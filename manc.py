"""Interactive Mandelbrot explorer rendered with Pygame and NumPy."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

import numpy as np
import pygame
import wgpu


WIDTH = 1024
HEIGHT = 1024
MAX_ITERATIONS = 256
THREADS = 16
START_CENTER = (np.float64(-0.5), np.float64(0.0))
START_SPAN = np.float64(3.0)
# [256,3] color RGB pallete, options: gray, hot, hsv, jet, rainbow, turbo, vir, pink, bone, fall, copper, summer, ocean, cool, cube, sin3
# sin3=np.array ( [  [(127.5 + 127.5 * np.sin(i*0.08)) for i in range(256)], [(127.5 + 127.5 * np.sin(i*0.08+2.1)) for i in range(256)], [(127.5 + 127.5 * np.sin(i*0.08+4.2)) for i in range(256)]] ).T
PALLETE_NAME = "sin3.csv" 

# 0=NumPy float64, 1=C float64, 2=WebGPU float32
RENDERER = 1

#gray_map = np.array([  [i for i in range(256) ], [i for i in range(256)], [i for i in range(256)] ] ).T

def color_iteration_counts(iterations: np.ndarray, max_iterations: int, map: np.ndarray) -> np.ndarray:
    """Map exact escape counts to an RGB image using a palette."""
    palette_indices = np.clip((iterations.astype(np.uint32) - 1) * 255 // max_iterations, 0, 254)
    return map[palette_indices]


def dump_iterations_to_file(iterations: np.ndarray, output_dir: str | os.PathLike[str] = ".") -> Path:
    """Write the exact Mandelbrot iteration matrix to a CSV file and return its path."""
    matrix = np.asarray(iterations, dtype=np.uint16)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = destination / f"mandelbrot_iterations_{timestamp}.csv"
    np.savetxt(filename, matrix, delimiter=",", fmt="%d")
    return filename

def render_mandelbrot(
    width: int,
    height: int,
    center_real: float,
    center_imag: float,
    span: float,
    max_iterations: int,
    iteration_buffer: np.ndarray,
) -> np.ndarray:
    """Calculate exact Mandelbrot escape counts with shape (width, height)."""
    real = np.linspace(center_real - span / 2, center_real + span / 2, width, dtype=np.float64)
    imag_span = np.float64(span * height / width)
    imag = np.linspace(center_imag + imag_span / 2, center_imag - imag_span / 2, height, dtype=np.float64)
    coordinates = real[:, np.newaxis] + np.complex128(1j) * imag[np.newaxis, :]
    values = np.zeros(coordinates.shape, dtype=np.complex128)
    escaped = np.zeros(coordinates.shape, dtype=bool)
    if iteration_buffer.shape != coordinates.shape or iteration_buffer.dtype != np.uint16:
        raise ValueError("iteration_buffer must be uint16 with shape (width, height)")
    iterations = iteration_buffer
    iterations.fill(max_iterations)

    for step in range(max_iterations):
        active = ~escaped
        values[active] = values[active] * values[active] + coordinates[active]
        escaped_now = active & (values.real * values.real + values.imag * values.imag > 4)
        iterations[escaped_now] = step + 1
        escaped |= escaped_now

    return iterations


def screen_to_complex(
    screen_x: int,
    screen_y: int,
    width: int,
    height: int,
    center_real: float,
    center_imag: float,
    span: float,
) -> complex:
    """Convert a screen coordinate to a complex plane coordinate."""
    imag_span = span * height / width
    real = center_real - span / 2.0 + span * screen_x / width
    imag = center_imag + imag_span / 2.0 - imag_span * screen_y / height
    return complex(real, imag)


def mandelbrot_path(
    complex_point: complex,
    width: int,
    height: int,
    center_real: float,
    center_imag: float,
    span: float,
    max_iterations: int,
    max_points: int = 128,
) -> list[tuple[int, int]]:
    """Return the orbit of a point in screen-space as a list of pixel coordinates."""
    imag_span = span * height / width
    z = 0j
    points: list[tuple[int, int]] = []

    for n in range(max_points):
        z = z * z + complex_point
        x = int(round((z.real - (center_real - span / 2.0)) / span * width))
        y = int(round(height - ((z.imag - (center_imag - imag_span / 2.0)) / imag_span * height)))
        #if 0 <= x < width and 0 <= y < height:
        points.append((x, y))
        if abs(z) > 2.0:
            break

    return points, z, n


class WebGPURenderer:
    """Compute and color the Mandelbrot set on a WebGPU device."""

    _SHADER = """
struct Params {
    width: u32,
    height: u32,
    max_iterations: u32,
    _padding: u32,
    center_real: f32,
    center_imag: f32,
    span: f32,
    _padding2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
@group(0) @binding(2) var<storage, read> palette: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }

    let x_fraction = f32(id.x) / f32(max(params.width - 1u, 1u));
    let y_fraction = f32(id.y) / f32(max(params.height - 1u, 1u));
    let real = params.center_real + (x_fraction - 0.5) * params.span;
    let imag = params.center_imag + (0.5 - y_fraction) * params.span * f32(params.height) / f32(params.width);
    var z_real = 0.0;
    var z_imag = 0.0;
    var iteration = params.max_iterations;
    var escaped = false;

    for (var step = 0u; step < params.max_iterations; step += 1u) {
        let next_real = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = next_real;
        if (z_real * z_real + z_imag * z_imag > 4.0) {
            iteration = step + 1u;
            escaped = true;
            break;
        }
    }

    var palette_index = 255u;
    if (escaped) {
        palette_index = min((iteration - 1u) * 255u / params.max_iterations, 254u);
    }
    pixels[id.y * params.width + id.x] = palette[palette_index];

}

"""

    def __init__(self, rgb_palette: np.ndarray) -> None:
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        palette_words = (
            rgb_palette[:, 0].astype(np.uint32)
            | (rgb_palette[:, 1].astype(np.uint32) << 8)
            | (rgb_palette[:, 2].astype(np.uint32) << 16)
            | np.uint32(0xFF000000)
        )
        self.palette_buffer = self.device.create_buffer(
            size=palette_words.nbytes,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(self.palette_buffer, 0, palette_words)
        shader = self.device.create_shader_module(code=self._SHADER)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": shader, "entry_point": "main"},
        )

    def render(
        self,
        width: int,
        height: int,
        center_real: float,
        center_imag: float,
        span: float,
        max_iterations: int,
    ) -> np.ndarray:
        pixel_count = width * height
        params = np.array(
            [
                width,
                height,
                max_iterations,
                0,
            ],
            dtype=np.uint32,
        )
        float_params = np.array([center_real, center_imag, span, 0.0], dtype=np.float32)
        uniform_data = params.tobytes() + float_params.tobytes()
        uniform_buffer = self.device.create_buffer(
            size=len(uniform_data),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )
        self.device.queue.write_buffer(uniform_buffer, 0, uniform_data)

        pixel_buffer = self.device.create_buffer(
            size=pixel_count * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
        )
        bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": uniform_buffer}},
                {"binding": 1, "resource": {"buffer": pixel_buffer}},
                {"binding": 2, "resource": {"buffer": self.palette_buffer}},
            ],
        )
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, bind_group)
        compute_pass.dispatch_workgroups((width + 7) // 8, (height + 7) // 8)
        compute_pass.end()
        self.device.queue.submit([encoder.finish()])

        raw_pixels = self.device.queue.read_buffer(pixel_buffer)
        rgba = np.frombuffer(raw_pixels, dtype=np.uint8).reshape(height, width, 4)
        return rgba[..., :3].transpose(1, 0, 2).copy()


def main() -> None:
    palette = np.rint(np.clip(np.loadtxt(PALLETE_NAME, delimiter=","), 0, 255)).astype(np.uint8)
    if palette.shape != (256, 3):
        raise ValueError(f"Palette must have shape (256, 3), got {palette.shape}")
    c_renderer = None
    if RENDERER == 2:
        try:
            gpu_renderer = WebGPURenderer(palette)
            use_gpu = True
            print(f"WebGPU device: {gpu_renderer.adapter.info['device']}")
        except Exception as error:
            gpu_renderer = None
            use_gpu = False
            print(f"WebGPU unavailable; using CPU float64 renderer: {error}")
    elif RENDERER == 1:
        gpu_renderer = None
        use_gpu = False
        from c_renderer import CRenderer

        c_renderer = CRenderer()
        print("Using native C float64 renderer")
    else:
        gpu_renderer = None
        use_gpu = False
        print("Using CPU float64 renderer")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Mandelbrot Explorer")
    #font = pygame.font.Font("monospace", 18)
    font = pygame.font.SysFont("monospace", 14)
    clock = pygame.time.Clock()
    render_ms = 0.0
    smoothed_render_ms = 0.0
    render_time_alpha = 0.2
    numpy_iteration_buffer = np.empty((WIDTH, HEIGHT), dtype=np.uint16)
    right_iteration_buffer = np.empty((WIDTH, HEIGHT), dtype=np.uint16)
    right_pixels = np.empty((WIDTH, HEIGHT, 3), dtype=np.uint8)
    last_iterations: np.ndarray | None = None

    center_real: np.float64 = START_CENTER[0]
    center_imag: np.float64 = START_CENTER[1]
    span: np.float64 = START_SPAN
    max_iterations = MAX_ITERATIONS
    dragging = False
    show_path = False
    path_dragging = False
    path_point: complex | None = None
    right_mouse_complex: complex | None = None
    dual_viewport = False
    left_dirty = True
    right_dirty = True
    right_base_dirty = True
    running = True

    while running:
        delta_time = min(clock.tick(60) / 1000.0, 0.1)
        path_disp = 0
        left_width = screen.get_width() // 2 if dual_viewport else screen.get_width()
        right_width = screen.get_width() - left_width if dual_viewport else 0
        height = screen.get_height()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                left_width = screen.get_width() // 2 if dual_viewport else screen.get_width()
                right_width = screen.get_width() - left_width if dual_viewport else 0
                height = screen.get_height()
                left_dirty = True
                right_dirty = True
                right_base_dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and event.pos[0] < left_width:
                dragging = True
                drag_start = event.pos
                center_start = (center_real, center_imag)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                if not dual_viewport or event.pos[0] < left_width:
                    path_point = screen_to_complex(
                        event.pos[0], event.pos[1], left_width, height, center_real, center_imag, span
                    )
                else:
                    path_point = screen_to_complex(
                        event.pos[0] - left_width,
                        event.pos[1],
                        right_width,
                        height,
                        START_CENTER[0],
                        START_CENTER[1],
                        START_SPAN,
                    )
                show_path = True
                path_dragging = True
                left_dirty = True
                right_dirty = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                path_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if not dual_viewport or event.pos[0] < left_width:
                    if right_mouse_complex is not None:
                        right_mouse_complex = None
                        right_dirty = True
                    if dragging:
                        center_real = np.float64(center_start[0] - (event.pos[0] - drag_start[0]) * span / left_width)
                        center_imag = np.float64(center_start[1] + (event.pos[1] - drag_start[1]) * span / left_width)
                        left_dirty = True
                    if path_dragging:
                        path_point = screen_to_complex(
                            event.pos[0], event.pos[1], left_width, height, center_real, center_imag, span
                        )
                        left_dirty = True
                        right_dirty = True
                elif dual_viewport:
                    right_mouse_complex = screen_to_complex(
                        event.pos[0] - left_width,
                        event.pos[1],
                        right_width,
                        height,
                        START_CENTER[0],
                        START_CENTER[1],
                        START_SPAN,
                    )
                    right_dirty = True
                    if path_dragging:
                        path_point = right_mouse_complex
                        left_dirty = True
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if mouse_x >= left_width:
                    continue
                old_span = span
                span *= np.float64(0.8) ** event.y
                anchor_real = np.float64(center_real + (mouse_x / left_width - 0.5) * old_span)
                anchor_imag = np.float64(center_imag + (0.5 - mouse_y / height) * old_span * height / left_width)
                center_real = np.float64(anchor_real - (mouse_x / left_width - 0.5) * span)
                center_imag = np.float64(anchor_imag - (0.5 - mouse_y / height) * span * height / left_width)
                left_dirty = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F2:
                    window_height = screen.get_height()
                    if dual_viewport:
                        window_width = max(1, screen.get_width() // 2)
                        dual_viewport = False
                    else:
                        window_width = screen.get_width() * 2
                        dual_viewport = True
                    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
                    left_width = screen.get_width() // 2 if dual_viewport else screen.get_width()
                    right_width = screen.get_width() - left_width if dual_viewport else 0
                    right_mouse_complex = None
                    left_dirty = True
                    right_dirty = True
                    right_base_dirty = True
                elif event.key == pygame.K_c and show_path:
                    show_path = False
                    path_dragging = False
                    path_point = None
                    left_dirty = True
                    right_dirty = True
                elif event.key == pygame.K_r:
                    center_real, center_imag = START_CENTER
                    span = np.float64(START_SPAN)
                    max_iterations = MAX_ITERATIONS
                    left_dirty = True
                    right_dirty = True
                    right_base_dirty = True
                elif event.key == pygame.K_g and gpu_renderer is not None:
                    use_gpu = not use_gpu
                    left_dirty = True
                    right_dirty = True
                    right_base_dirty = True
                elif event.key == pygame.K_d:
                    if last_iterations is not None:
                        dump_path = dump_iterations_to_file(last_iterations)
                    else:
                        width = left_width
                        temp_iterations = np.empty((width, height), dtype=np.uint16)
                        dump_matrix = render_mandelbrot(
                            width,
                            height,
                            center_real,
                            center_imag,
                            span,
                            max_iterations,
                            temp_iterations,
                        )
                        dump_path = dump_iterations_to_file(dump_matrix)
                    print(f"Dumped iteration matrix to {dump_path}")
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    max_iterations = min(max_iterations + 40, 1000)
                    left_dirty = True
                    right_dirty = True
                    right_base_dirty = True
                elif event.key == pygame.K_MINUS:
                    max_iterations = max(max_iterations - 40, 40)
                    left_dirty = True
                    right_dirty = True
                    right_base_dirty = True
                elif event.key in (pygame.K_w, pygame.K_s):
                    direction = 1 if event.key == pygame.K_w else -1
                    center_imag = np.float64(center_imag + direction * span * 0.1 * height / left_width)
                    left_dirty = True

        keys = pygame.key.get_pressed()
        horizontal_pan = int(keys[pygame.K_RIGHT]) - int(keys[pygame.K_LEFT])
        vertical_pan = int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])
        if horizontal_pan or vertical_pan:
            pan_scale = 0.7071067811865476 if horizontal_pan and vertical_pan else 1.0
            pan_step = np.float64(span * 0.6 * delta_time * pan_scale)
            center_real = np.float64(center_real + horizontal_pan * pan_step)
            center_imag = np.float64(center_imag + vertical_pan * pan_step * height / left_width)
            left_dirty = True

        left_width = screen.get_width() // 2 if dual_viewport else screen.get_width()
        right_width = screen.get_width() - left_width if dual_viewport else 0
        height = screen.get_height()
        frame_dirty = left_dirty or (dual_viewport and right_dirty)
        left_canvas = screen.subsurface(pygame.Rect(0, 0, left_width, height))
        right_canvas = screen.subsurface(pygame.Rect(left_width, 0, right_width, height)) if dual_viewport else None
        if left_dirty and running:
            width = left_width
            render_start = time.perf_counter()

            if use_gpu and gpu_renderer is not None:
                pixels = gpu_renderer.render(width, height, center_real, center_imag, span, max_iterations)
                render_mode = "WebGPU f32"
                last_iterations = None
            elif c_renderer is not None:
                if numpy_iteration_buffer.shape != (width, height):
                    numpy_iteration_buffer = np.empty((width, height), dtype=np.uint16)

                iterations = c_renderer.render(
                    width,
                    height,
                    center_real,
                    center_imag,
                    span,
                    max_iterations,
                    numpy_iteration_buffer,
                    THREADS,
                )
                pixels = color_iteration_counts(iterations, max_iterations, palette)
                render_mode = "C f64"
                last_iterations = iterations
            else:
                if numpy_iteration_buffer.shape != (width, height):
                    numpy_iteration_buffer = np.empty((width, height), dtype=np.uint16)
                iterations = render_mandelbrot(
                    width,
                    height,
                    center_real,
                    center_imag,
                    span,
                    max_iterations,
                    numpy_iteration_buffer,
                )
                pixels = color_iteration_counts(iterations, max_iterations, palette)
                render_mode = "CPU f64"
                last_iterations = iterations
            render_ms = (time.perf_counter() - render_start) * 1000.0
            smoothed_render_ms = render_time_alpha * render_ms + (1.0 - render_time_alpha) * smoothed_render_ms
            pygame.surfarray.blit_array(left_canvas, pixels)
            if show_path and path_point is not None:
                path_points, finalz, ct = mandelbrot_path(path_point, width, height, center_real, center_imag, span, max_iterations, max_iterations)
                if len(path_points) > 1:
                    pygame.draw.lines(left_canvas, (0, 0, 0), False, path_points, 1)
                    # Draw a dot at the final point
                    pygame.draw.circle(left_canvas, (255, 0, 0), path_points[-1], 4)
                    pygame.draw.circle(left_canvas, (0, 255, 0), path_points[0], 4)
                    #print(finalz, ct)
                    path_stat = font.render(
                        f"st:{path_point.real: 12.11f},{path_point.imag: 12.11f} end:{finalz.real: 12.11f},{finalz.imag: 12.11f}, ct={ct}",
                        True,
                        (255, 255, 255),
                    )                          
                    path_disp = 1;
            status = font.render(
                f"{render_mode} | Render {smoothed_render_ms:.1f} ms | Iterations {max_iterations}",
                True,
                (255, 255, 255),
            )
            center_status = font.render(
                f"C:{center_real: 15.14f},{center_imag: 15.14f}, SP:{span: 15.14g}",
                True,
                (255, 255, 255),
            )
            controls_text = "Wheel zoom | Drag pan | +/- detail | R reset | Esc quit"
            if gpu_renderer is not None:
                controls_text = "Arrows: pan while held | Wheel zoom/drag pan | Right-click: path | C clear | F2 single/dual | G GPU/CPU | D dump | +/- detail | R reset | Esc quit"
            else:
                controls_text = "Arrows: pan while held | Wheel zoom/drag pan | Right-click: path | C clear | F2 single/dual | D dump | +/- detail | R reset | Esc quit"
            controls = font.render(controls_text, True, (255, 255, 255))
            left_canvas.blit(status, (8, 6))
            left_canvas.blit(center_status, (8, 24))
            left_canvas.blit(controls, (8, 42))
            if path_disp:
                left_canvas.blit(path_stat, (8, 60))
            left_dirty = False

        if dual_viewport and right_dirty and running:
            width = right_width
            assert right_canvas is not None
            if right_base_dirty:
                if use_gpu and gpu_renderer is not None:
                    right_pixels = gpu_renderer.render(
                        width,
                        height,
                        START_CENTER[0],
                        START_CENTER[1],
                        START_SPAN,
                        max_iterations,
                    )
                else:
                    if right_iteration_buffer.shape != (width, height):
                        right_iteration_buffer = np.empty((width, height), dtype=np.uint16)
                    if c_renderer is not None:
                        right_iterations = c_renderer.render(
                            width,
                            height,
                            START_CENTER[0],
                            START_CENTER[1],
                            START_SPAN,
                            max_iterations,
                            right_iteration_buffer,
                            THREADS,
                        )
                    else:
                        right_iterations = render_mandelbrot(
                            width,
                            height,
                            START_CENTER[0],
                            START_CENTER[1],
                            START_SPAN,
                            max_iterations,
                            right_iteration_buffer,
                        )
                    right_pixels = color_iteration_counts(right_iterations, max_iterations, palette)
                right_base_dirty = False

            pygame.surfarray.blit_array(right_canvas, right_pixels)
            if show_path and path_point is not None:
                right_path_points, _, _ = mandelbrot_path(
                    path_point,
                    width,
                    height,
                    START_CENTER[0],
                    START_CENTER[1],
                    START_SPAN,
                    max_iterations,
                    max_iterations,
                )
                if len(right_path_points) > 1:
                    pygame.draw.lines(right_canvas, (0, 0, 0), False, right_path_points, 1)
                    pygame.draw.circle(right_canvas, (255, 0, 0), right_path_points[-1], 4)
                    pygame.draw.circle(right_canvas, (0, 255, 0), right_path_points[0], 4)
            if right_mouse_complex is not None:
                coordinate_text = font.render(
                    f"C: {right_mouse_complex.real: .12f}, {right_mouse_complex.imag: .12f}",
                    True,
                    (255, 255, 255),
                )
                background = pygame.Surface(
                    (coordinate_text.get_width() + 12, coordinate_text.get_height() + 8),
                    pygame.SRCALPHA,
                )
                background.fill((0, 0, 0, 180))
                right_canvas.blit(background, (4, 4))
                right_canvas.blit(coordinate_text, (10, 8))
            pygame.draw.line(right_canvas, (40, 40, 40), (0, 0), (0, height - 1), 2)
            right_dirty = False

        if frame_dirty:
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mandelbrot explorer")
    parser.add_argument("--render-type", type=int, default=RENDERER, choices=(0, 1, 2), help="0=NumPy, 1=C, 2=WebGPU")
    parser.add_argument("--palette", type=str, default=PALLETE_NAME, help="Palette CSV filename, e.g. hot.csv or sin3.csv")
    parser.add_argument("--width", type=int, default=WIDTH, help="Width of each viewport in pixels")
    parser.add_argument("--height", type=int, default=HEIGHT, help="Height of the window in pixels")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help="Maximum Mandelbrot iterations")
    parser.add_argument("--threads", type=int, default=THREADS, help="Native C renderer thread count")
    args = parser.parse_args()

    RENDERER = args.render_type
    PALLETE_NAME = args.palette
    WIDTH = args.width
    HEIGHT = args.height
    MAX_ITERATIONS = args.max_iterations
    THREADS = args.threads

    main()
