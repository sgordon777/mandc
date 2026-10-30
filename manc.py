"""Interactive Mandelbrot explorer rendered with Pygame and NumPy."""

from __future__ import annotations

import os
import sys
import time

if sys.platform.startswith("linux"):
    os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

import numpy as np
import pygame
import wgpu


WIDTH = 1024
HEIGHT = 1024
MAX_ITERATIONS = 256
THREADS = 4
START_CENTER = (np.float64(-0.5), np.float64(0.0))
START_SPAN = np.float64(3.0)
# [256,3] color RGB pallete, options: gray, hot, hsv, jet, rainbow, turbo, vir, pink, bone, fall, copper, summer, ocean, cool, cube, sin3
# sin3=np.array ( [  [(127.5 + 127.5 * np.sin(i*0.08)) for i in range(256)], [(127.5 + 127.5 * np.sin(i*0.08+2.1)) for i in range(256)], [(127.5 + 127.5 * np.sin(i*0.08+4.2)) for i in range(256)]] ).T
PALLETE_NAME = "hot.csv" 

# 0=NumPy float64, 1=C float64, 2=WebGPU float32
RENDERER = 2

#gray_map = np.array([  [i for i in range(256) ], [i for i in range(256)], [i for i in range(256)] ] ).T

def color_iteration_counts(iterations: np.ndarray, max_iterations: int, map: np.ndarray) -> np.ndarray:
    """Map exact escape counts to an RGB image using a palette."""
    palette_indices = np.clip((iterations.astype(np.uint32) - 1) * 255 // max_iterations, 0, 254)
    return map[palette_indices]

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

    center_real: np.float64 = START_CENTER[0]
    center_imag: np.float64 = START_CENTER[1]
    span: np.float64 = START_SPAN
    max_iterations = MAX_ITERATIONS
    dragging = False
    show_path = False
    path_mouse = (0, 0)
    dirty = True
    running = True

    while running:
        path_disp = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                dirty = True
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dragging = True
                drag_start = event.pos
                center_start = (center_real, center_imag)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                show_path = True
                path_mouse = event.pos
                dirty = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                show_path = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                width, height = screen.get_size()
                center_real = np.float64(center_start[0] - (event.pos[0] - drag_start[0]) * span / width)
                center_imag = np.float64(center_start[1] + (event.pos[1] - drag_start[1]) * span / width)
                dirty = True
            elif event.type == pygame.MOUSEMOTION and show_path:
                path_mouse = event.pos
                dirty = True
            elif event.type == pygame.MOUSEWHEEL:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                width, height = screen.get_size()
                old_span = span
                span *= np.float64(0.8) ** event.y
                anchor_real = np.float64(center_real + (mouse_x / width - 0.5) * old_span)
                anchor_imag = np.float64(center_imag + (0.5 - mouse_y / height) * old_span * height / width)
                center_real = np.float64(anchor_real - (mouse_x / width - 0.5) * span)
                center_imag = np.float64(anchor_imag - (0.5 - mouse_y / height) * span * height / width)
                dirty = True
            elif event.type == pygame.KEYDOWN:
                width, height = screen.get_size()
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    center_real, center_imag = START_CENTER
                    span = np.float64(START_SPAN)
                    max_iterations = MAX_ITERATIONS
                    dirty = True
                elif event.key == pygame.K_g and gpu_renderer is not None:
                    use_gpu = not use_gpu
                    dirty = True
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    max_iterations = min(max_iterations + 40, 1000)
                    dirty = True
                elif event.key == pygame.K_MINUS:
                    max_iterations = max(max_iterations - 40, 40)
                    dirty = True
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    direction = -1 if event.key == pygame.K_LEFT else 1
                    center_real = np.float64(center_real + direction * span * 0.1)
                    dirty = True
                elif event.key in (pygame.K_UP, pygame.K_DOWN):
                    direction = -1 if event.key == pygame.K_DOWN else 1
                    center_imag = np.float64(center_imag + direction * span * 0.1)
                    dirty = True
                elif event.key in (pygame.K_w, pygame.K_s):
                    direction = 1 if event.key == pygame.K_w else -1
                    center_imag = np.float64(center_imag + direction * span * 0.1 * height / width)
                    dirty = True

        if dirty and running:
            width, height = screen.get_size()
            render_start = time.perf_counter()

            if use_gpu and gpu_renderer is not None:
                pixels = gpu_renderer.render(width, height, center_real, center_imag, span, max_iterations)
                render_mode = "WebGPU f32"
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
            render_ms = (time.perf_counter() - render_start) * 1000.0
            smoothed_render_ms = render_time_alpha * render_ms + (1.0 - render_time_alpha) * smoothed_render_ms
            pygame.surfarray.blit_array(screen, pixels)
            if show_path:
                path_point = screen_to_complex(path_mouse[0], path_mouse[1], width, height, center_real, center_imag, span)
                path_points, finalz, ct = mandelbrot_path(path_point, width, height, center_real, center_imag, span, max_iterations, max_iterations)
                if len(path_points) > 1:
                    pygame.draw.lines(screen, (0, 0, 0), False, path_points, 1)
                    # Draw a dot at the final point
                    pygame.draw.circle(screen, (255, 0, 0), path_points[-1], 4)
                    pygame.draw.circle(screen, (0, 255, 0), path_points[0], 4)
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
                controls_text = "Wheel zoom | Drag pan | G GPU/CPU | +/- detail | R reset | Esc quit"
            controls = font.render(controls_text, True, (255, 255, 255))
            screen.blit(status, (8, 6))
            screen.blit(center_status, (8, 24))
            screen.blit(controls, (8, 42))
            if path_disp:
                screen.blit(path_stat, (8, 60))
            pygame.display.flip()
            dirty = False

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
