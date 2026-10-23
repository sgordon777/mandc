"""Interactive Mandelbrot explorer rendered with Pygame and NumPy."""

from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("WGPU_BACKEND_TYPE", "Vulkan")

import numpy as np
import pygame
import wgpu


WIDTH = 512
HEIGHT = 512
MAX_ITERATIONS = 256
START_CENTER = (np.float64(-0.5), np.float64(0.0))
START_SPAN = np.float64(3.0)

# 0=native python(double), 1=SSE(double), 2=webGPU (single)
RENDERER = 0

def render_mandelbrot(
    width: int,
    height: int,
    center_real: float,
    center_imag: float,
    span: float,
    max_iterations: int,
) -> np.ndarray:
    """Calculate an RGB Mandelbrot image with shape (width, height, 3)."""
    real = np.linspace(center_real - span / 2, center_real + span / 2, width, dtype=np.float64)
    imag_span = np.float64(span * height / width)
    imag = np.linspace(center_imag + imag_span / 2, center_imag - imag_span / 2, height, dtype=np.float64)
    coordinates = real[:, np.newaxis] + np.complex128(1j) * imag[np.newaxis, :]
    values = np.zeros(coordinates.shape, dtype=np.complex128)
    escaped = np.zeros(coordinates.shape, dtype=bool)
    iterations = np.zeros(coordinates.shape, dtype=np.uint16)

    for step in range(max_iterations):
        active = ~escaped
        values[active] = values[active] * values[active] + coordinates[active]
        escaped_now = active & (values.real * values.real + values.imag * values.imag > 4)
        iterations[escaped_now] = step + 1
        escaped |= escaped_now

    phase = iterations.astype(np.float64) * np.float64(0.08)
    image = np.empty((*coordinates.shape, 3), dtype=np.uint8)
    image[..., 0] = (127.5 + 127.5 * np.sin(phase)).astype(np.uint8)
    image[..., 1] = (127.5 + 127.5 * np.sin(phase + np.float64(2.1))).astype(np.uint8)
    image[..., 2] = (127.5 + 127.5 * np.sin(phase + np.float64(4.2))).astype(np.uint8)
    image[~escaped] = 0
    return image


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
    var escaped = false;
    var iteration = 0u;

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

    var color = 0u;
    if (escaped) {
        let phase = f32(iteration) * 0.08;
        let red = u32(127.5 + 127.5 * sin(phase));
        let green = u32(127.5 + 127.5 * sin(phase + 2.1));
        let blue = u32(127.5 + 127.5 * sin(phase + 4.2));
        color = red | (green << 8u) | (blue << 16u) | 0xff000000u;
    }
    pixels[id.y * params.width + id.x] = color;
}
"""

    def __init__(self) -> None:
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
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
    if RENDERER == 2:
        try:
            gpu_renderer = WebGPURenderer()
            use_gpu = True
            print(f"WebGPU device: {gpu_renderer.adapter.info['device']}")
        except Exception as error:
            gpu_renderer = None
            use_gpu = False
            print(f"WebGPU unavailable; using CPU float64 renderer: {error}")
    elif RENDERER == 1:
        gpu_renderer = None
        use_gpu = False
        print("Using CPU float32 renderer")
    else:
        gpu_renderer = None
        use_gpu = False
        print("Using CPU float64 renderer")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Mandelbrot Explorer")
    font = pygame.font.Font(None, 18)
    clock = pygame.time.Clock()

    center_real: np.float64 = START_CENTER[0]
    center_imag: np.float64 = START_CENTER[1]
    span: np.float64 = START_SPAN
    max_iterations = MAX_ITERATIONS
    dragging = False
    dirty = True
    running = True

    while running:
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
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                width, height = screen.get_size()
                center_real = np.float64(center_start[0] - (event.pos[0] - drag_start[0]) * span / width)
                center_imag = np.float64(center_start[1] + (event.pos[1] - drag_start[1]) * span / width)
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
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_UP):
                    max_iterations = min(max_iterations + 40, 1000)
                    dirty = True
                elif event.key in (pygame.K_MINUS, pygame.K_DOWN):
                    max_iterations = max(max_iterations - 40, 40)
                    dirty = True
                elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    direction = -1 if event.key == pygame.K_LEFT else 1
                    center_real = np.float64(center_real + direction * span * 0.1)
                    dirty = True
                elif event.key in (pygame.K_w, pygame.K_s):
                    direction = 1 if event.key == pygame.K_w else -1
                    center_imag = np.float64(center_imag + direction * span * 0.1 * height / width)
                    dirty = True

        if dirty and running:
            width, height = screen.get_size()
            if use_gpu and gpu_renderer is not None:
                pixels = gpu_renderer.render(width, height, center_real, center_imag, span, max_iterations)
                render_mode = "WebGPU f32"
            else:
                pixels = render_mandelbrot(width, height, center_real, center_imag, span, max_iterations)
                render_mode = "CPU f64"
            pygame.surfarray.blit_array(screen, pixels)
            status = font.render(
                f"{render_mode} | Zoom {START_SPAN / span:.1f}x | Iterations {max_iterations}",
                True,
                (255, 255, 255),
            )
            controls_text = "Wheel zoom | Drag pan | +/- detail | R reset | Esc quit"
            if gpu_renderer is not None:
                controls_text = "Wheel zoom | Drag pan | G GPU/CPU | +/- detail | R reset | Esc quit"
            controls = font.render(controls_text, True, (255, 255, 255))
            screen.blit(status, (8, 6))
            screen.blit(controls, (8, 24))
            pygame.display.flip()
            dirty = False

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
