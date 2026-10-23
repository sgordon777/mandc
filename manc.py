"""Interactive Mandelbrot explorer rendered with Pygame and NumPy."""

from __future__ import annotations

import numpy as np
import pygame


WIDTH = 256
HEIGHT = 256
MAX_ITERATIONS = 120
START_CENTER = (-0.5, 0.0)
START_SPAN = 3


def render_mandelbrot(
	width: int,
	height: int,
	center_real: float,
	center_imag: float,
	span: float,
	max_iterations: int,
) -> np.ndarray:
	"""Calculate an RGB Mandelbrot image with shape (width, height, 3)."""
	real = np.linspace(center_real - span / 2, center_real + span / 2, width)
	imag_span = span * height / width
	imag = np.linspace(center_imag + imag_span / 2, center_imag - imag_span / 2, height)
	coordinates = real[:, np.newaxis] + 1j * imag[np.newaxis, :]
	values = np.zeros(coordinates.shape, dtype=np.complex128)
	escaped = np.zeros(coordinates.shape, dtype=bool)
	iterations = np.zeros(coordinates.shape, dtype=np.uint16)

	for step in range(max_iterations):
		active = ~escaped
		values[active] = values[active] * values[active] + coordinates[active]
		escaped_now = active & (values.real * values.real + values.imag * values.imag > 4)
		iterations[escaped_now] = step + 1
		escaped |= escaped_now

	phase = iterations.astype(np.float32) * 0.08
	image = np.empty((*coordinates.shape, 3), dtype=np.uint8)
	image[..., 0] = (127.5 + 127.5 * np.sin(phase)).astype(np.uint8)
	image[..., 1] = (127.5 + 127.5 * np.sin(phase + 2.1)).astype(np.uint8)
	image[..., 2] = (127.5 + 127.5 * np.sin(phase + 4.2)).astype(np.uint8)
	image[~escaped] = 0
	return image


def main() -> None:
	pygame.init()
	screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
	pygame.display.set_caption("Mandelbrot Explorer")
	font = pygame.font.Font(None, 22)
	clock = pygame.time.Clock()

	center_real, center_imag = START_CENTER
	span = START_SPAN
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
				center_real = center_start[0] - (event.pos[0] - drag_start[0]) * span / width
				center_imag = center_start[1] + (event.pos[1] - drag_start[1]) * span / width
				dirty = True
			elif event.type == pygame.MOUSEWHEEL:
				mouse_x, mouse_y = pygame.mouse.get_pos()
				width, height = screen.get_size()
				old_span = span
				span *= 0.8 ** event.y
				anchor_real = center_real + (mouse_x / width - 0.5) * old_span
				anchor_imag = center_imag + (0.5 - mouse_y / height) * old_span * height / width
				center_real = anchor_real - (mouse_x / width - 0.5) * span
				center_imag = anchor_imag - (0.5 - mouse_y / height) * span * height / width
				dirty = True
			elif event.type == pygame.KEYDOWN:
				width, height = screen.get_size()
				if event.key == pygame.K_ESCAPE:
					running = False
				elif event.key == pygame.K_r:
					center_real, center_imag = START_CENTER
					span = START_SPAN
					max_iterations = MAX_ITERATIONS
					dirty = True
				elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_UP):
					max_iterations = min(max_iterations + 40, 1000)
					dirty = True
				elif event.key in (pygame.K_MINUS, pygame.K_DOWN):
					max_iterations = max(max_iterations - 40, 40)
					dirty = True
				elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
					direction = -1 if event.key == pygame.K_LEFT else 1
					center_real += direction * span * 0.1
					dirty = True
				elif event.key in (pygame.K_w, pygame.K_s):
					direction = 1 if event.key == pygame.K_w else -1
					center_imag += direction * span * 0.1 * height / width
					dirty = True

		if dirty and running:
			width, height = screen.get_size()
			pixels = render_mandelbrot(width, height, center_real, center_imag, span, max_iterations)
			pygame.surfarray.blit_array(screen, pixels)
			status = font.render(
				f"Zoom {START_SPAN / span:.1f}x  Iterations {max_iterations}  "
				"Wheel: zoom | Drag: pan | Arrows/W/S: move | +/-: detail | R: reset | Esc: quit",
				True,
				(255, 255, 255),
			)
			screen.blit(status, (10, 10))
			pygame.display.flip()
			dirty = False

		clock.tick(60)

	pygame.quit()


if __name__ == "__main__":
	main()
