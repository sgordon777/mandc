#include <math.h>
#include <stddef.h>
#include <stdint.h>

void render_mandelbrot_c(
    int width,
    int height,
    double center_real,
    double center_imag,
    double span,
    int max_iterations,
    int thread_count,
    uint16_t *iterations
) {
    const double imaginary_span = span * (double)height / (double)width;

#pragma omp parallel for schedule(dynamic, 1) num_threads(thread_count)
    for (int x = 0; x < width; ++x) {
        const double real = center_real - span / 2.0
            + span * (double)x / (double)(width - 1 > 0 ? width - 1 : 1);

        for (int y = 0; y < height; ++y) {
            const double imaginary = center_imag + imaginary_span / 2.0
                - imaginary_span * (double)y / (double)(height - 1 > 0 ? height - 1 : 1);
            double z_real = 0.0;
            double z_imag = 0.0;
            int iteration = max_iterations;

            int step;
            for (step = 0; step < max_iterations; ++step) {
                const double next_real = z_real * z_real - z_imag * z_imag + real;
                z_imag = 2.0 * z_real * z_imag + imaginary;
                z_real = next_real;

                if (z_real * z_real + z_imag * z_imag > 4.0) {
                    iteration = step + 1;
                    break;
                }
            }

            const size_t pixel_index = (size_t)x * (size_t)height + (size_t)y;
            iterations[pixel_index] = (uint16_t)iteration;
        }
    }
}
