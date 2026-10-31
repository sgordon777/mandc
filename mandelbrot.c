#include <math.h>
#include <omp.h>
#include <stddef.h>
#include <stdint.h>

#pragma omp declare target
static inline __attribute__((no_stack_protector)) uint16_t mandelbrot_escape_count(
    double real,
    double imaginary,
    int max_iterations
) {
    double z_real = 0.0;
    double z_imag = 0.0;
    int iteration = max_iterations;

    for (int step = 0; step < max_iterations; ++step) {
        const double next_real = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imaginary;
        z_real = next_real;

        if (z_real * z_real + z_imag * z_imag > 4.0) {
            iteration = step + 1;
            break;
        }
    }

    return (uint16_t)iteration;
}
#pragma omp end declare target

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
            const size_t pixel_index = (size_t)x * (size_t)height + (size_t)y;
            iterations[pixel_index] = mandelbrot_escape_count(real, imaginary, max_iterations);
        }
    }
}

__attribute__((no_stack_protector)) int mandelbrot_openmp_gpu_available(void) {
    int initial_device = 1;

#pragma omp target map(from: initial_device)
    {
        initial_device = omp_is_initial_device();
    }

    return !initial_device;
}

__attribute__((no_stack_protector)) void render_mandelbrot_openmp_gpu(
    int width,
    int height,
    double center_real,
    double center_imag,
    double span,
    int max_iterations,
    int thread_count,
    uint16_t *iterations
) {
    (void)thread_count;
    const double imaginary_span = span * (double)height / (double)width;
    const size_t pixel_count = (size_t)width * (size_t)height;

#pragma omp target teams distribute parallel for collapse(2) thread_limit(256) \
    map(to: width, height, center_real, center_imag, span, imaginary_span, max_iterations) \
    map(from: iterations[0:pixel_count])
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            const double real = center_real - span / 2.0
                + span * (double)x / (double)(width - 1 > 0 ? width - 1 : 1);
            const double imaginary = center_imag + imaginary_span / 2.0
                - imaginary_span * (double)y / (double)(height - 1 > 0 ? height - 1 : 1);
            const size_t pixel_index = (size_t)x * (size_t)height + (size_t)y;
            iterations[pixel_index] = mandelbrot_escape_count(real, imaginary, max_iterations);
        }
    }
}
