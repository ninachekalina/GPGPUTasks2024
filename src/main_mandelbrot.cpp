#include <cmath>
#include <gpu.hpp>  
#include <image.hpp>// Для сохранения PNG
#include <iostream>
#include <stdexcept>
#include <timer.hpp>// Таймер для замеров
#include <vector>

void executeMandelbrotGPU(unsigned int width, unsigned int height, float centerX, float centerY, float sizeX,
                          float sizeY, unsigned int iterationsLimit) {
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    ocl::Kernel kernel(mandelbrot_kernel, mandelbrot_kernel_length, "mandelbrot");
    bool printCompilationLog = false;
    kernel.compile(printCompilationLog);

    gpu::gpu_mem_32f gpuResults;
    gpuResults.resizeN(width * height);

    timer performanceTimer;
    const int benchmarkRuns = 5;
    for (int run = 0; run < benchmarkRuns; ++run) {
        kernel.exec(gpu::WorkSize(16, 16, width, height), gpuResults, width, height, centerX - sizeX / 2.0f,
                    centerY - sizeY / 2.0f, sizeX, sizeY, iterationsLimit, 1// 1 означает включение сглаживания
        );
        gpuResults.readN(gpu_results.ptr(), width * height);
        performanceTimer.nextLap();
    }

    size_t flopsPerIteration = 10;
    size_t totalFlops = width * height * iterationsLimit * flopsPerIteration;
    size_t gflops = 1000 * 1000 * 1000;
    std::cout << "Среднее время выполнения (GPU): " << performanceTimer.lapAvg() << " сек\n";
    std::cout << "Производительность (GPU): " << totalFlops / gflops / performanceTimer.lapAvg() << " GFLOPS\n";

    double realIterationsRatio = 0.0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            realIterationsRatio += gpu_results.ptr()[y * width + x];
        }
    }
    std::cout << "Доля реальных итераций: " << 100.0 * realIterationsRatio / (width * height) << "%\n";

    renderToColor(gpu_results.ptr(), image.ptr(), width, height);
    image.savePNG("mandelbrot_gpu_modified.png");

    double averageError = 0.0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            averageError += std::abs(gpu_results.ptr()[y * width + x] - cpu_results.ptr()[y * width + x]);
        }
    }
    averageError /= (width * height);
    std::cout << "Средняя разница между CPU и GPU результатами: " << 100.0 * averageError << "%\n";

    if (errorAvg > 0.03) {
        throw std::runtime_error("Too high difference between CPU and GPU results!");
    }
}