#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

template<typename T>
void handleError(const T &expected, const T &actual, std::string message, std::string filename, int line)
{
    std::cerr << "Error in " << filename << " at line " << line << ": " << message << std::endl;
    std::cerr << "Expected: " << expected << ", Actual: " << actual << std::endl;
    exit(1);
}

#define CHECK_EQUAL(a, b, message) handleError(a, b, message, __FILE__, __LINE__)

void executeSum(const std::vector<unsigned int>& data, unsigned int expectedSum, int iterations, gpu::Device device, ocl::Kernel kernel, std::string kernelName) {
    unsigned int n = data.size();
    unsigned int workGroupSize = 64;
    unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    gpu::gpu_mem_32u data_gpu;
    data_gpu.resizeN(n);
    data_gpu.writeN(data.data(), n);

    kernel.compile(false);
    {
        timer t;
        for (int iter = 0; iter < iterations; ++iter) {
            unsigned int sum = 0;

            gpu::gpu_mem_32u sum_gpu;
            sum_gpu.resizeN(1);
            sum_gpu.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), data_gpu, sum_gpu, n);
            sum_gpu.readN(&sum, 1);

            CHECK_EQUAL(expectedSum, sum, "GPU " + kernelName + " result should be consistent!");

            t.nextLap();
        }
        std::cout << "GPU " + kernelName + ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " + kernelName + ":     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }
}

int main(int argc, char **argv)
{
    std::vector<unsigned int> as = generateRandomData(); // Рандомные данные для примера
    unsigned int referenceSum = calculateSum(as); // Эталонная сумма

    int benchmarkingIters = 10;

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    // Ядра для тестирования
    ocl::Kernel atomicKernel(sum_kernel, sum_kernel_length, "sum_with_atomic");
    executeSum(as, referenceSum, benchmarkingIters, device, atomicKernel, "atomicKernel");

    ocl::Kernel loopKernel(sum_kernel, sum_kernel_length, "sum_with_loop");
    executeSum(as, referenceSum, benchmarkingIters, device, loopKernel, "loopKernel");

    ocl::Kernel coalescedKernel(sum_kernel, sum_kernel_length, "sum_with_local_memory_coalesced");
    executeSum(as, referenceSum, benchmarkingIters, device, coalescedKernel, "coalescedKernel");

    ocl::Kernel localMemoryKernel(sum_kernel, sum_kernel_length, "sum_with_local_memory");
    executeSum(as, referenceSum, benchmarkingIters, device, localMemoryKernel, "localMemoryKernel");

    ocl::Kernel treeReductionKernel(sum_kernel, sum_kernel_length, "sum_with_tree_reduction");
    executeSum(as, referenceSum, benchmarkingIters, device, treeReductionKernel, "treeReductionKernel");

    return 0;
}
