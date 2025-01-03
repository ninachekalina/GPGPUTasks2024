#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define ELEMENTS_PER_THREAD 32
#define GROUP_SIZE 64

// Ядро для суммы с использованием атомарных операций
__kernel void sum_with_atomic(__global const unsigned int* arr,
                               __global unsigned int* sum,
                               unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    // Если индекс глобального потока больше, чем количество элементов
    if (gid >= n)
        return;

    atomic_add(sum, arr[gid]);
}

// Ядро для суммы с использованием цикла
__kernel void sum_with_loop(__global const unsigned int* arr,
                             __global unsigned int* sum,
                             unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    unsigned int result = 0;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        unsigned int idx = gid * ELEMENTS_PER_THREAD + i;
        if (idx < n) {
            result += arr[idx];
        }
    }

    atomic_add(sum, result);
}

// Ядро с улучшенной локальной памятью и коалесценцией
__kernel void sum_with_local_memory_coalesced(__global const unsigned int* arr,
                                              __global unsigned int* sum,
                                              unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int result = 0;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        unsigned int idx = wid * grs * ELEMENTS_PER_THREAD + i * grs + lid;
        if (idx < n) {
            result += arr[idx];
        }
    }

    atomic_add(sum, result);
}

// Ядро с использованием локальной памяти для суммирования
__kernel void sum_with_local_memory(__global const unsigned int* arr,
                                     __global unsigned int* sum,
                                     unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[GROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_result = 0;
        for (unsigned int i = 0; i < GROUP_SIZE; i++) {
            group_result += buf[i];
        }
        atomic_add(sum, group_result);
    }
}

// Ядро с деревом редукции для эффективного сложения
__kernel void sum_with_tree_reduction(__global const unsigned int* arr,
                                       __global unsigned int* sum,
                                       unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buf[GROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nValues = GROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}
