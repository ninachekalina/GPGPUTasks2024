#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(
    __global float *matrixA,
    __global float *matrixB,
    __global float *matrixC,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);
    
    if (global_row >= N || global_col >= M) {
        return;
    }
    
    float total = 0.0f;
    for (int k = 0; k < K; ++k) {
        total += matrixA[global_col * K + k] * matrixB[k * N + global_row];
    }
    matrixC[global_col * N + global_row] = total;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(
    __global float *matrixA,
    __global float *matrixB,
    __global float *matrixC,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);
    
    if (global_row >= N || global_col >= M) {
        return;
    }
    
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);
    
    __local float localTileA[TILE_SIZE][TILE_SIZE];
    __local float localTileB[TILE_SIZE][TILE_SIZE];
    
    float total = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        localTileA[local_col][local_row] = matrixA[global_col * K + (tileK * TILE_SIZE + local_row)];
        localTileB[local_col][local_row] = matrixB[(tileK * TILE_SIZE + local_col) * K + global_row];
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int k = 0; k < TILE_SIZE; ++k) {
            total += localTileA[local_col][k] * localTileB[k][local_row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    matrixC[global_col * N + global_row] = total;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void local_matrix_multiplication_local_wpt(
    __global float *matrixA,
    __global float *matrixB,
    __global float *matrixC,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    int global_row = get_global_id(0);
    int global_col = get_global_id(1) * WORK_PER_THREAD;

    if (global_row >= N || global_col >= M) {
        return;
    }

    int local_row = get_local_id(0);
    int local_col = get_local_id(1) * WORK_PER_THREAD;

    __local float localTileA[TILE_SIZE][TILE_SIZE];
    __local float localTileB[TILE_SIZE][TILE_SIZE];

    float total[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; ++w) {
        total[w] = 0.0f;
    }

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        for (int w = 0; w < WORK_PER_THREAD; ++w) {
            localTileA[local_col + w][local_row] = matrixA[(global_col + w) * K + (tileK * TILE_SIZE + local_row)];
            localTileB[local_col + w][local_row] = matrixB[(tileK * TILE_SIZE + local_col + w) * K + global_row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                total[w] += localTileA[local_col + w][k] * localTileB[k][local_row];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; ++w) {
        matrixC[(global_col + w) * N + global_row] = total[w];
    }
}
#endif