#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define BLOCK_SIZE 16

__kernel void matrix_transpose_naive(
    __global float *input_matrix,
    __global float *transposed_matrix,
    unsigned int rows,
    unsigned int cols
) {
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);
    
    if (global_row < cols && global_col < rows) {
        transposed_matrix[global_row * rows + global_col] = input_matrix[global_col * cols + global_row];
    }
}

__kernel void matrix_transpose_local_bad_banks(
    __global float *input_matrix,
    __global float *transposed_matrix,
    unsigned int rows,
    unsigned int cols
) {
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    __local float local_tile[BLOCK_SIZE][BLOCK_SIZE];
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    if (global_row < cols && global_col < rows) {
        local_tile[local_col][local_row] = input_matrix[global_col * cols + global_row];
    } else {
        local_tile[local_col][local_row] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int target_col = (global_row - local_row) + local_col;
    int target_row = (global_col - local_col) + local_row;
    if (target_row < cols && target_col < rows) {
        transposed_matrix[target_col * cols + target_row] = local_tile[local_row][local_col];
    }
}

__kernel void matrix_transpose_local_good_banks(
    __global float *input_matrix,
    __global float *transposed_matrix,
    unsigned int rows,
    unsigned int cols
) {
    int global_row = get_global_id(0);
    int global_col = get_global_id(1);

    __local float local_tile[BLOCK_SIZE][BLOCK_SIZE + 1];
    int local_row = get_local_id(0);
    int local_col = get_local_id(1);

    if (global_row < cols && global_col < rows) {
        local_tile[local_col][local_row] = input_matrix[global_col * cols + global_row];
    } else {
        local_tile[local_col][local_row] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int target_col = (global_row - local_row) + local_col;
    int target_row = (global_col - local_col) + local_row;
    if (target_row < cols && target_col < rows) {
        transposed_matrix[target_col * cols + target_row] = local_tile[local_row][local_col];
    }
}