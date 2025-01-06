#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

__kernel void merge_global(__global const int *input_array, __global int *output_array, unsigned int block_size)
{
    unsigned int global_id = get_global_id(0);
    unsigned int block_id = global_id / block_size;
    unsigned int index_within_block = global_id % block_size;

    unsigned int paired_block_id;
    unsigned int start_of_left_block;
    bool is_left_block = block_id % 2 == 0;

    if (is_left_block) {
        paired_block_id = block_id + 1;
        start_of_left_block = block_id * block_size;
    } else {
        paired_block_id = block_id - 1;
        start_of_left_block = (block_id - 1) * block_size;
    }

    unsigned int start_of_pair_block = paired_block_id * block_size;
    unsigned int left = 0;
    unsigned int right = block_size;
    unsigned int mid;

    int current_value = input_array[global_id];

    while (left < right) {
        mid = (left + right) / 2;
        if (input_array[start_of_pair_block + mid] < current_value ||
            (is_left_block && input_array[start_of_pair_block + mid] == current_value)) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    output_array[start_of_left_block + right + index_within_block] = current_value;
}

__kernel void calculate_indices(__global const int *input_array, __global unsigned int *indices, unsigned int block_size)
{
    // Реализация функции для вычисления индексов
}

__kernel void merge_local(__global const int *input_array, __global const unsigned int *indices, __global int *output_array, unsigned int block_size)
{
    // Реализация функции для локального слияния
}