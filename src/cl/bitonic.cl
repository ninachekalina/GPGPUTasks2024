__kernel void bitonic(__global int *data, unsigned int step_size, unsigned int sub_step_size)
{
    unsigned int thread_id = get_global_id(0);
    unsigned int segment_index = thread_id / step_size;
    bool sort_direction = segment_index % 2 == 0;
    unsigned int local_idx = thread_id / sub_step_size * (sub_step_size * 2) + (thread_id % sub_step_size);

    unsigned int pair_idx = local_idx + sub_step_size;
    if (sort_direction && data[local_idx] > data[pair_idx] ||
        !sort_direction && data[local_idx] < data[pair_idx]
    ) {
        int temp = data[local_idx];
        data[local_idx] = data[pair_idx];
        data[pair_idx] = temp;
    }
}
