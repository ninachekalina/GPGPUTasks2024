
#define WORK_GROUP_SIZE 4
#define TILE_SIZE 16

__kernel void fill_with_zeros(__global unsigned int *as, unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    as[gid] = 0;
}

__kernel void count(__global unsigned int *as, __global unsigned int *counters, unsigned int n, unsigned int shift, unsigned int bits_count) {
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    unsigned int value = (as[gid] >> shift) & ((1 << bits_count) - 1); 
    unsigned int wgid = get_group_id(0);
    atomic_inc(&counters[wgid * (1 << bits_count) + value]);
}

__kernel void matrix_transpose_local_good_banks(
    __global float *a,
    __global float *at,
    unsigned int m,
    unsigned int k
) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < k && j < m) {
        tile[local_j][local_i] = a[j * k + i];
    } else {
        tile[local_j][local_i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int target_j = (i - local_i) + local_j;
    int target_i = (j - local_j) + local_i;
    if (target_i < k && target_j < m) {
        at[target_j * k + target_i] = tile[local_i][local_j];
    }
}

__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *bs, unsigned int i, unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    if (gid >= i) {
        bs[gid] = as[gid - i] + as[gid];
    } else {
        bs[gid] = as[gid];
    }
}

__kernel void radix_sort(__global unsigned int *as, __global unsigned int *bs, __global unsigned int *counters, unsigned int n, unsigned int shift, unsigned int bits_count)
{
    unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    unsigned int value = (as[gid] >> shift) & ((1 << bits_count) - 1);
    
    unsigned int wgid = get_group_id(0);
    
    unsigned int start = wgid * WORK_GROUP_SIZE;
    unsigned int end = gid;
    unsigned int offset = 0;
   
    for (unsigned int i = start; i < end; ++i) {
        unsigned int prev_value = (as[i] >> shift) & ((1 << bits_count) - 1);
        if (prev_value == value) {
            ++offset;
        }
    }

    unsigned int base_idx;
    unsigned int counters_idx = wgid + value * WORK_GROUP_SIZE;
    if (counters_idx > 0) {
        base_idx = counters[counters_idx - 1];
    } else {
        base_idx = 0;
    }

    bs[base_idx + offset] = as[gid];
}