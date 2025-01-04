#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>

// this can't be in the main cpp file since the file containing cuda kernels to
// be compiled by nvcc needs to have .cu extensions. See here:
// https://github.com/NVIDIA/thrust/issues/614

void thrust_sort(int* keys, int n) {
    thrust::device_ptr<int> keys_ptr(keys);
    thrust::sort(keys_ptr, keys_ptr + n);
}

void sortByCellIdAndOrder(int* d_pos_arr, long* point_cell_id, int n, int start) {
    thrust::device_ptr<int> d_new_pos_arr_ptr(d_pos_arr);
    thrust::device_ptr<long> d_point_cell_id_ptr(point_cell_id);
    thrust::sequence(d_new_pos_arr_ptr, d_new_pos_arr_ptr + n, start);
    auto comparator = [point_cell_id] __device__ (int i1, int i2) {
        return (point_cell_id[i1] == point_cell_id[i2]) ? (i1 < i2) : (point_cell_id[i1] < point_cell_id[i2]);
    };
    thrust::sort(d_new_pos_arr_ptr, d_new_pos_arr_ptr + n, comparator);
}