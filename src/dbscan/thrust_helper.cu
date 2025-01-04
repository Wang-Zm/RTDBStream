#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/partition.h>
#include "func.h"

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

void merge_by_cell_id_and_idx(int* pos_arr1, int* pos_arr2, int* pos_arr3, long* point_cell_id, int n, int m, int stride_left) {
    thrust::device_ptr<int> pos_ptr1(pos_arr1);
    thrust::device_ptr<int> pos_ptr2(pos_arr2);
    thrust::device_ptr<int> pos_ptr3(pos_arr3);
    int stride_right = stride_left + m;
    
    auto partition = [stride_left, stride_right] __device__ (int x) {
        return x < stride_left || x >= stride_right;
    };
    thrust::partition(pos_ptr1, pos_ptr1 + n, partition);
    
    auto custom_compare = [point_cell_id, stride_left, stride_right, n] __device__ (int a, int b) { // 把别的放到最后面
        if (point_cell_id[a] != point_cell_id[b]) 
            return point_cell_id[a] < point_cell_id[b];
        return a < b;
    };
    // 使用 thrust::merge 并传递自定义比较函数
    thrust::merge(pos_ptr1, pos_ptr1 + n - m,
                  pos_ptr2, pos_ptr2 + m,
                  pos_ptr3,
                  custom_compare);
}