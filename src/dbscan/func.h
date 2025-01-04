#pragma once

void thrust_sort(int* keys, int n);
void sortByCellIdAndOrder(int* d_pos_arr, long* point_cell_id, int n, int start);
void merge_by_cell_id_and_idx(int* pos_arr1, int* pos_arr2, int* pos_arr3, long* point_cell_id, int n, int m, int stride_left);