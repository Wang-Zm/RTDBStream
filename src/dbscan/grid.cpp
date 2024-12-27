#include <cstring>
#include <execution>
#include "state.h"

void update_grid(ScanState &state, int update_pos, int window_left, int window_right) {
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]--;
    }
    // 遍历 cell_points，查看 cell 中有多少点，若变少了，那么移除前面的点
    for (auto &t: state.cell_points) {
        if (int(t.second.size()) > state.cell_point_num[t.first]) {
            t.second.erase(t.second.begin(), t.second.begin() + t.second.size() - state.cell_point_num[t.first]);
        }
    }
    int pos_start = update_pos * state.stride_size - window_right;
    for (int i = window_right; i < window_right + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.cell_points[cell_id].push_back(pos_start + i);
        state.h_point_cell_id[pos_start + i] = cell_id;
    }
}

void update_grid_without_vector(ScanState &state, int update_pos, int window_left, int window_right) {    
    // 1.更新 h_point_cell_id
    timer.startTimer(&timer.update_h_point_cell_id);
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i); // TODO: Can be implemented in GPU 
        state.cell_point_num[cell_id]--;
    }
    int pos_start = update_pos * state.stride_size - window_right;
    for (int i = window_right; i < window_right + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.h_point_cell_id[pos_start + i] = cell_id;
#ifdef COMPUTE_CELL_CENTER
        if (!state.cell_centers.count(cell_id))
            state.cell_centers[cell_id] = compute_cell_center(state, state.h_data[i]);
#endif
    }
    timer.stopTimer(&timer.update_h_point_cell_id);

    // 2.对 h_point_cell_id 排序，返回 pos_array
    timer.startTimer(&timer.sort_h_point_cell_id);
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;

    // * 对整个 pos_arr 进行排序
    // std::sort(pos_arr, pos_arr + state.window_size, 
    //      [&point_cell_id](size_t i1, size_t i2) { 
    //         return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
    //      });
    
    // 对 point_cell_id 的 stride 部分排序，然后与原有的 pos_arr 合并
    int *new_pos_arr = state.new_pos_arr;
    int stride_left = update_pos * state.stride_size;
    for (int i = stride_left; i < stride_left + state.stride_size; i++) {
        new_pos_arr[i - stride_left] = i;
    }
    timer.startTimer(&timer.sort_new_stride);
    std::sort(new_pos_arr, new_pos_arr + state.stride_size, 
         [point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });
    
    // * 并行排序。结果：排序性能提高，但是其他部分性能降低
    // omp_set_num_threads(32);
    // std::sort(std::execution::par_unseq, new_pos_arr, new_pos_arr + state.stride_size, 
    //      [&point_cell_id](size_t i1, size_t i2) { 
    //         return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
    //      });
    // omp_set_num_threads(1);

    // * gnu 并行排序方法
    // omp_set_num_threads(32);
    // __gnu_parallel::sort(new_pos_arr, new_pos_arr + state.stride_size, 
    //     [&point_cell_id](size_t i1, size_t i2) { 
    //         return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
    // });
    // omp_set_num_threads(1);
    // * 分离编译
    // sort_stride(new_pos_arr, state.stride_size, point_cell_id);
    timer.stopTimer(&timer.sort_new_stride);
    timer.stopTimer(&timer.sort_h_point_cell_id);

    timer.startTimer(&timer.merge_pos_arr);
    int i1 = 0, i2 = 0, i3 = 0;
    int *tmp_pos_arr = state.tmp_pos_arr;
    while (i1 < state.window_size && i2 < state.stride_size) {
        if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
            i1++;
            continue;
        }
        if (point_cell_id[pos_arr[i1]] < point_cell_id[new_pos_arr[i2]]) {
            tmp_pos_arr[i3++] = pos_arr[i1++];
        } else if (point_cell_id[pos_arr[i1]] > point_cell_id[new_pos_arr[i2]]) {
            tmp_pos_arr[i3++] = new_pos_arr[i2++];
        } else { // ==
            if (pos_arr[i1] < new_pos_arr[i2]) {
                tmp_pos_arr[i3++] = pos_arr[i1++];
            } else {
                tmp_pos_arr[i3++] = new_pos_arr[i2++];
            }
        }
    }
    while (i1 < state.window_size) {
        if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
            i1++;
            continue;
        }
        tmp_pos_arr[i3++] = pos_arr[i1++];
    }
    memcpy(tmp_pos_arr + i3, new_pos_arr + i2, (state.stride_size - i2) * sizeof(int));
    memcpy(pos_arr, tmp_pos_arr, state.window_size * sizeof(int));
    timer.stopTimer(&timer.merge_pos_arr);
}

void update_grid_without_vector_parallel(ScanState &state, int update_pos, int window_left, int window_right) {    
    // 1.更新 h_point_cell_id
    timer.startTimer(&timer.update_h_point_cell_id);
    #pragma omp parallel for
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        #pragma omp critical
        {
            state.cell_point_num[cell_id]--;
        }
    }
    // printf("update_h_point_cell_id, first loop\n");

    int pos_start = update_pos * state.stride_size - window_right;
    #pragma omp parallel for
    for (int i = window_right; i < window_right + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        #pragma omp critical
        {
            state.cell_point_num[cell_id]++;
        }
        state.h_point_cell_id[pos_start + i] = cell_id;
        // if (!state.cell_centers.count(cell_id)) {
        //     #pragma omp critical
        //     {
        //         if (!state.cell_centers.count(cell_id)) {
        //             state.cell_centers[cell_id] = compute_cell_center(state, state.h_data[i]);
        //         }
        //     }
        // }
    }
    timer.stopTimer(&timer.update_h_point_cell_id);
    // printf("update_h_point_cell_id\n");

    // 2.对 h_point_cell_id 排序，返回 pos_array
    timer.startTimer(&timer.sort_h_point_cell_id);
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
    // 对 point_cell_id 的 stride 部分排序，然后与原有的 pos_arr 合并
    int *new_pos_arr = state.new_pos_arr;
    int stride_left = update_pos * state.stride_size;
    for (int i = stride_left; i < stride_left + state.stride_size; i++) {
        new_pos_arr[i - stride_left] = i;
    }
    timer.startTimer(&timer.sort_new_stride);
    std::sort(std::execution::par, new_pos_arr, new_pos_arr + state.stride_size, 
     [point_cell_id](size_t i1, size_t i2) { 
        return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
     });
    timer.stopTimer(&timer.sort_new_stride);
    timer.stopTimer(&timer.sort_h_point_cell_id);
    // printf("use std::execution::par\n");

    timer.startTimer(&timer.merge_pos_arr);
    int i1 = 0, i2 = 0, i3 = 0;
    int *tmp_pos_arr = state.tmp_pos_arr;
    while (i1 < state.window_size && i2 < state.stride_size) {
        if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
            i1++;
            continue;
        }
        if (point_cell_id[pos_arr[i1]] < point_cell_id[new_pos_arr[i2]]) {
            tmp_pos_arr[i3++] = pos_arr[i1++];
        } else if (point_cell_id[pos_arr[i1]] > point_cell_id[new_pos_arr[i2]]) {
            tmp_pos_arr[i3++] = new_pos_arr[i2++];
        } else { // ==
            if (pos_arr[i1] < new_pos_arr[i2]) {
                tmp_pos_arr[i3++] = pos_arr[i1++];
            } else {
                tmp_pos_arr[i3++] = new_pos_arr[i2++];
            }
        }
    }
    while (i1 < state.window_size) {
        if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
            i1++;
            continue;
        }
        tmp_pos_arr[i3++] = pos_arr[i1++];
    }
    memcpy(tmp_pos_arr + i3, new_pos_arr + i2, (state.stride_size - i2) * sizeof(int));
    memcpy(pos_arr, tmp_pos_arr, state.window_size * sizeof(int));
    // printf("merge_pos_arr\n");
    timer.stopTimer(&timer.merge_pos_arr);
}

void update_grid_using_unordered_map(ScanState &state, int update_pos, int window_left, int window_right) {
    // 1.更新 h_point_cell_id
    timer.startTimer(&timer.update_h_point_cell_id);
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i); // TODO: Can be implemented in GPU 
        state.cell_point_num[cell_id]--;
    }
    for (auto &t: state.cell_points) {
        if (int(t.second.size()) > state.cell_point_num[t.first]) {
            t.second.erase(t.second.begin(), t.second.begin() + t.second.size() - state.cell_point_num[t.first]);
        }
    }
    int pos_start = update_pos * state.stride_size - window_right;
    for (int i = window_right; i < window_right + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.h_point_cell_id[pos_start + i] = cell_id;
        state.cell_points[cell_id].push_back(pos_start + i);
    }
    timer.stopTimer(&timer.update_h_point_cell_id);

    // 2.对 h_point_cell_id 排序，返回 pos_array
    timer.startTimer(&timer.sort_h_point_cell_id);
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
    // 对 point_cell_id 的 stride 部分排序，然后与原有的 pos_arr 合并
    int *new_pos_arr = state.new_pos_arr;
    int stride_left = update_pos * state.stride_size;
    for (int i = stride_left; i < stride_left + state.stride_size; i++) {
        new_pos_arr[i - stride_left] = i;
    }
    timer.startTimer(&timer.sort_new_stride);
    std::sort(new_pos_arr, new_pos_arr + state.stride_size, 
         [point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });
    timer.stopTimer(&timer.sort_new_stride);
    timer.stopTimer(&timer.sort_h_point_cell_id);

    timer.startTimer(&timer.merge_pos_arr);
    int i1 = 0, i2 = 0, i3 = 0;
    int *tmp_pos_arr = state.tmp_pos_arr;
    while (i1 < state.window_size && i2 < state.stride_size) {
        if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
            i1++;
            continue;
        }
        if (point_cell_id[pos_arr[i1]] < point_cell_id[new_pos_arr[i2]]) {
            tmp_pos_arr[i3++] = pos_arr[i1++];
        } else if (point_cell_id[pos_arr[i1]] > point_cell_id[new_pos_arr[i2]]) {
            tmp_pos_arr[i3++] = new_pos_arr[i2++];
        } else { // ==
            if (pos_arr[i1] < new_pos_arr[i2]) {
                tmp_pos_arr[i3++] = pos_arr[i1++];
            } else {
                tmp_pos_arr[i3++] = new_pos_arr[i2++];
            }
        }
    }
    while (i1 < state.window_size) {
        if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
            i1++;
            continue;
        }
        tmp_pos_arr[i3++] = pos_arr[i1++];
    }
    memcpy(tmp_pos_arr + i3, new_pos_arr + i2, (state.stride_size - i2) * sizeof(int));
    memcpy(pos_arr, tmp_pos_arr, state.window_size * sizeof(int));
    timer.stopTimer(&timer.merge_pos_arr);
}