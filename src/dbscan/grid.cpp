#include <cstring>
// #include <execution>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include "state.h"
#include "func.h"

void update_grid(ScanState &state, int update_pos, int window_left, int window_right) {
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
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
        state.cell_points[cell_id].push_back(pos_start + i);
        state.h_point_cell_id[pos_start + i] = cell_id;
    }
}

void update_grid_without_vector(ScanState &state, int update_pos, int window_left, int window_right) {    
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

    timer.startTimer(&timer.sort_h_point_cell_id);
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
    
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

// void update_grid_without_vector_parallel(ScanState &state, int update_pos, int window_left, int window_right) {    
//     timer.startTimer(&timer.update_h_point_cell_id);
//     // #pragma omp parallel for
//     for (int i = window_left; i < window_left + state.stride_size; i++) {
//         CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
//         // #pragma omp critical
//         // {
//             state.cell_point_num[cell_id]--;
//         // }
//     }
//     // printf("update_h_point_cell_id, first loop\n");

//     int pos_start = update_pos * state.stride_size - window_right;
//     // #pragma omp parallel for
//     for (int i = window_right; i < window_right + state.stride_size; i++) {
//         CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
//         // #pragma omp critical
//         // {
//             state.cell_point_num[cell_id]++;
//         // }
//         state.h_point_cell_id[pos_start + i] = cell_id;
//         // if (!state.cell_centers.count(cell_id)) {
//         //     #pragma omp critical
//         //     {
//         //         if (!state.cell_centers.count(cell_id)) {
//         //             state.cell_centers[cell_id] = compute_cell_center(state, state.h_data[i]);
//         //         }
//         //     }
//         // }
//     }
//     timer.stopTimer(&timer.update_h_point_cell_id);

//     timer.startTimer(&timer.sort_h_point_cell_id);
//     int *pos_arr = state.pos_arr;
//     CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
//     int *new_pos_arr = state.new_pos_arr;
//     int stride_left = update_pos * state.stride_size;
//     for (int i = stride_left; i < stride_left + state.stride_size; i++) {
//         new_pos_arr[i - stride_left] = i;
//     }
//     timer.startTimer(&timer.sort_new_stride);
//     std::sort(std::execution::par, new_pos_arr, new_pos_arr + state.stride_size, 
//      [point_cell_id](size_t i1, size_t i2) { 
//         return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
//      });
//     // std::this_thread::yield();
//     // std::sort(new_pos_arr, new_pos_arr + state.stride_size, 
//     //  [point_cell_id](size_t i1, size_t i2) { 
//     //     return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
//     //  });
//     timer.stopTimer(&timer.sort_new_stride);
//     timer.stopTimer(&timer.sort_h_point_cell_id);

//     timer.startTimer(&timer.merge_pos_arr);
//     int i1 = 0, i2 = 0, i3 = 0;
//     int *tmp_pos_arr = state.tmp_pos_arr;
//     while (i1 < state.window_size && i2 < state.stride_size) {
//         if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
//             i1++;
//             continue;
//         }
//         if (point_cell_id[pos_arr[i1]] < point_cell_id[new_pos_arr[i2]]) {
//             tmp_pos_arr[i3++] = pos_arr[i1++];
//         } else if (point_cell_id[pos_arr[i1]] > point_cell_id[new_pos_arr[i2]]) {
//             tmp_pos_arr[i3++] = new_pos_arr[i2++];
//         } else { // ==
//             if (pos_arr[i1] < new_pos_arr[i2]) {
//                 tmp_pos_arr[i3++] = pos_arr[i1++];
//             } else {
//                 tmp_pos_arr[i3++] = new_pos_arr[i2++];
//             }
//         }
//     }
//     while (i1 < state.window_size) {
//         if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
//             i1++;
//             continue;
//         }
//         tmp_pos_arr[i3++] = pos_arr[i1++];
//     }
//     memcpy(tmp_pos_arr + i3, new_pos_arr + i2, (state.stride_size - i2) * sizeof(int));
//     memcpy(pos_arr, tmp_pos_arr, state.window_size * sizeof(int));
//     timer.stopTimer(&timer.merge_pos_arr);
// }

void update_grid_thrust(ScanState &state, int update_pos, int window_left, int window_right) {    
    timer.startTimer(&timer.update_h_point_cell_id);
    int stride_left = update_pos * state.stride_size;
    compute_cell_id(state.params.window + stride_left, state.params.point_cell_id + stride_left, state.stride_size,
                    state.params.min_value, state.params.cell_count, state.params.cell_length);
    timer.stopTimer(&timer.update_h_point_cell_id);

    timer.startTimer(&timer.sort_h_point_cell_id);
    timer.startTimer(&timer.sort_new_stride);
    sortByCellIdAndOrder(state.params.new_pos_arr, state.params.point_cell_id, state.stride_size, stride_left);
    timer.stopTimer(&timer.sort_new_stride);
    timer.stopTimer(&timer.sort_h_point_cell_id);

    timer.startTimer(&timer.merge_pos_arr);
    merge_by_cell_id_and_idx(state.params.pos_arr, state.params.new_pos_arr, state.params.tmp_pos_arr, 
                             state.params.point_cell_id, state.window_size, state.stride_size, stride_left);
    swap(state.params.pos_arr, state.params.tmp_pos_arr);
    timer.stopTimer(&timer.merge_pos_arr);
}

void update_grid_using_unordered_map(ScanState &state, int update_pos, int window_left, int window_right) {
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

    timer.startTimer(&timer.sort_h_point_cell_id);
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
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

void update_grid_with_timestamp(ScanState &state, int update_pos, int window_left, int window_right) {    
    unordered_map<CELL_ID_TYPE, int> remove_mp, add_mp;
    timer.startTimer(&timer.update_h_point_cell_id);
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i); // TODO: Can be implemented in GPU 
        // state.cell_point_num[cell_id]--;
        remove_mp[cell_id]++;
    }
    int pos_start = update_pos * state.stride_size - window_right;
    for (int i = window_right; i < window_right + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        // state.cell_point_num[cell_id]++;
        add_mp[cell_id]++;
        state.h_point_cell_id[pos_start + i] = cell_id;
        state.h_timestamp[pos_start + i] = i;
    }
    timer.stopTimer(&timer.update_h_point_cell_id);

    timer.startTimer(&timer.sort_h_point_cell_id);
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
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
            // tmp_pos_arr[i3++] = pos_arr[i1++];
            int num_points = state.cell_point_num[point_cell_id[pos_arr[i1]]] - remove_mp[point_cell_id[pos_arr[i1]]];
            memcpy(tmp_pos_arr + i3, pos_arr + i1, num_points * sizeof(int));
            i3 += num_points;
            i1 += num_points;
        } else if (point_cell_id[pos_arr[i1]] > point_cell_id[new_pos_arr[i2]]) {
            // tmp_pos_arr[i3++] = new_pos_arr[i2++];
            int& num_points = add_mp[point_cell_id[new_pos_arr[i2]]];
            memcpy(tmp_pos_arr + i3, new_pos_arr + i2, num_points * sizeof(int));
            i3 += num_points;
            i2 += num_points;
        } else { // ==
            // if (pos_arr[i1] < new_pos_arr[i2]) {
            //     tmp_pos_arr[i3++] = pos_arr[i1++];
            // } else {
            //     tmp_pos_arr[i3++] = new_pos_arr[i2++];
            // }
            int num_points = state.cell_point_num[point_cell_id[pos_arr[i1]]] - remove_mp[point_cell_id[pos_arr[i1]]];
            memcpy(tmp_pos_arr + i3, pos_arr + i1, num_points * sizeof(int));
            i3 += num_points;
            i1 += num_points;
            num_points = add_mp[point_cell_id[new_pos_arr[i2]]];
            memcpy(tmp_pos_arr + i3, new_pos_arr + i2, num_points * sizeof(int));
            i3 += num_points;
            i2 += num_points;
        }
    }
    while (i1 < state.window_size) {
        if (pos_arr[i1] >= stride_left && pos_arr[i1] < stride_left + state.stride_size) {
            i1++;
            continue;
        }
        // tmp_pos_arr[i3++] = pos_arr[i1++];
        int num_points = state.cell_point_num[point_cell_id[pos_arr[i1]]] - remove_mp[point_cell_id[pos_arr[i1]]];
        memcpy(tmp_pos_arr + i3, pos_arr + i1, num_points * sizeof(int));
        i3 += num_points;
        i1 += num_points;
    }
    memcpy(tmp_pos_arr + i3, new_pos_arr + i2, (state.stride_size - i2) * sizeof(int));
    memcpy(pos_arr, tmp_pos_arr, state.window_size * sizeof(int));
    for (auto& item : remove_mp) {
        state.cell_point_num[item.first] -= item.second;
    }
    for (auto& item : add_mp) {
        state.cell_point_num[item.first] += item.second;
    }
    timer.stopTimer(&timer.merge_pos_arr);
}