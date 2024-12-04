#include <optix.h>
// #include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <map>
#include <queue>
#include <thread>

#include "state.h"
#include "timer.h"

using namespace std;

Timer timer;

void initialize_params(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    state.cell_length   = state.radius / sqrt(3);
    for (int i = 0; i < 3; i++) {
        state.cell_count.push_back(int((state.max_value[i] - state.min_value[i] + state.cell_length) / state.cell_length));
    }
    state.radius_one_half = state.radius * 1.5;

    state.params.radius = state.radius;
    state.params.radius2 = state.radius * state.radius;
    state.params.radius_one_half2 = state.radius_one_half * state.radius_one_half;
    state.params.cell_length = state.cell_length;
    state.params.tmin   = 0.0f;
    state.params.tmax   = FLT_MIN;
    state.params.min_pts= state.min_pts;
    state.params.window_size = state.window_size;
    CUDA_CHECK(cudaMalloc(&state.params.cluster_id, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.label, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.nn, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.window, state.window_size * sizeof(DATA_TYPE_3)));
    CUDA_CHECK(cudaMalloc(&state.params.out_stride, state.stride_size * sizeof(DATA_TYPE_3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));
    CUDA_CHECK(cudaMallocHost(&state.h_cluster_id, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&state.h_label, state.window_size * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&state.params.centers, state.window_size * sizeof(DATA_TYPE_3)));
    CUDA_CHECK(cudaMalloc(&state.params.radii, state.window_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&state.params.point_cell_id, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.center_idx_in_window, state.window_size * sizeof(int)));
    state.h_point_cell_id = (CELL_ID_TYPE*) malloc(state.window_size * sizeof(int));

    CUDA_CHECK(cudaMalloc(&state.params.cell_points, state.window_size * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&state.params.cell_point_num, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.points_in_dense_cells, state.window_size * sizeof(int)));
    state.d_cell_points = (int**) malloc(state.window_size * sizeof(int*));
    for (int i = 0; i < state.window_size; i++) state.d_cell_points[i] = nullptr;
    state.points_in_dense_cells = (int*) malloc(state.window_size * sizeof(int));
    CUDA_CHECK(cudaMalloc(&state.params.pos_arr, 2 * state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.point_status, state.window_size * sizeof(bool)));
    CUDA_CHECK(cudaMallocHost(&state.pos_arr, state.window_size * sizeof(int)));
    state.tmp_pos_arr = (int*) malloc(state.window_size * sizeof(int));
    state.new_pos_arr = (int*) malloc(state.stride_size * sizeof(int));
    CUDA_CHECK(cudaMallocHost(&state.uniq_pos_arr, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&state.num_points, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.cell_count, 3 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.params.cell_count, state.cell_count.data(), 3 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&state.params.min_value, 3 * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMemcpy(state.params.min_value, state.min_value.data(), 3 * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaMalloc(&state.params.ray_intersections, state.query_num * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&state.params.ray_primitive_hits, state.query_num * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.query_num * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.query_num * sizeof(unsigned)));
    state.h_ray_intersections = (unsigned *) malloc(state.query_num * sizeof(unsigned));
    state.h_ray_hits = (unsigned *) malloc(state.query_num * sizeof(unsigned));
#endif
    CUDA_CHECK(cudaMalloc(&state.params.cluster_ray_intersections, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(state.params.cluster_ray_intersections, 0, sizeof(unsigned)));

    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_pos, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_num, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_list, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_capacity, state.window_size * sizeof(int)));

    state.neighbor_cells_pos = (int*) malloc(state.window_size * sizeof(int));
    state.neighbor_cells_num = (int*) malloc(state.window_size * sizeof(int));

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] initialize_params: " << 1.0 * used / (1 << 20) << std::endl;

    // Allocate memory for CUDA implementation
    if (state.check) {
        CUDA_CHECK(cudaMalloc(&state.params.check_nn, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&state.params.check_label, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&state.params.check_cluster_id, state.window_size * sizeof(int)));
        state.check_h_nn = (int*) malloc(state.window_size * sizeof(int)); 
        state.check_h_label = (int*) malloc(state.window_size * sizeof(int));
        state.check_h_cluster_id = (int*) malloc(state.window_size * sizeof(int));
        state.h_nn = (int*) malloc(state.window_size * sizeof(int));
    }
}

void log_common_info(ScanState &state) {
    std::cout << "Input file: " << state.data_file << std::endl;
    std::cout << "Data num: " << state.data_num << std::endl;
    std::cout << "Window size: " << state.window_size << ", Stride size: " << state.stride_size << ", K: " << state.min_pts << std::endl;
    std::cout << "Radius: " << state.radius << ", Radius2: " << state.params.radius2 << std::endl;
    std::cout << "Cell length: " << state.cell_length << std::endl;
    for (int i = 0; i < 3; i++) std::cout << "Cell count: " << state.cell_count[i] << ", ";
    std::cout << std::endl;
    std::cout << "OPTIMIZATION_LEVEL: " << OPTIMIZATION_LEVEL << std::endl;
}

inline CELL_ID_TYPE get_cell_id(DATA_TYPE_3* data, vector<DATA_TYPE>& min_value, vector<int>& cell_count, DATA_TYPE cell_length, int i) {
    CELL_ID_TYPE dim_id_x = (data[i].x - min_value[0]) / cell_length;
    CELL_ID_TYPE dim_id_y = (data[i].y - min_value[1]) / cell_length;
    CELL_ID_TYPE dim_id_z = (data[i].z - min_value[2]) / cell_length;
    CELL_ID_TYPE id = dim_id_x * cell_count[1] * cell_count[2] + dim_id_y * cell_count[2] + dim_id_z;
    return id;
}

void find_neighbors_cores(ScanState &state, int update_pos) {
    state.params.out = state.params.window + update_pos * state.stride_size;
    state.params.out_stride_handle = state.handle_list[update_pos];
    state.params.stride_left = update_pos * state.stride_size;
    state.params.stride_right = state.params.stride_left + state.stride_size;
    memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));

    CUDA_CHECK(cudaEventCreate(&timer.start2));
    CUDA_CHECK(cudaEventCreate(&timer.stop2));
    CUDA_CHECK(cudaEventRecord(timer.start2, state.stream));
    CUDA_CHECK(cudaMemcpyAsync(state.params.out_stride, state.params.out, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyDeviceToDevice, state.stream));
    CUDA_CHECK(cudaMemcpyAsync(state.params.out, state.h_window + update_pos * state.stride_size, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice, state.stream));
    CUDA_CHECK(cudaMemsetAsync(state.params.nn + update_pos * state.stride_size, 0, state.stride_size * sizeof(int), state.stream));
    rebuild_gas_stride(state, update_pos);
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice, state.stream));
    OPTIX_CHECK(optixLaunch(state.pipeline, state.stream, state.d_params, sizeof(Params), &state.sbt, state.window_size, 2, 1));
    
    find_cores(state.params.label, state.params.nn, state.params.cluster_id, state.window_size, state.min_pts, state.stream);
    CUDA_CHECK(cudaEventRecord(timer.stop2, state.stream));
    // CUDA_CHECK(cudaEventSynchronize(timer.stop2));
    // CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds2, timer.start2, timer.stop2));
    // timer.find_cores += timer.milliseconds2;
    // CUDA_CHECK(cudaEventDestroy(timer.start2));
    // CUDA_CHECK(cudaEventDestroy(timer.stop2));
}

void update_grid(ScanState &state, int update_pos, int window_left, int window_right) {
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
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
        int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.cell_points[cell_id].push_back(pos_start + i);
        state.h_point_cell_id[pos_start + i] = cell_id;
    }
}

void set_centers_radii_cpu(ScanState &state, int* pos_arr) {
    // 3.获取球的球心和半径
    timer.startTimer(&timer.get_centers_radii);
    state.h_centers.clear();
    state.h_radii.clear();
    state.h_center_idx_in_window.clear();
    state.h_cell_point_num.clear();
    int j = 0, sphere_id = 0;
    while (j < state.window_size) {
        int cell_id = state.h_point_cell_id[pos_arr[j]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num >= state.min_pts) {
            int i = pos_arr[j];
            int dim_id_x = (state.h_window[i].x - state.min_value[0]) / state.cell_length;
            int dim_id_y = (state.h_window[i].y - state.min_value[1]) / state.cell_length;
            int dim_id_z = (state.h_window[i].z - state.min_value[2]) / state.cell_length;
            DATA_TYPE_3 center = { state.min_value[0] + (dim_id_x + 0.5) * state.cell_length, 
                                   state.min_value[1] + (dim_id_y + 0.5) * state.cell_length, 
                                   state.min_value[2] + (dim_id_z + 0.5) * state.cell_length };
            state.h_centers.push_back(center);
            state.h_radii.push_back(state.radius_one_half);
            state.h_center_idx_in_window.push_back(i);
            state.h_cell_point_num.push_back(point_num); // 记录点数多少
            state.d_cell_points[sphere_id] = state.params.pos_arr + j;
            
            // 设置该 cell 中的点的 cid
            for (int t = 0; t < point_num; t++) {
                state.h_cluster_id[pos_arr[j++]] = i; 
            }

            sphere_id++;
        } else {
            for (int t = 0; t < point_num; t++) {
                int i = pos_arr[j];
                state.h_centers.push_back(state.h_window[i]);
                state.h_radii.push_back(state.radius);
                state.h_center_idx_in_window.push_back(i);
                state.h_cell_point_num.push_back(1); // 占空
                state.h_cluster_id[i] = i;
                j++;
                sphere_id++;
            }
        }
    }
    timer.stopTimer(&timer.get_centers_radii);

    // 4.传送到 GPU
    timer.startTimer(&timer.cell_points_memcpy); // TODO: Merge these cudaMemcpy calls
    CUDA_CHECK(cudaMemcpy(state.params.pos_arr, // 将 pos_arr 复制到 GPU
                          pos_arr, 
                          state.window_size * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.centers, state.h_centers.data(), state.h_centers.size() * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.radii, state.h_radii.data(), state.h_radii.size() * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.center_idx_in_window, 
                          state.h_center_idx_in_window.data(), 
                          state.h_center_idx_in_window.size() * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cell_points, 
                          state.d_cell_points, 
                          state.h_centers.size() * sizeof(int*), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cell_point_num, 
                          state.h_cell_point_num.data(), 
                          state.h_cell_point_num.size() * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cluster_id, 
                          state.h_cluster_id, 
                          state.window_size * sizeof(int), 
                          cudaMemcpyHostToDevice));
    state.params.center_num = state.h_centers.size();
    timer.stopTimer(&timer.cell_points_memcpy);
}

void set_centers_radii_gpu(ScanState &state, int* pos_arr) {
    timer.startTimer(&timer.compute_uniq_pos_arr);
    int *uniq_pos_arr = state.uniq_pos_arr;
    int *num_points = state.num_points;
    int num_centers = 0;
    for (int i1 = 0; i1 < state.window_size; i1++) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos_arr[i1]];
        int point_num = state.cell_point_num[cell_id];
        uniq_pos_arr[num_centers] = i1;
        num_points[num_centers] = 1;
        if (point_num >= state.min_pts) {
            i1 += point_num - 1;
            num_points[num_centers] = point_num;
        }
        num_centers++;
    }
    timer.stopTimer(&timer.compute_uniq_pos_arr);

    // TODO: Explore why recording time adds latency
    // ! When use recording, compute_uniq_pos_arr increases a lot.
    state.params.center_num = num_centers;
    // CUDA_CHECK(cudaEventCreate(&timer.start1));
    // CUDA_CHECK(cudaEventCreate(&timer.stop1));
    // CUDA_CHECK(cudaEventRecord(timer.start1, state.stream2));
    CUDA_CHECK(cudaMemcpyAsync(state.params.pos_arr + state.window_size, uniq_pos_arr, state.params.center_num * sizeof(int), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(state.params.cell_point_num, num_points, state.params.center_num * sizeof(int), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaMemcpyAsync(state.params.pos_arr, pos_arr, state.window_size * sizeof(int), cudaMemcpyHostToDevice, 0));
    set_centers_radii(
        state.params.window, state.params.radius, state.params.pos_arr, state.params.pos_arr + state.window_size, state.params.cell_point_num, 
        state.params.min_pts, state.params.min_value, state.params.cell_length, state.params.center_num,
        state.params.centers, state.params.radii, state.params.cluster_id, state.params.cell_points, state.params.center_idx_in_window,
        0
    );
    // CUDA_CHECK(cudaEventRecord(timer.stop1, state.stream2));
    // CUDA_CHECK(cudaEventSynchronize(timer.stop1));
    // CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds1, timer.start1, timer.stop1));
    // timer.set_centers_radii += timer.milliseconds1;
    // CUDA_CHECK(cudaEventDestroy(timer.start1));
    // CUDA_CHECK(cudaEventDestroy(timer.stop1));
}

void set_centers_sparse(ScanState &state) {
    // Put points in sparse cells into state.params.centers
    timer.startTimer(&timer.early_cluster);
    state.h_centers.clear();
    state.h_center_idx_in_window.clear();
    int num_dense_cells = 0, num_sparse_points = 0;
    for (auto& item : state.cell_points) {
        if ((int) item.second.size() < state.min_pts) {
            for (int& pos : item.second) {
                state.h_centers.push_back(state.h_window[pos]);
                state.h_cluster_id[pos] = pos;
            }
            state.h_center_idx_in_window.insert(state.h_center_idx_in_window.end(), item.second.begin(), item.second.end());
            num_sparse_points += item.second.size();
        } else {
            // Set cluster_id
            int id = item.second.front();
            for (int& pos : item.second) {
                state.h_cluster_id[pos] = id; // ! Early cluster is not effective
            }
            num_dense_cells++;
        }
    }
    CUDA_CHECK(cudaMemcpy(state.params.centers, state.h_centers.data(), state.h_centers.size() * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.center_idx_in_window, state.h_center_idx_in_window.data(), state.h_centers.size() * sizeof(int), cudaMemcpyHostToDevice));
    state.params.center_num = state.h_centers.size();
    timer.stopTimer(&timer.early_cluster);
    printf("num_dense_cells: %d\n", num_dense_cells);
    printf("num_sparse_points: %d\n", num_sparse_points);
}

void set_centers_sparse_without_vector(ScanState &state) {
    // Put points in sparse cells into state.params.centers, set centers from pos_arr
    timer.startTimer(&timer.early_cluster);
    state.h_centers.clear();
    state.h_center_idx_in_window.clear();
    state.pos_of_cell.clear();
    int *pos_arr = state.pos_arr;
    int j = 0;
    int num_dense_cells = 0, num_sparse_points = 0;
    while (j < state.window_size) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos_arr[j]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num < state.min_pts) {
            int pos_arr_start = j;
            for (int k = 0; k < point_num; k++) {
                state.h_cluster_id[pos_arr[j]] = pos_arr[j];
                state.h_centers.push_back(state.h_window[pos_arr[j++]]);
            }
            state.h_center_idx_in_window.insert(state.h_center_idx_in_window.end(), pos_arr + pos_arr_start, pos_arr + j);
            num_sparse_points += point_num;
        } else {
            state.pos_of_cell[cell_id] = j; // Set for dense cells
            num_dense_cells++;
            int id = pos_arr[j];
            for (int k = 0; k < point_num; k++) {
                state.h_cluster_id[pos_arr[j]] = id;
                j++;
            }
        }
    }
    CUDA_CHECK(cudaMemcpy(state.params.centers, state.h_centers.data(), state.h_centers.size() * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.center_idx_in_window, state.h_center_idx_in_window.data(), state.h_center_idx_in_window.size() * sizeof(int), cudaMemcpyHostToDevice));
    state.params.center_num = state.h_centers.size();
    timer.stopTimer(&timer.early_cluster);
    // printf("num_dense_cells: %d\n", num_dense_cells);
    // printf("num_sparse_points: %d\n", num_sparse_points);
}

void update_grid_without_vector(ScanState &state, int update_pos, int window_left, int window_right) {    
    // 1.更新 h_point_cell_id
    timer.startTimer(&timer.update_h_point_cell_id);
    for (int i = window_left; i < window_left + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i); // TODO: Can be implemented in GPU 
        state.cell_point_num[cell_id]--; // TODO: Use GPU's hashtable to maintain cell_point_num
    }
    int pos_start = update_pos * state.stride_size - window_right;
    for (int i = window_right; i < window_right + state.stride_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.h_point_cell_id[pos_start + i] = cell_id;
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
    sort(new_pos_arr, new_pos_arr + state.stride_size, 
         [&point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });
    // timer.stopTimer(&timer.sort_h_point_cell_id);

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
    timer.stopTimer(&timer.sort_h_point_cell_id);
}

void get_centers_radii_device(ScanState &state) {
    timer.startTimer(&timer.get_dense_sphere);
    state.h_centers.clear();
    state.h_radii.clear();
    state.h_center_idx_in_window.clear();
    state.h_cell_point_num.clear();
    unordered_map<int, int> &cell_repres = state.cell_repres;
    cell_repres.clear();
    vector<int> cell_sphere_num_list, cell_id_list;
    for (int i = 0; i < state.window_size; i++) {
        int cell_id = get_cell_id(state.h_window, state.min_value, state.cell_count, state.cell_length, i);
        if (state.cell_point_num[cell_id] >= state.min_pts) {
            if (cell_repres.count(cell_id) > 0) continue;
            int dim_id_x = (state.h_window[i].x - state.min_value[0]) / state.cell_length;
            int dim_id_y = (state.h_window[i].y - state.min_value[1]) / state.cell_length;
            int dim_id_z = (state.h_window[i].z - state.min_value[2]) / state.cell_length;
            DATA_TYPE_3 center = { state.min_value[0] + (dim_id_x + 0.5) * state.cell_length, 
                                   state.min_value[1] + (dim_id_y + 0.5) * state.cell_length, 
                                   state.min_value[2] + (dim_id_z + 0.5) * state.cell_length };
            state.h_centers.push_back(center);
            state.h_radii.push_back(state.radius_one_half);
            cell_repres[cell_id] = i;
            state.h_center_idx_in_window.push_back(i);

            // TODO: 可以使用原子操作计算当前是第一个 center，然后多线程加速
            int cell_sphere_num = state.h_centers.size() - 1; // 和 primIdx 保持一致
            cell_id_list.push_back(cell_id);
            cell_sphere_num_list.push_back(cell_sphere_num); // 表示第几个 point 是 sphere 的 center
            state.h_cell_point_num.push_back(state.cell_point_num[cell_id]); // 记录点数多少
        } else {
            state.h_centers.push_back(state.h_window[i]);
            state.h_radii.push_back(state.radius);
            state.h_center_idx_in_window.push_back(i);

            state.h_cell_point_num.push_back(1);
        }
    }
    timer.stopTimer(&timer.get_dense_sphere);

    timer.startTimer(&timer.dense_cell_points_copy);
    int curr_pos = 0;
    for (size_t i = 0; i < cell_sphere_num_list.size(); i++) {
        // Strategy 1: 为每个 dense cell 申请空间，将数据传输其中
        // CUDA_CHECK(cudaMalloc(&state.d_cell_points[cell_sphere_num_list[i]], state.cell_point_num[cell_id_list[i]] * sizeof(int)));
        // timer.startTimer(&timer.spheres_copy);
        // CUDA_CHECK(cudaMemcpy(state.d_cell_points[cell_sphere_num_list[i]], 
        //                         state.cell_points[cell_id_list[i]].data(),
        //                         state.cell_point_num[cell_id_list[i]] * sizeof(int),
        //                         cudaMemcpyHostToDevice));
        // timer.stopTimer(&timer.spheres_copy);

        // Strategy 2: 将数据复制到一维空间，然后设置 state.params.cell_points；
        memcpy(state.points_in_dense_cells + curr_pos, 
               state.cell_points[cell_id_list[i]].data(), 
               state.cell_point_num[cell_id_list[i]] * sizeof(int));
        state.d_cell_points[cell_sphere_num_list[i]] = state.params.points_in_dense_cells + curr_pos;
        curr_pos += state.cell_point_num[cell_id_list[i]];
    }
    CUDA_CHECK(cudaMemcpy(state.params.points_in_dense_cells,
                          state.points_in_dense_cells,
                          curr_pos * sizeof(int),
                          cudaMemcpyHostToDevice));
    timer.stopTimer(&timer.dense_cell_points_copy);

    timer.startTimer(&timer.cell_points_memcpy);
    CUDA_CHECK(cudaMemcpy(state.params.centers, state.h_centers.data(), state.h_centers.size() * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.radii, state.h_radii.data(), state.h_radii.size() * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.center_idx_in_window, 
                          state.h_center_idx_in_window.data(), 
                          state.h_center_idx_in_window.size() * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cell_points, 
                          state.d_cell_points, 
                          state.h_centers.size() * sizeof(int*), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cell_point_num, 
                          state.h_cell_point_num.data(), 
                          state.h_cell_point_num.size() * sizeof(int), 
                          cudaMemcpyHostToDevice));
    state.params.center_num = state.h_centers.size();
    timer.stopTimer(&timer.cell_points_memcpy);
}

void early_cluster(ScanState &state) { // 根据所属的 cell，快速设置 cluster id
    unordered_map<int, int> &cell_repres = state.cell_repres;
    for (int i = 0; i < state.window_size; i++) {
        int cell_id = state.h_point_cell_id[i];
        if (state.cell_point_num[cell_id] >= state.min_pts) {
            state.h_cluster_id[i] = cell_repres[cell_id];
        } else {
            state.h_cluster_id[i] = i;
        }
    }
    CUDA_CHECK(cudaMemcpy(state.params.cluster_id, state.h_cluster_id, state.window_size * sizeof(int), cudaMemcpyHostToDevice)); // TODO: 之前 find_cores 方法中对这个的设置是没必要的，可以去掉
}

vector<CELL_ID_TYPE> calc_neighbor_cells(ScanState &state, DATA_TYPE_3 point) {
    vector<CELL_ID_TYPE> res;
    CELL_ID_TYPE dim_id_x = (point.x - state.min_value[0]) / state.cell_length;
    CELL_ID_TYPE dim_id_y = (point.y - state.min_value[1]) / state.cell_length;
    CELL_ID_TYPE dim_id_z = (point.z - state.min_value[2]) / state.cell_length;
    // 2Eps-dist cell centers. Take three-dimensional situation as an example
    double dist_thres = 2 * state.radius;
    double bound = dist_thres / state.cell_length;
    int len = (int) bound;
    for (int dx = 0; dx <= len; dx++) {
        if (dx == 0) {
            for (int dy = 0; dy <= len; dy++) {
                for (int dz = (dy == 0 ? 0 : -len); dz <= len; dz++) {
                    if (dy == 0 && dz == 0) {
                        continue;
                    }
                    if (sqrt(dx * dx + dy * dy + dz * dz) * state.cell_length < dist_thres) {
                        CELL_ID_TYPE nx = dim_id_x + dx, ny = dim_id_y + dy, nz = dim_id_z + dz;
                        CELL_ID_TYPE id = nx * state.cell_count[1] * state.cell_count[2] + ny * state.cell_count[2] + nz;
                        if (state.cell_point_num[id] >= state.min_pts) { // 必须判别是否是 dense，是 dense 再放进去
                            res.push_back(id);
                        }
                    }
                }
            }
        } else {
            for (int dy = -len; dy <= len; dy++) {
                for (int dz = -len; dz <= len; dz++) {
                    if (sqrt(dx * dx + dy * dy + dz * dz) * state.cell_length < dist_thres) {
                        CELL_ID_TYPE nx = dim_id_x + dx, ny = dim_id_y + dy, nz = dim_id_z + dz;
                        CELL_ID_TYPE id = nx * state.cell_count[1] * state.cell_count[2] + ny * state.cell_count[2] + nz;
                        if (state.cell_point_num[id] >= state.min_pts) { // 必须判别是否是 dense，是 dense 再放进去
                            res.push_back(id);
                        }
                    }
                }
            }
        }
    }
    return res;
}

bool check_neighbor_cells(ScanState &state, DATA_TYPE_3 p1, DATA_TYPE_3 p2) {
    int x1 = (p1.x - state.min_value[0]) / state.cell_length;
    int y1 = (p1.y - state.min_value[1]) / state.cell_length;
    int z1 = (p1.z - state.min_value[2]) / state.cell_length;
    int x2 = (p2.x - state.min_value[0]) / state.cell_length;
    int y2 = (p2.y - state.min_value[1]) / state.cell_length;
    int z2 = (p2.z - state.min_value[2]) / state.cell_length;
    if (x2 < x1 || (x2 == x1 && y2 < y1) || (x2 == x1 && y2 == y1 && z2 < z1)) {
        return false;
    }
    int dx = x2 - x1, dy = y2 - y1, dz = z2 - z1;
    if (sqrt(dx * dx + dy * dy + dz * dz) * state.cell_length < 2 * state.radius) {
        return true;
    }
    return false;
}

// 基于从当前 cell 向后扩散的方式找 neighbor cells
map<CELL_ID_TYPE, vector<CELL_ID_TYPE>> find_neighbor_cells_extend(ScanState &state) {
    int *pos_arr = state.pos_arr;
    map<CELL_ID_TYPE, vector<CELL_ID_TYPE>> neighbor_cells_of_dense_cells;
    int j = 0;
    while (j < state.window_size) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos_arr[j]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num >= state.min_pts) {
            vector<CELL_ID_TYPE> neighbor_cells = calc_neighbor_cells(state, state.h_window[pos_arr[j]]);
            neighbor_cells_of_dense_cells[cell_id] = neighbor_cells;
        }
        j += point_num;
    }
    return neighbor_cells_of_dense_cells;
}

// Very slow
map<CELL_ID_TYPE, vector<CELL_ID_TYPE>> find_neighbor_cells(ScanState &state) {
    int *pos_arr = state.pos_arr;
    map<CELL_ID_TYPE, vector<CELL_ID_TYPE>> neighbor_cells_of_dense_cells;
    int i = 0;
    while (i < state.window_size) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos_arr[i]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num >= state.min_pts) {
            // 找 cell_id 的邻居
            int j = i + point_num;
            CELL_ID_TYPE other_cell_id;
            while (j < state.window_size) {
                other_cell_id = state.h_point_cell_id[pos_arr[j]];
                if (state.cell_point_num[other_cell_id] >= state.min_pts &&
                    check_neighbor_cells(state, state.h_window[pos_arr[i]], state.h_window[pos_arr[j]])) {
                    neighbor_cells_of_dense_cells[cell_id].push_back(other_cell_id);
                }
                j += state.cell_point_num[other_cell_id];
            }
        }
        i += point_num;
    }
    return neighbor_cells_of_dense_cells;
}

void check_found_neighbor_cells(map<CELL_ID_TYPE, vector<CELL_ID_TYPE>>& mp1,
                                map<CELL_ID_TYPE, vector<CELL_ID_TYPE>>& mp2) {
    for (auto& item : mp2) {
        if (item.second.size() != mp1[item.first].size()) {
            printf("Error on number of neighbor cells, verify_nei_cells[%ld].size = %lu, "
                   "neighbor_cells_of_dense_cells[%ld].size = %lu\n",
                   item.first, item.second.size(), item.first, mp1[item.first].size());
            exit(0);
        }
        for (int i = 0; i < (int) item.second.size(); i++) {
            if (item.second[i] != mp1[item.first][i]) {
                printf("Error on neighbor cells, cell_id = %lu\n", item.first);
                exit(0);
            }
        }
    }
    printf("neighbor cells check correct\n");
}

void find_neighbors_of_cells(ScanState &state) {
    timer.startTimer(&timer.find_neighbor_cells);
    map<CELL_ID_TYPE, vector<CELL_ID_TYPE>> neighbor_cells_of_dense_cells = find_neighbor_cells_extend(state);
    timer.stopTimer(&timer.find_neighbor_cells);
    
    // point->cell_id, cell_id->neighbors_cells, neighbors_cells->(start_pos, len) list
    timer.startTimer(&timer.put_neighbor_cells_list);
    vector<int> &neighbor_cells_list = state.neighbor_cells_list;
    vector<int> &neighbor_cells_capacity = state.neighbor_cells_capacity;
    neighbor_cells_list.clear();
    neighbor_cells_capacity.clear();
    unordered_map<int, pair<int, int>> neighbor_cells_pos_and_num;
    for (auto& item : neighbor_cells_of_dense_cells) { // List in cell_id's order
        neighbor_cells_pos_and_num[item.first] = {neighbor_cells_list.size(), item.second.size()};
        for (CELL_ID_TYPE& cell_id : item.second) {
            neighbor_cells_list.push_back(state.pos_of_cell[cell_id]);
            neighbor_cells_capacity.push_back(state.cell_point_num[cell_id]);
        }
    }
    // 将 neighbor_cells_list 和 neighbor_cells_capacity 放到邻居列表中
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_list, 
                          neighbor_cells_list.data(), 
                          neighbor_cells_list.size() * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_capacity, 
                          neighbor_cells_capacity.data(), 
                          neighbor_cells_capacity.size() * sizeof(int), 
                          cudaMemcpyHostToDevice));
    timer.stopTimer(&timer.put_neighbor_cells_list);

    // 也要记录每个 dense cell 的 neighbors 的开始位置，维护好一个这样的映射
    // 为每个 dense cell 中的点准备这些信息，让其通过 CUDA 加速的时候能够有效果
    timer.startTimer(&timer.prepare_for_points_in_dense_cells);
    int* neighbor_cells_pos = state.neighbor_cells_pos;
    int* neighbor_cells_num = state.neighbor_cells_num;
    int j = 0;
    // TODO：可以用多线程加速
    while (j < state.window_size) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[j];
        if (neighbor_cells_pos_and_num.count(cell_id)) {
            neighbor_cells_pos[j] = neighbor_cells_pos_and_num[cell_id].first;
            neighbor_cells_num[j] = neighbor_cells_pos_and_num[cell_id].second;
        } else {
            neighbor_cells_pos[j] = -1;
            neighbor_cells_num[j] = -1;
        }
        j++;
    }
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_pos, neighbor_cells_pos, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_num, neighbor_cells_num, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
    timer.stopTimer(&timer.prepare_for_points_in_dense_cells);
}

void cluster_dense_cells_cpu(ScanState &state) {
    // 用 map 记录每个 cell_id 在 pos_arr 中的起始位置
    map<CELL_ID_TYPE, int> start_pos_of_cell;
    for (int i = 0; i < state.window_size;) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[state.pos_arr[i]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num >= state.min_pts) {
            start_pos_of_cell[cell_id] = i;
        }
        i += point_num;
    }
    
    // 根据 h_cluster_id 设置 cluster_id
    memcpy(state.check_h_cluster_id, state.h_cluster_id, state.window_size * sizeof(int));
    // 使用低效的笨办法进行测试
    int i = 0;
    int* pos_arr = state.pos_arr;
    int* cid = state.check_h_cluster_id;
    while (i < state.window_size) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[i];
        int point_num = state.cell_point_num[cell_id];
        if (point_num < state.min_pts) {
            i++;
            continue;
        }
        CELL_ID_TYPE begin_cell_id = get_cell_id(state.h_window, state.min_value, state.cell_count, state.cell_length, i);
        auto begin = start_pos_of_cell.find(begin_cell_id);
        begin++;
        DATA_TYPE_3 point = state.h_window[i];
        for (auto it = begin; it != start_pos_of_cell.end(); it++) {
            // TODO：可以加一层判断，判别是否相邻。若不相邻，则直接跳过
            if (!check_neighbor_cells(state, point, state.h_window[pos_arr[it->second]])) {
                continue;
            }
            if (state.cell_point_num[it->first] >= state.min_pts) { // 只和 dense cell 进行判别
                for (int j = it->second; j < it->second + state.cell_point_num[it->first]; j++) {
                    if (find(i, cid) == find(pos_arr[j], cid)) {
                        break;
                    }
                    // 计算距离
                    DATA_TYPE_3 O = {point.x - state.h_window[pos_arr[j]].x, 
                                     point.y - state.h_window[pos_arr[j]].y,
                                     point.z - state.h_window[pos_arr[j]].z};
                    DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
                    if (d < state.radius * state.radius) {
                        unite(i, pos_arr[j], cid);
                        break;
                    }
                }
            }
        }
        i++;
        // if (i % 1000 == 0) {
        //     printf ("i = %d\n", i);
        // }
    }

    for (int i = 0; i < state.window_size; i++) {
        if (state.h_label[i] == 2) continue;
        find(i, cid);
    }
    printf("cluster_dense_cells_cpu done!\n");
}

void search_cuda(ScanState &state) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;

    log_common_info(state);

    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int)));
    find_neighbors(state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK();

    // * Start sliding
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    while (remaining_data_num >= state.stride_size) {
        cluster_with_cuda(state, timer);
        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void search_naive(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;

    log_common_info(state);

    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas(state);
    printf("[Step] Initialize the first window - build BVH tree...\n");
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int)));
    find_neighbors(state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK();
    printf("[Step] Initialize the first window - get NN...\n");

    // * Start sliding
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    unsigned long cluster_ray_intersections = 0;
    CUDA_CHECK(cudaMemset(state.params.cluster_ray_intersections, 0, sizeof(unsigned)));
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);

        state.params.out = state.params.window + update_pos * state.stride_size;
        memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));
        // out stride
        timer.startTimer(&timer.out_stride_ray);
        state.params.operation = 0; // nn--
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.stride_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.out_stride_ray);

        // 插入 in stride
        timer.startTimer(&timer.in_stride_bvh);
        CUDA_CHECK(cudaMemcpy(state.params.out, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        rebuild_gas(state, update_pos);
        // 重置 in/out stride 部分 nn
        CUDA_CHECK(cudaMemset(state.params.nn + update_pos * state.stride_size, 0, state.stride_size * sizeof(int)));
        timer.stopTimer(&timer.in_stride_bvh);

        timer.startTimer(&timer.in_stride_ray);
        state.params.operation = 1; // nn++
        state.params.stride_left = update_pos * state.stride_size;
        state.params.stride_right = state.params.stride_left + state.stride_size;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.stride_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.in_stride_ray);

        // set labels and find cores
        timer.startTimer(&timer.find_cores);
        find_cores(state.params.label, state.params.nn, state.params.cluster_id, state.window_size, state.min_pts, 0);
        timer.stopTimer(&timer.find_cores);

        // 根据获取到的 core 开始 union，设置 cluster_id
        timer.startTimer(&timer.set_cluster_id);
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.union_cluster_id);
        // * serial union-find
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);

        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;

        unsigned is;
        CUDA_CHECK(cudaMemcpy(&is, state.params.cluster_ray_intersections, sizeof(unsigned), cudaMemcpyDeviceToHost));
        cluster_ray_intersections += is;
        CUDA_CHECK(cudaMemset(state.params.cluster_ray_intersections, 0, sizeof(unsigned)));

        if (!timing) if (!check(state, stride_num, timer)) { exit(1); }

        // printf("[Step] Finish window %d\n", window_left / state.stride_size);
    }
    printf("[Step] Finish sliding the window...\n");
    printf("[Debug] cluster_ray_intersections=%lu\n", cluster_ray_intersections);
}

void search_identify_cores(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;

    log_common_info(state);

    // * 设置 buffer 数组
    state.d_gas_temp_buffer_list   = (CUdeviceptr*) malloc(unit_num * sizeof(CUdeviceptr));
    state.d_gas_output_buffer_list = (CUdeviceptr*) malloc(unit_num * sizeof(CUdeviceptr));
    state.handle_list              = (OptixTraversableHandle*) malloc(unit_num * sizeof(OptixTraversableHandle));

    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas_for_each_stride(state, unit_num);
    
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int)));
    find_neighbors(state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK();

    // * Start sliding
    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        find_neighbors_cores(state, update_pos);
        CUDA_CHECK(cudaEventSynchronize(timer.stop2));
        CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds2, timer.start2, timer.stop2));
        timer.find_cores += timer.milliseconds2;
        CUDA_CHECK(cudaEventDestroy(timer.start2));
        CUDA_CHECK(cudaEventDestroy(timer.stop2));

        timer.startTimer(&timer.whole_bvh);
        rebuild_gas(state); // 不再更新aabb，因为rebuild_gas已经更新好
        timer.stopTimer(&timer.whole_bvh);

        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        // printf("window_id=%d, before, h_label[%d]=%d, h_cluster_id[%d]=%d\n", stride_num + 1, 836, state.h_label[836], 836, state.h_cluster_id[836]);

        timer.startTimer(&timer.set_cluster_id);
        state.params.window_id = stride_num + 1;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        // 从 core 发射光线：1）只与小的 core 进行相交测试，若遇到 core，则向 idx 小的 core 聚类 2）若遇到 border，且 border 的 cid 已经设置过，则直接跳过；否则使用原子操作设置该 border
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.union_cluster_id);
        // * serial union-find
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        // printf("window_id=%d, after, h_label[%d]=%d, h_cluster_id[%d]=%d\n", stride_num + 1, 836, state.h_label[836], 836, state.h_cluster_id[836]);
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);

        swap(state.d_gas_temp_buffer_list[update_pos], state.d_gas_temp_buffer);
        swap(state.d_gas_output_buffer_list[update_pos], state.d_gas_output_buffer);
        state.handle_list[update_pos] = state.params.in_stride_handle;

        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        
        if (!timing) if (!check(state, stride_num, timer)) { exit(1); }

        // printf("[Step] Finish window %d\n", window_left / state.stride_size);
    }
    CUDA_CHECK(cudaStreamDestroy(state.stream));
    printf("[Step] Finish sliding the window...\n");
}

// ! `early_cluster` is useless
void search_with_grid(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;

    log_common_info(state);

    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas(state);
    printf("[Step] Initialize the first window - build BVH tree...\n");
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int)));
    find_neighbors(state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK();
    printf("[Step] Initialize the first window - get NN...\n");
    for (int i = 0; i < state.window_size; i++) {
        int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.h_point_cell_id[i] = cell_id;
    }
    CUDA_CHECK(cudaMemcpy(state.params.point_cell_id, state.h_point_cell_id, state.stride_size * sizeof(int), cudaMemcpyHostToDevice));
    printf("[Step] Initialize the first window - set grid...\n");

    // * Start sliding
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    unsigned long cluster_ray_intersections = 0;
    CUDA_CHECK(cudaMemset(state.params.cluster_ray_intersections, 0, sizeof(unsigned)));
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        
        state.params.out = state.params.window + update_pos * state.stride_size;
        timer.startTimer(&timer.out_stride_bvh);
        rebuild_gas_stride(state, update_pos, state.out_stride_gas_handle);
        timer.stopTimer(&timer.out_stride_bvh);
        timer.startTimer(&timer.out_stride_ray);
        state.params.operation = 0; // nn--
        state.params.out_stride_handle = state.out_stride_gas_handle;
        state.params.stride_left = update_pos * state.stride_size;
        state.params.stride_right = state.params.stride_left + state.stride_size;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.out_stride_ray);

        timer.startTimer(&timer.in_stride_bvh);
        CUDA_CHECK(cudaMemcpy(state.params.out, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(state.params.nn + update_pos * state.stride_size, 0, state.stride_size * sizeof(int)));
        rebuild_gas_stride(state, update_pos, state.in_stride_gas_handle);
        timer.stopTimer(&timer.in_stride_bvh);
        timer.startTimer(&timer.in_stride_ray);
        state.params.operation = 1; // nn++
        state.params.in_stride_handle = state.in_stride_gas_handle;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.in_stride_ray);

        timer.startTimer(&timer.find_cores);
        find_cores(state.params.label, state.params.nn, state.params.cluster_id, state.window_size, state.min_pts, 0);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.find_cores);

        timer.startTimer(&timer.early_cluster);
        for (int i = window_left; i < window_left + state.stride_size; i++) {
            int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i); // get_cell_id 方法需要改造一下
            state.cell_point_num.erase(cell_id); // TODO: state.cell_point_num[cell_id]--，不应该直接删除该item
        }
        for (int i = window_right; i < window_right + state.stride_size; i++) {
            int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
            state.cell_point_num[cell_id]++;
        }

        CUDA_CHECK(cudaMemcpy(state.h_window, state.params.window, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyDeviceToHost));
        unordered_map<int, int> cell_repres;
        int dense_core_num = 0;
        for (int i = 0; i < state.window_size; i++) {
            int cell_id = get_cell_id(state.h_window, state.min_value, state.cell_count, state.cell_length, i);
            if (state.cell_point_num[cell_id] >= state.min_pts) {
                dense_core_num++;
                if (cell_repres.count(cell_id)) { // 未统计每个 grid 中有多少数据，而是直接说明有多少个数
                    state.h_cluster_id[i] = cell_repres[cell_id];
                } else {
                    state.h_cluster_id[i] = i;
                    cell_repres[cell_id] = i;
                }
            } else {
                state.h_cluster_id[i] = i;
            }
        }
        CUDA_CHECK(cudaMemcpy(state.params.cluster_id, state.h_cluster_id, state.window_size * sizeof(int), cudaMemcpyHostToDevice)); // TODO: 之前 find_cores 方法中对这个的设置是没必要的，可以去掉
        timer.stopTimer(&timer.early_cluster);

        timer.startTimer(&timer.whole_bvh);
        rebuild_gas(state);
        timer.stopTimer(&timer.whole_bvh);

        timer.startTimer(&timer.set_cluster_id);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        // 从 core 发射光线：1）只与小的 core 进行相交测试，若遇到 core，则向 idx 小的 core 聚类 2）若遇到 border，且 border 的 cid 已经设置过，则直接跳过；否则使用原子操作设置该 border
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.union_cluster_id);
        // * serial union-find
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);

        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        
        unsigned is;
        CUDA_CHECK(cudaMemcpy(&is, state.params.cluster_ray_intersections, sizeof(unsigned), cudaMemcpyDeviceToHost));
        cluster_ray_intersections += is;
        CUDA_CHECK(cudaMemset(state.params.cluster_ray_intersections, 0, sizeof(unsigned)));

        if (!timing) if (!check(state, stride_num, timer)) { exit(1); }

        // printf("[Step] Finish window %d\n", window_left / state.stride_size);
    }
    printf("[Step] Finish sliding the window...\n");
    printf("[Debug] cluster_ray_intersections=%lu\n", cluster_ray_intersections);
}

void search_hybrid_bvh(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;

    log_common_info(state);

    // * 设置 buffer 数组
    state.d_gas_temp_buffer_list   = (CUdeviceptr*) malloc(unit_num * sizeof(CUdeviceptr));
    state.d_gas_output_buffer_list = (CUdeviceptr*) malloc(unit_num * sizeof(CUdeviceptr));
    state.handle_list              = (OptixTraversableHandle*) malloc(unit_num * sizeof(OptixTraversableHandle));

    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas_for_each_stride(state, unit_num);
    
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int)));
    find_neighbors(state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK(); // May conflict with un-synchronized stream
    
    for (int i = 0; i < state.window_size; i++) {
        int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.cell_points[cell_id].push_back(i);
        state.h_point_cell_id[i] = cell_id;
    }

    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
    for (int i = 0; i < state.window_size; i++) pos_arr[i] = i;
    sort(pos_arr, pos_arr + state.window_size, 
         [&point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });

    // * Start sliding
    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        find_neighbors_cores(state, update_pos);
        CUDA_CHECK(cudaEventSynchronize(timer.stop2));
        CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds2, timer.start2, timer.stop2));
        timer.find_cores += timer.milliseconds2;
        CUDA_CHECK(cudaEventDestroy(timer.start2));
        CUDA_CHECK(cudaEventDestroy(timer.stop2));
        update_grid_without_vector(state, update_pos, window_left, window_right);
        // set_centers_radii_cpu(state, pos_arr);
        set_centers_radii_gpu(state, pos_arr);
        make_gas_by_cell(state, timer);
        timer.stopTimer(&timer.pre_process);

        CUDA_CHECK(cudaEventCreate(&timer.start1));
        CUDA_CHECK(cudaEventCreate(&timer.stop1));
        CUDA_CHECK(cudaEventRecord(timer.start1));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_CHECK(cudaEventRecord(timer.stop1));
        CUDA_CHECK(cudaEventSynchronize(timer.stop1));
        CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds1, timer.start1, timer.stop1));
        CUDA_CHECK(cudaEventDestroy(timer.start1));
        CUDA_CHECK(cudaEventDestroy(timer.stop1));
        timer.set_cluster_id += timer.milliseconds1;

        timer.startTimer(&timer.union_cluster_id);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);

        swap(state.d_gas_temp_buffer_list[update_pos], state.d_gas_temp_buffer);
        swap(state.d_gas_output_buffer_list[update_pos], state.d_gas_output_buffer);
        state.handle_list[update_pos] = state.params.in_stride_handle;

        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        
        if (!timing) if (!check(state, stride_num, timer)) { exit(1); }

        // printf("[Step] Finish window %d\n", stride_num);
    }
    CUDA_CHECK(cudaStreamDestroy(state.stream));
    printf("[Step] Finish sliding the window...\n");
}

void search_async(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;

    log_common_info(state);

    // * 设置 buffer 数组
    state.d_gas_temp_buffer_list   = (CUdeviceptr*) malloc(unit_num * sizeof(CUdeviceptr));
    state.d_gas_output_buffer_list = (CUdeviceptr*) malloc(unit_num * sizeof(CUdeviceptr));
    state.handle_list              = (OptixTraversableHandle*) malloc(unit_num * sizeof(OptixTraversableHandle));

    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas_for_each_stride(state, unit_num);
    
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int)));
    find_neighbors(state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK(); // May conflict with un-synchronized stream
    
    for (int i = 0; i < state.window_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.cell_points[cell_id].push_back(i);
        state.h_point_cell_id[i] = cell_id;
    }

    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
    for (int i = 0; i < state.window_size; i++) pos_arr[i] = i;
    sort(pos_arr, pos_arr + state.window_size, 
         [&point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });

    // * Start sliding
    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        find_neighbors_cores(state, update_pos);
        update_grid_without_vector(state, update_pos, window_left, window_right);
        // set_centers_radii_cpu(state, pos_arr);
        set_centers_radii_gpu(state, pos_arr);
        make_gas_by_cell(state, timer);
        
        CUDA_CHECK(cudaEventSynchronize(timer.stop2));
        CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds2, timer.start2, timer.stop2));
        timer.find_cores += timer.milliseconds2;
        CUDA_CHECK(cudaEventDestroy(timer.start2));
        CUDA_CHECK(cudaEventDestroy(timer.stop2));
        timer.stopTimer(&timer.pre_process);

        CUDA_CHECK(cudaEventCreate(&timer.start1));
        CUDA_CHECK(cudaEventCreate(&timer.stop1));
        CUDA_CHECK(cudaEventRecord(timer.start1));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_CHECK(cudaEventRecord(timer.stop1));
        CUDA_CHECK(cudaEventSynchronize(timer.stop1));
        CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds1, timer.start1, timer.stop1));
        CUDA_CHECK(cudaEventDestroy(timer.start1));
        CUDA_CHECK(cudaEventDestroy(timer.stop1));
        timer.set_cluster_id += timer.milliseconds1;

        timer.startTimer(&timer.union_cluster_id);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);

        swap(state.d_gas_temp_buffer_list[update_pos], state.d_gas_temp_buffer);
        swap(state.d_gas_output_buffer_list[update_pos], state.d_gas_output_buffer);
        state.handle_list[update_pos] = state.params.in_stride_handle;

        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        
        if (!timing) {
            // calc_cluster_num(state.h_cluster_id, state.window_size, state.min_pts);
            if (!check(state, stride_num, timer)) { exit(1); }
        }
        // printf("[Step] Finish window %d\n", stride_num);
    }
    CUDA_CHECK(cudaStreamDestroy(state.stream));
    printf("[Step] Finish sliding the window...\n");
}

void search_grid_cores_like_rtod(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;
    log_common_info(state);
    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas(state);
    for (int i = 0; i < state.window_size; i++) {
        int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.cell_points[cell_id].push_back(i);
        state.h_point_cell_id[i] = cell_id;
    }
    // * Start sliding
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    state.h_nn = (int*) malloc(state.window_size * sizeof(int));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));
        CUDA_CHECK(cudaMemcpy(state.params.window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        timer.startTimer(&timer.update_grid);
        update_grid(state, update_pos, window_left, window_right);
        timer.stopTimer(&timer.update_grid);
        timer.startTimer(&timer.build_bvh);
        set_centers_sparse(state);
        make_gas_by_sparse_points(state, timer);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.build_bvh);
        // printf("Number of centers: %d\n", state.params.center_num);
        CUDA_CHECK(cudaMemset(state.params.nn, 0, state.params.center_num * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        timer.startTimer(&timer.find_cores);
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        CUDA_CHECK(cudaMemcpy(state.h_nn, state.params.nn, state.params.center_num * sizeof(int), cudaMemcpyDeviceToHost));
        timer.stopTimer(&timer.find_cores);
        memset(state.h_label, 0, state.window_size * sizeof(int)); // Set all points to cores
        for (int i = 0; i < state.params.center_num; i++) {
            if (state.h_nn[i] < state.min_pts) { // Mark noises
                state.h_label[state.h_center_idx_in_window[i]] = 2; // Noise
            }
        }
        // Copy label and cluster_id
        CUDA_CHECK(cudaMemcpy(state.params.label, state.h_label, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.params.cluster_id, state.h_cluster_id, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.pre_process);
        
        // Build whole BVH tree for clustering
        rebuild_gas_from_all_points_in_window(state);
        
        CUDA_CHECK(cudaEventCreate(&timer.start1));
        CUDA_CHECK(cudaEventCreate(&timer.stop1));
        CUDA_CHECK(cudaEventRecord(timer.start1));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_CHECK(cudaEventRecord(timer.stop1));
        CUDA_CHECK(cudaEventSynchronize(timer.stop1));
        CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds1, timer.start1, timer.stop1));
        CUDA_CHECK(cudaEventDestroy(timer.start1));
        CUDA_CHECK(cudaEventDestroy(timer.stop1));
        timer.set_cluster_id += timer.milliseconds1;

        timer.startTimer(&timer.union_cluster_id);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);
        
        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        if (!timing) if (!check(state, stride_num, timer)) { exit(1); }
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void search_grid_cores_like_rtod_friendly_gpu_grid_storing(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;
    log_common_info(state);
    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas(state);
    for (int i = 0; i < state.window_size; i++) {
        int cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.cell_points[cell_id].push_back(i);
        state.h_point_cell_id[i] = cell_id;
    }
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
    for (int i = 0; i < state.window_size; i++) pos_arr[i] = i;
    sort(pos_arr, pos_arr + state.window_size, 
         [&point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });
    // * Start sliding
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    state.h_nn = (int*) malloc(state.window_size * sizeof(int));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));
        CUDA_CHECK(cudaMemcpy(state.params.window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        timer.startTimer(&timer.update_grid);
        update_grid_without_vector(state, update_pos, window_left, window_right);
        timer.stopTimer(&timer.update_grid);
        timer.startTimer(&timer.build_bvh);
        set_centers_sparse_without_vector(state);
        make_gas_by_sparse_points(state, timer);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.build_bvh);
        // printf("Number of centers: %d\n", state.params.center_num);
        CUDA_CHECK(cudaMemset(state.params.nn, 0, state.params.center_num * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        timer.startTimer(&timer.find_cores);
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        CUDA_CHECK(cudaMemcpy(state.h_nn, state.params.nn, state.params.center_num * sizeof(int), cudaMemcpyDeviceToHost));
        timer.stopTimer(&timer.find_cores);
        memset(state.h_label, 0, state.window_size * sizeof(int)); // Set all points to cores
        for (int i = 0; i < state.params.center_num; i++) {
            if (state.h_nn[i] < state.min_pts) { // Mark noises
                state.h_label[state.h_center_idx_in_window[i]] = 2; // Noise
            }
        }
        for (int i = 0; i < state.window_size; i++) {
            state.h_cluster_id[i] = i;
        }
        // Copy label and cluster_id
        CUDA_CHECK(cudaMemcpy(state.params.label, state.h_label, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.params.cluster_id, state.h_cluster_id, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.pre_process);
        
        // Build whole BVH tree for clustering
        rebuild_gas_from_all_points_in_window(state);
        
        CUDA_CHECK(cudaEventCreate(&timer.start1));
        CUDA_CHECK(cudaEventCreate(&timer.stop1));
        CUDA_CHECK(cudaEventRecord(timer.start1));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_CHECK(cudaEventRecord(timer.stop1));
        CUDA_CHECK(cudaEventSynchronize(timer.stop1));
        CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds1, timer.start1, timer.stop1));
        CUDA_CHECK(cudaEventDestroy(timer.start1));
        CUDA_CHECK(cudaEventDestroy(timer.stop1));
        timer.set_cluster_id += timer.milliseconds1;

        timer.startTimer(&timer.union_cluster_id);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);
        
        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        if (!timing) if (!check(state, stride_num, timer)) { exit(1); }
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void search_grid_cores_like_rtod_early_cluster_dense_cells(ScanState &state, bool timing) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;
    log_common_info(state);
    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas(state);
    for (int i = 0; i < state.window_size; i++) {
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
        state.cell_point_num[cell_id]++;
        state.cell_points[cell_id].push_back(i);
        state.h_point_cell_id[i] = cell_id;
    }
    int *pos_arr = state.pos_arr;
    CELL_ID_TYPE *point_cell_id = state.h_point_cell_id;
    for (int i = 0; i < state.window_size; i++) pos_arr[i] = i;
    sort(pos_arr, pos_arr + state.window_size, 
         [&point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });
    // * Start sliding
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    state.h_nn = (int*) malloc(state.window_size * sizeof(int));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));
        CUDA_CHECK(cudaMemcpy(state.params.window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        timer.startTimer(&timer.update_grid);
        update_grid_without_vector(state, update_pos, window_left, window_right);
        timer.stopTimer(&timer.update_grid);
        timer.startTimer(&timer.build_bvh);
        set_centers_sparse_without_vector(state);
        make_gas_by_sparse_points(state, timer);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.build_bvh);
        // printf("Number of centers: %d\n", state.params.center_num);
        CUDA_CHECK(cudaMemset(state.params.nn, 0, state.params.center_num * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        timer.startTimer(&timer.find_cores);
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        CUDA_CHECK(cudaMemcpy(state.h_nn, state.params.nn, state.params.center_num * sizeof(int), cudaMemcpyDeviceToHost));
        timer.stopTimer(&timer.find_cores);
        memset(state.h_label, 0, state.window_size * sizeof(int)); // Set all points to cores
        for (int i = 0; i < state.params.center_num; i++) {
            if (state.h_nn[i] < state.min_pts) { // Mark noises
                state.h_label[state.h_center_idx_in_window[i]] = 2; // Noise
            }
        }
        // Copy label and cluster_id
        CUDA_CHECK(cudaMemcpy(state.params.label, state.h_label, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(state.params.cluster_id, state.h_cluster_id, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.pre_process);

        timer.startTimer(&timer.find_neighbors_of_cells);
        find_neighbors_of_cells(state);
        // cluster_dense_cells_cpu(state);
        // 需要提前对每个 dense cell 中的 cluster id 进行聚类
        timer.stopTimer(&timer.find_neighbors_of_cells);
        timer.startTimer(&timer.cluster_dense_cells);
        CUDA_CHECK(cudaMemcpy(state.params.pos_arr, state.pos_arr, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        cluster_dense_cells(state.params.pos_arr, state.params.window,
                            state.params.window_size, state.params.radius2,
                            state.params.cluster_id,
                            state.params.d_neighbor_cells_pos, state.params.d_neighbor_cells_num,
                            state.params.d_neighbor_cells_list, state.params.d_neighbor_cells_capacity,
                            0);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.cluster_dense_cells);
        // CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        // for (int i = 0; i < state.window_size; i++) {
        //     if (state.h_label[i] == 2) continue;
        //     find(i, state.h_cluster_id);
        // }
        // CUDA_CHECK(cudaMemcpy(state.params.cluster_id, state.h_cluster_id, state.window_size * sizeof(int), cudaMemcpyHostToDevice));

        // * Check
        // for (int i = 0; i < state.window_size; i++) {
        //     if (state.check_h_cluster_id[i] != state.h_cluster_id[i]) {
        //         printf("check error, state.h_cluster_id[%d] = %d, state.check_h_cluster_id[%d] = %d\n",
        //                 i, state.h_cluster_id[i], i, state.check_h_cluster_id[i]);
        //         int cell_id = state.h_point_cell_id[i];
        //         int point_num = state.cell_point_num[cell_id];
        //         printf("point_num = %d\n", point_num);
        //         exit(0);
        //     }
        // }
        // printf("cluster_dense_cells == cluster_dense_cells_cpu, correct!\n");

        timer.startTimer(&timer.set_cluster_id);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.union_cluster_id);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state.h_cluster_id);
        }
        timer.stopTimer(&timer.union_cluster_id);
        
        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        if (!timing) if (!check(state, stride_num, timer)) { exit(1); }
        printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void cleanup(ScanState &state) {
    // free host memory
    free(state.h_data);
    CUDA_CHECK(cudaFreeHost(state.h_label));
    CUDA_CHECK(cudaFreeHost(state.h_cluster_id));
    CUDA_CHECK(cudaFreeHost(state.h_window));
    free(state.h_point_cell_id);
    free(state.d_cell_points);
    CUDA_CHECK(cudaFreeHost(state.pos_arr));
    free(state.tmp_pos_arr);
    free(state.new_pos_arr);
    CUDA_CHECK(cudaFreeHost(state.uniq_pos_arr));
    CUDA_CHECK(cudaFreeHost(state.num_points));
#if OPTIMIZATION_LEVEL == 3
    free(state.d_gas_temp_buffer_list); // TODO: free 其中的每一项
    free(state.d_gas_output_buffer_list);
#endif

    // free device memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));

    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(state.params.window));
    CUDA_CHECK(cudaFree(state.params.nn));
    CUDA_CHECK(cudaFree(state.params.label));
    CUDA_CHECK(cudaFree(state.params.cluster_id));
    CUDA_CHECK(cudaFree(state.params.centers));
    CUDA_CHECK(cudaFree(state.params.radii));
    CUDA_CHECK(cudaFree(state.params.point_cell_id));
    CUDA_CHECK(cudaFree(state.params.out_stride));
    CUDA_CHECK(cudaFree(state.params.pos_arr));
    CUDA_CHECK(cudaFree(state.params.point_status));
    CUDA_CHECK(cudaFree(state.params.center_idx_in_window));

    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_pos));
    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_num));
    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_list));
    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_capacity));

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaFree(state.params.ray_primitive_hits));
    CUDA_CHECK(cudaFree(state.params.ray_intersections));
    free(state.h_ray_hits);
    free(state.h_ray_intersections);
#endif

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));

    // Free memory for CUDA implementation
    if (state.check) {
        CUDA_CHECK(cudaFree(state.params.check_nn));
        CUDA_CHECK(cudaFree(state.params.check_label));
        CUDA_CHECK(cudaFree(state.params.check_cluster_id));
        free(state.check_h_nn);
        free(state.check_h_label);
        free(state.check_h_cluster_id);
        free(state.h_nn);
    }
}

int main(int argc, char *argv[]) {
    setbuf(stdout, NULL); // https://blog.csdn.net/moshiyaofei/article/details/107297472
    ScanState state;
    parse_args(state, argc, argv);
    if (state.data_file.find("tao") != string::npos) {
        read_data_from_tao(state.data_file, state);
    } else if (state.data_file.find("geolife") != string::npos) {
        read_data_from_geolife(state.data_file, state);
    } else if (state.data_file.find("RBF") != string::npos) {
        read_data_from_rbf(state.data_file, state);
    } else if (state.data_file.find("EDS") != string::npos) {
        read_data_from_eds(state.data_file, state);
    } else if (state.data_file.find("stock") != string::npos) {
        read_data_from_stk(state.data_file, state);
    } else {
        printf("[Error] Dataset %s does not exist!\n", state.data_file.c_str());
    }

    initialize_optix(state);
    make_module(state);
    make_program_groups(state);
    make_pipeline(state);               // Link pipeline
    make_sbt(state);
    state.check = false;
    initialize_params(state);
    
    // Warmup
#if OPTIMIZATION_LEVEL == 7
    search_grid_cores_like_rtod_early_cluster_dense_cells(state, !state.check);
#elif OPTIMIZATION_LEVEL == 5
    search_grid_cores_like_rtod(state, !state.check);
    // search_grid_cores_like_rtod_friendly_gpu_grid_storing(state, !state.check);
#elif OPTIMIZATION_LEVEL == 4
    search_async(state, !state.check);
#elif OPTIMIZATION_LEVEL == 3
    search_hybrid_bvh(state, !state.check);
#elif OPTIMIZATION_LEVEL == 2
    search_with_grid(state, !state.check);
#elif OPTIMIZATION_LEVEL == 1
    search_identify_cores(state, !state.check);
#elif OPTIMIZATION_LEVEL == 0
    search_naive(state, !state.check);
#elif OPTIMIZATION_LEVEL == 10
    search_cuda(state);
#endif
    printf("[Step] Warmup\n");
    timer.clear();
    state.cell_point_num.clear();
    state.cell_points.clear();
    state.cell_repres.clear();
    // Timing
#if OPTIMIZATION_LEVEL == 7
    search_grid_cores_like_rtod_early_cluster_dense_cells(state, true);
#elif OPTIMIZATION_LEVEL == 5
    search_grid_cores_like_rtod(state, true);
    // search_grid_cores_like_rtod_friendly_gpu_grid_storing(state, true);
#elif OPTIMIZATION_LEVEL == 4
    search_async(state, true);
#elif OPTIMIZATION_LEVEL == 3
    search_hybrid_bvh(state, true);
#elif OPTIMIZATION_LEVEL == 2
    search_with_grid(state, true);
#elif OPTIMIZATION_LEVEL == 1
    search_identify_cores(state, true);
#elif OPTIMIZATION_LEVEL == 0
    search_naive(state, true);
#elif OPTIMIZATION_LEVEL == 10
    search_cuda(state);
#endif

    timer.showTime((state.data_num - state.window_size) / state.stride_size);
    cleanup(state);
    return 0;
}