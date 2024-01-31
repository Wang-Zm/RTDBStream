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

void printUsageAndExit(const char* argv0) {
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for data input\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --n <int>                   Set data num; defaults to 1e8\n";
    std::cerr << "         --primitive <int>           Set primitive type, 0 for cube, 1 for triangle with anyhit; defaults to 0\n";
    std::cerr << "         --nc                        No Comparison\n";
    exit(1);
}

void parse_args(ScanState &state, int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            printUsageAndExit(argv[0]);
        } else if (arg == "--file" || arg == "-f") {
            if (i < argc - 1) {
                state.data_file = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "-n") {
            if (i < argc - 1) {
                state.data_num = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--window" || arg == "-W") {
            if (i < argc - 1) {
                state.window_size = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--stride" || arg == "-S") {
            if (i < argc - 1) {
                state.stride_size = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--minPts" || arg == "-K") {
            if (i < argc - 1) {
                state.min_pts = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--radius" || arg == "-R") {
            if (i < argc - 1) {
                state.radius = stod(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }
}

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
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));
    state.h_cluster_id = (int*) malloc(state.window_size * sizeof(int));
    state.h_label = (int*) malloc(state.window_size * sizeof(int));

    CUDA_CHECK(cudaMalloc(&state.params.centers, state.window_size * sizeof(DATA_TYPE_3)));
    CUDA_CHECK(cudaMalloc(&state.params.radii, state.window_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&state.params.point_cell_id, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.center_idx_in_window, state.window_size * sizeof(int)));
    state.h_point_cell_id = (int*) malloc(state.window_size * sizeof(int));

    CUDA_CHECK(cudaMalloc(&state.params.cell_points, state.window_size * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&state.params.cell_point_num, state.window_size * sizeof(int)));
    state.d_cell_points = (int**) malloc(state.window_size * sizeof(int*));
    for (int i = 0; i < state.window_size; i++) state.d_cell_points[i] = nullptr;

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

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] initialize_params: " << 1.0 * used / (1 << 20) << std::endl;
}

void log_common_info(ScanState &state) {
    std::cout << "Input file: " << state.data_file << std::endl;
    std::cout << "Data num: " << state.data_num << std::endl;
    std::cout << "Window size: " << state.window_size << ", Stride size:" << state.stride_size << ", K:" << state.min_pts << std::endl;
    std::cout << "Radius: " << state.radius << ", Radius2: " << state.params.radius2 << std::endl;
    std::cout << "Cell length: " << state.cell_length << std::endl;
    for (int i = 0; i < 3; i++) std::cout << "Cell count: " << state.cell_count[i] << ", ";
    std::cout << std::endl;
}

inline int get_cell_id(DATA_TYPE_3* data, vector<DATA_TYPE>& min_value, vector<int>& cell_count, DATA_TYPE cell_length, int i) {
    int id = 0;
// #if DIMENSION == 1
//     id = (vertices[i].x - state.min_value[0]) / state.cell_length;
// #elif DIMENSION == 3
    int dim_id_x = (data[i].x - min_value[0]) / cell_length;
    int dim_id_y = (data[i].y - min_value[1]) / cell_length;
    int dim_id_z = (data[i].z - min_value[2]) / cell_length;
    id = dim_id_x * cell_count[1] * cell_count[2] + dim_id_y * cell_count[2] + dim_id_z;
// #endif
    return id;
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
// #ifdef OPTIMIZATION_BVH
        state.h_point_cell_id[pos_start + i] = cell_id;
// #endif
    }
// #ifdef OPTIMIZATION_BVH
//     CUDA_CHECK(cudaMemcpy(state.params.point_cell_id + update_pos * state.stride_size, 
//                             state.h_point_cell_id + update_pos * state.stride_size,
//                             state.stride_size * sizeof(int),
//                             cudaMemcpyHostToDevice));
// #endif
}

void get_centers_radii_device(ScanState &state) {
    CUDA_CHECK(cudaMemcpy(state.h_window, state.params.window, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyDeviceToHost));
    state.h_centers.clear();
    state.h_radii.clear();
    state.h_center_idx_in_window.clear();
    state.h_cell_point_num.clear();
    unordered_map<int, int> &cell_repres = state.cell_repres;
    cell_repres.clear();
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

            int cell_sphere_num = state.h_centers.size() - 1; // 和 primIdx 保持一致
            CUDA_CHECK(cudaMalloc(&state.d_cell_points[cell_sphere_num], state.cell_point_num[cell_id] * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(state.d_cell_points[cell_sphere_num], 
                                  state.cell_points[cell_id].data(),
                                  state.cell_point_num[cell_id] * sizeof(int),
                                  cudaMemcpyHostToDevice));
            // 记录点数多少
            state.h_cell_point_num.push_back(state.cell_point_num[cell_id]);
        } else {
            state.h_centers.push_back(state.h_window[i]);
            state.h_radii.push_back(state.radius);
            state.h_center_idx_in_window.push_back(i);

            state.h_cell_point_num.push_back(1);
        }
    }
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
                          state.h_cell_point_num.size() * sizeof(int*), 
                          cudaMemcpyHostToDevice));
    state.params.center_num = state.h_centers.size();
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

void search(ScanState &state) {
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
        state.cell_points[cell_id].push_back(i);
        state.h_point_cell_id[i] = cell_id;
    }
    CUDA_CHECK(cudaMemcpy(state.params.point_cell_id, state.h_point_cell_id, state.stride_size * sizeof(int), cudaMemcpyHostToDevice));
    printf("[Step] Initialize the first window - set grid...\n");

    // * Start sliding
    state.h_window = (DATA_TYPE_3*) malloc(state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
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
        find_cores(state.params.label, state.params.nn, state.params.cluster_id, state.window_size, state.min_pts);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.find_cores);

#ifdef OPTIMIZATION_GRID
        timer.startTimer(&timer.update_grid);
        update_grid(state, update_pos, window_left, window_right);
        timer.stopTimer(&timer.update_grid);
        
        // timer.startTimer(&timer.early_cluster);
        // early_cluster(state);
        // timer.stopTimer(&timer.early_cluster);
#endif

#ifndef OPTIMIZATION_BVH
        timer.startTimer(&timer.whole_bvh);
        rebuild_gas(state);
        timer.stopTimer(&timer.whole_bvh);
        state.params.handle = state.gas_handle;
#endif

#ifdef OPTIMIZATION_BVH
        timer.startTimer(&timer.hybrid_sphere);
        get_centers_radii_device(state);
        timer.stopTimer(&timer.hybrid_sphere);

        timer.startTimer(&timer.build_hybrid_bvh);
        make_gas_by_cell(state);
        state.params.handle = state.gas_handle;
        timer.stopTimer(&timer.build_hybrid_bvh);
#endif

        timer.startTimer(&timer.early_cluster);
        early_cluster(state);
        timer.stopTimer(&timer.early_cluster);

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

        // * 释放空间
        timer.startTimer(&timer.free_cell_points);
        for (size_t i = 0; i < state.h_centers.size(); i++) {
            if (state.d_cell_points[i] != nullptr) {
                CUDA_CHECK(cudaFree(state.d_cell_points[i]));
                state.d_cell_points[i] = nullptr;
            }
        }
        timer.stopTimer(&timer.free_cell_points);

        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        
        if (!check(state, window_left / state.stride_size, timer)) { exit(1); }

        printf("[Step] Finish window %d\n", window_left / state.stride_size);
    }
    printf("[Step] Finish sliding the window...\n");
}

// void result_d2h(ScanState &state) { // https://www.cnblogs.com/ybqjymy/p/17462877.html
//     timer.startTimer(&timer.copy_results_d2h);
//     for (int i = 0; i < state.query_num; i++) {
//         CUDA_CHECK(cudaMemcpy(state.h_dist[i], state.h_dist_temp[i], state.data_num * sizeof(DIST_TYPE), cudaMemcpyDeviceToHost));
//     }
//     long total_neighbor_num = 0;
//     for (int i = 0; i < state.query_num; i++) {
//         int neighbor_num = 0;
//         for (int j = 0; j < state.data_num; j++) {
//             // TODO: 只会计算到query R内的point的距离，超出的不计算，一开始dist仍然是0；可能之后的一些维度的和<R，会导致求解的邻居多余实际邻居数
//             if (state.h_dist[i][j] > 0 && state.h_dist[i][j] < state.params.radius2) {
//                 neighbor_num++;
//                 state.queries_neighbors[i].push_back(j);
//                 state.queries_neighbor_dist[i].push_back(state.h_dist[i][j]);
//             }
//         }
//         state.queries_neighbor_num[i] = neighbor_num;
//         total_neighbor_num += neighbor_num;
//     }
//     timer.stopTimer(&timer.copy_results_d2h);

//     std::cout << "Total number of neighbors:     " << total_neighbor_num << std::endl;
//     std::cout << "Ratio of returned data points: " << 1.0 * total_neighbor_num / (state.query_num * state.data_num) * 100 << "%" << std::endl;
//     // for (int i = 0; i < state.query_num; i++) {
//     //     std::cout << "Query[" << i << "]: " << state.queries_neighbor_num[i] << std::endl;
//     // }

// #if DEBUG_INFO == 1
//     CUDA_CHECK(cudaMemcpy(state.h_ray_intersections, state.params.ray_intersections, state.query_num * sizeof(unsigned), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(state.h_ray_hits, state.params.ray_primitive_hits, state.query_num * sizeof(unsigned), cudaMemcpyDeviceToHost));
//     long total_hits = 0, total_intersection_tests = 0;
//     for (int i = 0; i < state.query_num; i++) {
//         total_hits += state.h_ray_hits[i];
//         total_intersection_tests += state.h_ray_intersections[i];
//     }
//     std::cout << "Total hits:               " << total_hits << std::endl;
//     std::cout << "Total intersection tests: " << total_intersection_tests << std::endl;

//     long effective_write_num = total_neighbor_num * ((state.dim + 3 - 1) / 3);
//     long ineffective_write_num = total_hits - effective_write_num;
//     std::cout << "Effective write count:   " << effective_write_num << std::endl;
//     std::cout << "Ineffective write count: " << ineffective_write_num << std::endl;
//     std::cout << fixed << setprecision(0) << "Effective write count per query:   " << 1.0 * effective_write_num / state.query_num << std::endl;
//     std::cout << fixed << setprecision(0) << "Ineffective write count per query: " << 1.0 * ineffective_write_num / state.query_num << std::endl;
// #endif
// }

void cleanup(ScanState &state) {
    // free host memory
    free(state.h_data);
    free(state.h_label);
    free(state.h_cluster_id);
    free(state.h_window);
    free(state.h_point_cell_id);
    free(state.d_cell_points);

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

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaFree(state.params.ray_primitive_hits));
    CUDA_CHECK(cudaFree(state.params.ray_intersections));
    free(state.h_ray_hits);
    free(state.h_ray_intersections);
#endif

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));
}

int main(int argc, char *argv[]) {
    setbuf(stdout,NULL); // https://blog.csdn.net/moshiyaofei/article/details/107297472
    ScanState state;
    parse_args(state, argc, argv);
    read_data_from_tao(state.data_file, state);

    initialize_optix(state);
    make_module(state);
    make_program_groups(state);
    make_pipeline(state);               // Link pipeline
    make_sbt(state);
    initialize_params(state);
    search(state);
    // result_d2h(state);
    timer.showTime((state.data_num - state.window_size) / state.stride_size);
    cleanup(state);
    return 0;
}
