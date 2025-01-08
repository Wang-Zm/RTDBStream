#include <optix.h>
// #include <optix_function_table_definition.h>
// #include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>
// #include <sutil/Camera.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <cstring>
#include <unistd.h>
#include <map>
#include <queue>
#include <omp.h>
#include <cmath>
// #include <thread>

#include "state.h"
#include "timer.h"
#include "func.h"

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
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    state.h_nn = (int*) malloc(state.window_size * sizeof(int));

    CUDA_CHECK(cudaMalloc(&state.params.centers, state.window_size * sizeof(DATA_TYPE_3)));
    CUDA_CHECK(cudaMalloc(&state.params.radii, state.window_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&state.params.point_cell_id, state.window_size * sizeof(CELL_ID_TYPE)));
    CUDA_CHECK(cudaMalloc(&state.params.center_idx_in_window, state.window_size * sizeof(int)));
    state.h_point_cell_id = (CELL_ID_TYPE*) malloc(state.window_size * sizeof(CELL_ID_TYPE));
    CUDA_CHECK(cudaMalloc(&state.params.offsets, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.num_offsets, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.num_points_in_dense_cells, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.num_dense_cells, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.sparse_offset, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.dense_offset, sizeof(int)));

    CUDA_CHECK(cudaMalloc(&state.params.cell_points, state.window_size * sizeof(int*)));
    CUDA_CHECK(cudaMalloc(&state.params.cell_point_num, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.points_in_dense_cells, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&state.d_cell_points, state.window_size * sizeof(int*)));
    for (int i = 0; i < state.window_size; i++) state.d_cell_points[i] = nullptr;
    state.points_in_dense_cells = (int*) malloc(state.window_size * sizeof(int));
    CUDA_CHECK(cudaMalloc(&state.params.pos_arr, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.point_status, state.window_size * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&state.params.new_pos_arr, state.stride_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.tmp_pos_arr, state.window_size * sizeof(int)));
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
    CUDA_CHECK(cudaMalloc(&state.params.ray_intersections, state.window_size * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&state.params.ray_primitive_hits, state.window_size * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&state.params.ray_intersections_cluster, state.window_size * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&state.params.ray_primitive_hits_cluster, state.window_size * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&state.params.num_dist_calculations, sizeof(unsigned)));
    state.h_ray_intersections = (unsigned *) malloc(state.window_size * sizeof(unsigned));
    state.h_ray_hits = (unsigned *) malloc(state.window_size * sizeof(unsigned));
#endif
    CUDA_CHECK(cudaMalloc(&state.params.cluster_ray_intersections, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(state.params.cluster_ray_intersections, 0, sizeof(unsigned)));

    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_pos, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_num, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_list, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.d_neighbor_cells_capacity, state.window_size * sizeof(int)));

    state.neighbor_cells_pos = (int*) malloc(state.window_size * sizeof(int));
    state.neighbor_cells_num = (int*) malloc(state.window_size * sizeof(int));

    CUDA_CHECK(cudaMallocHost(&state.h_point_status, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.point_status, state.window_size * sizeof(int)));

    CUDA_CHECK(cudaMallocHost(&state.h_centers_p, state.window_size * sizeof(DATA_TYPE_3)));
    CUDA_CHECK(cudaMallocHost(&state.h_center_idx_in_window_p, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&state.h_cell_point_num_p, state.window_size * sizeof(int)));
    state.h_big_sphere = (int*) malloc(state.window_size * sizeof(int));

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

void print_rt_info(ScanState &state) {
    CUDA_CHECK(cudaMemcpy(state.h_ray_intersections, state.params.ray_intersections, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.h_ray_hits, state.params.ray_primitive_hits, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
    long total_intersections = 0, total_hits = 0;
    for (int i = 0; i < state.window_size; i++) {
        total_intersections += state.h_ray_intersections[i];
        total_hits += state.h_ray_hits[i];
    }
    state.intersections_all_window += total_intersections;
    state.hits_all_window += total_hits;
    // long avg_intersections = total_intersections / state.window_size;
    // long avg_hits = total_hits / state.window_size;
    // printf("total_intersections: %ld\n", total_intersections);
    // printf("total_hits: %ld\n", total_hits);
    // printf("avg_intersections: %ld\n", avg_intersections);
    // printf("avg_hits: %ld\n", avg_hits);
    CUDA_CHECK(cudaMemcpy(state.h_ray_intersections, state.params.ray_intersections_cluster, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.h_ray_hits, state.params.ray_primitive_hits_cluster, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
    total_intersections = 0, total_hits = 0;
    for (int i = 0; i < state.window_size; i++) {
        total_intersections += state.h_ray_intersections[i];
        total_hits += state.h_ray_hits[i];
    }
    state.intersections_all_window_cluster += total_intersections;
    state.hits_all_window_cluster += total_hits;
    
    state.num_spheres_all_window += state.params.center_num;
    state.num_points_in_dense_cells += state.params.dense_num;
    
    unsigned num_dist_calculations;
    CUDA_CHECK(cudaMemcpy(&num_dist_calculations, state.params.num_dist_calculations, sizeof(unsigned), cudaMemcpyDeviceToHost));
    state.num_dist_calculations_all_window += num_dist_calculations;
}

void print_overall_rt_info(ScanState &state, int num_strides) {
    long avg_intersections = state.intersections_all_window / num_strides;
    long avg_hits = state.hits_all_window / num_strides;
    double avg_intersections_per_ray = 1.0 * state.intersections_all_window / num_strides / state.num_rays_in_window;
    double avg_hits_per_ray = 1.0 * state.hits_all_window / num_strides / state.num_rays_in_window;
    
    long avg_intersections_cluster = state.intersections_all_window_cluster / num_strides;
    long avg_hits_cluster = state.hits_all_window_cluster / num_strides;
    double avg_intersections_per_ray_cluster = 1.0 * state.intersections_all_window_cluster / num_strides / state.num_rays_in_window;
    double avg_hits_per_ray_cluster = 1.0 * state.hits_all_window_cluster / num_strides / state.num_rays_in_window;
    
    printf("avg intersections for each window: %ld\n", avg_intersections);
    printf("avg hits for each window: %ld\n", avg_hits);
    printf("avg intersections for each window per ray: %lf\n", avg_intersections_per_ray);
    printf("avg hits for each window per ray: %lf\n", avg_hits_per_ray);
    
    printf("avg intersections for each window cluster: %ld\n", avg_intersections_cluster);
    printf("avg hits for each window cluster: %ld\n", avg_hits_cluster);
    printf("avg intersections for each window per ray cluster: %lf\n", avg_intersections_per_ray_cluster);
    printf("avg hits for each window per ray cluster: %lf\n", avg_hits_per_ray_cluster);
    
    double avg_num_spheres = 1.0 * state.num_spheres_all_window / num_strides;
    printf("avg num spheres for each window: %lf\n", avg_num_spheres);
    
    double avg_num_points_in_dense_cells = 1.0 * state.num_points_in_dense_cells / num_strides;
    printf("avg num points in dense cells for each window: %lf\n", avg_num_points_in_dense_cells);

    double avg_num_dist_calculations = 1.0 * state.num_dist_calculations_all_window / num_strides;
    printf("avg num dist calculations for each window: %lf\n", avg_num_dist_calculations);
}

inline DATA_TYPE_3 compute_cell_center(ScanState& state, DATA_TYPE_3& point) {
    int dim_id_x = (point.x - state.min_value[0]) / state.cell_length;
    int dim_id_y = (point.y - state.min_value[1]) / state.cell_length;
    int dim_id_z = (point.z - state.min_value[2]) / state.cell_length;
    DATA_TYPE_3 center = { state.min_value[0] + (dim_id_x + 0.5f) * state.cell_length, 
                           state.min_value[1] + (dim_id_y + 0.5f) * state.cell_length, 
                           state.min_value[2] + (dim_id_z + 0.5f) * state.cell_length };
    return center;
}

void find_neighbors_cores(ScanState &state, int update_pos, const cudaStream_t &stream) {
    state.params.out = state.params.window + update_pos * state.stride_size;
    state.params.out_stride_handle = state.handle_list[update_pos];
    state.params.stride_left = update_pos * state.stride_size;
    state.params.stride_right = state.params.stride_left + state.stride_size;
    memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));

    CUDA_CHECK(cudaMemcpy(state.params.out_stride, state.params.out, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.out, state.h_window + update_pos * state.stride_size, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.params.nn + update_pos * state.stride_size, 0, state.stride_size * sizeof(int)));
    rebuild_gas_stride(state, update_pos, 0);
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(state.pipeline, stream, state.d_params, sizeof(Params), &state.sbt, state.window_size, 2, 1));
    
    find_cores(state.params.label, state.params.nn, state.params.cluster_id, state.window_size, state.min_pts, stream);
}

void find_neighbors_cores_async(ScanState &state, int update_pos, const cudaStream_t &stream) {
    state.params.out = state.params.window + update_pos * state.stride_size;
    state.params.out_stride_handle = state.handle_list[update_pos];
    state.params.stride_left = update_pos * state.stride_size;
    state.params.stride_right = state.params.stride_left + state.stride_size;
    memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));

    CUDA_CHECK(cudaEventCreate(&timer.start2));
    CUDA_CHECK(cudaEventCreate(&timer.stop2));
    CUDA_CHECK(cudaEventRecord(timer.start2, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.params.out_stride, state.params.out, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(state.params.out, state.h_window + update_pos * state.stride_size, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(state.params.nn + update_pos * state.stride_size, 0, state.stride_size * sizeof(int), stream));
    rebuild_gas_stride(state, update_pos, 0);
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(state.pipeline, stream, state.d_params, sizeof(Params), &state.sbt, state.window_size, 2, 1));
    
    find_cores(state.params.label, state.params.nn, state.params.cluster_id, state.window_size, state.min_pts, stream);
    CUDA_CHECK(cudaEventRecord(timer.stop2, stream));
    // CUDA_CHECK(cudaEventSynchronize(timer.stop2));
    // CUDA_CHECK(cudaEventElapsedTime(&timer.milliseconds2, timer.start2, timer.stop2));
    // timer.find_cores += timer.milliseconds2;
    // CUDA_CHECK(cudaEventDestroy(timer.start2));
    // CUDA_CHECK(cudaEventDestroy(timer.stop2));
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
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos_arr[j]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num >= state.min_pts) {
            int i = pos_arr[j];
            int dim_id_x = (state.h_window[i].x - state.min_value[0]) / state.cell_length;
            int dim_id_y = (state.h_window[i].y - state.min_value[1]) / state.cell_length;
            int dim_id_z = (state.h_window[i].z - state.min_value[2]) / state.cell_length;
            DATA_TYPE_3 center = { state.min_value[0] + (dim_id_x + 0.5f) * state.cell_length, 
                                   state.min_value[1] + (dim_id_y + 0.5f) * state.cell_length, 
                                   state.min_value[2] + (dim_id_z + 0.5f) * state.cell_length };
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
                if (pos < id)
                    id = pos;
            }
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
    // printf("num_dense_cells: %d\n", num_dense_cells);
    // printf("num_sparse_points: %d\n", num_sparse_points);
}

void set_centers_sparse_without_vector(ScanState &state) {
    // Put points in sparse cells into state.params.centers, set centers from pos_arr
    state.h_centers.clear();
    state.h_center_idx_in_window.clear();
    int *pos_arr = state.pos_arr;
    int j = 0;
    int num_dense_cells = 0, num_sparse_points = 0;
    while (j < state.window_size) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos_arr[j]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num < state.min_pts) {
            int pos_arr_start = j;
            for (int k = 0; k < point_num; k++) {
                state.h_centers.push_back(state.h_window[pos_arr[j++]]);
            }
            state.h_center_idx_in_window.insert(state.h_center_idx_in_window.end(), pos_arr + pos_arr_start, pos_arr + j);
            num_sparse_points += point_num;
        } else {
            num_dense_cells++;
            j += point_num;
        }
    }
    CUDA_CHECK(cudaMemcpy(state.params.centers, state.h_centers.data(), state.h_centers.size() * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.center_idx_in_window, state.h_center_idx_in_window.data(), state.h_center_idx_in_window.size() * sizeof(int), cudaMemcpyHostToDevice));
    state.params.center_num = state.h_centers.size();
    timer.stopTimer(&timer.early_cluster);
    // printf("num_dense_cells: %d\n", num_dense_cells);
    // printf("num_sparse_points: %d\n", num_sparse_points);
}

void set_centers_sparse_gpu(ScanState &state) {
    // 1.计算 offsets
    timer.startTimer(&timer.compute_offsets);
    CUDA_CHECK(cudaMemset(state.params.num_offsets, 0, sizeof(int)));
    compute_offsets_of_cells(state.window_size, state.params.pos_arr, state.params.point_cell_id, state.params.offsets, state.params.num_offsets);
    int num_offsets;
    CUDA_CHECK(cudaMemcpy(&num_offsets, state.params.num_offsets, sizeof(int), cudaMemcpyDeviceToHost));
    thrust_sort(state.params.offsets, num_offsets);
    timer.stopTimer(&timer.compute_offsets);

    // 2.设置 centers, cell_points
    timer.startTimer(&timer.set_centers_radii);
    int num_cells = num_offsets - 1;
    CUDA_CHECK(cudaMemset(state.params.sparse_offset, 0, sizeof(int)));
    set_spheres_info_from_sparse_points(num_cells, state.params.min_pts, state.params.pos_arr, state.params.window, 
                                        state.params.sparse_offset, state.params.offsets, state.params.centers,
                                        state.params.cell_points);
    CUDA_CHECK(cudaMemcpy(&state.params.sparse_num, state.params.sparse_offset, sizeof(int), cudaMemcpyDeviceToHost));
    state.params.center_num = state.params.sparse_num;
    timer.stopTimer(&timer.set_centers_radii);
    // printf("sparse_num = %d\n", state.params.sparse_num);
}

void set_hybrid_aabb(ScanState &state) {
    timer.startTimer(&timer.early_cluster);
    int *pos_arr = state.pos_arr;
    int j = 0;
    int num_sparse_centers = 0, num_dense_centers = 0;
    
    timer.startTimer(&timer.set_sparse_spheres);
    while (j < state.window_size) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos_arr[j]];
        int point_num = state.cell_point_num[cell_id];
        if (point_num < state.min_pts) {
            for (int k = 0; k < point_num; k++) {
                state.h_cluster_id[pos_arr[j]] = pos_arr[j];
                state.h_centers_p[num_sparse_centers] = state.h_window[pos_arr[j]];
                state.h_cell_point_num_p[num_sparse_centers] = 1;
                state.d_cell_points[num_sparse_centers] = state.params.pos_arr + j; 
                num_sparse_centers++;
                j++;
            }
        } else {
            state.h_big_sphere[num_dense_centers++] = j;
            int id = pos_arr[j];
            for (int k = 0; k < point_num; k++) {
                state.h_cluster_id[pos_arr[j]] = id;
                j++;
            }
        }
    }
    state.params.sparse_num = num_sparse_centers;
    timer.stopTimer(&timer.set_sparse_spheres);

    timer.startTimer(&timer.set_dense_spheres);
    int idx = state.params.sparse_num;
    for (int i = 0; i < num_dense_centers; i++) {
        int pos_idx = state.h_big_sphere[i];
        int pos = pos_arr[pos_idx];
        CELL_ID_TYPE cell_id = state.h_point_cell_id[pos];
        int point_num = state.cell_point_num[cell_id];
#ifdef COMPUTE_CELL_CENTER
        state.h_centers_p[num_sparse_centers] = state.cell_centers[cell_id];
#else
        DATA_TYPE_3& point = state.h_window[pos];
        int dim_id_x = (point.x - state.min_value[0]) / state.cell_length;
        int dim_id_y = (point.y - state.min_value[1]) / state.cell_length;
        int dim_id_z = (point.z - state.min_value[2]) / state.cell_length;
        DATA_TYPE_3 center = { state.min_value[0] + (dim_id_x + 0.5f) * state.cell_length, 
                               state.min_value[1] + (dim_id_y + 0.5f) * state.cell_length, 
                               state.min_value[2] + (dim_id_z + 0.5f) * state.cell_length };
        state.h_centers_p[num_sparse_centers] = center;
#endif
        state.h_cell_point_num_p[num_sparse_centers] = point_num;
        num_sparse_centers++;
        state.d_cell_points[idx] = state.params.pos_arr + pos_idx; // 索引
        idx++;
    }
    state.params.center_num = num_sparse_centers;
    state.params.dense_num = num_dense_centers;
    timer.stopTimer(&timer.set_dense_spheres);
    
    CUDA_CHECK(cudaMemcpy(state.params.pos_arr, 
                          state.pos_arr, 
                          state.window_size * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.centers, 
                          state.h_centers_p, 
                          state.params.center_num * sizeof(DATA_TYPE_3), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cell_point_num, 
                          state.h_cell_point_num_p, 
                          state.params.center_num * sizeof(int), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cell_points, 
                          state.d_cell_points, 
                          state.params.center_num * sizeof(int*), 
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.cluster_id, 
                          state.h_cluster_id, 
                          state.window_size * sizeof(int), 
                          cudaMemcpyHostToDevice));
    timer.stopTimer(&timer.early_cluster);
    // printf("state.params.sparse_num = %d\n", state.params.sparse_num);
    // printf("state.params.dense_num = %d\n", state.params.dense_num);
    // printf("num_centers: %d\n", state.params.center_num);
}

void set_hybrid_aabb_gpu(ScanState &state) {
    // 1.计算 offsets
    timer.startTimer(&timer.compute_offsets);
    CUDA_CHECK(cudaMemset(state.params.num_offsets, 0, sizeof(int)));
    compute_offsets_of_cells(state.window_size, state.params.pos_arr, state.params.point_cell_id, state.params.offsets, state.params.num_offsets);
    int num_offsets;
    CUDA_CHECK(cudaMemcpy(&num_offsets, state.params.num_offsets, sizeof(int), cudaMemcpyDeviceToHost));
    thrust_sort(state.params.offsets, num_offsets);
    timer.stopTimer(&timer.compute_offsets);

    // 2.统计有多少 sparse point
    timer.startTimer(&timer.count_sparse_points);
    int num_cells = num_offsets - 1;
    CUDA_CHECK(cudaMemset(state.params.num_points_in_dense_cells, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(state.params.num_dense_cells, 0, sizeof(int)));
    count_points_in_dense_cells(state.params.offsets, num_cells, state.min_pts, state.params.num_points_in_dense_cells, state.params.num_dense_cells);
    int num_points_in_dense_cells;
    int num_dense_cells;
    CUDA_CHECK(cudaMemcpy(&num_points_in_dense_cells, state.params.num_points_in_dense_cells, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&num_dense_cells, state.params.num_dense_cells, sizeof(int), cudaMemcpyDeviceToHost));
    int num_points_in_sparse_cells = state.window_size - num_points_in_dense_cells;
    state.params.sparse_num = num_points_in_sparse_cells;
    state.params.dense_num = num_points_in_dense_cells;
    state.params.center_num = num_dense_cells + num_points_in_sparse_cells; // dense cell 的个数 + sparse point 的个数
    // printf("sparse_num = %d\n", num_points_in_sparse_cells);
    // printf("dense_num = %d\n", num_points_in_dense_cells);
    // printf("center_num = %d\n", state.params.center_num);
    // printf("num_cells = %d\n", num_cells);
    timer.stopTimer(&timer.count_sparse_points);

    // 3.设置 centers, center_idx_in_window, cluster_id
    timer.startTimer(&timer.set_centers_radii);
    CUDA_CHECK(cudaMemset(state.params.sparse_offset, 0, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(state.params.dense_offset, &num_points_in_sparse_cells, sizeof(int), cudaMemcpyHostToDevice));
    set_hybrid_spheres_info(num_cells, state.min_pts, state.params.pos_arr, state.params.window,
                            state.params.sparse_offset, state.params.dense_offset, state.params.offsets,
                            state.params.center_idx_in_window, state.params.centers, state.params.cluster_id,
                            state.params.cell_points, state.params.cell_point_num,
                            state.params.min_value, state.params.cell_length);
    // CUDA_SYNC_CHECK();
    timer.stopTimer(&timer.set_centers_radii);
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
        CELL_ID_TYPE cell_id = get_cell_id(state.h_window, state.min_value, state.cell_count, state.cell_length, i);
        if (state.cell_point_num[cell_id] >= state.min_pts) {
            if (cell_repres.count(cell_id) > 0) continue;
            int dim_id_x = (state.h_window[i].x - state.min_value[0]) / state.cell_length;
            int dim_id_y = (state.h_window[i].y - state.min_value[1]) / state.cell_length;
            int dim_id_z = (state.h_window[i].z - state.min_value[2]) / state.cell_length;
            DATA_TYPE_3 center = { state.min_value[0] + (dim_id_x + 0.5f) * state.cell_length, 
                                   state.min_value[1] + (dim_id_y + 0.5f) * state.cell_length, 
                                   state.min_value[2] + (dim_id_z + 0.5f) * state.cell_length };
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
        CELL_ID_TYPE cell_id = state.h_point_cell_id[i];
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
    while (j < state.window_size) { // TODO：似乎也能够使用多线程机制
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

void prepare_for_points_in_dense_cells(int* pos, int* num, CELL_ID_TYPE* h_point_cell_id, 
                                       unordered_map<CELL_ID_TYPE, pair<int, int>>& pos_and_num, 
                                       int block_size, int j, int window_size) {
    int up_bound = (j + 1) * block_size < window_size ? (j + 1) * block_size : window_size;
    for (int k = j * block_size; k < up_bound; k++) {
        CELL_ID_TYPE cell_id = h_point_cell_id[k];
        if (pos_and_num.count(cell_id)) {
            pos[k] = pos_and_num[cell_id].first;
            num[k] = pos_and_num[cell_id].second;
        } else {
            pos[k] = -1;
            num[k] = -1;
        }
    }
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
    unordered_map<CELL_ID_TYPE, pair<int, int>> neighbor_cells_pos_and_num;
      
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
    // #pragma omp parallel for // 会影响其他部分的性能
    for (int j = 0; j < state.window_size; j++) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[j];
        if (neighbor_cells_pos_and_num.count(cell_id)) {
            neighbor_cells_pos[j] = neighbor_cells_pos_and_num[cell_id].first;
            neighbor_cells_num[j] = neighbor_cells_pos_and_num[cell_id].second;
        } else {
            neighbor_cells_pos[j] = -1;
            neighbor_cells_num[j] = -1;
        }
    }

    // 试试使用 CPU 多线程的情况
    // thread threads[THREAD_NUM];
    // int block = (state.window_size + THREAD_NUM - 1) / THREAD_NUM;
    // for (int i = 0; i < THREAD_NUM; i++) {
    //     threads[i] = thread(prepare_for_points_in_dense_cells, 
    //         neighbor_cells_pos, neighbor_cells_num, state.h_point_cell_id, 
    //         std::ref(neighbor_cells_pos_and_num), block, i, state.window_size);
    // }
    // for (int i = 0; i < THREAD_NUM; i++) {
    //     threads[i].join();
    // }
    
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_pos, neighbor_cells_pos, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_num, neighbor_cells_num, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
    timer.stopTimer(&timer.prepare_for_points_in_dense_cells);
}

// 借助 cuCollection 来进行加速
void find_neighbors_of_cells_gpu(ScanState &state) {
    timer.startTimer(&timer.find_neighbor_cells);
    map<CELL_ID_TYPE, vector<CELL_ID_TYPE>> neighbor_cells_of_dense_cells = find_neighbor_cells_extend(state);
    timer.stopTimer(&timer.find_neighbor_cells);
    
    // point->cell_id, cell_id->neighbors_cells, neighbors_cells->(start_pos, len) list
    timer.startTimer(&timer.put_neighbor_cells_list);
    vector<int> &neighbor_cells_list = state.neighbor_cells_list;
    vector<int> &neighbor_cells_capacity = state.neighbor_cells_capacity;
    neighbor_cells_list.clear();
    neighbor_cells_capacity.clear();
    unordered_map<CELL_ID_TYPE, pair<int, int>> neighbor_cells_pos_and_num;
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
    // CUDA：将 neighbor_cells_pos_and_num 转移到 GPU 中，调用核函数来解决
    for (int j = 0; j < state.window_size; j++) {
        CELL_ID_TYPE cell_id = state.h_point_cell_id[j];
        if (neighbor_cells_pos_and_num.count(cell_id)) {
            neighbor_cells_pos[j] = neighbor_cells_pos_and_num[cell_id].first;
            neighbor_cells_num[j] = neighbor_cells_pos_and_num[cell_id].second;
        } else {
            neighbor_cells_pos[j] = -1;
            neighbor_cells_num[j] = -1;
        }
    }
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_pos, neighbor_cells_pos, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.params.d_neighbor_cells_num, neighbor_cells_num, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
    timer.stopTimer(&timer.prepare_for_points_in_dense_cells);
}

void cluster_dense_cells_cpu_check(ScanState &state) {
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
    }

    for (int i = 0; i < state.window_size; i++) {
        if (state.h_label[i] == 2) continue;
        find(i, cid);
    }
    printf("cluster_dense_cells_cpu done!\n");
}

void dbscan_with_cuda(ScanState &state) {
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

void dbscan_naive(ScanState &state, bool timing) {
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
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int)));
    find_neighbors(state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK();

    // * Start sliding
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    unsigned long cluster_ray_intersections = 0;
    CUDA_CHECK(cudaMemset(state.params.cluster_ray_intersections, 0, sizeof(unsigned)));
    while (remaining_data_num >= state.stride_size) {
#if DEBUG_INFO == 1
        CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.window_size * sizeof(int)));
#endif

        timer.startTimer(&timer.total);

        state.params.out = state.params.window + update_pos * state.stride_size;
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
#if DEBUG_INFO == 1
        print_rt_info(state);
#endif

        // set labels and find cores
        timer.startTimer(&timer.find_cores);
        find_cores(state.params.label, state.params.nn, state.params.cluster_id, state.window_size, state.min_pts, 0);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.find_cores);
        timer.startTimer(&timer.set_cluster_id);
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.union_cluster_id);
        // * serial union-find
        post_cluster(state.params.label, state.params.cluster_id, state.window_size, 0);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
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

void dbscan_identify_cores(ScanState &state, bool timing) {
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
        timer.startTimer(&timer.find_cores);
        find_neighbors_cores(state, update_pos, 0);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.find_cores);

        // CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        // int num_cores = 0;
        // for (int i = 0; i < state.window_size; i++) {
        //     if (state.h_label[i] == 0)
        //         num_cores++;
        // }
        // printf("number of cores: %d\n", num_cores);

        timer.startTimer(&timer.whole_bvh);
        rebuild_gas(state); // 不再更新aabb，因为rebuild_gas已经更新好
        timer.stopTimer(&timer.whole_bvh);

        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        // printf("window_id=%d, before, h_label[%d]=%d, h_cluster_id[%d]=%d\n", stride_num + 1, 836, state.h_label[836], 836, state.h_cluster_id[836]);

#if DEBUG_INFO == 1
        CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.window_size * sizeof(int)));
#endif
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
#if DEBUG_INFO == 1
        print_rt_info(state);
#endif
        // printf("[Step] Finish window %d\n", window_left / state.stride_size);
    }
    CUDA_CHECK(cudaStreamDestroy(state.stream));
    printf("[Step] Finish sliding the window...\n");
}

// ! `early_cluster` is useless
void dbscan_with_grid(ScanState &state, bool timing) {
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
        CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
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
            CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i); // get_cell_id 方法需要改造一下
            state.cell_point_num.erase(cell_id); // TODO: state.cell_point_num[cell_id]--，不应该直接删除该item
        }
        for (int i = window_right; i < window_right + state.stride_size; i++) {
            CELL_ID_TYPE cell_id = get_cell_id(state.h_data, state.min_value, state.cell_count, state.cell_length, i);
            state.cell_point_num[cell_id]++;
        }

        CUDA_CHECK(cudaMemcpy(state.h_window, state.params.window, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyDeviceToHost));
        unordered_map<int, int> cell_repres;
        int dense_core_num = 0;
        for (int i = 0; i < state.window_size; i++) {
            CELL_ID_TYPE cell_id = get_cell_id(state.h_window, state.min_value, state.cell_count, state.cell_length, i);
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

void dbscan_hybrid_bvh(ScanState &state, bool timing) {
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
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        timer.startTimer(&timer.find_cores);
        find_neighbors_cores(state, update_pos, 0);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.find_cores);
        update_grid_without_vector(state, update_pos, window_left, window_right);
        // set_centers_radii_cpu(state, pos_arr);
        timer.startTimer(&timer.set_centers_radii);
        set_centers_radii_gpu(state, pos_arr);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_centers_radii);
        make_gas_by_cell(state, timer);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.pre_process);

        timer.startTimer(&timer.set_cluster_id);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_cluster_id);

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

        printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void dbscan_async(ScanState &state, bool timing) {
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
        find_neighbors_cores_async(state, update_pos, state.stream);
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
        printf("[Step] Finish window %d\n", stride_num);
    }
    CUDA_CHECK(cudaStreamDestroy(state.stream));
    printf("[Step] Finish sliding the window...\n");
}

void dbscan_grid_cores_like_rtod(ScanState &state, bool timing) {
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
    // * Start sliding
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
        
#if DEBUG_INFO == 1
        CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.window_size * sizeof(int)));
#endif
        timer.startTimer(&timer.set_cluster_id);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
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
#if DEBUG_INFO == 1
        print_rt_info(state);
#endif
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void dbscan_grid_cores_like_rtod_friendly_gpu_grid_storing(ScanState &state, bool timing) {
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

void dbscan_grid_cores_like_rtod_early_cluster_dense_cells(ScanState &state, bool timing) {
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

void dbscan_grid_cores_hybrid_bvh(ScanState &state, bool timing) {
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
        for (int i = 0; i < state.params.center_num; i++) { // TODO: Can be accelerated by CUDA
            if (state.h_nn[i] < state.min_pts) { // Mark noises
                state.h_label[state.h_center_idx_in_window[i]] = 2; // Noise
            }
        }
        set_centers_radii_gpu(state, pos_arr); // 其中会设置 cluster_id
        make_gas_by_cell(state, timer);
        CUDA_CHECK(cudaMemcpy(state.params.label, state.h_label, state.window_size * sizeof(int), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.pre_process);
        
        timer.startTimer(&timer.set_cluster_id);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
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

void dbscan_grid_cores_hybrid_bvh_op(ScanState &state, bool timing) {
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
        state.h_point_cell_id[i] = cell_id;
        state.cell_points[cell_id].push_back(i);
#ifdef COMPUTE_CELL_CENTER
        if (!state.cell_centers.count(cell_id))
            state.cell_centers[cell_id] = compute_cell_center(state, state.h_data[i]);
#endif
    }
    CUDA_CHECK(cudaMemcpy(state.params.point_cell_id, state.h_point_cell_id, state.window_size * sizeof(CELL_ID_TYPE), cudaMemcpyHostToDevice));
    int *pos_arr = state.pos_arr;
    for (int i = 0; i < state.window_size; i++) pos_arr[i] = i;
    sort(pos_arr, pos_arr + state.window_size, 
         [point_cell_id = state.h_point_cell_id](size_t i1, size_t i2) { 
            return point_cell_id[i1] == point_cell_id[i2] ? i1 < i2 : point_cell_id[i1] < point_cell_id[i2];
         });
    // * Start sliding
    CUDA_CHECK(cudaMallocHost(&state.h_window, state.window_size * sizeof(DATA_TYPE_3)));
    state.h_nn = (int*) malloc(state.window_size * sizeof(int));
    memcpy(state.h_window, state.h_data, state.window_size * sizeof(DATA_TYPE_3));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
#ifdef USE_OMP
    omp_set_num_threads(72);
#endif
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        
        timer.startTimer(&timer.transfer_data);
        memcpy(state.h_window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3));
        CUDA_CHECK(cudaMemcpy(state.params.window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.transfer_data);
        
        timer.startTimer(&timer.update_grid);
#ifdef USE_OMP
        update_grid_without_vector_parallel(state, update_pos, window_left, window_right);
#else
    #ifdef GRID_VECTOR
        update_grid_using_unordered_map(state, update_pos, window_left, window_right);
    #else
        update_grid_without_vector(state, update_pos, window_left, window_right);
    #endif
#endif
        timer.stopTimer(&timer.update_grid);
        
        set_hybrid_aabb(state);
        timer.startTimer(&timer.build_bvh);
        make_gas_by_sparse_points(state, timer);
        // CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.build_bvh);
        
        CUDA_CHECK(cudaMemset(state.params.nn, 0, state.params.sparse_num * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        timer.startTimer(&timer.find_cores);
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        // CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.find_cores);
        timer.startTimer(&timer.set_label);
        CUDA_CHECK(cudaMemset(state.params.label, 0, state.window_size * sizeof(int)));
        set_label(state.params.cell_points, state.params.nn, state.min_pts, state.params.label, state.params.sparse_num);
        // CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_label);
        timer.stopTimer(&timer.pre_process);

        // 构建好 hybrid BVH tree 后聚类
        timer.startTimer(&timer.set_cluster_id);
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        // CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.union_cluster_id); 
        post_cluster(state.params.label, state.params.cluster_id, state.window_size, 0);
        // CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
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
#if DEBUG_INFO == 1
        print_rt_info(state);
#endif
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void dbscan_grid_cores_gpu(ScanState &state, bool timing) {
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
    compute_cell_id(state.params.window, state.params.point_cell_id, state.window_size, state.params.min_value,
                    state.params.cell_count, state.params.cell_length);
    sortByCellIdAndOrder(state.params.pos_arr, state.params.point_cell_id, state.window_size, 0);
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
#if DEBUG_INFO == 1
        CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_intersections_cluster, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits_cluster, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.num_dist_calculations, 0, sizeof(int)));
#endif
        
        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        
        timer.startTimer(&timer.transfer_data);
        CUDA_CHECK(cudaMemcpy(state.params.window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.transfer_data);
        
        timer.startTimer(&timer.update_grid);
        update_grid_thrust(state, update_pos, window_left, window_right);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.update_grid);

        timer.startTimer(&timer.early_cluster);
        set_centers_sparse_gpu(state);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.early_cluster);

        timer.startTimer(&timer.build_bvh);
        make_gas_by_sparse_points(state, timer);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.build_bvh);
        
        timer.startTimer(&timer.find_cores);
        CUDA_CHECK(cudaMemset(state.params.nn, 0, state.params.center_num * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaMemset(state.params.label, 0, state.window_size * sizeof(int)));
        set_label(state.params.cell_points, state.params.nn, state.min_pts, state.params.label, state.params.sparse_num);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.find_cores);
        timer.stopTimer(&timer.pre_process);
        
        // Build whole BVH tree for clustering
        rebuild_gas_from_all_points_in_window(state);
        CUDA_SYNC_CHECK();
        
        timer.startTimer(&timer.set_cluster_id);
        thrust_sequence(state.params.cluster_id, state.window_size, 0);
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.startTimer(&timer.union_cluster_id); 
        post_cluster(state.params.label, state.params.cluster_id, state.window_size, 0);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.union_cluster_id);
        timer.stopTimer(&timer.set_cluster_id);
        
        timer.startTimer(&timer.transfer_data);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        timer.stopTimer(&timer.transfer_data);

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
#if DEBUG_INFO == 1
        print_rt_info(state);
#endif
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void dbscan_with_two_bvh_trees(ScanState &state, bool timing) {
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
    compute_cell_id(state.params.window, state.params.point_cell_id, state.window_size, state.params.min_value,
                    state.params.cell_count, state.params.cell_length);
    sortByCellIdAndOrder(state.params.pos_arr, state.params.point_cell_id, state.window_size, 0);
    // * Start sliding
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
#if DEBUG_INFO == 1
        CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_intersections_cluster, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits_cluster, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.num_dist_calculations, 0, sizeof(int)));
#endif

        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        
        timer.startTimer(&timer.transfer_data);
        CUDA_CHECK(cudaMemcpy(state.params.window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.transfer_data);
        
        timer.startTimer(&timer.update_grid);
        update_grid_thrust(state, update_pos, window_left, window_right);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.update_grid);
        
        timer.startTimer(&timer.early_cluster);
        set_hybrid_aabb_gpu(state);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.early_cluster);
        
        timer.startTimer(&timer.build_bvh);
        make_gas_by_sparse_points(state, timer);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.build_bvh);
        
        timer.startTimer(&timer.find_cores);
        CUDA_CHECK(cudaMemset(state.params.nn, 0, state.params.sparse_num * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.startTimer(&timer.set_label);
        CUDA_CHECK(cudaMemset(state.params.label, 0, state.window_size * sizeof(int)));
        set_label(state.params.cell_points, state.params.nn, state.min_pts, state.params.label, state.params.sparse_num);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.set_label);
        timer.stopTimer(&timer.find_cores);
        timer.stopTimer(&timer.pre_process);

        make_gas_from_small_big_sphere(state, timer);
        timer.startTimer(&timer.set_cluster_id);
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.startTimer(&timer.union_cluster_id); 
        post_cluster(state.params.label, state.params.cluster_id, state.window_size, 0);
        timer.stopTimer(&timer.union_cluster_id);
 #ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.transfer_data);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        timer.stopTimer(&timer.transfer_data);

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
#if DEBUG_INFO == 1
        print_rt_info(state);
#endif
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void dbscan(ScanState &state, bool timing) {
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
    compute_cell_id(state.params.window, state.params.point_cell_id, state.window_size, state.params.min_value,
                    state.params.cell_count, state.params.cell_length);
    sortByCellIdAndOrder(state.params.pos_arr, state.params.point_cell_id, state.window_size, 0);
    // * Start sliding
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    if (!timing) printf("[Info] checking\n");
    while (remaining_data_num >= state.stride_size) {
#if DEBUG_INFO == 1
        CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_intersections_cluster, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits_cluster, 0, state.window_size * sizeof(int)));
        CUDA_CHECK(cudaMemset(state.params.num_dist_calculations, 0, sizeof(int)));
#endif

        timer.startTimer(&timer.total);
        timer.startTimer(&timer.pre_process);
        
        timer.startTimer(&timer.transfer_data);
        CUDA_CHECK(cudaMemcpy(state.params.window + update_pos * state.stride_size, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        timer.stopTimer(&timer.transfer_data);
        
        timer.startTimer(&timer.update_grid);
        update_grid_thrust(state, update_pos, window_left, window_right);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.update_grid);
        
        timer.startTimer(&timer.early_cluster);
        set_hybrid_aabb_gpu(state);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.early_cluster);
        
        timer.startTimer(&timer.build_bvh);
        make_gas_by_sparse_points(state, timer);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.build_bvh);
        
        timer.startTimer(&timer.find_cores);
        CUDA_CHECK(cudaMemset(state.params.nn, 0, state.params.sparse_num * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.startTimer(&timer.set_label);
        CUDA_CHECK(cudaMemset(state.params.label, 0, state.window_size * sizeof(int)));
        set_label(state.params.cell_points, state.params.nn, state.min_pts, state.params.label, state.params.sparse_num);
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.set_label);
        timer.stopTimer(&timer.find_cores);
        timer.stopTimer(&timer.pre_process);

        timer.startTimer(&timer.set_cluster_id);
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt_cluster, state.window_size, 1, 1));
#ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.startTimer(&timer.union_cluster_id); 
        post_cluster(state.params.label, state.params.cluster_id, state.window_size, 0);
        timer.stopTimer(&timer.union_cluster_id);
 #ifdef USE_SYNC
        CUDA_SYNC_CHECK();
#endif
        timer.stopTimer(&timer.set_cluster_id);

        timer.startTimer(&timer.transfer_data);
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        timer.stopTimer(&timer.transfer_data);

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
#if DEBUG_INFO == 1
        print_rt_info(state);
#endif
        // printf("[Step] Finish window %d\n", stride_num);
    }
    printf("[Step] Finish sliding the window...\n");
}

void cleanup(ScanState &state) {
    // free host memory
    free(state.h_data);
    free(state.h_point_cell_id);
    free(state.tmp_pos_arr);
    free(state.new_pos_arr);
    free(state.h_big_sphere);
    CUDA_CHECK(cudaFreeHost(state.d_cell_points));
    CUDA_CHECK(cudaFreeHost(state.h_label));
    CUDA_CHECK(cudaFreeHost(state.h_cluster_id));
    CUDA_CHECK(cudaFreeHost(state.h_window));
    CUDA_CHECK(cudaFreeHost(state.pos_arr));
    CUDA_CHECK(cudaFreeHost(state.uniq_pos_arr));
    CUDA_CHECK(cudaFreeHost(state.num_points));
    CUDA_CHECK(cudaFreeHost(state.h_point_status));
    CUDA_CHECK(cudaFreeHost(state.h_centers_p));
    CUDA_CHECK(cudaFreeHost(state.h_center_idx_in_window_p));
    CUDA_CHECK(cudaFreeHost(state.h_cell_point_num_p));
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
    CUDA_CHECK(cudaFree(state.params.new_pos_arr));
    CUDA_CHECK(cudaFree(state.params.tmp_pos_arr));
    CUDA_CHECK(cudaFree(state.params.center_idx_in_window));
    CUDA_CHECK(cudaFree(state.params.offsets));
    CUDA_CHECK(cudaFree(state.params.num_offsets));
    CUDA_CHECK(cudaFree(state.params.num_points_in_dense_cells));
    CUDA_CHECK(cudaFree(state.params.num_dense_cells));
    CUDA_CHECK(cudaFree(state.params.sparse_offset));
    CUDA_CHECK(cudaFree(state.params.dense_offset));

    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_pos));
    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_num));
    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_list));
    CUDA_CHECK(cudaFree(state.params.d_neighbor_cells_capacity));

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaFree(state.params.ray_primitive_hits));
    CUDA_CHECK(cudaFree(state.params.ray_intersections));
    CUDA_CHECK(cudaFree(state.params.ray_primitive_hits_cluster));
    CUDA_CHECK(cudaFree(state.params.ray_intersections_cluster));
    CUDA_CHECK(cudaFree(state.params.num_dist_calculations));
    free(state.h_ray_hits);
    free(state.h_ray_intersections);
#endif

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));

    // Free memory for CUDA implementation
    free(state.h_nn);
    if (state.check) {
        CUDA_CHECK(cudaFree(state.params.check_nn));
        CUDA_CHECK(cudaFree(state.params.check_label));
        CUDA_CHECK(cudaFree(state.params.check_cluster_id));
        free(state.check_h_nn);
        free(state.check_h_label);
        free(state.check_h_cluster_id);
    }
}

int main(int argc, char *argv[]) {
    setbuf(stdout, NULL); // https://blog.csdn.net/moshiyaofei/article/details/107297472
    ScanState state;
    parse_args(state, argc, argv);
    if (state.data_file.find("tao") != string::npos) {
        read_data_from_tao(state.data_file, state);
        state.check = true;
    } else if (state.data_file.find("geolife") != string::npos) {
        read_data_from_geolife(state.data_file, state);
        state.check = false;
    } else if (state.data_file.find("RBF") != string::npos) {
        read_data_from_rbf(state.data_file, state);
        state.check = true;
    } else if (state.data_file.find("EDS") != string::npos) {
        read_data_from_eds(state.data_file, state);
    } else if (state.data_file.find("stock") != string::npos) {
        read_data_from_stk(state.data_file, state);
        state.check = true;
    } else {
        printf("[Error] Dataset %s does not exist!\n", state.data_file.c_str());
    }

    initialize_optix(state);
    make_module(state);
    make_program_groups(state);
    make_pipeline(state);
    make_sbt(state);
    initialize_params(state);
    
    // Warmup
    for (int i = 0; i < 10; i++) {
#if OPTIMIZATION_LEVEL == 9
        dbscan(state, !state.check);
#elif OPTIMIZATION_LEVEL == 8
        dbscan_grid_cores_hybrid_bvh(state, !state.check);
#elif OPTIMIZATION_LEVEL == 7
        dbscan_grid_cores_like_rtod_early_cluster_dense_cells(state, !state.check);
#elif OPTIMIZATION_LEVEL == 5
        dbscan_grid_cores_gpu(state, !state.check);
#elif OPTIMIZATION_LEVEL == 4
        dbscan_async(state, !state.check);
#elif OPTIMIZATION_LEVEL == 3
        dbscan_hybrid_bvh(state, !state.check);
#elif OPTIMIZATION_LEVEL == 2
        dbscan_with_two_bvh_trees(state, true);
#elif OPTIMIZATION_LEVEL == 1
        dbscan_identify_cores(state, !state.check);
#elif OPTIMIZATION_LEVEL == 0
        dbscan_naive(state, !state.check);
#elif OPTIMIZATION_LEVEL == 100
        dbscan_with_cuda(state);
#endif
        printf("[Step] Warmup %d\n", i);
        timer.clear();
        state.cell_point_num.clear();
        state.cell_points.clear();
        state.cell_repres.clear();
    }

    state.intersections_all_window = 0;
    state.hits_all_window = 0;
    state.intersections_all_window_cluster = 0;
    state.hits_all_window_cluster = 0;
    state.num_spheres_all_window = 0;
    state.num_points_in_dense_cells = 0;
    state.num_dist_calculations_all_window = 0;
    // Timing
#if OPTIMIZATION_LEVEL == 9
    dbscan(state, true);
    state.num_rays_in_window = state.window_size;
#elif OPTIMIZATION_LEVEL == 8
    dbscan_grid_cores_hybrid_bvh(state, true);
#elif OPTIMIZATION_LEVEL == 7
    dbscan_grid_cores_like_rtod_early_cluster_dense_cells(state, true);
#elif OPTIMIZATION_LEVEL == 5
    dbscan_grid_cores_gpu(state, true);
    state.num_rays_in_window = state.window_size;
#elif OPTIMIZATION_LEVEL == 4
    dbscan_async(state, true);
#elif OPTIMIZATION_LEVEL == 3
    dbscan_hybrid_bvh(state, true);
#elif OPTIMIZATION_LEVEL == 2
    dbscan_with_two_bvh_trees(state, true);
#elif OPTIMIZATION_LEVEL == 1
    dbscan_identify_cores(state, true);
#elif OPTIMIZATION_LEVEL == 0
    dbscan_naive(state, true);
    state.num_rays_in_window = state.stride_size * 2;
#elif OPTIMIZATION_LEVEL == 100
    dbscan_with_cuda(state);
#endif

#if DEBUG_INFO == 1
    print_overall_rt_info(state, (state.data_num - state.window_size) / state.stride_size);
#endif
    timer.showTime((state.data_num - state.window_size) / state.stride_size);
    cleanup(state);
    return 0;
}