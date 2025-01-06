#ifndef TIMER_H
#define TIMER_H

#include <mutex>
#include <sys/time.h>
#include <iostream>

using namespace std;

class Timer{
  public:

  double timebase;

  double build_bvh;
  double out_stride;
  double rebuild_bvh;
  double in_stride;
  double find_cores;

  double whole_bvh;
  double hybrid_sphere;
  double build_hybrid_bvh;

  double transfer_data;
  double pre_process;
  double update_grid;
  double early_cluster;
  double set_cluster_id;
  double union_cluster_id;
  double copy_cluster_d2h;
  double free_cell_points;

  double out_stride_bvh;
  double out_stride_ray;
  double in_stride_bvh;
  double in_stride_ray;

  double cuda_find_neighbors;
  double cuda_set_clusters;

  double total;
  double cpu_cluter_total;
  double cuda_cluter_total;

  double dense_cell_points_copy;
  double cell_points_memcpy;
  double get_dense_sphere;

  double update_h_point_cell_id;
  double sort_h_point_cell_id;
  double get_centers_radii;
  double compute_uniq_pos_arr;
  double set_centers_radii;

  double find_neighbors_of_cells;
  double cluster_dense_cells;
  double find_neighbor_cells;
  double put_neighbor_cells_list;
  double prepare_for_points_in_dense_cells;
  double set_label;
  double set_sparse_spheres;
  double set_dense_spheres;
  double gen_aabb;
  double merge_pos_arr;
  double sort_new_stride;
  double compute_offsets;
  double count_sparse_points;

  cudaEvent_t start1, stop1;
  cudaEvent_t start2, stop2;
  float milliseconds1, milliseconds2;

  Timer() {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    timebase = t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
    clear();
    // CUDA_CHECK(cudaEventCreate(&start1));
    // CUDA_CHECK(cudaEventCreate(&stop1));
    // CUDA_CHECK(cudaEventCreate(&start2));
    // CUDA_CHECK(cudaEventCreate(&stop2));
    // * If the cudaEvent is not synchronized, calling `cudaSynchronize()` will toggle error
  }

  // ~Timer() {
  //   CUDA_CHECK(cudaEventDestroy(start1));
  //   CUDA_CHECK(cudaEventDestroy(stop1));
  //   CUDA_CHECK(cudaEventDestroy(start2));
  //   CUDA_CHECK(cudaEventDestroy(stop2));
  // }

  void clear() {
    build_bvh = 0;
    out_stride = 0;
    rebuild_bvh = 0;
    in_stride = 0;
    find_cores = 0;

    whole_bvh = 0;
    hybrid_sphere = 0;
    build_hybrid_bvh = 0;

    transfer_data = 0;
    pre_process = 0;
    update_grid = 0;
    early_cluster = 0;
    set_cluster_id = 0;
    union_cluster_id = 0;
    copy_cluster_d2h = 0;
    free_cell_points = 0;

    out_stride_bvh = 0;
    out_stride_ray = 0;
    in_stride_bvh  = 0;
    in_stride_ray  = 0;

    cuda_find_neighbors = 0;
    cuda_set_clusters = 0;

    total = 0;
    cpu_cluter_total = 0;
    cuda_cluter_total = 0;

    dense_cell_points_copy = 0;
    cell_points_memcpy = 0;
    get_dense_sphere = 0;

    update_h_point_cell_id = 0;
    sort_h_point_cell_id = 0;
    get_centers_radii = 0;
    compute_uniq_pos_arr = 0;
    set_centers_radii = 0;

    find_neighbors_of_cells = 0;
    cluster_dense_cells = 0;
    find_neighbor_cells = 0;
    put_neighbor_cells_list = 0;
    prepare_for_points_in_dense_cells = 0;
    set_label = 0;
    set_sparse_spheres = 0;
    set_dense_spheres = 0;
    gen_aabb = 0;
    merge_pos_arr = 0;
    sort_new_stride = 0;
    compute_offsets = 0;
    count_sparse_points = 0;
  }

  void startTimer(double *t) {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    *t -= (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }

  void stopTimer(double *t) {
    struct timeval t1;                           
    gettimeofday(&t1, NULL);
    *t += (t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0) - timebase;
  }


  // void average(int n) {
  //   copy_new_points_h2d /= n;
  //   copy_filtered_points_h2d /= n;
  //   copy_outlier_d2h /= n;
  //   prepare_cell /= n;
  //   build_bvh /= n;
  //   detect_outlier /= n;
  //   total /= n;
  // }

  void showTime(int query_num) {
    cout << endl;
    cout << "###########   Time  ##########" << endl;
    
    // cout << "[Time] out_stride: " << out_stride / query_num << " ms" << endl;
    // cout << "[Time] rebuild_bvh: " << rebuild_bvh / query_num << " ms" << endl;
    // cout << "[Time] in_stride: " << in_stride / query_num << " ms" << endl;

    cout << "[Time] out_stride_bvh: " << out_stride_bvh / query_num << " ms" << endl;
    cout << "[Time] out_stride_ray: " << out_stride_ray / query_num << " ms" << endl;
    cout << "[Time] in_stride_bvh: " << in_stride_bvh / query_num << " ms" << endl;
    cout << "[Time] in_stride_ray: " << in_stride_ray / query_num << " ms" << endl;

    cout << "[Time] transfer_data: " << transfer_data / query_num << " ms" << endl;
    cout << "[Time] update_grid: " << update_grid / query_num << " ms" << endl;
    cout << "[Time]  update_h_point_cell_id: " << update_h_point_cell_id / query_num << " ms" << endl;
    cout << "[Time]  sort_h_point_cell_id: " << sort_h_point_cell_id / query_num << " ms" << endl;
    cout << "[Time]   sort_new_stride: " << sort_new_stride / query_num << " ms" << endl;
    cout << "[Time]  merge_pos_arr: " << merge_pos_arr / query_num << " ms" << endl;
    cout << "[Time] early_cluster: " << early_cluster / query_num << " ms" << endl;
    cout << "[Time]  set_sparse_spheres: " << set_sparse_spheres / query_num << " ms" << endl;
    cout << "[Time]  set_dense_spheres: " << set_dense_spheres / query_num << " ms" << endl;
    cout << "[Time]  compute_offsets: " << compute_offsets / query_num << " ms" << endl;
    cout << "[Time]  count_sparse_points: " << count_sparse_points / query_num << " ms" << endl;
    cout << "[Time] build_bvh: " << build_bvh / query_num << " ms" << endl;
    cout << "[Time] gen_aabb: " << gen_aabb / query_num << " ms" << endl;
    cout << "[Time] find_cores: " << find_cores / query_num << " ms" << endl;
    cout << "[Time] set_label: " << set_label / query_num << " ms" << endl;
    
    // cout << "[Time] whole_bvh: " << whole_bvh / query_num << " ms" << endl;
    cout << "[Time] hybrid_sphere: " << hybrid_sphere / query_num << " ms" << endl;
    cout << "[Time] build_hybrid_bvh: " << build_hybrid_bvh / query_num << " ms" << endl;

    cout << "[Time] set_cluster_id: " << set_cluster_id / query_num << " ms" << endl;
    cout << "[Time] union_cluster_id: " << union_cluster_id / query_num << " ms" << endl;
    cout << "[Time] free_cell_points: " << free_cell_points / query_num << " ms" << endl;
    // cout << "[Time] copy_cluster_d2h: " << copy_cluster_d2h / query_num << " ms" << endl;
    
    cout << "[Time] pre_process: " << pre_process / query_num << " ms" << endl;
    cout << "[Time] total: " << total / query_num << " ms" << endl << endl;
    
    cout << "[Time] cpu_cluter_total: " << cpu_cluter_total / query_num << " ms" << endl;

    cout << "[Time] cuda_cluter_total: " << cuda_cluter_total / query_num << " ms" << endl;
    cout << "[Time] cuda_find_neighbors: " << cuda_find_neighbors / query_num << " ms" << endl;
    cout << "[Time] cuda_set_clusters: " << cuda_set_clusters / query_num << " ms" << endl;
    
    cout << "[Time] get_dense_sphere: " << get_dense_sphere / query_num << " ms" << endl;
    cout << "[Time] dense_cell_points_copy: " << dense_cell_points_copy / query_num << " ms" << endl;
    cout << "[Time] cell_points_memcpy: " << cell_points_memcpy / query_num << " ms" << endl;

    cout << "[Time] get_centers_radii: " << get_centers_radii / query_num << " ms" << endl;
    cout << "[Time] compute_uniq_pos_arr: " << compute_uniq_pos_arr / query_num << " ms" << endl;
    cout << "[Time] set_centers_radii: " << set_centers_radii / query_num << " ms" << endl;
    cout << endl;

    cout << "[Time] find_neighbors_of_cells: " << find_neighbors_of_cells / query_num << " ms" << endl;
    cout << "[Time] cluster_dense_cells: " << cluster_dense_cells / query_num << " ms" << endl;
    cout << "[Time] find_neighbor_cells: " << find_neighbor_cells / query_num << " ms" << endl;
    cout << "[Time] put_neighbor_cells_list: " << put_neighbor_cells_list / query_num << " ms" << endl;
    cout << "[Time] prepare_for_points_in_dense_cells: " << prepare_for_points_in_dense_cells / query_num << " ms" << endl;

    cout << "##############################" << endl;
    cout << endl;
  }
};
#endif