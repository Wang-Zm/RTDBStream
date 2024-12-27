#ifndef STATE_H
#define STATE_H

#include <float.h>
#include <vector_types.h>
#include <optix_types.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include "dbscan.h"
#include "timer.h"

// #define COMPUTE_CELL_CENTER // * ineffective
// #define USE_OMP // * Low performance
// #define GRID_VECTOR // * Low performance

using namespace std;

extern Timer timer;

struct ScanState
{
    Params                              params;
    CUdeviceptr                         d_params;
    OptixDeviceContext                  context                   = nullptr;
    OptixBuildInput                     vertex_input              = {};

    CUdeviceptr*                        d_gas_output_buffer_list;
    CUdeviceptr*                        d_gas_temp_buffer_list;
    CUdeviceptr                         d_gas_output_buffer       = 0;
    CUdeviceptr                         d_gas_temp_buffer         = 0;
    CUdeviceptr                         d_gas_output_buffer_hybrid= 0;
    CUdeviceptr                         d_gas_temp_buffer_hybrid  = 0;
    OptixTraversableHandle*             handle_list;
    OptixTraversableHandle              out_stride_gas_handle;
    OptixTraversableHandle              in_stride_gas_handle;


    OptixAccelBufferSizes               gas_buffer_sizes;
    const uint32_t                      vertex_input_flags[1]     = {OPTIX_GEOMETRY_FLAG_NONE};
    CUdeviceptr                         d_aabb_ptr                = 0;
    DATA_TYPE_3*                        new_stride;

    OptixModule                         module                    = nullptr;

    OptixProgramGroup                   raygen_prog_group         = nullptr;
    OptixProgramGroup                   miss_prog_group           = nullptr;
    OptixProgramGroup                   hitgroup_prog_group       = nullptr;
    OptixProgramGroup                   raygen_cluster            = nullptr;
    OptixProgramGroup                   hitgroup_cluster          = nullptr;

    OptixPipeline                       pipeline                  = nullptr;
    OptixPipeline                       pipeline_cluster          = nullptr;
    OptixPipelineCompileOptions         pipeline_compile_options  = {};

    OptixShaderBindingTable             sbt                       = {}; 
    OptixShaderBindingTable             sbt_cluster               = {}; 

    std::string                         data_file;
    int                                 data_num;
    DATA_TYPE_3*                        h_data;
    vector<DATA_TYPE>                   max_value;
    vector<DATA_TYPE>                   min_value;
    // int                                 dim;
    int                                 window_size;
    int                                 stride_size;
    DATA_TYPE                           radius;
    DATA_TYPE                           radius_one_half;
    int                                 min_pts;
    DATA_TYPE_3*                        h_window;
    int*                                h_nn;
    int*                                h_label;
    int*                                h_cluster_id;
    int*                                check_h_nn;
    int*                                check_h_label;
    int*                                check_h_cluster_id;
    bool                                check;

    vector<DATA_TYPE_3>                 h_centers;
    vector<DATA_TYPE>                   h_radii;
    vector<int>                         h_center_idx_in_window;
    vector<int>                         h_cell_point_num;
    DATA_TYPE_3*                        h_centers_p;
    // DATA_TYPE*                          h_radii_p;
    int*                                h_center_idx_in_window_p;
    int*                                h_cell_point_num_p;
    int*                                h_point_status;
    int*                                h_big_sphere;

    unsigned*                           h_ray_hits;
    unsigned*                           h_ray_intersections;

    unordered_map<CELL_ID_TYPE, int>            cell_point_num;
    unordered_map<CELL_ID_TYPE, vector<int>>    cell_points;
    unordered_map<int, int*>            cell_points_ptr;
    int**                               d_cell_points;  // 第一层是 GPU 中的指针，d_cell_points[i] (存放于 host mem 中) 的值是 GPU 中的指针
    int*                                points_in_dense_cells;
    int*                                pos_arr;
    int*                                tmp_pos_arr;
    int*                                new_pos_arr;
    DATA_TYPE                           cell_length;
    vector<int>                         cell_count;
    CELL_ID_TYPE*                       h_point_cell_id;
    unordered_map<int, int>             cell_repres;
    int*                                uniq_pos_arr;
    int*                                num_points;
    unordered_map<CELL_ID_TYPE, int>    pos_of_cell;
    unordered_map<CELL_ID_TYPE, DATA_TYPE_3> cell_centers;

    vector<int>                         neighbor_cells_list;
    vector<int>                         neighbor_cells_capacity;
    int*                                neighbor_cells_pos;
    int*                                neighbor_cells_num;

    cudaStream_t                        stream;
    cudaStream_t                        stream2;
};

inline CELL_ID_TYPE get_cell_id(DATA_TYPE_3* data, vector<DATA_TYPE>& min_value, vector<int>& cell_count, DATA_TYPE cell_length, int i) {
    CELL_ID_TYPE dim_id_x = (data[i].x - min_value[0]) / cell_length;
    CELL_ID_TYPE dim_id_y = (data[i].y - min_value[1]) / cell_length;
    CELL_ID_TYPE dim_id_z = (data[i].z - min_value[2]) / cell_length;
    CELL_ID_TYPE id = dim_id_x * cell_count[1] * cell_count[2] + dim_id_y * cell_count[2] + dim_id_z;
    return id;
}

void read_data_from_tao(string& data_file, ScanState &state);
void read_data_from_geolife(string& data_file, ScanState &state);
void read_data_from_rbf(string& data_file, ScanState &state);
void read_data_from_eds(string& data_file, ScanState &state);
void read_data_from_stk(string& data_file, ScanState &state);
size_t get_cpu_memory_usage();
void start_gpu_mem(size_t* avail_mem);
void stop_gpu_mem(size_t* avail_mem, size_t* used);
int find(int x, int* cid);
void unite(int x, int y, int* cid);
void cluster_with_cpu(ScanState &state, Timer &timer);
void cluster_with_cuda(ScanState &state, Timer &timer);
bool check(ScanState &state, int window_id, Timer &timer);
void printUsageAndExit(const char* argv0);
void parse_args(ScanState &state, int argc, char *argv[]);
void calc_cluster_num(int* cluster_id, int n, int min_pts);

void initialize_optix(ScanState &state);
void make_gas(ScanState &state);
void make_gas_for_each_stride(ScanState &state, int unit_num);
void make_gas_by_sparse_points(ScanState &state, Timer &timer);
void rebuild_gas(ScanState &state, int update_pos);
void rebuild_gas(ScanState &state);
void rebuild_gas_from_all_points_in_window(ScanState &state);
void rebuild_gas_stride(ScanState &state, int update_pos, const cudaStream_t &stream);
void rebuild_gas_stride(ScanState &state, int update_pos, OptixTraversableHandle& gas_handle);
void make_gas_by_cell_grid(ScanState &state);
void make_gas_by_cell(ScanState &state, Timer &timer);
void make_gas_from_small_big_sphere(ScanState &state, Timer &timer);
void make_module(ScanState &state);
void make_program_groups(ScanState &state);
void make_pipeline(ScanState &state);
void make_sbt(ScanState &state);

void update_grid(ScanState &state, int update_pos, int window_left, int window_right);
void update_grid_without_vector(ScanState &state, int update_pos, int window_left, int window_right);
void update_grid_without_vector_parallel(ScanState &state, int update_pos, int window_left, int window_right);
void update_grid_using_unordered_map(ScanState &state, int update_pos, int window_left, int window_right);
#endif