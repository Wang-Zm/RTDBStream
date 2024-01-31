#ifndef STATE_H
#define STATE_H

#include <float.h>
#include <vector_types.h>
#include <optix_types.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "dbscan.h"
#include "timer.h"

using namespace std;

struct ScanState
{
    Params                          params;
    CUdeviceptr                     d_params;
    OptixDeviceContext              context                   = nullptr;
    OptixBuildInput                 vertex_input              = {};

    CUdeviceptr*                    d_gas_output_buffer_list;
    CUdeviceptr*                    d_gas_temp_buffer_list;
    CUdeviceptr                     d_gas_output_buffer       = 0;
    CUdeviceptr                     d_gas_temp_buffer         = 0;
    CUdeviceptr                     d_gas_output_buffer_hybrid= 0;
    CUdeviceptr                     d_gas_temp_buffer_hybrid  = 0;
    OptixTraversableHandle*         handle_list;


    OptixAccelBufferSizes           gas_buffer_sizes;
    const uint32_t                  vertex_input_flags[1]     = {OPTIX_GEOMETRY_FLAG_NONE};
    CUdeviceptr                     d_aabb_ptr                = 0;
    DATA_TYPE_3*                    new_stride;

    OptixModule                     module                    = nullptr;

    OptixProgramGroup               raygen_prog_group         = nullptr;
    OptixProgramGroup               miss_prog_group           = nullptr;
    OptixProgramGroup               hitgroup_prog_group       = nullptr;
    OptixProgramGroup               raygen_cluster            = nullptr;
    OptixProgramGroup               hitgroup_cluster          = nullptr;

    OptixPipeline                   pipeline                  = nullptr;
    OptixPipeline                   pipeline_cluster          = nullptr;
    OptixPipelineCompileOptions     pipeline_compile_options  = {};

    OptixShaderBindingTable         sbt                       = {}; 
    OptixShaderBindingTable         sbt_cluster               = {}; 

    std::string                     data_file;
    int                             data_num;
    DATA_TYPE_3*                    h_data;
    vector<DATA_TYPE>               max_value;
    vector<DATA_TYPE>               min_value;
    // int                             dim;
    int                             window_size;
    int                             stride_size;
    DATA_TYPE                       radius;
    DATA_TYPE                       radius_one_half;
    int                             min_pts;
    DATA_TYPE_3*                    h_window;
    int*                            h_nn;
    int*                            h_label;
    int*                            h_cluster_id;
    int*                            check_h_nn;
    int*                            check_h_label;
    int*                            check_h_cluster_id;

    vector<DATA_TYPE_3>             h_centers;
    vector<DATA_TYPE>               h_radii;
    vector<int>                     h_center_idx_in_window;
    vector<int>                     h_cell_point_num;

    unsigned*                       h_ray_hits;
    unsigned*                       h_ray_intersections;

    unordered_map<int, int>         cell_point_num;
    unordered_map<int, vector<int>> cell_points;
    int**                           d_cell_points;  // 第一层是 GPU 中的指针，d_cell_points[i] (存放于 host mem 中) 的值是 GPU 中的指针
    DATA_TYPE                       cell_length;
    vector<int>                     cell_count;
    int*                            h_point_cell_id;
    unordered_map<int, int>         cell_repres;
};

void read_data_from_tao(string& data_file, ScanState &state);
size_t get_cpu_memory_usage();
void start_gpu_mem(size_t* avail_mem);
void stop_gpu_mem(size_t* avail_mem, size_t* used);
int find(int x, int* cid);
void cluster_with_cpu(ScanState &state, Timer &timer);
void cluster_with_cuda(ScanState &state, Timer &timer);
bool check(ScanState &state, int window_id, Timer &timer);

void initialize_optix(ScanState &state);
// void make_gas(ScanState &state);
void make_gas_for_each_stride(ScanState &state, int unit_num);
// void rebuild_gas(ScanState &state);
void rebuild_gas_stride(ScanState &state, int update_pos);
void make_gas_by_cell(ScanState &state);
void make_module(ScanState &state);
void make_program_groups(ScanState &state);
void make_pipeline(ScanState &state);
void make_sbt(ScanState &state);
#endif