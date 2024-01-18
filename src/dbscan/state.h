#ifndef STATE_H
#define STATE_H

#include <float.h>
#include <vector_types.h>
#include <optix_types.h>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include "optixScan.h"

using namespace std;

// class FixQueue {
// public:
//     int         arr[MK];
//     int         start;
//     int         num;

//     FixQueue() {
//         start = 0;
//         num = 0;
//     }

//     void enqueue(int val) {
//         arr[start] = val;
//         if ((++start) == MK) {
//             start = 0;
//         }
//         num++;
//     }

//     void copy(double3* dsc, int* dsc_idx, double3* h_current_window) {
//         int _start = (start - 1 + MK) % MK;
//         int _num   = num;
//         while (_num > 0) {
//             *dsc = h_current_window[arr[_start]];
//             *dsc_idx = arr[_start];
//             dsc++;
//             dsc_idx++;
//             _start = (_start - 1 + MK) % MK;
//             _num--;
//         }
//     }

//     void copy(double3* dsc, double3* h_current_window) {
//         int _start = (start - 1 + MK) % MK;
//         int _num   = num;
//         while (_num > 0) {
//             *dsc = h_current_window[arr[_start]];
//             dsc++;
//             _start = (_start - 1 + MK) % MK;
//             _num--;
//         }
//     }
// };

struct ScanState
{
    Params                          params;
    CUdeviceptr                     d_params;
    OptixDeviceContext              context                   = nullptr;
    // OptixTraversableHandle          pre_gas_handle;
    OptixTraversableHandle          gas_handle;
    CUdeviceptr                     d_gas_output_buffer       = 0;
    OptixBuildInput                 vertex_input              = {};
    CUdeviceptr                     d_temp_buffer_gas         = 0;
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
    DATA_TYPE*                      max_value;
    DATA_TYPE*                      min_value;
    // int                             dim;
    int                             window_size;
    int                             stride_size;
    DATA_TYPE                       radius;
    int                             min_pts;
    int*                            h_label;
    int*                            h_cluster_id;

    unsigned*                       h_ray_hits;
    unsigned*                       h_ray_intersections;
};

void read_data_from_tao(string& data_file, ScanState &state);
size_t get_cpu_memory_usage();
void start_gpu_mem(size_t* avail_mem);
void stop_gpu_mem(size_t* avail_mem, size_t* used);
#endif