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
    OptixTraversableHandle*         gas_handle;

    OptixModule                     module                    = nullptr;

    OptixProgramGroup               raygen_prog_group         = nullptr;
    OptixProgramGroup               miss_prog_group           = nullptr;
    OptixProgramGroup               hitgroup_prog_group       = nullptr;

    OptixPipeline                   pipeline                  = nullptr;
    OptixPipelineCompileOptions     pipeline_compile_options  = {};

    OptixShaderBindingTable         sbt                       = {}; 

    std::string                     data_file;
    std::string                     query_file;
    int                             data_num;
    int                             query_num;
    DATA_TYPE**                     vertices;
    DATA_TYPE**                     h_queries;
    int                             dim;
    int                             unsigned_len;
    double                          radius;
    DATA_TYPE*                      max_value;
    DATA_TYPE*                      min_value;
    DIST_TYPE**                     h_dist;
    DIST_TYPE**                     h_dist_temp;
    DATA_TYPE**                     h_queries_temp;
    unsigned*                       h_dist_flag;
    size_t                          dist_flag_len;
    int*                            queries_neighbor_num;
    // vector<int>*                    queries_neighbors;
    // vector<DIST_TYPE>*              queries_neighbor_dist;
    vector<vector<int>>             queries_neighbors;
    vector<vector<DIST_TYPE>>       queries_neighbor_dist;

    unsigned*                       h_ray_hits;
    unsigned*                       h_ray_intersections;
};

void read_data(string& data_file, string& query_file, ScanState &state);
void read_data_from_sift1m_128d(string& data_file, string& query_file, ScanState &state);
void read_data_from_gist_960d(string& data_file, string& query_file, ScanState &state);
size_t get_cpu_memory_usage();
void start_gpu_mem(size_t* avail_mem);
void stop_gpu_mem(size_t* avail_mem, size_t* used);
void check(ScanState &state);
void check_single_thread(ScanState &state);
#endif