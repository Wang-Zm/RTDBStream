#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

#define DEBUG_INFO 0
#define THREAD_NUM 80

#define MODE 0
#define OPTIMIZATION_GRID

typedef double DATA_TYPE;
typedef double3 DATA_TYPE_3;

struct Params {
    // OptixTraversableHandle  pre_handle;
    OptixTraversableHandle  handle;
    OptixTraversableHandle  out_stride_handle;
    OptixTraversableHandle  in_stride_handle;
    
    float                   tmin;
    float                   tmax;
    int                     data_num;
    // DATA_TYPE_3*            pre_window;
    DATA_TYPE_3*            window;
    DATA_TYPE_3*            out;
    int*                    label; // 0(core), 1(border), 2(noise)
    int*                    cluster_id;
    // int*                    tmp_cluster_id;
    int*                    nn; // number of neighbors
    int*                    check_label;
    int*                    check_cluster_id;
    int*                    check_nn;
    int                     operation;
    int                     window_size;
    int                     stride_left;
    int                     stride_right;
    // DATA_TYPE_3*            ex_cores;
    // DATA_TYPE_3*            neo_cores;
    // DATA_TYPE_3*            c_out;
    // DATA_TYPE_3*            out_stride;
    // int*                    c_out_num;
    // int*                    ex_cores_num;
    // int*                    neo_cores_num;
    // int*                    ex_cores_idx; // 在 pre_window 中的索引
    // int*                    R_out_f;
    // int*                    M_out_f;

    // int                     bvh_num;
    // double                  radius;
    DATA_TYPE               radius2;
    int                     min_pts;
    unsigned*               intersection_test_num;
    unsigned*               hit_num;

    unsigned*               ray_primitive_hits;
    unsigned*               ray_intersections;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
};


struct HitGroupData
{
};

extern "C" void kGenAABB(DATA_TYPE_3 *points, DATA_TYPE radius, unsigned numPrims, OptixAabb *d_aabb);
extern "C" void find_cores(int* label, int* nn, int* cluster_id, int window_size, int min_pts);
// extern "C" void union_cluster(int* tmp_cluster_id, int* cluster_id, int* label, int window_size);
extern "C" void find_neighbors(int* nn, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2, int min_pts);
extern "C" void set_cluster_id(int* nn, int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2);
#endif