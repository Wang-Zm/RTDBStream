#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

#define DEBUG_INFO 0
#define THREAD_NUM 80

// #ifndef DATA_N
// #define DATA_N  1e8
// #endif

#define MODE 1

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
    int*                    tmp_cluster_id;
    int*                    nn; // number of neighbors
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

#endif