#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

#define DEBUG_INFO 0
#define THREAD_NUM 80

#ifndef DATA_N
#define DATA_N  1e8
#endif

typedef double DIST_TYPE;
#if DATA_WIDTH == 32
typedef float DATA_TYPE;
typedef float3 DATA_TYPE_3;
#else
typedef double DATA_TYPE;
typedef double3 DATA_TYPE_3;
#endif

struct Params {
    OptixTraversableHandle  pre_handle;
    OptixTraversableHandle  handle;
    
    float                   tmin;
    float                   tmax;
    int                     data_num;
    DATA_TYPE_3*            pre_window;
    DATA_TYPE_3*            window;
    DATA_TYPE_3*            out;
    int*                    label; // 0(core), 1(border), 2(noise)
    int*                    cluster_id;
    int*                    nn; // number of neighbors
    int                     operation;
    DATA_TYPE_3*            ex_cores;
    DATA_TYPE_3*            neo_cores;
    DATA_TYPE_3*            c_out;
    DATA_TYPE_3*            out_stride;
    int*                    c_out_num;
    int*                    ex_cores_num;
    int*                    neo_cores_num;
    int*                    ex_cores_idx; // 在 pre_window 中的索引
    int*                    R_out_f;
    int*                    M_out_f;

    int                     bvh_num;
    double                  radius;
    double                  radius2;
    int                     min_pts
    int                     dim;
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