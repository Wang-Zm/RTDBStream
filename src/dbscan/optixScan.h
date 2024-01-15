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
#else
typedef double DATA_TYPE;
#endif

#define RT_FILTER 1 // 0时有正确结果，1时出现错误结果

struct Params {
    DATA_TYPE**             points;
    DATA_TYPE**             queries;    
    OptixTraversableHandle* handle;
    
    float                   tmin;
    float                   tmax;
    int                     data_num;
    int                     bvh_num;
    DIST_TYPE**             dist;
    unsigned*               dist_flag;
    double                  radius;
    double                  radius2;
    DATA_TYPE               sub_radius2;
    int                     dim;
    int                     unsigned_len;
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