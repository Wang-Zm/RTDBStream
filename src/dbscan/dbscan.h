#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

#define DEBUG_INFO 0
#define THREAD_NUM 80

#define OPTIMIZATION_LEVEL 1 // 2 无用，early cluster 无效果

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
    DATA_TYPE_3*            window;
    DATA_TYPE_3*            out;
    DATA_TYPE_3*            out_stride;
    int*                    label; // 0(core), 1(border), 2(noise)
    int*                    cluster_id;
    int*                    nn; // number of neighbors
    int*                    check_label;
    int*                    check_cluster_id;
    int*                    check_nn;
    int                     operation;
    int                     window_size;
    int                     stride_left;
    int                     stride_right;
    int                     window_id;

    DATA_TYPE*              min_value;
    int*                    cell_count;
    DATA_TYPE               cell_length;
    int*                    point_cell_id;
    int*                    center_idx_in_window;

    DATA_TYPE               radius;
    DATA_TYPE               radius2;
    DATA_TYPE               radius_one_half2;
    int                     min_pts;
    unsigned*               intersection_test_num;
    unsigned*               hit_num;

    DATA_TYPE_3*            centers;
    DATA_TYPE*              radii;
    int                     center_num;
    int**                   cell_points; // 仅存放第一层即可
    int*                    cell_point_num;
    int*                    points_in_dense_cells;
    int*                    pos_arr;

    unsigned*               ray_primitive_hits;
    unsigned*               ray_intersections;
    unsigned*               cluster_ray_intersections;
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
extern "C" void kGenAABB_by_center(DATA_TYPE_3* points, DATA_TYPE* width, unsigned numPrims, OptixAabb* d_aabb);
extern "C" void find_cores(int* label, int* nn, int* cluster_id, int window_size, int min_pts);
// extern "C" void union_cluster(int* tmp_cluster_id, int* cluster_id, int* label, int window_size);
extern "C" void find_neighbors(int* nn, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2, int min_pts);
extern "C" void set_cluster_id(int* nn, int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2);
#endif