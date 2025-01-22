#ifndef OPTIXSCAN_H
#define OPTIXSCAN_H

#include <cuda_runtime.h>

#define DEBUG_INFO 0
#define OPTIMIZATION_LEVEL 9

// typedef double DATA_TYPE;
// typedef double3 DATA_TYPE_3;
typedef float DATA_TYPE;
typedef float3 DATA_TYPE_3;
typedef long CELL_ID_TYPE;

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
    CELL_ID_TYPE*           point_cell_id;
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
    int                     sparse_num;
    int                     dense_num;
    int**                   cell_points;
    int*                    cell_point_num;
    int*                    points_in_dense_cells;
    int*                    pos_arr;
    // bool*                   point_status;
    int*                    tmp_pos_arr;
    int*                    new_pos_arr;
    int*                    offsets;
    int*                    num_offsets;
    int*                    num_points_in_dense_cells;
    int*                    num_dense_cells;
    int*                    sparse_offset;
    int*                    dense_offset;

    unsigned*               ray_primitive_hits;
    unsigned*               ray_intersections;
    unsigned*               ray_primitive_hits_cluster;
    unsigned*               ray_intersections_cluster;
    unsigned*               cluster_ray_intersections;
    unsigned*               num_dist_calculations;

    // DATA_TYPE_3*            c_centers;
    // DATA_TYPE*              c_radii;
    // int*                    c_cluster_id;
    // int**                   c_cell_points;
    // int*                    c_center_idx_in_window;

    int*                    d_neighbor_cells_list;
    int*                    d_neighbor_cells_capacity;
    int*                    d_neighbor_cells_pos;
    int*                    d_neighbor_cells_num;

    int*                    point_status;
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

extern "C" void kGenAABB(DATA_TYPE_3 *points, DATA_TYPE radius, unsigned numPrims, OptixAabb *d_aabb, cudaStream_t stream);
extern "C" void genAABB_hybrid_width(DATA_TYPE_3* points, DATA_TYPE width1, DATA_TYPE width2, unsigned N1, unsigned N2, OptixAabb* d_aabb, cudaStream_t stream);
extern "C" void kGenAABB_by_center(DATA_TYPE_3* points, DATA_TYPE* width, unsigned numPrims, OptixAabb* d_aabb, cudaStream_t stream);
extern "C" void find_cores(int* label, int* nn, int* cluster_id, int window_size, int min_pts, cudaStream_t stream);
// extern "C" void union_cluster(int* tmp_cluster_id, int* cluster_id, int* label, int window_size);
extern "C" void find_neighbors(int* nn, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2, int min_pts);
extern "C" void set_cluster_id(int* nn, int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2);
extern "C" void set_centers_radii(DATA_TYPE_3* window, DATA_TYPE radius, int* pos_arr, int* uniq_pos_arr, int* num_points, int min_pts, DATA_TYPE* min_value, DATA_TYPE cell_length, int num_centers,
								  DATA_TYPE_3* centers, DATA_TYPE* radii, int* cluster_id, int** cell_points, int* center_idx_in_window,
                                  cudaStream_t stream);
extern "C" void cluster_dense_cells(int* pos_arr,
									DATA_TYPE_3* window,
									int window_size,
									DATA_TYPE radius2,
									int* cluster_id,
									int* d_neighbor_cells_pos,
									int* d_neighbor_cells_num,
									int* d_neighbor_cells_list,
									int* d_neighbor_cells_capacity,
									cudaStream_t stream);
extern "C" void post_cluster(int* label, int* cluster_id, int window_size, cudaStream_t stream);
extern "C" void compute_offsets_of_cells(
	int window_size,
	int* pos_arr,
	CELL_ID_TYPE* point_cell_idx,
	int* offsets,
	int* num_offsets
);
extern "C" void count_points_in_dense_cells(const int* cell_offsets, int num_nonempty_cells, 
                                        	int min_pts, int* num_points_in_dense_cells,
                                            int* num_dense_cells);
extern "C" void set_hybrid_spheres_info(
	int num_cells,
	int min_pts,
	int* pos_arr,
	DATA_TYPE_3* window,
	int* sparse_offset,
	int* dense_offset,
	int* offsets,
	int* mixed_pos_arr,
	DATA_TYPE_3* centers,
	int* cid,
	int** points_in_dense_cells,
	int* num_points_in_dense_cell,
	DATA_TYPE* min_value,
	DATA_TYPE cell_length
);
void set_spheres_info_from_sparse_points(
	int num_cells, int min_pts, int* pos_arr, DATA_TYPE_3* window, int* sparse_offset, 
	int* offsets, DATA_TYPE_3* centers, int** points_in_dense_cells
);
void set_label(int** cell_points, int* nn, int min_pts, int* label, int num_points);
void compute_cell_id(DATA_TYPE_3* points, CELL_ID_TYPE* point_cell_id, int n,
					 DATA_TYPE* min_value, int* cell_count, DATA_TYPE cell_length);
#endif