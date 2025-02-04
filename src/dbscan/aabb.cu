#include <optix.h>
#include <sutil/vec_math.h>
#include "dbscan.h"

__global__ void kGenAABB_t (
      DATA_TYPE_3* points,
      DATA_TYPE radius,
      unsigned int N,
      OptixAabb* aabb
) {
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= N) return;

	DATA_TYPE_3 center = {points[particleIndex].x, points[particleIndex].y, points[particleIndex].z};

	// float3 m_min = center - radius;
	// float3 m_max = center + radius;
	float3 m_min;
	float3 m_max;
	m_min.x = center.x - radius;
	m_min.y = center.y - radius;
	m_min.z = center.z - radius;
	m_max.x = center.x + radius;
	m_max.y = center.y + radius;
	m_max.z = center.z + radius;

	aabb[particleIndex] =
	{
		m_min.x, m_min.y, m_min.z,
		m_max.x, m_max.y, m_max.z
	};
}

extern "C" void kGenAABB(DATA_TYPE_3* points, DATA_TYPE width, unsigned numPrims, OptixAabb* d_aabb, cudaStream_t stream) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  kGenAABB_t <<<numOfBlocks, threadsPerBlock, 0, stream>>> (
    points,
    width,
    numPrims,
    d_aabb
    );
}

__global__ void genAABB_hybrid_width_t (DATA_TYPE_3* points, DATA_TYPE radius1, DATA_TYPE radius2, unsigned N1, unsigned N2, OptixAabb* aabb) {
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= N2) return;

	DATA_TYPE_3 center = {points[particleIndex].x, points[particleIndex].y, points[particleIndex].z};
	DATA_TYPE radius = particleIndex < N1 ? radius1 : radius2;
	float3 m_min;
	float3 m_max;
	m_min.x = center.x - radius;
	m_min.y = center.y - radius;
	m_min.z = center.z - radius;
	m_max.x = center.x + radius;
	m_max.y = center.y + radius;
	m_max.z = center.z + radius;

	aabb[particleIndex] =
	{
		m_min.x, m_min.y, m_min.z,
		m_max.x, m_max.y, m_max.z
	};
}

extern "C" void genAABB_hybrid_width(DATA_TYPE_3* points, DATA_TYPE width1, DATA_TYPE width2, unsigned N1, unsigned N2, OptixAabb* d_aabb, cudaStream_t stream) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = (N2 + threadsPerBlock - 1) / threadsPerBlock;

  genAABB_hybrid_width_t <<<numOfBlocks, threadsPerBlock, 0, stream>>> (
    points,
    width1,
	width2,
    N1,
	N2,
    d_aabb
    );
}

__global__ void kGenAABB_by_center_t (DATA_TYPE_3* points, DATA_TYPE* radius, unsigned int N, OptixAabb* aabb) {
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= N) return;
	DATA_TYPE_3 center = {points[particleIndex].x, points[particleIndex].y, points[particleIndex].z};
	float3 m_min;
	float3 m_max;
	m_min.x = center.x - radius[particleIndex];
	m_min.y = center.y - radius[particleIndex];
	m_min.z = center.z - radius[particleIndex];
	m_max.x = center.x + radius[particleIndex];
	m_max.y = center.y + radius[particleIndex];
	m_max.z = center.z + radius[particleIndex];

	aabb[particleIndex] =
	{
		m_min.x, m_min.y, m_min.z,
		m_max.x, m_max.y, m_max.z
	};
}

extern "C" void kGenAABB_by_center(DATA_TYPE_3* points, DATA_TYPE* width, unsigned numPrims, OptixAabb* d_aabb, cudaStream_t stream) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  kGenAABB_by_center_t <<<numOfBlocks, threadsPerBlock, 0, stream>>> (
    points,
    width,
    numPrims,
    d_aabb
    );
}

static __forceinline__ __device__ int find_repres(int v, int* cid) {
    int par = cid[v];
    if (par != v) {
        int next, prev = v;
        while (par > (next = cid[par])) {
            cid[prev] = next;
            prev = par;
            par = next;
        }
    }
    return par;
}

static __forceinline__ __device__ void unite(int ray_id, int primIdx, int* cid) {
    int ray_rep = find_repres(ray_id, cid);
    int prim_rep = find_repres(primIdx, cid);
    bool repeat;
    do { // set core
        repeat = false;
        if (ray_rep != prim_rep) {
            int ret;
            if (ray_rep < prim_rep) {
                if ((ret = atomicCAS(cid + prim_rep, prim_rep, ray_rep)) != prim_rep) {
                    prim_rep = ret;
                    repeat = true;
                }
            } else {
                if ((ret = atomicCAS(cid + ray_rep, ray_rep, prim_rep)) != ray_rep) {
                    ray_rep = ret;
                    repeat = true;
                }
            }
        }
    } while (repeat);
}

__global__ void find_cores_t(int* label, int* nn, int* cluster_id, int window_size, int min_pts) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx >= window_size) return;
	if (nn[idx] >= min_pts) {
		label[idx] = 0;
	} else {
		label[idx] = 2;  	// init noise
	}
	cluster_id[idx] = idx; 	// init cluster
}

extern "C" void find_cores(int* label, int* nn, int* cluster_id, int window_size, int min_pts, cudaStream_t stream) {
	unsigned threadsPerBlock = 64;
	unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;
	find_cores_t <<<numOfBlocks, threadsPerBlock, 0, stream>>> (
		label,
		nn,
		cluster_id,
		window_size,
		min_pts
	);
}

static __forceinline__ __device__ int find_final_repres(int v, int* cid) {
    int par = cid[v];
    while (par != cid[par]) {
		par = cid[par];
	}
    return par;
}

__global__ void post_cluster_t(int* label, int* cluster_id, int window_size) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx >= window_size) return;
	if (label[idx] == 2) return;
	cluster_id[idx] = find_final_repres(idx, cluster_id);
}

extern "C" void post_cluster(int* label, int* cluster_id, int window_size, cudaStream_t stream) {
	unsigned threadsPerBlock = 64;
	unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;
	post_cluster_t <<<numOfBlocks, threadsPerBlock, 0, stream>>> (
		label,
		cluster_id,
		window_size
	);
}

__global__ void find_neighbors_t(int* nn, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2, int min_pts) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx >= window_size) return;
	DATA_TYPE_3 p = window[idx];
	for (int j = 0; j < window_size; j++) {
		DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
		DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
		if (d < radius2) {
			atomicAdd(nn + idx, 1); 
		}
	}
}

extern "C" void find_neighbors(int* nn, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2, int min_pts) {
	unsigned threadsPerBlock = 64;
	unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;
	find_neighbors_t <<<numOfBlocks, threadsPerBlock>>> (
		nn,
		window,
		window_size,
		radius2,
		min_pts
	);  
}

__global__ void set_cluster_id_op_t(int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  	if (i >= window_size) return;
	DATA_TYPE_3 p = window[i];
	if (label[i] == 0) { // compare only with the previous core
		for (int j = 0; j < i; j++) {
			if (label[j] != 0) continue;
			DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
			DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
			if (d >= radius2) continue;
			unite(i, j, cluster_id);
		}
	} else {			// border，set cid as the first core's repres found
		for (int j = 0; j < window_size; j++) {
			if (j == i) continue;
			if (label[j] == 0) {
				DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
				DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
				if (d < radius2) {
					cluster_id[i] = find_repres(j, cluster_id);
					label[i] = 1;
					break;
				}
			}
		}
	}
}

extern "C" void set_cluster_id(int* nn, int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2) {
	unsigned threadsPerBlock = 64;
	unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;
	set_cluster_id_op_t <<<numOfBlocks, threadsPerBlock>>> (
		label,
		cluster_id,
		window,
		window_size,
		radius2
	);
}

__global__ void set_centers_radii_t(DATA_TYPE_3* window, DATA_TYPE radius, int* pos_arr, int* uniq_pos_arr, int* num_points, int min_pts, DATA_TYPE* min_value, DATA_TYPE cell_length, int num_centers,
								    DATA_TYPE_3* centers, DATA_TYPE* radii, int* cluster_id, int** cell_points, int* center_idx_in_window) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_centers) {
		return;
	}
	int offset = uniq_pos_arr[idx];
	int i = pos_arr[offset];
	if (num_points[idx] < min_pts) {
		centers[idx] = window[i];
		radii[idx] = radius;
		center_idx_in_window[idx] = i;
		cluster_id[i] = i;
	} else {
		int dim_id_x = (window[i].x - min_value[0]) / cell_length;
		int dim_id_y = (window[i].y - min_value[1]) / cell_length;
		int dim_id_z = (window[i].z - min_value[2]) / cell_length;
	    centers[idx] = { min_value[0] + (dim_id_x + 0.5f) * cell_length, 
						 min_value[1] + (dim_id_y + 0.5f) * cell_length, 
						 min_value[2] + (dim_id_z + 0.5f) * cell_length };
		radii[idx] = 1.5 * radius;
		cell_points[idx] = pos_arr + offset;
		center_idx_in_window[idx] = i;
		for (int t = 0; t < num_points[idx]; t++) {
			cluster_id[pos_arr[offset + t]] = i;
		}
	}
}

extern "C" void set_centers_radii(DATA_TYPE_3* window, DATA_TYPE radius, int* pos_arr, int* uniq_pos_arr, int* num_points, int min_pts, DATA_TYPE* min_value, DATA_TYPE cell_length, int num_centers,
								  DATA_TYPE_3* centers, DATA_TYPE* radii, int* cluster_id, int** cell_points, int* center_idx_in_window,
								  cudaStream_t stream) {
	int block = 32;
	int grid = (num_centers + block - 1) / block;
	set_centers_radii_t <<<grid, block, 0, stream>>> (
		window, radius, pos_arr, uniq_pos_arr, num_points, min_pts, min_value, cell_length, num_centers,
		centers, radii, cluster_id, cell_points, center_idx_in_window
	);
}

__global__ void cluster_dense_cells_t(int* pos_arr, 
									  DATA_TYPE_3* window,
									  int window_size,
									  DATA_TYPE radius2,
									  int* cluster_id,
									  int* d_neighbor_cells_pos,
									  int* d_neighbor_cells_num,
									  int* d_neighbor_cells_list,
									  int* d_neighbor_cells_capacity) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= window_size) {
		return;
	}
	int pos = pos_arr[idx];
	if (d_neighbor_cells_pos[pos] < 0) {
		return;
	}
	DATA_TYPE_3 p = window[pos];
	for (int i = 0; i < d_neighbor_cells_num[pos]; i++) {
		int nei_cell_pos = d_neighbor_cells_pos[pos] + i;
		int start_pos = d_neighbor_cells_list[nei_cell_pos];
		int point_num = d_neighbor_cells_capacity[nei_cell_pos];
		if (find_repres(pos, cluster_id) == find_repres(pos_arr[start_pos], cluster_id)) {
			continue;
		}
		for (int j = start_pos; j < start_pos + point_num; j++) {
			int &pos_arrj = pos_arr[j];
			DATA_TYPE_3 O = {p.x - window[pos_arrj].x, p.y - window[pos_arrj].y, p.z - window[pos_arrj].z};
			DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
			if (d < radius2) {
				unite(pos, pos_arrj, cluster_id);
				break;
			}
		}
	}
}

extern "C" void cluster_dense_cells(int* pos_arr,
									DATA_TYPE_3* window,
									int window_size,
									DATA_TYPE radius2,
									int* cluster_id,
									int* d_neighbor_cells_pos,
									int* d_neighbor_cells_num,
									int* d_neighbor_cells_list,
									int* d_neighbor_cells_capacity,
									cudaStream_t stream) {
	int block = 32;
	int grid = (window_size + block - 1) / block;
	cluster_dense_cells_t <<<grid, block, 0, stream>>> (
		pos_arr,
		window,
		window_size,
		radius2,
		cluster_id,
		d_neighbor_cells_pos,
		d_neighbor_cells_num,
		d_neighbor_cells_list,
		d_neighbor_cells_capacity
	);
}

__global__ void prepare_for_points_in_dense_cells_t(int* pos_arr, 
													DATA_TYPE_3* window,
													int window_size,
													DATA_TYPE radius2,
													int* cluster_id,
													int* d_neighbor_cells_pos,
													int* d_neighbor_cells_num,
													int* d_neighbor_cells_list,
													int* d_neighbor_cells_capacity) {
	
}

__global__ void compute_offsets_of_cells_kernel(
	int n,
	int* pos_arr,
	CELL_ID_TYPE* point_cell_idx,
	int* offsets,
	int* num_offsets
) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx <= n) {
        bool is_cell_first_index = (idx == 0 || point_cell_idx[pos_arr[idx]] != point_cell_idx[pos_arr[idx - 1]] || idx == n);
        if (is_cell_first_index) {
            int old_offset = atomicAdd(num_offsets, 1);
            offsets[old_offset] = idx;
        }
    }
}

extern "C" void compute_offsets_of_cells(
	int window_size,
	int* pos_arr,
	CELL_ID_TYPE* point_cell_idx,
	int* offsets,
	int* num_offsets
) {
	int block = 256;
	int grid = (window_size + 1 + block - 1) / block;
	compute_offsets_of_cells_kernel <<<grid, block>>>(
		window_size,
		pos_arr,
		point_cell_idx,
		offsets,
		num_offsets
	);
}

// CUDA 核函数：计算满足条件的点数量并进行归约
__global__ void count_points_in_dense_cells_kernel(const int* cell_offsets, int num_nonempty_cells, 
                                        		   int min_pts, int* num_points_in_dense_cells,
												   int* num_dense_cells) {
    // 使用共享内存存储线程块内的部分和
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int local_sum = 0;

    for (int i = global_tid; i < num_nonempty_cells; i += stride) {
        int num_points_in_cell = cell_offsets[i + 1] - cell_offsets[i];
        if (num_points_in_cell >= min_pts) {
            local_sum += num_points_in_cell;
			atomicAdd(num_dense_cells, 1);
        }
    }

    shared_data[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(num_points_in_dense_cells, shared_data[0]);
    }
}

extern "C" void count_points_in_dense_cells(const int* cell_offsets, int num_nonempty_cells, 
                                        	int min_pts, int* num_points_in_dense_cells, 
											int* num_dense_cells) {
	int block = 256;
	int grid = (num_nonempty_cells + block - 1) / block;
	count_points_in_dense_cells_kernel <<<grid, block, block * sizeof(int)>>>(
		cell_offsets,
		num_nonempty_cells,
		min_pts,
		num_points_in_dense_cells,
		num_dense_cells
	);
}

__global__ void set_hybrid_spheres_info_kernel(
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
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_cells)
		return;
	const int num_points = offsets[idx + 1] - offsets[idx];
	if (num_points < min_pts) {
		int offset = atomicAdd(sparse_offset, num_points);
		for (int i = offsets[idx]; i < offsets[idx + 1]; i++, offset++) {
			mixed_pos_arr[offset] = pos_arr[i];
			centers[offset] = window[pos_arr[i]];
			cid[pos_arr[i]] = pos_arr[i];
			points_in_dense_cells[offset] = pos_arr + i;
		}
	} else {
		int offset = atomicAdd(dense_offset, 1);
		mixed_pos_arr[offset] = pos_arr[offsets[idx]];
		DATA_TYPE_3& point = window[pos_arr[offsets[idx]]];
		int dim_id_x = (point.x - min_value[0]) / cell_length;
        int dim_id_y = (point.y - min_value[1]) / cell_length;
        int dim_id_z = (point.z - min_value[2]) / cell_length;
        DATA_TYPE_3 center = { min_value[0] + (dim_id_x + 0.5f) * cell_length, 
                               min_value[1] + (dim_id_y + 0.5f) * cell_length, 
                               min_value[2] + (dim_id_z + 0.5f) * cell_length };
        centers[offset] = center;
		const int repres = pos_arr[offsets[idx]];
		for (int i = offsets[idx]; i < offsets[idx + 1]; i++) {
			cid[pos_arr[i]] = repres;
		}
		points_in_dense_cells[offset] = pos_arr + offsets[idx];
		num_points_in_dense_cell[offset] = num_points;
	}
}

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
) {
	int block = 256;
	int grid = (num_cells + block - 1) / block;
	set_hybrid_spheres_info_kernel <<<grid, block>>>(
		num_cells,
		min_pts,
		pos_arr,
		window,
		sparse_offset,
		dense_offset,
		offsets,
		mixed_pos_arr,
		centers,
		cid,
		points_in_dense_cells,
		num_points_in_dense_cell,
		min_value,
		cell_length
	);
}

__global__ void set_spheres_info_from_sparse_points_kernel(
	int num_cells, int min_pts, int* pos_arr, DATA_TYPE_3* window, int* sparse_offset, 
	int* offsets, DATA_TYPE_3* centers, int** points_in_dense_cells
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_cells)
		return;
	const int num_points = offsets[idx + 1] - offsets[idx];
	if (num_points < min_pts) {
		int offset = atomicAdd(sparse_offset, num_points);
		for (int i = offsets[idx]; i < offsets[idx + 1]; i++, offset++) {
			centers[offset] = window[pos_arr[i]];
			points_in_dense_cells[offset] = pos_arr + i;
		}
	}
}

void set_spheres_info_from_sparse_points(
	int num_cells, int min_pts, int* pos_arr, DATA_TYPE_3* window, int* sparse_offset, 
	int* offsets, DATA_TYPE_3* centers, int** points_in_dense_cells
) {
	int block = 256;
	int grid = (num_cells + block - 1) / block;
	set_spheres_info_from_sparse_points_kernel <<<grid, block>>>(
		num_cells, min_pts, pos_arr, window, sparse_offset, 
		offsets, centers, points_in_dense_cells
	);
}

__global__ void set_label_kernel(int** cell_points, int* nn, int min_pts, int* label, int num_points) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_points)
		return;
	if (nn[idx] < min_pts)
		label[*cell_points[idx]] = 2;
}

void set_label(int** cell_points, int* nn, int min_pts, int* label, int num_points) {
	int block = 256;
	int grid = (num_points + block - 1) / block;
	set_label_kernel <<<grid, block>>>(cell_points, nn, min_pts, label, num_points);
}

static __forceinline__ __device__ CELL_ID_TYPE get_cell_id(DATA_TYPE_3 point, DATA_TYPE* min_value, int* cell_count, DATA_TYPE cell_length) {
    CELL_ID_TYPE dim_id_x = (point.x - min_value[0]) / cell_length;
    CELL_ID_TYPE dim_id_y = (point.y - min_value[1]) / cell_length;
    CELL_ID_TYPE dim_id_z = (point.z - min_value[2]) / cell_length;
    CELL_ID_TYPE id = dim_id_x * cell_count[1] * cell_count[2] + dim_id_y * cell_count[2] + dim_id_z;
    return id;
}

__global__ void compute_cell_id_kernel(DATA_TYPE_3* points, CELL_ID_TYPE* point_cell_id, int n,
									   DATA_TYPE* min_value, int* cell_count, DATA_TYPE cell_length) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= n)
		return;
	CELL_ID_TYPE cell_id = get_cell_id(points[idx], min_value, cell_count, cell_length);
    point_cell_id[idx] = cell_id;
}

void compute_cell_id(DATA_TYPE_3* points, CELL_ID_TYPE* point_cell_id, int n,
					 DATA_TYPE* min_value, int* cell_count, DATA_TYPE cell_length) {
	int block = 256;
	int grid = (n + block - 1) / block;
	compute_cell_id_kernel <<<grid, block>>>(points, point_cell_id, n, min_value, cell_count, cell_length);
}