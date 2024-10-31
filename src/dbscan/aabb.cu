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

  double3 center = {points[particleIndex].x, points[particleIndex].y, points[particleIndex].z};

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

extern "C" void kGenAABB(DATA_TYPE_3* points, DATA_TYPE width, unsigned numPrims, OptixAabb* d_aabb) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  kGenAABB_t <<<numOfBlocks, threadsPerBlock>>> (
    points,
    width,
    numPrims,
    d_aabb
    );
}

__global__ void kGenAABB_by_center_t (DATA_TYPE_3* points, DATA_TYPE* radius, unsigned int N, OptixAabb* aabb) {
	unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (particleIndex >= N) return;
	double3 center = {points[particleIndex].x, points[particleIndex].y, points[particleIndex].z};
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

extern "C" void kGenAABB_by_center(DATA_TYPE_3* points, DATA_TYPE* width, unsigned numPrims, OptixAabb* d_aabb) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  kGenAABB_by_center_t <<<numOfBlocks, threadsPerBlock>>> (
    points,
    width,
    numPrims,
    d_aabb
    );
}

/**
 * 1.收集 c_out, ex_cores, neo_cores
 * 2.label 设置
*/
// extern "C" void set_label_collect_cores(int* label, 
// 										int* nn,
// 										DATA_TYPE_3* window,
// 										DATA_TYPE_3* out_stride,
// 										int window_size,
// 										int out_start,
// 										int out_end,
// 										DATA_TYPE_3* c_out,
//                     DATA_TYPE_3* ex_cores,
// 										DATA_TYPE_3* neo_cores,
// 										int* c_out_num,
// 										int* ex_cores_num,
// 										int* neo_cores_num,
// 										int min_pts) {
//   unsigned threadsPerBlock = 64;
//   unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;

//   collect_t <<<numOfBlocks, threadsPerBlock>>> (
// 	label,
// 	nn,	
// 	window,
// 	out_stride,
// 	window_size,
// 	out_start,
// 	out_end,
// 	c_out,
// 	ex_cores,
// 	neo_cores,
// 	c_out_num,
// 	ex_cores_num,
// 	neo_cores_num,
// 	min_pts
//   );
// }

__global__ void find_cores_t(int* label, int* nn, int* cluster_id, int window_size, int min_pts) {
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  	if (idx >= window_size) return;
	if (nn[idx] >= min_pts) {
		label[idx] = 0;
	} else {
		label[idx] = 2;  	// 初始化为 noise
	}
	cluster_id[idx] = idx; 	// 初始化 cluster
}

extern "C" void find_cores(int* label, int* nn, int* cluster_id, int window_size, int min_pts) {
	unsigned threadsPerBlock = 64;
	unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;
	find_cores_t <<<numOfBlocks, threadsPerBlock>>> (
		label,
		nn,
		cluster_id,
		window_size,
		min_pts
	);
}

// __global__ void union_t(int* tmp_cluster_id, int* cluster_id, int* label, int window_size) {
// 	int left = (window_size / blockDim.x) * threadIdx.x;
// 	int right = left + window_size / blockDim.x;
// 	if (threadIdx.x == blockDim.x - 1) right = window_size;
// 	for (int i = left; i < right; i++) {
// 		if (label[i] == 2) {
// 			cluster_id[i] = -1; // noise
// 			continue;
// 		}
// 		int p = tmp_cluster_id[i];
// 		while (p != tmp_cluster_id[p]) { // 不带路径压缩的 union
// 			p = tmp_cluster_id[p];
// 		}
// 		cluster_id[i] = p;
// 	}
// }

// extern "C" void union_cluster(int* tmp_cluster_id, int* cluster_id, int* label, int window_size) {
//   unsigned threadsPerBlock = 64;
//   unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;
//   union_t <<<numOfBlocks, threadsPerBlock>>> (
// 	tmp_cluster_id,
//     cluster_id,
//     label,
//     window_size
//   );                        
// }

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

// __global__ void set_cluster_id_op_t(int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2, int operation) {
// 	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
//   	if (i >= window_size) return;
// 	DATA_TYPE_3 p = window[i];
//  	if (operation != 2) {
// 		if (label[i] == 0) { // 只和前面的 core 进行比较
// 			for (int j = 0; j < i; j++) {
// 				if (label[j] == 0) {
// 					DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
// 					DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
// 					if (d < radius2) {
// 						if (i == 2747) {
// 							printf("set_cluster_id_op_t, i=%d, j=%d, d=%lf, window[%d]={%lf, %lf, %lf}, window[%d]={%lf, %lf, %lf}\n", i, j, d, j, window[j].x, window[j].y, window[j].z, i, window[i].x, window[i].y, window[i].z);
// 						}
// 						cluster_id[i] = j;
// 						break;
// 					}
// 				}
// 			}
// 		} else {			// border，设置 cid 为找到的第一个 core
// 			for (int j = 0; j < window_size; j++) {
// 				if (j == i) continue;
// 				if (label[j] == 0) {
// 					DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
// 					DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
// 					if (d < radius2) {
// 						cluster_id[i] = j;
// 						label[i] = 1;
// 						break;
// 					}
// 				}
// 			}
// 		}
// 	} else {
// 		if (label[i] == 0 && cluster_id[i] == i) { // 只和后面的 core 进行比较
// 			for (int j = i + 1; j < window_size; j++) {
// 				if (label[j] == 0 && i > cluster_id[j]) {
// 					DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
// 					DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
// 					if (d < radius2) {
// 						cluster_id[i] = cluster_id[j];
// 						printf("i=%d, params.check_cluster_id[%d]=%d\n", i, j, cluster_id[j]);
// 						break;
// 					}
// 				}
// 			}
// 		}
// 	}
// }

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

__global__ void set_cluster_id_op_t(int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  	if (i >= window_size) return;
	
	// if (label[i] != 0) return; // 从 core 开始计算
	// DATA_TYPE_3 p = window[i];
	// for (int j = 0; j < window_size; j++) { // 只和 i 前面的点进行测试
	// 	if (label[j] == 0 && j > i) return;
		
	// 	DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
	// 	DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
	// 	if (d >= radius2) continue;
	// 	if (label[j] == 0) {
	// 		int ray_rep = find_repres(i, cluster_id);
	// 		int prim_rep = find_repres(j, cluster_id);
	// 		bool repeat;
	// 		do { // 设置 core
	// 			repeat = false;
	// 			if (ray_rep != prim_rep) {
	// 				int ret;
	// 				if (ray_rep < prim_rep) {
	// 					if ((ret = atomicCAS(cluster_id + prim_rep, prim_rep, ray_rep)) != prim_rep) {
	// 						prim_rep = ret;
	// 						repeat = true;
	// 					}
	// 				} else {
	// 					if ((ret = atomicCAS(cluster_id + ray_rep, ray_rep, prim_rep)) != ray_rep) {
	// 						ray_rep = ret;
	// 						repeat = true;
	// 					}
	// 				}
	// 			}
	// 		} while (repeat);
	// 	} else { // border 处暂直接设置 direct parent 即可
	// 		if (cluster_id[j] == j) {
	// 			atomicCAS(cluster_id + j, j, i);
	// 		}
	// 		// 1) 若对应点的 cid 不是自己，说明已经设置，直接跳过 2) 若对应点的 cid 是自己，说明未设置，此时开始设置；设置过程中可能有其余的线程也在设置，这样可能连续设置两次，但是不会出现问题问题，就是多设置几次[暂时使用这种策略]
	// 		label[j] = 1; // 设置为 border
	// 	}
	// }

	DATA_TYPE_3 p = window[i];
	if (label[i] == 0) { // 只和前面的 core 进行比较
		for (int j = 0; j < i; j++) {
			if (label[j] != 0) continue;
			DATA_TYPE_3 O = {p.x - window[j].x, p.y - window[j].y, p.z - window[j].z};
			DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
			if (d >= radius2) continue;
			int ray_rep = find_repres(i, cluster_id);
			int prim_rep = find_repres(j, cluster_id);
			bool repeat;
			do { // 设置 core
				repeat = false;
				if (ray_rep != prim_rep) {
					int ret;
					if (ray_rep < prim_rep) {
						if ((ret = atomicCAS(cluster_id + prim_rep, prim_rep, ray_rep)) != prim_rep) {
							prim_rep = ret;
							repeat = true;
						}
					} else {
						if ((ret = atomicCAS(cluster_id + ray_rep, ray_rep, prim_rep)) != ray_rep) {
							ray_rep = ret;
							repeat = true;
						}
					}
				}
			} while (repeat);
		}
	} else {			// border，设置 cid 为找到的第一个 core 的 repres
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

// __global__ void update_grid_t(DATA_TYPE_3* data, int data_num, DATA_TYPE cell_width, uint3 grid_d, KeyValue* pHashTable) {
// 	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
//   	if (i >= data_num) return;
// 	unsigned cell_idx_x = data[i].x / cell_width;
// 	unsigned cell_idx_y = data[i].y / cell_width;
// 	unsigned cell_idx_z = data[i].z / cell_width;
// 	unsigned cell_id = cell_idx_x * grid_d.y * grid_d.z + cell_idx_y * grid_d.z + cell_idx_z;
// 	// cell_id 对应的 value + 1
// 	// 如果 gpu 比较困难，那么找别的东西
// 	// 需求就是 cell_id 对应的 slot 中的 value 进行原子操作，因此只要找到一个索引即可
// }

// extern "C" void update_grid(DATA_TYPE_3* data, int data_num, DATA_TYPE cell_width, uint3 grid_d, KeyValue* pHashTable) { // grid_d.x|y|z，cell_width, 
// 	unsigned threadsPerBlock = 64;
// 	unsigned numOfBlocks = (data_num + threadsPerBlock - 1) / threadsPerBlock;
// 	update_grid_t <<<numOfBlocks, threadsPerBlock>>> (
// 		data,
// 		data_num,
// 		cell_width,
// 		grid_d,
// 		pHashTable
// 	);
// }

/**
 * 已有信息：
 * window 窗口中的点
 * radius 半径
 * pos_arr 原始的 pos_arr
 * uniq_pos_arr 合并大球后的每个球的中心在 pos_arr 的 offset
 * num_points 每个点对应的球中的点数，用来确认大球还是小球
 * min_pts 邻居阈值
 * min_val 数据空间范围最小值
 * cell_length cell 的边长
 * num_centers center 的个数
 * 设置信息：
 * centors 球的中心
 * radii 球的半径
 * cluster_id 窗口中每个点的 cluster id 的设置
 * cell_points 每个大球对应的点的起始位置
 * center_idx_in_window 每个 center 在窗口中的位置，便于光追过程中的聚类；这个似乎可以通过 uniq_pos_arr 获取到，其实就是 i
 * 阶段：
 * 1.只使用全局内存实现功能
 */
__global__ void set_centers_radii_t(DATA_TYPE_3* window, DATA_TYPE radius, int* pos_arr, int* uniq_pos_arr, int* num_points, int min_pts, DATA_TYPE* min_value, DATA_TYPE cell_length, int num_centers,
								    DATA_TYPE_3* centers, DATA_TYPE* radii, int* cluster_id, int** cell_points, int* center_idx_in_window) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_centers) {
		return;
	}
	int offset = uniq_pos_arr[idx];
	int i = pos_arr[offset]; // 在 window 中的位置
	if (num_points[idx] < min_pts) {
		centers[idx] = window[i];
		radii[idx] = radius;
		center_idx_in_window[idx] = i;
		cluster_id[i] = i;
	} else {
		int dim_id_x = (window[i].x - min_value[0]) / cell_length;
		int dim_id_y = (window[i].y - min_value[1]) / cell_length;
		int dim_id_z = (window[i].z - min_value[2]) / cell_length;
	    centers[idx] = { min_value[0] + (dim_id_x + 0.5) * cell_length, 
						 min_value[1] + (dim_id_y + 0.5) * cell_length, 
						 min_value[2] + (dim_id_z + 0.5) * cell_length };
		radii[idx] = 1.5 * radius;
		cell_points[idx] = pos_arr + offset;
		center_idx_in_window[idx] = i;
		for (int t = 0; t < num_points[idx]; t++) {
			cluster_id[pos_arr[offset + t]] = i;
		}
		// int t = 0;
		// int times = num_points[idx] / 4;
		// for (; t < times; t++) {
		// 	cluster_id[pos_arr[offset + 4 * t]] = i;
		// 	cluster_id[pos_arr[offset + 4 * t + 1]] = i;
		// 	cluster_id[pos_arr[offset + 4 * t + 2]] = i;
		// 	cluster_id[pos_arr[offset + 4 * t + 3]] = i;
		// }
		// while (t < num_points[idx]) {
		// 	cluster_id[pos_arr[offset + t]] = i;
		// 	t++;
		// }
	}
}

extern "C" void set_centers_radii(DATA_TYPE_3* window, DATA_TYPE radius, int* pos_arr, int* uniq_pos_arr, int* num_points, int min_pts, DATA_TYPE* min_value, DATA_TYPE cell_length, int num_centers,
								  DATA_TYPE_3* centers, DATA_TYPE* radii, int* cluster_id, int** cell_points, int* center_idx_in_window) {
	int block = 32;
	int grid = (num_centers + block - 1) / block;
	set_centers_radii_t <<<grid, block>>> (
		window, radius, pos_arr, uniq_pos_arr, num_points, min_pts, min_value, cell_length, num_centers,
		centers, radii, cluster_id, cell_points, center_idx_in_window
	);
}