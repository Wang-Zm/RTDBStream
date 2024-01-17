#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

__global__ void kGenAABB_t (
      DATA_TYPE** points,
      int bvh_id,
      double radius,
      unsigned int N,
      OptixAabb* aabb
) {
  unsigned int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (particleIndex >= N) return;

  double3 center = {points[particleIndex][3*bvh_id], points[particleIndex][3*bvh_id+1], points[particleIndex][3*bvh_id+2]};

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

extern "C" void kGenAABB(DATA_TYPE** points, int bvh_id, double width, unsigned int numPrims, OptixAabb* d_aabb) {
  unsigned int threadsPerBlock = 64;
  unsigned int numOfBlocks = numPrims / threadsPerBlock + 1;

  kGenAABB_t <<<numOfBlocks, threadsPerBlock>>> (
    points,
    bvh_id,
    width,
    numPrims,
    d_aabb
    );
}


__global__ void refine_t(DIST_TYPE** dist, 
                         unsigned* dist_flag, 
                         DATA_TYPE** points,
                         DATA_TYPE** queries,
                         int query_num, 
                         int data_num, 
                         int bvh_num,
                         double radius2) {
  int left = (data_num / blockDim.x) * threadIdx.x;
  int right = left + data_num / blockDim.x;
  if (threadIdx.x == blockDim.x - 1) right = data_num; // 最后一个线程多一些
  int unsigned_len = (bvh_num + 32 - 1) / 32;
  for (int i = left; i < right; i++) {
    // 判别dist是否是0，如果是0则跳过；否则计算bvh
    if (dist[blockIdx.x][i] <= 0) continue;
    
    long pos_base = long(blockIdx.x) * data_num * unsigned_len + i * unsigned_len;
    int pos, bit_pos;
    for (int bvh_id = 0; bvh_id < bvh_num; bvh_id++) {
      pos = pos_base + bvh_id / 32;
      bit_pos = bvh_id % 32;
      // 判断是否set了flag(计算过距离)，若计算过则直接跳过
      if (dist_flag[pos] & (1 << (31 - bit_pos))) continue;

      // 未计算过距离则计算距离
      const DATA_TYPE point[3]    = { points[i][bvh_id * 3], 
                                      points[i][bvh_id * 3 + 1], 
                                      points[i][bvh_id * 3 + 2] };
      const DATA_TYPE query[3]    = { queries[blockIdx.x][bvh_id * 3],
                                      queries[blockIdx.x][bvh_id * 3 + 1],
                                      queries[blockIdx.x][bvh_id * 3 + 2] };
      const DATA_TYPE O[3]        = { query[0] - point[0], query[1] - point[1], query[2] - point[2] };
      const DIST_TYPE sqdist      = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
      dist[blockIdx.x][i]        += sqdist;
      if (dist[blockIdx.x][i] >= radius2) break;
    }
  }
}

// https://forums.developer.nvidia.com/t/maximum-number-of-threads-on-thread-block/46392 关于grid和block的设置
extern "C" void refine_with_cuda(DIST_TYPE** dist, 
                                 unsigned* dist_flag,
                                 DATA_TYPE** points,
                                 DATA_TYPE** queries, 
                                 int query_num, 
                                 int data_num, 
                                 int bvh_num,
                                 double radius2) {
  // query, data, bvh
  unsigned threadsPerBlock = 1024;
  unsigned numOfBlocks = query_num;

  // data_num / threadsPerBlock 是一个thread处理的线程数
  refine_t <<<numOfBlocks, threadsPerBlock>>> (
    dist,
    dist_flag,
    points,
    queries,
    query_num,
    data_num,
    bvh_num,
    radius2
  );
}

__global__ void search_t(DIST_TYPE** dist, 
                         DATA_TYPE** points,
                         DATA_TYPE** queries,
                         int query_num, 
                         int data_num, 
                         int bvh_num,
                         double radius2) {
  int left = (data_num / blockDim.x) * threadIdx.x;
  int right = left + data_num / blockDim.x;
  if (threadIdx.x == blockDim.x - 1) right = data_num; // 最后一个线程多一些
  int unsigned_len = (bvh_num + 32 - 1) / 32;
  for (int i = left; i < right; i++) {
    for (int bvh_id = 0; bvh_id < bvh_num; bvh_id++) {
      const DATA_TYPE point[3]    = { points[i][bvh_id * 3], 
                                      points[i][bvh_id * 3 + 1], 
                                      points[i][bvh_id * 3 + 2] };
      const DATA_TYPE query[3]    = { queries[blockIdx.x][bvh_id * 3],
                                      queries[blockIdx.x][bvh_id * 3 + 1],
                                      queries[blockIdx.x][bvh_id * 3 + 2] };
      const DATA_TYPE O[3]        = { query[0] - point[0], query[1] - point[1], query[2] - point[2] };
      const DIST_TYPE sqdist      = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
      dist[blockIdx.x][i]        += sqdist;
      if (dist[blockIdx.x][i] >= radius2) break;
    }
  }
}

extern "C" void search_with_cuda(DIST_TYPE** dist, 
                                 DATA_TYPE** points,
                                 DATA_TYPE** queries, 
                                 int query_num, 
                                 int data_num, 
                                 int bvh_num,
                                 double radius2) {
  // query, data, bvh
  unsigned threadsPerBlock = 1024;
  unsigned numOfBlocks = query_num;

  // data_num / threadsPerBlock 是一个thread处理的线程数
  search_t <<<numOfBlocks, threadsPerBlock>>> (
    dist,
    points,
    queries,
    query_num,
    data_num,
    bvh_num,
    radius2
  );
}

__global__ void collect_t(int* label,
						  int* nn,
						  DATA_TYPE_3* window,
						  DATA_TYPE_3* out_stride,
						  int window_size,
						  int out_start,
						  int out_end,
						  DATA_TYPE_3* c_out,
						  DATA_TYPE_3* ex_cores,
						  DATA_TYPE_3* neo_cores,
						  int* c_out_num,
						  int* ex_cores_num,
						  int* neo_cores_num,
						  int min_pts) {
	int left = (window_size / blockDim.x) * threadIdx.x;
	int right = left + window_size / blockDim.x;
	if (threadIdx.x == blockDim.x - 1) right = window_size; // 最后一个线程少一些
	for (int i = left; i < right; i++) {
		if (i >= out_start && i < out_end) {
			if (label[i] == 0) {							// 原来是 core，现在 out
				int idx = atomicAdd(c_out_num, 1);
				c_out[idx] = out_stride[i - out_start]; 	// 记录 out 的部分
				idx = atomicAdd(ex_cores_num, 1);
				ex_cores[idx] = out_stride[i - out_start];
			}
			if (nn[i] > min_pts) {							// 现在是 core
				label[i] = 0;
				int idx = atomicAdd(neo_cores_num, 1);
				neo_cores[idx] = window[i];
			} else {
				label[i] = 2; 								// 现在不是 core，可暂时初始化为 noise
			}
		} else {
			if (nn[i] > min_pts && label[i] != 0) {			// 原来不是现在是
				int idx = atomicAdd(neo_cores_num, 1);
				neo_cores[idx] = window[i];
				label[i] = 0;
			} else if (nn[i] <= min_pts && label[i] == 0) { // 原来是现在不是
				int idx = atomicAdd(c_out_num, 1);
				c_out[idx] = window[i];
				label[i] = 2;								// 将 Wcurr 中 ex-core label 初始化为 noise
			}
		}
	}
}

/**
 * 1.收集 c_out, ex_cores, neo_cores
 * 2.label 设置
*/
extern "C" void set_label_collect_cores(int* label, 
										int* nn,
										DATA_TYPE_3* window,
										DATA_TYPE_3* out_stride,
										int window_size,
										int out_start,
										int out_end,
										DATA_TYPE_3* c_out,
                                        DATA_TYPE_3* ex_cores,
										DATA_TYPE_3* neo_cores,
										int* c_out_num,
										int* ex_cores_num,
										int* neo_cores_num,
										int min_pts) {
  unsigned threadsPerBlock = 64;
  unsigned numOfBlocks = (window_size + threadsPerBlock - 1) / threadsPerBlock;

  collect_t <<<numOfBlocks, threadsPerBlock>>> (
	label,
	nn,	
	window,
	out_stride,
	window_size,
	out_start,
	out_end,
	c_out,
	ex_cores,
	neo_cores,
	c_out_num,
	ex_cores_num,
	neo_cores_num,
	min_pts
  );
}