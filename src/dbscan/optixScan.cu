#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();    
    const uint3 dim = optixGetLaunchDimensions(); 

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen 
    float3 ray_origin, ray_direction;
    ray_origin    = { float(params.queries[idx.x][idx.y * 3]), 
                      float(params.queries[idx.x][idx.y * 3 + 1]),
                      float(params.queries[idx.x][idx.y * 3 + 2]) };

    ray_direction = { 1, 0, 0 };

    // Trace the ray against our scene hierarchy
    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    unsigned int query_id = idx.x;
    unsigned int bvh_id  = idx.y;
    optixTrace(
            params.handle[idx.y],
            ray_origin,
            ray_direction,
            params.tmin,                   // Min intersection distance
            params.tmax,        // Max intersection distance
            0.0f,                   // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            intersection_test_num,
            hit_num,
            query_id,
            bvh_id
            );
#if DEBUG_INFO == 1
    atomicAdd(&params.ray_primitive_hits[idx.x], hit_num);
    atomicAdd(&params.ray_intersections[idx.x], intersection_test_num);
#endif
}

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __intersection__cube() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned bvh_id  = optixGetPayload_3();
    unsigned query_id= optixGetPayload_2();
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif
    if (params.dist[query_id][primIdx] >= params.radius2) return;

    const DATA_TYPE point[3] = {params.points[primIdx][bvh_id * 3], 
                             params.points[primIdx][bvh_id * 3 + 1], 
                             params.points[primIdx][bvh_id * 3 + 2]};
    const DATA_TYPE ray_orig[3] = {params.queries[query_id][bvh_id * 3],
                                params.queries[query_id][bvh_id * 3 + 1],
                                params.queries[query_id][bvh_id * 3 + 2]};
    DATA_TYPE O[] = { ray_orig[0] - point[0], ray_orig[1] - point[1], ray_orig[2] - point[2] };
    DIST_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
    atomicAdd(&params.dist[query_id][primIdx], sqdist); // 该操作耗时很长，无论是否是原子操作
    // params.dist[query_id][primIdx] += sqdist;
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
}

extern "C" __global__ void __intersection__flag() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned bvh_id  = optixGetPayload_3();
    unsigned query_id= optixGetPayload_2();
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif

    const DATA_TYPE point[3]    = { params.points[primIdx][bvh_id * 3], 
                                    params.points[primIdx][bvh_id * 3 + 1], 
                                    params.points[primIdx][bvh_id * 3 + 2] };
    const DATA_TYPE ray_orig[3] = { params.queries[query_id][bvh_id * 3],
                                    params.queries[query_id][bvh_id * 3 + 1],
                                    params.queries[query_id][bvh_id * 3 + 2] };
    DATA_TYPE O[3]   = { ray_orig[0] - point[0], ray_orig[1] - point[1], ray_orig[2] - point[2] };
    DIST_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
    if (sqdist < params.sub_radius2) {
        atomicAdd(&params.dist[query_id][primIdx], sqdist);
        long pos = long(query_id) * params.data_num * params.unsigned_len + primIdx * params.unsigned_len + bvh_id / 32;
        int bit_pos = bvh_id % 32;
        atomicOr(&params.dist_flag[pos], 1u << (31 - bit_pos));
    }
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
}

extern "C" __global__ void __anyhit__terminate_ray() {
    optixTerminateRay();
}
