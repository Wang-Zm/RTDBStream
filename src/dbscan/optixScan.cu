#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();    
    if (params.operation == 0 && idx.x >= params.stride_left && idx.x < params.stride_right) return; //! 减少部分光线的发射

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen 
    float3 ray_origin, ray_direction;
    ray_origin    = { float(params.window[idx.x].x), 
                      float(params.window[idx.x].y),
                      float(params.window[idx.x].z) };
    ray_direction = { 1, 0, 0 };

    // Trace the ray against our scene hierarchy
    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    unsigned int ray_id  = idx.x;
    if (params.operation == 0) {
        optixTrace(
                params.out_stride_handle,
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
                ray_id
                );
    } else {
        optixTrace(
                params.in_stride_handle,
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
                ray_id
                );
    }
#if DEBUG_INFO == 1
    atomicAdd(&params.ray_primitive_hits[idx.x], hit_num);
    atomicAdd(&params.ray_intersections[idx.x], intersection_test_num);
#endif
}

extern "C" __global__ void __intersection__cube() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned ray_id  = optixGetPayload_2();
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif
    const DATA_TYPE_3 point    = params.out[primIdx];
    const DATA_TYPE_3 ray_orig = params.window[ray_id];
    DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
    DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
    if (sqdist < params.radius2) {
        if (params.operation == 0) {
            atomicSub(params.nn + ray_id, 1);
        } else if (params.operation == 1) {
            atomicAdd(params.nn + ray_id, 1);
            if (ray_id < params.stride_left || ray_id >= params.stride_right) {
                atomicAdd(params.nn + params.stride_left + primIdx, 1); // stride 外部的点邻居增加后，stride 中的点的邻居也同步增加
            }
        }
    }
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
}

extern "C" __global__ void __anyhit__terminate_ray() {
    optixTerminateRay();
}

extern "C" __global__ void __miss__ms() {
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

extern "C" __global__ void __raygen__cluster() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    if (params.label[idx.x] != 0) return; // * 从 core 发射光线 

    // Map our launch idx to a screen location and create a ray from the camera location through the screen 
    float3 ray_origin, ray_direction;
    ray_origin    = { float(params.window[idx.x].x), 
                      float(params.window[idx.x].y),
                      float(params.window[idx.x].z) };
    ray_direction = { 1, 0, 0 };

    // Trace the ray against our scene hierarchy
    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    unsigned int ray_id  = idx.x;
    optixTrace(
            params.handle,      // 应当是针对 cores 构建的树，此时为方便，可直接基于当前窗口构建树
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
            ray_id
            );
#if DEBUG_INFO == 1
    atomicAdd(&params.ray_primitive_hits[idx.x], hit_num);
    atomicAdd(&params.ray_intersections[idx.x], intersection_test_num);
#endif
}

extern "C" __global__ void __intersection__cluster() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned ray_id  = optixGetPayload_2();
    if (params.label[primIdx] == 0 && primIdx > ray_id) return; // 是 core 且 id 靠后，则直接退出 => 减少距离计算次数
    const DATA_TYPE_3 ray_orig = params.window[ray_id];
    const DATA_TYPE_3 point    = params.window[primIdx];
    DATA_TYPE_3 O = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
    DATA_TYPE sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
    if (sqdist >= params.radius2) return; // 距离过大
    if (params.label[primIdx] == 0) {
        int ray_rep = find_repres(ray_id, params.cluster_id); // TODO: 哪个已经知道？
        int prim_rep = find_repres(primIdx, params.cluster_id);
        bool repeat;
        do { // 设置
            repeat = false;
            if (ray_rep != prim_rep) {
                int ret;
                if (ray_rep < prim_rep) {
                    if ((ret = atomicCAS(params.cluster_id + prim_rep, prim_rep, ray_rep)) != prim_rep) {
                        prim_rep = ret;
                        repeat = true;
                    }
                } else {
                    if ((ret = atomicCAS(params.cluster_id + ray_rep, ray_rep, prim_rep)) != ray_rep) {
                        ray_rep = ret;
                        repeat = true;
                    }
                }
            }
        } while (repeat);
    } else { // border 处暂直接设置 direct parent 即可
        if (params.cluster_id[primIdx] == primIdx) {
            atomicCAS(params.cluster_id + primIdx, primIdx, ray_id);
        }
        // 1) 若对应点的 cid 不是自己，说明已经设置，直接跳过 2) 若对应点的 cid 是自己，说明未设置，此时开始设置；设置过程中可能有其余的线程也在设置，这样可能连续设置两次，但是不会出现问题问题，就是多设置几次[暂时使用这种策略]
        params.label[primIdx] = 1; // 设置为 border
    }
}