#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

extern "C" {
__constant__ Params params;
}

// extern "C" __global__ void __raygen__rg() {
//     // Lookup our location within the launch grid
//     const uint3 idx = optixGetLaunchIndex();    

//     // Map our launch idx to a screen location and create a ray from the camera
//     // location through the screen 
//     float3 ray_origin, ray_direction;
//     ray_origin    = { float(params.out[idx.x].x), 
//                       float(params.out[idx.x].y),
//                       float(params.out[idx.x].z) };
//     ray_direction = { 1, 0, 0 };

//     // Trace the ray against our scene hierarchy
//     unsigned int intersection_test_num = 0;
//     unsigned int hit_num = 0;
//     unsigned int ray_id  = idx.x;
//     optixTrace(
//             params.handle,
//             ray_origin,
//             ray_direction,
//             params.tmin,                   // Min intersection distance
//             params.tmax,        // Max intersection distance
//             0.0f,                   // rayTime -- used for motion blur
//             OptixVisibilityMask( 255 ), // Specify always visible
//             OPTIX_RAY_FLAG_NONE,
//             0,                   // SBT offset   -- See SBT discussion
//             1,                   // SBT stride   -- See SBT discussion
//             0,                   // missSBTIndex -- See SBT discussion
//             intersection_test_num,
//             hit_num,
//             ray_id
//             );
// #if DEBUG_INFO == 1
//     atomicAdd(&params.ray_primitive_hits[idx.x], hit_num);
//     atomicAdd(&params.ray_intersections[idx.x], intersection_test_num);
// #endif
// }

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
            // params.nn[ray_id]--; // 只操作自己，无需原子操作
            atomicSub(params.nn + ray_id, 1);
        } else if (params.operation == 1) {
            // params.nn[ray_id]++; // 判别是否是 stride_left, stride_right 中间
            atomicAdd(params.nn + ray_id, 1);
            if (ray_id < params.stride_left || ray_id >= params.stride_right) {
                atomicAdd(params.nn + primIdx, 1); // stride 外部的点邻居增加后，stride 中的点的邻居也同步增加
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

extern "C" __global__ void __raygen__cluster() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    // if (params.label[idx.x] == 2) return; // TODO: 应该只从 core 或 border 发射光线，但是 border 目前无法识别，
    if (params.operation == 2) { // 第二次只从 cid[x] = x 的点发射光线
        if (params.label[idx.x] != 0 || params.cluster_id[idx.x] != idx.x) return;
    }

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
    if (params.label[primIdx] != 0) return;

    if (params.operation != 2) {
        if (params.label[ray_id] == 0) {
            if (primIdx >= ray_id) return; // 仅在 primIdx < core_id 时进行距离计算，减小距离计算次数
            #if DEBUG_INFO == 1
                optixSetPayload_0(optixGetPayload_0() + 1);
            #endif
            const DATA_TYPE_3 point    = params.window[primIdx];
            const DATA_TYPE_3 ray_orig = params.window[ray_id];
            DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
            DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
            if (sqdist < params.radius2) {
                #if DEBUG_INFO == 1
                    optixSetPayload_1(optixGetPayload_1() + 1);
                #endif
                if (ray_id == 2747) {
                    O[0] = params.window[ray_id].x - params.window[39].x;
                    O[1] = params.window[ray_id].y - params.window[39].y;
                    O[2] = params.window[ray_id].z - params.window[39].z;
                    sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
                    printf("__intersection__cluster, ray_id=2747, primIdx=%d, d=%lf, window[%d]={%lf, %lf, %lf}, window[%d]={%lf, %lf, %lf}\n", 
                            39, sqdist, 39, params.window[39].x, params.window[39].y, params.window[39].z, 
                            ray_id, params.window[ray_id].x, params.window[ray_id].y, params.window[ray_id].z);
                    printf("__intersection__cluster, ray_id=2747, primIdx=%d\n", primIdx);
                }
                params.cluster_id[ray_id] = primIdx;
                optixReportIntersection( 0, 0 );    // 直接停下光线
            }
        } else {
            #if DEBUG_INFO == 1
                optixSetPayload_0(optixGetPayload_0() + 1);
            #endif
            const DATA_TYPE_3 point    = params.window[primIdx];
            const DATA_TYPE_3 ray_orig = params.window[ray_id];
            DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
            DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
            if (sqdist < params.radius2) {
                #if DEBUG_INFO == 1
                    optixSetPayload_1(optixGetPayload_1() + 1);
                #endif
                params.cluster_id[ray_id] = primIdx;
                params.label[ray_id] = 1;        // set border
                optixReportIntersection( 0, 0 ); // 直接停下光线
            }
        }
    } else {
        if (primIdx <= ray_id || params.cluster_id[primIdx] >= ray_id) return; // 仅在 primIdx > core_id 时进行距离计算，减小距离计算次数
        #if DEBUG_INFO == 1
            optixSetPayload_0(optixGetPayload_0() + 1);
        #endif
        const DATA_TYPE_3 point    = params.window[primIdx];
        const DATA_TYPE_3 ray_orig = params.window[ray_id];
        DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
        DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
        if (sqdist < params.radius2 && ray_id > params.cluster_id[primIdx]) {
            #if DEBUG_INFO == 1
                optixSetPayload_1(optixGetPayload_1() + 1);
            #endif
            // if (ray_id == 2747) {
            //     O[0] = params.window[ray_id].x - params.window[39].x;
            //     O[1] = params.window[ray_id].y - params.window[39].y;
            //     O[1] = params.window[ray_id].z - params.window[39].z;
            //     printf("__intersection__cluster, ray_id=2747, primIdx=%d, d=%lf\n", 39, sqdist);
            // }
            printf("ray_id=%d, params.cluster_id[%d]=%d\n", ray_id, primIdx, params.cluster_id[primIdx]);
            params.cluster_id[ray_id] = params.cluster_id[primIdx];
            optixReportIntersection( 0, 0 );    // 直接停下光线
        }
    }
}