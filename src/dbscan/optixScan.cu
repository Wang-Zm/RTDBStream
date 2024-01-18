#include <optix.h>
#include <sutil/vec_math.h>
#include "optixScan.h"

extern "C" {
__constant__ Params params;
}

extern "C" __global__ void __raygen__rg() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();    

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen 
    float3 ray_origin, ray_direction;
    ray_origin    = { float(params.out[idx.x].x), 
                      float(params.out[idx.x].y),
                      float(params.out[idx.x].z) };
    ray_direction = { 1, 0, 0 };

    // Trace the ray against our scene hierarchy
    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    unsigned int ray_id  = idx.x;
    optixTrace(
            params.handle,
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

extern "C" __global__ void __intersection__cube() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned ray_id  = optixGetPayload_2();
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif

    const DATA_TYPE_3 point    = params.window[primIdx];
    const DATA_TYPE_3 ray_orig = params.out[ray_id];
    DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
    DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
    if (sqdist < params.radius2) {
        if (params.operation == 0) {
            params.nn[primIdx]--;
        } else if (params.operation == 1) {
            params.nn[primIdx]++;
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

// extern "C" __global__ void __raygen__R_out() {
//     // Lookup our location within the launch grid
//     const uint3 idx = optixGetLaunchIndex();    
//     const uint3 dim = optixGetLaunchDimensions(); 

//     // Map our launch idx to a screen location and create a ray from the camera
//     // location through the screen 
//     float3 ray_origin, ray_direction;
//     ray_origin    = { float(params.pre_window[params.ex_cores_idx[idx.x]].x), 
//                       float(params.pre_window[params.ex_cores_idx[idx.x]].y),
//                       float(params.pre_window[params.ex_cores_idx[idx.x]].z) };
//     ray_direction = { 1, 0, 0 };

//     // Trace the ray against our scene hierarchy
//     unsigned int intersection_test_num = 0;
//     unsigned int hit_num = 0;
//     unsigned int ray_id  = idx.x;
//     optixTrace(
//             params.pre_handle,
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

// extern "C" __global__ void __intersection__R_out() {
//     unsigned primIdx = optixGetPrimitiveIndex();
//     unsigned ray_id  = optixGetPayload_2();
// #if DEBUG_INFO == 1
//     optixSetPayload_0(optixGetPayload_0() + 1);
// #endif

//     const DATA_TYPE_3 point    = params.window[primIdx];
//     const DATA_TYPE_3 ray_orig = params.pre_window[params.ex_cores_idx[ray_id]];
//     DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
//     DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
//     if (sqdist < params.radius2) {
//         // 判别 point 是否在 ex_cores_idx 中，如果不在其中，则设置为
//         // 若邻居是最佳的
//     }
// #if DEBUG_INFO == 1
//     optixSetPayload_1(optixGetPayload_1() + 1);
// #endif
// }

extern "C" __global__ void __raygen__cluster() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    if (params.label[idx.x] != 0) return;

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
    optixTrace(
            params.handle,
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
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif

    const DATA_TYPE_3 point    = params.window[primIdx];
    const DATA_TYPE_3 ray_orig = params.window[ray_id];
    DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
    DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
    if (sqdist < params.radius2) {
        if (params.label[primIdx] == 0 && ray_id < primIdx) {
            params.cluster_id[primIdx] = ray_id; // 最终指向小的 idx
        } else if (params.label[primIdx] != 0) {
            params.label[primIdx] = 1; // border
            params.cluster_id[primIdx] = ray_id;
        }
    }
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
}