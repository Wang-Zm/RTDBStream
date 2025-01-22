#include <optix.h>
#include <sutil/vec_math.h>
#include "dbscan.h"

extern "C" {
__constant__ Params params;
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

static __forceinline__ __device__ DATA_TYPE compute_dist(int ray_id, int primIdx, DATA_TYPE_3* ray_points, DATA_TYPE_3* prim_points) {
    const DATA_TYPE_3 ray_orig = ray_points[ray_id];
    const DATA_TYPE_3 point    = prim_points[primIdx];
    DATA_TYPE_3 O = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
    DATA_TYPE sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
    // DATA_TYPE sqdist = (ray_orig.x - point.x) * (ray_orig.x - point.x);
    return sqdist;
}

extern "C" __global__ void __raygen__naive() {
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
    unsigned int op      = 0;
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
            ray_id,
            op
            );
#if DEBUG_INFO == 1
    atomicAdd(&params.ray_primitive_hits[idx.x], hit_num);
    atomicAdd(&params.ray_intersections[idx.x], intersection_test_num);
#endif
}

extern "C" __global__ void __raygen__identify_cores() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();    
    if (idx.y == 0 && idx.x >= params.stride_left && idx.x < params.stride_right) return;

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
    unsigned int op      = idx.y;
    if (op == 0) {
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
                ray_id,
                op
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
                ray_id,
                op
                );
    }
#if DEBUG_INFO == 1
    atomicAdd(&params.ray_primitive_hits[idx.x], hit_num);
    atomicAdd(&params.ray_intersections[idx.x], intersection_test_num);
#endif
}

extern "C" __global__ void __raygen__grid() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    // int pos = params.pos_arr[idx.x];    

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
    unsigned int op      = 0; // 占空
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            params.tmin,                   // Min intersection distance
            params.tmax,                   // Max intersection distance
            0.0f,                          // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ),    // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                             // SBT offset   -- See SBT discussion
            1,                             // SBT stride   -- See SBT discussion
            0,                             // missSBTIndex -- See SBT discussion
            intersection_test_num,
            hit_num,
            ray_id,
            op
            );
#if DEBUG_INFO == 1
    atomicAdd(&params.ray_primitive_hits[idx.x], hit_num);
    atomicAdd(&params.ray_intersections[idx.x], intersection_test_num);
#endif
}

extern "C" __global__ void __intersection__naive() {
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
            atomicSub(params.nn + primIdx, 1);
        } else if (params.operation == 1) {
            if (primIdx >= params.stride_left && primIdx < params.stride_right) {
                atomicAdd(params.nn + primIdx, 1);
            } else {
                atomicAdd(params.nn + primIdx, 1);
                atomicAdd(params.nn + params.stride_left + ray_id, 1);
            }
        }
    }
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
}

extern "C" __global__ void __intersection__identify_cores() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned ray_id  = optixGetPayload_2();
    unsigned op      = optixGetPayload_3();
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif
    const DATA_TYPE_3 point = op == 0 ? params.out_stride[primIdx] : params.out[primIdx];
    const DATA_TYPE_3 ray_orig = params.window[ray_id];
    DATA_TYPE O[] = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
    DATA_TYPE sqdist = O[0] * O[0] + O[1] * O[1] + O[2] * O[2];
    if (sqdist < params.radius2) {
        if (op == 0) {
            atomicSub(params.nn + ray_id, 1);
        } else if (op == 1) {
            atomicAdd(params.nn + ray_id, 1);
            if (ray_id < params.stride_left || ray_id >= params.stride_right) {
                atomicAdd(params.nn + params.stride_left + primIdx, 1);
            }
        }
    }
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
}

extern "C" __global__ void __intersection__grid() {
    unsigned primIdx = optixGetPrimitiveIndex();
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif
#if OPTIMIZATION_LEVEL == 9
    if (primIdx >= params.sparse_num)
        return;
#endif
    if (params.nn[primIdx] >= params.min_pts)
        return;

    unsigned ray_id  = optixGetPayload_2();
    DATA_TYPE sqdist = compute_dist(ray_id, primIdx, params.window, params.centers);
    if (sqdist < params.radius2) {
        atomicAdd(params.nn + primIdx, 1);
    }
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
}

// extern "C" __global__ void __anyhit__terminate_ray() {
//     optixTerminateRay();
// }

extern "C" __global__ void __miss__ms() {
}

extern "C" __global__ void __raygen__cluster() {
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    if (params.label[idx.x] != 0) return;

    // Map our launch idx to a screen location and create a ray from the camera location through the screen 
    // float3 ray_origin, ray_direction;
    // ray_origin    = { float(params.window[idx.x].x), 
    //                   float(params.window[idx.x].y),
    //                   float(params.window[idx.x].z) };
    float3& ray_origin   = params.window[idx.x];
    float3 ray_direction = { 1, 0, 0 };

    // Trace the ray against our scene hierarchy
    unsigned int intersection_test_num = 0;
    unsigned int hit_num = 0;
    unsigned int ray_id  = idx.x;
    // unsigned int op      = params.point_cell_id[ray_id];
    unsigned int op      = 0;
    // unsigned int op      = params.point_status[ray_id];
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            params.tmin,        // Min intersection distance
            params.tmax,        // Max intersection distance
            0.0f,               // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                  // SBT offset   -- See SBT discussion
            1,                  // SBT stride   -- See SBT discussion
            0,                  // missSBTIndex -- See SBT discussion
            intersection_test_num,
            hit_num,
            ray_id,
            op
            );
#if DEBUG_INFO == 1
    atomicAdd(&params.ray_primitive_hits_cluster[idx.x], hit_num);
    atomicAdd(&params.ray_intersections_cluster[idx.x], intersection_test_num);
#endif
    // atomicAdd(params.cluster_ray_intersections, intersection_test_num);
}

extern "C" __global__ void __intersection__cluster() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned ray_id  = optixGetPayload_2();
    if (params.label[primIdx] == 0 && primIdx > ray_id) return;
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif
    int ray_rep = find_repres(ray_id, params.cluster_id);
    int prim_rep = find_repres(primIdx, params.cluster_id);
    if (ray_rep == prim_rep) return;
#if DEBUG_INFO == 1
    optixSetPayload_1(optixGetPayload_1() + 1);
#endif
    const DATA_TYPE_3 ray_orig = params.window[ray_id];
    const DATA_TYPE_3 point    = params.window[primIdx];
    DATA_TYPE_3 O = { ray_orig.x - point.x, ray_orig.y - point.y, ray_orig.z - point.z };
    DATA_TYPE sqdist = O.x * O.x + O.y * O.y + O.z * O.z;
#if DEBUG_INFO == 1
    atomicAdd(params.num_dist_calculations, 1);
#endif
    if (sqdist >= params.radius2) return;
    if (params.label[primIdx] == 0) {
        unite(ray_id, primIdx, params.cluster_id);
    } else {
        if (params.cluster_id[primIdx] == primIdx) {
            atomicCAS(params.cluster_id + primIdx, primIdx, ray_id);
            params.label[primIdx] = 1; // set to border
        }
    }
}

extern "C" __global__ void __intersection__hybrid_radius_sphere() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned ray_id  = optixGetPayload_2();
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif
    int prim_idx_in_window = *params.cell_points[primIdx];
    if (find_repres(ray_id, params.cluster_id) == find_repres(prim_idx_in_window, params.cluster_id)) 
        return;

    if (primIdx < params.sparse_num) {
        DATA_TYPE sqdist = compute_dist(ray_id, primIdx, params.window, params.centers);
#if DEBUG_INFO == 1
        atomicAdd(params.num_dist_calculations, 1);
#endif
        if (sqdist >= params.radius2) return;
        if (params.label[prim_idx_in_window] == 0) {
            unite(ray_id, prim_idx_in_window, params.cluster_id);
        } else {
            if (params.cluster_id[prim_idx_in_window] == prim_idx_in_window) {
                params.cluster_id[prim_idx_in_window] = ray_id;
                params.label[prim_idx_in_window] = 1;
            }
        }
    } else { // 1.5Eps-radius sphere
        int *points_in_cell = params.cell_points[primIdx];
        int num_points = params.cell_point_num[primIdx];
        for (int i = 0; i < num_points; i++) {
            // if (find_repres(ray_id, params.cluster_id) == find_repres(points_in_cell[i], params.cluster_id))
            //     break;
            DATA_TYPE dist = compute_dist(ray_id, points_in_cell[i], params.window, params.window);
#if DEBUG_INFO == 1
            atomicAdd(params.num_dist_calculations, 1);
#endif
            if (dist < params.radius2) {
                unite(ray_id, points_in_cell[i], params.cluster_id);
                break;
            }
        }
    }
}

extern "C" __global__ void __intersection__cluster_bvh_from_sparse_points() {
    unsigned primIdx = optixGetPrimitiveIndex();
    unsigned ray_id  = optixGetPayload_2();
    int prim_idx_in_window = params.center_idx_in_window[primIdx];
    
    int ray_rep = find_repres(ray_id, params.cluster_id);
    int prim_rep = find_repres(prim_idx_in_window, params.cluster_id);
    if (ray_rep == prim_rep) return;
#if DEBUG_INFO == 1
    optixSetPayload_0(optixGetPayload_0() + 1);
#endif
    DATA_TYPE sqdist = compute_dist(ray_id, primIdx, params.window, params.centers);
    if (sqdist >= params.radius2) return;
    if (params.label[prim_idx_in_window] == 0) {
        unite(ray_id, prim_idx_in_window, params.cluster_id);
    } else {
        if (params.cluster_id[prim_idx_in_window] == prim_idx_in_window) {
            atomicCAS(params.cluster_id + prim_idx_in_window, prim_idx_in_window, ray_id);
        }
        params.label[prim_idx_in_window] = 1;
    }
}