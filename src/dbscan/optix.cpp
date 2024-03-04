#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <sutil/Camera.h>

#include <array>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <map>
#include <queue>
#include <thread>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include "state.h"
#include "timer.h"

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

void initialize_optix(ScanState &state) {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
}

// void make_gas(ScanState &state) {
//     OptixAccelBuildOptions accel_options = {};
//     accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
//     accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

//     OptixAabb *d_aabb;
//     CUDA_CHECK(cudaMalloc(&d_aabb, state.window_size * sizeof(OptixAabb)));
//     kGenAABB(state.params.window, state.radius, state.window_size, d_aabb);
//     state.d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabb);

//     OptixBuildInput &vertex_input = state.vertex_input;
//     vertex_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
//     vertex_input.customPrimitiveArray.aabbBuffers = &state.d_aabb_ptr;
//     vertex_input.customPrimitiveArray.flags = state.vertex_input_flags;
//     vertex_input.customPrimitiveArray.numSbtRecords = 1;
//     vertex_input.customPrimitiveArray.numPrimitives = state.window_size;
//     // it's important to pass 0 to sbtIndexOffsetBuffer
//     vertex_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
//     vertex_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
//     vertex_input.customPrimitiveArray.primitiveIndexOffset = 0;

//     OptixAccelBufferSizes gas_buffer_sizes;
//     OPTIX_CHECK(optixAccelComputeMemoryUsage(
//                 state.context,
//                 &accel_options,
//                 &vertex_input,
//                 1, // Number of build inputs
//                 &gas_buffer_sizes
//                 ));
//     state.gas_buffer_sizes = gas_buffer_sizes;
//     CUDA_CHECK(cudaMalloc(
//                reinterpret_cast<void **>(&state.d_temp_buffer_gas),
//                gas_buffer_sizes.tempSizeInBytes
//               ));

//     // non-compacted output and size of compacted GAS.
//     // CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
//     size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
//     CUDA_CHECK(cudaMalloc(
//                reinterpret_cast<void **>(&state.d_gas_output_buffer),
//                compactedSizeOffset + 8
//               ));

//     size_t final_gas_size;
//     OPTIX_CHECK(optixAccelBuild(
//                 state.context,
//                 0, // CUDA stream
//                 &accel_options,
//                 &vertex_input,
//                 1, // num build inputs
//                 state.d_temp_buffer_gas,
//                 gas_buffer_sizes.tempSizeInBytes,
//                 state.d_gas_output_buffer,
//                 gas_buffer_sizes.outputSizeInBytes,
//                 &state.gas_handle,
//                 nullptr,
//                 0
//         ));
//     final_gas_size = compactedSizeOffset;
//     std::cerr << "Final GAS size: " << (float)final_gas_size / (1024 * 1024) << " MB" << std::endl;
//     printf("Final GAS size: %f MB\n", (float)final_gas_size / (1024 * 1024));
// }

void make_gas_for_each_stride(ScanState &state, int unit_num) {
    // * 为每个 stride bvh 设置 temp buffer，output buffer，gas handle
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAabb *d_aabb;
    CUDA_CHECK(cudaMalloc(&d_aabb, state.window_size * sizeof(OptixAabb)));
    state.d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabb);
    
    OptixBuildInput &vertex_input = state.vertex_input;
    vertex_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    vertex_input.customPrimitiveArray.aabbBuffers = &state.d_aabb_ptr;
    vertex_input.customPrimitiveArray.flags = state.vertex_input_flags;
    vertex_input.customPrimitiveArray.numSbtRecords = 1;
    vertex_input.customPrimitiveArray.numPrimitives = state.stride_size;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    vertex_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    vertex_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    vertex_input.customPrimitiveArray.primitiveIndexOffset = 0;
    
    for (int i = 0; i < unit_num; i++) {
        kGenAABB(state.params.window + i * state.stride_size, state.radius, state.stride_size, d_aabb);
        
        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
                    state.context,
                    &accel_options,
                    &vertex_input,
                    1, // Number of build inputs
                    &gas_buffer_sizes
                    ));
        state.gas_buffer_sizes = gas_buffer_sizes;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_temp_buffer_list[i]),
                   gas_buffer_sizes.tempSizeInBytes));

        size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer_list[i]),
                   compactedSizeOffset + 8));

        OPTIX_CHECK(optixAccelBuild(
                    state.context,
                    0, // CUDA stream
                    &accel_options,
                    &vertex_input,
                    1, // num build inputs
                    state.d_gas_temp_buffer_list[i],
                    gas_buffer_sizes.tempSizeInBytes,
                    state.d_gas_output_buffer_list[i],
                    gas_buffer_sizes.outputSizeInBytes,
                    &state.handle_list[i],
                    nullptr,
                    0
            ));
        // printf("Final GAS size: %f MB\n", (float)compactedSizeOffset / (1024 * 1024));
    }

    // * 为 in_stride 申请空间
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_temp_buffer),
               state.gas_buffer_sizes.tempSizeInBytes));
    size_t compactedSizeOffset = roundUp<size_t>(state.gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer),
               compactedSizeOffset + 8));

    // * 为 hybrid gas 申请空间
    vertex_input.customPrimitiveArray.numPrimitives = state.window_size;
    kGenAABB(state.params.window, state.radius, state.window_size, d_aabb);
    OptixAccelBufferSizes gas_buffer_sizes_hybrid;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &vertex_input,
                1, // Number of build inputs
                &gas_buffer_sizes_hybrid
                ));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_temp_buffer_hybrid),
               gas_buffer_sizes_hybrid.tempSizeInBytes));
    compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes_hybrid.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_gas_output_buffer_hybrid),
               compactedSizeOffset + 8));
}

// void rebuild_gas(ScanState &state) {
//     OptixAccelBuildOptions accel_options = {};
//     accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD; // * bring higher performance compared to OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
//     // 改为 OPTIX_BUILD_FLAG_PREFER_FAST_TRACE 时会有性能提升但提升不多
//     accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

//     state.vertex_input.customPrimitiveArray.aabbBuffers = &state.d_aabb_ptr;
//     state.vertex_input.customPrimitiveArray.numPrimitives = state.window_size;

//     // recompute gas_buffer_sizes
//     OptixAccelBufferSizes gas_buffer_sizes;
//     OPTIX_CHECK(optixAccelComputeMemoryUsage(
//                 state.context,
//                 &accel_options,
//                 &state.vertex_input,
//                 1, // Number of build inputs
//                 &gas_buffer_sizes
//                 ));
//     OPTIX_CHECK(optixAccelBuild(
//                 state.context,
//                 0, // CUDA stream
//                 &accel_options,
//                 &state.vertex_input,
//                 1, // num build inputs
//                 state.d_temp_buffer_gas,
//                 gas_buffer_sizes.tempSizeInBytes,
//                 state.d_gas_output_buffer,
//                 gas_buffer_sizes.outputSizeInBytes,
//                 &state.gas_handle,
//                 nullptr,
//                 0
//         ));
//     CUDA_SYNC_CHECK();
// }

void rebuild_gas_stride(ScanState &state, int update_pos) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD; // * bring higher performance compared to OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAabb *d_aabb = reinterpret_cast<OptixAabb *>(state.d_aabb_ptr);
    kGenAABB(state.params.window + update_pos * state.stride_size,
             state.radius,
             state.stride_size,
             d_aabb);
    state.vertex_input.customPrimitiveArray.numPrimitives = state.stride_size;

    OPTIX_CHECK(optixAccelBuild(
                state.context,
                0, // CUDA stream
                &accel_options,
                &state.vertex_input,
                1, // num build inputs
                state.d_gas_temp_buffer,
                state.gas_buffer_sizes.tempSizeInBytes,
                state.d_gas_output_buffer,
                state.gas_buffer_sizes.outputSizeInBytes,
                &state.params.in_stride_handle,
                nullptr,
                0
        ));
    // CUDA_SYNC_CHECK();
}

void make_gas_by_cell(ScanState &state) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD; // * bring higher performance compared to OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAabb *d_aabb = reinterpret_cast<OptixAabb *>(state.d_aabb_ptr);
    kGenAABB_by_center(state.params.centers, state.params.radii, state.params.center_num, d_aabb);
    state.vertex_input.customPrimitiveArray.numPrimitives = state.params.center_num;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &state.vertex_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ));
    OPTIX_CHECK(optixAccelBuild(
                state.context,
                0, // CUDA stream
                &accel_options,
                &state.vertex_input,
                1, // num build inputs
                state.d_gas_temp_buffer_hybrid,
                gas_buffer_sizes.tempSizeInBytes,
                state.d_gas_output_buffer_hybrid,
                gas_buffer_sizes.outputSizeInBytes,
                &state.params.handle,
                nullptr,
                0
        ));
#ifndef OPTIMIZATION_HETEROGENEOUS
    CUDA_SYNC_CHECK();
#endif
}

void make_module(ScanState &state) {
    size_t make_module_start_mem;
    start_gpu_mem(&make_module_start_mem);

    char log[2048];

    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 4;
    state.pipeline_compile_options.numAttributeValues = 0;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    // By default (usesPrimitiveTypeFlags == 0) it supports custom and triangle primitives
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    size_t inputSize = 0;
    const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "dbscan.cu", inputSize);
    size_t sizeof_log = sizeof(log);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        log,
        &sizeof_log,
        &state.module));

    size_t used;
    stop_gpu_mem(&make_module_start_mem, &used);
    std::cout << "[Mem] make_module: " << 1.0 * used / (1 << 20) << std::endl;
}

void make_program_groups(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    char log[2048];

    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module; // 指定 cu 文件名
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &miss_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleIS = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
    // hitgroup_prog_group_desc.hitgroup.moduleAH = state.module;
    // hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__terminate_ray";

    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.hitgroup_prog_group));


    // * cluster
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module; // 指定 cu 文件名
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__cluster";
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &raygen_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.raygen_cluster));

    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleIS = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cluster";
    // hitgroup_prog_group_desc.hitgroup.moduleAH = state.module;
    // hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__terminate_ray";

    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.hitgroup_cluster));

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] make_program_groups: " << 1.0 * used / (1 << 20) << std::endl;
}

void make_pipeline(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    char log[2048];
    const uint32_t max_trace_depth = 1;
    std::vector<OptixProgramGroup> program_groups{state.raygen_prog_group, state.miss_prog_group, state.hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL; // TODO: 或可更改
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups.data(),
        program_groups.size(),
        log,
        &sizeof_log,
        &state.pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));


    // * cluster
    std::vector<OptixProgramGroup> program_groups_cluster{state.raygen_cluster, state.miss_prog_group, state.hitgroup_cluster};

    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups_cluster.data(),
        program_groups_cluster.size(),
        log,
        &sizeof_log,
        &state.pipeline_cluster));

    for (auto &prog_group : program_groups_cluster) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                           0, // maxCCDepth
                                           0, // maxDCDEpth
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline_cluster, direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state, continuation_stack_size,
                                          1 // maxTraversableDepth
                                          ));

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] make_pipeline: " << 1.0 * used / (1 << 20) << std::endl;
}

void make_sbt(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(miss_record),
        &ms_sbt,
        miss_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hitgroup_record),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = 1;


    // * cluster
    CUdeviceptr raygen_record_cluster;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record_cluster), raygen_record_size));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_cluster, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(raygen_record_cluster),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record_cluster;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record_cluster), hitgroup_record_size));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hitgroup_cluster, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(hitgroup_record_cluster),
        &hg_sbt,
        hitgroup_record_size,
        cudaMemcpyHostToDevice));

    state.sbt_cluster.raygenRecord = raygen_record_cluster;
    state.sbt_cluster.missRecordBase = miss_record;
    state.sbt_cluster.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt_cluster.missRecordCount = 1;
    state.sbt_cluster.hitgroupRecordBase = hitgroup_record_cluster;
    state.sbt_cluster.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt_cluster.hitgroupRecordCount = 1;

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] make_sbt: " << 1.0 * used / (1 << 20) << std::endl;
}
