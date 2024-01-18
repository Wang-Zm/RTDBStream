#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>

#include <sampleConfig.h>

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#include "state.h"
#include "timer.h"

#include <array>
#include <bitset>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <map>
#include <thread>
#include <sutil/Camera.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <fstream>
using namespace std;

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

Timer timer;

extern "C" void kGenAABB(DATA_TYPE_3 *points, DATA_TYPE radius, unsigned numPrims, OptixAabb *d_aabb);
extern "C" void find_cores(int* label, int* nn, int window_size, int min_pts);
extern "C" void union_cluster(int* tmp_cluster_id, int* cluster_id, int* label, int window_size);
extern "C" void find_neighbors(int* label, int* nn, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2, int min_pts);
extern "C" void set_cluster_id(int* label, int* cluster_id, DATA_TYPE_3* window, int window_size, DATA_TYPE radius2);

void printUsageAndExit(const char* argv0) {
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      Specify file for data input\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    std::cerr << "         --n <int>                   Set data num; defaults to 1e8\n";
    std::cerr << "         --primitive <int>           Set primitive type, 0 for cube, 1 for triangle with anyhit; defaults to 0\n";
    std::cerr << "         --nc                        No Comparison\n";
    exit(1);
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

void parse_args(ScanState &state, int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            printUsageAndExit(argv[0]);
        } else if (arg == "--file" || arg == "-f") {
            if (i < argc - 1) {
                state.data_file = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "-n") {
            if (i < argc - 1) {
                state.data_num = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--window" || arg == "-W") {
            if (i < argc - 1) {
                state.window_size = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--stride" || arg == "-S") {
            if (i < argc - 1) {
                state.stride_size = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--minPts" || arg == "-K") {
            if (i < argc - 1) {
                state.min_pts = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--radius" || arg == "-R") {
            if (i < argc - 1) {
                state.radius = stod(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else {
            std::cerr << "Unknown option '" << arg << "'\n";
            printUsageAndExit(argv[0]);
        }
    }
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

void make_gas(ScanState &state) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAabb *d_aabb;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabb), state.window_size * sizeof(OptixAabb)));
    kGenAABB(state.params.window, state.radius, state.window_size, d_aabb);
    state.d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabb);

    OptixBuildInput &vertex_input = state.vertex_input;
    vertex_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    vertex_input.customPrimitiveArray.aabbBuffers = &state.d_aabb_ptr;
    vertex_input.customPrimitiveArray.flags = state.vertex_input_flags;
    vertex_input.customPrimitiveArray.numSbtRecords = 1;
    vertex_input.customPrimitiveArray.numPrimitives = state.window_size;
    // it's important to pass 0 to sbtIndexOffsetBuffer
    vertex_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
    vertex_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    vertex_input.customPrimitiveArray.primitiveIndexOffset = 0;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                state.context,
                &accel_options,
                &vertex_input,
                1, // Number of build inputs
                &gas_buffer_sizes
                ));
    state.gas_buffer_sizes = gas_buffer_sizes;
    CUDA_CHECK(cudaMalloc(
               reinterpret_cast<void **>(&state.d_temp_buffer_gas),
               gas_buffer_sizes.tempSizeInBytes
              ));

    // non-compacted output and size of compacted GAS.
    // CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
               reinterpret_cast<void **>(&state.d_gas_output_buffer),
               compactedSizeOffset + 8
              ));

    size_t final_gas_size;
    OPTIX_CHECK(optixAccelBuild(
                state.context,
                0, // CUDA stream
                &accel_options,
                &vertex_input,
                1, // num build inputs
                state.d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                state.d_gas_output_buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                nullptr,
                0
        ));
    final_gas_size = compactedSizeOffset;
    std::cerr << "Final GAS size: " << (float)final_gas_size / (1024 * 1024) << " MB" << std::endl;
    printf("Final GAS size: %f MB\n", (float)final_gas_size / (1024 * 1024));
}

void rebuild_gas(ScanState &state, int update_pos) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD; // * bring higher performance compared to OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // update aabb
    OptixAabb *d_aabb = reinterpret_cast<OptixAabb *>(state.d_aabb_ptr);
    kGenAABB(state.params.window + update_pos * state.stride_size,
             state.radius,
             state.stride_size,
             d_aabb + update_pos * state.stride_size);

    // recompute gas_buffer_sizes
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
                state.d_temp_buffer_gas,
                gas_buffer_sizes.tempSizeInBytes,
                state.d_gas_output_buffer,
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                nullptr,
                0
        ));
    CUDA_SYNC_CHECK();
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
    state.pipeline_compile_options.numPayloadValues = 3;
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
    const char *input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixScan.cu", inputSize);
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

void initialize_params(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    // state.params.handle = state.gas_handle; // 已在 search 方法中写好
    state.params.radius2= state.radius * state.radius;
    state.params.tmin   = 0.0f;
    state.params.tmax   = FLT_MIN;
    state.params.min_pts= state.min_pts;
    state.params.window_size = state.window_size;
    CUDA_CHECK(cudaMalloc(&state.params.cluster_id, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.label, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.nn, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.window, state.window_size * sizeof(DATA_TYPE_3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));
    state.h_cluster_id = (int*) malloc(state.window_size * sizeof(int));

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaMalloc(&state.params.ray_intersections, state.query_num * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&state.params.ray_primitive_hits, state.query_num * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(state.params.ray_intersections, 0, state.query_num * sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(state.params.ray_primitive_hits, 0, state.query_num * sizeof(unsigned)));
    state.h_ray_intersections = (unsigned *) malloc(state.query_num * sizeof(unsigned));
    state.h_ray_hits = (unsigned *) malloc(state.query_num * sizeof(unsigned));
#endif

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] initialize_params: " << 1.0 * used / (1 << 20) << std::endl;
}

void log_common_info(ScanState &state) {
    std::cout << "Input file: " << state.data_file << std::endl;
    std::cout << "Data num: " << state.data_num << std::endl;
    std::cout << "Window size: " << state.window_size << ", Stride size:" << state.stride_size << ", K:" << state.min_pts << std::endl;
    std::cout << "Radius: " << state.radius << ", Radius2: " << state.params.radius2 << std::endl;
}

void cluster(ScanState &state) {
    // * split
    // 找到 ex-cores 的 R-
    // 1.从上一个 BVH tree 中查这些点的相邻关系，从每个点发射光线，记录邻居，之后找到集合；R- 集合中暂存放 ex-core 的具体的 index
    // 2.找到所有的core，记录下来 [暂时忽略这些工程优化]
    // 3.使用数组存下所有点的邻居关系

    // TODO: 改变 pipeline
    // TODO: 从所有 ex-cores 发射光线，设置 R_out_f[] 和 M_out_f[]
    OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.stride_size, 1, 1)); // 找新的邻居
    CUDA_SYNC_CHECK();
    // 得到 R_out_f[]，处理后得到每个点属于的 R-
    // 得到 M_out_f[]，处理后得到每个点属于的 M- 集合

    // TODO: 遍历每个 M- 集合，队列的处理？


    // 遍历每个 M- 集合，判别其中点的连通关系；从一个点开始 BFS，标记访问过的 M-(p) 中的点，如果队列已空，从未访问过的 M-(p) 中找点，再次 BFS
}

int find(int x, ScanState &state) {
    return state.h_cluster_id[x] == x ? x : state.h_cluster_id[x] = find(state.h_cluster_id[x], state);
}

void unite(int x, int y, ScanState &state) {
    int fx = find(x, state), fy = find(y, state);
    if (fx < fy) {
        state.h_cluster_id[fy] = fx;
    }
}

void search(ScanState &state) {
    int remaining_data_num  = state.data_num - state.window_size;
    int unit_num            = state.window_size / state.stride_size;
    int update_pos          = 0;
    int stride_num          = 0;
    int window_left         = 0;
    int window_right        = state.window_size;
    state.new_stride        = state.h_data + state.window_size;

    log_common_info(state);

    // * Initialize the first window
    CUDA_CHECK(cudaMemcpy(state.params.window, state.h_data, state.window_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
    make_gas(state);
    printf("[Step] Initialize the first window - build BVH tree...\n");
    state.params.handle = state.gas_handle;
    state.params.out = state.params.window;
    state.params.operation = 1;
    CUDA_CHECK(cudaMemset(state.params.nn, 0, state.window_size * sizeof(int))); // ! 无需设置 label，之前的 label 标记已无作用
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
    CUDA_SYNC_CHECK();
    printf("[Step] Initialize the first window - get NN...\n");

    // * Start sliding
    state.h_label = (int*) malloc(state.window_size * sizeof(int));
    CUDA_CHECK(cudaMalloc(&state.params.tmp_cluster_id, state.window_size * sizeof(int)));
    printf("[Info] Total stride num: %d\n", remaining_data_num / state.stride_size);
    while (remaining_data_num >= state.stride_size) {
        timer.startTimer(&timer.total);
        
#if MODE == 0
        // * use RT
        state.params.out = state.params.window + update_pos * state.stride_size;
        // out stride
        timer.startTimer(&timer.out_stride);
        state.params.operation = 0; // nn--
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.stride_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.out_stride);
        // printf("    [Step] out stride\n");

        // 插入 in stride
        timer.startTimer(&timer.rebuild_bvh);
        CUDA_CHECK(cudaMemcpy(state.params.out, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        rebuild_gas(state, update_pos);
        // 重置 in/out stride 部分 nn
        CUDA_CHECK(cudaMemset(state.params.nn + update_pos * state.stride_size, 0, state.stride_size * sizeof(int)));
        timer.stopTimer(&timer.rebuild_bvh);
        // printf("    [Step] rebuild BVH tree\n");

        timer.startTimer(&timer.in_stride);
        state.params.operation = 1; // nn++
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(state.d_params), &state.params, sizeof(Params), cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.stride_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.in_stride);
        // printf("    [Step] in stride\n");

        // set labels and find cores
        timer.startTimer(&timer.find_cores);
        find_cores(state.params.label, state.params.nn, state.window_size, state.min_pts);
        timer.stopTimer(&timer.find_cores);
        // printf("    [Step] find cores\n");

        // 根据获取到的 core 开始 union，设置 cluster_id
        timer.startTimer(&timer.set_cluster_id);
        OPTIX_CHECK(optixLaunch(state.pipeline_cluster, 0, state.d_params, sizeof(Params), &state.sbt, state.window_size, 1, 1));
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.set_cluster_id);
        // printf("    [Step] set cluster_id\n");


#else
        // ! 使用 cuda 暴搜替换：1）所有点的邻居个数，确定 core 2）再扫一遍，找 border
        // * use CUDA
        timer.startTimer(&timer.cuda_find_neighbors);
        // update window
        CUDA_CHECK(cudaMemcpy(state.params.out, state.new_stride, state.stride_size * sizeof(DATA_TYPE_3), cudaMemcpyHostToDevice));
        // reset nn
        CUDA_CHECK(cudaMemset(state.params.nn + update_pos * state.stride_size, 0, state.stride_size * sizeof(int)));
        find_neighbors(state.params.label, state.params.nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.cuda_find_neighbors);

        timer.startTimer(&timer.cuda_set_clusters);
        set_cluster_id(state.params.label, state.params.cluster_id, state.params.window, state.window_size, state.params.radius2);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.cuda_set_clusters);
#endif

        // 暂时实现并行的 union-find，不进行路径压缩而是直接设置当前值
        timer.startTimer(&timer.union_cluster_id);
        // * parallel union-find
        // CUDA_CHECK(cudaMemcpy(state.params.tmp_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToDevice));
        // union_cluster(state.params.tmp_cluster_id, state.params.cluster_id, state.params.label, state.window_size); // ! 这里停了下来，可能是并行引入了循环问题
        // CUDA_SYNC_CHECK();
        // * serial union-find
        CUDA_CHECK(cudaMemcpy(state.h_label, state.params.label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < state.window_size; i++) {
            if (state.h_label[i] == 2) continue;
            find(i, state);
        }
        timer.stopTimer(&timer.union_cluster_id);
        // printf("    [Step] union clusters\n");

        // 传回 cluster
        // timer.startTimer(&timer.copy_cluster_d2h);
        // CUDA_CHECK(cudaMemcpy(state.h_cluster_id, state.params.cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
        // timer.startTimer(&timer.copy_cluster_d2h);

        stride_num++;
        remaining_data_num  -= state.stride_size;
        state.new_stride    += state.stride_size;
        update_pos           = (update_pos + 1) % unit_num;
        window_left         += state.stride_size;
        window_right        += state.stride_size;
        timer.stopTimer(&timer.total);
        // printf("[Time] Total process: %lf ms\n", timer.total);
        // timer.total = 0.0;
        // printf("[Step] Finish window %d\n", window_left / state.stride_size);
    }
    printf("[Step] Finish sliding the window...\n");
}

// void result_d2h(ScanState &state) { // https://www.cnblogs.com/ybqjymy/p/17462877.html
//     timer.startTimer(&timer.copy_results_d2h);
//     for (int i = 0; i < state.query_num; i++) {
//         CUDA_CHECK(cudaMemcpy(state.h_dist[i], state.h_dist_temp[i], state.data_num * sizeof(DIST_TYPE), cudaMemcpyDeviceToHost));
//     }
//     long total_neighbor_num = 0;
//     for (int i = 0; i < state.query_num; i++) {
//         int neighbor_num = 0;
//         for (int j = 0; j < state.data_num; j++) {
//             // TODO: 只会计算到query R内的point的距离，超出的不计算，一开始dist仍然是0；可能之后的一些维度的和<R，会导致求解的邻居多余实际邻居数
//             if (state.h_dist[i][j] > 0 && state.h_dist[i][j] < state.params.radius2) {
//                 neighbor_num++;
//                 state.queries_neighbors[i].push_back(j);
//                 state.queries_neighbor_dist[i].push_back(state.h_dist[i][j]);
//             }
//         }
//         state.queries_neighbor_num[i] = neighbor_num;
//         total_neighbor_num += neighbor_num;
//     }
//     timer.stopTimer(&timer.copy_results_d2h);

//     std::cout << "Total number of neighbors:     " << total_neighbor_num << std::endl;
//     std::cout << "Ratio of returned data points: " << 1.0 * total_neighbor_num / (state.query_num * state.data_num) * 100 << "%" << std::endl;
//     // for (int i = 0; i < state.query_num; i++) {
//     //     std::cout << "Query[" << i << "]: " << state.queries_neighbor_num[i] << std::endl;
//     // }

// #if DEBUG_INFO == 1
//     CUDA_CHECK(cudaMemcpy(state.h_ray_intersections, state.params.ray_intersections, state.query_num * sizeof(unsigned), cudaMemcpyDeviceToHost));
//     CUDA_CHECK(cudaMemcpy(state.h_ray_hits, state.params.ray_primitive_hits, state.query_num * sizeof(unsigned), cudaMemcpyDeviceToHost));
//     long total_hits = 0, total_intersection_tests = 0;
//     for (int i = 0; i < state.query_num; i++) {
//         total_hits += state.h_ray_hits[i];
//         total_intersection_tests += state.h_ray_intersections[i];
//     }
//     std::cout << "Total hits:               " << total_hits << std::endl;
//     std::cout << "Total intersection tests: " << total_intersection_tests << std::endl;

//     long effective_write_num = total_neighbor_num * ((state.dim + 3 - 1) / 3);
//     long ineffective_write_num = total_hits - effective_write_num;
//     std::cout << "Effective write count:   " << effective_write_num << std::endl;
//     std::cout << "Ineffective write count: " << ineffective_write_num << std::endl;
//     std::cout << fixed << setprecision(0) << "Effective write count per query:   " << 1.0 * effective_write_num / state.query_num << std::endl;
//     std::cout << fixed << setprecision(0) << "Ineffective write count per query: " << 1.0 * ineffective_write_num / state.query_num << std::endl;
// #endif
// }

void cleanup(ScanState &state) {
    // free host memory
    free(state.h_data);
    free(state.h_cluster_id);

    // free device memory
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.hitgroupRecordBase)));

    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.hitgroup_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixModuleDestroy(state.module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(state.params.window));

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaFree(state.params.ray_primitive_hits));
    CUDA_CHECK(cudaFree(state.params.ray_intersections));
    free(state.h_ray_hits);
    free(state.h_ray_intersections);
#endif

    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.d_params)));
}

int main(int argc, char *argv[]) {
    setbuf(stdout,NULL); // https://blog.csdn.net/moshiyaofei/article/details/107297472
    ScanState state;
    parse_args(state, argc, argv);
    read_data_from_tao(state.data_file, state);

    initialize_optix(state);
    make_module(state);
    make_program_groups(state);
    make_pipeline(state);               // Link pipeline
    make_sbt(state);
    initialize_params(state);
    search(state);
    // result_d2h(state);
    timer.showTime((state.data_num - state.window_size) / state.stride_size);
    cleanup(state);
    return 0;
}
