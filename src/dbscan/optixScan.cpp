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

Timer                    timer;

extern "C" void kGenAABB(DATA_TYPE **points, int bvh_id, double radius, unsigned int numPrims, OptixAabb *d_aabb);
extern "C" void refine_with_cuda(DIST_TYPE** dist, 
                                 unsigned* dist_flag,
                                 DATA_TYPE** points,
                                 DATA_TYPE** queries, 
                                 int query_num, 
                                 int data_num, 
                                 int bvh_num,
                                 double radius2);
extern "C" void search_with_cuda(DIST_TYPE** dist, 
                                 DATA_TYPE** points,
                                 DATA_TYPE** queries, 
                                 int query_num, 
                                 int data_num, 
                                 int bvh_num,
                                 double radius2);

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
        } else if (arg == "--query_file") {
            if (i < argc - 1) {
                state.query_file = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--n") {
            if (i < argc - 1) {
                state.data_num = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--dim") {
            if (i < argc - 1) {
                state.dim = stoi(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--radius") {
            if (i < argc - 1) {
                state.radius = stod(argv[++i]);
            } else {
                printUsageAndExit(argv[0]);
            }
        } else if (arg == "--query_num") {
            if (i < argc - 1) {
                state.query_num = stoi(argv[++i]);
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

void data_h2d(ScanState &state) {
    DATA_TYPE *points[state.data_num];
    for (int i = 0; i < state.data_num; i++) {
        CUDA_CHECK(cudaMalloc(&points[i], state.dim * sizeof(DATA_TYPE)));
        CUDA_CHECK(cudaMemcpy(points[i], state.vertices[i], state.dim * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMalloc(&state.params.points, state.data_num * sizeof(DATA_TYPE*)));
    CUDA_CHECK(cudaMemcpy(state.params.points, points, state.data_num * sizeof(DATA_TYPE*), cudaMemcpyHostToDevice));
}

void make_gas(ScanState &state) {
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    const uint32_t vertex_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    
    size_t final_gas_size = 0;
    int bvh_num = state.dim / 3;
    state.gas_handle = (OptixTraversableHandle*) malloc(bvh_num * sizeof(OptixTraversableHandle));

    printf("make_gas, state.radius / sqrt(bvh_num) = %lf\n", state.radius / sqrt(bvh_num));
    for (int bvh_id = 0; bvh_id < bvh_num; bvh_id++) {
        OptixAabb *d_aabb;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_aabb), state.data_num * sizeof(OptixAabb)));
#if RT_FILTER == 1
        kGenAABB(state.params.points, bvh_id, state.radius / sqrt(bvh_num), state.data_num, d_aabb);
#else
        kGenAABB(state.params.points, bvh_id, state.radius, state.data_num, d_aabb);
#endif
        CUdeviceptr d_aabb_ptr = reinterpret_cast<CUdeviceptr>(d_aabb);

        // Our build input is a simple list of non-indexed triangle vertices
        OptixBuildInput vertex_input = {};
        vertex_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        vertex_input.customPrimitiveArray.aabbBuffers = &d_aabb_ptr;
        vertex_input.customPrimitiveArray.flags = vertex_input_flags;
        vertex_input.customPrimitiveArray.numSbtRecords = 1;
        vertex_input.customPrimitiveArray.numPrimitives = state.data_num;
        // it's important to pass 0 to sbtIndexOffsetBuffer
        vertex_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
        vertex_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
        vertex_input.customPrimitiveArray.primitiveIndexOffset = 0;

        OptixAccelBufferSizes gas_buffer_sizes;
        CUdeviceptr d_temp_buffer_gas, d_gas_output_buffer;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            state.context,
            &accel_options,
            &vertex_input,
            1, // Number of build inputs
            &gas_buffer_sizes));
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_temp_buffer_gas),
            gas_buffer_sizes.tempSizeInBytes));

        // non-compacted output and size of compacted GAS.
        // CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
        size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&d_gas_output_buffer),
            compactedSizeOffset + 8));

        OPTIX_CHECK(optixAccelBuild(
            state.context,
            0, // CUDA stream
            &accel_options,
            &vertex_input,
            1, // num build inputs
            d_temp_buffer_gas,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer,
            gas_buffer_sizes.outputSizeInBytes,
            &state.gas_handle[bvh_id],
            nullptr,
            0));
        final_gas_size += compactedSizeOffset;
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    }
    std::cerr << "Final GAS size: " << (float)final_gas_size / (1024 * 1024) << std::endl;
    printf("Final GAS size: %f MB\n", (float)final_gas_size / (1024 * 1024));
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
    raygen_prog_group_desc.raygen.module = state.module;
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
#if RT_FILTER == 1
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__flag";
#else
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__cube";
#endif
    hitgroup_prog_group_desc.hitgroup.moduleAH = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__terminate_ray";
      
    sizeof_log = sizeof(log);
    OPTIX_CHECK_LOG(optixProgramGroupCreate(
        state.context,
        &hitgroup_prog_group_desc,
        1, // num program groups
        &program_group_options,
        log,
        &sizeof_log,
        &state.hitgroup_prog_group));

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
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
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

    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] make_sbt: " << 1.0 * used / (1 << 20) << std::endl;
}

void initialize_params(ScanState &state) {
    size_t start;
    start_gpu_mem(&start);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&state.d_params), sizeof(Params)));
    
    CUDA_CHECK(cudaMalloc(&state.params.handle, state.dim / 3 * sizeof(OptixTraversableHandle)));
    CUDA_CHECK(cudaMemcpy(state.params.handle, state.gas_handle, state.dim / 3 * sizeof(OptixTraversableHandle), cudaMemcpyHostToDevice));
    

    DIST_TYPE *h_dist;
    state.h_dist_temp = (DIST_TYPE**) malloc(state.query_num * sizeof(DIST_TYPE*));
    h_dist = (DIST_TYPE*) malloc(state.data_num * sizeof(DIST_TYPE));
    for (int i = 0; i < state.data_num; i++) h_dist[i] = 0.0;
    for (int i = 0; i < state.query_num; i++) {
        CUDA_CHECK(cudaMalloc(&state.h_dist_temp[i], state.data_num * sizeof(DIST_TYPE)));
        CUDA_CHECK(cudaMemcpy(state.h_dist_temp[i], h_dist, state.data_num * sizeof(DIST_TYPE), cudaMemcpyHostToDevice)); // Set dist to all zero
    }
    CUDA_CHECK(cudaMalloc(&state.params.dist, state.query_num * sizeof(DIST_TYPE*)));
    CUDA_CHECK(cudaMemcpy(state.params.dist, state.h_dist_temp, state.query_num * sizeof(DIST_TYPE*), cudaMemcpyHostToDevice));
    free(h_dist);
    
    state.h_dist = (DIST_TYPE**) malloc(state.query_num * sizeof(DIST_TYPE*));
    for (int i = 0; i < state.query_num; i++) {
        state.h_dist[i] = (DIST_TYPE*) malloc(state.data_num * sizeof(DIST_TYPE));
    }

    state.unsigned_len = (state.dim + 96 - 1) / 96; // 96 = 3 * 4 * 8
    state.dist_flag_len = size_t(state.query_num) * state.data_num * state.unsigned_len * sizeof(unsigned);
    CUDA_CHECK(cudaMalloc(&state.params.dist_flag, state.dist_flag_len));
    CUDA_CHECK(cudaMemset(state.params.dist_flag, 0, state.dist_flag_len));
    std::cout << "dist_flag size: " << state.dist_flag_len << std::endl;

    state.h_queries_temp = (DATA_TYPE**) malloc(state.query_num * sizeof(DATA_TYPE*));
    for (int i = 0; i < state.query_num; i++) {
        CUDA_CHECK(cudaMalloc(&state.h_queries_temp[i], state.dim * sizeof(DATA_TYPE)));
    }
    CUDA_CHECK(cudaMalloc(&state.params.queries, state.query_num * sizeof(DATA_TYPE*)));

    state.queries_neighbor_num  = (int*) malloc(state.query_num * sizeof(int));
    state.queries_neighbors.resize(state.query_num);
    state.queries_neighbor_dist.resize(state.query_num);

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
    std::cout << "input file: " << state.data_file << std::endl;
    std::cout << "query file: " << state.query_file << std::endl;
    std::cout << "data num: " << state.data_num << std::endl;
    std::cout << "query num: " << state.query_num << std::endl;
    std::cout << "dim: " << state.dim << std::endl;
    std::cout << "bvh num: " << state.params.bvh_num << std::endl;
    std::cout << "unsigned len: " << state.unsigned_len << std::endl;
    std::cout << "dist flag len: " << state.dist_flag_len << std::endl;
    std::cout << "radius: " << state.radius << ", radius2: " << state.params.radius2 << ", sub_radius2: " << state.params.sub_radius2 << std::endl;
}

void search(ScanState &state) {
    timer.startTimer(&timer.copy_queries_h2d);
    // Prepare queries
    for (int i = 0; i < state.query_num; i++) {
        CUDA_CHECK(cudaMemcpy(state.h_queries_temp[i], state.h_queries[i], state.dim * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(state.params.queries, state.h_queries_temp, state.query_num * sizeof(DATA_TYPE*), cudaMemcpyHostToDevice));

    // Prepare parameters
    state.params.data_num   = state.data_num;
    state.params.dim        = state.dim;
    state.params.unsigned_len = state.unsigned_len;
    state.params.radius     = state.radius;
    state.params.radius2    = state.radius * state.radius;
    state.params.bvh_num    = state.dim / 3;
    state.params.sub_radius2= state.params.radius2 / state.params.bvh_num;
    state.params.tmin       = 0.0f;
    state.params.tmax       = FLT_MIN;
    timer.stopTimer(&timer.copy_queries_h2d);

    log_common_info(state);
    
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(state.d_params),
        &state.params,
        sizeof(Params),
        cudaMemcpyHostToDevice)); // 对于GPU中的查询，实际无需数据在host与device之间传输

    // Warmup
//     OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.query_num, state.dim / 3, 1));
//     CUDA_SYNC_CHECK();

// #if RT_FILTER == 1
//     refine_with_cuda(state.params.dist,
//                      state.params.dist_flag,
//                      state.params.points,
//                      state.params.queries,
//                      state.query_num,
//                      state.data_num,
//                      state.params.bvh_num,
//                      state.params.radius2);
//     CUDA_SYNC_CHECK();
// #endif

#if DEBUG_INFO == 0
    // Timing
    int runs = 5; // 5
    DIST_TYPE *tmp_dist = (DIST_TYPE *) malloc(state.data_num * sizeof(DIST_TYPE));
    for (int i = 0; i < state.data_num; i++) tmp_dist[i] = 0;
    for (int i = 0; i < runs; i++) {
        // Clear dist array
        for (int i = 0; i < state.query_num; i++) {
            CUDA_CHECK(cudaMemcpy(state.h_dist_temp[i], tmp_dist, state.data_num * sizeof(DIST_TYPE), cudaMemcpyHostToDevice));
        }

        timer.startTimer(&timer.cuda_refine);
        search_with_cuda(state.params.dist,
                         state.params.points,
                         state.params.queries,
                         state.query_num,
                         state.data_num,
                         state.params.bvh_num,
                         state.params.radius2);
        CUDA_SYNC_CHECK();
        timer.stopTimer(&timer.cuda_refine);

// #if RT_FILTER == 1
//         CUDA_CHECK(cudaMemset(state.params.dist_flag, 0, state.dist_flag_len));
// #endif

//         timer.startTimer(&timer.search_neighbors);
//         OPTIX_CHECK(optixLaunch(state.pipeline, 0, state.d_params, sizeof(Params), &state.sbt, state.query_num, state.dim / 3, 1));
//         CUDA_SYNC_CHECK(); // ! 若后面用cuda refine，则该同步方法似乎不再需要？
//         timer.stopTimer(&timer.search_neighbors);

// #if RT_FILTER == 1
//         // cuda来找剩下的点，使用kernel方法
//         timer.startTimer(&timer.cuda_refine);
//         refine_with_cuda(state.params.dist,
//                          state.params.dist_flag,
//                          state.params.points,
//                          state.params.queries,
//                          state.query_num,
//                          state.data_num,
//                          state.params.bvh_num,
//                          state.params.radius2);
//         CUDA_SYNC_CHECK();
//         timer.stopTimer(&timer.cuda_refine);
// #endif
    }
    timer.search_neighbors /= runs;
    timer.cuda_refine /= runs;
    free(tmp_dist);
#endif
}

void result_d2h(ScanState &state) { // https://www.cnblogs.com/ybqjymy/p/17462877.html
    timer.startTimer(&timer.copy_results_d2h);
    for (int i = 0; i < state.query_num; i++) {
        CUDA_CHECK(cudaMemcpy(state.h_dist[i], state.h_dist_temp[i], state.data_num * sizeof(DIST_TYPE), cudaMemcpyDeviceToHost));
    }
    long total_neighbor_num = 0;
    for (int i = 0; i < state.query_num; i++) {
        int neighbor_num = 0;
        for (int j = 0; j < state.data_num; j++) {
            // TODO: 只会计算到query R内的point的距离，超出的不计算，一开始dist仍然是0；可能之后的一些维度的和<R，会导致求解的邻居多余实际邻居数
            if (state.h_dist[i][j] > 0 && state.h_dist[i][j] < state.params.radius2) {
                neighbor_num++;
                state.queries_neighbors[i].push_back(j);
                state.queries_neighbor_dist[i].push_back(state.h_dist[i][j]);
            }
        }
        state.queries_neighbor_num[i] = neighbor_num;
        total_neighbor_num += neighbor_num;
    }
    timer.stopTimer(&timer.copy_results_d2h);

    std::cout << "Total number of neighbors:     " << total_neighbor_num << std::endl;
    std::cout << "Ratio of returned data points: " << 1.0 * total_neighbor_num / (state.query_num * state.data_num) * 100 << "%" << std::endl;
    // for (int i = 0; i < state.query_num; i++) {
    //     std::cout << "Query[" << i << "]: " << state.queries_neighbor_num[i] << std::endl;
    // }

#if DEBUG_INFO == 1
    CUDA_CHECK(cudaMemcpy(state.h_ray_intersections, state.params.ray_intersections, state.query_num * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.h_ray_hits, state.params.ray_primitive_hits, state.query_num * sizeof(unsigned), cudaMemcpyDeviceToHost));
    long total_hits = 0, total_intersection_tests = 0;
    for (int i = 0; i < state.query_num; i++) {
        total_hits += state.h_ray_hits[i];
        total_intersection_tests += state.h_ray_intersections[i];
    }
    std::cout << "Total hits:               " << total_hits << std::endl;
    std::cout << "Total intersection tests: " << total_intersection_tests << std::endl;
    
    long effective_write_num = total_neighbor_num * ((state.dim + 3 - 1) / 3);
    long ineffective_write_num = total_hits - effective_write_num;
    std::cout << "Effective write count:   " << effective_write_num << std::endl;
    std::cout << "Ineffective write count: " << ineffective_write_num << std::endl;
    std::cout << fixed << setprecision(0) << "Effective write count per query:   " << 1.0 * effective_write_num / state.query_num << std::endl;
    std::cout << fixed << setprecision(0) << "Ineffective write count per query: " << 1.0 * ineffective_write_num / state.query_num << std::endl;
#endif
}

// void calc_total_hit_intersection_each_window(ScanState &state) {
//     int ray_num = OPTIMIZATION == 1 ? state.params.ray_origin_num : state.window;
//     if (OPTIMIZATION == 0 && state.launch_ray_num != 0) {
//         ray_num = state.launch_ray_num;
//     }
//     int bvh_node_num = OPTIMIZATION == 2 ? state.params.ray_origin_num : state.window;
//     unsigned total_hit = 0, total_intersection_test = 0;
//     for (int i = 0; i < ray_num; i++) {
//         total_hit += state.h_ray_hits[i];
//         total_intersection_test += state.h_ray_intersections[i];
//     }
//     state.total_hit     += total_hit;
//     state.total_is_test += total_intersection_test;
//     state.total_is_test_per_ray += 1.0 * total_intersection_test / ray_num;
//     state.total_hit_per_ray     += 1.0 * total_hit / ray_num;
//     std::cout << "Total_hit: " << total_hit << ", " << "Total_intersection_test: " << total_intersection_test << std::endl;
//     std::cout << "BVH_Node: " << bvh_node_num << ", Cast_Ray_Num: " << ray_num << std::endl;
// }

// void calc_ray_hits(ScanState &state, unsigned *ray_hits) {
//     map<unsigned, int> hitNum_rayNum;
//     int sum = 0;
//     int ray_num = OPTIMIZATION == 1 ? state.params.ray_origin_num : state.window;
//     if (OPTIMIZATION == 0 && state.launch_ray_num != 0) {
//         ray_num = state.launch_ray_num;
//     }
//     for (int i = 0; i < ray_num; i++) {
//         sum += ray_hits[i];
//         if (hitNum_rayNum.count(ray_hits[i])) {
//             hitNum_rayNum[ray_hits[i]]++;
//         } else {
//             hitNum_rayNum[ray_hits[i]] = 1;
//         }       
//     }

//     int min, max, median = -1;
//     double avg;
//     int tmp_sum = 0;
//     min = hitNum_rayNum.begin()->first;
//     max = (--hitNum_rayNum.end())->first;
//     avg = 1.0 * sum / ray_num;
//     printf("hit num: ray num\n");
//     for (auto &item: hitNum_rayNum) {
//         fprintf(stdout, "%d: %d\n", item.first, item.second);
//         tmp_sum += item.second;
//         if (median == -1 && tmp_sum >= ray_num / 2) {
//             median = item.first;
//         }
//     }
//     printf("min: %d, max: %d, average: %lf, median: %d\n", min, max, avg, median);
// }

void cleanup(ScanState &state) {
    // free host memory
    free(state.vertices);

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

    CUDA_CHECK(cudaFree(state.params.points));

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
    if (state.data_file.find("uniform") != string::npos) {
        read_data(state.data_file, state.query_file, state);
    } else if (state.data_file.find("sift") != string::npos) {
        read_data_from_sift1m_128d(state.data_file, state.query_file, state);
    } else if (state.data_file.find("gist") != string::npos) {
        read_data_from_gist_960d(state.data_file, state.query_file, state);
    } else {
        cerr << "Invalid data file!" << endl;
        exit(-1);
    }
    
    size_t start;
    start_gpu_mem(&start);
    initialize_optix(state);
    data_h2d(state);
    make_gas(state);                    // Acceleration handling
    make_module(state);
    make_program_groups(state);
    make_pipeline(state);               // Link pipeline
    make_sbt(state);
    initialize_params(state);
    size_t used;
    stop_gpu_mem(&start, &used);
    std::cout << "[Mem] total gpu mem: " << 1.0 * used / (1 << 20) << std::endl;
    search(state);
    result_d2h(state);
    timer.showTime(state.query_num);
    // check(state);
    check_single_thread(state);
    cleanup(state);
    return 0;
}
