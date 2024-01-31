#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <thread>
#include <queue>
#include "state.h"
#include "timer.h"
using namespace std;

void read_data_from_tao(string& data_file, ScanState &state) {
    state.h_data = (DATA_TYPE_3*) malloc(state.data_num * sizeof(DATA_TYPE_3));
    state.max_value.resize(3);
    state.min_value.resize(3);

    ifstream fin;
    string line;
    fin.open(data_file, ios::in);
    if (!fin.is_open()) {
        std::cerr << "Fail to open [" << data_file << "]!" << std::endl;
    }
    for (int dim_id = 0; dim_id < 3; dim_id++) {
        state.max_value[dim_id] = -FLT_MAX;
        state.min_value[dim_id] = FLT_MAX;
    }
    for (int rid = 0; rid < state.data_num; rid++) {
        getline(fin, line);
        sscanf(line.c_str(), "%lf,%lf,%lf", &state.h_data[rid].x, &state.h_data[rid].y, &state.h_data[rid].z);
        if (state.max_value[0] < state.h_data[rid].x) {
            state.max_value[0] = state.h_data[rid].x;
        }
        if (state.min_value[0] > state.h_data[rid].x) {
            state.min_value[0] = state.h_data[rid].x;
        }

        if (state.max_value[1] < state.h_data[rid].y) {
            state.max_value[1] = state.h_data[rid].y;
        }
        if (state.min_value[1] > state.h_data[rid].y) {
            state.min_value[1] = state.h_data[rid].y;
        }

        if (state.max_value[2] < state.h_data[rid].z) {
            state.max_value[2] = state.h_data[rid].z;
        }
        if (state.min_value[2] > state.h_data[rid].z) {
            state.min_value[2] = state.h_data[rid].z;
        }
    }
    fin.close();

    for (int i = 0; i < 3; i++) {
        std::cout << "DIM[" << i << "]: " << state.min_value[i] << ", " << state.max_value[i] << std::endl;
    }
    // printf("h_data[0] = {%lf, %lf, %lf}\n", state.h_data[0].x, state.h_data[0].y, state.h_data[0].z);
    // printf("h_data[window_size-1] = {%lf, %lf, %lf}\n", state.h_data[state.window_size-1].x, state.h_data[state.window_size-1].y, state.h_data[state.window_size-1].z);
}

size_t get_cpu_memory_usage() {
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];
    while (fgets(line, 128, file) != nullptr) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            int len = strlen(line);
            const char* p = line;
            for (; std::isdigit(*p) == false; ++p) {}
            line[len - 3] = 0;
            result = atoi(p);
            break;
        }
    }
    fclose(file);
    return result; // KB
}

void start_gpu_mem(size_t* avail_mem) {
    size_t total_gpu_mem;
    CUDA_CHECK(cudaMemGetInfo( avail_mem, &total_gpu_mem ));
}

void stop_gpu_mem(size_t* avail_mem, size_t* used) {
    size_t total_gpu_mem, avail_mem_now;
    CUDA_CHECK(cudaMemGetInfo( &avail_mem_now, &total_gpu_mem ));
    *used = *avail_mem - avail_mem_now;
}

int find(int x, int* cid) {
    return cid[x] == x ? x : cid[x] = find(cid[x], cid);
}

void cluster_with_cpu(ScanState &state, Timer &timer) {
    // 1. 查找所有点的邻居，判别是否是 core，打上 core label
    int *check_nn = state.check_h_nn; 
    int *check_label = state.check_h_label;
    int *check_cluster_id = state.check_h_cluster_id;
    DATA_TYPE_3 *window = state.h_window;
    
    timer.startTimer(&timer.cpu_cluter_total);
    memset(check_nn, 0, state.window_size * sizeof(int));
    for (int i = 0; i < state.window_size; i++) {
        check_nn[i]++; // 自己
        for (int j = i + 1; j < state.window_size; j++) {
            DATA_TYPE_3 O = { window[i].x - window[j].x, 
                              window[i].y - window[j].y, 
                              window[i].z - window[j].z };
            DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
            if (d < state.params.radius2) {
                check_nn[i]++;
                check_nn[j]++;
            }
        }
    }
    for (int i = 0; i < state.window_size; i++) {
        if (check_nn[i] >= state.min_pts) check_label[i] = 0;
        else check_label[i] = 2;
    }

    // 2. 从一个 core 开始 bfs，设置所有的点是该 core_id
    bool *vis = (bool*) malloc(state.window_size * sizeof(bool));
    memset(vis, false, state.window_size * sizeof(bool));
    queue<int> q;
    for (int i = 0; i < state.window_size; i++) {
        if (vis[i] || check_label[i] != 0) continue; // 对于 border，应该特殊处理：1）不加到 queue 中
        vis[i] = true;
        check_cluster_id[i] = i;
        q.push(i);
        while (!q.empty()) {
            DATA_TYPE_3 p = window[q.front()];
            q.pop();
            for (int j = 0; j < state.window_size; j++) {
                if (!vis[j]) {
                    DATA_TYPE_3 O = { p.x - window[j].x, 
                                      p.y - window[j].y, 
                                      p.z - window[j].z };
                    DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
                    if (d < state.params.radius2) {
                        vis[j] = true;
                        check_cluster_id[j] = i;
                        if (check_label[j] == 0) q.push(j);
                        else check_label[j] = 1; // border
                    }
                }
            }
        }
    }
    free(vis);
    timer.stopTimer(&timer.cpu_cluter_total);
}

void cluster_with_cuda(ScanState &state, Timer &timer) {
    CUDA_CHECK(cudaMalloc(&state.params.check_nn, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.check_label, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.params.check_cluster_id, state.window_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(state.params.check_nn, 0, state.window_size * sizeof(int)));

    timer.startTimer(&timer.cuda_cluter_total);

    timer.startTimer(&timer.cuda_find_neighbors);
    find_neighbors(state.params.check_nn, state.params.window, state.window_size, state.params.radius2, state.min_pts);
    CUDA_SYNC_CHECK();
    find_cores(state.params.check_label, state.params.check_nn, state.params.check_cluster_id, state.window_size, state.min_pts);
    CUDA_SYNC_CHECK();
    timer.stopTimer(&timer.cuda_find_neighbors);

    timer.startTimer(&timer.cuda_set_clusters);
    set_cluster_id(state.params.check_nn, state.params.check_label, state.params.check_cluster_id, state.params.window, state.window_size, state.params.radius2);
    CUDA_SYNC_CHECK();
    timer.stopTimer(&timer.cuda_set_clusters);

    CUDA_CHECK(cudaMemcpy(state.check_h_nn, state.params.check_nn, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.check_h_label, state.params.check_label, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.check_h_cluster_id, state.params.check_cluster_id, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < state.window_size; i++) {
        if (state.check_h_label[i] == 2) continue;
        find(i, state.check_h_cluster_id);
    }
    timer.stopTimer(&timer.cuda_cluter_total);

    CUDA_CHECK(cudaFree(state.params.check_nn));
    CUDA_CHECK(cudaFree(state.params.check_label));
    CUDA_CHECK(cudaFree(state.params.check_cluster_id));
}

bool check(ScanState &state, int window_id, Timer &timer) {
    state.check_h_nn = (int*) malloc(state.window_size * sizeof(int)); 
    state.check_h_label = (int*) malloc(state.window_size * sizeof(int));
    state.check_h_cluster_id = (int*) malloc(state.window_size * sizeof(int));
    // cluster_with_cpu(state, timer);
    cluster_with_cuda(state, timer);
    
    // 将 gpu 中的结果传回来
    state.h_nn = (int*) malloc(state.window_size * sizeof(int));
    CUDA_CHECK(cudaMemcpy(state.h_nn, state.params.nn, state.window_size * sizeof(int), cudaMemcpyDeviceToHost));

    int *nn = state.h_nn, *check_nn = state.check_h_nn;
    int *label = state.h_label, *check_label = state.check_h_label;
    int *cid = state.h_cluster_id, *check_cid = state.check_h_cluster_id;
    for (int i = 0; i < state.window_size; i++) {
        if (nn[i] != check_nn[i]) {
            printf("Error on window %d: nn[%d] = %d, check_nn[%d] = %d\n", 
                    window_id, i, state.h_nn[i], i, state.check_h_nn[i]);
            return false;
        }
    }
    for (int i = 0; i < state.window_size; i++) {
        if (label[i] != check_label[i]) {
            printf("Error on window %d: label[%d] = %d, check_label[%d] = %d; nn[%d] = %d, check_nn[%d] = %d\n", 
                    window_id, i, label[i], i, check_label[i], i, state.h_nn[i], i, state.check_h_nn[i]);
            return false;
        }
    }
    for (int i = 0; i < state.window_size; i++) {
        if (label[i] == 0) {
            if (cid[i] != check_cid[i]) {
                printf("Error on window %d: cid[%d] = %d, check_cid[%d] = %d; "
                       "label[%d] = %d, check_label[%d] = %d; "
                       "nn[%d] = %d, check_nn[%d] = %d\n", 
                        window_id, i, cid[i], i, check_cid[i], 
                        i, label[i], i, check_label[i], 
                        i, nn[i], i, check_nn[i]);
                return false;
            }
        } else if (label[i] == 1) {
            DATA_TYPE_3 p = state.h_window[i];
            bool is_correct = false;
            for (int j = 0; j < state.window_size; j++) {
                if (j == i) continue;
                DATA_TYPE_3 O = {p.x - state.h_window[j].x, p.y - state.h_window[j].y, p.z - state.h_window[j].z};
                DATA_TYPE d = O.x * O.x + O.y * O.y + O.z * O.z;
                if (d < state.params.radius2) {
                    if (cid[j] == cid[i]) { // 验证成功
                        is_correct = true;
                        break;
                    }
                }
            }
            if (!is_correct) { // border 的 label 错误，打印问题
                printf("Error on window %d: cid[%d] = %d, but border[%d] doesn't have a core belonging to cluster %d\n", 
                        window_id, i, cid[i], i, cid[i]);
                return false;
            }
        }
    }
    free(state.check_h_nn);
    free(state.check_h_label);
    free(state.check_h_cluster_id);
    free(state.h_nn);
    return true;
}