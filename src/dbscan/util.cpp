#include <sutil/Exception.h>
#include <sutil/sutil.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <thread>
#include "state.h"
using namespace std;

void read_data(string& data_file, string& query_file, ScanState &state) {
    state.vertices = (DATA_TYPE**) malloc(state.data_num * sizeof(DATA_TYPE*));
    state.h_queries = (DATA_TYPE**) malloc(state.data_num * sizeof(DATA_TYPE*));
    for (int i = 0; i < state.data_num; i++) {
        state.vertices[i] = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
        state.h_queries[i] = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
    }
    state.max_value = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
    state.min_value = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));

    ifstream fin;
    fin.open(data_file, ios::in);
    if (!fin.is_open()) {
        std::cerr << "Fail to open [" << data_file << "]!" << std::endl;
    }
    for (int dim_id = 0; dim_id < state.dim; dim_id++) {
        state.max_value[dim_id] = -FLT_MAX;
        state.min_value[dim_id] = FLT_MAX;
    }
    for (int rid = 0; rid < state.data_num; rid++) {
        for (int j = 0; j < state.dim; j++) {
            fin.read((char *)&state.vertices[rid][j], sizeof(DATA_TYPE));
            streamsize read_count = fin.gcount();
            if (read_count <= 0) {
                fin.close();
                std::cout << "Fail to read data file: " << data_file << std::endl;
                std::cerr << "Fail to read data file: " << data_file << std::endl;
                exit(-1);
            }
            if (state.max_value[j] < state.vertices[rid][j]) {
                state.max_value[j] = state.vertices[rid][j];
            }
            if (state.min_value[j] > state.vertices[rid][j]) {
                state.min_value[j] = state.vertices[rid][j];
            }
        }
    }
    fin.close();

    // for (int i = 0; i < state.dim; i++) {
    //     std::cout << "DIM[" << i << "]: " << state.min_value[i] << ", " << state.max_value[i] << std::endl;
    // }

    // Read queries
    fin.open(query_file, ios::in);
    if (!fin.is_open()) {
        std::cerr << "Fail to open [" << query_file << "]!" << std::endl;
    }
    for (int rid = 0; rid < state.query_num; rid++) {
        for (int j = 0; j < state.dim; j++) {
            fin.read((char *)&state.h_queries[rid][j], sizeof(DATA_TYPE));
            streamsize read_count = fin.gcount();
            if (read_count <= 0) {
                fin.close();
                std::cout << "Fail to read query file: " << query_file << std::endl;
                std::cerr << "Fail to read query file: " << query_file << std::endl;
                exit(-1);
            }
        }
    }
    fin.close();
}

void read_data_from_sift1m_128d(string& data_file, string& query_file, ScanState &state) {
    state.vertices = (DATA_TYPE**) malloc(state.data_num * sizeof(DATA_TYPE*));
    state.h_queries = (DATA_TYPE**) malloc(state.data_num * sizeof(DATA_TYPE*));
    for (int i = 0; i < state.data_num; i++) {
        state.vertices[i] = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
        state.h_queries[i] = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
    }
    state.max_value = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
    state.min_value = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));

    ifstream fin;
    fin.open(data_file, ios::in|ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Fail to open [" << data_file << "]!" << std::endl;
        exit(-1);
    }
    for (int dim_id = 0; dim_id < state.dim; dim_id++) {
        state.max_value[dim_id] = -FLT_MAX;
        state.min_value[dim_id] = FLT_MAX;
    }
    for (int rid = 0; rid < state.data_num; rid++) {
        int d;
        fin.read((char *)&d, sizeof(int)); // read dim
        for (int j = 0; j < state.dim; j++) {
            fin.read((char *)&state.vertices[rid][j], sizeof(DATA_TYPE));
            streamsize read_count = fin.gcount();
            if (read_count <= 0) {
                fin.close();
                std::cout << "Fail to read data file: " << data_file << std::endl;
                std::cerr << "Fail to read data file: " << data_file << std::endl;
                exit(-1);
            }
            if (state.max_value[j] < state.vertices[rid][j]) {
                state.max_value[j] = state.vertices[rid][j];
            }
            if (state.min_value[j] > state.vertices[rid][j]) {
                state.min_value[j] = state.vertices[rid][j];
            }
        }
        DATA_TYPE tmp_data;
        for (int j = state.dim; j < 128; j++) {
            fin.read((char *)&tmp_data, sizeof(DATA_TYPE));
        }
    }
    fin.close();

    // Read queries
    fin.open(query_file, ios::in|ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Fail to open [" << query_file << "]!" << std::endl;
        exit(-1);
    }
    for (int rid = 0; rid < state.query_num; rid++) {
        int d;
        fin.read((char *)&d, sizeof(int)); // read dim
        for (int j = 0; j < state.dim; j++) {
            fin.read((char *)&state.h_queries[rid][j], sizeof(DATA_TYPE));
            streamsize read_count = fin.gcount();
            if (read_count <= 0) {
                fin.close();
                std::cout << "Fail to read query file: " << query_file << std::endl;
                std::cerr << "Fail to read query file: " << query_file << std::endl;
                exit(-1);
            }
        }
        DATA_TYPE tmp_data;
        for (int j = state.dim; j < 128; j++) {
            fin.read((char *)&tmp_data, sizeof(DATA_TYPE));
        }
    }
    fin.close();
}

void read_data_from_gist_960d(string& data_file, string& query_file, ScanState &state) {
    state.vertices = (DATA_TYPE**) malloc(state.data_num * sizeof(DATA_TYPE*));
    state.h_queries = (DATA_TYPE**) malloc(state.data_num * sizeof(DATA_TYPE*));
    for (int i = 0; i < state.data_num; i++) {
        state.vertices[i] = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
        state.h_queries[i] = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
    }
    state.max_value = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));
    state.min_value = (DATA_TYPE*) malloc(state.dim * sizeof(DATA_TYPE));

    ifstream fin;
    fin.open(data_file, ios::in|ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Fail to open [" << data_file << "]!" << std::endl;
        exit(-1);
    }
    for (int dim_id = 0; dim_id < state.dim; dim_id++) {
        state.max_value[dim_id] = -FLT_MAX;
        state.min_value[dim_id] = FLT_MAX;
    }
    for (int rid = 0; rid < state.data_num; rid++) {
        int d;
        fin.read((char *)&d, sizeof(int)); // read dim
        for (int j = 0; j < state.dim; j++) {
            fin.read((char *)&state.vertices[rid][j], sizeof(DATA_TYPE));
            streamsize read_count = fin.gcount();
            if (read_count <= 0) {
                fin.close();
                std::cout << "Fail to read data file: " << data_file << std::endl;
                std::cerr << "Fail to read data file: " << data_file << std::endl;
                exit(-1);
            }
            if (state.max_value[j] < state.vertices[rid][j]) {
                state.max_value[j] = state.vertices[rid][j];
            }
            if (state.min_value[j] > state.vertices[rid][j]) {
                state.min_value[j] = state.vertices[rid][j];
            }
        }
    }
    fin.close();

    // Read queries
    fin.open(query_file, ios::in|ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Fail to open [" << query_file << "]!" << std::endl;
        exit(-1);
    }
    for (int rid = 0; rid < state.query_num; rid++) {
        int d;
        fin.read((char *)&d, sizeof(int)); // read dim
        for (int j = 0; j < state.dim; j++) {
            fin.read((char *)&state.h_queries[rid][j], sizeof(DATA_TYPE));
            streamsize read_count = fin.gcount();
            if (read_count <= 0) {
                fin.close();
                std::cout << "Fail to read query file: " << query_file << std::endl;
                std::cerr << "Fail to read query file: " << query_file << std::endl;
                exit(-1);
            }
        }
    }
    fin.close();
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

void check(ScanState &state) {
    int *queries_neighbor_num_check = (int *) malloc(state.query_num * sizeof(int));
    std::thread threads[THREAD_NUM];
    for (int k = 0; k * THREAD_NUM < state.query_num; k++) {
        for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < state.query_num; j++) {
            int query_idx = k * THREAD_NUM + j;
            threads[j] = std::thread([&state, queries_neighbor_num_check, query_idx](){
                int neighbor_num = 0;
                for (int j = 0; j < state.data_num; j++) {
                    DIST_TYPE dist = 0.0;
                    for (int k = 0; k < state.dim; k++) {
                        dist += DIST_TYPE(state.h_queries[query_idx][k] - state.vertices[j][k]) * (state.h_queries[query_idx][k] - state.vertices[j][k]);
                        if (dist >= state.params.radius2) break;
                    }
                    if (dist < state.params.radius2) {
                        neighbor_num++;
                    }
                }
                queries_neighbor_num_check[query_idx] = neighbor_num;
                if (state.queries_neighbor_num[query_idx] != queries_neighbor_num_check[query_idx]) {
                    printf("Error for Query[%d]: Test(%d), Correct(%d)\n", query_idx, state.queries_neighbor_num[query_idx], queries_neighbor_num_check[query_idx]);
                    fprintf(stderr, "Error for Query[%d]: Test(%d), Correct(%d)\n", query_idx, state.queries_neighbor_num[query_idx], queries_neighbor_num_check[query_idx]);
                    exit(-1);
                }
            });
        }
        for (int j = 0; j < THREAD_NUM && (k * THREAD_NUM + j) < state.query_num; j++) {
            threads[j].join();
        }
    }
    free(queries_neighbor_num_check);
    std::cout << "Check Done!" << std::endl;
    std::cerr << "Check Done!" << std::endl;
}

void check_single_thread(ScanState &state) {
    int queries_neighbor_num_check[state.query_num];
    for (int i = 0; i < state.query_num; i++) {
        int neighbor_num = 0;
        for (int j = 0; j < state.data_num; j++) {
            DIST_TYPE dist = 0.0;
            for (int k = 0; k < state.dim; k++) {
                dist += DIST_TYPE(state.h_queries[i][k] - state.vertices[j][k]) * (state.h_queries[i][k] - state.vertices[j][k]);
                if (dist >= state.params.radius2) break;
            }
            if (dist < state.params.radius2) {
                neighbor_num++;
            }
        }
        queries_neighbor_num_check[i] = neighbor_num;
        if (state.queries_neighbor_num[i] != queries_neighbor_num_check[i]) {
            std::cout << "Error for Query[" << i << "]: Test(" << state.queries_neighbor_num[i] 
                      << "), Correct(" << queries_neighbor_num_check[i] <<")" << std::endl;
            
            std::cerr << "Error for Query[" << i << "]: Test(" << state.queries_neighbor_num[i] 
                      << "), Correct(" << queries_neighbor_num_check[i] <<")" << std::endl;
            exit(-1);
        }
    }
    std::cout << "Check Done!" << std::endl;
    std::cerr << "Check Done!" << std::endl;
}

// int main() {
//     string data_file = "/home/wzm/rnn/rtfrnn_hd/dataset/siftsmall/siftsmall_query.fvecs";
//     ifstream fin;
//     fin.open(data_file, ios::in|ios::binary);
//     if (!fin.is_open()) {
//         std::cerr << "Fail to open [" << data_file << "]!" << std::endl;
//     }
//     int d;
//     fin.read((char *)&d, sizeof(int));
//     cout << d << endl;
//     fin.close();
//     return 0;
// }