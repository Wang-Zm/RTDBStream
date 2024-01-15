import time
import os
import sys, getopt
import numpy as np

build_type = 'Release'

def perf_uniform(n, dim, q, r, data_file, query_file):
    output_file = f"log/rtfrnn/30d/{logtime}-N{n}-D{dim}-Q{q}-R{r}.log"
    args = f'--n {n} --dim {dim} --query_num {q} --radius {r} -f dataset/{data_file} --query_file query/{query_file}'
    # --n 20000 --dim 30 --query_num 100 --radius 1.8 -f dataset/uniform_2e4_30d.dat --query_file query/uniform_1e2_30d.dat
    cmd = f"./build/bin/optixScan {args} >> {output_file}"
    print(cmd)
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} && \
                make')
    os.system(cmd)
    

def perf_uniform_vary_r(n, dim, q, r, data_file, query_file, log_dir, data_width):
    output_file = f"log/rtfrnn/{log_dir}/{logtime}-N{n}-D{dim}-Q{q}-R{r[0]}to{r[-1]}.log"
    for sr in r:
        args = f'--n {n} --dim {dim} --query_num {q} --radius {sr} -f {data_file} --query_file {query_file}'
        # set args --n 100000 --dim 960 --query_num 100 --radius 1 -f dataset/gist/gist_base.fvecs --query_file dataset/gist/gist_query.fvecs
        cmd = f"./build/bin/optixScan {args} >> {output_file}"
        print(cmd)
        os.system(f'cd build/ && cmake ../src/ \
                    -D CMAKE_BUILD_TYPE={build_type} -D DATA_WIDTH={data_width} && \
                    make')
        os.system(cmd)
        

logtime = time.strftime("%y%m%d-%H%M%S")

# perf_uniform(20000, 30, 100, 1.8, 'uniform_2e4_30d.dat', 'uniform_1e2_30d.dat')
# perf_uniform(20000, 30, 1000, 1.8, 'uniform_2e4_30d.dat', 'uniform_1e3_30d.dat')
# perf_uniform(20000, 30, 10000, 1.8, 'uniform_2e4_30d.dat', 'uniform_1e4_30d.dat')
# perf_uniform(50000, 30, 5000, 1.8, data_file='uniform_5e4_30d_float32.dat', query_file='uniform_5e3_30d_float32.dat')

# perf_uniform_vary_r(int(1e5), 3, int(5e3), r=[0.1, 0.2, 0.3],
#                     data_file='dataset/uniform_1e5_3d_float64.dat', 
#                     query_file='query/uniform_5e3_3d_float64.dat', 
#                     log_dir='3d',
#                     data_width=64)

# perf_uniform_vary_r(50000, 30, 5000, r=[1.4, 1.5, 1.8], # 1.5, 1.8, 2.1
#                     data_file='dataset/uniform_5e4_30d_float64.dat', 
#                     query_file='query/uniform_5e3_30d_float64.dat', 
#                     log_dir='30d',
#                     data_width=64)
# perf_uniform_vary_r(50000, 90, 5000, r=[3.2, 3.6, 3.8], 
#                     data_file='dataset/uniform_5e4_90d_float64.dat', 
#                     query_file='query/uniform_5e3_90d_float64.dat', 
#                     log_dir='90d',
#                     data_width=64)
# perf_uniform_vary_r(50000, 300, 5000, r=[6.2, 6.5, 7], # [6, 7, 8]
#                     data_file='dataset/uniform_5e4_300d_float64.dat', 
#                     query_file='query/uniform_5e3_300d_float64.dat', 
#                     log_dir='300d',
#                     data_width=64)

perf_uniform_vary_r(int(1e5), 126, int(1e4), r=[230, 240, 250], # 300, 400, 500
                    data_file='dataset/sift/sift_base.fvecs', 
                    query_file='dataset/sift/sift_query.fvecs', 
                    log_dir='sift1m',
                    data_width=32)

# perf_uniform_vary_r(int(1e5), 960, int(1e3), r=[0.7, 0.8, 0.9], # 0.7, 0.8, 0.9; 1, 1.5, 1.7
#                     data_file='dataset/gist/gist_base.fvecs', 
#                     query_file='dataset/gist/gist_query.fvecs', 
#                     log_dir='gist',
#                     data_width=32)
# set args --n 100000 --dim 960 --query_num 1000 --radius 0.7 -f dataset/gist/gist_base.fvecs --query_file dataset/gist/gist_query.fvecs

