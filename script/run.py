import time
import os
import numpy as np

build_type = 'Release'

def perf(n, W, S, K, R, data_file, log_file):
    output_file = f"log/{logtime}-W{W}-S{S}-K{K}-R{R}-{log_file}.log"
    args = f'-n {n} -W {W} -S {S} -K {K} -R {R} -f dataset/{data_file}'
    cmd = f"./build/bin/dbscan {args} >> {output_file}"
    print(cmd)
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} -D CMAKE_C_COMPILER=/usr/bin/gcc-8 -D CMAKE_CXX_COMPILER=/usr/bin/g++-8 && \
                make')
    os.system(cmd)

def perf_vary_window_size():
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} -D CMAKE_C_COMPILER=/usr/bin/gcc-9 -D CMAKE_CXX_COMPILER=/usr/bin/g++-9 && \
                make')
    # # stk
    # window_size_list = [50000, 100000, 200000, 400000]
    # output_file = f"log/{logtime}-varyW-S5000-K2-R0.07-stk.log"
    # for w in window_size_list:
    #     args = f'-n 1048572 -W {w} -S 5000 -K 2 -R 0.07 -f dataset/stock.txt'
    #     cmd = f"./build/bin/dbscan {args} >> {output_file}"
    #     print(cmd)
    #     os.system(cmd)
    # # rbf
    # window_size_list = [2500, 5000, 10000, 20000]
    # output_file = f"log/{logtime}-varyW-S500-K4-R0.035-rbf.log"
    # for w in window_size_list:
    #     args = f'-n 40000 -W {w} -S 500 -K 4 -R 0.035 -f dataset/RBF4_40000.csv'
    #     cmd = f"./build/bin/dbscan {args} >> {output_file}"
    #     print(cmd)
    #     os.system(cmd)
    # # tao
    # window_size_list = [5000, 10000, 20000, 40000]
    # output_file = f"log/{logtime}-varyW-S500-K6-R1.3-tao.log"
    # for w in window_size_list:
    #     args = f'-n 575468 -W {w} -S 500 -K 6 -R 1.3 -f dataset/tao.txt'
    #     cmd = f"./build/bin/dbscan {args} >> {output_file}"
    #     print(cmd)
    #     os.system(cmd)
    # # geolife
    # window_size_list = [50000, 100000, 200000, 400000, 800000]
    # output_file = f"log/{logtime}-varyW-S10000-K6-R0.01-geolife.log"
    # for w in window_size_list:
    #     args = f'-n 24876978 -W {w} -S 10000 -K 6 -R 0.01 -f dataset/geolife.bin'
    #     cmd = f"./build/bin/dbscan {args} >> {output_file}"
    #     print(cmd)
    #     os.system(cmd)

    # geolife
    window_size_list = [50000, 100000, 200000, 400000, 800000]
    stride_size_list = [2500, 5000, 10000, 20000, 40000]
    output_file = f"log/{logtime}-varyW-varyS-K6-R0.01-geolife.log"
    for i in range(len(window_size_list)):
        w = window_size_list[i]
        s = stride_size_list[i]
        args = f'-n 24876978 -W {w} -S {s} -K 6 -R 0.01 -f dataset/geolife.bin'
        cmd = f"./build/bin/dbscan {args} >> {output_file}"
        print(cmd)
        os.system(cmd)

def perf_vary_stride_size():
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} -D CMAKE_C_COMPILER=/usr/bin/gcc-9 -D CMAKE_CXX_COMPILER=/usr/bin/g++-9 && \
                make')
    stride_size_list = [2000, 10000, 20000, 50000]
    output_file = f"log/{logtime}-W200000-varyS-K6-R0.01-geolife.log"
    for s in stride_size_list:
        args = f'-n 24876978 -W 200000 -S {s} -K 6 -R 0.01 -f dataset/geolife.bin'
        cmd = f"./build/bin/dbscan {args} >> {output_file}"
        print(cmd)
        os.system(cmd)

def perf_vary_eps():
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} -D CMAKE_C_COMPILER=/usr/bin/gcc-9 -D CMAKE_CXX_COMPILER=/usr/bin/g++-9 && \
                make')
    eps_list = [0.005, 0.015, 0.025, 0.035, 0.045]
    output_file = f"log/{logtime}-W200000-S10000-K6-varyR-geolife.log"
    for eps in eps_list:
        args = f'-n 24876978 -W 200000 -S 10000 -K 6 -R {eps} -f dataset/geolife.bin'
        cmd = f"./build/bin/dbscan {args} >> {output_file}"
        print(cmd)
        os.system(cmd)

def perf_vary_min_pts():
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} -D CMAKE_C_COMPILER=/usr/bin/gcc-9 -D CMAKE_CXX_COMPILER=/usr/bin/g++-9 && \
                make')
    min_pts_list = [2, 4, 6, 8, 10, 12, 14]
    output_file = f"log/{logtime}-W200000-S10000-varyK-R0.01-geolife.log"
    for min_pts in min_pts_list:
        args = f'-n 24876978 -W 200000 -S 10000 -K {min_pts} -R 0.01 -f dataset/geolife.bin'
        cmd = f"./build/bin/dbscan {args} >> {output_file}"
        print(cmd)
        os.system(cmd)

logtime = time.strftime("%y%m%d-%H%M%S")

# perf(n=1048572, W=100000, S=5000, K=2, R=0.07, data_file='stock.txt', log_file='stk')
# perf(n=1048572, W=100000, S=5000, K=50, R=0.45, data_file='stock.txt', log_file='stk')
# set args -n 1048572 -W 100000 -S 5000 -K 2 -R 0.07 -f dataset/stock.txt

# perf(n=40000, W=10000, S=500, K=4, R=0.035, data_file='RBF4_40000.csv', log_file='rbf')
# perf(n=40000, W=10000, S=500, K=4, R=0.038237, data_file='RBF4_40000.csv', log_file='rbf')
# set args -n 40000 -W 10000 -S 500 -K 4 -R 0.019607 -f dataset/RBF4_40000.csv

# perf(n=245000, W=10000, S=500, K=4, R=68.516460, data_file='EDS.txt', log_file='eds')

# perf(n=575468, W=10000, S=500, K=6, R=1.3, data_file='tao.txt', log_file='tao') # R is obtained by kth graph
# perf(n=575468, W=10000, S=500, K=50, R=1.9, data_file='tao.txt', log_file='tao')
# perf(n=575468, W=10000, S=500, K=50, R=1, data_file='tao.txt', log_file='tao')
# set args -n 575468 -W 10000 -S 500 -K 6 -R 1.3 -f dataset/tao.txt

# perf(n=24876978, W=200000, S=10000, K=6, R=0.01, data_file='geolife.bin', log_file='geolife')
# perf(n=24876978, W=100000, S=5000, K=765, R=0.002, data_file='geolife.bin', log_file='geolife')
# set args -n 24876978 -W 200000 -S 10000 -K 6 -R 0.01 -f dataset/geolife.bin

# perf_vary_window_size()
# perf_vary_stride_size()
# perf_vary_eps()
# perf_vary_min_pts()