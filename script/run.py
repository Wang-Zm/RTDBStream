import time
import os
import numpy as np

build_type = 'Release'

def perf_tao(n, W, S, K, R, data_file):
    output_file = f"log/rt_naive/{logtime}-W{W}-S{S}-K{K}-R{R}-tao.log"
    args = f'-n {n} -W {W} -S {S} -K {K} -R {R} -f dataset/{data_file}'
    # -n 575468 -W 10000 -S 500 -K 50 -R 1.9 -f dataset/tao.txt
    cmd = f"./build/bin/optixScan {args} >> {output_file}"
    print(cmd)
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} && \
                make')
    os.system(cmd)
        
logtime = time.strftime("%y%m%d-%H%M%S")
perf_tao(n=575468, W=10000, S=500, K=50, R=1, data_file='tao.txt')