import time
import os
import numpy as np

build_type = 'Release'

def perf(n, W, S, K, R, data_file, log_file):
    output_file = f"log/{logtime}-W{W}-S{S}-K{K}-R{R}-{log_file}.log"
    args = f'-n {n} -W {W} -S {S} -K {K} -R {R} -f dataset/{data_file}'
    # -n 575468 -W 10000 -S 500 -K 50 -R 1 -f dataset/tao.txt
    cmd = f"./build/bin/dbscan {args} >> {output_file}"
    print(cmd)
    os.system(f'cd build/ && cmake ../src/ \
                -D CMAKE_BUILD_TYPE={build_type} -D CMAKE_C_COMPILER=/usr/bin/gcc-8 -D CMAKE_CXX_COMPILER=/usr/bin/g++-8 && \
                make')
    os.system(cmd)
        
logtime = time.strftime("%y%m%d-%H%M%S")

# perf(n=575468, W=10000, S=500, K=50, R=1, data_file='tao.txt', log_file='tao')
perf(n=575468, W=10000, S=500, K=50, R=1.9, data_file='tao.txt', log_file='tao')
# set args -n 575468 -W 10000 -S 500 -K 50 -R 1.9 -f dataset/tao.txt

# perf(n=24876978, W=100000, S=5000, K=765, R=0.002, data_file='geolife.bin', log_file='geolife')

# perf(n=40000, W=10000, S=500, K=4, R=0.019607, data_file='RBF4_40000.csv', log_file='rbf')
# set args -n 40000 -W 10000 -S 500 -K 4 -R 0.019607 -f dataset/RBF4_40000.csv

# perf(n=1048572, W=100000, S=5000, K=50, R=0.45, data_file='stock.txt', log_file='stk')
# set args -n 24876978 -W 100000 -S 5000 -K 765 -R 0.002 -f dataset/geolife.bin
