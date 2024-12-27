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
                -D CMAKE_BUILD_TYPE={build_type} -D CMAKE_C_COMPILER=/usr/bin/gcc-9 -D CMAKE_CXX_COMPILER=/usr/bin/g++-9 && \
                make')
    os.system(cmd)
        
logtime = time.strftime("%y%m%d-%H%M%S")

# perf(n=575468, W=10000, S=500, K=6, R=1.3, data_file='tao.txt', log_file='tao') # R is obtained by kth graph
# perf(n=575468, W=10000, S=500, K=50, R=1.9, data_file='tao.txt', log_file='tao')
# perf(n=575468, W=10000, S=500, K=50, R=1, data_file='tao.txt', log_file='tao')
# set args -n 575468 -W 10000 -S 500 -K 6 -R 1.3 -f dataset/tao.txt

# perf(n=24876978, W=200000, S=10000, K=6, R=0.01, data_file='geolife.bin', log_file='geolife')
# perf(n=24876978, W=100000, S=5000, K=765, R=0.002, data_file='geolife.bin', log_file='geolife')
# set args -n 24876978 -W 200000 -S 10000 -K 6 -R 0.01 -f dataset/geolife.bin

# perf(n=40000, W=10000, S=500, K=4, R=0.035, data_file='RBF4_40000.csv', log_file='rbf')
# perf(n=40000, W=10000, S=500, K=4, R=0.038237, data_file='RBF4_40000.csv', log_file='rbf')
# set args -n 40000 -W 10000 -S 500 -K 4 -R 0.019607 -f dataset/RBF4_40000.csv

# perf(n=245000, W=10000, S=500, K=4, R=68.516460, data_file='EDS.txt', log_file='eds')

perf(n=1048572, W=100000, S=5000, K=2, R=0.07, data_file='stock.txt', log_file='stk')
# perf(n=1048572, W=100000, S=5000, K=50, R=0.45, data_file='stock.txt', log_file='stk')
# set args -n 1048572 -W 100000 -S 5000 -K 2 -R 0.07 -f dataset/stock.txt
