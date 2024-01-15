import numpy as np
import sys

# data_num = '1e7'
# N = int(float(data_num))
# dim = 4
# query_num = '1e4'
# Q = int(float(query_num))
# data = np.random.uniform(size=(N, dim))
# queries = np.random.uniform(size=(Q, dim))
# print(f'./dataset/uniform_{data_num}_{dim}d.dat')
# data.astype('float64').tofile(f'./dataset/uniform_{data_num}_{dim}d.dat')
# print(f'./query/uniform_{query_num}_{dim}d.dat')
# queries.astype('float64').tofile(f'./query/uniform_{query_num}_{dim}d.dat')


# data_num = '5e4'
# N = int(float(data_num))
# dim = 30
# query_num = '5e3'
# Q = int(float(query_num))
# data = np.random.uniform(size=(N, dim))
# queries = np.random.uniform(size=(Q, dim))
# print(f'./dataset/uniform_{data_num}_{dim}d_float32.dat')
# data.astype('float32').tofile(f'./dataset/uniform_{data_num}_{dim}d_float32.dat')
# print(f'./query/uniform_{query_num}_{dim}d_float32.dat')
# queries.astype('float32').tofile(f'./query/uniform_{query_num}_{dim}d_float32.dat')

data_num = '5e4'
N = int(float(data_num))
dim = 900
query_num = '5e3'
Q = int(float(query_num))
data = np.random.uniform(size=(N, dim))
queries = np.random.uniform(size=(Q, dim))
print(f'./dataset/uniform_{data_num}_{dim}d_float64.dat')
data.astype('float64').tofile(f'./dataset/uniform_{data_num}_{dim}d_float64.dat')
print(f'./query/uniform_{query_num}_{dim}d_float64.dat')
queries.astype('float64').tofile(f'./query/uniform_{query_num}_{dim}d_float64.dat')