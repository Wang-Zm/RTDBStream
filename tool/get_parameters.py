import numpy as np
from joblib import Parallel, delayed

def calc_epi(matrix, k, noise_ratio):
    rows, cols = matrix.shape
    result = []

    # for i in range(rows):
    #     distances = []
    #     for j in range(rows):
    #         if i != j:
    #             distance = np.linalg.norm(matrix[i] - matrix[j])
    #             distances.append(distance)

    #     kth_distance = np.partition(distances, k-1)[k-1]  # 第 k 大距离
    #     result.append(kth_distance)
    #     if i % 500 == 0:
    #         print("i=%d" % i)
    
    def calculate_kth_distance(i):
        distances = []
        for j in range(rows):
            if i != j:
                distance = np.linalg.norm(matrix[i] - matrix[j])
                distances.append(distance)

        kth_distance = np.partition(distances, k-1)[k-1]  # 第 k 小距离
        return kth_distance

    # 使用 joblib 并行计算
    result = Parallel(n_jobs=-1)(delayed(calculate_kth_distance)(i) for i in range(rows))

    result.sort(reverse=True)
    print("epi index=%d" % int(rows * noise_ratio))
    return result[int(rows * noise_ratio)]


def get_params_from_rbf():
    data = np.loadtxt('dataset/RBF4_40000.csv', delimiter=',', skiprows=1, usecols=(0, 1))
    print(data.shape)
    W = 10000
    # 随机获取若干行
    random_rows = np.random.choice(data.shape[0], size=W, replace=False)
    # 提取随机行
    selected_rows = data[random_rows, :]
    K = 4 # from Ester et al
    noise_ratio = 0.05
    epi = calc_epi(selected_rows, K, noise_ratio)
    print('RBF: epi=%f' % epi)
    
    '''
    epi index=500
    RBF: epi=0.019607
    '''

# * 暂时参照 outlier 中的设置
# def get_params_from_stk():
#     data = np.loadtxt('dataset/stock.txt')
#     print(data.shape)
#     W = 100000
#     # 随机获取若干行
#     random_rows = np.random.choice(data.shape[0], size=W, replace=False)
#     # 提取随机行
#     selected_rows = data[random_rows, :]
#     K = 4 # from Ester et al
#     noise_ratio = 0.05
#     epi = calc_epi(selected_rows, K, noise_ratio)
#     print('RBF: epi=%f' % epi)
    
    
get_params_from_rbf()
# get_params_from_stk()