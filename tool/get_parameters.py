import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

def plot_kth_graph(data, fig_name):
    plt.plot(data, marker='o', linestyle='-', color='b')

    plt.title(fig_name)
    plt.xlabel("Index")
    plt.ylabel("Value")

    plt.grid(True)

    plt.savefig(fig_name + ".pdf", format="pdf")

    # plt.show()

def calc_epi(matrix, k, noise_ratio, fig_name="Line Chart"):
    rows = matrix.shape[0]
    result = []

    def calculate_kth_distance(i):
        distances = []
        for j in range(rows):
            if i != j:
                distance = np.linalg.norm(matrix[i] - matrix[j])
                distances.append(distance)

        kth_distance = np.partition(distances, k-1)[k-1]
        return kth_distance

    result = Parallel(n_jobs=-1)(delayed(calculate_kth_distance)(i) for i in range(rows))

    result.sort(reverse=True)
    print(result[:5])
    print("epi index=%d" % int(rows * noise_ratio))
    plot_kth_graph(result, fig_name)
    return result[int(rows * noise_ratio)]


def get_params_from_rbf():
    data = np.loadtxt('dataset/RBF4_40000.csv', delimiter=',', skiprows=1, usecols=(0, 1))
    print(data.shape)
    W = 10000
    random_rows = np.random.choice(data.shape[0], size=W, replace=False)
    selected_rows = data[random_rows, :]
    K = 2 * 2 - 1 # from Ester et al
    noise_ratio = 0.01
    epi = calc_epi(selected_rows, K, noise_ratio, "RBF-2d Kth Graph")
    print('RBF: epi=%f' % epi)
    '''
    epi index=100
    RBF: epi=0.038237
    '''

def get_params_from_stk():
    data = np.loadtxt('dataset/stock.txt')
    print(data.shape)
    W = 10000 # 100000 is too slow
    random_rows = np.random.choice(data.shape[0], size=W, replace=False)
    selected_rows = data[random_rows]
    K = 1
    noise_ratio = 0.01
    epi = calc_epi(selected_rows, K, noise_ratio, "STK-1d Kth Graph")
    print('RBF: epi=%f' % epi)
    '''
    epi index=100
    RBF: epi=
    '''
    
def get_params_from_eds():
    data = np.loadtxt('dataset/EDS.txt', delimiter=' ', skiprows=0, usecols=(1, 2))
    print(data.shape)
    W = 10000
    random_rows = np.random.choice(data.shape[0], size=W, replace=False)
    selected_rows = data[random_rows, :]
    K = 3 # from Ester et al
    noise_ratio = 0.01
    epi = calc_epi(selected_rows, K, noise_ratio)
    print('EDS: epi=%f' % epi)
    
    '''
    epi index=100
    RBF: epi=
    '''

def get_params_from_tao():
    data = np.loadtxt('dataset/tao.txt', delimiter=',')
    print(data.shape)
    W = 10000
    random_rows = np.random.choice(data.shape[0], size=W, replace=False)
    selected_rows = data[random_rows, :]
    K = 2 * 3 - 1 # from Ester et al
    noise_ratio = 0.01
    epi = calc_epi(selected_rows, K, noise_ratio, "TAO-3d Kth Graph")
    print('EDS: epi=%f' % epi)
    
    '''
    epi index=100
    RBF: epi=1.247574
    '''

# get_params_from_rbf()
get_params_from_stk()
# get_params_from_eds()
# get_params_from_tao()