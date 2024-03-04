# from sklearn.datasets import load_iris
# X = load_iris()['data']
# X_1, X_2 = X[:100], X[100:]
# print(X_1)

# from incdbscan import IncrementalDBSCAN
# clusterer = IncrementalDBSCAN(eps=0.5, min_pts=5)

# # Insert 1st batch of data points and get their labels
# clusterer.insert(X_1)
# labels_part1 = clusterer.get_cluster_labels(X_1)

# # Insert 2nd batch and get labels of all points in a one-liner
# labels_all = clusterer.insert(X_2).get_cluster_labels(X)

# # Delete 1st batch and get labels for 2nd batch
# clusterer.delete(X_1)
# labels_part2 = clusterer.get_cluster_labels(X_2)

import time
from incdbscan import IncrementalDBSCAN
import numpy as np

def process_tao():
    clusterer = IncrementalDBSCAN(eps=1, min_pts=50)
    file_path = 'dataset/tao.txt'
    with open(file_path, 'r') as f:
        line = f.readline()
        data = []
        while line:
            line = line.strip('\n')
            record = line.split(',')
            for j in range(len(record)):
                record[j] = float(record[j])
            data.append(record)
            line = f.readline()
        
        window_size = 10000
        stride_size = 500
        n = 575468
        remaining_data_num = n - window_size
        new_stride_pos = window_size
        old_stride_pos = 0
        unit_num = int(window_size / stride_size)
        current_window = data[:window_size]
        # print(current_window)
        
        # First window
        clusterer.insert(current_window)
        print('Initialize first window')
        
        # Start sliding
        slide_num  = 0
        update_pos = 0
        new_stride = data[new_stride_pos : new_stride_pos + stride_size]
        old_stride = data[old_stride_pos : old_stride_pos + stride_size]
        while remaining_data_num >= stride_size and slide_num < 10:
            # Update current window
            start_time = time.time()
            start = update_pos * stride_size
            for i in range(start, start + stride_size):
                current_window[i] = new_stride[i - start]
            end_time = time.time()
            update_window_time = (end_time - start_time) * 1000
            print('[Step] Update current window, time: %.2f' % update_window_time)
            
            start_time = time.time()
            clusterer.delete(old_stride)
            end_time = time.time()
            delete_stride_time = (end_time - start_time) * 1000
            print('[Step] Delete old stride, time: %.2f' % delete_stride_time)
            
            start_time = time.time()
            clusterer.insert(new_stride)
            end_time = time.time()
            insert_stride_time = (end_time - start_time) * 1000
            print('[Step] Insert new stride, time: %.2f', insert_stride_time)
            
            start_time = time.time()
            clusterer.get_cluster_labels(current_window)
            end_time = time.time()
            get_label_time = (end_time - start_time) * 1000
            print('[Step] Cluster, time: %.2f', get_label_time)
            
            
            old_stride_pos += stride_size
            new_stride_pos += stride_size
            slide_num += 1
            remaining_data_num -= stride_size
            update_pos = (update_pos + 1) % unit_num
            
            new_stride = data[new_stride_pos : new_stride_pos + stride_size]
            old_stride = data[old_stride_pos : old_stride_pos + stride_size]
            
            print('Slide:', slide_num)
        
        print('slide_num:', slide_num)
    
def process_geolife():
    clusterer = IncrementalDBSCAN(eps=0.002, min_pts=765)
    file_path = 'dataset/geolife.bin'
    window_size = 100000
    stride_size = 5000
    n = 24876978
    data = np.fromfile(file_path)
    data.shape = n, 3
    
    remaining_data_num = n - window_size
    new_stride_pos = window_size
    old_stride_pos = 0
    unit_num = int(window_size / stride_size)
    current_window = data[:window_size]
    # print(current_window)
    
    # First window
    clusterer.insert(current_window)
    print('Initialize first window')
    
    # Start sliding
    slide_num  = 0
    update_pos = 0
    new_stride = data[new_stride_pos : new_stride_pos + stride_size]
    old_stride = data[old_stride_pos : old_stride_pos + stride_size]
    while remaining_data_num >= stride_size and slide_num < 10:
        # Update current window
        start_time = time.time()
        start = update_pos * stride_size
        for i in range(start, start + stride_size):
            current_window[i] = new_stride[i - start]
        end_time = time.time()
        update_window_time = (end_time - start_time) * 1000
        print('[Step] Update current window, time: %.2f' % update_window_time)
        
        start_time = time.time()
        clusterer.delete(old_stride)
        end_time = time.time()
        delete_stride_time = (end_time - start_time) * 1000
        print('[Step] Delete old stride, time: %.2f' % delete_stride_time)
        
        start_time = time.time()
        clusterer.insert(new_stride)
        end_time = time.time()
        insert_stride_time = (end_time - start_time) * 1000
        print('[Step] Insert new stride, time: %.2f', insert_stride_time)
        
        start_time = time.time()
        clusterer.get_cluster_labels(current_window)
        end_time = time.time()
        get_label_time = (end_time - start_time) * 1000
        print('[Step] Cluster, time: %.2f', get_label_time)
        
        
        old_stride_pos += stride_size
        new_stride_pos += stride_size
        slide_num += 1
        remaining_data_num -= stride_size
        update_pos = (update_pos + 1) % unit_num
        
        new_stride = data[new_stride_pos : new_stride_pos + stride_size]
        old_stride = data[old_stride_pos : old_stride_pos + stride_size]
        
        print('Slide:', slide_num)
    
    print('slide_num:', slide_num)

    
process_geolife()