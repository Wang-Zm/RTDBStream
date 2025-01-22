import numpy as np
import os

dirt = '/home/wzm/Geolife Trajectories 1.3/Data/'
person = '000'
data = []
for i in range(182):
    person = '%03d' % i
    person_dir = dirt + person + '/Trajectory/'
    print(person_dir)
    for filepath, dirnames, filenames in os.walk(person_dir):
        for filename in filenames:
            fp = person_dir + filename
            d = np.loadtxt(fp, delimiter=',', skiprows=6, usecols=(0, 1, 3, 4))
            data.extend(d)
            
print('record_num =', len(data))
for i in range(len(data)):
    data[i][2] /= 300000
    
def takeSecond(elem):
    return elem[3]

data = np.array(data)
# data.sort(key=takeSecond)
data = data[data[:, 3].argsort()]
data = data[:, :3]
print(data.shape)
# np.savetxt('dataset/geolife.txt', data, delimiter=',')
data.tofile('dataset/geolife.bin')