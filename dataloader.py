import scipy.io as sio
from sklearn import preprocessing
import os
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt


path = os.getcwd()
eeg_dir = '/data/feature/'
eeg_file_list = os.listdir(path+eeg_dir)
eeg_file_list.sort()

T = 10
fz = 200
channels = 62


for item in eeg_file_list:
    print(item)
    all_data = sio.loadmat(os.path.join(path+eeg_dir,item))
#print(all_data)
film_list = ['de_movingAve1','de_movingAve2','de_movingAve4','de_movingAve6','de_movingAve9']
data = {k:v for k,v in  all_data.items() if k in film_list}
print(data['de_movingAve1'][:,0:10,:].shape)
DEfeature_data_1 = data['de_movingAve1']
DEfeature_data_2 = data['de_movingAve2']
DEfeature_data_4 = data['de_movingAve4']
DEfeature_data_6 = data['de_movingAve6']
DEfeature_data_9 = data['de_movingAve9']
DEfeature_data = [DEfeature_data_1,DEfeature_data_2,DEfeature_data_4,DEfeature_data_6,DEfeature_data_9 ]
DEfeature = []
for item in DEfeature_data:
    result = np.mean(item[:,::10,:],axis=1)
    print(result.shape)
    DEfeature.append(result)


#film_list = ['djc_eeg1','djc_eeg2','djc_eeg4','djc_eeg6','djc_eeg9']
#data = {k:v for k,v in all_data.items() if k in film_list}
#print(type(data['djc_eeg1']))
#Brain_segment = np.arr_split(data['djc_eeg1'],200)
#print(len(Brain_segment))
#TimeSnap = 10
#data_list = []
#data_list.extend(data.items())
#print(data_list[0].shape)
#Input feature X.dimension:T*N*F*f   T是划分的脑图数 N是脑图结点数 F是结点特征数 f=5是五个不同的频段
#现在的DE feature 是 N*t*f
