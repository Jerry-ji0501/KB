import torch
import numpy as np
import torch.utils.data
import os
import scipy.sparse as sp
from data.FeatureDivided import ExtractFeatures

path = os.getcwd()

def  load_feature_data(path):
    dir_path = 'data/feature'
    data_list = []
    for single_dir in os.listdir(dir_path):
        paths =  'data/feature/' + single_dir
        data = ExtractFeatures(paths, single_dir)
        data_list.append(data)
    data_all = np.concatenate(data_list,axis=0)

    return data_all

def load_topo_data(path):
    topo_data = torch.randn(180,1,100,100)
    return topo_data



