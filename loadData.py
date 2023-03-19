import os
import numpy as np
import torch
import torch.utils.data
import scipy.sparse as sp
from data.FeatureDivided import ExtractFeatures

path = os.getcwd()


def load_feature_data():
    dir_path = os.path.join(os.getcwd(), 'data', 'feature')
    data_list = []
    for single_dir in os.listdir(dir_path):
        file_path = os.path.join(dir_path, single_dir)
        data = ExtractFeatures(file_path, single_dir)
        data_list.append(data)
    data_all = np.concatenate(data_list, axis=0)
    return data_all


def load_topo_data():
    topo_data = torch.randn(180, 1, 100, 100)
    return topo_data
