from scipy import io
import numpy as np
import os

def pad_last_dim(array,max_dim):
    last_dim = array.shape[-1]
    padding_len = max_dim-last_dim
    return np.pad(array,((0,0),(0,0),(0,padding_len)),mode='constant')

def padding_to_max(Feature_list):
    max_dim = max(array.shape[-1] for array in Feature_list)
    padding_list = [[pad_last_dim(array,max_dim) for array in Feature_list]]
    padding_array = np.concatenate(padding_list)
    return padding_array


def load_feature(file):
    fir_dir_list = os.listdir(file)
    data_list = []
    for fir_dir in fir_dir_list:
        sec_dir_list = os.listdir(file + fir_dir)
        for sec_dir in sec_dir_list:
            data = np.load(file + '/' + fir_dir + '/' + sec_dir)
            data_list.append(data)
    data_list = padding_to_max(data_list)
    data_all = np.stack(data_list)
    
    
    return data_all       
            
        
    

