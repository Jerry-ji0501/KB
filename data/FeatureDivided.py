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





def ExtractFeatures(path, name):
    file_name, file_extension = os.path.splitext(name)
    folder_path = 'features/' + file_name
    #os.makedirs(folder_path)

    # 第一次实验 de_LDS  psd_LDS 235s
    data = io.loadmat(path)
    data1 = list(data.values())[4]
    data2 = list(data.values())[6]

    final_1 = np.zeros([5, 62, 8])
    final_2 = np.zeros([5, 62, 8])

    for i in range(5):
        data = data1[:, :, 0]
        for j in range(7):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_1[i, :, j] = pre
        pre = np.average(data[:, 209:234])
        final_1[i, :, 7] = pre

    for i in range(5):
        data = data2[:, :, 0]
        for j in range(7):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_2[i, :, j] = pre
        pre = np.average(data[:, 209:234])
        final_2[i, :, 7] = pre

    final1 = np.concatenate([final_1, final_2], axis=0)
    print(final1.shape)
    #np.save(folder_path + '/1.npy', final1)

    # 第四次实验 238s
    data = io.loadmat(path)
    data1 = list(data.values())[40]
    data2 = list(data.values())[42]

    final_1 = np.zeros([5, 62, 8])
    final_2 = np.zeros([5, 62, 8])

    for i in range(5):
        data = data1[:, :, 0]
        for j in range(7):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_1[i, :, j] = pre
        pre = np.average(data[:, 209:237])
        final_1[i, :, 7] = pre

    for i in range(5):
        data = data2[:, :, 0]
        for j in range(7):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_2[i, :, j] = pre
        pre = np.average(data[:, 209:237])
        final_2[i, :, 7] = pre

    final2 = np.concatenate([final_1, final_2], axis=0)
    print(final2.shape)
    #np.save(folder_path + '/2.npy', final2)

    # 第六次实验
    data = io.loadmat(path)
    data1 = list(data.values())[64]
    data2 = list(data.values())[66]

    final_1 = np.zeros([5, 62, 7])
    final_2 = np.zeros([5, 62, 7])

    for i in range(5):
        data = data1[:, :, 0]
        for j in range(6):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_1[i, :, j] = pre
        pre = np.average(data[:, 179:194])
        final_1[i, :, 6] = pre

    for i in range(5):
        data = data2[:, :, 0]
        for j in range(6):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_2[i, :, j] = pre
        pre = np.average(data[:, 179:194])
        final_2[i, :, 6] = pre

    final3 = np.concatenate([final_1, final_2], axis=0)
    print(final3.shape)
    #np.save(folder_path + '/3.npy', final3)

    # 第九次实验
    data = io.loadmat(path)
    data1 = list(data.values())[100]
    data2 = list(data.values())[102]

    final_1 = np.zeros([5, 62, 9])
    final_2 = np.zeros([5, 62, 9])

    for i in range(5):
        data = data1[:, :, 0]
        for j in range(8):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_1[i, :, j] = pre
        pre = np.average(data[:, 239:264])
        final_1[i, :, 8] = pre

    for i in range(5):
        data = data2[:, :, 0]
        for j in range(8):
            pre = np.average(data[:, j * 30:30 * (j + 1) - 1], axis=1)
            final_2[i, :, j] = pre
        pre = np.average(data[:, 239:264])
        final_2[i, :, 8] = pre

    final4 = np.concatenate([final_1, final_2], axis=0)
    print(final4.shape)
    #np.save(folder_path + '/4.npy', final4)
    #data_all = np.concatenate((final1,final2,final3,final4),axis=0)
    #print(data_all.shape)
    data = padding_to_max([final1,final2,final3,final4])
    print(data.shape)
    return data



if __name__ == '__main__':
    dir_path = 'feature'
    data_list = []
    for single_dir in os.listdir(dir_path):
        paths =  'feature/' + single_dir
        data = ExtractFeatures(paths, single_dir)
        data_list.append(data)
    data_all = np.concatenate(data_list,axis=0)
    print(data_all.shape)
