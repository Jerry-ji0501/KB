import scipy.io
import numpy as np
import os

T = [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
N = 10


def Experiment(path, item, folder_path):
    # 第一次实验 de_LDS  psd_LDS 235s
    data = scipy.io.loadmat(path)
    data1 = list(data.values())[4 + item * 12]
    data2 = list(data.values())[6 + item * 12]

    final_1 = np.zeros([5, 62, int(T[item] / N) + 1])
    final_2 = np.zeros([5, 62, int(T[item] / N) + 1])

    for i in range(5):
        data = data1[:, :, 0]
        for j in range(int(T[item] / N)):
            pre = np.average(data[:, j * N: N * (j + 1) - 1], axis=1)
            final_1[i, :, j] = pre
        pre = np.average(data[:, T[item] - T[item] % 10 - 1:T[item] - 1])
        final_1[i, :, int(T[item] / N)] = pre

    for i in range(5):
        data = data2[:, :, 0]
        for j in range(int(T[item] / N)):
            pre = np.average(data[:, j * N: N * (j + 1) - 1], axis=1)
            final_2[i, :, j] = pre
        pre = np.average(data[:, T[item] - T[item] % 10 - 1:T[item] - 1])
        final_2[i, :, item] = pre

    final1 = np.concatenate([final_1, final_2], axis=0)
    np.save(folder_path + '/' + '{}.npy'.format(item), final1)


def ExtractFeatures(path, name):
    file_name, file_extension = os.path.splitext(name)
    folder_path = './data/features/' + file_name

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    item = 0
    for i in range(15):
        Experiment(path, item, folder_path)
        item += 1


if __name__ == '__main__':
    dir_path = './data/ExtractedFeatures'
    for single_dir in os.listdir(dir_path):
        paths = './data/ExtractedFeatures/' + single_dir
        ExtractFeatures(paths, single_dir)
