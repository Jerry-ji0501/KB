import os
import numpy as np
import scipy.io as sio
import pandas as pd
import re


def ad_p(file):  # 计算皮尔逊相关系数
    # 使用scipy.io转换matlab数据为numpy(成功)
    (filename, extension) = os.path.splitext(file)
    data = sio.loadmat(path + '/data/Preprocessed_EEG/' + file)
    numpy_list = []
    film_list = []  # record the key of dic
    key_list = list(data.keys())

    for key in key_list:
        film_num = re.findall(r'\d+', key)
        if len(film_num) == 0:
            continue
        else:
            film_list.append(key)
            numpy_list.append(data[key])

    for numpy_array in numpy_list:
        # 分割数组
        decide_list = []

        for m in range(int(list(numpy_array.shape)[1] / 2000)):
            decide_list.append(2000 * (m + 1))
        part = np.hsplit(numpy_array, decide_list)

        length = len(part)
        for n in range(length):

            # 计算协方差矩阵
            matrix = np.corrcoef(part[n])

            # 筛选邻接矩阵数据
            result = []
            for i in range(62):
                for j in range(62):
                    if matrix[i][j] > 0.8 and i < j:  # 筛选阈值设置为0.1
                        result.append([int(i + 1), int(j + 1), matrix[i][j]])

            # 写入csv文件
            for name in film_list:
                path1 = 'F:/ResearchData/Demo/data/adjacent_matrix/pm0.8/' + filename
                path2 = 'F:/ResearchData/Demo/data/adjacent_matrix/pm0.8/' + filename + '/' + name
                df = pd.DataFrame(result)
                if os.path.isdir(path1):
                    pass
                else:
                    os.makedirs(path1)
                if os.path.isdir(path2):
                    pass
                else:
                    os.makedirs(path2)
                csv_save_path = 'F:/ResearchData/Demo/data/adjacent_matrix/pm0.8/{}/{}/part{}.csv'.format(filename, name, n)
                df.to_csv(csv_save_path, sep=',', index=False, header=False)


path = os.getcwd()
# list all dir
dir_list = os.listdir(path + '/data/Preprocessed_EEG')
print(dir_list)
# deal with film [1,4,6,9] and compute the adj
for doc in dir_list:
    ad_p(doc)
