import os
import numpy as np
import scipy.io as sio
import pandas as pd
import re


def ad_p(file):  # 计算皮尔逊相关系数
    # 使用scipy.io转换matlab数据为numpy(成功)
    data = sio.loadmat(path + '/Preprocessed_EEG/' + file)
    numpy_list = []
    film_list = []  # record the key of dic
    key_list = list(data.keys())

    for key in key_list:
        film_num = re.findall(r'\d+', key)
        if len(film_num) == 0:
            continue
        else:
            ls = [1, 4, 6, 9]
            num = int(film_num[0])

            if num in ls:
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
                    if matrix[i][j] < 0.7 and i < j:  # 筛选阈值设置为0.1
                        result.append([int(i + 1), int(j + 1), matrix[i][j]])

            # 写入csv文件
            for name in film_list:
                paths = 'ad_pm0.7/' + name
                df = pd.DataFrame(result)
                if os.path.isdir(paths):
                    pass
                else:
                    os.mkdir(paths)
                csv_save_path = 'ad_pm0.7/{}/part{}.csv'.format(name, n)
                df.to_csv(csv_save_path, sep=',', index=False, header=False)


path = os.getcwd()
# list all dir
dir_list = os.listdir(path + '/Preprocessed_EEG')

# deal with film [1,4,6,9] and compute the adj
for doc in dir_list:
    ad_p(doc)
