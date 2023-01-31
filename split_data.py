import scipy.io
import numpy as np
import scipy.io
import pandas as pd
import os


def adjacent_matrix(file):
    # 使用scipy.io转换matlab数据为numpy(成功)
    (filename, extension) = os.path.splitext(file)
    matlab_data = scipy.io.loadmat('data/Preprocessed_EEG/' + file)
    key_list_all = list(matlab_data.keys())

    key_list = [key_list_all[3], key_list_all[6], key_list_all[8], key_list_all[11]]

    for key in key_list:
        numpy_array = matlab_data[key]
        divide_list = []
        for m in range(int(list(numpy_array.shape)[1] / 6000)):
            divide_list.append(6000 * (m + 1))
        part = np.hsplit(numpy_array, divide_list)

        length = len(part)
        for n in range(length):
            # 计算协方差矩阵
            matrix = np.corrcoef(part[n])

            # 筛选邻接矩阵数据
            result = []
            for i in range(62):
                for j in range(62):
                    if 0 < matrix[i][j] < 0.1 and i < j:  # 筛选阈值设置为0.1
                        result.append([int(i + 1), int(j + 1), matrix[i][j]])

            # 写入csv文件
            path1 = 'data/adjacent_matrix/pm0.1/' + filename
            path2 = 'data/adjacent_matrix/pm0.1/' + filename + '/' + key
            df = pd.DataFrame(result)
            if os.path.isdir(path1):
                pass
            else:
                os.mkdir(path1)

            if os.path.isdir(path2):
                pass
            else:
                os.mkdir(path2)

            csv_save_path = 'data/adjacent_matrix/pm0.1/{}/{}/part{}.csv'.format(filename, key, n)
            df.to_csv(csv_save_path, sep=',', index=False, header=False)


if __name__ == '__main__':
    path = os.getcwd()
    # list all dir
    dir_list = os.listdir(path + '/data/Preprocessed_EEG')

    # 依次处理1469
    for doc in dir_list:
        adjacent_matrix(doc)
