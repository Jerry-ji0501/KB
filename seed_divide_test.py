import os
import numpy as np
import scipy.io as sio
import pandas as pd
import re



def adjacent_matrix(file):
    # 使用scipy.io转换matlab数据为numpy(成功)
    data = sio.loadmat(path + '/Preprocessed_EEG/' + file)
    numpy_list = []
    film_list = []  # record the key of dic
    key_list = list(data.keys())
    print(key_list)
    for key in key_list:
        film_num = re.findall(r'\d+', key)
        if len(film_num)==0:
            continue
        else:
            ls = [1,4,6,9]
            num = int(film_num[0])
            print(num)
            if num in ls:
                film_list.append(key)
                print(film_list)
                numpy_list.append(data[key])


    for numpy_array in numpy_list:
        # 分割数组
        devide_list = []
        print(numpy_array.shape)
        for m in range(int(list(numpy_array.shape)[1] / 2000)):
            devide_list.append(2000 * (m + 1))
        part = np.hsplit(numpy_array, devide_list)
        #print(part)
        lenth = len(part)
        print(lenth)
        for n in range(lenth):
            # print(part[n].shape)
            # 计算协方差矩阵
            matrix = np.corrcoef(part[n])
            print('第段视频第{}部分'.format( n + 1))
            print(part[n])
            print('邻接矩阵：')
            print(matrix)
            # print(matrix.shape)

            # 筛选邻接矩阵数据
            result = []
            for i in range(62):
                for j in range(62):
                    if matrix[i][j] > 0.7 and i < j:  # 筛选阈值设置为0.1
                        result.append([int(i + 1), int(j + 1), matrix[i][j]])
            # print(result)

            # 写入csv文件
            for name in film_list:
                df = pd.DataFrame(result)
                os.mkdir()
                csv_save_path = './Distance/{}/{}_part{}.csv'.format(file, name, n)
                df.to_csv(csv_save_path, sep=',', index=False, header=False)




path = os.getcwd()
#list all dir
dir_list = os.listdir(path+'/Preprocessed_EEG')



#deal with film [1,4,6,9] and compute the adj
for dir in dir_list:
    adjacent_matrix(dir)
