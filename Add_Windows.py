import numpy as np
import torch
#对于原始的data数据(SampleNums,GraphNums,NodeNums,NodeNums)对第二维的GraphNums进行加窗
#Window_len = 3 and Stride = 1
#对于长度不足窗口数的解决办法：(1)循环队列 从图集成的任务上来看的话不太符合(2)对不足的窗口补0 算作下策(3)不然我能的预测的长度就定义为窗口数 = GraphNums-Windows_len+1
#个人来看更倾向于(3)
#para:inputdata:batch_size*graph_num*Node_Num*Node_Num
#output:batch_size*window_Num*graphNums*Node_Num*Node_Num

def Add_Windows(data,window_len=3,stride=1):
    length = data.shape[1]
    end_index = length-window_len+stride
    X = []
    index = 0
    while index <end_index:
        X.append(data[:,index:index+window_len,:])
        index += stride
    X = np.array(X)
    X = torch.FloatTensor(X)
    print(X.shape)
    return X





