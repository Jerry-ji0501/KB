import os
import numpy as np

#zpi load distancematrix->zpi
#gcn load 构建好的特征数据
paths = os.getcwd()

def load_DistanceMatrix(paths,GraphNum = 27,SampNum = 180):
    #DistanceMatrix is used in ZPI compution   SampleNum*GraphNum*NVertices*NVertices
    BrainGraphs = []
    edgeAucGraphs = []
    dir_list = os.listdir(paths + '/distancematrix')
    for i in range (SampNum):
        for j in range (GraphNum):
            path = os.path.join(paths+'/distancematrix'+dir_list[(i-1)*27+j])
            edgelist = np.loadtxt(path,delimiter=',')
            edgeAucGraphs.append(edgelist)
            edgeAuc = np.stack(edgeAucGraphs,axis=0)
        BrainGraphs.append(edgeAucGraphs)
        BrainTopoData = np.stack(BrainGraphs,axis=0)
    print(BrainTopoData.shape)
    return BrainTopoData #SampNum*GraphNum*NVertices*NVertices


def load_feature():
    #load feature of Vertices feature ->GCN SampleNum*GraphNum*NVertices*Feature




def load_zpi():
    #用于加载计算保存的zpi特征











