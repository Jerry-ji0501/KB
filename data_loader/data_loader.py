import os
import numpy as np

#zpi load distancematrix->zpi
#gcn load 构建好的特征数据
path = os.getcwd()

def load_DistanceMatrix(paths):
    #DistanceMatrix is used in ZPI compution   SampleNum*GraphNum*NVertices*NVertices
    BrainGraphs = []
    dir_list = os.listdir(paths + '/distancematrix')
    for i in range (len(dir_list)):
        path = os.path.join(paths+'/distancematrix'+dir_list[i])
        edgelist = np.loadtxt(path,delimiter=',')
        BrainGraphs.append(edgelist)
        if (i%27) == 0:




    for cur_file in dir_list:
        path = os.path.join(paths + '/distancematrix', cur_file)

        edgelist = np.loadtxt(path, delimiter=',')
        BrainGraphs.append(edgelist)
    return BrainGraphs


def load_feature():
    #load feature of Vertices feature ->GCN SampleNum*GraphNum*NVertices*Feature
















