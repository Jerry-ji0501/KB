import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from GRU import  BGI,Prediction_Linearlayer
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
#from loadData import load_feature_data,load_topo_data
from FeatureDivided import load_feature
from Add_Windows import Add_Windows
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.offline as offline




#from tensorflow.python import debug as tf_debug


def Smooth(train_acc,train_loss,train_loss1,train_loss2):
    train_acc_ave = []
    train_loss_ave = []
    train_loss1_ave = []
    train_loss2_ave = []
    for i in range (0,len(train_acc),10):
        sub_acc = train_acc[i:i+10]
        ave = sum(sub_acc)/len(sub_acc)
        train_acc_ave.append(ave)

    for i in range (0,len(train_loss),10):
        sub_loss = train_loss[i:i+10]
        ave = sum(sub_loss)/len(sub_loss)
        train_loss_ave.append(ave)

    for i in range (0,len(train_loss1),10):
        sub_loss1 = train_loss1[i:i+10]
        ave = sum(sub_loss1)/len(sub_loss1)
        train_loss1_ave.append(ave)

    for i in range (0,len(train_loss2),10):
        sub_loss2 = train_loss2[i:i+10]
        ave = sum(sub_loss2)/len(sub_loss2)
        train_loss2_ave.append(ave)
    return train_acc_ave,train_loss_ave,train_loss1_ave,train_loss2_ave


def train(network,criterion,train_dataloader,device,batch_size,num_epochs,lr,scheduler_type='Cosine'):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)#xavier_normal_

    network.apply(init_xavier)
    print ('training on:',device)
    #network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    logger = SummaryWriter(log_dir='./logs')
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer,T_max=num_epochs)
        print ('using CosineAnnealingLR')
    train_loss = []
    train_loss1 = []
    train_loss2 = []
    train_acces = []
    eval_acces = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print ('epoch:',epoch+1)
        network.train()
        train_acc = 0

        Brain_graph = []


        for batch_idx ,(x,zpi,node_embedding,labels) in enumerate(train_dataloader):#运行不了就删掉tqdm
            #x = x.to(device)batch_size*graph_Num*NodeNum*NodeNum
            #x = x.numpy()

            #print(x.shape)#(7,4,3,62,10)
            #x = Add_Windows(x,window_len=3,stride=3)
            #x = torch.FloatTensor(x)
            #print(x.shape)

            #zpi = zpi.to(device)

            node_embedding = node_embedding[:,:,:]
            #node_embedding = node_embedding.to(device)

            #labels = labels.to(device)
            outputs_pro,outputs_all = network(x,zpi,node_embedding)#batch_size*node_num*node_num
            Linear_layer = Prediction_Linearlayer(62)
            outputs_all = Linear_layer(outputs_all)
            Brain_graph.append(outputs_all)
            print(outputs_all.shape)
            print(labels.shape)

            loss1 = criterion(outputs_all, labels)
            #loss2 = criterion(outputs_pro, labels_pro)
            #logger.add_scalar('loss1',loss1.item(),batch_idx)
            #logger.add_scalar('loss2',loss2.item(),batch_idx)

            optimizer.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),max_norm=2)
            optimizer.step()
            _, pred = outputs_all.max(1)
            #print(pred)
            #print(labels)

            num_correct = (pred == labels).sum().item()

            acc = num_correct / (batch_size)
            train_acc += acc
            #logger.add_scalar('accuracy',acc.item(),batch_idx)


        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, loss1.item(), train_acc / len(train_dataloader)))
        #print(len(train_dataloader))
        writer.add_scalar('loss/loss1',loss1,epoch)
        #writer.add_scalar('loss/loss2',loss2,epoch)
        writer.add_scalar('acc',train_acc / len(train_dataloader),epoch)
        train_acces.append(train_acc / len(train_dataloader))
        #train_loss.append(loss.item())
        train_loss1.append(loss1.item())
        #train_loss2.append(loss2.item())







    return train_acces,train_loss1,Brain_graph



def obj_data_to_mesh3d(odata):
    # odata is the string read from an obj file
    vertices = []
    faces = []
    lines = odata.splitlines()

    for line in lines:
        slist = line.split()
        if slist:
            if slist[0] == 'v':
                vertex = np.array(slist[1:], dtype=float)
                vertices.append(vertex)
            elif slist[0] == 'f':
                face = []
                for k in range(1, len(slist)):
                    face.append([int(s) for s in slist[k].replace('//', '/').split('/')])
                if len(face) > 3:  # triangulate the n-polyonal face, n>3
                    faces.extend(
                        [[face[0][0] - 1, face[k][0] - 1, face[k + 1][0] - 1] for k in range(1, len(face) - 1)])
                else:
                    faces.append([face[j][0] - 1 for j in range(len(face))])
            else:
                pass

    return np.array(vertices), np.array(faces)














if __name__ == '__main__':
    path = os.getcwd()
    node_num = 62
    graph_num = 9
    dim_in = 10
    dim_out = 62
    window_len = 3
    link_len = 2
    emb_dim = 3
    num_layers = 2
    batch_size = 4



    data_all = load_feature(r'data/feature/')
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    x_all = torch.FloatTensor(data_all)#torch.randn(64,graph_num,62,10)
    x_all = x_all.permute((0,3,2,1))
    #x_all = normalize(x_all)
    print(x_all.shape)
    #zigzag_all = np.load('zpi.npy')
    #zigzag_all = zigzag_all.reshape(180,9,100,100)
    #zigzag_all = torch.FloatTensor(zigzag_all)


    #print(zigzag_all.shape)
    zigzag_all = torch.randn(675,9,100,100)
    node_embedding = torch.randn(675,62,3)
    labels = np.array(
        [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,
         2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0, 2, 1, 0, 0, 1, 2, 0,
         1, 2, 2, 1, 0, 1, 2, 0,


         ])  #

    labels = torch.LongTensor(labels)
    labels_part =np.array([[1,0,1,0,1,1,2,2,2],[2,2,0,1,0,2,0,1,0],[1,2,1,1,2,1,2,0,1],[0,0,0,0,0,0,0,1,2]])
    labels_pro = np.tile(labels_part,(45,1))
    labels_pro = torch.LongTensor(labels_pro)

    writer = SummaryWriter('./path/to/log')


    #labels = torch.randint(0,high=3,size=(180,))
    #one_hot_labels = torch.zeros(64,3,dtype=torch.long)
    #one_hot_labels[range(64), labels] = 1


    data = torch.utils.data.TensorDataset(x_all,zigzag_all,node_embedding,labels)
    #train_data, val_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])
    train_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True, drop_last=False)
    #val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = BGI(node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers=num_layers,Window_Num=9)

    #print('# Model parameters:', sum(param.numel() for param in network.parameters()))

    criterion = nn.CrossEntropyLoss()
    train_acc,train_loss1,Brain_graph= train(network,criterion,train_dataloader,device,batch_size=batch_size,num_epochs=20,lr=1e-5)


    #showpic(train_acc,train_loss1,500)
    #smooth(train_acc,train_loss1)
    for name, parms in network.named_parameters():
        print(name)
        print(parms.requires_grad)
        print(parms.grad)

    cmat = Brain_graph[0][0]
    cmat = cmat.detach().numpy()
    nodes = np.loadtxt('./brain_plot/fs_region_centers_68_sort.txt')

    labels = []
    with open("./brain_plot/fs_region_centers_68_sort.txt", "r") as f:
        for line in f:
            labels.append(line.strip('\n'))

    # %%

    # ! 这个程序必定报错，但是可以手动继续进行下面的工作
    [source, target] = np.nonzero(np.triu(cmat) > 0.01)

    nodes_x = nodes[:, 0]
    nodes_y = nodes[:, 1]
    nodes_z = nodes[:, 2]

    edge_x = []
    edge_y = []
    edge_z = []
    for s, t in zip(source, target):
        if s < len(nodes) and t < len(nodes):  # 检查索引是否有效
            edge_x += [nodes_x[s], nodes_x[t]]
            edge_y += [nodes_y[s], nodes_y[t]]
            edge_z += [nodes_z[s], nodes_z[t]]

    # %%

    with open(r"./brain_plot/lh.pial.obj", "r") as f:
        obj_data = f.read()
    [vertices, faces] = obj_data_to_mesh3d(obj_data)

    vert_x, vert_y, vert_z = vertices[:, :3].T
    face_i, face_j, face_k = faces.T

    # Download and prepare dataset from BrainNet repo
    coords = np.loadtxt(np.DataSource().open('./brain_graph/BrainMesh_Ch2_smoothed.nv'), skiprows=1, max_rows=53469)
    x, y, z = coords.T

    triangles = np.loadtxt(np.DataSource().open('./brain_graph/BrainMesh_Ch2_smoothed.nv'), skiprows=53471, dtype=int)
    triangles_zero_offset = triangles - 1
    i, j, k = triangles_zero_offset.T

    fig = go.Figure()

    fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                                    i=i, j=j, k=k,
                                    color='lightpink', opacity=0.5, name='', showscale=False, hoverinfo='none')])

    fig.add_trace(go.Scatter3d(x=nodes_x, y=nodes_y, z=nodes_z, text=labels,
                               mode='markers', hoverinfo='text', name='Nodes', ))
    fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                               mode='lines', hoverinfo='none', name='Edges'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
        ),
        width=1200, height=800
    )
    offline.plot(fig, filename='./brain_plot/model.html', auto_open=False)

    fig.show()







