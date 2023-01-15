import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GRU import  BGI
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from loadData import load_feature_data,load_topo_data
import os




def train(network,criterion,dataloader,device,batch_size,num_epochs,lr,scheduler_type='Cosine'):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)#xavier_normal_

    network.apply(init_xavier)
    print ('training on:',device)
    #network.to(device)
    optimizer = optim.Adam(network.parameters(), lr=lr)
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        print ('using CosineAnnealingLR')
    train_loss = []
    train_acces = []

    for epoch in range(num_epochs):
        print ('epoch:',epoch)
        network.train()
        train_acc = 0

        for batch_idx ,(x,zpi,node_embedding,labels) in enumerate(dataloader):
            #x = x.to(device)

            #zpi = zpi.to(device)
            node_embedding = node_embedding[-1,:,:]
            #node_embedding = node_embedding.to(device)

            #labels = labels.to(device)
            outputs = network(x,zpi,node_embedding)#batch_size*node_num*node_num

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = outputs.max(1)
            num_correct = (pred == labels).sum().item()

            acc = num_correct / (batch_size)
            train_acc += acc


        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, loss.item(), train_acc / len(dataloader)))
        train_acces.append(train_acc / len(dataloader))
        train_loss.append(loss.item())
    return train_acces, train_loss

def showpic(train_acc, train_loss,num_epoch):
    plt.plot(1 + np.arange(len(train_loss)), train_loss, linewidth=1.5, linestyle='dashed', label='train_loss')
    plt.plot(1 + np.arange(len(train_acc)), train_acc, linewidth=1.5, linestyle='dashed', label='train_acc')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch,20))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    path = os.getcwd()
    node_num = 62
    graph_num = 9
    dim_in = 10
    dim_out = 62
    link_len = 2
    emb_dim = 3
    num_layers = 2
    batch_size = 4

    data_all = load_feature_data(path)
    x_all = torch.FloatTensor(data_all)#torch.randn(64,graph_num,62,10)
    x_all = x_all.permute((0,3,2,1))
    print(type(x_all))
    zigzag_all = torch.randn(180,1,100,100)
    node_embedding = torch.randn(180,62,3)
    label_part = np.array([2,0,2,2])
    labels = np.tile(label_part,45)
    labels = torch.LongTensor(labels)
    print(labels)
    print(labels.shape)


    #labels = torch.randint(0,high=3,size=(180,))
    #one_hot_labels = torch.zeros(64,3,dtype=torch.long)
    #one_hot_labels[range(64), labels] = 1

    data = torch.utils.data.TensorDataset(x_all,zigzag_all,node_embedding,labels)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,shuffle=True, drop_last=False)

    device = torch.device("cuda:0"  "cpu")
    network = BGI(node_num,graph_num,dim_in,dim_out,link_len,emb_dim,num_layers=num_layers)

    #print('# Model parameters:', sum(param.numel() for param in network.parameters()))

    criterion = nn.CrossEntropyLoss()
    train_loss,train_acc = train(network,criterion,dataloader,device,batch_size=4,num_epochs=200,lr=1e-5)

    showpic(train_acc,train_loss,200)
    for name, parms in network.named_parameters():
        print(name)
        print(parms.requires_grad)
        print(parms.grad)




