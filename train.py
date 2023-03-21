import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from GRU import  BGI
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
#from loadData import load_feature_data,load_topo_data
from FeatureDivided import load_feature
from Add_Windows import Add_Windows
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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


def train(network,criterion,train_dataloader,val_dataloader,device,batch_size,num_epochs,lr,scheduler_type='Cosine'):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)#xavier_normal_

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


        for batch_idx ,(x,zpi,node_embedding,labels,labels_pro) in enumerate(train_dataloader):#运行不了就删掉tqdm
            #x = x.to(device)batch_size*graph_Num*NodeNum*NodeNum
            x = x.numpy()

            #print(x.shape)#(7,4,3,62,10)
            x = Add_Windows(x,window_len=3,stride=1)
            x = torch.FloatTensor(x)

            #zpi = zpi.to(device)

            node_embedding = node_embedding[-1,:,:]
            #node_embedding = node_embedding.to(device)

            #labels = labels.to(device)
            outputs_all,outputs_pro = network(x,zpi,node_embedding)#batch_size*node_num*node_num
            print(outputs_pro.shape)

            loss1 = criterion(outputs_all, labels)
            loss2 = criterion(outputs_pro, labels_pro)
            logger.add_scalar('loss1',loss1.item(),batch_idx)
            logger.add_scalar('loss2',loss2.item(),batch_idx)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),max_norm=2)
            optimizer.step()
            _, pred = outputs_all.max(1)

            num_correct = (pred == labels).sum().item()

            acc = num_correct / (batch_size)
            train_acc += acc
            #logger.add_scalar('accuracy',acc.item(),batch_idx)


        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, loss.item(), train_acc / len(train_dataloader)))
        print(len(train_dataloader))
        writer.add_scalar('loss/loss1',loss1,epoch)
        writer.add_scalar('loss/loss2',loss2,epoch)
        writer.add_scalar('acc',train_acc / len(train_dataloader),epoch)
        train_acces.append(train_acc / len(train_dataloader))
        train_loss.append(loss.item())
        train_loss1.append(loss1.item())
        train_loss2.append(loss2.item())


        network.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for  batch_idx ,(x,zpi,node_embedding,labels,labels_pro) in enumerate(val_dataloader):
                x = x.numpy()
                x = Add_Windows(x,window_len=3,stride=1)
                x = torch.FloatTensor(x)
                node_embedding = node_embedding[-1, :, :]
                outputs_all, outputs_pro = network(x, zpi, node_embedding)
                loss = criterion(outputs_all, labels)
                #loss2 = criterion(outputs_pro, labels_pro)
                #loss = loss1 + loss2
                _, pred = outputs_all.max(1)

                num_correct = (pred == labels).sum().item()
                eval_loss+=loss

                acc = num_correct / (batch_size)
                eval_acc += acc

            eval_losses = eval_loss / (len(val_dataloader))
            eval_acc = eval_acc / (len(val_dataloader))
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(network.state_dict(), 'best_acc.pth')
            eval_acces.append(eval_acc)
            print("整体验证集上的Loss: {}".format(eval_losses))
            print("整体验证集上的正确率: {}".format(eval_acc))




    return train_acces, train_loss,train_loss1,train_loss2,eval_acces

def showpic(train_acc, train_loss,train_loss1,train_loss2,num_epoch):
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(1 + np.arange(len(train_loss)), train_loss, linewidth=1.5, linestyle='dashed', label='train_loss')
    plt.subplot(2,2,2)
    plt.plot(1 + np.arange(len(train_loss1)), train_loss1, linewidth=1.5, linestyle='dashed', label='train_loss1')
    plt.subplot(2,2,3)
    plt.plot(1 + np.arange(len(train_loss2)), train_loss2, linewidth=1.5, linestyle='dashed', label='train_loss2')
    plt.subplot(2,2,4)
    plt.plot(1 + np.arange(len(train_acc)), train_acc, linewidth=1.5, linestyle='dashed', label='train_acc')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch,50))
    plt.legend()
    plt.show()
    plt.plot(1+np.arange(len(eval_acces)),eval_acces,linewidth=1.5, linestyle='dashed', label='eval_acces')
    plt.show()

def smooth(train_acc, train_loss,train_loss1,train_loss2):
    train_acc_ave,train_loss_ave,train_loss1_ave,train_loss2_ave =Smooth(train_acc, train_loss, train_loss1, train_loss2)
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(1 + np.arange(len(train_loss_ave)), train_loss_ave, linewidth=1.5, linestyle='dashed', label='train_loss_smooth')
    plt.subplot(2, 2, 2)
    plt.plot(1 + np.arange(len(train_loss1_ave)), train_loss1_ave, linewidth=1.5, linestyle='dashed', label='train_loss1_smooth')
    plt.subplot(2, 2, 3)
    plt.plot(1 + np.arange(len(train_loss2_ave)), train_loss2_ave, linewidth=1.5, linestyle='dashed', label='train_loss2_smooth')
    plt.subplot(2, 2, 4)
    plt.plot(1 + np.arange(len(train_acc_ave)), train_acc_ave, linewidth=1.5, linestyle='dashed', label='train_acc_smooth')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 50, 5))
    plt.legend()
    plt.show()

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

    data_all = load_feature('E:/JWX/Code/KB-main/data/feature/10s')
    print(data_all.shape)
    x_all = torch.FloatTensor(data_all)#torch.randn(64,graph_num,62,10)
    x_all = x_all.permute((0,3,2,1))
    zigzag_all = torch.randn(180,1,100,100)
    node_embedding = torch.randn(180,62,3)
    label_part = np.array([2,2,2,2])
    labels = np.tile(label_part,45)
    labels = torch.LongTensor(labels)
    labels_part =np.array([[1,0,1,0,1,1,2],[2,2,0,1,0,2,0],[1,2,1,1,2,1,2],[0,0,0,0,0,0,0]])
    labels_pro = np.tile(labels_part,(45,1))
    labels_pro = torch.LongTensor(labels_pro)

    writer = SummaryWriter('./path/to/log')


    #labels = torch.randint(0,high=3,size=(180,))
    #one_hot_labels = torch.zeros(64,3,dtype=torch.long)
    #one_hot_labels[range(64), labels] = 1


    data = torch.utils.data.TensorDataset(x_all,zigzag_all,node_embedding,labels,labels_pro)
    train_data, val_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)


    device = torch.device("cuda:0"  "cpu")
    network = BGI(node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers=num_layers)

    #print('# Model parameters:', sum(param.numel() for param in network.parameters()))

    criterion = nn.CrossEntropyLoss()
    train_acc,train_loss,train_loss1,train_loss2 ,eval_acces= train(network,criterion,train_dataloader,val_dataloader,device,batch_size=4,num_epochs=500,lr=1e-5)

    showpic(train_acc,train_loss,train_loss1,train_loss2,500)
    smooth(train_acc,train_loss,train_loss1,train_loss2)
    for name, parms in network.named_parameters():
        print(name)
        print(parms.requires_grad)
        print(parms.grad)






