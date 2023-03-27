import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import pandas as pd
import numpy as np
import torch.nn.functional as F
from Sychronization import *
from GRU import *
import torch.nn.functional as F
from FeatureDivided import load_feature
from torch.utils.tensorboard import SummaryWriter






'''
   This part of  the py is aim to achieve of the classification based on high level of feature
   KG_all ,BG_all and A_L_all
   KG_all and BG_all are Graph Structures should be passed into GCN to fuse the neighbors' node information
   input :
     KG_all:(batch_size,KG_num,embed_dim)
     
     BG_all:(batch_size,BG_num,embed_dim)
     A_L_all:(batch_size,max_KG_num,BG_num)
     
    output:
    labels:
'''


class GCNLayer(nn.Module):
    def __init__(self,input_dim,output_dim,bias = False):
        super(GCNLayer,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()


    def reset_parameters(self):
        ''' initialize parameters'''
        std = math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)


    def forward(self,adj,x):
        '''

        :param adj:
        :param x:
        '''
        print(x.shape)#(4,128,128)
        #weight(128,128)
        support = torch.matmul(x,self.weights)
        print(support.shape)


        output = torch.matmul(adj,support)
        if self.bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__+'('+ str(self.input_dim) + ')' + '->' + str(self.output_dim) + ')'



class GCN(nn.Module):
    '''
        Build  a two-layer GCN network
    '''
    def __init__(self,input_dim,hidden_dim,output_dim,dropout_rate,bias = False):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gcn1 = GCNLayer(input_dim,hidden_dim,bias=bias)
        self.gcn2 = GCNLayer(hidden_dim,output_dim,bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,adj,x):
        x = F.relu(self.gcn1(adj,x))
        x= self.dropout(x)
        x = self.gcn2(adj,x)
        return F.log_softmax(x,dim=1)


class DenseLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        output = self.fc(x)
        return output

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self,x):
        return x.view(x.shape[0],-1)




class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 181 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
















class FusionLayer(nn.Module):
    def __init__(self,node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers,Window_Num,
                 KG_input_dim, output_dim, dropout_rate,
                 KG_num,
                 KG_embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 embed_dim,
                 ffn_size,
                 num_layer
                 , num_decoder_layers,
                 attention_prob_dropout_prob,
                 ):
        super(FusionLayer,self).__init__()
        self.BrainGraphIntergration = BGI(node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers,Window_Num)
        self.Sychronization = SYN(KG_input_dim,output_dim,dropout_rate,
                 KG_num,
                 KG_embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 embed_dim,
                 ffn_size,
                 num_layer
                 ,num_decoder_layers,
                 attention_prob_dropout_prob,num_layers)
        #self.KG_GCN = GCN()
        #self.Dense = nn.Linear()
        self.conv = ConvModule()


    def forward(self,KG_embeddings,in_degree,out_degree,x_all,zigzag_PI,node_embeddings):
        output_pro, output_all = self.BrainGraphIntergration(x_all,zigzag_PI,node_embeddings)
        A_L_List =  []
        BG_Graph_Construct_list = []
        p_list = []
        q_list = []

        for i in range(9):
            BG_Graph_Construct, p, q, A_L = self.Sychronization(KG_embeddings[:,i,:,:],output_pro[:,i,:,:],in_degree[:,i,:],out_degree[:,i,:])
            #print(A_L.shape)
            BG_Graph_Construct_list.append(BG_Graph_Construct)
            p_list.append(p)
            q_list.append(q)
            A_L_List.append(A_L)
        A_L_all = torch.zeros(A_L.shape)
        for tensor in A_L_List:
            A_L_all += tensor

        # = torch.sum(A_L_List)
        KG_all = torch.sum(KG_embeddings,dim=1)
        max_row = max(A_L_all.shape[1],KG_all.shape[1],output_all.shape[1])
        max_line = max(A_L_all.shape[2],KG_all.shape[2],output_all.shape[2])


        #Padding first dimension
        A_L_all = F.pad(A_L_all,(0,0,0,max_row-A_L_all.shape[1],0,0),"constant",0)
        KG_all = F.pad(KG_all,(0,0,0,max_row-KG_all.shape[1],0,0),"constant",0)
        output_all = F.pad(output_all,(0,0,0,max_row-output_all.shape[1],0,0),"constant",0)

        A_L_all = F.pad(A_L_all,(0,max_line-A_L_all.shape[2],0,0,0,0),"constant",0)
        KG_all = F.pad(KG_all,(0,max_line-KG_all.shape[2],0,0,0,0),"constant",0)
        output_all = F.pad(output_all,(0,max_line-output_all.shape[2],0,0,0,0),"constant",0)

        #print(A_L_all.shape)
        #print(KG_all.shape)
        #print(output_all.shape)  #

        feature_all = torch.stack([A_L_all,KG_all,output_all],dim=1)
        #print(feature_all.shape)
        output  = self.conv(feature_all)
        return output,output_pro,BG_Graph_Construct_list,p_list,q_list,A_L_all



















def train(model,train_dataloader,mae,criterion,device,train_epoch,batch_size,lr,scheduler_type='Cosine'):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)#xavier_normal_

    model.apply(init_xavier)
    print(device)
    device = torch.device("cpu")
    optimizer = optim.Adam(model.parameters(),lr)
    #mae = loss1()
    #critirion = loss2()
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer,T_max=train_epoch)
        print ('using CosineAnnealingLR')
    train_loss = []
    train_loss1 = []
    train_loss2 = []
    train_loss_kl = []
    train_acces = []
    eval_acces = []
    best_acc = 0.0


    for epoch in range (train_epoch):
        model.train()
        train_acc = 0




        for batch_idx ,(x_all, zigzag_all, node_embeddings,kg_embeddings,in_degree,out_degree,labels) in enumerate(train_dataloader):
            output,output_pro, BG_Graph_Construct_list, p_list, q_list,A_L_all = model(kg_embeddings, in_degree, out_degree, x_all,
                                                                    zigzag_all, node_embeddings)


            BG_Graph_Construct = torch.stack(BG_Graph_Construct_list,dim=1)

            loss1 = 0.0001*mae(output_pro,BG_Graph_Construct)
            loss2 = criterion(output,labels)
            loss_kl = 0
            for p,q in zip(p_list,q_list):
                loss_kl += torch.distributions.kl.kl_divergence(p,q).sum()

            print("loss1",loss1)
            print("loss2",loss2)
            print("loss_kl",loss_kl)
            kl_lambda = 0.000001
            loss = loss1+loss2+kl_lambda*loss_kl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
            _, pred = output.max(1)

            num_correct = (pred == labels).sum().item()
            # print(num_correct)

            acc = num_correct / (batch_size)
            train_acc += acc



        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, loss.item(), train_acc / len(train_dataloader)))
        #print(len(train_dataloader))
        writer.add_scalar('loss',loss,epoch)
        writer.add_scalar('loss/loss1', loss1, epoch)
        writer.add_scalar('loss/loss2', loss2, epoch)
        writer.add_scalar('loss/loss_kl', loss_kl, epoch)
        writer.add_scalar('acc', train_acc / len(train_dataloader), epoch)
        train_acces.append(train_acc / len(train_dataloader))
        train_loss.append(loss.item())
        train_loss1.append(loss1.item())
        train_loss2.append(loss2.item())
        train_loss_kl.append(loss_kl.item())

        model.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for batch_idx, (
            x_all, zigzag_all, node_embeddings, kg_embeddings, in_degree, out_degree, labels) in enumerate(
                    val_dataloader):
                output, output_pro, BG_Graph_Construct_list, p_list, q_list,A_L_all = model(kg_embeddings, in_degree,
                                                                                    out_degree, x_all,
                                                                                    zigzag_all, node_embeddings)
                loss = criterion(output, labels)

                _, pred = output.max(1)

                num_correct = (pred == labels).sum().item()
                eval_loss += loss

                acc = num_correct / (batch_size)
                eval_acc += acc

            eval_losses = eval_loss / (len(val_dataloader))
            eval_acc = eval_acc / (len(val_dataloader))
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(model.state_dict(), 'best_acc.pth')
            eval_acces.append(eval_acces)
            print("整体验证集上的Loss: {}".format(eval_losses))
            print("整体验证集上的正确率: {}".format(eval_acc))
            writer.add_scalar('eval_loss', eval_losses,epoch)
            writer.add_scalar('eval_acces',eval_acc,epoch)

    return A_L_all












if __name__ == '__main__':
    node_num = 62#  This is the number of nodes in the Brain Graph
    graph_num = 9 # This is the number of
    dim_in = 10
    dim_out = 62
    window_len = 3
    link_len = 2
    emb_dim = 3
    num_layers = 2
    Window_Num = 9
    batch_size = 4
    KG_input_dim = 64
    KG_embed_dim = 64
    input_dim = 64
    output_dim = 64
    dropout_rate = 0.1
    KG_num = 1454
    embed_dim = 64
    num_in_degree = 1454
    num_out_degree = 1454
    num_heads = 8
    hidden_size = 64
    ffn_size = 64
    num_layer = 8
    attention_prob_dropout_prob = 0.1
    num_decoder_layers = 3
    num_layers = 3

    writer = SummaryWriter('./path/to/log')

    adjacent_matrix_list = []
    kg_embed_list = []

    path = r'./data/tripples_embedding/tripples_embedding/'
    dir_list = os.listdir(path)
    # print(dir_list)
    for dir in dir_list:
        file_list = os.listdir(path + dir)
        # print(file_list)

        adj_matrix = np.load(path + dir + '/' + file_list[0])
        kg_embedding = np.load(path + dir + '/' + file_list[1])
        adjacent_matrix_list.append(adj_matrix)
        kg_embed_list.append(kg_embedding)
    kg_embeddings = np.stack(kg_embed_list, axis=0)
    adj_matrixs = np.stack(adjacent_matrix_list, axis=0)
    kg_embeddings = np.transpose(kg_embeddings.reshape((15, 9, 1454, 64)), (0, 1, 2, 3))
    adj_matrixs = np.transpose(adj_matrixs.reshape((15, 9, 1454, 1454)), (0, 1, 2, 3))
    in_degree = np.sum(adj_matrixs, axis=2)
    out_degree = np.sum(adj_matrixs, axis=3)
    print("================================")
    kg_embeddings = np.tile(kg_embeddings,(12,1,1,1))
    in_degree = np.tile(in_degree,(12,1,1))
    out_degree = np.tile(out_degree,(12,1,1))
    kg_embeddings = torch.tensor(kg_embeddings)
    adj_matrixs = torch.tensor(adj_matrixs)
    in_degree = torch.LongTensor(in_degree)
    out_degree = torch.LongTensor(out_degree)
    print(kg_embeddings.shape)
    print(adj_matrixs.shape)
    print(in_degree.shape)
    print(out_degree.shape)




    #KG_embeddings = torch.randn(15, 9, 1454, 64)
    #in_degree = torch.randint(0, 5, (15, 9, 1454))
    #out_degree = torch.randint(0, 5, (15, 9, 1454))
    #x_all = torch.randn(15, 27, 62, 10)
    data_all = load_feature(r'data/feature/')
    x_all = torch.FloatTensor(data_all)  # torch.randn(64,graph_num,62,10)
    x_all = x_all.permute((0, 3, 2, 1))
    x_all = x_all[0:180]
    zigzag_all = np.load(r'zpi.npy')
    zigzag_all = torch.Tensor(zigzag_all)
    zigzag_all = zigzag_all.view(180,9,100,100)
    zigzag_all = zigzag_all[0:180]
    #zigzag_all = torch.randn(15, 9, 100, 100)
    #node_embeddings = torch.randn(15, 62, 3)
    node_embedding = pd.read_excel(r'./data/nodeEmbedding.xlsx', header=None)

    node_embedding = np.array(node_embedding)
    node_embedding = torch.Tensor(node_embedding)
    node_embedding = torch.nn.functional.normalize(node_embedding)  # normal
    node_embeddings = node_embedding.repeat(180,1,1)

    labels = np.array([2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,
                       2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,
                       2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,
                       2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,2,1,0,0,1,2,0,1,2,2,1,0,1,2,0,

                       ])  #
    #labels = labels.repeat(12)
    labels = torch.LongTensor(labels)
    device = torch.device('cpu')

    data = torch.utils.data.TensorDataset(x_all, zigzag_all, node_embeddings,kg_embeddings,in_degree,out_degree,labels)
    train_data, val_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False)

    mae = nn.L1Loss()
    critrion = nn.CrossEntropyLoss()

    model = FusionLayer(node_num, graph_num, dim_in, dim_out, window_len, link_len, emb_dim, num_layers, Window_Num,
                        KG_input_dim, output_dim, dropout_rate,
                        KG_num,
                        KG_embed_dim,
                        num_in_degree,
                        num_out_degree,
                        num_heads,
                        hidden_size,
                        embed_dim,
                        ffn_size,
                        num_layer
                        , num_decoder_layers,
                        attention_prob_dropout_prob, )

    A_L_all = train(model,train_dataloader,mae,critrion,device,batch_size=batch_size,train_epoch=100,lr=1e-5)
    plt.figure(figsize=(180,60))
    #aspect_rate = A_L_all.shape[0]/ A_L_all.shape[1]
    plt.imshow(A_L_all[0],cmap='jet',aspect=1)
    plt.show()









        









