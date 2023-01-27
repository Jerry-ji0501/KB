import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ZGCN import TLSGCN

class MLP(nn.Module):
    def __int__(self,input_size,hidden_size,output_size):
        super(MLP,self).__int__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x








class BGI(nn.Module):
    def __init__(self, node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers):
        super(BGI, self).__init__()
        self.node_num = node_num
        self.graph_num = graph_num
        self.window_len = window_len
        self.link_len = link_len
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=node_num,hidden_size=self.output_dim,num_layers=num_layers,batch_first = True)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.linear1 = nn.Linear(in_features=node_num,out_features=256)
        self.linear2 = nn.Linear(in_features=256,out_features=3)

    def forward(self,x_all,zigzag_PI,node_embeddings):#x is Batch_size*Graph_num*NVertices*NVertices
       #x_all:windowNum*batchSize*window_len*NodeNum*F
       h = []
       window_num = x_all.shape[0]
       emb_dim = node_embeddings[1]
       #ZGCN = TLSGCN(dim_in=10, dim_out=62, link_len=2, emb_dim=3, window_len=3)
       for t in range(0,window_num):
           #print(t)
           #print(x[:,t,:,:].shape)
           x_window = x_all[t,:,:,:,:]#Batchsi、ze*Window_len*NodeNum*F
           x_window = x_window.reshape(4,3,62,10)
           x = x_window[:,-1,:,:]#Batchsize*NodeNum*F
           x = x.reshape(4,62,10)
           ZGCN = TLSGCN(dim_in=10, dim_out=62, link_len=2, emb_dim=3, window_len=3)
           x_conv = ZGCN(x,x_window,node_embeddings,zigzag_PI)

           h.append(x_conv)
       x = torch.stack(h,dim=1)#batch_size*Window_Num*NVertices*NVertices

       x =  torch.where(torch.isnan(x) | torch.isinf(x),torch.tensor([1e-5]),x)


       #print(x[:,:,61,:].shape)
       print(x.shape)
       output_part = []
       output_all = []
       for i in range(self.node_num):
           hn = self.gru(x[:,:,int(i),:])
           output_part.append(hn[0])
           output_all.append(hn[0][:,-1,:])
       output_part = torch.stack(output_part,dim=2)
       output_part = torch.sum(output_part,dim=-1)
       output_all = torch.stack(output_all,dim=1)#
       output_all = torch.sum(output_all,dim=2)#
       print(output_part.shape)
       output_pro = []
       for i in range(7):
           auc = F.relu(self.linear1(output_part[:,i,:]))
           auc = F.relu(self.linear2(auc))
           output_pro.append(auc)
       output_pro = torch.stack(output_pro,dim=1)
       output_pro = output_pro.permute(0,2,1)
       output_all = F.relu(self.linear1(output_all))#batch_size*3
       #output = output.permute(0,2,1)
       output_all = F.relu(self.linear2(output_all))

       return output_all,output_pro







#if __name__ == '__main__':
#    node_num = 62
#    graph_num = 27
#    dim_in = 10
#    dim_out = 62
#    link_len = 2
#    emb_dim = 3
#    num_layers = 2
#    zigzag_PI=torch.randn(64,1,100,100)
#    x = torch.randn(64,27,62,10)
#    node_embedding = torch.randn(node_num,emb_dim)




#    model = BGI(node_num,graph_num,dim_in,dim_out,link_len,emb_dim,num_layers=num_layers)
#    output = model(x,zigzag_PI,node_embedding)
#    print(output)













