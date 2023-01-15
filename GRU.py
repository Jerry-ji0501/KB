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
    def __init__(self, node_num,graph_num,dim_in,dim_out,link_len,emb_dim,num_layers):
        super(BGI, self).__init__()
        self.node_num = node_num
        self.window_len = graph_num
        self.link_len = link_len
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=node_num,hidden_size=self.output_dim,num_layers=num_layers,batch_first = True)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.linear1 = nn.Linear(in_features=node_num,out_features=256)
        self.linear2 = nn.Linear(in_features=256,out_features=3)

    def forward(self,x,zigzag_PI,node_embeddings):#x is Batch_size*Graph_num*NVertices*NVertices
       h = []
       graph_num = 9
       emb_dim = node_embeddings[1]
       for t in range(1,graph_num):
           #print(t)
           #print(x[:,t,:,:].shape)
           ZGCN = TLSGCN(dim_in=10, dim_out=62, link_len=2, emb_dim=3, window_len=int(t))
           x_conv = ZGCN(x[:,t,:,:],x,node_embeddings,zigzag_PI)

           h.append(x_conv)
       x = torch.stack(h,dim=1)#batch_size*Seq_length*NVertices*NVertices

       x =  torch.where(torch.isnan(x) | torch.isinf(x),torch.tensor([0.0]),x)


       #print(x[:,:,61,:].shape)
       output = []
       for i in range(self.node_num):
           hn = self.gru(x[:,:,int(i),:])


           output.append(hn[0][:,-1,:])
       output = torch.stack(output,dim=1)
       output = torch.sum(output,dim=2)

       output = F.relu(self.linear1(output))#batch_size*3
       #output = output.permute(0,2,1)
       output = F.relu(self.linear2(output))

       return output







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













