import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ZGCN import TLSGCN



class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]




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
        #self.gru = nn.GRU(input_size=node_num,hidden_size=self.output_dim,num_layers=num_layers,batch_first = True)
        self.ZGCN = TLSGCN(dim_in=10, dim_out=62, link_len=2, emb_dim=3, window_len=3)
        self.flatten = nn.Flatten(start_dim=1,end_dim=-1)
        self.linear1 = nn.Linear(in_features=node_num,out_features=256)
        self.linear2 = nn.Linear(in_features=256,out_features=3)

        GRUs = [nn.GRU(input_size=node_num,hidden_size=self.output_dim,num_layers=num_layers,batch_first = True)  for _ in range(Window_Num)]
        self.GRU_Layer = nn.ModuleList(GRUs)



    def forward(self,brain_graph,zpi,node_embedding):
        '''

        :param brain_graph: (Sessions,Graph_Num,Node_Num,Node_Num)
        :param zpi: (Sessions,window_Num,100,100)
        :param node_embedding:(Sessions,Node_Num,3)
        :return:

        '''
        brain_conv = []

        Window_Num = zpi.shape[1]
        batch_size = zpi.shape[0]
        index = 0
        for i in range (Window_Num):
            brain_graph_window = brain_graph[:,index:index+self.window_len,:,:]
            brain_graph_window_final = brain_graph_window[:,-1,:,:]
            zpi_window = zpi[:,index:index+1,:,:]
            index+=1
            x_conv = self.ZGCN(brain_graph_window_final,brain_graph_window,node_embedding,zpi_window)
            brain_conv.append(x_conv)
        x = torch.stack(brain_conv, dim=1)  # batch_size*Window_Num*NVertices*NVertices
        x = torch.where(torch.isnan(x) | torch.isinf(x), torch.tensor([1e-5]), x)
        _,h = self.GRU_Layer[0](x[:,0,:,:])
        for i in range(1,Window_Num):
            output ,h = self.GRU_Layer[i](x[:,i,:],h)

        return x,output










if __name__ =='__main__':
    sessions  = 180
    Graph_Num = 24
    Node_Num = 62
    Window_Num = 8


    brain_graph = torch.randn(sessions,Graph_Num,Node_Num,10)
    zpi = torch.randn(sessions,Window_Num,100,100)
    node_embedding = torch.randn(sessions,Node_Num,3)
    model = BGI(node_num=Node_Num,graph_num=Graph_Num,dim_in=10,dim_out=62,window_len=3,link_len=2,emb_dim=3,num_layers=3)

    x,output = model(brain_graph,zpi,node_embedding)
    print(x.shape)
    print(output.shape)
















