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




class Prediction_Linearlayer(nn.Module):
    def __init__(self, node_num):
        super(Prediction_Linearlayer, self).__init__()
        self.Flatten_layer = nn.Flatten(start_dim=1,end_dim=-1)
        self.linear1 = nn.Linear(in_features=node_num*node_num,out_features=node_num*node_num)
        self.linear2 = nn.Linear(in_features=node_num*node_num,out_features=3)

    def forward(self,x):
        x = self.Flatten_layer(x)
        print(x.shape)
        x = self.linear1(x)
        output = self.linear2(x)

        return output



class ConvModule(nn.Module):
    def __init__(self):
        super(ConvModule, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x






class BGI(nn.Module):
    def __init__(self, node_num, graph_num, dim_in, dim_out, window_len, link_len, emb_dim, num_layers, Window_Num):
        super(BGI, self).__init__()
        self.node_num = node_num
        self.graph_num = graph_num
        self.window_len = window_len
        self.link_len = link_len
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.num_layers = num_layers
        self.Window_Num = Window_Num
        # self.gru = nn.GRU(input_size=node_num,hidden_size=self.output_dim,num_layers=num_layers,batch_first = True)
        self.ZGCN = TLSGCN(dim_in=dim_in, dim_out=dim_out, link_len=link_len, emb_dim=emb_dim, window_len=window_len)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear1 = nn.Linear(in_features=node_num, out_features=256)
        self.linear2 = nn.Linear(in_features=256, out_features=3)
        self.predict = Prediction_Linearlayer(62)

        GRUs = [nn.GRU(input_size=node_num, hidden_size=self.output_dim, num_layers=num_layers, batch_first=True) for _
                in range(Window_Num)]
        self.gcn = GCN(input_dim=62, hidden_dim=62, output_dim=62, dropout_rate=0.1)
        self.GRU_Layer = nn.ModuleList(GRUs)

    def forward(self, brain_graph, zpi, node_embedding,adj_window):
        '''

        :param brain_graph: (batch,Graph_Num,Node_Num,Node_Num)
        :param zpi: (batch,window_Num,100,100)
        :param node_embedding:(batch,Node_Num,3)
        :return:

        '''
        brain_conv = []

        batch_size = zpi.shape[0]
        index = 0
        for i in range(self.Window_Num):
            brain_graph_window = brain_graph[:, index:index + self.window_len, :, :]
            # print(brain_graph_window.shape)
            brain_graph_window_final = brain_graph_window[:, -1, :, :]
            # print(brain_graph_window_final.shape)
            zpi_window = zpi[:, i:i + 1, :, :]
            # print(zpi_window.shape)
            index += 3
            x_conv = self.ZGCN(brain_graph_window_final, brain_graph_window, node_embedding, zpi_window)
            brain_conv.append(x_conv)
        x = torch.stack(brain_conv, dim=1)  # batch_size*Window_Num*NVertices*NVertices

        gcn_list = []
        gcn_0 = self.gcn(x[:, 0, :, :], adj_window[:, 0, :, :])
        gcn_list.append(gcn_0)
        _, h = self.GRU_Layer[0](gcn_0)
        for i in range(1, self.Window_Num):
            gcn_output = self.gcn(x[:, i, :, :], adj_window[:, i, :, :])
            gcn_list.append(gcn_output)
            output, h = self.GRU_Layer[i](gcn_output, h)


        print(output.shape)
        x = torch.stack(gcn_list, dim=1)
        output_pred = self.predict(output)
        # print(output)

        return x, output, output_pred


























