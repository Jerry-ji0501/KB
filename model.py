import torch
import torch.nn as nn

from Fusion import  *
from Sychronization import *
from GRU import *
from loadData import load_feature_data,load_topo_data
from Add_Windows import Add_Windows
import os


'''
    This function is aim to combine the BrainGraphIntergration , Signal Sychronization and finally Fusion module.
    Input of Model:
    
    BGI:
    x_all,zigzag_PI,node_embeddings
    
    SYN:
    KG_embed_vector,BG_embed_vector,in_degree,out_degree
    
    Fusion:
    KG_feature,KG_adj,A_L,BG_feature,BG_adj
    


'''

class model(nn.Module):
    def __init__(self,node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers,
        input_dim, output_dim, dropout_rate,
        KG_num,
        embed_dim,
        num_in_degree,
        num_out_degree,
        num_heads,
        hidden_size,
        ffn_size,
        num_layer
        , num_decoder_layers,
        attention_prob_dropout_prob,
                 input_size,
         hidden_dim,  concat_dim,  bias = False

    ):
        super(model, self).__init__()
        self.bgi = BGI(node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers)
        #(x_all,zigzag_PI,node_embedding)-> Brain_Interagration(adj_all)

        self.syn = SYN(input_dim,output_dim,dropout_rate,
                 KG_num,
                 embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 ffn_size,
                 num_layer
                 ,num_decoder_layers,
                 attention_prob_dropout_prob,num_layers)
        #(KG_embed_vector,BG_embed_vector,in_degree,out_degree)->(BG_Construct,p,q,A_L)

        self.Fusion = Fusion(input_size,hidden_dim,output_dim,concat_dim,dropout_rate,bias)
        #(KG_embed_vector,KG_adj,A_L,BG_embed_vector,BG_adj)->prediction



    def forward(self,x_all,zigzag_PI,node_embeddings,KG_embed_vector,BG_embed_vector,in_degree,out_degree,KG_adj):
        outputs_all, outputs_pro = self.bgi(x_all, zigzag_PI, node_embeddings)

        BG_Graph_Construct, p, q, A_L = self.syn(KG_embed_vector, BG_embed_vector, in_degree, out_degree)
        print('KG_embed_vector',KG_embed_vector.shape)
        print('KG_adj',KG_adj.shape)
        print('BG_embed_vector',BG_embed_vector.shape)


        print('outputs_all',outputs_all.shape)
        output = self.Fusion(KG_embed_vector,KG_adj,A_L,BG_embed_vector,outputs_all)

        return output,BG_Graph_Construct,p,q


if __name__ == '__main__':
    path = os.getcwd()
    node_num = 62
    graph_num = 27
    dim_in = 10
    dim_out = 62
    window_len = 3
    link_len = 2
    emb_dim = 3
    num_layers = 3
    input_dim = 128
    output_dim = 128
    dropout_rate = 0.1
    KG_num = 128
    embed_dim = 128
    num_in_degree = 128
    num_out_degree = 128
    num_heads = 8
    hidden_size = 128
    ffn_size = 128
    num_layer = 8
    num_decoder_layers = 3
    attention_prob_dropout_prob = 0.1
    input_size = 128
    hidden_dim = 128
    concat_dim = 128
    bias = False

    data_all = load_feature_data(path)

    x_all = torch.FloatTensor(data_all)  # torch.randn(64,graph_num,62,10)
    x_all = x_all.permute((0, 3, 2, 1))
    x_all = x_all.numpy()
    #print(x_all.shape)
    x = Add_Windows(x_all, window_len=3, stride=1)
    x = torch.FloatTensor(x)
    x = x[:,0:4,:,:,:]
    zigzag_PI = torch.randn(4, 1, 100, 100)
    node_embeddings = torch.randn(4, 62, 3)
    node_embeddings = node_embeddings[-1,:,:]
    KG_embed_vector = torch.randn(4,128,128)
    in_degree = torch.randint(0,5,(4,128))
    out_degree = torch.randint(0,5,(4,128))
    BG_embed_vector = torch.randn(4,128,128)
    KG_adj = torch.randn(4, 128, 128)

    network = model(node_num,graph_num,dim_in,dim_out,window_len,link_len,emb_dim,num_layers,
        input_dim, output_dim, dropout_rate,
        KG_num,
        embed_dim,
        num_in_degree,
        num_out_degree,
        num_heads,
        hidden_size,
        ffn_size,
        num_layer
        , num_decoder_layers,
        attention_prob_dropout_prob,
                    input_size,
         hidden_dim,  concat_dim,  bias = False)
    #print(x.shape)
    output = network(x,zigzag_PI,node_embeddings,KG_embed_vector,BG_embed_vector,in_degree,out_degree,KG_adj)






