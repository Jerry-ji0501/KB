import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import datetime
import pandas as pd
import numpy as np
import torch.nn.functional as F
from CognitionAlignment import *
from BGI import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from adabelief_pytorch import AdaBelief

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix



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
        self.weights = nn.Parameter(torch.randn(input_dim,output_dim))
        if bias:
            self.bias = nn.Parameter(torch.randn(input_dim,output_dim))
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
        :param x:(node_num,embed_dim)
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
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 181 * 7, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 3)
        )
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.tanh(x)
        return x











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










class CA(nn.Module):
    def __init__(self,KG_input_dim,output_dim,dropout_rate,
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
                 attention_prob_dropout_prob,num_layers):
        super(CA,self).__init__()
        self.DenseLayer = DenseLayer(input_dim,output_dim)
        self.FlattenLayer = FlattenLayer()
        self.encoder = GraphTransformerEncoder(dropout_rate,
                 KG_num,
                 embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 ffn_size,
                 num_layer)
        self.decoder = GraphTransformerDecoder(hidden_size,dropout_rate,num_in_degree,embed_dim,num_out_degree,num_decoder_layers,num_heads,ffn_size)
        self.p_net = PriorNetwork(KG_embed_dim,num_heads,KG_input_dim,output_dim)
        self.q_net = RecognitionNetwork(num_layers,embed_dim,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate)
        self.self_att = nn.MultiheadAttention(embed_dim,num_heads)
        self.mult_att = nn.MultiheadAttention(embed_dim,num_heads)
        self.norm = LayerNorm(62)
        self.fn_fc = nn.Linear(embed_dim*KG_num,3)
        self.Sigmoid = torch.nn.Sigmoid()


    def forward(self,KG_embed_vector,BG_embed_vector,in_degree,out_degree):


        KG_hidden_state = self.encoder(KG_embed_vector,in_degree,out_degree)

        z_p = self.p_net(KG_hidden_state)
        z_q = self.q_net(BG_embed_vector)
        A_L = torch.matmul(z_p, z_q.transpose(1, 2))

        A_L = torch.softmax(A_L, dim=-1)



        z_q = F.pad(z_q,(0,z_p.shape[2]-z_q.shape[2],0,z_p.shape[1]-z_q.shape[1],0,0),"constant",0)

        cov1 = torch.ones_like(z_p)
        cov2 = torch.ones_like(z_q)
        p = torch.distributions.normal.Normal(z_p,cov1)
        q = torch.distributions.normal.Normal(z_q,cov2)




        BG_hidden_state = torch.matmul(KG_hidden_state.transpose(1,2),A_L).transpose(1,2)

        BG_Graph_Construct = self.decoder(BG_hidden_state)
        BG_Graph_Construct = self.norm(BG_Graph_Construct)


        return BG_Graph_Construct,p,q,A_L






class DenseLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        output = self.fc(x)
        return output



class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer,self).__init__()

    def forward(self,x):
        return x.view(x.shape[0],-1)



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

    def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias




#PriorNetwork
class PriorNetwork(nn.Module):
    def __init__(self,embed_dim,num_heads,input_dim,output_dim):
        super(PriorNetwork,self).__init__()
        self.attentionLayer = AttentionLayer(embed_dim,num_heads)
        self.Dense = DenseLayer(input_dim,output_dim)


    def forward(self,KG_embed_vector):
        KG_update_vector ,KG_attn= self.attentionLayer(KG_embed_vector)
        #print(KG_update_vector.shape)
        KG_update_vector = self.Dense(KG_update_vector)
        z,attn_score = self.attentionLayer(KG_update_vector)

        return z






class AttentionLayer(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super( AttentionLayer, self ).__init__()
        self.attention = nn.MultiheadAttention(embed_dim,num_heads)

    def forward(self,hidden_state):
        attn,context_vector = self.attention(query=hidden_state,key=hidden_state,value=hidden_state)
        return attn,context_vector







class DenseLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        output = self.fc(x)
        return output




class MultiAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiAttention, self).__init__()

        self.num_heads = num_heads
        self.attention_size = hidden_size // num_heads
        self.scale = self.attention_size ** -0.5

        self.query_layer = nn.Linear(hidden_size, num_heads * self.attention_size)
        self.key_layer = nn.Linear(hidden_size, num_heads * self.attention_size)
        self.value_layer = nn.Linear(hidden_size, num_heads * self.attention_size)

        self.dropout_layer = nn.Dropout(attention_dropout_rate)
        self.ouput_layer = nn.Linear(num_heads * self.attention_size, hidden_size)

    def forward(self, q, k, v, attention_bias=None):
        orig_q_size = q.size()

        d_k = self.attention_size
        d_v = self.attention_size
        batch_size = q.size(0)

        q = self.query_layer(q).view(batch_size, 1, self.num_heads, d_k)
        k = self.key_layer(k).view(batch_size, 1, self.num_heads, d_k)
        v = self.value_layer(v).view(batch_size, 1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        v = v.transpose(1, 2)

        # Attention_score computing
        q = q * self.scale
        x = torch.matmul(q, k)
        if attention_bias is not None:
            x = x + attention_bias
        x = torch.softmax(x, dim=3)
        x = self.dropout_layer()
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.ouput_layer(x)
        assert x == orig_q_size
        return x






#Recognition Network
class RecognitionNetwork(nn.Module):
    '''
    num_layers:  The number of GNN Layers
    embed_dim: BG_embedding dimension
    hidden_size:BG_hidden size
    num_heads:
    ...
    Input of the network:
    KG_embed_vector:
    A:

    Output of the network:
    z:


    '''
    def __init__(self,num_layers,embed_dim,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate):
        super(RecognitionNetwork, self).__init__()

        self.gnn = GNN(num_layers,embed_dim,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate)
        self.AttentionLayer = AttentionLayer(embed_dim,num_heads)
        self.Dense = DenseLayer(input_dim=64,output_dim=64)
        self.Dense1 = DenseLayer(input_dim=62,output_dim=64)

    def forward(self,BG_embed_vector):
        BG_embed_vector = self.Dense1(BG_embed_vector)
        print(BG_embed_vector.shape)
        hidden_state  = self.gnn(BG_embed_vector)
        hidde_state,attn = self.AttentionLayer(hidden_state)
        #print(attn.shape)
        #print(hidde_state.shape)
        z = self.Dense(hidde_state)

        return z











class GNN(nn.Module):
    def __init__(self,num_layers,embed_dim,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate):
        super(GNN, self).__init__()
        self.n_layers = num_layers
        self.LayerNorm = nn.LayerNorm(embed_dim)
        layers = GATlayers (hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate)
        self.gnn_layers = nn.ModuleList([copy.deepcopy(layers) for _ in range (num_layers)])


    def forward(self,graph_vector):
        graph_vector = self.gnn_layers[0](graph_vector)
        for i in range(1,self.n_layers,1):
            graph_vector = self.gnn_layers[i](graph_vector)

        return graph_vector







class GATlayers(nn.Module):
    def __init__(self,hidden_size,num_heads,attention_prob_dropout_prob,dropout_rate):
        super(GATlayers,self).__init__()
        self.att_layer = SelfAttention(hidden_size,num_heads,attention_prob_dropout_prob)
        self.Output = GATOutput(hidden_size,dropout_rate)

    def forward(self,graph_vector):

        attn_prob,graph_vector = self.att_layer(hidden_states=graph_vector)
        #print(attn_prob.shape)



        graph_vector = self.Output(graph_vector)

        return graph_vector







class AttentionLayer(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super( AttentionLayer, self ).__init__()
        self.attention = nn.MultiheadAttention(embed_dim,num_heads)

    def forward(self,hidden_state):
        context_vector,attn = self.attention(query=hidden_state,key=hidden_state,value=hidden_state)
        return context_vector,attn





class SelfAttention(nn.Module):
    def __init__(self,hidden_size,num_heads,attention_prob_dropout_prob,output_attention=False,keep_multi_head = False):
        super(SelfAttention, self).__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                    "The hidden size (%d) is not a multiple of the number of attention "
                    "heads (%d)" % (hidden_size, num_heads))
        self.att_drop = True
        self.output_attention = output_attention
        self.keep_multi_head = keep_multi_head

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size/num_heads)
        self.all_head_size = self.num_attention_heads*self.attention_head_size

        self.query = nn.Linear(hidden_size,self.all_head_size)
        self.key = nn.Linear(hidden_size,self.all_head_size)
        self.value = nn.Linear(hidden_size,self.all_head_size)

        self.dropout = nn.Dropout(attention_prob_dropout_prob)
        self.do_softmax = True

    def transpose_for_score(self,x):
        new_x_shape = x.size()[:-1]+(self.num_attention_heads,self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0,2,1,3)

    def forward(self,hidden_states):
        batch_size = hidden_states.shape[0]
        query_layers = self.query(hidden_states)
        key_layers = self.key(hidden_states)
        value_layers = self.value(hidden_states)


        query_layers =self.transpose_for_score(query_layers)
        key_layers =self.transpose_for_score(key_layers)
        value_layers = self.transpose_for_score(value_layers)

        attention_score = torch.matmul(query_layers,key_layers.transpose(-1,-2))
        attention_score = attention_score/math.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_score)
        attention_probs = self.dropout(attention_probs)
        context_layers =torch.matmul(attention_probs,value_layers)



        context_layers = context_layers.permute(0,2,1,3).contiguous()
        #Do  you need to shape reverse
        #print(self.all_head_size)
        #new_context_layer_shape = context_layers.size()[:2] + (self.all_head_size,)
        context_layers = context_layers.view((batch_size,62,64))#(*new_context_layer_shape)
        #print(context_layers.shape)


        return attention_probs,context_layers


class GATOutput(nn.Module):
    def __init__(self,hidden_size,dropout_rate):
        super(GATOutput,self).__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.relu = nn.ReLU()
        self.LayerNorm =  nn.LayerNorm(hidden_size,eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,hidden_state):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.relu(self.LayerNorm(hidden_state))
        hidden_state = self.LayerNorm(hidden_state)
        output = self.dropout(hidden_state)
        return output








class DenseLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        output = self.fc(x)
        return output




class MultiAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiAttention, self).__init__()

        self.num_heads = num_heads
        self.attention_size = hidden_size // num_heads
        self.scale = self.attention_size ** -0.5

        self.query_layer = nn.Linear(hidden_size, num_heads * self.attention_size)
        self.key_layer = nn.Linear(hidden_size, num_heads * self.attention_size)
        self.value_layer = nn.Linear(hidden_size, num_heads * self.attention_size)

        self.dropout_layer = nn.Dropout(attention_dropout_rate)
        self.ouput_layer = nn.Linear(num_heads * self.attention_size, hidden_size)

    def forward(self, q, k, v, attention_bias=None):
        orig_q_size = q.size()

        d_k = self.attention_size
        d_v = self.attention_size
        batch_size = q.size(0)

        q = self.query_layer(q).view(batch_size, 1, self.num_heads, d_k)
        k = self.key_layer(k).view(batch_size, 1, self.num_heads, d_k)
        v = self.value_layer(v).view(batch_size, 1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)
        v = v.transpose(1, 2)

        # Attention_score computing
        q = q * self.scale
        x = torch.matmul(q, k)
        if attention_bias is not None:
            x = x + attention_bias
        x = torch.softmax(x, dim=3)
        x = self.dropout_layer(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.ouput_layer(x)
        assert x == orig_q_size
        return x


#Encoder
class GraphTransformerEncoder(nn.Module):
    def __init__(self,
                 dropout_rate:float,
                 KG_num,
                 embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 ffn_size,
                 num_layer
                ):
        super(GraphTransformerEncoder, self).__init__()


        self.dropout = nn.Dropout(dropout_rate)
        self.LayerNorm = nn.LayerNorm(embed_dim)
        self.node_encoder = nn.Embedding(KG_num, embed_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, embed_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, embed_dim, padding_idx=0)

        encoders = [EncoderLayer(embed_dim, num_heads, hidden_size, ffn_size, num_in_degree, num_out_degree)
                    for _ in range(num_layer)]
        self.encoder_layers = nn.ModuleList(encoders)

    def forward(self, KG_embed_vector, in_degree, out_degree):
        #node_features = self.node_encoder(KG_embed_vector)

        node_features = (
                KG_embed_vector + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)
        ).to('cuda')
        output = self.dropout(node_features)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output)

        output = self.LayerNorm(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self,embed_dim,num_heads,hidden_size,ffn_size,num_indegree,num_outdegree):
        super(EncoderLayer, self).__init__()


        #Graph Information Embedding
        self.in_degree = nn.Embedding(num_indegree,hidden_size,padding_idx=0)
        self.out_degree = nn.Embedding(num_outdegree,hidden_size,padding_idx=0)


        self.selfattention = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads)
        self.LayerNorm =  nn.LayerNorm(embed_dim)
        self.layers1 = nn.Linear(embed_dim,ffn_size)
        self.gelu = nn.GELU()
        self.layers2 = nn.Linear(ffn_size,embed_dim)
        self.dropout = nn.Dropout(0.1)


    def  FeedForwardNetwork(self,x):
        x = self.layers1(x)
        x = self.gelu(x)
        x = self.layers2(x)
        return x

    def forward(self,x):
        '''

        :param x:
        :return:x,attn
        Build it upon the Transformer Architecture
        '''
        #Multi-Attention
        residual = x
        x_norm = self.LayerNorm(x)
        x, attn= self.selfattention(query=x_norm,key=x_norm,value=x_norm)
        x = self.dropout(x)
        x = x + residual
        #print(x.shape)

        #FeedForward Network
        residual = x
        x = self.LayerNorm(x)
        x = self.FeedForwardNetwork(x)
        x = self.dropout(x)
        x = residual + x
        return x



class GraphNodeFeatures(nn.Module):
    def __init__(self,KG_num,
                 embed_dim,
                 num_in_degree,
                 num_out_degree,
                 num_heads,
                 hidden_size,
                 ffn_size,
                 num_indegree,
                 num_outdegree,
                 num_layer
                 ):
        super(GraphNodeFeatures, self).__init__()

        self.Dropout = nn.Dropout(0.1)
        self.LayerNorm = nn.LayerNorm(embed_dim)
        self.node_encoder = nn.Embedding(KG_num,embed_dim,padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree,embed_dim,padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree,embed_dim,padding_idx=0)
        encoders = [EncoderLayer(embed_dim,num_heads,hidden_size,ffn_size,num_indegree,num_outdegree)
                   for _ in range(num_layer) ]
        self.encoder_layers = nn.ModuleList(encoders)



    def forward(self, KG_embed_vector, in_degree, out_degree):

        node_features = self.node_encoder(KG_embed_vector)
        node_features = (
            node_features+self.in_degree_encoder(in_degree)+self.out_degree_encoder(out_degree)
        )
        output = self.Dropout(node_features)
        for enc_layer in self.encoder_layers:
            output = enc_layer(output)

        output = torch.softmax(output).dim(-1)

        return output








class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

    def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias





#Decoder
class GraphTransformerDecoder(nn.Module):
    def __init__(self,hidden_size,dropout_rate,num_in_degree,embed_dim,num_out_degree,num_decoder_layers,num_heads,ffn_size):
        super(GraphTransformerDecoder,self).__init__()

        self.LayerNorm = nn.LayerNorm(hidden_size,eps=1e-12)
        self.softmax = nn.Softmax(dim=-1)
        self.Dropout = nn.Dropout(dropout_rate)
        self.in_degree_encoder = nn.Embedding(num_in_degree, embed_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, embed_dim, padding_idx=0)

        decoders = [DecoderLayer(embed_dim,num_heads,ffn_size)
                   for _ in range (num_decoder_layers)]
        self.decoder_layers =nn.ModuleList(decoders)
        self.layers_fn = nn.Linear(64, 62)


    def forward(self,BG_hidden_state):

        BG_hidden_state = self.Dropout(BG_hidden_state)
        for dec_layer in self.decoder_layers:
            output = dec_layer(BG_hidden_state)
        BG_Construct = output
        BG_Construct = self.layers_fn(BG_Construct)
        ##print(BG_Construct.shape)
        #BG_Construct = nn.Softmax(output)
        return BG_Construct


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim,num_heads,ffn_size):
        super(DecoderLayer,self).__init__()
        self.selfattention = nn.MultiheadAttention(embed_dim,num_heads)
        self.LayerNorm =  nn.LayerNorm(embed_dim)
        self.layers1 = nn.Linear(embed_dim,ffn_size)
        self.gelu = nn.GELU()
        self.layers2 = nn.Linear(ffn_size,embed_dim)

        self.dropout = nn.Dropout(0.1)


    def  FeedForwardNetwork(self,x):
        x = self.layers1(x)
        x = self.gelu(x)
        x = self.layers2(x)
        return x




    def forward(self,x):
        # Multi-Attention
        residual = x
        x_norm = self.LayerNorm(x)
        x, attn = self.selfattention(query=x_norm, key=x_norm, value=x_norm)
        x = self.dropout(x)
        x = x + residual
        #print(x.shape)

        # FeedForward Network
        residual = x
        x = self.LayerNorm(x)
        x = self.FeedForwardNetwork(x)
        x = self.dropout(x)
        x = residual + x

        # Multi-Attention
        residual = x
        x_norm = self.LayerNorm(x)
        x, attn = self.selfattention(query=x_norm, key=x_norm, value=x_norm)
        x = self.dropout(x)
        x = x + residual


        return x




class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(LayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

    def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


















class FusionLayer(nn.Module):
    def __init__(self, node_num, graph_num, dim_in, dim_out, window_len, link_len, emb_dim, num_layers, Window_Num,
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
        super(FusionLayer, self).__init__()
        self.BrainGraphIntergration = BGI(node_num, graph_num, dim_in, dim_out, window_len, link_len, emb_dim,
                                          num_layers, Window_Num)
        self.CognitionAlignment = CA(KG_input_dim, output_dim, dropout_rate,
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
                                  attention_prob_dropout_prob, num_layers)
        self.conv = ConvModule()
        self.BG_weight = nn.Parameter(torch.randn(1454, 62))
        self.KG_weight = nn.Parameter(torch.rand(64, 62))

    def forward(self, KG_embeddings, in_degree, out_degree, x_all, zigzag_PI, node_embeddings):
        output_pro, output_all, output_pred = self.BrainGraphIntergration(x_all, zigzag_PI, node_embeddings)
        A_L_List = []
        BG_Graph_Construct_list = []
        p_list = []
        q_list = []

        for i in range(9):
            BG_Graph_Construct, p, q, A_L = self.CognitionAlignment(KG_embeddings[:, i, :, :].to('cuda'),
                                                                output_pro[:, i, :, :].to('cuda'),
                                                                in_degree[:, i, :].to('cuda'),
                                                                out_degree[:, i, :].to('cuda'))
            # print(A_L.shape)
            BG_Graph_Construct_list.append(BG_Graph_Construct)
            p_list.append(p)
            q_list.append(q)
            A_L_List.append(A_L)
        A_L_all = torch.zeros(A_L.shape, ).cuda()
        for tensor in A_L_List:
            A_L_all += tensor

        # = torch.sum(A_L_List)
        KG_all = torch.sum(KG_embeddings, dim=1)

        KG_all = torch.matmul(KG_all, self.KG_weight)
        output_all = torch.matmul(self.BG_weight, output_all)

        # max_row = max(A_L_all.shape[1],KG_all.shape[1],output_all.shape[1])

        # max_line = max(A_L_all.shape[2],KG_all.shape[2],output_all.shape[2])

        # Padding first dimension
        # A_L_all = F.pad(A_L_all,(0,0,0,max_row-A_L_all.shape[1],0,0),"constant",0)
        # KG_all = F.pad(KG_all,(0,0,0,max_row-KG_all.shape[1],0,0),"constant",0)
        # output_all = F.pad(output_all,(0,0,0,max_row-output_all.shape[1],0,0),"constant",0)

        # A_L_all = F.pad(A_L_all,(0,max_line-A_L_all.shape[2],0,0,0,0),"constant",0)
        # KG_all = F.pad(KG_all,(0,max_line-KG_all.shape[2],0,0,0,0),"constant",0)
        # output_all = F.pad(output_all,(0,max_line-output_all.shape[2],0,0,0,0),"constant",0)

        # print(A_L_all.shape)
        # print(KG_all.shape)
        # print(output_all.shape)  #

        feature_all = torch.stack([A_L_all, KG_all, output_all], dim=1)
        # print(feature_all.shape)
        output = self.conv(feature_all)
        return output, output_pro, BG_Graph_Construct_list, p_list, q_list, A_L_all, output_pred,





def train(model, train_dataloader,val_dataloader, mae, criterion, device, train_epoch, batch_size, lr,
          scheduler_type='Cosine'):
    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)  # xavier_normal_

    model.apply(init_xavier)
    print(device)
    optimizer = AdaBelief(model.parameters(), lr=lr, weight_decay=0.3, eps=1e-16, betas=(0.9, 0.999))
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=train_epoch)
        print('using CosineAnnealingLR')

    now = datetime.datetime.now()
    log_dir = now.strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = './logs/' + log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)  # tensorboard可视化

    train_loss1 = []
    train_loss2 = []
    train_loss_kl = []
    train_acces = []
    eval_acces = []
    best_acc = 0.0

    for epoch in range(train_epoch):
        model.train()
        train_acc = 0
        train_acc_bgi = 0
        train_loss = 0

        for batch_idx, (x_all, zigzag_all, node_embeddings, kg_embeddings, in_degree, out_degree, labels) in enumerate(
                train_dataloader):
            x_all = x_all.to(device)
            zigzag_all = zigzag_all.to(device)
            node_embeddings = node_embeddings.to(device)
            kg_embeddings = kg_embeddings.to(device)
            in_degree = in_degree.to(device)
            out_degree = out_degree.to(device)
            labels = labels.to(device)

            output, output_pro, BG_Graph_Construct_list, p_list, q_list, A_L_all, output_pred = model(kg_embeddings,
                                                                                                      in_degree,
                                                                                                      out_degree, x_all,
                                                                                                      zigzag_all,
                                                                                                      node_embeddings)

            BG_Graph_Construct = torch.stack(BG_Graph_Construct_list, dim=1)

            print("output_pred", output_pred.shape)
            loss1 = mae(output_pro, BG_Graph_Construct)
            loss2 = criterion(output, labels)
            loss_bg = criterion(output_pred, labels)
            loss_kl = 0
            for p, q in zip(p_list, q_list):
                loss_kl += torch.distributions.kl.kl_divergence(p, q).sum()

            print("loss1", loss1)
            print("loss2", loss2)
            print("loss_kl", loss_kl)
            kl_lambda = 0.00001
            loss = loss1 + loss2 + kl_lambda * loss_kl + loss_bg
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
            optimizer.step()
            # print(output)
            _, pred = output.max(1)
            _, pred_bgi = output_pred.max(1)

            num_correct = (pred == labels).sum().item()
            num_correct_bgi = (pred_bgi == labels).sum().item()
            # print(num_correct)

            acc = num_correct / len(labels)
            acc_bgi = num_correct_bgi / len(labels)
            train_acc += acc
            train_acc_bgi += acc_bgi
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_dataloader)  # 计算平均值
        train_acc /= len(train_dataloader)  # 计算平均值
        train_acc_bgi /= len(train_dataloader)  # 计算平均值
        print(f"epoch: {epoch}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Acc_bgi: {train_acc_bgi:.4f}")
        # print(len(train_dataloader))
        writer.add_scalar('loss', train_loss, epoch + 1)
        writer.add_scalar('loss/loss1', loss1, epoch + 1)
        writer.add_scalar('loss/loss2', loss2, epoch + 1)
        writer.add_scalar('loss/loss_kl', loss_kl, epoch + 1)
        writer.add_scalar('loss/loss_bg', loss_bg, epoch+ 1)
        writer.add_scalar('acc', train_acc, epoch + 1)
        writer.add_scalar('acc_bgi', acc_bgi, epoch + 1)

        train_acces.append(train_acc)
        train_loss1.append(loss1.item())
        train_loss2.append(loss2.item())
        train_loss_kl.append(loss_kl.item())

        model.eval()
        val_loss = 0
        val_acc = 0
        val_acc_bgi = 0
        with torch.no_grad():
            for batch_idx, (
                    x_all, zigzag_all, node_embeddings, kg_embeddings, in_degree, out_degree, labels) in enumerate(
                val_dataloader):
                x_all = x_all.to(device)
                zigzag_all = zigzag_all.to(device)
                node_embeddings = node_embeddings.to(device)
                kg_embeddings = kg_embeddings.to(device)
                in_degree = in_degree.to(device)
                out_degree = out_degree.to(device)
                labels = labels.to(device)
                output, output_pro, BG_Graph_Construct_list, p_list, q_list, A_L_all, output_pred = model(kg_embeddings,
                                                                                                          in_degree,
                                                                                                          out_degree,
                                                                                                          x_all,
                                                                                                          zigzag_all,
                                                                                                          node_embeddings)

                loss = criterion(output, labels)
                loss_bgi = criterion(output_pred, labels)

                _, pred = output.max(1)
                _, pred_bgi = output_pred.max(1)

                num_correct = (pred == labels).sum().item()
                num_correct_bgi = (pred_bgi == labels).sum().item()

                val_loss += loss.item()

                acc = num_correct / len(labels)
                acc_bgi = num_correct_bgi / len(labels)
                val_acc += acc
                val_acc_bgi += acc_bgi

        val_loss /= len(val_dataloader)  # 计算平均值
        val_acc /= len(val_dataloader)  # 计算平均值
        val_acc_bgi /= len(val_dataloader)  # 计算平均值
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model, 'best_model.pkl')
        print(
            f"Validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}, validation accuracy_bgi: {val_acc_bgi:.4f}")
        writer.add_scalar('Val/Loss', val_loss, epoch + 1)
        writer.add_scalar('Val/Acc', val_acc, epoch + 1)
        writer.add_scalar('Val/Acc_bgi', val_acc_bgi, epoch + 1)





    return A_L_all


def test(network, test_dataloader, device,activate_node):
    network.eval()

    prediction = []
    true = []
    prediction_bgi = []

    back_dic = {}
    # criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for _, (x_all, zigzag_all, node_embeddings, kg_embeddings, in_degree, out_degree, labels) in enumerate(
                test_dataloader):
            x_all = x_all.to(device)
            zigzag_all = zigzag_all.to(device)
            node_embeddings = node_embeddings.to(device)
            kg_embeddings = kg_embeddings.to(device)
            in_degree = in_degree.to(device)
            out_degree = out_degree.to(device)
            labels = labels.to(device)
            output, output_pro, BG_Graph_Construct_list, p_list, q_list, A_L_all, output_pred = model(kg_embeddings,
                                                                                                      in_degree,
                                                                                                      out_degree, x_all,
                                                                                                      zigzag_all,
                                                                                                      node_embeddings)

            # loss = criterion(outputs_all, labels)
            activate_bg_node = np.argmax(A_L_all[:,activate_node,:].cpu().numpy(), axis=1)
            print(activate_bg_node)
            print(labels)
            for i in range (len(labels)):
                back_dic[labels[i].item()] = activate_bg_node[i]

            print(back_dic)

            _, pred = output.max(1)
            _, pred_bgi = output_pred.max(1)
            labels = labels.cpu()
            pred = pred.cpu()
            pred_bgi = pred_bgi.cpu()
            labels = labels.detach().numpy()
            pred = pred.detach().numpy()
            pred_bgi = pred_bgi.detach().numpy()
            labels = labels.tolist()
            pred = pred.tolist()
            pred_bgi = pred_bgi.tolist()

            true.extend(labels)

            prediction.extend(pred)
            prediction_bgi.extend(pred_bgi)
        print(prediction)
        print(true)
        print(prediction_bgi)

    accuracy = accuracy_score(true, prediction)
    f1 = f1_score(true, prediction, average='macro')
    recall = recall_score(true, prediction, average='macro')
    precision = precision_score(true, prediction, average='macro')
    kappa = cohen_kappa_score(true, prediction)

    print('acc', accuracy)
    print('f1', f1)
    print('recall', recall)
    print('precision', precision)
    print('confusion_matrix', confusion_matrix(true, prediction))
    print('kappa', kappa)

    accuracy_bgi = accuracy_score(true, prediction_bgi)
    f1_bgi = f1_score(true, prediction_bgi, average='macro')
    recall_bgi = recall_score(true, prediction_bgi, average='macro')
    precision_bgi = precision_score(true, prediction_bgi, average='macro')
    kappa_bgi = cohen_kappa_score(true, prediction_bgi)

    print('acc_bgi', accuracy_bgi)
    print('f1_bgi', f1_bgi)
    print('recall_bgi', recall_bgi)
    print('precision_bgi', precision_bgi)
    print('confusion_matrix_bgi', confusion_matrix(true, prediction_bgi))
    print('kappa_bgi', kappa_bgi)




    return accuracy, f1, recall, precision, kappa, accuracy_bgi, f1_bgi, recall_bgi, precision_bgi, kappa_bgi





































        









