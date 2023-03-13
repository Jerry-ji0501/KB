
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from Sychronization import *



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



class Fusion(nn.Module):
    def __init__(self,KG_input_dim,BG_input_dim,hidden_dim,output_dim,concat_dim,dropout_rate,bias = False):

        super(Fusion,self).__init__()

        self.concat_dim = concat_dim

        self.KG_Dense = DenseLayer(KG_input_dim,concat_dim)
        self.BG_Dense = DenseLayer(BG_input_dim,concat_dim)
        self.A_L_Dense = DenseLayer(input_dim,concat_dim)
        self.KG_gcn = GCN(KG_input_dim,hidden_dim,output_dim,dropout_rate,bias = bias)
        self.BG_gcn = GCN(BG_input_dim,hidden_dim,output_dim,dropout_rate,bias = bias)
        self.FlattenLayer = FlattenLayer()
        self.fc1 = nn.Linear(32256,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,3)
        self.softmax = nn.Softmax(dim=-1)



    def forward(self,KG_feature,KG_adj,A_L,BG_feature,BG_adj):
        '''

        :param KG_feature:
        :param KG_adj:
        :param A_L:
        :param BG_feature:
        :param BG_adj:
        KG_feature: KG_num*KG_feature_dim
        BG_feature: BG_num*BG_feature_dim
        A_L: max_KG_num*BG_num

        How can i concat of three high level feature for classification?
        '''

        KG_features = self.KG_gcn(KG_adj,KG_feature)
        BG_features = self.BG_gcn(BG_adj,BG_feature)
        KG_features = self.KG_Dense(KG_features)
        #print(KG_features.shape)
        BG_features = self.BG_Dense(BG_features)
        #print(BG_features.shape[-1])
        A_L = self.A_L_Dense(A_L)
        #print(A_L.shape)
        Concat_features = torch.cat((KG_features,BG_features,A_L),dim=1)
        print(Concat_features.shape)
        Flatten_features = self.FlattenLayer(Concat_features)


        hidden_features =self.fc1(Flatten_features)
        output = self.fc2(hidden_features)
        output = torch.nn.functional.softmax(output)
        print(output)


        return output




def train(model,train_dataloader,criterion,device,train_epoch,batch_size,lr,scheduler_type='Cosine'):
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




        for batch_idx ,(KG_adj,KG_features,BG_adj,BG_features,A_L,labels) in enumerate(train_dataloader):
            output = model(KG_features,KG_adj,A_L,BG_features,BG_adj)




            #loss1 = mae(BG_embed_vector,BG_Graph_Construct)
            loss = criterion(output,labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=2)
            optimizer.step()

            _, pred = output.max(1)
            print(pred)
            print(labels)

            num_correct = (pred == labels).sum().item()
            print(num_correct)

            acc = num_correct / (batch_size)
            train_acc += acc


        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, loss.item(), train_acc / len(train_dataloader)))
        #print(len(train_dataloader))
        train_acces.append(train_acc / len(train_dataloader))
        train_loss.append(loss.item())



    return train_acces, train_loss






def showpic(train_acc, train_loss,num_epoch):
    plt.figure()
    plt.subplot(2,1,1)
    print(train_loss)
    plt.plot(1 + np.arange(len(train_loss)), train_loss, linewidth=1.5, linestyle='dashed', label='train_loss')
    plt.subplot(2,1,2)
    plt.plot(1 + np.arange(len(train_acc)), train_acc, linewidth=1.5, linestyle='dashed', label='train_acc')
    #plt.plot(1+np.arange(len(train_loss_kl)),train_loss_kl,linewidth=1.5,linestyle='dashed',label='train_loss_kl')
    plt.grid()
    plt.xlabel('epoch')
    plt.xticks(range(1, 1 + num_epoch,5))
    plt.legend()
    plt.show()
    #plt.plot(1+np.arange(len(eval_acces)),eval_acces,linewidth=1.5, linestyle='dashed', label='eval_acces')
    #plt.show()





if __name__ == '__main__':
    input_dim=128
    hidden_dim = 128
    output_dim = 128
    concat_dim = 128
    dropout_rate = 0.5
    batch_size = 32


    KG_features = torch.randn(32,128,128)
    KG_adj = torch.randn(32,128,128)
    BG_features = torch.randn(32,62,128)
    BG_adj = torch.randn(32,62,62)
    A_L = torch.randn(32,62,128)
    #gcn = GCN(input_dim,hidden_dim,output_dim,dropout_rate)
    #output_KG = gcn(KG_adj,KG_features)
    #output_BG = gcn(BG_adj,BG_features)
    labels = np.array([2,1,1,0,2,0,1,2,2,1,0,0,1,2,2,0,1,0,2,1,0,2,0,0,2,2,0,1,2,0,2,0])
    print(len(labels))
    labels = torch.LongTensor(labels)
    data = torch.utils.data.TensorDataset(KG_adj,KG_features,BG_adj,BG_features,A_L,labels)
    train_dataloader = torch.utils.data.DataLoader(data,batch_size=batch_size,shuffle=True, drop_last=False)

    model = Fusion(input_dim,input_dim,hidden_dim,output_dim,concat_dim,dropout_rate)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')
    train_acc, train_loss = train(model,  train_dataloader,criterion=criterion, device=device, batch_size=batch_size,
                                                                        train_epoch=50, lr=1e-5)

    #showpic(train_acc, train_loss, 5)
    #plt.imshow(A_L[0],cmap='hot',interpolation='nearest')
    #plt.colorbar()
    #plt.show()

    for name, parms in model.named_parameters():
        print(name)
        print(parms.requires_grad)
        print(parms.grad)






        









