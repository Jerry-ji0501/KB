import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import pdist,squareform
from CNN import CNN
import numpy as np

# 计算 ZPI 的空间卷积和时间卷积层

class TSGCN(nn.Module):
    def __init__(self, dim_in, dim_out, link_len, emb_dim, window_len):  # dim_in 特征的数目 dim_out 通道数
            super(TSGCN,self).__init__()
            self.link_len = link_len
            self.weights_pool = nn.Parameter(torch.FloatTensor(emb))