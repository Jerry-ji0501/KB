import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, dim_out):
        super(CNN, self).__init__()
        # 接收zpi作为模型输入，然后对其进行卷积核最大池化的操作
        # 提取特征用于 Z-GCNet 的 persistence 表示学习
        # 重点是卷积核的选取和前后维度的变化
        self.dim_out = dim_out  # 通道数
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2),  # channel of ZPI is 1
            nn.Relu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(8, dim_out, kernel_size=3, stride=2),
            nn.Relu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.maxpool = nn.MaxPool2d(5, 5)

    def forward(self, zigzag_window_pi):
        feature = self.features(zigzag_window_pi)
        feature = self.maxpool(feature)
        feature = feature.view(-1, self.dim_out)  # Batch_size, dim_out
        return feature
