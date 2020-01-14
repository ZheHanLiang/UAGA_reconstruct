##########################################################################
# @File: model.py
# @Author: Zhehan Liang
# @Date: 1/10/2020
# @Intro: GAN的模型函数，其中Discriminator是鉴别器，mapping是生成器
##########################################################################

import time
import torch
from torch import nn

# from .utils import 

class Discriminator(nn.Module):
    """
    鉴别器函数
    """
    def __init__(self, params):
        super(Discriminator, self).__init__()

        # 传递params的参数
        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout

        layers = [] # 初始化layers
        for i in range(self.dis_layers + 1): # 依次往layers里面添加需要的网络层
            input_dim = self.emb_dim if i==0 else self.dis_hid_dim
            output_dim = 1 if i==self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim)) # 往layers里面添加线性层，即全连接层
            if i < self.dis_layers: # 最后一个线性层后面不添加LeakyReLU
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid()) # 最后一个线性层后不用LeakyReLU，用sigmoid
        self.layers = nn.Sequential(*layers) # 把得到的layers传入构造器

    def forward(self, x):
        assert x.dim()==2, "Dimension of x is error!" # 校验输入x的维度是否符合要求
        assert x.size(1)==self.emb_dim, "Length of x is error!" # 校验输入x第二维的size是否符合要求
        return self.layers(x).view(-1) # .view()的作用和numpy中的.resize()类似


def build_model(params):
    """
    构建GAN模型
    """
    time_head = time.time() # 记录开始时间
    # 初始化mapping，即生成器
    mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False) # mapping设置为线性层
    mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim))) # 将mapping初始化为对角矩阵
    # 初始化discriminator
    discriminator = Discriminator(params)
    # 初始化cuda
    if params.cuda:
        mapping.cuda()
        discriminator.cuda()
    time_tail = time.time() # 记录完成时间
    print("Model has been built!\tTime: %.3f"%(time_tail-time_head))

    return mapping, discriminator
