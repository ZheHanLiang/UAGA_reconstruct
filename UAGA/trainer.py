##########################################################################
# @File: main.py
# @Author: Zhehan Liang
# @Date: 1/11/2020
# @Intro: 训练器函数
##########################################################################

import os
import torch
import scipy
import scipy.linalg
from torch.autograd import Variable
from torch.nn import functional as func
from utils import get_optimizer, get_two_subgraph_scores

class Trainer(object):
    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        初始化训练器
        """
        # 传递params的参数
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
        self.best_result = -1e12

        # 设置优化器
        if hasattr(params, 'map_optimizer'): # 如果参数中设置了生成器优化器
            optim_fn, optim_lr = get_optimizer(params.map_optimizer) # 根据生成器参数选择对应的优化器
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_lr) # 将生成器的模型参数加入优化器
        if hasattr(params, 'dis_optimizer'): # 如果参数中设置了鉴别器优化器
            optim_fn, optim_lr = get_optimizer(params.dis_optimizer) # 根据鉴别器参数选择对应的优化器
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_lr) # 将鉴别器的模型参数加入优化器
        else:
            assert discriminator is None, "Without discriminator!" # 校验是否设置了鉴别器参数
    
    def orthogonalize(self):
        """
        正交化生成网络
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def get_dis_xy(self, requires_grad):
        """
        获取鉴别器每次输入的batch和对应的标签
        """
        # 随机选择节点
        bs = self.params.batch_size
        assert bs <= self.params.emb_num, "Batch size is greater than amount of nodes!" # 校验batch size是否小于总节点数
        src_node = torch.LongTensor(bs).random_(self.params.emb_num)
        tgt_node = torch.LongTensor(bs).random_(self.params.emb_num)
        if self.params.cuda:
            src_node = src_node.cuda()
            tgt_node = tgt_node.cuda()

        # 获取对应节点的嵌入，其中需要对tensor进行Variable化
        src_emb = self.src_emb[Variable(src_node, requires_grad=False)]
        tgt_emb = self.tgt_emb[Variable(tgt_node, requires_grad=False)]
        src_emb = self.mapping(Variable(src_emb.data, requires_grad=requires_grad)) # 源域嵌入部分先通过生成器，得到生成后的嵌入
        tgt_emb = Variable(tgt_emb.data, requires_grad=requires_grad)

        # 设置输入与标签
        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.FloatTensor(2 * bs).zero_()
        y[:bs] = 1 - self.params.dis_smooth # 源域部分标签设置为1-dis_smooth
        y[bs:] = self.params.dis_smooth # 目标域部分标签设置为dis_smooth
        y = Variable(y.cuda() if self.params.cuda else y) # Variable化

        return x, y

    def dis_step(self, dis_cost):
        """
        训练鉴别器
        """
        self.discriminator.train() # 鉴别器网络模式设置为train
        self.mapping.eval() # 生成器网络模式设置为eval
        # loss函数
        x, y = self.get_dis_xy(requires_grad=False) # 获取输入和标签
        preds = self.discriminator(Variable(x.data)) # 输入通过鉴别器的到预测结果
        loss = func.binary_cross_entropy(preds, y) # 用二元交叉熵计算loss
        dis_cost.append(loss.item()) # 记录鉴别器的loss
        # 优化
        self.dis_optimizer.zero_grad() # 清零上一次训练的梯度
        loss.backward() # 将loss进行反向传递
        self.dis_optimizer.step() # 执行一步网络优化，更新网络参数

    def map_step(self, map_acc):
        """
        训练生成器
        """
        self.discriminator.eval() # 鉴别器网络模式设置为eval
        self.mapping.train() # 生成器网络模式设置为train
        # loss函数
        x, y = self.get_dis_xy(requires_grad=True) # 获取输入和标签
        preds = self.discriminator(x) # 输入通过鉴别器的到预测结果
        loss = func.binary_cross_entropy(preds, 1-y) # 用二元交叉熵计算loss，但是这里的标签与训练鉴别器时是相反的，起到反向优化的效果
        # 计算准确率acc
        label = torch.ones_like(preds) # 创建存放二值化标签的tensor
        label[preds<0.5] = self.params.dis_smooth # 小的预测值都归为dis_smooth
        label[preds>=0.5] = 1 - self.params.dis_smooth # 大的预测值都归为1-dis_smooth
        num_correct = torch.eq(label, y).sum() # 求label和y相同元素的个数
        map_acc.append(num_correct.item()/self.params.batch_size/2.0) # 计算准确率
        # 优化
        self.map_optimizer.zero_grad() # 清零上一次训练的梯度
        loss.backward() # 将loss进行反向传递
        self.map_optimizer.step() # 执行一步网络优化，更新网络参数
        self.orthogonalize()

    def save_best(self, logging):
        """
        保存top10结果最好时的生成器参数
        """
        if logging['Graph matching accuracy'][2] > self.best_result:
            self.best_result = logging['Graph matching accuracy'][2] # 最好结果记录到self.best_result中
            W = self.mapping.weight.data.cpu().numpy() # 读取此时生成器的参数
            path = os.path.join(self.params.best_result_dir, 'best_mapping.pth')
            torch.save(W, path)

    def update_lr(self):
        """
        当优化器是SGD时，每次训练后缩小学习率
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            self.map_optimizer.param_groups[0]['lr'] = new_lr

    def reload_best(self):
        """
        重新载入结果最好的生成器参数
        """
        path = os.path.join(self.params.best_result_dir, 'best_mapping.pth')
        assert os.path.isfile(path), "Path of reloading isn't existing!" # 校验模型路径是否正确
        print("Reloading the best model ...")
        reload_model = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert reload_model.size() == W.size(), "Size of model is error!" # 校验模型尺寸是否相同
        W.copy_(reload_model.type_as(W)) # 将最优模型的参数进行赋值
        print("Reloaded the best model!")

    def build_nodes_pairs(self):
        """
        构建节点匹配对应关系
        """
        src_emb = self.mapping(self.src_emb).data
        tgt_emb = self.tgt_emb.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.pairs = get_two_subgraph_scores(src_emb.cpu().numpy(), tgt_emb.cpu().numpy(), self.params.evaluate_method, operation='match')

    def procrustes(self):
        """
        通过procrustes方法更新生成器参数
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        A = self.src_emb.data
        B = self.tgt_emb.data[self.pairs]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))
