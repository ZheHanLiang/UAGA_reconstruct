##########################################################################
# @File: main.py
# @Author: Zhehan Liang
# @Date: 1/9/2020
# @Intro: 训练的主函数
##########################################################################

import numpy as np
import pickle
import argparse
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import initialize_seed, load_embeddings
from model import build_model
from trainer import Trainer
from evaluator import Evaluator

## 限制显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'

## 随机数种子
np.random.seed(1)

## 参数设置
parser = argparse.ArgumentParser(description='UAGA') # 实例化ArgumenParser
parser.add_argument("--seed", type=int, default=1, help="Initialization seed(<=0 to disable)") # 使用add_argument函数添加参数
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension")
parser.add_argument("--map_beta", type=float, default=0.001, help="Beta for orthogonalization")
parser.add_argument("--dis_layers", type=int, default=2, help="Discriminator layers")
parser.add_argument("--dis_hid_dim", type=int, default=2048, help="Discriminator hidden layer dimensions")
parser.add_argument("--dis_dropout", type=float, default=0.1, help="Discriminator dropout")
parser.add_argument("--dis_steps", type=int, default=5, help="Discriminator steps")
parser.add_argument("--map_steps", type=int, default=1, help="Mapping steps")
parser.add_argument("--dis_smooth", type=float, default=0.2, help="Discriminator smooth predictions")
parser.add_argument("--adversarial", type=bool, default=True, help="Use adversarial training")
parser.add_argument("--n_epochs", "-e", type=int, default=30, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")
parser.add_argument("--dis_optimizer", type=str, default="sgd,lr=0.1", help="Discriminator optimizer and learning rate")
parser.add_argument("--map_optimizer", type=str, default="sgd,lr=0.1", help="Mapping optimizer and learning rate")
parser.add_argument("--lr_decay", type=float, default=0.98, help="Learning rate decay (SGD only)")
parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate (SGD only)")
parser.add_argument("--n_refinement", type=int, default=20, help="Number of refinement iterations (0 to disable the refinement procedure)")
parser.add_argument("--dataset", "-d", type=str, default="lastfm", help="Name of dataset")
parser.add_argument("--emb_dir", type=str, default="./data/graph_embedding/", help="Path of embeddings")
parser.add_argument("--best_result_dir", type=str, default="./data/best_mapping_model/", help="Path of best mapping model")
parser.add_argument("--emb_num", "-n", type=int, default=10000, help="Number of nodes")
parser.add_argument("--evaluate_method", type=str, default="CGSS", help="Mathod of evaluating: 'NN'/'CGSS'/'REGAL'")
parser.add_argument("--pair_method", type=str, default="NN", help="Mathod of get pair: 'NN'/'CGSS'")
params = parser.parse_args() # 进行解析

## 初始化随机数种子
initialize_seed(params)

## 读取节点嵌入，嵌入已经转换为tensor格式
src_emb = load_embeddings(params, True)
tgt_emb = load_embeddings(params, False)

## 构造GAN模型
mapping, discriminator = build_model(params)

## 构造训练器
trainer = Trainer(src_emb, tgt_emb, mapping, discriminator, params)

## 构造评估器
evaluator = Evaluator(trainer)

## 对抗训练过程
if params.adversarial:
    print("----> ADVERSARIAL TRAINING STARTING <----") # 显示开始对抗训练
    for n_epoch in range(params.n_epochs):
        # print("Starting adversarial training epoch: %i"%n_epoch) # 显示对抗训练轮数
        time_head = time.time() # 记录开始时间
        dis_cost = [] # 鉴别器loss结果记录
        map_acc = [] # 生成器acc结果记录
        for n_iter in range(int(params.emb_num/params.batch_size)):
            # 鉴别器训练过程
            for _ in range(params.dis_steps):
                trainer.dis_step(dis_cost)
            # 生成器训练过程
            for _ in range(params.map_steps):
                trainer.map_step(map_acc)
        # if n_epoch % 10 == 0:
        time_tail = time.time() # 记录完成时间
        print("Epoch %d  \tdis_cost: %0.5f\tmap_acc:  %0.4f\tTime:  %.5f"%(n_epoch, np.mean(dis_cost), np.mean(map_acc),time_tail-time_head))
        logging = OrderedDict({'n_epoch': n_epoch}) # 创建记录实验结果的有序字典==>print(logging)=OrderedDict([('n_epoch', n_epoch)])
        evaluator.graph_eval(logging) # 训练结果评估
        trainer.save_best(logging) # 保存结果最好时的生成器参数
        trainer.update_lr() # 优化器是SGD时更新学习率
    time_tail = time.time() # 记录完成时间
    print("Epoch %d  \tdis_cost: %0.5f\tmap_acc:  %0.4f\tTime:  %.5f"%(n_epoch, np.mean(dis_cost), np.mean(map_acc),time_tail-time_head)) # 打印最后一轮训练的结果

## 微调训练过程
if params.n_refinement > 0:
    print('----> ITERATIVE PROCRUSTES REFINEMENT <----') # 显示开始微调训练
    trainer.reload_best() # 重新载入结果最好的生成器参数
    for n_iter in range(params.n_refinement):
        print("Starting refinement iteration %i..."% n_iter)
        trainer.build_nodes_pairs() # 构建节点匹配对应关系
        trainer.procrustes() # 通过procrustes方法更新生成器参数
        logging = OrderedDict({'n_iter': n_iter}) # 创建记录实验结果的有序字典==>print(logging)=OrderedDict([('n_iter', n_iter)])
        evaluator.graph_eval(logging) # 训练结果评估
        trainer.save_best(logging) # 保存结果最好时的生成器参数

## 显示最后的最优效果
trainer.reload_best() # 重新载入结果最好的生成器参数
logging = OrderedDict({}) # 创建记录实验结果的有序字典==>print(logging)=OrderedDict([('n_epoch', n_epoch)])
evaluator.graph_eval(logging) # 训练结果评估
