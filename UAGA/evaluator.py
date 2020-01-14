##########################################################################
# @File: main.py
# @Author: Zhehan Liang
# @Date: 1/11/2020
# @Intro: 评估器函数
##########################################################################

import numpy as np
from utils import get_two_subgraph_scores, get_two_subgraph_scores_REGAL

class Evaluator(object):
    def __init__(self, trainer):
        """
        初始化评估器
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator
        self.params = trainer.params
        self.evaluate_method = self.params.evaluate_method

    def graph_eval(self, logging):
        """
        评估两个子图的节点匹配准确率
        """
        src_emb = self.mapping(self.src_emb).data.cpu().numpy() # 获取源域经过映射后的嵌入
        tgt_emb = self.tgt_emb.data.cpu().numpy()
        if self.evaluate_method=='REGAL':
            accuracy1, accuracy5, accuracy10 = get_two_subgraph_scores_REGAL(src_emb, tgt_emb) # REGAL方法
        else:
            assert self.evaluate_method in ['NN', 'CGSS'], "The evaluate mathod is unkown!" # 校验相似度计算方式是否在已知list中
            accuracy1, accuracy5, accuracy10 = get_two_subgraph_scores(src_emb, tgt_emb, self.evaluate_method, operation='evaluate') # 学长方法
        if accuracy1 is None: # 检查评估函数是否返回成功
            print("Fail to evaluate!")
            return
        print("Graph matching accuracy==>top-1: %.4f, top-5: %.4f, top-10: %.4f" %(accuracy1, accuracy5, accuracy10))
        logging['Graph matching accuracy'] = [accuracy1, accuracy5, accuracy10]
