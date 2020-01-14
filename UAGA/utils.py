##########################################################################
# @File: utils.py
# @Author: Zhehan Liang
# @Date: 1/10/2020
# @Intro: 一些需要用到的工具函数
##########################################################################

import numpy as np
import heapq as hq
import scipy.sparse as sp
import torch
import time
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix, coo_matrix

def initialize_seed(params):
    """
    初始化随机数种子
    """
    time_head = time.time() # 记录开始时间
    if params.seed>0: # 当设置了正的seed值时产生numpy和torch的随机数种子
        np.random.seed(params.seed) # numpy部分的随机数种子
        torch.manual_seed(params.seed) # torch中CPU部分的随机数种子
        if params.cuda: # 当使用cuda时
            torch.cuda.manual_seed(params.seed) # torch中GPU部分的随机数种子
    time_tail = time.time() # 记录完成时间
    print("Seeds have been initialized!\tTime: %.3f"%(time_tail-time_head))

def load_embeddings(params, source):
    """
    载入deepwalk训练好的节点嵌入
    """
    time_head = time.time() # 记录开始时间
    feature = np.ones(shape=(params.emb_num, params.emb_dim)) # 初始化嵌入矩阵
    emb_dir = params.emb_dir + params.dataset
    emb_dir += "_source.emb" if source else "_target.emb" # 设置嵌入文件路径
    with open(emb_dir, 'r', encoding='utf-8') as f: # 读取嵌入文件
        for i, line in enumerate(f.readlines()): # 逐行进行读取
            if i == 0: # deepwalk生成的嵌入文件第一行是节点数和嵌入维度
                result = line.strip().split(' ')
                # 校验嵌入信息
                assert len(result)==2, "Information format of embeddings is error!"
                assert int(result[0])==params.emb_num, "Amount of embeddings is error!"
                assert int(result[1])==params.emb_dim, "Dimension of embeddings is error!"
            else: # 第一行以后都是嵌入信息，格式为节点+对应的嵌入
                node, vector = line.strip().split(' ', 1) # 将单行文本分为节点和嵌入向量
                vector = np.fromstring(vector, sep=' ') # 将文本向量转化为numpy向量
                assert len(vector)==params.emb_dim, "Length of embeddings is error!" # 校验嵌入长度是否正确
                feature[int(node)] = vector # 将得到的向量存入嵌入矩阵的对应行
    embeddings = torch.from_numpy(feature).float() # 把numpy转换到tensor
    embeddings = embeddings.cuda() if params.cuda else embeddings # cuda转化
    time_tail = time.time() # 记录完成时间
    if source:
        print("Source embeddings have been loaded!\tTime: %.3f"%(time_tail-time_head))
    else:
        print("Target embeddings have been loaded!\tTime: %.3f"%(time_tail-time_head))
    assert embeddings.size() == (params.emb_num, params.emb_dim), "Size of embeddings is error!" # 校验embedding的尺寸是否正确
    # 初始化cuda
    if params.cuda:
        embeddings.cuda()

    return embeddings

def get_optimizer(optimizer):
    """
    获取参数对应的优化器和学习率
    """
    optim_params = optimizer.split(',')
    # 8种不同的优化器，可以按照需求自行添加
    if optim_params[0] == 'adadelta':
        optim_fn = torch.optim.Adadelta
    elif optim_params[0] == 'adagrad':
        optim_fn = torch.optim.Adagrad
    elif optim_params[0] == 'adam':
        optim_fn = torch.optim.Adam
    elif optim_params[0] == 'adamax':
        optim_fn = torch.optim.Adamax
    elif optim_params[0] == 'asgd':
        optim_fn = torch.optim.ASGD
    elif optim_params[0] == 'rmsprop':
        optim_fn = torch.optim.RMSprop
    elif optim_params[0] == 'rprop':
        optim_fn = torch.optim.Rprop
    elif optim_params[0] == 'sgd':
        optim_fn = torch.optim.SGD
    else:
        raise Exception('Unknown optimization function: "%s"' % optim_params[0])
    lr = optim_params[1].split('=')
    optim_lr = {}
    optim_lr['lr'] = float(lr[1])

    return optim_fn, optim_lr

"""
用学长代码中的方法计算匹配准确度
"""
def cosine_Matrix(matrix1, matrix2):
    """
    计算矩阵行向量计算余弦相似度
    """
    dot = matrix1.dot(matrix2.transpose())
    matrix1_norm = np.sqrt(np.multiply(matrix1, matrix1).sum(axis=1))
    matrix2_norm = np.sqrt(np.multiply(matrix2, matrix2).sum(axis=1))
    return np.divide(dot, matrix1_norm * matrix2_norm.transpose())

def mean_similarity(similarity_matrix):
    """
    计算平均相似度
    """
    ms1 = (similarity_matrix.sum(axis=1)/similarity_matrix.shape[0])
    ms2 = (similarity_matrix.sum(axis=0)/similarity_matrix.shape[1])
    return ms1, ms2

def get_two_subgraph_scores(embeddings1, embeddings2, method, operation):
    """
    计算两个子图之间节点在不同top值下的匹配准确率/计算最佳匹配点
    """
    assert embeddings1.shape==embeddings2.shape, "embeddings1.shape!=embeddings2.shape" # 校验嵌入的形状是否满足要求，该方案中要求节点的数量相等
    assert operation in ['evaluate', 'match'], "Unkown operation!" # 校验操作是否是评估和匹配中的一个
    top1_count = 0
    top5_count = 0
    top10_count = 0
    pairs = []
    similarity_matrix = cosine_Matrix(embeddings1, embeddings2)# 余弦相似度矩阵计算
    if method=='CGSS': # 如果用CGSS方法计算相似度，需要调整相似度矩阵
        ms1, ms2 = mean_similarity(similarity_matrix) # 先计算嵌入的平均相似度ms1和ms2
        similarity_matrix = 2 * similarity_matrix - np.tile(ms1, (len(embeddings1), 1)).T - np.tile(ms2, (len(embeddings2), 1)) # 再计算CGSS方法下的相似度矩阵
    assert similarity_matrix.shape==(len(embeddings1), len(embeddings2)), "Size of similarity matrix is error!" # 校验相似度矩阵的尺寸是否正确
    time_head = time.time() # 记录开始时间
    for i in range(len(embeddings1)):
        top = hq.nsmallest(10,range(len(embeddings2)), similarity_matrix[i].take)
        if operation=='evaluate': # 评估时，对top进行解析，得到各指标下匹配正确的数量
            for num in range(10):
                if top[9] == i:
                    top1_count += 1
                    top5_count += 1
                    top10_count += 1
                    continue
                elif num < 9 and num >4 and top[num] == i:
                    top5_count += 1
                    top10_count += 1
                elif num < 5 and top[num] == i:
                    top10_count += 1
        else: # 匹配时，记录对应最佳匹配点
            pairs.append(top[9])
        # 每1000个点打印一次结果
        # if i % 1000 == 0:
        #     time_tail = time.time() # 记录完成时间
        #     print("Have matched %d nodes\tTime: %.3f"%(i, time_tail-time_head))
        #     if operation=='evaluate':
        #         print("Accuracy number==>top-1: %d, top-5: %d, top-10: %d"%(top1_count, top5_count, top10_count))
        #     time_head = time.time() # 记录开始时间
    if operation=='evaluate':
        return (top1_count/len(embeddings1), top5_count/len(embeddings1), top10_count/len(embeddings1))
    else:
        assert len(pairs)==len(embeddings1), "Length of pairs is error!" # 校验最佳匹配pair长度是否正确
        return pairs

"""
用REGAL代码中的方法计算匹配准确度
"""
def kd_align(emb1, emb2, normalize=False, distance_metric = "euclidean", num_top = 50):
    kd_tree = KDTree(emb2, metric = distance_metric) # metric变量手册: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric
    row = np.array([])
    col = np.array([])
    data = np.array([])
    dist, ind = kd_tree.query(emb1, k = num_top)
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top)*i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))

    return sparse_align_matrix.tocsr()

def get_embedding_similarities(embed1, embed2, sim_measure = "euclidean", num_top = None):
    n_nodes, dim = embed1.shape
    if num_top is not None: #KD tree with only top similarities computed
        kd_sim = kd_align(embed1, embed2, distance_metric = sim_measure, num_top = num_top)

        return kd_sim

def score_alignment_matrix(alignment_matrix, topk = None, topk_score_weighted = False, true_alignments = None):
    n_nodes_s = alignment_matrix.shape[0]
    n_nodes_t = alignment_matrix.shape[1]
    correct_nodes = []
    if topk is None:
        row_sums = alignment_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 10e-6 #shouldn't affect much since dividing 0 by anything is 0
        alignment_matrix = alignment_matrix / row_sums[:, np.newaxis] #normalize
        alignment_score = score(alignment_matrix, true_alignments = true_alignments)
    else: 
        alignment_score = 0
        if not sp.issparse(alignment_matrix):
            sorted_indices = np.argsort(alignment_matrix)
        for node_index in range(n_nodes_s):
            target_alignment = node_index #default: assume identity mapping, and the node should be aligned to itself
            if true_alignments is not None: #if we have true alignments (which we require), use those for each node
                target_alignment = int(true_alignments[node_index])
            if sp.issparse(alignment_matrix):
                row, possible_alignments, possible_values = sp.find(alignment_matrix[node_index])
                node_sorted_indices = possible_alignments[possible_values.argsort()]
            else:
                node_sorted_indices = sorted_indices[node_index]
            if target_alignment in node_sorted_indices[-topk:]:
                if topk_score_weighted:
                  alignment_score += 1.0 / (n_nodes - np.argwhere(sorted_indices[node_index] == target_alignment)[0])
                else:
                  alignment_score += 1
                correct_nodes.append(node_index)
        alignment_score /= float(n_nodes_s)

    return alignment_score, set(correct_nodes)

def get_two_subgraph_scores_REGAL(embeddings1, embeddings2):
    topk = [1, 5, 10]
    alignment_matrix = get_embedding_similarities(embeddings1, embeddings2, num_top = max(topk)) # 构建相似度矩阵
    # 构建真实对应关系
    true_alignments = {}
    for index in range(embeddings1.shape[0]):
        true_alignments[index] = index
    # 按照top取值进行匹配准确度计算
    topk_scores = []
    for k in topk:
        score, correct_nodes = score_alignment_matrix(alignment_matrix, topk = k, true_alignments = true_alignments)
        topk_scores.append(score)

    return (topk_scores[0], topk_scores[1], topk_scores[2])


