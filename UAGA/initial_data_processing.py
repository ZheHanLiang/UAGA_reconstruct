##########################################################################
# @File: initial_data_processing.py
# @Author: Zhehan Liang
# @Date: 1/9/2020
# @Intro: 对原始数据进行调整，对节点的数量、删除边节点的最小度数、数据格式等
# 进行处理，并分别删去一定数量不重复的边，最后保存为源域和目标域两个新图，保存
# 格式为.txt
# @Data source: https://www.aminer.cn/cosnet
##########################################################################

import numpy as np
import os
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Initial data processing.")
    parser.add_argument('--dataset', '-d', type=int, default=0, help='Number of dataset.')
    parser.add_argument('--total_num', '-n', type=int, default=10000, help='Number of nodes.')
    parser.add_argument('--start_node', '-s', type=int, default=0, help='Start node.')
    parser.add_argument('--degree', '-g', type=int, default=3, help='The nodes which will discard edge with this degree at least.')
    parser.add_argument('--discard_edge_rate', '-r', type=float, default=0.1, help='The discard edge rate of the generated subgraph.')
    return parser.parse_args()

def data_processing(args):
    ## 参数设置
    data_name = ["lastfm", "flickr", "myspace"] # 数据集名称列表(list)
    dataset = data_name[args.dataset]
    input_dir = "../graph data/" # 原始数据文件路径
    output_dir = "./data/new_edge/" # 新数据文件存储路径
    total_num = args.total_num # 删减后总的节点数
    start_node = args.start_node # 开始选择的节点序号，默认从第一个节点，即0开始
    degree = args.degree # 考虑删除边的节点的最小度(degree)数
    nodes_range = range(start_node, start_node+total_num) # 节点搜索范围
    discard_edge_rate = args.discard_edge_rate # 单个图抛弃边的比例

    ## 路径设置
    input_data_dir = input_dir + "{0}/{0}.edges".format(dataset) # 原始数据路径
    output_new_data_dir = output_dir + "{0}_{1}_new.edges".format(dataset, total_num) # 保留目标数量节点文件路径
    output_source_dir = output_dir + "{0}_source_edges.txt".format(dataset) # 源域图路径
    output_target_dir = output_dir + "{0}_target_edges.txt".format(dataset) # 目标域图路径

    ## 读取原始图数据中的边，并保存，保存后可以避免二次运行该部分
    time_head = time.time()
    edges = ""
    f = open(input_data_dir, 'r', encoding='utf-8')
    for line in f.readlines(): # 依次读入每行数据
        nodes = line.strip().split(' ') # nodes = [node0, node1]
        if int(nodes[0]) in nodes_range and int(nodes[1]) in nodes_range: # 仅记录在目标节点范围内的数据
            edges += "{0} {1}\n".format(nodes[0], nodes[1])
        if int(nodes[0])>=total_num: break # 根据原始数据格式的特点，可以提前结束搜索
    f.close()
    if not os.path.exists(output_dir): # 检查输出路径是否存在，若不存在则进行创建
        os.makedirs(output_dir)
    f = open(output_new_data_dir, 'w', encoding='utf-8')
    f.write(edges)
    f.close()
    time_tail = time.time()
    print("%d nodes had been saved!\n1.Time:%.3f"%(total_num, time_tail-time_head))

    ## 计算新图中的各项数据
    time_head = time.time()
    present_node = start_node # 当前节点识别符，从上面设置的start_node开始
    index_edge = 0 # 边的序号
    num_degree = 0 # 记录当前节点的degree
    num_high_degree_edge = 0 # 记录满足degree要求的节点组成的子图中的边的数量，即可以删除的边的数量
    node_set = set() # 记录被记录的节点（方便后续补充中间被遗漏的孤岛节点）
    nodes_beyond_degree = set() # 记录满足degree要求的节点
    can_discard_list = [] # 记录可以删除的边的序号
    present_edge_list = [] # 当前节点所连接的边的序号
    f = open(output_new_data_dir, 'r', encoding='utf-8')
    for line in f.readlines():
        nodes = line.strip().split(' ') # 依次读入每行数据
        if int(nodes[0])==present_node: # 还是同个点时；因为是从start_node开始记录的，所以start_node肯定是第一个被选择的节点
            num_degree += 1 # 当前节点的degree加1
            present_edge_list.append(index_edge) # 把当前边加入当前节点所连接的边的list中
        else: # 记录到下一个点的时候
            if num_degree>=degree: # 当上一个节点的degree达到要求时
                nodes_beyond_degree.add(present_node) # 记录该节点
                num_high_degree_edge += num_degree # 增加子图边的数量
                can_discard_list += present_edge_list # 记录可以删除的边的序号
            num_degree = 1 # 初始化新节点的degree为1
            present_node = int(nodes[0]) # 更新节点识别符
            present_edge_list.clear() # 清空当前节点所连接的边的序号的list
            present_edge_list.append(index_edge) # 把当前边加入当前节点所连接的边的list中
        node_set.add(int(nodes[0])) # 添加被记录的节点
        node_set.add(int(nodes[1])) # 添加被记录的节点
        index_edge += 1 # 更新当前边的序号
    f.close()
    num_edge = index_edge
    time_tail = time.time()
    print("The num of nodes is %d, the max node is %d" %(len(node_set), max(node_set)))
    print("The num of edges is %d" % num_edge)
    print("The num of high-degree edges is %d" % (num_high_degree_edge))
    num_edge_discard = int(num_edge * discard_edge_rate) # 计算一下可以删除的边是否比需要删除的多
    print("The num of edges need to discard is %d" % num_edge_discard)
    if num_high_degree_edge>num_edge_discard:
        print("Edges are enough to discard!")
    else:
        print("Edges aren't enough to discard!\n################################################")
    print("2.Time:%.3f\n################################################"%(time_tail-time_head))

    ## 随机选择出两部分要删除的边
    time_head = time.time()
    edge_set_1 = set()
    edge_set_2 = set()
    while len(edge_set_1)<num_edge_discard:
        rd = np.random.randint(0, len(can_discard_list))
        edge_set_1.add(can_discard_list[rd])
        if len(edge_set_1)%10000==0: # 每10000步显示一下
            print("Have select %d edges to discard in G1"%(len(edge_set_1)))
    while len(edge_set_2)<num_edge_discard:
        rd = np.random.randint(0, len(can_discard_list))
        if can_discard_list[rd] not in edge_set_1:
            edge_set_2.add(can_discard_list[rd])
            if len(edge_set_2)%10000==0: # 有添加的话，每10000步显示一下
                print("Have select %d edges to discard in G2"%(len(edge_set_2)))
    time_tail = time.time()
    print("3.Time:%.3f\n################################################"%(time_tail-time_head))

    ## 源域和目标域子图构建，这里进行共存边计算，验证构建结果是否正确
    time_head = time.time()
    index_edge = 0 # 当前边的序号
    num_com = 0 # 同时保留在两个子图中的边的数目
    flag = 0 # 共存标识符，确定当前边是否同时保留在两个子图中
    edges_source = "" # 源域要保留的边
    edges_target = "" # 目标域要保留的边
    node_set_s = set() # 源域要保留的节点
    node_set_t = set() # 目标域要保留的节点
    f = open(output_new_data_dir, 'r', encoding='utf-8')
    for line in f.readlines():
        if index_edge in edge_set_1 and index_edge in edge_set_2: # 检查是否有边同时出现在两个要删除的集合中
            print("ERROR: Have the edge that both to discard in G1 ande G2!")
        else:
            nodes = line.strip().split(' ')
            if index_edge not in edge_set_1:
                edges_source += "{0} {1} ".format(nodes[0], nodes[1]) + "{'weight': 1.0}\n"
                node_set_s.add(int(nodes[0]))
                node_set_s.add(int(nodes[1]))
                flag += 1
            if index_edge not in edge_set_2:
                edges_target += "{0} {1} ".format(nodes[0], nodes[1]) + "{'weight': 1.0}\n"
                node_set_t.add(int(nodes[0]))
                node_set_t.add(int(nodes[1]))
                flag += 1
        if flag==2: # flag=2时说明当前边同时保留在两个子图中
            num_com += 1
        flag = 0 # 重置flag
        index_edge += 1 # 切换到下一条边
        if index_edge%100000==0:# 每10000条边显示一下
            print("%s edges have finished input" %index_edge)
        # if index_edge==num_edge:
        #     print("All edges have finished input")
        #     break
    rate = 1.0*num_com/num_edge # 计算共存率
    print("The rate of common edges is %f" % rate)
    f.close()
    time_tail = time.time()
    print("4.Time:%.3f\n################################################"%(time_tail-time_head))

    ## 如果有未被记录的孤岛节点，找出来，补上自连接
    time_head = time.time()
    missing_list_s = []
    missing_list_t = []
    if len(node_set_s)<total_num: # 当源域子图有缺失点
        nodes_list = list(node_set_s)
        nodes_list.sort() # 对源域子图的节点序号进行排序，方便查找
        num_missing = total_num - len(node_set) # 计算缺失节点的数目
        num = 0 # 用于下面补充缺失节点用到的一个辅助变量，表示已补充的节点数量
        for index, node in enumerate(nodes_list):
            if index==node:
                continue
            if index==node-num-1:
                missing_list_s.append(node-1)
                num += 1
        if len(set(nodes_list+missing_list_s))==total_num: print("The missing nodes of source have been all found!")
    if len(node_set_t)<total_num: # 当目标域子图有缺失点
        nodes_list = list(node_set_t)
        nodes_list.sort() # 对目标域子图的节点序号进行排序，方便查找
        num_missing = total_num - len(node_set) # 计算缺失节点的数目
        num = 0 # 用于下面补充缺失节点用到的一个辅助变量，表示已补充的节点数量
        for index, node in enumerate(nodes_list):
            if index==node:
                continue
            if index==node-num-1:
                missing_list_t.append(node-1)
                num += 1
        if len(set(nodes_list+missing_list_t))==total_num: print("The missing nodes of target have been all found!")
    if len(missing_list_s)>0:
        for node in missing_list_s:
            edges_source += "{0} {0} ".format(str(node)) + "{'weight': 1.0}\n" # 添加缺失点的自连接
    if len(missing_list_t)>0:
        for node in missing_list_t:
            edges_target += "{0} {0} ".format(str(node)) + "{'weight': 1.0}\n" # 添加缺失点的自连接
    time_tail = time.time()
    print("5.Time:%.3f\n################################################"%(time_tail-time_head))

    ## 保存源域和目标域子图
    f_s = open(output_source_dir, 'w', encoding='utf-8')
    f_t = open(output_target_dir, 'w', encoding='utf-8')
    f_s.write(edges_source)
    f_s.close()
    f_t.write(edges_target)
    f_t.close()
    print("Source graph and target graph have been saved!\n################################################")

if __name__ == "__main__":
    args = parse_args()
    data_processing(args)