from __future__ import division
from __future__ import print_function
import os, sys
import warnings
import argparse
import time
import torch
import hdbscan
import networkx as nx
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import mean_squared_log_error,accuracy_score
from local_embedding import data_for_dimins,scale
from scipy.sparse.csgraph import connected_components
from communities.utilities import modularity
from sklearn.cluster import AgglomerativeClustering

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[64], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--upth_st', type=float, default=0.013, help='Upper Threshold start.')
parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.01, help='Upper Threshold end.')
parser.add_argument('--lowth_ed', type=float, default=0.8, help='Lower Threshold end.')
parser.add_argument('--upd', type=int, default=5, help='Update epoch.')
parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


class FastNewman:
    def __init__(self, G):
        self.G = G
        # G = nx.read_gml('dolphins.gml')
        self.A = nx.to_numpy_array(self.G)  # 邻接矩阵
        self.num_node = len(self.A)  # 点数
        self.num_edge = sum(sum(self.A))  # 边数
        self.c = {}  # 记录所有Q值对应的社团分布

        # def merge_community(self, iter_num, detaQ, e, b):
    #     # 一起合并容易出bug  查询的结果I在遍历过程中 可能在已经前面某次作为J被合并了
    #     # 比如某次是[ 3, 11] [11, 54] 第一轮迭代中11被合并 第二轮54合并到旧的11中  会导致后面被删除 导致节点消失  需要将54合并到现在11所在位置  比较麻烦 不如一个个合并
    #     b_num = sum([len(i) for i in b])
    #     det_max = np.amax(detaQ)
    #
    #     (I, J) = np.where(detaQ == det_max)
    #     print((I, J) )
    #     # 由于先遍历的I I中可能有多个相同值  所以合并时候因应该将J合并到I中
    #     # 如果将I合并到J中 后续删除删不到
    #     for m in range(len(I)):
    #         # 确保J还未被合并
    #         if J.tolist().index(J[m]) == m:
    #             # 将第J合并到I 然后将J清零
    #             e[I[m], :] = e[J[m], :] + e[I[m], :]
    #             e[J[m], :] = 0
    #             e[:, I[m]] = e[:, J[m]] + e[:, I[m]]
    #             e[:, J[m]] = 0
    #             b[I[m]] = b[I[m]] + b[J[m]]
    #
    #     e = np.delete(e, J, axis=0)
    #     e = np.delete(e, J, axis=1)
    #     J = sorted(list(set(J)), reverse=True)
    #     for j in J:
    #         b.remove(b[j])  # 删除第J组社团，（都合并到I组中了）
    #     b_num2 = sum([len(i) for i in b])
    #     if b_num2 != b_num:
    #         print("111")
    #     self.c[iter_num] = b.copy()
    #     return e, b

    def merge_community(self, iter_num, detaQ, e, b):
        # 一个个合并
        (I, J) = np.where(detaQ == np.amax(detaQ))
        # 由于先遍历的I I中可能有多个相同值  所以合并时候因应该将J合并到I中
        # 如果将I合并到J中 后续删除删不到
        e[I[0], :] = e[J[0], :] + e[I[0], :]
        e[J[0], :] = 0
        e[:, I[0]] = e[:, J[0]] + e[:, I[0]]
        e[:, J[0]] = 0
        b[I[0]] = b[I[0]] + b[J[0]]

        e = np.delete(e, J[0], axis=0)
        e = np.delete(e, J[0], axis=1)
        b.remove(b[J[0]])  # 删除第J组社团，（都合并到I组中了）
        self.c[iter_num] = b.copy()
        return e, b

    def Run_FN(self):
        e = self.A / self.num_edge  # 社区i,j连边数量占总的边的比例
        a = np.sum(e, axis=0)  # e的列和，表示与社区i中节点相连的边占总边数的比例
        b = [[i] for i in range(self.num_node)]  # 本轮迭代的社团分布
        Q = []
        iter_num = 0
        while len(e) > 1:
            num_com = len(e)
            detaQ = -np.power(10, 9) * np.ones((self.num_node, self.num_node))  # detaQ可能为负数，初始设为负无穷
            for i in range(num_com - 1):
                for j in range(i + 1, num_com):
                    if e[i, j] != 0:
                        detaQ[i, j] = 2 * (e[i, j] - a[i] * a[j])
            if np.sum(detaQ + np.power(10, 9)) == 0:
                break

            e, b = self.merge_community(iter_num, detaQ, e, b)

            a = np.sum(e, axis=0)
            # 计算Q值
            Qt = 0.0
            for n in range(len(e)):
                Qt += e[n, n] - a[n] * a[n]
            Q.append(Qt)
            iter_num += 1
        max_Q, community = self.get_community(Q)
        return max_Q, community

    def get_community(self, Q):
        max_k = np.argmax(Q)
        community = self.c[max_k]
        return Q[max_k], community


def hdbscan_pred(mlist, adj):

    # f_adj = np.matmul(mlist, np.transpose(mlist))
    # distance = np.multiply(f_adj, adj)

    # adj[adj == 4]=0

    graph = []

    for i in range(len(mlist)):
        gtmp = []
        for j in range(len(mlist)):
            if i < j :
                cosdis = np.dot(mlist[i], mlist[j]) / (np.linalg.norm(mlist[i]) * (np.linalg.norm(mlist[j])))
                gtmp.append(cosdis*adj[i][j])
            elif i > j:
                gtmp.append(graph[j][i])
            else:
                gtmp.append(0)
        graph.append(gtmp)


    distance = np.multiply(graph, 1)
    # distance[np.where(distance < 10)] = 0

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    cluster_labels = clusterer.fit_predict(distance)
    n_clusters = len(set(cluster_labels))
    return n_clusters

def fn_predict(mlist, adj):
    f_adj = np.matmul(mlist, np.transpose(mlist))
    distance = np.multiply(f_adj, adj)
    # distance[np.where(distance < 6)] = 0
    G = nx.from_numpy_matrix(distance)
    fn = FastNewman(G)
    fn.Run_FN()
    max_q, communities = fn.Run_FN()
    n_clusters = len(communities)
    return n_clusters


class GN:
    def __init__(self, G):
        self.G_copy = G.copy()
        self.G = G
        self.partition = [[n for n in G.nodes()]]
        # print(self.partition)
        self.all_Q = [0.0]
        self.max_Q = 0.0
        #         self.zidian={0.0:[100]}
        self.zidian = {0: [0]}

    #     def GG(self):
    #         return self.G_copy

    # Using max_Q to divide communities
    def run(self):
        # Until there is no edge in the graph
        while len(self.G.edges()) != 0:
            # Find the most betweenness edge
            edge = max(nx.edge_betweenness_centrality(self.G).items(), key=lambda item: item[1])[0]
            # print(max(nx.edge_betweenness_centrality(self.G).items(),key=lambda item:item[1]))
            # ((1, 32), 0.1272599949070537)

            # Remove the most betweenness edge
            self.G.remove_edge(edge[0], edge[1])  # 一条边的两个点

            # List the the connected nodes
            components = [list(c) for c in list(nx.connected_components(self.G))]  # 找联通子图

            if len(components) != len(self.partition):  # 所有的边删掉后每个节点自己是一个联通子图
                # compute the Q
                cur_Q = self.cal_Q(components, self.G_copy)
                if cur_Q not in self.all_Q:
                    self.all_Q.append(cur_Q)
                if cur_Q > self.max_Q:
                    self.max_Q = cur_Q
                    self.partition = components
                    for i in range(len(self.partition)):
                        self.zidian[i] = self.partition[i]
        print('-----------the Divided communities and the Max Q-----------')
        print('Max_Q:', self.max_Q)
        print('The number of Communites:', len(self.partition))
        print("Communites:", self.partition)
        return self.partition

    def cal_Q(self, partition, G):
        m = len(G.edges(None, False))
        # print(G.edges(None,False))
        # print("=======6666666")
        a = []
        e = []
        for community in partition:  # 把每一个联通子图拿出来
            t = 0.0
            for node in community:  # 找出联通子图的每一个顶点
                t += len([x for x in G.neighbors(node)])  # G.neighbors(node)找node节点的邻接节点
            a.append(t / (2 * m))
        #             self.zidian[t/(2*m)]=community
        for community in partition:
            t = 0.0
            for i in range(len(community)):
                for j in range(len(community)):
                    if (G.has_edge(community[i], community[j])):
                        t += 1.0
            e.append(t / (2 * m))

        q = 0.0
        for ei, ai in zip(e, a):
            q += (ei - ai ** 2)
        return q

    def add_group(self):
        num = 0
        nodegroup = {}
        for partition in self.partition:
            for node in partition:
                nodegroup[node] = {'group': num}
            num = num + 1
        nx.set_node_attributes(self.G_copy, nodegroup)  # 给每个节点增加分组的属性值
        # print(nodegroup)

    def to_gml(self):
        nx.write_gml(self.G_copy, 'outtoGN.gml')

# GHAC《Unsupervised Author Disambiguation using Heterogeneous Graph Convolutional Network Embedding》使用的方法
def predict_clusters(mlist,adj):
    best_m = -10000000
    distance = []
    # adj[adj == 4]=0

    graph = []

    for i in range(len(mlist)):
        gtmp = []
        for j in range(len(mlist)):
            if i < j :
                cosdis = np.dot(mlist[i], mlist[j]) / (np.linalg.norm(mlist[i]) * (np.linalg.norm(mlist[j])))
                gtmp.append(cosdis*adj[i][j])
            elif i > j:
                gtmp.append(graph[j][i])
            else:
                gtmp.append(0)
        graph.append(gtmp)


    distance = np.multiply(graph, 1)



    graph = np.array(graph)

    n_components1, labels = connected_components(graph)
    graph[graph <= 0.5] = 0

    n_components, labels = connected_components(graph)


    for k in range(n_components, n_components1 - 1, -1):

        model_HAC = AgglomerativeClustering(linkage="average", affinity='precomputed', n_clusters=k)
        model_HAC.fit(distance)
        labels = model_HAC.labels_

        part = []
        for i in range(k):
            part.append(set())
        for j in range(len(labels)):
            part[labels[j]].add(j)

        mod = modularity(graph, part)
        if mod > best_m:
            best_m = mod
            best_labels = labels
    labels = best_labels
    return len(set(labels))


# <Dual-Channel Heterogeneous Graph Network for AuthorName Disambiguation>使用的方法
def dbscan_pred():
    pre = DBSCAN(eps=0.2, min_samples=4, metric="precomputed").fit_predict(sim)
    # print(pre)
    # print(type(pre))
    # for i in range(len(pre)):
    #   print('pre',[i],pre[i])
    # print('len:',len(pre))

    for i in range(len(pre)):
        if pre[i] == -1:
            outlier.add(i)
    # print('outlier:',outlier)
    ## assign each outlier a label
    paper_pair = generate_pair(pubs, outlier)
    # print('paper_pair:',paper_pair)
    paper_pair1 = paper_pair.copy()
    K = len(set(pre))

    listap = []
    for i in range(len(pubs)):
        listap.append(float(-1))
    na1 = np.array(listap)

    for i in range(len(pre)):
        if i not in outlier:
            continue
        j = np.argmax(paper_pair[i])
        # print('j1:',j)
        while j in outlier:
            paper_pair[i][j] = -1
            if any(na1 == paper_pair[i]) == True:
                break
            else:
                j = np.argmax(paper_pair[i])
            # print('j2:',j)
        if paper_pair[i][j] >= 1.5:
            pre[i] = pre[j]
        else:
            pre[i] = K
            K = K + 1

    ## find nodes in outlier is the same label or not
    for ii, i in enumerate(outlier):
        for jj, j in enumerate(outlier):
            if jj <= ii:
                continue
            else:
                if paper_pair1[i][j] >= 1.5:
                    pre[j] = pre[i]

def main(args):
    name_to_pubs_train = load_json("datasets/OAG-WhoisWh0-na-v3/train","train_author.json")
    name_to_pubs_test = load_json("datasets/OAG-WhoisWh0-na-v3/valid","sna_valid_ground_truth.json")
    wf = open('local_embedding_data/cluster_predicted_ghac.csv', 'w', encoding='utf-8')
    wf.write('name,num_nodes,true_clusters,pred_clusters\n')

    ktrue = []
    kpre = []
    for index,name in enumerate(name_to_pubs_test):
        curr_name_pubs = name_to_pubs_test[name]
        if len(curr_name_pubs) <2:
            continue
        start_time = time.time()
        print(index,name)
        # if name == 'jianzhou_wang':
        if index>=0:
            inx_author, adj_author, pos_num_author, labels_author = data_for_dimins(name, args, 1, 'author')
            inx_content, adj_content, pos_num_content, labels_content = data_for_dimins(name, args, 1,'content')
            inx_orgs, adj_orgs, pos_num_orgs, labels_orgs = data_for_dimins(name, args, 1, 'orgs')
            inx_venue, adj_venue, pos_num_venue, labels_venue = data_for_dimins(name, args, 1, 'venue')

            # inx_all = inx_venue * 0.15 + inx_orgs * 0.15 + inx_content * 0.4 + inx_author * 0.3
            inx_all = torch.cat((inx_content, inx_orgs, inx_venue, inx_author), 1)
            # inx_all = inx_author
            total_adj = adj_author + adj_content + adj_orgs + adj_venue
            # total_adj = adj_author


            line = len(inx_content)
            total_adj = total_adj.reshape(line, line)
            total_adj = total_adj.cpu().data.numpy()
            # total_adj[np.where(total_adj > 1)] = 1
            total_adj = total_adj / 2 + np.transpose(total_adj / 2)
            # total_adj[np.where(total_adj == 4)] = 0
            inx_all_feature = scale(inx_all).cpu().data.numpy()
            # # communities, frames = louvain_method(distance)
            # G = nx.from_numpy_matrix(distance)
            # fn = FastNewman(G)
            # fn.Run_FN()
            # max_q,communities = fn.Run_FN()
            # n_clusters = len(communities)
            # n_clusters = 0

            # n_clusters = predict_clusters(inx_all_feature,total_adj)
            # adj_orgs= adj_orgs.reshape(line, line)
            # adj_orgs = adj_orgs.cpu().data.numpy()
            # adj_orgs = adj_orgs/2+np.transpose(adj_orgs/2)
            # inx_orgs = scale(inx_orgs).cpu().data.numpy()

            # communities = DBSCAN(eps=0.6, min_samples=4).fit_predict(inx_all_feature)
            # 统计每一类的数量
            #     n_clusters = pd.value_counts(iris_db, sort=True)
            #
            n_clusters = hdbscan_pred(inx_all_feature, total_adj)
            # n_clusters = fn_predict(inx_all_feature,total_adj)
            # n_clusters = predict_clusters(inx_all_feature,total_adj)

            true_clusters = max(labels_content) + 1
            ktrue.append(true_clusters)
            kpre.append(n_clusters)

            # n_clusters = fn_predict(inx_all_feature,total_adj)
            #
            #
            #
            #

            print("预测的簇个数：", n_clusters)
            print("真实的簇个数:", true_clusters)
        #
            wf.write('{0},{1},{2},{3}\n'.format(
                name, line, true_clusters,n_clusters))
            wf.flush()

    msle = mean_squared_log_error(ktrue, kpre)
    acc = accuracy_score(ktrue, kpre)

    print("MSLE", msle)

    print("Accuracy", acc)

    wf.write('msle:,{0:.5f},acc,{1:.5f}\n'.format(
        msle, acc))
    wf.close()

if __name__ == '__main__':
    main(args)
