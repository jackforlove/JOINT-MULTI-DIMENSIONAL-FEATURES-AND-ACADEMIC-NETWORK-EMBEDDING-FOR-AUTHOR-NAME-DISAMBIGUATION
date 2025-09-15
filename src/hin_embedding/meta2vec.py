import os
import re
from xml.dom.minidom import parse
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import collections
from src.utils.clustering_metric import clustering_metrics
import pandas as pd
import scipy.sparse as sp
import torch
import random
import dgl
import scipy
import scipy.sparse as spp
import pickle
from tqdm import tqdm
import torch.nn as nn
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
from model import RGCN,SkipGramModel
# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
RWALK_DIR = "../../datas/OAG-WhoisWh0-na-v1/hin_embedding"
PRE_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
ADJ_DIR = "../../datas/OAG-WhoisWh0-na-v1/adj"
GLOBAL_EMB_DIR = "../../datas/OAG-WhoisWh0-na-v1/global_embedding"

valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/test_author_name_ids.pkl')
valid_author_name_paper_ids = pd.read_pickle(PRE_DATA_DIR+'/test_author_name_paper_ids.pkl')
valid_global_embedding = pd.read_pickle(GLOBAL_EMB_DIR+'/test_global_embedding.pkl')

SEED = 42
window = 2  # 这里是取metapath时的窗口大小

num_walks = 5  # 每个结点run 多少遍
walk_len = 20  # 每个path的长度
# metapath_type = ['coorgs', 'coauthor', 'covenue'，'coauthor'，'coorgs']
metapath_type = ['coauthor', 'coorgs', 'coauthor', 'covenue']

device = torch.device('cuda')

print('Using GPU')
# torch.cuda.manual_seed(42)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = list(map(lambda x: classes_dict[x], labels))
    return labels
def get_cograph(name,relation):
    adj_dir = ADJ_DIR + "/"+relation+"/" + name + "/pubs_network.txt"


    local_emb = valid_global_embedding.loc[valid_global_embedding["author_name"] == name]
    paper_id = local_emb["paper_id"]
    features = local_emb[relation]
    labels = local_emb["author_id"]


    paper_id = paper_id.values.tolist()
    paper_id = np.array(paper_id)
    paper_id_map = {j: i for i, j in enumerate(paper_id)}
    features = features.values
    features = features.tolist()
    features = np.array(features)
    labels = labels.values
    labels = encode_labels(labels)
    features = torch.FloatTensor(features)

    edges_unordered = np.genfromtxt(adj_dir, dtype=np.dtype(str),usecols =(0,1))
    edges_weight = np.genfromtxt(adj_dir, dtype=np.dtype(float),usecols =(2))

    edges = np.array(list(map(paper_id_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((edges_weight, (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    co_graph = dgl.from_scipy(adj)

    # coauthor_weight = torch.ones(co_graph.num_edges(), 1)
    #
    # for i, (src, dst) in enumerate(zip(co_graph.edges()[0], co_graph.edges()[1])):
    #     weight = edges_weight[i]
    #     if weight != 0:
    #         coauthor_weight[i] = weight
    co_graph.edata['weights'] = torch.FloatTensor(edges_weight)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
    weight_graph = adj_1st


    features = features
    weight_graph = weight_graph
    return co_graph,weight_graph,labels,features
def scale(z):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / (zmax - zmin)
    z_scaled = z_std
    return z_scaled
def meta2vec(name):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    metapaths = []  # 所有的metapath
    coauthor_graph,weights_coauthor,labels,coauthor_feature = get_cograph(name,"authors")
    coorgs_graph,weights_coorgs,labels,coorgs_feature = get_cograph(name,"author_org")
    covenue_graph,weights_covenue,labels,covenue_feature = get_cograph(name,"venue")

    # node_feature = covenue_feature+coorgs_feature+coauthor_feature
    # hete_graph = dgl.heterograph({
    #     ('paper', 'coauthor', 'paper'): coauthor_graph.edges(),
    #     ('paper', 'coorgs', 'paper'): coorgs_graph.edges(),
    #     ('paper', 'covenue', 'paper'): covenue_graph.edges()
    # })

    edge_per_graph = {}  # 对应每个图，建立个字典，每个字典的key为结点编号，value为key在该图中可以到达的结点编号
    edge_per_graph['coauthor'] = create_node2node_dict(coauthor_graph)
    edge_per_graph['coorgs'] = create_node2node_dict(coorgs_graph)
    edge_per_graph['covenue'] = create_node2node_dict(covenue_graph)
    weights_all_graph = {'coauthor': weights_coauthor, 'coorgs': weights_coorgs, 'covenue': weights_covenue}

    for walk in tqdm(range(num_walks)):
        for cur_node in list(range(len(labels))):
            stop = 0
            path = []
            path.append(cur_node)
            while len(path) < walk_len and stop == 0:
                for rel in metapath_type:
                    if len(path) == walk_len or Is_isolate(cur_node,edge_per_graph):
                        stop = 1
                        break
                    if edge_per_graph[rel].get(cur_node, -1) == -1:
                        continue

                    cand_nodes = edge_per_graph[rel][cur_node]
                    weights_per_candnodes = weights_all_graph[rel][cur_node][cand_nodes]
                    weighted_ratio = weights_per_candnodes * 1.0 / np.sum(weights_per_candnodes)
                    cur_node=np.random.choice(cand_nodes,size=1,p=weighted_ratio)[0]
                    path.append(cur_node)
            metapaths.append(path)

    pos_us, pos_vs, neg_vs = [], [], []
    nodes = list(range(len(labels)))
    ratio = get_negative_ratio(metapaths)
    for path in metapaths:
        pos_u, pos_v = positive_sampler(path)
        for u, v in zip(pos_u, pos_v):
            negative_nodes = negative_sampler(path, ratio, nodes)
            neg_vs.append(negative_nodes)
        pos_us.extend(pos_u)
        pos_vs.extend(pos_v)
    pos_us = torch.LongTensor(pos_us).cuda()
    pos_vs = torch.LongTensor(pos_vs).cuda()
    neg_vs = torch.LongTensor(neg_vs).cuda()

    # # 语义特征与关系特征联合优化
    # model = RGCN(node_feature.shape[-1], 100, 64)
    # skip_model = SkipGramModel()
    # optimizer = torch.optim.Adam(nn.ModuleList([skip_model, model]).parameters(), lr=0.001, weight_decay=0.0001)
    # losses = []
    # for epoch in range(200):
    #     model.train()
    #     optimizer.zero_grad()
    #     logits = model(coauthor_graph, coorgs_graph, covenue_graph, node_feature)
    #     loss, score, neg_score = skip_model(torch.tensor(pos_us), torch.tensor(pos_vs), torch.tensor(neg_vs), logits)
    #     loss.backward()
    #     optimizer.step()
    #     losses.append(loss.item())
    #     if epoch % 20 == 0:
    #         print('epoch {0}  loss {1}'.format(epoch, loss))
    # hin_embedding = model(coauthor_graph, coorgs_graph, covenue_graph, node_feature).detach().numpy()

    # 单纯的metapath2vec
    skip_model = SkipGramModel(len(labels), 64)
    skip_model.cuda()
    losses = []
    optimizer = torch.optim.Adam(skip_model.parameters(), lr=0.001)
    for epoch in range(500):
        optimizer.zero_grad()
        loss = skip_model(torch.tensor(pos_us).cuda(), torch.tensor(pos_vs).cuda(), torch.tensor(neg_vs).cuda())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            print('epoch {0}  loss {1}'.format(epoch, loss))


    embedding = skip_model.u_embeddings.weight.cpu().data
    embedding = torch.nn.functional.softmax(embedding)

    true_clusters = max(labels) + 1
    Cluster = SpectralClustering(n_clusters=true_clusters, affinity='precomputed', random_state=0)
    f1, nmi, adj = clustering(Cluster, embedding, labels)
    print('使用hin表示的聚类结果：f1: {}, nmi: {}, adj: {}'.format(f1, nmi, adj))
    return len(embedding),true_clusters,f1, nmi, adj,embedding
def clustering(Cluster, feature, true_labels):
    f_adj = np.matmul(feature, np.transpose(feature))
    # distance =np.multiply(f_adj,total_adj)
    predict_labels = Cluster.fit_predict(f_adj)

    cm = clustering_metrics(true_labels, predict_labels)
    nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)
    prec, rec, f1 = pairwise_precision_recall_f1(predict_labels, true_labels)

    return f1, nmi, adj
def positive_sampler(path):
    pos_u,pos_v=[],[]
    for i in range(len(path)):
        if len(path)==1:
            continue
        u=path[i]
        v=np.concatenate([path[max(i-window,0):i],path[i+1:i+window+1]],axis=0)
        pos_u.extend([u]*len(v))
        pos_v.extend(v)
    return pos_u,pos_v
def get_negative_ratio(metapath):
    node_frequency=dict()
    sentence_count,node_count=0,0
    for path in metapath:
        for node in path:
            node_frequency[node]=node_frequency.get(node,0)+1
            node_count+=1
    pow_frequency=np.array(list(map(lambda x:x[-1],sorted(node_frequency.items(),key=lambda asd:asd[0]))))**0.75
    node_pow=np.sum(pow_frequency)
    ratio=pow_frequency/node_pow
    return ratio
def negative_sampler(path,ratio,nodes):
    negtives_size=5
    negatives=[]
    while len(negatives)<5:
        temp=np.random.choice(nodes, size=negtives_size-len(negatives), replace=False, p=ratio)
        negatives.extend([node for node in temp if node not in path])
    return negatives
def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def create_node2node_dict(graph):
    src_dst = {}
    for src, dst in zip(graph.edges()[0], graph.edges()[1]):
        src, dst = src.item(), dst.item()
        if src not in src_dst.keys():
            src_dst[src] = []
        src_dst[src].append(dst)
    return src_dst
def normalize_vectors(vectors):
    scaler = StandardScaler()
    vectors_norm = scaler.fit_transform(vectors)
    return vectors_norm

def Is_isolate(node,edge_per_graph):
    for rel in metapath_type:
        if node in edge_per_graph[rel].keys():
            return 0
    return 1


def convert(name):
    rrrr = '[!“”"#$%&\'()*+,-./:)‘′(;<=>?@[\\]^-_` •！@#￥&*（）——+~}】【|、|？《》，。：“；‘{|}~—～’*《》<>「」{}【】()/\\\[\] ]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the']
    name = name.lower()
    name = re.sub(rrrr, ' ', name.strip().lower())
    name = re.sub(r'\s{2,}', '', name).strip()
    name = re.sub(r'[0-9]', '', name).strip()
    new_name = ""
    for word in name.split():
        if word not in stopword:
            new_name +=word
    return new_name

def main():
    wf = open(RWALK_DIR+'/out/results_test_加权0.7_作者-组织-作者-场所.csv', 'w', encoding='utf-8')
    wf.write('name,num_nodes,true_clusters,pf1,nmi,ari\n')
    metrics = np.zeros(3)
    cnt = 0

    hin_embedding = pd.DataFrame(columns=["name","hin_embedding"])
    for index,name in enumerate(valid_author_name_ids["author_name"]):
        name = convert(name)
        print(index,name)
        if index>=0:
            num_nodes, true_clusters, best_f1, best_nmi, best_adj,embedding =meta2vec(name)
            # embedding = embedding.cpu().data.numpy()
            hin_embedding.loc[index] = [name,embedding]
            wf.write('{0},{1},{2},{3},{4:.5f},{5:.5f}\n'.format(
                name, num_nodes, true_clusters, best_f1, best_nmi, best_adj))
            wf.flush()
            cur_metric = [best_f1, best_nmi, best_adj]
            for i, m in enumerate(cur_metric):
                metrics[i] += m
            cnt += 1
            average_f1 = metrics[0] / cnt
            average_nmi = metrics[1] / cnt
            average_adj = metrics[2] / cnt
            print('average until now', [average_f1, average_nmi, average_adj])
    hin_embedding.to_pickle(RWALK_DIR + "/hin_embedding.pkl")
    average_f1 = metrics[0] / cnt
    average_nmi = metrics[1] / cnt
    average_adj = metrics[2] / cnt
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        average_f1, average_nmi, average_adj))
    wf.close()



if __name__ == '__main__':
    main()
    # meta2vec("haifengqian")
    # meta2vec("huicai")
    # meta2vec("jianguowu")
    # meta2vec("weishi")
    # meta2vec("bozou")
    # meta2vec("pingxie")

