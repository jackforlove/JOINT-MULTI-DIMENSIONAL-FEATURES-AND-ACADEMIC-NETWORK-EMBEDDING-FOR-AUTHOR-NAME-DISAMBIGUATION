import argparse
import math
import multiprocessing
import random
import time
import re
import community
import hdbscan
import networkx as nx
from torch import optim
import pandas as pd
import warnings
from model import LocalModel
import scipy.sparse as sp
import torch
from sklearn.cluster import SpectralClustering, KMeans,AgglomerativeClustering,DBSCAN
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pandas as pd
import numpy as np
from src.utils.clustering_metric import clustering_metrics

import os
import sys
import pypinyin
from collections import defaultdict
from tqdm import tqdm_notebook,tqdm
from src.utils.utils import *
import pickle
from gensim.models import word2vec,Word2Vec
from sklearn.preprocessing import normalize, MinMaxScaler

# 配置参数
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
# For replicating the experiments
SEED = 42
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from scipy.sparse.csgraph import connected_components
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
parser.add_argument('--dims', type=int, default=[64], help='Number of units in hidden layer 1.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--upth_st', type=float, default=0.01, help='Upper Threshold start.')
parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
parser.add_argument('--lowth_ed', type=float, default=0.9, help='Lower Threshold end.')
parser.add_argument('--upd', type=int, default=5, help='Update epoch.')
parser.add_argument('--bs', type=int, default=10000, help='Batchsize.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda')


if args.cuda is True:
    print('Using GPU')
    torch.cuda.manual_seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
GLOBAL_EMB_DIR = "../../datas/OAG-WhoisWh0-na-v1/global_embedding"
LOCAL_EMB_DIR = "../../datas/OAG-WhoisWh0-na-v1/local_embedding"
ADJ_DIR = "../../datas/OAG-WhoisWh0-na-v1/adj"
PRE_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
RWALK_DIR = "../../datas/OAG-WhoisWh0-na-v1/hin_embedding"
IN_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"

# 导入数据
valid_data = pd.read_pickle(IN_DATA_DIR+'/test_data.pkl')
valid_data = valid_data.reset_index()
valid_global_embedding = pd.read_pickle(GLOBAL_EMB_DIR+'/test_global_embedding.pkl')
train_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/train_author_name_ids.pkl')
valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/test_author_name_ids.pkl')
hin_embedding_data = pd.read_pickle(RWALK_DIR + "/hin_embedding.pkl")

def encode_labels(labels):
    classes = set(labels)
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels = list(map(lambda x: classes_dict[x], labels))
    return labels

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

def normalize_vectors(vectors):
    scaler = StandardScaler()
    vectors_norm = scaler.fit_transform(vectors)
    return vectors_norm
def preprocess_graph(adj, layer, norm='sym', renorm=True):
    adj = sp.coo_matrix(adj)
    ident = sp.eye(adj.shape[0])
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))

    if norm == 'sym':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        laplacian = ident - adj_normalized
    elif norm == 'left':
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    reg = [1 / 2] * (layer)

    adjs = []
    for i in range(len(reg)):
        adjs.append(ident - (reg[i] * laplacian))
    return adjs

def data_for_dimins(name,relation,embedding,args):


    local_emb = embedding.loc[embedding["author_name"] == name]
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

    adj_dir = ADJ_DIR+"/"+relation+"/"+name+"/pubs_network.txt"
    edges_unordered = np.genfromtxt(adj_dir, dtype=np.dtype(str),usecols =(0,1))
    edges_weight = np.genfromtxt(adj_dir, dtype=np.dtype(float),usecols =(2))

    edges = np.array(list(map(paper_id_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((edges_weight, (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    adj_norm_s = preprocess_graph(adj, 3, norm='sym', renorm=False)
    sm_fea_s = sp.csr_matrix(features).toarray()
    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    print('Laplacian Smoothing-----'+relation)
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)

    # sm_fea_s = normalize_vectors(sm_fea_s)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
    adj_label = torch.FloatTensor(adj_1st)


    sm_fea_s = torch.FloatTensor(sm_fea_s)
    if args.cuda:
        inx = sm_fea_s.cuda()
        support = adj_label.cuda()

    pos_num = len(adj.indices)

    return inx,labels,support,pos_num
def get_common_data(features):
    n_nodes, feat_dim = features.shape
    # dims = [feat_dim//4] + args.dims
    dims = [feat_dim] + args.dims


    features = features.to_dense()
    pos_inds, neg_inds,pos_num,neg_num = update_similarity(features.cpu().data.numpy(), args.upth_st, args.lowth_st)

    bs = min(args.bs, len(pos_inds))

    return pos_inds,neg_inds,bs,n_nodes,pos_num,neg_num,dims
def update_similarity(z, upper_threshold, lower_treshold):
    f_adj = np.matmul(z, np.transpose(z))
    # f_adj_nor = normalize_max_min(f_adj)
    # total_adj_mor = normalize_max_min(total_adj)
    # distance = f_adj_nor+total_adj_mor


    cosine = f_adj
    cosine = cosine.reshape([-1, ])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds),pos_num,neg_num
def update_similarity_2(z,y, upper_threshold, lower_treshold):
    f_adj = np.matmul(z, np.transpose(z))
    f_adj_2 = np.matmul(y, np.transpose(y))
    f_adj = normalize_max_min(f_adj)
    f_adj_2 = normalize_max_min(f_adj_2)
    distance = f_adj+f_adj_2


    cosine = distance
    cosine = cosine.reshape([-1, ])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1 - lower_treshold) * len(cosine))

    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]

    return np.array(pos_inds), np.array(neg_inds),pos_num,neg_num
def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth


def normalize_max_min(f_adj):
    min = np.min(f_adj)
    max = np.max(f_adj)
    ranges = max-min
    f_adj_nor = (f_adj-min)/ranges
    return f_adj_nor
def clustering(Cluster, feature, true_labels,total_adj):
    f_adj = np.matmul(feature, np.transpose(feature))
    # mean = np.mean(f_adj)*2
    # for i in range(len(feature)):
    #     for j in range(len(feature)):
    #         # if i is not j:
    #             if total_adj[i][j] == 0 and f_adj[i][j]<mean:
    #                 f_adj[i][j]=0

    # f_adj_nor = normalize_max_min(f_adj)

    # total_adj_mor = normalize_max_min(total_adj)
    # distance = f_adj_nor+total_adj_mor
    # mean = np.mean(distance)
    # distance[distance<mean] = 0
    total_adj[total_adj>0] = 1
    # f_adj_nor[f_adj_nor>mean]=1
    # f_adj_nor[f_adj_nor<mean]=0
    # adj = total_adj+f_adj_nor

    # distance = f_adj+total_adj
    distance =np.multiply(f_adj,total_adj)
    # distance[distance<mean]=0

    predict_labels = Cluster.fit_predict(distance)
    cm = clustering_metrics(true_labels, predict_labels)
    nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)
    prec, rec, f1 = pairwise_precision_recall_f1(predict_labels, true_labels)

    return f1, nmi, adj
def loss_function(adj_preds, adj_labels, n_nodes):
    cost = 0.
    cost += F.binary_cross_entropy_with_logits(adj_preds, adj_labels)

    return cost
def scale(z):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / (zmax - zmin)
    z_scaled = z_std
    return z_scaled
def hdbscan_pred(mlist, adj,true_labels):

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

    distance = np.multiply(graph, -1)
    # distance[np.where(distance < 10)] = 0

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
    cluster_labels = clusterer.fit_predict(distance)
    # n_clusters = len(set(cluster_labels))

    cm = clustering_metrics(true_labels, cluster_labels)
    nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)
    prec, rec, f1 = pairwise_precision_recall_f1(cluster_labels, true_labels)
    return f1,nmi,adj

def GHAC(mlist, adj,true_labels):

    global best_labels
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

    distance = np.multiply(graph, -1)

    best_m = -10000000
    graph = np.array(graph)
    n_components1, labels = connected_components(graph)

    graph[graph <= 0.3] = 0
    G = nx.from_numpy_matrix(graph)

    n_components, labels = connected_components(graph)

    for k in range(n_components, n_components1 - 1, -1):

        model_HAC = AgglomerativeClustering(linkage="average", affinity='precomputed', n_clusters=k)
        # model_HAC = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=0)
        model_HAC.fit(distance)
        labels = model_HAC.labels_

        part = {}
        for j in range(len(labels)):
            part[j] = labels[j]

        mod = community.modularity(part, G)
        if mod > best_m:
            best_m = mod
            best_labels = labels
    labels = best_labels
    cm = clustering_metrics(true_labels, labels)
    nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)
    prec, rec, f1 = pairwise_precision_recall_f1(labels, true_labels)

    return f1, nmi, adj
def hdbscan_pred_2(embedding,name,mlist,adj,true_labels):
    outlier = set()
    local_emb = embedding.loc[embedding["author_name"] == name]
    paper_id = local_emb["paper_id"]
    pubs = paper_id.values
    # f_adj = np.matmul(feature, np.transpose(feature))
    # total_adj[np.where(total_adj==3)]=0
    # distance =np.multiply(f_adj,total_adj)

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

    distance = np.multiply(graph, -1)


    # pre = DBSCAN(eps=0.2, min_samples=4).fit_predict(distance)
    pre = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(distance)
    # Cluster =AgglomerativeClustering(n_clusters=true_cluster)
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
    paper_pair = generate_pair(pubs, outlier,valid_data,name)
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
    pre = np.array(pre)
    cm = clustering_metrics(true_labels, pre)
    nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)
    prec, rec, f1 = pairwise_precision_recall_f1(pre, true_labels)

    return f1,nmi,adj
def get_hin_embedding(name):
    hin_embedding_name = hin_embedding_data.loc[hin_embedding_data["name"] == name]["hin_embedding"]
    hin_embedding_name = hin_embedding_name.values.tolist()[0]

    # hin_embedding_name = normalize_vectors(hin_embedding_name)
    hin_embedding_name = scale(hin_embedding_name).cuda()
    # hin_embedding_name = np.array(hin_embedding_name)
    return hin_embedding_name
def total_local__embedding(name,embedding,args):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    hin_embedding = get_hin_embedding(name)
    inx_authors, labels_authors,adj_authors,pos_num_authors = data_for_dimins(name, 'authors',embedding,args)
    inx_title, labels_title,adj_title,pos_num_title = data_for_dimins(name,  'title',embedding,args)
    # inx_keywords, labels_keywords ,adj_keywords,pos_num_keywords= data_for_dimins(name, 'keywords',embedding,args)
    inx_venue, labels_venue,adj_venue,pos_num_venue = data_for_dimins(name, 'venue',embedding,args)
    inx_author_org, labels_author_org,adj_author_org,pos_num_author_org = data_for_dimins(name, 'author_org',embedding,args)
    try:
        inx_abstract, labels_abstract ,adj_abstrat,pos_num_abstract= data_for_dimins(name, 'abstract',embedding,args)
    except:
        inx_abstract, labels_abstract ,adj_abstrat,pos_num_abstract= inx_title, labels_title,adj_title,pos_num_title

    inx_all = inx_title +inx_authors+inx_venue+inx_author_org +inx_abstract
    # inx_all = inx_authors +  inx_title+inx_author_org+inx_venue+inx_abstract
    # inx_all = inx_venue

    total_adj = adj_author_org +adj_authors +adj_venue
    # total_adj = adj_venue
    total_adj = total_adj.cpu().data.numpy()

    pos_inds, neg_inds, bs, n_nodes, pos_num, neg_num, dims = get_common_data(inx_all)

    pos_inds_cuda = torch.LongTensor(pos_inds).cuda()
    up_eta = (args.upth_ed - args.upth_st) / (args.epochs / args.upd)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs / args.upd)
    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)

    model = LocalModel(dims)

    print(dims)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.cuda:
        model.cuda()

    line = len(inx_all)
    # total_adj = total_adj.reshape(line, line)

    # total_adj = total_adj / 2 + np.transpose(total_adj / 2)

    true_clusters = max(labels_title) + 1
    # inx_all = torch.nn.functional.softmax(inx_all)
    # inx_all_feature = torch.nn.functional.normalize(inx_all)
    inx_all_feature = scale(inx_all).cpu().data.numpy()

    # inx_authors = normalize_vectors(inx_authors)
    # inx_title = normalize_vectors(inx_title)
    # inx_venue = normalize_vectors(inx_venue)
    # inx_author_org = normalize_vectors(inx_author_org)
    # inx_abstract = normalize_vectors(inx_abstract)

    # 簇个数预测
    # f1, nmi, adj = hdbscan_pred(inx_all_feature,total_adj,labels_venue)
    # pre_clusters = int(float(pre_clusters.split('\t')[2].strip()))


    # f1, nmi, adj = hdbscan_pred_2(embedding,name,inx_all_feature,total_adj,labels_venue)
    # f1, nmi, adj = GHAC(inx_all_feature,total_adj,labels_venue)

    #
    Cluster = SpectralClustering(n_clusters=true_clusters, affinity='precomputed', random_state=0)
    try:
        f1, nmi, adj = clustering(Cluster, inx_all_feature, labels_title, total_adj)
    except:
        f1,nmi,adj = 0, 0, 0
    #
    #
    print('局部嵌入前：f1: {}, nmi: {}, adj: {}'.format(f1, nmi, adj))

    best_f1 = f1
    best_nmi = nmi
    best_adj = adj
    best_all = (f1 + nmi + adj) / 3


    print('Start Training...')
    for epoch in tqdm(range(args.epochs)):

        st, ed = 0, bs
        batch_num = 0
        model.train()
        length = len(pos_inds)

        while (ed <= length):
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=ed - st)).cuda()
            sampled_inds = torch.cat((pos_inds_cuda[st:ed], sampled_neg), 0)
            t = time.time()
            optimizer.zero_grad()
            #
            xind = sampled_inds // n_nodes
            yind = sampled_inds % n_nodes
            #
            x_authors = torch.index_select(inx_authors, 0, xind)
            x_abstract = torch.index_select(inx_abstract, 0, xind)
            x_title = torch.index_select(inx_title, 0, xind)
            x_author_org = torch.index_select(inx_author_org, 0, xind)
            x_venue = torch.index_select(inx_venue, 0, xind)
            x_hin = torch.index_select(hin_embedding, 0, xind)

            y_author = torch.index_select(inx_authors, 0, yind)
            y_abstract = torch.index_select(inx_abstract, 0, yind)
            y_title = torch.index_select(inx_title, 0, yind)
            y_author_org = torch.index_select(inx_author_org, 0, yind)
            y_venue = torch.index_select(inx_venue, 0, yind)
            y_hin = torch.index_select(hin_embedding, 0, yind)

            zx = model(x_authors, x_abstract, x_title,x_author_org,x_venue,x_hin)
            zy = model(y_author, y_abstract, y_title,y_author_org,y_venue,y_hin)

            batch_label = torch.cat((torch.ones(ed - st), torch.zeros(ed - st))).cuda()
            batch_pred = model.dcs(zx, zy)

            loss = loss_function(adj_preds=batch_pred, adj_labels=batch_label, n_nodes=ed - st)
            loss.backward()

            cur_loss = loss.item()
            optimizer.step()

            st = ed
            batch_num += 1
            if ed < length and ed + bs >= length:
                ed += length - ed
            else:
                ed += bs

        if (epoch + 1) % args.upd == 0:
            model.eval()
            mu = model(inx_authors, inx_abstract, inx_title,inx_author_org,inx_venue,hin_embedding)
            hidden_emb = mu.cpu().data.numpy()
            # inx_emb = inx_all.cpu().data.numpy()
            upth, lowth = update_threshold(upth, lowth, up_eta, low_eta)
            # # if epoch > epoch/2:
            # #
            pos_inds, neg_inds, pos_num, neg_num = update_similarity(hidden_emb, upth, lowth)
            bs = min(args.bs, len(pos_inds))
            pos_inds_cuda = torch.LongTensor(pos_inds).cuda()

            # hidden_emb = F.normalize(scale(mu)).cpu().data.numpy()
            f1, nmi, adj = clustering(Cluster, hidden_emb, labels_title,total_adj)
            # f1, nmi, adj = hdbscan_pred_2(embedding, name, hidden_emb, total_adj, labels_venue)
            # f1, nmi, adj = GHAC(hidden_emb, total_adj, labels_venue)
            # f1, nmi, adj = hdbscan_pred(hidden_emb, total_adj, labels_venue)
            tqdm.write("Epoch: {}, train_loss_gae={:.5f}, time={:.5f}".format(
                epoch + 1, cur_loss, time.time() - t))
            print('f1: {}, nmi: {}, adj: {}'.format(f1, nmi, adj))
            if (f1 + nmi + adj) / 3 > best_all:
                best_f1 = f1
                best_nmi = nmi
                best_adj = adj
                best_all = (f1 + nmi + adj) / 3
    tqdm.write("Optimization Finished!")
    tqdm.write('best_f1: {}, best_nmi: {}, best_adj: {}'.format(best_f1, best_nmi, best_adj))
    return line,true_clusters,best_f1, best_nmi, best_adj

def main(args):

    wf = open(LOCAL_EMB_DIR+'/out/results_test-2.18-distance.csv', 'w', encoding='utf-8')
    wf.write('name,num_nodes,true_clusters,pf1,nmi,ari\n')
    metrics = np.zeros(3)
    # metrics = [0.4739511096783046, 0.6130561209982938, 0.3641500385028091]
    # cnt = 1
    cnt = 0
    # f = open("../../datas/cluster_pred/n_clusters_rnn.txt","r")
    # pre_data = f.readlines()
    for index,name in enumerate(valid_author_name_ids["author_name"]):
        start_time = time.time()
        name = convert(name)
        print(index,name)
        if index>=0:
            num_nodes,true_clusters,best_f1, best_nmi, best_adj=total_local__embedding(name,valid_global_embedding,args)
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
            time_acc = time.time() - start_time
            print(cnt, 'names', time_acc, 'avg time', time_acc / cnt)
    average_f1 = metrics[0] / cnt
    average_nmi = metrics[1] / cnt
    average_adj = metrics[2] / cnt
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        average_f1, average_nmi, average_adj))
    wf.close()


if __name__ == '__main__':
    main(args)
    # total_local__embedding("haifengqian", valid_global_embedding, args)
    # total_local__embedding("huicai", valid_global_embedding, args)
    # total_local__embedding("jianguowu", valid_global_embedding, args)
    # total_local__embedding("weishi", valid_global_embedding, args)
    # total_local__embedding("bozou", valid_global_embedding, args)
    # total_local__embedding("jinyang", valid_global_embedding, args)
    # total_local__embedding("qinglingzhang", valid_global_embedding, args)
    # total_local__embedding("hailin", valid_global_embedding, args)
    # total_local__embedding("fenxu", valid_global_embedding, args)
    # total_local__embedding("xiaominli", valid_global_embedding, args)
    # total_local__embedding("jianrongli", valid_global_embedding, args)
    # total_local__embedding("hongxialiu", valid_global_embedding, args)
    # total_local__embedding("dongshengzhou", valid_global_embedding, args)

