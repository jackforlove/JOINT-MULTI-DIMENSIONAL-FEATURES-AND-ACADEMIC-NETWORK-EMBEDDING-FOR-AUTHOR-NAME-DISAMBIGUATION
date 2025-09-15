import math
import multiprocessing
import random
import time
import re

import pandas as pd
import warnings

import scipy.sparse as sp
import torch
from sklearn.cluster import SpectralClustering, KMeans,AgglomerativeClustering,DBSCAN
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
#%%
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


SEED = 42

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_embedding"
GLOBAL_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/global_embedding"
PRE_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
HIN_DIR = "../../datas/OAG-WhoisWh0-na-v1/hin_embedding"


# 导入数据
train_pub_pre_embedding = pd.read_pickle(DATA_DIR+'/train_pub_pre_embedding.pkl')
test_pub_pre_embedding = pd.read_pickle(DATA_DIR+'/test_pub_pre_embedding.pkl')
test_global_embedding = pd.read_pickle(GLOBAL_DATA_DIR+'/test_global_embedding.pkl')
hin_embedding = pd.read_pickle(HIN_DIR+'/hin_embedding.pkl')


train_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/train_author_name_ids.pkl')
valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/test_author_name_ids.pkl')
print()
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

def clustering(Cluster, feature, true_labels):

    f_adj = np.matmul(feature, np.transpose(feature))


    predict_labels = Cluster.fit_predict(f_adj)


    cm = clustering_metrics(true_labels, predict_labels)
    nmi, adj = cm.evaluationClusterModelFromLabel(tqdm)
    prec, rec, f1 = pairwise_precision_recall_f1(predict_labels, true_labels)

    return f1, nmi, adj
def scale(z):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / (zmax - zmin)
    z_scaled = z_std
    return z_scaled
def data_for_dimins(name,relation,embedding):


    local_emb = embedding.loc[embedding["author_name"] == name]
    paper_id = local_emb["paper_id"]
    features = local_emb[relation]
    labels = local_emb["author_id"]

    paper_id = paper_id.values
    features = features.values
    features = features.tolist()
    features = np.array(features)
    labels = labels.values
    labels = encode_labels(labels)


    features = torch.FloatTensor(features)


    # Store original adjacency matrix (without diagonal entries) for later

    sm_fea_s = sp.csr_matrix(features).toarray()
    # sm_fea_s = normalize_vectors(sm_fea_s)


    sm_fea_s = torch.FloatTensor(sm_fea_s)
    # sm_fea_s = scale(sm_fea_s)
    # sm_fea_s = torch.nn.functional.softmax(sm_fea_s)



    return sm_fea_s,labels

def evalution_embedding(name,embedding):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    inx_authors, labels_authors = data_for_dimins(name, 'authors',embedding)
    inx_title, labels_title = data_for_dimins(name,  'title',embedding)
    # inx_keywords, labels_keywords = data_for_dimins(name, 'keywords',embedding)
    inx_venue, labels_venue = data_for_dimins(name, 'venue',embedding)
    inx_author_org, labels_author_org = data_for_dimins(name, 'author_org',embedding)
    inx_abstract, labels_abstract = data_for_dimins(name, 'abstract',embedding)
    # inx, labels = data_for_dimins(name, 'hin_emb',embedding)

    inx_all = inx_authors +  inx_title+inx_author_org+inx_venue+inx_abstract
    line = len(inx_author_org)
    true_clusters = max(labels_author_org) + 1
    # inx_all_feature = scale(inx_all).cpu().data.numpy()
    Cluster = SpectralClustering(n_clusters=true_clusters, affinity='precomputed', random_state=0)
    # Cluster =AgglomerativeClustering(n_clusters=true_clusters)
    f1, nmi, adj = clustering(Cluster, inx_all, labels_author_org)
    print('聚类结果：f1: {}, nmi: {}, adj: {}'.format(f1, nmi, adj))
    return line, true_clusters, f1, nmi, adj

def main():
    wf = open(GLOBAL_DATA_DIR+'/out/results_test_2.17.csv', 'w', encoding='utf-8')
    wf.write('name,num_nodes,true_clusters,pf1,nmi,ari\n')
    metrics = np.zeros(3)
    cnt = 0

    for index,name in enumerate(valid_author_name_ids["author_name"]):
        start_time = time.time()
        name = convert(name)
        print(index,name)
        # if name == 'jianzhou_wang':
        if index>=0:
            try:

                num_nodes,true_clusters,best_f1, best_nmi, best_adj=evalution_embedding(name,test_global_embedding)
                    # break

                wf.write('{0},{1},{2},{3},{4:.5f},{5:.5f}\n'.format(
                    name, num_nodes, true_clusters,best_f1, best_nmi, best_adj))
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
            except Exception as e:
                print(e)
    average_f1 = metrics[0] / cnt
    average_nmi = metrics[1] / cnt
    average_adj = metrics[2] / cnt
    wf.write('average,,,{0:.5f},{1:.5f},{2:.5f}\n'.format(
        average_f1, average_nmi, average_adj))
    wf.close()


if __name__ == '__main__':
    main()
