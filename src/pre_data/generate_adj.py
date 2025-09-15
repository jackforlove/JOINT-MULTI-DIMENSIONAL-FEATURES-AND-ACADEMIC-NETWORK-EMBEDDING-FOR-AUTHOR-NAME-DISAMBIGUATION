from os.path import join
import os
import numpy as np
import re
from collections import defaultdict
import pandas as pd
import pypinyin
from numpy.random import shuffle
from rapidfuzz.distance import Levenshtein

from src.utils.utils import load_json,load_data,dump_data

DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
IN_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
OUT_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/adj"

# 导入数据
train_data = pd.read_pickle(IN_DATA_DIR+'/train_data.pkl')
valid_data = pd.read_pickle(IN_DATA_DIR+'/test_data.pkl')
valid_data = valid_data.reset_index()
valid_author_name_ids = pd.read_pickle(IN_DATA_DIR+'/test_author_name_ids.pkl')
def to_pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s

def check_chs(c):
    return '\u4e00' <= c <= '\u9fa5'


def CalSimJaccard(dataA,dataB):
    A_len,B_len=len(dataA),len(dataB)
    C=[i for i in dataA if i in dataB]#取交集
    C_len=len(C)#交集含有元素的个数
    return C_len/(A_len+B_len-C_len)

def score(n1, n2):
    n1 = ''.join(filter(str.isalpha, n1.lower()))
    if check_chs(n1):
#         print(n1)
        n1 = to_pinyin(n1)
#         print(n1)
    n2 = ''.join(filter(str.isalpha, n2.lower()))
    counter = defaultdict(int)
    score = 0
    for c in n1:
        counter[c] += 1
    for c in n2:
        if (c in counter) and (counter[c] > 0):
            counter[c] -= 1
        else:
            score += 1
    score += np.sum(list(counter.values()))
    return score

def convert(name):
    rrrr = '[!“”"#$%&\'()*+,-./:)‘′(;<=>?@[\\]^-_` •！@#￥&*（）——+~}】【|、|？《》，。：“；‘{|}~—～’*《》<>「」{}【】()/\\\[\] ]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the','by']
    name = name.lower()
    name = re.sub(rrrr, ' ', name.strip().lower())
    name = re.sub(r'\s{2,}', '', name).strip()
    name = re.sub(r'[0-9]', '', name).strip()
    new_name = ""
    for word in name.split():
        if word not in stopword:
            new_name +=word

    return new_name
def orgs_sim(n1,n2):
    return 1-Levenshtein.distance(n1,n2)/max(len(n1),len(n2))

def gen_local_adj(col = '',fname = ''):

    local_data = pd.DataFrame()
    local_data["paper_id"]= valid_data["paper_id"].loc[valid_data["author_name"] == name]
    local_data[col]= valid_data[col].loc[valid_data["author_name"] == name]
    local_data = local_data.reset_index()
    n_pubs = len(local_data)
    jaccard_sim = np.zeros((n_pubs,n_pubs))
    print('n_pubs', n_pubs)
    out_dir = OUT_DATA_DIR+"/"+col+"/"+fname
    os.makedirs(out_dir, exist_ok=True)
    wf_network = open(out_dir+"/pubs_network.txt", 'w')

    for i in range(n_pubs-1):
        if i % 100 == 0:
            print(i)
        paper_id_i = local_data["paper_id"].iloc[i]
        paper_info_i = local_data[col].iloc[i]
        split_cut_i = paper_info_i.split()

        if len(split_cut_i)<2:
            continue
        split_cut_i = np.array(split_cut_i)

        for j in range(n_pubs-1):
            if i != j:
                paper_id_j = local_data["paper_id"].iloc[j]
                paper_info_j = local_data[col].iloc[j]
                split_cut_j = paper_info_j.split()
                if len(split_cut_j) < 2:
                    continue
                split_cut_j = np.array(split_cut_j)
                if col == 'authors':
                    C = [i for i in split_cut_i if i in split_cut_j]  # 取交集
                    C_len = len(C)  # 交集含有元素的个数
                    if C_len>=2:
                        weight = CalSimJaccard(split_cut_i, split_cut_j)
                        wf_network.write('{}\t{}\t{}\n'.format(paper_id_i, paper_id_j,weight))
                elif col == "author_org" or col == "venue":
                     sim = orgs_sim(paper_info_i,paper_info_j)
                     if sim>=0.7:
                        wf_network.write('{}\t{}\t{}\n'.format(paper_id_i, paper_id_j, sim))
                else:
                    jaccard_sim[i,j] = CalSimJaccard(split_cut_i,split_cut_j)
    idf_threshold = np.mean(jaccard_sim) + 2 * np.std(jaccard_sim)
    print("idf_threshold", idf_threshold)
    if idf_threshold != 0:
        for i in range(n_pubs - 1):
            for j in range(n_pubs - 1):
                if i != j:
                    if jaccard_sim[i, j] > idf_threshold:
                        wf_network.write('{}\t{}\t{}\n'.format(local_data["paper_id"].iloc[i], local_data["paper_id"].iloc[j],jaccard_sim[i, j]))

    wf_network.close()

if __name__ == '__main__':

    for col in ['title', 'abstract', 'keywords','authors', 'venue','author_org']:
        graph_dir = OUT_DATA_DIR+"/"+col
        os.makedirs(graph_dir, exist_ok=True)
        for index, name in enumerate(valid_author_name_ids["author_name"]):
            if index >= 0:
                name = convert(name)
                print(index, name)
                gen_local_adj(col,name)
        print(col+"done!")

# hongjiezhang
# hvogel
# rgupta