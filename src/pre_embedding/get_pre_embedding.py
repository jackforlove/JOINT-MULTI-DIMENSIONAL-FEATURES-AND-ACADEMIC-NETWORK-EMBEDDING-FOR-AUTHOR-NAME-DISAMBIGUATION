import math
import multiprocessing
import random

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#%%
import pandas as pd
import numpy as np
import os
import sys
import pypinyin
from collections import defaultdict
from tqdm import tqdm_notebook
import pickle
from gensim.models import word2vec,Word2Vec

random.seed(42)
np.random.seed(42)

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
IN_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
OUT_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_embedding"

# 导入数据
train_data = pd.read_pickle(IN_DATA_DIR+'/train_data.pkl')
valid_data = pd.read_pickle(IN_DATA_DIR+'/test_data.pkl')

save_model_name = OUT_DATA_DIR+"/word2vec_xl_100.model"
model = word2vec.Word2Vec.load(save_model_name)
word_input = 100

cols = ['title', 'abstract','authors', 'venue','author_org']


def get_embedding(pub_info,style):
    pub_info_drop = pub_info.drop("year", 1)
    pub_info_drop = pub_info_drop.drop("author_id", 1)
    pub_info_drop = pub_info_drop.drop("author_name", 1)
    pub_info_drop = pub_info_drop.reset_index()
    pub_pre_embedding = pd.DataFrame()
    sentences = np.array(pub_info_drop).flatten().tolist()
    new_sentences = []
    for i in sentences:
        if len(i) > 2:
            for word in i.split():
                new_sentences.append(word)
    word_num = len(new_sentences)
    idf_word = pd.value_counts(new_sentences)
    pub_pre_embedding["paper_id"] = pub_info_drop["paper_id"]
    pub_info = pub_info.reset_index()
    pub_pre_embedding["author_id"] = pub_info["author_id"]
    pub_pre_embedding["author_name"] = pub_info["author_name"]
    pub_pre_embedding["year"] = pub_info["year"]
    for c in cols:
        print(c)
        randoms = 0
        list_emb = []
        paper_keys = []
        pub_info_c = list(pub_info_drop[c].values)
        for paper in pub_info_c:
            words_vec = 0
            tag = True
            if len(paper) > 1:
                split_cut = paper.split()
                for j in split_cut:
                    if j in model.wv:
                        idf = math.log(word_num / idf_word[j] + 1)
                        words_vec += model.wv[j] *idf
                        tag = False
                # if tag==False:
                    # words_vec = words_vec/len(split_cut)
            if tag:
                words_vec = 2 * np.random.random(word_input) - 1
                randoms+=1
                # print(paper)
            words_vec = np.array(words_vec)
            # if paper in paper_keys:
            #     print(paper)
            # else:
            #     paper_keys.append(paper)
            list_emb.append(words_vec)
        pub_pre_embedding[c] = list_emb
        print("随机生成的特征个数为:"+str(randoms))
    pub_pre_embedding.to_pickle(OUT_DATA_DIR + "/"+style+"_pub_pre_embedding.pkl")

if __name__ == "__main__":
    get_embedding(train_data,"train")
    get_embedding(valid_data,"test")