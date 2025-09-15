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
from tqdm.notebook import tqdm
import pickle

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"

random.seed(42)
np.random.seed(42)


# 导入数据
# pub_info = pd.read_pickle(DATA_DIR+'/pub_info.pkl').drop_duplicates(subset='paper_id', keep='first')
train_pub_info = pd.read_pickle(DATA_DIR+'/train_pub_info.pkl')
train_author_name_ids = pd.read_pickle(DATA_DIR+'/train_author_name_ids.pkl')
train_author_paper_ids = pd.read_pickle(DATA_DIR+'/train_author_paper_ids.pkl')
train_author_name_paper_ids = pd.read_pickle(DATA_DIR+'/train_author_name_paper_ids.pkl')

valid_pub_info = pd.read_pickle(DATA_DIR+'/test_pub_info.pkl')
valid_author_name_ids = pd.read_pickle(DATA_DIR+'/test_author_name_ids.pkl')
valid_author_name_paper_ids = pd.read_pickle(DATA_DIR+'/test_author_name_paper_ids.pkl')



def check_chs(c):
    return '\u4e00' <= c <= '\u9fa5'

def to_pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s


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


# 作者姓名处理
paper_authors = {}
for author_name, paper_ids in train_author_name_paper_ids[['author_name', 'paper_ids']].values:
    for pid in paper_ids:
        if not pid in paper_authors:
            paper_authors[pid] = [author_name]
        else:
            paper_authors[pid].append(author_name)
for author_name, paper_ids in valid_author_name_paper_ids[['author_name', 'paper_ids']].values:
    for pid in paper_ids:
        if not pid in paper_authors:
            paper_authors[pid] = [author_name]
        else:
            paper_authors[pid].append(author_name)
#%%
paper_authors_df = pd.DataFrame([(k, v) for k,v in paper_authors.items()], columns=['paper_id', 'author_ids'])

pub_info = pd.concat([train_pub_info, valid_pub_info])
pub_info['author_names'] = pub_info['authors'].apply(lambda x: [ao['name'] for ao in x])
#%%
pub_info = pub_info.merge(paper_authors_df, 'left', 'paper_id')
#%%
pub_info.head()


author_name_map = {}
for author_names, author_ids in tqdm(pub_info[['author_names', 'author_ids']].values):
    if type(author_ids) == float:
        continue
    for aid in author_ids:
        dis = []
        for an in author_names:
            dis.append(score(an, aid))
        cor = author_names[np.argmin(dis)]
        author_name_map[cor] = aid

with open(DATA_DIR+'/author_name_map.pkl', 'wb') as file:
    pickle.dump(author_name_map, file)

