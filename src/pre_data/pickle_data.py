import json
import os
import random

import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm_notebook
# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"


random.seed(42)
np.random.seed(42)

# 加载测试集
with open(DATASET_DIR+"/test/sna_test_pub.json",encoding="utf-8") as file:
    sna_valid_pub = json.load(file, object_pairs_hook=OrderedDict)
with open(DATASET_DIR+"/test/sna_test_author_ground_truth.json",encoding="utf-8") as file:
    sna_valid_ground_truth = json.load(file, object_pairs_hook=OrderedDict)

#加载训练集
with open(DATASET_DIR+"/train/train_pub.json",encoding="utf-8") as file:
    train_pub = json.load(file, object_pairs_hook=OrderedDict)
with open(DATASET_DIR+"/train/train_author.json",encoding="utf-8") as file:
    train_author = json.load(file, object_pairs_hook=OrderedDict)

# 作者姓名-作者ids
train_author_names = list(train_author.keys())
train_author_ids = [list(v.keys()) for v in train_author.values()]
train_author_name_ids = pd.DataFrame(list(zip(train_author_names, train_author_ids)), columns=['author_name', 'author_ids'])

valid_author_names = list(sna_valid_ground_truth.keys())
valid_author_ids = [list(v.keys()) for v in sna_valid_ground_truth.values()]
valid_author_name_ids = pd.DataFrame(list(zip(valid_author_names, valid_author_ids)), columns=['author_name', 'author_ids'])

train_author_name_ids.to_pickle(DATA_DIR+'/train_author_name_ids.pkl')
valid_author_name_ids.to_pickle(DATA_DIR+'/test_author_name_ids.pkl')

#作者id-论文id
train_author_paper_ids = pd.DataFrame([(k2, v2) for v1 in train_author.values() for k2, v2 in v1.items()], columns=['author_id', 'paper_ids'])
# valid_author_paper_ids = pd.DataFrame([(k2,name, v2) for name,v1 in sna_valid_ground_truth for k2, v2 in v1.values().items()], columns=['author_id', 'author_name', 'paper_ids'])
train_author_paper_list = []
valid_author_paper_list = []

for name in train_author:
    v1 = train_author[name]
    for k2, v2 in v1.items():
        train_author_paper_list.append((k2,name, v2))

for name in sna_valid_ground_truth:
    v1 = sna_valid_ground_truth[name]
    for k2, v2 in v1.items():
        valid_author_paper_list.append((k2,name, v2))

train_author_name_paper_ids = pd.DataFrame(train_author_paper_list, columns=['author_id', 'author_name', 'paper_ids'])
valid_author_name_paper_ids = pd.DataFrame(valid_author_paper_list, columns=['author_id', 'author_name', 'paper_ids'])

train_author_name_paper_ids.to_pickle(DATA_DIR+'/train_author_name_paper_ids.pkl')
valid_author_name_paper_ids.to_pickle(DATA_DIR+'/test_author_name_paper_ids.pkl')
train_author_paper_ids.to_pickle(DATA_DIR+'/train_author_paper_ids.pkl')


# 论文信息
train_pub_info = pd.DataFrame.from_dict(train_pub, orient='index').reset_index(drop=True).rename({'id':'paper_id'}, axis=1)
valid_pub_info = pd.DataFrame.from_dict(sna_valid_pub, orient='index').reset_index(drop=True).rename({'id':'paper_id'}, axis=1)

pub_info = pd.concat([train_pub_info, valid_pub_info]).drop_duplicates(subset='paper_id', keep='first')
pub_info['orgs'] = pub_info['authors'].apply(lambda x: [ao['org'] for ao in x if 'org' in ao])
pub_info['authors'] = pub_info['authors'].apply(lambda x: [ao['name'] for ao in x if 'name' in ao])
pub_info['year'] = pub_info['year'].fillna(0).replace('', 0).astype(int)
pub_info['abstract'] = pub_info['abstract'].fillna(' ').replace('', ' ')

#
#
#
train_pub_info.to_pickle(DATA_DIR+'/train_pub_info.pkl')
valid_pub_info.to_pickle(DATA_DIR+'/test_pub_info.pkl')

pub_info = pub_info.drop_duplicates(subset='paper_id', keep='first')
pub_info = pub_info.set_index('paper_id')
pub_info.to_pickle(DATA_DIR+'/pub_info.pkl')

author_pub_ids = valid_author_name_paper_ids[['author_id','paper_ids']].merge(train_author_paper_ids, 'left', 'author_id')

author_pub_ids['paper_ids_x_len'] = author_pub_ids['paper_ids_x'].apply(len)
author_pub_ids['paper_ids_y_len'] = author_pub_ids['paper_ids_y'].apply(lambda x: 0 if type(x) == float else len(x))

author_pub_ids['paper_ids'] = author_pub_ids.apply(lambda row: list(set(row['paper_ids_x']) | (set() if type(row['paper_ids_y']) == float else set(row['paper_ids_y']))), axis=1)

author_pub_ids['paper_ids_len'] = author_pub_ids['paper_ids'].apply(len)

author_pub_ids.drop(columns=['paper_ids_x', 'paper_ids_y', 'paper_ids_x_len', 'paper_ids_y_len'], inplace=True)

author_pub_ids.head()
#%%
author_pub_ids['paper_ids_len'].describe()

author_pub_ids_ = author_pub_ids[['author_id', 'paper_ids']].values
pub_col = ['abstract', 'keywords', 'title', 'venue', 'year', 'authors', 'orgs']
for pc in pub_col:
    print(pc)
    dat = []
    for author_id, paper_ids in tqdm_notebook(author_pub_ids_):
        d = []
        for pid in paper_ids:
            d.append(pub_info.loc[pid, pc])
        dat.append(d)
    author_pub_ids[pc] = dat

author_pub_ids.to_pickle(DATA_DIR+'/author_pub_detail.pkl')




