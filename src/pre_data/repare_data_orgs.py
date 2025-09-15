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
from tqdm.notebook import tqdm_notebook
import pickle


random.seed(42)
np.random.seed(42)

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"

# 导入数据
# pub_info = pd.read_pickle(DATA_DIR+'/pub_info.pkl').drop_duplicates(subset='paper_id', keep='first')
train_pub_info = pd.read_pickle(DATA_DIR+'/train_pub_info.pkl')
train_author_name_ids = pd.read_pickle(DATA_DIR+'/train_author_name_ids.pkl')
train_author_paper_ids = pd.read_pickle(DATA_DIR+'/train_author_paper_ids.pkl')
train_author_name_paper_ids = pd.read_pickle(DATA_DIR+'/train_author_name_paper_ids.pkl')

valid_pub_info = pd.read_pickle(DATA_DIR+'/test_pub_info.pkl')
valid_author_name_ids = pd.read_pickle(DATA_DIR+'/test_author_name_ids.pkl')
valid_author_name_paper_ids = pd.read_pickle(DATA_DIR+'/test_author_name_paper_ids.pkl')

pub_info = pd.concat([train_pub_info, valid_pub_info])
print(pub_info.shape)
pub_info = pub_info.drop_duplicates(subset='paper_id', keep='first')
print(pub_info.shape)

pub_info = pub_info.set_index('paper_id')
with open(DATA_DIR+'/author_name_map.pkl', 'rb') as file:
    author_name_map = pickle.load(file)

train_author_name_ids_ext = []
for author_name, author_ids in train_author_name_ids[['author_name', 'author_ids']].values:
    for aid in author_ids:
        train_author_name_ids_ext.append([author_name, aid])

train_author_name_ids_ext = pd.DataFrame(train_author_name_ids_ext, columns=['author_name', 'author_id'])
train_author_name_paper_ids = train_author_paper_ids.merge(train_author_name_ids_ext, 'left', 'author_id')

author_name_paper_ids = pd.concat([train_author_name_paper_ids, valid_author_name_paper_ids])
# author_name_paper_ids = author_name_paper_ids.drop_duplicates(subset='author_id', keep='last')


author_org_map = defaultdict(dict)
author_id_org_map = defaultdict(list)
for author_id, author_name, paper_ids in tqdm_notebook(author_name_paper_ids[['author_id', 'author_name', 'paper_ids']].values):
    for pid in paper_ids:
        org = np.nan
        author_orgs = pub_info.loc[pid, 'authors']
        for ao in author_orgs:
            if not ao['name'] in author_name_map:
                continue
            if author_name_map[ao['name']] == author_name:
                org = ao.get('org')
        author_org_map[author_name][pid] = org
        author_id_org_map[author_id].append(org)

with open(DATA_DIR+'/author_org_map.pkl', 'wb') as file:
    pickle.dump(author_org_map, file)

author_id_org_map_df = pd.DataFrame()
author_id_org_map_df['author_id'] = author_id_org_map.keys()
author_id_org_map_df['orgs'] = author_id_org_map.values()
author_id_org_map_df.to_pickle(DATA_DIR+'/author_id_org_map.pkl')