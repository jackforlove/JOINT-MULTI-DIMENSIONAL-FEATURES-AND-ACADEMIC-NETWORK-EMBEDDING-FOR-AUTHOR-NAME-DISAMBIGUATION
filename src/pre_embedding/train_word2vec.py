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
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v3"
IN_DATA_DIR = "../../datas/pre_data"
OUT_DATA_DIR = "../../datas/pre_embedding"

# 导入数据
train_data = pd.read_pickle(IN_DATA_DIR+'/train_data.pkl')
valid_data = pd.read_pickle(IN_DATA_DIR+'/valid_data.pkl')

save_model_name = OUT_DATA_DIR+"/word2vec512.model"


pub_info_embed = pd.DataFrame()
#%%
pub_info = pd.concat([train_data, valid_data]).drop_duplicates()
# pub_info = pub_info.reset_index()
pub_info = pub_info.drop("year",1)
pub_info = pub_info.drop("author_id",1)
pub_info = pub_info.drop("author_name",1)
sentences = np.array(pub_info).flatten().tolist()
new_sentences = []
for i in sentences:
    if len(i)>2:
        new_sentences.append(i.split())
print(multiprocessing.cpu_count())

word2vec_model = Word2Vec(new_sentences,min_count=2,workers=multiprocessing.cpu_count(),size=512)
word2vec_model.save(save_model_name)
print("训练完成")
# for c in cols:
#     print(c)
#     pub_info_c = list(pub_info[c].values)
#     pub_info_c_embed = model.encode(pub_info_c, show_progress_bar=True, batch_size=128)
#     pub_info_embed[c] = pub_info_c_embed
#     display(pub_info_embed.head(2))




