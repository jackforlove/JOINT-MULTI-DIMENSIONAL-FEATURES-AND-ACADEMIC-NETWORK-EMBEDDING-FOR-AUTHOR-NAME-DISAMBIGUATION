import random
import re

import fsspec
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

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"

# 乱码及停用词
rrrr = '[!“”"#$%&\'()*+,-./:)‘′(;<=>?@[\\]^-_` •！@#￥&*（）——+~}】【|、|？《》，。：“；‘{|}~—～’*《》<>「」{}【】()/\\\[\] ]+'
stop_words = []
with open(DATA_DIR+"/stop_words.txt","r")as f:
    for line in f.readlines():
        stop_words.append(line.replace("\n",""))


random.seed(42)
np.random.seed(42)

# 导入数据
# pub_info = pd.read_pickle(DATA_DIR+'/pub_info.pkl').drop_duplicates(subset='paper_id', keep='first')
pub_info = pd.read_pickle(DATA_DIR+'/pub_info.pkl')
train_pub_info = pd.read_pickle(DATA_DIR+'/train_pub_info.pkl')
train_author_name_ids = pd.read_pickle(DATA_DIR+'/train_author_name_ids.pkl')
train_author_paper_ids = pd.read_pickle(DATA_DIR+'/train_author_paper_ids.pkl')
train_author_name_paper_ids = pd.read_pickle(DATA_DIR+'/train_author_name_paper_ids.pkl')

valid_pub_info = pd.read_pickle(DATA_DIR+'/test_pub_info.pkl')
valid_author_name_ids = pd.read_pickle(DATA_DIR+'/test_author_name_ids.pkl')
valid_author_name_paper_ids = pd.read_pickle(DATA_DIR+'/test_author_name_paper_ids.pkl')


with open(DATA_DIR+'/author_name_map.pkl', 'rb') as file:
    author_name_map = pickle.load(file)
with open(DATA_DIR+'/author_org_map.pkl', 'rb') as file:
    author_org_map = pickle.load(file)

def to_pinyin(word):
    s = ''
    for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
        s += ''.join(i)
    return s

def convert_sentence(sentence):

    sentence = sentence.lower()
    sentence = to_pinyin(sentence)
    sentence = re.sub(rrrr, ' ', sentence.strip().lower())
    sentence = re.sub(r'\s{2,}', ' ', sentence).strip()
    sentence = re.sub(r'[0-9]', ' ', sentence).strip()
    return sentence.split()

def convert_words(word):
    word = word.strip().lower()
    word = to_pinyin(word)
    word = re.sub(rrrr, ' ', word)
    word = re.sub(r'\s{2,}', '', word).strip()
    word = re.sub(r'[0-9]', '', word).strip()
    new_word = ""
    for w in word.split():
        # if w not in stop_words:
            new_word +=w
    return new_word


def deal_venue(venue):
    s = []
    try:
        for i in convert_sentence(venue):
            s.append( convert_words(i))
    except:
        pass
    return ' '.join(s)

def deal_keywords(words):
    list_words = []
    if type(words) is not float:
        for i in words:
            list_words.append(deal_venue(i))
    return ' '.join(list_words)

# 训练数据准备
train_author_name_ids['author_num'] = train_author_name_ids['author_ids'].apply(len)
train_author_name_ids = train_author_name_ids[train_author_name_ids['author_num'] >= 2]
train_author_name_ids_ext = []
for author_name, author_ids in train_author_name_ids[['author_name', 'author_ids']].values:
    for aid in author_ids:
        train_author_name_ids_ext.append([author_name, aid])
train_author_name_ids_ext = pd.DataFrame(train_author_name_ids_ext, columns=['author_name', 'author_id'])

train_author_paper_ids['paper_num'] = train_author_paper_ids['paper_ids'].apply(len)
author_pub_more_than_k = train_author_paper_ids[train_author_paper_ids['paper_num'] > 5]['author_id'].unique()
train_author_name_ids_ext = train_author_name_ids_ext[train_author_name_ids_ext['author_id'].isin(author_pub_more_than_k)]
print(len(train_author_name_ids_ext))

train_author_paper_ids_ext = []
for author_id, paper_ids in train_author_paper_ids[['author_id', 'paper_ids']].values:
     for pid in paper_ids:
            train_author_paper_ids_ext.append([author_id, pid])
train_author_paper_ids_ext = pd.DataFrame(train_author_paper_ids_ext, columns=['author_id', 'paper_id'])

train_author_paper_ids_ext = train_author_paper_ids_ext.merge(train_author_name_ids_ext, 'left', 'author_id')

train_author_paper_ids_ext = train_author_paper_ids_ext.dropna(subset=['author_name'])
train_author_paper_ids_ext['author_org'] = train_author_paper_ids_ext.apply(lambda row: ' '.join([convert_words(str(i))for i in convert_sentence(str(author_org_map[row['author_name']][row['paper_id']]))]), axis=1)
train_data = pub_info.merge(train_author_paper_ids_ext,"left","paper_id")
train_data = train_data.dropna(subset=['author_name'])
# train_data = train_data.set_index('paper_id')

train_data["authors"] = train_data["authors"].apply(lambda name: ' '.join([convert_words(i) for i in name]))
train_data["title"] = train_data["title"].apply(lambda word:' '.join([convert_words(i) for i in convert_sentence(word)]))
train_data["abstract"] = train_data["abstract"].apply(lambda word:' '.join([convert_words(i) for i in convert_sentence(word)]))
# train_data["keywords"] = train_data["keywords"].apply(lambda words:' '.join([deal_venue(i) for i in words]))
train_data["keywords"] = train_data["keywords"].apply(lambda words: deal_keywords(words))
train_data["author_name"] = train_data["author_name"].apply(lambda i:convert_words(i))
train_data["venue"] = train_data["venue"].apply(lambda words:' '.join([convert_words(i) for i in convert_sentence(words)]))
train_data = train_data.drop('orgs', 1)
train_data = train_data.drop_duplicates(subset='paper_id')
train_data = train_data.set_index('paper_id')
train_data.to_pickle(DATA_DIR+'/train_data.pkl')





# 测试数据准备
valid_author_name_ids['author_num'] = valid_author_name_ids['author_ids'].apply(len)
valid_author_name_ids = valid_author_name_ids[valid_author_name_ids['author_num'] >= 2]
valid_author_name_ids_ext = []
for author_name, author_ids in valid_author_name_ids[['author_name', 'author_ids']].values:
    for aid in author_ids:
        valid_author_name_ids_ext.append([author_name, aid])
valid_author_name_ids_ext = pd.DataFrame(valid_author_name_ids_ext, columns=['author_name', 'author_id'])



valid_author_name_paper_ids['paper_num'] = valid_author_name_paper_ids['paper_ids'].apply(len)
author_pub_more_than_k = valid_author_name_paper_ids[valid_author_name_paper_ids['paper_num'] > 5]['author_id'].unique()
valid_author_name_ids_ext = valid_author_name_ids_ext[valid_author_name_ids_ext['author_id'].isin(author_pub_more_than_k)]
print(len(valid_author_name_ids_ext))
valid_author_paper_ids_ext = []
for author_id, paper_ids,author_name in valid_author_name_paper_ids[['author_id', 'paper_ids','author_name']].values:
     for pid in paper_ids:
            valid_author_paper_ids_ext.append([author_id, pid,author_name])


valid_author_paper_ids_ext = pd.DataFrame(valid_author_paper_ids_ext, columns=['author_id', 'paper_id','author_name'])

# valid_author_paper_ids_ext = valid_author_paper_ids_ext.merge(valid_author_name_ids_ext, 'left', 'author_id','author_name')

valid_author_paper_ids_ext = valid_author_paper_ids_ext.dropna(subset=['author_name'])
valid_author_paper_ids_ext['author_org'] = valid_author_paper_ids_ext.apply(lambda row: ' '.join([convert_words(str(i))for i in convert_sentence(str(author_org_map[row['author_name']][row['paper_id']]))]), axis=1)
valid_data = pub_info.merge(valid_author_paper_ids_ext,"left","paper_id")

valid_data = valid_data.dropna(subset=['author_name'])
# valid_data = valid_data.set_index('paper_id')
valid_data["authors"] = valid_data["authors"].apply(lambda name: ' '.join([convert_words(i) for i in name]))
valid_data["title"] = valid_data["title"].apply(lambda word:' '.join([convert_words(i) for i in convert_sentence(word)]))
valid_data["abstract"] = valid_data["abstract"].apply(lambda word:' '.join([convert_words(i) for i in convert_sentence(word)]))
# valid_data["keywords"] = valid_data["keywords"].apply(lambda words:' '.join([deal_venue(i) for i in words]))
valid_data["keywords"] = valid_data["keywords"].apply(lambda words:deal_keywords(words))
valid_data["venue"] = valid_data["venue"].apply(deal_venue)

valid_data["author_name"] = valid_data["author_name"].apply(lambda i:convert_words(i))

valid_data = valid_data.drop('orgs', 1)
valid_data = valid_data.drop_duplicates(subset='paper_id')
valid_data = valid_data.set_index('paper_id')
valid_data.to_pickle(DATA_DIR+'/test_data.pkl')

