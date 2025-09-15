import json
import pickle
import re
import pandas as pd
import pypinyin
from collections import defaultdict
import numpy as np
from rapidfuzz.distance import Levenshtein
from random import randint
from collections import OrderedDict
DATA_DIR = "../../datas/pre_data"
# def to_pinyin(word):
#     s = ''
#     for i in pypinyin.pinyin(word, style=pypinyin.NORMAL):
#         s += ''.join(i)
#     return s
#
# def check_chs(c):
#     return '\u4e00' <= c <= '\u9fa5'
#
# def score(n1, n2):
#     n1 = ''.join(filter(str.isalpha, n1.lower()))
#     if check_chs(n1):
# #         print(n1)
#         n1 = to_pinyin(n1)
# #         print(n1)
#     n2 = ''.join(filter(str.isalpha, n2.lower()))
#     counter = defaultdict(int)
#     score = 0
#     for c in n1:
#         counter[c] += 1
#     for c in n2:
#         if (c in counter) and (counter[c] > 0):
#             counter[c] -= 1
#         else:
#             score += 1
#     score += np.sum(list(counter.values()))
#     return score
# def convert(name):
#     rrrr = '[!“”"#$%&\'()*+,-./:)‘′(;<=>?@[\\]^-_` •！@#￥&*（）——+~}】【|、|？《》，。：“；‘{|}~—～’*《》<>「」{}【】()/\\\[\] ]+'
#     stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the']
#     name = name.lower()
#     name = to_pinyin(name)
#     name = re.sub(rrrr, ' ', name.strip().lower())
#     name = re.sub(r'\s{2,}', '', name).strip()
#     name = re.sub(r'[0-9]', '', name).strip()
#     new_name = ""
#     for word in name.split():
#         if word not in stopword:
#             new_name +=word
#
#     return new_name
#
# def orgs_sim(n1,n2):
#     return 1-Levenshtein.distance(n1,n2)/max(len(n1),len(n2))
#
# # print(score("shuxunge nongyue chunyang mingu chenghu","nongyue shuxunge  chunyang mingu chenghu"))
# # train_data = pd.read_pickle(DATA_DIR+'/train_author_name_ids.pkl')
# valid_data = pd.read_pickle(DATA_DIR+'/valid_author_name_ids.pkl')
# n1 = "School of Electrical Engineering & Automation, Henan, Polytechnic University,Jiaozuo,People’s Republic of China"
# n2 = "Department of Electrical Engineering and Automation, Tianjin University,Tianjin, People’s Republic of China"
# print(orgs_sim(n1,n2))
# print("-----------OAG-WhoisWh0-na-v1---------------")
# with open("../../datasets/OAG-WhoisWh0-na-v1/test/sna_test_author_raw.json",encoding="utf-8") as file:
#     sna_valid_pub = json.load(file, object_pairs_hook=OrderedDict)
#     names_v1 = list(dict(sna_valid_pub).keys())
#     for i in range(len(names_v1)):
#         if i % 5 == 0:
#             print()
#         print(names_v1[i],end='   ')
# print("-----------OAG-WhoisWh0-na-v2---------------")
# with open("../../datasets/OAG-WhoisWh0-na-v2/test/sna_test_author_raw.json",encoding="utf-8") as file:
#     sna_valid_pub = json.load(file, object_pairs_hook=OrderedDict)
#     names_v2 = list(dict(sna_valid_pub).keys())
#     for i in range(len(names_v2)):
#         if i % 5 == 0:
#             print()
#         print(names_v2[i],end='   ')
# print()
# print("-----------OAG-WhoisWh0-na-v3---------------")
# with open("../../datasets/OAG-WhoisWh0-na-v3/valid/sna_valid_raw.json",encoding="utf-8") as file:
#     sna_valid_pub = json.load(file, object_pairs_hook=OrderedDict)
#     names_v3 = list(dict(sna_valid_pub).keys())
#     for i in range(len(names_v3)):
#         if i % 5 == 0:
#             print()
#         print(names_v3[i],end='   ')
# v2_num = 0
# v3_num = 0
# for i in names_v1:
#     if i in names_v2:
#         v2_num+1
#     if i in names_v3:
#         v3_num+1
#
# print(v2_num)
# print(v3_num)

DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v2"
DATA_DIR = "../../datas/OAG-WhoisWh0-na-v2/hin_embedding"
res = pd.read_pickle(DATA_DIR+"/hin_embedding.pkl")
print(res)