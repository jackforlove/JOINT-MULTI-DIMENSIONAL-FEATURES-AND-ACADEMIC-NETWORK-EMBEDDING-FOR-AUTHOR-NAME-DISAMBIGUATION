import pickle
import pandas as pd
import pypinyin
from collections import defaultdict
import numpy as np
from random import randint
DATA_DIR = "../../datas/pre_data"
with open(DATA_DIR+'/author_org_map.pkl', 'rb') as file:
    author_org_map = pickle.load(file)
train_author_paper_ids = pd.read_pickle(DATA_DIR+'/train_author_paper_ids.pkl')
valid_author_name_paper_ids = pd.read_pickle(DATA_DIR+'/valid_author_name_paper_ids.pkl')
valid_author_name_paper_ids = valid_author_name_paper_ids.drop("author_name",1)
author_name_paper_ids = pd.concat([train_author_paper_ids,valid_author_name_paper_ids]).reset_index(drop=True)
author_id_org_map = pd.read_pickle(DATA_DIR+'/author_id_org_map.pkl')
# f = open(DATA_DIR+'/un_add_orgs_log.txt',"w")
f = open(DATA_DIR+'/un_add_orgs_log.txt',"r")
un_paper = []
for paper in f.readlines():
    un_paper.append(paper.replace("\n",""))
for name in author_org_map:
    for paper in author_org_map[name]:
        print(name,paper)
        if len(str(author_org_map[name][paper])) <2 and paper in un_paper:
            try:
                for index in list(author_name_paper_ids.index):
                    author_id = author_name_paper_ids.loc[index,"author_id"]
                    papers = author_name_paper_ids.loc[index,"paper_ids"]
                    if paper in papers:
                        orgs = list(author_id_org_map[author_id_org_map["author_id"] == author_id]["orgs"])[0]
                        while '' in orgs:
                            orgs.remove('')
                        if len(orgs) >=1:
                            author_org_map[name][paper] = orgs[randint(0,len(orgs))]
                        break
            except Exception as e:
                print(e)
f.close()

with open(DATA_DIR+'/author_org_map.pkl', 'wb') as file:
    pickle.dump(author_org_map, file)
