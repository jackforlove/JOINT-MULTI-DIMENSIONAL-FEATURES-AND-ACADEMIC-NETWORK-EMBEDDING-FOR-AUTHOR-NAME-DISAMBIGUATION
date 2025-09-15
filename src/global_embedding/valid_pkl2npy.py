import numpy as np
import pandas as pd
import pickle


DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
OUT_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/global_embedding"
IN_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_embedding"
PRE_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
# 导入数据
valid_data = pd.read_pickle(IN_DATA_DIR+'/test_pub_pre_embedding.pkl')



valid_data_npy = {}
cols = ['title','authors', 'abstract', 'venue', 'author_org']
for index in range(len(valid_data)):
    info = valid_data.loc[index]
    pid = info["paper_id"]
    dics = {}
    for clo in cols:
        valid_data_clo = info[clo]
        dics[clo] = valid_data_clo
    if pid in valid_data_npy.keys():
        print(pid)
    else:
        valid_data_npy[pid] = dics
#
valid_data_npy_dir = IN_DATA_DIR+"/test_pub_pre_embedding_npy.npy"
valid_data_npy_dir = np.save(valid_data_npy_dir,valid_data_npy)

# valid_data_npy_dir = np.load(valid_data_npy_dir,allow_pickle=True)
print(valid_data_npy_dir)