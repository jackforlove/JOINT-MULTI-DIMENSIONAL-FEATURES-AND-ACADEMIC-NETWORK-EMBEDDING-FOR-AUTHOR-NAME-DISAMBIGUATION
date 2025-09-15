import numpy as np
import pandas as pd
import pickle


DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
OUT_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/global_embedding"
IN_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_embedding"
PRE_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
# 导入数据
global_npy_data = np.load(OUT_DATA_DIR+'/global_embedding_npy.npy',allow_pickle=True).item()
pre_emb_data = pd.read_pickle(IN_DATA_DIR+'/test_pub_pre_embedding.pkl')

global_emb_data = pd.DataFrame()
global_emb_data["paper_id"] = pre_emb_data["paper_id"]
global_emb_data["author_id"] = pre_emb_data["author_id"]
global_emb_data["author_name"] = pre_emb_data["author_name"]
global_emb_data["year"] = pre_emb_data["year"]

for sty in global_npy_data:
    global_emb = global_npy_data[sty]
    global_emb = map(lambda x: x, global_emb)
    global_emb_data[sty] = pd.Series(global_emb)

#
global_emb_data.to_pickle(OUT_DATA_DIR+"/test_global_embedding.pkl")

# valid_data_npy_dir = np.load(valid_data_npy_dir,allow_pickle=True)
print("done!")