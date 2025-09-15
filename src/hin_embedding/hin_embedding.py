import multiprocessing
import random
import re

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
#%%
import pandas as pd
import numpy as np

from gensim.models import word2vec,Word2Vec

random.seed(42)
np.random.seed(42)

# 数据路径
HIN_DIR = "../../datas/hin_embedding"
PRE_DATA_DIR = "../../datas/pre_data"
GLOBAL_EMB_DIR = "../../datas/global_embedding"
# 导入数据
valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/valid_author_name_ids.pkl')
valid_global_embedding = pd.read_pickle(GLOBAL_EMB_DIR+'/valid_global_embedding.pkl')

save_model_name = HIN_DIR + "/" + "hin_word2vec.model"
model = word2vec.Word2Vec.load(save_model_name)

def convert(name):
    rrrr = '[!“”"#$%&\'()*+,-./:)‘′(;<=>?@[\\]^-_` •！@#￥&*（）——+~}】【|、|？《》，。：“；‘{|}~—～’*《》<>「」{}【】()/\\\[\] ]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the']
    name = name.lower()
    name = re.sub(rrrr, ' ', name.strip().lower())
    name = re.sub(r'\s{2,}', '', name).strip()
    name = re.sub(r'[0-9]', '', name).strip()
    new_name = ""
    for word in name.split():
        if word not in stopword:
            new_name +=word
    return new_name
def hin_emd(paper_id):
    if paper_id in model.wv:
        emb = model.wv[paper_id]
    else:
        print(paper_id+"不在词库中!")
        emb = 2 * np.random.random(64) - 1
    return np.array(emb)
def main():
    hin_embdding = pd.DataFrame()
    hin_embdding["paper_id"] = valid_global_embedding["paper_id"]
    hin_embdding["author_id"] = valid_global_embedding["author_id"]
    hin_embdding["author_name"] = valid_global_embedding["author_name"]
    hin_embdding["year"] = valid_global_embedding["year"]
    hin_embdding["hin_emb"] = hin_embdding["paper_id"].apply(hin_emd)
    print("训练完成！")

    hin_embdding.to_pickle(HIN_DIR+"/hin_embedding.pkl")



if __name__ == '__main__':
    main()
