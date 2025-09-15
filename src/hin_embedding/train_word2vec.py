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
# 导入数据
valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/valid_author_name_ids.pkl')
save_model_name = HIN_DIR + "/" + "hin_word2vec.model"


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

def main():
    new_sentences = []
    for index,name in enumerate(valid_author_name_ids["author_name"]):
        name = convert(name)
        print(index,name)
        with open(HIN_DIR + "/" + name + "/rand_walks.txt", "r") as f:
            for line in f.readlines():
                new_sentences.append(line.split())
    print(multiprocessing.cpu_count())
    word2vec_model = Word2Vec(new_sentences, min_count=1, workers=multiprocessing.cpu_count(),size=64)
    word2vec_model.save(save_model_name)
    print("训练完成！")



if __name__ == '__main__':
    main()


