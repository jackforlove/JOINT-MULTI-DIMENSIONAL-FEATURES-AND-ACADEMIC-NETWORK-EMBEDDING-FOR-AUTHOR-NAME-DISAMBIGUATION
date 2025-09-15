from os.path import join
import os
import multiprocessing as mp
import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import pandas as pd

from src.utils.utils import load_json
from src.utils.utils import dump_data
import numpy as np

start_time = datetime.now()

"""
This class generates triplets of author embeddings to train global model
"""
random.seed(42)
np.random.seed(42)

# 数据路径
DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
IN_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_embedding"
OUT_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/global_embedding"
PRE_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
# 导入数据
train_data = pd.read_pickle(IN_DATA_DIR+'/train_pub_pre_embedding.pkl')
valid_data = pd.read_pickle(IN_DATA_DIR+'/test_pub_pre_embedding.pkl')
train_author_name_paper_ids = pd.read_pickle(PRE_DATA_DIR+'/train_author_name_paper_ids.pkl')
valid_author_name_paper_ids = pd.read_pickle(PRE_DATA_DIR+'/test_author_name_paper_ids.pkl')
train_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/train_author_name_ids.pkl')
valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/test_author_name_ids.pkl')


def normalize_vectors(vectors):
    scaler = StandardScaler()
    vectors_norm = scaler.fit_transform(vectors)
    return vectors_norm


class TripletsGenerator:
    name2pubs_train = {}
    name2pubs_test = {}
    names_train = None
    names_test = None
    n_pubs_train = None
    n_pubs_test = None
    pids_train = []
    pids_test = []
    n_triplets = 0
    batch_size = 10000
    anchor_embs = []
    pos_embs = []
    neg_embs = []

    def __init__(self, train_scale=100000, feature_style = "authors"):

        self.save_size = train_scale
        self.o_idr = OUT_DATA_DIR+"/"+feature_style
        self.prepare_data()
        self.feature_style = feature_style

    def prepare_data(self):
        self.names_train = train_author_name_ids["author_name"].values
        print('names train', len(self.names_train))
        self.names_test = valid_author_name_ids["author_name"].values
        print('names test', len(self.names_test))

        self.name2pubs_train = train_author_name_paper_ids
        self.name2pubs_test = valid_author_name_paper_ids


        self.train_dict = train_data
        self.test_dict = valid_data


        self.pids_train = self.train_dict["paper_id"].values
        # random.shuffle(self.pids_train)
        self.n_pubs_train = len(self.pids_train)
        print('pubs2train', self.n_pubs_train)

        self.pids_test = self.test_dict["paper_id"].values
        # random.shuffle(self.pids_test)
        self.n_pubs_test = len(self.pids_test)
        print('pubs2test', self.n_pubs_test)




    def gen_neg_pid(self, not_in_pids, role='train'):
        if role == 'train':
            sample_from_pids = self.pids_train
        else:
            sample_from_pids = self.pids_test
        while True:
            idx = random.randint(0, len(sample_from_pids)-1)
            pid = sample_from_pids[idx]
            if pid not in not_in_pids:
                return pid

    def sample_triplet_ids(self, task_q,role='train', N_PROC=8):
        n_sample_triplets = 0
        if role == 'train':
            names = self.names_train
            name2pubs = self.name2pubs_train
        else:  # test
            names = self.names_test
            name2pubs = self.name2pubs_test
            self.save_size = 50000  # test save size
        for pub_items in name2pubs["paper_ids"]:
                if len(pub_items) == 1:
                    continue
                pids = pub_items
                cur_n_pubs = len(pids)
                random.shuffle(pids)
                for i in range(cur_n_pubs):
                    pid1 = pids[i]  # pid

                    # batch samples
                    n_samples_anchor = min(6, cur_n_pubs)
                    idx_pos = random.sample(range(cur_n_pubs), n_samples_anchor)
                    for ii, i_pos in enumerate(idx_pos):
                        if i_pos != i:
                            if n_sample_triplets % 100 == 0:
                                print('sampled triplet ids', n_sample_triplets)
                            pid_pos = pids[i_pos]
                            pid_neg = self.gen_neg_pid(pids, role)
                            n_sample_triplets += 1
                            task_q.append((pid1, pid_pos, pid_neg))


    def dump_triplets(self, role='train'):
        if role == 'train':
            out_dir = join(self.o_idr, 'triplets-{}'.format(self.save_size))
        else:
            out_dir = join(self.o_idr, 'test-triplets')
        os.makedirs(out_dir, exist_ok=True)

        f_idx = 0
        task_q = []
        self.sample_triplet_ids(task_q, role)
        n_sample_triplets=0

        if role == 'train':
            load_dict = self.train_dict
        else:  # test
            load_dict = self.test_dict
        # papers = load_dict["paper_id"].values
        new_load_dict = pd.DataFrame()
        new_load_dict["paper_id"] = load_dict["paper_id"]
        # feature = load_dict[self.feature_style].values
        # feature = feature.tolist()
        # feature = np.array(feature)
        # feature = normalize_vectors(feature)
        # data_list = map(lambda x: x, feature)
        # data_list_ser = pd.Series(data_list)
        new_load_dict[self.feature_style] = load_dict[self.feature_style]
        'authors', 'title', 'abstract', 'keywords', 'venue', 'author_org'
        # new_load_dict[self.feature_style] = load_dict['authors']+load_dict['title']+load_dict['abstract']+load_dict['venue']
        new_load_dict = new_load_dict.set_index("paper_id")
        paper_dict = new_load_dict.to_dict()
        paper_dict = paper_dict[self.feature_style]

        # train_idr = IN_DATA_DIR+"/all_paper_train_vec_1.npy"
        # paper_dict = np.load(train_idr,allow_pickle=True).item()

        for i in task_q:
            pid1, pid_pos, pid_neg = i[0],i[1],i[2]

            if pid1 in paper_dict.keys() and pid_pos in paper_dict.keys() and pid_neg in paper_dict.keys():

                emb1 = paper_dict[pid1]
                self.anchor_embs.append(emb1)

                emb_pos = paper_dict[pid_pos]
                self.pos_embs.append(emb_pos)

                emb_neg = paper_dict[pid_neg]
                self.neg_embs.append(emb_neg)
                if len(self.anchor_embs) == self.batch_size:
                    dump_data(self.anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
                    dump_data(self.pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
                    dump_data(self.neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
                    f_idx += 1
                    self.anchor_embs = []
                    self.pos_embs = []
                    self.neg_embs = []


                n_sample_triplets+=1
                if n_sample_triplets % 100 == 0:
                        print('得到的嵌入三运组', n_sample_triplets)


        if self.anchor_embs:
            dump_data(self.anchor_embs, out_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
            dump_data(self.pos_embs, out_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
            dump_data(self.neg_embs, out_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))

        print('dumped')


if __name__ == '__main__':
    # data_gen = TripletsGenerator(train_scale=100000,feature_style= "authors")
    # data_gen.dump_triplets(role='train')
    # data_gen.dump_triplets(role='test')
    #
    # data_gen = TripletsGenerator(train_scale=100000,feature_style= 'title')
    # data_gen.dump_triplets(role='train')
    # data_gen.dump_triplets(role='test')

    # data_gen = TripletsGenerator(train_scale=100000,feature_style= "abstract")
    # data_gen.dump_triplets(role='train')
    # data_gen.dump_triplets(role='test')
    # #
    # data_gen = TripletsGenerator(train_scale=100000,feature_style= "keywords")
    # data_gen.dump_triplets(role='train')
    # data_gen.dump_triplets(role='test')
    # #
    # data_gen = TripletsGenerator(train_scale=100000,feature_style= "venue")
    # data_gen.dump_triplets(role='train')
    # data_gen.dump_triplets(role='test')
    # #
    # data_gen = TripletsGenerator(train_scale=100000,feature_style= "author_org")
    # data_gen.dump_triplets(role='train')
    # data_gen.dump_triplets(role='test')


    data_gen = TripletsGenerator(train_scale=100000,feature_style= "all")
    data_gen.dump_triplets(role='train')
    data_gen.dump_triplets(role='test')

