


from os.path import join
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


seed = 42
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Input, Lambda,BatchNormalization
from keras.optimizers import Adam
from src.utils.utils import full_auc
from src.utils.utils import load_data,load_json,dump_data
from src.utils.utils import get_hidden_output,LMDBClient


DATASET_DIR = "../../datasets/OAG-WhoisWh0-na-v1"
OUT_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/global_embedding"
IN_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_embedding"
PRE_DATA_DIR = "../../datas/OAG-WhoisWh0-na-v1/pre_data"
# 导入数据
# valid_data = np.load(IN_DATA_DIR+"/valid_pub_pre_embedding_npy.npy",allow_pickle=True)
valid_data_npy = np.load(IN_DATA_DIR+"/test_pub_pre_embedding_npy.npy",allow_pickle=True).item()

# train_author_name_paper_ids = pd.read_pickle(PRE_DATA_DIR+'/train_author_name_paper_ids.pkl')
# valid_author_name_paper_ids = pd.read_pickle(PRE_DATA_DIR+'/valid_author_name_paper_ids.pkl')
# train_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/train_author_name_ids.pkl')
# valid_author_name_ids = pd.read_pickle(PRE_DATA_DIR+'/valid_author_name_ids.pkl')
EMB_DIM = 100
"""
global metric learning model
"""

def l2Norm(x):
    return K.l2_normalize(x, axis=-1)



def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def triplet_loss(_, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0,0]) - K.square(y_pred[:,1,0]) + margin))


def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] < y_pred[:,1,0])




class GlobalTripletModel:

    def __init__(self, data_scale,f_style):
        self.f_style = f_style
        self.data_scale = data_scale
        self.o_dir = OUT_DATA_DIR+"/"+f_style
        self.train_triplets_dir = self.o_dir+'/triplets-{}'.format(self.data_scale)
        self.test_triplets_dir = self.o_dir+'/test-triplets'
        self.train_triplet_files_num = self.get_triplets_files_num(self.train_triplets_dir)
        self.test_triplet_files_num = self.get_triplets_files_num(self.test_triplets_dir)
        print('train file num', self.train_triplet_files_num)
        print('test file num', self.test_triplet_files_num)

    @staticmethod
    def get_triplets_files_num(path_dir):
        files = []
        for f in os.listdir(path_dir):
            if f.startswith('anchor_embs_'):
                files.append(f)
        return len(files)

    def load_batch_triplets(self, f_idx, role='train'):
        if role == 'train':
            cur_dir = self.train_triplets_dir
        else:
            cur_dir = self.test_triplets_dir
        X1 = load_data(cur_dir, 'anchor_embs_{}_{}.pkl'.format(role, f_idx))
        X2 = load_data(cur_dir, 'pos_embs_{}_{}.pkl'.format(role, f_idx))
        X3 = load_data(cur_dir, 'neg_embs_{}_{}.pkl'.format(role, f_idx))
        return X1, X2, X3

    def load_triplets_data(self, role='train'):
        X1 = np.empty([0, EMB_DIM])
        X2 = np.empty([0, EMB_DIM])
        X3 = np.empty([0, EMB_DIM])
        if role == 'train':
            f_num = self.train_triplet_files_num
        else:
            f_num = self.test_triplet_files_num
        for i in range(f_num):
            print('load', i)
            x1_batch, x2_batch, x3_batch = self.load_batch_triplets(i, role)
            p = np.random.permutation(len(x1_batch))
            x1_batch = np.array(x1_batch)[p]
            x2_batch = np.array(x2_batch)[p]
            x3_batch = np.array(x3_batch)[p]
            X1 = np.concatenate((X1, x1_batch))
            X2 = np.concatenate((X2, x2_batch))
            X3 = np.concatenate((X3, x3_batch))
        return X1, X2, X3

    @staticmethod
    def create_triplet_model():
        emb_anchor = Input(shape=(EMB_DIM, ), name='anchor_input')
        emb_pos = Input(shape=(EMB_DIM, ), name='pos_input')
        emb_neg = Input(shape=(EMB_DIM, ), name='neg_input')

        # shared layers
        layer1 = Dense(EMB_DIM,activation='relu',  name='first_emb_layer')
        layer2 = Dense(EMB_DIM,activation='relu',  name='last_emb_layer')
        norm_layer = Lambda(l2Norm, name='norm_layer', output_shape=[EMB_DIM])
        # norm_layer = BatchNormalization()

        encoded_emb = norm_layer(layer2(layer1(emb_anchor)))
        encoded_emb_pos = norm_layer(layer2(layer1(emb_pos)))
        encoded_emb_neg = norm_layer(layer2(layer1(emb_neg)))


        pos_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_emb, encoded_emb_pos])
        neg_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_emb, encoded_emb_neg])

        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
        )([pos_dist, neg_dist])

        model = Model([emb_anchor, emb_pos, emb_neg], stacked_dists, name='triple_siamese')
        model.compile(loss=triplet_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])
        inter_layer = 0

        return model, inter_layer

    def load_triplets_model(self):
        model_dir = join(self.o_dir, 'model')
        rf = open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'r')
        model_json = rf.read()
        rf.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))
        return loaded_model

    def get_global_embdding(self):
        trained_global_model = self.load_triplets_model()
        print('triplets model loaded')
        test_triplets = self.load_triplets_data(role='test')
        auc_score = full_auc(trained_global_model, test_triplets)
        print('AUC', auc_score)

        embs_input = []
        for paper in valid_data_npy:
            cur_emb = valid_data_npy[paper][self.f_style]
            if cur_emb is None:
                continue
            embs_input.append(cur_emb)
        embs_input = np.stack(embs_input)
        inter_embs = get_hidden_output(trained_global_model, embs_input)
        return inter_embs


    def train_triplets_model(self):
        X1, X2, X3 = self.load_triplets_data()
        n_triplets = len(X1)
        print('loaded')
        model, inter_model = self.create_triplet_model()
        # print(model.summary())

        X_anchor, X_pos, X_neg = X1, X2, X3
        X = {'anchor_input': X_anchor, 'pos_input': X_pos, 'neg_input': X_neg}
        model.fit(X, np.ones((n_triplets, 2)), batch_size=128, epochs=5, shuffle=True, validation_split=0.2)

        model_json = model.to_json()
        model_dir = join(self.o_dir, 'model')
        os.makedirs(model_dir, exist_ok=True)
        with open(join(model_dir, 'model-triplets-{}.json'.format(self.data_scale)), 'w') as wf:
            wf.write(model_json)
        model.save_weights(join(model_dir, 'model-triplets-{}.h5'.format(self.data_scale)))

        # test_triplets = self.load_triplets_data(role='test')
        # auc_score = full_auc(model, test_triplets)
        # print('AUC', auc_score)

    def evaluate_triplet_model(self):
        test_triplets = self.load_triplets_data(role='test')
        loaded_model = self.load_triplets_model()
        print('triplets model loaded')
        auc_score = full_auc(loaded_model, test_triplets)

if __name__ == '__main__':
    global_embedding = {}
    # global_model = GlobalTripletModel(data_scale=100000, f_style='all')
    # global_model.train_triplets_model()
    cols = ['authors','title', 'abstract',  'venue', 'author_org']
    for clo in cols:
        print(clo)
        set_seed(seed)
        global_model = GlobalTripletModel(data_scale=100000, f_style=clo)
        global_model.train_triplets_model()
        glo_emb = global_model.get_global_embdding()
        global_embedding[clo] = glo_emb
        print(clo + ' done!')
    np.save(OUT_DATA_DIR+"/global_embedding_npy.npy",global_embedding)

    #


