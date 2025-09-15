# utils_v1 使用word2vec预训练向量作为文本embedding
import codecs
import json
from os.path import join
import pickle
import os
import re

import pandas as pd
# import torch
from gensim.models import word2vec
import numpy as np
from keras import backend as K
# import lmdb
from sklearn.metrics import roc_auc_score
################# Load and Save Data ################

def load_json(rfdir, rfname):
    with codecs.open(join(rfdir, rfname), 'r', encoding='utf-8') as rf:
        return json.load(rf)


def dump_json(obj, wfpath, wfname, indent=None):
    with codecs.open(join(wfpath, wfname), 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)



def dump_data(obj, wfpath, wfname):
    with open(os.path.join(wfpath, wfname), 'wb') as wf:
        pickle.dump(obj, wf)

def sparse_dropout(x, rate, noise_shape):
    """

    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out


def load_data(rfpath, rfname):
    with open(os.path.join(rfpath, rfname), 'rb') as rf:
        return pickle.load(rf)


def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def cal_f1(prec, rec):
    return 2*prec*rec/(prec+rec)


def get_hidden_output(model, inp):
    get_activations = K.function(model.inputs[:1] + [K.learning_phase()], [model.layers[5].get_output_at(0), ])
    print(model.layers)
    activations = get_activations([inp, 0])
    return activations[0]


def predict(anchor_emb, test_embs):
    score1 = np.linalg.norm(anchor_emb-test_embs[0])
    score2 = np.linalg.norm(anchor_emb-test_embs[1])
    return [score1, score2]


def full_auc(model, test_triplets):
    """
    Measure AUC for model and ground truth on all items.

    Returns:
    - float AUC
    """

    grnds = []
    preds = []
    preds_before = []
    embs_anchor, embs_pos, embs_neg = test_triplets

    inter_embs_anchor = get_hidden_output(model, embs_anchor)
    inter_embs_pos = get_hidden_output(model, embs_pos)
    inter_embs_neg = get_hidden_output(model, embs_neg)
    print(inter_embs_anchor)

    accs = []
    accs_before = []

    for i, e in enumerate(inter_embs_anchor):
        if i % 10000 == 0:
            print('test', i)

        emb_anchor = e
        emb_pos = inter_embs_pos[i]
        emb_neg = inter_embs_neg[i]
        test_embs = np.array([emb_pos, emb_neg])

        emb_anchor_before = embs_anchor[i]
        emb_pos_before = embs_pos[i]
        emb_neg_before = embs_neg[i]
        test_embs_before = np.array([emb_pos_before, emb_neg_before])

        predictions = predict(emb_anchor, test_embs)
        predictions_before = predict(emb_anchor_before, test_embs_before)

        acc_before = 1 if predictions_before[0] < predictions_before[1] else 0
        acc = 1 if predictions[0] < predictions[1] else 0
        accs_before.append(acc_before)
        accs.append(acc)

        grnd = [0, 1]
        grnds += grnd
        preds += predictions
        preds_before += predictions_before

    auc_before = roc_auc_score(grnds, preds_before)
    auc = roc_auc_score(grnds, preds)
    print('test accuracy before', np.mean(accs_before))
    print('test accuracy after', np.mean(accs))

    print('test AUC before', auc_before)
    print('test AUC after', auc)
    return auc


################# Compare Lists ################

def tanimoto(p,q):
    c = [v for v in p if v in q]
    return float(len(c) / (len(p) + len(q) - len(c)))



################# Paper similarity ################

def generate_pair(pubs,outlier,valid_data,name): ##求匹配相似度
    
    paper_org = {}
    paper_conf = {}
    paper_author = {}
    paper_word = {}
    cols = ['paper_id','title', 'venue', 'author_org','authors','abstract', 'keywords',]
    local_data = pd.DataFrame()
    local_data["paper_id"]= valid_data["paper_id"].loc[valid_data["author_name"] == name]
    for col in cols:
        local_data[col]= valid_data[col].loc[valid_data["author_name"] == name]
    local_data = local_data.reset_index()

    for i in range(len(pubs)):
        id = local_data.iloc[i,1]
        title = local_data.iloc[i,2].split()
        venue = local_data.iloc[i,3].split()
        org = local_data.iloc[i,4].split()
        author = local_data.iloc[i,5].split()
        for word in title:
            if id not in paper_word:
                paper_word[id] = []
            paper_word[id].append(word)
        for word in venue:
            if id not in paper_conf:
                paper_conf[id] = []
            paper_conf[id].append(word)
        for word in org:
            if id not in paper_org:
                paper_org[id] = []
            paper_org[id].append(word)
        for word in author:
            if id not in paper_author:
                paper_author[id] = []
            paper_author[id].append(word)
    
    paper_paper = np.zeros((len(pubs),len(pubs)))
    for i,pid in enumerate(pubs):
        if i not in outlier:
            continue
        for j,pjd in enumerate(pubs):
            if j==i:
                continue
            ca=0
            cv=0
            co=0
            ct=0
          
            if pid in paper_author and pjd in paper_author:
                ca = len(set(paper_author[pid]) & set(paper_author[pjd])) * 1.5
            if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
                cv = tanimoto(set(paper_conf[pid]), set(paper_conf[pjd]))
            if pid in paper_org and pjd in paper_org:
                co = tanimoto(set(paper_org[pid]), set(paper_org[pjd]))
            if pid in paper_word and pjd in paper_word:
                ct = len(set(paper_word[pid]) & set(paper_word[pjd])) / 3
            
            # 有共同合作作者
            # if pid in paper_author and pjd in paper_author:
            #     ca = len(set(paper_author[pid])&set(paper_author[pjd]))*1.5
            # # venue相似度
            # if pid in paper_conf and pjd in paper_conf and 'null' not in paper_conf[pid]:
            #     cv = len(set(paper_conf[pid])&set(paper_conf[pjd]))*0.2
            # # 机构相似度
            # if pid in paper_org and pjd in paper_org:
            #     co = len(set(paper_org[pid])&set(paper_org[pjd]))*0.5
            # # 题目关键词相似度
            # if pid in paper_word and pjd in paper_word:
            #     ct = len(set(paper_word[pid])&set(paper_word[pjd]))*0.3
                    
            paper_paper[i][j] =ca+cv+co+ct
            
    return paper_paper

def deserialize_embedding(s):
    return pickle.loads(s)

class LMDBClient(object):

    def __init__(self, dir,name, readonly=False):
        try:
            lmdb_dir = join(dir, 'lmdb')
            print(join(lmdb_dir, name))
            os.makedirs(lmdb_dir, exist_ok=True)
            self.db = lmdb.open(join(lmdb_dir, name), map_size=109951162777, readonly=readonly)
        except Exception as e:
            print("2131",e)

    def get(self, key):
        with self.db.begin() as txn:
            value = txn.get(key.encode())
        if value:
            return deserialize_embedding(value)
        else:
            return None

    def get_batch(self, keys):
        values = []
        with self.db.begin() as txn:
            for key in keys:
                value = txn.get(key.encode())
                if value:
                    values.append(deserialize_embedding(value))
        return values

    def set(self, key, vector):
        with self.db.begin(write=True) as txn:
            txn.put(key.encode("utf-8"), deserialize_embedding(vector))

    def set_batch(self, generator):
        with self.db.begin(write=True) as txn:
            for key, vector in generator:
                txn.put(key.encode("utf-8"), serialize_embedding(vector))
                print(key, self.get(key))

        
################# Evaluate ################
def serialize_embedding(embedding):
    return pickle.dumps(embedding)

def pairwise_evaluate(correct_labels,pred_labels):
    TP = 0.0  # Pairs Correctly Predicted To SameAuthor
    TP_FP = 0.0  # Total Pairs Predicted To SameAuthor
    TP_FN = 0.0  # Total Pairs To SameAuthor

    for i in range(len(correct_labels)):
        for j in range(i + 1, len(correct_labels)):
            if correct_labels[i] == correct_labels[j]:
                TP_FN += 1
            if pred_labels[i] == pred_labels[j]:
                TP_FP += 1
            if (correct_labels[i] == correct_labels[j]) and (pred_labels[i] == pred_labels[j]):
                TP += 1

    if TP == 0:
        pairwise_precision = 0
        pairwise_recall = 0
        pairwise_f1 = 0
    else:
        pairwise_precision = TP / TP_FP
        pairwise_recall = TP / TP_FN
        pairwise_f1 = (2 * pairwise_precision * pairwise_recall) / (pairwise_precision + pairwise_recall)
    return pairwise_precision, pairwise_recall, pairwise_f1


################# Save Paper Features ################

def save_relation(name_pubs_raw, name): # 保存论文的各种feature
    name_pubs_raw = load_json('genename', name_pubs_raw)
    # name_pubs_raw下存储了当前name下所有论文的信息
    ## trained by all text in the datasets. Training code is in the cells of "train word2vec"
    save_model_name = "word2vec/Aword2vec.model"
    model_w = word2vec.Word2Vec.load(save_model_name)
    
    r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～’]+'
    stopword = ['at','based','in','of','for','on','and','to','an','using','with','the','by','we','be','is','are','can']
    stopword1 = ['university','univ','china','department','dept','laboratory','lab','school','al','et',
                 'institute','inst','college','chinese','beijing','journal','science','international','arxiv']
    org_stopword = ['at','based','in','of','for','on','and','to','using','with','the','by','we','be','is','are','can',
                    'university','univ','china','department','dept','laboratory','lab','school','al','et',
                 'institute','inst','college','beijing']
                    #'university','univ','china','department','dept','institute','inst','laboratory','lab','technology',
                    #'al','et','science','sciences','school','chinese','college']
    # org_stopword = ['at','based','in','of','for','on','and','to','using','with','the','by','we','be','is','are','can','al','et']


    f1 = open ('gene/paper_author.txt','w',encoding = 'utf-8')
    f2 = open ('gene/paper_conf.txt','w',encoding = 'utf-8')
    f3 = open ('gene/paper_word.txt','w',encoding = 'utf-8')
    f4 = open ('gene/paper_org.txt','w',encoding = 'utf-8')


    
    taken = name.split("_")
    name = taken[0] + taken[1]
    name_reverse = taken[1]  + taken[0]
    if len(taken)>2:
        name = taken[0] + taken[1] + taken[2]
        name_reverse = taken[2]  + taken[0] + taken[1]
    
    authorname_dict={}
    ptext_emb = {}  
    
    tcp=set()  
    for i,pid in enumerate(name_pubs_raw):
        # pid为paper id
        pub = name_pubs_raw[pid]
        
        #save authors
        org=""
        for author in pub["authors"]:
            authorname = re.sub(r,'', author["name"]).lower()
            taken = authorname.split(" ")
            if len(taken)==2: ##检测目前作者名是否在作者词典中
                authorname = taken[0] + taken[1]
                authorname_reverse = taken[1]  + taken[0] 
            
                if authorname not in authorname_dict:
                    if authorname_reverse not in authorname_dict:
                        authorname_dict[authorname]=1
                    else:
                        authorname = authorname_reverse 
            else:
                authorname = authorname.replace(" ","")
            
            if authorname!=name and authorname!=name_reverse:
                # 如果这个作者不是当前的分析作者
                f1.write(pid + '\t' + authorname + '\n')
        
            else:
                # 如果这个是当前分析的作者 就记录org信息
                if "org" in author:
                    org = author["org"]
                    
                    
        #save org 待消歧作者的机构名
        pstr = org.strip()
        pstr = pstr.lower() #小写
        pstr = re.sub(r,' ', pstr) #去除符号
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip() #去除多余空格
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>1]
        pstr = [word for word in pstr if word not in org_stopword]
        #pstr = [word for word in pstr if word not in stopword]
        #pstr = [word for word in pstr if word not in stopword1]
        pstr=set(pstr)
        for word in pstr:
            f4.write(pid + '\t' + word + '\n')

        
        #save venue
        pstr = pub["venue"].strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>1]
        pstr = [word for word in pstr if word not in stopword1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f2.write(pid + '\t' + word + '\n')
        if len(pstr)==0:
            f2.write(pid + '\t' + 'null' + '\n')

            
        #save text 关键词和题目
        pstr = ""    
        keyword=""
        if "keywords" in pub:
            for word in pub["keywords"]:
                keyword=keyword+word+" "
        pstr = pstr + pub["title"]
        pstr=pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>1]
        pstr = [word for word in pstr if word not in stopword]
        for word in pstr:
            f3.write(pid + '\t' + word + '\n')
        
        #save all words' embedding
        # pstr = [关键词 题目 地点 机构 年份]
        pstr = keyword + " " + pub["title"] + " " + pub["venue"] + " " + org
        if "year" in pub:
              pstr = pstr +  " " + str(pub["year"])
        pstr=pstr.strip()
        pstr = pstr.lower()
        pstr = re.sub(r,' ', pstr)
        pstr = re.sub(r'\s{2,}', ' ', pstr).strip()
        pstr = pstr.split(' ')
        pstr = [word for word in pstr if len(word)>1]
        pstr = [word for word in pstr if word not in stopword]
        pstr = [word for word in pstr if word not in stopword1]
        #print (pstr)

        words_vec=[]
        for word in pstr:
            if (word in model_w):
                words_vec.append(model_w[word])
                # words_vec = [[word1],[word2],[word3],[word4],...]表示当前文章的embedding
        if len(words_vec)<1:
            words_vec.append(np.zeros(100))
            tcp.add(i)
            #print ('outlier:',pid,pstr)
        ptext_emb[pid] = np.mean(words_vec,0)
        
    #  ptext_emb: key is paper id, and the value is the paper's text embedding
    dump_data(ptext_emb,'gene','ptext_emb.pkl')
    # the paper index that lack text information
    dump_data(tcp,'gene','tcp.pkl')
            
 
    f1.close()
    f2.close()
    f3.close()
    f4.close()