from torch.nn import init
from tqdm import tqdm
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F
import torch

class RGCN(nn.Module):
    def __init__(self, in_features, hidden_features, n_classes):
        super(RGCN, self).__init__()
        self.conv1_1 = dglnn.SAGEConv(in_feats=in_features, out_feats=hidden_features, aggregator_type='gcn')
        self.conv1_2 = dglnn.SAGEConv(in_feats=hidden_features, out_feats=n_classes, aggregator_type='gcn')

        self.conv2_1 = dglnn.SAGEConv(in_feats=in_features, out_feats=hidden_features, aggregator_type='gcn')
        self.conv2_2 = dglnn.SAGEConv(in_feats=hidden_features, out_feats=n_classes, aggregator_type='gcn')

        self.conv3_1 = dglnn.SAGEConv(in_feats=in_features, out_feats=hidden_features, aggregator_type='gcn')
        self.conv3_2 = dglnn.SAGEConv(in_feats=hidden_features, out_feats=n_classes, aggregator_type='gcn')

        self.dropout = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(in_features)
        self.batch_norm2 = nn.BatchNorm1d(hidden_features)
        self.batch_norm3 = nn.BatchNorm1d(n_classes)
    def scale(self, z):

        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled
    def forward(self, graph1, graph2, graph3, x):
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x1_1 = self.conv1_1(graph1, x)
        x1_2 = self.conv2_1(graph2, x)
        x1_3 = self.conv3_1(graph3, x)
        x = F.relu(x1_1 + x1_2 + x1_3)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x2_1 = self.conv1_2(graph1, x)
        x2_2 = self.conv2_2(graph2, x)
        x2_3 = self.conv3_2(graph3, x)
        x = x2_1 + x2_2 + x2_3
        x = self.batch_norm3(x)

        out = F.sigmoid(x)
        out = self.scale(out)
        out = F.normalize(out)
        return out


# class SkipGramModel(nn.Module):
#
#     def __init__(self):
#         super(SkipGramModel, self).__init__()
#
#     def forward(self, pos_u, pos_v, neg_v, logits):
#         emb_u = logits[pos_u]
#         emb_v = logits[pos_v]
#         emb_neg_v = logits[neg_v]
#
#         score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
#         score = torch.clamp(score, max=10, min=-10)
#         score = -F.logsigmoid(score)
#
#         neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
#         neg_score = torch.clamp(neg_score, max=10, min=-10)
#         neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)
#
#         return torch.mean(score + neg_score), score, neg_score



# 单纯的metapath2vec
"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""
class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)