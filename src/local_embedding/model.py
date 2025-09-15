import torch
import torch.nn as nn
import torch.nn.functional as F
from    layer import ASGC,SampleDecoder,GetWeigit,GAT

# from layers import *


class LocalModel(nn.Module):
    def __init__(self, dims):
        super(LocalModel, self).__init__()
        # self.layers = nn.ModuleList()
        self.dcs = SampleDecoder(act=lambda x: x)
        self.getweight = GetWeigit(act=lambda x: x)
        self.input_dim = dims[0]
        self.output_dim = dims[1]
        #
        self.AdaptGcn_relation_authors = ASGC(self.input_dim,self.output_dim)
        self.AdaptGcn_relation_abstract = ASGC(self.input_dim,self.output_dim)
        self.AdaptGcn_relation_title = ASGC(self.input_dim,self.output_dim)
        # self.AdaptGcn_relation_key_words = ASGC(self.input_dim,self.output_dim)
        self.AdaptGcn_relation_orgs = ASGC(self.input_dim,self.output_dim)
        self.AdaptGcn_relation_venue = ASGC(self.input_dim,self.output_dim)
        #
        self.layers_relation = nn.Linear(self.output_dim,1,bias=True)
    def scale(self, z):

        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
        return z_scaled

    def forward(self, x_authors, x_abstract, x_title,x_author_org,x_venue,hin_embedding):

        out_authors = self.AdaptGcn_relation_authors(x_authors)
        out_abstract = self.AdaptGcn_relation_abstract(x_abstract)
        out_title = self.AdaptGcn_relation_title(x_title)
        out_orgs = self.AdaptGcn_relation_orgs(x_author_org)
        out_venue = self.AdaptGcn_relation_venue(x_venue)

        H = torch.stack([out_authors,out_abstract,out_title,out_orgs,out_venue,hin_embedding]).cuda()
        # H = torch.stack([out_authors,out_abstract,out_title,out_orgs,out_venue]).cuda()
        X = self.layers_relation(H)
        X = F.sigmoid(X)

        X = torch.permute(X,(1,0,2))
        # X = F.normalize(X,dim=1)
        S = F.softmax(X,dim=1)
        out_H = torch.permute(H,(1,2,0))
        out = torch.bmm(out_H,S).squeeze()
        # out = out_authors+out_abstract+out_title+out_orgs+out_venue


        # out = F.softmax(out)
        out = self.scale(out)
        out = F.normalize(out)
        return out


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class LinTrans(nn.Module):
    def __init__(self, layers, dims):
        super(LinTrans, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[ i+1]))
        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):

        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std

        return z_scaled

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.scale(out)
        out = F.normalize(out)
        return out
