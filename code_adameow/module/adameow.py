# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .preprocess import *
from .encoder import *

class AdaMEOW(nn.Module):
    def __init__(self, feats_dim_list, sub_num, hidden_dim, embed_dim, tau, adjs, dropout, nei_index, feat, dataset):
        super(AdaMEOW, self).__init__()
        self.sub_num = sub_num
        self.tau = tau
        self.dataset = dataset
        self.adj = torch.stack(adjs).mean(axis=0)
        self.nei_index = [i.cuda() for i in nei_index]
        self.nei_index_count = [i.to_dense().sum(axis=1).unsqueeze(axis=1).reshape(adjs[0].size(0),1) for i in self.nei_index]
        self.nei_index_count = [torch.where(i >0, i, 1) for i in self.nei_index_count]
        
        if dataset == 'dblp':
            agg_num = 7
        else:
            agg_num = sub_num

        if dataset == 'aminer':
            self.p_fc_list = nn.ModuleList([nn.Linear(feats_dim_list[0], hidden_dim, bias=True)
                                      for _ in range(sub_num)])

        self.dropout = nn.Dropout(dropout)
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        self.agg = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False)
                                      for _ in range(agg_num)])
        self.Encoder = GCN(hidden_dim, embed_dim, embed_dim, dropout)
        self.att = Attention(embed_dim, dropout)

        if self.dataset == 'aminer':
            proj_dim = 16
        else:
            proj_dim = embed_dim

        self.project = nn.Linear(embed_dim, proj_dim)
        
        self.MLP = nn.Sequential(
                                nn.Linear(proj_dim, 16),
                                 nn.Tanh(),
                                 nn.Linear(16, 1),
                                 nn.Sigmoid(),
                                )

        
    def forward(self, feats, mask_feat, adjs, mask_adjs):
        # --------- Feature transform & Agggregate neighbor -------
        if self.dataset == 'acm':
            h_mp_embed, h_mp_mask_embed, h_coarse = self.aggregate_nei_acm(feats, mask_feat)
        elif self.dataset == 'dblp':
            h_mp_embed, h_mp_mask_embed, h_coarse = self.aggregate_nei_dblp(feats, mask_feat)
        elif self.dataset == 'aminer':
            h_mp_embed, h_mp_mask_embed, h_coarse = self.aggregate_nei_aminer(feats, mask_feat)
        elif self.dataset == 'imdb':
            h_mp_embed, h_mp_mask_embed, h_coarse = self.aggregate_nei_acm(feats, mask_feat)

        # ------------- Coarse view ------------
        z_coarse = self.Encoder(h_coarse, self.adj)
        # -------- Fine-grained view -----------
        h_fine  = []
        for i in range(self.sub_num):
            h_fine.append(self.Encoder(h_mp_embed[i], adjs[i]))
            h_fine.append(self.Encoder(h_mp_mask_embed[i], mask_adjs[i]))
        h_fine = [F.normalize(h, dim=1) for h in h_fine]
        z_fine = self.att(h_fine)
        self.z = z_fine
        # --------- Projection -------------
        z_coarse = F.tanh(self.project(z_coarse))
        z_coarse = F.normalize(z_coarse, dim=1)
        z_fine = F.tanh(self.project(z_fine))
        z_fine = F.normalize(z_fine, dim=1)
        
        loss = self.weight_InfoNce(z_fine, z_coarse)

        return loss
    
    def aggregate_nei_acm(self, feats, mask_feat=None):

        h_tar = F.elu(self.dropout(self.fc_list[0](feats[0])))
        h_nei = []
        for i in range(1,len(feats)):
            h_nei.append(F.elu(self.dropout(self.fc_list[i](feats[i]))))
        h_view_embed = []
        for i in range(len(self.nei_index)):
            h_nei_agg = torch.mm(self.nei_index[i], h_nei[i]) / self.nei_index_count[i]
            h_view_embed.append(F.elu(h_tar + self.dropout(self.agg[i](h_nei_agg))))
        
        if mask_feat is not None:
            h_mask = F.elu(self.dropout(self.fc_list[0](mask_feat)))
            h_mask_embed = []
            for i in range(len(self.nei_index)):
                h_nei_agg = torch.mm(self.nei_index[i], h_nei[i]) / self.nei_index_count[i]
                h_mask_embed.append(F.elu(h_mask + self.dropout(self.agg[i](h_nei_agg))))
            return h_view_embed, h_mask_embed, h_tar
        else:
            return h_view_embed, h_tar

    def aggregate_nei_dblp(self, feats, mask_feat):
        nei_ap, nei_apc, nei_apcp, nei_apt, nei_aptp = self.nei_index
        
        [feat_a, feat_p, feat_t, feat_c] = [F.elu(self.dropout(self.fc_list[i](feats[i]))) for i in range(len(feats))]
        feat_a_mask = F.elu(self.fc_list[0](mask_feat))

        # apa 
        h_ap = torch.mm(nei_ap, feat_p) / self.nei_index_count[0]
        agg_apa = self.dropout(self.agg[0](h_ap))
        h_apa = feat_a + agg_apa
        mask_apa = feat_a_mask + agg_apa

        # apcpa
        h_apc = torch.mm(nei_apc, feat_c) / self.nei_index_count[1]
        h_apcp = torch.mm(nei_apcp, feat_p) / self.nei_index_count[2]
        agg_apcpa = self.dropout(self.agg[1](h_ap)) + self.dropout(self.agg[2](h_apc)) + self.dropout(self.agg[3](h_apcp))
        h_apcpa = feat_a + agg_apcpa
        mask_apcpa = feat_a_mask + agg_apcpa
        
        # aptpa
        h_apt = torch.mm(nei_apt, feat_t) / self.nei_index_count[3]
        h_aptp = torch.mm(nei_aptp, feat_p) / self.nei_index_count[4]
        agg_aptpa = self.dropout(self.agg[4](h_ap)) + self.dropout(self.agg[5](h_apt)) + self.dropout(self.agg[6](h_aptp))
        h_aptpa = feat_a + agg_aptpa
        mask_aptpa = feat_a_mask + agg_aptpa

        h_view_embed = [h_apa, h_apcpa, h_aptpa]
        h_mask_embed = [mask_apa, mask_apcpa, mask_aptpa]
        h_view_embed = [F.elu(i) for i in h_view_embed]
        h_mask_embed = [F.elu(i) for i in h_mask_embed]
        
        return h_view_embed, h_mask_embed, feat_a

    def aggregate_nei_aminer(self, feats, mask_feat):
        h_tar = [F.elu(self.dropout(self.p_fc_list[i](feats[0][i]))) for i in range(len(self.nei_index))]
        h_mask = [F.elu(self.dropout(self.p_fc_list[i](mask_feat[i]))) for i in range(len(self.nei_index))]
        h_nei = []
        for i in range(1,len(feats)):
            h_nei.append(F.elu(self.dropout(self.fc_list[i-1](feats[i]))))
        h_view_embed = [] 
        h_mask_embed = []
        for i in range(len(self.nei_index)):
            h_nei_agg = torch.mm(self.nei_index[i], h_nei[i]) / self.nei_index_count[i]
            h_view_embed.append(F.elu(h_tar[i] + self.dropout(self.agg[i](h_nei_agg))))
            h_mask_embed.append(F.elu(h_mask[i] + self.dropout(self.agg[i](h_nei_agg))))
            
        h_coarse = torch.stack(h_tar).mean(axis=0)
        return h_view_embed, h_mask_embed, h_coarse


    def weight_InfoNce(self, z_anchor, z):
        N = z.size(0)

        dots = torch.exp(torch.mm(z_anchor, z.t()) / self.tau)
        nominator = torch.diag(dots)

        w_anchor = torch.stack([z_anchor[i].unsqueeze(0).repeat(N,1) for i in range(N)]).reshape(N*N,-1)
        w = z.unsqueeze(0).repeat(N,1,1).reshape(N*N,-1)
        weight = w_anchor + w
        weight = self.MLP(weight).reshape(N,N)

        neg = torch.mul(dots,weight)

        denominator = neg.sum(axis=1)
        loss = ((-1) * (torch.log(nominator / denominator))).mean()
        
        return loss

    def get_embeds(self):
        return self.z.detach()