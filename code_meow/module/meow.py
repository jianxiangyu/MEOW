# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocess import *
from .encoder import *
from .cluster import *
from .contrast import *

class MEOW(nn.Module):
    def __init__(self, feats_dim_list, sub_num, hidden_dim, embed_dim, tau, adjs, lam_proto, \
                 dropout, nei_index, dataset):
        super(MEOW, self).__init__()
        self.sub_num = sub_num
        self.dataset= dataset
        self.tau = tau
        self.lam_proto = lam_proto
        self.adj = torch.stack(adjs).mean(axis=0)
        self.nei_index = [i.cuda() for i in nei_index]
        self.nei_index_count = [i.to_dense().sum(axis=1).unsqueeze(axis=1).reshape(adjs[0].size(0),1) for i in self.nei_index]
        self.nei_index_count = [torch.where(i >0, i, 1) for i in self.nei_index_count]

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = lambda x: x
        
        if dataset == 'dblp':
            agg_num = 7
        else:
            agg_num = sub_num
        if dataset == 'aminer':
            self.p_fc_list = nn.ModuleList([nn.Linear(feats_dim_list[0], hidden_dim, bias=True)
                                      for _ in range(sub_num)])
            
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True)
                                      for feats_dim in feats_dim_list])
        
        self.agg = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False)
                                      for _ in range(agg_num)])

        self.Encoder = GCN(hidden_dim, embed_dim, embed_dim, dropout)
        self.att = Attention(embed_dim, dropout)
        self.project = nn.Linear(embed_dim, embed_dim)

        
    def forward(self, feats, mask_feat, mask_adjs, adjs_norm, num_cluster):
        if self.dataset == 'acm':
            h_view_embed, h_mask_embed, h_coarse = self.aggregate_nei_acm(feats, mask_feat)
        elif self.dataset == 'dblp':
            h_view_embed, h_mask_embed, h_coarse = self.aggregate_nei_dblp(feats, mask_feat)
        elif self.dataset == 'aminer':
            h_view_embed, h_mask_embed, h_coarse = self.aggregate_nei_aminer(feats, mask_feat)
        elif self.dataset == 'imdb':
            h_view_embed, h_mask_embed, h_coarse = self.aggregate_nei_acm(feats, mask_feat)


        z_new = self.generate_loss(h_view_embed, h_mask_embed, mask_adjs, adjs_norm)
        loss_info, loss_proto = self.contrast_loss(h_coarse, z_new, num_cluster)
        loss = loss_info + self.lam_proto * loss_proto
        print('total loss: %f' % loss)
        return loss

    def contrast_loss(self, h_coarse, z_new, num_clusters):
        z_coarse = self.Encoder(h_coarse, self.adj)
        z_coarse = F.tanh(self.project(z_coarse))
        z_coarse = F.normalize(z_coarse, dim=1)
        
        z_new = [F.normalize(z_temp, dim=1) for z_temp in z_new]
        z = self.att(z_new)
        z_pro = F.tanh(self.project(z))
        z_pro = F.normalize(z_pro, dim=1)
    
        loss_info, loss_proto = Info_and_Proto(z_coarse, z_pro, num_clusters, self.tau)
        self.z_pos = z
        return loss_info , loss_proto

    def generate_loss(self, h_view_embed, h_mask_embed, mask_adjs, adjs_norm):
        z  = []
        for i in range(self.sub_num):
            z.append(self.Encoder(h_view_embed[i], adjs_norm[i]))
            z.append(self.Encoder(h_mask_embed[i], mask_adjs[i]))
        return z

    def aggregate_nei_acm(self, feats, mask_feat):
        h_tar = F.elu(self.dropout(self.fc_list[0](feats[0])))
        h_mask = F.elu(self.dropout(self.fc_list[0](mask_feat)))
        h_nei = []
        for i in range(1,len(feats)):
            h_nei.append(F.elu(self.dropout(self.fc_list[i](feats[i]))))
        h_view_embed = []
        h_mask_embed = []
        for i in range(len(self.nei_index)):
            h_nei_agg = torch.mm(self.nei_index[i], h_nei[i]) / self.nei_index_count[i]
            h_view_embed.append(F.elu(h_tar + self.dropout(self.agg[i](h_nei_agg))))
            h_mask_embed.append(F.elu(h_mask + self.dropout(self.agg[i](h_nei_agg))))
        return h_view_embed, h_mask_embed, h_tar

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

    def get_embeds(self):
        return self.z_pos.detach()
