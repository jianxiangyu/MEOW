# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:49:35 2022

@author: JasonYu
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, gc_drop):
        super(GraphConv, self).__init__()
        weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight = self.reset_parameters(weight)
        if gc_drop:
            self.gc_drop = nn.Dropout(gc_drop)
        else:
            self.gc_drop = lambda x: x
        self.act = nn.PReLU()

    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)
        return weight

    def glorot_init(self, input_dim, output_dim):
        init_range = np.sqrt(6.0/(input_dim + output_dim))
        initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
        return nn.Parameter(initial)

    def forward(self, x, adj, activation=None):
        x_hidden = self.gc_drop(torch.mm(x, self.weight))
        x = torch.spmm(adj, x_hidden)
        if activation is None:
            outputs = self.act(x)
        else:
            outputs = activation(x)
        return outputs


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(GCN, self).__init__()
        self.gc1 = GraphConv(input_dim, hidden_dim, dropout)
        self.gc2 = GraphConv(hidden_dim, output_dim, dropout)

    def forward(self, feat, adj, action=None):
        hidden = self.gc1(feat, adj)
        Z = self.gc2(hidden, adj, activation=lambda x: x)
        layernorm = nn.LayerNorm(Z.size(), eps=1e-05, elementwise_affine=False)
        outputs = layernorm(Z)
        return outputs

class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_drop=0):
        super(Attention, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

        self.tanh = nn.Tanh()
        self.att = nn.Parameter(torch.empty(
            size=(1, hidden_dim)), requires_grad=True)
        nn.init.xavier_normal_(self.att.data, gain=1.414)

        self.softmax = nn.Softmax()
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x

    def forward(self, embeds):
        beta = []
        attn_curr = self.attn_drop(self.att)
        for embed in embeds:
            sp = self.tanh(self.fc(embed)).mean(dim=0)
            beta.append(attn_curr.matmul(sp.t()))
            # matmul is attn_curr*sp.t() matrix mult   sp.t()=sp inverse
        beta = torch.cat(beta, dim=-1).view(-1)
        beta = self.softmax(beta)
        # print("the attention of type-level: ", beta.data.cpu().numpy())  # type-level attention
        z_mp = 0
        for i in range(len(embeds)):
            z_mp += embeds[i]*beta[i]
        return z_mp