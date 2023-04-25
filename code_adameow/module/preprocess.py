# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = adj.sum(axis=1)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm_coo = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().todense()

    adj_torch = torch.from_numpy(adj_norm_coo).float()
    if torch.cuda.is_available():
        adj_torch = adj_torch.cuda()
    return adj_torch

def Count(feats,num):
    count = 0
    for i in range(feats.size()[0]):
        for j in range(feats.size()[1]):
            if feats[i][j] == num:
                count = count + 1
    return count

def pathsim(adjs, max_nei):
    print("the number of edges:", [adj.getnnz() for adj in adjs])
    top_adjs = []
    adjs_num = []
    for t in range(len(adjs)):
        A = adjs[t].todense()
        value = []
        x,y = A.nonzero()
        for i,j in zip(x,y):
            value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
        pathsim_matrix = sp.coo_matrix((value, (x, y)), shape=A.shape).toarray()
        idx_x = np.array([np.ones(max_nei[t])*i for i in range(A.shape[0])], dtype=np.int32).flatten()
        idx_y = np.sort(np.argsort(pathsim_matrix, axis=1)[:,::-1][:,0:max_nei[t]]).flatten()
        new = []
        for i,j in zip(idx_x,idx_y):
            new.append(A[i,j])
        new = (np.int32(np.array(new)))
        adj_new = sp.coo_matrix((new, (idx_x,idx_y)), shape=adjs[t].shape)
        adj_num = np.array(new).nonzero()
        adjs_num.append(adj_num[0].shape[0])
        top_adjs.append(adj_new)
    print("the top-k number of edges:", [adj for adj in adjs_num])
    return top_adjs

def spcoo_to_torchcoo(adj):
    values = adj.data
    indices = np.vstack((adj.row ,adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    adj_torchcoo = torch.sparse_coo_tensor(i, v, adj.shape)
    return adj_torchcoo


def mask_edges(adjs, sub_num, adj_mask):
    #print("the number of edges:", [adj.values for adj in adjs])
    mask_adjs = []
    for i in range(sub_num):    
        adj = adjs[i]
        adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
        adj.eliminate_zeros()
        adj_tuple = sparse_to_tuple(adj)   
        edges = adj_tuple[0]  
        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(edges)
        mask_edges_num = int(adj_mask*len(edges))   # len(edges)=减去自环的边数
        rest_edges = edges[mask_edges_num:]
        data = np.ones(rest_edges.shape[0])
        adj = sp.coo_matrix((data, (rest_edges[:, 0], rest_edges[:, 1])), shape=adjs[i].shape)
        adj = (adj + np.eye(adjs[i].shape[0]))
        adj = normalize_adj(adj)
        mask_adjs.append(adj)

      #  print("the number of mask edges:", [mask_adj.getnnz() for mask_adj in mask_adjs])

    return mask_adjs

def mask_feature(feat, adj_mask):
    feats_coo = sp.coo_matrix(feat)
    feats_num = feats_coo.getnnz()
    feats_idx = [i for i in range(feats_num)]
    mask_num = int(feats_num * adj_mask)
    mask_idx = random.sample(feats_idx, mask_num)
    feats_data = feats_coo.data
    for j in mask_idx:
        feats_data[j] = 0
    mask_feat = torch.sparse.FloatTensor(torch.LongTensor([feats_coo.row.tolist(), feats_coo.col.tolist()]),
                          torch.FloatTensor(feats_data.astype(np.float)))

    if torch.cuda.is_available():
        mask_feat = mask_feat.cuda()
    return mask_feat

def mask_features(feats, adj_mask):
    if len(feats.size()) == 3:
        mask_feats = []
        for feat in feats:
            mask_feats.append(mask_feature(feat,adj_mask))
        if torch.cuda.is_available():
            mask_feats =  [f.cuda() for f in mask_feats]
    else:
        mask_feats = mask_feature(feats, adj_mask)
        if torch.cuda.is_available():
            mask_feats = mask_feats.cuda() 
    return mask_feats

def txt_to_coo(dataset, relation):
    txt = np.genfromtxt("../data/" + dataset + "/" + relation + ".txt")
    row = txt[:,0]
    col = txt[:,1]
    data = np.ones(len(row))
    return torch.sparse.FloatTensor(torch.LongTensor([row.tolist(), col.tolist()]),
                            torch.FloatTensor(data.astype(np.float))).to_dense()