import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder
 
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

def get_pos(pos):
    row = pos.row
    col = pos.col
    p = []
    flag = 0
    temp = []
    for i,j in zip(row,col):
        if flag!=i:
            temp = th.stack(temp)
            p.append(temp)
            temp = []
            flag = flag + 1
        temp.append(th.tensor(j).long())
    p.append(th.stack(temp))
    return p

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    # 4019 7167 60
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    
    nei_a  = sp.load_npz(path + "nei_a.npz")
    nei_s = sp.load_npz(path + "nei_s.npz")
    feat_p = sp.load_npz(path + "p_feat.npz")
    feat_a = sp.load_npz(path + "a_feat.npz")
    feat_s = sp.eye(type_num[2])
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = sparse_mx_to_torch_sparse_tensor(nei_a)
    nei_s = sparse_mx_to_torch_sparse_tensor(nei_s)
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_s = th.FloatTensor(preprocess_features(feat_s))
    # pap pap  type = <class 'scipy.sparse.coo.coo_matrix'>

    #pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    #psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

   # print(feat_p.size())
   # print(feat_p)
    return [nei_a, nei_s], [feat_p, feat_a, feat_s], [pap, psp], label, train, val, test


def load_dblp(ratio, type_num):
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_c = sp.eye(type_num[3])
    feat_t = np.load(path+"t_feat.npz")

    nei_ap = sp.load_npz(path + "nei_ap.npz")
    nei_apc = sp.load_npz(path + "nei_apc.npz")
    nei_apcp = sp.load_npz(path + "nei_apcp.npz")
    nei_apt = sp.load_npz(path + "nei_apt.npz")
    nei_aptp = sp.load_npz(path + "nei_aptp.npz")

    apa = sp.load_npz(path + "apa.npz")  
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz") 

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)

    nei_ap = sparse_mx_to_torch_sparse_tensor(nei_ap)
    nei_apc = sparse_mx_to_torch_sparse_tensor(nei_apc)
    nei_apcp = sparse_mx_to_torch_sparse_tensor(nei_apcp)
    nei_apt = sparse_mx_to_torch_sparse_tensor(nei_apt)
    nei_aptp = sparse_mx_to_torch_sparse_tensor(nei_aptp)
        
    feat_p = th.FloatTensor(preprocess_features(feat_p))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_t = th.FloatTensor(feat_t)
    feat_c = th.FloatTensor(preprocess_features(feat_c))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]

    return [nei_ap, nei_apc, nei_apcp, nei_apt, nei_aptp], [feat_a, feat_p, feat_t, feat_c], [apa, apcpa, aptpa], label, train, val, test


def load_aminer(ratio, type_num):
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    nei_a = sp.load_npz(path + "nei_a.npz")
    nei_r = sp.load_npz(path+ "nei_r.npz")

    feat_p_pap = np.load(path + "feat_p_pap.w1000.l100.npy").astype('float')
    feat_p_prp = np.load(path + "feat_p_prp.w1000.l100.npy").astype('float')
    feat_a = np.load(path + "feat_a.w1000.l100.npy").astype('float')
    feat_r = np.load(path + "feat_r.w1000.l100.npy").astype('float')

    feat_p = th.stack((th.FloatTensor(feat_p_pap),th.FloatTensor(feat_p_prp)))
    feat_a = th.FloatTensor(feat_a)
    feat_r = th.FloatTensor(feat_r)


    # feat_p = feat_p.mean(axis=0)
    # feat_p = sp.eye(type_num[0])
    # feat_a = sp.eye(type_num[1])
    # feat_r = sp.eye(type_num[2])
    # feat_p = th.FloatTensor(preprocess_features(feat_p))
    # feat_a = th.FloatTensor(preprocess_features(feat_a))
    # feat_r = th.FloatTensor(preprocess_features(feat_r))


    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    nei_a = sparse_mx_to_torch_sparse_tensor(nei_a)
    nei_r = sparse_mx_to_torch_sparse_tensor(nei_r)

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_r], [feat_p, feat_a, feat_r], [pap, prp], label, train, val, test


def load_imdb(ratio, type_num):
    # m a d k
    # 4275 5432 2083 7313
    path = "../data/imdb/"    
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    feat_m = sp.load_npz(path + "m_feat.npz").astype("float32")
    feat_a = sp.eye(type_num[1])
    feat_d = sp.eye(type_num[2])
    feat_k = sp.eye(type_num[3])

    nei_a = sp.load_npz(path + "nei_a.npz")
    nei_d = sp.load_npz(path + "nei_d.npz")
    nei_k = sp.load_npz(path + "nei_k.npz")
    nei_a = sparse_mx_to_torch_sparse_tensor(nei_a)
    nei_d = sparse_mx_to_torch_sparse_tensor(nei_d)
    nei_k = sparse_mx_to_torch_sparse_tensor(nei_k)

    mam = sp.load_npz(path + 'mam.npz')
    mdm = sp.load_npz(path + 'mdm.npz')
    mkm = sp.load_npz(path + 'mkm.npz')


    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    feat_m = th.FloatTensor(preprocess_features(feat_m))
    feat_a = th.FloatTensor(preprocess_features(feat_a))
    feat_d = th.FloatTensor(preprocess_features(feat_d))
    feat_k = th.FloatTensor(preprocess_features(feat_k))

    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return [nei_a, nei_d, nei_k], [feat_m, feat_a, feat_d, feat_k], [mam, mdm, mkm], label, train, val, test


def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == 'imdb':
        data = load_imdb(ratio, type_num)
    return data
