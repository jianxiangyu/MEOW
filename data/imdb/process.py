import os
from scipy.io import loadmat
import pickle
from scipy.sparse import coo_matrix, csr_matrix
import scipy.sparse as sp
import numpy as np
import random

# # Feature Process
# with open('./movie_feature_vector_6334.pickle', 'rb') as f:
#     feature = pickle.load(f)

# f_row = feature.row
# f_col = feature.col
# f_data = feature.data
# f_shape = feature.shape
# # print(f_data)
# csr_f = csr_matrix((f_data, (f_row, f_col)), shape=f_shape)

# feat_a = sp.save_npz("./m_feat.npz", csr_f)

# # Valid
# c1 = feature.todense()
# c2 = csr_f.todense()
# count = 0
# for i in range(f_shape[0]):
#     for j in range(f_shape[1]):
#         if c1[i,j]!=c2[i,j]:
#             count += 1
# print(count)

def relation(str):
    path = r"./IMDB.mat"  # mat文件路径
    data = loadmat(path)  # 读取mat文件
    print(data.keys())  # 查看mat文件中包含的变量    
    dict = {}

    for key, value in data.items(): 
        dict[key] = value

    r = str[0]+'_vs_'+str[1]

    relation = dict[r].todense()
    shape = dict[r].shape
    row = []
    col = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            if relation[i,j]!=0:
                row.append(i)
                col.append(j)

    row = np.array(row)
    col = np.array(col)
    relation_ = np.concatenate((row.reshape(-1,1), col.reshape(-1,1)),axis=1)

    np.savetxt(str + '.txt',relation_,fmt='%d')

# relation("md")
# relation("mk")

def neighbour(str):
    path = r"./IMDB.mat"  # mat文件路径
    data = loadmat(path)  # 读取mat文件
    print(data.keys())  # 查看mat文件中包含的变量    
    dict = {}

    for key, value in data.items(): 
        dict[key] = value

    r = 'm_vs_'+str
    
    nei_matrix = dict[r]
    nei_shape = nei_matrix.shape
    row = []
    col = []
    for i in range(nei_shape[0]):
        for j in range(nei_shape[1]):
            if nei_matrix[i,j]!=0:
                row.append(i)
                col.append(j)

    row = np.array(row)
    col = np.array(col)
    
    nei = coo_matrix((np.ones(len(row)), (row, col)), shape=nei_shape)

    sp.save_npz("./nei_" + str + ".npz", nei)
    
    # Valid
    c1 = nei_matrix.todense()
    c2 = nei.todense()
    count = 0
    for i in range(nei_shape[0]):   
        for j in range(nei_shape[1]):
            if c1[i,j]!=c2[i,j]:
                count += 1
    print(count)


# neighbour('a')
# neighbour('d')
# neighbour('k')


# # Label
# label = np.loadtxt("./index_label.txt", dtype=np.str)
# l = []
# for i in range(label.shape[0]):
#     l.append(int(label[i].split(',')[1]))

# np.save('labels.npy', l)

def split_dataset(n):
    M = 4275
    label = np.load('labels.npy')
    test_idx = np.load('test_' + str(n) + '.npy')
    val_idx = np.load('val_' + str(n) + '.npy')
    l1 = []
    l2 = []
    l3 = []
    for i in range(label.shape[0]):
        if label[i] == 1:
            l1.append(i)
        elif label[i] == 2:
            l2.append(i)
        elif label[i] == 3:
            l3.append(i)
    # print(len(l1))
    # print(len(l2))
    # print(len(l3))

    train_n = []
    for _ in range(10000):
        temp_idx = random.randint(0, len(l1)-1)
        if l1[temp_idx] in test_idx or l1[temp_idx] in val_idx:
            continue
        else:
            train_n.append(l1[temp_idx])
        
        if len(train_n) == n:
            break
    
    for _ in range(10000):
        temp_idx = random.randint(0, len(l2)-1)
        if l2[temp_idx] in test_idx or l2[temp_idx] in val_idx:
            continue
        else:
            train_n.append(l2[temp_idx])
        
        if len(train_n) == 2*n:
            break
    
    for _ in range(10000):
        temp_idx = random.randint(0, len(l3)-1)
        if l3[temp_idx] in test_idx or l3[temp_idx] in val_idx:
            continue
        else:
            train_n.append(l3[temp_idx])
        
        if len(train_n) == 3*n:
            break

    print(len(train_n))
    np.save('train_' + str(n) + '.npy', np.array(train_n))
    

    # train = index[0:n]
    # test = index[n:1000+n]
    # valid = index[1000+n:2000+n]
    # print(train)
    # np.save('train_' + str(n) + '.npy', train)
    # test_idx = np.save('test_' + str(n) + '.npy', test)
    # val_idx = np.save('val_' + str(n) + '.npy', valid)

split_dataset(20)
split_dataset(40)
split_dataset(60)