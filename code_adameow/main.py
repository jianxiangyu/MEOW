import numpy as np
import torch
from utils import load_data, set_params, evaluate, run_kmeans
from module.adameow import AdaMEOW
from module.preprocess import *
import warnings
import datetime
import pickle as pkl
import random
import matplotlib.pyplot as plt
import os

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset

## random seed ##
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def save_embeds(embeds, dataset, time):
    print("Start to save embeds.")
    f = open("./embeds/"+dataset+"/"+str(time)+".pkl", "wb")
    pkl.dump(embeds.cpu().data.numpy(), f)
    f.close()
    print("Save finish.")

def train():
    nei_index, feats, adjs, label, idx_train, idx_val, idx_test = \
        load_data(args.dataset, args.ratio, args.type_num)
    nb_classes = label.shape[-1]

    if args.dataset == 'aminer':
        feats_dim_list = [64,64,64]
    else:
        feats_dim_list = [i.shape[1] for i in feats]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("The dim of different kinds' nodes' feature: ", feats_dim_list)
    print("Label: ", label.sum(axis=0))
    print(args)
    feat = feats[0]
    mask_feat = mask_features(feat, args.feat_mask)
    adjs = pathsim(adjs, args.nei_max)
    mask_adjs = mask_edges(adjs, sub_num, args.adj_mask)
    adjs = [normalize_adj(adj) for adj in adjs]

    model = AdaMEOW(feats_dim_list, sub_num, args.hidden_dim, args.embed_dim, args.tau, adjs, args.dropout, nei_index, feat, args.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feats = [f.cuda() for f in feats]
        adjs = [adj.cuda() for adj in adjs]
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test] 
    
    cnt_wait = 0
    best = 1e9
    period = 100

    starttime = datetime.datetime.now()

    if not args.save_emb:
        print("Read embeds.")
        file = open("./embeds/"+args.dataset+"/2023-02-27 13:00.pkl","rb")
        embeds = torch.from_numpy(pkl.load(file)).cuda()
        file.close()
        for i in range(len(idx_train)):
            auc = evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
             args.eva_lr, args.eva_wd, starttime, 0)
            if i == 0 and auc < args.auc_limit:
                break
        exit()
    
    for epoch in range(args.nb_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model(feats, mask_feat, adjs, mask_adjs)
        loss.backward()
        optimizer.step()

        if best > loss:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1

        print("Epoch:", epoch)
        print('Total loss: ', loss)
        if (epoch + 1) % period == 0 and epoch > args.start_eval:
            embeds = model.get_embeds()
            print("---------------------------------------------------")
            embeds = model.get_embeds()

            run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset, epoch+1)
            for i in range(len(idx_train)):
                auc = evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd, starttime, epoch+1)
                if i == 0 and auc < args.auc_limit:
                    break

if __name__ == '__main__':
    train()