import numpy
import torch
from utils import load_data, set_params, evaluate, run_kmeans
from module.meow import MEOW
from module.preprocess import *
import warnings
import datetime
import pickle as pkl
import random

warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
seed = args.seed
numpy.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


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
    feat = feats[0]
    adjs = pathsim(adjs, args.nei_max)
    mask_feat = mask_features(feat, args.feat_mask)
    adjs_norm = [normalize_adj(adj) for adj in adjs]
    mask_adjs = mask_edges(adjs, sub_num, args.adj_mask)
    print("Feature and Edge Mask Finished!") 

    model = MEOW(feats_dim_list, sub_num, args.hidden_dim, args.embed_dim, args.tau, adjs_norm, args.lam_proto, \
                  args.dropout, nei_index, args.dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        feat = feat.cuda()
        feats = [f.cuda() for f in feats]
        label = label.cuda()
        idx_train = [i.cuda() for i in idx_train]
        idx_val = [i.cuda() for i in idx_val]
        idx_test = [i.cuda() for i in idx_test] 
    
    cnt_wait = 0
    best = 1e9

    starttime = datetime.datetime.now()
    epoch_times = args.nb_epochs
    
    num_clusters = args.num_cluster
    for epoch in range(epoch_times):
        if not args.save_emb:
            break
        print("---------------------------------------------------")
        print("Epoch:",epoch)
        model.train()
        optimizer.zero_grad()
        loss = model(feats, mask_feat, mask_adjs, adjs_norm, num_clusters)
        loss.backward()
        optimizer.step()
        print('best:', best)
        if best > loss:
            best = loss
            cnt_wait = 0
        else:
            cnt_wait += 1
        # print('current patience: ', cnt_wait)
        if cnt_wait >= args.patience:
            print('Early stopping!')
            break    
            
    if args.save_emb:
        print("Start to save embeds.")
        embeds = model.get_embeds()
        f = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl", "wb")
        pkl.dump(embeds.cpu().data.numpy(), f)
        f.close()
        print("Save finish.")
        run_kmeans(embeds.cpu(), torch.argmax(label.cpu(), dim=-1), nb_classes, starttime, args.dataset)
    else:
        print("Read embeds.")
        file = open("./embeds/"+args.dataset+"/"+str(args.turn)+".pkl","rb")
        embeds = torch.from_numpy(pkl.load(file)).cuda()
        file.close()
        
    for i in range(len(idx_train)):
        evaluate(embeds, args.ratio[i], idx_train[i], idx_val[i], idx_test[i], label, nb_classes, device, args.dataset,
                 args.eva_lr, args.eva_wd, starttime)
    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print("Total time: ", time, "s")

if __name__ == '__main__':
    train()