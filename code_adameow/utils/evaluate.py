import numpy as np
import torch
from .logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


##################################################
# This section of code adapted from pcy1302/DMGI #
##################################################

def evaluate(embeds, ratio, idx_train, idx_val, idx_test, label, nb_classes, device, dataset, lr, wd
             , starttime, epoch, isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    val_auc_score_list = []
    for _ in range(50):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []
        
        logits_list = []
        logits_val_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)
            logits_val_list.append(logits)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score_list.append(roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))
        
        best_logits = logits_val_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        val_auc_score_list.append(roc_auc_score(y_true=val_lbls.detach().cpu().numpy(),
                                            y_score=best_proba.detach().cpu().numpy(),
                                            multi_class='ovr'
                                            ))
        
    print("\t[Val Classification] auc: {:.2f} var: {:.2f}"
            .format(
                    np.mean(val_auc_score_list)*100,
                    np.std(val_auc_score_list)*100
                    )
        )
              
    if isTest:
        print("\t[Test Classification] Macro-F1_mean: {:.2f} var: {:.2f}  Micro-F1_mean: {:.2f} var: {:.2f} auc: {:.2f} var: {:.2f}"
              .format(np.mean(macro_f1s)*100,
                      np.std(macro_f1s)*100,
                      np.mean(micro_f1s)*100,
                      np.std(micro_f1s)*100,
                      np.mean(auc_score_list)*100,
                      np.std(auc_score_list)*100
                      )
              )
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)

    f = open("result_"+dataset+str(ratio)+".txt", "a")
    f.write(str(starttime.strftime('%Y-%m-%d %H:%M'))+"\t")
    f.write("Epoch: " + str(epoch) + "\t")
    f.write("Ma-F1_mean: "+str(np.around(np.mean(macro_f1s)*100, 2))+" +/- "+ str(np.around(np.std(macro_f1s)*100,2))+"\t")
    f.write(" Mi-F1_mean: "+str(np.around(np.mean(micro_f1s)*100,2))+" +/- "+ str(np.around(np.std(micro_f1s)*100,2))+"\t")
    f.write(" AUC_mean: "+str(np.around(np.mean(auc_score_list)*100,2))+ " +/- "+ str(np.around(np.std(auc_score_list)*100,2))+"\n")
    f.close()
    
    return np.mean(macro_f1s)*100 + np.mean(micro_f1s)*100 + np.mean(auc_score_list)*100
    # return np.mean(auc_score_list)*100
    # test_embs = np.array(test_embs.cpu())
    # test_lbls = np.array(test_lbls.cpu())

def run_kmeans(x, y, k, starttime, dataset, epoch):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    ARI_list = []
    for _ in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)

        n1 = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        a1 = adjusted_rand_score(y, y_pred)
        NMI_list.append(n1)
        ARI_list.append(a1)

    nmi = sum(NMI_list) / len(NMI_list)
    ari = sum(ARI_list) / len(ARI_list)

    print('\t[Clustering] Epoch: {:d} NMI: {:.2f}   ARI: {:.2f}'.format(epoch, np.round(nmi*100,2), np.round(ari*100,2)))
    f = open("result_" + dataset + "_NMI&ARI.txt", "a")
    f.write(str(starttime.strftime('%Y-%m-%d %H:%M'))+"\t Epoch: " + str(epoch) +"\t NMI: " + str(np.round(nmi*100,4)) +\
         "\t ARI: " + str(np.round(ari*100,4)) + "\n")
    f.close()