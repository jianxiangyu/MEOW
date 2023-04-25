import argparse
import sys

argv = sys.argv
dataset = argv[1]

def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="acm")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=1200)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=410)
    parser.add_argument('--auc_limit', type=int, default=97)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.03)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--dropout', type=float, default=0.3)
 
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--nei_max', type=int, default=[690,690])
    parser.add_argument('--feat_mask', type=float, default=0.3)
    parser.add_argument('--adj_mask', type=float, default=0.2)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=1500)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=810)
    parser.add_argument('--auc_limit', type=int, default=97)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--l2_coef', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_mask', type=float, default=0.1)
    parser.add_argument('--adj_mask', type=float, default=0.3)
    parser.add_argument('--nei_max', type=int, default=[25,200,40])

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    return args

def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=3000)
    parser.add_argument('--start_eval', type=int, default=2510)
    parser.add_argument('--auc_limit', type=int, default=90)
  
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.3)
    parser.add_argument('--eva_wd', type=float, default=1e-3)
    
   # The parameters of learning process
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.0)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_mask', type=float, default=0.)
    parser.add_argument('--adj_mask', type=float, default=0.)
    parser.add_argument('--nei_max', type=int, default=[8,31])
     
    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--start_eval', type=int, default=10)
    parser.add_argument('--auc_limit', type=int, default=60)
  
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    # parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--l2_coef', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_mask', type=float, default=0.3)
    parser.add_argument('--adj_mask', type=float, default=0.1)
    parser.add_argument('--nei_max', type=int, default=[70,10,70])

    args, _ = parser.parse_known_args()
    args.type_num = [4275, 5432, 2083, 7313]  # the number of every node type
    args.nei_num = 3  # the number of neighbors' types
    return args


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == "aminer":
        args = aminer_params()
    elif dataset == "freebase":
        args = freebase_params()
    elif dataset == 'imdb':
        args = imdb_params()
    return args
