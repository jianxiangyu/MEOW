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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.03)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0007)
    parser.add_argument('--dropout', type=float, default=0.2)
 
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--feat_mask', type=float, default=0.3)
    parser.add_argument('--adj_mask', type=float, default=[0.3,0.2])
    parser.add_argument('--nei_max', type=int, default=[110,700])
    parser.add_argument('--num_cluster', default=[100,300], type=int, help='number of clusters')    
    parser.add_argument('--lam_proto', type=float, default=1)
    
    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.01)
    parser.add_argument('--eva_wd', type=float, default=0)
    
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_mask', type=float, default=0.2)
    parser.add_argument('--adj_mask', type=float, default=[0.2,0.5,0.6])
    parser.add_argument('--lam_proto', type=float, default=1)
    parser.add_argument('--nei_max', type=int, default=[25,200,40])
    parser.add_argument('--num_cluster', default=[200,700])

    args, _ = parser.parse_known_args()
    args.type_num = [4057, 14328, 7723, 20]  # the number of every node type
    args.nei_num = 1  # the number of neighbors' types
    return args

def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="aminer")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.1)
    parser.add_argument('--eva_wd', type=float, default=8e-4)
    
   # The parameters of learning process
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.0007)
    parser.add_argument('--dropout', type=float, default=0.2)
    
    # model-specific parameters
    parser.add_argument('--tau', type=float, default=0.9)
    parser.add_argument('--feat_mask', type=float, default=0.2)
    parser.add_argument('--adj_mask', type=float, default=[0.7,0.4])
    parser.add_argument('--nei_max', type=int, default=[5,21])
    parser.add_argument('--num_cluster', default=[500,1200], type=int, help='number of clusters')
    parser.add_argument('--lam_proto', type=float, default=0.1)
     
    args, _ = parser.parse_known_args()
    args.type_num = [6564, 13329, 35890]  # the number of every node type
    args.nei_num = 2  # the number of neighbors' types
    return args

def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_emb', action="store_false")
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="imdb")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nb_epochs', type=int, default=10000)
    
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
    parser.add_argument('--feat_mask', type=float, default=0.3)
    parser.add_argument('--adj_mask', type=float, default=[0.1,0.1,0.1])
    parser.add_argument('--nei_max', type=int, default=[70,10,70])
    parser.add_argument('--num_cluster', default=[500,700], type=int, help='number of clusters')
    parser.add_argument('--lam_proto', type=float, default=1)
     
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
    elif dataset == "imdb":
        args = imdb_params()
    return args
