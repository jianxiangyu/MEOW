import torch
from .cluster import *

def InfoNce(z_tar, z, cluster_result, num_clusters, tau):
    dots = torch.exp(torch.mm(z_tar, z.t()) / tau)
    z_min = torch.diag(dots)
    
    im2cluster = cluster_result['im2cluster']
    im2cluster = torch.stack(im2cluster)
    k_times = len(num_clusters)
    N = z_tar.size(0)

    weight = torch.ones([N,N]).cuda() * k_times
    for i in range(k_times):
        node_idx = torch.range(0,N-1).long().cuda()
        cluster_idx = im2cluster[i]
        idx = torch.stack((node_idx, cluster_idx),0)
        data = torch.ones(N).cuda()
        coo_i = torch.sparse_coo_tensor(idx, data, [N,num_clusters[i]])
        weight = weight - torch.mm(coo_i, coo_i.to_dense().t())
    dots = torch.mul(dots,weight)
    
    nominator = z_min
    denominator = dots.mean(axis=1) * N
    loss = ((-1) * (torch.log(nominator / denominator))).mean()
    return loss

def ProtoNCE(z, cluster_result):
    loss = torch.tensor(0.0, requires_grad=True).cuda()
    for _, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],\
        cluster_result['centroids'],cluster_result['density'])):
        node_prototypes = prototypes[im2cluster]
        phi = density[im2cluster]
        pos_prototypes = torch.exp(torch.mul(z, node_prototypes).sum(axis=1) / phi)
        neg_prototypes = torch.exp(torch.mm(z, prototypes.t()) / density).mean(axis=1) * z.size(0)
        loss = loss + ((-1) * (torch.log(pos_prototypes / neg_prototypes))).mean()
    loss = loss / len(cluster_result['im2cluster'])
    return loss

def Info_and_Proto(z_anchor, z, num_clusters, tau):
    cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
    for num_cluster in num_clusters:
        cluster_result['im2cluster'].append(torch.zeros(z_anchor.size(0),dtype=torch.long).cuda())
        cluster_result['centroids'].append(torch.zeros(num_cluster,z_anchor.size(1)).cuda())
        cluster_result['density'].append(torch.zeros(num_cluster).cuda()) 
    cluster_result = run_kmeans(z, num_clusters, tau)

    loss_info = InfoNce(z_anchor, z, cluster_result, num_clusters, tau)
    loss_proto = ProtoNCE(z_anchor, cluster_result)
    return loss_info, loss_proto