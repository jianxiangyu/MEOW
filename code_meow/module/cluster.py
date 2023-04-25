import faiss
import numpy as np
import torch
import torch.nn.functional as F


def run_kmeans(x, clusters, tau):
    x = x.cpu().detach().numpy()
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(clusters):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = num_cluster
        clus = faiss.Clustering(d, k)
        clus.verbose = False
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.gpu = True
        clus.max_points_per_centroid = 4000
        clus.min_points_per_centroid = 5

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatIP(res, d, cfg)  

        clus.train(x, index)

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = []
        for n in I:
            im2cluster.append(n[0])
        
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for _ in range(k)]
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+5)        
                density[i] = d
                
        #if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = tau*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = F.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results