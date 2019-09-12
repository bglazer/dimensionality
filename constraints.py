import torch
import torch_geometric
from torch_geometric.transforms.knn_graph import KNNGraph
from torch_geometric.data import Data, Batch
from torch import tensor, zeros
from torch_scatter import scatter_mean #, scatter_max
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric

# load data
num_points = 400
num_nbrs = 10
dim = 2
#sample_size = 50

data = datasets.MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)
labels = data.targets[:num_points]
data = data.data
data = data.type(torch.FloatTensor)
data = Variable(data.view(-1, 28*28))
data = data[:num_points]
npdata = data.numpy()
#data = Data(pos=data)
#data = Batch(Data(x=data))

# compute knn graph
balltree = BallTree(npdata, metric=squared_euclidean, leaf_size=num_nbrs)
dists, idxs = balltree.query(npdata, k=num_nbrs)
dists = dists/np.max(dists)

# project data into lower dimension
# TODO better initial projection? T-SNE/UMAP/PCA etc? Maybe PCA, given results in paper?
projected = np.random.random((num_points, dim))

# find distance from source idx[:,0] to neighbors, using data so that gradient can be calculated
#for start_idx, nbrs in enumerate(idxs):
#    d = data[start_idx] - data[nbrs]
num_iters = 1
eps = .01

#def step(projected):
nbrs = projected[idxs[:,1:]]
srcs = projected[idxs[:,0]]

nbrs = nbrs.transpose((1,0,2))

d_nbrs = srcs - nbrs

# distance from source to non-neighbors
num_non_nbrs = 10

#non_nbrs = np.ndarray((total_non_nbrs, dim))
d_non_nbrs = np.ndarray((num_non_nbrs, num_points, dim))

for idx_row in idxs:
    src_idx = idx_row[0]
    src = projected[src_idx]

    p = np.ones(num_points)
    p = p * 1/(num_points - num_nbrs)
    p[idx_row] = 0.0
    random_non_nbrs = np.random.choice(num_points, size=(num_non_nbrs), replace=False, p=p)
    d_non_nbrs[:,src_idx,:] = src - projected[random_non_nbrs]
    
# find farthest neighbors
d_farthest = d_nbrs[-1,:,:]
dist_farthest = np.sum(d_nbrs[-1,:,:], axis=1)

# calculate gradient
dist_nbr = np.sum(d_nbrs, axis=2)
dist_non_nbr = np.sum(d_non_nbrs, axis=2)

nbr_mask = dist_nbr < dist_farthest
non_nbr_mask = dist_non_nbr > dist_farthest

grad = np.sum(d_farthest - d_nbrs, axis=0)

# optimize wrt constraints
#projected = projected + grad*eps
#return projected
