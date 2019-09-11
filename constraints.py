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
num_neighbors = 10
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
#data.requires_grad = True
#data = Data(pos=data)
#data = Batch(Data(x=data))

# compute knn graph
def squared_euclidean(x, y):
    return np.sum((x-y)**2)

balltree = BallTree(npdata, metric=squared_euclidean, leaf_size=num_neighbors)
dists, idxs = balltree.query(npdata, k=num_neighbors)
dists = dists/np.max(dists)

# project data into lower dimension
projected = torch.rand(num_points, dim)

# find distance from source idx[:,0] to neighbors
#for start_idx, nbrs in enumerate(idxs):
#    d = data[start_idx] - data[nbrs]
nbrs = projected[idxs[:,1:]]
srcs = projected[idxs[:,0]]

nbrs = nbrs.permute((1,0,2))  

d = srcs - nbrs

# distance from source to non-neighbors


#for d in dists:
#    d[-1]
# find farthest neighbors
# write constraints
# optimize wrt constraints


