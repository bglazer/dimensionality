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
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# load data
num_points = 400
num_nbrs = 10
num_non_nbrs = 50
dim = 2
#sample_size = 50

data = datasets.MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)
labels = data.targets[:num_points].numpy()
data = data.data
data = data.type(torch.FloatTensor)
data = Variable(data.view(-1, 28*28))
data = data[:num_points]
npdata = data.numpy()
#data = Data(pos=data)
#data = Batch(Data(x=data))

# compute knn graph
def squared_euclidean(x, y):
    d = np.sum((x-y)**2)
    return d

balltree = BallTree(npdata, metric=squared_euclidean, leaf_size=num_nbrs)
dists, idxs = balltree.query(npdata, k=num_nbrs)
dists = dists/np.max(dists)

# project data into lower dimension
#projected = np.random.random((num_points, dim))
pca = PCA(2)
projected = pca.fit_transform(data)
projected = projected/np.max(projected)

# find distance from source idx[:,0] to neighbors, using data so that gradient can be calculated
#for start_idx, nbrs in enumerate(idxs):
#    d = data[start_idx] - data[nbrs]
num_iters = 20
eps = .01
clip = .25

def clip(grad, maxgrad=.25):
    grad_nrm = np.sqrt(np.sum((grad)**2))
    if grad_nrm > maxgrad:
        grad = grad/grad_nrm * maxgrad
    return grad

def step(projected):
    nbrs = projected[idxs[:,1:]]
    srcs = projected[idxs[:,0]]

    nbrs = nbrs.transpose((1,0,2))

    d_nbrs = srcs - nbrs

    # distance from source to non-neighbors

    d_non_nbrs = np.ndarray((num_non_nbrs, num_points, dim))

    #for idx_row in idxs:
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
    dist_farthest = np.sum(d_farthest**2, axis=1)

    # calculate gradient
    grad = np.zeros((num_points, dim))
    dist_nbr = np.sum(d_nbrs**2, axis=2)
    dist_non_nbr = np.sum(d_non_nbrs**2, axis=2)

    # Points in the local neighborhood 
    nbr_mask = dist_nbr < dist_farthest
    non_nbr_mask = dist_non_nbr > dist_farthest

    grad_nbr = d_farthest - d_nbrs
    grad_nbr[nbr_mask] = 0.0
    grad_nbr = np.sum(grad_nbr, axis=0)
    grad += grad_nbr

    grad_non_nbr = d_farthest - d_non_nbrs
    grad_non_nbr[non_nbr_mask] = 0.0
    grad_non_nbr = np.sum(grad_non_nbr, axis=0)
    grad += grad_non_nbr

    # Compute gradient with respect to the threshold (t), farthest point
    #grad_t = -d_farthest
    #tidx = idxs[:,-1]
    #grad[tidx] += grad_t

    # Gradient wrt points in local neighborhood
    grad_n = d_nbrs
    grad_n[nbr_mask] = 0.0
    grad += np.sum(grad_n, axis=0)

    # Gradient wrt points not in local neighborhood
    grad_non = d_non_nbrs
    grad_non[non_nbr_mask] = 0.0
    grad += np.sum(grad_non, axis=0)

    # optimize wrt constraints
    grad = grad*eps
    #grad = clip(grad)
    projected = projected + grad
    return projected

#ax.clear()
#plt.scatter(projected[:,0], projected[:,1], c=labels)
#plt.pause(.05)
def plot(projected):
    fig, ax = plt.subplots()
    for i in range(num_iters):
        projected = step(projected)
    plt.scatter(projected[:,0], projected[:,1], c=labels)
    plt.show()

def animate(projected):
    fig, ax = plt.subplots()
    ln, = plt.plot(projected[:,0],projected[:,1], c=labels)

    def init():
        fig.gca().relim()
        fig.gca().autoscale_view()
        return ln,

    def update(frame):
        projected = step(projected)
        ln.set_data(projected[:,0],projected[:,1])
        return ln,

    ani = FuncAnimation(fig, update, frames=range(num_iters),
                        init_func=init, interval=1)
                        #init_func=init, blit=True, interval=1)
    plt.show()

#animate(projected)
plot(projected)
