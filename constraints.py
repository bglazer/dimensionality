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
from matplotlib import collections
from matplotlib.animation import FuncAnimation

# load data
num_points = 400
dim = 2

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

# compute knn graph
def squared_euclidean(x, y):
    d = np.sum((x-y)**2)
    return d

num_iters = 10
eps = .1
num_nbrs = 50
num_non_nbrs = 50

balltree = BallTree(npdata, metric=squared_euclidean, leaf_size=num_nbrs)
dists, idxs = balltree.query(npdata, k=num_nbrs)
dists = dists/np.max(dists)

# project data into lower dimension
projected = np.random.random((num_points, dim))
#pca = PCA(dim)
#projected = pca.fit_transform(data)
#projected = projected/np.max(projected)

# find distance from source idx[:,0] to neighbors, using data so that gradient can be calculated

def clip(grad, maxgrad=.25):
    grad_nrm = np.sqrt(np.sum((grad)**2))
    if grad_nrm > maxgrad:
        grad = grad/grad_nrm * maxgrad
    return grad
  
def average_jaccard(projected):
    balltree = BallTree(projected, metric=squared_euclidean, leaf_size=num_nbrs)
    _, proj_idxs = balltree.query(projected, k=num_nbrs)

    ajd = 0.0
    for i,proj_nbrs in enumerate(proj_idxs):
        original_nbrs = idxs[i,:]
        inter = len(set(original_nbrs) & set(proj_nbrs))
        union = len(set(original_nbrs) | set(proj_nbrs))
        ajd += (union - inter)/union

    return ajd/num_points


def step(projected, eps):
    nbrs = projected[idxs[:,1:]]
    srcs = projected[idxs[:,0]]

    nbrs = nbrs.transpose((1,0,2))

    # distance from source to non-neighbors
    d_nbrs = srcs - nbrs

    # find farthest neighbors
    d_farthest = d_nbrs[-1,:,:]

    d_non_nbrs = np.ndarray((num_non_nbrs, num_points, dim))
    # randomly select points that weren't in the original local neighborhood
    for idx_row in idxs:
        src_idx = idx_row[0]
        src = projected[src_idx]

        p = np.ones(num_points)
        p = p * 1/(num_points - num_nbrs)
        p[idx_row] = 0.0
        random_non_nbrs = np.random.choice(num_points, size=(num_non_nbrs), replace=False, p=p)
        d_non_nbrs[:,src_idx,:] = src - projected[random_non_nbrs]
        

    # Calculate distances
    dist_nbr = np.sum(d_nbrs**2, axis=2)
    dist_non_nbr = np.sum(d_non_nbrs**2, axis=2)
    dist_farthest = np.sum(d_farthest**2, axis=1)

    # calculate gradient
    grad = np.zeros((num_points, dim))

    # Points in the local neighborhood 
    nbr_mask = dist_nbr < dist_farthest
    non_nbr_mask = dist_non_nbr > dist_farthest

    # Compute gradient of source point
    grad_nbr = d_farthest - d_nbrs
    grad_nbr[nbr_mask] = 0.0
    # sum puts this into a vector form, which has to expanded so that it's broadcastable
    n_nrm = np.expand_dims(np.sum(~nbr_mask, axis=0), 1) + 0.00001
    grad_nbr = np.sum(grad_nbr, axis=0)/n_nrm
    grad += grad_nbr

    grad_non_nbr = d_farthest - d_non_nbrs
    grad_non_nbr[non_nbr_mask] = 0.0
    non_nrm = np.expand_dims(np.sum(~non_nbr_mask, axis=0), 1) + .00001
    grad_non_nbr = np.sum(grad_non_nbr, axis=0)/non_nrm
    grad += grad_non_nbr

    # Compute gradient with respect to the threshold (t), i.e. the farthest point in the original neighborhood
    grad_t = d_farthest
    grad[idxs[:,-1]] += grad_t

    # TODO add penalty for being too close to the source

    # Gradient wrt points in local neighborhood
    grad_n = d_nbrs
    grad_n[nbr_mask] = 0.0
    grad += -np.sum(grad_n, axis=0)/n_nrm

    # Gradient wrt points not in local neighborhood
    grad_non = d_non_nbrs
    grad_non[non_nbr_mask] = 0.0
    grad += -np.sum(grad_non, axis=0)/non_nrm

    # optimize wrt constraints
    grad = grad*eps
    #grad = clip(grad)
    # TODO this should be subtraction to minimize the gradient
    projected += grad
    print(f'neighbors in threshold {np.sum(nbr_mask)/(num_points*num_nbrs)}')
    print(f'non-neighbors not in threshold {np.sum(non_nbr_mask)/(num_points*num_non_nbrs)}')
    print(f'gradient {np.sum(grad**2)}')
    return projected, grad

#ax.clear()
#plt.scatter(projected[:,0], projected[:,1], c=labels)
#plt.pause(.05)
def run(projected, eps):
    start_ajd = average_jaccard(projected)
    for i in range(num_iters):
        eps = .99*eps
        projected, grad = step(projected, eps=eps)
    plot(projected,grad)
    end_ajd = average_jaccard(projected)
    print(f'start: average jaccard {start_ajd}')
    print(f'end  : average jaccard {end_ajd}')
    
def plot(projected, grad=None):
    fig, ax = plt.subplots()
    plt.scatter(projected[:,0], projected[:,1], c=labels, s=6)
    if grad is not None:
        lines = np.stack((projected,projected+grad)).transpose((1,0,2))
        lc = collections.LineCollection(lines)
        ax.add_collection(lc)
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
run(projected, eps)
