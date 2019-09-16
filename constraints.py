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

num_iters = 30
eps = .01
num_nbrs = 10

a = np.random.random((10,2))+10
b = np.random.random((10,2))
npdata = np.concatenate((a,b))
projected = npdata
num_points = 20
labels = np.ones(20)
labels[10:] = 2

balltree = BallTree(npdata, metric='euclidean', leaf_size=num_nbrs)
dists, idxs = balltree.query(npdata, k=num_nbrs)
dists = dists/np.max(dists)

# project data into lower dimension
projected = np.random.random((num_points, dim))
#pca = PCA(dim)
#projected = pca.fit_transform(data)
#TODO create a test data set of two completely disconnected normal distributions
#projected = projected/np.max(projected)

# find distance from source idx[:,0] to neighbors, using data so that gradient can be calculated

def clip(grad, maxgrad=.25):
    grad_nrm = np.sqrt(np.sum((grad)**2))
    if grad_nrm > maxgrad:
        grad = grad/grad_nrm * maxgrad
    return grad
  
def average_jaccard(projected):
    balltree = BallTree(projected, metric='euclidean', leaf_size=num_nbrs)
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

    # Calculate distances
    dist_nbrs = np.sqrt(np.sum(d_nbrs**2, axis=2))
    # find closest non-neighbor in projected space
    farthest_nbr_idx = np.argmax(dist_nbrs, axis=0)
    # distance to farthest neighbor
    balltree = BallTree(projected, metric='euclidean', leaf_size=num_nbrs)
    farthest_nbr_dist = dist_nbrs[farthest_nbr_idx, np.arange(num_points)]
    radius_idxs,radius_dists = balltree.query_radius(srcs, r=farthest_nbr_dist, sort_results=True, return_distance=True) 

    grad = np.zeros((num_points, dim))
    # find distance to closest non-neighbor in projected space
    radius_non_nbr_idxs = np.setdiff1d(radius_idxs, idxs, assume_unique=True)
    # getting element 1 and not 0 because element 0 is always the source
    closest_non_nbr_idx = radius_non_nbr_idxs[1]
    # have to use a for loop because there are a varying number of non neighbors for each point
    # can't use vectorized numpy operations
    for i,non_nbr_idxs in enumerate(radius_non_nbr_idxs):
        # get points inside the radius
        non_nbr_idxs = non_nbr_idxs[1:]
        non_nbrs = projected[non_nbr_idxs]
        # TODO should this be the opposite direction?
        # find direction from source to the non-neighbors
        d_non_nbrs = srcs[i] - non_nbrs
        dist_non_nbrs = np.sqrt(np.sum(d_non_nbrs**2, axis=1, keepdims=True)) + .00001
        # distance from src to non_neighbor < dist from src to neighbor
        constraints = dist_non_nbrs < dist_nbrs[:,i]
        # TODO these variables really need better names
        # Number of non_neighbors that are closer than each neighbor
        num_violated = np.sum(constraints, axis=0, keepdims=True)

        # np.newaxis is a hack to make dist_nbrs broadcastable
        grad[idxs[i,1:]] += -num_violated.T * d_nbrs[:,i]/dist_nbrs[:,i,np.newaxis]

        num_violated = np.sum(~constraints, axis=1, keepdims=True)
        grad[non_nbr_idxs] += num_violated * d_non_nbrs/dist_non_nbrs

    print(np.sum(constraints))
    print(np.sum(~constraints))

        #wh = np.where(constraints)
        ## select neighbor vectors where the constraints are true
        ## and update their gradient
        #nbr_idx = wh[0]
        #not_nbr_idx = wh[1]
        #grad[nbr_idx] += d_nbrs[wh[1], wh[0]]
        #breakpoint()

        ## select non-neighors where the constraints are false
        #nwh = np.where(~constraints)
        #grad[nwh[0]] += d_non_nbr[nwh[0]]
        ## TODO calculate mask statistics for diagnostic purposes

        #grad[non_nbr_idxs] += d_non_nbrs/dist_non_nbrs 

    #grad[idxs[:,1:]] += (d_nbrs/np.expand_dims(dist_nbr,2)).transpose((1,0,2))
        
    grad = grad*eps
    #grad = clip(grad)
    # TODO this should be subtraction to minimize the gradient
    projected -= grad
    print(f'gradient {np.sum(grad**2)}')
    return projected, grad

#ax.clear()
#plt.scatter(projected[:,0], projected[:,1], c=labels)
#plt.pause(.05)
def run(projected, eps):
    start_ajd = average_jaccard(projected)
    for i in range(num_iters):
        #eps = .99*eps
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
    plt.pause(1)

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
