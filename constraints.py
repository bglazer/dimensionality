import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation



# find distance from source idx[:,0] to neighbors, using data so that gradient can be calculated
class JaccardGradient():
    def __init__(self, dim, num_nbrs, metric='euclidean'):
        self.dim = dim
        self.num_nbrs = num_nbrs
        self.metric = metric

    # TODO fix this 
    def clip(self, grad, maxgrad=.25):
        grad_nrm = np.sqrt(np.sum((grad)**2))
        if grad_nrm > maxgrad:
            grad = grad/grad_nrm * maxgrad
        return grad
      
    def average_jaccard(self, projected):
        balltree = BallTree(projected, metric='euclidean', leaf_size=self.num_nbrs)
        _, proj_idxs = balltree.query(projected, k=self.num_nbrs)

        ajd = 0.0
        for i,proj_nbrs in enumerate(proj_idxs):
            original_nbrs = self.idxs[i,:]
            inter = len(set(original_nbrs) & set(proj_nbrs))
            union = len(set(original_nbrs) | set(proj_nbrs))
            ajd += (union - inter)/union

        return ajd/self.num_points


    def step(self, projected, eps, verbose=False):
        nbrs = projected[self.idxs[:,1:]]
        srcs = projected[self.idxs[:,0]]

        nbrs = nbrs.transpose((1,0,2))

        # distance from source to non-neighbors
        d_nbrs = srcs - nbrs

        # Calculate distances
        dist_nbrs = np.sqrt(np.sum(d_nbrs**2, axis=2))
        # find closest non-neighbor in projected space
        farthest_nbr_idx = np.argmax(dist_nbrs, axis=0)
        # distance to farthest neighbor
        balltree = BallTree(projected, metric='euclidean', leaf_size=self.num_nbrs)
        farthest_nbr_dist = dist_nbrs[farthest_nbr_idx, np.arange(self.num_points)]
        radius_idxs,radius_dists = balltree.query_radius(srcs, r=farthest_nbr_dist, sort_results=True, return_distance=True) 

        grad = np.zeros((self.num_points, self.dim))
        # find distance to closest non-neighbor in projected space
        radius_non_nbr_idxs = np.setdiff1d(radius_idxs, self.idxs, assume_unique=True)
        # getting element 1 and not 0 because element 0 is always the source
        closest_non_nbr_idx = radius_non_nbr_idxs[1]
        # have to use a for loop because there are a varying number of non neighbors for each point
        # can't use vectorized numpy operations
        for i,non_nbr_idxs in enumerate(radius_non_nbr_idxs):
            # get points inside the radius
            non_nbr_idxs = non_nbr_idxs[1:]
            non_nbrs = projected[non_nbr_idxs]
            # find direction from source to the non-neighbors
            d_non_nbrs = srcs[i] - non_nbrs
            dist_non_nbrs = np.sqrt(np.sum(d_non_nbrs**2, axis=1, keepdims=True)) + .00001
            # distance from src to non_neighbor < dist from src to neighbor
            constraints = dist_non_nbrs < dist_nbrs[:,i]
            # Number of non_neighbors that are closer than each neighbor
            num_violated = np.sum(constraints, axis=0, keepdims=True)

            # np.newaxis is a hack to make dist_nbrs broadcastable
            grad[self.idxs[i,1:]] += -num_violated.T * d_nbrs[:,i]/dist_nbrs[:,i,np.newaxis]

            num_violated = np.sum(~constraints, axis=1, keepdims=True)
            grad[non_nbr_idxs] += num_violated * d_non_nbrs/dist_non_nbrs

            # gradient with respect to source point
            grad[i] += \
                np.sum(d_nbrs[:,i]/dist_nbrs[:,i,np.newaxis], axis=0) - \
                np.sum(d_non_nbrs/dist_non_nbrs, axis=0)

            
        grad = grad*eps
        projected -= grad

        if verbose:
            # TODO more diagnostic info
            print(np.sum(constraints))
            print(np.sum(~constraints))
            print(f'gradient {np.sum(grad**2)}')
            print(self.average_jaccard(projected))

        return projected, grad

    def fit_transform(self, data, eps=1.0, num_iters=50, verbose=False):
        self.num_points = data.shape[0]
        balltree = BallTree(data, metric=self.metric, leaf_size=self.num_nbrs)
        _, self.idxs = balltree.query(data, k=self.num_nbrs)

        # project data into lower dimension
        #projected = np.random.random((num_points, dim))
        pca = PCA(self.dim)
        projected = pca.fit_transform(data)
        #projected = projected/np.max(projected)

        if verbose:
            start_ajd = self.average_jaccard(projected)
            
        for i in range(num_iters):
            # TODO learning rate decay?
            #eps = .99*eps
            projected, grad = self.step(projected, eps=eps, verbose=verbose)

        if verbose:
            end_ajd = self.average_jaccard(projected)
            print(f'start: average jaccard {start_ajd}')
            print(f'end  : average jaccard {end_ajd}')
         
        return projected, grad
            

    def plot(self, projected, labels=None, grad=None):
        fig, ax = plt.subplots()
        plt.scatter(projected[:,0], projected[:,1], c=labels, s=6)
        if grad is not None:
            lines = np.stack((projected,projected+grad)).transpose((1,0,2))
            lc = collections.LineCollection(lines)
            ax.add_collection(lc)
        plt.show()
        plt.pause(10)

    # TODO fix this
    def animate(self, projected):
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

