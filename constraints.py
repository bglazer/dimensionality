import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation



# find distance from source idx[:,0] to neighbors, using data so that gradient can be calculated
class JaccardGradient():
    def __init__(self, data, dim, num_nbrs, metric='euclidean', projection='pca'):
        self.dim = dim
        self.num_nbrs = num_nbrs
        self.metric = metric

        self.num_points = data.shape[0]
        balltree = BallTree(data, metric=self.metric, leaf_size=self.num_nbrs)
        _, self.idxs = balltree.query(data, k=self.num_nbrs)

        # project data into lower dimension
        if projection == 'random':
            self.initial_projected = np.random.random((self.num_points, self.dim))
        elif projection == 'pca':
            pca = PCA(self.dim)
            self.initial_projected = pca.fit_transform(data)
        else:
            raise Exception(f'The given projection: "{projection}" is not supported. Try "pca" or "random"')

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
        proj_dists, proj_idxs= balltree.query(srcs, k=self.num_nbrs)

        grad = np.zeros((self.num_points, self.dim))
        # find distance to closest non-neighbor in projected space
        # have to use a for loop because there are a varying number of non neighbors for each point
        # TODO somehow switch to vectorized operations?
        # can't use vectorized numpy operations
        for i,proj_idx in enumerate(proj_idxs):
            non_nbr_idxs = np.setdiff1d(proj_idx, self.idxs[i], assume_unique=True)
            if len(non_nbr_idxs) == 0:
                continue
            # get points inside the radius
            #non_nbr_idxs = non_nbr_idxs[1:]
            non_nbrs = projected[non_nbr_idxs]
            # find direction from source to the non-neighbors
            d_non_nbrs = srcs[i] - non_nbrs
            dist_non_nbrs = np.sqrt(np.sum(d_non_nbrs**2, axis=1, keepdims=True)) + .00001

            farthest_nbr = proj_dists[i,-1] 
            constraints = dist_nbrs[:,i] < farthest_nbr
            # where the constraint is not met, calculate the gradient
            unmet_idxs = self.idxs[i,1:][~constraints]
            unmet_nbrs = d_nbrs[:,i][~constraints]
            # np.newaxis is a hack to make dist_nbrs broadcastable
            unmet_dist = dist_nbrs[:,i,np.newaxis][~constraints]
            grad[unmet_idxs] += -unmet_nbrs/unmet_dist

            grad[non_nbr_idxs] += d_non_nbrs/dist_non_nbrs

            # gradient with respect to source point
            grad[i] += \
                np.sum(unmet_nbrs/unmet_dist, axis=0) - \
                np.sum(d_non_nbrs/dist_non_nbrs, axis=0)

            
        grad = grad*eps
        projected -= grad

        if verbose:
            # TODO more diagnostic info
            print(f'gradient {np.sum(grad**2)}')
            print(self.average_jaccard(projected))

        return projected, grad

    def fit_transform(self, eps=1.0, num_iters=50, verbose=False):
        if verbose:
            start_ajd = self.average_jaccard(self.initial_projected)
            
        projected, grad = self.step(self.initial_projected, eps=eps, verbose=verbose)
        for i in range(num_iters):
            # TODO learning rate decay?
            eps = .99*eps
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
    def animate(self, num_iters, labels):
        fig, ax = plt.subplots()
        ln = plt.scatter(self.initial_projected[:,0], self.initial_projected[:,1], c=labels)
        projected = self.initial_projected

        def init():
            #fig.gca().relim()
            #fig.gca().autoscale_view()
            return ln,

        # this is a closure that encloses the projected variable
        # that's necessary to keep updating the newest projection
        def update(frame):
            pr,_ = self.step(projected, eps=1.0, verbose=True)
            ln.set_offsets(pr)
            return ln,

        ani = FuncAnimation(fig, update, frames=range(num_iters),
                            init_func=init, interval=0.001, repeat=False)
        plt.show()
        ani.save(filename='constraints_animation.gif', writer='imagemagick')

