import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation

def l2(a):
    return np.sqrt(np.sum(l2**2, axis=1, keepdims=True))

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
        balltree = BallTree(projected, metric=self.metric, leaf_size=self.num_nbrs)
        _, proj_idxs = balltree.query(projected, k=self.num_nbrs)

        ajd = 0.0
        for i,proj_nbrs in enumerate(proj_idxs):
            original_nbrs = self.idxs[i,:]
            inter = len(set(original_nbrs) & set(proj_nbrs))
            union = len(set(original_nbrs) | set(proj_nbrs))
            ajd += (union - inter)/union

        return ajd/self.num_points

    def step(self, projected, num_updates=None, prev_grad=None, eps=1.0, gamma=.9, verbose=False):
        if num_updates is None:
            update_idxs = np.arange(self.num_points)
            num_updates = self.num_points
        else:
            update_idxs = np.random.choice(self.num_points, size=num_updates, replace=False)

        idxs = self.idxs[update_idxs]
        srcs = projected[idxs[:,0]]

        # distance to farthest neighbor
        balltree = BallTree(projected, metric=self.metric, leaf_size=self.num_nbrs)
        proj_dists, proj_idxs= balltree.query(srcs, k=self.num_nbrs)

        grad = np.zeros((self.num_points, self.dim))
        if prev_grad is None:
            prev_grad = np.zeros((self.num_points, self.dim))
        # TODO somehow switch to vectorized operations?
        # find distance to closest non-neighbor in projected space
        # have to use a for loop because there are a varying number of non neighbors for each point
        # so can't use vectorized numpy operations
        for i,proj_idx in enumerate(proj_idxs):
            # projected neighbors - true_neighbors = non-true neighbors in projected neighborhood
            non_nbr_idxs = np.setdiff1d(proj_idx, idxs[i], assume_unique=True)
            # true_neighbors - projected neighbors = true neighbors not in projected neighborhood
            unmet_idxs = np.setdiff1d(idxs[i], proj_idx, assume_unique=True)

            #if len(non_nbr_idxs) == 0:
            #    continue
            ## get points inside the radius
            ##non_nbr_idxs = non_nbr_idxs[1:]
            non_nbrs = projected[non_nbr_idxs]
            ## find direction from source to the non-neighbors
            d_non_nbrs = srcs[i] - non_nbrs
            dist_non_nbrs = np.sqrt(np.sum(d_non_nbrs**2, axis=1, keepdims=True)) + .00001

            d_unmet_nbrs = srcs[i] - projected[unmet_idxs]
            dist_unmet_nbrs = np.sqrt(np.sum(unmet_nbrs**2, axis=1, keepdims=True))
            farthest_nbr_idx = idxs[i, -1]
            # TODO add gradient for farthest_nbr idx
            grad[unmet_idxs] += -unmet_nbrs/dist_unmet_nbrs

            grad[non_nbr_idxs] += d_non_nbrs/dist_non_nbrs

            # gradient with respect to farthest neighbor (threshold of neighborhood)
            farthest_nbr = proj_dists[-1]
            farthest_nbr_dist = np.sqrt(np.sum(proj
            breakpoint()
            #grad[farthest_nbr_idx] += fartO

            # gradient with respect to source point
            grad[update_idxs[i]] += \
                np.sum(unmet_nbrs/unmet_dist, axis=0) - \
                np.sum(d_non_nbrs/dist_non_nbrs, axis=0)

            
        grad = (grad*eps + prev_grad*gamma)

        #projected[update_idxs] -= grad[update_idxs]
        projected -= grad

        ajd = self.average_jaccard(projected)

        if verbose:
            # TODO more diagnostic info
            print(f'gradient {np.sum(grad**2)}')
            print(ajd)

        return projected, grad, ajd

    def fit_transform(self, num_updates=None, eps=1.0, gamma=.9, num_iters=50, verbose=False):
        if verbose:
            start_ajd = self.average_jaccard(self.initial_projected)
            
        projected, grad, ajd = self.step(projected=self.initial_projected, 
                                         num_updates=num_updates,
                                         gamma=gamma,
                                         eps=eps,
                                         verbose=verbose)
        prev_grad = grad
        best_ajd = 1.0
        for i in range(num_iters-1):
            # TODO learning rate decay?
            #eps = .99*eps
            if verbose: print(i)
            projected, grad, ajd = self.step(projected=projected,
                                             num_updates=num_updates,
                                             prev_grad=prev_grad,
                                             gamma=gamma,
                                             eps=eps,
                                             verbose=verbose)
            prev_grad = grad
            if ajd < best_ajd:
                best_ajd = ajd
                best_projection = projected

        if verbose:
            print(f'start: average jaccard {start_ajd}')
            print(f'best : average jaccard {best_ajd}')
         
        return best_projection, best_ajd
            

    def plot(self, projected, labels=None, grad=None, filename=None):
        fig, ax = plt.subplots(figsize=(20,20))
        plt.scatter(projected[:,0], projected[:,1], c=labels, s=20)
        if grad is not None:
            lines = np.stack((projected,projected+grad)).transpose((1,0,2))
            lc = collections.LineCollection(lines)
            ax.add_collection(lc)
        if filename is None:
            plt.show(block=True)
        else:
            plt.savefig(filename)

    # TODO fix this
    def animate(self, num_iters, labels):
        fig, ax = plt.subplots()
        ln = plt.scatter(self.initial_projected[:,0], self.initial_projected[:,1], c=labels, s=3)
        projected = self.initial_projected
        prev_grad = np.zeros((self.num_points, self.dim))

        def init():
            #fig.gca().relim()
            #fig.gca().autoscale_view()
            return ln,

        # this is a closure that encloses the projected variable
        # that's necessary to keep updating the newest projection
        def update(frame):
            pr,gr = self.step(projected, prev_grad, eps=1.0, gamma=.9, verbose=True)
            ln.set_offsets(pr)
            return ln,

        ani = FuncAnimation(fig, update, frames=range(num_iters),
                            init_func=init, interval=0.001, repeat=False)
        plt.show()
        ani.save(filename='constraints_animation.gif', writer='imagemagick')

