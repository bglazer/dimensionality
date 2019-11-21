import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from itertools import combinations

def l2(a):
    return np.sqrt(np.sum(a**2, axis=1, keepdims=True))

# find distance from source idx[:,0] to neighbors, using data so that gradient can be calculated
class JaccardGradient():
    def __init__(self, data, dim, num_nbrs, metric='euclidean', projection='pca'):
        self.dim = dim
        self.num_nbrs = num_nbrs
        self.metric = metric

        self.num_points = data.shape[0]
        balltree = BallTree(data, metric=self.metric, leaf_size=self.num_nbrs)
        _, self.idxs = balltree.query(data, k=self.num_nbrs)

        self.rjd = self.relative_jaccard()

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
        grad_nrm = np.sqrt(np.sum(grad**2, axis=1))
        over = grad_nrm > g
        grad[over] = grad[over]/grad_nrm[over] * maxgrad
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
    
    def relative_jaccard(self):
        self.nbrhds = {i:[] for i in range(self.num_points)}
        for nbrs in self.idxs:
            # exclude the first idx which is just the source point
            nbrs = nbrs[1:]
            i = nbrs[0]
            for nbr in nbrs:
                self.nbrhds[nbr].append(i)
    
        # TODO these variable names suck
        rjd = {}
        for i,j in combinations(range(self.num_points), r=2):
            # neighbors of the source point
            a = self.idxs[i]
            ajd = 0.0
            # neighborhoods the other point is in
            for nbrhd in self.nbrhds[j]:
                # points in the other neighborhoods
                b = self.idxs[nbrhd]
                # jaccard distance between the source point and the points in the other neighborhood
                inter = len(set(a) & set(b))
                union = len(set(a) | set(b))
                ajd += (union - inter)/union
            if len(self.nbrhds[j]) > 0:
                rjd[(i,j)] = 1-ajd/len(self.nbrhds[j])
            else:
                rjd[(i,j)] = 0.0

        return rjd

    def step(self, projected, num_updates=None, prev_grad=None, eps=1.0, gamma=.9, verbose=False):
        # determine whether we're updating every point (num_updates = None) or if we're doing stochasting gradient descent
        if num_updates is None:
            update_idxs = np.arange(self.num_points)
            num_updates = self.num_points
        else:
            # stochastic updates
            update_idxs = np.random.choice(self.num_points, size=num_updates, replace=False)

        idxs = self.idxs[update_idxs]
        srcs = projected[idxs[:,0]]

        # distance to farthest neighbor
        balltree = BallTree(projected, metric=self.metric, leaf_size=self.num_nbrs)
        # find k nearest neighbors and their distances
        proj_dists, proj_idxs= balltree.query(srcs, k=self.num_nbrs)

        grad = np.zeros((self.num_points, self.dim))
        # initialize previous gradient on first step
        if prev_grad is None:
            prev_grad = np.zeros((self.num_points, self.dim))

        # find distance to closest non-neighbor in projected space
        # have to use a for loop because there are a varying number of non neighbors for each point
        # so can't use vectorized numpy operations
        for i,proj_idx in enumerate(proj_idxs):
            # projected neighbors - true_neighbors = non-true neighbors in projected neighborhood
            non_nbr_idxs = np.setdiff1d(proj_idx, idxs[i], assume_unique=True)
            # true_neighbors - projected neighbors = true neighbors not in projected neighborhood
            unmet_idxs = np.setdiff1d(idxs[i], proj_idx, assume_unique=True)

            non_nbrs = projected[non_nbr_idxs]
            ## find direction from source to the non-neighbors
            d_non_nbrs = srcs[i] - non_nbrs
            dist_non_nbrs = np.sqrt(np.sum(d_non_nbrs**2, axis=1, keepdims=True)) + .00001

            # distance to neighbors not in the projected neighborhood 
            d_unmet_nbrs = srcs[i] - projected[unmet_idxs]
            dist_unmet_nbrs = np.sqrt(np.sum(d_unmet_nbrs**2, axis=1, keepdims=True))
            grad[unmet_idxs] += (-d_unmet_nbrs/dist_unmet_nbrs)
            # adjust gradient wrt average jaccard distance of points
            # Not sure this is working
            #for unmet_idx in unmet_idxs:
            #    rjd_idx = (i,unmet_idx) if unmet_idx > i else (unmet_idx,i)
            #    adjustment = 1-self.rjd[rjd_idx]
                #adjustment = 1 if adjustment > .9 else 0
                #print(adjustment)
                #grad[unmet_idxs] *= adjustment
                #breakpoint()

            grad[non_nbr_idxs] += (d_non_nbrs/dist_non_nbrs)

            # gradient with respect to farthest neighbor (threshold of neighborhood)
            # TODO not sure  this is working right
            farthest_nbr_idx = proj_idxs[i, -1]
            d_farthest_nbr = srcs[i] - projected[farthest_nbr_idx]
            dist_farthest_nbr = proj_dists[i, -1]
            # TODO multiply by number of unmet neighbors and non-neighbors?
            grad[farthest_nbr_idx] += d_farthest_nbr/dist_farthest_nbr

            # gradient with respect to source point
            grad[update_idxs[i]] += \
                np.sum(d_unmet_nbrs/dist_unmet_nbrs, axis=0) - \
                np.sum(d_non_nbrs/dist_non_nbrs, axis=0)


            
        grad = (grad*eps + prev_grad*gamma) 
        #print(np.max(grad, axis=0))
        #print(np.min(grad, axis=0))

        #projected[update_idxs] -= grad[update_idxs]
        projected -= grad

        ajd = self.average_jaccard(projected)

        if verbose:
            # TODO more diagnostic info
            print(f'gradient: {np.sum(grad**2)}')
            print(f'     AJD: {ajd})')

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
            if verbose: print('step',i)
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

    def animate(self, num_iters, labels):
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
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
            pr,gr,ajd = self.step(projected, prev_grad=prev_grad, eps=1.0, gamma=.9, verbose=True)
            ln.set_offsets(pr)
            return ln,

        ani = FuncAnimation(fig, update, frames=range(num_iters),
                            init_func=init, interval=0.001, repeat=False)
        plt.show()
        ani.save(filename='constraints_animation.gif', writer='imagemagick')

