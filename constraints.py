import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from itertools import combinations

def l2(a):
    return np.sqrt(np.sum(a**2, axis=1, keepdims=True))

def clip(grad, maxgrad=.25):
    grad_nrm = np.sqrt(np.sum(grad**2, axis=1))
    over = grad_nrm > maxgrad
    grad[over] = grad[over]/grad_nrm[over].reshape((-1,1)) * maxgrad
    return grad 
    
class JaccardGradient():
    def __init__(self, data, dim, num_nbrs, metric='euclidean', projection='pca'):
        self.dim = dim
        self.num_nbrs = num_nbrs
        self.metric = metric

        self.num_points = data.shape[0]
        balltree = BallTree(data, metric=self.metric, leaf_size=self.num_nbrs)
        self.dists, self.idxs = balltree.query(data, k=self.num_nbrs)

        #self.rjd = self.relative_jaccard()

        # project data into lower dimension
        if projection == 'random':
            self.projected = np.random.random((self.num_points, self.dim))
        elif projection == 'pca':
            pca = PCA(self.dim)
            self.projected = pca.fit_transform(data)
        else:
            raise Exception(f'The given projection: "{projection}" is not supported. Try "pca" or "random"')

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

        # find k nearest neighbors and their distances in the projected space
        balltree = BallTree(projected, metric=self.metric, leaf_size=self.num_nbrs)
        proj_dists, proj_idxs= balltree.query(srcs, k=self.num_nbrs)

        grad = np.zeros((self.num_points, self.dim))
        # initialize previous gradient on first step
        if prev_grad is None:
            prev_grad = np.zeros((self.num_points, self.dim))

        # have to use a for loop because there are a varying number of non neighbors for each point
        # so can't use vectorized numpy operations
        for i,proj_idx in enumerate(proj_idxs):
            # projected neighbors - true_neighbors = non-true neighbors in projected neighborhood
            non_nbr_idxs = np.setdiff1d(proj_idx, idxs[i], assume_unique=True)
            # true_neighbors - projected neighbors = true neighbors not in projected neighborhood
            unmet_idxs = np.setdiff1d(idxs[i], proj_idx, assume_unique=True)
            non_nbrs = projected[non_nbr_idxs]

            dist_farthest_nbr = proj_dists[i, -1]

            # gradient with respect to non-neighbors in the projected neighborhood 
            d_non_nbrs = srcs[i] - non_nbrs
            dist_non_nbrs = np.sqrt(np.sum(d_non_nbrs**2, axis=1, keepdims=True)) + .00001
            grad[non_nbr_idxs] += (d_non_nbrs/dist_non_nbrs) #* (dist_non_nbrs - dist_farthest_nbr)

            # gradient with respect to neighbors not in the projected neighborhood 
            d_unmet_nbrs = srcs[i] - projected[unmet_idxs]
            dist_unmet_nbrs = np.sqrt(np.sum(d_unmet_nbrs**2, axis=1, keepdims=True))
            #breakpoint()
            grad[unmet_idxs] += -(d_unmet_nbrs/dist_unmet_nbrs) #* (dist_unmet_nbrs - dist_farthest_nbr)

            # gradient with respect to farthest neighbor (threshold of neighborhood)
            # TODO not sure  this is working right
            farthest_nbr_idx = proj_idxs[i, -1]
            d_farthest_nbr = srcs[i] - projected[farthest_nbr_idx]
            #grad[farthest_nbr_idx] += d_farthest_nbr/dist_farthest_nbr * np.sum(dist_non_nbrs - dist_unmet_nbrs, axis=0)

            # gradient with respect to source point
            grad[update_idxs[i]] += \
                (np.sum(d_unmet_nbrs/dist_unmet_nbrs - \
                        d_non_nbrs/dist_non_nbrs, 
                        axis=0))
            #* (dist_unmet_nbrs - dist_farthest_nbr) - 
            #* (dist_non_nbrs - dist_farthest_nbr), 

            
        # Update gradient
        grad = (grad*eps + prev_grad*gamma) 

        # Gradient descent step
        projected -= grad

        ajd = self.average_jaccard(projected)

        if verbose:
            # TODO more diagnostic info
            print(f'gradient: {np.sum(grad**2)}')
            print(f'     AJD: {ajd}')

        return projected, grad, ajd

    def fit_transform(self, num_updates=None, eps=1.0, gamma=.9, num_iters=50, verbose=False):
        if verbose:
            start_ajd = self.average_jaccard(self.projected)
            
        projected, grad, ajd = self.step(projected=self.projected, 
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
            #if ajd < best_ajd:
            #    best_ajd = ajd
            #    best_projection = projected

        if verbose:
            print(f'start: average jaccard {start_ajd}')
            print(f'best : average jaccard {best_ajd}')
         
        # TODO fix this
        #return best_projection, best_ajd
        return projected, ajd
            

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

    def animate(self, num_iters, labels, save_file, eps=.25, gamma=.9, verbose=True):
        fig, ax = plt.subplots()
        ax.ignore_existing_data_limits = True
        fig.set_size_inches(6, 6)
        ln = plt.scatter(self.projected[:,0], self.projected[:,1], c=labels, s=3)
        projected = self.projected
        prev_grad = np.zeros((self.num_points, self.dim))

        def init():
            return ln,

        # this is a closure that encloses the projected variable
        # that's necessary to keep updating the newest projection
        def update(frame):
            #fig.gca().autoscale_view()
            pr,gr,ajd = self.step(self.projected, prev_grad=prev_grad, eps=eps, gamma=gamma, verbose=verbose)
            ln.set_offsets(pr)
            ax.update_datalim(ln.get_datalim(ax.transData))
            ax.autoscale_view()
            return ln,

        ani = FuncAnimation(fig, update, frames=range(num_iters),
                            init_func=init, interval=0.001, repeat=False)
        ani.save(filename=save_file, writer='imagemagick')

class JaccardProjectionGradient(JaccardGradient):
    def __init__(self, data, dim, num_nbrs, metric='euclidean', projection='pca'):
        self.low_dim = dim
        self.high_dim = data.shape[1]
        # kept for compatibility with parent JaccardGradient methods
        self.dim = self.high_dim
        self.num_nbrs = num_nbrs
        self.metric = metric
        self.data = data

        self.num_points = data.shape[0]
        balltree = BallTree(data, metric=self.metric, leaf_size=self.num_nbrs)
        _, self.idxs = balltree.query(data, k=self.num_nbrs)

        # project data into lower dimension
        if projection == 'random':
            # subtract .5 to center the random variables at 0
            self.P = np.random.random((self.low_dim, self.high_dim))-.5
            self.PP = self.P.T @ self.P
            self.projected = self.data @ self.P.T
        elif projection == 'pca':
            self.pca = PCA(self.low_dim)
            fitted = self.pca.fit(data)
            # Orthogonal basis of PCA transform
            self.P = fitted.components_
            self.PP = self.P.T @ self.P
            self.projected = self.pca.fit_transform(data)
        else:
            raise Exception(f'The given projection: "{projection}" is not supported. Try "pca" or "random"')

    def step(self, projected, num_updates=None, prev_grad=None, eps=1.0, gamma=.9, verbose=False):
        # determine whether we're updating every point (num_updates = None) or if we're doing stochastic gradient descent
        if num_updates is None:
            update_idxs = np.arange(self.num_points)
            num_updates = self.num_points
        else:
            # stochastic updates
            update_idxs = np.random.choice(self.num_points, size=num_updates, replace=False)

        idxs = self.idxs[update_idxs]
        # rearrange data to match random indexing
        srcs = projected[idxs[:,0]]
        #data = self.data[idxs[:,0]]
        data = self.data

        # find k nearest neighbors and their distances in the projected space
        balltree = BallTree(projected, metric=self.metric, leaf_size=self.num_nbrs)
        proj_dists, proj_idxs= balltree.query(srcs, k=self.num_nbrs)

        grad = np.zeros((self.num_points, self.high_dim))
        # initialize previous gradient on first step
        if prev_grad is None:
            prev_grad = np.zeros((self.num_points, self.high_dim))

        # have to use a for loop because there are a varying number of non neighbors for each point
        # so can't use vectorized numpy operations
        for i,proj_idx in enumerate(proj_idxs):
            # projected neighbors = points in neighborhood in projected space
            # projected neighbors - true_neighbors = non-neighbors in projected neighborhood
            non_nbr_idxs = np.setdiff1d(proj_idx, idxs[i], assume_unique=True)
            # true_neighbors - projected neighbors = true neighbors not in projected neighborhood
            unmet_idxs = np.setdiff1d(idxs[i], proj_idx, assume_unique=True)
            non_nbrs = data[non_nbr_idxs]

            # gradient with respect to non-neighbors in the projected neighborhood 
            d_non_nbrs = data[i] - non_nbrs
            dist_non_nbrs = np.sqrt(np.sum(d_non_nbrs**2, axis=1, keepdims=True)) + .00001
            #breakpoint()
            grad[non_nbr_idxs] += (d_non_nbrs/dist_non_nbrs)

            # gradient with respect to neighbors not in the projected neighborhood 
            d_unmet_nbrs = data[i] - data[unmet_idxs]
            dist_unmet_nbrs = np.sqrt(np.sum(d_unmet_nbrs**2, axis=1, keepdims=True))
            #breakpoint()
            grad[unmet_idxs] += (-d_unmet_nbrs/dist_unmet_nbrs)

            # gradient with respect to farthest neighbor (threshold of neighborhood)
            # TODO not sure  this is working right
            farthest_nbr_idx = proj_idxs[i, -1]
            d_farthest_nbr = data[i] - data[farthest_nbr_idx]
            dist_farthest_nbr = proj_dists[i, -1]
            #breakpoint()
            #grad[farthest_nbr_idx] += d_farthest_nbr/dist_farthest_nbr

            # gradient with respect to source point
            #breakpoint()
            #grad[update_idxs[i]] += \
            #    np.sum(d_unmet_nbrs/dist_unmet_nbrs, axis=0) - \
            #    np.sum(d_non_nbrs/dist_non_nbrs, axis=0)


        #breakpoint()
        grad = (self.PP @ grad.T).T

        # Update gradient
        grad = (grad*eps + prev_grad*gamma) 

        # Gradient descent step
        self.data -= grad
        #projected = self.pca.fit_transform(data)
        projected = self.data @ self.P.T
        self.projected = projected

        ajd = self.average_jaccard(projected)

        if verbose:
            # TODO more diagnostic info
            print(f'gradient: {np.sum(grad**2)}')
            print(f'     AJD: {ajd}')

        return projected, grad, ajd


class DistanceConstraint(JaccardGradient):

    def __init__(self, data, dim, num_nbrs, metric='euclidean', projection='pca'):
        super().__init__(data, dim, num_nbrs, metric, projection)

        self.bndry_dists = self.dists[:,-1]
        # TODO alter boundary distance
        #self.bndry_dists = np.ones((self.num_points)) * 10
        self.data = data

        self.low_dim = dim
        self.high_dim = data.shape[1]
        # kept for compatibility with parent JaccardGradient methods
        self.dim = self.high_dim

        # project data into lower dimension
        if projection == 'random':
            # subtract .5 to center the random variables at 0
            self.P = np.random.random((self.low_dim, self.high_dim))-.5
            self.PP = self.P.T @ self.P
            self.projected = self.data @ self.P.T

    def step(self, projected, num_updates=None, prev_grad=None, eps=1.0, gamma=.9, verbose=False):
        #TODO make this work for stochastic updates
        # determine whether we're updating every point (num_updates = None) or if we're doing stochastic gradient descent
        if num_updates is None:
            update_idxs = np.arange(self.num_points)
            num_updates = self.num_points
        else:
            # stochastic updates
            update_idxs = np.random.choice(self.num_points, size=num_updates, replace=False)

        #TODO make this work for stochastic updates
        idxs = self.idxs[update_idxs]
        srcs = projected[idxs[:,0]]

        grad = np.zeros((self.num_points, self.high_dim))
        # initialize previous gradient on first step
        if prev_grad is None:
            prev_grad = np.zeros((self.num_points, self.high_dim))

        data = self.data
        
        balltree = BallTree(projected, metric=self.metric, leaf_size=self.num_nbrs)
        proj_idxs, proj_dists = balltree.query_radius(projected, self.bndry_dists, return_distance=True)

        total_nbr = 0
        total_non_nbr = 0
        for idx in self.idxs:
            i = idx[0]
            nbrs = np.setdiff1d(idx, proj_idxs[i], assume_unique=True)
            non_nbrs = np.setdiff1d(proj_idxs[i], idx, assume_unique=True)

            boundary = self.bndry_dists[i]

            # TODO check the math on how to calculate distances?
            dvectors = data[i] - data[nbrs]
            pdvectors = projected[i] - projected[nbrs]
            distances = np.sqrt(np.sum(pdvectors**2, axis=1, keepdims=True))
            grad[nbrs] += dvectors/distances * (distances - boundary)

            dvectors = data[i] - data[non_nbrs]
            pdvectors = projected[i] - projected[non_nbrs]
            distances = np.sqrt(np.sum(pdvectors**2, axis=1, keepdims=True))
            grad[non_nbrs] += dvectors/distances * (distances - boundary)

            total_nbr += len(nbrs)
            total_non_nbr += len(non_nbrs)
            
        # Update gradient
        #breakpoint()
        grad = (self.PP @ grad.T).T
        grad = (grad*eps + prev_grad*gamma) 
        # TODO make maxgrad a parameter
        #grad = clip(grad, maxgrad=10)
        projected = data @ self.P.T
        self.projected = projected

        # Gradient descent step
        self.data += grad
        ajd = self.average_jaccard(projected)

        if verbose:
            # TODO more diagnostic info
            print(f' num nbr: {total_nbr}')
            print(f' num non: {total_non_nbr}')
            print(f'gradient: {np.sum(grad**2)}')
            print(f'     AJD: {ajd}')

        return projected, grad, ajd

class ProjectionGradient(JaccardGradient):

    def __init__(self, data, dim, num_nbrs, metric='euclidean', projection='pca'):
        super().__init__(data, dim, num_nbrs, metric, projection)

        self.bndry_dists = self.dists[:,-1]
        # TODO alter boundary distance
        #self.bndry_dists = np.ones((self.num_points)) * 10
        self.data = data

        self.low_dim = dim
        self.high_dim = data.shape[1]
        # kept for compatibility with parent JaccardGradient methods
        self.dim = self.high_dim

        # project data into lower dimension
        if projection == 'random':
            # subtract .5 to center the random variables at 0
            self.P = np.random.random((self.low_dim, self.high_dim))-.5
            self.PP = self.P.T @ self.P
            self.projected = self.data @ self.P.T
            
    def step(self, projected, num_updates=None, prev_grad=None, eps=1.0, gamma=.9, verbose=False):
        #TODO make this work for stochastic updates
        # determine whether we're updating every point (num_updates = None) or if we're doing stochastic gradient descent
        if num_updates is None:
            update_idxs = np.arange(self.num_points)
            num_updates = self.num_points
        else:
            # stochastic updates
            update_idxs = np.random.choice(self.num_points, size=num_updates, replace=False)

        #TODO make this work for stochastic updates
        idxs = self.idxs[update_idxs]
        srcs = projected[idxs[:,0]]

        grad = np.zeros((self.num_points, self.high_dim))
        # initialize previous gradient on first step
        if prev_grad is None:
            prev_grad = np.zeros((self.num_points, self.high_dim))

        data = self.data
        
        balltree = BallTree(projected, metric=self.metric, leaf_size=self.num_nbrs)
        proj_idxs, proj_dists = balltree.query_radius(projected, self.bndry_dists, return_distance=True)

        total_nbr = 0
        total_non_nbr = 0

        grad = np.zeros((self.high_dim, self.high_dim))

        for idx in self.idxs:
            i = idx[0]
            nbrs = np.setdiff1d(idx, proj_idxs[i], assume_unique=True)
            non_nbrs = np.setdiff1d(proj_idxs[i], idx, assume_unique=True)

            boundary = self.bndry_dists[i]

            # TODO check the math on how to calculate distances?
            dvectors = data[i] - data[nbrs]

            # compute sum of outer products of all distance vectors
            # Einsum is dark magic, but I got this to do what I think is right, so...
            grad += np.einsum('ia,ib->ab',dvectors, dvectors)

            dvectors = data[i] - data[non_nbrs]
            grad += -np.einsum('ia,ib->ab',dvectors, dvectors)
        
        #breakpoint()
        grad = (self.P @ grad)*eps
        self.P -= grad*eps

        projected = data @ self.P.T
        self.projected = projected

        # Gradient descent step
        ajd = self.average_jaccard(projected)

        if verbose:
            # TODO more diagnostic info
            print(f' num nbr: {total_nbr}')
            print(f' num non: {total_non_nbr}')
            print(f'gradient: {np.sum(grad**2)}')
            print(f'     AJD: {ajd}')

        return projected, grad, ajd
