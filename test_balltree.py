from sklearn.neighbors import BallTree

import numpy as np
from numpy.random import random
#rng = np.random.RandomState(0)
X = random((10, 3))  # 10 points in 3 dimensions
tree = BallTree(X, leaf_size=2)
dist, ind = tree.query(X[:1], k=3)
print(ind)  # indices of 3 closest neighbors

print(dist)  # distances to 3 closest neighbors
