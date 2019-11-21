import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import JaccardGradient
from data import load_mnist, small_example

num_points = 1500
dim = 2

num_iters = 400
eps = .05
gamma=.9
num_nbrs = 10

data, labels = load_mnist(num_points = num_points)
#data, labels = small_example()

jg = JaccardGradient(data, dim=2, num_nbrs=num_nbrs, projection='pca')
projected, grad = jg.fit_transform(eps=eps, gamma=gamma, num_iters=num_iters, verbose=True)
jg.plot(projected, grad=grad, labels=labels)
