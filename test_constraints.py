import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import JaccardGradient
from data import load_mnist, small_example

num_points = 50
dim = 2

num_iters = 100
eps = 1
gamma=.9
num_nbrs = 5

data, labels = load_mnist(num_points = num_points)
#data, labels = small_example()

jg = JaccardGradient(data, dim=2, num_nbrs=num_nbrs, projection='random')
projected, grad = jg.fit_transform(eps=eps, gamma=gamma, num_iters=num_iters, verbose=True)
jg.plot(projected, grad=grad, labels=labels)
