import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import DistanceConstraint
from data import load_mnist, small_example

num_points = 2500
dim = 2

num_iters = 100
eps = 1e-4
gamma=.9
num_nbrs = 5

#data, labels = small_example(200)
data, labels = load_mnist(num_points = num_points)
data= data.astype('float64')

jg = DistanceConstraint(data, dim=2, num_nbrs=num_nbrs, projection='random')
projected, grad = jg.fit_transform(eps=eps, gamma=gamma, num_iters=num_iters, verbose=True)#, num_updates=1)
jg.plot(projected, labels=labels)#grad=grad, 

