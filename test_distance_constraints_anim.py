import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import DistanceConstraint
from data import load_mnist, small_example

num_points = 150
dim = 2

num_iters = 2000
eps = 1e-4
gamma = 0
num_nbrs = 10

data, labels = load_mnist(num_points = num_points)
data= data.astype('float64')
#data, labels = small_example()

jg = DistanceConstraint(data, dim=2, num_nbrs=num_nbrs, projection='random')
jg.animate(num_iters, labels, 'distance_constraints_proj_animation_clipped_boundary_reduced.gif', eps, gamma)


