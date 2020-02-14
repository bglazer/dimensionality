import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import DistanceConstraint
from data import load_mnist, small_example

num_points = 2500 
dim = 2

num_iters = 500
eps = 1e-4
gamma = .9
num_nbrs = 10

data, labels = load_mnist(num_points = num_points)
data= data.astype('float64')
#data, labels = small_example()

jg = DistanceConstraint(data, dim=2, num_nbrs=num_nbrs, projection='random')
jg.animate(num_iters, labels, 'distance_constraints_proj_animation.gif', eps, gamma)


