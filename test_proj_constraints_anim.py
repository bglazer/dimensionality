import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import JaccardProjectionGradient
from data import load_mnist, projection_example

num_points = 2500
dim = 2

num_iters = 400
eps = .1
gamma = .9
num_nbrs = 10

#TODO random projection doesn't work for this
#data, labels = projection_example(num_points, high_dim=100)
data, labels = load_mnist(num_points = num_points)
data= data.astype('float64')

jg = JaccardProjectionGradient(data, dim=2, num_nbrs=num_nbrs, projection='random')
jg.animate(num_iters, labels, 'constraints_proj_animation.gif', eps, gamma)

