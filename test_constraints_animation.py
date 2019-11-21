import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import JaccardGradient
from data import load_mnist, small_example

num_points = 1500
dim = 2

num_iters = 100
eps = .25
gamma = .9
num_nbrs = 50

data, labels = load_mnist(num_points = num_points)
#data, labels = small_example()

jg = JaccardGradient(data, dim=2, num_nbrs=num_nbrs, projection='pca')
jg.animate(num_iters, labels, 'constraints_animation_2.gif', eps, gamma)
