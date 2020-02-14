import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import JaccardGradient
from data import load_mnist, small_example

num_points = 500
dim = 2

num_iters = 100
eps = .1
gamma = .9
num_nbrs = 10

data, labels = small_example(num_points)
#data, labels = load_mnist(num_points = num_points)

jg = JaccardGradient(data, dim=2, num_nbrs=num_nbrs, projection='random')
jg.animate(num_iters, labels, 'constraints_animation_random.gif', eps, gamma)
