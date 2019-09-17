import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections
from matplotlib.animation import FuncAnimation
from constraints import JaccardGradient

def load_mnist(num_points):
    data = np.load('data/mnist_data.npy')
    labels = np.load('data/mnist_labels.npy')
    data = data.reshape(-1, 28*28)
    data = data[:num_points]
    labels = labels[:num_points]
    return data,labels


# Random completely separated data
def small_example():
    a = np.random.random((10,2))+10
    b = np.random.random((10,2))
    data = np.concatenate((a,b))
    num_points = 20
    labels = np.ones(20)
    labels[10:] = 2
    return data, labels

num_points = 600
dim = 2

num_iters = 200
eps = 1
num_nbrs = 50

data, labels = load_mnist(num_points = num_points)
#data, labels = small_example()

jg = JaccardGradient(data, dim=2, num_nbrs=num_nbrs, projection='pca')
jg.animate(20, labels)
