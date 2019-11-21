import numpy as np

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

