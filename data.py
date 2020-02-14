import numpy as np

def load_mnist(num_points):
    data = np.load('data/mnist_data.npy')
    labels = np.load('data/mnist_labels.npy')
    data = data.reshape(-1, 28*28)
    data = data[:num_points]
    labels = labels[:num_points]
    return data,labels


# Random completely separated data
def small_example(N):
    a = np.random.random((N,2))+10
    b = np.random.random((N,2))
    data = np.concatenate((a,b))
    num_points = N*2
    labels = np.ones(N*2)
    labels[N:] = 2
    return data, labels

# Random completely separated data
def projection_example(N, high_dim):
    a = (np.random.random((N,high_dim))-.5)+10
    b = np.random.random((N,high_dim))-.5
    data = np.concatenate((a,b))
    num_points = N*2
    labels = np.ones(N*2)
    labels[N:] = 2
    return data, labels

