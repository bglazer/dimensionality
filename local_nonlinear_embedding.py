import torch
import torch_geometric
from torch import tensor, zeros
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, ModuleList, Sigmoid
from torch.nn.functional import softmax, elu, relu
from torch import sigmoid
from torch.autograd import Variable
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_softmax
from torch_geometric.nn import MetaLayer, global_max_pool, GlobalAttention, Set2Set, GATConv
import numpy as np
import random
import pickle
import math
from torchvision import datasets
from torchvision import transforms
from sklearn.neighbors import BallTree, DistanceMetric

def MLP(channels):
    hidden = Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels)-1)])
    final = Linear(channels[len(channels)-2], channels[len(channels)-1])
    return Seq(hidden, final)


class NodeModel(torch.nn.Module):
    def __init__(self, layers):
        super(NodeModel, self).__init__()
        self.mlp = MLP(layers)
    
    def forward(self, src, dest, edge_attr, u, batch):
        return self.mlp(src)


class LNLE(torch.nn.Module):
    def __init__(self, data, num_neighbors, d):
        super(LNLE, self).__init__()

        self.data = data

        self.rows, self.cols = data.shape

        D = self.cols

        # TODO replace with torch_geometric knn
        balltree = BallTree(data)
        self.num_neighbors=num_neighbors
        self.dists, self.idxs = balltree.query(data, k=num_neighbors)
        # Getting index of neighbors. Exclude the first element, 
        # which is just the query node itself with distance 0
        src = self.idxs[:,1:]
        src = src.reshape(1,-1).squeeze() # reshape to a vector
        # create list of query nodes, repeated for each neighbor
        dst = np.repeat(range(len(data)), num_neighbors-1)
        # concatenate to make COO format
        self.edge_index = tensor(np.array((dst,src)), dtype=torch.int64)

        self.projection = MLP([D, 256, 256, d])
        self.graph_conv1 = GATConv(in_channels=d, out_channels=10, heads=4, 
                                      concat=True, negative_slope=0.2, dropout=0, bias=True)
        self.graph_conv2 = GATConv(in_channels=40, out_channels=D, heads=1, 
                                      concat=True, negative_slope=0.2, dropout=0, bias=True)
        self.reconstruction = MLP([400, D*2, D*2, D])

    def forward(self, x):
        x = self.projection(x)
        x = relu(self.graph_conv1(x, self.edge_index))
        x = sigmoid(self.graph_conv2(x, self.edge_index)) * 256
        #x = relu(self.graph_conv2(x, self.edge_index))
        #x = sigmoid(self.reconstruction(x)) * 256
        return x

if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
    
data = datasets.MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)
data = data.data
data = data.type(torch.FloatTensor)
data = Variable(data.view(-1, 28*28))

epochs = 1000

num_points = 1500
num_neighbors = 10

data = data.data[:num_points]

d = 2
lnle = LNLE(data = data, num_neighbors=num_neighbors, d=2)

optimizer = torch.optim.Adam(lnle.parameters())

mse = torch.nn.MSELoss()

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    optimizer.zero_grad()
    projected = lnle.forward(data)
    loss = mse(projected, data)
    loss.backward()
    optimizer.step()
    print(float(loss))

import matplotlib.pyplot as plt
images = np.zeros((28*10,28*2))
for i in range(10):
    im = data[i].reshape(28,28).detach().numpy()
    reconstructed = lnle(data)
    re = reconstructed[i].reshape(28,28).detach().numpy()
    images[28*i:28*i+28,:] = np.concatenate((im,re),1)
    
plt.imshow(images)
plt.show()

labels = pickle.load(open('./data/labels.pickle','rb'))[:num_points]

reduced = lnle.projection(data)
p = reduced.detach().numpy()
x,y = p[:,0],p[:,1]

plt.figure(figsize=(10,10))
plt.scatter(x,y,c=labels,s=8)
plt.show()
#plt.savefig('./data/reduced.png')
#plt.close()

