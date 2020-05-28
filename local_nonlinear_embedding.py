import torch
import torch_geometric
from torch import tensor, zeros
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, ModuleList, Sigmoid
from torch.nn.functional import softmax, elu, relu, sigmoid
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

        self.reconstruction = GATConv(in_channels=D, out_channels=D, heads=1, 
                                      concat=True, negative_slope=0.2, dropout=0, bias=True)
        self.model = MetaLayer(node_model=self.reconstruction)#, edge_model=self.projection)

    def forward(self, x):
        x = relu(self.reconstruction(x, self.edge_index))#, edge_attr=None, u=None, batch=None)
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

epochs = 100

num_points = 100
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
    print(loss)

import matplotlib.pyplot as plt
images = np.zeros((28*10,28*2))
for i in range(10):
    im = data[i].reshape(28,28).detach().numpy()
    re = projected[i].reshape(28,28).detach().numpy()
    images[28*i:28*i+28,:] = np.concatenate((im,re),1)
    
plt.imshow(images)
plt.show()

print('optimizing reduced dimension points')
transform_epochs = 2000
reduced = torch.rand((data.shape[0], d))
reduced.requires_grad = True
# TODO, change number/size of layers in up projection
#up_projection = MLP([d, 256, 512, 1024, 2048, data.shape[1]])
up_projection = Linear(d, data.shape[1])
projection_optimizer = torch.optim.Adam(up_projection.parameters())
point_optimizer = torch.optim.Adam([reduced], lr=.8)
for epoch in range(transform_epochs):
    print(f'epoch: {epoch}')
    projection_optimizer.zero_grad()
    point_optimizer.zero_grad()
    up = up_projection(reduced)
    projected = lnle.forward(up)
    loss = mse(projected, data)
    loss.backward()
    point_optimizer.step()
    #if epoch%20==0:
    projection_optimizer.step()
    print(float(loss))

##import pickle
##pickle.dump(reduced, open('./data/proj.pickle','wb'))
#
labels = pickle.load(open('./data/labels.pickle','rb'))[:num_points]

p = reduced.detach().numpy()
x,y = p[:,0],p[:,1]

plt.figure(figsize=(10,10))
plt.scatter(x,y,c=labels,s=8)
plt.show()
#plt.savefig('./data/reduced.png')
#plt.close()

