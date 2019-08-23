import torch
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN
from sklearn.neighbors import BallTree, DistanceMetric
from torch import tensor, zeros
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import numpy as np

def MLP(channels):
    return Seq(*[
        Seq(
        Linear(channels[i - 1], channels[i]), 
        ReLU()
        )
        #, BN(channels[i]))
        for i in range(1, len(channels))
    ])

def squared_euclidean(x, y):
    return np.sum((x-y)**2)

class Reducer(torch.nn.Module):
    def __init__(self, data, output_size, num_neighbors=100, metric_p=2):
        super(Reducer, self).__init__()
        # TODO this probably wont work
        self.rows, self.cols = data.shape
        # TODO layer sizes?
        # TODO dropout?
        self.mlp = MLP([self.cols, 128, 64, 32])
        self.out_layer = Linear(32, output_size)
        self.data = data
        self.balltree = BallTree(self.data, metric=squared_euclidean)
        self.num_neighbors=num_neighbors

        dists, self.idxs = self.balltree.query(self.data, k=self.num_neighbors)
        self.true_distances = torch.as_tensor(dists,dtype=torch.float32)
        self.true_distances = self.true_distances.reshape(-1,1)
        
    def forward(self):
        projected = self.mlp(self.data)
        projected = self.out_layer(projected)

        proj_dists = zeros([self.rows*self.num_neighbors,1])

        k=0
        for i,idxs in enumerate(self.idxs):
            # proj = projected, i.e. a point in the reduced dimensionality space
            # orig = original i.e. a point in the original dimensionality
            proj_neighbors = projected[idxs,:]

            proj_point = projected[i,:]

            for j in range(self.num_neighbors):
                proj_neighbor = proj_neighbors[j]
                # TODO something to keep torch from calculating the gradient for 
                # both the point and its neighbor???
                #proj_dist = torch.sqrt(torch.sum((proj_point - proj_neighbor)**2))
                proj_dist = torch.sum((proj_point - proj_neighbor)**2)
                proj_dists[k] = proj_dist
                k+=1

        return proj_dists

data = datasets.MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)
data = data.data
data = data.type(torch.FloatTensor)
data = Variable(data.view(-1, 28*28))
#loader = torch.utils.data.DataLoader(dataset=data, 
#                                     #batch_size=batch_size, 
#                                     shuffle=True)

epochs = 200 

num_points = 1000
num_neighbors = 10
reducer = Reducer(data.data[:num_points],
                  output_size=2, 
                  num_neighbors=num_neighbors,
                  metric_p=2)

optimizer = torch.optim.Adam(reducer.parameters())#, lr=1e-6)

mse = torch.nn.MSELoss()

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    optimizer.zero_grad()
    projected_distances = reducer.forward()
    loss = mse(projected_distances, reducer.true_distances)
    loss.backward()
    #torch.nn.utils.clip_grad_value_(reducer.parameters(), 10)
    optimizer.step()
    print(loss)

