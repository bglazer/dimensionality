import torch
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN
from sklearn.neighbors import BallTree, DistanceMetric
from torch import tensor, zeros
from torchvision import datasets

def MLP(channels):
    return Seq(*[
        Seq(
        Linear(channels[i - 1], channels[i]), 
        ReLU(), 
        BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Reducer(torch.nn.Module):
    def __init__(self, data, output_size, num_neighbors=100, metric_p=2):
        # TODO this probably wont work
        rows, cols = data.shape
        # TODO layer sizes?
        # TODO dropout?
        self.mlp = MLP([input_size, 128, 64, 32, output_size])
        self.data = data
        self.balltree = BallTree(self.data)
        self.metric = DistanceMetric.get_metric('minkowski',metric_p))
        self.num_neighbors=num_neighbors

        self.neighbor_dists = zeros([rows, num_neighbors])
        self.idxs = []
        for i,point in enumerate(self.data):
            dists, idxs = balltree.query(self.data[i,:], k=self.num_neighbors)
            self.neighbor_dists[i,:] = dists
            self.idxs.append(idxs)
        
    def neighbor_loss(self, output):
        loss = 0.0
        for i,idxs in enumerate(self.idxs):
            # proj = projected, i.e. a point in the reduced dimensionality space
            # orig = original i.e. a point in the original dimensionality
            orig_neighbors = self.data[idxs,:]
            proj_neighbors = output[idxs,:]

            orig_point = self.data[i,:]
            proj_point = output[i,:]

            for i in range(self.num_neighbors):
                orig_neighbor = orig_neighbors[i]
                proj_neighbor = proj_neighbors[i]
                orig_dist = orig_point - orig_neighbor
                proj_dist = proj_point - proj_neighbor
                loss += (orig_dist - proj_dist)**2

        return loss
    
    def forward(self, data):
        projected = self.mlp(data)
        return projected

data = datasets.MNIST(root='./data'
                      train=True,
                      transform=transformers.ToTensor(),
                      download=True)

loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                     batch_size=batch_size, 
                                     shuffle=True)

reducer = Reducer(
self.optimizer = torch.optim.Adam(

for 
