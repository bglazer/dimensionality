import torch
import torch_geometric
from torch import tensor, zeros
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN
from torch.autograd import Variable
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torchvision import datasets
from torchvision import transforms
import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric

# TODO make sure this gets removed at some point
from IPython.core.debugger import set_trace

def MLP(channels):
    return Seq(*[
        Seq(
        Linear(channels[i - 1], channels[i]), 
        ReLU(), 
        BN(channels[i]))
        for i in range(1, len(channels))
    ])


# node_model (Module, optional): A callable which updates a graph's node
#     features based on its current node features, its graph
#     connectivity, its edge features and its global features.
#     (default: :obj:`None`)
class NodeModel(torch.nn.Module):
    def __init__(self, node_in_size=2, node_out_size=2, msg_size=128):
        super(NodeModel, self).__init__()
        # TODO layer sizes?
        msg_h1 = 128
        node_h1 = 128
        edge_size = 1
        self.msg_mlp = MLP([node_in_size+edge_size, msg_h1, msg_size])
        self.node_mlp = MLP([node_in_size+msg_size, node_h1, node_out_size])

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes. F_x is size of node features
        # edge_index: [2, E] with max entry N - 1. E - number of edges in graph
        # edge_attr: [E, F_e], F_e edge feature size
        # u: [B, F_u], F_u size of global feature, B - number of graphs in batch
        # batch: [N] with max entry B - 1.  row, col = edge_index
        #set_trace()
        dst, src = edge_index
        msg = torch.cat([x[src], edge_attr], dim=1)
        msg = self.msg_mlp(msg)
        # TODO different type of aggregation across node features?
        agg_msg = scatter_mean(src=msg,  # sum mlp transformed inputs
                               index=src, # per node 
                               dim=0,  
                               dim_size=x.size(0)) # make same size as input
         
        out = self.node_mlp(torch.cat([x, agg_msg], dim=1))
        
        return out

class GraphMap(torch.nn.Module):
    def __init__(self, data, node_sizes, num_neighbors=10):
        super(GraphMap, self).__init__()
        
        self.rows, self.cols = data.shape

        balltree = BallTree(data, metric=squared_euclidean)
        self.num_neighbors=num_neighbors
        self.dists, self.idx = balltree.query(data, k=num_neighbors)
        # Getting index of neighbors. Exclude the first element, 
        # which is just the query node itself with distance 0
        src = self.idx[:,1:]
        src = src.reshape(1,-1) # reshape to a vector
        # create list of query nodes, repeated for each neighbor
        dst = np.repeat(range(len(data)), num_neighbors-1)
        # concatenate to make COO format
        self.edge_index = np.array((dst,src)).T

        self.edge_attr = self.dists[:,1:].reshape(-1,1)
        # TODO still need to normalize?
        true_distances = torch.as_tensor(self.dists, dtype=torch.float32)
        true_distances = true_distances.reshape(-1,1)
        max_dist = torch.max(true_distances)
        self.true_distances_norm = true_distances / max_dist

        self.layers = []
        for i in range(1,len(node_sizes)):
            nm = NodeModel(node_in_size=node_sizes[i-1],
                           node_out_size=node_sizes[i],
                           msg_size=node_sizes[i-1])
            layer = MetaLayer(node_model=nm)
            self.layers.append(layer)

    def forward(self): #x , edge_index, edge_attr=None, u=None, batch=None):
        for layer in self.layers:
            x, _, _ = layer.forward(x, self.edge_index, self.edge_attr, u=None, batch=None)

        proj_dists = zeros([self.rows*self.num_neighbors,1])

        k=0
        for i,idxs in enumerate(self.idxs):
            # proj = projected, i.e. a point in the reduced dimensionality space
            # orig = original i.e. a point in the original dimensionality
            proj_neighbors = projected[idxs,:]

            proj_point = projected[i,:]

            # TODO do this in one shot?
            for j in range(self.num_neighbors):
                proj_neighbor = proj_neighbors[j]
                proj_dist = torch.sum((proj_point - proj_neighbor)**2)
                proj_dists[k] = proj_dist
                k+=1

        return proj_dists
        

def squared_euclidean(x, y):
    return np.sum((x-y)**2)

data = datasets.MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)
data = data.data
data = data.type(torch.FloatTensor)
data = Variable(data.view(-1, 28*28))

epochs = 10

num_points = 100
num_neighbors = 10
# TODO optimize layer number/size
reducer = GraphMap(data.data[:num_points],
                   node_sizes=[28**2, 256, 64, 2],
                   num_neighbors=num_neighbors)

optimizer = torch.optim.Adam(reducer.parameters())

def sse(x,y):
    return torch.sum((x-y)**2)

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    optimizer.zero_grad()
    projected_distances = reducer.forward()
    true = reducer.true_distances_norm
    loss = sse(projected, true)
    loss.backward()
    optimizer.step()
    print(loss)

reduced = reducer.reduce()
import pickle
pickle.dump(reduced, open('./data/proj.pickle','wb'))
