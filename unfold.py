import torch
import torch_geometric
from torch import tensor, zeros
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, ModuleList
from torch.autograd import Variable
from torch_scatter import scatter_mean, scatter_add, scatter_max
from torch_geometric.nn import MetaLayer
from torchvision import datasets
from torchvision import transforms
import numpy as np
from sklearn.neighbors import BallTree, DistanceMetric

# TODO make sure this gets removed at some point
from IPython.core.debugger import set_trace

def MLP(channels):
    hidden = Seq(*[
        Seq(Linear(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels)-1)])
    final = Linear(channels[len(channels)-2], channels[len(channels)-1])
    return Seq(hidden, final)


# node_model (Module, optional): A callable which updates a graph's node
#     features based on its current node features, its graph
#     connectivity, its edge features and its global features.
#     (default: :obj:`None`)
class NodeModel(torch.nn.Module):
    def __init__(self, node_in_size=2, node_out_size=2):
        super(NodeModel, self).__init__()
        # TODO layer sizes?
        msg_h1 = 128
        node_h1 = 128
        edge_size = 3
        msg_size = node_in_size+3
        #self.msg_mlp = MLP([node_in_size+edge_size, msg_h1, msg_size])
        self.node_mlp = MLP([node_in_size+msg_size, node_h1, node_out_size])

    # edge_attr = true_dist, dist, dir_x, dir_y
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes. F_x is size of node features
        # edge_index: [2, E] with max entry N - 1. E - number of edges in graph
        # edge_attr: [E, F_e], F_e edge feature size
        # u: [B, F_u], F_u size of global feature, B - number of graphs in batch
        # batch: [N] with max entry B - 1.  row, col = edge_index
        dst, src = edge_index
        msg = torch.cat([x[src], edge_attr], dim=1)
        #msg = self.msg_mlp(msg)
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
        
        self.data = data

        self.rows, self.cols = data.shape

        balltree = BallTree(data, metric=squared_euclidean)
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

        # TODO replace hardcoded 2 with number of dimensions?
        self.random_start = tensor(np.random.normal(loc=0.0, scale=1.0, size=(self.rows, 2)),dtype=torch.float32)

        # TODO still need to normalize?
        true_distances = torch.as_tensor(self.dists, dtype=torch.float32)
        true_distances = true_distances[:,1:].reshape(-1,1)
        max_dist = torch.max(true_distances)
        self.true_distances_norm = true_distances / max_dist
        self.direction = self.random_start[dst] - self.random_start[src]

        #self.edge_attr = tensor(self.true_distances_norm, dtype=torch.float32)
        self.edge_attr = torch.cat(
                            (tensor(self.true_distances_norm, dtype=torch.float32), 
                             tensor(self.direction, dtype=torch.float32)),
                             dim=1)

        self.layers = ModuleList()
        for i in range(1,len(node_sizes)):
            nm = NodeModel(node_in_size=node_sizes[i-1],
                           node_out_size=node_sizes[i])
                           #msg_size=node_sizes[i-1])
            layer = MetaLayer(node_model=nm)
            self.layers.append(layer)
        #self.model = Seq(*self.layers)

    def reduce(self):
        x = self.random_start
        for layer in self.layers:
            x, _, _ = layer.forward(x, self.edge_index, self.edge_attr, u=None, batch=None)
        return x

    def forward(self):
        x = self.random_start
        for layer in self.layers:
            x, _, _ = layer.forward(x, self.edge_index, self.edge_attr, u=None, batch=None)
        projected = x
        #projected, _, _ = self.model.forward(self.data, self.edge_index, self.edge_attr, u=None, batch=None)

        # TODO fix num neighbors to account for self
        proj_dists = zeros([self.rows*(self.num_neighbors-1),1])

        k=0
        idxs = self.idxs[:,1:]
        for i,idx in enumerate(idxs):
            # proj = projected, i.e. a point in the reduced dimensionality space
            # orig = original i.e. a point in the original dimensionality
            proj_neighbors = projected[idx,:]

            proj_point = projected[i,:]

            for j in range(self.num_neighbors-1):
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

epochs = 200

num_points = 100
num_neighbors = 50
# TODO optimize layer number/size
reducer = GraphMap(data.data[:num_points],
                   node_sizes=[2, 2, 2, 2, 2],
                   num_neighbors=num_neighbors)

#device = torch.device(0)
#reducer = reducer.to(device)

optimizer = torch.optim.Adam(reducer.parameters())

def sse(x,y):
    return torch.sum((x-y)**2)

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    optimizer.zero_grad()
    projected = reducer.forward()
    true = reducer.true_distances_norm
    loss = sse(projected, true)
    loss.backward()
    optimizer.step()
    print(loss)

reduced = reducer.reduce()
import pickle
pickle.dump(reduced, open('./data/proj.pickle','wb'))

import matplotlib.pyplot as plt
labels = pickle.load(open('./data/labels.pickle','rb'))[:num_points]

p = reduced.detach().numpy()
x,y = p[:,0],p[:,1]

plt.figure(figsize=(10,10))
plt.scatter(x,y,c=labels,s=8)
plt.savefig('./data/reduced.png')
plt.close()

