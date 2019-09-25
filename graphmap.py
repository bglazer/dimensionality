import torch
import torch_geometric
from torch import tensor, zeros
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN, ModuleList
from torch.autograd import Variable
from torch_scatter import scatter_mean #, scatter_max
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
        dst, src = edge_index
        msg = torch.cat([x[src], edge_attr], dim=1)
        msg = self.msg_mlp(msg)
        # TODO different type of aggregation across node features?
        #agg_msg,_ = scatter_max(src=msg,  # sum mlp transformed inputs
        agg_msg = scatter_mean(src=msg,  # sum mlp transformed inputs
                               index=src, # per node 
                               dim=0,  
                               dim_size=x.size(0)) # make same size as input
         
        out = self.node_mlp(torch.cat([x, agg_msg], dim=1))
        
        return out

class GraphMap(torch.nn.Module):
    def __init__(self, data, node_sizes, num_neighbors=10):
        super(GraphMap, self).__init__()
        
        self.num_neighbors = num_neighbors

        self.layers = ModuleList()
        for i in range(1,len(node_sizes)):
            nm = NodeModel(node_in_size=node_sizes[i-1],
                           node_out_size=node_sizes[i],
                           msg_size=node_sizes[i-1])
            layer = MetaLayer(node_model=nm)
            self.layers.append(layer)

    def compute_edges(self, data):
        self.rows, self.cols = data.shape
        # TODO still need to normalize?
        balltree = BallTree(data, metric=squared_euclidean, leaf_size=num_neighbors)
        self.dists, self.idxs = balltree.query(data, k=num_neighbors)
        # Getting index of neighbors. Exclude the first element, 
        # which is just the query node itself with distance 0
        src = self.idxs[:,1:]
        src = src.reshape(1,-1).squeeze() # reshape to a vector
        # create list of query nodes, repeated for each neighbor
        dst = np.repeat(range(len(data)), num_neighbors-1)
        # concatenate to make COO format
        edge_index = torch.stack((tensor(dst),tensor(src)))

        # Transform distances into vector, normalize
        true_distances = torch.as_tensor(self.dists, dtype=torch.float32)
        true_distances = true_distances[:,1:].reshape(-1,1)
        max_dist = torch.max(true_distances)
        self.true_distances_norm = (true_distances / max_dist).float()
        edge_attr = self.true_distances_norm

        edge_attr = edge_attr.cuda()
        edge_index = edge_index.cuda()
        return edge_attr, edge_index

    def reduce(self, x):
        self.edge_attr, self.edge_index = self.compute_edges(x)
        x = x.cuda()
        for layer in self.layers:
            x, _, _ = layer.forward(x, self.edge_index, self.edge_attr, u=None, batch=None)
        return x

    def forward(self,x):
        self.edge_attr, self.edge_index = self.compute_edges(x)
        x = x.cuda()
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

        del projected
        del x

        return proj_dists
        

def squared_euclidean(x, y):
    return np.sum((x-y)**2)

epochs = 400
num_points = 4000
num_neighbors = 50
sample_size = 50

data = datasets.MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)
labels = data.targets[:num_points]
data = data.data
data = data.type(torch.FloatTensor)
data = Variable(data.view(-1, 28*28))
data = data[:num_points]

# TODO optimize layer number/size
reducer = GraphMap(data,
                   node_sizes=[28**2, 512, 256, 64, 2, 2, 2],
                   num_neighbors=num_neighbors)

reducer = reducer.cuda()

optimizer = torch.optim.Adam(reducer.parameters())

def sse(x,y):
    return torch.sum((x-y)**2)

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    perm = torch.randperm(data.size(0))
    idx = perm[:sample_size]
    samples = data[idx]
    
    optimizer.zero_grad()
    projected = reducer.forward(samples)
    true = reducer.true_distances_norm
    loss = sse(projected, true)
    loss.backward()
    optimizer.step()
    print(f'loss: {loss.item()}')

import pickle
import matplotlib.pyplot as plt

num_slices = int(num_points/sample_size)
reduced = zeros((num_points,2), requires_grad=False)
for i in range(num_slices):
    torch.cuda.empty_cache()
    start = i*sample_size
    end = (i+1)*(sample_size)
    r = reducer.reduce(data[start:end])
    r = r.detach().cpu()
    reduced[start:end] = r
    del r

#labels = pickle.load(open('./data/labels.pickle','rb'))[:num_points]

reduced = reduced.cpu()
p = reduced.detach().numpy()

pickle.dump(p, open('./data/proj.pickle','wb'))

x,y = p[:,0],p[:,1]

plt.figure(figsize=(10,10))
plt.scatter(x,y,c=labels,s=8)
plt.savefig('./data/reduced.png')

bt = BallTree(p, metric=squared_euclidean, leaf_size=num_neighbors)
_,proj_neighbors = bt.query(p, k=num_neighbors)

bt = BallTree(data, metric=squared_euclidean, leaf_size=num_neighbors)
_,true_neighbors = bt.query(data, k=num_neighbors)

# Compare neighborhoods in projected data to true neighborhoods
t = 0
for i in range(num_points):
    for j in range(num_neighbors):
        if true_neighbors[i,j] in proj_neighbors[i]: t+=1
print(t, t/(num_points*num_neighbors))

#random_start = 
#bt = BallTree(r)
#_,proj_neighb_idx = bt.query(r, k=num_neighbors)
#
#t = 0
#for i in range(num_points):
#    for j in range(num_neighbors):
#        if reducer.idxs[i,j] in nn[i]: t+=1
#print(t, t/(num_points*num_neighbors))
