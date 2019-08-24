from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d as BN
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
import torch
from torch import tensor, sigmoid
import torch_geometric

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
    def __init__(self, node_size=2, msg_size=128):
        super(NodeModel, self).__init__()
        # TODO layer sizes?
        msg_h1 = 128
        node_h1 = 128
        # 2 = original and projected distances
        # node_size * 2 = original coordinates + projected coordinates
        edge_size = 2 + node_size * 2
        self.msg_mlp = MLP([node_size+edge_size, msg_h1, msg_size])
        self.node_mlp = MLP([node_size+msg_size, node_h1, node_size])

    # features:
    #   node - x, y
    #   edge - orig_dist, proj_x, proj_y, proj_dist_x, proj_dist_y, proj_dist
    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes. F_x is size of node features
        # edge_index: [2, E] with max entry N - 1. E - number of edges in graph
        # edge_attr: [E, F_e], F_e edge feature size
        # u: [B, F_u], F_u size of global feature, B - number of graphs in batch
        # batch: [N] with max entry B - 1.  row, col = edge_index
        #set_trace()
        row, col = edge_index
        msg = torch.cat([x[col], edge_attr], dim=1)
        msg = self.msg_mlp(out)
        # TODO different type of aggregation across node features?
        agg_msg = scatter_mean(src=msg,  # sum mlp transformed inputs
                               index=row, # per node 
                               dim=0,  
                               dim_size=x.size(0)) # make same size as input
         
        out = self.node_mlp(torch.cat([x, out], dim=1))
        
        return out

#class Network(torch_geometric.nn.MetaLayer):
class GraphMap(torch.nn.Module):
    def __init__(self, data, dim=2, num_neighbors=10, num_steps=5)
        super(Network, self).__init__()

        nm = NodeModel(node_size=dim)
        self.model = MetaLayer(node_model=nm)

        self.num_steps = num_steps
        self.dim = dim

        balltree = BallTree(data, metric=squared_euclidean)
        num_neighbors=num_neighbors
        self.dists, self.idxs = balltree.query(data, k=num_neighbors)
        true_distances = torch.as_tensor(dists,dtype=torch.float32)
        true_distances = true_distances.reshape(-1,1)
        max_dist = torch.max(true_distances)
        self.true_distances_norm = true_distances / max_dist

    #   edge - orig_dist, proj_x, proj_y, proj_dist_x, proj_dist_y, proj_dist
    def compute_edges(self, points):
        norms = zeros(points.shape)
        distances = zeros(len(points), 1)
        neighbors = zeros(len(points)*self.num_neighbors, self.dim)
        for i,idxs in enumerate(self.idxs):
            point = points[i]
            n = points[idxs]
            neighbors[i,:] = n
            vectors = neighbors - point
            d = np.sum(vectors**2, dim=1)
            distances[i] = d
            norms[i,:] = vectors / np.sqrt(d)

        return np.concatenate((neighbors, norms, distances))

    def forward(self, x, edge_index, edge_attr=None, u=None, batch=None):
        for step in range(num_steps):
            x, _, _ = self.model.forward(x, edge_index, edge_attr, u, batch)
            orig_dist = edge_attr[:,0]
            new_attr = compute_edges(x)
            edge_attr = np.concatenate((orig_dist, new_attr))

        return edge_attr
        

def squared_euclidean(x, y):
    return np.sum((x-y)**2)

data = datasets.MNIST(root='./data',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)
data = data.data
data = data.type(torch.FloatTensor)
data = Variable(data.view(-1, 28*28))

rows,cols = data.shape

epochs = 100

num_points = 100
num_neighbors = 10
reducer = Reducer(data.data[:num_points],
                  output_size=2, 
                  num_neighbors=num_neighbors,
                  metric_p=2)

#print(torch.cuda.is_available())
#reducer = reducer.cuda()
optimizer = torch.optim.Adam(reducer.parameters(), lr=1e-2)
#optimizer = torch.optim.SGD(reducer.parameters(), lr=.1)#, lr=1e-6)

mse = torch.nn.MSELoss()

def sse(x,y):
    return torch.sum((x-y)**2)

sample_size = 100

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    optimizer.zero_grad()
    projected_distances = reducer.forward()
    if sample_size != 0:
        sample_idxs = torch.randint(reducer.rows*reducer.num_neighbors, (sample_size,))
        projected = projected_distances[sample_idxs]
        true = reducer.true_distances_norm[sample_idxs]
    else:
        projected = projected_distances
        true = reducer.true_distances_norm
    loss = sse(projected, true)
    loss.backward()
    #torch.nn.utils.clip_grad_value_(reducer.parameters(), 10)
    optimizer.step()
    print(loss)

reduced = reducer.reduce()
import pickle
pickle.dump(reduced, open('./data/proj.pickle','wb'))
