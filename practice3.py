##### make convolution network in graph

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.data import Data
import torch.optim as optim
import networkx as nx
from torch_geometric.utils.convert import to_networkx

edge_list = torch.tensor([[0,1,1,1,2,2,3,3],[1,0,2,3,1,3,1,2]],dtype= torch.long)

node_features = torch.tensor([[0,1],[2,3],[4,5],[6,7]],dtype=torch.long)

import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))
data = Data(x=node_features,edge_index=edge_list)
G = to_networkx(data)
nx.draw_networkx(G)
plt.show()

#our first convolution

node_features = torch.arange(8,dtype=torch.float32).view(1,4,2)
adj_matrix = torch.tensor([[[1,1,0,0],[1,1,1,1],[0,1,1,1],[0,1,1,1]]],dtype=torch.float32)
print('node features:',node_features)
print(',adj mat',adj_matrix)

class SimpleConvolution(nn.Module):

    def __init__(self,c_in,c_out):
        super().__init__()
        self.projection = nn.Linear(c_in,c_out)
        # y = xAT + b 선형회귀

    def forward(self,node_features,adj_matrix):
        # Input:
        # node_features - Tensor with node features of shape[batch size, num node]
        # adj_matrix - batch of adj [batch size, num_nodes,num_nodes]

        #num neighbor = num of incoming edges

        num_neighbours = adj_matrix.sum(dim=-1,keepdims = True)
        print('Num of neighbor per node',num_neighbours)
        node_features = self.projection(node_features)
        node_features = torch.bmm(adj_matrix,node_features)#matrix multiplication
        node_features = node_features/num_neighbours
        return node_features

layer = SimpleConvolution(c_in=2,c_out=2)
layer.projection.weight.data = torch.Tensor([[1.,0.],[0.,1.]])# init weight
layer.projection.bias.data = torch.Tensor([0.,0.])#bias term equal to zero

with torch.no_grad():
    out_features = layer(node_features,adj_matrix)

print('adj mat',adj_matrix)
print('input features',node_features)
print('output features',out_features)