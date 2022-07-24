##### graph representation

import torch
import torch_geometric
import torch_sparse
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
#define a graph

# a graph with 4 nodes

edge_list = torch.tensor([[0,0,0,1,2,2,3,3],[1,2,3,0,0,3,2,0]],dtype= torch.long)
# 1번째 줄이 노드를 시작으로 2번째줄의 노드에 연결됨을 나타냄.
# 0->1 0->2 0->3 ,,,,, 하지만 닫아줘야함 노드가 연결되어있다는 것을 나타냄

# G features for each node (4*6 - number of nodes * number of features)

node_features = torch.tensor([[-8,1,5,8,2,-3],[-1,0,2,-3,0,1],[1,-1,0,-1,2,1],[0,1,4,-2,3,4]],dtype=torch.long)
# define features of nodes(0-3)

# 1 weight for each edge

edge_weight = torch.tensor([[35.],[48.],[12.],[10.],[70.],[5.],[15.],[8.]],dtype=torch.long)

# make a data object to store graph information

data = Data(x=node_features,edge_index = edge_list,edge_attr = edge_weight)

# print the graph infomation

print('num of nodes', data.num_nodes)
print('num of edges', data.num_edges)
#양방향 그래프이기 때문에 edge의 수가 8개이다
print('num of features per node(len of feature vector)', data.num_node_features,'\n')
print('num of weights per node(edge-features)', data.num_edge_features,'\n')

#plot the graph

plt.figure(figsize=(15,15))

G = to_networkx(data)
nx.draw_networkx(G)

plt.show()