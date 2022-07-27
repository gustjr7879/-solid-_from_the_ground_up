# Node2vec in pytorch

from sklearn.utils import shuffle
import torch
from torch_geometric.datasets import Planetoid # here have some dataset (cora,, else,,,,)
from torch_geometric.nn import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy
# Import data

path = './' # directory to download file
dataset = Planetoid(path,'cora') # in path install 'cora' dataset
data = dataset[0] # 'cora' data files shape
print('cora',data) # see shape

device = 'cuda'

Node2Vec_model = Node2Vec(data.edge_index,embedding_dim= 128, walk_length=20,context_size=10,walks_per_node=10,num_negative_samples=1,p=0.4,q=0.5,sparse=True).to(device)


loader = Node2Vec_model.loader(batch_size = 128,shuffle = True,num_workers =4) # this code is simlilar to fit , load data in batch size
optimizer = torch.optim.SparseAdam(list(Node2Vec_model.parameters()),lr=0.01)


def train():
    Node2Vec_model.train() # set training as true for the model
    total_loss = 0
    for pos_rw,neg_rw in loader:
        pos_rw = torch.as_tensor(pos_rw,device='cuda')
        neg_rw = torch.as_tensor(neg_rw,device='cuda')
        optimizer.zero_grad() # reset of gradient of all variables
        loss = Node2Vec_model.loss(pos_rw,neg_rw) # compute the loss
        loss.backward() # update the gradient
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(loader)

for epoch in range(10):
    loss = train()
    print('epoch:',epoch,'loss:',loss)

# Plot 2D of embedded rep

@torch.no_grad() # deactivate autograd function
def plot_point(colors):
    Node2Vec_model.eval()
    z = Node2Vec_model(torch.arange(data.num_nodes,device=device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy()) # like PCA, node dimension 2 and show node embedding
    y = data.y.cpu().numpy()
    plt.figure()
    for i in range(dataset.num_classes):
        plt.scatter(z[y==i,0],z[y==i,1],s=20,color = colors[i])
    plt.axis('off')
    plt.show()

colors = ['#ffc0cb','#bada55','#008080','#420420','#7fe5f0','#065535','#ffd700']

plot_point(colors)

def test():
    Node2Vec_model.eval() #evaluate the model based on the trained parameters
    z = Node2Vec_model() #evaluate the model based on the trained parameters
    acc = Node2Vec_model.test(z[data.train_mask],data.y[data.train_mask],z[data.test_mask],data.y[data.test_mask],max_iter = 150) # using logistic regression
    return acc

print('acc',test())

