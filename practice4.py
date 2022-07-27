# node2vec using karate_club dataset

import networkx as nx
from karateclub import DeepWalk
from karateclub import Node2Vec
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# load data set

G = nx.karate_club_graph()
print('number of node in karate_club',len(G.nodes))


labels = []

for i in G.nodes:
    club_names = G.nodes[i]['club']
    labels.append(1 if club_names == 'Officer' else 0)

layout_pos = nx.spring_layout(G)
plt.figure(figsize=(15,15))
nx.draw_networkx(G,pos = layout_pos,node_color = labels,cmap = 'coolwarm')
plt.show()

N2vec_model = Node2Vec(walk_number=10,walk_length=80,p=0.4,q=0.5,dimensions=124)
N2vec_model.fit(G)
N2vec_embedding = N2vec_model.get_embedding()
print('Embedding array shape node*features',N2vec_embedding.shape)

PCA_model = sklearn.decomposition.PCA(n_components = 2)
lowdimension_embedding = PCA_model.fit_transform(N2vec_embedding)
print('Low dimensional embedding representation (nodes*2)',lowdimension_embedding.shape)
plt.scatter(lowdimension_embedding[:,0],lowdimension_embedding[:,1],c=labels,s=15,cmap='coolwarm')
plt.show() # plt.show 2-dimension node embedding


x_train,x_test,y_train,y_test = train_test_split(N2vec_embedding,labels,test_size=0.3,random_state=42)
ML_model = LogisticRegression(random_state=0).fit(x_train,y_train)
y_predict = ML_model.predict(x_test)#x test about y
ML_acc = roc_auc_score(y_test,y_predict)

print('acc',ML_acc)