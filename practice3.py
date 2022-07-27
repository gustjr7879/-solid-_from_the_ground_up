# this file is using deepwalk in karateclub data
# training about Graph Neural Network
# using fastcampus lecture 



import networkx as nx

from karateclub import DeepWalk
import sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # using in node classification
from sklearn.metrics import roc_auc_score
from torch import embedding

# load data

G = nx.karate_club_graph()
print('number of nodes(club-members)',len(G.nodes))

#nx.draw_networkx(G)


#plot the graph with labels

labels = []
for i in G.nodes:
    club_names = G.nodes[i]['club']
    labels.append(1 if club_names == 'Officer' else 0)# clubs : officer is 1 else 0

layout_pos = nx.spring_layout(G)
plt.figure(figsize=(15,15))
nx.draw_networkx(G,pos= layout_pos,node_color = labels,cmap = 'coolwarm')
plt.show()

# DeepWalk

DeepWalk_model = DeepWalk(walk_number=10,walk_length=80,dimensions=124) 
# walk number is how many repeat deepwalk
# walk length is hop number
# dimension is embedding dimension / features dimension

DeepWalk_model.fit(G) # karate club graph into Deepwalk
embedding = DeepWalk_model.get_embedding()
print('embedding array shape (nodes*features)',embedding.shape)


#PCA -> 124 dimension into 2 dimension

PCA_model = sklearn.decomposition.PCA(n_components = 2)
lowdimension_embedding = PCA_model.fit_transform(embedding)
print('Low dimensional embedding representation (nodes*2)',lowdimension_embedding)
plt.scatter(lowdimension_embedding[:,0],lowdimension_embedding[:,1],c=labels,s=15,cmap='coolwarm')
plt.show() # plt.show 2-dimension node embedding

# node classification using embedded model

x_train,x_test,y_train,y_test = train_test_split(embedding,labels,test_size=0.3)
ML_model = LogisticRegression(random_state=0).fit(x_train,y_train)
y_predict = ML_model.predict(x_test)#x test about y
ML_acc = roc_auc_score(y_test,y_predict)

print('acc',ML_acc)