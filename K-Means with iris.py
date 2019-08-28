#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#Part I : Please load your dataset, and use K-means to build your model and groups your whole dataset into 10 clusters.

#Part II: Visualize your dataset. Please plot the following figure in your source code.

#Part III: Please evaluate your K-means model performance by using confusion matrix. 
          #Please plot your confusion matrix figure as the following.

#Part IV: Please use Elbow finding method to pick your best K. Please plot the following figure.

#Part V: Please use t-SNE to map your 10-Dimension data to 2D space. 
        #Please plot the following figure in your source code.


# Import pyplot
import matplotlib.pyplot as plt
# Import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

'''
   Part 1: How to build the clustering model
'''
# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters = 3)
# Fit model to points
model.fit(X)
# Determine the cluster labels of new_points : labels
clusters = model.predict(X)

#to convert the random cluster into the correct labels
from scipy.stats import mode
labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    labels[mask] = mode(y[mask])[0]
    
'''
    Part 2: How to evaluate the quality of the clustering
'''
score = accuracy_score(y, labels)
##Confusion Matrix Table
plt.figure()
import seaborn as sns
sns.set()
mat = confusion_matrix(y, labels)
sns.heatmap(mat, square = True, annot = True, cbar = False, cmap = "YlGnBu")
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()

'''
   Part 3 : How to visulize the clustering
'''
# Assign the colums of new_points: xs and ys
xs = X[:,2] #Petal length
ys = X[:,3] #Petal width
# Create a KMeans instance with 3 clusters: model
model = KMeans (n_clusters = 3)
# Fit model to points
model.fit(X[:,[2,3]])
colors = labels.astype(object)
colors[y == 0] = 'red'
colors[y == 1] = 'yellow'
colors[y == 2] = 'blue'
# Make a scatter plot of xs and ys, using labels to define the 
plt.figure()
plt.scatter(xs,ys,c = colors,label=None)
plt.title("Iris clustering analysis")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
# Assign  the cluster centers: centroids
centroids = model.cluster_centers_
#Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, s=100, color='black')
plt.show()

'''
    Part 4 : How to pick the best K
'''
###################################################################
'''
Using only sample and their cluster labels
     A good clustering has tight clusters
     ...and samples in each cluster bunched together
     
Measure how spread out the cluster are (lower is better)
Distance from each sample to centroid of its cluster
After fit(), available as attribute inertia_
KMeans attempt to minimize the inertia when choosing clusters
'''

#How we evaluate the quality of this clustering
ks = range(1, 10)
inertias =[]

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    # Fit model to sample
    model.fit(X)
    # Append the inertia to the list of inertia
    #inertia_ : Sum of squared distance of sample to their closes
    inertias.append(model.inertia_)
    
plt.figure()
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

'''
   Part 5: How to map high-dimension sample to 2D space
#############################################################
t-SNE - "t-distributed stochastic neighbor embedding
Maps samples to 2D space (or 3D)
Map approxmately oreserve nearness of samples
Great for inspecting dataset
'''
from sklearn.manifold import TSNE

model = TSNE(learning_rate= 100)

transformed = model.fit_transform(X)

xs = transformed[:,0]
ys = transformed[:,1]
plt.figure()
plt.scatter(xs,ys,c=colors)
plt.show()
