import numpy as n
import pandas as p
import matplotlib.pyplot as plt
import sklearn.metrics as so
from sklearn.cluster import KMeans
from sklearn import datasets

iris=datasets.load_iris()
print(type(iris))
print(iris.data)
print(iris.target)
x=p.DataFrame(iris.data)
print(x)
x.columns=['sepallength','sepalwidth','petallength','petalwidth']
colormap=n.array(['red','lime','black']) # giving color for each 
y=p.DataFrame(iris.target)
print(y)
y.columns=['Target']
plt.figure(figsize=(14,7))
plt.scatter(x.sepallength,x.sepalwidth,c=colormap[y.Target],s=40)
plt.title("Sepal data before model")
plt.show()
plt.figure(figsize=(14,7))
plt.scatter(x.petallength,x.petalwidth,c=colormap[y.Target])
plt.title("petal data before model")
plt.show()
model =KMeans(n_clusters=2)
model.fit(x)
centroids=model.cluster_centers_ #give the centroid 
print("centroids",centroids)
labels=model.labels_  ##labels of result 
print(labels)
plt.figure(figsize=(14,7))
plt.scatter(x.petallength,x.petalwidth,c=colormap[labels])
plt.title("petal data after model")
plt.show()

pred_y=n.choose(labels,[1,0,2]).astype(n.int64)
#print(labels)
print(pred_y)
print(so.accuracy_score(y,pred_y))
print(so.confusion_matrix(y,pred_y))




