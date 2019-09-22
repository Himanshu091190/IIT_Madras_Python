from scipy.cluster.hierarchy import dendrogram,linkage
import matplotlib.pyplot as plt
import numpy as n
import pandas as p

data =p.read_csv(r'C:\Users\AKASH\Desktop\sheet physi\word\data2.csv')
print(data)
f1=data['meat'].values
f2=data['Region'].values
plt.rcParams['figure.figsize']=(5,5)
plt.style.use('ggplot')
ar=n.array(list(zip(f1,f2)))
d=linkage(ar,'ward')
dn=dendrogram(d)
b=linkage(ar,'single')
plt.figure(figsize=(5,5))
dn=dendrogram(b)
plt.show()
