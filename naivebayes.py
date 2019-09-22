import numpy as n
import pandas as p
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as so
from sklearn import datasets
from sklearn.model_selection import train_test_split 
data=datasets.load_wine()
print(type(data))
X_train, X_test, Y_train, Y_test=train_test_split(data.data,data.target, test_size=0.3, random_state=5)

model=GaussianNB()
model.fit(X_train,Y_train)
pred_y=model.predict(X_test)
print(pred_y)
print(so.confusion_matrix(Y_train,pred_y))
