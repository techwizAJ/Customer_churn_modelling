# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:47:37 2018

@author: techwiz
"""
""" Data Preprocessing"""
#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,[13]].values
#Prepaeing The Dataset
from sklearn.preprocessing import LabelEncoder
label_en_X = LabelEncoder()
X[:,1] = label_en_X.fit_transform(X[:,1])
label_en_X1 = LabelEncoder()
X[:,2] = label_en_X1.fit_transform(X[:,2])
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(categorical_features=[1])
X = one_hot.fit_transform(X).toarray()
X = X[:,1:]
#Scaling the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#making training and test sets
from sklearn.model_selection import train_test_split
X_train , X_test, y_train ,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

""" Making of Neural Net """
#importing keras nad requried modules
import keras
from keras.models import Sequential
from keras.layers import Dense

clf = Sequential()
clf.add(Dense())
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
