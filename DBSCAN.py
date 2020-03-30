# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:50:30 2020

@author: suyog
"""

#DBSCAN to classify the clusters and use silhouette score to evaluate our model

#Importing Libraries
import pandas as pd
import numpy as np

#Importing Dataset 
dataset = pd.read_csv('Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Apply DBSCAN alogrithm
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3,min_samples=4)

#Fitting the model
model = dbscan.fit(X)
labels = model.labels_

#Identifying the points which makes up our core points
sample_cores = np.zeros_like(labels,dtype=bool)
sample_cores[dbscan.core_sample_indices_]=True

#calculate the number of clusters
from sklearn import metrics
n_clusters = len(set(labels)) - ( 1 if -1 in labels else 0)

print(metrics.silhouette_score(X,labels)) --> -0.19 score