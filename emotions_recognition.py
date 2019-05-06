#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intelligent and Interactive Systems 
Spring 2019
Assignment 1, Lingyan Duan
"""
# Import datasets, classifiers and performance metrics
import pandas as pd
import numpy as np
import cv2

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn import manifold, neighbors, metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import LinearSVC

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

'''
Global Variables 
'''
SIMPLE_EMBEDDING=False
MANUAL_SPLIT    =True
DATA_2D         =True

def holdOut(fnData, labelsData, nSamples, percentSplit=0.8):
    '''
    This function splits the data into training and test sets
    '''
    if(MANUAL_SPLIT):
        n_trainSamples = int(nSamples*percentSplit)
        trainData = fnData[:n_trainSamples]
        trainLabels = labelsData[:n_trainSamples]
        
        testData = fnData[n_trainSamples:]
        expectedLabels = labelsData[n_trainSamples:]
    else:
        trainData, testData, trainLabels, expectedLabels = train_test_split(fnData, labelsData,
                                                                            test_size=(1.0-percentSplit), random_state=0)

    return trainData, trainLabels, testData, expectedLabels

def plotData(X, labelsData):
    '''
    This function plots either 2D data or 3D data
    '''
    if(DATA_2D):
        # Convert labels from string into integer, then plot it in different colors
        labelsData = pd.Categorical(pd.factorize(labelsData)[0])
        plt.scatter(X[:,0], X[:,1], c=labelsData, cmap=plt.cm.summer)
        plt.show()
    else:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(X[:,0], X[:,1], X[:,2])
        
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        
        plt.show()

# Load datasets 
data1 = pd.read_csv('dataset/Face-Feat-with-image-file-names.txt', delimiter=",\t", header=None, engine='python')
data1 = data1.dropna(axis=1)
data2 = pd.read_csv('dataset/HogFeat-with-image-file-names.txt', delimiter=",\t", header=None, error_bad_lines=False, engine='python')
data2 = data2.dropna(axis=1)

# Remove commas in last row of data1
data1.iloc[:, 23] = data1.iloc[:, 23].str.strip(",")

# Preprocess the image name before use it as conjuction 
data1.iloc[:, 0] = data1.iloc[:, 0].str.strip("Geo_dis_")
data2.iloc[:, 0] = data2.iloc[:, 0].str.strip("Hog_body_")

# Merge datasets into one
data3 = pd.merge(data1, data2, how="inner", on=[0,1])

# Shuffle DataFrame rows
data3 = data3.sample(frac=1).reset_index(drop=True)

# Remove image file names, seperate feature data and label data
labelsData = data3.iloc[:, 1]
data3 = data3.iloc[:, 2:]

n_samples = len(data3)

# Data Dimension
if(DATA_2D):
    nComp = 2
else:
    nComp = 3

# Data Reduction 
if(SIMPLE_EMBEDDING):
    #  Dimensionality Reduction PCA 
    pca = PCA(n_components = 2)
    X_trans = pca.fit_transform(data3)
else:
    # Manifold embedding with tSNE/Users/Lingyan/Desktop/IIS2019/Assignment 1/emotions_recognition.py
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    X_trans = tsne.fit_transform(data3)
    
# Ploting Data
plotData(X_trans, labelsData)

# Manually Split your data
X_train, X_labels, X_test, X_trueLabels = holdOut(data3, labelsData, n_samples)
print(data3.shape)
print(X_train.shape)
print(X_test.shape)
print(X_labels.shape)
print(X_trueLabels.shape)

n_neighbors = 6
# k-NearestNeighbour Classifier 
kNNClassifier = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")

# Training the model
kNNClassifier.fit(X_train, X_labels)
predictedLabels = kNNClassifier.predict(X_test)

# Display classifier results
print("Classification report for classifier %s:\n%s\n" 
      % ('k-NearestNeighbour', metrics.classification_report(X_trueLabels, predictedLabels)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(X_trueLabels, predictedLabels))

# Cross Validation 
scores = cross_val_score(kNNClassifier, data3, labelsData, cv=5)
print(scores)

# Support Vector Machines
clf_svm = LinearSVC()

# Training 
clf_svm.fit(X_train, X_labels)

# Prediction
y_pred_svm = clf_svm.predict(X_test)
acc_svm = metrics.accuracy_score(X_trueLabels,y_pred_svm)

print ("Linear SVM accuracy: ",acc_svm)

# Cross Validation
scores = cross_val_score(clf_svm, data3, labelsData, cv=5)
print(scores)