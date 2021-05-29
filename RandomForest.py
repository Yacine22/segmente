#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:59:44 2020

@author: ymerabet
"""

import dataframe as data
from sklearn.ensemble import RandomForestClassifier as rf




x = data.x_train 
y = data.y_train 

hyperparameters = {"n_estimators": 100, "criterion":
                           'gini', "max_depth":  None,
                           "min_samples_split": 5,
                           "min_samples_leaf": 2,
                           "max_features": 'auto'}
    

forest = rf(n_estimators = hyperparameters["n_estimators"],
            max_depth = hyperparameters["max_depth"],
            max_features = hyperparameters["max_features"],
            min_samples_leaf = hyperparameters["min_samples_leaf"],
            min_samples_split = hyperparameters["min_samples_split"],
            criterion = hyperparameters["criterion"],
            n_jobs=-1
            )

model = forest.fit(x, y)


import numpy as np
from glob import glob 
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


subvol15 = '/home/ymerabet/Projet_Covid/Random_Forest/features/p15/'
boite15 = np.load('/home/ymerabet/Projet_Covid/Random_Forest/volumes/boite_patient015.npy')
annot15 = np.load('/home/ymerabet/Projet_Covid/Random_Forest/volumes/Segmentation_pat015-label.npy')
nonzero = np.nonzero(boite15)
subannot = annot15[nonzero]

flist = glob(subvol15+'/*.npy')

for i in flist:
    ftr = np.load(i)
    vect = ftr[nonzero]
    name = str(i).split('/')[-1].split('npy')[0]
    np.save(subvol15+'arrays/'+name+'.npy', vect)

    
vs15 = glob(subvol15+'arrays/*.npy')    
l15 = []
for i in vs15:
    npy = np.load(i)
    l15.append([npy])   
data15 = np.concatenate(tuple(l15))
data15 = data15.T

X = np.load()
X_test = data15
Y_test = subannot

y_pred = model.predict(X_test)

mask_annot = y_pred.reshape(80, 87, 92)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

conf_mat = confusion_matrix(Y_test, y_pred)
print(conf_mat)

sns.heatmap(conf_mat, annot=True, fmt="d")


TP = conf_mat[0][0]
FP = conf_mat[0][1]
FN = conf_mat[1][0]
TN = conf_mat[1][1]

print("Sensibilité = ", TP/(TP+FN) )
print("Spécificité = ", TN/(TN+FP) )
print("MCC = ", matthews_corrcoef(Y_test, y_pred))
print("Dice = ", (2*TP)/(2*TP+TN+FP) )

metrics.plot_roc_curve(model, X_test, Y_test)  
plt.show()           
   
