#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:06:25 2020

@author: ymerabet
"""
import RandomForest as rndf
import patient15test as p15
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


X_test = p15.data15
Y_test = p15.subannot

y_pred = rndf.model.predict(X_test)

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

metrics.plot_roc_curve(rndf.model, X_test, Y_test)  
plt.show()           