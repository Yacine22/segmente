#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:21:56 2020

@author: ymerabet
"""
import matrix_construction as mtx
import numpy as np
from sklearn.ensemble import RandomForestClassifier as rf
import matplotlib.pyplot as plt
import pandas as pd
import vesselsValues as ves


data = mtx.data

d = {}
for i in range(data.shape[1]-1):
    d["des"+str(i)] = data[:,i]
d["label"] = data[:,-1]

data_frame = pd.DataFrame(data=d)

x = data_frame.drop("label", axis=1)
y = data_frame["label"]

hyperparameters = {"n_estimators": 175, "criterion":
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
            n_jobs=6
            )

model = forest.fit(x, y)


X = ves.subvolume3()


X_test = mtx.dat15
Y_test = ves.annot_pat015

y_pred=model.predict(X_test)

mask_annot = y_pred.reshape(80, 87, 92)


from sklearn import metrics
import seaborn as sns

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

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


