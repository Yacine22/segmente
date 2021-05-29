#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:43:02 2020

@author: ymerabet
"""
import dataFusion as df
import pandas as pd
import os

data = df.data


lstfname = []
for fname in os.listdir(df.data1vs):
    lstfname.append(fname.split('npy')[0]) ## liste des nom de features
    
d = {}
for i in range(data.shape[1]-1):
    d[lstfname[i]] = data[:,i]
d["label"] = data[:,-1]

data_frame = pd.DataFrame(data=d)

x_train = data_frame.drop("label", axis=1)
y_train = data_frame["label"]