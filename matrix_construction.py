#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 18:10:43 2020

@author: ymerabet
"""
from glob import glob
import numpy as np
import features
import vesselsValues as ves


data1 = '/home/ymerabet/Projet_Covid/RF_files/img_after_features/data1/'
data5 = '/home/ymerabet/Projet_Covid/RF_files/img_after_features/data5/'

data15 = '/home/ymerabet/Projet_Covid/RF_files/img_after_features/p15/'
data15list = glob(data15+'/*.npy')

nvnpy1 = glob(data1+'/*.npy')
vnpy1 = glob(data1+'1/*.npy')


nvnpy5 = glob(data5+'/*.npy')
vnpy5 = glob(data5+'5/*.npy')

##-------------------- Non vessels data --- for pat 001
lnv1 = []
for i in nvnpy1:
    npy = np.load(i)
    lnv1.append([npy])
    
datanv1 = np.concatenate(tuple(lnv1))
datanv1 = np.c_[datanv1.T, np.zeros(datanv1.shape[1])] # label 0

##----------------------------------------------------

##-------------------- --- vessels data --- for pat 001
lv1 = []
for i in vnpy1:
    npy = np.load(i)
    lv1.append([npy])
    
datav1 = np.concatenate(tuple(lv1))
datav1 = np.c_[datav1.T, np.ones(datav1.shape[1])] # label 1
##----------------------------------------------------

##-------------------- Non vessels data --- for pat 005
lnv5 = []
for i in nvnpy5:
    npy = np.load(i)
    lnv5.append([npy])
    
datanv5 = np.concatenate(tuple(lnv5))
datanv5 = np.c_[datanv5.T, np.zeros(datanv5.shape[1])]
##----------------------------------------------------

##-------------------- --- vessels data --- for pat 005
lv5 = []
for i in vnpy5:
    npy = np.load(i)
    lv5.append([npy])
    
datav5 = np.concatenate(tuple(lv5))
datav5 = np.c_[datav5.T, np.ones(datav5.shape[1])] # label 1
##----------------------------------------------------
data001 = np.concatenate((datav1, datanv1))
data005 = np.concatenate((datav5, datanv5))

data = np.c_[data001.T, data005.T]
data = data.T


l15 = []
for i in data15list:
    npy = np.load(i)
    l15.append([npy])
    
dat15 = np.concatenate(tuple(l15)).T
# datanv1 = np.c_[datanv1.T, np.zeros(datanv1.shape[1])]

#features.BuildDataFrame(path_featur_dir=features1, annotations_file = path_raw+'anot/annotations.npy', points_set=None)
