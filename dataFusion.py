#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:57:48 2020

@author: ymerabet
"""
from glob import glob
import numpy as np


### Change path here
data1vs = '/home/ymerabet/Projet_Covid/Random_Forest/features/p1/arrays/'
data5vs = '/home/ymerabet/Projet_Covid/Random_Forest/features/p5/arrays/'

data1nvs = '/home/ymerabet/Projet_Covid/Random_Forest/features/nonVessels/p1/'
data5nvs = '/home/ymerabet/Projet_Covid/Random_Forest/features/nonVessels/p5/'

vs1 = glob(data1vs+'/*.npy')
vs5 = glob(data5vs+'/*.npy')
nvs1 = glob(data1nvs+'/*.npy')
nvs5 = glob(data5nvs+'/*.npy')
##-------------------- --- vessels data --- for patient 1
lv1 = []
for i in vs1:
    npy = np.load(i)
    lv1.append([npy])   
datav1 = np.concatenate(tuple(lv1))
datav1 = np.c_[datav1.T, np.ones(datav1.shape[1])] # with label 1
##-------------------- --- vessels data --- for pat 5
lv5 = []
for i in vs5:
    npy = np.load(i)
    lv5.append([npy])   
datav5 = np.concatenate(tuple(lv5))
datav5 = np.c_[datav5.T, np.ones(datav5.shape[1])] # with label 1
##-------------------- Non vessels data --- for pat 1
lnv1 = []
for i in nvs1:
    npy = np.load(i)
    lnv1.append([npy])   
datanv1 = np.concatenate(tuple(lnv1))
datanv1 = np.c_[datanv1.T, np.zeros(datanv1.shape[1])] #with label 0 
##-------------------- Non vessels data --- for pat 5
lnv5 = []
for i in nvs5:
    npy = np.load(i)
    lnv5.append([npy])  
datanv5 = np.concatenate(tuple(lnv5))
datanv5 = np.c_[datanv5.T, np.zeros(datanv5.shape[1])] #with label 0 
##----------------------------------------------------
data001 = np.concatenate((datav1, datanv1))
data005 = np.concatenate((datav5, datanv5))

data = np.c_[data001.T, data005.T]
data = data.T
