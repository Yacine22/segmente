#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:55:30 2020

@author: ymerabet
"""
import numpy as np
import cv2 as cv
import vesselsValues as vess 
from scipy import ndimage




path_raw = '/home/ymerabet/Projet_Covid/RF_files/npy/'


img1 = np.load(path_raw+'reduced/reduced001.npy') 
#img5 = cv.imread(path_raw+'reduced/array005.npy', 0) 



# size = []
# for i in range(100):
#     s = np.ones((i+20)) 
#     size.append(s)
    
# for j in range(len(size)):    
#     img_dilation1 = cv.dilate(img1, np.array(size[j]), iterations=1) 
#     cv.imshow('Dilation'+str(j), img_dilation1) 
    
    