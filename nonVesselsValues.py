#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:16:57 2020

@author: ymerabet
"""
import numpy as np 
import vesselsValues as ves
import random

new_img = '/home/ymerabet/Projet_Covid/RF_files/img_after_features/'

vessels1 = ves.ctVessels_values(ves.index_ves1, ves.p001npy) # vaisseaux du patient 001 
vessels5 = ves.ctVessels_values(ves.index_ves5, ves.p005npy) # vaisseaux du patient 005 

nbr_vess1 = vessels1.shape[0]
nbr_vess5 = vessels5.shape[0]


ind1 = ves.ind1
ind5 = ves.ind5

mask1 = ves.annot_001npy[ind1[0][0]:ind1[0][-1], ind1[1][0]:ind1[1][-1], ind1[2][0]:ind1[2][-1]]
mask5 = ves.annot_005npy[ind5[0][0]:ind5[0][-1], ind5[1][0]:ind5[1][-1], ind5[2][0]:ind5[2][-1]]


index1 = np.argwhere(mask1 == 0)
random_coord1 = random.choices(index1, k=nbr_vess1)

index5 = np.argwhere(mask5 == 0)
random_coord5 = random.choices(index5, k=nbr_vess5)



nv_list1 = []
for i in random_coord1:
    non_ves1 = ves.p001npy[i[0], i[1], i[2]]
    nv_list1.append(non_ves1)
nv1_array = np.array(nv_list1)
np.save(new_img+'non_vessels1/non_vessels1.npy', nv1_array)
    
nv_list2 = []
for i in random_coord5:
    non_ves1 = ves.p001npy[i[0], i[1], i[2]]
    nv_list2.append(non_ves1)
nv5_array = np.array(nv_list2)
np.save(new_img+'non_vessels5/non_vessels5.npy', nv5_array)
