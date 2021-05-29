#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:51:05 2020

@author: ymerabet
"""
import nibabel as nib 
import numpy as np
from glob import glob 


path_s = '/home/ymerabet/Projet_Covid/CTPRED/Pat-1-5-15/'
liste_patients = glob(path_s+'/*.nii')
liste_patients.sort()

path_an = '/home/ymerabet/Projet_Covid/CTPRED/annot/'
liste_annot = glob(path_an+'/*.nii')
liste_annot.sort()

path_b = '/home/ymerabet/Projet_Covid/CTPRED/boites/'
liste_boites = glob(path_b+'/*.nii')
liste_boites.sort()

path_raw = '/home/ymerabet/Projet_Covid/RF_files/npy/'



def load_nifti2array(): 
    """
    
    Parameters 
        img = nifti ct image
    
    Returns
    -------
    Charger les images nifti des pat001 et pat005 -- les convertir en array

    """
    liste = []
    
    for i in liste_patients : 
        pat_nifti = nib.load(i)
        pat_array = np.array(pat_nifti.dataobj)
        liste.append(pat_array)
    
    j = 0
    for i in liste: 
        np.save(path_raw+liste_patients[j].split('/')[-1].split('.')[0], i)
        j=j+1
    

def load_annotations():
    """
    

    Returns
    -------
    Charger les images des annotations en array

    """
    liste2 = []
    
    for i in liste_annot : 
        pat_a_nifti = nib.load(i)
        pat_a_array = np.array(pat_a_nifti.dataobj)
        liste2.append(pat_a_array)
    
    j = 0
    for i in liste2: 
        np.save(path_raw+liste_annot[j].split('/')[-1].split('.')[0], i)
        j=j+1
    


def load_boites():
    """
    
    Returns
    -------
    Charger les images des masques en array

    """
    
    liste3 = []
    
    
    for i in liste_boites : 
        pat_b_nifti = nib.load(i)
        pat_b_array = np.array(pat_b_nifti.dataobj)
        liste3.append(pat_b_array)
    
    j = 0
    for i in liste3: 
        np.save(path_raw+liste_boites[j].split('/')[-1].split('.')[0], i)
        j=j+1
    

load_nifti2array()
load_annotations()
load_boites()
  