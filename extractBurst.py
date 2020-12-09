#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This program is meant to help build features 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from os import listdir
import os
from os.path import isfile, join
import shutil
from scipy import signal
import sys
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features

def getDeviceName(subdir):
    i = -1
    device = ''
    while abs(i) < len(subdir):
        if not subdir[i] == '\':
            device += subdir[i]
            i -= 1
        else:
            return device[::-1]
        
def getBurstNum(file):
    i = -1
    num = ''
    
    while abs(i) < len(file):
        if not file[i] == '.':
            i -= 1
            
        else:
            i-=1
            while abs(i) < len(file) and file[i] != 't': 
                num += file[i]
                i-=1
                    
            return int(num[::-1])
        
        
def deviceArr(path):
    
    devices = []
    
    subdirs = [f.path for f in os.scandir(path) if f.is_dir() ]
    
    i = 0
    for subdir in subdirs:
        devices.insert(i, getDeviceName(subdir))
        i+=1
    return devices
    
    
if __name__ == '__main__':
    path = r'AsExcel'
    devices = deviceArr(path)
    
    subdirs = [f.path for f in os.scandir(path) if f.is_dir() ]
    
    
    for device in devices:
        subdir = path + '\' + device
        bursts = [f for f in listdir(subdir) if isfile(join(subdir, f))]
        
        size = len(bursts) * 2 # we expect each burst to contain 2 bursts
        
        df = pd.DataFrame(index = range(size), columns = ['Time', 'Amplitude', 'PeakNumber'])
        i = 0
        for file in bursts:
            fileDir = path + '\' + device + '\' + file
            print(fileDir)
            data = pd.read_excel(fileDir)
            data['Device'] = device
            
            data.to_excel(fileDir)


# In[ ]:




