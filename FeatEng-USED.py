#!/usr/bin/env python
# coding: utf-8

# In[8]:



# This program is meant to help build features 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import spline
#from path import Path
from os import listdir
import os
from os.path import isfile, join
import shutil
from scipy import signal
import sys
from tsfresh import extract_features
from tsfresh import feature_extraction
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import extract_features, EfficientFCParameters

backSlash = '/'
noise = -92

class Peak:
    def __init__(self, start, end, height):
        self.start = start
        self.end = end
        self.width = end - start
        self.height = height
    def getStart(self):
        return self.start
    def getEnd(self):
        return self.end
    def getWidth(self):
        return self.width
    def getHeight(self):
        return self.height               
    def __str__(self):
        return "START: " + str(self.start) + "; END: " + str(self.end) + "; WIDTH: " + str(self.width) + "; HEIGHT: " + str(self.height)
    def __repr__(self):
        return str(self) + '\n'


       
def whiskerStDev(df, peak):
     # this method is meant to find the standard deviation of between 
    # the top and bottom end of the whisker. To achieve this, I will 
    # find the center of the peak, and look for the max and min of the 
    # center left and center right of the whisker
    # samples around the center
    
    start = peak.getStart()
    end = peak.getEnd()
    
    center = (start + end) // 2
    
    centerLeft = (center + start)//2
    centerRight = (center + end)//2
    
    inRange = df[df['Time'] > centerLeft]
    inRange = inRange[inRange['Time'] < centerRight]
    
    stdev = inRange['Amplitude'].std()
    
    if math.isnan(stdev):
        raise SystemExit
        return 0
    return stdev
    
    
    

    
def whiskerHeight(df, peak):
    # this method is meant to find the difference between the top 
    # and bottom end of the whisker. To achieve this, I will find 
    # the center of the peak, and look for the max and min of the 
    # center left and center right of the whisker
    # samples around the center

    start = peak.getStart()
    end = peak.getEnd()
    
    center = (start + end) // 2
    
    centerLeft = (center + start)//2
    centerRight = (center + end)//2
    
    inRange = df[df['Time'] > centerLeft]
    inRange = inRange[inRange['Time'] < centerRight]
    
    
    maximum = inRange['Amplitude'].max()
    minimum = inRange['Amplitude'].min()

    if math.isnan(maximum) or math.isnan(minimum):
        raise SystemExit
        return 0
    return maximum - minimum

    
def getDeviceName(subdir):
    i = -1
    device = ''
    while abs(i) < len(subdir):
        if not subdir[i] == backSlash:
            device += subdir[i]
            i -= 1
        else:
            return device[::-1]
        
def deviceArr(path):
    
    devices = []
    
    subdirs = [f.path for f in os.scandir(path) if f.is_dir() ]  
    for subdir in subdirs:
        print(subdir)
    
    i = 0
    for subdir in subdirs:
        devices.insert(i, getDeviceName(subdir))
        i+=1
    return devices

def makeDir(pathWay):
    if os.path.isdir(pathWay):
        shutil.rmtree(pathWay)
        
    os.mkdir(pathWay)
    
def getBurstNum(file):
    i = -1
    num = ''
    
    while abs(i) < len(file):
        if not file[i] == '(':
            i -= 1
            
        else:
            i-=1
            while abs(i) < len(file) and file[i] != 't': 
                num += file[i]
                i-=1
                    
            return int(num[::-1])
        

def mostUseful(fileDirectory):
    allFeatures = pd.read_excel(fileDirectory)
    useful = []
    size = 0
    for feat in allFeatures.columns:
        if size < 200:
            useful.append(feat)
            size+=1
    return useful
    
        
def ts(timeseries):
    
    y = timeseries["Device"]
    y = y.reindex(index = timeseries.index)
    useful = mostUseful('legalFeatures.xlsx')
    extractionSettings = feature_extraction.settings.from_columns(useful)
    extracted_features = extract_features(timeseries, column_id="Device", column_sort="Time", 
                                           impute_function = impute, kind_to_fc_parameters=extractionSettings)
    return extracted_features
    
    
    
    
if __name__ == "__main__":
    # assumptions, there is a directory which has folders labelled by device name. The folder contains amplitude information. 
    # requirements: must be able to pick up the two strongest plateaus
    # will use python peak detection to find out the two or one peaks worth looking at. Will look at the most prominent peaks
    # first.
    path = r'smallBursts'
    savePlotsTo = r'Plots'
    featuresFolder = r'ReplayFeatures'
    samplingRate = 112*(10**6) # sampling rate of collection device in samples/second
    minLength = 40/(10**6) # min length of a bluetooth packet (8us for preamble and 32us for access address) in seconds
    minWidth = minLength * samplingRate #minwidth is 4480samples
    
    features = pd.DataFrame(columns = ['Bursts', 'Device', 'Burst Width','Burst-Top Height',
                                                          'Burst-Top Stdev'])
    mainseries = pd.DataFrame()
    devices = deviceArr(path)
    
    subdirs = [f.path for f in os.scandir(path) if f.is_dir() ]  
    makeDir(savePlotsTo)
    makeDir(featuresFolder)
    
    mainseries = pd.DataFrame()
    for device in devices:
        makeDir(savePlotsTo + backSlash + device) #make folder to save plots to
        
        print(device)
        subdir = path + backSlash + device
        
        bursts = [f for f in listdir(subdir) if isfile(join(subdir, f))]
        
        size = len(bursts) * 2 # we expect each burst to contain 2 bursts
        
        df = pd.DataFrame(index = range(size), columns = ['Bursts', 'Device', 'Burst Width','Burst-Top Height',
                                                          'Burst-Top Stdev'])
        i = 0
        
        for file in bursts:
            
            fileDir = path + backSlash + device + backSlash + file
            print(fileDir)
            data = pd.read_excel(fileDir)
            
            peak = Peak(data['Time'].iloc[0], data['Time'].iloc[-1], data['Amplitude'].mean())
            print(peak)
            burstNum = getBurstNum(file)
            df['Bursts'][i] = burstNum
            df['Device'][i] = device
            df['Burst Width'][i] = peak.getWidth()
            df['Burst-Top Height'][i] = whiskerHeight(data, peak)
            df['Burst-Top Stdev'][i] = whiskerStDev(data, peak)
            
            data = data.reset_index(drop = True) # Has to be here to avoid messing up peak above
            
            timeseries = ts(data)
                
            tsFeatures = list(timeseries)

            for feat in tsFeatures:
                if feat not in df.columns:
                    df[feat] = 'NaaaaN'
                df[feat][i] = timeseries[feat][0]

            mainseries = mainseries.append(data)
            i+=1
                
        df = df.reindex(range(i))
        df.to_excel(featuresFolder + backSlash + device + '.xlsx')
        features = features.append(df)
    
features = features.reset_index(drop = True)
print(features)

features.to_excel(featuresFolder + backSlash + 'Features.xlsx')
            
            #smooth the data


# In[ ]:




