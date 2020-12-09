#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This program is meant to help save the bursts in a large dataframe/excel file containing a lot of noise
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from os import listdir
import os
from os.path import isfile, join
import shutil
from scipy import signal
import sys

noise = -80

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

    
def amplitude(val):
    global noise
    if val == '0j' or val == '' or val == '0':
        return noise
    comp = complex(val)
    real = comp.real  #extract real part and convert to a float
    imaginary = comp.imag #remove the space between the sign and number for imaginary and convert to float
    
    refLevDBM = 30
    absComplex = abs(complex(real, imaginary))
    
    try:
        sigDB = 20* math.log(absComplex, 10)
        
    except:
        print("val is " + str(val))
        print("abscomplex is " + str(absComplex))
        print(sigDB)
        sys.exit()
        
    sigDBM = sigDB + 30 - refLevDBM
    
    return sigDBM #complex takes two floats, real and imaginary, and creates a complex number, return that complex number

def imaginary(val):
    comp = complex(val)
    imag = comp.imag
    return imag

def real(val):
    r = complex(val)
    r = r.real
    return r

# what I want to do is write something which tells me when the slope has suddenly changed....
# alternatively I could have something which tells me when the mean amplitude has changed,
# tells me the exact location for the change if the average is above noise for 4000 samples
# add start end and width to possible peaks
def identifyPeaks(data):
    df = data.copy()
    
    numSamples = df.index.max()
    stdScale = 2.5
    
    
    i = 0
    stds = []
    avgAmplitude = df['Amplitude'].mean()
    peaks = []
    
    if numSamples < minWidth:
        print("num samples too few")
        return []
    
    splits = np.array_split(data['Amplitude'], len(data)//roll)
    
    for split in splits:
        stds.append(split.std())
        
    startEnd = np.array(sorted(np.argsort(stds)[-4:])) * roll # Find the points with the four largest standard deviations 
    start1, end1 = startEnd[0], startEnd[1]
    
    if (end1 + roll) - start1 >= minWidth:
        
        center = (start1 + end1) // 2
        centerLeft = (center + start1)//2
        centerRight = (center + end1)//2
        height = df['Amplitude'][centerLeft: centerRight].mean()
        
        if height > avgAmplitude: # If this region has a higher amplitude than the average amplitude, it is probably a burst
            newPeak = Peak(start1, end1 + roll, height)
            peaks.append(newPeak)
    start2, end2 = startEnd[2], startEnd[3]
    if (end2 + roll) - start2 >= minWidth:
        
        center = (start2 + end2) // 2
        centerLeft = (center + start2)//2
        centerRight = (center + end2)//2
        height = df['Amplitude'][centerLeft: centerRight].mean()
        
        if height > avgAmplitude: # If this region has a higher amplitude than the average amplitude, it is probably a burst
            newPeak = Peak(start2, end2 + roll, height)
            peaks.append(newPeak)
        
    print(start1, end1, start2, end2)

    peaks = padPeaks(peaks)
    return peaks

def padPeaks(peaks):
    
    if len(peaks) == 2:
        return peaks
    
    elif len(peaks) == 1:
        zeroPeak = Peak(0, 0, 0)
        peaks.append(zeroPeak)
        return peaks
    
    elif len(peaks) == 0:
        zeroPeak = Peak(0, 0, 0)
        zeroPeak2 = Peak(0, 0, 0)
        peaks.append(zeroPeak)
        peaks.append(zeroPeak2)
        return peaks

def getDeviceName(subdir):
    i = -1
    device = ''
    print(subdir)
    while abs(i) < len(subdir):
        if not subdir[i] == '\\':
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

def makeDir(pathWay):
    if os.path.isdir(pathWay):
        shutil.rmtree(pathWay)
        
    os.mkdir(pathWay)

if __name__ == "__main__":
    # assumptions, there is a directory which has folders labelled by device name. The folder contains amplitude information. 
    # requirements: must be able to pick up the two strongest plateaus
    # will use python peak detection to find out the two or one peaks worth looking at. Will look at the most prominent peaks
    # first.
    path = r'AsExcel'
    savePlotsTo = r'PlotsAll'
    newBursts = r'smallBursts'
    samplingRate = 112*(10**6) # sampling rate of collection device in samples/second
    minLength = 40/(10**6) # min length of a bluetooth packet (8us for preamble and 32us for access address) in seconds
    minWidth = minLength * samplingRate #minwidth is 4480samples
    roll = 100
    
    devices = deviceArr(path)
    print("Running" + str(devices))
    subdirs = [f.path for f in os.scandir(path) if f.is_dir() ]  
    
    makeDir(savePlotsTo)
    makeDir(newBursts)
    
    
    for device in devices:
        makeDir(savePlotsTo + '/' + device) #make folder to save plots to
        makeDir(newBursts + '/' + device)
        
        print(device)
        subdir = path + '/' + device
        
        bursts = [f for f in listdir(subdir) if isfile(join(subdir, f))]
        
        size = len(bursts) * 2 # we expect each burst to contain 2 bursts
        
        df = pd.DataFrame(index = range(size), columns = ['Time', 'Amplitude', 'real', 'imaginary', 'PeakNumber'])
        i = 0
        for file in bursts:
            burstNum = getBurstNum(file)
            fileDir = path + '/' + device + '/' + file
            print(fileDir)
            try:
                data = pd.read_excel(fileDir)
                
            except:
                print("Failed to read " + fileDir)
                continue
            data['real'] = np.nan
            data['imaginary'] = np.nan
            data = data.rename(columns = {0 : 'Amplitude'})
            data['real'] = data['Amplitude'].apply(real)
            data['imaginary'] = data['Amplitude'].apply(imaginary)
            data['Amplitude'] = data['Amplitude'].apply(amplitude) # convert to amplitude
            data = data.rename(columns = {'Unnamed: 0': 'Time'})
            data['Time'] = data.index
            data['Device'] = device
            plt.plot(data['Amplitude'])
            plt.savefig(savePlotsTo + '/' + device + '/' + str(burstNum))
            plt.cla()
            
            peaks = identifyPeaks(data)
            print(peaks)
            toSave = pd.DataFrame(columns = df.columns)
            
#             break
            
            
            label = '(a)'
            for peak in peaks:
                if peak.start == peak.end - roll:
                    continue
                
                toSave = data[peak.start : peak.end ]
                toSave.to_excel(newBursts + '/' + device + '/' + file[:-5] + label + '.xlsx')
                myPlot = toSave['Amplitude']
                plt.plot(myPlot)
                plt.savefig(savePlotsTo + '/' + device + '/' + str(burstNum) + label)
                plt.cla()
                label = '(b)'
                
            


# In[ ]:




