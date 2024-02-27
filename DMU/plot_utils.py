# -*- coding: utf-8 -*-
"""
Lumerical Data Handling
Created on Tue Aug 18 17:06:05 2020
@author: Vidar Flodgren
Github: https://github.com/DeltaMod
"""

#use this to set current directory without running code: os.chdir(os.path.dirname(sys.argv[0]))
import os
import sys
import time
import h5py
import hdf5storage
import matplotlib
import matplotlib as mpl
import tkinter as tk

from tkinter.filedialog import askopenfilename, askdirectory
from matplotlib import patches as ptc
from matplotlib import colormaps as cmaps
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d #If you want to be able to use projection="3D", then you need this:
import scipy
import numpy as np
from scipy import integrate, interpolate, constants
import json
from collections import Counter
import natsort
import csv
import xlrd

#%% Importing and executing logging
import logging
from custom_logger import get_custom_logger
logger = get_custom_logger("DMU_PLOTUTILS")

class CustomFormatter(logging.Formatter):

    magenta  = "\033[1;35m"
    lblue    = "\033[1;34m"
    yellow   = "\033[1;33m"
    red      = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset    = "\x1b[0m"
    format   = "%(asctime)s - %(name)s \n %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: magenta + format + reset,
        logging.INFO: lblue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# create logger with 'spam_application'
logger = logging.getLogger("DMU_UTILS")
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())
if (logger.hasHandlers()):
    logger.handlers.clear()

logger.addHandler(ch)

"""
PLOT TOOLS
"""

def dummy_text_params(string,FIG,fontsize=12,usetex=True,visible=False):
    #This function just guesses based on fontsize, there's no way to do this nicely so we won't try:
    
    dummy = {"width":None,"height":None} 
    HMOD = 2;  WMOD = 0.175;
    dummy["height"] = (HMOD*fontsize)/FIG.fig.dpi
    dummy["width"] = (WMOD * len(string) * fontsize )/FIG.fig.dpi
       
    return(dummy)
    
#Getting Legend entries
def get_combined_legend(FIG):
    leg_handles = []
    leg_labels  = [] 
    for key in FIG.ax.keys():
        lh,ll = FIG.ax[key].get_legend_handles_labels()
        leg_labels.append(ll)
        leg_handles.append(lh)
    
    def flatten(xss):
        return [x for xs in xss for x in xs]
    return(flatten(leg_handles),flatten(leg_labels))

# Automatically adjust tick locations to intervals that do not require two decimal places
def adjust_ticks(ticks):
    nums      = []
    ooms      = []
    
    #First, we find out the highest oom that we will base all our nums on
    for tick in ticks:
        if tick != 0:
            ooms.append(np.floor(np.log10(abs(tick))))
    oom = np.max(ooms)
    
    nums = np.array(ticks)/10**oom
    diffs = np.diff(nums)
    
    def diffnum_converter(diff,nums):
        proxval = [0.1,0.2,0.5,1.0,2.0]
        diffnum = min(proxval, key=lambda x: abs(x - diff))
        new_low =  nums[0] - (nums[0] % diffnum)
        new_high = (nums[-1] + diffnum) - (nums[-1] % diffnum)
        
        if len(np.round(np.arange(new_low-diffnum,new_high+diffnum,diffnum),2)) >=12:
            try:
                diffnum = proxval[proxval.index(diffnum) + 1]
            except:
                diffnum = diffnum*2
                
            new_low =  nums[0] - (nums[0] % diffnum)
            new_high = (nums[-1] + diffnum) - (nums[-1] % diffnum)
        
        new_nums = np.round(np.arange(new_low-5*diffnum,new_high+5*diffnum,diffnum),2)
        return(new_nums)

    new_nums = diffnum_converter(diffs[0],nums)
    
    for num in new_nums:
        new_ticks = new_nums * np.power(10,oom)
        
    return(new_ticks)


def align_axis_zeros(axes):

    ylims_current = {}   #  Current ylims
    ylims_mod     = {}   #  Modified ylims
    deltas        = {}   #  ymax - ymin for ylims_current
    ratios        = {}   #  ratio of the zero point within deltas

    for ax in axes:
        ylims_current[ax] = list(ax.get_ylim())
                        # Need to convert a tuple to a list to manipulate elements.
        deltas[ax]        = ylims_current[ax][1] - ylims_current[ax][0]
        ratios[ax]        = -ylims_current[ax][0]/deltas[ax]
    
    for ax in axes:      # Loop through all axes to ensure each ax fits in others.
        ylims_mod[ax]     = [np.nan,np.nan]   # Construct a blank list
        ylims_mod[ax][1]  = max(deltas[ax] * (1-np.array(list(ratios.values()))))
                        # Choose the max value among (delta for ax)*(1-ratios),
                        # and apply it to ymax for ax
        ylims_mod[ax][0]  = min(-deltas[ax] * np.array(list(ratios.values())))
                        # Do the same for ymin
        ax.set_ylim(tuple(ylims_mod[ax]))        
        
class ScalarFormatterForceFormat(mpl.ticker.ScalarFormatter):
    def __init__(self, useOffset=True, useMathText=False):
        super().__init__(useOffset=useOffset, useMathText=useMathText)
        self._set_format()

    def _set_format(self):
        self.format = "%1.1f"  # Give format here
        
#%%
def DefaultGrid(ax):
    ax.grid(which='major', color='darkgrey', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')  


#%%
class cmap_seq(object):
    """
    Goal of class: use for colourmapping profiles such that:
        - cmap types can be selected
        - The range of values before their full colour contents is selected
        - Alternatively the increment between each cmap value can be selected
        - Invoking the class allows the next value to be given (as to not rely on the for loop i)
        - Allowing for the class to be reset
    """
    def __init__(self):
        self.cmap = None
        self.i    = 0
        
        """
        kwargs:
            ==============
            cmap:  the name of the colormap you wish to use
            steps: the number of increments of the colormap between first and last values
            custom: enables custom mode where colour 1 and colour 2 (and the interp between them) can be seleced
            col1: colour 1
            col2: colour 2
            interp: value for interpolation speed (number of points between col1 and col2)
        """
        
    def set_cmap(self,**kwargs):
        kwargdict = {'cmap':'cmap','colormap':'cmap','colourmap':'cmap',
                     'steps':'steps','step':'steps','N':'steps',
                     'custom':'custom','cust':'custom',
                     'col1':'col1','color1':'col1','colour1':'col1',
                     'col2':'col2','color2':'col2','colour2':'col2',
                     'interp':'interp','interpolation':'interp'}
        kuniq = np.unique(list(kwargdict.keys()))
        kw = KwargEval(kwargs, kwargdict, cmap='viridis',steps=100,custom=False,col1=None,col2=None,interp=None)
        self.istep = 1/kw.steps
        self.cmap = mpl.colormaps(kw.cmap)
        self.col  = self.cmap(self.i)

        
    def iter_cmap(self):
        self.i += self.istep
        self.col  = self.cmap(self.i)
        
    def reset(self):
        self.i = 0
        
   
#%%  
def KwargEval(fkwargs,kwargdict,**kwargs):
    """
    A short function that handles kwarg assignment and definition using the same kwargdict as before. To preassign values in the kw.class, you use the kwargs at the end
    use: provide the kwargs fed to the function as fkwargs, then give a kwarg dictionary. 
    
    Example:
        
        kw = KwargEval(kwargs,kwargdict,pathtype='rel',co=True,data=None)
        does the same as what used to be in each individual function. Note that this function only has error handling when inputting the command within which this is called.
        Ideally, you'd use this for ALL kwarg assigments to cut down on work needed.
    """
    #create kwarg class 
    class kwclass:
        co = True
        pass
    
    #This part initialises the "default" values inside of kwclass using **kwargs. If you don't need any defaults, then you can ignore this.
    if len(kwargs)>0:
        for kwarg in kwargs:
            kval = kwargs.get(kwarg,False)
            try:
                setattr(kwclass,kwarg, kval)
                
            except:
                cprint(['kwarg =',kwarg,'does not exist!',' Skipping kwarg eval.'],mt = ['wrn','err','wrn','note'],co=kwclass.co)
    #Setting the class kwargs from the function kwargs!     
    for kwarg in fkwargs:
        fkwarg_key = kwarg
        if kwarg not in kwargdict.keys():
            kwarg_low = [key.lower() for key in kwargdict.keys()]
            if kwarg in kwarg_low:
                kidx = kwarg_low.index(kwarg)
                kwarg = list(kwargdict.keys())[kidx]
                
        kval = kwargs.get(kwarg,False)
        fkval = fkwargs.get(fkwarg_key,False)
        
        try:
            setattr(kwclass,kwargdict[kwarg], fkval)
            
        except:
            cprint(['kwarg =',kwarg,'does not exist!',' Skipping kwarg eval.'],mt = ['wrn','err','wrn','note'])
    return(kwclass)


#%%Finds all indices where the data set changes direction
def find_turning_points(data):
    
    turning_points = [0]
    try:
        if data[1] > data[0]:
            increasing = True
        else:
            increasing = False
        step_size = abs(data[int(len(data)/4-2)] - data[int(len(data)/4-3)])
        sep_tp = []
        tp_i = 1
        for i in range(1, len(data)-1):
            dx = abs(data[i+1] - data[i])
            if data[i] > data[i+1] and dx > 0.5*step_size:
                if increasing == True:
                    turning_points.append(i)
                    sep_tp.append(turning_points[tp_i] - turning_points[tp_i-1])
                    tp_i += 1
                    increasing = False
                    
            elif data[i] < data[i+1] and dx > 0.1*step_size:
                if increasing == False:
                    turning_points.append(i)
                    sep_tp.append(turning_points[tp_i] - turning_points[tp_i-1])
                    tp_i += 1
                    increasing = True
                    
        if len(turning_points) == 1:
            turning_points = [0,len(data)]
        elif len(turning_points)>1:
            turning_points.append(len(data))
            
        if max(sep_tp) - min(sep_tp)>3:
            turning_points = [0,len(data)]
        return turning_points
    except:
        return turning_points


#%%Takes a list of lists, then returns them split up where the dependent variable changes direction. 

def segment_sweep(L,indices):
    swdat = []
    inds = indices
    slices = [[inds[i],inds[i+1]+1] for i in range(len(inds)-1)]
    
    slices[-1][1] = len(L)
    for slc in slices:
        swdat.append(L[slc[0]:slc[1]])
    return(swdat)

