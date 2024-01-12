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


#%%        
def AbsPowIntegrator(Data,x,y,z,WL):
    
    """
    "A function that uses a RectBivariateSpline function to determine the total absorbed power from a pabs_adv lumerical file."
    Calculating total power absorption as a power fraction:s
    Lumerical initially gives P_abs in terms of W/m^3, but then converts it by dividing by the source power - which is why the values are seemingly massive. 
    SP   = meshgrid4d(4,x,y,z,sourcepower(f));
    Pabs = Pabs / SP;
    
    If we then look inside the P_abs_tot analysis group script, we can see that this simply becomes an integration over each 2d slice:
    Pabs_integrated = integrate2(Pabs,1:3,x,y,z);
    We could simply export Pabs_tot, but I think we get more control if we do it manually, and we also save data!
    """
    
    P_tot = []
    for i in range(len(WL)):
        BivarSpline = [np.abs(z[0]-z[1])*scipy.interpolate.RectBivariateSpline(y,x,Data[i,k,:,:]).integral(y[0],y[-1],x[0],x[-1]) for k in range(len(z))]

        P_tot.append(np.sum(BivarSpline))
    
    return(P_tot)

#%%

def bias_plotter(data,FIG,**kwargs): 
    FIG.hold_on = False
    IFIG = False
    IDF = None
    keys = list(data.keys())
    kwargdict = {'fwd':'fwd','f':'fwd','forward':'fwd',
                 'bwd':'rev','rev':'rev','reverse':'rev',
                 'title':'title','tit':'title',
                 'tool':'tool','experiment':'exp','exp':'exp',
                 'ideality':'ideality',
                 'plot':'plot',
                 'altplot':"altplot",
                 'cols':'cols'}
    
    kuniq = np.unique(list(kwargdict.keys()))
    Pkwargs = {}
    #Filtering out the function kwargs from the plot kwargs
    for key in kwargs.keys():
        if key not in kuniq:
            kwargdict[key] = key
    #Collecting kwargs
    kw = KwargEval(kwargs, kwargdict,fwd = True, rev = True, title=None,tool="Nanonis",exp='2IV',cols=None,ideality=False,plot=True,altplot=False)
    xkey = []; ykey = []; fwdbwd = []
    if kw.plot == True:
        ax = FIG.ax[0]
        
    #Separating plot kwargs into a different dict
    for key,val in kwargs.items():
        if key not in kuniq:
            Pkwargs[key] = val
            
    
    if kw.tool == "Nanonis":
        if kw.plot == True:
            for k in keys:
                if "bias" in k.lower():
                    xkey.append(k)
                    
                if "bwd" not in k.lower() and "bias" not in k.lower():
                    ykey.append(k)
                    fwdbwd.append(0) #Note: 0  is forward, 1 is backwards
                
                if "bwd" in k.lower() and "bias" not in k.lower():
                    ykey.append(k)
                    fwdbwd.append(1) #Note: 0  is forward, 1 is backwards
            
            for n,i in enumerate(fwdbwd):    
                plotkwargs = {}
                for k,v in Pkwargs.items():
                    try: 
                        plotkwargs[k] = v[i]
                    except:
                        plotkwargs[k] = v
            
                if kw.fwd == True and i == 0:
                    ax.plot(data[xkey[0]],data[ykey[i]],label='Forward',**plotkwargs)
                    
                if kw.rev == True and i == 1:
                    ax.plot(data[xkey[0]],data[ykey[i]],label='Backward',**plotkwargs)
                
            ax.set_xlabel(xkey[0])
            ax.set_ylabel(ykey[fwdbwd[0]])
            ax.legend()
            ax.set_title(kw.title)
        
        if kw.ideality == True:
        
            tab20 = mpl.colormaps["tab20"]
            IDF = Ideality_Factor(data[ykey[0]],data[xkey[0]])
            
            if kw.plot == True:
                IFIG = True
                IFIG = ezplot()
    
                IFIG.ax[0].plot(IDF['V_new'],IDF['I_new'],'.-',linewidth=2,color=tab20(0),label='ideality='+"{0:.5g}".format(IDF['n']))
                IFIG.ax[0].plot(IDF['V'],IDF['I'],'.',linewidth=2,c=tab20(1),label='IV data')
    
    
                IFIG.ax[0].legend()
            
                
    elif kw.tool == "Keithley":
        keys = data['col headers']
        plotkwargs = {}
        
        for i,ll in enumerate(xkey):
            for k,v in Pkwargs.items():
                try: 
                    plotkwargs[k] = v[i]
                except:
                    plotkwargs[k] = v
                    
        for k in keys:
            if "voltage" in k.lower():
                xkey.append(k)
            elif "current" in k.lower():
                ykey.append(k)
        
        xkey.sort(); ykey.sort()
        if kw.plot == True:
            if "Voltage Linear Sweep" in data["Settings"]["Operation Mode"] and "5_emitter_sweep" not in data["Settings"]["Test Name"]:
                print(data[xkey[0]])
                if len(xkey) == 1 and len(ykey) == 1:
                    for m,data_x in enumerate(data[xkey[0]]): 
                        ax.plot(data[xkey[0]][m],data[ykey[0]][m],label=ykey[0],**plotkwargs)
                
                elif len(xkey) > 1 and len(ykey)>1:
                    for n,key in enumerate(xkey):
                        for m,data_x in enumerate(data[xkey[0]]): 
                            ax.plot(data[xkey[n]][m],data[ykey[n]][m],label=ykey[n],**plotkwargs)
                        
                elif len(xkey) == 1 and len(ykey)>1:
                    for n,key in enumerate(ykey):
                        for m,data_x in enumerate(data[xkey[0]]):
                            ax.plot(data[xkey[0]][m],data[ykey[n]][m],label=ykey[n],**plotkwargs)
                        
                elif len(xkey) > 1 and len(ykey)==1:
                    for n,key in enumerate(xkey):
                        for m,data_x in enumerate(data[xkey[0]]):
                            ax.plot(data[xkey[n]][m],data[ykey[0]][m],label=ykey[0],**plotkwargs) 
                
                    
                    
                ax.set_xlabel(xkey[0])
                ax.set_ylabel(ykey[0])
                ax.legend()
                ax.set_title(kw.title)
            
                if kw.ideality == True:
                    IFIG = False
                    try:
                        
                        tab20 = mpl.colormaps["tab20"]
                        
                        IDF = Ideality_Factor(data[ykey[0]][0],data[xkey[0]][0],T=273,plot_range =None,fit_range = None,N=100,p0=None)
                        if kw.plot == True:
                            IFIG = ezplot()
            
                            IFIG.ax[0].plot(IDF['V_new'],IDF['I_new'],'.-',linewidth=1,color=tab20(0),label='ideality='+"{0:.5g}".format(IDF['n']))
                            IFIG.ax[0].plot(IDF['V'],IDF['I'],'.',linewidth=2,c=tab20(1),label='IV data')
            
            
                            IFIG.ax[0].legend()
                        
                    except:
                        None
                    
            elif "Voltage List Sweep" in data["Settings"]["Operation Mode"]:
                if kw.cols == None:
                    cols = {"IE":[cmaps["tab20c"](0),cmaps["tab20c"](1)],
                            "VE":[cmaps["tab20c"](4),cmaps["tab20c"](5)],
                            "ID":[cmaps["tab20c"](8),cmaps["tab20c"](9)]}
                if kw.plot == True:
                    emitter  = data['emitter']
                    detector = data['detector']
                        
                    #We want to show: emitter voltage on both top and bottom plots, and only detector current on the top plot. 
                    for key in data.keys():
                        if emitter['NWID'] in key:
                            if "voltage" in key:
                                Em_V_key = key
                            if "current" in key:
                                Em_I_key = key
                        
                        if detector['NWID'] in key:
                            if "voltage" in key:
                                Det_V_key = key
                            if "current" in key:
                                Det_I_key = key
                            
                    Em_V     =   data[Em_V_key][0]
                    Em_I     =   data[Em_I_key][0]
                    Det_V     =  data[Det_V_key][0]
                    Det_I     =  data[Det_I_key][0]
                    #We want all currents to show "foward current", so we must check if the abs(min) > abs(max)
                    
                    
                    
                    
                    if abs(np.max(Det_I)) < abs(np.min(Det_I)):
                        Det_I = -np.array(Det_I)
                    
                    if abs(np.max(Em_I)) < abs(np.min(Em_I)):
                        if Em_V(Em_I.index(np.min(Em_I))) < 0:
                            Em_V = -np.array(Em_V)
                            
                        Em_I = -np.array(Em_I)
                        
                        
                    if kw.altplot == False:
                        FIG.fig,(FIG.ax[0],FIG.ax[1]) = plt.subplots(nrows=2, sharex=True)
                        fig, ax_top, ax_bottom = [FIG.fig,FIG.ax[0],FIG.ax[1]]
                        ax_top.spines["bottom"].set_visible(False)
                        ax_bottom.spines["top"].set_visible(False)
                        ax_top.tick_params(bottom=False)
                        ax_bottom.tick_params(top=False)
                        plt.subplots_adjust(hspace=0.1)
                
                        ax_top_r = ax_top.twinx()
                        ax_bottom_r = ax_bottom.twinx()
                        
                        axyy  = [ax_top,ax_bottom]
                        axxy  = [ax_top_r,ax_bottom_r]
                    
                        
                        ax_top.plot(Det_I,label='Detector Current [I]',color=cols["ID"][1],**plotkwargs)
                        ax_top_r.plot(Em_V,label='Emitter Voltage [V]',color=cols["VE"][1]**plotkwargs)
                        ax_top.set_ylabel('$I_{Detector}$ [I]',color=cols["ID"][0])
                        ax_top_r.set_ylabel('$V_{Emitter}$ [V]',color=cols["VE"][0])
                        
                        ax_bottom.plot(Em_I,label='Emitter Current [I]',color=cols["IE"][1],**plotkwargs)
                        ax_bottom_r.plot(Em_V,label='Emitter Voltage [V]',color=cols["VE"][1],**plotkwargs)
                        
                        ax_bottom.set_ylabel('$I_{Emitter}$ [I]',   color  = cols["IE"][0])
                        ax_bottom_r.set_ylabel('$V_{Emitter}$ V [V]',color = cols["VE"][0])
                        
                        
                            #Fix colours
                        if len(Pkwargs['c'])<4:
                            Pkwargs['c'] = mpl.colormaps["tab20"]
                        for axis in axyy:
                            for i,line in enumerate(axis.get_lines()):
                                line.set_color(Pkwargs['c'](i))
                        for axis in axxy:
                            for i,line in enumerate(axis.get_lines()):
                                line.set_color(Pkwargs['c'](i+3))
                        ax_bottom.set_xlabel("index")
                        ax_top.set_xlabel("index")
                        handles1,labels = ax_top.get_legend_handles_labels()
                        handles2,labels = ax_top_r.get_legend_handles_labels()
                        fig.legend(handles=handles1+handles2,labels=['$I_{NW}$','$V_{NW}$'])
             
                        FIG.ax[0].set_title(kw.title)
                        
                    if kw.altplot==True:

                        FIG.fig,FIG.ax[0] = plt.subplots()
                        FIG.fig.subplots_adjust(right=0.75)
                        FIG.ax[1] = FIG.ax[0].twinx()
                        FIG.ax[2] = FIG.ax[0].twinx()
                        FIG.ax[2].spines.right.set_position(("axes", 1.25))
                        
                        p1, = FIG.ax[0].plot(Det_I,label='Detector Current [I]',color=cols["ID"][1],**plotkwargs)
                        p2, = FIG.ax[1].plot(Em_I,label='Emitter Current [I]',color=cols["IE"][1],**plotkwargs)
                        p3, = FIG.ax[2].plot(Em_V,'-.',label='Emitter Voltage [V]',color=cols["VE"][1],linewidth = 1)
                        
                        FIG.ax[0].set_xlabel("Index")
                        FIG.ax[0].set_ylabel("$I_{Detector}$ [I]" ,color=cols["ID"][0])
                        FIG.ax[1].set_ylabel('$I_{Emitter}$ [I]'  , color=cols["IE"][0])
                        FIG.ax[2].set_ylabel('$V_{Emitter}$ V [V]',color=cols["VE"][0])
                        
                        #We want to set axis limits so that Voltage = 90% of the ylim
                        def rpad(data,ratio):
                            drange = np.max(data)-np.min(data)
                            dpad   = (drange*(1/ratio) - drange)/2
                            return([np.min(data)-dpad,np.max(data)+dpad],dpad)
                        
                        Vlim,Vpad = rpad(Em_V,0.9)
                        FIG.ax[2].set_ylim(Vlim)
                        
                        #This is the ratio of the positive axis over the negative one such that the currents fit underneath the voltage every time 
                        RatioMod = (np.max(Em_V) + Vpad)/(Vlim[1] - Vlim[0]) 
                        FIG.ax[0].set_ylim(rpad(Det_I,0.8*RatioMod)[0])
                        FIG.ax[1].set_ylim(rpad(Em_I,0.85*RatioMod)[0])
                        
                        
                        
                        align_axis_zeros([FIG.ax[0],FIG.ax[1],FIG.ax[2]])
                        
                    
                        #Setting Spine colours and tickparameters
                        FIG.ax[0].yaxis.label.set_color(cols["ID"][0])
                        FIG.ax[1].yaxis.label.set_color(cols["IE"][0])
                        FIG.ax[2].yaxis.label.set_color(cols["VE"][0])
                        
                        tkw = dict(size=4, width=1.5)
                        FIG.ax[0].tick_params(axis='y', colors=cols["ID"][0], **tkw)
                        
                        FIG.ax[1].tick_params(axis='y', colors=cols["IE"][0], **tkw)
                        FIG.ax[1].spines["right"].set_color(cols["IE"][0])
                        
                        
                        FIG.ax[2].tick_params(axis='y', colors=cols["VE"][0], **tkw)
                        FIG.ax[0].tick_params(axis='x', **tkw)
                        
                        FIG.ax[0].legend(handles=[p1, p2, p3],fontsize=7)
                        
                        for ax in FIG.ax:
                            FIG.ax[ax].spines["left"].set_color(cols["ID"][0])
                            FIG.ax[ax].yaxis.grid(False, which='both')
                            FIG.ax[0].yaxis.grid(True, which='both',color=cmaps["tab20c"](11))
            
            elif "Voltage Linear Sweep" in data["Settings"]["Operation Mode"] and "5_emitter_sweep" in data["Settings"]["Test Name"]:
                FIG.hold_on = True
                if "iteration" not in FIG.keys():
                    FIG.iteration = 0
                    
                if kw.cols == None:
                    cols = [cmaps["Set3"](i) for i in range(12)]
                    
                if kw.plot == True:
                    emitter  = data['emitter']
                    detector = data['detector']
                        
                    #We want to show: emitter voltage on both top and bottom plots, and only detector current on the top plot. 
                    for key in data.keys():
                        if emitter['NWID'] in key:
                            if "voltage" in key:
                                Em_V_key = key
                            if "current" in key:
                                Em_I_key = key
                        
                        if detector['NWID'] in key:
                            if "voltage" in key:
                                Det_V_key = key
                            if "current" in key:
                                Det_I_key = key
                            
                    Em_V     =   data[Em_V_key][0]
                    Em_I     =   data[Em_I_key][0]
                    Det_V     =  data[Det_V_key][0]
                    Det_I     =  data[Det_I_key][0]
                    #We want all currents to show "foward current", so we must check if the abs(min) > abs(max)
                    
                    if abs(np.max(Det_I)) < abs(np.min(Det_I)):
                        Det_I = -np.array(Det_I)
                    
                    if abs(np.max(Em_I)) < abs(np.min(Em_I)):
                        if Em_V(Em_I.index(np.min(Em_I))) < 0:
                            Em_V = -np.array(Em_V)
                            
                        Em_I = -np.array(Em_I)
                    Det_Vmean = np.mean(Det_V)
                    FIG.ax[0].plot(Em_V,Det_I,color=cols[FIG.iteration],label="$V_{Detector}=$"+f'{Det_Vmean:.3f}')
                    FIG.ax[0].set_xlabel("V_{Emitter}")
                    FIG.ax[0].set_ylabel("$I_{Detector}$ [I]")
                    FIG.ax[0].legend()
                
        return(IFIG,IDF)

#%%
def Single_NW_Diagram_Plotter(fig,ax,NanonisDat):
    """
    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    labels : list: [port 1, port 2, port 3, port 4]
    types : list: [bias, sweep, ground, ground]
    plots a nanowire diagram
    """ 
    arrow_style = {
    "head_width": 0.1,
    "head_length": 0.2,
    "color":"k"}
    axobjs = []
    Device = NanonisDat['device']
    Portd =  [NanonisDat['Port'+n] for n in ['1','2','3','4']]
    
    if "Comment06" in list(NanonisDat.keys()):
        comment = NanonisDat["Comment06"]
    
    dxW = 0.05
    dyH = 0.05
    
    nwl = 0.065
    nwh = 0.018
    nwp = 0.003
    c   = [0.825,0.175]
    gctf = 1
    
    gcbl = gctf*nwl/3
    gcbh = gctf*nwl/2
    gcsl = nwh*gctf/3
    gcsh = gctf*nwl
    
    ##Nanowire Creation
    nw1 = ax.add_patch(ptc.Rectangle(
    (c[0]-nwp-nwl, c[1]-nwh/2), # lower left point of rectangle
    nwl, nwh,   # width/height of rectangle
    facecolor="purple", edgecolor="darkorchid",alpha=1,zorder=2,transform=fig.transFigure
    ))
    labels = []
    types  = []
    
    for i,port in enumerate(Portd):
        if port['pos'] != None:
            labels.append('Port'+str(i+1))
            if port['current'] == True:
                types.append('current')
            
            elif port['bias'] != False: 
                types.append('bias')
            
            elif port['sweep'] != False:
                types.append('sweep')
            
            elif port['ground'] == True:
                types.append('ground')
            
            if "tip" in port['NW'] and port['pos'] == 2:
                gp1_pos = (c[0]-nwp, c[1])
            elif "tip" in port['NW'] and port['pos'] == 1:
                gp1_pos = (c[0]-nwp-nwl, c[1])

    gp1 = ax.add_patch(plt.Circle(gp1_pos, radius=nwh/2, facecolor="gold",edgecolor='orange',transform=fig.transFigure))
    
    axobjs = [nw1,gp1]
    
    portlocs      = [(c[0]-nwp-nwl-gcbl, c[1]),(c[0]+gcbl, c[1])]
    porttextloc   = [(c[0]-nwp-nwl-4*gcbl, c[1]),(c[0]+gcbl, c[1])]
    typetextloc   = [(c[0]-nwp-nwl-4*gcbl, c[1]-1.3*gcbl),(c[0]+gcbl, c[1]+1.3*gcbl)]
    # porttextloc   = [(c[0]-nwp-nwl-4.8*gcbl, c[1]),(c[0]-nwp-nwh*gctf+1/2*gcsl, c+gcsh),(c[0]+nwp+nwh*gctf-1/2*gcsl, c[1]5-gcsh),(c[0]+nwp+nwl+2*gcbl, c[1])]
    for i,lab in enumerate(labels):
        axobjs.append(ax.annotate(labels[i],
                  xy=portlocs[i],
                  xytext=porttextloc[i], verticalalignment='center',
                  xycoords = fig.transFigure, textcoords=fig.transFigure))
        
        typecolour = {'ground':"brown",'current':"blue",'bias':"green",'sweep':'red'}
    
        axobjs.append(ax.annotate(types[i],
                  xy=portlocs[i],
                  xytext=typetextloc[i], verticalalignment='center', color=typecolour[types[i].strip()],
                  xycoords = fig.transFigure, textcoords=fig.transFigure))
        #,arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth=5,headlength=5)

    
def Nanowire_Diagram_Plotter(fig,ax,NanonisDat):
    """
    Parameters
    ----------
    ax : TYPE
        DESCRIPTION.
    labels : list: [port 1, port 2, port 3, port 4]
    types : list: [bias, sweep, ground, ground]
    plots a nanowire diagram
    """ 
    arrow_style = {
    "head_width": 0.1,
    "head_length": 0.2,
    "color":"k"}
    axobjs = []
    Device = NanonisDat['device']
    Portd =  [NanonisDat['Port'+n] for n in ['1','2','3','4']]
    if "Comment06" in list(NanonisDat.keys()):
        comment = NanonisDat["Comment06"]
    
    dxW = 0.05
    dyH = 0.05
    
    nwl = 0.035
    nwh = 0.009
    nwp = 0.003
    c   = [0.75,0.25]
    gctf = 2
    
    gcbl = gctf*nwl/3
    gcbh = gctf*nwl/2
    gcsl = nwh*gctf/3
    gcsh = gctf*nwl
    
    ##Nanowire Creation
    nw1 = ax.add_patch(ptc.Rectangle(
    (c[0]-nwp-nwl, c[1]-nwh/2), # lower left point of rectangle
    nwl, nwh,   # width/height of rectangle
    facecolor="purple", edgecolor="darkorchid",alpha=1,zorder=2,transform=fig.transFigure
    ))
    labels = np.empty(4,dtype=object)
    types  = np.empty(4,dtype=object)
    
    for i,port in enumerate(Portd):
        
        labels[port['pos']-1] = 'Port'+str(i+1)
        if port['current'] == True:
            types[port['pos']-1] = 'current'
        
        elif port['bias'] != False: 
            types[port['pos']-1] = 'bias'
        
        elif port['sweep'] != False:
            types[port['pos']-1] = 'sweep'
        
        elif port['ground'] == True:
            types[port['pos']-1] = 'ground'
        
        if "tip" in port['NW'] and port['pos'] == 2:
            gp1_pos = (c[0]-nwp, c[1])
        elif "end" in port['NW'] and port['pos'] == 2:
            gp1_pos = (c[0]-nwp-nwl, c[1])
            
        if "tip" in port['NW'] and port['pos'] == 3:
            gp2_pos = (c[0]+nwp+nwl, c[1])
        elif "end" in port['NW'] and port['pos'] == 3:
            gp2_pos = (c[0]+nwp, c[1])
        
    gp1 = ax.add_patch(plt.Circle(gp1_pos, radius=nwh/2, facecolor="gold",edgecolor='orange',transform=fig.transFigure))
    
    nw2 = ax.add_patch(ptc.Rectangle(
    (c[0]+nwp, c[1]-nwh/2), # lower left point of rectangle
    nwl, nwh,   # width/height of rectangle
    facecolor="purple", edgecolor='darkorchid',alpha=1, zorder=2,transform=fig.transFigure
    ))
    
    gp2 = ax.add_patch(plt.Circle(gp2_pos, radius=nwh/2,facecolor="gold",edgecolor='orange',transform=fig.transFigure))
    
    gc1 = ax.add_patch(ptc.Rectangle(
    (c[0]-nwp-nwl-gcbl+1/8*nwl, c[1]-gcbh/2), # lower left point of rectangle
    gcbl, gcbh,   # width/height of rectangle
    facecolor="gold", edgecolor='orange',alpha=1, zorder=2,transform=fig.transFigure
    ))
    
    gc2 = ax.add_patch(ptc.Rectangle(
    (c[0]-nwp-nwh*gctf, c[1]-gctf*nwl/2), # lower left point of rectangle
    gcsl,gcsh-1/3*gcsh,   # width/height of rectangle
    facecolor="gold", edgecolor='orange',alpha=1, zorder=2,transform=fig.transFigure
    ))
    
    gc3 = ax.add_patch(ptc.Rectangle(
    (c[0]+nwp+nwh*gctf-gcsl, c[1]-gctf*nwl/2+1/3*gcsh), # lower left point of rectangle
    gcsl,gcsh-1/3*gcsh,   # width/height of rectangle
    facecolor="gold", edgecolor='orange',alpha=1, zorder=2,transform=fig.transFigure
    ))
    
    gc4 = ax.add_patch(ptc.Rectangle(
    (c[0]+nwp+nwl-1/8*nwl, c[1]-gcbh/2), # lower left point of rectangle
    gcbl, gcbh,   # width/height of rectangle
    facecolor="gold", edgecolor='orange',alpha=1, zorder=2,transform=fig.transFigure
    ))
    
    axobjs = [nw1,gp1,nw2,gp2,gc1,gc2,gc3,gc4]
    
    portlocs      = [(c[0]-nwp-nwl-gcbl, c[1]),(c[0]-nwp-nwh*gctf+1/2*gcsl, c[1]+1/3*gcsh),(c[0]+nwp+nwh*gctf-1/2*gcsl, c[1]-1/3*gcsh),(c[0]+nwp+nwl+gcbl, c[1])]
    porttextloc   = [(c[0]-nwp-nwl-4*gcbl, c[1]),(c[0]-nwp-nwh*gctf, c[1]-0.75*gcsh),(c[0]+nwp+nwh*gctf-gcsl, c[1]+0.85*gcsh),(c[0]+nwp+nwl+2*gcbl, c[1])]
    typetextloc   = [(c[0]-nwp-nwl-4*gcbl, c[1]-1.3*gcbl),(c[0]-nwp-nwh*gctf, c[1]-0.75*gcsh-1.3*gcbl),(c[0]+nwp+nwh*gctf-gcsl, c[1]+0.85*gcsh+1.3*gcbl),(c[0]+nwp+nwl+2*gcbl, c[1]+1.3*gcbl)]
    # porttextloc   = [(c[0]-nwp-nwl-4.8*gcbl, c[1]),(c[0]-nwp-nwh*gctf+1/2*gcsl, c+gcsh),(c[0]+nwp+nwh*gctf-1/2*gcsl, c[1]5-gcsh),(c[0]+nwp+nwl+2*gcbl, c[1])]
    for i,lab in enumerate(labels):
        axobjs.append(ax.annotate(labels[i],
                  xy=portlocs[i],
                  xytext=porttextloc[i], verticalalignment='center',
                  xycoords = fig.transFigure, textcoords=fig.transFigure))
        
        typecolour = {'ground':"brown",'current':"blue",'bias':"green",'sweep':'red'}
    
        axobjs.append(ax.annotate(types[i],
                  xy=portlocs[i],
                  xytext=typetextloc[i], verticalalignment='center', color=typecolour[types[i].strip()],
                  xycoords = fig.transFigure, textcoords=fig.transFigure))
        #,arrowprops=dict(facecolor='black', shrink=0.05,width=1,headwidth=5,headlength=5)


#%%
def Bulk_CSV_Load(filename):
    if ".csv" not in filename:
        filename = filename+'.csv'
    key_value = np.loadtxt(filename, delimiter=",")
    return({ k:v for k,v in key_value })

#%%    
def CUV(**kwargs):
    """
    Change_User_Variables -- or CUV -- is a function used to save and load user defined variables at the start, and then at the end, of any session.
    Parameters
    ----------
    **kwargs : 
        [act,action,a]              : 
            ['reset','r','res'] - fully replaces the current DataImportSettings.json default file with default settings. This action cannot be undone
            ['load','l']        - loads a specific file. This function opens up a file dialog for selection, so you don't need to add anything else. This also saves the location to Aux_File.
            ['init','i','initialise'] - initialises your file with the current DataImportSettings. It will load Aux_File if the field is not None
            ['sesh','save session','session'] - requires a data kwarg field with a dictionary listed. It will accept ANY dictionary, and save this to the currently active DataImportSettings file (or Aux_File, if loaded)
            ['ddir','data dir','directories'] - will allow you to select a new data directories file. If the file does not exist, you can save it as a new file by writing a new name for it. 
            
        [co, console, console out]  = Select if console output is set to [True/False]
        [path, pathtype, pt]        = Choose path type preference ['rel','abs']. Selecting 'rel' will save the directory of selected files in using a relative address, but only if it can! It the start of the address does not match the current working directory, absolute address will be used automatically.
        [data, dat, d]              = Specify DataImportSettings data <type: Dict>. Must be included in act='sesh' and 'save' (when implemented), but is ignored otherwise. 

    Returns 
    -------
    Dictionary data saved to DataImportSettings.json or Aux_File indicated within DataImportSettings.json!

    """
    S_ESC = LinWin()
    kwargdict = {'act':'act','action':'act','a':'act',
                 'co':'co','console':'co','console out':'console',
                 'path':'pathtype','pathtype':'pathtype','pt':'pathtype',
                 'data':'data','dat':'data','d':'data'}
    
    actdict = {'reset':'reset','r':'reset','res':'reset',
               'l':'load','load':'load',
               'i':'init','init':'init','initialise':'init',
               'sesh':'session','save session':'session','session':'session',
               'ddir':'ddir','data dir':'ddir','directories':'ddir'}
    
    ACTLIST = list(np.unique(list(actdict.values())))
    kw =  KwargEval(kwargs,kwargdict,pathtype='rel',co=True,data=None,act=None)
    if len(kwargs) == 0:
        kw.act = str(input(cprint(['Please enter one of the following actions',' [', ",".join(ACTLIST),']'],mt=['note','stat'],tr=True)))
        if kw.act not in ['reset','load']:
           cprint('Ignoring command, you did not select a valid entry',mt='err',co=kw.co)
           
    try:
        kw.act = actdict[kw.act]
    except:
        cprint(['Note that','kw.act = ',str(kw.act),' does not correspond to an action!',' Skipping kwarg eval.'],mt = ['wrn','err','wrn','note'])
    
   
    #list all acceptable "get/set" inputs - consider using .lower() in the future to remove duplicates/case sensitivity - I think, however, we won't do this! 
    #Instead, import to variable using init or load - change that variable - then save using session.
    getdict = {'debug':'Debug','Debug':'Debug',
               'FL':'File_Load','File_Load':'File_Load','fileload':'File_Load','file_load':'File_Load',
               'DF':'Default_File','default':'Default_File','default_file':'Default_File','Default_File':'Default_File',
               'DDF':'Data_Directories_File','data_directories_file':'Data_Directories_File','data_directory':'Data_Directories_File','ddf':'Data_Directories_File',
               'console':'Console_Output','CO':'Console_Output','Console_Output':'Console_Output',
               'txt':'txt_import','text_import':'txt_import','TI':'txt_import'}
    
    
    
    #Give default filename and try to load default data
    RFile = "DataImportSettings.json"
    try:
        ddata =  jsonhandler(f = RFile,pt=kw.pathtype,a='r')
    except:
        cprint('You don\'t have any default data! Run Init_LDI() or use CUV(act=\'reset\') to reset to default!',mt='err')

    
    #We make sure to check if we have provided data! If we have, we will check ddata['Alt_File'] and write to the correct file.
    if kw.data != None:
        if ddata['Alt_File'] !=None:
            Target_File = ddata['Alt_File']
            
        else:
            Target_File = RFile
    if kw.act == 'reset':
        print(kw.pathtype)
        cprint('Writing default settings to file',mt='note',co=kw.co)
        Default = {"Debug": True, "File_Load": True, "Alt_File": None, "Default_File": RFile,"Data_Directories_File":"DataDirectories.json", "Console_Output": True, "txt_import": True}    
        jsonhandler(f = Default['Default_File'],pt=kw.pathtype,d = Default,a='w')
        return(jsonhandler(f=RFile,pt=kw.pathtype,a='r'))
          
    if kw.data == None:
        kw.data = jsonhandler(f = RFile ,pt=kw.pathtype,a='r')
        
        if kw.data['Alt_File'] is not None:
            Target_File = kw.data['Alt_File']
            kw.data = jsonhandler(f = kw.data['Alt_File'] ,pt=kw.pathtype,a='r')
        else:
            Target_File  = RFile
    

    
    if kw.act == 'session':
            try:
                cprint(['Saving user set settings to path = ',Target_File],mt=['note','stat'],co=kw.co)
                jsonhandler(f = Target_File,pt=kw.pathtype, d = kw.data, a='w')
                
            except:
                cprint(['Alt_File failed, setting user set settings to path = ',ddata['Default_File']],mt=['wrn','stat'],co=kw.co)
                jsonhandler(f = RFile,pt=kw.pathtype, d = kw.data, a='w')

       
    if kw.act == 'load':
        root = tk.Tk()
        file_path = askopenfilename(title = 'Select a settings file',filetypes=[('json files','*.json'),('All Files','*.*')]).replace('/',S_ESC)    
        tk.Tk.withdraw(root)
        file_path,kw.pathtype = Rel_Checker(file_path) 
        if file_path != "":
            kw.data = jsonhandler(f = file_path,pt=kw.pathtype, a='r')
            if ddata['Alt_File'] == None:
                ddata['Alt_File'] = Rel_Checker(file_path)[0]
                jsonhandler(f = RFile,pt=kw.pathtype, d = ddata, a='w')  
            return(kw.data)
        else:
            cprint("Cancelled file loading",mt='note',co=kw.co)
        
    if kw.act == 'init':
            try:
                kw.data = jsonhandler(f = Target_File,a='r')
                cprint(['Loading user set settings from path = ',Target_File],mt=['note','stat'],co=kw.co)
                return(kw.data)
            except:
                cprint(['Failed to load alt user settings file, using defaults instead'],mt=['err'],co=kw.co)
                return(ddata)
        
        
    
    if kw.act == 'ddir':
        RFile = "DataImportSettings.json"
        ddata = jsonhandler(f = RFile,pt=kw.pathtype,a='r')
        root = tk.Tk()
        file_path = askopenfilename(title = 'Select or write in a new Data Directories file',filetypes=[('json files','*.json'),('All Files','*.*')])
        tk.Tk.withdraw(root)
        
        file_path,kw.pathtype = Rel_Checker(file_path) 
        kw.data['Data_Directories_File'] = file_path
        
        if ddata['Alt_File'] is not None:
            try:
                cprint(['Saving user set settings to path = ',ddata['Alt_File']],mt=['note','stat'],co=kw.co)
                
                jsonhandler(f = ddata['Alt_File'],pt=kw.pathtype, d = kw.data, a='w')
                
            except:
                cprint(['Alt_File failed, setting user set settings to path = ',ddata['Default_File']],mt=['wrn','stat'],co=kw.co)
                jsonhandler(f = ddata['Default_File'],pt=kw.pathtype, d = kw.data, a='w')
        else:
            cprint(['Writing current user settings to path = ',PathSet(ddata['Default_File'],pt=kw.pathtype)],mt=['note','stat'],co=kw.co)
            jsonhandler(f = ddata['Default_File'],pt=kw.pathtype, d = kw.data, a='w')  
 
#%%
def coltxt_read(filename,**kwargs):
    """
    **kwargs :
        cn/colnames/columns - optional list of strings that contain names for all colums - will be used in dict item to store the variables afterwards
        dl/delimiter/delim - if known, provide the delimeter for the data - will make the function faster, but it should still run 
        dt/datatype        - If you have string data only, you need to type in "string", otherwise it's not going to find your delimeter properly. Default is "mixed"
    Returns
    -------
    

    """
    
    kwargdict = {'cn':'cn','colnames':'cn','columns':'cn',
                 'delimiter':'dl','delim':'dl','dl':'dl',
                 'datatype':'dt','dt':'dt',
                 'cu':'cleanup','cleanup':'cleanup','clean':'clean'}
    #Collecting kwargs
    kw = KwargEval(kwargs, kwargdict, cn = [],dl=None,dt='mixed',cleanup = True)
    
    if kw.dl is None:
        #If the delimeter is None, then we need to guess which delimiter it could be
        #TLDR: check if any of ['\t', '\s' , ',' , ';'] split to match the total number of columns
        with open(filename,'r') as f:
            fline = f.readline()
            
            #First, we check if the total number of strings is higher than the total number of integers
            #This is done to check what we need to do in order to guess the correct delimeter.
            if kw.dt == "mixed":
                chars = 0
                nums  = 0
                for letter in fline:
                    try:
                        int(letter)
                        nums  += 1
                    except:
                        chars +=1
                
                if chars>nums:
                    fline = f.readline()

                    
            esc_chars =  ['\t', ' ' , ',' , ';']
            for esc in esc_chars:
                char,num = maxRepeating(fline,guess=esc)
                line = fline
                if char == esc and num!=1:
                    esc = "".join([esc for i in range(num)])
                
                if "\n" in line:
                    line = line.split('\n')[0]

                    
                line = line.split(esc)
                
                if line[-1] == '':
                    line.pop(-1)
                    
                if len(line) == 1:
                    try:
                        int(line)
                        colnum = 1
                    except:
                        None
                if len(line)>1:
                    try:
                        for splt in line:
                            test = float(splt)
                        colnum = len(line)
                        kw.dl = esc
                        
                    except:
                        None
    
    try:
        data = np.genfromtxt(filename,delimiter=kw.dl)
    except ValueError:
        cprint('Delimeter was probably not found automatically, please provide one next time you run this command',mt='err')
    
    colnum = data.shape[1]
    rownum = data.shape[0]

    if np.isnan(data[:,-1]).all():
        colnum -= 1
        data = np.delete(data, -1, 1)
    
    if np.isnan(data[0,:]).all():
        rownum -=1
        data = np.delete(data, 0, 0)
    
    
        
    with open(filename,'r') as f:
        if kw.dl is not None:
            line = f.readline().split(kw.dl)
            if len(line) == colnum + 1:
                del(line[-1])
            if len(line) == colnum:
                colnames = line
                    
    dictnames = []
    dictnum   = {}
    if len(kw.cn) !=0:
        dictnames = kw.cn
        if len(dictnames) != colnum:
            for i in range(len(dictnames),colnum):
                dictnames.append(str(i))
    else:
        if len(max(colnames))<5:
            dictnames = colnames
            unames    = set(dictnames)
            
            if len(dictnames) != len(unames):
                for name in unames:
                    dictnum[name] = 1
                for i in range(len(dictnames)):
                    dictnum[dictnames[i]] += 1
                    dictnames[i] = dictnames[i]+'_'+str(dictnum[dictnames[i]]-1)
                    
        elif len(max(colnames))>5:
            dictnames = [str(i) for i in range(colnum)]
            
    data_out = {}
    data_out['colnames'] = colnames
    
    for i in range(data.shape[1]):
        data_out[dictnames[i]] = data[:,i]
    
    if kw.cleanup == True:
        for key in data_out.keys():
            if type(data_out[key]) == np.ndarray:
                data_out[key] = data_out[key][~np.isnan(data_out[key])]
        
    return(data_out)
  
  
#%%    
def cprint(String,**kwargs):
    """
    WARNING: The format of this script's kwarg importing is severely different from all other functions in this sheet - consider revising to make use instead of the kwargdict and assigning variables through that!
    Note that some light colour variations do not work on all terminals! To get light colours on style-inedpendent terminals, you can use ts = bold!
    kwargs:
    
    mt: Message type - a string that defines one from a list of preset message formats. 
        List of acceptable entries: ['err','error','note','warning','caution','wrn','curio','status','stat','custom']. Note: only custom supports further kwargs
    fg: Foreground colour - a string with the full name or abbrviation of a colour to give to the text.
        List of acceptable entries: ['black','k','red','r','green','g','orange','o','blue','b','purple','p','cyan','c','lightgrey','lg',
                                     'darkgrey','dg','lightgreen','lgr','yellow','y','lightblue','lb','pink','pk','lightcyan','lc']
                                    Note that some light colours are accessed using bold style instead of this!
    bg: Background colour - a string with the full name or abbrviation of a colour to highlight text with.
        List of acceptable entries: ['black','k','red','r','green','g','orange','o','blue','b','purple','p','cyan','c','lightgrey','lg']
    ts: Text Style - a string indicating what style to apply to the text. Some do not work as intended.
        List of acceptable entries: ['bold','b','italic','it','underline','ul','strikethrough','st','reverse','rev','disable','db','invisible','inv']
    sc: StartCode - A custom startcode if you want to use specific colours not listed in the code. 
        Note: overwrites any bs/fg inputs, is compatible only with "custom" message type, but supports text style ts kwargs!
    jc: Join character - This is the character that will join the strings in a list together, recommend '\n' or ' ' but anything works 
    cprint also supports lists with different styles and options applied. Use:
        cprint([string1,string2],fg = [fg1,fg2],bg = [bg1,bg2],ts = [ts1,ts2])
    tr: textreturn - returns the escape character strng instead - does not produce a print output!
    co: console output - a global variable if you want an option to disable console ouput throughout your code!
        list of acceptable entries: [True,False], default: False
    """
    kwargdict = {'mt':'mt','message type':'mt','message':'mt',
                 'fg':'fg', 'foreground':'fg',
                 'bg':'bg', 'background':'bg',
                 'ts':'ts', 'text style':'ts', 'style':'ts',
                 'sc':'sc', 'start code':'sc',
                 'jc':'jc', 'join character':'jc', 'join char':'jc',
                 'tr':'tr', 'text return':'tr', 'return':'tr',
                 'co':'co', 'console':'co','console output':'co'}
    
    kw = KwargEval(kwargs,kwargdict,mt='custom',fg=None,bg=None,ts=None,sc=None,jc=' ',tr=False,co=True)
    
    #We convert all of these to lists to make sure that we can give the software strings or lists without any problems
    if type(String) == str:
        String    = [String]
    
    def ListMatcher(SoL,matchwith):
        if type(SoL) == type(matchwith) == list:
            if len(SoL) != len(matchwith):
                TM = ['' for i in range(len(matchwith))]
                for j in range(len(SoL)):
                    if j<len(matchwith):
                        TM[j] = SoL[j]
                    
                for k in range(len(SoL),len(matchwith)):
                    TM[k] = SoL[-1]
                SoL = TM
                
        elif type(SoL) != type(matchwith):
            if type(SoL) == str:
                SoL = [SoL for i in range(len(matchwith))]
            if SoL == None:
                SoL = [None for i in range(len(matchwith))]
        
        return(SoL)
    kw.mt    = ListMatcher(kw.mt,String)
    kw.fg    = ListMatcher(kw.fg,String)
    kw.bg    = ListMatcher(kw.bg,String)
    kw.ts    = ListMatcher(kw.ts,String)
    kw.sc    = ListMatcher(kw.sc,String)
    kw.jc    = ListMatcher(kw.jc,String) 
  
    reset ='\033[0m'
   
    EXITCODE  = reset
    #Note: These can eventually be generated from a script - but probably still need manual definition. Consider to replace with KwargEval in the future, but it's fine for now! 
    class ts:
        bold          = b   ='\033[01m'
        italic        = it  = '\33[3m'
        disable       = db  = '\033[02m'
        underline     = ul  = '\033[04m'
        reverse       = rev = '\033[07m'
        strikethrough = st  = '\033[09m'
        invisible     = inv = '\033[08m'
    
    class fg:
        white      =  w  = '\33[37m'
        black      =  k  = '\033[30m'
        red        =  r  = '\033[31m'
        green      =  g  = '\033[32m'
        orange     =  o  = '\033[33m'
        blue       =  b  = '\033[34m'
        purple     =  p  = '\033[35m'
        cyan       =  c  = '\033[36m'
        lightgrey  =  lg = '\033[37m'
        darkgrey   =  dg = '\033[90m'
        lightred   =  lr = '\033[91m'
        lightgreen = lgr = '\033[92m'
        yellow     =  y  = '\033[93m'
        lightblue  =  lb = '\033[94m'
        pink       =  pk = '\033[95m'
        lightcyan  =  lc = '\033[96m'
        
    class bg:
        white     =  w  = '\33[47m'
        black     =  k  = '\033[40m'
        red       =  r  = '\033[41m'
        green     =  g  = '\033[42m'
        orange    =  o  = '\033[43m'
        blue      =  b  = '\033[44m'
        purple    =  p  = '\033[45m'
        cyan      =  c  = '\033[46m'
        lightgrey = lg  = '\033[47m'
    
    #Message preset function
    class mps: 
        err  = error =            fg.red+ts.bold
        note =                    fg.cyan+ts.bold
        wrn = warning = caution = fg.orange+ts.bold
        status = stat =           fg.green+ts.bold
        curio  =                  fg.purple+ts.bold
        frun   = funct =          bg.c+fg.o
    
    PRINTSTR = []
    for i in range(len(String)):
        STARTCODE = ''
        if kw.mt[i] == 'custom':    
            
            try: 
                style = getattr(ts,kw.ts[i])
            except:
                if kw.ts[i] is not None:
                    cprint(['Attribute ts =',str(kw.ts[i]),'does not exist - reverting to default value'],mt='err')
                style = ''
            
            try:
                 STARTCODE = STARTCODE + getattr(fg,kw.fg[i])
            except:
                if kw.fg[i] is not None:
                    cprint(['Attribute fg =',str(kw.fg[i]),'does not exist - reverting to default value'],mt='err')
                STARTCODE = STARTCODE 
                
            try:
                 STARTCODE = STARTCODE + getattr(bg,kw.bg[i])
            except:
                if kw.bg[i] is not None:
                    cprint(['Attribute bg =',str(kw.bg[i]),'does not exist - reverting to default value'],mt='err')
                STARTCODE = STARTCODE 
            
            if kw.sc[i] is not None:
                STARTCODE = kw.sc[i]
            STARTCODE = STARTCODE+style
        else:
            try:
                STARTCODE = getattr(mps,kw.mt[i])
            except:
                cprint(['Message preset', 'mt = '+str(kw.mt[i]),'does not exist. Printing normal text instead!'],mt = ['wrn','err','wrn'])
   
        PRINTSTR.append(STARTCODE+String[i]+EXITCODE+kw.jc[i])
    if kw.co == True:     
        if kw.tr == False:
            print(''.join(PRINTSTR))
        else:
            return(''.join(PRINTSTR))

          
#%%
def DataDir(**kwargs):
    """
    Function to handle loading new data from other directories - should be expanded to support an infinitely large list of directories, by appending new data to the file.
    Note: to change currently active data-dir, you need to select a new file in CUV. I'm going to set up a function that allows you to both select a file, and to make a new one! 
    
    What to do here? I'm saving a file with directories, and I'm giving an option to set the save location in a different directory, isn't that a bit much?
    Maybe I should just have the option in CUV to select a new DataDirectories file, and let this one only pull the directory from CUV?
    
    Current implementation:
        UVAR = CUV(act='init') means UVAR now contains all your variables, and to save you would do CUV(d=UVAR,act = 'session'), which keeps all changes and additions you made to UVAR.
        If you want to add a data directory using DataDir, it too will use CUV(act='init') to load the file, but this does not take into account any changes made in UVAR.
        Solution: Add CUV(act='data_dir') to add a new empty .json file with a particular name, or to select a previously created data_dir file, and make that file the new 
        UVAR['Data_Directories_File']. 
        
        How do you make sure that UVAR is updated properly? 
        Current solution is to give a cprint call telling you to load UVAR again if you make this change, or to return the newly edited file with the function...
    """ 
    S_ESC = LinWin()
    kwargdict = {"a":"act","act":"act","action":"act"}
    
    actdict =   {"a":"add","add":"add","addfile":"add",
                 "d":"delete","del":"delete","delete":"delete",
                 "dupl":"dupes","dupes":"dupes","duplicates":"dupes",
                 "list":"list","lst":"list","show":"list",
                 "load":"load"}
      
    if len(kwargs) == 0:
        
        kw_keys  = np.unique(list(kwargdict.values()))
        act_keys = np.unique(list(actdict.values()))
        act_keydict    = {}
        for i in range(len(act_keys)):    
            act_keydict[i]    = act_keys[i] 
        kwText   = ":".join(["List of Actions"]+[str(i)+" : "+ act_keys[i] for i in range(len(act_keys))]+["INPUT SELECTION"]).split(":")
        kwjc     = [":\n"]+list(np.concatenate([[":"]+["\n"] for i in range(int((len(kwText)-1)/2)) ]))+[":"]
        kwFull   = np.concatenate([[kwText[i]]+[kwjc[i]] for i in range(len(kwjc))])
        kwmt     = ["note"]+["note"]+list(np.concatenate([["stat"]+["wrn"]+["stat"]+["stat"] for i in range(len(act_keys)) ]))+["curio"]+["curio"]
        kwID = input(cprint(kwFull,mt=kwmt,tr=True))
        
        kwargs = {"act":act_keydict[int(kwID)]}
    
    kw = KwargEval(kwargs, kwargdict, act=False)    
    
    UV_dir = CUV(act="init",co=False)["Data_Directories_File"]
    UV_dir,UV_pt = Rel_Checker(UV_dir)
    UV_dir = PathSet(UV_dir,p=UV_pt)
    setattr(kw,"ddir", UV_dir)

    def NewDict(Dict):
        NewDict  = {}
        if type(Dict) == dict:
            Dkeys= list(Dict.keys())
            DictList = list(Dict.items())
        elif type(Dict) == list:
            DictList = Dict
            pass
        
        #We check if the dictionary needs refreshing by comparing the keys to a sequence. If it does not follow the 1,2,3,4,5... format, this will correct it!
        newdictbool = False
        for i in range(len(Dkeys)):
            if i+1 != int(Dkeys[i]):
                newdictbool = True
        if newdictbool == True:
            cprint("Correcting provided dictionary to contain the right order of entires!",mt="wrn")
            for i in range(len(Dict)):
                NewDict[str(i+1)]  = DictList[i][1]
            return(NewDict)
        elif newdictbool == False:
            return(Dict)
    
        
    try:
        kw.act = actdict[kw.act]
    except:
        cprint(["Note that ","kw.act"," = ",str(kw.act)," does not correspond to an action!"," Skipping kwarg eval."],mt = ["wrn","err","wrn","note"])
        
    #Check if file exists, else write an empty file:
    if os.path.isfile(kw.ddir) == False:
            jsonhandler(f = kw.ddir,d={},pt="abs",a="w")    
    DirDict = jsonhandler(f = kw.ddir,pt="abs", a="r")
    print(DirDict)
    if kw.act == "add":
        DirDict = NewDict(DirDict) #Make sure that the add command adds in the correct format
        print(DirDict)
        root = tk.Tk()
        file_path = askdirectory(title = "Please select a data directory to append to your data directories list!").replace("/",S_ESC)
        
        tk.Tk.withdraw(root)
        #First, we need to check that the dictionary has sequential keys, and if not, we need to rebuild these!
        
        DirDict[str(len(DirDict)+1)] = file_path
        if file_path != "":
            jsonhandler(f = kw.ddir,d=DirDict,pt="abs", a="w")
        else:
            cprint("No file selected, aborting!",mt="err")
    
    if kw.act == "delete":
        listdel  = ["Select a data directory to delete:\n"]
        cplist   = ["note"] 
        DDI = list(DirDict.items())
        for i in range(len(DDI)):
            cplist = cplist + ["wrn","note","stat","stat"]
            listdel = listdel+ [str(i)," : ",DDI[i][1], "\n"]
        cplist = cplist + ["curio"]
        listdel = listdel+["Enter number here: "]
        
        IPT = cprint(listdel,mt=cplist,jc="",tr=True)
        index = input(IPT)
        try:
            index = int(index)
        except:
            cprint("Non integer string entered! No fields will be deleted!",mt="err")
        if type(index) == int:
            DirDict.pop(DDI[index][0])
            DirDict = NewDict(DirDict)
                
            jsonhandler(f = kw.ddir,d=DirDict,pt="abs", a="w")
            
            cprint(["Deleted ", "{"+str(DDI[index][0])," : ",DDI[index][1],"}", " from directory list file"],mt = ["note","wrn","note","stat","wrn","note"])
            
    if kw.act == "dupes":
        DDK = list(DirDict.keys())
        DDI = list(DirDict.values())
        UNQ = np.unique(DDI)
        if len(DDI) == len(UNQ):
            cprint("No duplicates found!",mt="note")
        else:
            dupeID = []
            for unique in UNQ:
                hits = [i for i,val in enumerate(DDI) if val == unique]

                if len(hits) > 1:
                    dupeID += hits[1:]
                    
            print(DirDict)
            for dID in dupeID:
                cprint(["Deleting", "{",DDK[dID], ":",DDI[dID],"}","from",kw.ddir],mt = ["note","wrn","curio","wrn","curio","wrn","note","stat"],jc=" ")
                DirDict.pop(DDK[dID])
                
            cprint(["A total of","[",str(len(dupeID)),"]","dupes were deleted."],mt = ["note","wrn","curio","wrn","note"])
            DirDict = NewDict(DirDict)
            jsonhandler(f = kw.ddir,d=DirDict,pt="abs", a="w")
            
        
    if kw.act == "list":
        listshow  = ["List of currently saved directories:\n"]
        cplist   = ["note"] 
        DDN = list(DirDict.keys())
        DDI = list(DirDict.items())
        for i in range(len(DDI)):
            cplist = cplist + ["wrn","note","stat","stat"]
            listshow = listshow+ [DDN[i]," : ",DDI[i][1], "\n"]
        cplist = cplist
        cprint(listshow,mt=cplist)
        
    if kw.act == "load":
        return(DirDict)

#%%
def DefaultGrid(ax):
    ax.grid(which='major', color='darkgrey', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')  

#%%    
def DProc_Plot(Dproc,gtype):
    """
    Function to guess what needs to be plotted from the data it has been given.
    fig, which is an array containing figures.

    """
    
    fig = [ezplot(fid=1)]
    
    note_dict = {}
    
    xvar,yvar,xlab,ylab,pltype = {
                 'Director Rounding':['roundingradius dir','AbsPow',
                                      'Director Rounding radius [m]','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','AbsPow_OneParam'],
                 
                 'Multi Dipole Director Rounding':['roundingradius dir','AbsPow',
                                                   'Director Rounding radius [m]','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','AbsPow_OneParam'],
                 
                 'Variable Contacts':['GCx_span','AbsPow',
                                      'Gold Contact span [m]','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','AbsPow_OneParam'], 
                 
                 'pabs variable contacts':['GCx_span','AbsPow',
                                           'Gold Contact span [m]','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','AbsPow_OneParam'],
                 
                 "Director Separation":['dir_sep','AbsPow',
                                        'Director Separation [m]','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','Dir_Sep'],
                 
                 "no contacts":['s_d','AbsPow',
                                        'Distance from source [m]','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','gold_contacts'],
                 
                 "contacts":['s_d','AbsPow',
                                        'Distance from source [m]','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','director_MIE'],
                 
                 "Waveguide Thickness":['wg_zspan','AbsPow',
                                        'Waveguide Thickness','Power Fraction: '+r'${P_{abs}}/{P_{tot}}$','AbsPow_OneParam']
                       }[Dproc['note'][0]]
    
    
    if pltype == "AbsPow_OneParam":

        try:
            labl = {'Director Rounding':'Single-Dipole','Multi Dipole Director Rounding':'Multi-Dipole'}[Dproc['note'][0]]
        except: 
            labl=None

        fig[0].ax[0].scatter(Dproc[xvar],Dproc[yvar],label=labl)
        fig[0].ax[0].legend()
    
    elif pltype == "Dir_Sep":
        NWuniq = list(np.unique(Dproc['NW_sep_dir']))
        Dproc['dir_sep_p.NW'] = [[] for n in range(len(NWuniq))]
        Dproc['abspow p.NW']     = [[] for n in range(len(NWuniq))]
        
        for i in range(0,len(Dproc['dir_sep'])):
            NWind = NWuniq.index(Dproc['NW_sep_dir'][i])
            Dproc['dir_sep_p.NW'][NWind].append(Dproc['dir_sep'][i])
            Dproc['abspow p.NW'][NWind].append(Dproc['AbsPow'][i])

        for i,nuq in enumerate(NWuniq):
            if nuq != 0:
                sca1 = fig[0].ax[0].scatter(Dproc['dir_sep_p.NW'][i],Dproc['abspow p.NW'][i],label='NW sep = ' + str(NWuniq[i]) +'m')
            elif nuq == 0:
                fig[0].ax[0].plot([min(x for x in Dproc['dir_sep'] if x !=0),max(Dproc['dir_sep'])],[Dproc['abspow p.NW'][i][0],Dproc['abspow p.NW'][i][0]],label='no antenna')
                fig[0].ax[0].plot([min(x for x in Dproc['dir_sep'] if x !=0),max(Dproc['dir_sep'])],[Dproc['abspow p.NW'][i][1],Dproc['abspow p.NW'][i][1]],label='reflector only')
        
        lmbd = [910e-9*1/n for n in range(2,6)]
        for i in range(len(lmbd)):    
            fig[0].ax[0].plot([lmbd[i],lmbd[i]],[min(min(Dproc['abspow p.NW'])),max(max(Dproc['abspow p.NW']))],label='$\lambda/$'+str(i+2))
        
        fig[0].ax[0].set_xlim([1.6e-7,max(Dproc['dir_sep'])])
        fig[0].ax[0].legend(ncol=2)
    
    ##########YOU WERE WORKING RIGHT HERE!
    elif pltype == "gold_contacts":
        if "no contacts" in Dproc['note']:
            title = "Absorbed Power without Gold Contacts"
        else:
            title = "Absorbed Power with Gold Contacts"
            
        #2D plot
        fig[0].ax[0].set_title(title)

        sca1 = fig[0].ax[0].scatter(Dproc['s_d'],Dproc['AbsPow'],c=Dproc['rel_rot'],cmap='gnuplot')
        cb1 = fig[0].ax[0].colorbar(sca1)
        cb1.set_label('Relative Rotation '+r'$[\theta$]')
    
        """
        LComp,LCompFi = MatLoader(DList['.mat'][0])
        LComp['P_abs'] = np.reshape(LComp['Pabs'],[LComp['lambda'].shape[0],LComp['z'].shape[0],LComp['y'].shape[0],LComp['x'].shape[0]])   
        #plt.imshow(np.rot90(LComp['P_abs'][0,:,:,6])) displays the same (xslice for [y,z]) as in lumerical
        
        LComp['P_tot'] = AbsPowIntegrator(LComp['P_abs'],LComp['x'],LComp['y'],LComp['z'],LComp['lambda'])
        plt.figure(2)
        plt.plot(LComp['lambda'],LComp['P_tot'])
        I have no idea what this is?
        """
        
    
    
        
        PDP = []
        PDR = []
        PDL = []
        PLP = []
        PLR = []
        PLL = []
        TPDP = []
        TPDL = []
        TPDR = []
        
        for i in range(len(Dproc['AbsPow'])):
            if Dproc['rel_rot'][i] < 60 or Dproc['rel_rot'][i] > 120:
                PDP.append(Dproc['AbsPow'][i])
                PDR.append(Dproc['rel_rot'][i])
                PDL.append(Dproc['s_d'][i])
            else:
                PLP.append(Dproc['AbsPow'][i])
                PLR.append(Dproc['rel_rot'][i])
                PLL.append(Dproc['s_d'][i])
            if Dproc['rel_rot'][i] < 10 or Dproc['rel_rot'][i] > 170:
                if Dproc['ENW_x'][i] == 0:
                     TPDP.append(Dproc['AbsPow'][i])
                     TPDR.append(Dproc['rel_rot'][i])
                     TPDL.append(Dproc['s_d'][i])
        
        fig.append(ezplot(fid=2)) 
        
        #2D plot
        fig[1].ax[0].set_title(title)
        fig[1].ax[0].set_xlabel('distance from source [m]')
        fig[1].ax[0].set_ylabel('Power Fraction: '+r'${P_{abs}}/{P_{tot}}$')
        fig[1].ax[0].grid(True)
        fig[1].ax[0].scatter(PDL,PDP,c='cyan')
        sca3 = fig[1].ax[0].scatter(PLL,PLP,c=PLR ,cmap='gnuplot')
        fig[1].ax[0].scatter(TPDL,TPDP,c='green')
        cb3 = fig[1].ax[0].colorbar(sca3)
        cb3.set_label('Relative Rotation '+r'$[\theta$]')
    
    elif pltype == "director_MIE":
        None
    
    #Labelling and Gridding
    fig[0].ax[0].set_xlabel(xlab)
    fig[0].ax[0].set_ylabel(ylab)
    fig[0].ax[0].grid(True)
    return(fig)
    
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
class ezplot(object):
    """
    goal of testing:
    Plot.fig = plt.figure(num)
    Plot.add_subplot => fig.add_subplot => Plot.ax[ind] = ax_new
    """
    def __init__(self,**kwargs): #Initialise by assigning a figure, creating an axis, and selecting the axis
        """
        kwargs:
            ==============
            fid:  the figure ID - can be set manually if you want to overwrite a specific figure, but will be set automatically otherwise
                format: [<int: fid>]
            gspec: uses matplotlib gridspec, this allows you to define how many subplot grid spaces you want to include in your plot.
                format: [<int: row>,<int: col>]
            spd: defines how many subplots you want, and how you want them arranged. The total number subplots cannot exceed the total number of gpec grids
                format: (all are <int>:) ([r1_min,r1_max,c1_min,c1_max],[r2_min,r2_max,c2_min,c2_max],...)
                If this is not set, then it will be assumed that you want the same number of axes as you have grid spaces in your gspec.
            
            Note: any kwargs that is accepted by a plot function will be included as well, so any errors that get raised will come from a kwarg that is not accepted by an add_subplot function.
            for instance: including projection='3d' will change the subplot projection for all plots - so only use this if you need a uniform plot format for all plots - otherwise, change it later using self.ax[n].set_prop instead.
       
           General Use case examples:
               Creating a new figure with only one axis - and no figure ID:
                   FIG = ezplot()    
                   FIG.ax[0].plot(x,y)
               Creating a new figure with 4 axes, a specific fid, and a 3D projection in each.
                   FIG = ezplot(gspec=[2,2],fid=1,projection='3d')
                   FIG.ax[0].title_set('ax0'); FIG.ax[1].title_set('ax1'); FIG.ax[2].title_set('ax2'); FIG.ax[3].title_set('ax3')
               Creating a new figure with 4 axes, a specific fid, and a 3D projection in each - but only three figures.
                   FIG = ezplot(gspec=[2,2],spd=[([0,1,0,1]),([1,2,0,2]),([0,1,1,2])],fid=1,projection='3d')
                   
                   
               fields you can access:
                   FIG.fig    = figure handle
                   FIG.fid    = figure ID
                   FIG.ax[n]  = axes handles (list from 0 to n)
            All plt.plot parameters can be passed with **kwargs, so if you want a specific window size, resolution, font size, title, etc. you can pass them in the command as you would normally
       """
        kwargdict = {'f_id':'fid','fid':'fid','fignum':'fid','fi':'fid',
                     'gridspec':'gspec','colrow':'gspec','gspec':'gspec','gs':'gspec',
                     'spd':'spd','sub_plot_dim':'spd'}
        kuniq = np.unique(list(kwargdict.keys()))
        
        def sp_id_checker(kw):
            if kw.ax_id == None:
                for i in range(len(self.ax)):
                    if self.ax[i].has_data() == False:
                        kw.ax_id = i
                        break
                    
            if kw.ax_id == None:
                kw.ax_id = 0
            return(kw.ax_id)
        for key in kwargs.keys():
            if key not in kuniq:
                kwargdict[key] = key
        kw = KwargEval(kwargs,kwargdict,gspec=[1,1],fid=None,spd=None)        
        
        if kw.fid == None:
            if len(plt.get_fignums()) != 0:
                kw.fid = plt.get_fignums()[-1]+1
            else:
                kw.fid = 1
        self.fig = plt.figure(kw.fid)
        self.fig.fid = kw.fid
        self.fig.clf()           # This clears the old figure - clearing will only be done if: you aer initialising using funct on it's own, or if specifying 'new' in kwargs when plotting in axis. 
        
        #define the size of the grid intended for plotting: Default is 1x1 (as in, only one plotting grid available)
        
        self.gspec = matplotlib.gridspec.GridSpec(kw.gspec[0],kw.gspec[1]) #This makes a 3x2 sized grid
        """
        #We test to see if there is an overlap - or, do we need to test this? It will throw an error regardless?
        Tot_coord = []        
        for spd in kw.spd:
            for x in list(range(spd[0],spd[1]+1)):
                for y in list(range(spd[2],spd[3]+1)):
                    Tot_coord.append([x,y])
        """
        self.sps = []
        self.ax  = {}
        self.mappable = {} 
        self.Pkwargs = {}
        for key,val in kwargs.items():
            if key not in kuniq:
                self.Pkwargs[key] = val
                
        if kw.spd != None:
            for spd in kw.spd:
                self.sps.append((slice(spd[0],spd[1]),slice(spd[2],spd[3])))
            
            for i in range(len(self.sps)):
                self.ax[i] =  self.fig.add_subplot(self.gspec[self.sps[i][0],self.sps[i][1]],**self.Pkwargs)
                self.mappable[i] = i
        else:
            for i in range(kw.gspec[0]*kw.gspec[1]):
                self.ax[i] =  self.fig.add_subplot(self.gspec[i],**self.Pkwargs)
                
                
    def sp_id_checker(self,kw):
        if kw.ax_id == None:
            for i in range(len(self.ax)):
                if self.ax[i].has_data() == False:
                    kw.ax_id = i
                    break
                
        if kw.ax_id == None:
            kw.ax_id = 0
        return(kw.ax_id)
    
    def fquiver(self,data,**kwargs):
        """        
        This function makes a quiver plot in the selected AXIS! (so self.ax[int].quiver(data=[x],[y],**kwargs)
        data must be in the format: [x,y,z,xdir,ydir,zdir] where x,y,z are vector locations, and xdir,ydir,zdir give the magnitude and direction of the vector.
        x/y/z/xdir/ydir/zdir all need to come in the same np.array shape - buy ANY number of 4d/3d/2d/1d matrices are supported. 
        This means that even if your x.shape = [50,50,50], this function will create a list of appropriate vectors to plot it correctly!
        
        this function also requires an ax_id (this is the subplot ID given as a list from sp1 -> spN). If this is not provided, the function will look for empty plots to fill, and if no empty plots are found, it will plot in the first plot.
        """
        
        kwargdict = {'ax_id':'ax_id','axisid':'ax_id','axid':'ax_id'}
        kuniq = np.unique(list(kwargdict.keys()))
        
        for key in kwargs.keys():
            if key not in kuniq:
                kwargdict[key] = key
        kw = KwargEval(kwargs,kwargdict,ax_id = None)        
        
        data2 = []
        for dat in data:
            if type(dat) == list:
                data2.append(np.array(dat))
            
            elif type(dat) == np.ndarray:
                data2.append(np.array(dat))
        
        if len(data2[0].shape) != 1:
            if data2[0].shape == data2[1].shape == data2[2].shape:
                vnum = np.prod(data2[0].shape)
                for i in range(len(data2)):
                    data2[i] = data2[i].reshape(vnum)

        x = data2[0]; y = data2[1]; z = data2[2]
        xdir = data2[3]; ydir = data2[4]; zdir = data2[5]                
                
        kw.ax_id = self.sp_id_checker(kw)    
  
        self.Pkwargs = {}
        for key,val in kwargs.items():
            if key not in kuniq:
                self.Pkwargs[key] = val
                
        self.ax[kw.ax_id].quiver(x,y,z,xdir,ydir,zdir, **self.Pkwargs)
        pass
    
    def DFT_Intensity_Plot(self,xgrid,ygrid,Eabs,gridscale,**kwargs):
        """
        xgrid is the x_min->x_max parameters of your grid (set in Lumerical)
        ygrid is the y_min->y_max parameters of your grid (set in Lumerical)
        Eabs is your DFT E field data, modified as per the lumerical document - you can likely do this here later, but for now this will have to do
        gridscale is the length unit per grid square (usually 1e-6)
        """
        kwargdict = {'ax_id':'ax_id','axisid':'ax_id','axid':'ax_id','cmap':'cmap','vmin':'vmin','vmax':'vmax'}
        kuniq = np.unique(list(kwargdict.keys()))
        
        for key in kwargs.keys():
            if key not in kuniq:
                kwargdict[key] = key
        kw = KwargEval(kwargs,kwargdict,ax_id = None,cmap='jet',vmin=None,vmax=None)
        kw.ax_id = self.sp_id_checker(kw)    
        
        self.Pkwargs = {}
        for key,val in kwargs.items():
            if key not in kuniq:
                self.Pkwargs[key] = val
                
        xgmod = xgrid*gridscale
        ygmod = ygrid*gridscale

        
        self.ax[kw.ax_id].set_xlabel('x coordinate [$\mu$ m]')
        self.ax[kw.ax_id].set_ylabel('y coordinate [$\mu$ m]')
        
        # Linear scale region
        self.mappable[kw.ax_id] = self.ax[kw.ax_id].pcolor(xgmod,ygmod,Eabs,cmap=kw.cmap,vmin=kw.vmin,vmax=kw.vmax)
        
        # Important
        self.ax[kw.ax_id].set_aspect('equal')


#%%
def ezquiver(data,**kwargs):
    """
    What should this function do?
    Accept a list of vectors to return the fig handle, and axis, of a plotted quiver plot.
    Problems:
        You cannot use -self-, because this returns __main__ instead of the figure.

    """
    kwargdict = {'f_id':'fid','fid':'fid','fignum':'fid','fi':'fid'}
    kw = KwargEval(kwargs, kwargdict,fid=None)
    
    if kw.fid !=None:
        if len(plt.get_fignums()) != 0:
            kw.fid = plt.get_fignums()[-1]+1
        else:
            kw.fid = 1
            
    fig = plt.figure(kw.fid)
    fig.clf()

    ax = fig.add_subplot(projection='3d',proj_type='persp') # Makes a graph that covers a 2x2 size and has 3D projection        
    ax.quiver(data)

#%%   
def Get_FileList(path,**kwargs):
    """
    A function that gives you a list of filenames from a specific folder
    path = path to files. Is relative unless you use kwarg pathtype = 'abs'
    kwargs**:
        pathtype: enum in ['rel','abs'], default = 'rel'. Allows you to manually enter absolute path    
        ext: file extension to look for, use format '.txt'. You can use a list e.g. ['.txt','.mat','.png'] to collect multiple files. Default is all files
        sorting: "alphabetical" or "numeric" sorting, default is "alphabetical"
    """
    
    S_ESC = LinWin() #check if an escape character \ is needed or if / should be used
    kwargdict = {'pathtype':'pt','pt':'pt','path':'pt','p':'pt',
                 'extension':'ext','ext':'ext','ex':'ext','e':'ext',
                 's':'sort','sort':'sort','sorting':'sort'}
    #Collecting kwargs
    kw = KwargEval(kwargs, kwargdict, pt = 'rel',ext = None, sort = None)
    
    cprint('=-=-=-=-=-=-=-=-=-=-=- Running: Get_FileList -=-=-=-=-=-=-=-=-=-=-=',mt = 'funct')
    Dpath = PathSet(path,pt=kw.pt)
    
    #Checking that kw.sort has been selected correctly
    if kw.sort not in [None,'alphabetical','numeric']:
        cprint('sorting was not set correctly, reverting to using alphabetical sorting (no extra sorting)',mt='note')
        kw.sort = None
    
    #Filtering out the intended file types from the filenames
    #First, checking that the format for ext is correct.
    extreplacer = []
    if type(kw.ext) is str:
        if kw.ext.startswith('.') is False:
            kw.ext = '.'+kw.ext
            cprint('Correcting incorrect extension from ext = \''+kw.ext[1:]+ '\' to ext = \''+kw.ext+'\'',mt='caution')
        extreplacer.append(kw.ext)
        kw.ext = tuple(extreplacer)
    elif type(kw.ext) is tuple: 
        
        for i in range(len(kw.ext)):
            if kw.ext[i].startswith('.') is False:
                extreplacer.append('.'+kw.ext[i])
                cprint('tuple ext['+str(i)+'] was corrected from ext['+str(i)+'] = \''+kw.ext[i]+'\' to ext['+str(i)+'] = \'.'+kw.ext[i]+'\'',mt='caution')
            else:
                extreplacer.append(kw.ext[i])
        kw.ext = tuple(extreplacer)
    else:
        kw.ext = None
        cprint('ext must be in string or tuple format - setting ext = None and gathering all files instead',mt='err')
        
    summary = []
    if kw.ext is not None:
        NList = {}
        DList = {}
        summary = ['\nSummary:']
        for ex in kw.ext:
            NList[ex] = [file for file in os.listdir(Dpath) if file.endswith(ex)]
            if kw.sort == 'numeric':
                NList[ex] = natsort.natsorted(NList[ex], key=lambda y: y.lower())
                cprint([ex, ' files were sorted numerically'],fg=['g','c'],ts='b')
            DList[ex] = [Dpath+S_ESC+name for name in NList[ex]]
            
        
            DSum = len(DList[ex])
            summary.append(str(DSum) + ' ' + ex + ' files')
                       
    else:
        NList = [file for file in os.listdir(Dpath)]
        if kw.sort == 'numeric':
            NList = natsort.natsorted(NList, key=lambda y: y.lower())
            cprint([ex, ' files were sorted numerically'],fg=['g','c'])
        DList = [Dpath+S_ESC+name for name in NList]
    
    cprint(['A total of',str(len(DList)), 'file extensions were scanned.']+summary,ts='b',fg=['c','g','c',None,'g'],jc = [' ',' ','\n'])
    
    
    return(DList,NList)
 
 
#%%
def Init_LDI():
    """
    A function that aims to set up the file structure of a new project. Running this function first will create the DataImportSettings.json and populate it with the default settings.
    Then, it will call an "add" command for DataDirectories.json, prompting you to select a data folder. 
    """
    
    #First, check if each of the files exist in your working directory:
    DIS = "DataImportSettings.json"
    Ddir= "DataDirectories.json"
    if os.path.isfile(PathSet(DIS, pt = 'rel')) and os.path.isfile(PathSet(Ddir,pt='rel')) == True:
        cprint(['Comment out your Init_LDI! You already have both',DIS,'and',Ddir],mt = ['wrn','stat','wrn','stat'])
    if os.path.isfile(PathSet(DIS, pt = 'rel')) == False:
        cprint(['Creating',DIS],mt=['curio','stat'])
        CUV(act='reset',pt='rel')
    
    if os.path.isfile(PathSet(Ddir,pt='rel')) == False:
        cprint(['Creating',DIS,'Please also select the first Data folder to append!'],mt=['curio','stat','note'],jc=[' ',':\n ',''])
        DataDir(act='add')

#%%
def jsonhandler(**kwargs):
    """
     DESCRIPTION.
     A simple script that handles saving/loading json files from/to python dictionaries. 

    Parameters
    ----------
    **kwargs :
            kwargdict = {'f':'filename','fn':'filename','filename':'filename',
                 'd':'data','dat':'data','data':'data',
                 'a':'action','act':'action','action':'action',
                 'p':'pathtype','pt':'pathtype','pathtype':'pathtype'}

    Returns
    -------
    Depends: If loading, returns the file, if saving - returns nothing

    """
    kwargdict = {'f':'filename','fn':'filename','filename':'filename',
                 'd':'data','dat':'data','data':'data',
                 'a':'action','act':'action','action':'action',
                 'p':'pathtype','pt':'pathtype','pathtype':'pathtype'}
    
    kw = KwargEval(kwargs, kwargdict, pathtype='rel')
   
    if hasattr(kw,"filename") and hasattr(kw,"action") == True:    
        if kw.action in ['read','r']:
            with open(PathSet(kw.filename,pt=kw.pathtype),'r') as fread:
                data = json.load(fread)
                return(data)
            
        elif kw.action in ['write','w']:
            try:
                with open(PathSet(kw.filename,pt=kw.pathtype),'w') as outfile:
                    cprint(['saved',str(kw.data),'to',str(outfile)],mt=['note','stat','note','stat'])
                    json.dump(kw.data,outfile)
            except:
                cprint('Data does not exist! Remember to enter d/dat/data = dict',mt='err')
    else:
        cprint('No filename given! Cannot read or write to json file!',mt='err')
    
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

#%%
def LDI_Data_Import(FLTuple,**kwargs):
    kwargdict = {'ikey':'ikey','ikeys':'ikey'}
    
    kw = KwargEval(kwargs, kwargdict, ikey=[])
    if type(kw.ikey) == "str":
        kw.ikey = [kw.ikey]
    print(kw.ikey)
    DList = FLTuple[0]
    NList = FLTuple[1]
    Dproc = {} # Create an empty dictionary to store your data in 
    for i,txtfile in enumerate(NList['.txt']):
        mfiles  = list(filter(lambda x: txtfile.split('.txt')[0] in x, DList['.mat']))
        MDat = {}; 
   # try:
        for mfile in mfiles:
            MDat = {**MDat,**MatLoader(mfile)[0]}
            #Then do calculations on the longer data sets and store only the specific single data needed in MDat
        if 'Pabs' in MDat.keys():
            MDat['P_abs'] = np.reshape(MDat['Pabs'],[MDat['lambda'].shape[0],MDat['z'].shape[0],MDat['y'].shape[0],MDat['x'].shape[0]])
            cprint(['Now processing file: ',str(mfile)],mt='curio')
        if "Pabs_total" in MDat.keys():
            MDat['P_tot']  = AbsPowIntegrator(MDat['P_abs'],MDat['x'],MDat['y'],MDat['z'],MDat['lambda'])
            MDat['AbsPow'] = max(MDat['P_tot'])
        TDat = txt_dict_loader(DList['.txt'][i])
        try:
            #In the case that you are calculating the cross section
            MDat['Q_Abs'] = np.max(MDat['Qabs'])
            MDat['Q_Scat'] = np.max(MDat['Qscat'])
        except:
            None
        Dproc = Prog_Dict_Importer(Dproc,{**TDat,**MDat},ikey=kw.ikey)

        if "ENW_x" and "ENW_y" and "ENW_z" in Dproc.keys():
            anw_l = np.array([0,0,Dproc['ENW_z'][-1]])
            enw_l = np.array([Dproc['ENW_x'][-1],Dproc['ENW_y'][-1],Dproc['ENW_z'][-1]])
            squared_dist = np.sum((anw_l-enw_l)**2, axis=0)
            if "ENW_z_rot" in Dproc.keys():
                enw_rot = Dproc['ENW_z_rot'][-1] 
    
    
            if Dproc['ENW_y'][-1] == 0:
                atdeg =  0
            else:
                atdeg = np.degrees(np.arctan(abs(Dproc['ENW_x'][-1]/Dproc['ENW_y'][-1]))) 
        
            rel_rot = enw_rot + atdeg
            
            if rel_rot>180:
                rel_rot = atdeg - (180 - enw_rot)  
            Dproc = Prog_Dict_Importer(Dproc,{'rel_rot':rel_rot})
            Dproc = Prog_Dict_Importer(Dproc,{'s_d':np.sqrt(squared_dist)})
        
  #  except:
                
            
        if 'lambda' not in Dproc.keys():
            cprint("You need to modify your data to contain lambda, or provide it separately")
                
        #Here, you can add a **function** that calculates something from each file's dataset, but not as a big paragraph.
        
        
    return(Dproc)

#%%
def LinWin():
    """
    Literally just checks if we need \\ or / for our directories by seeing how os.getcwd() returns your working directory
    """
    if '\\' in os.getcwd():
        S_ESC = '\\'
    else:
        S_ESC = '/'
    return(S_ESC)


#%%
def PathSet(filename,**kwargs):
    """"
    p/pt/pathtype in [rel,relative,abs,absolute]
    Note that rel means you input a relative path, and it auto-completes it to be an absolute path, 
    whereas abs means that you input an absolute path!
    """
    #Check if we need \\ or / for our directories
    S_ESC = LinWin()
        
    ## Setting up the path and pathtype type correctly:  
    kwargdict = {'p':'pathtype','pt':'pathtype','pathtype':'pathtype'}
    kw = KwargEval(kwargs, kwargdict,pathtype='rel')
            
    if kw.pathtype not in ['rel','abs']: 
        cprint('pathtype set incorrectly, correcting to \'rel\' and assuming your path is correct',mt='caution')
        kw.pathtype = 'rel'
    
    if kw.pathtype in ['abs','absolute']:
        WorkDir = ""
        
        
    elif kw.pathtype in ['rel','relative']:
        WorkDir = os.getcwd()+S_ESC

    if filename == None:
        filename = ''
    return(WorkDir+filename)


#%%
def Rel_Checker(path):

    """
    Simple function that checks if a file location is relative to the working directory, and if so - replaces the file directory with a relative coordinate version.
    Returns: (path,pt). So, relative (path) (if found to be relative), and path type (pt) just in case
    """
    #Check if we need \\ or / for our directories
    S_ESC = LinWin()
        
    #First, we check if the current working directory is actually correct!
    DIS  = "DataImportSettings.json"
    Ddir = "DataDirectories.json"
    if os.path.isfile(PathSet(DIS, pt = 'rel')) or os.path.isfile(PathSet(Ddir,pt='rel')) == True:
        WorkDir = os.getcwd()+S_ESC
        if os.path.isabs(path) == True:
            if WorkDir in path:
                path = path.replace(WorkDir,'')
                pt = 'rel'
            else:
                pt = 'abs'
        else:
            pt = 'rel'
        return(path,pt)
    else:
        cprint(['One of the required ',DIS,' or ',Ddir,' files are missing! Consider running ','Init_LDI()',' again before continuing!'],mt=['wrn','err','wrn','err','wrn','curio','wrn'])

             
#%%          
def MatLoader(file,**kwargs):
    """

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    [.mat data, field names]

    """
    cprint('=-=-=-=-=-=-=-=-=-=-=- Running: MatLoader -=-=-=-=-=-=-=-=-=-=-=',mt = 'funct')
    S_ESC = LinWin()
    kwargdict = {'txt':'txt','textfile':'txt',
                 'json':'json','jsonfile':'json',
                 'dir':'path','directory':'path','path':'path','p':'path',
                 'tf':'tf','txtfile':'tf',
                 'esc':'esc','escape_character':'esc','e':'esc','esc_char':'esc'}
    kw = KwargEval(kwargs, kwargdict,json=False,txt  = False, path = 'same', tf  = None, esc  = None)
    
    #Mat File Loading
    FIELDDICT = {}
    f = h5py.File(file,'r')
   
    for k, v in f.items():
        FIELDDICT[k] = np.array(v)

    # FILEDICT used once to get the keys then unused
    FIELDLIST = list(FIELDDICT.keys()) 
    
    data = {}
    dfields = []
    if '#refs#' in FIELDLIST: 
        FIELDLIST.remove('#refs#')
        cprint(["Scanning fields:"]+FIELDLIST,fg='c',ts='b')
    
    for i in range(len(FIELDLIST)):
        try:
            dfields.append(list(f[FIELDLIST[i]].keys()))
            twokeys = True
        except:
            dfields.append(FIELDLIST)
            twokeys = False
        for field in dfields[i]:
            if twokeys == True:
                data[field] = np.array(f[FIELDLIST[i]][field])
            elif twokeys == False:
                data[field] = np.array(f[field])
            # perform a special check against Lumerical's complex notation
            if data[field].dtype == np.dtype([('real', '<f8'), ('imag', '<f8')]) :
                data[field] = data[field].view(complex)
                
            if len(data[field].shape) == 2 and data[field].shape[0] == 1:
                oldshape    = data[field].shape
                data[field] = data[field][0]
                cprint(['corrected','data['+str(field)+'].shape','from',str(oldshape),'to',str(data[field].shape)],mt=['note','status','note','wrn','note','status'])
    mname = file.split(S_ESC)[-1]
    data['matfilepath'] = file
    data['matname'] = mname
    
    #%% .txt File Loading
    #Always guess the text filename
    fname = mname.split('.')[0]
    if kw.tf == None and kw.path == 'same':
        path = os.path.dirname(file)
    txtlp = Get_FileList(path, ext='.txt',pathtype='abs')[0]
    jsonlp = Get_FileList(path, ext='.json',pathtype='abs')[0]
    txtind = [i for i, s in enumerate(txtlp['.txt']) if (s.split(S_ESC)[-1]).split('.')[0] in fname]
    jsonind= [i for i, s in enumerate(jsonlp['.json']) if (s.split(S_ESC)[-1]).split('.')[0] in fname]
    try:
        data['txtfilepath'] = txtlp['.txt'][txtind[0]]
        data['txtname'] = data['txtfilepath'].split(S_ESC)[-1]
        d = []
    except:
        None
    try:
        data['jsonfilepath'] = jsonlp['.json'][jsonind[0]]
        data['jsonname'] = data['jsonfilepath'].split(S_ESC)[-1]
        d = []
    except:
        None
        
    if kw.json == True:
        f = open(data["jsonfilepath"])
        dat = json.load(f)
        f.close()
        namedata  = dat[list(dat.keys())[0]]["name"]["_data"]
        valuedata = dat[list(dat.keys())[0]]["value"]["_data"]
        dictdat   = {}
        for i,val in enumerate(valuedata):
            if type(val) == dict:
                try:
                    value = val["_data"]
                except:
                    value = val
            else:
                value = val
            dictdat[namedata[i]] = value
        data.update(dictdat)
        
    if kw.txt == True:
            #determine escape character if none is given
        try:
            with open(data['txtfilepath'],'r') as source:
                line1 = source.readline()
                skipline1 = False
                if len(line1)<=1:
                    line1 = source.readline()
                    skipline1 = True
                if kw.esc == None or kw.esc not in line1:
                    if '\t' not in line1:
                        numspc = maxRepeating(line1,guess=' ')
                        kw.esc = "".join([numspc[0] for i in range(numspc[1])])
                                        
                    else:
                        kw.esc = '\t'
            
            with open(data['txtfilepath'],'r') as source:
                if skipline1 == True:
                    source.readline()
                    
                for line in source:
                    line = line.strip('\n')
                    fields = line.split(kw.esc)
                    d.append(fields)  
            if len(d[-1]) < len(d[0]):
                d.pop(-1)
            floatcol = np.zeros([len(d),len(d[0])])
            for i in range(len(d)):         
                for k in range(len(d[0])):
                    try:
                        A = float(d[i][k])
                        floatcol[i,k] = 1
                    except:
                        floatcol[i,k] = 0
            LC = sum(floatcol[:,0])
            RC = sum(floatcol[:,1])
            
            if LC > RC:
                VarI = 0
                FieldI = 1
            else:
                VarI = 1
                FieldI = 0
                        
            for ent in d:
                try:
                    ent[VarI] = float(ent[VarI])
                    data[ent[FieldI]] = float(ent[VarI])
                except:
                    data[ent[FieldI]] =  ent[VarI]
            cprint(['Loaded auxilary variables from file =',data['txtfilepath'], 'successfully!\n','Added:',str(d)],mt=['note','stat','note','note','stat'])
        except:
            cprint("Something went wrong - I probably didn\'t find a matching .txt file :/. The script continued none the less.",mt='err' )
        
    tranconf = 0
    powconf  = 0 
    
    if 'trans' in FIELDLIST[0].lower():
        cprint('Best guess is that you just loaded the data from a Transfer Box analysis group!', mt = 'curio')
        tranconf = 1
    if any(substring in FIELDLIST[0].lower() for substring in ['pow','pabs']):
        cprint('Best guess is that you just loaded the data from a power absorption analysis group!',mt = 'curio')
        powconf = 1
    if [tranconf,powconf] == [1,1]:
        cprint('Naming convention might be strange - you should know better what type of file you loaded...',fg='o')    
    return(data,dfields) 


#%%
def maxRepeating(str, **kwargs): 
    """
    DESCRIPTION.
    A function used to find and count the max repeating string, can be used to guess
    Parameters
    ----------
    str : TYPE
        DESCRIPTION.
    **kwargs : 
        guess : TYPE = str
        allows you to guess the escape character, and it will find the total number of that character only!

    Returns
    -------
    res,count
    Character and total number consecutive

    """
    guess = kwargs.get('guess',None) 
    l = len(str) 
    count = 0
  
    # Find the maximum repeating  
    # character starting from str[i] 
    res = str[0] 
    for i in range(l): 
          
        cur_count = 1
        for j in range(i + 1, l): 
            if guess is not None:
                if (str[i] != str[j] or str[j] != guess):
                        break
            else:
                if (str[i] != str[j]):
                        break
            cur_count += 1
  
        # Update result if required 
        if cur_count > count : 
            count = cur_count 
            res = str[i] 
    return(res,count)

        
#%%        
def MergeList(path,target,**kwargs):
        """
        target is filetype, so ".txt"
        kwargs:
            c/contains - type:str. Specify that a specific substring must ALSO be included in the mergelist
        
        """
        kwargdict = {'contains':'contains','c':'c',
                      'sort':'sort','name':'name','n':'name'}
        kw = KwargEval(kwargs, kwargdict,c=False,sort='numeric',name='MergeList')
        
        MListOUT = path+'\\'+kw.name+'.txt'
        if kw.c == False:
            DirList  = [file for file in Get_FileList(path,ext=(target),pt='abs',sort=kw.sort)[0][target] if file.endswith(target)]
        elif type(kw.c) == str:
            DirList  = [file for file in Get_FileList(path,ext=(target),pt='abs',sort=kw.sort)[0][target] if file.endswith(target) and kw.c in file]
                
            
        FList    = ['file \''+DirList[n]+'\' \n' for n in range(len(DirList))]
        MergeList = open(MListOUT, "w")
        MergeList.write("".join(FList))
        MergeList.close()
        #ffmpeg -f concat -i MergeList.txt -c copy output.mp4         
            

#%%        
def MultChoiceCom(**kwargs):
    pass

#%%
def npy_filehandler(**kwargs):
    kwargdict = {'f':'f','fn':'f','f':'f',
                 'd':'d','dat':'d','data':'d',
                 'a':'a','act':'a','a':'a',
                 'p':'pt','pt':'pt','pathtype':'pt'}
    
    kw = KwargEval(kwargs, kwargdict, pt='rel',d=None,f=None,a=None)
    
    if os.path.exists(kw.f) == False:
        kw.f,kw.pt = Rel_Checker(kw.f)
    
    if kw.a in ['s','save']:
        np.save(kw.f,kw.d)
        
    if kw.a in ['l','load']:
        try:
            data = np.load(kw.f)
        except ValueError:
            data = np.load(kw.f,allow_pickle=True)
        except FileNotFoundError:
            if ".npy" not in kw.f:
                try:
                    data = np.load(kw.f+'.npy',allow_pickle=True)
                except:
                    cprint('No file found with that name')
        
        if len(data.shape) == 0 and type(data) != dict:
            data = dict(data.item())
        
        return(data)

#%%
def Prog_Dict_Importer(Dict, data, **kwargs):
    """
    Automatically finds any and all fields inside of data that have a set length of less than maxlen = 1 (or higher, if set manually)
    Use: import many unique variables other than the main data into a dictionary for later use!
    Parameters
    ----------
    data : This is the dictionary that you create when you use Matloader - Specifically: data,Fields = MatLoader(file,txt=UV['txt_import'])    
    
    Dict: This can be an empty or partially filled dictionary. This is where the values from each key in data get stored - ergo the "progressive" part
    
    **kwargs : A list of optional settings that you can use 
        [maxlen,length,ml] - maximum length of array to keep. If the length of an array exceeds this value, it will not be stored within
        Default value: 1
        

    Returns
    -------
    Dict - it appends same named keys from data into dict, and adds any new fields that were not present 
    warnings: If you append a new field into a partially filled dictionary, the script will warn you - but continue anyways. 
    If this comes from you wanting to merge data sets - make sure that no two fields share the same name, else you will get arrays of different lengths.

    """
    kwargdict = {'maxlen':'ml','length':'ml','ml':'ml',
                 'ikeys':'ikey','ikey':'ikey'}
    kw = KwargEval(kwargs, kwargdict, ml = 1,ikey=[])
    for key in data.keys():
        #test if iterable:
        try:
            iter(data[key])
            Iterable = True
        except:
            Iterable = False
        
        #check if the key exists, and if not - create it
        if key not in Dict.keys():
            Dict[key] = []
    
        if Iterable == True:
            if (len(data[key]) <= kw.ml or type(data[key]) == str) or key in kw.ikey:
                Dict[key].append(data[key])
            
        else:
            Dict[key].append(data[key])
        
    return(Dict)


#%%
def txt_dict_loader(txtfile,**kwargs):
    S_ESC = LinWin()
    kwargdict = {'dir':'path','directory':'path','path':'path','p':'path',
                 'esc':'esc','escape_character':'esc','e':'esc','esc_char':'esc'}
    kw = KwargEval(kwargs, kwargdict, path = 'same', esc  = None)
    d = []
    with open(txtfile,'r') as source:
        line1 = source.readline()
        skipline1 = False
        if len(line1)<=1:
            line1 = source.readline()
            skipline1 = True
        if kw.esc == None or kw.esc not in line1:
            if '\t' not in line1:
                numspc = maxRepeating(line1,guess=' ')
                kw.esc = "".join([numspc[0] for i in range(numspc[1])])
                                
            else:
                kw.esc = '\t'
    
    with open(txtfile,'r') as source:
        if skipline1 == True:
            source.readline()
            
        for line in source:
            line = line.strip('\n')
            fields = line.split(kw.esc)
            d.append(fields)  
    if len(d[-1]) < len(d[0]):
        d.pop(-1)
    floatcol = np.zeros([len(d),len(d[0])])
    for i in range(len(d)):         
        for k in range(len(d[0])):
            try:
                A = float(d[i][k])
                floatcol[i,k] = 1
            except:
                floatcol[i,k] = 0
    LC = sum(floatcol[:,0])
    RC = sum(floatcol[:,1])
    
    if LC > RC:
        VarI = 0
        FieldI = 1
    else:
        VarI = 1
        FieldI = 0
    d_dict = {}            
    for ent in d:
        try:
            ent[VarI] = float(ent[VarI])
            d_dict[ent[FieldI]] = float(ent[VarI])
        except:
            d_dict[ent[FieldI]] =  ent[VarI]
    return(d_dict)

#%% Help Functions to determine discrete lists of forward and reverse bias
#Checks if a list is not discrete
def is_not_discrete(data):
    for i in range(1, int(len(data)/5)):
        if data[i] <= data[i - 1] or data[i] >= data[i + 1]:
            return True

    return False

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

##
def Keithley_xls_read(directory,**kwargs):
    if directory == None:
        directory = os.getcwd()
        
    # Get a list of all the Excel files in the current directory
    files = [directory+"\\"+f for f in os.listdir(directory) if f.endswith('.xls')]
    
    # Create an empty dictionary to store the data
    data = {}
    
    """
    We must first handle the Logbook
    """
    logfiles = [file for file in files if "LOG" in file]
    for file in logfiles:
        filename = os.path.splitext(file.split('\\')[-1])[0]
        
        xls = xlrd.open_workbook(file)  
        file_data = {}
        """
        Logbook Handling
        -This section deals with extracting logbook information, and storing that in a dict that can then be imported 
        into the full data structure!.
        """
        
        if 'LOG' in filename.upper():
            
            # If filename contains LOG, read data as a flat dictionary
            sheet = xls.sheet_by_index(0)
            rows = []
            for row_index in range(1, sheet.nrows):  # Exclude the first row
                row_data = sheet.row_values(row_index)
                rows.append(row_data)
            # Extract the keys from the first row
            keys = [str(cell_value) for cell_value in sheet.row_values(0) if cell_value != ""]
            
            # Convert data to a dictionary with the values in the first row as the keys
            flat_data = {}
            #Determine the SMU position data and create a key for this in flat_data
            smu_ind = [i for i, item in enumerate(keys) if "SMU" in item]
            flat_data['positions'] = {}
            
            for ind in smu_ind:
                flat_data['positions']["pos"+str(1+ind - min(smu_ind))] = {}
                flat_data['positions']["pos"+str(1+ind - min(smu_ind))]['SMU'] = keys[ind]
            
            
            #position_indices = [i for i, header in enumerate(keys) if 'POSITION' in header]
            #This section finds rows that are empty, and excludes them from our search
            for row in rows:
                try:
                    if all(x == '' for x in row) != True:
                        if row[0] == "":
                            if any(string in element for element in row for string in ["NW","n-i-p","p-i-n"]) == True:
                                for i,var in enumerate(smu_ind):
                                #print("smu_ind " + str(var) + " --> produces for -1 = " + str(row[var-1]) +" for 0" + str(row[var]) + " for +1" + str(row[var+1]) )
                                    
                                    if row[var] == "":
                                        print(row)
                                        if "NW" in row[var-1]:
                                            flat_data['positions']["pos"+str(1+var-min(smu_ind))]['NW'] = row[var-1]
                                        if "n-i-p" in row[var-1] or "p-i-n" in row[var-1]:
                                            flat_data['positions']["pos"+str(1+var-min(smu_ind))]['NW Orientation'] = row[var-1]
                                    elif "NW" in row[var]:
                                        if "NW" in row[var]:
                                            flat_data['positions']["pos"+str(1+var-min(smu_ind))]['NW'] = row[var]
                                        if "n-i-p" in row[var] or "p-i-n" in row[var]:
                                            flat_data['positions']["pos"+str(1+var-min(smu_ind))]['NW Orientation'] = row[var]
                        if row[0] != "":
                            break
                except:
                    print("WARN: First run row is likely empty")
                    break    
                    
            for row in rows:
                if all(x == '' for x in row) != True:
         
                    #This section restructures "position" and "nw" to fit with the format desired.    
                    if row[0] !="":
                        run_key = str(int(row[0]))
                        row_dict = {}
                        for i, header in enumerate(keys):
                            if i in smu_ind:
                                pos_idx = 1+i-min(smu_ind)
                                nw_key = 'pos' + str(pos_idx)
                                row_dict[nw_key] = {'SMU': flat_data['positions']['pos' + str(pos_idx)]['SMU'], 'NW': flat_data['positions']['pos' + str(pos_idx)]['NW']}
                                    
                            row_dict[header] = row[i]
                                
                        flat_data[run_key] = row_dict
                        if "device" in flat_data[run_key].keys() and "Device" not in flat_data[run_key].keys():
                            flat_data[run_key]["Device"] = flat_data[run_key]["device"]
                            del flat_data[run_key]['device']
                        if "device" not in [entry.lower() for entry in list(flat_data[run_key].keys())]:
                            flat_data[run_key]["Device"] = 'Unlabelled'
                        if type(flat_data[run_key]["Device"]) != str:
                            flat_data[run_key]["Device"] ="Unlabelled"
            file_data = flat_data
            
        data[filename] = file_data
            
    data_files = [file for file in files if "LOG" not in file ]    
# Loop over each file and each sheet within the file
    for file in data_files:
        file_data = {}
        filename = os.path.splitext(file.split('\\')[-1])[0]
        xls = xlrd.open_workbook(file)
        
        #First we need to find and import the settings:
        settings_sheet = [sheet for sheet in xls.sheet_names() if "settings" in sheet.lower()]
        run_sheets     = [sheet for sheet in xls.sheet_names() if "run" in sheet.lower()]
        data[filename] = {}
        """
        Settings Sheet handling
        Each xls file has one sheet called "Settings", this will need to be read in a row by row format.
        Column headers are no longer important
        """
        #####!!!!#####!!! use "string".isDigit() to figure out if it can be a float. Then do list comprehension of contents of string lists.  
        # Otherwise, read data as nested sheets
        for sheet_name in settings_sheet:
            sheet = xls.sheet_by_name(sheet_name)
            # Read data row by row into a list
            rows = []
            for row_index in range(sheet.nrows):  # Exclude header row
                row_data = sheet.row_values(row_index)
                rows.append(row_data)
                
            settings_data = {}
            for row in rows:
                if any("===" in s for s in row) or (all(not s.strip() for s in row)):  #We strip out all the === that separates the Settings.
                    
                    continue
        
                if any("Run" in s for s in row): # We detect where to swap key into a new settings file
                    key = str(row[0])
                    settings_data[key] = {}
                    continue
                
                header = row[0]
                
                if len(row) > 1:
                    values = row[1:]
                    values = [x for x in values if x != '']
                    if len(values) == 1:
                        settings_data[key][header] = str(values[0])
                    else:
                        settings_data[key][header] = str(values)
            file_data[sheet_name] = settings_data
            
            
        """
        Data sheet handling
        """
        # Otherwise, read data as nested sheets
        for sheet_name in run_sheets:
            sheet = xls.sheet_by_name(sheet_name)
            
            if sheet_name != "Calc":
                # Read data column by column into a dictionary with the first item in each column as the key
                cols = {}
                cols["col headers"] = []
                for col_index in range(sheet.ncols):
                    col_data = [x for x in sheet.col_values(col_index) if x != '']
                    key = col_data.pop(0)
                    if len(col_data) == 1:
                        col_data = col_data[0]
                    cols[key] = col_data 
                   
                    if (key.endswith("V") == True and any(s.lower() in key.lower() for s in ["START","STOP"])==False):
                        cols['voltage'] = col_data
                        key = 'voltage'
                    elif (key.endswith("I") == True and any(s.lower() in key.lower() for s in ["START","STOP"])==False):
                        cols['current'] = col_data
                        key = 'current'
                    if "SD" in key and "voltage" in key:
                        cols['voltage'] = col_data
                        key = 'voltage'
                    if "SD" in key and "current" in key:
                        cols['current'] = col_data
                        key = 'current'
                        
                            
                    cols["col headers"].append(key)
                
                stats    = {"Npts"   : file_data["Settings"][sheet_name]["Number of Points"].strip("[]").replace('\'','').split(', '),
                            "VStep"  : file_data["Settings"][sheet_name]["Step"].strip("[]").replace('\'','').split(', ') ,
                            "VStart" : file_data["Settings"][sheet_name]["Start/Bias"].strip("[]").replace('\'','').split(', '),
                            "VStop"  : file_data["Settings"][sheet_name]["Stop"].strip("[]").replace('\'','').split(', '),
                            "OpMode"  : file_data["Settings"][sheet_name]["Operation Mode"].strip("[]").replace('\'','').split(', '),
                            "NWID"  : file_data["Settings"][sheet_name]["Name"].strip("[]").replace('\'','').split(', '),
                            "SMU"  : file_data["Settings"][sheet_name]["Instrument"].strip("[]").replace('\'','').split(', '),
                            "FBSweep": file_data["Settings"][sheet_name]["Dual Sweep"].strip("[]").replace('\'','').split(', ')}
                
                for key in stats.keys():
                    for n,element in enumerate(stats[key]):
                        if "Npts" in key:
                            try:
                                stats[key][n] = int(element)
                            except:
                                stats[key][n] = 0
                        if key in ["Vstep","VStart","VStop","VStep"]:
                            try: 
                                stats[key][n] = float(element)
                            except:
                                stats[key][n] = None
                        if key in ["FBSweep"]:
                            if element == "Enabled":
                                stats[key][n] = True
                            else:
                                stats[key][n] = False
                for key in stats.keys():
                    if len(stats[key]) == 1:
                        stats[key] = stats[key][0]
                
                    
                #Swap linear sweep data to segmented data
                main_col = stats['Npts'].index(max(stats['Npts']))    
                if stats["FBSweep"][main_col] == True and stats['Npts'][main_col] == 2*(1+int(abs(stats["VStart"][main_col] - stats["VStop"][main_col])/abs(stats["VStep"][main_col]))):
                    sweep_indices = [0,int(stats['Npts'][main_col]/2),stats['Npts'][main_col]]
                else:
                    sweep_indices = [0,max(stats["Npts"])]
                print(sheet_name+": "+file_data["Settings"][sheet_name]["Operation Mode"] )    
                if  "Voltage Linear Sweep" in file_data["Settings"][sheet_name]["Operation Mode"]:
                    
                    list_keys = [key for key, value in cols.items() if isinstance(value, list) and "headers" not in key]
                    for key in list_keys:
                        cols[key]  = segment_sweep(cols[key],sweep_indices)
                        
                        
                        
                    # Store the data in the dictionary
                    file_data[sheet_name] = cols
                
                if all(any(i in j for j in ["Voltage Bias", "Voltage List Sweep" ]) for i in stats["OpMode"]):
                    #Determine which nanowire is the emitter, and which is the detector:
                    emitter_ID = stats["OpMode"].index('Voltage List Sweep')
                    detector_ID = stats["OpMode"].index('Voltage Bias')
                    emitter_OP = "Voltage List Sweep"; detector_OP = "Voltage Bias"
                
                if all(any(i in j for j in ["Voltage Bias","Voltage Linear Sweep"]) for i in stats["OpMode"]):
                    #Determine which nanowire is the emitter, and which is the detector:

                    emitter_ID = stats["OpMode"].index('Voltage Linear Sweep')
                    detector_ID = stats["OpMode"].index('Voltage Bias')
                    emitter_OP = "Voltage Linear Sweep"; detector_OP = "Voltage Bias"
                    
                try:
                    cols["emitter"] = {"SMU":stats["SMU"][emitter_ID],'NWID':stats["NWID"][emitter_ID].split(' ')[0],"OpMode":emitter_OP}
                    cols["detector"] = {"SMU":stats["SMU"][detector_ID],'NWID':stats["NWID"][detector_ID].split(' ')[0],"OpMode":detector_OP}
                except:
                    None
                    
                # Store the data in the dictionary
                list_keys = [key for key, value in cols.items() if isinstance(value, list) and "headers" not in key]
                for key in list_keys:
                    cols[key] = cols[key]
                file_data[sheet_name] = cols    
        
        # Store the data for the file in the top-level dictionary
        data[filename] = file_data

        
    #Now we need to check the logbook run info against all included excel sheets so that we can import the correct logbook data.
    log_keys  = [x for x in data.keys() if "LOG" in x.upper()]
    data_keys = [x for x in data.keys() if "LOG" not in x.upper()]

    
    for log_key in log_keys: 
        for lkey in data[log_key].keys():
            for key in data_keys:
                if key == log_key:
                    continue
                
                #Checks RunID in the logbook against the sheetnames to move the logbook into the right dict.
                if any(lkey in s for s in data[key]):
                    for runID in data[key].keys():
                        if str(lkey) in str(runID):
                            data[key][runID]['LOG'] = data[log_key][lkey]
                            continue
        
        del(data[log_key])
    #Now we need to move the settings file into the 
    for rkey in data.keys():
        for runID in data[rkey]["Settings"].keys():
            if runID in data[rkey].keys():
                data[rkey][runID]['Settings'] = data[rkey]["Settings"][runID]
                continue
        del(data[rkey]["Settings"])
    return(data)

def Nanonis_dat_read(file,**kwargs):
    """
    Reads the data structure in nanonis data files, and outputs a dictionary containing the data. 
    """
    Raw   = {}
    if 'portformat' not in list(kwargs.keys()):
        portF = False

    elif 'portformat' in  list(kwargs.keys()):
        portF = kwargs['portformat']
        
    
    #First we find the first level dict we need to save. There are two options, predata or [Data]
    with open(file,'r') as f:
        line1 = f.readline()
        if line1.startswith('[') == True and line1.endswith == ']\n':
            topdict = line1
        else:
            topdict = '[Pre-Data]'
        Raw[topdict] = {}
        
        f.close()
 
    
    with open(file,'r') as f:        
        for line in f:
            line = line.strip()
            if line.startswith('[') == True and line.endswith(']'):
                topdict = line 
                Raw[topdict] = {}
                l2 = f.readline()
                items = l2.split('\t')
                for item in items:
                    Raw[topdict][item] = []
                    
            else:
                ld = line.split('\t')
                
                if topdict == '[Pre-Data]':

                    try:
                        Raw[topdict][ld[0]] = ld[1]
                    except:
                        if ld[0] != '\n':
                            Raw[topdict][ld[0]] = None
                else:
                    for num,item in enumerate(ld):
                        try:
                            Raw[topdict][items[num]].append(float(ld[num]))
                        except:
                            Raw[topdict][items[num]].append(ld[num])
            
                
        if portF == True:
            keylist = []
            
            for key in Raw['[Pre-Data]'].keys():
                if 'Comment' in key:
                    keylist.append(key)
                    
            for key in keylist:
                
                cdata = Raw['[Pre-Data]'][key].split(' : ')
                
                cdstr = "".join(cdata).lower()
                
                if "device" in cdstr:
                    itlist = cdstr.split(' ; ')
                    Raw['[Pre-Data]'][itlist[0]] = itlist[1]
                
                if "light" in cdstr:
                    try:
                        itlist = cdstr.split(' ; ')
                        if itlist[1].lower() in ['false','fal','off','none',None,'no']:
                            itlist[1] = False
                        
                        elif itlist[1].lower() in ['true','tru','on','light','yes']:
                            itlist[1] = True
                            
                        Raw['[Pre-Data]'][itlist[0]] = itlist[1]
                    except:
                        None
                    
                else:
                    Raw['[Pre-Data]'][cdata[0]] = {'NW':None,'bias':False,'sweep':False,'current':False,'ground':False,'pos':None}
                
                for item in cdata:
                    try:
                        itlist = item.split(' ; ')
                    except:
                        itlist = item
                        
                    if "NW" in item:   
                        Raw['[Pre-Data]'][cdata[0]][itlist[0]] = itlist[1]
                    
                    if 'bias' in item:
                        Raw['[Pre-Data]'][cdata[0]][itlist[0]] = itlist[1]
                        
                    if 'sweep' in item:
                        Raw['[Pre-Data]'][cdata[0]][itlist[0]] = json.loads(itlist[1])
                        
                    if 'current' in item:
                        Raw['[Pre-Data]'][cdata[0]][itlist[0]] = True
                        
                    if  'ground' in item:
                        Raw['[Pre-Data]'][cdata[0]][itlist[0]] = True
                        
                    if 'pos' in item:
                        Raw['[Pre-Data]'][cdata[0]][itlist[0]] = int(itlist[1])
        Raw['info'] = {}
        Raw['info']['filename'] = file.split('\\')[-1]
        Raw['info']['path'] = file
        
        return(Raw)
                
   
   
#%%
def txtparse(**kwargs):
    kwargdict = {'file':'file','fn':'file','f':'file'}
    kw = KwargEval(kwargs, kwargdict,file=None)
    S_ESC = LinWin()
    if kw.file == None:
        root = tk.Tk()
        file_path = askopenfilename(title = 'Select a file to load',filetypes=[('txt file','*.txt'),('All Files','*.*')]).replace('/',S_ESC)    
        tk.Tk.withdraw(root)
        kw.file,kw.pathtype = Rel_Checker(file_path)
    else:
        kw.file,kw.pathtype = Rel_Checker(kw.file)
    Raw = {}
    top_dict = []
    with open(kw.file,'r') as f:
        for line in f:
            if line.startswith('[') == True and line.endswith(']\n') == True:
                top_dict.append(line)
                Raw[top_dict[-1].split('[')[-1].split(']')[0]] = []
            if top_dict[-1] not in line:
                linestr = line.strip()
                if len(linestr) != 0:
                    Raw[top_dict[-1].split('[')[-1].split(']')[0]].append(linestr)
        out_dict = {}
        for key in Raw.keys():
            skdict = {}
            for item in Raw[key]:
                if len(item.split('=')) !=2:
                    if type(skdict) == dict:
                        skdict = []
                    delim = ''.join([' ' for n in range(maxRepeating(item,guess=' ')[1])])  
                    field_item = item.split(delim)
                    skdict.append(field_item)
                    
                elif len(item.split('=')) == 2:
                    try: 
                        field_item = item.split('=')
                        skdict[field_item[0]] = field_item[1]
                        spctot =  maxRepeating(field_item[1],guess=' ')[1]
                        if spctot >= 1:
                            try:
                                delim = ''.join([' ' for n in range(maxRepeating(item,guess=' ')[1])])  
                                float(field_item[1].split(delim)[0])
                                ftrue = True
                            except:
                                ftrue = False
                        if ftrue == True:
                            skdict[field_item[0]] = np.array(field_item[1].split(delim),dtype='float')
                        else:
                            skdict[field_item[0]] = field_item[1]
                    except:
                        if type(skdict) == dict and '=' in item:
                            skdict[item.split('=')[0]] = item.split('=')[1]

            if type(skdict) == list:
                skdict = np.array(skdict)
            out_dict[key] = skdict 
                 
                    
                    
    return(out_dict)

def Ideality_Factor(I,V,**kwargs):
    kwargdict = {'T':'T','temp':'T','temperature':'T',
                  'fit_range':'fit_range','fr':'fit_range',
                  'plot_range':'plot_range','pr':'plot_range',
                  "data_range":"data_range","dr":"data_range",
                  'N':'N','pts':'N',
                  "p0":"p0",'guess':"p0"}
    
    kw = KwargEval(kwargs, kwargdict,T=273,fit_range=None,plot_range=None,N=200,p0=None,data_range=None)
    
    q = constants.e
    k = constants.Boltzmann
    
    def Diode_EQ(V,I_0,n):
        return(I_0 * np.exp((q*V)/(n*k*kw.T)))    
    

    if max(I) < np.abs(min(I)):
        I = list(np.multiply(-1,I))
        V = list(np.multiply(-1,V))

    if V[-1]<V[0]:
        V = np.flip(V)
        I = np.flip(I)
   
        
    if kw.fit_range == None:
        kw.fit_range = 0.01
    
    
    if type(kw.fit_range) == float:
        I_range = [kw.fit_range*max(I),max(I)]
        
        v_i =  next((i for i, x in enumerate(I) if x > I_range[0]), -1)
        v_f =  next((i for i, x in enumerate(I) if x > I_range[1]), -1)
        kw.fit_range = [V[v_i],V[v_f]]
        

    
    elif type(kw.fit_range) == list:        
        v_i =  next((i for i, x in enumerate(V) if x > kw.fit_range[0]), -1)
        v_f =  next((i for i, x in enumerate(V) if x > kw.fit_range[1]), -1)
     
    V_fit = V[v_i:v_f]
    I_fit = I[v_i:v_f]
     
    if kw.plot_range == "all":
        kw.plot_range = [min(V),max(V)]
    
    elif kw.plot_range == "positive":
        kw.plot_range = [0,max(V)]
        
    elif kw.plot_range == None:
        kw.plot_range = [min(V_fit),max(V_fit)]
        
    if type(kw.data_range) == list:
        v_id =  next((i for i, x in enumerate(V) if x > kw.data_range[0]), -1)
        v_fd =  next((i for i, x in enumerate(V) if x > kw.data_range[1]), -1)
        
    if kw.data_range == "positive":
        v_id =  next((i for i, x in enumerate(V) if x > 0), -1)
        v_fd =  next((i for i, x in enumerate(V) if x > max(V)), -1)
     
        V_data = V[v_id:v_fd]
        I_data = I[v_id:v_fd]
    else:
        V_data = V
        I_data = I
    
    if kw.p0 == None:
        kw.p0 = [kw.fit_range[0]*max(np.abs(I)),3] 
    

    popt, pcov = scipy.optimize.curve_fit(Diode_EQ, V_fit, I_fit,p0=kw.p0)
    V_new  = np.linspace(kw.plot_range[0],kw.plot_range[1],kw.N)
    I_new  = Diode_EQ(V_new, *popt)


    return({"n":popt[1],"V_new":V_new,"I_new":I_new,"V_fit":V_fit,"I_fit":I_fit,'V':V,'I':I,'V_data':V_data,"I_data":I_data})

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