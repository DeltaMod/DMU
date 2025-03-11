import os
import sys
import xlrd
import matplotlib.cm as mcm
import matplotlib as mpl

GHIN = False

if GHIN:
    sys.path.append(r"C:\Users\vidar\Documents")
    import GitHub.DMU.DMU.utils as dm
    import GitHub.DMU.DMU.plot_utils as dmp
    import GitHub.DMU.DMU.graph_styles as gss
else:
    from DMU import utils as dm
    from DMU import plot_utils as dmp
    from DMU import graph_styles as gss

import numpy as np 
import matplotlib.pyplot as plt
#%%

gss.graph_style("PP1_Wide") # Comment out to customise your own style
        

DefBBOX = {"single":gss.DEF_BBOX(style="PP1_Wide",bboxstyle="symmetric"),"communication":gss.DEF_BBOX(style="PP1_Wide",bboxstyle="right asymmetric")}
   
data = dm.Keithley_xls_read(None)

if "exclude_title" in locals():
    title_tag = False
else:
    title_tag = True
#%% 
FIG   = [dm.ezplot()]; FIG[0].hold_on = True    
idFIG = []

ind = 0; iind =0;
idict = {"NW1":{"fit_range":[0.5,1],"n0":3},"NW2":{"fit_range":[0.5,1.5],"n0":2.5}} #NW1 and NW2 ideality fit parameters - change these so that the ideality region and estimated n0 fits your data.    
for dkey in data.keys():
    if "LOG" not in dkey:
        for key in [k for k in data[dkey].keys() if "Run" in k]: 
            d = data[dkey][key]
            if not FIG[ind].hold_on:
                FIG.append(dm.ezplot())
                ind += 1 
            
            filename = str(dkey+" "+key).replace("_", " ") +" "+ d["Operation"]
            FIG[ind].file = filename
            
            #Attempt to find the device name. If it can't be found, assign "Log Missing" as the device name
            try:
                device   = d['LOG']['Device']
                NWID = d["emitter"]["NWID"]
            except:
                NWID = "NW1" #This is set so that ideality plots use NW1 settings as a fallback
                device="Log Missing"
            
            # Look in the settings file generated automatically by the Keithley system. If "Voltage List Sweep" is there, then we use this preset legend configuration
            if d["Operation"] in ["Voltage List Sweep"]:
                bbkey = "communication"
                legend_loc="upper center"
                legend_off=(0,0.015,0,0)
                ncols=3
            else:
                legend_loc="best"
                legend_off=(0,0,0,0)
                bbkey = "single"
                ncols=1
                
            
            IFIG,IDF = dm.bias_plotter(d,FIG[ind],tool='Keithley',title=filename+' dev:'+device,c=['b','r','g'],
                                       ideality=True,altplot=True,idict = idict[NWID],plot_fit_points=True,
                                       legend_loc=legend_loc,legend_off=legend_off,ncols=ncols)
            if title_tag:
                KeithDL = dmp.Keithley_Plot_Tagger(FIG[ind],d)
                bbox_offset = (0,0,0,-0.05)
            else:
                bbox_offset = (0,0,0,0)

            bbox = FIG[ind].ax[0].get_position()
            bbox.x0 = DefBBOX[bbkey][0];  bbox.x1 = DefBBOX[bbkey][1];
            bbox.y0 = DefBBOX[bbkey][2]; bbox.y1 = DefBBOX[bbkey][3]+bbox_offset[3];
            FIG[ind].ax[0].set_position(bbox)


            if IFIG != False:
                idFIG.append(IFIG)
                idFIG[iind].IDF = IDF
                idFIG[iind].file = filename+'_ideal'
                if title_tag:
                    KeithDL = dmp.Keithley_Plot_Tagger(idFIG[iind],d)
                    bbox_offset = (0,0,0,-0.05)
                else:
                    bbox_offset = (0,0,0,0)
                
                bbox = idFIG[iind].ax[0].get_position()
                bbox.x0 = DefBBOX[bbkey][0];  bbox.x1 = DefBBOX[bbkey][1];
                bbox.y0 = DefBBOX[bbkey][2]; bbox.y1 = DefBBOX[bbkey][3];
                idFIG[iind].ax[0].set_position(bbox)
                
                iind +=1
            #Check if the figure is meant to "HOLD ON", and if not, make a new one
            
                
#%%

savefig = True
figformat = [".png",".pdf"]
if savefig == True:
    for fig in FIG:
        for ext in figformat:
            fig.fig.savefig("Figures\\"+fig.file+ext)
    for fig in idFIG:
        for ext in figformat:
            fig.fig.savefig("Figures\\"+fig.file+ext)

"""
Information that is displayed in each image:
    Device Info: Device Name (DFR1-GG-BR1) || NW ID (NW1//NW2) || Attempt to show Date of Data Plotted (2024-01-01)
    Type of Plot: Ideality // Voltage Sweep // Pulse Sweep // Pulse Ladder || Light Condition: Light on//Light Off
    File Info: Filename
    
We don't want to make this another addition to the plotter, so we instead write this into the dm.plot_utils where you can pass the fig and data dict to e.g. func(FIG[id],d = data[dkey][key]). All the data should be listed in the "settings"
This also means that you can simply turn off labelling
"""
