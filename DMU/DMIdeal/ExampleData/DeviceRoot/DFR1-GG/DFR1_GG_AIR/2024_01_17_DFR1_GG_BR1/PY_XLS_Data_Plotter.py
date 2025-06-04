import os
import sys
import xlrd
import matplotlib.cm as mcm
import matplotlib as mpl

# sys.path.append(r"C:\Users\vidar\Documents\GitHub\DMU\DMU") #Home PC
sys.path.append(r"C:\Users\vidar\Documents\GitHub\DMU\DMU") #Office PC
import utils as dm
#from DMU import utils as dm
from DMU import graph_styles as gss
import numpy as np 
import matplotlib.pyplot as plt

#%%
plt.rcParams.update({
    		"text.usetex": True,
    		"font.family": "serif",
    		"font.serif": ["CMU"],
    		"font.size": 22,
    		"axes.grid.which":'both', 
    		"grid.linestyle":'dashed',
    		"grid.linewidth":0.4,
    		"xtick.minor.visible":True,
    		"ytick.minor.visible":True,
    		"figure.figsize":[16/2,9/1.5],
    		'xtick.labelsize':16,
            'ytick.labelsize':16,
            'legend.fontsize':16,
            'figure.dpi':200,   
    		'axes.grid':True,
    		'axes.axisbelow':True,
    		'figure.autolayout':True })

gss.graph_style("WideNarrow") # Comment out to customise your own style
        

    
data = dm.Keithley_xls_read(None)
#%% 
FIG   = [dm.ezplot()]; FIG[0].hold_on = True    
idFIG = []

ind = 0; iind =0;
for dkey in data.keys():
    if "LOG" not in dkey:
        for key in [k for k in data[dkey].keys() if "Run" in k]: 
            d = data[dkey][key]
            if not FIG[ind].hold_on:
                FIG.append(dm.ezplot())
                ind += 1 
            
            filename = str(dkey+" "+key).replace("_", " ")
            FIG[ind].file = filename
            
            try:
                device   = d['LOG']['Device']
                print(filename +"light: "+ str(d['LOG']["Light Microscope"]))
            except:
                device=""
                
            IFIG,IDF = dm.bias_plotter(d,FIG[ind],tool='Keithley',title=filename+' dev:'+device,c=['b','r','g'],ideality=True,altplot=True)
            FIG[ind].ax[0].set_title(filename+' dev:'+device,fontsize=12)
            if IFIG != False:
                idFIG.append(IFIG)
                idFIG[iind].IDF = IDF
                idFIG[iind].file = filename+'_ideal'
                iind +=1
            #Check if the figure is meant to "HOLD ON", and if not, make a new one
            
                
#%%

savefig = False
figformat = [".png",".svg",".eps"]
if savefig == True:
    for fig in FIG:
        for ext in figformat:
            fig.fig.savefig("Figures\\"+fig.file+ext)
    for fig in idFIG:
        for ext in figformat:
            fig.fig.savefig("Figures\\"+fig.file+ext)
                    