import numpy as np
import json
import os
from matplotlib import pyplot as plt
from DMU import plot_utils as dmp
from DMU import graph_styles as gss
from DMU import utils as dm

def json_loaddata(filepath,importtype = "normal"):
    if importtype == "normal":
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
        return(existing_data)

### UTILITY  and SAVING ###
def json_savedata(data, filename, directory=os.getcwd(), overwrite=False):
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if filename in files and not overwrite:
        print("RENAME YOUR FILE OR RUN DELETE IN NEXT CELL")
    else:
        pathname = os.sep.join([directory, filename]) 
        with open(pathname, "w") as f:
            json.dump(convert_to_json_serializable(data), f, indent=2)
            
def convert_to_json_serializable(item):
    if isinstance(item, dict):
        return {key: convert_to_json_serializable(value) for key, value in item.items()}
    elif isinstance(item, np.ndarray):
        try:
            return item.unique().tolist()
        except AttributeError:
            return item.tolist()
    elif isinstance(item, (list, tuple)):
        return [convert_to_json_serializable(element) for element in item]
    elif isinstance(item, (int, float)):
        return item
    elif callable(getattr(item, 'tolist', None)):
        return item.tolist()
    else:
        return str(item)
            
datafilepath = os.sep.join([os.getcwd() + r"\DataStorage\Ideality"])
DATA = {}

for file in [f for f in os.listdir(datafilepath) if f.endswith(".json")]:    
    DATA[file.replace(".json","")] = json_loaddata(os.sep.join([datafilepath,file]))

#"Auto" Idealities Only
IDF = dict(Auto=dict(),All=dict(),Best=dict(),Tabulated = dict(),Everything=dict())

#limit_device     = ["DFR1-GG","DFR1-FJ","DFR1-HE"]
limit_device     = ["DFR1-JI"]
exclude_material = ["PMMA"] 

tt = "All"
dummy_dict = {"NoRun":{"editdict":{"n":None,"mrange":"None","light_on":None,"accept":None}}} #Made so we can append a single empty run if NW2 was not measured

for dname,device in DATA.items():
    
        IDF["Everything"][dname] = {"n list":[],"device":[],"NWID":[], "Rs":[],"Range":[],"RunID":[],"Light":[]}
        IDF["All"][dname] = {"n list":[],"device":[],"NWID":[], "Rs":[],"Range":[],"RunID":[],"Light":[]}
        IDF["Auto"][dname] = {"n list":[],"device":[],"NWID":[], "Rs":[],"Range":[],"RunID":[],"Light":[]}
        IDF["Tabulated"][dname] = {}
        for sname,sdevice in device.items():
            
                IDF["Tabulated"][dname][sname] = {}
                NWkeys = ["NW1","NW2"]
                for NWID in NWkeys:
                    if NWID not in sdevice.keys():
                        sdevice[NWID] = dummy_dict
                    tt = "Everything"
                    for runID, run in sdevice[NWID].items():
                       
                            IDF[tt][dname]["n list"].append(run["editdict"]["n"])
                            IDF[tt][dname]["device"].append(sname)
                            IDF[tt][dname]["NWID"].append(NWID)
                            IDF[tt][dname]["RunID"].append(runID)
                            IDF[tt][dname]["Range"].append(run["editdict"]["mrange"])
                            IDF[tt][dname]["Light"].append(run["editdict"]["light_on"])
                            try:
                                Vfit = run["editdict"]["ideality_fit"]["Vfit"]
                                Ifit = run["editdict"]["ideality_fit"]["Ifit"]
                                Rs = (np.mean(np.gradient(tuple(Vfit[-100:]),tuple(Ifit[-100:]))))
                            except:
                                Rs = "N/A"
                                
                            IDF[tt][dname]["Rs"].append(Rs)
                    
                    tt = "All"
                    
                    for runID, run in sdevice[NWID].items():
                            if run["editdict"]["accept"] == True:
                                IDF[tt][dname]["n list"].append(run["editdict"]["n"])
                                IDF[tt][dname]["device"].append(sname)
                                IDF[tt][dname]["NWID"].append(NWID)
                                IDF[tt][dname]["RunID"].append(runID)
                                IDF[tt][dname]["Range"].append(run["editdict"]["mrange"])
                                IDF[tt][dname]["Light"].append(run["editdict"]["light_on"])
                                try:
                                    Vfit = run["editdict"]["ideality_fit"]["Vfit"]
                                    Ifit = run["editdict"]["ideality_fit"]["Ifit"]
                                    Rs = (np.mean(np.gradient(tuple(Vfit[-100:]),tuple(Ifit[-100:]))))
                                except:
                                    Rs = "N/A"
                    tt = "Auto"
                    for runID, run in sdevice[NWID].items():
                            if (run["editdict"]["accept"] == True) & (run["editdict"]["mrange"] == "Auto"):
                                IDF[tt][dname]["n list"].append(run["editdict"]["n"])
                                IDF[tt][dname]["device"].append(dname.split("_")[0] + sname)
                                IDF[tt][dname]["NWID"].append(NWID)
                                IDF[tt][dname]["RunID"].append(runID)
                                IDF[tt][dname]["Range"].append(run["editdict"]["mrange"])
                                IDF[tt][dname]["Light"].append(run["editdict"]["light_on"])
                                try:
                                    Vfit = run["editdict"]["ideality_fit"]["Vfit"]
                                    Ifit = run["editdict"]["ideality_fit"]["Ifit"]
                                    Rs = (np.mean(np.gradient(tuple(Vfit[-100:]),tuple(Ifit[-100:]))))
                                except:
                                    Rs = "N/A"
                
                    tt = "Tabulated"
                    IDF[tt][dname][sname][NWID] = {}
                    nind   = []
                    mrange = [] 
                    runlist    = []
                    light  = []
                    for runID, run in sdevice[NWID].items():
                        if ((isinstance(run["editdict"]["n"],float)) & (run["editdict"]["accept"] == True)):
                            nind.append(run["editdict"]["n"])
                            mrange.append(run["editdict"]["mrange"])
                            runlist.append(runID)
                            light.append(run["editdict"]["light_on"])
                        
                    if len(nind) != 0:
                        IDF[tt][dname][sname][NWID]["n"]      = np.array(nind)
                        IDF[tt][dname][sname][NWID]["mrange"] = np.array(mrange)
                        IDF[tt][dname][sname][NWID]["RunID"]  = np.array(runlist)
                        IDF[tt][dname][sname][NWID]["Light"]  = np.array(light)
                        nind_list = np.array([n for n in nind if not isinstance(n,(type(None),str))])
                        if len(nind_list) >=1:
                            IDF[tt][dname][sname][NWID]["n best"]      = np.min(nind_list)
                            IDF[tt][dname][sname][NWID]["range best"]  = np.array(mrange)[np.where(np.array(nind)==np.min(nind_list))[0][0]]
                            IDF[tt][dname][sname][NWID]["run best"]  = np.array(runlist)[np.where(np.array(nind)==np.min(nind_list))[0][0]]
                            IDF[tt][dname][sname][NWID]["light"]  = np.array(light)[np.where(np.array(nind)==np.min(nind_list))[0][0]]
                    else:
                       IDF[tt][dname][sname][NWID]["n best"]      = "None"
                       IDF[tt][dname][sname][NWID]["range best"]  = "None"
                       IDF[tt][dname][sname][NWID]["run best"]  = "None"
                       IDF[tt][dname][sname][NWID]["light"]  = "None"
                        


Tablestring = "\\begin{table}\n\\begin{tabular}{ccccccc}\n" 
Tablestring += "Device & Subdevice & NWID & RunID & Ideality & Range & Light \\\\ \n"
last_subdevice = ""
last_NW = ""
ttt = "Tabulated"

for dname,device in IDF["Tabulated"].items():
    Firstline = True
    Tablestring+= dname 
    for sname,subdevice in device.items():
        
        Tablestring+= "&" + sname
        Secondline = True
            
        for NWID,NWData in subdevice.items():           

            if Secondline:
                Tablestring+= "&" + NWID
            else:
                Tablestring+= "& & "+ NWID
            
            try:
                nbest =  "{:.4f}".format(NWData["n best"]) 
            except:
                nbest = NWData["n best"]
                
            Tablestring += " & " + NWData["run best"] + " & " + nbest + " & " + NWData["range best"] +" & " + str(NWData["light"]) + "\\\\ \n"
            Firstline  = False
            Secondline = False
    Tablestring += "\\\\ \n"

Tablestring += "\\end{tabular} \n \\end{table}"    



with open(os.sep.join([os.getcwd(),"Tabulated","DFR1_Everything_"+ttt+".json"]),"w") as f:
    f.write(Tablestring)        


Tablestring = "\\begin{table}\n\\begin{tabular}{cccccccc}\n" 
Tablestring += "Device & Subdevice & NWID & RunID & Ideality & Range & Light & R0 [MOhm] \\\\ \n"
last_subdevice = ""
last_NW = ""
ttt = "Everything"

for dname,device in IDF[ttt].items():
    Tablestring+= dname 
    for i,run in enumerate(device["device"]):
        if last_subdevice != device["device"][i]:
            last_subdevice = device["device"][i]
            Tablestring += " & " + last_subdevice
        else:
            last_subdevice = device["device"][i]
            Tablestring += " & " 
        
        if last_NW != device["NWID"][i]:
            last_NW = device["NWID"][i]
            Tablestring += " & " + last_NW
        else:
            last_NW = device["NWID"][i]
            
            Tablestring += " & " 
        
        Tablestring += " & " + device["RunID"][i]
        if not isinstance(device["n list"][i],(str,type(None),list)):
            Tablestring += " & " + "{:.4f}".format(device["n list"][i])
        else:
            Tablestring += " & None "
        Tablestring += " & " + device["Range"][i]
        
        Tablestring += " & " + str(device["Light"][i])

        Tablestring += "\\\\ \n"

Tablestring += "\\end{tabular} \n \\end{table}"    



with open(os.sep.join([os.getcwd(),"Tabulated","DFR1-Everything_"+ttt+".json"]),"w") as f:
    f.write(Tablestring) 

#%%


materials = ["AIR","PMMA","ALD"]
DD = {}
for dname,device in IDF["Tabulated"].items():
    if dname not in DD.keys():
        DD[dname] = {}
    for sname,subdevice in device.items():
        
        mat = sname.split("_")[1]
        
        if mat not in DD[dname].keys():
            DD[dname][mat] = []
            
        for NWID,NWData in subdevice.items():      
            print(NWData["n best"])
            if NWData["n best"] != "None":                
                DD[dname][mat].append(NWData["n best"])

"""
Each device occupies n->n+1, each new material has a 1 gap, each new device a 2 gap
"""

FIG = dm.ezplot()
ax = FIG.ax[0]
cmap = dmp.get_tab20bc(output="cmap",grouping="all")

gss.graph_style("PP1_Wide") # Comment out to customise your own style
        
DefBBOX = gss.DEF_BBOX(style="PP1_Wide",bboxstyle="symmetric")

cn = -1

pcount = -2 
tickname = []
tickposition = []
tickdevice = []
device_position = [-2]
def sorted_from_middle(lst, reverse=False):
    if len(lst) <= 1:
        return lst
    tail = sorted([lst[-1], lst[0]], reverse=reverse)
    return sorted_from_middle(lst[1:-1], reverse) + tail
Chrono = ["EG","CE","FF","IG","FF","IG","GG","FJ","HE","FE","GK","JI"]
DKEY = ["DFR1-"+ch for ch in Chrono]
for device in DKEY:
    pcount+=1
    cn+=1
    mn = 0
    for mname,mat in DD[device].items():
        mat = np.array(mat)
        mat = mat[np.where(mat<10)[0]]
        
        ub = .95
        lb = .05
        mod=-.5
        if "PMMA" in DD[device].keys():
            if mn == 0:
                mnmod = -0.2
            if mn == 1:
                mnmod = 0.2
        else:
            mnmod = 0.1
                
        dx = np.array(sorted_from_middle(list(np.linspace(lb,ub,len(mat)+2))))[:len(mat)]+pcount+mod +mnmod
        
        y = np.array(mat)
        if len(y) != 0:
            if cn == 10:
                cn=0
            ax.plot(dx,y,"x",c=cmap(cn*4+mn),markersize=8,markeredgewidth=2)
            tickname.append(mname)
            tickposition.append(pcount+mnmod)
            mn += 1
            pcount+=1
    if len(y) !=0:
        tickdevice.append(device.replace("DFR1-",""))
        device_position.append(pcount)
    
ax.set_ylabel("Ideality Factor")     
ax.set_xticks(tickposition)  # Set tick positions
ax.set_xticklabels(tickname,rotation=90)  # Set tick labels
ax.tick_params(axis="x", which="both", length=0)  # No major or minor tick marks
for i in range(len(device_position)-1):
    ax.annotate(tickdevice[i],xy=(np.mean([device_position[i+1],device_position[i]]),8),annotation_clip=False,ha="center",)
ax.set_xlim([-2,np.max(device_position)])
# Highlight each device column with alternating colors
highlight_colors = ["white", "lightgrey"]
for i in range(len(device_position) - 1):
    start = device_position[i]
    end = device_position[i + 1]
    color = highlight_colors[i % len(highlight_colors)]
    ax.axvspan(start, end, color=color, alpha=0.3)  # Adjust alpha for transparency
    
bbox = [DefBBOX[0]-0.07,DefBBOX[1]+0.07,DefBBOX[2]+0.05,DefBBOX[3] + 0.1]
FIG.apply_bbox(0,bbox)
FIG.fig.savefig("IdealiySpread.pdf")