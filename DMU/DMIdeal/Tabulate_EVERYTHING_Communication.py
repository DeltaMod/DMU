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
            
datafilepath = os.sep.join([os.getcwd() + r"\DataStorage\Communication"])
DATA = {}
# #%% OLD CODE THAT FIXED PROBLEMS WITH LEVEL PLOTTING
# for filename in [f for f in os.listdir(datafilepath) if f.endswith(".json")] :
#     file = json_loaddata(os.sep.join([datafilepath,filename]))
#     for subdevice in file.keys():
#         for nwID in ["NW1","NW2"]:
#             if nwID in file[subdevice].keys():
#                 for runID in file[subdevice][nwID].keys():
#                     sort_data = {"I_E":[],"V_E":[],"I_R":[]}
#                     avg_data  = {"I_E":[],"V_E":[],"I_R":[]}
#                     emitter_voltage  = file[subdevice][nwID][runID]["data"]["emitter voltage"]
#                     receiver_voltage = file[subdevice][nwID][runID]["data"]["receiver voltage"]
#                     emitter_current  = file[subdevice][nwID][runID]["data"]["emitter current"]
#                     receiver_current = file[subdevice][nwID][runID]["data"]["receiver current"]
                    
#                     for i,EVarr in enumerate(emitter_voltage):        
#                         sort_data["I_E"].append({})
#                         sort_data["V_E"].append({})
#                         sort_data["I_R"].append({})
#                         for j,volt in enumerate(EVarr):
#                             #We go through and find all consecutive values measured at each voltage. This means we build up an average pulse voltage on emitter = average current on Receiver.
#                             rvolt = round(volt,1) 
#                             if rvolt not in sort_data["V_E"][i].keys():
#                                 sort_data["I_E"][i][rvolt] = []
#                                 sort_data["V_E"][i][rvolt] = []
#                                 sort_data["I_R"][i][rvolt] = []
#                             else:
#                                 sort_data["I_E"][i][rvolt].append(emitter_current[i][j])
#                                 sort_data["V_E"][i][rvolt].append(rvolt)
#                                 sort_data["I_R"][i][rvolt].append(receiver_current[i][j])
#                     for i,VEdict in enumerate(sort_data["V_E"]):     
#                         avg_data["I_E"].append([])
#                         avg_data["V_E"].append([])
#                         avg_data["I_R"].append([])
                        
#                         for kkey in VEdict.keys():
#                             #avg_data now contains the average emitter and Receiver current for each unique applied voltage.
#                             avg_data["I_E"][i].append(np.mean(sort_data["I_E"][i][kkey]))
#                             avg_data["V_E"][i].append(np.mean(sort_data["V_E"][i][kkey]))
#                             avg_data["I_R"][i].append(np.mean(sort_data["I_R"][i][kkey]))
#                     file[subdevice][nwID][runID]["data"]["I_E_steps"] = avg_data["I_E"]
#                     file[subdevice][nwID][runID]["data"]["I_R_steps"] = avg_data["I_R"]
#                     file[subdevice][nwID][runID]["data"]["V_E_steps"] = avg_data["V_E"]
#                     file[subdevice][nwID][runID]["data"]["I_E_sort"] = sort_data["I_E"]
#                     file[subdevice][nwID][runID]["data"]["I_R_sort"] = sort_data["I_R"]
#                     file[subdevice][nwID][runID]["data"]["V_E_sort"] = sort_data["V_E"]
#     json_savedata(file, filename, directory=os.getcwd(), overwrite=True)

                    
#%%            
for file in [f for f in os.listdir(datafilepath) if f.endswith(".json")]:    
    DATA[file.replace(".json","")] = json_loaddata(os.sep.join([datafilepath,file]))


#%%
#"Auto" Idealities Only
IDF = dict(Auto=dict(),All=dict(),Best=dict(),Tabulated = dict(),Everything=dict())

#limit_device     = ["DFR1-GG","DFR1-FJ","DFR1-HE"]
#limit_device     = ["DFR1-JI"]
#exclude_material = ["PMMA"] 

tt = "All"
dummy_dict = {"NoRun":{"editdict":{"n":None,"mrange":"None","light_on":None,"accept":None},"data":{"I_E_steps":["None"],"V_E_steps":["None"],"I_R_steps":["None"],"Settings":{"Test Name":"None"}}}} #Made so we can append a single empty run if NW2 was not measured

for dname,device in DATA.items():
    
        IDF["Everything"][dname] = {"n list":[],"device":[],"NWID":[], "Rs":[],"Range":[],"RunID":[],"Light":[],"I_E_steps":[],"V_E_steps":[],"I_R_steps":[],"operation":[]}
        IDF["All"][dname] = {"n list":[],"device":[],"NWID":[], "Rs":[],"Range":[],"RunID":[],"Light":[],"I_E_steps":[],"V_E_steps":[],"I_R_steps":[],"operation":[]}
        IDF["Auto"][dname] = {"n list":[],"device":[],"NWID":[], "Rs":[],"Range":[],"RunID":[],"Light":[],"I_E_steps":[],"V_E_steps":[],"I_R_steps":[],"operation":[]}
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
                            IDF[tt][dname]["I_E_steps"].append(run["data"]["I_E_steps"][0])
                            IDF[tt][dname]["I_R_steps"].append(run["data"]["I_R_steps"][0])
                            IDF[tt][dname]["V_E_steps"].append(run["data"]["V_E_steps"][0])
                            IDF[tt][dname]["operation"].append(run["data"]["Settings"]["Test Name"].split("#")[0])
                            
                    
                    tt = "All"
                    
                    for runID, run in sdevice[NWID].items():
                            if run["editdict"]["accept"] == True:
                                IDF[tt][dname]["n list"].append(run["editdict"]["n"])
                                IDF[tt][dname]["device"].append(sname)
                                IDF[tt][dname]["NWID"].append(NWID)
                                IDF[tt][dname]["RunID"].append(runID)
                                IDF[tt][dname]["Range"].append(run["editdict"]["mrange"])
                                IDF[tt][dname]["Light"].append(run["editdict"]["light_on"])
                                IDF[tt][dname]["I_E_steps"].append(run["data"]["I_E_steps"][0])
                                IDF[tt][dname]["I_R_steps"].append(run["data"]["I_R_steps"][0])
                                IDF[tt][dname]["V_E_steps"].append(run["data"]["V_E_steps"][0])
                                IDF[tt][dname]["operation"].append(run["data"]["Settings"]["Test Name"].split("#")[0])
                    tt = "Auto"
                    for runID, run in sdevice[NWID].items():
                            if (run["editdict"]["accept"] == True) & (run["editdict"]["mrange"] == "Auto"):
                                IDF[tt][dname]["n list"].append(run["editdict"]["n"])
                                IDF[tt][dname]["device"].append(dname.split("_")[0] + sname)
                                IDF[tt][dname]["NWID"].append(NWID)
                                IDF[tt][dname]["RunID"].append(runID)
                                IDF[tt][dname]["Range"].append(run["editdict"]["mrange"])
                                IDF[tt][dname]["Light"].append(run["editdict"]["light_on"])
                                IDF[tt][dname]["I_E_steps"].append(run["data"]["I_E_steps"][0])
                                IDF[tt][dname]["I_R_steps"].append(run["data"]["I_R_steps"][0])
                                IDF[tt][dname]["V_E_steps"].append(run["data"]["V_E_steps"][0])
                                IDF[tt][dname]["operation"].append(run["data"]["Settings"]["Test Name"].split("#")[0])
                    
                    tt = "Tabulated"
                    IDF[tt][dname][sname][NWID] = {}
                    VE = []
                    IE = []
                    IR = []
                    mrange = [] 
                    runlist    = []
                    light  = []
                    operation = []
                    for runID, run in sdevice[NWID].items():
                        if "ladder" not in run["data"]["Settings"]["Test Name"].split("#")[0]:
                            if run["editdict"]["accept"]:
                                try:        
                                    VEps = np.nanmax(run["data"]["V_E_steps"])
                                    
                                except:
                                    VEps = None
                                try:
                                    IRps = np.nanmax(np.abs(run["data"]["I_R_steps"]-np.nanmean(run["data"]["I_R_steps"][0][0])))
                                    
                                except:
                                    IRps = None
                                try:
                                    IEps = np.nanmax(np.abs(run["data"]["I_E_steps"]-np.nanmean(run["data"]["I_E_steps"][0][0])))
                                    
                                    
                                except:
                                    IEps = None
                                
                                VE.append(VEps)
                                IE.append(IEps)
                                IR.append(IRps)
                                mrange.append(run["editdict"]["mrange"])
                                runlist.append(runID)
                                light.append(run["editdict"]["light_on"])
                                operation.append(run["data"]["Settings"]["Test Name"].split("#")[0])
                        
                    if len(VE) != 0:
                        IDF[tt][dname][sname][NWID]["VE"]      = np.array(VE)
                        IDF[tt][dname][sname][NWID]["IE"]      = np.array(IE)
                        IDF[tt][dname][sname][NWID]["IR"]      = np.array(IR)
                        IDF[tt][dname][sname][NWID]["mrange"] = np.array(mrange)
                        IDF[tt][dname][sname][NWID]["RunID"]  = np.array(runlist)
                        IDF[tt][dname][sname][NWID]["Light"]  = np.array(light)
                        IDF[tt][dname][sname][NWID]["operation"] = np.array(operation)
                        VEList = np.array([v for v in VE if not isinstance(v,(type(None),str))])
                        IEList = np.array([i for i in IE  if not isinstance(i,(type(None),str))])
                        IRList = np.array([i for i in IR if not isinstance(i,(type(None),str))])
                        
                        if len(VEList) >=1:
                            IEabs                                   = np.abs(IE)
                            maxIE                                   = np.max(IEabs)
                            
                            IDF[tt][dname][sname][NWID]["IE_best"]  = maxIE
                            IDF[tt][dname][sname][NWID]["IR_best"]  = np.array(IRList)[np.where(IEabs==maxIE)[0][0]]
                            IDF[tt][dname][sname][NWID]["VE_best"]  = np.array(VEList)[np.where(IEabs==maxIE)[0][0]]
                            IDF[tt][dname][sname][NWID]["range best"]  = np.array(mrange)[np.where(IEabs==maxIE)[0][0]]
                            IDF[tt][dname][sname][NWID]["run best"]  = np.array(runlist)[np.where(IEabs==maxIE)[0][0]]
                            IDF[tt][dname][sname][NWID]["light"]  = np.array(light)[np.where(IEabs==maxIE)[0][0]]
                            IDF[tt][dname][sname][NWID]["operation"]  = np.array(operation)[np.where(IEabs==maxIE)[0][0]]
                    else:
                       IDF[tt][dname][sname][NWID]["IR_best"]      = "None"
                       IDF[tt][dname][sname][NWID]["IE_best"]  = "None"
                       IDF[tt][dname][sname][NWID]["VE_best"]  = "None"
                       
                       IDF[tt][dname][sname][NWID]["range best"]  = "None"
                       IDF[tt][dname][sname][NWID]["run best"]  = "None"
                       IDF[tt][dname][sname][NWID]["light"]  = "None"
                       IDF[tt][dname][sname][NWID]["operation"]  = "Unknown"


Tablestring = "\\begin{table}\n\\begin{tabular}{cccccccccc}\n" 
Tablestring += "Device & Subdevice & NWID & RunID & Operation & $V_E$ steps [V] & $I_R$ steps [fA] & $I_E steps$ [uA]  & Range & Light & R0 [MOhm] \\\\ \n"
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
            
            operation = NWData["operation"]
            try:
                VE_best = "{:.4f}".format(NWData["VE_best"])
                IE_best =  "{:.4f}".format(NWData["IE_best"])
                IR_best =  "{:.4f}".format(NWData["IR_best"]) 
            except:
                VE_best = NWData["VE_best"] 
                IE_best = NWData["IE_best"] 
                IR_best = NWData["IR_best"] 
                
            Tablestring += " & " + NWData["run best"] + " & " + operation + " & " + VE_best + " & " + IR_best + " & " + IE_best + " & " + NWData["range best"] +" & " + str(NWData["light"]) + "\\\\ \n"
            Firstline  = False
            Secondline = False
    Tablestring += "\\\\ \n"

Tablestring += "\\end{tabular} \n \\end{table}"    



with open(os.sep.join([os.getcwd(),"Tabulated","DFR1_Communication_"+ttt+".json"]),"w") as f:
    f.write(Tablestring)        
    

Tablestring = "\\begin{table}\n\\begin{tabular}{cccccccccc}\n" 
Tablestring += "Device & Subdevice & NWID & RunID & Operation & $V_E$ steps [V] & $I_R$ steps [fA] & $I_E steps$ [uA]  & Range & Light & R0 [MOhm] \\\\ \n"
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
        
        Tablestring += " & " + device["operation"][i]
        try:        
            Tablestring += " & " + "{:.4f}".format(np.nanmax(device["V_E_steps"][i]))
        except:
            Tablestring += " & None "
        try:
            Tablestring += " & " + "{:.4f}".format(1e+15*np.nanmax(device["I_R_steps"][i]-np.nanmean(device["I_R_steps"][0])))
        except:
            Tablestring += " & None "
        try:
            Tablestring += " & " + "{:.4f}".format(1e+6*np.nanmax(device["I_E_steps"][i]-np.manmean(device["I_E_steps"][0])))
        except:
            Tablestring += " & None "
            
        Tablestring += " & " + device["Range"][i]
        
        Tablestring += " & " + str(device["Light"][i])

        Tablestring += "\\\\ \n"

Tablestring += "\\end{tabular} \n \\end{table}"    



with open(os.sep.join([os.getcwd(),"Tabulated","DFR1-Communication_Everything_"+ttt+".json"]),"w") as f:
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
            DD[dname][mat] = {"VE":[],"IE":[],"IR":[]}
            
        for NWID,NWData in subdevice.items():      
            
            if NWData["IR_best"] != "None":                
                DD[dname][mat]["VE"].append(NWData["VE_best"])
                DD[dname][mat]["IE"].append(NWData["IE_best"])
                DD[dname][mat]["IR"].append(NWData["IR_best"])

#%%
"""
Each device occupies n->n+1, each new material has a 1 gap, each new device a 2 gap
"""
#Communication Colours
cols = {"IE":[dmp.get_rgbhex_color("dark blue",ctype="rgba"),dmp.get_rgbhex_color("blue",ctype="rgba")],
        "VE":[dmp.get_rgbhex_color("dark orange",ctype="rgba"),dmp.get_rgbhex_color("orange",ctype="rgba")],
        "IR":[dmp.get_rgbhex_color("dark green",ctype="rgba"),dmp.get_rgbhex_color("green",ctype="rgba")]}


axs = [] #new ax handles
ph  = [] #Plot handles
FIG = dm.ezplot()
ax = FIG.ax[0]
 
axs.append(ax)
axs.append(ax.twinx())
axs.append(ax.twinx())
axs[0].patch.set_alpha(0)  # Set the background behind everything
ax.set_zorder(2)
axs[2].patch.set_zorder(0)
axs[1].set_zorder(1)

plotkwargs = {}

cmap = dmp.get_tab20bc(output="cmap",grouping="all")

gss.graph_style("PP1_Wide") # Comment out to customise your own style
        
DefBBOX = gss.DEF_BBOX(style="PP1_Wide",bboxstyle="symmetric")

cn = -1

pcount = -2 
tickname = []
tickposition = []
tickdevice = []
device_position = [-2]

firstrun = True
DD.pop("DFR1-EE")
DD["DFR1-IG"].pop("ALD")
def sorted_from_middle(lst, reverse=False):
    if len(lst) <= 1:
        return lst
    tail = sorted([lst[-1], lst[0]], reverse=reverse)
    return sorted_from_middle(lst[1:-1], reverse) + tail
Chrono = ["EG","IG","FF","IG","GG","FJ","HE","GK","JI"]
DKEY = ["DFR1-"+ch for ch in Chrono]
for device in DKEY:
    pcount+=1
    cn+=1
    mn = 0
    for mname,mat in DD[device].items():
        
        inds = np.where(np.array(mat["IR"])<1e-12)[0]
        ub = .95
        lb = .05
        mod=-.5
        if "PMMA" in DD[device].keys():
            if mn == 0:
                mnmod = -0.2
            if mn == 1:
                mnmod = 0.2
        else:
            mnmod = 0

        dx = np.array(sorted_from_middle(list(np.linspace(lb,ub,len(inds)+2))))[:len(inds)]+pcount+mod +mnmod
        
        yVE = np.array(mat["VE"])[inds]
        yIE = np.array(mat["IE"])[inds]
        yIR = np.array(mat["IR"])[inds]
        if len(yVE) != 0:
            if cn == 10:
                cn=0
            #ax.plot(dx,y,"x",c=cmap(cn*4+mn))
            if firstrun:
                ph.append(axs[1].semilogy(dx,yIR,"*",label='$I_{Receiver}$ [A]',color=cols["IR"][0],markersize=8,markeredgewidth=2,zorder=55)[0] ) 
                ph.append(axs[2].plot(dx,yIE,"x",label='$I_{Emitter}$ [A]',color=cols["IE"][0],markersize=8,markeredgewidth=2,zorder=44)[0] ) 
                ph.append(axs[0].plot(dx,yVE,'.',label='$V_{Emitter}$ [V]',color=cols["VE"][0],markersize=8,markeredgewidth=2,zorder=66)[0])
                firstrun = False
            else:
                axs[1].semilogy(dx,yIR,"*",color=cols["IR"][0],markersize=8,markeredgewidth=2,zorder=55) 
                axs[2].plot(dx,yIE,"x",color=cols["IE"][0],markersize=8,markeredgewidth=2,zorder=44) 
                axs[0].plot(dx,yVE,'.',color=cols["VE"][0],markersize=8,markeredgewidth=2,zorder=66)
            dxx  = [[dxx,dxx] for dxx in dx]
            yIRR = [[0,yy] for yy in yIR] 
            yIEE = [[0,yy] for yy in yIE] 
            yVEE = [[0,yy] for yy in yVE] 
            for i,xxx in enumerate(dxx):
                axs[1].semilogy(xxx,yIRR[i],"-",color=cols["IR"][1],linewidth=2,zorder=0,alpha=.3) 
                axs[2].plot(xxx,yIEE[i],"-",color=cols["IE"][1],linewidth=1.5,zorder=0,alpha=.3) 
                axs[0].plot(xxx,yVEE[i],'--',color=cols["VE"][1],linewidth=1,zorder=0,alpha=.3)
            tickname.append(mname)
            tickposition.append(pcount+mnmod+0.05)
            mn += 1
            pcount+=1
    if len(yVE) !=0:
        tickdevice.append(device.replace("DFR1-",""))
        device_position.append(pcount)
    

ax.set_xticks(tickposition)  # Set tick positions
ax.set_xticklabels(tickname,rotation=90,ha="center")  # Set tick labels
ax.tick_params(axis="x", which="both", length=0)  # No major or minor tick marks

for i in range(len(device_position)-1):
    ax.annotate(tickdevice[i],xy=(np.mean([device_position[i+1],device_position[i]]),6.0),annotation_clip=False,ha="center")
ax.set_xlim([-2,np.max(device_position)])
# Highlight each device column with alternating colors
highlight_colors = ["white", "lightgrey"]
for i in range(len(device_position) - 1):
    start = device_position[i]
    end = device_position[i + 1]
    color = highlight_colors[i % len(highlight_colors)]
    ax.axvspan(start, end, color=color, alpha=0.3)  # Adjust alpha for transparency
    
bbox = [DefBBOX[0]-0.07,DefBBOX[1]-0.05,DefBBOX[2]+0.05,DefBBOX[3] + 0.025]
FIG.apply_bbox(0,bbox)



axs[1].set_ylabel("$I_{Receiver}$ [A]" ,color=cols["IR"][0])
axs[2].set_ylabel('$I_{Emitter}$ [A]'  , color=cols["IE"][0])
axs[0].set_ylabel('$V_{Emitter}$ [V]',color=cols["VE"][0])

axs[0].set_ylim([1.8,6.5])
axs[1].set_ylim([1e-16,1e-11])
axs[2].set_ylim([0,1.2e-6])


# # Get the right spine of ax[2] 
# bboxes = {}
# for ax in axs:
#     bbox = ax.get_position()
    
#     bbox.x0 = kw.bounds_padding[0]; bbox.x1 = kw.bounds_padding[1]; 
#     bbox.y0 = kw.bounds_padding[2]; bbox.y1 = kw.bounds_padding[3]; 
#     ax.set_position(bbox)
#     bboxes[ax] = bbox




#dmp.align_axis_zeros([axs[1],axs[2],axs[0]])

# for nax in axs:
dmp.adjust_ticks(axs[2],which="yticks",Nx=2,Ny=2,xpad=1,ypad=1,respect_zero =True, whole_numbers_only = True)       #adjust ticks based on original ticks

   
ticklabelwidth = dmp.dummy_text_params("−0.00",FIG,fontsize=plt.rcParams["ytick.labelsize"],usetex=plt.rcParams["text.usetex"])["width"] # Get the width of the bounding box in figure coordinates
#We will get the width of a single "-" in figure coordinates too.
minuslabelwidth = dmp.dummy_text_params("−",FIG,fontsize=plt.rcParams["ytick.labelsize"],usetex=plt.rcParams["text.usetex"])["width"] # Get the width of the bounding box in figure coordinates

spine_move = FIG.fig.dpi*4.01*ticklabelwidth
spine_move_fig = axs[2].transAxes.inverted().transform((spine_move, 0))

axs[2].spines.right.set_position(("outward",spine_move*1.2))


#Setting Spine colours and tickparameters
axs[1].yaxis.label.set_color(cols["IR"][0])
axs[2].yaxis.label.set_color(cols["IE"][0])
axs[0].yaxis.label.set_color(cols["VE"][0])


axs[0].tick_params(which="both", axis='y', colors=cols["VE"][0])
axs[1].tick_params(which="both", axis='y', colors=cols["IR"][0])
axs[2].tick_params(which="both",axis='y', colors=cols["IE"][0])

axs[0].spines["left"].set_color(cols["VE"][0])
axs[1].spines["right"].set_color(cols["IR"][0])
axs[2].spines["right"].set_color(cols["IE"][0])

axs[1].tick_params(axis='x')
legend = axs[1].legend(ncol=3,handles=[ph[0], ph[1], ph[2]],loc="upper center",frameon=False,columnspacing=0.8,handlelength=1.5) 
# Get the font size for the legend text
   
legendheight = dmp.dummy_text_params("DUMMY",FIG,fontsize=plt.rcParams["legend.fontsize"])["height"] # Get the width of the bounding box in figure coordinates
for t in legend.get_texts(): t.set_va('bottom')
legend.set_bbox_to_anchor([sum(x) for x in zip((0, 0.6*legendheight, 1, 1),(0,0,0,0))])


t1 = axs[1].yaxis.get_offset_text().get_position()
t2 = axs[2].yaxis.get_offset_text().get_position()

axs[1].yaxis.get_offset_text().set_position((t1[0] ,t1[1] ))
axs[2].yaxis.get_offset_text().set_position((t2[0]+0.22 ,t2[1]+0.07 ))
# axs[1].yaxis.get_offset_text().set_position((t1[0] +0.125 ,t1[1] + 0.125))
# axs[2].yaxis.get_offset_text().set_position((t2[0] + -spine_move_fig[0]+0.04,t2[1] + 0.125))

for ax in axs:
    ax.spines["left"].set_color(cols["VE"][0])

 

for ax in axs:
    if ax != 0:
        move_ax = False
        
        y_min, y_max = ax.get_ylim()
        
        # Get the y-tick positions
        yticks = ax.get_yticks()
        
        # Filter tick labels based on whether they are within the y-axis limits
        in_bounds_labels = [tick for tick in yticks if y_min <= tick <= y_max]
        
        if np.min(in_bounds_labels)<0:
            move_ax = True
                    
        if move_ax:
            for j, tickobj in enumerate(ax.get_yticklabels()):
                                                    
                ticktext   = tickobj.get_text()
                if ticktext == "":
                    ticktext = "-1"
                    
                if float(ticktext.replace("$\\mathdefault{","").replace("}$","").replace("−","-"))>=0:
                    tick_position = tickobj.get_position()

                    # Create a new position with the x-axis translation
                    new_position = (tick_position[0] + 0.8*minuslabelwidth, tick_position[1])
     
                    # Set the new position for the tick label
                    tickobj.set_position(new_position)
                    #ticktrans = mpl.transforms.Affine2D().translate(minuslabelwidth*FIG.fig.dpi,0) 
                    #tickobj.set_transform(tickobj.get_transform() + ticktrans)
                    
FIG.fig.savefig("CommunicationSpread.pdf")                       
