import os
import json 
import numpy as np
from functools import partial
import sys
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets as qtw
from PyQt5.QtGui import QColor,QPalette  # Make sure this is imported
import matplotlib.pyplot as plt
import matplotlib as mpl
import xlrd
from DMU import utils as dm
from DMU import utils_utils as dmm
import scipy 
from scipy.optimize import curve_fit, root, least_squares
from scipy.interpolate import interp1d
def get_tab20bc(output="list",grouping="pairs"):
    t20b = plt.get_cmap("tab20b")
    t20c = plt.get_cmap("tab20c")

    
    cmap20bc = []
    if grouping == "pairs":
        for i in [0,1,2,3]:
            for j in [0,2]:
                 cmap20bc.append(t20c(4*i + j))
    
        for i in [4,3,0,1,2]:
            for j in [0,2]:
                 cmap20bc.append(t20b(4*i + j))
                 
        for i in [4]:
            for j in [0,2]:
                 cmap20bc.append(t20c(4*i + j))
    elif grouping == "all":
        for i in [0,1,2,3,4]:
            for j in [0,1,2,3]:
                 cmap20bc.append(t20b(4*i + j))
                 
        for i in [0,1,2,3,4]:
            for j in [0,1,2,3]:
                 cmap20bc.append(t20c(4*i + j))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("tab20bc", cmap20bc,N=len(cmap20bc))
    
    if output == "list":    
        return(cmap20bc)
    if output == "cmap":
        return(cmap)
    if output == "both":
        return(cmap20bc,cmap)
    
def get_rgbhex_color(color_name,ctype = "hex"):
    
    cmap = get_tab20bc(output="cmap",grouping="all")

    # Select a color from the colormap (normalized index between 0 and 1)
    colornames = ["dark violet", "violet", "light violet", "pale violet",
                  "dark lime", "lime", "light lime", "pale lime",
                  "dark tan", "tan", "light tan", "pale tan",
                  "dark red", "red", "light red", "pale red",
                  "dark lilac", "lilac", "light lilac", "pale lilac",
                  "dark blue", "blue", "light blue", "pale blue",
                  "dark orange", "orange", "light orange", "pale orange",
                  "dark green", "green", "light green", "pale green",
                  "dark lilac", "lilac", "light lilac", "pale lilac",
                  "dark grey", "grey", "light grey", "pale grey"]
    colordict = {colorname:colorID for colorID,colorname in enumerate(colornames)}
    if color_name not in colornames:
        raise ValueError(f"Invalid colour name: '{color_name}'\nValid names are: {', '.join(colornames)}")
        
    rgb = cmap(colordict[color_name]) 
    if ctype == "rgba":
        return(rgb)
   
    if ctype == "rgb":
        return(rgb[:3])
    
    if ctype == "rgb255":
        return(tuple([int(val*255) for val in rgb[:3]]))
    
    if ctype == "rgba255":
        return(tuple([int(val*255) for val in rgb]))    
        
    
    elif ctype == "hex":
        # Include alpha channel (opacity) if needed
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

        # If alpha exists, we handle it separately (adding a transparency value if RGBA)
        if len(rgb) > 3:
            # Normalize alpha (it should be between 0 and 1)
            alpha = int(rgb[3] * 255)
            hex_color += '{:02x}'.format(alpha)  # Add alpha value to the hex code

        return hex_color

def set_nested_dict_value(self, selfvar, keylist, value):
    """
    Sets a value in a nested dictionary within a specified attribute of self.
    Automatically creates missing intermediate dictionaries.

    Args:
        selfvar (str): Name of the attribute in self to modify (e.g., 'session').
        keylist (list): List of keys specifying the nested path.
        value (any): The value to set at the specified path.

    Raises:
        TypeError: If a non-dictionary is encountered before the final key.
    """
    # Get the target attribute in self
    target_dict = getattr(self, selfvar, None)
    if target_dict is None:
        raise AttributeError(f"'{selfvar}' not found in 'self'.")
        
    if keylist == None:
        setattr(self, selfvar, value)
        return
    

    # Traverse and set value in the nested dictionary
    d = target_dict
    for i, key in enumerate(keylist):
        if i == len(keylist) - 1:
            # Set value at the last key
            d[key] = value
        else:
            # Create missing intermediate dictionaries if needed
            if key not in d:
                d[key] = {}  # Initialize empty dict
            elif not isinstance(d[key], dict):
                raise TypeError(f"Key '{key}' exists but is not a dictionary.")
            # Move deeper into the dictionary
            d = d[key]
    
    # Update the attribute in self (not needed if target_dict is modified in place)
    setattr(self, selfvar, target_dict)
            
def get_nested_dict_value(self, selfvar, keylist, default_text="NO KEY"):
    
    target_dict = getattr(self, selfvar, None)
    
    if keylist == None:
        try:
            return(target_dict)
        except:
            return "No Variable name"
    if target_dict is None:
        return default_text
    try:
        for key in keylist:
            target_dict = target_dict[key]
        return target_dict
    except (KeyError, TypeError):
        
        return default_text
    
    
### UTILITY  and SAVING ###
def json_savedata(data, filename, directory=os.getcwd(), overwrite=False):
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if filename in files and not overwrite:
        print("RENAME YOUR FILE OR RUN DELETE IN NEXT CELL")
    else:
        pathname = os.sep.join([directory, filename]) 
        with open(pathname, "w") as f:
            json.dump(convert_to_json_serializable(data), f, indent=2)
            
def json_loaddata(filepath,importtype = "normal"):
    if importtype == "normal":
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
        return(existing_data)
    
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
            
def check_and_mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def merge_dicts(a, b):
    """
    Recursively merge missing keys and values from dict `a` into dict `b`.
    Existing keys in `b` are never overwritten.

    Args:
        a (dict): Source dictionary with new data.
        b (dict): Target dictionary to be updated.
    Returns:
        dict: Updated `b` with missing keys and values from `a`.
    """
    for key, value in a.items():
        # If the value is a nested dict, recurse into both dictionaries
        if isinstance(value, dict):
            # If `b` doesn't have the key or it isn't a dict, initialize it as a dict
            if key not in b.keys() or not isinstance(b[key], dict):
                b[key] = {}
            # Recursive call to merge nested dictionaries
            merge_dicts(value, b[key])
        # If the key doesn't exist in `b`, add it from `a`
        elif key not in b.keys():
            b[key] = value
    return b

def create_devicedata_dict(self, KeithleyData, runID):
    """
    Function should return a dict in the form:
        subdevicedict = dict(NWID = dict(runID = dict(fitdict,editdict,data))) 
    """
    
    if self.USERMODE == "Ideality":
        kd = KeithleyData
        fitdict  = self.session_reset["fitdict"].copy() 
        editdict = dict(n = None, n_series = None, Rs_lin=None, Rs=None, accept=None, light_on=bool(kd["LOG"]["Light Microscope"]),mrange=" ".join([item for item in kd["Settings"]["Current Range"] if str(item) !=str(None)]), NWID=kd["emitter"]["NWID"], ideality_fit = dict(n=None,I0=None,Vfit=None,Ifit=None),series_fit = dict(n=None,Rs_lin=None,Rs=None,I0=None,Vfit=None,Ifit=None,Rs_fit = None,n_fit = None))
        return {runID: dict(fitdict=fitdict, editdict=editdict, data=kd)}


def save_and_update_USERMODE_jsondicts(self,selfvar="session",datavar="DATA",listwidgets = None,keylists=""):
    device_widget,subdevice_widget,run_table = listwidgets
    usermode           = self.USERMODE_Types[self.get_nested_dict_value(selfvar,["List_Indices","USERMODEID"])]
    save_directory     = self.get_nested_dict_value(selfvar,["Directories","Data_Storage_Current"])
    root_data_search   = self.get_nested_dict_value(selfvar,["Directories","Data_Search"])
    old_data = self.DATA.copy()
    
    DeviceFolders = dm.find_folders_containing_filetype(root_data_search,must_include="",filetype=".xls",skip_if_name=None,return_progress=False)
    FileExclusions = []
    nolog = []
    devicedict = {}

    for file in DeviceFolders:   
        if FileExclusions:
            if any(exl in file for exl in FileExclusions):
                continue
        data = dm.Keithley_xls_read(file)
        
        if usermode == "Ideality":
            for dkey in data.keys():
                if "LOG" not in dkey:
                    for key in [k for k in data[dkey].keys() if "Run" in k]: 
                        d = data[dkey][key] 
                        try:
                            d["LOG"]
                        except:
                            nolog.append(file.split("\\")[-1] + key + "Has no log entry!")
                            break
                        
                        if len(d["Settings"]["Npts"]) == 2:
                            if len(d["current"]) <5:
                                material = "AIR"
                                if any(mat in file for mat in self.MATERIAL_Types):
                                    material = next((mat for mat in self.MATERIAL_Types if mat in file), "AIR")    
                                
                                fulldevice = d["LOG"]["Device"].replace(" ","-").replace("_","-")
                                device     = "-".join(fulldevice.split("-")[:2]) 
                                filtered_device = [part for part in fulldevice.split("-") if part not in self.MATERIAL_Types] #We are just double checking that we don't have material duplicates
                                subdevice  = "-".join(filtered_device[1:]) + "_" + material
                                
                                if device not in devicedict.keys():
                                    devicedict[device] = {}
                                    
                                if subdevice not in devicedict[device]:
                                    devicedict[device][subdevice] = {}
                                    devicedict[device][subdevice]["Sample"]     = device.split("-")[0]
                                    devicedict[device][subdevice]["Device"]     = device
                                    devicedict[device][subdevice]["Waveguide"]  = material
                                    devicedict[device][subdevice]["filename"]   = device+".json"    
                                nwid = d["emitter"]["NWID"]
                                if nwid not in devicedict[device][subdevice]:
                                    devicedict[device][subdevice][nwid] = {}
                                
                                # Add new runs to existing NWID
                                jsondict = self.create_devicedata_dict(d,key)
                                devicedict[device][subdevice][nwid].update(jsondict)

    # Sort keys at the upper level alphabetically
    devicedict = {k: devicedict[k] for k in sorted(devicedict)}
    
    # Load existing JSON data and merge
    for dname, ddict in devicedict.items():
        filepath = os.path.join(save_directory,dname+".json")
        if os.path.exists(filepath):
            existing_data = json_loaddata(filepath) 
            merge_dicts(ddict, existing_data)  # Merge new data without overwriting
            json_savedata(existing_data,directory=save_directory,filename=dname+".json",overwrite=True)
            print("Merged new dict into: " + dname + ".json")
        else:
            json_savedata(ddict,directory=save_directory,filename=dname+".json",overwrite=True) 
            print("Saved new dict into: " + dname + ".json")
    
    self.read_data_storage()
    self.populate_device_list(selfvar=selfvar,datavar = datavar,device_widget = listwidgets[0])

    
### GUI CREATION AND UPDATES ###
def update_rundata_variable(self, selfvar="session", datavar="DATA", alter_type="Overwrite", partial_keylist=[], keylists = [["List_Indices", "DeviceID"],["List_Indices", "SubvdeviceID"],["List_Indices", "RunID"]], listwidgets=None, toggle_options=None, new_value=None):
    
    """
    Update the specified nested variable in the data dictionary.

    Parameters:
    - selfvar: Reference to self.
    - datavar: The main data variable to be updated.
    - alter_type: Type of alteration ('Toggle' or 'Overwrite').
    - keylist: Partial key path specifying the nested variable to update.
    - listwidgets: Widgets to determine the current selection keys.
    - toggle_options: Options for toggling values.
    - new_value: Value to overwrite in case of 'Overwrite'.
    """

    # Get device, subdevice, nw, and run keys from the current selections
    device_widget, subdevice_widget, run_table = listwidgets
    leftkeys,rightkeys,runkeys = keylists
    try:
        devicekey = device_widget.item(device_widget.currentRow()).text()
        subdevicekey = subdevice_widget.item(subdevice_widget.currentRow()).text()
        nwkey = run_table.item(run_table.currentRow(), 0).text()
        runkey = run_table.item(run_table.currentRow(), 1).text()
    except:
        deviceID    = self.get_nested_dict_value(selfvar, leftkeys) 
        subdeviceID = self.get_nested_dict_value(selfvar, rightkeys)
        runID       = self.get_nested_dict_value(selfvar, runkeys)
        devicekey = device_widget.item(deviceID).text()
        subdevicekey = subdevice_widget.item(subdeviceID).text()
        nwkey = run_table.item(runID, 0).text()
        runkey = run_table.item(runID, 1).text()
        
    # Construct the complete keylist
    keylist = [devicekey, subdevicekey, nwkey, runkey] + partial_keylist

    # Retrieve the current value using the constructed keylist
    current_value = self.get_nested_dict_value(datavar, keylist)

    if alter_type == "Toggle":
        # Determine the new value based on the toggle options
        if current_value in toggle_options:
            current_index = toggle_options.index(current_value)
            new_value = toggle_options[(current_index + 1) % len(toggle_options)]
        else:
            new_value = toggle_options[0]  # Default to first option if current value is invalid
        
        if "accept" in partial_keylist:
            if isinstance(self.get_nested_dict_value(datavar,[devicekey, subdevicekey, nwkey, runkey] + ["editdict","n"]),(type(None),str)):
                new_value = False 
                print("IDEALITY NOT FITTED, CANNOT SET TRUE")
        
        print("Toggled " + " - ".join((keylist)) +  " to: " +str(new_value)  )
    elif alter_type == "Overwrite" and new_value is not None:
        None  # new_value is already specified
        
    
    # Update the nested data using the complete keylist
   
    self.set_nested_dict_value(datavar, keylist, new_value)
   
    self.update_all_on_run_change_and_colors(selfvar="session", listwidgets=listwidgets, keylists=keylists)
    

    
from PyQt5 import QtWidgets as qtw

def create_multi_choice_button(self, selfvar="session", label_text="", options=[""], start_index=0, keylist=None, layout=""):
    # Ensure inputs are in the correct format
    if not isinstance(options, list):
        options = list(options)
    if not isinstance(keylist, list):
        keylist = list(keylist)
        
    # Get the current index value dynamically using getattr and set the initial value if necessary
    current_index = self.get_nested_dict_value(selfvar, keylist)  

    # Set the initial value in the session if needed
    self.set_nested_dict_value(selfvar, keylist, current_index)

    # Create a horizontal layout for the button
    container_layout = qtw.QHBoxLayout()

    # Create and set the label
    label = qtw.QLabel(label_text)
    container_layout.addWidget(label, stretch=1)

    # Create the dropdown (QComboBox)
    dropdown = qtw.QComboBox()
    dropdown.addItems(options)

    # Set the initial index based on the dynamically accessed value
    dropdown.setCurrentIndex(current_index if 0 <= current_index < len(options) else start_index)

    # Connect the dropdown selection change to update session and index_variable
    def on_index_changed(index):
        # Use setattr to dynamically set the value of the variable specified by indexvar
        self.set_nested_dict_value(selfvar, keylist, index)  # Update the variable dynamically
    

    dropdown.currentIndexChanged.connect(on_index_changed)

    # Add the dropdown to the layout
    container_layout.addWidget(dropdown, stretch=5)

    # Add the layout to the parent layout
    layout.addLayout(container_layout)

    return dropdown  # Optionally return dropdown if needed later

                 

def save_highest_level_json(self,selfvar="session",keylists="",listwidgets=""):
    try:
        device_widget,subdevice_widget,run_table = listwidgets
        leftkeys,rightkeys,runkeys = keylists
        device_id          = self.get_nested_dict_value(selfvar, leftkeys)
        subdevice_id       = self.get_nested_dict_value(selfvar, rightkeys) 
        selected_device    = device_widget.item(device_id).text()
        selected_subdevice = subdevice_widget.item(subdevice_id).text()
        filename           = self.DATA[selected_device][selected_subdevice]["filename"]
        save_directory     = self.get_nested_dict_value(selfvar,["Directories","Data_Storage_Current"])
        json_savedata(self.DATA[selected_device],directory=save_directory,filename=filename,overwrite=True) 
        print("saved " + filename)
    except:
        print("Saving JSON skipped since you just loaded your data")
    
            
def update_device_list(self,selfvar="session",listwidgets=None,keylists=None):
    """
    Top tier hierachy update of device list, involves clearing the subdevice and run indices so they can be populated in correct order
    """
    
    
    device_widget, subdevice_widget, run_table = listwidgets
    leftkeys,rightkeys,runkeys = keylists
    
    device_id       = self.get_nested_dict_value(selfvar,leftkeys)
    
    if device_widget.currentRow() <0 :
        device_widget.blockSignals(True)
        device_widget.setCurrentRow(device_id)
        self.set_nested_dict_value(selfvar,leftkeys,device_id)
        device_widget.blockSignals(False)
        
    run_table.blockSignals(True)
    subdevice_widget.blockSignals(True)
    subdevice_widget.clearSelection()    
    run_table.clearSelection()     
    
    #We populate the new subdevice list 
    self.populate_subdevice_list(listwidgets=listwidgets, keylists=keylists)

def populate_subdevice_list(self, selfvar="session", listwidgets=None,keylists=None):
    
    device_widget, subdevice_widget, run_table = listwidgets
    leftkeys,rightkeys,runkeys = keylists
    
    device_id       = self.get_nested_dict_value(selfvar,leftkeys)
    selected_device = device_widget.item(device_id).text()
    
    subdevice_id     = self.get_nested_dict_value(selfvar, rightkeys) 
    
    subdevice_widget.clear()
    if selected_device in self.DATA.keys():
        subdevice_data  = self.DATA[selected_device]
        
        for subdevice_name, subdevice_info in subdevice_data.items():
            item = qtw.QListWidgetItem(subdevice_name)
            subdevice_widget.addItem(item)
    
    #Here, we activate the "index change behaviour of the subdevice widget - this means we can run the index check, and update_device_list again
    
    subdevice_widget.blockSignals(False)
    subdevice_widget.setCurrentRow(subdevice_id) #Remember that we already set this to zero before. It should not possibly be any different now.

def update_subdevice_list(self,selfvar="session",listwidgets=None,keylists=None):
    """
    Top tier hierachy update of device list, involves clearing the run table it can be populated
    """
    device_widget, subdevice_widget, run_table = listwidgets
    leftkeys,rightkeys,runkeys = keylists
    
    run_table.blockSignals(True)
    run_table.clearSelection()  # Clear selection so that it doesn't trigger the change index function     
    #We populate the new run table
    self.populate_run_table(selfvar=selfvar, listwidgets=listwidgets, keylists=keylists)
            
def populate_run_table(self, selfvar="session", listwidgets=None,keylists=None):
    device_widget, subdevice_widget, run_table = listwidgets
    leftkeys,rightkeys,runkeys = keylists
    
    device_id        = self.get_nested_dict_value(selfvar, leftkeys)
    subdevice_id     = self.get_nested_dict_value(selfvar, rightkeys) 
    run_id           = self.get_nested_dict_value(selfvar, runkeys) 
    
    
    
    selected_device    = device_widget.item(device_id).text() 
    selected_subdevice = subdevice_widget.item(subdevice_id).text()
    
    device_data    = self.DATA.copy()
    

    run_table.clearContents()
    run_table.setRowCount(0)

    if selected_device and selected_subdevice:
        subdevice_data = device_data[selected_device][selected_subdevice]
        NWkeys = [key for key in subdevice_data.keys() if key not in ["Sample","Device","Waveguide","filename"]]
        for nw_key in NWkeys:
            nw_data = subdevice_data[nw_key]
            
            for runID, run_info in nw_data.items():
                
                run_edit   = run_info["editdict"]
                nwid       = run_edit.get("NWID", "")
                ideality   = run_edit.get("n", "N/A")
                if type(ideality) !=str: ideality = f"{ideality:.5f}"
                ideality_s = run_edit.get("n_series", "N/A")
                if type(ideality_s) !=str: ideality_s = f"{ideality_s:.5f}"
                Rs_lin     = run_edit.get("Rs_lin", "N/A")
                if type(Rs_lin) !=str: Rs_lin = f"{Rs_lin*1e-6:.5f}"
                Rs_series  = run_edit.get("Rs", "N/A")
                if type(Rs_series) !=str: Rs_series = f"{Rs_series*1e-6:.5f}"
                accept     = run_edit.get("accept",None)
                lighton    = run_edit.get("light_on",False) 
                mrange     = run_edit.get("mrange","Unknown")
                if run_edit.get("accept") == True:
                    status = "Accepted"
                elif run_edit.get("accept") == False:
                    status = "Rejected"
                else:
                    status = "Not Done"
                
                row = run_table.rowCount()
                run_table.insertRow(row)
                run_table.setItem(row, 0, qtw.QTableWidgetItem(f"{nwid}"))
                run_table.setItem(row, 1, qtw.QTableWidgetItem(f"{runID}"))
                run_table.setItem(row, 2, qtw.QTableWidgetItem(ideality))
                run_table.setItem(row, 3, qtw.QTableWidgetItem(ideality_s))
                run_table.setItem(row, 4, qtw.QTableWidgetItem(Rs_lin))
                run_table.setItem(row, 5, qtw.QTableWidgetItem(Rs_series))
                run_table.setItem(row, 6, qtw.QTableWidgetItem(f"{lighton}"))
                run_table.setItem(row, 7, qtw.QTableWidgetItem(f"{mrange}"))
                run_table.setItem(row, 8, qtw.QTableWidgetItem(f"{status}"))
    run_table.blockSignals(False)    
    run_table.setCurrentCell(run_id,0) #If we just reset, this run_id will be 0 anyway, but this will trigger the run_table index update function anwyay.

def update_all_on_run_change_and_colors(self, selfvar="session", listwidgets=None, keylists=None):
    """
    Updates run table contents, colors for devices, subdevices, and rows, as well as highlights.
    """

    # Unpack widgets and keys
    device_widget, subdevice_widget, run_table = listwidgets
    leftkeys, rightkeys, runkeys = keylists

    # Get selected IDs
    device_id     = self.get_nested_dict_value(selfvar, leftkeys)
    subdevice_id  = self.get_nested_dict_value(selfvar, rightkeys)
    run_id        = self.get_nested_dict_value(selfvar, runkeys)
    
    # Get selected device and subdevice
    selected_device = device_widget.item(device_id).text()
    selected_subdevice = subdevice_widget.item(subdevice_id).text()
    selected_nw = run_table.item(run_id, 0).text()
    selected_run = run_table.item(run_id, 1).text()
    
    # Fetch hierarchical data
    device_data = self.DATA[selected_device].copy()
    statusdict = {}
    # === Update Device Colors ===
    for index in range(device_widget.count()):
        item = device_widget.item(index)
        device_name = item.text()
        statusdict[device_name] = {}
        subdevices = self.DATA[device_name].copy()
        
        # Aggregate subdevice accept values
        for subdevice_name,subdevice_info in subdevices.items():
            statusdict[device_name][subdevice_name] = {"color":[]}
            
            NWkeys = [key for key in subdevice_info.keys() if key not in ["Sample", "Device", "Waveguide", "filename"]]
            for nw_key in NWkeys:
                nw_data = subdevice_info[nw_key]
                statusdict[device_name][subdevice_name][nw_key] = {"accept_lighton":[]}
                
                for runkey,run_info in nw_data.items():
                    if runkey not in statusdict[device_name][subdevice_name][nw_key].keys():
                        statusdict[device_name][subdevice_name][nw_key][runkey] = {"accept_lighton":[]}
                    statusdict[device_name][subdevice_name][nw_key]["accept_lighton"].append([run_info["editdict"]["accept"],run_info["editdict"]["light_on"]])                    
                    statusdict[device_name][subdevice_name][nw_key][runkey]["accept_lighton"].append([run_info["editdict"]["accept"],run_info["editdict"]["light_on"]])

            
            
            values_A = statusdict[device_name][subdevice_name][NWkeys[0]]["accept_lighton"]
            try:
                values_B = statusdict[device_name][subdevice_name][NWkeys[1]]["accept_lighton"]
            except:
                values_B = [[False,False]]
                
            # Evaluate pairwise conditions across both keys
            has_true_false = any(a == [True, False] and b == [True, False] for a, b in zip(values_A, values_B))
            has_one_true_false = (any(a == [True, False] for a in values_A) or any(b == [True, False] for b in values_B)) and not has_true_false
            has_true_true = all(a == [True, True] and b == [True, True] for a, b in zip(values_A, values_B))
            has_false_true = all(a == [False, True] and b == [False, True] for a, b in zip(values_A, values_B))
            has_false_false = all(a == [False, False] and b == [False, False] for a, b in zip(values_A, values_B))
            has_none = any(a[0] is None or b[0] is None for a, b in zip(values_A, values_B))
            
            # Update the color based on conditions
            if has_none:
                color = 'pale tan'  # At least one entry is [None, anything]
            elif has_true_false:
                color = 'pale green'  # Both NW keys have [True, False]
            elif has_one_true_false:
                color = 'pale lime'   # Only one NW key has [True, False]
            elif has_true_true:
                color = 'pale blue'  # Both NW keys have [True, True]
            elif has_false_true:
                color = 'pale violet'  # Both NW keys have [False, True]
            elif has_false_false:
                color = 'pale red'  # Both NW keys have [False, True]
            else:
                color = 'pale tan'   # Default case, all [False, False]
            
            # Assign the color to the status dictionary
            statusdict[device_name][subdevice_name]["color"] = color
            

        if all(subdevice["color"] in ["pale green", "pale blue","pale lime"] for subdevice in statusdict[device_name].values()):
            color = "pale green"
            
        elif all(subdevice["color"] in ["pale green", "pale blue","pale lime","pale red"] for subdevice in statusdict[device_name].values()):
             color = "pale lime"
            
        elif all(subdevice["color"] == "pale blue" for subdevice in statusdict[device_name].values()):
            color = "pale blue"

        elif all(subdevice["color"] == "pale violet" for subdevice in statusdict[device_name].values()):
            color = "pale violet"
            
        elif all(subdevice["color"] == "pale red" for subdevice in statusdict[device_name].values()):
            color = "pale red"
            
        elif any(subdevice["color"] == "pale tan" for subdevice in statusdict[device_name].values()) and any(subdevice["color"] in ["pale green","pale lime","pale blue","pale violet"] for subdevice in statusdict[device_name].values()):
            color = "light orange"
        
        else:
            color = "pale tan"
        
        item.setBackground(QColor(*get_rgbhex_color(color, ctype="rgba255")))

    # === Update Subdevice Colors ===
    for index in range(subdevice_widget.count()):
        item = subdevice_widget.item(index)
        subdevice_name = item.text()
        color = statusdict[selected_device][subdevice_name]["color"]
        item.setBackground(QColor(*get_rgbhex_color(color, ctype="rgba255")))

    # === Update Run Table ===
    if selected_device and selected_subdevice:
        subdevice_data = device_data[selected_subdevice]
        NWkeys = [key for key in subdevice_data.keys() if key not in ["Sample", "Device", "Waveguide", "filename"]]
        row = 0
        for nw_key in NWkeys:
            nw_data = subdevice_data[nw_key]
            for runID, run_info in nw_data.items():
                run_edit   = run_info["editdict"]
                nwid       = run_edit.get("NWID", "")
                ideality   = run_edit.get("n", "N/A")
                if type(ideality) !=str: ideality = f"{ideality:.5f}"
                ideality_s = run_edit.get("n_series", "N/A")
                if type(ideality_s) !=str: ideality_s = f"{ideality_s:.5f}"
                Rs_lin     = run_edit.get("Rs_lin", "N/A")
                if type(Rs_lin) !=str: Rs_lin = f"{Rs_lin*1e-6:.5f}"
                Rs_series  = run_edit.get("Rs", "N/A")
                if type(Rs_series) !=str: Rs_series = f"{Rs_series*1e-6:.5f}"
                accept     = run_edit.get("accept",None)
                lighton    = run_edit.get("light_on",False) 
                mrange     = run_edit.get("mrange","Unknown")
                if run_edit.get("accept") == True:
                    status = "Accepted"
                elif run_edit.get("accept") == False:
                    status = "Rejected"
                else:
                    status = "Not Done"
                
                run_table.setItem(row, 0, qtw.QTableWidgetItem(f"{nwid}"))
                run_table.setItem(row, 1, qtw.QTableWidgetItem(f"{runID}"))
                run_table.setItem(row, 2, qtw.QTableWidgetItem(ideality))
                run_table.setItem(row, 3, qtw.QTableWidgetItem(ideality_s))
                run_table.setItem(row, 4, qtw.QTableWidgetItem(Rs_lin))
                run_table.setItem(row, 5, qtw.QTableWidgetItem(Rs_series))
                run_table.setItem(row, 6, qtw.QTableWidgetItem(f"{lighton}"))
                run_table.setItem(row, 7, qtw.QTableWidgetItem(f"{mrange}"))
                run_table.setItem(row, 8, qtw.QTableWidgetItem(f"{status}"))

                if lighton and accept:
                    color = 'pale blue'     # Light on + Accepted
                elif lighton and accept is False:
                    color = 'pale violet'   # Light on + Rejected
                elif lighton and accept is None:
                    color = 'pale red'     # Light on + Not Done
                elif accept is True:
                    color = 'pale green'    # Accepted
                elif accept is False:
                    color = 'pale red'      # Rejected
                else:
                    color = 'pale tan'      # Not Done

                for col in range(9):
                    run_table.item(row, col).setBackground(QColor(*get_rgbhex_color(color, ctype="rgba255")))

                row += 1

    # === Update Highlight Colors ===
    palette = run_table.palette()
    selected_items = run_table.selectedItems()
    if selected_items:
        current_color = selected_items[0].background().color()
    else:
        current_color = QColor(255, 255, 255)

    darkened_color = current_color.darker(120)
    palette.setColor(QPalette.Highlight, darkened_color)
    palette.setColor(QPalette.HighlightedText, Qt.white)
    run_table.setPalette(palette)
    

    # ==== Update Button Values ===== #
    
    run_data = self.DATA[selected_device][selected_subdevice][selected_nw][selected_run]
    self.CURRENT_RUN = run_data.copy()
    
    
    
    try:
        self.textbox_update(self.textbox_I0guess,[selected_device,selected_subdevice,selected_nw,selected_run],["fitdict","Initial_Guess","I0"])
        self.textbox_update(self.textbox_nguess,[selected_device,selected_subdevice,selected_nw,selected_run],["fitdict","Initial_Guess","n"])
        
        self.textbox_update(self.textbox_I0fit,[selected_device,selected_subdevice,selected_nw,selected_run],["editdict","ideality_fit","I0"])
        self.textbox_update(self.textbox_nfit,[selected_device,selected_subdevice,selected_nw,selected_run],["editdict","n"])
        
        self.textbox_update(self.textbox_Rslin,[selected_device,selected_subdevice,selected_nw,selected_run],["editdict","Rs_lin"])
        self.textbox_update(self.textbox_Rss,[selected_device,selected_subdevice,selected_nw,selected_run],["editdict","Rs"])
        self.textbox_update(self.textbox_ns_fit,[selected_device,selected_subdevice,selected_nw,selected_run],["editdict","n_series"])
       
        
        self.textbox_update(self.textbox_Imin_plot,[selected_device,selected_subdevice,selected_nw,selected_run],["fitdict","Fit_plot","Imin"])
        self.textbox_update(self.textbox_Imax_plot,[selected_device,selected_subdevice,selected_nw,selected_run],["fitdict","Fit_plot","Imax"])
        
        self.textbox_update(self.textbox_Vmin_plot,[selected_device,selected_subdevice,selected_nw,selected_run],["fitdict","Fit_plot","Vmin"])
        self.textbox_update(self.textbox_Vmax_plot,[selected_device,selected_subdevice,selected_nw,selected_run],["fitdict","Fit_plot","Vmax"])
        self.textbox_update(self.textbox_Npts_plot,[selected_device,selected_subdevice,selected_nw,selected_run],["fitdict","Fit_plot","Npts"])
    
    except Exception as e:  # Catch any exception
        print(f"An error occurred: {e}")  # Print the error message
        print("Textbox update skipped - Initialisation caused early call of function.")
        
    try:
        
        self.set_nested_dict_value(selfvar,["fitdict","Fit_range","V"],self.CURRENT_RUN["fitdict"]["Fit_range"]["V"])
        self.set_nested_dict_value(selfvar,["fitdict","Fit_range","I0"],self.CURRENT_RUN["fitdict"]["Fit_range"]["I0"])
        self.set_nested_dict_value(selfvar,["fitdict","Fit_range","n_range"],self.CURRENT_RUN["fitdict"]["Fit_range"]["n_range"])
        self.set_nested_dict_value(selfvar,["fitdict","Fit_range","Rs_lin"],self.CURRENT_RUN["fitdict"]["Fit_range"]["Rs_lin"])
        self.set_nested_dict_value(selfvar,["fitdict","Fit_range","Rs_mean"],self.CURRENT_RUN["fitdict"]["Fit_range"]["Rs_mean"])
        
        
        
    except Exception as e:
        print(f"An error occurred: {e}")  
        print("Update selection ranges")

def populate_device_list(self,selfvar="session", datavar="DATA", device_widget ="",):
    device_widget.blockSignals(True)
    device_widget.clearSelection()
    device_widget.clear()
    device_data = self.get_nested_dict_value(datavar, None)
    # Set initial device and subdevice selections
    device_widget.addItems(device_data.keys())
    device_widget.blockSignals(False)
    device_widget.setCurrentRow(0)

def create_device_subdevice_list(self, selfvar="session", datavar="DATA", leftkeylist="", rightkeylist="", runkeylist="", left_label="", right_label="", layout=""):
    # Ensure inputs are in the correct format
    if isinstance(leftkeylist, str):
        leftkeylist = list(leftkeylist)
    if isinstance(rightkeylist, str):
        rightkeylist = list(rightkeylist)
    keylists = [leftkeylist,rightkeylist,runkeylist]
    # Get the device data dynamically using getattr and set the initial value if necessary
    device_data = self.get_nested_dict_value(datavar, None)  # Get device data (dict) from the session or other source

    # Set initial device and subdevice selections
    deviceID = self.get_nested_dict_value(selfvar, leftkeylist)
    subdeviceID = self.get_nested_dict_value(selfvar, rightkeylist)
    runID = self.get_nested_dict_value(selfvar, runkeylist)
    
    if any(i<0 for i in [deviceID,subdeviceID,runID]):
        self.set_nested_dict_value(selfvar,leftkeylist,0)
        self.set_nested_dict_value(selfvar,rightkeylist,0)
        self.set_nested_dict_value(selfvar,runkeylist,0)
        deviceID    = self.get_nested_dict_value(selfvar, leftkeylist)
        subdeviceID = self.get_nested_dict_value(selfvar, rightkeylist)
        runID       = self.get_nested_dict_value(selfvar, runkeylist)
    
    # Create layouts
    container_layout = qtw.QVBoxLayout()
    main_splitter = qtw.QSplitter(Qt.Vertical)
    top_splitter = qtw.QSplitter(Qt.Horizontal)

    # Device List
    device_widget = qtw.QListWidget()
    device_widget.addItems(device_data.keys())
    device_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
    # Subdevice List
    subdevice_widget = qtw.QListWidget()
    device_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
    # Run Table
    run_table = qtw.QTableWidget()
    run_table.setColumnCount(9)
    run_table.setHorizontalHeaderLabels(["NWID", "Run ID", "n linear", "n series", "Rs est [MΩ]", "Rs fit [MΩ]", "Light On", "Range", "Status"])
    run_table.horizontalHeader().setSectionResizeMode(qtw.QHeaderView.Stretch)
    run_table.verticalHeader().setVisible(False)
    run_table.setSelectionBehavior(qtw.QAbstractItemView.SelectRows)
    run_table.setSizeAdjustPolicy(qtw.QAbstractScrollArea.AdjustToContents)
    device_widget.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
    
    # Add widgets to the splitters
    top_splitter.addWidget(device_widget)
    top_splitter.addWidget(subdevice_widget)
    main_splitter.addWidget(top_splitter)
    main_splitter.addWidget(run_table)

    container_layout.addWidget(main_splitter)
    listwidgets = [device_widget, subdevice_widget, run_table]
    
    try:
        try:
            self.update_device_list(selfvar="session",listwidgets=listwidgets,keylists=keylists)
            self.update_subdevice_list(selfvar="session",listwidgets=listwidgets,keylists=keylists)
            self.update_all_on_run_change_and_colors(selfvar="session", listwidgets=listwidgets, keylists=keylists)
            
        except:
            self.update_and_reset_listIDs(selfvar="session",listwidgets=listwidgets, keylists=keylists , which="Device")
            self.update_device_list(selfvar="session",listwidgets=listwidgets,keylists=keylists)
            self.update_subdevice_list(selfvar="session",listwidgets=listwidgets,keylists=keylists)
            self.update_all_on_run_change_and_colors(selfvar="session", listwidgets=listwidgets, keylists=keylists)
            
        #Now we try to set the last sessions items, or failing that, resetting as required.
        try:
            device_widget.setCurrentRow(self.get_nested_dict_value(selfvar, leftkeylist))
        except:
            device_widget.setCurrentRow(0)
            self.update_and_reset_listIDs(selfvar="session",listwidgets=listwidgets, keylists=keylists , which="Device")
        try:
            subdevice_widget.setCurrentRow(self.get_nested_dict_value(selfvar, rightkeylist))
        except:
            subdevice_widget.setCurrentRow(0)
            self.update_and_reset_listIDs(selfvar="session",listwidgets=listwidgets, keylists=keylists , which="Subdevice")
        try:
             run_table.setCurrentRow(self.get_nested_dict_value(selfvar, runkeylist))
        except:
             run_table.setCurrentCell(0,0)
             self.update_and_reset_listIDs(selfvar="session",listwidgets=listwidgets, keylists=keylists , which="Run")
    except:
        print("NO DATA LOADED - PLEASE SELECT A DATA STORAGE DIRECTORY OR CRAWL A NEW ONE!")
    # Connect events
    device_widget.currentItemChanged.connect(lambda: 
                                            [ 
                                                self.save_highest_level_json(selfvar="session",keylists=keylists,listwidgets=listwidgets),
                                                self.update_and_reset_listIDs(selfvar="session",listwidgets=listwidgets, keylists=keylists , which="Device"),
                                                self.update_device_list(selfvar="session",listwidgets=listwidgets,keylists=keylists)
                                            ]
                                            )

    subdevice_widget.currentItemChanged.connect(lambda: 
                                                [   
                                                    self.update_and_reset_listIDs(selfvar="session",listwidgets=listwidgets, keylists=keylists , which="Subdevice"),
                                                    self.update_subdevice_list(selfvar="session",listwidgets=listwidgets,keylists=keylists)
                                                 ]
                                                )

    run_table.itemSelectionChanged.connect(lambda: 
                                           [
                                               self.update_and_reset_listIDs(selfvar="session",listwidgets=listwidgets, keylists=keylists , which="Run"),
                                               self.update_all_on_run_change_and_colors(selfvar="session", listwidgets=listwidgets, keylists=keylists),
                                               self.update_spinbox_range(selfvar,self.spinbox_sweep_index),
                                               self.plot_current_data(),
                                               self.replot_saved_ranges()
                                           ]
                                           )
        
    # Finalize layout
    try:
        self.update_all_on_run_change_and_colors(selfvar="session", listwidgets=listwidgets, keylists=keylists)
    except:
        None
    layout.addLayout(container_layout)
    return(listwidgets)

    
def update_and_reset_listIDs(self,selfvar="session",listwidgets=None, keylists ="", which="Device"):
    #Call this function when an index change happens on ANY LEVEL. All this does is save over the new IDs whenever an item selection change happens.
    #At the run level, no resets are made.
    device_widget, subdevice_widget, run_table = listwidgets
    leftkeys, rightkeys, runkeys = keylists
    #We first grab all new device IDs so we don't need to grab them later. Becuase this should only be called during an index change, the data should be in focus.
    if which == "Device":
        self.set_nested_dict_value(selfvar, leftkeys, device_widget.currentRow())
        self.set_nested_dict_value(selfvar, rightkeys, 0)
        self.set_nested_dict_value(selfvar, runkeys, 0)
    if which == "Subdevice":
        self.set_nested_dict_value(selfvar, rightkeys, subdevice_widget.currentRow())
        self.set_nested_dict_value(selfvar, runkeys, 0)
        
    if which == "Run":
        self.set_nested_dict_value(selfvar, runkeys, run_table.currentRow())
    


def textbox_update(self,textbox,runkeylist,partial_keylist):

    value = self.get_nested_dict_value("CURRENT_RUN",partial_keylist)  
    self.set_nested_dict_value("session",partial_keylist,value)
    self.set_nested_dict_value("DATA",runkeylist+partial_keylist,value)
    textbox.setText(str(value))
    
def simple_textbox(self, datavar="CURRENT_RUN", varlist=["editdict","n"], label_text="", layout="", stretch=5):
    """
    Creates a QLineEdit widget (textbox) with a label that contains a known variable.
    The textbox is non-editable and updates the variable when the variable changes.
    """
    # Create a horizontal layout to hold the label and textbox
    container_layout = qtw.QHBoxLayout()

    # Create and set the label
    label = qtw.QLabel(label_text)
    label.setAlignment(Qt.AlignRight)  # Right-align text inside the textbox
    label.setStyleSheet("border: none;")  # Remove any border from the label
    container_layout.addWidget(label, stretch=1)
    
    # Create the textbox
    textbox = qtw.QLineEdit()
    textbox.setText(str(self.get_nested_dict_value(datavar,varlist)))  # Set the initial text to the current value of the variable
    textbox.setStyleSheet("background-color: white;")  # Set white background for the textbox
    textbox.setReadOnly(True)  # Make the textbox non-editable
    container_layout.addWidget(textbox, stretch=stretch)

    # Add the layout to the parent layout
    layout.addLayout(container_layout)

    return textbox
    
def simple_function_textbox(self, selfvar="session", datavar="CURRENT_RUN", runkeylist=None, label_text="", layout="", default_valuetext="1", return_type = "float" , function=None, function_variables={}, stretch=5):
    """
    Creates a QLineEdit widget (textbox) with a label and updates self.CURRENT_RUN['fitdict']
    when Enter is pressed or focus is lost. It only updates if the value is a valid number.
    """
    # Create a horizontal layout to hold the label and textbox
    container_layout = qtw.QHBoxLayout()

    # Create and set the label
    label = qtw.QLabel(label_text)
    label.setAlignment(Qt.AlignRight)  # Right-align text inside the textbox
    label.setStyleSheet("border: none;")  # Remove any border from the label
    container_layout.addWidget(label)
    if runkeylist:
        original_value = self.get_nested_dict_value(datavar,runkeylist,default_text=default_valuetext)

    else:
        original_value = default_valuetext
        
    # Create the textbox
    textbox = qtw.QLineEdit()
    textbox.setStyleSheet("background-color: white;")  # Set white background for the textbox
    label.setAlignment(Qt.AlignLeft)
    textbox.setText(str(original_value))  # Set the default text
    container_layout.addWidget(textbox, stretch=stretch)
    
    # Add the layout to the parent layout
    layout.addLayout(container_layout)
    
    # Function to validate and update fitdict
    def update_fitdict():
        text = textbox.text().strip()  # Get and clean the text
        try:
            # Attempt to convert text to a number (supports scientific notation like 1e-15)
            if return_type == "float":
                value = float(text)
            elif return_type == "int":
                value = int(text)
            # Update the fitdict value
            
            if function:
                if "new_value" in function_variables.keys():

                    function_variables["new_value"] = value
                function(**function_variables)
            
        except ValueError:
            # Invalid input; do nothing
            print(f"Invalid input: {text}")  # Debugging print statement
            textbox.setText(str(original_value))  # Reset to default if invalid
        
    # Connect signals for validation and update
    textbox.editingFinished.connect(update_fitdict)  # Triggered on focus loss or Enter
    
    return textbox
    
def simple_spinbox(self, selfvar="session", label_text="", layout="",stretch=2):
    """
    Creates a QSpinBox widget with a label, and sets its range based on the length of the data.
    If an update function is provided, it will be called whenever the range changes.
    """
    # Get the upper and lower limits based on the length of the data
    min_value = 0  # Set minimum value to 0 (or a different value if needed)
    try:
        data_length = len(self.CURRENT_RUN["data"]["voltage"])
        max_value = data_length - 1  # Set maximum value to length of the data - 1
    except:
        data_length = 0
        max_value = 0
    
    # Create a horizontal layout to hold the label and the spin box
    container_layout = qtw.QHBoxLayout()

    # Create and set the label
    label = qtw.QLabel(label_text)
    container_layout.addWidget(label, stretch=stretch)

    # Create the QSpinBox widget
    spin_box = qtw.QSpinBox(self)
    spin_box.setRange(min_value, max_value)  # Set the range based on the data length
    spin_box.setValue(min_value)  # Set initial value (optional)

    # Optional: set step size (for example, step of 1)
    spin_box.setSingleStep(1)
    
    # Add the spin box to the layout
    container_layout.addWidget(spin_box, stretch=1)

    # Add the layout to the parent layout
    layout.addLayout(container_layout)

    # Automatically update the range when CURRENT_RUN changes
    self.update_spinbox_range(selfvar, spin_box)

    return spin_box

def update_spinbox_value(self,spinbox):
    
    try: 
        self.set_nested_dict_value("CURRENT_RUN",["fitdict","sweep_index"],spinbox.value())
        self.plot_current_data()
    except:
        print("No data loaded, please choose a run")
    
            
def update_spinbox_range(self, selfvar, spinbox):
    """
    Updates the range and value of the spin box whenever self.CURRENT_RUN changes.
    """
    
    try:
        data_length = len(self.CURRENT_RUN["data"]["voltage"])
        max_value = data_length - 1  # Set the maximum value to length of the data - 1
    except:
        data_length = 0
        max_value = 0

    # Update the spin box range and value
    spinbox.setRange(0, max_value)
    # Optionally reset the spin box value (for example, to the first value)
    try:
        savedval =  int(self.get_nested_dict_value("CURRENT_RUN",["fitdict","sweep_index"]))
    except:
        savedval = 0
        
    if savedval > max_value:
        spinbox.setValue(0)  # You can modify this based on your specific logic
    else:
        spinbox.setValue(savedval)
        
    self.update_spinbox_value(spinbox)

def simple_function_button(self, selfvar="session", label_text="",button_text="", default_text="", keylist="", layout="",function=None,function_variables={},select_string = None,stretch=5):
    # Get text from session if available; otherwise, use default_text
    button_text = button_text
    
    # Create a horizontal layout to hold the label and the button
    container_layout = qtw.QHBoxLayout()
    
    # Create and set the label
    label = qtw.QLabel(label_text)
    container_layout.addWidget(label,stretch=1)
    
    # Create the button
    button = qtw.QPushButton()
    button.setText(button_text)
    function_variables["selfvar"] = selfvar
    button.clicked.connect(lambda: function(**function_variables))
    
    container_layout.addWidget(button,stretch=stretch)
    container_layout.setAlignment(Qt.AlignCenter)
    # Add the layout to the parent layout
    layout.addLayout(container_layout)

    return button

        
def create_directory_selection_button(self, selfvar="session", label_text="", default_text="", keylist="", layout="",select_string = None):
    # Get text from session if available; otherwise, use default_text
    
    button_text = self.get_nested_dict_value(selfvar, keylist,default_text=default_text)
    
    # Create a horizontal layout to hold the label and the button
    container_layout = qtw.QHBoxLayout()
    
    # Create and set the label
    label = qtw.QLabel(label_text)
    container_layout.addWidget(label,stretch=1)
    
    # Create the button
    button = qtw.QPushButton()
    button.setText(button_text)
    button.clicked.connect(lambda: self.select_directory(selfvar,button, keylist,select_string = select_string))
    
    container_layout.addWidget(button,stretch=5)
    
    # Add the layout to the parent layout
    layout.addLayout(container_layout)
  
    return button

def update_directory_button(self,selfvar, button, keylist):
    
    directory = self.get_nested_dict_value(selfvar, keylist)
    
    if directory:
        folders = [filename for filename in os.listdir(directory) if os.path.isdir(os.sep.join([directory,filename]))]
        for dfolder in self.USERMODE_Types:
            if dfolder not in folders:
                folderdir = os.sep.join([directory,dfolder])
                os.mkdir(os.path.abspath(folderdir))
                print("created new data folder at" + folderdir)
                
        button.setText(directory)
        qcol = QColor(*get_rgbhex_color("light green",ctype="rgba255"))
        button.setStyleSheet(f"background-color: rgb({qcol.red()}, {qcol.green()}, {qcol.blue()});")
    else:
        button.setText("CLICK TO SELECT DIRECTORY")
        qcol = QColor(*get_rgbhex_color("light red",ctype="rgba255"))
        button.setStyleSheet(f"background-color: rgb({qcol.red()}, {qcol.green()}, {qcol.blue()});")
    
    if "Data_Storage" in keylist:
        self.read_data_storage()
        try:
            self.populate_device_list(selfvar=selfvar,datavar = "DATA",device_widget = self.device_subdevice_list[0])
            print("Data succesfully loaded from DataStorage. If you just selected it, the last error was wrong!")
        except:
            print("Data does not yet exist in DataStorage. Add some files first, then select it again!")
def update_usermode_button_directory(self,selfvar, button, rootlist,usermodeIDkeys,keylist):
    rootdir        = self.get_nested_dict_value(selfvar, rootlist)
    self.USERMODE  = self.USERMODE_Types[self.get_nested_dict_value(selfvar,usermodeIDkeys)] 
    newdir         = os.sep.join([rootdir,self.USERMODE ])
    self.set_nested_dict_value(selfvar,keylist,newdir)
    print("Changed user mode directory to "+newdir)
        
def read_data_storage(self):
    #We will do data storage in this way: 
    #device (DFR1-EG) -> #subdevice (BR1) -> runID + NWID -> [deviceID, subdeviceID, NWID, Vdata, Idata, Vfitdata, Ifitdata, inital_guess, plot_range,fit_range,light_on, accept_data]
    
    datafilepath = os.sep.join([self.session["Directories"]["Data_Storage"],self.USERMODE])
    try:
        for file in [f for f in os.listdir(datafilepath) if f.endswith(".json")]:    
            self.DATA[file.replace(".json","")] = json_loaddata(os.sep.join([datafilepath,file]))
    except:
        print("Please select new data directories")
    
def select_directory(self,selfvar, button, keylist,select_string=None):
    if select_string == None:
        select_string = "Select " + keylist[-1].title() + " Directory"
    
    directory = os.path.abspath(qtw.QFileDialog.getExistingDirectory(self, select_string))
    if directory:
        self.set_nested_dict_value(selfvar, keylist, directory) 
        
        self.update_directory_button(selfvar, button, keylist)
        if keylist == "Data_Storage":
            self.read_data_storage()
            
            
#Custom Console Class
class ConsoleOutput(qtw.QPlainTextEdit):
    """Custom console output widget."""
    def __init__(self, max_lines=50, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)  # Make the console output read-only
        self.max_lines = max_lines  # Maximum lines to store
        self.buffer = []  # Internal buffer for lines
    
    def write(self, message):
        """Append text to the console and trim old lines."""
        self.buffer.append(message.strip())  # Add new line to buffer
        if len(self.buffer) > self.max_lines:
            self.buffer.pop(0)  # Remove oldest line if buffer exceeds limit
        
        # Update displayed text
        self.setPlainText('\n'.join(self.buffer))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def flush(self):
        """Flush method for compatibility, no-op."""
        pass
            
#PLOTTING FUNNCTIONS!
def Correct_Forward_Voltage(self,I,V):
    #We need to turn the data the right way round (forward bias is positive voltages for positive currents)
    if max(I) < np.abs(min(I)):
        I = np.multiply(-1,I)
        V = np.multiply(-1,V)

    if V[-1]<V[0]:
        V = np.flip(V)
        I = np.flip(I)

    if type(V) == list:
        V = np.array(V)
    if type(I) == list:
        I = np.array(I)
    return(I,V)

def alter_Vmax(self,selfvar="session",increment=1):
    Vmax = self.get_nested_dict_value("session",["fitdict","Fit_plot","Vmax"]) + increment
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=Vmax,partial_keylist=["fitdict","Fit_plot","Vmax"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])

def alter_n_guess(self,selfvar="session",increment=1):
    n = self.get_nested_dict_value("session",["fitdict","Initial_Guess","n"]) + increment
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=n,partial_keylist=["fitdict","Initial_Guess","n"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])

def alter_Vmax_and_n_guess(self,selfvar="session",increment=1):
    Vmax = self.get_nested_dict_value("session",["fitdict","Fit_plot","Vmax"]) + increment
    n = self.get_nested_dict_value("session",["fitdict","Initial_Guess","n"]) + increment
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=n,partial_keylist=["fitdict","Initial_Guess","n"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite",new_value=Vmax,partial_keylist=["fitdict","Fit_plot","Vmax"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])

def extract_negative_segments_and_dots(V, I):
    """
    Extracts continuous negative segments and single negative dots from paired data (V, I).

    Args:
        V (np.array): Array of V values.
        I (np.array): Array of I values.
    Returns:
        segments (list): List of tuples, where each tuple contains (V_segment, I_segment) for continuous negative segments.
        dots (list): List of tuples, where each tuple contains (V_dot, I_dot) for single negative points.
    """
    I = np.array(I)
    V = np.array(V)

    negative_indices = np.where(I < 0)[0] #all negative values

    segments_indices = np.split(negative_indices, np.where(np.diff(negative_indices) != 1)[0] + 1)     # Split negative indices into continuous segments

    segments = [];    dots = []

    for segment in segments_indices:
        if len(segment) >= 2:  # Continuous segment
            segments.append((V[segment], I[segment]))
        elif len(segment) == 1:  # Single negative point
            dots.append((V[segment], I[segment]))
    return segments, dots
    
    
def plot_current_data(self):
    lw_wide   = 5
    lw_mid    = 3
    lw_narrow = 2
    self.ax.clear()
    c_IV    = get_rgbhex_color("green",ctype="rgba")
    c_IVneg = get_rgbhex_color("pale lime",ctype="rgba")
    c_ideal = get_rgbhex_color("light violet",ctype="rgba")
    c_series = get_rgbhex_color("red",ctype="rgba")
    c_seriesn = get_rgbhex_color("tan",ctype="rgba")
    
    RD = self.get_nested_dict_value("CURRENT_RUN",None)
    fitdict   = RD["fitdict"]
    data      = RD["data"]
    edit_vars = RD["editdict"]
    current,voltage = self.Correct_Forward_Voltage(data["current"][fitdict["sweep_index"]],data["voltage"][fitdict["sweep_index"]])
    if self.USERMODE == "Ideality":
        if self.session["PlotWhich"]["Ideality"]["IV"]:    
            self.ax.set_xlabel("Voltage [V]")
            self.ax.set_ylabel("Current [A]")
            Vfit = edit_vars["ideality_fit"]["Vfit"]
            Ifit = edit_vars["ideality_fit"]["Ifit"]
            
            Vfits = edit_vars["series_fit"]["Vfit"]
            Ifits = edit_vars["series_fit"]["Ifit"]
            
            try:
                IVfitneg =  extract_negative_segments_and_dots(Vfit,Ifit)
            except:
                IVfitneg = False
            try:
                IVneg =  extract_negative_segments_and_dots(voltage,current)
            except:
                IVneg = False
            
            
            try:
                IVfitsneg =  extract_negative_segments_and_dots(Vfits,Ifits)
            except:
                IVfitsneg = False

            n    =  edit_vars["n"]
            
            if self.is_log_scale: 
                self.ax.plot(voltage,np.abs(current),linewidth=lw_wide,c=c_IV)
                if IVfitneg:
                    for v_segment, i_segment in IVfitneg[0]:
                        self.ax.plot(v_segment, np.abs(i_segment),linewidth=lw_mid, color=c_IVneg)
    
                    # Plot dots
                    for v_dot, i_dot in IVfitneg[1]:
                        self.ax.plot(v_dot, np.abs(i_dot), 'o', color=c_IVneg)
                
                        
                if IVneg:
                    for v_segment, i_segment in IVneg[0]:
                        self.ax.plot(v_segment, np.abs(i_segment),linewidth=lw_wide, color=c_IVneg)
    
                    # Plot dots
                    for v_dot, i_dot in IVneg[1]:
                        self.ax.plot(v_dot, np.abs(i_dot), 'o', color=c_IVneg)
                
                    
                if IVfitsneg:
                    for v_segment, i_segment in IVfitsneg[0]:
                        self.ax.plot(v_segment, np.abs(i_segment), color=c_seriesn)
    
                    # Plot dots
                    for v_dot, i_dot in IVfitsneg[1]:
                        self.ax.plot(v_dot, np.abs(i_dot), 'o', color=c_seriesn)
                        
                self.ax.set_yscale("log")
                self.toggle_log_button.setText("Log")
        
                if not isinstance(Vfit,(type(None), str)) and not isinstance(Ifit,(type(None),str)):
                    self.ax.plot(Vfit,np.abs(Ifit),linewidth=lw_mid,c=c_ideal)
                
                if not isinstance(Vfits,(type(None), str)) and not isinstance(Ifits,(type(None),str)):
                    self.ax.plot(Vfits,np.abs(Ifits),linewidth=lw_narrow,c=c_series)
                    
               
                
                
            else:
                self.ax.plot(voltage,current,c=c_IV)
                if not isinstance(Vfit,(type(None), str)) and not isinstance(Ifit,(type(None),str)):
                    self.ax.plot(Vfit,Ifit,c=c_ideal)
                
                if IVfitneg:
                    for v_segment, i_segment in IVfitneg[0]:
                        self.ax.plot(v_segment, i_segment, color=c_IVneg)
    
                    # Plot dots
                    for v_dot, i_dot in IVfitneg[1]:
                        self.ax.plot(v_dot, np.abs(i_dot), 'o', color=c_IVneg)
                if IVneg:
                    for v_segment, i_segment in IVneg[0]:
                        self.ax.plot(v_segment, i_segment, color=c_IVneg)
    
                    # Plot dots
                    for v_dot, i_dot in IVneg[1]:
                        self.ax.plot(v_dot, i_dot, 'o', color=c_IVneg)
                        
                
                self.ax.set_yscale("linear")
                self.toggle_log_button.setText("Linear")
         
        if self.session["PlotWhich"]["Ideality"]["Rs"]:
            self.ax.set_xlabel("Voltage [V]")
            self.ax.set_ylabel("Rs [$\Omega$]")
            
            Vfits = edit_vars["series_fit"]["Vfit"]
            
            
            try:
                Rs_fit = edit_vars["series_fit"]["Rs_fit"]
                Rsn =  extract_negative_segments_and_dots(Vfit,Rs_fit)
            except:
                Rsn = False
            try:            
                if self.is_log_scale: 
                    self.ax.plot(voltage,np.abs(Rs_fit),c=c_IV)
                    if Rsn:
                        for v_segment, r_segment in Rsn[0]:
                            self.ax.plot(v_segment, np.abs(r_segment), color=c_IVneg)
        
                        # Plot dots
                        for v_dot, r_dot in Rsn[1]:
                            self.ax.plot(v_dot, np.abs(r_dot), 'o', color=c_IVneg)
                    
                    self.ax.set_yscale("log")
                    self.toggle_log_button.setText("Log")
   
                
                else:
                    self.ax.plot(voltage,Rs_fit,c=c_IV)
                    
                    if Rsn:
                        for v_segment, r_segment in Rsn[0]:
                            self.ax.plot(v_segment, r_segment, color=c_IVneg)
        
                        # Plot dots
                        for v_dot, r_dot in Rsn[1]:
                            self.ax.plot(v_dot, r_dot, 'o', color=c_IVneg)
                            
                    
                    self.ax.set_yscale("linear")
                    self.toggle_log_button.setText("Linear")
            
            
                        
            except:
                None
            
        if self.session["PlotWhich"]["Ideality"]["ns"]:
            self.ax.set_xlabel("Voltage [V]")
            self.ax.set_ylabel("Ideality")
            
            Vfits = edit_vars["series_fit"]["Vfit"]
            n_fit = edit_vars["series_fit"]["n_fit"] 
            
            try:
                nsn =  extract_negative_segments_and_dots(Vfit,n_fit)
            except:
                nsn = False
            try:            
                if self.is_log_scale: 
                    self.ax.plot(voltage,np.abs(n_fit),c=c_IV)
                    if nsn:
                        for v_segment, r_segment in nsn[0]:
                            self.ax.plot(v_segment, np.abs(r_segment), color=c_IVneg)
        
                        # Plot dots
                        for v_dot, r_dot in nsn[1]:
                            self.ax.plot(v_dot, np.abs(r_dot), 'o', color=c_IVneg)
                    
                    self.ax.set_yscale("log")
                    self.toggle_log_button.setText("Log")
   
                
                else:
                    self.ax.plot(voltage,n_fit,c=c_IV)
                    
                    if nsn:
                        for v_segment, r_segment in nsn[0]:
                            self.ax.plot(v_segment, r_segment, color=c_IVneg)
        
                        # Plot dots
                        for v_dot, r_dot in nsn[1]:
                            self.ax.plot(v_dot, r_dot, 'o', color=c_IVneg)
                            
                    
                    self.ax.set_yscale("linear")
                    self.toggle_log_button.setText("Linear")
            except:
                None
                    
        self.get_axscale_set_lim()
            
        
        if all(lim is None for lim in fitdict["axis_lim"]["xlim"]):
            self.get_axscale_set_lim(operation="manual",xlim="auto",ylim="none")
        else:
            xmin = fitdict["axis_lim"]["xlim"][0] or np.min(voltage) 
            xmax = fitdict["axis_lim"]["xlim"][1] or np.max(voltage)    
            #self.ax.set_xlim([xmin,xmax])    
        
        if all(lim is None for lim in fitdict["axis_lim"]["ylim"]):
            self.get_axscale_set_lim(operation="manual",xlim="none",ylim="auto")
        else:
            ymax = fitdict["axis_lim"]["ylim"][1] or np.max(current)
            #self.ax.set_ylim([ymin,ymax])    
        
        if "fit_data" in data.keys():
            vfit = data["fit_data"]["voltage"]
            ifit = data["fit_data"]["current"]
            self.ax.plot(vfit,ifit,"--",c = get_rgbhex_color("light green",type="rgba"))
        
    self.canvas.draw()
    
def add_plot(self, canvas):
    """Initial plot setup."""
    self.ax = canvas.figure.add_subplot(111)  # Initialize ax
    p = self.splash_screen_polygon
   
    self.ax.add_patch(p)
    
    self.ax.set_xlim([-15,15]); self.ax.set_ylim([-10,10])
    self.ax.set_xlabel("Voltage [V]")
    self.ax.set_ylabel("Current [A]")
    self.canvas.draw()


def get_axscale_set_lim(self,operation="Auto Both",xlim="auto",ylim="auto"):
    
    tbb = self.ax.dataLim
    
    if "auto" in operation.lower():
        if "both" in operation.lower():
            operation += "xy"
        if "x" in operation.lower():
            self.ax.set_xlim([tbb.x0,tbb.x1])
        if "y" in operation.lower():
            self.ax.set_ylim([tbb.y0,tbb.y1])
        
    elif "manual" in operation.lower():
        if xlim == "auto": 
            xlim = [tbb.x0,tbb.x1]
            self.ax.set_xlim(xlim);
        if ylim == "auto": 
            ylim = [tbb.y0,tbb.y1]    
            self.ax.set_ylim(ylim)
    
def toggle_axis_scale(self):
    """Toggle the y-axis scale between linear and logarithmic."""
   
    self.is_log_scale = not self.is_log_scale
    
    if self.is_log_scale:
        self.ax.set_yscale("log")
        self.toggle_log_button.setText("Log")
        self.get_axscale_set_lim()
        
    else:
        self.ax.set_yscale("linear")
        self.toggle_log_button.setText("Linear")
        self.get_axscale_set_lim()
    # Re-plot the data
    
    self.canvas.draw()
    
    
    
    
"""
FIT FUNCTIONS

"""
def fit_guide_hide_ranges(self,selfvar=""):
    print("Hiding Range")
    visibility = not(self.session["Cursors"]["I0_cursor"]["visible"])
    for key in self.session["Cursors"].keys():
        self.session["Cursors"][key]["visible"] =  visibility

        
    self.alter_span_visibility(self.spans,visibility)
    
    if self.session["Cursors"]["I0_cursor"]["visible"]:
        self.fit_guide_toggle_hide.setStyleSheet("background-color: lightgrey;")
        
    if not(self.session["Cursors"]["I0_cursor"]["visible"]):
        self.fit_guide_toggle_hide.setStyleSheet("background-color: red;")
        

def fit_guide_setactive(self,selector=None,**kwargs):
    print("Setting new active")
    if self.get_attribute_name(selector).replace("range_","") in ["I0","n_linear"]:
        self.alter_span_visibility([self.range_Rs_linear,self.range_n_series, self.range_Rs_mean],False)
        
        for key in ["Rs_linear_cursor","n_series_cursor","Rs_mean_cursor"]:
            self.session["Cursors"][key]["visible"] =  False
    
    if self.get_attribute_name(selector).replace("range_","") in ["Rs_linear","n_series","Rs_mean"]:
        self.alter_span_visibility([self.range_I0,self.range_n_linear],False)
        
        for key in ["Rs_linear_cursor","n_series_cursor","Rs_mean_cursor"]:
            self.session["Cursors"][key]["visible"] =  False
        
    for sel in self.fit_guide_buttons:
        fitname = self.get_attribute_name(sel).replace("fit_guide_","")
        
                
        for span in self.spans:
            if fitname != self.get_attribute_name(selector).replace("range_",""):
                if (fitname in self.get_attribute_name(span).replace("range_","")) and span.get_active():
                    sel.setStyleSheet("background-color: lightgrey;")
                    
        if fitname == self.get_attribute_name(selector).replace("range_",""):
            sel.setStyleSheet("background-color: lightgreen;")
        
    self.update_all_spans_and_make_active(selector)
    
            
    
    
def fit_guide_fitvals(self,selfvar="session",datavar="DATA",currentvar="CURRENT_RUN",listwidgets="", keylists=""):
    RD = self.get_nested_dict_value("CURRENT_RUN",None)
    fitdict   = RD["fitdict"]
    data      = RD["data"]
    edit_vars = RD["editdict"]
    current,voltage = self.Correct_Forward_Voltage(data["current"][fitdict["sweep_index"]],data["voltage"][fitdict["sweep_index"]])
    
    IDF = Ideality_Factor(current,voltage, Vrange=fitdict["Fit_range"]["V"],p0=[fitdict["Initial_Guess"]["I0"],fitdict["Initial_Guess"]["n"]], Iplot_range=[fitdict["Fit_plot"]["Imin"],fitdict["Fit_plot"]["Imax"]],Vplot_range=[fitdict["Fit_plot"]["Vmin"],fitdict["Fit_plot"]["Vmax"]],N=fitdict["Fit_plot"]["Npts"] )
    
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite", new_value=IDF, 
                            partial_keylist=["editdict","ideality_fit"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
    
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite", new_value=IDF["n"], 
                            partial_keylist=["editdict","n"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
    
    
    self.plot_current_data()
    

def fit_guide_fitseriesvals(self,selfvar="session",datavar="DATA",currentvar="CURRENT_RUN",listwidgets="", keylists=""):
    RD = self.get_nested_dict_value("CURRENT_RUN",None)
    fitdict   = RD["fitdict"]
    data      = RD["data"]
    edit_vars = RD["editdict"]
    current,voltage = self.Correct_Forward_Voltage(data["current"][fitdict["sweep_index"]],data["voltage"][fitdict["sweep_index"]])
    n_guess   = edit_vars["n"]
    I_0       = edit_vars["ideality_fit"]["I0"]
    Rs_lin   = fitdict["Fit_range"]["Rs_lin"]
    n_range   = fitdict["Fit_range"]["n_range"]
    Rs_range   = fitdict["Fit_range"]["Rs_mean"]
    
    IDF = Ideality_Factor_Series(current,voltage,n_guess,I_0, Rs_lin, n_range, Rs_range)
    
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite", new_value=IDF, 
                            partial_keylist=["editdict","series_fit"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
    
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite", new_value=IDF["n"], 
                            partial_keylist=["editdict","n_series"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
    
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite", new_value=IDF["Rs"], 
                            partial_keylist=["editdict","Rs"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
    
    self.update_rundata_variable(selfvar="selfvar",datavar="DATA", alter_type="Overwrite", new_value=IDF["Rs_lin"], 
                            partial_keylist=["editdict","Rs_lin"],listwidgets=self.device_subdevice_list,keylists=[["List_Indices","DeviceID"],["List_Indices","SubdeviceID"],["List_Indices","RunID"]])
    
    self.plot_current_data()
    
def Ideality_Factor(I,V, T = 273, Vrange=[0,1],N=1000,p0=[2,1e-15],Iplot_range=None,Vplot_range=None):


    q = scipy.constants.e
    k = scipy.constants.Boltzmann 
    
    #def Diode_EQ(V,I_0,n):
    #    return(I_0 * np.exp((q*V)/(n*k*T)))    
    
    def Diode_EQ(V,I_0,n):
        return(I_0 * (np.exp((q*V)/(n*k*T)) - 1)) 
    
    if type(V) == list:
        V = np.array(V)
    if type(I) == list:
        I = np.array(I)

    V_fit = V[np.where((V>Vrange[0]) & (V<Vrange[1]))]
    I_fit = I[np.where((V>Vrange[0]) & (V<Vrange[1]))]
  
    
    popt, pcov = scipy.optimize.curve_fit(Diode_EQ, V_fit, I_fit,p0=p0)
    
    V_new  = np.linspace(np.min(Vplot_range[0]),np.min(Vplot_range[1]),N)
    I_new  = Diode_EQ(V_new, *popt)
    
    V_new = V_new[np.where((I_new>Iplot_range[0]) & (I_new < Iplot_range[1]))]
    I_new = I_new[np.where((I_new>Iplot_range[0]) & (I_new < Iplot_range[1]))]


    return({"I0":popt[0],"n":popt[1],"par":popt,"covar":pcov,"Vfit":V_new,"Ifit":I_new})

def Ideality_Factor_Series(I,V,n_guess,I_0, Rs_lin, n_range, Rs_range,T=273):

    q = scipy.constants.e
    k = scipy.constants.Boltzmann 
    if type(V) == list:
        V = np.array(V)
    if type(I) == list:
        I = np.array(I)
        
    # Estimate Rs from the high-voltage slope
    
    Rsl_inds = np.where((V>Rs_lin[0])&(V<Rs_lin[1]))[0]
    n_sinds  = np.where((V>n_range[0])&(V<n_range[1]))[0]
    Rs_inds  = np.where((V>Rs_range[0])&(V<Rs_range[1]))[0]
    
    high_voltage_slope = np.polyfit(I[Rsl_inds], V[Rsl_inds], 1)[0]
    Rs_guess = high_voltage_slope
    
    V_data = V # Voltage data
    I_data = I  # Current data
    N = len(V_data)
    V_fit = np.linspace(min(V_data), max(V_data), N)  # Voltage range for fitted curve
    # Create an interpolator for experimental I-V data
    I_interp = np.interp(V_fit,V_data,I_data)
    
    
    
    ### DIODE EQUATION ###
    def diode_eq_RS(I, V, I_0, n, Rs):
        # Clip the argument of exp to avoid overflow
        exponent = (q * V - I * Rs) / (n * k * T)
        exponent = np.clip(exponent, -100, 100)  # Limit exponent to a reasonable range
        return I_0 * (np.exp(exponent) - 1) - I
    
    ### DIODE EQUATION SOLVED FOR Rs ###
    def diode_eq_RS_solve_Rs(Rs, V, I, I_0, n):
        # Rearranged diode equation to solve for Rs
        exponent = (q * (V - I * Rs)) / (n * k * T)
        exponent = np.clip(exponent, -100, 100)  # Limit exponent to avoid overflow
        return I_0 * (np.exp(exponent) - 1) - I
    
    ### DIODE EQUATION SOLVED FOR n ###
    def diode_eq_n_solve_n(n, V, I, I_0, Rs):
        # Rearranged diode equation to solve for n
        exponent = (q * (V - I * Rs)) / (n * k * T)
        exponent = np.clip(exponent, -100, 100)  # Limit exponent to avoid overflow
        return I_0 * (np.exp(exponent) - 1) - I
    
    # Wrapper function to solve for I given V and parameters
    def solve_diode_current(vpt, I_0, n, Rs, max_retries=5,suppress_text=False):
        # Use interpolation to get a better initial guess for I
        I_guess = I_interp[np.where(V_fit >= vpt)[0][0]]  # Interpolate experimental current at vpt
        I_upper = I_guess
        I_lower = I_guess
        retries = 0
        
        while retries < max_retries:
            # Solve for I using root-finding
            result = root(diode_eq_RS, I_upper, args=(vpt, I_0, n, Rs))
            if result.success:
                return result.x[0], Rs  # Return the computed current I and the provided Rs
            else:
                I_upper *= 1.05 
                I_lower *= 0.95  # Adjust the initial guess (e.g., increase by 10%)
                
            
            # Solve for I using lower bound root-finding
            result = root(diode_eq_RS, I_lower, args=(vpt, I_0, n, Rs))
            if result.success:
                return result.x[0], Rs  # Return the computed current I and the provided Rs
            else:
                retries += 1
    
        # If all retries fail, return the last guess
        if not suppress_text:
            print(f"Error: Root-finding failed for V = {vpt}, I_0 = {I_0}, n = {n}, Rs = {Rs} after {max_retries} retries.")
        return I_guess, Rs
    
    # Function to compute the model current for given parameters
    def model_current(V, I_0, n, Rs,suppress_text=False):
        I_fit = np.zeros_like(V)
        Rs_array = np.full_like(V, Rs)  # Array to store Rs values (constant for this step)
        for i, v in enumerate(V):
            I_fit[i], _ = solve_diode_current(v, I_0, n, Rs,suppress_text=suppress_text)
        return I_fit, Rs_array
    
    
    # Wrapper function to solve for Rs given V, I, I0, and n
    def solve_diode_resistance(vpt, I_pt, I_0, n, max_retries=5,suppress_text=False):
        # Initial guess for Rs (e.g., from the high-voltage slope)
        Rs_guess = np.polyfit(I_data[-10:], V_data[-10:], 1)[0]
        Rs_upper = Rs_guess
        Rs_lower = Rs_guess
        retries = 0
    
        while retries < max_retries:
            # Solve for Rs using root-finding
            result = root(diode_eq_RS_solve_Rs, Rs_upper, args=(vpt, I_pt, I_0, n))
            if result.success:
                return result.x[0]  # Return the computed Rs
            else:
                
                Rs_upper *= 1.05  # Adjust the initial guess by 5%
                Rs_lower *= 0.95
    
            # Retry solve for Rs using lower bound root-finding
            result = root(diode_eq_RS_solve_Rs, Rs_lower, args=(vpt, I_pt, I_0, n))
            if result.success:
                return result.x[0]  # Return the computed Rs
            else:
                retries += 1
                
        # If all retries fail, return the last guess
        if not suppress_text:    
            print(f"Error: Root-finding failed for V = {vpt}, I = {I_pt}, I_0 = {I_0}, n = {n} after {max_retries} retries.")
        return Rs_guess
    
    # Wrapper function to solve for n given V, I, I0, and Rs
    def solve_diode_ideality(vpt, I_pt, I_0, Rs, n_guess, max_retries=5,suppress_text=False):
        retries = 0
        n_upper = n_guess
        n_lower = n_guess
        while retries < max_retries:
            # Solve for n using root-finding
            result = root(diode_eq_n_solve_n, n_upper, args=(vpt, I_pt, I_0, Rs))
            if result.success:
                return result.x[0]  # Return the computed n
            else:
                n_upper *= 1.01  # Adjust the initial guess by 5%
                n_lower *=0.95
                
            # Retry solve for lower bound using root-finding
            result = root(diode_eq_n_solve_n, n_lower, args=(vpt, I_pt, I_0, Rs))
            if result.success:
                return result.x[0]  # Return the computed n
            else:
                retries += 1
        # If all retries fail, return the last guess
        if not suppress_text:
            print(f"Error: Root-finding failed for V = {vpt}, I = {I_pt}, I_0 = {I_0}, Rs = {Rs} after {max_retries} retries.")
        return n_guess
    
    # Function to compute Rs for each point in the I-V curve
    def compute_rs_values(V_data, I_data, I_0, n,suppress_text=True):
        Rs_values = np.zeros_like(V_data)
        for i, (v, I_pt) in enumerate(zip(V_data, I_data)):
            Rs_values[i] = solve_diode_resistance(v, I_pt, I_0, n,suppress_text=suppress_text)
        return Rs_values
    
    # Function to compute n for each point in the I-V curve
    def compute_n_values(V_data, I_data, I_0, Rs,suppress_text=True):
        n_values = np.zeros_like(V_data)
        for i, (v, I_pt) in enumerate(zip(V_data, I_data)):
            n_values[i] = solve_diode_ideality(v, I_pt, I_0, Rs, n_guess,suppress_text=suppress_text)
        return n_values
    
    ### DIODE EQUATION SOLVED FOR I0 ###
    def diode_eq_I0_solve_I0(I_0, V, I, n, Rs):
        exponent = (q * (V - I * Rs)) / (n * k * T)
        exponent = np.clip(exponent, -100, 100)  # Limit exponent to avoid overflow
        return I_0 * (np.exp(exponent) - 1) - I
    
    # Wrapper function to solve for I0 given V, I, n, and Rs
    def solve_diode_I0(vpt, I_pt, n, Rs, I0_guess, max_retries=10,suppress_text=False):
        retries = 0
        I0_upper = I0_guess
        I0_lower = I0_guess
        while retries < max_retries:
            # Solve for I0 using root-finding
            result = root(diode_eq_I0_solve_I0, I0_upper, args=(vpt, I_pt, n, Rs))
            if result.success:
                return result.x[0]  # Return the computed I0
            else:
                I0_upper *= 1.05  # Adjust upper bound
                I0_lower *=0.95 # Adjust the lower bound for the next step     
            
            # attempt resolve with lower bound I0 
            result = root(diode_eq_I0_solve_I0, I0_lower, args=(vpt, I_pt, n, Rs))
            if result.success:
                return result.x[0]  # Return the computed I0
            else:
                retries += 1
                
        # If all retries fail, return the last guess
        if not suppress_text:
            print(f"Error: Root-finding failed for V = {vpt}, I = {I_pt}, n = {n}, Rs = {Rs} after {max_retries} retries.")
        return I0_guess
    
    # Function to compute I0 for each point in the I-V curve
    def compute_I0_values(V_data, I_data, n, Rs):
        I0_values = np.zeros_like(V_data)
        for i, (v, I_pt) in enumerate(zip(V_data, I_data)):
            I0_values[i] = solve_diode_I0(v, I_pt, n, Rs, I_0)
        return I0_values
    
    
    def iterative_fit(V_data, I_data, I0, n_guess, num_iterations=2):
        """
        Perform iterative fitting for Rs and n.
        Parameters:
            V_data (np.array): Voltage data.
            I_data (np.array): Current data.
            I0 (float): Reverse Bias saturation current
            n_guess (float): Initial ideality factor guess.
            num_iterations (int): Number of iterations to repeat the process.

        Returns:
            Rs_mean (float): Mean Rs from the last iteration.
            n_mean (float): Mean n from the last iteration.
        """
        Rs_mean = None
        n_mean = n_guess

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}:")

            # Step 1: Compute Rs for each point in the I-V curve
            Rs_values = compute_rs_values(V_data, I_data, I0, n_mean,suppress_text=True)

            # Step 2: Calculate the mean Rs for the last 10 points
            Rs_mean = np.mean(Rs_values[-10:])
            print(f"Mean Rs (last 10 points) = {Rs_mean:.4f} Ω")

            # Step 3: Compute n for each point in the I-V curve with Rs fixed
            n_values = compute_n_values(V_data, I_data, I0, Rs_mean,suppress_text=True)

            # Step 4: Calculate the mean n for the last 10 points
            n_mean = np.mean(n_values[np.where(V_data > 0.5)[0][:5]])
            print(f"Mean n (last 10 points) = {n_mean:.4f}")

        return Rs_mean, n_mean, Rs_values, n_values

    # Perform iterative fitting
    Rs_mean, n_mean,Rs_values, n_values = iterative_fit(V_data, I_data, I_0, n_guess, num_iterations=4)
    
    # Step 8: Perform a final curve fit with all parameters (I0, n, Rs)
    def fit_all(V_data, I_data, I0_guess, n_guess, Rs_guess):
        def model_wrapper(V, I_0, n, Rs):
            I_fit, _ = model_current(V, I_0, n, Rs)
            return I_fit
    
        # Define bounds for I0, n, and Rs
        bounds = (
            [np.min([I0_guess * 1e-1,I0_guess * 1e+1]), 0.8 * n_guess, 0.01 * Rs_guess],  # Lower bounds
            [np.max([I0_guess * 1e-1,I0_guess * 1e+1]), 1.2 * n_guess, 100 * Rs_guess]    # Upper bounds
        )
        
        popt, pcov = curve_fit(model_wrapper, V_data, I_data, p0=[I0_guess, n_guess, Rs_guess], bounds=bounds)
        return popt, pcov
    
    # Final curve fit with all parameters
    popt_final, pcov_final = fit_all(V_data, I_data, I_0, n_mean, Rs_mean)
    I0_final, n_final, Rs_final = popt_final
    
    
    # Generate fitted curve using the final optimized parameters
    V_fit = np.linspace(min(V_data), max(V_data), N)  # Voltage range for fitted curve
    I_fit, _ = model_current(V_fit, I0_final, n_final, Rs_final)  # Current values for fitted curve
    return({"I0":I0_final,"n":n_final,"Rs":Rs_final, "Rs_lin":high_voltage_slope,"Vfit":V_fit,"Ifit":I_fit,"Rs_fit":Rs_values,"n_fit":n_values})

