"""
data_handlers manages importing/exporting data.
"""
import numpy as np
from scipy import constants
import os

from ...custom_logger import get_custom_logger
logger = get_custom_logger("LMU_datahandler")

def loaddata(filepath):
    """
    Load a 3-column material file (wl, n, k).
    Accepts comma- or space-delimited formats automatically.
    Returns wl, n, k as numpy arrays.
    """

    # genfromtxt automatically detects commas or whitespace
    data = np.genfromtxt(
        filepath,
        comments="#",
        delimiter=None,      # auto-detect
        dtype=float,
        invalid_raise=False  # tolerate stray commas at end
    )

    # Remove empty rows that load as NaN
    data = data[~np.isnan(data).any(axis=1)]

    if data.shape[1] < 3:
        raise ValueError(f"{filepath} must contain at least 3 columns (wl, n, k).")
    cc = constants.c
    wl = np.array(data[:, 0])
    freq = 2 * np.pi * cc / (wl*1e-6)  # rad/s
    n  = np.array(data[:, 1])
    k  = np.array(data[:, 2])
    nik = n + 1j * k
    data_2col = np.column_stack((freq, nik))
    return wl, n, k,np.array(data_2col)

def import_materials_from_folder(sim,path,extlist=[".csv",".txt"],mat_type="Sampled 3D data",overwrite=True):
    filelist = [file for file in os.listdir(path) if any(ext in file for ext in extlist)]
    for file in filelist:
        fpath = os.path.join(path, file)
        material_name = file.split(".")[0]
        wl,n,k,data = loaddata(fpath)
        
        print(f"Importing material: {material_name}")

        # --- Create a new material in the simulation ---
        materiallist = sim.getmaterial().split("\n")
        if material_name in materiallist and overwrite==False:
            print(f"{material_name} already exists, skipping")
            return
        
        if material_name in materiallist and overwrite==True:
            sim.deletematerial(material_name)
        newmat = sim.addmaterial(mat_type)                   # creates a plain material
        sim.setmaterial(newmat,"name",material_name);
        sim.setmaterial(material_name, mat_type.lower(),data)
        
        print(f"Loaded {material_name} ({len(wl)} points)")