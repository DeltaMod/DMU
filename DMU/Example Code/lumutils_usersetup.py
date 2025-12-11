import lumapi as lum
import sys


GHIN=True
if GHIN:
    sys.path.append(r"E:\Vidar Flodgren")
    import DMUlocal.DMU.sem_tools as dmsem
    import DMUlocal.DMU.plot_utils as dmp
    import DMUlocal.DMU.utils as dm
    import DMUlocal.DMU.lumutils as dlu
    
else:
    from DMU import sem_tools as dmsem
    from DMU import plot_utils as dmp
    from DMU import utils as dm
    
try:
    sim = lum.connect("FDTD")
except:
    sim = lum.FDTD(hide=False)


#Initialising the FDTD Volume
bounds = {"analysis":[],"geometry":[]}
material_folder = r'E:\Vidar Flodgren\RedoxMe\Materials'
dlu.import_materials_from_folder(sim,material_folder,mat_type="Sampled 3D data")

mats = dict(Au=dlu.get_material_list(sim,substr=["Au","Palik"],legacy=False)[0],
            TiO2=dlu.get_material_list(sim,substr=["TiO2","Devore"],legacy=False)[0],
            SiO2=dlu.get_material_list(sim,substr=["SiO2","Palik"],legacy=False)[0],
            Ag=dlu.get_material_list(sim,substr=["Ag","Palik"],legacy=False)[0],
            NiO=dlu.get_material_list(sim,substr=["NiO","Dibyashree"],legacy=False)[0])
 
fdtd_props = {
    "x":0,"y":0,"z":0,
    "x span":1e-6,"y span":1e-6,"z span":1e-6,
    "x min":-0.5e-6,"x max":0.5e-6,
    "y min":-0.5e-6,"y max":0.5e-6,
    "z min":-0.5e-6,"z max":0.5e-6,
    "mesh type":"auto non-uniform", 
    "mesh accuracy":3, "dt stability factor": 0.5,
    "mesh refinement":"conformal variant 1",
    "min mesh step": 0.00025,
    "x min bc":"PML","x max bc":"PML",
    "y min bc":"PML","y max bc":"PML",
    "z min bc":"PML","z max bc":"PML",
    "auto shutoff min":1e-5, "auto shutoff max":10000
    }

FDTD = sim.addfdtd()

dlu.select_and_set_props(sim, FDTD.name, fdtd_props)

group_list = dict(structure=["Geometry::Substrate","Geometry::SurfaceComponents"],
                  analysis=["Analysis::DFTMonitors","Analysis::VideoMonitors","Analysis::Pabs","Analysis::Transmission"])
              
dlu.create_groups_from_dict(sim, group_list)


dlu.L_addDFT(sim, "Test",group="Analysis::DFTMonitors",bounds=bounds["analysis"])
