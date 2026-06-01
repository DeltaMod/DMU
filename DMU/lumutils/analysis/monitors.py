# lumutils/scene_object.py
import numpy as np
from ..scene_object import SceneObject
from ..geometry.OBB import OBB
from .. import helpers as hlp

class AnalysisMixin(hlp.HelpersMixin):
    
    def _register_monitor(self, adict, name, group):
        """Build OBB from adict and register as an analysis SceneObject."""
        obb = OBB(
            center=[adict["loc"]["x"], adict["loc"]["y"], adict["loc"]["z"]],
            spans =[adict["rng"]["x"][1] - adict["rng"]["x"][0],
                    adict["rng"]["y"][1] - adict["rng"]["y"][0],
                    adict["rng"]["z"][1] - adict["rng"]["z"][0]]
        )
        return self._register(SceneObject.from_obb(obb, name=name, group=group, kind="analysis"))
    
    def _set_monitor_bounds(self, adict, skip):
        """Push min/max values to sim."""
        for key, item in adict["loc"].items():
            self.sim.set(key, item)
        for key, item in adict["rng"].items():
            if key not in skip:
                self.sim.set(key + " min", item[0])
                self.sim.set(key + " max", item[1])
                
    def add_DFT(self, name, group=None, xyz=(0,0,0), xrange=1, yrange=1, zrange=0,prop_dict={},bounds=None):
        """
        Add a 2D or 3D DFT monitor to the simulation.
        
        Two modes: span or range. You may provide either xrange=5 or xrange=[-2.5,2.5].
        
        Parameters:
            sim       : Lumerical simulation object
            name      : Name of the DFT monitor
            group     : Optional parent group (currently not used)
            xyz       : centre coordinates (tuple)
            xrange    : float or [min,max] for x
            yrange    : float or [min,max] for y
            zrange    : float or [min,max] for z
        """
        xyz = tuple(xyz)
        adict,monitor_type,skip = hlp.determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=True)
        
        # Create the DFT monitor
        self.sim.adddftmonitor()
        self.sim.set("name",name)
        self.sim.addtogroup(group)
        self.sim.set("monitor type",monitor_type)
        self._set_monitor_bounds(adict, skip)
        self.select_and_set_props(None, prop_dict)

        return self._register_monitor(adict, name, group)
                    
    def add_movie(self, name, group=None, xyz=(0,0,0), xrange=1, yrange=1, zrange=0,prop_dict={},bounds=None):
        """
        Add a 2D moviemonitor
        
        Two modes: span or range. You may provide either xrange=5 or xrange=[-2.5,2.5].
        Parameters:
            sim       : Lumerical simulation object
            name      : Name of  monitor
            group     : format: "model::group1::group2"
            xyz       : centre coordinates (tuple) - if using min/max, this can be anything
            xrange    : float or [min,max] for x
            yrange    : float or [min,max] for y
            zrange    : float or [min,max] for z
        """
        xyz = tuple(xyz)
        adict,monitor_type,skip = hlp.determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=False)
        
        # Create the movie monitor
        self.sim.addmovie()
        self.sim.set("name",name)
        self.sim.addtogroup(group) 
        self.sim.set("monitor type",monitor_type)
        
        self._set_monitor_bounds(adict, skip)
        self.select_and_set_props(None, prop_dict)

        return self._register_monitor(adict, name, group)
            
    def add_analysis(self, atype, name, group=None, xyz=(0,0,0), xrange=1, yrange=1, zrange=0,prop_dict={}):
        """
        Add a 2D moviemonitor
        
        Two modes: span or range. You may provide either xrange=5 or xrange=[-2.5,2.5].
        Parameters:
            name      : Name of  monitor
            group     : format: "model::group1::group2"
            xyz       : centre coordinates (tuple) - if using min/max, this can be anything
            xrange    : float or [min,max] for x
            yrange    : float or [min,max] for y
            zrange    : float or [min,max] for z
        """
        xyz = tuple(xyz)
        adict,monitor_type,skip = hlp.determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=False)
        
        # Create the movie monitor
        self.sim.addmovie()
        self.sim.set("name",name)
        self.sim.addtogroup(group)
        
        self.sim.set("monitor type",monitor_type)
        
        self._set_monitor_bounds(adict, skip)
        self.select_and_set_props(None, prop_dict)
        return self._register_monitor(adict, name, group)
    
    def add_FDTD(self,name="FDTD",xyz=(0,0,0),xrange=1e-6,yrange=1e-6,zrange=1e-6,prop_dict=None):
        """
        Add a FDTD simulation object
        
        Two modes: span or range. You may provide either xrange=5 or xrange=[-2.5,2.5].
        Parameters:
            name      : Name of  monitor
            xyz       : centre coordinates (tuple) - if using min/max, this can be anything
            xrange    : float or [min,max] for x
            yrange    : float or [min,max] for y
            zrange    : float or [min,max] for z
            prop_dict : dict object, example dict:
                prop_dict = {
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
        """
        xyz = tuple(xyz)
        adict,monitor_type,skip = hlp.determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=True)