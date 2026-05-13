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
            
    def add_analysis(self, atype, name, group=None, xyz=(0,0,0), xrange=1, yrange=1, zrange=0,prop_dict={},bounds = None):
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
            