# lumutils/SCENE.py
import numpy as np
from .scene_object import SceneObject
import lumapi as lum
from ..custom_logger import get_custom_logger

logger = get_custom_logger("DMU_SCENE")


from .geometry import *
from .analysis import *
from .data import *
from .helpers import *

class Scene(ObjectsMixin, AnalysisMixin, HelpersMixin):
    """
    Wrapper around a lumapi sim session.
    Owns all SceneObjects and exposes aggregate bounds.
    Mixin classes attach add_nanowire, add_DFT_monitor, etc.
    """
    def __init__(self):
        try:
            sim = lum.connect("FDTD")
        except:
            sim = lum.FDTD(hide=False)
        self.sim     = sim
        self.objects = []
        
        self.FDTD    = self.add_FDTD()
       
    # ------------------------------------------------------------------
    # Object registration (called by mixins after lumapi instantiation)
    # ------------------------------------------------------------------
    def _register(self, scene_obj):
        scene_obj.sim = self.sim
        scene_obj.scene_ref = weakref.ref(self)  # for delete
        self.objects.add(scene_obj)   # set handles duplicates gracefully
        return scene_obj

    def _unregister(self, obj):
        self.objects.discard(obj)   # discard = no error if missing

    # ------------------------------------------------------------------
    # Bound aggregation
    # ------------------------------------------------------------------

    def get_bounds(self, padding=0, include=("geometry",)):
        """
        Aggregate world-space AABB over all registered objects.
        Returns (min (3,), max (3,)).
        """
    
        objects = [o for o in self.objects 
                   if o.kind in include 
                   and not o.exclude_from_bounds]
        if not objects:
            raise ValueError("No matching objects in scene.")
        all_min = np.array([o.obb.aabb_min for o in objects])
        all_max = np.array([o.obb.aabb_max for o in objects])
        return all_min.min(axis=0) - padding, all_max.max(axis=0) + padding

    @property
    def bounds_min(self):
        return self.get_bounds()[0]

    @property
    def bounds_max(self):
        return self.get_bounds()[1]

    @property
    def bounds_center(self):
        mn, mx = self.get_bounds()
        return (mn + mx) / 2

    @property
    def bounds_spans(self):
        mn, mx = self.get_bounds()
        return mx - mn

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_by_name(self, name):
        matches = [o for o in self.objects if o.name == name]
        if not matches:
            raise KeyError(f"No object named '{name}' in scene.")
        return matches if len(matches) > 1 else matches[0]

    def __repr__(self):
        return f"Scene({len(self.objects)} objects)"