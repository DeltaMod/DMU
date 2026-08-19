# lumutils/scene_object.py
import numpy as np
from .geometry.OBB import OBB

class SceneObject:
    """
    Wrapper around a geometry object (Nanowire, RoundedCuboid, etc.)
    Holds world-space position and orientation, and owns the world-space OBB. We must also use this for other primitives we add to the scene normally.
    This function can also be used to modify the sizes of such primitives, but complex structures must be deleted and moved instead.
    """
    def __init__(self, geo, x=0, y=0, z=0, rotx=0, roty=0, rotz=0, xspan=0,yspan=0,zspan=0,
                 pathdict=None, kind="geometry", axis_offset=(0,0,0),
                 exclude_from_bounds=False, max_bounds=None, subobj=None ,permit_resize=True):
        self.geo    = geo
        if not pathdict:
            self.name, self.group, self.fullpath = ["Group",None,"Group"]

        else:
            self.name, self.group, self.fullpath = [pathdict[key] for key in ["name","group","fullpath"]]

        self.kind   = kind   # ["geometry"|"analysis"]
        self.pos    = np.array([x, y, z], dtype=float)
        self.subobj = subobj
        self.spans  = np.array([0,0,0])
        self.minmax  = np.array([[0,0][0,0],[0,0]])
        self.rot    = np.array([rotx, roty, rotz], dtype=float)  # degrees, Euler XYZ
        self.axis_offset = np.array(axis_offset, dtype=float)
        self.obb  = self._compute_obb()
        
    def _check_selection(self):
        """Checks if the object is actually selected, and if not - selects it. (This outputs True if the object is selected, meaning you can technically call it yourself to verify.)""" 
        if not hasattr(self, "sim"):
            raise RuntimeError(f"SceneObject '{self.name}' has no sim reference — was it registered with a Scene?")
    
        selected = self.sim.getAllSelectedObjects()
        if len(selected) == 1 and selected[0]["name"] == self.name: ####WARNING!!! CHECK IF THIS OUTPUT IS FULLPATH, OR JUST THE NAME OF THE STRUCT - 
            return True
        
        self.sim.unselectall()
        self.sim.select(self.fullpath)
        return False
    
    def _compute_obb(self):
        if self.geo is None:
            return OBB(center=self.pos, spans=[0, 0, 0])  # placeholder until geo wrappers exist
        obb = self.geo.get_obb()                    # local space, centred at origin
        obb.translate(self.axis_offset)             # shift pivot in local space
        obb.translate(self.pos)                     # apply world position
        if any(self.rot != 0):
            obb.rotate_euler(*self.rot)
        return obb

    def translate(self, x=None, y=None, z=None, dx=None, dy=None, dz=None):
        """
        Translates the object and updates bounds. 
        Absolute (x/y/z) sets position, relative (dx/dy/dz) shifts it from the current position.
        If both are provided for a single axis, the absolute is applied first, then the relative.
        e.g. translate(x=6, dx=0.5) → pos = 6.5. 
        You only need to provide the dimension you intend to move.
        """
        for i, (abs_val, rel_val) in enumerate(zip((x, y, z), (dx, dy, dz))):
            if abs_val is not None:
                self.pos[i] = abs_val
            if rel_val is not None:
                self.pos[i] += rel_val
        self.obb = self._compute_obb()
        return self
    
    def rotate(self, rotx=0, roty=0, rotz=0):
        """Cumulative rotation in degrees."""
        self.rot += np.array([rotx, roty, rotz])
        self.obb = self._compute_obb()
        return self

    def set_rotation(self, rotx=0, roty=0, rotz=0):
        """Absolute rotation in degrees."""
        self.rot = np.array([rotx, roty, rotz])
        self.obb = self._compute_obb()
        return self
    
    def set_spans(self,xrange=None,yrange=None,zrange=None):
        None
    
    def set_bounds(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        None
    #%% ANALYSIS EXCLUSIVE FUNCTIONS
    def set_monitor_bounds(self, ref_obj, dim="3d", normal="z",
                       padding=0,
                       xpad=None, ypad=None, zpad=None):
        """
        Set bounds of analysis object from a provided reference object's OBB. Note that if you want this to be a primitive, you must have enabled "standalone = True"
        Updates self.obb to reflect the new bounds so Scene.get_bounds() stays as a valid method.
    
        ref_obj : SceneObject providing the spatial reference - can be either geometry or a monitor. !!! This should probably a warning if the object has a zero-bound on one axis.
        dim     : "3d" | "2d"
        normal  : "x"|"y"|"z", collapsed axis for 2d monitors
        padding : scalar, symmetric additive on all non-normal axes - default value = 0
        xpad/ypad/zpad: scalar → [-du,+du], list → [min,max] asymmetric
                        ignored on normal axis
        """
        assert self.kind == "analysis", "set_monitor_bounds requires an analysis SceneObject"
        assert ref_obj is not None, "A reference SceneObject must be provided"
    
        def resolve_pad(pad):
            if pad is None:
                return [0, 0]
            if isinstance(pad, (list, tuple)):
                return list(pad)
            return [-abs(pad), +abs(pad)]
    
        pads = {"x": resolve_pad(xpad),
                "y": resolve_pad(ypad),
                "z": resolve_pad(zpad)}
    
        mn, mx = ref_obj.obb.aabb
        axes   = ["x", "y", "z"]
        bounds = {}
    
        for i, ax in enumerate(axes):
            if dim == "2d" and ax == normal:
                centre     = (mn[i] + mx[i]) / 2
                bounds[ax] = [centre, centre]
            else:
                bounds[ax] = [
                    mn[i] - padding + pads[ax][0],
                    mx[i] + padding + pads[ax][1]
                ]
    
        # Push to lumapi
        self._check_selection()
        for ax, (lo, hi) in bounds.items():
            self.sim.set(f"{ax} min", lo)
            self.sim.set(f"{ax} max", hi)
    
        # Rebuild own OBB from the new bounds so Scene.get_bounds() stays valid
        center = np.array([(bounds[ax][0] + bounds[ax][1]) / 2 for ax in axes])
        spans  = np.array([ bounds[ax][1] - bounds[ax][0]      for ax in axes])
        self.obb = OBB(center, spans)
    
        return bounds

    def __repr__(self):
        return f"SceneObject(name={self.name}, pos={self.pos}, rot={self.rot}, geo={type(self.geo).__name__})"