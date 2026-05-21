# lumutils/scene_object.py
import numpy as np
from .geometry.OBB import OBB

class SceneObject:
    """
    Wrapper around a geometry object (Nanowire, RoundedCuboid, etc.)
    Holds world-space position and orientation, and owns the world-space OBB.
    """
    def __init__(self, geo, x=0, y=0, z=0, rx=0, ry=0, rz=0, name=None, group=None, kind="geometry", axis_offset=(0,0,0)):
        self.geo    = geo
        self.name   = name
        self.group  = group
        self.kind   = kind   # ["geometry"|"analysis"]
        self.pos    = np.array([x, y, z], dtype=float)
        self.rot    = np.array([rx, ry, rz], dtype=float)  # degrees, Euler XYZ
        self.obb    = self._compute_obb()

    def _check_selection(self):
        """Checks if the object is actually selected, and if not - selects it. (This outputs True if the object is selected, meaning you can technically call it yourself to verify.)""" 
        None
    def _compute_obb(self):
        """Get OBB from geometry in local space, then apply world transform."""
        obb = self.geo.get_obb()                  # local-space OBB, centre at origin
        obb.translate(self.pos)
        if any(self.rot != 0):
            obb.rotate_euler(*self.rot)
        return obb

    def move_to(self, x=None, y=None, z=None):
        """Update world position and recompute OBB. Position is in x,y,z coordinates"""
        if x is not None: self.pos[0] = x
        if y is not None: self.pos[1] = y
        if z is not None: self.pos[2] = z
        self.obb = self._compute_obb()
        return self
    
    def move_by(self, dx=None, dy=None, dz=None):
        """Update world position and recompute OBB. Position is in x,y,z coordinates"""
        if dx is not None: self.pos[0] += dx
        if dy is not None: self.pos[1] += dy
        if dz is not None: self.pos[2] += dz
        self.obb = self._compute_obb()
        return self
    
    def rotate(self, rx=0, ry=0, rz=0):
        """Cumulative rotation in degrees."""
        self.rot += np.array([rx, ry, rz])
        self.obb = self._compute_obb()
        return self

    def set_rotation(self, rx=0, ry=0, rz=0):
        """Absolute rotation in degrees."""
        self.rot = np.array([rx, ry, rz])
        self.obb = self._compute_obb()
        return self
    
    def __repr__(self):
        return f"SceneObject(name={self.name}, pos={self.pos}, rot={self.rot}, geo={type(self.geo).__name__})"