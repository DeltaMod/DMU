# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:53:59 2026

@author: vidar
"""
import numpy as np

from ...custom_logger import get_custom_logger
logger = get_custom_logger("LMU_OBB")

    

class FDTD_SCENE:
    def __init__(self):
        None

class OBB:
    """
    Oriented Bounding Box defined by centre, spans, and local axes.
    All rotations update only the axes matrix; spans remain in local space.
    """

    def __init__(self, center, spans, axes=None):
        """
        Parameters
        ----------
        center: array-like (3,)
        spans : array-like (3,)  spans, so full-lengths along each local axis
        axes  : array-like (3,3) row vectors = local X/Y/Z axes
                  defaults to identity (axis-aligned)
        """
        self.center  = np.asarray(center,  dtype=float)
        self.spans = np.asarray(spans, dtype=float)
        self.axes    = np.asarray(axes, dtype=float) if axes is not None else np.eye(3)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_D(cls, x_span, y_span, z_span, center=None):
        spans_arr = np.array([x_span, y_span, z_span], dtype=float)
        spans = spans_arr[:, 1] - spans_arr[:, 0]  
        if center is None:
            center = spans_arr.mean(axis=1)
        return cls(center, spans)
    
    @classmethod
    def from_points(cls, points):
        pts = np.asarray(points, dtype=float)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = (mn + mx) / 2
        spans  = mx - mn  
        return cls(center, spans)

    @classmethod
    def from_vertices(cls, vertices):
        """Same as from_points — alias for the 8-vertex cube format you had."""
        return cls.from_points(vertices)

    # ------------------------------------------------------------------
    # Transformations (all return self for chaining)
    # ------------------------------------------------------------------

    def rotate(self, R):
        """Apply a (3,3) rotation matrix to the box orientation."""
        R = np.asarray(R, dtype=float)
        self.axes = R @ self.axes
        return self

    def rotate_euler(self, rx=0.0, ry=0.0, rz=0.0, order="xyz", mode="degrees"):
        """
        Rotate by Euler angles (radians).
        Builds R from elemental rotations applied in `order` (e.g. 'xyz').
        """
        if mode == "degrees":
            rx = np.radians(rx)
            ry = np.radians(ry)
            rz = np.radians(rz)
            
        def Rx(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[1,0,0],[0,c,-s],[0,s,c]])
        def Ry(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c,0,s],[0,1,0],[-s,0,c]])
        def Rz(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c,-s,0],[s,c,0],[0,0,1]])

        mats = {"x": Rx(rx), "y": Ry(ry), "z": Rz(rz)}
        R = np.eye(3)
        for axis in order:
            R = mats[axis] @ R
        return self.rotate(R)

    def translate(self, offset):
        """Translate the centre by offset (3,)."""
        self.center = self.center + np.asarray(offset, dtype=float)
        return self

    def scale(self, factor):
        """
        Uniform or per-axis scale.
        factor: scalar or array-like (3,)
        """
        self.spans = self.spans * np.asarray(factor, dtype=float)
        return self

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def vertices(self):
        signs = np.array([[sx,sy,sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], dtype=float)
        return self.center + (signs * (self.spans / 2)) @ self.axes
    
    @property
    def aabb(self):
        half = np.abs((self.spans / 2) @ self.axes)
        return self.center - half, self.center + half
    
    

    @property
    def aabb_min(self):
        return self.aabb[0]

    @property
    def aabb_max(self):
        return self.aabb[1]

    def contains_point(self, point):
        p = np.asarray(point, dtype=float) - self.center
        local = self.axes @ p
        return bool(np.all(np.abs(local) <= self.spans / 2))

    def intersects(self, other):
        """
        SAT-based OBB vs OBB intersection test.
        Returns True if the two boxes overlap.
        """
        t = other.center - self.center
        axes_to_test = []

        # Face normals of both boxes
        for i in range(3):
            axes_to_test.append(self.axes[i])
            axes_to_test.append(other.axes[i])

        # Edge cross products
        for i in range(3):
            for j in range(3):
                cross = np.cross(self.axes[i], other.axes[j])
                if np.linalg.norm(cross) > 1e-10:
                    axes_to_test.append(cross / np.linalg.norm(cross))

        for axis in axes_to_test:
            # Project both boxes onto axis
            proj_self  = sum(abs(np.dot((self.spans[i]  / 2) * self.axes[i],  axis)) for i in range(3))
            proj_other = sum(abs(np.dot((other.spans[i] / 2) * other.axes[i], axis)) for i in range(3))
            dist = abs(np.dot(t, axis))
            if dist > proj_self + proj_other:
                return False  # Separating axis found

        return True

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"OBB(center={self.center}, spans={self.spans},\n"
                f"    axes=\n{self.axes})")