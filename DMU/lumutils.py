# -*- coding: utf-8 -*-
"""
Lumerical functions - How best to generate objects in python.
Every single object function must return a dict which details their x-y-z coordinate, their bounding box normalised to x-y coordinates), object rotation etc. 
We note that the objects cannot be modified FROM this unless lumemrical outputs an object control struct, but we'll see about that
Created on Thu Nov 13 13:36:03 202
@author: vidar
"""
import numpy as np
from scipy import constants
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import lumapi
import os
import numpy as np
#%%

def select_and_set_props(sim, name, propdict):
    """
    If name=None, we assume it's already been selected in the scope, so you can use this after creation without passing a name
    """
    if name:
        sim.select(name)
    for key, item in propdict.items():    
        sim.set(key,item)

def create_groups_from_dict(sim, grouplist):
    """
    Create structure and analysis groups in Lumerical without checking for existence.
    Uses temporary SETTER groups and Lumerical's addtogroup behavior.
    Only maximal (non-subset) paths are used.
    
    grouplist example:
    {
        "structure": ["Geometry", "Geometry::Substrate", "Geometry::SurfaceComponents"],
        "analysis": ["Analysis::DFTMonitors","Analysis::VideoMonitors"]
    }
    """
    
    def filter_maximal_paths(paths):
        """Keep only paths that are not prefixes of other paths"""
        paths_sorted = sorted(paths, key=lambda x: -len(x))  # longest first
        maximal = []
        for p in paths_sorted:
            if not any(other.startswith(p + "::") for other in maximal):
                maximal.append(p)
        return maximal
    
    # --- Structure groups ---
    struct_paths = filter_maximal_paths(grouplist.get("structure", []))
    for path in struct_paths:
        sim.addstructuregroup()
        sim.set("name", "structure_setter")
        sim.addtogroup(path)
        sim.delete()
    
    # --- Analysis groups ---
    analysis_paths = filter_maximal_paths(grouplist.get("analysis", []))
    
    if analysis_paths:
        # Find all unique top-level analysis groups
        top_level_analysis = set(p.split("::")[0] for p in analysis_paths)
        
        # Create each top-level analysis group
        for top_group in top_level_analysis:
            sim.addanalysisgroup()
            sim.set("name", top_group)
        
        # Add all paths under their respective top-level analysis groups
        for path in analysis_paths:
            top_group = path.split("::")[0]
            sim.addanalysisgroup()
            sim.set("name", "analysis_setter")
            sim.addtogroup(path)
            sim.delete()
            
def are_all_dict_values_type(d,ttype=None):
    if isinstance(d, dict):  # If d is a dictionary, check all values
        return all(are_all_dict_values_type(value) for value in d.values())
    elif isinstance(d, list):  # If d is a list, check all elements
        return all(are_all_dict_values_type(value) for value in d)
    else:  # If it's neither a dict nor a list, just check if the value is None
        return d is None


class RoundedCuboid:
    def __init__(self, rx=0, ry=0, rz=0, Dx=1, Dy=1, Dz=1, rx2=None, ry2=None, rz2=None):
        """
        x1,y2,z2 ----> o--------o <- x2,y2,z2
                      /        /|                         xy2
                     /        / |                          ^
        x1,y1,z2 -> o--------o  |                          I
                    |  o-----|  o <- x2,y2,z1     yz1<- |     | ->yz2  (out of cube = xz1, into cube xz2)
                    | /      | /                          I
                    |/       |/                           v
        x1,y1,z1--> o--------o <---- x2,y1,z1            xy1
                    
                    <---Dx--->
        
        Rounding Radius Explanation:
            
        Rounding radius can be different for the normal axis of planes with shared normals. So the zy plane can have different rx values, but not different ry and rz values.
        To make this easier, we will run rounding as a "per exception": rx/ry/rz are FOR ALL, and then you can elect to set rx2/ry2/rz2
        Examples: 
            I want rx1 == rx2: set rx = value, and ignore rx2
            I want rx1 = val, and rx2 = 0: Set rx = value, and rx2=0
            I want rx1 = 0.5 and rx2 = 1, you just set rx=0.5 and rx2=1.  
            
        This way, we can also remove planes completely for FLAT faces (if you want to make plus symbols or something)
        
        note that x1 = -1, anx x2 = +1 and vice versa!        

        Dx/Dy/Dz Explanation:
        This represents the TOTAL bounds of the object. This means that the rounding radius of any corner cannot exceed D/2, since then 2r == D
        If greater, then the sphere will clip out of the opposite side. Note, this also means that rx1 can be small, while rx2 can be D/2.
        
        We build objects by doing:
            rx1 -> Dxp -> rx2 (where Dxp = Dx-(rx1+rx2))
            This naturally also means that rx1+rx2 !> Dx
            But you can have: Dx = 100, rx1 = 5, Dxp = 45, rx2 = 50 and the maths will still work out! 
        """
        
        #Fix rx/ry/rz overlap, maximum permitted is D/2. After this, you might as well make a custom shape from spheres instead.
        
        #Assign new r2 values if not assigned, then fix the r_overlap. We will do a rudimentary check to see if rounding can be ratiod, since if r1 = 1.5r2, then r1 should equal D/2, and r2 = D/(2*1.5) 
        r1_list  = [rx,  ry,  rz]
        r2_list  = [rx2, ry2, rz2]
        D_list   = [Dx,  Dy,  Dz]
        self.rr = {}
        for i, axis in enumerate("xyz"):
            self.rr[axis] = list(self._fix_overlap_and_r2(r1_list[i], r2_list[i], D_list[i]))
        
        self.base_corners = {}
        self.base_edges   = {}
        
        self.corner_props = {}
        self.edge_props = {}
        self.core_props = {}
        
        self.Dx = Dx; self.Dy = Dy; self.Dz = Dz
        
        #Get base locations of all corners, generating the keys for future use
        self._get_base_corners()
        #Get real locations of all corners and rounding radiuses.
        self._get_corner_props()
        
        #Get the base keys for edges, and generate keys for future use. Remember to use frozenset() when calling edge keys.
        self._get_base_edges() 
        
        self._get_edge_props()
        
        self._get_core_props()
        
        self._generate_primitives()

    def _fix_overlap_and_r2(self, r1, r2, D):
        # 1. assign r2 if missing
        if r2 is None:
            r2 = r1
        r_lim = D / 2
    
        # 2. if nothing exceeds the limit, return unchanged
        if r1 <= r_lim and r2 <= r_lim:
            return r1, r2
    
        # 3. ratio k = min / max
        radii = [r1, r2]
        rmin = min(radii); rmax = max(radii)
        k = rmin / rmax if rmax != 0 else 0   # safe even if both 0
    
        # 4. index positions of min and max in original order
        idx_min = radii.index(rmin); idx_max = radii.index(rmax)
    
        # 5. build new radii in correct original order
        new = [None, None]
        new[idx_max] = r_lim
        new[idx_min] = k * r_lim
    
        return new[0], new[1]
    
    #CODE FOR BASE PROPERTIES, THESE WILL ALWYS BE THE SAME FOR ALL RESULTS!
    #~~~~~~~~~~~~~~~~ START ~~~~~~~~~~~~~~~~#
    
    def _get_base_corners(self):
        for xi,x_coord in enumerate((-1, 1)):
            for yi,y_coord in enumerate((-1, 1)):
                for zi,z_coord in enumerate((-1, 1)):
                    self.base_corners[(x_coord, y_coord, z_coord)] = {"rr":{"rx": self.rr["x"][xi], "ry": self.rr["y"][yi], "rz": self.rr["z"][zi]},
                                                                       "loc":{"x":x_coord,"y":y_coord,"z":z_coord}}
    def _get_base_edges(self):
        corners = list(self.corner_props.keys())
        edges = []
        
        for i, c1 in enumerate(corners):
            for j in range(i+1, len(corners)):
                c2 = corners[j]
                diffs = sum(a != b for a,b in zip(c1, c2))
                if diffs == 1:
                    edges.append((c1, c2))
        self.base_edges = edges
    #~~~~~~~~~~~~~~~~ END ~~~~~~~~~~~~~~~~#
    
    #CODE FOR REAL CORNER, EDGE AND CORE PROPS#
    #~~~~~~~~~~~~~~~~ START ~~~~~~~~~~~~~~~~#
    def _get_corner_props(self):
        for xi,yi,zi in self.base_corners.keys():
            
            x = xi*self.Dx/2 - xi*self.base_corners[(xi,yi,zi)]["rr"]["rx"]
            y = yi*self.Dy/2 - yi*self.base_corners[(xi,yi,zi)]["rr"]["ry"]
            z = zi*self.Dz/2 - zi*self.base_corners[(xi,yi,zi)]["rr"]["rz"]
            
            
            self.corner_props[(xi,yi,zi)] = {"rr":{"x": self.base_corners[(xi,yi,zi)]["rr"]["rx"], "y": self.base_corners[(xi,yi,zi)]["rr"]["ry"], "z": self.base_corners[(xi,yi,zi)]["rr"]["rz"]},             
                                             "loc":{"x":x,"y":y,"z":z}}

    def _get_edge_props(self):
        self.edge_props = {}
    
        for c1, c2 in self.base_edges:
            loc1 = self.corner_props[c1]["loc"].copy()
            loc2 = self.corner_props[c2]["loc"].copy()
    
            # Determine the edge axis (where the corner coordinates differ)
            diffs = [a != b for a, b in zip(c1, c2)]
            normal_axis = ["x", "y", "z"][diffs.index(True)]
    
            # Get the in-plane radii from one of the corners (they should be equal)
            rr_corner = self.corner_props[c1]["rr"].copy()
            rr_edge = {}
            for axis in "xyz":
                if axis == normal_axis:
                    rr_edge[axis] = None
                else:
                    rr_edge[axis] = rr_corner[axis]
    
            # Store in edge_props
            self.edge_props[(c1, c2)] = {
                "loc": [[loc1["x"], loc1["y"], loc1["z"]],
                        [loc2["x"], loc2["y"], loc2["z"]]],
                "rr": rr_edge}
    
    def _get_core_props(self):
        #We assume that each rounded cube has 3 core cubes, and these will lie along the x/y/z normals. 
        #This is actually trivial, since all cubes are extruded from the core cube that we can extract from all corner radii. 
        dim_mod = {"x":{"val":self.Dx,"ind":0},"y":{"val":self.Dy,"ind":1},"z":{"val":self.Dz,"ind":2}}
        for norm in ["x","y","z"]:
            self.core_props[norm] = {}
            for corner,val in self.corner_props.items():                
                self.core_props[norm][corner] = {"loc":val["loc"].copy()}
                
                self.core_props[norm][corner]["loc"][norm] = corner[dim_mod[norm]["ind"]]*dim_mod[norm]["val"]/2
               
    #~~~~~~~~~~~~~~~~ END ~~~~~~~~~~~~~~~~#
    
    def _generate_primitives(self):
        self.r_spheres = []
        self.r_cylinders = []
        self.c_cubes = []
        
        
        # ----------- ROUNDING SPHERES -----------
         
        for corner, props in self.corner_props.items():
            rr = props["rr"]
            if any(rr[ax] == 0 for ax in "xyz"):
                continue
            entry = {"loc": props["loc"].copy(), "radius": rr.copy()}
            if entry not in self.r_spheres:
                self.r_spheres.append(entry)
        
        
       # ----------- CYLINDERS -----------
        for edge, props in self.edge_props.items():
            loc0, loc1 = props["loc"]
            rr = props["rr"]
        
            # skip zero-length edges
            if loc0 == loc1:
                continue
        
            # find cylinder axis (where radius is None)
            norm = next(ax for ax, v in rr.items() if v is None)
        
            # skip if any non-axis radius is zero
            if any(rr[ax] == 0 for ax in "xyz" if ax != norm):
                continue
        
            # compute cylinder center
            center = {ax: 0.5 * (loc0[i] + loc1[i]) for i, ax in enumerate("xyz")}
        
            # compute cylinder height along norm axis
            height = abs(loc1["xyz".index(norm)] - loc0["xyz".index(norm)])
        
            # perpendicular radii
            radius = {ax: rr[ax] for ax in "xyz" if ax != norm}
        
            entry = {
                "norm": norm,
                "loc": center,
                "radius": radius,
                "height": height
            }
        
            # avoid duplicates
            if entry not in self.r_cylinders:
                self.r_cylinders.append(entry)
    
        # ----------- CORE CUBES -----------
        for norm, cubes in self.core_props.items():
            cube_range = {}
            degenerate = False
            
            for ax in "xyz":
                # grab all loc[ax] for all corners in this norm
                vals = [props["loc"][ax] for props in cubes.values()]
                lo, hi = min(vals), max(vals)
                
                if lo == hi:
                    degenerate = True
                    break
                
                cube_range[f"{ax}min"] = lo
                cube_range[f"{ax}max"] = hi
        
            if not degenerate:
                self.c_cubes.append({"range": cube_range})

                
    def __repr__(self):
        return f"RoundedCuboidRadii({self.corner_props})"


class Nanowire:
    def __init__(self,
        radius,
        length,
        shape="circle",
        endcaps="both",          # "none", "top", "bottom", "both"
        cap_factor=0.5,          # squash factor for z-radius
        seed="none",             # "none", "top", "bottom", "both"
        seed_rfactor = 0.1,      # Percentage of radius to offset particle by. 0.1 = 0.1r, -0.1 = -0.1r
        seed_z_offset=0.1,):
        
        self.radius   = radius
        self.length   = length
                
        self.endcaps = endcaps
        self.cap_factor = cap_factor
        
        self.seed = seed
        self.seed_rfactor = seed_rfactor
        self.shape = shape
        
        # Storage (mirrors your RoundedCuboid structure)
        self.core_cylinder = []
        self.endcaps_list = []
        self.seed_list = []
        
        # Build geometry
        self._make_cylinder()
        self._make_endcaps()
        self._make_seeds()
    
    # -----------------------------------------------------------
    #  MAIN CYLINDER (your cylinder dict format)
    # -----------------------------------------------------------
    def _make_cylinder(self):
        cyl = {
            "loc": {"x": 0, "y": 0, "z": self.length/2},
            "radius": {"x": self.radius, "y": self.radius, "z": self.length/2},
            "range": {"zmin": -self.length/2, "zmax": self.length/2},
            "norm": [0, 0, 1] #Technically, we cannot choose this at all...
        }
        self.core_cylinder.append(cyl)
    
    # -----------------------------------------------------------
    #  ENDCAPS (deformed spheres) Note that this is only supported for shape=circle!
    # -----------------------------------------------------------
    def _make_endcaps(self):

        if self.endcaps not in ("bottom", "top", "both") or self.shape!="circle":
            return
        
        r = self.radius
        rz = r * self.cap_factor
        
        # Bottom
        if self.endcaps in ("bottom", "both"):
            cap = {
                "loc": {"x": 0, "y": 0, "z": -self.length/2},
                "radius": {"x": r, "y": r, "z": rz}
            }
            self.endcaps_list.append(cap)
        
        # Top
        if self.endcaps in ("top", "both"):
            cap = {
                "loc": {"x": 0, "y": 0, "z": self.length/2},
                "radius": {"x": r, "y": r, "z": rz}
            }
            self.endcaps_list.append(cap)
    
    
    # -----------------------------------------------------------
    #  SEEDS (small spheres with different material)
    # We need to make both, but we set the z-order of the "air" sphere to be at the bottom of everything.
    # -----------------------------------------------------------
    def _make_seeds(self):
        if self.seed not in ("bottom", "top", "both"):
            return
        
        seed_rz = self.radius * self.seed_rfactor
        
        # Bottom seed
        if self.seed in ("bottom", "both"):
            s = {
                "loc": {"x": 0, "y": 0, "z": -self.length/2 - self.seed_rfactor*self.radius},
                "radius": {"x": seed_rz, "y": seed_rz, "z": seed_rz}
            }
            self.seed_list.append(s)
        
        # Top seed
        if self.seed in ("top", "both"):
            s = {
                "loc": {"x": 0, "y": 0, "z": self.length/2 + self.seed_rfactor*self.radius},
                "radius": {"x": seed_rz, "y": seed_rz, "z": seed_rz},
            }
            self.seed_list.append(s)

def coordinate_standardisation(method = "span", x=None,y=None,z=None,Dx=None,Dy=None,Dz=None,rx=None,ry=None,rz=None,xmm=None,ymm=None,zmm=None):
    """
    This function works by taking a limited set of co-ordinates, and outputting the full set.
    This means that if you provide: x,y,z and Dx,Dy,Rz, it will give you xyz = [x,y,z], Dxyz = [Dx,Dy,Dz], rxyz = [rx,ry,rz], mmxyz = [xmm,ymm,zmm]
    How does it solve redundancy? Well, if you give it Dx and xmm, these will compete for space.
    The solution is to always prefer Dx/rx outputs over min-max, unless not provided...
    
    If the MODE is set to "span", then we ignore mmxyz until the end, and calculate Dx/rx from whichever is not None. 
    If both are provided, then we will raise an error - but continue as intended. Since if a sphere, rx will be used instead of Dx unless it doesn't exist, at which point it will be calculated 
    if the MODE is set to "minmax", then we calculate Dx from mm
    
    What this func needs to cover:
        x = None, and xmm is not -> calculate x from xmm
        x = value, and xmm range does not match -> recalculate x from xmm
    """    
    xyz    = [x,y,z]
    Dxyz   = [Dx,Dy,Dz]
    rxyz   = [rx,ry,rz] 
    mmxyz  = [xmm,ymm,zmm]
    ## First check if this is a default generation, and produce defaults in that case.
    for i,vals in enumerate(zip(xyz,Dxyz,rxyz,mmxyz)):
        coord, D, r, mm = vals
        if all(a == None for a in [coord,D,r,mm]):
            xyz[i] = 0
            Dxyz[i] = 1e-6
            rxyz[i]  = Dxyz[i]/2 
            mmxyz[i] = [-rxyz[i],rxyz[i]]
        
        if method == "span":
            #Check if all span entries are none
            if all (a == None for a in [D,r]):
                #use min-max fallback, if it exists EVEN if mode is set to span.
                if mmxyz[i] != None:
                    Dxyz[i] = mmxyz[i][1] - mmxyz[i][0]; rxyz[i] = Dxyz[i]/2
                    xyz[i] = np.mean(mmxyz[i])
                #Else, set defaults for all values!
                else:
                    Dxyz[i] = 1e-6; rxyz[i] = Dxyz[i]/2; 
                    if coord == None:
                        xyz[i] = 0
                    mmxyz[i] = [xyz[i]-rxyz[i],xyz[i]+rxyz[i]]
            else:        
                #Since not all values are none, we check first if r is none, and then alter r and D accordingly.
                if r == None: rxyz[i] = D/2
                if D == None: Dxyz[i] = r*2
                if coord == None: xyz[i] = 0
                #And now we alter the minmax values to match the mode
                mmxyz[i] = [xyz[i]-rxyz[i],xyz[i]+rxyz[i]]
                
        if method == "minmax":
            # Check if min-max is none
            if mm == None:
                # attempt to use span fallback
                if not all(a == None for a in [D, r]):
                    # prefer rx, since this is closer analogue to minmax and will work on spheres too
                    if r == None: rxyz[i] = D/2
                    if D == None: Dxyz[i] = r*2
                    if coord == None: xyz[i] = 0

                    mmxyz[i] = [xyz[i] - rxyz[i], xyz[i] + rxyz[i]]

                # Else, set defaults
                else:
                    Dxyz[i] = 1e-6; rxyz[i] = Dxyz[i]/2; 
                    if coord == None:
                        xyz[i] = 0
                    mmxyz[i] = [xyz[i]-rxyz[i],xyz[i]+rxyz[i]]

            else:
                # min-max single-handedly otherwise governs all parameter spaces
                mmxyz[i] = mm
                xyz[i]  = np.mean(mm)
                Dxyz[i] = mm[1] - mm[0]
                rxyz[i] = Dxyz[i] / 2
    return(xyz,Dxyz,rxyz,mmxyz)
                
        
    
def L_primitive(sim,primitive="rect", method="span",x=0,y=0,z=0, Dx=None,rx=None,Dy=None,ry=None,Dz=None,rz=None,norm="z",xminmax = [0,0], yminmax=[0,0],zminmax = [0,0],material=None,zorder=0):
    
    #First we handle the alias case, since we don't want to run the function with illegitimate values.
    #In this case, we want all primitives to have the same reasonable dimension of 1e-6 (where Dx = 2rx)
    xyz,D,r,mm = coordinate_standardisation(method = method, x=x,y=y,z=z,Dx=Dx,Dy=Dy,Dz=Dz,rx=rx,ry=ry,rz=rz,xmm=xminmax,ymm=yminmax,zmm=zminmax)
    axstr = ["x","y","z"]
        
    if primitive == "sphere":
        
        name = sim.addsphere()
        for i,axs in enumerate(axstr):
            sim.setnamed(name, axs, xyz[i])
            sim.setnamed(name, "radius "+axs, r[i])

        if material: sim.setnamed(name, "material", material)
        sim.setnamed(name, "z order", zorder)
    
    if primitive == "rect":
        name = sim.addsphere()
        for i,axs in enumerate(axstr):
            sim.setnamed(name, axs, xyz[i])
            sim.setnamed(name, axs+" span", D[i])
        if material: sim.setnamed(name, "material", material)
        sim.setnamed(name, "z order", zorder)
    
    if primitive == "cylinder":
        name = sim.addcylinder()
        # Default orientation: along z
        # axis_map defines which axis is the primary cylinder axis
        axis_map = {"x": "X", "y": "Y", "z": "Z"}
        rot_map  = {"x": [0, 1, 0], "y": [1, 0, 0], "z": [0, 0, 1]}
        
        primary_axis = axis_map.get(norm.lower(), "Z")  # fallback to z if unknown
        rotation     = rot_map.get(norm.lower(), [0, 0, 1])
        
        # Set cylinder center
        for i, axs in enumerate(["x", "y", "z"]):
            sim.setnamed(name, axs, xyz[i])
        
        # Set cylinder radius and height
        # Convention: radius = r perpendicular to axis, height = D along axis
        # Map: primary axis gets D, other axes use radius
        for i, axs in enumerate(["x", "y", "z"]):
            if axs.lower() == norm.lower():
                sim.setnamed(name, "height " + axs, D[i])
            else:
                sim.setnamed(name, "radius " + axs, r[i])
        
        # Set rotation
        sim.setnamed(name, "primary axis", primary_axis)
        sim.setnamed(name, "rotation", rotation)
        
        if material: sim.setnamed(name, "material", material)
        sim.setnamed(name, "z order", zorder)


def get_minmax_items(xmm,ymm,zmm):
    return([["x","y","z"],["min","max"],[xmm,ymm,zmm]])

def rot_matrix(xr, yr, zr):
    cx, sx = np.cos(np.radians(xr)), np.sin(np.radians(xr))
    cy, sy = np.cos(np.radians(yr)), np.sin(np.radians(yr))
    cz, sz = np.cos(np.radians(zr)), np.sin(np.radians(zr))

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx,  cx]])

    Ry = np.array([[ cy, 0, sy],
                   [  0, 1,  0],
                   [-sy, 0, cy]])

    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [ 0,   0, 1]])

    # Lumerical uses Rz * Ry * Rx
    return Rz @ Ry @ Rx

def aabb_of_rotated_cylinder(center, rx, ry, length, xr, yr, zr):
    cx, cy, cz = center
    h = length / 2

    # Local extreme points
    pts = np.array([
        [ rx,  0,  0],
        [-rx,  0,  0],
        [  0, ry,  0],
        [  0,-ry,  0],
        [  0,  0,  h],
        [  0,  0, -h],
    ])

    R = rot_matrix(xr, yr, zr)

    # Rotate + translate
    world_pts = (R @ pts.T).T + np.array(center)

    xmin, ymin, zmin = world_pts.min(axis=0)
    xmax, ymax, zmax = world_pts.max(axis=0)

    return({"rng":{"x":[xmin, xmax], "y":[ymin, ymax], "z":[zmin, zmax]}})

def quick_sphere(sim, xyz, rx, ry, rz, zorder=0, material=None,name=None):
    sim.addsphere()
    if name:
        sim.set("name",name)
    x,y,z = xyz
    sim.set("x", x)
    sim.set("y", y)
    sim.set("z", z)
    sim.set("radius", rx)
    sim.set("radius 2", ry)
    sim.set("radius 3", rz)
    if material: sim.set("material", material)
    sim.set("override mesh order from material database",1); 
    sim.set("mesh order",zorder);
    

def quick_cuboid(sim, xyz, xmm,ymm,zmm, zorder=0, material=None,name=None):
    sim.addrect()
    if name:
        sim.set("name",name)
    axnorm,mim,xyzmm = get_minmax_items(xmm,ymm,zmm) 
    for i,norm in enumerate(axnorm):
        for ii,mm in enumerate(mim):
            sim.set(norm, xyz[i])
            sim.set(" ".join([norm,mm]),xyzmm[i][ii])

    if material: sim.set("material", material)
    sim.set("override mesh order from material database",1); 
    sim.set("mesh order",zorder);
    

def quick_cylinder(sim, xyz,rx=1e-6,ry=1e-6,L=1e-6, xr=0,yr=0,zr=0, zorder=0, normal="z", material=None,name=None):
    sim.addcircle() #note that the circle is always instaned in the xy plane, normal to the z axis
    if name:
        sim.set("name",name)
    sim.set("first axis","x")
    sim.set("second axis","y")
    sim.set("third axis","z")
    if normal == "x":
        yr += 90
        
    if normal == "y":
        xr += 90
    sim.set("make ellipsoid",1)
    sim.set("radius",rx)
    sim.set("radius 2",ry)  
    sim.set("z span",L)
    sim.set("rotation 1",xr)
    sim.set("rotation 2",yr)
    sim.set("rotation 3",zr)     
    
    if material: sim.set("material", material)
    sim.set("override mesh order from material database",1); 
    sim.set("mesh order",zorder);
    
def L_primitive_old(sim,primitive="rect",xyz=None, x=0,y=0,z=0, Dx=1e-6, Dy=1e-6, Dz=1e-6, xmm = None, ymm=None,zmm = None,
                rx=1e-6, ry=1e-6, rz=1e-6, L=1e-6,xr=0,yr=0,zr=0,normal="z", material=None, zorder=0,bounds=None,name=None,group=None):
    if not xyz:
        xyz = (x,y,z)
    
    if primitive == "sphere":
        quick_sphere(sim,xyz,rx,ry,rz,zorder=zorder,material=material)
        xyz,minmax = radius_to_minmax(xyz, rx, ry, rz)
        adict = range_dict(xyz, *minmax)
        
    if primitive == "rect":
        if all(val == None for val in [xmm,ymm,zmm]):
            xyz,minmax = span_to_minmax(xyz, Dx,Dy,Dz)
        else:
            
            xyz,minmax = span_to_minmax(xyz, xmm,ymm,zmm)
        
        
        quick_cuboid(sim,xyz,*minmax)
        adict = range_dict(xyz, *minmax)
    if primitive == "cylinder":
        quick_cylinder(sim,xyz,rx=rx,ry=ry,L=L, xr=xr,yr=yr,zr=zr, zorder=zorder, normal=normal, material=material,name=name)
        adict = aabb_of_rotated_cylinder(center=(0,0,0), rx=rx, ry=ry, length=L, xr=xr, yr=yr, zr=zr)     
        
    if group:
        sim.addtogroup(group)
        
    if bounds:
        bounds.append(adict["rng"])
        
def L_nanowire(sim, NW, mat = None, seed_mat = None, zorder=0, axis_offset=(0,0,0),group=None):
    for sphere in NW.seed_list:
        sim.addsphere()
        
            
def L_roundedcube(sim, RC, material=None, zorder=0, axis_offset=(0,0,0), group=None,bounds=None):
    """
    Instantiate a RoundedCuboid in a Lumerical simulation using lumapi.
    
    sim: lumapi simulation object (e.g. FDTD(), varFDTD(), CHARGE(), etc.)
    RC: RoundedCuboid instance
    material: Lumerical material string
    zorder: drawing order
    axis_offset: (dx, dy, dz)
    group: name of group to insert objects into (string)
    """
    aox, aoy, aoz = axis_offset

    # -------------------------------------------------------------
    # CORE CUBES
    # -------------------------------------------------------------
    for cube in RC.c_cubes:
        cube_range = cube["range"]
        x = (cube_range["xmin"] + cube_range["xmax"]) / 2 + aox
        y = (cube_range["ymin"] + cube_range["ymax"]) / 2 + aoy
        z = (cube_range["zmin"] + cube_range["zmax"]) / 2 + aoz
        Dx = cube_range["xmax"] - cube_range["xmin"]
        Dy = cube_range["ymax"] - cube_range["ymin"]
        Dz = cube_range["zmax"] - cube_range["zmin"]

        L_primitive(sim,primitive="rect",xyz=(x,y,z), Dx=Dx, Dy=Dy, Dz=Dz,material=material,zorder=zorder,bounds=bounds,group=group)        

    # -------------------------------------------------------------
    # SPHERES
    # -------------------------------------------------------------
    for sphere in RC.r_spheres:
        loc = sphere["loc"]
        radius = sphere["radius"]

        x = loc["x"] + aox
        y = loc["y"] + aoy
        z = loc["z"] + aoz

        rx, ry, rz = radius["x"], radius["y"], radius["z"]
        L_primitive(sim,primitive="sphere",xyz=(x,y,z), rx=rx, ry=ry, rz=rz,material=material,zorder=zorder,bounds=bounds,group=group)      
        

    # -------------------------------------------------------------
    # CYLINDERS
    # -------------------------------------------------------------
    for cyl in RC.r_cylinders:
        loc = cyl["loc"]
        radius = cyl["radius"]
        axis_range = cyl["range"]
        norm = cyl["norm"]

        x = loc["x"] + aox
        y = loc["y"] + aoy
        z = loc["z"] + aoz

        rx = radius.get("x", 0)
        ry = radius.get("y", 0)
        rz = radius.get("z", 0)

        name = sim.addobject("cylinder")
        sim.setnamed(name, "x", x)
        sim.setnamed(name, "y", y)
        sim.setnamed(name, "z", z)

        sim.setnamed(name, "radius x", rx)
        sim.setnamed(name, "radius y", ry)
        sim.setnamed(name, "radius z", rz)

        sim.setnamed(name, "axis", norm)
        sim.setnamed(name, "z min", axis_range["zmin"])
        sim.setnamed(name, "z max", axis_range["zmax"])

        if material: sim.setnamed(name, "material", material)
        sim.setnamed(name, "z order", zorder)

        if group:
            sim.addtogroup(group)

def radius_to_minmax_oneax(loc, rad):
    """Returns [min,max] of a span, or [min,max] if span is already [min,max]"""
    if isinstance(rad, (list, tuple)):
        if len(rad) == 1:
            rad = rad[0]
        elif len(rad) == 2:
            return rad
    return(loc, [loc - rad, loc + rad])

def radius_to_minmax(xyz, rx,ry,rz):
    radii = [rx,ry,rz]
    minmax    = []
    for i,loc in enumerate(xyz):
        locnew,mm = radius_to_minmax_oneax(loc,radii[i])
        minmax.append(mm)
    return(xyz,minmax)


def span_to_minmax_oneax(loc, span):
    """Returns [min,max] of a span, or [min,max] if span is already [min,max]"""
    if isinstance(span, (list, tuple)):
        if len(span) == 1:
            span = span[0]
        elif len(span) == 2:
            locnew = np.mean(span)
            return(locnew,span)
    mm = [loc - span/2, loc + span/2]
    
    return(loc,mm)

def span_to_minmax(xyz,Dx,Dy,Dz):
    spans = [Dx,Dy,Dz]
    xyzn      = []
    minmax    = []
    for i,loc in enumerate(xyz):
        locnew,mm = span_to_minmax_oneax(loc,spans[i])
        xyzn.append(locnew)
        minmax.append(mm)
    
    return(xyzn,minmax)
    

def range_dict(xyz,xrange,yrange,zrange):
    adict = dict(loc={},rng = {"x":xrange,"y":yrange,"z":zrange}) 
    for i,dim in enumerate(["x","y","z"]):
        loc            = xyz[i]

        adict["loc"][dim], adict["rng"][dim] = span_to_minmax_oneax(loc, adict["rng"][dim]) 
    return(adict)


def determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=True):
    adict = range_dict(xyz, xrange, yrange, zrange)
    
    # Determine which dimensions have non-zero span
    spans = {dim: adict["rng"][dim][1] - adict["rng"][dim][0] for dim in ["x","y","z"]}
    nonzero = [d for d in spans if spans[d] > 0]
    zero    = [d for d in spans if spans[d] == 0]

    # Determine 2D vs 3D
    if len(nonzero) not in [3,2]:
        raise ValueError(f"Invalid span combination: zero spans in {zero}. Provide either all 3 spans (3D) or exactly one zero span (2D).")
    if len(nonzero) == 3 and not allow3D:
        raise ValueError("Attempted to set 3 dimensions when only two are allowed. Please provide exactly one zero span (2D).")
    
    if len(nonzero) == 3:
        monitor_type = "3D"
        
    elif len(nonzero) == 2 and len(zero) == 1:
        monitor_type = "2D"
        for key,item in adict["rng"].items():
            if min(item) == max(item):
                normal = key
        monitor_type = "2D "+normal.upper()+"-normal"
    return(adict,monitor_type,[normal])
        
def L_addDFT(sim, name, group=None, xyz=(0,0,0), xrange=1, yrange=1, zrange=0,prop_dict={},bounds=None):
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
    adict,monitor_type,skip = determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=True)
    
    # Create the DFT monitor
    sim.adddftmonitor()
    sim.set("name",name)
    sim.addtogroup(group)
    sim.set("monitor type",monitor_type)
    
    for key,item in adict["loc"].items():
            sim.set(key,item)
            
    for key,item in adict["rng"].items():
        if key not in skip:
            sim.set(key+" min",item[0])
            sim.set(key+" max",item[1])
    select_and_set_props(sim, None, prop_dict)    
    if bounds:
        bounds.append(adict["rng"])
                
def L_addmovie(sim, name, group=None, xyz=(0,0,0), xrange=1, yrange=1, zrange=0,prop_dict={},bounds=None):
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
    adict,monitor_type,skip = determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=False)
    
    # Create the movie monitor
    sim.addmovie()
    sim.set("name",name)
    sim.addtogroup(group)
    
    sim.set("monitor type",monitor_type)
    
    for key,item in adict["loc"].items():
            sim.set(key,item)
            
    for key,item in adict["rng"].items():
        if key not in skip:
            sim.set(key+" min",item[0])
            sim.set(key+" max",item[1])
    select_and_set_props(sim, None, prop_dict)        
    if bounds:
        bounds.append(adict["rng"])
     

def L_addanalysis(sim, atype, name, group=None, xyz=(0,0,0), xrange=1, yrange=1, zrange=0,prop_dict={},bounds = None):
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
    adict,monitor_type,skip = determine_2D_3D_spans_normals(xyz,xrange,yrange,zrange,allow3D=False)
    
    # Create the movie monitor
    sim.addmovie()
    sim.set("name",name)
    sim.addtogroup(group)
    
    sim.set("monitor type",monitor_type)
    
    for key,item in adict["loc"].items():
            sim.set(key,item)
            
    for key,item in adict["rng"].items():
        if key not in skip:
            sim.set(key+" min",item[0])
            sim.set(key+" max",item[1])
    
    select_and_set_props(sim, None, prop_dict)
    
    if bounds:
        bounds.append(adict["rng"])


def get_material_list(sim,substr=None,legacy=False):
    if not sim:
        legacy=True
        
    if legacy:
        Matlist = [ "Ag (Silver) - Johnson and Christy",
                    "Al (Aluminium) - CRC",
                    "Ag (Silver) - Palik (0-2um)",
                    "W (Tungsten) - Palik",
                    "InAs - Palik",
                    "TiO2 (Titanium Dioxide) - Sarkar",
                    "Cr (Chromium) - CRC",
                    "5CB - Li",
                    "Ge (Germanium) - Palik",
                    "Si3N4 (Silicon Nitride) - Luke",
                    "5PCH - Li",
                    "TiO2 (Titanium Dioxide) - Kischkat,"
                    "Ag (Silver) - CRC",
                    "PEC (Perfect Electrical Conductor),"
                    "Al2O3 - Palik",
                    "Al (Aluminium) - Palik",
                    "E44 - Li",
                    "W (Tungsten) - CRC",
                    "TiO2 (Titanium Dioxide) - Devore",
                    "Si3N4 (Silicon Nitride) - Phillip",
                    "Sn (Tin) - Palik",
                    "Cr (Chromium) - Palik",
                    "Fe (Iron) - Palik",
                    "MLC-6608 - Li",
                    "C (graphene) - Falkovsky (mid-IR)",
                    "Pt (Platinum) - Palik",
                    "Au (Gold) - CRC",
                    "Si (Silicon) - Palik",
                    "Fe (Iron) - CRC",
                    "SiO2 (Glass) - Palik",
                    "Si3N4 (Silicon Nitride) - Kischkat,"
                    "Pd (Palladium) - Palik",
                    "In (Indium) - Palik",
                    "Ni (Nickel) - CRC",
                    "MLC-9200-100 - Li",
                    "InP - Palik",
                    "MLC-9200-000 - Li",
                    "Ni (Nickel) - Palik",
                    "Cu (Copper) - CRC",
                    "Ge (Germanium) - CRC",
                    "etch",
                    "Au (Gold) - Palik",
                    "Ti (Titanium) - Palik",
                    "TiN - Palik",
                    "TiO2 (Titanium Dioxide) - Siefke",
                    "Ti (Titanium) - CRC",
                    "H2O (Water) - Palik",
                    "Au (Gold) - Johnson and Christy",
                    "GaAs - Palik",
                    "V (Vanadium ) - CRC",
                    "6241-000 - Li",
                    "Cu (Copper) - Palik",
                    "Ta (Tantalum) - CRC",
                    "E7 - Li",
                    "Rh (Rhodium) - Palik",
                    "Ag (Silver) - Palik (1-10um)",
                    "TL-216 - Li"]
    else:
        Matlist = sim.getmaterial().split("\n")
        
    if substr:
        # 1. If substr is a str → convert to list
        if isinstance(substr, str):
            substr_list = [substr]
        else:
            # 2. If it's already a list or tuple → use as-is
            substr_list = list(substr)

        # 3. Apply AND-filter:
        # Keep only materials that contain *all* substrings
        def match_all(mat):
            return all(s in mat for s in substr_list)

        Matlist = [mat for mat in Matlist if match_all(mat)]

    return(Matlist)

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

"""
HELPER SCRIPTS!!!
SCRIPTS BELOW ARE USED TO VERIFY FUNCTIONALITY OF THE CODE ABOVE, SUCH AS MATPLOTLIB PLOTTING OF CUBES, CYLINDER NANOWIRES, AND MORE!
"""
# ---------- CYLINDERS ----------
def check_ax(ax):
    try:
        bbox = ax.bbox
    except Exception as e:
        raise Exception("No axis provided") from e
        
def plot_cylinder(center, radius, height, ax=None, axis="z", resolution=20):
    # axis-aligned cylinder
    check_ax(ax)
        
    u = np.linspace(0, 2 * np.pi, resolution)
    h = np.linspace(-height/2, height/2, 2)
    U, H = np.meshgrid(u, h)
    if axis == "x":
        X = H + center["x"]
        Y = radius["y"] * np.cos(U) + center["y"]
        Z = radius["z"] * np.sin(U) + center["z"]
    elif axis == "y":
        X = radius["x"] * np.cos(U) + center["x"]
        Y = H + center["y"]
        Z = radius["z"] * np.sin(U) + center["z"]
    else:  # z-axis
        X = radius["x"] * np.cos(U) + center["x"]
        Y = radius["y"] * np.sin(U) + center["y"]
        Z = H + center["z"]
    ax.plot_surface(X, Y, Z, color="green", alpha=0.6)
    
def plot_cube(cube,ax=None):
    check_ax(ax)
    rng = cube["range"]
    x = [rng["xmin"], rng["xmax"]]
    y = [rng["ymin"], rng["ymax"]]
    z = [rng["zmin"], rng["zmax"]]

    # Create 2D surfaces for each face
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, z[0]*np.ones_like(X), color="cyan", alpha=0.3)  # bottom
    ax.plot_surface(X, Y, z[1]*np.ones_like(X), color="cyan", alpha=0.3)  # top

    Y, Z = np.meshgrid(y, z)
    ax.plot_surface(x[0]*np.ones_like(Y), Y, Z, color="cyan", alpha=0.3)  # left
    ax.plot_surface(x[1]*np.ones_like(Y), Y, Z, color="cyan", alpha=0.3)  # right

    X, Z = np.meshgrid(x, z)
    ax.plot_surface(X, y[0]*np.ones_like(X), Z, color="cyan", alpha=0.3)  # front
    ax.plot_surface(X, y[1]*np.ones_like(X), Z, color="cyan", alpha=0.3)  # back
    
def plot_sphere(sphere,resolution=20,ax=None):
    check_ax(ax)
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    cx, cy, cz = sphere["loc"]["x"], sphere["loc"]["y"], sphere["loc"]["z"]
    rx, ry, rz = sphere["radius"]["x"], sphere["radius"]["y"], sphere["radius"]["z"]
    X = rx * np.outer(np.cos(u), np.sin(v)) + cx
    Y = ry * np.outer(np.sin(u), np.sin(v)) + cy
    Z = rz * np.outer(np.ones_like(u), np.cos(v)) + cz
    ax.plot_surface(X, Y, Z, color="red", alpha=0.6)
    
def plot_rounded_cuboid(CG,ax=None):
    check_ax(ax)
    for cube in CG.c_cubes:
        plot_cube(cube,ax=ax)

    # ---------- ROUNDING SPHERES ----------
    for sphere in CG.r_spheres:
        plot_sphere(sphere,resolution=20,ax=ax)
        
    for cyl in CG.r_cylinders:
        plot_cylinder(cyl["loc"], cyl["radius"], cyl["height"],ax=ax, axis=cyl["norm"])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1,1,1])
    plt.show()

# --- Example usage ---
helper_functions = False
if helper_functions:
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    A = RoundedCuboid(rx=1, ry=1, rz=1, Dx=6, Dy=6, Dz=6, rx2=2, ry2=2, rz2=2)
    plot_rounded_cuboid(A,ax=ax)


def plot_Nanowire(NWG):       
    None
                                    
