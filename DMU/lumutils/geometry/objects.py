# lumutils/objects/objects.py
import numpy as np
from ..geometry.OBB import OBB
from ..scene_object import SceneObject
from .. import helpers as hlp
from .. import defaults as dflt

from ...custom_logger import get_custom_logger
_S = object()
logger = get_custom_logger("DMU_NANOWIRE")


class Nanowire:
    def __init__(self,
                 radius=0.1e-6,
                 length=3e-6,
                 shape="circle",
                 endcaps="both",
                 cap_factor=0.5,
                 seed="none",
                 seed_rfactor=0.1,
                 seed_z_offset=0.1):
        """
        inputs:
            radius       : Radius of the nanowire, can be symmetric (scalar) or provided as a float (radx,rady) or a dict ["x":1e-7,"y":1e-7]
            length       : Length of the nanowire (Scalar) 
            shape        : For now, only circle is accepted - but square should also be an option later.                      
            endcaps      : "both|top|bottom" 
            cap_factor   : How smushed the z-axis of the spherical endcap is. (Scalar) Default is .5.
            seed         : "bottom|top|both"
            seed_rfactor : describes the shrink ratio of the seed radius. If set to 0.9, then the radius is 90% of the nanowire radius.
            seed_z_offset: range of [-1,1]. This should be a "sink" ratio where -1 is fully inside (-r), and 1 fully outside (r) ("balanced like a ball on top") and 0 is perfectly centered inside the flat part of the nanowire cylinder ends.
        """
        if isinstance(radius, (int, float)):
            self.radius = {"rx": radius, "ry": radius}
        elif isinstance(radius, (tuple, list)):
            self.radius = {"rx": radius[0], "ry": radius[1]}
        elif isinstance(radius, dict):
            self.radius = radius
        else:
            raise TypeError(f"radius must be scalar, tuple, or dict, got {type(radius)}")
    
        self.length       = length
        self.shape        = shape
        self.endcaps      = endcaps
        self.cap_factor   = cap_factor
        self.seed         = seed
        self.seed_rfactor = seed_rfactor
        self.seed_z_offset = seed_z_offset
        self.recalculate()

    def recalculate(self):
        self.core_cylinder = []
        self.endcaps_list  = []
        self.seed_list     = []
        self._make_cylinder()
        self._make_endcaps()
        self._make_seeds()

    def get_obb(self):
        """
        Compute local-space OBB (centre at origin).
        Nanowire is always along Z in local space.
        """
        rx = self.radius["rx"]
        ry = self.radius["ry"]
        cap_ext = max(rx, ry) * self.cap_factor if self.endcaps != "none" else 0
        return OBB(center=[0,0,0], spans=[rx*2, ry*2, self.length + 2*cap_ext])

    # --- geometry builders unchanged from your original ---
    def _make_cylinder(self):
        cyl = {
            "loc":    {"x": 0, "y": 0, "z": self.length / 2},
            "radius": {"rx": self.radius["rx"], "ry": self.radius["ry"], "rz": self.length / 2},
            "range":  {"zmin": -self.length / 2, "zmax": self.length / 2},
            "norm":   "z"
        }
        self.core_cylinder.append(cyl)

    def _make_endcaps(self):
        if self.endcaps not in ("bottom", "top", "both") or self.shape != "circle":
            return
        rz = (self.radius["rx"]+self.radius["ry"])/2 * self.cap_factor
        if self.endcaps in ("bottom", "both"):
            self.endcaps_list.append({"loc": {"x": 0, "y": 0, "z": -self.length / 2},
                                      "radius": {"rx": self.radius["rx"], "ry": self.radius["ry"], "rz": rz}})
        if self.endcaps in ("top", "both"):
            self.endcaps_list.append({"loc": {"x": 0, "y": 0, "z":  self.length / 2},
                                      "radius": {"rx": self.radius["rx"], "ry": self.radius["ry"], "rz": rz}})

    def _make_seeds(self):
        if self.seed not in ("bottom", "top", "both"):
            return
        seed_rx,seed_ry,seed_rz = [self.radius["rx"] * self.seed_rfactor, self.radius["ry"] * self.seed_rfactor,(self.radius["rx"] + self.radius["ry"])/2* self.seed_rfactor]
        if self.seed in ("bottom", "both"):
            self.seed_list.append({"loc": {"x": 0, "y": 0,
                                           "z": -self.length / 2 + self.seed_z_offset * seed_rz},
                                   "radius": {"rx": seed_rx, "ry": seed_ry, "rz": seed_rz}})
        if self.seed in ("top", "both"):
            self.seed_list.append({"loc": {"x": 0, "y": 0,
                                           "z":  self.length / 2 + self.seed_z_offset * seed_rz},
                                   "radius": {"rx": seed_rx, "ry": seed_ry, "rz": seed_rz}})

    def __repr__(self):
        return f"Nanowire(rx={self.radius['rx']:.2e}, ry={self.radius['ry']:.2e}, L={self.length:.2e})"


class RoundedCuboid:
    def __init__(self, Dx=1e-6, Dy=1e-6, Dz=1e-6, 
                 rrx=1e-7, rry=1e-7, rrz=1e-7, rrx2=None, rry2=None, rrz2=None):
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
            
        Rounding radius can be different for the normal axis of planes with shared normals. So the zy plane can have different rrx values, but not different rry and rrz values.
        To make this easier, we will run rounding as a "per exception": rrx/rry/rrz are FOR ALL, and then you can elect to set rrx2/rry2/rrz2
        Examples: 
            I want rrx1 == rrx2: set rrx = value, and ignore rrx2
            I want rrx1 = val, and rrx2 = 0: Set rrx = value, and rrx2=0
            I want rrx1 = 0.5 and rrx2 = 1, you just set rrx=0.5 and rrx2=1.  
            
        This way, we can also remove planes completely for FLAT faces (if you want to make plus symbols or something)
        
        note that x1 = -1, anx x2 = +1 and vice versa!        

        Dx/Dy/Dz Explanation:
        This represents the TOTAL bounds of the object. This means that the rounding radius of any corner cannot exceed D/2, since then 2r == D
        If greater, then the sphere will clip out of the opposite side. Note, this also means that rrx1 can be small, while rrx2 can be D/2.
        
        We build objects by doing:
            rrx1 -> Dxp -> rrx2 (where Dxp = Dx-(rrx1+rrx2))
            This naturally also means that rrx1+rrx2 !> Dx
            But you can have: Dx = 100, rrx1 = 5, Dxp = 45, rrx2 = 50 and the maths will still work out! 
        """
        
        #Fix rrx/rry/rrz overlap, maximum permitted is D/2. After this, you might as well make a custom shape from spheres instead.
        
        #Assign new r2 values if not assigned, then fix the r_overlap. We will do a rudimentarry check to see if rounding can be ratiod, since if r1 = 1.5r2, then r1 should equal D/2, and r2 = D/(2*1.5) 
        r1_list  = [rrx,  rry,  rrz]
        r2_list  = [rrx2, rry2, rrz2]
        D_list   = [Dx,  Dy,  Dz]
        
        self.Dx = Dx  
        self.Dy = Dy
        self.Dz = Dz
        self.rr = {}
        for i, axis in enumerate("xyz"):
            self.rr[axis] = list(self._fix_overlap_and_r2(r1_list[i], r2_list[i], D_list[i]))
        
        self.recalculate()

    def recalculate(self):
        self.base_corners = {}
        self.base_edges   = {}
        self.corner_props = {}
        self.edge_props   = {}
        self.core_props   = {}
        self.r_spheres    = []
        self.r_cylinders  = []
        self.c_cubes      = []
    
        self._get_base_corners()
        self._get_corner_props()
        self._get_base_edges()
        self._get_edge_props()
        self._get_core_props()
        self._generate_primitives()
    
    def get_obb(self):
        """Local-space OBB — centre at origin, spans are Dx/Dy/Dz."""
        return OBB(center=[0, 0, 0], spans=[self.Dx, self.Dy, self.Dz])
    
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
            
            
            self.corner_props[(xi,yi,zi)] = {"rr":{"rx": self.base_corners[(xi,yi,zi)]["rr"]["rx"], "ry": self.base_corners[(xi,yi,zi)]["rr"]["ry"], "rz": self.base_corners[(xi,yi,zi)]["rr"]["rz"]},             
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
                    rr_edge[f"r{axis}"] = None
                else:
                    rr_edge[f"r{axis}"] = rr_corner[f"r{axis}"]
    
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
            if any(rr[f"r{ax}"] == 0 for ax in "xyz"):
                continue
            entrry = {"loc": props["loc"].copy(), "radius": rr.copy()}
            if entrry not in self.r_spheres:
                self.r_spheres.append(entrry)
        
        
       # ----------- CYLINDERS -----------
        for edge, props in self.edge_props.items():
            loc0, loc1 = props["loc"]
            rr = props["rr"]
        
            # skip zero-length edges
            if loc0 == loc1:
                continue
        
            # find cylinder axis (where radius is None)
            norm = next(ax[1] for ax, v in rr.items() if v is None)  # "rx"→"x" etc.
            
            # skip if any non-axis radius is zero
            if any(rr[f"r{ax}"] == 0 for ax in "xyz" if ax != norm):
                continue
            
                
            ax_idx = {"x": 0, "y": 1, "z": 2}
            # compute cylinder height along norm axis
            height = abs(loc1[ax_idx[norm]] - loc0[ax_idx[norm]])
            # compute cylinder center
            center = {ax: 0.5 * (loc0[ax_idx[ax]] + loc1[ax_idx[ax]]) for ax in "xyz"}
        
            # perpendicular radii
            radius = {f"r{ax}": rr[f"r{ax}"] for ax in "xyz" if ax != norm}
        
        
            entrry = {
                "norm": norm,
                "loc": center,
                "radius": radius,
                "height": height
            }
        
            # avoid duplicates
            if entrry not in self.r_cylinders:
                self.r_cylinders.append(entrry)
    
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
    
class RectGeo:
    def __init__(self, Dx, Dy, Dz):
        self.Dx = Dx
        self.Dy = Dy
        self.Dz = Dz

    def get_obb(self):
        return OBB(center=[0, 0, 0], spans=[self.Dx, self.Dy, self.Dz])

    def __repr__(self):
        return f"RectGeo(Dx={self.Dx:.2e}, Dy={self.Dy:.2e}, Dz={self.Dz:.2e})"


class SphereGeo:
    def __init__(self, rx, ry, rz):
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def get_obb(self):
        return OBB(center=[0, 0, 0], spans=[self.rx*2, self.ry*2, self.rz*2])

    def __repr__(self):
        return f"SphereGeo(rx={self.rx:.2e}, ry={self.ry:.2e}, rz={self.rz:.2e})"


class CylinderGeo:
    def __init__(self, rx, ry, rz, norm="z"):
        # rx/ry/rz are the three semi-axes; norm is the length axis
        # the two non-norm axes are radii, norm axis is half-length
        self.rx   = rx
        self.ry   = ry
        self.rz   = rz
        self.norm = norm

    def get_obb(self):
        # spans = diameter on radial axes, full length on norm axis
        ax_idx = {"x": 0, "y": 1, "z": 2}
        spans  = [self.rx*2, self.ry*2, self.rz*2]
        # norm axis span is already length (rz passed as half-length from _make_cylinder)
        return OBB(center=[0, 0, 0], spans=spans)

    def __repr__(self):
        return f"CylinderGeo(rx={self.rx:.2e}, ry={self.ry:.2e}, rz={self.rz:.2e}, norm={self.norm!r})"   
# ------------------------------------------------------------------
# Mixin — attached to Scene, has access to self.sim
# ------------------------------------------------------------------
class ObjectsMixin(hlp.HelpersMixin):
    """
    Do I want to do this for primitives or not? I could either do this, or add it as an easier means to get and set props after creation?
    Tough choice tbh
    props = sim.getnamed("object_name", "?")
    # or if already selected:
    props = sim.get("?")
    
    def get_all_props(self, name=None):
    if name:
        self.sim.select(name)
    prop_names = self.sim.get("?")  # returns newline-separated string or list depending on version
    if isinstance(prop_names, str):
        prop_names = [p for p in prop_names.strip().split("\n") if p]
    
    return {prop: self.sim.get(prop) for prop in prop_names}
    """        
    def add_primitive(self, primitive="rect",
                  x=_S, y=_S, z=_S,
                  Dx=_S, Dy=_S, Dz=_S,
                  rx=_S, ry=_S, rz=_S,
                  rotx=_S, roty=_S, rotz=_S,
                  norm="z",
                  material=None, zorder=0,
                  name=None, group=None, fullpath=None,
                  standalone=True, axis_offset=(0,0,0),
                  prop_dict=None, **geo):
        
        defaults = dflt.get_default_prop_dicts()[dflt._get_primitive_mapping()[primitive]]
        
        props = hlp.resolve_and_merge(
        defaults, prop_dict,
        explicit={"x": x, "y": y, "z": z,
                  "Dx": Dx, "Dy": Dy, "Dz": Dz,
                  "rx": rx, "ry": ry, "rz": rz,
                  "rotx": rotx, "roty": roty, "rotz": rotz},
        geo=geo
        )
        
        rot = hlp._extract_rotation(props)  # always pop from props
        rx_, ry_, rz_ = rot

        if primitive == "sphere":
            self.sim.addsphere()
            geo = SphereGeo(
            rx=props.pop("x span") / 2,
            ry=props.pop("y span") / 2,
            rz=props.pop("z span") / 2
            )
            for ax, key in zip(("x","y","z"), ("radius","radius 2","radius 3")):
                props[key] = props.pop(f"{ax} span") / 2

        elif primitive == "rect":
            self.sim.addrect()

        elif primitive == "cylinder":
            self.sim.addcircle()
            props["make ellipsoid"] = 1
            radial = [ax for ax in ("x","y","z") if ax != norm]
            props["radius"]   = props.pop(f"{radial[0]} span") / 2
            props["radius 2"] = props.pop(f"{radial[1]} span") / 2
            if norm == "x": ry_ += 90
            if norm == "y": rx_ += 90

        # If not standalone, rotation goes back into props for set_obj_props
        if not standalone:
            props["first axis"]  = "x"; props["rotation 1"] = rx_
            props["second axis"] = "y"; props["rotation 2"] = ry_
            props["third axis"]  = "z"; props["rotation 3"] = rz_

        pathdict = self.resolve_fullpath(name, group=group, fullpath=fullpath)
        self.create_groups_from_dict({"structure": [pathdict["fullpath"]]})
        props["name"] = pathdict["name"]
        props["mesh order"] = zorder
        if material:
            props["material"] = material
            props["override mesh order from material database"] = 1
        if group:
            self.sim.addtogroup(pathdict["group"])

        self.set_obj_props(props)

        if standalone:
            scene_obj = SceneObject(
                geo=None,
                x=props["x"], y=props["y"], z=props["z"],
                rotx=rx_, roty=ry_, rotz=rz_,
                pathdict=pathdict,
                axis_offset=axis_offset
            )
            return self._register(scene_obj)
    
    def add_roundedcube(self, RC, x=0, y=0, z=0, rotx = 0, roty = 0, rotz = 0, axis_offset = (0,0,0), 
                        name="RoundedCube", group=None, fullpath = None, material=None, zorder=0):
        """
        Instantiate a RoundedCuboid in a Lumerical simulation using lumapi.
        
        sim: lumapi simulation object (e.g. FDTD(), varFDTD(), CHARGE(), etc.)
        RC: RoundedCuboid instance
        material: Lumerical material string
        zorder: drawing order
        axis_offset = (dx,dy,dz), moves the local origin so that rotations can be done geometrically around the centre.
        x,y,z = location of the object relative to the local centre defined by the axis offset in 3d space.
        group: name of group to insert objects into (string)
        """
        aox, aoy, aoz = axis_offset
        
        pathdict   = self.resolve_fullpath(name,group=group,fullpath=fullpath) 
        name, group, fullpath = [pathdict["name"],pathdict["group"], pathdict["fullpath"]]
        self.create_groups_from_dict( {"structure": [fullpath]})
            

        # -------------------------------------------------------------
        # CORE CUBES
        # -------------------------------------------------------------
        for cube in RC.c_cubes:
            cube_range = cube["range"]
            xx = (cube_range["xmin"] + cube_range["xmax"]) / 2 + aox 
            yy = (cube_range["ymin"] + cube_range["ymax"]) / 2 + aoy 
            zz = (cube_range["zmin"] + cube_range["zmax"]) / 2 + aoz 
            Dx = cube_range["xmax"] - cube_range["xmin"]
            Dy = cube_range["ymax"] - cube_range["ymin"]
            Dz = cube_range["zmax"] - cube_range["zmin"]

            self.add_primitive(primitive="rect",x=xx,y=yy,z=zz, Dx=Dx, Dy=Dy, Dz=Dz,material=material,zorder=zorder,group=fullpath,standalone=False)        

        # -------------------------------------------------------------
        # SPHERES
        # -------------------------------------------------------------
        for sphere in RC.r_spheres:
            loc = sphere["loc"]
            radius = sphere["radius"]

            xx = loc["x"] + aox 
            yy = loc["y"] + aoy 
            zz = loc["z"] + aoz 

            rcx, rcy, rcz = radius["rx"], radius["ry"], radius["rz"]
            self.add_primitive(primitive="sphere",x=xx,y=yy,z=zz, rx=rcx, ry=rcy, rz=rcz,material=material,zorder=zorder,group=fullpath,standalone=False)      
            

        # -------------------------------------------------------------
        # CYLINDERS
        # -------------------------------------------------------------
        for cyl in RC.r_cylinders:
            loc = cyl["loc"]
            radius = cyl["radius"]
            norm = cyl["norm"]

            xx = loc["x"] + aox 
            yy = loc["y"] + aoy 
            zz = loc["z"] + aoz 

            rcx = radius.get("rx", 0)
            rcy = radius.get("ry", 0)
            rcz = radius.get("rz", 0)
            
            self.add_primitive(primitive="cylinder",x=xx,y=yy,z=zz, rx=rcx, ry=rcy, rz=rcz,material=material,zorder=zorder,group=fullpath,norm=norm,standalone=False)  
        
        scene_obj = SceneObject(RC, x=x, y=y, z=z, rotx=rotx, roty=roty, rotz=rotz,axis_offset=axis_offset,
                                    pathdict=pathdict)
        return self._register(scene_obj)
    
    def add_nanowire(self, nw, x=0, y=0, z=0, rotx=0, roty=0, rotz=0, axis_offset=(0,0,0),
                     material=None, seed_material=None,
                     zorder=0, name="Nanowire", group=None,fullpath=None):
       
        pathdict   = self.resolve_fullpath(name,group=group,fullpath=fullpath) 
        name, group, fullpath = [pathdict["name"],pathdict["group"], pathdict["fullpath"]]
        self.create_groups_from_dict( {"structure": [fullpath]})
        
        aox, aoy, aoz = axis_offset
        offset = {"x": aox, "y": aoy, "z": aoz}

        for sphere in nw.seed_list:
            loc = {k: v + offset[k] for k, v in sphere["loc"].items()}
            self.add_primitive(primitive="sphere", **loc,
                        **sphere["radius"], material=seed_material,
                        zorder=zorder + 1, name="SeedSphere", group=fullpath,standalone=False)

        for sphere in nw.endcaps_list:
            loc = {k: v + offset[k] for k, v in sphere["loc"].items()}
            self.add_primitive(primitive="sphere", **loc,
                        **sphere["radius"], material=material,
                        zorder=zorder, name="EndcapSphere", group=fullpath,standalone=False)

        for cyl in nw.core_cylinder:
            loc = {k: v + offset[k] for k, v in cyl["loc"].items()}
            self.add_primitive(primitive="cylinder", **loc,
                        **cyl["radius"], material=material,
                        zorder=zorder, name="NWCoreCylinder", group=fullpath,standalone=False)

        scene_obj = SceneObject(nw, x=x, y=y, z=z, rotx=rotx, roty=roty, rotz=rotz,
                                pathdict=pathdict, axis_offset=axis_offset)
        return self._register(scene_obj)
                
    
