# lumutils/objects/nanowire.py
import numpy as np
from ..geometry.OBB import OBB
from ..scene_object import SceneObject
from .. import helpers as hlp

from ...custom_logger import get_custom_logger

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

        self.radius       = radius
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
        xy_span = self.radius * 2
        # endcaps add cap_factor*radius beyond each end
        cap_ext = self.radius * self.cap_factor if self.endcaps != "none" else 0
        z_span  = self.length + 2 * cap_ext
        return OBB(center=[0, 0, 0], spans=[xy_span, xy_span, z_span])

    # --- geometry builders unchanged from your original ---
    def _make_cylinder(self):
        cyl = {
            "loc":    {"x": 0, "y": 0, "z": self.length / 2},
            "radius": {"x": self.radius, "y": self.radius, "z": self.length / 2},
            "range":  {"zmin": -self.length / 2, "zmax": self.length / 2},
            "norm":   [0, 0, 1]
        }
        self.core_cylinder.append(cyl)

    def _make_endcaps(self):
        if self.endcaps not in ("bottom", "top", "both") or self.shape != "circle":
            return
        r  = self.radius
        rz = r * self.cap_factor
        if self.endcaps in ("bottom", "both"):
            self.endcaps_list.append({"loc": {"x": 0, "y": 0, "z": -self.length / 2},
                                      "radius": {"x": r, "y": r, "z": rz}})
        if self.endcaps in ("top", "both"):
            self.endcaps_list.append({"loc": {"x": 0, "y": 0, "z":  self.length / 2},
                                      "radius": {"x": r, "y": r, "z": rz}})

    def _make_seeds(self):
        if self.seed not in ("bottom", "top", "both"):
            return
        seed_rz = self.radius * self.seed_rfactor
        if self.seed in ("bottom", "both"):
            self.seed_list.append({"loc": {"x": 0, "y": 0,
                                           "z": -self.length / 2 - self.seed_rfactor * self.radius},
                                   "radius": {"x": seed_rz, "y": seed_rz, "z": seed_rz}})
        if self.seed in ("top", "both"):
            self.seed_list.append({"loc": {"x": 0, "y": 0,
                                           "z":  self.length / 2 + self.seed_rfactor * self.radius},
                                   "radius": {"x": seed_rz, "y": seed_rz, "z": seed_rz}})

    def __repr__(self):
        return f"Nanowire(r={self.radius:.2e}, L={self.length:.2e})"


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
    
# ------------------------------------------------------------------
# Mixin — attached to Scene, has access to self.sim
# ------------------------------------------------------------------
class ObjectsMixin(hlp.HelpersMixin):
    
    def _set_mat_zo_name_group(self,material,zorder,name,group):
        if material: 
            self.sim.set("material", material)
        self.sim.set("override mesh order from material database", 1)
        self.sim.set("mesh order", zorder)
            
        if name:
            self.sim.set("name",name)    
        if group:
            self.sim.addtogroup(group)
            
    def add_primitive(self, primitive="rect", method="span", x=0, y=0, z=0, Dx=None, rx=None, Dy=None, ry=None, Dz=None, rz=None, norm="z", xminmax=[0,0], yminmax=[0,0], zminmax=[0,0], material=None, zorder=0,name=None,group=None,standalone=True):
        
        xyz, D, r, mm = hlp.coordinate_standardisation(method=method, x=x, y=y, z=z, Dx=Dx, Dy=Dy, Dz=Dz, rx=rx, ry=ry, rz=rz, xmm=xminmax, ymm=yminmax, zmm=zminmax)
        axstr = ["x", "y", "z"]

        if primitive == "sphere":
            self.sim.addsphere()
            for i, axs in enumerate(axstr):
                self.sim.set(axs, xyz[i])
            self.sim.set("radius",   r[0])
            self.sim.set("radius 2", r[1])
            self.sim.set("radius 3", r[2])
        
        if primitive == "rect":
            self.sim.addrect()
            for i, axs in enumerate(axstr):
                self.sim.set(axs, xyz[i])
                self.sim.set(axs + " span", D[i])
        
        if primitive == "cylinder":
            self.sim.addcircle()
            self.sim.set("first axis",  "x")
            self.sim.set("second axis", "y")
            self.sim.set("third axis",  "z")
            xr, yr = 0, 0
            if norm == "x": yr += 90
            if norm == "y": xr += 90
            self.sim.set("make ellipsoid", 1)
            self.sim.set("x", xyz[0])
            self.sim.set("y", xyz[1])
            self.sim.set("z", xyz[2])
            self.sim.set("radius",   r[0])
            self.sim.set("radius 2", r[1])
            self.sim.set("z span",   D[2])
            self.sim.set("rotation 1", xr)
            self.sim.set("rotation 2", yr)
            self.sim.set("rotation 3", 0)
        self._set_mat_zo_name_group(material, zorder, name, group)
        
        
        return()
    
    def add_roundedcube(self, RC, name="RoundedCube", material=None, zorder=0, axis_offset=(0,0,0), group=None,bounds=None):
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
        gdir = group if group else name
        self.create_groups_from_dict( {"structure": [gdir]})
            

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

            self.add_primitive(primitive="rect",xyz=(x,y,z), Dx=Dx, Dy=Dy, Dz=Dz,material=material,zorder=zorder,group=gdir,standalone=False)        

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
            self.add_primitive(primitive="sphere",xyz=(x,y,z), rx=rx, ry=ry, rz=rz,material=material,zorder=zorder,group=gdir,standalone=False)      
            

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
            
            self.add_primitive(primitive="cylinder",xyz=(x,y,z), rx=rx, ry=ry, rz=rz,material=material,zorder=zorder,group=gdir,norm=norm,standalone=False)  
            scene_obj = SceneObject(RC, x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                                    name=name, group=gdir)
            return self._register(scene_obj)
    
    def add_nanowire(self, nw, x=0, y=0, z=0, rx=0, ry=0, rz=0,
                     material=None, seed_material=None,
                     zorder=0, name="Nanowire", group=None):

        gdir = group if group else name
        self.create_groups_from_dict( {"structure": [gdir]})

        offset = {"x": x, "y": y, "z": z}

        for sphere in nw.seed_list:
            sphere = hlp.fix_single_or_nonstandard_rxkeys(sphere)
            loc = {k: v + offset[k] for k, v in sphere["loc"].items()}
            self.add_primitive( primitive="sphere", **loc,
                        **sphere["radius"], material=seed_material,
                        zorder=zorder + 1, name="SeedSphere", group=gdir,standalone=False)

        for sphere in nw.endcaps_list:
            sphere = hlp.fix_single_or_nonstandard_rxkeys(sphere)
            loc = {k: v + offset[k] for k, v in sphere["loc"].items()}
            self.add_primitive(primitive="sphere", **loc,
                        **sphere["radius"], material=material,
                        zorder=zorder, name="EndcapSphere", group=gdir,standalone=False)

        for cyl in nw.core_cylinder:
            loc = {k: v + offset[k] for k, v in cyl["loc"].items()}
            self.add_primitive(primitive="cylinder", **loc,
                        **cyl["radius"], material=material,
                        zorder=zorder, name="NWCoreCylinder", group=gdir,standalone=False)

        scene_obj = SceneObject(nw, x=x, y=y, z=z, rx=rx, ry=ry, rz=rz,
                                name=name, group=gdir)
        return self._register(scene_obj)
                
    
