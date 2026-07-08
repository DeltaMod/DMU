import numpy as np 
_S = object() #Add sentinel object for empty fields
from ..custom_logger import get_custom_logger
logger = get_custom_logger("DLUM_HELPERS")
from . import defaults as dflt

def are_all_dict_values_type(d,ttype=None):
    if isinstance(d, dict):  # If d is a dictionary, check all values
        return all(are_all_dict_values_type(value) for value in d.values())
    elif isinstance(d, list):  # If d is a list, check all elements
        return all(are_all_dict_values_type(value) for value in d)
    else:  # If it's neither a dict nor a list, just check if the value is None
        return d is None

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

def filter_maximal_paths(paths):
    """Keep only paths that are not prefixes of other paths"""
    paths_sorted = sorted(paths, key=lambda x: -len(x))  # longest first
    maximal = []
    for p in paths_sorted:
        if not any(other.startswith(p + "::") for other in maximal):
            maximal.append(p)
    return maximal        
    
##########################
#%% GEOMETRY HELPERS!!!###
##########################

def _resolve_geometry(explicit: dict, geo: dict) -> dict:
    """
    Resolves all geometry inputs into a canonical flat dict of Lumerical keys.

    Priority (highest → lowest):
      explicit kwargs  >  geo kwargs  >  nothing (key absent, defaults fill later)

    Position/span resolution per axis:
      - xmm / (xmin+xmax)  →  centre + span  (min/max discarded after)
      - Dx / rx            →  span (Dx preferred over rx if both given)
      - x                  →  centre
     Shorthands accepted:
        xyz=(x,y,z)         Dxyz=(Dx,Dy,Dz)      radxyz=(rx,ry,rz)
        rotxyz=(rx,ry,rz)   xmm=(lo,hi)           xmin+xmax
        rotx/roty/rotz      Dx/Dy/Dz              rx/ry/rz
    
    Rotation:
      rotxyz=(a,b,c) or rotx/roty/rotz  →  "first axis":"x","rotation 1":a, etc.
      Axes are always x/y/z in order.
    
    Output keys are always Lumerical strings:
        "x","y","z"  /  "x span","y span","z span"  /  "rotation 1" etc.
    """
    out = {}

    #### Unpack tuple shorthands from geo ####
    _unpack = [
       ("xyz",    "xyz",               ("x",   "y",   "z"  )),
       ("Dxyz",   "Dxyz",              ("Dx",  "Dy",  "Dz" )),
       ("radxyz", "radxyz",            ("rx",  "ry",  "rz" )),
       ("rotxyz", "rotxyz",            ("rotx","roty","rotz")),
    ]
    for key, _, targets in _unpack:
        if key in geo:
            for t, v in zip(targets, geo.pop(key)):
                geo.setdefault(t, v)
    
    for ax in ("x", "y", "z"):
        if f"{ax}mm" in geo:
            lo, hi = geo.pop(f"{ax}mm")
            geo.setdefault(f"{ax}min", lo)
            geo.setdefault(f"{ax}max", hi)
    
    #### Merge explicit on top explicit kwargs beat geo kwargs, sentinels excluded ####
    merged = {**geo, **{k: v for k, v in explicit.items() if v is not _S}}

    # ── Per-axis position + span resolution ──────────────────────────────────
    for ax in ("x", "y", "z"):
        centre = merged.get(ax)
        D      = merged.get(f"D{ax}")
        r      = merged.get(f"r{ax}")
        lo     = merged.get(f"{ax}min")
        hi     = merged.get(f"{ax}max")

        if (D is not None or r is not None) and (lo is not None or hi is not None):
            logger.warn(
                f"Both span and min/max provided for {ax!r} — using span.",
                UserWarning, stacklevel=3
            )
            lo = hi = None

        if lo is not None and hi is not None:
            out[ax]          = (lo + hi) / 2
            out[f"{ax} span"] = hi - lo
        else:
            if D is not None:
                out[f"{ax} span"] = D
            elif r is not None:
                out[f"{ax} span"] = r * 2
            if centre is not None:
                out[ax] = centre

    # ── Rotation ─────────────────────────────────────────────────────────────
    _rot_map = (
        ("rotx", "first axis",  "x", "rotation 1"),
        ("roty", "second axis", "y", "rotation 2"),
        ("rotz", "third axis",  "z", "rotation 3"),
    )
    for short, axis_key, axis_val, rot_key in _rot_map:
        angle = merged.get(short)
        if angle is not None:
            out[axis_key] = axis_val
            out[rot_key]  = angle

    return out


#### prop_dict cleaner ####

def _clean_prop_dict(prop_dict: dict, resolved_geo: dict) -> dict:
    """
    Remove from prop_dict any geometry keys that are already covered by
    resolved_geo, to prevent stale min/max conflicting with resolved span.
    Non-geometry keys pass through untouched.
    """
    # Which lumerical geo keys did we actually resolve?
    covered = set(resolved_geo.keys())
    # If we resolved x span, also evict x min/x max (and vice versa)
    for ax in ("x", "y", "z"):
        if f"{ax} span" in covered:
            covered |= {f"{ax} min", f"{ax} max"}
        if f"{ax} min" in covered or f"{ax} max" in covered:
            covered |= {f"{ax} span"}

    return {k: v for k, v in prop_dict.items() if k not in covered}


#### Main merge entry point ####

def resolve_and_merge(defaults: dict, prop_dict: dict | None,
                      explicit: dict, geo: dict) -> dict:
    """
    Produces the final flat Lumerical property dict.

    Priority: explicit kwargs > prop_dict > defaults
    Geometry in prop_dict is evicted where explicit/geo kwargs cover the same axis.
    """
    prop_dict   = prop_dict or {}
    resolved    = _resolve_geometry(explicit, geo)
    clean_props = _clean_prop_dict(prop_dict, resolved)

    return {**defaults, **clean_props, **resolved}

#### Since rotations should not be set on object instancing, it must be removed from the dict and passed elsewhere

def _extract_rotation(props: dict) -> tuple:
    """
    Pops rotation keys from props and returns (rx, ry, rz).
    Also pops reference to first axis / second axis / third axis. 
    """
    r1 = props.pop("rotation 1", 0)
    r2 = props.pop("rotation 2", 0)
    r3 = props.pop("rotation 3", 0)
    for k in ("first axis", "second axis", "third axis"):
        props.pop(k, None)
    return (r1, r2, r3)

class HelpersMixin:
    
    def set_mat_zorder_group_name(self,material,zorder,name,group):
        if material: 
            self.sim.set("material", material)
        self.sim.set("override mesh order from material database", 1)
        self.sim.set("mesh order", zorder)
            
        if name:
            self.sim.set("name",name)    
        if group:
            self.sim.addtogroup(group)
            
    def select_and_set_props(self, name, propdict):
        """
        If name=None, we assume it's already been selected in the scope, so you can use this after creation without passing a name
        """
        if name:
            self.sim.select(name)
        for key, item in propdict.items():
            self.sim.set(key,item)

    def resolve_fullpath(name, group=None, fullpath=None):
        """
        Sorts out what the user intended for the input. Resolves down fullpath-> name,group,fullpath or figures out which combination of group and name was used.
        Resolve the lumapi fullpath for a structure group.
        If fullpath is provided, then name and group are ignored.
            name="Name", group = None outputs "Name"
            name="Name", group="Group1::Group2") outputs "Group1::Group2::Name"
    
        Output: PathDict = {"name":name,"group":group,"fullpath":fullpath}
        """
        if fullpath:
            composite = fullpath.split("::") 
            name = composite[-1]
            if len(composite) == 1:
                group = None
            else:
                group = "::".join(composite[:-1])
            return({"name":name,"group":group,"fullpath":fullpath})
        
        if group:
            if not name:
                raise NameError("name field has been intentionally left blank, use fullpath = instead of group if you want this syntax.")
            
            fullpath = "::".join([group,name])
            return({"name":name,"group":group,"fullpath":fullpath})
        
        if name:
            fullpath = name
            return({"name":name,"group":group,"fullpath":fullpath})
        
        raise NameError("At least one of name, group, or fullpath must be provided.")

    def create_groups_from_dict(self, grouplist):
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

        # --- Structure groups ---
        struct_paths = filter_maximal_paths(grouplist.get("structure", []))
        for path in struct_paths:
            self.sim.addstructuregroup()
            self.sim.set("name", "structure_setter")
            self.sim.addtogroup(path)
            self.sim.delete()

        # --- Analysis groups ---
        analysis_paths = filter_maximal_paths(grouplist.get("analysis", []))

        if analysis_paths:
            # Find all unique top-level analysis groups
            top_level_analysis = set(p.split("::")[0] for p in analysis_paths)

            # Create each top-level analysis group
            for top_group in top_level_analysis:
                self.sim.addanalysisgroup()
                self.sim.set("name", top_group)

            # Add all paths under their respective top-level analysis groups
            for path in analysis_paths:
                top_group = path.split("::")[0]
                self.sim.addanalysisgroup()
                self.sim.set("name", "analysis_setter")
                self.sim.addtogroup(path)
                self.sim.delete()
    
    def set_obj_props(self,setprops):
        """
        setprops is a dict that contains "key":value for a variety of named parameters of the SELECTED object. 
        This will be invoked at the END of the function, meaning that you can provide a setprops dict instead of x=,y=,z etc
        The order of the entries is their order of execution, meaning if you want to set a rotation, you must provide primary axis 
        Example:
            setprops = {"x":1,"y":1,"z":1,"first axis":x,"second axis":y,"third axis":"z",...}
            
        Note that if you use **setprops then x,y,z will not be processed a second time, as they will take over from the normal function.
        """
        for setprop,val in setprops.items():
            self.sim.set(setprop,val)
            
    def set_named(self,name,setnamed):
        """
        setnamed is a dict that contains "key":value for a variety of named parameters of a NAMED object (struct/analysis, you name it). 
        This will be invoked at the END of the function, meaning that you can provide a setnamed dict instead of x=,y=,z etc
        The order of the entries is their order of execution, meaning if you want to set a rotation, you must provide primary axis 
        Example:
            setnamed = {"x":1,"y":1,"z":1,"first axis":x,"second axis":y,"third axis":"z",...}
            
        Note that if you use **setnamed then x,y,z will not be processed a second time, as they will take over from the normal function.
        """
        for setname,val in setnamed.items():
            self.sim.setnamed(name, setname, val)
        
    def set_loc_rot(self,xyz,r):
        self.sim.set("x",xyz[0])
        self.sim.set("y",xyz[1])
        self.sim.set("z",xyz[2])
    
    #%% OBB HELPERS
    def get_monitor_bounds(self, dim="3d", normal="z", 
                           padding=0,
                           xpad=None, ypad=None, zpad=None,
                           transform_coords="local"):
        """
        Calculate monitor bounds from OBB, with optional padding.
        
        padding      : scalar, symmetric additive on all non-normal axes
        xpad/ypad/zpad: scalar → symmetric [-v,+v], list → asymmetric [min,max]
                       normal axis pad is ignored
        transform_coords: "local" → bounds relative to OBB centre
                          "world" → absolute world bounds from OBB aabb
        dim          : "2d" → normal axis collapsed to OBB centre
                       "3d" → all axes have extent
        normal       : "x"|"y"|"z", only relevant for dim="2d"
        """
        
        def resolve_pad(pad):
            """Scalar → [-v,+v], list/tuple → as-is"""
            if pad is None:
                return [0, 0]
            if isinstance(pad, (list, tuple)):
                return pad
            return [-pad, +pad]
    
        pads = {"x": resolve_pad(xpad), 
                "y": resolve_pad(ypad), 
                "z": resolve_pad(zpad)}
    
        if transform_coords == "local":
            mn = self.obb.center - self.obb.spans / 2
            mx = self.obb.center + self.obb.spans / 2
        else:
            mn, mx = self.obb.aabb
    
        axes = ["x", "y", "z"]
        bounds = {}
        for i, ax in enumerate(axes):
            if dim == "2d" and ax == normal:
                centre = (mn[i] + mx[i]) / 2
                bounds[ax] = [centre, centre]
            else:
                bounds[ax] = [
                    mn[i] - padding + pads[ax][0],
                    mx[i] + padding + pads[ax][1]
                ]
    
        return bounds