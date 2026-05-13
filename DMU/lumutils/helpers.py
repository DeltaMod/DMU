import numpy as np 

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

def fix_single_or_nonstandard_rxkeys(dictitem):
    newdict = dictitem.copy()

    if "radius" in dictitem.keys():
        if type(dictitem["radius"]) != dict:
            radius = dictitem["radius"]
            newdict["radius"] = {"rx":radius,"ry":radius,"rz":radius}
        else:
            if "x" in dictitem["radius"].keys():
                newdict["radius"]["rx"] = dictitem["radius"]["x"]
                newdict["radius"]["ry"] = dictitem["radius"]["y"]
                newdict["radius"]["rz"] = dictitem["radius"]["z"]
    return(newdict)

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

class HelpersMixin:
    def select_and_set_props(self, name, propdict):
        """
        If name=None, we assume it's already been selected in the scope, so you can use this after creation without passing a name
        """
        if name:
            self.sim.select(name)
        for key, item in propdict.items():
            self.sim.set(key,item)

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

    
