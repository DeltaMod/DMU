# lumutils/defaults.py
from types import MappingProxyType as _frozen

# ── Building blocks (immutable) ───────────────────────────────────────────────
_GEO_XYZ      = _frozen({"x": 0, "y": 0, "z": 0})
_GEO_SPAN     = _frozen({"x span": 1e-6, "y span": 1e-6, "z span": 1e-6})
_GEO_MINMAX   = _frozen({"x min": -5e-7, "x max": 5e-7,
                          "y min": -5e-7, "y max": 5e-7,
                          "z min": -5e-7, "z max": 5e-7})
_GEO_ROT_AXIS = _frozen({"first axis": "x", "second axis": "y", "third axis": "z"})
_GEO_ROT      = _frozen({"rotation 1": 0, "rotation 2": 0, "rotation 3": 0})
_GEO_MATERIAL = _frozen({"override mesh order from material database": 1, "mesh order": 1})

# ── Composed defaults (plain dicts, safe to use as **defaults) ────────────────
DEFAULTS_GEOSPAN = {**_GEO_XYZ, **_GEO_SPAN,
                     **_GEO_ROT_AXIS, **_GEO_ROT, **_GEO_MATERIAL}

DEFAULTS_GEORADIUS = {**_GEO_XYZ, **_GEO_SPAN,
                     **_GEO_ROT_AXIS, **_GEO_ROT, **_GEO_MATERIAL}

DEFAULTS_FDTD     = {**_GEO_XYZ, **_GEO_SPAN,
                     "mesh type": "auto non-uniform",
                     "mesh accuracy": 3,
                     "dt stability factor": 0.5,
                     "mesh refinement": "conformal variant 1",
                     "min mesh step": 0.00025,
                     "x min bc": "PML", "x max bc": "PML",
                     "y min bc": "PML", "y max bc": "PML",
                     "z min bc": "PML", "z max bc": "PML",
                     "auto shutoff min": 1e-5,
                     "auto shutoff max": 10000}

DEFAULTS_MONITOR  = {**_GEO_XYZ, **_GEO_SPAN}


_DEFAULTS_MAP = {
    "FDTD":     DEFAULTS_FDTD,
    "geospan":  DEFAULTS_GEOSPAN,
    "georadius":DEFAULTS_GEORADIUS, 
    "monitor":  DEFAULTS_MONITOR,
}

_PRIMITIVE_DEFAULTS = {
    "rect":     "geospan",
    "sphere":   "georadius",
    "cylinder": "georadius",
}

def  get_default_prop_dicts() -> dict[str, dict]:
    """
    Returns a dict[str, dict]
    -------
    Dict contains default set entries for all objects in lumutils. The different types are as follows:
        FDTD: Defaults for all parametrs of the FDTD box. From x/y/z/x span/y span/z span etc to auto shutoff min.
        geospan: Defaults for geometry that makes use of x span
        georadius: Defaults for geometry that makes use of radius. NOTE: If you pass this to a cylinder, the norm will dictate which of these axes are popped to avoid trying to access inactive parameters
        monitor: Defaults for generic monitor groups - you add the rest to the dict yourself :)
    
    How to use: grab the dict you want, and add it to prop_dict entries with modified values if you want.
    """
    return {k: dict(v) for k, v in _DEFAULTS_MAP.items()}

def _get_primitive_mapping() -> dict[str,dict]:
    return {k: dict(v) for k, v in _PRIMITIVE_DEFAULTS.items()}