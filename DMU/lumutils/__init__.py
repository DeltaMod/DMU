# lumutils/__init__.py
# Group1 Imports - no local deps
from .helpers import *
from .geometry.OBB import OBB
from .defaults import get_default_prop_dicts

# Group2 Imports - depends on Group1
from .scene_object import SceneObject

# Group3 Imports - depends on Groups1+2
from .geometry import *
from .data import *

# Group4 Imports - Only ran when lumapi is needed
try:
    import lumapi
    from .analysis import *
    from .SCENE import Scene
except ImportError:
    pass