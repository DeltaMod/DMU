"""
Data containers and file-loading logic for DMUI.
"""
from dataclasses import dataclass, field
from typing import Protocol, Optional, Any
import numpy as np
from pathlib import Path

#Local Packages:
try:
    
    from .Plotting_Tools import *
    from ..data_import_tools import Keithley_xls_read_file, build_device_data, save_device_data
except ImportError:
    from DMU.DMUI.Plotting_Tools import *
    from DMU.data_import_tools import Keithley_xls_read_file, build_device_data, save_device_data
    

"""
################################################################### 
### DATA CLASSES FOR CONTAINING THE DATA OF EACH MEASUREMENT FILE #
###################################################################
""" 

@dataclass
class CurveData:
    """
    Each curve gets its own container class. 
    This way, we can more easily alter the contents and titles of each curve without modifying the base data.
    """
    name: str                      # display name, e.g. "Current [A]"
    x: np.ndarray
    y: np.ndarray
    x_unit: str = ""
    y_unit: str = ""
    source_key: str = ""           # for Keithley: the column name in the file

@dataclass
class Measurement:
    """Many measurements will contain multiple curves, and therefore each measurement should be made up of curves that can all individually identify themselves."""
    measurement_id: int
    title: str
    timestamp: str = ""
    comment: str = ""
    curves: list[CurveData] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    log: dict = field(default_factory=dict)      # <-- NEW: raw LOG dict (Keithley)
    

@dataclass
class LoadedFile:
    path: Path
    source_type: str
    measurements: list[Measurement]
    group: Optional[str] = None
    subdevice: Optional[str] = None
    # NEW: for write-back
    device_data: Optional[dict] = None      # whole pkl structure in memory
    pkl_path: Optional[Path] = None

    def save(self):
        """Persist edits (e.g. LOG) back into the folder's pkl."""
        if self.pkl_path is None or self.device_data is None:
            return
        import pickle
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self.device_data, f)


"""
################################################################### 
###         DATA HANDLING, LOADING, AND CATEGORISATION            #
###################################################################
""" 
@dataclass
class Data_Mapping:
   """
   Contains default key names and similar. Should probably used together with a fallback
   """
   KeithleyMap = {
    "x_patterns": ["V", "voltage", "bias"],
    "y_patterns": ["I", "current", "A"],
    # Explicit overrides take priority
    "explicit_pairs": [],   # [["Voltage", "Current"], ...]
    }
                
class DataSource(Protocol):
    name: str  # "palmsens", "keithley"

    def can_load(self, path: Path) -> bool: ...

    def load(self, path: Path, **kwargs) -> LoadedFile: ...

    def list_measurements(self, path: Path) -> list[tuple[int, str]]:
        """Cheap peek — used to populate the tree without loading curves."""
        ...

class PalmSensSource:
    name = "palmsens"

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".pssession"

    def load(self, path: Path, **kwargs) -> LoadedFile:
        import pypalmsens as ps
        DATA = ps.load_session_file(str(path))
        measurements = []
        for i, dat in enumerate(DATA):
            curves = []
            for c in (dat.curves if getattr(dat, "n_curves", 0) else []):
                x, y = curve_to_xy(c)
                curves.append(CurveData(
                    name=curve_display_name(c),
                    x=x, y=y,
                    x_unit=getattr(c, "x_unit", ""),
                    y_unit=getattr(c, "y_unit", ""),
                ))
            measurements.append(Measurement(
                measurement_id=i,
                title=getattr(dat, "title", "") or path.stem,
                timestamp=getattr(dat, "timestamp", ""),
                comment=getattr(
                    getattr(getattr(dat, "_psmeasurement", None), "Method", None),
                    "Notes", ""
                ),
                curves=curves,
            ))
        return LoadedFile(path=path, source_type=self.name, measurements=measurements)

    def list_measurements(self, path: Path) -> list[tuple[int, str]]:
        # Reuse load() but discard curves — pypalmsens doesn't give a cheaper path
        lf = self.load(path)
        return [(m.measurement_id, m.curves[0].name if m.curves else "no curves")
                for m in lf.measurements]
    
class KeithleySource:
    name = "keithley"

    # Which column pairs do we plot? (applied_col, measured_col)
    # Configurable via self.column_map below.
    DEFAULT_PAIRS = [("voltage", "current")]

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".xls" and "LOG" not in path.stem.upper()

    def load(self, path: Path, **kwargs) -> LoadedFile:
        folder = path.parent
        pkl_path = folder / f".dmui_{folder.name}.pkl"
    
        # build_device_data loads from pkl if present, else parses xls and writes pkl
        device_data = build_device_data(folder)
    
        # filter to runs that came from *this* file
        file_name = path.name
        measurements = []
        i = 0
        for np_key, devs in device_data.items():
            if np_key == "_meta":
                continue        # do not load "_meta" key
            for dev, subs in devs.items():
                for sub, runs in subs.items():
                    for run_key, run_data in runs.items():
                        if run_data.get("filename") != file_name:
                            continue
                        curves = self._extract_curves(run_data)
                        settings = run_data.get("Settings", {})
                        log = run_data.get("LOG", {}) or {}
                        measurements.append(Measurement(
                            measurement_id=i,
                            title=f"{run_key}: {settings.get('Test Name', '')}".strip(": "),
                            timestamp=settings.get("Execution Time", ""),
                            comment=log.get("Comment", ""),
                            curves=curves,
                            metadata={"run_key": run_key, "settings": settings},
                            log=log,
                        ))
                        i += 1
    
        return LoadedFile(
            path=path, source_type=self.name, measurements=measurements,
            group=kwargs.get("group"), subdevice=kwargs.get("subdevice"),
            device_data=device_data, pkl_path=pkl_path,
        )

    def _extract_curves(self, run_data: dict) -> list[CurveData]:
        curves = []
        applied = run_data.get("applied_cols", [])
        measured = run_data.get("measured_cols", [])
        settings = run_data.get("Settings", {})
        colnames = settings.get("Colname", [])
        if isinstance(colnames, str):
            colnames = [colnames]

        for idx, (a_col, m_col) in enumerate(zip(applied, measured)):
            x = np.asarray(run_data.get(a_col, []), dtype=float)
            y = np.asarray(run_data.get(m_col, []), dtype=float)
            if x.size == 0 or y.size == 0:
                continue
            # Display name: use Colname if available, else "sweepN"
            if idx < len(colnames) and colnames[idx]:
                disp = str(colnames[idx])
            else:
                disp = f"sweep{idx + 1}"
            curves.append(CurveData(
                name=disp, x=x, y=y,
                x_unit="V", y_unit="A",
                source_key=f"{a_col}|{m_col}",
            ))

        # Fallback: if no pairs found, try the legacy single pair
        if not curves and "voltage" in run_data and "current" in run_data:
            curves.append(CurveData(
                name="sweep1",
                x=np.asarray(run_data["voltage"], dtype=float),
                y=np.asarray(run_data["current"], dtype=float),
                x_unit="V", y_unit="A",
                source_key="voltage|current",
            ))
        return curves

    def list_measurements(self, path: Path) -> list[tuple[int, str]]:
        lf = self.load(path)
        return [(m.measurement_id, m.title) for m in lf.measurements]

class DataSourceRegistry:
    def __init__(self):
        self._sources: list[DataSource] = []

    def register(self, src: DataSource):
        self._sources.append(src)

    def find(self, path: Path) -> Optional[DataSource]:
        for s in self._sources:
            if s.can_load(path):
                return s
        return None

    def load(self, path: Path, **kw) -> LoadedFile:
        src = self.find(path)
        if src is None:
            raise ValueError(f"No data source can load {path}")
        return src.load(path, **kw)