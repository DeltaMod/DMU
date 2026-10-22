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
    segments: Optional[list[list[int]]] = None   # [[start, end], …] or None
    @property
    def unit_pair(self) -> tuple[str, str]:
        return (self.x_unit, self.y_unit)

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
    device_data: Optional[dict] = None       # reference into the folder store
    pkl_path: Optional[Path] = None
    folder_store: Optional["FolderStore"] = None

    def save(self):
        """
        Deprecated: prefer FolderStore.save_log(run_key, log).
        Kept for external callers that still edit device_data directly.
        Bypasses the store's LoadedFile-mirroring; do not mix with save_log().
        """
        if self.pkl_path is None or self.device_data is None:
            return
        import pickle
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self.device_data, f)
            
# ---------------------------------------------------------------------------
# FolderStore — one per (folder, source); owns the pkl-backed device_data
# and the identity-stable LoadedFile cache. Plain class, not a @dataclass:
# see module notes.
# ---------------------------------------------------------------------------

_STORE_CACHE: dict[tuple[Path, str], "FolderStore"] = {}


def get_folder_store(folder: Path,
                     source_type: str,
                     build_measurement) -> "FolderStore":
    """
    Return (and lazily create) the singleton store for a folder.

    build_measurement: callable(i, run_key, run_data) -> Measurement.
                       Supplied by the DataSource so the store stays
                       source-agnostic.
    """
    key = (Path(folder).resolve(), source_type)
    if key not in _STORE_CACHE:
        _STORE_CACHE[key] = FolderStore(folder, source_type, build_measurement)
    return _STORE_CACHE[key]


def evict_folder_store(folder: Path) -> None:
    """Forget a folder's store (e.g. when the user removes it from the list)."""
    resolved = Path(folder).resolve()
    for key in [k for k in _STORE_CACHE if k[0] == resolved]:
        _STORE_CACHE.pop(key, None)


class FolderStore:
    """
    Owns one folder's pkl-backed `device_data` dict and every LoadedFile
    derived from it.

    Responsibilities:
      * lazy-load the pkl (or build it from raw XLS + LOG if missing)
      * hand out identity-stable LoadedFile instances per file
      * provide a cheap peek() (titles only, no curve extraction)
      * persist LOG edits back to the pkl and update live LoadedFiles
      * reload from disk when the user asks for a full re-sync

    Not thread-safe; the GUI is single-threaded.

    Not a @dataclass — see module notes:
      * `_data` / `_error` / `_loaded_files` are lazy-mutated internals
      * `pkl_path` is derived, not an input
      * identity is guaranteed by get_folder_store() (value semantics
        would be wrong: two callers must share the same device_data).
    """

    def __init__(self, folder: Path, source_type: str, build_measurement):
        self.folder: Path         = Path(folder).resolve()
        self.source_type: str     = source_type
        self.pkl_path: Path       = self.folder / f".dmui_{self.folder.name}.pkl"
        self._build_measurement   = build_measurement

        self._data: Optional[dict]                  = None
        self._loaded_files: dict[Path, LoadedFile]  = {}
        self._error: Optional[str]                  = None

    # -- lazy load ----------------------------------------------------------

    def _ensure(self) -> None:
        """Populate self._data from the pkl, or build it if missing."""
        if self._data is not None:
            return
        try:
            self._data = build_device_data(self.folder)
            self._error = None
        except Exception as exc:
            self._error = f"{type(exc).__name__}: {exc}"
            raise

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def data(self) -> dict:
        """Read-only view. Use save_log()/reload() to change it."""
        self._ensure()
        return self._data

    # -- iteration ----------------------------------------------------------

    def _iter_runs(self):
        """Yield (numports, device, subdevice, run_key, run_data)."""
        assert self._data is not None
        for np_key, devices in self._data.items():
            if np_key == "_meta" or not isinstance(devices, dict):
                continue
            for dev, subs in devices.items():
                for sub, runs in subs.items():
                    for run_key, run_data in runs.items():
                        yield np_key, dev, sub, run_key, run_data

    # -- public API ---------------------------------------------------------

    def peek(self, file_name: str) -> list[tuple[int, str]]:
        """
        Cheap listing: (measurement_id, title) for each run in this file.
        Does not extract curves. Returns [] on a load failure so that
        callers populating a tree can degrade gracefully.
        """
        try:
            self._ensure()
        except Exception:
            return []
        out: list[tuple[int, str]] = []
        i = 0
        for _np, _dev, _sub, run_key, run_data in self._iter_runs():
            if run_data.get("filename") != file_name:
                continue
            title = (run_data.get("Settings") or {}).get("Test Name", "")
            out.append((i, f"{run_key}: {title}".strip(": ")))
            i += 1
        return out

    def load_file(self, path: Path) -> LoadedFile:
        """Return the (cached) LoadedFile for this file, backed by self._data."""
        path = Path(path).resolve()
        if path in self._loaded_files:
            return self._loaded_files[path]

        self._ensure()
        file_name = path.name
        measurements: list[Measurement] = []
        first_dev = first_sub = None
        i = 0
        for _np, dev, sub, run_key, run_data in self._iter_runs():
            if run_data.get("filename") != file_name:
                continue
            if first_dev is None:
                first_dev, first_sub = dev, sub
            measurements.append(self._build_measurement(i, run_key, run_data))
            i += 1

        lf = LoadedFile(
            path=path,
            source_type=self.source_type,
            measurements=measurements,
            device_data=self._data,    # reference, not copy — single truth
            pkl_path=self.pkl_path,
            group=first_dev,
            subdevice=first_sub,
            folder_store=self,
        )
        self._loaded_files[path] = lf
        return lf

    def save_log(self, run_key: str, log: dict) -> bool:
        """
        Persist `log` for `run_key` into device_data and the pkl, and
        mirror the change into any live LoadedFile holding that run.
        Returns True if the run was found.
        """
        self._ensure()
        for _np, _dev, _sub, rk, run_data in self._iter_runs():
            if rk != run_key:
                continue
            log_copy = dict(log)
            run_data["LOG"] = log_copy
            save_device_data(self.folder, self._data)

            # Mirror into live LoadedFiles so callers holding a
            # Measurement reference see the update without reloading.
            for lf in self._loaded_files.values():
                for meas in lf.measurements:
                    if meas.metadata.get("run_key") == run_key:
                        meas.log = dict(log_copy)
                        meas.comment = log_copy.get("Comment", "")
            return True
        return False

    def reload(self, force: bool = False, compare: bool = False,
               overwrite_existing: bool = True) -> None:
        """
        Re-read (or re-build) the pkl from disk and drop every cached
        LoadedFile. Call after any operation that mutates the pkl on disk
        outside this store (Force-rebuild, Full-sync, LOG migration).
        """
        self._loaded_files.clear()
        self._data = build_device_data(
            self.folder,
            force_reload=force,
            compare=compare,
            overwrite_existing=overwrite_existing,
        )

    def invalidate_file(self, path: Path) -> None:
        """Drop one file's cached LoadedFile (rarely needed)."""
        self._loaded_files.pop(Path(path).resolve(), None)
            
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

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() == ".xls" and "LOG" not in path.stem.upper()

    def load(self, path: Path, **kwargs) -> LoadedFile:
        store = get_folder_store(
            path.parent, self.name, self._build_measurement,
        )
        return store.load_file(path)

    def list_measurements(self, path: Path) -> list[tuple[int, str]]:
        store = get_folder_store(
            path.parent, self.name, self._build_measurement,
        )
        return store.peek(path.name)

    # -- measurement builder (unchanged logic, now called by FolderStore) --

    def _build_measurement(self, i: int, run_key: str,
                           run_data: dict) -> Measurement:
        settings = run_data.get("Settings", {}) or {}
        log      = run_data.get("LOG", {}) or {}
        return Measurement(
            measurement_id=i,
            title=f"{run_key}: {settings.get('Test Name', '')}".strip(": "),
            timestamp=settings.get("Execution Time", ""),
            comment=log.get("Comment", ""),
            curves=self._extract_curves(run_data),
            metadata={"run_key": run_key, "settings": settings},
            log=log,
        )


    def _extract_curves(self, run_data: dict) -> list[CurveData]:
        curves = []
        applied  = run_data.get("applied_cols",  []) or []
        measured = run_data.get("measured_cols", []) or []
        settings = run_data.get("Settings", {}) or {}
    
        # Segment boundaries — None means "not segmented" (single sweep)
        raw_segments = run_data.get("sweep_segments", []) or []
        segments = [list(s) for s in raw_segments] if len(raw_segments) > 1 else None
    
        colnames = settings.get("Colname", []) or []
        if isinstance(colnames, str):
            colnames = [colnames]
    
        op_modes = settings.get("Operation Mode", []) or []
        if isinstance(op_modes, str):
            op_modes = [op_modes]
    
        for idx, (a_col, m_col) in enumerate(zip(applied, measured)):
            x = np.asarray(run_data.get(a_col, []), dtype=float)
            y = np.asarray(run_data.get(m_col, []), dtype=float)
            if x.size == 0 or y.size == 0:
                continue
    
            x_unit, y_unit = self._infer_units(m_col, op_modes, idx)
    
            if idx < len(colnames) and colnames[idx]:
                disp = str(colnames[idx])
            else:
                disp = f"sweep{idx + 1}"
    
            curves.append(CurveData(
                name=disp, x=x, y=y,
                x_unit=x_unit, y_unit=y_unit,
                source_key=f"{a_col}|{m_col}",
                segments=[list(s) for s in segments] if segments else None,
            ))
    
        if not curves and "voltage" in run_data and "current" in run_data:
            curves.append(CurveData(
                name="sweep1",
                x=np.asarray(run_data["voltage"], dtype=float),
                y=np.asarray(run_data["current"], dtype=float),
                x_unit="V", y_unit="A",
                source_key="voltage|current",
                segments=[list(s) for s in segments] if segments else None,
            ))
        return curves
    
    
    @staticmethod
    def _infer_units(m_col: str, op_modes: list, idx: int) -> tuple[str, str]:
        m = str(m_col).lower()
        if   "current" in m: y_unit = "A"
        elif "volt"    in m: y_unit = "V"
        elif "time"    in m: y_unit = "s"
        elif "resist"  in m: y_unit = "Ω"
        else:                y_unit = ""
    
        op = ""
        if idx < len(op_modes):
            op = str(op_modes[idx]).lower()
        else:
            for candidate in op_modes:
                if "sweep" in str(candidate).lower():
                    op = str(candidate).lower()
                    break
    
        if   "voltage" in op: x_unit = "V"
        elif "current" in op: x_unit = "A"
        elif y_unit == "A":   x_unit = "V"
        elif y_unit == "V":   x_unit = "A"
        else:                 x_unit = "V"
    
        # Guard: x and y cannot both be V or both be A on a source-measure pair.
        # A contradictory record means the raw data is suspect; prefer the
        # measured column's unit for y, and mark x as unknown rather than lie.
        if x_unit == y_unit and x_unit in ("V", "A"):
            x_unit = ""
    
        return x_unit, y_unit

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