"""
PalmSens data browser / approximate background removal fit.

Three-column PyQt5 layout:
  Left   - data folder list (add/delete); below it, a flat list of every file's
           measurements, grouped under an unselectable grey filename header,
           each row labelled "<measurementID>: <first curve name>"
  Middle - matplotlib canvas + toolbar
  Right  - multi-selection panel showing either fitting, plotting or advanced settings. 
           display: alter fit type, figure size, font size, etc. More functionality can be added later.
           fit: background range list (+/- to add/remove, click a row to highlight it green), fit, remove-background toggle, annotated/raw plot
           toggle (global, not per-file), save figure

Data Import model, from palmsense documentation.
----------------------------------------
    for i, file in enumerate(data_files):
        DATA = ps.load_session_file(file)
        for j, meas in enumerate(DATA):
            plot_psense(DATA, measurementID=j, curveID="all")

Clicking a measurement in a new data-file row loads the *whole* file via ps.load_session_file()
otherwise, cached data is used for enhanced speed. A big no-no is to load ALL the data at once, this would not spark joy.

Design Principles
------------
* NEVER modify the raw data.
  All operator state (background ranges, fit coefficients, toggle states) lives in a small JSON sidecar per
  (file, measurementID) pair, in a ".psense_gui_state" folder next to the data
  files. This means other users can add the data folder, and receive another users modifications without issue.
* currently, we set the curveID plot to show all, but I hope to add a tab-selection for specific data. More "if this data type" shinanegans.
  Since fits will only be done on CA data, for now, we do not care that more data can be found elsewhere as it does not need background fits.
* Curve x/y arrays are pulled off the Line2D object that `curve.plot(ax=ax)`
  creates, drawn on a disposable scratch axes so it never touches the real plot.
* There is a global option to draw the comment that was made during the measurement. 
  This also crudely adds a fileID and measurementID for reference outside the GUI.
* On-screen display size and exported-figure size/DPI are independent -
  export always uses the figure width and height specified in the UI, regardless of what stretch option is enabled.
"""

import os
import re
import sys
import json
import textwrap
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from typing import Optional
import io 
import numpy as np
import shutil
import time 
import platform
import pickle
# Must be set before QApplication is constructed to avoid pyQT5 high DPI scaling.
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector
from matplotlib.ticker import ScalarFormatter, FixedLocator, FuncFormatter
from matplotlib.legend_handler import HandlerTuple

try:
    from .Widgets import *
    from .Plotting_Tools import *
    from .Data_Containers import *
    from .Helpers import *
    from . import data_import_tools as dit
    from . import plot_utils as dmp
except ImportError:
    from Widgets import *
    from Plotting_Tools import *
    from Data_Containers import *
    from Helpers import *
    from DMU import data_import_tools as dit
    from DMU import plot_utils as dmp

### Scipy imports
from scipy.interpolate import splrep, splev
from scipy.optimize import curve_fit
  
# ---------------------------------------------------------------------------
# pypalmsens integration
# ---------------------------------------------------------------------------

PSSESSION_GLOB = "*.pssession"  # file pattern shown in the file list


def load_session_file(filepath: Path):
    """Load a .pssession file and return the list of measurements (DATA)."""
    import pypalmsens as ps
    return ps.load_session_file(str(filepath))


# ---------------------------------------------------------------------------
# Per-measurement operator state, persisted as JSON, never touches raw data
# ---------------------------------------------------------------------------

def state_key(filename_or_relpath: str, measurement_id: int) -> str:
    """Build a filesystem-safe key. Accepts either a bare filename or a
    path relative to the state root; separators are folded to '__'."""
    safe = str(filename_or_relpath).replace("\\", "__").replace("/", "__")
    return f"{safe}__m{measurement_id}"

def file_path_match(a, b) -> bool:
    """Compare two file paths tolerantly (None-safe, str/Path-safe)."""
    if a is None or b is None:
        return False
    return Path(a).resolve() == Path(b).resolve()
@dataclass
class CurveState:
    """
     ── Per-curve fit data (framework) ──────────────────────────────────
     Keys are _fit_key(curve_name, segments). Values are dicts of the form:
       {
         "mode": int,                # FitMode enum
         "coefficients": list|None,  # what the fit produced (poly coeffs, etc.)
         "use_bg_removed": bool,     # fit input was bg-subtracted or raw
         "x_range": [xmin, xmax]|None,  # x domain of the fit
         "segments": list[int],      # redundant copy of key's segment list
         "extra": dict,              # future per-mode params
         "display_x": list|None,     # optional precomputed fit points
         "display_y": list|None,     # (populated by fit code; plotted verbatim)
       }
    
     fit_by_curve is for the *curve* fit (e.g. diode / poly shape).
     bg_fit_by_curve is for the BG-removal fit — separate storage because
     they answer different questions and can coexist.
    """
    hidden_curves: list[str] = field(default_factory=list)
    # Per-curve-segment visibility: key = curve name, value = list of segment
    # indices hidden from the plot. Empty dict or missing entry = all shown.
    hidden_segments: dict[str, list[int]] = field(default_factory=dict)
    bg_ranges_by_curve: dict[str, list] = field(default_factory=dict)
    anchor_points_by_curve: dict[str, list] = field(default_factory=dict)
    curve_overrides: dict[str, dict] = field(default_factory=dict)

    # Global fit settings
    fit_coeffs: Optional[list] = None
    remove_background: bool = False
    bg_removal_mode: int = 0
    poly_degree: int = 2
    spline_smoothing: float = 0.0

    bg_ranges: list = field(default_factory=list)
    anchor_points: list = field(default_factory=list)

    
    fit_by_curve: dict[str, dict] = field(default_factory=dict)
    bg_fit_by_curve: dict[str, dict] = field(default_factory=dict)

    # Which (curve, segment-selection) the fit panel is currently targeting.
    # None = nothing selected (fit button / display disabled).
    active_fit_key: Optional[str] = None
    # Optional explicit plot layout. None = "auto" (use meas.curves as
    # before). [] = nothing. List of dicts = each is one line on the plot:
    #   {"x_column": "Time", "y_column": "Current",
    #    "segments": [], "label": "I vs t", "axis": "auto"}
    #   - segments: [] means "full array", [0] means "forward only", [0,1] both
    #   - axis: "auto" | "left" | "right"   (auto = one axis per y-unit)
    plot_template: Optional[list[dict]] = None
    # True once plot_template has been set by either the user, an explicit
    # "Reset to automatic", or an auto-apply from the library. Prevents
    # library templates from re-clobbering the user's choice on next load.
    plot_template_auto_applied: bool = False
    
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclass_fields(cls)}
        inst = cls(**{k: v for k, v in d.items() if k in known})
        # Migrate legacy flat ranges onto first curve
        if inst.bg_ranges and not inst.bg_ranges_by_curve:
            inst.bg_ranges_by_curve["__first__"] = inst.bg_ranges
            inst.anchor_points_by_curve["__first__"] = inst.anchor_points
            inst.bg_ranges = []
            inst.anchor_points = []
        return inst
    
class StateStore:
    """One JSON sidecar per (file, measurementID), in <folder>/.dmui_gui_state/"""
    def __init__(self, folder: Path):
        self.state_dir = Path(folder) / ".dmui_gui_state"
        self.state_dir.mkdir(exist_ok=True)

    def _path_for(self, key: str) -> Path:
        return self.state_dir / (key + ".json")

    def load(self, key: str) -> CurveState:
        p = self._path_for(key)
        if p.exists():
            with open(p) as f:
                return CurveState.from_dict(json.load(f))
        return CurveState()

    def save(self, key: str, state: CurveState):
        with open(self._path_for(key), "w") as f:
            json.dump(state.to_dict(), f, indent=2)

def _measurement_signature(meas: "Measurement") -> str:
    """Structural identity for template matching.

    Two measurements with the same column set are the same 'shape' of
    data, and any template that works on one works on the other.
    """
    return "|".join(sorted(meas.columns.keys()))

# ---------------------------------------------------------------------------
# Small config: remembers which folders you've added, across sessions. Change this to a local directory later.
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".config" / "DMUI"
WORKSPACE_PATH = CONFIG_PATH/ "workspace.json"
DISPLAY_PATH = CONFIG_PATH / "display.json"
RCPARAMS_OVERRIDE_PATH = CONFIG_PATH / "rcparams_override.json"

def load_workspace():
    """Return the workspace dict.
    Shape:
        {
          "folders": [
              {"path": "/abs/path/to/folder", "checked": true|false},
              ...
          ],
          "ticked_files": ["/abs/path/to/file", ...]
        }
    """
    if not WORKSPACE_PATH.exists():
        return {"folders": [], "ticked_files": []}
    with open(WORKSPACE_PATH) as f:
        data = json.load(f)
    data.setdefault("folders", [])
    data.setdefault("ticked_files", [])
    return data


def save_workspace(workspace: dict):
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    with open(WORKSPACE_PATH, "w") as f:
        json.dump(workspace, f, indent=2)


# ---------------------------------------------------------------------------
# rcParams override file: a dedicated, explicit "startup defaults" layer.
# Separate from display.json (which remembers raw EDIT_* field values) -
# this file is only ever written by the "Save" button on the Display tab,
# only ever re-applied to the running session by "Load", and only ever
# cleared by "Reset". It is loaded automatically once at startup so that
# whatever was last explicitly saved here becomes the new baseline on top of
# get_rcparam_defaults().
# ---------------------------------------------------------------------------

def load_rcparams_override():
    if RCPARAMS_OVERRIDE_PATH.exists():
        with open(RCPARAMS_OVERRIDE_PATH) as f:
            return json.load(f)
    return {}


def save_rcparams_override(overrides: dict):
    RCPARAMS_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RCPARAMS_OVERRIDE_PATH, "w") as f:
        json.dump(overrides, f, indent=2, default=str)


def delete_rcparams_override():
    if RCPARAMS_OVERRIDE_PATH.exists():
        RCPARAMS_OVERRIDE_PATH.unlink()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

HEADER_BG = QtGui.QColor("#c8c8c8")

# BG REMOVAL options
class BgRemovalMode:
    LINEAR = 0           # Linear fit across selected ranges
    POLY = 1             # Polynomial fit (user-selectable degree)
    SPLINE = 2           # Spline interpolation through selected points
    CONNECT_END = 3      # Connect end of one range to start of next with straight line
    EXPONENTIAL_DECAY = 4  # Fit exponential decay through baseline points
    MANUAL_GRADIENT = 5   # Manual piecewise gradient construction

class FitMode:
    """Curve-fit algorithm identifier. Extend as real fits are added."""
    NONE = 0
    LINEAR = 1
    POLY = 2
    # Future: DIODE, IDEAL_DIODE, EXPONENTIAL, SPLINE, CUSTOM


def _fit_key(curve_name: str, segments=None) -> str:
    """Stable key for a (curve, segment-selection) fit target.

    segments=None or [] → "all segments".
    Otherwise, sorted segment indices comma-joined.

        _fit_key("Sweep 1", None)   -> "Sweep 1|all"
        _fit_key("Sweep 1", [0])    -> "Sweep 1|0"
        _fit_key("Sweep 1", [0, 1]) -> "Sweep 1|0,1"
    """
    if not segments:
        return f"{curve_name}|all"
    return f"{curve_name}|{','.join(str(s) for s in sorted(segments))}"


def _parse_fit_key(key: str) -> tuple[str, list[int]]:
    """Inverse of _fit_key: returns (curve_name, segment_indices)."""
    if "|" not in key:
        return key, []
    name, segs = key.rsplit("|", 1)
    if segs == "all":
        return name, []
    try:
        return name, [int(s) for s in segs.split(",")]
    except ValueError:
        return name, []
    
# BG REMOVAL span types 
class RangeType:
    SPAN = 0          # Background range to remove
    ANCHOR = 1        # Manual anchor point (stored as [x, y] in separate list)

                
class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Browser")
        self.resize(1300, 800)
        
        self.current_folder: Optional[Path] = None

        # ── Added folders / roots, plus their tick state ─────────────
        self._added_entries: list[dict] = []      # {"path", "kind"}
        self._leaf_check: dict[str, bool] = {}    # abs_leaf_path -> bool
        self._state_roots: dict[str, str] = {}    # abs_leaf_path -> abs_root
        
        # Deferred-rebuild bookkeeping — see _schedule_tree_rebuild().
        self._tree_rebuild_pending: bool = False
        self._pending_focus: Optional[str] = None
        self._in_tree_rebuild: bool = False
        
        self.current_file_path: Optional[str] = None
        self.loaded_file: Optional[LoadedFile] = None   # current LoadedFile
        self.loaded_files: dict[str, LoadedFile] = {}   # cache of every ticked file                                # ps.load_session_file() result for current_file_path

        self.current_file_index: Optional[int] = None
        self.current_meas_index: Optional[int] = None
        self.current_filename: Optional[str] = None
        
        # Ticked-measurement model (file_path, meas_index) -> ticked.
        self._ticked_measurements_from_workspace: set = set()
        self.device_registry: dict = {}
        
        ## -- plotting parameters:
        
        self.RCPARAM_DEFAULTS = get_rcparam_defaults()
        self.UI_RCPCONFIG_MAP = {
            # Figure - index for list items
            'EDIT_FIGSIZE_X': ('figure.figsize', 0),
            'EDIT_FIGSIZE_Y': ('figure.figsize', 1),
            'EDIT_FIGURE_DPI': ('figure.dpi', None),
            'EDIT_COMMENT_WIDTH_FACTOR': None,  # Not an rcParam
            
            # Font sizes
            'EDIT_TEXT_HEADER_SIZE': ('axes.titlesize', None),
            'EDIT_TEXT_BODY_SIZE': ('font.size', None),
            'EDIT_LEGEND_HEADER_SIZE': ('legend.title_fontsize', None),
            'EDIT_LEGEND_BODY_SIZE': ('legend.fontsize', None),
            'EDIT_COMMENT_HEADER_SIZE': None,  # Not an rcParam
            'EDIT_COMMENT_BODY_SIZE': None,    # Not an rcParam
            'EDIT_AXIS_LABEL_SIZE': ('axes.labelsize', None),
            'EDIT_TICK_SIZE': ('xtick.labelsize', None),
            # Line styles
            'EDIT_LINEWIDTH': ('lines.linewidth', None),
            'EDIT_MARKERSIZE': ('lines.markersize', None),
            'EDIT_AXES_LINEWIDTH': ('axes.linewidth', None),
        }
        
        self._toggle_widgets = {}   # variable_name -> QPushButton (checkable)
        self.UI_ONLY_DEFAULTS = {
            'EDIT_COMMENT_WIDTH_FACTOR': 1.3,
            'EDIT_COMMENT_HEADER_SIZE': 14.0,
            'EDIT_COMMENT_BODY_SIZE': 8,
            'EDIT_TOGGLE_STRETCH': 0,
            'EDIT_SHOW_COMMENT': False,
            'EDIT_HIDE_SPANS': False,
            
            #Plot Labels and Titles
            'EDIT_XLABEL_OVERRIDE':None,
            'EDIT_YLABEL_OVERRIDE':None,
            'EDIT_SHOW_TITLE':True,
            'EDIT_TITLE_OVERRIDE':None,
            'EDIT_BOUNDS_LEFT':  0.1,
            'EDIT_BOUNDS_RIGHT': 0.98,
            'EDIT_BOUNDS_TOP':   0.92,
            'EDIT_BOUNDS_BOTTOM':0.125,
            'EDIT_XMIN':None,
            'EDIT_XMAX':None,
            'EDIT_YMIN':None,
            'EDIT_YMAX':None,
             #Window Persistance
            'EDIT_WINDOW_WIDTH': 1300,
            'EDIT_WINDOW_HEIGHT': 800,
            'EDIT_WINDOW_X': 0,
            'EDIT_WINDOW_Y': 0,
            'EDIT_SPLITTER_SIZES': [384, 780, 384],
            #LegendConstruction
            'EDIT_LEGEND_SHOW_RUN':       True,
            'EDIT_LEGEND_SHOW_DEVICE':    True,
            'EDIT_LEGEND_SHOW_SUBDEVICE': True,
            'EDIT_LEGEND_SHOW_MEASURED':  True,
            'EDIT_LEGEND_SHOW_SWEEP':     True,
        }
        
        
        self.current_state: CurveState = CurveState()
        self.raw_xy_list = []       # [(x, y), ...] one per curve in the current measurement, cached on plot
        self.span_selector = None
        # Data source registry — one place where new file types get plugged in.
        self.registry = DataSourceRegistry()
        self.registry.register(PalmSensSource())
        self.registry.register(KeithleySource())

        
        self._selected_row: Optional[int] = None  # row currently highlighted in the range list

        # Folders auto-ticked by a row-click.
        # Cleared/updated on the next folder focus.
        self._temp_ticked_folders: set[str] = set()
        
        # Advanced option: refuse to load more than one file at a time.
        self.EDIT_DISABLE_MULTILOAD: bool = False

        # --- filter model ---
        self._filter_enabled: bool = False
        self._filter_mode: str = "AND"                     # "AND" | "OR"
        self._filter_conditions: list[dict] = []           # [{field, operator, value, enabled}, …]
        self._filter_field_types: dict[str, set] = {}      # field_name -> {type names seen}
        self._filter_tab_index: Optional[int] = None
        # Per-folder log cache for pkl fallback (see _get_log_for_run).
        # Populated lazily; cleared on tree rebuild.
        self._filter_log_cache: dict[str, dict] = {}
        
        # --- multi-selection model ---
        # Every highlighted measurement (file_path, meas_index), in click order.
        self.selected_measurements: list[tuple[str, int]] = []
        # The most-recently highlighted one. Tabs and edits target this.
        self.active_measurement: Optional[tuple[str, int]] = None
        # Toggle: draw only active, or draw every selected.
        self.EDIT_MULTIPLOT: bool = True
        
        # --- classification view ---
        self.EDIT_TREE_VIEW: str = "flat"          # "flat" | "classified"
        self.EDIT_CLASSIFY_PATTERN: str = r"(?P<device>[^_\-]+)[_\-](?P<subdevice>[^_\.]+)"
        self._classify_cache: dict[str, tuple[str, str]] = {}   # file_path -> (dev, sub)

        self._init_session_variables() #This references the UI_ONLY_DEFAULTS and UI_RCPCONFIG_MAP
        
        # User rcParams overrides - loaded from the dedicated override file (see RCPARAMS_OVERRIDE_PATH)
        self._user_rcparams = load_rcparams_override()
        self._sync_edit_vars_from_rcparams(self._user_rcparams)
        self._apply_all_rcparams()
        
        self._build_ui()
        self._restore_folders()
        self._load_session_parameters()
        self._apply_figsize(self.EDIT_SHOW_COMMENT)  
        self._update_canvas_size_policy()
        self._refresh_plot()   
    
    def get_default_for_edit(self,edit_name):
        """Get the default value for an EDIT_* variable from rcParams or UI_ONLY_DEFAULTS"""
        # Check if it's a UI-only value first
        if edit_name in self.UI_ONLY_DEFAULTS:
            return self.UI_ONLY_DEFAULTS[edit_name]
        
        # Otherwise get from rcParams
        mapping = self.UI_RCPCONFIG_MAP.get(edit_name)
        if mapping is None:
            return None
        
        rc_key, index = mapping
        
        if rc_key not in self.RCPARAM_DEFAULTS:
            return None
        
        value = self.RCPARAM_DEFAULTS[rc_key]
        if index is not None and isinstance(value, (list, tuple)):
            return value[index]
        return value
    
    def _init_session_variables(self):
        """Initialize all EDIT_* variables from rcParams or UI_ONLY_DEFAULTS"""
        # rcParam-linked variables 
        for edit_name in self.UI_RCPCONFIG_MAP.keys():
            default_value = self.get_default_for_edit(edit_name)
            if default_value is not None:
                setattr(self, edit_name, default_value)
                setattr(self, f"DEFAULT_{edit_name}", default_value)
            else:
                setattr(self, edit_name, 0)
                setattr(self, f"DEFAULT_{edit_name}", 0)
        
        # UI-only variables 
        for edit_name, default_value in self.UI_ONLY_DEFAULTS.items():
            setattr(self, edit_name, default_value)
            setattr(self, f"DEFAULT_{edit_name}", default_value)
        

                
    def _apply_all_rcparams(self):
        """rcParams are applied in the following order: HARDCODED Defaults -> UI defined defaults -> user defined overrides"""
        
        #HARDCODED Defaults
        mpl.rcParams.update(self.RCPARAM_DEFAULTS)
        
        #UI Defaults
        for edit_name, mapping in self.UI_RCPCONFIG_MAP.items():
            if mapping is None:
                continue  
            rc_key, index = mapping
            value = getattr(self, edit_name, None)
            if value is None:
                continue
            if index is not None:
                # Some entries are lists, and we must handle those differently 
                current = mpl.rcParams.get(rc_key, [])
                if isinstance(current, list) and len(current) > index:
                    current[index] = value
                    mpl.rcParams[rc_key] = current
            else:
                mpl.rcParams[rc_key] = value
    
        #USER overrides
        if hasattr(self, '_user_rcparams') and self._user_rcparams:
            mpl.rcParams.update(self._user_rcparams)
            
    def _update_ui_from_config(self):
        """Update all UI elements from EDIT_* variables"""

        for widget_name in ['save_width_row', 'save_height_row', 'dpi_row', 'width_factor_row']:
            if hasattr(self, widget_name):
                widget = getattr(self, widget_name)
                param_name = widget.param_name
                if hasattr(self, param_name):
                    widget.line_edit.setText(str(getattr(self, param_name)))
        
        # Update MultiParamRow
        if hasattr(self, 'font_sizes_row'):
            values = {}
            for param_name in self.font_sizes_row.line_edits.keys():
                if hasattr(self, param_name):
                    values[param_name] = getattr(self, param_name)
            self.font_sizes_row.set_values(values)
        
        self._update_button_text()  
    
    def _get_default_title(self):
         """Return the default title for the current measurement."""
         if self.loaded_file is not None and self.current_meas_index is not None:
             dat = self.loaded_file.measurements[self.current_meas_index]
             title = getattr(dat, "title", "")
             if not title:
                 title = self.current_filename or "Untitled"
             return title
         return "Title"
    
    def _toggle_show_title(self, checked):
        self.EDIT_SHOW_TITLE = checked
        self._refresh_plot()
        self._save_session_parameters()
    
    def _update_override_placeholders(self):
        """Update placeholder texts for all OverrideParamRow widgets."""
        if hasattr(self, 'xlabel_override_row'):
            self.xlabel_override_row.update_placeholder_text()
        if hasattr(self, 'ylabel_override_row'):
            self.ylabel_override_row.update_placeholder_text()
        if hasattr(self, 'title_override_row'):
            self.title_override_row.update_placeholder_text()
    
    def _sync_override_rows(self):
        """Update the override line edits from the current EDIT_* attributes."""
        for row in (self.xlabel_override_row, self.ylabel_override_row, self.title_override_row):
            if row is not None:
                val = getattr(self, row.param_name, None)
                row.line_edit.setText(val if val is not None else "")
                row._update_placeholder()   # refresh grey placeholder text
    # -- Data-management actions ---------------------------------------------

    def _get_ticked_folders(self) -> list[Path]:
        out: list[Path] = []
        for i in range(self.folder_tree.topLevelItemCount()):
            self._collect_ticked_leaves(
                self.folder_tree.topLevelItem(i), out)
        return out

    def _collect_ticked_leaves(self, item, out: list[Path]):
        if item is None:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload and payload[0] == "folder_leaf" \
                and item.checkState(0) == QtCore.Qt.Checked:
            out.append(Path(payload[1]))
            return
        for j in range(item.childCount()):
            self._collect_ticked_leaves(item.child(j), out)
    

    
    def _convert_logs_to_json(self):
        """Convert xls/txt logs to strict JSON and seed the GUI copy."""

        folders = self._dm_target_folders()
        if not folders:
            QtWidgets.QMessageBox.information(
                self, "No folders", "Tick at least one data folder first.")
            return

        converted, seeded, failed = 0, 0, []
        touched: list[Path] = []
        for folder in folders:
            folder_touched = False
            try:
                written = dit.migrate_logs_to_json(folder)
                converted += len(written)
                if written:
                    folder_touched = True
            except Exception as e:
                failed.append(f"{folder.name}: convert failed — {e}")

            source = dit.find_log_file(folder)
            if source is not None and dit.find_gui_log_file(folder) is None:
                try:
                    shutil.copyfile(source, dit.gui_log_path(source))
                    seeded += 1
                    folder_touched = True
                except OSError as e:
                    failed.append(f"{folder.name}: GUI seed failed — {e}")

            if folder_touched:
                touched.append(folder)

        if touched:
            self._evict_loaded_files_for(touched)

        msg = (f"Converted {converted} file(s); seeded {seeded} GUI copy(ies) "
               f"across {len(folders)} folder(s).")
        if failed:
            msg += "\n\nFailures:\n" + "\n".join(failed)
        QtWidgets.QMessageBox.information(self, "Conversion complete", msg)
        
    def _on_dm_target_changed(self):
        if not hasattr(self, "_dm_target_hint"):
            return
        if self._dm_target_one.isChecked():
            self._dm_target_hint.setText(
                "Operations apply only to the currently-focused folder "
                "(highlighted in the folder tree on the left).")
        else:
            self._dm_target_hint.setText(
                "Operations apply to every ticked folder in the folder "
                "tree on the left.")

    def _dm_target_folders(self) -> list[Path]:
        """Return the folder list the Data Management buttons should act on."""
        if (hasattr(self, "_dm_target_one")
                and self._dm_target_one.isChecked()):
            if self.current_folder is None:
                QtWidgets.QMessageBox.information(
                    self, "No folder focused",
                    "Click a folder in the folder tree first, or switch "
                    "the target back to 'All ticked folders'.")
                return []
            return [Path(self.current_folder)]
        return self._get_ticked_folders()
    
    def _add_new_measurements(self):
        self._run_sync(overwrite_existing=False,
                       title="Add new measurements")
        # -- LOG merge / regenerate / migration --------------------------------

    def _regenerate_gui_log(self):
        """Overwrite each target folder's GUI copy from its source."""
        

        folders = self._dm_target_folders()
        if not folders:
            QtWidgets.QMessageBox.information(
                self, "No folders", "Tick at least one data folder first.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Regenerate GUI LOG",
            f"Overwrite _LOGBOOK_GUI.json for {len(folders)} folder(s) "
            f"from their _LOGBOOK.json?\n\n"
            f"All GUI-side edits in these folders will be LOST.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        touched, failed = [], []
        for folder in folders:
            source = dit.find_log_file(folder)
            if source is None:
                continue
            target = dit.gui_log_path(source)
            try:
                shutil.copyfile(source, target)
                touched.append(folder)
            except OSError as e:
                failed.append(f"{folder.name}: {e}")

        if touched:
            self._evict_loaded_files_for(touched)

        msg = f"Regenerated GUI LOG for {len(touched)} folder(s)."
        if failed:
            msg += "\n\nFailures:\n" + "\n".join(failed)
        QtWidgets.QMessageBox.information(self, "Regenerate complete", msg)

    def _export_gui_log_to_source(self):
        """Promote GUI file to be the new source file (with backup)."""


        folders = self._dm_target_folders()
        if not folders:
            QtWidgets.QMessageBox.information(
                self, "No folders", "Tick at least one data folder first.")
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Export GUI LOG → source",
            f"Replace _LOGBOOK.json with _LOGBOOK_GUI.json for "
            f"{len(folders)} folder(s)?\n\n"
            f"The old source is backed up to _LOGBOOK_BACKUP.json "
            f"in each folder.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        touched, failed = [], []
        for folder in folders:
            source = dit.find_log_file(folder)
            gui = dit.find_gui_log_file(folder)
            if source is None or gui is None:
                continue
            backup = source.with_name(
                f"{source.stem}_BACKUP_{int(time.time())}{source.suffix}")
            try:
                source.rename(backup)
                shutil.copyfile(gui, source)
                touched.append(folder)
            except OSError as e:
                failed.append(f"{folder.name}: {e}")

        if touched:
            self._evict_loaded_files_for(touched)

        msg = f"Promoted GUI LOG → source in {len(touched)} folder(s)."
        if failed:
            msg += "\n\nFailures:\n" + "\n".join(failed)
        QtWidgets.QMessageBox.information(self, "Export complete", msg)

    def _compare_and_merge_logs(self):
        """Three-tier merge: global mode → per-run review for differing runs."""
        folders = self._dm_target_folders()
        if not folders:
            QtWidgets.QMessageBox.information(
                self, "No folders", "Tick at least one data folder first.")
            return

        work_items = []
        for folder in folders:
            source = dit.find_log_file(folder)
            if source is None:
                continue
            try:
                src_data = dit.load_log_file(source)
            except Exception as e:
                logger.error(f"Could not read source log in {folder}: {e}")
                continue

            gui = dit.find_gui_log_file(folder)
            gui_data = {}
            if gui is not None:
                try:
                    gui_data = dit.load_log_file(gui)
                except Exception as e:
                    logger.error(f"Could not read GUI log in {folder}: {e}")

            diff = dit.compare_logs(src_data, gui_data, folder=folder)
            if (diff["new_runs"] or diff["differing_runs"]
                    or diff["only_in_gui"]):
                work_items.append((folder, diff, src_data, gui_data,
                                   source, gui))

        if not work_items:
            QtWidgets.QMessageBox.information(
                self, "Already in sync",
                "Source and GUI logbooks are already in sync across "
                "every target folder.")
            return

        touched: list[Path] = []
        results: list[tuple[str, str]] = []   # (folder_name, summary)
        for folder, diff, src_data, gui_data, source, gui in work_items:
            mode = self._dialog_merge_mode(folder, diff)
            if mode is None:
                results.append((folder.name, "cancelled"))
                continue

            per_run_choices: dict = {}
            n_reviewed = n_skipped = 0
            if mode == "reviewed":
                for run_key in sorted(diff["differing_runs"].keys()):
                    choice = self._dialog_merge_run(
                        run_key, diff["differing_runs"][run_key])
                    if choice is None:
                        n_skipped += 1
                        continue
                    per_run_choices[run_key] = choice
                    n_reviewed += 1

            merged = dit.merge_logs(src_data, gui_data, mode,
                                    per_run_choices, folder=folder)

            target = gui if gui is not None else dit.gui_log_path(source)
            try:
                with open(target, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False,
                              default=str)
                touched.append(folder)
                extra = ""
                if mode == "reviewed":
                    extra = f"; {n_reviewed} reviewed, {n_skipped} skipped"
                results.append((
                    folder.name,
                    f"{mode} (+{len(diff['new_runs'])} new, "
                    f"{len(diff['differing_runs'])} compared{extra})"
                ))
            except OSError as e:
                results.append((folder.name, f"write failed — {e}"))

        if touched:
            self._evict_loaded_files_for(touched)

        if not touched:
            return

        summary = "\n".join(f"  • {name}: {status}" for name, status in results)
        reply = QtWidgets.QMessageBox.question(
            self, "Merge complete",
            f"Updated {len(touched)} folder(s):\n\n{summary}\n\n"
            f"Rebuild the pkl(s) now to apply?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._rebuild_pkl()

    def _dialog_merge_mode(self, folder, diff) -> Optional[str]:
        """Tier 1 — global mode for one folder. Returns None on cancel."""
        n_new   = len(diff["new_runs"])
        n_diff  = len(diff["differing_runs"])
        n_orphan = len(diff["only_in_gui"])

        summary = (
            f"Folder: {folder.name}\n\n"
            f"  • {n_new} new run(s) in source\n"
            f"  • {n_diff} run(s) with field differences\n"
            f"  • {n_orphan} run(s) only in GUI (will be preserved)\n\n"
            f"How do you want to merge?"
        )

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Merge LOGs")
        box.setText(summary)
        box.setIcon(QtWidgets.QMessageBox.Question)

        b_overwrite = box.addButton("Overwrite all",      QtWidgets.QMessageBox.DestructiveRole)
        b_addonly   = box.addButton("Only add new",       QtWidgets.QMessageBox.AcceptRole)
        b_keepgui   = box.addButton("Keep GUI",           QtWidgets.QMessageBox.AcceptRole)
        b_review    = box.addButton("Review each run…",   QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)

        box.exec_()
        clicked = box.clickedButton()
        if clicked is b_overwrite: return "overwrite_all"
        if clicked is b_addonly:   return "add_only"
        if clicked is b_keepgui:   return "keep_gui"
        if clicked is b_review:    return "reviewed"
        return None

    def _dialog_merge_run(self, run_key, run_diff):
        """Tier 2/3 — one dialog per differing run, showing all fields."""
        fields = run_diff["fields"]

        dlg = QtWidgets.QDialog(self)
        fname, run = run_key
        dlg.setWindowTitle(f"{fname} :: {run} — {len(fields)} field(s) differ")
        dlg.resize(560, 400)

        outer = QtWidgets.QVBoxLayout(dlg)
        outer.addWidget(QtWidgets.QLabel(
            "Pick a value for each field. 'Apply' uses your choices; "
            "'Whole run → source' / '→ GUI' shortcut the picks below."))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(body)

        choices: dict[str, QtWidgets.QButtonGroup] = {}
        for field, (gui_val, src_val) in fields.items():
            group_box = QtWidgets.QGroupBox(field)
            gb_layout = QtWidgets.QVBoxLayout(group_box)

            grp = QtWidgets.QButtonGroup(group_box)
            rb_gui = QtWidgets.QRadioButton(f"Keep GUI:    {gui_val!r}")
            rb_src = QtWidgets.QRadioButton(f"Use source:  {src_val!r}")
            rb_gui.setChecked(True)
            grp.addButton(rb_gui, 0)
            grp.addButton(rb_src, 1)
            gb_layout.addWidget(rb_gui)
            gb_layout.addWidget(rb_src)

            form.addWidget(group_box)
            choices[field] = grp

        form.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        b_src_all  = QtWidgets.QPushButton("Whole run → source")
        b_gui_all  = QtWidgets.QPushButton("Whole run → GUI")
        b_apply    = QtWidgets.QPushButton("Apply")
        b_cancel   = QtWidgets.QPushButton("Cancel")
        btn_row.addWidget(b_src_all)
        btn_row.addWidget(b_gui_all)
        btn_row.addStretch(1)
        btn_row.addWidget(b_apply)
        btn_row.addWidget(b_cancel)
        outer.addLayout(btn_row)

        result = {"value": None}

        def _apply():
            result["value"] = {
                f: ("src" if grp.checkedId() == 1 else "gui")
                for f, grp in choices.items()
            }
            dlg.accept()

        b_src_all.clicked.connect(
            lambda *_: (result.__setitem__("value", "src"), dlg.accept()))
        b_gui_all.clicked.connect(
            lambda *_: (result.__setitem__("value", "gui"), dlg.accept()))
        b_apply.clicked.connect(lambda *_: _apply())
        b_cancel.clicked.connect(lambda *_: dlg.reject())

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return result["value"]

        
    def _rebuild_pkl(self):
        """Single, safe rebuild path. Re-derives each folder's pkl from
        XLS + _LOGBOOK_GUI.json. Cannot lose GUI LOG edits (they live in
        the GUI file, not the pkl)."""
        self._run_sync(overwrite_existing=True, title="Rebuild pkl")
    
    def _run_sync(self, overwrite_existing: bool, title: str):
        folders = self._dm_target_folders()
        if not folders:
            QtWidgets.QMessageBox.information(
                self, "No folders", "Tick at least one data folder first.")
            return
        ok, failed = 0, []
        for folder in folders:
            try:
                dit.build_device_data(
                    folder,
                    force_reload=True,
                    compare=True,
                    overwrite_existing=overwrite_existing,
                )
                ok += 1
            except Exception as e:
                failed.append(f"{folder.name}: {e}")
        self._evict_loaded_files_for(folders)
        msg = f"{title}: synced {ok} folder(s)."
        if failed:
            msg += "\n\nFailures:\n" + "\n".join(failed)
        QtWidgets.QMessageBox.information(self, title, msg)
    
    
    def _evict_loaded_files_for(self, folders):
        """Drop in-memory LoadedFile + FolderStore instances for those folders."""
        roots = [str(Path(f).resolve()) for f in folders]
        drop = []
        for fp in list(self.loaded_files.keys()):
            try:
                rp = str(Path(fp).resolve())
            except Exception:
                continue
            if any(rp.startswith(r + os.sep) or rp == r for r in roots):
                drop.append(fp)
        for fp in drop:
            del self.loaded_files[fp]
        # ── NEW: also evict the FolderStore so _ensure re-reads the pkl ──
        for f in folders:
            evict_folder_store(Path(f))
        # Reset active pointer if we just evicted it.
        if self.current_file_path in drop:
            self._reset_loaded_data()
        self._rebuild_measurements_tree()
    
    def _update_device_registry(self):
        folders = self._get_ticked_folders()
        if not folders:
            QtWidgets.QMessageBox.information(
                self, "No folders", "Tick at least one data folder first.")
            return
        registry = {}
        for folder in folders:
            if not folder.is_dir():
                continue
            pkl_path = folder / f".dmui_{folder.name}.pkl"
            for f in folder.iterdir():
                if not f.is_file():
                    continue
                if self.registry.find(f) is None:
                    continue
                dev, sub = self._classify_path(f)
                registry[str(f)] = {
                    "device": dev,
                    "subdevice": sub,
                    "folder": str(folder),
                    "pkl_present": pkl_path.exists(),
                }
        self.device_registry = registry
        if hasattr(self, "_registry_status"):
            self._registry_status.setText(
                f"Registry: {len(registry)} file(s) indexed."
            )
        QtWidgets.QMessageBox.information(
            self, "Registry updated",
            f"Indexed {len(registry)} data file(s) across {len(folders)} folder(s)."
        )
    
    # -- UI construction ----------------------------------------------------
    def _build_ui(self):
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
    
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(self._build_left_column())
        left_widget.setMinimumWidth(200)
        self.main_splitter.addWidget(left_widget)
    
        middle_widget = QtWidgets.QWidget()
        middle_widget.setLayout(self._build_middle_column())
        middle_widget.setMinimumWidth(220)
        self.main_splitter.addWidget(middle_widget)
    
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(self._build_right_column())
        right_widget.setMinimumWidth(240)
        self.main_splitter.addWidget(right_widget)
    
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)   # extra window width → middle
        self.main_splitter.setStretchFactor(2, 0)
    
        # Defaults; _load_session_parameters() overrides with the saved value
        self.main_splitter.setSizes([260, 780, 260])
    
        self.setCentralWidget(self.main_splitter)

    def _build_left_column(self):
        col = QtWidgets.QVBoxLayout()
    
        col.addWidget(QtWidgets.QLabel("Data folders"))
        self.folder_tree = QtWidgets.QTreeWidget()
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        # We handle wrapping manually for headers (see _format_tree_header),
        # so the widget-level word wrap stays off. This keeps the widget's
        # layout predictable and gives the leaves a single line each.
        self.folder_tree.setWordWrap(False)
        self.folder_tree.setUniformRowHeights(False)
        self.folder_tree.setTextElideMode(QtCore.Qt.ElideNone)
        # ResizeToContents + horizontal scrollbar → long paths extend past
        # the viewport, so the user can scroll to see the tail.
        self.folder_tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)
        self.folder_tree.header().setStretchLastSection(False)
        self.folder_tree.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded)
        self.folder_tree.currentItemChanged.connect(
            self._on_folder_tree_selected)
        self.folder_tree.itemChanged.connect(
            self._on_folder_tree_check_changed)
        col.addWidget(self.folder_tree)
    
        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add folder")
        add_root_btn = QtWidgets.QPushButton("Add root…")
        del_btn = QtWidgets.QPushButton("Delete")
        add_btn.setToolTip("Add a single measurement folder.")
        add_root_btn.setToolTip(
            "Add a parent folder — all measurement folders underneath "
            "it will be discovered and listed together.")
        add_btn.clicked.connect(self._add_folder)
        add_root_btn.clicked.connect(self._add_root_folder)
        del_btn.clicked.connect(self._delete_folder)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(add_root_btn)
        btn_row.addWidget(del_btn)
        col.addLayout(btn_row)
    
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        col.addWidget(line)
    
        col.addWidget(QtWidgets.QLabel("Measurements"))
        self.file_list = QtWidgets.QTreeWidget()
        self.file_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(
            self._on_selection_changed)
        self.file_list.setHeaderHidden(True)
        # Compact chevron spacing between Folder → XLS file → Run.
        # Default QTreeView indentation is ~20px; 12px keeps them tight
        # while staying legible.
        self.file_list.setIndentation(12)
        self.file_list.setWordWrap(False)
        self.file_list.setTextElideMode(QtCore.Qt.ElideNone)
        self.file_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded)
        self.file_list.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents)
        self.file_list.header().setStretchLastSection(False)
        self.file_list.itemClicked.connect(self._on_entry_selected)
        self.file_list.itemChanged.connect(self._on_item_check_changed)
        col.addWidget(self.file_list)
    
        return col

    def _build_middle_column(self):
        col = QtWidgets.QVBoxLayout()
        
        # Create stacked widget to switch between canvas and image
        self.display_stack = QtWidgets.QStackedWidget()
        
        # --- MAIN PLOTTING CANVAS ---
        self.fig = Figure(figsize=(self.EDIT_FIGSIZE_X, self.EDIT_FIGSIZE_Y), dpi=self.EDIT_FIGURE_DPI)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # canvas_container centers the canvas when it's given a fixed size (Scale to Fit mode). 
        # A QStackedWidget always forces its *page* widget to fill the whole stack area, so letterboxing is an exception rule we must add via this wrapper's centered layout.
        self.canvas_container = QtWidgets.QWidget()
        canvas_container_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        canvas_container_layout.setContentsMargins(0, 0, 0, 0)
        canvas_container_layout.addWidget(self.canvas)
        self.display_stack.addWidget(self.canvas_container)  
        
        # --- Image label (for Real Size mode) ---
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")  # dark grey
        
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.image_label.setMinimumSize(0, 0)
        self.canvas_container.layout().addWidget(self.canvas)
        self.canvas_container.layout().addWidget(self.image_label)
        self.image_label.hide()
        
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setIconSize(QtCore.QSize(20, 20))
        toolbar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        
        col.addWidget(toolbar)
        col.addWidget(self.display_stack, stretch=1)
        return col
    
    def _build_right_column(self):
        col = QtWidgets.QVBoxLayout()
        
        # TAB WIDGET for multi-selection of plotting modes
        self.tab_widget = QtWidgets.QTabWidget()
        col.addWidget(self.tab_widget)
        
        # We build the three columns used in the GUI.
        self._build_display_tab()
        self._build_fitting_tab()
        self._build_advanced_tab()
        self._build_curves_tab()
        self._build_logbook_tab()
        self._build_filter_tab()
        self._build_classify_tab()
        self._build_problems_tab()     

        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        col.addStretch(1)
        return col

    def _build_fitting_tab(self):
        """Build the Fitting tab with background range controls"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Background ranges
        layout.addWidget(QtWidgets.QLabel("Background ranges"))
        self.range_list = QtWidgets.QListWidget()
        self.range_list.itemClicked.connect(self._on_range_item_clicked)
        self.range_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        layout.addWidget(self.range_list)
        
        range_btn_row = QtWidgets.QHBoxLayout()
        
        # SPAN button
        self.span_btn = QtWidgets.QPushButton("|━━|")
        self.span_btn.setCheckable(True)
        self.span_btn.toggled.connect(lambda checked: self._toggle_select_range(checked, RangeType.SPAN))
        range_btn_row.addWidget(self.span_btn)
        
        # ANCHOR button
        self.anchor_btn = QtWidgets.QPushButton("•")
        self.anchor_btn.setCheckable(True)
        self.anchor_btn.toggled.connect(self._toggle_anchor_selection)
        range_btn_row.addWidget(self.anchor_btn)
        
        # Delete button
        delete_btn = QtWidgets.QPushButton("✕")
        delete_btn.clicked.connect(self._remove_range)
        delete_btn.setToolTip("Delete selected range")
        range_btn_row.addWidget(delete_btn)
        
        # Clear all button
        clear_btn = QtWidgets.QPushButton("Clear All")
        clear_btn.clicked.connect(self._clear_all_ranges)
        range_btn_row.addWidget(clear_btn)
        
        layout.addLayout(range_btn_row)
        
        # Background removal mode
        mode_label = QtWidgets.QLabel("Background Removal Mode:")
        layout.addWidget(mode_label)
        
        self.bg_mode_combo = QtWidgets.QComboBox()
        self.bg_mode_combo.addItems([
            "Linear Fit",
            "Polynomial Fit",
            "Spline Interpolation",
            "Connect Region Ends",
            "Exponential Decay",
            "Manual Gradient"
        ])
        self.bg_mode_combo.currentIndexChanged.connect(self._on_bg_mode_changed)
        layout.addWidget(self.bg_mode_combo)
        
        # Polynomial degree 
        poly_layout = QtWidgets.QHBoxLayout()
        poly_layout.addWidget(QtWidgets.QLabel("Polynomial Degree:"))
        self.poly_degree_spin = QtWidgets.QSpinBox()
        self.poly_degree_spin.setRange(1, 10)
        self.poly_degree_spin.setValue(2)
        self.poly_degree_spin.valueChanged.connect(self._on_poly_degree_changed)
        poly_layout.addWidget(self.poly_degree_spin)
        poly_layout.addStretch()
        layout.addLayout(poly_layout)
        
        # Spline smoothing 
        spline_layout = QtWidgets.QHBoxLayout()
        spline_layout.addWidget(QtWidgets.QLabel("Spline Smoothing:"))
        self.spline_smoothing_spin = QtWidgets.QDoubleSpinBox()
        self.spline_smoothing_spin.setRange(0, 1)
        self.spline_smoothing_spin.setSingleStep(0.01)
        self.spline_smoothing_spin.setValue(0.0)
        self.spline_smoothing_spin.valueChanged.connect(self._on_spline_smoothing_changed)
        spline_layout.addWidget(self.spline_smoothing_spin)
        spline_layout.addStretch()
        layout.addLayout(spline_layout)
        
        # --- Curve selection for fitting ---
        layout.addWidget(QtWidgets.QLabel("Curve for fitting"))
        self.fit_curve_list = QtWidgets.QListWidget()
        self.fit_curve_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.fit_curve_list.itemClicked.connect(self._on_fit_curve_selected)
        layout.addWidget(self.fit_curve_list)
        
        self.fit_use_bg_toggle = QtWidgets.QPushButton(
            "Fit on bg-removed data")
        self.fit_use_bg_toggle.setCheckable(True)
        self.fit_use_bg_toggle.setToolTip(
            "When off, fits use raw curve data. When on, fits use the "
            "curve after background subtraction (if any BG fit exists).")
        self.fit_use_bg_toggle.setEnabled(False)
        self.fit_use_bg_toggle.toggled.connect(self._on_fit_use_bg_toggled)
        layout.addWidget(self.fit_use_bg_toggle)
        
        #------ Fit from Selection -----#
        self.fit_bg_btn = QtWidgets.QPushButton("Fit from selection")
        self.fit_bg_btn.clicked.connect(self._fit_background)
        layout.addWidget(self.fit_bg_btn)
        
        self.remove_bg_toggle = QtWidgets.QPushButton("Remove background")
        self.remove_bg_toggle.setCheckable(True)
        self.remove_bg_toggle.toggled.connect(self._toggle_remove_background)
        layout.addWidget(self.remove_bg_toggle)
        
        self.multiplot_toggle = QtWidgets.QPushButton("Multi-plot")
        self.multiplot_toggle.setCheckable(True)
        self.multiplot_toggle.setChecked(True)
        self.multiplot_toggle.toggled.connect(self._toggle_multiplot)
        layout.addWidget(self.multiplot_toggle)
        
        self.comment_toggle = QtWidgets.QPushButton("Plot with comment")
        self.comment_toggle.setCheckable(True)
        self.comment_toggle.toggled.connect(self._toggle_comment)
        layout.addWidget(self.comment_toggle)
        self._toggle_widgets['EDIT_SHOW_COMMENT'] = self.comment_toggle
        
        self.span_show_toggle = QtWidgets.QPushButton("Hide Fit")
        self.span_show_toggle.setCheckable(True)
        self.span_show_toggle.toggled.connect(self._toggle_spans)
        layout.addWidget(self.span_show_toggle)
        self._toggle_widgets['EDIT_HIDE_SPANS'] = self.span_show_toggle
        
        save_btn = QtWidgets.QPushButton("Save figure")
        save_btn.clicked.connect(self._save_figure)
        layout.addWidget(save_btn)
        
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Fitting")
        
    def _build_advanced_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        layout.addWidget(QtWidgets.QLabel("Loading"))
        self.disable_multiload_toggle = QtWidgets.QPushButton("Disable multi-load")
        self.disable_multiload_toggle.setCheckable(True)
        self.disable_multiload_toggle.setToolTip(
            "When on, only one file is kept in memory. Ticking a second file "
            "or activating a measurement in a second file is refused."
        )
        self.disable_multiload_toggle.toggled.connect(self._toggle_disable_multiload)
        layout.addWidget(self.disable_multiload_toggle)

        # ---- Data management ----
        dm_lbl = QtWidgets.QLabel("Data management")
        dm_lbl.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(dm_lbl)

        target_row = QtWidgets.QHBoxLayout()
        target_row.addWidget(QtWidgets.QLabel("Target:"))
        self._dm_target_all = QtWidgets.QPushButton("All ticked folders")
        self._dm_target_all.setCheckable(True)
        self._dm_target_one = QtWidgets.QPushButton("Focused only")
        self._dm_target_one.setCheckable(True)
        self._dm_target_group = QtWidgets.QButtonGroup(self)
        self._dm_target_group.addButton(self._dm_target_all, 0)
        self._dm_target_group.addButton(self._dm_target_one, 1)
        self._dm_target_all.setChecked(True)
        self._dm_target_group.buttonClicked.connect(
            self._on_dm_target_changed)
        target_row.addWidget(self._dm_target_all)
        target_row.addWidget(self._dm_target_one)
        target_row.addStretch(1)
        layout.addLayout(target_row)

        self._dm_target_hint = QtWidgets.QLabel("")
        self._dm_target_hint.setStyleSheet("color: gray; font-size: 10px;")
        self._dm_target_hint.setWordWrap(True)
        layout.addWidget(self._dm_target_hint)

        # ---- LOG file operations ----
        log_lbl = QtWidgets.QLabel("LOG file operations")
        log_lbl.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(log_lbl)

        btn_logs = QtWidgets.QPushButton("Convert LOGs to JSON")
        btn_logs.setToolTip(
            "Convert every *LOG*.xls/.txt to a strict .json (with backup). "
            "Also seeds _LOGBOOK_GUI.json for any folder missing one.")
        btn_logs.clicked.connect(self._convert_logs_to_json)
        layout.addWidget(btn_logs)

        btn_regen = QtWidgets.QPushButton("Regenerate GUI LOG from source")
        btn_regen.setToolTip(
            "Overwrite _LOGBOOK_GUI.json with the current contents of "
            "_LOGBOOK.json. All GUI-side edits are lost. Confirmation "
            "required.")
        btn_regen.clicked.connect(self._regenerate_gui_log)
        layout.addWidget(btn_regen)

        btn_merge = QtWidgets.QPushButton("Compare & merge LOGs…")
        btn_merge.setToolTip(
            "Diff the source LOG against the GUI working copy. Adds new "
            "runs, and lets you review field-level changes one run at a "
            "time.")
        btn_merge.clicked.connect(self._compare_and_merge_logs)
        layout.addWidget(btn_merge)

        btn_export = QtWidgets.QPushButton("Export GUI LOG → source")
        btn_export.setToolTip(
            "Promote _LOGBOOK_GUI.json to be the new source file "
            "(with a timestamped backup of the old source).")
        btn_export.clicked.connect(self._export_gui_log_to_source)
        layout.addWidget(btn_export)

        # ---- pkl operations ----
        pkl_lbl = QtWidgets.QLabel("pkl operations")
        pkl_lbl.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(pkl_lbl)

        btn_add = QtWidgets.QPushButton("Find & add new measurements")
        btn_add.setToolTip(
            "Add runs that aren't in the pkl yet. Existing pkl entries "
            "are kept, including in-GUI LOG edits.")
        btn_add.clicked.connect(self._add_new_measurements)
        layout.addWidget(btn_add)

        btn_rebuild = QtWidgets.QPushButton("Rebuild pkl")
        btn_rebuild.setToolTip(
            "Fully re-derive each target folder's pkl from XLS + "
            "_LOGBOOK_GUI.json. Safe: GUI LOG edits are preserved "
            "because they live in the GUI file, not the pkl.")
        btn_rebuild.clicked.connect(self._rebuild_pkl)
        layout.addWidget(btn_rebuild)

        # ---- migration ----
        mig_lbl = QtWidgets.QLabel("Migration")
        mig_lbl.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(mig_lbl)


        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Advanced")

        # Initialise the hint text once
        self._on_dm_target_changed()
    
    def _toggle_disable_multiload(self, checked: bool):
        self.EDIT_DISABLE_MULTILOAD = bool(checked)
    
    def _build_display_tab(self):
        """Build the Display tab with figure and font settings"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Label for the FIGURE DISPLAY PARAMETERS
        fig_display_label = QtWidgets.QLabel("Figure Display Parameters")
        fig_display_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(fig_display_label)
        
        
        self.show_title_toggle = QtWidgets.QPushButton("Show Title")
        self.show_title_toggle.setCheckable(True)
        self.show_title_toggle.setChecked(True)  # default True
        self.show_title_toggle.toggled.connect(self._toggle_show_title)
        layout.addWidget(self.show_title_toggle)
        self._toggle_widgets['EDIT_SHOW_TITLE'] = self.show_title_toggle
        
        # ---- Title Override ----
        # The placeholder will be updated when a measurement is loaded.
        self.title_override_row = OverrideParamRow(
            "Title:", "EDIT_TITLE_OVERRIDE", self,
            placeholder_getter=self._get_default_title
        )
        layout.addWidget(self.title_override_row)
        
         # ---- X Label Override ----
        self.xlabel_override_row = OverrideParamRow(
            "X Label:", "EDIT_XLABEL_OVERRIDE", self,
            placeholder_getter=lambda: "Time [s]"
        )
        layout.addWidget(self.xlabel_override_row)
 
        # ---- Y Label Override ----
        self.ylabel_override_row = OverrideParamRow(
            "Y Label:", "EDIT_YLABEL_OVERRIDE", self,
            placeholder_getter=lambda: "Current [A]"
        )
        layout.addWidget(self.ylabel_override_row)
 

        # Save parameters
        self.save_width_row = EditableParamRow(
            "Fig Width:", self.EDIT_FIGSIZE_X, "EDIT_FIGSIZE_X", self
        )
        layout.addWidget(self.save_width_row)
        
        self.save_height_row = EditableParamRow(
            "Fig Height:", self.EDIT_FIGSIZE_Y, "EDIT_FIGSIZE_Y", self
        )
        layout.addWidget(self.save_height_row)
        
        # DPI row
        self.dpi_row = EditableParamRow(
            "Figure DPI:", self.EDIT_FIGURE_DPI, "EDIT_FIGURE_DPI", self
        )
        layout.addWidget(self.dpi_row)
        
        # Width factor row
        self.width_factor_row = EditableParamRow(
            "Comment W:", self.EDIT_COMMENT_WIDTH_FACTOR, "EDIT_COMMENT_WIDTH_FACTOR", self
        )
        
        
        
        layout.addWidget(self.width_factor_row)
        
        self.ui_toggle_stretchplot = QtWidgets.QPushButton("Stretch to Fit")
        self.ui_toggle_stretchplot.setCheckable(True)
        self.ui_toggle_stretchplot.clicked.connect(self._cycle_stretch_mode)
        layout.addWidget(self.ui_toggle_stretchplot)
        self._update_button_text()
        
        # ---- Legend label builder ----
        legend_label = QtWidgets.QLabel("Legend label builder")
        legend_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(legend_label)
        
        legend_hint = QtWidgets.QLabel(
            "Choose which parts to include in each curve's legend entry. "
            "The active measurement is used as a live example below."
        )
        legend_hint.setWordWrap(True)
        legend_hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(legend_hint)
        
        legend_btn_row = QtWidgets.QHBoxLayout()
        self._legend_buttons = {}
        for text, var in (
            ("RunID",     "EDIT_LEGEND_SHOW_RUN"),
            ("Device",    "EDIT_LEGEND_SHOW_DEVICE"),
            ("Subdevice", "EDIT_LEGEND_SHOW_SUBDEVICE"),
            ("Measured",  "EDIT_LEGEND_SHOW_MEASURED"),
            ("Sweep",     "EDIT_LEGEND_SHOW_SWEEP"),
        ):
            btn = QtWidgets.QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(getattr(self, var, True))
            btn.toggled.connect(
                lambda checked, v=var: self._on_legend_toggle(v, checked))
            legend_btn_row.addWidget(btn)
            self._legend_buttons[var] = btn
            # Register so _update_toggle_buttons() re-syncs from disk on startup
            self._toggle_widgets[var] = btn
        layout.addLayout(legend_btn_row)
        
        self.legend_preview_label = QtWidgets.QLabel("Preview: —")
        self.legend_preview_label.setWordWrap(True)
        self.legend_preview_label.setStyleSheet(
            "color: #333; background: #f4f4f4; padding: 6px; "
            "font-size: 11px; font-family: monospace;")
        layout.addWidget(self.legend_preview_label)
        
        # -- Separator --
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(line2)
        
        # Font size controls
        font_label = QtWidgets.QLabel("Font Sizes")
        font_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(font_label)
        
        self.font_sizes_row = MultiParamRow(
            headers=["Header", "Body"],
            row_labels=["Text", "Legend", "Comment", "Axis"],
            param_names=[
                ["EDIT_TEXT_HEADER_SIZE", "EDIT_TEXT_BODY_SIZE"],
                ["EDIT_LEGEND_HEADER_SIZE", "EDIT_LEGEND_BODY_SIZE"],
                ["EDIT_COMMENT_HEADER_SIZE", "EDIT_COMMENT_BODY_SIZE"],
                ["EDIT_AXIS_LABEL_SIZE", "EDIT_TICK_SIZE"]
            ],
            parent=self,
            default_type = "float"
        )
        layout.addWidget(self.font_sizes_row)
        
        # Bounds controls
        font_label = QtWidgets.QLabel("Plot Bounds")
        font_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(font_label)
        
        self.bounds_row = MultiParamRow(
            rows=[
                [("Left", "EDIT_BOUNDS_LEFT"), ("Right", "EDIT_BOUNDS_RIGHT")],
                [("Bottom", "EDIT_BOUNDS_BOTTOM"), ("Top", "EDIT_BOUNDS_TOP")]
            ],
            parent=self,
            default_type = "float"
        )
        layout.addWidget(self.bounds_row)
        
        
        # Bounds controls
        font_label = QtWidgets.QLabel("Plot Limits")
        font_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(font_label)
        # GET BOUNDS to make altering them easier
        self.get_bounds_btn = QtWidgets.QPushButton("Get Plot Bounds")
        self.get_bounds_btn.clicked.connect(self.on_get_bounds_clicked)
        layout.addWidget(self.get_bounds_btn)
        
        self.plot_lim_row = MultiParamRow(
            headers=["min", "max"],
            row_labels=["x-axis", "y-axis"],
            param_names=[
                ["EDIT_XMIN", "EDIT_XMAX"],
                ["EDIT_YMIN", "EDIT_YMAX"],
            ],
            parent=self,
            allow_none = True,
            default_type = "float"
        )
        layout.addWidget(self.plot_lim_row)
        
        reset_all_btn = QtWidgets.QPushButton("↺ Reset All Parameters")
        reset_all_btn.clicked.connect(self._reset_all_parameters)
        layout.addWidget(reset_all_btn)
        
        # -- Separator --
        line3 = QtWidgets.QFrame()
        line3.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(line3)
        
        # rcParam override file handling
        rcparam_label = QtWidgets.QLabel("RC Param Handling")
        rcparam_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(rcparam_label)
        
        rcparam_hint = QtWidgets.QLabel(
            "Save: write current settings as the file that loads on every "
            "launch. Load: re-apply that saved file now. Reset: delete it "
            "and go back to the defaults hardcoded in the script."
        )
        rcparam_hint.setWordWrap(True)
        rcparam_hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(rcparam_hint)
        
        rcparam_btn_row = QtWidgets.QHBoxLayout()
        save_rc_btn = QtWidgets.QPushButton("Save")
        save_rc_btn.setToolTip("Save current rcParams as the startup override file")
        save_rc_btn.clicked.connect(self._save_rcparam_overrides)
        load_rc_btn = QtWidgets.QPushButton("Load")
        load_rc_btn.setToolTip("Reload the saved override file into this session")
        load_rc_btn.clicked.connect(self._load_rcparam_overrides)
        reset_rc_btn = QtWidgets.QPushButton("Reset")
        reset_rc_btn.setToolTip("Delete the override file and revert to code defaults")
        reset_rc_btn.clicked.connect(self._reset_rcparam_overrides)
        rcparam_btn_row.addWidget(save_rc_btn)
        rcparam_btn_row.addWidget(load_rc_btn)
        rcparam_btn_row.addWidget(reset_rc_btn)
        layout.addLayout(rcparam_btn_row)
        
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Display")    
    
    def _build_logbook_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
    
        # ---- Active measurement (editable) ----
        active_lbl = QtWidgets.QLabel("Active measurement")
        active_lbl.setStyleSheet("font-weight: bold;")
        # ---- Identity (read-only) ----
        identity_row = QtWidgets.QHBoxLayout()
        identity_row.addWidget(QtWidgets.QLabel("File:"))
        self._log_identity_file = QtWidgets.QLineEdit()
        self._log_identity_file.setReadOnly(True)
        self._log_identity_file.setStyleSheet(
            "background: #f4f4f4; color: #333;")
        identity_row.addWidget(self._log_identity_file, stretch=2)
       
        identity_row.addWidget(QtWidgets.QLabel("RunID:"))
        self._log_identity_runid = QtWidgets.QLineEdit()
        self._log_identity_runid.setReadOnly(True)
        self._log_identity_runid.setStyleSheet("background: #f4f4f4; color: #333;")
        identity_row.addWidget(self._log_identity_runid, stretch=1)
        layout.addLayout(identity_row) 
        layout.addWidget(active_lbl)
        
        # Comment first, always visible (multi-line)
        self._log_comment = QtWidgets.QPlainTextEdit()
        self._log_comment.setFixedHeight(80)
        self._log_comment.textChanged.connect(self._on_log_comment_changed)
        layout.addWidget(self._log_comment)
    
        # Dynamic fields, scrollable in case user adds many
        self._log_fields = {}
        self._log_fields_container = QtWidgets.QWidget()
        self._log_form_layout = QtWidgets.QFormLayout(
            self._log_fields_container)
        self._log_form_layout.setContentsMargins(0, 0, 0, 0)
        self._log_form_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    
        self._log_fields_scroll = QtWidgets.QScrollArea()
        self._log_fields_scroll.setWidgetResizable(True)
        self._log_fields_scroll.setWidget(self._log_fields_container)
        layout.addWidget(self._log_fields_scroll, stretch=1)
    
        # ---- New Entry row ----
        new_sep = QtWidgets.QFrame()
        new_sep.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(new_sep)
    
        new_lbl = QtWidgets.QLabel("New Entry")
        new_lbl.setStyleSheet("font-weight: bold;")
        layout.addWidget(new_lbl)
    
        new_row = QtWidgets.QHBoxLayout()
        self._log_new_field_name = QtWidgets.QLineEdit()
        self._log_new_field_name.setPlaceholderText("Field name")
    
        self._log_new_field_type = QtWidgets.QComboBox()
        self._log_new_field_type.addItems(["string", "bool", "int", "float"])
    
        btn_focused = QtWidgets.QPushButton("Add to focused")
        btn_focused.setToolTip("Add this field to the active measurement only")
        btn_focused.clicked.connect(self._on_add_new_field_focused)
    
        btn_ticked = QtWidgets.QPushButton("Add to ticked")
        btn_ticked.setToolTip("Add this field to every ticked measurement")
        btn_ticked.clicked.connect(self._on_add_new_field_ticked)
    
        new_row.addWidget(self._log_new_field_name, stretch=1)
        new_row.addWidget(self._log_new_field_type)
        new_row.addWidget(btn_focused)
        new_row.addWidget(btn_ticked)
        layout.addLayout(new_row)
    
        # ---- Collapsible list of other selected/ticked measurements ----
        if not hasattr(self, "_log_other_expanded"):
            self._log_other_expanded = False
    
        other_sep = QtWidgets.QFrame()
        other_sep.setFrameShape(QtWidgets.QFrame.HLine)
        layout.addWidget(other_sep)
    
        others_row = QtWidgets.QHBoxLayout()
        others_lbl = QtWidgets.QLabel("Other selected / ticked measurements")
        others_lbl.setStyleSheet("font-weight: bold;")
        others_row.addWidget(others_lbl, stretch=1)
    
        self._log_other_toggle = QtWidgets.QPushButton(
            "Collapse All" if self._log_other_expanded else "Expand All")
        self._log_other_toggle.setFixedWidth(100)
        self._log_other_toggle.setToolTip(
            "Set whether 'Other measurements' sections open expanded or "
            "collapsed by default when the selection changes.")
        self._log_other_toggle.clicked.connect(self._on_log_other_toggle)
        others_row.addWidget(self._log_other_toggle)
        layout.addLayout(others_row)
    
        self._log_other_scroll = QtWidgets.QScrollArea()
        self._log_other_scroll.setWidgetResizable(True)
        self._log_other_container = QtWidgets.QWidget()
        self._log_other_layout = QtWidgets.QVBoxLayout(
            self._log_other_container)
        self._log_other_layout.setContentsMargins(0, 0, 0, 0)
        self._log_other_layout.addStretch(1)
        self._log_other_scroll.setWidget(self._log_other_container)
        layout.addWidget(self._log_other_scroll, stretch=1)
    
        self._logbook_tab_index = self.tab_widget.addTab(tab, "Logbook")
        
    def _on_log_other_toggle(self):
        """Flip the global 'always expanded' state for other-measurement
        sections, then rebuild the tab so every section honours it."""
        self._log_other_expanded = not self._log_other_expanded
        self._log_other_toggle.setText(
            "Collapse All" if self._log_other_expanded else "Expand All")
        self._refresh_logbook_tab()


    # -- Filter Menu stuff ---#
    def _build_filter_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
    
        hint = QtWidgets.QLabel(
            "Show only runs whose logbook fields match the conditions below. "
            "Filtering is a view operation — tick state is preserved, and "
            "hidden runs still receive 'Add to ticked' / 'Copy to ticked' "
            "edits. Combine conditions with AND or OR.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint)
    
        # Enable toggle
        self.filter_enable_btn = QtWidgets.QPushButton("Filtering: OFF")
        self.filter_enable_btn.setCheckable(True)
        self.filter_enable_btn.setChecked(False)
        self.filter_enable_btn.setStyleSheet(
            "QPushButton { padding: 4px; } "
            "QPushButton:checked { background-color: #4CAF50; color: white; }")
        self.filter_enable_btn.toggled.connect(self._on_filter_enabled_toggled)
        layout.addWidget(self.filter_enable_btn)
    
        # Status line
        self._filter_status_label = QtWidgets.QLabel("")
        self._filter_status_label.setStyleSheet(
            "color: #555; font-size: 10px;")
        self._filter_status_label.setWordWrap(True)
        layout.addWidget(self._filter_status_label)
    
        # Conditions scroll area
        self.filter_scroll = QtWidgets.QScrollArea()
        self.filter_scroll.setWidgetResizable(True)
        self.filter_container = QtWidgets.QWidget()
        self.filter_layout = QtWidgets.QVBoxLayout(self.filter_container)
        self.filter_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_layout.addStretch(1)
        self.filter_scroll.setWidget(self.filter_container)
        layout.addWidget(self.filter_scroll, stretch=1)
    
        # Add condition
        add_btn = QtWidgets.QPushButton("+ Add condition")
        add_btn.clicked.connect(self._on_add_filter_condition)
        layout.addWidget(add_btn)
    
        # Mode row
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Combine:"))
        self.filter_mode_and = QtWidgets.QRadioButton("AND")
        self.filter_mode_or = QtWidgets.QRadioButton("OR")
        self.filter_mode_and.setChecked(True)
        self.filter_mode_group = QtWidgets.QButtonGroup(self)
        self.filter_mode_group.addButton(self.filter_mode_and, 0)
        self.filter_mode_group.addButton(self.filter_mode_or, 1)
        self.filter_mode_group.buttonClicked.connect(self._on_filter_mode_changed)
        mode_row.addWidget(self.filter_mode_and)
        mode_row.addWidget(self.filter_mode_or)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)
    
        # Action row
        btn_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh fields")
        refresh_btn.setToolTip(
            "Re-scan ticked folders' logbooks for available field names.")
        refresh_btn.clicked.connect(self._refresh_filter_field_options)
        clear_btn = QtWidgets.QPushButton("Clear all")
        clear_btn.clicked.connect(self._on_clear_filter_conditions)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
    
        self._filter_tab_index = self.tab_widget.addTab(tab, "Filter")
    
    def _on_tab_changed(self, idx):
        if idx == self._filter_tab_index and not self._filter_field_types:
            self._refresh_filter_field_options()
   
    def _gather_log_fields(self) -> dict[str, set]:
        """Return {field_name: set_of_type_names} across every ticked folder.
    
        Loaded measurements contribute their in-memory logs (including
        uncommitted edits). Unloaded folders contribute their pkl data.
        """
        import pickle
        fields: dict[str, set] = {}
    
        def _scan(log):
            for k, v in (log or {}).items():
                if str(k).startswith("_"):
                    continue
                fields.setdefault(str(k), set()).add(type(v).__name__)
    
        for lf in self.loaded_files.values():
            for meas in lf.measurements:
                _scan(meas.log)
    
        loaded_folders = {
            str(Path(fp).parent.resolve()) for fp in self.loaded_files
        }
        for folder in self._get_ticked_folders():
            if str(Path(folder).resolve()) in loaded_folders:
                continue
            pkl = folder / f".dmui_{folder.name}.pkl"
            if not pkl.exists():
                continue
            try:
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
            except Exception:
                continue
            if (data.get("_meta") or {}).get("structure_version") != 2:
                continue
            for fname, runs in data.items():
                if fname == "_meta" or not isinstance(runs, dict):
                    continue
                for _run_key, run_data in runs.items():
                    _scan(run_data.get("LOG"))
    
        return fields
    # -- filter: row construction ------------------------------------------
    
    _FILTER_OPERATORS = {
        "bool":  ["==", "!=", "is None", "is not None"],
        "int":   ["==", "!=", ">", "<", ">=", "<=", "is None", "is not None"],
        "float": ["==", "!=", ">", "<", ">=", "<=", "is None", "is not None"],
        "str":   ["==", "!=", "contains", "is None", "is not None"],
    }
    
    def _operators_for_field(self, field):
        types = self._filter_field_types.get(field, set())
        if types.issubset({"bool"}):
            return self._FILTER_OPERATORS["bool"]
        if types and types.issubset({"int", "float"}):
            return self._FILTER_OPERATORS["float"]
        return self._FILTER_OPERATORS["str"]
    
    def _value_widget_type(self, field):
        types = self._filter_field_types.get(field, set())
        if types.issubset({"bool"}) and types:
            return "bool"
        if types and types.issubset({"int", "float"}):
            return "numeric"
        return "string"
    
    def _make_filter_condition_row(self, idx, cond):
        """Build one condition row and wire its signals to cond mutation."""
        row = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(row)
        hbox.setContentsMargins(0, 2, 0, 2)
    
        enabled_cb = QtWidgets.QCheckBox()
        enabled_cb.setChecked(cond.get("enabled", True))
        enabled_cb.toggled.connect(
            lambda v, i=idx: self._on_condition_enabled_changed(i, v))
        hbox.addWidget(enabled_cb)
    
        field_combo = QtWidgets.QComboBox()
        for f in sorted(self._filter_field_types):
            field_combo.addItem(f, userData=f)
        fi = field_combo.findData(cond.get("field", ""))
        if fi >= 0:
            field_combo.setCurrentIndex(fi)
        hbox.addWidget(field_combo, stretch=2)
    
        op_combo = QtWidgets.QComboBox()
        hbox.addWidget(op_combo, stretch=1)
    
        value_container = QtWidgets.QWidget()
        vc = QtWidgets.QHBoxLayout(value_container)
        vc.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(value_container, stretch=2)
    
        rm_btn = QtWidgets.QPushButton("×")
        rm_btn.setFixedWidth(26)
        rm_btn.clicked.connect(
            lambda _checked=False, i=idx: self._on_remove_filter_condition(i))
        hbox.addWidget(rm_btn)
    
        cond["_field_combo"] = field_combo
        cond["_op_combo"] = op_combo
        cond["_value_container"] = value_container
    
        self._populate_operator_combo(idx)
        self._populate_value_widget(idx)
    
        field_combo.currentIndexChanged.connect(
            lambda _i, i=idx: self._on_condition_field_changed(i))
        return row
    
    def _populate_operator_combo(self, idx):
        cond = self._filter_conditions[idx]
        field = cond.get("field") or (
            sorted(self._filter_field_types)[0]
            if self._filter_field_types else "")
        cond["field"] = field
    
        combo = cond["_op_combo"]
        combo.blockSignals(True)
        combo.clear()
        for op in self._operators_for_field(field):
            combo.addItem(op, userData=op)
        want = cond.get("operator") or "=="
        ci = combo.findData(want)
        combo.setCurrentIndex(ci if ci >= 0 else 0)
        cond["operator"] = combo.currentData() or "=="
        combo.blockSignals(False)
        combo.currentIndexChanged.connect(
            lambda _i, i=idx: self._on_condition_operator_changed(i))
    
    def _populate_value_widget(self, idx):
        cond = self._filter_conditions[idx]
        container = cond["_value_container"]
        while container.layout().count():
            item = container.layout().takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        layout = container.layout()
    
        if cond.get("operator") in ("is None", "is not None"):
            lbl = QtWidgets.QLabel("—")
            lbl.setStyleSheet("color: #aaa;")
            layout.addWidget(lbl)
            return
    
        field = cond.get("field", "")
        vtype = self._value_widget_type(field)
        current = cond.get("value")
    
        if vtype == "bool":
            combo = QtWidgets.QComboBox()
            combo.addItem("True", True)
            combo.addItem("False", False)
            combo.addItem("None", None)
            ci = combo.findData(current)
            combo.setCurrentIndex(ci if ci >= 0 else 0)
            combo.currentIndexChanged.connect(
                lambda _i, i=idx, c=combo:
                self._on_condition_value_changed(i, c.currentData()))
            layout.addWidget(combo)
            if "value" not in cond:
                cond["value"] = combo.currentData()
        else:
            le = QtWidgets.QLineEdit()
            if vtype == "numeric":
                v = QtGui.QDoubleValidator()
                v.setNotation(QtGui.QDoubleValidator.StandardNotation)
                le.setValidator(v)
            if current is not None:
                le.setText(str(current))
            le.editingFinished.connect(
                lambda i=idx, w=le, t=vtype:
                self._on_condition_value_changed(
                    i, self._parse_filter_value_str(w.text(), t)))
            layout.addWidget(le)
            if "value" not in cond:
                cond["value"] = self._parse_filter_value_str(
                    le.text(), vtype)
    
    @staticmethod
    def _parse_filter_value_str(text, vtype):
        s = text.strip()
        if s == "":
            return None
        if vtype == "numeric":
            try:
                return float(s)
            except ValueError:
                return None
        low = s.lower()
        if low in ("true", "yes"):    return True
        if low in ("false", "no"):    return False
        if low in ("none", "null"):   return None
        return s
    
    # -- filter: condition handlers ----------------------------------------
    
    def _on_add_filter_condition(self):
        if not self._filter_field_types:
            self._filter_field_types = self._gather_log_fields()
        if not self._filter_field_types:
            QtWidgets.QMessageBox.information(
                self, "No fields available",
                "Load or tick at least one folder with a logbook first.")
            return
        first = sorted(self._filter_field_types)[0]
        self._filter_conditions.append({
            "field": first, "operator": "==", "value": None,
            "enabled": True,
        })
        self._render_filter_conditions()
        self._apply_filter_to_tree()
        self._save_workspace()
    
    def _on_remove_filter_condition(self, idx):
        if 0 <= idx < len(self._filter_conditions):
            del self._filter_conditions[idx]
        self._render_filter_conditions()
        self._apply_filter_to_tree()
        self._save_workspace()
    
    def _on_condition_enabled_changed(self, idx, checked):
        if 0 <= idx < len(self._filter_conditions):
            self._filter_conditions[idx]["enabled"] = bool(checked)
            self._apply_filter_to_tree()
            self._save_workspace()
    
    def _on_condition_field_changed(self, idx):
        if not (0 <= idx < len(self._filter_conditions)):
            return
        cond = self._filter_conditions[idx]
        cond["field"] = cond["_field_combo"].currentData()
        # Reset operator + value to sensible defaults for the new type
        ops = self._operators_for_field(cond["field"])
        cond["operator"] = ops[0]
        cond.pop("value", None)
        self._populate_operator_combo(idx)
        self._populate_value_widget(idx)
        self._apply_filter_to_tree()
        self._save_workspace()
    
    def _on_condition_operator_changed(self, idx):
        if not (0 <= idx < len(self._filter_conditions)):
            return
        cond = self._filter_conditions[idx]
        cond["operator"] = cond["_op_combo"].currentData() or "=="
        self._populate_value_widget(idx)
        self._apply_filter_to_tree()
        self._save_workspace()
    
    def _on_condition_value_changed(self, idx, new_value):
        if 0 <= idx < len(self._filter_conditions):
            self._filter_conditions[idx]["value"] = new_value
            self._apply_filter_to_tree()
            self._save_workspace()
    
    def _on_filter_enabled_toggled(self, checked):
        self._filter_enabled = bool(checked)
        self.filter_enable_btn.setText(
            "Filtering: ON" if checked else "Filtering: OFF")
        self._apply_filter_to_tree()
        self._save_workspace()
    
    def _on_filter_mode_changed(self, _btn):
        self._filter_mode = "AND" if self.filter_mode_and.isChecked() else "OR"
        self._apply_filter_to_tree()
        self._save_workspace()
    
    def _on_clear_filter_conditions(self):
        if not self._filter_conditions:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Clear filters",
            f"Remove all {len(self._filter_conditions)} filter condition(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        self._filter_conditions = []
        self._render_filter_conditions()
        self._apply_filter_to_tree()
        self._save_workspace()
    def _refresh_filter_field_options(self):
        self._filter_field_types = self._gather_log_fields()
        self._render_filter_conditions()
        self._apply_filter_to_tree()
    
    def _render_filter_conditions(self):
        """Rebuild all condition rows from self._filter_conditions."""
        while self.filter_layout.count() > 1:  # keep trailing stretch
            item = self.filter_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
    
        if not self._filter_field_types:
            self._filter_field_types = self._gather_log_fields()
    
        for idx, cond in enumerate(self._filter_conditions):
            row = self._make_filter_condition_row(idx, cond)
            self.filter_layout.insertWidget(
                self.filter_layout.count() - 1, row)
    
    # -- filter: evaluation + tree application -----------------------------

    @staticmethod
    def _filter_values_equal(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if a == b:
            return True
        try:
            return str(a) == str(b)
        except Exception:
            return False
   
    @staticmethod
    def _filter_to_number(v):
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        return float(v)
   
    def _evaluate_filter_condition(self, cond, log) -> bool:
        field    = cond.get("field")
        operator = cond.get("operator", "==")
        target   = cond.get("value")
        actual   = log.get(field)
   
        if operator == "is None":       return actual is None
        if operator == "is not None":   return actual is not None
        if operator == "==":            return self._filter_values_equal(actual, target)
        if operator == "!=":            return not self._filter_values_equal(actual, target)
        if operator == "contains":
            if actual is None or target is None:
                return False
            return str(target) in str(actual)
   
        try:
            a = self._filter_to_number(actual)
            b = self._filter_to_number(target)
        except (TypeError, ValueError):
            return False
        if operator == ">":  return a > b
        if operator == "<":  return a < b
        if operator == ">=": return a >= b
        if operator == "<=": return a <= b
        return False
   
    def _apply_filter_to_tree(self):
        """Re-apply filter visibility across the tree. Cheap; call freely."""
        if not hasattr(self, "file_list") or self.file_list is None:
            return
        for i in range(self.file_list.topLevelItemCount()):
            self._apply_filter_to_subtree(self.file_list.topLevelItem(i))
        self._update_filter_status()
   
    def _apply_filter_to_subtree(self, item) -> bool:
        """Recursively apply filter. Returns True if item should be visible."""
        payload = item.data(0, QtCore.Qt.UserRole)
        kind = payload[0] if payload else None
   
        if kind == "meas":
            visible = self._measurement_passes_filter(item, payload)
            item.setHidden(not visible)
            return visible
   
        any_visible = False
        for i in range(item.childCount()):
            if self._apply_filter_to_subtree(item.child(i)):
                any_visible = True
        item.setHidden(not any_visible)
        return any_visible
   
    def _measurement_passes_filter(self, _item, payload) -> bool:
        if not self._filter_enabled:
            return True
        active = [c for c in self._filter_conditions if c.get("enabled", True)]
        if not active:
            return True
    
        _, fp, _, mi = payload
        log = self._get_log_for_run(fp, mi)
        if log is None:
            return True  # neither loaded nor in pkl — cannot evaluate
    
        results = [self._evaluate_filter_condition(c, log) for c in active]
        return all(results) if self._filter_mode == "AND" else any(results)
    
    def _get_log_for_run(self, fp, mi):
        """Return a run's log dict, preferring in-memory over pkl.
    
        In-memory `loaded_files` reflects any edits the user has made
        this session (including unflushed ones), so it's always the
        freshest source. For unloaded files, fall back to the folder's
        pkl — its `LOG` entries are updated on every flush.
    
        Returns None if the run cannot be resolved at all.
        """
        lf = self.loaded_files.get(fp)
        if lf is not None and 0 <= mi < len(lf.measurements):
            return lf.measurements[mi].log or {}
    
        folder = Path(fp).parent
        folder_key = str(folder.resolve())
        if folder_key not in self._filter_log_cache:
            self._filter_log_cache[folder_key] = self._load_folder_logs(folder)
        return self._filter_log_cache[folder_key].get(
            (str(Path(fp).resolve()), mi))
    
    def _load_folder_logs(self, folder: Path) -> dict:
        """Return {(abs_fp_str, meas_index): log} from a folder's pkl."""
        import pickle
        import re
    
        result: dict[tuple[str, int], dict] = {}
        pkl = folder / f".dmui_{folder.name}.pkl"
        if not pkl.exists():
            return result
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
        except Exception:
            return result
        if (data.get("_meta") or {}).get("structure_version") != 2:
            return result  # stale pkl, skip filter cache
    
        def _run_sort_key(run_key):
            s = str(run_key)
            m = re.search(r"(\d+)", s)
            if m:
                return (0, int(m.group(1)), s)
            return (1, 0, s)
    
        for fname in sorted(data.keys()):
            if fname == "_meta" or not isinstance(data[fname], dict):
                continue
            runs = data[fname]
            fp_str = str((folder / fname).resolve())
            for mi, run_key in enumerate(sorted(runs.keys(),
                                                 key=_run_sort_key)):
                result[(fp_str, mi)] = runs[run_key].get("LOG") or {}
        return result
   
    def _update_filter_status(self):
        if not hasattr(self, "_filter_status_label"):
            return
        active = [c for c in self._filter_conditions if c.get("enabled", True)]
        if not self._filter_enabled or not active:
            self._filter_status_label.setText(
                "No active filter." if not active
                else "Filtering is off.")
            return
   
        total = visible = 0
        for item in self._iter_all_items():
            p = item.data(0, QtCore.Qt.UserRole)
            if p and p[0] == "meas":
                total += 1
                if not item.isHidden():
                    visible += 1
        self._filter_status_label.setText(
            f"Showing {visible} of {total} run(s) "
            f"({total - visible} hidden by filter).")     
   
    # -- Filter Handlers
    def _wrap_log_field_with_copy(self, key, widget):
        """Wrap a field widget with a '→' button that copies this field's
        current value to every other ticked measurement."""
        container = QtWidgets.QWidget()
        hbox = QtWidgets.QHBoxLayout(container)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(widget, stretch=1)
    
        btn = QtWidgets.QPushButton("→")
        btn.setFixedWidth(24)
        btn.setToolTip(
            f"Copy '{key}' from this measurement to all other ticked "
            f"measurements.")
        btn.clicked.connect(
            lambda _checked=False, k=key: self._copy_field_to_ticked(k))
        hbox.addWidget(btn)
        return container
    
    def _copy_field_to_ticked(self, key):
        """Batch edit: apply the active measurement's `key` value to every
        other ticked measurement. Writes through the folder store."""
        if self.loaded_file is None or self.current_meas_index is None:
            return
        src = self.loaded_file.measurements[self.current_meas_index]
        val = src.log.get(key)
    
        active = (self.current_file_path, self.current_meas_index)
        targets = sorted(
            t for t in self._get_ticked_measurements() if t != active)
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "No targets",
                "Tick at least one other measurement before copying.")
            return
    
        preview = repr(val)
        if len(preview) > 80:
            preview = preview[:77] + "…"
        reply = QtWidgets.QMessageBox.question(
            self, f"Copy '{key}'",
            f"Copy {key} = {preview}\n\n"
            f"…to {len(targets)} other ticked measurement(s)?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
    
        import copy
        updated, failed = 0, []
        for fp, mi in targets:
            if fp not in self.loaded_files:
                try:
                    self.loaded_files[fp] = self.registry.load(Path(fp))
                except Exception as e:
                    failed.append(f"{Path(fp).name}: {e}")
                    continue
            lf = self.loaded_files[fp]
            if mi >= len(lf.measurements):
                continue
            meas = lf.measurements[mi]
            run_key = meas.metadata.get("run_key")
            meas.log[key] = copy.deepcopy(val)
            if lf.folder_store is not None and run_key is not None:
                try:
                    lf.folder_store.save_log(
                        Path(fp).name, run_key, dict(meas.log))
                except Exception as e:
                    failed.append(f"{Path(fp).name}: {e}")
                    continue
            updated += 1
    
        # Persist the source's own pending edit
        if self.loaded_file.folder_store is not None:
            rk = src.metadata.get("run_key")
            if rk is not None:
                self.loaded_file.folder_store.save_log(
                    Path(self.current_file_path).name, rk, dict(src.log))
    
        msg = f"Copied '{key}' to {updated} measurement(s)."
        if failed:
            msg += "\n\nFailures:\n" + "\n".join(failed[:8])
            if len(failed) > 8:
                msg += f"\n… and {len(failed) - 8} more"
        QtWidgets.QMessageBox.information(self, "Copy complete", msg)
    # -- Add-new-field ------------------------------------------------------

    def _on_add_new_field_focused(self):
        self._flush_pending_log_commit()
        name, default = self._read_new_field_spec()
        if name is None:
            return
        if self.current_file_path is None or self.current_meas_index is None:
            QtWidgets.QMessageBox.information(
                self, "No active measurement",
                "Select a measurement first.")
            return
        targets = [(self.current_file_path, self.current_meas_index)]
        self._add_field_to_measurements(name, default, targets)

    def _on_add_new_field_ticked(self):
        self._flush_pending_log_commit()
        name, default = self._read_new_field_spec()
        if name is None:
            return
        targets = sorted(self._get_ticked_measurements())
        if not targets:
            QtWidgets.QMessageBox.information(
                self, "No ticked measurements",
                "Tick at least one measurement first.")
            return
        self._add_field_to_measurements(name, default, targets)

    def _read_new_field_spec(self):
        """Return (name, default_value), or (None, None) on cancel/invalid."""
        name = self._log_new_field_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(
                self, "Field name required",
                "Enter a name for the new field.")
            return (None, None)

        ftype = self._log_new_field_type.currentText()
        defaults = {"string": None, "bool": False,
                    "int": None, "float": None}
        return (name, defaults[ftype])

    def _add_field_to_measurements(self, name, default, targets):
        """Add or reset `name` = `default` on every target measurement."""
        ftype = self._log_new_field_type.currentText()

        # Count how many targets we already know have this field so the
        # confirmation text is honest. Un-loaded targets are loaded below.
        loaded_with_field = 0
        for fp, mi in targets:
            lf = self.loaded_files.get(fp)
            if lf is not None and mi < len(lf.measurements):
                if name in lf.measurements[mi].log:
                    loaded_with_field += 1

        extra = ""
        if loaded_with_field:
            extra = (f"\n\nNote: {loaded_with_field} target(s) already "
                     f"have this field; their values will be reset to the "
                     f"type's default.")
        reply = QtWidgets.QMessageBox.question(
            self, "Add field",
            f"Add '{name}' ({ftype}) to {len(targets)} measurement(s)?"
            f"{extra}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        updated = 0
        failed: list[str] = []
        for fp, mi in targets:
            if getattr(self, "EDIT_DISABLE_MULTILOAD", False):
                others = [p for p in self.loaded_files.keys() if p != fp]
                if others:
                    failed.append(
                        f"{Path(fp).name} m{mi}: multi-load disabled")
                    continue
            if fp not in self.loaded_files:
                try:
                    self.loaded_files[fp] = self.registry.load(Path(fp))
                except Exception as e:
                    failed.append(f"{Path(fp).name}: load failed — {e}")
                    continue

            lf = self.loaded_files[fp]
            if mi >= len(lf.measurements):
                continue
            meas = lf.measurements[mi]
            run_key = meas.metadata.get("run_key")

            new_log = dict(meas.log)
            new_log[name] = default
            try:
                if lf.folder_store is not None and run_key is not None:
                    ok = lf.folder_store.save_log(run_key, new_log)
                    if not ok:
                        failed.append(
                            f"{Path(fp).name} m{mi}: run not found in pkl")
                        continue
                else:
                    # PalmSens — no folder store, in-memory only
                    meas.log[name] = default
                updated += 1
            except Exception as e:
                failed.append(f"{Path(fp).name} m{mi}: {e}")

        # Rebuild the active view
        self._refresh_logbook_tab()

        # Clear the entry, but keep the last type selected (convenience)
        self._log_new_field_name.clear()

        msg = f"Added '{name}' ({ftype}) to {updated} measurement(s)."
        if failed:
            msg += "\n\nFailures:\n" + "\n".join(failed[:10])
            if len(failed) > 10:
                msg += f"\n… and {len(failed) - 10} more"
        QtWidgets.QMessageBox.information(self, "Field added", msg)
        
    def _refresh_logbook_indicator(self):
        """Set the Logbook tab title to indicate unsaved-to-source diff."""
        if not hasattr(self, "_logbook_tab_index"):
            return

        folder = (Path(self.current_file_path).parent
                  if self.current_file_path else None)
        if folder is None:
            self.tab_widget.setTabText(self._logbook_tab_index, "Logbook")
            self.tab_widget.setTabToolTip(self._logbook_tab_index, "")
            return

        source = dit.find_log_file(folder)
        gui    = dit.find_gui_log_file(folder)
        if source is None or gui is None:
            self.tab_widget.setTabText(self._logbook_tab_index, "Logbook")
            self.tab_widget.setTabToolTip(self._logbook_tab_index, "")
            return

        try:
            diff = dit.compare_logs(dit.load_log_file(source), dit.load_log_file(gui),folder=folder)
            count = (len(diff["differing_runs"])
                     + len(diff["new_runs"])
                     + len(diff["only_in_gui"]))
        except Exception:
            count = 0

        if count == 0:
            self.tab_widget.setTabText(self._logbook_tab_index, "Logbook")
            self.tab_widget.setTabToolTip(self._logbook_tab_index, "")
        else:
            self.tab_widget.setTabText(self._logbook_tab_index, "Logbook ●")
            self.tab_widget.setTabToolTip(
                self._logbook_tab_index,
                f"{count} run(s) differ between _LOGBOOK.json and the "
                f"GUI copy. Use Advanced → 'Export GUI LOG → source' "
                f"to promote, or 'Compare & merge LOGs…' to reconcile.")

    def _refresh_logbook_tab(self):
        # ---------- Active section: comment + identity ----------
        if self.loaded_file is None or self.current_meas_index is None:
            if hasattr(self, "_log_identity_file"):
                self._log_identity_file.setText("")
            if hasattr(self, "_log_identity_runid"):
                self._log_identity_runid.setText("")
            self._log_comment.blockSignals(True)
            self._log_comment.setPlainText("")
            self._log_comment.blockSignals(False)
            self._rebuild_log_fields({})
        else:
            meas = self.loaded_file.measurements[self.current_meas_index]
            log = getattr(meas, "log", {}) or {}

            if hasattr(self, "_log_identity_file"):
                self._log_identity_file.setText(
                    Path(self.current_file_path).name
                    if self.current_file_path else "")
            if hasattr(self, "_log_identity_runid"):
                self._log_identity_runid.setText(
                    str(meas.metadata.get("run_key", "")))

            self._log_comment.blockSignals(True)
            self._log_comment.setPlainText(
                str(log.get("Comment") or meas.comment or ""))
            self._log_comment.blockSignals(False)

            self._rebuild_log_fields(log)

        # ---------- Other selected / ticked sections ----------
        while self._log_other_layout.count() > 1:
            item = self._log_other_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        active = (
            (self.current_file_path, self.current_meas_index)
            if self.current_file_path is not None else None
        )
        combined = (self._get_ticked_measurements()
                    | set(self.selected_measurements))
        others = sorted(t for t in combined if t != active)

        if not others:
            hint = QtWidgets.QLabel("(nothing else ticked or highlighted)")
            hint.setStyleSheet("color: gray; font-style: italic;")
            self._log_other_layout.insertWidget(0, hint)
            self._refresh_logbook_indicator()
            return

        readonly_keys = ("Timestamp", "Device", "Subdevice", "RunID",
                         "SMU Order", "LightMicroscope",
                         "SMU1", "SMU2", "SMU3", "SMU4")

        for fp, mi in others:
            lf = self.loaded_files.get(fp)
            if lf is None or mi >= len(lf.measurements):
                continue
            meas = lf.measurements[mi]
            log = getattr(meas, "log", {}) or {}
            title = f"{Path(fp).name}  ::  {meas.title}"
            section = CollapsibleSection(
                title, expanded=getattr(self, "_log_other_expanded", False))

            inner = QtWidgets.QFormLayout()
            inner.setContentsMargins(20, 2, 2, 2)

            cv = QtWidgets.QPlainTextEdit(
                str(log.get("Comment") or meas.comment or ""))
            cv.setReadOnly(True)
            cv.setFixedHeight(50)
            inner.addRow("Comment:", cv)

            for k in readonly_keys:
                v = log.get(k, "")
                le = QtWidgets.QLineEdit("" if v is None else str(v))
                le.setReadOnly(True)
                inner.addRow(f"{k}:", le)

            section.set_layout(inner)
            self._log_other_layout.insertWidget(
                self._log_other_layout.count() - 1, section)

        self._refresh_logbook_indicator()
    def _rebuild_log_fields(self, log: dict):
        """Clear and rebuild dynamic field widgets from a log dict.

        Always shows Device and Subdevice. All other keys come from the
        log itself, so user-added fields appear here automatically.
        """
        self._log_fields = {}
        while self._log_form_layout.rowCount() > 0:
            self._log_form_layout.removeRow(0)

        forced = []
        readonly_keys = {"Filename", "RunID"}
        ordered = list(forced)
        for k in log.keys():
            if k == "Comment" or k in readonly_keys:
                continue
            if k not in ordered:
                ordered.append(k)

        for key in ordered:
            value = log.get(key)
            widget = self._build_log_field_widget(key, value)
            self._log_fields[key] = widget
            wrapped = self._wrap_log_field_with_copy(key, widget)
            self._log_form_layout.addRow(f"{key}:", wrapped)

    def _build_log_field_widget(self, key, value):
        """Type-dispatch a log value to the appropriate editor widget."""
        if isinstance(value, bool):
            return BoolToggleButton(
                key, value, self._on_log_value_changed,
                true_color=dmp.get_rgbhex_color("light green"),
                false_color=dmp.get_rgbhex_color("light red"))
        if isinstance(value, int):
            return TypedLogField(key, value, "int",
                                 self._on_log_value_changed)
        if isinstance(value, float):
            return TypedLogField(key, value, "float",
                                 self._on_log_value_changed)
        if isinstance(value, list):
            if all(isinstance(v, (int, float, str)) for v in value):
                return EditableListField(
                    key, value, self._on_log_value_changed)
            return ReadOnlyLogValue(key, value)
        if isinstance(value, dict):
            return ReadOnlyLogValue(key, value)
        # str or None
        return TypedLogField(key, value, "str",
                             self._on_log_value_changed)
    def _on_log_comment_changed(self):
        """Update in-memory comment; defer disk + plot to the debounce.
    
        Empty comment maps to None (not "") — consistent with field
        handling. Unchanged state is a no-op.
        """
        if self.loaded_file is None or self.current_meas_index is None:
            return
        meas = self.loaded_file.measurements[self.current_meas_index]
        txt = self._log_comment.toPlainText()
        new_val = txt if txt else None
    
        current = meas.log.get("Comment")
        if current == "":
            current = None
        if current == new_val:
            return
    
        meas.comment = new_val
        meas.log["Comment"] = new_val
        self._schedule_log_commit(affect_plot=True)
    
    def _on_log_value_changed(self, key: str, new_value):
        """Unified handler for every dynamic field widget.
    
        Widgets already do their own type conversion, so new_value is
        typed (None / bool / int / float / str / list). Empty fields
        are stored as None (or [] for lists).
        """
        if self.loaded_file is None or self.current_meas_index is None:
            return
        meas = self.loaded_file.measurements[self.current_meas_index]
        current = meas.log.get(key)
        if current == "":
            current = None
        if current == new_value:
            return
    
        meas.log[key] = new_value
        self._schedule_log_commit(affect_plot=False)
    
    def _persist_log_to_pkl(self):
        """Persist GUI edits through the folder store."""
        if self.loaded_file is None or self.loaded_file.folder_store is None:
            return
        if self.current_meas_index is None:
            return
    
        meas = self.loaded_file.measurements[self.current_meas_index]
        run_key = meas.metadata.get("run_key")
        if run_key is None:
            return
    
        filename = Path(self.current_file_path).name
        self.loaded_file.folder_store.save_log(
            filename, run_key, dict(meas.log))
        self._refresh_logbook_indicator()
        
    def _schedule_log_commit(self, affect_plot: bool = False):
        """Start (or restart) the debounce timer for a GUI log edit.

        The in-memory meas.log is already updated by the caller — this
        only defers the disk write and optional plot refresh. Captures
        the target (file_path, meas_index) so a mid-typing measurement
        switch cannot commit the wrong run.
        """
        if not hasattr(self, "_log_commit_timer"):
            self._log_commit_timer = QtCore.QTimer(self)
            self._log_commit_timer.setSingleShot(True)
            self._log_commit_timer.setInterval(400)
            self._log_commit_timer.timeout.connect(self._commit_log_edit_now)

        self._pending_log_commit_target = (
            self.current_file_path, self.current_meas_index)
        self._pending_log_commit_affects_plot = (
            getattr(self, "_pending_log_commit_affects_plot", False)
            or affect_plot)
        self._log_commit_timer.start()

    def _commit_log_edit_now(self):
        """Flush the pending log edit to disk and refresh the indicator."""
        target = getattr(self, "_pending_log_commit_target", None)
        affect_plot = getattr(self, "_pending_log_commit_affects_plot", False)
        self._pending_log_commit_target = None
        self._pending_log_commit_affects_plot = False

        if target is None:
            return
        fp, mi = target
        lf = self.loaded_files.get(fp)
        if lf is None or lf.folder_store is None:
            return
        if mi is None or mi >= len(lf.measurements):
            return

        meas = lf.measurements[mi]
        run_key = meas.metadata.get("run_key")
        if run_key is None:
            return

        filename = Path(fp).name
        lf.folder_store.save_log(Path(fp).name, run_key, dict(meas.log))
        self._refresh_logbook_indicator()

        if (affect_plot
                and fp == self.current_file_path
                and mi == self.current_meas_index
                and self.EDIT_SHOW_COMMENT):
            self._refresh_plot()

    def _flush_pending_log_commit(self):
        """Force-commit any pending edit immediately (before a switch
        or a close). Safe to call when nothing is pending."""
        if (hasattr(self, "_log_commit_timer")
                and self._log_commit_timer.isActive()):
            self._log_commit_timer.stop()
            self._commit_log_edit_now()
    def _build_classify_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
    
        info = QtWidgets.QLabel(
            "The classified view groups every run by device + subdevice.\n\n"
            "• Keithley files: taken directly from each run's log entry "
            "(no regex).\n"
            "• Fallback (PalmSens, or Keithley runs without log entries): "
            "the patterns below are tried, and the *best* match wins "
            "(longest device+subdevice, ties broken by list order).\n\n"
            "Patterns are stored per-root in .dmui_gui_state/, so they "
            "travel with the data folder. Unset roots fall back to the "
            "built-in defaults."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(info)
    
        # ── Root selector ─────────────────────────────────────────
        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(QtWidgets.QLabel("Editing patterns for:"))
        self.classify_root_combo = QtWidgets.QComboBox()
        self.classify_root_combo.currentIndexChanged.connect(
            self._on_classify_root_changed)
        root_row.addWidget(self.classify_root_combo, stretch=1)
        layout.addLayout(root_row)
    
        # ── List of patterns ──────────────────────────────────────
        layout.addWidget(QtWidgets.QLabel(
            "Patterns (checkbox = enabled; best match wins):"))
        self.classify_list = QtWidgets.QListWidget()
        self.classify_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.classify_list.itemChanged.connect(self._on_classify_item_changed)
        self.classify_list.itemSelectionChanged.connect(
            self._on_classify_selection_changed)
        layout.addWidget(self.classify_list, stretch=1)
    
        # ── Editor ────────────────────────────────────────────────
        edit_box = QtWidgets.QGroupBox("Edit / add pattern")
        ef = QtWidgets.QFormLayout(edit_box)
    
        self.classify_device_edit = QtWidgets.QLineEdit()
        self.classify_device_edit.setPlaceholderText(
            "shorthand: DFR[x]-[AA]    raw: DFR\\d+-[A-Z]{2}")
        self.classify_device_edit.textChanged.connect(
            self._refresh_classify_effective_regex)
        ef.addRow("Device pattern:", self.classify_device_edit)
    
        sub_row = QtWidgets.QHBoxLayout()
        self.classify_subdevice_edit = QtWidgets.QLineEdit()
        self.classify_subdevice_edit.setPlaceholderText(
            "shorthand: [AAx]    raw: [A-Z]{2}\\d+")
        self.classify_subdevice_edit.textChanged.connect(
            self._refresh_classify_effective_regex)
        self.classify_anchor_check = QtWidgets.QCheckBox(
            "Anchor after separator")
        self.classify_anchor_check.setChecked(True)
        self.classify_anchor_check.setToolTip(
            "Prefix the subdevice regex with (?<=[-_]) so it can only "
            "match right after '-' or '_' — prevents it from matching "
            "inside the device name.")
        self.classify_anchor_check.toggled.connect(
            self._refresh_classify_effective_regex)
        sub_row.addWidget(self.classify_subdevice_edit, stretch=1)
        sub_row.addWidget(self.classify_anchor_check)
        ef.addRow("Subdevice pattern:", sub_row)
    
        self.classify_label_edit = QtWidgets.QLineEdit()
        self.classify_label_edit.setPlaceholderText(
            "Optional description (e.g. 'DFR naming')")
        ef.addRow("Label:", self.classify_label_edit)
    
        mode_row = QtWidgets.QHBoxLayout()
        self.classify_mode_shorthand = QtWidgets.QRadioButton("Shorthand")
        self.classify_mode_raw = QtWidgets.QRadioButton("Raw regex")
        self.classify_mode_shorthand.setChecked(True)
        self.classify_mode_group = QtWidgets.QButtonGroup(self)
        self.classify_mode_group.addButton(self.classify_mode_shorthand, 0)
        self.classify_mode_group.addButton(self.classify_mode_raw, 1)
        self.classify_mode_group.buttonClicked.connect(
            self._refresh_classify_effective_regex)
        mode_row.addWidget(self.classify_mode_shorthand)
        mode_row.addWidget(self.classify_mode_raw)
        mode_row.addStretch(1)
        ef.addRow("Mode:", mode_row)
    
        self.classify_effective_label = QtWidgets.QLabel("")
        self.classify_effective_label.setStyleSheet(
            "color: #333; background: #f4f4f4; padding: 4px; "
            "font-family: monospace; font-size: 10px;")
        self.classify_effective_label.setWordWrap(True)
        ef.addRow(self.classify_effective_label)
    
        btn_row = QtWidgets.QHBoxLayout()
        add_btn    = QtWidgets.QPushButton("Add / Update")
        remove_btn = QtWidgets.QPushButton("Remove")
        up_btn     = QtWidgets.QPushButton("↑"); up_btn.setFixedWidth(30)
        dn_btn     = QtWidgets.QPushButton("↓"); dn_btn.setFixedWidth(30)
        import_btn = QtWidgets.QPushButton("Import from…")
        add_btn.clicked.connect(self._on_classify_add_or_update)
        remove_btn.clicked.connect(self._on_classify_remove)
        up_btn.clicked.connect(lambda: self._on_classify_move(-1))
        dn_btn.clicked.connect(lambda: self._on_classify_move(+1))
        import_btn.clicked.connect(self._on_classify_import_from)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addWidget(up_btn)
        btn_row.addWidget(dn_btn)
        btn_row.addWidget(import_btn)
        btn_row.addStretch(1)
        ef.addRow(btn_row)
    
        layout.addWidget(edit_box)
    
        # ── Preview ──────────────────────────────────────────────
        self.classify_preview = QtWidgets.QLabel("")
        self.classify_preview.setWordWrap(True)
        self.classify_preview.setStyleSheet(
            "color: #333; background: #f4f4f4; padding: 6px; "
            "font-size: 10px;")
        layout.addWidget(self.classify_preview)
    
        # ── Actions ──────────────────────────────────────────────
        action_row = QtWidgets.QHBoxLayout()
        toggle_btn = QtWidgets.QPushButton("Toggle view (flat / classified)")
        toggle_btn.clicked.connect(self._toggle_tree_view)
        rebuild_btn = QtWidgets.QPushButton("Rebuild measurement tree")
        rebuild_btn.clicked.connect(self._on_classify_rebuild)
        action_row.addWidget(toggle_btn)
        action_row.addWidget(rebuild_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)
    
        self._refresh_classify_root_combo()
        self.tab_widget.addTab(tab, "Classify")
    
    def _apply_classify_pattern(self):
        pat = self.classify_pattern_edit.text()
        self.EDIT_CLASSIFY_PATTERN = pat
        self._classify_cache.clear()
        self._rebuild_measurements_tree()
    
    def _toggle_tree_view(self):
        self.EDIT_TREE_VIEW = "classified" if self.EDIT_TREE_VIEW == "flat" else "flat"
        self._rebuild_measurements_tree()
    
    def _classify_path(self, path: Path) -> tuple[str, str]:
        """Return (device, subdevice) for a file, cached.
    
        Used only as a *fallback* for files without log data (e.g.
        PalmSens). Keithley files get their device/subdevice from the
        logbook via FolderStore.peek_classified().
    
        Fallback scheme:
          • Empty EDIT_CLASSIFY_PATTERN → folder name / file stem
          • Custom regex (must name groups 'device' and 'subdevice')
        """
        key = str(path)
        if key in self._classify_cache:
            return self._classify_cache[key]
    
        pattern = (self.EDIT_CLASSIFY_PATTERN or "").strip()
        result: tuple[str, str]
    
        if not pattern:
            result = (path.parent.name, path.stem)
        else:
            try:
                m = re.search(pattern, path.stem)
                if (m and "device" in m.groupdict()
                        and "subdevice" in m.groupdict()):
                    result = (m.group("device"), m.group("subdevice"))
                else:
                    result = (path.parent.name, path.stem)
            except re.error:
                result = (path.parent.name, path.stem)
    
        self._classify_cache[key] = result
        return result

    def _build_curves_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        # ── Mode toggle ────────────────────────────────────────────
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Plot mode:"))
        self.curves_mode_auto = QtWidgets.QRadioButton("Automatic")
        self.curves_mode_template = QtWidgets.QRadioButton("Template")
        self.curves_mode_auto.setChecked(True)
        self.curves_mode_group = QtWidgets.QButtonGroup(self)
        self.curves_mode_group.addButton(self.curves_mode_auto, 0)
        self.curves_mode_group.addButton(self.curves_mode_template, 1)
        self.curves_mode_group.buttonClicked.connect(
            self._on_curves_mode_changed)
        mode_row.addWidget(self.curves_mode_auto)
        mode_row.addWidget(self.curves_mode_template)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self.curves_mode_hint = QtWidgets.QLabel("")
        self.curves_mode_hint.setStyleSheet(
            "color: gray; font-size: 10px;")
        self.curves_mode_hint.setWordWrap(True)
        layout.addWidget(self.curves_mode_hint)

        # ── Curves tree (unchanged) ────────────────────────────────
        layout.addWidget(QtWidgets.QLabel("Ticked measurements"))
        self.curves_tree = QtWidgets.QTreeWidget()
        self.curves_tree.setHeaderLabels(["Curve", "Points", "Source"])
        self.curves_tree.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.curves_tree.itemChanged.connect(self._on_curve_item_changed)
        self.curves_tree.itemSelectionChanged.connect(
            self._on_curve_selection_changed)
        layout.addWidget(self.curves_tree)

        # ── Template editor (hidden in auto mode) ──────────────────
        self.curves_template_box = QtWidgets.QGroupBox("Plot template")
        tpl = QtWidgets.QVBoxLayout(self.curves_template_box)

        self.curves_spec_table = QtWidgets.QTableWidget(0, 4)
        self.curves_spec_table.setHorizontalHeaderLabels(
            ["X column", "Y column", "Segments", "Label"])
        hdr = self.curves_spec_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.curves_spec_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.curves_template_box)
        tpl.addWidget(self.curves_spec_table)

        spec_btn_row = QtWidgets.QHBoxLayout()
        add_row_btn = QtWidgets.QPushButton("+ Add")
        rm_row_btn = QtWidgets.QPushButton("– Remove")
        up_btn = QtWidgets.QPushButton("↑")
        dn_btn = QtWidgets.QPushButton("↓")
        up_btn.setFixedWidth(30)
        dn_btn.setFixedWidth(30)
        add_row_btn.clicked.connect(self._on_template_add_row)
        rm_row_btn.clicked.connect(self._on_template_remove_row)
        up_btn.clicked.connect(lambda: self._on_template_move_row(-1))
        dn_btn.clicked.connect(lambda: self._on_template_move_row(+1))
        spec_btn_row.addWidget(add_row_btn)
        spec_btn_row.addWidget(rm_row_btn)
        spec_btn_row.addWidget(up_btn)
        spec_btn_row.addWidget(dn_btn)
        spec_btn_row.addStretch(1)
        tpl.addLayout(spec_btn_row)

        self.curves_template_sig_label = QtWidgets.QLabel("Signature: —")
        self.curves_template_sig_label.setStyleSheet(
            "color: gray; font-size: 10px;")
        tpl.addWidget(self.curves_template_sig_label)

        lib_row = QtWidgets.QHBoxLayout()
        save_tpl_btn = QtWidgets.QPushButton("Save as template")
        apply_tpl_btn = QtWidgets.QPushButton("Apply from template…")
        reset_auto_btn = QtWidgets.QPushButton("Reset to automatic")
        save_tpl_btn.clicked.connect(self._on_template_save)
        apply_tpl_btn.clicked.connect(self._on_template_apply)
        reset_auto_btn.clicked.connect(self._on_template_reset)
        lib_row.addWidget(save_tpl_btn)
        lib_row.addWidget(apply_tpl_btn)
        lib_row.addWidget(reset_auto_btn)
        lib_row.addStretch(1)
        tpl.addLayout(lib_row)

        self.curves_template_box.setVisible(False)   # hidden in auto mode

        # ── Curve editor (unchanged) ───────────────────────────────
        edit_box = QtWidgets.QGroupBox("Edit selected curve")
        edit_form = QtWidgets.QFormLayout(edit_box)
        self._curve_edit_name = QtWidgets.QLineEdit()
        self._curve_edit_xunit = QtWidgets.QLineEdit()
        self._curve_edit_yunit = QtWidgets.QLineEdit()
        self._curve_edit_name.editingFinished.connect(
            lambda: self._on_curve_edit_commit("name"))
        self._curve_edit_xunit.editingFinished.connect(
            lambda: self._on_curve_edit_commit("x_unit"))
        self._curve_edit_yunit.editingFinished.connect(
            lambda: self._on_curve_edit_commit("y_unit"))
        edit_form.addRow("Name:", self._curve_edit_name)
        edit_form.addRow("X unit:", self._curve_edit_xunit)
        edit_form.addRow("Y unit:", self._curve_edit_yunit)
        layout.addWidget(edit_box)

        self._curve_edit_target: Optional[str] = None

        # ── Show/Hide/Untick buttons (unchanged) ──────────────────
        btn_row = QtWidgets.QHBoxLayout()
        show_all_btn = QtWidgets.QPushButton("Show all")
        show_all_btn.clicked.connect(
            lambda: self._set_all_curves_visible(True))
        hide_all_btn = QtWidgets.QPushButton("Hide all")
        hide_all_btn.clicked.connect(
            lambda: self._set_all_curves_visible(False))
        untick_root_btn = QtWidgets.QPushButton("Untick roots")
        untick_root_btn.setToolTip(
            "Untick every file/measurement group (leaves curves as-is)")
        untick_root_btn.clicked.connect(self._untick_all_curve_roots)
        btn_row.addWidget(show_all_btn)
        btn_row.addWidget(hide_all_btn)
        btn_row.addWidget(untick_root_btn)
        layout.addLayout(btn_row)

        self.tab_widget.addTab(tab, "Curves")
    
    def _untick_all_curve_roots(self):
        """Untick the file- and measurement-level nodes in the Curves tab,
        leaving curve ticks alone. Has no effect on the left file_list."""
        self.curves_tree.blockSignals(True)
        for i in range(self.curves_tree.topLevelItemCount()):
            source_root = self.curves_tree.topLevelItem(i)
            source_root.setCheckState(0, QtCore.Qt.Unchecked)
            for j in range(source_root.childCount()):
                file_root = source_root.child(j)
                file_root.setCheckState(0, QtCore.Qt.Unchecked)
                for k in range(file_root.childCount()):
                    meas_root = file_root.child(k)
                    meas_root.setCheckState(0, QtCore.Qt.Unchecked)
        self.curves_tree.blockSignals(False)
    
    def _build_curve_label(self, meas, y_unit="", seg_tag=None, override=None):
        """Build a curve legend label according to the Display-tab builder toggles.
    
        Order: RunID, Device, Subdevice, then Measured+Sweep combined.
    
        Rules for the trailing "Measured / Sweep" pair:
          * Both on  → $I_{fwd/rev}$  (or $I_{fwd}$ for single-segment)
          * Measured on, Sweep off → $I$
          * Measured off, Sweep on → (fwd/rev) plain-text fallback
            (no LaTeX base to hang a subscript onto)
          * Neither  → nothing appended
    
        If Measured cannot be identified (y_unit has no known symbol)
        → treated as "Measured off" for the purpose of the assembly.
    
        If *all* parts are disabled, falls back to the measurement title so
        the label is never empty.
        """
        if override:
            return override
    
        show_run       = getattr(self, 'EDIT_LEGEND_SHOW_RUN',       True)
        show_device    = getattr(self, 'EDIT_LEGEND_SHOW_DEVICE',    True)
        show_subdevice = getattr(self, 'EDIT_LEGEND_SHOW_SUBDEVICE', True)
        show_measured  = getattr(self, 'EDIT_LEGEND_SHOW_MEASURED',  True)
        show_sweep     = getattr(self, 'EDIT_LEGEND_SHOW_SWEEP',     True)
    
        # -- RunID --
        run_key = ""
        if meas.metadata:
            run_key = str(meas.metadata.get("run_key", "") or "")
        if not run_key:
            m = re.match(r"(Run\d+)", meas.title or "")
            if m:
                run_key = m.group(1)
    
        # -- Device / Subdevice (from log) --
        log = meas.log or {}
        dev = str(log.get("Device", "")    or "").strip()
        sub = str(log.get("Subdevice", "") or "").strip()
    
        # -- Measured symbol --
        sym = {"A": "I", "V": "V", "Ω": "R", "Ohm": "R"}.get(y_unit, "")
        measured_ok = show_measured and bool(sym)
        sweep_ok    = show_sweep    and bool(seg_tag)
    
        parts = []
        if show_run and run_key:
            parts.append(run_key)
        if show_device and dev and dev.lower() != "unlabelled":
            parts.append(dev)
        if show_subdevice and sub and sub != "SD0":
            parts.append(sub)
    
        if measured_ok and sweep_ok:
            parts.append(f"${sym}_{{{seg_tag}}}$")
        elif measured_ok:
            parts.append(f"${sym}$")
        elif sweep_ok:
            # No math base to subscript onto → use plain-text parenthetical
            parts.append(f"({seg_tag})")
    
        if not parts:
            return meas.title or "measurement"
    
        return " ".join(parts)
    
    def _on_legend_toggle(self, var_name, checked):
        """User clicked one of the legend-builder toggles."""
        setattr(self, var_name, bool(checked))
        self._refresh_legend_preview()
        self._refresh_plot()
        self._save_session_parameters()
    
    
    def _refresh_legend_preview(self):
        """Live example underneath the toggle row: what the active
        measurement's legend entry will look like with current settings."""
        if not hasattr(self, "legend_preview_label"):
            return
    
        if self.loaded_file is None or self.current_meas_index is None:
            self.legend_preview_label.setText("Preview: (no active measurement)")
            return
    
        meas = self.loaded_file.measurements[self.current_meas_index]
    
        # Representative y_unit: first visible curve's unit
        y_unit = ""
        for c in meas.curves:
            if c.name not in self.current_state.hidden_curves:
                y_unit = c.y_unit
                break
    
        has_segs = any(
            getattr(c, "segments", None) and len(c.segments) >= 2
            for c in meas.curves)
        seg_tag = "fwd/rev" if has_segs else None
    
        example = self._build_curve_label(
            meas, y_unit=y_unit, seg_tag=seg_tag)
    
        self.legend_preview_label.setText(f"Preview: {example}")
    def _build_problems_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
    
        hint = QtWidgets.QLabel(
            "Issues found while loading every ticked folder. "
            "Click a folder name to expand. If a folder's data looks wrong, "
            "use Advanced → Force rebuild pkl."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(hint)
    
        self.problems_tree = QtWidgets.QTreeWidget()
        self.problems_tree.setHeaderLabels(["Issue", "Path", "Where"])
        self.problems_tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.problems_tree.setColumnWidth(0, 480)
        layout.addWidget(self.problems_tree, stretch=1)
    
        btn_row = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_problems_tab)
        copy_btn = QtWidgets.QPushButton("Copy all")
        copy_btn.clicked.connect(self._copy_problems_to_clipboard)
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
    
        self.tab_widget.addTab(tab, "Problems")
    
    def _refresh_problems_tab(self):
    
        if not hasattr(self, "problems_tree"):
            return
        from Data_Containers import _STORE_CACHE
        self.problems_tree.clear()
    
        ticked = self._get_ticked_folders()
        if not ticked:
            item = QtWidgets.QTreeWidgetItem(["(no folders ticked)"])
            item.setForeground(0, QtGui.QColor("gray"))
            self.problems_tree.addTopLevelItem(item)
            return
    
        total = 0
        any_store_seen = False
    
        for folder in ticked:
            key = (Path(folder).resolve(), "keithley")
            store = _STORE_CACHE.get(key)
            if store is None:
                continue
            any_store_seen = True
            if not store.problems:
                continue
    
            folder_item = QtWidgets.QTreeWidgetItem(
                [f"{folder.name}  ({len(store.problems)} issue(s))",
                 str(folder), ""])
            font = folder_item.font(0)
            font.setBold(True)
            folder_item.setFont(0, font)
            folder_item.setExpanded(True)
            self.problems_tree.addTopLevelItem(folder_item)
    
            for p in store.problems:
                sev = (p.get("severity") or "error").lower()
                msg = p.get("message") or ""
                path = p.get("path") or ""
                line = p.get("line")
                col  = p.get("column")
                loc = ""
                if line is not None:
                    loc = f"L{line}"
                    if col is not None:
                        loc += f":{col}"
    
                text = f"[{sev.upper():7}] {msg}"
                child = QtWidgets.QTreeWidgetItem([text, path, loc])
    
                if sev == "error":
                    child.setForeground(0, QtGui.QColor("#b00020"))
                elif sev == "warning":
                    child.setForeground(0, QtGui.QColor("#a05000"))
                else:
                    child.setForeground(0, QtGui.QColor("#404040"))
    
                folder_item.addChild(child)
                total += 1
    
        if total == 0:
            label = ("No problems in any loaded folder."
                     if any_store_seen
                     else "No folders have been loaded yet.")
            item = QtWidgets.QTreeWidgetItem([label])
            item.setForeground(0, QtGui.QColor("darkgreen"))
            self.problems_tree.addTopLevelItem(item)
    
    def _copy_problems_to_clipboard(self):
        lines = []
        for i in range(self.problems_tree.topLevelItemCount()):
            folder_item = self.problems_tree.topLevelItem(i)
            lines.append(folder_item.text(0))
            for j in range(folder_item.childCount()):
                child = folder_item.child(j)
                lines.append(f"  - {child.text(0)}")
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
    # -- Folder management ----------------------------------------------------
    
    def _schedule_tree_rebuild(self, focus_after: Optional[str] = None):
        """Defer a tree rebuild to the next event-loop tick, as otherwise we crash!
    
        Calling _rebuild_folder_tree_ui() synchronously from an
        itemChanged / currentItemChanged handler destroys the QTreeWidgetItem
        that Qt is currently dispatching a signal for, which segfaults
        inside Qt's C++ layer. Deferring via QTimer avoids that.
        """
        if focus_after is not None:
            self._pending_focus = focus_after
        if self._tree_rebuild_pending:
            return
        self._tree_rebuild_pending = True
        QtCore.QTimer.singleShot(0, self._do_scheduled_rebuild)
    
    def _do_scheduled_rebuild(self):
        self._tree_rebuild_pending = False
        focus = self._pending_focus
        self._pending_focus = None
        self._rebuild_folder_tree_ui()
        if focus:
            self._focus_folder_tree_item(focus)
            
    def _rebuild_folder_tree_ui(self):
        """Flatten the leaves (via _added_entries + registry) and display
        them grouped by their state_root. Within each group, the longest
        shared directory prefix is shown as a header, and each leaf is
        shown as its relative path below it.
    
        This is intentionally NOT a per-segment tree — the user asked for
        a flat list under a shared header, e.g.:
    
            home/stuffA/stuffB/otherThings/Whatwhaat/ProbablyUpperDataroot/
              ☐ ActualDataFolder
              ☐ ActualDataFolder2
              ☐ ActualDataFolder3
        """
        self._in_tree_rebuild = True
        self.folder_tree.blockSignals(True)
        try:
            self.folder_tree.clear()
            # ── everything you already have, indented one level ──
            # (leaves collection, groups loop, header/leaf item creation)
            ...
        finally:
            self.folder_tree.blockSignals(False)
            self._in_tree_rebuild = False
    
        # ── Collect leaves (dedupe by resolved path) ────────────────
        leaves: list[tuple[Path, Path]] = []
        seen_leaves: set[str] = set()
        for entry in self._added_entries:
            p = Path(entry["path"])
            if entry.get("kind") == "root":
                for c in find_measurement_folders(p, self.registry):
                    rp = c.resolve()
                    if str(rp) in seen_leaves:
                        continue
                    seen_leaves.add(str(rp))
                    leaves.append((rp, p.resolve()))
            else:
                rp = p.resolve()
                if str(rp) in seen_leaves:
                    continue
                seen_leaves.add(str(rp))
                leaves.append((rp, rp))
    
        # ── state_root mapping + prune stale check-state ────────────
        self._state_roots = {str(lp): str(sr) for lp, sr in leaves}
        current_leaf_paths = set(self._state_roots.keys())
        for k in list(self._leaf_check.keys()):
            if k not in current_leaf_paths:
                del self._leaf_check[k]
    
        if not leaves:
            self.folder_tree.blockSignals(False)
            return
    
        # ── Group by state_root, one section per added root ─────────
        groups: dict[str, list[Path]] = {}
        for leaf_path, state_root in leaves:
            groups.setdefault(str(state_root), []).append(leaf_path)
    
        for state_root_str in sorted(groups):
            leaf_paths = sorted(groups[state_root_str])
    
            # ── Single-leaf groups: one row, full path, no header ──
            if len(leaf_paths) == 1:
                leaf = leaf_paths[0]
                item = QtWidgets.QTreeWidgetItem([str(leaf)])
                item.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsSelectable
                    | QtCore.Qt.ItemIsUserCheckable)
                is_ticked = self._leaf_check.get(str(leaf), False)
                item.setCheckState(
                    0, QtCore.Qt.Checked if is_ticked
                       else QtCore.Qt.Unchecked)
                item.setData(0, QtCore.Qt.UserRole,
                             ("folder_leaf", str(leaf)))
                item.setToolTip(0, str(leaf))
                self.folder_tree.addTopLevelItem(item)
                continue
    
            # ── 2+ leaves: common-prefix header + relative rows ─────
            common = self._common_path_prefix([str(p) for p in leaf_paths])
            if common:
                header = QtWidgets.QTreeWidgetItem([self._format_tree_header(common + os.sep)])
                header.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsUserCheckable
                    | QtCore.Qt.ItemIsAutoTristate)
                header.setData(
                    0, QtCore.Qt.UserRole,
                    ("folder_group", [str(lp) for lp in leaf_paths]))
                # Initial state from _leaf_check (all/any/none)
                _states = [self._leaf_check.get(str(lp), False)
                           for lp in leaf_paths]
                if all(_states):
                    header.setCheckState(0, QtCore.Qt.Checked)
                elif any(_states):
                    header.setCheckState(0, QtCore.Qt.PartiallyChecked)
                else:
                    header.setCheckState(0, QtCore.Qt.Unchecked)
                header.setToolTip(
                    0, f"{state_root_str}  (state root for sidecars)")
                header.setExpanded(True)
                self.folder_tree.addTopLevelItem(header)
                parent = header
            else:
                parent = None
    
            for leaf in leaf_paths:
                if common:
                    try:
                        rel = leaf.relative_to(Path(common))
                        label = str(rel) or leaf.name
                    except ValueError:
                        label = str(leaf)
                else:
                    label = str(leaf)
    
                item = QtWidgets.QTreeWidgetItem([label])
                item.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsSelectable
                    | QtCore.Qt.ItemIsUserCheckable)
                is_ticked = self._leaf_check.get(str(leaf), False)
                item.setCheckState(
                    0, QtCore.Qt.Checked if is_ticked
                       else QtCore.Qt.Unchecked)
                item.setData(0, QtCore.Qt.UserRole,
                             ("folder_leaf", str(leaf)))
                item.setToolTip(0, str(leaf))
                if parent is not None:
                    parent.addChild(item)
                else:
                    self.folder_tree.addTopLevelItem(item)
    
            if parent is not None:
                parent.setExpanded(True)
    
        self.folder_tree.blockSignals(False)
    
    def _format_tree_header(self, text: str, max_chars_per_line: int = 60) -> str:
        """Render a folder-group header as two lines: start of path on row 1,
        tail (the actual folder name at the end) on row 2. Anything in between
        is reachable via the widget's horizontal scrollbar.
    
        Keeping the tail visible on row 2 is the whole point: it tells the
        user *which* folder they're looking at without requiring them to
        scroll every time.
        """
        if len(text) <= max_chars_per_line:
            return text
    
        head = text[:max_chars_per_line]
        tail = "…" + text[-max_chars_per_line:]
        return f"{head}\n{tail}"
    
    def _common_path_prefix(self, paths: list[str]) -> str:
        """Longest directory prefix shared by all paths. Returns "" if
        the shared part is only the filesystem anchor (e.g. '/' on Unix,
        'C:\\' on Windows) — callers treat "" as 'no header'."""
        if not paths:
            return ""
        if len(paths) == 1:
            return str(Path(paths[0]).parent)
        try:
            common = os.path.commonpath(paths)
        except ValueError:
            return ""
        # Nested case: one leaf is an ancestor of another → back off
        # one level so no leaf renders as an empty row.
        if any(Path(p) == Path(common) for p in paths):
            common = str(Path(common).parent)
        # If the common prefix is just the filesystem root, treat as none
        if common in ("", ".", str(Path(common).anchor)):
            return ""
        return common    
    
    def _restore_folders(self):
        workspace = load_workspace()
    
        # Migrate old format if present
        if "added_entries" in workspace:
            self._added_entries = [
                {"path": e["path"], "kind": e.get("kind", "leaf")}
                for e in workspace["added_entries"]
            ]
            self._leaf_check = {
                str(k): bool(v)
                for k, v in workspace.get("leaf_check", {}).items()
            }
        else:
            old = workspace.get("folders", [])
            self._added_entries = [
                {"path": e["path"], "kind": "leaf"} for e in old
            ]
            self._leaf_check = {
                e["path"]: bool(e.get("checked", False)) for e in old
            }
    
        self._ticked_files_from_workspace = set(
            workspace.get("ticked_files", []))
        self._ticked_measurements_from_workspace = set()
        for row in workspace.get("ticked_measurements", []):
            try:
                fp, mi = row
                self._ticked_measurements_from_workspace.add(
                    (str(fp), int(mi)))
            except (TypeError, ValueError):
                continue
    
        active = workspace.get("active_measurement")
        active_fp = active_mi = None
        if isinstance(active, list) and len(active) == 2:
            try:
                active_fp = str(active[0])
                active_mi = int(active[1])
            except (ValueError, TypeError):
                active_fp = active_mi = None
    
        # Load filter state before the tree is (re)built so the first
        # rebuild already applies it.
        filter_state = workspace.get("filter", {}) or {}
        self._filter_enabled = bool(filter_state.get("enabled", False))
        self._filter_mode = filter_state.get("mode", "AND")
        self._filter_conditions = [
            dict(c) for c in filter_state.get("conditions", [])
        ]
        # Sync the toggle button to the restored state
        if hasattr(self, "filter_enable_btn"):
            self.filter_enable_btn.blockSignals(True)
            self.filter_enable_btn.setChecked(self._filter_enabled)
            self.filter_enable_btn.setText(
                "Filtering: ON" if self._filter_enabled else "Filtering: OFF")
            self.filter_enable_btn.blockSignals(False)
        if hasattr(self, "filter_mode_and"):
            self.filter_mode_and.blockSignals(True)
            self.filter_mode_or.blockSignals(True)
            (self.filter_mode_and if self._filter_mode == "AND"
             else self.filter_mode_or).setChecked(True)
            self.filter_mode_and.blockSignals(False)
            self.filter_mode_or.blockSignals(False)
        
        saved_focus = workspace.get("current_folder")
        
        self._rebuild_folder_tree_ui()
    
        # Transiently tick the leaf containing the active measurement
        if active_fp is not None:
            parent = str(Path(active_fp).parent.resolve())
            if parent in self._leaf_check and not self._leaf_check[parent]:
                self._leaf_check[parent] = True
                self._temp_ticked_folders.add(parent)
                self._rebuild_folder_tree_ui()
    
        focus_path = (saved_focus
                      or (str(Path(active_fp).parent.resolve())
                          if active_fp is not None else None))
        if focus_path:
            self._focus_folder_tree_item(focus_path)
    
        self._rebuild_measurements_tree()
    
        if active_fp is not None and active_mi is not None:
            try:
                self._set_active_measurement(active_fp, 0, active_mi)
            except Exception as e:
                print(f"Could not restore active measurement: {e}")
    
    def _focus_folder_tree_item(self, path: str):
        def _walk(item):
            if item is None:
                return None
            payload = item.data(0, QtCore.Qt.UserRole)
            if payload and payload[0] == "folder_leaf" \
                    and payload[1] == path:
                return item
            for j in range(item.childCount()):
                hit = _walk(item.child(j))
                if hit is not None:
                    return hit
            return None
    
        for i in range(self.folder_tree.topLevelItemCount()):
            hit = _walk(self.folder_tree.topLevelItem(i))
            if hit is not None:
                self.folder_tree.blockSignals(True)
                self.folder_tree.setCurrentItem(hit)
                self.folder_tree.blockSignals(False)
                return
    def _add_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add data folder")
        if not directory:
            return
        p = str(Path(directory).resolve())
        for entry in self._added_entries:
            if entry["path"] == p:
                return
        self._added_entries.append({"path": p, "kind": "leaf"})
        self._leaf_check[p] = False
        self._save_workspace()
        self._schedule_tree_rebuild()
        self._refresh_classify_root_combo()
        
    def _add_root_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add root folder (auto-scan for measurement folders)")
        if not directory:
            return
        p = str(Path(directory).resolve())
        for entry in self._added_entries:
            if entry["path"] == p:
                return
        found = find_measurement_folders(Path(p), self.registry)
        if not found:
            QtWidgets.QMessageBox.information(
                self, "No measurement folders",
                f"No folders containing loadable files were found under:\n{p}")
            return
        self._added_entries.append({"path": p, "kind": "root"})
        self._save_workspace()
        self._schedule_tree_rebuild()
        self._refresh_classify_root_combo()
        self._rebuild_measurements_tree()
        
    def _delete_folder(self):
        item = self.folder_tree.currentItem()
        if item is None:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None or payload[0] != "folder_leaf":
            return
        leaf_path = payload[1]

        target_path = None
        kind = None
        for entry in self._added_entries:
            if entry["path"] == leaf_path:
                target_path, kind = entry["path"], "leaf"
                break
            if entry["kind"] == "root":
                try:
                    Path(leaf_path).relative_to(entry["path"])
                    target_path, kind = entry["path"], "root"
                    break
                except ValueError:
                    continue
        if target_path is None:
            return

        msg = (f"Remove this root and all its discovered folders?\n{target_path}"
               if kind == "root"
               else f"Remove folder?\n{target_path}")
        reply = QtWidgets.QMessageBox.question(
            self, "Confirm removal", msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._added_entries = [
            e for e in self._added_entries if e["path"] != target_path
        ]
        self._leaf_check.pop(leaf_path, None)
        self._save_workspace()
        self._schedule_tree_rebuild()
        self._rebuild_measurements_tree()

    def _reset_loaded_data(self):
        self.current_file_path = None
        self.loaded_file = None
        self.current_file_index = None
        self.current_meas_index = None
        self.current_filename = None
        self.fig.clear()
        self.canvas.draw()
    # -- Measurement list---------------

    def _on_folder_tree_selected(self, item, _prev=None):
        # Ignore selection events that fire while we're rebuilding the
        # tree — Qt auto-sets currentItem to the first leaf on rebuild,
        # which would otherwise transient-tick a folder the user never
        # clicked.
        if self._in_tree_rebuild:
            return
        if item is None:
            self.current_folder = None
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None or payload[0] != "folder_leaf":
            return
        leaf_path = payload[1]

        for p in list(self._temp_ticked_folders):
            if p != leaf_path:
                self._leaf_check[p] = False
                self._temp_ticked_folders.discard(p)

        if not self._leaf_check.get(leaf_path, False):
            self._leaf_check[leaf_path] = True
            self._temp_ticked_folders.add(leaf_path)
            self._schedule_tree_rebuild(focus_after=leaf_path)

        self.current_folder = Path(leaf_path)
        self._save_workspace()
        self._rebuild_measurements_tree()

    def _on_folder_tree_check_changed(self, item, column):
        if self._in_tree_rebuild:
            return
        if column != 0:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None:
            return

        if payload[0] == "folder_group":
            new_state = item.checkState(0)
            if new_state == QtCore.Qt.PartiallyChecked:
                return
            checked = (new_state == QtCore.Qt.Checked)
            for lp in payload[1]:
                self._leaf_check[lp] = checked
                self._temp_ticked_folders.discard(lp)
            self._save_workspace()
            self._schedule_tree_rebuild()
            self._rebuild_measurements_tree()
            return

        if payload[0] != "folder_leaf":
            return
        leaf_path = payload[1]
        checked = item.checkState(0) == QtCore.Qt.Checked
        self._leaf_check[leaf_path] = checked
        self._temp_ticked_folders.discard(leaf_path)
        self._save_workspace()
        self._schedule_tree_rebuild()
        self._rebuild_measurements_tree()

    def _on_entry_selected(self, item, _column=0):
        """Click handler for 'click on already-current item' — the case where
        itemSelectionChanged won't fire. Handles the plain re-click case only.
        """
        if item is None:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None or payload[0] != "meas":
            return
        _, file_path, file_index, meas_index = payload
        if (file_path, meas_index) == self.active_measurement:
            return
        self._set_active_measurement(file_path, file_index, meas_index)
    
    def _on_selection_changed(self):
        """Qt fires this whenever the highlighted set changes, including
        ctrl/shift multi-selects and arrow-key navigation. Update the
        selection list, promote the newest addition to 'active', then replot.
    
        'Active' is driven by QTreeWidget.currentItem() — the cursor that
        moves on every arrow key, shift-extension, and click. This makes
        arrow-down, shift+arrow-down, shift+click, and ctrl+click all
        promote the newly-touched measurement to active, so the Logbook tab
        and Curves tab follow the user's focus without extra clicks.
        """
        sel = []
        for item in self.file_list.selectedItems():
            payload = item.data(0, QtCore.Qt.UserRole)
            if payload is None or payload[0] != "meas":
                continue
            _, file_path, _, meas_index = payload
            sel.append((file_path, meas_index))
        self.selected_measurements = sel
    
        # ── Candidate for active = Qt's currentItem, if it's a selected
        #    measurement. Falls through if current is deselected (ctrl+click
        #    removal) or a non-measurement row.
        candidate = None
        current = self.file_list.currentItem()
        if current is not None and current.isSelected():
            payload = current.data(0, QtCore.Qt.UserRole)
            if payload is not None and payload[0] == "meas":
                _, fp, _, mi = payload
                candidate = (fp, mi)
    
        if candidate is not None and candidate != self.active_measurement:
            self._set_active_measurement(candidate[0], 0, candidate[1])
            if self.active_measurement == candidate:
                return  # _set_active_measurement already refreshed
    
        # Fallback: no active yet (first ever selection after load)
        if self.active_measurement is None and sel:
            fp, mi = sel[-1]
            self._set_active_measurement(fp, 0, mi)
            if self.active_measurement == (fp, mi):
                return
    
        # Fallback: refresh using whatever state we have (also covers
        # _set_active_measurement failure, e.g. file load error).
        self._refresh_curves_tab()
        self._refresh_logbook_tab()
        self._refresh_plot()
    
    def _on_curve_selection_changed(self):
        items = self.curves_tree.selectedItems()
        if not items:
            self._curve_edit_target = None
            for w in (self._curve_edit_name, self._curve_edit_xunit, self._curve_edit_yunit):
                w.blockSignals(True); w.clear(); w.blockSignals(False)
            return
        item = items[-1]
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None or payload[0] != "curve":
            self._curve_edit_target = None
            return
        _, file_path, meas_idx, curve_name = payload
        # only allow editing the active measurement's curves (else no state to save into)
        if not (file_path_match(file_path, self.current_file_path)
                and meas_idx == self.current_meas_index):
            self._curve_edit_target = None
            return
        self._curve_edit_target = curve_name
        self._refresh_curve_editor()
    
    def _refresh_curve_editor(self):
        if self._curve_edit_target is None or self.loaded_file is None:
            return
        meas = self.loaded_file.measurements[self.current_meas_index]
        curve = next((c for c in meas.curves if c.name == self._curve_edit_target), None)
        if curve is None:
            return
        ov = self.current_state.curve_overrides.get(curve.name, {})
        for w, key, default in (
            (self._curve_edit_name, "name", curve.name),
            (self._curve_edit_xunit, "x_unit", curve.x_unit),
            (self._curve_edit_yunit, "y_unit", curve.y_unit),
        ):
            w.blockSignals(True)
            w.setText(str(ov.get(key, default)))
            w.blockSignals(False)
    
    def _on_curve_edit_commit(self, field: str):
        if self._curve_edit_target is None:
            return
        ov = self.current_state.curve_overrides.setdefault(self._curve_edit_target, {})
        widget = {"name": self._curve_edit_name,
                  "x_unit": self._curve_edit_xunit,
                  "y_unit": self._curve_edit_yunit}[field]
        ov[field] = widget.text()
        self._save_state()
        self._refresh_plot()
        self._refresh_curves_tab()
        
    def _set_active_measurement(self, file_path: str, file_index: int, meas_index: int):
        """Set the one measurement that tabs/logbook/curves operate on and that
        fits are performed against. Loads file if not cached."""
        # Flush any in-flight log edit before we swap targets, so the
        # debounce timer can't commit the wrong run.
        self._flush_pending_log_commit()

        if getattr(self, "EDIT_DISABLE_MULTILOAD", False):
            others = [p for p in self.loaded_files.keys() if p != file_path]
            if others:
                QtWidgets.QMessageBox.information(
                    self, "Multi-load disabled",
                    "Advanced settings forbid loading more than one file at a time."
                )
                return
    
        if file_path not in self.loaded_files:
            try:
                self.loaded_files[file_path] = self.registry.load(Path(file_path))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Load failed", f"Could not load {file_path}:\n{exc}"
                )
                return
    
        self.loaded_file = self.loaded_files[file_path]
        self.current_file_path = file_path
        self.current_file_index = file_index
        self.current_meas_index = meas_index
        self.current_filename = Path(file_path).name
        self.active_measurement = (file_path, meas_index)
    
        # Load sidecar state
        store, key_prefix = self._state_store_for(file_path)
        key = state_key(key_prefix, meas_index)
        self.current_state = store.load(key)
        # Auto-apply library template if this measurement has never
        # seen one. plot_template_auto_applied prevents re-application
        # after the user explicitly resets or edits.
        if (self.current_state.plot_template is None
                and not self.current_state.plot_template_auto_applied):
            sig = self._measurement_signature(
                self.loaded_file.measurements[meas_index])
            if sig:
                lib = self._load_plot_template_library(file_path)
                if sig in lib:
                    self.current_state.plot_template = [
                        dict(s) for s in lib[sig]]
                    self.current_state.plot_template_auto_applied = True
                    self._save_state()
        # Reflect state in UI (existing code, unchanged)
        self.remove_bg_toggle.blockSignals(True)
        self.remove_bg_toggle.setChecked(self.current_state.remove_background)
        self.remove_bg_toggle.blockSignals(False)
    
        self.bg_mode_combo.blockSignals(True)
        self.bg_mode_combo.setCurrentIndex(self.current_state.bg_removal_mode)
        self.bg_mode_combo.blockSignals(False)
    
        self.poly_degree_spin.blockSignals(True)
        self.poly_degree_spin.setValue(self.current_state.poly_degree)
        self.poly_degree_spin.blockSignals(False)
    
        self.spline_smoothing_spin.blockSignals(True)
        self.spline_smoothing_spin.setValue(self.current_state.spline_smoothing)
        self.spline_smoothing_spin.blockSignals(False)
    
        self.poly_degree_spin.setVisible(self.current_state.bg_removal_mode == BgRemovalMode.POLY)
        self.spline_smoothing_spin.setVisible(self.current_state.bg_removal_mode == BgRemovalMode.SPLINE)
        if self.current_state.plot_template is not None:
            self.curves_mode_template.setChecked(True)
            self.curves_template_box.setVisible(True)
            self.curves_mode_hint.setText(
                "Template mode: the plot shows exactly the rows below. "
                "Curve ticks affect only the fit-target list.")
        else:
            self.curves_mode_auto.setChecked(True)
            self.curves_template_box.setVisible(False)
            self.curves_mode_hint.setText(
                "Automatic mode: every non-hidden curve is drawn, "
                "one axis per Y unit.")
        self._selected_row = None
        self._refresh_curves_tab()
        self._refresh_range_list()
        self._refresh_fit_curve_list()
        self._refresh_logbook_tab()          
        self._refresh_curve_editor()
        self._refresh_classify_root_combo()
        self._refresh_classify_preview()        
        self._refresh_template_editor() 
        self._refresh_plot()
    
    def _rebuild_measurements_tree(self):
        """Populate self.file_list from every ticked folder."""
        # Pkl may have been rewritten; drop any cached pkl log lookups
        # so the next filter application re-reads fresh data.
        if hasattr(self, "_filter_log_cache"):
            self._filter_log_cache.clear()
    
        # ── Snapshot current tick state (works in flat AND classified) ──
        previously_ticked_meas = self._get_ticked_measurements()
        if not previously_ticked_meas:
            previously_ticked_meas = set(
                getattr(self, "_ticked_measurements_from_workspace", set()))
        previously_ticked_files = self._get_ticked_file_paths()
        if not previously_ticked_files:
            previously_ticked_files = set(
                getattr(self, "_ticked_files_from_workspace", set()))
    
        self.file_list.blockSignals(True)
        self.file_list.clear()
    
        ticked_folders = self._get_ticked_folders()
    
        # ── Classified view ─────────────────────────────────────────
        if self.EDIT_TREE_VIEW == "classified":
            # tree[dev][sub] = [(file_path, meas_id, title), …]
            tree: dict[str, dict[str, list[tuple[str, int, str]]]] = {}
    
            for folder_path in ticked_folders:
                if not folder_path.is_dir():
                    continue
                try:
                    all_paths = sorted(p for p in folder_path.iterdir()
                                       if p.is_file())
                except OSError:
                    continue
    
                for f in all_paths:
                    src = self.registry.find(f)
                    if src is None:
                        continue
                
                    file_dev, file_sub = self._classify_path(f)
                
                    try:
                        classified = src.peek_classified(f)
                    except Exception as exc:
                        logger.error(
                            f"peek_classified failed for {f}: {exc}")
                        continue
                
                    for meas_id, title, log_dev, log_sub in classified:
                        dev = log_dev or file_dev
                        sub = log_sub or file_sub
                        tree.setdefault(dev, {}) \
                            .setdefault(sub, []) \
                            .append((str(f), int(meas_id), title))
    
            def _run_sort(entry):
                m = re.search(r"Run(\d+)", entry[2])
                if m:
                    return (0, int(m.group(1)), entry[2])
                return (1, 0, entry[2])
            
            # Count run_id occurrences within each (dev, sub) to know
            # which need a (1)/(2)/(3) suffix.
            run_id_totals: dict[tuple, int] = {}
            for dev, subs in tree.items():
                for sub, entries in subs.items():
                    for _fp, _mid, title in entries:
                        m = re.search(r"(Run\d+)", title)
                        if m:
                            key = (dev, sub, m.group(1))
                            run_id_totals[key] = run_id_totals.get(key, 0) + 1
            
            run_id_seen: dict[tuple, int] = {}
            for dev, subs in sorted(tree.items()):
                dev_node = QtWidgets.QTreeWidgetItem([dev])
                dev_node.setFlags(QtCore.Qt.ItemIsEnabled)
                dev_node.setExpanded(True)
                self.file_list.addTopLevelItem(dev_node)
            
                for sub, entries in sorted(subs.items()):
                    sub_node = QtWidgets.QTreeWidgetItem([sub])
                    sub_node.setFlags(QtCore.Qt.ItemIsEnabled)
                    sub_node.setExpanded(True)
                    dev_node.addChild(sub_node)
            
                    for fp, meas_id, title in sorted(entries, key=_run_sort):
                        m = re.search(r"(Run\d+)", title)
                        display = title
                        if m:
                            key = (dev, sub, m.group(1))
                            if run_id_totals.get(key, 0) > 1:
                                run_id_seen[key] = run_id_seen.get(key, 0) + 1
                                display = f"{title} ({run_id_seen[key]})"
            
                        was_ticked = (
                            (fp, meas_id) in previously_ticked_meas
                            or fp in previously_ticked_files
                        )
                        child = QtWidgets.QTreeWidgetItem([display])
                        child.setFlags(
                            QtCore.Qt.ItemIsEnabled
                            | QtCore.Qt.ItemIsSelectable
                            | QtCore.Qt.ItemIsUserCheckable
                        )
                        child.setCheckState(
                            0, QtCore.Qt.Checked if was_ticked
                               else QtCore.Qt.Unchecked)
                        child.setData(0, QtCore.Qt.UserRole,
                                      ("meas", fp, 0, meas_id))
                        sub_node.addChild(child)
    
            self.file_list.expandAll()
            self.file_list.blockSignals(False)
            self._apply_filter_to_tree()
            self._refresh_curves_tab()
            self._refresh_logbook_tab()
            self._refresh_problems_tab()
            return
    
        # ── Flat view ──────────────────────────────────────────────
        for folder_path in ticked_folders:
            if not folder_path.is_dir():
                continue
        
            folder_node = QtWidgets.QTreeWidgetItem([str(folder_path.name)])
            folder_node.setFlags(QtCore.Qt.ItemIsEnabled)
            folder_node.setExpanded(True)
            self.file_list.addTopLevelItem(folder_node)
        
            try:
                all_paths = sorted(
                    p for p in folder_path.iterdir() if p.is_file())
            except OSError:
                continue
        
            for f in all_paths:
                src = self.registry.find(f)
                if src is None:
                    continue
        
                header = QtWidgets.QTreeWidgetItem([f.name])
                header.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsUserCheckable)
                was_ticked = str(f) in previously_ticked_files
                header.setCheckState(
                    0, QtCore.Qt.Checked if was_ticked
                       else QtCore.Qt.Unchecked)
                header.setData(0, QtCore.Qt.UserRole, ("file", str(f), 0))
                folder_node.addChild(header)
        
                try:
                    measurements = src.list_measurements(f)
                except Exception as exc:
                    err = QtWidgets.QTreeWidgetItem(
                        [f"    (failed to load: {exc})"])
                    err.setFlags(QtCore.Qt.NoItemFlags)
                    header.addChild(err)
                    header.setExpanded(True)
                    continue
        
                for meas_index, meas_title in measurements:
                    child = QtWidgets.QTreeWidgetItem(
                        [f"    {meas_index}: {meas_title}"])
                    child.setFlags(
                        QtCore.Qt.ItemIsEnabled
                        | QtCore.Qt.ItemIsSelectable
                        | QtCore.Qt.ItemIsUserCheckable)
                    was_ticked = (
                        (str(f), meas_index) in previously_ticked_meas
                        or str(f) in previously_ticked_files)
                    child.setCheckState(
                        0, QtCore.Qt.Checked if was_ticked
                           else QtCore.Qt.Unchecked)
                    child.setData(
                        0, QtCore.Qt.UserRole,
                        ("meas", str(f), 0, meas_index))
                    header.addChild(child)
        
                header.setExpanded(True)
        
        self.file_list.expandAll()
        self.file_list.blockSignals(False)
        self._apply_filter_to_tree()
        self._refresh_curves_tab()
        self._refresh_logbook_tab()
        self._refresh_problems_tab()
        
        
    def _activate_measurement(self, file_path, file_index, meas_index):
        """Load (from cache if possible) and display a single measurement.
        This is the transient-focus path; it loads the file into memory but
        does not tick the file or its children."""
        if getattr(self, "EDIT_DISABLE_MULTILOAD", False):
            others = [p for p in self.loaded_files.keys() if p != file_path]
            if others:
                QtWidgets.QMessageBox.information(
                    self, "Multi-load disabled",
                    "Advanced settings forbid loading more than one file at a time."
                )
                return

        if file_path not in self.loaded_files:
            try:
                self.loaded_files[file_path] = self.registry.load(Path(file_path))
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self, "Load failed", f"Could not load {file_path}:\n{exc}"
                )
                return

        self.loaded_file = self.loaded_files[file_path]
        self.current_file_path = file_path
        self.current_file_index = file_index
        self.current_meas_index = meas_index
        self.current_filename = Path(file_path).name

        # The StateStore folder is whichever folder this file lives in.
        store, key_prefix = self._state_store_for(file_path)
        key = state_key(key_prefix, meas_index)
        self.current_state = store.load(key)

        # Update UI to reflect loaded state
        self.remove_bg_toggle.blockSignals(True)
        self.remove_bg_toggle.setChecked(self.current_state.remove_background)
        self.remove_bg_toggle.blockSignals(False)

        self.bg_mode_combo.blockSignals(True)
        self.bg_mode_combo.setCurrentIndex(self.current_state.bg_removal_mode)
        self.bg_mode_combo.blockSignals(False)

        self.poly_degree_spin.blockSignals(True)
        self.poly_degree_spin.setValue(self.current_state.poly_degree)
        self.poly_degree_spin.blockSignals(False)

        self.spline_smoothing_spin.blockSignals(True)
        self.spline_smoothing_spin.setValue(self.current_state.spline_smoothing)
        self.spline_smoothing_spin.blockSignals(False)

        self.poly_degree_spin.setVisible(self.current_state.bg_removal_mode == BgRemovalMode.POLY)
        self.spline_smoothing_spin.setVisible(self.current_state.bg_removal_mode == BgRemovalMode.SPLINE)

        self._selected_row = None
        self._refresh_curves_tab()
        self._refresh_range_list()
        self._refresh_fit_curve_list()
        self._refresh_logbook_tab()
        self._refresh_curve_editor()
        self._refresh_plot()
        
    def _iter_all_items(self):
        """Yield every tree item in file_list (DFS), regardless of shape."""
        stack = [self.file_list.topLevelItem(i)
                 for i in range(self.file_list.topLevelItemCount())]
        while stack:
            item = stack.pop()
            yield item
            for j in range(item.childCount()):
                stack.append(item.child(j))    
    
    
    
    def _save_workspace(self):
        for p in list(self._temp_ticked_folders):
            self._temp_ticked_folders.discard(p)
            self._leaf_check[p] = False
    
        ticked = self._get_ticked_measurements()
    
        active = None
        if (self.current_file_path is not None
                and self.current_meas_index is not None):
            active = [str(self.current_file_path),
                      int(self.current_meas_index)]
    
        current_folder = None
        if self.folder_tree.currentItem() is not None:
            payload = self.folder_tree.currentItem().data(
                0, QtCore.Qt.UserRole)
            if payload and payload[0] == "folder_leaf":
                current_folder = payload[1]
    
        save_workspace({
            "added_entries": [dict(e) for e in self._added_entries],
            "leaf_check": dict(self._leaf_check),
            "ticked_files": list({fp for fp, _ in ticked}),
            "ticked_measurements": [list(t) for t in sorted(ticked)],
            "active_measurement": active,
            "current_folder": current_folder,
            "filter": {
                "enabled": self._filter_enabled,
                "mode": self._filter_mode,
                "conditions": [
                    {k: v for k, v in c.items()
                     if not str(k).startswith("_")}
                    for c in self._filter_conditions
                ],
            },
        })
    
    # -- classify tab: root selector --------------------------------------

    def _refresh_classify_root_combo(self):
        if not hasattr(self, "classify_root_combo"):
            return
        self.classify_root_combo.blockSignals(True)
        self.classify_root_combo.clear()
        for entry in self._added_entries:
            r = str(Path(entry["path"]).resolve())
            self.classify_root_combo.addItem(r, userData=r)
        # Default selection: current file's root if any
        if self.current_file_path:
            store, _ = self._state_store_for(self.current_file_path)
            cur = str(store.state_dir.parent)
            idx = self.classify_root_combo.findData(cur)
            if idx >= 0:
                self.classify_root_combo.setCurrentIndex(idx)
        self.classify_root_combo.blockSignals(False)
        self._on_classify_root_changed()
    
    def _on_classify_root_changed(self):
        root = self.classify_root_combo.currentData()
        if not root:
            self._classify_patterns_editing = []
        else:
            self._classify_patterns_editing = load_classify_patterns(Path(root))
        self._refresh_classify_list()
        self._refresh_classify_preview()
    
    # -- classify tab: pattern list ---------------------------------------
    
    def _refresh_classify_list(self):
        if not hasattr(self, "classify_list"):
            return
        self.classify_list.blockSignals(True)
        self.classify_list.clear()
        for i, entry in enumerate(self._classify_patterns_editing):
            mode = entry.get("mode", "shorthand")
            tag = "[S]" if mode == "shorthand" else "[R]"
            label = entry.get("label", "")
            display = (f"{tag} {entry['device_pattern']}  |  "
                       f"{entry['subdevice_pattern']}")
            if label:
                display += f"   — {label}"
            item = QtWidgets.QListWidgetItem(display)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked if entry.get("enabled", True)
                else QtCore.Qt.Unchecked)
            item.setData(QtCore.Qt.UserRole, i)
            self.classify_list.addItem(item)
        self.classify_list.blockSignals(False)
    
    def _on_classify_item_changed(self, item):
        idx = item.data(QtCore.Qt.UserRole)
        if idx is None or not (0 <= idx < len(self._classify_patterns_editing)):
            return
        self._classify_patterns_editing[idx]["enabled"] = (
            item.checkState() == QtCore.Qt.Checked)
        self._persist_classify_patterns()
    
    def _on_classify_selection_changed(self):
        items = self.classify_list.selectedItems()
        if not items:
            return
        idx = items[-1].data(QtCore.Qt.UserRole)
        if idx is None or not (0 <= idx < len(self._classify_patterns_editing)):
            return
        entry = self._classify_patterns_editing[idx]
        self.classify_device_edit.setText(entry.get("device_pattern", ""))
        self.classify_subdevice_edit.setText(entry.get("subdevice_pattern", ""))
        self.classify_label_edit.setText(entry.get("label", ""))
        self.classify_anchor_check.setChecked(
            entry.get("anchor_subdevice", True))
        if entry.get("mode", "shorthand") == "raw":
            self.classify_mode_raw.setChecked(True)
        else:
            self.classify_mode_shorthand.setChecked(True)
        self._refresh_classify_effective_regex()
    
    def _on_classify_add_or_update(self):
        dev = self.classify_device_edit.text().strip()
        sub = self.classify_subdevice_edit.text().strip()
        label = self.classify_label_edit.text().strip()
        mode = "raw" if self.classify_mode_raw.isChecked() else "shorthand"
        anchor = self.classify_anchor_check.isChecked()
        if not dev or not sub:
            return
        # Validate by trying to compile both halves now
        try:
            re.compile(_effective_regex(dev, mode))
            re.compile(_effective_regex(sub, mode))
        except re.error as e:
            QtWidgets.QMessageBox.warning(
                self, "Invalid pattern",
                f"Could not compile pattern:\n{e}")
            return
    
        entry = {
            "device_pattern":    dev,
            "subdevice_pattern": sub,
            "label":             label,
            "enabled":           True,
            "mode":              mode,
            "anchor_subdevice":  anchor,
        }
        items = self.classify_list.selectedItems()
        if items:
            idx = items[-1].data(QtCore.Qt.UserRole)
            if idx is not None and 0 <= idx < len(self._classify_patterns_editing):
                entry["enabled"] = self._classify_patterns_editing[idx].get(
                    "enabled", True)
                self._classify_patterns_editing[idx] = entry
            else:
                self._classify_patterns_editing.append(entry)
        else:
            self._classify_patterns_editing.append(entry)
    
        self._persist_classify_patterns()
        self._refresh_classify_list()
        self._refresh_classify_preview()
    
    def _on_classify_remove(self):
        items = self.classify_list.selectedItems()
        if not items:
            return
        idx = items[-1].data(QtCore.Qt.UserRole)
        if idx is None or not (0 <= idx < len(self._classify_patterns_editing)):
            return
        del self._classify_patterns_editing[idx]
        self._persist_classify_patterns()
        self._refresh_classify_list()
        self._refresh_classify_preview()
        self.classify_device_edit.clear()
        self.classify_subdevice_edit.clear()
        self.classify_label_edit.clear()
    
    def _on_classify_move(self, delta):
        items = self.classify_list.selectedItems()
        if not items:
            return
        idx = items[-1].data(QtCore.Qt.UserRole)
        if idx is None:
            return
        new_idx = idx + delta
        if not (0 <= new_idx < len(self._classify_patterns_editing)):
            return
        p = self._classify_patterns_editing
        p[idx], p[new_idx] = p[new_idx], p[idx]
        self._persist_classify_patterns()
        self._refresh_classify_list()
        if 0 <= new_idx < self.classify_list.count():
            self.classify_list.setCurrentRow(new_idx)
        self._refresh_classify_preview()
    
    def _on_classify_import_from(self):
        """Copy patterns from another root into the currently-edited one."""
        target_root = self.classify_root_combo.currentData()
        if not target_root:
            return
        sources = []
        for entry in self._added_entries:
            r = str(Path(entry["path"]).resolve())
            if r != target_root:
                sources.append(r)
        if not sources:
            QtWidgets.QMessageBox.information(
                self, "No other roots",
                "Add at least one other root folder first.")
            return
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Import patterns from…",
            "Copy patterns from which root? (replaces current list)",
            sources, 0, False)
        if not ok or not choice:
            return
        imported = load_classify_patterns(Path(choice))
        self._classify_patterns_editing = [dict(p) for p in imported]
        self._persist_classify_patterns()
        self._refresh_classify_list()
        self._refresh_classify_preview()
    
    # -- classify tab: helpers --------------------------------------------
    
    def _persist_classify_patterns(self):
        root = self.classify_root_combo.currentData()
        if not root:
            return
        save_classify_patterns(Path(root), self._classify_patterns_editing)
        self._classify_cache.clear()
    
    def _refresh_classify_effective_regex(self):
        if not hasattr(self, "classify_effective_label"):
            return
        mode = "raw" if self.classify_mode_raw.isChecked() else "shorthand"
        dev = _effective_regex(self.classify_device_edit.text().strip(), mode)
        sub = _effective_regex(self.classify_subdevice_edit.text().strip(), mode)
        if sub and self.classify_anchor_check.isChecked():
            sub = r"(?<=[-_])" + sub
        self.classify_effective_label.setText(
            f"Effective (device):    {dev or '—'}\n"
            f"Effective (subdevice): {sub or '—'}")
    
    def _refresh_classify_preview(self):
        if not hasattr(self, "classify_preview"):
            return
        root = self.classify_root_combo.currentData()
        if not root:
            self.classify_preview.setText("No roots added yet.")
            return
        if not self.current_file_path:
            self.classify_preview.setText(
                f"Editing root: {root}\n(no active file to preview against)")
            return
        try:
            dev, sub = classify_with_patterns(
                Path(self.current_file_path), self._classify_patterns_editing)
        except Exception as e:
            self.classify_preview.setText(f"Error: {e}")
            return
        actual_root = str(self._state_store_for(
            self.current_file_path)[0].state_dir.parent)
        same_root = actual_root == root
        note = ("" if same_root else
                f"\n(file is under {actual_root}; that root's patterns "
                f"are used by the classified tree)")
        self.classify_preview.setText(
            f"Active file: {Path(self.current_file_path).name}\n"
            f"Using these patterns → device={dev!r}  subdevice={sub!r}"
            f"{note}")
    
    def _on_classify_rebuild(self):
        self._classify_cache.clear()
        self._rebuild_measurements_tree()
        self._refresh_classify_preview()
    # -- Tick / check handling ------------------------------------------------
    def _on_item_check_changed(self, item, column):
        if column != 0:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None:
            return
        kind = payload[0]
        if kind == "file":
            self._on_file_check_changed(item)
        elif kind == "meas":
            self._on_measurement_check_changed(item)

    def _load_file_for_tick(self, file_path: str, item_to_revert) -> bool:
        """Ensure the file is in memory; on failure, revert the checkbox and
        return False."""
        if file_path in self.loaded_files:
            return True
        if getattr(self, "EDIT_DISABLE_MULTILOAD", False):
            others = [p for p in self.loaded_files.keys() if p != file_path]
            if others:
                self.file_list.blockSignals(True)
                item_to_revert.setCheckState(0, QtCore.Qt.Unchecked)
                self.file_list.blockSignals(False)
                QtWidgets.QMessageBox.information(
                    self, "Multi-load disabled",
                    "Advanced settings forbid loading more than one file at a time."
                )
                return False
        try:
            self.loaded_files[file_path] = self.registry.load(Path(file_path))
            return True
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Load failed", f"Could not load {file_path}:\n{exc}"
            )
            self.file_list.blockSignals(True)
            item_to_revert.setCheckState(0, QtCore.Qt.Unchecked)
            self.file_list.blockSignals(False)
            return False

    def _on_file_check_changed(self, item):
        payload = item.data(0, QtCore.Qt.UserRole)
        _, file_path, _ = payload
        ticked = item.checkState(0) == QtCore.Qt.Checked

        # Cascade to children.
        self.file_list.blockSignals(True)
        for j in range(item.childCount()):
            child = item.child(j)
            cpayload = child.data(0, QtCore.Qt.UserRole)
            if cpayload is None or cpayload[0] != "meas":
                continue
            child.setCheckState(
                0, QtCore.Qt.Checked if ticked else QtCore.Qt.Unchecked
            )
        self.file_list.blockSignals(False)

        if ticked and not self._load_file_for_tick(file_path, item):
            return

        self._after_tick_changed()

    def _on_measurement_check_changed(self, item):
        payload = item.data(0, QtCore.Qt.UserRole)
        _, file_path, _, _meas_index = payload
        ticked = item.checkState(0) == QtCore.Qt.Checked

        if ticked and not self._load_file_for_tick(file_path, item):
            return

        # Update parent's aggregate state.
        parent = item.parent()
        if parent is not None:
            states = []
            for j in range(parent.childCount()):
                c = parent.child(j)
                cpayload = c.data(0, QtCore.Qt.UserRole)
                if cpayload is None or cpayload[0] != "meas":
                    continue
                states.append(c.checkState(0))
            self.file_list.blockSignals(True)
            if states and all(s == QtCore.Qt.Checked for s in states):
                parent.setCheckState(0, QtCore.Qt.Checked)
            elif states and all(s == QtCore.Qt.Unchecked for s in states):
                parent.setCheckState(0, QtCore.Qt.Unchecked)
            else:
                parent.setCheckState(0, QtCore.Qt.PartiallyChecked)
            self.file_list.blockSignals(False)

        self._after_tick_changed()

    def _after_tick_changed(self):
        self._save_workspace()
        self._refresh_curves_tab()
        self._refresh_logbook_tab()
        self._refresh_problems_tab()
        self._refresh_plot()

    def _get_ticked_measurements(self) -> set:
        ticked = set()
        for item in self._iter_all_items():
            payload = item.data(0, QtCore.Qt.UserRole)
            if payload is None or payload[0] != "meas":
                continue
            if item.checkState(0) == QtCore.Qt.Checked:
                ticked.add((payload[1], payload[3]))
        return ticked

    def _get_ticked_file_paths(self) -> set:
        ticked = set()
        for item in self._iter_all_items():
            payload = item.data(0, QtCore.Qt.UserRole)
            if payload is None or payload[0] != "file":
                continue
            if item.checkState(0) == QtCore.Qt.Checked:
                ticked.add(payload[1])
        return ticked

    def _on_curves_mode_changed(self, btn):
        template_mode = self.curves_mode_template.isChecked()
        self.curves_template_box.setVisible(template_mode)
        self.curves_mode_hint.setText(
            "Template mode: the plot shows exactly the rows below. "
            "Curve ticks affect only the fit-target list."
            if template_mode else
            "Automatic mode: every non-hidden curve is drawn, "
            "one axis per Y unit.")

        if self.loaded_file is None or self.current_meas_index is None:
            return

        if template_mode:
            # First switch → seed from currently visible curves
            if self.current_state.plot_template is None:
                meas = self.loaded_file.measurements[self.current_meas_index]
                self.current_state.plot_template = \
                    self._default_specs_from_curves(meas)
                self.current_state.plot_template_auto_applied = True
                self._save_state()
        else:
            if self.current_state.plot_template is not None:
                self.current_state.plot_template = None
                self.current_state.plot_template_auto_applied = True
                self._save_state()

        self._refresh_template_editor()
        self._refresh_plot()

    def _default_specs_from_curves(self, meas) -> list[dict]:
        """Turn the current visible-curves view into equivalent specs."""
        specs = []
        for curve in meas.curves:
            if curve.name in self.current_state.hidden_curves:
                continue
            parts = (curve.source_key or "").split("|")
            x_col = parts[0] if len(parts) > 0 and parts[0] in meas.columns else None
            y_col = parts[1] if len(parts) > 1 and parts[1] in meas.columns else None
            if not x_col or not y_col:
                continue
            specs.append({
                "x_column": x_col,
                "y_column": y_col,
                "segments": [],
                "label": "",
            })
        return specs    
    
    def _refresh_template_editor(self):
        if not hasattr(self, "curves_spec_table"):
            return
    
        active = (self.loaded_file is not None
                  and self.current_meas_index is not None)
        if not active:
            self.curves_spec_table.setRowCount(0)
            self.curves_template_sig_label.setText("Signature: —")
            return
    
        meas = self.loaded_file.measurements[self.current_meas_index]
        sig = self._measurement_signature(meas)
        self.curves_template_sig_label.setText(
            f"Signature: {sig or '— (no raw columns; template disabled)'}")
    
        specs = self.current_state.plot_template or []
        self.curves_spec_table.blockSignals(True)
        self.curves_spec_table.setRowCount(0)
        for spec in specs:
            self._append_spec_row(spec, meas)
        self.curves_spec_table.blockSignals(False)
    
    def _append_spec_row(self, spec: dict, meas):
        row = self.curves_spec_table.rowCount()
        self.curves_spec_table.insertRow(row)
        columns = list(meas.columns.keys())
    
        def _mk_combo(field, value, entries=None):
            cb = QtWidgets.QComboBox()
            if entries is None:
                for c in columns:
                    unit = meas.column_units.get(c, "")
                    cb.addItem(f"{c} [{unit}]" if unit else c, userData=c)
                cb.setCurrentIndex(max(0, cb.findData(value)))
            else:
                for label, val in entries:
                    cb.addItem(label, userData=val)
                cb.setCurrentIndex(max(0, cb.findText(value)))
            cb.currentIndexChanged.connect(
                lambda idx, r=row, f=field: self._on_spec_changed(r, f))
            return cb
    
        self.curves_spec_table.setCellWidget(
            row, 0, _mk_combo("x_column", spec.get("x_column", "")))
    
        self.curves_spec_table.setCellWidget(
            row, 1, _mk_combo("y_column", spec.get("y_column", "")))
    
        seg_entries = self._segment_combo_entries(meas)
        cur_label = self._segments_to_label(
            spec.get("segments", []) or [], meas)
        self.curves_spec_table.setCellWidget(
            row, 2, _mk_combo("segments", cur_label, seg_entries))
    
        lbl = QtWidgets.QLineEdit(spec.get("label", ""))
        lbl.setPlaceholderText("auto")
        lbl.editingFinished.connect(
            lambda r=row: self._on_spec_changed(r, "label"))
        self.curves_spec_table.setCellWidget(row, 3, lbl)
    
    def _segment_combo_entries(self, meas):
        n = len(meas.segments)
        if n <= 1:
            return [("all", [])]
        out = [("all", []), ("fwd", [0]), ("rev", [1])]
        for i in range(2, n):
            out.append((f"seg{i}", [i]))
        return out
    
    def _segments_to_label(self, segs, meas):
        segs = list(segs) if segs else []
        if not segs:                 return "all"
        if segs == [0]:              return "fwd"
        if segs == [1]:              return "rev"
        if len(segs) == 1:           return f"seg{segs[0]}"
        return "all"
    
    def _on_spec_changed(self, row, field):
        if (self.current_state.plot_template is None
                or row < 0
                or row >= len(self.current_state.plot_template)):
            return
        spec = self.current_state.plot_template[row]
        col_idx = {"x_column": 0, "y_column": 1, "segments": 2,
                   "label": 3}[field]
        w = self.curves_spec_table.cellWidget(row, col_idx)
        if field == "label":
            spec["label"] = w.text().strip()
        elif field == "segments":
            spec["segments"] = w.currentData() or []
        else:
            spec[field] = w.currentData() or ""
        self._save_state()
        self._refresh_plot()
        
    def _on_template_add_row(self):
        if self.loaded_file is None or self.current_meas_index is None:
            return
        if self.current_state.plot_template is None:
            self.current_state.plot_template = []
        meas = self.loaded_file.measurements[self.current_meas_index]
        cols = list(meas.columns.keys())
        if not cols:
            return
        x_col = "Time" if "Time" in cols else cols[0]
        y_col = next((c for c in cols if c != x_col), cols[0])
        self.current_state.plot_template.append({
            "x_column": x_col, "y_column": y_col,
            "segments": [], "label": "",
        })
        self._save_state()
        self._refresh_template_editor()
        self._refresh_plot()

    def _on_template_remove_row(self):
        if self.current_state.plot_template is None:
            return
        rows = sorted(
            {i.row() for i in
             self.curves_spec_table.selectionModel().selectedRows()},
            reverse=True)
        for r in rows:
            if 0 <= r < len(self.current_state.plot_template):
                del self.current_state.plot_template[r]
        self._save_state()
        self._refresh_template_editor()
        self._refresh_plot()

    def _on_template_move_row(self, delta):
        if self.current_state.plot_template is None:
            return
        rows = [i.row() for i in
                self.curves_spec_table.selectionModel().selectedRows()]
        if len(rows) != 1:
            return
        r = rows[0]
        nr = r + delta
        specs = self.current_state.plot_template
        if 0 <= nr < len(specs):
            specs[r], specs[nr] = specs[nr], specs[r]
            self._save_state()
            self._refresh_template_editor()
            self.curves_spec_table.selectRow(nr)
            self._refresh_plot()
    
    def _plot_template_library_path(self, file_path) -> Path:
        store, _ = self._state_store_for(file_path)
        return store.state_dir / "_plot_templates.json"

    def _load_plot_template_library(self, file_path) -> dict:
        p = self._plot_template_library_path(file_path)
        if not p.exists():
            return {}
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_plot_template_library(self, file_path, library):
        p = self._plot_template_library_path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(library, f, indent=2)

    def _measurement_signature(self, meas) -> str:
        if not meas.columns:
            return ""
        return "|".join(sorted(meas.columns.keys()))

    def _on_template_save(self):
        if self.loaded_file is None or self.current_meas_index is None:
            return
        meas = self.loaded_file.measurements[self.current_meas_index]
        sig = self._measurement_signature(meas)
        if not sig:
            QtWidgets.QMessageBox.information(
                self, "No signature",
                "This measurement has no raw columns, so a template "
                "cannot be saved for it.")
            return
        specs = self.current_state.plot_template or []
        if not specs:
            QtWidgets.QMessageBox.information(
                self, "Empty template",
                "Add at least one row before saving.")
            return
        lib = self._load_plot_template_library(self.current_file_path)
        lib[sig] = [dict(s) for s in specs]
        self._save_plot_template_library(self.current_file_path, lib)
        QtWidgets.QMessageBox.information(
            self, "Saved",
            f"Template saved for signature:\n{sig}")

    def _on_template_apply(self):
        if self.loaded_file is None or self.current_meas_index is None:
            return
        lib = self._load_plot_template_library(self.current_file_path)
        if not lib:
            QtWidgets.QMessageBox.information(
                self, "No templates",
                "No templates have been saved in this root yet.")
            return
        meas = self.loaded_file.measurements[self.current_meas_index]
        sig = self._measurement_signature(meas)
        sigs = sorted(lib.keys())
        default_idx = sigs.index(sig) if sig in sigs else 0
        choice, ok = QtWidgets.QInputDialog.getItem(
            self, "Apply template",
            "Choose a template to apply:", sigs, default_idx, False)
        if not ok or not choice:
            return
        self.current_state.plot_template = [
            dict(s) for s in lib[choice]]
        self.current_state.plot_template_auto_applied = True
        self._save_state()
        self.curves_mode_template.setChecked(True)
        self.curves_template_box.setVisible(True)
        self._refresh_template_editor()
        self._refresh_plot()

    def _on_template_reset(self):
        self.current_state.plot_template = None
        self.current_state.plot_template_auto_applied = True
        self._save_state()
        self.curves_mode_auto.setChecked(True)
        self.curves_template_box.setVisible(False)
        self.curves_mode_hint.setText(
            "Automatic mode: every non-hidden curve is drawn, "
            "one axis per Y unit.")
        self._refresh_plot()
    def _refresh_curves_tab(self):
        """Rebuild the Curves tab from active + selected + ticked measurements.
    
        Order:
          • the active measurement's curves first (★ marker on its header row)
          • then any selected/ticked measurements, in the order they appear
            in the measurement tree (works for both flat and classified views)
    
        Visibility is per-curve; toggling a curve updates `hidden_curves` for
        the active measurement only (other measurements' ticks are cosmetic).
        """
        self.curves_tree.blockSignals(True)
        self.curves_tree.clear()
    
        active = ((self.current_file_path, self.current_meas_index)
                  if self.current_file_path is not None else None)
        wanted = (set(self._get_ticked_measurements())
                  | set(self.selected_measurements))
        if active is not None:
            wanted.add(active)
    
        if not wanted:
            self.curves_tree.blockSignals(False)
            return
    
        # ── Ordered traversal of file_list (display order) ────────────
        def _collect_in_order(item, out):
            payload = item.data(0, QtCore.Qt.UserRole)
            if payload is not None and payload[0] == "meas":
                # measurement row (leaf); do not recurse
                tgt = (payload[1], payload[3])
                if tgt in wanted and tgt != active:
                    out.append(tgt)
                return
            for j in range(item.childCount()):
                _collect_in_order(item.child(j), out)
    
        ordered_others: list[tuple[str, int]] = []
        for i in range(self.file_list.topLevelItemCount()):
            _collect_in_order(self.file_list.topLevelItem(i), ordered_others)
    
        ordered: list[tuple[str, int]] = []
        if active is not None:
            ordered.append(active)
        ordered.extend(ordered_others)
    
        # ── Build the tree: one meas_root per target, curves as children ──
        for fp, mi in ordered:
            lf = self.loaded_files.get(fp)
            if lf is None or mi >= len(lf.measurements):
                continue
            meas = lf.measurements[mi]
            is_active = (fp == self.current_file_path
                         and mi == self.current_meas_index)
    
            title = ("★ " + meas.title) if is_active else meas.title
            meas_root = QtWidgets.QTreeWidgetItem([title, "", ""])
            meas_root.setFlags(
                meas_root.flags() | QtCore.Qt.ItemIsUserCheckable)
            meas_root.setCheckState(0, QtCore.Qt.Checked)
            meas_root.setData(0, QtCore.Qt.UserRole,
                              ("meas_root", fp, mi))
            self.curves_tree.addTopLevelItem(meas_root)
    
            for curve in meas.curves:
                hidden = (is_active
                          and curve.name in self.current_state.hidden_curves)
                child = QtWidgets.QTreeWidgetItem([
                    curve.name,
                    str(len(curve.x)),
                    curve.source_key or "",
                ])
                child.setFlags(
                    child.flags() | QtCore.Qt.ItemIsUserCheckable)
                child.setCheckState(
                    0, QtCore.Qt.Unchecked if hidden else QtCore.Qt.Checked)
                child.setData(0, QtCore.Qt.UserRole,
                              ("curve", fp, mi, curve.name))
                meas_root.addChild(child)
            
                # Segment sub-rows for dual-sweep data
                segs = curve.segments or []
                if len(segs) > 1:
                    hidden_segs = (self.current_state.hidden_segments
                                   .get(curve.name, [])
                                   if is_active else [])
                    for seg_idx, (start, end) in enumerate(segs):
                        if seg_idx == 0:
                            seg_label = "fwd"
                        elif seg_idx == 1:
                            seg_label = "rev"
                        else:
                            seg_label = f"seg{seg_idx}"
                        seg_child = QtWidgets.QTreeWidgetItem([
                            f"    {seg_label}",
                            str(end - start),
                            "",
                        ])
                        seg_child.setFlags(
                            seg_child.flags() | QtCore.Qt.ItemIsUserCheckable)
                        seg_child.setCheckState(
                            0,
                            (QtCore.Qt.Unchecked
                             if seg_idx in hidden_segs
                             else QtCore.Qt.Checked))
                        seg_child.setData(
                            0, QtCore.Qt.UserRole,
                            ("segment", fp, mi, curve.name, seg_idx))
                        child.addChild(seg_child)
    
            meas_root.setExpanded(True)
    
        self.curves_tree.blockSignals(False)
    
    def _on_curve_item_changed(self, item, column):
        if column != 0:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None:
            return
        kind = payload[0]
    
        # --- cascade for root nodes ---
        if kind == "meas_root":
            ticked = item.checkState(0) == QtCore.Qt.Checked
            _, fp, mi = payload
        
            # If this measurement is the active one, mirror the tick into
            # the sidecar's hidden_curves so the plot actually updates.
            if (file_path_match(fp, self.current_file_path)
                    and mi == self.current_meas_index):
                lf = self.loaded_files.get(fp)
                if lf is not None and mi < len(lf.measurements):
                    if ticked:
                        self.current_state.hidden_curves = []
                    else:
                        self.current_state.hidden_curves = [
                            c.name for c in lf.measurements[mi].curves
                        ]
                    self._save_state()
        
            self.curves_tree.blockSignals(True)
            self._cascade_check(item, ticked)
            self.curves_tree.blockSignals(False)
            self._refresh_plot()
            return
        if kind == "segment":
            _, fp, mi, curve_name, seg_idx = payload
            # Only the active measurement's segments can be toggled.
            if not (file_path_match(fp, self.current_file_path)
                    and mi == self.current_meas_index):
                return
            visible = item.checkState(0) == QtCore.Qt.Checked
            hidden = self.current_state.hidden_segments.setdefault(
                curve_name, [])
            if visible and seg_idx in hidden:
                hidden.remove(seg_idx)
            elif not visible and seg_idx not in hidden:
                hidden.append(seg_idx)
            if not hidden:
                del self.current_state.hidden_segments[curve_name]
            self._save_state()
            self._refresh_plot()
            return
        if kind != "curve":
            return
    
        _, file_path, meas_index, curve_name = payload
        # Only mutate the active measurement's hidden_curves; other
        # measurements' ticks are visual-only for now.
        if not (file_path_match(file_path, self.current_file_path)
                and meas_index == self.current_meas_index):
            return
    
        visible = item.checkState(0) == QtCore.Qt.Checked
        if visible and curve_name in self.current_state.hidden_curves:
            self.current_state.hidden_curves.remove(curve_name)
        elif not visible and curve_name not in self.current_state.hidden_curves:
            self.current_state.hidden_curves.append(curve_name)
        self._save_state()
        self._refresh_plot()
    
    def _cascade_check(self, item, ticked: bool):
        """Recursively set check state on item and all children."""
        item.setCheckState(0, QtCore.Qt.Checked if ticked else QtCore.Qt.Unchecked)
        for j in range(item.childCount()):
            self._cascade_check(item.child(j), ticked)
    

    
    def _set_all_curves_visible(self, visible: bool):
        """Tick/untick every curve row. Only affects the currently-active measurement's state."""
        self.curves_tree.blockSignals(True)
        for curve_item in self._iter_curve_items():
            curve_item.setCheckState(0, QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked)
        self.curves_tree.blockSignals(False)

        if visible:
            self.current_state.hidden_curves = []
        else:
            hidden = []
            for curve_item in self._iter_curve_items():
                payload = curve_item.data(0, QtCore.Qt.UserRole)
                if payload is None or payload[0] != "curve":
                    continue
                _, file_path, meas_index, curve_name = payload
                if file_path == self.current_file_path and meas_index == self.current_meas_index:
                    hidden.append(curve_name)
            self.current_state.hidden_curves = hidden

        self._save_state()
        self._refresh_plot()

    def _iter_curve_items(self):
        """Yield every curve-level QTreeWidgetItem in the Curves tab (DFS)."""
        stack = [self.curves_tree.topLevelItem(i) for i in range(self.curves_tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            payload = item.data(0, QtCore.Qt.UserRole)
            if payload is not None and payload[0] == "curve":
                yield item
            else:
                stack.extend(item.child(j) for j in range(item.childCount()))  
        
    def _resolve_curve_key(self, curve_name: str, idx: int) -> str:
        """Map '__first__' (legacy) to the actual first curve's name."""
        if "__first__" in self.current_state.bg_ranges_by_curve:
            if idx == 0:
                # Migrate on the fly
                self.current_state.bg_ranges_by_curve[curve_name] = \
                    self.current_state.bg_ranges_by_curve.pop("__first__")
                self.current_state.anchor_points_by_curve[curve_name] = \
                    self.current_state.anchor_points_by_curve.pop("__first__", [])
                self._save_state()
            return curve_name
        return curve_name 
    
    def _save_state(self):
        if self.current_file_path and self.current_filename and self.current_meas_index is not None:
            store, key_prefix = self._state_store_for(self.current_file_path)
            key = state_key(key_prefix, self.current_meas_index)
            store.save(key, self.current_state)
            
    def _state_store_for(self, file_path) -> tuple["StateStore", str]:
        fp = Path(file_path).resolve()
        leaf_dir = str(fp.parent)
        state_root_str = self._state_roots.get(leaf_dir)
        if state_root_str is None:
            state_root = fp.parent
            key_prefix = fp.name
        else:
            state_root = Path(state_root_str)
            try:
                key_prefix = str(fp.relative_to(state_root))
            except ValueError:
                key_prefix = fp.name
        return StateStore(state_root), key_prefix
    def on_get_bounds_clicked(self):
        # Assuming you have access to plotted_x and plotted_y lists of arrays
        if not self.plotted_x or not self.plotted_y:
            return  
    
        all_x = np.concatenate(self.plotted_x)
        all_y = np.concatenate(self.plotted_y)
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        x_range = x_max - x_min
        y_range = y_max - y_min
    
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1
    
        bounds = {
        'EDIT_XMIN': x_min - x_pad,
        'EDIT_XMAX': x_max + x_pad,
        'EDIT_YMIN': y_min - y_pad,
        'EDIT_YMAX': y_max + y_pad,
        }
    
        # This updates both the line edits AND the parent attributes
        self.plot_lim_row.set_values(bounds)
        
    # -- Plotting -------------------------------------------------------------
    def _render_figure_to_pixmap(self):
        """Render the current figure to a QPixmap at the current figure size and DPI."""
        buf = io.BytesIO()
        # Save the figure as PNG using the current figure's DPI and size
        
        self.fig.savefig(buf, format='png', dpi=self.fig.get_dpi())
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        buf.close()
        return pixmap

    def _cycle_stretch_mode(self):
        """Cycle through the 3 stretch modes with proper cleanup"""
        self.EDIT_TOGGLE_STRETCH = (self.EDIT_TOGGLE_STRETCH + 1) % 3
        self._update_button_text()
    
        # Clean up span selector if entering image‑preview mode
        if self.EDIT_TOGGLE_STRETCH == 2:
            if self.span_selector is not None:
                self.span_selector.disconnect_events()
                self.span_selector = None
                self.add_range_btn.setChecked(False)
    
        # Always clear the image label (previous pixmap)
        self.image_label.clear()
    
        # Update canvas size policy and refresh the plot
        self._update_canvas_size_policy()
        self._refresh_plot()   
    
        self._save_session_parameters()
    
    def _update_button_text(self):
        """Update button text based on current mode"""
        mode_texts = {
            0: "Stretch to Fit",
            1: "Scale to Fit", 
            2: "Real Size"
        }
        self.ui_toggle_stretchplot.setText(mode_texts[self.EDIT_TOGGLE_STRETCH])
        # We change button color to indicate active state, return and change to more pleasant colours later.
        if self.EDIT_TOGGLE_STRETCH == 0:
            self.ui_toggle_stretchplot.setStyleSheet("background-color: #4CAF50; color: white;")  # Green
        elif self.EDIT_TOGGLE_STRETCH == 1:
            self.ui_toggle_stretchplot.setStyleSheet("background-color: #FF9800; color: white;")  # Orange
        else:
            self.ui_toggle_stretchplot.setStyleSheet("background-color: #f44336; color: white;")  # Red
            
    def _toggle_comment(self, checked):
        self.EDIT_SHOW_COMMENT = checked
        self._refresh_plot()
        #Save comment state to preserve it between sessions
        self._save_session_parameters()  
    
    def _toggle_spans(self,checked):
        self.EDIT_HIDE_SPANS = checked
        self._refresh_plot()
        #Save spans  state to preserve it between sessions
        self._save_session_parameters()  
    
    def _toggle_multiplot(self, checked):
        self.EDIT_MULTIPLOT = bool(checked)
        self._refresh_plot()
        
    def closeEvent(self, event):
        """Save session when closing the window."""
        self._flush_pending_log_commit()
        self._save_session_parameters()
        event.accept()  
        
    def _apply_figsize(self, comment: bool):
        # Skip if in Real Size mode so that the preview for the saved figure can be seen
        if self.EDIT_TOGGLE_STRETCH == 2:
            return
        
        w, h = self.EDIT_FIGSIZE_X, self.EDIT_FIGSIZE_Y
        if comment:
            w = w * self.EDIT_COMMENT_WIDTH_FACTOR
        self.fig.set_size_inches(w, h)
        self.fig.set_dpi(self.EDIT_FIGURE_DPI)
    
    def _apply_scale_to_fit_canvas_size(self):
        container = self.canvas_container
        container_size = container.size()
        if container_size.width() <= 0 or container_size.height() <= 0:
            return
    
        fig_w = self.EDIT_FIGSIZE_X
        fig_h = self.EDIT_FIGSIZE_Y
        if self.EDIT_SHOW_COMMENT:
            fig_w *= self.EDIT_COMMENT_WIDTH_FACTOR
            
        aspect = fig_w / fig_h
    
        avail_w = container_size.width() - 4
        avail_h = container_size.height() - 4
    
        if avail_w / avail_h > aspect:
            new_w = int(avail_h * aspect)
            new_h = avail_h
        else:
            new_w = avail_w
            new_h = int(avail_w / aspect)
    
        new_w = max(new_w, 1)
        new_h = max(new_h, 1)
    
        current = self.canvas.size()
        if current.width() != new_w or current.height() != new_h:
            self.canvas.resize(new_w, new_h) 
            
    def _on_bg_mode_changed(self, index):
        """Handle background removal mode change"""
        self.current_state.bg_removal_mode = index
        self._save_state()
        self._refresh_plot()
        
        # Show/hide relevant controls
        self.poly_degree_spin.setVisible(index == BgRemovalMode.POLY)
        self.spline_smoothing_spin.setVisible(index == BgRemovalMode.SPLINE)
    
    def _on_poly_degree_changed(self, value):
        """Handle polynomial degree change"""
        self.current_state.poly_degree = value
        self._save_state()
        self._refresh_plot()
    
    def _on_spline_smoothing_changed(self, value):
        """Handle spline smoothing change"""
        self.current_state.spline_smoothing = value
        self._save_state()
        self._refresh_plot()
        
    def _update_canvas_size_policy(self):
        if self.EDIT_TOGGLE_STRETCH == 0:
            self.canvas.setMinimumSize(0, 0)
            self.canvas.setMaximumSize(16777215, 16777215)
            self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        elif self.EDIT_TOGGLE_STRETCH == 1:
            self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            
        # mode 2 uses the label, canvas not shown
       
    

    
    def _save_window_state(self):
        """Save window position and size"""
        geometry = self.geometry()
        self.EDIT_WINDOW_X = geometry.x()
        self.EDIT_WINDOW_Y = geometry.y()
        self.EDIT_WINDOW_WIDTH = geometry.width()
        self.EDIT_WINDOW_HEIGHT = geometry.height()
        
    def _restore_window_state(self):
        """Restore window position and size from saved values"""
        if hasattr(self, 'EDIT_WINDOW_X') and hasattr(self, 'EDIT_WINDOW_Y'):
            self.move(self.EDIT_WINDOW_X, self.EDIT_WINDOW_Y)
        if hasattr(self, 'EDIT_WINDOW_WIDTH') and hasattr(self, 'EDIT_WINDOW_HEIGHT'):
            self.resize(self.EDIT_WINDOW_WIDTH, self.EDIT_WINDOW_HEIGHT)
            
    def _update_image_preview(self):
        """Scale the cached native pixmap to fit the label, maintaining aspect ratio."""
        if not hasattr(self, '_native_pixmap') or self._native_pixmap is None:
            return
        label_size = self.image_label.size()
        if label_size.width() <= 0 or label_size.height() <= 0:
            return
        # Scale to fit inside label, keeping aspect ratio, using smooth transformation
        scaled = self._native_pixmap.scaled(
            label_size.width() - 2,   # small margin
            label_size.height() - 2,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)        
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display_scaling()
    
    def _apply_span_ranges(self, x, y, span_ranges, anchor_points=None):
        """Build baseline (y_subtract) from SPANs and anchor points."""
        if not span_ranges and not anchor_points:
            return np.zeros_like(y)
    
        ranges = []                 # list of [start, end] for each SPAN and anchor
        anchor_values = {}          # index -> clicked y value for anchors
    
        # Process SPANs
        for xmin, xmax in span_ranges:
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            idx = np.where((x >= xmin) & (x <= xmax))[0]
            if len(idx) > 0:
                start, end = idx[0], idx[-1]
                ranges.append([start, end])
    
        # Process anchors
        if anchor_points:
            for xp, yp in anchor_points:
                idx = np.argmin(np.abs(x - xp))
                ranges.append([idx, idx])
                anchor_values[idx] = yp
    
        if not ranges:
            return np.zeros_like(y)
    
        # Sort ranges by start index
        ranges.sort(key=lambda r: r[0])
    
        y_subtract = np.zeros_like(y)
    
        # 1. Set values for SPANs and anchors
        for start, end in ranges:
            if start == end:
                # anchor – use the clicked y value
                if start in anchor_values:
                    y_subtract[start] = anchor_values[start]
                else:
                    y_subtract[start] = y[start]   # fallback
            else:
                # SPAN – original y values
                y_subtract[start:end+1] = y[start:end+1]
    
        # 2. Fill gaps between consecutive ranges
        for i in range(len(ranges) - 1):
            cur_end = ranges[i][1]
            next_start = ranges[i+1][0]
            if cur_end < next_start:
                num = next_start - cur_end + 1
                y_subtract[cur_end:next_start+1] = np.linspace(
                    y_subtract[cur_end], y_subtract[next_start], num
                )
    
        return y_subtract
    
    
    def _apply_linear_fit(self, x, y, span_ranges, anchor_points=None):
        x_fit, y_fit = self._collect_fit_points(x, y, span_ranges, anchor_points or [])
        if len(x_fit) < 2:
            return y
        coeffs = np.polyfit(x_fit, y_fit, 1)
        return y - np.polyval(coeffs, x)
    
    
    def _apply_poly_fit(self, x, y, span_ranges, anchor_points=None):
        x_fit, y_fit = self._collect_fit_points(x, y, span_ranges, anchor_points or [])
        degree = self.current_state.poly_degree
        if len(x_fit) < degree + 1:
            return self._apply_linear_fit(x, y, span_ranges, anchor_points)
        coeffs = np.polyfit(x_fit, y_fit, degree)
        return y - np.polyval(coeffs, x)
    
    def _apply_exponential_decay_fit(self, x, y, span_ranges, anchor_points=None):
        """Exponential decay fit through SPANs + anchors."""
        x_fit, y_fit = self._collect_fit_points(x, y, span_ranges, anchor_points or [])
        if len(x_fit) < 3:
            return y  # not enough points
    
        sort = np.argsort(x_fit)
        x_fit = np.array(x_fit)[sort]
        y_fit = np.array(y_fit)[sort]
    
        try:
          
            def decay(x, A, tau, C, m):
                return A * np.exp(-x / tau) + C + m * x
            C_est = np.min(y_fit)
            A_est = np.max(y_fit) - C_est
            tau_est = (x_fit[-1] - x_fit[0]) / 3
            m_est = (y_fit[-1] - y_fit[0]) / (x_fit[-1] - x_fit[0]) if x_fit[-1] > x_fit[0] else 0
            p0 = [A_est, tau_est, C_est, m_est]
            popt, _ = curve_fit(decay, x_fit, y_fit, p0=p0, maxfev=5000)
            baseline = decay(x, *popt)
            return y - baseline
        except:
            return self._apply_manual_gradient(x, y, span_ranges, anchor_points)
    
    
    def _apply_manual_gradient(self, x, y, span_ranges, anchor_points=None):
        """Polynomial fit through median points of SPANs + anchors."""
        x_fit, y_fit = self._collect_fit_points(x, y, span_ranges, anchor_points or [])
        if len(x_fit) < 2:
            return y
    
        # Use median of each SPAN? Already using all points; this is fine.
        degree = min(3, len(x_fit) - 1)
        coeffs = np.polyfit(x_fit, y_fit, degree)
        baseline = np.polyval(coeffs, x)
        return y - baseline
    
    
    
    
    def _collect_fit_points(self, x, y, span_ranges, anchor_points):
        """Collect (x, y) points from SPANs and anchor points for fitting."""
        x_fit, y_fit = [], []
        for xmin, xmax in span_ranges:
            mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
            x_fit.extend(x[mask])
            y_fit.extend(y[mask])
        for xp, yp in anchor_points:
            x_fit.append(xp)
            y_fit.append(yp)
        return x_fit, y_fit
    """
    def _run_curve_fit(self):
        key = self.current_state.active_fit_key
        if key is None:
            return
        curve_name, seg_idxs = _parse_fit_key(key)
        curve = <curve in active measurement with this name>
        x, y = curve.x, curve.y
        if self.current_state.fit_by_curve.get(key, {}).get("use_bg_removed"):
            baseline = <compute baseline from bg_fit_by_curve, if any>
            if baseline is not None:
                y = y - baseline
        if seg_idxs:
            # restrict x/y to selected segments
            ...
        coeffs, fit_x, fit_y = run_fit(mode, x, y)
        self.current_state.fit_by_curve[key] = {
            "mode": mode,
            "coefficients": list(coeffs),
            "use_bg_removed": bool(...),
            "x_range": [float(fit_x[0]), float(fit_x[-1])],
            "segments": list(seg_idxs),
            "extra": {},
            "display_x": list(map(float, fit_x)),
            "display_y": list(map(float, fit_y)),
        }
        self._save_state()
        self._refresh_plot()
    """
    def _plot_from_template(self, primary_ax, meas, specs, is_active):
        """Draw one measurement's custom plot spec on primary_ax.
    
        Called only when st.plot_template is not None. Fits and BG
        removal are NOT applied here — template mode is 'raw as configured'.
        """
        axes_by_yunit = {"": primary_ax}
        styles = ("-", "--", ":", "-.")
    
        def _get_axes_for(unit):
            if unit in axes_by_yunit:
                return axes_by_yunit[unit]
            new_ax = primary_ax.twinx()
            new_ax.yaxis.set_label_position("right")
            axes_by_yunit[unit] = new_ax
            return new_ax
    
        for spec in specs:
            x_col = spec.get("x_column")
            y_col = spec.get("y_column")
            if not x_col or not y_col:
                continue
            x = meas.columns.get(x_col)
            y = meas.columns.get(y_col)
            if x is None or y is None:
                continue
    
            y_unit = meas.column_units.get(y_col, "")
            ax = _get_axes_for(y_unit)
            segs = list(spec.get("segments") or [])
            user_label = spec.get("label") or None
    
            # Base colour index for this spec. All segments in one spec share
            # the pair, so fwd=2i, rev=2i+1.
            c_base = self._color_idx * 2
    
            if segs:
                seg_handles = {}
                for seg_idx in segs:
                    if seg_idx >= len(meas.segments):
                        continue
                    s0, s1 = meas.segments[seg_idx]
                    color = dmp.get_color_at(self._color_palette, c_base + seg_idx)
                    line, = ax.plot(
                        x[s0:s1], y[s0:s1],
                        linestyle=styles[seg_idx % len(styles)],
                        color=color)
                    seg_handles[seg_idx] = line
                    self.plotted_x.append(x[s0:s1])
                    self.plotted_y.append(y[s0:s1])
    
                # Consolidated fwd/rev handling (same shape as auto-mode)
                has_fwd = 0 in seg_handles
                has_rev = 1 in seg_handles
                if has_fwd and has_rev:
                    lbl = user_label or self._build_curve_label(
                        meas, y_unit=y_unit, seg_tag="fwd/rev")
                    self._legend_entries.append(
                        (lbl, (seg_handles[0], seg_handles[1]), 'tuple'))
                elif has_fwd:
                    lbl = user_label or self._build_curve_label(
                        meas, y_unit=y_unit, seg_tag="fwd")
                    self._legend_entries.append((lbl, seg_handles[0], 'single'))
                elif has_rev:
                    lbl = user_label or self._build_curve_label(
                        meas, y_unit=y_unit, seg_tag="rev")
                    self._legend_entries.append((lbl, seg_handles[1], 'single'))
    
                # Extra segments (>=2): individual entries, never consolidated
                for seg_idx in sorted(seg_handles.keys()):
                    if seg_idx >= 2:
                        lbl = user_label or self._build_curve_label(
                            meas, y_unit=y_unit, seg_tag=f"seg{seg_idx}")
                        self._legend_entries.append(
                            (lbl, seg_handles[seg_idx], 'single'))
            else:
                color = dmp.get_color_at(self._color_palette, c_base)
                line, = ax.plot(x, y, color=color)
                lbl = user_label or self._build_curve_label(
                    meas, y_unit=y_unit)
                self._legend_entries.append((lbl, line, 'single'))
                self.plotted_x.append(x)
                self.plotted_y.append(y)
    
            self._color_idx += 1
                
    def _refresh_plot(self):
        if self.span_selector is not None:
            try:
                self.span_selector.disconnect_events()
            except Exception:
                pass
            self.span_selector = None
        self._refresh_legend_preview() 
        self.fig.clear()
        self.image_label.clear()
    
        if self.loaded_file is None or self.current_meas_index is None:
            self.canvas.draw()
            return
    
        # --- target list: ticked ∪ Qt-selected ---
        if self.EDIT_MULTIPLOT:
            ticked = self._get_ticked_measurements()
            highlighted = set(self.selected_measurements)
            combined = ticked | highlighted
            targets = list(combined) if combined else [
                (self.current_file_path, self.current_meas_index)
            ]
        else:
            targets = [(self.current_file_path, self.current_meas_index)]
        
        # Lazy-load any file we need but haven't read into memory yet.
        for fp, _ in targets:
            if fp not in self.loaded_files:
                try:
                    self.loaded_files[fp] = self.registry.load(Path(fp))
                except Exception as exc:
                    logger.error(f"Could not auto-load {fp}: {exc}")
        targets = [t for t in targets if t[0] in self.loaded_files]
    
        # active measurement's Measurement (for title etc)
        active_lf = self.loaded_file
        dat = active_lf.measurements[self.current_meas_index]
    
        fig_width = self.EDIT_FIGSIZE_X
        fig_height = self.EDIT_FIGSIZE_Y
        if self.EDIT_SHOW_COMMENT:
            fig_width *= self.EDIT_COMMENT_WIDTH_FACTOR
        self.fig.set_size_inches(fig_width, fig_height)
        self.fig.set_dpi(self.EDIT_FIGURE_DPI)
    
        if self.EDIT_SHOW_COMMENT:
            ax = self._build_annotated_axes()
        else:
            ax = self.fig.add_subplot(111)
            self.fig.subplots_adjust(
                left=self.EDIT_BOUNDS_LEFT, right=self.EDIT_BOUNDS_RIGHT,
                top=self.EDIT_BOUNDS_TOP, bottom=self.EDIT_BOUNDS_BOTTOM,
            )
        primary_ax = ax
        self.plotted_x = []
        self.plotted_y = []

        self._color_palette = dmp.get_tab20bc(output="list", grouping="maxpairs")
        self._color_idx = 0
        self._legend_entries = []   # list[tuple[str, Line2D | tuple, 'single'|'tuple']]
        # Selected range (applies only to active measurement's list)
        selected_typ = None
        selected_idx = None
        if self._selected_row is not None:
            item = self.range_list.item(self._selected_row)
            if item is not None:
                data = item.data(QtCore.Qt.UserRole)
                if data is not None:
                    selected_typ, selected_idx = data
    
        any_baseline = False
        any_fit = False

        for file_path, meas_idx in targets:
            lf = self.loaded_files[file_path]
            if meas_idx >= len(lf.measurements):
                continue
            meas = lf.measurements[meas_idx]
            is_active = (file_path == self.current_file_path
                         and meas_idx == self.current_meas_index)

            # ── Resolve sidecar state ─────────────────────────────────
            if is_active:
                st = self.current_state
            else:
                st_store, st_prefix = self._state_store_for(file_path)
                st = st_store.load(state_key(st_prefix, meas_idx))

            # ── Branch: template-mode or auto-mode ────────────────────
            if getattr(st, "plot_template", None) is not None:
                # Template path: only the user-specified lines are drawn.
                # No per-curve loop; no hidden_curves logic; no fitting
                # overlay (yet — see note at bottom).
                self._plot_from_template(
                    primary_ax, meas, st.plot_template, is_active)
                continue                # skip the auto path for this target

            # ── Auto path: unchanged per-curve loop ───────────────────
            hidden = st.hidden_curves
            bg_by_curve = st.bg_ranges_by_curve
            anchors_by_curve = st.anchor_points_by_curve
            remove_bg = st.remove_background
            bg_mode = st.bg_removal_mode

            prefix = "" if is_active else f"[m{meas_idx}] "
            overrides = st.curve_overrides
    
            for idx, curve in enumerate(meas.curves):
                if curve.name in hidden:
                    continue
                x, y = curve.x, curve.y
    
                # Curve display-name override (feature 5)
                ov = overrides.get(curve.name, {})
                disp = ov.get("name", curve.name)
    
                key = self._resolve_curve_key(curve.name, idx)
                raw_ranges = bg_by_curve.get(key, [])
                span_ranges = []
                for rd in raw_ranges:
                    if len(rd) >= 3:
                        if rd[2] == RangeType.SPAN:
                            span_ranges.append([rd[0], rd[1]])
                    else:
                        span_ranges.append([rd[0], rd[1]])
                anchor_points = anchors_by_curve.get(key, [])
    
                # ---- Baseline / subtraction (same logic as before, using local vars) ----
                y_plot = y.copy()
                baseline = None
                if (remove_bg or span_ranges or anchor_points):
                    if bg_mode == BgRemovalMode.CONNECT_END:
                        baseline = self._apply_span_ranges(x, y, span_ranges, anchor_points)
                    elif bg_mode == BgRemovalMode.LINEAR:
                        xf, yf = self._collect_fit_points(x, y, span_ranges, anchor_points)
                        if len(xf) >= 2:
                            coeffs = np.polyfit(xf, yf, 1)
                            baseline = np.polyval(coeffs, x)
                    elif bg_mode == BgRemovalMode.POLY:
                        xf, yf = self._collect_fit_points(x, y, span_ranges, anchor_points)
                        degree = self.current_state.poly_degree
                        if len(xf) >= degree + 1:
                            coeffs = np.polyfit(xf, yf, degree)
                            baseline = np.polyval(coeffs, x)
                    elif bg_mode == BgRemovalMode.SPLINE:
                        xf, yf = self._collect_fit_points(x, y, span_ranges, anchor_points)
                        if len(xf) >= 4:
                            try:
                                si = np.argsort(xf)
                                xf, yf = np.array(xf)[si], np.array(yf)[si]
                                sm = self.current_state.spline_smoothing or None
                                tck = splrep(xf, yf, s=sm)
                                baseline = splev(x, tck)
                            except ImportError:
                                pass
                    elif bg_mode == BgRemovalMode.EXPONENTIAL_DECAY:
                        baseline = self._apply_exponential_decay_fit(x, y, span_ranges, anchor_points)
                    elif bg_mode == BgRemovalMode.MANUAL_GRADIENT:
                        baseline = self._apply_manual_gradient(x, y, span_ranges, anchor_points)
    
                    if remove_bg and baseline is not None:
                        y_plot = y - baseline
    
                ov = overrides.get(curve.name, {})
                disp_override = ov.get("name") if ov.get("name") != curve.name else None
                
                if getattr(curve, "segments", None) and len(curve.segments) >= 2:
                    hidden_segs = (
                        self.current_state.hidden_segments.get(curve.name, [])
                        if is_active else [])
                
                    c_base = self._color_idx * 2
                    _SEG_STYLES = ("-", "--", ":", "-.")
                
                    seg_handles = {}
                    for seg_idx, (start, end) in enumerate(curve.segments):
                        if seg_idx in hidden_segs:
                            continue
                        color = dmp.get_color_at(self._color_palette, c_base + seg_idx)
                        line, = ax.plot(
                            x[start:end], y_plot[start:end],
                            linestyle=_SEG_STYLES[seg_idx % len(_SEG_STYLES)],
                            color=color)
                        seg_handles[seg_idx] = line
                
                    has_fwd = 0 in seg_handles
                    has_rev = 1 in seg_handles
                    if has_fwd and has_rev:
                        lbl = self._build_curve_label(
                            meas, y_unit=curve.y_unit,
                            seg_tag="fwd/rev", override=disp_override)
                        self._legend_entries.append(
                            (lbl, (seg_handles[0], seg_handles[1]), 'tuple'))
                    elif has_fwd:
                        lbl = self._build_curve_label(
                            meas, y_unit=curve.y_unit,
                            seg_tag="fwd", override=disp_override)
                        self._legend_entries.append((lbl, seg_handles[0], 'single'))
                    elif has_rev:
                        lbl = self._build_curve_label(
                            meas, y_unit=curve.y_unit,
                            seg_tag="rev", override=disp_override)
                        self._legend_entries.append((lbl, seg_handles[1], 'single'))
                
                    for seg_idx in sorted(seg_handles.keys()):
                        if seg_idx >= 2:
                            lbl = self._build_curve_label(
                                meas, y_unit=curve.y_unit,
                                seg_tag=f"seg{seg_idx}", override=disp_override)
                            self._legend_entries.append(
                                (lbl, seg_handles[seg_idx], 'single'))
                
                    self._color_idx += 1
                else:
                    label = self._build_curve_label(
                        meas, y_unit=curve.y_unit, override=disp_override)
                    color = dmp.get_color_at(self._color_palette, self._color_idx * 2)
                    line, = ax.plot(x, y_plot, color=color)
                    self._legend_entries.append((label, line, 'single'))
                    self._color_idx += 1
                
                    # ── Show stored fit for the ACTIVE fit target only ──
                    active_fit_key = self.current_state.active_fit_key
                    if (is_active and active_fit_key and not self.EDIT_HIDE_SPANS):
                        fit_curve_name, _fit_segs = _parse_fit_key(active_fit_key)
                        if fit_curve_name == curve.name:
                            fit = self.current_state.fit_by_curve.get(active_fit_key) or {}
                            fx = fit.get("display_x")
                            fy = fit.get("display_y")
                            if fx is not None and fy is not None:
                                fit_line, = ax.plot(
                                    fx, fy, linestyle="--", color="green", linewidth=1.5)
                                self._legend_entries.append(("Fit", fit_line, 'single'))
                
                # Baseline visualisation
                if (baseline is not None and not remove_bg
                        and not self.EDIT_HIDE_SPANS and is_active):
                    bl_line, = ax.plot(
                        x, baseline, '--', color='red', alpha=0.7, linewidth=1.5)
                    self._legend_entries.append(("Baseline", bl_line, 'single'))
                
                # SPAN / anchor overlays (unchanged except anchors now push a handle)
                if not self.EDIT_HIDE_SPANS and is_active:
                    for xmin, xmax in span_ranges:
                        ax.axvspan(xmin, xmax, color="orange", alpha=0.25)
                    for j, (xp, yp) in enumerate(anchor_points):
                        sc = ax.scatter(xp, yp, color='magenta', s=40, zorder=5)
                        if j == 0:
                            self._legend_entries.append(("Anchors", sc, 'single'))
                
                self.plotted_x.append(x)
                self.plotted_y.append(y_plot)
    
        # ---- Title / labels / cosmetics ----
        base_title = getattr(dat, "title") or self.current_filename or ""
        if self.current_state.remove_background:
            base_title += " (rm BG)"
        if self.EDIT_SHOW_TITLE:
            final_title = self.EDIT_TITLE_OVERRIDE if self.EDIT_TITLE_OVERRIDE is not None else base_title
        else:
            final_title = ""
        ax.set_title(final_title, fontweight="bold")
        ax.set_xlabel(self.EDIT_XLABEL_OVERRIDE if self.EDIT_XLABEL_OVERRIDE is not None else "Time [s]")
        ax.set_ylabel(self.EDIT_YLABEL_OVERRIDE if self.EDIT_YLABEL_OVERRIDE is not None else "Current [A]")
    
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.xaxis.get_major_formatter().set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.yaxis.get_major_formatter().set_powerlimits((-3, 3))
        ax.tick_params(axis='both', labelsize=self.EDIT_TICK_SIZE)
        if self._legend_entries:
            handles, labels_, seen = [], [], set()
            handler_map = {}
            for lbl, hnd, kind in self._legend_entries:
                if lbl in seen:
                    continue
                seen.add(lbl)
                labels_.append(lbl)
                if kind == 'tuple':
                    handles.append(tuple(hnd))
                    handler_map[tuple] = HandlerTuple(ndivide=None)
                else:
                    handles.append(hnd)
            primary_ax.legend(handles, labels_, loc="best",
                              handler_map=handler_map)
    
        if self.EDIT_SHOW_COMMENT:
            self._draw_comment_panel(dat, ax)
            ax.annotate(
                f"fileID: {self.current_file_index}  measurementID: {self.current_meas_index}",
                xy=(0.1, 0.9), xycoords="axes fraction",
            )
    
        if self.span_btn.isChecked() and self.EDIT_TOGGLE_STRETCH != 2:
            self._enable_span_selector(ax, RangeType.SPAN)
    
        x_limits, y_limits = self.get_plot_limits()
        ax.set_aspect('auto')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_frame_on(True)
        if y_limits[0] is not None and y_limits[0] < 0:
            ax.axhline(y=0, color='grey', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
    
        # ---- Render mode handling (unchanged) ----
        if self.EDIT_TOGGLE_STRETCH == 2:
            native_pixmap = self._render_figure_to_pixmap()
            self._native_pixmap = native_pixmap
            self._update_image_preview()
            self.canvas.hide()
            self.image_label.show()
        else:
            self.canvas.show()
            self.image_label.hide()
            self.canvas.draw()
    
        QtCore.QTimer.singleShot(0, self._update_display_scaling)
        size = self.canvas_container.size()
        self.canvas_container.resize(size.width() + 1, size.height() + 1)
        self.canvas_container.resize(size.width(), size.height())
        
    def get_plot_limits(self):
        # Start with all None
        x_min = x_max = y_min = y_max = None
    
        # If data exists, compute padded bounds (these become the defaults)
        if self.plotted_x and self.plotted_y:
            all_x = np.concatenate(self.plotted_x)
            all_y = np.concatenate(self.plotted_y)
            x_min_data, x_max_data = all_x.min(), all_x.max()
            y_min_data, y_max_data = all_y.min(), all_y.max()
            x_range = x_max_data - x_min_data
            y_range = y_max_data - y_min_data
            x_pad = x_range * 0.05 if x_range > 0 else 1
            y_pad = y_range * 0.05 if y_range > 0 else 1
            x_min = x_min_data - x_pad
            x_max = x_max_data + x_pad
            y_min = y_min_data - y_pad
            y_max = y_max_data + y_pad
    
        # Override with user values if they are not None
        if self.EDIT_XMIN is not None:
            x_min = self.EDIT_XMIN
        if self.EDIT_XMAX is not None:
            x_max = self.EDIT_XMAX
        if self.EDIT_YMIN is not None:
            y_min = self.EDIT_YMIN
        if self.EDIT_YMAX is not None:
            y_max = self.EDIT_YMAX

        return((x_min, x_max), (y_min, y_max))
    
    def _update_display_scaling(self):
        if self.EDIT_TOGGLE_STRETCH == 1:
            self._apply_scale_to_fit_canvas_size()
        elif self.EDIT_TOGGLE_STRETCH == 2:
            self._update_image_preview()
      
            
            
    def _build_annotated_axes(self):
        """Layout port of plot_psense(): main axes squished to the left,
        a lightcyan comment panel occupying the widened right portion."""
        original_right = 1.0 / self.EDIT_COMMENT_WIDTH_FACTOR
        o_r_mod = original_right - 0.125
        self._plot_pos = [0.125, 0.125, o_r_mod, 0.82]
        self._text_pos = [original_right, 0.1, 1.0 - original_right - 0.005, 0.825]
        return self.fig.add_axes(self._plot_pos)

    def _draw_comment_panel(self, dat, ax):
        comment_text = getattr(dat, "comment", "") or ""
        timestamp = getattr(dat, "timestamp", "")
    
        ax_text = self.fig.add_axes(self._text_pos)
        ax_text.axis("off")
        ax_text.patch.set_visible(True)
        
        # Make sure the background is visible
        ax_text.patch.set_alpha(1.0)
        ax_text.patch.set_edgecolor('none')
    
        # Add text first
        header = ax_text.text(
            0.05, 0.97, "Comment",
            transform=ax_text.transAxes, verticalalignment="top",
            fontsize=self.EDIT_COMMENT_HEADER_SIZE, fontweight="bold",
        )
    
        # Calculate wrap width
        ax_text_width_px = ax_text.get_window_extent().width
        fontsize = self.EDIT_COMMENT_BODY_SIZE
        print(type(fontsize))
        dpi = self.fig.get_dpi()
        char_width_px = fontsize * dpi / 72 * 0.5
        wrap_width = max(int(ax_text_width_px / char_width_px), 10)
        wrapped = textwrap.fill(f"{comment_text}\n{timestamp}", width=wrap_width)
    
        body = ax_text.text(
            0.05, 0.9, wrapped,
            transform=ax_text.transAxes, verticalalignment="top",
            fontsize=fontsize, clip_on=False,
        )
    
        # Force a draw to get proper extents
        self.fig.canvas.draw()
    
        # Get the bounding box of the text
        bbox_header = header.get_window_extent()
        bbox_body = body.get_window_extent()
        
        # Combine the bounding boxes properly - use Bbox.union() with a list
        combined = mpl.transforms.Bbox.union([bbox_header, bbox_body])
    
        # Convert from display to axes coordinates
        inv = ax_text.transAxes.inverted()
        x0, y0 = inv.transform((combined.x0, combined.y0))
        x1, y1 = inv.transform((combined.x1, combined.y1))
        
        # Calculate the height of one line of text in axes coordinates
        # Get the body text height
        bbox_body_display = body.get_window_extent()
        # Get the height of the body text in axes units
        y0_body, _ = inv.transform((0, bbox_body_display.y0))
        _, y1_body = inv.transform((0, bbox_body_display.y1))
        row_height_px = fontsize * dpi / 72
        # Convert from pixels to axes coordinates using the axes height
        ax_height_px = ax_text.get_window_extent().height
        ax_height_axes = 1.0  # axes coordinates go from 0 to 1
        row_height_axes = row_height_px / ax_height_px * ax_height_axes
    
        # Extend y0 downward by one row height
        y0_extended = y0 - row_height_axes
    
        # Add some padding
        pad = 0.03
        
        # Create the background patch with border so it's visible
        patch = mpl.patches.Rectangle(
            (x0 - pad, y0_extended - pad),  # Extended y0
            (x1 - x0) + 2 * pad, 
            (y1 - y0_extended) + 2 * pad,   # Extended height
            facecolor="lightblue", 
            linewidth=1,
            transform=ax_text.transAxes, 
            zorder=0,
            alpha=0.9
        )
        ax_text.add_patch(patch)
        
        # Make sure the patch is behind the text
        patch.set_zorder(0)
        header.set_zorder(1)
        body.set_zorder(1)
    
        # Ensure the positions are set correctly
        ax.set_position(self._plot_pos)
        ax_text.set_position(self._text_pos)
    
        # Handle legend
        handles, labels = ax.get_legend_handles_labels()
        for handle in handles:
            if hasattr(handle, 'set_linewidth'):
                handle.set_linewidth(2)
            if hasattr(handle, 'set_markersize'):
                handle.set_markersize(8)
        legend = ax.legend(handles, labels, prop={"size": 10}, handlelength=0.9)
        ax_bottom = ax.get_position().y0
        ax_right = ax.get_position().x1
        legend.set_bbox_to_anchor((ax_right, ax_bottom), transform=self.fig.transFigure)
        legend._loc = 3  # lower left

    # -- Background selection / fitting ---------------------------------------
    def _toggle_anchor_selection(self, checked):
        if self.EDIT_TOGGLE_STRETCH == 2:
            self.anchor_btn.setChecked(False)
            QtWidgets.QMessageBox.information(self, "Not available", "Anchor selection not available in Real Size mode.")
            return
        if checked:
            self.span_btn.setChecked(False)
            self._anchor_click_cid = self.canvas.mpl_connect('button_press_event', self._on_anchor_click)
        else:
            if hasattr(self, '_anchor_click_cid'):
                self.canvas.mpl_disconnect(self._anchor_click_cid)
                self._anchor_click_cid = None
    
    def _on_anchor_click(self, event):
        if event.inaxes is None:
            return
        x, y = event.xdata, event.ydata
        self.current_state.anchor_points.append([float(x), float(y)])
        self._save_state()
        self._refresh_range_list()
        self._refresh_plot()
        
    def _toggle_select_range(self, checked, range_type):
        if self.EDIT_TOGGLE_STRETCH == 2:
            QtWidgets.QMessageBox.information(
                self, "Not available", "Span selection not available in Real Size mode."
            )
            self.span_btn.setChecked(False)
            return
        if checked:
            axes = self.fig.axes
            if axes:
                self._enable_span_selector(axes[0], range_type)
        else:
            if self.span_selector is not None:
                self.span_selector.disconnect_events()
                self.span_selector = None
            self._refresh_plot()
    
    def _enable_span_selector(self, ax, range_type):
        """Enable span selector with type"""
        self.span_selector = SpanSelector(
            ax, lambda xmin, xmax: self._on_span_select(xmin, xmax, range_type),
            "horizontal",
            useblit=True, 
            props=dict(alpha=0.35, facecolor="orange"),
        )
    
    def _on_span_select(self, xmin, xmax, range_type, skip_overlap_check=False):
        """Handle span selection; if a span is selected and the span button is active,
        the selected span is replaced by the newly drawn one."""
        # Ensure min/max order
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
    
        # Check if we are in "replacement mode": span button active and a span row is selected
        selected_span_idx = None
        if self.span_btn.isChecked() and self._selected_row is not None:
            item = self.range_list.item(self._selected_row)
            if item is not None:
                data = item.data(QtCore.Qt.UserRole)
                if data is not None:
                    typ, idx = data
                    if typ == 'span':
                        selected_span_idx = idx  # index into self.current_state.bg_ranges
    
        if not skip_overlap_check:
            # Check for overlaps against all spans EXCEPT the one we may replace
            for i, existing in enumerate(self.current_state.bg_ranges):
                if i == selected_span_idx:
                    continue  # skip the selected one – it will be deleted if we proceed
                ex_min, ex_max = existing[0], existing[1]
                if not (xmax <= ex_min or xmin >= ex_max):
                    # Overlap detected – snap to nearest valid position
                    dist_to_min = abs(xmin - ex_min)
                    dist_to_max = abs(xmax - ex_max)
                    if dist_to_min < dist_to_max:
                        xmax = ex_min
                    else:
                        xmin = ex_max
                    if xmin >= xmax:
                        QtWidgets.QMessageBox.information(
                            self, "Range too small",
                            "The range would be too small after avoiding overlap."
                        )
                        return  # do NOT delete the selected span
    
        # If we get here, the new span is valid – delete the selected one (if any)
        if selected_span_idx is not None:
            del self.current_state.bg_ranges[selected_span_idx]
            self._selected_row = None
    
        # Add the new span
        self.current_state.bg_ranges.append([float(xmin), float(xmax), range_type])
        self._sort_ranges()
        self._save_state()
        self._refresh_range_list()
        self._refresh_plot()
    
    
    def _refresh_fit_curve_list(self):
        """Repopulate the 'Curve for fitting' list from the active measurement.
    
        For segmented curves (fwd/rev sweeps), one row is shown per segment
        plus an '(all)' row — each row corresponds to a distinct fit target
        so that forward and reverse fits are stored separately.
        """
        if not hasattr(self, "fit_curve_list"):
            return
        self.fit_curve_list.blockSignals(True)
        self.fit_curve_list.clear()
    
        if self.loaded_file is None or self.current_meas_index is None:
            self.fit_curve_list.blockSignals(False)
            self._sync_fit_use_bg_toggle()
            return
    
        meas = self.loaded_file.measurements[self.current_meas_index]
        active_key = self.current_state.active_fit_key
    
        for curve in meas.curves:
            segs = curve.segments or []
            if len(segs) > 1:
                all_key = _fit_key(curve.name, [])
                item = QtWidgets.QListWidgetItem(f"{curve.name}  (all)")
                item.setData(QtCore.Qt.UserRole, all_key)
                self.fit_curve_list.addItem(item)
    
                for i in range(len(segs)):
                    if i == 0:
                        label = "fwd"
                    elif i == 1:
                        label = "rev"
                    else:
                        label = f"seg{i}"
                    key = _fit_key(curve.name, [i])
                    item = QtWidgets.QListWidgetItem(
                        f"    {curve.name}  ({label})")
                    item.setData(QtCore.Qt.UserRole, key)
                    self.fit_curve_list.addItem(item)
            else:
                key = _fit_key(curve.name, [])
                item = QtWidgets.QListWidgetItem(curve.name)
                item.setData(QtCore.Qt.UserRole, key)
                self.fit_curve_list.addItem(item)
    
        # Restore the saved selection if it exists
        if active_key:
            for i in range(self.fit_curve_list.count()):
                if (self.fit_curve_list.item(i).data(QtCore.Qt.UserRole)
                        == active_key):
                    self.fit_curve_list.setCurrentRow(i)
                    break
    
        self.fit_curve_list.blockSignals(False)
        self._sync_fit_use_bg_toggle()
    
    def _sync_fit_use_bg_toggle(self):
        """Sync the bg-removed toggle to the current active fit target."""
        if not hasattr(self, "fit_use_bg_toggle"):
            return
        key = self.current_state.active_fit_key
        if key is None:
            self.fit_use_bg_toggle.setEnabled(False)
            self.fit_use_bg_toggle.blockSignals(True)
            self.fit_use_bg_toggle.setChecked(False)
            self.fit_use_bg_toggle.blockSignals(False)
            return
        self.fit_use_bg_toggle.setEnabled(True)
        fit = self.current_state.fit_by_curve.get(key, {})
        self.fit_use_bg_toggle.blockSignals(True)
        self.fit_use_bg_toggle.setChecked(
            bool(fit.get("use_bg_removed", False)))
        self.fit_use_bg_toggle.blockSignals(False)
    
    def _on_fit_curve_selected(self, item):
        if item is None:
            return
        key = item.data(QtCore.Qt.UserRole)
        if not key:
            return
        self.current_state.active_fit_key = key
        self._save_state()
        self._sync_fit_use_bg_toggle()
        self._refresh_plot()
    
    def _on_fit_use_bg_toggled(self, checked):
        key = self.current_state.active_fit_key
        if key is None:
            return
        fit = self.current_state.fit_by_curve.setdefault(key, {})
        fit["use_bg_removed"] = bool(checked)
        self._save_state()
        
    def _sort_ranges(self):
        """Sort ranges by xmin"""
        self.current_state.bg_ranges.sort(key=lambda r: r[0])
    
    def _refresh_range_list(self):
        """Refresh the range list with both SPANs and anchor points."""
        self.range_list.blockSignals(True)
        self.range_list.clear()
    
        # SPAN ranges
        for i, r in enumerate(self.current_state.bg_ranges):
            if len(r) >= 3:
                xmin, xmax = r[0], r[1]
            else:
                xmin, xmax = r[0], r[1]
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            item = QtWidgets.QListWidgetItem(f"🟧 SPAN {i}: [{xmin:.4g}, {xmax:.4g}]")
            item.setData(QtCore.Qt.UserRole, ('span', i))
            self.range_list.addItem(item)
    
        # Anchor points
        for j, (xp, yp) in enumerate(self.current_state.anchor_points):
            item = QtWidgets.QListWidgetItem(f"🟣 ANCHOR {j}: x={xp:.4g}, y={yp:.4g}")
            item.setData(QtCore.Qt.UserRole, ('anchor', j))
            self.range_list.addItem(item)
    
        # Restore selection if the row still exists
        if self._selected_row is not None and self._selected_row < self.range_list.count():
            self.range_list.setCurrentRow(self._selected_row)
        else:
            self._selected_row = None
    
        self.range_list.blockSignals(False)   
    
    def _remove_range(self):
        """Remove the selected range (SPAN or anchor) and select the previous one."""
        # If nothing selected, delete the last item
        if self._selected_row is None:
            # Delete the last item in the combined list
            total = len(self.current_state.bg_ranges) + len(self.current_state.anchor_points)
            if total == 0:
                return
            # Determine the last item's type and index by examining the last row of the list widget
            last_row = total - 1
            item = self.range_list.item(last_row)
            if item is None:
                return
            data = item.data(QtCore.Qt.UserRole)
            if data is None:
                return
            typ, idx = data
            if typ == 'span':
                if 0 <= idx < len(self.current_state.bg_ranges):
                    del self.current_state.bg_ranges[idx]
            elif typ == 'anchor':
                if 0 <= idx < len(self.current_state.anchor_points):
                    del self.current_state.anchor_points[idx]
            else:
                return
            # After deletion, we have no selection, but we can set target row to last_row-1 if exists
            target_row = last_row - 1
            self._selected_row = target_row if target_row >= 0 else None
        else:
            # Delete the selected item
            row = self._selected_row
            item = self.range_list.item(row)
            if item is None:
                return
            data = item.data(QtCore.Qt.UserRole)
            if data is None:
                return
            typ, idx = data
            if typ == 'span':
                if 0 <= idx < len(self.current_state.bg_ranges):
                    del self.current_state.bg_ranges[idx]
            elif typ == 'anchor':
                if 0 <= idx < len(self.current_state.anchor_points):
                    del self.current_state.anchor_points[idx]
            else:
                return
            # Determine new selection: try to stay at the same row, else go to previous
            total = len(self.current_state.bg_ranges) + len(self.current_state.anchor_points)
            if total == 0:
                self._selected_row = None
            else:
                # If the same row now exists (i.e., the item that shifted up), keep it
                if row < total:
                    self._selected_row = row
                else:
                    # Otherwise, the last item was removed, so go to the previous row
                    self._selected_row = total - 1 if total > 0 else None
    
        self._save_state()
        self._refresh_range_list()
        self._refresh_plot()
    def _on_range_item_clicked(self, item):
        """Clicking a row selects/deselects it; no automatic deletion."""
        row = self.range_list.row(item)
        data = item.data(QtCore.Qt.UserRole)
        if data is None:
            return
    
        if self._selected_row == row:
            # Already selected → deselect
            self._selected_row = None
            self.range_list.blockSignals(True)
            self.range_list.setCurrentRow(-1)
            self.range_list.blockSignals(False)
            self._refresh_plot()
            return
    
        # New selection
        self._selected_row = row
        self.range_list.blockSignals(True)
        self.range_list.setCurrentRow(row)
        self.range_list.blockSignals(False)
        self._refresh_plot()
    
    
    def _clear_all_ranges(self):
        """Clear all ranges and anchor points."""
        if not self.current_state.bg_ranges and not self.current_state.anchor_points:
            return
        reply = QtWidgets.QMessageBox.question(
            self, "Clear All Ranges",
            "Remove all background ranges and anchor points?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.current_state.bg_ranges = []
            self.current_state.anchor_points = []
            self._selected_row = None
            self._save_state()
            self._refresh_range_list()
            self._refresh_plot()

    
    def _fit_curve(self):
        ### This needs a fit panel, where we can choose what type of fit to apply - mainly, this will be the ideal diode equation.
        None
    
    
    def _fit_background(self):
        """Fit background using selected mode - for when user clicks 'Fit from selection'"""
        if not self.raw_xy_list:
            QtWidgets.QMessageBox.information(
                self, "No selection", "Load a measurement first."
            )
            return
        # Target the first visible curve.
        dat = self.loaded_file.measurements[self.current_meas_index]
        visible = [c for c in dat.curves if c.name not in self.current_state.hidden_curves]
        if not visible:
            QtWidgets.QMessageBox.information(
                self, "No selection", "No visible curve to fit."
            )
            return
        target_name = visible[0].name
        curve_ranges = self.current_state.bg_ranges_by_curve.get(target_name, [])
        if not curve_ranges:
            QtWidgets.QMessageBox.information(
                self, "No selection", "Select at least one background range first."
            )
            return
    
        mode = self.current_state.bg_removal_mode
        new_coeffs = []
        fitted_any = False
        
        # Fit only the first visible curve.
        x, y = self.raw_xy_list[0]
        
        # Separate SPAN ranges from anything else.
        span_ranges = []
        for range_data in curve_ranges:
            if len(range_data) >= 3:
                xmin, xmax, rtype = range_data[0], range_data[1], range_data[2]
            else:
                xmin, xmax = range_data[0], range_data[1]
                rtype = RangeType.SPAN
            if rtype == RangeType.SPAN:
                span_ranges.append([xmin, xmax])
        
        if mode == BgRemovalMode.CONNECT_END:
            new_coeffs.append(None)
            fitted_any = True
        
        elif mode == BgRemovalMode.LINEAR:
            x_bg = []
            y_bg = []
            for xmin, xmax in span_ranges:
                mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                x_bg.extend(x[mask])
                y_bg.extend(y[mask])
            if len(x_bg) >= 2:
                coeffs = np.polyfit(x_bg, y_bg, 1).tolist()
                new_coeffs.append(coeffs)
                fitted_any = True
            else:
                new_coeffs.append(None)
        
        elif mode == BgRemovalMode.POLY:
            x_bg = []
            y_bg = []
            for xmin, xmax in span_ranges:
                mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                x_bg.extend(x[mask])
                y_bg.extend(y[mask])
            degree = self.current_state.poly_degree
            if len(x_bg) >= degree + 1:
                coeffs = np.polyfit(x_bg, y_bg, degree).tolist()
                new_coeffs.append(coeffs)
                fitted_any = True
            else:
                new_coeffs.append(None)
        
        elif mode == BgRemovalMode.SPLINE:
            new_coeffs.append(None)
            fitted_any = True
        
        elif mode == BgRemovalMode.EXPONENTIAL_DECAY:
            new_coeffs.append(None)
            fitted_any = True
        
        elif mode == BgRemovalMode.MANUAL_GRADIENT:
            new_coeffs.append(None)
            fitted_any = True
        
        if not fitted_any:
            QtWidgets.QMessageBox.warning(
                self, "Fit failed",
                "Not enough points in the selection for the selected mode."
            )
            return
        
        self.current_state.fit_coeffs = new_coeffs
        self._save_state()
        self._refresh_plot()

    def _toggle_remove_background(self, checked):
        self.current_state.remove_background = checked
        self._save_state()
        self._refresh_plot()
    
    
    # -- Selecting, editing and restoring text entry fields ---
    
    def _update_rcparams_from_ui(self):
        """
        Update matplotlib rcParams from UI controls. 
        Note that if you then load an rc_param dict from the advanced tab, that this will update the UI. This is how we can save plotting defaults.
        """
    
        rc_updates = {
            # Font sizes
            'font.size'             : self.EDIT_TEXT_BODY_SIZE,
            'axes.titlesize'        : self.EDIT_TEXT_HEADER_SIZE,
            'axes.labelsize'        : self.EDIT_AXIS_LABEL_SIZE,
            'xtick.labelsize'       : self.EDIT_TICK_SIZE,
            'ytick.labelsize'       : self.EDIT_TICK_SIZE,
            'legend.fontsize'       : self.EDIT_LEGEND_BODY_SIZE,
            'legend.title_fontsize' : self.EDIT_LEGEND_HEADER_SIZE,
            
            # Figure defaults
            'figure.dpi'            : self.EDIT_FIGURE_DPI,
            'figure.figsize'        :[self.EDIT_FIGSIZE_X, self.EDIT_FIGSIZE_Y],

            # Lines and Markers
            'lines.linewidth'       : self.EDIT_LINEWIDTH,
            'lines.markersize'      : self.EDIT_MARKERSIZE,
            'axes.grid': False,
            'grid.alpha': 0.3,
            'legend.frameon': True,
            'legend.edgecolor': 'lightgray',
        }
        
        mpl.rcParams.update(rc_updates)
    
    def _reset_all_parameters(self):
        """Reset all parameters to defaults"""
        
        # First, update DEFAULT_* values from current defaults
        for edit_name in self.UI_RCPCONFIG_MAP.keys():
            default_value = self.get_default_for_edit(edit_name)
            if default_value is not None:
                # Set the DEFAULT_* variable for reference
                setattr(self, f"DEFAULT_{edit_name}", default_value)
                setattr(self, edit_name, default_value)
        
        # Reset UI-only variables
        for edit_name, default_value in self.UI_ONLY_DEFAULTS.items():
            setattr(self, f"DEFAULT_{edit_name}", default_value)
            setattr(self, edit_name, default_value)
        
        
        # Reset user rcParams
        self._user_rcparams = {}
        
        # Reset matplotlib rcParams to defaults
        mpl.rcParams.update(self.RCPARAM_DEFAULTS)
        
        # Update UI elements
        self._update_ui_from_config()
        self._sync_override_rows()
        self._apply_splitter_sizes()
        # Only apply figsize if NOT Real Size mode
        if self.EDIT_TOGGLE_STRETCH != 2:
            self._apply_figsize(self.EDIT_SHOW_COMMENT)
            
        self._refresh_plot()
        self._save_session_parameters()
        
        QtWidgets.QMessageBox.information(
            self, "Reset Complete", "All parameters reset to defaults."
        )
        
    def _on_param_changed(self):
        """Called when any parameter is changed - updates the plot if needed"""
        self.EDIT_COMMENT_WIDTH_FACTOR = self.width_factor_row.get_value()
        self.EDIT_FIGURE_DPI = self.dpi_row.get_value()
        self.EDIT_FIGSIZE_X = self.save_width_row.get_value()
        self.EDIT_FIGSIZE_Y = self.save_height_row.get_value()
        
        self._apply_all_rcparams()
        self._refresh_plot()
        self._save_session_parameters()
        
    def _update_toggle_buttons(self):
        """Set all toggle buttons from their EDIT_* variables, blocking signals."""
        for var_name, btn in self._toggle_widgets.items():
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(getattr(self, var_name, False))
                btn.blockSignals(False)
                
    def _apply_splitter_sizes(self):
        if not hasattr(self, 'main_splitter'):
            return
        sizes = getattr(self, 'EDIT_SPLITTER_SIZES', None)
        if (isinstance(sizes, (list, tuple)) and len(sizes) == 3
                and all(isinstance(x, (int, float)) and x > 0
                        for x in sizes)):
            self.main_splitter.setSizes([int(x) for x in sizes])        
            
    def _load_session_parameters(self):
        """Load saved parameter values from the display config"""
        if DISPLAY_PATH.exists():
            try:
                with open(DISPLAY_PATH) as f:
                    session_data = json.load(f)
                
                # Load UI config - iterate over both maps
                ui_config = session_data.get('ui_config', {})
                
                # Load rcParam-linked variables
                for edit_name in self.UI_RCPCONFIG_MAP.keys():
                    if edit_name in ui_config:
                        setattr(self, edit_name, ui_config[edit_name])
                
                # Load UI-only variables
                for edit_name in self.UI_ONLY_DEFAULTS.keys():
                    if edit_name in ui_config:
                        setattr(self, edit_name, ui_config[edit_name])
                    
                # Restore window state
                if hasattr(self, 'EDIT_WINDOW_X'):
                    self._restore_window_state()
                
                # NOTE: rcParam overrides are read from session.json which lives in their own dedicated file (RCPARAMS_OVERRIDE_PATH, see load_rcparams_override()).
                # Save/Load/Reset in the Display tab has one unambiguous source of truth. self._user_rcparams is already populated from that file in __init__.
                
                # Update UI elements
                self._update_ui_from_config()
                                
                #Synchronise buttons
                self._update_toggle_buttons()
                
                # Apply rcParams after loading, then make sure the EDIT_* fields reflect whatever the (higher-priority) rcParam overrides actually set.
                self._apply_all_rcparams()
                self._sync_edit_vars_from_rcparams(self._user_rcparams)
                self._update_ui_from_config()
                self._apply_splitter_sizes() 
                
            except Exception as e:
                print(f"Error loading session: {e}")
                
    def _save_session_parameters(self):
        """Save all settings to a single session file"""
        
        # Save window state
        self._save_window_state()
        
        # Build UI config from both maps
        ui_config = {}
        # Snapshot actual splitter sizes so a manual drag is persisted
        # even though splitterMoved() isn't connected.
        if hasattr(self, 'main_splitter'):
            self.EDIT_SPLITTER_SIZES = list(self.main_splitter.sizes())
            
        # Save rcParam-linked variables
        for edit_name in self.UI_RCPCONFIG_MAP.keys():
            if hasattr(self, edit_name):
                ui_config[edit_name] = getattr(self, edit_name)
        
        # Save UI-only variables
        for edit_name in self.UI_ONLY_DEFAULTS.keys():
            if hasattr(self, edit_name):
                ui_config[edit_name] = getattr(self, edit_name)
        
        # Save show_comment state
        ui_config['show_comment'] = self.EDIT_SHOW_COMMENT
        
        # Build rcParams config (only user-modified rcParams that differ from defaults)
        rc_defaults = self.RCPARAM_DEFAULTS
        current_rc = {k: v for k, v in mpl.rcParams.items() 
                      if k in rc_defaults and v != rc_defaults.get(k)}
        
        session_data = {
                        'ui_config': ui_config,
                        'current_rcparams': current_rc,
                        }
        
        DISPLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DISPLAY_PATH, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

    # -- rcParam override file handling (Save / Load / Reset buttons) --------

    def _sync_edit_vars_from_rcparams(self, rc_dict, fallback_to_defaults=False):
        """
        Keep the EDIT_* UI variables consistent with a given rcParams dict for every EDIT_* that maps to an rcParam key present in it.
        Fallback simply forces rc_dict to be pulled from RCPARAM_DEFAULTS instead.
        """
        for edit_name, mapping in self.UI_RCPCONFIG_MAP.items():
            if mapping is None:
                continue
            rc_key, index = mapping
            if rc_key in rc_dict:
                value = rc_dict[rc_key]
            elif fallback_to_defaults and rc_key in self.RCPARAM_DEFAULTS:
                value = self.RCPARAM_DEFAULTS[rc_key]
            else:
                continue
            if index is not None and isinstance(value, (list, tuple)):
                value = value[index]
            setattr(self, edit_name, value)

    def _get_current_rcparam_diffs(self):
        """
        Return the subset of rcParams (matching RCPARAM_DEFAULTS keys) whose current mpl.rcParams value differs from the code default.
        """
        diffs = {}
        for key, default_val in self.RCPARAM_DEFAULTS.items():
            current_val = mpl.rcParams.get(key, default_val)
            if current_val != default_val:
                diffs[key] = current_val
        return diffs

    def _save_rcparam_overrides(self):
        """
        Persist the current rcParams (on top of the code defaults) to disk.
        This then becomes the new baseline that loads automatically on
        every future launch.
        """
        self._apply_all_rcparams()  # make sure EDIT_* edits are reflected in mpl.rcParams first
        diffs = self._get_current_rcparam_diffs()
        save_rcparams_override(diffs)
        self._user_rcparams = diffs
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Saved {len(diffs)} rcParam override(s) as the new startup defaults."
        )

    def _load_rcparam_overrides(self):
        """
        Reload the saved override file and apply it to the current session (does not touch the file itself).
        """
        self._user_rcparams = load_rcparams_override()
        self._sync_edit_vars_from_rcparams(self._user_rcparams)
        self._apply_all_rcparams()
        self._update_ui_from_config()
        if self.EDIT_TOGGLE_STRETCH != 2:
            self._apply_figsize(self.EDIT_SHOW_COMMENT)
        self._refresh_plot()
        self._save_session_parameters()

    def _reset_rcparam_overrides(self):
        """
        Discard saved overrides and return to hardcoded defaults get_rcparam_defaults(), deleting the override file.
        """
        reply = QtWidgets.QMessageBox.question(
            self, "Reset rcParam overrides",
            "This deletes the saved rcParam override file and reverts to "
            "the defaults defined in the code. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        delete_rcparams_override()
        self._user_rcparams = {}
        self._sync_edit_vars_from_rcparams({}, fallback_to_defaults=True)
        mpl.rcParams.update(self.RCPARAM_DEFAULTS)
        self._apply_all_rcparams()
        self._update_ui_from_config()
        if self.EDIT_TOGGLE_STRETCH != 2:
            self._apply_figsize(self.EDIT_SHOW_COMMENT)
        self._refresh_plot()
        self._save_session_parameters()

    # -- Export -----------------------------------------------------------

    def _save_figure(self):
        if self.current_filename is None:
            return
        default_name = f"{Path(self.current_filename).stem}_m{self.current_meas_index}.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save figure", default_name)
        if not path:
            return

        on_screen_size = self.fig.get_size_inches().copy()
        on_screen_dpi = self.fig.get_dpi()

        self.fig.set_size_inches(self.EDIT_FIGSIZE_X, self.EDIT_FIGSIZE_Y)
        self.fig.set_dpi(self.EDIT_FIGURE_DPI)
        self.fig.savefig(path, dpi=self.EDIT_FIGURE_DPI)

        self.fig.set_size_inches(on_screen_size)
        self.fig.set_dpi(on_screen_dpi)
        self.canvas.draw()

def get_rcparam_defaults():
    fontfamily = "Arial" if platform.system() == "Windows" else "Liberation Sans"
    titlefont = 15.0
    bigfont = 14.0
    mediumfont = 12.0
    defaultsDict = {
                    'axes.formatter.use_mathtext': True,
                    'text.usetex': False,
                    'font.family': fontfamily,
                    
                    # Figure Dimensions 4:3 default
                    'figure.figsize': [6, 4.5],      
                    'figure.dpi': 200.0,               
                    
                    # Label Fontsizes
                    'font.size': bigfont,                 
                    'xtick.labelsize': mediumfont,
                    'ytick.labelsize': mediumfont,           
                    'legend.fontsize': mediumfont,           
                    'legend.title_fontsize': bigfont,   
                    'axes.titlesize':titlefont,
                    'axes.labelsize': bigfont, 
                    
                    #Standard Linewidths and Markers
                    'lines.linewidth': 1.5,
                    'lines.markeredgewidth': 1.5,
                    'lines.markersize': 5.0,
                    'axes.linewidth': 1.0,           
                    
                    # Tick Dimensions and Padding
                    'xtick.major.size': 4,           
                    'xtick.minor.size': 2.5,         
                    'xtick.major.width': 1.2,        
                    'xtick.minor.width': 0.8,        
                    'xtick.major.pad': 4,            
                    'xtick.minor.pad': 4,
                    'xtick.minor.visible': True,
                    
                    'ytick.major.size': 4,
                    'ytick.minor.size': 2.5,
                    'ytick.major.width': 1.2,
                    'ytick.minor.width': 0.8,
                    'ytick.major.pad': 4,
                    'ytick.minor.pad': 4,
                    'ytick.minor.visible': True,
                    
                    # Legend Structural Padding (Kept intact)
                    'legend.borderpad': 0.4,
                    'legend.labelspacing': 0.4,
                    'legend.handlelength': 1.25,
                    'legend.handleheight': 0.7,
                    'legend.handletextpad': 0.5,
                    'legend.borderaxespad': 0.5,
                    'legend.columnspacing': 1.0, 
                    
                    'axes.grid': False,
                    'axes.axisbelow': True,
                    'figure.autolayout': False,
                    'figure.constrained_layout.use': False
                    }
    rcparamdefaults = dict(mpl.rcParamsDefault)
    rcparamdefaults.update(defaultsDict)
    return(rcparamdefaults)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()