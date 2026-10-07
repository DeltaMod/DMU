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
import sys
import json
import textwrap
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from typing import Optional
import io 
import numpy as np

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

#Local Packages:
try:
    from .Widgets import *
    from .Plotting_Tools import *
    from .Data_Containers import *
except ImportError:
    from Widgets import *
    from Plotting_Tools import *
    from Data_Containers import *

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

def state_key(filename: str, measurement_id: int) -> str:
    return f"{filename}__m{measurement_id}"

def file_path_match(a, b) -> bool:
    """Compare two file paths tolerantly (None-safe, str/Path-safe)."""
    if a is None or b is None:
        return False
    return Path(a).resolve() == Path(b).resolve()
@dataclass
class CurveState:
    # Per-curve visibility — key is the curve's `name` (stable across reloads)
    hidden_curves: list[str] = field(default_factory=list)
    bg_ranges_by_curve: dict[str, list] = field(default_factory=dict)
    anchor_points_by_curve: dict[str, list] = field(default_factory=dict)
    # NEW: per-curve display overrides, keyed by ORIGINAL curve name
    # value = {"name": ..., "x_unit": ..., "y_unit": ...}
    curve_overrides: dict[str, dict] = field(default_factory=dict)

    # Global fit settings
    fit_coeffs: Optional[list] = None       # kept for backward compat
    remove_background: bool = False
    bg_removal_mode: int = 0
    poly_degree: int = 2
    spline_smoothing: float = 0.0

    # Legacy single-curve fields — loaded from old sidecars, migrated on read
    bg_ranges: list = field(default_factory=list)
    anchor_points: list = field(default_factory=list)

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
        
        self.current_file_path: Optional[str] = None   # path of the file currently loaded into DATA
        self.loaded_file: Optional[LoadedFile] = None   # current LoadedFile
        self.loaded_files: dict[str, LoadedFile] = {}   # cache of every ticked file                                # ps.load_session_file() result for current_file_path

        self.current_file_index: Optional[int] = None
        self.current_meas_index: Optional[int] = None
        self.current_filename: Optional[str] = None
        
        
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
    # -- UI construction ----------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        root.addLayout(self._build_left_column(), stretch=0)
        root.addLayout(self._build_middle_column(), stretch=1)
        root.addLayout(self._build_right_column(), stretch=0)

    def _build_left_column(self):
        col = QtWidgets.QVBoxLayout()

        col.addWidget(QtWidgets.QLabel("Data folders"))
        self.folder_list = QtWidgets.QListWidget()
        self.folder_list.currentItemChanged.connect(self._on_folder_selected)
        self.folder_list.itemChanged.connect(self._on_folder_check_changed)
        col.addWidget(self.folder_list)

        btn_row = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add")
        del_btn = QtWidgets.QPushButton("Delete")
        add_btn.clicked.connect(self._add_folder)
        del_btn.clicked.connect(self._delete_folder)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        col.addLayout(btn_row)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        col.addWidget(line)

        col.addWidget(QtWidgets.QLabel("Measurements"))
        self.file_list = QtWidgets.QTreeWidget()
        self.file_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.file_list.setHeaderHidden(True)
        self.file_list.itemClicked.connect(self._on_entry_selected)
        self.file_list.itemChanged.connect(self._on_file_item_check_changed)
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
        self._build_classify_tab()
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
        """Build an Advanced tab for less frequently used settings"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
    
        layout.addWidget(QtWidgets.QLabel("Keithley column mapping"))
        self.keithley_x_col = QtWidgets.QLineEdit("V")
        self.keithley_y_col = QtWidgets.QLineEdit("I")
        layout.addWidget(self.keithley_x_col)
        layout.addWidget(self.keithley_y_col)
    
        layout.addWidget(QtWidgets.QLabel("Loading"))
        self.disable_multiload_toggle = QtWidgets.QPushButton("Disable multi-load")
        self.disable_multiload_toggle.setCheckable(True)
        self.disable_multiload_toggle.setToolTip(
            "When on, only one file is kept in memory. Ticking a second file "
            "or activating a measurement in a second file is refused."
        )
        self.disable_multiload_toggle.toggled.connect(self._toggle_disable_multiload)
        layout.addWidget(self.disable_multiload_toggle)
    
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Advanced")
    
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
        layout.addWidget(QtWidgets.QLabel("LOG entry for active measurement"))
    
        form = QtWidgets.QFormLayout()
        self._log_fields = {}
    
        # Comment gets its own multi-line edit
        self._log_comment = QtWidgets.QPlainTextEdit()
        self._log_comment.setFixedHeight(80)
        self._log_comment.textChanged.connect(self._on_log_comment_changed)
        form.addRow("Comment:", self._log_comment)
    
        for key in ("Timestamp", "Device", "Subdevice", "RunID", "SMU Order", "LightMicroscope",
                    "SMU1", "SMU2", "SMU3", "SMU4"):
            le = QtWidgets.QLineEdit()
            le.editingFinished.connect(lambda k=key: self._on_log_field_changed(k))
            self._log_fields[key] = le
            form.addRow(f"{key}:", le)
    
        layout.addLayout(form)
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Logbook")
    
    def _refresh_logbook_tab(self):
        """Populate the Logbook tab from the active measurement's log dict."""
        for w in (self._log_comment, *self._log_fields.values()):
            w.blockSignals(True)
    
        if self.loaded_file is None or self.current_meas_index is None:
            self._log_comment.setPlainText("")
            for w in self._log_fields.values():
                w.setText("")
        else:
            meas = self.loaded_file.measurements[self.current_meas_index]
            log = getattr(meas, "log", {}) or {}
            self._log_comment.setPlainText(str(log.get("Comment", meas.comment or "")))
            for k, w in self._log_fields.items():
                v = log.get(k, "")
                w.setText("" if v is None else str(v))
    
        for w in (self._log_comment, *self._log_fields.values()):
            w.blockSignals(False)
    
    def _on_log_comment_changed(self):
        if self.loaded_file is None or self.current_meas_index is None:
            return
        meas = self.loaded_file.measurements[self.current_meas_index]
        txt = self._log_comment.toPlainText()
        meas.comment = txt
        meas.log["Comment"] = txt
        self._persist_log_to_pkl()
        self._refresh_plot()
    
    def _on_log_field_changed(self, key: str):
        if self.loaded_file is None or self.current_meas_index is None:
            return
        meas = self.loaded_file.measurements[self.current_meas_index]
        val = self._log_fields[key].text()
        # light conversion: keep ints as ints where possible
        try:
            if val.strip().lstrip("-").isdigit():
                val_conv = int(val)
            elif val.strip().lower() in ("true", "false"):
                val_conv = val.strip().lower() == "true"
            else:
                val_conv = val
        except Exception:
            val_conv = val
        meas.log[key] = val_conv
        self._persist_log_to_pkl()
    
    def _persist_log_to_pkl(self):
        """Walk the in-memory device_data tree and rewrite the LOG dict for the
        active measurement, then dump to pkl."""
        if self.loaded_file is None or self.loaded_file.device_data is None:
            return
        run_key = self.loaded_file.measurements[self.current_meas_index].metadata.get("run_key")
        if run_key is None:
            return
        # find and replace in device_data
        target = None
        for np_k, devs in self.loaded_file.device_data.items():
            for dev, subs in devs.items():
                for sub, runs in subs.items():
                    if run_key in runs:
                        target = runs[run_key]
                        break
        if target is None:
            return
        target["LOG"] = dict(self.loaded_file.measurements[self.current_meas_index].log)
        self.loaded_file.save()
    
    def _build_classify_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
    
        layout.addWidget(QtWidgets.QLabel("Regex (must name groups 'device' and 'subdevice')"))
        self.classify_pattern_edit = QtWidgets.QLineEdit(self.EDIT_CLASSIFY_PATTERN)
        layout.addWidget(self.classify_pattern_edit)
    
        apply_btn = QtWidgets.QPushButton("Apply pattern & switch view")
        apply_btn.clicked.connect(self._apply_classify_pattern)
        layout.addWidget(apply_btn)
    
        toggle_btn = QtWidgets.QPushButton("Toggle view (flat / classified)")
        toggle_btn.clicked.connect(self._toggle_tree_view)
        layout.addWidget(toggle_btn)
    
        clear_btn = QtWidgets.QPushButton("Clear classification cache")
        clear_btn.clicked.connect(lambda: self._classify_cache.clear())
        layout.addWidget(clear_btn)
    
        layout.addStretch(1)
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
        """Return (device, subdevice) for a file/folder name, cached."""
        key = str(path)
        if key in self._classify_cache:
            return self._classify_cache[key]
        import re as _re
        try:
            m = _re.search(self.EDIT_CLASSIFY_PATTERN, path.stem)
            if m and "device" in m.groupdict() and "subdevice" in m.groupdict():
                result = (m.group("device"), m.group("subdevice"))
            else:
                result = ("unclassified", path.stem)
        except _re.error:
            result = ("unclassified", path.stem)
        self._classify_cache[key] = result
        return result

    def _build_curves_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
    
        layout.addWidget(QtWidgets.QLabel("Ticked measurements"))
        self.curves_tree = QtWidgets.QTreeWidget()
        self.curves_tree.setHeaderLabels(["Curve", "Points", "Source"])
        self.curves_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.curves_tree.itemChanged.connect(self._on_curve_item_changed)
        layout.addWidget(self.curves_tree)
        # --- curve editor (edits the active measurement's curves) ---
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
        
        # remember the currently-edited original curve name
        self._curve_edit_target: Optional[str] = None
        self.curves_tree.itemSelectionChanged.connect(self._on_curve_selection_changed)
        
        #Button rows
        btn_row = QtWidgets.QHBoxLayout()
        show_all_btn = QtWidgets.QPushButton("Show all")
        show_all_btn.clicked.connect(lambda: self._set_all_curves_visible(True))
        hide_all_btn = QtWidgets.QPushButton("Hide all")
        hide_all_btn.clicked.connect(lambda: self._set_all_curves_visible(False))
        untick_root_btn = QtWidgets.QPushButton("Untick roots")
        untick_root_btn.setToolTip("Untick every file/measurement group (leaves curves as-is)")
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
    

    # -- Folder management ----------------------------------------------------

    def _restore_folders(self):
        workspace = load_workspace()
        self.folder_list.blockSignals(True)
        for entry in workspace.get("folders", []):
            path = entry["path"]
            checked = entry.get("checked", False)
            item = QtWidgets.QListWidgetItem(path)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
            self.folder_list.addItem(item)
        self.folder_list.blockSignals(False)
    
        # Cache the ticked-file set from the workspace so the first
        # _rebuild_measurements_tree call has somewhere to read it from.
        self._ticked_files_from_workspace = set(workspace.get("ticked_files", []))
        self._rebuild_measurements_tree()

    def _add_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Add data folder")
        if not directory:
            return
        existing = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        if directory not in existing:
            item = QtWidgets.QListWidgetItem(directory)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked)
            self.folder_list.addItem(item)
            self._save_workspace()

    def _delete_folder(self):
        item = self.folder_list.currentItem()
        if item is None:
            return
        row = self.folder_list.row(item)
        self.folder_list.takeItem(row)
        self._save_workspace()
        self.file_list.clear()
        self._reset_loaded_data()
        self._refresh_curves_tab()

    def _reset_loaded_data(self):
        self.current_file_path = None
        self.loaded_file = None
        self.current_file_index = None
        self.current_meas_index = None
        self.current_filename = None
        self.fig.clear()
        self.canvas.draw()
    # -- Measurement list---------------

    def _on_folder_selected(self, item, _prev=None):
        """Focus a folder. Also transiently ticks it (unless the user had
        explicitly ticked it before, in which case it stays ticked)."""
        if item is None:
            self.current_folder = None
            return

        new_path = item.text()

        # 1. Untick any previously transient ticks that the user didn't confirm.
        self.folder_list.blockSignals(True)
        for i in range(self.folder_list.count()):
            other = self.folder_list.item(i)
            if other is item:
                continue
            if other.text() in self._temp_ticked_folders:
                other.setCheckState(QtCore.Qt.Unchecked)
                self._temp_ticked_folders.discard(other.text())
        self.folder_list.blockSignals(False)

        # 2. Transiently tick the new folder if it isn't already ticked.
        if item.checkState() != QtCore.Qt.Checked:
            self.folder_list.blockSignals(True)
            item.setCheckState(QtCore.Qt.Checked)
            self.folder_list.blockSignals(False)
            self._temp_ticked_folders.add(new_path)

        self.current_folder = Path(new_path)
        self._save_workspace()
        self._rebuild_measurements_tree()

    def _on_folder_check_changed(self, item):
        """User explicitly clicked the checkbox → not transient any more."""
        if item is None:
            return
        # Explicit click clears the transient mark.
        self._temp_ticked_folders.discard(item.text())
        self._save_workspace()
        self._rebuild_measurements_tree()

    def _on_entry_selected(self, item, _column=0):
        """A click on a measurement row makes it active. Selection set is
        handled separately in _on_selection_changed (fired by Qt on any change)."""
        if item is None:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None or payload[0] != "meas":
            return
        _, file_path, file_index, meas_index = payload
        self._set_active_measurement(file_path, file_index, meas_index)
    
    def _on_selection_changed(self):
        """Qt fires this whenever the highlighted set changes, including for
        ctrl/shift multi-selects. Update selected list, then replot."""
        sel = []
        for item in self.file_list.selectedItems():
            payload = item.data(0, QtCore.Qt.UserRole)
            if payload is None or payload[0] != "meas":
                continue
            _, file_path, _, meas_index = payload
            sel.append((file_path, meas_index))
        self.selected_measurements = sel
        if self.active_measurement is None and sel:
            # Fall back to the first highlighted item if nothing active yet
            fp, mi = sel[-1]
            payloads = [i.data(0, QtCore.Qt.UserRole) for i in self.file_list.selectedItems()]
            self._set_active_measurement(fp, 0, mi)
            return
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
        store = StateStore(Path(file_path).parent)
        key = state_key(self.current_filename, meas_index)
        self.current_state = store.load(key)
    
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
    
        self._selected_row = None
        self._refresh_curves_tab()
        self._refresh_range_list()
        self._refresh_logbook_tab()          # NEW
        self._refresh_curve_editor()         # NEW
        self._refresh_plot()
    
    def _rebuild_measurements_tree(self):
        """Populate self.file_list with the union of every ticked folder's files."""
        # Remember which files were ticked so we can restore tick state after rebuild.
        previously_ticked = self._get_ticked_file_paths()
        if not previously_ticked:
            previously_ticked = getattr(self, "_ticked_files_from_workspace", set())
    
        self.file_list.blockSignals(True)
        self.file_list.clear()
    
        ticked_folders = [
            Path(self.folder_list.item(i).text())
            for i in range(self.folder_list.count())
            if self.folder_list.item(i).checkState() == QtCore.Qt.Checked
        ]
        if self.EDIT_TREE_VIEW == "classified":
            # Build a {device: {subdevice: [file_path, ...]}} map across all ticked folders
            tree: dict[str, dict[str, list[Path]]] = {}
            for folder_path in ticked_folders:
                if not folder_path.is_dir():
                    continue
                for f in sorted(p for p in folder_path.iterdir() if p.is_file()):
                    if self.registry.find(f) is None:
                        continue
                    dev, sub = self._classify_path(f)
                    tree.setdefault(dev, {}).setdefault(sub, []).append(f)
        
            for dev, subs in sorted(tree.items()):
                dev_node = QtWidgets.QTreeWidgetItem([dev])
                dev_node.setFlags(QtCore.Qt.ItemIsEnabled)
                dev_node.setExpanded(True)
                self.file_list.addTopLevelItem(dev_node)
                for sub, files in sorted(subs.items()):
                    sub_node = QtWidgets.QTreeWidgetItem([sub])
                    sub_node.setFlags(QtCore.Qt.ItemIsEnabled)
                    sub_node.setExpanded(True)
                    dev_node.addChild(sub_node)
                    for f in files:
                        # ... same header + measurement children code as the flat branch ...
                        pass
            self.file_list.expandAll()
            self.file_list.blockSignals(False)
            self._refresh_curves_tab()
            return
        
        for folder_path in ticked_folders:
            if not folder_path.is_dir():
                continue
    
            # Top-level node per folder, so the user can see which folder a file belongs to.
            folder_node = QtWidgets.QTreeWidgetItem([str(folder_path.name)])
            folder_node.setFlags(QtCore.Qt.ItemIsEnabled)
            folder_node.setExpanded(True)
            self.file_list.addTopLevelItem(folder_node)
    
            try:
                all_paths = sorted([p for p in folder_path.iterdir() if p.is_file()])
            except OSError:
                continue
    
            for f in all_paths:
                src = self.registry.find(f)
                if src is None:
                    continue
    
                header = QtWidgets.QTreeWidgetItem([f.name])
                header.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                was_ticked = str(f) in previously_ticked
                header.setCheckState(0, QtCore.Qt.Checked if was_ticked else QtCore.Qt.Unchecked)
                header.setData(0, QtCore.Qt.UserRole, ("file", str(f), 0))
                folder_node.addChild(header)
    
                try:
                    measurements = src.list_measurements(f)
                except Exception as exc:
                    err = QtWidgets.QTreeWidgetItem([f"    (failed to load: {exc})"])
                    err.setFlags(QtCore.Qt.NoItemFlags)
                    header.addChild(err)
                    header.setExpanded(True)
                    continue
    
                for meas_index, meas_title in measurements:
                    child = QtWidgets.QTreeWidgetItem([f"    {meas_index}: {meas_title}"])
                    child.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                    child.setData(0, QtCore.Qt.UserRole,
                                  ("meas", str(f), 0, meas_index))
                    header.addChild(child)
    
                header.setExpanded(True)
    
        self.file_list.expandAll()
        self.file_list.blockSignals(False)
        self._refresh_curves_tab()
        
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
        state_folder = Path(file_path).parent
        store = StateStore(state_folder)
        key = state_key(self.current_filename, meas_index)
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
        self._refresh_plot()
        
    def _get_ticked_file_paths(self) -> set:
        """Return the set of file paths whose rows are currently ticked."""
        ticked = set()
        for i in range(self.file_list.topLevelItemCount()):
            folder_node = self.file_list.topLevelItem(i)
            for j in range(folder_node.childCount()):
                header = folder_node.child(j)
                payload = header.data(0, QtCore.Qt.UserRole)
                if payload is None or payload[0] != "file":
                    continue
                if header.checkState(0) == QtCore.Qt.Checked:
                    ticked.add(payload[1])
        return ticked
    
    def _save_workspace(self):
        """Persist folder list (with check states) and ticked file paths."""
        folders = [
            {
                "path": self.folder_list.item(i).text(),
                "checked": self.folder_list.item(i).checkState() == QtCore.Qt.Checked,
            }
            for i in range(self.folder_list.count())
        ]
        ticked_files = list(self._get_ticked_file_paths())
        save_workspace({"folders": folders, "ticked_files": ticked_files})    
    def _on_file_item_check_changed(self, item, column):
        """Tick/untick a file row. Cascades to measurement children, loads
        the file into memory if newly ticked, and persists tick state."""
        if column != 0:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None or payload[0] != "file":
            return
        _, file_path, _ = payload
        ticked = item.checkState(0) == QtCore.Qt.Checked

        # --- cascade to measurement children ---
        self.file_list.blockSignals(True)
        for j in range(item.childCount()):
            child = item.child(j)
            cpayload = child.data(0, QtCore.Qt.UserRole)
            if cpayload is None or cpayload[0] != "meas":
                continue
            child.setCheckState(0, QtCore.Qt.Checked if ticked else QtCore.Qt.Unchecked)
        self.file_list.blockSignals(False)

        if ticked:
            # Multi-load guard.
            if getattr(self, "EDIT_DISABLE_MULTILOAD", False):
                others = [p for p in self.loaded_files.keys() if p != file_path]
                if others:
                    self.file_list.blockSignals(True)
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                    self.file_list.blockSignals(False)
                    QtWidgets.QMessageBox.information(
                        self, "Multi-load disabled",
                        "Advanced settings forbid loading more than one file "
                        "at a time. Untick the current one first."
                    )
                    return
            # Load into memory if not already loaded.
            if file_path not in self.loaded_files:
                try:
                    self.loaded_files[file_path] = self.registry.load(Path(file_path))
                except Exception as exc:
                    QtWidgets.QMessageBox.warning(
                        self, "Load failed", f"Could not load {file_path}:\n{exc}"
                    )
                    self.file_list.blockSignals(True)
                    item.setCheckState(0, QtCore.Qt.Unchecked)
                    self.file_list.blockSignals(False)
                    return
        # Unticking does not evict from memory (cheap); it just drops the file
        # from the Curves tab.

        self._save_workspace()
        self._refresh_curves_tab()

        
    def _refresh_curves_tab(self):
        """Rebuild the Curves tab from every ticked file in file_list."""
        self.curves_tree.blockSignals(True)
        self.curves_tree.clear()
        

        # Gather every ticked file from the file_list tree.
        ticked: list[tuple[str, LoadedFile]] = []
        for i in range(self.file_list.topLevelItemCount()):
            folder_node = self.file_list.topLevelItem(i)
            for j in range(folder_node.childCount()):
                header = folder_node.child(j)
                payload = header.data(0, QtCore.Qt.UserRole)
                if payload is None or payload[0] != "file":
                    continue
                if header.checkState(0) != QtCore.Qt.Checked:
                    continue
                file_path = payload[1]
                lf = self.loaded_files.get(file_path)
                if lf is not None:
                    ticked.append((file_path, lf))

        # Group by source type so mixed file types sort together.
        by_source: dict[str, list[tuple[str, LoadedFile]]] = {}
        for fp, lf in ticked:
            by_source.setdefault(lf.source_type, []).append((fp, lf))

        for source_type in sorted(by_source.keys()):
            entries = by_source[source_type]
            source_root = QtWidgets.QTreeWidgetItem([source_type, "", ""])
            source_root.setFlags(
                source_root.flags() | QtCore.Qt.ItemIsUserCheckable
            )
            source_root.setCheckState(0, QtCore.Qt.Checked)
            source_root.setData(0, QtCore.Qt.UserRole, ("source_root", source_type))
            self.curves_tree.addTopLevelItem(source_root)

            for fp, lf in entries:
                file_root = QtWidgets.QTreeWidgetItem([Path(fp).name, "", ""])
                file_root.setFlags(
                    file_root.flags() | QtCore.Qt.ItemIsUserCheckable
                )
                file_root.setCheckState(0, QtCore.Qt.Checked)
                file_root.setData(0, QtCore.Qt.UserRole, ("file_root", fp))
                source_root.addChild(file_root)

                for meas in lf.measurements:
                    meas_root = QtWidgets.QTreeWidgetItem([meas.title, "", ""])
                    meas_root.setFlags(
                        meas_root.flags() | QtCore.Qt.ItemIsUserCheckable
                    )
                    meas_root.setCheckState(0, QtCore.Qt.Checked)
                    meas_root.setData(0, QtCore.Qt.UserRole,
                                      ("meas_root", fp, meas.measurement_id))
                    file_root.addChild(meas_root)

                    for curve in meas.curves:
                        hidden = (
                            file_path_match(fp, self.current_file_path)
                            and meas.measurement_id == self.current_meas_index
                            and curve.name in self.current_state.hidden_curves
                        )
                        child = QtWidgets.QTreeWidgetItem([
                            curve.name,
                            str(len(curve.x)),
                            curve.source_key or "",
                        ])
                        child.setFlags(child.flags() | QtCore.Qt.ItemIsUserCheckable)
                        child.setCheckState(0, QtCore.Qt.Unchecked if hidden else QtCore.Qt.Checked)
                        child.setData(0, QtCore.Qt.UserRole,
                                      ("curve", fp, meas.measurement_id, curve.name))
                        meas_root.addChild(child)

        self.curves_tree.expandAll()
        self.curves_tree.blockSignals(False)
    
    def _on_curve_item_changed(self, item, column):
        if column != 0:
            return
        payload = item.data(0, QtCore.Qt.UserRole)
        if payload is None:
            return
        kind = payload[0]
    
        # --- cascade for root nodes ---
        if kind in ("source_root", "file_root", "meas_root"):
            ticked = item.checkState(0) == QtCore.Qt.Checked
            self.curves_tree.blockSignals(True)
            self._cascade_check(item, ticked)
            self.curves_tree.blockSignals(False)
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
            state_folder = Path(self.current_file_path).parent
            key = state_key(self.current_filename, self.current_meas_index)
            StateStore(state_folder).save(key, self.current_state)

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

    def _update_axes_frame(self, ax, mode):
        """Update axes frame appearance based on mode"""
        if mode == 2:  # Real Size
            # Make frame visible
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('red')
            # Add tick marks to show real size
            ax.tick_params(axis='both', which='major', labelsize=self.EDIT_TICK_SIZE)
        else:
            # Normal frame
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')
            
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
    
    def _update_display_mode(self):
        """Switch between canvas and image display"""
        if self.EDIT_TOGGLE_STRETCH == 2:
            # Real Size mode - show image
            self.display_stack.setCurrentIndex(1)  # Image label
            # Disable toolbar interactions for image mode
            self._set_toolbar_enabled(False)
        else:
            # Stretch/Scale modes - show interactive canvas
            self.display_stack.setCurrentIndex(0)  # Canvas
            self._set_toolbar_enabled(True)
    
    def _set_toolbar_enabled(self, enabled):
        """Enable/disable toolbar interactions"""
        # Find the navigation toolbar and toggle its interactivity
        for child in self.findChildren(NavigationToolbar):
            child.setEnabled(enabled)
    
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
        """Save session when closing the window"""
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
            from scipy.optimize import curve_fit
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
    
            
    def _refresh_plot(self):
        if self.span_selector is not None:
            try:
                self.span_selector.disconnect_events()
            except Exception:
                pass
            self.span_selector = None
    
        self.fig.clear()
        self.image_label.clear()
    
        if self.loaded_file is None or self.current_meas_index is None:
            self.canvas.draw()
            return
    
        # --- target list ---
        if self.EDIT_MULTIPLOT and self.selected_measurements:
            targets = list(self.selected_measurements)
        else:
            targets = [(self.current_file_path, self.current_meas_index)]
        # drop any targets whose file isn't in memory yet
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
    
        self.plotted_x = []
        self.plotted_y = []
    
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
    
        for file_path, meas_idx in targets:
            lf = self.loaded_files[file_path]
            if meas_idx >= len(lf.measurements):
                continue
            meas = lf.measurements[meas_idx]
            is_active = (file_path == self.current_file_path
                         and meas_idx == self.current_meas_index)
    
            # Curve filter: active uses state.hidden_curves; others use their own sidecar
            if is_active:
                hidden = self.current_state.hidden_curves
                bg_by_curve = self.current_state.bg_ranges_by_curve
                anchors_by_curve = self.current_state.anchor_points_by_curve
                remove_bg = self.current_state.remove_background
                bg_mode = self.current_state.bg_removal_mode
            else:
                st = StateStore(Path(file_path).parent).load(
                    state_key(Path(file_path).name, meas_idx))
                hidden = st.hidden_curves
                bg_by_curve = st.bg_ranges_by_curve
                anchors_by_curve = st.anchor_points_by_curve
                remove_bg = st.remove_background
                bg_mode = st.bg_removal_mode
    
            prefix = "" if is_active else f"[m{meas_idx}] "
            overrides = self.current_state.curve_overrides if is_active else st.curve_overrides
    
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
                                from scipy.interpolate import splrep, splev
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
    
                label = prefix + disp
                ax.plot(x, y_plot, label=label)
    
                # Baseline visualisation (only if not removed and not hidden)
                if (baseline is not None and not remove_bg
                        and not self.EDIT_HIDE_SPANS and is_active):
                    ax.plot(x, baseline, '--', color='red', alpha=0.7, linewidth=1.5,
                            label='Baseline' if not any_baseline else '')
                    any_baseline = True
    
                # SPAN/anchor overlays (active measurement only)
                if not self.EDIT_HIDE_SPANS and is_active:
                    for xmin, xmax in span_ranges:
                        ax.axvspan(xmin, xmax, color="orange", alpha=0.25)
                    for j, (xp, yp) in enumerate(anchor_points):
                        ax.scatter(xp, yp, color='magenta', s=40, zorder=5,
                                   label='Anchors' if j == 0 else "")
    
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
        if self.plotted_x:
            ax.legend(loc="best")
    
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
                
            except Exception as e:
                print(f"Error loading session: {e}")
                
    def _save_session_parameters(self):
        """Save all settings to a single session file"""
        
        # Save window state
        self._save_window_state()
        
        # Build UI config from both maps
        ui_config = {}
        
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
    import platform
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