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

import numpy as np

# Must be set before QApplication is constructed - avoids Qt's automatic
# HiDPI scaling double-counting against matplotlib's own devicePixelRatio
# handling, which was producing oversized text and a clipped toolbar.
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

# ---------------------------------------------------------------------------
# pypalmsens integration
# ---------------------------------------------------------------------------

PSSESSION_GLOB = "*.pssession"  # file pattern shown in the file list


def load_session_file(filepath: Path):
    """Load a .pssession file and return the list of measurements (DATA)."""
    import pypalmsens as ps
    return ps.load_session_file(str(filepath))


def curve_display_name(curve, index=0):
    """Best-effort curve label. TODO: confirm the right attribute for your
    curve objects - this tries the common candidates in order."""
    for attr in ("title", "name", "label", "array_type"):
        val = getattr(curve, attr, None)
        if val:
            return str(val)
    return f"curve{index}"


def curve_to_xy(curve):
    """Extract (x, y) arrays from a pypalmsens curve via a disposable scratch
    axes, rather than relying on internal attribute names or touching the
    real plot."""
    scratch_fig = Figure()
    scratch_ax = scratch_fig.add_subplot(111)
    curve.plot(ax=scratch_ax)
    if not scratch_ax.lines:
        raise RuntimeError("curve.plot(ax=ax) did not add a line to the axes")
    line = scratch_ax.lines[0]
    return np.asarray(line.get_xdata(), dtype=float), np.asarray(line.get_ydata(), dtype=float)


# ---------------------------------------------------------------------------
# Per-measurement operator state, persisted as JSON, never touches raw data
# ---------------------------------------------------------------------------

@dataclass
class CurveState:
    """Per-measurement, saved to disk. "show_comment" deliberately lives on
    MainWindow instead (see self.show_comment) - it's a global display
    toggle, not something that should vary per file."""
    bg_ranges: list = field(default_factory=list)       # [[xmin, xmax], ...], shared across curves
    fit_coeffs: Optional[list] = None                    # one np.polyfit()-style list per curve, or None per curve
    remove_background: bool = False

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # tolerate old sidecars that still have a "show_comment" key
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


def state_key(filename: str, measurement_id: int) -> str:
    return f"{filename}__m{measurement_id}"


class StateStore:
    """One JSON sidecar per (file, measurementID), in <folder>/.psense_gui_state/"""

    def __init__(self, folder: Path):
        self.state_dir = Path(folder) / ".psense_gui_state"
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
# Small config: remembers which folders you've added, across sessions
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".config" / "palmsens_gui" / "folders.json"


def load_folder_list():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return []


def save_folder_list(folders):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(folders, f, indent=2)


# ---------------------------------------------------------------------------
# rcParams override file: a dedicated, explicit "startup defaults" layer.
# Separate from session.json (which just remembers raw EDIT_* field values) -
# this file is only ever written by the "Save" button on the Display tab,
# only ever re-applied to the running session by "Load", and only ever
# cleared by "Reset". It is loaded automatically once at startup so that
# whatever was last explicitly saved here becomes the new baseline on top of
# get_rcparam_defaults().
# ---------------------------------------------------------------------------

RCPARAMS_OVERRIDE_PATH = Path.home() / ".config" / "palmsens_gui" / "rcparams_override.json"


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

class EditableParamRow(QtWidgets.QWidget):
    def __init__(self, label, default_value, param_name, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self._default_value = default_value
        self.parent_window = parent
        self._is_updating = False  # Flag to prevent recursive updates
        
        # Create layout FIRST
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Label
        self.label = QtWidgets.QLabel(label)
        self.label.setFixedWidth(80)
        layout.addWidget(self.label)
        
        # Editable field - create BEFORE any potential access
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setText(str(default_value))
        self.line_edit.textChanged.connect(self._on_text_changed)
        
        # Add validator to only allow numbers (optional but helpful)
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.line_edit.setValidator(validator)
        
        layout.addWidget(self.line_edit)
        
        # Reset button
        self.reset_btn = QtWidgets.QPushButton("⟳")
        self.reset_btn.setFixedWidth(30)
        self.reset_btn.setToolTip("Reset to default")
        self.reset_btn.clicked.connect(self._reset_to_default)
        layout.addWidget(self.reset_btn)
    
    def _on_text_changed(self, text):
        """Only update if the text is valid and not empty"""
        if self._is_updating:
            return
        
        # Skip empty strings
        if not text or text.strip() == "":
            return
        
        try:
            # Try to convert to float
            value = float(text)
            
            # Check for special cases like NaN, Inf
            if not np.isfinite(value):
                return
            
            # Update the parent
            setattr(self.parent_window, self.param_name, value)
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        except ValueError:
            # Invalid input - do nothing
            pass
    
    def _reset_to_default(self):
        # Use DEFAULT_* if available
        default_name = f"DEFAULT_{self.param_name}"
        if hasattr(self.parent_window, default_name):
            default_value = getattr(self.parent_window, default_name)
            self._default_value = default_value
        self.line_edit.setText(str(self._default_value))
        setattr(self.parent_window, self.param_name, float(self._default_value))
        if hasattr(self.parent_window, '_on_param_changed'):
            self.parent_window._on_param_changed()
    
    def get_value(self):
        try:
            return float(self.line_edit.text())
        except ValueError:
            return self._default_value
    
    def reload_default(self):
        """Reload the default value from parent"""
        default_name = f"DEFAULT_{self.param_name}"
        if hasattr(self.parent_window, default_name):
            self._default_value = getattr(self.parent_window, default_name)
            self.line_edit.setText(str(self._default_value))
            setattr(self.parent_window, self.param_name, float(self._default_value))
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
# Add this at the top with other constants
class BgRemovalMode:
    LINEAR = 0           # Linear fit across selected ranges
    POLY = 1             # Polynomial fit (user-selectable degree)
    SPLINE = 2           # Spline interpolation through selected points
    REGION_DELETE = 3    # Delete data in selected regions entirely
    CONNECT_END = 4      # Connect end of one range to start of next with straight line
    SMOOTH = 5           # Smooth interpolation through selected points

@dataclass
class CurveState:
    """Per-measurement, saved to disk."""
    bg_ranges: list = field(default_factory=list)       # [[xmin, xmax], ...]
    fit_coeffs: Optional[list] = None                    # one np.polyfit()-style list per curve, or None per curve
    remove_background: bool = False
    bg_removal_mode: int = 0  # Default to LINEAR
    poly_degree: int = 2      # For POLY mode
    spline_smoothing: float = 0.0  # For SPLINE mode
    
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
        
class MultiParamRow(QtWidgets.QWidget):
    def __init__(self, headers, row_labels, param_names, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.param_names = param_names
        self.row_labels = row_labels
        self.headers = headers
        self._is_updating = False  # Add this flag
        
        # Don't call _load_defaults_from_parent here - do it later
        self.default_values = {}
        
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(2)
        
        # Create header labels (top row, starting at column 1)
        for col_idx, header_text in enumerate(headers):
            header = QtWidgets.QLabel(header_text)
            header.setStyleSheet("font-weight: bold;")
            layout.addWidget(header, 0, col_idx * 2 + 1, 1, 2)
        
        # Add a "Reset All" button in the top right corner
        reset_all_btn = QtWidgets.QPushButton("↺ Reset All")
        reset_all_btn.setFixedWidth(80)
        reset_all_btn.setToolTip("Reset all values to defaults")
        reset_all_btn.clicked.connect(self.reload_defaults)
        layout.addWidget(reset_all_btn, 0, len(headers) * 2 + 1)
        
        # Create rows with labels, entry boxes and reset buttons
        self.line_edits = {}
        self.reset_btns = {}
        
        for row_idx, row_label in enumerate(row_labels):
            # Row label (column 0)
            label = QtWidgets.QLabel(row_label)
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, row_idx + 1, 0)
            
            for col_idx, param_name in enumerate(param_names[row_idx]):
                # Entry box
                line_edit = QtWidgets.QLineEdit()
                line_edit.setFixedWidth(60)
                # Set initial value from parent if available
                if parent and hasattr(parent, param_name):
                    line_edit.setText(str(getattr(parent, param_name)))
                line_edit.textChanged.connect(self._on_text_changed)
                
                # Add validator
                validator = QtGui.QDoubleValidator()
                validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
                line_edit.setValidator(validator)
                
                layout.addWidget(line_edit, row_idx + 1, col_idx * 2 + 1)
                self.line_edits[param_name] = line_edit
                
                # Reset button
                reset_btn = QtWidgets.QPushButton("⟳")
                reset_btn.setFixedWidth(30)
                reset_btn.setToolTip(f"Reset {row_label} {headers[col_idx]} to default")
                reset_btn.clicked.connect(lambda checked, p=param_name: self._reset_to_default(p))
                layout.addWidget(reset_btn, row_idx + 1, col_idx * 2 + 2)
                self.reset_btns[param_name] = reset_btn
        
        # Now load defaults after everything is created
        self._load_defaults_from_parent()
    
    def _load_defaults_from_parent(self):
        """Reload default values from parent window"""
        self.default_values = {}
        for row_idx, row_params in enumerate(self.param_names):
            for col_idx, param_name in enumerate(row_params):
                default_name = f"DEFAULT_{param_name}"
                if hasattr(self.parent_window, default_name):
                    self.default_values[param_name] = getattr(self.parent_window, default_name)
    
    def _on_text_changed(self):
        """Called when any text changes - only update if all values are valid"""
        if self._is_updating:
            return
        
        # Check that ALL entries are valid before updating
        all_valid = True
        for param_name, line_edit in self.line_edits.items():
            text = line_edit.text()
            if not text or text.strip() == "":
                all_valid = False
                break
            try:
                value = float(text)
                if not np.isfinite(value):
                    all_valid = False
                    break
            except ValueError:
                all_valid = False
                break
        
        if not all_valid:
            return
        
        # All entries are valid, update them
        try:
            for param_name, line_edit in self.line_edits.items():
                value = float(line_edit.text())
                setattr(self.parent_window, param_name, value)
            
            # Trigger refresh once after updating all
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        except ValueError:
            pass
    
    def _reset_to_default(self, param_name):
        """Reset a specific entry to its default value"""
        if self._is_updating:
            return
        
        self._is_updating = True
        try:
            if param_name in self.default_values:
                default_value = self.default_values[param_name]
                self.line_edits[param_name].setText(str(default_value))
                setattr(self.parent_window, param_name, float(default_value))
                if hasattr(self.parent_window, '_on_param_changed'):
                    self.parent_window._on_param_changed()
        finally:
            self._is_updating = False
    
    def reload_defaults(self):
        """Reload defaults and reset all entries to their defaults"""
        if self._is_updating:
            return
        
        self._is_updating = True
        try:
            self._load_defaults_from_parent()
            for param_name, default_value in self.default_values.items():
                if param_name in self.line_edits:
                    self.line_edits[param_name].setText(str(default_value))
                    setattr(self.parent_window, param_name, float(default_value))
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        finally:
            self._is_updating = False
    
    def get_values(self):
        """Return dictionary of current values"""
        values = {}
        for param_name, line_edit in self.line_edits.items():
            try:
                values[param_name] = float(line_edit.text())
            except ValueError:
                values[param_name] = 0.0
        return values
    
    def set_values(self, values_dict):
        """Set values from dictionary"""
        if self._is_updating:
            return
        
        self._is_updating = True
        try:
            for param_name, value in values_dict.items():
                if param_name in self.line_edits:
                    self.line_edits[param_name].setText(str(value))
        finally:
            self._is_updating = False
                
class MainWindow(QtWidgets.QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PalmSens Data Browser")
        self.resize(1300, 800)
        
        self.current_folder: Optional[Path] = None

        self.current_file_path: Optional[str] = None   # path of the file currently loaded into DATA
        self.DATA = None                                # ps.load_session_file() result for current_file_path
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
        
        self.UI_ONLY_DEFAULTS = {
            'EDIT_COMMENT_WIDTH_FACTOR': 1.3,
            'EDIT_COMMENT_HEADER_SIZE': 14.0,
            'EDIT_COMMENT_BODY_SIZE': 8,
            'EDIT_TOGGLE_STRETCH': 0,
            'EDIT_SHOW_COMMENT': False,  
             #Window Persistance
            'EDIT_WINDOW_WIDTH': 1300,
            'EDIT_WINDOW_HEIGHT': 800,
            'EDIT_WINDOW_X': 0,
            'EDIT_WINDOW_Y': 0,
        }
        
        
        self.current_state: CurveState = CurveState()
        self.raw_xy_list = []       # [(x, y), ...] one per curve in the current measurement, cached on plot
        self.span_selector = None

        self.show_comment = False       # global toggle, not persisted per file
        self.selected_range_index: Optional[int] = None  # row currently highlighted in the range list
        
        # Initialize all EDIT_* variables from UI_CONFIG_MAP
        self._init_session_variables()
        
        # User rcParams overrides - loaded from the dedicated override file
        # (see RCPARAMS_OVERRIDE_PATH), not from session.json. This is the
        # "startup defaults" layer managed by the Save/Load/Reset buttons.
        self._user_rcparams = load_rcparams_override()
        self._sync_edit_vars_from_rcparams(self._user_rcparams)
        self._apply_all_rcparams()
        
        self._build_ui()
        self._restore_folders()
        # Load saved parameters
        self._load_session_parameters()
        # Apply initial figure size after UI is built  # Added
        self._apply_figsize(self.show_comment)  # Added
        self._update_canvas_size_policy()
        self._refresh_plot()  # Added (or call after everything is loaded)
    
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
        # Initialize rcParam-linked variables from UI_RCPCONFIG_MAP
        for edit_name in self.UI_RCPCONFIG_MAP.keys():
            default_value = self.get_default_for_edit(edit_name)
            if default_value is not None:
                # Set both the EDIT_* and DEFAULT_EDIT_* variables
                setattr(self, edit_name, default_value)
                setattr(self, f"DEFAULT_{edit_name}", default_value)
            else:
                setattr(self, edit_name, 0)
                setattr(self, f"DEFAULT_{edit_name}", 0)
        
        # Initialize UI-only variables from UI_ONLY_DEFAULTS
        for edit_name, default_value in self.UI_ONLY_DEFAULTS.items():
            setattr(self, edit_name, default_value)
            setattr(self, f"DEFAULT_{edit_name}", default_value)
        

                
    def _apply_all_rcparams(self):
        """Apply rcParams in order: defaults -> UI -> user overrides"""
        
        # 1. Start with base defaults
        mpl.rcParams.update(self.RCPARAM_DEFAULTS)
        
        # 2. Apply UI overrides (only for EDIT_* that map to rcParams)
        for edit_name, mapping in self.UI_RCPCONFIG_MAP.items():
            if mapping is None:
                continue  # Not an rcParam
            rc_key, index = mapping
            value = getattr(self, edit_name, None)
            if value is None:
                continue
            if index is not None:
                # Handle list items like figure.figsize
                current = mpl.rcParams.get(rc_key, [])
                if isinstance(current, list) and len(current) > index:
                    current[index] = value
                    mpl.rcParams[rc_key] = current
            else:
                mpl.rcParams[rc_key] = value
    
        # 3. Apply user overrides (highest priority)
        if hasattr(self, '_user_rcparams') and self._user_rcparams:
            mpl.rcParams.update(self._user_rcparams)
    def _update_ui_from_config(self):
        """Update all UI elements from EDIT_* variables"""
        # Update EditableParamRow widgets
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
        
        # Update stretch button
        self._update_button_text()       
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
        self.file_list = QtWidgets.QListWidget()
        self.file_list.currentItemChanged.connect(self._on_entry_selected)
        col.addWidget(self.file_list)

        return col

    def _build_middle_column(self):
        col = QtWidgets.QVBoxLayout()
        
        # Create stacked widget to switch between canvas and image
        self.display_stack = QtWidgets.QStackedWidget()
        
        # --- Canvas (for Stretch/Scale modes) ---
        self.fig = Figure(figsize=(self.EDIT_FIGSIZE_X, self.EDIT_FIGSIZE_Y), dpi=self.EDIT_FIGURE_DPI)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # canvas_container centers the canvas when it's given a fixed size
        # (Scale to Fit mode). A QStackedWidget always forces its *page*
        # widget to fill the whole stack area, so letterboxing has to be
        # done one level in, via this wrapper's centered layout - the
        # canvas itself is the thing we fix the size of, the container just
        # provides the centering.
        self.canvas_container = QtWidgets.QWidget()
        canvas_container_layout = QtWidgets.QVBoxLayout(self.canvas_container)
        canvas_container_layout.setContentsMargins(0, 0, 0, 0)
        canvas_container_layout.setAlignment(Qt.AlignCenter)
        canvas_container_layout.addWidget(self.canvas)
        self.display_stack.addWidget(self.canvas_container)  # Index 0
        
        # --- Image label (for Real Size mode) ---
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        self.display_stack.addWidget(self.image_label)  # Index 1
        
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setIconSize(QtCore.QSize(20, 20))
        toolbar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        
        col.addWidget(toolbar)
        col.addWidget(self.display_stack, stretch=1)
        return col
    
    def _build_right_column(self):
        col = QtWidgets.QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QtWidgets.QTabWidget()
        col.addWidget(self.tab_widget)
        
        # Create tabs
        self._build_display_tab()
        self._build_fitting_tab()
        self._build_advanced_tab()
        col.addStretch(1)
        return col

    def _build_fitting_tab(self):
        """Build the Fitting tab with background range controls"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Background ranges
        layout.addWidget(QtWidgets.QLabel("Background ranges"))
        self.range_list = QtWidgets.QListWidget()
        self.range_list.currentRowChanged.connect(self._on_range_row_changed)
        layout.addWidget(self.range_list)
        
        range_btn_row = QtWidgets.QHBoxLayout()
        self.add_range_btn = QtWidgets.QPushButton("+")
        self.add_range_btn.setCheckable(True)
        self.add_range_btn.toggled.connect(self._toggle_select_background)
        remove_range_btn = QtWidgets.QPushButton("-")
        remove_range_btn.clicked.connect(self._remove_range)
        range_btn_row.addWidget(self.add_range_btn)
        range_btn_row.addWidget(remove_range_btn)
        layout.addLayout(range_btn_row)
        
        # Background removal mode
        mode_label = QtWidgets.QLabel("Background Removal Mode:")
        layout.addWidget(mode_label)
        
        self.bg_mode_combo = QtWidgets.QComboBox()
        self.bg_mode_combo.addItems([
            "Linear Fit",
            "Polynomial Fit",
            "Spline Interpolation",
            "Delete Regions",
            "Connect Region Ends"
        ])
        self.bg_mode_combo.currentIndexChanged.connect(self._on_bg_mode_changed)
        layout.addWidget(self.bg_mode_combo)
        
        # Polynomial degree (only for POLY mode)
        poly_layout = QtWidgets.QHBoxLayout()
        poly_layout.addWidget(QtWidgets.QLabel("Polynomial Degree:"))
        self.poly_degree_spin = QtWidgets.QSpinBox()
        self.poly_degree_spin.setRange(1, 10)
        self.poly_degree_spin.setValue(2)
        self.poly_degree_spin.valueChanged.connect(self._on_poly_degree_changed)
        poly_layout.addWidget(self.poly_degree_spin)
        poly_layout.addStretch()
        layout.addLayout(poly_layout)
        
        # Spline smoothing (only for SPLINE mode)
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
        
        self.comment_toggle = QtWidgets.QPushButton("Plot with comment")
        self.comment_toggle.setCheckable(True)
        self.comment_toggle.toggled.connect(self._toggle_comment)
        layout.addWidget(self.comment_toggle)
        
        save_btn = QtWidgets.QPushButton("Save figure")
        save_btn.clicked.connect(self._save_figure)
        layout.addWidget(save_btn)
        
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Fitting")
        
    def _build_advanced_tab(self):
        """Build an Advanced tab for less frequently used settings"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Add any advanced settings here
        # e.g., default folder path, export settings, etc.
        
        layout.addStretch(1)
        self.tab_widget.addTab(tab, "Advanced")
    
    def _build_display_tab(self):
        """Build the Display tab with figure and font settings"""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        
        # FIGURE DISPLAY PARAMETERS
        fig_display_label = QtWidgets.QLabel("Figure Display Parameters")
        fig_display_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(fig_display_label)
        
        # Save parameters
        self.save_width_row = EditableParamRow(
            "Fig Width:", self.EDIT_FIGSIZE_X, "EDIT_FIGSIZE_X", self
        )
        layout.addWidget(self.save_width_row)
        
        self.save_height_row = EditableParamRow(
            "Fig Height:", self.EDIT_FIGSIZE_Y, "EDIT_FIGSIZE_Y", self
        )
        layout.addWidget(self.save_height_row)
        
        # Add DPI row
        self.dpi_row = EditableParamRow(
            "Figure DPI:", self.EDIT_FIGURE_DPI, "EDIT_FIGURE_DPI", self
        )
        layout.addWidget(self.dpi_row)
        
        # Add width factor row
        self.width_factor_row = EditableParamRow(
            "Comment W:", self.EDIT_COMMENT_WIDTH_FACTOR, "EDIT_COMMENT_WIDTH_FACTOR", self
        )
        layout.addWidget(self.width_factor_row)
        
        self.ui_toggle_stretchplot = QtWidgets.QPushButton("Stretch to Fit")
        self.ui_toggle_stretchplot.setCheckable(True)
        self.ui_toggle_stretchplot.clicked.connect(self._cycle_stretch_mode)
        layout.addWidget(self.ui_toggle_stretchplot)
        self._update_button_text()
        
        # Separator
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
            parent=self
        )
        layout.addWidget(self.font_sizes_row)
        
        reset_all_btn = QtWidgets.QPushButton("↺ Reset All Parameters")
        reset_all_btn.clicked.connect(self._reset_all_parameters)
        layout.addWidget(reset_all_btn)
        
        # Separator
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
    #

    # -- Folder management ----------------------------------------------------

    def _restore_folders(self):
        for folder in load_folder_list():
            self.folder_list.addItem(folder)

    def _add_folder(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Add data folder")
        if not directory:
            return
        existing = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        if directory not in existing:
            self.folder_list.addItem(directory)
            save_folder_list(existing + [directory])

    def _delete_folder(self):
        item = self.folder_list.currentItem()
        if item is None:
            return
        row = self.folder_list.row(item)
        self.folder_list.takeItem(row)
        remaining = [self.folder_list.item(i).text() for i in range(self.folder_list.count())]
        save_folder_list(remaining)
        self.file_list.clear()
        self._reset_loaded_data()

    def _reset_loaded_data(self):
        self.current_file_path = None
        self.DATA = None
        self.current_file_index = None
        self.current_meas_index = None
        self.current_filename = None
        self.fig.clear()
        self.canvas.draw()

    # -- Measurement list (grouped under grey filename headers) ---------------

    def _on_folder_selected(self, item, _prev=None):
        self.file_list.clear()
        self._reset_loaded_data()
        if item is None:
            self.current_folder = None
            return
        self.current_folder = Path(item.text())

        files = sorted(self.current_folder.glob(PSSESSION_GLOB))
        for file_index, f in enumerate(files):
            header = QtWidgets.QListWidgetItem(f.name)
            header.setFlags(QtCore.Qt.NoItemFlags)
            header.setBackground(HEADER_BG)
            self.file_list.addItem(header)

            try:
                data = load_session_file(f)
            except Exception as exc:
                err_item = QtWidgets.QListWidgetItem(f"    (failed to load: {exc})")
                err_item.setFlags(QtCore.Qt.NoItemFlags)
                self.file_list.addItem(err_item)
                continue

            for meas_index, dat in enumerate(data):
                curve0 = dat.curves[0] if getattr(dat, "n_curves", 0) else None
                curve0_name = curve_display_name(curve0, 0) if curve0 is not None else "no curves"
                entry = QtWidgets.QListWidgetItem(f"    {meas_index}: {curve0_name}")
                entry.setData(QtCore.Qt.UserRole, (str(f), file_index, meas_index))
                self.file_list.addItem(entry)

            del data  # only the file you click into gets kept in memory

    def _on_entry_selected(self, item, _prev=None):
        if item is None:
            return
        payload = item.data(QtCore.Qt.UserRole)
        if payload is None:
            return  # header row - not selectable, but guard anyway
        file_path, file_index, meas_index = payload
    
        if self.current_file_path != file_path:
            self.DATA = load_session_file(Path(file_path))
            self.current_file_path = file_path
    
        self.current_file_index = file_index
        self.current_meas_index = meas_index
        self.current_filename = Path(file_path).name
    
        store = StateStore(self.current_folder)
        key = state_key(self.current_filename, meas_index)
        self.current_state = store.load(key)
    
        # Update UI to reflect loaded state
        self.remove_bg_toggle.blockSignals(True)
        self.remove_bg_toggle.setChecked(self.current_state.remove_background)
        self.remove_bg_toggle.blockSignals(False)
        
        # Update background removal mode combo box
        self.bg_mode_combo.blockSignals(True)
        self.bg_mode_combo.setCurrentIndex(self.current_state.bg_removal_mode)
        self.bg_mode_combo.blockSignals(False)
        
        # Update polynomial degree spin box
        self.poly_degree_spin.blockSignals(True)
        self.poly_degree_spin.setValue(self.current_state.poly_degree)
        self.poly_degree_spin.blockSignals(False)
        
        # Update spline smoothing spin box
        self.spline_smoothing_spin.blockSignals(True)
        self.spline_smoothing_spin.setValue(self.current_state.spline_smoothing)
        self.spline_smoothing_spin.blockSignals(False)
        
        # Show/hide controls based on mode
        self.poly_degree_spin.setVisible(self.current_state.bg_removal_mode == BgRemovalMode.POLY)
        self.spline_smoothing_spin.setVisible(self.current_state.bg_removal_mode == BgRemovalMode.SPLINE)
    
        self.selected_range_index = None
        self._refresh_range_list()
        self._refresh_plot()
        

    def _save_state(self):
        if self.current_folder and self.current_filename and self.current_meas_index is not None:
            key = state_key(self.current_filename, self.current_meas_index)
            StateStore(self.current_folder).save(key, self.current_state)

    def _refresh_range_list(self):
        self.range_list.blockSignals(True)
        self.range_list.clear()
        for i, (xmin, xmax) in enumerate(self.current_state.bg_ranges):
            self.range_list.addItem(f"{i}: [{xmin:.4g}, {xmax:.4g}]")
        self.range_list.blockSignals(False)
    
    # -- Plotting -------------------------------------------------------------
    def _render_figure_to_pixmap(self):
        """Render the current figure to a QPixmap at the specified DPI"""
        
        # CRITICAL: Ensure figure is properly sized for export
        fig_w = self.EDIT_FIGSIZE_X
        fig_h = self.EDIT_FIGSIZE_Y
        if self.show_comment:
            fig_w = fig_w * self.EDIT_COMMENT_WIDTH_FACTOR
        
        self.fig.set_size_inches(fig_w, fig_h)
        self.fig.set_dpi(self.EDIT_FIGURE_DPI)
        
        # Apply tight_layout for the export
        if self.show_comment:
            self.fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.08)
        else:
            self.fig.tight_layout(pad=0.8)
        
        # Render at full resolution
        width = int(fig_w * self.EDIT_FIGURE_DPI)
        height = int(fig_h * self.EDIT_FIGURE_DPI)
        
        # Use the canvas's renderer for proper rendering
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        self.fig.canvas.render(painter)
        painter.end()
        
        return pixmap

    def _update_axes_frame(self, ax, mode):
        """Update axes frame appearance based on mode"""
        if mode == 2:  # Real Size
            # Make frame visible and prominent
            ax.set_frame_on(True)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_color('red')
            # Add tick marks to show real size
            ax.tick_params(axis='both', which='major', labelsize=10)
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
        
        # Clear any stale image or canvas artifacts
        if self.EDIT_TOGGLE_STRETCH == 2:
            # Switch to Real Size mode - disable selector if active
            if self.span_selector is not None:
                self.span_selector.disconnect_events()
                self.span_selector = None
                self.add_range_btn.setChecked(False)
            self.image_label.clear()
        else:
            # Switch to Stretch/Scale mode
            self.image_label.clear()
            if self.EDIT_TOGGLE_STRETCH == 1:
                self._apply_scale_to_fit_canvas_size()
        
        self._update_canvas_size_policy()
        self._refresh_plot()
        self._save_session_parameters()  # Save the mode state
    
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
        # Optional: change button color or style to indicate active state
        if self.EDIT_TOGGLE_STRETCH == 0:
            self.ui_toggle_stretchplot.setStyleSheet("background-color: #4CAF50; color: white;")  # Green
        elif self.EDIT_TOGGLE_STRETCH == 1:
            self.ui_toggle_stretchplot.setStyleSheet("background-color: #FF9800; color: white;")  # Orange
        else:
            self.ui_toggle_stretchplot.setStyleSheet("background-color: #f44336; color: white;")  # Red
            
    def _toggle_comment(self, checked):
        self.show_comment = checked
        self._refresh_plot()
        #Save comment state to preserve it between sessions
        self._save_session_parameters()  
    
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
        """Resize the canvas widget so it preserves the figure's own aspect ratio"""
        if not hasattr(self, "display_stack"):
            return
        container_size = self.display_stack.size()
        avail_w = max(container_size.width() - 20, 1)
        avail_h = max(container_size.height() - 20, 1)
    
        fig_w = self.EDIT_FIGSIZE_X
        fig_h = self.EDIT_FIGSIZE_Y
        if self.show_comment:
            fig_w = fig_w * self.EDIT_COMMENT_WIDTH_FACTOR
        
        # Get the actual figure size in pixels at current DPI
        dpi = self.EDIT_FIGURE_DPI
        actual_w = fig_w * dpi
        actual_h = fig_h * dpi
        
        # Calculate scaling to fit within container
        scale_w = avail_w / actual_w if actual_w > 0 else 1
        scale_h = avail_h / actual_h if actual_h > 0 else 1
        scale = min(scale_w, scale_h, 1.0)  # Don't scale up beyond 1x
        
        new_w = int(actual_w * scale)
        new_h = int(actual_h * scale)
        
        self.canvas.setFixedSize(max(new_w, 1), max(new_h, 1))
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
        """Update canvas size policy based on current mode"""
        if self.EDIT_TOGGLE_STRETCH == 0:
            # Stretch to Fit - canvas fills the available area, no letterboxing
            self.canvas.setMinimumSize(0, 0)
            self.canvas.setMaximumSize(16777215, 16777215)  # Qt's QWIDGETSIZE_MAX
            self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        elif self.EDIT_TOGGLE_STRETCH == 1:
            # Scale to Fit - fixed size, aspect-locked, centered by canvas_container
            self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            self._apply_scale_to_fit_canvas_size()
        # mode 2 (Real Size) doesn't use the live canvas at all - the
        # rasterized preview goes through image_label instead.
        
        # Force layout update
        self.canvas.updateGeometry()
    
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
            
    def resizeEvent(self, event):
        """Handle window resize with proper mode handling"""
        super().resizeEvent(event)
        
        if self.EDIT_TOGGLE_STRETCH == 1:
            # Scale to Fit: update letterboxed canvas size
            self._apply_scale_to_fit_canvas_size()
            self.canvas.draw()
        elif self.EDIT_TOGGLE_STRETCH == 2:
            # Real Size: re-render the image at new size
            self._refresh_plot()
    def _remove_background_from_curve(self, x, y):
        """Apply background removal based on current mode"""
        if not self.current_state.bg_ranges:
            return y
        
        ranges = self.current_state.bg_ranges
        mode = self.current_state.bg_removal_mode
        
        # Create a mask for points inside background ranges
        in_range_mask = np.zeros_like(x, dtype=bool)
        for xmin, xmax in ranges:
            in_range_mask |= (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
        
        if mode == BgRemovalMode.REGION_DELETE:
            # Simply delete data in ranges - return NaN for those points
            y_cleaned = y.copy()
            y_cleaned[in_range_mask] = np.nan
            return y_cleaned
        
        elif mode == BgRemovalMode.CONNECT_END:
            # Connect end of one range to start of next with straight line
            y_cleaned = y.copy()
            
            # Get points just outside each range
            for i, (xmin, xmax) in enumerate(ranges):
                # Find indices just outside the range
                idx_before = np.where(x < xmin)[0]
                idx_after = np.where(x > xmax)[0]
                
                if len(idx_before) > 0 and len(idx_after) > 0:
                    # Get points just outside
                    x_before = x[idx_before[-1]]
                    y_before = y[idx_before[-1]]
                    x_after = x[idx_after[0]]
                    y_after = y[idx_after[0]]
                    
                    # Linear interpolation between these points
                    if x_after > x_before:
                        # Replace the range with linear interpolation
                        mask_range = (x >= xmin) & (x <= xmax)
                        x_range = x[mask_range]
                        y_range = y_before + (y_after - y_before) * (x_range - x_before) / (x_after - x_before)
                        y_cleaned[mask_range] = y_range
            
            return y_cleaned
        
        elif mode == BgRemovalMode.POLY:
            # Polynomial fit through selected ranges
            poly_deg = self.current_state.poly_degree
            
            # Collect points from all background ranges
            x_bg = []
            y_bg = []
            for xmin, xmax in ranges:
                mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                x_bg.extend(x[mask])
                y_bg.extend(y[mask])
            
            if len(x_bg) >= poly_deg + 1:
                # Fit polynomial
                coeffs = np.polyfit(x_bg, y_bg, poly_deg)
                # Subtract the polynomial from the entire curve
                y_fit = np.polyval(coeffs, x)
                return y - y_fit
            else:
                return y  # Not enough points for fit
        
        elif mode == BgRemovalMode.SPLINE:
            # Spline interpolation through selected points
            try:
                from scipy.interpolate import splrep, splev
            except ImportError:
                # Fallback to linear if scipy not available
                return y
            
            # Collect points from all background ranges
            x_bg = []
            y_bg = []
            for xmin, xmax in ranges:
                mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                x_bg.extend(x[mask])
                y_bg.extend(y[mask])
            
            if len(x_bg) >= 4:  # Need at least 4 points for spline
                # Sort by x
                idx_sorted = np.argsort(x_bg)
                x_bg = np.array(x_bg)[idx_sorted]
                y_bg = np.array(y_bg)[idx_sorted]
                
                # Create spline
                smoothing = self.current_state.spline_smoothing or None
                tck = splrep(x_bg, y_bg, s=smoothing)
                y_fit = splev(x, tck)
                return y - y_fit
            else:
                return y
        
        else:  # LINEAR mode (default)
            # Linear fit through selected ranges
            x_bg = []
            y_bg = []
            for xmin, xmax in ranges:
                mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                x_bg.extend(x[mask])
                y_bg.extend(y[mask])
            
            if len(x_bg) >= 2:
                coeffs = np.polyfit(x_bg, y_bg, 1)
                y_fit = np.polyval(coeffs, x)
                return y - y_fit
            else:
                return y    
            
    def _refresh_plot(self):
        # Drop any stale SpanSelector before tearing down its axes - leaving
        # one connected across a fig.clear() is what caused the leftover
        # selection artifacts on redraw.
        if self.span_selector is not None:
            try:
                self.span_selector.disconnect_events()
            except Exception:
                pass
            self.span_selector = None
    
        self.fig.clear()
    
        if self.DATA is None or self.current_meas_index is None:
            self.canvas.draw()
            if self.EDIT_TOGGLE_STRETCH == 2:
                self.image_label.clear()
            return
    
        dat = self.DATA[self.current_meas_index]
        curves = dat.curves if getattr(dat, "n_curves", 0) else []
    
        self.raw_xy_list = [curve_to_xy(c) for c in curves]
    
        # ALWAYS apply figsize for rendering (the image needs the correct size,
        # and this is also the size Real Size / Scale to Fit letterbox against)
        fig_width = self.EDIT_FIGSIZE_X
        fig_height = self.EDIT_FIGSIZE_Y
        if self.show_comment:
            fig_width = fig_width * self.EDIT_COMMENT_WIDTH_FACTOR
        self.fig.set_size_inches(fig_width, fig_height)
        self.fig.set_dpi(self.EDIT_FIGURE_DPI)
    
        # Build axes
        if self.show_comment:
            ax = self._build_annotated_axes()
        else:
            ax = self.fig.add_subplot(111)
    
        fit_coeffs = self.current_state.fit_coeffs or [None] * len(curves)
        
        # Store the plotted data for bounds calculation
        plotted_x = []
        plotted_y = []
        
        for idx, (curve, (x, y)) in enumerate(zip(curves, self.raw_xy_list)):
            y_plot = y
            fit_line = None
            
            if self.current_state.remove_background:
                # Apply background removal based on mode
                if self.current_state.bg_removal_mode in [BgRemovalMode.REGION_DELETE, BgRemovalMode.CONNECT_END]:
                    # These modes modify the data directly
                    y_plot = self._remove_background_from_curve(x, y)
                else:
                    # These modes use coefficients or compute on the fly
                    coeffs = fit_coeffs[idx] if idx < len(fit_coeffs) else None
                    if coeffs is not None:
                        fit_line = np.polyval(coeffs, x)
                        y_plot = y - fit_line
                    elif self.current_state.bg_removal_mode == BgRemovalMode.SPLINE:
                        # Spline interpolation
                        try:
                            from scipy.interpolate import splrep, splev
                            # Collect background points
                            x_bg = []
                            y_bg = []
                            for xmin, xmax in self.current_state.bg_ranges:
                                mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                                x_bg.extend(x[mask])
                                y_bg.extend(y[mask])
                            if len(x_bg) >= 4:
                                # Sort by x
                                idx_sorted = np.argsort(x_bg)
                                x_bg = np.array(x_bg)[idx_sorted]
                                y_bg = np.array(y_bg)[idx_sorted]
                                smoothing = self.current_state.spline_smoothing or None
                                tck = splrep(x_bg, y_bg, s=smoothing)
                                y_fit = splev(x, tck)
                                y_plot = y - y_fit
                        except ImportError:
                            # Fallback to linear if scipy not available
                            pass
            
            ax.plot(x, y_plot, label=curve_display_name(curve, idx))
            if fit_line is not None and not self.current_state.remove_background:
                ax.plot(x, fit_line, "--", color="grey", alpha=0.6)
            
            # Store the plotted data for bounds
            plotted_x.append(x)
            plotted_y.append(y_plot)
    
        for i, (xmin, xmax) in enumerate(self.current_state.bg_ranges):
            if i == self.selected_range_index:
                ax.axvspan(xmin, xmax, color="green", alpha=0.35)
            else:
                ax.axvspan(xmin, xmax, color="orange", alpha=0.2)
    
        ax.set_title(getattr(dat, "title", self.current_filename), fontweight="bold")
        ax.legend(loc="best")
        ax.annotate(
            f"fileID: {self.current_file_index}  measurementID: {self.current_meas_index}",
            xy=(0.1, 0.9), xycoords="axes fraction",
        )
    
        if self.show_comment:
            self._draw_comment_panel(dat, ax)
    
        # Only enable SpanSelector in Stretch/Scale modes (the underlying
        # data coordinates in Real Size mode are only ever seen via the
        # rasterized preview, so span-selection there wouldn't map to
        # anything meaningful)
        if self.add_range_btn.isChecked() and self.EDIT_TOGGLE_STRETCH != 2:
            self._enable_span_selector(ax)
    
        # Data bounds - use the PLOTTED data, not the raw data
        if len(plotted_x) > 0 and len(plotted_y) > 0:
            all_x = np.concatenate(plotted_x)
            all_y = np.concatenate(plotted_y)
            x_min, x_max = all_x.min(), all_x.max()
            y_min, y_max = all_y.min(), all_y.max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            # Add a small padding (5%) for better visualization
            x_pad = x_range * 0.05 if x_range > 0 else 1
            y_pad = y_range * 0.05 if y_range > 0 else 1
            x_limits = (x_min - x_pad, x_max + x_pad)
            y_limits = (y_min - y_pad, y_max + y_pad)
        else:
            x_limits = (0, 1)
            y_limits = (0, 1)
    
        # The data itself is always shown "auto" (no aspect locking to data
        # units) - what differs between the three modes is only how the
        # *figure as a whole* gets placed on screen afterwards.
        ax.set_aspect('auto')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_frame_on(True)
        # Add a pale grey dotted x-axis (y=0) if the lower bound goes below zero
        if y_limits[0] < 0:
            ax.axhline(y=0, color='grey', linestyle=':', linewidth=1, alpha=0.7, zorder=1)
            
        # --- Render/Display based on mode ---
        if self.EDIT_TOGGLE_STRETCH == 2:
            # Real Size: render and display as image with NEAREST NEIGHBOR scaling
            self.fig.subplots_adjust(left=0.1, right=0.98, top=0.98, bottom=0.08)
            pixmap = self._render_figure_to_pixmap()
            label_size = self.image_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                # Use FastTransformation for nearest neighbor (no interpolation)
                scaled_pixmap = pixmap.scaled(
                    label_size.width() - 10,
                    label_size.height() - 10,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation  # Nearest neighbor - no interpolation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setPixmap(pixmap)
            self.canvas.draw()
        else:
            # Stretch/Scale modes: interactive canvas. Scale to Fit keeps its
            # letterboxed fixed size in sync 
            if self.EDIT_TOGGLE_STRETCH == 1:
                self._apply_scale_to_fit_canvas_size()
            self.canvas.draw()
            self.image_label.clear()
            
            if self.EDIT_TOGGLE_STRETCH == 0:
                # Force canvas update (nudge to clear stale Stretch-to-Fit repaint artifacts)
                self.canvas.updateGeometry()
                current_size = self.canvas.size()
                self.canvas.resize(current_size.width() + 1, current_size.height())
                self.canvas.resize(current_size.width(), current_size.height())


    def _build_annotated_axes(self):
        """Layout port of plot_psense(): main axes squished to the left,
        a lightcyan comment panel occupying the widened right portion."""
        original_right = 1.0 / self.EDIT_COMMENT_WIDTH_FACTOR
        o_r_mod = original_right - 0.125
        self._plot_pos = [0.125, 0.1, o_r_mod, 0.825]
        self._text_pos = [original_right, 0.1, 1.0 - original_right - 0.005, 0.825]
        return self.fig.add_axes(self._plot_pos)

    def _draw_comment_panel(self, dat, ax):
        comment = getattr(getattr(dat, "_psmeasurement", None), "Method", None)
        comment_text = getattr(comment, "Notes", "") if comment else ""
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
            handle.set_linewidth(2)
            handle.set_markersize(8)
        legend = ax.legend(handles, labels, prop={"size": 10}, handlelength=0.9)
        ax_bottom = ax.get_position().y0
        ax_right = ax.get_position().x1
        legend.set_bbox_to_anchor((ax_right, ax_bottom), transform=self.fig.transFigure)
        legend._loc = 3  # lower left

    # -- Background selection / fitting ---------------------------------------

    def _toggle_select_background(self, checked):
        if self.EDIT_TOGGLE_STRETCH == 2:
            QtWidgets.QMessageBox.information(
                self, "Not available", 
                "Background selection is not available in Real Size mode. Switch to Stretch or Scale mode."
            )
            self.add_range_btn.setChecked(False)
            return
        
        if checked:
            axes = self.fig.axes
            if axes:
                self._enable_span_selector(axes[0])
        else:
            if self.span_selector is not None:
                self.span_selector.disconnect_events()
                self.span_selector = None
            self._refresh_plot()

    def _enable_span_selector(self, ax):
        self.span_selector = SpanSelector(
            ax, self._on_span_select, "horizontal",
            useblit=True, props=dict(alpha=0.2, facecolor="orange"),
        )

    def _on_span_select(self, xmin, xmax):
        self.current_state.bg_ranges.append([float(xmin), float(xmax)])
        self._save_state()
        self._refresh_range_list()
        self._refresh_plot()

    def _on_range_row_changed(self, row):
        self.selected_range_index = row if row >= 0 else None
        self._refresh_plot()

    def _remove_range(self):
        if not self.current_state.bg_ranges:
            return
        if self.selected_range_index is not None and 0 <= self.selected_range_index < len(self.current_state.bg_ranges):
            idx = self.selected_range_index
        else:
            idx = len(self.current_state.bg_ranges) - 1  # bottom of the list, for rapid-tap clear-all
        del self.current_state.bg_ranges[idx]
        self.selected_range_index = None
        self._save_state()
        self._refresh_range_list()
        self._refresh_plot()

    def _fit_background(self):
        """Fit background using selected mode"""
        if not self.raw_xy_list or not self.current_state.bg_ranges:
            QtWidgets.QMessageBox.information(
                self, "No selection", "Select at least one background range first."
            )
            return
    
        mode = self.current_state.bg_removal_mode
        new_coeffs = []
        fitted_any = False
        
        for x, y in self.raw_xy_list:
            # Collect points from all background ranges
            x_bg = []
            y_bg = []
            for xmin, xmax in self.current_state.bg_ranges:
                mask = (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
                x_bg.extend(x[mask])
                y_bg.extend(y[mask])
            
            if mode == BgRemovalMode.LINEAR:
                # Linear fit
                if len(x_bg) >= 2:
                    coeffs = np.polyfit(x_bg, y_bg, 1).tolist()
                    new_coeffs.append(coeffs)
                    fitted_any = True
                else:
                    new_coeffs.append(None)
            
            elif mode == BgRemovalMode.POLY:
                # Polynomial fit
                degree = self.current_state.poly_degree
                if len(x_bg) >= degree + 1:
                    coeffs = np.polyfit(x_bg, y_bg, degree).tolist()
                    new_coeffs.append(coeffs)
                    fitted_any = True
                else:
                    new_coeffs.append(None)
            
            elif mode == BgRemovalMode.SPLINE:
                # Spline - store coefficients as None since we'll compute on the fly
                new_coeffs.append(None)
                fitted_any = True
            
            elif mode == BgRemovalMode.REGION_DELETE:
                # Delete regions - no coefficients needed
                new_coeffs.append(None)
                fitted_any = True
            
            elif mode == BgRemovalMode.CONNECT_END:
                # Connect ends - no coefficients needed
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
        """Update matplotlib rcParams from UI controls. 
        Note that if you then load an rc_param dict from the advanced tab, that this will update the UI. This is how we can save plotting defaults."""
    
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
        
        # Update UI elements - this should now work properly
        self._update_ui_from_config()
        
        # Only apply figsize if NOT Real Size mode
        if self.EDIT_TOGGLE_STRETCH != 2:
            self._apply_figsize(self.show_comment)
        
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
        
    def _load_session_parameters(self):
        """Load saved parameter values from the session config"""
        SESSION_PATH = Path.home() / ".config" / "palmsens_gui" / "session.json"
        if SESSION_PATH.exists():
            try:
                with open(SESSION_PATH) as f:
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
                
                # Load show_comment state
                if 'show_comment' in ui_config:
                    self.show_comment = ui_config['show_comment']
                    # Update the toggle button
                    if hasattr(self, 'comment_toggle'):
                        self.comment_toggle.blockSignals(True)
                        self.comment_toggle.setChecked(self.show_comment)
                        self.comment_toggle.blockSignals(False)
                
                # Restore window state
                if hasattr(self, 'EDIT_WINDOW_X'):
                    self._restore_window_state()
                
                # NOTE: rcParam overrides are read from session.json which lives in their own dedicated file (RCPARAMS_OVERRIDE_PATH, see load_rcparams_override()).
                # Save/Load/Reset in the Display tab has one unambiguous source of truth. self._user_rcparams is already populated from that file in __init__.
                
                # Update UI elements
                self._update_ui_from_config()
                
                # Apply rcParams after loading, then make sure the EDIT_*
                # fields reflect whatever the (higher-priority) rcParam
                # overrides actually set, so the displayed values are honest.
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
        ui_config['show_comment'] = self.show_comment
        
        # Build rcParams config (only user-modified rcParams that differ from defaults)
        rc_defaults = self.RCPARAM_DEFAULTS
        current_rc = {k: v for k, v in mpl.rcParams.items() 
                      if k in rc_defaults and v != rc_defaults.get(k)}
        
        session_data = {
            'ui_config': ui_config,
            'current_rcparams': current_rc,
        }
        
        SESSION_PATH = Path.home() / ".config" / "palmsens_gui" / "session.json"
        SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SESSION_PATH, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

    # -- rcParam override file handling (Save / Load / Reset buttons) --------

    def _sync_edit_vars_from_rcparams(self, rc_dict, fallback_to_defaults=False):
        """Keep the EDIT_* UI variables consistent with a given rcParams
        dict, for every EDIT_* that maps to an rcParam key present in it.
        With fallback_to_defaults=True, any mapped EDIT_* not present in
        rc_dict is pulled from RCPARAM_DEFAULTS instead (used by Reset)."""
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
        """Return the subset of rcParams (matching RCPARAM_DEFAULTS keys)
        whose current mpl.rcParams value differs from the code default."""
        diffs = {}
        for key, default_val in self.RCPARAM_DEFAULTS.items():
            current_val = mpl.rcParams.get(key, default_val)
            if current_val != default_val:
                diffs[key] = current_val
        return diffs

    def _save_rcparam_overrides(self):
        """Persist the current rcParams (on top of the code defaults) to
        disk, so this becomes the new baseline that loads automatically on
        every future launch."""
        self._apply_all_rcparams()  # make sure EDIT_* edits are reflected in mpl.rcParams first
        diffs = self._get_current_rcparam_diffs()
        save_rcparams_override(diffs)
        self._user_rcparams = diffs
        QtWidgets.QMessageBox.information(
            self, "Saved", f"Saved {len(diffs)} rcParam override(s) as the new startup defaults."
        )

    def _load_rcparam_overrides(self):
        """Reload the saved override file and apply it to the current
        session (does not touch the file itself)."""
        self._user_rcparams = load_rcparams_override()
        self._sync_edit_vars_from_rcparams(self._user_rcparams)
        self._apply_all_rcparams()
        self._update_ui_from_config()
        if self.EDIT_TOGGLE_STRETCH != 2:
            self._apply_figsize(self.show_comment)
        self._refresh_plot()
        self._save_session_parameters()

    def _reset_rcparam_overrides(self):
        """Discard saved overrides and go back to the defaults hardcoded in
        get_rcparam_defaults(), deleting the override file."""
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
            self._apply_figsize(self.show_comment)
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
    titlefont = 15
    bigfont = 14
    mediumfont = 12
    defaultsDict = {
                    'axes.formatter.use_mathtext': True,
                    'text.usetex': False,
                    'font.family': fontfamily,
                    
                    # Figure Dimensions 4:3 default
                    'figure.figsize': [6, 4.5],      
                    'figure.dpi': 200,               
                    
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
                    'lines.markersize': 5,
                    'axes.linewidth': 1.0,           # Standard edge thickness
                    
                    # 4. Clean Tick Dimensions and Padding
                    'xtick.major.size': 4,           # Clean standard major tick length
                    'xtick.minor.size': 2.5,         # Clean standard minor tick length
                    'xtick.major.width': 1.2,        # Clean standard major tick width
                    'xtick.minor.width': 0.8,        # Clean standard minor tick width
                    'xtick.major.pad': 4,            # Clean standard pad spacing
                    'xtick.minor.pad': 4,
                    'xtick.minor.visible': True,
                    
                    'ytick.major.size': 4,
                    'ytick.minor.size': 2.5,
                    'ytick.major.width': 1.2,
                    'ytick.minor.width': 0.8,
                    'ytick.major.pad': 4,
                    'ytick.minor.pad': 4,
                    'ytick.minor.visible': True,
                    
                    # 5. Legend Structural Padding (Kept intact)
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