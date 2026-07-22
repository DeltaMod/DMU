"""
PalmSens data browser / background-fitting applet.

Three-column PyQt5 layout:
  Left   - data folder list (add/delete); below it, a flat list of every file's
           measurements, grouped under an unselectable grey filename header,
           each row labelled "<measurementID>: <first curve name>"
  Middle - matplotlib canvas + toolbar
  Right  - background range list (+/- to add/remove, click a row to highlight
           it green), fit, remove-background toggle, annotated/raw plot
           toggle (global, not per-file), save figure

Data model (matches your plotting loop)
----------------------------------------
    for i, file in enumerate(data_files):
        DATA = ps.load_session_file(file)
        for j, meas in enumerate(DATA):
            plot_psense(DATA, measurementID=j, curveID="all")

Clicking a measurement row loads the *whole* file via ps.load_session_file()
only if it isn't already the loaded file; if you're clicking another
measurement within the same file, the cached DATA is reused. Only one file's
DATA is ever held in memory at a time.

Design notes
------------
* Nothing here mutates the raw measurement data. All operator state (background
  ranges, fit coefficients, toggle states) lives in a small JSON sidecar per
  (file, measurementID) pair, in a ".psense_gui_state" folder next to the data
  files. Deleting that folder resets everything back to raw.
* curveID is always "all", per your loop - every curve in the measurement is
  plotted and independently available for background fitting.
* Curve x/y arrays are pulled off the Line2D object that `curve.plot(ax=ax)`
  creates, drawn on a disposable scratch axes so it never touches the real plot.
* Annotated ("with comment") layout is a direct port of your plot_psense()
  panel: squished main axes + lightcyan comment panel, dynamic text wrap,
  restyled legend, plus the "fileID / measurementID" annotation you add
  after calling plot_psense(). Toggling it off gives the main axes full width.
* On-screen display size and exported-figure size/DPI are independent -
  export always uses SAVE_WIDTH_IN / SAVE_HEIGHT_IN / SAVE_DPI below.

TODO before running:
* `curve_display_name()` guesses which curve attribute holds its display
  name (tries "title", "name", "label", "array_type"). Confirm/replace with
  whatever attribute your curve objects actually expose.
* Comment/device text pulls dat._psmeasurement.Method.Notes and dat.timestamp,
  as in your function - adjust if your SDK version differs.

Requires: PyQt5, matplotlib, numpy, pypalmsens
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
from PyQt5.QtCore import Qt
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
# Main window
# ---------------------------------------------------------------------------

HEADER_BG = QtGui.QColor("#c8c8c8")

class EditableParamRow(QtWidgets.QWidget):
    def __init__(self, label, default_value, param_name, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.default_value = default_value
        self.parent_window = parent
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Label
        self.label = QtWidgets.QLabel(label)
        self.label.setFixedWidth(80)
        layout.addWidget(self.label)
        
        # Editable field
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setText(str(default_value))
        self.line_edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.line_edit)
        
        # Reset button
        self.reset_btn = QtWidgets.QPushButton("⟳")
        self.reset_btn.setFixedWidth(30)
        self.reset_btn.setToolTip("Reset to default")
        self.reset_btn.clicked.connect(self._reset_to_default)
        layout.addWidget(self.reset_btn)
    
    def _on_text_changed(self, text):
        try:
            # Try to parse as float
            value = float(text)
            # Update the parent's attribute
            setattr(self.parent_window, self.param_name, value)
            # Trigger a refresh if needed
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        except ValueError:
            # Invalid input - you might want to highlight the field
            pass
    
    def _reset_to_default(self):
        self.line_edit.setText(str(self.default_value))
        # Ensure the value is updated
        setattr(self.parent_window, self.param_name, float(self.default_value))
        if hasattr(self.parent_window, '_on_param_changed'):
            self.parent_window._on_param_changed()
    
    def get_value(self):
        try:
            return float(self.line_edit.text())
        except ValueError:
            return self.default_value

class MainWindow(QtWidgets.QMainWindow):
    #Defaults:
    BASE_FIGSIZE = (7.0, 6.0)
    COMMENT_WIDTH_FACTOR = 1.3
    FIGURE_DPI = 200

    SAVE_WIDTH_IN = 10.0
    SAVE_HEIGHT_IN = 6.0
    SAVE_DPI = 300
    
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
        
        #plotting parameters:
        
        self.EDIT_FIGSIZE_X = 10.0
        self.EDIT_FIGSIZE_Y = 6.0 
        self.EDIT_COMMENT_WIDTH_FACTOR = 1.3
        self.EDIT_FIGURE_DPI = 200
        self.EDIT_TOGGLE_STRETCH = False
        
        self.current_state: CurveState = CurveState()
        self.raw_xy_list = []       # [(x, y), ...] one per curve in the current measurement, cached on plot
        self.span_selector = None

        self.show_comment = False       # global toggle, not persisted per file
        self.selected_range_index: Optional[int] = None  # row currently highlighted in the range list

        self._build_ui()
        self._restore_folders()
        # Load saved parameters
        self._load_session_parameters()
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
        self.fig = Figure(figsize=self.BASE_FIGSIZE, dpi=self.FIGURE_DPI)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.setIconSize(QtCore.QSize(20, 20))
        toolbar.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        col.addWidget(toolbar)
        col.addWidget(self.canvas)
        return col

    def _build_right_column(self):
        col = QtWidgets.QVBoxLayout()
        
        # FIGURE DISPLAY PARAMETERS
        fig_display_label = QtWidgets.QLabel("Figure Display Parameters")
        fig_display_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        col.addWidget(fig_display_label)
        
        self.ui_toggle_stretchplot = QtWidgets.QPushButton("Fill Axes")
        self.ui_toggle_stretchplot.setCheckable(True)
        self.ui_toggle_stretchplot.toggled.connect(self._toggle_stretchplot)
        col.addWidget(self.ui_toggle_stretchplot)
        
        # Editable parameters
        self.figsize_row = EditableParamRow(
            "Fig size:", self.BASE_FIGSIZE[0], "EDIT_FIGSIZE", self
        )
        col.addWidget(self.figsize_row)
        
        # Add width factor row
        self.width_factor_row = EditableParamRow(
            "Width factor:", self.COMMENT_WIDTH_FACTOR, "EDIT_COMMENT_WIDTH_FACTOR", self
        )
        col.addWidget(self.width_factor_row)
        
        # Add DPI row
        self.dpi_row = EditableParamRow(
            "Figure DPI:", self.FIGURE_DPI, "EDIT_FIGURE_DPI", self
        )
        col.addWidget(self.dpi_row)
        
        # Separator
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        col.addWidget(line)
        
        # Save parameters label
        save_label = QtWidgets.QLabel("Save Parameters")
        save_label.setStyleSheet("font-weight: bold; margin-top: 5px;")
        col.addWidget(save_label)
        
        # Save parameters
        self.save_width_row = EditableParamRow(
            "Save W:", self.SAVE_WIDTH_IN, "EDIT_FIGSIZE_X", self
        )
        col.addWidget(self.save_width_row)
        
        self.save_height_row = EditableParamRow(
            "Save H:", self.SAVE_HEIGHT_IN, "EDIT_FIGSIZE_Y", self
        )
        col.addWidget(self.save_height_row)
        
        self.save_dpi_row = EditableParamRow(
            "Save DPI:", self.EDIT_FIGURE_DPI, "EDIT_FIGURE_DPI", self
        )
        col.addWidget(self.save_dpi_row)
        
        # Separator
        line2 = QtWidgets.QFrame()
        line2.setFrameShape(QtWidgets.QFrame.HLine)
        col.addWidget(line2)
        
        # Original UI elements
        col.addWidget(QtWidgets.QLabel("Background ranges"))
        self.range_list = QtWidgets.QListWidget()
        self.range_list.currentRowChanged.connect(self._on_range_row_changed)
        col.addWidget(self.range_list)
        
        range_btn_row = QtWidgets.QHBoxLayout()
        self.add_range_btn = QtWidgets.QPushButton("+")
        self.add_range_btn.setCheckable(True)
        self.add_range_btn.toggled.connect(self._toggle_select_background)
        remove_range_btn = QtWidgets.QPushButton("-")
        remove_range_btn.clicked.connect(self._remove_range)
        range_btn_row.addWidget(self.add_range_btn)
        range_btn_row.addWidget(remove_range_btn)
        col.addLayout(range_btn_row)
        
        self.fit_bg_btn = QtWidgets.QPushButton("Fit from selection")
        self.fit_bg_btn.clicked.connect(self._fit_background)
        col.addWidget(self.fit_bg_btn)
        
        self.remove_bg_toggle = QtWidgets.QPushButton("Remove background")
        self.remove_bg_toggle.setCheckable(True)
        self.remove_bg_toggle.toggled.connect(self._toggle_remove_background)
        col.addWidget(self.remove_bg_toggle)
        
        self.comment_toggle = QtWidgets.QPushButton("Plot with comment")
        self.comment_toggle.setCheckable(True)
        self.comment_toggle.toggled.connect(self._toggle_comment)
        col.addWidget(self.comment_toggle)
        
        save_btn = QtWidgets.QPushButton("Save figure")
        save_btn.clicked.connect(self._save_figure)
        col.addWidget(save_btn)
        
        col.addStretch(1)
        return col

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

        self.remove_bg_toggle.blockSignals(True)
        self.remove_bg_toggle.setChecked(self.current_state.remove_background)
        self.remove_bg_toggle.blockSignals(False)

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

    def _apply_figsize(self, comment: bool):
        w, h = self.BASE_FIGSIZE
        if comment:
            w = w * self.COMMENT_WIDTH_FACTOR
        self.fig.set_size_inches(w, h)
        self.fig.set_dpi(self.FIGURE_DPI)

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
            return

        dat = self.DATA[self.current_meas_index]
        curves = dat.curves if getattr(dat, "n_curves", 0) else []

        self.raw_xy_list = [curve_to_xy(c) for c in curves]

        self._apply_figsize(self.show_comment)

        if self.show_comment:
            ax = self._build_annotated_axes()
        else:
            ax = self.fig.add_subplot(111)

        fit_coeffs = self.current_state.fit_coeffs or [None] * len(curves)
        for idx, (curve, (x, y)) in enumerate(zip(curves, self.raw_xy_list)):
            y_plot = y
            fit_line = None
            coeffs = fit_coeffs[idx] if idx < len(fit_coeffs) else None
            if coeffs is not None:
                fit_line = np.polyval(coeffs, x)
                if self.current_state.remove_background:
                    y_plot = y - fit_line

            ax.plot(x, y_plot, label=curve_display_name(curve, idx))
            if fit_line is not None and not self.current_state.remove_background:
                ax.plot(x, fit_line, "--", color="grey", alpha=0.6)

        for i, (xmin, xmax) in enumerate(self.current_state.bg_ranges):
            if i == self.selected_range_index:
                ax.axvspan(xmin, xmax, color="green", alpha=0.35)
            else:
                ax.axvspan(xmin, xmax, color="orange", alpha=0.2)

        ax.set_title(getattr(dat, "title", self.current_filename), fontsize=16, fontweight="bold")
        ax.legend(loc="best")
        ax.annotate(
            f"fileID: {self.current_file_index}  measurementID: {self.current_meas_index}",
            xy=(0.1, 0.9), xycoords="axes fraction",
        )

        if self.show_comment:
            self._draw_comment_panel(dat, ax)

        if self.add_range_btn.isChecked():
            self._enable_span_selector(ax)

        self.canvas.draw()

    def _build_annotated_axes(self):
        """Layout port of plot_psense(): main axes squished to the left,
        a lightcyan comment panel occupying the widened right portion."""
        original_right = 1.0 / self.COMMENT_WIDTH_FACTOR
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
        ax_text.set_facecolor("lightcyan")
        ax_text.patch.set_visible(True)

        header = ax_text.text(
            0.05, 0.97, "Comment",
            transform=ax_text.transAxes, verticalalignment="top",
            fontsize=18, fontweight="bold",
        )

        ax_text_width_px = ax_text.get_window_extent().width
        fontsize = 11
        dpi = self.fig.get_dpi()
        char_width_px = fontsize * dpi / 72 * 0.5
        wrap_width = max(int(ax_text_width_px / char_width_px), 10)
        wrapped = textwrap.fill(f"{comment_text}\n{timestamp}", width=wrap_width)

        body = ax_text.text(
            0.05, 0.9, wrapped,
            transform=ax_text.transAxes, verticalalignment="top",
            fontsize=fontsize, clip_on=False,
        )

        ax.set_position(self._plot_pos)
        ax_text.set_position(self._text_pos)
        self.fig.canvas.draw()

        bbox_header = header.get_window_extent()
        bbox_body = body.get_window_extent()
        combined = bbox_header.union([bbox_header, bbox_body])

        inv = ax_text.transAxes.inverted()
        x0, y0 = inv.transform((combined.x0, combined.y0))
        x1, y1 = inv.transform((combined.x1, combined.y1))

        pad = 0.03
        patch = FancyBboxPatch(
            (x0 - pad, y0 - pad), (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad,
            boxstyle="square,pad=0", facecolor="lightcyan", edgecolor="none",
            transform=ax_text.transAxes, zorder=0,
        )
        ax_text.add_patch(patch)

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
        if not self.raw_xy_list or not self.current_state.bg_ranges:
            QtWidgets.QMessageBox.information(
                self, "No selection", "Select at least one background range first."
            )
            return

        new_coeffs = []
        fitted_any = False
        for x, y in self.raw_xy_list:
            mask = np.zeros_like(x, dtype=bool)
            for xmin, xmax in self.current_state.bg_ranges:
                mask |= (x >= min(xmin, xmax)) & (x <= max(xmin, xmax))
            if mask.sum() >= 2:
                new_coeffs.append(np.polyfit(x[mask], y[mask], deg=1).tolist())
                fitted_any = True
            else:
                new_coeffs.append(None)

        if not fitted_any:
            QtWidgets.QMessageBox.warning(self, "Fit failed", "Not enough points in the selection.")
            return

        self.current_state.fit_coeffs = new_coeffs
        self._save_state()
        self._refresh_plot()

    def _toggle_remove_background(self, checked):
        self.current_state.remove_background = checked
        self._save_state()
        self._refresh_plot()
    
    def _toggle_stretchplot(self, checked):
        self.EDIT_TOGGLE_STRETCH = checked
        self._refresh_plot()
        
    def _toggle_comment(self, checked):
        self.show_comment = checked
        self._refresh_plot()
    
    # -- Selecting, editing and restoring text entry fields ---
    
    def _on_param_changed(self):
        """Called when any parameter is changed - updates the plot if needed"""
        # Update the figure size and DPI
        self.BASE_FIGSIZE = self.figsize_row.get_value(), self.figsize_row.get_value()
        self.COMMENT_WIDTH_FACTOR = self.width_factor_row.get_value()
        self.FIGURE_DPI = self.dpi_row.get_value()
        
        # Update save parameters
        self.SAVE_WIDTH_IN = self.save_width_row.get_value()
        self.SAVE_HEIGHT_IN = self.save_height_row.get_value()
        self.EDIT_FIGURE_DPI = self.save_dpi_row.get_value()
        
        # Refresh the plot with new parameters
        self._refresh_plot()
        
        # Save the parameters to the session
        self._save_session_parameters()
    def _load_session_parameters(self):
        """Load saved parameter values from the session config"""
        PARAMS_CONFIG_PATH = Path.home() / ".config" / "palmsens_gui" / "params.json"
        if PARAMS_CONFIG_PATH.exists():
            try:
                with open(PARAMS_CONFIG_PATH) as f:
                    params = json.load(f)
                
                # Update the parameters
                if 'EDIT_COMMENT_WIDTH_FACTOR' in params:
                    self.COMMENT_WIDTH_FACTOR = params['EDIT_COMMENT_WIDTH_FACTOR']
                if 'EDIT_FIGURE_DPI' in params:
                    self.FIGURE_DPI = params['EDIT_FIGURE_DPI']
                if 'EDIT_FIGSIZE_X' in params:
                    self.SAVE_WIDTH_IN = params['EDIT_FIGSIZE_X']
                if 'EDIT_FIGSIZE_Y' in params:
                    self.SAVE_HEIGHT_IN = params['EDIT_FIGSIZE_Y']

                    
                # Update the line edits if they exist
                if hasattr(self, 'figsize_row'):
                    self.figsize_row.line_edit.setText(str(self.BASE_FIGSIZE[0]))
                    self.width_factor_row.line_edit.setText(str(self.COMMENT_WIDTH_FACTOR))
                    self.dpi_row.line_edit.setText(str(self.FIGURE_DPI))
                    self.save_width_row.line_edit.setText(str(self.SAVE_WIDTH_IN))
                    self.save_height_row.line_edit.setText(str(self.SAVE_HEIGHT_IN))
                    self.save_dpi_row.line_edit.setText(str(self.EDIT_FIGURE_DPI))
            except Exception as e:
                print(f"Error loading parameters: {e}")
    def _save_session_parameters(self):
        """Save the current parameter values to the session config"""
        params = {
            'EDIT_COMMENT_WIDTH_FACTOR': self.COMMENT_WIDTH_FACTOR,
            'EDIT_FIGSIZE_X': self.EDIT_FIGSIZE_X,
            'EDIT_FIGSIZE_Y': self.EDIT_FIGSIZE_Y,
            'EDIT_FIGURE_DPI': self.EDIT_FIGURE_DPI
        }
        
        # Save to a separate config file
        PARAMS_CONFIG_PATH = Path.home() / ".config" / "palmsens_gui" / "params.json"
        PARAMS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PARAMS_CONFIG_PATH, "w") as f:
            json.dump(params, f, indent=2)
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

        self.fig.set_size_inches(self.SAVE_WIDTH_IN, self.SAVE_HEIGHT_IN)
        self.fig.set_dpi(self.EDIT_FIGURE_DPI)
        self.fig.savefig(path, dpi=self.EDIT_FIGURE_DPI)

        self.fig.set_size_inches(on_screen_size)
        self.fig.set_dpi(on_screen_dpi)
        self.canvas.draw()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()