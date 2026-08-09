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

import numpy as np

from PyQt5 import QtWidgets, QtGui


HEADER_BG = QtGui.QColor("#c8c8c8")
class OverrideParamRow(QtWidgets.QWidget):
    """A row with a label, a QLineEdit (no validator, can hold string), and a reset button.
       The line edit shows a grey placeholder text (the default value) when empty.
       The stored value is None when empty, else the entered string."""
    def __init__(self, label, param_name, parent=None, placeholder_getter=None):
        """
        label: displayed text (e.g., "X Label")
        param_name: name of the attribute on the parent window (e.g., "EDIT_XLABEL_OVERRIDE")
        parent: the MainWindow instance
        placeholder_getter: callable that returns the current default string (e.g., lambda: "Time [s]")
        """
        super().__init__(parent)
        self.param_name = param_name
        self.parent_window = parent
        self._placeholder_getter = placeholder_getter
        self._is_updating = False

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        # Label
        self.label = QtWidgets.QLabel(label)
        self.label.setFixedWidth(80)
        layout.addWidget(self.label)

        # Line edit
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.textChanged.connect(self._on_text_changed)
        # Set initial value from parent if any
        if parent and hasattr(parent, param_name):
            val = getattr(parent, param_name)
            if val is not None:
                self.line_edit.setText(val)
        self._update_placeholder()
        layout.addWidget(self.line_edit)

        # Reset button
        self.reset_btn = QtWidgets.QPushButton("⟳")
        self.reset_btn.setFixedWidth(30)
        self.reset_btn.setToolTip("Reset to default (clear override)")
        self.reset_btn.clicked.connect(self._reset_to_default)
        layout.addWidget(self.reset_btn)

    def _on_text_changed(self, text):
        if self._is_updating:
            return
        # If text is empty, store None, else store the string
        value = text.strip() if text.strip() else None
        setattr(self.parent_window, self.param_name, value)
        if hasattr(self.parent_window, '_on_param_changed'):
            self.parent_window._on_param_changed()

    def _reset_to_default(self):
        self._is_updating = True
        self.line_edit.setText("")
        setattr(self.parent_window, self.param_name, None)
        self._update_placeholder()
        if hasattr(self.parent_window, '_on_param_changed'):
            self.parent_window._on_param_changed()
        self._is_updating = False

    def _update_placeholder(self):
        """Update the placeholder text using the getter."""
        if self._placeholder_getter is not None:
            placeholder = self._placeholder_getter()
            self.line_edit.setPlaceholderText(placeholder)
            # Set grey colour for placeholder – Qt does this automatically.

    def update_placeholder_text(self):
        """Public method to refresh the placeholder (e.g., after loading new data)."""
        self._update_placeholder()

    def get_value(self):
        """Return the current override string or None."""
        text = self.line_edit.text().strip()
        return text if text else None
    
class EditableParamRow(QtWidgets.QWidget):
    def __init__(self, label, default_value, param_name, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self._default_value = default_value
        self.parent_window = parent
        self._is_updating = False  # This will prevent recursive updates
        
        # Create layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        
        # Label
        self.label = QtWidgets.QLabel(label)
        self.label.setFixedWidth(80)
        layout.addWidget(self.label)
        
        # This is the editable field generated in the row.
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setText(str(default_value))
        self.line_edit.textChanged.connect(self._on_text_changed)
        
        # Add validator to only allow numbers 
        validator = QtGui.QDoubleValidator()
        validator.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.line_edit.setValidator(validator)
        
        layout.addWidget(self.line_edit)
        
        # Add Reset Button
        self.reset_btn = QtWidgets.QPushButton("⟳")
        self.reset_btn.setFixedWidth(30)
        self.reset_btn.setToolTip("Reset to default")
        self.reset_btn.clicked.connect(self._reset_to_default)
        layout.addWidget(self.reset_btn)
    
    def _on_text_changed(self, text):
        """Only update if the text is valid and not empty"""
        if self._is_updating:
            return
        
        if not text or text.strip() == "":
            return
        
        try: #try float, and check if inf.
            value = float(text)
            
            if not np.isfinite(value):
                return
            

            setattr(self.parent_window, self.param_name, value)
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        except ValueError:
            pass
    
    def _reset_to_default(self):
        # Use provided DEFAULT_* parameters if available
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

class MultiParamRow(QtWidgets.QWidget):
    """
    Two modes: header (with headers and row labels) or cell (per‑cell labels).
    Supports typed parameters: 'float' or 'string' (default).
    Empty fields: no update until focus lost; then reset to default.
    Invalid floats: no update; reset to default on focus loss.

    Parameters:
        default_type: type for ALL parameters ('float' or 'string')
        type_overrides: dict {param_name: 'float' or 'string'} for exceptions
        types: (deprecated) full dict; if provided, overrides default_type/overrides
    """
    def __init__(self, parent=None, headers=None, row_labels=None, param_names=None,
             rows=None, types=None, defaults=None,
             default_type='string', type_overrides=None,
             allow_none=False):   # <-- new
        super().__init__(parent)
        self.parent_window = parent
        self.allow_none = allow_none
        self._is_updating = False
        self.line_edits = {}
        self.reset_btns = {}
        self.default_values = {}
        # self.param_types will be set after UI construction
        self.param_types = {}
        self._last_valid = {}

        # Build UI
        if headers is not None:
            self.mode = 'header'
            self.headers = headers
            self.row_labels = row_labels
            self.param_names = param_names
            self._build_header_mode()
        else:
            self.mode = 'cell'
            self.rows = rows
            self._build_cell_mode()

        # --- NEW: determine param types ---
        if types is not None:
            # Backward compatibility: use provided types dict
            self.param_types = types
        else:
            # All parameters get default_type, then overrides are applied
            self.param_types = {name: default_type for name in self.line_edits}
            if type_overrides:
                self.param_types.update(type_overrides)

        # Load defaults (from parent's DEFAULT_* or from provided `defaults`)
        self._load_defaults(parent, defaults)

    def _build_header_mode(self):
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(2)

        for col_idx, header_text in enumerate(self.headers):
            header = QtWidgets.QLabel(header_text)
            header.setStyleSheet("font-weight: bold;")
            layout.addWidget(header, 0, col_idx * 2 + 1, 1, 2)

        reset_all_btn = QtWidgets.QPushButton("↺ Reset All")
        reset_all_btn.setFixedWidth(80)
        reset_all_btn.setToolTip("Reset all values to defaults")
        reset_all_btn.clicked.connect(self.reload_defaults)
        layout.addWidget(reset_all_btn, 0, len(self.headers) * 2 + 1)

        for row_idx, row_label in enumerate(self.row_labels):
            label = QtWidgets.QLabel(row_label)
            label.setStyleSheet("font-weight: bold;")
            layout.addWidget(label, row_idx + 1, 0)

            for col_idx, param_name in enumerate(self.param_names[row_idx]):
                le = self._create_line_edit(param_name)
                layout.addWidget(le, row_idx + 1, col_idx * 2 + 1)
                self.line_edits[param_name] = le

                rb = self._create_reset_button(param_name, f"{row_label} {self.headers[col_idx]}")
                layout.addWidget(rb, row_idx + 1, col_idx * 2 + 2)
                self.reset_btns[param_name] = rb

    def _build_cell_mode(self):
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setHorizontalSpacing(5)
        layout.setVerticalSpacing(2)

        for row_idx, row_items in enumerate(self.rows):
            col = 0
            for label_text, param_name in row_items:
                lbl = QtWidgets.QLabel(label_text)
                lbl.setFixedWidth(60)
                layout.addWidget(lbl, row_idx, col * 3)

                le = self._create_line_edit(param_name)
                layout.addWidget(le, row_idx, col * 3 + 1)
                self.line_edits[param_name] = le

                rb = self._create_reset_button(param_name, label_text)
                layout.addWidget(rb, row_idx, col * 3 + 2)
                self.reset_btns[param_name] = rb

                col += 1

    def _create_line_edit(self, param_name):
        le = QtWidgets.QLineEdit()
        # Set initial value from parent or default
        if hasattr(self.parent_window, param_name):
            val = getattr(self.parent_window, param_name)
            if val is not None:
                le.setText(str(val))
                self._last_valid[param_name] = val
        le.textChanged.connect(lambda text, p=param_name: self._on_text_changed(p, text))
        le.editingFinished.connect(lambda p=param_name: self._on_editing_finished(p))
        return le

    def _create_reset_button(self, param_name, tooltip):
        rb = QtWidgets.QPushButton("⟳")
        rb.setFixedWidth(30)
        rb.setToolTip(f"Reset {tooltip} to default")
        rb.clicked.connect(lambda checked, p=param_name: self._reset_to_default(p))
        return rb

    def _load_defaults(self, parent, explicit_defaults):
        """Read defaults from parent's DEFAULT_* attributes or explicit dict."""
        self.default_values = {}
        for param_name in self.line_edits.keys():
            if explicit_defaults and param_name in explicit_defaults:
                self.default_values[param_name] = explicit_defaults[param_name]
            else:
                default_name = f"DEFAULT_{param_name}"
                if hasattr(parent, default_name):
                    self.default_values[param_name] = getattr(parent, default_name)
                else:
                    self.default_values[param_name] = None
            # If no valid initial value, store default as last valid
            if param_name not in self._last_valid:
                self._last_valid[param_name] = self.default_values[param_name]

    def _convert_value(self, param_name, text):
        """Convert text to the appropriate type (float or string)."""
        if self.param_types.get(param_name) == 'float':
            try:
                return float(text)
            except ValueError:
                return None  # invalid
        else:
            return text

    def _on_text_changed(self, param_name, text):
        """Real‑time update only if text is non‑empty and convertible."""
        if self._is_updating:
            return
        if not text.strip():
            # Empty: do nothing (wait for focus loss to reset)
            return
        # Try to convert
        value = self._convert_value(param_name, text)
        if value is not None:
            # Valid conversion – store, update parent, refresh
            self._last_valid[param_name] = value
            setattr(self.parent_window, param_name, value)
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        # else: invalid input – ignore, keep previous value

    def _on_editing_finished(self, param_name):
        if self._is_updating:
            return
        le = self.line_edits[param_name]
        text = le.text().strip()
    
        if not text:
            if self.allow_none:
                # Empty -> treat as None (no reset to default)
                self._last_valid[param_name] = None
                setattr(self.parent_window, param_name, None)
                if hasattr(self.parent_window, '_on_param_changed'):
                    self.parent_window._on_param_changed()
            else:
                # Original behavior: reset to default
                self._reset_to_default(param_name)
        else:
            value = self._convert_value(param_name, text)
            if value is None:
                # Invalid -> reset to default (still the same)
                self._reset_to_default(param_name)
            else:
                # Valid – update
                current = getattr(self.parent_window, param_name, None)
                if current != value:
                    self._last_valid[param_name] = value
                    setattr(self.parent_window, param_name, value)
                    if hasattr(self.parent_window, '_on_param_changed'):
                        self.parent_window._on_param_changed()

    def _reset_to_default(self, param_name):
        if self._is_updating:
            return
        self._is_updating = True
        try:
            default = self.default_values.get(param_name, None)
            self.line_edits[param_name].setText(str(default) if default is not None else "")
            self._last_valid[param_name] = default
            setattr(self.parent_window, param_name, default)
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        finally:
            self._is_updating = False

    def reload_defaults(self):
        """Reset all entries to their defaults."""
        if self._is_updating:
            return
        self._is_updating = True
        try:
            for param_name in self.line_edits.keys():
                default = self.default_values.get(param_name, None)
                self.line_edits[param_name].setText(str(default) if default is not None else "")
                self._last_valid[param_name] = default
                setattr(self.parent_window, param_name, default)
            if hasattr(self.parent_window, '_on_param_changed'):
                self.parent_window._on_param_changed()
        finally:
            self._is_updating = False

    def get_values(self):
        """Return a dict param_name -> current value (converted type or None)."""
        values = {}
        for param_name, le in self.line_edits.items():
            text = le.text().strip()
            if not text:
                values[param_name] = None
            else:
                value = self._convert_value(param_name, text)
                values[param_name] = value  # may be None if invalid
        return values

    def set_values(self, values_dict):
        """Set entry texts from a dict of param_name -> value (any type)."""
        if self._is_updating:
            return
        self._is_updating = True
        try:
            for param_name, value in values_dict.items():
                if param_name in self.line_edits:
                    self.line_edits[param_name].setText(str(value) if value is not None else "")
                    self._last_valid[param_name] = value
                    setattr(self.parent_window, param_name, value)
        finally:
            self._is_updating = False

                
