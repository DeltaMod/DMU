"""
Plotting logic and other helper functions
"""
from dataclasses import dataclass, field
from typing import Protocol, Optional, Any
import numpy as np
from pathlib import Path
from matplotlib.figure import Figure


def curve_display_name(curve, index=0):
    """Finds details for curves for display."""
    for attr in ("title", "name", "label", "array_type"):
        val = getattr(curve, attr, None)
        if val:
            return str(val)
    return f"curve{index}"

def get_unit_oom_from_string(unit):
    UNIT_CONVERSION = {"f":1e-15,"p":1e-12,"n":1e-9,"µ":1e-6, "m":1e-3, "":1e+0, "k":1e+3, "M":1e+6, "G":1e+9, "T":1e+12}
    if len(unit) == 2 and type(unit) ==  str:
        base_unit = unit[1:]
        oom = UNIT_CONVERSION[unit[0]]
    else:
        base_unit = unit
        oom = UNIT_CONVERSION[""]
    return(oom,base_unit)


def nice_ticks(vmin, vmax, max_ticks=8):
    """
    Return an array of nice tick values between vmin and vmax.
    Step is chosen from 1, 2, 5 times a power of 10.
    """
    if vmax <= vmin:
        return np.array([vmin])
    range_val = vmax - vmin
    # Desired step size
    raw_step = range_val / max_ticks
    # Get magnitude (power of 10)
    magnitude = 10 ** np.floor(np.log10(raw_step))
    # Normalize
    norm = raw_step / magnitude
    # Choose step: 1, 2, or 5
    if norm < 1.5:
        step = magnitude
    elif norm < 3.5:
        step = 2 * magnitude
    else:
        step = 5 * magnitude
    # Ensure step is positive
    step = max(step, 1e-15)
    # Compute first and last tick
    first = np.ceil(vmin / step) * step
    last = np.floor(vmax / step) * step
    # Generate ticks
    ticks = np.arange(first, last + step/2, step)
    # Limit to a reasonable number
    if len(ticks) > max_ticks + 2:
        # If too many, increase step (try doubling)
        step *= 2
        first = np.ceil(vmin / step) * step
        last = np.floor(vmax / step) * step
        ticks = np.arange(first, last + step/2, step)
    return ticks

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