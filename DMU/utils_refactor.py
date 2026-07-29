# -*- coding: utf-8 -*-
"""
Lumerical Data Handling
Created on Tue Aug 18 17:06:05 2020
@author: Vidar Flodgren
Github: https://github.com/DeltaMod
"""
import os
from collections import defaultdict
import sys
import time
import re
import h5py
import matplotlib
import matplotlib as mpl
import tkinter as tk

from tkinter.filedialog import askopenfilename, askdirectory
from matplotlib import patches as ptc
from matplotlib import colormaps as cmaps
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d #If you want to be able to use projection="3D", then you need this:
import scipy
import numpy as np
from scipy import integrate, interpolate, constants
import json
from collections import Counter
import natsort
import csv
import xlrd
import mat73
import pickle
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
# ---------- HANDLE IMPORTS FOR BOTH PACKAGE AND DEBUG ----------
try:
    # Running directly from file in debug
    from .custom_logger import get_custom_logger
    from .plot_utils import *
    from .utils_utils import *
except ImportError:
    # Relative imports for when it is being run as a package
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from DMU.custom_logger import get_custom_logger
    from DMU.plot_utils import *
    from DMU.utils_utils import *

logger = get_custom_logger("DMU_UTILS")

# =============================================================================
#  Keithley data importer Helper functions
# =============================================================================

def strip_measurement_suffix(stem: str) -> str:
    return re.sub(r'_(2Term|4Term|LOG)$', '', stem, flags=re.IGNORECASE)

def find_common_prefix(stems):
    if not stems:
        return ""
    prefix = stems[0]
    for s in stems[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

def get_sec(time_str: str) -> float:
    try:
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return 0

def num_only(data):
    if isinstance(data, list):
        return [float(x) for x in data if isinstance(x, (int, float))]
    return [float(data)] if isinstance(data, (int, float)) else []

def segment_sweep(data, indices):
    if not data or not indices:
        return data
    idx0, idx1 = int(indices[0]), int(indices[1])
    if len(indices) == 2:
        return data[idx0:idx1]
    return [data[int(i):int(j)] for i, j in zip(indices[:-1], indices[1:])]

def turning_points(data):
    if not data or len(data) < 3:
        return [0, len(data)]
    points = [0]
    for i in range(1, len(data)-1):
        if (data[i] > data[i-1] and data[i] > data[i+1]) or \
           (data[i] < data[i-1] and data[i] < data[i+1]):
            points.append(i)
    points.append(len(data))
    return points

def Convert_to_type(data_dict):
    for key, value in data_dict.items():
        if isinstance(value, str):
            try:
                data_dict[key] = float(value)
            except ValueError:
                pass
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    try:
                        value[i] = float(item)
                    except ValueError:
                        pass
    return data_dict

NEWDEF = {"N/A": None, "Enabled": True, "Disabled": False, "OFF": False, "ON": True}
KEYDEF = {
    "Number of Points": "Npts",
    "Step": "VStep",
    "Start/Bias": "VStart",
    "Stop": "VStop",
    "Name": "Colname",
    "Instrument": "SMU",
    "Dual Sweep": "FBSweep"
}

# =============================================================================
# LogBook class
# =============================================================================
class LogBook:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._parse()

    def _parse(self) -> Dict[str, Dict]:
        with open(os.devnull, "w") as devnull:
            xls = xlrd.open_workbook(self.file_path, logfile=devnull)
        try:
            sheet = xls.sheet_by_index(0)
            rows = [sheet.row_values(i) for i in range(1, sheet.nrows)]
            headers = [str(cell) for cell in sheet.row_values(0) if cell != ""]
            smu_indices = [i for i, h in enumerate(headers) if "SMU" in h]
            flat_data = {'positions': {}}
            for ind in smu_indices:
                pos_key = f"pos{1 + ind - min(smu_indices)}"
                flat_data['positions'][pos_key] = {'SMU': headers[ind]}
            # first pass: NW info
            for row in rows:
                if not any(row):
                    continue
                if row[0] == "":
                    if any(sub in str(el) for el in row for sub in ("NW", "n-i-p", "p-i-n")):
                        for i, var in enumerate(smu_indices):
                            pos_key = f"pos{1 + var - min(smu_indices)}"
                            if row[var] == "":
                                if "NW" in str(row[var-1]):
                                    flat_data['positions'][pos_key]['NW'] = row[var-1]
                                if "n-i-p" in str(row[var-1]) or "p-i-n" in str(row[var-1]):
                                    flat_data['positions'][pos_key]['NW Orientation'] = row[var-1]
                            elif "NW" in str(row[var]):
                                flat_data['positions'][pos_key]['NW'] = row[var]
                                if "n-i-p" in str(row[var]) or "p-i-n" in str(row[var]):
                                    flat_data['positions'][pos_key]['NW Orientation'] = row[var]
                if row[0] != "":
                    break
            # second pass: run rows
            for row in rows:
                if not any(row):
                    continue
                if row[0] != "":
                    try:
                        run_key = str(int(row[0]))
                    except ValueError:
                        continue
                    row_dict = {}
                    for i, header in enumerate(headers):
                        if i in smu_indices:
                            pos_idx = 1 + i - min(smu_indices)
                            pos_key = f'pos{pos_idx}'
                            row_dict[pos_key] = {
                                'SMU': flat_data['positions'][pos_key]['SMU'],
                                'NW': flat_data['positions'][pos_key].get('NW')
                            }
                        row_dict[header] = row[i]
                    if "device" in row_dict and "Device" not in row_dict:
                        row_dict["Device"] = row_dict["device"]
                        del row_dict['device']
                    if "device" not in [entry.lower() for entry in row_dict.keys()]:
                        row_dict["Device"] = 'Unlabelled'
                    if type(row_dict["Device"]) != str:
                        row_dict["Device"] = "Unlabelled"
                    row_dict["LOG Directory"] = self.file_path
                    flat_data[run_key] = row_dict
            return flat_data
        finally:
            xls.release_resources()

    def get_run_info(self, run_key: str) -> Optional[Dict]:
        if run_key in self.data:
            return self.data[run_key]
        if run_key.startswith("Run"):
            num = run_key[3:]
            if num in self.data:
                return self.data[num]
        return None

# =============================================================================
# KeithleyDataReader class – redesigned column identification
# =============================================================================
class KeithleyDataReader:
    def __init__(self, logbook: Optional[LogBook] = None):
        self.logbook = logbook
        self.newdef = NEWDEF
        self.keydef = KEYDEF

    def read_file(self, file_path: str) -> Dict[str, Any]:
        with open(os.devnull, "w") as devnull:
            xls = xlrd.open_workbook(file_path, logfile=devnull)
        try:
            file_data = {}
            settings_sheets = [s for s in xls.sheet_names() if "settings" in s.lower()]
            run_sheets = [s for s in xls.sheet_names() if s.lower().startswith("run")]

            if settings_sheets:
                settings_data = self._parse_settings_sheet(xls, settings_sheets[0])
                file_data["Settings"] = settings_data

            for sheet_name in run_sheets:
                if sheet_name == "Calc":
                    continue
                run_key = self._extract_run_key(sheet_name)
                if run_key is None:
                    continue
                settings = file_data.get("Settings", {}).get(run_key)
                if settings is None:
                    logger.warning(f"No settings for {run_key} in {os.path.basename(file_path)}")
                    continue

                cols = self._parse_run_sheet(xls, sheet_name, file_path, settings)
                if cols is None:
                    continue

                stats = settings.copy()
                self._process_run_data(cols, stats, sheet_name, file_path)
                self._detect_emitter_detector(cols, stats)

                cols['Settings'] = stats
                file_data[sheet_name] = cols

            return file_data
        finally:
            xls.release_resources()

    def _parse_settings_sheet(self, xls, sheet_name: str) -> Dict[str, Dict]:
        sheet = xls.sheet_by_name(sheet_name)
        rows = [sheet.row_values(i) for i in range(sheet.nrows)]
        settings_data = {}
        current_run = None

        for row in rows:
            if any("===" in str(s) for s in row) or all(not str(s).strip() for s in row):
                continue
            if any("Run" in str(s) for s in row):
                current_run = str(row[0]).strip().replace(" ", "")
                settings_data[current_run] = {}
                continue
            if current_run is None:
                continue
            header = str(row[0]).strip()
            if len(row) > 1:
                values = [x for x in row[1:] if x != ""]
                if len(values) == 1:
                    settings_data[current_run][header] = str(values[0])
                elif values:
                    settings_data[current_run][header] = str(values)

        for run in settings_data:
            settings_data[run] = Convert_to_type(settings_data[run])

        for run_no, run_settings in settings_data.items():
            run_settings["Formulas"] = {}
            for key in list(run_settings.keys()):
                value = run_settings[key]
                if isinstance(value, list):
                    run_settings[key] = [self.newdef.get(v, v) for v in value]
                if "=" in str(key):
                    formula_name = str(key).split("=")[0].strip()
                    run_settings["Formulas"][formula_name] = key
                    del run_settings[key]
                    continue
                if key in self.keydef:
                    run_settings[self.keydef[key]] = run_settings.pop(key)
        return settings_data

    def _extract_run_key(self, sheet_name: str) -> Optional[str]:
        match = re.search(r"Run\s*(\d+)", sheet_name, re.IGNORECASE)
        if match:
            return f"Run{match.group(1)}"
        if sheet_name.isdigit():
            return f"Run{sheet_name}"
        return None

    def _parse_run_sheet(self, xls, sheet_name: str, file_path: str, settings: Dict) -> Optional[Dict]:
        sheet = xls.sheet_by_name(sheet_name)
        cols = {}
        cols["Data directory"] = file_path
        col_dict = {}
        header_order = []
    
        for col_index in range(sheet.ncols):
            col_data = [x for x in sheet.col_values(col_index) if x != ""]
            if not col_data:
                continue
            header = str(col_data[0])
            data = col_data[1:]
            if len(data) == 1:
                data = data[0]
            col_dict[header] = data
            header_order.append(header)
    
        cols["column_order"] = header_order
        for h in header_order:
            cols[h] = col_dict[h]
    
        # ---- Identify applied voltage and measured current columns (ALL of them) ----
        applied_cols = []
        measured_cols = []
    
        colnames = settings.get("Colname")
        if colnames is not None:
            if isinstance(colnames, str):
                colnames = [colnames]
            for name in colnames:
                if name in col_dict:
                    applied_cols.append(name)
                    idx = header_order.index(name)
                    if idx > 0:
                        measured_cols.append(header_order[idx - 1])
    
        # If nothing found via Colname, try heuristic (columns ending with 'V')
        if not applied_cols:
            for h in header_order:
                if h.endswith('V') and not any(s in h for s in ["START", "STOP"]):
                    applied_cols.append(h)
                    idx = header_order.index(h)
                    if idx > 0:
                        measured_cols.append(header_order[idx - 1])
                    break   # only grab the first pair in heuristic mode
    
        # Fallback: first two columns
        if not applied_cols and len(header_order) >= 2:
            applied_cols.append(header_order[1])
            measured_cols.append(header_order[0])
    
        if not applied_cols or not measured_cols:
            logger.warning(f"Could not identify applied/measured columns in {sheet_name}")
            return None
    
        # Store all pairs for multi‑SMU handling
        cols['applied_cols'] = applied_cols
        cols['measured_cols'] = measured_cols
    
        # Keep primary pair for legacy access
        cols['voltage'] = col_dict.get(applied_cols[0], [])
        cols['current'] = col_dict.get(measured_cols[0], [])
        cols['applied_col'] = applied_cols[0]
        cols['measured_col'] = measured_cols[0]
    
        return cols

    def _process_run_data(self, cols: Dict, stats: Dict, sheet_name: str, file_path: str):
        # ---------- Ensure all stats values that will be indexed are lists ----------
        if not isinstance(stats.get('Npts', []), list):
            for key in list(stats.keys()):
                if not isinstance(stats[key], list):
                    stats[key] = [stats[key]]
    
        # ---------- Robust Npts handling ----------
        npts_raw = stats.get('Npts', 0)
        # Now npts_raw is guaranteed to be a list
        if not isinstance(npts_raw, list):
            npts_raw = [npts_raw]   # just in case
    
        # Convert each entry to an integer, treating non‑numeric as 0
        npts_ints = []
        for v in npts_raw:
            try:
                npts_ints.append(int(float(v)))
            except (ValueError, TypeError):
                npts_ints.append(0)
    
        # Choose the largest valid value and its index
        max_npts = max(npts_ints)
        main_col = npts_ints.index(max_npts) if max_npts > 0 else 0
    
        # If no valid Npts found, fall back to the length of the current column
        if max_npts == 0:
            if 'current' in cols and isinstance(cols['current'], list):
                max_npts = len(cols['current'])
                main_col = 0
            else:
                logger.warning(f"No valid Npts and no current data in {sheet_name}, skipping")
                return
    
        npts = int(max_npts)
    
        # ---------- Time per point ----------
        try:
            exec_time = stats.get('Execution Time', '0:00:00')
            time_per_point = get_sec(exec_time) / npts if npts else 0
        except:
            time_per_point = 0
        stats["Time Per Point"] = time_per_point
        stats["NWID"] = ["NW1", "NW1", "NW2", "NW2"]
    
        try:
            cols["Time"] = np.linspace(0, npts * time_per_point, npts)
        except:
            cols["Time"] = np.linspace(0, npts, npts)
    
        # ---------- Ensure VStep is float ----------
        try:
            if isinstance(stats["VStep"][main_col], str):
                stats["VStep"][main_col] = float(stats["VStep"][main_col])
        except (ValueError, TypeError):
            stats["VStep"][main_col] = 0.0   # safe fallback
    
        # ---------- Detect sweep indices ----------
        try:
            if (stats["FBSweep"][main_col] == True and
                npts == 2 * (1 + int(abs(stats["VStart"][main_col] - stats["VStop"][main_col]) / abs(stats["VStep"][main_col])))):
                sweep_indices = [0, int(npts / 2), npts]
            else:
                sweep_indices = [0, npts]
        except:
            sweep_indices = [0, npts]
    
        print(f"{os.sep.join(file_path.split(os.sep)[-3:])} - {sheet_name}: {stats['Operation Mode'][main_col]}")
    
        # ---------- Segment sweep for linear sweeps ----------
        if "Voltage Linear Sweep" in stats["Operation Mode"]:
            list_keys = [k for k, v in cols.items() if isinstance(v, list) and k not in ("Data directory", "column_order")]
            for k in list_keys:
                cols[k] = segment_sweep(cols[k], sweep_indices)
    
        # ---------- Special handling for curing sweeps ----------
        if ("Voltage List Sweep" in stats["Operation Mode"] and
            "curing" in stats["Test Name"].lower()):
            list_keys = [k for k, v in cols.items() if isinstance(v, list) and k not in ("Data directory", "column_order")]
            tps = turning_points(cols["voltage"])
            for k in list_keys:
                cols[k] = segment_sweep(cols[k], tps)

    def _detect_emitter_detector(self, cols: Dict, stats: Dict):
        modes = stats["Operation Mode"]
        if len(modes) == 1:
            if "Voltage List Sweep" in modes:
                em_key = "Voltage List Sweep"
            elif "Voltage Linear Sweep" in modes:
                em_key = "Voltage Linear Sweep"
            else:
                em_key = modes[0]
            detector_ID = None
            emitter_OP = em_key
            detector_OP = None
        elif len(modes) == 2:
            if "Voltage List Sweep" in modes:
                em_key = "Voltage List Sweep"
            elif "Voltage Linear Sweep" in modes:
                em_key = "Voltage Linear Sweep"
            else:
                em_key = modes[0]
            emitter_ID = modes.index(em_key)
            detector_ID = None
            emitter_OP = em_key
            detector_OP = None
        else:
            if "Voltage Bias" in modes and "Voltage List Sweep" in modes:
                emitter_ID = modes.index('Voltage List Sweep')
                detector_ID = modes.index('Voltage Bias')
                emitter_OP = "Voltage List Sweep"
                detector_OP = "Voltage Bias"
            elif "Voltage Bias" in modes and "Voltage Linear Sweep" in modes:
                emitter_ID = modes.index('Voltage Linear Sweep')
                detector_ID = modes.index('Voltage Bias')
                emitter_OP = "Voltage Linear Sweep"
                detector_OP = "Voltage Bias"
            else:
                emitter_ID = next((i for i, m in enumerate(modes) if "Sweep" in m), 0)
                detector_ID = None
                emitter_OP = modes[emitter_ID] if emitter_ID < len(modes) else "Unknown"
                detector_OP = None

        try:
            emitter_dict = {
                "SMU": stats["SMU"][emitter_ID],
                "colname": stats["Colname"][emitter_ID],
                'NWID': stats["NWID"][emitter_ID].split(' ')[0],
                "Operation Mode": emitter_OP
            }
        except:
            emitter_dict = {"SMU": None, "colname": None, "NWID": None, "Operation Mode": emitter_OP}

        if detector_ID is not None:
            try:
                detector_dict = {
                    "SMU": stats["SMU"][detector_ID],
                    "colname": stats["Colname"][detector_ID],
                    'NWID': stats["NWID"][detector_ID].split(' ')[0],
                    "Operation Mode": detector_OP
                }
            except:
                detector_dict = {"SMU": None, "colname": None, "NWID": None, "Operation Mode": detector_OP}
        else:
            detector_dict = {"SMU": None, "colname": None, "NWID": None, "Operation Mode": detector_OP}

        cols["emitter"] = emitter_dict
        cols["detector"] = detector_dict

def _find_matching_log(data_stem: str, log_files: List[Path]) -> Optional[Path]:
    """
    Given a data file stem (without extension) and a list of LOG file Paths,
    return the LOG file Path whose stem best matches the data file stem.
    Dashes and underscores are treated as identical for matching.

    Strategy:
      1. Perfect candidate: data_stem with last '_...' replaced by '_LOG'.
         Example: DFR1_IG_BL3_2Term -> DFR1_IG_BL3_LOG
      2. Longest common prefix with any log file stem (minimum 50% length of data_stem).
    """
    # Normalise dashes/underscores for comparison
    data_clean = data_stem.replace('-', '_')

    # --- perfect candidate ---
    if '_' in data_clean:
        candidate = data_clean.rsplit('_', 1)[0] + '_LOG'
        for lf in log_files:
            if lf.stem.replace('-', '_') == candidate:
                return lf

    # --- longest common prefix fallback ---
    best_match = None
    best_len = 0
    min_len = max(1, len(data_clean) // 2)

    for lf in log_files:
        log_clean = lf.stem.replace('-', '_')
        common = os.path.commonprefix([data_clean, log_clean])
        if len(common) >= min_len and len(common) > best_len:
            best_len = len(common)
            best_match = lf

    return best_match


# =============================================================================
# Public API
# =============================================================================


def Keithley_xls_read_file(file_path: str, logbook: Optional[LogBook] = None) -> Dict[str, Any]:
    """
    Read a single Keithley .xls file.
    If logbook is not provided, the function automatically searches for a matching
    LOG file in the same directory (using _find_matching_log).
    """
    # Auto-detect logbook if none supplied
    if logbook is None:
        directory = Path(file_path).parent
        log_files = [f for f in directory.glob("*.xls") if "LOG" in f.stem.upper()]
        matched = _find_matching_log(Path(file_path).stem, log_files)
        if matched is not None:
            try:
                logbook = LogBook(str(matched))
            except Exception as e:
                logger.warning(f"Could not parse logbook {matched.name}: {e}")

    reader = KeithleyDataReader(logbook)
    file_data = reader.read_file(file_path)
    result = {}

    if "Settings" in file_data:
        settings_dict = file_data.pop("Settings")
        for run_key, run_data in file_data.items():
            if run_key in settings_dict:
                run_data['Settings'] = settings_dict[run_key]
            else:
                num_part = run_key.replace("Run", "")
                if num_part in settings_dict:
                    run_data['Settings'] = settings_dict[num_part]
                else:
                    run_data['Settings'] = {}
            result[run_key] = run_data
    else:
        result = file_data

    if logbook is not None:
        for run_key, run_data in result.items():
            log_info = logbook.get_run_info(run_key)
            if log_info:
                run_data['LOG'] = log_info

    return result


def Keithley_xls_read(directory: str, **kwargs) -> Dict[str, Any]:
    """
    Read all .xls Keithley data files in a directory.
    For each data file, the best matching LOG file is automatically detected
    (using _find_matching_log) and used as the logbook.
    """
    directory = Path(directory)
    all_files = list(directory.glob("*.xls"))
    log_files = [f for f in all_files if "LOG" in f.stem.upper()]
    data_files = [f for f in all_files if "LOG" not in f.stem.upper()]

    results = {}
    for df in data_files:
        matched = _find_matching_log(df.stem, log_files)
        logbook = None
        if matched is not None:
            try:
                logbook = LogBook(str(matched))
            except Exception as e:
                logger.warning(f"Could not parse logbook {matched.name}: {e}")

        try:
            results[df.stem] = Keithley_xls_read_file(str(df), logbook)
        except Exception as e:
            logger.error(f"Error reading {df.name}: {e}")

    return results

def convert_log_xls_to_dict(xls_path: str) -> dict:
    """
    Returns a dict: { device_name: { '2port': {...}, '4port': {...} } }
    Uses original NW labels; common ground SMUs are labelled as their nanowire.
    """

    log = LogBook(xls_path)
    flat = log.data
    positions = flat.get('positions', {})

    # Physical column order and NW mapping from header
    smu_to_nw = {}          # SMU number → 'NW1'/'NW2'
    smu_order = []          # physical column order of SMU numbers
    for pos_key in sorted(positions.keys()):   # pos1, pos2, pos3, pos4
        info = positions[pos_key]
        smu_num = int(info['SMU'].replace('SMU', ''))
        nw = info.get('NW')
        smu_to_nw[smu_num] = nw
        smu_order.append(smu_num)

    device_name = None
    runs_2port = {}
    runs_4port = {}

    for key, entry in flat.items():
        if key == 'positions' or key.startswith('pos'):
            continue
        if not key.isdigit():
            continue

        if device_name is None:
            device_name = entry.get('Device', 'Unlabelled')

        # Build SMU1..SMU4 assignments
        smu_assignments = {}
        for smu_num in range(1, 5):
            raw = entry.get(f'SMU{smu_num}', 'NA')
            if isinstance(raw, float) and raw == 0.0:
                raw = 'NA'
            raw = str(raw).strip().lower()

            # NA or empty → null (not used)
            if raw in ('na', ''):
                role = None
            else:
                # Any other value (sweep, pulse, common ground, etc.)
                # → assign the nanowire label from the header
                nw = smu_to_nw.get(smu_num)
                role = nw if nw else None   # fallback to None if unknown
            smu_assignments[f'SMU{smu_num}'] = role

        # Determine measurement type: presence of both NW1 and NW2 → 4‑port
        present_nws = set(v for v in smu_assignments.values() if v is not None)
        mtype = '4port' if ('NW1' in present_nws and 'NW2' in present_nws) else '2port'

        # SMU Order = all SMU channels (physical order) that have a non‑null assignment
        smu_order_list = [
            n for n in smu_order
            if smu_assignments.get(f'SMU{n}') is not None
        ]

        light_mic = entry.get('Light Microscope', 0)
        try:
            light_mic = bool(int(light_mic))
        except:
            light_mic = False

        run_info = {
            **smu_assignments,               # SMU1 … SMU4 (NW1/NW2 or null)
            'SMU Order': smu_order_list,
            'LightMicroscope': light_mic,
            'Comment': ''
        }

        if mtype == '2port':
            runs_2port[key] = run_info
        else:
            runs_4port[key] = run_info

    if device_name is None:
        device_name = 'UnknownDevice'

    return {device_name: {'2port': runs_2port, '4port': runs_4port}}


# ----------------------------------------------------------------------
# Process one folder: merge all LOG.xls into a single aggregate JSON
# ----------------------------------------------------------------------
def process_folder(folder: Path):
    """Convert all LOG.xls files in the folder and save an aggregated JSON."""
    xls_files = list(folder.glob("*.xls"))
    log_files = [f for f in xls_files if "LOG" in f.stem.upper()]
    data_files = [f for f in xls_files if "LOG" not in f.stem.upper()]

    if not log_files or not data_files:
        return

    # Common prefix, truncated at the last underscore
    stems = [strip_measurement_suffix(f.stem).replace('-', '_') for f in data_files]
    common = find_common_prefix(stems)
    if common:
        if '_' in common:
            common = common.rsplit('_', 1)[0]      # keep only up to last segment
    else:
        common = Path(data_files[0].stem).split('_')[0]
    common = common.rstrip('_')

    # Merge all logbooks in this folder
    aggregated = {}
    for log_path in log_files:
        try:
            dev_dict = convert_log_xls_to_dict(str(log_path))
            for device, content in dev_dict.items():
                if device in aggregated:
                    for mtype in ['2port', '4port']:
                        aggregated[device].setdefault(mtype, {}).update(content.get(mtype, {}))
                else:
                    aggregated[device] = content
        except Exception as e:
            print(f"Error converting {log_path}: {e}")

    if not aggregated:
        return

    output_name = f"{common}_LOG.json"
    output_path = folder / output_name
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)
    print(f"Created {output_path}")


# ----------------------------------------------------------------------
# Walk the entire directory tree and process every folder
# ----------------------------------------------------------------------
def aggregate_all_logs(root_dir: str,skip_dupes=True):
    """
    Example use:
        root_folder_1 = "/home/vidar/mnt/Box/PhD/Lab Data/Device Data/Kiethley Probe Station/DFR1_RAW_DATA/"
        root_folder_2 = "/home/vidar/mnt/Box/PhD/Lab Data/Device Data/Kiethley Probe Station/DFR2_RAW_DATA/"

        aggregate_all_logs(root_folder_1)
        aggregate_all_logs(root_folder_2)
    """
    root = Path(root_dir)
    for current_dir, dirs, files in os.walk(root):
        # Process only if the folder contains .xls files
        
        if any(f.endswith('.xls') for f in files):
            if skip_dupes:
                if any(f.endswith('.json') for f in files):
                    print("Json already present in " + current_dir)
                    continue
            
            process_folder(Path(current_dir))



# ----------------------------------------------------------------------
# Helper: group name from full device name
# ----------------------------------------------------------------------
def get_device_group(device_name: str) -> str:
    """Return the group prefix (everything before the last underscore)."""
    # Normalise dashes to underscores for consistency
    normalised = device_name.replace('-', '_')
    if '_' in normalised:
        return normalised.rsplit('_', 1)[0]
    return normalised

# ----------------------------------------------------------------------
# Scan root and build group → subdevice → log_info mapping
# ----------------------------------------------------------------------
def discover_device_groups(root_dir: str) -> Dict[str, Dict[str, Dict]]:
    """
    Walk root_dir, find all *_LOG.json files, and build:
        group -> {
            subdevice_name -> {
                'log_path': str,          # path to the log JSON
                'runs': {                 # all runs for this subdevice
                    run_number: {         # e.g. '758'
                        '2port': {...},   # run info from JSON
                        '4port': {...}
                    }
                }
            }
        }
    """
    root = Path(root_dir)
    groups = {}
    
    for log_file in root.rglob('*_LOG.json'):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {log_file}: {e}")
            continue
        
        # Each JSON has top-level device names
        for device, content in data.items():
            group = get_device_group(device)
            # Subdevice is the last part after the group prefix + underscore
            # e.g. device = DFR1_GG_BL1, group = DFR1_GG, sub = BL1
            normalised_device = device.replace('-', '_')
            if normalised_device.startswith(group + '_'):
                sub = normalised_device[len(group)+1:]
            else:
                sub = normalised_device  # fallback
            
            if group not in groups:
                groups[group] = {}
            if sub not in groups[group]:
                groups[group][sub] = {
                    'log_path': str(log_file),
                    'runs': {}
                }
            # Merge runs from both 2port and 4port
            for mtype in ['2port', '4port']:
                runs = content.get(mtype, {})
                for run_num, run_info in runs.items():
                    groups[group][sub]['runs'][run_num] = {
                        'measurement_type': mtype,
                        **run_info
                    }
    return groups

# ----------------------------------------------------------------------
# Load full measurement data for a subdevice
# ----------------------------------------------------------------------
def load_subdevice_data(log_path: str, device_name: str) -> Dict[str, Any]:
    """
    Given the path to the LOG.json and the device name,
    find the matching .xls files in that folder and load all runs.
    Returns dict: run_key -> run_data (with 'LOG' already inserted).
    """
    folder = Path(log_path).parent
    # Find .xls files that belong to this device
    # Normalise device name to match file naming conventions
    device_pattern = device_name.replace('-', '_')
    xls_files = [
        f for f in folder.glob('*.xls')
        if 'LOG' not in f.stem.upper()
        and device_pattern in f.stem.upper().replace('-', '_')
    ]
    if not xls_files:
        return {}

    # We'll use your existing Keithley_xls_read_file but pass logbook=None
    # because we'll inject the log info ourselves from the aggregated data.
    # However, we need the function imported.

    all_runs = {}
    for xls_path in xls_files:
        try:
            file_data = Keithley_xls_read_file(str(xls_path), logbook=None)
        except Exception as e:
            print(f"Error reading {xls_path}: {e}")
            continue
        if file_data is None:
            continue
        # Enrich with log info later when we merge
        for run_key, run_data in file_data.items():
            all_runs[run_key] = run_data
    return all_runs

# ----------------------------------------------------------------------
# Build aggregated data for a specific group (with cache)
# ----------------------------------------------------------------------
def build_group_data(root_dir: str, group: str,
                     force_reload: bool = False,
                     cache_dir: Optional[str] = None) -> Dict[str, Dict]:
    """
    For the given group, load all measurement data for all its subdevices.
    Returns dict: { subdevice_name -> { run_key: run_data } }
    The run_data already contains the 'LOG' field from the log JSON.
    
    Caching: if cache_dir is provided, a pickle file <group>_data.pkl is saved/loaded.
    """
    if cache_dir is None:
        cache_dir = root_dir
    cache_path = Path(cache_dir) / f"{group}_data.pkl"

    # Try to load from cache if not forced
    if not force_reload and cache_path.exists():
        # Check if any log file is newer than the cache
        cache_mtime = cache_path.stat().st_mtime
        # We need the list of log files used for this group
        groups = discover_device_groups(root_dir)
        if group in groups:
            subdevices = groups[group]
            log_files = set(info['log_path'] for info in subdevices.values())
            if all(Path(lf).stat().st_mtime <= cache_mtime for lf in log_files):
                print(f"Loading cached data from {cache_path}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

    # Load fresh data
    groups = discover_device_groups(root_dir)
    if group not in groups:
        print(f"Group '{group}' not found.")
        return {}

    subdevices = groups[group]
    aggregated = {}

    for sub, info in subdevices.items():
        log_path = info['log_path']
        device_name = f"{group}_{sub}".replace('_', '-')  # reconstruct original device name
        # Load measurement data
        runs = load_subdevice_data(log_path, device_name)
        # Now attach log metadata to each run
        for run_num, run_info in info['runs'].items():
            run_key = f"Run{run_num}" if not run_num.startswith("Run") else run_num
            if run_key in runs:
                runs[run_key]['LOG'] = run_info
        aggregated[sub] = runs

    # Save cache
    if cache_dir:
        with open(cache_path, 'wb') as f:
            pickle.dump(aggregated, f)
        print(f"Cached data saved to {cache_path}")

    return aggregated

# ----------------------------------------------------------------------
# Convenience: list all groups
# ----------------------------------------------------------------------
def list_groups(root_dir: str) -> List[str]:
    groups = discover_device_groups(root_dir)
    return sorted(groups.keys())

