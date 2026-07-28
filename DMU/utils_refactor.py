# -*- coding: utf-8 -*-
"""
Lumerical Data Handling
Created on Tue Aug 18 17:06:05 2020
@author: Vidar Flodgren
Github: https://github.com/DeltaMod
"""
import os
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
# LogBook class (unchanged)
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
        cols["col headers"] = []
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

        cols["column_order"] = header_order   # preserve order
        # keep all raw columns
        for h in header_order:
            cols[h] = col_dict[h]

        # ---- Identify main applied voltage and measured current ----
        applied_col = None
        measured_col = None

        # 1) Try using settings "Colname"
        colnames = settings.get("Colname")
        if colnames is not None:
            if isinstance(colnames, str):
                colnames = [colnames]
            for name in colnames:
                if name in col_dict:
                    applied_col = name
                    # The measured column is the one immediately before applied in order
                    # because order is measured|applied|measured|applied...
                    idx = header_order.index(applied_col)
                    if idx > 0:
                        measured_col = header_order[idx - 1]
                    break

        # 2) If not found, try heuristic (ends with 'V' or contains "voltage")
        if applied_col is None:
            for h in header_order:
                if h.endswith('V') and not any(s in h for s in ["START", "STOP"]):
                    applied_col = h
                    break
                elif "voltage" in h.lower():
                    applied_col = h
                    break
            if applied_col is not None:
                idx = header_order.index(applied_col)
                if idx > 0:
                    measured_col = header_order[idx - 1]

        # 3) Fallback: first two columns as (measured, applied)
        if applied_col is None and len(header_order) >= 2:
            applied_col = header_order[1]
            measured_col = header_order[0]

        if applied_col is None or measured_col is None:
            logger.warning(f"Could not identify applied/measured columns in {sheet_name}")
            return None

        cols['voltage'] = col_dict.get(applied_col, [])
        cols['current'] = col_dict.get(measured_col, [])
        # Store which columns were chosen
        cols['applied_col'] = applied_col
        cols['measured_col'] = measured_col

        return cols

    def _process_run_data(self, cols: Dict, stats: Dict, sheet_name: str, file_path: str):
        # ensure lists
        if not isinstance(stats['Npts'], list):
            for key in list(stats.keys()):
                if not isinstance(stats[key], list):
                    stats[key] = [stats[key]]

        npts_values = [float(x) for x in stats['Npts'] if isinstance(x, (int, float))]
        if not npts_values:
            logger.warning(f"No valid Npts in {sheet_name}, skipping sweep processing")
            return

        main_col = stats['Npts'].index(max(npts_values))
        npts = int(stats['Npts'][main_col])

        try:
            exec_time = stats.get('Execution Time', '0:00:00')
            time_per_point = get_sec(exec_time) / max(npts_values) if npts_values else 0
        except:
            time_per_point = 0
        stats["Time Per Point"] = time_per_point
        stats["NWID"] = ["NW1", "NW1", "NW2", "NW2"]

        try:
            cols["Time"] = np.linspace(0, npts * time_per_point, npts)
        except:
            cols["Time"] = np.linspace(0, npts, npts)

        if isinstance(stats["VStep"][main_col], str):
            stats["VStep"][main_col] = float(stats["VStep"][main_col])

        try:
            if (stats["FBSweep"][main_col] == True and
                npts == 2 * (1 + int(abs(stats["VStart"][main_col] - stats["VStop"][main_col]) / abs(stats["VStep"][main_col])))):
                sweep_indices = [0, int(npts / 2), npts]
            else:
                sweep_indices = [0, max(npts_values)]
        except:
            sweep_indices = [0, npts]

        print(f"{os.sep.join(file_path.split(os.sep)[-3:])} - {sheet_name}: {stats['Operation Mode'][main_col]}")

        if "Voltage Linear Sweep" in stats["Operation Mode"]:
            list_keys = [k for k, v in cols.items() if isinstance(v, list) and k not in ("col headers", "Data directory", "column_order")]
            for k in list_keys:
                cols[k] = segment_sweep(cols[k], sweep_indices)

        if ("Voltage List Sweep" in stats["Operation Mode"] and
            "curing" in stats["Test Name"].lower()):
            list_keys = [k for k, v in cols.items() if isinstance(v, list) and k not in ("col headers", "Data directory", "column_order")]
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

# =============================================================================
# Public API
# =============================================================================
def Keithley_xls_read_file(file_path: str, logbook: Optional[LogBook] = None) -> Dict[str, Any]:
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
    directory = Path(directory)
    all_files = list(directory.glob("*.xls"))
    log_files = [f for f in all_files if "LOG" in f.stem.upper()]
    data_files = [f for f in all_files if "LOG" not in f.stem.upper()]

    logbook = None
    if log_files:
        try:
            logbook = LogBook(str(log_files[0]))
        except:
            logger.warning("Could not parse logbook, proceeding without.")

    results = {}
    for df in data_files:
        try:
            results[df.stem] = Keithley_xls_read_file(str(df), logbook)
        except Exception as e:
            logger.error(f"Error reading {df.name}: {e}")
    return results