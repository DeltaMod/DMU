# -*- coding: utf-8 -*-
"""
Lumerical Data Handling
Created on Tue Aug 18 17:06:05 2020
@author: DeltaMod
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
import ast
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


def _convert(val):
    if isinstance(val, list):
        return [_convert(x) for x in val]
    if isinstance(val, tuple):
        return tuple(_convert(x) for x in val)
    if isinstance(val, dict):
        return {k: _convert(v) for k, v in val.items()}
    if isinstance(val, str):
        for typ in (int, float):
            try:
                return typ(val)
            except ValueError:
                continue
        # optional bool/None
        if val.lower() == 'true': return True
        if val.lower() == 'false': return False
        if val.lower() == 'none': return None
        return val
    return val

def convert_values(d):
    out = {}
    for k, v in d.items():
        try:
            parsed = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            parsed = v
        out[k] = _convert(parsed)
    return out

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

# -----------------------------------------------------------------------------
# Run-key normalisation
# -----------------------------------------------------------------------------
_RUN_DIGITS_RE = re.compile(r'(\d+)')


def _run_number(key: Any) -> Optional[str]:
    """
    Extract the numeric run number and format strictly as 'Runxx'.
    
    Examples:
        '65' -> 'Run65', 'run65' -> 'Run65', 'Run 65' -> 'Run65', 65 -> 'Run65'
    """
    if key is None:
        return None
    s = str(key).strip()
    if not s:
        return None
        
    # If it's pure digits (e.g., "65" or 65), convert to "Run65"
    if s.isdigit():
        return f"Run{s}"
        
    # Prefer digits that follow 'run' (e.g., "run65" -> "Run65")
    m = re.search(r'run[\s_\-]*(\d+)', s, re.IGNORECASE)
    if m:
        return f"Run{m.group(1)}"
        
    # Fallback for isolated numbers if applicable
    m = re.search(r'(\d+)', s)
    if m:
        return f"Run{m.group(1)}"
        
    return None

# -----------------------------------------------------------------------------
# LOG file discovery & loading (JSON / TXT preferred, XLS as fallback)
# -----------------------------------------------------------------------------
def _match_log_in_list(data_clean: str, log_files: List[Path]) -> Optional[Path]:
    """
    Try the perfect '_..._LOG' candidate first (also after stripping common
    '2Term_' / '4Term_' prefixes), then the longest common prefix.
    """
    log_files = list(log_files)
    if not log_files:
        return None

    candidates: List[str] = [data_clean]
    stripped = re.sub(r'^(2term|4term)_', '', data_clean, flags=re.IGNORECASE)
    if stripped != data_clean:
        candidates.append(stripped)
    stripped2 = strip_measurement_suffix(data_clean)
    if stripped2 not in candidates:
        candidates.append(stripped2)

    # 1) perfect candidate
    for dc in candidates:
        if '_' in dc:
            candidate = dc.rsplit('_', 1)[0] + '_LOG'
            for lf in log_files:
                if lf.stem.replace('-', '_').upper() == candidate.upper():
                    return lf

    # 2) longest common prefix over all candidates
    best_match, best_len = None, 0
    for dc in candidates:
        min_len = max(1, len(dc) // 2)
        for lf in log_files:
            log_clean = lf.stem.replace('-', '_')
            common = os.path.commonprefix([dc, log_clean])
            if len(common) >= min_len and len(common) > best_len:
                best_len, best_match = len(common), lf
    return best_match


def _find_matching_log(data_stem: str, folder: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Find best LOG file for `data_stem` in `folder`.
    Preference: .json > .txt > .xls
    Returns (path, kind) or (None, None).
    """
    folder = Path(folder)
    data_clean = data_stem.replace('-', '_')

    json_logs = [f for f in folder.glob("*.json") if "LOG" in f.stem.upper()]
    txt_logs  = [f for f in folder.glob("*.txt")  if "LOG" in f.stem.upper()]
    xls_logs  = [f for f in folder.glob("*.xls")  if "LOG" in f.stem.upper()]

    for logs, kind in ((json_logs, 'json'), (txt_logs, 'txt'), (xls_logs, 'xls')):
        m = _match_log_in_list(data_clean, logs)
        if m is not None:
            return m, kind

    # Folder-level fallback: a single JSON/TXT LOG covers everything in the dir
    if len(json_logs) == 1:
        return json_logs[0], 'json'
    if len(txt_logs) == 1 and not json_logs:
        return txt_logs[0], 'txt'

    return None, None




def _load_log_source(path: Path):
    """
    Load a LOG file and return a callable: run_key -> Optional[dict].

    Supported formats:
      * .json / .txt  – either the aggregated structure
                        {device: {'2port': {run: {...}}, '4port': {...}}}
                        or a flat {run: info} dict.
      * .xls          – legacy LogBook.
    """
    suffix = path.suffix.lower()

    if suffix in ('.json', '.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Detect the aggregated 2port/4port layout
        is_aggregated = any(
            isinstance(v, dict) and ('2port' in v or '4port' in v)
            for v in content.values()
        )

        # Initialize flat outside the branches to guarantee correct scoping
        flat: Dict[str, Dict] = {}

        if is_aggregated:
            for device, dev_content in content.items():
                if not isinstance(dev_content, dict):
                    continue
                for mtype in ('2port', '4port'):
                    for run_num, run_info in dev_content.get(mtype, {}).items():
                        # Direct key normalization on ingestion
                        normalized_key = _run_number(run_num) or str(run_num)
                        flat[normalized_key] = {
                            **run_info,
                            'measurement_type': mtype,
                            'Device': device,
                        }
        else:
            for k, v in content.items():
                # Direct key normalization on ingestion
                normalized_key = _run_number(k) or str(k)
                flat[normalized_key] = v
            
        # Build a number-indexed map for loose runtime query fallback variations
        by_number: Dict[str, Dict] = {}
        for k, v in flat.items():
            n = _run_number(k)
            if n is not None:
                by_number.setdefault(n, v)
            
        def lookup(run_key: str):
            # 1. First normalize the incoming query to matching "Runxx" layout
            n = _run_number(run_key)
            
            # 2. Check the strictly normalized flat structure first
            if n in flat:
                return flat[n]
                
            # 3. Fallback check for raw text key match if normalization was empty
            if run_key in flat:
                return flat[run_key]
                
            # 4. Final loose variations dictionary lookup
            if n is not None:
                return by_number.get(n)
                
            return None
        
        return lookup

    if suffix == '.xls':
        logbook = LogBook(str(path))
        return logbook.get_run_info

    raise ValueError(f"Unsupported LOG file type: {suffix}")


''' Design Statutes 

We want to convert all data into pkl files so that we do not have to parse new data to save/modify it. This will also compress the data.
When loading data, if a .pkl exists, we should SKIP all other processing. There should be an option to create new and compare keys between the new data and the pkl (non destructive)
Each DEVICE gets its own .pkl, but then remains the question what constitutes a device? 
We will classify each measurement folder as its own device, meaning that any future measurements should be placed into the same folder (preferably with dates on the file)
If a device has subdevices, these must be in the logbook - otherwise, we will always assume that "device_x" has subdevice "SD0"

Raw Data dict key structure: filename -> RunID -> datakeys and settings
If there is a matching logfile, we must check the devicename/subdevicename/RunID against the Raw data, to create a data dict structure that replace the filename with device->subdevice (and then puts filename into the datakeys section like we normally do) 

Data dict key structure: numports->device->subdevice->RunID->[Data | Log | Settings] (note that Data is not a key, RunID simply contains e.g. "current/voltage" etc.)
We can completely ignore the name to determine the portnumber, as we can simmply read the length of the "Instrument" key in filename->runID->Settings->Instrument and use len(val)+"port" for this.
Why place numports at the front? no repeats, and all 2port measurements from all devices can be grouped

To be able to match the log files, we need the format to make sense. But we also need to be flexible... So we need the following edge cases:
We must accept: 
    device->runID ("SD0" to be subdevice key)
    runID->(force filename as device, and SD0 as subdevice key)

file loading order:
    check if pkl exists
    if recalculate is enabled, create temp dict from the xls and log (if exists), comparing and merging contents (if new subdevice, add entire contents, if new runID, add to subdevice)
    if recalculate is enabled, but no pkl exists, then we treat it as a first creation (see below)
    if not, create new from the following logic:
        load DATA xls files into a single dict (filename->RunID->datakeys and settings)
        check for log
        if log is xls, convert it to .json
        if log is txt, convert it from .txt to .json (if direct conversion does not work, simply copy it to a same named .json and delete the .txt)
        now that we have a .json, we load that as the log
        from the data filename key, we must: strip any dates (e.g. 2026-01-01_2Term_Devicename_subdevicename -> 2Term_Devicename_subdevicename)
        
        We often use the naming convention (2Term|3Term|4Term|2port|3port|4port) to separate measurements from each other, as they cannot be saved in the same xls. 
        Here, we run into some issues - as the individual xls files might actually have overlapping runIDs...
        
        Because I want the log to be easy to use, if you are exclusively using only 2port or 4port measurements (3port possible, so plan for reduncancy), then your run ID's should be easy to correlate.
        The default log format should be any of the following:
            
        numports->Device->Subdevice->Runnum (either e.g. "123" or "Run123")->{"SMU1":val,"SMU2":val,...comment} 
        Note that here, if only e.g. SMU1 and SMU2 are provided, SMU3 and SMU4 should be added with "SMU3":False,"SMU4":False. 
        If no "comment" is provided, one should be added (so we can access and edit it later in the GUI)
        If no "SMU Order" is provided, it should be presumed to be in numeric order [1,2,3,4] or [1,2] if only SMU1->SMU2
        Ideally, these can be exposed in the GUI to be edited.
        If not present, the field "Light Microscope" can be set to "unknown" (I'd prefer this to be a bool, but you can't set unknown as a bool)
        Also, it should work if the log just contains RunID's. In this case, we fallback on the Filename + SD0 I described earlier, but of course add numports based on the Instrument field. 
        More importantly, if NO LOG IS PROVIDED, we should build it from the data we have, but fill the comment field with "NO LOG" and the SMU's measured on with "Probe1" and "Probe2" in order
        
        Finally, the old log conversion (from old XLS, further down) needs to be rewritten to respect this. 
        There needs to be a GUI button that allows wildcard forms to identify device and subdevice.
        
        Example: DFR1-GG-BL1 -> Device: DFR1-GG, subdevice: BL1 (this conversion process should let the user know if it could not cleanly find these names, or was able to split it, and in that case each such instance should be fixable by the user with a name entry exceptions) 
        

'''    
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
                run_key = sheet_name

                
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
                cols['_sheet_name'] = sheet_name          # keep original for debugging
                cols['_run_key'] = run_key                # 'Run65'
                if run_key in file_data:
                    logger.warning(f"Duplicate run key {run_key} in {file_path}; overwriting")
                file_data[run_key] = cols

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
        stats = convert_values(stats)
        
        # Choose the largest valid value and its index
        max_npts = max([val if type(val)==int else 0 for val in stats["Npts"]])
        main_col = stats["Npts"].index(max_npts) if max_npts > 0 else 0
    
        # If no valid Npts found, fall back to the length of the current column
        max_npts = 0
        if max_npts == 0:
            if 'current' in cols and isinstance(cols['current'], list):
                max_npts = len(cols['current'])
                print(max_npts)
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




# =============================================================================
# Public API
# =============================================================================


def Keithley_xls_read_file(file_path: str,
                           logbook: Optional[LogBook] = None,
                           log_source: Optional[Any] = None) -> Dict[str, Any]:
    """
    Read a single Keithley .xls file and associate logbook data.

    LOG resolution priority:
      1. explicit `log_source` callable (run_key -> dict), if supplied
      2. explicit `logbook` (legacy)
      3. auto-discovered LOG file in the same folder (JSON > TXT > XLS)
    """
    if log_source is None and logbook is None:
        matched, kind = _find_matching_log(Path(file_path).stem, Path(file_path).parent)
        if matched is not None:
            try:
                log_source = _load_log_source(matched)
                
                logger.info(f"Using LOG file: {matched.name} ({kind})")
                
            except Exception as e:
                logger.warning(f"Could not parse logbook {matched.name}: {e}")

    if log_source is None and logbook is not None:
        log_source = logbook.get_run_info
    
    reader = KeithleyDataReader(logbook)
    
    file_data = reader.read_file(file_path)
    result: Dict[str, Any] = {}

    if "Settings" in file_data:
        settings_dict = file_data.pop("Settings")
        for run_key, run_data in file_data.items():
            run_data['Settings'] = settings_dict.get(run_key, {})
            result[run_key] = run_data
    else:
        result = dict(file_data)
    
    if log_source is not None:
        
        for run_key, run_data in result.items():
            
            log_info = log_source(run_key)
            if log_info:
                result[run_key]['LOG'] = log_info  
                
    return result


def Keithley_xls_read(directory: str, **kwargs) -> Dict[str, Any]:
    """
    Read every Keithley .xls data file in a directory.

    LOG files are auto-discovered per data file, preferring:
        *_LOG.json  >  *_LOG.txt  >  *_LOG.xls
    """
    directory = Path(directory)
    data_files = [
        f for f in directory.glob("*.xls")
        if "LOG" not in f.stem.upper()
    ]

    results: Dict[str, Any] = {}
    for df in data_files:
        try:
            results[df.stem] = Keithley_xls_read_file(str(df))
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

