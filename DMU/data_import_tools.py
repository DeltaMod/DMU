# -*- coding: utf-8 -*-
"""
Lumerical Data Handling
Created on Tue Aug 18 17:06:05 2020
@author: DeltaMod
Github: https://github.com/DeltaMod
"""

# ---------- Standard library ----------
import os
import sys
import re
import ast
import copy
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Optional, Any

# ---------- Third-party ----------
import numpy as np
import xlrd

# ---------- Local packages ----------
try:
    from .custom_logger import get_custom_logger
    from .plot_utils import *      # noqa: F401,F403  (kept for external wildcard consumers)
    from .utils_utils import *     # noqa: F401,F403
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from DMU.custom_logger import get_custom_logger
    from DMU.plot_utils import *     # noqa: F401,F403
    from DMU.utils_utils import *    # noqa: F401,F403


logger = get_custom_logger("DMU_UTILS")


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
#  Keithley data importer Helper functions
# =============================================================================

KNOWN_MATERIALS = {
    "ALD":    "Al2O3",
    "PMMA":   "PMMA",
    "Al2O3":  "Al2O3",
    "S1813":  "S1813",
}

def detect_material_from_path(path) -> str:
    """Scan every path component (case-insensitively) for a known
    material flag. First match wins. Returns 'AIR' if nothing matches."""
    for part in Path(path).parts:
        lowered = part.lower()
        for flag, material in KNOWN_MATERIALS.items():
            if flag.lower() in lowered:
                return material
    return "AIR"

def migrate_logs_to_json(root_dir, force=False):
    """Walk root_dir, convert every *LOG*.xls to a sibling .json using the
    existing convert_log_xls_to_dict(). Skips files whose .json already
    exists unless force=True. Returns list of (xls, json) pairs written."""
    root = Path(root_dir)
    written = []
    for xls in root.rglob("*LOG*.xls"):
        json_path = xls.with_suffix(".json")
        if json_path.exists() and not force:
            continue
        data = convert_log_xls_to_dict(xls)   # your existing converter
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        written.append((xls, json_path))
    return written

def _parse_log_text(text: str, path: Path) -> tuple[dict, str]:
    """Return (data, status).

    status is 'strict_json' when the raw text parsed as strict JSON.
    Anything else means a fallback was used and the file should be
    migrated to the canonical (strict-JSON) format.
    """
    try:
        return json.loads(text), "strict_json"
    except json.JSONDecodeError as json_exc:
        first_exc = json_exc

    try:
        return ast.literal_eval(text), "python_literal"
    except (ValueError, SyntaxError):
        pass

    try:
        return (json.loads(_normalise_lax_json(text, "json")),
                "normalised_json")
    except json.JSONDecodeError:
        pass

    try:
        return (ast.literal_eval(_normalise_lax_json(text, "py")),
                "normalised_python")
    except (ValueError, SyntaxError):
        pass

    raise LogParseError(
        f"{path.name}: {first_exc.msg} at line "
        f"{first_exc.lineno} col {first_exc.colno}",
        lineno=first_exc.lineno,
        colno=first_exc.colno,
        original=first_exc,
    )

def _migrate_log_file(path: Path, parsed_data: dict,
                      problems: Optional[list] = None) -> None:
    """Back up a non-strict-JSON log and write a canonical version in its
    place. Safe to call multiple times — subsequent calls hit 'strict_json'
    and never reach this function."""
    # Backup name: replace 'LOGBOOK' with 'BACKUP_LGF' (LGF does NOT
    # contain 'LOG', so the backup will never match find_log_file's
    # *LOG*.json glob → no confusion about which file to load).
    name = path.name
    backup_name = re.sub(r"LOGBOOK", "BACKUP_LGF", name, flags=re.IGNORECASE)
    if backup_name == name:
        backup_name = path.stem + "_BACKUP_LGF" + path.suffix

    backup_path = path.parent / backup_name

    # Don't clobber a pre-existing backup; append _N suffix
    if backup_path.exists():
        i = 1
        while True:
            candidate = path.parent / (
                f"{backup_path.stem}_{i}{backup_path.suffix}")
            if not candidate.exists():
                backup_path = candidate
                break
            i += 1

    # Move original → backup (atomic rename within same dir)
    os.replace(str(path), str(backup_path))

    # Write canonical version at the original path
    canonical = _canonicalize_log_data(parsed_data)

    def _json_default(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, (set, tuple)): return list(obj)
        raise TypeError(
            f"Cannot serialize {type(obj).__name__} to JSON")

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(canonical, f, indent=2, ensure_ascii=False,
                      default=_json_default)
    except Exception:
        # Write failed — restore the backup so we don't lose data
        if path.exists():
            path.unlink()
        os.replace(str(backup_path), str(path))
        raise

    if problems is not None:
        problems.append({
            "severity": "info",
            "category": "log_migrated",
            "folder":   str(path.parent.resolve()),
            "path":     str(path),
            "run":      None,
            "message":  (f"Log was not strict JSON — original backed up to "
                         f"{backup_path.name}, canonical version written to "
                         f"{path.name}. Missing fields "
                         f"(SMU1–4, SMU Order, LightMicroscope, Comment) "
                         f"were filled with defaults."),
            "line":     None,
            "column":   None,
        })
def _canonicalize_log_data(data: dict) -> dict:
    """Return a copy of the log dict with every run info normalised:
    SMU1..SMU4 present, SMU Order computed, LightMicroscope + Comment
    defaulted, Device/Subdevice/RunID ensured."""
    out: dict = {}
    for np_key, devices in data.items():
        if not isinstance(devices, dict):
            out[np_key] = devices
            continue
        new_devs = {}
        for dev, subs in devices.items():
            if not isinstance(subs, dict):
                new_devs[dev] = subs
                continue
            new_subs = {}
            for sub, runs in subs.items():
                if not isinstance(runs, dict):
                    new_subs[sub] = runs
                    continue
                new_runs = {}
                for run_key, run_info in runs.items():
                    if not isinstance(run_info, dict) or not _is_log_info(run_info):
                        new_runs[run_key] = run_info
                        continue
                    info = normalise_log_info(dict(run_info))
                    info.setdefault("Device", dev)
                    info.setdefault("Subdevice", sub)
                    info.setdefault("RunID", normalise_run_key(run_key))
                    new_runs[run_key] = info
                new_subs[sub] = new_runs
            new_devs[dev] = new_subs
        out[np_key] = new_devs
    return out

def save_device_data(folder: Path, data: dict, cache_name: Optional[str] = None) -> Path:
    """Write the device_data dict back to its pkl. Returns the pkl path."""
    folder = Path(folder)
    cache_path = folder / (cache_name or f".dmui_{folder.name}.pkl")
    _save_pickle(cache_path, data)
    return cache_path

def _split_device_subdevice(name: str) -> tuple[str, str]:
    """
    Best-effort split of a legacy device string into (device, subdevice).

    Convention assumed: the subdevice is the trailing segment, and we only
    split when there are at least two separators (so that 'DFR1-GG' stays
    whole and does not become device='DFR1', sub='GG').

        'DFR1-GG-BL1'   -> ('DFR1-GG', 'BL1')
        'DFR1_GG_BL1'   -> ('DFR1_GG', 'BL1')
        'DFR1-GG'       -> ('DFR1-GG', 'SD0')
        'DFR1'          -> ('DFR1', 'SD0')
        ''              -> ('Unlabelled', 'SD0')

    This is only a default. The GUI is expected to let the user override
    the split with a wildcard/exception table later.
    """
    s = str(name).strip()
    if not s:
        return "Unlabelled", "SD0"

    for sep in ("-", "_"):
        if s.count(sep) >= 2:
            head, tail = s.rsplit(sep, 1)
            if head and tail:
                return head, tail

    return s, "SD0"


def convert_log_xls_to_dict(xls_path: Path) -> dict:
    """
    Convert a legacy .xls LOG book straight into the new structure:

        numports -> device -> subdevice -> RunID -> info

    RunID is normalised to 'Runxx'. Device/Subdevice are split from the
    'Device' column of the XLS with _split_device_subdevice().

    SMU1..SMU4 hold the nanowire label ('NW1' / 'NW2') when the SMU is in
    use, and False when it is not. 'SMU Order' is the physical column order
    of the used SMUs, and numports is len('SMU Order').
    """
    log = LogBook(str(xls_path))
    flat = log.data
    positions = flat.get("positions", {})

    # Physical column order and NW label per SMU, taken from the header.
    smu_to_nw: dict[int, Optional[str]] = {}
    physical_order: list[int] = []
    for pos_key in sorted(positions.keys()):
        info = positions[pos_key]
        smu_num = int(str(info["SMU"]).replace("SMU", ""))
        smu_to_nw[smu_num] = info.get("NW")
        physical_order.append(smu_num)

    result: dict = {}

    for key, entry in flat.items():
        if not str(key).isdigit():
            continue

        raw_device = str(entry.get("Device") or "Unlabelled")
        device, subdevice = _split_device_subdevice(raw_device)
        run_id = normalise_run_key(key)

        # SMU1..SMU4 -> nanowire label or False.
        smu_assignments: dict[str, Any] = {}
        for smu_num in range(1, 5):
            raw = entry.get(f"SMU{smu_num}", "NA")
            if isinstance(raw, float) and raw == 0.0:
                raw = "NA"
            raw_s = str(raw).strip().lower()

            if raw_s in ("na", "", "nan", "none"):
                smu_assignments[f"SMU{smu_num}"] = False
            else:
                smu_assignments[f"SMU{smu_num}"] = smu_to_nw.get(smu_num) or False

        smu_order_used = [
            n for n in physical_order
            if smu_assignments.get(f"SMU{n}") not in (False, None)
        ]
        numports = len(smu_order_used) if smu_order_used else 2

        light_mic = entry.get("Light Microscope", "unknown")
        if light_mic in (0, "0", False):
            light_mic = False
        elif light_mic in (1, "1", True):
            light_mic = True
        # else: leave as "unknown"

        run_info = {
            **smu_assignments,
            "SMU Order": smu_order_used,
            "LightMicroscope": light_mic,
            "Comment": entry.get("Comment", ""),
            "Device": device,
            "Subdevice": subdevice,
            "RunID": run_id,
        }

        result.setdefault(numports, {}) \
              .setdefault(device, {}) \
              .setdefault(subdevice, {})[run_id] = run_info

    return result

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

def get_sec(time_str: str) -> float:
    """
    Converts time of the format dd:hh:mm:ss into seconds, or bluntly assumes you just gave it a string of seconds.
    """
    if type(time_str) == float:
        logger.warning("Provided TIME is a float... Returning it as if it were seconds")
        return(time_str)
    if len(time_str.split(":")) == 1:
        try:
            seconds = float(time_str[0])
        except:
            logger.warning("Provided TIME is neither in d:h:m:s format, or a float convertable string")
            return 0
    sec_per_par = [86400,3600,60,1]
    try:
        seconds = 0
        for i,val in enumerate(map(float, time_str.split(':'))):
            seconds += val*sec_per_par[i]
        return seconds
    except:
        logger.warning("Provided TIME is not a string in d:h:m:s format, please double check your data sheet.")
        return 0

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

# Keithley Helpers
def _convert_cell(value):
    """
    Convert a single XLS cell value to its natural Python type.

    Strings that are valid Python literals (int, float, bool, None, list,
    tuple, dict, quoted string) are parsed. Anything that isn't a valid
    literal is returned as a stripped string.

    Examples:
        "2"                     -> 2          (int)
        "2.25"                  -> 2.25       (float)
        "1e5"                   -> 100000.0   (float)
        "'hello'"               -> "hello"    (str, quoted)
        "['test', 2]"           -> ['test', 2]
        "['thing', 2.25]"       -> ['thing', 2.25]
        "SMU1"                  -> "SMU1"     (str, unquoted)
        "N/A"                   -> "N/A"      (str)
        "Enabled"               -> "Enabled"  (str, then NEWDEF maps it)
    """
    if not isinstance(value, str):
        return value

    s = value.strip()
    if not s:
        return s

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _convert_cell_list(values):
    """Apply _convert_cell to every element of an iterable."""
    return [_convert_cell(v) for v in values]

def _apply_newdef(value, mapping):
    """
    Apply the NEWDEF substitution map safely.

    - Lists are recursed element-wise.
    - Dicts and other unhashable containers are returned untouched
      (they can never be keys in NEWDEF anyway).
    - Everything else is looked up normally, falling back to itself.
    """
    if isinstance(value, list):
        return [_apply_newdef(v, mapping) for v in value]
    if isinstance(value, (dict, set)):
        return value
    try:
        return mapping.get(value, value)
    except TypeError:
        return value
    
#Main funct
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
                if not values:
                    continue
    
                # Convert each cell up-front. This is what stops
                # "['SMU1', 'SMU3']" from surviving as a string, and turns
                # "2" / "2.25" into real ints / floats.
                converted = _convert_cell_list(values)
    
                if len(converted) == 1:
                    settings_data[current_run][header] = converted[0]
                else:
                    settings_data[current_run][header] = converted
    
        # ---- post-processing (formulas, key renaming, definition map) ----
        for run_no, run_settings in settings_data.items():
            run_settings["Formulas"] = {}
            for key in list(run_settings.keys()):
                # Safe NEWDEF substitution for scalars, lists, and stray dicts.
                run_settings[key] = _apply_newdef(run_settings[key], self.newdef)
        
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
        
        
        # We no longer split the raw data, but rather we tell it where the indices of turning points are. This means we can check this later in the plotting tool to plot all as separate, or just the first/second etc.
        sweep_segments: list[tuple[int, int]] = []

        # ---------- Segment sweep for linear sweeps ----------
        if ("Voltage Linear Sweep" in stats["Operation Mode"]and len(sweep_indices) > 2):
            
            sweep_segments = [
                (int(sweep_indices[i]), int(sweep_indices[i + 1]))
                for i in range(len(sweep_indices) - 1)
            ]
    
        # ---------- Special handling for curing sweeps ----------
        elif ("Voltage List Sweep" in stats["Operation Mode"] and "curing" in stats["Test Name"].lower()):
            tps = turning_points(cols["voltage"])
            sweep_segments = [
                (int(tps[i]), int(tps[i + 1]))
                for i in range(len(tps) - 1)]
            
        cols["sweep_segments"] = sweep_segments

    
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
# New Keithley / LOG aggregation layer
# =============================================================================

# ---------------------------------------------------------------------------
# Run-key and instrument helpers
# ---------------------------------------------------------------------------

def normalise_run_key(key: Any) -> str:
    """Normalise run keys to 'Runxx' where possible."""
    s = str(key).strip()
    m = re.search(r'(\d+)', s)
    if m:
        return f"Run{m.group(1)}"
    return s


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def get_instrument_list(run_data: dict) -> list[str]:
    """Return the raw Instrument/SMU list from a run's Settings."""
    settings = run_data.get("Settings", {}) or {}
    instruments = settings.get("SMU") or settings.get("Instrument") or []
    return [str(x) for x in _as_list(instruments) if str(x).strip()]


def get_smu_order(run_data: dict) -> list[int]:
    """
    Determine SMU order from the Instrument list.
    Example: ['SMU1', 'SMU3'] -> [1, 3]
    """
    order = []
    for inst in get_instrument_list(run_data):
        m = re.search(r'(\d+)', inst)
        if m:
            order.append(int(m.group(1)))
    if not order:
        order = [1, 2]
    return order


def get_num_ports(run_data: dict) -> int:
    """Number of ports is the length of the Instrument list."""
    return len(get_smu_order(run_data))


# ---------------------------------------------------------------------------
# Fallback LOG creation
# ---------------------------------------------------------------------------

def make_fallback_log(run_data: dict) -> dict:
    """
    Build a LOG dict when no log file exists.
    SMU order comes from Settings['SMU'] / Settings['Instrument'].
    The first SMU in that list becomes Probe1, the second Probe2, etc.
    """
    order = get_smu_order(run_data)
    info = {f"SMU{i}": False for i in range(1, 5)}

    for probe_idx, smu_num in enumerate(order, start=1):
        if 1 <= smu_num <= 4:
            info[f"SMU{smu_num}"] = f"Probe{probe_idx}"

    info["SMU Order"] = order
    info["Comment"] = "NO LOG"
    info["LightMicroscope"] = "unknown"
    return info


def normalise_log_info(info: dict, run_data: Optional[dict] = None,
                       default_comment: str = "") -> dict:
    """Ensure a LOG dict has all expected keys."""
    if info is None:
        info = {}

    out = dict(info)

    for i in range(1, 5):
        out.setdefault(f"SMU{i}", False)

    order = out.get("SMU Order") or out.get("SMU_Order") or out.get("smu_order")
    if not order:
        order = [
            i for i in range(1, 5)
            if out.get(f"SMU{i}") not in (False, None, "")
        ]

    if not order and run_data is not None:
        order = get_smu_order(run_data)

    out["SMU Order"] = [
        int(x) for x in _as_list(order)
        if str(x).strip().lstrip("-").isdigit()
    ]
    out["Comment"] = out.get("Comment", out.get("comment", default_comment))
    out["LightMicroscope"] = out.get(
        "LightMicroscope",
        out.get("Light Microscope", "unknown")
    )
    return out

def flatten_log_runs(log_data: dict) -> dict[str, dict]:
    """Return {run_key: info_dict} from any nested log structure.

    Unlike `_iter_log_runs`, this ignores device/subdevice path info —
    the goal is a flat, key-addressable view for diffing. Device /
    Subdevice renames show up naturally as field-level diffs.
    Later entries win if the same run_key appears twice.
    """
    out: dict[str, dict] = {}

    def walk(obj, path):
        if isinstance(obj, dict):
            if _is_log_info(obj):
                if path:
                    run = normalise_run_key(path[-1])
                    out[run] = obj
                return
            for k, v in obj.items():
                walk(v, path + [str(k)])

    walk(log_data or {}, [])
    return out




def nest_log_runs(flat: dict, folder: Optional[Path] = None) -> dict:
    """{(filename_or_None, run): info} -> canonical filename -> run -> info.

    Entries with a None filename go under "_unresolved_" (orphan bucket).
    """
    out: dict = {}
    for key, info in flat.items():
        if isinstance(key, tuple) and len(key) == 2:
            fname, run = key
        else:
            fname, run = None, key
        bucket = fname if fname is not None else "_unresolved_"
        out.setdefault(bucket, {})[run] = info
    return out


def compare_logs(source: dict, gui: dict,
                 folder: Optional[Path] = None,
                 xls_data: Optional[dict] = None) -> dict:
    """Diff a source log against a GUI working copy.

    If `folder` is provided, both sides are resolved to (filename, run)
    identity via _resolve_log_to_identity — this handles both canonical
    and lazy source formats. Otherwise both are assumed canonical.

    Returns:
        {
            "new_runs":       [(filename, run), ...],
            "only_in_gui":    [(filename, run), ...],
            "differing_runs": {
                (filename, run): {"fields": {field: (gui_val, src_val)}}
            },
        }
    """
    if folder is not None:
        xls_data = xls_data or {}
        src_runs = _resolve_log_to_identity(source or {}, Path(folder), xls_data)[0]
        gui_runs = _flatten_canonical(gui or {})
    else:
        src_runs = _flatten_canonical(source or {})
        gui_runs = _flatten_canonical(gui or {})

    new_runs = sorted(set(src_runs) - set(gui_runs))
    only_in_gui = sorted(set(gui_runs) - set(src_runs))

    differing_runs = {}
    for key in sorted(set(src_runs) & set(gui_runs)):
        src_info = src_runs[key]
        gui_info = gui_runs[key]
        field_diffs = {}
        for k in set(src_info) | set(gui_info):
            if str(k).startswith("_"):
                continue
            sv, gv = src_info.get(k), gui_info.get(k)
            if sv != gv:
                field_diffs[k] = (gv, sv)
        if field_diffs:
            differing_runs[key] = {"fields": field_diffs}

    return {
        "new_runs": new_runs,
        "only_in_gui": only_in_gui,
        "differing_runs": differing_runs,
    }


def merge_logs(source: dict, gui: dict, mode: str,
               per_run_choices: Optional[dict] = None,
               folder: Optional[Path] = None,
               xls_data: Optional[dict] = None) -> dict:
    """Produce merged log data in canonical form (filename -> run -> info)."""
    if folder is not None:
        xls_data = xls_data or {}
        src_runs = _resolve_log_to_identity(source or {}, Path(folder), xls_data)[0]
        gui_runs = _flatten_canonical(gui or {})
    else:
        src_runs = _flatten_canonical(source or {})
        gui_runs = _flatten_canonical(gui or {})

    per_run_choices = per_run_choices or {}

    if mode == "overwrite_all":
        merged = {k: dict(v) for k, v in src_runs.items()}
        for k, v in gui_runs.items():
            merged.setdefault(k, dict(v))
    elif mode == "add_only":
        merged = {k: dict(v) for k, v in gui_runs.items()}
        for k, v in src_runs.items():
            merged.setdefault(k, dict(v))
    elif mode == "keep_gui":
        merged = {k: dict(v) for k, v in gui_runs.items()}
    else:  # reviewed
        merged = {k: dict(v) for k, v in gui_runs.items()}
        for key, src_info in src_runs.items():
            if key not in gui_runs:
                merged[key] = dict(src_info)
                continue
            choice = per_run_choices.get(key)
            if choice is None:
                continue
            if choice == "src":
                merged[key] = dict(src_info)
            elif choice == "gui":
                continue
            elif isinstance(choice, dict):
                patched = dict(gui_runs[key])
                for field, pick in choice.items():
                    if pick == "src":
                        if field in src_info:
                            patched[field] = src_info[field]
                        else:
                            patched.pop(field, None)
                merged[key] = patched

    out: dict = {}
    for (fname, run), info in merged.items():
        out.setdefault(fname, {})[run] = info
    return out

# ---------------------------------------------------------------------------
# LOG loading and lookup
# ---------------------------------------------------------------------------

def _extract_numports_key(key: Any) -> Optional[int]:
    """Extract 2/3/4 from keys like '2port', '4Term', '2'."""
    s = str(key).lower()
    m = re.search(r'(\d+)\s*(?:port|term)', s)
    if m:
        return int(m.group(1))
    if s.isdigit():
        return int(s)
    return None


def _is_log_info(d: dict) -> bool:
    """Heuristic: does this dict look like a run's LOG entry?"""
    if not isinstance(d, dict):
        return False
    keys = {str(k) for k in d.keys()}
    return (
        any(k.startswith("SMU") for k in keys)
        or "Comment" in keys
        or "comment" in keys
        or "LightMicroscope" in keys
        or "Light Microscope" in keys
    )


# ── Lax-JSON support ────────────────────────────────────────────────
class LogParseError(ValueError):
    """Raised when a LOG file cannot be parsed. Carries line/col for the UI."""
    def __init__(self, message, *, lineno=None, colno=None, original=None):
        super().__init__(message)
        self.lineno = lineno
        self.colno = colno
        self.original = original


# Bare-word tokens that people habitually write instead of the strict JSON equivalents. 
# Both maps cover the same set of bad spellings plus the *other* syntax's keywords, so either parser can retry.

_NULL_LIKE  = {"none", "null", "nul", "nan"}     # case-insensitive
_TRUE_LIKE  = {"true"}
_FALSE_LIKE = {"false"}

_JSON_TARGETS = {"null": "null", "true": "true", "false": "false"}
_PY_TARGETS   = {"null": "None", "true": "True", "false": "False"}


def _normalise_lax_json(text: str, target: str) -> str:
    """Prepare log text for a specific parser.

    target = "json"  → targets json.loads  (null / true / false)
    target = "py"    → targets ast.literal_eval (None / True / False)

    Two transformations, both string-literal-aware:
      1. Bare-word tokens replaced case-insensitively:
         Nul / nul / NULL / nUll / nan / NaN / None / … → null (or None)
         True / TRUE / tRUe → true (or True)
         False / FALSE      → false (or False)
      2. Trailing commas stripped before `}` or `]`:
         {"a":1,} → {"a":1}    [1,2,] → [1,2]
    Both operations skip anything inside string literals.
    """
    target_map = _JSON_TARGETS if target == "json" else _PY_TARGETS

    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    quote  = ""
    esc    = False

    while i < n:
        ch = text[i]

        # ── Inside a string literal: copy verbatim ──
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            i += 1
            continue

        # ── Entering a string literal ──
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            out.append(ch)
            i += 1
            continue

        # ── Punctuation: possibly a trailing comma to strip ──
        if not (ch.isalnum() or ch == "_"):
            if ch == ",":
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j] in "}]":
                    i += 1              # skip the comma
                    continue
            out.append(ch)
            i += 1
            continue

        # ── Bare word: consume, decide, replace ──
        word_end = i
        while word_end < n and (text[word_end].isalnum()
                                or text[word_end] == "_"):
            word_end += 1
        word  = text[i:word_end]
        lower = word.lower()
        if lower in _NULL_LIKE:
            out.append(target_map["null"])
        elif lower in _TRUE_LIKE:
            out.append(target_map["true"])
        elif lower in _FALSE_LIKE:
            out.append(target_map["false"])
        else:
            out.append(word)
        i = word_end

    return "".join(out)



def load_log_file(path: Path, *, migration_callback=None) -> dict:
    """Load a LOG file (.json / .txt / .xls) into a dict.

    Tries in order:
      1. Strict JSON on the raw text.
      2. Python literal-eval (handles None / True / False).
      3. Strict JSON after normalizing lax tokens (Nul / NaN / nan / None).
      4. Python literal-eval after normalizing strict-JSON tokens.

    If (2), (3) or (4) was used and migration_callback is provided, the
    callback is invoked with (path, parsed_data) so the caller can back
    up the original and rewrite it in canonical form.

    Raises LogParseError (with .lineno / .colno) if all four fail.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".json", ".txt"):
        text = path.read_text(encoding="utf-8-sig")
        data, status = _parse_log_text(text, path)
        if status != "strict_json" and migration_callback is not None:
            try:
                migration_callback(path, data)
            except Exception as e:
                logger.warning(f"Log migration failed for {path}: {e}")
        return data

    if suffix == ".xls":
        return convert_log_xls_to_dict(path)

    raise ValueError(f"Unsupported LOG file type: {suffix}")


def find_log_file(folder: Path) -> Optional[Path]:
    """Find the user-editable source LOG file in a folder.

    Excludes *_GUI.* working copies — those are app-managed and are
    returned by find_gui_log_file() instead.
    """
    folder = Path(folder)

    for pattern in (
        "*_LOG.json", "*_LOG.txt", "*_LOG.xls",
        "*LOG*.json", "*LOG*.txt", "*LOG*.xls",
    ):
        matches = [
            m for m in sorted(folder.glob(pattern))
            if not m.stem.endswith("_GUI")
        ]
        if matches:
            return matches[0]

    return None

def find_gui_log_file(folder: Path) -> Optional[Path]:
    """Find the app-managed GUI working copy of the log.

    Priority:
      1. <source_stem>_GUI.<suffix> — the normal case, paired with a
         source file found by find_log_file().
      2. Any *_GUI.{json,txt,xls} in the folder — GUI-only setup, where
         no source file exists (e.g. fallback-log folder that got edited).

    Returns None if no GUI file exists.
    """
    folder = Path(folder)

    source = find_log_file(folder)
    if source is not None:
        candidate = gui_log_path(source)
        if candidate.exists():
            return candidate

    for pattern in ("*_GUI.json", "*_GUI.txt", "*_GUI.xls"):
        matches = sorted(folder.glob(pattern))
        if matches:
            return matches[0]

    return None


def gui_log_path(source_path: Path) -> Path:
    """Return the canonical GUI working-copy path for a source log.

    _LOGBOOK.json  →  _LOGBOOK_GUI.json
    2024_LOG.txt   →  2024_LOG_GUI.txt
    """
    source_path = Path(source_path)
    return source_path.with_name(source_path.stem + "_GUI" + source_path.suffix)

def build_log_lookup(log_data: dict,
                     default_device: Optional[str] = None,
                     default_subdevice: str = "SD0") -> dict:
    """Legacy shim — returns {(filename_or_None, run): info}.

    Retained so nothing importing it crashes. New code should use
    _resolve_log_to_identity(log_data, folder, xls_data) which can
    resolve lazy entries to actual filenames.
    """
    lookup = {}
    for path, run, info in _iter_log_runs(log_data):
        fname = path[0] if path else None
        lookup[(fname, run)] = info
    return lookup


def find_log_info(lookup: dict,
                  run_data: dict,
                  run_key: str,
                  device: Optional[str] = None,
                  subdevice: Optional[str] = None) -> Optional[dict]:
    """Find a LOG entry for one run."""
    np = get_num_ports(run_data)
    run_norm = normalise_run_key(run_key)

    if subdevice is None:
        subdevice = "SD0"

    keys = [
        (str(np), device, subdevice, run_norm),
        (None, device, subdevice, run_norm),
        (None, device, "SD0", run_norm),
        (None, None, None, run_norm),
    ]

    for k in keys:
        if k in lookup:
            return lookup[k]

    return None


# ---------------------------------------------------------------------------
# Single-file parsing
# ---------------------------------------------------------------------------

def parse_keithley_file(file_path: Path,
                        log_lookup=None,
                        device: Optional[str] = None,
                        subdevice: str = "SD0") -> dict:
    """Parse one Keithley .xls file. Returns run_key -> run_data.

    LOG attachment is deferred to build_device_data — this function
    only attaches a fallback LOG when called standalone, so external
    consumers still see complete run_data. When called from the build
    path with log_lookup=None, run_data["LOG"] is a fallback that the
    caller is expected to overwrite with the resolved entry.
    """
    file_path = Path(file_path)
    reader = KeithleyDataReader()
    file_data = reader.read_file(str(file_path))

    if "Settings" in file_data:
        settings_dict = file_data.pop("Settings")
        for run_key, run_data in file_data.items():
            base = settings_dict.get(run_key, {})
            existing = run_data.get("Settings") or {}
            run_data["Settings"] = {**base, **existing}

    result = {}
    for run_key, run_data in file_data.items():
        run_data["filename"] = file_path.name
        run_data["LOG"] = make_fallback_log(run_data)
        result[normalise_run_key(run_key)] = run_data
    return result


# ---------------------------------------------------------------------------
# Public API kept for Data_Containers
# ---------------------------------------------------------------------------

def Keithley_xls_read_file(file_path: str,
                           logbook: Optional[LogBook] = None,
                           log_source: Optional[Any] = None,
                           device: Optional[str] = None,
                           subdevice: str = "SD0") -> dict:
    """
    Read one Keithley .xls file.

    LOG resolution priority:
      1. explicit log_source callable
      2. explicit legacy logbook
      3. auto-discovered LOG file in the same folder
      4. fallback LOG built from Settings['Instrument']
    """
    file_path = Path(file_path)

    if log_source is None and logbook is not None:
        log_source = logbook.get_run_info

    log_lookup = None

    if log_source is not None:
        log_lookup = log_source
    else:
        log_path = find_log_file(file_path.parent)
        if log_path is not None:
            try:
                log_data = load_log_file(log_path)
                log_lookup = build_log_lookup(log_data)
            except Exception as e:
                logger.warning(f"Could not parse LOG file {log_path}: {e}")

    return parse_keithley_file(
        file_path,
        log_lookup=log_lookup,
        device=device,
        subdevice=subdevice,
    )


def Keithley_xls_read(directory: str, **kwargs) -> dict:
    """Read every Keithley .xls data file in a directory."""
    directory = Path(directory)

    data_files = [
        f for f in directory.glob("*.xls")
        if "LOG" not in f.stem.upper()
    ]

    results = {}

    for df in data_files:
        try:
            results[df.stem] = Keithley_xls_read_file(str(df), **kwargs)
        except Exception as e:
            logger.error(f"Error reading {df.name}: {e}")

    return results


# ---------------------------------------------------------------------------
# New pkl-based device aggregation
# ---------------------------------------------------------------------------

def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def _log_fingerprint(folder: Path) -> Optional[dict]:
    """Snapshot both the source LOG and its GUI working copy.

    Returns None only when neither exists (folder has no log at all).
    Otherwise:

        {"source": {...} | None, "gui": {...} | None}

    - source = the user-editable file (find_log_file)
    - gui    = the app-edited working copy (find_gui_log_file)

    The pkl is built from "gui if present, else source", so only the
    gui stat needs to gate rebuilds. The source stat is captured purely
    so we can warn the user when the source changes without triggering
    a rebuild (which would silently discard GUI edits).
    """
    source = find_log_file(Path(folder))
    gui = find_gui_log_file(Path(folder))
    if source is None and gui is None:
        return None

    out = {"source": None, "gui": None}
    for key, p in (("source", source), ("gui", gui)):
        if p is None:
            continue
        try:
            st = p.stat()
            out[key] = {"path": str(p), "mtime": st.st_mtime, "size": st.st_size}
        except OSError:
            pass
    return out


def _log_fingerprint_matches(folder: Path, stored) -> bool:
    """True when the pkl can be reused without rebuilding.

    Rules:
      - No logs on disk: keep whatever the pkl has (never wipe data
        because a log file was removed).
      - Logs appeared since build: rebuild (the pkl is missing them).
      - Legacy/corrupt fingerprint format: rebuild once to reseed.
      - GUI working copy changed since build: rebuild (GUI is the
        pkl's declared source of truth).
      - Only the source changed: MATCH. Pkl stays valid; the caller
        emits a Problems-tab warning via _source_log_changed().
    """
    current = _log_fingerprint(folder)
    if current is None:
        return True                              # no logs on disk
    if stored is None:
        return False                             # logs appeared since build
    if not isinstance(stored, dict) or "source" not in stored:
        return False                             # legacy format → reseed
    return stored.get("gui") == current.get("gui")

def _source_log_changed(folder: Path, stored_fp) -> bool:
    """True when the source LOG exists and differs from what the pkl
    was built against.

    Used only to emit a 'source may be stale' Problems-tab entry.
    Never triggers a rebuild — that would discard GUI edits, which is
    the exact bug the two-file model exists to prevent.
    """
    if not isinstance(stored_fp, dict) or "source" not in stored_fp:
        return False
    current = _log_fingerprint(folder)
    if current is None:
        return False
    return current.get("source") != stored_fp.get("source")

def _merge_device_data(old: dict, new: dict,
                       overwrite_existing: bool = True) -> dict:
    merged = copy.deepcopy(old)
    merged.setdefault("_meta", {"material": "AIR", "material_stack": {}})

    for fname, runs in new.items():
        if fname == "_meta":
            merged["_meta"] = {**merged.get("_meta", {}), **runs}
            continue
        if not isinstance(runs, dict):
            continue
        target = merged.setdefault(fname, {})
        for run_key, run_data in runs.items():
            if not overwrite_existing and run_key in target:
                continue
            target[run_key] = run_data

    return merged

def _iter_log_runs(log_data: dict):
    """Yield (path_tuple, run_norm, info) for each run entry.

    `path` is every key between the root and the run key, in tree order.
    Skips keys that start with '_'.
    """
    def walk(obj, path):
        if isinstance(obj, dict):
            if _is_log_info(obj):
                if not path:
                    return
                yield (tuple(path[:-1]), normalise_run_key(path[-1]), obj)
                return
            for k, v in obj.items():
                if str(k).startswith("_"):
                    continue
                yield from walk(v, path + [str(k)])

    yield from walk(log_data or {}, [])
    
def _iter_data_runs(xls_data: dict):
    """Yield (filename, run_key, run_data) across parsed XLS data."""
    for fname, file_data in xls_data.items():
        for run_key, run_data in file_data.items():
            yield fname, run_key, run_data


def _flatten_canonical(log_data: dict) -> dict:
    """Return {(filename, run_key): info} from canonical-shape log data.

    Canonical shape is filename -> run_key -> info. Anything else (lazy
    shapes, orphan bucket) is ignored — those must go through
    _resolve_log_to_identity() which has folder context.
    """
    out = {}
    if not isinstance(log_data, dict):
        return out
    for fname, runs in log_data.items():
        if str(fname).startswith("_") or not isinstance(runs, dict):
            continue
        for run_key, info in runs.items():
            if not isinstance(info, dict) or not _is_log_info(info):
                continue
            out[(str(fname), normalise_run_key(run_key))] = info
    return out


def _resolve_log_to_identity(log_data: dict, folder: Path,
                             xls_data: dict) -> tuple[dict, list]:
    """Normalise any log structure to {(filename, run_key): info}.

    Supported inputs:
      * canonical     — filename -> run_key -> info
      * lazy, ports   — numports -> device -> subdevice -> run_key -> info
      * lazy, no-sub  — device -> subdevice -> run_key -> info
      * lazy, bare    — device -> run_key -> info
      * bare          — run_key -> info

    Resolution for lazy entries:
      1. Find all files with this run_key.
      2. If exactly one → use it.
      3. If multiple and a numports key was in the path → filter by port count.
      4. Still ambiguous or no match → orphan.

    Returns (identity_dict, orphan_list). Each orphan entry is
    {"path": path_tuple, "run": run_key, "info": log_info_dict}.
    """
    folder = Path(folder)
    known_filenames = set(xls_data.keys())

    data_runs_by_id: dict[str, list[tuple[str, int]]] = {}
    for fname, run_key, run_data in _iter_data_runs(xls_data):
        n = get_num_ports(run_data)
        data_runs_by_id.setdefault(run_key, []).append((fname, n))

    identity: dict[tuple[str, str], dict] = {}
    orphans: list[dict] = []

    for path, run, info in _iter_log_runs(log_data):
        if not path:
            orphans.append({"path": path, "run": run, "info": info})
            continue

        if path[0] in known_filenames:
            identity[(path[0], run)] = info
            continue

        numports_hint = None
        for part in path[:-1]:
            n = _extract_numports_key(part)
            if n is not None:
                numports_hint = n
                break

        candidates = data_runs_by_id.get(run, [])
        if len(candidates) == 1:
            identity[(candidates[0][0], run)] = info
        elif len(candidates) > 1 and numports_hint is not None:
            matched = [(f, n) for f, n in candidates if n == numports_hint]
            if len(matched) == 1:
                identity[(matched[0][0], run)] = info
            else:
                orphans.append({"path": path, "run": run, "info": info})
        else:
            orphans.append({"path": path, "run": run, "info": info})

    return identity, orphans
    
def build_device_data(folder: Path,
                      force_reload: bool = False,
                      compare: bool = False,
                      cache_name: Optional[str] = None,
                      overwrite_existing: bool = True,
                      problems: Optional[list] = None) -> dict:
    """Build/load one folder's pkl.

    New pkl layout: filename -> run_key -> run_data.

    Log precedence:
      1. _LOGBOOK_GUI.json if it exists
      2. source log if it exists
      3. auto-create a stub GUI log from fallback logs

    Reconciliation problems are added to `problems` and surfaced in the
    Problems tab.
    """
    if problems is None:
        problems = []

    folder = Path(folder)
    folder_str = str(folder.resolve())

    def _add(severity, category, message, path=None, run=None,
             line=None, column=None):
        problems.append({
            "severity": severity,
            "category": category,
            "folder": folder_str,
            "path": str(path) if path is not None else None,
            "run": run,
            "message": message,
            "line": line,
            "column": column,
        })

    cache_path = folder / (cache_name or f".dmui_{folder.name}.pkl")

    # Ensure GUI copy exists before cache check so first load reseeds
    # the fingerprint format.
    _source_pre = find_log_file(folder)
    _gui_pre = find_gui_log_file(folder)
    if _source_pre is not None and _gui_pre is None:
        try:
            shutil.copyfile(_source_pre, gui_log_path(_source_pre))
            _gui_pre = gui_log_path(_source_pre)
        except OSError as exc:
            _add("warning", "gui_log_create_failed",
                 f"Could not create GUI working copy: {exc}",
                 path=_source_pre)

    if cache_path.exists() and not force_reload:
        cached = _load_pickle(cache_path)
        cached_meta = cached.get("_meta") or {}
        stored_fp = cached_meta.get("log_fingerprint")
        # Old-format pkls (numports -> dev -> sub -> run) fail the
        # version check via the structure marker; we force a rebuild.
        structure_version = cached_meta.get("structure_version")
        if structure_version != 2:
            logger.info(f"Old pkl structure detected, rebuilding: {cache_path}")
        elif _log_fingerprint_matches(folder, stored_fp):
            if _source_log_changed(folder, stored_fp):
                _add("info", "source_log_changed",
                     "Source LOG changed since pkl build. GUI is "
                     "authoritative; use Compare & merge to propagate.",
                     path=find_log_file(folder))
            return cached
        else:
            logger.info(f"GUI log changed, rebuilding: {cache_path}")

    # ── Load log data ─────────────────────────────────────────────
    gui_path = find_gui_log_file(folder)
    source_path = find_log_file(folder)
    log_path = gui_path or source_path
    log_data = None
    log_existed = log_path is not None

    if log_path is not None:
        is_gui = (gui_path is not None and log_path == gui_path)
        migration_cb = (
            (lambda p, d: _migrate_log_file(p, d, problems))
            if is_gui else None
        )
        try:
            log_data = load_log_file(log_path, migration_callback=migration_cb)
        except Exception as e:
            _add("error", "log_parse_failed",
                 f"Could not parse LOG: {e}",
                 path=log_path,
                 line=getattr(e, "lineno", None),
                 column=getattr(e, "colno", None))

    # ── Parse every data XLS (single pass) ────────────────────────
    xls_files = [f for f in folder.glob("*.xls")
                 if "LOG" not in f.stem.upper()]
    xls_data: dict[str, dict] = {}

    if not log_existed and xls_files:
        _add("warning", "missing_log",
             f"No LOG file found. Auto-creating a GUI log for "
             f"{len(xls_files)} data file(s). Edit Device/Subdevice "
             f"in the Logbook tab to classify.")

    for xls in xls_files:
        try:
            xls_data[xls.name] = parse_keithley_file(xls, log_lookup=None)
        except Exception as e:
            _add("error", "data_parse_failed",
                 f"Could not read {xls.name}: {e}",
                 path=xls,
                 line=getattr(e, "lineno", None),
                 column=getattr(e, "colno", None))

    # ── Resolve log to (filename, run) identity ────────────────────
    identity: dict = {}
    orphans: list = []
    if log_data is not None:
        identity, orphans = _resolve_log_to_identity(
            log_data, folder, xls_data)

    # ── Attach logs to runs ───────────────────────────────────────
    missing_log_count = 0
    for fname, file_data in xls_data.items():
        for run_key, run_data in file_data.items():
            real_log = identity.get((fname, run_key))
            if real_log is not None:
                run_data["LOG"] = normalise_log_info(real_log, run_data)
            # else keep fallback LOG already set by parse_keithley_file
            if run_data["LOG"].get("Comment") == "NO LOG":
                missing_log_count += 1
            run_data["LOG"].setdefault("Device", "Unlabelled")
            run_data["LOG"].setdefault("Subdevice", "SD0")

    if missing_log_count:
        _add("warning", "run_missing_from_log",
             f"{missing_log_count} data run(s) have no log entry "
             f"(showing Comment='NO LOG').")

    if orphans:
        _add("info", "run_orphaned_in_log",
             f"{len(orphans)} log entry(ies) could not be matched to "
             f"any data run: {', '.join(str(o['run']) for o in orphans[:5])}"
             + (" …" if len(orphans) > 5 else ""),
             path=log_path)

    # ── Auto-create GUI log when none existed ─────────────────────
    if not log_existed and xls_data:
        target = folder / "_LOGBOOK_GUI.json"
        gui_data = {}
        for fname, file_data in xls_data.items():
            gui_data[fname] = {}
            for run_key, run_data in file_data.items():
                gui_data[fname][run_key] = dict(run_data["LOG"])
        try:
            with open(target, "w", encoding="utf-8") as f:
                json.dump(gui_data, f, indent=2,
                          ensure_ascii=False, default=str)
            _add("info", "gui_log_created",
                 f"Auto-created {target.name} with "
                 f"{sum(len(v) for v in gui_data.values())} stub(s).",
                 path=target)
        except OSError as e:
            _add("error", "gui_log_create_failed",
                 f"Could not write {target.name}: {e}",
                 path=target)

    # ── Build fresh pkl in new structure ──────────────────────────
    fresh: dict = {}
    for fname, file_data in xls_data.items():
        fresh[fname] = file_data

    fresh["_meta"] = {
        "material": detect_material_from_path(folder),
        "material_stack": {},
        "log_fingerprint": _log_fingerprint(folder),
        "structure_version": 2,
    }

    if compare and cache_path.exists():
        old = _load_pickle(cache_path)
        old_meta = old.get("_meta", {})
        if old_meta.get("structure_version") == 2:
            merged = _merge_device_data(old, fresh,
                                        overwrite_existing=overwrite_existing)
        else:
            merged = fresh  # old format → discard entirely
        _save_pickle(cache_path, merged)
        return merged

    _save_pickle(cache_path, fresh)
    return fresh

