"""
Helper scripts that are kept separate to avoid cluttering the GUI file
"""
from dataclasses import dataclass, field
from typing import Protocol, Optional, Any
import numpy as np
import re
import json 
from pathlib import Path


"""
################################################################### 
                 REGEX AND CLASSIFY HELPERS 
Stored in <root>/.dmui_gui_state/_classify_patterns.json so patterns
travel with the data. A root with no file falls back to the built-in
defaults below. Two-field (device + subdevice) because the shorthand
is far cleaner when each half gets its own minimal syntax.
###################################################################
""" 
# ── Classify patterns (per-root) ─────────────────────────────────────

_DEFAULT_CLASSIFY_PATTERNS: list[dict] = [
    {
        "device_pattern":    r"[^_\-\.]+",
        "subdevice_pattern": r"[^_\-\.]+",
        "label": "Split at first separator (device / rest)",
        "enabled": True,
        "mode": "raw",
        "anchor_subdevice": True,
    },
    {
        "device_pattern":    r"^.+?(?=[_\-][^_\-\.]+$)",
        "subdevice_pattern": r"[^_\-\.]+$",
        "label": "Split at last separator (prefix / subdevice)",
        "enabled": False,
        "mode": "raw",
        "anchor_subdevice": True,
    },
    {
        "device_pattern":    r"DFR[x]-[AA]",
        "subdevice_pattern": r"[AAx]",
        "label": "DFR-style — demonstrates bracketed shorthand",
        "enabled": True,
        "mode": "shorthand",
        "anchor_subdevice": True,
    },
]

_BRACKET_SHORTHAND = {
    "x": r"\d+",
    "A": r"[A-Z]",
    "a": r"[a-z]",
}


def _shorthand_to_regex(pattern: str) -> str:
    """Expand bracketed shorthand to a regex string.

    * Outside brackets: every character is literal, EXCEPT '-' and '_'
      which both expand to '[-_]' (so 'DFR1-GG' and 'DFR1_GG' are
      interchangeable, always).
    * Inside brackets: per-character expansion
          x -> \\d+      A -> [A-Z]      a -> [a-z]
      any other character stays literal (escaped).
    * An unclosed '[' is treated as a literal '['.
    """
    out, i, n = [], 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "[":
            j = pattern.find("]", i + 1)
            if j == -1:
                out.append(re.escape(ch))
                i += 1
                continue
            for c in pattern[i + 1:j]:
                out.append(_BRACKET_SHORTHAND.get(c, re.escape(c)))
            i = j + 1
        elif ch in "-_":
            out.append(r"[-_]")
            i += 1
        else:
            out.append(re.escape(ch))
            i += 1
    return "".join(out)


def _effective_regex(pattern: str, mode: str) -> str:
    if not pattern:
        return ""
    return _shorthand_to_regex(pattern) if mode == "shorthand" else pattern


def _classify_patterns_path(root: Path) -> Path:
    return Path(root) / ".dmui_gui_state" / "_classify_patterns.json"


def load_classify_patterns(root: Path) -> list[dict]:
    """Load a root's patterns. Missing or unreadable → defaults."""
    p = _classify_patterns_path(root)
    if p.exists():
        try:
            with open(p) as f:
                raw = json.load(f).get("patterns", [])
            out = []
            for e in raw:
                if not isinstance(e, dict):
                    continue
                dev = str(e.get("device_pattern", "")).strip()
                sub = str(e.get("subdevice_pattern", "")).strip()
                if not dev or not sub:
                    continue
                mode = e.get("mode", "shorthand")
                if mode not in ("shorthand", "raw"):
                    mode = "shorthand"
                out.append({
                    "device_pattern":    dev,
                    "subdevice_pattern": sub,
                    "label":             str(e.get("label", "")),
                    "enabled":           bool(e.get("enabled", True)),
                    "mode":              mode,
                    "anchor_subdevice":  bool(e.get("anchor_subdevice", True)),
                })
            if out:
                return out
        except Exception:
            pass
    return [dict(p) for p in _DEFAULT_CLASSIFY_PATTERNS]


def save_classify_patterns(root: Path, patterns: list[dict]) -> None:
    p = _classify_patterns_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"patterns": patterns}, f, indent=2)


def classify_with_patterns(
    path: Path, patterns: list[dict]
) -> tuple[str, str]:
    """Best-fit classification. Pure function, no caching.

    Score = (len(device)+len(subdevice), -list_index).
    Longer extractions win; ties broken by list order.
    Both fields must match and produce ≥2 chars, else the entry is
    rejected. Falls back to (folder_name, file_stem) if nothing matches.
    """
    stem = path.stem
    best = None  # (device, subdevice, score)
    for idx, entry in enumerate(patterns):
        if not entry.get("enabled", True):
            continue
        mode = entry.get("mode", "shorthand")
        dev_src = _effective_regex(entry.get("device_pattern", ""), mode)
        sub_src = _effective_regex(entry.get("subdevice_pattern", ""), mode)
        if not dev_src or not sub_src:
            continue
        if entry.get("anchor_subdevice", True):
            sub_src = r"(?<=[-_])" + sub_src
        try:
            dev_re = re.compile(dev_src)
            sub_re = re.compile(sub_src)
        except re.error:
            continue
        dm = dev_re.search(stem)
        sm = sub_re.search(stem)
        if dm is None or sm is None:
            continue
        device    = (dm.group(0) or "").strip()
        subdevice = (sm.group(0) or "").strip()
        if len(device) < 2 or len(subdevice) < 2:
            continue
        if device == subdevice:
            continue  # not a useful split
        score = (len(device) + len(subdevice), -idx)
        if best is None or score > best[2]:
            best = (device, subdevice, score)
    if best is None:
        return (path.parent.name, stem)
    return (best[0], best[1])

