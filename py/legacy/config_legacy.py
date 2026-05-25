"""
Legacy config sections extracted from py/config.py.

Supports deprecated modules in py/legacy/:
  - ID_RECOGNITION   -> id_recognize.py
  - METADATA_GETTER  -> metadata_getter.py
  - MULTIPLE_CHOICE  -> mc_detect.py

To reintegrate, move the relevant dict back into py/config.py and update
the module's import from ``config_legacy`` to ``config``.
"""

import sys
from pathlib import Path

_py_dir = Path(__file__).resolve().parent.parent
if str(_py_dir) not in sys.path:
    sys.path.insert(0, str(_py_dir))

from config import PATHS

# id_recognize: OCR on "id" ROI from schema
# ocr_engine: "tesseract" | "easyocr" | "ocr_engine" (see ocr_*.py modules)
# crop_cache_root: where to store cropped ID ROI images before OCR
# id_trim_and_fix: True = extract after colon, apply replacements, fixed length
# id_char_length: fixed length for all IDs (e.g. 8 for "9010526A")
# id_char_replacements: common OCR misreads -> correct char (applied case-insensitive)
ID_RECOGNITION = {
    "schema_path": PATHS["data"] / "roi_schemas" / "schema_sidea.json",
    "ocr_engine": "ocr_engine",
    "crop_cache_root": PATHS["cache"] / "id_crops",
    "id_trim_and_fix": True,
    "id_char_length": 8,
    "id_char_replacements": {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "G": "6",
        "B": "8",
    },
}

# Metadata getter: OCR on configurable ROI fields (date, school, teacher, etc.)
# Odd pages, side a. schema_key for data/roi_schemas/<key>.json.
# field_names: ROI names to read (excludes id). ocr_engine: tesseract | easyocr | ocr_engine.
METADATA_GETTER = {
    "schema_key": "schema_sidea",
    "schema_dir": PATHS["data"] / "roi_schemas",
    "field_names": ["date", "school", "teacher"],
    "ocr_engine": "ocr_engine",
    "crop_cache_root": PATHS["cache"] / "metadata_crops",
}

# Multiple-choice (bubble) detection: filled = grayscale-weighted darkness >= threshold
# filled_threshold_by_label: label "a".."h" -> min average darkness (0-1)
# use_graduated_region: if True, use distance-weighted average darkness
# graduated_falloff: dict with center_weight, edge_weight, transition_distance, falloff_power
MULTIPLE_CHOICE = {
    "schema_dir": PATHS["data"] / "roi_schemas",
    "filled_threshold_by_label": {
        "a": 0.15,
        "b": 0.15,
        "c": 0.15,
        "d": 0.15,
        "e": 0.15,
        "f": 0.15,
        "g": 0.15,
        "h": 0.15,
    },
    "use_graduated_region": False,
    "graduated_falloff": {
        "center_weight": 0.0,
        "edge_weight": 1.0,
        "transition_distance": .6,
        "falloff_power": 6,
    },
}
