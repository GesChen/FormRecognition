"""Release-scoped form-to-workbook-sheet metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from config import DATA_RELEASE, PATHS
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parent.parent
    DATA_RELEASE = "2026"
    PATHS = {"data": ROOT / "data"}


def form_sheet_map_path(release: str | None = None) -> Path:
    data_root = Path(PATHS.get("data", Path(__file__).resolve().parent.parent / "data"))
    rel = str(release or DATA_RELEASE).strip()
    return data_root / "xlsx" / "form_sheet_maps" / f"{rel}.json"


def load_form_sheet_map(release: str | None = None) -> dict[str, dict[str, Any]]:
    path = form_sheet_map_path(release)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for form, meta in data.items():
        if isinstance(meta, dict):
            out[str(form)] = dict(meta)
    return out


def form_sheet_meta(form: str, release: str | None = None) -> dict[str, Any]:
    return load_form_sheet_map(release).get(str(form or "").strip(), {})


def form_sheet_name(form: str, release: str | None = None) -> str | None:
    value = form_sheet_meta(form, release).get("sheet")
    if value:
        return str(value)
    return None


def workbook_template_path(release: str | None = None) -> Path:
    """
    Return the release-scoped data-entry workbook template.

    Templates live under data/xlsx/workbook_templates/<release>/. A release is
    expected to have one workbook; if multiple exist, prefer one whose filename
    contains the release string, then fall back to deterministic name order.
    """
    data_root = Path(PATHS.get("data", Path(__file__).resolve().parent.parent / "data"))
    rel = str(release or DATA_RELEASE).strip()
    root = data_root / "xlsx" / "workbook_templates" / rel
    if not root.is_dir():
        raise FileNotFoundError(f"Workbook template directory not found for release {rel!r}: {root}")
    candidates = sorted(
        [
            p for p in root.iterdir()
            if p.is_file() and p.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm"}
        ],
        key=lambda p: p.name.lower(),
    )
    if not candidates:
        raise FileNotFoundError(f"No workbook template found for release {rel!r}: {root}")
    for p in candidates:
        if rel.lower() in p.name.lower():
            return p.resolve()
    return candidates[0].resolve()
