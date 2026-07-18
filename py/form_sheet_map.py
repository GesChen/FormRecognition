"""Workbook-template-scoped form-to-sheet metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from config import PATHS, XLSX_TEMPLATE_NAME
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parent.parent
    PATHS = {"data": ROOT / "data"}
    XLSX_TEMPLATE_NAME = "2026 template ver 1"


WORKBOOK_SUFFIXES = {".xlsx", ".xlsm", ".xltx", ".xltm"}


def _template_name(template_name: str | None = None) -> str:
    name = str(template_name or XLSX_TEMPLATE_NAME).strip()
    if not name:
        raise ValueError("XLSX template name is not configured.")
    if "/" in name or "\\" in name or name in {".", ".."}:
        raise ValueError(f"Invalid XLSX template name: {name!r}")
    return name


def form_sheet_map_path(template_name: str | None = None) -> Path:
    data_root = Path(PATHS.get("data", Path(__file__).resolve().parent.parent / "data"))
    name = _template_name(template_name)
    return data_root / "xlsx" / "form_sheet_maps" / f"{name}.json"


def load_form_sheet_map(template_name: str | None = None) -> dict[str, dict[str, Any]]:
    path = form_sheet_map_path(template_name)
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


def form_sheet_meta(form: str, template_name: str | None = None) -> dict[str, Any]:
    return load_form_sheet_map(template_name).get(str(form or "").strip(), {})


def form_sheet_name(form: str, template_name: str | None = None) -> str | None:
    value = form_sheet_meta(form, template_name).get("sheet")
    if value:
        return str(value)
    return None


def workbook_template_path(template_name: str | None = None) -> Path:
    """
    Return the configured data-entry workbook template.

    Templates live directly under data/xlsx/workbook_templates/. The configured
    template name is matched first as an exact workbook stem, then as an exact
    filename, then as a case-insensitive stem. As a compatibility fallback, a
    directory with the template name is searched using the old one-workbook
    convention.
    """
    data_root = Path(PATHS.get("data", Path(__file__).resolve().parent.parent / "data"))
    name = _template_name(template_name)
    root = data_root / "xlsx" / "workbook_templates"
    if not root.is_dir():
        raise FileNotFoundError(f"Workbook template directory not found: {root}")

    if Path(name).suffix.lower() in WORKBOOK_SUFFIXES:
        candidate = root / name
        if candidate.is_file():
            return candidate.resolve()

    for suffix in sorted(WORKBOOK_SUFFIXES):
        candidate = root / f"{name}{suffix}"
        if candidate.is_file():
            return candidate.resolve()

    candidates = sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in WORKBOOK_SUFFIXES],
        key=lambda p: p.name.lower(),
    )
    for p in candidates:
        if p.stem.lower() == name.lower() or p.name.lower() == name.lower():
            return p.resolve()

    legacy_root = root / name
    if legacy_root.is_dir():
        candidates = sorted(
            [p for p in legacy_root.iterdir() if p.is_file() and p.suffix.lower() in WORKBOOK_SUFFIXES],
            key=lambda p: p.name.lower(),
        )
        if candidates:
            return candidates[0].resolve()

    if not candidates:
        raise FileNotFoundError(f"No workbook templates found in: {root}")
    raise FileNotFoundError(f"No workbook template named {name!r} found in: {root}")
