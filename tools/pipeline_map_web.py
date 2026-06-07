#!/usr/bin/env python3
"""Interactive ROI-to-XLSX pipeline visualizer."""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
PY_DIR = ROOT / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from flask import Flask, jsonify, request, send_file

from form_sheet_map import form_sheet_name, workbook_template_path

try:
    from config import DATA_RELEASE, PATHS, PROJECT_ROOT
except Exception:  # pragma: no cover
    PROJECT_ROOT = ROOT
    DATA_RELEASE = "2025"
    PATHS = {
        "data": ROOT / "data",
        "templates_root": ROOT / "data" / "templates" / DATA_RELEASE,
        "roi_schemas_root": ROOT / "data" / "roi_schemas" / DATA_RELEASE,
        "xlsx_mappings_root": ROOT / "data" / "xlsx" / "mappings" / DATA_RELEASE,
    }

try:
    from xlsx_mapping_auto_generator import extract_sheet_structure
except Exception:  # pragma: no cover
    extract_sheet_structure = None  # type: ignore[assignment]

SERVER_INSTANCE_ID = uuid.uuid4().hex
FORM_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
RESERVED_SOURCES = {"id", "form_type", "page_odd", "page_even"}
VALID_TYPES = {"direct", "lookup", "multi_column", "static"}


def _resolve_collection_root(path_value: Path | str, expected_leaf: str) -> Path:
    p = Path(path_value).resolve()
    if p.name == expected_leaf:
        return p
    if p.parent.name == expected_leaf:
        return p.parent
    return p


DATA_ROOT = Path(PATHS.get("data", ROOT / "data")).resolve()
TEMPLATES_ROOT = _resolve_collection_root(
    PATHS.get("templates_root", DATA_ROOT / "templates"),
    "templates",
)
ROI_SCHEMAS_ROOT = _resolve_collection_root(
    PATHS.get("roi_schemas_root", DATA_ROOT / "roi_schemas"),
    "roi_schemas",
)
MAPPINGS_ROOT = _resolve_collection_root(
    PATHS.get("xlsx_mappings_root", DATA_ROOT / "xlsx" / "mappings"),
    "mappings",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(PROJECT_ROOT).resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _safe_name(value: str | None, label: str) -> str:
    name = str(value or "").strip()
    if not name or not FORM_NAME_RE.match(name):
        raise ValueError(f"Invalid {label}")
    return name


def _source_sort_key(value: str) -> tuple[int, int, str]:
    s = str(value)
    if s in RESERVED_SOURCES:
        return (-1, 0, s)
    if s.isdigit():
        return (0, int(s), s)
    m = re.match(r"^(\d+)([A-Za-z]+)$", s)
    if m:
        return (1, int(m.group(1)), m.group(2).lower())
    return (2, 0, s.lower())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_releases() -> list[str]:
    ignored = {"example", "examples", ".backups"}
    releases: set[str] = set()
    for root in (TEMPLATES_ROOT, ROI_SCHEMAS_ROOT, MAPPINGS_ROOT):
        if root.is_dir():
            releases.update(
                p.name for p in root.iterdir()
                if p.is_dir() and p.name not in ignored and FORM_NAME_RE.match(p.name)
            )
    return sorted(releases)


def _forms_from_release_root(root: Path, release: str, suffixes: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    rel_root = root / release
    if not rel_root.is_dir():
        return out
    for suffix in suffixes:
        for p in rel_root.glob(f"*{suffix}"):
            stem = p.stem
            stem = re.sub(r"_[ab]$", "", stem, flags=re.I)
            if stem and FORM_NAME_RE.match(stem):
                out.add(stem)
    return out


def _available_forms(release: str) -> list[str]:
    forms: set[str] = set()
    forms.update(_forms_from_release_root(TEMPLATES_ROOT, release, (".png",)))
    forms.update(_forms_from_release_root(ROI_SCHEMAS_ROOT, release, (".json",)))
    forms.update(_forms_from_release_root(MAPPINGS_ROOT, release, (".json",)))
    forms.discard("schema_sidea")
    forms.discard("schema_sideb")
    return sorted(forms, key=str.lower)


def _browse_root(kind: str) -> Path:
    if kind == "template":
        return TEMPLATES_ROOT.resolve()
    raise ValueError("Invalid browse kind")


def _browse_files(kind: str, rel_dir: str = "") -> tuple[list[str], list[str]]:
    root = _browse_root(kind)
    base = Path(PROJECT_ROOT).resolve()
    rel_dir = str(rel_dir or "").strip().replace("\\", "/").rstrip("/")
    target = root if not rel_dir else (base / rel_dir).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return [], []
    if not target.is_dir():
        return [], []
    dirs: list[str] = []
    files: list[str] = []
    allowed = {".png"}
    for p in target.iterdir():
        try:
            rel = str(p.resolve().relative_to(base)).replace("\\", "/")
        except ValueError:
            continue
        if p.is_dir() and p.name not in {".backups", "__pycache__"}:
            dirs.append(rel)
        elif p.is_file() and p.suffix.lower() in allowed:
            files.append(rel)
    return sorted(dirs, key=str.lower), sorted(files, key=str.lower)


def _template_info_from_path(raw: str | None) -> dict[str, str]:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("Missing template path")
    if ".." in value.replace("\\", "/").split("/"):
        raise ValueError("Invalid template path")
    p = Path(value)
    if not p.is_absolute():
        p = Path(PROJECT_ROOT).resolve() / p
    p = p.resolve()
    p.relative_to(TEMPLATES_ROOT.resolve())
    if not p.is_file() or p.suffix.lower() != ".png":
        raise ValueError("Template must be a PNG inside the templates root")
    rel = p.relative_to(TEMPLATES_ROOT.resolve())
    if len(rel.parts) < 2:
        raise ValueError("Template path must be inside a release directory")
    release = _safe_name(rel.parts[0], "release")
    stem = p.stem
    side = ""
    m = re.match(r"^(.+)_([ab])$", stem, re.I)
    if m:
        form = m.group(1)
        side = m.group(2).lower()
    else:
        form = stem
    return {"release": release, "form": _safe_name(form, "form"), "side": side, "path": _rel(p)}


def _mapping_path(release: str, form: str) -> Path:
    path = (MAPPINGS_ROOT / release / f"{form}.json").resolve()
    path.relative_to(MAPPINGS_ROOT.resolve())
    return path


def _schema_path(release: str, form: str, side: str) -> Path:
    path = (ROI_SCHEMAS_ROOT / release / f"{form}_{side}.json").resolve()
    path.relative_to(ROI_SCHEMAS_ROOT.resolve())
    return path


def _template_path(release: str, form: str, side: str) -> Path:
    path = (TEMPLATES_ROOT / release / f"{form}_{side}.png").resolve()
    path.relative_to(TEMPLATES_ROOT.resolve())
    return path


def _mapping_destinations(row: dict[str, Any]) -> list[dict[str, str]]:
    mtype = str(row.get("type", "") or "").strip().lower()
    out: list[dict[str, str]] = []
    if mtype in {"direct", "lookup", "static"}:
        col = str(row.get("column", "") or "").strip().upper()
        if col:
            out.append({"column": col, "choice": ""})
    elif mtype == "multi_column":
        choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
        for choice, col in choices.items():
            c = str(col or "").strip().upper()
            if c:
                out.append({"column": c, "choice": str(choice)})
        no_answer = str(row.get("no_answer", "") or "").strip().upper()
        if no_answer:
            out.append({"column": no_answer, "choice": "no_answer"})
    return out


def _column_number(col: str) -> int:
    total = 0
    for ch in str(col or "").strip().upper():
        if not ("A" <= ch <= "Z"):
            return 0
        total = total * 26 + (ord(ch) - ord("A") + 1)
    return total


def _column_letter(num: int) -> str:
    if num <= 0:
        return ""
    out = ""
    while num:
        num, rem = divmod(num - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def _parse_range(value: str) -> tuple[int, int, int, int] | None:
    raw = str(value or "").strip().upper()
    m = re.match(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$", raw)
    if not m:
        return None
    c1, r1, c2, r2 = m.groups()
    return _column_number(c1), int(r1), _column_number(c2), int(r2)


def _roi_kind(name: str, all_names: set[str]) -> str:
    if re.match(r"^\d+[A-Za-z]+$", name):
        return "mcq_choice"
    if name.isdigit() and any(re.match(rf"^{re.escape(name)}[A-Za-z]+$", n) for n in all_names):
        return "mcq"
    return "text"


def _load_roi_nodes(release: str, form: str, side: str, issues: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    path = _schema_path(release, form, side)
    template = _template_path(release, form, side)
    if not path.is_file():
        issues.append({"severity": "error", "code": "missing_schema", "message": f"Missing ROI schema: {_rel(path)}"})
        return [], {}
    data = _load_json(path)
    rois = data.get("rois") if isinstance(data, dict) else []
    names = {
        str(r.get("name", "")).strip()
        for r in rois or []
        if isinstance(r, dict) and str(r.get("name", "")).strip()
    }
    nodes: list[dict[str, Any]] = []
    source_index: dict[str, list[str]] = {}
    for roi in rois or []:
        if not isinstance(roi, dict):
            continue
        name = str(roi.get("name", "") or "").strip()
        if not name:
            continue
        node_id = f"roi:{side}:{name}"
        kind = _roi_kind(name, names)
        node = {
            "id": node_id,
            "side": side,
            "name": name,
            "kind": kind,
            "bbox": {
                "x": float(roi.get("x", 0) or 0),
                "y": float(roi.get("y", 0) or 0),
                "w": float(roi.get("w", 0) or 0),
                "h": float(roi.get("h", 0) or 0),
            },
            "image_width": data.get("image_width"),
            "image_height": data.get("image_height"),
            "schema_path": _rel(path),
            "template_path": _rel(template) if template.is_file() else None,
            "mapped": False,
        }
        nodes.append(node)
        source_index.setdefault(name, []).append(node_id)
    if not template.is_file():
        issues.append({"severity": "warning", "code": "missing_template", "message": f"Missing template image: {_rel(template)}"})
    return nodes, source_index


def _load_workbook_columns(
    workbook_path: Path | None,
    *,
    release: str,
    form: str,
    mapping: dict[str, Any],
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any] | None]:
    if workbook_path is None:
        issues.append({
            "severity": "warning",
            "code": "missing_workbook",
            "message": "Release workbook template unavailable; spreadsheet nodes use mapping column letters only.",
        })
        return [], set(), None
    if extract_sheet_structure is None:
        issues.append({"severity": "warning", "code": "openpyxl_unavailable", "message": "Workbook headers unavailable: openpyxl helper is not importable."})
        return [], set(), None
    try:
        sheet = str(mapping.get("sheet", "") or "") or None
        start_row = int(mapping.get("start_row", 4) or 4)
        structure = extract_sheet_structure(
            workbook_path,
            form=form,
            release=release,
            sheet_hint=sheet,
            start_row_hint=start_row,
        )
    except Exception as exc:
        issues.append({"severity": "warning", "code": "workbook_load_failed", "message": f"Workbook headers unavailable: {exc}"})
        return [], set(), None
    nodes: list[dict[str, Any]] = []
    allowed: set[str] = set()
    for col in structure.get("columns") or []:
        if not isinstance(col, dict):
            continue
        letter = str(col.get("column", "") or "").strip().upper()
        if not letter:
            continue
        allowed.add(letter)
        nodes.append({
            "id": f"col:{letter}",
            "column": letter,
            "header_values": col.get("header_values") or [],
            "start_row_value": col.get("start_row_value", ""),
            "width": col.get("width"),
            "sheet": structure.get("sheet_name"),
            "mapped": False,
        })
    return nodes, allowed, structure


def _build_sheet_grid(
    *,
    mapping: dict[str, Any],
    workbook_structure: dict[str, Any] | None,
    columns_by_id: dict[str, dict[str, Any]],
    mapped_columns: set[str],
) -> dict[str, Any]:
    start_row = int(mapping.get("start_row", 4) or 4)
    sheet_name = str(mapping.get("sheet", "") or "")
    workbook_backed = isinstance(workbook_structure, dict)
    workbook_columns: list[dict[str, Any]] = []
    if workbook_backed:
        for col in workbook_structure.get("columns") or []:
            if isinstance(col, dict) and col.get("column"):
                workbook_columns.append(col)

    visible_letters: set[str] = set(mapped_columns)
    if workbook_columns:
        by_num = {
            _column_number(str(col.get("column", ""))): str(col.get("column", "")).strip().upper()
            for col in workbook_columns
            if _column_number(str(col.get("column", ""))) > 0
        }
        for letter in list(mapped_columns):
            n = _column_number(letter)
            for neighbor in (n - 1, n, n + 1):
                if neighbor in by_num:
                    visible_letters.add(by_num[neighbor])
    if not visible_letters:
        visible_letters.update(str(n.get("column", "")).strip().upper() for n in columns_by_id.values() if n.get("column"))

    column_lookup: dict[str, dict[str, Any]] = {}
    for col in workbook_columns:
        letter = str(col.get("column", "") or "").strip().upper()
        if letter:
            column_lookup[letter] = col
    for node in columns_by_id.values():
        letter = str(node.get("column", "") or "").strip().upper()
        if letter and letter not in column_lookup:
            column_lookup[letter] = node

    visible_rows = [1, 2]
    if start_row not in visible_rows:
        visible_rows.append(start_row)
    visible_rows = sorted(r for r in visible_rows if r > 0)
    visible_col_nums = sorted(_column_number(c) for c in visible_letters if _column_number(c) > 0)
    visible_col_num_set = set(visible_col_nums)
    merged_cells: dict[tuple[str, int], dict[str, Any]] = {}
    merged_ranges: list[dict[str, Any]] = []
    if workbook_backed:
        for raw_range in (workbook_structure or {}).get("merged_ranges_in_header_area", []):
            parsed = _parse_range(str(raw_range))
            if parsed is None:
                continue
            c1, r1, c2, r2 = parsed
            if r1 not in {1, 2} or r2 < r1:
                continue
            covered = [n for n in visible_col_nums if c1 <= n <= c2]
            if not covered:
                continue
            start_visible = min(covered)
            span = len(covered)
            start_letter = _column_letter(start_visible)
            merge_info = {
                "range": str(raw_range),
                "row": r1,
                "column": start_letter,
                "colspan": span,
                "rowspan": max(1, min(r2, 2) - r1 + 1),
                "hidden": False,
            }
            merged_cells[(start_letter, r1)] = merge_info
            merged_ranges.append(merge_info)
            for n in covered:
                letter = _column_letter(n)
                if letter == start_letter:
                    continue
                for row in range(r1, min(r2, 2) + 1):
                    merged_cells[(letter, row)] = {"hidden": True, "range": str(raw_range)}

    columns: list[dict[str, Any]] = []
    for letter in sorted(visible_letters, key=_column_number):
        col_data = column_lookup.get(letter, {})
        header_by_row = {
            int(hv.get("row")): str(hv.get("value", "") or "")
            for hv in (col_data.get("header_values") or [])
            if isinstance(hv, dict) and hv.get("row") is not None
        }
        cells = []
        for row in visible_rows:
            value = header_by_row.get(row, "")
            kind = "header"
            if row == start_row:
                value = str(col_data.get("start_row_value", "") or "")
                kind = "data"
            merge_info = merged_cells.get((letter, row), {})
            cells.append(
                {
                    "id": f"cell:{letter}:{row}",
                    "column": letter,
                    "row": row,
                    "value": value,
                    "kind": kind,
                    "mapped": row == start_row and letter in mapped_columns,
                    "merged": merge_info,
                }
            )
        columns.append(
            {
                "column": letter,
                "width": col_data.get("width"),
                "mapped": letter in mapped_columns,
                "cells": cells,
            }
        )

    return {
        "sheet_name": sheet_name or (workbook_structure or {}).get("sheet_name"),
        "start_row": start_row,
        "workbook_backed": workbook_backed,
        "visible_rows": visible_rows,
        "columns": columns,
        "merged_ranges_in_header_area": (workbook_structure or {}).get("merged_ranges_in_header_area", []),
        "visible_merged_ranges": merged_ranges,
    }


def build_visualization(
    release: str,
    form: str,
    *,
    workbook_path: Path | None = None,
) -> dict[str, Any]:
    release = _safe_name(release, "release")
    form = _safe_name(form, "form")
    issues: list[dict[str, Any]] = []
    if workbook_path is None:
        try:
            workbook_path = workbook_template_path(release)
        except Exception as exc:
            workbook_path = None
            issues.append({
                "severity": "warning",
                "code": "missing_release_workbook",
                "message": f"Release workbook template unavailable: {exc}",
            })
    mapping_path = _mapping_path(release, form)
    if mapping_path.is_file():
        mapping = _load_json(mapping_path)
        if not isinstance(mapping, dict):
            mapping = {"form_type": form, "sheet": form_sheet_name(form, release) or "", "start_row": 4, "mappings": []}
            issues.append({"severity": "error", "code": "invalid_mapping", "message": f"Mapping root is not an object: {_rel(mapping_path)}"})
    else:
        mapping = {"form_type": form, "sheet": form_sheet_name(form, release) or "", "start_row": 4, "mappings": []}
        issues.append({"severity": "warning", "code": "missing_mapping", "message": f"Missing mapping: {_rel(mapping_path)}"})
    configured_sheet = form_sheet_name(form, release)
    if configured_sheet:
        mapping["sheet"] = configured_sheet

    roi_nodes: list[dict[str, Any]] = []
    source_index: dict[str, list[str]] = {}
    for side in ("a", "b"):
        nodes, idx = _load_roi_nodes(release, form, side, issues)
        roi_nodes.extend(nodes)
        for source, ids in idx.items():
            source_index.setdefault(source, []).extend(ids)

    column_nodes, workbook_columns, workbook_structure = _load_workbook_columns(
        workbook_path,
        release=release,
        form=form,
        mapping=mapping,
        issues=issues,
    )
    column_by_id = {str(n.get("id")): n for n in column_nodes}
    column_letters_from_mapping: set[str] = set()
    virtual_nodes = [
        {"id": f"virtual:{name}", "name": name, "kind": "item_field", "mapped": False}
        for name in sorted(RESERVED_SOURCES)
    ]
    virtual_by_source = {n["name"]: n for n in virtual_nodes}
    static_nodes: list[dict[str, Any]] = []

    mapping_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    direct_edges: list[dict[str, Any]] = []
    mapped_sources: set[str] = set()
    mapped_columns: set[str] = set()
    mappings = mapping.get("mappings") if isinstance(mapping.get("mappings"), list) else []
    for idx, row in enumerate(mappings, start=1):
        if not isinstance(row, dict):
            issues.append({"severity": "error", "code": "invalid_mapping_row", "row": idx, "message": f"Mapping row {idx} is not an object."})
            continue
        mtype = str(row.get("type", "") or "").strip().lower()
        node_id = f"map:{idx}"
        source = str(row.get("source", "") or "").strip()
        destinations = _mapping_destinations(row)
        map_node = {
            "id": node_id,
            "row": idx,
            "type": mtype,
            "source": source,
            "column": str(row.get("column", "") or "").strip().upper(),
            "destinations": destinations,
            "transform": row.get("transform"),
            "lookup_count": len(row.get("map") or {}) if isinstance(row.get("map"), dict) else 0,
            "has_default": row.get("default") is not None,
            "choice_count": len(row.get("choices") or {}) if isinstance(row.get("choices"), dict) else 0,
            "mark": row.get("mark"),
            "value": row.get("value"),
            "mapping": row,
        }
        mapping_nodes.append(map_node)
        if mtype not in VALID_TYPES:
            issues.append({"severity": "error", "code": "unknown_mapping_type", "row": idx, "message": f"Unknown mapping type {mtype!r} at row {idx}."})

        if mtype == "static":
            static_id = f"static:{idx}"
            source_ids = [static_id]
            static_nodes.append({
                "id": static_id,
                "name": f"static row {idx}",
                "kind": "static",
                "value": row.get("value"),
                "mapped": True,
            })
            edges.append({"id": f"edge:static:{idx}", "from": static_id, "to": node_id, "kind": "static_to_mapping", "mapping_type": mtype})
        elif source:
            source_ids = []
            if source in virtual_by_source:
                virtual_by_source[source]["mapped"] = True
                mapped_sources.add(source)
                source_ids = [f"virtual:{source}"]
                edges.append({"id": f"edge:source:{idx}:virtual", "from": f"virtual:{source}", "to": node_id, "kind": "source_to_mapping", "mapping_type": mtype})
            else:
                source_ids = source_index.get(source, [])
                if not source_ids:
                    issues.append({"severity": "error", "code": "missing_source", "row": idx, "message": f"Mapping row {idx} references missing source {source!r}."})
                for src_id in source_ids:
                    mapped_sources.add(source)
                    edges.append({"id": f"edge:source:{idx}:{src_id}", "from": src_id, "to": node_id, "kind": "source_to_mapping", "mapping_type": mtype})
        elif mtype != "static":
            source_ids = []
            issues.append({"severity": "error", "code": "missing_source_field", "row": idx, "message": f"Mapping row {idx} has no source."})
        else:
            source_ids = []

        if not destinations and mtype != "static":
            issues.append({"severity": "warning", "code": "missing_destination", "row": idx, "message": f"Mapping row {idx} has no target column."})
        for dest in destinations:
            col = str(dest.get("column", "") or "").strip().upper()
            if not col:
                continue
            column_letters_from_mapping.add(col)
            mapped_columns.add(col)
            if workbook_columns and col not in workbook_columns:
                issues.append({"severity": "error", "code": "missing_column", "row": idx, "message": f"Mapping row {idx} targets missing workbook column {col!r}."})
            if f"col:{col}" not in column_by_id:
                column_by_id[f"col:{col}"] = {
                    "id": f"col:{col}",
                    "column": col,
                    "header_values": [],
                    "start_row_value": "",
                    "sheet": mapping.get("sheet"),
                    "mapped": False,
                }
            column_by_id[f"col:{col}"]["mapped"] = True
            edges.append({
                "id": f"edge:dest:{idx}:{col}:{dest.get('choice','')}",
                "from": node_id,
                "to": f"col:{col}",
                "kind": "mapping_to_column",
                "mapping_type": mtype,
                "choice": dest.get("choice", ""),
            })
            for src_id in source_ids:
                direct_edges.append(
                    {
                        "id": f"direct:{idx}:{src_id}:{col}:{dest.get('choice','')}",
                        "from": src_id,
                        "to": f"cell:{col}:{int(mapping.get('start_row', 4) or 4)}",
                        "column": col,
                        "row": int(mapping.get("start_row", 4) or 4),
                        "choice": dest.get("choice", ""),
                        "mapping_row": idx,
                        "mapping_type": mtype,
                        "type": mtype,
                        "source": source,
                        "transform": row.get("transform"),
                        "lookup_count": len(row.get("map") or {}) if isinstance(row.get("map"), dict) else 0,
                        "has_default": row.get("default") is not None,
                        "choice_count": len(row.get("choices") or {}) if isinstance(row.get("choices"), dict) else 0,
                        "mark": row.get("mark"),
                        "value": row.get("value"),
                        "mapping": row,
                    }
                )

    mapped_roi_ids = {e["from"] for e in edges if str(e.get("from", "")).startswith("roi:")}
    for node in roi_nodes:
        node["mapped"] = node["id"] in mapped_roi_ids
    for node in column_by_id.values():
        node["mapped"] = str(node.get("column", "")).upper() in mapped_columns

    column_nodes_out = sorted(column_by_id.values(), key=lambda n: _column_sort_key(str(n.get("column", ""))))
    sheet_grid = _build_sheet_grid(
        mapping=mapping,
        workbook_structure=workbook_structure,
        columns_by_id=column_by_id,
        mapped_columns=mapped_columns,
    )
    unmapped_roi_count = sum(1 for n in roi_nodes if not n.get("mapped") and n.get("kind") != "mcq_choice")
    return {
        "release": release,
        "form": form,
        "mapping_path": _rel(mapping_path),
        "mapping": {
            "form_type": mapping.get("form_type"),
            "sheet": mapping.get("sheet"),
            "start_row": mapping.get("start_row"),
            "mapping_count": len(mappings),
        },
        "workbook_path": _rel(workbook_path) if workbook_path else None,
        "workbook_structure": workbook_structure,
        "roi_nodes": roi_nodes,
        "virtual_nodes": virtual_nodes,
        "static_nodes": static_nodes,
        "mapping_nodes": mapping_nodes,
        "column_nodes": column_nodes_out,
        "edges": edges,
        "direct_edges": direct_edges,
        "sheet_grid": sheet_grid,
        "issues": issues,
        "stats": {
            "roi_count": len(roi_nodes),
            "mapping_count": len(mapping_nodes),
            "column_count": len(column_nodes_out),
            "edge_count": len(direct_edges),
            "unmapped_roi_count": unmapped_roi_count,
        },
    }


def _column_sort_key(col: str) -> tuple[int, str]:
    total = 0
    for ch in col.upper():
        if not ("A" <= ch <= "Z"):
            return (10_000, col)
        total = total * 26 + (ord(ch) - ord("A") + 1)
    return (total, col)


def create_app() -> Flask:
    app = Flask(__name__, static_folder=TOOLS_DIR / "static", template_folder=TOOLS_DIR / "templates")

    @app.route("/")
    def index():
        return send_file(TOOLS_DIR / "templates" / "pipeline_map.html")

    @app.route("/api/instance")
    def api_instance():
        return jsonify({"instance_id": SERVER_INSTANCE_ID})

    @app.route("/api/options")
    def api_options():
        releases = _available_releases()
        active_release = DATA_RELEASE if DATA_RELEASE in releases else (releases[0] if releases else DATA_RELEASE)
        release_arg = request.args.get("release")
        form_arg = request.args.get("form")
        try:
            selected_release = _safe_name(release_arg, "release") if release_arg else active_release
        except Exception:
            selected_release = active_release
        forms = _available_forms(selected_release)
        selected_form = form_arg if form_arg in forms else (forms[0] if forms else "")
        try:
            workbook = _rel(workbook_template_path(selected_release)) if selected_release else None
        except Exception:
            workbook = None
        return jsonify({
            "active_release": active_release,
            "selected_release": selected_release,
            "selected_form": selected_form,
            "releases": [
                {"release": rel, "forms": _available_forms(rel)}
                for rel in releases
            ],
            "workbook": workbook,
            "roots": {
                "templates": _rel(TEMPLATES_ROOT),
                "roi_schemas": _rel(ROI_SCHEMAS_ROOT),
                "mappings": _rel(MAPPINGS_ROOT),
            },
        })

    @app.route("/api/browse-root")
    def api_browse_root():
        try:
            kind = str(request.args.get("kind") or "template").strip().lower()
            return jsonify({"root": _rel(_browse_root(kind))})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/browse")
    def api_browse():
        try:
            kind = str(request.args.get("kind") or "template").strip().lower()
            rel_dir = request.args.get("dir", "").strip()
            dirs, files = _browse_files(kind, rel_dir)
            return jsonify({"dirs": dirs, "files": files})
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/template-info")
    def api_template_info():
        try:
            return jsonify(_template_info_from_path(request.args.get("path")))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.route("/api/visualization")
    def api_visualization():
        try:
            release = _safe_name(request.args.get("release") or DATA_RELEASE, "release")
            form = _safe_name(request.args.get("form"), "form")
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        try:
            return jsonify(build_visualization(release, form))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/template-image")
    def api_template_image():
        try:
            release = _safe_name(request.args.get("release") or DATA_RELEASE, "release")
            form = _safe_name(request.args.get("form"), "form")
            side = str(request.args.get("side") or "").strip().lower()
            if side not in {"a", "b"}:
                raise ValueError("Invalid side")
            path = _template_path(release, form, side)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        if not path.is_file():
            return jsonify({"error": f"Template image not found: {_rel(path)}"}), 404
        return send_file(path)

    return app


def run(port: int = 5006, debug: bool = False) -> None:
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=debug)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ROI-to-XLSX pipeline visualizer")
    p.add_argument("--port", type=int, default=5006, help="Port (default: 5006)")
    p.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = p.parse_args()
    print(f"Pipeline map visualizer: http://127.0.0.1:{args.port}/")
    run(port=args.port, debug=args.debug)
