"""
Fill Excel data-entry templates using human-authored mapping JSON files.

Mapping files (one per form_type, stored as ``data/<form_type>.json``) define
how pipeline recognition items map onto template columns.  Four mapping types:

- **direct** — write value as-is (optional transform: upper, lower, number, date)
- **lookup** — map value through a dictionary
- **multi_column** — value selects which column gets a mark
- **static** — fixed value every row

See ``docs/xlsx_mapping_spec.md`` for the full specification.

Usage::

    from xlsx_data_entry import load_mapping, fill_template, items_from_payload

    mapping = load_mapping("6post")
    items = items_from_payload(pipeline_json_dict)
    fill_template(mapping, items, "output/filled.xlsx")

CLI::

    python3 py/xlsx_data_entry.py 6post pipeline.json output.xlsx
    python3 py/xlsx_data_entry.py 6post pipeline.json --staging
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, datetime
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

from config import PATHS, XLSX_DATA_ENTRY
from form_sheet_map import form_sheet_name, workbook_template_path

try:
    from openpyxl.comments import Comment
    from openpyxl.utils import column_index_from_string
    from openpyxl.worksheet.worksheet import Worksheet
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except ImportError as e:  # pragma: no cover
    Comment = None  # type: ignore
    column_index_from_string = None  # type: ignore
    Worksheet = None  # type: ignore
    ILLEGAL_CHARACTERS_RE = None  # type: ignore
    _OPENPYXL_ERR = e
else:
    _OPENPYXL_ERR = None


def _require_openpyxl() -> None:
    if _OPENPYXL_ERR is not None:
        raise ImportError(
            "openpyxl is required for xlsx_data_entry. Install: pip install openpyxl"
        ) from _OPENPYXL_ERR


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_mapping(
    form_type: str,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load a mapping file for *form_type* from ``cfg["mapping_dir"]``.

    Resolves to ``<mapping_dir>/<form_type>.json`` (e.g. ``data/xlsx/mappings/2026/6post.json``).
    """
    cfg = cfg or XLSX_DATA_ENTRY
    root = Path(cfg.get("mapping_dir", PATHS["data"] / "xlsx" / "mappings"))
    name = f"{form_type.strip()}.json"
    p = (root / name).resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"Mapping not found: {p}  (form_type={form_type!r}, mapping_dir={root})"
        )
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Mapping root must be a JSON object: {p}")
    for key in ("form_type", "sheet", "start_row", "mappings"):
        if key not in data:
            raise ValueError(f"Mapping file missing required key {key!r}: {p}")
    configured_sheet = form_sheet_name(str(data.get("form_type") or form_type))
    if configured_sheet:
        data["sheet"] = configured_sheet
    data["_mapping_release"] = p.parent.name
    return data


def items_from_payload(
    payload: Any,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Extract pipeline items from a JSON payload.

    Accepts:

    - ``{"items": [ ... ]}`` — typical ``pdf_recognize`` output
    - A bare JSON array of items
    - A single item dict

    Returns a list of raw item dicts (each with ``id``, ``form_type``, ``data[]``).
    """
    seq: list[Any] = []
    if isinstance(payload, dict) and "items" in payload:
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        seq = list(items)
    elif isinstance(payload, list):
        seq = list(payload)
    elif isinstance(payload, dict):
        seq = [payload]
    else:
        return []

    out: list[dict[str, Any]] = []
    for x in seq:
        if not isinstance(x, dict):
            continue
        out.append(x)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def resolve_template_path(
    cfg: Mapping[str, Any] | None = None,
    *,
    release: str | None = None,
) -> Path:
    """Resolve the release-scoped master template workbook from data/."""
    _require_openpyxl()
    return workbook_template_path(release)


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

_ITEM_LEVEL_SOURCES = frozenset({"id", "form_type", "page_odd", "page_even"})


def _find_data_entry(item: Mapping[str, Any], source: str) -> Mapping[str, Any] | None:
    for entry in item.get("data") or []:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("name", "")).strip() == source:
            return entry
    return None


def resolve_source(item: Mapping[str, Any], source: str) -> str | None:
    """
    Read a value from a pipeline item given a source string.

    - ``"id"``, ``"form_type"``, ``"page_odd"``, ``"page_even"`` read item-level fields.
    - Anything else (e.g. ``"1"``, ``"25"``) looks up ``data[]`` by ``name`` and returns ``text``.
    """
    if source in _ITEM_LEVEL_SOURCES:
        v = item.get(source)
        return str(v).strip() if v is not None else None

    entry = _find_data_entry(item, source)
    if entry is not None:
        t = entry.get("text")
        if t is not None:
            s = str(t).strip()
            return s if s else None
        return None
    return None


# ---------------------------------------------------------------------------
# Cell writers / transforms
# ---------------------------------------------------------------------------

def _col_letter_to_idx(letter: str) -> int:
    if column_index_from_string is None:
        _require_openpyxl()
    return column_index_from_string(letter.strip().upper())


def _write_cell(ws: Worksheet, row: int, col_letter: str, value: Any):
    if value is None:
        return None
    # Guard against control characters that Excel/openpyxl reject.
    if isinstance(value, str):
        txt = value
        if ILLEGAL_CHARACTERS_RE is not None:
            txt = ILLEGAL_CHARACTERS_RE.sub("", txt)
        value = txt
    return ws.cell(row=row, column=_col_letter_to_idx(col_letter), value=value)


def _parse_date(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (datetime, date)):
        return v
    if isinstance(v, str):
        s = v.strip()
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
    return v


def _parse_number(v: str) -> int | float | str:
    try:
        return float(v) if "." in v else int(v)
    except (TypeError, ValueError):
        return v


def _pdf_pages_string(item: Mapping[str, Any]) -> str:
    odd = item.get("page_odd")
    even = item.get("page_even")
    if odd in (None, "") and even in (None, ""):
        return ""
    if even in (None, ""):
        return str(odd)
    return f"{odd}-{even}"


def _source_file_name(item: Mapping[str, Any], context: Mapping[str, Any] | None) -> str:
    for key in ("source_file_name", "file_name", "pdf_file_name", "pdf_name"):
        v = item.get(key)
        if v not in (None, ""):
            return Path(str(v)).name
    if context:
        for key in ("source_file_name", "file_name", "pdf_file_name", "pdf_name"):
            v = context.get(key)
            if v not in (None, ""):
                return Path(str(v)).name
        pdf_stem = str(context.get("pdf_stem") or "").strip()
        if pdf_stem:
            return pdf_stem
    return ""


def _format_confidence(row_meta: Mapping[str, Any] | None, item: Mapping[str, Any], source: str) -> tuple[str, str, str]:
    if row_meta:
        label = str(row_meta.get("ocr_confidence_label") or "").strip()
        score = row_meta.get("ocr_confidence_score")
    elif source == "id":
        label = str(item.get("id_ocr_confidence_label") or "").strip()
        score = item.get("id_ocr_confidence_score")
    else:
        label = ""
        score = None

    score_text = ""
    if score is not None:
        try:
            score_text = f"{float(score):.3f}"
        except (TypeError, ValueError):
            score_text = str(score)
    if label and score_text:
        return f"{label} ({score_text})", score_text, label
    return label or score_text, score_text, label


def _comment_template_and_replacements(comment_cfg: Any) -> tuple[str, dict[str, str]]:
    if isinstance(comment_cfg, str):
        return comment_cfg, {}
    if not isinstance(comment_cfg, Mapping):
        return "", {}

    template = str(
        comment_cfg.get("text")
        or comment_cfg.get("template")
        or comment_cfg.get("comment")
        or ""
    )
    replacements: dict[str, str] = {}
    raw_replacements = comment_cfg.get("replacements")
    if isinstance(raw_replacements, Mapping):
        replacements.update({str(k): "" if v is None else str(v) for k, v in raw_replacements.items()})
    elif isinstance(raw_replacements, list):
        for row in raw_replacements:
            if not isinstance(row, Mapping):
                continue
            find = row.get("find")
            if find is None:
                find = row.get("from")
            if find is None:
                continue
            value = row.get("text")
            if value is None:
                value = row.get("replace")
            if value is None:
                value = row.get("to")
            replacements[str(find)] = "" if value is None else str(value)
    return template, replacements


def render_comment_text(
    comment_cfg: Any,
    *,
    item: Mapping[str, Any],
    entry: Mapping[str, Any],
    source: str,
    raw: Any,
    value: Any,
    context: Mapping[str, Any] | None = None,
) -> str | None:
    """Render a mapping comment config into Excel comment text."""
    template, replacements = _comment_template_and_replacements(comment_cfg)
    if not template:
        return None

    row_meta = _find_data_entry(item, source) if source else None
    confidence, confidence_score, confidence_label = _format_confidence(row_meta, item, source)
    drop_ins = {
        "{value}": "" if value is None else str(value),
        "{raw}": "" if raw is None else str(raw),
        "{source}": source,
        "{type}": str(entry.get("type") or ""),
        "{ocr_confidence}": confidence,
        "{ocr_confidence_score}": confidence_score,
        "{ocr_confidence_label}": confidence_label,
        "{file_name}": _source_file_name(item, context),
        "{pdf_pages}": _pdf_pages_string(item),
        "{page_numbers}": _pdf_pages_string(item),
        "{page_odd}": "" if item.get("page_odd") is None else str(item.get("page_odd")),
        "{page_even}": "" if item.get("page_even") is None else str(item.get("page_even")),
    }
    if context:
        pdf_stem = str(context.get("pdf_stem") or "").strip()
        if pdf_stem:
            drop_ins["{pdf_stem}"] = pdf_stem
    for key, replacement in replacements.items():
        template = template.replace(key, replacement)
    for key, replacement in drop_ins.items():
        template = template.replace(key, replacement)
    text = template.strip()
    return text or None


def _apply_cell_comment(
    cell: Any,
    entry: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    source: str,
    raw: Any,
    value: Any,
    context: Mapping[str, Any] | None = None,
) -> None:
    comment_cfg = entry.get("comment")
    if not comment_cfg or cell is None:
        return
    if Comment is None:
        _require_openpyxl()
    text = render_comment_text(
        comment_cfg,
        item=item,
        entry=entry,
        source=source,
        raw=raw,
        value=value,
        context=context,
    )
    if text:
        cell.comment = Comment(text, str(entry.get("comment_author") or "EVMS OCR"))


# ---------------------------------------------------------------------------
# Mapping application
# ---------------------------------------------------------------------------

def apply_mapping_entry(
    ws: Worksheet,
    row: int,
    entry: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Apply one mapping entry to a worksheet row for a given pipeline item."""
    mtype = str(entry.get("type", "")).strip().lower()

    if mtype == "static":
        value = entry.get("value")
        cell = _write_cell(ws, row, str(entry["column"]), value)
        _apply_cell_comment(cell, entry, item, source="", raw=value, value=value, context=context)
        return

    source = entry.get("source")
    if not source:
        return
    raw = resolve_source(item, str(source))
    if raw is None and mtype != "multi_column":
        return

    if mtype == "direct":
        value: Any = raw
        transform = str(entry.get("transform", "")).strip().lower()
        if transform == "upper":
            value = raw.upper()
        elif transform == "lower":
            value = raw.lower()
        elif transform == "number":
            value = _parse_number(raw)
        elif transform == "date":
            value = _parse_date(raw) or raw
        cell = _write_cell(ws, row, str(entry["column"]), value)
        _apply_cell_comment(cell, entry, item, source=str(source), raw=raw, value=value, context=context)
        return

    if mtype == "lookup":
        lookup_map = entry.get("map") or {}
        key = raw.strip().lower()
        result = lookup_map.get(key) or lookup_map.get(raw)
        if result is None:
            result = entry.get("default")
        if result is not None:
            cell = _write_cell(ws, row, str(entry["column"]), result)
            _apply_cell_comment(cell, entry, item, source=str(source), raw=raw, value=result, context=context)
        return

    if mtype == "multi_column":
        choices = entry.get("choices") or {}
        mark = entry.get("mark", "Yes")
        blank = entry.get("blank")
        no_answer_col = entry.get("no_answer")

        if blank is not None:
            for col in choices.values():
                _write_cell(ws, row, str(col), blank)

        if raw is None:
            if no_answer_col:
                cell = _write_cell(ws, row, str(no_answer_col), mark)
                _apply_cell_comment(
                    cell, entry, item, source=str(source), raw=raw, value=mark, context=context
                )
            return

        key = raw.strip().lower()
        target_col = choices.get(key) or choices.get(raw)
        if target_col:
            cell = _write_cell(ws, row, str(target_col), mark)
            _apply_cell_comment(cell, entry, item, source=str(source), raw=raw, value=mark, context=context)
        return

    raise ValueError(f"Unknown mapping type: {mtype!r}")


def fill_row(
    ws: Worksheet,
    row: int,
    mapping: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Apply all mappings to one row for a single pipeline item."""
    for entry in mapping.get("mappings") or []:
        if isinstance(entry, Mapping):
            apply_mapping_entry(ws, row, entry, item, context=context)


# ---------------------------------------------------------------------------
# Staging helpers
# ---------------------------------------------------------------------------

def make_staging_output_path(
    form_type: str,
    cfg: Mapping[str, Any] | None = None,
) -> Path:
    """Build a unique path under ``cfg["staging_dir"]`` for one staging workbook."""
    cfg = cfg or XLSX_DATA_ENTRY
    root = Path(cfg.get("staging_dir", PATHS["cache"] / "xlsx_staging"))
    root.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.-]+", "_", str(form_type).strip().lower() or "unknown")
    return root / f"xlsx_staging_{safe}_{time.time_ns()}.xlsx"


# ---------------------------------------------------------------------------
# High-level fill
# ---------------------------------------------------------------------------

def _stamp_data_rows(dst_ws: Worksheet, src_ws: Worksheet, start_row: int, count: int) -> None:
    """
    Copy the start row's cell values, styles, and row height from *src_ws*
    into *dst_ws* for *count* consecutive rows beginning at *start_row*.

    Rows that already exist in *dst_ws* are overwritten with the template
    row so every data row has identical formatting before data is filled.
    """
    from copy import copy

    max_col = src_ws.max_column or 1
    src_dim = src_ws.row_dimensions.get(start_row)

    for offset in range(count):
        dst_row = start_row + offset
        if src_dim and src_dim.height:
            dst_ws.row_dimensions[dst_row].height = src_dim.height
        for col_idx in range(1, max_col + 1):
            src_cell = src_ws.cell(row=start_row, column=col_idx)
            dst_cell = dst_ws.cell(row=dst_row, column=col_idx, value=src_cell.value)
            if src_cell.has_style:
                dst_cell.font = copy(src_cell.font)
                dst_cell.border = copy(src_cell.border)
                dst_cell.fill = copy(src_cell.fill)
                dst_cell.number_format = src_cell.number_format
                dst_cell.protection = copy(src_cell.protection)
                dst_cell.alignment = copy(src_cell.alignment)
            if src_cell.comment:
                dst_cell.comment = copy(src_cell.comment)


def _freeze_header_rows(ws: Worksheet, start_row: int) -> None:
    """Freeze rows above the first data row so headers stay visible in Excel."""
    if start_row > 1:
        ws.freeze_panes = f"A{start_row}"
    else:
        ws.freeze_panes = None


def original_sheet_name(form_type: str) -> str:
    """
    Concise original/raw sheet name. Excel sheet titles are limited to 31 chars.
    """
    ft = str(form_type or "").strip().lower()
    m = re.match(r"^([678h])(pre|post)$", ft)
    if not m:
        base = re.sub(r"[\[\]:*?/\\]", " ", str(form_type or "Data")).strip() or "Data"
        base = " ".join(base.split())[:20].strip() or "Data"
        return f"{base} (Original)"[:31]
    grade, timing = m.group(1), m.group(2)
    grade_label = {"6": "6th", "7": "7th", "8": "8th", "h": "HS"}[grade]
    timing_label = "Pre" if timing == "pre" else "Post"
    return f"{grade_label} {timing_label} Data (Original)"


def _copy_sheet_structure(
    src_ws: Worksheet,
    dst_ws: Worksheet,
    *,
    start_row: int,
    row_count: int,
) -> None:
    """
    Copy source worksheet cells/styles/dimensions into destination and stamp data rows.
    """
    from copy import copy

    for row in src_ws.iter_rows():
        for cell in row:
            dst_cell = dst_ws.cell(row=cell.row, column=cell.column, value=cell.value)
            if cell.has_style:
                dst_cell.font = copy(cell.font)
                dst_cell.border = copy(cell.border)
                dst_cell.fill = copy(cell.fill)
                dst_cell.number_format = cell.number_format
                dst_cell.protection = copy(cell.protection)
                dst_cell.alignment = copy(cell.alignment)
            if cell.comment:
                dst_cell.comment = copy(cell.comment)

    for merged in src_ws.merged_cells.ranges:
        dst_ws.merge_cells(str(merged))

    for i, dim in src_ws.column_dimensions.items():
        dst_ws.column_dimensions[i].width = dim.width
    for i, dim in src_ws.row_dimensions.items():
        dst_ws.row_dimensions[i].height = dim.height

    _stamp_data_rows(dst_ws, src_ws, start_row, row_count)
    _freeze_header_rows(dst_ws, start_row)


def fill_template(
    mapping: Mapping[str, Any],
    items: list[Mapping[str, Any]],
    output_path: str | Path,
    *,
    template_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """
    Copy only the target sheet from the master template into a new workbook,
    stamp the start row for as many data rows as needed, write one row per
    pipeline item, and save to *output_path*.

    Returns the resolved output path.
    """
    _require_openpyxl()
    from openpyxl import Workbook, load_workbook

    cfg = cfg or XLSX_DATA_ENTRY
    context = context or {}
    tpl = Path(template_path).resolve() if template_path else resolve_template_path(
        cfg,
        release=str(mapping.get("_mapping_release") or "").strip() or None,
    )
    out = Path(output_path).resolve()

    if not tpl.is_file():
        raise FileNotFoundError(f"Template not found: {tpl}")

    sheet_name = str(mapping["sheet"])
    start_row = int(mapping.get("start_row", 4))

    src_wb = load_workbook(tpl)
    try:
        if sheet_name not in src_wb.sheetnames:
            raise KeyError(f"No sheet {sheet_name!r} in workbook; have {src_wb.sheetnames}")
        src_ws = src_wb[sheet_name]

        dst_wb = Workbook()
        dst_ws = dst_wb.active
        dst_ws.title = sheet_name

        _copy_sheet_structure(src_ws, dst_ws, start_row=start_row, row_count=len(items))

        for i, item in enumerate(items):
            fill_row(dst_ws, start_row + i, mapping, item, context=context)

        out.parent.mkdir(parents=True, exist_ok=True)
        dst_wb.save(out)
        dst_wb.close()
    finally:
        src_wb.close()
    return out


def fill_template_pair(
    mapping: Mapping[str, Any],
    items: list[Mapping[str, Any]],
    original_items: list[Mapping[str, Any]],
    output_path: str | Path,
    *,
    template_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """
    Fill one staging workbook with normalized and original sheets for a form type.
    """
    _require_openpyxl()
    from openpyxl import Workbook, load_workbook

    cfg = cfg or XLSX_DATA_ENTRY
    context = context or {}
    tpl = Path(template_path).resolve() if template_path else resolve_template_path(
        cfg,
        release=str(mapping.get("_mapping_release") or "").strip() or None,
    )
    out = Path(output_path).resolve()

    if not tpl.is_file():
        raise FileNotFoundError(f"Template not found: {tpl}")

    sheet_name = str(mapping["sheet"])
    start_row = int(mapping.get("start_row", 4))
    original_name = original_sheet_name(str(mapping.get("form_type", "")))

    src_wb = load_workbook(tpl)
    try:
        if sheet_name not in src_wb.sheetnames:
            raise KeyError(f"No sheet {sheet_name!r} in workbook; have {src_wb.sheetnames}")
        src_ws = src_wb[sheet_name]

        dst_wb = Workbook()
        norm_ws = dst_wb.active
        norm_ws.title = sheet_name
        _copy_sheet_structure(src_ws, norm_ws, start_row=start_row, row_count=len(items))
        for i, item in enumerate(items):
            fill_row(norm_ws, start_row + i, mapping, item, context=context)

        original_ws = dst_wb.create_sheet(title=original_name)
        _copy_sheet_structure(src_ws, original_ws, start_row=start_row, row_count=len(original_items))
        for i, item in enumerate(original_items):
            fill_row(original_ws, start_row + i, mapping, item, context=context)

        out.parent.mkdir(parents=True, exist_ok=True)
        dst_wb.save(out)
        dst_wb.close()
    finally:
        src_wb.close()
    return out


def fill_staging(
    mapping: Mapping[str, Any],
    items: list[Mapping[str, Any]],
    *,
    output_path: str | Path | None = None,
    template_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """
    Like ``fill_template`` but auto-generates a staging output path when
    *output_path* is ``None``.
    """
    cfg = cfg or XLSX_DATA_ENTRY
    form_type = str(mapping.get("form_type", "unknown"))
    out = Path(output_path).resolve() if output_path else make_staging_output_path(form_type, cfg)
    return fill_template(mapping, items, out, template_path=template_path, cfg=cfg, context=context)


# ---------------------------------------------------------------------------
# Multi-type merge
# ---------------------------------------------------------------------------

def merge_workbooks(
    staging_paths: dict[str, Path],
    output_path: str | Path,
) -> Path:
    """
    Merge single-sheet staging workbooks into one output workbook.

    *staging_paths* maps ``form_type`` to a staging ``.xlsx`` that contains
    exactly one data sheet (produced by ``fill_template``).  Each sheet is
    copied into the final workbook with its original tab name.

    Returns the resolved output path.
    """
    _require_openpyxl()
    from copy import copy

    from openpyxl import Workbook, load_workbook

    out = Path(output_path).resolve()
    dst_wb = Workbook()
    created_default = True

    for _form_type, src_path in staging_paths.items():
        src_wb = load_workbook(src_path)
        try:
            for src_ws in src_wb.worksheets:
                if src_ws is None:
                    continue
                if created_default:
                    dst_ws = dst_wb.active
                    dst_ws.title = src_ws.title
                    created_default = False
                else:
                    dst_ws = dst_wb.create_sheet(title=src_ws.title)

                for row in src_ws.iter_rows():
                    for cell in row:
                        dst_cell = dst_ws.cell(
                            row=cell.row, column=cell.column, value=cell.value,
                        )
                        if cell.has_style:
                            dst_cell.font = copy(cell.font)
                            dst_cell.border = copy(cell.border)
                            dst_cell.fill = copy(cell.fill)
                            dst_cell.number_format = cell.number_format
                            dst_cell.protection = copy(cell.protection)
                            dst_cell.alignment = copy(cell.alignment)
                        if cell.comment:
                            dst_cell.comment = copy(cell.comment)

                for merged in src_ws.merged_cells.ranges:
                    dst_ws.merge_cells(str(merged))
                for i, dim in src_ws.column_dimensions.items():
                    dst_ws.column_dimensions[i].width = dim.width
                for i, dim in src_ws.row_dimensions.items():
                    dst_ws.row_dimensions[i].height = dim.height
                dst_ws.freeze_panes = src_ws.freeze_panes
        finally:
            src_wb.close()

    if created_default:
        raise ValueError("No staging workbooks to merge.")

    out.parent.mkdir(parents=True, exist_ok=True)
    dst_wb.save(out)
    dst_wb.close()
    return out


def fill_from_pipeline(
    items: list[dict[str, Any]],
    output_path: str | Path | None = None,
    *,
    pdf_stem: str | None = None,
    source_file_name: str | None = None,
    cfg: Mapping[str, Any] | None = None,
    verbose: bool = True,
    original_items: list[dict[str, Any]] | None = None,
) -> Path | None:
    """
    Group pipeline items by ``form_type``, fill one staging workbook per type,
    then merge all sheets into a single output workbook.

    *items*: list of pipeline item dicts (each with ``form_type``, ``id``, ``data[]``).
    *original_items*: optional pre-post-normalization item list. When provided,
      each form_type staging workbook includes a normalized sheet and an
      ``(Original)`` sheet.
    *output_path*: explicit path for the merged workbook.  When ``None``, auto-generates
      ``<output_dir>/<pdf_stem>.xlsx`` (or a timestamped name if *pdf_stem* is also ``None``).
    *pdf_stem*: used for the default output filename (e.g. ``"maury_1"``).
    *source_file_name*: optional original PDF filename for XLSX comment drop-ins.

    Returns the path to the merged workbook, or ``None`` if no items had a form_type.
    """
    _require_openpyxl()
    cfg = cfg or XLSX_DATA_ENTRY
    comment_context = {
        "pdf_stem": str(pdf_stem or "").strip(),
        "source_file_name": str(source_file_name or "").strip(),
    }

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        ft = (item.get("form_type") or "").strip()
        if not ft:
            continue
        groups.setdefault(ft, []).append(item)

    original_groups: dict[str, list[dict[str, Any]]] = {}
    if original_items is not None:
        for item in original_items:
            ft = (item.get("form_type") or "").strip()
            if not ft:
                continue
            original_groups.setdefault(ft, []).append(item)

    if not groups:
        if verbose:
            print("  [xlsx] No items with form_type — skipping xlsx fill.")
        return None

    staging: dict[str, Path] = OrderedDict()
    for ft, ft_items in groups.items():
        try:
            mapping = load_mapping(ft, cfg)
        except FileNotFoundError:
            if verbose:
                print(f"  [xlsx] No mapping for {ft!r} — skipping ({len(ft_items)} items).")
            continue

        staging_path = make_staging_output_path(ft, cfg)
        ft_original_items = original_groups.get(ft) if original_items is not None else None
        if ft_original_items is not None:
            fill_template_pair(mapping, ft_items, ft_original_items, staging_path, cfg=cfg, context=comment_context)
        else:
            fill_template(mapping, ft_items, staging_path, cfg=cfg, context=comment_context)
        staging[ft] = staging_path
        if verbose:
            suffix = ""
            if ft_original_items is not None:
                suffix = f" + {original_sheet_name(ft)!r}"
            print(f"  [xlsx] {ft}: {len(ft_items)} item(s) → {mapping['sheet']!r}{suffix}")

    if not staging:
        if verbose:
            print("  [xlsx] No mappings matched any form_type — no output.")
        return None

    if output_path is None:
        out_dir = Path(cfg.get("output_dir", PATHS["output"] / "xlsx"))
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{pdf_stem}.xlsx" if pdf_stem else f"xlsx_output_{time.time_ns()}.xlsx"
        output_path = out_dir / name

    merged = merge_workbooks(staging, output_path)
    if verbose:
        print(f"  [xlsx] Merged {len(staging)} sheet(s) → {merged}")
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    _py_dir = Path(__file__).resolve().parent
    if str(_py_dir) not in sys.path:
        sys.path.insert(0, str(_py_dir))

    parser = argparse.ArgumentParser(
        description="Fill an Excel template from a mapping file + pipeline JSON.",
    )
    parser.add_argument(
        "form_type",
        help="Form type (resolves to data/xlsx/mappings/<release>/<form_type>.json mapping file)",
    )
    parser.add_argument(
        "pipeline_json",
        help="Path to pipeline output JSON (items[]), or '-' for stdin",
    )
    parser.add_argument(
        "output_xlsx",
        nargs="?",
        default=None,
        help="Output workbook path (required unless --staging)",
    )
    parser.add_argument(
        "--staging",
        action="store_true",
        help="Auto-generate output path under staging_dir",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max items to process",
    )
    args = parser.parse_args()

    cfg = XLSX_DATA_ENTRY
    mapping = load_mapping(args.form_type, cfg)

    if args.pipeline_json == "-":
        payload = json.load(sys.stdin)
    else:
        with open(args.pipeline_json, encoding="utf-8") as f:
            payload = json.load(f)

    items = items_from_payload(payload, limit=args.limit)
    if not items:
        print("No items found in pipeline JSON.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded mapping for {args.form_type!r} → sheet {mapping['sheet']!r}")
    print(f"  {len(items)} item(s), {len(mapping.get('mappings', []))} mapping(s)")

    if args.staging:
        out = fill_staging(
            mapping, items,
            cfg=cfg,
        )
    else:
        if not args.output_xlsx:
            print("output_xlsx is required unless --staging is set", file=sys.stderr)
            sys.exit(2)
        out = fill_template(
            mapping, items, args.output_xlsx,
            cfg=cfg,
        )

    print(f"Wrote {out}  ({len(items)} rows)")
