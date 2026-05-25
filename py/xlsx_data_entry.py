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
from pathlib import Path
from typing import Any, Mapping

from config import PATHS, XLSX_DATA_ENTRY

try:
    from openpyxl.utils import column_index_from_string
    from openpyxl.worksheet.worksheet import Worksheet
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except ImportError as e:  # pragma: no cover
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

    Resolves to ``<mapping_dir>/<form_type>.json`` (e.g. ``data/xlsx_mappings/6post.json``).
    """
    cfg = cfg or XLSX_DATA_ENTRY
    root = Path(cfg.get("mapping_dir", PATHS["data"] / "xlsx_mappings"))
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
) -> Path:
    """Resolve the master template workbook from config."""
    _require_openpyxl()
    cfg = cfg or XLSX_DATA_ENTRY
    master = cfg.get("master_template_workbook")
    if master:
        mp = Path(master).resolve()
        if mp.is_file():
            return mp
    raise FileNotFoundError(
        'Set XLSX_DATA_ENTRY["master_template_workbook"] to an existing .xlsx'
    )


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------

_ITEM_LEVEL_SOURCES = frozenset({"id", "form_type", "page_odd", "page_even"})


def resolve_source(item: Mapping[str, Any], source: str) -> str | None:
    """
    Read a value from a pipeline item given a source string.

    - ``"id"``, ``"form_type"``, ``"page_odd"``, ``"page_even"`` read item-level fields.
    - Anything else (e.g. ``"1"``, ``"25"``) looks up ``data[]`` by ``name`` and returns ``text``.
    """
    if source in _ITEM_LEVEL_SOURCES:
        v = item.get(source)
        return str(v).strip() if v is not None else None

    for entry in item.get("data") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("name", "")).strip() == source:
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


def _write_cell(ws: Worksheet, row: int, col_letter: str, value: Any) -> None:
    if value is None:
        return
    # Guard against control characters that Excel/openpyxl reject.
    if isinstance(value, str):
        txt = value
        if ILLEGAL_CHARACTERS_RE is not None:
            txt = ILLEGAL_CHARACTERS_RE.sub("", txt)
        value = txt
    ws.cell(row=row, column=_col_letter_to_idx(col_letter), value=value)


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


# ---------------------------------------------------------------------------
# Mapping application
# ---------------------------------------------------------------------------

def apply_mapping_entry(
    ws: Worksheet,
    row: int,
    entry: Mapping[str, Any],
    item: Mapping[str, Any],
) -> None:
    """Apply one mapping entry to a worksheet row for a given pipeline item."""
    mtype = str(entry.get("type", "")).strip().lower()

    if mtype == "static":
        _write_cell(ws, row, str(entry["column"]), entry.get("value"))
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
        _write_cell(ws, row, str(entry["column"]), value)
        return

    if mtype == "lookup":
        lookup_map = entry.get("map") or {}
        key = raw.strip().lower()
        result = lookup_map.get(key) or lookup_map.get(raw)
        if result is None:
            result = entry.get("default")
        if result is not None:
            _write_cell(ws, row, str(entry["column"]), result)
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
                _write_cell(ws, row, str(no_answer_col), mark)
            return

        key = raw.strip().lower()
        target_col = choices.get(key) or choices.get(raw)
        if target_col:
            _write_cell(ws, row, str(target_col), mark)
        return

    raise ValueError(f"Unknown mapping type: {mtype!r}")


def fill_row(
    ws: Worksheet,
    row: int,
    mapping: Mapping[str, Any],
    item: Mapping[str, Any],
) -> None:
    """Apply all mappings to one row for a single pipeline item."""
    for entry in mapping.get("mappings") or []:
        if isinstance(entry, Mapping):
            apply_mapping_entry(ws, row, entry, item)


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


def fill_template(
    mapping: Mapping[str, Any],
    items: list[Mapping[str, Any]],
    output_path: str | Path,
    *,
    template_path: str | Path | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> Path:
    """
    Copy only the target sheet from the master template into a new workbook,
    stamp the start row for as many data rows as needed, write one row per
    pipeline item, and save to *output_path*.

    Returns the resolved output path.
    """
    _require_openpyxl()
    from copy import copy

    from openpyxl import Workbook, load_workbook

    cfg = cfg or XLSX_DATA_ENTRY
    tpl = Path(template_path).resolve() if template_path else resolve_template_path(cfg)
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

        for merged in src_ws.merged_cells.ranges:
            dst_ws.merge_cells(str(merged))

        for i, dim in src_ws.column_dimensions.items():
            dst_ws.column_dimensions[i].width = dim.width
        for i, dim in src_ws.row_dimensions.items():
            dst_ws.row_dimensions[i].height = dim.height

        _stamp_data_rows(dst_ws, src_ws, start_row, len(items))

        for i, item in enumerate(items):
            fill_row(dst_ws, start_row + i, mapping, item)

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
) -> Path:
    """
    Like ``fill_template`` but auto-generates a staging output path when
    *output_path* is ``None``.
    """
    cfg = cfg or XLSX_DATA_ENTRY
    form_type = str(mapping.get("form_type", "unknown"))
    out = Path(output_path).resolve() if output_path else make_staging_output_path(form_type, cfg)
    return fill_template(mapping, items, out, template_path=template_path, cfg=cfg)


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

    for form_type, src_path in staging_paths.items():
        src_wb = load_workbook(src_path)
        try:
            src_ws = src_wb.active
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

            for merged in src_ws.merged_cells.ranges:
                dst_ws.merge_cells(str(merged))
            for i, dim in src_ws.column_dimensions.items():
                dst_ws.column_dimensions[i].width = dim.width
            for i, dim in src_ws.row_dimensions.items():
                dst_ws.row_dimensions[i].height = dim.height
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
    cfg: Mapping[str, Any] | None = None,
    verbose: bool = True,
) -> Path | None:
    """
    Group pipeline items by ``form_type``, fill one staging workbook per type,
    then merge all sheets into a single output workbook.

    *items*: list of pipeline item dicts (each with ``form_type``, ``id``, ``data[]``).
    *output_path*: explicit path for the merged workbook.  When ``None``, auto-generates
      ``<output_dir>/<pdf_stem>.xlsx`` (or a timestamped name if *pdf_stem* is also ``None``).
    *pdf_stem*: used for the default output filename (e.g. ``"maury_1"``).

    Returns the path to the merged workbook, or ``None`` if no items had a form_type.
    """
    _require_openpyxl()
    cfg = cfg or XLSX_DATA_ENTRY

    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        ft = (item.get("form_type") or "").strip()
        if not ft:
            continue
        groups.setdefault(ft, []).append(item)

    if not groups:
        if verbose:
            print("  [xlsx] No items with form_type — skipping xlsx fill.")
        return None

    staging: dict[str, Path] = {}
    for ft, ft_items in groups.items():
        try:
            mapping = load_mapping(ft, cfg)
        except FileNotFoundError:
            if verbose:
                print(f"  [xlsx] No mapping for {ft!r} — skipping ({len(ft_items)} items).")
            continue

        staging_path = make_staging_output_path(ft, cfg)
        fill_template(mapping, ft_items, staging_path, cfg=cfg)
        staging[ft] = staging_path
        if verbose:
            print(f"  [xlsx] {ft}: {len(ft_items)} item(s) → {mapping['sheet']!r}")

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
        help="Form type (resolves to data/xlsx_mappings/<form_type>.json mapping file)",
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
        "--template",
        default=None,
        help="Override master template .xlsx path",
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
            template_path=args.template,
            cfg=cfg,
        )
    else:
        if not args.output_xlsx:
            print("output_xlsx is required unless --staging is set", file=sys.stderr)
            sys.exit(2)
        out = fill_template(
            mapping, items, args.output_xlsx,
            template_path=args.template,
            cfg=cfg,
        )

    print(f"Wrote {out}  ({len(items)} rows)")
