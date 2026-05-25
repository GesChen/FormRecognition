#!/usr/bin/env python3
"""
Test xlsx_data_entry: fill a template from a mapping JSON + pipeline JSON.

Run from project root.  All generated files go under testing/output/.

Usage::

    # Defaults: form_type 6post, sample pipeline JSON, output xlsx
    python3 testing/test_xlsx_data_entry.py

    python3 testing/test_xlsx_data_entry.py --form-type 6post \
        -r output/recognition/post_crossroads_6_Davis.json -o testing/output/filled.xlsx

    python3 testing/test_xlsx_data_entry.py --form-type 6post -n 3

    python3 testing/test_xlsx_data_entry.py --staging --form-type 6post
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

OUTPUT_DIR = ROOT / "testing" / "output"

DEFAULT_FORM_TYPE = "6post"
DEFAULT_RECORDS = ROOT / "output" / "recognition" / "post_crossroads_6_Davis.json"
DEFAULT_OUT = OUTPUT_DIR / "xlsx_data_entry_test.xlsx"


def main() -> int:
    p = argparse.ArgumentParser(
        description="Fill an xlsx using xlsx_data_entry (mapping + pipeline JSON).",
    )
    p.add_argument(
        "--form-type",
        default=DEFAULT_FORM_TYPE,
        metavar="TYPE",
        help=f"Form type — resolves to data/<TYPE>.json (default: {DEFAULT_FORM_TYPE})",
    )
    p.add_argument(
        "-r",
        "--records",
        type=Path,
        default=DEFAULT_RECORDS,
        help="Pipeline output JSON with items[] (default: post_crossroads_6_Davis)",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help=f"Output workbook (default: {DEFAULT_OUT}; with --staging → auto name)",
    )
    p.add_argument("--template", type=Path, default=None, help="Override template .xlsx path")
    p.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Use only first N items",
    )
    p.add_argument(
        "--staging",
        action="store_true",
        help="Use fill_staging; writes under testing/output/.",
    )
    args = p.parse_args()
    if args.out is None and not args.staging:
        args.out = DEFAULT_OUT

    from config import XLSX_DATA_ENTRY
    from xlsx_data_entry import (
        fill_staging,
        fill_template,
        items_from_payload,
        load_mapping,
        resolve_template_path,
    )

    rec_path = args.records.resolve()
    if not rec_path.is_file():
        print(
            f"Pipeline JSON not found: {rec_path}\n"
            "Pass -r PATH to a recognition output JSON.",
            file=sys.stderr,
        )
        return 1

    with rec_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    items = items_from_payload(payload, limit=args.limit)
    print(f"Loaded {len(items)} pipeline item(s).")

    if not items:
        print("No items to write.", file=sys.stderr)
        return 1

    mapping = load_mapping(args.form_type)
    cfg = dict(XLSX_DATA_ENTRY)
    if args.staging:
        cfg["staging_dir"] = OUTPUT_DIR

    tpl = Path(args.template).resolve() if args.template else resolve_template_path(cfg)
    print(f"Template:  {tpl}")
    print(f"Mapping:   data/xlsx_mappings/{args.form_type}.json → sheet {mapping['sheet']!r}")
    print(f"  {len(mapping.get('mappings', []))} mapping entries")

    if args.staging:
        out_path = args.out.resolve() if args.out is not None else None
        out = fill_staging(
            mapping,
            items,
            output_path=out_path,
            cfg=cfg,
            template_path=args.template,
        )
        print(f"Staging:   {out} ({out.stat().st_size} bytes)")
        return 0

    assert args.out is not None
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    fill_template(
        mapping, items, out,
        template_path=args.template,
        cfg=cfg,
    )
    print(f"Wrote:     {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
