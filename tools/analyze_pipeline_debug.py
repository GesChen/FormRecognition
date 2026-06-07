#!/usr/bin/env python3
"""
Extract the pipeline-debug trace for a single page-pair item from a
``pdf_recognize`` ``*_debug.json`` file.

Items are 0-based: item 0 = pages 1–2, item 1 = pages 3–4, etc.

Usage::

    python3 tools/analyze_pipeline_debug.py output/recognition/my_pdf_debug.json --item 0
    python3 tools/analyze_pipeline_debug.py ... --item 2 -o testing/output/item2_debug.json
    python3 tools/analyze_pipeline_debug.py ... --item 0 --recognition output/recognition/my_pdf.json

Requires the debug file produced with ``pdf_recognize.py --debug``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _slice_id_form_step(s2: dict[str, Any], item_index: int) -> dict[str, Any]:
    """Subset step ``2_id_form`` to one side-a page (pair index)."""
    out2: dict[str, Any] = {}

    if "llm" in s2 and isinstance(s2["llm"], dict):
        llm = s2["llm"]
        if isinstance(llm.get("attempts"), list):
            filt = [
                a
                for a in llm["attempts"]
                if isinstance(a, dict) and a.get("page_index") == item_index
            ]
            out2["llm"] = {**llm, "attempts": filt}
        else:
            out2["llm"] = dict(llm)

    id_form = s2.get("id_form")
    if isinstance(id_form, dict):
        sliced: dict[str, Any] = {}
        ocr_per = id_form.get("ocr_per_page") or []
        if item_index < len(ocr_per):
            sliced["ocr_per_page"] = [ocr_per[item_index]]
        else:
            sliced["ocr_per_page"] = []
        llm_inner = id_form.get("llm")
        if isinstance(llm_inner, dict):
            new_llm = dict(llm_inner)
            pr = llm_inner.get("parsed_results") or []
            if item_index < len(pr):
                new_llm["parsed_results"] = [pr[item_index]]
            sliced["llm"] = new_llm
        for k in ("form_type_source",):
            if k in id_form:
                sliced[k] = id_form[k]
        out2["id_form"] = sliced

    return out2


def _slice_text_llm_deferred_step(
    deferred: dict[str, Any],
    page_indexes: tuple[int, ...],
) -> dict[str, Any]:
    """
    Subset step ``4_roi.text_llm_deferred`` to only the selected item pages.

    Per-ROI entries use names like ``p12:r3:16``. We filter those by page index
    while preserving top-level metadata fields.
    """
    out = dict(deferred)
    details = deferred.get("details")
    if not isinstance(details, dict):
        return out

    page_prefixes = tuple(f"p{idx}:" for idx in page_indexes)
    details_out = dict(details)

    per_roi = details.get("per_roi")
    if isinstance(per_roi, list):
        filtered_per_roi = []
        for row in per_roi:
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            if isinstance(name, str) and name.startswith(page_prefixes):
                filtered_per_roi.append(row)
        details_out["per_roi"] = filtered_per_roi

    result_map = details.get("result")
    if isinstance(result_map, dict):
        details_out["result"] = {
            k: v
            for k, v in result_map.items()
            if isinstance(k, str) and k.startswith(page_prefixes)
        }

    # Keep original run-level totals and add slice-level counts for clarity.
    total_slice = len(details_out.get("per_roi") or [])
    details_out["slice_total_text_rows"] = total_slice
    details_out["slice_page_indexes"] = list(page_indexes)

    out["details"] = details_out
    out["slice_total_text_rows"] = total_slice
    out["slice_page_indexes"] = list(page_indexes)
    return out


def extract_item_pipeline_debug(
    data: dict[str, Any],
    item_index: int,
    *,
    source_path: str | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable dict: same ``steps`` keys as the input debug file,
    but scoped to one item (one odd + one even page, or odd only if last page).

    *item_index*: 0-based index into page pairs (same order as ``items[]`` in recognition JSON).
    """
    steps = data.get("steps") or {}
    page_paths = (steps.get("1_pdf_to_images") or {}).get("page_paths") or []
    n_pages = len(page_paths)
    pi_odd = 2 * item_index
    pi_even = 2 * item_index + 1

    if n_pages == 0:
        raise ValueError("Debug JSON has no steps['1_pdf_to_images']['page_paths'].")
    if pi_odd >= n_pages:
        n_items = (n_pages + 1) // 2
        raise ValueError(
            f"item_index={item_index} out of range: PDF has {n_pages} page(s) "
            f"({n_items} item(s), indices 0..{n_items - 1})."
        )

    out_steps: dict[str, Any] = {}

    # 1 — raster paths for this pair only
    pair_paths = [page_paths[pi_odd]]
    if pi_even < n_pages:
        pair_paths.append(page_paths[pi_even])
    out_steps["1_pdf_to_images"] = {"page_paths": pair_paths, "count": len(pair_paths)}

    # 2 — ID / form-type (one entry per side-a page == per item)
    s2 = steps.get("2_id_form")
    if isinstance(s2, dict) and s2:
        out_steps["2_id_form"] = _slice_id_form_step(s2, item_index)

    # 3 — normalize: one entry per physical page in pair
    s3 = steps.get("3_normalize") or {}
    per_page = s3.get("per_page") or []
    pair_norm: list[Any] = []
    if pi_odd < len(per_page):
        pair_norm.append(per_page[pi_odd])
    if pi_even < len(per_page):
        pair_norm.append(per_page[pi_even])
    if pair_norm:
        out_steps["3_normalize"] = {"per_page": pair_norm}

    # 4 — ROI extraction: one entry per page
    s4 = steps.get("4_roi") or {}
    roi_per = s4.get("roi_per_page") or []
    pair_roi: list[Any] = []
    if pi_odd < len(roi_per):
        pair_roi.append(roi_per[pi_odd])
    if pi_even < len(roi_per):
        pair_roi.append(roi_per[pi_even])
    if pair_roi:
        out_steps["4_roi"] = {"roi_per_page": pair_roi}
    deferred = s4.get("text_llm_deferred")
    if isinstance(deferred, dict) and deferred:
        page_indexes = tuple(i for i in (pi_odd, pi_even) if i < n_pages)
        out_steps.setdefault("4_roi", {})
        out_steps["4_roi"]["text_llm_deferred"] = _slice_text_llm_deferred_step(
            deferred,
            page_indexes,
        )

    if "5_post_normalize" in steps:
        out_steps["5_post_normalize"] = steps["5_post_normalize"]

    # 6 — pointer to full output + which item this slice is
    s6 = steps.get("6_output") or steps.get("5_output") or {}
    out_steps["6_output"] = {
        "output_path": s6.get("output_path"),
        "item_count_full_run": s6.get("item_count"),
        "item_index_this_slice": item_index,
        "note": "This object describes one item; the full run wrote item_count_full_run items.",
    }

    if "7_xlsx" in steps:
        out_steps["7_xlsx"] = steps["7_xlsx"]
    elif "6_xlsx" in steps:
        out_steps["6_xlsx"] = steps["6_xlsx"]

    return {
        "meta": {
            "source_debug_path": source_path,
            "description": data.get("description"),
            "pdf_path": data.get("pdf_path"),
            "pdf_stem": data.get("pdf_stem"),
            "timestamp_start": data.get("timestamp_start"),
            "timestamp_end": data.get("timestamp_end"),
            "item_index": item_index,
            "page_odd_1based": item_index * 2 + 1,
            "page_even_1based": item_index * 2 + 2 if pi_even < n_pages else None,
            "slice_note": "Single page-pair slice of pdf_recognize --debug output.",
        },
        "steps": out_steps,
    }


def load_recognition_item(path: Path, item_index: int) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    items = payload.get("items")
    if not isinstance(items, list):
        return None
    if item_index < 0 or item_index >= len(items):
        return None
    it = items[item_index]
    return it if isinstance(it, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract pipeline debug steps for one item from *_debug.json.",
    )
    parser.add_argument(
        "debug_json",
        type=Path,
        help="Path to pdf_recognize debug JSON (*_debug.json)",
    )
    parser.add_argument(
        "--item",
        type=int,
        required=True,
        metavar="N",
        help="0-based item index (item 0 = pages 1–2)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path (default: stdout)",
    )
    parser.add_argument(
        "--recognition",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional recognition JSON (same run); merges items[N] under top-level 'item'",
    )
    args = parser.parse_args()

    path = args.debug_json.resolve()
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(data, dict):
        print("Error: debug JSON root must be an object.", file=sys.stderr)
        return 1

    item_index = int(args.item)
    if item_index < 0:
        print("Error: item index must be >= 0", file=sys.stderr)
        return 1

    try:
        out = extract_item_pipeline_debug(data, item_index, source_path=str(path))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.recognition is not None:
        rp = args.recognition.resolve()
        if not rp.is_file():
            print(f"Error: --recognition file not found: {rp}", file=sys.stderr)
            return 1
        merged = load_recognition_item(rp, item_index)
        if merged is None:
            print(
                f"Warning: no items[{item_index}] in {rp} (or missing items[]).",
                file=sys.stderr,
            )
        else:
            out["item"] = merged

    text = json.dumps(out, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output.resolve()}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
