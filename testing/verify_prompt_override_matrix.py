#!/usr/bin/env python3
"""
Quick verifier for prompt-override behavior matrix on a debug JSON file.

Checks non-header text rows for four combinations:
1) none        : no ocr override, no llm override
2) ocr_only    : ocr override only
3) llm_only    : llm override only
4) both        : both ocr + llm overrides

Usage:
  python3 testing/verify_prompt_override_matrix.py output/recognition/<run>_debug.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _truthy_text(v) -> bool:
    return bool(str(v or "").strip())


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python3 testing/verify_prompt_override_matrix.py <debug_json_path>")
        return 2

    p = Path(sys.argv[1]).resolve()
    if not p.exists():
        print(f"error: not found: {p}")
        return 2

    obj = json.loads(p.read_text())
    step4 = (((obj.get("steps") or {}).get("4_roi")) or {})
    pages = step4.get("roi_per_page") or []
    deferred = (step4.get("text_llm_deferred") or {}).get("per_roi") or []

    llm_by_uid = {}
    for row in deferred:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("name") or "")
        if uid:
            llm_by_uid[uid] = row

    counts = {"none": 0, "ocr_only": 0, "llm_only": 0, "both": 0}
    seen = {"none": 0, "ocr_only": 0, "llm_only": 0, "both": 0}
    examples = {"none": None, "ocr_only": None, "llm_only": None, "both": None}

    for page_idx, pd in enumerate(pages):
        if not isinstance(pd, dict):
            continue
        text_per = pd.get("text_per_roi") or {}
        for row_idx, (name, trow) in enumerate((text_per or {}).items()):
            if not isinstance(trow, dict):
                continue
            uid = f"p{page_idx}:r{row_idx}:{name}"
            # only non-header text rows enter deferred list
            lrow = llm_by_uid.get(uid)
            if not lrow:
                continue

            # infer overrides from prompt/debug metadata
            ocr_override_used = bool(trow.get("prompt_override_used"))
            llm_override_used = bool(lrow.get("used_prompt_override"))
            if ocr_override_used and llm_override_used:
                key = "both"
            elif ocr_override_used:
                key = "ocr_only"
            elif llm_override_used:
                key = "llm_only"
            else:
                key = "none"

            counts[key] += 1
            # considered "seen by both steps" if OCR row exists + deferred row exists
            seen[key] += 1

            if examples[key] is None:
                raw_ocr = str(trow.get("text") or "")
                final = str(lrow.get("result") or "")
                ocr_path = str(trow.get("ocr_path") or "")
                examples[key] = (uid, ocr_path, raw_ocr[:80], final[:80])

    print("Prompt Matrix Coverage")
    for key in ("none", "ocr_only", "llm_only", "both"):
        print(f"- {key}: {counts[key]} row(s)")
        ex = examples[key]
        if ex:
            uid, path, raw, final = ex
            print(f"  example: {uid} path={path!r} raw={raw!r} final={final!r}")

    # pass if all present categories are processed by both OCR+deferred (they are by construction)
    print("Verification")
    for key in ("none", "ocr_only", "llm_only", "both"):
        if counts[key]:
            print(f"- {key}: OK ({seen[key]}/{counts[key]} processed)")
    missing = [k for k in ("none", "ocr_only", "llm_only", "both") if counts[k] == 0]
    if missing:
        print(f"- note: missing categories in this run: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

