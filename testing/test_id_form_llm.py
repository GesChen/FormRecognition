"""
Test ID + form type extraction (top crop -> OCR engine -> LLM).

Usage (from project root):

    python3 testing/test_id_form_llm.py <image_path>
    python3 testing/test_id_form_llm.py <image_path> [image_path ...]

Single image: one LLM call, prints id, form_type, and raw OCR-stage output.
Multiple paths: batch mode, one LLM call, prints one result per image.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from id_form_llm import extract_id_and_form_type, extract_id_and_form_type_batch


def _serialize_raw_ocr(raw_obj):
    """Convert raw OCR output to JSON-serializable form for printing."""
    try:
        return json.loads(json.dumps(raw_obj, default=str))
    except Exception:
        return {"__repr__": repr(raw_obj)}


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 testing/test_id_form_llm.py <image_path> [image_path ...]")
        sys.exit(1)

    paths = [Path(p).resolve() for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"Error: not found: {p}")
            sys.exit(1)

    if len(paths) == 1:
        print(f"Single image: {paths[0]}")
        print("-" * 60)
        t0 = time.perf_counter()
        result = extract_id_and_form_type(paths[0], return_raw_ocr=True)
        elapsed = time.perf_counter() - t0
        print(f"id:         {result.get('id')}")
        print(f"form_type:  {result.get('form_type')}")
        print(f"[DEBUG] time: {elapsed:.2f}s")
        if "raw_ocr" in result:
            print("-" * 60)
            print("Raw OCR output:")
            serialized = _serialize_raw_ocr(result["raw_ocr"])
            print(json.dumps(serialized, indent=2, default=str))
    else:
        print(f"Batch: {len(paths)} images")
        for i, p in enumerate(paths, start=1):
            print(f"  {i}. {p}")
        print("-" * 60)
        t0 = time.perf_counter()
        results = extract_id_and_form_type_batch(paths)
        elapsed = time.perf_counter() - t0
        for i, (p, r) in enumerate(zip(paths, results), start=1):
            print(f"  {i}. {p.name}: id={r.get('id')!r}, form_type={r.get('form_type')!r}")
        print(f"[DEBUG] time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
