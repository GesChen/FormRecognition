"""Test full PDF recognition workflow: pdf → images, normalize, id, metadata, MCQ; pair pages; JSON output."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from config import PDF_RECOGNITION
from pdf_recognize import run_workflow, _sanitize_pdf_stem

PDF_PATH = ROOT / "data" / "maury 1.pdf"


def main():
    if not PDF_PATH.exists():
        print(f"Skip: PDF not found: {PDF_PATH}")
        return
    print(f"Running workflow on: {PDF_PATH}")
    items = run_workflow(PDF_PATH, write_json=True)
    print(f"Recognized {len(items)} item(s).")
    out_dir = Path(PDF_RECOGNITION["output_dir"])
    stem = _sanitize_pdf_stem(PDF_PATH.name)
    out_file = out_dir / f"{stem}.json"
    assert out_file.exists(), f"Expected output file: {out_file}"
    data = json.loads(out_file.read_text())
    assert data == items
    for i, item in enumerate(items):
        assert "page_odd" in item and "id" in item and "metadata" in item and "mcq" in item
        print(f"  Item {i + 1}: pages {item['page_odd']}-{item.get('page_even') or '-'}, id={item['id']!r}")
    print(f"Output: {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
