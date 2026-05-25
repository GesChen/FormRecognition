"""Run legacy ID recognition with OCR engine on 10 odd-numbered pages (1, 3, 5, ..., 19)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "py" / "legacy"))

from config_legacy import ID_RECOGNITION
ID_RECOGNITION["ocr_engine"] = "ocr_engine"

from image_normalize import normalize_images, template_key_for_page
from id_recognize import id_recognize

CACHE_DIR = ROOT / "output" / "cache" / "maury_1"
# 10 odd-numbered pages: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
ODD_PAGE_INDICES = range(1, 90, 2)

if __name__ == "__main__":
    paths = [CACHE_DIR / f"page_{i:04d}.png" for i in ODD_PAGE_INDICES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("Missing cache images:", missing[0])
        print("Run testing/test_pdf_to_images.py first.")
        sys.exit(1)

    print("Step 1: Normalizing 10 odd-numbered pages (1,3,...,19)...")
    page_keys = [template_key_for_page(int(Path(p).stem.split("_")[1])) for p in paths]
    normalized = normalize_images(paths, template_key=page_keys)
    print(f"  -> {len(normalized)} normalized")

    print("Step 2: ID recognition (OCR engine) on normalized images...")
    results = []
    for path in normalized:
        try:
            text = id_recognize(path)
            results.append(text.strip() if text else "(unidentified)")
        except Exception as e:
            results.append(f"(unidentified: {e})")

    print("\nResults (page -> id text):")
    print("-" * 60)
    for i, (norm_path, text) in enumerate(zip(normalized, results), start=1):
        name = norm_path.name
        print(f"  {i:2}. {name}: {repr(text)}")
    print("-" * 60)
