"""
Convert a PDF to page images (if needed) and normalize them (odd=a, even=b template).

Uses config: PDF_TO_IMAGES["cache_root"], IMAGE_NORMALIZE["cache_root"], template_key per page.
Output: normalized/<pdf_stem>/page_<N>_bin.png under the normalized cache.

Usage (from project root):
    python3 testing/test_image_normalize.py [pdf_path]
    Default PDF: data/maury 1.pdf
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from config import PDF_TO_IMAGES
from image_normalize import normalize_image, template_key_for_page
from pdf_to_images import pdf_to_images, _sanitize_name

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, unit=None, **kwargs):
        return iterable


def _pdf_page_cache_dir(pdf_path: Path) -> Path:
    """Directory where pdf_to_images stores page images for this PDF."""
    return Path(PDF_TO_IMAGES["cache_root"]) / _sanitize_name(pdf_path.name)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        pdf_path = Path(sys.argv[1]).resolve()
    else:
        pdf_path = ROOT / "data" / "maury 1.pdf"

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        print("Usage: python3 testing/test_image_normalize.py [pdf_path]")
        sys.exit(1)

    cache_dir = _pdf_page_cache_dir(pdf_path)
    paths = sorted(cache_dir.glob("page_*.png"))
    if not paths:
        print(f"No page images in {cache_dir}. Converting PDF to images...")
        paths = pdf_to_images(pdf_path, use_tqdm=True)
        if not paths:
            print("No pages produced.")
            sys.exit(1)
        paths = sorted(paths)

    cache_subdir = _sanitize_name(pdf_path.name)
    print(f"Normalizing {len(paths)} page(s) from {pdf_path.name} (odd=a, even=b)...")
    page_keys = [template_key_for_page(int(p.stem.split("_")[1])) for p in paths]
    normalized = []
    for path, key in tqdm(list(zip(paths, page_keys)), desc="Normalize", unit="page"):
        out = normalize_image(path, template_key=key, cache_subdir=cache_subdir)
        normalized.append(out)
    print(f"  -> {len(normalized)} normalized:")
    for out in normalized:
        print(f"    {out}")
