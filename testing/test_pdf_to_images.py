"""Test pdf_to_images with a PDF from data/."""

import sys
from pathlib import Path

# Run from project root so py and data paths work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from pdf_to_images import pdf_to_images

PDF_PATH = ROOT / "data" / "maury 1.pdf"

if __name__ == "__main__":
    print(f"Testing pdf_to_images with: {PDF_PATH}")
    paths = pdf_to_images(PDF_PATH)
    print(f"Cached {len(paths)} images:")
    for p in paths:
        print(f"  {p}")
