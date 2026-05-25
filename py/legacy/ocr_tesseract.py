"""
Tesseract OCR: image path in, text out.

Usage:

    from ocr_tesseract import ocr
    text = ocr("path/to/crop.png")
"""

from __future__ import annotations

from pathlib import Path

try:
    import cv2
    import pytesseract
except ImportError:
    cv2 = None
    pytesseract = None


def ocr(image_path: str | Path) -> str:
    """
    Run Tesseract OCR on an image file.

    Args:
        image_path: Path to the image file (e.g. cached ID ROI crop).

    Returns:
        Recognized text (stripped).
    """
    if pytesseract is None or cv2 is None:
        raise ImportError(
            "pytesseract and opencv-python are required. Install: pip install pytesseract opencv-python "
            "(and tesseract-ocr system package)"
        )
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return pytesseract.image_to_string(image).strip()
