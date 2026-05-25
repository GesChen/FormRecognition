"""
EasyOCR: image path in, text out.

Usage:

    from ocr_easyocr import ocr
    text = ocr("path/to/crop.png")
"""

from __future__ import annotations

from pathlib import Path

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except ImportError:
            raise ImportError(
                "EasyOCR is required. Install: pip install easyocr"
            )
    return _reader


def ocr(image_path: str | Path) -> str:
    """
    Run EasyOCR on an image file.

    Args:
        image_path: Path to the image file (e.g. cached ID ROI crop).

    Returns:
        Recognized text (stripped), words joined by space.
    """
    import cv2
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    reader = _get_reader()
    results = reader.readtext(image)
    return " ".join(t[1] for t in results).strip()
