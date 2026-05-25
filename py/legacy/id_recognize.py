"""
Standalone module: OCR the "id" ROI from a schema on one or more images.

Uses config.ID_RECOGNITION: schema_path, ocr_engine, crop_cache_root.
Cropped ROI is saved to cache and OCR is given the file path. Engines have ocr(image_path) -> str.
"""

from pathlib import Path
import json
import re

from config_legacy import ID_RECOGNITION

try:
    import cv2
except ImportError:
    cv2 = None


def _load_id_roi(schema_path: Path) -> tuple[int, int, int, int, int, int]:
    """Load schema JSON and return (x, y, w, h, ref_w, ref_h) for the ROI named 'id'."""
    path = Path(schema_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    data = json.loads(path.read_text())
    ref_w = data.get("image_width", 0)
    ref_h = data.get("image_height", 0)
    for roi in data.get("rois", []):
        if roi.get("name") == "id":
            return (
                int(roi["x"]),
                int(roi["y"]),
                int(roi["w"]),
                int(roi["h"]),
                ref_w,
                ref_h,
            )
    raise KeyError(f"No ROI named 'id' in schema: {path}")


def _crop_to_roi(
    image_path: Path,
    x: int,
    y: int,
    w: int,
    h: int,
    ref_w: int,
    ref_h: int,
) -> "cv2.Mat":
    """Load image, scale ROI if image size != reference, return cropped region (BGR or gray)."""
    if cv2 is None:
        raise ImportError("OpenCV is required. Install: pip install opencv-python")
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    H, W = img.shape[:2]
    if ref_w > 0 and ref_h > 0 and (W != ref_w or H != ref_h):
        scale_x = W / ref_w
        scale_y = H / ref_h
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return img[y : y + h, x : x + w]


def _crop_to_bw(crop: "cv2.Mat") -> "cv2.Mat":
    """Convert crop to strict B&W: grayscale then binary threshold (0 or 255 only)."""
    if cv2 is None:
        raise ImportError("OpenCV is required. Install: pip install opencv-python")
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _crop_cache_path(image_path: Path, cache_root: Path) -> Path:
    """Unique cache path for this image's id ROI crop (same stem as normalized naming)."""
    stem = image_path.stem
    parent = image_path.parent.name
    if parent and parent not in (".", ".."):
        stem = parent + "_" + stem
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{stem}_id_crop.png"


# Registry: engine name -> module name (module must define ocr(image_path) -> str).
OCR_ENGINES = {
    "tesseract": "ocr_tesseract",
    "easyocr": "ocr_easyocr",
    "ocr_engine": "ocr_engine",
}


def get_ocr(engine: str | None = None):
    """
    Return the ocr(image) -> str function for the given engine.
    For use by other modules that need text recognition.
    Engine names: tesseract, easyocr, ocr_engine.

    Args:
        engine: Engine name (default: from config.ID_RECOGNITION["ocr_engine"]).

    Returns:
        Callable ocr(image_path) -> str, where image_path is a path to an image file.
    """
    if engine is None:
        engine = ID_RECOGNITION.get("ocr_engine", "ocr_engine")
    engine = engine.lower()
    if engine not in OCR_ENGINES:
        raise ValueError(
            f"Unknown ocr_engine: {engine}. Use one of: {list(OCR_ENGINES.keys())}."
        )
    mod = __import__(OCR_ENGINES[engine], fromlist=["ocr"])
    return mod.ocr


def _run_ocr(crop_path: Path, engine: str) -> str:
    """Run OCR engine on image file at crop_path; returns recognized text. Engine from config."""
    return get_ocr(engine)(crop_path)


def _extract_id_value(ocr_text: str) -> str:
    """
    Extract ID token from OCR output: everything after the colon that looks like
    the ID (alphanumeric). Number sequences follow a colon (e.g. "ID:9010526A" or
    "ID:901I526A" -> full token so replacements can fix I->1 across the whole string).
    """
    ocr_text = ocr_text.replace(" ", "")
    
    if not ocr_text or not ocr_text.strip():
        return (ocr_text or "").strip()
    # After a colon: optional space, then capture all alphanumeric (keeps going so
    # e.g. 901I526A is captured and I can be replaced with 1 in _normalize_id_value)
    m = re.search(r":\s*([0-9A-Za-z]+)", ocr_text)
    if m:
        return m.group(1).strip()
    return ocr_text.strip()


def _normalize_id_value(
    raw: str,
    length: int,
    replacements: dict[str, str],
) -> str:
    """
    Accumulate the ID string to a fixed length, applying replacement dict for common OCR
    mismatches (case-insensitive). Truncate if longer, pad with "?" if shorter.
    All spaces are removed from the string before processing.
    """
    if not raw:
        return "?" * length if length > 0 else ""
    raw = raw.replace(" ", "")
    # Replacements: keys are uppercase in config; look up by char.upper()
    out = []
    for c in raw:
        r = replacements.get(c.upper(), c)
        out.append(r)
    result = "".join(out)
    if len(result) > length:
        return result[:length]
    if length > len(result):
        return result + "?" * (length - len(result))
    return result


def id_recognize(
    image_path: str | Path,
    schema_path: str | Path | None = None,
) -> str:
    """
    Run OCR on the "id" ROI of a single image.

    Args:
        image_path: Path to the image (e.g. normalized page).
        schema_path: Path to ROI schema JSON (default: from config).

    Returns:
        If id_trim_and_fix: extracted + normalized ID (e.g. "9010526A"). Else: full OCR output (stripped).
    """
    cfg = ID_RECOGNITION
    if schema_path is None:
        schema_path = cfg["schema_path"]
    schema_path = Path(schema_path).resolve()
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    crop_cache = Path(cfg.get("crop_cache_root", Path(cfg["schema_path"]).parent / "cache" / "id_crops")).resolve()
    x, y, w, h, ref_w, ref_h = _load_id_roi(schema_path)
    crop = _crop_to_roi(image_path, x, y, w, h, ref_w, ref_h)
    bw = _crop_to_bw(crop)
    crop_path = _crop_cache_path(image_path, crop_cache)
    cv2.imwrite(str(crop_path), bw)
    raw = _run_ocr(crop_path, cfg.get("ocr_engine", "ocr_engine"))
    if not cfg.get("id_trim_and_fix", True):
        return (raw or "").strip()
    extracted = _extract_id_value(raw)
    length = cfg.get("id_char_length")
    replacements = cfg.get("id_char_replacements") or {}
    if length is not None:
        return _normalize_id_value(extracted, length, replacements)
    return extracted


def id_recognize_batch(
    image_paths: list[str | Path],
    schema_path: str | Path | None = None,
) -> list[str]:
    """
    Run OCR on the "id" ROI for each image (same order as input).

    Args:
        image_paths: List of image paths.
        schema_path: Path to ROI schema JSON (default: from config).

    Returns:
        List of strings: trimmed/fixed ID per image if id_trim_and_fix, else full OCR output per image.
    """
    cfg = ID_RECOGNITION
    if schema_path is None:
        schema_path = cfg["schema_path"]
    schema_path = Path(schema_path).resolve()
    crop_cache = Path(cfg.get("crop_cache_root", Path(cfg["schema_path"]).parent / "cache" / "id_crops")).resolve()
    x, y, w, h, ref_w, ref_h = _load_id_roi(schema_path)
    engine = cfg.get("ocr_engine", "ocr_engine").lower()

    results: list[str] = []
    for p in image_paths:
        p = Path(p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        crop = _crop_to_roi(p, x, y, w, h, ref_w, ref_h)
        bw = _crop_to_bw(crop)
        crop_path = _crop_cache_path(p, crop_cache)
        cv2.imwrite(str(crop_path), bw)
        raw = _run_ocr(crop_path, engine)
        if not cfg.get("id_trim_and_fix", True):
            results.append((raw or "").strip())
            continue
        extracted = _extract_id_value(raw)
        length = cfg.get("id_char_length")
        replacements = cfg.get("id_char_replacements") or {}
        if length is not None:
            results.append(_normalize_id_value(extracted, length, replacements))
        else:
            results.append(extracted)
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python id_recognize.py <image_path> [image_path ...]")
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) == 1:
        print(id_recognize(paths[0]))
    else:
        for p, t in zip(paths, id_recognize_batch(paths)):
            print(f"{p}: {t}")
