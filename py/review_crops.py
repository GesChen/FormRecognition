"""
PNG crops for human review UI: ID top band (same geometry as id_form_llm) and ROI rectangles from schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2  # type: ignore


def safe_resolve_under_project(project_root: Path, rel: str) -> Path | None:
    """
    Return absolute path if *rel* points at a file under *project_root*.

    Accepts project-relative paths (``output/cache/...``) or absolute paths that
    still lie under *project_root* (so JSON from different machines can resolve).
    """
    rel = (rel or "").strip().replace("\\", "/")
    if not rel:
        return None
    parts = rel.split("/")
    if ".." in parts:
        return None
    root = project_root.resolve()
    p = Path(rel)
    if p.is_absolute():
        full = p.resolve()
    else:
        full = (root / rel).resolve()
    try:
        full.relative_to(root)
    except ValueError:
        return None
    return full if full.is_file() else None


def imread_grayscale(path: Path):
    """Read image as 2D grayscale; more tolerant than IMREAD_GRAYSCALE alone."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None


def _top_rows_from_percent(image_height: int, percent: float) -> int:
    if image_height <= 0:
        return 0
    p = max(0.01, min(100.0, float(percent)))
    return max(1, min(image_height, int(round(image_height * p / 100.0))))


def png_bytes_id_top_crop(page_image: Path, crop_top_percent: float) -> bytes:
    """Grayscale PNG bytes for the top *crop_top_percent* of *page_image* (ID region)."""
    img = imread_grayscale(page_image)
    if img is None:
        return b""
    h = img.shape[0]
    rows = _top_rows_from_percent(h, crop_top_percent)
    crop = img[0:rows, :]
    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else b""


def _find_roi_dict(raw_schema: dict[str, Any], roi_name: str) -> dict[str, Any] | None:
    for roi in raw_schema.get("rois") or []:
        if str(roi.get("name", "")).strip() == str(roi_name).strip():
            return roi
    return None


def png_bytes_full_page(page_image: Path) -> bytes:
    """Grayscale PNG bytes for the full *page_image* (for human review context)."""
    img = imread_grayscale(page_image)
    if img is None:
        return b""
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def roi_rect_xyxy_in_page_pixels(
    page_image: Path,
    raw_schema: dict[str, Any] | None,
    roi_name: str,
    kind: str,
    crop_top_percent: float,
) -> tuple[int, int, int, int, int, int] | None:
    """
    ROI rectangle on the full page in pixel coordinates.

    Returns (x1, y1, x2, y2, iw, ih) with x2/y2 as exclusive upper bounds (slice-like),
    matching the crop used in png_bytes_id_top_crop / png_bytes_roi_crop.
    """
    kind = (kind or "").lower()
    img = imread_grayscale(page_image)
    if img is None:
        return None
    ih, iw = int(img.shape[0]), int(img.shape[1])
    if kind == "id":
        rows = _top_rows_from_percent(ih, crop_top_percent)
        return (0, 0, iw, rows, iw, ih)
    if kind not in ("text", "mcq") or not raw_schema:
        return None
    roi = _find_roi_dict(raw_schema, roi_name)
    if roi is None:
        return None
    x = float(roi.get("x", 0))
    y = float(roi.get("y", 0))
    w = float(roi.get("w", 0))
    h = float(roi.get("h", 0))
    ref_w = int(raw_schema.get("image_width") or 0)
    ref_h = int(raw_schema.get("image_height") or 0)
    if ref_w > 0 and ref_h > 0 and (ref_w, ref_h) != (iw, ih):
        sx, sy = iw / ref_w, ih / ref_h
    else:
        sx = sy = 1.0
    x1 = max(int(round(x * sx)), 0)
    y1 = max(int(round(y * sy)), 0)
    x2 = min(int(round((x + w) * sx)), iw)
    y2 = min(int(round((y + h) * sy)), ih)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2, iw, ih)


def png_bytes_roi_crop(page_image: Path, raw_schema: dict[str, Any], roi_name: str) -> bytes:
    """Grayscale PNG bytes for ROI *roi_name* using schema coordinates (with scale)."""
    roi = _find_roi_dict(raw_schema, roi_name)
    if roi is None:
        return b""
    x = float(roi.get("x", 0))
    y = float(roi.get("y", 0))
    w = float(roi.get("w", 0))
    h = float(roi.get("h", 0))
    img = imread_grayscale(page_image)
    if img is None:
        return b""
    ih, iw = img.shape[:2]
    ref_w = int(raw_schema.get("image_width") or 0)
    ref_h = int(raw_schema.get("image_height") or 0)
    if ref_w > 0 and ref_h > 0 and (ref_w, ref_h) != (iw, ih):
        sx, sy = iw / ref_w, ih / ref_h
    else:
        sx = sy = 1.0
    x1 = max(int(round(x * sx)), 0)
    y1 = max(int(round(y * sy)), 0)
    x2 = min(int(round((x + w) * sx)), iw)
    y2 = min(int(round((y + h) * sy)), ih)
    if x2 <= x1 or y2 <= y1:
        return b""
    crop = img[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".png", crop)
    return buf.tobytes() if ok else b""


def load_schema_raw(project_root: Path, form_type: str | None, side: str) -> dict[str, Any] | None:
    """Load raw ROI schema JSON for (form_type, side)."""
    from config import PDF_RECOGNITION

    schema_dir = Path(PDF_RECOGNITION.get("schema_dir", project_root / "data" / "roi_schemas")).resolve()
    s = (form_type or "").strip()
    if s:
        path = schema_dir / f"{s}_{side}.json"
    else:
        fallback = (
            PDF_RECOGNITION.get("schema_key_sidea", "schema_sidea")
            if side == "a"
            else PDF_RECOGNITION.get("schema_key_sideb", "schema_sideb")
        )
        path = schema_dir / f"{fallback}.json"
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))
