"""
Standalone module: OCR configurable ROI fields (date, school, teacher, etc.) from normalized images.

Images are assumed odd pages, side a; schema is managed via config. Same OCR procedure as id_recognize
(engine from config, crop to ROI, no filtering). Excludes id (use id_recognize for that).

Ref: prompts/module — single-call usage after import.

Usage:
    from metadata_getter import get_metadata, get_metadata_batch

    meta = get_metadata("output/cache/normalized/maury_1/page_0001_normalized.png")
    # -> {"date": "...", "school": "...", "teacher": "..."}

    list_meta = get_metadata_batch([path1, path2])
    # -> [{"date": "...", ...}, {"date": "...", ...}]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import overload

from config_legacy import METADATA_GETTER

try:
    import cv2
except ImportError:
    cv2 = None


def _schema_path(schema_key: str | None = None) -> Path:
    """Resolve schema key to JSON path."""
    cfg = METADATA_GETTER
    key = schema_key or cfg.get("schema_key", "schema_sidea")
    schema_dir = Path(cfg.get("schema_dir", Path(__file__).resolve().parent.parent.parent / "data" / "roi_schemas"))
    return schema_dir.resolve() / f"{key}.json"


def _load_schema(schema_path: Path) -> dict:
    path = Path(schema_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    return json.loads(path.read_text())


def _get_roi_by_name(schema_data: dict, field_name: str) -> tuple[int, int, int, int, int, int]:
    """Return (x, y, w, h, ref_w, ref_h) for the named ROI."""
    ref_w = schema_data.get("image_width", 0)
    ref_h = schema_data.get("image_height", 0)
    for roi in schema_data.get("rois", []):
        if roi.get("name") == field_name:
            return (
                int(roi["x"]),
                int(roi["y"]),
                int(roi["w"]),
                int(roi["h"]),
                ref_w,
                ref_h,
            )
    raise KeyError(f"No ROI named {field_name!r} in schema")


def _crop_to_roi(
    image_path: Path,
    x: int,
    y: int,
    w: int,
    h: int,
    ref_w: int,
    ref_h: int,
) -> "cv2.Mat":
    """Load image, scale ROI if image size != reference, return cropped region."""
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
    """Convert crop to B&W for OCR (Otsu threshold)."""
    if cv2 is None:
        raise ImportError("OpenCV is required. Install: pip install opencv-python")
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _crop_cache_path(image_path: Path, cache_root: Path, field_name: str) -> Path:
    """Unique cache path for this image's field ROI crop."""
    stem = image_path.stem
    parent = image_path.parent.name
    if parent and parent not in (".", ".."):
        stem = parent + "_" + stem
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{stem}_{field_name}_crop.png"


def _run_ocr(crop_path: Path, engine: str) -> str:
    """Run OCR on crop file; return raw text (stripped, no filtering)."""
    from id_recognize import get_ocr
    return (get_ocr(engine)(crop_path) or "").strip()


def get_metadata(
    image_path: str | Path,
    *,
    schema_path: Path | str | None = None,
    schema_key: str | None = None,
    field_names: list[str] | None = None,
) -> dict[str, str]:
    """
    OCR configured ROI fields from one normalized image (odd page, side a). No filtering.

    Args:
        image_path: Path to normalized page image.
        schema_path: Override path to schema JSON (optional).
        schema_key: Override schema key (optional).
        field_names: Override list of ROI names to read (default: from config).

    Returns:
        Dict mapping each field name to OCR text (raw, stripped).
    """
    cfg = METADATA_GETTER
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if schema_path is not None:
        sp = Path(schema_path).resolve()
    else:
        sp = _schema_path(schema_key)
    schema_data = _load_schema(sp)
    fields = field_names if field_names is not None else list(cfg.get("field_names", ["date", "school", "teacher"]))
    engine = cfg.get("ocr_engine", "ocr_engine").lower()
    cache_root = Path(cfg.get("crop_cache_root", Path(__file__).resolve().parent.parent / "output" / "cache" / "metadata_crops")).resolve()

    out: dict[str, str] = {}
    for name in fields:
        try:
            x, y, w, h, ref_w, ref_h = _get_roi_by_name(schema_data, name)
        except KeyError:
            out[name] = ""
            continue
        crop = _crop_to_roi(image_path, x, y, w, h, ref_w, ref_h)
        bw = _crop_to_bw(crop)
        crop_path = _crop_cache_path(image_path, cache_root, name)
        cv2.imwrite(str(crop_path), bw)
        out[name] = _run_ocr(crop_path, engine)
    return out


def get_metadata_batch(
    image_paths: list[str | Path],
    *,
    schema_path: Path | str | None = None,
    schema_key: str | None = None,
    field_names: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    OCR configured ROI fields for each image. Same order as input. No filtering.

    Args:
        image_paths: List of normalized page image paths.
        schema_path: Override path to schema JSON (optional).
        schema_key: Override schema key (optional).
        field_names: Override list of ROI names (default: from config).

    Returns:
        List of dicts, one per image: each dict maps field name -> OCR text.
    """
    return [
        get_metadata(p, schema_path=schema_path, schema_key=schema_key, field_names=field_names)
        for p in image_paths
    ]


@overload
def get_metadata_one_or_batch(
    image_path: str | Path,
    *,
    schema_path: Path | str | None = None,
    schema_key: str | None = None,
    field_names: list[str] | None = None,
) -> dict[str, str]: ...
@overload
def get_metadata_one_or_batch(
    image_paths: list[str | Path],
    *,
    schema_path: Path | str | None = None,
    schema_key: str | None = None,
    field_names: list[str] | None = None,
) -> list[dict[str, str]]: ...
def get_metadata_one_or_batch(
    image_path_or_paths: str | Path | list[str | Path],
    *,
    schema_path: Path | str | None = None,
    schema_key: str | None = None,
    field_names: list[str] | None = None,
) -> dict[str, str] | list[dict[str, str]]:
    """
    Single entry point: one image -> dict; list of images -> list of dicts (like mc_detect).
    """
    if isinstance(image_path_or_paths, list):
        return get_metadata_batch(
            image_path_or_paths,
            schema_path=schema_path,
            schema_key=schema_key,
            field_names=field_names,
        )
    return get_metadata(
        image_path_or_paths,
        schema_path=schema_path,
        schema_key=schema_key,
        field_names=field_names,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python metadata_getter.py <image_path> [image_path ...]")
        sys.exit(1)
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) == 1:
        for k, v in get_metadata(paths[0]).items():
            print(f"  {k}: {repr(v)}")
    else:
        for p, meta in zip(paths, get_metadata_batch(paths)):
            print(f"{p}:")
            for k, v in meta.items():
                print(f"  {k}: {repr(v)}")
    print("Done.")
