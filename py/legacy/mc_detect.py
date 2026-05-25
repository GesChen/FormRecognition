"""
Standalone module: multiple-choice (bubble) filled detection on normalized images.

Uses a schema key, list of ROI names, and page(s). Crops each ROI from the normalized
image and detects filled vs empty via black pixel percentage (total or graduated
outer-cropped region). Threshold per label (a–h) from config.

Ref: prompts/module — single-call usage after import.

Usage:
    from mc_detect import detect_page, detect_batch, detect_roi_on_page

    results = detect_page(normalized_image_path, schema_key="schema_sidea", roi_names=["1a", "1b", "1c", "1d"])
    # -> {"1a": True, "1b": False, ...}

    results_list = detect_batch(pages=[1, 2], schema_key="...", roi_names=["1a", "1b"], get_normalized_path=my_path_fn)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from config_legacy import MULTIPLE_CHOICE

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


# Labels we support for threshold lookup (multiple choice options)
MC_LABELS = ("a", "b", "c", "d", "e", "f", "g", "h")


def _schema_path(schema_key: str) -> Path:
    """Resolve schema key to JSON path (data/roi_schemas/<key>.json)."""
    schema_dir = Path(MULTIPLE_CHOICE.get("schema_dir", Path(__file__).resolve().parent.parent.parent / "data" / "roi_schemas"))
    return schema_dir.resolve() / f"{schema_key}.json"


def _load_schema(schema_path: Path) -> dict:
    """Load schema JSON; return dict with image_width, image_height, rois."""
    path = Path(schema_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path}")
    return json.loads(path.read_text())


def _get_roi_by_name(schema_data: dict, roi_name: str) -> tuple[int, int, int, int, int, int]:
    """Return (x, y, w, h, ref_w, ref_h) for the named ROI."""
    ref_w = schema_data.get("image_width", 0)
    ref_h = schema_data.get("image_height", 0)
    for roi in schema_data.get("rois", []):
        if roi.get("name") == roi_name:
            return (
                int(roi["x"]),
                int(roi["y"]),
                int(roi["w"]),
                int(roi["h"]),
                ref_w,
                ref_h,
            )
    raise KeyError(f"No ROI named {roi_name!r} in schema")


def _label_from_roi_name(roi_name: str) -> str | None:
    """Extract option label from ROI name (e.g. '1a' -> 'a', '2d' -> 'd')."""
    if not roi_name or roi_name not in (roi_name.strip(),):
        return None
    last = roi_name.strip()[-1].lower()
    return last if last in MC_LABELS else None


def _crop_to_roi(
    image: "np.ndarray",
    x: int,
    y: int,
    w: int,
    h: int,
    ref_w: int,
    ref_h: int,
) -> "np.ndarray":
    """Crop image to ROI; scale x,y,w,h if image size != ref. Returns ROI patch (gray or BGR)."""
    H, W = image.shape[:2]
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
    return image[y : y + h, x : x + w].copy()


def _to_gray(crop: "np.ndarray") -> "np.ndarray":
    """Ensure crop is 2D grayscale."""
    if crop.ndim == 3:
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


def _darkness_fraction(crop: "np.ndarray") -> float:
    """
    Grayscale-weighted darkness: mean of (255 - gray) / 255 over all pixels.
    Returns a value in [0, 1]: 0 = all white, 1 = all black. No binary cutoff.
    """
    gray = _to_gray(crop).astype(np.float64)
    if gray.size == 0:
        return 0.0
    darkness = (255.0 - gray) / 255.0
    return float(np.mean(darkness))


def _get_graduated_weight_map(crop: "np.ndarray", falloff_config: dict | None = None) -> "np.ndarray":
    """
    Get the graduated weight map (coefficient map) for a crop.
    Returns weight array matching crop dimensions.
    """
    gray = _to_gray(crop).astype(np.float64)
    h, w = gray.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((h, w), dtype=np.float64)
    
    if falloff_config is None:
        falloff_config = MULTIPLE_CHOICE.get("graduated_falloff", {})
        if isinstance(falloff_config, (int, float)):
            falloff_config = {
                "center_weight": 0.0,
                "edge_weight": 1.0,
                "transition_distance": 0.3,
                "falloff_power": 2.0,
            }
    
    center_w = float(falloff_config.get("center_weight", 0.0))
    edge_w = float(falloff_config.get("edge_weight", 1.0))
    transition_dist = float(falloff_config.get("transition_distance", 0.3))
    falloff_power = float(falloff_config.get("falloff_power", 2.0))
    
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    max_dist = np.sqrt(cx * cx + cy * cy)
    if max_dist <= 0:
        max_dist = 1.0
    
    yy = np.arange(h, dtype=np.float64) - cy
    xx = np.arange(w, dtype=np.float64) - cx
    dist = np.sqrt((xx[np.newaxis, :] ** 2) + (yy[:, np.newaxis] ** 2))
    normalized_dist = dist / max_dist
    
    # Original simple falloff: power curve
    t = np.clip(normalized_dist, 0.0, 1.0)
    t_scaled = np.clip(t / max(transition_dist, 0.01), 0.0, 1.0)
    t_curved = t_scaled ** falloff_power
    weight = center_w + (edge_w - center_w) * t_curved
    
    return weight


def _darkness_fraction_graduated(crop: "np.ndarray", falloff_config: dict | None = None) -> float:
    """
    Grayscale-weighted darkness using graduated coefficient map.
    Process: invert image -> multiply by weight map -> invert -> calculate darkness (same as flat).
    This reduces center contribution and focuses on weighted regions.
    
    Args:
        falloff_config: Dict with center_weight, edge_weight, transition_distance, falloff_power.
                        If None, uses MULTIPLE_CHOICE["graduated_falloff"].
    """
    gray = _to_gray(crop).astype(np.float64)
    if gray.size == 0:
        return 0.0
    
    # Get weight map (coefficient map)
    weight_map = _get_graduated_weight_map(crop, falloff_config)
    
    # Step 1: Invert original image (dark -> light, light -> dark)
    inverted = 255.0 - gray
    
    # Step 2: Multiply by weight map
    multiplied = weight_map * inverted
    
    # Step 3: Invert result back
    final = 255.0 - multiplied
    
    # Step 4: Calculate darkness using same method as flat
    darkness = (255.0 - final) / 255.0
    return float(np.mean(darkness))


def is_roi_filled(
    crop: "np.ndarray",
    label: str | None,
    *,
    use_graduated: bool | None = None,
    threshold_override: float | None = None,
) -> bool:
    """
    Detect whether a single ROI crop is filled (bubble marked).

    Uses grayscale-weighted darkness: mean of (255 - gray)/255 over the ROI.
    Filled when this darkness value >= config threshold for the label (a–h).
    Threshold 0.15 = 15% average darkness (no binary cutoff).

    Args:
        crop: ROI image (BGR or grayscale).
        label: Option label "a".."h" for threshold lookup; None = use default.
        use_graduated: If True, use graduated darkness (invert->multiply->invert); if False, use flat mean.
                       If None, uses MULTIPLE_CHOICE["use_graduated_region"] config value.
        threshold_override: If set, use this darkness threshold (0–1) instead of config.

    Returns:
        True if darkness >= threshold (filled).
    """
    cfg = MULTIPLE_CHOICE
    if threshold_override is not None:
        th = threshold_override
    else:
        by_label = cfg.get("filled_threshold_by_label") or {}
        th = by_label.get((label or "").lower(), 0.15)
    use_grad = use_graduated if use_graduated is not None else cfg.get("use_graduated_region", False)
    if use_grad:
        darkness = _darkness_fraction_graduated(crop)
    else:
        darkness = _darkness_fraction(crop)
    return darkness >= th


def detect_roi_on_page(
    normalized_image_path: str | Path,
    schema_key: str,
    roi_name: str,
    *,
    schema_path: Path | None = None,
    schema_data: dict | None = None,
) -> bool:
    """
    Single-ROI filled detection on one page: load normalized image, crop to ROI, run filled check.

    Args:
        normalized_image_path: Path to normalized page image (e.g. from cache).
        schema_key: Key for schema file (e.g. "schema_sidea").
        roi_name: ROI name in schema (e.g. "1a", "2b").
        schema_path: Override path to schema JSON (optional).
        schema_data: Pre-loaded schema dict (optional; avoids re-read).

    Returns:
        True if the ROI is detected as filled.
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and numpy required. Install: pip install opencv-python numpy")
    path = Path(normalized_image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if schema_data is None:
        sp = Path(schema_path) if schema_path is not None else _schema_path(schema_key)
        schema_data = _load_schema(sp)
    x, y, w, h, ref_w, ref_h = _get_roi_by_name(schema_data, roi_name)
    crop = _crop_to_roi(img, x, y, w, h, ref_w, ref_h)
    label = _label_from_roi_name(roi_name)
    use_grad = MULTIPLE_CHOICE.get("use_graduated_region", False)
    return is_roi_filled(crop, label, use_graduated=use_grad)


def detect_page(
    normalized_image_path: str | Path,
    schema_key: str,
    roi_names: list[str],
    *,
    schema_path: Path | None = None,
) -> dict[str, bool]:
    """
    Run filled detection for multiple ROIs on one normalized page image.

    Args:
        normalized_image_path: Path to normalized page image.
        schema_key: Schema key (e.g. "schema_sidea").
        roi_names: List of ROI names (e.g. ["1a", "1b", "1c", "1d"]).
        schema_path: Override path to schema JSON (optional).

    Returns:
        Dict mapping each roi_name to filled (True/False).
    """
    filled, _ = detect_page_with_darkness(
        normalized_image_path, schema_key, roi_names, schema_path=schema_path
    )
    return filled


def detect_page_with_darkness(
    normalized_image_path: str | Path,
    schema_key: str,
    roi_names: list[str],
    *,
    schema_path: Path | None = None,
) -> tuple[dict[str, bool], dict[str, dict]]:
    """
    Run filled detection and return darkness values per ROI for debug output.

    Returns:
        (filled_dict, darkness_dict) where filled_dict is roi_name -> bool,
        and darkness_dict is roi_name -> {"darkness_flat": float, "darkness_graduated": float, "filled": bool}.
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and numpy required. Install: pip install opencv-python numpy")
    path = Path(normalized_image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    sp = Path(schema_path) if schema_path is not None else _schema_path(schema_key)
    schema_data = _load_schema(sp)
    ref_w = schema_data.get("image_width", 0)
    ref_h = schema_data.get("image_height", 0)

    use_grad = MULTIPLE_CHOICE.get("use_graduated_region", False)
    out: dict[str, bool] = {}
    darkness_out: dict[str, dict] = {}
    for name in roi_names:
        try:
            x, y, w, h, _, _ = _get_roi_by_name(schema_data, name)
        except KeyError:
            out[name] = False
            darkness_out[name] = {"darkness_flat": 0.0, "darkness_graduated": 0.0, "filled": False}
            continue
        crop = _crop_to_roi(img, x, y, w, h, ref_w, ref_h)
        label = _label_from_roi_name(name)
        flat = _darkness_fraction(crop)
        grad = _darkness_fraction_graduated(crop)
        filled = is_roi_filled(crop, label, use_graduated=use_grad)
        out[name] = filled
        darkness_out[name] = {
            "darkness_flat": round(flat, 4),
            "darkness_graduated": round(grad, 4),
            "filled": filled,
        }
    return out, darkness_out


def detect_page_with_crops(
    normalized_image_path: str | Path,
    schema_key: str,
    roi_names: list[str],
    *,
    schema_path: Path | None = None,
) -> tuple[dict[str, bool], list[tuple[str, "np.ndarray", bool]]]:
    """
    Like detect_page but also return crops for preview. Returns (results_dict, list of (roi_name, crop, filled)).
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and numpy required. Install: pip install opencv-python numpy")
    path = Path(normalized_image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    sp = Path(schema_path) if schema_path is not None else _schema_path(schema_key)
    schema_data = _load_schema(sp)
    ref_w = schema_data.get("image_width", 0)
    ref_h = schema_data.get("image_height", 0)

    use_grad = MULTIPLE_CHOICE.get("use_graduated_region", False)
    results: dict[str, bool] = {}
    crops: list[tuple[str, "np.ndarray", bool]] = []
    for name in roi_names:
        try:
            x, y, w, h, _, _ = _get_roi_by_name(schema_data, name)
        except KeyError:
            results[name] = False
            continue
        crop = _crop_to_roi(img, x, y, w, h, ref_w, ref_h)
        label = _label_from_roi_name(name)
        filled = is_roi_filled(crop, label, use_graduated=use_grad)
        results[name] = filled
        crops.append((name, crop, filled))
    return results, crops


def detect_batch_with_crops(
    pages: list[int],
    schema_key: str,
    roi_names: list[str],
    get_normalized_path: Callable[[int], Path | str],
    *,
    schema_path: Path | None = None,
) -> tuple[list[dict[str, bool]], list[tuple[int, str, "np.ndarray", bool]]]:
    """
    Like detect_batch but also return crops for preview. Returns
    (list of results per page, list of (page, roi_name, crop, filled)).
    """
    results_list: list[dict[str, bool]] = []
    all_crops: list[tuple[int, str, "np.ndarray", bool]] = []
    for page in pages:
        path = get_normalized_path(page)
        try:
            res, crops = detect_page_with_crops(path, schema_key, roi_names, schema_path=schema_path)
        except FileNotFoundError:
            res = {name: False for name in roi_names}
            crops = []
        results_list.append(res)
        for name, crop, filled in crops:
            all_crops.append((page, name, crop, filled))
    return results_list, all_crops


def detect(
    schema_key: str,
    roi_names: list[str],
    page_or_pages: int | list[int],
    get_normalized_path: Callable[[int], Path | str],
    *,
    schema_path: Path | None = None,
) -> dict[str, bool] | list[dict[str, bool]]:
    """
    Single entry point: run filled detection for the given ROIs on one page or a batch of pages.

    Args:
        schema_key: Schema key (e.g. "schema_sidea").
        roi_names: List of ROI names (e.g. ["1a", "1b", "1c", "1d"]).
        page_or_pages: One page number (int) or list of page numbers.
        get_normalized_path: Callable(page: int) -> path to normalized image for that page.
        schema_path: Override path to schema JSON (optional).

    Returns:
        If page_or_pages is a single int: dict mapping roi_name -> filled.
        If page_or_pages is a list: list of such dicts, one per page (same order).
    """
    if isinstance(page_or_pages, int):
        return detect_page(
            get_normalized_path(page_or_pages),
            schema_key,
            roi_names,
            schema_path=schema_path,
        )
    return detect_batch(
        list(page_or_pages),
        schema_key,
        roi_names,
        get_normalized_path,
        schema_path=schema_path,
    )


def detect_batch(
    pages: list[int],
    schema_key: str,
    roi_names: list[str],
    get_normalized_path: Callable[[int], Path | str],
    *,
    schema_path: Path | None = None,
) -> list[dict[str, bool]]:
    """
    Run filled detection for multiple ROIs on a batch of pages.

    Args:
        pages: List of 1-based page numbers.
        schema_key: Schema key (e.g. "schema_sidea").
        roi_names: List of ROI names (e.g. ["1a", "1b", "1c", "1d"]).
        get_normalized_path: Callable(page: int) -> path to normalized image for that page.
        schema_path: Override path to schema JSON (optional).

    Returns:
        List of dicts, one per page (same order as pages): each dict maps roi_name -> filled.
    """
    results: list[dict[str, bool]] = []
    for page in pages:
        path = get_normalized_path(page)
        try:
            results.append(detect_page(path, schema_key, roi_names, schema_path=schema_path))
        except FileNotFoundError:
            results.append({name: False for name in roi_names})
    return results


if __name__ == "__main__":
    import sys
    from config import IMAGE_NORMALIZE

    # Example: single-call usage (ref: prompts/module)
    # python mc_detect.py <schema_key> <page> [roi_name ...]
    if len(sys.argv) < 3:
        print("Usage: python mc_detect.py <schema_key> <page> [roi_name ...]")
        print("Example: python mc_detect.py schema_sidea 1 1a 1b 1c 1d")
        sys.exit(1)
    schema_key = sys.argv[1]
    page = int(sys.argv[2])
    roi_names = sys.argv[3:] if len(sys.argv) > 3 else ["1a", "1b", "1c", "1d"]
    cache = Path(IMAGE_NORMALIZE["cache_root"]).resolve()
    suffix = "_bin" if IMAGE_NORMALIZE.get("binarize", True) else "_normalized"
    norm_path = cache / "maury_1" / f"page_{page:04d}{suffix}.png"
    res = detect_page(norm_path, schema_key, roi_names)
    for name, filled in res.items():
        print(f"  {name}: {'filled' if filled else 'empty'}")
    print("Done.")
