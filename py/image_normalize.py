"""
Standalone module: normalize document images for pure 0/1 use cases.

Pipeline (all toggles via config.IMAGE_NORMALIZE):
  - Optional feature-based registration to a template (ORB keypoints, RANSAC homography).
    Templates can be selected by:
      - explicit template key (e.g. "6pre_a"), or
      - form_type + page_side ("a"/"b"), combined as "<form_type>_<side>".
    Resolution checks template_registration_templates dict first, then
    template_registration_templates_dir / "<key>.png".
  - Optional perspective correction (document quad); optional deskew (minAreaRect).
  - Denoise + CLAHE, contrast stretch, then binarization by default (adaptive threshold).

Usage:

    from image_normalize import normalize_image, normalize_images

    out_path = normalize_image("output/cache/maury_1/page_0001.png")
    out_paths = normalize_images(["path1.png", "path2.png"])
    out_path = normalize_image(path, form_type="6pre", page_side="a", binarize=False)
"""

from pathlib import Path
import hashlib
import json
import re

import numpy as np

from config import IMAGE_NORMALIZE

try:
    import cv2
except ImportError:
    cv2 = None


def _sanitize_stem(name: str) -> str:
    """Safe filename stem from path."""
    stem = Path(name).stem
    return re.sub(r"[^\w\-.]", "_", stem).strip("_") or "image"


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stable content hash for cache validation."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _norm_meta_path(out_path: Path) -> Path:
    """Sidecar metadata path for one normalized output."""
    return out_path.with_suffix(out_path.suffix + ".meta.json")


def _load_norm_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_norm_meta(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        # Metadata persistence is best-effort only.
        pass


def _to_gray(img: np.ndarray) -> np.ndarray:
    """Ensure B/W: convert to single channel (all input is always black and white)."""
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.uint8)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Deskew using min-area rect on text/non-background pixels. Input: grayscale."""
    gray = img.astype(np.uint8) if img.ndim == 2 else _to_gray(img)
    # Invert so text is white (minAreaRect expects foreground)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size < 100:
        return img
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _find_doc_quad(gray: np.ndarray) -> np.ndarray | None:
    """Find largest quadrilateral contour (document outline). Returns 4x2 points or None."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(
        edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    area_thresh = gray.size * 0.05
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_thresh:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and area > best_area:
            best_area = area
            best = approx.reshape(4, 2)
    return best


def _register_to_template(gray: np.ndarray, template_path: Path) -> np.ndarray:
    """
    Feature-based registration: align image to template using ORB keypoints,
    match, RANSAC homography, warp. Returns warped image or original if template
    missing / too few matches. Best when scans vary but printed background is stable.
    """
    template_path = Path(template_path).resolve()
    if not template_path.exists():
        return gray
    tpl = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        return gray
    cfg = IMAGE_NORMALIZE if isinstance(IMAGE_NORMALIZE, dict) else {}
    orb_nfeatures = int(cfg.get("template_registration_orb_nfeatures", 5000) or 5000)
    ratio_test = float(cfg.get("template_registration_ratio_test", 0.75) or 0.75)
    min_good_matches = int(cfg.get("template_registration_min_good_matches", 10) or 10)
    ransac_reproj_threshold = float(
        cfg.get("template_registration_ransac_reproj_threshold", 5.0) or 5.0
    )
    min_inliers = int(cfg.get("template_registration_min_inliers", 8) or 8)

    # ORB (no contrib; works on binary/grayscale)
    orb = cv2.ORB_create(nfeatures=max(100, orb_nfeatures))
    kp1, desc1 = orb.detectAndCompute(gray, None)
    kp2, desc2 = orb.detectAndCompute(tpl, None)
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        return gray
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_test * n.distance:
            good.append(m)
    if len(good) < max(4, min_good_matches):
        return gray
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        max(0.5, ransac_reproj_threshold),
    )
    if H is None:
        return gray
    if mask is not None:
        inliers = int(mask.ravel().sum())
        if inliers < max(4, min_inliers):
            return gray
    h, w = tpl.shape[:2]
    return cv2.warpPerspective(gray, H, (w, h), flags=cv2.INTER_LINEAR)


def _apply_geometric_pipeline(
    img: np.ndarray,
    *,
    template_key: str | None,
    form_type: str | None,
    page_side: str | None,
) -> np.ndarray:
    """
    Geometric-only stage:
      - optional template registration (homography)
      - optional perspective correction
      - optional deskew
    """
    out = _to_gray(img)
    if IMAGE_NORMALIZE.get("template_registration", False):
        tpl_path = _get_template_path(
            template_key=template_key,
            form_type=form_type,
            page_side=page_side,
        )
        if tpl_path is not None:
            out = _register_to_template(out, tpl_path)
    out = _perspective_correct(
        out, enabled=IMAGE_NORMALIZE.get("perspective_correct", False)
    )
    if IMAGE_NORMALIZE.get("deskew", False):
        out = _deskew(out)
    return out


def _apply_postprocess_pipeline(
    gray: np.ndarray,
    *,
    binarize: bool,
    binarize_block_size: int,
    adaptive_c: int,
) -> np.ndarray:
    """
    Postprocess stage:
      - denoise + clahe
      - contrast pass
      - optional binarization
    """
    out = _to_gray(gray)
    out = _denoise_and_clahe(out)
    out = _contrast_pass(out)
    if binarize:
        out = _binarize_per_roi(out, binarize_block_size, adaptive_c)
    return out


def _perspective_correct(img: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Apply perspective correction if enabled and a document quad is detected (e.g. photographed). Input: grayscale."""
    if not enabled:
        return img
    gray = img if img.ndim == 2 else _to_gray(img)
    quad = _find_doc_quad(gray)
    if quad is None:
        return img
    # Order points: top-left, top-right, bottom-right, bottom-left
    s = quad.sum(axis=1)
    tl = quad[np.argmin(s)]
    br = quad[np.argmax(s)]
    diff = np.diff(quad, axis=1)
    tr = quad[np.argmin(diff)]
    bl = quad[np.argmax(diff)]
    src = np.float32([tl, tr, br, bl])
    w1 = np.linalg.norm(tr - tl)
    w2 = np.linalg.norm(br - bl)
    h1 = np.linalg.norm(bl - tl)
    h2 = np.linalg.norm(br - tr)
    w = max(int(w1), int(w2))
    h = max(int(h1), int(h2))
    # Avoid 90° flip: if quad would make portrait→landscape or vice vs, skip perspective
    # (e.g. largest quad was an inner rotated feature, not the page border)
    img_h, img_w = gray.shape[:2]
    if (w > h) != (img_w > img_h):
        return img
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)


def _contrast_pass(gray: np.ndarray) -> np.ndarray:
    """Linear stretch (percentile → 0–255) then power curve; grays move toward black or white."""
    cfg = IMAGE_NORMALIZE
    low_pct = cfg["contrast_low_percentile"]
    high_pct = cfg["contrast_high_percentile"]
    power = cfg["contrast_power"]
    lo, hi = np.percentile(gray, [low_pct, high_pct])
    if hi <= lo:
        stretched = gray.astype(np.float32)
    else:
        stretched = np.clip((gray.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    # Power curve: >1 pushes grays toward black, <1 toward white
    if power != 1.0:
        stretched = (stretched / 255.0) ** power * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _denoise_and_clahe(gray: np.ndarray) -> np.ndarray:
    """Denoise then CLAHE. Input: grayscale only (all input is B/W)."""
    cfg = IMAGE_NORMALIZE
    h = cfg["denoise_h"]
    tw = cfg["denoise_template_window_size"]
    sw = cfg["denoise_search_window_size"]
    clip = cfg["clahe_clip_limit"]
    grid = cfg["clahe_tile_grid_size"]
    denoised = cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=tw, searchWindowSize=sw)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(denoised)


def _binarize_per_roi(gray: np.ndarray, block_size: int, c: int) -> np.ndarray:
    """Adaptive (per-ROI) binarization; not a single global threshold. Input: grayscale."""
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
    )


def _cache_stem(img_path: Path) -> str:
    """Unique stem for cache filename: parent + name to avoid collisions (e.g. maury_1_page_0001)."""
    stem = _sanitize_stem(img_path.name)
    parent = img_path.parent.name
    if parent and parent not in (".", ".."):
        stem = _sanitize_stem(parent) + "_" + stem
    return stem


def _normalize_page_side(page_side: str | None) -> str | None:
    if page_side is None:
        return None
    s = str(page_side).strip().lower()
    return s if s in {"a", "b"} else None


def _resolve_template_key(
    template_key: str | None,
    form_type: str | None,
    page_side: str | None,
) -> str | None:
    """
    Resolve template key priority:
      1) explicit template_key
      2) form_type + side => "<form_type>_<side>"
      3) config default key
    """
    if template_key:
        return str(template_key).strip()
    side = _normalize_page_side(page_side)
    if form_type and side:
        ft = str(form_type).strip()
        if ft:
            return f"{ft}_{side}"
    cfg = IMAGE_NORMALIZE
    return cfg.get("template_registration_default_key") or cfg.get("template_registration_key")


def _get_template_path(
    template_key: str | None = None,
    form_type: str | None = None,
    page_side: str | None = None,
) -> Path | None:
    """Resolve template path from key or form_type+side, then config dict/dir."""
    cfg = IMAGE_NORMALIZE
    key = _resolve_template_key(template_key, form_type, page_side)
    if key is None:
        return None

    # First: explicit mapping dict
    templates = cfg.get("template_registration_templates") or {}
    if isinstance(templates, dict):
        path = templates.get(key)
        if path is not None:
            return Path(path).resolve()

    # Second: templates directory with "<key>.png"
    templates_dir = cfg.get("template_registration_templates_dir")
    if templates_dir:
        candidate = Path(templates_dir).resolve() / f"{key}.png"
        if candidate.exists():
            return candidate
    return None


def template_key_for_page(page: int) -> str:
    """Return template key for a 1-based page number: odd pages -> side a, even -> side b (from config)."""
    cfg = IMAGE_NORMALIZE
    return cfg.get("template_key_even", "b") if page % 2 == 0 else cfg.get("template_key_odd", "a")


def _normalize_one(
    img_path: Path,
    cache_dir: Path,
    binarize: bool,
    binarize_block_size: int,
    adaptive_c: int,
    template_key: str | None = None,
    form_type: str | None = None,
    page_side: str | None = None,
    mode: str = "full",
    output_tag: str | None = None,
    *,
    out_stem: str | None = None,
) -> Path:
    """Run full pipeline on one image (B/W); save to cache_dir; return path.
    Skips processing if output already exists and input is not newer (cache hit).
    If out_stem is set, use it for the output filename (e.g. page_0001); else use _cache_stem(img_path)."""
    stem = out_stem if out_stem is not None else _cache_stem(img_path)
    norm_mode = str(mode or "full").strip().lower()
    if norm_mode not in {"full", "geometric", "postprocess"}:
        raise ValueError(
            f"Invalid normalization mode {mode!r}; expected 'full', 'geometric', or 'postprocess'."
        )
    if output_tag is not None:
        suffix = f"_{str(output_tag).strip()}"
    elif norm_mode == "geometric":
        suffix = "_homography"
    else:
        suffix = "_bin" if binarize else "_normalized"
    out_name = f"{stem}{suffix}.png"
    out_path = cache_dir / out_name
    meta_path = _norm_meta_path(out_path)
    resolved_template_key = _resolve_template_key(template_key, form_type, page_side)
    params_signature = {
        "mode": norm_mode,
        "output_tag": str(output_tag) if output_tag is not None else None,
        "binarize": bool(binarize),
        "binarize_block_size": int(binarize_block_size),
        "adaptive_c": int(adaptive_c),
        "template_key": resolved_template_key,
        "form_type": str(form_type) if form_type is not None else None,
        "page_side": str(page_side) if page_side is not None else None,
    }

    input_hash: str | None = None
    if out_path.exists():
        try:
            in_mtime = img_path.stat().st_mtime
            out_mtime = out_path.stat().st_mtime
        except OSError:
            in_mtime = None
            out_mtime = None

        meta = _load_norm_meta(meta_path)
        has_meta = bool(meta)
        meta_params = meta.get("params")
        meta_hash = str(meta.get("input_sha256", "") or "")
        params_match = isinstance(meta_params, dict) and (meta_params == params_signature)

        # Fast path: existing output with matching params and non-newer input.
        if params_match and in_mtime is not None and out_mtime is not None and in_mtime <= out_mtime:
            return out_path.resolve()

        # If only timestamp changed, verify unchanged content and reuse.
        if params_match and meta_hash and in_mtime is not None and out_mtime is not None and in_mtime > out_mtime:
            try:
                input_hash = _file_sha256(img_path)
            except OSError:
                input_hash = None
            if input_hash and input_hash == meta_hash:
                return out_path.resolve()

        # Backward compatibility for old cache entries without metadata:
        # seed metadata without forcing a recompute.
        if (not has_meta) and in_mtime is not None and out_mtime is not None and in_mtime <= out_mtime:
            try:
                input_hash = _file_sha256(img_path)
            except OSError:
                input_hash = ""
            _write_norm_meta(
                meta_path,
                {
                    "input_sha256": input_hash,
                    "params": params_signature,
                },
            )
            return out_path.resolve()

    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = _to_gray(raw)

    if norm_mode in {"full", "geometric"}:
        img = _apply_geometric_pipeline(
            img,
            template_key=template_key,
            form_type=form_type,
            page_side=page_side,
        )
    if norm_mode in {"full", "postprocess"}:
        img = _apply_postprocess_pipeline(
            img,
            binarize=binarize,
            binarize_block_size=binarize_block_size,
            adaptive_c=adaptive_c,
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    if input_hash is None:
        try:
            input_hash = _file_sha256(img_path)
        except OSError:
            input_hash = ""
    _write_norm_meta(
        meta_path,
        {
            "input_sha256": input_hash,
            "params": params_signature,
        },
    )
    return out_path.resolve()


def normalize_image(
    image_path: str | Path,
    cache_root: str | Path | None = None,
    binarize: bool | None = None,
    binarize_block_size: int | None = None,
    template_key: str | None = None,
    form_type: str | None = None,
    page_side: str | None = None,
    mode: str = "full",
    output_tag: str | None = None,
    cache_name_stem: str | None = None,
    *,
    cache_subdir: str | None = None,
) -> Path:
    """
    Normalize a single image: deskew, perspective (if doc detected), denoise, CLAHE, then binarization by default.
    Saves result under cache and returns its path. Default is binary (0/1) output.

    Args:
        image_path: Path to input image.
        cache_root: Folder for normalized outputs (default: from config).
        binarize: If True, apply adaptive binarization; if False, keep grayscale. Default: from config (True).
        binarize_block_size: Block size for adaptive threshold (default: from config); odd; larger = larger ROIs.
        template_key: Explicit template key to select registration template (e.g. "6pre_a").
        form_type: Form type used with page_side to resolve template key "<form_type>_<side>".
        page_side: Page side ("a" or "b"), used with form_type.
        mode: One of:
            - "full": geometric + postprocess (legacy behavior)
            - "geometric": homography/perspective/deskew only
            - "postprocess": denoise/contrast/binarize only
        output_tag: Optional output filename suffix tag (without leading underscore).
            When provided, output becomes "<stem>_<output_tag>.png".
        cache_name_stem: Optional output stem override (useful when input path is an intermediate file).
        cache_subdir: If set, save under cache_root/cache_subdir/ and use image stem for filename (e.g. page_0001_bin.png).

    Returns:
        Absolute path to the cached normalized image.
    """
    if cv2 is None:
        raise ImportError("OpenCV is required. Install with: pip install opencv-python")

    cfg = IMAGE_NORMALIZE
    if cache_root is None:
        cache_root = cfg["cache_root"]
    if binarize is None:
        binarize = cfg.get("binarize", True)
    if binarize_block_size is None:
        binarize_block_size = cfg["binarize_block_size"]
    adaptive_c = cfg["adaptive_threshold_c"]

    image_path = Path(image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cache_dir = Path(cache_root).resolve()
    if cache_subdir:
        cache_dir = cache_dir / cache_subdir
    out_stem = cache_name_stem or (image_path.stem if cache_subdir else None)

    return _normalize_one(
        image_path,
        cache_dir,
        binarize,
        binarize_block_size,
        adaptive_c,
        template_key=template_key,
        form_type=form_type,
        page_side=page_side,
        mode=mode,
        output_tag=output_tag,
        out_stem=out_stem,
    )


def normalize_images(
    image_paths: list[str | Path],
    cache_root: str | Path | None = None,
    binarize: bool | None = None,
    binarize_block_size: int | None = None,
    template_key: str | list[str] | None = None,
    form_type: str | list[str] | None = None,
    page_side: str | list[str] | None = None,
    mode: str = "full",
    output_tag: str | None = None,
    *,
    cache_subdir: str | None = None,
) -> list[Path]:
    """
    Normalize multiple images (batch). Same pipeline as normalize_image; binarization by default.

    Args:
        image_paths: List of paths to input images.
        cache_root: Folder for normalized outputs (default: from config).
        binarize: If True, apply adaptive binarization; if False, grayscale. Default: from config (True).
        binarize_block_size: Block size for adaptive threshold (default: from config).
        template_key: Key (or list of keys, one per image) for template_registration_templates.
        form_type: Form type (or list) to combine with page_side as "<form_type>_<side>".
        page_side: Side "a"/"b" (or list) used with form_type.
        mode: "full" (default), "geometric", or "postprocess".
        output_tag: Optional output filename suffix tag for all outputs.
        cache_subdir: If set, save under cache_root/cache_subdir/ with image stem per file (e.g. page_0001_bin.png).

    Returns:
        List of absolute paths to the cached normalized images (same order as input).
    """
    if cv2 is None:
        raise ImportError("OpenCV is required. Install with: pip install opencv-python")

    cfg = IMAGE_NORMALIZE
    if cache_root is None:
        cache_root = cfg["cache_root"]
    if binarize is None:
        binarize = cfg.get("binarize", True)
    if binarize_block_size is None:
        binarize_block_size = cfg["binarize_block_size"]
    adaptive_c = cfg["adaptive_threshold_c"]
    cache_dir = Path(cache_root).resolve()
    if cache_subdir:
        cache_dir = cache_dir / cache_subdir

    keys: list[str] | None = None
    if isinstance(template_key, list):
        if len(template_key) != len(image_paths):
            raise ValueError("template_key list length must match image_paths")
        keys = template_key
    form_types: list[str] | None = None
    if isinstance(form_type, list):
        if len(form_type) != len(image_paths):
            raise ValueError("form_type list length must match image_paths")
        form_types = form_type
    sides: list[str] | None = None
    if isinstance(page_side, list):
        if len(page_side) != len(image_paths):
            raise ValueError("page_side list length must match image_paths")
        sides = page_side

    out_paths: list[Path] = []
    for i, p in enumerate(image_paths):
        p = Path(p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        key = keys[i] if keys is not None else template_key
        ft = form_types[i] if form_types is not None else form_type
        side = sides[i] if sides is not None else page_side
        out_stem = p.stem if cache_subdir else None
        out_paths.append(
            _normalize_one(
                p,
                cache_dir,
                binarize,
                binarize_block_size,
                adaptive_c,
                template_key=key,
                form_type=ft,
                page_side=side,
                mode=mode,
                output_tag=output_tag,
                out_stem=out_stem,
            )
        )
    return out_paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_normalize.py <image_path> [--no-binarize]")
        sys.exit(1)
    path = sys.argv[1]
    binarize = "--no-binarize" not in sys.argv
    out = normalize_image(path, binarize=binarize)
    print(out)
