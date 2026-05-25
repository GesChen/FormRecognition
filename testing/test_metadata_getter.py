"""Test metadata getter: OCR date, school, teacher (and optional fields) on odd pages, side a.

Uses config METADATA_GETTER (default OCR engine, default fields date/school/teacher).
Normalized images must exist for the given pages (odd = side a). Optional --save writes
results to testing/output/metadata_getter_results.json. Optional previews: set
DEFAULT_OUTPUT_PREVIEW_FILES / DEFAULT_SHOW_PREVIEW_WINDOWS or use --no-output-previews / --no-show.

Usage:
    python test_metadata_getter.py [--page N] [--pages N M ...] [--save] [--no-show] [--no-output-previews]
    python test_metadata_getter.py --page 1
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "py" / "legacy"))

import cv2
import numpy as np
from config import IMAGE_NORMALIZE
from config_legacy import METADATA_GETTER

# --- Options (CLI overrides) ---
DEFAULT_PAGE = 1
DEFAULT_PAGES = [1, 3, 5]  # odd pages, side a
DEFAULT_OUTPUT_PREVIEW_FILES = False
DEFAULT_SHOW_PREVIEW_WINDOWS = True

CACHE_NORMALIZED = ROOT / "output" / "cache" / "normalized"
TESTING_OUTPUT_DIR = ROOT / "testing" / "output"
SCHEMA_DIR = ROOT / "data" / "roi_schemas"
PREVIEW_SIDE = 160


def normalized_image_path(page: int) -> Path:
    suffix = "_bin" if IMAGE_NORMALIZE.get("binarize", True) else "_normalized"
    return CACHE_NORMALIZED / "maury_1" / f"page_{page:04d}{suffix}.png"


def _parse_args():
    argv = list(sys.argv[1:])
    save = "--save" in argv
    no_show = "--no-show" in argv
    output_preview_files = DEFAULT_OUTPUT_PREVIEW_FILES and "--no-output-previews" not in argv
    argv = [a for a in argv if a not in ("--save", "--no-show", "--no-output-previews")]
    page = DEFAULT_PAGE
    pages = list(DEFAULT_PAGES)
    i = 0
    while i < len(argv):
        if argv[i] in ("--page", "-p") and i + 1 < len(argv):
            page = int(argv[i + 1])
            pages = [page]
            i += 2
            continue
        if argv[i] in ("--pages",) and i + 1 < len(argv):
            i += 1
            pages = []
            while i < len(argv) and argv[i].isdigit():
                pages.append(int(argv[i]))
                i += 1
            continue
        if argv[i].isdigit():
            page = int(argv[i])
            pages = [page]
            i += 1
            continue
        i += 1
    show_preview_windows = DEFAULT_SHOW_PREVIEW_WINDOWS and not no_show
    return page, pages, save, output_preview_files, show_preview_windows


def _load_schema_roi(schema_path: Path, roi_name: str):
    """Return (x, y, w, h, ref_w, ref_h) for roi_name from schema."""
    data = json.loads(schema_path.read_text())
    ref_w = data.get("image_width", 0)
    ref_h = data.get("image_height", 0)
    for roi in data.get("rois", []):
        if roi.get("name") == roi_name:
            return int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"]), ref_w, ref_h
    raise KeyError(f"No ROI {roi_name!r} in schema")


def _crop_roi_from_image(img, x: int, y: int, w: int, h: int, ref_w: int, ref_h: int):
    """Crop image to ROI; scale coords if image size != ref. Returns BGR or gray patch."""
    H, W = img.shape[:2]
    if ref_w > 0 and ref_h > 0 and (W != ref_w or H != ref_h):
        x = int(x * (W / ref_w))
        y = int(y * (H / ref_h))
        w = int(w * (W / ref_w))
        h = int(h * (H / ref_h))
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return img[y : y + h, x : x + w].copy()


def _preview_image(crop):
    """Put ROI crop in a square BGR image (letterboxed) for imshow."""
    crop = np.ascontiguousarray(crop)
    if crop.dtype != np.uint8:
        if np.issubdtype(crop.dtype, np.floating):
            crop = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
        else:
            crop = np.clip(crop, 0, 255).astype(np.uint8)
    else:
        crop = crop.copy()
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((PREVIEW_SIDE, PREVIEW_SIDE, 3), dtype=np.uint8)
    scale = min(PREVIEW_SIDE / h, PREVIEW_SIDE / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((PREVIEW_SIDE, PREVIEW_SIDE, 3), 128, dtype=np.uint8)
    y0 = (PREVIEW_SIDE - new_h) // 2
    x0 = (PREVIEW_SIDE - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = scaled
    return np.ascontiguousarray(canvas)


if __name__ == "__main__":
    from metadata_getter import get_metadata, get_metadata_batch

    page, pages, save, output_preview_files, show_preview_windows = _parse_args()

    # Odd pages only (side a); ensure normalized images exist
    paths = [normalized_image_path(p) for p in pages]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"Missing normalized images: {missing[0]}")
        print("Run test_pdf_to_images then normalize (e.g. test_image_normalize or test_crop_preview_cycle) first.")
        sys.exit(1)

    schema_key = METADATA_GETTER.get("schema_key", "schema_sidea")
    schema_path = (SCHEMA_DIR / f"{schema_key}.json").resolve()
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        sys.exit(1)

    field_names = list(METADATA_GETTER.get("field_names", ["date", "school", "teacher"]))
    print(f"Metadata getter: pages {pages}, fields {field_names}, engine {METADATA_GETTER.get('ocr_engine', 'ocr_engine')}")

    if len(paths) == 1:
        meta = get_metadata(paths[0])
        results = [meta]
        print(f"\nPage {pages[0]} — metadata:")
        for k, v in meta.items():
            print(f"  {k}: {repr(v)}")
    else:
        results = get_metadata_batch(paths)
        print("\nBatch — metadata:")
        for p, meta in zip(pages, results):
            print(f"  Page {p}: {meta}")

    # Previews of ROI regions (like mc_detect tester)
    display_list = []
    TESTING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for p, path in zip(pages, paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        for field_name in field_names:
            try:
                x, y, w, h, ref_w, ref_h = _load_schema_roi(schema_path, field_name)
            except KeyError:
                continue
            crop = _crop_roi_from_image(img, x, y, w, h, ref_w, ref_h)
            title = f"p{p} {field_name}" if len(pages) > 1 else f"metadata — {field_name}"
            disp = _preview_image(crop)
            display_list.append((title, disp))
            if output_preview_files:
                out_path = TESTING_OUTPUT_DIR / f"meta_preview_p{p}_{field_name}.png"
                cv2.imwrite(str(out_path), disp)
    if display_list and output_preview_files:
        print(f"\nSaved {len(display_list)} preview images to {TESTING_OUTPUT_DIR}")
    if show_preview_windows and display_list:
        for title, disp in display_list:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, disp)
            cv2.resizeWindow(title, PREVIEW_SIDE, PREVIEW_SIDE)
            cv2.waitKey(1)
        print("Preview windows open. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    if save:
        out_data = {f"page_{p}": meta for p, meta in zip(pages, results)}
        out_file = TESTING_OUTPUT_DIR / "metadata_getter_results.json"
        out_file.write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved: {out_file}")

    print("Done.")
