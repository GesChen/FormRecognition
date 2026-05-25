"""Test multiple-choice (bubble) filled detection on normalized pages.

Runs mc_detect on one page or a batch of pages for the given ROIs and prints
filled/empty per ROI. By default, preview images are saved to testing/output/mc_preview_*.png unless
DEFAULT_OUTPUT_PREVIEW_FILES is False or --no-output-previews. Preview windows open
unless DEFAULT_SHOW_PREVIEW_WINDOWS is False or --no-show.

Usage:
    python test_mc_detect.py [--page N] [--pages N M ...] [--schema KEY] [--roi 1a 1b 1c 1d] [--save] [--no-show] [--no-output-previews]
    python test_mc_detect.py 1 1a 1b 1c 1d
    python test_mc_detect.py --page 1 --no-show   # write preview PNGs (unless --no-output-previews)
    python test_mc_detect.py --no-output-previews   # skip writing mc_preview_*.png files
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

# --- Options (CLI overrides) ---
DEFAULT_PAGE = 1
DEFAULT_PAGES = range(1, 20, 2)
DEFAULT_SCHEMA_KEY = "schema_sidea"
DEFAULT_ROI_NAMES = ["1a", "1b", "1c", "1d"]
# Write preview images (mc_preview_*.png) to testing/output; use --no-output-previews to disable
DEFAULT_OUTPUT_PREVIEW_FILES = False
# Show cv2 preview windows (press key to close); use --no-show to disable
DEFAULT_SHOW_PREVIEW_WINDOWS = True

CACHE_NORMALIZED = ROOT / "output" / "cache" / "normalized"
TESTING_OUTPUT_DIR = ROOT / "testing" / "output"


def normalized_image_path(page: int) -> Path:
    suffix = "_bin" if IMAGE_NORMALIZE.get("binarize", True) else "_normalized"
    return CACHE_NORMALIZED / "maury_1" / f"page_{page:04d}{suffix}.png"


def get_normalized_path(page: int) -> Path:
    return normalized_image_path(page)


def _parse_args():
    argv = list(sys.argv[1:])
    save = "--save" in argv
    no_show = "--no-show" in argv
    output_preview_files = DEFAULT_OUTPUT_PREVIEW_FILES and "--no-output-previews" not in argv
    argv = [a for a in argv if a not in ("--save", "--no-show", "--no-output-previews")]
    page = DEFAULT_PAGE
    pages = list(DEFAULT_PAGES)
    schema_key = DEFAULT_SCHEMA_KEY
    roi_names = list(DEFAULT_ROI_NAMES)
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
        if argv[i] in ("--schema", "-s") and i + 1 < len(argv):
            schema_key = argv[i + 1]
            i += 2
            continue
        if argv[i] in ("--roi", "-r") and i + 1 < len(argv):
            i += 1
            roi_names = []
            while i < len(argv) and not argv[i].startswith("-") and argv[i] not in ("--page", "--pages", "--schema", "--roi"):
                roi_names.append(argv[i])
                i += 1
            continue
        if argv[i].isdigit():
            page = int(argv[i])
            pages = [page]
            i += 1
            if i < len(argv) and not argv[i].startswith("-") and not argv[i].isdigit():
                roi_names = [argv[i]]
                i += 1
                while i < len(argv) and not argv[i].startswith("-"):
                    roi_names.append(argv[i])
                    i += 1
            continue
        i += 1
    show_preview_windows = DEFAULT_SHOW_PREVIEW_WINDOWS and not no_show
    return page, pages, schema_key, roi_names, save, no_show, output_preview_files, show_preview_windows


# Square preview side length (each ROI shown in a square window with letterboxing)
PREVIEW_SIDE = 160
SCHEMA_DIR = ROOT / "data" / "roi_schemas"


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
    """Put ROI crop in a square BGR image (letterboxed) for imshow. Ensures uint8 and keeps detail."""
    crop = np.ascontiguousarray(crop)
    # Normalize to uint8 0-255
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
    # Scale crop to fit inside PREVIEW_SIDE x PREVIEW_SIDE, preserve aspect ratio, center on square canvas
    scale = min(PREVIEW_SIDE / h, PREVIEW_SIDE / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    scaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((PREVIEW_SIDE, PREVIEW_SIDE, 3), 128, dtype=np.uint8)  # gray background
    y0 = (PREVIEW_SIDE - new_h) // 2
    x0 = (PREVIEW_SIDE - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = scaled
    return np.ascontiguousarray(canvas)


if __name__ == "__main__":
    from mc_detect import detect_page, detect_batch

    page, pages, schema_key, roi_names, save, no_show, output_preview_files, show_preview_windows = _parse_args()

    if not roi_names:
        print("No ROI names given. Use --roi 1a 1b 1c 1d or set DEFAULT_ROI_NAMES.")
        sys.exit(1)

    # Check normalized images exist
    missing = [p for p in pages if not normalized_image_path(p).exists()]
    if missing:
        print(f"Missing normalized images for page(s): {missing}")
        print("Run test_pdf_to_images then test_image_normalize (or test_crop_preview_cycle) first.")
        sys.exit(1)

    schema_path = (SCHEMA_DIR / f"{schema_key}.json").resolve()
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        sys.exit(1)

    print(f"Schema: {schema_key}, ROIs: {roi_names}, page(s): {pages}")

    if len(pages) == 1:
        results = detect_page(get_normalized_path(pages[0]), schema_key, roi_names)
        print(f"\nPage {pages[0]} — filled detection:")
        for name, filled in results.items():
            print(f"  {name}: {'filled' if filled else 'empty'}")
        out_data = {f"page_{pages[0]}": results}
        page_list = pages
    else:
        results_list = detect_batch(pages, schema_key, roi_names, get_normalized_path)
        print("\nBatch — filled detection:")
        for p, res in zip(pages, results_list):
            print(f"  Page {p}: {res}")
        out_data = {f"page_{p}": res for p, res in zip(pages, results_list)}
        page_list = pages

    # Build preview crops ourselves: load each image, crop each ROI, so display is guaranteed correct
    TESTING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    display_list = []
    for p in page_list:
        path = get_normalized_path(p)
        img = cv2.imread(str(path))
        if img is None:
            print(f"Warning: could not read {path}, skipping previews for page {p}")
            continue
        for roi_name in roi_names:
            try:
                x, y, w, h, ref_w, ref_h = _load_schema_roi(schema_path, roi_name)
            except KeyError:
                continue
            crop = _crop_roi_from_image(img, x, y, w, h, ref_w, ref_h)
            filled = out_data.get(f"page_{p}", {}).get(roi_name, False)
            label = "filled" if filled else "empty"
            title = f"p{p} {roi_name} — {label}" if len(page_list) > 1 else f"ROI {roi_name} — {label}"
            disp = _preview_image(crop)
            display_list.append((title, disp))
            if output_preview_files:
                out_path = TESTING_OUTPUT_DIR / f"mc_preview_p{p}_{roi_name}_{label}.png"
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
        TESTING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_file = TESTING_OUTPUT_DIR / "mc_detect_results.json"
        out_file.write_text(json.dumps(out_data, indent=2))
        print(f"\nSaved: {out_file}")

    print("Done.")
