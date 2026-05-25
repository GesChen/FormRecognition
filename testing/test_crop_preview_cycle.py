"""Cycle through cropped ROI from multiple normalized pages. Display each for a set time. Optional save."""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "py" / "legacy"))

from config_legacy import ID_RECOGNITION
from config import IMAGE_NORMALIZE
import cv2
import json

# --- Easy pick: customize pages, ROI, schema, and display time (CLI overrides these) ---
DEFAULT_PAGES = [3]
DEFAULT_ROI_NAME = "1a"
DEFAULT_SCHEMA_KEY = "schema_sidea"  # e.g. "schema_sidea" for data/roi_schemas/<key>.json; None = config schema_path
DEFAULT_DISPLAY_SECONDS = 10.1

PDF_PAGE_CACHE = ROOT / "output" / "cache" / "maury_1"

# Normalized images: cache_normalized / pdf_subdir / page_N_bin.png
def normalized_image_path(page: int) -> Path:
    suffix = "_bin" if IMAGE_NORMALIZE.get("binarize", True) else "_normalized"
    return ROOT / "output" / "cache" / "normalized" / "maury_1" / f"page_{page:04d}{suffix}.png"


def source_page_path(page: int) -> Path:
    return PDF_PAGE_CACHE / f"page_{page:04d}.png"


TESTING_OUTPUT_DIR = ROOT / "testing" / "output"
SCHEMA_DIR = ROOT / "data" / "roi_schemas"


def crop_output_path(page: int, roi_name: str) -> Path:
    stem = f"{roi_name}_crop_preview"
    if page != 1:
        stem += f"_page{page}"
    return TESTING_OUTPUT_DIR / f"{stem}.png"


def load_roi(schema_path: Path, roi_name: str):
    data = json.loads(schema_path.read_text())
    ref_w = data.get("image_width", 0)
    ref_h = data.get("image_height", 0)
    for roi in data.get("rois", []):
        if roi.get("name") == roi_name:
            return int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"]), ref_w, ref_h
    raise KeyError(f"No ROI named {roi_name!r} in schema")


def _crop_page(schema_path: Path, roi_name: str, page: int):
    """Load or normalize image, crop to ROI; return crop array or None on failure."""
    norm_path = normalized_image_path(page)
    if not norm_path.exists():
        source = source_page_path(page)
        if not source.exists():
            print(f"Page {page}: source not found: {source}. Skip.")
            return None
        print(f"Page {page}, ROI {roi_name!r}: normalizing...")
        from image_normalize import normalize_image, template_key_for_page
        normalize_image(source, template_key=template_key_for_page(page), cache_subdir="maury_1")
        if not norm_path.exists():
            print(f"Page {page}: normalization failed. Skip.")
            return None
    img = cv2.imread(str(norm_path))
    if img is None:
        print(f"Page {page}, ROI {roi_name!r}: cannot read {norm_path}. Skip.")
        return None
    x, y, w, h, ref_w, ref_h = load_roi(schema_path, roi_name)
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
    return img[y : y + h, x : x + w]


def _parse_args():
    argv = list(sys.argv[1:])
    no_show = "--no-show" in argv
    save = "--save" in argv
    argv = [a for a in argv if a not in ("--no-show", "--save")]
    pages = list(DEFAULT_PAGES)
    roi_name = DEFAULT_ROI_NAME
    schema_key = DEFAULT_SCHEMA_KEY
    display_sec = DEFAULT_DISPLAY_SECONDS
    i = 0
    while i < len(argv):
        if argv[i] in ("--schema", "-s") and i + 1 < len(argv):
            schema_key = argv[i + 1] or None
            i += 2
            continue
        if argv[i] in ("--pages", "-p") and i + 1 < len(argv):
            # --pages 1 3 5 7 (consume until next option or end)
            i += 1
            pages = []
            while i < len(argv) and argv[i].isdigit():
                pages.append(int(argv[i]))
                i += 1
            continue
        if argv[i] in ("--roi", "-r") and i + 1 < len(argv):
            roi_name = argv[i + 1]
            i += 2
            continue
        if argv[i] in ("--time", "-t") and i + 1 < len(argv):
            try:
                display_sec = float(argv[i + 1])
            except ValueError:
                display_sec = DEFAULT_DISPLAY_SECONDS
            i += 2
            continue
        if argv[i].isdigit():
            pages = [int(argv[i])]
            i += 1
            if i < len(argv) and not argv[i].startswith("-") and not argv[i].isdigit():
                roi_name = argv[i]
                i += 1
            continue
        if argv[i].replace(".", "").isdigit():
            display_sec = float(argv[i])
            i += 1
            continue
        roi_name = argv[i]
        i += 1
    return pages, roi_name, schema_key, display_sec, no_show, save


if __name__ == "__main__":
    pages, roi_name, schema_key, display_sec, no_show, save = _parse_args()
    if not pages:
        print("No pages given. Use --pages 1 3 5 or set DEFAULT_PAGES.")
        sys.exit(1)

    schema_path = (SCHEMA_DIR / f"{schema_key}.json").resolve() if schema_key else Path(ID_RECOGNITION["schema_path"]).resolve()
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        sys.exit(1)

    print(f"Pages: {pages}, ROI: {roi_name!r}, schema: {schema_key or schema_path}, display: {display_sec}s, save: {save}")

    win = f"{roi_name} ROI cycle"

    for idx, page in enumerate(pages):
        crop = _crop_page(schema_path, roi_name, page)
        if crop is None:
            continue
        if save:
            out_path = crop_output_path(page, roi_name)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop)
            print(f"Page {page}, ROI {roi_name!r}: saved {out_path}")
        if no_show:
            continue
        try:
            cv2.imshow(win, crop)
            cv2.setWindowTitle(win, f"{roi_name} ROI — page {page} ({idx + 1}/{len(pages)})")
        except Exception as e:
            print(f"Page {page}: display error: {e}")
            break
        t0 = time.monotonic()
        user_quit = False
        while (time.monotonic() - t0) < display_sec:
            key = cv2.waitKey(100)
            if key >= 0:
                print("Key pressed — exiting cycle.")
                user_quit = True
                break
            try:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 0:
                    user_quit = True
                    break
            except Exception:
                pass
        if user_quit:
            break

    if not no_show:
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
    print("Done.")
