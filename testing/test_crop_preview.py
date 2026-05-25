"""Show the cropped ROI from one normalized test image. Saves crop to output and displays it if possible."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
sys.path.insert(0, str(ROOT / "py" / "legacy"))

from config_legacy import ID_RECOGNITION
from config import IMAGE_NORMALIZE
import cv2
import json

# --- Easy pick: change these to switch test image and ROI (CLI overrides these) ---
DEFAULT_PAGE = 3
DEFAULT_ROI_NAME = "1a"

PDF_PAGE_CACHE = ROOT / "output" / "cache" / "maury_1"

# Normalized cache: cache_normalized / pdf_subdir / page_N_bin.png
def normalized_image_path(page: int) -> Path:
    suffix = "_bin" if IMAGE_NORMALIZE.get("binarize", True) else "_normalized"
    return ROOT / "output" / "cache" / "normalized" / "maury_1" / f"page_{page:04d}{suffix}.png"


def source_page_path(page: int) -> Path:
    return PDF_PAGE_CACHE / f"page_{page:04d}.png"


# All testing output goes under testing/ (see .cursor/rules/testing-output.mdc)
TESTING_OUTPUT_DIR = ROOT / "testing" / "output"


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


def _parse_args():
    argv = [a for a in sys.argv[1:] if a != "--no-show"]
    no_show = "--no-show" in sys.argv
    page, roi_name = DEFAULT_PAGE, DEFAULT_ROI_NAME
    i = 0
    while i < len(argv):
        if argv[i] in ("--page", "-p") and i + 1 < len(argv):
            page = int(argv[i + 1])
            i += 2
        elif argv[i] in ("--roi", "-r") and i + 1 < len(argv):
            roi_name = argv[i + 1]
            i += 2
        elif argv[i].isdigit():
            page = int(argv[i])
            i += 1
            if i < len(argv):
                roi_name = argv[i]
                i += 1
        else:
            roi_name = argv[i]
            i += 1
    return page, roi_name, no_show


if __name__ == "__main__":
    page, roi_name, no_show = _parse_args()

    NORMALIZED_IMAGE = normalized_image_path(page)
    CROP_OUTPUT = crop_output_path(page, roi_name)

    schema_path = Path(ID_RECOGNITION["schema_path"]).resolve()
    if not schema_path.exists():
        print(f"Schema not found: {schema_path}")
        sys.exit(1)

    if not NORMALIZED_IMAGE.exists():
        source = source_page_path(page)
        if not source.exists():
            print(f"Page {page}: source image not found: {source}")
            print("Run test_pdf_to_images first.")
            sys.exit(1)
        print(f"Page {page}: normalizing (ROI {roi_name!r})...")
        from image_normalize import normalize_image, template_key_for_page
        normalize_image(source, template_key=template_key_for_page(page), cache_subdir="maury_1")
        if not NORMALIZED_IMAGE.exists():
            print(f"Page {page}: normalization did not produce {NORMALIZED_IMAGE}")
            sys.exit(1)

    x, y, w, h, ref_w, ref_h = load_roi(schema_path, roi_name)
    img = cv2.imread(str(NORMALIZED_IMAGE))
    if img is None:
        print(f"Page {page}, ROI {roi_name!r}: cannot read {NORMALIZED_IMAGE}")
        sys.exit(1)

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
    crop = img[y : y + h, x : x + w]

    CROP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(CROP_OUTPUT), crop)
    print(f"Page {page}, ROI {roi_name!r}: cropped to {CROP_OUTPUT}")
    print(f"Page {page}, ROI {roi_name!r}: region ({x}, {y}) size {w}x{h}")

    if not no_show:
        try:
            win = f"{roi_name} ROI crop"
            cv2.imshow(win, crop)
            print(f"Page {page}, ROI {roi_name!r}: close the preview window or press any key to exit.")
            while True:
                key = cv2.waitKey(100)
                if key >= 0:
                    break
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 0:
                        break
                except Exception:
                    pass
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        except Exception as e:
            print(f"Page {page}, ROI {roi_name!r}: display skipped: {e}. Open the saved image to view.")
