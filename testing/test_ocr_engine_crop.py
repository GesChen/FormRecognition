"""
Test OCR engine on an image keeping only the top x rows.
Shows a preview of the crop, then prints timing and OCR text.

Usage (from project root):

    python3 testing/test_ocr_engine_crop.py <image_path> <x> [--no-preview]

Example:

    python3 testing/test_ocr_engine_crop.py output/cache/normalized/maury_1/page_0001_bin.png 80
"""

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import cv2
from ocr_engine import ocr


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python3 testing/test_ocr_engine_crop.py <image_path> <x>")
        print("  x = number of rows to keep from the top (only top x rows are OCR'd)")
        print("  --no-preview = skip opening the crop preview window")
        sys.exit(1)
    image_path = Path(sys.argv[1]).resolve()
    try:
        x = int(sys.argv[2])
    except ValueError:
        print("Error: x must be an integer")
        sys.exit(1)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    if x < 0:
        print("Error: x must be >= 0")
        sys.exit(1)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        sys.exit(1)
    h, w = img.shape[:2]
    if x <= 0 or x > h:
        print(f"Error: x must be in 1..{h} (image height)")
        sys.exit(1)

    # Keep only top x rows
    cropped = img[0:x, :]
    print(f"Image: {image_path}")
    print(f"Crop: top {x} rows only -> size {w}x{cropped.shape[0]}")
    print("-" * 60)

    # Preview crop (skip if --no-preview)
    show_preview = "--no-preview" not in sys.argv
    if show_preview:
        win = "Crop preview (top {} rows) — press any key to close".format(x)
        try:
            # Ensure BGR for display (imread can return 2D grayscale)
            if len(cropped.shape) == 2:
                preview = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
            else:
                preview = cropped.copy()
            # Scale to fit screen: max 1000px on the longer side, keep aspect ratio
            ph, pw = preview.shape[:2]
            max_side = 1000
            if max(ph, pw) > max_side:
                r = max_side / max(ph, pw)
                new_w = int(pw * r)
                new_h = int(ph * r)
                preview = cv2.resize(preview, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, preview)
            print("Preview: press any key in the window to continue (Ctrl+C works)...")
            while True:
                key = cv2.waitKey(100)
                if key != -1:
                    break
                try:
                    if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break  # window was closed
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Preview skipped ({e})")

    t_start = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        cv2.imwrite(tmp_path, cropped)
        t_ocr_start = time.perf_counter()
        text = ocr(tmp_path)
        t_ocr = time.perf_counter() - t_ocr_start
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    t_total = time.perf_counter() - t_start
    print(f"[DEBUG] time OCR: {t_ocr:.3f}s")
    print(f"[DEBUG] time total (crop + write + OCR): {t_total:.3f}s")
    print()
    print("Text:")
    print("-" * 60)
    print(text.strip() if text else "(empty)")
    print("-" * 60)


if __name__ == "__main__":
    main()
