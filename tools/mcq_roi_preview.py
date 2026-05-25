"""MCQ ROI preview + darkness inspector for threshold tuning.

Run this against a normalized page image, schema, and ROI name to:
- crop the ROI using the same geometry as `mc_detect`
- preview the crop in a window
- print the darkness value used by MC detection and the current threshold
- show whether it is considered filled with the *current* settings

Usage (from project root):

    python3 tools/mcq_roi_preview.py \\
        --image output/cache/normalized/maury_1/page_0001_bin.png \\
        --schema-key schema_sidea \\
        --roi 1a

Options:
    --image PATH           Path to normalized page image (required).
    --schema-key KEY       Schema key (default: schema_sidea).
    --schema-path PATH     Override schema JSON path.
    --roi NAME             ROI name in schema, e.g. 1a, 2b (required).
    --threshold FLOAT      Override darkness threshold (0–1) for this run.
    --graduated            Force graduated (Gaussian) darkness.
    --flat                 Force flat mean darkness.

Requires OpenCV + numpy:
    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2  # type: ignore
import numpy as np  # type: ignore

# Allow imports from py/
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "py") not in sys.path:
    sys.path.insert(0, str(ROOT / "py"))

from mc_detect import (  # type: ignore
    MULTIPLE_CHOICE,
    _schema_path as mc_schema_path,
    _load_schema,
    _get_roi_by_name,
    _crop_to_roi,
    _label_from_roi_name,
    _to_gray,
    _darkness_fraction,
    _darkness_fraction_graduated,
    _get_graduated_weight_map,
    is_roi_filled,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview MCQ ROI crop + darkness for threshold tuning.")
    p.add_argument(
        "--image",
        required=True,
        help="Path to normalized page image (e.g. output/cache/normalized/maury_1/page_0001_bin.png).",
    )
    p.add_argument(
        "--schema-key",
        default="schema_sidea",
        help="Schema key (default: schema_sidea).",
    )
    p.add_argument(
        "--schema-path",
        help="Override schema JSON path (otherwise resolved from schema-key and config.MULTIPLE_CHOICE['schema_dir']).",
    )
    p.add_argument(
        "--roi",
        required=True,
        help="ROI name in schema, e.g. 1a, 2b.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override darkness threshold (0–1) for this run.",
    )
    grad_group = p.add_mutually_exclusive_group()
    grad_group.add_argument(
        "--graduated",
        action="store_true",
        help="Force graduated (Gaussian-weighted) darkness.",
    )
    grad_group.add_argument(
        "--flat",
        action="store_true",
        help="Force flat mean darkness (no Gaussian).",
    )
    return p.parse_args()




def _create_gradient(weights: "np.ndarray") -> "np.ndarray":
    """Create a grayscale gradient visualization of weights (0-1 float -> 0-255 uint8 grayscale)."""
    # Weights are already in [0, 1] range, convert directly to grayscale
    gradient = (weights * 255.0).astype(np.uint8)
    # Convert to BGR for display (single channel -> 3 channels)
    gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    return gradient_bgr


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.schema_path:
        # If it looks like a key (no / and no .json), treat as schema key; otherwise as path
        schema_path_str = args.schema_path
        if "/" not in schema_path_str and not schema_path_str.endswith(".json"):
            schema_path = mc_schema_path(schema_path_str)
        else:
            schema_path = Path(schema_path_str).resolve()
    else:
        schema_path = mc_schema_path(args.schema_key)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    schema = _load_schema(schema_path)
    ref_w = schema.get("image_width", 0)
    ref_h = schema.get("image_height", 0)

    # Load full page image
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Crop ROI using same geometry as mc_detect
    x, y, w, h, _, _ = _get_roi_by_name(schema, args.roi)
    crop = _crop_to_roi(img, x, y, w, h, ref_w, ref_h)
    label = _label_from_roi_name(args.roi)

    # Compute darkness metrics
    flat_dark = _darkness_fraction(crop)
    grad_dark = _darkness_fraction_graduated(crop)  # Uses new calculation: invert -> multiply -> invert -> darkness
    weight_map = _get_graduated_weight_map(crop)  # Get weight map for visualization
    
    # Calculate multiplied image for preview (inverted * weight_map, before final inversion)
    gray_crop = _to_gray(crop).astype(np.float64)
    inverted = 255.0 - gray_crop  # Invert original
    multiplied = weight_map * inverted  # Multiply by weight map

    cfg = MULTIPLE_CHOICE
    by_label = cfg.get("filled_threshold_by_label") or {}
    default_th = by_label.get(label, 0.15)
    threshold = args.threshold if args.threshold is not None else default_th

    if args.graduated:
        use_grad = True
    elif args.flat:
        use_grad = False
    else:
        use_grad = cfg.get("use_graduated_region", False)

    darkness_used = grad_dark if use_grad else flat_dark

    # Use the same detector for truth (uses original graduated, not inverted)
    filled = is_roi_filled(
        crop,
        label,
        use_graduated=use_grad,
        threshold_override=threshold if args.threshold is not None else None,
    )

    print(f"Image:        {image_path}")
    print(f"Schema:       {schema_path}")
    print(f"ROI:          {args.roi}  (label={label!r})")
    print(f"Crop size:    {crop.shape[1]}x{crop.shape[0]} px (w x h)")
    print(f"Flat darkness:      {flat_dark:.4f}")
    print(f"Graduated darkness: {grad_dark:.4f}")
    print(f"Using graduated:    {use_grad}")
    print(f"Threshold:          {threshold:.4f}")
    print(f"Filled (>= th):     {filled}")

    # Visual preview: show crop, gradient, and weighted image side-by-side, scaled up
    scale = 8
    crop_h, crop_w = crop.shape[:2]
    vis_w = crop_w * scale
    vis_h = crop_h * scale
    
    # Left: crop image (scaled up, convert to BGR if needed)
    crop_vis = crop.copy()
    if crop_vis.ndim == 2:
        crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_GRAY2BGR)
    crop_vis = cv2.resize(crop_vis, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)

    # Middle: Weight gradient (scaled up to match crop, pixel-matched)
    # Show weights directly: center = dark/low weight (0), edges = bright/high weight (1)
    weight_map_scaled = cv2.resize(weight_map, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    gradient = _create_gradient(weight_map_scaled)
    
    # Right: Multiplied image (inverted * weight_map) - mid-step before final inversion
    # This shows the result of: invert original -> multiply by weight map
    # Scale up to match visualization size (pixel-matched scaling)
    multiplied_scaled = cv2.resize(multiplied, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    # Convert to uint8 for display (multiplied is in [0, 255] range)
    multiplied_vis = np.clip(multiplied_scaled, 0, 255).astype(np.uint8)
    multiplied_vis = cv2.cvtColor(multiplied_vis, cv2.COLOR_GRAY2BGR)
    
    # Combine side-by-side
    combined = np.hstack([crop_vis, gradient, multiplied_vis])

    window_name = "MCQ ROI preview | Crop (left) | Weight map (middle) | Multiplied image (right, inverted * weights) - Press any key or close window"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)
    
    # Wait for key press or window close; check window status periodically
    try:
        while True:
            key = cv2.waitKey(100) & 0xFF  # Check every 100ms
            if key != 255:  # Any key pressed (255 = no key)
                break
            # Check if window was closed (getWindowProperty returns -1 if closed)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                # Window was destroyed
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

