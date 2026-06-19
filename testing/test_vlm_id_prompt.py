"""
Unified VLM ID-recognition tester.

Supports either:
1) image input directly, or
2) PDF + page number, with automatic top-crop generation.

The tester then runs the ID/form workflow and prints:
- raw OCR JSON output
- interpreted ID output
- detailed debug metadata (especially with --verbose)

Usage examples:
    python3 testing/test_vlm_id_prompt.py --pdf-path data/sample.pdf --page 1
    python3 testing/test_vlm_id_prompt.py image.png --verbose
    python3 testing/test_vlm_id_prompt.py --pdf-path data/sample.pdf --page 3 --json-only
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from config import ID_FORM_LLM, PATHS, PDF_TO_IMAGES
from id_form_llm import extract_id_and_form_type

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None


ID_FULL_RE = re.compile(r"\bID:[A-Z0-9]{8}\b", flags=re.IGNORECASE)
ID_SUFFIX_RE = re.compile(r"\b([A-Z0-9]{8})\b", flags=re.IGNORECASE)


def _vprint(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[vlm_id_tester] {message}")


def _extract_id_candidates(parsed: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    id_from_json = parsed.get("id")
    if id_from_json is not None:
        id_from_json = str(id_from_json).strip() or None

    detected_text = str(parsed.get("detected_text") or raw.get("detected_text") or "")

    id_from_detected_text = None
    m = ID_FULL_RE.search(detected_text)
    if m:
        id_from_detected_text = m.group(0).upper()
    else:
        m2 = ID_SUFFIX_RE.search(detected_text)
        if m2:
            id_from_detected_text = f"ID:{m2.group(1).upper()}"

    chosen = None
    if id_from_json:
        if id_from_json.upper().startswith("ID:"):
            chosen = id_from_json.upper()
        else:
            v = id_from_json.upper()
            if re.fullmatch(r"[A-Z0-9]{8}", v):
                chosen = f"ID:{v}"
            else:
                chosen = v
    elif id_from_detected_text:
        chosen = id_from_detected_text

    return {
        "id_from_json": id_from_json,
        "id_from_detected_text": id_from_detected_text,
        "chosen_id": chosen,
        "detected_text": detected_text,
    }


def _normalize_crop_pct(percent: float) -> float:
    return max(0.01, min(100.0, float(percent)))


def _top_rows_from_percent(image_height: int, percent: float) -> int:
    if image_height <= 0:
        return 0
    p = _normalize_crop_pct(percent)
    return max(1, min(image_height, int(round(image_height * p / 100.0))))


def _crop_pct_filename_suffix(pct: float) -> str:
    s = f"{float(pct):g}"
    return s.replace(".", "p")


def _default_id_crop_dir() -> Path:
    out_dir = Path(PATHS["output"]) / "debug_images" / "id_crop"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _render_pdf_page_to_png(pdf_path: Path, page: int, dpi: int, *, verbose: bool) -> Path:
    if fitz is None:  # pragma: no cover
        raise RuntimeError("PyMuPDF is required for --pdf-path mode. Install: pip install pymupdf")
    if page < 1:
        raise ValueError("--page must be >= 1")

    doc = fitz.open(pdf_path)
    try:
        if page > len(doc):
            raise ValueError(f"--page {page} out of range; PDF has {len(doc)} page(s)")
        zoom = float(dpi) / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = doc[page - 1].get_pixmap(matrix=matrix, alpha=False)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp = Path(f.name)
        pix.save(str(tmp))
        _vprint(verbose, f"rendered PDF page {page} at {dpi} dpi -> {tmp}")
        return tmp
    finally:
        doc.close()


def _prepare_image_input(
    *,
    image_path_arg: str | None,
    pdf_path_arg: str | None,
    page: int | None,
    pdf_dpi: int,
    crop_top_percent: float,
    write_image: bool,
    write_image_path: str | None,
    verbose: bool,
) -> tuple[Path, dict[str, Any], list[Path]]:
    """
    Resolve actual image path used for OCR.

    Returns:
      (ocr_image_path, input_meta, temp_paths_to_cleanup)
    """
    cleanup_paths: list[Path] = []
    input_meta: dict[str, Any] = {}

    if pdf_path_arg:
        pdf_path = Path(pdf_path_arg).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if page is None:
            raise ValueError("--page is required when using --pdf-path")
        rendered_page = _render_pdf_page_to_png(pdf_path, int(page), int(pdf_dpi), verbose=verbose)
        cleanup_paths.append(rendered_page)
        pct = _normalize_crop_pct(crop_top_percent)
        if write_image_path:
            crop_out = Path(write_image_path).expanduser().resolve()
            crop_out.parent.mkdir(parents=True, exist_ok=True)
        else:
            suf = _crop_pct_filename_suffix(pct)
            crop_out = _default_id_crop_dir() / f"{pdf_path.stem}_page_{int(page):04d}_crop_top{suf}pct.png"
        input_meta = {
            "input_mode": "pdf_page_crop",
            "pdf_path": str(pdf_path),
            "page": int(page),
            "pdf_dpi": int(pdf_dpi),
            "crop_top_percent": float(pct),
            "rendered_page_image": str(rendered_page),
            "crop_image": str(crop_out),
            "write_image_forced": True,
        }
        return rendered_page, input_meta, cleanup_paths

    if not image_path_arg:
        raise ValueError("Provide either <image_path> or --pdf-path + --page")

    image_path = Path(image_path_arg).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    written_copy = None
    if write_image:
        if write_image_path:
            written_copy = Path(write_image_path).expanduser().resolve()
            written_copy.parent.mkdir(parents=True, exist_ok=True)
        else:
            written_copy = _default_id_crop_dir() / f"{image_path.stem}_vlm_input{image_path.suffix or '.png'}"
        shutil.copy2(image_path, written_copy)
        _vprint(verbose, f"wrote input image copy -> {written_copy}")

    input_meta = {
        "input_mode": "image",
        "image_path": str(image_path),
        "written_image_copy": str(written_copy) if written_copy else None,
    }
    return image_path, input_meta, cleanup_paths


def run_test(
    image_path: Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    input_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _vprint(verbose, f"running ID workflow on image={image_path}")

    crop_top_percent = None
    crop_debug_out = None
    input_info = input_meta or {}
    if isinstance(input_info, dict):
        try:
            crop_top_percent = float(input_info.get("crop_top_percent"))
        except (TypeError, ValueError):
            crop_top_percent = None
        crop_img = input_info.get("crop_image")
        if crop_img:
            crop_debug_out = Path(str(crop_img))

    t0 = time.perf_counter()
    workflow_result = extract_id_and_form_type(
        image_path,
        crop_top_percent=crop_top_percent,
        return_raw_ocr=True,
        verbose=verbose,
        crop_debug_out=crop_debug_out,
    )
    out = workflow_result.get("raw_ocr") if isinstance(workflow_result, dict) else {}
    if not isinstance(out, dict):
        out = {}
    elapsed = round(time.perf_counter() - t0, 4)

    stages = out.get("stages") or []
    selected_stage = stages[-1] if stages else {}
    parsed = selected_stage.get("parsed") if isinstance(selected_stage.get("parsed"), dict) else {}

    id_data = _extract_id_candidates(parsed, out)
    workflow_id = workflow_result.get("id") if isinstance(workflow_result, dict) else None
    if workflow_id:
        id_data["chosen_id"] = str(workflow_id).strip()

    return {
        "meta": {
            "ocr_image_path": str(image_path.resolve()),
            "elapsed_sec": elapsed,
            "model": out.get("selected_model"),
            "selected_stage_index": out.get("selected_stage_index"),
            "needs_human_review": bool(out.get("needs_human_review", False)),
            "input": input_meta or {},
        },
        "debug": {},
        "raw_output": out,
        "workflow_output": workflow_result,
        "interpreted": {
            "id": id_data.get("chosen_id"),
            "id_from_json": id_data.get("id_from_json"),
            "id_from_detected_text": id_data.get("id_from_detected_text"),
            "detected_text": id_data.get("detected_text"),
            "workflow_form_type": (
                workflow_result.get("form_type") if isinstance(workflow_result, dict) else None
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified VLM tester for ID recognition (image or PDF page crop).")
    p.add_argument("image_path", nargs="?", help="Optional direct input image path.")
    p.add_argument("--pdf-path", default=None, help="PDF input path (use with --page).")
    p.add_argument("--page", type=int, default=None, help="1-based page number for --pdf-path.")
    p.add_argument("--pdf-dpi", type=int, default=int(PDF_TO_IMAGES.get("dpi", 150)), help="PDF render DPI.")
    p.add_argument(
        "--crop-top-percent",
        type=float,
        default=float(ID_FORM_LLM.get("crop_top_percent", 15.0)),
        help="Top-percent crop for PDF mode.",
    )
    p.add_argument("--timeout", type=int, default=None, help="OCR timeout (seconds).")
    p.add_argument(
        "--write-image",
        action="store_true",
        help="Image mode only: write a copy of input image into output/debug_images/id_crop.",
    )
    p.add_argument(
        "--write-image-path",
        default=None,
        help="Optional explicit output path for written image/crop.",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose running/debug logs.")
    p.add_argument("--json-only", action="store_true", help="Print only JSON output.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.image_path and args.pdf_path:
        print("Error: provide either <image_path> or --pdf-path, not both.")
        sys.exit(1)

    try:
        ocr_image_path, input_meta, cleanup_paths = _prepare_image_input(
            image_path_arg=args.image_path,
            pdf_path_arg=args.pdf_path,
            page=args.page,
            pdf_dpi=int(args.pdf_dpi),
            crop_top_percent=float(args.crop_top_percent),
            write_image=bool(args.write_image),
            write_image_path=args.write_image_path,
            verbose=bool(args.verbose),
        )
    except Exception as exc:
        print(f"Error preparing input: {exc}")
        sys.exit(1)

    if args.verbose:
        print("=" * 72)
        print("RUN INPUT")
        print("=" * 72)
        print(json.dumps(input_meta, indent=2, default=str))

    try:
        result = run_test(
            ocr_image_path,
            timeout=args.timeout,
            verbose=bool(args.verbose),
            input_meta=input_meta,
        )
    finally:
        for p in cleanup_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    if args.json_only:
        print(json.dumps(result, indent=2, default=str))
        return

    print("=" * 72)
    print("RAW JSON OUTPUT")
    print("=" * 72)
    print(json.dumps(result.get("raw_output") or {}, indent=2, default=str))

    interp = result.get("interpreted") or {}
    meta = result.get("meta") or {}
    print("\n" + "=" * 72)
    print("INTERPRETED OUTPUT")
    print("=" * 72)
    print(f"OCR image: {meta.get('ocr_image_path')}")
    print(f"Elapsed: {meta.get('elapsed_sec')}s")
    print(f"Model: {meta.get('model')}")
    print(f"Needs human review: {meta.get('needs_human_review')}")
    print(f"Input mode: {(meta.get('input') or {}).get('input_mode')}")
    if (meta.get("input") or {}).get("crop_image"):
        print(f"Crop image: {(meta.get('input') or {}).get('crop_image')}")
    if (meta.get("input") or {}).get("written_image_copy"):
        print(f"Written image copy: {(meta.get('input') or {}).get('written_image_copy')}")
    print(f"Detected text: {interp.get('detected_text')!r}")
    print(f"ID (chosen): {interp.get('id')!r}")
    print(f"ID from JSON field: {interp.get('id_from_json')!r}")
    print(f"ID from detected_text regex: {interp.get('id_from_detected_text')!r}")

    if args.verbose:
        print("\n" + "=" * 72)
        print("FULL RESULT JSON")
        print("=" * 72)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
