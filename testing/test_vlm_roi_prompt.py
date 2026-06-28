"""
VLM ROI prompt tester for one PDF page in an odd/even page pair.

Given a PDF, a 1-based page number, the page-pair form type, a target ROI, and
a custom VLM prompt, this script:
  1) renders the requested page pair,
  2) normalizes both pages using form_type + side a/b template registration,
  3) crops the target ROI from the requested page,
  4) saves the crop under output/debug_images/vlm_roi_tester/,
  5) sends the crop to the configured Ollama VLM with the custom prompt, and
  6) prints the full model output plus status updates at major steps.

Examples:
    python3 -u testing/test_vlm_roi_prompt.py \\
      data/sample.pdf 1 6pre id '{"detected_text":""}'

    python3 -u testing/test_vlm_roi_prompt.py \\
      data/sample.pdf 2 6pre 118,120,80,34 prompts/my_roi_prompt.txt

    python3 -u testing/test_vlm_roi_prompt.py \\
      data/sample.pdf 1 6pre page prompts/my_page_prompt.txt
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from config import LLM, OCR_ENGINE, PATHS, PDF_RECOGNITION, PDF_TO_IMAGES
from image_normalize import normalize_image
from ocr_engine import (
    _completed_json_from_stream as _engine_completed_json_from_stream,
    _stream_stop_completion as _engine_stream_stop_completion,
)
from pdf_to_images import pdf_to_images


def _status(message: str) -> None:
    print(f"[vlm_roi_tester] {message}", flush=True)


def _cfg() -> dict[str, Any]:
    return OCR_ENGINE if isinstance(OCR_ENGINE, dict) else {}


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_cfg().get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _cfg_bool(key: str, default: bool) -> bool:
    v = _cfg().get(key, default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _page_side(page: int) -> str:
    return "a" if int(page) % 2 == 1 else "b"


def _pair_pages(page: int) -> list[int]:
    page = int(page)
    if page < 1:
        raise ValueError("--page must be >= 1")
    return [page, page + 1] if page % 2 == 1 else [page - 1, page]


def _schema_path(form_type: str, side: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    schema_dir = Path(
        PDF_RECOGNITION.get("schema_dir", ROOT / "data" / "roi_schemas")
    ).resolve()
    return schema_dir / f"{form_type}_{side}.json"


def _load_schema_roi(
    *,
    form_type: str,
    side: str,
    roi_name: str,
    schema_path_arg: str | None,
) -> tuple[tuple[float, float, float, float], dict[str, Any], Path]:
    schema_path = _schema_path(form_type, side, schema_path_arg)
    if not schema_path.exists():
        raise FileNotFoundError(f"ROI schema not found: {schema_path}")
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    for roi in data.get("rois") or []:
        if str(roi.get("name", "")).strip() != roi_name:
            continue
        box = (
            float(roi.get("x", 0)),
            float(roi.get("y", 0)),
            float(roi.get("w", 0)),
            float(roi.get("h", 0)),
        )
        return box, data, schema_path
    raise KeyError(f"ROI {roi_name!r} not found in {schema_path}")


def _parse_roi(
    *,
    roi_arg: str,
    form_type: str,
    side: str,
    schema_path_arg: str | None,
) -> tuple[tuple[float, float, float, float], dict[str, Any] | None, Path | None, str]:
    roi = str(roi_arg).strip()
    m = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*",
        roi,
    )
    if m:
        vals = tuple(float(x) for x in m.groups())
        return vals, None, None, "explicit_xywh"

    box, schema, schema_path = _load_schema_roi(
        form_type=form_type,
        side=side,
        roi_name=roi,
        schema_path_arg=schema_path_arg,
    )
    return box, schema, schema_path, "schema_roi"


def _scale_xywh(
    box: tuple[float, float, float, float],
    *,
    image_size: tuple[int, int],
    schema: dict[str, Any] | None,
) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    ref_w = int(schema.get("image_width") or 0) if isinstance(schema, dict) else 0
    ref_h = int(schema.get("image_height") or 0) if isinstance(schema, dict) else 0
    sx = image_w / ref_w if ref_w > 0 and ref_w != image_w else 1.0
    sy = image_h / ref_h if ref_h > 0 and ref_h != image_h else 1.0

    x, y, w, h = box
    x1 = max(0, int(round(x * sx)))
    y1 = max(0, int(round(y * sy)))
    x2 = min(image_w, int(round((x + w) * sx)))
    y2 = min(image_h, int(round((y + h) * sy)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"ROI crop is empty after scaling/clamping: xywh={box}, scaled={(x1, y1, x2, y2)}, image={image_size}"
        )
    return x1, y1, x2, y2


def _expand_box(
    box: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    expand_px: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    image_w, image_h = image_size
    pad = max(0, int(expand_px or 0))
    if pad <= 0:
        return box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(image_w, x2 + pad),
        min(image_h, y2 + pad),
    )


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        path = Path(args.prompt_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")
    prompt = str(args.prompt or "")
    if prompt and getattr(args, "prompt_from_pos", False):
        looks_like_path = (
            len(prompt) < 240
            and "\n" not in prompt
            and not any(ch in prompt for ch in "{}:")
            and (
                "/" in prompt
                or "\\" in prompt
                or Path(prompt).suffix.lower() in {".txt", ".md", ".prompt"}
            )
        )
        if looks_like_path:
            try:
                maybe_path = Path(prompt).expanduser()
                if maybe_path.exists() and maybe_path.is_file():
                    return maybe_path.resolve().read_text(encoding="utf-8")
            except OSError:
                pass
    if not prompt.strip():
        raise ValueError("Provide --prompt or --prompt-file")
    return prompt


def _save_crop(
    *,
    normalized_page: Path,
    scaled_box: tuple[int, int, int, int],
    out_dir: Path,
    pdf_path: Path,
    page: int,
    form_type: str,
    roi_label: str,
) -> tuple[Path, tuple[int, int], tuple[int, int]]:
    with Image.open(normalized_page) as img:
        gray = img.convert("L")
        crop = gray.crop(scaled_box)
        safe_pdf = re.sub(r"[^\w\-.]", "_", pdf_path.stem).strip("_") or "pdf"
        safe_roi = re.sub(r"[^\w\-.]", "_", roi_label).strip("_") or "roi"
        crop_path = out_dir / f"{safe_pdf}_p{page:04d}_{form_type}_{_page_side(page)}_{safe_roi}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path)
        return crop_path.resolve(), gray.size, crop.size


def _remove_horizontal_lines_from_file(
    image_path: Path,
    *,
    min_line_len: int,
    thickness: int,
) -> dict[str, Any]:
    import cv2

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Cannot read crop for horizontal line removal: {image_path}")

    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    line_len = max(2, int(min_line_len or 80))
    line_thickness = max(1, int(thickness or 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_len, 1))
    line_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    if line_thickness > 1:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, line_thickness))
        line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)

    removed_pixels = int(cv2.countNonZero(line_mask))
    cleaned = cv2.inpaint(gray, line_mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(str(image_path), cleaned)
    mask_path = image_path.with_name(f"{image_path.stem}_horizontal_line_mask.png")
    cv2.imwrite(str(mask_path), line_mask)
    return {
        "enabled": True,
        "min_line_len": line_len,
        "thickness": line_thickness,
        "removed_pixels": removed_pixels,
        "mask_path": str(mask_path.resolve()),
    }


def _encode_image_for_vlm(image_path: Path, jpeg_quality: int) -> tuple[str, dict[str, Any]]:
    with Image.open(image_path) as img:
        gray = img.convert("L")
        quality = max(1, min(100, int(jpeg_quality)))
        subsampling = 0 if quality >= 95 else 2
        buf = BytesIO()
        gray.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=False,
            subsampling=subsampling,
        )
    data = buf.getvalue()
    return base64.b64encode(data).decode("ascii"), {
        "jpeg_quality": quality,
        "jpeg_subsampling": "4:4:4" if subsampling == 0 else "4:2:0",
        "jpeg_bytes": len(data),
    }


def _build_payload(image_b64: str, prompt: str, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "stream": bool(args.stream),
        "keep_alive": int(args.keep_alive),
        "images": [image_b64],
    }
    extra = _cfg().get("extra_params")
    if isinstance(extra, dict):
        payload.update(extra)
    if args.temperature is not None:
        options = dict(payload.get("options") or {})
        options["temperature"] = float(args.temperature)
        payload["options"] = options
    return payload


def _call_vlm(payload: dict[str, Any], args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    url = f"http://{args.host}:{int(args.port)}/api/generate"
    _status(f"calling VLM model={args.model} url={url} stream={bool(args.stream)}")
    started = time.perf_counter()

    if bool(args.stream):
        parts: list[str] = []
        terminal: dict[str, Any] = {}
        chunk_count = 0
        stream_stop: dict[str, int | str] | None = None
        interrupted = False
        try:
            with requests.post(
                url,
                json=payload,
                timeout=(int(args.connect_timeout), int(args.timeout)),
                stream=True,
            ) as resp:
                _status(f"VLM HTTP status={resp.status_code}")
                resp.raise_for_status()
                for raw_line in resp.iter_lines(chunk_size=1, decode_unicode=False):
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace")
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk_count += 1
                    delta = str(obj.get("response", "") or "")
                    if delta:
                        parts.append(delta)
                        stream_stop = _engine_stream_stop_completion("".join(parts))
                        if stream_stop is not None:
                            terminal = obj
                            break
                    if obj.get("done"):
                        terminal = obj
                        break
        except KeyboardInterrupt:
            interrupted = True
            _status("KeyboardInterrupt during VLM stream; printing accumulated output before returning")
            print("\n--- ACCUMULATED VLM OUTPUT START ---", flush=True)
            print("".join(parts), flush=True)
            print("--- ACCUMULATED VLM OUTPUT END ---\n", flush=True)
        elapsed = time.perf_counter() - started
        full_text = "".join(parts)
        text = _engine_completed_json_from_stream(full_text, stream_stop)
        terminal["streaming"] = {
            "enabled": True,
            "chunk_count": chunk_count,
            "interrupted": interrupted,
            "stopped_for_stream_completion": stream_stop is not None,
            "stop_reason": (
                str(stream_stop.get("stop_reason"))
                if isinstance(stream_stop, dict) and stream_stop.get("stop_reason")
                else None
            ),
            "stream_completion": stream_stop,
            "raw_response_chars": len(full_text),
            "returned_response_chars": len(text),
        }
        terminal["elapsed_sec"] = round(elapsed, 4)
        terminal["response"] = text
        return text, terminal

    resp = requests.post(
        url,
        json=payload,
        timeout=(int(args.connect_timeout), int(args.timeout)),
    )
    _status(f"VLM HTTP status={resp.status_code}")
    resp.raise_for_status()
    body = resp.json()
    elapsed = time.perf_counter() - started
    text = str(body.get("response", "") or "")
    body["streaming"] = {"enabled": False}
    body["elapsed_sec"] = round(elapsed, 4)
    return text, body


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a PDF page pair, crop one ROI, and test a custom VLM prompt on the crop.",
    )
    parser.add_argument("pdf_path_pos", nargs="?", help="Input PDF path.")
    parser.add_argument("page_pos", nargs="?", type=int, help="1-based page number to crop and run through the VLM.")
    parser.add_argument("form_type_pos", nargs="?", help="Form type for the page pair, e.g. 6pre, 8post, hpre.")
    parser.add_argument("roi_pos", nargs="?", help="ROI name from schema, explicit x,y,w,h, or 'page' for the whole page.")
    parser.add_argument("prompt_pos", nargs="?", help="Custom VLM prompt text, or a path to a prompt file.")
    parser.add_argument("--pdf", "--pdf-path", dest="pdf_path", default=None, help="Input PDF path.")
    parser.add_argument("--page", type=int, default=None, help="1-based page number to crop and run through the VLM.")
    parser.add_argument("--form-type", default=None, help="Form type for the page pair, e.g. 6pre, 8post, hpre.")
    parser.add_argument("--roi", default=None, help="ROI name from schema, explicit x,y,w,h, or 'page' for the whole page.")
    parser.add_argument("--prompt", default=None, help="Custom VLM prompt text.")
    parser.add_argument("--prompt-file", default=None, help="Path containing custom VLM prompt text.")
    parser.add_argument("--schema-path", default=None, help="Explicit schema path for named --roi lookup.")
    parser.add_argument(
        "--normalization-mode",
        default="full",
        choices=["full", "geometric", "postprocess"],
        help="Normalization mode passed to image_normalize.normalize_image.",
    )
    parser.add_argument("--no-binarize", action="store_true", help="Disable binarization during normalization.")
    parser.add_argument(
        "--remove-horizontal-lines",
        action="store_true",
        help="Apply conservative horizontal-line removal to the saved crop before VLM.",
    )
    parser.add_argument(
        "--expand-px",
        type=int,
        default=0,
        help="Expand the target ROI crop by N pixels in all directions after schema scaling.",
    )
    parser.add_argument(
        "--horizontal-line-min-len",
        type=int,
        default=80,
        help="Minimum horizontal line length, in crop pixels, for --remove-horizontal-lines.",
    )
    parser.add_argument(
        "--horizontal-line-thickness",
        type=int,
        default=2,
        help="Mask dilation thickness for --remove-horizontal-lines.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(PATHS["output"]) / "debug_images" / "vlm_roi_tester",
        help="Directory where the crop image is saved.",
    )
    parser.add_argument("--dpi", type=int, default=int(PDF_TO_IMAGES.get("dpi", 150)), help="PDF render DPI.")
    parser.add_argument("--fmt", default=str(PDF_TO_IMAGES.get("fmt", "png")), help="PDF render image format.")
    parser.add_argument("--host", default=str(_cfg().get("host") or LLM.get("host", "127.0.0.1")))
    parser.add_argument("--port", type=int, default=int(_cfg().get("port") or LLM.get("port", 11434)))
    parser.add_argument("--model", default=str(_cfg().get("model") or "qwen2.5vl"))
    parser.add_argument("--keep-alive", type=int, default=int(_cfg().get("keep_alive") or LLM.get("keep_alive", 0)))
    parser.add_argument("--timeout", type=int, default=_cfg_int("call_timeout_sec", 90))
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=_cfg_int("jpeg_quality", 60))
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=_cfg_bool("stream", True))
    parser.add_argument("--payload-json", action="store_true", help="Print payload metadata, excluding base64 image data.")
    args = parser.parse_args()

    args.prompt_from_pos = args.prompt is None and args.prompt_pos is not None
    args.pdf_path = args.pdf_path or args.pdf_path_pos
    args.page = args.page if args.page is not None else args.page_pos
    args.form_type = args.form_type or args.form_type_pos
    args.roi = args.roi or args.roi_pos
    args.prompt = args.prompt if args.prompt is not None else args.prompt_pos

    missing = []
    if not args.pdf_path:
        missing.append("pdf")
    if args.page is None:
        missing.append("page")
    if not args.form_type:
        missing.append("form_type")
    if not args.roi:
        missing.append("roi")
    if not args.prompt and not args.prompt_file:
        missing.append("prompt")
    if missing:
        parser.error(
            "missing required argument(s): "
            + ", ".join(missing)
            + "\npositional usage: test_vlm_roi_prompt.py PDF PAGE FORM_TYPE ROI PROMPT"
        )
    return args


def main() -> None:
    args = _parse_args()
    pdf_path = Path(args.pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    prompt = _read_prompt(args)
    target_side = _page_side(args.page)
    pair_pages = _pair_pages(args.page)
    max_pair_page = max(pair_pages)

    _status(
        f"start pdf={pdf_path} page={args.page} side={target_side} pair_pages={pair_pages} form_type={args.form_type}"
    )

    _status(f"rendering PDF pages through page {max_pair_page} at {args.dpi} dpi")
    rendered = pdf_to_images(
        pdf_path,
        dpi=int(args.dpi),
        fmt=str(args.fmt),
        use_tqdm=False,
        max_pages=max_pair_page,
    )
    if len(rendered) < max_pair_page:
        raise ValueError(f"PDF has only {len(rendered)} rendered page(s); requested pair needs page {max_pair_page}")
    pair_rendered = {page: rendered[page - 1] for page in pair_pages}
    _status("PDF render complete")

    _status("normalizing page pair with form-type templates")
    normalized: dict[int, Path] = {}
    cache_subdir = re.sub(r"[^\w\-.]", "_", pdf_path.stem).strip("_") or None
    for page in pair_pages:
        side = _page_side(page)
        normalized[page] = normalize_image(
            pair_rendered[page],
            binarize=not bool(args.no_binarize),
            form_type=str(args.form_type),
            page_side=side,
            mode=str(args.normalization_mode),
            cache_subdir=cache_subdir,
        )
        _status(f"normalized page {page} side={side} -> {normalized[page]}")

    _status("resolving target ROI")
    target_page = normalized[int(args.page)]
    with Image.open(target_page) as img:
        page_size = img.size
        roi_token = str(args.roi or "").strip().lower()
        if roi_token in {"page", "full-page", "full_page", "entire-page", "entire_page"}:
            raw_xywh = (0.0, 0.0, float(page_size[0]), float(page_size[1]))
            schema = None
            schema_path = None
            roi_source = "entire_page"
            scaled_box_raw = (0, 0, page_size[0], page_size[1])
        else:
            raw_xywh, schema, schema_path, roi_source = _parse_roi(
                roi_arg=args.roi,
                form_type=str(args.form_type),
                side=target_side,
                schema_path_arg=args.schema_path,
            )
            scaled_box_raw = _scale_xywh(raw_xywh, image_size=page_size, schema=schema)
        scaled_box = _expand_box(
            scaled_box_raw,
            image_size=page_size,
            expand_px=int(args.expand_px),
        )
    _status(
        f"ROI ready source={roi_source} raw_xywh={raw_xywh} "
        f"scaled_box={scaled_box_raw} expanded_box={scaled_box} "
        f"expand_px={int(args.expand_px)} normalized_page_size={page_size}"
    )
    if schema_path:
        _status(f"schema={schema_path}")

    _status("cropping ROI and saving PNG")
    crop_path, normalized_size, crop_size = _save_crop(
        normalized_page=target_page,
        scaled_box=scaled_box,
        out_dir=args.output_dir.expanduser().resolve(),
        pdf_path=pdf_path,
        page=int(args.page),
        form_type=str(args.form_type),
        roi_label=str(args.roi),
    )
    _status(f"crop saved -> {crop_path} size={crop_size}")
    horizontal_line_cleanup: dict[str, Any] = {"enabled": False}
    if args.remove_horizontal_lines:
        _status("removing horizontal lines from crop")
        horizontal_line_cleanup = _remove_horizontal_lines_from_file(
            crop_path,
            min_line_len=int(args.horizontal_line_min_len),
            thickness=int(args.horizontal_line_thickness),
        )
        _status(
            "horizontal line cleanup complete "
            f"removed_pixels={horizontal_line_cleanup.get('removed_pixels')} "
            f"mask={horizontal_line_cleanup.get('mask_path')}"
        )

    _status("encoding crop for VLM")
    image_b64, encode_meta = _encode_image_for_vlm(crop_path, int(args.jpeg_quality))
    _status(
        "crop encoded "
        f"jpeg_quality={encode_meta['jpeg_quality']} subsampling={encode_meta['jpeg_subsampling']} bytes={encode_meta['jpeg_bytes']}"
    )

    payload = _build_payload(image_b64, prompt, args)
    if args.payload_json:
        preview = dict(payload)
        preview["images"] = [f"<base64 {len(image_b64)} chars>"]
        print(json.dumps(preview, indent=2, sort_keys=True), flush=True)

    response_text, response_body = _call_vlm(payload, args)
    _status(f"VLM complete elapsed_sec={response_body.get('elapsed_sec')}")

    print("\n--- VLM OUTPUT START ---", flush=True)
    print(response_text, flush=True)
    print("--- VLM OUTPUT END ---\n", flush=True)

    print("--- RUN METADATA START ---", flush=True)
    metadata = {
        "pdf_path": str(pdf_path),
        "page": int(args.page),
        "pair_pages": pair_pages,
        "form_type": str(args.form_type),
        "side": target_side,
        "rendered_pages": {str(k): str(v) for k, v in pair_rendered.items()},
        "normalized_pages": {str(k): str(v) for k, v in normalized.items()},
        "normalization_mode": str(args.normalization_mode),
        "binarize": not bool(args.no_binarize),
        "roi": {
            "input": str(args.roi),
            "source": roi_source,
            "schema_path": str(schema_path) if schema_path else None,
            "raw_xywh": raw_xywh,
            "scaled_box_xyxy": scaled_box_raw,
            "expanded_box_xyxy": scaled_box,
            "expand_px": int(args.expand_px),
            "normalized_page_size": normalized_size,
            "crop_size": crop_size,
            "crop_path": str(crop_path),
            "horizontal_line_cleanup": horizontal_line_cleanup,
        },
        "vlm": {
            "host": str(args.host),
            "port": int(args.port),
            "model": str(args.model),
            "stream": bool(args.stream),
            "prompt_chars": len(prompt),
            "image_payload": encode_meta,
            "response_body": response_body,
        },
    }
    print(json.dumps(metadata, indent=2, sort_keys=True), flush=True)
    print("--- RUN METADATA END ---", flush=True)


if __name__ == "__main__":
    main()
