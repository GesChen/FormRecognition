"""
Live VLM stream viewer for one OCR ROI crop.

This is a debugging utility for Ollama /api/generate streams. It recreates the
vision call used by ocr_engine for a single cropped ROI and prints each chunk's
"response" value as soon as it arrives, without adding line breaks between
chunks.

Default replay target:
    8_Romsak_Pre_4, page 13, form 8pre side a, ROI id

Usage:
    python3 -u testing/test_vlm_stream_viewer.py
    python3 -u testing/test_vlm_stream_viewer.py --page-image output/cache/normalized/8_Romsak_Pre_4/page_0013_homography.png --form-type 8pre --side a --roi id
    python3 -u testing/test_vlm_stream_viewer.py --ndjson
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

from config import LLM, OCR_ENGINE, PDF_RECOGNITION
from ocr_engine import (
    _completed_json_from_stream as _engine_completed_json_from_stream,
    _stream_stop_completion as _engine_stream_stop_completion,
)


DEFAULT_PAGE_IMAGE = (
    ROOT
    / "output"
    / "cache"
    / "normalized"
    / "8_Romsak_Pre_4"
    / "page_0013_homography.png"
)
DEFAULT_FORM_TYPE = "8pre"
DEFAULT_SIDE = "a"
DEFAULT_ROI = "id"

TEMP_TEXT_VLM_PROMPT = """Text recognition:
```json
{
"text":""
}
```"""


def _cfg() -> dict[str, Any]:
    return OCR_ENGINE if isinstance(OCR_ENGINE, dict) else {}


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_cfg().get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _schema_path(form_type: str, side: str) -> Path:
    schema_dir = Path(
        PDF_RECOGNITION.get("schema_dir", ROOT / "data" / "roi_schemas")
    ).resolve()
    return schema_dir / f"{form_type}_{side}.json"


def _load_roi_box(schema_path: Path, roi_name: str) -> tuple[int, int, int, int, dict[str, Any]]:
    data = json.loads(schema_path.read_text())
    for roi in data.get("rois") or []:
        if str(roi.get("name", "")).strip() != roi_name:
            continue
        x = int(round(float(roi.get("x", 0))))
        y = int(round(float(roi.get("y", 0))))
        w = int(round(float(roi.get("w", 0))))
        h = int(round(float(roi.get("h", 0))))
        return x, y, x + w, y + h, data
    raise KeyError(f"ROI {roi_name!r} not found in {schema_path}")


def _scaled_box(
    box: tuple[int, int, int, int],
    *,
    image_size: tuple[int, int],
    schema: dict[str, Any],
) -> tuple[int, int, int, int]:
    image_w, image_h = image_size
    ref_w = int(schema.get("image_width") or 0)
    ref_h = int(schema.get("image_height") or 0)
    if ref_w > 0 and ref_h > 0 and (ref_w, ref_h) != (image_w, image_h):
        sx = image_w / ref_w
        sy = image_h / ref_h
    else:
        sx = sy = 1.0
    x1, y1, x2, y2 = box
    return (
        max(int(round(x1 * sx)), 0),
        max(int(round(y1 * sy)), 0),
        min(int(round(x2 * sx)), image_w),
        min(int(round(y2 * sy)), image_h),
    )


def _encode_crop(
    page_image: Path,
    box: tuple[int, int, int, int],
    *,
    jpeg_quality: int,
    write_crop: Path | None,
) -> tuple[str, bytes, tuple[int, int], tuple[int, int]]:
    img = Image.open(page_image).convert("L")
    crop = img.crop(box)
    if write_crop is not None:
        write_crop.parent.mkdir(parents=True, exist_ok=True)
        crop.save(write_crop)

    quality = max(1, min(100, int(jpeg_quality)))
    subsampling = 0 if quality >= 95 else 2
    buf = BytesIO()
    crop.save(
        buf,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=False,
        subsampling=subsampling,
    )
    jpeg_bytes = buf.getvalue()
    return (
        base64.b64encode(jpeg_bytes).decode("ascii"),
        jpeg_bytes,
        img.size,
        crop.size,
    )


def _build_payload(image_b64: str, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "stream": True,
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


def _stream_text_json_completion(text: str) -> dict[str, int | str] | None:
    """
    Match ocr_engine's early-stop detector for a complete {"text": ...} JSON block.
    """
    s = str(text or "")
    if not s:
        return None

    fence = re.search(r"```(?:json)?\s*", s, flags=re.IGNORECASE)
    if fence:
        close = s.find("```", fence.end())
        if close >= 0:
            block = s[fence.end() : close].strip()
            try:
                obj = json.loads(block)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and "text" in obj:
                return {"mode": "fenced_json", "end_index": close + 3}

    start = s.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(s[start:])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "text" in obj:
        return {"mode": "raw_json", "end_index": start + end}
    return None


def _completed_json_from_stream(
    text: str,
    json_completion: dict[str, int | str] | None,
) -> str:
    s = str(text or "")
    if isinstance(json_completion, dict):
        try:
            end_index = int(json_completion.get("end_index", 0))
        except (TypeError, ValueError):
            end_index = 0
        if end_index > 0:
            return s[:end_index]
    return s


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print a live Ollama VLM stream for one OCR ROI crop.",
    )
    p.add_argument("--page-image", default=str(DEFAULT_PAGE_IMAGE), help="Normalized page image to crop.")
    p.add_argument("--form-type", default=DEFAULT_FORM_TYPE, help="Form type for schema lookup.")
    p.add_argument("--side", default=DEFAULT_SIDE, choices=["a", "b"], help="Schema side.")
    p.add_argument("--roi", default=DEFAULT_ROI, help="ROI name in the schema.")
    p.add_argument("--schema-path", default=None, help="Explicit ROI schema path.")
    p.add_argument("--host", default=str(_cfg().get("host") or LLM.get("host", "127.0.0.1")))
    p.add_argument("--port", type=int, default=int(_cfg().get("port") or LLM.get("port", 11434)))
    p.add_argument("--model", default=str(_cfg().get("model") or "qwen2.5vl"))
    p.add_argument("--keep-alive", type=int, default=int(_cfg().get("keep_alive") or LLM.get("keep_alive", 0)))
    p.add_argument("--timeout", type=int, default=_cfg_int("call_timeout_sec", 90))
    p.add_argument("--connect-timeout", type=int, default=10)
    p.add_argument("--jpeg-quality", type=int, default=_cfg_int("jpeg_quality", 60))
    p.add_argument("--temperature", type=float, default=None, help="Override options.temperature.")
    p.add_argument("--prompt", default=TEMP_TEXT_VLM_PROMPT, help="Prompt sent to /api/generate.")
    p.add_argument("--write-crop", default=None, help="Optional path to save the exact PNG crop.")
    p.add_argument("--ndjson", action="store_true", help="Print raw NDJSON stream lines instead of response deltas.")
    p.add_argument(
        "--no-stop-after-json",
        action="store_true",
        help="Keep reading until the server sends done instead of stopping after one JSON block.",
    )
    p.add_argument(
        "--raw-delta-only",
        action="store_true",
        help="Deprecated alias for the default behavior.",
    )
    p.add_argument("--payload-json", action="store_true", help="Print payload metadata, excluding image data.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    page_image = Path(args.page_image).expanduser().resolve()
    if not page_image.exists():
        raise FileNotFoundError(f"Page image not found: {page_image}")

    schema_path = Path(args.schema_path).expanduser().resolve() if args.schema_path else _schema_path(args.form_type, args.side)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    raw_box, schema = None, None
    x1, y1, x2, y2, schema = _load_roi_box(schema_path, args.roi)
    raw_box = (x1, y1, x2, y2)
    with Image.open(page_image) as img:
        scaled_box = _scaled_box(raw_box, image_size=img.size, schema=schema)

    write_crop = Path(args.write_crop).expanduser().resolve() if args.write_crop else None
    image_b64, jpeg_bytes, page_size, crop_size = _encode_crop(
        page_image,
        scaled_box,
        jpeg_quality=args.jpeg_quality,
        write_crop=write_crop,
    )
    payload = _build_payload(image_b64, args)

    url = f"http://{args.host}:{int(args.port)}/api/generate"
    print(f"POST {url}", flush=True)
    print(f"page_image={page_image}", flush=True)
    print(f"schema_path={schema_path}", flush=True)
    print(f"roi={args.roi} raw_box={raw_box} scaled_box={scaled_box}", flush=True)
    print(f"page_size={page_size} crop_size={crop_size} jpeg_bytes={len(jpeg_bytes)}", flush=True)
    if write_crop:
        print(f"crop_written={write_crop}", flush=True)
    if args.payload_json:
        payload_preview = dict(payload)
        payload_preview["images"] = [f"<base64 {len(image_b64)} chars>"]
        print(json.dumps(payload_preview, indent=2, sort_keys=True), flush=True)
    if args.ndjson:
        print("--- live VLM stream (raw NDJSON) ---", flush=True)
    else:
        print("--- live VLM response stream ---", flush=True)

    started = time.perf_counter()
    assembled: list[str] = []
    json_completion: dict[str, int | str] | None = None
    with requests.post(
        url,
        json=payload,
        timeout=(int(args.connect_timeout), int(args.timeout)),
        stream=True,
    ) as resp:
        print(f"HTTP {resp.status_code}", flush=True)
        resp.raise_for_status()
        for raw_line in resp.iter_lines(chunk_size=1, decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if args.ndjson:
                    print(line, flush=True)
                continue
            delta = str(obj.get("response", "") or "")
            if delta:
                assembled.append(delta)
            if args.ndjson:
                print(line, flush=True)
            elif delta:
                print(delta, end="", flush=True)
            if delta and not args.no_stop_after_json:
                json_completion = _engine_stream_stop_completion("".join(assembled))
                if json_completion is not None:
                    if args.ndjson:
                        print(f"--- stopped_for_json_completion={json_completion} ---", flush=True)
                    break
            if obj.get("done"):
                break

    if args.ndjson:
        print("\n--- assembled response ---", flush=True)
        print(_engine_completed_json_from_stream("".join(assembled), json_completion), flush=True)
    else:
        print("", flush=True)
    if json_completion is not None:
        print(f"stopped_for_json_completion={json_completion}", flush=True)
    print(f"elapsed_sec={time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()
