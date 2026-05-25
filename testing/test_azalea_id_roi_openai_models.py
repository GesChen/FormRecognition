"""
Benchmark ID ROI extraction on 20 normalized Azalea cache images using OpenAI vision models.

This script:
1) Loads ID ROI geometry from `data/roi_schemas/6pre_a.json`.
2) Selects 20 normalized images from Azalea cache (default: *_homography.png).
3) Crops the `id` ROI from each image.
4) Sends each crop + English ID prompt to each model:
   - gpt-5.3-chat-latest
   - gpt-4.1-mini
   - gpt-4.1-nano
5) Writes a JSON report with raw responses and parsed IDs.

It does not run automatically; invoke it manually.

Usage (from project root):
  python3 testing/test_azalea_id_roi_openai_models.py
  python3 testing/test_azalea_id_roi_openai_models.py --count 20 --save-crops
  python3 testing/test_azalea_id_roi_openai_models.py --pattern "*_bin.png"
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from config import PATHS

DEFAULT_MODELS = [
    "gpt-5.3-chat-latest",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Roi:
    name: str
    x: float
    y: float
    w: float
    h: float


def _load_id_roi(schema_path: Path) -> tuple[Roi, int | None, int | None, str | None]:
    raw = _read_json(schema_path)
    rois = raw.get("rois", [])
    for r in rois:
        if str(r.get("name", "")).strip().lower() == "id":
            roi = Roi(
                name="id",
                x=float(r.get("x", 0.0)),
                y=float(r.get("y", 0.0)),
                w=float(r.get("w", 0.0)),
                h=float(r.get("h", 0.0)),
            )
            ref_w = raw.get("image_width")
            ref_h = raw.get("image_height")
            ocr_prompt = r.get("ocr_prompt_override")
            return (
                roi,
                int(ref_w) if ref_w is not None else None,
                int(ref_h) if ref_h is not None else None,
                str(ocr_prompt) if ocr_prompt is not None else None,
            )
    raise ValueError(f"ROI named 'id' not found in schema: {schema_path}")


def _prompt_from_schema_or_fallback(schema_prompt: str | None) -> str:
    # Accuracy-focused prompt for ID extraction from this ROI.
    # Keep output machine-parseable and normalized.
    _ = schema_prompt  # keep signature stable
    return (
        "Read the handwritten ID from this image crop.\n"
        "Return exactly one JSON object with this schema:\n"
        '{"id":"<value or null>"}\n'
        "\n"
        "Rules:\n"
        "1) Output ONLY JSON. No markdown, no code fences, no extra text.\n"
        "2) If an ID is visible, return only the ID value (no 'ID:' prefix).\n"
        "3) Remove spaces and punctuation from the ID value.\n"
        "4) Use uppercase letters.\n"
        "5) If unreadable/uncertain, return {\"id\":null}.\n"
    )


def _pick_images(cache_dir: Path, pattern: str, count: int) -> list[Path]:
    files = sorted(cache_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No images matched pattern {pattern!r} in {cache_dir}")
    odd_files: list[Path] = []
    for p in files:
        m = re.search(r"page_(\d+)", p.stem)
        if not m:
            continue
        page_num = int(m.group(1))
        if page_num % 2 == 1:
            odd_files.append(p)
    if not odd_files:
        raise FileNotFoundError(
            f"No odd-numbered page images matched pattern {pattern!r} in {cache_dir}"
        )
    return odd_files[:count]


def _scale_roi(roi: Roi, img_w: int, img_h: int, ref_w: int | None, ref_h: int | None) -> tuple[int, int, int, int]:
    if ref_w and ref_h and ref_w > 0 and ref_h > 0 and (img_w != ref_w or img_h != ref_h):
        sx = img_w / ref_w
        sy = img_h / ref_h
    else:
        sx = 1.0
        sy = 1.0

    x1 = max(0, int(round(roi.x * sx)))
    y1 = max(0, int(round(roi.y * sy)))
    x2 = min(img_w, int(round((roi.x + roi.w) * sx)))
    y2 = min(img_h, int(round((roi.y + roi.h) * sy)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid ROI crop bounds after scaling: {(x1, y1, x2, y2)} "
            f"for image size {(img_w, img_h)}"
        )
    return x1, y1, x2, y2


def _crop_id_roi(
    image_path: Path,
    roi: Roi,
    ref_w: int | None,
    ref_h: int | None,
) -> tuple[bytes, dict[str, Any]]:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    x1, y1, x2, y2 = _scale_roi(roi, w, h, ref_w, ref_h)
    crop = img[y1:y2, x1:x2]
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise RuntimeError(f"Failed to encode crop PNG for {image_path}")

    meta = {
        "source_image": str(image_path),
        "source_width": w,
        "source_height": h,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "crop_width": x2 - x1,
        "crop_height": y2 - y1,
    }
    return bytes(encoded), meta


def _extract_output_text(resp_obj: dict[str, Any]) -> str:
    direct = resp_obj.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    # Fallback parse of structured output blocks
    out = resp_obj.get("output")
    if not isinstance(out, list):
        return ""

    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict):
                continue
            if c.get("type") in {"output_text", "text"}:
                txt = c.get("text")
                if isinstance(txt, str) and txt:
                    parts.append(txt)
    return "\n".join(parts).strip()


def _extract_id_from_text(text: str) -> str | None:
    if not text:
        return None
    s = str(text).strip()

    # First try code-fenced JSON if present.
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    candidates = [s]
    if m:
        candidates.insert(0, m.group(1).strip())

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                val = obj.get("id")
                if val is None:
                    return None
                cleaned = str(val).strip()
                if not cleaned:
                    return None
                cleaned = cleaned.upper()
                cleaned = re.sub(r"^\s*ID\s*:\s*", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
                return cleaned or None
        except json.JSONDecodeError:
            # Try decoding first JSON object inside noisy text.
            try:
                dec = json.JSONDecoder()
                i0 = cand.find("{")
                if i0 >= 0:
                    obj, _ = dec.raw_decode(cand[i0:])
                    if isinstance(obj, dict):
                        val = obj.get("id")
                        if val is None:
                            return None
                        cleaned = str(val).strip()
                        if not cleaned:
                            return None
                        cleaned = cleaned.upper()
                        cleaned = re.sub(r"^\s*ID\s*:\s*", "", cleaned, flags=re.IGNORECASE)
                        cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
                        return cleaned or None
            except Exception:
                continue
    return None


def _call_openai_responses(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_png: bytes,
    timeout_sec: int,
) -> tuple[dict[str, Any], float]:
    image_b64 = base64.b64encode(image_png).decode("ascii")
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_b64}",
                        "detail": "high",
                    },
                ],
            }
        ],
    }

    data = json.dumps(body).encode("utf-8")
    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    t0 = time.perf_counter()
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            payload = resp.read().decode("utf-8")
            out = json.loads(payload)
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"OpenAI HTTPError {e.code}: {raw}") from e
    except error.URLError as e:
        raise RuntimeError(f"OpenAI URLError: {e}") from e
    elapsed = time.perf_counter() - t0

    return out, elapsed


def _write_crop(path: Path, image_png: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(image_png)


def _call_openai_with_retries(
    *,
    api_key: str,
    model: str,
    prompt: str,
    image_png: bytes,
    timeout_sec: int,
    max_retries: int,
) -> tuple[dict[str, Any], float]:
    attempts = max(1, int(max_retries) + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return _call_openai_responses(
                api_key=api_key,
                model=model,
                prompt=prompt,
                image_png=image_png,
                timeout_sec=timeout_sec,
            )
        except Exception as e:
            last_error = e
            if attempt < attempts:
                time.sleep(min(5.0, 0.5 * attempt))
                continue
            break
    raise RuntimeError(f"Failed after {attempts} attempts: {last_error}") from last_error


def _write_report_checkpoint(output_json: Path, report: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_json.with_suffix(output_json.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(output_json)


def _preflight_checks(
    *,
    api_key: str,
    output_json: Path,
    images: list[Path],
    roi: Roi,
    ref_w: int | None,
    ref_h: int | None,
    prompt_en: str,
    models: list[str],
    timeout_sec: int,
    max_retries: int,
) -> None:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    # Validate output path is writable.
    output_json.parent.mkdir(parents=True, exist_ok=True)
    probe = output_json.parent / ".write_probe.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink(missing_ok=True)

    if not images:
        raise RuntimeError("No input images selected.")

    # Validate ROI crop is feasible before long-running calls.
    first_png, first_meta = _crop_id_roi(images[0], roi, ref_w, ref_h)
    if not first_png:
        raise RuntimeError("Preflight crop failed: empty image payload.")
    print(
        "Preflight crop OK:",
        Path(first_meta["source_image"]).name,
        f"{first_meta['crop_width']}x{first_meta['crop_height']}",
    )

    # Cheap network sanity check.
    try:
        socket.gethostbyname("api.openai.com")
    except OSError as e:
        raise RuntimeError(f"Cannot resolve api.openai.com: {e}") from e

    # Validate auth/model access once per model before full run.
    for model in models:
        resp_obj, elapsed = _call_openai_with_retries(
            api_key=api_key,
            model=model,
            prompt=prompt_en,
            image_png=first_png,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
        )
        txt = _extract_output_text(resp_obj)
        print(f"Preflight API OK: {model} ({elapsed:.2f}s, text_len={len(txt)})")


def main() -> None:
    p = argparse.ArgumentParser(description="ID ROI benchmark on Azalea normalized cache with OpenAI vision models.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(PATHS["cache_normalized"]) / "azalea_ms_6_earle_pre",
        help="Directory containing normalized Azalea page images.",
    )
    p.add_argument(
        "--pattern",
        default="*_homography.png",
        help="Glob pattern used to pick normalized images from --cache-dir.",
    )
    p.add_argument("--count", type=int, default=20, help="Number of images to test (default: 20).")
    p.add_argument(
        "--schema-path",
        type=Path,
        default=ROOT / "data" / "roi_schemas" / "6pre_a.json",
        help="ROI schema path containing the id ROI.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model list to test.",
    )
    p.add_argument(
        "--timeout-sec",
        type=int,
        default=90,
        help="Timeout per API call.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per API request after the initial attempt.",
    )
    p.add_argument(
        "--save-crops",
        action="store_true",
        help="Save cropped ID PNGs under output/debug_images/id_crop_openai_bench/.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "output" / "debug_images" / "id_crop_openai_bench" / "azalea_id_roi_openai_bench.json",
        help="Path to write result report JSON.",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run checks (including one API call/model) and exit without full benchmark.",
    )
    args = p.parse_args()

    api_key = str(os.environ.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required via environment variable.")

    cache_dir = args.cache_dir.expanduser().resolve()
    if not cache_dir.exists():
        raise SystemExit(f"Cache dir not found: {cache_dir}")

    schema_path = args.schema_path.expanduser().resolve()
    if not schema_path.exists():
        raise SystemExit(f"Schema not found: {schema_path}")

    roi, ref_w, ref_h, schema_prompt = _load_id_roi(schema_path)
    prompt_en = _prompt_from_schema_or_fallback(schema_prompt)

    images = _pick_images(cache_dir, args.pattern, max(1, int(args.count)))
    models = [str(m).strip() for m in args.models if str(m).strip()]
    if not models:
        raise SystemExit("At least one model is required.")

    print(f"Selected {len(images)} image(s) from {cache_dir} with pattern {args.pattern!r}")
    print(f"Models: {', '.join(models)}")
    print(f"ROI: id x={roi.x} y={roi.y} w={roi.w} h={roi.h} (schema ref={ref_w}x{ref_h})")
    print(f"Output report: {args.output_json.expanduser().resolve()}")

    report: dict[str, Any] = {
        "created_at_utc": _utc_now_iso(),
        "cache_dir": str(cache_dir),
        "pattern": args.pattern,
        "image_count": len(images),
        "schema_path": str(schema_path),
        "roi": {
            "name": roi.name,
            "x": roi.x,
            "y": roi.y,
            "w": roi.w,
            "h": roi.h,
            "schema_image_width": ref_w,
            "schema_image_height": ref_h,
        },
        "models": models,
        "prompt_english": prompt_en,
        "preflight_completed": False,
        "results": [],
    }

    crops_dir = ROOT / "output" / "debug_images" / "id_crop_openai_bench"

    _preflight_checks(
        api_key=api_key,
        output_json=args.output_json.expanduser().resolve(),
        images=images,
        roi=roi,
        ref_w=ref_w,
        ref_h=ref_h,
        prompt_en=prompt_en,
        models=models,
        timeout_sec=int(args.timeout_sec),
        max_retries=int(args.max_retries),
    )
    report["preflight_completed"] = True
    _write_report_checkpoint(args.output_json.expanduser().resolve(), report)
    if args.preflight_only:
        print("Preflight-only requested; exiting before benchmark run.")
        return

    try:
        for idx, img_path in enumerate(images, start=1):
            image_entry: dict[str, Any] = {
                "index": idx,
                "image_path": str(img_path),
                "crop": None,
                "model_runs": [],
            }

            try:
                crop_png, crop_meta = _crop_id_roi(img_path, roi, ref_w, ref_h)
                image_entry["crop"] = crop_meta
                if args.save_crops:
                    crop_name = f"{img_path.stem}_id_crop.png"
                    _write_crop(crops_dir / crop_name, crop_png)
                    crop_meta["saved_crop_path"] = str((crops_dir / crop_name).resolve())
            except Exception as e:
                image_entry["error"] = f"crop_failed: {e}"
                report["results"].append(image_entry)
                _write_report_checkpoint(args.output_json.expanduser().resolve(), report)
                print(f"[{idx}/{len(images)}] {img_path.name} crop failed")
                continue

            for model in models:
                run: dict[str, Any] = {
                    "model": model,
                    "ok": False,
                    "elapsed_sec": None,
                    "id": None,
                    "response_text": "",
                    "usage": None,
                    "error": None,
                }
                try:
                    resp_obj, elapsed = _call_openai_with_retries(
                        api_key=api_key,
                        model=model,
                        prompt=prompt_en,
                        image_png=crop_png,
                        timeout_sec=int(args.timeout_sec),
                        max_retries=int(args.max_retries),
                    )
                    text = _extract_output_text(resp_obj)
                    run["ok"] = True
                    run["elapsed_sec"] = round(elapsed, 4)
                    run["response_text"] = text
                    run["id"] = _extract_id_from_text(text)
                    run["usage"] = resp_obj.get("usage")
                except Exception as e:
                    run["error"] = str(e)

                image_entry["model_runs"].append(run)

            report["results"].append(image_entry)
            _write_report_checkpoint(args.output_json.expanduser().resolve(), report)
            print(f"[{idx}/{len(images)}] {img_path.name} done")
    except KeyboardInterrupt:
        _write_report_checkpoint(args.output_json.expanduser().resolve(), report)
        print("Interrupted. Partial results checkpoint saved.")
        raise SystemExit(130)

    _write_report_checkpoint(args.output_json.expanduser().resolve(), report)
    print(f"Wrote report: {args.output_json.expanduser().resolve()}")


if __name__ == "__main__":
    main()
