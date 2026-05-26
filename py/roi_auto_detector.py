"""
ROI auto-detector for template authoring in ROI editor.

Flow:
1) Run PaddleOCR on the page image and collect text boxes.
2) Provide document structure + a reference schema example to OpenAI.
3) Require strict JSON-schema output for ROI template structure.
4) Validate output geometry before returning.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai_client import call_json, resolve_model

try:
    from config import PATHS
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parent.parent
    PATHS = {"data": ROOT / "data"}


ROI_MIN_SIZE = 5


def _schema_dir() -> Path:
    return Path(PATHS.get("roi_schemas_root", Path(PATHS["data"]) / "roi_schemas")).resolve()


def _templates_dir() -> Path:
    return Path(PATHS.get("templates_root", Path(PATHS["data"]) / "templates")).resolve()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _load_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as im:
        w, h = im.size
    return int(w), int(h)


def _box_from_xyxy(raw: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    x1 = _coerce_int(raw[0])
    y1 = _coerce_int(raw[1])
    x2 = _coerce_int(raw[2])
    y2 = _coerce_int(raw[3])
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _box_from_poly(raw: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for pt in raw:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        xs.append(_safe_float(pt[0]))
        ys.append(_safe_float(pt[1]))
    if not xs or not ys:
        return None
    x_min = int(round(min(xs)))
    y_min = int(round(min(ys)))
    x_max = int(round(max(xs)))
    y_max = int(round(max(ys)))
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return None
    return x_min, y_min, w, h


def _collect_document_structure(image_path: Path) -> dict[str, Any]:
    from ocr_engine import _get_paddle_engine

    engine = _get_paddle_engine()
    outputs = engine.predict(input=str(image_path.resolve()))

    lines: list[dict[str, Any]] = []
    for res in outputs or []:
        j = getattr(res, "json", None)
        if not isinstance(j, dict):
            continue
        block = j.get("res")
        if not isinstance(block, dict):
            continue

        texts = block.get("rec_texts") or []
        scores = block.get("rec_scores") or []
        boxes = block.get("rec_boxes") or []
        polys = block.get("rec_polys") or []

        n = max(len(texts), len(boxes), len(polys))
        for i in range(n):
            text = str(texts[i] if i < len(texts) else "").strip()
            if not text:
                continue
            score = _safe_float(scores[i] if i < len(scores) else 0.0, 0.0)

            box = _box_from_xyxy(boxes[i]) if i < len(boxes) else None
            if box is None and i < len(polys):
                box = _box_from_poly(polys[i])
            if box is None:
                continue
            x, y, w, h = box
            lines.append(
                {
                    "text": text,
                    "score": round(score, 6),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                }
            )

    lines.sort(key=lambda r: (int(round(r["y"] / 8.0)), r["x"]))
    mean_score = round(sum(r["score"] for r in lines) / len(lines), 6) if lines else 0.0
    min_score = round(min((r["score"] for r in lines), default=0.0), 6)
    return {
        "line_count": len(lines),
        "min_score": min_score,
        "mean_score": mean_score,
        "lines": lines,
    }


def _load_nonempty_schema(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    rois = data.get("rois")
    if not isinstance(rois, list) or not rois:
        return None
    return data


def _pick_reference_schema(image_path: Path) -> tuple[Path, dict[str, Any]]:
    schema_dir = _schema_dir()
    templates_dir = _templates_dir()
    stem = image_path.stem.lower()

    exact: Path | None = None
    try:
        rel = image_path.resolve().relative_to(templates_dir)
        exact = schema_dir / rel.with_suffix(".json")
    except ValueError:
        exact = schema_dir / f"{image_path.stem}.json"
    exact_data = _load_nonempty_schema(exact)
    if exact_data is not None:
        return exact, exact_data

    form_keys = ["6pre", "6post", "7pre", "7post", "8pre", "8post", "hpre", "hpost"]
    inferred_form = next((k for k in form_keys if k in stem), None)
    inferred_side = "a" if stem.endswith("_a") else ("b" if stem.endswith("_b") else None)

    if inferred_form and inferred_side:
        p = schema_dir / f"{inferred_form}_{inferred_side}.json"
        data = _load_nonempty_schema(p)
        if data is not None:
            return p, data

    if inferred_side:
        side_candidates = sorted(schema_dir.rglob(f"*_{inferred_side}.json"))
        for p in side_candidates:
            data = _load_nonempty_schema(p)
            if data is not None:
                return p, data

    for p in sorted(schema_dir.rglob("*.json")):
        data = _load_nonempty_schema(p)
        if data is not None:
            return p, data

    raise RuntimeError("No reference ROI schema with ROIs found in data/roi_schemas.")


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rois": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 80},
                        "x": {"type": "integer", "minimum": 0},
                        "y": {"type": "integer", "minimum": 0},
                        "w": {"type": "integer", "minimum": ROI_MIN_SIZE},
                        "h": {"type": "integer", "minimum": ROI_MIN_SIZE},
                    },
                    "required": ["name", "x", "y", "w", "h"],
                },
            }
        },
        "required": ["rois"],
    }


def _validate_and_normalize_rois(raw_obj: dict[str, Any], image_w: int, image_h: int) -> list[dict[str, Any]]:
    rois_raw = raw_obj.get("rois")
    if not isinstance(rois_raw, list) or not rois_raw:
        raise ValueError("Generated template has no ROIs.")

    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for i, roi in enumerate(rois_raw):
        if not isinstance(roi, dict):
            raise ValueError(f"ROI #{i + 1} is not an object.")
        name = str(roi.get("name", "")).strip()
        if not name:
            raise ValueError(f"ROI #{i + 1} has empty name.")
        if name in names:
            raise ValueError(f"Duplicate ROI name: {name!r}")
        names.add(name)

        x = _coerce_int(roi.get("x"))
        y = _coerce_int(roi.get("y"))
        w = _coerce_int(roi.get("w"))
        h = _coerce_int(roi.get("h"))

        if x < 0 or y < 0 or w < ROI_MIN_SIZE or h < ROI_MIN_SIZE:
            raise ValueError(f"ROI {name!r} has invalid geometry.")
        if x + w > image_w or y + h > image_h:
            raise ValueError(
                f"ROI {name!r} is out of bounds for image size {image_w}x{image_h}."
            )

        out.append(
            {
                "id": "r" + uuid.uuid4().hex[:9],
                "name": name,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
            }
        )

    return out


def detect_rois_with_openai(
    image_path: Path,
    *,
    model_profile: str = "balanced",
) -> dict[str, Any]:
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_w, image_h = _load_image_size(image_path)
    doc_structure = _collect_document_structure(image_path)
    if not doc_structure.get("lines"):
        raise RuntimeError("PaddleOCR detected no text lines.")

    ref_path, ref_schema = _pick_reference_schema(image_path)
    ref_example = {
        "image_width": ref_schema.get("image_width"),
        "image_height": ref_schema.get("image_height"),
        "rois": ref_schema.get("rois", []),
    }

    instructions = (
        "You generate ROI templates for fixed-layout assessment forms.\n"
        "Return ONE JSON object only that matches the provided JSON schema exactly.\n"
        "Do not include markdown, prose, comments, or extra keys.\n"
        "Rules:\n"
        "1) Use the OCR line boxes and text as the source of layout truth.\n"
        "2) Follow the naming style and granularity pattern of the reference template.\n"
        "3) For MCQ blocks, include both question ROI (numeric name) and choice ROIs (e.g., 1a, 1b...).\n"
        "4) Coordinates must be integer pixels in the target image coordinate space.\n"
        "5) Every ROI must stay fully inside the image bounds.\n"
        "6) Do not hallucinate fields not supported by the document structure.\n"
    )

    payload = {
        "target_image": {
            "path": str(image_path),
            "width": image_w,
            "height": image_h,
        },
        "document_structure_from_paddleocr": doc_structure,
        "reference_template_example": ref_example,
    }

    raw_obj = call_json(
        json.dumps(payload, ensure_ascii=False),
        schema=_output_schema(),
        schema_name="roi_template",
        instructions=instructions,
        model_profile=model_profile,
        max_output_tokens=12000,
    )
    rois = _validate_and_normalize_rois(raw_obj, image_w, image_h)

    return {
        "rois": rois,
        "image_width": image_w,
        "image_height": image_h,
        "reference_schema_path": str(ref_path.resolve()),
        "ocr_line_count": int(doc_structure.get("line_count", 0)),
        "model": resolve_model(model_profile=model_profile),
    }
