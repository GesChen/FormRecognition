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
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai_client import call_json, resolve_model
from llm_client import generate

try:
    from config import PATHS, ROI_AUTO_DETECT
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parent.parent
    PATHS = {"data": ROOT / "data"}
    ROI_AUTO_DETECT = {"use_openai": True, "local_model": "qwen3.5:9b", "local_reasoning": True}


ROI_MIN_SIZE = 5


def _resolve_collection_root(path_value: Path | str, expected_leaf: str) -> Path:
    p = Path(path_value).resolve()
    if p.name == expected_leaf:
        return p
    if p.parent.name == expected_leaf:
        return p.parent
    return p


def _schema_dir() -> Path:
    data_root = Path(PATHS.get("data", Path(__file__).resolve().parent.parent / "data")).resolve()
    return _resolve_collection_root(
        PATHS.get("roi_schemas_root", data_root / "roi_schemas"),
        "roi_schemas",
    )


def _templates_dir() -> Path:
    data_root = Path(PATHS.get("data", Path(__file__).resolve().parent.parent / "data")).resolve()
    return _resolve_collection_root(
        PATHS.get("templates_root", data_root / "templates"),
        "templates",
    )


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


def _load_custom_reference_schema() -> tuple[str, dict[str, Any]]:
    schema_dir = _schema_dir()
    ref_a_path = schema_dir / "reference" / "6pre_a_vlm_reference.json"
    ref_b_path = schema_dir / "reference" / "6pre_b_vlm_reference.json"
    if not ref_a_path.exists():
        raise RuntimeError(f"Custom reference schema not found: {ref_a_path}")
    if not ref_b_path.exists():
        raise RuntimeError(f"Custom reference schema not found: {ref_b_path}")
    try:
        side_a = json.loads(ref_a_path.read_text(encoding="utf-8"))
        side_b = json.loads(ref_b_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read custom reference schema pair: {ref_a_path} | {ref_b_path}"
        ) from exc
    if not isinstance(side_a, dict):
        raise RuntimeError(f"Custom reference schema must be a JSON object: {ref_a_path}")
    if not isinstance(side_b, dict):
        raise RuntimeError(f"Custom reference schema must be a JSON object: {ref_b_path}")
    merged = {
        "form_type": side_a.get("form_type") or side_b.get("form_type") or "6pre",
        "side_a": {
            "image_width": side_a.get("image_width"),
            "image_height": side_a.get("image_height"),
            "rois": side_a.get("rois", []),
        },
        "side_b": {
            "image_width": side_b.get("image_width"),
            "image_height": side_b.get("image_height"),
            "rois": side_b.get("rois", []),
        },
    }
    ref_label = f"{ref_a_path.resolve()} + {ref_b_path.resolve()}"
    return ref_label, merged


def _load_custom_reference_paddle(side: str) -> dict[str, Any]:
    schema_dir = _schema_dir()
    side_key = "a" if str(side).strip().lower() == "a" else "b"
    path = schema_dir / "reference" / f"6pre_{side_key}_paddle_raw.json"
    if not path.exists():
        raise RuntimeError(f"Custom Paddle reference not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to read custom Paddle reference: {path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Custom Paddle reference must be a JSON object: {path}")
    return data


def _reference_rois_without_id(ref_example: dict[str, Any], side: str) -> list[dict[str, Any]]:
    side_key = "side_a" if str(side).strip().lower() == "a" else "side_b"
    side_obj = ref_example.get(side_key) or {}
    rois = side_obj.get("rois") or []
    out: list[dict[str, Any]] = []
    for roi in rois:
        if not isinstance(roi, dict):
            continue
        name = str(roi.get("name", "")).strip().lower()
        if name == "id":
            continue
        out.append(roi)
    return out


def _extract_json_object(text: str) -> dict[str, Any]:
    s = str(text or "").strip()
    if not s:
        raise RuntimeError("Local model returned empty output.")
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: extract first top-level JSON object block.
    start = s.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start : i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = s.find("{", start + 1)
    raise RuntimeError("Local model output did not contain a valid JSON object.")


def _call_local_json(
    *,
    payload: dict[str, Any],
    instructions: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    model = str(ROI_AUTO_DETECT.get("local_model", "qwen3.5:9b") or "qwen3.5:9b")
    reasoning = bool(ROI_AUTO_DETECT.get("local_reasoning", True))
    prompt = (
        f"{instructions}\n\n"
        "Strict output requirements:\n"
        "- Return exactly one JSON object.\n"
        "- No markdown, no code fences, no explanation.\n"
        "- JSON must match this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Input payload:\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    extra_params: dict[str, Any] = {
        "options": {"temperature": 0},
        "format": schema,
    }
    if reasoning:
        extra_params["think"] = True

    out = generate(
        prompt,
        model=model,
        timeout=300,
        extra_params=extra_params,
    )
    text = str(out.get("text", "") or "")
    elapsed = float(out.get("elapsed") or 0.0)
    raw = out.get("raw")

    if not text.strip():
        print("[roi_auto_detect][local] Empty model output.")
        print(f"[roi_auto_detect][local] elapsed_sec={elapsed:.3f}")
        print(f"[roi_auto_detect][local] raw_response={raw}")
        raise RuntimeError(f"Local model returned empty output (elapsed={elapsed:.2f}s).")

    try:
        return _extract_json_object(text)
    except Exception as exc:
        print("[roi_auto_detect][local] Failed to parse JSON output.")
        print(f"[roi_auto_detect][local] elapsed_sec={elapsed:.3f}")
        print(f"[roi_auto_detect][local] raw_response={raw}")
        print(f"[roi_auto_detect][local] text_output={text}")
        raise RuntimeError(
            f"Local model output parse failed (elapsed={elapsed:.2f}s): {exc}"
        ) from exc


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
                        "llm_prompt_override": {"type": ["string", "null"]},
                        "ocr_prompt_override": {"type": ["string", "null"]},
                    },
                    "required": [
                        "name",
                        "x",
                        "y",
                        "w",
                        "h",
                        "llm_prompt_override",
                        "ocr_prompt_override",
                    ],
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
        llm_override = roi.get("llm_prompt_override")
        if isinstance(llm_override, str):
            llm_clean = llm_override.strip()
            out[-1]["llm_prompt_override"] = llm_clean if llm_clean else None
        else:
            out[-1]["llm_prompt_override"] = None

        ocr_override = roi.get("ocr_prompt_override")
        if isinstance(ocr_override, str):
            ocr_clean = ocr_override.strip()
            out[-1]["ocr_prompt_override"] = ocr_clean if ocr_clean else None
        else:
            out[-1]["ocr_prompt_override"] = None

    return out


def paddle_preview_rois(image_path: Path) -> dict[str, Any]:
    """
    Build lightweight preview ROIs directly from Paddle detections:
      - ROI name = detected text (truncated)
      - ROI geometry = Paddle line box
    """
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_w, image_h = _load_image_size(image_path)
    doc_structure = _collect_document_structure(image_path)
    lines = doc_structure.get("lines") or []
    rois: list[dict[str, Any]] = []
    for i, line in enumerate(lines, start=1):
        text = " ".join(str(line.get("text", "") or "").split())
        if not text:
            continue
        name = text[:80]
        rois.append(
            {
                "id": "p" + uuid.uuid4().hex[:9],
                "name": name,
                "x": int(_coerce_int(line.get("x"))),
                "y": int(_coerce_int(line.get("y"))),
                "w": max(ROI_MIN_SIZE, int(_coerce_int(line.get("w"), ROI_MIN_SIZE))),
                "h": max(ROI_MIN_SIZE, int(_coerce_int(line.get("h"), ROI_MIN_SIZE))),
                "llm_prompt_override": None,
                "ocr_prompt_override": None,
            }
        )
    return {
        "rois": rois,
        "image_width": image_w,
        "image_height": image_h,
        "ocr_line_count": int(doc_structure.get("line_count", 0)),
        "document_structure_from_paddleocr": doc_structure,
    }


def detect_rois_with_openai(
    image_path: Path,
    *,
    model_profile: str = "max_quality",
    precomputed_doc_structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_w, image_h = _load_image_size(image_path)
    doc_structure = precomputed_doc_structure or _collect_document_structure(image_path)
    if not doc_structure.get("lines"):
        raise RuntimeError("PaddleOCR detected no text lines.")

    ref_path, ref_example = _load_custom_reference_schema()

    instructions = (
        "You generate ROI templates for fixed-layout assessment forms.\n"
        "Return ONE JSON object only that matches the provided JSON schema exactly.\n"
        "Do not include markdown, prose, comments, or extra keys.\n"
        "Rules:\n"
        "1) Use the OCR line boxes and text as the source of layout truth.\n"
        "2) Follow the naming style and granularity pattern of the reference templates (side_a + side_b).\n"
        "2a) Use the provided reference_paddle_to_schema_examples to map Paddle line patterns into schema ROI structure.\n"
        "2b) Do not copy ROI geometry directly from references.\n"
        "3) For MCQ blocks, include both question ROI (numeric name) and choice ROIs (e.g., 1a, 1b...).\n"
        "4) Coordinates must be integer pixels in the target image coordinate space.\n"
        "5) Every ROI must stay fully inside the image bounds.\n"
        "6) For free-response/text ROIs, generate concise, field-specific prompt overrides.\n"
        "6a) Prompt overrides must closely model the reference style and wording patterns.\n"
        "6b) ocr_prompt_override must follow the Chinese template style used in the reference, including a JSON object to fill.\n"
        "6c) ocr_prompt_override should begin with: 请按下列JSON格式输出图中信息:\n"
        "6d) ocr_prompt_override JSON keys must be specific to the ROI use case (e.g., date parts, school_name, teacher_name, id, answer_letter, age).\n"
        "6e) llm_prompt_override must be strict-normalizer style, ROI-specific, and return a single normalized value (or null when appropriate).\n"
        "7) Always include llm_prompt_override and ocr_prompt_override keys for every ROI.\n"
        "8) For MCQ ROIs, set llm_prompt_override=null and ocr_prompt_override=null.\n"
        "9) For free-response/text ROIs, set both prompt override values to non-empty strings.\n"
        "10) Do not hallucinate fields not supported by the document structure.\n"
        "11) Keep prompt text compact and consistent across similar ROI types.\n"
        "12) For ROI named 'id', infer bounds from target OCR header evidence only; do not use reference 'id' geometry priors.\n"
    )

    schema = _output_schema()
    payload = {
        "target_image": {
            "path": str(image_path),
            "width": image_w,
            "height": image_h,
        },
        "document_structure_from_paddleocr": doc_structure,
        "reference_template_example": ref_example,
        "reference_paddle_to_schema_examples": {
            "side_a": {
                "paddle_output": _load_custom_reference_paddle("a"),
                "schema_rois": _reference_rois_without_id(ref_example, "a"),
            },
            "side_b": {
                "paddle_output": _load_custom_reference_paddle("b"),
                "schema_rois": _reference_rois_without_id(ref_example, "b"),
            },
        },
    }

    use_openai = bool(ROI_AUTO_DETECT.get("use_openai", True))
    openai_model_override = str(ROI_AUTO_DETECT.get("openai_model") or "").strip() or None
    openai_reasoning_enabled = bool(ROI_AUTO_DETECT.get("openai_reasoning", True))
    openai_reasoning_cfg = None if openai_reasoning_enabled else {"effort": "none"}
    if use_openai:
        raw_obj = call_json(
            json.dumps(payload, ensure_ascii=False),
            schema=schema,
            schema_name="roi_template",
            instructions=instructions,
            model=openai_model_override,
            model_profile=None if openai_model_override else model_profile,
            max_output_tokens=12000,
            reasoning=openai_reasoning_cfg,
        )
    else:
        raw_obj = _call_local_json(
            payload=payload,
            instructions=instructions,
            schema=schema,
        )
    rois = _validate_and_normalize_rois(raw_obj, image_w, image_h)

    return {
        "rois": rois,
        "image_width": image_w,
        "image_height": image_h,
        "reference_schema_path": ref_path,
        "ocr_line_count": int(doc_structure.get("line_count", 0)),
        "model": (
            (openai_model_override or resolve_model(model_profile=model_profile))
            if use_openai
            else str(ROI_AUTO_DETECT.get("local_model", "qwen3.5:9b") or "qwen3.5:9b")
        ),
    }
