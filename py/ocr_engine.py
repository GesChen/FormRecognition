"""
Unified OCR module: image path in, text out.

Workflows:
- ocr_raw_vision_only: direct local vision OCR call.
- ocr_raw_paddle_only: Paddle OCR only, no vision fallback.
- ocr_raw_paddle_then_vision: Paddle first (min-confidence gate), then local vision fallback.
- ocr_raw_vision_with_paddle_confidence: vision text with Paddle confidence.
- ocr_raw_paddle_vlm_fusion: vision text plus Paddle sidecar for deferred LLM fusion.

Default ocr_raw() uses the configured default workflow.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from config import LLM, OCR_ENGINE

_LABEL_TO_SCORE = {
    "low": 0.25,
    "medium": 0.6,
    "high": 0.9,
}

_paddle_engine = None


def _vprint(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[ocr_engine] {message}")


def _cfg() -> dict[str, Any]:
    return OCR_ENGINE if isinstance(OCR_ENGINE, dict) else {}


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(_cfg().get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_cfg().get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _cfg_str(key: str, default: str) -> str:
    v = _cfg().get(key, default)
    s = str(v).strip()
    return s or str(default)


def _cfg_bool(key: str, default: bool) -> bool:
    v = _cfg().get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return bool(default)


def _repeat_cfg() -> dict[str, Any]:
    v = _cfg().get("vlm_repeat_stop", {})
    return v if isinstance(v, dict) else {}


def _repeat_cfg_bool(key: str, default: bool) -> bool:
    v = _repeat_cfg().get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return bool(default)


def _repeat_cfg_int(key: str, default: int) -> int:
    try:
        return int(_repeat_cfg().get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _llm_host() -> str:
    return str(_cfg().get("host") or LLM.get("host", "127.0.0.1"))


def _llm_port() -> int:
    try:
        return int(_cfg().get("port") if _cfg().get("port") is not None else LLM.get("port", 11434))
    except (TypeError, ValueError):
        return 11434


def _llm_keep_alive() -> int:
    try:
        return int(_cfg().get("keep_alive") if _cfg().get("keep_alive") is not None else LLM.get("keep_alive", 0))
    except (TypeError, ValueError):
        return 0


def _workflow_default() -> str:
    return _cfg_str("workflow_default", "paddle_then_vision").lower()


def _normalize_jpeg_quality(quality: int) -> int:
    try:
        q = int(quality)
    except (TypeError, ValueError):
        q = 60
    return max(1, min(100, q))


def _encode_image(image_path: str | Path, *, quality: int) -> tuple[str, dict[str, Any]]:
    img = Image.open(image_path).convert("L")
    requested_quality = quality
    applied_quality = _normalize_jpeg_quality(quality)
    subsampling = 0 if applied_quality >= 95 else 2
    buf = BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=applied_quality,
        optimize=True,
        progressive=False,
        subsampling=subsampling,
    )
    data = buf.getvalue()
    meta = {
        "jpeg_quality_requested": int(requested_quality),
        "jpeg_quality_applied": int(applied_quality),
        "jpeg_subsampling": "4:4:4" if subsampling == 0 else "4:2:0",
        "jpeg_bytes": len(data),
    }
    return base64.b64encode(data).decode("ascii"), meta


def _first_nonempty_scalar_value(obj: dict[str, Any]) -> Any:
    for key in ("detected_text", "text"):
        if key not in obj:
            continue
        value = obj.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        if isinstance(value, (int, float, bool)):
            return value
    for key, value in obj.items():
        if key in {"confidence", "confidence_score", "confidence_label", "needs_human_review"}:
            continue
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        if isinstance(value, (int, float, bool)):
            return value
    return None


def _normalize_parsed_json_obj(obj: Any) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    out = dict(obj)
    if "detected_text" not in out:
        if "text" in out:
            out["detected_text"] = out.get("text")
        else:
            scalar = _first_nonempty_scalar_value(out)
            if scalar is not None:
                out["detected_text"] = scalar
    return out


def _json_text_candidates(text: str) -> list[str]:
    s = str(text or "").strip()
    if not s:
        return []
    candidates: list[str] = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", s, flags=re.IGNORECASE):
        block = m.group(1).strip()
        if block:
            candidates.append(block)
    candidates.append(s)
    for start in [m.start() for m in re.finditer(r"\{", s)]:
        end = s.find("}", start)
        if end >= 0:
            candidates.append(s[start : end + 1].strip())
        else:
            candidates.append(s[start:].strip())

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        c = candidate.strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _clean_malformed_scalar(value: str) -> str:
    v = str(value or "").strip()
    v = v.strip("`")
    v = re.sub(r"\s*```\s*$", "", v).strip()
    v = v.strip().strip(",")
    v = v.strip().strip('"').strip("'").strip()
    return v


def _malformed_json_like_from_text(text: str) -> dict[str, Any]:
    """
    Best-effort extraction for common VLM JSON mistakes.

    Examples handled:
      {"Record ID: "8011576A"}
      {"Record ID" "8011576A"}
      {"Record ID":8011576A}
      Record ID: 8011576A
    """
    for raw in _json_text_candidates(text):
        s = raw.strip()
        if not s:
            continue
        s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        s = re.sub(r",\s*}", "}", s)

        repaired = s
        if "'" in repaired and '"' not in repaired:
            repaired = repaired.replace("'", '"')
        try:
            obj = json.loads(repaired)
            normalized = _normalize_parsed_json_obj(obj)
            if normalized:
                return normalized
        except json.JSONDecodeError:
            pass

        patterns = (
            # Missing JSON colon between a quoted key and quoted value; the key
            # may itself end with a human-readable colon.
            r'^\{\s*"(?P<key>[^"]+?)"\s*:?\s*"(?P<value>[^"]+)"\s*\}\s*$',
            # Missing JSON colon and value is effectively bare after the key quote:
            # {"Record ID: "8011576A"}
            r'^\{\s*"(?P<key>[^"]+?:\s*)"\s*(?P<value>[^"}]+)"?\s*\}\s*$',
            # Quoted key with bare alphanumeric-ish value.
            r'^\{\s*"(?P<key>[^"]+?)"\s*:?\s*(?P<value>[^"}][^}]*)\s*\}\s*$',
            # Unbraced label/value response.
            r'^(?P<key>[A-Za-z][^:\n]{0,80}?)\s*:\s*(?P<value>[^\n{}]+)\s*$',
        )
        for pattern in patterns:
            m = re.match(pattern, s, flags=re.DOTALL)
            if not m:
                continue
            value = _clean_malformed_scalar(m.group("value"))
            if value:
                key = str(m.groupdict().get("key") or "detected_text").strip()
                return _normalize_parsed_json_obj({key: value})

    return {}


def _json_from_text(text: str) -> dict[str, Any]:
    if not text:
        return {}
    for s in _json_text_candidates(text):
        try:
            obj = json.loads(s)
            parsed = _normalize_parsed_json_obj(obj)
            if parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        dec = json.JSONDecoder()
        i0 = s.find("{")
        if i0 < 0:
            continue
        try:
            obj, _ = dec.raw_decode(s[i0:])
            parsed = _normalize_parsed_json_obj(obj)
            if parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    return _malformed_json_like_from_text(text)


def _normalize_label(label: Any) -> str:
    x = str(label or "").strip().lower()
    if x in {"low", "medium", "high"}:
        return x
    return "low"


def _score_to_label(score: float) -> str:
    """
    Derive low/medium/high from numeric score.
    """
    s = max(0.0, min(1.0, float(score)))
    if s >= 0.85:
        return "high"
    if s >= 0.5:
        return "medium"
    return "low"


def _coerce_confidence_score(parsed: dict[str, Any]) -> float:
    """
    Internal confidence coercion.
    VLM prompts no longer require confidence output; when missing, default to 1.0.
    Legacy confidence fields are still accepted when present.
    """
    has_score = "confidence_score" in parsed
    has_conf = "confidence" in parsed
    has_label = "confidence_label" in parsed
    score_raw = parsed.get("confidence_score") if has_score else None
    if score_raw is None and has_conf:
        score_raw = parsed.get("confidence")
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        if has_label:
            label = _normalize_label(parsed.get("confidence_label"))
            score = _LABEL_TO_SCORE.get(label, 1.0)
        else:
            score = 1.0
    return max(0.0, min(1.0, float(score)))


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value or "").strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _has_uncertainty_signal(parsed: dict[str, Any]) -> bool:
    se = parsed.get("self_evaluation")
    if not isinstance(se, dict):
        return False
    for key in ("legibility_issues", "ambiguities", "missing_regions"):
        v = se.get(key)
        if isinstance(v, list):
            if any(str(x).strip() for x in v):
                return True
        elif isinstance(v, str) and v.strip():
            return True
    reason = str(se.get("reason_for_confidence", "") or "").lower()
    uncertainty_terms = (
        "unclear",
        "illegible",
        "ambiguous",
        "cropped",
        "cut off",
        "partial",
        "blur",
        "faint",
        "uncertain",
    )
    return any(tok in reason for tok in uncertainty_terms)


def _is_confident(parsed: dict[str, Any], *, min_score: float) -> bool:
    if not isinstance(parsed, dict) or not parsed:
        return False
    score = _coerce_confidence_score(parsed)
    return score >= float(min_score)


def _is_paddle_confident(parsed: dict[str, Any]) -> bool:
    try:
        score = float(parsed.get("confidence_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return score > _cfg_float("paddle_min_confidence_for_accept", 0.90)


def _build_prompt() -> str:
    return (
        "You are an OCR evaluator. Extract text from the image.\n"
        "Return ONE valid JSON object only, no markdown and no extra text.\n"
        "Required schema:\n"
        "{\n"
        '  "detected_text": "string"\n'
        "}\n"
        "Rules:\n"
        "- Keep detected_text concise and faithful to visible text only.\n"
        "- Include unclear but visible text as best-effort in detected_text; do not omit it just because it is uncertain.\n"
        "- Mark uncertain spans inline with [[? ... ]] when needed.\n"
        "- Do not invent text that has no visible evidence in the image.\n"
        "Task:\n"
        "Extract all visible text (including unclear best-effort spans)."
    )


def default_text_prompt() -> str:
    """Public accessor for the base OCR text prompt template."""
    return _cfg_str(
        "vlm_default_query_prompt",
        "Text recognition:\n```json\n{\n\"text\":\"\"\n}\n```",
    )


def build_vlm_roi_prompt(query: str | None) -> str | None:
    """Build the configured custom VLM prompt for a per-ROI query."""
    if not _cfg_bool("vlm_custom_query_enabled", True):
        return None
    q = str(query or "").strip()
    if not q:
        return None
    template = str(
        _cfg().get(
            "vlm_roi_prompt_template",
            '请按下列JSON格式输 出图中信息: {"{query}":""}',
        )
        or ""
    )
    if not template.strip():
        return None
    if "{query}" in template:
        return template.replace("{query}", q)
    return f"{template}{q}"


def _is_text_json_obj(obj: Any) -> bool:
    return isinstance(obj, dict) and _first_nonempty_scalar_value(obj) is not None


def _canonical_stream_json_obj(obj: dict[str, Any]) -> str:
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(obj)


def _stream_text_json_blocks(text: str) -> list[dict[str, Any]]:
    """
    Return completed OCR JSON objects found anywhere in streamed text.

    This intentionally does not require markdown fences to close. Some VLMs emit:
        ```json
        {"text": "..."}
        ```
    and then keep generating more fenced JSON blocks. As soon as the object itself
    is complete, the caller can safely stop reading.
    """
    s = str(text or "")
    if not s:
        return []

    decoder = json.JSONDecoder()
    blocks: list[dict[str, Any]] = []
    seen_spans: set[tuple[int, int]] = set()
    for match in re.finditer(r"\{", s):
        start = match.start()
        try:
            obj, rel_end = decoder.raw_decode(s[start:])
        except json.JSONDecodeError:
            continue
        if not _is_text_json_obj(obj):
            continue
        end = start + rel_end
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        blocks.append(
            {
                "mode": "json_object",
                "start_index": start,
                "end_index": end,
                "obj": obj,
                "canonical": _canonical_stream_json_obj(obj),
            }
        )
    blocks.sort(key=lambda b: int(b.get("start_index", 0)))
    return blocks


def _stream_text_json_completion(text: str) -> dict[str, int | str] | None:
    """
    Return metadata once the streamed text contains a complete {"text": ...} JSON block.
    """
    s = str(text or "")
    if not s:
        return None

    blocks = _stream_text_json_blocks(s)
    if blocks:
        first = blocks[0]
        return {
            "mode": str(first.get("mode") or "json_object"),
            "start_index": int(first.get("start_index", 0) or 0),
            "end_index": int(first.get("end_index", 0) or 0),
        }

    fence = re.search(r"```(?:json)?\s*", s, flags=re.IGNORECASE)
    if fence:
        close = s.find("```", fence.end())
        if close >= 0:
            block = s[fence.end() : close].strip()
            try:
                obj = json.loads(block)
            except json.JSONDecodeError:
                obj = None
            if _is_text_json_obj(obj):
                return {"mode": "fenced_json", "end_index": close + 3}

    start = s.find("{")
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(s[start:])
    except json.JSONDecodeError:
        close = s.find("}", start)
        if close >= 0 and _malformed_json_like_from_text(s[start : close + 1]):
            return {
                "mode": "malformed_json_object",
                "start_index": start,
                "end_index": close + 1,
            }
        return None
    if _is_text_json_obj(obj):
        return {"mode": "raw_json", "end_index": start + end}
    return None


def _stream_text_json_repeat_completion(text: str) -> dict[str, int | str] | None:
    """
    Return metadata when streamed OCR output starts repeating JSON answers.

    The returned end_index points to the end of the first completed JSON block,
    so downstream parsing keeps the first answer and discards repeated tails.
    """
    blocks = _stream_text_json_blocks(text)
    if len(blocks) < max(2, _repeat_cfg_int("json_min_blocks", 2)):
        return None

    first = blocks[0]
    second = blocks[1]
    stop: dict[str, int | str] = {
        "mode": "repeated_json_blocks",
        "start_index": int(first.get("start_index", 0) or 0),
        "end_index": int(first.get("end_index", 0) or 0),
        "block_count": len(blocks),
    }
    if first.get("canonical") == second.get("canonical"):
        stop["mode"] = "repeated_identical_json"
    return stop


def _stream_repeated_tail_completion(text: str) -> dict[str, int | str] | None:
    """
    Detect exact repeated suffix loops in a stream.

    This is a fallback for malformed generations that never yield a parseable
    JSON object. It is intentionally conservative: the repeated unit must be
    reasonably long and repeated several times at the end of the stream.
    """
    if not _repeat_cfg_bool("tail_enabled", True):
        return None

    s = str(text or "")
    min_unit = max(8, _repeat_cfg_int("tail_min_unit_chars", 24))
    max_unit = max(min_unit, _repeat_cfg_int("tail_max_unit_chars", 240))
    repeats = max(2, _repeat_cfg_int("tail_repeats", 3))
    min_total = max(min_unit * repeats, _repeat_cfg_int("tail_min_total_chars", 80))
    if len(s) < min_total:
        return None

    upper_unit = min(max_unit, len(s) // repeats)
    for unit_len in range(min_unit, upper_unit + 1):
        unit = s[-unit_len:]
        if not unit.strip():
            continue
        repeated = unit * repeats
        if not s.endswith(repeated):
            continue
        return {
            "mode": "repeated_tail",
            "end_index": len(s) - (unit_len * (repeats - 1)),
            "repeat_unit_chars": unit_len,
            "repeat_count": repeats,
        }
    return None


def _stream_stop_completion(text: str) -> dict[str, int | str] | None:
    """
    Decide whether the current streamed text is complete enough to stop reading.
    """
    json_completion = _stream_text_json_completion(text)
    if json_completion is not None:
        json_completion = dict(json_completion)
        json_completion["stop_reason"] = "json_completion"
        return json_completion

    if _repeat_cfg_bool("json_enabled", True):
        repeated_json = _stream_text_json_repeat_completion(text)
        if repeated_json is not None:
            repeated_json = dict(repeated_json)
            repeated_json["stop_reason"] = "json_repeat"
            return repeated_json

    repeated_tail = _stream_repeated_tail_completion(text)
    if repeated_tail is not None:
        repeated_tail = dict(repeated_tail)
        repeated_tail["stop_reason"] = "tail_repeat"
        return repeated_tail
    return None


def _completed_json_from_stream(
    text: str,
    json_completion: dict[str, int | str] | None = None,
) -> str:
    """
    Return the completed JSON block when streaming found its end.
    """
    s = str(text or "")
    if isinstance(json_completion, dict):
        try:
            end_index = int(json_completion.get("end_index", 0))
        except (TypeError, ValueError):
            end_index = 0
        if end_index > 0:
            return s[:end_index]
    return s


def _local_vision_call(
    *,
    model: str,
    image_b64: str,
    prompt: str,
    timeout: int,
    host: str,
    port: int,
    keep_alive: int,
    stream: bool,
    extra_params: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    effective_prompt = str(prompt or "").strip() or default_text_prompt()
    used_default_prompt = effective_prompt == default_text_prompt()
    payload: dict[str, Any] = {
        "model": model,
        "prompt": effective_prompt,
        "stream": bool(stream),
        "keep_alive": int(keep_alive),
        "images": [image_b64],
    }
    if extra_params:
        payload.update(extra_params)

    url = f"http://{host}:{int(port)}/api/generate"
    if bool(stream):
        response_parts: list[str] = []
        terminal_obj: dict[str, Any] = {}
        json_completion: dict[str, int | str] | None = None
        chunk_count = 0
        with requests.post(url, json=payload, timeout=(10, timeout), stream=True) as resp:
            if resp.status_code >= 400:
                detail = (resp.text or "").strip()
                if len(detail) > 1500:
                    detail = detail[:1500] + "...(truncated)"
                raise RuntimeError(f"HTTP {resp.status_code} error from {url}: {detail}")

            for raw_line in resp.iter_lines(chunk_size=1, decode_unicode=False):
                if not raw_line:
                    continue
                if isinstance(raw_line, bytes):
                    line = raw_line.decode("utf-8", errors="replace").strip()
                else:
                    line = str(raw_line).strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chunk_count += 1
                delta = str(obj.get("response", "") or "")
                if delta:
                    response_parts.append(delta)
                    full_so_far = "".join(response_parts)
                    json_completion = _stream_stop_completion(full_so_far)
                    if json_completion is not None:
                        terminal_obj = obj
                        break

                if bool(obj.get("done")):
                    terminal_obj = obj
                    break

        full_text = "".join(response_parts)
        text = _completed_json_from_stream(
            full_text,
            json_completion,
        )
        body: dict[str, Any] = dict(terminal_obj or {})
        body["response"] = text
        body["streaming"] = {
            "enabled": True,
            "chunk_count": chunk_count,
            "stopped_for_json_completion": json_completion is not None,
            "stopped_for_stream_completion": json_completion is not None,
            "stop_reason": (
                str(json_completion.get("stop_reason"))
                if isinstance(json_completion, dict) and json_completion.get("stop_reason")
                else None
            ),
            "json_completion": json_completion,
            "raw_response_chars": len(full_text),
            "returned_response_chars": len(text),
        }
        body["prompt_override"] = {
            "default_query_prompt": used_default_prompt,
            "ignored_passed_prompt": False,
            "custom_prompt": not used_default_prompt,
        }
        return text, body

    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        detail = (resp.text or "").strip()
        if len(detail) > 1500:
            detail = detail[:1500] + "...(truncated)"
        raise RuntimeError(f"HTTP {resp.status_code} error from {url}: {detail}")
    body = resp.json()
    text = str(body.get("response", "") or "")
    if isinstance(body, dict):
        body["streaming"] = {"enabled": False}
        body["prompt_override"] = {
            "default_query_prompt": used_default_prompt,
            "ignored_passed_prompt": False,
            "custom_prompt": not used_default_prompt,
        }
    return text, body


def _apply_paddle_env_overrides() -> None:
    raw = _cfg().get("paddle_env_overrides", {})
    if not isinstance(raw, dict):
        return
    for env_key, env_val in raw.items():
        os.environ.setdefault(str(env_key), str(env_val))


def _get_paddle_engine():
    global _paddle_engine
    if _paddle_engine is not None:
        return _paddle_engine

    _apply_paddle_env_overrides()
    try:
        import logging
        from paddleocr import PaddleOCR, logger as paddleocr_logger
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PaddleOCR unavailable. Install with: pip install paddlepaddle paddleocr"
        ) from exc

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paddleocr_logger.setLevel(logging.CRITICAL)
            _paddle_engine = PaddleOCR(
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
    return _paddle_engine


def _run_paddle_stage(
    path: Path,
    *,
    verbose: bool,
    force_failure: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    stage: dict[str, Any] = {
        "stage_index": 0,
        "model": _cfg_str("paddle_model_name", "paddle-ppocrv5"),
        "elapsed_sec": None,
        "raw_response": None,
        "response_text": "",
        "parsed": {},
        "accepted": False,
        "error": None,
        "image_payload": {},
    }

    t0 = time.time()
    paddle_full_out: dict[str, Any] | None = None
    try:
        if force_failure:
            raise RuntimeError("Forced Paddle failure for testing.")
        _vprint(verbose, f"stage 0 begin model={stage['model']} (paddle)")
        engine = _get_paddle_engine()
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = engine.predict(input=str(path.resolve()))

        rec_texts: list[str] = []
        rec_scores: list[float] = []
        raw_items: list[dict[str, Any]] = []
        for res in outputs or []:
            j = getattr(res, "json", None)
            if isinstance(j, dict):
                raw_items.append(j)
                block = j.get("res") or {}
                texts = block.get("rec_texts", [])
                scores = block.get("rec_scores", [])
                if isinstance(texts, list):
                    rec_texts.extend(str(t) for t in texts if str(t).strip())
                elif texts:
                    rec_texts.append(str(texts))
                if isinstance(scores, list):
                    for s in scores:
                        try:
                            rec_scores.append(float(s))
                        except (TypeError, ValueError):
                            pass

        text = " ".join(rec_texts).strip()
        min_score = min(rec_scores) if rec_scores else 0.0
        mean_score = (sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0
        label = "high" if min_score > 0.90 else ("medium" if min_score >= 0.60 else "low")
        needs_review = min_score <= _cfg_float("paddle_min_confidence_for_accept", 0.90)

        parsed = {
            "detected_text": text,
            "confidence_label": label,
            "confidence_score": float(min_score),
            "needs_human_review": bool(needs_review),
            "self_evaluation": {
                "engine": "paddle",
                "line_count": len(rec_texts),
                "min_rec_score": min_score,
                "mean_rec_score": mean_score,
                "reason_for_confidence": (
                    f"paddle min_rec_score={min_score:.4f}, mean_rec_score={mean_score:.4f}"
                ),
            },
        }
        paddle_full_out = {
            "detected_text": text,
            "confidence_label": label,
            "confidence_score": float(min_score),
            "needs_human_review": bool(needs_review),
            "self_evaluation": parsed["self_evaluation"],
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "raw_response": raw_items,
        }

        stage["parsed"] = parsed
        stage["raw_response"] = raw_items
        stage["response_text"] = text
        stage["accepted"] = _is_paddle_confident(parsed)
        _vprint(
            verbose,
            f"stage 0 complete accepted={stage['accepted']} confidence={parsed.get('confidence_score')}",
        )
    except Exception as exc:  # pragma: no cover
        stage["error"] = str(exc)
        _vprint(verbose, f"stage 0 error={exc}")
    finally:
        stage["elapsed_sec"] = round(time.time() - t0, 4)
        _vprint(verbose, f"stage 0 elapsed={stage['elapsed_sec']}s")

    return stage, paddle_full_out


def _run_vision_stage(
    path: Path,
    *,
    timeout: int,
    min_conf: float,
    verbose: bool,
    stage_index: int,
    prompt_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _cfg_str("model", "qwen2.5vl")
    host = _llm_host()
    port = _llm_port()
    keep_alive = _llm_keep_alive()
    stream = _cfg_bool("stream", False)
    extra_params_raw = _cfg().get("extra_params", {})
    extra_params = extra_params_raw if isinstance(extra_params_raw, dict) else None
    max_call_ms = max(0, _cfg_int("vlm_max_call_ms", 0))
    slow_call_retries = max(0, _cfg_int("vlm_slow_call_retries", 0))

    image_encode_meta: dict[str, Any] = {}
    stage: dict[str, Any] = {
        "stage_index": stage_index,
        "model": model,
        "elapsed_sec": None,
        "raw_response": None,
        "response_text": "",
        "parsed": {},
        "accepted": False,
        "error": None,
        "image_payload": {},
        "vlm_call_ms": None,
        "vlm_attempt": None,
        "vlm_slow_dropped_attempts": [],
    }

    prompt_text = str(prompt_override or "").strip() or default_text_prompt()
    stage["prompt_source"] = "roi_override" if str(prompt_override or "").strip() else "default"
    stage["prompt"] = prompt_text

    _vprint(verbose, f"stage {stage_index} begin model={model} (vision)")
    t0 = time.time()
    try:
        image_b64, image_encode_meta = _encode_image(path, quality=_cfg_int("jpeg_quality", 60))
        stage["image_payload"] = dict(image_encode_meta)
        _vprint(
            verbose,
            "vision image encoded "
            f"quality_requested={image_encode_meta.get('jpeg_quality_requested')} "
            f"quality_applied={image_encode_meta.get('jpeg_quality_applied')} "
            f"subsampling={image_encode_meta.get('jpeg_subsampling')} "
            f"bytes={image_encode_meta.get('jpeg_bytes')}",
        )
        response_text = ""
        body: dict[str, Any] = {}
        attempts = 1 + slow_call_retries if max_call_ms > 0 else 1
        for attempt in range(1, attempts + 1):
            t_call = time.perf_counter()
            response_text, body = _local_vision_call(
                model=model,
                image_b64=image_b64,
                prompt=prompt_text,
                timeout=timeout,
                host=host,
                port=port,
                keep_alive=keep_alive,
                stream=stream,
                extra_params=extra_params,
            )
            call_ms = (time.perf_counter() - t_call) * 1000.0
            stage["vlm_call_ms"] = round(call_ms, 2)
            stage["vlm_attempt"] = attempt
            if max_call_ms > 0 and call_ms > float(max_call_ms):
                stage["vlm_slow_dropped_attempts"].append(
                    {
                        "attempt": attempt,
                        "elapsed_ms": round(call_ms, 2),
                        "max_ms": int(max_call_ms),
                    }
                )
                if attempt < attempts:
                    _vprint(
                        verbose,
                        f"stage {stage_index} slow VLM call dropped "
                        f"(attempt={attempt}/{attempts}, elapsed_ms={call_ms:.2f}, max_ms={max_call_ms}); retrying",
                    )
                    continue
                raise TimeoutError(
                    f"VLM call exceeded max duration: {call_ms:.2f}ms > {max_call_ms}ms "
                    f"after {attempts} attempt(s)"
                )
            break
        parsed = _json_from_text(response_text)
        stage["raw_response"] = body
        stage["response_text"] = response_text
        stage["parsed"] = parsed
        stage["accepted"] = _is_confident(parsed, min_score=min_conf)
        _vprint(
            verbose,
            f"stage {stage_index} complete accepted={stage['accepted']} "
            f"confidence_label={parsed.get('confidence_label')} confidence_score={parsed.get('confidence_score')}",
        )
    except Exception as exc:
        stage["error"] = str(exc)
        _vprint(verbose, f"stage {stage_index} error={exc}")
    finally:
        stage["elapsed_sec"] = round(time.time() - t0, 4)
        _vprint(verbose, f"stage {stage_index} elapsed={stage['elapsed_sec']}s")

    return stage, image_encode_meta


def _build_output(
    *,
    chosen: dict[str, Any],
    stages: list[dict[str, Any]],
    min_conf: float,
    image_encode_meta: dict[str, Any],
    paddle_full_out: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed = dict(chosen.get("parsed") or {})
    score = _coerce_confidence_score(parsed)
    label = _score_to_label(score)

    last_stage = stages[-1] if stages else {}
    needs_review = not (score >= float(min_conf))
    if last_stage.get("error"):
        needs_review = True

    detected_text = str(parsed.get("detected_text", "") or "").strip()
    raw_response_text = str(chosen.get("response_text", "") or "").strip()
    # Preserve plain-text model responses as a compatibility fallback, but never
    # promote a successfully parsed JSON envelope into OCR text.  An empty JSON
    # value (for example {"text": ""}) means the model detected no text.
    if not detected_text and not parsed and raw_response_text and not chosen.get("error"):
        detected_text = raw_response_text

    selected_model = str(chosen.get("model", _cfg_str("model", "qwen2.5vl")))
    selected_stage_index = int(chosen.get("stage_index", 0))

    if (
        selected_model == _cfg_str("paddle_model_name", "paddle-ppocrv5")
        and isinstance(paddle_full_out, dict)
    ):
        rec_scores_raw = list(paddle_full_out.get("rec_scores") or [])
        rec_scores: list[float] = []
        for x in rec_scores_raw:
            try:
                rec_scores.append(float(x))
            except (TypeError, ValueError):
                pass
        rec_texts = list(paddle_full_out.get("rec_texts") or [])
        min_rec_score = float(paddle_full_out.get("confidence_score", score))
        mean_rec_score = float(sum(rec_scores) / len(rec_scores)) if rec_scores else min_rec_score
        raw_response = paddle_full_out.get("raw_response")
    else:
        rec_scores = [score]
        rec_texts = [detected_text] if detected_text else []
        min_rec_score = score
        mean_rec_score = score
        raw_response = chosen.get("raw_response")

    return {
        "detected_text": detected_text,
        "confidence_label": label,
        "confidence_score": score,
        "needs_human_review": bool(needs_review),
        "selected_model": selected_model,
        "selected_stage_index": selected_stage_index,
        "min_rec_score": min_rec_score,
        "mean_rec_score": mean_rec_score,
        "rec_scores": rec_scores,
        "rec_texts": rec_texts,
        "raw_response": raw_response,
        "self_evaluation": parsed.get("self_evaluation", {}),
        "image_payload": image_encode_meta,
        "stages": stages,
    }


def _paddle_confidence_summary(
    *,
    paddle_stage: dict[str, Any],
    paddle_full_out: dict[str, Any] | None,
    include_raw: bool,
) -> dict[str, Any]:
    parsed = dict(paddle_stage.get("parsed") or {})
    full = paddle_full_out if isinstance(paddle_full_out, dict) else {}
    error = paddle_stage.get("error")

    if error and not full and not parsed:
        return {
            "detected_text": "",
            "confidence_label": None,
            "confidence_score": None,
            "needs_human_review": True,
            "min_rec_score": None,
            "mean_rec_score": None,
            "rec_scores": [],
            "rec_texts": [],
            "selected_model": paddle_stage.get("model"),
            "selected_stage_index": paddle_stage.get("stage_index"),
            "self_evaluation": {},
            "error": error,
        }

    rec_scores_raw = full.get("rec_scores") if full else []
    rec_scores: list[float] = []
    if isinstance(rec_scores_raw, list):
        for x in rec_scores_raw:
            try:
                rec_scores.append(float(x))
            except (TypeError, ValueError):
                pass

    rec_texts_raw = full.get("rec_texts") if full else []
    rec_texts: list[str] = []
    if isinstance(rec_texts_raw, list):
        rec_texts = [str(t) for t in rec_texts_raw if str(t).strip()]

    detected_text = str(full.get("detected_text") or parsed.get("detected_text") or "").strip()
    score = _coerce_confidence_score(full or parsed)
    label = _score_to_label(score)
    min_rec_score = min(rec_scores) if rec_scores else score
    mean_rec_score = (sum(rec_scores) / len(rec_scores)) if rec_scores else score
    needs_review_raw = full.get("needs_human_review") if full else parsed.get("needs_human_review")
    needs_review = _coerce_bool(needs_review_raw)
    if needs_review is None:
        needs_review = score < _cfg_float("paddle_min_confidence_for_accept", 0.90)

    summary = {
        "detected_text": detected_text,
        "confidence_label": label,
        "confidence_score": score,
        "needs_human_review": bool(needs_review),
        "min_rec_score": float(min_rec_score),
        "mean_rec_score": float(mean_rec_score),
        "rec_scores": rec_scores,
        "rec_texts": rec_texts,
        "selected_model": paddle_stage.get("model"),
        "selected_stage_index": paddle_stage.get("stage_index"),
        "self_evaluation": full.get("self_evaluation") or parsed.get("self_evaluation", {}),
        "error": error,
    }
    if include_raw and full:
        summary["raw_response"] = full.get("raw_response")
    return summary


def ocr_raw_vision_only(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    call_timeout = int(timeout or _cfg_int("call_timeout_sec", 90))
    min_conf = _cfg_float("min_confidence_for_accept", 0.82)

    _vprint(
        verbose,
        "start OCR vision_only "
        f"image={Path(path).resolve()} model={_cfg_str('model', 'qwen2.5vl')} "
        f"host={_llm_host()} port={_llm_port()} keep_alive={_llm_keep_alive()}",
    )

    vision_kwargs = {}
    if str(prompt_override or "").strip():
        vision_kwargs["prompt_override"] = prompt_override
    stage, image_meta = _run_vision_stage(
        path,
        timeout=call_timeout,
        min_conf=min_conf,
        verbose=verbose,
        stage_index=0,
        **vision_kwargs,
    )
    out = _build_output(
        chosen=stage,
        stages=[stage],
        min_conf=min_conf,
        image_encode_meta=image_meta,
        paddle_full_out=None,
    )
    _vprint(
        verbose,
        f"final selected_model={out.get('selected_model')} selected_stage_index={out.get('selected_stage_index')} "
        f"needs_human_review={out.get('needs_human_review')}",
    )
    return out


def ocr_raw_vision_with_paddle_confidence(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    force_paddle_failure: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    call_timeout = int(timeout or _cfg_int("call_timeout_sec", 90))
    min_conf = _cfg_float("min_confidence_for_accept", 0.82)

    _vprint(verbose, f"start OCR vision_with_paddle_confidence image={Path(path).resolve()}")

    vision_kwargs = {}
    if str(prompt_override or "").strip():
        vision_kwargs["prompt_override"] = prompt_override
    vision_stage, image_meta = _run_vision_stage(
        path,
        timeout=call_timeout,
        min_conf=min_conf,
        verbose=verbose,
        stage_index=0,
        **vision_kwargs,
    )
    paddle_stage, paddle_full = _run_paddle_stage(
        path,
        verbose=verbose,
        force_failure=force_paddle_failure,
    )
    stages = [vision_stage, paddle_stage]

    out = _build_output(
        chosen=vision_stage,
        stages=[vision_stage],
        min_conf=min_conf,
        image_encode_meta=image_meta,
        paddle_full_out=None,
    )
    out["stages"] = stages
    out["text_source"] = "vision"

    include_raw = _cfg_bool("include_paddle_confidence_raw", False)
    paddle_confidence = _paddle_confidence_summary(
        paddle_stage=paddle_stage,
        paddle_full_out=paddle_full,
        include_raw=include_raw,
    )
    out["paddle_confidence"] = paddle_confidence

    if isinstance(paddle_full, dict) and not paddle_stage.get("error"):
        out["confidence_source"] = "paddle"
        out["confidence_label"] = paddle_confidence["confidence_label"]
        out["confidence_score"] = paddle_confidence["confidence_score"]
        out["needs_human_review"] = paddle_confidence["needs_human_review"]
        out["min_rec_score"] = paddle_confidence["min_rec_score"]
        out["mean_rec_score"] = paddle_confidence["mean_rec_score"]
        out["rec_scores"] = paddle_confidence["rec_scores"]
        out["rec_texts"] = paddle_confidence["rec_texts"]
    else:
        out["confidence_source"] = "fallback_vlm"

    _vprint(
        verbose,
        f"final selected_model={out.get('selected_model')} selected_stage_index={out.get('selected_stage_index')} "
        f"confidence_source={out.get('confidence_source')} needs_human_review={out.get('needs_human_review')}",
    )
    return out


def ocr_raw_paddle_vlm_fusion(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    force_paddle_failure: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    """
    Run VLM and Paddle, carrying both guesses for deferred text-ROI fusion.

    The OCR result exposes VLM text as detected_text. Paddle output is preserved
    in paddle_confidence so pdf_recognize can run one deferred fusion pass before
    the deferred ROI-specific LLM normalizer.
    """
    out = ocr_raw_vision_with_paddle_confidence(
        image_path,
        timeout=timeout,
        verbose=verbose,
        force_paddle_failure=force_paddle_failure,
        prompt_override=prompt_override,
    )
    vlm_detected_text = str(out.get("detected_text", "") or "").strip()
    paddle_confidence = out.get("paddle_confidence")
    out["workflow"] = "paddle_vlm_fusion"
    out["vlm_detected_text"] = vlm_detected_text
    out["text_source"] = "vision"
    out["fusion"] = {
        "enabled": True,
        "deferred": True,
        "normalizer_inputs": ["deferred_fusion.detected_text"],
        "downstream_raw_ocr_source": "deferred_text_llm_input",
        "detected_text": None,
        "vlm_detected_text": vlm_detected_text,
        "paddle_detected_text": str(
            (paddle_confidence or {}).get("detected_text", "")
            if isinstance(paddle_confidence, dict)
            else ""
        ).strip(),
    }
    return out


def ocr_raw_paddle_then_vision(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    force_paddle_failure: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    call_timeout = int(timeout or _cfg_int("call_timeout_sec", 90))
    min_conf = _cfg_float("min_confidence_for_accept", 0.82)

    _vprint(verbose, f"start OCR paddle_then_vision image={Path(path).resolve()}")

    stages: list[dict[str, Any]] = []
    image_meta: dict[str, Any] = {}

    paddle_stage, paddle_full = _run_paddle_stage(
        path,
        verbose=verbose,
        force_failure=force_paddle_failure,
    )
    stages.append(paddle_stage)

    if paddle_stage.get("accepted"):
        chosen = paddle_stage
        _vprint(verbose, "paddle accepted; skipping vision fallback")
    else:
        vision_kwargs = {}
        if str(prompt_override or "").strip():
            vision_kwargs["prompt_override"] = prompt_override
        vision_stage, image_meta = _run_vision_stage(
            path,
            timeout=call_timeout,
            min_conf=min_conf,
            verbose=verbose,
            stage_index=1,
            **vision_kwargs,
        )
        stages.append(vision_stage)
        if vision_stage.get("error") and not vision_stage.get("parsed"):
            chosen = paddle_stage
        else:
            chosen = vision_stage

    out = _build_output(
        chosen=chosen,
        stages=stages,
        min_conf=min_conf,
        image_encode_meta=image_meta,
        paddle_full_out=paddle_full,
    )
    _vprint(
        verbose,
        f"final selected_model={out.get('selected_model')} selected_stage_index={out.get('selected_stage_index')} "
        f"needs_human_review={out.get('needs_human_review')}",
    )
    return out


def ocr_raw_paddle_only(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    force_paddle_failure: bool = False,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    min_conf = _cfg_float("min_confidence_for_accept", 0.82)
    _vprint(verbose, f"start OCR paddle_only image={Path(path).resolve()}")

    paddle_stage, paddle_full = _run_paddle_stage(
        path,
        verbose=verbose,
        force_failure=force_paddle_failure,
    )
    out = _build_output(
        chosen=paddle_stage,
        stages=[paddle_stage],
        min_conf=min_conf,
        image_encode_meta={},
        paddle_full_out=paddle_full,
    )
    _vprint(
        verbose,
        f"final selected_model={out.get('selected_model')} selected_stage_index={out.get('selected_stage_index')} "
        f"needs_human_review={out.get('needs_human_review')}",
    )
    return out


def ocr_raw(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    force_paddle_failure: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    workflow = _workflow_default()
    if workflow in {"vision_only", "vision"}:
        return ocr_raw_vision_only(
            image_path,
            timeout=timeout,
            verbose=verbose,
            prompt_override=prompt_override,
        )
    if workflow in {
        "vision_with_paddle_confidence",
        "vlm_paddle_confidence",
        "vision_paddle_confidence",
        "vlm_with_paddle_confidence",
    }:
        return ocr_raw_vision_with_paddle_confidence(
            image_path,
            timeout=timeout,
            verbose=verbose,
            force_paddle_failure=force_paddle_failure,
            prompt_override=prompt_override,
        )
    if workflow in {
        "paddle_vlm_fusion",
        "vlm_paddle_fusion",
        "vision_paddle_fusion",
        "paddle_vision_fusion",
    }:
        return ocr_raw_paddle_vlm_fusion(
            image_path,
            timeout=timeout,
            verbose=verbose,
            force_paddle_failure=force_paddle_failure,
            prompt_override=prompt_override,
        )
    if workflow in {"paddle_only", "paddle"}:
        return ocr_raw_paddle_only(
            image_path,
            timeout=timeout,
            verbose=verbose,
            force_paddle_failure=force_paddle_failure,
        )
    return ocr_raw_paddle_then_vision(
        image_path,
        timeout=timeout,
        verbose=verbose,
        force_paddle_failure=force_paddle_failure,
        prompt_override=prompt_override,
    )


def ocr(image_path: str | Path) -> str:
    out = ocr_raw(image_path)
    return str(out.get("detected_text", "") or "").strip()


def ocr_confidence_stats(raw_result: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_result or {}
    if not isinstance(raw, dict):
        raw = {}
    score = _coerce_confidence_score(raw if isinstance(raw, dict) else {})
    label = _score_to_label(score)

    rec_scores_raw = raw.get("rec_scores")
    rec_scores: list[float] = []
    if isinstance(rec_scores_raw, list):
        for s in rec_scores_raw:
            try:
                rec_scores.append(float(s))
            except (TypeError, ValueError):
                pass

    if rec_scores:
        min_rec = float(min(rec_scores))
        mean_rec = float(sum(rec_scores) / len(rec_scores))
    else:
        min_rec = score
        mean_rec = score
        rec_scores = [score]

    rec_texts_raw = raw.get("rec_texts")
    rec_texts: list[str] = []
    if isinstance(rec_texts_raw, list):
        rec_texts = [str(t) for t in rec_texts_raw if str(t).strip()]
    if not rec_texts:
        text = str(raw.get("detected_text", "") or "")
        rec_texts = [text] if text else []

    out = {
        "min_rec_score": min_rec,
        "mean_rec_score": mean_rec,
        "rec_scores": rec_scores,
        "rec_texts": rec_texts,
        "confidence_label": label,
        "needs_human_review": bool(score < _cfg_float("min_confidence_for_accept", 0.82)),
        "selected_model": raw.get("selected_model"),
        "selected_stage_index": raw.get("selected_stage_index"),
    }
    if "confidence_score" in raw:
        out["confidence_score"] = score
    if "confidence_source" in raw:
        out["confidence_source"] = raw.get("confidence_source")
    if "text_source" in raw:
        out["text_source"] = raw.get("text_source")
    if "workflow" in raw:
        out["workflow"] = raw.get("workflow")
    if isinstance(raw.get("paddle_confidence"), dict):
        out["paddle_confidence"] = raw.get("paddle_confidence")
    return out
