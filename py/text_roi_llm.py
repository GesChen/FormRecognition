"""
Post-processing LLM module for non-header text ROIs.

Workflow:
1) OCR extracts raw text from ROI crops using the default OCR prompt.
2) This module applies optional per-ROI LLM prompts/instructions on that text.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from config import LLM, TEXT_ROI_LLM
from llm_client import generate
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, desc=None, **kwargs):
        return iterable


def _cfg() -> dict[str, Any]:
    raw = TEXT_ROI_LLM if isinstance(TEXT_ROI_LLM, dict) else {}
    if raw:
        return raw
    # Backward-compatible fallback for older config layout.
    legacy = LLM.get("text_roi_postprocess") if isinstance(LLM, dict) else {}
    return legacy if isinstance(legacy, dict) else {}


def _enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def _timeout_sec() -> int:
    try:
        return int(_cfg().get("timeout_sec", 120))
    except Exception:
        return 120


def _max_reruns() -> int:
    try:
        return max(0, int(_cfg().get("max_reruns", 3)))
    except Exception:
        return 3


def _model() -> str | None:
    m = str(_cfg().get("model", "") or "").strip()
    return m or None


def _extra_params() -> dict[str, Any]:
    raw = _cfg().get("extra_params", {"think": False})
    return raw if isinstance(raw, dict) else {"think": False}


def _meta_str(meta: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        s = str(value).strip()
        if s:
            return s
    return ""


def _meta_value(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in meta:
            return meta.get(key)
    return None


def _meta_obj(meta: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    value = _meta_value(meta, *keys)
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _parse_format_override(value: Any) -> Any:
    """
    Optional structured-output format override for llm_client.generate.
    Accepts either:
      - "json" string
      - JSON-schema dict
      - JSON string that decodes to a dict
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s == "json":
            return "json"
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None
    return None


def _force_temperature_zero(extra_params: dict[str, Any]) -> dict[str, Any]:
    """
    Enforce deterministic decoding for normalization/extraction tasks.
    """
    out = dict(extra_params or {})
    opts = out.get("options")
    if not isinstance(opts, dict):
        opts = {}
    opts = dict(opts)
    opts["temperature"] = 0
    out["options"] = opts
    return out


def _merge_extra_params(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base or {})
    if not isinstance(override, dict):
        return merged
    for k, v in override.items():
        if isinstance(merged.get(k), dict) and isinstance(v, dict):
            tmp = dict(merged.get(k) or {})
            tmp.update(v)
            merged[k] = tmp
        else:
            merged[k] = v
    return merged


def _parse_llm_json(text: str) -> Any:
    s = str(text or "").strip()
    if not s:
        return {}
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        return {}
    tail = s[min(starts) :]
    try:
        obj, _end = decoder.raw_decode(tail)
        return obj
    except json.JSONDecodeError:
        return {}


def default_text_roi_llm_prompt_template() -> str:
    """
    Human-readable base template used by the LLM postprocessor for one ROI.
    """
    return (
        "You are a strict text normalizer for one OCR ROI field.\n"
        "You are NOT reading an image. You only receive OCR-extracted text.\n"
        "Return ONE valid JSON object only, no markdown and no extra text.\n"
        "Required schema:\n"
        "{\n"
        '  "detected_text": "string"\n'
        "}\n"
        "Rules:\n"
        "- Preserve meaning from OCR text.\n"
        "- Apply only conservative cleanup (trim/collapse whitespace, obvious punctuation cleanup).\n"
        "- Do not invent unseen content.\n"
        "- Use null when value is missing/unclear.\n"
        "ROI name: <roi_name>\n"
        "Field data type: <field_data_type>\n"
        "Validation rules: <validation_rules>\n"
        "Task instruction: <llm_prompt_instruction>\n"
        "OCR extracted text:\n"
        "<ocr_text>\n"
    )


def _build_default_prompt(
    *,
    roi_name: str,
    field_data_type: str,
    validation_rules: str,
    instruction: str,
    ocr_text: str,
) -> str:
    rules = validation_rules if validation_rules else "(none)"
    instr = instruction if instruction else "Normalize the OCR text conservatively."
    return (
        "You are a strict text normalizer for one OCR ROI field.\n"
        "You are NOT reading an image. You only receive OCR-extracted text.\n"
        "Return ONE valid JSON object only, no markdown and no extra text.\n"
        "Required schema:\n"
        "{\n"
        '  "detected_text": "string"\n'
        "}\n"
        "Rules:\n"
        "- Preserve meaning from OCR text.\n"
        "- Apply conservative cleanup only.\n"
        "- Do not invent unseen content.\n"
        "- Use null when value is missing/unclear.\n"
        f"ROI name: {roi_name}\n"
        f"Field data type: {field_data_type or 'text'}\n"
        f"Validation rules: {rules}\n"
        f"Task instruction: {instr}\n\n"
        "OCR extracted text:\n"
        f"{ocr_text}\n"
    )


def _build_prompt(
    *,
    roi_name: str,
    field_data_type: str,
    validation_rules: str,
    instruction: str,
    prompt_override: str,
    ocr_text: str,
) -> str:
    if prompt_override:
        return (
            "You are a strict text normalizer for one OCR ROI field.\n"
            "You are NOT reading an image. You only receive OCR-extracted text.\n"
            "Apply the custom instructions below.\n"
            "Return ONE valid JSON object only, no markdown and no extra text.\n"
            "Required schema:\n"
            "{\n"
            '  "detected_text": "string"\n'
            "}\n"
            "Custom instructions:\n"
            f"{prompt_override}\n\n"
            f"ROI name: {roi_name}\n"
            f"Field data type: {field_data_type or 'text'}\n"
            f"Validation rules: {validation_rules or '(none)'}\n"
            f"Creator instruction: {instruction or '(none)'}\n\n"
            "OCR extracted text:\n"
            f"{ocr_text}\n"
        )
    return _build_default_prompt(
        roi_name=roi_name,
        field_data_type=field_data_type,
        validation_rules=validation_rules,
        instruction=instruction,
        ocr_text=ocr_text,
    )


def _coerce_detected_text(parsed: Any, raw_text: str, fallback: str) -> str:
    if isinstance(parsed, dict):
        if "detected_text" in parsed:
            value = parsed.get("detected_text")
            if value is None:
                return ""
            return " ".join(str(value).split())
    s = str(raw_text or "").strip()
    if not s:
        return fallback
    return " ".join(s.split())


def _is_conforming_detected_text_json(parsed: Any) -> bool:
    """
    Strict conformance check for required output schema:
      {"detected_text": "string or null"}
    """
    if not isinstance(parsed, dict):
        return False
    if "detected_text" not in parsed:
        return False
    val = parsed.get("detected_text")
    return val is None or isinstance(val, str)


def postprocess_text_rois(
    text_by_roi: Dict[str, Any],
    *,
    roi_meta_by_name: Dict[str, Dict[str, Any]] | None = None,
    debug_out: Dict[str, Any] | None = None,
    use_tqdm: bool = False,
    tqdm_desc: str = "      Text LLM deferred",
) -> Dict[str, str]:
    """
    Postprocess non-header text ROIs using per-ROI LLM prompts.

    Returns a dict with the same keys as input.
    Falls back to original OCR text on errors or unparseable responses.
    """
    src: dict[str, str] = {
        str(k): " ".join(str(v or "").split())
        for k, v in (text_by_roi or {}).items()
    }
    if not src:
        return {}
    if not _enabled():
        if debug_out is not None:
            debug_out["enabled"] = False
            debug_out["skipped"] = "disabled"
        return src

    result = dict(src)
    all_debug_rows: list[dict[str, Any]] = []
    rows_iter = list(src.items())
    if use_tqdm and rows_iter:
        rows_iter = tqdm(
            rows_iter,
            total=len(rows_iter),
            desc=tqdm_desc,
            unit="roi",
            dynamic_ncols=True,
            leave=False,
        )

    for name, ocr_text in rows_iter:
        meta = (roi_meta_by_name or {}).get(name) or {}
        field_data_type = _meta_str(meta, "llm_field_data_type", "ocr_field_data_type")
        validation_rules = _meta_str(meta, "llm_validation_rules", "ocr_validation_rules")
        instruction = _meta_str(meta, "llm_prompt_instruction", "ocr_prompt_instruction")
        prompt_override = _meta_str(meta, "llm_prompt_override", "ocr_prompt_override")
        per_roi_extra = _merge_extra_params(
            _extra_params(),
            _meta_obj(meta, "llm_extra_params", "ocr_extra_params"),
        )
        format_override = _parse_format_override(
            _meta_value(meta, "llm_response_format", "ocr_response_format", "llm_format", "ocr_format")
        )
        if format_override is not None:
            per_roi_extra["format"] = format_override
        options_override = _meta_obj(meta, "llm_options", "ocr_options")
        if options_override:
            per_roi_extra = _merge_extra_params(per_roi_extra, {"options": options_override})
        per_roi_extra = _force_temperature_zero(per_roi_extra)
        prompt = _build_prompt(
            roi_name=name,
            field_data_type=field_data_type,
            validation_rules=validation_rules,
            instruction=instruction,
            prompt_override=prompt_override,
            ocr_text=ocr_text,
        )

        attempts: list[dict[str, Any]] = []
        final_value = ocr_text
        for attempt in range(_max_reruns() + 1):
            out = generate(
                prompt,
                model=_model(),
                timeout=_timeout_sec(),
                extra_params=per_roi_extra,
            )
            raw_text = str(out.get("text", "") or "")
            parsed = _parse_llm_json(raw_text)
            parsed_ok = _is_conforming_detected_text_json(parsed)
            candidate = _coerce_detected_text(parsed, raw_text, fallback=ocr_text) if parsed_ok else ocr_text
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "parsed_ok": bool(parsed_ok),
                    "elapsed": out.get("elapsed"),
                    "eval_count": out.get("eval_count"),
                    "eval_duration": out.get("eval_duration"),
                    "raw_response": raw_text,
                }
            )
            final_value = candidate
            if parsed_ok:
                break

        result[name] = final_value
        all_debug_rows.append(
            {
                "name": name,
                "prompt": prompt,
                "used_prompt_override": bool(prompt_override),
                "field_data_type": field_data_type or None,
                "llm_extra_params": per_roi_extra,
                "attempts": attempts,
                "result": final_value,
            }
        )

    if debug_out is not None:
        debug_out["enabled"] = True
        debug_out["model"] = _model() or LLM.get("model")
        debug_out["per_roi"] = all_debug_rows
        debug_out["result"] = dict(result)
    return result
