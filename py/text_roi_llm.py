"""
Post-processing LLM module for non-header text ROIs.

Workflow:
1) OCR extracts raw text from ROI crops using the default OCR prompt.
2) This module applies optional per-ROI LLM prompts/instructions on that text.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import textwrap
import traceback
from typing import Any, Dict

from config import LLM, LLM_POSTPROCESS, SUPPORTED_LLM_MODELS, TEXT_ROI_LLM
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
    global_cfg = LLM_POSTPROCESS if isinstance(LLM_POSTPROCESS, dict) else {}
    return bool(global_cfg.get("enabled", True)) and bool(_cfg().get("enabled", True))


def _timeout_sec() -> int:
    try:
        return int(_cfg().get("timeout_sec", 120))
    except Exception:
        return 120


def _code_timeout_sec() -> float:
    try:
        return max(0.1, float(_cfg().get("code_timeout_sec", 2.0)))
    except Exception:
        return 2.0


def _max_reruns() -> int:
    try:
        return max(0, int(_cfg().get("max_reruns", 3)))
    except Exception:
        return 3


def _model() -> str | None:
    m = str(_cfg().get("model", "") or "").strip()
    return m or None


def _supported_llm_models() -> set[str]:
    return {
        str(item).strip()
        for item in (SUPPORTED_LLM_MODELS if isinstance(SUPPORTED_LLM_MODELS, list) else [])
        if str(item).strip()
    }


def _pass_model_selection(pass_cfg: dict[str, Any], fallback: str | None) -> dict[str, Any]:
    raw = str(pass_cfg.get("model") or "").strip()
    fallback_model = str(fallback or LLM.get("model") or "").strip() or None
    if not raw:
        return {
            "requested_model": None,
            "fallback_model": fallback_model,
            "selected_model": fallback_model,
            "source": "fallback",
            "fallback_reason": "no_pass_model",
        }
    supported = _supported_llm_models()
    if supported and raw not in supported:
        return {
            "requested_model": raw,
            "fallback_model": fallback_model,
            "selected_model": fallback_model,
            "source": "fallback",
            "fallback_reason": "unsupported_pass_model",
        }
    return {
        "requested_model": raw,
        "fallback_model": fallback_model,
        "selected_model": raw,
        "source": "pass",
        "fallback_reason": None,
    }


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


def _looks_internal_uid(value: Any) -> bool:
    return re.fullmatch(r"p\d+:r\d+:[^:\s]+", str(value or "").strip()) is not None


def _candidate_echoes_numeric_roi_name(
    *,
    candidate: str,
    roi_name: str,
    ocr_text: str,
) -> bool:
    value = str(candidate or "").strip()
    prompt_name = str(roi_name or "").strip()
    if not value or value != prompt_name or not re.fullmatch(r"\d{1,3}", prompt_name):
        return False
    return re.search(rf"(?<!\d){re.escape(value)}(?!\d)", str(ocr_text or "")) is None


def _paddle_sidecar(meta: dict[str, Any]) -> dict[str, Any] | None:
    raw = _meta_value(
        meta,
        "ocr_paddle_confidence",
        "paddle_confidence",
        "_ocr_paddle_confidence",
    )
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
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
        "<input_text>\n"
    )


def _build_default_prompt(
    *,
    roi_name: str,
    field_data_type: str,
    validation_rules: str,
    instruction: str,
    input_text: str,
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
        "Input text:\n"
        f"{input_text}\n"
    )


_INPUT_TEXT_PLACEHOLDERS = ("<input_text>", "<INPUT_TEXT>", "{{input_text}}")


def _inject_input_text(prompt_override: str, input_text: str) -> tuple[str, bool]:
    """Allow prompt overrides to choose exactly where pass input text is inserted."""
    prompt = str(prompt_override or "")
    injected = False
    for placeholder in _INPUT_TEXT_PLACEHOLDERS:
        if placeholder in prompt:
            prompt = prompt.replace(placeholder, input_text)
            injected = True
    return prompt, injected


def _build_prompt(
    *,
    roi_name: str,
    field_data_type: str,
    validation_rules: str,
    instruction: str,
    prompt_override: str,
    input_text: str,
) -> str:
    if prompt_override:
        custom_instructions, _injected_input_text = _inject_input_text(prompt_override, input_text)
        return custom_instructions
    return _build_default_prompt(
        roi_name=roi_name,
        field_data_type=field_data_type,
        validation_rules=validation_rules,
        instruction=instruction,
        input_text=input_text,
    )


def _normalize_postprocess_pass(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        text = value.strip()
        return {"type": "llm", "prompt": text} if text else None
    if not isinstance(value, dict):
        return None
    pass_type = str(value.get("type") or value.get("kind") or "llm").strip().lower()
    if pass_type == "code":
        code = str(value.get("code") or value.get("snippet") or "").strip()
        return {"type": "code", "code": code} if code else None
    prompt = str(value.get("prompt") or value.get("text") or "").strip()
    if not prompt:
        return None
    out: dict[str, Any] = {"type": "llm", "prompt": prompt}
    model = str(value.get("model") or value.get("llm_model") or "").strip()
    if model:
        out["model"] = model
    return out


def _postprocess_passes_from_meta(meta: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """
    Return configured postprocess passes for a ROI.

    New ROI schemas store `postprocess_passes`, a list of typed pass objects.
    Older string-only pass lists remain readable as LLM passes.
    """
    if "postprocess_passes" in meta:
        raw_passes = meta.get("postprocess_passes")
        if isinstance(raw_passes, list):
            passes = [
                p for p in (_normalize_postprocess_pass(item) for item in raw_passes)
                if p is not None
            ]
            return passes, "postprocess_passes"
        return [], "postprocess_passes"
    if "llm_passes" in meta:
        raw_passes = meta.get("llm_passes")
        if isinstance(raw_passes, list):
            passes = [
                p for p in (_normalize_postprocess_pass(item) for item in raw_passes)
                if p is not None
            ]
            return passes, "legacy_llm_passes"
        return [], "legacy_llm_passes"
    prompt_override = _meta_str(meta, "llm_prompt_override")
    if prompt_override:
        return [{"type": "llm", "prompt": prompt_override}], "legacy_llm_prompt_override"
    return [], "none"


_CODE_ALLOWED_BUILTINS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "sorted": sorted,
}


def _code_worker(conn: Any, code: str, input_text: str, roi_name: str, meta: dict[str, Any]) -> None:
    try:
        globals_dict: dict[str, Any] = {
            "__builtins__": _CODE_ALLOWED_BUILTINS,
            "re": re,
            "input_text": input_text,
            "roi_name": roi_name,
            "meta": meta,
        }
        locals_dict: dict[str, Any] = {}
        try:
            result = eval(compile(code, "<postprocess_pass>", "eval"), globals_dict, locals_dict)
        except SyntaxError:
            fn_src = "def __postprocess_fn__():\n" + textwrap.indent(code, "    ")
            exec(compile(fn_src, "<postprocess_pass>", "exec"), globals_dict, locals_dict)
            result = locals_dict["__postprocess_fn__"]()
        conn.send({"ok": True, "result": result})
    except Exception:
        conn.send({"ok": False, "error": traceback.format_exc(limit=3)})
    finally:
        conn.close()


def _run_code_pass(
    *,
    code: str,
    input_text: str,
    roi_name: str,
    meta: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    try:
        ctx = mp.get_context("fork")
    except ValueError:  # pragma: no cover - platform dependent
        ctx = mp.get_context()
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_code_worker,
        args=(child_conn, code, input_text, roi_name, dict(meta or {})),
    )
    proc.start()
    child_conn.close()
    proc.join(_code_timeout_sec())
    if proc.is_alive():
        proc.terminate()
        proc.join(0.5)
        return False, input_text, {"error": "code_pass_timeout", "timeout_sec": _code_timeout_sec()}
    if not parent_conn.poll():
        return False, input_text, {"error": "code_pass_no_result", "exitcode": proc.exitcode}
    msg = parent_conn.recv()
    if not isinstance(msg, dict) or not msg.get("ok"):
        return False, input_text, {"error": (msg or {}).get("error", "code_pass_error")}
    value = msg.get("result")
    if value is None:
        return True, "", {"raw_result": None}
    return True, " ".join(str(value).split()), {"raw_result": value}


def _single_output_value(parsed: Any) -> tuple[bool, Any]:
    if not isinstance(parsed, dict):
        return False, None
    if "detected_text" in parsed:
        return True, parsed.get("detected_text")
    if len(parsed) == 1:
        return True, next(iter(parsed.values()))
    return False, None


def _coerce_detected_text(parsed: Any, raw_text: str, fallback: str) -> str:
    if parsed is None:
        return ""
    if isinstance(parsed, dict):
        has_value, value = _single_output_value(parsed)
        if has_value:
            if value is None:
                return ""
            normalized = " ".join(str(value).split())
            if normalized.strip().lower() in {"null", "none", "n/a", "na", "unknown", "unreadable"}:
                return ""
            return normalized
    s = str(raw_text or "").strip()
    if not s:
        return fallback
    return " ".join(s.split())


def _regex_mismatch(candidate: str, pattern: str) -> bool:
    if not pattern or not candidate:
        return False
    try:
        return re.fullmatch(pattern, candidate) is None
    except re.error:
        return False


def _is_conforming_detected_text_json(parsed: Any) -> bool:
    """
    Conformance check for an LLM JSON scalar response.

    Preferred shape is {"detected_text": "string or null"}, but schema-specific
    prompts may return a single alternate key such as {"date": "..."} or
    {"age": 12}. Treat any one-key JSON object as the extracted value.
    """
    if parsed is None:
        return True
    has_value, value = _single_output_value(parsed)
    if not has_value:
        return False
    return value is None or isinstance(value, (str, int, float, bool))


def _first_llm_model_for_meta(meta: dict[str, Any]) -> str | None:
    postprocess_passes, _pass_source = _postprocess_passes_from_meta(meta)
    model_override = _meta_str(meta, "llm_model") or _model()
    for pass_cfg in postprocess_passes:
        pass_type = str(pass_cfg.get("type") or "llm").strip().lower()
        if pass_type != "llm":
            continue
        selected = _pass_model_selection(pass_cfg, model_override).get("selected_model")
        selected_model = str(selected or "").strip()
        return selected_model or None
    return None


def _group_roi_items_by_first_llm_model(
    items: list[tuple[str, str]],
    *,
    roi_meta_by_name: Dict[str, Dict[str, Any]] | None,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    batches: list[dict[str, Any]] = []
    batch_index_by_model: dict[str, int] = {}
    for name, _ocr_text in items:
        meta = (roi_meta_by_name or {}).get(name) or {}
        model = _first_llm_model_for_meta(meta)
        key = model or ""
        if key not in batch_index_by_model:
            batch_index_by_model[key] = len(batches)
            batches.append({"model": model, "roi_names": []})
        batches[batch_index_by_model[key]]["roi_names"].append(name)

    item_by_name = {name: (name, ocr_text) for name, ocr_text in items}
    grouped_items = [
        item_by_name[name]
        for batch in batches
        for name in batch.get("roi_names", [])
        if name in item_by_name
    ]
    for batch in batches:
        batch["roi_count"] = len(batch.get("roi_names", []))
    return grouped_items, batches


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
    rows_iter, call_batches = _group_roi_items_by_first_llm_model(
        list(src.items()),
        roi_meta_by_name=roi_meta_by_name,
    )
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
        prompt_roi_name = _meta_str(meta, "roi_name", "name") or name
        field_data_type = _meta_str(meta, "llm_field_data_type")
        validation_rules = _meta_str(meta, "llm_validation_rules")
        output_regex = _meta_str(meta, "output_regex", "ocr_output_regex")
        instruction = _meta_str(meta, "llm_prompt_instruction")
        postprocess_passes, pass_source = _postprocess_passes_from_meta(meta)
        model_override = _meta_str(meta, "llm_model") or _model()
        paddle_sidecar = _paddle_sidecar(meta)
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
        attempts: list[dict[str, Any]] = []
        pass_debug: list[dict[str, Any]] = []
        final_value = ocr_text
        prompts_used: list[str] = []
        for pass_index, pass_cfg in enumerate(postprocess_passes, start=1):
            pass_type = str(pass_cfg.get("type") or "llm").strip().lower()
            input_before_pass = final_value
            if pass_type == "code":
                ok, candidate, details = _run_code_pass(
                    code=str(pass_cfg.get("code") or ""),
                    input_text=final_value,
                    roi_name=prompt_roi_name,
                    meta=meta,
                )
                final_value = candidate if ok else final_value
                pass_debug.append(
                    {
                        "pass": pass_index,
                        "type": "code",
                        "input": input_before_pass,
                        "ok": bool(ok),
                        "details": details,
                        "result": final_value,
                    }
                )
                continue
            model_selection = _pass_model_selection(pass_cfg, model_override)
            selected_model = model_selection.get("selected_model")
            prompt = _build_prompt(
                roi_name=prompt_roi_name,
                field_data_type=field_data_type,
                validation_rules=validation_rules,
                instruction=instruction,
                prompt_override=str(pass_cfg.get("prompt") or ""),
                input_text=final_value,
            )
            prompts_used.append(prompt)
            for attempt in range(_max_reruns() + 1):
                out = generate(
                    prompt,
                    model=selected_model,
                    timeout=_timeout_sec(),
                    extra_params=per_roi_extra,
                )
                raw_text = str(out.get("text", "") or "")
                parsed = _parse_llm_json(raw_text)
                schema_ok = _is_conforming_detected_text_json(parsed)
                candidate = _coerce_detected_text(parsed, raw_text, fallback=final_value) if schema_ok else final_value
                rejection_reason = None
                if _looks_internal_uid(candidate):
                    candidate = ""
                    rejection_reason = "internal_uid"
                if _candidate_echoes_numeric_roi_name(
                    candidate=candidate,
                    roi_name=prompt_roi_name,
                    ocr_text=ocr_text,
                ):
                    candidate = ""
                    rejection_reason = "numeric_roi_name_echo"
                if _regex_mismatch(candidate, output_regex):
                    candidate = ""
                    rejection_reason = "output_regex_mismatch"
                parsed_ok = bool(schema_ok)
                attempts.append(
                    {
                        "pass": pass_index,
                        "type": "llm",
                        "model": selected_model,
                        "requested_model": model_selection.get("requested_model"),
                        "fallback_model": model_selection.get("fallback_model"),
                        "model_source": model_selection.get("source"),
                        "model_fallback_reason": model_selection.get("fallback_reason"),
                        "attempt": attempt + 1,
                        "parsed_ok": bool(parsed_ok),
                        "rejection_reason": rejection_reason,
                        "elapsed": out.get("elapsed"),
                        "eval_count": out.get("eval_count"),
                        "eval_duration": out.get("eval_duration"),
                        "raw_response": raw_text,
                    }
                )
                final_value = candidate
                if parsed_ok:
                    break
            pass_debug.append(
                {
                    "pass": pass_index,
                    "type": "llm",
                    "model": selected_model,
                    "requested_model": model_selection.get("requested_model"),
                    "fallback_model": model_selection.get("fallback_model"),
                    "model_source": model_selection.get("source"),
                    "model_fallback_reason": model_selection.get("fallback_reason"),
                    "input": input_before_pass,
                    "prompt": prompt,
                    "result": final_value,
                }
            )

        result[name] = final_value
        all_debug_rows.append(
            {
                "name": name,
                "roi_name": prompt_roi_name,
                "prompt": prompts_used[0] if prompts_used else None,
                "prompts": prompts_used,
                "postprocess_pass_count": len(postprocess_passes),
                "postprocess_pass_source": pass_source,
                "llm_pass_count": len([p for p in postprocess_passes if p.get("type") == "llm"]),
                "llm_prompt_source": pass_source,
                "used_prompt_override": bool(postprocess_passes),
                "used_paddle_vlm_fusion": False,
                "preserved_ocr_paddle_confidence": isinstance(paddle_sidecar, dict),
                "field_data_type": field_data_type or None,
                "llm_extra_params": per_roi_extra,
                "passes": pass_debug,
                "attempts": attempts,
                "result": final_value,
            }
        )

    if debug_out is not None:
        debug_out["enabled"] = True
        debug_out["model"] = _model() or LLM.get("model")
        debug_out["call_batches"] = call_batches
        debug_out["per_roi"] = all_debug_rows
        debug_out["result"] = dict(result)
    return result
