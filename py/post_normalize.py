"""
Batch post-normalization for selected ROI text fields.

This pass runs after ordinary ROI OCR/LLM cleanup. It sees repeated values for
configured fields across the whole run and asks the LLM to return normalized
values in the same order.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from math import ceil
from typing import Any, Mapping

from config import LLM, POST_NORMALIZE
from llm_client import generate

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, desc=None, **kwargs):
        return iterable


POST_NORMALIZE_MODES = {"casual", "aggressive", "consensus"}
LEGACY_POST_NORMALIZE_MODE_ALIASES = {
    "strict": "casual",
    "conservative": "casual",
    "balanced": "aggressive",
}
DEFAULT_POST_NORMALIZE_MODE = "aggressive"


def _cfg() -> dict[str, Any]:
    return POST_NORMALIZE if isinstance(POST_NORMALIZE, dict) else {}


def _enabled() -> bool:
    return bool(_cfg().get("enabled", True))


def _model() -> str | None:
    raw = str(_cfg().get("model", "") or "").strip()
    return raw or None


def _timeout_sec() -> int:
    try:
        return int(_cfg().get("timeout_sec", 180) or 180)
    except Exception:
        return 180


def _max_retries() -> int:
    try:
        return max(0, int(_cfg().get("max_retries", 3) or 0))
    except Exception:
        return 3


def _extra_params() -> dict[str, Any]:
    raw = _cfg().get("extra_params", {"think": False, "options": {"temperature": 0}})
    out = dict(raw) if isinstance(raw, dict) else {"think": False}
    opts = out.get("options")
    if not isinstance(opts, dict):
        opts = {}
    opts = dict(opts)
    opts["temperature"] = 0
    out["options"] = opts
    return out


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _looks_structured_identifier(value: str) -> bool:
    s = _norm_text(value)
    if not s:
        return True
    if re.fullmatch(r"\d+(?:[/-]\d+)*[A-Za-z]?", s):
        return True
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", s):
        return True
    if re.fullmatch(r"[A-Za-z]?\d+[A-Za-z]?", s):
        return True
    return False


def _looks_placeholder(value: str, roi_name: str | None = None) -> bool:
    s = _norm_text(value).strip().lower()
    if not s:
        return True
    if s in {"null", "none", "n/a", "na", "unknown", "unreadable"}:
        return True
    field = _norm_text(roi_name).strip().lower()
    if not field:
        return False
    field_words = " ".join(re.split(r"[_-]+", field)).strip()
    return s in {
        field,
        field_words,
        f"name of {field}",
        f"name of {field_words}",
        f"{field} name",
        f"{field_words} name",
    }


def _alpha_signature(value: str) -> str:
    return "".join(ch.lower() for ch in _norm_text(value) if ch.isalpha())


def _aggressive_cluster_map(values: list[str], roi_name: str) -> tuple[dict[str, str], dict[str, Any]]:
    counts = Counter(_norm_text(v) for v in values if _norm_text(v))
    detail: dict[str, Any] = {
        "enabled": True,
        "reason": None,
        "canonicals": [],
        "assignments": {},
    }
    canonicals = [
        value
        for value, count in counts.most_common()
        if count >= 2 and not _looks_placeholder(value, roi_name) and not _looks_structured_identifier(value)
    ]
    detail["canonicals"] = canonicals
    if not canonicals:
        detail["reason"] = "no_repeated_canonical"
        return {}, detail

    out: dict[str, str] = {}
    for value in sorted(counts):
        if _looks_placeholder(value, roi_name):
            out[value] = ""
            detail["assignments"][value] = {"canonical": None, "reason": "placeholder"}
            continue
        sig = _alpha_signature(value)
        if not sig:
            detail["assignments"][value] = {"canonical": None, "reason": "no_alpha_signature"}
            continue
        best: tuple[float, str] | None = None
        for canonical in canonicals:
            can_sig = _alpha_signature(canonical)
            if not can_sig:
                continue
            score = SequenceMatcher(None, sig, can_sig).ratio()
            if best is None or score > best[0]:
                best = (score, canonical)
        if best is None:
            continue
        score, canonical = best
        can_sig = _alpha_signature(canonical)
        substring_match = len(sig) >= 3 and (sig in can_sig or can_sig in sig)
        if score >= 0.72 or substring_match:
            out[value] = canonical
            detail["assignments"][value] = {
                "canonical": canonical,
                "score": round(score, 4),
                "reason": "similarity" if not substring_match else "substring",
            }
        else:
            detail["assignments"][value] = {
                "canonical": None,
                "score": round(score, 4),
                "reason": "below_threshold",
            }

    detail["reason"] = "ok"
    return out, detail


def _dominant_consensus_value(values: list[str], roi_name: str) -> tuple[str | None, dict[str, Any]]:
    """
    Pick a field-level consensus for consensus mode.

    This is intentionally blunt: consensus mode means the user expects this ROI to
    contain a very small number of true values and wants noisy singletons pulled
    toward the run-level agreement.
    """
    counts = Counter(_norm_text(v) for v in values if _norm_text(v))
    detail: dict[str, Any] = {
        "enabled": True,
        "reason": None,
        "canonical": None,
        "canonical_count": 0,
        "row_count": len(values),
        "min_count": max(2, min(3, ceil(len(values) * 0.12))),
    }
    if not counts:
        detail["reason"] = "no_values"
        return None, detail

    canonical, count = counts.most_common(1)[0]
    detail["canonical"] = canonical
    detail["canonical_count"] = count
    if count < int(detail["min_count"]):
        detail["reason"] = "no_clear_dominant_value"
        return None, detail
    if _looks_placeholder(canonical, roi_name):
        detail["reason"] = "dominant_value_looks_placeholder"
        return None, detail
    if _looks_structured_identifier(canonical):
        detail["reason"] = "dominant_value_looks_structured"
        return None, detail
    detail["reason"] = "clear_dominant_value"
    return canonical, detail


def _parse_jsonish(text: str) -> Any:
    s = str(text or "").strip()
    if not s:
        return {}
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        return {}
    try:
        obj, _ = decoder.raw_decode(s[min(starts):])
        return obj
    except json.JSONDecodeError:
        return {}


def _normalize_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode in POST_NORMALIZE_MODES:
        return mode
    if mode in LEGACY_POST_NORMALIZE_MODE_ALIASES:
        return LEGACY_POST_NORMALIZE_MODE_ALIASES[mode]
    return DEFAULT_POST_NORMALIZE_MODE


def _mode_guidance(mode: str) -> list[str]:
    mode = _normalize_mode(mode)
    if mode == "casual":
        return [
            "Use casual cleanup: remove obvious OCR artifacts, title remnants, spacing problems, and casing noise.",
            "Merge only near-identical values where the intended text is very clear from the strings themselves.",
            "Preserve different-looking names or values as separate; when uncertain, leave the original value unchanged.",
        ]
    if mode == "consensus":
        return [
            "Use single-consensus normalization: this field is expected to have one true value across the run.",
            "Use only the observed strings and counts; do not use external knowledge.",
            "Prefer the most frequent exact observed value as the canonical value when it is not a placeholder or structured identifier.",
            "Collapse plausible noisy spellings, abbreviations, partial reads, and artifacts to that one consensus value.",
        ]
    return [
        "Use aggressive multi-consensus normalization: identify one or more consensus clusters from the observed strings.",
        "Use only the observed strings and counts; do not use external knowledge.",
        "Within each cluster, prefer the most frequent exact observed value as the canonical spelling.",
        "Merge plausible noisy spellings, abbreviations, partial reads, and label-stripped variants into their cluster canonical.",
        "Keep clearly different names or values in separate clusters; do not force every value into one cluster.",
        "Treat obvious labels/placeholders/artifacts as null or map them only when they clearly belong to a cluster from the observed strings.",
    ]


def _attempt_guidance(attempt_index: int, mode: str) -> str:
    variants = [
        " ".join(_mode_guidance(mode)),
        (
            "Second pass: focus on returning exactly the same object keys as the input. "
            + " ".join(_mode_guidance(mode))
        ),
        (
            "Retry pass: output must be only a JSON object with exactly the input keys. "
            + " ".join(_mode_guidance(mode))
        ),
    ]
    return variants[min(attempt_index, len(variants) - 1)]


def _build_prompt(*, form_type: str, roi_name: str, values: list[str], attempt_index: int, mode: str) -> str:
    counts = Counter(values)
    unique_values = sorted(counts)
    return (
        "You normalize repeated OCR/handwriting text values for one data field.\n"
        "You are NOT reading an image. You only receive OCR-extracted strings.\n"
        "Use only the provided strings and their counts; do not use external knowledge.\n"
        "Return ONE valid JSON object only, no markdown and no extra text.\n"
        "The JSON object MUST map every input string key to its normalized replacement string.\n"
        "The output object MUST contain exactly the same keys as the input object.\n"
        "Rules:\n"
        "- Each output value must be the normalized replacement for its exact input key.\n"
        "- Never output an empty string. If preserving a value, repeat the original input key as the value.\n"
        "- Use null for labels/placeholders/artifacts that should not become real data.\n"
        "- Keep genuinely different values distinct.\n"
        "- Do not invent a value unrelated to the inputs.\n"
        "- If a value is already clean or cannot be improved, return it unchanged.\n"
        f"- Normalization mode: {_normalize_mode(mode)}.\n"
        f"- {_attempt_guidance(attempt_index, mode)}\n"
        f"Form type: {form_type}\n"
        f"ROI field name: {roi_name}\n"
        "Observed value counts JSON object:\n"
        f"{json.dumps(dict(sorted(counts.items())), ensure_ascii=False)}\n"
        "Input JSON object whose keys are the unique observed strings and whose values are placeholders to replace:\n"
        f"{json.dumps({value: '<normalize>' for value in unique_values}, ensure_ascii=False)}\n"
    )


def _coerce_normalized_mapping(parsed: Any, expected_keys: set[str]) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(parsed, dict):
        return None, "not_object"
    keys = set(str(k) for k in parsed.keys())
    if not (keys & expected_keys):
        return None, "no_expected_keys"
    out: dict[str, str] = {}
    for key in sorted(expected_keys):
        if key not in parsed:
            out[key] = key
            continue
        value = parsed.get(key)
        if value is not None and not isinstance(value, str):
            return None, f"non_string:{key}"
        normalized = _norm_text(value) if value is not None else ""
        nullish_string = normalized.lower() in {"null", "none", "n/a", "na"}
        if nullish_string:
            normalized = ""
        if value is not None and not normalized and not nullish_string:
            return None, f"empty_string:{key}"
        out[key] = normalized
    return out, None


def _enabled_for(enabled_fields: Mapping[str, set[str]] | set[tuple[str, str]], form_type: str, roi_name: str) -> bool:
    if isinstance(enabled_fields, set):
        return (form_type, roi_name) in enabled_fields
    names = enabled_fields.get(form_type) or set()
    if isinstance(names, Mapping):
        return roi_name in names
    return roi_name in names


def _field_mode(enabled_fields: Mapping[str, Any] | set[tuple[str, str]], form_type: str, roi_name: str) -> str:
    if isinstance(enabled_fields, set):
        return DEFAULT_POST_NORMALIZE_MODE
    names = enabled_fields.get(form_type) or {}
    if isinstance(names, Mapping):
        raw = names.get(roi_name)
        if isinstance(raw, Mapping):
            return _normalize_mode(raw.get("mode"))
        return _normalize_mode(raw)
    return DEFAULT_POST_NORMALIZE_MODE


def normalize_items(
    items: list[dict[str, Any]],
    enabled_fields: Mapping[str, Any] | set[tuple[str, str]],
    *,
    debug_out: dict[str, Any] | None = None,
    use_tqdm: bool = False,
    tqdm_desc: str = "      Post normalize",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (normalized_items, original_items).

    ``enabled_fields`` is scoped by form type and ROI name. Only text rows matching
    that scope are candidates for this batch pass.
    """
    original_items = copy.deepcopy(items or [])
    normalized_items = copy.deepcopy(items or [])

    status: dict[str, Any] = {
        "enabled": bool(_enabled()),
        "model": _model() or LLM.get("model"),
        "max_retries": _max_retries(),
        "groups": [],
        "applied_count": 0,
        "skipped_count": 0,
    }
    if debug_out is not None:
        debug_out.update(status)

    if not items:
        status["skipped_reason"] = "no_items"
        if debug_out is not None:
            debug_out.update(status)
        return normalized_items, original_items
    if not _enabled():
        status["skipped_reason"] = "disabled"
        if debug_out is not None:
            debug_out.update(status)
        return normalized_items, original_items

    row_refs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in normalized_items:
        if not isinstance(item, dict):
            continue
        form_type = _norm_text(item.get("form_type"))
        if not form_type:
            continue
        for row in item.get("data") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("kind", "")).strip().lower() != "text":
                continue
            roi_name = _norm_text(row.get("name"))
            if not roi_name or not _enabled_for(enabled_fields, form_type, roi_name):
                continue
            text = _norm_text(row.get("text"))
            if not text:
                continue
            row_refs[(form_type, roi_name)].append(row)

    if not row_refs:
        status["skipped_reason"] = "no_matching_values"
        if debug_out is not None:
            debug_out.update(status)
        return normalized_items, original_items

    group_items = sorted(row_refs.items())
    group_iter = group_items
    if use_tqdm and group_items:
        group_iter = tqdm(group_items, total=len(group_items), desc=tqdm_desc, unit="group", dynamic_ncols=True, leave=False)

    for (form_type, roi_name), rows in group_iter:
        values = [_norm_text(row.get("text")) for row in rows]
        counts = Counter(values)
        mode = _field_mode(enabled_fields, form_type, roi_name)
        expected_keys = set(values)
        group_debug: dict[str, Any] = {
            "form_type": form_type,
            "roi_name": roi_name,
            "mode": mode,
            "unique_value_count": len(counts),
            "row_count": len(values),
            "input_values": values,
            "input_unique_values": sorted(expected_keys),
            "value_counts": dict(sorted(counts.items())),
            "attempts": [],
            "applied": [],
            "skipped": [],
            "error": None,
        }
        status["groups"].append(group_debug)
        normalized_map: dict[str, str] | None = None

        for attempt_index in range(_max_retries() + 1):
            prompt = _build_prompt(
                form_type=form_type,
                roi_name=roi_name,
                values=values,
                attempt_index=attempt_index,
                mode=mode,
            )
            attempt_debug: dict[str, Any] = {
                "attempt_index": attempt_index + 1,
                "prompt": prompt,
                "raw_response": None,
                "elapsed": None,
                "valid": False,
                "error": None,
            }
            group_debug["attempts"].append(attempt_debug)
            try:
                response = generate(
                    prompt,
                    model=_model(),
                    timeout=_timeout_sec(),
                    extra_params=_extra_params(),
                )
                raw_text = str(response.get("text", "") or "")
                attempt_debug["raw_response"] = raw_text
                attempt_debug["elapsed"] = response.get("elapsed")
                parsed = _parse_jsonish(raw_text)
                normalized_map, error = _coerce_normalized_mapping(parsed, expected_keys)
                if error:
                    attempt_debug["error"] = error
                    normalized_map = None
                    continue
                attempt_debug["valid"] = True
                break
            except Exception as exc:
                attempt_debug["error"] = str(exc)

        if normalized_map is None:
            group_debug["error"] = "no_valid_response"
            status["skipped_count"] += len(values)
            group_debug["skipped"] = [
                {"index": i, "original": value, "skip_reason": "no_valid_response"}
                for i, value in enumerate(values)
            ]
            continue

        consensus_detail: dict[str, Any] | None = None
        if mode == "consensus":
            consensus_value, consensus_detail = _dominant_consensus_value(values, roi_name)
            if consensus_value:
                normalized_map = {key: consensus_value for key in expected_keys}
            group_debug["consensus"] = consensus_detail
        elif mode == "aggressive":
            cluster_map, cluster_detail = _aggressive_cluster_map(values, roi_name)
            if cluster_map:
                normalized_map.update(cluster_map)
            group_debug["aggressive_clusters"] = cluster_detail

        for key in expected_keys:
            if not normalized_map.get(key) and not _looks_placeholder(key, roi_name):
                normalized_map[key] = key

        group_debug["normalized_map"] = dict(sorted(normalized_map.items()))
        for i, (row, original) in enumerate(zip(rows, values)):
            normalized = normalized_map.get(original, original)
            if not normalized:
                group_debug["skipped"].append(
                    {"index": i, "original": original, "normalized": normalized, "skip_reason": "empty_normalized"}
                )
                status["skipped_count"] += 1
                continue
            if normalized == original:
                group_debug["skipped"].append(
                    {"index": i, "original": original, "normalized": normalized, "skip_reason": "unchanged"}
                )
                status["skipped_count"] += 1
                continue
            row["text"] = normalized
            group_debug["applied"].append(
                {"index": i, "original": original, "normalized": normalized}
            )
            status["applied_count"] += 1

    if debug_out is not None:
        debug_out.update(status)
    return normalized_items, original_items
