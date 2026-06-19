"""
ROI prompt creator helpers for LLM text postprocess prompts (llm_prompt_override).
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from config import ROI_PROMPT_CREATOR
from llm_client import generate, generate_stream
from text_roi_llm import default_text_roi_llm_prompt_template


def _clean_model_text(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    m = re.match(r"^```(?:[a-zA-Z0-9_-]+)?\s*([\s\S]*?)\s*```$", s)
    if m:
        s = m.group(1).strip()
    return s


def _field_contract(dtype: str) -> str:
    d = (dtype or "").strip().lower()
    return {
        "choice": "detected_text must be exactly one uppercase letter (A-Z) or null.",
        "id": "detected_text must be one canonical ID token (no spaces, no extra text) or null.",
        "identifier": "detected_text must be one canonical ID token (no spaces, no extra text) or null.",
        "date": "detected_text must be ISO date YYYY-MM-DD or null.",
        "number": "detected_text must contain digits only (0-9) or null.",
        "boolean": "detected_text must be true, false, or null.",
        "text": "detected_text must be plain extracted text or null.",
        "json": "detected_text must be a single compact JSON string literal or null.",
    }.get(d, "detected_text must be plain extracted text or null.")


def _build_llm_generator_prompt(
    *,
    instruction: str,
    roi_name: str | None = None,
    field_data_type: str | None = None,
    validation_rules: str | None = None,
    existing_prompt: str | None = None,
) -> str:
    roi_label = (roi_name or "").strip() or "(unnamed ROI)"
    dtype = (field_data_type or "").strip() or "text"
    rules = (validation_rules or "").strip()
    existing = str(existing_prompt or "").strip()
    base_prompt = default_text_roi_llm_prompt_template().strip()
    rules_block = rules if rules else "(not provided)"
    existing_block = (
        f"\nExisting prompt to revise (if useful):\n{existing}\n"
        if existing
        else "\nNo existing prompt was provided.\n"
    )
    return (
        "You are a prompt engineer for OCR text postprocessing.\n"
        "Create ONE high-quality prompt text that will be sent to a text LLM for a single ROI field.\n"
        "Return only the final prompt text. Do not include markdown fences, explanations, or notes.\n\n"
        "Requirements for the prompt you produce:\n"
        "- It must demand a strict structured output compatible with the text-ROI LLM workflow.\n"
        "- It must clearly state that input is OCR-extracted text, not an image.\n"
        "- It must instruct the model not to invent unseen content.\n"
        "- It must be specific to the user instruction.\n"
        "- Keep it concise and production-ready.\n"
        '- Include: "Return ONE valid JSON object only, no markdown and no extra text."\n'
        '- Include this required schema key exactly: {"detected_text": "..."}\n'
        "- Do not require custom JSON keys outside the workflow schema.\n\n"
        "Strictness requirements:\n"
        "- Force a single final value only (no explanations, no alternatives, no multiple candidates).\n"
        "- Forbid placeholder/filler outputs (e.g., string, detected_text, ---, ____, N/A).\n"
        "- Forbid markup/noise artifacts (e.g., code fences, TeX snippets, arrows like x->A).\n"
        "- Explicitly require null when value is not clearly readable.\n"
        f"- Field-type contract: {_field_contract(dtype)}\n\n"
        "Important:\n"
        "- Model the final prompt after the provided base text-LLM prompt style.\n"
        "- Preserve core safety/faithfulness constraints from the base prompt.\n"
        "- Adapt wording to ROI-specific requirements while keeping the output contract.\n\n"
        f"Base text-ROI LLM prompt template:\n{base_prompt}\n\n"
        f"ROI name: {roi_label}\n"
        f"Field data type: {dtype}\n"
        f"Validation rules:\n{rules_block}\n"
        f"User instruction:\n{instruction.strip()}\n"
        f"{existing_block}"
    )


def _generate_common(
    ask: str,
    *,
    model: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    model_to_use = str(model or ROI_PROMPT_CREATOR.get("model") or "").strip() or None
    out = generate(
        ask,
        model=model_to_use,
        timeout=int(timeout),
        stream=False,
        extra_params={"think": False},
    )
    raw_text = str(out.get("text", "") or "")
    prompt = _clean_model_text(raw_text)
    if not prompt:
        raise RuntimeError("Model returned an empty prompt.")
    raw = out.get("raw")
    model_name = ""
    if isinstance(raw, dict):
        model_name = str(raw.get("model", "") or "")
    return {
        "prompt": prompt,
        "raw_text": raw_text,
        "elapsed": out.get("elapsed"),
        "model": model_name or str(model_to_use or ""),
        "raw": raw,
    }


def _generate_stream_common(
    ask: str,
    *,
    model: str | None = None,
    timeout: int = 600,
) -> Iterator[dict[str, Any]]:
    model_to_use = str(model or ROI_PROMPT_CREATOR.get("model") or "").strip() or None
    model_label = str(model_to_use or "")
    yield {"type": "start", "model": model_label}

    full_raw = ""
    elapsed = None
    done_text = ""
    stream_failed = False
    try:
        for event in generate_stream(
            ask,
            model=model_to_use,
            timeout=int(timeout),
            extra_params={"think": False},
        ):
            et = str(event.get("type", "") or "")
            if et == "chunk":
                delta = str(event.get("delta", "") or "")
                if delta:
                    full_raw += delta
                    yield {"type": "chunk", "delta": delta}
            elif et == "done":
                elapsed = event.get("elapsed")
                done_text = str(event.get("text", "") or "")
    except Exception:
        stream_failed = True

    prompt = _clean_model_text(full_raw or done_text)
    if not prompt:
        out = generate(
            ask,
            model=model_to_use,
            timeout=int(timeout),
            stream=False,
            extra_params={"think": False},
        )
        prompt = _clean_model_text(str(out.get("text", "") or ""))
        if elapsed is None:
            elapsed = out.get("elapsed")
        if stream_failed and prompt:
            yield {"type": "chunk", "delta": prompt}
    if not prompt:
        raise RuntimeError("Model returned an empty prompt.")
    yield {
        "type": "done",
        "prompt": prompt,
        "elapsed": elapsed,
        "model": model_label,
    }


def generate_roi_llm_prompt(
    instruction: str,
    *,
    roi_name: str | None = None,
    field_data_type: str | None = None,
    validation_rules: str | None = None,
    existing_prompt: str | None = None,
    model: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    ins = str(instruction or "").strip()
    if not ins:
        raise ValueError("instruction is required")
    ask = _build_llm_generator_prompt(
        instruction=ins,
        roi_name=roi_name,
        field_data_type=field_data_type,
        validation_rules=validation_rules,
        existing_prompt=existing_prompt,
    )
    return _generate_common(ask, model=model, timeout=timeout)


def generate_roi_llm_prompt_stream(
    instruction: str,
    *,
    roi_name: str | None = None,
    field_data_type: str | None = None,
    validation_rules: str | None = None,
    existing_prompt: str | None = None,
    model: str | None = None,
    timeout: int = 600,
) -> Iterator[dict[str, Any]]:
    ins = str(instruction or "").strip()
    if not ins:
        raise ValueError("instruction is required")
    ask = _build_llm_generator_prompt(
        instruction=ins,
        roi_name=roi_name,
        field_data_type=field_data_type,
        validation_rules=validation_rules,
        existing_prompt=existing_prompt,
    )
    yield from _generate_stream_common(ask, model=model, timeout=timeout)


def llm_generator_prompt_template_preview() -> str:
    return _build_llm_generator_prompt(
        instruction="<user_instruction>",
        roi_name="<roi_name>",
        field_data_type="<field_data_type>",
        validation_rules="<validation_rules>",
        existing_prompt=None,
    )


# Backward-compatible aliases (old naming).
def generate_roi_vlm_prompt(*args, **kwargs):
    return generate_roi_llm_prompt(*args, **kwargs)


def generate_roi_vlm_prompt_stream(*args, **kwargs):
    return generate_roi_llm_prompt_stream(*args, **kwargs)


def generator_prompt_template_preview() -> str:
    return llm_generator_prompt_template_preview()
