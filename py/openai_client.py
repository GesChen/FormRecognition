"""Simple OpenAI API client helpers for this project.

Usage:

    from openai_client import call_text, call_json

    # Simple text call
    text = call_text("Summarize this in one line.")

    # Structured JSON call
    data = call_json(
        "Extract fields.",
        schema={
            "type": "object",
            "properties": {
                "id": {"type": ["string", "null"]},
            },
            "required": ["id"],
            "additionalProperties": False,
        },
        schema_name="id_extract",
    )

    # Profile-based model selection
    text = call_text("Draft this email.", model_profile="cheap")

Environment:
    OPENAI_API_KEY   Required API key.
    OPENAI_MODEL     Optional default model (default: "gpt-5.4-mini").
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

if TYPE_CHECKING:
    from openai import OpenAI

DEFAULT_MODEL = "gpt-5.4-mini"
MODEL_PROFILES: Dict[str, str] = {
    "cheap": "gpt-5.4-nano",
    "balanced": "gpt-5.4-mini",
    "max_quality": "gpt-5.5",
}


def _default_model(model: Optional[str] = None) -> str:
    """Return model override or OPENAI_MODEL/default."""
    return str(model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)


def available_model_profiles() -> Dict[str, str]:
    """Return profile -> model mapping."""
    return dict(MODEL_PROFILES)


def resolve_model(
    *,
    model: Optional[str] = None,
    model_profile: Optional[str] = None,
) -> str:
    """Resolve final model with precedence: explicit model > profile > env/default."""
    if model:
        return str(model).strip()
    if model_profile:
        key = str(model_profile).strip().lower()
        picked = MODEL_PROFILES.get(key)
        if not picked:
            allowed = ", ".join(sorted(MODEL_PROFILES.keys()))
            raise ValueError(f"Unknown model_profile {model_profile!r}. Use one of: {allowed}")
        return picked
    return _default_model(None)


def get_client(*, api_key: Optional[str] = None) -> "OpenAI":
    """Build an OpenAI client using explicit key or OPENAI_API_KEY env var."""
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "OpenAI SDK is not installed. Install with: pip install openai"
        ) from exc
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before calling OpenAI API helpers."
        )
    return OpenAI(api_key=key)


def _normalize_input(input_text: str | Iterable[str]) -> str:
    if isinstance(input_text, str):
        return input_text
    return "\n".join(str(x) for x in input_text)


def call_text(
    input_text: str | Iterable[str],
    *,
    model: Optional[str] = None,
    model_profile: Optional[str] = None,
    instructions: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
) -> str:
    """Run a plain text response call and return output text."""
    client = get_client(api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": resolve_model(model=model, model_profile=model_profile),
        "input": _normalize_input(input_text),
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = int(max_output_tokens)

    resp = client.responses.create(**kwargs)
    return (resp.output_text or "").strip()


def call_json(
    input_text: str | Iterable[str],
    *,
    schema: Dict[str, Any],
    schema_name: str = "result",
    model: Optional[str] = None,
    model_profile: Optional[str] = None,
    instructions: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a structured-output call and return parsed JSON (as dict).

    Args:
        input_text: User input content for the model.
        schema: JSON Schema object describing required output format.
        schema_name: Name attached to the schema in API request.
    """
    client = get_client(api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": resolve_model(model=model, model_profile=model_profile),
        "input": _normalize_input(input_text),
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": schema,
                "strict": True,
            }
        },
    }
    if instructions is not None:
        kwargs["instructions"] = instructions
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = int(max_output_tokens)
    if reasoning is not None:
        kwargs["reasoning"] = reasoning

    resp = client.responses.create(**kwargs)
    parsed = getattr(resp, "output_parsed", None)
    if isinstance(parsed, dict):
        return parsed

    # Fallback for SDK variants where parsed output may not be populated.
    text = (resp.output_text or "").strip()
    if not text:
        raise RuntimeError("OpenAI response did not contain parsed or text output.")

    import json

    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Expected JSON object output, got: {type(obj).__name__}")
    return obj


def call_messages(
    messages: List[Dict[str, Any]],
    *,
    model: Optional[str] = None,
    model_profile: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
) -> str:
    """Low-level helper for callers that want full messages control.

    `messages` should be a Responses API-style input list.
    """
    client = get_client(api_key=api_key)
    kwargs: Dict[str, Any] = {
        "model": resolve_model(model=model, model_profile=model_profile),
        "input": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = int(max_output_tokens)

    resp = client.responses.create(**kwargs)
    return (resp.output_text or "").strip()
