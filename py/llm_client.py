"""
LLM client module: talk to a local model server (Ollama) using config.LLM.

Ref: prompts/module — this is a reusable module with a simple Python API, no CLI.

Usage:

    from llm_client import generate

    result = generate("Fix this CSV row and return JSON.")
    print(result["text"])

Configuration:

- Host / port / model / keep_alive are taken from config.LLM by default:

    from config import LLM
    LLM = {
        "host": "192.168.182.1",
        "port": 11434,
        "model": "deepseek-coder:6.7b",
        "keep_alive": 0,
    }

You can override any of these per-call via function arguments.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterator, Optional

import requests

from config import LLM


def _get_base_url(host: Optional[str] = None, port: Optional[int] = None) -> str:
    """Return base URL (host:port) for the LLM server."""
    h = host or LLM.get("host", "127.0.0.1")
    p = int(port or LLM.get("port", 11434))
    return f"http://{h}:{p}"


def _default_model(model: Optional[str] = None) -> str:
    """Return model name, falling back to config.LLM['model']."""
    return str(model or LLM.get("model", "deepseek-coder:6.7b"))


def _default_keep_alive(keep_alive: Optional[int] = None) -> int:
    """Return keep_alive seconds, default from config (0 = disabled)."""
    return int(keep_alive if keep_alive is not None else LLM.get("keep_alive", 0))


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    keep_alive: Optional[int] = None,
    stream: bool = False,
    timeout: int = 300,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Call the LLM server via Ollama's /api/generate endpoint.

    Args:
        prompt: The text prompt to send to the model.
        model: Optional model name; defaults to config.LLM['model'].
        host: Optional host override; defaults to config.LLM['host'].
        port: Optional port override; defaults to config.LLM['port'].
        keep_alive: Optional keep-alive seconds; defaults to config.LLM['keep_alive'].
        stream: Whether to use streaming. For testing, False is simpler.
        timeout: HTTP timeout in seconds.
        extra_params: Additional keys merged into the JSON payload.

    Returns:
        Dict with keys:
            - text: generated text (response field)
            - raw: full JSON response from the server
            - elapsed: total wall time in seconds
            - eval_count / eval_duration: optional stats from server (if present)
    """
    base_url = _get_base_url(host=host, port=port)
    model_name = _default_model(model)
    ka = _default_keep_alive(keep_alive)

    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": ka,
    }
    if extra_params:
        payload.update(extra_params)

    t0 = time.time()
    resp = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()

    # Extract common fields
    text = data.get("response", "")
    eval_count = data.get("eval_count")
    eval_duration = data.get("eval_duration")

    return {
        "text": text,
        "raw": data,
        "elapsed": elapsed,
        "eval_count": eval_count,
        "eval_duration": eval_duration,
    }


def generate_stream(
    prompt: str,
    *,
    model: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    keep_alive: Optional[int] = None,
    timeout: int = 600,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Stream generation chunks from Ollama /api/generate (NDJSON over HTTP).

    Yields dict events:
      - {"type": "chunk", "delta": "...", "raw": {...}}
      - {"type": "done", "text": "...", "raw": {...}, "elapsed": <sec>}
    """
    base_url = _get_base_url(host=host, port=port)
    model_name = _default_model(model)
    ka = _default_keep_alive(keep_alive)

    payload: Dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "keep_alive": ka,
    }
    if extra_params:
        payload.update(extra_params)

    t0 = time.time()
    full_text_parts: list[str] = []
    done_obj: Dict[str, Any] | None = None
    with requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=(10, int(timeout)),
        stream=True,
    ) as resp:
        resp.raise_for_status()
        # Use a tiny chunk size so token events are surfaced immediately
        # instead of waiting for larger buffered reads.
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

            delta = str(obj.get("response", "") or "")
            if delta:
                full_text_parts.append(delta)
                yield {"type": "chunk", "delta": delta, "raw": obj}

            if bool(obj.get("done")):
                done_obj = obj
                # Some servers only attach output on the terminal done event.
                if not delta:
                    tail = str(obj.get("response", "") or "")
                    if tail:
                        full_text_parts.append(tail)
                        yield {"type": "chunk", "delta": tail, "raw": obj}
                break

    elapsed = time.time() - t0
    full_text = "".join(full_text_parts)
    yield {
        "type": "done",
        "text": full_text,
        "raw": done_obj or {},
        "elapsed": elapsed,
    }


def ping(
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: int = 5,
) -> Dict[str, Any]:
    """
    Lightweight connectivity check against /api/tags.

    Returns:
        Dict with 'ok' (bool) and either 'tags' or 'error' keys.
    """
    base_url = _get_base_url(host=host, port=port)
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        resp.raise_for_status()
        tags = resp.json()
    except Exception as e:  # pragma: no cover - simple manual test helper
        return {"ok": False, "error": str(e)}
    return {"ok": True, "tags": tags}
