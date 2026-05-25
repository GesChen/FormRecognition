"""
Simple sanity-check script for a local LLM served by Ollama on the host machine.

Uses configuration from `py/config.py` (LLM["host"], LLM["port"], LLM["model"]).

Usage (from project root):
    python3 -m testing.test_llm_connection
"""

from __future__ import annotations

import json
import textwrap
import time
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from config import LLM
from llm_client import generate, ping


def _print_box(title: str, content: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")
    print(content)
    print(bar)


def main() -> None:
    host = LLM.get("host", "127.0.0.1")
    port = int(LLM.get("port", 11434))
    model = str(LLM.get("model", "deepseek-coder:6.7b"))
    keep_alive = int(LLM.get("keep_alive", 0))

    base_url = f"http://{host}:{port}"
    print(f"Testing LLM via Ollama at: {base_url}")
    print(f"Model: {model}")

    # 1) Check /api/tags to ensure server is reachable (via llm_client.ping)
    ping_result = ping(host=host, port=port)
    if not ping_result.get("ok"):
        raise SystemExit(
            f"ERROR: Could not reach LLM server at {base_url}/api/tags: {ping_result.get('error')}"
        )
    tags_json = ping_result.get("tags", {})
    _print_box("Available models (/api/tags)", json.dumps(tags_json, indent=2))

    # 2) Run a simple generate call
    prompt = """\
You are a helpful assistant. Write a detailed explanation (at least 600 words).

Topic: How to normalize noisy tabular survey data, infer missing values, and align it to a fixed schema.
Cover:
- Common types of noise (typos, inconsistent formats, missing fields)
- Strategies for cleaning and standardizing fields
- Approaches for estimating missing values
- How to validate and reconcile against a master schema
Use short sections and bullet points.
"""

    print("Sending test prompt...")
    t0 = time.time()
    try:
        result = generate(
            prompt,
            model=model,
            host=host,
            port=port,
            keep_alive=keep_alive,
            stream=False,
            timeout=120,
        )
    except Exception as e:  # pragma: no cover - simple manual test helper
        raise SystemExit(f"ERROR: generate request failed: {e}")

    dt = time.time() - t0
    response_text = result.get("text", "")
    eval_count = result.get("eval_count")
    eval_duration = result.get("eval_duration")

    _print_box(
        "Prompt",
        textwrap.indent(prompt.strip(), "  "),
    )
    _print_box(
        "Model response",
        textwrap.indent(response_text.strip(), "  "),
    )

    stats_lines = [f"Total wall time: {dt:.2f} s"]
    if eval_count is not None and eval_duration:
        seconds = eval_duration / 1e9
        tok_s = eval_count / seconds if seconds > 0 else None
        stats_lines.append(f"Tokens generated: {eval_count}")
        stats_lines.append(f"Eval time: {seconds:.2f} s")
        if tok_s is not None:
            stats_lines.append(f"Throughput: {tok_s:.1f} tok/s")

    _print_box("Stats", "\n".join("  " + line for line in stats_lines))

    print("\nIf you see a sensible response and reasonable throughput, your LLM setup is working.")


if __name__ == "__main__":
    main()

