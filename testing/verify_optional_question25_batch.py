#!/usr/bin/env python3
"""Live-model verifier for the optional-question q25 batch.

This reproduces the comparison run against the rosemontall 6post batch by:
- loading the same per-page ROI text from output/recognition/rosemontall_debug.json
- extracting the q25 ROI text for every 6post page
- sending the same short header-removal prompt to the selected model(s)
- timing each model and summarizing the results

The script exits non-zero if the observed counts differ from the expected
snapshot captured during the prompt-tuning pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from llm_client import generate  # noqa: E402


DEFAULT_SOURCE = ROOT / "output" / "recognition" / "rosemontall_debug.json"
DEFAULT_MODELS = ("qwen3.5:9b", "llama3.2:latest")
DEFAULT_PROMPT = (
    'Return ONE valid JSON object only: {"detected_text":"string or null"}. '
    'If the OCR text contains any part of the header "What, if anything, could have made this experience better for you?" '
    'remove that header text completely, including partial fragments like "what", "if anything", or '
    '"could have made this experience better for you". If nothing remains, return null. '
    "Do not summarize, improve, or explain."
)

EXPECTED = {
    "qwen3.5:9b": {"null": 76, "unchanged": 11, "header_cleanup": 1, "other_text": 0, "invalid_json": 0},
    "llama3.2:latest": {"null": 79, "unchanged": 6, "header_cleanup": 0, "other_text": 3, "invalid_json": 0},
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split())


def _extract_detected_text(payload: str) -> Any:
    s = str(payload or "").strip()
    if not s:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if fenced:
        s = fenced.group(1).strip()
    else:
        obj_match = re.search(r"\{[\s\S]*\}", s)
        if obj_match:
            s = obj_match.group(0)
    # Keep the parser simple and deterministic: this verifier expects a JSON object.
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("response is not a JSON object")
    if "detected_text" not in obj:
        raise ValueError("missing detected_text key")
    value = obj["detected_text"]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("detected_text is not a string or null")
    value = _norm(value)
    return value


def _load_inputs(source: Path) -> list[str]:
    obj = json.loads(source.read_text(encoding="utf-8"))
    step4 = (((obj.get("steps") or {}).get("4_roi")) or {})
    pages = step4.get("roi_per_page") or []
    inputs: list[str] = []
    for page in pages:
        if not isinstance(page, dict) or page.get("form_type") != "6post":
            continue
        text_roi = page.get("text_per_roi") or {}
        q25 = text_roi.get("25")
        if not isinstance(q25, dict):
            continue
        inputs.append(_norm(q25.get("text")))
    return inputs


def _classify(inp: str, out: Any) -> str:
    if out is None:
        return "null"
    if not isinstance(out, str):
        return "invalid_json"
    inp_n = _norm(inp)
    out_n = _norm(out)
    if out_n == inp_n:
        return "unchanged"
    header_markers = (
        "what, if anything, could have made this experience better for you",
        "if anything, could have made this experience better for you",
        "could have made this experience better for you",
        "optional question",
    )
    if any(marker in inp_n.lower() for marker in header_markers):
        return "header_cleanup"
    return "other_text"


def _run_model(model: str, prompt: str, inputs: list[str]) -> dict[str, Any]:
    counts = Counter()
    invalid_examples: list[tuple[int, str, str]] = []
    t0 = time.perf_counter()
    for idx, text in enumerate(inputs):
        result = generate(
            prompt=f"{prompt}\n\nOCR text:\n{text}",
            model=model,
            timeout=180,
            extra_params={"think": False, "options": {"temperature": 0}},
        )
        try:
            detected = _extract_detected_text(result["text"])
        except Exception as exc:
            counts["invalid_json"] += 1
            if len(invalid_examples) < 3:
                invalid_examples.append((idx, text, f"{exc}: {result['text']!r}"))
            continue

        counts[_classify(text, detected)] += 1

    elapsed = time.perf_counter() - t0
    return {
        "counts": dict(counts),
        "elapsed_sec": elapsed,
        "invalid_examples": invalid_examples,
    }


def _fmt_counts(counts: dict[str, int]) -> str:
    order = ("null", "unchanged", "header_cleanup", "other_text", "invalid_json")
    return ", ".join(f"{k}={counts.get(k, 0)}" for k in order)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Debug JSON source file.")
    ap.add_argument("--model", action="append", dest="models", help="Model name to test. May be repeated.")
    ap.add_argument("--no-assert", action="store_true", help="Print results without enforcing the snapshot.")
    args = ap.parse_args()

    source = args.source.resolve()
    if not source.exists():
        print(f"error: not found: {source}")
        return 2

    models = tuple(args.models) if args.models else DEFAULT_MODELS
    inputs = _load_inputs(source)
    if len(inputs) != 88:
        print(f"error: expected 88 q25 inputs from rosemontall 6post, found {len(inputs)}")
        return 2

    print(f"source: {source}")
    print(f"q25 inputs: {len(inputs)}")
    print(f"prompt: {DEFAULT_PROMPT}")
    print()

    failures: list[str] = []
    for model in models:
        print(f"model: {model}")
        result = _run_model(model, DEFAULT_PROMPT, inputs)
        counts = result["counts"]
        elapsed = result["elapsed_sec"]
        print(f"elapsed_sec: {elapsed:.3f}")
        print(f"counts: {_fmt_counts(counts)}")
        if result["invalid_examples"]:
            print("invalid_examples:")
            for idx, raw, err in result["invalid_examples"]:
                print(f"  - row={idx} input={raw!r} error={err}")

        expected = EXPECTED.get(model)
        if expected and not args.no_assert:
            if any(int(counts.get(k, 0)) != v for k, v in expected.items()):
                failures.append(f"{model}: expected {_fmt_counts(expected)}")
        print()

    if failures and not args.no_assert:
        print("snapshot mismatch:")
        for line in failures:
            print(f"- {line}")
        return 1

    if not args.no_assert:
        print("snapshot: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
