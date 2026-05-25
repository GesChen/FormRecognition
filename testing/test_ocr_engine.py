"""
Test OCR engine detector on a single raw image.

Usage (from project root):

    python3 testing/test_ocr_engine.py <image_path> [--verbose] [--force-paddle-failure] [--repeat N]

Example:

    python3 testing/test_ocr_engine.py data/sample.png --verbose --force-paddle-failure --repeat 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from ocr_engine import ocr_raw


def _safe_str(value) -> str:
    if value is None:
        return "-"
    return str(value)


def _overall_status(result: dict) -> str:
    stages = result.get("stages") or []
    any_accepted = any(bool(s.get("accepted", False)) for s in stages)
    has_text = bool(str(result.get("detected_text") or "").strip())
    needs_review = bool(result.get("needs_human_review", False))
    if needs_review:
        return "REVIEW_REQUIRED"
    if any_accepted and has_text:
        return "AUTO_ACCEPTED"
    if any_accepted and not has_text:
        return "ACCEPTED_EMPTY_TEXT"
    return "NO_ACCEPTED_STAGE"


def print_standardized_human_summary(result: dict) -> None:
    """Print a fixed-format summary block for quick human scanning."""
    stages = result.get("stages") or []
    text = str(result.get("detected_text") or "")
    text_preview = text.replace("\n", " ").strip()
    if len(text_preview) > 240:
        text_preview = text_preview[:240] + "...(truncated)"

    print("\n" + "=" * 72)
    print("STANDARDIZED OCR TEST SUMMARY")
    print("=" * 72)
    print(f"overall_status: {_overall_status(result)}")
    print(f"image_path: {_safe_str(result.get('image_path'))}")
    print(f"elapsed_sec: {_safe_str(result.get('elapsed_sec'))}")
    print(f"selected_stage_index: {_safe_str(result.get('selected_stage_index'))}")
    print(f"selected_model: {_safe_str(result.get('selected_model'))}")
    print(f"confidence_label: {_safe_str(result.get('confidence_label'))}")
    print(f"confidence_score: {_safe_str(result.get('confidence_score'))}")
    print(f"needs_human_review: {_safe_str(result.get('needs_human_review'))}")
    print(f"stage_count: {len(stages)}")
    print("stage_results:")
    for st in stages:
        print(
            "  - "
            f"stage_index={_safe_str(st.get('stage_index'))} | "
            f"model={_safe_str(st.get('model'))} | "
            f"accepted={_safe_str(st.get('accepted'))} | "
            f"elapsed_sec={_safe_str(st.get('elapsed_sec'))} | "
            f"error={_safe_str(st.get('error'))}"
        )
    print(f"text_preview: {_safe_str(text_preview)}")
    print("=" * 72)


def run_ocr_detector_test(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    trace_steps: bool = False,
    verbose: bool = False,
    force_paddle_failure: bool = False,
) -> dict:
    """Test helper for running OCR detection on one raw image."""
    t0 = time.perf_counter()
    raw = ocr_raw(
        image_path,
        timeout=timeout,
        verbose=trace_steps,
        force_paddle_failure=force_paddle_failure,
    )
    elapsed = round(time.perf_counter() - t0, 4)

    stages_out = []
    for st in raw.get("stages") or []:
        row = {
            "stage_index": st.get("stage_index"),
            "model": st.get("model"),
            "elapsed_sec": st.get("elapsed_sec"),
            "accepted": bool(st.get("accepted", False)),
            "error": st.get("error"),
            "parsed": st.get("parsed") or {},
        }
        if verbose:
            row["response_text"] = st.get("response_text")
            row["raw_response"] = st.get("raw_response")
        stages_out.append(row)

    return {
        "image_path": str(Path(image_path).resolve()),
        "elapsed_sec": elapsed,
        "detected_text": raw.get("detected_text", ""),
        "confidence_label": raw.get("confidence_label"),
        "confidence_score": raw.get("confidence_score"),
        "needs_human_review": bool(raw.get("needs_human_review", False)),
        "selected_model": raw.get("selected_model"),
        "selected_stage_index": raw.get("selected_stage_index"),
        "stage_count": len(stages_out),
        "stages": stages_out,
    }


def run_ocr_detector_repeated(
    image_path: str | Path,
    *,
    repeat: int,
    timeout: int | None = None,
    trace_steps: bool = False,
    verbose: bool = False,
    force_paddle_failure: bool = False,
) -> dict:
    runs: list[dict] = []
    for i in range(max(1, int(repeat))):
        run = run_ocr_detector_test(
            image_path,
            timeout=timeout,
            trace_steps=trace_steps,
            verbose=verbose,
            force_paddle_failure=force_paddle_failure,
        )
        run["run_index"] = i + 1
        runs.append(run)

    elapsed_values = [float(r.get("elapsed_sec", 0.0) or 0.0) for r in runs]
    avg_elapsed = round(sum(elapsed_values) / len(elapsed_values), 4) if elapsed_values else 0.0
    min_elapsed = round(min(elapsed_values), 4) if elapsed_values else 0.0
    max_elapsed = round(max(elapsed_values), 4) if elapsed_values else 0.0

    status_counts: dict[str, int] = {}
    selected_model_counts: dict[str, int] = {}
    for r in runs:
        status = _overall_status(r)
        status_counts[status] = status_counts.get(status, 0) + 1
        model = str(r.get("selected_model") or "unknown")
        selected_model_counts[model] = selected_model_counts.get(model, 0) + 1

    confidence_scores: list[float] = []
    for r in runs:
        try:
            confidence_scores.append(float(r.get("confidence_score")))
        except (TypeError, ValueError):
            pass
    avg_confidence_score = (
        round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else None
    )

    aggregate = {
        "repeat_count": len(runs),
        "avg_elapsed_sec": avg_elapsed,
        "min_elapsed_sec": min_elapsed,
        "max_elapsed_sec": max_elapsed,
        "avg_confidence_score": avg_confidence_score,
        "status_counts": status_counts,
        "selected_model_counts": selected_model_counts,
    }
    return {
        "image_path": str(Path(image_path).resolve()),
        "aggregate": aggregate,
        "runs": runs,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run OCR detector test on one raw image.",
    )
    p.add_argument("image_path", help="Path to source image.")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Include raw per-stage model response payloads in output.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="OCR call timeout (seconds).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full result as JSON only.",
    )
    p.add_argument(
        "--force-paddle-failure",
        action="store_true",
        help="Force Paddle stage failure to test vision fallback behavior.",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the test multiple times and aggregate metrics (default: 1).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    repeat = max(1, int(args.repeat or 1))
    multi = run_ocr_detector_repeated(
        image_path,
        repeat=repeat,
        timeout=args.timeout,
        trace_steps=bool(args.verbose and not args.json and repeat == 1),
        verbose=bool(args.verbose),
        force_paddle_failure=bool(args.force_paddle_failure),
    )
    result = (multi.get("runs") or [{}])[-1]
    aggregate = multi.get("aggregate") or {}

    if args.json:
        print(json.dumps(multi, indent=2, default=str))
        return

    print(f"Image: {multi.get('image_path')}")
    print(f"Repeat count: {aggregate.get('repeat_count')}")
    print(
        "Average elapsed: "
        f"{aggregate.get('avg_elapsed_sec')}s "
        f"(min={aggregate.get('min_elapsed_sec')}s, max={aggregate.get('max_elapsed_sec')}s)"
    )
    print(f"Last run elapsed: {result.get('elapsed_sec')}s")
    print(f"Selected stage: {result.get('selected_stage_index')} ({result.get('selected_model')})")
    print(
        "Confidence: "
        f"{result.get('confidence_label')} ({result.get('confidence_score')})"
    )
    print(f"Needs human review: {result.get('needs_human_review')}")
    print(f"Stages run: {result.get('stage_count')}")
    print(f"Status counts: {aggregate.get('status_counts')}")
    print(f"Selected model counts: {aggregate.get('selected_model_counts')}")
    print("-" * 60)
    print("Detected text:")
    print(result.get("detected_text") or "(empty)")

    print("-" * 60)
    print("Stages:")
    for st in result.get("stages") or []:
        print(
            f"  [{st.get('stage_index')}] model={st.get('model')} "
            f"accepted={st.get('accepted')} elapsed={st.get('elapsed_sec')}s "
            f"error={st.get('error')!r}"
        )

    if args.verbose:
        print("-" * 60)
        print("Verbose stage payloads:")
        print(json.dumps(result, indent=2, default=str))

    # Always end non-JSON mode with a fixed-format human-readable summary.
    print_standardized_human_summary(result)


if __name__ == "__main__":
    main()
