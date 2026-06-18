"""
Temporary OCR benchmark tool for model + JPEG quality sweeps.

Runs the OCR engine repeatedly for each (model, quality) combination and
prints a pretty summary table with timing and outcome stats.

Usage examples:

  python3 testing/benchmark_ocr_model_quality.py /path/to/image.png
  python3 testing/benchmark_ocr_model_quality.py /path/to/image.png --trials 20
  python3 testing/benchmark_ocr_model_quality.py /path/to/image.png \
      --models "blaifa/InternVL3_5:4B,blaifa/InternVL3_5:8B,openbmb/minicpm-v4.5:8b" \
      --qualities "10,50,90"
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import ocr_engine  # type: ignore
from config import OCR_ENGINE  # type: ignore

# Script-level defaults (not config.py defaults)
DEFAULT_MODELS = [
    "blaifa/InternVL3_5:4B",
    "blaifa/InternVL3_5:8B",
    "openbmb/minicpm-v4.5:8b",
]
DEFAULT_QUALITIES = [10, 50, 90]
DEFAULT_TRIALS = 20
DEFAULT_WORKFLOW = "vision_only"


def _parse_csv_models(value: str) -> list[str]:
    out = [x.strip() for x in str(value or "").split(",") if x.strip()]
    return out or list(DEFAULT_MODELS)


def _parse_csv_qualities(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            q = int(token)
        except ValueError:
            continue
        out.append(max(1, min(100, q)))
    return out or list(DEFAULT_QUALITIES)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OCR model/quality combinations.")
    p.add_argument("image_path", help="Path to one image to benchmark.")
    p.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated model list.",
    )
    p.add_argument(
        "--qualities",
        default=",".join(str(x) for x in DEFAULT_QUALITIES),
        help="Comma-separated JPEG quality list (1-100).",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"Trials per (model, quality) pair (default: {DEFAULT_TRIALS}).",
    )
    p.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        choices=["vision_only", "paddle_only", "paddle_then_vision"],
        help="OCR workflow to use during benchmark (default: vision_only).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional OCR call timeout override (seconds).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-trial progress lines.",
    )
    return p.parse_args()


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


def _fmt_float(value: float | None, ndigits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{ndigits}f}"


def _run_once(image_path: Path, timeout: int | None) -> tuple[dict, float]:
    t0 = time.perf_counter()
    raw = ocr_engine.ocr_raw(image_path, timeout=timeout, verbose=False)
    elapsed = time.perf_counter() - t0
    return raw, elapsed


def _run_combo(
    image_path: Path,
    *,
    model: str,
    quality: int,
    trials: int,
    timeout: int | None,
    verbose: bool,
) -> dict:
    elapsed_values: list[float] = []
    confidence_values: list[float] = []
    status_counts: dict[str, int] = {}
    review_count = 0
    error_count = 0

    for i in range(1, trials + 1):
        raw, elapsed = _run_once(image_path, timeout)
        elapsed_values.append(elapsed)

        try:
            confidence_values.append(float(raw.get("confidence_score")))
        except (TypeError, ValueError):
            pass

        status = _overall_status(raw)
        status_counts[status] = status_counts.get(status, 0) + 1

        if bool(raw.get("needs_human_review", False)):
            review_count += 1

        stages = raw.get("stages") or []
        if any(bool(s.get("error")) for s in stages):
            error_count += 1

        if verbose:
            print(
                f"  trial {i}/{trials}: elapsed={elapsed:.4f}s "
                f"status={status} confidence={raw.get('confidence_score')}"
            )

    avg_elapsed = statistics.fmean(elapsed_values) if elapsed_values else None
    stdev_elapsed = statistics.pstdev(elapsed_values) if len(elapsed_values) > 1 else 0.0
    avg_conf = statistics.fmean(confidence_values) if confidence_values else None

    return {
        "model": model,
        "quality": int(quality),
        "trials": int(trials),
        "avg_elapsed_sec": round(avg_elapsed, 4) if avg_elapsed is not None else None,
        "stdev_elapsed_sec": round(stdev_elapsed, 4),
        "min_elapsed_sec": round(min(elapsed_values), 4) if elapsed_values else None,
        "max_elapsed_sec": round(max(elapsed_values), 4) if elapsed_values else None,
        "avg_confidence_score": round(avg_conf, 4) if avg_conf is not None else None,
        "needs_review_count": review_count,
        "error_count": error_count,
        "status_counts": status_counts,
    }


def _print_summary(image_path: Path, workflow: str, rows: list[dict]) -> None:
    print("\n" + "=" * 118)
    print("OCR MODEL/QUALITY BENCHMARK SUMMARY")
    print("=" * 118)
    print(f"Image: {image_path}")
    print(f"Workflow: {workflow}")
    print("-" * 118)
    print(
        f"{'Model':36}  {'Q':>3}  {'Trials':>6}  {'Avg(s)':>8}  {'Std(s)':>8}  "
        f"{'Min(s)':>8}  {'Max(s)':>8}  {'AvgConf':>8}  {'Review':>6}  {'Errors':>6}"
    )
    print("-" * 118)
    for r in rows:
        print(
            f"{str(r.get('model', '')):36}  "
            f"{int(r.get('quality', 0)):>3}  "
            f"{int(r.get('trials', 0)):>6}  "
            f"{_fmt_float(r.get('avg_elapsed_sec'), 4):>8}  "
            f"{_fmt_float(r.get('stdev_elapsed_sec'), 4):>8}  "
            f"{_fmt_float(r.get('min_elapsed_sec'), 4):>8}  "
            f"{_fmt_float(r.get('max_elapsed_sec'), 4):>8}  "
            f"{_fmt_float(r.get('avg_confidence_score'), 4):>8}  "
            f"{int(r.get('needs_review_count', 0)):>6}  "
            f"{int(r.get('error_count', 0)):>6}"
        )
    print("-" * 118)
    print("Per-row status_counts:")
    for r in rows:
        print(f"  {r['model']} @ q={r['quality']}: {r.get('status_counts', {})}")
    print("=" * 118)


def main() -> None:
    args = _parse_args()
    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    models = _parse_csv_models(args.models)
    qualities = _parse_csv_qualities(args.qualities)
    trials = max(1, int(args.trials or DEFAULT_TRIALS))

    original = {
        "model": OCR_ENGINE.get("model"),
        "jpeg_quality": OCR_ENGINE.get("jpeg_quality"),
        "workflow_default": OCR_ENGINE.get("workflow_default"),
    }

    rows: list[dict] = []
    try:
        OCR_ENGINE["workflow_default"] = str(args.workflow)
        for model in models:
            for quality in qualities:
                OCR_ENGINE["model"] = str(model)
                OCR_ENGINE["jpeg_quality"] = int(quality)
                print(
                    f"Running: model={model} quality={quality} trials={trials} "
                    f"workflow={args.workflow}"
                )
                row = _run_combo(
                    image_path,
                    model=model,
                    quality=quality,
                    trials=trials,
                    timeout=args.timeout,
                    verbose=bool(args.verbose),
                )
                rows.append(row)
    finally:
        OCR_ENGINE["model"] = original["model"]
        OCR_ENGINE["jpeg_quality"] = original["jpeg_quality"]
        OCR_ENGINE["workflow_default"] = original["workflow_default"]

    _print_summary(image_path, str(args.workflow), rows)


if __name__ == "__main__":
    main()
