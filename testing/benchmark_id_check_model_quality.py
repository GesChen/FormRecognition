"""
Temporary benchmark tool for ID-check OCR quality across model + JPEG quality.

It runs OCR repeatedly on a single ID crop image, extracts a normalized ID guess,
computes Levenshtein distance against a known true ID, and prints Excel-friendly CSV.

Default sweep:
- models:
  - blaifa/InternVL3_5:4B
  - blaifa/InternVL3_5:8B
  - openbmb/minicpm-v4.5:8b
- JPEG quality: 10, 50, 90
- trials per combo: 20
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

import ocr_engine  # type: ignore
from config import LLM, OCR_ENGINE  # type: ignore

DEFAULT_IMAGE = ROOT / "output" / "debug_images" / "id_crop" / "page_0001_0000_crop_top15pct.png"
DEFAULT_TRUE_ID = "ID:6010985A"
DEFAULT_MODELS = [
    "blaifa/InternVL3_5:4B",
    "blaifa/InternVL3_5:8b",
    "openbmb/minicpm-v4.5:8b",
    "ministral-3:8b",
    "glm-ocr:latest",
    "deepseek-ocr:latest",
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
    p = argparse.ArgumentParser(description="Benchmark ID OCR model/quality combinations.")
    p.add_argument(
        "image_path",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help="Path to ID-crop image (default: output/debug_images/id_crop/page_0001_0000_crop_top15pct.png)",
    )
    p.add_argument("--true-id", default=DEFAULT_TRUE_ID, help=f"Ground-truth ID (default: {DEFAULT_TRUE_ID}).")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model list.")
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
        help="OCR workflow for benchmark (default: vision_only).",
    )
    p.add_argument("--timeout", type=int, default=None, help="Optional OCR call timeout override (seconds).")
    p.add_argument("--verbose", action="store_true", help="Print per-trial lines.")
    return p.parse_args()


def _llm_host() -> str:
    return str(OCR_ENGINE.get("host") or LLM.get("host", "127.0.0.1"))


def _llm_port() -> int:
    raw = OCR_ENGINE.get("port")
    if raw is None:
        raw = LLM.get("port", 11434)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 11434


def _fetch_installed_models(host: str, port: int, timeout_sec: int = 10) -> set[str]:
    url = f"http://{host}:{port}/api/tags"
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    body = resp.json() if resp.content else {}
    out: set[str] = set()
    for item in (body.get("models") or []):
        if not isinstance(item, dict):
            continue
        for k in ("name", "model"):
            v = str(item.get(k) or "").strip()
            if v:
                out.add(v)
    return out


def _precheck_models_or_exit(models: list[str]) -> None:
    host = _llm_host()
    port = _llm_port()
    print(f"Model precheck: querying {host}:{port} ...")
    try:
        installed = _fetch_installed_models(host, port, timeout_sec=10)
    except Exception as exc:
        print(f"Error: model precheck failed against {host}:{port}: {exc}")
        sys.exit(2)

    missing = [m for m in models if m not in installed]
    if missing:
        print("Error: benchmark aborted; missing model(s):")
        for m in missing:
            print(f"  - {m}")
        print("Installed models seen on server:")
        for name in sorted(installed):
            print(f"  - {name}")
        sys.exit(2)

    print("Model precheck passed.")


def _normalize_text(s: str) -> str:
    x = unicodedata.normalize("NFKC", str(s or "")).upper()
    x = re.sub(r"\s+", "", x)
    return x


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def _extract_normalized_id_from_text(text: str) -> str | None:
    s = _normalize_text(text)
    if not s:
        return None

    region = s.split(":", 1)[1] if ":" in s else s
    # Strict extraction only: exactly 7 digits + trailing A/B, no manual char correction.
    matches = re.findall(r"\d{7}[AB]", region)
    if matches:
        return matches[0]
    return None


def _score_run(raw: dict, true_id_fmt: str) -> dict:
    detected_text = str(raw.get("detected_text") or "")
    extracted = _extract_normalized_id_from_text(detected_text)
    pred_fmt = f"ID:{extracted}" if extracted else ""
    dist = _levenshtein(pred_fmt, true_id_fmt)
    denom = max(1, len(true_id_fmt))
    similarity = max(0.0, 1.0 - (dist / denom))
    return {
        "detected_text": detected_text,
        "pred_id": extracted,
        "pred_formatted": pred_fmt,
        "levenshtein": dist,
        "similarity": similarity,
        "exact_match": bool(pred_fmt == true_id_fmt),
        "needs_human_review": bool(raw.get("needs_human_review", False)),
        "selected_model": str(raw.get("selected_model") or ""),
    }


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
    true_id_fmt: str,
    verbose: bool,
) -> dict:
    elapsed_values: list[float] = []
    distance_values: list[int] = []
    similarity_values: list[float] = []
    confidence_values: list[float] = []
    exact_match_count = 0
    review_count = 0
    error_count = 0

    for i in range(1, trials + 1):
        raw, elapsed = _run_once(image_path, timeout)
        elapsed_values.append(elapsed)

        scored = _score_run(raw, true_id_fmt)
        distance_values.append(int(scored["levenshtein"]))
        similarity_values.append(float(scored["similarity"]))
        if scored["exact_match"]:
            exact_match_count += 1
        if scored["needs_human_review"]:
            review_count += 1

        try:
            confidence_values.append(float(raw.get("confidence_score")))
        except (TypeError, ValueError):
            pass

        stages = raw.get("stages") or []
        if any(bool(s.get("error")) for s in stages):
            error_count += 1

        if verbose:
            print(
                f"  trial {i}/{trials}: elapsed={elapsed:.4f}s "
                f"lev={scored['levenshtein']} sim={scored['similarity']:.4f} "
                f"pred={scored['pred_formatted'] or '-'}"
            )

    avg_elapsed = statistics.fmean(elapsed_values) if elapsed_values else 0.0
    avg_distance = statistics.fmean(distance_values) if distance_values else 0.0
    avg_similarity = statistics.fmean(similarity_values) if similarity_values else 0.0
    avg_conf = statistics.fmean(confidence_values) if confidence_values else None

    return {
        "model": model,
        "quality": int(quality),
        "trials": int(trials),
        "avg_elapsed_sec": round(avg_elapsed, 4),
        "min_elapsed_sec": round(min(elapsed_values), 4) if elapsed_values else None,
        "max_elapsed_sec": round(max(elapsed_values), 4) if elapsed_values else None,
        "avg_levenshtein": round(avg_distance, 4),
        "avg_similarity": round(avg_similarity, 4),
        "exact_match_count": exact_match_count,
        "exact_match_rate": round(exact_match_count / max(1, trials), 4),
        "avg_confidence_score": round(avg_conf, 4) if avg_conf is not None else None,
        "needs_review_count": review_count,
        "error_count": error_count,
    }


def _print_human_summary(image_path: Path, true_id: str, workflow: str, rows: list[dict]) -> None:
    print("\n" + "=" * 132)
    print("ID CHECK OCR BENCHMARK SUMMARY")
    print("=" * 132)
    print(f"Image: {image_path}")
    print(f"True ID: {true_id}")
    print(f"Workflow: {workflow}")
    print("-" * 132)
    print(
        f"{'Model':36}  {'Q':>3}  {'Trials':>6}  {'Avg(s)':>8}  {'Min(s)':>8}  {'Max(s)':>8}  "
        f"{'AvgLev':>8}  {'AvgSim':>8}  {'Exact':>6}  {'Exact%':>8}  {'AvgConf':>8}  {'Review':>6}  {'Errors':>6}"
    )
    print("-" * 132)
    for r in rows:
        print(
            f"{r['model']:36}  {r['quality']:>3}  {r['trials']:>6}  {r['avg_elapsed_sec']:>8.4f}  "
            f"{(r['min_elapsed_sec'] if r['min_elapsed_sec'] is not None else 0):>8.4f}  "
            f"{(r['max_elapsed_sec'] if r['max_elapsed_sec'] is not None else 0):>8.4f}  "
            f"{r['avg_levenshtein']:>8.4f}  {r['avg_similarity']:>8.4f}  {r['exact_match_count']:>6}  "
            f"{r['exact_match_rate']:>8.4f}  "
            f"{(r['avg_confidence_score'] if r['avg_confidence_score'] is not None else 0):>8.4f}  "
            f"{r['needs_review_count']:>6}  {r['error_count']:>6}"
        )
    print("=" * 132)


def _print_excel_csv(rows: list[dict], image_path: Path, true_id: str, workflow: str) -> None:
    print("\nEXCEL_CSV_BEGIN")
    print("image_path,true_id,workflow,model,quality,trials,avg_elapsed_sec,min_elapsed_sec,max_elapsed_sec,avg_levenshtein,avg_similarity,exact_match_count,exact_match_rate,avg_confidence_score,needs_review_count,error_count")
    for r in rows:
        line = [
            str(image_path),
            true_id,
            workflow,
            str(r.get("model", "")),
            str(r.get("quality", "")),
            str(r.get("trials", "")),
            str(r.get("avg_elapsed_sec", "")),
            str(r.get("min_elapsed_sec", "")),
            str(r.get("max_elapsed_sec", "")),
            str(r.get("avg_levenshtein", "")),
            str(r.get("avg_similarity", "")),
            str(r.get("exact_match_count", "")),
            str(r.get("exact_match_rate", "")),
            str(r.get("avg_confidence_score", "")),
            str(r.get("needs_review_count", "")),
            str(r.get("error_count", "")),
        ]
        print(",".join(x.replace(",", " ") for x in line))
    print("EXCEL_CSV_END")


def main() -> None:
    args = _parse_args()
    image_path = Path(args.image_path).resolve()
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    models = _parse_csv_models(args.models)
    qualities = _parse_csv_qualities(args.qualities)
    trials = max(1, int(args.trials or DEFAULT_TRIALS))
    _precheck_models_or_exit(models)

    true_id_norm = _normalize_text(str(args.true_id or DEFAULT_TRUE_ID))
    if not true_id_norm.startswith("ID:"):
        true_id_norm = f"ID:{true_id_norm}"

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
                    f"Running: model={model} quality={quality} trials={trials} workflow={args.workflow}"
                )
                row = _run_combo(
                    image_path,
                    model=model,
                    quality=quality,
                    trials=trials,
                    timeout=args.timeout,
                    true_id_fmt=true_id_norm,
                    verbose=bool(args.verbose),
                )
                rows.append(row)
    finally:
        OCR_ENGINE["model"] = original["model"]
        OCR_ENGINE["jpeg_quality"] = original["jpeg_quality"]
        OCR_ENGINE["workflow_default"] = original["workflow_default"]

    _print_human_summary(image_path, true_id_norm, str(args.workflow), rows)
    _print_excel_csv(rows, image_path, true_id_norm, str(args.workflow))


if __name__ == "__main__":
    main()
