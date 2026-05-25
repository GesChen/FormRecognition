"""
Standalone module: full PDF recognition workflow.

Converts PDF to images, runs form-type detection on side-a headers, normalizes with
form-specific templates (e.g. 6pre_a, 6pre_b), then runs ROI extraction via
roi_page_module (ID handled as a normal text ROI in step 4); groups pairs of pages
(1+2, 3+4, ...), merges results, and writes a list of items to a JSON file named
after the PDF.

Ref: prompts/module — single-call usage after import.

Usage:
    from pdf_recognize import run_workflow

    items = run_workflow("data/maury 1.pdf")
    # Writes output/recognition/maury_1.json (with pdf_path, pdf_stem, generated_at, item_count, items)
    # and returns list of dicts (the items).
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import config as _config_module
from config import IMAGE_NORMALIZE, PATHS, PDF_RECOGNITION, ROI_PAGE_RECOGNITION

try:
    from ocr_human_review import build_human_review_block
except ImportError:  # pragma: no cover
    build_human_review_block = None  # type: ignore[misc, assignment]


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable


def _tqdm_enabled() -> bool:
    """
    Render live tqdm bars only on interactive terminals.
    Prevents carriage-return bars from turning into log spam in web previews.
    """
    env = str(os.environ.get("EVMS_TQDM", "auto")).strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    if str(os.environ.get("TERM", "")).strip().lower() == "dumb":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())

# Lazy imports for heavy deps
_pdf_to_images = None
_normalize_image = None
_extract_id_form_batch = None
_analyze_pages_batch = None


def _get_pdf_to_images():
    global _pdf_to_images
    if _pdf_to_images is None:
        from pdf_to_images import pdf_to_images
        _pdf_to_images = pdf_to_images
    return _pdf_to_images


def _get_normalize_image():
    global _normalize_image
    if _normalize_image is None:
        from image_normalize import normalize_image
        _normalize_image = normalize_image
    return _normalize_image


def _get_extract_id_form_batch():
    global _extract_id_form_batch
    if _extract_id_form_batch is None:
        from id_form_llm import extract_id_and_form_type_batch
        _extract_id_form_batch = extract_id_and_form_type_batch
    return _extract_id_form_batch


def _get_analyze_pages_batch():
    global _analyze_pages_batch
    if _analyze_pages_batch is None:
        from roi_page_module import analyze_pages_batch
        _analyze_pages_batch = analyze_pages_batch
    return _analyze_pages_batch


def _sanitize_pdf_stem(name: str) -> str:
    """Safe filename stem from PDF name."""
    stem = Path(name).stem
    return re.sub(r"[^\w\-.]", "_", stem).strip("_") or "pdf"


def _sanitize_output_suffix(value: str | None) -> str:
    """
    Safe optional output suffix (prefixed with "_"), or "" when unset.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    safe = re.sub(r"[^\w\-.]", "_", raw).strip("_")
    if not safe:
        return ""
    return f"_{safe}"


def _rel_from_project_root(full: Path) -> str:
    """Path relative to project root for JSON portability."""
    root = Path(__file__).resolve().parent.parent
    try:
        return str(full.resolve().relative_to(root))
    except ValueError:
        return str(full.resolve())


def _form_type_predicted_debug(form_type: str | None) -> str:
    """
    Human-readable summary of the detected form type (grade + pre/post) for verbose CLI output.

    Examples: 6post -> "6th Grade post", hpre -> "High School pre", None -> "(none)".
    """
    if form_type is None or not str(form_type).strip():
        return "predicted_grade=(none)"
    ft = str(form_type).strip().lower()
    m = re.match(r"^([678h])(pre|post)$", ft)
    if not m:
        return f"form_type={form_type!r}"
    g, timing = m.group(1), m.group(2)
    grade_word = {"6": "6th Grade", "7": "7th Grade", "8": "8th Grade", "h": "High School"}[g]
    timing_word = "pre" if timing == "pre" else "post"
    return f"form_type={form_type!r} (predicted: {grade_word} {timing_word})"


def _schema_path(key: str) -> Path:
    cfg = PDF_RECOGNITION
    schema_dir = Path(cfg.get("schema_dir", Path(__file__).resolve().parent.parent / "data" / "roi_schemas"))
    return schema_dir.resolve() / f"{key}.json"


def _schema_path_for_form(form_type: str | None, side: str) -> Path:
    """Resolve per-form schema: data/roi_schemas/<form_type>_<side>.json with config fallback."""
    cfg = PDF_RECOGNITION
    schema_dir = Path(cfg.get("schema_dir", Path(__file__).resolve().parent.parent / "data" / "roi_schemas")).resolve()
    if form_type:
        return schema_dir / f"{form_type}_{side}.json"
    fallback = cfg.get("schema_key_sidea", "schema_sidea") if side == "a" else cfg.get("schema_key_sideb", "schema_sideb")
    return schema_dir / f"{fallback}.json"


def _get_mc_roi_names(schema_path: Path) -> list[str]:
    """ROI names that look like MCQ bubbles: digits + one of a-h."""
    data = json.loads(schema_path.read_text())
    names = []
    for roi in data.get("rois", []):
        n = roi.get("name", "")
        if n and len(n) >= 2 and n[-1].lower() in "abcdefgh" and n[:-1].isdigit():
            names.append(n)
    return sorted(names, key=lambda x: (int(re.match(r"\d+", x).group()), x[-1]))


def _bool_dict_to_letter_per_question(roi_filled: dict[str, bool]) -> dict[str, str]:
    """Convert ROI name -> filled (bool) to question number -> chosen letter (a-h). When multiple filled, returns ""."""
    by_question: dict[str, list[tuple[str, bool]]] = {}
    for roi_name, filled in roi_filled.items():
        m = re.match(r"^(\d+)([a-h])$", roi_name, re.I)
        if not m:
            continue
        q, letter = m.group(1), m.group(2).lower()
        by_question.setdefault(q, []).append((letter, filled))
    out: dict[str, str] = {}
    for q, options in sorted(by_question.items(), key=lambda x: int(x[0])):
        chosen = [letter for letter, filled in options if filled]
        out[q] = chosen[0] if len(chosen) == 1 else ""
    return out


def _letter_per_question_from_filled_and_darkness(
    roi_filled: dict[str, bool],
    roi_darkness: dict[str, dict],
    use_graduated: bool,
) -> dict[str, str]:
    """
    Convert ROI filled + darkness to question -> chosen letter.
    When exactly one option is filled, return that letter.
    When multiple are filled, return the letter with the highest darkness (flat or graduated per config).
    When none filled, return "".
    """
    key = "darkness_graduated" if use_graduated else "darkness_flat"
    by_question: dict[str, list[tuple[str, bool, float]]] = {}
    for roi_name, filled in roi_filled.items():
        m = re.match(r"^(\d+)([a-h])$", roi_name, re.I)
        if not m:
            continue
        q, letter = m.group(1), m.group(2).lower()
        dark_data = roi_darkness.get(roi_name, {})
        darkness = float(dark_data.get(key, 0.0))
        by_question.setdefault(q, []).append((letter, filled, darkness))
    out: dict[str, str] = {}
    for q, options in sorted(by_question.items(), key=lambda x: int(x[0])):
        chosen = [(letter, darkness) for letter, filled, darkness in options if filled]
        if not chosen:
            out[q] = ""
        elif len(chosen) == 1:
            out[q] = chosen[0][0]
        else:
            out[q] = max(chosen, key=lambda x: x[1])[0]
    return out


def _merge_mcq_letters(odd_letters: dict[str, str], even_letters: dict[str, str]) -> dict[str, str]:
    """Merge odd and even page question->letter dicts into one with keys 1, 2, ..., N (odd first, then even)."""
    n_odd = len(odd_letters)
    merged: dict[str, str] = {}
    for i, (q, letter) in enumerate(sorted(odd_letters.items(), key=lambda x: int(x[0])), start=1):
        merged[str(i)] = letter
    for i, (q, letter) in enumerate(sorted(even_letters.items(), key=lambda x: int(x[0])), start=1):
        merged[str(n_odd + i)] = letter
    return merged


def _darkness_by_question(roi_darkness: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """Group ROI darkness dict (roi_name -> {flat, graduated, filled}) by question number. Returns question -> letter -> {darkness_flat, darkness_graduated, filled}."""
    by_q: dict[str, dict[str, dict]] = {}
    for roi_name, data in roi_darkness.items():
        m = re.match(r"^(\d+)([a-h])$", roi_name, re.I)
        if not m:
            continue
        q, letter = m.group(1), m.group(2).lower()
        by_q.setdefault(q, {})[letter] = {
            "darkness_flat": data.get("darkness_flat", 0.0),
            "darkness_graduated": data.get("darkness_graduated", 0.0),
            "filled": data.get("filled", False),
        }
    return by_q


def _merge_mcq_darknesses(
    by_question_odd: dict[str, dict[str, dict]],
    by_question_even: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """Merge odd/even question->letter->darkness into one dict with keys 1..N (odd then even)."""
    n_odd = len(by_question_odd)
    merged: dict[str, dict[str, dict]] = {}
    for i, (q, choices) in enumerate(sorted(by_question_odd.items(), key=lambda x: int(x[0])), start=1):
        merged[str(i)] = choices
    for i, (q, choices) in enumerate(sorted(by_question_even.items(), key=lambda x: int(x[0])), start=1):
        merged[str(n_odd + i)] = choices
    return merged


def _normalized_path_for_page(page_path: Path) -> Path:
    """Path to normalized image for this cached page: cache_normalized / subdir / page_N_bin.png."""
    cache = Path(IMAGE_NORMALIZE["cache_root"]).resolve()
    suffix = "_bin" if IMAGE_NORMALIZE.get("binarize", True) else "_normalized"
    subdir = page_path.parent.name
    return cache / subdir / f"{page_path.stem}{suffix}.png"


def _roi_name_sort_key(entry: dict[str, Any]) -> tuple[float | int, str]:
    """Sort key: numeric names (1,2,...,10,11) first in order; non-numeric last by name."""
    name = entry.get("name", "")
    if isinstance(name, str) and name.isdigit():
        return (int(name), "")
    return (float("inf"), str(name))


def _sort_data_by_roi_name(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort data list by ROI name (numeric order: 1, 2, ..., 9, 10, 11, ...)."""
    return sorted(data, key=_roi_name_sort_key)


def _merge_page_data(
    odd_items: list[dict[str, Any]],
    even_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge per-page ROI extraction outputs into one list for a page pair.

    Each row includes ``page_in_pair``: ``\"odd\"`` (side-a page) or ``\"even\"`` (side-b page)
    so downstream tools (e.g. human OCR review) can patch the correct ``data[]`` row.
    """
    merged: list[dict[str, Any]] = []
    for item in odd_items or []:
        d = dict(item)
        d["page_in_pair"] = "odd"
        merged.append(d)
    for item in even_items or []:
        d = dict(item)
        d["page_in_pair"] = "even"
        merged.append(d)
    return merged


def _log(msg: str, verbose: bool, file=None) -> None:
    if verbose:
        (file or sys.stdout).write(msg + "\n")
        (file or sys.stdout).flush()


def _write_debug_snapshot(debug_data: dict[str, Any], dbg_path: Path, verbose: bool) -> None:
    """
    Atomically write one compact debug snapshot.
    """
    try:
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dbg_path.with_suffix(dbg_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(debug_data, default=str, separators=(",", ":")))
        tmp_path.replace(dbg_path)
    except Exception as exc:  # pragma: no cover - debug I/O should not kill pipeline
        _log(f"      [debug] Warning: failed to update {dbg_path}: {exc}", verbose)


def _config_snapshot_b64_gzip() -> str:
    """
    Return a base64(gzip(JSON(config))) snapshot string for debug artifacts.
    Includes uppercase module globals from config.py.
    """
    payload: dict[str, Any] = {}
    for key in dir(_config_module):
        if not key.isupper():
            continue
        value = getattr(_config_module, key, None)
        if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
            payload[key] = value
    raw = json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(raw)
    return base64.b64encode(compressed).decode("ascii")


def _load_pdf_pages(pdf_path: Path, verbose: bool, max_pages: int | None = None) -> list[Path]:
    """Step 1: PDF → cached page images. Respects max_pages when set."""
    _log("      Converting PDF to images...", verbose)
    pdf_to_images = _get_pdf_to_images()
    page_paths = pdf_to_images(pdf_path, use_tqdm=verbose, max_pages=max_pages)
    if not page_paths:
        _log("      No pages produced.", verbose)
        return []
    _log(f"      Cached {len(page_paths)} page(s).", verbose)
    return page_paths


def _detect_ids_and_form_types(
    page_paths: list[Path],
    verbose: bool,
    debug_out: dict | None = None,
) -> list[dict[str, Any]]:
    """Step 2: Form-type detection from side-a (odd) page headers; result is used for each page pair."""
    _log("[2/6] Detecting form type from page headers (side-a only)...", verbose)
    # Only odd-indexed pages (side a of each pair): 0, 2, 4, ...
    odd_indices = list(range(0, len(page_paths), 2))
    odd_paths = [page_paths[i] for i in odd_indices]
    extract_id_and_form_type_batch = _get_extract_id_form_batch()
    id_form_results = extract_id_and_form_type_batch(
        [str(p) for p in odd_paths],
        include_id=False,
        verbose=verbose,
        debug_out=debug_out,
    )
    while len(id_form_results) < len(odd_paths):
        id_form_results.append({"id": None, "form_type": None})
    # One result per pair: assign to both pages of the pair (idx 0,1 → result[0]; 2,3 → result[1]; ...)
    id_form_by_pair = id_form_results[: len(odd_indices)]

    page_infos: list[dict[str, Any]] = []
    for idx, p in enumerate(page_paths):
        page_num = int(p.stem.split("_")[1])
        side = "a" if page_num % 2 == 1 else "b"
        pair_idx = idx // 2
        id_form = id_form_by_pair[pair_idx] if pair_idx < len(id_form_by_pair) else {}
        form_type = (id_form or {}).get("form_type")
        schema_path = _schema_path_for_form(form_type, side)
        page_infos.append(
            {
                "page_num": page_num,
                "side": side,
                "form_type": form_type,
                # ID is extracted in Step 4 as a normal ROI-style OCR+LLM flow.
                "id": "",
                "schema_path": schema_path,
            }
        )

    detected_form_types = sorted({(x.get("form_type") or "") for x in page_infos if x.get("form_type")})
    _log(
        f"      Detected form types: {', '.join(detected_form_types) if detected_form_types else '(none)'}",
        verbose,
    )
    return page_infos


def _normalize_page_job(
    args: tuple[int, str, str | None, str | None, str | None, str]
) -> tuple[int, str, str]:
    """
    Worker for one page normalization:
      geometric normalization -> postprocess normalization.
    Returns string paths for stable inter-process transport.
    """
    idx, page_path_s, form_type, side, cache_subdir_s, stem = args
    normalize_image = _get_normalize_image()
    page_path = Path(page_path_s)
    out_homography = normalize_image(
        page_path,
        form_type=form_type,
        page_side=side,
        cache_subdir=cache_subdir_s,
        cache_name_stem=stem,
        mode="geometric",
        output_tag="homography",
    )
    out_mcq = normalize_image(
        out_homography,
        cache_subdir=cache_subdir_s,
        cache_name_stem=stem,
        mode="postprocess",
    )
    return idx, str(out_homography), str(out_mcq)


def _normalize_pages(
    page_paths: list[Path],
    page_infos: list[dict[str, Any]],
    verbose: bool,
) -> tuple[list[Path], list[Path]]:
    """Step 3: Normalize all pages with form-specific templates."""
    _log("[3/6] Normalizing pages with form-specific templates...", verbose)
    cache_subdir = page_paths[0].parent.name if page_paths else None
    try:
        worker_count = int(PDF_RECOGNITION.get("normalize_workers", 2) or 1)
    except Exception:
        worker_count = 1
    worker_count = max(1, worker_count)

    jobs: list[tuple[int, str, str | None, str | None, str | None, str]] = []
    for idx, p in enumerate(page_paths):
        info = page_infos[idx]
        jobs.append(
            (
                idx,
                str(p),
                info.get("form_type"),
                info.get("side"),
                cache_subdir,
                p.stem,
            )
        )

    results_by_idx: list[tuple[Path, Path] | None] = [None] * len(jobs)
    progress = None
    if verbose:
        progress = tqdm(
            total=len(jobs),
            desc="      Normalize",
            unit="page",
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            leave=False,
        )
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            future_to_idx = {ex.submit(_normalize_page_job, job): job[0] for job in jobs}
            for fut in as_completed(future_to_idx):
                idx, out_h_s, out_m_s = fut.result()
                results_by_idx[idx] = (Path(out_h_s), Path(out_m_s))
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    homography_paths: list[Path] = []
    mcq_paths: list[Path] = []
    for idx, pair in enumerate(results_by_idx):
        if pair is None:
            raise RuntimeError(f"Normalization missing result for page index {idx}")
        out_homography, out_mcq = pair
        homography_paths.append(out_homography)
        mcq_paths.append(out_mcq)
        info = page_infos[idx]
        info["normalized_homography_path"] = out_homography
        info["normalized_mcq_path"] = out_mcq
        info["normalized_path"] = out_homography
    _log(
        f"      Normalized {len(homography_paths)} page(s) "
        f"(homography + postprocess variants) with {worker_count} worker(s).",
        verbose,
    )
    return homography_paths, mcq_paths


def _analyze_all_pages(
    text_paths: list[Path],
    mcq_paths: list[Path],
    page_infos: list[dict[str, Any]],
    verbose: bool,
    debug_out: dict | None = None,
    *,
    pdf_stem: str = "",
    text_review_queue: list[dict[str, Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Step 4a: ROI extraction for all pages in one batch."""
    _log("[4/6] Extracting ROI data for all pages...", verbose)
    analyze_pages_batch = _get_analyze_pages_batch()
    use_processed_for_text = bool(PDF_RECOGNITION.get("use_processed_images_for_text_roi", False))
    batch_inputs: list[dict[str, Any]] = []
    for i in range(len(text_paths)):
        pair_index = i // 2
        page_in_pair = "odd" if i % 2 == 0 else "even"
        text_ocr_path = mcq_paths[i] if use_processed_for_text else text_paths[i]
        inp: dict[str, Any] = {
            # Primary page path is metadata/debug context.
            "page_image_path": text_paths[i],
            # Text ROI OCR path is configurable so we can keep default dual-channel
            # behavior or optionally route postprocessed pages into text OCR too.
            "text_image_path": text_ocr_path,
            # MCQ uses the post-processed variant tuned for mark detection.
            "mcq_image_path": mcq_paths[i],
            "form_type": page_infos[i].get("form_type"),
            "side": page_infos[i].get("side"),
            "pair_index": pair_index,
            "page_in_pair": page_in_pair,
            "pdf_stem": pdf_stem,
        }
        if text_review_queue is not None:
            inp["review_text_queue_out"] = text_review_queue
        batch_inputs.append(inp)
    return analyze_pages_batch(
        batch_inputs,
        debug_out=debug_out,
        use_tqdm=verbose,
        tqdm_desc="      ROI analyze",
        apply_text_llm_postprocess=False,
    )


def _regex_retry_cfg() -> dict[str, Any]:
    cfg = ROI_PAGE_RECOGNITION if isinstance(ROI_PAGE_RECOGNITION, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _normalize_text_value(value: Any) -> str:
    return " ".join(str(value or "").split())


def _parse_bbox_xyxy(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in value)
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _compile_output_regex(pattern_text: str | None) -> re.Pattern[str] | None:
    raw = str(pattern_text or "").strip()
    if not raw:
        return None
    try:
        return re.compile(raw)
    except re.error:
        return None


def _crop_gray_with_pad(gray, bbox: tuple[int, int, int, int], pad: int):
    x1, y1, x2, y2 = bbox
    h, w = gray.shape[:2]
    px = max(int(pad or 0), 0)
    cx1 = max(x1 - px, 0)
    cy1 = max(y1 - px, 0)
    cx2 = min(x2 + px, w)
    cy2 = min(y2 + px, h)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return gray[cy1:cy2, cx1:cx2]


def _ocr_retry_step_pad_8px(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    pad = int(cfg.get("ocr_regex_retry_pad_px", 8) or 8)
    return _crop_gray_with_pad(gray, bbox, pad)


def _ocr_retry_step_clahe_light(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    import cv2

    crop = _crop_gray_with_pad(gray, bbox, 0)
    if crop is None:
        return None
    clip = float(cfg.get("ocr_regex_retry_clahe_clip", 2.0) or 2.0)
    tile_n = int(cfg.get("ocr_regex_retry_clahe_tile", 8) or 8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_n, tile_n))
    return clahe.apply(crop)


def _ocr_retry_step_adaptive_mean(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    import cv2

    crop = _crop_gray_with_pad(gray, bbox, 0)
    if crop is None:
        return None
    block = int(cfg.get("ocr_regex_retry_adaptive_block", 31) or 31)
    if block % 2 == 0:
        block += 1
    if block < 3:
        block = 3
    c = float(cfg.get("ocr_regex_retry_adaptive_c_mean", 8) or 8)
    return cv2.adaptiveThreshold(
        crop,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )


def _ocr_retry_step_adaptive_gauss(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    import cv2

    crop = _crop_gray_with_pad(gray, bbox, 0)
    if crop is None:
        return None
    block = int(cfg.get("ocr_regex_retry_adaptive_block", 31) or 31)
    if block % 2 == 0:
        block += 1
    if block < 3:
        block = 3
    c = float(cfg.get("ocr_regex_retry_adaptive_c_gauss", 10) or 10)
    return cv2.adaptiveThreshold(
        crop,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )


def _ocr_retry_step_adaptive_binarize(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    """
    Dedicated adaptive binarization step (explicitly named for easy workflow tuning).
    """
    import cv2

    crop = _crop_gray_with_pad(gray, bbox, 0)
    if crop is None:
        return None
    block = int(cfg.get("ocr_regex_retry_adaptive_block", 31) or 31)
    if block % 2 == 0:
        block += 1
    if block < 3:
        block = 3
    c = float(cfg.get("ocr_regex_retry_adaptive_c_binarize", 9) or 9)
    bin_img = cv2.adaptiveThreshold(
        crop,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )
    # Light median cleanup to reduce pepper noise.
    return cv2.medianBlur(bin_img, 3)


_FSRCNN_MODEL_CACHE: dict[tuple[str, int], Any] = {}
_FSRCNN_UNAVAILABLE: set[tuple[str, int]] = set()


def _get_fsrcnn_upsampler(cfg: dict[str, Any]):
    """
    Lazy-load FSRCNN model once per (path, scale). Returns None when unavailable.
    """
    model_path = str(cfg.get("ocr_regex_retry_fsrcnn_model_path", "") or "").strip()
    if not model_path:
        return None
    scale = int(cfg.get("ocr_regex_retry_fsrcnn_scale", 2) or 2)
    key = (model_path, scale)
    if key in _FSRCNN_MODEL_CACHE:
        return _FSRCNN_MODEL_CACHE[key]
    if key in _FSRCNN_UNAVAILABLE:
        return None
    try:
        import cv2

        if not Path(model_path).exists():
            _FSRCNN_UNAVAILABLE.add(key)
            return None
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("fsrcnn", scale)
        _FSRCNN_MODEL_CACHE[key] = sr
        return sr
    except Exception:
        _FSRCNN_UNAVAILABLE.add(key)
        return None


def _ocr_retry_step_fsrcnn_x2(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    """
    FSRCNN super-resolution step.
    If model/runtime unavailable, optionally falls back to standard xN resize.
    """
    import cv2

    crop = _crop_gray_with_pad(gray, bbox, 0)
    if crop is None:
        return None
    scale = int(cfg.get("ocr_regex_retry_fsrcnn_scale", 2) or 2)
    scale = max(2, scale)
    fallback_resize = bool(cfg.get("ocr_regex_retry_fsrcnn_fallback_resize", True))

    def _fallback():
        if not fallback_resize:
            return None
        h, w = crop.shape[:2]
        return cv2.resize(
            crop,
            (max(1, w * scale), max(1, h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    upsampler = _get_fsrcnn_upsampler(cfg)
    if upsampler is None:
        return _fallback()
    try:
        up = upsampler.upsample(crop)
        if up is None:
            return _fallback()
        if len(up.shape) == 3:
            up = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        return up
    except Exception:
        return _fallback()


def _ocr_retry_step_pad_8px_clahe_light(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    import cv2

    pad = int(cfg.get("ocr_regex_retry_pad_px", 8) or 8)
    crop = _crop_gray_with_pad(gray, bbox, pad)
    if crop is None:
        return None
    clip = float(cfg.get("ocr_regex_retry_clahe_clip", 2.0) or 2.0)
    tile_n = int(cfg.get("ocr_regex_retry_clahe_tile", 8) or 8)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_n, tile_n))
    return clahe.apply(crop)


def _ocr_retry_step_pad_8px_adaptive_gauss(gray, bbox: tuple[int, int, int, int], cfg: dict[str, Any]):
    import cv2

    pad = int(cfg.get("ocr_regex_retry_pad_px", 8) or 8)
    crop = _crop_gray_with_pad(gray, bbox, pad)
    if crop is None:
        return None
    block = int(cfg.get("ocr_regex_retry_adaptive_block", 31) or 31)
    if block % 2 == 0:
        block += 1
    if block < 3:
        block = 3
    c = float(cfg.get("ocr_regex_retry_adaptive_c_gauss", 10) or 10)
    return cv2.adaptiveThreshold(
        crop,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        c,
    )


# Each OCR retry step is a dedicated method above.
# To change workflow order, edit ROI_PAGE_RECOGNITION["ocr_regex_retry_steps"] in config.py.
_OCR_REGEX_RETRY_STEP_METHODS = {
    "pad_8px": _ocr_retry_step_pad_8px,
    "clahe_light": _ocr_retry_step_clahe_light,
    "fsrcnn_x2": _ocr_retry_step_fsrcnn_x2,
    "adaptive_binarize": _ocr_retry_step_adaptive_binarize,
    "adaptive_mean": _ocr_retry_step_adaptive_mean,
    "adaptive_gauss": _ocr_retry_step_adaptive_gauss,
    "pad_8px_clahe_light": _ocr_retry_step_pad_8px_clahe_light,
    "pad_8px_adaptive_gauss": _ocr_retry_step_pad_8px_adaptive_gauss,
}


def _run_ocr_regex_retry_step_queue(
    pending_entries: list[dict[str, Any]],
    *,
    step_name: str,
    use_tqdm: bool = False,
    tqdm_desc: str | None = None,
) -> dict[str, str]:
    """
    Execute one queued OCR retry step for all pending rows, then return OCR texts by uid.
    """
    from ocr_engine import ocr_raw
    import cv2
    import tempfile

    step_fn = _OCR_REGEX_RETRY_STEP_METHODS.get(step_name)
    if step_fn is None:
        return {}
    cfg = _regex_retry_cfg()
    image_cache: dict[str, Any] = {}
    out: dict[str, str] = {}

    entries_iter = pending_entries
    if use_tqdm and pending_entries:
        entries_iter = tqdm(
            pending_entries,
            total=len(pending_entries),
            desc=tqdm_desc or f"      OCR retry [{step_name}]",
            unit="roi",
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            leave=False,
        )
    for entry in entries_iter:
        uid = str(entry.get("uid", "") or "")
        image_path = str(entry.get("image_path", "") or "").strip()
        bbox = _parse_bbox_xyxy(entry.get("bbox_xyxy"))
        if not uid or not image_path or bbox is None:
            continue
        gray = image_cache.get(image_path)
        if gray is None:
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image_cache[image_path] = gray
        if gray is None:
            continue
        transformed = step_fn(gray, bbox, cfg)
        if transformed is None or getattr(transformed, "size", 0) == 0:
            continue
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), transformed)
            prompt_override = str(entry.get("ocr_prompt_override", "") or "").strip() or None
            raw = ocr_raw(tmp_path, prompt_override=prompt_override)
            raw_out = raw if isinstance(raw, dict) else {}
            out[uid] = _normalize_text_value(raw_out.get("detected_text", ""))
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
    return out


def _run_deferred_text_llm_postprocess(
    all_page_data: list[list[dict[str, Any]]],
    *,
    verbose: bool,
    debug_out: dict | None = None,
) -> None:
    """
    Run non-header text ROI postprocessing once after all OCR extraction is complete.
    Then, for ROIs with `ocr_output_regex`, run queued OCR retry steps in rounds:
      1) OCR all pending rows for current step
      2) LLM normalize all those OCR outputs together
      3) re-check regex; keep unresolved rows queued for next step
    """
    from text_roi_llm import postprocess_text_rois

    text_by_uid: dict[str, str] = {}
    meta_by_uid: dict[str, dict[str, Any]] = {}
    row_refs: dict[str, dict[str, Any]] = {}
    retry_ctx_by_uid: dict[str, dict[str, Any]] = {}

    for page_idx, page_rows in enumerate(all_page_data):
        for row_idx, row in enumerate(page_rows or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("kind", "")) != "text":
                continue
            uid = f"p{page_idx}:r{row_idx}:{row.get('name','')}"
            text_by_uid[uid] = _normalize_text_value(row.get("text", ""))
            meta_by_uid[uid] = {
                "llm_field_data_type": row.get("_llm_field_data_type"),
                "llm_validation_rules": row.get("_llm_validation_rules"),
                "llm_prompt_instruction": row.get("_llm_prompt_instruction"),
                "llm_prompt_override": row.get("_llm_prompt_override"),
            }
            retry_ctx_by_uid[uid] = {
                "ocr_output_regex": row.get("_ocr_output_regex"),
                "image_path": row.get("_ocr_retry_image_path"),
                "bbox_xyxy": row.get("_ocr_retry_bbox_xyxy"),
                "ocr_prompt_override": row.get("_ocr_retry_prompt_override"),
            }
            row_refs[uid] = row

    total_text_rows = sum(
        1
        for page_rows in all_page_data
        for row in (page_rows or [])
        if isinstance(row, dict) and str(row.get("kind", "")) == "text"
    )

    if not text_by_uid:
        if isinstance(debug_out, dict):
            debug_out["text_llm_deferred"] = {
                "enabled": False,
                "reason": "no_text_rows",
                "total_text_rows": total_text_rows,
                "queued_rows": 0,
                "processed_rows": 0,
            }
        return

    _log(
        f"      Deferred text-LLM postprocess on {len(text_by_uid)} text row(s)...",
        verbose,
    )
    llm_debug_initial: dict[str, Any] | None = {} if isinstance(debug_out, dict) else None
    initial_processed = postprocess_text_rois(
        text_by_uid,
        roi_meta_by_name=meta_by_uid,
        debug_out=llm_debug_initial,
        use_tqdm=bool(verbose and _tqdm_enabled()),
        tqdm_desc="      Text LLM deferred",
    )

    current_values: dict[str, str] = dict(text_by_uid)
    for uid, value in (initial_processed or {}).items():
        current_values[uid] = _normalize_text_value(value)

    retry_cfg = _regex_retry_cfg()
    retry_enabled = bool(retry_cfg.get("ocr_regex_retry_enabled", True))
    configured_steps = retry_cfg.get("ocr_regex_retry_steps") or []
    retry_steps = [
        s for s in configured_steps
        if isinstance(s, str) and s in _OCR_REGEX_RETRY_STEP_METHODS
    ]

    regex_by_uid: dict[str, re.Pattern[str]] = {}
    invalid_regex_uids: list[str] = []
    for uid, ctx in retry_ctx_by_uid.items():
        pat = _compile_output_regex(ctx.get("ocr_output_regex"))
        if pat is not None:
            regex_by_uid[uid] = pat
            continue
        raw = str(ctx.get("ocr_output_regex", "") or "").strip()
        if raw:
            invalid_regex_uids.append(uid)

    pending = [
        uid for uid, pat in regex_by_uid.items()
        if not bool(pat.fullmatch(_normalize_text_value(current_values.get(uid, ""))))
    ]

    # Per-ROI retry trace for debug visibility.
    retry_trace_by_uid: dict[str, dict[str, Any]] = {}
    for uid, pat in regex_by_uid.items():
        row = row_refs.get(uid) if isinstance(row_refs.get(uid), dict) else {}
        retry_trace_by_uid[uid] = {
            "uid": uid,
            "roi_name": (row or {}).get("name"),
            "regex": pat.pattern,
            "initial_text_after_llm": _normalize_text_value(current_values.get(uid, "")),
            "initial_match": bool(pat.fullmatch(_normalize_text_value(current_values.get(uid, "")))),
            "steps": [],
            "final_text_after_retries": None,
            "final_match": None,
        }

    retry_rounds: list[dict[str, Any]] = []
    if retry_enabled and pending and retry_steps:
        _log(f"      OCR regex retries queued for {len(pending)} row(s)...", verbose)
        step_iter = retry_steps
        step_bar = None
        if verbose:
            step_bar = tqdm(
                retry_steps,
                total=len(retry_steps),
                desc="      Regex retry",
                unit="step",
                disable=not _tqdm_enabled(),
                dynamic_ncols=True,
                leave=False,
            )
            step_iter = step_bar
        for step_name in step_iter:
            if not pending:
                break
            if step_bar is not None:
                step_bar.set_postfix_str(f"pending={len(pending)}", refresh=False)
            queue = []
            for uid in pending:
                ctx = retry_ctx_by_uid.get(uid, {})
                queue.append(
                    {
                        "uid": uid,
                        "image_path": ctx.get("image_path"),
                        "bbox_xyxy": ctx.get("bbox_xyxy"),
                        "ocr_prompt_override": ctx.get("ocr_prompt_override"),
                    }
                )
            step_ocr_texts = _run_ocr_regex_retry_step_queue(
                queue,
                step_name=step_name,
                use_tqdm=bool(verbose),
                tqdm_desc=f"      OCR retry [{step_name}]",
            )
            queue_uids = [str(q.get("uid", "") or "") for q in queue]
            if not step_ocr_texts:
                for uid in queue_uids:
                    tr = retry_trace_by_uid.get(uid)
                    if isinstance(tr, dict):
                        tr.setdefault("steps", []).append(
                            {
                                "step": step_name,
                                "was_pending_at_step_start": True,
                                "ocr_output_generated": False,
                                "ocr_text_before_llm": None,
                                "llm_text_after_step": None,
                                "matched_after_step": False,
                            }
                        )
                retry_rounds.append(
                    {
                        "step": step_name,
                        "pending_in": len(pending),
                        "ocr_outputs": 0,
                        "llm_processed": 0,
                        "pending_out": len(pending),
                    }
                )
                continue
            llm_step_debug: dict[str, Any] | None = {} if isinstance(debug_out, dict) else None
            step_meta = {uid: meta_by_uid.get(uid, {}) for uid in step_ocr_texts}
            step_processed = postprocess_text_rois(
                step_ocr_texts,
                roi_meta_by_name=step_meta,
                debug_out=llm_step_debug,
                use_tqdm=bool(verbose and _tqdm_enabled()),
                tqdm_desc=f"      Text LLM retry [{step_name}]",
            )
            processed_norm = {
                uid: _normalize_text_value(val)
                for uid, val in (step_processed or {}).items()
            }
            processed_count = 0
            for uid, norm_value in processed_norm.items():
                current_values[uid] = norm_value
                processed_count += 1
            for uid in queue_uids:
                tr = retry_trace_by_uid.get(uid)
                if not isinstance(tr, dict):
                    continue
                pat = regex_by_uid.get(uid)
                llm_value = processed_norm.get(uid)
                matched = bool(pat and llm_value is not None and pat.fullmatch(llm_value))
                tr.setdefault("steps", []).append(
                    {
                        "step": step_name,
                        "was_pending_at_step_start": True,
                        "ocr_output_generated": uid in step_ocr_texts,
                        "ocr_text_before_llm": step_ocr_texts.get(uid),
                        "llm_text_after_step": llm_value,
                        "matched_after_step": matched,
                    }
                )
            pending = [
                uid for uid in pending
                if not bool(regex_by_uid[uid].fullmatch(_normalize_text_value(current_values.get(uid, ""))))
            ]
            retry_rounds.append(
                {
                    "step": step_name,
                    "pending_in": len(queue),
                    "ocr_outputs": len(step_ocr_texts),
                    "llm_processed": processed_count,
                    "pending_out": len(pending),
                    "llm_details": llm_step_debug if isinstance(llm_step_debug, dict) else {},
                }
            )
        if step_bar is not None:
            step_bar.close()

    for uid, tr in retry_trace_by_uid.items():
        if not isinstance(tr, dict):
            continue
        final_text = _normalize_text_value(current_values.get(uid, ""))
        tr["final_text_after_retries"] = final_text
        tr["final_match"] = bool(regex_by_uid[uid].fullmatch(final_text))

    for uid, row in row_refs.items():
        if isinstance(row, dict):
            row["text"] = _normalize_text_value(current_values.get(uid, ""))

    if isinstance(debug_out, dict):
        details = llm_debug_initial if isinstance(llm_debug_initial, dict) else {}
        debug_out["text_llm_deferred"] = {
            "enabled": True,
            "total_text_rows": total_text_rows,
            "queued_rows": len(text_by_uid),
            "processed_rows": len(current_values),
            "model": details.get("model"),
            "details": details,
            "regex_retry": {
                "enabled": retry_enabled,
                "configured_steps": retry_steps,
                "rows_with_regex": len(regex_by_uid),
                "invalid_regex_uids": invalid_regex_uids,
                "initial_mismatch_count": len(
                    [
                        uid for uid, pat in regex_by_uid.items()
                        if not bool(pat.fullmatch(_normalize_text_value((initial_processed or {}).get(uid, ""))))
                    ]
                ),
                "remaining_mismatch_count": len(pending),
                "rounds": retry_rounds,
                "per_roi": retry_trace_by_uid,
            },
        }

    # Remove internal metadata from rows before output assembly.
    for page_rows in all_page_data:
        for row in page_rows or []:
            if not isinstance(row, dict):
                continue
            row.pop("_llm_field_data_type", None)
            row.pop("_llm_validation_rules", None)
            row.pop("_llm_prompt_instruction", None)
            row.pop("_llm_prompt_override", None)
            row.pop("_ocr_output_regex", None)
            row.pop("_ocr_retry_image_path", None)
            row.pop("_ocr_retry_bbox_xyxy", None)
            row.pop("_ocr_retry_prompt_override", None)


def _header_id_schema_and_roi() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load side-a header schema and return raw schema + ROI entry for name='id'."""
    schema_dir = Path(
        PDF_RECOGNITION.get("schema_dir", Path(__file__).resolve().parent.parent / "data" / "roi_schemas")
    ).resolve()
    key = str(PDF_RECOGNITION.get("schema_key_sidea", "schema_sidea") or "schema_sidea").strip()
    schema_path = schema_dir / f"{key}.json"
    raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    rois = raw_schema.get("rois")
    if not isinstance(rois, list):
        raise ValueError(f"Invalid header schema rois list: {schema_path}")
    for roi in rois:
        if isinstance(roi, dict) and str(roi.get("name", "")).strip().lower() == "id":
            return raw_schema, roi
    raise ValueError(f"Header schema missing ROI named 'id': {schema_path}")


def _extract_text_from_roi_on_page(
    page_path: Path,
    *,
    raw_schema: dict[str, Any],
    roi: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """OCR one ROI from one page image using the same OCR path as text ROIs."""
    import cv2
    import tempfile
    from ocr_engine import ocr_confidence_stats, ocr_raw

    img = cv2.imread(str(page_path))
    if img is None:
        return "", {}, ocr_confidence_stats({})
    ih, iw = img.shape[:2]
    try:
        rx = float(roi.get("x", 0))
        ry = float(roi.get("y", 0))
        rw = float(roi.get("w", 0))
        rh = float(roi.get("h", 0))
    except Exception:
        return "", {}, ocr_confidence_stats({})
    if rw <= 0 or rh <= 0:
        return "", {}, ocr_confidence_stats({})

    ref_w_raw = raw_schema.get("image_width")
    ref_h_raw = raw_schema.get("image_height")
    try:
        ref_w = int(ref_w_raw) if ref_w_raw is not None else None
    except Exception:
        ref_w = None
    try:
        ref_h = int(ref_h_raw) if ref_h_raw is not None else None
    except Exception:
        ref_h = None

    if ref_w and ref_h and ref_w > 0 and ref_h > 0 and (ref_w != iw or ref_h != ih):
        sx = iw / ref_w
        sy = ih / ref_h
    else:
        sx = sy = 1.0

    x1 = max(int(round(rx * sx)), 0)
    y1 = max(int(round(ry * sy)), 0)
    x2 = min(int(round((rx + rw) * sx)), iw)
    y2 = min(int(round((ry + rh) * sy)), ih)
    if x2 <= x1 or y2 <= y1:
        return "", {}, ocr_confidence_stats({})

    crop = img[y1:y2, x1:x2]
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cv2.imwrite(str(tmp_path), crop)
        roi_meta = roi if isinstance(roi, dict) else {}
        prompt_override = str(
            roi_meta.get("ocr_prompt_override") or roi_meta.get("prompt_override") or ""
        ).strip() or None
        raw_out = ocr_raw(tmp_path, prompt_override=prompt_override)
        raw_result = raw_out if isinstance(raw_out, dict) else {}
        text = str(raw_result.get("detected_text", "") or "").strip()
        stats = ocr_confidence_stats(raw_result)
        return text, raw_result, stats
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _extract_pair_ids_from_roi(
    text_paths: list[Path],
    *,
    verbose: bool,
    debug_out: dict | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Extract one ID per odd page (one per pair) using ROI OCR + deferred LLM postprocess.
    Returns: pair_index -> {id, ocr_text_pre_llm, ocr_text_post_llm, raw_ocr, stats}
    """
    from id_form_llm import _extract_id_from_ocr_raw
    from text_roi_llm import postprocess_text_rois

    raw_schema, id_roi = _header_id_schema_and_roi()
    odd_indices = list(range(0, len(text_paths), 2))
    id_text_by_uid: dict[str, str] = {}
    id_raw_by_uid: dict[str, dict[str, Any]] = {}
    id_stats_by_uid: dict[str, dict[str, Any]] = {}
    pair_by_uid: dict[str, int] = {}

    iterator = odd_indices
    if verbose:
        iterator = tqdm(
            odd_indices,
            desc="      ID ROI OCR (odd pages)",
            unit="pair",
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            leave=False,
        )
    for page_idx in iterator:
        pair_index = page_idx // 2
        uid = f"pair{pair_index}"
        text, raw_ocr, stats = _extract_text_from_roi_on_page(
            text_paths[page_idx],
            raw_schema=raw_schema,
            roi=id_roi,
        )
        id_text_by_uid[uid] = " ".join(str(text or "").split())
        id_raw_by_uid[uid] = raw_ocr if isinstance(raw_ocr, dict) else {}
        id_stats_by_uid[uid] = stats if isinstance(stats, dict) else {}
        pair_by_uid[uid] = pair_index

    meta = {
        "id": {
            "llm_field_data_type": id_roi.get("llm_field_data_type"),
            "llm_validation_rules": id_roi.get("llm_validation_rules"),
            "llm_prompt_instruction": id_roi.get("llm_prompt_instruction"),
            "llm_prompt_override": id_roi.get("llm_prompt_override"),
        }
    }
    llm_inputs = {uid: id_text_by_uid.get(uid, "") for uid in id_text_by_uid}
    llm_meta = {uid: meta["id"] for uid in llm_inputs}
    llm_debug: dict[str, Any] | None = {} if isinstance(debug_out, dict) else None
    try:
        processed = postprocess_text_rois(
            llm_inputs,
            roi_meta_by_name=llm_meta,
            debug_out=llm_debug,
            use_tqdm=bool(verbose and _tqdm_enabled()),
            tqdm_desc="      ID LLM deferred",
        )
    except Exception as exc:
        processed = dict(llm_inputs)
        if llm_debug is not None:
            llm_debug["error"] = str(exc)

    out: dict[int, dict[str, Any]] = {}
    for uid, pair_index in pair_by_uid.items():
        text_pre = id_text_by_uid.get(uid, "")
        text_post = "" if processed.get(uid) is None else str(processed.get(uid, ""))
        id_value = _extract_id_from_ocr_raw(id_raw_by_uid.get(uid), text_post or text_pre)
        out[pair_index] = {
            "id": id_value,
            "ocr_text_pre_llm": text_pre,
            "ocr_text_post_llm": text_post,
            "raw_ocr": id_raw_by_uid.get(uid, {}),
            "stats": id_stats_by_uid.get(uid, {}),
        }

    if isinstance(debug_out, dict):
        debug_out["id_roi_deferred"] = {
            "pair_count": len(out),
            "llm": llm_debug if isinstance(llm_debug, dict) else {},
            "per_pair": {
                str(k): {
                    "id": v.get("id"),
                    "ocr_text_pre_llm": v.get("ocr_text_pre_llm"),
                    "ocr_text_post_llm": v.get("ocr_text_post_llm"),
                    "raw_ocr": v.get("raw_ocr"),
                    "stats": v.get("stats"),
                }
                for k, v in out.items()
            },
        }
    return out


def _build_pair_items(
    text_paths: list[Path],
    mcq_paths: list[Path],
    page_infos: list[dict[str, Any]],
    all_page_data: list[list[dict[str, Any]]],
    verbose: bool,
) -> list[dict[str, Any]]:
    """Step 4b: Group pages into pairs and merge their ROI data."""
    num_pairs = (len(text_paths) + 1) // 2
    _log(f"      Building {num_pairs} page pair item(s)...", verbose)

    items: list[dict[str, Any]] = []
    pair_indices = list(range(0, len(text_paths), 2))
    if verbose:
        pair_indices = tqdm(
            pair_indices,
            desc="      Pair",
            unit="item",
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            leave=False,
        )
    for i in pair_indices:
        page_odd = i + 1
        page_even = i + 2 if i + 1 < len(text_paths) else None
        info_odd = page_infos[i]
        info_even = page_infos[i + 1] if page_even is not None else None
        form_type_value = (info_odd.get("form_type") or (info_even.get("form_type") if info_even else None))

        odd_data = all_page_data[i]
        even_data: list[dict[str, Any]] = []
        if page_even is not None and info_even is not None:
            even_data = all_page_data[i + 1]
        merged_data = _merge_page_data(odd_data, even_data)
        if PDF_RECOGNITION.get("sort_data_by_roi_name", False):
            merged_data = _sort_data_by_roi_name(merged_data)

        def _find_id_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("kind", "")).strip().lower() != "text":
                    continue
                if str(row.get("name", "")).strip().lower() != "id":
                    continue
                return row
            return None

        id_row_odd = _find_id_row(odd_data)
        id_row_even = _find_id_row(even_data)
        id_odd = str((id_row_odd or {}).get("text", "") or "").strip()
        id_even = str((id_row_even or {}).get("text", "") or "").strip()
        id_value = id_odd or id_even

        entry: dict[str, Any] = {
            "id": id_value,
            "form_type": form_type_value,
            "page_odd": page_odd,
            "page_even": page_even,
            "data": merged_data,
        }
        odd_text_path = text_paths[i]
        odd_mcq_path = mcq_paths[i]
        entry["normalized_page_odd"] = _rel_from_project_root(odd_text_path)
        entry["normalized_page_odd_mcq"] = _rel_from_project_root(odd_mcq_path)
        if page_even is not None and i + 1 < len(text_paths):
            entry["normalized_page_even"] = _rel_from_project_root(text_paths[i + 1])
            entry["normalized_page_even_mcq"] = _rel_from_project_root(mcq_paths[i + 1])
        else:
            entry["normalized_page_even"] = None
            entry["normalized_page_even_mcq"] = None

        id_row_for_stats = id_row_odd if id_odd else id_row_even
        if isinstance(id_row_for_stats, dict):
            if id_row_for_stats.get("ocr_min_score") is not None:
                try:
                    id_score = round(float(id_row_for_stats.get("ocr_min_score")), 6)
                except (TypeError, ValueError):
                    id_score = None
                if id_score is not None:
                    entry["id_ocr_min_score"] = id_score
                    entry["id_ocr_confidence_score"] = id_score
            if id_row_for_stats.get("ocr_mean_score") is not None:
                try:
                    entry["id_ocr_mean_score"] = round(float(id_row_for_stats.get("ocr_mean_score")), 6)
                except (TypeError, ValueError):
                    pass
            if id_row_for_stats.get("ocr_confidence_label") is not None:
                entry["id_ocr_confidence_label"] = str(id_row_for_stats.get("ocr_confidence_label")).lower()
            if id_row_for_stats.get("ocr_needs_human_review") is not None:
                entry["id_ocr_needs_human_review"] = bool(id_row_for_stats.get("ocr_needs_human_review"))
            if id_row_for_stats.get("ocr_selected_model") is not None:
                entry["id_ocr_selected_model"] = str(id_row_for_stats.get("ocr_selected_model"))
            if id_row_for_stats.get("ocr_selected_stage_index") is not None:
                entry["id_ocr_selected_stage_index"] = id_row_for_stats.get("ocr_selected_stage_index")

        items.append(entry)

    _log(f"      Extracted {len(items)} item(s).", verbose)
    return items


def _write_output(
    pdf_path: Path,
    pdf_stem: str,
    items: list[dict[str, Any]],
    output_dir: Path | str | None,
    write_json: bool,
    verbose: bool,
    *,
    human_review: dict[str, Any] | None = None,
) -> Path | None:
    """Step 5: Write final JSON. Returns output file path when write_json True, else None."""
    cfg = PDF_RECOGNITION
    _log("[5/6] Writing JSON output...", verbose)
    out_dir = Path(output_dir) if output_dir is not None else Path(
        cfg.get("output_dir", Path(__file__).resolve().parent.parent / "output" / "recognition")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{pdf_stem}.json"
    if not write_json:
        _log("      Skipped (write_json=False).", verbose)
        _log("Done.", verbose)
        return None
    body: dict[str, Any] = {
        "pdf_path": str(pdf_path.resolve()),
        "pdf_stem": pdf_stem,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "item_count": len(items),
        "items": items,
    }
    if human_review is not None:
        body["human_review"] = human_review
    out_file.write_text(json.dumps(body, indent=2))
    _log(f"      Wrote {out_file}.", verbose)
    _log("Done.", verbose)
    return out_file

def _run_workflow(
    pdf_path: str | Path,
    output_dir: Path | str | None = None,
    *,
    write_json: bool = True,
    verbose: bool = True,
    max_pages: int | None = None,
    recache: bool = False,
    debug: bool = False,
    debug_path: Path | str | None = None,
    output_suffix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run full workflow: PDF → images → ID/form-type check → form-based normalization →
    ROI extraction; pair pages; merge; optionally write JSON.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory for output JSON (default: from config PDF_RECOGNITION["output_dir"]).
        write_json: If True, write results to {output_dir}/{pdf_stem}.json.
        verbose: If True, print step messages and show tqdm progress bars.
        debug: If True, collect and write debug data (OCR, LLM, normalization, ROI) to a file.
        debug_path: Where to write debug JSON; default {output_dir}/{pdf_stem}_debug.json.

    Returns:
        List of items, one per pair of pages. Each item:
        - id: str (from odd page, or even if only one page)
        - form_type: str | None (odd page preferred, else even page)
        - page_odd: int (1-based)
        - page_even: int | None
        - data: list[dict] — merged ROI outputs from odd/even pages
          (each entry has at least: name, kind, text)
    """
    workflow_start = time.perf_counter()
    workflow_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cfg = PDF_RECOGNITION
    out_dir = Path(output_dir) if output_dir is not None else Path(
        cfg.get("output_dir", Path(__file__).resolve().parent.parent / "output" / "recognition")
    )
    stem = _sanitize_pdf_stem(pdf_path.name)
    output_stem = f"{stem}{_sanitize_output_suffix(output_suffix)}"

    debug_data: dict[str, Any] | None = None
    dbg_path: Path | None = None
    if debug:
        dbg_path = Path(debug_path) if debug_path else out_dir / f"{output_stem}_debug.json"
        debug_data = {
            "description": "Pipeline debug: OCR (form-type header + text ROIs incl. ID), LLM (form-type + deferred text), normalization, MCQ recognition.",
            "pdf_path": str(pdf_path),
            "pdf_stem": output_stem,
            "config_b64_gzip": _config_snapshot_b64_gzip(),
            "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "running",
            "current_step": None,
            "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "steps": {},
            "step_status": {},
        }
        _write_debug_snapshot(debug_data, dbg_path, verbose)

    def _debug_step_begin(step_key: str) -> float:
        t0 = time.perf_counter()
        if debug_data is None or dbg_path is None:
            return t0
        meta = debug_data.setdefault("step_status", {}).setdefault(step_key, {})
        meta["status"] = "running"
        meta["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        debug_data["current_step"] = step_key
        debug_data["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_debug_snapshot(debug_data, dbg_path, verbose)
        return t0

    def _debug_step_done(step_key: str, payload: dict[str, Any], started_at: float) -> None:
        if debug_data is None or dbg_path is None:
            return
        debug_data["steps"][step_key] = payload
        meta = debug_data.setdefault("step_status", {}).setdefault(step_key, {})
        meta["status"] = "done"
        meta["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta["elapsed_sec"] = round(time.perf_counter() - started_at, 4)
        debug_data["current_step"] = None
        debug_data["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_debug_snapshot(debug_data, dbg_path, verbose)

    _log(f"[1/6] PDF: {pdf_path.name}", verbose)
    _log(f"      Workflow start (UTC): {workflow_start_utc}", verbose)

    try:
        # Optional: clear existing caches for this PDF
        if recache:
            cache_root = Path(PATHS["cache"]).resolve()
            norm_root = Path(IMAGE_NORMALIZE["cache_root"]).resolve()
            subdir_name = stem
            for root in (cache_root, norm_root):
                target = root / subdir_name
                if target.exists():
                    _log(f"      Removing existing cache: {target}", verbose)
                    shutil.rmtree(target, ignore_errors=True)

        # Step 1: PDF → images (only up to max_pages when set)
        t_step_1 = _debug_step_begin("1_pdf_to_images")
        page_paths = _load_pdf_pages(pdf_path, verbose, max_pages=max_pages)
        if max_pages is not None and max_pages > 0:
            _log(f"      Limited to first {len(page_paths)} page(s) (--max-pages={max_pages}).", verbose)
        _debug_step_done(
            "1_pdf_to_images",
            {
                "page_paths": [str(p) for p in page_paths],
                "count": len(page_paths),
                "max_pages": max_pages,
            },
            t_step_1,
        )
        if not page_paths:
            if debug_data is not None and dbg_path is not None:
                debug_data["status"] = "completed"
                debug_data["completion_reason"] = "no_pages_produced"
                debug_data["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                debug_data["elapsed_sec"] = round(time.perf_counter() - workflow_start, 4)
                debug_data["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _write_debug_snapshot(debug_data, dbg_path, verbose)
            elapsed = time.perf_counter() - workflow_start
            _log(f"[done] Workflow total time: {elapsed:.2f}s (no pages produced)", verbose)
            return []

        # Step 2: form type (header OCR + LLM)
        t_step_2 = _debug_step_begin("2_id_form")
        step2_debug: dict | None = {} if debug_data is not None else None
        page_infos = _detect_ids_and_form_types(
            page_paths,
            verbose,
            debug_out=step2_debug if debug_data else None,
        )
        if debug_data is not None:
            payload_2 = dict(step2_debug or {})
            id_form = payload_2.get("id_form")
            if isinstance(id_form, dict):
                # Step 2 is now form-type-only; keep only form-type debug fields.
                for row in (id_form.get("ocr_per_page") or []):
                    if not isinstance(row, dict):
                        continue
                    row.pop("ocr_text_id", None)
                    row.pop("raw_ocr_id", None)
                    parsed = row.get("parsed_result")
                    if isinstance(parsed, dict):
                        parsed.pop("id", None)
                parsed_results = id_form.get("parsed_results")
                if isinstance(parsed_results, list):
                    for row in parsed_results:
                        if isinstance(row, dict):
                            row.pop("id", None)
            payload_2["page_info_count"] = len(page_infos)
            payload_2["note"] = "Step 2 runs form-type detection only; ID is handled as a normal ROI in Step 4."
            _debug_step_done("2_id_form", payload_2, t_step_2)

        # Step 3: Normalize pages
        t_step_3 = _debug_step_begin("3_normalize")
        normalized_text_paths, normalized_mcq_paths = _normalize_pages(
            page_paths, page_infos, verbose
        )
        if debug_data is not None:
            _debug_step_done(
                "3_normalize",
                {
                    "per_page": [
                        {
                            "page_index": i,
                            "page_path": str(page_paths[i]),
                            "template_key": (
                                f"{page_infos[i].get('form_type') or ''}_{page_infos[i].get('side') or ''}".strip("_")
                                or None
                            ),
                            "normalized_text_path": str(normalized_text_paths[i]),
                            "normalized_mcq_path": str(normalized_mcq_paths[i]),
                        }
                        for i in range(len(page_paths))
                    ],
                },
                t_step_3,
            )

        # Step 4: ROI extraction + pair assembly (OCR text OCR + MCQ)
        t_step_4 = _debug_step_begin("4_roi")
        step4_debug: dict | None = {} if debug_data is not None else None
        text_review_queue: list[dict[str, Any]] = []
        all_page_data = _analyze_all_pages(
            normalized_text_paths,
            normalized_mcq_paths,
            page_infos,
            verbose,
            debug_out=step4_debug if debug_data else None,
            pdf_stem=output_stem,
            text_review_queue=text_review_queue,
        )
        _run_deferred_text_llm_postprocess(
            all_page_data,
            verbose=verbose,
            debug_out=step4_debug if debug_data else None,
        )
        items = _build_pair_items(
            normalized_text_paths,
            normalized_mcq_paths,
            page_infos,
            all_page_data,
            verbose,
        )
        if debug_data is not None:
            payload_4 = dict(step4_debug or {})
            payload_4["text_review_queue_count"] = len(text_review_queue)
            payload_4["pair_item_count"] = len(items)
            _debug_step_done("4_roi", payload_4, t_step_4)

        # Human-in-the-loop: low text OCR confidence (optional metadata block in JSON)
        human_review_block: dict[str, Any] | None = None
        if build_human_review_block is not None:
            human_review_block = build_human_review_block(
                items, output_stem, cfg, text_ocr_queue=text_review_queue
            )

        # Step 5: Output
        t_step_5 = _debug_step_begin("5_output")
        out_file = _write_output(
            pdf_path,
            output_stem,
            items,
            output_dir,
            write_json,
            verbose,
            human_review=human_review_block,
        )
        if debug_data is not None:
            pending = (human_review_block or {}).get("pending")
            _debug_step_done(
                "5_output",
                {
                    "output_path": str(out_file) if out_file else None,
                    "item_count": len(items),
                    "human_review_pending_count": (
                        len(pending) if isinstance(pending, list) else 0
                    ),
                },
                t_step_5,
            )

        # Step 6: XLSX data entry — group items by form_type, fill templates, merge
        _log("[6/6] Filling Excel data-entry workbook...", verbose)
        t_step_6 = _debug_step_begin("6_xlsx")
        xlsx_payload: dict[str, Any] = {"output_path": None}
        try:
            from xlsx_data_entry import fill_from_pipeline

            xlsx_path = fill_from_pipeline(items, pdf_stem=output_stem, verbose=verbose)
            if xlsx_path:
                xlsx_payload["output_path"] = str(xlsx_path)
        except Exception as exc:
            _log(f"      [xlsx] Warning: {exc}", verbose)
            xlsx_payload["warning"] = str(exc)
        if debug_data is not None:
            _debug_step_done("6_xlsx", xlsx_payload, t_step_6)

        if debug_data is not None and dbg_path is not None:
            debug_data["status"] = "completed"
            debug_data["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            debug_data["elapsed_sec"] = round(time.perf_counter() - workflow_start, 4)
            debug_data["current_step"] = None
            debug_data["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_debug_snapshot(debug_data, dbg_path, verbose)
            _log(f"      Debug snapshot: {dbg_path}", verbose)

        elapsed = time.perf_counter() - workflow_start
        _log(f"[done] Workflow total time: {elapsed:.2f}s · items={len(items)}", verbose)
        return items
    except Exception as exc:
        if debug_data is not None and dbg_path is not None:
            debug_data["status"] = "error"
            debug_data["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            debug_data["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            debug_data["elapsed_sec"] = round(time.perf_counter() - workflow_start, 4)
            debug_data["current_step"] = None
            debug_data["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_debug_snapshot(debug_data, dbg_path, verbose)
        raise


def run_workflow(
    pdf_path: str | Path,
    output_dir: Path | str | None = None,
    *,
    write_json: bool = True,
    verbose: bool = True,
    max_pages: int | None = None,
    recache: bool = False,
    debug: bool = False,
    debug_path: Path | str | None = None,
    output_suffix: str | None = None,
) -> list[dict[str, Any]]:
    """
    Run full PDF recognition workflow and return list of items (one per page pair).
    Writes output to {output_dir}/{pdf_stem}.json by default.
    When verbose=True, prints step messages and shows tqdm progress bars.
    max_pages optionally limits the number of pages processed per PDF.
    recache=True clears any existing cached images/normalized pages for this PDF.
    debug=True writes a {pdf_stem}_debug.json with OCR, LLM, and ROI debug data.
    """
    return _run_workflow(
        pdf_path,
        output_dir=output_dir,
        write_json=write_json,
        verbose=verbose,
        max_pages=max_pages,
        recache=recache,
        debug=debug,
        debug_path=debug_path,
        output_suffix=output_suffix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full PDF ROI recognition workflow.")
    parser.add_argument("pdf_path", help="Path to the PDF file to process.")
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not write JSON output file; only return items.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging and progress bars.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional limit on number of pages to process from the PDF.",
    )
    parser.add_argument(
        "--recache",
        action="store_true",
        help="Delete any existing cached images/normalized pages for this PDF before processing.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write a debug JSON file with OCR, LLM, normalization, and ROI extraction details.",
    )
    parser.add_argument(
        "--debug-path",
        type=str,
        default=None,
        help="Path for debug output file (default: {output_dir}/{pdf_stem}_debug.json).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to output stem for JSON/XLSX/debug filenames.",
    )
    args = parser.parse_args()

    items = run_workflow(
        args.pdf_path,
        write_json=not args.no_json,
        verbose=args.verbose,
        max_pages=args.max_pages,
        recache=args.recache,
        debug=args.debug,
        debug_path=args.debug_path,
        output_suffix=args.output_suffix,
    )
    if args.verbose:
        for i, item in enumerate(items):
            grade_dbg = _form_type_predicted_debug(item.get("form_type"))
            print(
                f"  Item {i + 1}: page_odd={item['page_odd']}, page_even={item.get('page_even')}, "
                f"id={item.get('id', '')!r}, {grade_dbg}"
            )
