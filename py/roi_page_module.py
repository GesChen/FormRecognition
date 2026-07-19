"""
Standalone module: ROI-based page interpretation (text + MCQ).

Pipeline:
  - Given a normalized page image and its form type + side ("a" | "b"),
    select the matching ROI schema from data/roi_schemas using the
    same naming convention as normalization templates:

        <form_type>_<side>.json   e.g.  6pre_a.json, 6pre_b.json

  - Interpret ROIs in that schema as:
      * Text ROIs:
          - ROI name is exactly a number (e.g. "1", "42")
          - There are NO ROIs whose name is that number followed by
            a letter (e.g. "1a", "1b"...). These ROIs are treated as
            standalone text regions.

      * MCQ ROIs:
          - ROI name is a number (e.g. "1") AND there exist other
            ROIs in the same schema whose names are that number +
            one letter (e.g. "1a", "1b"...).
          - The pure number ROI is the "main" MCQ ROI.
          - The numbered+letter ROIs ("1a", "1b", ...) are the
            per-choice sub-ROIs attached to the main ROI for
            future MCQ recognition.

The module exposes:

    from roi_page_module import (
        PageRoiSchema,
        TextRoi,
        McqChoiceRoi,
        McqRoi,
        load_page_schema,
        recognize_text_fields,
        recognize_mcq_fields,
    )

For now, recognize_text_fields / recognize_mcq_fields are stubs that
will be filled in by future prompts. They exist to keep text vs MCQ
handling cleanly separated.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import IMAGE_NORMALIZE, PDF_RECOGNITION, ROI_PAGE_RECOGNITION


def _tqdm_enabled() -> bool:
    """
    Render live tqdm bars only on interactive terminals.
    Prevents carriage-return bar updates from spamming web preview logs.
    """
    env = str(os.environ.get("EVMS_TQDM", "auto")).strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    if str(os.environ.get("TERM", "")).strip().lower() == "dumb":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _suppress_horizontal_lines(crop: Any) -> tuple[Any, dict[str, Any]]:
    """Remove long horizontal rules from a grayscale text ROI crop."""
    cfg = ROI_PAGE_RECOGNITION if isinstance(ROI_PAGE_RECOGNITION, dict) else {}
    enabled = bool(cfg.get("text_roi_horizontal_line_suppression_enabled", False))
    try:
        min_line_len = max(2, int(cfg.get("text_roi_horizontal_line_min_len", 80) or 80))
    except (TypeError, ValueError):
        min_line_len = 80
    try:
        thickness = max(1, int(cfg.get("text_roi_horizontal_line_thickness", 2) or 2))
    except (TypeError, ValueError):
        thickness = 2
    try:
        inpaint_radius = max(1, int(cfg.get("text_roi_horizontal_line_inpaint_radius", 3) or 3))
    except (TypeError, ValueError):
        inpaint_radius = 3

    details = {
        "enabled": enabled,
        "min_line_len": min_line_len,
        "thickness": thickness,
        "inpaint_radius": inpaint_radius,
        "removed_pixels": 0,
    }
    if not enabled:
        return crop, details

    import cv2

    gray = crop if getattr(crop, "ndim", 0) == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1))
    line_mask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    if thickness > 1:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, thickness))
        line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)

    details["removed_pixels"] = int(cv2.countNonZero(line_mask))
    if not details["removed_pixels"]:
        return gray, details
    return cv2.inpaint(gray, line_mask, inpaint_radius, cv2.INPAINT_TELEA), details


@dataclass
class BaseRoi:
    """Common fields shared by all ROI types."""

    name: str
    x: float
    y: float
    w: float
    h: float
    meta: Dict[str, Any]


@dataclass
class TextRoi(BaseRoi):
    """ROI interpreted as a text field."""

    kind: str = field(default="text", init=False)


@dataclass
class McqChoiceRoi(BaseRoi):
    """One choice ROI within a multiple-choice question (e.g. '1a')."""

    letter: str
    kind: str = field(default="mcq_choice", init=False)


@dataclass
class McqRoi(BaseRoi):
    """
    Multiple-choice ROI, with a main number ROI (e.g. '1') and
    per-choice sub-ROIs (e.g. '1a', '1b', ...).
    """

    choices: List[McqChoiceRoi] = field(default_factory=list)
    kind: str = field(default="mcq", init=False)


@dataclass
class PageRoiSchema:
    """
    Interpreted ROI schema for a single page:
      - text_rois: all standalone text ROIs
      - mcq_rois: all MCQ main ROIs with nested choices
    """

    form_type: Optional[str]
    side: str
    schema_path: Path
    text_rois: List[TextRoi]
    mcq_rois: List[McqRoi]
    raw_schema: Dict[str, Any]


def _schema_path_for_form(form_type: Optional[str], side: str) -> Path:
    """
    Resolve per-form schema path from config:
        data/roi_schemas/<form_type>_<side>.json

    If form_type is None or empty, fall back to legacy keys
    from PDF_RECOGNITION config.
    """
    cfg = PDF_RECOGNITION
    schema_dir = Path(
        cfg.get(
            "schema_dir",
            Path(__file__).resolve().parent.parent / "data" / "roi_schemas",
        )
    ).resolve()
    s = (form_type or "").strip()
    if s:
        return schema_dir / f"{s}_{side}.json"
    fallback = (
        cfg.get("schema_key_sidea", "schema_sidea")
        if side == "a"
        else cfg.get("schema_key_sideb", "schema_sideb")
    )
    return schema_dir / f"{fallback}.json"


def _is_pure_number(name: str) -> bool:
    return name.isdigit()


def _split_number_letter(name: str) -> Optional[tuple[str, str]]:
    """
    If name is of the form "<number><letter>" (e.g. "12a"), return (number, letter).
    Otherwise return None.
    """
    if not name:
        return None
    num_part = ""
    letter_part = ""
    for ch in name:
        if ch.isdigit() and not letter_part:
            num_part += ch
        else:
            letter_part += ch
    if not num_part or len(letter_part) != 1 or not letter_part.isalpha():
        return None
    return num_part, letter_part.lower()


def _interpret_rois(schema: Dict[str, Any]) -> tuple[List[TextRoi], List[McqRoi]]:
    """
    Interpret raw schema rois into text and MCQ structures.

    Rules:
      - Text ROI: name is a pure number and there are no "<number><letter>" ROIs.
      - MCQ ROI: there exist "<number><letter>" ROIs (e.g. 1a, 1b); the sub-ROIs
        become McqChoiceRoi entries. If the schema has a main "N" ROI, use its bbox;
        otherwise create an MCQ ROI from the lettered sub-ROIs only (bbox = union of choices).
    """
    rois = schema.get("rois", []) or []

    # Index ROIs by name for quick lookup
    by_name: Dict[str, Dict[str, Any]] = {}
    # Track which numbers have lettered sub-ROIs
    number_has_letters: Dict[str, bool] = {}
    lettered_by_number: Dict[str, List[tuple[str, Dict[str, Any]]]] = {}

    for roi in rois:
        name = str(roi.get("name", "")).strip()
        if not name:
            continue
        by_name[name] = roi
        split = _split_number_letter(name)
        if split is not None:
            num, letter = split
            number_has_letters[num] = True
            lettered_by_number.setdefault(num, []).append((letter, roi))

    text_rois: List[TextRoi] = []
    mcq_rois: List[McqRoi] = []

    def make_choices(num: str) -> List[McqChoiceRoi]:
        choices_raw = lettered_by_number.get(num, [])
        choices_raw = sorted(choices_raw, key=lambda p: p[0])
        choices: List[McqChoiceRoi] = []
        for letter, sub_roi in choices_raw:
            cx = float(sub_roi.get("x", 0))
            cy = float(sub_roi.get("y", 0))
            cw = float(sub_roi.get("w", 0))
            ch = float(sub_roi.get("h", 0))
            cmeta = {
                k: v
                for k, v in sub_roi.items()
                if k not in {"name", "x", "y", "w", "h"}
            }
            choices.append(
                McqChoiceRoi(
                    name=f"{num}{letter}",
                    letter=letter,
                    x=cx,
                    y=cy,
                    w=cw,
                    h=ch,
                    meta=cmeta,
                )
            )
        return choices

    # Pass 1: pure-number ROIs that exist in the schema
    for name, roi in by_name.items():
        if not _is_pure_number(name):
            continue

        num = name
        if number_has_letters.get(num):
            # MCQ: schema has main "N" ROI
            x = float(roi.get("x", 0))
            y = float(roi.get("y", 0))
            w = float(roi.get("w", 0))
            h = float(roi.get("h", 0))
            meta = {k: v for k, v in roi.items() if k not in {"name", "x", "y", "w", "h"}}
            mcq_rois.append(
                McqRoi(
                    name=num,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    meta=meta,
                    choices=make_choices(num),
                )
            )
        else:
            # Standalone text ROI
            x = float(roi.get("x", 0))
            y = float(roi.get("y", 0))
            w = float(roi.get("w", 0))
            h = float(roi.get("h", 0))
            meta = {k: v for k, v in roi.items() if k not in {"name", "x", "y", "w", "h"}}
            text_rois.append(
                TextRoi(
                    name=num,
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    meta=meta,
                )
            )

    # Pass 1b: non-numeric ROI names that aren't MCQ choice sub-ROIs → text ROIs
    handled_names: set[str] = set()
    for name in by_name:
        if _is_pure_number(name):
            handled_names.add(name)
            continue
        if _split_number_letter(name) is not None:
            continue
        handled_names.add(name)
        roi = by_name[name]
        x = float(roi.get("x", 0))
        y = float(roi.get("y", 0))
        w = float(roi.get("w", 0))
        h = float(roi.get("h", 0))
        meta = {k: v for k, v in roi.items() if k not in {"name", "x", "y", "w", "h"}}
        text_rois.append(
            TextRoi(name=name, x=x, y=y, w=w, h=h, meta=meta)
        )

    # Pass 2: numbers that have lettered sub-ROIs but no main "N" ROI in schema
    # (e.g. schema has 1a, 1b, 1c, 1d but no "1") -> still create MCQ ROI from choices
    for num in lettered_by_number:
        if num in by_name:
            continue
        choices = make_choices(num)
        if not choices:
            continue
        x_min = min(c.x for c in choices)
        y_min = min(c.y for c in choices)
        x_max = max(c.x + c.w for c in choices)
        y_max = max(c.y + c.h for c in choices)
        mcq_rois.append(
            McqRoi(
                name=num,
                x=x_min,
                y=y_min,
                w=x_max - x_min,
                h=y_max - y_min,
                meta={},
                choices=choices,
            )
        )

    # Stable order: by numeric name (1, 2, ..., 10, 11, ...)
    def _sort_key(roi: Any) -> tuple:
        n = getattr(roi, "name", roi) if hasattr(roi, "name") else str(roi)
        try:
            return (int(n), "")
        except ValueError:
            return (0, str(n))

    text_rois.sort(key=_sort_key)
    mcq_rois.sort(key=_sort_key)
    return text_rois, mcq_rois


def load_page_schema(form_type: Optional[str], side: str) -> PageRoiSchema:
    """
    Load and interpret the ROI schema for a single page.

    Args:
        form_type: Form type string (e.g. "6pre", "6post", "7pre").
        side:      Page side, "a" for odd/front, "b" for even/back.

    Returns:
        PageRoiSchema with text_rois and mcq_rois lists populated.
    """
    side = (side or "").strip().lower()
    if side not in {"a", "b"}:
        raise ValueError(f"side must be 'a' or 'b', got {side!r}")

    schema_path = _schema_path_for_form(form_type, side)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    raw = json.loads(schema_path.read_text())
    text_rois, mcq_rois = _interpret_rois(raw)
    return PageRoiSchema(
        form_type=(form_type or None),
        side=side,
        schema_path=schema_path,
        text_rois=text_rois,
        mcq_rois=mcq_rois,
        raw_schema=raw,
    )


def _serialize_raw_ocr(raw_result: dict[str, Any] | None) -> dict[str, Any]:
    """JSON-serializable object from structured OCR output."""
    try:
        return json.loads(json.dumps(raw_result or {}, default=str))
    except Exception:
        return {"__repr__": repr(raw_result)}


def _roi_output_regex(meta: Dict[str, Any] | None) -> str | None:
    """
    Optional strict regex for final normalized ROI text.
    Stored in schema metadata as `output_regex`.
    """
    if not bool(ROI_PAGE_RECOGNITION.get("ocr_regex_check_enabled", True)):
        return None
    if not isinstance(meta, dict):
        return None
    value = meta.get("output_regex")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def recognize_text_fields(
    page_image_path: str | Path,
    page_schema: PageRoiSchema,
    *,
    debug_collector: Optional[Dict[str, Any]] = None,
    ocr_confidence_out: Optional[Dict[str, Any]] = None,
    review_text_queue_out: Optional[List[Dict[str, Any]]] = None,
    review_context: Optional[Dict[str, Any]] = None,
    ocr_retry_meta_out: Optional[Dict[str, Dict[str, Any]]] = None,
    use_tqdm: bool = False,
    tqdm_desc: str = "        Text OCR",
    tqdm_leave: bool = False,
    apply_llm_postprocess: bool = True,
) -> Dict[str, Any]:
    """
    Recognize text content for all TextRoi entries on the page.

    Returns a mapping: ROI name -> recognized string, using OCR.
    If debug_collector is provided, it is filled with text_per_roi: { name: { text, raw_ocr } }.
    If ocr_confidence_out is provided (mutable dict), it is filled with ROI name -> stats from
    :func:`ocr_engine.ocr_confidence_stats` (requires ``ocr_raw`` on each crop).

    When *review_text_queue_out* and *review_context* are set and ``PDF_RECOGNITION["human_review"]``
    is enabled, OCR-flagged crops are appended to *review_text_queue_out* with a
    ``review_target_ref`` (see ``ocr_human_review``). *review_context* must include
    ``pdf_stem``, ``pair_index``, ``page_in_pair`` (``odd``/``even``).
    """
    from ocr_engine import build_vlm_roi_prompt, ocr_raw, ocr_confidence_stats
    from ocr_human_review import append_low_confidence_text_roi
    import cv2

    img_path = Path(page_image_path).resolve()
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found for text recognition: {img_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    h, w = img.shape[:2]
    # Scale ROI coordinates when schema was defined for a different image size.
    raw = page_schema.raw_schema or {}
    ref_w = int(raw.get("image_width") or 0)
    ref_h = int(raw.get("image_height") or 0)
    if ref_w > 0 and ref_h > 0 and (ref_w, ref_h) != (w, h):
        scale_x, scale_y = w / ref_w, h / ref_h
    else:
        scale_x = scale_y = 1.0

    results: Dict[str, Any] = {}
    non_header_texts: Dict[str, str] = {}
    non_header_meta: Dict[str, Dict[str, Any]] = {}
    if debug_collector is not None:
        debug_collector["text_per_roi"] = {}
    if ocr_confidence_out is not None:
        ocr_confidence_out.clear()
    if ocr_retry_meta_out is not None:
        ocr_retry_meta_out.clear()

    hr_cfg = PDF_RECOGNITION.get("human_review") or {}
    hr_thresh = float(hr_cfg.get("min_text_ocr_confidence", 0.85))
    hr_use_review_flag = bool(hr_cfg.get("text_review_use_needs_human_review", True))
    queue_enabled = (
        bool(hr_cfg.get("enabled"))
        and review_text_queue_out is not None
        and isinstance(review_context, dict)
        and review_context.get("pdf_stem")
    )
    try:
        text_roi_expand_px = max(0, int(ROI_PAGE_RECOGNITION.get("text_roi_expand_px", 0) or 0))
    except (TypeError, ValueError):
        text_roi_expand_px = 0

    roi_bar = None
    roi_iterable = page_schema.text_rois
    total_text_rois = len(page_schema.text_rois)
    if use_tqdm and total_text_rois > 0:
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            def tqdm(iterable, desc=None, **kwargs):
                return iterable
        roi_bar = tqdm(
            page_schema.text_rois,
            total=total_text_rois,
            desc=tqdm_desc,
            unit="roi",
            leave=tqdm_leave,
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            position=1,
        )
        roi_iterable = roi_bar

    # ROIs are in normalized image coordinate space (same as roi_editor schemas).
    for roi_index, roi in enumerate(roi_iterable, start=1):
        if roi_bar is not None:
            roi_bar.set_postfix_str(
                f"{roi_index}/{total_text_rois} roi={roi.name}",
                refresh=False,
            )
        x1_raw = int(round(roi.x * scale_x))
        y1_raw = int(round(roi.y * scale_y))
        x2_raw = int(round((roi.x + roi.w) * scale_x))
        y2_raw = int(round((roi.y + roi.h) * scale_y))
        x1 = max(x1_raw - text_roi_expand_px, 0)
        y1 = max(y1_raw - text_roi_expand_px, 0)
        x2 = min(x2_raw + text_roi_expand_px, w)
        y2 = min(y2_raw + text_roi_expand_px, h)
        if x2 <= x1 or y2 <= y1:
            results[roi.name] = ""
            if debug_collector is not None:
                debug_collector["text_per_roi"][roi.name] = {"text": "", "raw_ocr": None}
            if ocr_confidence_out is not None:
                ocr_confidence_out[roi.name] = ocr_confidence_stats({})
            continue

        crop = img[y1:y2, x1:x2]
        crop, horizontal_line_cleanup = _suppress_horizontal_lines(crop)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), crop)
            roi_meta = getattr(roi, "meta", None)
            vlm_query = ""
            vlm_prompt = None
            if isinstance(roi_meta, dict):
                vlm_query = str(roi_meta.get("vlm_query", "") or "").strip()
                vlm_prompt = build_vlm_roi_prompt(vlm_query)
            is_header_prompt_roi = str(roi.name or "").strip().lower() in {"id", "form_type"}
            strict_regex = _roi_output_regex(roi_meta)
            if ocr_retry_meta_out is not None:
                ocr_retry_meta_out[str(roi.name)] = {
                    "image_path": str(img_path),
                    "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "text_roi_expand_px": int(text_roi_expand_px),
                    "horizontal_line_cleanup": dict(horizontal_line_cleanup),
                    "output_regex": strict_regex,
                    "vlm_query": vlm_query,
                    "vlm_prompt_override": vlm_prompt,
                }
            need_raw = (
                debug_collector is not None
                or ocr_confidence_out is not None
                or queue_enabled
            )
            if need_raw:
                raw_out = (
                    ocr_raw(tmp_path, prompt_override=vlm_prompt)
                    if vlm_prompt
                    else ocr_raw(tmp_path)
                )
                raw_result = raw_out if isinstance(raw_out, dict) else {}
                if vlm_prompt:
                    raw_result = dict(raw_result)
                    raw_result["vlm_query"] = vlm_query
                    raw_result["vlm_prompt_override"] = vlm_prompt
                text = str(raw_result.get("detected_text", "") or "")
                if debug_collector is not None:
                    debug_collector["text_per_roi"][roi.name] = {
                        "text": text,
                        "raw_ocr": _serialize_raw_ocr(raw_result),
                        "ocr_path": "text_vlm_query" if vlm_prompt else "text_default",
                        "horizontal_line_cleanup": dict(horizontal_line_cleanup),
                    }
                stats = ocr_confidence_stats(raw_result)
                if ocr_confidence_out is not None:
                    ocr_confidence_out[roi.name] = stats
            else:
                raw_out = (
                    ocr_raw(tmp_path, prompt_override=vlm_prompt)
                    if vlm_prompt
                    else ocr_raw(tmp_path)
                )
                raw_result = raw_out if isinstance(raw_out, dict) else {}
                if vlm_prompt:
                    raw_result = dict(raw_result)
                    raw_result["vlm_query"] = vlm_query
                    raw_result["vlm_prompt_override"] = vlm_prompt
                text = str(raw_result.get("detected_text", "") or "")
                stats = ocr_confidence_stats({})
            cleaned = " ".join(str(text).split())
            results[roi.name] = cleaned
            if not is_header_prompt_roi:
                non_header_texts[roi.name] = cleaned
                if isinstance(roi_meta, dict):
                    non_header_meta[roi.name] = dict(roi_meta)
                if isinstance(raw_result.get("paddle_confidence"), dict):
                    meta_for_name = non_header_meta.setdefault(roi.name, {})
                    meta_for_name["ocr_paddle_confidence"] = json.loads(
                        json.dumps(raw_result.get("paddle_confidence"), default=str)
                    )
            if queue_enabled:
                append_low_confidence_text_roi(
                    review_text_queue_out,
                    pdf_stem=str(review_context["pdf_stem"]),
                    pair_index=int(review_context["pair_index"]),
                    page_in_pair=str(review_context["page_in_pair"]),
                    roi_name=roi.name,
                    stats=stats if need_raw else ocr_confidence_stats({}),
                    text_current=cleaned,
                    threshold=hr_thresh,
                    use_needs_review_flag=hr_use_review_flag,
                )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Non-header text ROIs can optionally go through LLM postprocessing.
    if non_header_texts and apply_llm_postprocess:
        llm_debug: Dict[str, Any] | None = {} if debug_collector is not None else None
        try:
            from text_roi_llm import postprocess_text_rois

            llm_processed = postprocess_text_rois(
                non_header_texts,
                roi_meta_by_name=non_header_meta,
                debug_out=llm_debug,
            )
        except Exception as exc:
            llm_processed = dict(non_header_texts)
            if llm_debug is not None:
                llm_debug["error"] = str(exc)
        for name, value in llm_processed.items():
            if name in results:
                results[name] = "" if value is None else str(value)
            if debug_collector is not None and name in debug_collector.get("text_per_roi", {}):
                entry = debug_collector["text_per_roi"][name]
                if isinstance(entry, dict):
                    entry["text_before_llm"] = non_header_texts.get(name, "")
                    entry["text"] = "" if value is None else str(value)
                    entry["text_postprocessed_by_llm"] = True
    elif non_header_texts:
        # Keep default OCR text for now; caller may apply deferred LLM postprocess later.
        for name, value in non_header_texts.items():
            if name in results:
                results[name] = "" if value is None else str(value)

    return results


def recognize_mcq_fields(
    page_image_path: str | Path,
    page_schema: PageRoiSchema,
    *,
    debug_collector: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Recognize multiple-choice selections for all McqRoi entries on the page.

    Soft template-subtracted darkness scorer:
      1. Convert template and target ROI crops to 0–1 darkness (0=white, 1=black).
      2. Optionally blur both to smooth scan noise.
      3. Compute extra_dark = clip(target_dark - template_dark - eps, 0, 1).
      4. Down-weight pixels already dark in the template: w = exp(-k * template_dark).
      5. Score = sum(w * extra_dark^power).
      6. Answer = letter with highest score; "" if all scores ≤ 0.

    If debug_collector is provided, it is filled with mcq_template_path and
    mcq_per_roi: { name: { answer, scores: {letter: float} } }.
    """
    import cv2
    import numpy as np
    import re

    def _resolve_source_template_path(form_type: Optional[str], side: str) -> Optional[Path]:
        key = f"{form_type}_{side}" if form_type else None
        if key is None:
            key = IMAGE_NORMALIZE.get("template_registration_default_key")
        if key is None:
            return None
        templates = IMAGE_NORMALIZE.get("template_registration_templates") or {}
        if isinstance(templates, dict):
            path = templates.get(key)
            if path is not None:
                return Path(path).resolve()
        templates_dir = IMAGE_NORMALIZE.get("template_registration_templates_dir")
        if templates_dir:
            candidate = Path(templates_dir).resolve() / f"{key}.png"
            if candidate.exists():
                return candidate
        return None

    def _expanded_bounds(x: float, y: float, w: float, h: float, pad: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        x1 = max(int(round(x)) - pad, 0)
        y1 = max(int(round(y)) - pad, 0)
        x2 = min(int(round(x + w)) + pad, img_w)
        y2 = min(int(round(y + h)) + pad, img_h)
        return x1, y1, x2, y2

    cfg = ROI_PAGE_RECOGNITION
    pad = int(cfg.get("mcq_subroi_expand_px", 0) or 0)
    eps = float(cfg.get("mcq_eps", 0.05) or 0.05)
    suppress_k = float(cfg.get("mcq_print_suppression_k", 5.0) or 5.0)
    power = float(cfg.get("mcq_power", 1.0) or 1.0)
    blur_sigma = float(cfg.get("mcq_blur_sigma", 0) or 0)
    min_score_to_accept = float(cfg.get("mcq_min_score_to_accept", 0.0) or 0.0)
    return_debug = bool(cfg.get("mcq_return_raw_darkness_debug", False)) or (debug_collector is not None)
    write_debug_images = bool(cfg.get("mcq_write_debug_images", False))
    debug_images_root_cfg = cfg.get("mcq_debug_images_dir")

    if debug_collector is not None:
        debug_collector["mcq_per_roi"] = {}

    target_path = Path(page_image_path).resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Image not found for MCQ recognition: {target_path}")
    target_gray = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if target_gray is None:
        raise FileNotFoundError(f"Cannot read image: {target_path}")

    source_path = _resolve_source_template_path(page_schema.form_type, page_schema.side)
    if debug_collector is not None:
        debug_collector["mcq_template_path"] = str(source_path) if source_path else None

    debug_image_paths: list[str] = []
    debug_page_dir: Path | None = None
    if write_debug_images and debug_images_root_cfg:
        root_dir = Path(debug_images_root_cfg).resolve()
        safe_form = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(page_schema.form_type or "unknown")).strip("_") or "unknown"
        safe_side = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(page_schema.side or "x")).strip("_") or "x"
        safe_page = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_path.stem).strip("_") or "page"
        debug_page_dir = root_dir / f"{safe_form}_{safe_side}" / safe_page
        debug_page_dir.mkdir(parents=True, exist_ok=True)
        if debug_collector is not None:
            debug_collector["mcq_debug_image_dir"] = str(debug_page_dir)

    def _write_question_debug_image(
        *,
        question_name: str,
        answer: str,
        choice_debug: list[tuple[str, float, np.ndarray]],
        choice_crops: list[tuple[str, np.ndarray]],
    ) -> None:
        if not write_debug_images or debug_page_dir is None or not choice_debug:
            return
        tile_h = max(int(img.shape[0]) for _, _, img in choice_debug)
        tile_w = max(int(img.shape[1]) for _, _, img in choice_debug)
        n = len(choice_debug)
        pad_px = 8
        header_h = 36
        label_h = 20
        row_label_h = 16
        crop_row_h = tile_h + row_label_h if choice_crops else 0
        diff_row_h = tile_h + row_label_h
        canvas_h = header_h + label_h + crop_row_h + diff_row_h + 2 * pad_px
        canvas_w = pad_px + n * (tile_w + pad_px)
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        title = f"Q{question_name} answer={answer or '-'}"
        cv2.putText(canvas, title, (pad_px, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 1, cv2.LINE_AA)

        # Row 1: raw target crops
        if choice_crops:
            row1_label_y = header_h + label_h + 11
            cv2.putText(canvas, "crop", (2, row1_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 140, 1, cv2.LINE_AA)
            row1_y0 = header_h + label_h + row_label_h
            for i, (letter, crop) in enumerate(choice_crops):
                x0 = pad_px + i * (tile_w + pad_px)
                gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if gray.shape[:2] != (tile_h, tile_w):
                    tile = cv2.resize(gray, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                else:
                    tile = gray
                canvas[row1_y0 : row1_y0 + tile_h, x0 : x0 + tile_w] = tile
                cv2.rectangle(canvas, (x0, row1_y0), (x0 + tile_w - 1, row1_y0 + tile_h - 1), 180, 1)

        # Row 2: weighted diff with scores
        row2_label_y = header_h + label_h + crop_row_h + 11
        cv2.putText(canvas, "diff", (2, row2_label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 140, 1, cv2.LINE_AA)
        row2_y0 = header_h + label_h + crop_row_h + row_label_h
        for i, (letter, score, img) in enumerate(choice_debug):
            x0 = pad_px + i * (tile_w + pad_px)
            if img.shape[:2] != (tile_h, tile_w):
                tile = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
            else:
                tile = img
            canvas[row2_y0 : row2_y0 + tile_h, x0 : x0 + tile_w] = tile
            marker = "*" if letter == answer else " "
            lbl = f"{marker}{letter}: {score:.1f}"
            cv2.putText(canvas, lbl, (x0, header_h + label_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 255, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x0, row2_y0), (x0 + tile_w - 1, row2_y0 + tile_h - 1), 180, 1)

        safe_q = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(question_name)).strip("_") or "q"
        out_path = debug_page_dir / f"q_{safe_q}_diff.png"
        cv2.imwrite(str(out_path), canvas)
        debug_image_paths.append(str(out_path))

    def _empty_results():
        if debug_collector is not None:
            for mcq in page_schema.mcq_rois:
                debug_collector["mcq_per_roi"][mcq.name] = {"answer": "", "scores": {}}
        return {mcq.name: ({"answer": "", "scores": {}} if return_debug else "") for mcq in page_schema.mcq_rois}

    if source_path is None or not source_path.exists():
        return _empty_results()
    source_gray = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    if source_gray is None:
        return _empty_results()

    tgt_h, tgt_w = target_gray.shape[:2]
    if source_gray.shape[:2] != target_gray.shape[:2]:
        source_gray = cv2.resize(source_gray, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

    # Convert to 0–1 darkness (0 = white, 1 = black).
    tgt_dark = 1.0 - target_gray.astype(np.float32) / 255.0
    src_dark = 1.0 - source_gray.astype(np.float32) / 255.0

    # Optional blur to smooth scan noise.
    if blur_sigma > 0:
        ksize = int(round(blur_sigma * 6)) | 1
        tgt_dark = cv2.GaussianBlur(tgt_dark, (ksize, ksize), blur_sigma)
        src_dark = cv2.GaussianBlur(src_dark, (ksize, ksize), blur_sigma)

    # Scale ROI coordinates when schema was defined for a different image size.
    raw = page_schema.raw_schema or {}
    ref_w = int(raw.get("image_width") or 0)
    ref_h = int(raw.get("image_height") or 0)
    if ref_w > 0 and ref_h > 0 and (ref_w, ref_h) != (tgt_w, tgt_h):
        scale_x = tgt_w / ref_w
        scale_y = tgt_h / ref_h
    else:
        scale_x = scale_y = 1.0

    results: Dict[str, Any] = {}
    for mcq in page_schema.mcq_rois:
        choice_scores: Dict[str, float] = {}
        choice_debug_images: list[tuple[str, float, np.ndarray]] = []
        choice_crop_images: list[tuple[str, np.ndarray]] = []
        for choice in mcq.choices:
            cx = choice.x * scale_x
            cy = choice.y * scale_y
            cw = choice.w * scale_x
            ch = choice.h * scale_y
            x1, y1, x2, y2 = _expanded_bounds(cx, cy, cw, ch, pad, tgt_w, tgt_h)
            if x2 <= x1 or y2 <= y1:
                choice_scores[choice.letter] = 0.0
                choice_debug_images.append((choice.letter, 0.0, np.zeros((8, 8), dtype=np.uint8)))
                choice_crop_images.append((choice.letter, np.zeros((8, 8), dtype=np.uint8)))
                continue

            src_crop = src_dark[y1:y2, x1:x2]
            tgt_crop = tgt_dark[y1:y2, x1:x2]

            # Keep raw grayscale crop for debug output.
            choice_crop_images.append((choice.letter, target_gray[y1:y2, x1:x2].copy()))

            extra_dark = np.clip(tgt_crop - src_crop - eps, 0.0, 1.0)
            # Down-weight pixels already dark in the template (printed structure).
            weight = np.exp(-suppress_k * src_crop)
            weighted = weight * (extra_dark ** power if power != 1.0 else extra_dark)
            score = float(weighted.sum())
            choice_scores[choice.letter] = score
            # B/W difference visualization: weighted new-darkness contribution map.
            diff_img = np.clip(weighted * 255.0, 0, 255).astype(np.uint8)
            choice_debug_images.append((choice.letter, score, diff_img))

        best_letter = ""
        best_score = 0.0
        for letter in sorted(choice_scores):
            s = float(choice_scores[letter])
            if s > best_score:
                best_score = s
                best_letter = letter

        answer = best_letter if best_score > min_score_to_accept else ""
        if debug_collector is not None:
            debug_collector["mcq_per_roi"][mcq.name] = {
                "answer": answer,
                "scores": {k: round(v, 4) for k, v in choice_scores.items()},
            }
        _write_question_debug_image(
            question_name=mcq.name,
            answer=answer,
            choice_debug=choice_debug_images,
            choice_crops=choice_crop_images,
        )
        if return_debug:
            results[mcq.name] = {"answer": answer, "scores": choice_scores}
        else:
            results[mcq.name] = answer

    if debug_collector is not None and debug_image_paths:
        debug_collector["mcq_debug_images"] = debug_image_paths

    return results


def analyze_page(
    page_image_path: str | Path,
    *,
    text_image_path: str | Path | None = None,
    mcq_image_path: str | Path | None = None,
    form_type: Optional[str],
    side: str,
    debug_collector: Optional[Dict[str, Any]] = None,
    pair_index: int = 0,
    page_in_pair: str = "odd",
    pdf_stem: str = "",
    review_text_queue_out: Optional[List[Dict[str, Any]]] = None,
    use_tqdm: bool = False,
    tqdm_desc_prefix: str = "",
    apply_text_llm_postprocess: bool = True,
) -> List[Dict[str, Any]]:
    """
    Central entry point for the ROI page module.

    This is designed to plug into the existing pipeline in pdf_recognize
    immediately after the normalization step:

        result = analyze_page(
            normalized_page_path,
            form_type=detected_form_type,
            side="a" or "b",
        )

    It:
      1) Loads and interprets the ROI schema for (form_type, side).
      2) Runs text recognition for all TextRoi entries (stub for now).
      3) Runs MCQ recognition for all McqRoi entries (stub for now).
      4) Consolidates all outputs into a single list of objects, each with:
           - name: ROI name / logical field name
           - kind: "text" or "mcq"
           - text: string representation of the content

    If debug_collector is provided, it is filled with schema_path, form_type, side,
    and the per-ROI debug from text and MCQ recognition.
    """
    schema = load_page_schema(form_type, side)
    text_src = Path(text_image_path).resolve() if text_image_path is not None else Path(page_image_path).resolve()
    mcq_src = Path(mcq_image_path).resolve() if mcq_image_path is not None else Path(page_image_path).resolve()
    if debug_collector is not None:
        debug_collector["page_image_path"] = str(Path(page_image_path).resolve())
        debug_collector["text_image_path"] = str(text_src)
        debug_collector["mcq_image_path"] = str(mcq_src)
        debug_collector["form_type"] = form_type
        debug_collector["side"] = side
        debug_collector["schema_path"] = str(schema.schema_path)
    from ocr_human_review import make_review_target_ref

    hr_cfg = PDF_RECOGNITION.get("human_review") or {}
    # Keep OCR confidence in pipeline rows even when human review is disabled;
    # later stages need the value as durable data, not only as review metadata.
    collect_ocr = True
    ocr_conf: Dict[str, Any] = {}
    ocr_retry_meta: Dict[str, Dict[str, Any]] = {}
    review_ctx = None
    if pdf_stem and (collect_ocr or review_text_queue_out is not None):
        review_ctx = {
            "pdf_stem": pdf_stem,
            "pair_index": pair_index,
            "page_in_pair": page_in_pair,
        }
    text_tqdm_desc = "        Text OCR"
    if tqdm_desc_prefix:
        text_tqdm_desc = f"        Text OCR {tqdm_desc_prefix}"
    text_data = recognize_text_fields(
        text_src,
        schema,
        debug_collector=debug_collector,
        ocr_confidence_out=ocr_conf if collect_ocr else None,
        ocr_retry_meta_out=ocr_retry_meta,
        review_text_queue_out=review_text_queue_out,
        review_context=review_ctx,
        use_tqdm=use_tqdm,
        tqdm_desc=text_tqdm_desc,
        tqdm_leave=False,
        apply_llm_postprocess=apply_text_llm_postprocess,
    )
    text_meta_by_name: Dict[str, Dict[str, Any]] = {
        str(r.name): (dict(r.meta) if isinstance(getattr(r, "meta", None), dict) else {})
        for r in (schema.text_rois or [])
    }
    mcq_data = recognize_mcq_fields(mcq_src, schema, debug_collector=debug_collector)
    items = []
    for name, value in (text_data or {}).items():
        row = {
            "name": str(name),
            "kind": "text",
            "text": "" if value is None else str(value),
            "_raw_ocr_text": "" if value is None else str(value),
            "ocr_confidence_score": None,
            "ocr_confidence_label": None,
            "ocr_confidence_source": None,
            "ocr_text_source": None,
        }
        if collect_ocr and name in ocr_conf:
            st = ocr_conf[name]
            mn = st.get("confidence_score")
            if mn is None:
                mn = st.get("min_rec_score")
            mean = st.get("mean_rec_score")
            label = st.get("confidence_label")
            review_flag = st.get("needs_human_review")
            model = st.get("selected_model")
            stage_index = st.get("selected_stage_index")
            confidence_source = st.get("confidence_source")
            text_source = st.get("text_source")
            workflow = st.get("workflow")
            paddle_confidence = st.get("paddle_confidence")
            if mn is not None:
                try:
                    score = round(float(mn), 6)
                except (TypeError, ValueError):
                    score = None
                if score is not None:
                    row["ocr_min_score"] = score
                    row["ocr_confidence_score"] = score
            if mean is not None:
                try:
                    row["ocr_mean_score"] = round(float(mean), 6)
                except (TypeError, ValueError):
                    pass
            if label is not None:
                row["ocr_confidence_label"] = str(label).lower()
            if confidence_source is not None:
                row["ocr_confidence_source"] = str(confidence_source)
            if text_source is not None:
                row["ocr_text_source"] = str(text_source)
            if workflow is not None:
                row["ocr_workflow"] = str(workflow)
            if review_flag is not None:
                row["ocr_needs_human_review"] = bool(review_flag)
            if model is not None:
                row["ocr_selected_model"] = str(model)
            if stage_index is not None:
                row["ocr_selected_stage_index"] = stage_index
            if isinstance(paddle_confidence, dict):
                row["ocr_paddle_confidence"] = json.loads(json.dumps(paddle_confidence, default=str))
        if pdf_stem:
            row["review_target_ref"] = make_review_target_ref(
                pdf_stem, pair_index, page_in_pair, str(name), "text"
            )
        # Keep ROI LLM metadata on rows so deferred end-of-pipeline postprocess can run once.
        meta = text_meta_by_name.get(str(name), {})
        if meta:
            row["_llm_field_data_type"] = str(
                meta.get("llm_field_data_type", "") or ""
            ).strip() or None
            row["_llm_validation_rules"] = str(
                meta.get("llm_validation_rules", "") or ""
            ).strip() or None
            row["_llm_prompt_instruction"] = str(
                meta.get("llm_prompt_instruction", "") or ""
            ).strip() or None
            row["_llm_prompt_override"] = str(
                meta.get("llm_prompt_override", "") or ""
            ).strip() or None
            postprocess_passes = meta.get("postprocess_passes")
            if postprocess_passes is None:
                postprocess_passes = meta.get("llm_passes")
            row["_postprocess_passes"] = (
                json.loads(json.dumps(postprocess_passes, default=str))
                if isinstance(postprocess_passes, list)
                else None
            )
            row["_output_regex"] = str(meta.get("output_regex", "") or "").strip() or None
        if isinstance(row.get("ocr_paddle_confidence"), dict):
            row["_ocr_paddle_confidence"] = json.loads(
                json.dumps(row.get("ocr_paddle_confidence"), default=str)
            )
        if row.get("ocr_workflow") is not None:
            row["_ocr_workflow"] = str(row.get("ocr_workflow"))
        retry_meta = ocr_retry_meta.get(str(name), {})
        if retry_meta:
            row["_ocr_retry_image_path"] = str(retry_meta.get("image_path", "") or "").strip() or None
            row["_ocr_retry_bbox_xyxy"] = retry_meta.get("bbox_xyxy")
            if row.get("_output_regex") in {None, ""}:
                row["_output_regex"] = str(
                    retry_meta.get("output_regex", "") or ""
                ).strip() or None
        items.append(row)
    for name, value in (mcq_data or {}).items():
        if isinstance(value, dict):
            answer_text = "" if value.get("answer") is None else str(value.get("answer"))
            mrow: Dict[str, Any] = {
                "name": str(name),
                "kind": "mcq",
                "text": answer_text,
            }
            if pdf_stem:
                mrow["review_target_ref"] = make_review_target_ref(
                    pdf_stem, pair_index, page_in_pair, str(name), "mcq"
                )
            items.append(mrow)
            continue
        mrow = {
            "name": str(name),
            "kind": "mcq",
            "text": "" if value is None else str(value),
        }
        if pdf_stem:
            mrow["review_target_ref"] = make_review_target_ref(
                pdf_stem, pair_index, page_in_pair, str(name), "mcq"
            )
        items.append(mrow)
    return items


def analyze_pages_batch(
    page_inputs: List[Dict[str, Any]],
    *,
    debug_out: Optional[Dict[str, Any]] = None,
    use_tqdm: bool = False,
    tqdm_desc: str = "      ROI analyze",
    apply_text_llm_postprocess: bool = True,
) -> List[List[Dict[str, Any]]]:
    """
    Batch wrapper around analyze_page.

    Args:
        page_inputs: list of dicts with keys:
            - "page_image_path" (required)
            - "form_type" (optional)
            - "side" ("a"|"b", required)
            - "pair_index" (optional, default 0) — index into merged ``items[]`` for this PDF
            - "page_in_pair" (optional, default ``\"odd\"``) — ``\"odd\"`` or ``\"even\"``
            - "pdf_stem" (optional) — for ``review_target_ref`` and OCR review queue
            - "review_text_queue_out" (optional) — shared list; text ROIs needing review append here
        debug_out: if provided, filled with roi_per_page: [ {...}, ... ] (one entry per page).

    Returns:
        List aligned to page_inputs, where each element is the same list
        structure returned by analyze_page for that page.
    """
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        def tqdm(iterable, desc=None, **kwargs):
            return iterable

    out: List[List[Dict[str, Any]]] = []
    if debug_out is not None:
        debug_out["roi_per_page"] = []
    total_pages = len(page_inputs)
    indexed_pages = list(enumerate(page_inputs, start=1))
    iterator = indexed_pages
    page_bar = None
    if use_tqdm:
        page_bar = tqdm(
            indexed_pages,
            total=total_pages,
            desc=tqdm_desc,
            unit="page",
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            leave=False,
            position=0,
        )
        iterator = page_bar
    for page_index, page in iterator:
        pair_display = int(page.get("pair_index", 0) or 0) + 1
        page_side = str(page.get("page_in_pair", "odd") or "odd")
        if page_bar is not None:
            page_bar.set_postfix_str(
                f"{page_index}/{total_pages} pair={pair_display} side={page_side}",
                refresh=False,
            )
        page_debug: Optional[Dict[str, Any]] = {} if debug_out is not None else None
        result = analyze_page(
            page.get("page_image_path"),
            text_image_path=page.get("text_image_path"),
            mcq_image_path=page.get("mcq_image_path"),
            form_type=page.get("form_type"),
            side=page.get("side"),
            debug_collector=page_debug,
            pair_index=int(page.get("pair_index", 0) or 0),
            page_in_pair=str(page.get("page_in_pair", "odd") or "odd"),
            pdf_stem=str(page.get("pdf_stem", "") or ""),
            review_text_queue_out=page.get("review_text_queue_out"),
            use_tqdm=use_tqdm,
            tqdm_desc_prefix=f"({page_index}/{total_pages} {page_side})",
            apply_text_llm_postprocess=apply_text_llm_postprocess,
        )
        out.append(result)
        if debug_out is not None and page_debug is not None:
            debug_out["roi_per_page"].append(page_debug)
    return out
