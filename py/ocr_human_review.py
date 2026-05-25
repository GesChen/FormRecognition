"""
Human-in-the-loop review: detection of fields that may need manual correction (ID, text ROIs, MCQ).

Text OCR rows needing review are enqueued in ``roi_page_module.recognize_text_fields`` using
``review_target_ref`` identifiers. :func:`build_human_review_block` merges that queue into
``human_review.pending`` on the recognition JSON.

Use :func:`apply_manual_corrections` to merge ``corrected_*`` values from ``pending[]`` back into
``items[]`` (then re-run XLSX filling as needed).

See ``docs/ocr_human_review_schema.md``.
"""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any, Mapping

from config import PDF_RECOGNITION

# Stable pointer: pdf_stem must not contain "|".
REVIEW_REF_PREFIX = "evms_review_v1"

# Canonical student id: 7 digits + A or B (same as pipeline expectations).
_STUDENT_ID_RE = re.compile(r"^\d{7}[AB]$")


def is_valid_student_id(value: str | None) -> bool:
    """Return True if *value* matches the normalized 7-digit + A/B id pattern."""
    if value is None:
        return False
    s = str(value).strip().upper()
    return bool(_STUDENT_ID_RE.match(s))


def make_review_target_ref(
    pdf_stem: str,
    pair_index: int,
    page_in_pair: str,
    roi_name: str,
    kind: str,
) -> str:
    """
    Canonical string identifying where this field lands in recognition output:

    ``items[pair_index].data[*]`` for ROI rows (match ``name``, ``kind``, ``page_in_pair``),
    or ``items[pair_index].id`` for ``kind == "id"`` (``roi_name`` is ``__id__``).

    Format: ``evms_review_v1|{pdf_stem}|{pair_index}|{page_in_pair}|{roi_name}|{kind}``
    """
    return (
        f"{REVIEW_REF_PREFIX}|{pdf_stem}|{pair_index}|{page_in_pair}|{roi_name}|{kind}"
    )


def parse_review_target_ref(ref: str) -> dict[str, Any] | None:
    """Parse :func:`make_review_target_ref` string, or return ``None`` if invalid."""
    parts = str(ref).split("|")
    if len(parts) != 6 or parts[0] != REVIEW_REF_PREFIX:
        return None
    try:
        pair_index = int(parts[2])
    except ValueError:
        return None
    return {
        "pdf_stem": parts[1],
        "pair_index": pair_index,
        "page_in_pair": parts[3],
        "roi_name": parts[4],
        "kind": parts[5],
    }


def _coerce_optional_bool(value: Any) -> bool | None:
    """Parse bool-ish values; return None when unknown."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _coerce_float_or_none(value: Any) -> float | None:
    """Best-effort float parse with None fallback."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def append_low_confidence_text_roi(
    queue: list[dict[str, Any]],
    *,
    pdf_stem: str,
    pair_index: int,
    page_in_pair: str,
    roi_name: str,
    stats: Mapping[str, Any],
    text_current: str,
    threshold: float,
    use_needs_review_flag: bool = True,
) -> bool:
    """
    Append one text ROI queue entry when OCR indicates human review is needed.

    Preferred signal is ``stats["needs_human_review"]`` from OCR engine output.
    If missing/unparseable, falls back to ``stats["min_rec_score"] < threshold``
    for backward compatibility.

    Call site: ``roi_page_module.recognize_text_fields`` (only OCR module for text ROIs).

    Returns True if an entry was appended.
    """
    score = _coerce_float_or_none(stats.get("min_rec_score"))
    mean_score = _coerce_float_or_none(stats.get("mean_rec_score"))
    needs_flag = _coerce_optional_bool(stats.get("needs_human_review"))

    should_enqueue = False
    if use_needs_review_flag and needs_flag is not None:
        should_enqueue = bool(needs_flag)
    elif score is not None:
        should_enqueue = score < float(threshold)
    if not should_enqueue:
        return False

    ref = make_review_target_ref(
        pdf_stem, pair_index, page_in_pair, str(roi_name), "text"
    )
    queue.append(
        {
            "review_target_ref": ref,
            "item_index": pair_index,
            "page_in_pair": page_in_pair,
            "roi_name": str(roi_name),
            "kind": "text",
            # Legacy confidence aliases kept for existing tools/scripts.
            "ocr_min_score": round(score, 6) if score is not None else None,
            "ocr_mean_score": round(mean_score, 6) if mean_score is not None else None,
            # Canonical OCR confidence fields.
            "ocr_confidence_label": str(stats.get("confidence_label") or "").lower() or None,
            "ocr_confidence_score": round(score, 6) if score is not None else None,
            "ocr_needs_human_review": (
                bool(needs_flag) if needs_flag is not None else bool(should_enqueue)
            ),
            "ocr_selected_model": stats.get("selected_model"),
            "ocr_selected_stage_index": stats.get("selected_stage_index"),
            "text_current": str(text_current or ""),
        }
    )
    return True


def make_entry_id(
    pdf_stem: str,
    item_index: int,
    page_in_pair: str,
    roi_name: str,
    kind: str,
) -> str:
    """
    Stable id for a pending review row (hex). Same inputs => same id across re-runs.
    """
    key = f"{pdf_stem}\x00{item_index}\x00{page_in_pair}\x00{roi_name}\x00{kind}".encode(
        "utf-8"
    )
    return hashlib.sha256(key).hexdigest()[:24]


def _pending_dict_from_text_queue_entry(
    pdf_stem: str,
    e: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one ``human_review.pending`` row from a text OCR queue entry."""
    item_index = int(e["item_index"])
    page_in_pair = str(e.get("page_in_pair", "") or "unknown")
    roi_name = str(e.get("roi_name", ""))
    ref = str(e.get("review_target_ref") or make_review_target_ref(
        pdf_stem, item_index, page_in_pair, roi_name, "text"
    ))
    return {
        "review_target_ref": ref,
        "entry_id": make_entry_id(pdf_stem, item_index, page_in_pair, roi_name, "text"),
        "item_index": item_index,
        "page_in_pair": page_in_pair,
        "roi_name": roi_name,
        "kind": "text",
        "ocr_min_score": e.get("ocr_min_score"),
        "ocr_mean_score": e.get("ocr_mean_score"),
        "ocr_confidence_label": e.get("ocr_confidence_label"),
        "ocr_confidence_score": e.get("ocr_confidence_score"),
        "ocr_needs_human_review": e.get("ocr_needs_human_review"),
        "ocr_selected_model": e.get("ocr_selected_model"),
        "ocr_selected_stage_index": e.get("ocr_selected_stage_index"),
        "text_current": str(e.get("text_current", "")),
        "corrected_text": None,
        "status": "pending",
        "review_status": "pending",
    }


def build_human_review_block(
    items: list[dict[str, Any]],
    pdf_stem: str,
    cfg: Mapping[str, Any] | None = None,
    *,
    text_ocr_queue: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """
    Build the ``human_review`` object for the recognition JSON, or ``None`` if disabled.

    **Text** review rows: if *text_ocr_queue* is not ``None`` (filled by
    ``recognize_text_fields`` / pipeline), use it exclusively. If ``None``, fall back
    to scanning ``items`` (legacy callers).

    Primary decision signal is OCR's ``needs_human_review`` flag (API stage output).
    Confidence-threshold checks remain as a fallback when the flag is absent.

    **ID** / **MCQ**: unchanged; include ``review_target_ref`` on each pending row.
    """
    cfg = cfg or PDF_RECOGNITION
    hr = cfg.get("human_review") or {}
    if not bool(hr.get("enabled")):
        return None

    thresh = float(hr.get("min_text_ocr_confidence", 0.85))
    thresh_id = float(hr.get("min_id_ocr_confidence", 0.85))
    text_use_review_flag = bool(hr.get("text_review_use_needs_human_review", True))
    id_use_review_flag = bool(hr.get("id_review_use_needs_human_review", True))
    pending_id = bool(hr.get("pending_id_review", True))
    pending_all_ids = bool(hr.get("pending_all_ids", False))
    pending_mcq_empty = bool(hr.get("pending_mcq_empty_review", True))

    pending: list[dict[str, Any]] = []

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        # --- Student ID (item-level) ---
        raw_id = str(item.get("id") or "").strip()
        id_min_f = _coerce_float_or_none(item.get("id_ocr_min_score"))
        id_mean_f = _coerce_float_or_none(item.get("id_ocr_mean_score"))
        id_label = str(item.get("id_ocr_confidence_label") or "").lower() or None
        id_review_flag = _coerce_optional_bool(item.get("id_ocr_needs_human_review"))
        id_model = item.get("id_ocr_selected_model")
        id_stage_idx = item.get("id_ocr_selected_stage_index")

        if pending_all_ids:
            need_id_review = True
        elif pending_id:
            need_id_review = not is_valid_student_id(raw_id)
            if not need_id_review and id_use_review_flag and id_review_flag is not None:
                need_id_review = bool(id_review_flag)
            if not need_id_review and id_min_f is not None:
                need_id_review = id_min_f < thresh_id
        else:
            need_id_review = False
        if need_id_review:
            id_ref = make_review_target_ref(
                pdf_stem, item_index, "_", "__id__", "id"
            )
            id_row: dict[str, Any] = {
                "review_target_ref": id_ref,
                "entry_id": make_entry_id(
                    pdf_stem, item_index, "_", "__id__", "id"
                ),
                "item_index": item_index,
                "page_in_pair": None,
                "roi_name": "__id__",
                "kind": "id",
                "id_current": raw_id,
                "corrected_id": None,
                "status": "pending",
                "review_status": "pending",
            }
            if id_min_f is not None:
                id_row["id_ocr_min_score"] = round(id_min_f, 6)
            if id_mean_f is not None:
                id_row["id_ocr_mean_score"] = round(id_mean_f, 6)
            if id_label:
                id_row["id_ocr_confidence_label"] = id_label
            if id_min_f is not None:
                id_row["id_ocr_confidence_score"] = round(id_min_f, 6)
            if id_review_flag is not None:
                id_row["id_ocr_needs_human_review"] = bool(id_review_flag)
            if id_model is not None:
                id_row["id_ocr_selected_model"] = id_model
            if id_stage_idx is not None:
                id_row["id_ocr_selected_stage_index"] = id_stage_idx
            pending.append(id_row)

        # --- Text ROI: from OCR module queue (preferred) ---
        if text_ocr_queue is not None:
            for e in text_ocr_queue:
                if not isinstance(e, dict):
                    continue
                if str(e.get("kind", "")).lower() != "text":
                    continue
                if int(e.get("item_index", -1)) != item_index:
                    continue
                pending.append(_pending_dict_from_text_queue_entry(pdf_stem, e))
        else:
            for row in item.get("data") or []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("kind", "")).lower() != "text":
                    continue
                score = _coerce_float_or_none(row.get("ocr_confidence_score"))
                if score is None:
                    score = _coerce_float_or_none(row.get("ocr_min_score"))
                mean_score = _coerce_float_or_none(row.get("ocr_mean_score"))
                review_flag = _coerce_optional_bool(row.get("ocr_needs_human_review"))
                should_enqueue = False
                if text_use_review_flag and review_flag is not None:
                    should_enqueue = bool(review_flag)
                elif score is not None:
                    should_enqueue = score < thresh
                if not should_enqueue:
                    continue

                page_in_pair = str(row.get("page_in_pair", "") or "unknown")
                roi_name = str(row.get("name", ""))

                ref = row.get("review_target_ref") or make_review_target_ref(
                    pdf_stem, item_index, page_in_pair, roi_name, "text"
                )
                pending.append(
                    {
                        "review_target_ref": ref,
                        "entry_id": make_entry_id(
                            pdf_stem, item_index, page_in_pair, roi_name, "text"
                        ),
                        "item_index": item_index,
                        "page_in_pair": page_in_pair,
                        "roi_name": roi_name,
                        "kind": "text",
                        "ocr_min_score": round(score, 6) if score is not None else None,
                        "ocr_mean_score": round(mean_score, 6)
                        if mean_score is not None
                        else None,
                        "ocr_confidence_label": (
                            str(row.get("ocr_confidence_label") or "").lower() or None
                        ),
                        "ocr_confidence_score": (
                            round(score, 6) if score is not None else None
                        ),
                        "ocr_needs_human_review": (
                            bool(review_flag) if review_flag is not None else bool(should_enqueue)
                        ),
                        "ocr_selected_model": row.get("ocr_selected_model"),
                        "ocr_selected_stage_index": row.get("ocr_selected_stage_index"),
                        "text_current": str(row.get("text", "")),
                        "corrected_text": None,
                        "status": "pending",
                        "review_status": "pending",
                    }
                )

        # --- MCQ empty ---
        for row in item.get("data") or []:
            if not isinstance(row, dict):
                continue
            rk = str(row.get("kind", "")).lower()
            if rk != "mcq" or not pending_mcq_empty:
                continue
            txt = str(row.get("text", "")).strip()
            if txt != "":
                continue
            page_in_pair = str(row.get("page_in_pair", "") or "unknown")
            roi_name = str(row.get("name", ""))
            mcq_ref = row.get("review_target_ref") or make_review_target_ref(
                pdf_stem, item_index, page_in_pair, roi_name, "mcq"
            )
            pending.append(
                {
                    "review_target_ref": mcq_ref,
                    "entry_id": make_entry_id(
                        pdf_stem, item_index, page_in_pair, roi_name, "mcq"
                    ),
                    "item_index": item_index,
                    "page_in_pair": page_in_pair,
                    "roi_name": roi_name,
                    "kind": "mcq",
                    "text_current": "",
                    "corrected_text": None,
                    "status": "pending",
                    "review_status": "pending",
                }
            )

    # Dedupe text pending by review_target_ref (queue + fallback could overlap)
    seen_text_ref: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for p in pending:
        ref = p.get("review_target_ref")
        k = str(p.get("kind", "")).lower()
        if k == "text" and ref:
            if ref in seen_text_ref:
                continue
            seen_text_ref.add(str(ref))
        deduped.append(p)

    return {
        "text_review_use_needs_human_review": text_use_review_flag,
        "id_review_use_needs_human_review": id_use_review_flag,
        "min_text_ocr_confidence": thresh,
        "min_id_ocr_confidence": thresh_id,
        "pending": deduped,
    }


def apply_manual_corrections(
    items: list[dict[str, Any]],
    human_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Apply ``corrected_id`` / ``corrected_text`` from ``human_review["pending"]`` onto a copy of *items*.

    Matches ``review_target_ref`` on ``data[]`` rows when present; otherwise uses
    ``item_index`` + ``roi_name`` + ``kind`` + ``page_in_pair``.
    """
    out = copy.deepcopy(items)
    for p in human_review.get("pending") or []:
        if not isinstance(p, dict):
            continue
        if str(p.get("status", "pending")).lower() in ("skipped", "cancelled"):
            continue
        try:
            idx = int(p["item_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(out):
            continue

        kind = str(p.get("kind", "")).lower()
        pref = p.get("review_target_ref")

        if kind == "id":
            cid = p.get("corrected_id")
            if cid is None:
                continue
            if str(cid).strip() == "":
                continue
            out[idx]["id"] = str(cid).strip().upper()
            continue

        if kind not in ("text", "mcq"):
            continue
        ct = p.get("corrected_text")
        if ct is None:
            continue
        ct = str(ct).strip() if isinstance(ct, str) else str(ct)
        if ct == "":
            continue

        roi_name = str(p.get("roi_name", ""))
        page_in_pair = p.get("page_in_pair")
        page_str = str(page_in_pair) if page_in_pair not in (None, "") else None

        matched = False
        if pref:
            for row in out[idx].get("data") or []:
                if not isinstance(row, dict):
                    continue
                if str(row.get("review_target_ref", "")) == str(pref):
                    row["text"] = ct
                    matched = True
                    break
        if matched:
            continue

        for row in out[idx].get("data") or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("name", "")) != roi_name:
                continue
            if str(row.get("kind", "")).lower() != kind:
                continue
            if page_str is not None and str(row.get("page_in_pair", "")) != page_str:
                continue
            row["text"] = ct
            break

    return out
