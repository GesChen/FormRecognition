"""
Extract ID and form type from image(s): crop top fraction of image → dual OCR calls.

Current workflow:
1) ID from side-a schema ROI ("id")
2) form_type from configurable top-crop OCR prompt

Ref: prompts/module — single-call usage after import.

Usage:

    from id_form_llm import extract_id_and_form_type, extract_id_and_form_type_batch

    result = extract_id_and_form_type("path/to/page.png")
    # -> {"id": "9010526A", "form_type": "6post"}

    results = extract_id_and_form_type_batch(["path/to/p1.png", "path/to/p2.png"])
    # -> [{"id": "9010526A", "form_type": "6post"}, ...]

Config: HEADER_RECOGNITION["crop_top_percent"], HEADER_RECOGNITION["form_types"],
HEADER_RECOGNITION["crop_write_debug_image"] / ["crop_debug_dir"] (optional PNG of the OCR crop).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from concurrent.futures import ProcessPoolExecutor, as_completed

from config import HEADER_RECOGNITION, OCR_ENGINE, PATHS, PDF_RECOGNITION

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm not installed
    def tqdm(iterable, desc=None, **kwargs):
        return iterable


def _tqdm_enabled() -> bool:
    """
    Render live tqdm bars only on interactive terminals.
    Prevents CR-updated bars from appearing as many lines in web log previews.
    """
    env = str(os.environ.get("EVMS_TQDM", "auto")).strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    if str(os.environ.get("TERM", "")).strip().lower() == "dumb":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _normalize_header_ocr(ocr_text: str) -> str:
    """Uppercase NFKC text with punctuation collapsed to spaces for rule matching."""
    if not ocr_text:
        return ""
    s = unicodedata.normalize("NFKC", ocr_text).upper()
    s = s.replace("：", " ").replace("，", " ").replace("－", "-")
    s = re.sub(r"[^A-Z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _detect_grade_level(norm: str) -> str | None:
    """
    Return '6', '7', '8', or 'h' from noisy header OCR, or None.
    Tolerates spacing/punctuation loss (e.g. 6 T H GRADE, 6TH GRADE).
    """
    if not norm:
        return None
    compact = re.sub(r"\s+", "", norm)

    if re.search(r"HIGH\s*SCHOOL", norm) or ("HIGH" in norm and "SCHOOL" in norm):
        return "h"

    for digit in ("8", "7", "6"):
        if re.search(rf"\b{digit}\s*T\s*H\b", norm):
            return digit
        if re.search(rf"\b{digit}\s+T\s+G\s*R\s*A\s*D\s*E\b", norm):
            return digit
        if re.search(rf"\b{digit}\s*TH\s*GRADE\b", norm):
            return digit
        frag = f"{digit}TH"
        if frag in compact and "GRADE" in compact:
            idx = compact.find(frag)
            j = compact.find("GRADE")
            if idx != -1 and j != -1 and abs(idx - j) < 18:
                return digit
    return None


def _detect_pre_post(norm: str) -> str | None:
    """Return 'pre' or 'post' from assessment wording, or None."""
    if not norm:
        return None
    pre_m = re.search(r"PRE[\s\-_]*ASSESS", norm)
    post_m = re.search(r"POST[\s\-_]*ASSESS", norm)
    if pre_m and post_m:
        return "pre" if pre_m.start() < post_m.start() else "post"
    if pre_m:
        return "pre"
    if post_m:
        return "post"
    head = norm[: min(350, len(norm))]
    if "POST" in head:
        if re.search(r"\bPRE\b", head) and head.find("PRE") < head.find("POST"):
            return "pre"
        return "post"
    if re.search(r"\bPRE\b", head) or "PRE-" in head:
        return "pre"
    return None


def infer_form_type_from_ocr(ocr_text: str, allowed: list[str]) -> str | None:
    """
    Deterministic form_type from top-crop OCR only (noise-tolerant).
    Returns one of allowed (e.g. '6post') or None if no confident match.
    """
    allowed_set = set(allowed or [])
    norm = _normalize_header_ocr(ocr_text)
    g = _detect_grade_level(norm)
    t = _detect_pre_post(norm)
    if g and t:
        cand = f"{g}{t}"
        if cand in allowed_set:
            return cand
    return None


def _id_instruction(form_types_str: str) -> str:
    """Reusable extraction rules for noisy OCR -> id + form_type."""
    return (
        "Your job has only two parts.\n"
        "1. Extract the id.\n"
        "   - The final id must be exactly 7 digits followed by one letter A or B.\n"
        "   - OCR is noisy. Common confusions: O/Q/D->0, I/L->1, S->5, B->8, G->6.\n"
        "   - If the text contains a colon, the part before the first colon is a label, not part of the id.\n"
        "     Example: '10:6010014A' -> id = '6010014A'.\n"
        "   - Prefer the most plausible normalized 7-digit+A/B sequence in the OCR text.\n"
        "   - If no plausible id is present, return null.\n"
        "2. Choose the form_type.\n"
        f"   - Pick the single allowed type that is best represented by the OCR text: {form_types_str}.\n"
        "   - Determine grade/level ONLY from explicit words like '6th Grade', '7th Grade', '8th Grade', or 'High School'.\n"
        "     Do NOT infer grade from digits inside the id.\n"
        "   - Determine pre vs post ONLY from explicit words like 'Pre-Assessment'/'Pre Assessment'/'PRE' or 'Post-Assessment'/'Post Assessment'/'POST'.\n"
        "   - Use this direct mapping:\n"
        "       * '6th Grade' + pre  -> 6pre\n"
        "       * '6th Grade' + post -> 6post\n"
        "       * '7th Grade' + pre  -> 7pre\n"
        "       * '7th Grade' + post -> 7post\n"
        "       * '8th Grade' + pre  -> 8pre\n"
        "       * '8th Grade' + post -> 8post\n"
        "       * 'High School' + pre  -> hpre\n"
        "       * 'High School' + post -> hpost\n"
        "   - If no allowed type is clearly supported, return null."
    )


def build_prompt(ocr_content: str | list[str], form_types: list[str]) -> str:
    """
    Build the LLM prompt for ID + form-type extraction. Override or replace this
    function to change behavior.

    Args:
        ocr_content: For single image, one string. For batch, list of strings (one per image).
        form_types: Allowed form type values from config.

    Returns:
        Prompt string. The LLM must respond with JSON only.
    """
    types_str = ", ".join(form_types)
    id_instr = _id_instruction(types_str)
    if isinstance(ocr_content, list):
        blocks = []
        for i, text in enumerate(ocr_content, start=1):
            t = text or "(no text)"
            block = f"--- Image {i} ---\n{t}"
            if ":" in t:
                after = t.split(":", 1)[1].strip()
                if after:
                    block += f"\n(For this image, use only the part after the first colon for the id: {after!r})"
            blocks.append(block)
        content = "\n\n".join(blocks)
        return (
            "You will receive OCR text from the top crop of multiple form pages.\n"
            "For each image, do only these tasks:\n"
            "- extract one normalized 7-digit+A/B id from the noisy OCR text\n"
            "- choose the best matching allowed form_type from the OCR text\n"
            "- return strict JSON only\n\n"
            "OCR text:\n\n"
            f"{content}\n\n"
            "Rules:\n"
            f"{id_instr}\n\n"
            "Return ONE JSON object only.\n"
            "Do NOT include any prose/explanation.\n"
            "Do NOT use markdown.\n"
            "Do NOT wrap the JSON in code fences.\n"
            "Use exactly this shape:\n"
            '{"results": [{"id": "... or null", "form_type": "... or null"}, ...]}\n'
            "One object per image in the same order as the images above."
        )
    else:
        content = ocr_content or "(no text)"
        # When there's a colon, spell out the after-colon segment so the model can't merge label + id.
        colon_hint = ""
        if ":" in content:
            after_colon = content.split(":", 1)[1].strip()
            if after_colon:
                colon_hint = f"\n(The part after the first colon is: {after_colon!r}. Use only this part for the id; do not use anything before the colon.)\n"
        return (
            "You will receive OCR text from the top crop of one form page.\n"
            "Do only these tasks:\n"
            "- extract one normalized 7-digit+A/B id from the noisy OCR text\n"
            "- choose the best matching allowed form_type from the OCR text\n"
            "- return strict JSON only\n\n"
            "OCR text:\n\n"
            f"{content}\n\n"
            f"{colon_hint}"
            "Rules:\n"
            f"{id_instr}\n\n"
            "Return ONE JSON object only.\n"
            "Do NOT include any prose/explanation.\n"
            "Do NOT use markdown.\n"
            "Do NOT wrap the JSON in code fences.\n"
            "Use exactly this shape:\n"
            '{"id": "... or null", "form_type": "... or null"}'
        )


def build_prompt_id_only(ocr_content: str | list[str]) -> str:
    """
    LLM prompt for ID extraction only (form_type comes from infer_form_type_from_ocr).
    """
    if isinstance(ocr_content, list):
        blocks = []
        for i, text in enumerate(ocr_content, start=1):
            t = text or "(no text)"
            block = f"--- Image {i} ---\n{t}"
            if ":" in t:
                after = t.split(":", 1)[1].strip()
                if after:
                    block += f"\n(For this image, use only the part after the first colon for the id: {after!r})"
            blocks.append(block)
        content = "\n\n".join(blocks)
        return (
            "You will receive OCR text from the top crop of multiple form pages.\n"
            "For EACH image extract ONLY the student id.\n"
            "The id must be exactly 7 digits followed by one letter A or B (after normalizing OCR noise).\n"
            "OCR confusions: O/Q/D->0, I/L->1, S->5, B->8 in digit positions, G->6.\n"
            "If the text contains a colon, the part before the first colon is a label, NOT part of the id.\n"
            "If no plausible id exists, return null for that image.\n\n"
            "OCR text:\n\n"
            f"{content}\n\n"
            "Return ONE JSON object only.\n"
            "No prose, no markdown, no code fences.\n"
            'Use exactly: {"results": [{"id": "... or null"}, ...]}\n'
            "One entry per image in the same order."
        )
    content = ocr_content or "(no text)"
    colon_hint = ""
    if ":" in content:
        after_colon = content.split(":", 1)[1].strip()
        if after_colon:
            colon_hint = f"\n(The part after the first colon is: {after_colon!r}. Use only this part for the id.)\n"
    return (
        "You will receive OCR text from the top crop of one form page.\n"
        "Extract ONLY the student id.\n"
        "The id must be exactly 7 digits followed by one letter A or B (after normalizing OCR noise).\n"
        "OCR confusions: O/Q/D->0, I/L->1, S->5, B->8 in digit positions, G->6.\n"
        "If the text contains a colon, the part before the first colon is a label, NOT part of the id.\n"
        "If no plausible id exists, return null for id.\n\n"
        "OCR text:\n\n"
        f"{content}\n\n"
        f"{colon_hint}"
        "Return ONE JSON object only.\n"
        "No prose, no markdown, no code fences.\n"
        'Use exactly: {"id": "... or null"}'
    )


def _top_rows_from_percent(image_height: int, percent: float) -> int:
    """Rows to keep from the top; *percent* is 0–100 of image height (at least 1 row when h > 0)."""
    if image_height <= 0:
        return 0
    p = max(0.01, min(100.0, float(percent)))
    return max(1, min(image_height, int(round(image_height * p / 100.0))))


def _crop_pct_filename_suffix(pct: float) -> str:
    """Safe fragment for filenames, e.g. 11.0 -> '11', 8.5 -> '8p5'."""
    s = f"{float(pct):g}"
    return s.replace(".", "p")


def _resolve_crop_debug_path_single(
    cfg: dict,
    image_path: Path,
    pct: float,
    crop_debug_out: Path | None,
) -> Path | None:
    """Path for one PNG of the top crop, or None to skip writing."""
    if crop_debug_out is not None:
        p = Path(crop_debug_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    if not bool(cfg.get("crop_write_debug_image")):
        return None
    d = Path(cfg.get("crop_debug_dir") or (PATHS["output"] / "debug_images" / "id_crop"))
    d.mkdir(parents=True, exist_ok=True)
    suf = _crop_pct_filename_suffix(pct)
    return d / f"{image_path.stem}_crop_top{suf}pct.png"


def _resolve_crop_debug_path_batch(
    cfg: dict,
    image_path: Path,
    pct: float,
    index: int,
    crop_debug_dir: Path | None,
) -> Path | None:
    """Path for batch item *index* PNG, or None to skip."""
    d = crop_debug_dir
    if d is None and bool(cfg.get("crop_write_debug_image")):
        d = cfg.get("crop_debug_dir") or (PATHS["output"] / "debug_images" / "id_crop")
    if d is None:
        return None
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    suf = _crop_pct_filename_suffix(pct)
    return d / f"{image_path.stem}_{index:04d}_crop_top{suf}pct.png"


def _crop_debug_will_write(cfg: dict, crop_debug_dir: Path | None) -> bool:
    return crop_debug_dir is not None or bool(cfg.get("crop_write_debug_image"))


def _cfg() -> dict[str, Any]:
    return HEADER_RECOGNITION if isinstance(HEADER_RECOGNITION, dict) else {}


def _header_ocr_workflow_override() -> str | None:
    cfg = _cfg()
    raw = cfg.get("ocr_workflow_override")
    if raw is None:
        raw = cfg.get("workflow_override")
    workflow = str(raw or "").strip()
    return workflow or None


def _ocr_raw_header(image_path: str | Path) -> dict[str, Any]:
    """
    OCR helper for header ID/form-type extraction.

    HEADER_RECOGNITION["ocr_workflow_override"] accepts the same workflow values
    as OCR_ENGINE["workflow_default"]. When unset, this uses the normal OCR
    engine default.
    """
    from ocr_engine import ocr_raw

    workflow = _header_ocr_workflow_override()
    if not workflow:
        raw = ocr_raw(image_path)
        return raw if isinstance(raw, dict) else {}

    old_workflow = OCR_ENGINE.get("workflow_default") if isinstance(OCR_ENGINE, dict) else None
    try:
        if isinstance(OCR_ENGINE, dict):
            OCR_ENGINE["workflow_default"] = workflow
        raw = ocr_raw(image_path)
        return raw if isinstance(raw, dict) else {}
    finally:
        if isinstance(OCR_ENGINE, dict):
            OCR_ENGINE["workflow_default"] = old_workflow


def _crop_top_and_ocr(
    image_path: Path,
    crop_top_percent: float,
    *,
    crop_debug_out: Path | None = None,
):
    """Load image, keep top ``crop_top_percent`` % of height, run OCR; return text."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    h = img.shape[0]
    rows = _top_rows_from_percent(h, crop_top_percent)
    cropped = img[0:rows, :]
    if crop_debug_out is not None:
        dbg = Path(crop_debug_out)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg), cropped)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, cropped)
        raw_out = _ocr_raw_header(tmp)
        return str(raw_out.get("detected_text", "") or "").strip()
    finally:
        Path(tmp).unlink(missing_ok=True)


def _crop_top_and_ocr_with_raw(
    image_path: Path,
    crop_top_percent: float,
    *,
    crop_debug_out: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Load image, keep top ``crop_top_percent`` % of height, run OCR; return (text, raw_ocr_output)."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return "", {}
    h = img.shape[0]
    rows = _top_rows_from_percent(h, crop_top_percent)
    cropped = img[0:rows, :]
    if crop_debug_out is not None:
        crop_debug_out = Path(crop_debug_out)
        crop_debug_out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_debug_out), cropped)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        cv2.imwrite(tmp, cropped)
        raw_out = _ocr_raw_header(tmp)
        text = str(raw_out.get("detected_text", "") or "").strip()
        return text, raw_out
    finally:
        Path(tmp).unlink(missing_ok=True)


def _crop_top_and_ocr_form_type_with_raw(
    image_path: Path,
    crop_top_percent: float,
    *,
    crop_debug_out: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Top-crop OCR call dedicated to form_type extraction."""
    import cv2
    img = cv2.imread(str(image_path))
    if img is None:
        return "", {}
    h = img.shape[0]
    rows = _top_rows_from_percent(h, crop_top_percent)
    cropped = img[0:rows, :]
    if crop_debug_out is not None:
        dbg = Path(crop_debug_out)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg), cropped)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    try:
        cv2.imwrite(str(tmp), cropped)
        raw_out = _ocr_raw_header(tmp)
        text = str(raw_out.get("detected_text", "") or "").strip()
        return text, raw_out
    finally:
        tmp.unlink(missing_ok=True)


def _header_schema_path_sidea() -> Path:
    cfg = PDF_RECOGNITION if isinstance(PDF_RECOGNITION, dict) else {}
    schema_dir = Path(
        cfg.get("schema_dir", Path(__file__).resolve().parent.parent / "data" / "roi_schemas")
    ).resolve()
    key = str(cfg.get("schema_key_sidea", "schema_sidea") or "schema_sidea").strip()
    return schema_dir / f"{key}.json"


def _roi_meta_value(meta: dict[str, Any] | None, *keys: str) -> str:
    if not isinstance(meta, dict):
        return ""
    for k in keys:
        v = meta.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _find_named_roi(
    raw_schema: dict[str, Any],
    *,
    name: str,
) -> dict[str, Any] | None:
    rois = raw_schema.get("rois")
    if not isinstance(rois, list):
        return None
    target = str(name).strip().lower()
    for r in rois:
        if not isinstance(r, dict):
            continue
        if str(r.get("name", "")).strip().lower() == target:
            return r
    return None


def _crop_named_roi_image(
    image_path: Path,
    *,
    raw_schema: dict[str, Any],
    roi: dict[str, Any],
    crop_debug_out: Path | None = None,
) -> Path | None:
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    ih, iw = img.shape[:2]
    try:
        rx = float(roi.get("x", 0))
        ry = float(roi.get("y", 0))
        rw = float(roi.get("w", 0))
        rh = float(roi.get("h", 0))
    except Exception:
        return None
    if rw <= 0 or rh <= 0:
        return None

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
        return None

    crop = img[y1:y2, x1:x2]
    if crop_debug_out is not None:
        dbg = Path(crop_debug_out)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg), crop)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    cv2.imwrite(str(tmp), crop)
    return tmp


def _ocr_roi_with_raw(
    image_path: Path,
    *,
    raw_schema: dict[str, Any],
    roi: dict[str, Any],
    crop_debug_out: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    tmp = _crop_named_roi_image(
        image_path,
        raw_schema=raw_schema,
        roi=roi,
        crop_debug_out=crop_debug_out,
    )
    if tmp is None:
        return "", {}
    try:
        raw_out = _ocr_raw_header(tmp)
        text = str(raw_out.get("detected_text", "") or "").strip()
        return text, raw_out
    finally:
        tmp.unlink(missing_ok=True)


def _id_llm_meta_from_roi(roi: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "id": {
            "llm_field_data_type": _roi_meta_value(roi, "llm_field_data_type"),
            "llm_validation_rules": _roi_meta_value(roi, "llm_validation_rules"),
            "llm_prompt_instruction": _roi_meta_value(roi, "llm_prompt_instruction"),
            "llm_prompt_override": _roi_meta_value(roi, "llm_prompt_override"),
        }
    }


def _form_type_llm_meta(allowed_form_types: list[str]) -> dict[str, dict[str, Any]]:
    cfg = _cfg()
    prompt_override = str(cfg.get("form_type_prompt_override", "") or "").strip()
    allowed = ", ".join(str(x).strip() for x in (allowed_form_types or []) if str(x).strip())
    validation = f"Must be one of {allowed} or null" if allowed else "Must be canonical form_type or null"
    enum_values = [str(x).strip() for x in (allowed_form_types or []) if str(x).strip()]
    if enum_values:
        response_format: dict[str, Any] = {
            "type": "object",
            "properties": {
                "detected_text": {
                    "anyOf": [
                        {"type": "string", "enum": enum_values},
                        {"type": "null"},
                    ]
                }
            },
            "required": ["detected_text"],
            "additionalProperties": False,
        }
    else:
        response_format = {
            "type": "object",
            "properties": {
                "detected_text": {"type": ["string", "null"]},
            },
            "required": ["detected_text"],
            "additionalProperties": False,
        }
    return {
        "form_type": {
            "llm_model": cfg.get("model"),
            "llm_field_data_type": "form_type",
            "llm_validation_rules": validation,
            "llm_prompt_instruction": "Normalize header OCR text into canonical form_type value.",
            "llm_prompt_override": prompt_override,
            "llm_response_format": response_format,
            "llm_options": {"temperature": 0},
        }
    }


def _postprocess_header_fields_with_llm(
    text_map: dict[str, str],
    *,
    roi_meta_by_name: dict[str, dict[str, Any]],
    use_tqdm: bool = False,
    tqdm_desc: str = "      Header LLM deferred",
) -> dict[str, str]:
    """
    Apply built-in ROI LLM overrides on header fields (id/form_type) using the same
    postprocess module as text ROIs.
    """
    try:
        from text_roi_llm import postprocess_text_rois
    except Exception:
        return dict(text_map)
    return postprocess_text_rois(
        text_map,
        roi_meta_by_name=roi_meta_by_name,
        debug_out=None,
        use_tqdm=use_tqdm,
        tqdm_desc=tqdm_desc,
    )


def id_top_crop_rec_stats(image_path: str | Path) -> dict[str, Any]:
    """
    OCR recognition score stats for the ID top crop (same geometry as extract_id).

    Returns a dict compatible with :func:`ocr_engine.ocr_confidence_stats`
    (``min_rec_score``, ``mean_rec_score``, etc.).
    """
    from ocr_engine import ocr_confidence_stats

    cfg = _cfg()
    pct = float(cfg.get("crop_top_percent", 8.0))
    path = Path(image_path)
    if not path.is_file():
        return {
            "min_rec_score": None,
            "mean_rec_score": None,
            "rec_scores": [],
            "rec_texts": [],
        }
    _, raw_result = _crop_top_and_ocr_with_raw(path, pct)
    return ocr_confidence_stats(raw_result)


def _parse_llm_json(text: str) -> Any:
    """
    Extract JSON from an LLM response.

    The model sometimes returns:
      - Markdown code fences (```json ... ```)
      - Valid JSON followed by extra explanation text
      - Leading prose before the JSON

    This function extracts the first valid JSON object/array it can decode and
    ignores any trailing text.
    """
    if not text or not text.strip():
        return {}
    s = str(text).strip()

    # Prefer content inside markdown code fences when present.
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    if not s:
        return {}

    # Fast path: response is pure JSON.
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Robust path: decode the first JSON value and ignore trailing text.
    decoder = json.JSONDecoder()
    starts = [i for i in (s.find("{"), s.find("[")) if i != -1]
    if not starts:
        return {}
    i0 = min(starts)
    tail = s[i0:]
    try:
        obj, _end = decoder.raw_decode(tail)
        return obj
    except json.JSONDecodeError:
        pass
    return {}


def _normalize_form_type(value: Any, allowed: list[str]) -> str | None:
    """Return value if it is exactly one of allowed; otherwise None."""
    if value is None:
        return None
    s = str(value).strip()
    return s if s in allowed else None


def _normalize_id(value: Any) -> str | None:
    """
    Normalize OCR/LLM id candidate.

    Preference order:
      1) Return a correctly shaped id: 7 digits + trailing A/B (after common OCR fixes).
      2) Otherwise, return a best-effort cleaned string (uppercased, stripped, OCR fixes),
         so the pipeline still has something usable in the `id` field.
    """
    if value is None:
        return None

    cfg = _cfg()
    raw_toggle = cfg.get("normalize_id_output", True)
    if isinstance(raw_toggle, bool):
        normalize_enabled = raw_toggle
    elif isinstance(raw_toggle, (int, float)):
        normalize_enabled = bool(raw_toggle)
    else:
        normalize_enabled = str(raw_toggle).strip().lower() in {"1", "true", "yes", "y", "on"}

    # Toggleable behavior: when disabled, keep model output as-is (trim only).
    if not normalize_enabled:
        raw = str(value).strip()
        return raw or None

    s = str(value).strip().upper()
    if not s:
        return None
    # Find a plausible 7-digit run followed by A/B. We intentionally do NOT apply
    # "B->8" globally because the trailing letter is semantically part of the ID.
    #
    # Instead, we normalize OCR confusions only inside the 7-digit portion.
    m = re.search(r"([0-9OQDI LSBG]{7})([AB])", s.replace("\t", " ").replace("\n", " "))
    if m is None:
        # Fallback: try to find any 7 digits + A/B without normalization.
        m2 = re.search(r"(\d{7})([AB])", s)
        if m2:
            return f"{m2.group(1)}{m2.group(2)}"
    else:
        digits_raw, suffix = m.group(1), m.group(2)
        digits = (
            digits_raw.replace(" ", "")
            .replace("O", "0")
            .replace("Q", "0")
            .replace("D", "0")
            .replace("I", "1")
            .replace("L", "1")
            .replace("S", "5")
            .replace("B", "8")
            .replace("G", "6")
        )
        digits = re.sub(r"\D", "", digits)
        if len(digits) == 7:
            return f"{digits}{suffix}"

    # Best-effort fallback: keep something usable even when strict shape isn't present.
    # Apply common OCR confusions and keep only alphanumerics.
    cleaned = (
        s.replace(" ", "")
        .replace("\t", "")
        .replace("\n", "")
        .replace("O", "0")
        .replace("Q", "0")
        .replace("D", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("S", "5")
        .replace("G", "6")
    )
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    if not cleaned:
        return None
    # Prefer keeping a plausible 8-char tail if it ends with A/B, else keep first 16 chars.
    m3 = re.search(r"(\d{7}[AB])", cleaned)
    if m3:
        return m3.group(1)
    return cleaned[:16]


def _extract_stage_parsed(raw_ocr: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw_ocr, dict):
        return {}
    stages = raw_ocr.get("stages")
    if isinstance(stages, list) and stages:
        last = stages[-1] if isinstance(stages[-1], dict) else {}
        parsed = last.get("parsed")
        if isinstance(parsed, dict):
            return parsed
    parsed = raw_ocr.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    return {}


def _extract_id_from_ocr_raw(raw_ocr: dict[str, Any] | None, fallback_text: str = "") -> str | None:
    def _is_canonical(v: Any) -> bool:
        return isinstance(v, str) and re.fullmatch(r"[0-9]{7}[AB]", v or "") is not None

    parsed = _extract_stage_parsed(raw_ocr)
    val = _normalize_id(parsed.get("id"))
    if _is_canonical(val):
        return val
    if isinstance(raw_ocr, dict):
        val = _normalize_id(raw_ocr.get("id"))
        if _is_canonical(val):
            return val
    val = _normalize_id(fallback_text)
    if _is_canonical(val):
        return val
    return None


def _extract_form_type_from_ocr_raw(
    raw_ocr: dict[str, Any] | None,
    allowed: list[str],
    fallback_text: str = "",
) -> str | None:
    # Prefer deterministic extraction from OCR text signals first.
    # This avoids accepting an incorrect canonical label returned by
    # downstream normalization when the raw header text already contains
    # explicit grade/timing evidence.
    ocr_text_candidates: list[str] = []
    parsed = _extract_stage_parsed(raw_ocr)
    if isinstance(parsed, dict):
        ocr_text_candidates.append(str(parsed.get("detected_text") or ""))
    if isinstance(raw_ocr, dict):
        ocr_text_candidates.append(str(raw_ocr.get("detected_text") or ""))
    ocr_text_candidates.append(str(fallback_text or ""))

    for txt in ocr_text_candidates:
        det = infer_form_type_from_ocr(txt, allowed)
        if det:
            return det

    candidate = _normalize_form_type(parsed.get("form_type"), allowed)
    if candidate:
        return candidate
    if isinstance(raw_ocr, dict):
        candidate = _normalize_form_type(raw_ocr.get("form_type"), allowed)
        if candidate:
            return candidate
    # accept explicit canonical value from fallback text directly
    direct = _normalize_form_type(fallback_text, allowed)
    if direct:
        return direct
    # fallback to deterministic pattern inference from OCR text
    return infer_form_type_from_ocr(fallback_text, allowed)


def _coerce_llm_batch_results(data: Any) -> list[dict[str, Any]] | None:
    """
    Accept a few common shapes and coerce to a flat list of dicts:
      - {"results": [ {...}, {...} ]}
      - {"id": ..., "form_type": ...}  (single-image call)
      - [ {"id":..., "form_type":...}, ... ]
      - [ {"results": [ {...} ]}, {"results":[{...}]} ]  (one wrapper per image)
    Returns None if not comprehensible.
    """
    if data is None:
        return None
    # {"id":..., "form_type":...}
    if isinstance(data, dict) and ("id" in data or "form_type" in data):
        return [data]
    # {"results": [...]}
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        out = [r for r in data.get("results") if isinstance(r, dict)]
        return out
    # [ ... ]
    if isinstance(data, list):
        # If list is already dicts with id/form_type, keep them.
        if all(isinstance(x, dict) for x in data):
            # Flatten any per-item {"results":[...]} wrappers.
            flattened: list[dict[str, Any]] = []
            for x in data:
                if isinstance(x.get("results"), list):
                    nested = [r for r in (x.get("results") or []) if isinstance(r, dict)]
                    if nested:
                        flattened.extend(nested)
                    else:
                        flattened.append({})
                else:
                    flattened.append(x)
            return flattened
    return None


def _llm_generate_with_retries(
    prompt: str,
    *,
    expected_count: int,
    form_types: list[str],
    max_reruns: int,
    verbose: bool = False,
    debug_out: dict | None = None,
    id_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Call LLM and enforce a strict, comprehensible output shape.
    Retries (reruns) the LLM call when parsing/normalization fails, up to max_reruns.
    If id_only is True, only \"id\" is read from the model; form_type is left None here.
    """
    from llm_client import generate

    attempts: list[dict[str, Any]] = []
    last_raw_text = ""
    if debug_out is not None:
        # Always store the exact prompt used for this retry loop.
        debug_out["prompt"] = prompt
    for attempt in range(max_reruns + 1):
        if verbose and attempt > 0:
            print(f"[HEADER_RECOGNITION] Re-running LLM (attempt {attempt + 1}/{max_reruns + 1}) due to invalid JSON shape...")
        out = generate(prompt)
        raw_text = out.get("text") or ""
        last_raw_text = raw_text
        data = _parse_llm_json(raw_text)
        coerced = _coerce_llm_batch_results(data)

        ok = isinstance(coerced, list) and len(coerced) >= expected_count
        normalized: list[dict[str, Any]] = []
        if ok:
            for i in range(expected_count):
                r = coerced[i] if i < len(coerced) and isinstance(coerced[i], dict) else {}
                normalized.append(
                    {
                        "id": _normalize_id(r.get("id")),
                        "form_type": (
                            None
                            if id_only
                            else _normalize_form_type(r.get("form_type"), form_types)
                        ),
                    }
                )
            # Strict shape achieved: list length == expected_count and dict keys present.
            ok = len(normalized) == expected_count and all(isinstance(x, dict) for x in normalized)

        attempts.append(
            {
                "attempt": attempt + 1,
                "prompt": prompt,
                "raw_response": raw_text,
                "parsed_ok": bool(ok),
                "parsed_type": type(data).__name__,
                "eval_count": out.get("eval_count"),
                "eval_duration": out.get("eval_duration"),
                "elapsed": out.get("elapsed"),
            }
        )
        if ok:
            if debug_out is not None:
                debug_out.setdefault("llm_attempts", []).extend(attempts)
                debug_out["last_ok"] = True
            return normalized

    if debug_out is not None:
        debug_out.setdefault("llm_attempts", []).extend(attempts)
        debug_out["last_ok"] = False
        debug_out["llm_failure"] = {
            "reason": "Could not coerce LLM output into strict expected shape",
            "expected_count": expected_count,
            "last_raw_response": last_raw_text,
        }
    # Return strict fallback shape (all nulls) if we still failed.
    return [{"id": None, "form_type": None} for _ in range(expected_count)]


def extract_id_and_form_type(
    image_path: str | Path,
    *,
    crop_top_percent: float | None = None,
    form_types: list[str] | None = None,
    return_raw_ocr: bool = False,
    verbose: bool = False,
    crop_debug_out: Path | None = None,
) -> dict[str, Any]:
    """
    Single image:
    - ID from side-a schema ROI named "id"
    - form_type from top-crop OCR using HEADER_RECOGNITION["form_type_prompt_override"]

    Config defaults from HEADER_RECOGNITION["crop_top_percent"] and HEADER_RECOGNITION["form_types"].
    If return_raw_ocr is True, result includes raw OCR outputs for both calls.
    If crop_debug_out is set, writes the top-crop PNG to that path. If None and
    HEADER_RECOGNITION["crop_write_debug_image"] is True, writes under HEADER_RECOGNITION["crop_debug_dir"].
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    cfg = _cfg()
    pct = float(crop_top_percent if crop_top_percent is not None else cfg.get("crop_top_percent", 15.0))
    ft = form_types if form_types is not None else cfg.get("form_types", [])
    raw_ocr_id: dict[str, Any] | None = None
    raw_ocr_form_type: dict[str, Any] | None = None

    schema_path = _header_schema_path_sidea()
    if not schema_path.exists():
        raise FileNotFoundError(f"Header schema not found: {schema_path}")
    raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    id_roi = _find_named_roi(raw_schema, name="id")
    if id_roi is None:
        raise ValueError(f"Header schema missing ROI named 'id': {schema_path}")

    crop_dbg_id = _resolve_crop_debug_path_single(cfg, path, pct, crop_debug_out)
    crop_dbg_form = None
    if crop_dbg_id is not None:
        crop_dbg_form = crop_dbg_id.with_name(f"{crop_dbg_id.stem}_form_type{crop_dbg_id.suffix}")

    if verbose:
        print("[HEADER_RECOGNITION] form_type OCR call (top-crop)...")
    ft_text_ocr, raw_ocr_form_type = _crop_top_and_ocr_form_type_with_raw(
        path,
        crop_top_percent=pct,
        crop_debug_out=crop_dbg_form,
    )

    if verbose:
        print("[HEADER_RECOGNITION] ID OCR call (ROI='id')...")
    id_text_final, raw_ocr_id = _ocr_roi_with_raw(
        path,
        raw_schema=raw_schema,
        roi=id_roi,
        crop_debug_out=crop_dbg_id,
    )
    id_llm_out = _postprocess_header_fields_with_llm(
        {"id": id_text_final},
        roi_meta_by_name=_id_llm_meta_from_roi(id_roi),
    )
    id_text_final = str(id_llm_out.get("id", id_text_final) or "").strip()
    ft_llm_out = _postprocess_header_fields_with_llm(
        {"form_type": ft_text_ocr},
        roi_meta_by_name=_form_type_llm_meta(ft),
    )
    ft_text_final = str(ft_llm_out.get("form_type", ft_text_ocr) or "").strip()

    id_value = _extract_id_from_ocr_raw(raw_ocr_id, id_text_final)
    form_type_value = _extract_form_type_from_ocr_raw(raw_ocr_form_type, ft, ft_text_final)
    result = {
        "id": id_value,
        "form_type": form_type_value,
    }
    if raw_ocr_id is not None or raw_ocr_form_type is not None:
        # Backward-compatible key + explicit per-call payloads.
        result["raw_ocr"] = raw_ocr_id or {}
        result["raw_ocr_id"] = raw_ocr_id or {}
        result["raw_ocr_form_type"] = raw_ocr_form_type or {}
        result["ocr_text_id"] = id_text_final
        result["ocr_text_form_type"] = ft_text_final
    return result


def _serialize_raw_ocr_for_debug(raw_result: dict[str, Any] | None) -> dict[str, Any]:
    """Convert raw OCR output to a JSON-serializable object."""
    try:
        return json.loads(json.dumps(raw_result or {}, default=str))
    except Exception:
        return {"__repr__": repr(raw_result)}


def extract_id_and_form_type_batch(
    image_paths: list[str] | list[Path],
    *,
    crop_top_percent: float | None = None,
    form_types: list[str] | None = None,
    include_id: bool = True,
    verbose: bool = False,
    debug_out: dict | None = None,
    crop_debug_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Batch entrypoint:
    - ID from side-a schema ROI named "id" (when include_id=True)
    - form_type from top-crop OCR using HEADER_RECOGNITION["form_type_prompt_override"]

    Config defaults from HEADER_RECOGNITION. Returns one dict per image in order.
    If debug_out is provided (mutable dict), includes per-page OCR payloads for both calls.
    If crop_debug_dir is set, writes crop debug PNG(s) per page.
    """
    paths = [Path(p) for p in image_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
    cfg = _cfg()
    pct = float(crop_top_percent if crop_top_percent is not None else cfg.get("crop_top_percent", 15.0))
    ft = form_types if form_types is not None else cfg.get("form_types", [])
    raw_schema: dict[str, Any] | None = None
    id_roi: dict[str, Any] | None = None
    if include_id:
        schema_path = _header_schema_path_sidea()
        if not schema_path.exists():
            raise FileNotFoundError(f"Header schema not found: {schema_path}")
        raw_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        id_roi = _find_named_roi(raw_schema, name="id")
        if id_roi is None:
            raise ValueError(f"Header schema missing ROI named 'id': {schema_path}")
    if verbose:
        if include_id:
            print(
                f"[HEADER_RECOGNITION] ID/form OCR on {len(paths)} page(s): "
                "ID ROI + top-crop form_type."
            )
        else:
            print(
                f"[HEADER_RECOGNITION] form-type OCR on {len(paths)} page(s): "
                "top-crop only."
            )

    out_list: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    deferred_form_type_texts: dict[str, str] = {}
    deferred_form_type_raw: dict[str, dict[str, Any]] = {}
    deferred_id_values: dict[str, str | None] = {}
    deferred_id_texts: dict[str, str] = {}
    deferred_id_raw: dict[str, dict[str, Any]] = {}
    deferred_page_paths: dict[str, str] = {}

    iterator = enumerate(paths)
    if verbose:
        iterator = tqdm(
            iterator,
            total=len(paths),
            desc=("      ID/Form OCR (top crop, side-a)" if include_id else "      Form-type OCR (top crop, side-a)"),
            unit="page",
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            leave=False,
        )

    for i, p in iterator:
        dbg = None
        if _crop_debug_will_write(cfg, crop_debug_dir):
            d = Path(crop_debug_dir) if crop_debug_dir is not None else Path(
                cfg.get("crop_debug_dir") or (PATHS["output"] / "debug_images" / "id_crop")
            )
            d.mkdir(parents=True, exist_ok=True)
            dbg = d / f"{p.stem}_{i:04d}_crop_roi_id.png"
        dbg_form = dbg.with_name(f"{dbg.stem}_form_type{dbg.suffix}") if dbg is not None else None

        form_text_ocr, form_raw = _crop_top_and_ocr_form_type_with_raw(
            p,
            crop_top_percent=pct,
            crop_debug_out=dbg_form,
        )
        uid = f"p{i}"
        if include_id and isinstance(raw_schema, dict) and isinstance(id_roi, dict):
            id_text_final, id_raw = _ocr_roi_with_raw(
                p,
                raw_schema=raw_schema,
                roi=id_roi,
                crop_debug_out=dbg,
            )
            id_llm_out = _postprocess_header_fields_with_llm(
                {"id": id_text_final},
                roi_meta_by_name=_id_llm_meta_from_roi(id_roi),
            )
            id_text_final = str(id_llm_out.get("id", id_text_final) or "").strip()
            id_value = _extract_id_from_ocr_raw(id_raw, id_text_final)
            deferred_id_values[uid] = id_value
            deferred_id_texts[uid] = id_text_final
            deferred_id_raw[uid] = id_raw if isinstance(id_raw, dict) else {}
        else:
            deferred_id_values[uid] = None
            deferred_id_texts[uid] = ""
            deferred_id_raw[uid] = {}
        deferred_form_type_texts[uid] = " ".join(str(form_text_ocr or "").split())
        deferred_form_type_raw[uid] = form_raw if isinstance(form_raw, dict) else {}
        deferred_page_paths[uid] = str(p)

    ft_llm_processed = _postprocess_header_fields_with_llm(
        deferred_form_type_texts,
        roi_meta_by_name={uid: _form_type_llm_meta(ft)["form_type"] for uid in deferred_form_type_texts},
        use_tqdm=bool(verbose and _tqdm_enabled()),
        tqdm_desc="      Form-type LLM deferred",
    )

    for i, _p in enumerate(paths):
        uid = f"p{i}"
        form_text_final = str(ft_llm_processed.get(uid, deferred_form_type_texts.get(uid, "")) or "").strip()
        id_value = deferred_id_values.get(uid)
        form_type_value = _extract_form_type_from_ocr_raw(
            deferred_form_type_raw.get(uid),
            ft,
            form_text_final,
        )
        out_list.append({"id": id_value, "form_type": form_type_value})
        if debug_out is not None:
            debug_rows.append(
                {
                    "page_path": deferred_page_paths.get(uid, ""),
                    "ocr_text_id": deferred_id_texts.get(uid, ""),
                    "ocr_text_form_type_pre_llm": deferred_form_type_texts.get(uid, ""),
                    "ocr_text_form_type": form_text_final,
                    "raw_ocr_id": _serialize_raw_ocr_for_debug(deferred_id_raw.get(uid)),
                    "raw_ocr_form_type": _serialize_raw_ocr_for_debug(deferred_form_type_raw.get(uid)),
                    "parsed_result": {
                        "id": id_value,
                        "form_type": form_type_value,
                    },
                }
            )

    if debug_out is not None:
        debug_out["id_form"] = {
            "extraction_mode": "dual_ocr",
            "ocr_per_page": debug_rows,
            "parsed_results": [
                {"id": r.get("id"), "form_type": r.get("form_type")} for r in out_list
            ],
        }

    return out_list
