"""
Unified OCR module: image path in, text out.

Workflows:
- ocr_raw_vision_only: direct local vision OCR call.
- ocr_raw_paddle_then_vision: Paddle first (min-confidence gate), then local vision fallback.

Default ocr_raw() uses the configured default workflow.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from config import LLM, OCR_ENGINE

_LABEL_TO_SCORE = {
    "low": 0.25,
    "medium": 0.6,
    "high": 0.9,
}

_paddle_engine = None


def _vprint(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[ocr_engine] {message}")


def _cfg() -> dict[str, Any]:
    return OCR_ENGINE if isinstance(OCR_ENGINE, dict) else {}


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(_cfg().get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(_cfg().get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _cfg_str(key: str, default: str) -> str:
    v = _cfg().get(key, default)
    s = str(v).strip()
    return s or str(default)


def _cfg_bool(key: str, default: bool) -> bool:
    v = _cfg().get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return bool(default)


def _llm_host() -> str:
    return str(_cfg().get("host") or LLM.get("host", "127.0.0.1"))


def _llm_port() -> int:
    try:
        return int(_cfg().get("port") if _cfg().get("port") is not None else LLM.get("port", 11434))
    except (TypeError, ValueError):
        return 11434


def _llm_keep_alive() -> int:
    try:
        return int(_cfg().get("keep_alive") if _cfg().get("keep_alive") is not None else LLM.get("keep_alive", 0))
    except (TypeError, ValueError):
        return 0


def _workflow_default() -> str:
    return _cfg_str("workflow_default", "paddle_then_vision").lower()


def _normalize_jpeg_quality(quality: int) -> int:
    try:
        q = int(quality)
    except (TypeError, ValueError):
        q = 60
    return max(1, min(100, q))


def _encode_image(image_path: str | Path, *, quality: int) -> tuple[str, dict[str, Any]]:
    img = Image.open(image_path).convert("L")
    requested_quality = quality
    applied_quality = _normalize_jpeg_quality(quality)
    subsampling = 0 if applied_quality >= 95 else 2
    buf = BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=applied_quality,
        optimize=True,
        progressive=False,
        subsampling=subsampling,
    )
    data = buf.getvalue()
    meta = {
        "jpeg_quality_requested": int(requested_quality),
        "jpeg_quality_applied": int(applied_quality),
        "jpeg_subsampling": "4:4:4" if subsampling == 0 else "4:2:0",
        "jpeg_bytes": len(data),
    }
    return base64.b64encode(data).decode("ascii"), meta


def _json_from_text(text: str) -> dict[str, Any]:
    if not text:
        return {}
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    i0 = s.find("{")
    if i0 < 0:
        return {}
    try:
        obj, _ = dec.raw_decode(s[i0:])
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _normalize_label(label: Any) -> str:
    x = str(label or "").strip().lower()
    if x in {"low", "medium", "high"}:
        return x
    return "low"


def _score_to_label(score: float) -> str:
    """
    Derive low/medium/high from numeric score.
    """
    s = max(0.0, min(1.0, float(score)))
    if s >= 0.85:
        return "high"
    if s >= 0.5:
        return "medium"
    return "low"


def _coerce_confidence_score(parsed: dict[str, Any]) -> float:
    """
    Internal confidence coercion.
    VLM prompts no longer require confidence output; when missing, default to 1.0.
    Legacy confidence fields are still accepted when present.
    """
    has_score = "confidence_score" in parsed
    has_conf = "confidence" in parsed
    has_label = "confidence_label" in parsed
    score_raw = parsed.get("confidence_score") if has_score else None
    if score_raw is None and has_conf:
        score_raw = parsed.get("confidence")
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        if has_label:
            label = _normalize_label(parsed.get("confidence_label"))
            score = _LABEL_TO_SCORE.get(label, 1.0)
        else:
            score = 1.0
    return max(0.0, min(1.0, float(score)))


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value or "").strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _has_uncertainty_signal(parsed: dict[str, Any]) -> bool:
    se = parsed.get("self_evaluation")
    if not isinstance(se, dict):
        return False
    for key in ("legibility_issues", "ambiguities", "missing_regions"):
        v = se.get(key)
        if isinstance(v, list):
            if any(str(x).strip() for x in v):
                return True
        elif isinstance(v, str) and v.strip():
            return True
    reason = str(se.get("reason_for_confidence", "") or "").lower()
    uncertainty_terms = (
        "unclear",
        "illegible",
        "ambiguous",
        "cropped",
        "cut off",
        "partial",
        "blur",
        "faint",
        "uncertain",
    )
    return any(tok in reason for tok in uncertainty_terms)


def _is_confident(parsed: dict[str, Any], *, min_score: float) -> bool:
    if not isinstance(parsed, dict) or not parsed:
        return False
    score = _coerce_confidence_score(parsed)
    return score >= float(min_score)


def _is_paddle_confident(parsed: dict[str, Any]) -> bool:
    try:
        score = float(parsed.get("confidence_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return score > _cfg_float("paddle_min_confidence_for_accept", 0.90)


def _build_prompt() -> str:
    return (
        "You are an OCR evaluator. Extract text from the image.\n"
        "Return ONE valid JSON object only, no markdown and no extra text.\n"
        "Required schema:\n"
        "{\n"
        '  "detected_text": "string"\n'
        "}\n"
        "Rules:\n"
        "- Keep detected_text concise and faithful to visible text only.\n"
        "- Include unclear but visible text as best-effort in detected_text; do not omit it just because it is uncertain.\n"
        "- Mark uncertain spans inline with [[? ... ]] when needed.\n"
        "- Do not invent text that has no visible evidence in the image.\n"
        "Task:\n"
        "Extract all visible text (including unclear best-effort spans)."
    )


def default_text_prompt() -> str:
    """Public accessor for the base OCR text prompt template."""
    return _build_prompt()


def _resolve_prompt(
    *,
    prompt_override: str | None = None,
) -> tuple[str, str]:
    """
    Resolve the prompt text and effective mode.

    Precedence:
    1) explicit prompt_override (if non-empty and OCR prompts enabled)
    2) default built-in prompt
    """
    if not bool(_cfg().get("use_ocr_prompts", True)):
        return _build_prompt(), "default"

    override = str(prompt_override or "").strip()
    if override:
        return override, "custom"

    return _build_prompt(), "default"


def _local_vision_call(
    *,
    model: str,
    image_b64: str,
    prompt: str,
    timeout: int,
    host: str,
    port: int,
    keep_alive: int,
    stream: bool,
    extra_params: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": bool(stream),
        "keep_alive": int(keep_alive),
        "images": [image_b64],
    }
    if extra_params:
        payload.update(extra_params)

    url = f"http://{host}:{int(port)}/api/generate"
    resp = requests.post(url, json=payload, timeout=timeout)
    if resp.status_code >= 400:
        detail = (resp.text or "").strip()
        if len(detail) > 1500:
            detail = detail[:1500] + "...(truncated)"
        raise RuntimeError(f"HTTP {resp.status_code} error from {url}: {detail}")
    body = resp.json()
    text = str(body.get("response", "") or "")
    return text, body


def _apply_paddle_env_overrides() -> None:
    raw = _cfg().get("paddle_env_overrides", {})
    if not isinstance(raw, dict):
        return
    for env_key, env_val in raw.items():
        os.environ.setdefault(str(env_key), str(env_val))


def _get_paddle_engine():
    global _paddle_engine
    if _paddle_engine is not None:
        return _paddle_engine

    _apply_paddle_env_overrides()
    try:
        import logging
        from paddleocr import PaddleOCR, logger as paddleocr_logger
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PaddleOCR unavailable. Install with: pip install paddlepaddle paddleocr"
        ) from exc

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            paddleocr_logger.setLevel(logging.CRITICAL)
            _paddle_engine = PaddleOCR(
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
    return _paddle_engine


def _run_paddle_stage(
    path: Path,
    *,
    verbose: bool,
    force_failure: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    stage: dict[str, Any] = {
        "stage_index": 0,
        "model": _cfg_str("paddle_model_name", "paddle-ppocrv5"),
        "elapsed_sec": None,
        "raw_response": None,
        "response_text": "",
        "parsed": {},
        "accepted": False,
        "error": None,
        "image_payload": {},
    }

    t0 = time.time()
    paddle_full_out: dict[str, Any] | None = None
    try:
        if force_failure:
            raise RuntimeError("Forced Paddle failure for testing.")
        _vprint(verbose, f"stage 0 begin model={stage['model']} (paddle)")
        engine = _get_paddle_engine()
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = engine.predict(input=str(path.resolve()))

        rec_texts: list[str] = []
        rec_scores: list[float] = []
        raw_items: list[dict[str, Any]] = []
        for res in outputs or []:
            j = getattr(res, "json", None)
            if isinstance(j, dict):
                raw_items.append(j)
                block = j.get("res") or {}
                texts = block.get("rec_texts", [])
                scores = block.get("rec_scores", [])
                if isinstance(texts, list):
                    rec_texts.extend(str(t) for t in texts if str(t).strip())
                elif texts:
                    rec_texts.append(str(texts))
                if isinstance(scores, list):
                    for s in scores:
                        try:
                            rec_scores.append(float(s))
                        except (TypeError, ValueError):
                            pass

        text = " ".join(rec_texts).strip()
        min_score = min(rec_scores) if rec_scores else 0.0
        mean_score = (sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0
        label = "high" if min_score > 0.90 else ("medium" if min_score >= 0.60 else "low")
        needs_review = min_score <= _cfg_float("paddle_min_confidence_for_accept", 0.90)

        parsed = {
            "detected_text": text,
            "confidence_label": label,
            "confidence_score": float(min_score),
            "needs_human_review": bool(needs_review),
            "self_evaluation": {
                "engine": "paddle",
                "line_count": len(rec_texts),
                "min_rec_score": min_score,
                "mean_rec_score": mean_score,
                "reason_for_confidence": (
                    f"paddle min_rec_score={min_score:.4f}, mean_rec_score={mean_score:.4f}"
                ),
            },
        }
        paddle_full_out = {
            "detected_text": text,
            "confidence_label": label,
            "confidence_score": float(min_score),
            "needs_human_review": bool(needs_review),
            "self_evaluation": parsed["self_evaluation"],
            "rec_texts": rec_texts,
            "rec_scores": rec_scores,
            "raw_response": raw_items,
        }

        stage["parsed"] = parsed
        stage["raw_response"] = raw_items
        stage["response_text"] = text
        stage["accepted"] = _is_paddle_confident(parsed)
        _vprint(
            verbose,
            f"stage 0 complete accepted={stage['accepted']} confidence={parsed.get('confidence_score')}",
        )
    except Exception as exc:  # pragma: no cover
        stage["error"] = str(exc)
        _vprint(verbose, f"stage 0 error={exc}")
    finally:
        stage["elapsed_sec"] = round(time.time() - t0, 4)
        _vprint(verbose, f"stage 0 elapsed={stage['elapsed_sec']}s")

    return stage, paddle_full_out


def _run_vision_stage(
    path: Path,
    *,
    timeout: int,
    min_conf: float,
    verbose: bool,
    stage_index: int,
    prompt_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _cfg_str("model", "qwen2.5vl")
    host = _llm_host()
    port = _llm_port()
    keep_alive = _llm_keep_alive()
    stream = _cfg_bool("stream", False)
    extra_params_raw = _cfg().get("extra_params", {})
    extra_params = extra_params_raw if isinstance(extra_params_raw, dict) else None
    max_call_ms = max(0, _cfg_int("vlm_max_call_ms", 0))
    slow_call_retries = max(0, _cfg_int("vlm_slow_call_retries", 0))

    image_encode_meta: dict[str, Any] = {}
    stage: dict[str, Any] = {
        "stage_index": stage_index,
        "model": model,
        "elapsed_sec": None,
        "raw_response": None,
        "response_text": "",
        "parsed": {},
        "accepted": False,
        "error": None,
        "image_payload": {},
        "vlm_call_ms": None,
        "vlm_attempt": None,
        "vlm_slow_dropped_attempts": [],
    }

    prompt_text, prompt_source = _resolve_prompt(
        prompt_override=prompt_override,
    )
    stage["prompt_source"] = prompt_source

    _vprint(verbose, f"stage {stage_index} begin model={model} (vision)")
    t0 = time.time()
    try:
        image_b64, image_encode_meta = _encode_image(path, quality=_cfg_int("jpeg_quality", 60))
        stage["image_payload"] = dict(image_encode_meta)
        _vprint(
            verbose,
            "vision image encoded "
            f"quality_requested={image_encode_meta.get('jpeg_quality_requested')} "
            f"quality_applied={image_encode_meta.get('jpeg_quality_applied')} "
            f"subsampling={image_encode_meta.get('jpeg_subsampling')} "
            f"bytes={image_encode_meta.get('jpeg_bytes')}",
        )
        response_text = ""
        body: dict[str, Any] = {}
        attempts = 1 + slow_call_retries if max_call_ms > 0 else 1
        for attempt in range(1, attempts + 1):
            t_call = time.perf_counter()
            response_text, body = _local_vision_call(
                model=model,
                image_b64=image_b64,
                prompt=prompt_text,
                timeout=timeout,
                host=host,
                port=port,
                keep_alive=keep_alive,
                stream=stream,
                extra_params=extra_params,
            )
            call_ms = (time.perf_counter() - t_call) * 1000.0
            stage["vlm_call_ms"] = round(call_ms, 2)
            stage["vlm_attempt"] = attempt
            if max_call_ms > 0 and call_ms > float(max_call_ms):
                stage["vlm_slow_dropped_attempts"].append(
                    {
                        "attempt": attempt,
                        "elapsed_ms": round(call_ms, 2),
                        "max_ms": int(max_call_ms),
                    }
                )
                if attempt < attempts:
                    _vprint(
                        verbose,
                        f"stage {stage_index} slow VLM call dropped "
                        f"(attempt={attempt}/{attempts}, elapsed_ms={call_ms:.2f}, max_ms={max_call_ms}); retrying",
                    )
                    continue
                raise TimeoutError(
                    f"VLM call exceeded max duration: {call_ms:.2f}ms > {max_call_ms}ms "
                    f"after {attempts} attempt(s)"
                )
            break
        parsed = _json_from_text(response_text)
        stage["raw_response"] = body
        stage["response_text"] = response_text
        stage["parsed"] = parsed
        stage["accepted"] = _is_confident(parsed, min_score=min_conf)
        _vprint(
            verbose,
            f"stage {stage_index} complete accepted={stage['accepted']} "
            f"confidence_label={parsed.get('confidence_label')} confidence_score={parsed.get('confidence_score')}",
        )
    except Exception as exc:
        stage["error"] = str(exc)
        _vprint(verbose, f"stage {stage_index} error={exc}")
    finally:
        stage["elapsed_sec"] = round(time.time() - t0, 4)
        _vprint(verbose, f"stage {stage_index} elapsed={stage['elapsed_sec']}s")

    return stage, image_encode_meta


def _build_output(
    *,
    chosen: dict[str, Any],
    stages: list[dict[str, Any]],
    min_conf: float,
    image_encode_meta: dict[str, Any],
    paddle_full_out: dict[str, Any] | None,
) -> dict[str, Any]:
    parsed = dict(chosen.get("parsed") or {})
    score = _coerce_confidence_score(parsed)
    label = _score_to_label(score)

    last_stage = stages[-1] if stages else {}
    needs_review = not (score >= float(min_conf))
    if last_stage.get("error"):
        needs_review = True

    detected_text = str(parsed.get("detected_text", "") or "").strip()
    if not detected_text and not chosen.get("error"):
        detected_text = str(chosen.get("response_text", "") or "").strip()

    selected_model = str(chosen.get("model", _cfg_str("model", "qwen2.5vl")))
    selected_stage_index = int(chosen.get("stage_index", 0))

    if (
        selected_model == _cfg_str("paddle_model_name", "paddle-ppocrv5")
        and isinstance(paddle_full_out, dict)
    ):
        rec_scores_raw = list(paddle_full_out.get("rec_scores") or [])
        rec_scores: list[float] = []
        for x in rec_scores_raw:
            try:
                rec_scores.append(float(x))
            except (TypeError, ValueError):
                pass
        rec_texts = list(paddle_full_out.get("rec_texts") or [])
        min_rec_score = float(paddle_full_out.get("confidence_score", score))
        mean_rec_score = float(sum(rec_scores) / len(rec_scores)) if rec_scores else min_rec_score
        raw_response = paddle_full_out.get("raw_response")
    else:
        rec_scores = [score]
        rec_texts = [detected_text] if detected_text else []
        min_rec_score = score
        mean_rec_score = score
        raw_response = chosen.get("raw_response")

    return {
        "detected_text": detected_text,
        "confidence_label": label,
        "confidence_score": score,
        "needs_human_review": bool(needs_review),
        "selected_model": selected_model,
        "selected_stage_index": selected_stage_index,
        "min_rec_score": min_rec_score,
        "mean_rec_score": mean_rec_score,
        "rec_scores": rec_scores,
        "rec_texts": rec_texts,
        "raw_response": raw_response,
        "self_evaluation": parsed.get("self_evaluation", {}),
        "image_payload": image_encode_meta,
        "stages": stages,
    }


def ocr_raw_vision_only(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    call_timeout = int(timeout or _cfg_int("call_timeout_sec", 90))
    min_conf = _cfg_float("min_confidence_for_accept", 0.82)

    _vprint(
        verbose,
        "start OCR vision_only "
        f"image={Path(path).resolve()} model={_cfg_str('model', 'qwen2.5vl')} "
        f"host={_llm_host()} port={_llm_port()} keep_alive={_llm_keep_alive()}",
    )

    stage, image_meta = _run_vision_stage(
        path,
        timeout=call_timeout,
        min_conf=min_conf,
        verbose=verbose,
        stage_index=0,
        prompt_override=prompt_override,
    )
    out = _build_output(
        chosen=stage,
        stages=[stage],
        min_conf=min_conf,
        image_encode_meta=image_meta,
        paddle_full_out=None,
    )
    _vprint(
        verbose,
        f"final selected_model={out.get('selected_model')} selected_stage_index={out.get('selected_stage_index')} "
        f"needs_human_review={out.get('needs_human_review')}",
    )
    return out


def ocr_raw_paddle_then_vision(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    force_paddle_failure: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    call_timeout = int(timeout or _cfg_int("call_timeout_sec", 90))
    min_conf = _cfg_float("min_confidence_for_accept", 0.82)

    _vprint(verbose, f"start OCR paddle_then_vision image={Path(path).resolve()}")

    stages: list[dict[str, Any]] = []
    image_meta: dict[str, Any] = {}

    paddle_stage, paddle_full = _run_paddle_stage(
        path,
        verbose=verbose,
        force_failure=force_paddle_failure,
    )
    stages.append(paddle_stage)

    if paddle_stage.get("accepted"):
        chosen = paddle_stage
        _vprint(verbose, "paddle accepted; skipping vision fallback")
    else:
        vision_stage, image_meta = _run_vision_stage(
            path,
            timeout=call_timeout,
            min_conf=min_conf,
            verbose=verbose,
            stage_index=1,
            prompt_override=prompt_override,
        )
        stages.append(vision_stage)
        if vision_stage.get("error") and not vision_stage.get("parsed"):
            chosen = paddle_stage
        else:
            chosen = vision_stage

    out = _build_output(
        chosen=chosen,
        stages=stages,
        min_conf=min_conf,
        image_encode_meta=image_meta,
        paddle_full_out=paddle_full,
    )
    _vprint(
        verbose,
        f"final selected_model={out.get('selected_model')} selected_stage_index={out.get('selected_stage_index')} "
        f"needs_human_review={out.get('needs_human_review')}",
    )
    return out


def ocr_raw(
    image_path: str | Path,
    *,
    timeout: int | None = None,
    verbose: bool = False,
    force_paddle_failure: bool = False,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    workflow = _workflow_default()
    if workflow in {"vision_only", "vision"}:
        return ocr_raw_vision_only(
            image_path,
            timeout=timeout,
            verbose=verbose,
            prompt_override=prompt_override,
        )
    return ocr_raw_paddle_then_vision(
        image_path,
        timeout=timeout,
        verbose=verbose,
        force_paddle_failure=force_paddle_failure,
        prompt_override=prompt_override,
    )


def ocr(image_path: str | Path) -> str:
    out = ocr_raw(image_path)
    return str(out.get("detected_text", "") or "").strip()


def ocr_confidence_stats(raw_result: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw_result or {}
    score = _coerce_confidence_score(raw if isinstance(raw, dict) else {})
    label = _score_to_label(score)

    rec_scores_raw = raw.get("rec_scores")
    rec_scores: list[float] = []
    if isinstance(rec_scores_raw, list):
        for s in rec_scores_raw:
            try:
                rec_scores.append(float(s))
            except (TypeError, ValueError):
                pass

    if rec_scores:
        min_rec = float(min(rec_scores))
        mean_rec = float(sum(rec_scores) / len(rec_scores))
    else:
        min_rec = score
        mean_rec = score
        rec_scores = [score]

    rec_texts_raw = raw.get("rec_texts")
    rec_texts: list[str] = []
    if isinstance(rec_texts_raw, list):
        rec_texts = [str(t) for t in rec_texts_raw if str(t).strip()]
    if not rec_texts:
        text = str(raw.get("detected_text", "") or "")
        rec_texts = [text] if text else []

    return {
        "min_rec_score": min_rec,
        "mean_rec_score": mean_rec,
        "rec_scores": rec_scores,
        "rec_texts": rec_texts,
        "confidence_label": label,
        "needs_human_review": bool(score < _cfg_float("min_confidence_for_accept", 0.82)),
        "selected_model": raw.get("selected_model"),
        "selected_stage_index": raw.get("selected_stage_index"),
    }
