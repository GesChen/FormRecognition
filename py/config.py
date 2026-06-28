"""Central configuration for EVMS pipeline modules."""

from pathlib import Path

# Root directory containing data/, output/, py/, docs/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RELEASE = "2026"
DATA_DIR = PROJECT_ROOT / "data"
DATA_XLSX_ROOT = DATA_DIR / "xlsx"
DATA_TEMPLATES_ROOT = DATA_DIR / "templates" / DATA_RELEASE
DATA_ROI_SCHEMAS_ROOT = DATA_DIR / "roi_schemas" / DATA_RELEASE
DATA_XLSX_MAPPINGS_ROOT = DATA_XLSX_ROOT / "mappings" / DATA_RELEASE
DATA_XLSX_WORKBOOK_TEMPLATES_ROOT = DATA_XLSX_ROOT / "workbook_templates" / DATA_RELEASE

# Canonical project paths (absolute).
PATHS = {
    "data": DATA_DIR,  # Input assets root.
    "xlsx_root": DATA_XLSX_ROOT,  # XLSX-related assets root.
    "templates_root": DATA_TEMPLATES_ROOT,  # Active template root for current data release.
    "roi_schemas_root": DATA_ROI_SCHEMAS_ROOT,  # Active ROI schema root for current data release.
    "xlsx_mappings_root": DATA_XLSX_MAPPINGS_ROOT,  # Active XLSX mapping root for current data release.
    "xlsx_workbook_templates_root": DATA_XLSX_WORKBOOK_TEMPLATES_ROOT,  # Active workbook templates root.
    "output": PROJECT_ROOT / "output",  # Pipeline outputs (json/xlsx/debug).
    "docs": PROJECT_ROOT / "docs",  # Documentation.
    "cache": PROJECT_ROOT / "output" / "cache",  # Intermediate cache root.
    "cache_normalized": PROJECT_ROOT / "output" / "cache" / "normalized",  # Normalized page cache.
}

# ---------------------------------------------------------------------------
# API / MODEL CONNECTIONS (top-level, high-priority)
# ---------------------------------------------------------------------------

LLM = {
    "host": "192.168.182.1",  # Model server host/IP (shared by LLM + VLM OCR unless overridden).
    "port": 11434,  # Model server port.
    # "model": "qwen3.5:9b",  # Default text model for llm_client/testing scripts.
    "model": "llama3.2:latest",  # Default text model for llm_client/testing scripts.
    "keep_alive": 20,  # Seconds to keep model loaded between calls (0 disables).
}

LLM_POSTPROCESS = {
    "enabled": True,  # Global toggle for optional LLM cleanup passes; does not disable required OCR/header LLM steps.
}

ROI_PROMPT_CREATOR = {
    "model": "qwen3.5:9b",  # Dedicated text model for ROI prompt generation.
}

ROI_AUTO_DETECT = {
    "use_openai": True,  # True => OpenAI Responses API, False => local Ollama model.
    "openai_model": "gpt-5.5",  # Optional explicit OpenAI model override (e.g. "gpt-5.5"). None => use profile.
    "openai_reasoning": True,  # True => omit reasoning arg (model default). False => send reasoning={"effort":"none"}.
    "local_model": "qwen3.5:9b",  # Local model for ROI auto-detector when use_openai=False.
    "local_reasoning": False,  # Enable local model reasoning/thinking mode when supported.
}

XLSX_MAPPING_AUTO_GENERATE = {
    "use_openai": True,  # True => OpenAI Responses API, False => local Ollama model.
    "openai_model": "gpt-5.5",  # Optional explicit OpenAI model override. None/"" => use profile.
    "openai_model_profile": "max_quality",  # Used only when openai_model is empty.
    "openai_reasoning": True,  # True => omit reasoning arg (model default). False => reasoning={"effort":"none"}.
    "max_output_tokens": 24000,
    "local_model": "qwen3.5:9b",
    "local_reasoning": False,
    "local_timeout_sec": 600,
    "default_start_row": 4,
    "xlsx_context_rows": 6,
    "max_validation_attempts": 2,
    "example_form": "6pre",
    "example_dir": DATA_XLSX_ROOT / "mappings" / "example" / "6pre",
    "example_mapping_path": DATA_XLSX_ROOT / "mappings" / "example" / "6pre" / "mapping.json",
    "example_paddle_a_path": DATA_XLSX_ROOT / "mappings" / "example" / "6pre" / "paddle_a.json",
    "example_paddle_b_path": DATA_XLSX_ROOT / "mappings" / "example" / "6pre" / "paddle_b.json",
    "example_template_workbook": DATA_XLSX_ROOT / "mappings" / "example" / "6pre" / "template.xlsx",
}

TEXT_ROI_LLM = {
    "enabled": True,  # LLM post-step for non-header text ROIs (after default OCR extraction).
    # "model": "qwen3.5:9b",  # None => use LLM["model"].
    "model": "llama3.2:latest",  # None => use LLM["model"].
    "timeout_sec": 120,  # Timeout per ROI post-step call.
    "max_reruns": 3,  # Retry count when model output does not conform to {"detected_text": string|null}.
    "extra_params": {"think": False},  # Prefer concise deterministic outputs.
    "paddle_vlm_fusion_enabled": True,  # When Paddle sidecar text exists, give both Paddle + VLM guesses to this normalizer.
}

POST_NORMALIZE = {
    "enabled": True,  # Batch LLM normalization pass for ROI fields marked post_normalize=true.
    "model": "qwen3.5:9b",  # Dedicated model for batch post-normalization.
    "timeout_sec": 180,
    "max_retries": 3,  # Retry when output is not a JSON object mapping observed values.
    "extra_params": {"think": False, "options": {"temperature": 0}},
}

# ---------------------------------------------------------------------------
# CORE PIPELINE BEHAVIOR (high-priority)
# ---------------------------------------------------------------------------

PDF_TO_IMAGES = {
    "cache_root": PATHS["cache"],  # Destination for rendered page images.
    "dpi": 150,  # Page render DPI.
    "fmt": "png",  # Output image format.
}

OCR_ENGINE = {
    # Main OCR workflow mode.
    # - "paddle_then_vision": Paddle first, vision fallback.
    # - "paddle_only": Paddle only, no vision fallback.
    # - "vision_only": vision only.
    # - "vision_with_paddle_confidence": vision text with Paddle confidence.
    # - "paddle_vlm_fusion": vision text + Paddle sidecar for downstream LLM fusion.
    "workflow_default": "paddle_vlm_fusion",

    # Vision OCR model name.
    "model": "glm-ocr:latest",
    # "model": "ministral-3:8b",

    # Paddle stage settings.
    "paddle_model_name": "paddle-ppocrv5",
    "paddle_min_confidence_for_accept": 0.90,  # If Paddle min score <= this, use vision fallback.
    "include_paddle_confidence_raw": True,  # Include Paddle raw_response in vision-with-Paddle-confidence debug output.
    "paddle_env_overrides": {
        "FLAGS_use_pir": "0",
        "FLAGS_use_mkldnn": "0",
        "FLAGS_enable_pir_api": "0",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "1",
        "DISABLE_AUTO_LOGGING_CONFIG": "1",
    },

    # Vision call behavior.
    "min_confidence_for_accept": 0.82,  # Internal acceptance threshold for downstream review flags.
    "call_timeout_sec": 90,  # Timeout per vision call.
    "vlm_max_call_ms": 2000,  # 0 disables slow-call drop; otherwise drop/retry VLM calls slower than this many ms.
    "vlm_slow_call_retries": 2,  # Extra retries after a slow-call drop.
    "vlm_repeat_json_stop_enabled": True,  # Stop streamed VLM output if it starts emitting repeated JSON answers.
    "vlm_repeat_json_min_blocks": 2,  # Number of completed JSON blocks that indicates a JSON-answer loop.
    "vlm_repeat_tail_stop_enabled": True,  # Fallback: stop malformed streams with exact repeated text tails.
    "vlm_repeat_tail_min_unit_chars": 24,  # Smallest repeated suffix unit considered a loop.
    "vlm_repeat_tail_max_unit_chars": 240,  # Largest repeated suffix unit considered a loop.
    "vlm_repeat_tail_repeats": 3,  # Required exact suffix repetitions before cutting off.
    "vlm_repeat_tail_min_total_chars": 80,  # Minimum accumulated stream size before tail-loop detection.
    "vlm_roi_prompt_template": '请按下列JSON格式输 出图中信息: {"{query}":""}',  # Per-ROI VLM prompt template; {query} is inserted exactly.
    "jpeg_quality": 20,  # JPEG quality used for VLM image payload.
    "stream": True,  # Stream VLM responses so completed JSON can be detected mid-call.
    "extra_params": 
    {
        "options": {"temperature": 0}
     },  # Extra model-server payload params.

}

HEADER_RECOGNITION = {
    "model": "qwen3.5:9b",  # Dedicated text model for form-type inference.
    "crop_top_percent": 15.0,  # Top portion of page used for ID/form_type OCR.
    "crop_write_debug_image": False,  # Write top-crop debug PNGs.
    "crop_debug_dir": PATHS["output"] / "debug_images" / "id_crop",  # Debug output dir for top-crops.
    "form_types": ["6pre", "7pre", "8pre", "hpre", "6post", "7post", "8post", "hpost"],  # Allowed form_type values.
    "normalize_id_output": False,  # Normalize ID OCR output to canonical 7-digit+A/B pattern.
    # Optional OCR workflow override for header ID/form-type extraction.
    # None/"" => use OCR_ENGINE["workflow_default"].
    # Same choices/aliases as OCR_ENGINE: vision_only, paddle_only,
    # paddle_then_vision, vision_with_paddle_confidence, paddle_vlm_fusion.
    "ocr_workflow_override": "paddle_only",
    "form_type_prompt_override":
"""
You normalize OCR header text into one canonical form_type value.
You are NOT reading an image. You only receive OCR-extracted text.
Return ONE valid JSON object only: {"detected_text":"6pre|6post|7pre|7post|8pre|8post|hpre|hpost|null"}.
No prose, no markdown, no code fences.

Direct mapping:
- If text indicates 6th Grade + Pre-Assessment, output "6pre".
- If text indicates 6th Grade + Post-Assessment, output "6post".
- If text indicates 7th Grade + Pre-Assessment, output "7pre".
- If text indicates 7th Grade + Post-Assessment, output "7post".
- If text indicates 8th Grade + Pre-Assessment, output "8pre".
- If text indicates 8th Grade + Post-Assessment, output "8post".
- If text indicates High School + Pre-Assessment, output "hpre".
- If text indicates High School + Post-Assessment, output "hpost".

Accept common OCR variants:
- Pre-Assessment, Pre Assessment, PRE
- Post-Assessment, Post Assessment, POST

Rules:
- Use only evidence explicitly present in OCR text.
- Do not infer grade from ID numbers.
- If required grade/timing evidence is missing or ambiguous, output null.
""",
}

# Backward-compatible alias for older tools/scripts. Prefer HEADER_RECOGNITION.
ID_FORM_LLM = HEADER_RECOGNITION

PDF_RECOGNITION = {
    "output_dir": PATHS["output"] / "recognition",  # Final recognition JSON output directory.
    "schema_key_sidea": "schema_sidea",  # key name for side-a schema in schema map.
    "schema_key_sideb": "schema_sideb",  # key name for side-b schema in schema map.
    "schema_dir": PATHS["roi_schemas_root"],  # ROI schema directory (release-scoped).
    "mc_roi_names": [],  # Empty => infer from schema (all names matching \d+[a-h]).
    "debug_output": True,  # Write pipeline debug JSON snapshots.
    "sort_data_by_roi_name": True,  # Sort ROI rows numerically by name.
    "page_count": {
        "required_multiple": 2,  # Processed PDF page count must divide evenly by this packet size.
    },
    # Keep current dual-channel behavior by default:
    # - text ROI OCR uses homography-only normalized images
    # - MCQ recognition uses postprocessed normalized images
    # Set True to route postprocessed ("mcq") images into text ROI OCR too.
    "use_processed_images_for_text_roi": False,
    "human_review": {
        "enabled": True,  # Enable human-review block generation.
        "text_review_use_needs_human_review": True,  # Prefer OCR engine review flag for text ROIs.
        "id_review_use_needs_human_review": True,  # Prefer OCR engine review flag for ID.
        "min_text_ocr_confidence": 0.85,  # Fallback threshold when review flag is absent.
        "min_id_ocr_confidence": 0.85,  # Fallback threshold for ID review.
        "pending_id_review": True,  # Include ID rows in pending review when needed.
        "pending_all_ids": True,  # Force all IDs into pending review queue.
    "pending_mcq_empty_review": True,  # Review empty MCQ answers.
    },
    # Normalization worker count.
    # Set to 1 for direct serial execution.
    "normalize_workers": 4,
}

# ---------------------------------------------------------------------------
# DETAILED BEHAVIOR / TUNING
# ---------------------------------------------------------------------------

IMAGE_NORMALIZE = {
    "cache_root": PATHS["cache_normalized"],  # Normalized image cache root.
    "perspective_correct": False,  # PDFs are already flat; keep disabled by default.
    "deskew": False,  # Avoid rotating already-upright pages.
    "binarize": True,  # Output binary images by default.
    "template_registration": True,  # Align pages to template using ORB + RANSAC.
    "template_registration_templates": {},  # Optional explicit key -> template path overrides.
    "template_registration_templates_dir": PATHS["templates_root"],  # Directory containing <form_type>_<side>.png templates.
    "template_registration_default_key": None,  # Fallback template key when form/side is unknown.
    "template_registration_orb_nfeatures": 5000,  # ORB feature cap used during template registration.
    "template_registration_ratio_test": 0.75,  # Lowe ratio threshold for knn match filtering.
    "template_registration_min_good_matches": 10,  # Minimum good matches required to solve homography.
    "template_registration_ransac_reproj_threshold": 5.0,  # RANSAC reprojection threshold for homography.
    "template_registration_min_inliers": 8,  # Minimum inlier correspondences after RANSAC to accept homography.
    "binarize_block_size": 31,  # Adaptive threshold block size.
    "clahe_clip_limit": 2.0,  # CLAHE clip limit.
    "clahe_tile_grid_size": (8, 8),  # CLAHE tile size.
    "denoise_h": 10,  # NLM denoise strength.
    "denoise_template_window_size": 7,  # NLM template window size.
    "denoise_search_window_size": 21,  # NLM search window size.
    "adaptive_threshold_c": 8,  # Adaptive threshold bias constant.
    "contrast_low_percentile": 1.0,  # Lower percentile for contrast stretch.
    "contrast_high_percentile": 99.0,  # Upper percentile for contrast stretch.
    "contrast_power": 1.0,  # >1 darkens midtones, <1 lightens midtones.
}

ROI_PAGE_RECOGNITION = {
    "text_roi_expand_px": 0,  # Expand each text ROI crop by N pixels before OCR.
    # Remove long form rules from text crops before the shared VLM/Paddle OCR call.
    "text_roi_horizontal_line_suppression_enabled": True,
    "text_roi_horizontal_line_min_len": 80,  # Minimum detected rule length in crop pixels.
    "text_roi_horizontal_line_thickness": 2,  # Vertical mask dilation used to cover the full rule.
    "text_roi_horizontal_line_inpaint_radius": 3,  # OpenCV Telea inpaint radius in pixels.
    "mcq_subroi_expand_px": 5,  # Expand each MCQ sub-ROI by N pixels.
    "mcq_eps": 0.05,  # Ignore tiny residual darkness as noise.
    "mcq_print_suppression_k": 5.0,  # Higher = stronger suppression of pre-printed dark pixels.
    "mcq_power": 3,  # Exponent applied to residual darkness.
    "mcq_blur_sigma": 5,  # Gaussian blur sigma before differencing.
    "mcq_min_score_to_accept": 0.0,  # Minimum best-choice score required to emit an MCQ answer.
    "mcq_return_raw_darkness_debug": False,  # Include per-choice raw scores in output.
    "mcq_write_debug_images": False,  # Write per-question debug images.
    "mcq_debug_images_dir": PATHS["output"] / "debug_images" / "mcq",  # MCQ debug image root.
    # Regex-gated OCR redo loop for text ROIs (applied after deferred LLM normalization).
    # If a text ROI defines `ocr_output_regex` in schema metadata and normalized text does
    # not match, each step below is attempted in order until match or steps are exhausted.
    "ocr_regex_check_enabled": False,  # Master toggle for checking/enforcing schema `ocr_output_regex`.
    "ocr_regex_retry_enabled": True,
    # After all regex retry steps are exhausted, clear values that still do not match
    # their schema-provided `ocr_output_regex`. This is generic schema enforcement,
    # not field-specific cleanup; invalid values remain visible in debug/review traces.
    "ocr_regex_retry_clear_on_final_mismatch": True,
    # Ordered, editable list of retry method names. Keep this short to bound runtime.
    "ocr_regex_retry_steps": [
        "pad_8px",
        "clahe_light",
        "fsrcnn_x2",
        "pad_8px_clahe_light",
    ],
    # Small set of tunables used by retry methods.
    "ocr_regex_retry_pad_px": 8,
    "ocr_regex_retry_clahe_clip": 2.0,
    "ocr_regex_retry_clahe_tile": 8,
    # FSRCNN super-resolution retry step:
    # - If model path is empty or unavailable, the step is skipped safely.
    # - Typical model file is OpenCV FSRCNN_x2.pb.
    "ocr_regex_retry_fsrcnn_model_path": "/home/ges/Documents/evms_3/data/models/FSRCNN-small_x2.pb",
    "ocr_regex_retry_fsrcnn_scale": 2,
    # If FSRCNN runtime/model fails (e.g., unsupported OpenCV ops), fall back
    # to standard resize upscaling so the step is still useful.
    "ocr_regex_retry_fsrcnn_fallback_resize": True,
}

XLSX_DATA_ENTRY = {
    "mapping_dir": PATHS["xlsx_mappings_root"],  # Directory with <form_type>.json mapping files.
    "output_dir": PATHS["output"] / "xlsx",  # Final XLSX output directory.
    "staging_dir": PATHS["cache"] / "xlsx_staging",  # Temporary filled workbook staging area.
    "force_text_cells": True,  # Write mapped values as explicit Excel text; disable to retain inferred types.
    "include_original_sheet": False,  # Add pre-normalization "(Original)" sheets when items_original is available.
    "confidence_heatmap": {
        "enabled": True,  # Shade text ROI cells by OCR confidence.
        "max_red": "F4B6B6",  # Fill at confidence=0; kept light enough for readable black text.
    },
}
