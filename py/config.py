"""Central configuration for EVMS pipeline modules."""

from pathlib import Path

# Root directory containing data/, output/, py/, docs/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical project paths (absolute).
PATHS = {
    "data": PROJECT_ROOT / "data",  # Input assets (schemas, templates, mappings).
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
    "model": "llama3.2",  # Default text model for llm_client/testing scripts.
    "keep_alive": 20,  # Seconds to keep model loaded between calls (0 disables).
}

ROI_PROMPT_CREATOR = {
    "model": "qwen3.5:9b",  # Dedicated text model for ROI prompt generation.
}

TEXT_ROI_LLM = {
    "enabled": True,  # LLM post-step for non-header text ROIs (after default OCR extraction).
    "model": None,  # None => use LLM["model"].
    "timeout_sec": 120,  # Timeout per ROI post-step call.
    "max_reruns": 3,  # Retry count when model output does not conform to {"detected_text": string|null}.
    "extra_params": {"think": False},  # Prefer concise deterministic outputs.
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
    # - "vision_only": vision only.
    "workflow_default": "vision_only",

    # Vision OCR model name.
    "model": "glm-ocr:latest",
    # "model": "ministral-3:8b",

    # Paddle stage settings.
    "paddle_model_name": "paddle-ppocrv5",
    "paddle_min_confidence_for_accept": 0.90,  # If Paddle min score <= this, use vision fallback.
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
    "jpeg_quality": 20,  # JPEG quality used for VLM image payload.
    "stream": False,  # Stream responses from model server.
    "extra_params": 
    {
        "options": {"temperature": 0}
     },  # Extra model-server payload params.

    # Global toggle: when False, ignore per-ROI OCR prompt overrides.
    "use_ocr_prompts": True,
}

ID_FORM_LLM = {
    "crop_top_percent": 15.0,  # Top portion of page used for ID/form_type OCR.
    "crop_write_debug_image": False,  # Write top-crop debug PNGs.
    "crop_debug_dir": PATHS["output"] / "debug_images" / "id_crop",  # Debug output dir for top-crops.
    "form_types": ["6pre", "7pre", "8pre", "hpre", "6post", "7post", "8post", "hpost"],  # Allowed form_type values.
    "normalize_id_output": False,  # Normalize ID OCR output to canonical 7-digit+A/B pattern.
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

PDF_RECOGNITION = {
    "output_dir": PATHS["output"] / "recognition",  # Final recognition JSON output directory.
    "schema_key_sidea": "schema_sidea",  # key name for side-a schema in schema map.
    "schema_key_sideb": "schema_sideb",  # key name for side-b schema in schema map.
    "schema_dir": PATHS["data"] / "roi_schemas",  # ROI schema directory.
    "mc_roi_names": [],  # Empty => infer from schema (all names matching \d+[a-h]).
    "debug_output": True,  # Write pipeline debug JSON snapshots.
    "sort_data_by_roi_name": True,  # Sort ROI rows numerically by name.
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
    "template_registration_templates_dir": PATHS["data"] / "templates",  # Directory containing <form_type>_<side>.png templates.
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
    "ocr_regex_retry_enabled": True,
    # Ordered, editable list of retry method names. Keep this short to bound runtime.
    "ocr_regex_retry_steps": [
        "pad_8px",
        "clahe_light",
        "fsrcnn_x2",
        "adaptive_binarize",
        "adaptive_mean",
        "adaptive_gauss",
        "pad_8px_clahe_light",
        "pad_8px_adaptive_gauss",
    ],
    # Small set of tunables used by retry methods.
    "ocr_regex_retry_pad_px": 8,
    "ocr_regex_retry_clahe_clip": 2.0,
    "ocr_regex_retry_clahe_tile": 8,
    "ocr_regex_retry_adaptive_block": 31,
    "ocr_regex_retry_adaptive_c_mean": 8,
    "ocr_regex_retry_adaptive_c_gauss": 10,
    "ocr_regex_retry_adaptive_c_binarize": 9,
    # FSRCNN super-resolution retry step:
    # - If model path is empty or unavailable, the step is skipped safely.
    # - Typical model file is OpenCV FSRCNN_x2.pb.
    "ocr_regex_retry_fsrcnn_model_path": "/home/ges/Documents/evms_3/data/models/FSRCNN-small_x2.pb",
    "ocr_regex_retry_fsrcnn_scale": 2,
    # If FSRCNN runtime/model fails (e.g., unsupported OpenCV ops), fall back
    # to standard resize upscaling so the step is still useful.
    "ocr_regex_retry_fsrcnn_fallback_resize": True,
}

# Shared master workbook used for all form types.
_XLSX_MASTER = PATHS["data"] / "2025 EVMS NPS Data Entry tool (Blank Template, Do not write on) .xlsx"

XLSX_DATA_ENTRY = {
    "mapping_dir": PATHS["data"] / "xlsx_mappings",  # Directory with <form_type>.json mapping files.
    "master_template_workbook": _XLSX_MASTER,  # Multi-sheet template workbook.
    "output_dir": PATHS["output"] / "xlsx",  # Final XLSX output directory.
    "form_type_sheet_map": {
        "6pre": "6th Grade Pre-Assessment Data",
        "7pre": "7th Grade Pre-Assessment Data",
        "8pre": "8th Grade Pre-Assessment Data",
        "6post": "6th Grade Post-Assessment Data",
        "7post": "7th Grade Post-Assessment Data",
        "8post": "8th Grade Post-Assessment Data",
        "hpre": "HS Pre-Assessment Data",
        "hpost": "HS Post-Assessment Data",
    },
    "staging_dir": PATHS["cache"] / "xlsx_staging",  # Temporary filled workbook staging area.
}
