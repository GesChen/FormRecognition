# Module reference

All pipeline modules live in `py/`. Each is designed as a standalone importable unit
with a single public entry point (function or small set of functions) and lazy-loaded
heavy dependencies.

---

## py/config.py

Central configuration consumed by every other module. Contains no logic — only
`Path` constants and plain dicts.

### Exports

| Name | Type | Purpose |
|---|---|---|
| `PROJECT_ROOT` | `Path` | Absolute path to the repo root (parent of `py/`). |
| `PATHS` | `dict` | Standard directory paths (`data`, `output`, `docs`, `cache`, `cache_normalized`). |
| `PDF_TO_IMAGES` | `dict` | Defaults for `pdf_to_images` (cache root, DPI, format). |
| `IMAGE_NORMALIZE` | `dict` | Normalization pipeline toggles and parameters (see below). |
| `PDF_RECOGNITION` | `dict` | Full-pipeline settings (output dir, schema dir, debug, sort, **human_review** OCR threshold). |
| `LLM` | `dict` | Ollama server connection (host, port, model, keep_alive). |
| `ID_FORM_LLM` | `dict` | ID + form-type extraction (crop height, allowed types, retry/MP options). |
| `ROI_PAGE_RECOGNITION` | `dict` | MCQ scorer tuning (eps, suppression, power, blur, expand, debug). |
| `XLSX_DATA_ENTRY` | `dict` | Mapping dir + template path for `xlsx_data_entry` Excel filling. |

### IMAGE_NORMALIZE keys

| Key | Default | Effect |
|---|---|---|
| `cache_root` | `output/cache/normalized` | Where normalized images are saved. |
| `perspective_correct` | `False` | Detect document quad and warp (for photos, not PDFs). |
| `deskew` | `False` | Rotate via `minAreaRect` (skip for PDF-sourced pages). |
| `binarize` | `True` | Adaptive threshold to produce black/white output. |
| `template_registration` | `True` | Align to template via ORB + RANSAC homography. |
| `template_registration_templates` | `{}` | Key → path overrides for templates. |
| `template_registration_templates_dir` | `data/templates` | Directory of `<key>.png` template files. |
| `template_registration_default_key` | `None` | Fallback when no form_type/side provided. |
| `binarize_block_size` | `31` | Adaptive threshold block size (must be odd). |
| `clahe_clip_limit` | `2.0` | CLAHE contrast-limit clip. |
| `denoise_h` | `10` | `fastNlMeansDenoising` filter strength. |
| `contrast_power` | `1.2` | Power curve: >1 darkens grays, <1 lightens. |

### ROI_PAGE_RECOGNITION keys

| Key | Default | Effect |
|---|---|---|
| `mcq_subroi_expand_px` | `0` | Pixel padding added to each sub-ROI boundary. |
| `mcq_eps` | `0.05` | Residual below this (0–1 scale) treated as noise. |
| `mcq_print_suppression_k` | `5.0` | Exponential decay weight for already-dark template pixels. |
| `mcq_power` | `1.0` | Exponent on residual; >1 boosts strong marks. |
| `mcq_blur_sigma` | `0` | Gaussian blur sigma before comparison (0 = off). |
| `mcq_return_raw_darkness_debug` | `False` | Include per-choice scores in MCQ output. |

### ID_FORM_LLM keys

| Key | Default | Effect |
|---|---|---|
| `crop_top_percent` | `8.0` | Top fraction of image height (0–100) used for OCR; scales with resolution (~200 px at ~2500 px height). |
| `crop_write_debug_image` | `False` | If `True`, save the top-crop region (pixels sent to OCR) as PNG(s) under `crop_debug_dir`. |
| `crop_debug_dir` | `output/debug_images/id_crop` | Directory for crop debug PNGs: single image `stem_crop_top<pct>pct.png`; batch `stem_NNNN_crop_top<pct>pct.png`. |
| `form_types` | `["6pre","7pre",…]` | Allowed form-type values for LLM selection. |
| `debug_prompt` | `False` | Print full LLM prompt before each call. |
| `max_llm_reruns` | `2` | Retry LLM when output shape is invalid. |
| `use_multiprocessing` | `False` | Parallel OCR crop step. |
| `num_workers` | `4` | Worker processes when MP is on. |

### PDF_RECOGNITION keys

| Key | Default | Effect |
|---|---|---|
| `output_dir` | `output/recognition` | Directory for `<pdf_stem>.json`. |
| `schema_dir` | `data/roi_schemas` | Per-form ROI JSON schemas. |
| `debug_output` | `True` | (Legacy naming.) |
| `sort_data_by_roi_name` | `True` | Sort each item’s `data[]` by ROI name. |
| `human_review` | see below | Optional manual-review queue; text/ID rows primarily use OCR `needs_human_review` flags. |

Nested `human_review`:

| Key | Default | Effect |
|---|---|---|
| `enabled` | `False` | Collect OCR metadata; emit `human_review` plus OCR confidence fields on rows. |
| `text_review_use_needs_human_review` | `True` | Queue **text** review using OCR `needs_human_review` (primary signal). |
| `id_review_use_needs_human_review` | `True` | Queue **id** review using `id_ocr_needs_human_review` when ID is otherwise valid. |
| `min_text_ocr_confidence` | `0.85` | Fallback threshold for text when `ocr_needs_human_review` is missing. |
| `min_id_ocr_confidence` | `0.85` | Fallback threshold for ID when `id_ocr_needs_human_review` is missing. |
| `pending_id_review` | `True` | Enable **id** pending rows (all, invalid, and/or OCR-driven low-confidence). |
| `pending_all_ids` | `True` | If `True`, add `kind: "id"` pending for **every** item (full ID QC). |
| `pending_mcq_empty_review` | `True` | Add `kind: "mcq"` pending when the MCQ answer string is empty. |

Full spec: `docs/ocr_human_review_schema.md`. Detection and `apply_manual_corrections()` live in `py/ocr_human_review.py`.

---

## py/pdf_to_images.py

Convert a PDF file to per-page images and cache them on disk.

### Dependencies

- `PyMuPDF` (`fitz`) — rendering.
- `tqdm` — optional progress bar.

### Public functions

#### `pdf_to_images(pdf_path, cache_root=None, dpi=None, fmt=None, *, use_tqdm=False, max_pages=None) -> list[Path]`

Render each page of the PDF at the configured DPI and save to
`<cache_root>/<sanitized_stem>/page_NNNN.png`. Skips re-rendering if the cache
directory already contains images (no mtime check — use `--recache` at the pipeline
level to clear).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `pdf_path` | `str \| Path` | required | Path to the PDF. |
| `cache_root` | `str \| Path \| None` | config | Override `PDF_TO_IMAGES["cache_root"]`. |
| `dpi` | `int \| None` | config | Override `PDF_TO_IMAGES["dpi"]` (default 150). |
| `fmt` | `str \| None` | config | `"png"` or `"jpeg"`. |
| `use_tqdm` | `bool` | `False` | Show progress bar. |
| `max_pages` | `int \| None` | `None` | Render at most N pages. |

**Returns:** list of absolute `Path` objects (one per rendered page, in order).

### CLI

```
python3 py/pdf_to_images.py <pdf_path>
```

Prints the cached image paths to stdout.

---

## py/image_normalize.py

Normalize scanned/rendered document images for downstream OCR and ROI extraction.

### Pipeline (per image)

1. **Template registration** (if enabled): ORB keypoints + RANSAC homography warp
   to align the page to a blank template.
2. **Perspective correction** (if enabled): detect document quad, four-point warp.
3. **Deskew** (if enabled): `minAreaRect` rotation.
4. **Denoise + CLAHE**: `fastNlMeansDenoising` then CLAHE.
5. **Contrast stretch**: percentile clip + power curve.
6. **Binarize** (if enabled): adaptive Gaussian threshold.

Results are cached; a file is skipped when the output is newer than the input.

### Template resolution priority

1. Explicit `template_key` argument (e.g. `"6pre_a"`).
2. `form_type` + `page_side` combined as `"<form_type>_<side>"`.
3. `IMAGE_NORMALIZE["template_registration_default_key"]` fallback.

Resolved key is looked up first in the `template_registration_templates` dict,
then as `<templates_dir>/<key>.png`.

### Public functions

#### `normalize_image(image_path, ..., form_type=None, page_side=None, *, cache_subdir=None) -> Path`

Normalize a single image. Returns the absolute path to the cached result.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `image_path` | `str \| Path` | required | Input image. |
| `cache_root` | `str \| Path \| None` | config | Override cache directory. |
| `binarize` | `bool \| None` | config | `True` for B/W, `False` for grayscale. |
| `binarize_block_size` | `int \| None` | config | Adaptive threshold block size. |
| `template_key` | `str \| None` | `None` | Explicit template selection. |
| `form_type` | `str \| None` | `None` | Combined with `page_side` for template key. |
| `page_side` | `str \| None` | `None` | `"a"` (odd) or `"b"` (even). |
| `cache_subdir` | `str \| None` | `None` | Save under `cache_root/cache_subdir/`. |

#### `normalize_images(image_paths, ...) -> list[Path]`

Batch variant — same parameters (scalars or parallel lists).

#### `template_key_for_page(page: int) -> str`

Legacy helper: returns `"a"` for odd pages, `"b"` for even.

### Internal helpers

| Function | Purpose |
|---|---|
| `_to_gray(img)` | Ensure single-channel uint8. |
| `_deskew(img)` | `minAreaRect`-based rotation. |
| `_find_doc_quad(gray)` | Largest 4-sided contour. |
| `_register_to_template(gray, template_path)` | ORB + RANSAC + warp. |
| `_perspective_correct(img, enabled)` | Four-point warp if quad found. |
| `_contrast_pass(gray)` | Percentile stretch + power curve. |
| `_denoise_and_clahe(gray)` | Denoise then CLAHE. |
| `_binarize_per_roi(gray, block_size, c)` | Adaptive threshold. |
| `_normalize_one(...)` | Full pipeline on one image; cache-aware. |
| `_resolve_template_key(...)` | Priority chain for template selection. |
| `_get_template_path(...)` | Dict/dir lookup from resolved key. |

---

## py/ocr_engine.py

Staged OCR module using a vision API. It escalates through model stages from `config.OCR_ENGINE`
and accepts early when confidence is high.

### Public functions

#### `ocr_raw(image_path, *, model_stages=None, timeout=None, force_paddle_failure=False) -> dict`

Runs OCR on one image and returns structured output:

- `detected_text`
- `confidence_label` (`low|medium|high`)
- `confidence_score` (0..1)
- `needs_human_review`
- `selected_model`, `selected_stage_index`
- `self_evaluation`
- `stages` (per-stage response/debug data)

`force_paddle_failure=True` is a test hook to force stage-0 Paddle to fail so API stages are exercised.

#### `ocr(image_path, *, model_stages=None) -> str`

Convenience wrapper returning only `detected_text`.

#### `ocr_confidence_stats(raw_result) -> dict`

Compatibility stats used by downstream pipeline blocks:
`min_rec_score`, `mean_rec_score`, `rec_scores`, `rec_texts`, plus label/review fields.

---

## py/llm_client.py

HTTP client for a local Ollama LLM server. No heavy dependencies beyond `requests`.

### Public functions

#### `generate(prompt, *, model=None, host=None, port=None, keep_alive=None, stream=False, timeout=300, extra_params=None) -> dict`

POST to `/api/generate` and return a dict:

| Key | Type | Content |
|---|---|---|
| `text` | `str` | Generated response text. |
| `raw` | `dict` | Full JSON body from server. |
| `elapsed` | `float` | Wall-clock seconds. |
| `eval_count` | `int \| None` | Tokens generated (if reported). |
| `eval_duration` | `int \| None` | Eval time in nanoseconds (if reported). |

All connection parameters default to `config.LLM`.

#### `ping(*, host=None, port=None, timeout=5) -> dict`

GET `/api/tags` and return `{"ok": True, "tags": ...}` or `{"ok": False, "error": ...}`.

---

## py/id_form_llm.py

Extract a 7-digit student ID and form type from page header images using
OCR engine + a single LLM call.

### Pipeline

1. **Crop**: keep top `crop_top_percent` of image height (default 8%, ~200 px on a ~2500 px-tall page).
2. **OCR**: run OCR engine on the crop to get noisy text.
3. **Prompt**: build a structured prompt listing the OCR text and the allowed
   `form_types`, asking for strict JSON output.
4. **LLM call**: send prompt to Ollama; parse response.
5. **Normalize**: apply OCR-correction heuristics to the ID (`O→0`, `I→1`, etc.)
   and validate form type against the allowed list.
6. **Retry** (optional): if parsing fails, retry the LLM call up to
   `max_llm_reruns` times (OCR is run once per image/crop).

### Public functions

#### `extract_id_and_form_type(image_path, *, crop_top_percent=None, form_types=None, return_raw_ocr=False, verbose=False, crop_debug_out=None) -> dict`

Single-image extraction. Returns `{"id": str|None, "form_type": str|None}`.
When `return_raw_ocr=True`, the result also contains `"raw_ocr"`.
With `verbose=True`, prints LLM retry messages; full prompt text prints only if `verbose=True` and `ID_FORM_LLM["debug_prompt"]` is true.
Optional `crop_debug_out` writes the top-crop PNG to that path; if unset, use `ID_FORM_LLM["crop_write_debug_image"]` and `crop_debug_dir`.

#### `extract_id_and_form_type_batch(image_paths, *, crop_top_percent=None, form_types=None, verbose=False, debug_out=None, crop_debug_dir=None) -> list[dict]`

Batch extraction: OCR all images, build one combined prompt, one LLM call.
Returns a list of `{"id", "form_type"}` dicts aligned to input order.

| Parameter | Type | Notes |
|---|---|---|
| `image_paths` | `list[str\|Path]` | One path per page to process. |
| `verbose` | `bool` | Print LLM timing and status messages. |
| `debug_out` | `dict \| None` | Mutable dict filled with `ocr_per_page` and `llm` debug data. |

**Multiprocessing:** when `ID_FORM_LLM["use_multiprocessing"]` is `True` and debug
is not active **and** crop debug output is not enabled, the OCR crop step runs across `num_workers` processes via
`ProcessPoolExecutor`.

#### `build_prompt(ocr_content, form_types) -> str`

Build the LLM prompt string. Accepts a single string (single image) or list of
strings (batch). Includes colon-hint logic when OCR text contains `:`.

### Internal helpers

| Function | Purpose |
|---|---|
| `_id_instruction(form_types_str)` | Reusable extraction rules text block. |
| `_top_rows_from_percent(h, percent)` | Row count for top ``percent`` of height. |
| `_crop_top_and_ocr(image_path, crop_top_percent, crop_debug_out=…)` | Crop + OCR → text; optional PNG of the crop. |
| `_crop_top_and_ocr_with_raw(image_path, crop_top_percent, crop_debug_out=…)` | Same, also returns raw OCR output. |
| `_parse_llm_json(text)` | Strip fences, extract `{...}` from LLM response. |
| `_normalize_form_type(value, allowed)` | Validate against allowed list. |
| `_normalize_id(value)` | OCR corrections + regex match for 7-digit + A/B. |
| `_coerce_llm_batch_results(data)` | Flatten various LLM JSON shapes to `list[dict]`. |
| `_llm_generate_with_retries(prompt, ...)` | Call LLM, validate shape, retry on failure. |
| `_serialize_raw_ocr_for_debug(raw_result)` | JSON-serializable form of OCR output. |

---

## py/roi_page_module.py

ROI-based page interpretation: load a form-specific schema, classify ROIs into
text and MCQ types, run recognition for each, and return a consolidated result list.

### Dataclasses

```
BaseRoi
├── name: str
├── x, y, w, h: float
└── meta: Dict[str, Any]

TextRoi(BaseRoi)          kind = "text"
McqChoiceRoi(BaseRoi)     kind = "mcq_choice", letter: str
McqRoi(BaseRoi)           kind = "mcq", choices: List[McqChoiceRoi]

PageRoiSchema
├── form_type: Optional[str]
├── side: str
├── schema_path: Path
├── text_rois: List[TextRoi]
├── mcq_rois: List[McqRoi]
└── raw_schema: Dict[str, Any]
```

### ROI classification rules

Given a schema JSON with a list of `rois` (each having `name`, `x`, `y`, `w`, `h`):

- **Text ROI**: `name` is a pure integer (e.g. `"5"`) AND no other ROI has a name
  of the form `"5a"`, `"5b"`, etc.
- **MCQ ROI**: there exist ROIs named `"<N><letter>"` (e.g. `"5a"`, `"5b"`). These
  become `McqChoiceRoi` entries. If a main `"5"` ROI also exists, its bbox is used;
  otherwise the union bbox of the choices is computed.

### MCQ scoring algorithm (soft template-subtracted darkness)

1. Load template and target as grayscale; convert to 0–1 **darkness** (0 = white, 1 = black).
2. Optionally Gaussian-blur both (`mcq_blur_sigma`).
3. For each sub-ROI choice region (with optional `mcq_subroi_expand_px` padding):
   - `extra_dark = clip(target_dark − template_dark − eps, 0, 1)`
   - `weight = exp(−k × template_dark)` (suppresses printed structure).
   - `score = Σ(weight × extra_dark ^ power)`
4. Answer = letter with the highest score; `""` if all scores ≤ 0.

### Public functions

#### `load_page_schema(form_type, side) -> PageRoiSchema`

Load `data/roi_schemas/<form_type>_<side>.json`, interpret ROIs.

#### `recognize_text_fields(page_image_path, page_schema, *, debug_collector=None) -> dict[str, str]`

Crop each `TextRoi` from the normalized image, run OCR engine, return
`{roi_name: recognized_text}`.

Coordinates are scaled when the schema's `image_width`/`image_height` differ from
the actual image dimensions.

#### `recognize_mcq_fields(page_image_path, page_schema, *, debug_collector=None) -> dict[str, Any]`

Score each `McqRoi` using the soft template-subtracted algorithm.
Returns `{roi_name: answer_letter}` or `{roi_name: {"answer": ..., "scores": ...}}`
when debug is active.

#### `analyze_page(page_image_path, *, form_type, side, debug_collector=None, pair_index=0, page_in_pair="odd", pdf_stem="", review_text_queue_out=None) -> list[dict]`

Central entry point. Loads schema, runs both recognizers, consolidates output into:

```python
[
    {"name": "1", "kind": "text", "text": "recognized content", "review_target_ref": "..."},
    {"name": "10", "kind": "mcq", "text": "b", "review_target_ref": "..."},
    ...
]
```

When `pdf_stem` is set, each row includes **`review_target_ref`** (see `docs/ocr_human_review_schema.md`). With **`review_text_queue_out`** and human-review config, text ROIs flagged by OCR `needs_human_review` (or fallback threshold logic) are appended to that list from `recognize_text_fields`.

#### `analyze_pages_batch(page_inputs, *, debug_out=None) -> list[list[dict]]`

Batch wrapper. Each element of `page_inputs` is a dict:

```python
{
    "page_image_path": Path,
    "form_type": str | None,
    "side": "a" | "b",
    "pair_index": int,  # optional, default 0
    "page_in_pair": "odd" | "even",  # optional
    "pdf_stem": str,  # optional
    "review_text_queue_out": list,  # optional shared queue for OCR-flagged text review rows
}
```

Returns one list per page, aligned to input order. When `debug_out` is provided,
`debug_out["roi_per_page"]` is populated with per-page debug data.

### Internal helpers

| Function | Purpose |
|---|---|
| `_schema_path_for_form(form_type, side)` | Resolve schema JSON path with config fallback. |
| `_is_pure_number(name)` | `name.isdigit()`. |
| `_split_number_letter(name)` | Parse `"12a"` → `("12", "a")` or `None`. |
| `_interpret_rois(schema)` | Classify raw ROIs into `TextRoi` / `McqRoi` lists. |
| `_serialize_raw_ocr(raw_result)` | JSON-safe form of OCR output (for debug). |

---

## py/pdf_recognize.py

Orchestrator: drives the full PDF → structured JSON pipeline.

### 5-step workflow

| Step | Method | What it does |
|---|---|---|
| 1 | `_load_pdf_pages` | PDF → cached page images (via `pdf_to_images`). |
| 2 | `_detect_ids_and_form_types` | Top-crop OCR + LLM on side-a pages only (via `id_form_llm`). |
| 3 | `_normalize_pages` | Form-specific template registration + binarization (via `image_normalize`). |
| 4a | `_analyze_all_pages` | ROI extraction for all pages (via `roi_page_module`). |
| 4b | `_build_pair_items` | Group pages into odd/even pairs, merge ROI data, sort. |
| 5 | `_write_output` | Write final JSON to `output/recognition/<stem>.json`. |

### Output JSON shape

```json
{
    "pdf_path": "/absolute/path/to/file.pdf",
    "pdf_stem": "file",
    "generated_at": "2026-03-15T12:00:00Z",
    "item_count": 2,
    "items": [
        {
            "id": "9010526A",
            "form_type": "6pre",
            "page_odd": 1,
            "page_even": 2,
            "data": [
                {"name": "1", "kind": "text", "text": "...", "page_in_pair": "odd"},
                {"name": "10", "kind": "mcq", "text": "b", "page_in_pair": "odd"},
                ...
            ]
        }
    ],
    "human_review": {
        "text_review_use_needs_human_review": true,
        "id_review_use_needs_human_review": true,
        "min_text_ocr_confidence": 0.85,
        "pending": []
    }
}
```

`human_review` is present only when `PDF_RECOGNITION["human_review"]["enabled"]` is `true`. Each `data[]` row includes `page_in_pair` (`"odd"` \| `"even"`). `pending[]` may list **`kind`**: `"id"` (manual student id), `"text"` (OCR flagged), `"mcq"` (empty answer), per config.

### Public functions

#### `run_workflow(pdf_path, output_dir=None, *, write_json=True, verbose=True, max_pages=None, recache=False, debug=False, debug_path=None) -> list[dict]`

Run the full pipeline and return the `items` list.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `pdf_path` | `str \| Path` | required | Input PDF. |
| `output_dir` | `Path \| None` | config | Override output directory. |
| `write_json` | `bool` | `True` | Write result JSON to disk. |
| `verbose` | `bool` | `True` | Step messages + tqdm bars. |
| `max_pages` | `int \| None` | `None` | Process at most N pages. |
| `recache` | `bool` | `False` | Delete existing cache first. |
| `debug` | `bool` | `False` | Write `<stem>_debug.json` with full pipeline trace. |
| `debug_path` | `Path \| None` | `None` | Custom path for debug file. |

### CLI

```
python3 py/pdf_recognize.py <pdf_path> [options]
```

| Flag | Effect |
|---|---|
| `-v`, `--verbose` | Enable verbose output. |
| `--no-json` | Skip writing output file. |
| `--max-pages N` | Limit pages processed. |
| `--recache` | Clear cache before processing. |
| `--debug` | Write debug JSON. |
| `--debug-path PATH` | Custom debug file path. |

### Internal helpers

| Function | Purpose |
|---|---|
| `_sanitize_pdf_stem(name)` | Safe filename from PDF name. |
| `_schema_path(key)` | Legacy schema path by key. |
| `_schema_path_for_form(form_type, side)` | Per-form schema resolution. |
| `_get_mc_roi_names(schema_path)` | Legacy: MCQ bubble names from schema. |
| `_bool_dict_to_letter_per_question(...)` | Legacy: filled bools → letter. |
| `_letter_per_question_from_filled_and_darkness(...)` | Legacy: darkness-based letter pick. |
| `_merge_mcq_letters(...)` | Legacy: merge odd/even MCQ letters. |
| `_darkness_by_question(...)` | Legacy: group darkness by question number. |
| `_merge_mcq_darknesses(...)` | Legacy: merge odd/even darkness dicts. |
| `_normalized_path_for_page(page_path)` | Expected normalized image path. |
| `_roi_name_sort_key(entry)` | Sort key: numeric names first. |
| `_sort_data_by_roi_name(data)` | Sort data list by ROI name. |
| `_merge_page_data(odd, even)` | Concatenate page data lists; add `page_in_pair` per row. |
| `_log(msg, verbose, file)` | Conditional print. |

---

## py/ocr_human_review.py

Builds the optional `human_review` object for recognition JSON and applies manual edits.

| Function | Purpose |
|---|---|
| `build_human_review_block(items, pdf_stem, cfg, *, text_ocr_queue=None)` | Return `human_review` dict or `None` if disabled. **Text** rows: prefer OCR `needs_human_review` from `text_ocr_queue` (from `recognize_text_fields`), with threshold fallback for legacy rows. Also **id** (invalid/all per config), **mcq** (empty answer). |
| `apply_manual_corrections(items, human_review)` | Deep-copy `items` and merge `corrected_id` / `corrected_text` from `pending[]` into `items[].id` and `items[].data[]`. |
| `is_valid_student_id(value)` | Whether `id` matches `^\d{7}[AB]$` (after trim/upper). |
| `make_entry_id(...)` | Stable `entry_id` for a pending row. |

See `docs/ocr_human_review_schema.md`.

---

## py/xlsx_data_entry.py

Fills **row-oriented** Excel templates from **human-authored mapping JSON** files
and raw **pipeline items** (from `pdf_recognize`).  Each form type has a mapping
file at `data/xlsx/mappings/<release>/<form_type>.json`.  See `docs/xlsx_mapping_spec.md` for the full spec.

### Config: `XLSX_DATA_ENTRY`

| Key | Default | Purpose |
|---|---|---|
| `mapping_dir` | `PATHS["data"] / "xlsx" / "mappings" / <release>` | Directory for `<form_type>.json` mapping files. |
| `output_dir` | `PATHS["output"] / "xlsx"` | Directory for merged per-PDF output workbooks. |
| `staging_dir` | `PATHS["cache"] / "xlsx_staging"` | Default directory for staging filled copies. |

### Mapping types

| Type | Effect |
|---|---|
| `direct` | Write value as-is. Optional `transform`: `upper`, `lower`, `number`, `date`. |
| `lookup` | Map value through a `map` dictionary (e.g. `"b"` → `"True"`). |
| `multi_column` | Value selects which column gets `mark`; others get `blank`. |
| `static` | Write a fixed `value` every row. |

### Public functions

| Function | Purpose |
|---|---|
| `load_mapping(form_type, cfg)` | Load `data/<form_type>.json`. |
| `items_from_payload(payload, limit=None)` | Extract pipeline items from `{"items":[]}`, array, or single dict. |
| `resolve_template_path(cfg)` | Resolve the release-scoped workbook template from `data/xlsx/workbook_templates/<release>/`. |
| `resolve_source(item, source)` | Read value from a pipeline item by source string. |
| `apply_mapping_entry(ws, row, entry, item)` | Apply one mapping entry to a cell. |
| `fill_row(ws, row, mapping, item)` | Apply all mappings for one item. |
| `fill_template(mapping, items, output_path, …)` | Copy template, stamp data rows, write items, save. |
| `fill_staging(mapping, items, …)` | Like `fill_template` with auto-generated output path. |
| `merge_workbooks(staging_paths, output_path)` | Merge single-sheet staging workbooks into one multi-sheet workbook. |
| `fill_from_pipeline(items, output_path, …, verbose=True)` | Group items by form_type, fill per-type, merge into final workbook. Called automatically by `pdf_recognize` step 6. |
| `make_staging_output_path(form_type, cfg)` | Unique path under `staging_dir`. |

### CLI

```
python3 py/xlsx_data_entry.py <form_type> <pipeline.json> [output.xlsx]
python3 py/xlsx_data_entry.py <form_type> <pipeline.json> --staging
python3 py/xlsx_data_entry.py 6post output/recognition/post_crossroads_6_Davis.json output/filled.xlsx
```

`<form_type>` resolves to `data/<form_type>.json`.  `<pipeline.json>` is the recognition
output (with `items[]`), or `"-"` for stdin.  `--staging` auto-generates the output path.
`--limit N` caps items processed.

### Mapping spec

See `docs/xlsx_mapping_spec.md`.
