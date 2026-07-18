# Testing scripts reference

All test scripts live in `testing/`. Run them from the project root:

```bash
python3 testing/<script>.py [args]
```

Each script adds `py/` to `sys.path` automatically. Test outputs go under
`testing/output/` (per project rules), though some scripts display results
in-terminal only.

## test_xlsx_data_entry.py

Fill a workbook using `xlsx_data_entry` (mapping JSON + pipeline JSON).

```bash
# Defaults: form_type 6post, post_crossroads_6_Davis.json, testing/output/xlsx_data_entry_test.xlsx
python3 testing/test_xlsx_data_entry.py

python3 testing/test_xlsx_data_entry.py --form-type 6post \
  -r output/recognition/my_run.json -o testing/output/filled.xlsx -n 5

# Staging mode: auto-generated output path under testing/output/
python3 testing/test_xlsx_data_entry.py --staging --form-type 6post
```

| Flag | Purpose |
|---|---|
| `--form-type` | Form type — resolves to `data/xlsx/mappings/<xlsx_template_name>/<TYPE>.json` mapping (default `6post`). |
| `-r`, `--records` | Pipeline output JSON with `items[]`. |
| `-o`, `--out` | Output `.xlsx` (default under `testing/output/`). With `--staging`, omit for auto path. |
| `--staging` | Call `fill_staging`; `staging_dir` forced to `testing/output/`. |
| `-n`, `--limit` | First N items only. |

---

## test_pdf_recognize.py

End-to-end test of the full pipeline.

```bash
python3 testing/test_pdf_recognize.py
```

- Hardcoded PDF: `data/maury 1.pdf`.
- Calls `run_workflow(PDF_PATH, write_json=True)`.
- Asserts the output JSON file exists and the returned data matches.
- Prints a summary line per item (page numbers, ID).

**Note:** the assert checks `"metadata"` and `"mcq"` keys which no longer exist in
the current pipeline output. This test needs updating to match the new `data` list
structure.

---

## test_id_form_llm.py

Test ID + form-type extraction (top crop -> OCR engine -> LLM).

```bash
# Single image (prints id, form_type, raw OCR output)
python3 testing/test_id_form_llm.py <image_path>

# Batch (one LLM call for all)
python3 testing/test_id_form_llm.py <image1> <image2> ...
```

**Single-image mode:**

- Calls `extract_id_and_form_type(path, return_raw_ocr=True)`.
- Prints `id`, `form_type`, elapsed time, and the full raw OCR JSON.

**Batch mode:**

- Calls `extract_id_and_form_type_batch(paths)`.
- Prints one line per image with `id` and `form_type`.
- Prints total elapsed time.

### Internal

`_serialize_raw_ocr(raw_obj)` — converts raw OCR output to
JSON-serializable dicts (mirrors `id_form_llm._serialize_raw_ocr_for_debug`).

---

## test_ocr_engine.py

Run OCR detector test on a single raw image.

```bash
python3 testing/test_ocr_engine.py <image_path> [--verbose] [--json]
```

- Uses a local tester helper in `testing/test_ocr_engine.py` built on `ocr_raw(...)`.
- Supports optional overrides:
  - `--models model1,model2,...`
  - `--timeout <seconds>`
  - `--force-paddle-failure` (force stage-0 failure so API stages are exercised)
  - `--verbose` (include raw per-stage API payloads)
  - `--json` (machine-friendly full JSON output)
- Useful for debugging OCR quality on individual crops.

---

## test_ocr_engine_crop.py

Run OCR engine on only the top N rows of an image, with an optional visual preview.

```bash
python3 testing/test_ocr_engine_crop.py <image_path> <rows> [--no-preview]
```

| Argument | Type | Notes |
|---|---|---|
| `image_path` | positional | Path to the image. |
| `rows` | positional `int` | Number of rows to keep from the top. |
| `--no-preview` | flag | Skip the OpenCV preview window. |

- Crops the image, optionally shows a resized preview (max 1000 px), then OCRs the
  crop via a temp file.
- Prints crop dimensions, OCR timing, and recognized text.

---

## test_llm_connection.py

Sanity check for the Ollama LLM server.

```bash
python3 testing/test_llm_connection.py
```

- Reads `LLM["host"]`, `LLM["port"]`, `LLM["model"]` from config.
- Calls `ping()` to verify server reachability; prints available models.
- Sends a long test prompt (600+ words on data normalization) via `generate()`.
- Prints the prompt, model response, and throughput stats (tokens, tok/s).

---

## test_image_normalize.py

Convert a PDF to page images (if needed) and normalize all pages.

```bash
python3 testing/test_image_normalize.py [pdf_path]
```

- Default PDF: `data/maury 1.pdf`.
- Checks `PDF_TO_IMAGES["cache_root"]` for existing page images; converts if absent.
- Normalizes each page using `template_key_for_page(N)` (odd=a, even=b).
- Prints normalized output paths.

**Note:** uses the legacy `template_key_for_page` helper, not form-type-based
template selection. To test form-aware normalization, use the full pipeline via
`test_pdf_recognize.py` or `pdf_recognize.py` directly.

---

## test_pdf_to_images.py

Minimal test of `pdf_to_images`.

```bash
python3 testing/test_pdf_to_images.py
```

- Hardcoded PDF: `data/maury 1.pdf`.
- Calls `pdf_to_images(PDF_PATH)` and prints cached image paths.

---

## test_id_recognize.py

Legacy: run ID recognition with Tesseract on 10 normalized images.

```bash
python3 testing/test_id_recognize.py
```

- Normalizes 10 pages from cache, then calls the old `id_recognize` module.
- Prints recognized IDs in a table.

---

## test_id_recognize_easyocr_odd.py

Legacy: run ID recognition with EasyOCR on 10 odd-numbered pages.

```bash
python3 testing/test_id_recognize_easyocr_odd.py
```

- Similar to `test_id_recognize.py` but uses EasyOCR and only odd pages.

---

## test_id_recognize_ocr_engine_odd.py

Legacy: run ID recognition with OCR engine on 10 odd-numbered pages.

```bash
python3 testing/test_id_recognize_ocr_engine_odd.py
```

- Similar to `test_id_recognize.py` but uses OCR engine and only odd pages.

---

## test_metadata_getter.py

Legacy: test OCR on metadata fields (date, school, teacher).

```bash
python3 testing/test_metadata_getter.py
```

- Normalizes pages, crops metadata ROIs, runs OCR per field.
- Prints results in a table.

---

## test_mc_detect.py

Legacy: test multiple-choice bubble filled detection.

```bash
python3 testing/test_mc_detect.py
```

- Uses the old `mc_detect` module with graduated-falloff darkness scoring.
- Prints per-question results (filled/unfilled, chosen letter).

---

## test_crop_preview.py

Show the cropped ROI from one normalized page.

```bash
python3 testing/test_crop_preview.py
```

- Loads an ROI schema, crops each ROI from the normalized image.
- Displays in an OpenCV window or saves under `testing/output/`.

---

## test_crop_preview_cycle.py

Cycle through cropped ROIs across multiple normalized pages.

```bash
python3 testing/test_crop_preview_cycle.py
```

- Iterates over pages and ROIs, displaying each crop sequentially.
- Useful for visually inspecting ROI alignment across different scans.
