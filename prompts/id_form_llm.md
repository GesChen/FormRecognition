# ID and form type extraction (LLM)

Module: `py/id_form_llm.py`. Single-call API after import (ref: prompts/module).

## Behavior

1. **Input:** One image path or a list of image paths.
2. **Crop:** Each image is cropped to a configured **fraction of height** from the top (config: `ID_FORM_LLM["crop_top_percent"]`, 0–100).
3. **OCR:** The OCR engine runs on each cropped region. For a single image, that text is used as-is. For a batch, all texts are collected.
4. **LLM:** One LLM call per invocation. The prompt is built by `build_prompt()` at the top of the module (easy to reconfigure). The prompt asks the LLM to:
   - **ID:** Extract the value matching the pattern *7 digits in a row with a letter at the end, either A or B* (e.g. `9010526A`, `9010527B`). Use `null` if not found.
   - **Form type:** Choose exactly one from the configured list: `6pre`, `7pre`, `8pre`, `hpre`, `6post`, `7post`, `8post`, `hpost`. Use `null` if unclear.
5. **Output:** The LLM must respond with a JSON object only. The module parses it and returns:
   - Single: `{"id": "... or null", "form_type": "... or null"}`
   - Batch: list of such dicts, one per image, in order.

## Config (config.py)

- `ID_FORM_LLM["crop_top_percent"]`: percent of image height to keep from the top (default `8.0`, roughly equivalent to the former 200 px crop on a ~2500 px-tall page).
- `ID_FORM_LLM["crop_write_debug_image"]` / `["crop_debug_dir"]`: optional; save the top-crop region as PNG file(s) for debugging (same pixels as OCR input).
- `ID_FORM_LLM["form_types"]`: list of valid form type strings.

## Prompt builder

The prompt text is built in `build_prompt(ocr_content, form_types)` in `py/id_form_llm.py`. Override or edit that function to change instructions or JSON shape.
