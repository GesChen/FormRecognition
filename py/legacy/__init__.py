"""
Legacy modules superseded by the current pipeline.

Modules here were part of the original recognition workflow and have been replaced by:
  - id_form_llm.py (replaces id_recognize.py)
  - roi_page_module.py (replaces metadata_getter.py, mc_detect.py)
  - ocr_engine.py (replaces ocr_tesseract.py and ocr_easyocr.py)

Config for these modules lives in config_legacy.py (this directory).

To reintegrate a module:
  1. Move it back to py/.
  2. Move its config section from py/legacy/config_legacy.py back to py/config.py.
  3. Update imports in the module (config_legacy -> config).
  4. Update any test scripts that reference py/legacy/.
"""
