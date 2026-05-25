# Project setup (global Python)

This project uses the **global Python environment** and your existing global packages. No virtual environment is required.

## 1. Install dependencies (if needed)

From the project root, install packages into your user site or system (depending on your setup):

```bash
cd /home/ges/Documents/evms_3
pip install -r py/requirements.txt
```

If you see **externally-managed-environment**, your distro blocks normal `pip install`. Use:

```bash
pip install --break-system-packages -r py/requirements.txt
```

To install only EasyOCR:

```bash
pip install --break-system-packages easyocr
```

Other options (if the above is not desired):

- **User install:** `pip install --user -r py/requirements.txt` (often also blocked)
- **System packages (Debian/Ubuntu):** `sudo apt install python3-tk tesseract-ocr` for Tesseract; EasyOCR is not in apt, so use the `--break-system-packages` pip command above for EasyOCR

Optional helper script (no venv):

```bash
./scripts/install_deps.sh
```

## 2. Run scripts

Use your normal `python3` (or `python`):

```bash
cd /home/ges/Documents/evms_3
python3 testing/test_id_recognize.py
python3 testing/test_id_recognize_easyocr_odd.py
python3 tools/roi_editor.py
python3 py/pdf_to_images.py data/maury\ 1.pdf
```

One-off with project modules:

```bash
cd /home/ges/Documents/evms_3
python3 -c "
import sys
sys.path.insert(0, 'py')
from id_recognize import id_recognize
print(id_recognize('output/cache/normalized/maury_1/page_0001_normalized.png'))
"
```

Tests under `testing/` add `py` to `sys.path` automatically, so run them from the project root with `python3 testing/<script>.py`.

## 3. OCR engine (ID recognition)

In `py/config.py`, `ID_RECOGNITION["ocr_engine"]` controls which engine is used:

- `"tesseract"` — uses system/pip Tesseract (e.g. `sudo apt install tesseract-ocr` + `pip install pytesseract`)
- `"easyocr"` — uses EasyOCR (`pip install easyocr`)

Use whichever engine you have installed globally.
