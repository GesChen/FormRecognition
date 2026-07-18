# Tools reference

Tools live in `tools/` and are user-facing utilities (editors, viewers). They are
not imported by the pipeline modules.

---

## Start all web UIs

Flask-based tools default ports:

| Port | Tool | Script |
|------|------|--------|
| 4999 | **EVMS Hub** (PDF recognition + links) | `tools/evms_hub_web.py` |
| 5000 | ROI editor (web) | `tools/roi_editor_web.py` |
| 5001 | ROI previewer | `tools/roi_previewer_web.py` |
| 5002 | Pipeline debug viewer | `tools/pipeline_debug_web.py` |
| 5003 | Human review editor | `tools/human_review_web.py` |

**Theme** (dark/light, accent): see [`docs/tools_theme.md`](tools_theme.md).

Helper (from repo root):

```bash
# Background — returns immediately; logs under testing/output/web_tools/*.log
python3 tools/start_web_tools.py

# New terminal window — servers run there; Ctrl+C stops all
python3 tools/start_web_tools.py --terminal

# This shell — foreground (Ctrl+C stops all)
python3 tools/start_web_tools.py --watch

# Stop background servers (uses testing/output/web_tools/web_tools_pids.json)
python3 tools/start_web_tools.py --stop

# No automatic browser tab (hub is still http://127.0.0.1:4999/)
python3 tools/start_web_tools.py --no-browser
```

Starting the tools (**default**, `--watch`, or background after `--terminal` fallback) opens the **EVMS hub** in a **new browser tab** once port `4999` accepts connections. Use **`--no-browser`** to skip (e.g. SSH/headless).

### Linux: double-click like a `.bat` file

Simple wrappers call Python with a fixed path and wait for Enter:

| Script | Runs |
|--------|------|
| `tools/start_web_tools.sh` | `python3 …/tools/start_web_tools.py` then **Press Enter** |
| `tools/stop_web_tools.sh` | same with `--stop` |

Edit the path inside those `.sh` files if your clone is not at `/home/ges/Documents/evms_3`. `chmod +x` them; for `.desktop` launchers use **`Terminal=true`** (see `tools/evms_web_tools*.desktop.example`).

---

## EVMS Hub (`tools/evms_hub_web.py`)

Main entry page for the web tools: **PDF recognition** plus links to ROI editor, previewer, pipeline debug, and human review.

```bash
python3 tools/evms_hub_web.py [--port 4999]
```

- **PDF**: drag-and-drop onto the zone, click to pick files, or choose a folder to recursively include all PDFs in it and its subfolders (stored under `output/uploads/`). Alternatively enter a **project-relative** path to an existing PDF (no upload).
- **CLI parity**: checkboxes / fields for the same options as `python3 py/pdf_recognize.py` (`--verbose`, `--no-json`, `--max-pages`, `--recache`, `--debug`, `--debug-path`).
- **Output**: plain **CLI-style** stream in a monospace panel (subprocess stdout/stderr), no separate progress UI. Runs Python with **`-u`** and **`PYTHONUNBUFFERED=1`**; reads the pipe with **`read1`** where available so tqdm flushes often. **Carriage returns** (`\\r`) are preserved (CRLF → LF only); the hub page applies terminal-style line overwrite so tqdm updates the same line instead of spamming newlines. ANSI escapes are stripped server-side for a plain-text preview.
- **After run**: a highlighted panel shows the expected **XLSX** path (`output/xlsx/<stem>.xlsx` per `XLSX_DATA_ENTRY`) and the **recognition JSON** path (`output/recognition/<stem>.json`).
- **Cancel**: **Cancel** sends `POST /api/cancel-recognition` with the job id from `X-EVMS-Job-Id`; the server **SIGTERM**s the pipeline’s **process group** (`start_new_session` + `killpg`) so child workers stop too.

---

## ROI editor

An interactive editor for creating, modifying, deleting, and naming rectangular
regions of interest (ROIs) on normalized page images. ROI schemas are saved as JSON
under `data/roi_schemas/` and consumed by `roi_page_module.py` at pipeline time.

Three files implement the editor:

| File | Role |
|---|---|
| `tools/roi_editor.py` | Entry point + Tk GUI (`ROIEditorApp`). |
| `tools/roi_editor_core.py` | Shared data model (`ROI` class), schema I/O, browsing. |
| `tools/roi_editor_web.py` | Flask web server with REST API + HTML frontend. |

### Launch

```bash
# Tk GUI — open a specific image
python3 tools/roi_editor.py data/templates/6pre_a.png

# Tk GUI — default image (output/cache/normalized/maury_1/page_0001_bin.png)
python3 tools/roi_editor.py

# Web UI
python3 tools/roi_editor.py --serve [--port 5000]
```

### CLI arguments (roi_editor.py)

| Argument | Type | Default | Notes |
|---|---|---|---|
| `image_path` | positional, optional | default image | Path to a `.png` page image. |
| `--serve` | flag | — | Run Flask web server instead of Tk. |
| `--port` | `int` | `5000` | Port for the web server. |

---

### tools/roi_editor_core.py

Shared logic used by both the Tk and web editors.

#### Constants

| Name | Value | Purpose |
|---|---|---|
| `DEFAULT_BROWSE_ROOT` | `"data/templates"` | Root directory for the web file picker. |
| `DEFAULT_IMAGE` | `cache_normalized/.../page_0001_bin.png` | Fallback for the Tk editor. |
| `SCHEMA_DIR` | `data/roi_schemas` | Where schema JSONs are read/written. |
| `HANDLE_SIZE` | `8` | Pixel radius for resize handles. |
| `MIN_ROI_SIZE` | `5` | Minimum width/height in pixels. |

#### `class ROI`

Represents one rectangular region.

| Attribute | Type | Description |
|---|---|---|
| `id` | `str` | Unique 8-char UUID prefix. |
| `name` | `str` | User-assigned label (e.g. `"1"`, `"5a"`). |
| `x`, `y` | `int` | Top-left corner in image coordinates. |
| `w`, `h` | `int` | Width and height in pixels. |

**Methods:**

| Method | Signature | Returns |
|---|---|---|
| `to_dict()` | `() -> dict` | `{"id", "name", "x", "y", "w", "h"}` |
| `from_dict(d)` | `classmethod(dict) -> ROI` | Construct from dict. |
| `contains_image_point(px, py)` | `(int, int) -> bool` | Hit test. |
| `hit_handle(px, py)` | `(int, int) -> str \| None` | Returns handle name (`"n"`, `"se"`, etc.) or `None`. |

#### Functions

| Function | Signature | Purpose |
|---|---|---|
| `schema_path_for_image(image_path)` | `(Path) -> Path` | `SCHEMA_DIR / "<stem>.json"`. |
| `load_schema(image_path)` | `(Path) -> (list[dict], tuple\|None)` | Load ROIs + optional `(w, h)`. |
| `save_schema(image_path, rois, w, h)` | `(Path, list, int, int)` | Write schema JSON. |
| `list_normalized_images()` | `() -> list[str]` | Relative paths under `DEFAULT_BROWSE_ROOT`. |
| `browse_normalized(rel_dir)` | `(str) -> (list[str], list[str])` | List subdirs and `.png` files. |

#### Schema JSON shape

```json
{
    "image_path": "/absolute/path/to/image.png",
    "image_name": "image.png",
    "image_width": 1275,
    "image_height": 1650,
    "rois": [
        {"id": "a1b2c3d4", "name": "1", "x": 100, "y": 200, "w": 300, "h": 50},
        {"id": "e5f6g7h8", "name": "1a", "x": 110, "y": 260, "w": 30, "h": 30}
    ]
}
```

The `image_width` / `image_height` fields are used by `roi_page_module` to scale
coordinates when the normalized image has different dimensions than the image used
to author the schema.

---

### tools/roi_editor_web.py

Flask web server exposing the editor as a browser application.

#### `create_app() -> Flask`

Factory that registers all routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve `roi_editor.html`. |
| `/api/images` | GET | List all `.png` files under browse root. |
| `/api/browse?dir=<rel>` | GET | List subdirs and files in a directory. |
| `/api/image?path=<rel>` | GET | Serve an image file. |
| `/api/schema?path=<rel>` | GET | Load ROI schema for an image. |
| `/api/schema` | POST | Save ROI schema (`{path, rois, image_width, image_height}`). |

#### `run(port=5000, debug=False)`

Start the Flask dev server on `0.0.0.0`.

---

### tools/roi_editor.py — Tk GUI

`ROIEditorApp` is the main class. It provides:

- **Canvas** with the image rendered at a zoom/pan-adjusted scale.
- **ROI list panel** on the right with selection, double-click rename, and buttons.
- **Interaction modes**: click canvas background to start drawing a new ROI
  (rubber-band rectangle); click an existing ROI to select it; drag to move;
  drag handles to resize.
- **Zoom**: scroll wheel (anchored at cursor).
- **Pan**: middle-mouse drag.
- **Auto-save**: schema is written to disk on window close.

#### Key methods

| Method | Trigger | Action |
|---|---|---|
| `_load_image()` | Init | Open image with PIL, compute base scale. |
| `_load_schema()` | Init | Load existing ROIs from JSON. |
| `_redraw()` | Any change | Clear canvas, redraw image + all ROIs + handles. |
| `_on_press(event)` | Left click | Detect handle hit → resize; ROI hit → move; empty → create. |
| `_on_drag(event)` | Left drag | Update position/size per drag mode. |
| `_on_release(event)` | Left release | Finalize new ROI creation. |
| `_rename_selected()` | Button/dbl-click | Dialog to rename. |
| `_duplicate_selected()` | Button | Clone ROI with offset. |
| `_delete_selected()` | Button | Remove selected ROI (with confirmation). |
| `_save_schema()` | Window close | Persist all ROIs to JSON. |

---

## Pipeline debug — single item (`tools/analyze_pipeline_debug.py`)

`pdf_recognize.py --debug` writes a large `*_debug.json` with steps 1–6 (PDF paths, ID/form OCR+LLM, normalization, ROI traces, output paths, optional xlsx). To inspect **one page-pair** (one recognition `items[]` entry) without loading the whole file into an editor, slice it with:

```bash
python3 tools/analyze_pipeline_debug.py output/recognition/my_run_debug.json --item 0
python3 tools/analyze_pipeline_debug.py output/recognition/my_run_debug.json --item 2 \
  -o testing/output/item2_pipeline_debug.json
python3 tools/analyze_pipeline_debug.py output/recognition/my_run_debug.json --item 0 \
  --recognition output/recognition/my_run.json
```

| Argument | Meaning |
|---|---|
| `--item N` | 0-based index (same order as `items[]` in the recognition JSON). |
| `-o PATH` | Write JSON to a file; default is stdout. |
| `--recognition PATH` | Optional: merge `items[N]` from the recognition output into the result as top-level `"item"`. |

The output has `meta` (pdf stem, page numbers, item index) and `steps` with the same keys as the full debug file, scoped to the two pages (or one page if the PDF ended on an odd page).

### Web UI — tree viewer (`tools/pipeline_debug_web.py`)

Browser-based file picker under `output/recognition/`, item index (or odd page number), optional recognition JSON merge, and a **collapsible hierarchical tree** (JSON/XML-style inspector) plus raw JSON tab.

```bash
python3 tools/pipeline_debug_web.py [--port 5002]
```

Open `http://127.0.0.1:5002/`. Pick `*_debug.json`, set the **item** index (same as `items[]` in the recognition JSON), **Load slice**. Use **Tree view** for nested expand/collapse or **Raw JSON** for copy/paste.

---

## Human review — recognition JSON editor (`tools/human_review_web.py`)

Browser UI to open a **`output/recognition/<stem>.json`** file, step through **`human_review.pending`** with a **full-page image** (ROI outlined in green), correction field, and **save** back to disk. **Apply pending** runs `apply_manual_corrections()` so `corrected_id` / `corrected_text` on pending rows are merged into `items` (then save to persist). A **What to review** box at the top of the side panel gives plain-language instructions: for **ID** steps, confirm the student ID matches the scan; for **MCQ** and **text**, generic guidance to compare the page to **Predicted** and fix if needed. The main **page preview** (left) uses a canvas with **wheel zoom** and **left- or middle-drag pan**; on load it **auto-frames the ROI** and **dims the rest of the page** so the field region is obvious (see `tools/templates/human_review.html`).

**Saves** are only allowed under `output/recognition/` (same as the pipeline output).

```bash
python3 tools/human_review_web.py [--port 5003]
```

Open `http://127.0.0.1:5003/`. **Browse** starts under `output/recognition/`. Select a `.json`, use the queue / **Prev**/**Next**, **Save**. **Update XLSX** calls `POST /api/update-xlsx` and runs `fill_from_pipeline()` (same as `pdf_recognize` step 6) using the **current** JSON in the browser, with `apply_manual_corrections()` applied first so pending corrections are included in the export; output goes to `output/xlsx/<pdf_stem>.xlsx` per `XLSX_DATA_ENTRY` in `config.py`.

See `docs/ocr_human_review_schema.md` for the `human_review` / correction model.
