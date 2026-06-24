"""
Web UI: open a recognition JSON (output/recognition/*.json) and review
human_review.pending entries; corrections save back to disk.

Run:  python3 tools/human_review_web.py [--port 5003]

Requires: flask (same as other tools). Saves are allowed only under output/recognition/.
POST /api/update-xlsx regenerates output/xlsx/{stem}.xlsx via xlsx_data_entry.fill_from_pipeline.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
PY_DIR = ROOT / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename

# Same project root as the pipeline / recognition JSON (do not use a different ROOT).
try:
    from config import PROJECT_ROOT
except Exception:  # pragma: no cover
    PROJECT_ROOT = ROOT

try:
    from config import HEADER_RECOGNITION
except Exception:  # pragma: no cover
    HEADER_RECOGNITION = {"crop_top_percent": 15.0}

# Only allow writing recognition outputs here (relative to project root).
RECOGNITION_REL = Path("output") / "recognition"
DEFAULT_BROWSE_ROOT = RECOGNITION_REL
SERVER_INSTANCE_ID = uuid.uuid4().hex


def _resolve_read_path(rel_path: str) -> Path | None:
    """Resolve a project-relative path for reading JSON."""
    if not rel_path or ".." in rel_path or rel_path.startswith("/"):
        return None
    base = PROJECT_ROOT.resolve()
    full = (base / rel_path).resolve()
    try:
        full.relative_to(base)
    except ValueError:
        return None
    return full if full.is_file() and full.suffix.lower() == ".json" else None


def _resolve_write_path(rel_path: str) -> Path | None:
    """Resolve path for writing; must stay under output/recognition/."""
    if not rel_path or ".." in rel_path or rel_path.startswith("/"):
        return None
    base = PROJECT_ROOT.resolve()
    full = (base / rel_path).resolve()
    allowed = (base / RECOGNITION_REL).resolve()
    try:
        full.relative_to(allowed)
    except ValueError:
        return None
    if full.suffix.lower() != ".json":
        return None
    return full


def browse_json_dir(rel_dir: str = "") -> tuple[list[str], list[str]]:
    """List subdirs and .json files under PROJECT_ROOT/rel_dir (newest first)."""
    base_root = PROJECT_ROOT.resolve()
    rel_dir = (rel_dir or "").strip().rstrip("/")
    if not rel_dir:
        rel_dir = DEFAULT_BROWSE_ROOT.as_posix()
    if rel_dir and (".." in rel_dir or rel_dir.startswith("/")):
        return [], []
    target = (base_root / rel_dir).resolve() if rel_dir else base_root
    try:
        target.relative_to(base_root)
    except ValueError:
        return [], []
    if not target.is_dir():
        return [], []
    dir_entries: list[tuple[float, str]] = []
    file_entries: list[tuple[float, str]] = []
    for p in target.iterdir():
        try:
            rel = p.relative_to(base_root)
            name = str(rel).replace("\\", "/")
        except ValueError:
            continue
        try:
            mtime = float(p.stat().st_mtime)
        except OSError:
            mtime = 0.0
        if p.is_dir():
            dir_entries.append((mtime, name))
        elif p.suffix.lower() == ".json":
            file_entries.append((mtime, name))
    dir_entries.sort(key=lambda x: (-x[0], x[1].lower()))
    file_entries.sort(key=lambda x: (-x[0], x[1].lower()))
    return [name for _, name in dir_entries], [name for _, name in file_entries]


def _load_pending_review_context(rel: str, pidx: int) -> dict:
    """
    Resolve recognition JSON path + pending index to normalized page image and ROI metadata.

    Keys: page_abs (Path), kind (str), raw (dict | None), roi_name (str), pct (float).
    Raises ValueError with a user-facing message on failure.
    """
    from review_crops import load_schema_raw, safe_resolve_under_project

    path = _resolve_read_path(rel)
    if not path:
        raise ValueError("Invalid or missing path")
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(str(e)) from e
    if not isinstance(data, dict):
        raise ValueError("Root must be an object")
    pending = (data.get("human_review") or {}).get("pending") or []
    if pidx < 0 or pidx >= len(pending):
        raise ValueError("pending_index out of range")
    p = pending[pidx]
    if not isinstance(p, dict):
        raise ValueError("bad pending row")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be an array")
    try:
        iidx = int(p.get("item_index", -1))
    except (TypeError, ValueError):
        raise ValueError("bad item_index") from None
    if iidx < 0 or iidx >= len(items):
        raise ValueError("item_index out of range")
    item = items[iidx]
    if not isinstance(item, dict):
        raise ValueError("bad item")

    root = PROJECT_ROOT.resolve()
    kind = str(p.get("kind", "")).lower()
    pct = float(HEADER_RECOGNITION.get("crop_top_percent", 8.0))

    if kind == "id":
        rel_img = item.get("normalized_page_odd")
        page_abs = safe_resolve_under_project(root, str(rel_img or ""))
        if not page_abs:
            raise ValueError(
                "Cannot open ID page image (need items[].normalized_page_odd). "
                f"Tried: {rel_img!r}"
            )
        return {
            "page_abs": page_abs,
            "kind": kind,
            "raw": None,
            "roi_name": str(p.get("roi_name", "")),
            "pct": pct,
        }

    if kind in ("text", "mcq"):
        page_in_pair = str(p.get("page_in_pair") or "odd")
        side = "a" if page_in_pair == "odd" else "b"
        if kind == "mcq":
            rel_img = (
                item.get("normalized_page_odd_mcq")
                if page_in_pair == "odd"
                else item.get("normalized_page_even_mcq")
            )
            if not rel_img:
                rel_img = (
                    item.get("normalized_page_odd")
                    if page_in_pair == "odd"
                    else item.get("normalized_page_even")
                )
        else:
            rel_img = (
                item.get("normalized_page_odd")
                if page_in_pair == "odd"
                else item.get("normalized_page_even")
            )
        page_abs = safe_resolve_under_project(root, str(rel_img or ""))
        if not page_abs:
            raise ValueError(
                "Cannot open normalized page for ROI crop. " f"Tried: {rel_img!r}"
            )
        raw = load_schema_raw(root, item.get("form_type"), side)
        if not raw:
            raise ValueError(
                f"No ROI schema for form_type={item.get('form_type')!r} side={side}"
            )
        return {
            "page_abs": page_abs,
            "kind": kind,
            "raw": raw,
            "roi_name": str(p.get("roi_name", "")),
            "pct": pct,
        }

    raise ValueError("unsupported kind")


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=TOOLS_DIR / "static",
        template_folder=TOOLS_DIR / "templates",
    )

    @app.route("/")
    def index():
        return send_file(TOOLS_DIR / "templates" / "human_review.html")

    @app.route("/api/instance")
    def api_instance():
        return jsonify({"instance_id": SERVER_INSTANCE_ID})

    @app.route("/api/browse")
    def api_browse():
        rel_dir = request.args.get("dir", "").strip()
        dirs, files = browse_json_dir(rel_dir)
        return jsonify({"dirs": dirs, "files": files})

    @app.route("/api/upload-json", methods=["POST"])
    def api_upload_json():
        up = request.files.get("file")
        if not up or not up.filename:
            return jsonify({"error": "Missing file"}), 400
        name = secure_filename(up.filename)
        if not name.lower().endswith(".json"):
            return jsonify({"error": "Only JSON files are supported"}), 400
        upload_dir = PROJECT_ROOT / "testing" / "output" / "web_tools" / "review_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / name
        up.save(str(save_path))
        try:
            rel = save_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(save_path)
        return jsonify({"ok": True, "path": rel, "filename": name})

    @app.route("/api/load")
    def api_load():
        rel = request.args.get("path", "").strip()
        path = _resolve_read_path(rel)
        if not path:
            return jsonify({"error": "Invalid or missing path (need .json under project)"}), 400
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON: {e}"}), 400
        if not isinstance(data, dict):
            return jsonify({"error": "Root must be an object"}), 400
        items = data.get("items")
        if items is not None and not isinstance(items, list):
            return jsonify({"error": "items must be an array"}), 400
        return jsonify(
            {
                "ok": True,
                "path": rel,
                "data": data,
            }
        )

    @app.route("/api/save", methods=["POST"])
    def api_save():
        body = request.get_json(silent=True) or {}
        rel = (body.get("path") or "").strip()
        path = _resolve_write_path(rel)
        if not path:
            return jsonify(
                {"error": "Save path must be a .json file under output/recognition/"},
            ),
            400
        payload = body.get("data")
        if not isinstance(payload, dict):
            return jsonify({"error": "data must be a JSON object"}), 400
        items = payload.get("items")
        if items is not None and not isinstance(items, list):
            return jsonify({"error": "items must be an array"}), 400
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        path.write_text(text, encoding="utf-8")
        return jsonify({"ok": True, "path": rel, "bytes": len(text.encode("utf-8"))})

    @app.route("/api/apply-pending", methods=["POST"])
    def api_apply_pending():
        """Merge human_review.pending corrected_* into items (server-side)."""
        try:
            from ocr_human_review import apply_manual_corrections
        except ImportError as e:
            return jsonify({"error": f"ocr_human_review: {e}"}), 500
        body = request.get_json(silent=True) or {}
        items = body.get("items")
        hr = body.get("human_review")
        if not isinstance(items, list):
            return jsonify({"error": "items must be an array"}), 400
        if hr is not None and not isinstance(hr, dict):
            return jsonify({"error": "human_review must be an object"}), 400
        if not hr:
            return jsonify({"ok": True, "items": items})
        merged = apply_manual_corrections(items, hr)
        return jsonify({"ok": True, "items": merged})

    @app.route("/api/update-xlsx", methods=["POST"])
    def api_update_xlsx():
        """
        Regenerate the data-entry workbook from recognition JSON (same as pipeline step 6).

        Request JSON: ``{ "path": "<rel to project .json>", "data": { ... } }``.
        If ``data`` is omitted, loads ``path`` from disk. ``items`` are merged with
        ``apply_manual_corrections`` so pending human_review corrections are reflected.
        Writes to ``XLSX_DATA_ENTRY["output_dir"]/<pdf_stem>.xlsx`` (see ``fill_from_pipeline``).
        """
        body = request.get_json(silent=True) or {}
        rel = (body.get("path") or "").strip()
        read_path = _resolve_read_path(rel)
        if not read_path:
            return jsonify(
                {"error": "Invalid or missing path (need .json under project)"},
            ), 400

        data = body.get("data")
        if data is None:
            try:
                with read_path.open(encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                return jsonify({"error": str(e)}), 400
        elif not isinstance(data, dict):
            return jsonify({"error": "data must be a JSON object"}), 400

        items = data.get("items")
        if not isinstance(items, list):
            return jsonify({"error": "items must be an array"}), 400

        try:
            from ocr_human_review import apply_manual_corrections
            from xlsx_data_entry import fill_from_pipeline
        except ImportError as e:
            return jsonify({"error": str(e)}), 500

        hr = data.get("human_review")
        merged = apply_manual_corrections(items, hr if isinstance(hr, dict) else {})

        pdf_stem = data.get("pdf_stem")
        if not pdf_stem:
            pdf_stem = read_path.stem

        original_items = data.get("items_original")
        if not isinstance(original_items, list):
            original_items = None

        try:
            source_file_name = ""
            pdf_path_value = data.get("pdf_path")
            if pdf_path_value:
                source_file_name = Path(str(pdf_path_value)).name
            out = fill_from_pipeline(
                merged,
                pdf_stem=str(pdf_stem),
                source_file_name=source_file_name,
                verbose=False,
                original_items=original_items,
            )
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        if out is None:
            return jsonify(
                {
                    "ok": True,
                    "xlsx_path": None,
                    "message": "No workbook written (no items with form_type, or no mapping).",
                }
            )

        root = PROJECT_ROOT.resolve()
        try:
            rel_xlsx = out.resolve().relative_to(root).as_posix()
        except ValueError:
            rel_xlsx = str(out)

        return jsonify({"ok": True, "xlsx_path": rel_xlsx, "message": f"Wrote {rel_xlsx}"})

    @app.route("/api/review-crop")
    def api_review_crop():
        """PNG crop for human_review.pending[pending_index] (ID top band or ROI)."""
        rel = request.args.get("path", "").strip()
        try:
            pidx = int(request.args.get("pending_index", "-1"))
        except ValueError:
            return jsonify({"error": "bad pending_index"}), 400
        try:
            from review_crops import png_bytes_id_top_crop, png_bytes_roi_crop
        except ImportError as e:
            return jsonify({"error": f"review_crops: {e}"}), 500
        try:
            ctx = _load_pending_review_context(rel, pidx)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        kind = ctx["kind"]
        page_abs = ctx["page_abs"]
        if kind == "id":
            png = png_bytes_id_top_crop(page_abs, ctx["pct"])
        else:
            png = png_bytes_roi_crop(page_abs, ctx["raw"], ctx["roi_name"])
            if not png:
                return jsonify(
                    {"error": f"ROI {ctx['roi_name']!r} not found in schema or empty crop."}
                ), 400

        if not png:
            return jsonify(
                {"error": "OpenCV could not read image or produced an empty crop."}
            ), 400
        return Response(png, mimetype="image/png")

    @app.route("/api/review-page-full")
    def api_review_page_full():
        """Full normalized page as PNG (grayscale) for context preview."""
        rel = request.args.get("path", "").strip()
        try:
            pidx = int(request.args.get("pending_index", "-1"))
        except ValueError:
            return jsonify({"error": "bad pending_index"}), 400
        try:
            from review_crops import png_bytes_full_page
        except ImportError as e:
            return jsonify({"error": f"review_crops: {e}"}), 500
        try:
            ctx = _load_pending_review_context(rel, pidx)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        png = png_bytes_full_page(ctx["page_abs"])
        if not png:
            return jsonify({"error": "Could not read full page image."}), 400
        return Response(png, mimetype="image/png")

    @app.route("/api/review-page-roi")
    def api_review_page_roi():
        """ROI rectangle on the full page in image pixel coordinates (for overlay)."""
        rel = request.args.get("path", "").strip()
        try:
            pidx = int(request.args.get("pending_index", "-1"))
        except ValueError:
            return jsonify({"error": "bad pending_index"}), 400
        try:
            from review_crops import roi_rect_xyxy_in_page_pixels
        except ImportError as e:
            return jsonify({"error": f"review_crops: {e}"}), 500
        try:
            ctx = _load_pending_review_context(rel, pidx)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        rect = roi_rect_xyxy_in_page_pixels(
            ctx["page_abs"],
            ctx["raw"],
            ctx["roi_name"],
            ctx["kind"],
            ctx["pct"],
        )
        if not rect:
            return jsonify({"error": "Could not compute ROI rectangle."}), 400
        x1, y1, x2, y2, iw, ih = rect
        return jsonify(
            {
                "ok": True,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "iw": iw,
                "ih": ih,
            }
        )

    return app


def run(port: int = 5003, debug: bool = False) -> None:
    app = create_app()
    print(f"Human review editor: http://127.0.0.1:{port}/")
    print(f"  (saves allowed only under {RECOGNITION_REL.as_posix()}/)")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Human review web editor for recognition JSON")
    p.add_argument("--port", type=int, default=5003, help="Port (default: 5003)")
    p.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = p.parse_args()
    run(port=args.port, debug=args.debug)
