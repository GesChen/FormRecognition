"""
Web UI for pipeline debug inspection: pick a *_debug.json, choose an item index,
view the sliced trace in a hierarchical tree (JSON/XML-style viewer).

Run:  python3 tools/pipeline_debug_web.py [--port 5002]
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from flask import Flask, jsonify, request, send_file
from werkzeug.utils import secure_filename

try:
    from analyze_pipeline_debug import extract_item_pipeline_debug, load_recognition_item
except ImportError:  # pragma: no cover
    extract_item_pipeline_debug = None  # type: ignore

# Reuse project root resolution like roi_editor_core
try:
    from roi_editor_core import PROJECT_ROOT
except Exception:  # pragma: no cover
    PROJECT_ROOT = ROOT

SERVER_INSTANCE_ID = uuid.uuid4().hex
DEFAULT_BROWSE_ROOT = Path("output") / "recognition"


def _resolve_project_path(rel_path: str) -> Path | None:
    if not rel_path or ".." in rel_path or rel_path.startswith("/"):
        return None
    base = PROJECT_ROOT.resolve()
    full = (base / rel_path).resolve()
    try:
        full.relative_to(base)
    except ValueError:
        return None
    return full if full.is_file() else None


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


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=TOOLS_DIR / "static",
        template_folder=TOOLS_DIR / "templates",
    )

    @app.route("/")
    def index():
        return send_file(TOOLS_DIR / "templates" / "pipeline_debug.html")

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
        upload_dir = PROJECT_ROOT / "testing" / "output" / "web_tools" / "debug_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / name
        up.save(str(save_path))
        try:
            rel = save_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(save_path)
        return jsonify({"ok": True, "path": rel, "filename": name})

    @app.route("/api/debug-meta")
    def api_debug_meta():
        rel = request.args.get("path", "").strip()
        path = _resolve_project_path(rel)
        if not path:
            return jsonify({"error": "Invalid or missing path"}), 400
        if extract_item_pipeline_debug is None:
            return jsonify({"error": "analyze_pipeline_debug not importable"}), 500
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON: {e}"}), 400
        if not isinstance(data, dict):
            return jsonify({"error": "Root must be an object"}), 400
        steps = data.get("steps") or {}
        s1 = steps.get("1_pdf_to_images") or {}
        page_paths = s1.get("page_paths") or []
        n_pages = len(page_paths)
        output_step = steps.get("6_output") or steps.get("5_output") or {}
        item_count = output_step.get("item_count")
        if item_count is None and n_pages:
            item_count = (n_pages + 1) // 2
        try:
            item_count = int(item_count) if item_count is not None else 0
        except (TypeError, ValueError):
            item_count = 0
        return jsonify(
            {
                "pdf_stem": data.get("pdf_stem"),
                "pdf_path": data.get("pdf_path"),
                "n_pages": n_pages,
                "item_count": item_count,
                "item_index_max": max(0, item_count - 1),
            }
        )

    @app.route("/api/slice", methods=["POST"])
    def api_slice():
        if extract_item_pipeline_debug is None:
            return jsonify({"error": "analyze_pipeline_debug not importable"}), 500
        body = request.get_json(silent=True) or {}
        rel = (body.get("path") or "").strip()
        path = _resolve_project_path(rel)
        if not path:
            return jsonify({"error": "Invalid or missing path"}), 400
        try:
            item_index = int(body.get("item_index", 0))
        except (TypeError, ValueError):
            return jsonify({"error": "item_index must be an integer"}), 400

        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Invalid JSON: {e}"}), 400
        if not isinstance(data, dict):
            return jsonify({"error": "Root must be an object"}), 400
        try:
            out = extract_item_pipeline_debug(data, item_index, source_path=str(path))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        rec_rel = (body.get("recognition_path") or "").strip()
        if rec_rel:
            rec_path = _resolve_project_path(rec_rel)
            if rec_path:
                merged = load_recognition_item(rec_path, item_index)
                if merged is not None:
                    out["item"] = merged

        return jsonify({"ok": True, "data": out})

    return app


def run(port: int = 5002, debug: bool = False) -> None:
    app = create_app()
    print(f"Pipeline debug viewer: http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pipeline debug web viewer")
    p.add_argument("--port", type=int, default=5002, help="Port (default: 5002)")
    p.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = p.parse_args()
    run(port=args.port, debug=args.debug)
