"""
Read-only ROI previewer.  Shows each ROI rectangle and its name over a
selected image.  Run via:  python3 tools/roi_previewer_web.py [--port 5001]
"""

from __future__ import annotations

import uuid
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
import sys
if str(ROOT / "py") not in sys.path:
    sys.path.insert(0, str(ROOT / "py"))

from flask import Flask, send_file, request, jsonify

import roi_editor_core as core

SERVER_INSTANCE_ID = uuid.uuid4().hex


def _resolve_image_path(rel_path: str) -> Path | None:
    if not rel_path or ".." in rel_path or rel_path.startswith("/"):
        return None
    base = core.PROJECT_ROOT
    full = (base / rel_path).resolve()
    try:
        full.relative_to(base.resolve())
    except ValueError:
        return None
    return full if full.exists() else None


def create_app() -> Flask:
    app = Flask(__name__, static_folder=TOOLS_DIR / "static",
                template_folder=TOOLS_DIR / "templates")

    @app.route("/")
    def index():
        return send_file(TOOLS_DIR / "templates" / "roi_previewer.html")

    @app.route("/api/instance")
    def api_instance():
        return jsonify({"instance_id": SERVER_INSTANCE_ID})

    @app.route("/api/browse")
    def api_browse():
        rel_dir = request.args.get("dir", "").strip()
        dirs, files = core.browse_normalized(rel_dir)
        return jsonify({"dirs": dirs, "files": files})

    @app.route("/api/browse-root")
    def api_browse_root():
        return jsonify({"root": core.DEFAULT_BROWSE_ROOT})

    @app.route("/api/image")
    def api_get_image():
        rel = request.args.get("path", "").strip()
        path = _resolve_image_path(rel)
        if not path:
            return jsonify({"error": "Invalid or missing path"}), 400
        return send_file(path, mimetype="image/png")

    @app.route("/api/schema")
    def api_get_schema():
        rel = request.args.get("path", "").strip()
        path = _resolve_image_path(rel)
        if not path:
            return jsonify({"error": "Invalid or missing path"}), 400
        rois, size = core.load_schema(path)
        return jsonify({
            "rois": rois,
            "image_width": size[0] if size else None,
            "image_height": size[1] if size else None,
            "schema_path": core.schema_relpath_for_image(path),
        })

    return app


def run(port: int = 5001, debug: bool = False) -> None:
    app = create_app()
    print(f"ROI Previewer: http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ROI Previewer (read-only)")
    p.add_argument("--port", type=int, default=5001, help="Port (default: 5001)")
    p.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = p.parse_args()
    run(port=args.port, debug=args.debug)
