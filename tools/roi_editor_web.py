"""
Web server for the ROI editor. Run via: python tools/roi_editor.py --serve [--port 5000]
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

# Ensure we can import from tools and py
TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
import sys
if str(ROOT / "py") not in sys.path:
    sys.path.insert(0, str(ROOT / "py"))

from flask import Flask, send_file, send_from_directory, request, jsonify, Response, stream_with_context

import roi_editor_core as core
from roi_prompt_creator import (
    generate_roi_llm_prompt,
    generate_roi_llm_prompt_stream,
    generate_roi_ocr_prompt,
    generate_roi_ocr_prompt_stream,
    llm_generator_prompt_template_preview,
    ocr_generator_prompt_template_preview,
)
from roi_auto_detector import detect_rois_with_openai

SERVER_INSTANCE_ID = uuid.uuid4().hex


def _resolve_image_path(rel_path: str) -> Path | None:
    """Resolve relative path (from project root) to an absolute path under the project root."""
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
    app = Flask(__name__, static_folder=TOOLS_DIR / "static", template_folder=TOOLS_DIR / "templates")

    @app.route("/")
    def index():
        return send_file(TOOLS_DIR / "templates" / "roi_editor.html")

    @app.route("/api/instance")
    def api_instance():
        return jsonify({"instance_id": SERVER_INSTANCE_ID})

    @app.route("/api/images")
    def api_list_images():
        paths = core.list_normalized_images()
        return jsonify({"images": paths})

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
        return jsonify(
            {
                "rois": rois,
                "image_width": size[0] if size else None,
                "image_height": size[1] if size else None,
                "schema_path": core.schema_relpath_for_image(path),
            }
        )

    @app.route("/api/schema", methods=["POST"])
    def api_save_schema():
        data = request.get_json()
        if not data or "path" not in data:
            return jsonify({"error": "Missing path"}), 400
        path = _resolve_image_path(data["path"].strip())
        if not path:
            return jsonify({"error": "Invalid path"}), 400
        rois = data.get("rois", [])
        w = data.get("image_width", 0)
        h = data.get("image_height", 0)
        if not w or not h:
            return jsonify({"error": "Missing image_width/image_height"}), 400
        core.save_schema(path, rois, int(w), int(h))
        return jsonify({"ok": True})

    @app.route("/api/roi-prompt-generate", methods=["POST"])
    def api_roi_prompt_generate():
        body = request.get_json(silent=True) or {}
        instruction = str(body.get("instruction", "")).strip()
        if not instruction:
            return jsonify({"error": "instruction is required"}), 400
        roi_name = str(body.get("roi_name", "")).strip() or None
        field_data_type = str(body.get("field_data_type", "")).strip() or None
        validation_rules = str(body.get("validation_rules", "")).strip() or None
        existing_prompt = str(body.get("existing_prompt", "")).strip() or None
        try:
            result = generate_roi_llm_prompt(
                instruction,
                roi_name=roi_name,
                field_data_type=field_data_type,
                validation_rules=validation_rules,
                existing_prompt=existing_prompt,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Prompt generation failed: {exc}"}), 500
        return jsonify(
            {
                "ok": True,
                "prompt": result.get("prompt", ""),
                "elapsed": result.get("elapsed"),
                "model": result.get("model", ""),
            }
        )

    @app.route("/api/roi-prompt-generate-stream", methods=["POST"])
    def api_roi_prompt_generate_stream():
        body = request.get_json(silent=True) or {}
        instruction = str(body.get("instruction", "")).strip()
        if not instruction:
            return jsonify({"error": "instruction is required"}), 400
        roi_name = str(body.get("roi_name", "")).strip() or None
        field_data_type = str(body.get("field_data_type", "")).strip() or None
        validation_rules = str(body.get("validation_rules", "")).strip() or None
        existing_prompt = str(body.get("existing_prompt", "")).strip() or None

        def _stream():
            try:
                for event in generate_roi_llm_prompt_stream(
                    instruction,
                    roi_name=roi_name,
                    field_data_type=field_data_type,
                    validation_rules=validation_rules,
                    existing_prompt=existing_prompt,
                ):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except ValueError as exc:
                yield json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False) + "\n"
            except Exception as exc:
                yield json.dumps(
                    {"type": "error", "error": f"Prompt generation failed: {exc}"},
                    ensure_ascii=False,
                ) + "\n"

        return Response(stream_with_context(_stream()), mimetype="application/x-ndjson")

    @app.route("/api/roi-prompt-template")
    def api_roi_prompt_template():
        try:
            template = llm_generator_prompt_template_preview()
        except Exception as exc:
            return jsonify({"error": f"Failed to build template: {exc}"}), 500
        return jsonify({"ok": True, "template": template})

    @app.route("/api/roi-ocr-prompt-generate", methods=["POST"])
    def api_roi_ocr_prompt_generate():
        body = request.get_json(silent=True) or {}
        instruction = str(body.get("instruction", "")).strip()
        if not instruction:
            return jsonify({"error": "instruction is required"}), 400
        roi_name = str(body.get("roi_name", "")).strip() or None
        field_data_type = str(body.get("field_data_type", "")).strip() or None
        validation_rules = str(body.get("validation_rules", "")).strip() or None
        existing_prompt = str(body.get("existing_prompt", "")).strip() or None
        try:
            result = generate_roi_ocr_prompt(
                instruction,
                roi_name=roi_name,
                field_data_type=field_data_type,
                validation_rules=validation_rules,
                existing_prompt=existing_prompt,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"Prompt generation failed: {exc}"}), 500
        return jsonify(
            {
                "ok": True,
                "prompt": result.get("prompt", ""),
                "elapsed": result.get("elapsed"),
                "model": result.get("model", ""),
            }
        )

    @app.route("/api/roi-ocr-prompt-generate-stream", methods=["POST"])
    def api_roi_ocr_prompt_generate_stream():
        body = request.get_json(silent=True) or {}
        instruction = str(body.get("instruction", "")).strip()
        if not instruction:
            return jsonify({"error": "instruction is required"}), 400
        roi_name = str(body.get("roi_name", "")).strip() or None
        field_data_type = str(body.get("field_data_type", "")).strip() or None
        validation_rules = str(body.get("validation_rules", "")).strip() or None
        existing_prompt = str(body.get("existing_prompt", "")).strip() or None

        def _stream():
            try:
                for event in generate_roi_ocr_prompt_stream(
                    instruction,
                    roi_name=roi_name,
                    field_data_type=field_data_type,
                    validation_rules=validation_rules,
                    existing_prompt=existing_prompt,
                ):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except ValueError as exc:
                yield json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False) + "\n"
            except Exception as exc:
                yield json.dumps(
                    {"type": "error", "error": f"Prompt generation failed: {exc}"},
                    ensure_ascii=False,
                ) + "\n"

        return Response(stream_with_context(_stream()), mimetype="application/x-ndjson")

    @app.route("/api/roi-ocr-prompt-template")
    def api_roi_ocr_prompt_template():
        try:
            template = ocr_generator_prompt_template_preview()
        except Exception as exc:
            return jsonify({"error": f"Failed to build template: {exc}"}), 500
        return jsonify({"ok": True, "template": template})

    @app.route("/api/roi-auto-detect", methods=["POST"])
    def api_roi_auto_detect():
        body = request.get_json(silent=True) or {}
        rel = str(body.get("path", "")).strip()
        if not rel:
            return jsonify({"error": "path is required"}), 400
        path = _resolve_image_path(rel)
        if not path:
            return jsonify({"error": "Invalid path"}), 400

        model_profile = str(body.get("model_profile", "balanced") or "balanced").strip().lower()
        try:
            result = detect_rois_with_openai(path, model_profile=model_profile)
            rois = result.get("rois", [])
            w = int(result.get("image_width", 0) or 0)
            h = int(result.get("image_height", 0) or 0)
            if not w or not h:
                return jsonify({"error": "Detector returned invalid image size."}), 500
            core.save_schema(path, rois, w, h)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"error": f"ROI auto-detect failed: {exc}"}), 500

        return jsonify(
            {
                "ok": True,
                "path": rel,
                "rois": rois,
                "image_width": w,
                "image_height": h,
                "schema_path": core.schema_relpath_for_image(path),
                "reference_schema_path": result.get("reference_schema_path"),
                "ocr_line_count": result.get("ocr_line_count"),
                "model": result.get("model"),
            }
        )

    return app


def run(port: int = 5000, debug: bool = False) -> None:
    app = create_app()
    print(f"ROI Editor web server: http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ROI Editor web server")
    p.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    p.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = p.parse_args()
    run(port=args.port, debug=args.debug)
