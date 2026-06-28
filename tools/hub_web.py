#!/usr/bin/env python3
"""
Project hub: PDF recognition launcher + links to other web tools.

Run from repo root:
  python3 tools/hub_web.py [--port 4999]

Upload a PDF or use a project-relative path; streams CLI output (like a terminal).
"""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import signal
import subprocess
import sys
import threading
import traceback
import uuid
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
PY_DIR = ROOT / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.utils import secure_filename
import fitz

try:
    from config import PDF_RECOGNITION, PROJECT_ROOT, XLSX_DATA_ENTRY
except Exception:  # pragma: no cover
    PROJECT_ROOT = ROOT
    PDF_RECOGNITION = {"output_dir": ROOT / "output" / "recognition"}
    XLSX_DATA_ENTRY = {"output_dir": ROOT / "output" / "xlsx"}

UPLOAD_DIR = Path(PDF_RECOGNITION.get("output_dir", ROOT / "output" / "recognition")).parent / "uploads"
MAX_UPLOAD_BYTES = 250 * 1024 * 1024

# Subprocess env: unbuffered Python so chunks (incl. tqdm \\r updates) reach the hub promptly.
_TQDM_STREAM_ENV = {
    "PYTHONUNBUFFERED": "1",
    # Force tqdm on for hub-streamed jobs; hub client emulates terminal CR refresh.
    "EVMS_TQDM": "1",
}

_jobs_lock = threading.Lock()
# job_id -> Popen for in-flight pdf_recognize runs (cancel stops the whole process group).
_active_recognition_jobs: dict[str, subprocess.Popen[bytes]] = {}
SERVER_INSTANCE_ID = uuid.uuid4().hex


def _terminate_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Stop the pipeline process and (on POSIX) its whole process group."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    try:
        if sys.platform == "win32":
            proc.terminate()
        else:
            try:
                gid = os.getpgid(pid)
                os.killpg(gid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                proc.terminate()
    except ProcessLookupError:
        pass
    except OSError:
        try:
            proc.terminate()
        except OSError:
            pass

def _normalize_tty_stream_chunk(text: str) -> str:
    """Normalize CRLF to LF; keep ANSI + lone \\r so the web client can emulate terminal redraw."""
    return text.replace("\r\n", "\n")


def _iter_subprocess_merged_output(proc: subprocess.Popen[bytes]) -> str:
    """Yield decoded stdout+stderr chunks (binary read) so \\r-only tqdm lines flush to the client."""
    assert proc.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            # read1: return pipe data as soon as available (helps tqdm \\r updates flush).
            read_fn = getattr(proc.stdout, "read1", None)
            chunk = read_fn(8192) if callable(read_fn) else proc.stdout.read(8192)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                yield _normalize_tty_stream_chunk(text)
        tail = decoder.decode(b"", final=True)
        if tail:
            yield _normalize_tty_stream_chunk(tail)
    finally:
        proc.stdout.close()


def _pdf_stem_from_name(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^\w\-.]", "_", stem).strip("_") or "pdf"


def _sanitize_output_suffix(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    safe = re.sub(r"[^\w\-.]", "_", raw).strip("_")
    if not safe:
        return ""
    return f"_{safe}"


def _sanitize_output_filename(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    name = Path(raw.replace("\\", "/")).name
    stem = Path(name).stem
    safe = re.sub(r"[^\w\-.]", "_", stem).strip("_")
    return safe or ""


def _resolve_pdf_under_project(rel: str) -> Path | None:
    rel = (rel or "").strip().replace("\\", "/")
    if not rel or ".." in rel or rel.startswith("/"):
        return None
    base = PROJECT_ROOT.resolve()
    full = (base / rel).resolve()
    try:
        full.relative_to(base)
    except ValueError:
        return None
    if not full.is_file() or full.suffix.lower() != ".pdf":
        return None
    return full


def _build_merged_upload_name(files: list) -> str:
    """Create merged filename from per-file acronyms plus a short unique suffix."""
    acronyms: list[str] = []
    for up in files:
        name = getattr(up, "filename", "") or ""
        stem = Path(name).stem
        tokens = re.findall(r"[A-Za-z0-9]+", stem)
        if not tokens:
            continue
        parts: list[str] = []
        for t in tokens:
            if t.isdigit():
                parts.append(t)
            else:
                parts.append(t[0].upper())
        ac = "".join(parts)
        ac = re.sub(r"[^A-Z0-9]", "", ac).strip()
        if ac:
            acronyms.append(ac)
    if not acronyms:
        return f"merged_{uuid.uuid4().hex[:8]}.pdf"
    preview = "_".join(acronyms[:8])
    if len(acronyms) > 8:
        preview += f"_P{len(acronyms) - 8}"
    preview = preview[:80].strip("_") or "MERGED"
    return f"merged_{preview}_{uuid.uuid4().hex[:6]}.pdf"


def _project_rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _merge_uploaded_pdfs_to_one(files: list, merged_path: Path) -> dict[str, object]:
    """Merge uploaded PDF file objects into one PDF at merged_path and write source-page metadata."""
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    out_doc = fitz.open()
    manifest: dict[str, object] = {
        "version": 1,
        "merged_pdf_path": _project_rel_or_abs(merged_path),
        "sources": [],
        "pages": [],
    }
    try:
        merged_page = 1
        for source_index, up in enumerate(files):
            if not up or not up.filename:
                continue
            if not up.filename.lower().endswith(".pdf"):
                raise ValueError(f"File must be a .pdf: {up.filename}")
            original_name = secure_filename(up.filename) or f"source_{source_index + 1}.pdf"
            if not original_name.lower().endswith(".pdf"):
                original_name += ".pdf"
            source_path = merged_path.parent / f"{merged_path.stem}__src{source_index + 1:02d}_{original_name}"
            data = up.read()
            source_path.write_bytes(data)
            src = fitz.open(stream=data, filetype="pdf")
            try:
                page_count = int(src.page_count)
                source_row = {
                    "source_index": source_index,
                    "source_file_name": up.filename,
                    "stored_pdf_path": _project_rel_or_abs(source_path),
                    "page_count": page_count,
                    "merged_page_start": merged_page,
                    "merged_page_end": merged_page + page_count - 1,
                }
                manifest_sources = manifest.setdefault("sources", [])
                if isinstance(manifest_sources, list):
                    manifest_sources.append(source_row)
                manifest_pages = manifest.setdefault("pages", [])
                if isinstance(manifest_pages, list):
                    for source_page in range(1, page_count + 1):
                        manifest_pages.append(
                            {
                                "merged_page": merged_page,
                                "source_index": source_index,
                                "source_file_name": up.filename,
                                "source_pdf_path": _project_rel_or_abs(source_path),
                                "source_page": source_page,
                            }
                        )
                        merged_page += 1
                out_doc.insert_pdf(src)
            finally:
                src.close()
        if out_doc.page_count <= 0:
            raise ValueError("No valid PDF pages were provided")
        out_doc.save(str(merged_path))
        manifest_path = merged_path.with_suffix(".sources.json")
        manifest["manifest_path"] = _project_rel_or_abs(manifest_path)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    finally:
        out_doc.close()
    return manifest


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=TOOLS_DIR / "static",
        template_folder=TOOLS_DIR / "templates",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    @app.route("/")
    def index():
        from flask import render_template

        return render_template("hub.html")

    @app.route("/api/instance")
    def api_instance():
        return jsonify({"instance_id": SERVER_INSTANCE_ID})

    @app.route("/api/run-recognition", methods=["POST"])
    def api_run_recognition():
        """Stream stdout/stderr of pdf_recognize.py; footer line __RESULT__{json}."""
        pdf_abs: Path | None = None
        upload_rel: str | None = None
        used_name_for_stem: str

        rel_path = (request.form.get("pdf_rel_path") or "").strip()
        uploads = [u for u in request.files.getlist("pdf") if u and u.filename]
        if not uploads:
            one = request.files.get("pdf")
            if one and one.filename:
                uploads = [one]

        if uploads:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            if len(uploads) == 1:
                up = uploads[0]
                if not up.filename.lower().endswith(".pdf"):
                    return jsonify({"error": "File must be a .pdf"}), 400
                safe = secure_filename(up.filename) or "upload.pdf"
                if not safe.lower().endswith(".pdf"):
                    safe = safe + ".pdf"
                pdf_abs = UPLOAD_DIR / safe
                up.save(str(pdf_abs))
                used_name_for_stem = safe
            else:
                merged_name = _build_merged_upload_name(uploads)
                pdf_abs = UPLOAD_DIR / merged_name
                try:
                    _merge_uploaded_pdfs_to_one(uploads, pdf_abs)
                except Exception as exc:
                    return jsonify({"error": f"Failed to merge uploaded PDFs: {exc}"}), 400
                used_name_for_stem = merged_name
            try:
                upload_rel = str(pdf_abs.resolve().relative_to(PROJECT_ROOT.resolve()))
            except ValueError:
                upload_rel = str(pdf_abs)
        elif rel_path:
            pdf_abs = _resolve_pdf_under_project(rel_path)
            if not pdf_abs:
                return jsonify(
                    {"error": "Invalid path or not a PDF under project root: " + rel_path},
                ), 400
            used_name_for_stem = pdf_abs.name
        else:
            return jsonify({"error": "Provide a PDF file or a project-relative pdf path."}), 400

        script = PY_DIR / "pdf_recognize.py"
        if not script.is_file():
            return jsonify({"error": f"Missing {script}"}), 500

        cmd: list[str] = [sys.executable, "-u", str(script), str(pdf_abs)]
        if request.form.get("no_json"):
            cmd.append("--no-json")
        # Web CLI should stream progress by default (tqdm + verbose step logs).
        # Accept explicit falsey values to disable when needed.
        verbose_raw = (request.form.get("verbose") or "").strip().lower()
        verbose_enabled = True
        if verbose_raw in {"0", "false", "off", "no"}:
            verbose_enabled = False
        if verbose_enabled:
            cmd.append("--verbose")
        max_pages = (request.form.get("max_pages") or "").strip()
        if max_pages:
            try:
                n = int(max_pages)
                if n > 0:
                    cmd.extend(["--max-pages", str(n)])
            except ValueError:
                pass
        if request.form.get("recache"):
            cmd.append("--recache")
        if request.form.get("debug"):
            cmd.append("--debug")
        debug_path = (request.form.get("debug_path") or "").strip()
        if debug_path:
            cmd.extend(["--debug-path", debug_path])
        output_suffix_raw = (request.form.get("output_suffix") or "").strip()
        if output_suffix_raw:
            cmd.extend(["--output-suffix", output_suffix_raw])

        output_filename_raw = (request.form.get("output_filename") or "").strip()
        output_filename_stem = ""
        if output_filename_raw:
            if len(uploads) <= 1:
                return jsonify({"error": "Custom output filename is only available for merged PDF uploads."}), 400
            output_filename_stem = _sanitize_output_filename(output_filename_raw)
            if not output_filename_stem:
                return jsonify({"error": "Custom output filename must include at least one valid filename character."}), 400
            cmd.extend(["--output-filename", output_filename_raw])

        stem = _pdf_stem_from_name(used_name_for_stem)
        output_stem = output_filename_stem or f"{stem}{_sanitize_output_suffix(output_suffix_raw)}"
        out_dir = Path(PDF_RECOGNITION.get("output_dir", ROOT / "output" / "recognition"))
        xlsx_dir = Path(XLSX_DATA_ENTRY.get("output_dir", ROOT / "output" / "xlsx"))
        json_rel = str((out_dir / f"{output_stem}.json").resolve().relative_to(PROJECT_ROOT.resolve()))
        xlsx_rel = str((xlsx_dir / f"{output_stem}.xlsx").resolve().relative_to(PROJECT_ROOT.resolve()))

        env = {**os.environ, **_TQDM_STREAM_ENV}
        job_id = str(uuid.uuid4())

        def generate():
            proc: subprocess.Popen[bytes] | None = None
            code: int | None = None
            stream_error: str | None = None
            stream_traceback: str | None = None
            cancelled = False
            try:
                yield f"$ {' '.join(cmd)}\n\n"
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(PROJECT_ROOT.resolve()),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    env=env,
                    start_new_session=True,
                )
                with _jobs_lock:
                    _active_recognition_jobs[job_id] = proc
                try:
                    for piece in _iter_subprocess_merged_output(proc):
                        yield piece
                except (BrokenPipeError, OSError):
                    cancelled = True
                    _terminate_process_tree(proc)
                    return
                finally:
                    if proc.stdout and not proc.stdout.closed:
                        try:
                            proc.stdout.close()
                        except OSError:
                            pass
                code = proc.wait()
            except Exception as exc:
                stream_error = f"{type(exc).__name__}: {exc}"
                stream_traceback = traceback.format_exc()
                if proc is not None and proc.poll() is None:
                    _terminate_process_tree(proc)
                    code = proc.wait()
                elif proc is not None:
                    code = proc.poll()
                else:
                    code = -1
                yield "\n--- Hub stream error ---\n"
                yield stream_traceback
            finally:
                try:
                    pdf_used_rel = str(pdf_abs.resolve().relative_to(PROJECT_ROOT.resolve()))
                except ValueError:
                    pdf_used_rel = str(pdf_abs.resolve())
                result = {
                    "exit_code": code if code is not None else -1,
                    "stem": output_stem,
                    "output_suffix": output_suffix_raw,
                    "output_filename": output_filename_stem,
                    "json_rel": json_rel,
                    "xlsx_rel": xlsx_rel,
                    "upload_rel": upload_rel,
                    "pdf_rel_path_used": pdf_used_rel,
                    "no_json": bool(request.form.get("no_json")),
                    "cancelled": bool(cancelled or (code is not None and code < 0)),
                }
                if stream_error:
                    result["stream_error"] = stream_error
                try:
                    yield "\n__RESULT__" + json.dumps(result) + "\n"
                except (BrokenPipeError, OSError):
                    if proc is not None and proc.poll() is None:
                        _terminate_process_tree(proc)
                with _jobs_lock:
                    _active_recognition_jobs.pop(job_id, None)

        resp = Response(
            stream_with_context(generate()),
            mimetype="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-store",
                "X-Accel-Buffering": "no",
                "X-Job-Id": job_id,
            },
        )
        return resp

    @app.route("/api/cancel-recognition", methods=["POST"])
    def api_cancel_recognition():
        body = request.get_json(silent=True) or {}
        jid = str(body.get("job_id", "")).strip()
        if not jid:
            return jsonify({"ok": False, "error": "Missing job_id"}), 400
        with _jobs_lock:
            proc = _active_recognition_jobs.pop(jid, None)
        if proc is None:
            return jsonify({"ok": False, "error": "Job not found or already finished"}), 404
        _terminate_process_tree(proc)
        return jsonify({"ok": True})

    return app


def main() -> None:
    p = argparse.ArgumentParser(description="Web tools hub (PDF recognition + tool links)")
    p.add_argument("--port", type=int, default=4999, help="Port (default: 4999)")
    p.add_argument("--debug", action="store_true", help="Flask debug")
    args = p.parse_args()
    app = create_app()
    print(f"Web tools hub: http://127.0.0.1:{args.port}/")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
