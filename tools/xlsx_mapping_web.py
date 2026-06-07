#!/usr/bin/env python3
"""Web editor/previewer for XLSX mapping JSON files."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
PY_DIR = ROOT / "py"
if str(PY_DIR) not in sys.path:
    sys.path.insert(0, str(PY_DIR))

from flask import Flask, jsonify, request, send_file

from form_sheet_map import form_sheet_name

try:
    from config import DATA_RELEASE, PATHS, PROJECT_ROOT, XLSX_MAPPING_AUTO_GENERATE
except Exception:  # pragma: no cover
    PROJECT_ROOT = ROOT
    DATA_RELEASE = "2025"
    PATHS = {
        "data": ROOT / "data",
        "xlsx_mappings_root": ROOT / "data" / "xlsx" / "mappings" / DATA_RELEASE,
        "roi_schemas_root": ROOT / "data" / "roi_schemas" / DATA_RELEASE,
    }
    XLSX_MAPPING_AUTO_GENERATE = {}

from xlsx_mapping_auto_generator import generate_xlsx_mapping

SERVER_INSTANCE_ID = uuid.uuid4().hex
FORM_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
COLUMN_RE = re.compile(r"^[A-Z]{1,3}$")
VALID_TYPES = {"direct", "lookup", "multi_column", "static"}
VALID_TRANSFORMS = {"", "upper", "lower", "number", "date"}
RESERVED_SOURCES = {"id", "form_type", "page_odd", "page_even"}


def _resolve_collection_root(path_value: Path | str, expected_leaf: str) -> Path:
    p = Path(path_value).resolve()
    if p.name == expected_leaf:
        return p
    if p.parent.name == expected_leaf:
        return p.parent
    return p


DATA_ROOT = Path(PATHS.get("data", ROOT / "data")).resolve()
MAPPINGS_ROOT = _resolve_collection_root(
    PATHS.get("xlsx_mappings_root", DATA_ROOT / "xlsx" / "mappings"),
    "mappings",
)
ROI_SCHEMAS_ROOT = _resolve_collection_root(
    PATHS.get("roi_schemas_root", DATA_ROOT / "roi_schemas"),
    "roi_schemas",
)
TEMPLATES_ROOT = _resolve_collection_root(
    PATHS.get("templates_root", DATA_ROOT / "templates"),
    "templates",
)
UPLOADS_DIR = Path(PROJECT_ROOT).resolve() / "output" / "uploads"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path(PROJECT_ROOT).resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _safe_release(value: str | None) -> str:
    release = str(value or DATA_RELEASE or "").strip()
    if not release or not FORM_NAME_RE.match(release):
        raise ValueError("Invalid release")
    return release


def _safe_form(value: str | None) -> str:
    form = str(value or "").strip()
    if not form or not FORM_NAME_RE.match(form):
        raise ValueError("Invalid form type")
    return form


def _mapping_path(release: str, form: str) -> Path:
    path = (MAPPINGS_ROOT / release / f"{form}.json").resolve()
    path.relative_to(MAPPINGS_ROOT.resolve())
    return path


def _resolve_mapping_relpath(rel_path: str) -> Path | None:
    rel = str(rel_path or "").strip().replace("\\", "/")
    if not rel or ".." in rel or rel.startswith("/"):
        return None
    full = (Path(PROJECT_ROOT).resolve() / rel).resolve()
    try:
        full.relative_to(MAPPINGS_ROOT.resolve())
    except ValueError:
        return None
    return full if full.is_file() and full.suffix.lower() == ".json" else None


def _mapping_info_from_path(path: Path) -> tuple[str, str]:
    rel = path.resolve().relative_to(MAPPINGS_ROOT.resolve())
    if len(rel.parts) < 2:
        raise ValueError("Mapping path must be inside a release directory")
    release = _safe_release(rel.parts[0])
    form = _safe_form(Path(rel.parts[-1]).stem)
    return release, form


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_releases() -> list[str]:
    ignored = {"example", "examples", ".backups"}
    releases: set[str] = set()
    if MAPPINGS_ROOT.exists():
        releases.update(p.name for p in MAPPINGS_ROOT.iterdir() if p.is_dir() and p.name not in ignored)
    if TEMPLATES_ROOT.exists():
        releases.update(p.name for p in TEMPLATES_ROOT.iterdir() if p.is_dir() and p.name not in ignored)
    return sorted(releases)


def _available_forms(release: str) -> list[str]:
    root = MAPPINGS_ROOT / release
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.json"))


def _browse_mappings(rel_dir: str = "") -> tuple[list[str], list[str]]:
    base_root = Path(PROJECT_ROOT).resolve()
    default_root_rel = _rel(MAPPINGS_ROOT)
    rel_dir = str(rel_dir or "").strip().replace("\\", "/").rstrip("/")
    if not rel_dir:
        rel_dir = default_root_rel
    if ".." in rel_dir or rel_dir.startswith("/"):
        return [], []
    target = (base_root / rel_dir).resolve()
    try:
        target.relative_to(MAPPINGS_ROOT.resolve())
    except ValueError:
        return [], []
    if not target.is_dir():
        return [], []
    dirs: list[str] = []
    files: list[str] = []
    for p in target.iterdir():
        try:
            rel = str(p.relative_to(base_root)).replace("\\", "/")
        except ValueError:
            continue
        if p.is_dir() and p.name != ".backups":
            dirs.append(rel)
        elif p.is_file() and p.suffix.lower() == ".json":
            files.append(rel)
    return sorted(dirs, key=str.lower), sorted(files, key=str.lower)


def _expected_forms_from_templates(release: str) -> list[str]:
    root = TEMPLATES_ROOT / release
    forms: set[str] = set()
    if not root.is_dir():
        return []
    for p in root.glob("*.png"):
        stem = p.stem
        form = re.sub(r"_[ab]$", "", stem, flags=re.IGNORECASE)
        if form and FORM_NAME_RE.match(form):
            forms.add(form)
    return sorted(forms, key=str.lower)


def _coverage_for_release(release: str) -> dict[str, Any]:
    expected = set(_expected_forms_from_templates(release))
    existing = set(_available_forms(release))
    missing = sorted(expected - existing, key=str.lower)
    extra = sorted(existing - expected, key=str.lower)
    return {
        "release": release,
        "expected": sorted(expected, key=str.lower),
        "existing": sorted(existing, key=str.lower),
        "missing": missing,
        "extra": extra,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "ok": not missing,
    }


def _coverage_for_dir(rel_dir: str = "") -> dict[str, Any]:
    base_root = Path(PROJECT_ROOT).resolve()
    default_root_rel = _rel(MAPPINGS_ROOT)
    rel_dir = str(rel_dir or "").strip().replace("\\", "/").rstrip("/") or default_root_rel
    target = (base_root / rel_dir).resolve()
    try:
        rel = target.relative_to(MAPPINGS_ROOT.resolve())
    except ValueError:
        rel = Path()
    releases = _available_releases()
    if rel.parts:
        candidate = rel.parts[0]
        releases = [candidate] if candidate in releases or (TEMPLATES_ROOT / candidate).is_dir() else []
    checks = [_coverage_for_release(r) for r in releases]
    missing_total = sum(int(c["missing_count"]) for c in checks)
    return {
        "root": _rel(MAPPINGS_ROOT),
        "dir": rel_dir,
        "releases": checks,
        "missing_total": missing_total,
        "ok": missing_total == 0,
    }


def _blank_mapping(form: str) -> dict[str, Any]:
    return {
        "form_type": form,
        "sheet": form_sheet_name(form) or f"{form} Data",
        "start_row": 4,
        "mappings": [],
    }


def _roi_sources(release: str, form: str) -> dict[str, Any]:
    root = ROI_SCHEMAS_ROOT / release
    names: set[str] = set(RESERVED_SOURCES)
    by_side: dict[str, list[str]] = {}
    for side in ("a", "b"):
        path = root / f"{form}_{side}.json"
        side_names: list[str] = []
        if path.is_file():
            try:
                data = _load_json(path)
                for roi in data.get("rois") or []:
                    if not isinstance(roi, dict):
                        continue
                    name = str(roi.get("name", "")).strip()
                    if name:
                        names.add(name)
                        side_names.append(name)
            except Exception:
                pass
        by_side[side] = sorted(set(side_names), key=_source_sort_key)
    return {"all": sorted(names, key=_source_sort_key), "by_side": by_side}


def _source_sort_key(value: str) -> tuple[int, int, str]:
    s = str(value)
    if s.isdigit():
        return (0, int(s), s)
    m = re.match(r"^(\d+)([A-Za-z]+)$", s)
    if m:
        return (1, int(m.group(1)), m.group(2).lower())
    if s in RESERVED_SOURCES:
        return (-1, 0, s)
    return (2, 0, s.lower())


def _row_destinations(row: dict[str, Any]) -> list[str]:
    mtype = str(row.get("type", "")).strip().lower()
    cols: list[str] = []
    if mtype in {"direct", "lookup", "static"}:
        col = str(row.get("column", "")).strip().upper()
        if col:
            cols.append(col)
    elif mtype == "multi_column":
        choices = row.get("choices") or {}
        if isinstance(choices, dict):
            for col in choices.values():
                c = str(col or "").strip().upper()
                if c:
                    cols.append(c)
        no_answer = str(row.get("no_answer", "")).strip().upper()
        if no_answer:
            cols.append(no_answer)
    return cols


def _validate_mapping(mapping: Any, *, release: str, form: str) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    def issue(severity: str, message: str, row: int | None = None) -> None:
        issues.append({"severity": severity, "message": message, "row": row})

    if not isinstance(mapping, dict):
        return {"ok": False, "issues": [{"severity": "error", "message": "Mapping root must be an object.", "row": None}], "stats": {}}

    for key in ("form_type", "sheet", "start_row", "mappings"):
        if key not in mapping:
            issue("error", f"Missing required top-level key: {key}")
    if str(mapping.get("form_type", "")).strip() != form:
        issue("warning", f"form_type is {mapping.get('form_type')!r}, expected {form!r}")
    try:
        if int(mapping.get("start_row", 0)) < 1:
            issue("error", "start_row must be a positive integer")
    except Exception:
        issue("error", "start_row must be a positive integer")

    rows = mapping.get("mappings") or []
    if not isinstance(rows, list) or not rows:
        issue("error", "mappings must be a non-empty array")
        rows = []

    roi = _roi_sources(release, form)
    known_sources = set(roi["all"])
    destinations: dict[str, list[int]] = {}
    source_counts: dict[str, int] = {}

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            issue("error", "Row must be an object", idx)
            continue
        mtype = str(row.get("type", "")).strip().lower()
        if mtype not in VALID_TYPES:
            issue("error", f"Unknown mapping type: {mtype or '(blank)'}", idx)
            continue
        source = str(row.get("source", "")).strip()
        if mtype != "static":
            if not source:
                issue("error", "Missing source", idx)
            elif source not in known_sources:
                issue("warning", f"Source {source!r} not found in ROI schemas or reserved fields", idx)
            source_counts[source] = source_counts.get(source, 0) + 1
        if mtype in {"direct", "lookup", "static"}:
            col = str(row.get("column", "")).strip().upper()
            if not col:
                issue("error", "Missing target column", idx)
            elif not COLUMN_RE.match(col):
                issue("error", f"Invalid column: {col!r}", idx)
        if mtype == "direct":
            transform = str(row.get("transform", "")).strip().lower()
            if transform not in VALID_TRANSFORMS:
                issue("warning", f"Unrecognized transform: {transform!r}", idx)
        if mtype == "lookup":
            mp = row.get("map")
            if not isinstance(mp, dict) or not mp:
                issue("error", "lookup rows need a non-empty map object", idx)
        if mtype == "multi_column":
            choices = row.get("choices")
            if not isinstance(choices, dict) or not choices:
                issue("error", "multi_column rows need a non-empty choices object", idx)
            else:
                for key, col in choices.items():
                    c = str(col or "").strip().upper()
                    if not c or not COLUMN_RE.match(c):
                        issue("error", f"Invalid choice column for {key!r}: {col!r}", idx)
            mark = row.get("mark")
            if mark is None or str(mark) == "":
                issue("warning", "multi_column row has no mark value; pipeline defaults to Yes", idx)
        for col in _row_destinations(row):
            destinations.setdefault(col, []).append(idx)

    for col, row_nums in sorted(destinations.items()):
        unique_rows = sorted(set(row_nums))
        if len(unique_rows) > 1:
            issue("warning", f"Column {col} is written by multiple rows: {', '.join(map(str, unique_rows))}")

    for source, count in sorted(source_counts.items(), key=lambda kv: _source_sort_key(kv[0])):
        if count > 1:
            issue("info", f"Source {source!r} is used {count} times")

    severity_rank = {"error": 3, "warning": 2, "info": 1}
    issues.sort(key=lambda x: (-severity_rank.get(x["severity"], 0), x.get("row") or 10**9, x["message"]))
    stats = {
        "rows": len(rows),
        "destinations": len(destinations),
        "sources": len(source_counts),
        "errors": sum(1 for i in issues if i["severity"] == "error"),
        "warnings": sum(1 for i in issues if i["severity"] == "warning"),
        "infos": sum(1 for i in issues if i["severity"] == "info"),
    }
    return {"ok": stats["errors"] == 0, "issues": issues, "stats": stats, "roi_sources": roi}


def _transform_value(raw: Any, transform: str) -> Any:
    if raw is None:
        return None
    s = str(raw)
    t = str(transform or "").strip().lower()
    if t == "upper":
        return s.upper()
    if t == "lower":
        return s.lower()
    if t == "number":
        try:
            n = float(s.strip())
            return int(n) if n.is_integer() else n
        except Exception:
            return s
    return s


def _sample_value(source: str) -> str:
    if source == "id":
        return "601034A"
    if source == "form_type":
        return "6post"
    if source in {"page_odd", "page_even"}:
        return "1"
    if source.isdigit():
        return "a"
    return f"sample {source}"


def _preview_mapping(mapping: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    rows = mapping.get("mappings") or []
    if not isinstance(rows, list):
        return out
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        mtype = str(row.get("type", "")).strip().lower()
        source = str(row.get("source", "")).strip()
        raw = _sample_value(source) if source else ""
        if mtype == "static":
            out.append({"row": idx, "column": str(row.get("column", "")), "source": "static", "raw": "", "value": row.get("value", "")})
        elif mtype == "direct":
            out.append({"row": idx, "column": str(row.get("column", "")), "source": source, "raw": raw, "value": _transform_value(raw, str(row.get("transform", "")))})
        elif mtype == "lookup":
            mp = row.get("map") if isinstance(row.get("map"), dict) else {}
            key = raw.strip().lower()
            value = mp.get(key, mp.get(raw, row.get("default", "")))
            out.append({"row": idx, "column": str(row.get("column", "")), "source": source, "raw": raw, "value": value})
        elif mtype == "multi_column":
            choices = row.get("choices") if isinstance(row.get("choices"), dict) else {}
            key = raw.strip().lower()
            col = choices.get(key) or choices.get(raw) or row.get("no_answer", "")
            out.append({"row": idx, "column": str(col), "source": source, "raw": raw, "value": row.get("mark", "Yes")})
    return out


def _save_mapping_with_backup(path: Path, mapping: dict[str, Any]) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if path.exists():
        backups = path.parent / ".backups"
        backups.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = backups / f"{path.stem}.{stamp}.json"
        shutil.copy2(path, backup_path)
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return backup_path


def create_app() -> Flask:
    app = Flask(__name__, static_folder=TOOLS_DIR / "static", template_folder=TOOLS_DIR / "templates")

    @app.route("/")
    def index():
        return send_file(TOOLS_DIR / "templates" / "xlsx_mapping.html")

    @app.route("/api/instance")
    def api_instance():
        return jsonify({"instance_id": SERVER_INSTANCE_ID})

    @app.route("/api/mappings")
    def api_mappings():
        releases = _available_releases()
        active = DATA_RELEASE if DATA_RELEASE in releases else (releases[0] if releases else DATA_RELEASE)
        return jsonify({
            "active_release": active,
            "releases": [
                {"release": rel, "forms": _available_forms(rel)}
                for rel in releases
            ],
            "root": _rel(MAPPINGS_ROOT),
        })

    @app.route("/api/browse-root")
    def api_browse_root():
        return jsonify({"root": _rel(MAPPINGS_ROOT)})

    @app.route("/api/browse")
    def api_browse():
        rel_dir = request.args.get("dir", "").strip()
        dirs, files = _browse_mappings(rel_dir)
        return jsonify({"dirs": dirs, "files": files})

    @app.route("/api/coverage")
    def api_coverage():
        rel_dir = request.args.get("dir", "").strip()
        return jsonify(_coverage_for_dir(rel_dir))

    @app.route("/api/coverage/generate", methods=["POST"])
    def api_generate_missing_mappings():
        body = request.get_json(silent=True) or {}
        rel_dir = str(body.get("dir", "") or "").strip()
        coverage = _coverage_for_dir(rel_dir)
        created: list[str] = []
        for item in coverage.get("releases", []):
            release = str(item.get("release", "")).strip()
            if not release or not FORM_NAME_RE.match(release):
                continue
            for form in item.get("missing", []):
                try:
                    safe_form = _safe_form(str(form))
                    path = _mapping_path(release, safe_form)
                except Exception:
                    continue
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(_blank_mapping(safe_form), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                created.append(_rel(path))
        return jsonify(
            {
                "ok": True,
                "created": created,
                "created_count": len(created),
                "coverage": _coverage_for_dir(rel_dir),
            }
        )

    @app.route("/api/mapping")
    def api_mapping():
        try:
            rel_path = request.args.get("path", "").strip()
            if rel_path:
                path = _resolve_mapping_relpath(rel_path)
                if not path:
                    return jsonify({"error": "Invalid or missing mapping path"}), 400
                release, form = _mapping_info_from_path(path)
            else:
                release = _safe_release(request.args.get("release"))
                form = _safe_form(request.args.get("form"))
                path = _mapping_path(release, form)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        if not path.is_file():
            return jsonify({"error": f"Mapping not found: {_rel(path)}"}), 404
        try:
            mapping = _load_json(path)
        except Exception as exc:
            return jsonify({"error": f"Failed to load mapping: {exc}"}), 500
        validation = _validate_mapping(mapping, release=release, form=form)
        return jsonify({
            "mapping": mapping,
            "path": _rel(path),
            "release": release,
            "form": form,
            "validation": validation,
            "preview": _preview_mapping(mapping if isinstance(mapping, dict) else {}),
        })

    @app.route("/api/validate", methods=["POST"])
    def api_validate():
        body = request.get_json(silent=True) or {}
        try:
            release = _safe_release(body.get("release"))
            form = _safe_form(body.get("form"))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        mapping = body.get("mapping")
        validation = _validate_mapping(mapping, release=release, form=form)
        return jsonify({"validation": validation, "preview": _preview_mapping(mapping if isinstance(mapping, dict) else {})})

    @app.route("/api/mapping/auto-generate-config")
    def api_mapping_auto_generate_config():
        use_openai = bool(XLSX_MAPPING_AUTO_GENERATE.get("use_openai", True))
        model = (
            str(XLSX_MAPPING_AUTO_GENERATE.get("openai_model") or XLSX_MAPPING_AUTO_GENERATE.get("openai_model_profile") or "")
            if use_openai
            else str(XLSX_MAPPING_AUTO_GENERATE.get("local_model", ""))
        )
        return jsonify(
            {
                "ok": True,
                "provider": "openai" if use_openai else "local",
                "model": model,
                "example_dir": _rel(Path(XLSX_MAPPING_AUTO_GENERATE.get("example_dir", MAPPINGS_ROOT / "example")).resolve())
                if XLSX_MAPPING_AUTO_GENERATE.get("example_dir")
                else "",
            }
        )

    @app.route("/api/mapping/auto-generate", methods=["POST"])
    def api_auto_generate_mapping():
        body = request.get_json(silent=True) or {}
        try:
            rel_path = str(body.get("path", "") or "").strip()
            if rel_path:
                path = _resolve_mapping_relpath(rel_path)
                if not path:
                    return jsonify({"error": "Invalid or missing mapping path"}), 400
                release, form = _mapping_info_from_path(path)
            else:
                release = _safe_release(body.get("release"))
                form = _safe_form(body.get("form"))
                path = _mapping_path(release, form)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        existing: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = _load_json(path)
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}

        sheet_hint = str(body.get("sheet") or existing.get("sheet") or "").strip() or None
        try:
            start_row_hint = int(body.get("start_row") or existing.get("start_row") or 4)
        except Exception:
            start_row_hint = 4
        max_attempts = max(1, int(XLSX_MAPPING_AUTO_GENERATE.get("max_validation_attempts", 2) or 2))
        feedback: list[dict[str, Any]] = []
        last_result: dict[str, Any] | None = None
        validation: dict[str, Any] | None = None
        errors: list[str] = []
        for attempt in range(1, max_attempts + 1):
            try:
                last_result = generate_xlsx_mapping(
                    release=release,
                    form=form,
                    sheet_hint=sheet_hint,
                    start_row_hint=start_row_hint,
                    validation_feedback=feedback,
                )
            except (FileNotFoundError, ValueError) as exc:
                return jsonify({"error": f"XLSX mapping auto-generation failed: {exc}"}), 400
            except Exception as exc:
                return jsonify({"error": f"XLSX mapping auto-generation failed: {exc}"}), 500

            mapping = last_result.get("mapping") if isinstance(last_result, dict) else None
            validation = _validate_mapping(mapping, release=release, form=form)
            stats = validation.get("stats") or {}
            workbook_issues = last_result.get("workbook_issues") if isinstance(last_result, dict) else []
            if validation.get("ok") and int(stats.get("warnings", 0) or 0) == 0 and not workbook_issues:
                break
            feedback = [
                {
                    "attempt": attempt,
                    "severity": str(issue.get("severity", "")),
                    "row": issue.get("row"),
                    "message": str(issue.get("message", "")),
                }
                for issue in validation.get("issues", [])[:40]
                if isinstance(issue, dict) and issue.get("severity") in {"error", "warning"}
            ]
            if isinstance(workbook_issues, list):
                feedback.extend(
                    {
                        "attempt": attempt,
                        "severity": str(issue.get("severity", "error")),
                        "row": issue.get("row"),
                        "message": str(issue.get("message", "")),
                    }
                    for issue in workbook_issues[:20]
                    if isinstance(issue, dict)
                )
            errors.append(f"attempt {attempt}: {len(feedback)} validation error/warning(s)")

        if not last_result or not isinstance(last_result.get("mapping"), dict) or not validation:
            return jsonify({"error": "XLSX mapping auto-generation returned no mapping."}), 500
        mapping = last_result["mapping"]
        validation_stats = validation.get("stats") or {}
        workbook_issues = last_result.get("workbook_issues") if isinstance(last_result, dict) else []
        if not validation.get("ok") or int(validation_stats.get("warnings", 0) or 0) > 0 or workbook_issues:
            return jsonify(
                {
                    "error": "Generated mapping did not pass strict validation; not saved.",
                    "attempts": errors,
                    "workbook_issues": workbook_issues,
                    "mapping": mapping,
                    "validation": validation,
                    "preview": _preview_mapping(mapping),
                    "metadata": {k: v for k, v in last_result.items() if k != "mapping" and k != "raw_model_output"},
                }
            ), 422

        backup_path = _save_mapping_with_backup(path, mapping)
        return jsonify(
            {
                "ok": True,
                "path": _rel(path),
                "backup_path": _rel(backup_path) if backup_path else None,
                "mapping": mapping,
                "validation": validation,
                "preview": _preview_mapping(mapping),
                "metadata": {k: v for k, v in last_result.items() if k != "mapping" and k != "raw_model_output"},
            }
        )

    @app.route("/api/mapping", methods=["POST"])
    def api_save_mapping():
        body = request.get_json(silent=True) or {}
        try:
            rel_path = str(body.get("path", "") or "").strip()
            if rel_path:
                path = _resolve_mapping_relpath(rel_path)
                if not path:
                    return jsonify({"error": "Invalid or missing mapping path"}), 400
                release, form = _mapping_info_from_path(path)
            else:
                release = _safe_release(body.get("release"))
                form = _safe_form(body.get("form"))
                path = _mapping_path(release, form)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        mapping = body.get("mapping")
        validation = _validate_mapping(mapping, release=release, form=form)
        if not validation.get("ok"):
            return jsonify({"error": "Fix validation errors before saving.", "validation": validation}), 400
        backup_path = _save_mapping_with_backup(path, mapping)
        return jsonify({
            "ok": True,
            "path": _rel(path),
            "backup_path": _rel(backup_path) if backup_path else None,
            "validation": validation,
            "preview": _preview_mapping(mapping),
        })

    return app


def run(port: int = 5005, debug: bool = False) -> None:
    app = create_app()
    print(f"XLSX Mapping Editor: http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="XLSX Mapping Editor")
    p.add_argument("--port", type=int, default=5005, help="Port (default: 5005)")
    p.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = p.parse_args()
    run(port=args.port, debug=args.debug)
