"""
Shared ROI and schema logic for both CLI (Tk) and web ROI editor.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "py") not in sys.path:
    sys.path.insert(0, str(ROOT / "py"))

try:
    # Prefer full config (project root + paths) when available
    from config import PROJECT_ROOT, PATHS  # type: ignore
except Exception:
    PROJECT_ROOT = ROOT
    PATHS = {
        "cache_normalized": ROOT / "output" / "cache" / "normalized",
        "templates_root": ROOT / "data" / "templates",
        "roi_schemas_root": ROOT / "data" / "roi_schemas",
    }

def _resolve_collection_root(path_value: Path | str, expected_leaf: str) -> Path:
    """
    Resolve a collection root that may be release-scoped.
    Example:
      - .../data/templates/2025 -> .../data/templates
      - .../data/roi_schemas/2025 -> .../data/roi_schemas
    """
    p = Path(path_value).resolve()
    if p.name == expected_leaf:
        return p
    if p.parent.name == expected_leaf:
        return p.parent
    return p


_data_root = Path(PATHS.get("data", ROOT / "data")).resolve()
TEMPLATES_ROOT = _resolve_collection_root(
    PATHS.get("templates_root", _data_root / "templates"),
    "templates",
)
SCHEMA_DIR = _resolve_collection_root(
    PATHS.get("roi_schemas_root", _data_root / "roi_schemas"),
    "roi_schemas",
)

# Default browse root inside the project (relative to PROJECT_ROOT).
try:
    DEFAULT_BROWSE_ROOT = str(TEMPLATES_ROOT.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
except Exception:
    DEFAULT_BROWSE_ROOT = "data/templates"

DEFAULT_IMAGE = PATHS["cache_normalized"] / "maury_1" / "page_0001_bin.png"

HANDLE_SIZE = 8
MIN_ROI_SIZE = 5


def schema_path_for_image(image_path: Path) -> Path:
    """Path to the ROI schema JSON for this image, mirrored by template-relative path."""
    img = image_path.resolve()
    try:
        rel = img.relative_to(TEMPLATES_ROOT)
    except ValueError:
        # Fallback for non-template images: legacy stem-only path.
        return SCHEMA_DIR / f"{img.stem}.json"
    return SCHEMA_DIR / rel.with_suffix(".json")


def schema_relpath_for_image(image_path: Path) -> str:
    """Return schema path relative to project root, for UI display."""
    path = schema_path_for_image(image_path)
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


class ROI:
    __slots__ = ("id", "name", "x", "y", "w", "h", "meta")

    def __init__(
        self,
        name: str,
        x: int,
        y: int,
        w: int,
        h: int,
        id_: str | None = None,
        meta: dict | None = None,
    ):
        self.id = id_ or str(uuid.uuid4())[:8]
        self.name = name
        self.x = max(0, x)
        self.y = max(0, y)
        self.w = max(MIN_ROI_SIZE, w)
        self.h = max(MIN_ROI_SIZE, h)
        self.meta = dict(meta or {})

    def to_dict(self) -> dict:
        out = {"id": self.id, "name": self.name, "x": self.x, "y": self.y, "w": self.w, "h": self.h}
        out.update(self.meta)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> ROI:
        core_keys = {"id", "name", "x", "y", "w", "h"}
        meta = {k: v for k, v in d.items() if k not in core_keys}
        return cls(
            d["name"],
            int(d["x"]),
            int(d["y"]),
            int(d["w"]),
            int(d["h"]),
            id_=d.get("id"),
            meta=meta,
        )

    def contains_image_point(self, px: int, py: int) -> bool:
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def hit_handle(self, px: int, py: int) -> str | None:
        """Return which handle (n, s, e, w, ne, nw, se, sw) is at (px, py), or None."""
        hs = HANDLE_SIZE
        left, right = self.x, self.x + self.w
        top, bottom = self.y, self.y + self.h
        if abs(px - left) <= hs and abs(py - top) <= hs:
            return "nw"
        if abs(px - right) <= hs and abs(py - top) <= hs:
            return "ne"
        if abs(px - left) <= hs and abs(py - bottom) <= hs:
            return "sw"
        if abs(px - right) <= hs and abs(py - bottom) <= hs:
            return "se"
        if abs(py - top) <= hs and left <= px <= right:
            return "n"
        if abs(py - bottom) <= hs and left <= px <= right:
            return "s"
        if abs(px - left) <= hs and top <= py <= bottom:
            return "w"
        if abs(px - right) <= hs and top <= py <= bottom:
            return "e"
        return None


def load_schema(image_path: Path) -> tuple[list[dict], tuple[int, int] | None]:
    """Load ROI list (as dicts) and optional (width, height) from schema file. Returns ([], None) if missing/invalid."""
    path = schema_path_for_image(image_path)
    if not path.exists():
        return [], None
    try:
        data = json.loads(path.read_text())
        stored_path = data.get("image_path") or ""
        path_ok = False
        if stored_path:
            try:
                stored_abs = Path(stored_path)
                if not stored_abs.is_absolute():
                    stored_abs = (PROJECT_ROOT / stored_abs).resolve()
                path_ok = stored_abs == image_path.resolve()
            except Exception:
                path_ok = (stored_path == str(image_path))
        if not path_ok and data.get("image_name") == image_path.name:
            # Backward compatibility for older schema files lacking path details.
            path_ok = True
        if path_ok:
            rois = [r for r in data.get("rois", [])]
            size = (data.get("image_width"), data.get("image_height"))
            if size[0] and size[1]:
                return rois, (int(size[0]), int(size[1]))
            return rois, None
    except Exception:
        pass
    return [], None


def save_schema(image_path: Path, rois: list[dict], image_width: int, image_height: int) -> None:
    """Write schema JSON for the image."""
    path = schema_path_for_image(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "image_width": image_width,
        "image_height": image_height,
        "rois": rois,
    }
    path.write_text(json.dumps(data, indent=2))


def list_normalized_images() -> list[str]:
    """Return list of relative paths (e.g. data/templates/6post_a.png) from PROJECT_ROOT."""
    base_root = PROJECT_ROOT
    base = PROJECT_ROOT / DEFAULT_BROWSE_ROOT
    if not base.exists():
        return []
    out: list[str] = []
    for p in base.rglob("*.png"):
        try:
            rel = p.relative_to(base_root)
            out.append(str(rel).replace("\\", "/"))
        except ValueError:
            continue
    return sorted(out)


def browse_normalized(rel_dir: str = "") -> tuple[list[str], list[str]]:
    """List subdirs and .png files under PROJECT_ROOT/rel_dir (newest first)."""
    base_root = PROJECT_ROOT
    if not base_root.exists():
        return [], []
    rel_dir = (rel_dir or "").strip().rstrip("/")
    if not rel_dir:
        rel_dir = DEFAULT_BROWSE_ROOT
    if rel_dir and (".." in rel_dir or rel_dir.startswith("/")):
        return [], []
    target = (base_root / rel_dir).resolve() if rel_dir else base_root
    try:
        target.relative_to(base_root.resolve())
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
        elif p.suffix.lower() == ".png":
            file_entries.append((mtime, name))
    dir_entries.sort(key=lambda x: (-x[0], x[1].lower()))
    file_entries.sort(key=lambda x: (-x[0], x[1].lower()))
    return [name for _, name in dir_entries], [name for _, name in file_entries]
