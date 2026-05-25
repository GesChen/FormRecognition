"""
Standalone module: convert a PDF to images and cache them.

Usage from another script:

    from pdf_to_images import pdf_to_images

    paths = pdf_to_images("data/maury 1.pdf")
"""

from pathlib import Path
import hashlib
import json
import os
import re
import sys

from config import PDF_TO_IMAGES

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, unit=None, **kwargs):
        return iterable

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


def _tqdm_enabled() -> bool:
    """
    Render live tqdm bars only on interactive terminals.
    Prevents carriage-return bars from spamming line-based web log previews.
    """
    env = str(os.environ.get("EVMS_TQDM", "auto")).strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    if str(os.environ.get("TERM", "")).strip().lower() == "dumb":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _sanitize_name(name: str) -> str:
    """Make a safe folder name from PDF basename."""
    stem = Path(name).stem
    return re.sub(r"[^\w\-.]", "_", stem).strip("_") or "pdf"


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Content hash for robust cache reuse even when file mtime changes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _meta_path_for_subdir(subdir: Path) -> Path:
    return subdir / ".pdf_to_images_meta.json"


def _load_meta(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_meta(path: Path, meta: dict) -> None:
    try:
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        # Cache metadata failures should not break conversion.
        pass


def pdf_to_images(
    pdf_path: str | Path,
    cache_root: str | Path | None = None,
    dpi: int | None = None,
    fmt: str | None = None,
    *,
    use_tqdm: bool = False,
    max_pages: int | None = None,
) -> list[Path]:
    """
    Convert a PDF to images and store them in a cache folder.
    Reuses cached page images when the source PDF content hash matches.

    Args:
        pdf_path: Path to the PDF file.
        cache_root: Root folder for cache (default: from config).
        dpi: Resolution for rendered pages (default: from config).
        fmt: Image format: "png" or "jpeg" (default: from config).
        use_tqdm: If True, show a progress bar while converting pages.
        max_pages: If set, convert at most this many pages (1-based count from start).

    Returns:
        List of absolute paths to the cached image files.

    Raises:
        ImportError: If PyMuPDF is not installed.
        FileNotFoundError: If the PDF does not exist.
    """
    if fitz is None:
        raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")

    cfg = PDF_TO_IMAGES
    if cache_root is None:
        cache_root = cfg["cache_root"]
    if dpi is None:
        dpi = cfg["dpi"]
    if fmt is None:
        fmt = cfg["fmt"]

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cache_root = Path(cache_root).resolve()
    subdir = cache_root / _sanitize_name(pdf_path.name)
    subdir.mkdir(parents=True, exist_ok=True)
    ext = "png" if fmt.lower() == "png" else "jpg"
    pdf_hash = _file_sha256(pdf_path)
    meta_path = _meta_path_for_subdir(subdir)
    meta = _load_meta(meta_path)

    meta_hash = str(meta.get("pdf_sha256", "") or "")
    meta_dpi = int(meta.get("dpi", -1) or -1)
    meta_fmt = str(meta.get("fmt", "") or "").lower()
    meta_total_pages_raw = meta.get("total_pages", 0)
    try:
        meta_total_pages = int(meta_total_pages_raw)
    except (TypeError, ValueError):
        meta_total_pages = 0

    cache_matches = (
        meta_hash == pdf_hash
        and meta_dpi == int(dpi)
        and meta_fmt == ext
        and meta_total_pages > 0
    )

    if cache_matches:
        total = meta_total_pages
        if max_pages is not None and max_pages > 0:
            total = min(total, max_pages)
        paths = [subdir / f"page_{i + 1:04d}.{ext}" for i in range(total)]
        if total > 0 and all(p.exists() for p in paths):
            return paths

    doc = fitz.open(pdf_path)
    total_all = len(doc)
    total = total_all
    if max_pages is not None and max_pages > 0:
        total = min(total, max_pages)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    paths: list[Path] = [subdir / f"page_{i + 1:04d}.{ext}" for i in range(total)]

    page_indices = range(total)
    if use_tqdm:
        page_indices = tqdm(
            page_indices,
            desc="      Convert",
            unit="page",
            disable=not _tqdm_enabled(),
            dynamic_ncols=True,
            leave=False,
        )

    try:
        for i in page_indices:
            out_path = paths[i]
            if cache_matches and out_path.exists():
                continue
            page = doc[i]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(out_path))
    finally:
        doc.close()

    _write_meta(
        meta_path,
        {
            "pdf_sha256": pdf_hash,
            "dpi": int(dpi),
            "fmt": ext,
            "total_pages": int(total_all),
        },
    )

    return paths


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_to_images.py <pdf_path>")
        sys.exit(1)
    paths = pdf_to_images(sys.argv[1])
    for p in paths:
        print(p)
