"""
Interactive ROI editor: create, modify, delete, and name ROIs on an image.
Saves one schema JSON per image on exit (data/roi_schemas/<image_stem>.json).
Run with a normalized page image.

CLI (Tk):  python3 tools/roi_editor.py [image_path]
Web:       python3 tools/roi_editor.py --serve [--port 5000]

Requires tkinter for CLI (system package on Linux: sudo apt install python3-tk).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Allow imports from py/ and tools/
TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
if str(ROOT / "py") not in sys.path:
    sys.path.insert(0, str(ROOT / "py"))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import roi_editor_core as core
from roi_editor_core import ROI, HANDLE_SIZE, MIN_ROI_SIZE

# Tk and PIL (ImageTk optional; fallback uses PPM for PhotoImage)
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image

try:
    from PIL import ImageTk
except ImportError:
    ImageTk = None


def _pil_to_photoimage(pil_image: Image.Image) -> tk.PhotoImage:
    """Display a PIL Image on a Tk canvas. Use ImageTk if available, else PPM."""
    if ImageTk is not None:
        return ImageTk.PhotoImage(pil_image)
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, "PPM")
    buf.seek(0)
    return tk.PhotoImage(data=buf.read())


def _image_to_canvas(x: float, y: float, scale: float, ox: float, oy: float) -> tuple[float, float]:
    return (x * scale + ox, y * scale + oy)


def _canvas_to_image(x: float, y: float, scale: float, ox: float, oy: float) -> tuple[int, int]:
    if scale <= 0:
        return (0, 0)
    return (int(round((x - ox) / scale)), int(round((y - oy) / scale)))


class ROIEditorApp:
    def __init__(self, image_path: str | Path):
        self.image_path = Path(image_path).resolve()
        self.schema_path = core.schema_path_for_image(self.image_path)
        self.rois: list[ROI] = []
        self.selected_id: str | None = None
        self.drag_mode: str | None = None
        self.drag_start: tuple[int, int] | None = None
        self.create_start: tuple[int, int] | None = None
        self.img_size: tuple[int, int] = (0, 0)
        self.base_scale = 1.0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom_factor = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._pan_start: tuple[float, float] | None = None
        self._photo: tk.PhotoImage | None = None
        self._roi_counter = 0
        self._zoom_min, self._zoom_max = 0.25, 8.0
        self._zoom_after_id: str | None = None

        self.root = tk.Tk()
        self.root.title(f"ROI Editor — {self.image_path.name}")
        self.root.geometry("1200x800")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._load_image()
        self._load_schema()
        self._redraw()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=5)
        main.pack(fill=tk.BOTH, expand=True)

        canvas_frame = ttk.Frame(main)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="#2b2b2b",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_press)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom_linux)
        self.canvas.bind("<Button-5>", self._on_zoom_linux)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.focus_set()

        panel = ttk.Frame(main, width=350)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)
        ttk.Label(panel, text="ROIs", font=("", 11, "bold")).pack(anchor=tk.W)
        self.listbox = tk.Listbox(panel, height=12, selectmode=tk.SINGLE, font=("", 10))
        self.listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        self.listbox.bind("<Double-1>", self._on_list_double)
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="New ROI", command=self._start_create_mode).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Rename", command=self._rename_selected).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Duplicate", command=self._duplicate_selected).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_frame, text="Delete", command=self._delete_selected).pack(side=tk.LEFT)
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)
        ttk.Label(panel, text="Draw a rectangle to add ROI.\nSelect then drag handles to resize.\nScroll to zoom, middle-drag to pan.", font=("", 9), foreground="gray").pack(anchor=tk.W)

    def _load_image(self) -> None:
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        pil_img = Image.open(self.image_path).convert("RGB")
        self._pil_image = pil_img
        self.img_size = (pil_img.width, pil_img.height)
        self._update_scale_and_photo()

    def _update_scale_and_photo(self) -> None:
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        iw, ih = self.img_size
        if iw == 0 or ih == 0:
            return
        self.base_scale = min(cw / iw, ch / ih)
        self.scale = self.base_scale * self.zoom_factor
        self.offset_x = (cw - iw * self.scale) / 2 + self.pan_x
        self.offset_y = (ch - ih * self.scale) / 2 + self.pan_y
        new_w = max(1, int(iw * self.scale))
        new_h = max(1, int(ih * self.scale))
        resized = self._pil_image.resize((new_w, new_h), Image.Resampling.NEAREST)
        self._photo = _pil_to_photoimage(resized)
        self._redraw()

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._update_scale_and_photo()

    def _on_pan_press(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        self.pan_x += event.x - self._pan_start[0]
        self.pan_y += event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self._update_scale_and_photo()

    def _on_pan_release(self, event: tk.Event) -> None:
        self._pan_start = None

    def _zoom_at(self, canvas_x: float, canvas_y: float, delta: float) -> None:
        iw, ih = self.img_size
        ix = (canvas_x - self.offset_x) / self.scale
        iy = (canvas_y - self.offset_y) / self.scale
        self.zoom_factor = max(self._zoom_min, min(self._zoom_max, self.zoom_factor * delta))
        new_scale = self.base_scale * self.zoom_factor
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.pan_x = canvas_x - ix * new_scale - (cw - iw * new_scale) / 2
        self.pan_y = canvas_y - iy * new_scale - (ch - ih * new_scale) / 2
        if self._zoom_after_id is not None:
            self.root.after_cancel(self._zoom_after_id)
        self._zoom_after_id = self.root.after(40, self._zoom_apply)

    def _zoom_apply(self) -> None:
        self._zoom_after_id = None
        self._update_scale_and_photo()

    def _on_zoom(self, event: tk.Event) -> None:
        delta = 1.15 if event.delta > 0 else 1.0 / 1.15
        self._zoom_at(event.x, event.y, delta)

    def _on_zoom_linux(self, event: tk.Event) -> None:
        delta = 1.15 if event.num == 4 else 1.0 / 1.15
        self._zoom_at(event.x, event.y, delta)

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._photo:
            self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self._photo, tags=("image",))
        for roi in self.rois:
            self._draw_roi(roi)
        if self.drag_mode == "create" and self.create_start:
            x0, y0 = _image_to_canvas(self.create_start[0], self.create_start[1], self.scale, self.offset_x, self.offset_y)
            x1, y1 = _image_to_canvas(self._last_mouse_im[0], self._last_mouse_im[1], self.scale, self.offset_x, self.offset_y)
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="lime", width=2, dash=(4, 2), tags=("preview",))
        self._sync_listbox()

    def _last_mouse_im(self) -> tuple[int, int]:
        return getattr(self, "_last_im", (0, 0))

    def _draw_roi(self, roi: ROI) -> None:
        x0, y0 = _image_to_canvas(roi.x, roi.y, self.scale, self.offset_x, self.offset_y)
        x1, y1 = _image_to_canvas(roi.x + roi.w, roi.y + roi.h, self.scale, self.offset_x, self.offset_y)
        is_sel = roi.id == self.selected_id
        outline = "cyan" if is_sel else "yellow"
        width = 3 if is_sel else 2
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=width, tags=("roi", roi.id))
        if is_sel:
            hs = HANDLE_SIZE
            for (hx, hy) in self._handle_positions(roi):
                self.canvas.create_rectangle(hx - hs, hy - hs, hx + hs, hy + hs, outline="white", fill="cyan", width=1, tags=("handle", roi.id))

    def _handle_positions(self, roi: ROI) -> list[tuple[float, float]]:
        cx = roi.x + roi.w / 2
        cy = roi.y + roi.h / 2
        left, right = roi.x, roi.x + roi.w
        top, bottom = roi.y, roi.y + roi.h
        pts = [
            (left, top), (right, top), (right, bottom), (left, bottom),
            (cx, top), (right, cy), (cx, bottom), (left, cy),
        ]
        return [_image_to_canvas(px, py, self.scale, self.offset_x, self.offset_y) for px, py in pts]

    def _on_press(self, event: tk.Event) -> None:
        px, py = _canvas_to_image(event.x, event.y, self.scale, self.offset_x, self.offset_y)
        self._last_im = (px, py)
        iw, ih = self.img_size
        if px < 0 or py < 0 or px >= iw or py >= ih:
            return
        if self.selected_id:
            sel = self._get_roi(self.selected_id)
            if sel:
                handle = sel.hit_handle(px, py)
                if handle:
                    self.drag_mode = "resize"
                    self._resize_handle = handle
                    self.drag_start = (px, py)
                    return
        for roi in reversed(self.rois):
            if roi.contains_image_point(px, py):
                self.selected_id = roi.id
                self.drag_mode = "move"
                self.drag_start = (px - roi.x, py - roi.y)
                self._redraw()
                return
        self.selected_id = None
        self.drag_mode = "create"
        self.create_start = (px, py)
        self._redraw()

    def _on_drag(self, event: tk.Event) -> None:
        px, py = _canvas_to_image(event.x, event.y, self.scale, self.offset_x, self.offset_y)
        self._last_im = (px, py)
        if self.drag_mode == "create" and self.create_start:
            self._redraw()
            return
        if self.drag_mode == "move" and self.selected_id and self.drag_start:
            roi = self._get_roi(self.selected_id)
            if roi:
                dx, dy = self.drag_start
                roi.x = max(0, px - dx)
                roi.y = max(0, py - dy)
                iw, ih = self.img_size
                roi.x = min(roi.x, iw - roi.w)
                roi.y = min(roi.y, ih - roi.h)
                self._redraw()
        if self.drag_mode == "resize" and self.selected_id and self.drag_start:
            roi = self._get_roi(self.selected_id)
            if roi:
                h = getattr(self, "_resize_handle", None)
                if h and h in ("n", "s", "e", "w", "nw", "ne", "sw", "se"):
                    self._apply_resize(roi, h, px, py)
                    self._redraw()

    def _apply_resize(self, roi: ROI, handle: str, px: int, py: int) -> None:
        if "e" in handle:
            roi.w = max(MIN_ROI_SIZE, px - roi.x)
        if "w" in handle:
            new_w = roi.x + roi.w - px
            if new_w >= MIN_ROI_SIZE:
                roi.x = px
                roi.w = new_w
        if "s" in handle:
            roi.h = max(MIN_ROI_SIZE, py - roi.y)
        if "n" in handle:
            new_h = roi.y + roi.h - py
            if new_h >= MIN_ROI_SIZE:
                roi.y = py
                roi.h = new_h
        iw, ih = self.img_size
        roi.x = max(0, min(roi.x, iw - MIN_ROI_SIZE))
        roi.y = max(0, min(roi.y, ih - MIN_ROI_SIZE))
        roi.w = max(MIN_ROI_SIZE, min(roi.w, iw - roi.x))
        roi.h = max(MIN_ROI_SIZE, min(roi.h, ih - roi.y))

    def _on_release(self, event: tk.Event) -> None:
        px, py = _canvas_to_image(event.x, event.y, self.scale, self.offset_x, self.offset_y)
        if self.drag_mode == "create" and self.create_start:
            x0, y0 = self.create_start
            w = abs(px - x0)
            h = abs(py - y0)
            if w >= MIN_ROI_SIZE and h >= MIN_ROI_SIZE:
                self._roi_counter += 1
                name = f"ROI {self._roi_counter}"
                x = min(x0, px)
                y = min(y0, py)
                self.rois.append(ROI(name, x, y, w, h))
                self.selected_id = self.rois[-1].id
            self.create_start = None
        self.drag_mode = None
        self.drag_start = None
        if hasattr(self, "_resize_handle"):
            delattr(self, "_resize_handle")
        self._redraw()

    def _get_roi(self, id_: str) -> ROI | None:
        for r in self.rois:
            if r.id == id_:
                return r
        return None

    def _start_create_mode(self) -> None:
        self.selected_id = None
        self.drag_mode = None
        self._redraw()

    def _sync_listbox(self) -> None:
        yview = self.listbox.yview()
        self.listbox.delete(0, tk.END)
        for roi in self.rois:
            self.listbox.insert(tk.END, roi.name)
        if self.selected_id:
            idx = next((i for i, r in enumerate(self.rois) if r.id == self.selected_id), None)
            if idx is not None:
                self.listbox.selection_set(idx)
                self.listbox.see(idx)
        if yview and self.listbox.size() > 0:
            self.listbox.yview_moveto(yview[0])

    def _on_list_select(self, event: tk.Event) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.rois):
            self.selected_id = self.rois[idx].id
            self._redraw()

    def _on_list_double(self, event: tk.Event) -> None:
        self._rename_selected()

    def _rename_selected(self) -> None:
        if not self.selected_id:
            messagebox.showinfo("Rename", "Select an ROI first (click on list or canvas).")
            return
        roi = self._get_roi(self.selected_id)
        if not roi:
            return
        new_name = simpledialog.askstring("Rename ROI", "Name:", initialvalue=roi.name)
        if new_name is not None and new_name.strip():
            roi.name = new_name.strip()
            self._redraw()

    def _duplicate_selected(self) -> None:
        if not self.selected_id:
            messagebox.showinfo("Duplicate", "Select an ROI first (click on list or canvas).")
            return
        roi = self._get_roi(self.selected_id)
        if not roi:
            return
        offset = 10
        copy_name = roi.name.replace(" (copy)", "").rstrip() + " (copy)"
        dup = ROI(
            copy_name,
            roi.x + offset,
            roi.y + offset,
            roi.w,
            roi.h,
            meta=dict(getattr(roi, "meta", {}) or {}),
        )
        self.rois.append(dup)
        self.selected_id = dup.id
        self._redraw()

    def _delete_selected(self) -> None:
        if not self.selected_id:
            messagebox.showinfo("Delete", "Select an ROI first.")
            return
        if not messagebox.askyesno("Delete", "Delete selected ROI?"):
            return
        self.rois = [r for r in self.rois if r.id != self.selected_id]
        self.selected_id = None
        self._redraw()

    def _load_schema(self) -> None:
        rois_dict, size = core.load_schema(self.image_path)
        for d in rois_dict:
            self.rois.append(ROI.from_dict(d))
        self._roi_counter = len(self.rois)

    def _save_schema(self) -> None:
        core.save_schema(self.image_path, [r.to_dict() for r in self.rois], self.img_size[0], self.img_size[1])

    def _on_close(self) -> None:
        if self._zoom_after_id is not None:
            self.root.after_cancel(self._zoom_after_id)
            self._zoom_after_id = None
        self._save_schema()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ROI editor: create, modify, delete, and name ROIs on an image. Use --serve for web UI."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to normalized page image (default: cache/normalized/maury_1/page_0001_bin.png)",
    )
    parser.add_argument("--serve", action="store_true", help="Run web server instead of Tk GUI")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server (default: 5000)")
    args = parser.parse_args()

    if args.serve:
        try:
            from roi_editor_web import run as run_web
        except ImportError as e:
            if "flask" in str(e).lower():
                sys.exit("Flask is required for the web server. Install with: pip install flask")
            raise
        run_web(port=args.port)
        return

    image_path = args.image_path if args.image_path is not None else core.DEFAULT_IMAGE
    app = ROIEditorApp(image_path)
    app.run()


if __name__ == "__main__":
    main()
