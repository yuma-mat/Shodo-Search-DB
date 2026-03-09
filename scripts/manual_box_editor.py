#!/usr/bin/env python3
"""
Manual 5-box editor for calligraphy PDF pages.

Usage:
  python3 scripts/manual_box_editor.py --pdf "/path/to/file.pdf"

Mouse:
  - Drag inside a box: move
  - Drag corner handle: resize

Keys:
  - 1..5 : select box
  - n / Enter : next page
  - p : previous page
  - r : reset current page to default 5 slots
  - s : save JSON
  - q : save and quit
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np


WINDOW_NAME = "Shodo Box Editor"
HANDLE_PX = 9
MIN_BOX_SIDE = 24

# right-top, right-middle, right-bottom, left-top, left-middle
DEFAULT_SLOTS = [
    (0.50, 0.00, 0.50, 1.0 / 3.0),
    (0.50, 1.0 / 3.0, 0.50, 1.0 / 3.0),
    (0.50, 2.0 / 3.0, 0.50, 1.0 / 3.0),
    (0.00, 0.00, 0.50, 1.0 / 3.0),
    (0.00, 1.0 / 3.0, 0.50, 1.0 / 3.0),
]


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


class Editor:
    def __init__(self, pdf_path: Path, out_json: Path, dpi: int) -> None:
        self.pdf_path = pdf_path
        self.out_json = out_json
        self.dpi = dpi
        self.doc = fitz.open(pdf_path)
        self.page_count = self.doc.page_count
        self.page_idx = 0
        self.selected = 0

        self.cache: dict[int, np.ndarray] = {}
        self.boxes_by_page: dict[int, list[Box]] = {}

        self.display_scale = 1.0
        self.display_img: np.ndarray | None = None
        self.base_img: np.ndarray | None = None

        self.drag_mode: str | None = None  # "move" or "resize"
        self.drag_corner: int | None = None
        self.drag_start = (0, 0)
        self.drag_box_start: Box | None = None
        self.drag_box_idx: int | None = None

    def render_page(self, page_idx: int) -> np.ndarray:
        if page_idx in self.cache:
            return self.cache[page_idx]
        page = self.doc.load_page(page_idx)
        scale = self.dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.cache[page_idx] = img
        return img

    def default_boxes(self, img: np.ndarray) -> list[Box]:
        h, w = img.shape[:2]
        boxes: list[Box] = []
        for rx, ry, rw, rh in DEFAULT_SLOTS:
            x = int(round(w * rx))
            y = int(round(h * ry))
            bw = int(round(w * rw))
            bh = int(round(h * rh))
            boxes.append(Box(x, y, bw, bh))
        return boxes

    def current_boxes(self) -> list[Box]:
        if self.page_idx not in self.boxes_by_page:
            img = self.render_page(self.page_idx)
            self.boxes_by_page[self.page_idx] = self.default_boxes(img)
        return self.boxes_by_page[self.page_idx]

    def clamp_box(self, box: Box, img_w: int, img_h: int) -> Box:
        w = max(MIN_BOX_SIDE, min(box.w, img_w))
        h = max(MIN_BOX_SIDE, min(box.h, img_h))
        x = max(0, min(box.x, img_w - w))
        y = max(0, min(box.y, img_h - h))
        return Box(x, y, w, h)

    def to_display(self, x: int, y: int) -> tuple[int, int]:
        s = self.display_scale
        return int(round(x * s)), int(round(y * s))

    def from_display(self, x: int, y: int) -> tuple[int, int]:
        s = max(1e-8, self.display_scale)
        return int(round(x / s)), int(round(y / s))

    def get_corner_points(self, box: Box) -> list[tuple[int, int]]:
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def hit_test(self, px: int, py: int) -> tuple[int | None, str | None, int | None]:
        boxes = self.current_boxes()
        click_x, click_y = self.from_display(px, py)

        for i in range(len(boxes) - 1, -1, -1):
            b = boxes[i]
            corners = self.get_corner_points(b)
            for ci, (cx, cy) in enumerate(corners):
                dx = click_x - cx
                dy = click_y - cy
                if dx * dx + dy * dy <= (int(HANDLE_PX / self.display_scale) + 2) ** 2:
                    return i, "resize", ci

        for i in range(len(boxes) - 1, -1, -1):
            b = boxes[i]
            if b.x <= click_x <= b.x + b.w and b.y <= click_y <= b.y + b.h:
                return i, "move", None

        return None, None, None

    def update_page_view(self) -> None:
        self.base_img = self.render_page(self.page_idx)
        h, w = self.base_img.shape[:2]
        max_w, max_h = 1400, 900
        self.display_scale = min(max_w / w, max_h / h, 1.0)
        disp = self.base_img
        if self.display_scale < 0.999:
            disp = cv2.resize(
                self.base_img,
                (int(round(w * self.display_scale)), int(round(h * self.display_scale))),
                interpolation=cv2.INTER_AREA,
            )

        boxes = self.current_boxes()
        canvas = disp.copy()
        for i, box in enumerate(boxes):
            x1, y1 = self.to_display(box.x, box.y)
            x2, y2 = self.to_display(box.x + box.w, box.y + box.h)
            color = (40, 80, 220) if i == self.selected else (200, 120, 120)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = str(i + 1)
            cv2.putText(canvas, label, (x1 + 6, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            for cx, cy in self.get_corner_points(box):
                dx, dy = self.to_display(cx, cy)
                cv2.circle(canvas, (dx, dy), HANDLE_PX, color, -1)

        info = (
            f"page {self.page_idx + 1}/{self.page_count}  "
            "keys: 1-5 select, drag move/resize, n/Enter next, p prev, r reset, s save, q quit"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (255, 255, 255), -1)
        cv2.putText(canvas, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (30, 30, 30), 1)
        self.display_img = canvas

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.base_img is None:
            return
        boxes = self.current_boxes()
        img_h, img_w = self.base_img.shape[:2]

        if event == cv2.EVENT_LBUTTONDOWN:
            idx, mode, corner = self.hit_test(x, y)
            if idx is not None and mode is not None:
                self.selected = idx
                self.drag_mode = mode
                self.drag_corner = corner
                self.drag_box_idx = idx
                self.drag_start = self.from_display(x, y)
                b = boxes[idx]
                self.drag_box_start = Box(b.x, b.y, b.w, b.h)
                self.update_page_view()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_mode is None or self.drag_box_start is None or self.drag_box_idx is None:
                return
            cur_x, cur_y = self.from_display(x, y)
            start_x, start_y = self.drag_start
            dx, dy = cur_x - start_x, cur_y - start_y
            b0 = self.drag_box_start
            b = Box(b0.x, b0.y, b0.w, b0.h)

            if self.drag_mode == "move":
                b = Box(b0.x + dx, b0.y + dy, b0.w, b0.h)
            elif self.drag_mode == "resize":
                x1, y1 = b0.x, b0.y
                x2, y2 = b0.x + b0.w, b0.y + b0.h
                c = self.drag_corner
                if c == 0:
                    x1 += dx
                    y1 += dy
                elif c == 1:
                    x2 += dx
                    y1 += dy
                elif c == 2:
                    x2 += dx
                    y2 += dy
                elif c == 3:
                    x1 += dx
                    y2 += dy
                if x2 - x1 < MIN_BOX_SIDE:
                    if c in (0, 3):
                        x1 = x2 - MIN_BOX_SIDE
                    else:
                        x2 = x1 + MIN_BOX_SIDE
                if y2 - y1 < MIN_BOX_SIDE:
                    if c in (0, 1):
                        y1 = y2 - MIN_BOX_SIDE
                    else:
                        y2 = y1 + MIN_BOX_SIDE
                b = Box(x1, y1, x2 - x1, y2 - y1)

            boxes[self.drag_box_idx] = self.clamp_box(b, img_w, img_h)
            self.update_page_view()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = None
            self.drag_corner = None
            self.drag_box_start = None
            self.drag_box_idx = None

    def save_json(self) -> None:
        data = {
            "pdf_path": str(self.pdf_path),
            "page_count": self.page_count,
            "boxes": {},
        }
        for p in range(self.page_count):
            boxes = self.boxes_by_page.get(p)
            if boxes is None:
                img = self.render_page(p)
                boxes = self.default_boxes(img)
            img = self.render_page(p)
            h, w = img.shape[:2]
            page_boxes = []
            for b in boxes:
                item = asdict(b)
                item["x_ratio"] = b.x / max(1.0, float(w))
                item["y_ratio"] = b.y / max(1.0, float(h))
                item["w_ratio"] = b.w / max(1.0, float(w))
                item["h_ratio"] = b.h / max(1.0, float(h))
                page_boxes.append(item)
            data["boxes"][str(p + 1)] = page_boxes
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        self.out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved: {self.out_json}")

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.update_page_view()

        while True:
            if self.display_img is not None:
                cv2.imshow(WINDOW_NAME, self.display_img)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if ord("1") <= key <= ord("5"):
                self.selected = key - ord("1")
                self.update_page_view()
            elif key in (ord("n"), 13):
                if self.page_idx < self.page_count - 1:
                    self.page_idx += 1
                    self.update_page_view()
            elif key == ord("p"):
                if self.page_idx > 0:
                    self.page_idx -= 1
                    self.update_page_view()
            elif key == ord("r"):
                img = self.render_page(self.page_idx)
                self.boxes_by_page[self.page_idx] = self.default_boxes(img)
                self.update_page_view()
            elif key == ord("s"):
                self.save_json()
            elif key == ord("q"):
                self.save_json()
                break

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Manual editor for 5 character boxes on PDF pages.")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--out-json", default="data/manual_boxes.json", help="Output JSON path")
    parser.add_argument("--dpi", type=int, default=260, help="Render DPI for editing")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    editor = Editor(pdf_path=pdf_path, out_json=out_json, dpi=args.dpi)
    editor.run()


if __name__ == "__main__":
    main()
