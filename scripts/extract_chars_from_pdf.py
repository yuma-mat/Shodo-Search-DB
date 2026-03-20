#!/usr/bin/env python3
"""
Extract 4 or 5 calligraphy character images per page from a PDF.

Methods:
- slots: fixed layout (4 or 5 cells)
- cluster: connected components + KMeans clustering (4 or 5 groups)

Both methods include noise filtering to suppress thin lines / memo scribbles.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import fitz  # PyMuPDF
import numpy as np


@dataclass(frozen=True)
class Slot:
    name: str
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class Component:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


SLOTS_5 = [
    Slot("right_top", 0.50, 0.00, 0.50, 1.00 / 3.0),
    Slot("right_middle", 0.50, 1.00 / 3.0, 0.50, 1.00 / 3.0),
    Slot("right_bottom", 0.50, 2.00 / 3.0, 0.50, 1.00 / 3.0),
    Slot("left_top", 0.00, 0.00, 0.50, 1.00 / 3.0),
    Slot("left_middle", 0.00, 1.00 / 3.0, 0.50, 1.00 / 3.0),
]

SLOTS_4 = [
    Slot("right_top", 0.50, 0.00, 0.50, 0.50),
    Slot("right_bottom", 0.50, 0.50, 0.50, 0.50),
    Slot("left_top", 0.00, 0.00, 0.50, 0.50),
    Slot("left_bottom", 0.00, 0.50, 0.50, 0.50),
]


def get_slot_layout(char_count: int) -> list[Slot]:
    if char_count == 4:
        return SLOTS_4
    return SLOTS_5


def get_slot_names(char_count: int) -> list[str]:
    if char_count == 4:
        return ["right_top", "right_bottom", "left_top", "left_bottom"]
    return ["right_top", "right_middle", "right_bottom", "left_top", "left_middle"]


def parse_crop(crop_text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in crop_text.split(",")]
    if len(parts) != 4:
        raise ValueError("crop must be 4 comma-separated values: x,y,w,h")
    x, y, w, h = map(float, parts)
    for v in (x, y, w, h):
        if v < 0.0 or v > 1.0:
            raise ValueError("crop values must be between 0.0 and 1.0")
    if x + w > 1.0 or y + h > 1.0:
        raise ValueError("crop rectangle must fit inside the page")
    return x, y, w, h


def render_page_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def rotate_image(img: np.ndarray, rotate: int) -> np.ndarray:
    if rotate == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate == -90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def crop_ratio_rect(img: np.ndarray, rect_ratio: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    rx, ry, rw, rh = rect_ratio
    x1 = max(0, int(round(w * rx)))
    y1 = max(0, int(round(h * ry)))
    x2 = min(w, int(round(w * (rx + rw))))
    y2 = min(h, int(round(h * (ry + rh))))
    return img[y1:y2, x1:x2]


def slot_to_pixels(img: np.ndarray, slot: Slot) -> tuple[int, int, int, int]:
    h, w = img.shape[:2]
    x1 = int(round(w * slot.x))
    y1 = int(round(h * slot.y))
    x2 = int(round(w * (slot.x + slot.w)))
    y2 = int(round(h * (slot.y + slot.h)))
    return x1, y1, x2, y2


def binarize_ink(img: np.ndarray, method: str, block_size: int, c_value: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    if method == "adaptive":
        block_size = max(3, block_size)
        if block_size % 2 == 0:
            block_size += 1
        binary_inv = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_value,
        )
    else:
        _, binary_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary_inv


def extract_components(
    binary_inv: np.ndarray,
    min_area_ratio: float,
    min_short_ratio: float,
    max_line_aspect: float,
) -> list[Component]:
    h, w = binary_inv.shape[:2]
    area_img = h * w
    min_area = max(4, int(area_img * min_area_ratio))
    min_short = max(2, int(min(h, w) * min_short_ratio))

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_inv, connectivity=8)
    comps: list[Component] = []
    for idx in range(1, num_labels):
        x, y, ww, hh, area = stats[idx]
        if area < min_area:
            continue

        short_side = min(ww, hh)
        long_side = max(ww, hh)
        aspect = long_side / max(1.0, float(short_side))
        fill_ratio = area / float(max(1, ww * hh))

        # Remove thin line-like scribbles.
        if short_side < min_short and aspect > max_line_aspect:
            continue
        if short_side < max(2, int(min_short * 0.7)):
            continue
        if aspect > (max_line_aspect * 0.75) and fill_ratio < 0.06:
            continue

        comps.append(Component(x=x, y=y, w=ww, h=hh, area=area))

    return comps


def union_bbox(comps: list[Component]) -> tuple[int, int, int, int]:
    x1 = min(c.x for c in comps)
    y1 = min(c.y for c in comps)
    x2 = max(c.x + c.w for c in comps)
    y2 = max(c.y + c.h for c in comps)
    return x1, y1, x2, y2


def pad_bbox(
    bbox: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    pad_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    side = max(x2 - x1, y2 - y1)
    pad = int(round(side * pad_ratio))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(img_w, x2 + pad)
    y2 = min(img_h, y2 + pad)
    return x1, y1, x2, y2


def tight_crop_with_components(
    img: np.ndarray,
    pad_ratio: float,
    min_area_ratio: float,
    min_short_ratio: float,
    max_line_aspect: float,
    binarize_method: str,
    adaptive_block_size: int,
    adaptive_c: int,
) -> np.ndarray:
    binary = binarize_ink(img, binarize_method, adaptive_block_size, adaptive_c)
    comps = extract_components(binary, min_area_ratio, min_short_ratio, max_line_aspect)
    if not comps:
        return img
    x1, y1, x2, y2 = union_bbox(comps)
    x1, y1, x2, y2 = pad_bbox((x1, y1, x2, y2), img.shape[1], img.shape[0], pad_ratio)
    return img[y1:y2, x1:x2]


def build_clean_mask(
    img: np.ndarray,
    min_area_ratio: float,
    min_short_ratio: float,
    max_line_aspect: float,
    binarize_method: str,
    adaptive_block_size: int,
    adaptive_c: int,
) -> np.ndarray:
    binary = binarize_ink(img, binarize_method, adaptive_block_size, adaptive_c)
    comps = extract_components(binary, min_area_ratio, min_short_ratio, max_line_aspect)
    mask = np.zeros(binary.shape, dtype=np.uint8)
    for c in comps:
        mask[c.y : c.y + c.h, c.x : c.x + c.w] = cv2.bitwise_or(
            mask[c.y : c.y + c.h, c.x : c.x + c.w],
            binary[c.y : c.y + c.h, c.x : c.x + c.w],
        )
    return mask


def apply_mask_gray(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = np.full_like(gray, 255)
    out[mask > 0] = gray[mask > 0]
    return out


def extract_chars_by_slots(
    paper_img: np.ndarray,
    slot_layout: list[Slot],
    pad_ratio: float,
    min_area_ratio: float,
    min_short_ratio: float,
    max_line_aspect: float,
    binarize_method: str,
    adaptive_block_size: int,
    adaptive_c: int,
) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    for slot in slot_layout:
        x1, y1, x2, y2 = slot_to_pixels(paper_img, slot)
        slot_img = paper_img[y1:y2, x1:x2]
        char_img = tight_crop_with_components(
            slot_img,
            pad_ratio=pad_ratio,
            min_area_ratio=min_area_ratio,
            min_short_ratio=min_short_ratio,
            max_line_aspect=max_line_aspect,
            binarize_method=binarize_method,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
        )
        out.append((slot.name, char_img))
    return out


def _order_cluster_boxes(
    bboxes: list[tuple[int, int, int, int]],
    expected_right: int,
    expected_left: int,
) -> list[tuple[int, int, int, int]] | None:
    if len(bboxes) != (expected_right + expected_left):
        return None

    centers = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0] for (x1, y1, x2, y2) in bboxes], dtype=np.float32)
    x_data = centers[:, :1]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, x_centers = cv2.kmeans(x_data, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)
    x_centers = x_centers.reshape(-1)

    right_label = int(np.argmax(x_centers))
    right = []
    left = []
    for i, bbox in enumerate(bboxes):
        item = (bbox, centers[i][1])
        if labels[i] == right_label:
            right.append(item)
        else:
            left.append(item)

    right.sort(key=lambda t: t[1])
    left.sort(key=lambda t: t[1])

    if len(right) < expected_right or len(left) < expected_left:
        return None

    ordered = [x[0] for x in right[:expected_right]] + [x[0] for x in left[:expected_left]]
    return ordered


def extract_chars_by_cluster(
    paper_img: np.ndarray,
    char_count: int,
    pad_ratio: float,
    min_area_ratio: float,
    min_short_ratio: float,
    max_line_aspect: float,
    binarize_method: str,
    adaptive_block_size: int,
    adaptive_c: int,
) -> list[tuple[str, np.ndarray]] | None:
    binary = binarize_ink(paper_img, binarize_method, adaptive_block_size, adaptive_c)
    comps = extract_components(binary, min_area_ratio, min_short_ratio, max_line_aspect)
    if len(comps) < char_count:
        return None

    points = np.array([[c.cx, c.cy] for c in comps], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 0.2)
    _, labels, _ = cv2.kmeans(points, char_count, None, criteria, 12, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    grouped: list[list[Component]] = [[] for _ in range(char_count)]
    for i, c in enumerate(comps):
        grouped[int(labels[i])].append(c)

    char_boxes: list[tuple[int, int, int, int]] = []
    for g in grouped:
        if not g:
            return None
        bbox = union_bbox(g)
        bbox = pad_bbox(bbox, paper_img.shape[1], paper_img.shape[0], pad_ratio)
        char_boxes.append(bbox)

    expected_right, expected_left = (2, 2) if char_count == 4 else (3, 2)
    ordered = _order_cluster_boxes(char_boxes, expected_right=expected_right, expected_left=expected_left)
    if ordered is None:
        return None

    names = get_slot_names(char_count)
    out: list[tuple[str, np.ndarray]] = []
    for i, (x1, y1, x2, y2) in enumerate(ordered):
        out.append((names[i], paper_img[y1:y2, x1:x2]))
    return out


def extract_chars_by_manual_boxes(
    page_img: np.ndarray,
    page_no: int,
    manual_boxes: dict[str, list[dict[str, float]]],
    slot_names: list[str],
    pad_ratio: float,
    min_area_ratio: float,
    min_short_ratio: float,
    max_line_aspect: float,
    binarize_method: str,
    adaptive_block_size: int,
    adaptive_c: int,
    keep_manual_box_exact: bool,
) -> list[tuple[str, np.ndarray]] | None:
    entries = manual_boxes.get(str(page_no))
    if not entries or len(entries) < len(slot_names):
        return None

    h, w = page_img.shape[:2]
    out: list[tuple[str, np.ndarray]] = []

    for idx, slot_name in enumerate(slot_names):
        e = entries[idx]
        if {"x_ratio", "y_ratio", "w_ratio", "h_ratio"} <= set(e.keys()):
            x = int(round(float(e["x_ratio"]) * w))
            y = int(round(float(e["y_ratio"]) * h))
            bw = int(round(float(e["w_ratio"]) * w))
            bh = int(round(float(e["h_ratio"]) * h))
        else:
            x = int(round(float(e["x"])))
            y = int(round(float(e["y"])))
            bw = int(round(float(e["w"])))
            bh = int(round(float(e["h"])))

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        box_img = page_img[y : y + bh, x : x + bw]
        if keep_manual_box_exact:
            char_img = box_img
        else:
            char_img = tight_crop_with_components(
                box_img,
                pad_ratio=pad_ratio,
                min_area_ratio=min_area_ratio,
                min_short_ratio=min_short_ratio,
                max_line_aspect=max_line_aspect,
                binarize_method=binarize_method,
                adaptive_block_size=adaptive_block_size,
                adaptive_c=adaptive_c,
            )
        out.append((slot_name, char_img))
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_pages(doc: fitz.Document, page_spec: str) -> Iterable[int]:
    if page_spec.lower() == "all":
        return range(doc.page_count)

    result: set[int] = set()
    for chunk in page_spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a_str, b_str = chunk.split("-", 1)
            a, b = int(a_str), int(b_str)
            start, end = min(a, b), max(a, b)
            for p in range(start, end + 1):
                result.add(p - 1)
        else:
            result.add(int(chunk) - 1)

    valid = [p for p in sorted(result) if 0 <= p < doc.page_count]
    return valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 4 or 5 calligraphy chars per page from a PDF.")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--out-dir", default="data/images", help="Output directory for char images")
    parser.add_argument("--csv", default="data/extracted_template.csv", help="Output CSV path")
    parser.add_argument("--sheet-prefix", default="S001", help="Prefix for generated IDs")
    parser.add_argument("--dpi", type=int, default=320, help="Render DPI (300-400 recommended)")
    parser.add_argument(
        "--crop",
        default="0,0,1,1",
        help="Crop region ratio x,y,w,h in page coordinates (default full page)",
    )
    parser.add_argument(
        "--pages",
        default="all",
        help='Pages to process: "all", "1,3,5", or "1-10"',
    )
    parser.add_argument(
        "--method",
        choices=["slots", "cluster"],
        default="cluster",
        help="Extraction method. cluster is more algorithmic, slots is deterministic fallback.",
    )
    parser.add_argument(
        "--char-count",
        type=int,
        choices=[4, 5],
        default=5,
        help="Number of characters per page to extract (4 or 5).",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[-90, 0, 90, 180],
        default=0,
        help="Rotate each page image before extraction",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.08,
        help="Padding ratio around detected ink bbox",
    )
    parser.add_argument(
        "--min-area-ratio",
        type=float,
        default=0.00008,
        help="Minimum connected-component area ratio for noise filtering",
    )
    parser.add_argument(
        "--min-short-ratio",
        type=float,
        default=0.008,
        help="Minimum short-side ratio of components; helps remove thin notes",
    )
    parser.add_argument(
        "--max-line-aspect",
        type=float,
        default=12.0,
        help="Max aspect threshold to reject long thin line-like components",
    )
    parser.add_argument(
        "--binarize-method",
        choices=["adaptive", "otsu"],
        default="adaptive",
        help="Binarization method for detection mask",
    )
    parser.add_argument(
        "--adaptive-block-size",
        type=int,
        default=35,
        help="Adaptive threshold block size (odd number)",
    )
    parser.add_argument(
        "--adaptive-c",
        type=int,
        default=12,
        help="Adaptive threshold C value",
    )
    parser.add_argument(
        "--save-mode",
        choices=["gray", "color", "mask-gray"],
        default="gray",
        help="Output image mode. mask-gray = white background + gray ink only.",
    )
    parser.add_argument(
        "--boxes-json",
        default="",
        help="Manual box JSON from scripts/manual_box_editor.py (uses page-level boxes)",
    )
    parser.add_argument(
        "--keep-manual-box-exact",
        action="store_true",
        help="With --boxes-json, crop exactly by manual boxes without re-tight-cropping",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    crop_ratio = parse_crop(args.crop)
    manual_boxes: dict[str, list[dict[str, float]]] | None = None
    slot_layout = get_slot_layout(args.char_count)
    slot_names = get_slot_names(args.char_count)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if args.boxes_json:
        boxes_path = Path(args.boxes_json).expanduser().resolve()
        if not boxes_path.exists():
            raise FileNotFoundError(f"boxes json not found: {boxes_path}")
        payload = json.loads(boxes_path.read_text(encoding="utf-8"))
        manual_boxes = payload.get("boxes", {})

    ensure_dir(out_dir)
    ensure_dir(csv_path.parent)

    doc = fitz.open(pdf_path)
    page_indices = list(iter_pages(doc, args.pages))
    if not page_indices:
        raise ValueError("No valid pages selected.")

    rows: list[dict[str, str]] = []
    saved_count = 0

    for page_idx in page_indices:
        page_no = page_idx + 1
        page = doc.load_page(page_idx)
        page_img = render_page_to_bgr(page, args.dpi)
        page_img = rotate_image(page_img, args.rotate)
        paper_img = crop_ratio_rect(page_img, crop_ratio)

        extracted: list[tuple[str, np.ndarray]] | None = None
        if manual_boxes is not None:
            extracted = extract_chars_by_manual_boxes(
                page_img=page_img,
                page_no=page_no,
                manual_boxes=manual_boxes,
                slot_names=slot_names,
                pad_ratio=args.pad_ratio,
                min_area_ratio=args.min_area_ratio,
                min_short_ratio=args.min_short_ratio,
                max_line_aspect=args.max_line_aspect,
                binarize_method=args.binarize_method,
                adaptive_block_size=args.adaptive_block_size,
                adaptive_c=args.adaptive_c,
                keep_manual_box_exact=args.keep_manual_box_exact,
            )
            if extracted is None:
                print(f"[warn] page {page_no}: manual boxes not found, fallback to method={args.method}")

        if extracted is None and args.method == "cluster":
            extracted = extract_chars_by_cluster(
                paper_img,
                char_count=args.char_count,
                pad_ratio=args.pad_ratio,
                min_area_ratio=args.min_area_ratio,
                min_short_ratio=args.min_short_ratio,
                max_line_aspect=args.max_line_aspect,
                binarize_method=args.binarize_method,
                adaptive_block_size=args.adaptive_block_size,
                adaptive_c=args.adaptive_c,
            )
            if extracted is None:
                print(f"[warn] page {page_no}: cluster failed, fallback to slots")

        if extracted is None:
            extracted = extract_chars_by_slots(
                paper_img,
                slot_layout=slot_layout,
                pad_ratio=args.pad_ratio,
                min_area_ratio=args.min_area_ratio,
                min_short_ratio=args.min_short_ratio,
                max_line_aspect=args.max_line_aspect,
                binarize_method=args.binarize_method,
                adaptive_block_size=args.adaptive_block_size,
                adaptive_c=args.adaptive_c,
            )

        for slot_idx, (slot_name, char_img) in enumerate(extracted, start=1):
            if args.save_mode == "gray":
                char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)
            elif args.save_mode == "mask-gray":
                clean_mask = build_clean_mask(
                    char_img,
                    min_area_ratio=args.min_area_ratio,
                    min_short_ratio=args.min_short_ratio,
                    max_line_aspect=args.max_line_aspect,
                    binarize_method=args.binarize_method,
                    adaptive_block_size=args.adaptive_block_size,
                    adaptive_c=args.adaptive_c,
                )
                char_img = apply_mask_gray(char_img, clean_mask)
            filename = f"{args.sheet_prefix}_p{page_no:03d}_{slot_idx:02d}.png"
            save_path = out_dir / filename
            ok = cv2.imwrite(str(save_path), char_img)
            if not ok:
                raise RuntimeError(f"Failed to save image: {save_path}")

            rows.append(
                {
                    "sheet_id": f"{args.sheet_prefix}_p{page_no:03d}",
                    "page": str(page_no),
                    "slot": str(slot_idx),
                    "slot_name": slot_name,
                    "image_path": str(Path("data/images") / filename),
                    "char": "",
                    "yomi": "",
                    "style": "",
                    "note": "",
                }
            )
            saved_count += 1

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sheet_id",
                "page",
                "slot",
                "slot_name",
                "image_path",
                "char",
                "yomi",
                "style",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done: pages={len(page_indices)}, images={saved_count}, method={args.method}")
    print(f"Images: {out_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
