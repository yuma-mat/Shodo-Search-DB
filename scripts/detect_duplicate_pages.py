#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np


@dataclass(frozen=True)
class PageFeature:
    page_no: int
    img: np.ndarray  # uint8 gray
    phash: int


def parse_crop(crop_text: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in crop_text.split(",")]
    if len(parts) != 4:
        raise ValueError("crop must be 4 comma-separated values: x,y,w,h")
    x, y, w, h = map(float, parts)
    for v in (x, y, w, h):
        if not (0.0 <= v <= 1.0):
            raise ValueError("crop values must be in [0.0, 1.0]")
    if x + w > 1.0 or y + h > 1.0:
        raise ValueError("crop rectangle must fit inside page")
    return x, y, w, h


def iter_pages(page_count: int, page_spec: str) -> list[int]:
    if page_spec.lower() == "all":
        return list(range(page_count))

    result: set[int] = set()
    for chunk in page_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a_str, b_str = chunk.split("-", 1)
            a, b = int(a_str), int(b_str)
            start, end = min(a, b), max(a, b)
            for p in range(start, end + 1):
                result.add(p - 1)
        else:
            result.add(int(chunk) - 1)
    return [p for p in sorted(result) if 0 <= p < page_count]


def render_page_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def crop_ratio_rect(img: np.ndarray, rect_ratio: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    rx, ry, rw, rh = rect_ratio
    x1 = max(0, int(round(w * rx)))
    y1 = max(0, int(round(h * ry)))
    x2 = min(w, int(round(w * (rx + rw))))
    y2 = min(h, int(round(h * (ry + rh))))
    return img[y1:y2, x1:x2]


def build_manual_boxes_composite(
    page_img: np.ndarray,
    page_no: int,
    manual_boxes: dict[str, list[dict[str, float]]],
    cell_size: int = 128,
) -> np.ndarray | None:
    entries = manual_boxes.get(str(page_no))
    if not entries or len(entries) < 5:
        return None

    h, w = page_img.shape[:2]
    gray_page = cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY)
    canvas = np.full((cell_size * 3, cell_size * 2), 255, dtype=np.uint8)
    # index: 0=right_top,1=right_middle,2=right_bottom,3=left_top,4=left_middle
    positions = [(0, 1), (1, 1), (2, 1), (0, 0), (1, 0)]

    for idx in range(5):
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

        cell = gray_page[y : y + bh, x : x + bw]
        cell = cv2.resize(cell, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        row, col = positions[idx]
        y1, y2 = row * cell_size, (row + 1) * cell_size
        x1, x2 = col * cell_size, (col + 1) * cell_size
        canvas[y1:y2, x1:x2] = cell

    return canvas


def normalize_gray(img: np.ndarray, size: int = 256) -> np.ndarray:
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return norm


def compute_phash(gray_img: np.ndarray) -> int:
    small = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    low = dct[:8, :8]
    values = low.flatten()
    median = float(np.median(values[1:]))  # skip DC
    bits = 0
    for i, v in enumerate(values):
        if v > median:
            bits |= 1 << i
    return bits


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def ssim_score(img1: np.ndarray, img2: np.ndarray) -> float:
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = num / np.maximum(den, 1e-12)
    return float(np.mean(ssim_map))


def orb_match_score(img1: np.ndarray, img2: np.ndarray) -> tuple[float, int]:
    orb = cv2.ORB_create(nfeatures=700)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    n1, n2 = len(kp1), len(kp2)
    min_kp = min(n1, n2)
    if des1 is None or des2 is None or min_kp == 0:
        return 0.0, min_kp

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = 0
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good += 1
    score = good / float(max(1, min_kp))
    return float(score), min_kp


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicated (or near-duplicated) calligraphy pages.")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--pages", default="all", help='Pages to process: "all", "1,3,5", or "1-10"')
    parser.add_argument("--dpi", type=int, default=140, help="Render DPI for duplicate detection")
    parser.add_argument("--crop", default="0,0,1,1", help="Crop region ratio x,y,w,h (used when no boxes-json)")
    parser.add_argument(
        "--boxes-json",
        default="",
        help="Manual box JSON. If provided, compares page by 5-box composite instead of full page",
    )
    parser.add_argument("--phash-threshold", type=int, default=8, help="Max pHash Hamming distance for candidates")
    parser.add_argument("--ssim-threshold", type=float, default=0.970, help="SSIM threshold for duplicate")
    parser.add_argument("--orb-threshold", type=float, default=0.18, help="ORB good-match ratio threshold")
    parser.add_argument("--out-csv", default="data/duplicate_pages.csv", help="Output CSV for duplicate pairs")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    crop_ratio = parse_crop(args.crop)
    manual_boxes: dict[str, list[dict[str, float]]] | None = None
    if args.boxes_json:
        boxes_path = Path(args.boxes_json).expanduser().resolve()
        if not boxes_path.exists():
            raise FileNotFoundError(f"boxes json not found: {boxes_path}")
        payload = json.loads(boxes_path.read_text(encoding="utf-8"))
        manual_boxes = payload.get("boxes", {})
        if not isinstance(manual_boxes, dict):
            raise ValueError("Invalid boxes JSON: 'boxes' must be object")

    doc = fitz.open(pdf_path)
    page_indices = iter_pages(doc.page_count, args.pages)
    if not page_indices:
        raise ValueError("No valid pages selected")

    feats: list[PageFeature] = []
    for page_idx in page_indices:
        page_no = page_idx + 1
        page = doc.load_page(page_idx)
        page_img = render_page_to_bgr(page, args.dpi)

        if manual_boxes is not None:
            comp = build_manual_boxes_composite(page_img, page_no, manual_boxes)
            if comp is None:
                cropped = crop_ratio_rect(page_img, crop_ratio)
                norm = normalize_gray(cropped)
            else:
                norm = normalize_gray(comp)
        else:
            cropped = crop_ratio_rect(page_img, crop_ratio)
            norm = normalize_gray(cropped)

        feats.append(PageFeature(page_no=page_no, img=norm, phash=compute_phash(norm)))

    pair_rows: list[dict[str, str]] = []
    uf = UnionFind(len(feats))
    candidate_count = 0

    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            ham = hamming_distance(feats[i].phash, feats[j].phash)
            if ham > args.phash_threshold:
                continue

            candidate_count += 1
            ssim = ssim_score(feats[i].img, feats[j].img)
            orb, min_kp = orb_match_score(feats[i].img, feats[j].img)
            orb_ok = orb >= args.orb_threshold
            low_texture_ok = min_kp < 20 and ssim >= max(args.ssim_threshold, 0.985)
            is_dup = ssim >= args.ssim_threshold and (orb_ok or low_texture_ok)

            if is_dup:
                uf.union(i, j)
                pair_rows.append(
                    {
                        "page_a": str(feats[i].page_no),
                        "page_b": str(feats[j].page_no),
                        "phash_hamming": str(ham),
                        "ssim": f"{ssim:.5f}",
                        "orb_score": f"{orb:.5f}",
                        "min_keypoints": str(min_kp),
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["page_a", "page_b", "phash_hamming", "ssim", "orb_score", "min_keypoints"],
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    groups: dict[int, list[int]] = {}
    for idx, feat in enumerate(feats):
        root = uf.find(idx)
        groups.setdefault(root, []).append(feat.page_no)
    dup_groups = [sorted(v) for v in groups.values() if len(v) > 1]
    dup_groups.sort(key=lambda g: (len(g), g[0]), reverse=True)

    print(f"pages={len(feats)}")
    print(f"candidates(phash<={args.phash_threshold})={candidate_count}")
    print(f"duplicate_pairs={len(pair_rows)}")
    print(f"duplicate_groups={len(dup_groups)}")
    if dup_groups:
        print("groups:")
        for g in dup_groups:
            print("  - " + ",".join(map(str, g)))
    print(f"csv={out_csv}")


if __name__ == "__main__":
    main()
