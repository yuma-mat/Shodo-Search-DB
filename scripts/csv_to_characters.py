#!/usr/bin/env python3
"""
Convert CSV to JSON/JS for this app, with optional yomi auto-generation.

Input CSV columns (recommended):
  char, yomi, style, image_path, sheet_id, note
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def to_hiragana(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def build_kakasi_converter():
    try:
        import pykakasi  # type: ignore
    except ImportError:
        return None
    return pykakasi.kakasi()


def infer_yomi(text: str, kakasi_converter: Any) -> str:
    if not text:
        return ""
    # If already kana-only, normalize to hiragana and use as-is.
    if all(
        ("\u3040" <= ch <= "\u309f")
        or ("\u30a0" <= ch <= "\u30ff")
        or ch.isspace()
        for ch in text
    ):
        return to_hiragana(text).replace(" ", "").strip()
    if kakasi_converter is None:
        return ""
    parts = kakasi_converter.convert(text)
    return "".join(p.get("hira", "") for p in parts).strip()


def pick(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            return str(row[key]).strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data/characters.js from CSV.")
    parser.add_argument("--input-csv", required=True, help="Input CSV path")
    parser.add_argument("--output-js", default="data/characters.js", help="Output JS path")
    parser.add_argument("--output-json", default="data/characters.json", help="Output JSON path")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding")
    parser.add_argument(
        "--readings-json",
        default="",
        help='Optional JSON map for multi-readings. e.g. {"月":["つき","げつ"]}',
    )
    parser.add_argument(
        "--allow-empty-yomi",
        action="store_true",
        help="Allow empty yomi when conversion is unavailable",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_js = Path(args.output_js).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")
    readings_map: dict[str, str] = {}
    if args.readings_json:
        readings_path = Path(args.readings_json).expanduser().resolve()
        if not readings_path.exists():
            raise FileNotFoundError(f"readings json not found: {readings_path}")
        payload = json.loads(readings_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("readings json must be an object map")
        for k, v in payload.items():
            key = str(k).strip()
            if not key:
                continue
            if isinstance(v, list):
                vals = [str(x).strip() for x in v if str(x).strip()]
                readings_map[key] = "、".join(vals)
            else:
                readings_map[key] = str(v).strip()

    output_js.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    kakasi_converter = build_kakasi_converter()
    if kakasi_converter is None:
        print("[warn] pykakasi is not installed. yomi auto-generation is disabled.")

    rows_out: list[dict[str, str]] = []
    with input_csv.open("r", encoding=args.encoding, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing.")

        for i, row in enumerate(reader, start=2):
            char = pick(row, "char", "文字")
            style = pick(row, "style", "書体")
            if not char:
                continue
            if not style:
                raise ValueError(f"line {i}: style is required")

            yomi = pick(row, "yomi", "読み")
            if char in readings_map and readings_map[char]:
                yomi = readings_map[char]
            if not yomi:
                yomi = infer_yomi(char, kakasi_converter)
            if not yomi:
                note_for_yomi = pick(row, "note", "備考")
                yomi = infer_yomi(note_for_yomi, kakasi_converter)
            if not yomi and not args.allow_empty_yomi:
                raise ValueError(
                    f"line {i}: yomi is empty. Install pykakasi or fill yomi/note in CSV "
                    f"(char={char}, style={style})"
                )

            image_path = pick(row, "image_path", "imagePath", "画像")
            sheet_id = pick(row, "sheet_id", "sheetId", "手本ID")
            note = pick(row, "note", "備考")

            rows_out.append(
                {
                    "char": char,
                    "yomi": yomi,
                    "style": style,
                    "imagePath": image_path,
                    "sheetId": sheet_id,
                    "note": note,
                }
            )

    output_json.write_text(
        json.dumps(rows_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    js_text = "window.CHARACTERS = " + json.dumps(rows_out, ensure_ascii=False, indent=2) + ";\n"
    output_js.write_text(js_text, encoding="utf-8")

    print(f"Done: {len(rows_out)} rows")
    print(f"JSON: {output_json}")
    print(f"JS: {output_js}")


if __name__ == "__main__":
    main()
