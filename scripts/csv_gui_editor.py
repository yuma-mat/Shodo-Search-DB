#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk  # type: ignore
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None


EDITABLE_COLUMNS = ["char", "yomi", "style", "note"]


class CsvGuiEditor:
    def __init__(self, root: tk.Tk, csv_path: Path) -> None:
        self.root = root
        self.csv_path = csv_path
        self.base_dir = csv_path.parent
        self.rows: list[dict[str, str]] = []
        self.columns: list[str] = []
        self.selected_index: int | None = None
        self.preview_img = None

        self.root.title("CSV GUI Editor")
        self.root.geometry("1180x760")

        self._build_ui()
        self._load_csv()

    def _build_ui(self) -> None:
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main, padding=10)
        main.add(left, weight=4)
        main.add(right, weight=2)

        toolbar = ttk.Frame(left, padding=6)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="保存", command=self.save_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="別名で保存", command=self.save_as_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="再読込", command=self._load_csv).pack(side=tk.LEFT, padx=4)
        self.status_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.LEFT, padx=10)

        self.tree = ttk.Treeview(left, show="headings")
        yscroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.tree.yview)
        xscroll = ttk.Scrollbar(left, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.TOP)
        yscroll.pack(fill=tk.Y, side=tk.RIGHT)
        xscroll.pack(fill=tk.X, side=tk.BOTTOM)

        self.tree.bind("<<TreeviewSelect>>", self.on_row_select)
        self.tree.bind("<Double-1>", self.on_double_click)

        form = ttk.LabelFrame(right, text="編集", padding=10)
        form.pack(fill=tk.X, pady=(0, 10))

        self.entry_vars: dict[str, tk.StringVar] = {}
        for col in EDITABLE_COLUMNS:
            row = ttk.Frame(form)
            row.pack(fill=tk.X, pady=4)
            ttk.Label(row, text=col, width=9).pack(side=tk.LEFT)
            var = tk.StringVar()
            ent = ttk.Entry(row, textvariable=var)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.entry_vars[col] = var

        btns = ttk.Frame(form)
        btns.pack(fill=tk.X, pady=(8, 2))
        ttk.Button(btns, text="行に反映", command=self.apply_form_to_row).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="次の行", command=self.select_next).pack(side=tk.LEFT, padx=4)

        preview_frame = ttk.LabelFrame(right, text="画像プレビュー", padding=8)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        self.preview_label = ttk.Label(preview_frame, text="画像なし")
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self.path_var = tk.StringVar(value="")
        ttk.Label(preview_frame, textvariable=self.path_var, wraplength=300).pack(fill=tk.X, pady=(6, 0))

    def _load_csv(self) -> None:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("CSVヘッダがありません")
            self.columns = list(reader.fieldnames)
            self.rows = [{k: (v or "") for k, v in row.items()} for row in reader]

        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = self.columns
        for col in self.columns:
            w = 110
            if col in ("image_path", "sheet_id"):
                w = 190
            if col in ("char", "slot", "page"):
                w = 70
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor=tk.CENTER if col in ("char", "slot", "page") else tk.W)

        for i, row in enumerate(self.rows):
            values = [row.get(c, "") for c in self.columns]
            self.tree.insert("", tk.END, iid=str(i), values=values)

        self.status_var.set(f"{len(self.rows)} 行を読込: {self.csv_path}")
        if self.rows:
            self.tree.selection_set("0")
            self.tree.focus("0")
            self.on_row_select()

    def on_double_click(self, _event=None) -> None:
        self.on_row_select()

    def on_row_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        self.selected_index = idx
        row = self.rows[idx]
        for col in EDITABLE_COLUMNS:
            self.entry_vars[col].set(row.get(col, ""))
        self._update_preview(row.get("image_path", ""))

    def _update_preview(self, image_path_text: str) -> None:
        image_path_text = (image_path_text or "").strip()
        self.path_var.set(image_path_text)
        if not image_path_text:
            self.preview_label.configure(image="", text="画像なし")
            self.preview_img = None
            return

        p = Path(image_path_text)
        if not p.is_absolute():
            p = (self.base_dir.parent / image_path_text).resolve()
        if not p.exists() or Image is None or ImageTk is None:
            self.preview_label.configure(image="", text="プレビュー不可")
            self.preview_img = None
            return

        img = Image.open(p).convert("RGB")
        img.thumbnail((360, 540))
        self.preview_img = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=self.preview_img, text="")

    def apply_form_to_row(self) -> None:
        if self.selected_index is None:
            return
        row = self.rows[self.selected_index]
        for col in EDITABLE_COLUMNS:
            row[col] = self.entry_vars[col].get().strip()
        values = [row.get(c, "") for c in self.columns]
        self.tree.item(str(self.selected_index), values=values)
        self.status_var.set(f"行 {self.selected_index + 1} を更新")

    def select_next(self) -> None:
        if self.selected_index is None:
            return
        nxt = min(len(self.rows) - 1, self.selected_index + 1)
        self.tree.selection_set(str(nxt))
        self.tree.focus(str(nxt))
        self.tree.see(str(nxt))
        self.on_row_select()

    def save_csv(self) -> None:
        self._save_to(self.csv_path)

    def save_as_csv(self) -> None:
        target = filedialog.asksaveasfilename(
            title="CSVを保存",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=self.csv_path.name,
        )
        if not target:
            return
        self._save_to(Path(target))

    def _save_to(self, path: Path) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.rows)
        self.status_var.set(f"保存しました: {path}")
        messagebox.showinfo("保存", f"保存しました\n{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI editor for extracted template CSV.")
    parser.add_argument("--csv", required=True, help="CSV file path")
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    root = tk.Tk()
    CsvGuiEditor(root, csv_path)
    root.mainloop()


if __name__ == "__main__":
    main()
