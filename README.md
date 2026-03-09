# 書道手本 1文字検索（ローカル運用）

父親向けに、ブラウザだけで使える「1文字検索」画面です。  
サーバー不要・無料で運用できます。

## 1. できること

- 文字で検索（例: `永`）
- 読みで検索（ひらがな/カタカナどちらでも可）
- 書体で絞り込み
- 結果件数の表示
- 手本作成タブで、1〜6文字を半紙レイアウトに自動配置

## 2. ファイル構成

- `index.html`: 検索画面本体
- `data/characters.js`: 1文字データ（このファイルを編集）
- `data/images/`: 事前に切り出した1文字画像を置くフォルダ

## 3. データ登録ルール

`data/characters.js` の `characters` 配列に、1文字ごとに1レコード追加します。

最低限必要な項目:

- `char`: 文字
- `yomi`: 読み（検索用）
- `style`: 書体
- `imagePath`: 文字画像ファイルのパス（例: `./data/images/S001_01_永.png`）

運用上おすすめの項目:

- `sheetId`: どの手本から切り出したか（例: `S023`）
- `note`: 補足

例:

```javascript
{
  char: "永",
  yomi: "えい",
  style: "楷書",
  imagePath: "./data/images/S001_01_永.png",
  sheetId: "S001",
  note: "永字八法"
}
```

## 4. 父親が使う手順

1. `index.html` をダブルクリックしてブラウザで開く
2. 検索欄に文字または読みを入力
3. 必要なら書体を選ぶ
4. `検索する` を押す（検索結果に文字画像が表示される）

手本作成を使う場合:

1. `手本作成` タブを開く
2. 1〜6文字を入力して `配置する` を押す
3. 半紙比率（縦33.4 × 横24.3）に配置された結果を確認する

配置順（6マス時）:

- 右上 → 右真ん中 → 右下 → 左上 → 左真ん中 → 左下

文字数ごとの配置:

- 1文字: 全面
- 2文字: 上半分 / 下半分
- 3文字: 4等分して右上・右下・左上
- 4文字: 4等分して右上・右下・左上・左下
- 5文字: 2列3行で右列3マス + 左列上2マス
- 6文字: 2列3行すべて

## 5. あなた（管理者）が行う作業

1. PDFから1文字ずつ切り出し（画像化）
2. 画像を `data/images/` に保存
3. 各文字の `char` / `yomi` / `style` / `imagePath` を `data/characters.js` に入力
4. `index.html` を開いて検索確認

手本が増えない前提なので、この静的ファイル構成が最小コストです。  
将来、元の手本ページへのリンクを足したくなった場合も、同じ構成のまま拡張できます。

## 6. OpenCVでPDFから5文字を自動切り出し

### 追加ファイル

- `scripts/extract_chars_from_pdf.py`
- `requirements.txt`

### 事前準備

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 実行例

```bash
python3 scripts/extract_chars_from_pdf.py \
  --pdf "/Users/yuma/projects/opencv-test/assets/①楷書（五文字）.pdf" \
  --out-dir "data/images" \
  --csv "data/extracted_template.csv" \
  --sheet-prefix "KAI" \
  --method "cluster" \
  --binarize-method "adaptive" \
  --save-mode "mask-gray"
```

### 生成されるもの

- `data/images/KAI_p001_01.png` のような1文字画像（ページごとに5枚）
- `data/extracted_template.csv`（`char`, `yomi`, `style` は空欄で出力）

### 補正パラメータ（必要時）

- `--crop x,y,w,h`
  - PDF全体のどこを半紙領域として使うかを比率指定
  - 例: `--crop 0.08,0.04,0.84,0.92`
- `--dpi`
  - 解像度（300〜400推奨）
- `--pad-ratio`
  - 文字の外接枠まわりの余白
- `--rotate`
  - PDFが回転している場合（`-90`, `90`, `180`）
- `--method`
  - `cluster`: クラスター分析で5文字を推定（推奨）
  - `slots`: 固定5枠で切り出し
- `--min-area-ratio`
  - 小さすぎる成分を除去（ノイズ除去）
- `--min-short-ratio`
  - 短辺が細すぎる成分を除去（細いメモ線対策）
- `--max-line-aspect`
  - 長く細い線状成分を除去
- `--binarize-method`
  - `adaptive`（推奨）/ `otsu` を選択
- `--adaptive-block-size`
  - 適応的二値化の局所窓サイズ（奇数）
- `--adaptive-c`
  - 適応的二値化のC値
- `--save-mode`
  - `gray`: グレースケール保存
  - `color`: カラー保存
  - `mask-gray`: 背景白 + 文字のみグレー保存（文字マスク適用）

推奨方針:

- 二値化は抽出用だけに使う
- 最終保存は `--save-mode mask-gray` か `--save-mode gray` にする

### 配置ルール（5文字）

- 右上
- 右真ん中
- 右下
- 左上
- 左真ん中

## 7. GUIで5枠を手動調整してから抽出

### 1) 枠をGUIで調整

```bash
python3 scripts/manual_box_editor.py \
  --pdf "/Users/yuma/projects/opencv-test/assets/①楷書（五文字）.pdf" \
  --out-json "data/manual_boxes.json"
```

操作:

- マウスドラッグ: 枠の移動
- 角ハンドルドラッグ: 枠サイズ変更
- `1`〜`5`: 枠選択
- `n` or `Enter`: 次ページ
- `p`: 前ページ
- `r`: 現在ページを初期枠に戻す
- `s`: JSON保存
- `q`: 保存して終了

### 2) 調整済み枠で抽出

```bash
python3 scripts/extract_chars_from_pdf.py \
  --pdf "/Users/yuma/projects/opencv-test/assets/①楷書（五文字）.pdf" \
  --boxes-json "data/manual_boxes.json" \
  --out-dir "data/images" \
  --csv "data/extracted_template.csv" \
  --sheet-prefix "KAI" \
  --save-mode "mask-gray"
```

補足:

- `--boxes-json` を指定した場合、ページ単位の手動枠を優先して抽出します。

## 8. CSVから読みを補完してJSON/JSを自動生成

### 期待するCSV列

- `char`（必須）
- `style`（必須）
- `yomi`（任意。空なら自動生成）
- `image_path`（任意）
- `sheet_id`（任意）
- `note`（任意）

### 実行

```bash
python3 scripts/csv_to_characters.py \
  --input-csv "data/extracted_template.csv" \
  --output-json "data/characters.json" \
  --output-js "data/characters.js"
```

### 補足

- `pykakasi` が入っていれば `yomi` を自動補完します。
- `yomi` を空許容にしたい場合は `--allow-empty-yomi` を指定します。
- 複数読みを優先したい場合は `--readings-json "data/readings_map.example.json"` を指定します。
