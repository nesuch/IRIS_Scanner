#!/bin/zsh
# ocr.sh <input.pdf> <output.pdf>
#
# Re-OCRs a scanned gazette into a SEARCHABLE pdf: the original page image with a
# real text layer behind it. Keeps the page images so the document stays readable
# and citable, and makes the text selectable/editable for later use.
#
# pdftoppm renders each page, tesseract OCRs them and — given a list file — emits a
# single multi-page PDF, which avoids needing a separate merge tool.
set -e
IN="$1"; OUT="$2"; DPI="${3:-300}"
[ -f "$IN" ] || { echo "no such file: $IN"; exit 1; }
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

pdftoppm -r "$DPI" -gray -png "$IN" "$WORK/pg"
ls "$WORK"/pg-*.png | sort > "$WORK/list.txt"
N=$(wc -l < "$WORK/list.txt" | tr -d ' ')

# --psm 6 (assume a single uniform block) beats the automatic mode on these
# single-column gazettes: measured on the 2000 IAC scan it recovered a section
# heading that psm 1 lost ("3. Procedure for meetings" vs nothing), which matters
# because a lost NUMBER means a lost clause boundary downstream. Same cost.
tesseract "$WORK/list.txt" "${OUT%.pdf}" --psm 6 -l eng pdf 2>/dev/null

echo "  $N pages -> $(basename "$OUT") ($(du -h "$OUT" | cut -f1))"
