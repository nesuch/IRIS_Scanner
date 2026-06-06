#!/usr/bin/env python3
"""Convert IRDAI Handbook 'Summary' sheets into IRIS tidy financial rows.

The Handbook ships ~100 print-formatted statistical tables. The Summary sheets
A/B/C are clean industry time-series (Particular | Remarks | Unit | <years...>).
This script un-pivots A/B/C into the long format IRIS's data engine ingests and
writes a single tidy .xlsx into knowledge_base/raw_submissions/, so the existing
Admin -> "Sync Data" flow picks it up automatically (no engine changes needed).

Output columns: dimension, entity, metric, value, financial_year, quarter,
                line_of_business, class_of_business

Usage:
    python tools/handbook_to_iris.py "/path/to/Summary.xlsx"
    python tools/handbook_to_iris.py "/path/to/Summary.xlsx" --out knowledge_base/raw_submissions
"""
import argparse
import os
import re
import sys

import openpyxl
import pandas as pd

# Summary sheet -> the industry entity it describes.
SHEET_ENTITY = {
    "A": "Indian Insurance Sector (All)",
    "B": "Life Insurance Sector",
    "C": "General Insurance Sector",
}
YEAR_RE = re.compile(r"^\d{4}-\d{2}$")        # e.g. 2024-25
DIMENSION = "Industry"
QUARTER = "Annual"
DEFAULT_LOB = "All Lines"
DEFAULT_CLASS = "All Classes"
# Contextless aggregate rows to drop entirely: a bare "Total" is just the sum of
# the segment cells above it, so on its own (no context) it is meaningless.
SKIP_PARTICULARS = {"total"}

# Segment names that are Lines of Business (not metrics). When one of these
# appears as a row under a LOB-breakdown section, the *section* is the metric
# (e.g. "Incurred Claims Ratio") and the segment is the Line of Business.
LOB_VOCAB = {
    "fire", "marine", "marine cargo", "marine hull", "motor", "motor od",
    "motor tp", "health", "engineering", "aviation", "liability",
    "personal accident", "crop", "credit", "miscellaneous", "others",
}

# A section whose header matches one of these keys is a per-LOB breakdown; the
# value maps the messy header to a clean metric name.
LOB_SECTIONS = [
    ("gross direct premium", "Gross Direct Premium"),
    ("net retention", "Net Retention"),
    ("incurred claims ratio", "Incurred Claims Ratio"),
]

# Sections whose children are members of a category (a channel, a region, a
# fund) rather than distinct metrics. The child name is ambiguous on its own
# (e.g. "Online", "Others if any"), so the metric is prefixed with this label.
BREAKDOWN_SECTIONS = [
    (("channel", "premium"), "New Business Premium by Channel"),
    (("channel", "lives"), "New Business Lives Covered by Channel"),
    (("region",), "Offices by Region"),
    (("assets under management",), "Assets Under Management"),
]


def _lob_section_metric(section):
    low = section.lower()
    for key, name in LOB_SECTIONS:
        if key in low:
            return name
    return None


def _breakdown_label(section):
    low = section.lower()
    for keys, name in BREAKDOWN_SECTIONS:
        if all(k in low for k in keys):
            return name
    return None


def _clean_text(v):
    return re.sub(r"\s+", " ", str(v or "").replace("\n", " ")).strip()


def _clean_metric(particular, unit):
    name = _clean_text(particular).rstrip("#").strip()
    unit = _clean_text(unit)
    # Keep the unit on the metric name (IRIS has no unit column) when meaningful.
    if unit and unit.lower() not in {"none", "nan", "unit", ""}:
        name = f"{name} ({unit})"
    return name


def _clean_section(particular):
    """Tidy a section-header label, e.g. 'INCURRED CLAIMS RATIO' -> 'Incurred Claims Ratio'."""
    s = _clean_text(particular).rstrip("#").strip()
    # Drop trailing parentheticals/footnotes that add no meaning to the label.
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s).strip(" .")
    return s.title() if s.isupper() else s


def _to_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        # Strip floating-point noise (e.g. 3.9000000000001 -> 3.9).
        return round(float(v), 4)
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "NA", "N.A.", "na", "nan", "None"}:
        return None
    try:
        return round(float(s), 4)
    except ValueError:
        return None


def _find_header(ws, max_scan=6):
    """Return (header_row_index, {col_index: financial_year}, particular_col, unit_col)."""
    for r, row in enumerate(ws.iter_rows(min_row=1, max_row=max_scan, values_only=True), start=1):
        first = _clean_text(row[0]).lower() if row else ""
        if first.startswith("particular"):
            years, particular_col, unit_col = {}, 0, None
            for ci, cell in enumerate(row):
                txt = _clean_text(cell)
                if YEAR_RE.match(txt):
                    years[ci] = txt
                elif txt.lower() == "unit":
                    unit_col = ci
            if years:
                return r, years, particular_col, unit_col
    return None


def convert_sheet(ws, entity, source):
    info = _find_header(ws)
    if not info:
        print(f"   [!] no header/year row found in '{ws.title}' — skipped")
        return []
    header_row, year_cols, p_col, unit_col = info
    rows = []
    current_section = ""
    section_metric = None  # set when the current section is a per-LOB breakdown
    # Track which sections each (non-LOB) metric name appears under, so any name
    # that recurs across sections can be disambiguated — otherwise duplicates
    # collapse and the pivot SUMS them into a meaningless figure.
    metric_sections = {}
    for row in ws.iter_rows(min_row=header_row + 1, values_only=True):
        particular = row[p_col] if p_col < len(row) else None
        if not _clean_text(particular):
            continue
        unit = row[unit_col] if (unit_col is not None and unit_col < len(row)) else None
        values = {fy: _to_number(row[ci]) for ci, fy in year_cols.items() if ci < len(row)}
        has_data = any(v is not None for v in values.values())

        # A section-header row has a label but no unit and no numeric values.
        if not has_data and not _clean_text(unit):
            current_section = _clean_section(particular)
            section_metric = _lob_section_metric(current_section)
            continue

        p_clean = _clean_text(particular).rstrip("#").strip()
        if p_clean.lower() in SKIP_PARTICULARS:
            continue  # drop contextless "Total" aggregate rows

        is_lob_row = section_metric and p_clean.lower() in LOB_VOCAB
        if is_lob_row:
            # The segment is the Line of Business; the section is the metric.
            metric = _clean_metric(section_metric, unit)
            lob = p_clean.title()
        else:
            breakdown = _breakdown_label(current_section)
            if breakdown:
                # Child is a category member (channel/region/fund) — prefix the
                # section so e.g. "Online" reads "...by Channel — Online".
                metric = _clean_metric(f"{breakdown} — {p_clean}", unit)
            else:
                metric = _clean_metric(particular, unit)
                metric_sections.setdefault(metric, set()).add(current_section)
            lob = DEFAULT_LOB

        for fy, value in values.items():
            if value is None:
                continue
            rows.append({
                "dimension": DIMENSION,
                "entity": entity,
                "metric": metric,
                "section": "" if is_lob_row else current_section,
                "value": value,
                "financial_year": fy,
                "quarter": QUARTER,
                "line_of_business": lob,
                "class_of_business": DEFAULT_CLASS,
            })

    # Disambiguate only the (non-LOB) metric names that genuinely collide across
    # sections; unique metrics keep their clean "<Particular> (<Unit>)" name.
    for r in rows:
        sec = r.pop("section")
        if sec and len(metric_sections.get(r["metric"], ())) > 1:
            r["metric"] = f"{sec} — {r['metric']}"
    return rows


def main():
    ap = argparse.ArgumentParser(description="Convert IRDAI Handbook Summary into IRIS tidy rows.")
    ap.add_argument("summary_xlsx", help="Path to the Handbook 'Summary.xlsx'")
    ap.add_argument("--out", default=os.path.join("knowledge_base", "raw_submissions"),
                    help="Output directory (default: knowledge_base/raw_submissions)")
    ap.add_argument("--name", default="handbook_2024-25_summary.xlsx", help="Output filename")
    args = ap.parse_args()

    if not os.path.exists(args.summary_xlsx):
        sys.exit(f"Input not found: {args.summary_xlsx}")

    wb = openpyxl.load_workbook(args.summary_xlsx, read_only=True, data_only=True)
    source = os.path.basename(args.summary_xlsx)
    all_rows = []
    for sheet, entity in SHEET_ENTITY.items():
        if sheet not in wb.sheetnames:
            print(f"   [!] sheet '{sheet}' not in workbook — skipped")
            continue
        rows = convert_sheet(wb[sheet], entity, source)
        print(f"   sheet {sheet} -> {entity}: {len(rows)} rows")
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No rows produced — nothing to write.")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, args.name)
    df = pd.DataFrame(all_rows, columns=[
        "dimension", "entity", "metric", "value", "financial_year",
        "quarter", "line_of_business", "class_of_business",
    ])
    df.to_excel(out_path, index=False)
    print(f"\n[+] Wrote {len(df)} tidy rows -> {out_path}")
    print(f"    entities: {sorted(df['entity'].unique())}")
    print(f"    years: {sorted(df['financial_year'].unique())}")
    print(f"    metrics: {df['metric'].nunique()} unique")


if __name__ == "__main__":
    main()
