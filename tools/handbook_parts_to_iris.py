#!/usr/bin/env python3
"""Convert IRDAI Handbook *Part I-V* per-insurer tables into IRIS tidy rows.

The Handbook Parts hold ~104 statistical tables in many shapes. This tool
handles **Phase 1**: the clean "insurer (or reinsurer) × year" tables, which map
directly onto IRIS's existing Insurer dimension (entity = insurer, one metric
per table, value per financial year). Wide multi-segment matrices, state-wise
and balance-sheet tables are out of scope here (later phases).

Output columns: dimension, entity, metric, value, financial_year, quarter,
                line_of_business, class_of_business
Writes knowledge_base/raw_submissions/handbook_2024-25_parts.xlsx so Admin ->
"Sync Data" ingests it with no engine changes.
"""
import argparse
import os
import re
import sys

import openpyxl
import pandas as pd

DIMENSION = "Insurer"
QUARTER = "Annual"
DEFAULT_CLASS = "All Classes"

YEAR_RE = re.compile(r"^(?:\d{4}-\d{2}|\d{4})$")     # 2023-24 or 2023
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")              # first number, ignoring footnote marks
ENTITY_AXIS = {"insurer", "reinsurer", "reinsurers", "company", "name of the insurer"}

# Phase-2 LOB matrices: (file, sheet). LOB + metric are read from the multi-row
# column headers; the parser auto-detects 2-level (LOB>year) vs 3-level
# (LOB>metric>year) layouts. Rows are insurers.
PHASE2 = [
    ("Part II", "41"),   # Segment-wise Gross Direct Premium (2-level)
    ("Part II", "44"),   # Net Premium / Claims Incurred / ICR by segment (3-level)
    ("Part IV", "83"),   # Reinsurers: NEP / Claims / ICR by segment (metric>LOB>year)
]

# Phase-2b/2c: transposed (insurers/reinsurers as columns; metrics are rows).
PHASE2B_TRANSPOSED = [
    ("Part II", "45", "General"),       # Underwriting Experience of insurers
    ("Part IV", "84", "Reinsurance"),   # Underwriting Experience of reinsurers
]
PHASE2B_CLASS = [
    ("Part III", "58", "Health"),
    ("Part III", "59", "Personal Accident"),
    ("Part III", "60", "Travel (Overseas)"),
    ("Part III", "61", "Travel (Domestic)"),
    ("Part III", "62", "Health"),
    ("Part III", "63", "Personal Accident"),
    ("Part III", "64", "Travel (Overseas)"),
    ("Part III", "65", "Travel (Domestic)"),
]

# Phase-1 tables: (file, sheet, line_of_business). Metric is derived from the title.
PHASE1 = [
    ("Part I", "2", "Life"),
    ("Part I", "3", "Life"),
    ("Part I", "9", "Life"),
    ("Part I", "22", "Life"),
    ("Part II", "40", "General"),
    ("Part IV", "86", "Reinsurance"),
    # Tables 90-92 are agents *of life insurers* — so Life, not "All Lines".
    ("Part V", "90", "Life"),
    ("Part V", "91", "Life"),
    ("Part V", "92", "Life"),
]

# Rows whose entity label is a group header / aggregate, not a real insurer.
SKIP_ENTITIES = re.compile(
    r"^(public sector|private sector|standalone health|stand-alone health|"
    r"speciali[sz]ed|grand total|industry|total|sub[- ]?total).*?$|.*\btotal$",
    re.I,
)
UNIT_RE = re.compile(r"\((₹\s?crore|in lakhs|amount in ₹lakh|₹\s?lakh|lakhs|in crore|"
                     r"us ?\$|per ?cent|nos\.?|number|₹.*?)\)", re.I)


def _clean(v):
    return re.sub(r"\s+", " ", str(v or "").replace("\n", " ")).strip()


# Short forms some tables use, mapped to a fuller name (subset-subsumption then
# folds these into the single canonical spelling). The Table-45 short forms are
# general/health insurers, so the ambiguous ones resolve to their General arm.
ENTITY_ALIASES = {
    "lic": "Life Insurance Corporation of India",
    "acko life": "Acko Life Insurance Ltd.",
    "credit access life": "Credit Access Life Insurance Ltd.",
    "go digit life": "Go Digit Life Insurance Ltd.",
    "maxlife insurance ltd.": "Axis MaxLife Insurance Ltd.",
    "aic": "Agriculture Insurance of India Ltd.",
    "aditya birla": "Aditya Birla Health Insurance Co. Ltd.",
    "bajaj allianz": "Bajaj Allianz General Insurance Co. Ltd.",
    "bharti axa": "Bharti AXA General Insurance Co. Ltd.",
    "go digit": "Go Digit General Insurance Ltd.",
    "hdfc ergo": "HDFC ERGO General Insurance Co. Ltd.",
    "shriram": "Shriram General Insurance Co. Ltd.",
    # Reinsurer variants across Part IV tables.
    "gic": "General Insurance Corporation of India (GIC Re)",
    "gic re. (public)": "General Insurance Corporation of India (GIC Re)",
    "general insurance corporation (gic re)": "General Insurance Corporation of India (GIC Re)",
    "iti": "ITI Reinsurance Ltd.",
    "iti re": "ITI Reinsurance Ltd.",
    "iti (private)": "ITI Reinsurance Ltd.",
}


def _canon_metric(m):
    """Unify equivalent metric names across tables (insurers vs reinsurers)."""
    m = m.replace("(%)", "(Per cent)").replace("(Percent)", "(Per cent)")
    m = re.sub(r"\bIncurred Claim Ratio\b", "Incurred Claims Ratio", m)
    m = re.sub(r"\bNet Premium Earned\b", "Net Earned Premium", m)
    m = re.sub(r"\bNet Incurred Claims\b", "Claims Incurred (Net)", m)
    return re.sub(r"\s{2,}", " ", m).strip()


# Corporate-filler words that don't distinguish one insurer from another.
_FILLER = {"insurance", "co", "company", "ltd", "limited", "india", "the",
           "and", "assurance", "services", "branch", "branches", "of"}


def _sig(name):
    """Distinctive token set of an insurer name (drops corporate fillers)."""
    toks = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    return frozenset(t for t in toks if t not in _FILLER)


def _norm_entity(name):
    """Light canonicalisation so spelling variants of one insurer don't split
    into separate entities across tables (e.g. 'Ltd' vs 'Ltd.', 'Sunlife')."""
    s = re.sub(r"^[\s@#*$^%]+|[\s@#*$^%.]+$", "", _clean(name))  # strip footnote marks
    s = re.sub(r"\bLimited\b", "Ltd", s, flags=re.I)
    s = re.sub(r"\bLtd\.?\s*$", "Ltd.", s)           # normalise trailing Ltd.
    s = re.sub(r"\bSun\s*[Ll]ife\b", "Sun Life", s)
    s = re.sub(r"\bCredit\s*Access\b", "Credit Access", s, flags=re.I)
    s = re.sub(r"\bCompany\b\s*", "", s)             # filler word; safe to drop
    s = s.replace("Limtied", "Limited")              # source typo
    s = re.sub(r"(Lloyd's of India)\s*-\s*", r"\1 - ", s)  # tidy "India- Markel"
    s = re.sub(r"\bLtd\.?\s*$", "Ltd.", s)           # re-normalise tail after edits
    s = re.sub(r"\s{2,}", " ", s).strip()
    return ENTITY_ALIASES.get(s.lower(), s)


def _to_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    s = str(v).strip()
    if s.startswith("(") and s.endswith(")"):
        return None  # parenthetical growth-% annotation, not a value
    m = NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return round(float(m.group()), 4)
    except ValueError:
        return None


def _norm_unit(u):
    return {"₹ crore": "₹Crore", "crore": "₹Crore", "₹crore": "₹Crore",
            "₹lakh": "₹Lakh", "₹ lakh": "₹Lakh", "amount in ₹lakh": "₹Lakh",
            "lakhs": "Lakhs", "lakh": "Lakh", "nos.": "Nos.", "nos": "Nos.",
            "number": "Nos.", "per cent": "Per cent", "percent": "Per cent",
            "us $": "US $", "us$": "US $"}.get(u.strip().lower(), u.strip())


def _find_unit(grid):
    for r in grid:
        for c in r:
            m = UNIT_RE.search(c)
            if m:
                return _norm_unit(re.sub(r"^(in|amount in)\s+", "", m.group(1).strip(), flags=re.I))
    return ""


def _ffill(row):
    """Forward-fill a header row across merged-cell gaps."""
    out, cur = [], ""
    for c in row:
        t = _clean(c)
        if t:
            cur = t
        out.append(cur)
    return out


def _tidy_metric(name):
    """Title-case an all-caps metric, normalising its trailing unit."""
    s = _clean(name)
    m = re.search(r"\(([^()]*)\)\s*$", s)
    unit = ""
    if m and UNIT_RE.fullmatch(f"({m.group(1)})"):
        unit = _norm_unit(m.group(1))
        s = s[:m.start()].strip()
    if s.isupper():
        s = s.title()
    return f"{s} ({unit})" if unit else s


# Aggregate LOB labels to drop (contextless sums of the segments above them).
SKIP_LOBS = {"total", "all segments", "total segments", "grand total", "all", "total insurance"}

# Segment names used to tell which header level is the Line of Business (vs the
# metric) in a matrix, since the two can be nested in either order.
SEGMENT_VOCAB = {"fire", "marine", "marine cargo", "marine hull", "motor", "motor od",
                 "motor tp", "health", "life", "engineering", "aviation", "liability",
                 "personal accident", "pa", "crop", "credit", "travel", "misc",
                 "miscellaneous", "others", "health + pa + travel"}


def _seg_score(level, year_cols):
    """Fraction of a header level's values that look like Line-of-Business names."""
    vals = [_norm_lob(level[i]) for i in year_cols if i < len(level) and level[i]]
    return sum(1 for v in vals if v.lower() in SEGMENT_VOCAB) / len(vals) if vals else 0


def _norm_lob(s):
    s = _clean(s)
    s = re.sub(r"\s+insurance\s*$", "", s, flags=re.I)
    s = re.sub(r"^other segments?$", "Others", s, flags=re.I)
    s = re.sub(r"\bPA\b\s*\+?\s*", "PA + ", s)            # tidy "Health + PA+ TRAVEL"
    s = re.sub(r"\bTRAVEL\b", "Travel", s)
    s = re.sub(r"\s{2,}", " ", s).replace("+ +", "+").strip()
    return s.title() if s.isupper() else s


def _metric_from_title(title, unit):
    t = _clean(title)
    t = re.sub(r"^(table|statement)\s*\d+\s*[:.\-]?\s*", "", t, flags=re.I)
    t = re.sub(r"\bsegment[- ]?wise\s+", "", t, flags=re.I)  # redundant once LOB is a column
    # Drop the "... of <sector> (re)insurers" tail that's redundant once entity
    # and line-of-business are columns.
    t = re.sub(r"\bof\s+(life|general|health|non-life|general and health|"
               r"life and general)\s+(re)?insurers?\b", "", t, flags=re.I)
    t = re.sub(r"\bof\s+(re)?insurers?\b", "", t, flags=re.I)
    t = re.sub(r"\s*[-–]\s*insurer[- ]?wise.*$", "", t, flags=re.I)  # drop "- Insurer-wise" tail
    t = re.sub(r"\s{2,}", " ", t).strip(" -")
    t = t.title() if t.isupper() else t
    if unit and unit.lower() not in {"", "none", "nan"}:
        t = f"{t} ({unit})"
    return t


def convert_matrix(ws):
    """Parse an LOB matrix: 2-level (LOB>year) or 3-level (LOB>metric>year)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    if not grid:
        return []
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1:
        return []
    seg = _ffill(grid[yr - 1])                       # level directly above years
    up = _ffill(grid[yr - 2]) if yr >= 2 else [""] * len(seg)
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}

    ent_col = None
    for hr in (grid[yr - 1], grid[yr - 2] if yr >= 2 else []):
        ent_col = next((i for i, c in enumerate(hr) if c.lower() in ENTITY_AXIS), None)
        if ent_col is not None:
            break
    if ent_col is None:
        ent_col = min(years) - 1

    # 3-level if the row above the segment row carries >1 distinct grouping value.
    three = len({up[i] for i in years}) > 1
    base_metric = _metric_from_title(grid[0][0] if grid[0] else "", _find_unit(grid[:3]))
    # The two header levels can be nested either way (LOB>metric as in Table 44,
    # or metric>LOB as in Table 83). Pick the LOB level by segment-vocabulary.
    if three:
        lob_level, met_level = (seg, up) if _seg_score(seg, years) >= _seg_score(up, years) else (up, seg)

    out = []
    for r in raw[yr + 1:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci, fy in years.items():
            if ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            lob = _norm_lob(lob_level[ci]) if three else _norm_lob(seg[ci])
            metric = _tidy_metric(met_level[ci]) if three else base_metric
            if not lob or lob.lower() in SKIP_LOBS:
                continue
            out.append({
                "dimension": DIMENSION, "entity": entity, "metric": metric,
                "value": value, "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": DEFAULT_CLASS,
            })
    return out


def convert_table(ws, lob):
    rows = [[_clean(c) for c in r] for r in ws.iter_rows(values_only=True)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    raw = list(ws.iter_rows(values_only=True))  # untouched values for numbers

    title = rows[0][0] if rows[0] else ""
    unit = ""
    for r in rows[:5]:
        for c in r:
            m = UNIT_RE.search(c)
            if m:
                unit = m.group(1).replace("In ", "").replace("in ", "").strip()
                break
        if unit:
            break
    # Normalise common unit spellings.
    unit = {"₹ crore": "₹Crore", "crore": "₹Crore", "lakhs": "Lakhs",
            "nos.": "Nos.", "nos": "Nos.", "number": "Nos.", "per cent": "Per cent",
            "us $": "US $", "us$": "US $"}.get(unit.lower(), unit)

    # Header row: the first row with >=2 year cells; entity column = the label
    # column (the 'Insurer'/'Reinsurers' header, else the col left of the years).
    hdr_idx = next((i for i, r in enumerate(rows[:8])
                    if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if hdr_idx is None:
        return []
    header = rows[hdr_idx]
    year_cols = {i: c for i, c in enumerate(header) if YEAR_RE.match(c)}
    ent_col = next((i for i, c in enumerate(header) if c.lower() in ENTITY_AXIS), None)
    if ent_col is None:
        first_year = min(year_cols)
        ent_col = first_year - 1 if first_year > 0 else 0

    metric = _metric_from_title(title, unit)
    out = []
    for r in raw[hdr_idx + 1:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci, fy in year_cols.items():
            if ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": DIMENSION,
                "entity": entity,
                "metric": metric,
                "value": value,
                "financial_year": fy,
                "quarter": QUARTER,
                "line_of_business": lob,
                "class_of_business": DEFAULT_CLASS,
            })
    return out


def _tidy_class(c):
    c = _clean(c)
    low = c.lower()
    if "total" in low:                              # any total / grand-total variant
        return "Total"
    if "irctc" in low:
        return "IRCTC Scheme"
    if "pmjdy" in low or "jan dhan" in low:
        return "PMJDY"
    if "pmsby" in low or "suraksha bima" in low:
        return "PMSBY"
    if low.startswith("government sponsored"):
        return "Government Sponsored"
    if low.startswith("group"):
        return "Group (excl. Govt)" if "exclud" in low else "Group"
    if "family" in low and "floater" in low and "excluding individual" in low:
        return "Family Floater (excl. Individual)"
    if "individual" in low and "excluding family" in low:
        return "Individual (excl. Family Floater)"
    if "individual" in low and "family floater" in low and "other" not in low:
        return "Individual - Family Floater"
    if "individual" in low and "other" in low:
        return "Individual - Other"
    if low == "individual business":
        return "Individual"
    return c


def _submetric(sub, table_unit):
    s = _clean(sub).replace("Permium", "Premium")
    low = s.lower()
    if "ratio" in low:
        unit = "Per cent"
    elif re.search(r"\([^)]*\)\s*$", s):       # already carries a unit/qualifier
        unit = ""
    elif "polic" in low:
        s, unit = "No. of Policies", "Nos."
    else:
        unit = table_unit
    return f"{s} ({unit})" if unit else s


def convert_transposed(ws, lob, fallback_unit="₹Crore"):
    """Insurers are column groups, metrics are row labels (e.g. Table 45)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1:
        return []
    ent_row = _ffill(grid[yr - 1])
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}
    unit = _find_unit(grid[:yr]) or fallback_unit
    out = []
    for r in raw[yr + 1:]:
        metric = _clean(r[0]).rstrip("*# ").strip() if r else ""
        if not metric or "=" in metric or metric.lower().startswith("note"):
            continue
        name = f"{metric} ({unit})" if unit else metric
        for ci, fy in years.items():
            entity = _norm_entity(ent_row[ci]) if ci < len(ent_row) else ""
            if not entity or SKIP_ENTITIES.match(entity) or ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": DIMENSION, "entity": entity, "metric": name,
                "value": value, "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": DEFAULT_CLASS,
            })
    return out


def convert_class_matrix(ws, lob, fallback_unit="₹Lakh"):
    """year > class > sub-metric column header, rows = insurers (Tables 58-65)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr + 2 >= len(grid):
        return []
    ylevel = _ffill(grid[yr])
    clevel = _ffill(grid[yr + 1])
    slevel = grid[yr + 2]                       # sub-metric, one per column
    unit = _find_unit(grid[:yr + 1]) or fallback_unit
    ent_col = next((i for i, c in enumerate(grid[yr]) if c.lower() in ENTITY_AXIS), 1)

    out = []
    for r in raw[yr + 3:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci in range(len(r)):
            if not (YEAR_RE.match(ylevel[ci] if ci < len(ylevel) else "")):
                continue
            cls = _tidy_class(clevel[ci] if ci < len(clevel) else "")
            sub = slevel[ci] if ci < len(slevel) else ""
            if not cls or "total" in cls.lower() or not _clean(sub):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": DIMENSION, "entity": entity,
                "metric": _submetric(sub, unit), "value": value,
                "financial_year": ylevel[ci], "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": cls,
            })
    return out


def _open(parts_dir, part):
    path = os.path.join(parts_dir, f"{part}.xlsx")
    if not os.path.exists(path):
        print(f"   [!] {path} missing — skipped")
        return None
    return openpyxl.load_workbook(path, read_only=True, data_only=True)


def main():
    ap = argparse.ArgumentParser(description="Convert Handbook Part tables (Phase 1) to IRIS rows.")
    ap.add_argument("parts_dir", help="Folder containing 'Part I.xlsx' ... 'Part V.xlsx'")
    ap.add_argument("--out", default=os.path.join("knowledge_base", "raw_submissions"))
    ap.add_argument("--name", default="handbook_2024-25_parts.xlsx")
    args = ap.parse_args()

    all_rows = []
    for part, sheet, lob in PHASE1:
        path = os.path.join(args.parts_dir, f"{part}.xlsx")
        if not os.path.exists(path):
            print(f"   [!] {path} missing — skipped")
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        match = next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_table(wb[match], lob)
        metric = rows[0]["metric"] if rows else "?"
        print(f"   {part} t{sheet:>3} [{lob:11}] -> {metric}: {len(rows)} rows")
        all_rows.extend(rows)

    for part, sheet in PHASE2:
        path = os.path.join(args.parts_dir, f"{part}.xlsx")
        if not os.path.exists(path):
            print(f"   [!] {path} missing — skipped")
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        match = next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_matrix(wb[match])
        lobs = sorted({r["line_of_business"] for r in rows})
        mets = sorted({r["metric"] for r in rows})
        print(f"   {part} t{sheet:>3} [matrix]      -> {len(rows)} rows | "
              f"LOBs={lobs} | metrics={len(mets)}")
        all_rows.extend(rows)

    for part, sheet, lob in PHASE2B_TRANSPOSED:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_transposed(wb[match], lob)
        print(f"   {part} t{sheet:>3} [transposed]  -> {len(rows)} rows | "
              f"metrics={len(set(r['metric'] for r in rows))}")
        all_rows.extend(rows)

    for part, sheet, lob in PHASE2B_CLASS:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_class_matrix(wb[match], lob)
        cls = sorted({r["class_of_business"] for r in rows})
        print(f"   {part} t{sheet:>3} [{lob:11}] class -> {len(rows)} rows | classes={cls}")
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("No rows produced.")

    # Canonicalise insurer names into a standard list: a short/variant name folds
    # into the fullest name whose words are a superset (e.g. "Star Health" ->
    # "Star Health & Allied Insurance Co. Ltd.", "Tata AIG" -> "Tata AIG General
    # Insurance Co. Ltd."). Ambiguous prefixes (e.g. a bare "Reliance" that could
    # be General/Health/Life) are left untouched.
    names = sorted({r["entity"] for r in all_rows})
    sig_names = {}
    for n in names:
        sig_names.setdefault(_sig(n), set()).add(n)
    sigs = list(sig_names)
    resolve = {}
    for n in names:
        s = _sig(n)
        supers = [o for o in sigs if s < o]
        minimal = [o for o in supers if not any(p < o for p in supers if p != o)]
        if len(minimal) == 1:                      # unique fuller name → fold in
            resolve[n] = max(sig_names[minimal[0]], key=len)
        else:                                      # keep within its own variant group
            resolve[n] = max(sig_names[s], key=len)
    merged = sum(1 for n in names if resolve[n] != n)
    for r in all_rows:
        r["entity"] = resolve[r["entity"]]
        r["metric"] = _canon_metric(r["metric"])
    print(f"\n[i] Canonicalised insurer names: folded {merged} variants into the standard list.")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, args.name)
    df = pd.DataFrame(all_rows, columns=[
        "dimension", "entity", "metric", "value", "financial_year",
        "quarter", "line_of_business", "class_of_business",
    ])
    df.to_excel(out_path, index=False)
    print(f"\n[+] Wrote {len(df)} rows -> {out_path}")
    print(f"    insurers: {df['entity'].nunique()} | metrics: {df['metric'].nunique()} | "
          f"years: {df['financial_year'].min()}..{df['financial_year'].max()}")


if __name__ == "__main__":
    main()
