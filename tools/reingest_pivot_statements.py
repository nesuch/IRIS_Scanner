#!/usr/bin/env python3
"""
Ingest the four deep-pivot Handbook tables that the per-insurer parsers skipped
(4, 21, 100, 102) as **statement-style views** under the Financials dimension,
NOT as new Insurer-view rows. This keeps them out of the Insurer metric/class
filters entirely (zero added filter clutter) — each appears only as a new entry
in the Financials "Statement" picker and renders via the existing StatementView
(section -> line item -> value per year). The user explicitly asked for the
"separate dimension, whole statement appears" pattern.

Mapping (all rows: dimension='Financials'):
  Table 4   -> Entity "Life Insurance Sector", statement "Life Segment-wise Premium",
               section = Linked / Non-Linked / Linked + Non-Linked,
               item = "{segment} — {First Year|Renewal|Single|Total} (₹Crore)".
  Table 21  -> Entity = insurer, statement "AUM — Insurer-wise (Investments)",
               section = fund, item = "{investment category} (₹Crore)", year = 2022..2025.
  Table 100 -> Entity = insurer, statement "Life Individual New Business by Channel (2024-25)",
               section = channel, item = "Policies (Nos.)" / "New Business Premium (₹Crore)".
  Table 102 -> Entity = insurer, statement "Life Group New Business by Channel (2024-25)",
               section = channel, item = "Schemes (Nos.)" / "New Business Premium (₹Crore)" /
               "Lives Covered (Nos.)".

Safe: writes a CSV preview by default; only touches a DB with --apply (run it on a
throwaway copy first). Re-run canonicalize + categorize after applying.
"""
import argparse
import os
import re
import sys

import pandas as pd

SRC = "handbook_2024-25_parts.xlsx"
DIM = "Financials"


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _year4(x):
    """Normalise a year cell that may arrive as '2022', 2023.0 or '2024-25'."""
    s = _clean(x)
    m = re.match(r"(\d{4})(?:-\d{2})?(?:\.0)?$", s)
    return m.group(0) if (m and "-" in s) else (m.group(1) if m else None)


def _num(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    if s in ("", "-", "–", "—", "NA", "N.A.", "nan"):
        return None
    v = pd.to_numeric(s.replace(",", "").strip("()"), errors="coerce")
    return None if pd.isna(v) else float(v)


# Canonical life-insurer display names — the spellings the EXISTING Financials
# statements use, so each insurer's statements co-locate under one pickable entity.
# Maps every raw spelling seen in Tables 21 (short) and 100/102 (long) -> canonical.
# Hand-verified: auto-fuzzy-matching is unsafe (it confuses an insurer's life arm
# with its general/health arm, e.g. "Future Generali Life" vs the general entity).
_LIFE_CANON = {
    "Acko Life Insurance Ltd.": ["Acko Life"],
    "Aditya Birla Sun Life Insurance Ltd.": ["Aditya Birla Sun Life", "Aditya Birla Sunlife Insurance Company Ltd."],
    "Ageas Federal Life Insurance Ltd.": ["Ageas Federal Life", "Ageas Federal Life Insurance Company Ltd."],
    "Aviva Life Insurance India Ltd.": ["Aviva Life", "Aviva Life Insurance Company India Ltd."],
    "Axis Max Life Insurance Co. Ltd.": ["Max Life", "Axis MaxLife Insurance Company Ltd.", "MaxLife Insurance Company Ltd."],
    "Bajaj Allianz Life Insurance Co Ltd.": ["Bajaj Allianz Life", "Bajaj Allianz Life Insurance Company Ltd."],
    "Bandhan Life Insurance Ltd.": ["Bandhan Life"],
    "Bharti-AXA Life Insurance Co Ltd.": ["Bharti AXA Life", "Bharti AXA Life Insurance Company Ltd."],
    "Canara HSBC OBC Life Insurance Ltd.": ["Canara HSBC OBC Life", "Canara HSBC Life Insurance Company Ltd."],
    "Credit Access Life Insurance Ltd.": ["Credit Access Life"],
    "Edelweiss Life Insurance Ltd.": ["Edelweiss Tokio Life", "Edelweiss Life Insurance Company Ltd.", "Edelweiss Tokio Life Insurance Company Ltd."],
    "Exide Life Insurance Ltd.": ["Exide Life"],
    "Future Generali India Life Insurance Ltd.": ["Future Generali Life", "Future Generali India Life Insurance Company Ltd."],
    "Go Digit Life Insurance Ltd.": ["Go Digit Life"],
    "HDFC Life Insurance Ltd.": ["HDFC Life", "HDFC Life Insurance Company Ltd."],
    "ICICI Prudential Life Insurance Ltd.": ["ICICI Prudential Life", "ICICI Prudential Life Insurance Company Ltd."],
    "India First Life Insurance Ltd.": ["IndiaFirst Life", "IndiaFirst Life Insurance Company Ltd."],
    "Kotak Mahindra OM Life Insurance Co. Ltd.": ["Kotak Mahindra Life", "Kotak Mahindra Life Insurance Ltd."],
    "Life Insurance Corporation of India": ["LIC"],
    "PNB MetLife India Insurance Co. Ltd.": ["PNB Metlife", "PNB Metlife India Insurance Company Ltd."],
    "Pramerica Life Insurance Ltd.": ["Pramerica Life", "Pramerica Life Insurance Company Ltd."],
    "Reliance Nippon Life Insurance Ltd.": ["Reliance Nippon Life", "Reliance Nippon Life Insurance Company Ltd."],
    "SBI Life Insurance Ltd.": ["SBI Life", "SBI Life Insurance Company Ltd."],
    "Sahara India Life Insurance Ltd.": ["Sahara India Life", "Sahara India Life Insurance Company Ltd."],
    "Shriram Life Insurance Co. Ltd.": ["Shriram Life", "Shriram Life Insurance Company Ltd."],
    "Star Union Dai-ichi Life Insurance Ltd.": ["Star Union Dai-ichi Life", "Star Union Dai-ichi Life Insurance Company Ltd."],
    "Tata AIA Life Insurance Ltd.": ["Tata AIA Life", "TATA AIA Life Insurance Company Ltd."],
}


def _norm_name(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


_ALIAS = {}
for _c, _vs in _LIFE_CANON.items():
    _ALIAS[_norm_name(_c)] = _c
    for _v in _vs:
        _ALIAS[_norm_name(_v)] = _c


def canon_insurer(raw):
    """Map a raw insurer spelling to its canonical display name (unchanged if
    not a known life insurer, e.g. the 'Life Insurance Sector' aggregate)."""
    return _ALIAS.get(_norm_name(raw), _clean(raw))


def _row(entity, stmt, section, item, fy, val):
    return {"dimension": DIM, "insurer": entity, "financial_year": fy,
            "quarter": "Annual", "metric": item, "value": val,
            "line_of_business": stmt, "class_of_business": section,
            "source_file": SRC}


# ---------------------------------------------------------------- Table 4 -----
T4_STMT = "Life Segment-wise Premium"
T4_ENTITY = "Life Insurance Sector"
T4_SECTIONS = {
    "linked (individual and group)": "Linked",
    "non-linked (individual and group)": "Non-Linked",
    "linked and non-linked (individual and group)": "Linked + Non-Linked",
}
# Year blocks start at col 1 and step by 10. Within a block:
#  +0..3 Non-Par FY/Renewal/Single/Total, +4..7 Par FY/Renewal/Single/Total,
#  +8 Both Grand Total, +9 Percentage.
T4_YEARS = {1: "2021-22", 11: "2022-23", 21: "2023-24", 31: "2024-25"}


def parse_t4(path):
    df = pd.read_excel(path, sheet_name="4", header=None)
    out, section = [], None
    for r in range(5, df.shape[0]):
        lab = df.iloc[r, 0]
        if pd.isna(lab):
            continue
        key = _clean(lab).lower()
        if key in T4_SECTIONS:
            section = T4_SECTIONS[key]
            continue
        if section is None:
            continue
        seg = _clean(lab)
        if seg.lower() in ("note:",) or seg.lower().startswith("note"):
            continue
        # subtotal rows -> a single "All Segments" line
        if re.search(r"\b(total)\b", seg, re.I):
            seg = "All Segments"
        for c0, fy in T4_YEARS.items():
            fy_block = {
                "First Year": (_num(df.iloc[r, c0]), _num(df.iloc[r, c0 + 4])),
                "Renewal":    (_num(df.iloc[r, c0 + 1]), _num(df.iloc[r, c0 + 5])),
                "Single":     (_num(df.iloc[r, c0 + 2]), _num(df.iloc[r, c0 + 6])),
            }
            for ptype, (npar, par) in fy_block.items():
                if npar is None and par is None:
                    continue
                val = (npar or 0) + (par or 0)
                out.append(_row(T4_ENTITY, T4_STMT, section,
                                f"{seg} — {ptype} (₹Crore)", fy, val))
            tot = _num(df.iloc[r, c0 + 8])      # Both Grand Total
            if tot is not None:
                out.append(_row(T4_ENTITY, T4_STMT, section,
                                f"{seg} — Total Premium (₹Crore)", fy, tot))
    return pd.DataFrame(out)


# --------------------------------------------------------------- Table 21 -----
T21_STMT = "AUM — Insurer-wise (Investments)"


def parse_t21(path):
    df = pd.read_excel(path, sheet_name="21", header=None)
    # Build column map: col -> (fund, category, year) by forward-filling the
    # fund row (R3, anchored at fund starts) and category row (R4) across cols.
    funds, cats = {}, {}
    cur = None
    for c in range(2, df.shape[1]):
        v = df.iloc[3, c]
        if pd.notna(v) and _clean(v):
            cur = _clean(v)
        funds[c] = cur
    cur = None
    for c in range(2, df.shape[1]):
        v = df.iloc[4, c]
        if pd.notna(v) and _clean(v):
            cur = _clean(v)
        cats[c] = cur
    years = {c: _year4(df.iloc[5, c]) for c in range(2, df.shape[1])
             if _year4(df.iloc[5, c])}
    out = []
    for r in range(6, df.shape[0]):
        sno, name = df.iloc[r, 0], df.iloc[r, 1]
        if pd.isna(sno) or pd.isna(name):     # section sub-headers (Public/Private Sector)
            continue
        entity = _clean(name)
        if not entity or entity.lower() in ("insurer",):
            continue
        entity = canon_insurer(entity)
        for c, yr in years.items():
            val = _num(df.iloc[r, c])
            if val is None:
                continue
            fund = funds.get(c) or "General"
            cat = cats.get(c) or "Total"
            out.append(_row(entity, T21_STMT, fund, f"{cat} (₹Crore)", yr, val))
    return pd.DataFrame(out)


# ----------------------------------------------------------- Tables 100/102 ---
def _channel_map(df, n_cols):
    """col -> (channel, sub-metric). Channel header on R1 (anchored at channel
    start), Bank/Others split on R2, sub-metric (Policies/Premium/...) on R3."""
    out = {}
    chan = sub = None
    for c in range(2, n_cols):
        h1 = _clean(df.iloc[1, c]) if pd.notna(df.iloc[1, c]) else ""
        if h1:                                 # a new top channel -> reset sub-split
            chan, sub = h1, None
        h2 = _clean(df.iloc[2, c]) if pd.notna(df.iloc[2, c]) else ""
        if h2:                                 # Corporate Agents -> Banks / Others
            sub = h2
        item = _clean(df.iloc[3, c]) if pd.notna(df.iloc[3, c]) else ""
        if not chan:
            continue
        label = f"{chan} ({sub})" if sub else chan
        out[c] = (label, item)
    return out


def _parse_channel(path, sheet, stmt, fy, item_rename):
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    cmap = _channel_map(df, df.shape[1])
    out = []
    for r in range(4, df.shape[0]):
        sno, name = df.iloc[r, 0], df.iloc[r, 1]
        if pd.isna(name):
            continue
        entity = _clean(name)
        if pd.isna(sno) or not entity:         # Public/Private Sector headers
            continue
        if entity.lower() in ("insurer", "grand total", "total"):
            continue
        entity = canon_insurer(entity)
        for c, (chan, sub) in cmap.items():
            val = _num(df.iloc[r, c])
            if val is None:
                continue
            item = item_rename.get(sub, sub) or "Value"
            out.append(_row(entity, stmt, chan, item, fy, val))
    return pd.DataFrame(out)


def parse_t100(path):
    return _parse_channel(
        path, "100", "Life Individual New Business by Channel (2024-25)", "2024-25",
        {"Policies": "Policies (Nos.)", "Premium (₹Crore)": "New Business Premium (₹Crore)"})


def parse_t102(path):
    return _parse_channel(
        path, "102", "Life Group New Business by Channel (2024-25)", "2024-25",
        {"Schemes": "Schemes (Nos.)", "Premium (₹Crore)": "New Business Premium (₹Crore)",
         "Lives covered": "Lives Covered (Nos.)"})


# ---------------------------------------------------------------- driver ------
TABLES = [
    ("Part I.xlsx", parse_t4, T4_STMT),
    ("Part I.xlsx", parse_t21, T21_STMT),
    ("Part V.xlsx", parse_t100, "Life Individual New Business by Channel (2024-25)"),
    ("Part V.xlsx", parse_t102, "Life Group New Business by Channel (2024-25)"),
]


_CANON_SET = set(_LIFE_CANON) | {T4_ENTITY}


def validate(name, new):
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    dups = int((new.groupby(KEY).size() > 1).sum()) if len(new) else 0
    # Any insurer not resolved to a canonical name = an alias we missed (would
    # fragment that insurer's statements under a separate, hard-to-find entity).
    unmapped = sorted(e for e in new.insurer.unique() if e not in _CANON_SET)
    print(f"  [{name}] rows={len(new):,} entities={new.insurer.nunique()} "
          f"sections={new.class_of_business.nunique()} items={new.metric.nunique()} "
          f"years={sorted(new.financial_year.unique())} dups={dups}")
    if unmapped:
        print(f"     !! UNMAPPED insurer spellings ({len(unmapped)}): {unmapped}")
    return dups, len(unmapped)


def apply(db, frames):
    import sqlite3
    con = sqlite3.connect(db); cur = con.cursor()
    cols = ["insurer", "financial_year", "quarter", "metric", "value",
            "line_of_business", "class_of_business", "dimension", "source_file"]
    total = 0
    for new in frames:
        stmt = new["line_of_business"].iloc[0]
        cur.execute("DELETE FROM financial_metrics WHERE dimension=? AND line_of_business=?",
                    (DIM, stmt))
        cur.executemany(
            f"INSERT INTO financial_metrics ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
            [tuple(r[c] for c in cols) for _, r in new.iterrows()])
        total += len(new)
    con.commit(); con.close()
    print(f"applied to {db}: inserted {total} statement rows across {len(frames)} statements")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handbook", default=os.path.expanduser(
        "~/Desktop/Publication of Handbook 2024-25 on IRDAI Website"))
    ap.add_argument("--csv", default="/tmp/pivot_statements_reingested.csv")
    ap.add_argument("--apply", metavar="DB")
    args = ap.parse_args()

    frames, bad, unmapped = [], 0, 0
    for fname, fn, name in TABLES:
        new = fn(os.path.join(args.handbook, fname))
        d, u = validate(name, new)
        bad += d; unmapped += u
        frames.append(new)
    allrows = pd.concat(frames, ignore_index=True)
    allrows.to_csv(args.csv, index=False)
    print(f"=> total {len(allrows):,} rows | {args.csv} | dup keys: {bad} | unmapped insurers: {unmapped} (want 0/0)")
    if args.apply:
        if bad or unmapped:
            print("ABORT: dup keys or unmapped insurers present; not applying."); return 1
        apply(args.apply, frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
