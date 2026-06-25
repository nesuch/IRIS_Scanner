#!/usr/bin/env python3
"""
Surgical re-ingestion of Handbook Table 66 — "Details of Claims Development and
Aging (Health Insurance)" — which lost two dimensions in the original tidy
transform (claim-TYPE: Cashless/Reimbursement/Benefit/Total, and MEASURE:
Number vs Amount), collapsing many distinct cells onto one key.

This re-parses the original wide matrix preserving every dimension and maps it
back into the financial_metrics schema as:
    dimension          = Industry
    insurer (entity)   = General, Health & RE (Industry)
    line_of_business   = Claims Development & Aging
    class_of_business  = channel  (For Claims Handled through TPAs / directly / Grand Total)
    metric             = "{status} - {type} ({unit})"   unit = Nos. | ₹Lakh
    financial_year, quarter=Annual, value

By default it only PARSES + VALIDATES and writes the candidate rows to a CSV.
With --apply <db> it surgically DELETEs the old section rows and inserts the new
ones (use a throwaway copy first).

  python tools/reingest_table66.py [--handbook DIR] [--csv OUT] [--apply DB]
"""
import argparse
import os
import re
import sqlite3
import sys

import pandas as pd

ENTITY = "General, Health & RE (Industry)"
LOB = "Claims Development & Aging"
SHEET = "66"
PART = "Part III.xlsx"

# Rows (0-indexed) in col 0 that are section headers / notes, not data.
SKIP_LABELS = {
    "Details of Claims Development", "Ageing of Paid claims*",
    "Ageing of repudiated claims**", "Ageing of outstanding claims**", "Note:",
}


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def parse(handbook_dir):
    path = os.path.join(handbook_dir, PART)
    df = pd.read_excel(path, sheet_name=SHEET, header=None)

    year = df.iloc[2].ffill()
    channel = df.iloc[3].ffill()
    ctype = df.iloc[4].ffill()
    measure = df.iloc[5]            # per-column Number / Amount (not ffilled)

    # data columns: have a real year, channel, type and a Number/Amount measure
    data_cols = []
    for c in range(1, df.shape[1]):
        y, ch, ty, me = year[c], channel[c], ctype[c], measure[c]
        if pd.isna(y) or str(y).startswith("Particular"):
            continue
        if pd.isna(me) or _clean(me) not in ("Number", "Amount"):
            continue
        if pd.isna(ch) or pd.isna(ty):
            continue
        data_cols.append(c)

    records = []
    for r in range(6, df.shape[0]):
        label = df.iloc[r, 0]
        if pd.isna(label):
            continue
        status = _clean(label)
        if status in SKIP_LABELS or status.startswith("*"):
            continue
        for c in data_cols:
            val = df.iloc[r, c]
            val = pd.to_numeric(val, errors="coerce")
            if pd.isna(val):
                continue
            me = _clean(measure[c])
            unit = "Nos." if me == "Number" else "₹Lakh"
            metric = f"{status} - {_clean(ctype[c])} ({unit})"
            records.append({
                "dimension": "Industry",
                "insurer": ENTITY,
                "financial_year": _clean(year[c]),
                "quarter": "Annual",
                "metric": metric,
                "value": float(val),
                "line_of_business": LOB,
                "class_of_business": _clean(channel[c]),
                "source_file": "handbook_2024-25_parts.xlsx",
            })
    return pd.DataFrame(records)


def validate(new):
    print(f"parsed {len(new):,} rows")
    print("  years:", sorted(new.financial_year.unique()))
    print("  channels:", sorted(new.class_of_business.unique()))
    print("  distinct metrics:", new.metric.nunique())
    # uniqueness — the whole point: no key should have >1 value now
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    dup = new.groupby(KEY).size()
    print("  duplicate keys remaining:", int((dup > 1).sum()), "(want 0)")
    # Internal consistency: for each (status, year, channel, measure) the sum of
    # all non-Total claim-types must equal the Total. Robust to type-name variants.
    new = new.copy()
    new["_status"] = new.metric.str.replace(r" - .+ \((Nos\.|₹Lakh)\)$", "", regex=True)
    new["_type"] = new.metric.str.extract(r" - (.+) \((?:Nos\.|₹Lakh)\)$")[0]
    new["_unit"] = new.metric.str.extract(r"\((Nos\.|₹Lakh)\)$")[0]
    # The table carries TWO parallel type schemes; sum only the mutually-exclusive
    # one (Only Cashless / Only Reimbursement / Both / Benefit Based) vs Total.
    ADDITIVE = {"Only Cashless", "Only Reimbursement",
                "Both Cashless and Reimbursement", "Benefit Based"}
    checked = ok = 0
    for (st, yr, ch, un), g in new.groupby(["_status", "financial_year",
                                            "class_of_business", "_unit"]):
        tot = g[g._type == "Total"].value
        parts = g[g._type.isin(ADDITIVE)].value.sum()
        if len(tot) == 1 and tot.iloc[0] not in (0, None) and g._type.isin(ADDITIVE).any():
            checked += 1
            if abs(parts - tot.iloc[0]) / abs(tot.iloc[0]) < 0.02:
                ok += 1
    print(f"  sum-of-types == Total: {ok}/{checked} groups consistent "
          f"({100*ok/checked:.1f}%)" if checked else "  no Total rows to check")


def apply(db, new):
    con = sqlite3.connect(db)
    cur = con.cursor()
    old = cur.execute(
        "SELECT COUNT(*) FROM financial_metrics WHERE dimension='Industry' "
        "AND line_of_business=?", (LOB,)).fetchone()[0]
    cur.execute("DELETE FROM financial_metrics WHERE dimension='Industry' "
                "AND line_of_business=?", (LOB,))
    cols = ["insurer", "financial_year", "quarter", "metric", "value",
            "line_of_business", "class_of_business", "dimension", "source_file"]
    cur.executemany(
        f"INSERT INTO financial_metrics ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        [tuple(r[c] for c in cols) for _, r in new.iterrows()])
    con.commit()
    con.close()
    print(f"applied to {db}: removed {old} old rows, inserted {len(new)} new rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handbook", default=os.path.expanduser(
        "~/Desktop/Publication of Handbook 2024-25 on IRDAI Website"))
    ap.add_argument("--csv", default="/tmp/table66_reingested.csv")
    ap.add_argument("--apply", metavar="DB", help="surgically replace section in this DB")
    args = ap.parse_args()

    new = parse(args.handbook)
    validate(new)
    new.to_csv(args.csv, index=False)
    print(f"=> candidate rows written to {args.csv}")
    if args.apply:
        apply(args.apply, new)


if __name__ == "__main__":
    sys.exit(main())
