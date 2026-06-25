#!/usr/bin/env python3
"""
Re-source Handbook Table 82 — "Net Retention of Non-Life Insurers (as a percent
of Gross Direct Premium)", Part IV. This restores the Net Retention data that was
previously POC-sourced (and removed with the unified_database.csv cleanup).

Structure: Segment (Aviation/Engineering/Fire/Marine/Motor/Misc/Industry) x Year,
one Net Retention % per cell.

    dimension=Industry, insurer="General Insurance Sector",
    line_of_business=segment ("Industry" row -> "All Lines"),
    class_of_business="All Classes", metric="Net Retention (Per cent)"
"""
import argparse
import os
import re
import sys

import pandas as pd

LOB_ENTITY = "General Insurance Sector"
METRIC = "Net Retention (Per cent)"


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _num(x):
    if pd.isna(x) or str(x).strip() in ("-", ""):
        return None
    v = pd.to_numeric(str(x).replace(",", "").strip("()"), errors="coerce")
    return None if pd.isna(v) else float(v)


def parse(path):
    df = pd.read_excel(path, sheet_name="82", header=None)
    fy_re = re.compile(r"^\d{4}-\d{2}$")
    years = {c: _clean(df.iloc[2, c]) for c in range(1, df.shape[1])
             if pd.notna(df.iloc[2, c]) and fy_re.match(_clean(df.iloc[2, c]))}
    out = []
    for r in range(3, df.shape[0]):
        seg = df.iloc[r, 0]
        if pd.isna(seg):
            continue
        seg = _clean(seg)
        lob = "All Lines" if seg.lower() == "industry" else seg
        for c, fy in years.items():
            v = _num(df.iloc[r, c])
            if v is None:
                continue
            out.append({
                "dimension": "Industry", "insurer": LOB_ENTITY,
                "financial_year": fy, "quarter": "Annual",
                "metric": METRIC, "value": v,
                "line_of_business": lob, "class_of_business": "All Classes",
                "source_file": "handbook_2024-25_parts.xlsx"})
    return pd.DataFrame(out)


def validate(new):
    print(f"parsed {len(new):,} rows | segments {new.line_of_business.nunique()} | "
          f"years {new.financial_year.nunique()}")
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    print("  duplicate keys remaining:", int((new.groupby(KEY).size() > 1).sum()), "(want 0)")
    print("  value range:", round(new.value.min(), 1), "-", round(new.value.max(), 1), "(% retention)")
    print("  segments:", sorted(new.line_of_business.unique()))


def apply(db, new):
    import sqlite3
    con = sqlite3.connect(db); cur = con.cursor()
    # remove any prior Net Retention rows in Industry, then insert fresh
    cur.execute("DELETE FROM financial_metrics WHERE dimension='Industry' AND metric=?", (METRIC,))
    cols = ["insurer", "financial_year", "quarter", "metric", "value",
            "line_of_business", "class_of_business", "dimension", "source_file"]
    cur.executemany(
        f"INSERT INTO financial_metrics ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        [tuple(r[c] for c in cols) for _, r in new.iterrows()])
    con.commit(); con.close()
    print(f"applied to {db}: inserted {len(new)} Net Retention rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handbook", default=os.path.expanduser(
        "~/Desktop/Publication of Handbook 2024-25 on IRDAI Website"))
    ap.add_argument("--csv", default="/tmp/table82_reingested.csv")
    ap.add_argument("--apply", metavar="DB")
    args = ap.parse_args()
    new = parse(os.path.join(args.handbook, "Part IV.xlsx"))
    validate(new)
    new.to_csv(args.csv, index=False)
    print(f"=> {args.csv}")
    if args.apply:
        apply(args.apply, new)


if __name__ == "__main__":
    sys.exit(main())
