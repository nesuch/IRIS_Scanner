#!/usr/bin/env python3
"""
Surgical re-ingestion of Handbook Table 78/80 — "State-wise Details on Number of
Network Providers" (Part III) — which lost the AGREEMENT-TYPE dimension. The
table has two identical Category x Region blocks: "Network Providers with whom
Insurers DIRECTLY have an agreement" vs "under a TRIPARTITE agreement with TPAs".
The old flatten collapsed them onto one key (e.g. 48 vs 151 for the same cell).

    dimension=State, insurer(entity)=State/UT, line_of_business=Network Providers,
    class_of_business=Insurer, metric="{agreement} - {category} - {region} (Nos.)",
    financial_year=2024-25 (snapshot), quarter=Annual
"""
import argparse
import os
import re
import sys

import pandas as pd

LOB = "Network Providers"
YEAR = "2024-25"


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _agree(s):
    t = _clean(s).lower()
    if "directly" in t:
        return "Direct agreement"
    if "tripartite" in t:
        return "Tripartite (TPA) agreement"
    return _clean(s)


def _num(x):
    if pd.isna(x) or str(x).strip() in ("-", ""):
        return None
    v = pd.to_numeric(str(x).replace(",", "").strip("()"), errors="coerce")
    return None if pd.isna(v) else float(v)


def parse(path):
    df = pd.read_excel(path, sheet_name="78", header=None)
    agree = df.iloc[1].ffill()
    cat = df.iloc[2].ffill()
    region = df.iloc[3]

    cols = [c for c in range(2, df.shape[1])
            if pd.notna(agree[c]) and pd.notna(cat[c]) and pd.notna(region[c])]

    # The insurer is a MERGED cell whose value lands on an inconsistent row within
    # each 37-state block, so ffill mis-attributes rows. Instead segment into blocks
    # (each starts at "Andhra Pradesh") and assign the block's lone label to all rows.
    data_rows = [r for r in range(4, df.shape[0]) if pd.notna(df.iloc[r, 1])]
    blocks, cur = [], []
    for r in data_rows:
        if _clean(df.iloc[r, 1]).lower().startswith("andhra pradesh") and cur:
            blocks.append(cur); cur = []
        cur.append(r)
    if cur:
        blocks.append(cur)
    row_insurer = {}
    for blk in blocks:
        labels = [_clean(df.iloc[r, 0]) for r in blk if pd.notna(df.iloc[r, 0])]
        ins = labels[0] if labels else None
        for r in blk:
            row_insurer[r] = ins

    out = []
    for r in data_rows:
        state = df.iloc[r, 1]
        ins = row_insurer.get(r)
        if pd.isna(state) or not ins:
            continue
        state = _clean(state)
        if state.lower() in ("total", "grand total"):
            state = "All India (Total)"
        for c in cols:
            v = _num(df.iloc[r, c])
            if v is None:
                continue
            metric = (f"{_agree(agree[c])} - {_clean(cat[c])} - "
                      f"{_clean(region[c])} (Nos.)")
            out.append({
                "dimension": "State", "insurer": state,
                "financial_year": YEAR, "quarter": "Annual",
                "metric": metric, "value": v,
                "line_of_business": LOB, "class_of_business": _clean(ins),
                "source_file": "handbook_2024-25_parts.xlsx"})
    return pd.DataFrame(out)


def validate(new):
    print(f"parsed {len(new):,} rows | states {new.insurer.nunique()} | "
          f"insurers {new.class_of_business.nunique()} | metrics {new.metric.nunique()}")
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    print("  duplicate keys remaining:", int((new.groupby(KEY).size() > 1).sum()), "(want 0)")
    print("  agreement types:", sorted(set(m.split(" - ")[0] for m in new.metric.unique())))


def apply(db, new):
    import sqlite3
    con = sqlite3.connect(db); cur = con.cursor()
    old = cur.execute("SELECT COUNT(*) FROM financial_metrics WHERE dimension='State' "
                      "AND line_of_business=?", (LOB,)).fetchone()[0]
    cur.execute("DELETE FROM financial_metrics WHERE dimension='State' AND line_of_business=?", (LOB,))
    cols = ["insurer", "financial_year", "quarter", "metric", "value",
            "line_of_business", "class_of_business", "dimension", "source_file"]
    cur.executemany(
        f"INSERT INTO financial_metrics ({','.join(cols)}) VALUES ({','.join('?'*len(cols))})",
        [tuple(r[c] for c in cols) for _, r in new.iterrows()])
    con.commit(); con.close()
    print(f"applied to {db}: removed {old} old, inserted {len(new)} new")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--handbook", default=os.path.expanduser(
        "~/Desktop/Publication of Handbook 2024-25 on IRDAI Website"))
    ap.add_argument("--csv", default="/tmp/table78_reingested.csv")
    ap.add_argument("--apply", metavar="DB")
    args = ap.parse_args()
    new = parse(os.path.join(args.handbook, "Part III.xlsx"))
    validate(new)
    new.to_csv(args.csv, index=False)
    print(f"=> {args.csv}")
    if args.apply:
        apply(args.apply, new)


if __name__ == "__main__":
    sys.exit(main())
