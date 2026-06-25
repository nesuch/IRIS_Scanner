#!/usr/bin/env python3
"""
Surgical re-ingestion of Handbook AUM tables (Investments) — Table 46
(General/Health/RE) and Table 20 (Life) — which lost the Item dimension
(Amount / Growth% / Share%) and, for Life, the Fund × Category hierarchy.

Schema mapping:
    dimension         = Industry
    insurer (entity)  = "General, Health & RE (Industry)" | "Life Insurers (Industry)"
    line_of_business  = Investments (AUM)
    class_of_business = fund ("Life Fund"/"Pension..."/"ULIP Funds"/"GRAND TOTAL")
                        for Life; "All Funds" for General
    metric            = "{category} (₹Crore)"  (amount)
                        "{category} - Growth (Per cent)"
                        "{category} - Share (Per cent)"
                        "Share of Total AUM (Per cent)" (Life fund-share sub-table)
    financial_year    = FY form (handbook calendar year Y -> "{Y-1}-{Y2}")

Parse+validate by default; --apply <db> surgically replaces the section.
"""
import argparse
import os
import re
import sys

import pandas as pd

LOB = "Investments (AUM)"
GEN = "General, Health & RE (Industry)"
LIFE = "Life Insurers (Industry)"


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _fy(yr):
    """Calendar 'as on 31 March Y' -> fiscal '{Y-1}-{Y2}'."""
    y = int(float(yr))
    return f"{y - 1}-{str(y)[2:]}"


def _num(x):
    if pd.isna(x):
        return None
    s = str(x).strip().replace(",", "")
    s = s.strip("()")                    # "(-4.81)" -> "-4.81" (minus is inside)
    v = pd.to_numeric(s, errors="coerce")
    return None if pd.isna(v) else float(v)


def _rec(entity, fy, metric, value, cob):
    return {"dimension": "Industry", "insurer": entity, "financial_year": fy,
            "quarter": "Annual", "metric": metric, "value": value,
            "line_of_business": LOB, "class_of_business": cob,
            "source_file": "handbook_2024-25_parts.xlsx"}


def parse_general(path):
    df = pd.read_excel(path, sheet_name="46", header=None)
    years = {c: _fy(df.iloc[2, c]) for c in range(2, df.shape[1])
             if pd.notna(df.iloc[2, c])}
    cat = df.iloc[:, 0].ffill()
    out = []
    for r in range(3, df.shape[0]):
        item = df.iloc[r, 1]
        if pd.isna(item):
            continue
        category = _clean(cat[r]).rstrip(".")
        item = _clean(item)
        if item.startswith("Amount"):
            mk = f"{category} (₹Crore)"
        elif item.startswith("Growth"):
            mk = f"{category} - Growth (Per cent)"
        elif item.startswith("Share"):
            mk = f"{category} - Share (Per cent)"
        else:
            continue
        for c, fy in years.items():
            v = _num(df.iloc[r, c])
            if v is not None:
                out.append(_rec(GEN, fy, mk, v, "All Funds"))
    return pd.DataFrame(out)


def parse_life(path):
    df = pd.read_excel(path, sheet_name="20", header=None)
    out = []
    # --- main block: Fund x Category x {Amount(r), Growth(r+1)} ---
    years = {c: _fy(df.iloc[3, c]) for c in range(2, df.shape[1])
             if pd.notna(df.iloc[3, c])}
    fund = None
    r = 4
    while r < df.shape[0]:
        c0, c1 = df.iloc[r, 0], df.iloc[r, 1]
        c0s = _clean(c0) if pd.notna(c0) else ""
        if c0s.startswith("Note") or c0s.startswith("SHARE OF") or c0s == "":
            if c0s.startswith("SHARE OF"):
                break
        if pd.notna(c0) and c0s and not c0s.startswith("GRAND"):
            fund = c0s
        category = None
        if c0s.startswith("GRAND"):
            category, this_fund = "GRAND TOTAL", "GRAND TOTAL"
        elif pd.notna(c1):
            category, this_fund = _clean(c1), fund
        if category:
            # amount row r, growth row r+1 (col1 blank)
            for c, fy in years.items():
                a = _num(df.iloc[r, c])
                if a is not None:
                    out.append(_rec(LIFE, fy, f"{category} (₹Crore)", a, this_fund))
            if r + 1 < df.shape[0] and pd.isna(df.iloc[r + 1, 1]) and pd.isna(df.iloc[r + 1, 0]):
                for c, fy in years.items():
                    g = _num(df.iloc[r + 1, c])
                    if g is not None:
                        out.append(_rec(LIFE, fy, f"{category} - Growth (Per cent)", g, this_fund))
                r += 2
                continue
        r += 1
    # --- secondary block: Fund share of total AUM (%) ---
    hdr = None
    for rr in range(df.shape[0]):
        if _clean(df.iloc[rr, 0]).startswith("SHARE OF"):
            hdr = rr
            break
    if hdr is not None:
        yrow = next(rr for rr in range(hdr, df.shape[0])
                    if _clean(df.iloc[rr, 0]) == "Particulars")
        syears = {c: _fy(df.iloc[yrow, c]) for c in range(2, df.shape[1])
                  if pd.notna(df.iloc[yrow, c])}
        for rr in range(yrow + 1, df.shape[0]):
            f = df.iloc[rr, 0]
            if pd.isna(f):
                continue
            for c, fy in syears.items():
                v = _num(df.iloc[rr, c])
                if v is not None:
                    out.append(_rec(LIFE, fy, "Share of Total AUM (Per cent)",
                                    v, _clean(f)))
    return pd.DataFrame(out)


def validate(new):
    print(f"parsed {len(new):,} rows ({(new.insurer==LIFE).sum()} Life, "
          f"{(new.insurer==GEN).sum()} General)")
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    dup = new.groupby(KEY).size()
    print("  duplicate keys remaining:", int((dup > 1).sum()), "(want 0)")
    # General: category amounts sum to TOTAL
    g = new[(new.insurer == GEN) & new.metric.str.endswith("(₹Crore)")].copy()
    g["_cat"] = g.metric.str.replace(r" \(₹Crore\)$", "", regex=True)
    chk = ok = 0
    for fy, sub in g.groupby("financial_year"):
        tot = sub[sub._cat == "TOTAL"].value
        parts = sub[sub._cat != "TOTAL"].value.sum()
        if len(tot) == 1 and tot.iloc[0]:
            chk += 1; ok += abs(parts - tot.iloc[0]) / tot.iloc[0] < 0.02
    print(f"  General: categories sum to TOTAL: {ok}/{chk} years")
    # Life: fund-share sums to 100%
    s = new[(new.insurer == LIFE) & (new.metric == "Share of Total AUM (Per cent)")
            & (new.class_of_business != "TOTAL")]
    chk2 = ok2 = 0
    for fy, sub in s.groupby("financial_year"):
        chk2 += 1; ok2 += abs(sub.value.sum() - 100) < 1.5
    print(f"  Life: fund shares sum to ~100%: {ok2}/{chk2} years")


def apply(db, new):
    import sqlite3
    con = sqlite3.connect(db); cur = con.cursor()
    old = cur.execute("SELECT COUNT(*) FROM financial_metrics WHERE dimension='Industry' "
                      "AND line_of_business=?", (LOB,)).fetchone()[0]
    cur.execute("DELETE FROM financial_metrics WHERE dimension='Industry' "
                "AND line_of_business=?", (LOB,))
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
    ap.add_argument("--csv", default="/tmp/aum_reingested.csv")
    ap.add_argument("--apply", metavar="DB")
    args = ap.parse_args()
    gen = parse_general(os.path.join(args.handbook, "Part II.xlsx"))
    life = parse_life(os.path.join(args.handbook, "Part I.xlsx"))
    new = pd.concat([gen, life], ignore_index=True)
    validate(new)
    new.to_csv(args.csv, index=False)
    print(f"=> {args.csv}")
    if args.apply:
        apply(args.apply, new)


if __name__ == "__main__":
    sys.exit(main())
