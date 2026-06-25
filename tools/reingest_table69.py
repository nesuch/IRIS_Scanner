#!/usr/bin/env python3
"""
Surgical re-ingestion of Handbook Table 69 — "State-wise Personal Accident
Insurance Business" (Part III) — whose BUSINESS-TYPE dimension was lost/mangled,
producing conflicts concentrated in the IRCTC/scheme classes.

Table 69's business-type labels are highly varied and exclusionary across years
("Group Business (other than PMSBY, PMJDY, IRCTC & Govt Sponsored Schemes)"), so
we keep them VERBATIM (whitespace-normalised only) rather than risk mis-bucketing.
Each verbatim label is unique per year -> keys become unique -> 0 conflicts, and
the data stays exactly faithful to the published handbook.

    dimension=State, insurer(entity)=State/UT, line_of_business=Personal Accident,
    class_of_business=verbatim business-type, metric=measure, quarter=Annual
"""
import argparse
import os
import re
import sys

import pandas as pd

LOB = "Personal Accident"
MEAS = {
    "no.of policies issued": "No. of Policies (Nos.)",
    "no. of persons covered ('000s)": "No. of Persons Covered ('000s)",
    "gross premium (₹lakh)": "Gross Premium (₹Lakh)",
    "no. of claims paid": "No. of claims paid (Nos.)",
    "claims paid (₹lakh)": "Claims Paid (₹Lakh)",
}


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _num(x):
    if pd.isna(x) or str(x).strip() in ("-", ""):
        return None
    v = pd.to_numeric(str(x).replace(",", "").strip("()"), errors="coerce")
    return None if pd.isna(v) else float(v)


def _fill_within_years(year_row, label_row, cols):
    """Forward-fill label_row but RESET at each year change (avoids a prior year's
    last business-type bleeding into the next year's first columns)."""
    out = {}
    cur_year = None
    cur_label = None
    for c in cols:
        y = year_row[c]
        if pd.notna(y):
            cur_year = _clean(y)
            cur_label = None             # reset business-type at a new year block
        lab = label_row[c]
        if pd.notna(lab):
            cur_label = _clean(lab)
        out[c] = cur_label
    return out


def parse(path):
    df = pd.read_excel(path, sheet_name="69", header=None)
    year = df.iloc[2]
    fy_re = re.compile(r"^\d{4}-\d{2}$")
    allcols = list(range(2, df.shape[1]))
    btype = _fill_within_years(year, df.iloc[3], allcols)
    yfill = {}
    cur = None
    for c in allcols:
        if pd.notna(year[c]) and fy_re.match(_clean(year[c])):
            cur = _clean(year[c])
        yfill[c] = cur
    measure = df.iloc[4]

    cols = [c for c in allcols
            if yfill[c] and fy_re.match(yfill[c]) and btype[c]
            and pd.notna(measure[c]) and _clean(measure[c]).lower() in MEAS]

    out = []
    for r in range(5, df.shape[0]):
        state = df.iloc[r, 1]
        if pd.isna(state):
            continue
        state = _clean(state)
        if state.lower() in ("total", "grand total"):
            state = "All India (Total)"
        for c in cols:
            v = _num(df.iloc[r, c])
            if v is None:
                continue
            out.append({
                "dimension": "State", "insurer": state,
                "financial_year": yfill[c], "quarter": "Annual",
                "metric": MEAS[_clean(measure[c]).lower()], "value": v,
                "line_of_business": LOB,
                "class_of_business": _clean(btype[c]).rstrip("*# "),
                "source_file": "handbook_2024-25_parts.xlsx"})
    return pd.DataFrame(out)


def validate(new):
    print(f"parsed {len(new):,} rows | states {new.insurer.nunique()} | "
          f"business-types {new.class_of_business.nunique()}")
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    print("  duplicate keys remaining:", int((new.groupby(KEY).size() > 1).sum()), "(want 0)")
    # Grand Total >= any single segment (per state/year/metric)
    chk = ok = 0
    for (st, fy, mt), g in new.groupby(["insurer", "financial_year", "metric"]):
        gt = g[g.class_of_business.str.lower().str.startswith("grand total")].value
        seg = g[~g.class_of_business.str.lower().str.startswith("grand total")].value
        if len(gt) == 1 and len(seg) and gt.iloc[0] > 0:
            chk += 1; ok += gt.iloc[0] >= seg.max() - 1
    print(f"  Grand Total >= max segment: {ok}/{chk} ({100*ok/chk:.0f}%)" if chk else "")


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
    ap.add_argument("--csv", default="/tmp/table69_reingested.csv")
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
