#!/usr/bin/env python3
"""
Surgical re-ingestion of insurer-wise Personal Accident — Handbook Tables 59
(Policies / Persons / Premium) and 63 (Net Earned Premium / Claims Incurred /
Incurred Claims Ratio), Part III. Same lost-business-type problem as the State PA
table (Table 69): the scheme/IRCTC business-type collapsed, producing conflicts
concentrated in the IRCTC/scheme classes.

Business-type labels are highly varied/exclusionary across years, so kept VERBATIM
(whitespace-normalised) -> unique keys -> 0 conflicts, fully faithful to source.

    dimension=Insurer, insurer=Insurer name, line_of_business=Personal Accident,
    class_of_business=verbatim business-type, metric=measure, quarter=Annual
"""
import argparse
import os
import re
import sys

import pandas as pd

LOB = "Personal Accident"
# measure (lowercased, trimmed) -> canonical metric name
MEAS = {
    "no. of policies": "No. of Policies (Nos.)",
    "no.of policies": "No. of Policies (Nos.)",
    "no. of persons covered ('000s)": "No. of Persons Covered ('000s)",
    "gross premium (₹lakh)": "Gross Premium (₹Lakh)",
    "net earned premium": "Net Earned Premium (₹Lakh)",
    "net earned permium": "Net Earned Premium (₹Lakh)",   # source typo
    "claims incurred (net)": "Claims Incurred (Net) (₹Lakh)",
    "incurred claims ratio": "Incurred Claims Ratio (Per cent)",
}
SHEETS = ["59", "63"]


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _num(x):
    if pd.isna(x) or str(x).strip() in ("-", ""):
        return None
    v = pd.to_numeric(str(x).replace(",", "").strip("()"), errors="coerce")
    return None if pd.isna(v) else float(v)


def _fill_within_years(year_row, label_row, cols):
    out, cur_label = {}, None
    for c in cols:
        if pd.notna(year_row[c]):
            cur_label = None            # reset business-type at each new year block
        if pd.notna(label_row[c]):
            cur_label = _clean(label_row[c])
        out[c] = cur_label
    return out


def parse_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    year = df.iloc[2]
    fy_re = re.compile(r"^\d{4}-\d{2}$")
    allcols = list(range(2, df.shape[1]))
    btype = _fill_within_years(year, df.iloc[3], allcols)
    yfill, cur = {}, None
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
        ins = df.iloc[r, 1]
        if pd.isna(ins):
            continue
        ins = _clean(ins)
        if ins.lower() in ("total", "grand total", "insurers"):
            continue
        for c in cols:
            v = _num(df.iloc[r, c])
            if v is None:
                continue
            out.append({
                "dimension": "Insurer", "insurer": ins,
                "financial_year": yfill[c], "quarter": "Annual",
                "metric": MEAS[_clean(measure[c]).lower()], "value": v,
                "line_of_business": LOB,
                "class_of_business": _clean(btype[c]).rstrip("*# "),
                "source_file": "handbook_2024-25_parts.xlsx"})
    return pd.DataFrame(out)


def validate(new):
    print(f"parsed {len(new):,} rows | insurers {new.insurer.nunique()} | "
          f"business-types {new.class_of_business.nunique()} | metrics {new.metric.nunique()}")
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    print("  duplicate keys remaining:", int((new.groupby(KEY).size() > 1).sum()), "(want 0)")


def apply(db, new):
    import sqlite3
    con = sqlite3.connect(db); cur = con.cursor()
    old = cur.execute("SELECT COUNT(*) FROM financial_metrics WHERE dimension='Insurer' "
                      "AND line_of_business=?", (LOB,)).fetchone()[0]
    cur.execute("DELETE FROM financial_metrics WHERE dimension='Insurer' AND line_of_business=?", (LOB,))
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
    ap.add_argument("--csv", default="/tmp/insurer_pa_reingested.csv")
    ap.add_argument("--apply", metavar="DB")
    args = ap.parse_args()
    path = os.path.join(args.handbook, "Part III.xlsx")
    new = pd.concat([parse_sheet(path, s) for s in SHEETS], ignore_index=True)
    # A few years repeat a business-type column verbatim (e.g. PMJDY in 2016-17);
    # keep the last (rightmost/final) column for those exact-key duplicates.
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    before = len(new)
    new = new.drop_duplicates(subset=KEY, keep="last").reset_index(drop=True)
    if before != len(new):
        print(f"  collapsed {before - len(new)} duplicate-column rows (kept last)")
    validate(new)
    new.to_csv(args.csv, index=False)
    print(f"=> {args.csv}")
    if args.apply:
        apply(args.apply, new)


if __name__ == "__main__":
    sys.exit(main())
