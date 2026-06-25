#!/usr/bin/env python3
"""
Surgical re-ingestion of Handbook Table 104 — "Channel-wise Health Insurance
Business" (Part V) — which lost the BUSINESS-TYPE dimension (Individual / Group /
Government / Grand Total). The table's segment naming changed across editions
(early years: "Group incl. Govt & RSBY"; later: "Group excl. Govt" + "Government
Business"), and claims measures were added in later years — which is why the old
flatten produced colliding keys (e.g. one channel/metric/year with two values).

Schema mapping:
    dimension         = Channel
    insurer (entity)  = channel name (Brokers, Individual Agents, ...)
    line_of_business  = Health
    class_of_business = business type (normalised: Individual / Group (excl. Govt)
                        / Group (incl. Govt & RSBY) / Government (incl. RSBY) / Grand Total)
    metric            = measure (No. of Policies / Persons Covered / Gross Premium
                        / No. of claims paid / Claims Paid)
    financial_year    = FY form (already in the header), quarter = Annual
"""
import argparse
import os
import re
import sys

import pandas as pd

LOB = "Health"
MEAS = {
    "no.of policies issued": "No. of Policies (Nos.)",
    "no. of persons covered ('000s)": "No. of Persons Covered ('000s)",
    "gross premium (₹lakh)": "Gross Premium (₹Lakh)",
    "no. of claims paid": "No. of claims paid (Nos.)",
    "claims paid (₹lakh)": "Claims Paid (₹Lakh)",
}
STOP_CHANNELS = ("share of", "name of the channel")


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _norm_bt(s):
    t = _clean(s).lower()
    if "grand total" in t:
        return "Grand Total"
    if "individual" in t:
        return "Individual"
    # NB: check group BEFORE government — "Group Business (Excluding Government
    # Business)" contains the substring "government business".
    if "group" in t:
        return "Group (excl. Govt)" if "exclud" in t else "Group (incl. Govt & RSBY)"
    if "government" in t:
        return "Government (incl. RSBY)"
    return _clean(s)


def _num(x):
    if pd.isna(x):
        return None
    v = pd.to_numeric(str(x).replace(",", "").strip("()"), errors="coerce")
    return None if pd.isna(v) else float(v)


def parse(path):
    df = pd.read_excel(path, sheet_name="104", header=None)
    year = df.iloc[1].ffill()
    btype = df.iloc[2].ffill()
    measure = df.iloc[3]
    fy_re = re.compile(r"^\d{4}-\d{2}$")

    cols = []
    for c in range(1, df.shape[1]):
        y, b, m = year[c], btype[c], measure[c]
        if pd.isna(y) or not fy_re.match(_clean(y)):
            continue
        if pd.isna(b) or pd.isna(m) or _clean(m).lower() not in MEAS:
            continue
        cols.append(c)

    out = []
    for r in range(4, df.shape[0]):
        ch = df.iloc[r, 0]
        if pd.isna(ch):
            continue
        channel = _clean(ch)
        if channel.lower().startswith(STOP_CHANNELS):
            break
        for c in cols:
            v = _num(df.iloc[r, c])
            if v is None:
                continue
            out.append({
                "dimension": "Channel", "insurer": channel,
                "financial_year": _clean(year[c]), "quarter": "Annual",
                "metric": MEAS[_clean(measure[c]).lower()], "value": v,
                "line_of_business": LOB,
                "class_of_business": _norm_bt(btype[c]),
                "source_file": "handbook_2024-25_parts.xlsx"})
    return pd.DataFrame(out)


def validate(new):
    print(f"parsed {len(new):,} rows | channels {new.insurer.nunique()} | "
          f"classes {sorted(new.class_of_business.unique())}")
    KEY = ["dimension", "insurer", "financial_year", "quarter", "metric",
           "line_of_business", "class_of_business"]
    print("  duplicate keys remaining:", int((new.groupby(KEY).size() > 1).sum()), "(want 0)")
    # consistency: all non-Grand-Total segments ~= Grand Total (per chan/yr/metric);
    # scheme-agnostic so it works for both early (2-segment) and later (3-segment) years.
    chk = ok = 0
    for (ch, fy, mt), g in new.groupby(["insurer", "financial_year", "metric"]):
        gt = g[g.class_of_business == "Grand Total"].value
        parts = g[g.class_of_business != "Grand Total"].value.sum()
        if len(gt) == 1 and gt.iloc[0] and parts > 0:
            chk += 1; ok += abs(parts - gt.iloc[0]) / abs(gt.iloc[0]) < 0.02
    print(f"  segments sum to Grand Total: {ok}/{chk} groups ({100*ok/chk:.0f}%)" if chk else "  n/a")


def apply(db, new):
    import sqlite3
    con = sqlite3.connect(db); cur = con.cursor()
    old = cur.execute("SELECT COUNT(*) FROM financial_metrics WHERE dimension='Channel' "
                      "AND line_of_business=?", (LOB,)).fetchone()[0]
    cur.execute("DELETE FROM financial_metrics WHERE dimension='Channel' AND line_of_business=?", (LOB,))
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
    ap.add_argument("--csv", default="/tmp/table104_reingested.csv")
    ap.add_argument("--apply", metavar="DB")
    args = ap.parse_args()
    new = parse(os.path.join(args.handbook, "Part V.xlsx"))
    validate(new)
    new.to_csv(args.csv, index=False)
    print(f"=> {args.csv}")
    if args.apply:
        apply(args.apply, new)


if __name__ == "__main__":
    sys.exit(main())
