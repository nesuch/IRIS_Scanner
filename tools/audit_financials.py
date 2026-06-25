#!/usr/bin/env python3
"""
Internal-consistency audit for `financial_metrics` — READ-ONLY.

Flags values that are almost certainly wrong WITHOUT needing the published
Handbook, by checking the data against itself:

  A. Conflicting duplicates  — same (dim,entity,metric,year,quarter,lob,class)
     appearing with DIFFERENT values (the pivot silently sums these -> wrong).
  B. Impossible percentages   — unit=PERCENT with |value| absurdly large.
  C. Negative counts          — unit=COUNT with value < 0 (can't have -ve offices).
  D. Per-metric magnitude outliers — a value >> the typical scale of that metric
     (classic unit/typo error, e.g. a ratio of 2,874,524).
  E. Year-over-year explosions — a series that jumps >50x between adjacent years.

Writes a CSV of every flagged row to /tmp/iris_audit_flags.csv and prints a
summary with examples. Nothing is modified.

  python tools/audit_financials.py [--db iris.db] [--csv /tmp/iris_audit_flags.csv]
"""
import argparse
import sqlite3
import sys

import pandas as pd


KEY = ["dimension", "insurer", "metric", "financial_year", "quarter",
       "line_of_business", "class_of_business"]


def load(db):
    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM financial_metrics", con)
    con.close()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def check_conflicting_dupes(df):
    g = df.groupby(KEY)["value"].nunique(dropna=True)
    bad_keys = g[g > 1].index
    if len(bad_keys) == 0:
        return pd.DataFrame()
    mask = df.set_index(KEY).index.isin(bad_keys)
    out = df[mask].copy()
    out["_check"] = "conflicting_duplicate"
    return out


def check_bad_percent(df):
    if "unit_code" not in df.columns:
        return pd.DataFrame()
    m = (df["unit_code"] == "PERCENT") & (df["value"].abs() > 1000)
    out = df[m].copy(); out["_check"] = "impossible_percent(|v|>1000)"
    return out


def check_negative_counts(df):
    if "unit_code" not in df.columns:
        return pd.DataFrame()
    m = (df["unit_code"] == "COUNT") & (df["value"] < 0)
    out = df[m].copy(); out["_check"] = "negative_count"
    return out


SERIES_COLS = ["dimension", "insurer", "metric", "quarter",
               "line_of_business", "class_of_business"]


def check_series_outliers(df, factor=100):
    """Flag a value >> the median of ITS OWN series (same insurer/metric/line over
    years). Comparing within-series — not across insurers — avoids flagging large
    players (LIC's whole series is big but internally consistent); it catches a
    single value that is wildly off its own history (a unit/typo error)."""
    rows = []
    for _, sub in df.groupby(SERIES_COLS):
        v = sub["value"].abs()
        v = v[v > 0].dropna()
        if len(v) < 5:
            continue
        med = v.median()
        if med <= 0:
            continue
        out = sub[sub["value"].abs() > factor * med]
        if len(out):
            o = out.copy()
            o["_check"] = f"series_outlier(>{factor}x own-series median {med:.4g})"
            rows.append(o)
    return pd.concat(rows) if rows else pd.DataFrame()


def check_isolated_spikes(df, factor=20):
    """Flag an ISOLATED spike: a value > factor x BOTH its previous and next year
    (a spike that returns). A genuine growth trend rises and stays, so requiring
    the spike to revert removes the false positives from real volatility (e.g.
    Crop premium swings)."""
    if "period_type" in df.columns:
        d = df[df["period_type"] == "FISCAL"].copy()
    else:
        d = df.copy()
    d = d.dropna(subset=["value"])
    d["_yr"] = d["financial_year"].astype(str).str.extract(r"^(\d{4})").astype(float)
    rows = []
    for _, sub in d.groupby(SERIES_COLS):
        sub = sub.sort_values("_yr")
        vals = sub["value"].tolist(); recs = sub.to_dict("records")
        for i in range(1, len(vals) - 1):
            a, b, c = vals[i - 1], vals[i], vals[i + 1]
            if min(abs(a), abs(c)) < 1 or abs(b) < 1:
                continue
            up = abs(b) > factor * abs(a) and abs(b) > factor * abs(c)
            down = abs(b) * factor < abs(a) and abs(b) * factor < abs(c)
            if up or down:
                rows.append({**recs[i],
                             "_check": f"isolated_spike(neighbors {a:.4g} / {c:.4g})"})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="iris.db")
    ap.add_argument("--csv", default="/tmp/iris_audit_flags.csv")
    args = ap.parse_args()

    df = load(args.db)
    print(f"loaded {len(df):,} rows from {args.db}\n")

    checks = [
        ("A. conflicting duplicates", check_conflicting_dupes(df)),
        ("B. impossible percentages", check_bad_percent(df)),
        ("C. negative counts", check_negative_counts(df)),
        ("D. within-series magnitude outliers", check_series_outliers(df)),
        ("E. isolated spikes (spike-and-return)", check_isolated_spikes(df)),
    ]

    all_flags = []
    show = ["dimension", "insurer", "metric", "financial_year",
            "line_of_business", "class_of_business", "value", "_check"]
    for name, res in checks:
        n = len(res)
        print(f"{name}: {n} flagged")
        if n:
            ex = res[[c for c in show if c in res.columns]].head(5)
            for _, r in ex.iterrows():
                print("    •", " | ".join(f"{r[c]}" for c in ex.columns))
            all_flags.append(res)
        print()

    if all_flags:
        combined = pd.concat(all_flags, ignore_index=True)
        cols = [c for c in show if c in combined.columns] + \
               [c for c in combined.columns if c not in show]
        combined[cols].to_csv(args.csv, index=False)
        print(f"=> {len(combined):,} total flagged rows written to {args.csv}")
    else:
        print("No issues flagged. 🎉")


if __name__ == "__main__":
    sys.exit(main())
