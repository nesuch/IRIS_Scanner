#!/usr/bin/env python3
"""
Compute data-quality flags for `financial_metrics` and write them to a
`data_quality_flags` table — so the Data Explorer can WARN users on sections
whose source tables lost a sub-dimension in the tidy transformation (multiple
distinct data points collapse onto one key -> values silently sum/duplicate).

A (dimension, line_of_business) section is flagged when a meaningful share of
its rows fall in "conflicting" keys (same key, different values). Major LOBs
that are mostly clean (e.g. Insurer/Health) are NOT flagged — only the genuinely
broken breakdown tables.

Additive + idempotent (rebuilds the flag table each run). READ-ONLY on the
financial data itself.

  python tools/flag_data_quality.py [--db iris.db] [--min-rows 50] [--min-frac 0.25]
"""
import argparse
import sqlite3
import sys

import pandas as pd

KEY = ["dimension", "insurer", "metric", "financial_year", "quarter",
       "line_of_business", "class_of_business"]


def compute(db, min_rows, min_frac):
    con = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT * FROM financial_metrics", con)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["_k"] = df[KEY].astype(str).agg("||".join, axis=1)
    df["_conf"] = df.groupby("_k")["value"].transform("nunique") > 1
    g = df.groupby(["dimension", "line_of_business"]).agg(
        total=("value", "size"), conf=("_conf", "sum"))
    g["frac"] = g["conf"] / g["total"]
    flagged = g[(g["conf"] >= min_rows) & (g["frac"] >= min_frac)].reset_index()

    con.execute("DROP TABLE IF EXISTS data_quality_flags")
    con.execute("""CREATE TABLE data_quality_flags (
        dimension TEXT, line_of_business TEXT, conflict_rows INTEGER,
        total_rows INTEGER, conflict_frac REAL, severity TEXT, note TEXT)""")
    note = ("This source table has unresolved sub-categories collapsed onto the "
            "same key, so granular values here may be duplicated or summed. Use "
            "totals with caution; per-line figures may be unreliable.")
    rows = []
    for _, r in flagged.iterrows():
        sev = "broken" if r["frac"] >= 0.5 else "partial"
        rows.append((r["dimension"], r["line_of_business"], int(r["conf"]),
                     int(r["total"]), round(float(r["frac"]), 3), sev, note))
    con.executemany("INSERT INTO data_quality_flags VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="iris.db")
    ap.add_argument("--min-rows", type=int, default=50)
    ap.add_argument("--min-frac", type=float, default=0.25)
    args = ap.parse_args()
    rows = compute(args.db, args.min_rows, args.min_frac)
    print(f"wrote {len(rows)} data-quality flags to {args.db}:")
    for d, lob, conf, tot, frac, sev, _ in sorted(rows, key=lambda x: -x[4]):
        print(f"   [{sev:7}] {d} / {lob}: {conf}/{tot} rows conflicted ({frac:.0%})")


if __name__ == "__main__":
    sys.exit(main())
