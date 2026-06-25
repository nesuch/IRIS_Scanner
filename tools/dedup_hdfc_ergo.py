#!/usr/bin/env python3
"""
De-duplicate HDFC ERGO General Insurance (Insurer dimension) — Option 1.

HDFC ERGO General is the merged survivor of HDFC General, L&T General (2016-17)
and Apollo Munich / HDFC ERGO Health (2020). For the merger-era years the source
carries 2-4 rows under the single name "HDFC ERGO General Insurance Co. Ltd." for
the same key, which the pivot otherwise SUMS into a nonsensical figure.

The merged-entity series can't be disambiguated from the data alone, so we apply
a transparent, reproducible rule: keep the LARGEST (most-consolidated) reported
value per key and drop the rest. A disclaimer is surfaced in the app for any
HDFC ERGO figure (see iris_brain.load_data_quality_flags / entity disclaimer).

    python tools/dedup_hdfc_ergo.py --db iris.db        # parse + report (dry run)
    python tools/dedup_hdfc_ergo.py --db iris.db --apply
"""
import argparse
import sqlite3
import sys

ENTITY = "HDFC ERGO General Insurance Co. Ltd."
KEY = ["metric", "financial_year", "quarter", "line_of_business", "class_of_business"]


def run(db, do_apply):
    con = sqlite3.connect(db)
    rows = con.execute(
        f"SELECT id, {', '.join(KEY)}, value FROM financial_metrics "
        "WHERE dimension='Insurer' AND insurer=?", (ENTITY,)).fetchall()
    groups = {}
    for r in rows:
        rid = r[0]
        key = tuple("" if x is None else str(x) for x in r[1:1 + len(KEY)])
        val = r[-1]
        groups.setdefault(key, []).append((rid, val))

    delete_ids = []
    conflicted = 0
    for key, items in groups.items():
        vals = {round(v, 4) for _, v in items if v is not None}
        if len(vals) <= 1:
            continue                     # exact dupes / single value -> leave dedup to nothing
        conflicted += 1
        # keep the row with the largest |value|; delete the others
        keep = max(items, key=lambda iv: abs(iv[1]) if iv[1] is not None else -1)
        for rid, _ in items:
            if rid != keep[0]:
                delete_ids.append(rid)

    print(f"HDFC ERGO General: {len(groups)} keys, {conflicted} conflicting, "
          f"{len(delete_ids)} duplicate rows to remove")
    if do_apply and delete_ids:
        con.executemany("DELETE FROM financial_metrics WHERE id=?",
                        [(i,) for i in delete_ids])
        con.commit()
        # verify
        left = con.execute(
            "SELECT COUNT(*) FROM (SELECT 1 FROM financial_metrics WHERE dimension='Insurer' "
            f"AND insurer=? GROUP BY {', '.join(KEY)} HAVING COUNT(DISTINCT round(value,4))>1)",
            (ENTITY,)).fetchone()[0]
        print(f"applied: removed {len(delete_ids)} rows; conflicting keys remaining: {left}")
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="iris.db")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    run(args.db, args.apply)


if __name__ == "__main__":
    sys.exit(main())
