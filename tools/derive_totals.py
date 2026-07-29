"""Derive what every `total`/`sub_total` row is the sum OF, then let them be dropped.

WHY THESE ROWS EXIST AT ALL
Canonicalisation normalised metric NAMES, units and financial years. It never
classified rows semantically. The segmenter faithfully extracts every row of a
handbook table — including its "Total", "Sub Total (A)" and "S.No" rows — and
canonicalisation then handed those tidy ids exactly like real facts. So the
database has always mixed LEAF facts with ROLL-UPS of those same facts, with
nothing marking which is which. That is double counting waiting to happen: any
"sum this metric" query over a table that contains its own total is wrong.

THE TARGET STATE
Hold only canonical leaf items. Totals are DERIVED on demand from a recorded
composition, so the number is reproducible and its provenance is explicit.

HOW THE COMPOSITION IS DERIVED
Not by subset-sum guessing. The rows carry document order in `id`, so within one
(insurer, year, line_of_business, class_of_business) block a total is the sum of
the leaf rows that PRECEDE it and follow the previous total. That candidate set is
then VERIFIED by arithmetic against the reported total, per instance. A
composition is only recorded when it reconciles across a large majority of the
instances it appears in — never on the strength of one example.

    python tools/derive_totals.py <db>              # derive + verify, write JSON
    python tools/derive_totals.py <db> --delete     # also drop verified total rows
"""
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict

# Rows that are pure scaffolding: they carry no fact and no composition worth
# keeping. Unlike totals there is nothing to derive — they simply leave.
SCAFFOLD = {
    "s_no", "sl_no", "sr_no", "serial_no", "particulars", "description",
    "a", "b", "c", "d", "e",
}
TOTAL_HINTS = ("total", "sub_total", "grand_total")
# Reconciliation tolerance: handbook tables are rounded, so an exact match is not
# required. 0.5% (or ±100 base units for near-zero rows) still proves the set.
REL_TOL, ABS_TOL = 0.005, 100.0


def is_scaffold(mid):
    base = mid.split("__")[0]
    return base in SCAFFOLD


def is_total(mid):
    base = mid.split("__")[0]
    return any(base == h or base.startswith(h + "_") or base == h for h in TOTAL_HINTS)


def reconciles(total, parts):
    s = sum(parts)
    if abs(total) < 1:
        return abs(s - total) <= ABS_TOL
    return abs(s - total) <= max(abs(total) * REL_TOL, ABS_TOL)


def derive(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT id, insurer_id, fy_canonical, line_of_business, class_of_business,
               metric_id, value_base
        FROM financial_metrics
        WHERE entity_type='insurer' AND metric_id IS NOT NULL AND value_base IS NOT NULL
        ORDER BY insurer_id, fy_canonical, line_of_business, class_of_business, id
    """).fetchall()
    con.close()

    blocks = defaultdict(list)
    for r in rows:
        blocks[(r["insurer_id"], r["fy_canonical"],
                r["line_of_business"], r["class_of_business"])].append(r)

    # candidate composition -> how often it reconciled / how often it was tried
    tally = defaultdict(lambda: Counter())
    for _, block in blocks.items():
        run = []                     # leaf rows since the previous total
        for r in block:
            mid = r["metric_id"]
            if is_scaffold(mid):
                continue
            if is_total(mid):
                if run:
                    parts = tuple(x["metric_id"] for x in run)
                    ok = reconciles(float(r["value_base"]),
                                    [float(x["value_base"]) for x in run])
                    tally[mid][("parts", parts)] += 0      # ensure key exists
                    tally[mid][("try", parts)] += 1
                    if ok:
                        tally[mid][("ok", parts)] += 1
                run = []             # a total closes its section
            else:
                run.append(r)

    out = {}
    for mid, c in tally.items():
        best, best_ok, best_try = None, 0, 0
        for key, n in c.items():
            if key[0] != "try":
                continue
            parts = key[1]
            tried = n
            okn = c[("ok", parts)]
            # Prefer the composition that reconciles most often; break ties on how
            # many instances back it, so a one-off coincidence cannot win.
            if (okn, tried) > (best_ok, best_try):
                best, best_ok, best_try = parts, okn, tried
        if best is None:
            continue
        total_tries = sum(n for k, n in c.items() if k[0] == "try")
        total_ok = sum(n for k, n in c.items() if k[0] == "ok")
        out[mid] = {
            "components": list(best),
            "reconciled_instances": best_ok,
            "instances_with_this_shape": best_try,
            "all_instances": total_tries,
            "all_reconciled": total_ok,
            "confidence": round(total_ok / total_tries, 3) if total_tries else 0.0,
        }
    return out


def main():
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        sys.exit("usage: derive_totals.py <path-to-COPY-of-iris.db> [--delete]")
    db = sys.argv[1]
    comps = derive(db)

    verified = {k: v for k, v in comps.items() if v["confidence"] >= 0.7}
    weak = {k: v for k, v in comps.items() if v["confidence"] < 0.7}

    print(f"\n=== derived {len(comps)} total definitions ===")
    print(f"  verified (>=70% of instances reconcile): {len(verified)}")
    print(f"  weak / not reconciled                 : {len(weak)}")

    for k, v in sorted(verified.items(), key=lambda x: -x[1]["all_instances"])[:12]:
        print(f"\n  {k}  [{v['confidence']:.0%} of {v['all_instances']} instances]")
        for c in v["components"]:
            print(f"      + {c}")

    if weak:
        print("\n  --- NOT verified (kept in DB, composition unknown) ---")
        for k, v in sorted(weak.items(), key=lambda x: -x[1]["all_instances"])[:10]:
            print(f"      {k}  {v['confidence']:.0%} of {v['all_instances']}")

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "total_definitions.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"verified": verified, "unverified": weak}, fh, indent=1, sort_keys=True)
    print(f"\n  definitions written -> {path}")

    if "--delete" in sys.argv:
        con = sqlite3.connect(db)
        # Scaffolding goes unconditionally; totals only where the composition is
        # recorded, so nothing becomes unrecoverable-by-derivation.
        scaffold_ids = [r[0] for r in con.execute(
            "SELECT DISTINCT metric_id FROM financial_metrics WHERE metric_id IS NOT NULL")
            if is_scaffold(r[0])]
        n1 = n2 = 0
        try:
            con.execute("BEGIN")
            for mid in scaffold_ids:
                n1 += con.execute("DELETE FROM financial_metrics WHERE metric_id=?",
                                  (mid,)).rowcount
            for mid in verified:
                n2 += con.execute("DELETE FROM financial_metrics WHERE metric_id=?",
                                  (mid,)).rowcount
            con.commit()
            print(f"  deleted {n1} scaffold rows and {n2} verified-total rows")
        except Exception as e:
            con.rollback()
            print(f"  ROLLED BACK: {e}")
            raise
        finally:
            con.close()
    else:
        print("\n  DRY RUN — nothing deleted. Re-run with --delete once the "
              "compositions above look right.")


if __name__ == "__main__":
    main()
