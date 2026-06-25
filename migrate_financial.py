#!/usr/bin/env python3
"""
Version-gated swap of the FINANCIAL-DATA tables from the baked image seed into
the live (Litestream-restored) DB.

Why: production restores iris.db from the GCS replica on every deploy, so changes
baked into the committed/seed iris.db never reach it. But the seed holds the
reconciled, handbook-sourced financial data. financial_metrics (and its derived
dim/flag tables) are PURELY handbook-sourced, so we can safely replace just those
tables on the live DB — leaving all operational tables (auth_users, flags,
departments, regulatory_clauses, announcements, search_logs, ...) untouched.

Runs at container start, AFTER `litestream restore` and BEFORE `litestream
replicate` (so the swap is captured into a fresh GCS generation). Idempotent:
a version marker in schema_meta means it runs once per data version.

Bump FINANCIAL_DATA_VERSION whenever the committed financial data changes.
"""
import argparse
import os
import sqlite3
import sys

FINANCIAL_DATA_VERSION = "2026-06-20-reconciled-v5"  # v5: signature-gated re-swap

# Tables derived purely from the handbook — safe to replace wholesale.
FINANCIAL_TABLES = [
    "financial_metrics", "insurer_dim", "metric_dim", "year_dim",
    "lob_dim", "cob_dim", "data_quality_flags",
]


def _get_version(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)")
    r = conn.execute("SELECT value FROM schema_meta WHERE key='financial_data_version'").fetchone()
    return r[0] if r else None


def _seed_signature(cur):
    """A cheap content fingerprint of the seed's financial data. Whenever the
    committed iris.db changes (rows added/edited/re-sourced), this changes too,
    so the swap re-fires automatically — no more manually bumping a version
    string (which kept getting burned into production and skipping real updates).
    """
    n, m, l, ins, s = cur.execute(
        "SELECT COUNT(*), COUNT(DISTINCT metric), COUNT(DISTINCT line_of_business), "
        "COUNT(DISTINCT insurer), ROUND(COALESCE(SUM(value), 0), 2) "
        "FROM seed.financial_metrics").fetchone()
    return f"{FINANCIAL_DATA_VERSION}|n={n}|m={m}|l={l}|i={ins}|sum={s}"


def migrate(live, seed):
    if not os.path.exists(seed):
        print(f"[migrate] seed {seed} missing — nothing to do")
        return
    conn = sqlite3.connect(live)
    cur = conn.cursor()
    cur.execute("ATTACH ? AS seed", (seed,))
    current = _get_version(conn)
    target = _seed_signature(cur)
    if current == target:
        print(f"[migrate] financial data already current ({current}); skip")
        cur.execute("DETACH seed")
        conn.close()
        return
    print(f"[migrate] swapping financial tables: {current} -> {target}")
    swapped = []
    for t in FINANCIAL_TABLES:
        meta = cur.execute(
            "SELECT type, sql FROM seed.sqlite_master WHERE tbl_name=? AND sql IS NOT NULL",
            (t,)).fetchall()
        table_sql = [s for ty, s in meta if ty == "table"]
        if not table_sql:
            continue                       # seed doesn't have this table — leave live as-is
        cur.execute(f'DROP TABLE IF EXISTS main."{t}"')
        cur.execute(table_sql[0])          # recreate with the seed's exact schema
        cur.execute(f'INSERT INTO main."{t}" SELECT * FROM seed."{t}"')
        for ty, s in meta:                 # recreate the table's indexes
            if ty == "index" and s:
                try:
                    cur.execute(s)
                except sqlite3.OperationalError:
                    pass
        swapped.append(t)
    # Defensive scrub: financial_metrics is purely handbook-sourced. Remove any
    # non-handbook rows (e.g. legacy POC unified_database.csv) so the result is
    # clean even if a stale seed ever carried them.
    scrubbed = cur.execute(
        "DELETE FROM financial_metrics "
        "WHERE source_file IS NOT NULL AND source_file NOT LIKE 'handbook%'").rowcount
    if scrubbed:
        print(f"[migrate] scrubbed {scrubbed} non-handbook (POC) rows")

    cur.execute(
        "INSERT INTO schema_meta(key, value) VALUES('financial_data_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (target,))
    conn.commit()
    cur.execute("DETACH seed")
    conn.close()
    print(f"[migrate] swapped {len(swapped)} tables {swapped} -> {target}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default=os.getenv("IRIS_DB_PATH", "/data/iris.db"))
    ap.add_argument("--seed", default="/app/iris.db")
    args = ap.parse_args()
    try:
        migrate(args.live, args.seed)
    except Exception as e:                  # never block container start on a migration error
        print(f"[migrate] ERROR (continuing): {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
