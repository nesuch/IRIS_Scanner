#!/usr/bin/env python3
"""
Categorization layer for `line_of_business` and `class_of_business`.

These two columns are OVERLOADED — each holds several different KINDS of value
(real business lines, totals, account/statement sections, breakdown axes, govt
schemes, and — in State-dimension rows — insurer names mis-filed as a "class").
A flat 79/164-item dropdown is therefore unusable. This layer:

  - merges pure format variants ("Marine (Hull)" == "Marine Hull", "Misc" ==
    "Miscellaneous") into one canonical id, WITHOUT merging genuinely distinct
    lines (parent "Marine" stays separate from "Marine Hull");
  - tags every value with a TYPE so the UI can group / hide them:
      LOB : business_line | aggregate | account_section | breakdown | other
      COB : class | aggregate | fund | scheme | entity | other
    ("entity" = an insurer name that leaked into class_of_business);
  - adds lob_id / lob_type / cob_id / cob_type onto financial_metrics and builds
    lob_dim / cob_dim reference tables.

Additive, idempotent, reversible — same contract as canonicalize_financials.py.
  python tools/categorize_dimensions.py --db iris.db --report
  python tools/categorize_dimensions.py --db iris.db --revert
"""
import argparse
import re
import sqlite3
import sys

from canonicalize_financials import slug, insurer_key   # reuse helpers

# ----------------------------------------------------------------------------
# Variant normalisation (merge format variants only — keep real lines distinct)
# ----------------------------------------------------------------------------
def norm_variant(v: str) -> str:
    s = (v or "").strip().lower()
    s = s.replace("&", " and ")
    s = s.replace("(", " ").replace(")", " ")        # "Marine (Hull)" -> "marine hull"
    s = s.replace("-", " ")                             # "on-line" -> "on line"
    s = re.sub(r"\bmisc\b", "miscellaneous", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def canonical_display(variants):
    """Prefer the most decorated form (has parens/caps) as the display label."""
    return sorted(variants, key=lambda v: (-("(" in v), -len(v), v))[0]


# ----------------------------------------------------------------------------
# Type classification
# ----------------------------------------------------------------------------
# Generic business words that, ALONE, are real classes/lines — never an insurer.
_GENERIC = {"general", "life", "health", "marine", "fire", "motor", "credit",
            "group", "individual", "miscellaneous", "misc", "aviation", "crop",
            "engineering", "liability", "others", "other", "total", "non", "all",
            "insurance", "reinsurance", "personal", "accident", "travel"}

# "Life" / "Non-Life" / "Reinsurance" are parallel sector lines (the natural
# top-level picks in Country/Industry views) — keep them as business_line, not
# aggregates. Only things that explicitly say "all" (or the composite combos) are
# treated as aggregates.
_LOB_AGG = re.compile(r"\b(all lines|all segments|^total$|health \+ pa)", re.I)
_LOB_ACCOUNT = re.compile(r"(account|balance sheet)", re.I)
_LOB_BREAKDOWN = re.compile(
    r"(in[- ]?force|offices|agents|network|claims development|status of claims|"
    r"investments? \(aum\)|persistency|grievance|ombudsman|settlement duration|"
    r"by region|by instrument|per agent|per policy|schedule|category|aging)", re.I)

_COB_SCHEME = re.compile(
    r"(rsby|pmsby|pmjay|pmjjby|pmjdy|irctc|pmjby|ab[- ]?pmjay|government sponsored|"
    r"\bscheme\b)", re.I)
_COB_FUND = re.compile(
    r"(funds?|sources of funds|income from investments|shareholders|policyholders)", re.I)
_COB_SUFFIX = re.compile(
    r"(insurance (co|ltd|company|limited)|life insurance|general insurance|"
    r"\bassurance\b|dai[- ]?ichi|nippon|sompo|lombard|ergo|allianz|generali)", re.I)
_COB_ARTIFACT = re.compile(r"(\(₹|benefit amount|particulars|premiums earned)", re.I)


def classify_lob(raw: str) -> str:
    if _LOB_AGG.search(raw):
        return "aggregate"
    if _LOB_ACCOUNT.search(raw):
        return "account_section"
    if _LOB_BREAKDOWN.search(raw):
        return "breakdown"
    return "business_line"


def make_cob_classifier(insurer_token_sets):
    def cob_tokens(raw):
        # Drop generic business words so a brand like "Star Health" reduces to its
        # distinctive token {star} and can match the insurer token-set (which is
        # built the same way). Purely-generic values reduce to {} -> not an entity.
        return {t for t in insurer_key(raw).split() if t and t not in _GENERIC}

    def is_entity(raw):
        if _COB_SUFFIX.search(raw):
            return True
        toks = cob_tokens(raw)
        if not toks:                            # purely generic -> a real class
            return False
        for iset in insurer_token_sets:         # brand tokens subset of an insurer
            if toks <= iset:
                return True
        return False

    def classify(raw):
        r = raw.strip()
        if r.lower() in ("all classes",) or re.fullmatch(r"(?i)total", r):
            return "aggregate"
        if is_entity(r):
            return "entity"
        if _COB_SCHEME.search(r):
            return "scheme"
        if _COB_ARTIFACT.search(r):
            return "other"
        if _COB_FUND.search(r):
            return "fund"
        return "class"
    return classify


# ----------------------------------------------------------------------------
# Apply / revert
# ----------------------------------------------------------------------------
ADDED_COLS = [("lob_id", "TEXT"), ("lob_type", "TEXT"),
              ("cob_id", "TEXT"), ("cob_type", "TEXT")]
DIM_TABLES = ["lob_dim", "cob_dim"]


def _cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def _build_dim(cur, col, classify, dim_table, id_prefix):
    raws = [r[0] for r in cur.execute(
        f"SELECT DISTINCT {col} FROM financial_metrics WHERE {col} IS NOT NULL AND {col}<>''")]
    groups = {}
    for v in raws:
        groups.setdefault(norm_variant(v), []).append(v)
    id_of, used = {}, set()
    rows = []
    for _, variants in groups.items():
        disp = canonical_display(variants)
        vid = slug(disp)
        while vid in used:
            vid += "_x"
        used.add(vid)
        vtype = classify(disp)
        rows.append((vid, disp, vtype, len(variants)))
        for v in variants:
            id_of[v] = (vid, vtype)
    cur.execute(f"CREATE TABLE IF NOT EXISTS {dim_table} ("
                f"{id_prefix}_id TEXT PRIMARY KEY, canonical TEXT, "
                f"{id_prefix}_type TEXT, variants INTEGER)")
    cur.execute(f"DELETE FROM {dim_table}")
    cur.executemany(f"INSERT INTO {dim_table} VALUES (?,?,?,?)", rows)
    return id_of


def apply(conn):
    cur = conn.cursor()
    existing = _cols(conn, "financial_metrics")
    for col, typ in ADDED_COLS:
        if col not in existing:
            cur.execute(f"ALTER TABLE financial_metrics ADD COLUMN {col} {typ}")

    # Insurer token sets (for detecting insurer names leaked into class column).
    insurers = [r[0] for r in cur.execute(
        "SELECT DISTINCT insurer FROM financial_metrics WHERE insurer IS NOT NULL")]
    try:
        insurers += [r[0] for r in cur.execute("SELECT canonical_name FROM insurer_dim")]
    except sqlite3.OperationalError:
        pass
    token_sets = []
    for name in set(insurers):
        toks = {t for t in insurer_key(name).split() if t and t not in _GENERIC}
        if toks:
            token_sets.append(toks)

    lob_id_of = _build_dim(cur, "line_of_business", classify_lob, "lob_dim", "lob")
    cob_classifier = make_cob_classifier(token_sets)
    cob_id_of = _build_dim(cur, "class_of_business", cob_classifier, "cob_dim", "cob")

    rows = cur.execute(
        "SELECT id, line_of_business, class_of_business FROM financial_metrics").fetchall()
    updates = []
    for rid, lob, cob in rows:
        lid, ltype = lob_id_of.get(lob, (None, None))
        cid, ctype = cob_id_of.get(cob, (None, None))
        updates.append((lid, ltype, cid, ctype, rid))
    cur.executemany(
        "UPDATE financial_metrics SET lob_id=?, lob_type=?, cob_id=?, cob_type=? WHERE id=?",
        updates)
    for col in ("lob_id", "cob_id"):
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_fm_{col} ON financial_metrics({col})")
    conn.commit()
    return {
        "lob_raw": len(set(r[1] for r in rows if r[1])),
        "lob_canonical": cur.execute("SELECT COUNT(*) FROM lob_dim").fetchone()[0],
        "cob_raw": len(set(r[2] for r in rows if r[2])),
        "cob_canonical": cur.execute("SELECT COUNT(*) FROM cob_dim").fetchone()[0],
        "cob_entities_evicted": cur.execute(
            "SELECT COUNT(*) FROM cob_dim WHERE cob_type='entity'").fetchone()[0],
        "rows_updated": len(updates),
    }


def revert(conn):
    cur = conn.cursor()
    for t in DIM_TABLES:
        cur.execute(f"DROP TABLE IF EXISTS {t}")
    for col in ("lob_id", "cob_id"):
        cur.execute(f"DROP INDEX IF EXISTS idx_fm_{col}")
    existing = _cols(conn, "financial_metrics")
    for col, _ in ADDED_COLS:
        if col in existing:
            try:
                cur.execute(f"ALTER TABLE financial_metrics DROP COLUMN {col}")
            except sqlite3.OperationalError:
                cur.execute(f"UPDATE financial_metrics SET {col}=NULL")
    conn.commit()


def report(conn):
    q = conn.execute
    print("\n--- categorization report ---")
    print("LOB types:", dict(q("SELECT lob_type, COUNT(*) FROM lob_dim GROUP BY lob_type ORDER BY 2 DESC")))
    print("COB types:", dict(q("SELECT cob_type, COUNT(*) FROM cob_dim GROUP BY cob_type ORDER BY 2 DESC")))
    print("\nclean business-line filter (was 79 mixed):")
    for (n,) in q("SELECT canonical FROM lob_dim WHERE lob_type='business_line' ORDER BY canonical LIMIT 40"):
        print("  ", n)
    print("\ninsurer names EVICTED from class filter (sample):")
    for (n,) in q("SELECT canonical FROM cob_dim WHERE cob_type='entity' ORDER BY canonical LIMIT 15"):
        print("  ", n)
    print("\nmerged format variants (LOB):")
    for cid, canon, t, v in q("SELECT * FROM lob_dim WHERE variants>1 ORDER BY variants DESC LIMIT 10"):
        print(f"   x{v}: {canon}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="iris.db")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    try:
        if args.revert:
            revert(conn)
            print("reverted categorization layer from", args.db)
            return
        print("applied to %s:" % args.db, apply(conn))
        if args.report:
            report(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
