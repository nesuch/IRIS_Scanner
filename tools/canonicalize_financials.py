#!/usr/bin/env python3
"""
Canonicalization layer for `financial_metrics`.

Turns the queryable-but-messy long table into an AI-query-SAFE one by adding:
  - insurer_id  : canonical entity id (collapses "Star Health &" vs "and",
                  "Co. Ltd." vs "Ltd." so SUM/GROUP BY can't silently split)
  - metric_id   : id of a curated metric, with the unit pulled OUT of the name
  - unit_code   : controlled unit (INR / PERCENT / COUNT / USD / RATIO / ...)
  - unit_scale  : multiply `value` by this to get base units (Crore -> 1e7, ...)
  - fy_canonical: one canonical year string
  - period_type : FISCAL ("2024-25") vs CALENDAR ("2024")

Everything is ADDITIVE (new nullable columns + new dim tables) and IDEMPOTENT
(safe to re-run). Existing columns and existing app queries are untouched.

  python tools/canonicalize_financials.py --db iris.db            # apply
  python tools/canonicalize_financials.py --db iris.db --report   # apply + verify
  python tools/canonicalize_financials.py --db iris.db --revert   # remove the layer
"""
import argparse
import json
import os
import re
import sqlite3
import sys
import unicodedata

# ----------------------------------------------------------------------------
# Insurer canonicalization
# ----------------------------------------------------------------------------
# CONSERVATIVE key: normalise case, & / and, company-suffix punctuation and
# whitespace ONLY. We deliberately do NOT strip distinguishing words like
# "Life" / "General" / "Health" — merging those would fuse different companies
# (e.g. "Bajaj Allianz Life" vs "Bajaj Allianz General").
_SUFFIX = re.compile(
    r"\b(co\.?|company|ltd\.?|limited|pvt\.?|private|corporation|corp\.?)\b")


def insurer_key(name: str) -> str:
    s = (name or "").strip().lower()
    s = s.replace("&", " and ")
    s = s.replace(".", " ")
    s = _SUFFIX.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def slug(text: str, maxlen: int = 60) -> str:
    s = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()
    return s[:maxlen] or "x"


def canonical_display(variants, pinned=None):
    """Prefer the most complete/long form as the display name (it usually
    carries the full legal suffix), tie-broken alphabetically for stability.

    `pinned` (from the alias file) wins outright. The length heuristic is only a
    guess at which rendering is most complete, and it guesses wrong exactly when
    an alias group exists: "Galaxy Health and Allied Insurance Ltd." is longer
    than "Galaxy Health Insurance Co. Ltd" but is not the name to show.
    """
    if pinned:
        return pinned
    return sorted(variants, key=lambda v: (-len(v), v))[0]


# ----------------------------------------------------------------------------
# Verified same-company aliases
# ----------------------------------------------------------------------------
# insurer_key() merges typographic variants. It cannot merge two genuinely
# different NAMES for one company — that is a real-world fact, not a string
# operation — so those live in tools/insurer_aliases.json and are applied here.
# Kept as data rather than code so a merge is reviewable in a diff, and shared
# with the runtime (iris_brain) so a deploy carries the fix to production, which
# restores its DB from the Litestream replica and never sees a local migration.
_ALIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "insurer_aliases.json")


def load_aliases(path=None):
    """-> (key_remap, display_pin) both keyed on insurer_key().

    key_remap sends every variant's key to the canonical variant's key, so all
    of them group together. display_pin fixes the label for that group.
    Returns empty maps when the file is missing: the alias layer is an override,
    never a prerequisite.
    """
    path = path or _ALIAS_FILE
    try:
        with open(path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except (OSError, ValueError):
        return {}, {}
    key_remap, display_pin = {}, {}
    for group in spec.get("aliases") or []:
        canonical = (group.get("canonical") or "").strip()
        variants = [v for v in (group.get("variants") or []) if (v or "").strip()]
        if not canonical or len(variants) < 2:
            continue
        target = insurer_key(canonical)
        if not target:
            continue
        display_pin[target] = canonical
        for v in variants:
            k = insurer_key(v)
            if k:
                key_remap[k] = target
    return key_remap, display_pin


# ----------------------------------------------------------------------------
# Metric unit parsing
# ----------------------------------------------------------------------------
# Controlled unit vocabulary. Maps a normalised trailing-parenthetical token to
# (unit_code, unit_scale, display_unit). unit_scale converts `value` to base
# units. Anything NOT in here is treated as a qualifier, not a unit (so formula
# codes like (G=C-D-E-F) or (A) stay in the label and unit_code = UNKNOWN).
_UNIT_MAP = {
    "rs crore": ("INR", 1e7, "Rs Crore"),
    "₹crore":   ("INR", 1e7, "Rs Crore"),
    "₹ crore":  ("INR", 1e7, "Rs Crore"),
    "₹lakh":    ("INR", 1e5, "Rs Lakh"),
    "₹ lakh":   ("INR", 1e5, "Rs Lakh"),
    "₹":        ("INR", 1.0, "Rs"),
    "per cent": ("PERCENT", 1.0, "Per cent"),
    "percent":  ("PERCENT", 1.0, "Per cent"),
    "%":        ("PERCENT", 1.0, "Per cent"),
    "nos":      ("COUNT", 1.0, "Nos."),
    "count":    ("COUNT", 1.0, "Nos."),
    "000s":     ("COUNT", 1e3, "'000s"),
    "lakh":     ("NUMBER", 1e5, "Lakh"),
    "lakhs":    ("NUMBER", 1e5, "Lakh"),
    "us $":     ("USD", 1.0, "US $"),
    "us$":      ("USD", 1.0, "US $"),
    "ratio":    ("RATIO", 1.0, "Ratio"),
    "years":    ("YEARS", 1.0, "Years"),
    "year":     ("YEARS", 1.0, "Years"),
}

_TRAIL_PAREN = re.compile(r"\s*\(([^()]*)\)\s*$")
_ALL_PAREN = re.compile(r"\(([^()]*)\)")
# Money words used to disambiguate a bare "(Lakh)" — count (policies/lives) vs
# rupees (e.g. "Gross Premium (Lakh)" is money, "New Policies (Lakhs)" is a count).
_MONEY_WORDS = re.compile(
    r"\b(premium|amount|capital|income|expense|commission|claims? paid|assets?|"
    r"reserves?|profit|fund|deposit|investment|surplus|provision|payout|benefit)\b",
    re.I)


def _norm_unit_token(tok: str) -> str:
    t = tok.strip().lower()
    t = t.replace("'", "").replace("’", "")
    t = t.rstrip(".")
    t = re.sub(r"\s+", " ", t)
    return t


def _resolve_unit(tok: str, raw: str):
    """Map a normalised token to (code, scale, display) or None if not a unit."""
    if tok not in _UNIT_MAP:
        return None
    code, scale, disp = _UNIT_MAP[tok]
    # bare Lakh/Lakhs next to a money word is rupees, not a count.
    if code == "NUMBER" and tok in ("lakh", "lakhs") and _MONEY_WORDS.search(raw):
        return "INR", 1e5, "Rs Lakh"
    return code, scale, disp


def parse_metric(raw: str):
    """Return (clean_label, unit_code, unit_scale, display_unit).
    1) Strip a trailing parenthetical if it is a recognised unit.
    2) Else (trailing paren is a formula code like (A) / (G=C-D-E-F)), scan the
       whole name for an embedded unit such as "...Paid (₹Crore) - ... (A)".
    3) Else report UNKNOWN (genuinely unitless / sub-total)."""
    raw = (raw or "").strip()
    m = _TRAIL_PAREN.search(raw)
    if m:
        u = _resolve_unit(_norm_unit_token(m.group(1)), raw)
        if u:
            return raw[: m.start()].strip(), u[0], u[1], u[2]
    for pm in _ALL_PAREN.finditer(raw):
        u = _resolve_unit(_norm_unit_token(pm.group(1)), raw)
        if u:   # keep full label — stripping a mid-string unit is messy/unsafe
            return raw, u[0], u[1], u[2]
    return raw, "UNKNOWN", 1.0, ""


# ----------------------------------------------------------------------------
# Year normalization
# ----------------------------------------------------------------------------
def parse_year(raw: str):
    """Return (fy_canonical, period_type, start_year).
    "2024-25" -> ("2024-25", FISCAL, 2024); "2024" -> ("2024", CALENDAR, 2024)."""
    raw = (raw or "").strip()
    mfy = re.match(r"^(\d{4})\s*[-/]\s*(\d{2,4})$", raw)
    if mfy:
        start = int(mfy.group(1))
        return f"{start}-{str(start + 1)[-2:]}", "FISCAL", start
    mcal = re.match(r"^(\d{4})$", raw)
    if mcal:
        return raw, "CALENDAR", int(raw)
    return raw, "UNKNOWN", None


# ----------------------------------------------------------------------------
# Apply / revert
# ----------------------------------------------------------------------------
ADDED_COLS = [
    ("insurer_id", "TEXT"),
    ("metric_id", "TEXT"),
    ("unit_code", "TEXT"),
    ("unit_scale", "REAL"),
    ("value_base", "REAL"),     # value in base units (value * unit_scale)
    ("fy_canonical", "TEXT"),
    ("period_type", "TEXT"),
]
DIM_TABLES = ["insurer_dim", "metric_dim", "year_dim"]


def _cols(conn, table):
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def apply(conn):
    cur = conn.cursor()
    existing = _cols(conn, "financial_metrics")
    for col, typ in ADDED_COLS:
        if col not in existing:
            cur.execute(f"ALTER TABLE financial_metrics ADD COLUMN {col} {typ}")

    # --- Build insurer dimension -------------------------------------------
    insurers = [r[0] for r in cur.execute(
        "SELECT DISTINCT insurer FROM financial_metrics WHERE insurer IS NOT NULL AND insurer<>''")]
    key_remap, display_pin = load_aliases()
    groups = {}
    for name in insurers:
        k = insurer_key(name)
        groups.setdefault(key_remap.get(k, k), []).append(name)
    insurer_id_of = {}          # raw name -> insurer_id
    cur.execute("""CREATE TABLE IF NOT EXISTS insurer_dim (
        insurer_id TEXT PRIMARY KEY, canonical_name TEXT, variants INTEGER)""")
    cur.execute("DELETE FROM insurer_dim")
    used = set()
    for key, variants in groups.items():
        disp = canonical_display(variants, display_pin.get(key))
        iid = slug(disp)
        while iid in used:
            iid += "_x"
        used.add(iid)
        cur.execute("INSERT INTO insurer_dim VALUES (?,?,?)", (iid, disp, len(variants)))
        for v in variants:
            insurer_id_of[v] = iid

    # --- Build metric dimension --------------------------------------------
    metrics = [r[0] for r in cur.execute(
        "SELECT DISTINCT metric FROM financial_metrics WHERE metric IS NOT NULL AND metric<>''")]
    metric_id_of = {}           # raw metric -> (metric_id, unit_code, unit_scale)
    cur.execute("""CREATE TABLE IF NOT EXISTS metric_dim (
        metric_id TEXT PRIMARY KEY, clean_label TEXT, unit_code TEXT,
        unit_scale REAL, display_unit TEXT, raw_count INTEGER)""")
    cur.execute("DELETE FROM metric_dim")
    dim = {}                    # metric_id -> [clean_label, unit_code, scale, disp, count]
    used_m = set()
    for raw in metrics:
        label, code, scale, disp = parse_metric(raw)
        base = slug(label)
        mid = base if code in ("UNKNOWN",) else f"{base}__{code.lower()}"
        # stable: same (label,unit) always lands on same id
        if mid not in dim:
            while mid in used_m and dim.get(mid, [label, code])[:2] != [label, code]:
                mid += "_x"
            used_m.add(mid)
            dim[mid] = [label, code, scale, disp, 0]
        dim[mid][4] += 1
        metric_id_of[raw] = (mid, code, scale)
    for mid, (label, code, scale, disp, cnt) in dim.items():
        cur.execute("INSERT INTO metric_dim VALUES (?,?,?,?,?,?)",
                    (mid, label, code, scale, disp, cnt))

    # --- Build year dimension ----------------------------------------------
    years = [r[0] for r in cur.execute(
        "SELECT DISTINCT financial_year FROM financial_metrics WHERE financial_year IS NOT NULL")]
    year_of = {}
    cur.execute("""CREATE TABLE IF NOT EXISTS year_dim (
        raw TEXT PRIMARY KEY, fy_canonical TEXT, period_type TEXT, start_year INTEGER)""")
    cur.execute("DELETE FROM year_dim")
    for raw in years:
        fy, ptype, start = parse_year(raw)
        year_of[raw] = (fy, ptype)
        cur.execute("INSERT INTO year_dim VALUES (?,?,?,?)", (raw, fy, ptype, start))

    # --- Populate the canonical columns on financial_metrics ---------------
    rows = cur.execute(
        "SELECT id, insurer, metric, financial_year, value FROM financial_metrics").fetchall()
    updates = []
    for rid, ins, met, yr, val in rows:
        iid = insurer_id_of.get(ins)
        mid, code, scale = metric_id_of.get(met, (None, None, None))
        fy, ptype = year_of.get(yr, (yr, None))
        vbase = (val * scale) if (val is not None and scale is not None) else None
        updates.append((iid, mid, code, scale, vbase, fy, ptype, rid))
    cur.executemany(
        """UPDATE financial_metrics SET insurer_id=?, metric_id=?, unit_code=?,
           unit_scale=?, value_base=?, fy_canonical=?, period_type=? WHERE id=?""", updates)

    # Indexes for the canonical query paths.
    for col in ("insurer_id", "metric_id", "fy_canonical"):
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_fm_{col} ON financial_metrics({col})")
    conn.commit()
    return {
        "insurer_groups": len(groups),
        "insurer_merges": sum(1 for v in groups.values() if len(v) > 1),
        "metrics_raw": len(metrics),
        "metrics_canonical": len(dim),
        "metrics_unit_unknown": sum(1 for v in dim.values() if v[1] == "UNKNOWN"),
        "years_raw": len(years),
        "rows_updated": len(updates),
    }


def revert(conn):
    cur = conn.cursor()
    for t in DIM_TABLES:
        cur.execute(f"DROP TABLE IF EXISTS {t}")
    # Drop indexes first — SQLite refuses to DROP an indexed column.
    for col in ("insurer_id", "metric_id", "fy_canonical"):
        cur.execute(f"DROP INDEX IF EXISTS idx_fm_{col}")
    # SQLite >= 3.35 supports DROP COLUMN; guard for older versions.
    existing = _cols(conn, "financial_metrics")
    for col, _ in ADDED_COLS:
        if col in existing:
            try:
                cur.execute(f"ALTER TABLE financial_metrics DROP COLUMN {col}")
            except sqlite3.OperationalError:
                cur.execute(f"UPDATE financial_metrics SET {col}=NULL")
    conn.commit()


_SPLIT_NOISE = {"ltd", "limited", "co", "company", "india", "the", "assurance",
                "pvt", "private", "corporation", "of", "plc", "se", "insurance",
                "and", "allied"}


def _report_suspected_splits(conn):
    """Flag one company that may have landed under two insurer_ids.

    insurer_key() cannot see that "Galaxy Health and Allied Insurance Ltd." and
    "Galaxy Health Insurance Co. Ltd" are the same company, and that split is
    almost invisible downstream — it just makes a profile look thin and a market
    look one insurer more crowded than it is. This surfaces the candidates at
    ingest, when a new name first appears, instead of when someone notices two
    entries in a picker.

    Reported, never merged: only the alias file merges, and only after a human has
    confirmed it. Entries already covered there are excluded, so a resolved case
    stops nagging.

    Heuristic: same licence class, and identical once group/suffix words and the
    class words are stripped. Different classes are left alone — Bajaj Allianz
    Life and Bajaj Allianz General really are two companies.
    """
    q = conn.execute
    if not q("SELECT 1 FROM sqlite_master WHERE type='table' AND name='insurer_dim'").fetchone():
        return
    key_remap, _pin = load_aliases()
    rows = q("""SELECT d.insurer_id, d.canonical_name,
                       (SELECT COUNT(*) FROM financial_metrics f WHERE f.insurer_id=d.insurer_id)
                FROM insurer_dim d""").fetchall()

    def dkey(name):
        s = re.sub(r"[#^*$~]+", "", name or "")
        toks = [t for t in re.split(r"[^a-z0-9]+", s.lower())
                if t and t not in _SPLIT_NOISE]
        return tuple(sorted(toks))

    groups = {}
    for iid, name, n in rows:
        cls = [r[0] for r in q("""SELECT DISTINCT insurer_class FROM financial_metrics
                                  WHERE insurer_id=? AND insurer_class IS NOT NULL""", (iid,))]
        groups.setdefault((dkey(name), tuple(sorted(cls))), []).append((iid, name, n))

    flagged = []
    for (k, cls), members in sorted(groups.items()):
        if len(members) < 2 or not k:
            continue
        # Already declared as one company in the alias file? Then it is resolved.
        if all(insurer_key(nm) in key_remap for _i, nm, _n in members):
            continue
        flagged.append((k, cls, members))

    if not flagged:
        print("   no suspected same-company splits")
        return
    print(f"   ⚠ {len(flagged)} suspected same-company split(s) "
          f"— confirm, then add to tools/insurer_aliases.json:")
    for k, cls, members in flagged:
        print(f"      [{' '.join(k)}] class={'/'.join(cls) or '?'}")
        for iid, name, n in members:
            print(f"          {n:>7} rows  {name}  [{iid}]")


def report(conn):
    q = conn.execute
    print("\n--- canonicalization report ---")
    print("insurers: %d raw names -> %d canonical (%d merged groups)" % (
        q("SELECT COUNT(DISTINCT insurer) FROM financial_metrics").fetchone()[0],
        q("SELECT COUNT(*) FROM insurer_dim").fetchone()[0],
        q("SELECT COUNT(*) FROM insurer_dim WHERE variants>1").fetchone()[0]))
    for iid, name, n in q("SELECT * FROM insurer_dim WHERE variants>1 ORDER BY variants DESC"):
        print(f"   merged x{n}: {name}  [{iid}]")
    _report_suspected_splits(conn)
    print("metrics: %d raw -> %d canonical, %d still unit=UNKNOWN" % (
        q("SELECT COUNT(DISTINCT metric) FROM financial_metrics").fetchone()[0],
        q("SELECT COUNT(*) FROM metric_dim").fetchone()[0],
        q("SELECT COUNT(*) FROM metric_dim WHERE unit_code='UNKNOWN'").fetchone()[0]))
    print("   unit spread:", dict(q(
        "SELECT unit_code, COUNT(*) FROM metric_dim GROUP BY unit_code ORDER BY 2 DESC")))
    print("years:", dict(q(
        "SELECT period_type, COUNT(*) FROM year_dim GROUP BY period_type")))
    nulls = q("SELECT SUM(insurer_id IS NULL AND insurer IS NOT NULL), "
              "SUM(metric_id IS NULL AND metric IS NOT NULL) FROM financial_metrics").fetchone()
    print(f"unmapped rows: insurer_id={nulls[0] or 0}, metric_id={nulls[1] or 0}")
    # Proof: Star Health no longer splits.
    star = q("""SELECT canonical_name, COUNT(DISTINCT insurer)
                FROM financial_metrics JOIN insurer_dim USING(insurer_id)
                WHERE canonical_name LIKE 'Star Health%' GROUP BY 1""").fetchall()
    if star:
        print("proof — Star Health raw variants now under one id:", star)


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
            print("reverted canonicalization layer from", args.db)
            return
        stats = apply(conn)
        print("applied to %s:" % args.db, stats)
        if args.report:
            report(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
