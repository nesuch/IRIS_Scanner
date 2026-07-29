"""Entity + metric classification for financial_metrics (handbook data).

WHY: `insurer_id` is polymorphic — it holds insurers, states, distribution
channels, countries, ombudsman centres AND sector aggregates. Nothing marks which
is which, so a query like "sum this metric across insurers" silently mixes a
state, a channel and an industry total into the same number. That is how the
naive market-share calculation produced "General Insurance Sector 36.3%,
Maharashtra 4.7%".

Separately, metrics are NOT comparable across insurer classes: of the metrics
reported at insurer level, only a small core is reported by ALL classes (Life,
General, SAHI, Reinsurer, FRB) — the rest are class-specific (persistency is a
life concept, combined ratio a general one). Ranking every insurer on one of
those produces a confident, meaningless league table.

This adds three derived columns, all COMPUTED and reviewable (not opinion):

  entity_type       insurer | state | channel | country | ombudsman_centre
                    | tpa | aggregate
  insurer_class     Life | General | SAHI | Reinsurer | FRB  (NULL unless insurer)
  metric_scope      universal | broad | class_specific | non_insurer
                    (on a companion table, keyed by metric_id)

Dry-run by default: prints the full classification and changes nothing.
`--apply` performs the migration inside a transaction.

    python tools/classify_entities.py <path-to-COPY-of-iris.db>
    python tools/classify_entities.py <db> --apply
"""
import os
import re
import sqlite3
import sys
from collections import defaultdict

# --- entity typing -----------------------------------------------------------
# The handbook's own `dimension` already separates most entity kinds; it just was
# never surfaced as a property of the row.
DIM_TO_TYPE = {
    "State": "state",
    "Channel": "channel",
    "Country": "country",
    "Ombudsman": "ombudsman_centre",
    "TPA": "tpa",
    "Industry": "aggregate",
}

# Aggregate rows that live INSIDE the insurer-bearing dimensions. These are the
# dangerous ones: they look like entities to any query that filters on dimension
# alone. For some metrics (e.g. pension premium) the sector row is the ONLY row,
# so an insurer view shows blank while an industry total silently reads it.
_AGG_EXACT = {"total", "grand_total", "all_india", "industry_total"}
_AGG_SUFFIX = ("_sector", "_industry", "_industry_all", "_total")
# Ownership/segment subtotals that sit in dim='Insurer' looking exactly like
# companies ("Private Sector Insurers Total" = 460 rows). Left unflagged they
# enter league tables as if they were firms.
_AGG_CONTAINS = ("private_sector_insurers", "public_sector_insurers",
                 "stand_alone_health_insurers", "standalone_health_insurers")


def entity_type(dimension, insurer_id):
    # Aggregate detection runs FIRST, before the dimension mapping. Every typed
    # dimension carries its own subtotal row — dimension='Channel' holds a "Total"
    # alongside Brokers and Individual Agents — and mapping by dimension alone made
    # that Total a channel, so a channel-mix chart showed "Total: 50%" as if it were
    # a distribution channel.
    iid = (insurer_id or "").lower()
    if (iid in _AGG_EXACT or iid.endswith(_AGG_SUFFIX)
            or any(a in iid for a in _AGG_CONTAINS)):
        return "aggregate"
    return DIM_TO_TYPE.get(dimension, "insurer")


# --- insurer class -----------------------------------------------------------
# Order matters: reinsurer before health/life (a reinsurer may carry either word),
# health before life ("... Health Insurance" is a SAHI, not a life company).
#
# Hand-verified exceptions — public-sector general insurers and specialists whose
# names carry no class word at all. Without these six they fall to the default and
# would still land on General, but they are pinned explicitly so the intent is
# recorded rather than accidental.
CLASS_OVERRIDES = {
    "national_insurance_co_ltd": "General",
    "the_new_india_assurance_co_ltd": "General",
    "the_oriental_insurance_co_ltd": "General",
    "united_india_insurance_co_ltd": "General",
    "ecgc_ltd": "General",          # export credit — a specialised general insurer
}

# Reinsurance entities are identified from DATA, not names: only they file
# `equity_share_capital_assigned_capital_of_branches_of_foreign`. That marker is
# carried by Indian reinsurers AND foreign branches alike, so it separates the
# reinsurance sector from Life/General/SAHI but not FRB from Indian reinsurer.
REINSURER_MARKER = "%assigned_capital_of_branches_of_foreign%"

# India has exactly two domestically-incorporated reinsurers; everything else
# holding the marker is a branch/syndicate of a foreign reinsurer (FRB). This is a
# regulatory fact, so an explicit list is more honest than a name heuristic.
INDIAN_REINSURERS = {
    "general_insurance_corporation_of_india_gic_re",
    "iti_reinsurance_ltd",
}


def insurer_class(insurer_id, insurer_name="", reinsurance_ids=frozenset()):
    iid = (insurer_id or "").lower()
    if iid in reinsurance_ids:
        return "Reinsurer" if iid in INDIAN_REINSURERS else "FRB"
    if iid in CLASS_OVERRIDES:
        return CLASS_OVERRIDES[iid]
    name = (insurer_name or "").lower()
    # Lloyd's syndicates and service companies sit under the Lloyd's of India
    # platform — foreign-branch business even when the name carries no class word.
    if "lloyd" in iid or "lloyd" in name:
        return "FRB"
    if "reinsur" in iid or re.search(r"(^|_)re(_|$)", iid) or " re " in f" {name} ":
        return "Reinsurer"
    if "health" in iid:
        return "SAHI"
    if "life" in iid:
        return "Life"
    return "General"


# --- metric scope ------------------------------------------------------------
# Derived from OBSERVED coverage, not judgement: a metric is universal when every
# insurer class actually reports it (with at least MIN_INSURERS distinct insurers,
# so one stray filing can't promote a class-specific metric).
MIN_INSURERS = 3
ALL_CLASSES = ("Life", "General", "SAHI", "Reinsurer", "FRB")


def classify(db):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT DISTINCT dimension, insurer_id, insurer FROM financial_metrics"
    ).fetchall()

    # Data-driven: who files the foreign-branch assigned-capital line?
    reins = {r[0] for r in con.execute(
        "SELECT DISTINCT insurer_id FROM financial_metrics WHERE metric_id LIKE ?",
        (REINSURER_MARKER,)) if r[0]}

    ent = {}
    for r in rows:
        et = entity_type(r["dimension"], r["insurer_id"])
        key = (r["dimension"], r["insurer_id"])
        ent[key] = {
            "entity_type": et,
            "insurer_class": insurer_class(r["insurer_id"], r["insurer"], reins) if et == "insurer" else None,
            "name": r["insurer"],
        }

    # metric -> which classes report it, and with how many distinct insurers
    ins_class = {}
    for (dim, iid), v in ent.items():
        if v["entity_type"] == "insurer":
            ins_class[iid] = v["insurer_class"]

    cover = defaultdict(lambda: defaultdict(set))
    for mid, iid in con.execute(
        "SELECT metric_id, insurer_id FROM financial_metrics WHERE metric_id IS NOT NULL"
    ):
        c = ins_class.get(iid)
        if c:
            cover[mid][c].add(iid)

    # Threshold must ADAPT to class size. India has only two domestic reinsurers,
    # so a flat ">=3 insurers" would make it impossible for any metric to qualify
    # for the Reinsurer class — and therefore impossible for anything to be
    # universal. Require 3, or the whole class where the class is smaller.
    class_size = defaultdict(set)
    for iid, c in ins_class.items():
        class_size[c].add(iid)
    threshold = {c: max(1, min(MIN_INSURERS, len(ids))) for c, ids in class_size.items()}

    scope = {}
    for mid in {r[0] for r in con.execute(
            "SELECT DISTINCT metric_id FROM financial_metrics WHERE metric_id IS NOT NULL")}:
        classes = [c for c in ALL_CLASSES
                   if len(cover[mid].get(c, ())) >= threshold.get(c, MIN_INSURERS)]
        if not classes:
            scope[mid] = ("non_insurer", [])
        elif len(classes) == len(ALL_CLASSES):
            scope[mid] = ("universal", classes)
        elif len(classes) >= 3:
            scope[mid] = ("broad", classes)
        else:
            scope[mid] = ("class_specific", classes)
    con.close()
    return ent, scope


def apply_migration(db, ent, scope):
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(financial_metrics)")}
    try:
        con.execute("BEGIN")
        if "entity_type" not in cols:
            con.execute("ALTER TABLE financial_metrics ADD COLUMN entity_type TEXT")
        if "insurer_class" not in cols:
            con.execute("ALTER TABLE financial_metrics ADD COLUMN insurer_class TEXT")
        for (dim, iid), v in ent.items():
            con.execute(
                "UPDATE financial_metrics SET entity_type=?, insurer_class=? "
                "WHERE dimension=? AND insurer_id IS ?",
                (v["entity_type"], v["insurer_class"], dim, iid),
            )
        # Metric scope lives in its own table: it is a property of the METRIC, not
        # of each of the 253k rows, and keeping it separate means recomputing it
        # later (as new years arrive) touches one small table.
        con.execute("""CREATE TABLE IF NOT EXISTS metric_scope (
            metric_id TEXT PRIMARY KEY, scope TEXT NOT NULL, classes TEXT)""")
        con.execute("DELETE FROM metric_scope")
        con.executemany("INSERT INTO metric_scope (metric_id,scope,classes) VALUES (?,?,?)",
                        [(m, s, ",".join(c)) for m, (s, c) in scope.items()])
        con.execute("CREATE INDEX IF NOT EXISTS idx_fm_entity_type ON financial_metrics(entity_type)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_fm_insurer_class ON financial_metrics(insurer_class)")
        con.commit()
        print("  migration applied.")
    except Exception as e:
        con.rollback()
        print(f"  ROLLED BACK: {e}")
        raise
    finally:
        con.close()


def main():
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        sys.exit("usage: classify_entities.py <path-to-COPY-of-iris.db> [--apply]")
    db = sys.argv[1]
    apply_it = "--apply" in sys.argv
    ent, scope = classify(db)

    types = defaultdict(int)
    for v in ent.values():
        types[v["entity_type"]] += 1
    print("\n=== ENTITY TYPES (distinct dimension+id pairs) ===")
    for t, n in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:18s} {n:5d}")

    print("\n=== INSURER CLASSIFICATION ===")
    by_class = defaultdict(list)
    for v in ent.values():
        if v["entity_type"] == "insurer":
            by_class[v["insurer_class"]].append(v["name"])
    for c in ALL_CLASSES:
        names = sorted(set(by_class.get(c, [])))
        print(f"\n  --- {c} ({len(names)}) ---")
        for n in names:
            mark = "  *" if any(k in (n or "").lower() for k in
                                ("national insurance", "new india", "oriental", "united india",
                                 "ecgc", "markel")) else "   "
            print(f"  {mark} {n}")

    print("\n=== AGGREGATE ROWS hiding in insurer dimensions ===")
    aggs = sorted({v["name"] for k, v in ent.items()
                   if v["entity_type"] == "aggregate" and k[0] in ("Insurer", "Financials")})
    for a in aggs:
        print(f"     {a}")
    if not aggs:
        print("     (none)")

    sc = defaultdict(int)
    for s, _ in scope.values():
        sc[s] += 1
    print("\n=== METRIC SCOPE ===")
    for s in ("universal", "broad", "class_specific", "non_insurer"):
        print(f"  {s:16s} {sc.get(s,0):5d}")

    print("\n  * = hand-pinned override (name carries no class word)")
    if apply_it:
        print("\n=== APPLYING ===")
        apply_migration(db, ent, scope)
    else:
        print("\n  DRY RUN — nothing written. Re-run with --apply to migrate.")


if __name__ == "__main__":
    main()
