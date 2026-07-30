#!/usr/bin/env python3
"""
Reconcile the IRDAI handbook, cell by cell, against what IRIS actually holds.

WHY THIS AND NOT A DIFF AGAINST OUR OWN EXTRACT
    handbook_2024-25_parts.xlsx is already the OUTPUT of extraction. Comparing the
    database against it proves only that the load was faithful; it cannot see a value
    the extraction dropped. That is exactly how the incurred-claims-ratio gap survived
    - Table 62's Total column prints a ratio for Galaxy and Narayana, the extract took
    the four class columns and not that one, and nothing downstream could tell.
    So this reads the ORIGINAL workbooks.

HOW A HANDBOOK TABLE IS SHAPED
    A stack of header rows, then one row per insurer:

        row  1   TABLE 62: HEALTH INSURANCE ...
        row  2   (Amount in Rs Lakh)                 <- the unit, and it varies per table
        row  3   S.No | Insurers | 2015-16 ...       <- year, merged across its block
        row  4                   | Govt Sponsored... <- class of business, merged
        row  5                   | Net Earned Prem.. <- the metric
        row  6+  1 | National Insurance Co. Ltd. | numbers...

    Merged cells read as None in every position but the first, so each header row is
    forward-filled to recover the full path of a column: (year, class, metric).

WHAT A FINDING MEANS
    Matching is done on VALUE, not on label. Our metric ids and context labels are
    canonicalised and will never string-match the handbook's headers, but a number is
    a number. For each handbook cell we ask whether IRIS holds that value for that
    insurer in that year, then report what it holds it under:

      ok              present, and the metric IRIS filed it under agrees with the header
      value_mismatch  IRIS holds THIS metric in THIS context for THIS insurer and year,
                      but with a different number. The most serious class: the figure is
                      not absent, it is wrong, and nothing downstream can tell.
      relocated       the value is present for that insurer and year, but under a
                      different metric or context - worth a human look
      missing         the value is nowhere for that insurer and year

    "relocated" is deliberately not called an error. The same number legitimately
    appears in several places, and our canonical labels differ from the handbook's
    wording by design.

USAGE
    python tools/audit_handbook.py --part "Part III" --table 62
    python tools/audit_handbook.py --all --out audit.csv

    Read-only. Touches no database and writes nothing but the report.
"""
import argparse
import csv
import glob
import os
import re
import sys

HANDBOOK_DIR = ("/Users/sudeepchandranemalikanti/Desktop/Parliament Questions/"
                "Publication of Handbook 2024-25 on IRDAI Website")

# A handbook amount is scaled; IRIS stores base rupees. Which scale applies is stated
# in the table's own subtitle, so it is read per table rather than assumed.
# The word "in" is optional. Table 62 writes "(Amount in ₹Lakh)" and Table 50 just
# "(₹Crore)" — requiring "in" silently left every Crore table unscaled, which made
# correct values look absent and produced a 57% "missing" rate that was entirely mine.
_SCALE_HINTS = (
    (re.compile(r"\bcrore", re.I), 1e7),
    (re.compile(r"\blakh", re.I), 1e5),
    (re.compile(r"\bthousand|'000", re.I), 1e3),
)
_YEAR_RE = re.compile(r"^(19|20)\d{2}\s*-\s*\d{2,4}$")
_PCT_RE = re.compile(r"ratio|per\s*cent|%|share", re.I)
# IRIS deliberately stores leaf items only; totals were removed and are recomputed from
# tools/total_definitions.json. So a handbook Total column is EXPECTED to be absent, and
# counting it as a loss overstates the gap badly.
#
# It is also the strongest test available. The handbook prints the whole, we hold the
# parts, and neither was derived from the other - so if our leaves sum to their total,
# the leaves are right. That checks far more than a label comparison can.
_TOTAL_RE = re.compile(r"^\s*(grand\s+)?total\b|\ball\s+classes\b|\btotal\s*$", re.I)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def ffill(row):
    """Forward-fill a header row so merged cells carry their label across the block."""
    out, last = [], ""
    for c in row:
        v = "" if c is None else str(c).strip()
        if v:
            last = v
        out.append(last)
    return out


# The unit is often stated per COLUMN rather than per table - "Benefit Amount
# (₹crore)" sits beside "No. of Persons Covered ('000s)" in the same sheet. Reading
# only the table subtitle made every crore column look 1e7 wrong when IRIS was right.
_COL_UNITS = (
    (re.compile(r"\bcrore\b", re.I), 1e7),
    (re.compile(r"\blakh\b", re.I), 1e5),
    (re.compile(r"'?000s?\b|\bthousand\b", re.I), 1e3),
)


def column_scale(metric, ctx, default):
    """Scale for one column, preferring a unit named in its own header."""
    text = f"{metric} {ctx}"
    for rx, mult in _COL_UNITS:
        if rx.search(text):
            return mult
    return default


def table_scale(rows):
    """Multiplier turning a printed amount into base rupees, from the table subtitle."""
    head = " ".join(str(c) for r in rows[:4] for c in r if c)
    for rx, mult in _SCALE_HINTS:
        if rx.search(head):
            return mult
    return 1.0


def _numeric_count(row, skip=2):
    return sum(1 for c in row[skip:] if isinstance(c, (int, float))
               and not isinstance(c, bool))


def find_header_block(rows):
    """(first data row, header rows).

    The header ends where DATA begins, not at the first insurer row. Those differ:
    Table 62 puts a "Public Sector Insurers" section heading between the headers and
    the first insurer, and taking the row above the insurer as the metric level made
    every metric read "Public Sector Insurers". So the first data row is the first row
    carrying several numbers, and header rows are the label-only rows above it - with
    section headings (a lone label, no numbers) dropped, since they describe the rows
    beneath rather than the columns.
    """
    first = None
    for i, r in enumerate(rows):
        if _numeric_count(r) >= 3:
            first = i
            break
    if first is None:
        return None, []
    hdr = []
    for r in rows[max(0, first - 6):first]:
        labels = [c for c in r if isinstance(c, str) and c.strip()]
        if not labels:
            continue
        # A section heading occupies one cell and governs no column.
        if len(labels) == 1 and _numeric_count(r) == 0:
            continue
        hdr.append(r)
    return first, hdr


def parse_sheet_transposed(rows, is_insurer, first, filled, width):
    """The other layout: insurers ACROSS the top, metrics down the side.

    Every account statement is shaped this way - Policyholders, Shareholders, Balance
    Sheet, for both life and general - so a parser that only understands insurer-per-row
    silently skipped six of the largest tables in the handbook. Here the row label is the
    metric ("Premiums earned (Net)"), and a column resolves to (insurer, year, line of
    business) through the header stack.
    """
    def at(level, ci):
        if level is None or level < 0 or level >= len(filled):
            return ""
        row = filled[level]
        return row[ci] if ci < len(row) else ""

    ins_lvl = max(range(len(filled)),
                  key=lambda l: sum(1 for v in filled[l] if v and is_insurer(v)))
    year_lvl = next((l for l in range(len(filled))
                     if any(_YEAR_RE.match(str(v).strip()) for v in filled[l])), None)

    for r in rows[first:]:
        metric = next((c for c in r[:2] if isinstance(c, str) and c.strip()), None)
        if not metric:
            continue
        for ci in range(width):
            v = r[ci] if ci < len(r) else None
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            label = at(ins_lvl, ci)
            if not label or not is_insurer(label):
                continue
            year = at(year_lvl, ci) if year_lvl is not None else ""
            ctx = " · ".join(dict.fromkeys(
                x for lvl in range(len(filled))
                if lvl not in (ins_lvl, year_lvl) and (x := at(lvl, ci))))
            yield label, str(year).strip(), ctx, str(metric).strip(), float(v)


def parse_sheet(rows, is_insurer):
    """Yield (insurer_label, year, context, metric, value) for every numeric cell."""
    first, hdr = find_header_block(rows)
    if first is None:
        return
    filled = [ffill(h) for h in hdr]
    if not filled:
        # No header rows survived, so no column can be attributed to a metric. Auditing
        # such a sheet would compare numbers against labels we do not have.
        return
    width = max((len(r) for r in rows), default=0)

    def at(level, ci):
        if level is None or level < 0 or level >= len(filled):
            return ""
        row = filled[level]
        return row[ci] if ci < len(row) else ""

    # Which way round is the table? If insurer names sit in the HEADER rather than the
    # label column, it is an account-statement layout and needs the transposed reader.
    hdr_names = sum(1 for lv in filled for v in lv if v and is_insurer(v))
    lbl_names = sum(1 for r in rows[first:]
                    for c in r[:4] if isinstance(c, str) and is_insurer(c))
    if hdr_names > lbl_names:
        yield from parse_sheet_transposed(rows, is_insurer, first, filled, width)
        return

    # Whichever header row holds year strings is the year level; the row closest to the
    # data is the metric; anything between is context.
    year_lvl = None
    for lvl in range(len(filled)):
        if any(_YEAR_RE.match(str(v).strip()) for v in filled[lvl]):
            year_lvl = lvl
            break
    metric_lvl = len(filled) - 1

    for r in rows[first:]:
        label = next((c for c in r[:4] if isinstance(c, str) and is_insurer(c)), None)
        if not label:
            continue
        for ci in range(width):
            v = r[ci] if ci < len(r) else None
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            # The S.No column is numeric but is not data; so is anything whose column
            # carries no metric label at all.
            deepest = at(metric_lvl, ci)
            if not deepest or norm(deepest) in ("s no", "sno", "insurers", "insurer"):
                continue
            year = at(year_lvl, ci) if year_lvl is not None else ""
            metric = at(metric_lvl, ci)
            ctx = " · ".join(dict.fromkeys(
                x for lvl in range(len(filled))
                if lvl not in (year_lvl, metric_lvl) and (x := at(lvl, ci))))
            yield label, str(year).strip(), ctx, str(metric).strip(), float(v)


def toks(s):
    return frozenset(t for t in norm(s).split() if len(t) > 2)


def build_iris_index(brain):
    """(insurer_id, fy) -> list of (value_base, metric_id, lob, cob), plus a name map."""
    df = brain.UNIFIED_DF
    ins = df[(df["entity_type"] == "insurer") & df["value_base"].notna()]
    idx = {}
    for iid, fy, val, mid, lob, cob in zip(
            ins["insurer_id"], ins["fy_canonical"].fillna(""), ins["value_base"],
            ins["metric_id"].fillna(""), ins["Line_of_Business"].fillna(""),
            ins["Class_of_Business"].fillna("")):
        idx.setdefault((str(iid), str(fy)), []).append(
            (float(val), str(mid), str(lob), str(cob)))
    names = {}
    for iid, nm in ins[["insurer_id", "Entity"]].drop_duplicates("insurer_id").values:
        names[norm(nm)] = str(iid)
    return idx, names


def resolve_insurer(label, names):
    k = norm(label)
    if k in names:
        return names[k]
    # The handbook carries footnote marks and small wording differences; fall back to
    # the longest name that is a prefix of the label or vice versa.
    best, blen = None, 0
    for nk, iid in names.items():
        if len(nk) < 8:
            continue
        if (k.startswith(nk[:20]) or nk.startswith(k[:20])) and len(nk) > blen:
            best, blen = iid, len(nk)
    return best


def audit_sheet(rows, iris_idx, names, sheet, part, tol=0.02):
    scale = table_scale(rows)
    is_insurer = lambda s: resolve_insurer(s, names) is not None and len(s) > 8
    out = []
    for label, year, ctx, metric, val in parse_sheet(rows, is_insurer):
        iid = resolve_insurer(label, names)
        if iid is None or not year:
            continue
        pool = iris_idx.get((iid, year), [])
        # A percentage is stored as filed; an amount is scaled to base rupees. Try both
        # rather than trusting the header, since a "Ratio" column can print 0.61 or 61.
        col_scale = column_scale(metric, ctx, scale)
        is_total = bool(_TOTAL_RE.search(metric) or _TOTAL_RE.search(ctx))
        cands = [val] if _PCT_RE.search(metric) else [val * col_scale, val]
        hit = None
        for want in cands:
            if want == 0:
                continue
            for v, mid, lob, cob in pool:
                if abs(v - want) <= abs(want) * tol:
                    hit = (mid, lob, cob)
                    break
            if hit:
                break
        if hit is None:
            # No value matched. Before calling it missing, ask whether IRIS holds this
            # very metric in this very context — if it does, the number is WRONG rather
            # than absent, which is the more serious finding and the one a
            # missing-cells-only audit would never surface.
            # Only claim a mismatch when the slot is UNAMBIGUOUS. Taking the first
            # token-overlapping row compared a handbook cell against an arbitrary
            # neighbour - three different Benefit Amount cells all "mismatched" the same
            # IRIS row. If several rows fit, we cannot say which column this cell is,
            # so we say nothing.
            mt, ct = toks(metric), toks(ctx)
            slots = []
            for v, mid2, lob2, cob2 in pool:
                if not mt or not (mt & toks(mid2)):
                    continue
                if ct and not (ct & toks(f"{lob2} {cob2}")):
                    continue
                slots.append((v, mid2, lob2, cob2))
                if len(slots) > 1:
                    break
            slot = slots[0] if len(slots) == 1 else None

            # A total we chose not to store: rebuild it from the leaves and see whether
            # it agrees. This is the reconciliation that actually proves the data.
            if slot is None and is_total and not _PCT_RE.search(metric):
                mt = toks(metric)
                # Sum ONE metric, not everything sharing a word. "Premiums earned (Net)"
                # overlaps both premiums_earned_net__inr and net_earned_premium__inr, and
                # adding both double-counted the total. Pick the single best-matching
                # metric_id, then sum only its non-total contexts.
                best, best_score = None, 0
                for _v, mid2, _l, _c in pool:
                    sc = len(mt & toks(mid2))
                    if sc > best_score:
                        best, best_score = mid2, sc
                parts_sum = sum(v for v, mid2, lob2, cob2 in pool
                                if best and mid2 == best
                                and not _TOTAL_RE.search(f"{lob2} {cob2}")) if best else 0
                want = val * col_scale
                if parts_sum and want:
                    agree = abs(parts_sum - want) <= abs(want) * 0.02
                    out.append({"part": part, "table": sheet, "insurer": label[:44],
                                "year": year, "hb_context": ctx[:60],
                                "hb_metric": metric[:44], "hb_value": val,
                                "status": "total_ok" if agree else "total_mismatch",
                                "iris_metric": "(sum of leaves)", "iris_lob": "",
                                "iris_cob": "", "iris_value": parts_sum,
                                "ratio": parts_sum / want})
                    continue

            if slot is not None:
                v, mid, lob, cob = slot
                out.append({"part": part, "table": sheet, "insurer": label[:44],
                            "year": year, "hb_context": ctx[:60], "hb_metric": metric[:44],
                            "hb_value": val, "status": "value_mismatch",
                            "iris_metric": mid, "iris_lob": lob, "iris_cob": cob,
                            "iris_value": v,
                            "ratio": (v / (val * (1 if _PCT_RE.search(metric) else col_scale))
                                      if val else "")})
                continue
            status = "missing" if val != 0 else "missing_zero"
            mid = lob = cob = ""
        else:
            mid, lob, cob = hit
            hm, hn = norm(metric), norm(mid)
            agree = bool(hm) and (hm[:12] in hn or hn[:12] in hm
                                  or norm(ctx)[:12] in norm(f"{lob} {cob}"))
            status = "ok" if agree else "relocated"
        out.append({"part": part, "table": sheet, "insurer": label[:44], "year": year,
                    "hb_context": ctx[:60], "hb_metric": metric[:44], "hb_value": val,
                    "status": status, "iris_metric": mid, "iris_lob": lob,
                    "iris_cob": cob, "iris_value": "", "ratio": ""})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=HANDBOOK_DIR)
    ap.add_argument("--part")
    ap.add_argument("--table")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--limit-sheets", type=int, default=0)
    a = ap.parse_args()

    import openpyxl
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import app  # noqa: F401  (loads the engine)
    import iris_brain as brain

    iris_idx, names = build_iris_index(brain)
    print(f"IRIS index: {len(iris_idx)} (insurer, year) buckets, {len(names)} insurers")

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "*.xlsx"))
                   if not os.path.basename(f).startswith("~$"))
    if a.part:
        files = [f for f in files if a.part.lower() in os.path.basename(f).lower()]

    rowsout, tally, skipped = [], {}, []
    for f in files:
        part = os.path.splitext(os.path.basename(f))[0]
        wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        sheets = [s for s in wb.sheetnames if s.strip().lower() != "index"]
        if a.table:
            sheets = [s for s in sheets if s.strip() == a.table.strip()]
        if a.limit_sheets:
            sheets = sheets[:a.limit_sheets]
        for sh in sheets:
            # One malformed sheet must not end a 109-table run; record it and continue.
            try:
                rows = list(wb[sh].iter_rows(values_only=True))
                res = audit_sheet(rows, iris_idx, names, sh, part)
            except Exception as exc:
                print(f"  {part:<10} table {sh:<7} SKIPPED: {type(exc).__name__}: {exc}")
                skipped.append((part, sh, f"{type(exc).__name__}: {exc}"))
                continue
            rowsout += res
            c = {}
            for r in res:
                c[r["status"]] = c.get(r["status"], 0) + 1
            tally[f"{part}/{sh}"] = c
            print(f"  {part:<10} table {sh:<7} cells={len(res):<6} {c}")
        wb.close()

    agg = {}
    for c in tally.values():
        for k, v in c.items():
            agg[k] = agg.get(k, 0) + v
    total = sum(agg.values()) or 1
    print("\n=== totals ===")
    for k in ("ok", "total_ok", "total_mismatch", "value_mismatch",
              "relocated", "missing", "missing_zero"):
        print(f"  {k:<14} {agg.get(k,0):>8}  {agg.get(k,0)/total*100:5.1f}%")

    if skipped:
        print(f"\nskipped {len(skipped)} sheet(s):")
        for p_, sh, why in skipped:
            print(f"   {p_}/{sh}: {why}")

    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rowsout[0].keys()) if rowsout else [])
            w.writeheader()
            w.writerows(rowsout)
        print(f"\nwrote {len(rowsout)} rows to {a.out}")


if __name__ == "__main__":
    main()
