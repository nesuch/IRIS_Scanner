#!/usr/bin/env python3
"""Convert IRDAI Handbook *Part I-V* per-insurer tables into IRIS tidy rows.

The Handbook Parts hold ~104 statistical tables in many shapes. This tool
handles **Phase 1**: the clean "insurer (or reinsurer) × year" tables, which map
directly onto IRIS's existing Insurer dimension (entity = insurer, one metric
per table, value per financial year). Wide multi-segment matrices, state-wise
and balance-sheet tables are out of scope here (later phases).

Output columns: dimension, entity, metric, value, financial_year, quarter,
                line_of_business, class_of_business
Writes knowledge_base/raw_submissions/handbook_2024-25_parts.xlsx so Admin ->
"Sync Data" ingests it with no engine changes.
"""
import argparse
import os
import re
import sys

import openpyxl
import pandas as pd

DIMENSION = "Insurer"
QUARTER = "Annual"
DEFAULT_CLASS = "All Classes"

YEAR_RE = re.compile(r"^(?:\d{4}-\d{2}|\d{4})$")     # 2023-24 or 2023
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")              # first number, ignoring footnote marks
ENTITY_AXIS = {"insurer", "insurers", "reinsurer", "reinsurers", "company", "name of the insurer"}


def _is_entity_axis(c):
    """True if a header cell labels the entity column (insurer/reinsurer/state)."""
    low = c.lower().strip()
    return (low in ENTITY_AXIS or "insurer" in low or "reinsurer" in low.replace("-", "")
            or "state" in low or "union territory" in low)

# Phase-2 LOB matrices: (file, sheet). LOB + metric are read from the multi-row
# column headers; the parser auto-detects 2-level (LOB>year) vs 3-level
# (LOB>metric>year) layouts. Rows are insurers.
PHASE2 = [
    ("Part II", "41"),   # Segment-wise Gross Direct Premium (2-level)
    ("Part II", "44"),   # Net Premium / Claims Incurred / ICR by segment (3-level)
    ("Part IV", "83"),   # Reinsurers: NEP / Claims / ICR by segment (metric>LOB>year)
]

# Phase-2b/2c: transposed (insurers/reinsurers as columns; metrics are rows).
PHASE2B_TRANSPOSED = [
    ("Part II", "45", "General"),       # Underwriting Experience of insurers
    ("Part IV", "84", "Reinsurance"),   # Underwriting Experience of reinsurers
]
# Phase-3: State dimension (entity = state).
PHASE3_YEAR_SUBMETRIC = [
    ("Part I", "5", "Life", "Individual New Business"),
    ("Part I", "7", "Life", "Group New Business"),
]
PHASE3_MATRIX = [
    ("Part II", "42"),   # State-wise Gross Direct Premium (General) by segment
]
# Simple state x year tables (one metric from the title).
PHASE3_STATE_SIMPLE = [
    ("Part V", "97", "All Lines"),   # State-wise Registered Brokers
    ("Part V", "98", "All Lines"),   # State-wise Insurance Marketing Firms
]
# Phase-3b: state-wise health/PA/travel, class-split (year>class>sub-metric).
PHASE3_CLASS = [
    ("Part III", "67", "Health"),
    ("Part III", "69", "Personal Accident"),
    ("Part III", "70", "Travel (Overseas)"),
    ("Part III", "71", "Travel (Domestic)"),
    ("Part III", "68", "Health"),    # state individual health (New/Renewal/In-Force)
    ("Part III", "72", "Health"),    # state claims settlement (Individual/Group)
]

# Phase-4: line-item financial statements -> their own "Financials" view. The
# statement becomes the Line of Business and the statement section the Class.
FIN_DIMENSION = "Financials"
PHASE4 = [
    ("Part I", "24", "Policyholders Account"),   # Life
    ("Part I", "25", "Shareholders Account"),    # Life
    ("Part I", "26", "Balance Sheet"),           # Life
    ("Part II", "52", "Balance Sheet"),          # General & Health
    ("Part II", "51", "Shareholders Account"),   # General & Health
]

# Segmented statements: entity > year > segment > particulars.
PHASE7_SEGMENTED = [
    ("Part II", "50", "Policyholders Account"),   # General & Health (by segment)
    ("Part IV", "87", "Policyholders Account"),   # Reinsurers (by segment)
]

# Phase-8: more reports into Statements & Reports.
#   transposed (insurer columns, particulars rows): (part, sheet, report, unit)
PHASE8_TRANSPOSED = [
    ("Part I", "32", "Micro-Insurance Death Claims - Individual", "₹Lakh"),
    ("Part I", "33", "Micro-Insurance Death Claims - Group", "₹Lakh"),
    ("Part I", "34", "Micro-Insurance Claim Settlement - Individual", "Nos."),
    ("Part I", "35", "Micro-Insurance Claim Settlement - Group", "Nos."),
]
#   insurer rows x (year > sub-metric): (part, sheet, report, unit)
PHASE8_PERIODIC = [
    ("Part I", "37", "Grievances (Life)", "Nos."),
    ("Part II", "56", "Grievances (General & Health)", "Nos."),
    ("Part I", "28", "Persistency", "Per cent"),
    ("Part IV", "80", "Reinsurance Premium Schedule", "₹Crore"),
    ("Part IV", "81", "Segment-wise Reinsurance Premium Accepted", "₹Crore"),
]
#   insurer rows x (year > stage > sub-metric) -> Reports: (part, sheet, report, unit)
PHASE9_REPORTS_CLASS = [
    ("Part I", "15", "Individual Death Claims", "Nos."),
    ("Part I", "17", "Group Death Claims", "Nos."),
    ("Part II", "53", "Status of Claims", "Nos."),
]
# Health business by LIFE insurers (entities are life insurers -> Life sector).
PHASE9B_LIFE_HEALTH = [
    ("Part III", "73", "Health Business by Life Insurers - New", "Nos."),
    ("Part III", "74", "Health Business by Life Insurers - Renewal", "Nos."),
    ("Part III", "75", "Health Riders on Life Products - New", "Nos."),
    ("Part III", "76", "Health Riders on Life Products - Renewal", "Nos."),
]

# Phase-10: Channel-wise distribution -> new Channel view (entity = channel).
PHASE10_CHANNEL_MEASURES = [          # channel rows x (measure > year)
    ("Part V", "99", "Life - Individual New Business"),
    ("Part V", "101", "Life - Group New Business"),
]
PHASE10_CHANNEL_SEGMENT = [           # channel cols x segment rows (GDP)
    ("Part V", "103", "General"),
]
PHASE10_CHANNEL_CLASS = [            # channel rows x (year > class > sub-metric)
    ("Part V", "104", "Health"),
]

# Phase-11: insurer rows x (category > year) -> Reports (category = line item).
# (part, sheet, report name, unit)
PHASE11_MEASURES = [
    ("Part II", "47", "Investments (AUM) by Instrument", "₹Crore"),
    ("Part I", "30", "Offices by Region", "Nos."),
    ("Part V", "93", "Avg Individual Policies Sold per Agent", "Nos."),
    ("Part V", "94", "Avg New Business Premium per Agent", "₹Lakh"),
    ("Part V", "95", "Avg Premium per Policy", "₹"),
    ("Part I", "27", "Lapsed / Forfeited Policies (Non-Linked)", "'000s"),
]

# State x insurer cross-tabs -> entity=insurer, class=state. (part, sheet, lob,
# base metric, unit). Sectors come from the part (6/8/29 Life, 54 Non-Life, 96 Life).
PHASE12_CROSSTAB = [
    ("Part I", "6", "Individual New Business by State", "New Business", "₹Crore"),
    ("Part I", "8", "Group New Business by State", "New Business", "₹Crore"),
    ("Part I", "29", "Offices by State", "No. of Offices", "Nos."),
    ("Part II", "54", "Offices by State", "No. of Offices", "Nos."),
    ("Part V", "96", "Individual Agents by State", "No. of Agents", "Nos."),
]

# Industry-aggregate tables -> Industry view. (part, sheet, entity, lob/base, row_as, unit)
PHASE13_INDUSTRY_ROWS = [
    ("Part I", "14", "Life Insurers (Industry)", "Individual Death Claims", "metric", "Nos."),
    ("Part I", "16", "Life Insurers (Industry)", "Group Death Claims", "metric", "Nos."),
    ("Part IV", "82", "Non-Life & Reinsurers (Industry)", "Net Retention (Per cent)", "lob", "Per cent"),
]
# (part, sheet, entity, lob, unit)
PHASE13_INDUSTRY_2KEY = [
    ("Part I", "20", "Life Insurers (Industry)", "Investments (AUM)", "₹Crore"),
    ("Part II", "46", "General, Health & RE (Industry)", "Investments (AUM)", "₹Crore"),
]

# Phase-6: more per-insurer transposed tables routed into the Statements &
# Reports view as named reports. (part, sheet, report name, fallback unit)
PHASE6_REPORTS = [
    ("Part I", "10", "In-Force Policies", "'000s"),
    ("Part I", "11", "In-Force Sum Assured", "₹Crore"),
    ("Part I", "18", "Death Claim Settlement Duration - Individual", "Nos."),
    ("Part I", "19", "Death Claim Settlement Duration - Group", "Nos."),
    ("Part IV", "88", "Shareholders Account", "₹Crore"),
    ("Part IV", "89", "Balance Sheet", "₹Crore"),   # reinsurers
]

PHASE2B_CLASS = [
    ("Part III", "58", "Health"),
    ("Part III", "59", "Personal Accident"),
    ("Part III", "60", "Travel (Overseas)"),
    ("Part III", "61", "Travel (Domestic)"),
    ("Part III", "62", "Health"),
    ("Part III", "63", "Personal Accident"),
    ("Part III", "64", "Travel (Overseas)"),
    ("Part III", "65", "Travel (Domestic)"),
]

# Phase-1 tables: (file, sheet, line_of_business). Metric is derived from the title.
PHASE1 = [
    ("Part I", "2", "Life"),
    ("Part I", "3", "Life"),
    ("Part I", "9", "Life"),
    ("Part I", "22", "Life"),
    ("Part II", "40", "General"),
    ("Part IV", "86", "Reinsurance"),
    ("Part IV", "85", "Reinsurance"),   # Equity/Assigned Capital of Reinsurers
    ("Part II", "48", "General"),   # Equity Share Capital of General & Health
    # Tables 90-92 are agents *of life insurers* — so Life, not "All Lines".
    ("Part V", "90", "Life"),
    ("Part V", "91", "Life"),
    ("Part V", "92", "Life"),
]

# Phase-5: quarterly insurer tables — columns are "Month YYYY" periods.
PHASE5_QUARTERLY = [
    ("Part I", "23", "Life", False),    # Solvency Ratio of Life Insurers
    ("Part II", "49", None, True),      # Solvency of General/Health/Reinsurance
]

# Rows whose entity label is a group header / aggregate, not a real insurer.
SKIP_ENTITIES = re.compile(
    r"^(public sector|private sector|standalone health|stand-alone health|"
    r"speciali[sz]ed|grand total|industry|total|sub[- ]?total|all india).*?$"
    r"|.*\btotal\b\)?$|.*\baverage$|.*cancelled.*",
    re.I,
)

# Canonical state/UT names (variant spelling -> standard), keyed by a squashed form.
def _state_key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


_STATE_GROUPS = {
    "Tamil Nadu": ["Tamil Nadu", "TamilNadu"],
    "Chhattisgarh": ["Chhattisgarh", "Chattisgarh"],
    "Jharkhand": ["Jharkhand", "Jharkand"],
    "Maharashtra": ["Maharashtra", "Maharasthra", "Maharastra"],
    "Odisha": ["Odisha", "Orissa"],
    "Rajasthan": ["Rajasthan", "Rajashtan"],
    "Uttarakhand": ["Uttarakhand", "Uttrakhand"],
    "Andaman & Nicobar Islands": ["Andaman & Nicobar Islands", "Andaman & Nicobar Is"],
    "Dadra & Nagar Haveli and Daman & Diu": [
        "Dadra & Nagar Haveli and Daman & Diu", "Dadra & Nagara Haveli and Daman & Diu"],
    "Delhi (NCT)": ["Delhi (NCT)", "Delhi", "New Delhi", "NCT of Delhi"],
    "Jammu & Kashmir": ["Jammu & Kashmir", "Jammu and Kashmir"],
}
STATE_CANON = {_state_key(v): canon for canon, vs in _STATE_GROUPS.items() for v in vs}


def _canon_state(name):
    return STATE_CANON.get(_state_key(name), name)
UNIT_RE = re.compile(r"\((₹\s?crore|in lakhs|amount in ₹lakh|₹\s?lakh|lakhs|in crore|"
                     r"us ?\$|per ?cent|nos\.?|number|₹.*?)\)", re.I)


def _clean(v):
    return re.sub(r"\s+", " ", str(v or "").replace("\n", " ")).strip()


def _to_fy(year):
    """Normalise a bare calendar year (an 'As on 31 March YYYY' balance-sheet /
    equity column) to the financial year it closes: 2022 -> '2021-22'.
    Leaves already-financial-year strings ('2022-23') unchanged."""
    s = str(year).strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})", s)
    if m:                                            # fix typos like "2023-23"
        start = int(m.group(1))
        exp = (start + 1) % 100
        return s if int(m.group(2)) == exp else f"{start}-{exp:02d}"
    if re.fullmatch(r"\d{4}", s):
        y = int(s)
        return f"{y - 1}-{s[-2:]}"
    return s


# UNAMBIGUOUS aliases only (one company regardless of sector). Ambiguous short
# forms (Bajaj Allianz, HDFC, Reliance, Shriram, Bharti AXA, Go Digit, Aditya
# Birla...) are resolved per-sector in the canonicalisation pass instead, so a
# Life table never maps them to the General arm.
ENTITY_ALIASES = {
    "lic": "Life Insurance Corporation of India",
    "lic of india": "Life Insurance Corporation of India",
    "acko life": "Acko Life Insurance Ltd.",
    "credit access life": "Credit Access Life Insurance Ltd.",
    "adity birla sun life": "Aditya Birla Sun Life Insurance Ltd.",
    "tata aia": "Tata AIA Life Insurance Ltd.",
    "star union dai-ichi": "Star Union Dai-ichi Life Insurance Ltd.",
    "factorty mutual": "Factory Mutual",
    "aic": "Agriculture Insurance of India Ltd.",
    # Reinsurer variants across Part IV tables.
    "gic": "General Insurance Corporation of India (GIC Re)",
    "gic re. (public)": "General Insurance Corporation of India (GIC Re)",
    "gic re": "General Insurance Corporation of India (GIC Re)",
    "general insurance corporation (gic re)": "General Insurance Corporation of India (GIC Re)",
    "general insurance corporation (gic)": "General Insurance Corporation of India (GIC Re)",
    "general insurance corporation of india": "General Insurance Corporation of India (GIC Re)",
    "iti": "ITI Reinsurance Ltd.",
    "iti re": "ITI Reinsurance Ltd.",
    "iti (private)": "ITI Reinsurance Ltd.",
    "sud life": "Star Union Dai-ichi Life Insurance Ltd.",
    "rga life": "RGA",
    "gen re": "General Reinsurance AG",
}


# Sector of each Handbook Part (per the index), used to disambiguate insurer
# short forms — a "Bajaj Allianz" in a Part I table is the Life company.
# General & Health share one "Non-Life" sector so a SAHI insurer appearing in
# both Part II and Part III resolves to one entity; Life and Reinsurance stay
# distinct (that's where short forms like "Bajaj Allianz" must not cross).
# Part IV (reinsurance) also folds into "Non-Life" for name resolution, because
# reinsurers appear in some Part II tables too (e.g. AUM 47); the Reinsurance LOB
# still distinguishes them in the data.
SECTOR_OF = {"Part I": "Life", "Part II": "Non-Life", "Part III": "Non-Life",
             "Part IV": "Non-Life", "Part V": "Life"}

# Explicit per-sector resolutions for names that stay ambiguous even within a
# sector (confirmed with the user). Key: (sector, squashed-name) -> canonical.
SECTOR_ALIASES = {
    ("Non-Life", "hdfcergo"): "HDFC ERGO General Insurance Co. Ltd.",
    ("Non-Life", "reliance"): "Reliance General Insurance Co. Ltd.",
    ("Life", "reliance"): "Reliance Nippon Life Insurance Ltd.",
    ("Life", "reliancelife"): "Reliance Nippon Life Insurance Ltd.",
    ("Life", "reliancenippon"): "Reliance Nippon Life Insurance Ltd.",
    ("Life", "fg"): "Future Generali India Life Insurance Ltd.",
    ("Life", "pnb life"): "PNB MetLife India Insurance Co. Ltd.",
    ("Life", "pnblife"): "PNB MetLife India Insurance Co. Ltd.",
}


def _stamp(rows, part, sector=None):
    """Tag rows with their sector so name resolution stays within-sector.
    State and Channel entities are isolated (not insurers). `sector` overrides
    the part default (e.g. health-by-life tables hold *life* insurers)."""
    sec = sector or SECTOR_OF.get(part, "")
    for r in rows:
        r["_sector"] = r["dimension"] if r["dimension"] in ("State", "Channel", "Industry") else sec
    return rows


def _canon_metric(m):
    """Unify equivalent metric names across tables (insurers vs reinsurers)."""
    m = m.replace("(%)", "(Per cent)").replace("(Percent)", "(Per cent)")
    m = re.sub(r"\bIncurred Claim Ratio\b", "Incurred Claims Ratio", m)
    m = re.sub(r"\bNet Premium Earned\b", "Net Earned Premium", m)
    m = re.sub(r"\bNet Incurred Claims\b", "Claims Incurred (Net)", m)
    return re.sub(r"\s{2,}", " ", m).strip()


# Corporate-filler words that don't distinguish one insurer from another.
_FILLER = {"insurance", "co", "company", "ltd", "limited", "india", "the",
           "and", "assurance", "services", "branch", "branches", "of"}


def _name_quality(n):
    """Rank candidate spellings: prefer mixed-case, ending in 'Ltd.', then length."""
    return (0 if n.isupper() else 1, 1 if re.search(r"\bLtd\.?$", n) else 0, len(n))


def _sig(name):
    """Distinctive token set of an insurer name (drops corporate fillers)."""
    toks = re.sub(r"[^a-z0-9 ]", " ", name.lower()).split()
    return frozenset(t for t in toks if t not in _FILLER)


def _norm_entity(name):
    """Light canonicalisation so spelling variants of one insurer don't split
    into separate entities across tables (e.g. 'Ltd' vs 'Ltd.', 'Sunlife')."""
    s = re.sub(r"^[\s@#*$^%&\-]+|[\s@#*$^%.&]+$", "", _clean(name))  # strip footnote marks
    s = re.sub(r"\s*\(\d+\)\s*$", "", s).strip()      # strip trailing "(1)" footnotes
    s = re.sub(r"\bLimited\b", "Ltd", s, flags=re.I)
    s = re.sub(r"\bLtd\.?\s*$", "Ltd.", s)           # normalise trailing Ltd.
    # Normalise compound brand words (case-insensitive — some tables are ALL CAPS).
    s = re.sub(r"\bsun\s*life\b", "Sun Life", s, flags=re.I)
    s = re.sub(r"\bmax\s*life\b", "Max Life", s, flags=re.I)
    s = re.sub(r"\bcredit\s*access\b", "Credit Access", s, flags=re.I)
    s = re.sub(r"\bcompany\b\s*", "", s, flags=re.I)  # filler word; safe to drop
    s = re.sub(r"\bGo\s*digit\b", "Go Digit", s, flags=re.I)        # compound spacing
    s = re.sub(r"\bIndia\s*First\b", "India First", s, flags=re.I)
    s = re.sub(r"\bKshema\s*General\b", "Kshema General", s, flags=re.I)
    s = re.sub(r"\bAlianz\b", "Allianz", s, flags=re.I)            # typos
    s = re.sub(r"\bFuture\s*generali\b", "Future Generali", s, flags=re.I)
    s = re.sub(r"\bManipal\s*Cigna\b", "ManipalCigna", s, flags=re.I)
    s = re.sub(r"\bMet\s*Life\b", "MetLife", s, flags=re.I)
    s = re.sub(r"\bScore\s+SE\b", "SCOR SE", s, flags=re.I)        # typo
    s = s.replace("Limtied", "Limited")              # source typo
    s = re.sub(r"(Lloyd's of India)\s*-\s*", r"\1 - ", s)  # tidy "India- Markel"
    s = re.sub(r"\bLtd\.?\s*$", "Ltd.", s)           # re-normalise tail after edits
    s = re.sub(r"\s{2,}", " ", s).strip()
    if "edelweiss" in s.lower():                  # Edelweiss Tokio Life -> Edelweiss Life (rename)
        return "Edelweiss Life Insurance Ltd."
    if "magma" in s.lower():                       # Magma General / Magma HDI -> one company
        return "Magma HDI General Insurance Co. Ltd."
    return ENTITY_ALIASES.get(s.lower(), s)


def _to_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return round(float(v), 4)
    s = str(v).strip()
    if s.startswith("(") and s.endswith(")"):
        return None  # parenthetical growth-% annotation, not a value
    m = NUM_RE.search(s.replace(",", ""))
    if not m:
        return None
    try:
        return round(float(m.group()), 4)
    except ValueError:
        return None


def _norm_unit(u):
    return {"₹ crore": "₹Crore", "crore": "₹Crore", "₹crore": "₹Crore",
            "₹lakh": "₹Lakh", "₹ lakh": "₹Lakh", "amount in ₹lakh": "₹Lakh",
            "lakhs": "Lakhs", "lakh": "Lakh", "nos.": "Nos.", "nos": "Nos.",
            "number": "Nos.", "per cent": "Per cent", "percent": "Per cent",
            "us $": "US $", "us$": "US $"}.get(u.strip().lower(), u.strip())


def _find_unit(grid):
    for r in grid:
        for c in r:
            m = UNIT_RE.search(c)
            if m:
                return _norm_unit(re.sub(r"^(in|amount in)\s+", "", m.group(1).strip(), flags=re.I))
    return ""


def _ffill(row):
    """Forward-fill a header row across merged-cell gaps."""
    out, cur = [], ""
    for c in row:
        t = _clean(c)
        if t:
            cur = t
        out.append(cur)
    return out


def _tidy_metric(name):
    """Title-case an all-caps metric, normalising its trailing unit."""
    s = _clean(name)
    m = re.search(r"\(([^()]*)\)\s*$", s)
    unit = ""
    if m and UNIT_RE.fullmatch(f"({m.group(1)})"):
        unit = _norm_unit(m.group(1))
        s = s[:m.start()].strip()
    if s.isupper():
        s = s.title()
    return f"{s} ({unit})" if unit else s


# Aggregate LOB labels to drop (contextless sums of the segments above them).
SKIP_LOBS = {"total", "all segments", "total segments", "grand total", "all", "total insurance"}

# Segment names used to tell which header level is the Line of Business (vs the
# metric) in a matrix, since the two can be nested in either order.
SEGMENT_VOCAB = {"fire", "marine", "marine cargo", "marine hull", "motor", "motor od",
                 "motor tp", "health", "life", "engineering", "aviation", "liability",
                 "personal accident", "pa", "crop", "credit", "travel", "misc",
                 "miscellaneous", "others", "health + pa + travel"}


def _seg_score(level, year_cols):
    """Fraction of a header level's values that look like Line-of-Business names."""
    vals = [_norm_lob(level[i]) for i in year_cols if i < len(level) and level[i]]
    return sum(1 for v in vals if v.lower() in SEGMENT_VOCAB) / len(vals) if vals else 0


def _norm_lob(s):
    s = _clean(s)
    s = re.sub(r"\s+insurance\s*$", "", s, flags=re.I)
    s = re.sub(r"^other segments?$", "Others", s, flags=re.I)
    s = re.sub(r"\bPA\b\s*\+?\s*", "PA + ", s)            # tidy "Health + PA+ TRAVEL"
    s = re.sub(r"\bTRAVEL\b", "Travel", s)
    s = re.sub(r"\s{2,}", " ", s).replace("+ +", "+").strip()
    return s.title() if s.isupper() else s


def _metric_from_title(title, unit):
    t = _clean(title)
    t = re.sub(r"^(table|statement)\s*\d+\s*[:.\-]?\s*", "", t, flags=re.I)
    t = re.sub(r"\b(segment|state)(\s+and\s+ut)?[- ]?wise\s+", "", t, flags=re.I)  # redundant once it's a column
    t = re.sub(r"\s*[-–]\s*general insurance\s*$", "", t, flags=re.I)
    # Drop the "... of <sector> (re)insurers" tail that's redundant once entity
    # and line-of-business are columns.
    t = re.sub(r"\bof\s+(life|general|health|non-life|general and health|"
               r"life and general)\s+(re)?insurers?\b", "", t, flags=re.I)
    t = re.sub(r"\bof\s+(re)?insurers?\b", "", t, flags=re.I)
    t = re.sub(r"\s*[-–]\s*insurer[- ]?wise.*$", "", t, flags=re.I)  # drop "- Insurer-wise" tail
    t = re.sub(r"\s{2,}", " ", t).strip(" -")
    t = t.title() if t.isupper() else t
    if unit and unit.lower() not in {"", "none", "nan"}:
        t = f"{t} ({unit})"
    return t


def convert_matrix(ws, dimension=DIMENSION, fallback_unit="₹Crore"):
    """Parse an LOB matrix: 2-level (LOB>year) or 3-level (LOB>metric>year).
    Rows are insurers (default) or states (dimension="State")."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    if not grid:
        return []
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1:
        return []
    seg = _ffill(grid[yr - 1])                       # level directly above years
    up = _ffill(grid[yr - 2]) if yr >= 2 else [""] * len(seg)
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}

    ent_col = None
    for hr in (grid[yr - 1], grid[yr - 2] if yr >= 2 else []):
        ent_col = next((i for i, c in enumerate(hr) if _is_entity_axis(c)), None)
        if ent_col is not None:
            break
    if ent_col is None:
        ent_col = min(years) - 1

    # 3-level only if the row above carries >1 *meaningful* grouping label
    # (ignore blanks and stray unit tokens like a lone "(₹Crore)").
    def _meaningful(v):
        return bool(v) and not re.fullmatch(r"\(?\s*[₹]?\s*(crore|lakh|lakhs|per ?cent|nos\.?|us ?\$|%)\s*\)?", v.strip(), re.I)
    three = len({up[i] for i in years if _meaningful(up[i])}) > 1
    base_metric = _metric_from_title(grid[0][0] if grid[0] else "", _find_unit(grid[:3]) or fallback_unit)
    # The two header levels can be nested either way (LOB>metric as in Table 44,
    # or metric>LOB as in Table 83). Pick the LOB level by segment-vocabulary.
    if three:
        lob_level, met_level = (seg, up) if _seg_score(seg, years) >= _seg_score(up, years) else (up, seg)

    out = []
    for r in raw[yr + 1:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci, fy in years.items():
            if ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            lob = _norm_lob(lob_level[ci]) if three else _norm_lob(seg[ci])
            metric = _tidy_metric(met_level[ci]) if three else base_metric
            if not lob or lob.lower() in SKIP_LOBS:
                continue
            out.append({
                "dimension": dimension, "entity": entity, "metric": metric,
                "value": value, "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": DEFAULT_CLASS,
            })
    return out


def convert_year_submetric(ws, lob, dimension, prefix="", fallback_unit="₹Crore"):
    """year (top, merged) > sub-metric (per column); rows = states/insurers.
    e.g. Table 5/7 — State-wise New Business: columns are year × {Policies, Premium}."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr + 1 >= len(grid):
        return []
    ylevel = _ffill(grid[yr])
    slevel = grid[yr + 1]                             # sub-metric, one per column
    ent_col = next((i for i, c in enumerate(grid[yr]) if _is_entity_axis(c)), 1)
    out = []
    for r in raw[yr + 2:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci in range(len(r)):
            if not YEAR_RE.match(ylevel[ci] if ci < len(ylevel) else ""):
                continue
            sub = _clean(slevel[ci]) if ci < len(slevel) else ""
            if not sub:
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            name = f"{prefix} {sub}".strip()
            if re.search(r"\([^)]*\)\s*$", sub):          # already carries a unit
                metric = name
            else:
                low = sub.lower()
                if any(k in low for k in ("no.", "number", "polic", "scheme", "lives", "person", "claims")):
                    unit = "Nos."
                else:
                    unit = fallback_unit                  # premium / sum assured / amount
                metric = f"{name} ({unit})"
            out.append({
                "dimension": dimension, "entity": entity, "metric": metric,
                "value": value, "financial_year": ylevel[ci], "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": DEFAULT_CLASS,
            })
    return out


def convert_table(ws, lob, dimension=DIMENSION):
    rows = [[_clean(c) for c in r] for r in ws.iter_rows(values_only=True)]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []
    raw = list(ws.iter_rows(values_only=True))  # untouched values for numbers

    title = rows[0][0] if rows[0] else ""
    unit = ""
    for r in rows[:5]:
        for c in r:
            m = UNIT_RE.search(c)
            if m:
                unit = m.group(1).replace("In ", "").replace("in ", "").strip()
                break
        if unit:
            break
    # Normalise common unit spellings.
    unit = {"₹ crore": "₹Crore", "crore": "₹Crore", "lakhs": "Lakhs",
            "nos.": "Nos.", "nos": "Nos.", "number": "Nos.", "per cent": "Per cent",
            "us $": "US $", "us$": "US $"}.get(unit.lower(), unit)

    # Header row: the first row with >=2 year cells; entity column = the label
    # column (the 'Insurer'/'Reinsurers' header, else the col left of the years).
    hdr_idx = next((i for i, r in enumerate(rows[:8])
                    if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if hdr_idx is None:
        return []
    header = rows[hdr_idx]
    year_cols = {i: c for i, c in enumerate(header) if YEAR_RE.match(c)}
    ent_col = next((i for i, c in enumerate(header) if _is_entity_axis(c)), None)
    if ent_col is None:
        first_year = min(year_cols)
        ent_col = first_year - 1 if first_year > 0 else 0

    metric = _metric_from_title(title, unit)
    out = []
    for r in raw[hdr_idx + 1:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci, fy in year_cols.items():
            if ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": dimension,
                "entity": entity,
                "metric": metric,
                "value": value,
                "financial_year": fy,
                "quarter": QUARTER,
                "line_of_business": lob,
                "class_of_business": DEFAULT_CLASS,
            })
    return out


def _tidy_class(c):
    c = _clean(c)
    low = c.lower()
    if "total" in low:                              # any total / grand-total variant
        return "Total"
    if "irctc" in low:
        return "IRCTC Scheme"
    if "pmjdy" in low or "jan dhan" in low:
        return "PMJDY"
    if "pmsby" in low or "suraksha bima" in low:
        return "PMSBY"
    if "ayushman" in low or "pmjay" in low:
        return "AB-PMJAY"
    if "rsby" in low and "group" not in low:
        return "RSBY"
    if "other govt" in low or "other government" in low:
        return "Government Sponsored (Other)"
    if low.startswith("government sponsored"):
        return "Government Sponsored"
    if low.startswith("group"):
        govt = ("rsby" in low or "govt sponsor" in low or "government sponsor" in low)
        if "other than" in low or "exclud" in low:
            return "Group (excl. Govt)"
        if govt:
            return "Government Sponsored"
        return "Group"
    if "family" in low and "floater" in low and "excluding individual" in low:
        return "Family Floater (excl. Individual)"
    if low.startswith("individual"):
        if "excluding family" in low:
            return "Individual (excl. Family Floater)"
        if "other than family" in low or ("other" in low and "floater" in low):
            return "Individual - Other"
        if "family floater" in low:
            return "Individual - Family Floater"
        return "Individual"
    return c


def _submetric(sub, table_unit):
    s = _clean(sub).replace("Permium", "Premium")
    low = s.lower()
    if "ratio" in low:
        unit = "Per cent"
    elif re.search(r"\([^)]*\)\s*$", s):       # already carries a unit/qualifier
        unit = ""
    elif "polic" in low:
        s, unit = "No. of Policies", "Nos."
    elif low.startswith(("no.", "number")) or "no. of" in low or "no.of" in low:
        unit = "Nos."                          # counts (e.g. "No. of claims paid")
    else:
        unit = table_unit
    return f"{s} ({unit})" if unit else s


_GEN_ITEM = re.compile(r"^(\(|total\b|sub[- ]?total|others?\b|less[: ]|add[: ])", re.I)
# Footnote / disclaimer rows that aren't real line items.
_FOOTNOTE = re.compile(r"^\s*[#*$^@]|^\s*(note\b|source\b|disclaimer|refer\b)|w\.e\.f|demerger", re.I)


def convert_transposed(ws, lob, fallback_unit="₹Crore", track_sections=False,
                       dimension=DIMENSION, section_as_class=False):
    """Insurers are column groups, metrics are row labels (e.g. Table 45).
    With track_sections (balance-sheet/P&L tables) a no-value row becomes a
    section header. section_as_class puts that section in the Class column (used
    by the Financial Statements view); otherwise it prefixes colliding items."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1:
        return []
    ent_row = _ffill(grid[yr - 1])
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}
    unit = _find_unit(grid[:yr]) or fallback_unit

    out, current_section = [], ""
    metric_sections = {}
    # Disambiguate distinct source rows that share the same (section,label):
    # each such row gets a stable ordinal so they never collapse in the pivot.
    seen_keys, ordinal = {}, {}
    for ri, r in enumerate(raw[yr + 1:]):
        raw_label = _clean(r[0]) if r else ""
        label = raw_label.rstrip("*#: ").strip()
        if not label or "=" in label or _FOOTNOTE.search(raw_label):
            continue
        row_vals = [(_to_number(r[ci]) if ci < len(r) else None) for ci in years]
        if track_sections and (not any(v is not None for v in row_vals) or raw_label.endswith(":")):
            current_section = label.title() if label.isupper() else label
            continue
        key = (current_section, label)
        if ri not in ordinal:
            seen_keys[key] = seen_keys.get(key, 0) + 1
            ordinal[ri] = seen_keys[key]
        metric_sections.setdefault(label, set()).add(current_section)
        for ci, fy in years.items():
            entity = _norm_entity(ent_row[ci]) if ci < len(ent_row) else ""
            if not entity or SKIP_ENTITIES.match(entity) or ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": dimension, "entity": entity, "metric": label,
                "section": current_section, "ord": ordinal[ri], "value": value,
                "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": DEFAULT_CLASS,
            })
    for row in out:
        sec, n = row.pop("section"), row.pop("ord")
        base = row["metric"]
        if section_as_class:
            row["class_of_business"] = sec or "General"
        elif sec and (len(metric_sections.get(base, ())) > 1 or _GEN_ITEM.match(base)):
            base = f"{sec} — {base}"
        if n > 1:                                # repeated label → stable ordinal
            base = f"{base} ({n})"
        row["metric"] = f"{base} ({unit})" if unit else base
    return out


CHANNEL_DIMENSION = "Channel"
_SUBMETRIC_VOCAB = ("polic", "premium", "lives", "scheme", "persons", "amount",
                    "sum assured", "no.", "number", "offices", "agents", "providers")


def convert_state_insurer_crosstab(ws, lob, base_metric, fallback_unit="₹Crore",
                                   dimension=DIMENSION):
    """State (rows) x insurer (col group) x year [x sub-metric]. Flattened to
    entity=insurer, class_of_business=state so all three axes survive (6/8/29/54/96)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1:
        return []
    insurer_row = _ffill(grid[yr - 1])
    year_row = _ffill(grid[yr])
    # optional sub-metric row directly below the years
    sub_row = None
    if yr + 1 < len(grid):
        cand = grid[yr + 1]
        if sum(1 for c in cand if c and any(v in c.lower() for v in _SUBMETRIC_VOCAB)) >= 2:
            sub_row = cand
    unit = _find_unit(grid[:yr]) or fallback_unit
    state_col = next((i for i, c in enumerate(grid[yr - 1]) if _is_entity_axis(c)
                      or "state" in c.lower()), 1)
    data_start = yr + 2 if sub_row else yr + 1
    out = []
    for r in raw[data_start:]:
        cells = [_clean(c) for c in r]
        state = _canon_state(cells[state_col]) if state_col < len(cells) else ""
        if not state or SKIP_ENTITIES.match(state) or _is_entity_axis(state):
            continue
        for ci in range(len(r)):
            fy = year_row[ci] if ci < len(year_row) else ""
            insurer = _norm_entity(insurer_row[ci]) if ci < len(insurer_row) else ""
            if not YEAR_RE.match(fy) or not insurer or _is_entity_axis(insurer):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            if sub_row:
                sub = _clean(sub_row[ci]) if ci < len(sub_row) else ""
                metric = _submetric(sub, unit) if sub else f"{base_metric} ({unit})"
            else:
                metric = f"{base_metric} ({unit})"
            out.append({
                "dimension": dimension, "entity": insurer, "metric": metric,
                "value": value, "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": state,
            })
    return out


def convert_channel_measures(ws, lob, fallback_unit="Nos.", dimension=CHANNEL_DIMENSION,
                             class_of_business=DEFAULT_CLASS):
    """Row-entity x (measure > year). e.g. 99/101 channel new business, and (for
    dimension=Financials) AUM by instrument (47), offices by region (30),
    agent productivity (93-95) — the measure becomes the line item."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1:
        return []
    measure_row = _ffill(grid[yr - 1])
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}
    ent_col = min(years) - 1 if years else 1
    out = []
    for r in raw[yr + 1:]:
        cells = [_clean(c) for c in r]
        ch = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not ch or SKIP_ENTITIES.match(ch):
            continue
        for ci, fy in years.items():
            if ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            meas = measure_row[ci] if ci < len(measure_row) else ""
            low = meas.lower()
            unit = ("Nos." if any(k in low for k in ("polic", "scheme", "number", "lives", "no."))
                    else "₹Crore" if ("premium" in low or "amount" in low) else fallback_unit)
            metric = f"{meas} ({unit})" if meas else "Value"
            out.append({
                "dimension": dimension, "entity": ch, "metric": metric,
                "value": value, "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": class_of_business,
            })
    return out


def convert_channel_segment(ws, lob_unused=None, fallback_unit="₹Crore"):
    """Channel column-groups > year, segment rows (Table 103). LOB = segment."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1:
        return []
    chan_row = _ffill(grid[yr - 1])
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}
    unit = _find_unit(grid[:yr]) or fallback_unit
    metric = f"Gross Direct Premium ({unit})"
    out = []
    for r in raw[yr + 1:]:
        seg = _norm_lob(_clean(r[0])) if r else ""
        if not seg or seg.lower() in SKIP_LOBS:
            continue
        for ci, fy in years.items():
            entity = _norm_entity(chan_row[ci]) if ci < len(chan_row) else ""
            if not entity or _is_entity_axis(entity) or ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": CHANNEL_DIMENSION, "entity": entity, "metric": metric,
                "value": value, "financial_year": fy, "quarter": QUARTER,
                "line_of_business": seg, "class_of_business": DEFAULT_CLASS,
            })
    return out


INDUSTRY_DIMENSION = "Industry"


def convert_industry_rows(ws, entity, lob_or_base, row_as="metric",
                          fallback_unit="Nos.", dimension=INDUSTRY_DIMENSION):
    """Single industry-aggregate table: rows = line items, columns = [measure>]year.
    row_as='metric' -> metric=row label, lob fixed (14/16); row_as='lob' ->
    lob=row label, metric fixed (82)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None:
        return []
    year_row = _ffill(grid[yr])
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}
    measure_row = _ffill(grid[yr - 1]) if (yr >= 1 and row_as == "metric") else []
    unit = _find_unit(grid[:yr]) or fallback_unit
    out = []
    for r in raw[yr + 1:]:
        label = _clean(r[0]) if r else ""
        if not label or SKIP_ENTITIES.match(label) or _FOOTNOTE.search(label):
            continue
        for ci, fy in years.items():
            value = _to_number(r[ci]) if ci < len(r) else None
            if value is None:
                continue
            if row_as == "metric":
                meas = measure_row[ci] if ci < len(measure_row) else ""
                mlabel = f"{meas} - {label}" if (meas and not YEAR_RE.match(meas)) else label
                metric, lob = _submetric(mlabel, unit), lob_or_base
            else:
                lob, metric = _norm_lob(label), f"{lob_or_base} ({unit})"
            out.append({
                "dimension": dimension, "entity": entity, "metric": metric,
                "value": value, "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": DEFAULT_CLASS,
            })
    return out


def convert_industry_2key(ws, entity, lob, fallback_unit="₹Crore",
                          dimension=INDUSTRY_DIMENSION):
    """Industry AUM: col0=instrument (merged), col1=item (Amount/% of total),
    columns=year. metric=instrument, class=item (20/46)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None:
        return []
    year_row = _ffill(grid[yr])
    years = {i: c for i, c in enumerate(grid[yr]) if YEAR_RE.match(c)}
    out, cur_k0 = [], ""
    for r in raw[yr + 1:]:
        cells = [_clean(c) for c in r]
        k0 = cells[0] if cells else ""
        k1 = cells[1] if len(cells) > 1 else ""
        if k0:
            cur_k0 = k0
        if not cur_k0 or SKIP_ENTITIES.match(cur_k0):
            continue
        low = k1.lower()
        klass = ("% of Total" if ("%" in k1 or "percent" in low or "per cent" in low)
                 else "Amount")
        for ci, fy in years.items():
            value = _to_number(r[ci]) if ci < len(r) else None
            if value is None:
                continue
            unit = "Per cent" if klass == "% of Total" else fallback_unit
            out.append({
                "dimension": dimension, "entity": entity,
                "metric": _submetric(cur_k0, unit), "value": value,
                "financial_year": fy, "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": klass,
            })
    return out


def convert_insurer_periodic(ws, report, fallback_unit="Nos.", dimension=FIN_DIMENSION):
    """Insurer rows x (year > sub-metric) columns, e.g. grievances (37/56),
    persistency (28). Routed to Statements & Reports: sub-metric = line item."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr + 1 >= len(grid):
        return []
    year_row = _ffill(grid[yr])
    sub_row = grid[yr + 1]
    unit = _find_unit(grid[:yr + 1]) or fallback_unit
    ent_col = next((i for i, c in enumerate(grid[yr]) if _is_entity_axis(c)), 1)
    out = []
    for r in raw[yr + 2:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci in range(len(r)):
            if not YEAR_RE.match(year_row[ci] if ci < len(year_row) else ""):
                continue
            sub = _clean(sub_row[ci]) if ci < len(sub_row) else ""
            if not sub:
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            sub = re.sub(r"^(\d+)\s*\*+$", r"\1th Month", sub)   # persistency buckets
            metric = _submetric(sub, unit)
            out.append({
                "dimension": dimension, "entity": entity, "metric": metric,
                "value": value, "financial_year": year_row[ci], "quarter": QUARTER,
                "line_of_business": report, "class_of_business": "General",
            })
    return out


def convert_segmented_statement(ws, statement, fallback_unit="₹Crore", dimension=FIN_DIMENSION):
    """Segmented financial statement: columns are entity > year > segment,
    rows are particulars (Tables 50, 87). The segment becomes the section."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr < 1 or yr + 1 >= len(grid):
        return []
    ent_row = _ffill(grid[yr - 1])
    year_row = _ffill(grid[yr])
    seg_row = grid[yr + 1]
    unit = _find_unit(grid[:yr]) or fallback_unit
    out = []
    for r in raw[yr + 2:]:
        raw_label = _clean(r[0]) if r else ""
        label = raw_label.rstrip("*#: ").strip()
        if not label or "=" in label or _FOOTNOTE.search(raw_label):
            continue
        for ci in range(len(r)):
            if not YEAR_RE.match(year_row[ci] if ci < len(year_row) else ""):
                continue
            seg = _clean(seg_row[ci]) if ci < len(seg_row) else ""
            entity = _norm_entity(ent_row[ci]) if ci < len(ent_row) else ""
            if not seg or not entity or SKIP_ENTITIES.match(entity):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": dimension, "entity": entity,
                "metric": f"{label} ({unit})" if unit else label, "value": value,
                "financial_year": year_row[ci], "quarter": QUARTER,
                "line_of_business": statement, "class_of_business": _norm_lob(seg),
            })
    return out


def convert_class_matrix(ws, lob, dimension=DIMENSION, fallback_unit="₹Lakh", ent_col=None):
    """year > class > sub-metric column header; rows = insurers/states (58-65, 67-71)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    yr = next((i for i, r in enumerate(grid[:8])
               if sum(1 for c in r if YEAR_RE.match(c)) >= 2), None)
    if yr is None or yr + 2 >= len(grid):
        return []
    ylevel = _ffill(grid[yr])
    clevel = _ffill(grid[yr + 1])
    slevel = grid[yr + 2]                       # sub-metric, one per column
    unit = _find_unit(grid[:yr + 1]) or fallback_unit
    if ent_col is None:
        ent_col = next((i for i, c in enumerate(grid[yr]) if _is_entity_axis(c)), 1)

    out = []
    for r in raw[yr + 3:]:
        cells = [_clean(c) for c in r]
        entity = _norm_entity(cells[ent_col]) if ent_col < len(cells) else ""
        if not entity or SKIP_ENTITIES.match(entity):
            continue
        for ci in range(len(r)):
            if not (YEAR_RE.match(ylevel[ci] if ci < len(ylevel) else "")):
                continue
            cls = _tidy_class(clevel[ci] if ci < len(clevel) else "")
            sub = slevel[ci] if ci < len(slevel) else ""
            if not cls or "total" in cls.lower() or not _clean(sub):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": dimension, "entity": entity,
                "metric": _submetric(sub, unit), "value": value,
                "financial_year": ylevel[ci], "quarter": QUARTER,
                "line_of_business": lob, "class_of_business": cls,
            })
    return out


PERIOD_RE = re.compile(r"^[A-Za-z]{3,9}\.?\s*[\-\s]?\s*\d{4}$")
_MONTH_Q = {"march": ("Q4", -1), "jun": ("Q1", 0), "june": ("Q1", 0),
            "sep": ("Q2", 0), "sept": ("Q2", 0), "september": ("Q2", 0),
            "dec": ("Q3", 0), "december": ("Q3", 0)}


def _period_to_fy_q(text):
    """'June 2015' -> ('2015-16','Q1'); 'March 2016' -> ('2015-16','Q4')."""
    m = re.match(r"([A-Za-z]+)\.?\s*[\-\s]?\s*(\d{4})$", _clean(text))
    if not m or m.group(1).lower() not in _MONTH_Q:
        return None
    q, off = _MONTH_Q[m.group(1).lower()]
    start = int(m.group(2)) + off
    return f"{start}-{str(start + 1)[-2:]}", q


def _group_lob(g):
    g = g.lower()
    if "reinsur" in g:
        return "Reinsurance"
    if "health" in g:
        return "Health"
    return "General"


def convert_quarterly(ws, default_lob, group_lob=False):
    """Insurer x 'Month YYYY' period columns (e.g. Solvency tables 23/49)."""
    raw = list(ws.iter_rows(values_only=True))
    grid = [[_clean(c) for c in r] for r in raw]
    hdr = next((i for i, r in enumerate(grid[:8])
                if sum(1 for c in r if PERIOD_RE.match(c)) >= 2), None)
    if hdr is None:
        return []
    periods = {i: _period_to_fy_q(c) for i, c in enumerate(grid[hdr]) if PERIOD_RE.match(c)}
    periods = {i: p for i, p in periods.items() if p}
    ent_col = next((i for i, c in enumerate(grid[hdr]) if _is_entity_axis(c)),
                   (min(periods) - 1 if periods else 1))
    metric = _metric_from_title(grid[0][0] if grid[0] else "", "")
    out, current_lob = [], default_lob
    for r in raw[hdr + 1:]:
        cells = [_clean(c) for c in r]
        entity = cells[ent_col] if ent_col < len(cells) else ""
        if not entity:
            continue
        has_data = any(_to_number(r[ci]) is not None for ci in periods if ci < len(r))
        if not has_data:                         # group header row
            if group_lob:
                current_lob = _group_lob(entity)
            continue
        if SKIP_ENTITIES.match(entity):
            continue
        entity = _norm_entity(entity)
        for ci, (fy, q) in periods.items():
            if ci >= len(r):
                continue
            value = _to_number(r[ci])
            if value is None:
                continue
            out.append({
                "dimension": DIMENSION, "entity": entity, "metric": metric,
                "value": value, "financial_year": fy, "quarter": q,
                "line_of_business": current_lob, "class_of_business": DEFAULT_CLASS,
            })
    return out


def _open(parts_dir, part):
    path = os.path.join(parts_dir, f"{part}.xlsx")
    if not os.path.exists(path):
        print(f"   [!] {path} missing — skipped")
        return None
    return openpyxl.load_workbook(path, read_only=True, data_only=True)


def main():
    ap = argparse.ArgumentParser(description="Convert Handbook Part tables (Phase 1) to IRIS rows.")
    ap.add_argument("parts_dir", help="Folder containing 'Part I.xlsx' ... 'Part V.xlsx'")
    ap.add_argument("--out", default=os.path.join("knowledge_base", "raw_submissions"))
    ap.add_argument("--name", default="handbook_2024-25_parts.xlsx")
    args = ap.parse_args()

    all_rows = []
    for part, sheet, lob in PHASE1:
        path = os.path.join(args.parts_dir, f"{part}.xlsx")
        if not os.path.exists(path):
            print(f"   [!] {path} missing — skipped")
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        match = next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_table(wb[match], lob)
        metric = rows[0]["metric"] if rows else "?"
        print(f"   {part} t{sheet:>3} [{lob:11}] -> {metric}: {len(rows)} rows")
        all_rows.extend(_stamp(rows, part))

    for part, sheet in PHASE2:
        path = os.path.join(args.parts_dir, f"{part}.xlsx")
        if not os.path.exists(path):
            print(f"   [!] {path} missing — skipped")
            continue
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        match = next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_matrix(wb[match])
        lobs = sorted({r["line_of_business"] for r in rows})
        mets = sorted({r["metric"] for r in rows})
        print(f"   {part} t{sheet:>3} [matrix]      -> {len(rows)} rows | "
              f"LOBs={lobs} | metrics={len(mets)}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob in PHASE2B_TRANSPOSED:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_transposed(wb[match], lob)
        print(f"   {part} t{sheet:>3} [transposed]  -> {len(rows)} rows | "
              f"metrics={len(set(r['metric'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob in PHASE2B_CLASS:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_class_matrix(wb[match], lob)
        cls = sorted({r["class_of_business"] for r in rows})
        print(f"   {part} t{sheet:>3} [{lob:11}] class -> {len(rows)} rows | classes={cls}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob, prefix in PHASE3_YEAR_SUBMETRIC:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_year_submetric(wb[match], lob, "State", prefix)
        print(f"   {part} t{sheet:>3} [State]       -> {len(rows)} rows | "
              f"states={len(set(r['entity'] for r in rows))} | metrics={sorted(set(r['metric'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet in PHASE3_MATRIX:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_matrix(wb[match], dimension="State")
        print(f"   {part} t{sheet:>3} [State matrix]-> {len(rows)} rows | "
              f"states={len(set(r['entity'] for r in rows))} | LOBs={sorted(set(r['line_of_business'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob in PHASE3_STATE_SIMPLE:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_table(wb[match], lob, dimension="State")
        for r in rows:
            r["dimension"] = "State"
        print(f"   {part} t{sheet:>3} [State simple] -> {len(rows)} rows | "
              f"metric={rows[0]['metric'] if rows else '?'}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob in PHASE3_CLASS:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_class_matrix(wb[match], lob, dimension="State")
        cls = sorted({r["class_of_business"] for r in rows})
        print(f"   {part} t{sheet:>3} [State {lob:8}]-> {len(rows)} rows | classes={cls}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, statement in PHASE4:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_transposed(wb[match], statement, track_sections=True,
                                  dimension=FIN_DIMENSION, section_as_class=True)
        secs = sorted({r["class_of_business"] for r in rows})
        print(f"   {part} t{sheet:>3} [Financials/{statement}] -> {len(rows)} rows | sections={len(secs)}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, statement in PHASE7_SEGMENTED:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_segmented_statement(wb[match], statement)
        segs = sorted({r["class_of_business"] for r in rows})
        print(f"   {part} t{sheet:>3} [Seg/{statement}] -> {len(rows)} rows | "
              f"entities={len(set(r['entity'] for r in rows))} | segments={segs}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, report, fb_unit in PHASE8_TRANSPOSED:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_transposed(wb[match], report, track_sections=True,
                                  dimension=FIN_DIMENSION, section_as_class=True, fallback_unit=fb_unit)
        print(f"   {part} t{sheet:>3} [Reports/{report[:26]}] -> {len(rows)} rows | "
              f"entities={len(set(r['entity'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, report, fb_unit in PHASE8_PERIODIC:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_insurer_periodic(wb[match], report, fallback_unit=fb_unit)
        print(f"   {part} t{sheet:>3} [Reports/{report[:26]}] -> {len(rows)} rows | "
              f"line items={len(set(r['metric'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, entity, lob_base, row_as, fb_unit in PHASE13_INDUSTRY_ROWS:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_industry_rows(wb[match], entity, lob_base, row_as=row_as, fallback_unit=fb_unit)
        print(f"   {part} t{sheet:>3} [Industry/{lob_base[:20]}] -> {len(rows)} rows | "
              f"LOBs={len(set(r['line_of_business'] for r in rows))} | metrics={len(set(r['metric'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, entity, lob, fb_unit in PHASE13_INDUSTRY_2KEY:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_industry_2key(wb[match], entity, lob, fallback_unit=fb_unit)
        print(f"   {part} t{sheet:>3} [Industry/{lob[:20]}] -> {len(rows)} rows | "
              f"instruments={len(set(r['metric'] for r in rows))} | classes={sorted(set(r['class_of_business'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob, base_metric, fb_unit in PHASE12_CROSSTAB:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_state_insurer_crosstab(wb[match], lob, base_metric, fallback_unit=fb_unit)
        print(f"   {part} t{sheet:>3} [Insurer/{lob[:22]}] -> {len(rows)} rows | "
              f"insurers={len(set(r['entity'] for r in rows))} | states={len(set(r['class_of_business'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, report, fb_unit in PHASE11_MEASURES:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_channel_measures(wb[match], report, fallback_unit=fb_unit,
                                        dimension=FIN_DIMENSION, class_of_business="General")
        print(f"   {part} t{sheet:>3} [Reports/{report[:24]}] -> {len(rows)} rows | "
              f"entities={len(set(r['entity'] for r in rows))} | items={len(set(r['metric'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, report, fb_unit in PHASE9B_LIFE_HEALTH:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_class_matrix(wb[match], report, dimension=FIN_DIMENSION, fallback_unit=fb_unit)
        print(f"   {part} t{sheet:>3} [Reports/{report[:24]}] -> {len(rows)} rows | "
              f"entities={len(set(r['entity'] for r in rows))}")
        all_rows.extend(_stamp(rows, part, sector="Life"))

    for part, sheet, lob in PHASE10_CHANNEL_MEASURES:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_channel_measures(wb[match], lob)
        print(f"   {part} t{sheet:>3} [Channel/{lob[:22]}] -> {len(rows)} rows | "
              f"channels={len(set(r['entity'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob in PHASE10_CHANNEL_SEGMENT:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_channel_segment(wb[match])
        print(f"   {part} t{sheet:>3} [Channel/General GDP] -> {len(rows)} rows | "
              f"channels={len(set(r['entity'] for r in rows))} | LOBs={sorted(set(r['line_of_business'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob in PHASE10_CHANNEL_CLASS:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            continue
        rows = convert_class_matrix(wb[match], lob, dimension=CHANNEL_DIMENSION, fallback_unit="Nos.", ent_col=0)
        print(f"   {part} t{sheet:>3} [Channel/Health] -> {len(rows)} rows | "
              f"channels={len(set(r['entity'] for r in rows))} | classes={sorted(set(r['class_of_business'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, report, fb_unit in PHASE9_REPORTS_CLASS:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_class_matrix(wb[match], report, dimension=FIN_DIMENSION, fallback_unit=fb_unit)
        secs = sorted({r["class_of_business"] for r in rows})
        print(f"   {part} t{sheet:>3} [Reports/{report[:24]}] -> {len(rows)} rows | stages={len(secs)}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, report, fb_unit in PHASE6_REPORTS:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_transposed(wb[match], report, track_sections=True,
                                  dimension=FIN_DIMENSION, section_as_class=True, fallback_unit=fb_unit)
        print(f"   {part} t{sheet:>3} [Reports/{report[:28]}] -> {len(rows)} rows | "
              f"entities={len(set(r['entity'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    for part, sheet, lob, group_lob in PHASE5_QUARTERLY:
        wb = _open(args.parts_dir, part)
        match = wb and next((s for s in wb.sheetnames if s.strip() == sheet), None)
        if not match:
            print(f"   [!] {part}: sheet {sheet!r} not found — skipped")
            continue
        rows = convert_quarterly(wb[match], lob, group_lob=group_lob)
        lobs = sorted({r["line_of_business"] for r in rows})
        print(f"   {part} t{sheet:>3} [quarterly]   -> {len(rows)} rows | LOBs={lobs} | "
              f"quarters={sorted(set(r['quarter'] for r in rows))}")
        all_rows.extend(_stamp(rows, part))

    if not all_rows:
        sys.exit("No rows produced.")

    # Fold state spelling variants into the canonical state name first.
    for r in all_rows:
        if r["dimension"] == "State":
            r["entity"] = _canon_state(r["entity"])

    # Canonicalise insurer names PER SECTOR (driven by the index): a short/variant
    # name folds into the fullest name whose words are a superset, but only within
    # its own sector — so "Bajaj Allianz" in a Life table resolves to Bajaj Allianz
    # *Life*, never the General arm. Names that stay ambiguous (a short form that
    # could be >1 company even inside the sector) are left as-is and reported.
    by_sector = {}
    for r in all_rows:
        by_sector.setdefault(r["_sector"], set()).add(r["entity"])
    resolve, ambiguous = {}, {}
    for sec, nameset in by_sector.items():
        names = sorted(nameset)
        sig_names = {}
        for n in names:
            sig_names.setdefault(_sig(n), set()).add(n)
        sigs = list(sig_names)
        for n in names:
            key = SECTOR_ALIASES.get((sec, _state_key(n)))
            if key:
                resolve[(sec, n)] = key
                continue
            s = _sig(n)
            # Fold into the MAXIMAL superset name (transitive), so chains like
            # Cholamandalam -> Cholamandalam MS -> ...General all land on the full
            # name. Ambiguous only when there are several incomparable maximals.
            cands = [o for o in sigs if s <= o]
            maximal = [o for o in cands if not any(o < p for p in cands)]
            if len(maximal) == 1:
                resolve[(sec, n)] = max(sig_names[maximal[0]], key=_name_quality)
            else:
                resolve[(sec, n)] = max(sig_names[s], key=_name_quality)
                ambiguous.setdefault((sec, n), sorted({max(sig_names[m], key=_name_quality) for m in maximal}))
    merged = sum(1 for r in all_rows if resolve[(r["_sector"], r["entity"])] != r["entity"])
    for r in all_rows:
        r["entity"] = resolve[(r["_sector"], r["entity"])]
        r["metric"] = _canon_metric(r["metric"])
        r["financial_year"] = _to_fy(r["financial_year"])   # unify calendar -> FY
        if r["line_of_business"] == "General":              # disambiguate the
            r["line_of_business"] = "General (All Segments)"  # company-wide total
        del r["_sector"]
    print(f"\n[i] Canonicalised insurer names per sector: folded {merged} variants.")
    if ambiguous:
        print(f"[!] {len(ambiguous)} ambiguous name(s) left as-is (need a rule):")
        for (sec, n), opts in sorted(ambiguous.items()):
            print(f"      [{sec}] {n!r} -> could be {opts}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, args.name)
    df = pd.DataFrame(all_rows, columns=[
        "dimension", "entity", "metric", "value", "financial_year",
        "quarter", "line_of_business", "class_of_business",
    ])
    df.to_excel(out_path, index=False)
    print(f"\n[+] Wrote {len(df)} rows -> {out_path}")
    print(f"    insurers: {df['entity'].nunique()} | metrics: {df['metric'].nunique()} | "
          f"years: {df['financial_year'].min()}..{df['financial_year'].max()}")


if __name__ == "__main__":
    main()
