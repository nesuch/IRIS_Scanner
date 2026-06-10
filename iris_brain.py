import sqlite3
import pandas as pd # type: ignore
import os
import re
from datetime import datetime
import glob
import difflib
import io  # Required for Excel Export
from pathlib import Path
from typing import cast

# --- NLTK SETUP ---
try:
    from nltk.stem import PorterStemmer # type: ignore
    stemmer = PorterStemmer()
    USE_NLTK = True
    print("[+] NLTK loaded.")
except ImportError:
    USE_NLTK = False
    print("[!] NLTK not found. Using manual stemmer.")

# ==========================================
# 1. CONFIGURATION
# ==========================================
DB_NAME = os.path.abspath(os.getenv("IRIS_DB_PATH", "iris.db"))
KB_FOLDER = "knowledge_base"
RAW_SUBMISSIONS_FOLDER = os.path.join(KB_FOLDER, "raw_submissions")

# Ensure folders exist (for admin upload staging)
if not os.path.exists(RAW_SUBMISSIONS_FOLDER): os.makedirs(RAW_SUBMISSIONS_FOLDER)

DOC_HIERARCHY = { "ACT": 1, "REGULATION": 2, "MASTER": 3, "CIRCULAR": 4, "GUIDELINE": 5, "UNKNOWN": 99 }
GREETINGS = { "hi", "hello", "hey", "iris", "help", "greetings" }

SYNONYM_MAP = {
    "money back": ["refund"], "older": ["senior"], "pay": ["premium"],
    "ncb": ["no", "claim", "bonus"], "doc": ["hospital", "doctor"],
    "baby": ["newborn"], "kid": ["child"], "many": ["multiple"], "wait": ["waiting"],
    "rejection": ["repudiation", "reject", "denial"]
}

STOP_WORDS = { "is", "am", "are", "can", "i", "get", "a", "the", "to", "for", "in", "on", "of", "about", "me", "does", "will", "should" }
STOPWORDS_STRONG = { "all", "any", "every", "shall", "may", "must", "including", "such", "other" }
MIN_KEYWORD_LENGTH = 2

# Unit Mapping for nicer headers
UNIT_MAP = {
    "Solvency Margin": "Ratio",
    "Solvency Ratio": "Ratio",
    "GDPI": "Cr", 
    "Net Profit": "Cr",
    "Incurred Claims Ratio": "%", 
    "Net Incurred Claims Ratio": "%",
    "Gross Commission Ratio": "%", 
    "Combined Ratio": "%",
    "Repudiation Ratio (Nos)": "%", 
    "Repudiation Ratio (Amount)": "%",
    "Expense of Management to NWP Ratio": "%",
    "Expense of Management to GDP Ratio": "%",
    "Liquidity Ratio": "Ratio"
}

UNIFIED_DF = pd.DataFrame()

# ==========================================
# 2. CORE UTILS (TEXT SEARCH)
# ==========================================
def get_stem(word):
    word = word.lower()
    if USE_NLTK: return stemmer.stem(word)
    if len(word) < 4: return word 
    if word.endswith("ing"): return word[:-3]
    return word

def get_doc_type(filename):
    fname = filename.upper()
    if "ACT" in fname: return "ACT"
    if "REGULATION" in fname: return "REGULATION"
    if "MASTER" in fname: return "MASTER"
    if "CIRCULAR" in fname: return "CIRCULAR"
    return "UNKNOWN"

# ==========================================
# 3. KNOWLEDGE BASE LOADER (SQL INTEGRATED)
# ==========================================
ALL_UNIQUE_TAGS = set()
ALL_DOC_NAMES = set()
TAG_INGREDIENTS = {} 
KNOWN_VOCAB = set()
NORMALIZED_TAG_LOOKUP = {}
KB_CACHE_DF = None

def normalize_tag_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).lower().replace("_", " "))).strip()

def load_knowledge_base(force_reload=False):
    global ALL_UNIQUE_TAGS, ALL_DOC_NAMES, TAG_INGREDIENTS, KNOWN_VOCAB, NORMALIZED_TAG_LOOKUP, KB_CACHE_DF
    
    # Pre-load financial data if needed
    if force_reload: load_master_data_engine()

    if not force_reload and KB_CACHE_DF is not None and not KB_CACHE_DF.empty:
        return KB_CACHE_DF

    # 1. Fetch from SQL
    try:
        conn = sqlite3.connect(DB_NAME)
        # Load directly into DataFrame
        df = pd.read_sql_query("SELECT * FROM regulatory_clauses", conn)
        conn.close()
    except Exception as e:
        print(f"[!] Error loading regulatory_clauses from DB: {e}")
        return pd.DataFrame()

    if df.empty:
        KB_CACHE_DF = df
        print("[!] Knowledge Base Loaded: 0 regulatory clauses from SQL.")
        return df
    print(f"[+] Knowledge Base Loaded: {len(df)} regulatory clauses from SQL.")

    # 2. Map SQL columns to Application Logic (PascalCase)
    df = df.rename(columns={
        "source_doc": "Source_Doc",
        "doc_category": "Doc_Category",
        "doc_type": "Doc_Type",
        "clause_id": "Clause_ID",
        "clause_text": "Clause_Text",
        "context_header": "Context_Header",
        "regulatory_tags": "Regulatory_Tags",
        "priority": "Priority",
        "is_header": "Is_Header"
    })

    # 3. Build Vocab / Tags (Only if not already built)
    if not ALL_UNIQUE_TAGS or force_reload:
        _rebuild_vocab(df)

    KB_CACHE_DF = df
    return df


_KB_RENAME = {
    "source_doc": "Source_Doc", "doc_category": "Doc_Category", "doc_type": "Doc_Type",
    "clause_id": "Clause_ID", "clause_text": "Clause_Text", "context_header": "Context_Header",
    "regulatory_tags": "Regulatory_Tags", "priority": "Priority", "is_header": "Is_Header",
}


def refresh_kb():
    """Re-read regulatory_clauses into the in-memory KB and rebuild the tag vocab,
    WITHOUT reloading the financial engine. Used after an import/bulk change."""
    global KB_CACHE_DF
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM regulatory_clauses", conn)
    conn.close()
    df = df.rename(columns=_KB_RENAME)
    KB_CACHE_DF = df
    _rebuild_vocab(df)
    return df


def _rebuild_vocab(df):
    """(Re)build the tag vocabulary / autocomplete data from the clause DataFrame.
    Called on load and after an in-app tag edit so search reflects changes."""
    ALL_UNIQUE_TAGS.clear(); ALL_DOC_NAMES.clear(); TAG_INGREDIENTS.clear(); KNOWN_VOCAB.clear(); NORMALIZED_TAG_LOOKUP.clear()

    for k in SYNONYM_MAP.keys(): KNOWN_VOCAB.add(k)
    for v_list in SYNONYM_MAP.values():
        for v in v_list: KNOWN_VOCAB.add(v)

    for _, row in df.iterrows():
        ALL_DOC_NAMES.add(row["Source_Doc"])
        row_tags = str(row["Regulatory_Tags"])
        if row_tags:
            for tag in [t.strip().lower() for t in row_tags.split(",")]:
                clean_tag = tag.replace("_", " ")
                if len(clean_tag) < 2: continue
                ALL_UNIQUE_TAGS.add(clean_tag)
                normalized = normalize_tag_text(clean_tag)
                if normalized:
                    NORMALIZED_TAG_LOOKUP[normalized] = clean_tag
                ingredients = set()
                for w in re.findall(r"\w+", clean_tag):
                    if w in STOP_WORDS: continue
                    KNOWN_VOCAB.add(str(w))
                    ingredients.add(get_stem(w))
                if ingredients: TAG_INGREDIENTS[clean_tag] = ingredients


def clause_tags(clause_id, source):
    """Current tags (list) for one clause, from the in-memory knowledge base."""
    df = KB_CACHE_DF
    if df is None or df.empty: return []
    mask = (df["Clause_ID"].astype(str) == str(clause_id)) & (df["Source_Doc"].astype(str) == str(source))
    sub = df.loc[mask, "Regulatory_Tags"]
    if not len(sub): return []
    return [t.strip() for t in str(sub.iloc[0]).split(",") if t.strip()]


def clause_html(clause_id, source):
    """The rich edited HTML body for a clause, or '' if it hasn't been edited
    in-app (in which case the plain clause_text is rendered instead)."""
    df = KB_CACHE_DF
    if df is None or df.empty or "clause_html" not in df.columns:
        return ""
    mask = (df["Clause_ID"].astype(str) == str(clause_id)) & (df["Source_Doc"].astype(str) == str(source))
    sub = df.loc[mask, "clause_html"]
    if not len(sub):
        return ""
    v = sub.iloc[0]
    return str(v) if pd.notna(v) and str(v).strip() else ""


def update_clause_content(clause_id, source, html, text, editor=None):
    """Admin in-app rich edit of a clause. Snapshots the current content to
    clause_versions (revert), persists the new HTML + derived plain text to SQL,
    and updates the in-memory KB so search/render reflect it. Returns True, or
    None if the clause wasn't found."""
    global KB_CACHE_DF
    html = html or ""
    text = (text or "").strip()
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(DB_NAME)
    cur = conn.execute(
        "SELECT clause_html, clause_text, regulatory_tags FROM regulatory_clauses WHERE clause_id=? AND source_doc=?",
        (str(clause_id), str(source)))
    prev = cur.fetchone()
    if prev is None:
        conn.close()
        return None
    conn.execute(
        "INSERT INTO clause_versions (clause_id, source_doc, html, body_text, tags, edited_by, edited_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (str(clause_id), str(source), prev[0], prev[1], prev[2], editor or "", now))
    conn.execute(
        "UPDATE regulatory_clauses SET clause_html=?, clause_text=?, updated_at=?, updated_by=? "
        "WHERE clause_id=? AND source_doc=?",
        (html, text, now, editor or "", str(clause_id), str(source)))
    conn.commit()
    conn.close()
    if KB_CACHE_DF is not None and not KB_CACHE_DF.empty:
        if "clause_html" not in KB_CACHE_DF.columns:
            KB_CACHE_DF["clause_html"] = None
        mask = (KB_CACHE_DF["Clause_ID"].astype(str) == str(clause_id)) & (KB_CACHE_DF["Source_Doc"].astype(str) == str(source))
        KB_CACHE_DF.loc[mask, "clause_html"] = html
        KB_CACHE_DF.loc[mask, "Clause_Text"] = text
    return True


def delete_document(source, editor=None):
    """Delete an entire document and all its clauses. Snapshots each clause to
    clause_versions first (recoverable), then refreshes the in-memory KB.
    Returns the number of clauses removed."""
    global KB_CACHE_DF
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT clause_id, clause_html, clause_text, regulatory_tags FROM regulatory_clauses WHERE source_doc=?",
        (str(source),)).fetchall()
    if not rows:
        conn.close()
        return 0
    tag = (editor or "") + " (doc delete)"
    for r in rows:
        conn.execute(
            "INSERT INTO clause_versions (clause_id, source_doc, html, body_text, tags, edited_by, edited_at) "
            "VALUES (?,?,?,?,?,?,?)", (r[0], str(source), r[1], r[2], r[3], tag, now))
    conn.execute("DELETE FROM regulatory_clauses WHERE source_doc=?", (str(source),))
    conn.commit()
    conn.close()
    refresh_kb()
    return len(rows)


def update_clause_tags(clause_id, source, tags):
    """Admin in-app edit: persist a clause's tags to SQL, update the in-memory KB
    and rebuild the tag vocabulary so search reflects it immediately. Returns the
    new tag list, or None if the clause wasn't found."""
    global KB_CACHE_DF
    tags = (tags or "").strip()
    conn = sqlite3.connect(DB_NAME)
    cur = conn.execute(
        "UPDATE regulatory_clauses SET regulatory_tags=? WHERE clause_id=? AND source_doc=?",
        (tags, str(clause_id), str(source)))
    conn.commit(); changed = cur.rowcount; conn.close()
    if not changed:
        return None
    if KB_CACHE_DF is not None and not KB_CACHE_DF.empty:
        mask = (KB_CACHE_DF["Clause_ID"].astype(str) == str(clause_id)) & (KB_CACHE_DF["Source_Doc"].astype(str) == str(source))
        KB_CACHE_DF.loc[mask, "Regulatory_Tags"] = tags
        _rebuild_vocab(KB_CACHE_DF)
    return [t.strip() for t in tags.split(",") if t.strip()]

def get_autocomplete_data():
    vocab = {"CONCEPTS": []}
    concepts = set(ALL_UNIQUE_TAGS)

    # Keep suggestion list anchored to real tags + curated synonym keys only.
    # Do not include raw synonym value tokens (e.g. "claim") because they can
    # appear as suggestions but not resolve to a meaningful tag result.
    concepts.update(SYNONYM_MAP.keys())

    vocab["CONCEPTS"] = sorted(list(concepts))
    return vocab

# ==========================================
# 4. TEXT SEARCH LOGIC
# ==========================================
def filter_df_by_module(df, module) -> pd.DataFrame:
    if df is None: return pd.DataFrame()
    if df.empty: return df
    if module == "health": return df[df["Doc_Category"] == "HEALTH"]
    elif module == "life": return df[df["Doc_Category"] == "LIFE"]
    elif module == "nonlife": return df[df["Doc_Category"] == "NONLIFE"]
    elif module == "data": return pd.DataFrame(columns=df.columns)
    else: return df

def check_greeting(query):
    return re.sub(r'[^\w\s]', '', query.lower().strip()) in GREETINGS

def get_clean_keywords(query: str):
    query = query.lower().replace("_", " ")

    # Strict mode: if user selects/types an exact known tag phrase,
    # do not expand to ingredient-matched related tags.
    normalized_query = normalize_tag_text(query)
    if normalized_query in NORMALIZED_TAG_LOOKUP:
        canonical = NORMALIZED_TAG_LOOKUP[normalized_query]
        return [(canonical, canonical)]

    # Handle minor punctuation/hyphen variations from selected suggestions
    if len(normalized_query.split()) >= 4 and NORMALIZED_TAG_LOOKUP:
        close_norm = difflib.get_close_matches(normalized_query, list(NORMALIZED_TAG_LOOKUP.keys()), n=1, cutoff=0.95)
        if close_norm:
            canonical = NORMALIZED_TAG_LOOKUP[close_norm[0]]
            return [(canonical, canonical)]

    final_tuples = []
    raw_words = re.findall(r'\w+', query)
    soup_ingredients = set()
    
    for w in raw_words:
        if w in STOP_WORDS: continue
        valid_word = w
        corrected_word = None

        if w not in KNOWN_VOCAB:
            matches = difflib.get_close_matches(w, list(KNOWN_VOCAB), n=1, cutoff=0.8)
            if matches:
                valid_word = matches[0]
                corrected_word = valid_word

        # Always keep user-entered token so deep scan can offer exact-user intent.
        raw_stem = get_stem(w)
        if len(raw_stem) >= MIN_KEYWORD_LENGTH:
            final_tuples.append((w, raw_stem)); soup_ingredients.add(raw_stem)

        # Also keep smart corrected token (if any) for better typo recovery.
        if corrected_word:
            corr_stem = get_stem(corrected_word)
            if len(corr_stem) >= MIN_KEYWORD_LENGTH:
                final_tuples.append((corrected_word, corr_stem)); soup_ingredients.add(corr_stem)

        if valid_word in SYNONYM_MAP:
            for s in SYNONYM_MAP[valid_word]: soup_ingredients.add(get_stem(s))

    sorted_tags = sorted(TAG_INGREDIENTS.keys(), key=lambda x: len(x), reverse=True)
    for tag in sorted_tags:
        required = TAG_INGREDIENTS[tag]
        match_count = sum(1 for req in required if any((s == req) or (len(s) > 3 and req.startswith(s)) for s in soup_ingredients))
        if match_count == len(required): 
            final_tuples.append((tag, tag)) 

    final_tuples.sort(key=lambda x: (x[0] != x[1], len(x[1])), reverse=True)
    seen = set(); unique = []
    for raw, clean in final_tuples:
        if clean not in seen: seen.add(clean); unique.append((raw, clean))
    return unique

def sort_matches(matches):
    return sorted(matches, key=lambda x: (x['priority'], x['source'], x['id']))

def search_tags_only(keyword_tuples, df, module="universal"):
    scoped_df: pd.DataFrame = filter_df_by_module(df, module)
    if scoped_df.empty: return []

    matches = []
    detected_tags = [t[1] for t in keyword_tuples]
    
    for _, row in scoped_df.iterrows():
        if row.get("Is_Header"): continue
        raw_tags = str(row.get("Regulatory_Tags", "")).lower()
        if not raw_tags: continue
        
        tag_list = [t.strip().replace("_", " ") for t in raw_tags.split(",")]
        
        found = False
        for target in detected_tags:
            target_stem = get_stem(target)
            for doc_tag in tag_list:
                if doc_tag == target or normalize_tag_text(doc_tag) == normalize_tag_text(target):
                    found = True; break
                if target_stem == get_stem(doc_tag) and len(target.split()) == 1 and len(doc_tag.split()) == 1:
                    found = True; break
            if found: break
        
        if found:
            matches.append({
                "source": row.get("Source_Doc", "UNKNOWN"),
                "type": row.get("Doc_Type", "UNKNOWN"),
                "priority": row.get("Priority", 99),
                "id": str(row.get("Clause_ID", "")).strip(),
                "header": row.get("Context_Header", ""),
                "raw_text": str(row.get("Clause_Text", ""))
            })
    return sort_matches(matches)

def _num_key(s):
    """Strip everything but letters/digits for space/bracket-insensitive matching:
    '64 V B' / '64vb' / '64(VB)' all collapse to '64vb'."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def search_by_clause_number(raw_query, df, sources=None, limit=12):
    """Find clauses by their number (e.g. 64VB, 4(1)(i), 110), ignoring spaces and
    brackets. Triggered when the user prefixes the query with '/'. Matches the
    clause id and the clause's leading text; clause-id hits rank first."""
    if df is None or df.empty:
        return []
    q = _num_key(str(raw_query).lstrip("/"))
    if not q:
        return []
    scoped = df
    if sources:
        scoped = df[df["Source_Doc"].isin(set(sources))]

    def _payload(row):
        return {
            "source": row.get("Source_Doc", "UNKNOWN"),
            "type": row.get("Doc_Type", "UNKNOWN"),
            "priority": row.get("Priority", 99),
            "id": str(row.get("Clause_ID", "")).strip(),
            "header": row.get("Context_Header", ""),
            "raw_text": str(row.get("Clause_Text", "")),
        }

    # Tier the matches: a clause's OWN number (exact, then prefix) is what the user
    # means by "/64". Body mentions ("clause 55 cites section 64") are only a
    # fallback used when nothing is actually numbered like the query.
    exact, prefix, loose = [], [], []
    for _, row in scoped.iterrows():
        if row.get("Is_Header"):
            continue
        text = str(row.get("Clause_Text", ""))
        cid = str(row.get("Clause_ID", ""))
        lead = re.match(r"^[\W_]*([0-9]+[a-z]*)", text.lower())
        lead_num = _num_key(lead.group(1)) if lead else ""
        cid_tail = _num_key(cid.rsplit("-", 1)[-1])
        if q == lead_num or q == cid_tail:
            exact.append(_payload(row))
        elif lead_num.startswith(q) or cid_tail.startswith(q):
            prefix.append((len(lead_num or cid_tail), _payload(row)))
        elif len(q) >= 2 and (q in _num_key(cid) or q in _num_key(text)):
            loose.append(_payload(row))
    prefix.sort(key=lambda x: x[0])
    primary = exact + [p for _, p in prefix]
    return (primary or loose)[:limit]


def deep_scan_brain(keyword_tuples, df, exclude_ids=None, module="universal"):
    scoped_df: pd.DataFrame = filter_df_by_module(df, module)
    if scoped_df.empty: return []

    search_stems = set()
    for raw, clean in keyword_tuples:
        if clean not in STOPWORDS_STRONG:
            search_stems.add(get_stem(clean))
            if clean in SYNONYM_MAP:
                for syn in SYNONYM_MAP[str(clean)]:
                    search_stems.add(get_stem(syn))

    exclude_set: set[str] = set(exclude_ids) if exclude_ids else set()
    matches = []
    
    for _, row in scoped_df.iterrows():
        if row.get("Is_Header"): continue
        c_id = str(row.get("Clause_ID", "")).strip()
        if c_id in exclude_set: continue
        
        text = str(row.get("Clause_Text", "")).lower()
        found = False
        for stem in search_stems:
            if re.search(rf"\b{re.escape(stem)}\w*", text): found = True; break
        
        if found:
            matches.append({
                "source": row.get("Source_Doc", "UNKNOWN"),
                "type": row.get("Doc_Type", "UNKNOWN"),
                "priority": row.get("Priority", 99),
                "id": c_id,
                "header": row.get("Context_Header", ""),
                "raw_text": str(row.get("Clause_Text", ""))
            })
    return sort_matches(matches)

# ==========================================
# 5. EARLY WARNING SYSTEM (RISK LOGIC - TRENDS)
# ==========================================
def _analyze_risk(selected_entities, dimension, selected_years=None):
    """
    Applies thresholds AND trend analysis for Data Explorer alerts.
    Only applicable if Dimension is 'Insurer'.

    Alerts are strictly scoped to the user's current selection: only the chosen
    entities (companies) are analyzed, and when specific financial years are
    selected the analysis window is restricted to those years too.
    """
    if UNIFIED_DF.empty or not selected_entities: return []
    if dimension != "Insurer": return []

    alerts = []
    # Filter by specific Entities (Insurers) — scope EWS to the selected companies only.
    df = UNIFIED_DF[UNIFIED_DF['Entity'].isin(selected_entities)].copy()

    # Scope to the selected financial years when the user has filtered them.
    if selected_years:
        df = df[df['Financial_Year'].isin(selected_years)]
    if df.empty:
        return []

    # Ensure correct sorting for trend analysis
    df['Sortable_Year'] = df['Financial_Year'].astype(str).str.extract(r'(\d+)').astype(float)
    df = df.sort_values(by=['Entity', 'Metric', 'Sortable_Year', 'Quarter'])

    SAHI_INSURERS = ["Star", "Care", "Aditya Birla", "Niva Bupa", "Manipal", "Galaxy", "Narayana"]

    def _is_pct(m):
        m = m.lower()
        return ("per cent" in m) or ("ratio" in m) or ("%" in m)

    def _fmt(v, pct):
        return f"{v:,.2f}%" if pct else f"{v:,.2f}"

    # Group by the FULL series (entity + metric + line of business + class) so we
    # never mix, e.g., Fire and Motor "Claims Incurred" into one trend. Each series
    # is then collapsed to one value per financial year.
    keys = ['Entity', 'Metric', 'Line_of_Business', 'Class_of_Business']
    keys = [k for k in keys if k in df.columns]
    for grp_keys, group in df.groupby(keys):
        entity = grp_keys[0] if isinstance(grp_keys, tuple) else grp_keys
        metric = grp_keys[1] if isinstance(grp_keys, tuple) else ""
        group = cast(pd.DataFrame, group)
        pct = _is_pct(metric)

        # one row per financial year (latest quarter), chronologically
        series = (group.dropna(subset=['Value'])
                       .drop_duplicates(subset=['Financial_Year'], keep='last')
                       .sort_values('Sortable_Year'))
        if series.empty:
            continue
        val = float(series.iloc[-1]['Value'])

        # THRESHOLD CHECKS (latest value) — only on the relevant metric kinds.
        if "Solvency" in metric and val < 1.5:
            alerts.append({"level": "critical",
                           "msg": f"Regulatory Violation: {entity} - Solvency {val:.2f} < 1.5 limit"})
        elif "Expense" in metric and pct:
            limit = 35 if any(s in entity for s in SAHI_INSURERS) else 30
            if val > limit:
                alerts.append({"level": "critical",
                               "msg": f"Regulatory Violation: {entity} - EoM {val:.2f}% exceeds {limit}% limit"})
        elif "Repudiation" in metric and pct and val > 10:
            alerts.append({"level": "warning",
                           "msg": f"High Repudiation: {entity} - {metric} {val:.2f}% exceeds 10% limit"})

        # TREND ANALYSIS — needs 3 distinct years.
        if len(series) < 3:
            continue
        vals = [float(v) for v in series['Value'].tolist()[-3:]]
        years = series['Financial_Year'].tolist()[-3:]

        # Rising "bad" ratios only (ICR, repudiation, expense, combined) — never on
        # absolute ₹Crore / Nos. metrics (those legitimately grow with the business).
        if pct and any(x in metric for x in ["Repudiation", "Claims", "Expense", "Combined", "Ratio"]):
            if vals[0] < vals[1] < vals[2] and (vals[2] - vals[0]) >= 3:
                alerts.append({
                    "level": "warning",
                    "msg": f"Rising Trend: {entity} - {metric} rose from {_fmt(vals[0], pct)} "
                           f"to {_fmt(vals[2], pct)} ({years[0]} to {years[2]})."
                })

        if "Solvency" in metric and vals[0] > vals[1] > vals[2] and (vals[0] - vals[2]) >= 0.2:
            alerts.append({
                "level": "warning",
                "msg": f"Deteriorating Solvency: {entity} - dropped from {vals[0]:.2f} "
                       f"to {vals[2]:.2f} ({years[0]} to {years[2]})."
            })

    return alerts

# ==========================================
# 6. DATA INTELLIGENCE ENGINE (DIMENSION AWARE + CSV SUPPORT)
# ==========================================

def _get_doc_category_from_path(file_path, clean_filename):
    normalized_parts = {part.lower() for part in Path(file_path).parts}

    if "health" in normalized_parts:
        return "HEALTH"
    if {"nonlife", "non-life", "non_life", "general"} & normalized_parts:
        return "NONLIFE"
    if "life" in normalized_parts:
        return "LIFE"

    tokens = set(re.split(r"[^A-Z0-9]+", clean_filename.upper()))
    health_hints = {"HEALTH", "PRODUCT", "PPHI", "HOSPITAL", "MEDICLAIM"}
    life_hints = {"LIFE", "ULIP", "ANNUITY", "PENSION"}
    nonlife_hints = {"MOTOR", "FIRE", "MARINE", "GENERAL", "MISCELLANEOUS",
                     "LIABILITY", "ENGINEERING", "PROPERTY", "NONLIFE"}
    if tokens & health_hints:
        return "HEALTH"
    if tokens & nonlife_hints:
        return "NONLIFE"
    if tokens & life_hints:
        return "LIFE"
    return "OTHER"


def aggregate_regulatory_documents():
    # Rebuilds regulatory_clauses from all Excel files under knowledge_base/.
    # Enables Admin sync to refresh regulatory docs without manual migrate_raw.py runs.
    all_files = glob.glob(os.path.join(KB_FOLDER, "**", "*.xlsx"), recursive=True)
    all_files += glob.glob(os.path.join(KB_FOLDER, "**", "*.xls"), recursive=True)
    all_files += glob.glob(os.path.join(KB_FOLDER, "**", "*.csv"), recursive=True)

    doc_files = []
    for file_path in all_files:
        if os.path.basename(file_path).startswith("~$"):
            continue
        if "raw_submissions" in file_path or "master_data" in file_path:
            continue
        doc_files.append(file_path)

    if not doc_files:
        return "[-] No regulatory Excel files found under knowledge_base/."

    total_rows = 0
    total_files = 0

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("DELETE FROM regulatory_clauses")

    for file_path in doc_files:
        filename = os.path.basename(file_path)
        try:
            normalized_frames = []
            if file_path.lower().endswith(".csv"):
                normalized_frames.append(pd.read_csv(file_path).fillna(""))
            else:
                sheet_map = pd.read_excel(file_path, sheet_name=None)
                normalized_frames.extend([sheet_df.fillna("") for sheet_df in sheet_map.values()])

            if not normalized_frames:
                continue

            df = pd.concat(normalized_frames, ignore_index=True)
            if df.empty:
                continue

            source_doc = re.sub(r"\.(xlsx|xls|csv|xlxs)$", "", filename, flags=re.IGNORECASE).replace("_", " ").upper()
            category = _get_doc_category_from_path(file_path, source_doc)
            doc_type = get_doc_type(source_doc)
            priority = DOC_HIERARCHY.get(doc_type, 99)

            cols_norm = {str(col).strip().lower(): col for col in df.columns}
            clause_text_col = cols_norm.get("clause_text") or cols_norm.get("clause text") or cols_norm.get("text") or cols_norm.get("clause")
            clause_id_col = cols_norm.get("clause_id") or cols_norm.get("clause id") or cols_norm.get("id")
            context_col = cols_norm.get("context_header") or cols_norm.get("context header")
            tags_col = cols_norm.get("regulatory_tags") or cols_norm.get("regulatory tags")
            if not clause_text_col:
                continue

            insert_rows = []
            for _, row in df.iterrows():
                clause_text = str(row.get(clause_text_col, "")).strip()
                if not clause_text:
                    continue

                is_header = 0
                if (clause_text.lower().startswith("chapter") or clause_text.lower().startswith("part")) and len(clause_text) < 120:
                    is_header = 1

                insert_rows.append((
                    source_doc.title(),
                    category,
                    doc_type,
                    str(row.get(clause_id_col, "")).strip() if clause_id_col else "",
                    clause_text,
                    str(row.get(context_col, "General")) if context_col else "General",
                    str(row.get(tags_col, "")) if tags_col else "",
                    priority,
                    is_header
                ))

            if insert_rows:
                c.executemany(
                    "INSERT INTO regulatory_clauses (source_doc, doc_category, doc_type, clause_id, clause_text, context_header, regulatory_tags, priority, is_header) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    insert_rows,
                )
                total_files += 1
                total_rows += len(insert_rows)

        except Exception as e:
            print(f"[!] Error processing regulatory file {filename}: {e}")

    conn.commit()
    conn.close()

    load_knowledge_base(force_reload=True)
    return f"[+] Regulatory sync complete: {total_files} files ({total_rows} clauses)."

def aggregate_submissions():
    """
    Reads Excel AND CSV files from raw_submissions and INSERTs them into SQL DB.
    Supports 'dimension' column. Defaults to 'Insurer' if not present.
    Automatically converts '-' quarters to 'Annual'.
    """
    # --- UPDATED: Look for files in ALL subdirectories using os.walk ---
    all_files = []
    for root, dirs, files in os.walk(RAW_SUBMISSIONS_FOLDER):
        for file in files:
            if file.startswith("~$"): continue # Skip Excel temp files
            if file.endswith(".xlsx") or file.endswith(".csv"):
                all_files.append(os.path.join(root, file))
    
    if not all_files: 
        return "[-] No new files found in 'raw_submissions' or subfolders."

    total_rows: int = 0
    total_files: int = 0

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # --- MIGRATION: Ensure columns exist in DB ---
        try: c.execute("ALTER TABLE financial_metrics ADD COLUMN dimension TEXT DEFAULT 'Insurer'")
        except sqlite3.OperationalError: pass 
        try: c.execute("ALTER TABLE financial_metrics ADD COLUMN source_file TEXT")
        except sqlite3.OperationalError: pass

        # --- CLEAN SLATE: Wipe table before import to prevent duplicates ---
        c.execute("DELETE FROM financial_metrics")

        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            try:
                # --- READ BASED ON EXTENSION ---
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)

                if df.empty: continue

                # Standardize columns to match SQL
                df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]
                
                rows_to_insert = []
                for _, row in df.iterrows():
                    # --- SMART ADAPTER LOGIC ---
                    # 1. Detect Dimension
                    dim_raw = row.get('dimension')
                    if not dim_raw:
                        # Infer based on available columns
                        if 'insurer' in df.columns: dim_raw = 'Insurer'
                        elif 'state' in df.columns: dim_raw = 'State'
                        elif 'tpa' in df.columns: dim_raw = 'TPA'
                        else: dim_raw = 'Insurer'

                    # 2. Detect Entity Name
                    entity_raw = (row.get('insurer') or 
                                  row.get('entity') or 
                                  row.get('state') or 
                                  row.get('tpa') or 
                                  "Unknown")

                    # 3. Clean and Standardize (Strip Whitespace)
                    dim = str(dim_raw).strip()
                    entity_name = str(entity_raw).strip()
                    metric_name = str(row.get('metric', '')).strip()
                    
                    # 4. Handle Quarter
                    quarter_val = str(row.get('quarter', 'Annual')).strip()
                    if quarter_val in ['-', 'nan', 'None', '', 'nan']: 
                        quarter_val = 'Annual'

                    # Skip invalid rows
                    if not metric_name or pd.isna(row.get('value')): continue

                    rows_to_insert.append((
                        entity_name, 
                        str(row.get('financial_year')).replace('.0','').strip(), 
                        quarter_val,
                        metric_name, 
                        row.get('value'), 
                        str(row.get('line_of_business', 'General')).strip(), 
                        str(row.get('class_of_business', 'General')).strip(),
                        dim,
                        filename # Task 3: Store Source File
                    ))
                
                if rows_to_insert:
                    c.executemany('''
                        INSERT INTO financial_metrics (insurer, financial_year, quarter, metric, value, line_of_business, class_of_business, dimension, source_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', rows_to_insert)
                    
                    total_files += 1  # type: ignore
                    total_rows += len(rows_to_insert)  # type: ignore
                
            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")

        conn.commit()
        conn.close()
        
        # Reload the engine to reflect new data
        load_master_data_engine()
        return f"[+] Success! Synced {total_files} files ({total_rows} rows) into SQL Database."

    except Exception as e:
        return f"Error syncing data: {e}"

_ENTITY_FILLER = {"insurance", "co", "company", "ltd", "limited", "india", "the",
                  "and", "assurance", "services", "branch", "branches", "of"}


def _ent_sig(name):
    toks = re.sub(r"[^a-z0-9 ]", " ", str(name).lower()).split()
    return frozenset(t for t in toks if t not in _ENTITY_FILLER)


def _ent_quality(n):
    return (0 if n.isupper() else 1, 1 if re.search(r"\bLtd\.?$", n) else 0, len(n))


def _canonicalize_insurer_entities():
    """Merge insurer-name spelling variants that arrive from different sources
    (e.g. SAHI 'unified_database.csv' vs Handbook) so the same company isn't
    listed twice. Scoped to the Insurer/Financials dimensions; folds a name into
    the maximal superset-by-words name (transitive)."""
    global UNIFIED_DF
    if UNIFIED_DF.empty or 'Dimension' not in UNIFIED_DF.columns:
        return
    mask = UNIFIED_DF['Dimension'].isin(['Insurer', 'Financials'])
    names = sorted(UNIFIED_DF.loc[mask, 'Entity'].dropna().unique().tolist())
    sig_names = {}
    for n in names:
        sig_names.setdefault(_ent_sig(n), set()).add(n)
    sigs = list(sig_names)
    resolve = {}
    for n in names:
        s = _ent_sig(n)
        cands = [o for o in sigs if s <= o]
        maximal = [o for o in cands if not any(o < p for p in cands)]
        target_sig = maximal[0] if len(maximal) == 1 else s
        resolve[n] = max(sig_names[target_sig], key=_ent_quality)
    if any(resolve[n] != n for n in names):
        UNIFIED_DF.loc[mask, 'Entity'] = UNIFIED_DF.loc[mask, 'Entity'].map(lambda e: resolve.get(e, e))


# Memoised results that only change when the underlying data is (re)loaded.
# Cleared by _invalidate_caches() at the end of every load_master_data_engine().
_CACHE = {}


def _invalidate_caches():
    _CACHE.clear()


def load_master_data_engine():
    global UNIFIED_DF
    try:
        conn = sqlite3.connect(DB_NAME)
        # Attempt to load with dimension and source_file
        try:
            UNIFIED_DF = pd.read_sql_query("SELECT * FROM financial_metrics", conn)
        except:
            # Fallback for old schema
            UNIFIED_DF = pd.read_sql_query("SELECT *, 'Insurer' as dimension, 'Unknown' as source_file FROM financial_metrics", conn)
        conn.close()

        if UNIFIED_DF.empty:
            print("[!] SQL Database 'financial_metrics' is empty.")
            return

        # Map SQL columns (snake_case) to Application (PascalCase)
        UNIFIED_DF = UNIFIED_DF.rename(columns={
            "insurer": "Entity",
            "dimension": "Dimension",
            "financial_year": "Financial_Year",
            "quarter": "Quarter",
            "metric": "Metric",
            "value": "Value",
            "line_of_business": "Line_of_Business",
            "class_of_business": "Class_of_Business",
            "source_file": "Source_File"
        })

        # Ensure correct types and Clean Data in Memory
        UNIFIED_DF['Value'] = pd.to_numeric(UNIFIED_DF['Value'], errors='coerce')
        for col in ['Entity', 'Dimension', 'Financial_Year', 'Quarter', 'Metric', 'Source_File']:
            if col in UNIFIED_DF.columns:
                UNIFIED_DF[col] = UNIFIED_DF[col].astype(str).str.strip()
            
        # Fill Missing Dimensions
        if 'Dimension' in UNIFIED_DF.columns:
            UNIFIED_DF['Dimension'] = UNIFIED_DF['Dimension'].replace(['None', 'nan', ''], 'Insurer')

        # Fix Dashes in Quarter
        UNIFIED_DF['Quarter'] = UNIFIED_DF['Quarter'].replace(['-', 'nan', 'None', ''], 'Annual')

        _canonicalize_insurer_entities()

        # Unify bare calendar years (e.g. an AUM "As on 31 March 2024" snapshot)
        # into financial-year form so "2024" and "2024-25" don't both appear.
        if 'Financial_Year' in UNIFIED_DF.columns:
            fy = UNIFIED_DF['Financial_Year'].astype(str)
            bare = fy.str.fullmatch(r'\d{4}')
            UNIFIED_DF.loc[bare, 'Financial_Year'] = fy[bare].apply(
                lambda y: f"{int(y) - 1}-{y[2:]}")

        # Collapse the Part tables' invented industry-aggregate names onto the
        # Handbook's official sector names so the Industry view shows one entity
        # per sector instead of near-duplicates.
        _INDUSTRY_ALIASES = {
            "Life Insurers (Industry)": "Life Insurance Sector",
            "General Insurers (Industry)": "General Insurance Sector",
            "General, Health & RE (Industry)": "General Insurance Sector",
            "Non-Life & Reinsurers (Industry)": "General Insurance Sector",
            "Health Industry": "Health Insurance Sector",
        }
        if {'Dimension', 'Entity'} <= set(UNIFIED_DF.columns):
            ind = UNIFIED_DF['Dimension'] == 'Industry'
            UNIFIED_DF.loc[ind, 'Entity'] = UNIFIED_DF.loc[ind, 'Entity'].map(
                lambda e: _INDUSTRY_ALIASES.get(e, e))

        _invalidate_caches()   # data changed → drop memoised filter options / compliance
        print(f"[+] Data Engine Loaded: {len(UNIFIED_DF)} rows from SQL.")

    except Exception as e:
        print(f"[!] Error loading SQL data: {e}")
        UNIFIED_DF = pd.DataFrame()

def get_filter_options():
    """
    Returns options structured by Dimension for the new UI. Memoised — the result
    only changes when data is reloaded (which clears the cache), so repeated modal
    opens are instant instead of rebuilding ~50k combo tuples each time.
    """
    if UNIFIED_DF.empty:
        load_master_data_engine()

    if UNIFIED_DF.empty:
        return {"dimensions": [], "entities": {}, "metrics": [], "years": [], "quarters": [], "lobs": [], "classes": []}

    if 'filter_options' in _CACHE:
        return _CACHE['filter_options']

    entities_by_dim = {}
    unique_dims = sorted(UNIFIED_DF['Dimension'].unique().tolist())

    def _uniq(d, col):
        return sorted(d[col].dropna().unique().tolist()) if col in d.columns else []

    # Per-dimension option sets — metrics/years/etc. differ by dimension (e.g.
    # Industry metrics and Insurer metrics don't overlap), so the UI must scope
    # options to the selected dimension to avoid empty "no data" combinations.
    # Column -> position in the cascade tuple the UI sends as "combos".
    CASCADE = ['Entity', 'Line_of_Business', 'Class_of_Business', 'Metric', 'Financial_Year', 'Quarter']
    # The Financials view reproduces whole statements, so its filter only cascades
    # Entity -> Statement(LOB) -> Year -> Quarter; dropping Class/Metric keeps the
    # combos payload small (statements have hundreds of line items per insurer).
    CASCADE_BY_DIM = {'Financials': ['Entity', 'Line_of_Business', 'Financial_Year', 'Quarter']}

    by_dim = {}
    for dim in unique_dims:
        d = UNIFIED_DF[UNIFIED_DF['Dimension'] == dim]
        entities_by_dim[dim] = sorted(d['Entity'].unique().tolist())
        # Distinct valid combinations drive a cascading filter in the UI: each
        # step only offers options consistent with earlier picks, so impossible
        # "no data" combos can't be built.
        avail = [c for c in CASCADE_BY_DIM.get(dim, CASCADE) if c in d.columns]
        combos = d[avail].drop_duplicates().astype(str).values.tolist()
        by_dim[dim] = {
            "cascade": avail,            # column order of each combo tuple
            "combos": combos,            # distinct [Entity, LOB, Class, Metric, Year, Quarter]
            # Plain per-dimension unions kept as a fallback for older clients.
            "metrics": _uniq(d, 'Metric'),
            "years": _uniq(d, 'Financial_Year'),
            "quarters": _uniq(d, 'Quarter'),
            "lobs": _uniq(d, 'Line_of_Business'),
            "classes": _uniq(d, 'Class_of_Business'),
        }

    # Quick-filter groups for the Insurer picker: {group -> [insurer names]}.
    insurer_groups = {}
    for ent in entities_by_dim.get('Insurer', []):
        for tag in _insurer_tags(ent):
            insurer_groups.setdefault(tag, []).append(ent)

    result = {
        "dimensions": unique_dims,
        "entities": entities_by_dim,
        "by_dim": by_dim,
        "insurer_groups": insurer_groups,
        # Global unions kept for backward compatibility.
        "metrics": _uniq(UNIFIED_DF, 'Metric'),
        "years": _uniq(UNIFIED_DF, 'Financial_Year'),
        "quarters": _uniq(UNIFIED_DF, 'Quarter'),
        "lobs": _uniq(UNIFIED_DF, 'Line_of_Business'),
        "classes": _uniq(UNIFIED_DF, 'Class_of_Business'),
    }
    _CACHE['filter_options'] = result
    return result

# --- PIVOT LOGIC ---
def _create_pivoted_view(filters):
    if UNIFIED_DF.empty: return pd.DataFrame(), [], []
    
    df = UNIFIED_DF.copy()
    
    # 1. Filter by Dimension
    target_dim = filters.get('dimension', 'Insurer')
    df = df[df['Dimension'] == target_dim]

    # 2. Apply Filters
    if filters.get('entities'): df = df[df['Entity'].isin(filters['entities'])]
    if filters.get('years'): df = df[df['Financial_Year'].isin(filters['years'])]
    if filters.get('metrics'): df = df[df['Metric'].isin(filters['metrics'])]
    if filters.get('quarters'): df = df[df['Quarter'].isin(filters['quarters'])]
    if filters.get('lobs'): df = df[df['Line_of_Business'].isin(filters['lobs'])]
    if filters.get('classes'): df = df[df['Class_of_Business'].isin(filters['classes'])]
    
    if df.empty: return pd.DataFrame(), ["No data found."], []
    
    # 3. Risk Analysis
    risk_alerts = []
    if filters.get('entities'):
        risk_alerts = _analyze_risk(filters['entities'], target_dim, filters.get('years'))

    try:
        # --- KEY UPDATE: Include Source_File in Index to separate duplicates ---
        index_cols = ['Entity', 'Financial_Year']
        if 'Line_of_Business' in df.columns: index_cols.append('Line_of_Business')
        if 'Class_of_Business' in df.columns: index_cols.append('Class_of_Business')
        if 'Quarter' in df.columns: index_cols.append('Quarter')
        if 'Source_File' in df.columns: index_cols.append('Source_File')
        
        pivot_df = df.pivot_table(index=index_cols, columns='Metric', values='Value', aggfunc='sum').reset_index()
        
        # Rename 'Entity' to Dimension Name
        pivot_df = pivot_df.rename(columns={'Entity': target_dim})
        
        # Add Units to Header
        new_cols = []
        for c in pivot_df.columns:
            if str(c) in UNIT_MAP: new_cols.append(f"{c} ({UNIT_MAP[str(c)]})")
            else: new_cols.append(str(c))
        pivot_df.columns = new_cols
        
        return pivot_df.fillna('-'), [], risk_alerts

    except Exception as e:
        print(f"Pivot Error: {e}")
        return pd.DataFrame(), ["Error processing table."], []

# --- PUBLIC FUNCTIONS (UPDATED FOR COLUMN ORDER & EXCEL CLEANUP) ---
def filter_data(filters):
    pivot_df, missing_alerts, risk_alerts = _create_pivoted_view(filters)
    if pivot_df.empty: return {'columns': [], 'rows': [], 'missing': missing_alerts, 'risks': risk_alerts}
    
    target_dim = filters.get('dimension', 'Insurer')
    
    # 1. Define Standard Left-Side Columns
    possible_headers = [target_dim, 'Financial_Year', 'Line_of_Business', 'Class_of_Business', 'Quarter']
    
    # 2. Extract Base Columns that actually exist
    base_cols = [c for c in possible_headers if c in pivot_df.columns]
    
    # 3. Extract Metric Columns (Excluding Source_File)
    metric_cols = [c for c in pivot_df.columns if c not in base_cols and c != 'Source_File']
    
    # 4. Construct Final Order: Base -> Metrics -> Source_File (Last)
    final_cols = base_cols + metric_cols
    # Force Source_File to append at the end
    if 'Source_File' in pivot_df.columns:
        final_cols.append('Source_File')
    
    return {'columns': final_cols, 'rows': pivot_df.to_dict('records'), 'missing': missing_alerts, 'risks': risk_alerts}

def generate_excel(filters):
    pivot_df, _, _ = _create_pivoted_view(filters)
    if pivot_df.empty: return None
    
    # --- CRITICAL FIX: DROP SOURCE FILE FOR EXCEL ---
    if 'Source_File' in pivot_df.columns:
        pivot_df = pivot_df.drop(columns=['Source_File'])
        
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer: pivot_df.to_excel(writer, index=False, sheet_name='IRIS_Report')
    output.seek(0)
    return output

# --- FINANCIAL STATEMENT VIEW (whole-statement reproduction) ---
def get_statement_options():
    """Entities + statements available in the Financials dimension."""
    if UNIFIED_DF.empty:
        load_master_data_engine()
    if UNIFIED_DF.empty or 'Dimension' not in UNIFIED_DF.columns:
        return {"entities": [], "statements": []}
    d = UNIFIED_DF[UNIFIED_DF['Dimension'] == 'Financials']
    return {
        "entities": sorted(d['Entity'].dropna().unique().tolist()),
        "statements": sorted(d['Line_of_Business'].dropna().unique().tolist()),
    }

def build_financial_statement(entities, statement, years=None):
    """Reproduce whole financial statements: ordered sections -> line items ->
    value per year, preserving the statement's natural row order."""
    if UNIFIED_DF.empty:
        load_master_data_engine()
    if UNIFIED_DF.empty:
        return {"statement": statement, "entities": []}

    df = UNIFIED_DF[(UNIFIED_DF['Dimension'] == 'Financials') &
                    (UNIFIED_DF['Line_of_Business'] == statement)]
    if entities:
        df = df[df['Entity'].isin(entities)]
    if years:
        df = df[df['Financial_Year'].isin(years)]
    if df.empty:
        return {"statement": statement, "years": [], "entities": []}

    def _strip_unit(metric):
        m = re.search(r"\(([^()]*)\)\s*$", metric)
        if m and re.search(r"₹|crore|lakh|per ?cent|nos|us \$|%", m.group(1), re.I):
            return metric[:m.start()].strip(), m.group(1).strip()
        return metric, ""

    all_years = sorted(df['Financial_Year'].dropna().unique().tolist())
    out_entities = []
    for entity in (entities or sorted(df['Entity'].unique().tolist())):
        edf = df[df['Entity'] == entity]
        if edf.empty:
            continue
        unit = ""
        sections, sec_index, item_index = [], {}, {}
        for _, row in edf.iterrows():
            section = str(row.get('Class_of_Business') or 'General')
            label, u = _strip_unit(str(row['Metric']))
            unit = unit or u
            if section not in sec_index:
                sec_index[section] = len(sections)
                sections.append({"name": section, "items": []})
            sec = sections[sec_index[section]]
            ikey = (section, label)
            if ikey not in item_index:
                item_index[ikey] = len(sec["items"])
                sec["items"].append({"label": label, "values": {}})
            sec["items"][item_index[ikey]]["values"][str(row['Financial_Year'])] = row['Value']
        # Flatten each item's values into the year order.
        for sec in sections:
            for it in sec["items"]:
                it["values"] = [it["values"].get(y) for y in all_years]
        out_entities.append({"entity": entity, "unit": unit, "sections": sections})

    return {"statement": statement, "years": all_years, "entities": out_entities}

def generate_statement_excel(entities, statement, years=None):
    """Excel workbook of whole statements — one sheet per insurer."""
    data = build_financial_statement(entities, statement, years)
    if not data["entities"]:
        return None
    yrs = data["years"]
    output = io.BytesIO()
    used = set()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for ent in data["entities"]:
            recs = []
            for sec in ent["sections"]:
                if sec["name"] and sec["name"] != "General":
                    recs.append({"Particulars": sec["name"], **{y: "" for y in yrs}})
                for it in sec["items"]:
                    recs.append({"Particulars": it["label"],
                                 **{y: ("" if it["values"][i] is None else it["values"][i])
                                    for i, y in enumerate(yrs)}})
            df = pd.DataFrame(recs, columns=["Particulars", *yrs])
            name = re.sub(r"[^A-Za-z0-9 ]", "", ent["entity"])[:28].strip() or "Sheet"
            base, n = name, 1
            while name.lower() in used:
                n += 1
                name = f"{base[:25]} {n}"
            used.add(name.lower())
            df.to_excel(writer, index=False, sheet_name=name)
    output.seek(0)
    return output

# ==========================================
# COMPLIANCE ENGINE UPDATES (TREND AWARE)
# ==========================================

_PSU_KEYS = ("life insurance corporation", "new india", "national insurance", "united india",
             "oriental insurance", "general insurance corporation", "gic re", "ecgc",
             "agriculture insurance")
_SAHI_KEYS = ("star health", "care health", "aditya birla health", "niva bupa",
              "manipalcigna", "manipal cigna", "galaxy health", "narayana health")
_RE_KEYS = ("reinsur", "gic re", "gen re", "munich", "swiss re", "hannover", "scor ", "scor se",
            "rga", "lloyd", "markel", "axa france", "allianz global", "general reinsurance",
            "valueattics", "iti reinsurance")


def _insurer_tags(name):
    """Classify an insurer into quick-filter groups (it can hold several)."""
    low = str(name).lower()
    tags = set()
    if any(k in low for k in _RE_KEYS):
        tags.add("Reinsurance")
    if any(k in low for k in _SAHI_KEYS):
        tags.add("Health (SAHI)")
    if "life" in low:
        tags.add("Life")
    if not (tags & {"Reinsurance", "Health (SAHI)", "Life"}):
        tags.add("General")
    tags.add("PSU" if any(k in low for k in _PSU_KEYS) else "Private")
    return tags


def _insurer_group(name):
    tags = _insurer_tags(name)
    for g in ("Reinsurance", "Health (SAHI)", "Life", "General"):
        if g in tags:
            return g
    return "Other"


def get_compliance_years():
    if UNIFIED_DF.empty: 
        load_master_data_engine()

    if UNIFIED_DF.empty: return []
    if 'Financial_Year' not in UNIFIED_DF.columns: return []
    return sorted(UNIFIED_DF['Financial_Year'].dropna().unique().tolist(), reverse=True)

def get_compliance_dashboard(target_year=None):
    """
    Analyzes data against thresholds AND historical trends.
    Only runs for Dimension = 'Insurer'. Memoised per target year — the result is
    identical for every user until data is reloaded, so we compute it once.
    """
    if UNIFIED_DF.empty:
        load_master_data_engine()
    if UNIFIED_DF.empty:
        return []

    cache_key = ('compliance', target_year or 'Latest')
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    df = UNIFIED_DF
    # Compliance only makes sense for Insurers
    if 'Dimension' in df.columns:
        df = df[df['Dimension'] == 'Insurer']

    dashboard_data = []
    insurers = df['Entity'].unique()

    for insurer in insurers:
        ins_df = df[df['Entity'] == insurer]
        if ins_df.empty: continue
        group = _insurer_group(insurer)
        is_sahi = group == "SAHI"

        # Scope to the chosen year, else use everything (each metric then resolves
        # to its own latest available value — solvency is quarterly, others annual).
        scoped = ins_df
        if target_year and target_year != "Latest":
            scoped = ins_df[ins_df['Financial_Year'] == target_year]
        if scoped.empty: continue
        try:
            curr_year = sorted(scoped['Financial_Year'].dropna().unique(),
                               key=lambda y: str(y))[-1]
        except Exception:
            continue

        def latest_val(names, lob=None, agg="last"):
            """Latest value of the first matching metric (optionally a LOB), across
            quarters. agg='mean' averages across a metric's segments for the latest year."""
            names = names if isinstance(names, list) else [names]
            for nm in names:
                rows = scoped[scoped['Metric'].str.contains(nm, case=False, na=False, regex=False)]
                if lob is not None and 'Line_of_Business' in rows.columns:
                    rows = rows[rows['Line_of_Business'].astype(str).str.contains(lob, case=False, na=False, regex=False)]
                rows = rows.dropna(subset=['Value'])
                if rows.empty:
                    continue
                rows = rows.assign(_sy=rows['Financial_Year'].astype(str))
                ly = sorted(rows['_sy'].unique())[-1]
                rows = rows[rows['_sy'] == ly]
                try:
                    if agg == "mean":
                        return round(float(rows['Value'].astype(float).mean()), 2)
                    r = rows.sort_values('Quarter').iloc[-1]
                    return float(str(r['Value']).replace(',', '').replace('%', ''))
                except Exception:
                    continue
            return None

        # Available directly:
        solvency = latest_val(["Solvency Ratio", "Solvency Margin"])
        gdp = latest_val(["Gross Direct Premium (Within And Outside", "Gross Direct Premium (Within India", "Gross Direct Premium"])
        # Company-level Incurred Claims Ratio = total net claims / total net earned
        # premium; fall back to the segment-average ICR if the totals aren't there.
        claims_amt = latest_val(["Claims Incurred (Net) (₹Crore)"], lob="General (All Segments)")
        nep_co = latest_val(["Net Earned Premium (₹Crore)"], lob="General (All Segments)")
        if claims_amt is not None and nep_co:
            claims_ratio = round(claims_amt / nep_co * 100, 2)
        else:
            claims_ratio = latest_val(["Incurred Claims Ratio"], agg="mean")
        underwriting = latest_val(["Underwriting Profit"])
        repudiation_val = latest_val(["Repudiation Ratio"])
        # Derived where not reported directly:
        eom_amt = latest_val(["Commission, Expenses of Management", "Expense of Management"])
        expense_ratio = latest_val(["Expense of Management to GDP", "EoM Ratio"])
        if expense_ratio is None and eom_amt is not None and gdp:
            expense_ratio = round(eom_amt / gdp * 100, 2)
        combined_ratio = latest_val(["Combined Ratio"])
        if combined_ratio is None and claims_ratio is not None and expense_ratio is not None:
            combined_ratio = round(claims_ratio + expense_ratio, 2)

        curr_qtr = ""
        status = "COMPLIANT"
        alerts = []

        # --- B. THRESHOLD CHECKS (Aligned with EWS Logic) ---
        if solvency is not None and solvency < 1.5:
            status = "VIOLATION"
            alerts.append({"level": "critical", "msg": f"Regulatory Violation: {insurer} - Solvency {solvency} < 1.5 limit"})

        if expense_ratio is not None:
            limit = 35 if is_sahi else 30
            if expense_ratio > limit:
                status = "VIOLATION"
                alerts.append({"level": "critical", "msg": f"Regulatory Violation: {insurer} - EoM {expense_ratio}% exceeds {limit}% limit"})
        
        if repudiation_val is not None and repudiation_val > 10:
            # High Repudiation is a Warning/Watchlist, not necessarily a status change to VIOLATION
            alerts.append({"level": "warning", "msg": f"High Repudiation: {insurer} - Repudiation Ratio {repudiation_val}% exceeds 10% limit"})

        # --- C. TREND CHECKS — only on a single percent/ratio series, one value
        # per (distinct) year, so absolute ₹Crore values never read as percentages.
        def check_trend_alert(metric_name, label, alert_type="rising"):
            rows = ins_df[ins_df['Metric'].str.contains(metric_name, case=False, na=False, regex=False)].copy()
            if rows.empty:
                return
            rows['_sy'] = rows['Financial_Year'].astype(str).str.extract(r'(\d{4})').astype(float)
            rows = (rows.dropna(subset=['Value', '_sy'])
                        .groupby('Financial_Year', as_index=False)
                        .agg({'Value': 'mean', '_sy': 'first'})
                        .sort_values('_sy'))
            if len(rows) < 3:
                return
            vals = [float(v) for v in rows['Value'].tolist()[-3:]]
            yrs = rows['Financial_Year'].tolist()[-3:]
            if alert_type == "rising" and vals[0] < vals[1] < vals[2] and (vals[2] - vals[0]) >= 3:
                alerts.append({"level": "warning",
                               "msg": f"Rising Trend: {insurer} - {label} rose from {vals[0]:.2f}% to {vals[2]:.2f}% ({yrs[0]} to {yrs[2]})."})
            elif alert_type == "falling" and vals[0] > vals[1] > vals[2] and (vals[0] - vals[2]) >= 0.2:
                alerts.append({"level": "warning",
                               "msg": f"Deteriorating Solvency: {insurer} - dropped from {vals[0]:.2f} to {vals[2]:.2f} ({yrs[0]} to {yrs[2]})."})

        check_trend_alert("Incurred Claims Ratio", "Incurred Claims Ratio", "rising")
        check_trend_alert("Solvency Ratio", "Solvency", "falling")

        # --- D. Finalize Data ---
        has_critical = any(a['level'] == 'critical' for a in alerts)
        has_warning = any(a['level'] == 'warning' for a in alerts)
        if has_critical:
            status = "VIOLATION"
        elif has_warning and status != "VIOLATION":
            status = "WATCHLIST"

        def _fv(v):
            return v if (v is not None and v != "") else "N/A"

        dashboard_data.append({
            "name": insurer,
            "group": group,
            "tags": sorted(_insurer_tags(insurer)),
            "metrics": {
                "solvency": _fv(solvency),
                "expenses": _fv(expense_ratio),
                "combined": _fv(combined_ratio),
                "claims": _fv(claims_ratio),
                "premium": _fv(round(gdp, 0) if gdp else None),
                "underwriting": _fv(underwriting),
            },
            "status": status,
            "alerts": alerts,
            "has_critical": has_critical,
            "has_warning": has_warning,
            "last_updated": str(curr_year),
        })

    priority = {"VIOLATION": 0, "WATCHLIST": 1, "COMPLIANT": 2}
    dashboard_data.sort(key=lambda x: priority.get(x['status'], 3))

    _CACHE[cache_key] = dashboard_data
    return dashboard_data
