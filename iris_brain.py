import sqlite3
import pandas as pd # type: ignore
import os
import re
import json
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

# --- Departments registry -------------------------------------------------
# Each department is a Knowledge Base module (own sidebar entry + search page,
# scoped to its Doc_Category + cross-cutting GENERAL). Admins manage the list
# in-app; it persists to knowledge_base/departments.json. GENERAL is implicit
# (it surfaces in every module) and is never itself a department.
DEPARTMENTS_PATH = os.path.join(KB_FOLDER, "departments.json")
DEFAULT_DEPARTMENTS = [
    {"key": "HEALTH",  "label": "Health",   "icon": "fa-heart-pulse",  "general": True},
    {"key": "LIFE",    "label": "Life",     "icon": "fa-umbrella",     "general": True},
    {"key": "NONLIFE", "label": "Non-Life", "icon": "fa-shield-halved", "general": True},
]

def _ensure_dept_table(conn):
    """Create the departments table on first use, seeding it from a legacy
    departments.json if present, otherwise from the built-in defaults. Stored in
    iris.db so it persists across redeploys (Litestream-replicated) — a JSON file
    on the container filesystem would be wiped on every deploy."""
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='departments'"
    ).fetchone() is not None
    conn.execute("""CREATE TABLE IF NOT EXISTS departments (
        key TEXT PRIMARY KEY, label TEXT, icon TEXT, general INTEGER DEFAULT 0, ord INTEGER DEFAULT 0)""")
    if not exists:
        seed = DEFAULT_DEPARTMENTS
        try:
            if os.path.exists(DEPARTMENTS_PATH):
                with open(DEPARTMENTS_PATH, "r", encoding="utf-8") as f:
                    js = json.load(f)
                if js:
                    seed = js
        except Exception:
            pass
        for i, d in enumerate(seed):
            k = str(d.get("key", "")).strip().upper()
            if not k or k == "GENERAL":
                continue
            conn.execute("INSERT OR IGNORE INTO departments (key,label,icon,general,ord) VALUES (?,?,?,?,?)",
                         (k, (d.get("label") or k.title()), (d.get("icon") or "fa-folder"),
                          1 if d.get("general") else 0, i))
        conn.commit()

def load_departments():
    try:
        conn = sqlite3.connect(DB_NAME)
        _ensure_dept_table(conn)
        rows = conn.execute("SELECT key,label,icon,general FROM departments ORDER BY ord, key").fetchall()
        conn.close()
        out, seen = [], set()
        for (k, label, icon, general) in rows:
            k = str(k or "").strip().upper()
            if not k or k == "GENERAL" or k in seen:
                continue
            seen.add(k)
            out.append({"key": k,
                        "label": (str(label or "").strip() or k.title()),
                        "icon": (str(icon or "").strip() or "fa-folder"),
                        # Whether this module also surfaces cross-cutting GENERAL
                        # documents (insurance lines do; e.g. HR does not).
                        "general": bool(general)})
        if out:
            return out
    except Exception:
        pass
    return [dict(d) for d in DEFAULT_DEPARTMENTS]

def save_departments(depts):
    conn = sqlite3.connect(DB_NAME)
    _ensure_dept_table(conn)
    conn.execute("DELETE FROM departments")
    for i, d in enumerate(depts):
        k = str(d.get("key", "")).strip().upper()
        if not k or k == "GENERAL":
            continue
        conn.execute("INSERT OR REPLACE INTO departments (key,label,icon,general,ord) VALUES (?,?,?,?,?)",
                     (k, (d.get("label") or k.title()), (d.get("icon") or "fa-folder"),
                      1 if d.get("general") else 0, i))
    conn.commit()
    conn.close()
    return load_departments()

def department_keys():
    return [d["key"] for d in load_departments()]

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

def common_prefix(a, b):
    """Longest common prefix of two strings."""
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]: i += 1
    return a[:i]

def search_root(word, stem=None):
    """Prefix safe to use for `\\broot\\w*` matching/highlighting. Porter sometimes
    restores letters so the stem is NOT a prefix of the word it came from
    (pricing -> 'price', which then can't match 'pricing'). Fall back to the
    common prefix of the word and its stem so the root always *is* a real prefix
    of the typed word. No-op when the stem is already a prefix (the usual case)."""
    w = str(word).lower()
    s = stem if stem is not None else get_stem(w)
    if s and w.startswith(s):
        return s                       # normal case — stem is a clean prefix
    cp = common_prefix(w, s or "")
    return cp if len(cp) >= 3 else (s or w)

def _root_present(root, text, text_join):
    """Does `root` prefix-match a word in the text? Checks the text AND a
    hyphen-collapsed copy so a typed 'deempanelment' matches 'de-empanelment'
    (and vice-versa) — hyphenation in regulatory terms shouldn't break search."""
    pat = rf"\b{re.escape(root)}\w*"
    return bool(re.search(pat, text)) or bool(re.search(pat, text_join))

def _root_count(root, text, text_join):
    pat = rf"\b{re.escape(root)}\w*"
    a = re.findall(pat, text)
    return len(a) if a else len(re.findall(pat, text_join))

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


def _html_to_text(html):
    """Plain text from clause HTML (for search/indexing) — block tags -> newlines."""
    import html as _h
    t = re.sub(r"(?i)<(?:br|/p|/div|/li|/tr|/h[1-6])\s*/?>", "\n", html or "")
    t = re.sub(r"<[^>]+>", "", t)
    t = _h.unescape(t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def doc_revision(source):
    """A content hash of a document's current clauses (from the DB — the source of
    truth across instances). Used for optimistic-concurrency on save."""
    import hashlib
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT clause_id, clause_text, regulatory_tags, clause_html, sort_order "
        "FROM regulatory_clauses WHERE source_doc=? ORDER BY clause_id", (str(source),)).fetchall()
    conn.close()
    h = hashlib.sha1()
    for r in rows:
        h.update(repr(r).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def replace_document_clauses(source, clauses, editor=None):
    """Bulk-replace all clauses of a document with the given ordered list
    (each {id, html, tags}). Snapshots the prior clauses to clause_versions, then
    rewrites them in order. Derives plain text from HTML for search."""
    global KB_CACHE_DF
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(DB_NAME)
    meta = conn.execute("SELECT doc_category, doc_type FROM regulatory_clauses WHERE source_doc=? LIMIT 1",
                        (str(source),)).fetchone()
    cat, dtype = (meta[0], meta[1]) if meta else ("GENERAL", "REGULATION")
    existing = {}
    for r in conn.execute("SELECT clause_id, clause_html, clause_text, regulatory_tags, updated_by, updated_at "
                          "FROM regulatory_clauses WHERE source_doc=?", (str(source),)).fetchall():
        existing[r[0]] = {"html": r[1], "text": r[2], "tags": r[3], "by": r[4], "at": r[5]}

    # normalise incoming ids (unique)
    seen = set(); norm = []
    for i, c in enumerate(clauses, 1):
        cid = (str(c.get("id", "")).strip() or f"C-{i}")
        while cid in seen:
            cid = f"{cid}-{i}"
        seen.add(cid); norm.append((cid, c))

    # snapshot + tombstone clauses that are being removed (so a later sync won't
    # resurrect them); lift tombstones for clauses present in the saved set.
    removed = [oid for oid in existing if oid not in seen]
    for oid in removed:
        ex = existing[oid]
        _snapshot(conn, oid, source, ex["html"], ex["text"], ex["tags"], editor, " (doc save: removed)")
    _tombstone(conn, source, removed, editor)
    _clear_tombstones(conn, source, list(seen))

    conn.execute("DELETE FROM regulatory_clauses WHERE source_doc=?", (str(source),))
    for i, (cid, c) in enumerate(norm, 1):
        html = c.get("html") or ""
        text = _html_to_text(html)
        tags = c.get("tags", "")
        tags = ", ".join(tags) if isinstance(tags, list) else str(tags or "")
        header = (text.split("\n", 1)[0])[:120]
        is_header = 1 if (text.strip().endswith(":") and "\n" not in text.strip()) else 0
        changed = bool(c.get("changed")) or (cid not in existing)
        if changed and cid in existing:        # snapshot the prior content of a changed clause
            ex = existing[cid]
            _snapshot(conn, cid, source, ex["html"], ex["text"], ex["tags"], editor, " (doc save)")
        if changed:
            up_by, up_at = (editor or ""), now
        else:                                  # untouched -> keep its prior edit attribution
            up_by, up_at = existing[cid]["by"], existing[cid]["at"]
        conn.execute(
            "INSERT INTO regulatory_clauses (source_doc, doc_category, doc_type, clause_id, clause_text, "
            "context_header, regulatory_tags, priority, is_header, sort_order, clause_html, updated_by, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (str(source), cat, dtype, cid, text, header, tags, 99, is_header, i * 10,
             html or None, up_by, up_at))
    conn.commit()
    conn.close()
    refresh_kb()
    return len(clauses)


def _ensure_tombstone_table(conn):
    """Tombstones for Studio-deleted clauses. The data Sync rebuilds Excel-backed
    docs from scratch, so without this a deleted clause is re-created next sync."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS deleted_clauses ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, source_doc TEXT NOT NULL, "
        "clause_id TEXT NOT NULL, deleted_by TEXT, deleted_at TEXT, "
        "UNIQUE(source_doc, clause_id))")


def _tombstone(conn, source, clause_ids, editor=None):
    """Record (source, clause_id) deletions so the next sync won't resurrect them."""
    _ensure_tombstone_table(conn)
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn.executemany(
        "INSERT OR IGNORE INTO deleted_clauses (source_doc, clause_id, deleted_by, deleted_at) "
        "VALUES (?,?,?,?)",
        [(str(source), str(cid), editor or "", now) for cid in clause_ids if str(cid)])


def _clear_tombstones(conn, source, clause_ids=None):
    """Lift tombstones when a clause/doc is (re)added or re-imported."""
    _ensure_tombstone_table(conn)
    if clause_ids is None:
        conn.execute("DELETE FROM deleted_clauses WHERE source_doc=?", (str(source),))
    else:
        conn.executemany("DELETE FROM deleted_clauses WHERE source_doc=? AND clause_id=?",
                         [(str(source), str(cid)) for cid in clause_ids])


def _load_tombstones(conn):
    """Return the set of (source_doc, clause_id) the sync must skip re-creating."""
    _ensure_tombstone_table(conn)
    return {(str(s), str(c)) for s, c in
            conn.execute("SELECT source_doc, clause_id FROM deleted_clauses").fetchall()}


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
    _tombstone(conn, source, [r[0] for r in rows], editor)
    conn.commit()
    conn.close()
    refresh_kb()
    return len(rows)


def _snapshot(conn, clause_id, source, html, text, tags, editor, suffix=""):
    conn.execute(
        "INSERT INTO clause_versions (clause_id, source_doc, html, body_text, tags, edited_by, edited_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (clause_id, str(source), html, text, tags, (editor or "") + suffix,
         datetime.utcnow().isoformat(sep=" ", timespec="seconds")))


def _renumber(conn, source):
    """Re-sequence a document's clauses (sort_order = 10, 20, 30 …) by current order."""
    ids = conn.execute(
        "SELECT clause_id FROM regulatory_clauses WHERE source_doc=? ORDER BY COALESCE(sort_order, rowid)",
        (str(source),)).fetchall()
    for i, (cid,) in enumerate(ids, 1):
        conn.execute("UPDATE regulatory_clauses SET sort_order=? WHERE source_doc=? AND clause_id=?",
                     (i * 10, str(source), cid))


def add_clause(source, after_id, editor=None):
    """Insert a new blank clause right after `after_id`. Returns the new id."""
    conn = sqlite3.connect(DB_NAME)
    meta = conn.execute("SELECT doc_category, doc_type FROM regulatory_clauses WHERE source_doc=? LIMIT 1",
                        (str(source),)).fetchone()
    if not meta:
        conn.close(); return None
    existing = {r[0] for r in conn.execute("SELECT clause_id FROM regulatory_clauses WHERE source_doc=?", (str(source),))}
    n = 1
    new_id = f"NEW-{n}"
    while new_id in existing:
        n += 1; new_id = f"NEW-{n}"
    arow = conn.execute("SELECT COALESCE(sort_order, rowid) FROM regulatory_clauses WHERE source_doc=? AND clause_id=?",
                        (str(source), str(after_id))).fetchone()
    after_so = arow[0] if arow else 0
    conn.execute(
        "INSERT INTO regulatory_clauses (source_doc, doc_category, doc_type, clause_id, clause_text, "
        "context_header, regulatory_tags, priority, is_header, sort_order, updated_by) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (str(source), meta[0], meta[1], new_id, "New clause:", "New clause", "", 99, 0, after_so + 5, editor or ""))
    _clear_tombstones(conn, source, [new_id])
    _renumber(conn, source)
    conn.commit(); conn.close()
    refresh_kb()
    return new_id


def delete_clause(source, clause_id, editor=None):
    conn = sqlite3.connect(DB_NAME)
    r = conn.execute("SELECT clause_html, clause_text, regulatory_tags FROM regulatory_clauses WHERE source_doc=? AND clause_id=?",
                     (str(source), str(clause_id))).fetchone()
    if not r:
        conn.close(); return False
    _snapshot(conn, clause_id, source, r[0], r[1], r[2], editor, " (delete)")
    conn.execute("DELETE FROM regulatory_clauses WHERE source_doc=? AND clause_id=?", (str(source), str(clause_id)))
    _tombstone(conn, source, [clause_id], editor)
    _renumber(conn, source)
    conn.commit(); conn.close()
    refresh_kb()
    return True


def merge_clause(source, clause_id, editor=None):
    """Merge a clause into the one immediately above it (by order)."""
    conn = sqlite3.connect(DB_NAME)
    seq = conn.execute("SELECT clause_id, clause_html, clause_text, regulatory_tags FROM regulatory_clauses "
                       "WHERE source_doc=? ORDER BY COALESCE(sort_order, rowid)", (str(source),)).fetchall()
    ids = [row[0] for row in seq]
    if str(clause_id) not in ids:
        conn.close(); return False
    idx = ids.index(str(clause_id))
    if idx == 0:
        conn.close(); return False
    prev, this = seq[idx - 1], seq[idx]
    new_text = (prev[2] or "") + "\n" + (this[2] or "")
    new_html = ((prev[1] or "") + (this[1] or "")) or None
    _snapshot(conn, prev[0], source, prev[1], prev[2], prev[3], editor, " (merge)")
    _snapshot(conn, this[0], source, this[1], this[2], this[3], editor, " (merge)")
    now = datetime.utcnow().isoformat(sep=" ", timespec="seconds")
    conn.execute("UPDATE regulatory_clauses SET clause_text=?, clause_html=?, updated_by=?, updated_at=? "
                 "WHERE source_doc=? AND clause_id=?", (new_text, new_html, editor or "", now, str(source), prev[0]))
    conn.execute("DELETE FROM regulatory_clauses WHERE source_doc=? AND clause_id=?", (str(source), str(clause_id)))
    _tombstone(conn, source, [clause_id], editor)
    _renumber(conn, source)
    conn.commit(); conn.close()
    refresh_kb()
    return True


def move_clause(source, clause_id, direction):
    conn = sqlite3.connect(DB_NAME)
    seq = conn.execute("SELECT clause_id, COALESCE(sort_order, rowid) FROM regulatory_clauses "
                       "WHERE source_doc=? ORDER BY COALESCE(sort_order, rowid)", (str(source),)).fetchall()
    ids = [row[0] for row in seq]
    if str(clause_id) not in ids:
        conn.close(); return False
    idx = ids.index(str(clause_id))
    swap = idx - 1 if direction == "up" else idx + 1
    if swap < 0 or swap >= len(seq):
        conn.close(); return False
    a, b = seq[idx], seq[swap]
    conn.execute("UPDATE regulatory_clauses SET sort_order=? WHERE source_doc=? AND clause_id=?", (b[1], str(source), a[0]))
    conn.execute("UPDATE regulatory_clauses SET sort_order=? WHERE source_doc=? AND clause_id=?", (a[1], str(source), b[0]))
    conn.commit(); conn.close()
    refresh_kb()
    return True


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

def update_document_meta(source, new_source=None, doc_type=None, doc_category=None):
    """Admin: rename a document and/or change its type-band / department across ALL
    its clauses (and cascade the rename to related tables). Returns (ok, error)."""
    source = str(source)
    new_source = (new_source or "").strip() or None
    conn = sqlite3.connect(DB_NAME)
    try:
        if not conn.execute("SELECT 1 FROM regulatory_clauses WHERE source_doc=? LIMIT 1", (source,)).fetchone():
            return False, "Document not found."
        if new_source and new_source != source and conn.execute(
                "SELECT 1 FROM regulatory_clauses WHERE source_doc=? LIMIT 1", (new_source,)).fetchone():
            return False, f"A document named '{new_source}' already exists."
        sets, params = [], []
        if new_source and new_source != source:
            sets.append("source_doc=?"); params.append(new_source)
        if doc_type:
            sets.append("doc_type=?"); params.append(doc_type)
        if doc_category:
            sets.append("doc_category=?"); params.append(doc_category)
        if not sets:
            return True, None
        conn.execute(f"UPDATE regulatory_clauses SET {', '.join(sets)} WHERE source_doc=?", (*params, source))
        if new_source and new_source != source:
            for tbl in ("clause_versions", "document_assets", "editing_sessions"):
                try:
                    conn.execute(f"UPDATE {tbl} SET source_doc=? WHERE source_doc=?", (new_source, source))
                except sqlite3.OperationalError:
                    pass
        conn.commit()
    finally:
        conn.close()
    refresh_kb()
    return True, None


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
    m = (module or "").strip().lower()
    if m in ("", "universal", "all"): return df
    if m == "data": return pd.DataFrame(columns=df.columns)
    # A department module shows its own Doc_Category, plus cross-cutting GENERAL
    # documents only if that department opts in. The category key is the module
    # slug upper-cased (health->HEALTH, hr->HR), so new departments need no code.
    key = m.upper()
    dept = next((d for d in load_departments() if d["key"] == key), None)
    cats = [key]
    if dept is None or dept.get("general"):
        cats.append("GENERAL")
    return df[df["Doc_Category"].isin(cats)]

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
            single_target = len(target.split()) == 1
            for doc_tag in tag_list:
                if doc_tag == target or normalize_tag_text(doc_tag) == normalize_tag_text(target):
                    found = True; break
                if target_stem == get_stem(doc_tag) and single_target and len(doc_tag.split()) == 1:
                    found = True; break
                # A single query word also matches a multi-word tag when it equals
                # (by stem) one of the words inside that tag, so "criticism"
                # surfaces the tag "criticism of authority or government".
                if single_target and len(doc_tag.split()) > 1:
                    if any(target == tw or target_stem == get_stem(tw) for tw in doc_tag.split()):
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


def deep_scan_brain(keyword_tuples, df, exclude_ids=None, module="universal", phrase=None):
    scoped_df: pd.DataFrame = filter_df_by_module(df, module)
    if scoped_df.empty: return []

    # Keep the core query stems (the words the user typed) separate from synonym
    # expansions, so relevance can reward the words actually typed.
    core_stems, search_stems = [], set()
    # The literal surface words the user typed (e.g. "criticism"), kept apart from
    # their stem ("critic"). Porter collapses "criticism"/"critical"/"criticize"
    # to the same stem, so without this the ranker can't tell them apart and a
    # "critical"-heavy clause outranks the "criticism" the user actually wanted.
    core_words = []
    for raw, clean in keyword_tuples:
        if clean not in STOPWORDS_STRONG:
            # Use a prefix that is genuinely a prefix of the typed word. Porter can
            # restore letters so the stem ("price" from "pricing") won't match the
            # word via `\broot\w*`; search_root collapses to the common prefix
            # ("pric") only in that case, otherwise the stem is unchanged.
            root = search_root(raw, clean)
            if root not in core_stems:
                core_stems.append(root)
            search_stems.add(root)
            # Only genuine single typed words (skip synthesized multi-word tags).
            rw = str(raw).lower().strip()
            if rw and " " not in rw and rw not in core_words:
                core_words.append(rw)
            if clean in SYNONYM_MAP:
                for syn in SYNONYM_MAP[str(clean)]:
                    search_stems.add(get_stem(syn))

    # Normalised verbatim phrase the user typed (for exact-phrase ranking).
    phrase_l = re.sub(r"\s+", " ", str(phrase or "").strip().lower()) or None
    exclude_set: set[str] = set(exclude_ids) if exclude_ids else set()
    matches = []

    for _, row in scoped_df.iterrows():
        if row.get("Is_Header"): continue
        c_id = str(row.get("Clause_ID", "")).strip()
        if c_id in exclude_set: continue

        text = str(row.get("Clause_Text", "")).lower()
        text_join = text.replace("-", "")     # hyphen-collapsed copy for matching
        hit_stems = [s for s in search_stems if _root_present(s, text, text_join)]
        if not hit_stems:
            continue

        # Relevance score (dominates the regulatory-hierarchy order below):
        #   verbatim phrase >> all typed words present >> more distinct words >> density.
        score = 0
        if phrase_l and " " in phrase_l and (phrase_l in text or phrase_l.replace("-", "") in text_join):
            score += 1000
        core_hits = [s for s in core_stems if _root_present(s, text, text_join)]
        if core_stems and len(core_hits) == len(core_stems):
            score += 200            # every word the user typed appears in this clause
        score += 10 * len(core_hits)
        # Reward the exact surface word the user typed over same-stem cousins:
        # a clause with literal "criticism" beats one that only has "critical"
        # (prefix on the full word, so "criticism" also catches "criticisms" but
        # never "critical"). 50 dominates the density term below.
        score += 50 * sum(1 for w in core_words if _root_present(w, text, text_join))
        score += sum(_root_count(s, text, text_join) for s in core_hits)

        matches.append({
            "source": row.get("Source_Doc", "UNKNOWN"),
            "type": row.get("Doc_Type", "UNKNOWN"),
            "priority": row.get("Priority", 99),
            "id": c_id,
            "header": row.get("Context_Header", ""),
            "raw_text": str(row.get("Clause_Text", "")),
            "_score": score,
        })

    # Hierarchy order first, then a stable sort by relevance so the best textual
    # matches (e.g. the clause with the exact phrase) rise to the top.
    matches = sort_matches(matches)
    matches.sort(key=lambda m: -m.get("_score", 0))
    return matches

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

    # --- Preserve in-app work that the Excel rebuild would otherwise wipe -----
    # The sync rebuilds clauses from the knowledge_base Excel/CSV files, but
    # Document Studio imports (PDF->clauses) and in-app clause edits/retags live
    # only in the DB. Snapshot them so we can restore them after the rebuild.
    cols = [r[1] for r in c.execute("PRAGMA table_info(regulatory_clauses)").fetchall()]
    snapshot = [dict(zip(cols, row))
                for row in c.execute(f"SELECT {', '.join(cols)} FROM regulatory_clauses").fetchall()]
    # Clauses the user deleted in Document Studio — never re-create them from Excel.
    tombstones = _load_tombstones(c)

    def _excel_src(fn):
        return re.sub(r"\.(xlsx|xls|csv|xlxs)$", "", fn, flags=re.IGNORECASE).replace("_", " ").upper().title()
    excel_sources = {_excel_src(os.path.basename(p)) for p in doc_files}

    db_only_rows = []   # whole documents not backed by any Excel file (Studio imports)
    edits = {}          # (source, clause_id) -> row, for clauses edited/retagged in-app
    for r in snapshot:
        src = str(r.get("source_doc") or "")
        if src not in excel_sources:
            db_only_rows.append(r)
        elif str(r.get("clause_html") or "").strip() or str(r.get("updated_by") or "").strip():
            edits[(src, str(r.get("clause_id") or ""))] = r

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

                cid = str(row.get(clause_id_col, "")).strip() if clause_id_col else ""
                # Don't resurrect a clause the user deleted in Document Studio.
                if cid and (source_doc.title(), cid) in tombstones:
                    continue

                insert_rows.append((
                    source_doc.title(),
                    category,
                    doc_type,
                    cid,
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

    # --- Restore the preserved in-app work -----------------------------------
    restored_docs, restored_edits = 0, 0
    if db_only_rows:
        ins_cols = [col for col in cols if col != "id"]   # let id autoincrement
        placeholders = ", ".join(["?"] * len(ins_cols))
        c.executemany(
            f"INSERT INTO regulatory_clauses ({', '.join(ins_cols)}) VALUES ({placeholders})",
            [tuple(r.get(col) for col in ins_cols) for r in db_only_rows],
        )
        restored_docs = len({r.get("source_doc") for r in db_only_rows})
    preserve = [col for col in ("clause_html", "regulatory_tags", "clause_text",
                                "sort_order", "updated_at", "updated_by") if col in cols]
    if preserve:
        set_sql = ", ".join(f"{col}=?" for col in preserve)
        for (src, cid), r in edits.items():
            c.execute(
                f"UPDATE regulatory_clauses SET {set_sql} WHERE source_doc=? AND clause_id=?",
                tuple(r.get(col) for col in preserve) + (src, cid),
            )
            if c.rowcount:
                restored_edits += 1

    conn.commit()
    conn.close()

    load_knowledge_base(force_reload=True)
    extra = ""
    if restored_docs or restored_edits:
        extra = f" | preserved {restored_docs} imported doc(s) + {restored_edits} in-app edit(s)"
    return f"[+] Regulatory sync complete: {total_files} files ({total_rows} clauses){extra}."

# Obsolete seed/POC files that must never be re-ingested by a sync: the curated
# iris.db supersedes them, and re-adding their rows would resurrect data we
# deliberately removed (POC unified_database.csv) or de-duplicated (HDFC ERGO).
SYNC_SKIP_FILES = {"unified_database.csv",
                   "handbook_2024-25_parts.xlsx", "handbook_2024-25_summary.xlsx"}


def aggregate_submissions():
    """
    Ingest Excel/CSV files from raw_submissions into financial_metrics.

    NON-DESTRUCTIVE / ADDITIVE: the curated DB is authoritative. We no longer wipe
    the table — that used to destroy every surgical handbook re-ingest (Tables
    4/21/66/69/78/82/100/102/AUM …) and resurrect removed POC/HDFC-ERGO rows. A
    row is inserted only if its key (dimension, insurer, year, quarter, metric,
    LOB, class) isn't already present, so existing curated rows always win and a
    new submission file only adds genuinely new keys. Supports a 'dimension'
    column (defaults to 'Insurer'); converts '-' quarters to 'Annual'.
    """
    all_files = []
    for root, dirs, files in os.walk(RAW_SUBMISSIONS_FOLDER):
        for file in files:
            if file.startswith("~$"): continue                 # Excel temp files
            if file in SYNC_SKIP_FILES: continue               # obsolete seed/POC
            if file.endswith(".xlsx") or file.endswith(".csv"):
                all_files.append(os.path.join(root, file))

    if not all_files:
        return "[-] No new files found in 'raw_submissions' or subfolders."

    added_rows = 0
    skipped_existing = 0
    total_files = 0

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()

        # --- MIGRATION: Ensure columns exist in DB ---
        try: c.execute("ALTER TABLE financial_metrics ADD COLUMN dimension TEXT DEFAULT 'Insurer'")
        except sqlite3.OperationalError: pass
        try: c.execute("ALTER TABLE financial_metrics ADD COLUMN source_file TEXT")
        except sqlite3.OperationalError: pass

        # Snapshot existing keys — curated data is authoritative and is preserved.
        def _key(insurer, fy, q, metric, lob, cob, dim):
            return (str(dim).strip(), str(insurer).strip(), str(fy).strip(),
                    str(q).strip(), str(metric).strip(), str(lob).strip(), str(cob).strip())
        existing = set(c.execute(
            "SELECT dimension, insurer, financial_year, quarter, metric, "
            "line_of_business, class_of_business FROM financial_metrics"))
        existing = {tuple((x or "").strip() for x in row) for row in existing}

        for file_path in all_files:
            filename = os.path.basename(file_path)
            try:
                df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
                if df.empty: continue
                df.columns = [str(col).strip().replace(" ", "_").lower() for col in df.columns]

                rows_to_insert = []
                for _, row in df.iterrows():
                    dim_raw = row.get('dimension')
                    if not dim_raw:
                        if 'insurer' in df.columns: dim_raw = 'Insurer'
                        elif 'state' in df.columns: dim_raw = 'State'
                        elif 'tpa' in df.columns: dim_raw = 'TPA'
                        else: dim_raw = 'Insurer'
                    entity_raw = (row.get('insurer') or row.get('entity') or
                                  row.get('state') or row.get('tpa') or "Unknown")

                    dim = str(dim_raw).strip()
                    entity_name = str(entity_raw).strip()
                    metric_name = str(row.get('metric', '')).strip()
                    fy = str(row.get('financial_year')).replace('.0', '').strip()
                    quarter_val = str(row.get('quarter', 'Annual')).strip()
                    if quarter_val in ['-', 'nan', 'None', '']:
                        quarter_val = 'Annual'
                    lob = str(row.get('line_of_business', 'General')).strip()
                    cob = str(row.get('class_of_business', 'General')).strip()

                    if not metric_name or pd.isna(row.get('value')):
                        continue

                    k = _key(entity_name, fy, quarter_val, metric_name, lob, cob, dim)
                    if k in existing:                       # curated/existing wins
                        skipped_existing += 1
                        continue
                    existing.add(k)                         # also dedups within the file
                    rows_to_insert.append((entity_name, fy, quarter_val, metric_name,
                                           row.get('value'), lob, cob, dim, filename))

                if rows_to_insert:
                    c.executemany(
                        "INSERT INTO financial_metrics (insurer, financial_year, quarter, "
                        "metric, value, line_of_business, class_of_business, dimension, source_file) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows_to_insert)
                    total_files += 1
                    added_rows += len(rows_to_insert)
            except Exception as e:
                print(f"[!] Error processing {filename}: {e}")

        conn.commit()
        conn.close()

        load_master_data_engine()
        return (f"[+] Sync complete: added {added_rows} new rows from {total_files} file(s); "
                f"preserved existing data ({skipped_existing} duplicate rows skipped).")

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

HDFC_ERGO_DISCLAIMER = (
    "HDFC ERGO General Insurance is a merged entity — it absorbed L&T General "
    "Insurance (2016-17) and Apollo Munich / HDFC ERGO Health (2020). Figures for "
    "the merger-era years reflect entity consolidation and may not be directly "
    "comparable across years. For merger-era figures, verify against the IRDAI "
    "Handbook before relying on them.")


def load_data_quality_flags():
    """Sections (dimension, line_of_business) whose source table lost a
    sub-dimension — surfaced as cautions in the Data Explorer. Cached; empty list
    on un-migrated DBs (no data_quality_flags table)."""
    if 'dq_flags' in _CACHE:
        return _CACHE['dq_flags']
    flags = []
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.execute("SELECT dimension, line_of_business, severity, note, "
                           "conflict_rows, total_rows FROM data_quality_flags")
        flags = [{"dimension": d, "line_of_business": lob, "severity": sev,
                  "note": note, "conflict_rows": cr, "total_rows": tr}
                 for d, lob, sev, note, cr, tr in cur.fetchall()]
        conn.close()
    except Exception:
        flags = []
    _CACHE['dq_flags'] = flags
    return flags


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

    # Type metadata from the categorization layer (categorize_dimensions.py):
    # value -> type, so the UI can group the dropdowns and hide insurer names
    # ('entity') that leaked into class_of_business. Absent on un-migrated DBs.
    def _meta(col, typecol):
        if col in UNIFIED_DF.columns and typecol in UNIFIED_DF.columns:
            sub = UNIFIED_DF[[col, typecol]].dropna().drop_duplicates()
            return {str(k): str(v) for k, v in zip(sub[col], sub[typecol])}
        return {}
    result["lob_meta"] = _meta('Line_of_Business', 'lob_type')
    result["cob_meta"] = _meta('Class_of_Business', 'cob_type')
    result["section_flags"] = load_data_quality_flags()

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

        # Computed totals: the Handbook omits industry/group totals, so users used
        # to export every insurer and sum in Excel. Append a TOTAL row = sum across
        # the selected entities for each Year/LOB/Class/Quarter. Unit-safe: only
        # additive metrics (amounts, counts) are summed; percentages and ratios are
        # left blank (summing them is meaningless), and value columns are consistent
        # within a metric so a plain column-sum is correct.
        if filters.get('add_total'):
            n_ent = df['Entity'].nunique()
            if n_ent >= 2:
                t_index = [c for c in ['Financial_Year', 'Line_of_Business',
                           'Class_of_Business', 'Quarter', 'Source_File'] if c in df.columns]
                tot = df.pivot_table(index=t_index, columns='Metric',
                                     values='Value', aggfunc='sum').reset_index()
                tot.insert(0, 'Entity', f'TOTAL ({n_ent})')
                if 'unit_code' in UNIFIED_DF.columns:
                    munit = dict(zip(UNIFIED_DF['Metric'].astype(str),
                                     UNIFIED_DF['unit_code'].astype(str)))
                    ADDITIVE = {'INR', 'COUNT', 'NUMBER', 'USD'}
                    for c in list(tot.columns):
                        if str(c) in munit and munit[str(c)] not in ADDITIVE:
                            tot[c] = pd.NA   # don't sum a % / ratio / unitless metric
                pivot_df = pd.concat([pivot_df, tot], ignore_index=True)

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
    
    # Surface data-quality cautions for the sections actually in this report.
    cautions = []
    if 'Line_of_Business' in pivot_df.columns:
        shown_lobs = set(pivot_df['Line_of_Business'].astype(str).unique())
        for f in load_data_quality_flags():
            if f['dimension'] == target_dim and f['line_of_business'] in shown_lobs:
                cautions.append(f)

    # Entity disclaimer: HDFC ERGO merger lineage. Show whenever HDFC ERGO appears
    # anywhere in the report (as the entity, or as a class — e.g. State views).
    text_cells = set()
    for col in (target_dim, 'Class_of_Business'):
        if col in pivot_df.columns:
            text_cells |= set(pivot_df[col].astype(str).unique())
    if any('HDFC ERGO' in s for s in text_cells):
        cautions.append({"dimension": "Entity note", "line_of_business": "HDFC ERGO",
                         "severity": "info", "note": HDFC_ERGO_DISCLAIMER})

    return {'columns': final_cols, 'rows': pivot_df.to_dict('records'),
            'missing': missing_alerts, 'risks': risk_alerts, 'cautions': cautions}

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
