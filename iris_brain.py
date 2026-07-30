import sqlite3
import pandas as pd # type: ignore
import os
import re
import functools
import bisect
import json
import unicodedata
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
# Function words that must never become SEARCH keywords. Left in, they match huge
# numbers of irrelevant clauses (a tag like "Returns_to_be_Published" matches "be",
# "given"/"as" appear in nearly every clause) — and because these words are also
# excluded from highlighting, those results show up with nothing highlighted. This
# is the single source of truth: get_clean_keywords drops these, and the highlighter
# uses the same set, so every search keyword is highlightable and vice-versa.
FUNCTION_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "nor", "for", "so", "yet",
    "of", "to", "in", "on", "at", "by", "with", "from", "into", "onto", "upon",
    "over", "under", "out", "off", "up", "down",
    "as", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "have", "has", "had", "having",
    "that", "this", "these", "those", "there", "here", "it", "its",
    "their", "his", "her", "our", "your", "my",
    "no", "not", "any", "all", "each", "some", "such", "other", "another", "same",
    "shall", "may", "must", "will", "would", "can", "could", "should", "might",
    "which", "who", "whom", "whose", "what", "when", "where", "why", "how",
    "than", "then", "if", "unless", "until", "while", "whether", "because",
    "although", "though", "per", "via", "he", "she", "they", "we", "you",
    "him", "them", "us", "made", "make", "given", "give",
})
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
@functools.lru_cache(maxsize=100000)
def get_stem(word):
    # Pure + deterministic -> safe to memoize. Stemming the same words repeatedly
    # (per row, per request) is a hot path; caching removes ~all of that cost.
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

@functools.lru_cache(maxsize=100000)
def search_root(word, stem=None):
    """Prefix safe to use for `\\broot\\w*` matching/highlighting. Porter sometimes
    restores letters so the stem is NOT a prefix of the word it came from
    (pricing -> 'price', which then can't match 'pricing'). Fall back to the
    common prefix of the word and its stem so the root always *is* a real prefix
    of the typed word. No-op when the stem is already a prefix (the usual case)."""
    w = str(word).lower()
    s = stem if stem is not None else get_stem(w)
    # A root shorter than 3 chars is dangerously broad: "cis" (an acronym) Porter-
    # stems to "ci", and `\bci\w*` then matches "circular", "citizen", etc. Match
    # such short words VERBATIM instead of by a tiny prefix.
    if s and w.startswith(s) and len(s) >= 3:
        return s                       # normal case — stem is a clean prefix
    cp = common_prefix(w, s or "")
    return cp if len(cp) >= 3 else w

@functools.lru_cache(maxsize=4096)
def _root_re(root):
    """Compiled `\\broot\\w*` matcher, memoized on the root.

    These two functions are the hottest code in the whole search path: profiled on
    a 19-word universal query, _root_present alone accounts for 37,730 calls and
    re.Pattern.search is 62% of the request. Building the pattern STRING on every
    call meant re.escape ran 60,005 times and re's internal cache was re-entered
    93,204 times per request, purely to rediscover the same handful of patterns.
    The root vocabulary per query is tiny (a dozen), so caching the compiled object
    removes all of that. Bounded at 4096 because roots come from user queries.
    """
    return re.compile(rf"\b{re.escape(root)}\w*")

def _root_present(root, text, text_join):
    """Does `root` prefix-match a word in the text? Checks the text AND a
    hyphen-collapsed copy so a typed 'deempanelment' matches 'de-empanelment'
    (and vice-versa) — hyphenation in regulatory terms shouldn't break search."""
    rx = _root_re(root)
    return bool(rx.search(text)) or bool(rx.search(text_join))

def _root_count(root, text, text_join):
    rx = _root_re(root)
    a = rx.findall(text)
    return len(a) if a else len(rx.findall(text_join))

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

# Precomputed, search-ready text columns. Building these ONCE at load time avoids
# lower-casing + hyphen-collapsing every clause on every search request.
#   _CT_LC  = Clause_Text, lower-cased
#   _CT_LCJ = _CT_LC with hyphens removed (matches "de-empanelment" <-> "deempanelment")
#
# INVARIANT: _CT_LC and _CT_LCJ must always mirror Clause_Text. Do NOT assign to
# KB_CACHE_DF["Clause_Text"] directly — write it ONLY through set_clause_text(),
# which refreshes the derived columns in the same call so they can never go stale.
# (Bulk changes go through refresh_kb()/load_knowledge_base(), which rebuild both.)
# Search code also falls back to computing on the fly if the columns are absent, so
# a missing column is only slower, never wrong.
_CT_LC, _CT_LCJ = "_ct_lc", "_ct_lcj"

def _derive_search_columns(text):
    """Canonical normalization for the search columns — the single source of truth
    for how Clause_Text maps to its searchable forms. If the rule ever changes
    (e.g. stripping more punctuation), change it here only."""
    lc = (text or "").lower()
    return lc, lc.replace("-", "")

def _add_search_cols(df):
    if df is None or getattr(df, "empty", True) or "Clause_Text" not in df.columns:
        return df
    derived = df["Clause_Text"].fillna("").astype(str).map(_derive_search_columns)
    df[_CT_LC] = derived.str[0]
    df[_CT_LCJ] = derived.str[1]
    # The index is derived from these columns, so it is rebuilt in the same call
    # that rebuilds them — the two can never drift. Both KB load paths
    # (load_knowledge_base, refresh_kb) route through here.
    build_prefix_index(df)
    return df

# ---------------------------------------------------------------------------
# Prefix index over the clause corpus.
#
# `\broot\w*` asks exactly one question: "does some word here START with root?"
# The regex answers it by rescanning ~1,875 chars of clause text, once per root,
# per clause, on every request — profiled at 37,730 _root_present calls and 62%
# of a 19-word universal query. That is a prefix LOOKUP being computed as a scan.
#
# So invert it once at load: a sorted vocabulary + postings (which clauses hold
# each word). bisect to the prefix range, union the postings, and the answer for
# ALL clauses arrives in one lookup per root. Measured on the real KB: 195.1ms of
# regex for 11 roots becomes 0.08ms, with byte-identical result sets.
#
# Both text forms feed ONE index. _root_present is `raw OR hyphen-collapsed`, so
# indexing the tokens of both and taking the union reproduces it exactly — no
# second structure needed. (_root_count is NOT a union — it counts in the raw text
# and only falls back to the collapsed form when that is zero — so it deliberately
# stays on the regex path.)
#
# Postings hold DataFrame index LABELS, not row positions: module scoping filters
# the frame with a boolean mask, which preserves labels but renumbers positions.
#
# Each posting also carries WHERE in the clause the word occurs, as a bitmask of
# sentence numbers. That serves the proximity score (_line_cover: "the most query
# roots any ONE sentence holds"), which is otherwise a finditer per root over the
# full clause text and, once presence was indexed, became the single largest cost
# in the request at 31%.
#
# The sentence bits come from the RAW lower-cased text only, never the
# hyphen-collapsed form, because _line_cover matches against the raw text. Tokens
# that exist only after collapsing hyphens ("deempanelment" from "de-empanelment")
# are therefore stored with an EMPTY mask: present for the presence query, and
# contributing no sentence to the proximity query — which is exactly what the
# regex does.
_IDX_TOKEN = re.compile(r"\w+")
# Canonical sentence splitter. api._line_cover imports THIS rather than keeping its
# own copy: the index bakes the segmentation in at load time, so two definitions
# drifting apart would silently corrupt every proximity score.
SENT_SPLIT = re.compile(r"[\n;]+|\.\s")
_IDX_MAP = None      # word -> {index label: sentence bitmask}
_IDX_WORDS = ()      # sorted tuple of _IDX_MAP keys, the bisect target


def _clause_tokens(lc, lcj):
    """{word: sentence bitmask} for one clause, over both search forms."""
    starts = [0]
    for mm in SENT_SPLIT.finditer(lc):
        starts.append(mm.end())
    out = {}
    for mm in _IDX_TOKEN.finditer(lc):
        i = bisect.bisect_right(starts, mm.start()) - 1
        if i < 0:
            continue
        w = mm.group(0)
        out[w] = out.get(w, 0) | (1 << i)
    if "-" in lc:
        for w in _IDX_TOKEN.findall(lcj):
            if w not in out:
                out[w] = 0
    return out


def build_prefix_index(df):
    """(Re)build the whole index from a frame. ~100ms for 1,299 clauses / ~10MB."""
    global _IDX_MAP, _IDX_WORDS
    if df is None or getattr(df, "empty", True) or _CT_LC not in getattr(df, "columns", ()):
        _IDX_MAP, _IDX_WORDS = None, ()
        return
    m = {}
    for label, lc, lcj in zip(df.index, df[_CT_LC].fillna("").astype(str),
                              df[_CT_LCJ].fillna("").astype(str)):
        for w, bits in _clause_tokens(lc, lcj).items():
            d = m.get(w)
            if d is None:
                m[w] = {label: bits}
            else:
                d[label] = bits
    _IDX_MAP = m
    _IDX_WORDS = tuple(sorted(m))


def _idx_update(label, old_tokens, new_tokens):
    """Incremental edit for one clause. Keeps _IDX_WORDS sorted by splicing rather
    than re-sorting 9k words per edit; edits are rare, lookups are not."""
    global _IDX_WORDS
    if _IDX_MAP is None:
        return
    for w in old_tokens.keys() - new_tokens.keys():
        d = _IDX_MAP.get(w)
        if d is not None:
            d.pop(label, None)
            if not d:
                del _IDX_MAP[w]
                i = bisect.bisect_left(_IDX_WORDS, w)
                if i < len(_IDX_WORDS) and _IDX_WORDS[i] == w:
                    _IDX_WORDS = _IDX_WORDS[:i] + _IDX_WORDS[i + 1:]
    # Every surviving word is re-stamped, not just the new ones: an edit moves text
    # around, so a word the clause already had can land in a different sentence.
    for w, bits in new_tokens.items():
        d = _IDX_MAP.get(w)
        if d is None:
            _IDX_MAP[w] = {label: bits}
            i = bisect.bisect_left(_IDX_WORDS, w)
            _IDX_WORDS = _IDX_WORDS[:i] + (w,) + _IDX_WORDS[i:]
        else:
            d[label] = bits


def clauses_with_root(root):
    """Index labels of every clause where `root` prefix-matches a word — the whole
    corpus answer that _root_present gives one clause at a time.

    Returns None when the index cannot answer, and the caller MUST fall back to the
    regex path. That happens when the index isn't built, or when the root is not a
    pure \\w+ token: a compound like "actl/ibnr" contains a separator, so `\\b...\\w*`
    no longer means "a word starts with this" and the equivalence breaks.
    """
    if _IDX_MAP is None or not root:
        return None
    r = str(root).lower()
    if not _IDX_TOKEN.fullmatch(r):
        return None
    i = bisect.bisect_left(_IDX_WORDS, r)
    out = set()
    words = _IDX_WORDS
    n = len(words)
    while i < n and words[i].startswith(r):
        out |= _IDX_MAP[words[i]].keys()
        i += 1
    return out


def sentence_masks(root):
    """{index label: sentence bitmask} for every clause where `root` prefix-matches,
    merged across the words it matches. Bit s set means "a word starting with root
    occurs in sentence s". None on the same conditions as clauses_with_root.

    This is the whole-corpus form of _line_cover's inner loop: with one of these per
    query root, the proximity score for a clause is a popcount over the ANDed bits
    instead of a finditer per root over the full text.
    """
    if _IDX_MAP is None or not root:
        return None
    r = str(root).lower()
    if not _IDX_TOKEN.fullmatch(r):
        return None
    i = bisect.bisect_left(_IDX_WORDS, r)
    words = _IDX_WORDS
    n = len(words)
    out = {}
    while i < n and words[i].startswith(r):
        for label, bits in _IDX_MAP[words[i]].items():
            if bits:
                out[label] = out.get(label, 0) | bits
        i += 1
    return out


def set_clause_text(mask, text):
    """The ONLY sanctioned way to write Clause_Text into the in-memory cache. Keeps
    the derived search columns AND the prefix index in lock-step with the text so
    they can never go stale (see the INVARIANT above). `mask` is any pandas row
    selector."""
    lc, lcj = _derive_search_columns(text)
    # Read the OLD tokens before overwriting, so the index can drop them. Doing this
    # from the frame means we never need a second label->words map to mirror.
    stale = []
    if _IDX_MAP is not None and _CT_LC in KB_CACHE_DF.columns:
        old = KB_CACHE_DF.loc[mask, [_CT_LC, _CT_LCJ]]
        for label, o_lc, o_lcj in zip(old.index, old[_CT_LC].fillna("").astype(str),
                                      old[_CT_LCJ].fillna("").astype(str)):
            stale.append((label, _clause_tokens(o_lc, o_lcj)))
    KB_CACHE_DF.loc[mask, "Clause_Text"] = text
    if _CT_LC in KB_CACHE_DF.columns:
        KB_CACHE_DF.loc[mask, _CT_LC] = lc
        KB_CACHE_DF.loc[mask, _CT_LCJ] = lcj
    if stale:
        fresh = _clause_tokens(lc, lcj)
        for label, old_tokens in stale:
            _idx_update(label, old_tokens, fresh)

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
        # This branch bypasses _add_search_cols, so clear the index explicitly —
        # otherwise a reload that finds an empty table would leave the postings of
        # the PREVIOUS corpus in place, and search would answer from clauses the
        # cache no longer holds.
        build_prefix_index(df)
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

    _add_search_cols(df)
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
    _add_search_cols(df)
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
        set_clause_text(mask, text)   # writes Clause_Text + derived search columns atomically
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
        if w in STOP_WORDS or w in FUNCTION_WORDS: continue
        valid_word = w
        corrected_word = None

        if w not in KNOWN_VOCAB:
            # Only spell-correct a GENUINELY unknown word. A valid word whose root
            # already prefixes known vocabulary (e.g. "deductions" -> root "deduct",
            # which matches "deduction") must NOT be rewritten to a lexically-near
            # but unrelated word ("directions"). Also use a stricter cutoff (0.86):
            # the false rewrites ("deductions"->"directions", "actual"->"actuarial")
            # sit at 0.80, while real single-char typos score higher.
            _rs = get_stem(w)
            root_known = len(_rs) >= 3 and any(v.startswith(_rs) for v in KNOWN_VOCAB)
            if not root_known:
                matches = difflib.get_close_matches(w, list(KNOWN_VOCAB), n=1, cutoff=0.86)
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
    # Precompute each target's stem + single-word flag ONCE (was recomputed for
    # every clause row); pure and identical, just hoisted out of the hot loop.
    targets = [(target, get_stem(target), len(target.split()) == 1) for target in detected_tags]
    # Skip clauses with no tags without materialising a row (the loop still guards).
    scoped_df = scoped_df[scoped_df["Regulatory_Tags"].fillna("").astype(str).str.strip() != ""]

    for _, row in scoped_df.iterrows():
        if row.get("Is_Header"): continue
        raw_tags = str(row.get("Regulatory_Tags", "")).lower()
        if not raw_tags: continue

        tag_list = [t.strip().replace("_", " ") for t in raw_tags.split(",")]

        found = False
        for target, target_stem, single_target in targets:
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
                "tag": str(row.get("Regulatory_Tags", "")),
                "raw_text": str(row.get("Clause_Text", ""))
            })
    return sort_matches(matches)

def search_citation(code, df):
    """Literal lookup of a regulatory citation / reference code (e.g.
    'IRDAI/Actl/IBNR/AIC/2009-10'). The code is a single reference, so match it as a
    substring of the clause text / heading / tags (case-insensitive, and ignoring
    incidental whitespace) instead of tokenising it into unrelated keywords. Returns
    the same match dicts as the other searchers; empty when the code isn't present."""
    if df is None or df.empty:
        return []
    needle = str(code).strip().lower()
    needle_ns = re.sub(r"\s+", "", needle)
    if len(needle_ns) < 3:
        return []
    out = []
    for _, row in df.iterrows():
        if row.get("Is_Header"):
            continue
        hay = " ".join(str(row.get(c, "")) for c in
                       ("Clause_Text", "Context_Header", "Regulatory_Tags")).lower()
        if needle in hay or needle_ns in re.sub(r"\s+", "", hay):
            out.append({
                "source": row.get("Source_Doc", "UNKNOWN"),
                "type": row.get("Doc_Type", "UNKNOWN"),
                "priority": row.get("Priority", 99),
                "id": str(row.get("Clause_ID", "")).strip(),
                "header": row.get("Context_Header", ""),
                "tag": str(row.get("Regulatory_Tags", "")),
                "raw_text": str(row.get("Clause_Text", "")),
            })
    return sort_matches(out)


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

    # Vectorized candidate pre-filter: only clauses whose text contains at least one
    # search stem (hyphen-insensitive, the SAME condition the loop applies) can ever
    # score, so run the expensive per-row logic on just those instead of all ~1300
    # clauses. Result set is IDENTICAL — the loop below still re-checks every row.
    if not search_stems:
        return []
    _alt = "|".join(re.escape(s) for s in search_stems)
    _pat = rf"\b(?:{_alt})\w*"
    if _CT_LC in scoped_df.columns:                       # precomputed (fast path)
        _lc, _lcj = scoped_df[_CT_LC], scoped_df[_CT_LCJ]
        _mask = _lc.str.contains(_pat, regex=True) | _lcj.str.contains(_pat, regex=True)
    else:                                                 # fallback: compute on the fly
        _txt = scoped_df["Clause_Text"].fillna("").astype(str).str.lower()
        _mask = (_txt.str.contains(_pat, regex=True)
                 | _txt.str.replace("-", "", regex=False).str.contains(_pat, regex=True))
    scoped_df = scoped_df[_mask]

    # Whole-corpus presence for every root this loop will test, resolved ONCE from
    # the prefix index instead of once per (root, clause). This is the hot spot:
    # the loop below asks _root_present up to ~40 times per candidate, and there can
    # be 664 candidates. A root whose value is None is one the index cannot answer
    # (a non-\w compound), and it alone falls back to the regex.
    _idx_sets = {r: clauses_with_root(r)
                 for r in set(search_stems) | set(core_stems) | set(core_words)}

    def _has(root, label, text, text_join):
        s = _idx_sets.get(root)
        if s is None:
            return _root_present(root, text, text_join)
        return label in s

    for label, row in scoped_df.iterrows():
        if row.get("Is_Header"): continue
        c_id = str(row.get("Clause_ID", "")).strip()
        if c_id in exclude_set: continue

        text = str(row.get("Clause_Text", "")).lower()
        text_join = text.replace("-", "")     # hyphen-collapsed copy for matching
        hit_stems = [s for s in search_stems if _has(s, label, text, text_join)]
        if not hit_stems:
            continue

        # Relevance score (dominates the regulatory-hierarchy order below):
        #   verbatim phrase >> all typed words present >> more distinct words >> density.
        score = 0
        if phrase_l and " " in phrase_l and (phrase_l in text or phrase_l.replace("-", "") in text_join):
            score += 1000
        core_hits = [s for s in core_stems if _has(s, label, text, text_join)]
        if core_stems and len(core_hits) == len(core_stems):
            score += 200            # every word the user typed appears in this clause
            # ...and if they all appear together in ONE line/provision, prefer it
            # strongly over clauses where the words are merely scattered about.
            # Stays on regex: this asks about a single LINE, not the clause, and the
            # index is clause-granular. It is also gated behind "all core stems
            # present", which is rare, so it is not a hot path.
            if len(core_stems) >= 2 and any(
                    all(_root_present(s, ln, ln.replace("-", "")) for s in core_stems)
                    for ln in text.split("\n")):
                score += 500
        score += 10 * len(core_hits)
        # Reward the exact surface word the user typed over same-stem cousins:
        # a clause with literal "criticism" beats one that only has "critical"
        # (prefix on the full word, so "criticism" also catches "criticisms" but
        # never "critical"). 50 dominates the density term below.
        score += 50 * sum(1 for w in core_words if _has(w, label, text, text_join))
        score += sum(_root_count(s, text, text_join) for s in core_hits)

        matches.append({
            "source": row.get("Source_Doc", "UNKNOWN"),
            "type": row.get("Doc_Type", "UNKNOWN"),
            "priority": row.get("Priority", 99),
            "id": c_id,
            "header": row.get("Context_Header", ""),
            "tag": str(row.get("Regulatory_Tags", "")),
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


# ---------------------------------------------------------------------------
# Entity classification, derived IN MEMORY at load time.
#
# tools/classify_entities.py writes entity_type/insurer_class into the database,
# but the app must never DEPEND on that having been run: production restores its
# SQLite from the litestream GCS replica, not from the committed seed, so a
# migration applied to a local file never reaches prod. Deriving here means the
# Insurer 360 screens work on any environment, migrated or not, and the DB columns
# (when present) are purely a convenience for ad-hoc SQL.
#
# Rules mirror tools/classify_entities.py exactly. If they ever diverge, this one
# is authoritative for the app.
_DIM_TO_ENTITY = {
    "State": "state", "Channel": "channel", "Country": "country",
    "Ombudsman": "ombudsman_centre", "TPA": "tpa", "Industry": "aggregate",
}
_AGG_EXACT = {"total", "grand_total", "all_india", "industry_total"}
_AGG_SUFFIX = ("_sector", "_industry", "_industry_all", "_total")
_AGG_CONTAINS = ("private_sector_insurers", "public_sector_insurers",
                 "stand_alone_health_insurers", "standalone_health_insurers")
_INDIAN_REINSURERS = {"general_insurance_corporation_of_india_gic_re", "iti_reinsurance_ltd"}
_CLASS_OVERRIDES = {
    "national_insurance_co_ltd": "General", "the_new_india_assurance_co_ltd": "General",
    "the_oriental_insurance_co_ltd": "General", "united_india_insurance_co_ltd": "General",
    "ecgc_ltd": "General",
}


_INSURER_ALIAS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "tools", "insurer_aliases.json")


def _apply_insurer_aliases(df):
    """Merge insurer_ids that the alias file says are one company.

    Applied to the in-memory frame, deliberately, rather than as a DB migration:
    production restores iris.db from the Litestream replica, so a local UPDATE
    would never arrive. Same reasoning as _derive_entity_columns — the fix ships
    with the code.

    Splitting one insurer across two ids is quiet but corrosive: it strands part
    of the book (Galaxy's Personal Accident line, 6,390 policies, sat outside its
    own profile) and inflates every count that treats ids as companies, so SAHI
    read as 10 insurers rather than 9 in the concentration figures.
    """
    if df is None or getattr(df, "empty", True) or "insurer_id" not in df.columns:
        return df
    try:
        with open(_INSURER_ALIAS_FILE, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except (OSError, ValueError):
        return df                      # an override, never a prerequisite

    def _slug(text):
        s = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
        return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()[:60]

    id_remap, name_remap, class_pin = {}, {}, {}
    for group in spec.get("aliases") or []:
        canonical = (group.get("canonical") or "").strip()
        variants = [v for v in (group.get("variants") or []) if (v or "").strip()]
        if not canonical or len(variants) < 2:
            continue
        target = _slug(canonical)
        if group.get("insurer_class"):
            class_pin[target] = group["insurer_class"]
        for v in variants:
            id_remap[_slug(v)] = target
            name_remap[v.strip()] = canonical
    if not id_remap:
        return df

    hit = df["insurer_id"].isin(id_remap)
    if hit.any():
        df.loc[hit, "insurer_id"] = df.loc[hit, "insurer_id"].map(id_remap)
        # Keep the display name in step, or the picker shows the merged entity
        # under whichever raw label happened to come first.
        for col in ("insurer", "Insurer", "Entity"):
            if col in df.columns:
                df[col] = df[col].map(lambda v: name_remap.get(str(v).strip(), v))
    # An explicitly declared class overrides whatever is stored or derived. The
    # name-based derivation cannot be relied on here: _derive_entity_columns returns
    # early whenever the DB already carries insurer_class, so a stored misclassification
    # would survive the rename untouched. CAL sat in a life-only micro-insurance table
    # while filed as General.
    if class_pin and "insurer_class" in df.columns:
        for iid, cls in class_pin.items():
            df.loc[df["insurer_id"] == iid, "insurer_class"] = cls
    return df


_INSURER_REGISTRY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "tools", "insurer_registry.json")

# Words that carry no identity, dropped before matching a name against the registry.
# Class words (life/general/health/re) are deliberately KEPT — they are what separates
# SBI Life from SBI General, and dropping them would fuse two real companies.
# "corporation" is NOT noise. Stripping it reduced "General Insurance Corporation of
# India" to the single token {general}, which then EXACTLY matched "L&T General
# Insurance Co. Ltd." - declaring a general insurer to be the national reinsurer. Few
# insurers are a Corporation, so the word carries real identity.
_REG_NOISE = {"ltd", "limited", "co", "company", "india", "the", "assurance", "pvt",
              "private", "of", "plc", "and", "ins", "insurance",
              # "allied" is noise because Galaxy dropped it from its name while the
              # Annual Report still prints it; Star Health carries it in both forms.
              "allied", "branch", "indian"}


def _reg_key(name):
    """Identity tokens of an insurer name, order-independent.

    Matching on tokens rather than strings absorbs the differences that separate the
    handbook from the Annual Report without needing an alias for each: '&' vs 'and',
    'Co. Ltd.' vs 'Ltd', punctuation, and word order.
    """
    n = unicodedata.normalize("NFKD", str(name or "")).encode("ascii", "ignore").decode()
    n = n.replace("&", " and ")
    # Single characters are dropped: they carry no identity and are usually an
    # artefact of an apostrophe. "Lloyd's of India" tokenised to {lloyd, s}, which
    # cleared the two-token floor and then uniquely subset-matched Volante Global
    # Services - Lloyd's India, silently declaring the Lloyd's platform to be one of
    # its service companies.
    return frozenset(t for t in re.split(r"[^a-z0-9]+", n.lower())
                     if len(t) > 1 and t not in _REG_NOISE)


def _apply_insurer_status(df):
    """Add insurer_status: active | not_writing | deregistered.

    Registration is a legal fact that the filings cannot express. The handbook keeps
    publishing an insurer for years after it stops writing — Sahara India Life and
    Reliance Health both still show 2024-25 — so neither the data nor the last filed
    year can distinguish a live insurer from a closed one.

      active        registered per Annexure 1 and writing business
      not_writing   registered, so it counts toward segment totals, but barred or
                    transferred out (Sahara, Reliance Health, ITI Re)
      deregistered  absent from Annexure 1; real history, but not a current insurer
                    (Exide Life, HDFC ERGO Health, Aegon Life, Bharti AXA General)

    Nothing is dropped. Cohorts, market-share denominators and HHI counts should use
    active only; a historical series must still resolve for all three.
    """
    if df is None or getattr(df, "empty", True) or "insurer_id" not in df.columns:
        return df
    try:
        with open(_INSURER_REGISTRY_FILE, "r", encoding="utf-8") as fh:
            reg = json.load(fh)
    except (OSError, ValueError):
        return df

    # Ambiguity guard: if two DIFFERENT registry names reduce to the same token set,
    # that key identifies nothing and must not match anything. Silently keeping the
    # last one is how a generic key can attach to the wrong segment.
    registered, seen_names = {}, {}
    for seg, d in (reg.get("segments") or {}).items():
        for own in ("public", "private"):
            for nm in d.get(own) or []:
                k = _reg_key(nm)
                if not k:
                    continue
                if k in seen_names and seen_names[k] != nm:
                    registered.pop(k, None)      # collision -> unusable
                    continue
                seen_names[k] = nm
                registered[k] = seg
    if not registered:
        return df
    not_writing = set(k for k in (reg.get("registered_but_not_writing") or {})
                      if not k.startswith("_"))

    name_col = next((c for c in ("Entity", "insurer", "Insurer") if c in df.columns), None)
    if name_col is None:
        return df
    def _lookup(nm):
        """Registry segment for a name, or None.

        Exact token match first. Then a subset fallback, because the handbook
        abbreviates where the report is formal — "Allianz Global" against "Allianz
        Global Corporate & Speciality SE, India Branch", "GIC Re" against "General
        Insurance Corporation of India". Guarded twice: the shorter side must carry at
        least two identity tokens, and the match must be UNIQUE. Without the
        uniqueness test a one-token name would sweep up every registry entry
        containing it, quietly marking unrelated insurers active.
        """
        k = _reg_key(nm)
        if not k:
            return None
        # Exact match accepts a single token, because several real names reduce to one:
        # "National Insurance Co. Ltd." -> {national}, "ECGC Ltd." -> {ecgc}, "The New
        # India Assurance Co. Ltd." -> {new}. The two-token floor below therefore guards
        # only the fuzzy subset path.
        if k in registered:
            return registered[k]
        if len(k) < 2:
            return None
        cands = {seg for rk, seg in registered.items()
                 if len(rk) >= 2 and (k <= rk or rk <= k)}
        hits = [rk for rk in registered if len(rk) >= 2 and (k <= rk or rk <= k)]
        if len(hits) == 1:
            return registered[hits[0]]
        return None

    status, seg_of = {}, {}
    for iid, nm in df[["insurer_id", name_col]].dropna().drop_duplicates("insurer_id").values:
        seg = _lookup(nm)
        if str(iid) in not_writing:
            status[iid] = "not_writing"
            if seg:
                seg_of[iid] = seg
        elif seg:
            status[iid] = "active"
            seg_of[iid] = seg
        else:
            status[iid] = "deregistered"
    df["insurer_status"] = df["insurer_id"].map(status)
    # Specialised is a registry segment with no insurer_class of its own; surface it
    # so AIC and ECGC stop being counted as ordinary general insurers.
    df["registry_segment"] = df["insurer_id"].map(seg_of)
    return df


def _derive_entity_columns(df):
    """Add entity_type / insurer_class to the in-memory frame when absent."""
    if df is None or getattr(df, "empty", True):
        return df
    if "insurer_id" not in df.columns or "Dimension" not in df.columns:
        return df
    if "entity_type" in df.columns and df["entity_type"].notna().any():
        return df          # already migrated in the DB — trust it

    iid = df["insurer_id"].fillna("").astype(str).str.lower()
    is_agg = (iid.isin(_AGG_EXACT) | iid.str.endswith(_AGG_SUFFIX)
              | iid.apply(lambda x: any(a in x for a in _AGG_CONTAINS)))
    # Aggregate wins over the dimension mapping: every typed dimension carries its
    # own subtotal row (Channel holds a "Total" beside Brokers), and mapping by
    # dimension alone would make that subtotal a channel.
    df["entity_type"] = df["Dimension"].map(_DIM_TO_ENTITY).fillna("insurer")
    df.loc[is_agg, "entity_type"] = "aggregate"

    # Reinsurance entities identified from DATA: only they file the foreign-branch
    # assigned-capital line. That marker covers Indian reinsurers and foreign
    # branches alike, so FRB is split off by the two-name domestic list.
    reins = set()
    if "metric_id" in df.columns:
        m = df["metric_id"].fillna("").astype(str)
        reins = set(df.loc[m.str.contains("assigned_capital_of_branches_of_foreign",
                                          na=False), "insurer_id"].dropna().unique())

    def _cls(i):
        i = str(i or "").lower()
        if i in reins:
            return "Reinsurer" if i in _INDIAN_REINSURERS else "FRB"
        if i in _CLASS_OVERRIDES:
            return _CLASS_OVERRIDES[i]
        if "lloyd" in i:
            return "FRB"
        if "reinsur" in i or re.search(r"(^|_)re(_|$)", i):
            return "Reinsurer"
        if "health" in i:
            return "SAHI"
        if "life" in i:
            return "Life"
        return "General"

    ins = df["entity_type"] == "insurer"
    df["insurer_class"] = None
    df.loc[ins, "insurer_class"] = df.loc[ins, "insurer_id"].map(_cls)
    return df


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

        # Merge verified same-company aliases BEFORE the entity/class layer, so
        # classification and every downstream count see one insurer, not two.
        UNIFIED_DF = _apply_insurer_aliases(UNIFIED_DF)

        # Derive the entity/class layer if the DB was not migrated (production
        # restores from the GCS replica, which has no such columns).
        UNIFIED_DF = _derive_entity_columns(UNIFIED_DF)

        # Registration status, from the Annual Report. After the entity layer, because
        # it needs the aliased names and adds a dimension rather than changing one.
        UNIFIED_DF = _apply_insurer_status(UNIFIED_DF)

        _invalidate_caches()   # data changed → drop memoised filter options / compliance
        _et = UNIFIED_DF["entity_type"].notna().sum() if "entity_type" in UNIFIED_DF.columns else 0
        print(f"[+] Data Engine Loaded: {len(UNIFIED_DF)} rows from SQL. "
              f"({_et} entity-typed)")

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
