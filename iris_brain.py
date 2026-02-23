import sqlite3
import pandas as pd # type: ignore
import os
import re
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
DB_NAME = "iris.db"
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

def normalize_tag_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", str(text).lower().replace("_", " "))).strip()

def load_knowledge_base(force_reload=False):
    global ALL_UNIQUE_TAGS, ALL_DOC_NAMES, TAG_INGREDIENTS, KNOWN_VOCAB, NORMALIZED_TAG_LOOKUP
    
    # Pre-load financial data if needed
    if force_reload: load_master_data_engine()

    # 1. Fetch from SQL
    try:
        conn = sqlite3.connect(DB_NAME)
        # Load directly into DataFrame
        df = pd.read_sql_query("SELECT * FROM regulatory_clauses", conn)
        conn.close()
    except Exception as e:
        print(f"[!] Error loading regulatory_clauses from DB: {e}")
        return pd.DataFrame()

    if df.empty: return df

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
        ALL_UNIQUE_TAGS.clear(); ALL_DOC_NAMES.clear(); TAG_INGREDIENTS.clear(); KNOWN_VOCAB.clear(); NORMALIZED_TAG_LOOKUP.clear()
        
        # Load Synonyms
        for k in SYNONYM_MAP.keys(): KNOWN_VOCAB.add(k)
        for v_list in SYNONYM_MAP.values(): 
            for v in v_list: KNOWN_VOCAB.add(v)
        
        # Process Tags from DB
        for idx, row in df.iterrows():
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

    return df

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
        if w not in KNOWN_VOCAB:
            # Avoid aggressive autocorrect on short terms (e.g. sign -> design).
            # Fuzzy correction is only for longer tokens where typo risk is higher.
            if len(w) >= 5:
                matches = difflib.get_close_matches(w, list(KNOWN_VOCAB), n=1, cutoff=0.85)
                if matches: valid_word = matches[0]
        
        stem = get_stem(valid_word)
        if len(stem) >= MIN_KEYWORD_LENGTH: final_tuples.append((valid_word, stem)); soup_ingredients.add(stem)
        
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
def _analyze_risk(selected_entities, dimension):
    """
    Applies thresholds AND trend analysis for Data Explorer alerts.
    Only applicable if Dimension is 'Insurer'.
    """
    if UNIFIED_DF.empty or not selected_entities: return []
    if dimension != "Insurer": return []
    
    alerts = []
    # Filter by specific Entities (Insurers)
    df = UNIFIED_DF[UNIFIED_DF['Entity'].isin(selected_entities)].copy()
    
    # Ensure correct sorting for trend analysis
    df['Sortable_Year'] = df['Financial_Year'].astype(str).str.extract(r'(\d+)').astype(float)
    df = df.sort_values(by=['Entity', 'Metric', 'Sortable_Year', 'Quarter'])

    SAHI_INSURERS = ["Star", "Care", "Aditya Birla", "Niva Bupa", "Manipal", "Galaxy", "Narayana"]

    # 1. GROUP BY INSURER & METRIC TO CHECK TRENDS
    for (entity, metric), group in df.groupby(['Entity', 'Metric']):
        group = cast(pd.DataFrame, group)
        
        # UNIVERSAL THRESHOLD CHECKS (Latest Value)
        # Always check the most recent data point regardless of history length
        latest = group.iloc[-1]
        val = latest['Value']
        
        if "Solvency" in metric and val < 1.5:
            alerts.append({"level": "critical", "msg": f"Regulatory Violation: {entity} - Solvency {val} < 1.5 limit"})
        
        elif "Expense" in metric:
            limit = 35 if any(s in entity for s in SAHI_INSURERS) else 30
            if val > limit:
                alerts.append({"level": "critical", "msg": f"Regulatory Violation: {entity} - EoM {val}% exceeds {limit}% limit"})
        
        elif "Repudiation" in metric:
            if val > 10:
                alerts.append({"level": "warning", "msg": f"High Repudiation: {entity} - {metric} {val}% exceeds 10% limit"})

        # --- SAFETY FIX: TREND ANALYSIS ---
        # We need at least 3 points to compare v0, v1, and v2
        if len(group) < 3:
            continue

        # Get last 3 values specifically
        last_3 = group.tail(3)
        vals = [float(str(v).replace(',','').replace('%','')) for v in last_3['Value'].tolist()]
        years = last_3['Financial_Year'].tolist()
        
        # Double check vals list length before indexing
        if len(vals) < 3:
            continue

        # TREND 1: RISING "BAD" METRICS
        if any(x in metric for x in ["Repudiation", "Claims", "Expense", "Combined"]):
            if vals[0] < vals[1] < vals[2]:
                growth = vals[2] - vals[0]
                if growth >= 3:
                    alerts.append({
                        "level": "warning", 
                        "msg": f"Rising Trend: {entity} - {metric} rose from {vals[0]}% to {vals[2]}% ({years[0]} to {years[2]})."
                    })

        # TREND 2: FALLING "GOOD" METRICS
        if "Solvency" in metric:
            if vals[0] > vals[1] > vals[2]:
                drop = vals[0] - vals[2]
                if drop >= 0.2:
                    alerts.append({
                        "level": "warning", 
                        "msg": f"Deteriorating Solvency: {entity} - Dropped from {vals[0]} to {vals[2]} ({years[0]} to {years[2]})."
                    })

    return alerts

# ==========================================
# 6. DATA INTELLIGENCE ENGINE (DIMENSION AWARE + CSV SUPPORT)
# ==========================================

def _get_doc_category_from_path(file_path, clean_filename):
    normalized_parts = {part.lower() for part in Path(file_path).parts}

    if "health" in normalized_parts:
        return "HEALTH"
    if "life" in normalized_parts:
        return "LIFE"

    tokens = set(re.split(r"[^A-Z0-9]+", clean_filename.upper()))
    health_hints = {"HEALTH", "PRODUCT", "PPHI", "HOSPITAL", "MEDICLAIM"}
    life_hints = {"LIFE", "ULIP", "ANNUITY", "PENSION"}
    if tokens & health_hints:
        return "HEALTH"
    if tokens & life_hints:
        return "LIFE"
    return "OTHER"


def aggregate_regulatory_documents():
    # Rebuilds regulatory_clauses from all Excel files under knowledge_base/.
    # Enables Admin sync to refresh regulatory docs without manual migrate_raw.py runs.
    all_files = glob.glob(os.path.join(KB_FOLDER, "**", "*.xlsx"), recursive=True)
    all_files += glob.glob(os.path.join(KB_FOLDER, "**", "*.xlxs"), recursive=True)

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
            df = pd.read_excel(file_path).fillna("")
            if df.empty:
                continue

            source_doc = re.sub(r"\.(xlsx|xlxs)$", "", filename, flags=re.IGNORECASE).replace("_", " ").upper()
            category = _get_doc_category_from_path(file_path, source_doc)
            doc_type = get_doc_type(source_doc)
            priority = DOC_HIERARCHY.get(doc_type, 99)

            insert_rows = []
            for _, row in df.iterrows():
                clause_text = str(row.get("Clause_Text", "")).strip()
                if not clause_text:
                    continue

                is_header = 0
                if (clause_text.lower().startswith("chapter") or clause_text.lower().startswith("part")) and len(clause_text) < 120:
                    is_header = 1

                insert_rows.append((
                    source_doc.title(),
                    category,
                    doc_type,
                    str(row.get("Clause_ID", "")).strip(),
                    clause_text,
                    str(row.get("Context_Header", "General")),
                    str(row.get("Regulatory_Tags", "")),
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

        print(f"[+] Data Engine Loaded: {len(UNIFIED_DF)} rows from SQL.")
        
    except Exception as e:
        print(f"[!] Error loading SQL data: {e}")
        UNIFIED_DF = pd.DataFrame()

def get_filter_options():
    """
    Returns options structured by Dimension for the new UI.
    """
    if UNIFIED_DF.empty: 
        load_master_data_engine()

    if UNIFIED_DF.empty: 
        return {"dimensions": [], "entities": {}, "metrics": [], "years": [], "quarters": [], "lobs": [], "classes": []}

    entities_by_dim = {}
    unique_dims = sorted(UNIFIED_DF['Dimension'].unique().tolist())

    for dim in unique_dims:
        entities = sorted(UNIFIED_DF[UNIFIED_DF['Dimension'] == dim]['Entity'].unique().tolist())
        entities_by_dim[dim] = entities

    return {
        "dimensions": unique_dims,
        "entities": entities_by_dim,
        "metrics": sorted(UNIFIED_DF['Metric'].dropna().unique().tolist()) if 'Metric' in UNIFIED_DF else [],
        "years": sorted(UNIFIED_DF['Financial_Year'].dropna().unique().tolist()) if 'Financial_Year' in UNIFIED_DF else [],
        "quarters": sorted(UNIFIED_DF['Quarter'].dropna().unique().tolist()) if 'Quarter' in UNIFIED_DF else [],
        "lobs": sorted(UNIFIED_DF['Line_of_Business'].dropna().unique().tolist()) if 'Line_of_Business' in UNIFIED_DF else [],
        "classes": sorted(UNIFIED_DF['Class_of_Business'].dropna().unique().tolist()) if 'Class_of_Business' in UNIFIED_DF else []
    }

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
        risk_alerts = _analyze_risk(filters['entities'], target_dim)

    try:
        # --- KEY UPDATE: Include Source_File in Index to separate duplicates ---
        index_cols = ['Entity', 'Financial_Year']
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
    possible_headers = [target_dim, 'Financial_Year', 'Quarter', 'Class_of_Business']
    
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

# ==========================================
# COMPLIANCE ENGINE UPDATES (TREND AWARE)
# ==========================================

def get_compliance_years():
    if UNIFIED_DF.empty: 
        load_master_data_engine()

    if UNIFIED_DF.empty: return []
    if 'Financial_Year' not in UNIFIED_DF.columns: return []
    return sorted(UNIFIED_DF['Financial_Year'].dropna().unique().tolist(), reverse=True)

def get_compliance_dashboard(target_year=None):
    """
    Analyzes data against thresholds AND historical trends.
    Only runs for Dimension = 'Insurer'.
    """
    load_master_data_engine()
    df = UNIFIED_DF
    
    if df.empty: return []

    # Compliance only makes sense for Insurers
    if 'Dimension' in df.columns:
        df = df[df['Dimension'] == 'Insurer']

    SAHI_INSURERS = ["Star", "Care", "Aditya Birla", "Niva Bupa", "Manipal", "Galaxy", "Narayana"]
    dashboard_data = []
    insurers = df['Entity'].unique()

    for insurer in insurers:
        ins_df = df[df['Entity'] == insurer]
        if ins_df.empty: continue

        # --- A. GET LATEST SNAPSHOT ---
        snapshot_df = ins_df.copy()
        if target_year and target_year != "Latest":
            snapshot_df = ins_df[ins_df['Financial_Year'] == target_year]
        
        if snapshot_df.empty: continue

        try:
            latest_row = snapshot_df.sort_values(by=['Financial_Year', 'Quarter'], ascending=False).iloc[0]
            curr_year = latest_row['Financial_Year']
            curr_qtr = latest_row['Quarter']
        except: continue

        current_data = snapshot_df[
            (snapshot_df['Financial_Year'] == curr_year) & 
            (snapshot_df['Quarter'] == curr_qtr)
        ]

        def get_val(metric_name_part):
            try:
                if isinstance(metric_name_part, list):
                    for name in metric_name_part:
                        row = current_data[current_data['Metric'].str.contains(name, case=False, na=False)]
                        if not row.empty:
                            val_str = str(row.iloc[0]['Value']).replace(',', '').replace('%', '')
                            return float(val_str)
                else:
                    row = current_data[current_data['Metric'].str.contains(metric_name_part, case=False, na=False)]
                    if not row.empty:
                        val_str = str(row.iloc[0]['Value']).replace(',', '').replace('%', '')
                        return float(val_str)
            except: pass
            return None

        # Metrics
        solvency = get_val(["Solvency Margin", "Solvency Ratio"])
        combined_ratio = get_val("Combined Ratio")
        claims_ratio = get_val(["Net Incurred Claims", "Incurred Claims"])
        expense_ratio = get_val(["Expense of Management to GDP Ratio", "Expense of Management", "EoM"])
        repudiation_val = get_val(["Repudiation Ratio", "Claims Repudiated"])
        
        status = "COMPLIANT"
        alerts = []
        is_sahi = any(s.lower() in insurer.lower() for s in SAHI_INSURERS)

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

        # --- C. TREND CHECKS (Safety Fixed) ---
        def check_trend_alert(metric_patterns, alert_type="rising"):
            pattern = '|'.join(metric_patterns)
            metric_rows = ins_df[ins_df['Metric'].str.contains(pattern, case=False, na=False)].copy()
            
            # SAFETY FIX: Ensure at least 3 points exist for trend analysis
            if len(metric_rows) < 3: return

            metric_rows['Sort_Y'] = metric_rows['Financial_Year'].astype(str).str.extract(r'(\d+)').astype(float)
            metric_rows = metric_rows.sort_values(by=['Sort_Y', 'Quarter'])
            
            last_3 = metric_rows.tail(3)
            vals = [float(str(v).replace(',','').replace('%','')) for v in last_3['Value'].tolist()]
            years = last_3['Financial_Year'].tolist()
            
            # Double check list length before indexing
            if len(vals) < 3: return

            if alert_type == "rising":
                if vals[0] < vals[1] < vals[2]:
                    growth = vals[2] - vals[0]
                    if growth >= 3:
                        alerts.append({
                            "level": "warning",
                            "msg": f"Rising Trend: {insurer} - {metric_patterns[0]} rose from {vals[0]}% to {vals[2]}% ({years[0]} to {years[2]})."
                        })
            elif alert_type == "falling":
                if vals[0] > vals[1] > vals[2]:
                    drop = vals[0] - vals[2]
                    if drop >= 0.2: 
                        alerts.append({
                            "level": "warning",
                            "msg": f"Deteriorating Solvency: {insurer} - Dropped from {vals[0]} to {vals[2]} ({years[0]} to {years[2]})."
                        })

        check_trend_alert(["Repudiation Ratio", "Claims Repudiated"], "rising")
        check_trend_alert(["Net Incurred Claims", "Incurred Claims"], "rising")
        check_trend_alert(["Expense of Management"], "rising")
        check_trend_alert(["Solvency"], "falling")

        # --- D. Finalize Data ---
        has_critical = any(a['level'] == 'critical' for a in alerts)
        has_warning = any(a['level'] == 'warning' for a in alerts)

        # Adjust overall status based on critical alerts
        if has_critical:
            status = "VIOLATION"
        elif has_warning and status != "VIOLATION":
            status = "WATCHLIST"

        dashboard_data.append({
            "name": insurer,
            "metrics": {
                "solvency": solvency if solvency else "N/A",
                "expenses": expense_ratio if expense_ratio else "N/A",
                "combined": combined_ratio if combined_ratio else "N/A",
                "claims": claims_ratio if claims_ratio else "N/A"
            },
            "status": status,
            "alerts": alerts,
            "has_critical": has_critical,
            "has_warning": has_warning,
            "last_updated": f"{curr_qtr} {curr_year}"
        })

    priority = {"VIOLATION": 0, "WATCHLIST": 1, "COMPLIANT": 2}
    dashboard_data.sort(key=lambda x: priority.get(x['status'], 3))
    
    return dashboard_data