import sqlite3
import pandas as pd # type: ignore
import os
import re
import glob
import difflib
import io  # Required for Excel Export

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

def load_knowledge_base(force_reload=False):
    global ALL_UNIQUE_TAGS, ALL_DOC_NAMES, TAG_INGREDIENTS, KNOWN_VOCAB
    
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
        ALL_UNIQUE_TAGS.clear(); ALL_DOC_NAMES.clear(); TAG_INGREDIENTS.clear(); KNOWN_VOCAB.clear()
        
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
                    
                    ingredients = set()
                    for w in clean_tag.split():
                        if w in STOP_WORDS: continue
                        KNOWN_VOCAB.add(str(w)) 
                        ingredients.add(get_stem(w))
                    if ingredients: TAG_INGREDIENTS[clean_tag] = ingredients

    return df

def get_autocomplete_data():
    vocab = {"CONCEPTS": []}
    concepts = set(ALL_UNIQUE_TAGS)
    concepts.update(SYNONYM_MAP.keys())
    for synonym_list in SYNONYM_MAP.values():
        for word in synonym_list:
            if len(word) > 2: concepts.add(word)
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
    final_tuples = []
    raw_words = re.findall(r'\w+', query)
    soup_ingredients = set()
    
    for w in raw_words:
        if w in STOP_WORDS: continue
        valid_word = w
        if w not in KNOWN_VOCAB:
            matches = difflib.get_close_matches(w, list(KNOWN_VOCAB), n=1, cutoff=0.8)
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
                if doc_tag == target: found = True; break
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
def _analyze_risk(selected_insurers):
    """
    Applies thresholds AND trend analysis for Data Explorer alerts.
    """
    if UNIFIED_DF.empty or not selected_insurers: return []
    
    alerts = []
    df = UNIFIED_DF[UNIFIED_DF['Insurer'].isin(selected_insurers)].copy()
    
    # Ensure correct sorting for trend analysis
    df['Sortable_Year'] = df['Financial_Year'].astype(str).str.extract(r'(\d+)').astype(float)
    df = df.sort_values(by=['Insurer', 'Metric', 'Sortable_Year', 'Quarter'])

    SAHI_INSURERS = ["Star", "Care", "Aditya Birla", "Niva Bupa", "Manipal", "Galaxy", "Narayana"]

    # 1. GROUP BY INSURER & METRIC TO CHECK TRENDS
    for (insurer, metric), group in df.groupby(['Insurer', 'Metric']):
        # We need at least 3 points to establish a "Trend" (e.g. 25 -> 30 -> 34)
        if len(group) < 3: 
            # Fallback to single latest value check
            latest = group.iloc[-1] # type: ignore
            val = latest['Value']
            
            # Simple threshold checks
            if "Solvency" in metric and val < 1.5:
                alerts.append({"level": "critical", "msg": f"{insurer}: Solvency {val} < 1.5."})
            elif "Expense" in metric:
                limit = 35 if any(s in insurer for s in SAHI_INSURERS) else 30
                if val > limit:
                    alerts.append({"level": "critical", "msg": f"{insurer}: EoM {val}% exceeds limit."})
            continue

        # Get last 3 values for trend
        last_3 = group.tail(3) # type: ignore
        vals = [float(str(v).replace(',','')) for v in last_3['Value'].tolist()]
        years = last_3['Financial_Year'].tolist()

        # TREND 1: RISING "BAD" METRICS (Expenses, Claims, Repudiation)
        if any(x in metric for x in ["Repudiation", "Claims", "Expense", "Combined"]):
            # Check for strictly increasing values: v1 < v2 < v3
            if vals[0] < vals[1] < vals[2]:
                growth = vals[2] - vals[0]
                # Only alert if the growth is noticeable (> 3% absolute change)
                if growth >= 3:
                    alerts.append({
                        "level": "warning", 
                        "msg": f"Rising Trend: {metric} rose from {vals[0]}% to {vals[2]}% ({years[0]}-{years[2]})."
                    })

        # TREND 2: FALLING "GOOD" METRICS (Solvency)
        if "Solvency" in metric:
            if vals[0] > vals[1] > vals[2]:
                drop = vals[0] - vals[2]
                if drop >= 0.2: # Significant solvency drop
                    alerts.append({
                        "level": "warning", 
                        "msg": f"Deteriorating Solvency: Dropped from {vals[0]} to {vals[2]} ({years[0]}-{years[2]})."
                    })

    return alerts

# ==========================================
# 6. DATA INTELLIGENCE ENGINE (SQL INTEGRATED)
# ==========================================
def aggregate_submissions():
    """
    Reads Excel files from raw_submissions and INSERTs them into SQL DB.
    """
    all_files = glob.glob(os.path.join(RAW_SUBMISSIONS_FOLDER, "*.xlsx"))
    
    if not all_files: 
        return "[-] No new files found in 'raw_submissions'."

    total_rows: int = 0
    total_files: int = 0

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        for file in all_files:
            if os.path.basename(file).startswith("~$"): continue
            
            try:
                df = pd.read_excel(file)
                # Standardize columns to match SQL
                df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
                
                rows_to_insert = []
                for _, row in df.iterrows():
                    rows_to_insert.append((
                        row.get('insurer'), 
                        str(row.get('financial_year')).replace('.0',''), 
                        row.get('quarter', 'Annual'), 
                        row.get('metric'), 
                        row.get('value'), 
                        row.get('line_of_business', 'General'), 
                        row.get('class_of_business', 'General')
                    ))
                
                c.executemany('''
                    INSERT INTO financial_metrics (insurer, financial_year, quarter, metric, value, line_of_business, class_of_business)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', rows_to_insert)
                
                total_files = total_files + 1 # type: ignore
                total_rows = total_rows + len(rows_to_insert) # type: ignore
                
            except Exception as e:
                print(f"[!] Error processing {file}: {e}")

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
        # Load all metrics into memory (pandas is fast enough for <1M rows)
        UNIFIED_DF = pd.read_sql_query("SELECT * FROM financial_metrics", conn)
        conn.close()

        if UNIFIED_DF.empty:
            print("[!] SQL Database 'financial_metrics' is empty.")
            return

        # Map SQL columns (snake_case) to Application (PascalCase)
        UNIFIED_DF = UNIFIED_DF.rename(columns={
            "insurer": "Insurer",
            "financial_year": "Financial_Year",
            "quarter": "Quarter",
            "metric": "Metric",
            "value": "Value",
            "line_of_business": "Line_of_Business",
            "class_of_business": "Class_of_Business"
        })

        # Ensure correct types
        UNIFIED_DF['Value'] = pd.to_numeric(UNIFIED_DF['Value'], errors='coerce')
        for col in ['Insurer', 'Financial_Year', 'Quarter', 'Metric', 'Class_of_Business']:
            UNIFIED_DF[col] = UNIFIED_DF[col].astype(str).str.strip()

        print(f"[+] Data Engine Loaded: {len(UNIFIED_DF)} rows from SQL.")
        
    except Exception as e:
        print(f"[!] Error loading SQL data: {e}")
        UNIFIED_DF = pd.DataFrame()

def get_filter_options():
    if UNIFIED_DF.empty: return {"insurers": [], "metrics": [], "years": [], "quarters": [], "lobs": [], "classes": []}
    return {
        "insurers": sorted(UNIFIED_DF['Insurer'].dropna().unique().tolist()) if 'Insurer' in UNIFIED_DF else [],
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
    missing_alerts = []
    
    risk_alerts = []
    if filters.get('insurers'): risk_alerts = _analyze_risk(filters['insurers'])

    if filters.get('insurers'): df = df[df['Insurer'].isin(filters['insurers'])]
    if filters.get('years'): df = df[df['Financial_Year'].isin(filters['years'])]
    if filters.get('metrics'): df = df[df['Metric'].isin(filters['metrics'])]
    if filters.get('quarters'): df = df[df['Quarter'].isin(filters['quarters'])]
    if filters.get('lobs'): df = df[df['Line_of_Business'].isin(filters['lobs'])]
    if filters.get('classes'): df = df[df['Class_of_Business'].isin(filters['classes'])]
    
    if df.empty: return pd.DataFrame(), ["No data found."], risk_alerts

    try:
        index_cols = ['Insurer', 'Financial_Year']
        if 'Class_of_Business' in df.columns and (len(df['Class_of_Business'].unique()) > 1 or filters.get('classes')):
             index_cols.append('Class_of_Business')
        if 'Quarter' in df.columns: index_cols.append('Quarter')
        
        pivot_df = df.pivot_table(index=index_cols, columns='Metric', values='Value', aggfunc='sum').reset_index()
        
        new_cols = []
        for c in pivot_df.columns:
            if str(c) in UNIT_MAP: new_cols.append(f"{c} ({UNIT_MAP[str(c)]})")
            else: new_cols.append(str(c))
        pivot_df.columns = new_cols
        
        return pivot_df.fillna('-'), missing_alerts, risk_alerts

    except Exception as e:
        print(f"Pivot Error: {e}")
        return pd.DataFrame(), ["Error processing table."], []

# --- PUBLIC FUNCTIONS ---
def filter_data(filters):
    pivot_df, missing_alerts, risk_alerts = _create_pivoted_view(filters)
    if pivot_df.empty: return {'columns': [], 'rows': [], 'missing': missing_alerts, 'risks': risk_alerts}
    
    base_cols = [c for c in ['Insurer', 'Financial_Year', 'Quarter', 'Class_of_Business'] if c in pivot_df.columns]
    metric_cols = [c for c in pivot_df.columns if c not in base_cols]
    
    return {'columns': base_cols + metric_cols, 'rows': pivot_df.to_dict('records'), 'missing': missing_alerts, 'risks': risk_alerts}

def generate_excel(filters):
    pivot_df, _, _ = _create_pivoted_view(filters)
    if pivot_df.empty: return None
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: pivot_df.to_excel(writer, index=False, sheet_name='IRIS_Report')
    output.seek(0)
    return output

# ==========================================
# COMPLIANCE ENGINE UPDATES (TREND AWARE)
# ==========================================

def get_compliance_years():
    if UNIFIED_DF.empty: return []
    if 'Financial_Year' not in UNIFIED_DF.columns: return []
    return sorted(UNIFIED_DF['Financial_Year'].dropna().unique().tolist(), reverse=True)

def get_compliance_dashboard(target_year=None):
    """
    Analyzes data against thresholds AND historical trends.
    """
    # FIX: Use the Financial Data Engine (UNIFIED_DF), not the Text Loader
    load_master_data_engine()
    df = UNIFIED_DF
    
    if df.empty: return []

    # 1. Base Filter for target year (for the 'Current Values')
    # However, for TRENDS, we need history regardless of target year.
    # So we'll grab history per insurer separately.
    
    SAHI_INSURERS = ["Star", "Care", "Aditya Birla", "Niva Bupa", "Manipal", "Galaxy", "Narayana"]

    dashboard_data = []
    insurers = df['Insurer'].unique()

    for insurer in insurers:
        ins_df = df[df['Insurer'] == insurer]
        if ins_df.empty: continue

        # --- A. GET LATEST SNAPSHOT (for Card Display) ---
        snapshot_df = ins_df.copy()
        if target_year and target_year != "Latest":
            snapshot_df = ins_df[ins_df['Financial_Year'] == target_year]
        
        if snapshot_df.empty: continue # Skip if no data for selected year

        # Find latest available period in this snapshot
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
        retention_ratio = get_val("Retention Ratio")
        expense_ratio = get_val(["Expense of Management to GDP Ratio", "Expense of Management", "EoM"])
        repud_policies = get_val(["Repudiation Ratio", "Claims Repudiated"])
        
        status = "COMPLIANT"
        alerts = []
        is_sahi = any(s.lower() in insurer.lower() for s in SAHI_INSURERS)

        # --- B. THRESHOLD CHECKS ---
        if solvency is not None and solvency < 1.5:
            status = "VIOLATION"
            alerts.append({"level": "critical", "msg": f"Solvency ({solvency}) < 1.5 minimum."})
        elif solvency is not None and 1.5 <= solvency <= 1.55:
            if status != "VIOLATION": status = "WATCHLIST"
            alerts.append({"level": "warning", "msg": f"Solvency ({solvency}) near limit."})

        if expense_ratio is not None:
            limit = 35 if is_sahi else 30
            if expense_ratio > limit:
                status = "VIOLATION"
                alerts.append({"level": "critical", "msg": f"EoM ({expense_ratio}%) exceeds {limit}% limit."})

        if combined_ratio and combined_ratio > 100:
            if status != "VIOLATION": status = "WATCHLIST"
            alerts.append({"level": "warning", "msg": f"Combined Ratio ({combined_ratio}%) > 100%."})

        if claims_ratio and claims_ratio > 90:
            if status != "VIOLATION": status = "WATCHLIST"
            alerts.append({"level": "warning", "msg": f"Claims Ratio ({claims_ratio}%) > 90%."})

        if retention_ratio and retention_ratio > 96:
            status = "VIOLATION"
            alerts.append({"level": "critical", "msg": f"Retention ({retention_ratio}%) violates 4% cession."})

        if repud_policies and repud_policies > 10:
            if status != "VIOLATION": status = "WATCHLIST"
            alerts.append({"level": "warning", "msg": f"Repudiation Ratio ({repud_policies}%) is high."})

        # --- C. TREND CHECKS (New!) ---
        # Helper to check trend on the FULL history (ins_df), not just snapshot
        def check_trend_alert(metric_patterns, alert_type="rising"):
            # Get all rows for this metric for this insurer
            pattern = '|'.join(metric_patterns)
            metric_rows = ins_df[ins_df['Metric'].str.contains(pattern, case=False, na=False)].copy()
            if len(metric_rows) < 3: return

            # Sort by Year/Quarter (Create sortable column)
            metric_rows['Sort_Y'] = metric_rows['Financial_Year'].astype(str).str.extract(r'(\d+)').astype(float)
            metric_rows = metric_rows.sort_values(by=['Sort_Y', 'Quarter'])
            
            # Take last 3 points
            last_3 = metric_rows.tail(3)
            vals = [float(str(v).replace(',','').replace('%','')) for v in last_3['Value'].tolist()]
            years = last_3['Financial_Year'].tolist()

            if alert_type == "rising":
                # Check for consistent increase
                if vals[0] < vals[1] < vals[2]:
                    growth = vals[2] - vals[0]
                    if growth >= 3: # Min 3% growth to warn
                        alerts.append({
                            "level": "warning",
                            "msg": f"Rising Trend: {metric_patterns[0]} rose from {vals[0]}% to {vals[2]}% ({years[0]}-{years[2]})."
                        })
            elif alert_type == "falling":
                if vals[0] > vals[1] > vals[2]:
                    drop = vals[0] - vals[2]
                    if drop >= 0.2: 
                        alerts.append({
                            "level": "warning",
                            "msg": f"Deteriorating Trend: {metric_patterns[0]} dropped from {vals[0]} to {vals[2]} ({years[0]}-{years[2]})."
                        })

        # Apply Trend Checks
        check_trend_alert(["Repudiation Ratio", "Claims Repudiated"], "rising")
        check_trend_alert(["Net Incurred Claims", "Incurred Claims"], "rising")
        check_trend_alert(["Expense of Management"], "rising")
        check_trend_alert(["Solvency"], "falling")

        # --- D. Finalize Data ---
        has_critical = any(a['level'] == 'critical' for a in alerts)
        has_warning = any(a['level'] == 'warning' for a in alerts)

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