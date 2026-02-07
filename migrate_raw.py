import sqlite3
import pandas as pd
import json
import glob
import os
import re

# Configuration
DB_NAME = "iris.db"
KB_FOLDER = "knowledge_base"
# Point directly to where your master CSV sits
MASTER_FILE = os.path.join(KB_FOLDER, "master_data", "unified_database.csv")
LOG_FILE = "iris_system_logs.json"

# Helper for Document Types
def get_doc_type(filename):
    fname = filename.upper()
    if "ACT" in fname: return "ACT"
    if "REGULATION" in fname: return "REGULATION"
    if "MASTER" in fname: return "MASTER"
    if "CIRCULAR" in fname: return "CIRCULAR"
    return "UNKNOWN"

DOC_HIERARCHY = { "ACT": 1, "REGULATION": 2, "MASTER": 3, "CIRCULAR": 4, "GUIDELINE": 5, "UNKNOWN": 99 }

def migrate():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    print("--- STARTING RAW FILE MIGRATION ---")

    # 1. MIGRATE FINANCIAL DATA (Directly from CSV)
    if os.path.exists(MASTER_FILE):
        print(f"[*] Reading Financial Data from {MASTER_FILE}...")
        try:
            df = pd.read_csv(MASTER_FILE)
            
            # Clean Data headers to match DB columns
            # We strip whitespace and standardize
            df.columns = [col.strip().replace(" ", "_") for col in df.columns]
            
            if 'Quarter' in df.columns: df['Quarter'] = df['Quarter'].fillna('Annual')
            if 'Line_of_Business' in df.columns: df['Line_of_Business'] = df['Line_of_Business'].fillna('General')
            if 'Class_of_Business' in df.columns: df['Class_of_Business'] = df['Class_of_Business'].fillna('General')
            
            rows_to_insert = []
            for _, row in df.iterrows():
                # Handle numeric value cleanup
                val = str(row.get('Value', 0)).replace(',', '')
                try: val = float(val)
                except: val = 0.0

                rows_to_insert.append((
                    row.get('Insurer'),
                    str(row.get('Financial_Year', '')).replace('.0', ''),
                    row.get('Quarter'),
                    row.get('Metric'),
                    val,
                    row.get('Line_of_Business', 'General'),
                    row.get('Class_of_Business', 'General')
                ))
            
            c.executemany('''
                INSERT INTO financial_metrics (insurer, financial_year, quarter, metric, value, line_of_business, class_of_business)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', rows_to_insert)
            print(f"[+] Migrated {len(rows_to_insert)} financial records.")
        except Exception as e:
            print(f"[!] Financial migration error: {e}")
    else:
        print(f"[!] File not found: {MASTER_FILE}")

    # 2. MIGRATE KNOWLEDGE BASE (Directly from Excel)
    print("[*] Reading Regulatory Documents (Excel)...")
    all_files = glob.glob(os.path.join(KB_FOLDER, "*.xlsx"))
    kb_count = 0
    
    for file_path in all_files:
        if os.path.basename(file_path).startswith("~$"): continue
        if "raw_submissions" in file_path or "master_data" in file_path: continue
        
        try:
            df = pd.read_excel(file_path).fillna("")
            clean_filename = os.path.basename(file_path).replace(".xlsx", "").replace("_", " ").upper()
            
            category = "OTHER"
            if "HEALTH" in clean_filename or "PRODUCT" in clean_filename: category = "HEALTH"
            elif "LIFE" in clean_filename: category = "LIFE"
            
            doc_type = get_doc_type(clean_filename)
            priority = DOC_HIERARCHY.get(doc_type, 99)

            kb_rows = []
            
            for _, row in df.iterrows():
                text = str(row.get("Clause_Text", "")).strip()
                # Simple Header Detection logic
                is_header = 0
                if (text.lower().startswith("chapter") or text.lower().startswith("part")) and len(text) < 120:
                    is_header = 1
                
                kb_rows.append((
                    clean_filename.title(),
                    category,
                    doc_type,
                    str(row.get("Clause_ID", "")).strip(),
                    text,
                    str(row.get("Context_Header", "General")),
                    str(row.get("Regulatory_Tags", "")),
                    priority,
                    is_header
                ))
            
            c.executemany('''
                INSERT INTO regulatory_clauses (source_doc, doc_category, doc_type, clause_id, clause_text, context_header, regulatory_tags, priority, is_header)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', kb_rows)
            kb_count += len(kb_rows)
            
        except Exception as e:
            print(f"[!] Failed to process {os.path.basename(file_path)}: {e}")
            
    print(f"[+] Migrated {kb_count} regulatory clauses.")

    # 3. MIGRATE LOGS (from JSON)
    if os.path.exists(LOG_FILE):
        print("[*] Migrating System Logs...")
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
            
            log_rows = []
            for l in logs:
                log_rows.append((
                    l.get('timestamp'),
                    l.get('endpoint'),
                    l.get('method'),
                    l.get('ip'),
                    l.get('status'),
                    l.get('error')
                ))
            
            c.executemany('''
                INSERT INTO system_logs (timestamp, endpoint, method, ip, status, error_msg)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', log_rows)
            print(f"[+] Migrated {len(log_rows)} logs.")
        except Exception as e:
            print(f"[!] Log migration failed: {e}")

    conn.commit()
    conn.close()
    print("--- MIGRATION COMPLETE ---")

if __name__ == "__main__":
    migrate()