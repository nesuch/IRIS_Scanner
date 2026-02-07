import sqlite3
import pandas as pd
import json
import glob
import os
import iris_brain as brain # To use your existing loading logic

DB_NAME = "iris.db"
LOG_FILE = "iris_system_logs.json"

def migrate():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    print("--- STARTING MIGRATION ---")

    # 1. MIGRATE FINANCIAL DATA
    # We leverage your existing brain logic to load the master CSV correctly first
    brain.load_master_data_engine()
    if not brain.UNIFIED_DF.empty:
        print(f"[*] Migrating {len(brain.UNIFIED_DF)} financial records...")
        
        # Prepare data for insertion
        fin_data = brain.UNIFIED_DF.copy()
        # Rename columns to match SQL schema keys if needed, or map them explicitly
        # We'll just insert row by row to be safe and handle the types
        rows_to_insert = []
        for _, row in fin_data.iterrows():
            rows_to_insert.append((
                row.get('Insurer'),
                row.get('Financial_Year'),
                row.get('Quarter'),
                row.get('Metric'),
                row.get('Value'),
                row.get('Line_of_Business', 'General'),
                row.get('Class_of_Business', 'General')
            ))
            
        c.executemany('''
            INSERT INTO financial_metrics (insurer, financial_year, quarter, metric, value, line_of_business, class_of_business)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', rows_to_insert)
        print("[+] Financial Data Migrated.")
    else:
        print("[!] No financial data found in CSV.")

    # 2. MIGRATE KNOWLEDGE BASE (TEXT)
    # We use your existing loader to process the Excels into a DF, then save to SQL
    print("[*] Loading Knowledge Base from Excel files...")
    kb_df = brain.load_knowledge_base(force_reload=False) 
    
    if not kb_df.empty:
        print(f"[*] Migrating {len(kb_df)} regulatory clauses...")
        kb_rows = []
        for _, row in kb_df.iterrows():
            kb_rows.append((
                row.get('Source_Doc'),
                row.get('Doc_Category'),
                row.get('Doc_Type'),
                str(row.get('Clause_ID', '')),
                row.get('Clause_Text'),
                row.get('Context_Header'),
                str(row.get('Regulatory_Tags', '')),
                row.get('Priority', 99),
                1 if row.get('Is_Header') else 0
            ))
            
        c.executemany('''
            INSERT INTO regulatory_clauses (source_doc, doc_category, doc_type, clause_id, clause_text, context_header, regulatory_tags, priority, is_header)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', kb_rows)
        print("[+] Knowledge Base Migrated.")

    # 3. MIGRATE LOGS
    if os.path.exists(LOG_FILE):
        print("[*] Migrating System Logs...")
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
                
            log_rows = [(l['timestamp'], l['endpoint'], l['method'], l['ip'], l['status'], l.get('error')) for l in logs]
            
            c.executemany('''
                INSERT INTO system_logs (timestamp, endpoint, method, ip, status, error_msg)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', log_rows)
            print(f"[+] {len(log_rows)} Logs Migrated.")
        except Exception as e:
            print(f"[!] Log migration failed: {e}")

    conn.commit()
    conn.close()
    print("--- MIGRATION COMPLETE ---")

if __name__ == "__main__":
    migrate()