import sqlite3
import pandas as pd
import json
import os

import iris_brain as brain

# Configuration
DB_NAME = "iris.db"
KB_FOLDER = "knowledge_base"
MASTER_FILE = os.path.join(KB_FOLDER, "master_data", "unified_database.csv")
LOG_FILE = "iris_system_logs.json"


def migrate():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    print("--- STARTING RAW FILE MIGRATION ---")

    # 1. MIGRATE FINANCIAL DATA (Directly from legacy master CSV)
    if os.path.exists(MASTER_FILE):
        print(f"[*] Reading Financial Data from {MASTER_FILE}...")
        try:
            df = pd.read_csv(MASTER_FILE)
            df.columns = [col.strip().replace(" ", "_") for col in df.columns]

            if 'Quarter' in df.columns:
                df['Quarter'] = df['Quarter'].fillna('Annual')
            if 'Line_of_Business' in df.columns:
                df['Line_of_Business'] = df['Line_of_Business'].fillna('General')
            if 'Class_of_Business' in df.columns:
                df['Class_of_Business'] = df['Class_of_Business'].fillna('General')

            rows_to_insert = []
            for _, row in df.iterrows():
                val = str(row.get('Value', 0)).replace(',', '')
                try:
                    val = float(val)
                except Exception:
                    val = 0.0

                rows_to_insert.append((
                    row.get('Insurer'),
                    str(row.get('Financial_Year', '')).replace('.0', ''),
                    row.get('Quarter'),
                    row.get('Metric'),
                    val,
                    row.get('Line_of_Business', 'General'),
                    row.get('Class_of_Business', 'General')
                ))

            c.executemany(
                '''
                INSERT INTO financial_metrics (insurer, financial_year, quarter, metric, value, line_of_business, class_of_business)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                rows_to_insert,
            )
            print(f"[+] Migrated {len(rows_to_insert)} financial records.")
        except Exception as e:
            print(f"[!] Financial migration error: {e}")
    else:
        print(f"[!] File not found: {MASTER_FILE}")

    # 2. MIGRATE KNOWLEDGE BASE (Delegated to shared admin sync engine)
    print("[*] Syncing Regulatory Documents via shared admin pipeline...")
    conn.commit()
    conn.close()
    print(brain.aggregate_regulatory_documents())

    # 3. MIGRATE LOGS (from JSON)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
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

            c.executemany(
                '''
                INSERT INTO system_logs (timestamp, endpoint, method, ip, status, error_msg)
                VALUES (?, ?, ?, ?, ?, ?)
                ''',
                log_rows,
            )
            print(f"[+] Migrated {len(log_rows)} logs.")
        except Exception as e:
            print(f"[!] Log migration failed: {e}")

    conn.commit()
    conn.close()
    print("--- MIGRATION COMPLETE ---")


if __name__ == "__main__":
    migrate()
