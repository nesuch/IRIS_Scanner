import sqlite3
import os

DB_NAME = "iris.db"

def reset_database():
    print(f"[*] Connecting to {DB_NAME}...")
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 1. DROP the old table (Wipe it out)
    print("[-] Dropping old table...")
    c.execute("DROP TABLE IF EXISTS financial_metrics")

    # 2. CREATE the new table (With the new 'source_file' column)
    print("[+] Creating new 'Industrial Strength' table structure...")
    c.execute('''
        CREATE TABLE financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insurer TEXT,
            financial_year TEXT,
            quarter TEXT,
            metric TEXT,
            value REAL,
            line_of_business TEXT,
            class_of_business TEXT,
            dimension TEXT DEFAULT 'Insurer',
            source_file TEXT
        )
    ''')
    
    # 3. Create Logs table if missing
    c.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            timestamp TEXT, 
            endpoint TEXT, 
            method TEXT, 
            ip TEXT, 
            status INTEGER, 
            error_msg TEXT
        )
    ''')
    
    # 4. Create Regulatory Knowledge Base if missing
    c.execute('''
        CREATE TABLE IF NOT EXISTS regulatory_clauses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_doc TEXT,
            doc_category TEXT,
            doc_type TEXT,
            clause_id TEXT,
            clause_text TEXT,
            context_header TEXT,
            regulatory_tags TEXT,
            priority INTEGER,
            is_header BOOLEAN
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ SUCCESS: Database has been reset and upgraded.")
    print("   You may now run 'run_production.py' and click SYNC.")

if __name__ == "__main__":
    reset_db_path = os.path.join(os.getcwd(), DB_NAME)
    if os.path.exists(reset_db_path):
        choice = input(f"Are you sure you want to wipe {DB_NAME}? (y/n): ")
        if choice.lower() == 'y':
            reset_database()
        else:
            print("Cancelled.")
    else:
        reset_database()