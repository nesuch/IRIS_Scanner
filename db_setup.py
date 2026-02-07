import sqlite3
import os

DB_NAME = "iris.db"

def init_db():
    if os.path.exists(DB_NAME):
        print(f"[*] Database {DB_NAME} already exists.")
    else:
        print(f"[*] Creating new database: {DB_NAME}")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 1. TABLE: FINANCIAL DATA (Replaces unified_database.csv)
    # indexed for fast filtering in Data Explorer
    c.execute('''
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insurer TEXT NOT NULL,
            financial_year TEXT,
            quarter TEXT,
            metric TEXT,
            value REAL,
            line_of_business TEXT DEFAULT 'General',
            class_of_business TEXT DEFAULT 'General'
        )
    ''')
    
    # Indexes for speed
    c.execute('CREATE INDEX IF NOT EXISTS idx_insurer ON financial_metrics (insurer)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_metric ON financial_metrics (metric)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_year ON financial_metrics (financial_year)')

    # 2. TABLE: KNOWLEDGE BASE (Replaces Excel Files)
    # Stores the text data for Health/Life/Universal search
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
            is_header BOOLEAN DEFAULT 0
        )
    ''')
    
    # 3. TABLE: SYSTEM LOGS (Replaces iris_system_logs.json)
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

    conn.commit()
    conn.close()
    print("[+] Database initialized successfully.")

if __name__ == "__main__":
    init_db()