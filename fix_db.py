import sqlite3

DB_NAME = "iris.db"

def fix_database():
    print(f"Connecting to {DB_NAME}...")
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        
        # 1. Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='financial_metrics'")
        if not c.fetchone():
            print("Table 'financial_metrics' does not exist. Skipping.")
            return

        # 2. Check if column exists
        c.execute("PRAGMA table_info(financial_metrics)")
        columns = [row[1] for row in c.fetchall()]
        
        if 'dimension' in columns:
            print("✅ Column 'dimension' already exists. No action needed.")
        else:
            print("⚠️ Column 'dimension' missing. Adding it now...")
            # Add the column with a default value of 'Insurer'
            c.execute("ALTER TABLE financial_metrics ADD COLUMN dimension TEXT DEFAULT 'Insurer'")
            conn.commit()
            print("✅ Successfully added 'dimension' column.")

        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    fix_database()