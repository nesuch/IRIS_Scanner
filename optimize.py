import sqlite3
try:
    conn = sqlite3.connect("iris.db")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()
    print("✅ Database Optimized (WAL Mode Enabled)")
except Exception as e:
    print(f"Error: {e}")