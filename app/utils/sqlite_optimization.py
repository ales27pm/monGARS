
import sqlite3

def optimize_sqlite_database(db_path):
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging for concurrent writes
        conn.execute("PRAGMA synchronous=NORMAL;")  # Reduce I/O sync overhead
        conn.execute("PRAGMA cache_size=10000;")  # Increase cache size
        conn.execute("PRAGMA temp_store=MEMORY;")  # Store temp data in memory for faster operations
        conn.close()
        print(f"SQLite database '{db_path}' optimized successfully.")
    except Exception as e:
        print(f"Error optimizing SQLite database '{db_path}': {e}")

if __name__ == "__main__":
    db_path = "memories.db"
    optimize_sqlite_database(db_path)
