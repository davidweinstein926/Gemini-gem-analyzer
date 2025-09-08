import sqlite3

# Connect to your existing database
conn = sqlite3.connect(r"c:\users\david\oneDrive\desktop\gemini_matcher\multi_structural_gem_data.db")
cursor = conn.cursor()

print("=== DATABASE SCHEMA ===")
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='structural_features'")
schema = cursor.fetchone()
if schema:
    print(schema[0])

print("\n=== COLUMN INFO ===")
cursor.execute("PRAGMA table_info(structural_features)")
columns = cursor.fetchall()
for col in columns:
    print(f"{col[1]} - {col[2]}")

print("\n=== SAMPLE DATA (5 records) ===")
cursor.execute("SELECT * FROM structural_features LIMIT 5")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()