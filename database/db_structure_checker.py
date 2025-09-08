# db_structure_checker.py

import pandas as pd

# Input file
input_file = "gemini_db_long.csv"

# Load the CSV file
try:
    db = pd.read_csv(input_file)
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# Print a preview of the database
print(f"\n✅ Database loaded successfully. Here is a preview:")
print(db.head(20))  # Print the first 20 rows

# Print information about the columns
print(f"\n🔍 Columns in the database:")
print(db.columns)

# Print data types for each column
print(f"\n🔍 Data types:")
print(db.dtypes)

# Print unique values for 'full_name' column (if exists)
if 'full_name' in db.columns:
    print(f"\n🔍 Unique prefixes in 'full_name' (first 20 shown):")
    print(db['full_name'].apply(lambda x: x[:10]).unique()[:20])

print("\n✅ Done displaying the database structure.")
