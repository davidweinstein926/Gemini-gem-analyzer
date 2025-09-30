import pandas as pd

# Fix files IN structural_data (where the importer reads from)
files = [
    'data/structural_data/199BC4_halogen_structural_20250926_094055.csv',
    'data/structural_data/199LC1_laser_structural_20250926_094305.csv',
    'data/structural_data/199UC1_uv_structural_auto_20250926_051400.csv',
    'data/structural_data/199UP2_uv_structural_auto_20250926_093915.csv'
]

for file_path in files:
    df = pd.read_csv(file_path)
    neg_count = (df['Intensity'] < 0).sum()
    if neg_count > 0:
        print(f"Fixing {file_path.split('/')[-1]}: {neg_count} negative values -> 0")
        df.loc[df['Intensity'] < 0, 'Intensity'] = 0
        df.to_csv(file_path, index=False)
        print(f"  Fixed and saved")
    else:
        print(f"{file_path.split('/')[-1]}: Already clean")

print("\nAll files fixed. Run Option 6 now.")