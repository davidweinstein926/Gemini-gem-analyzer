# append full name to structural db
import os
import pandas as pd

DB_PATH = 'gemini_structural_db.csv'
source_files = ['unkgemB_source.txt', 'unkgemL_source.txt', 'unkgemU_source.txt']

entries = []

def parse_full_name_components(full_name):
    light_source = next((c for c in full_name if c.upper() in 'BLU'), '')
    orientation = next((c for c in full_name if c.upper() in 'CP'), '')
    scan_number = full_name[-1] if full_name[-1].isdigit() else ''
    gem_id_end = full_name.find(light_source)
    gem_number = full_name[:gem_id_end] if gem_id_end != -1 else full_name
    return gem_number, orientation, scan_number, light_source

for source in source_files:
    if os.path.exists(source):
        with open(source) as f:
            full_name = os.path.splitext(f.read().strip())[0]
            gem_id, orientation, scan_no, light_source = parse_full_name_components(full_name)
            entries.append([gem_id, orientation, scan_no, light_source, ''])
            print(f"✅ Parsed {source}: {gem_id}, {orientation}, {scan_no}, {light_source}")
    else:
        print(f"⚠️ Missing source file: {source}")

if entries:
    df_new = pd.DataFrame(entries, columns=['GemID', 'Orientation', 'ScanNo', 'LightSource', 'Expected_Features'])
    df_new['ScanNo'] = df_new['ScanNo'].astype(str)

    if os.path.exists(DB_PATH):
        df_db = pd.read_csv(DB_PATH, dtype={'ScanNo': str})

        keys_to_remove = set((r[0], r[1], r[2], r[3]) for r in entries)
        df_db = df_db[~df_db.apply(lambda r: (r['GemID'], r['Orientation'], r['ScanNo'], r['LightSource']) in keys_to_remove, axis=1)]

        merged = pd.concat([df_db, df_new], ignore_index=True)
    else:
        merged = df_new

    merged.to_csv(DB_PATH, index=False)
    print(f"💾 Appended metadata rows to {DB_PATH}. Total rows now: {len(merged)}")
else:
    print("⚠️ No metadata entries created. Nothing appended.")
