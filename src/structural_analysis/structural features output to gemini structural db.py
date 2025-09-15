# structural features output to gemini structural db
import os
import pandas as pd

OUTPUT_B = 'structural_features_output_B.csv'
OUTPUT_L = 'structural_features_output_L.csv'
OUTPUT_U = 'structural_features_output_U.csv'
DB_PATH = 'gemini_structural_db.csv'

source_files = {
    'B': OUTPUT_B,
    'L': OUTPUT_L,
    'U': OUTPUT_U,
}

def extract_full_name_from_filename(filename):
    name = os.path.splitext(filename)[0]
    return name if any(c.isdigit() for c in name) else None

def parse_full_name_components(full_name):
    light_source = next((c for c in full_name if c.upper() in 'BLU'), '')
    orientation = next((c for c in full_name if c.upper() in 'CP'), '')
    scan_number = full_name[-1] if full_name[-1].isdigit() else ''
    gem_id_end = full_name.find(light_source)
    gem_number = full_name[:gem_id_end] if gem_id_end != -1 else full_name
    return gem_number, orientation, scan_number, light_source

def classify_code(label):
    return {
        'Peak': 'P', 'Trough': 'T', 'Crest': 'C',
        'Valley': 'V', 'Shoulder': 'S'
    }.get(label, '?')

all_entries = []

for typ, path in source_files.items():
    if not os.path.exists(path):
        print(f"⚠️ Skipping {typ}: {path} not found.")
        continue

    try:
        df = pd.read_csv(path)

        col_map = {c.lower(): c for c in df.columns}
        required = ['wavelength', 'intensity', 'type']
        if all(k in col_map for k in required):
            df = df.rename(columns={
                col_map['type']: 'Type',
                col_map['wavelength']: 'Wavelength',
                col_map['intensity']: 'Intensity'
            })

        if not {'Wavelength', 'Intensity', 'Type'}.issubset(df.columns):
            raise ValueError(f"Missing required columns: Wavelength, Intensity, Type. Found: {df.columns.tolist()}")

        if 'full_name' not in df.columns:
            print(f"🔍 Attempting to recover full_name from unkgem{typ}_source.txt...")
            source_path = f"unkgem{typ}_source.txt"
            if os.path.exists(source_path):
                with open(source_path) as f:
                    full_name = os.path.splitext(f.read().strip())[0]
                df.insert(0, 'full_name', [full_name] * len(df))
            else:
                df.insert(0, 'full_name', [f'unknown_{i}' for i in range(len(df))])

        full_name = df['full_name'].iloc[0]
        gem_id, orientation, scan_no, light_source = parse_full_name_components(full_name)

        features = []
        skew_value = None
        for _, row in df.iterrows():
            if row['Type'] == 'Skew':
                skew_value = row['Intensity']
                continue
            code = classify_code(row['Type'])
            wl = str(round(row['Wavelength']))  # ✅ Rounded to nearest whole number
            intensity = int(round(row['Intensity']))
            features.append(f"{wl}{code}:{intensity}")

        if skew_value is not None:
            features.append(f"skew:{skew_value:.2f}")

        if features:
            feature_str = ', '.join(features)
            all_entries.append([gem_id, orientation, scan_no, light_source, feature_str])
            print(f"✅ Collapsed {path} into one row with {len(features)} features.")
        else:
            print(f"⚠️ Skipped {path}: no usable features.")

    except Exception as e:
        print(f"❌ Error reading {path}: {e}")

if all_entries:
    df_result = pd.DataFrame(all_entries, columns=['GemID', 'Orientation', 'ScanNo', 'LightSource', 'Expected_Features'])
    df_result['ScanNo'] = df_result['ScanNo'].astype(str)

    if os.path.exists(DB_PATH):
        existing = pd.read_csv(DB_PATH, dtype={'ScanNo': str})
        keys_to_remove = set((row[0], row[1], row[2], row[3]) for row in all_entries)
        existing = existing[~existing.apply(lambda r: (r['GemID'], r['Orientation'], r['ScanNo'], r['LightSource']) in keys_to_remove, axis=1)]
        merged = pd.concat([existing, df_result], ignore_index=True)
    else:
        merged = df_result

    merged.to_csv(DB_PATH, index=False)
    print(f"💾 Structural database saved to {DB_PATH} with {len(merged)} rows.")
else:
    print("⚠️ No structural features loaded. Database not updated.")