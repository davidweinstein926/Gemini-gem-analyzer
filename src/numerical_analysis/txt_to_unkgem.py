# txt_to_unkgem.py

import os
import pandas as pd

def normalize(df, kind):
    if kind == 'B':
        anchor = df.iloc[(df['wavelength'] - 650).abs().idxmin()]
        scale = 50000 / anchor['intensity'] if anchor['intensity'] != 0 else 1
    elif kind == 'L':
        anchor = df.iloc[(df['wavelength'] - 450).abs().idxmin()]
        scale = 50000 / anchor['intensity'] if anchor['intensity'] != 0 else 1
    elif kind == 'U':
        anchor_region = df[(df['wavelength'] >= 810) & (df['wavelength'] <= 812)]
        if anchor_region.empty:
            print("⚠️ No 810–812 nm region found in U spectrum.")
            return df
        peak = anchor_region['intensity'].max()
        scale = 15000 / peak if peak != 0 else 1
    else:
        scale = 1
    df['intensity'] *= scale
    return df

def convert_txt_to_csv(folder='raw_txt'):
    print("📁 Looking in folder:", folder)
    print("📂 Current working directory:", os.getcwd())
    folder_path = os.path.abspath(folder)
    print("📂 Absolute path to folder:", folder_path)

    try:
        files = os.listdir(folder)
        print("📄 Found files:", files)
    except FileNotFoundError:
        print("❌ Folder 'raw_txt' not found. Please make sure it exists next to this script.")
        return

    if not files:
        print("⚠️ No files found in the directory. Please check file paths and extensions.")

    for fname in files:
        print(f"🔍 Checking file: {fname}")
        if fname.lower().endswith('.txt'):
            full_path = os.path.join(folder, fname)
            print(f"🔄 Processing: {fname}")

            # Count lines in the raw file before pandas reads it
            with open(full_path, 'r') as raw_file:
                raw_lines = raw_file.readlines()
            print(f"🔍 Raw file {fname} has {len(raw_lines)} lines (before pandas).")

            # Read file with pandas and preserve blank lines if any
            df = pd.read_csv(full_path, sep='\s+', header=None, names=['wavelength', 'intensity'], skip_blank_lines=False)
            print(f"✅ Loaded {fname} with {len(df)} lines (after pandas).")

            label = fname.upper()
            kind = None
            if "B" in label:
                kind = "B"
                df = normalize(df, kind)
                df.to_csv("unkgemB.csv", index=False)
                print(f"✅ Converted {fname} -> unkgemB.csv with {len(df)} lines.")
                with open("unkgemB_source.txt", "w") as f:
                    f.write(fname.strip())
            elif "L" in label:
                kind = "L"
                df = normalize(df, kind)
                df.to_csv("unkgemL.csv", index=False)
                print(f"✅ Converted {fname} -> unkgemL.csv with {len(df)} lines.")
                with open("unkgemL_source.txt", "w") as f:
                    f.write(fname.strip())
            elif "U" in label:
                kind = "U"
                df = normalize(df, kind)
                df.to_csv("unkgemU.csv", index=False)
                print(f"✅ Converted {fname} -> unkgemU.csv with {len(df)} lines.")
                with open("unkgemU_source.txt", "w") as f:
                    f.write(fname.strip())
            else:
                print(f"⚠️ Skipped {fname} (no recognized light source).")

if __name__ == "__main__":
    try:
        convert_txt_to_csv()
    except Exception as e:
        print("❌ An unexpected error occurred:", e)
