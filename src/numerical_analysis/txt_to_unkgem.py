# txt_to_unkgem.py - Fixed for your specific directory structure
# Input:  gemini_gemological_analysis/src/numerical_analysis/raw_txt/
# Output: gemini_gemological_analysis/data/unknown/

import os
import pandas as pd
from pathlib import Path

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
            print("⚠️ No 810—812 nm region found in U spectrum.")
            return df
        peak = anchor_region['intensity'].max()
        scale = 15000 / peak if peak != 0 else 1
    else:
        scale = 1
    df['intensity'] *= scale
    return df

def convert_txt_to_csv():
    # Set up paths relative to script location
    script_dir = Path(__file__).parent  # src/numerical_analysis/
    project_root = script_dir.parent.parent  # Go up to gemini_gemological_analysis
    
    # Input and output directories
    input_folder = script_dir / 'raw_txt'  # src/numerical_analysis/raw_txt
    output_folder = project_root / 'data' / 'unknown'  # gemini_gemological_analysis/data/unknown
    
    print("🔍 Looking in folder:", input_folder)
    print("📂 Current working directory:", os.getcwd())
    print("📂 Absolute path to input folder:", input_folder.absolute())
    print("📂 Output folder:", output_folder.absolute())
    print("📂 Expected input path: gemini_gemological_analysis/src/numerical_analysis/raw_txt/")
    print("📂 Expected output path: gemini_gemological_analysis/data/unknown/")
    
    # Create output directory if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    print("✅ Output directory ready")

    try:
        files = list(input_folder.glob("*.txt"))  # Get all .txt files
        file_names = [f.name for f in files]
        print("📄 Found files:", file_names)
    except FileNotFoundError:
        print(f"❌ Folder '{input_folder}' not found.")
        print("📁 Please create: gemini_gemological_analysis/src/numerical_analysis/raw_txt/")
        print("📄 And put your .txt spectrum files there.")
        return

    if not files:
        print("⚠️ No .txt files found in the directory. Please check file paths and extensions.")
        return

    for file_path in files:
        fname = file_path.name
        print(f"🔍 Checking file: {fname}")
        
        print(f"📄 Processing: {fname}")

        # Count lines in the raw file before pandas reads it
        with open(file_path, 'r') as raw_file:
            raw_lines = raw_file.readlines()
        print(f"📖 Raw file {fname} has {len(raw_lines)} lines (before pandas).")

        # Read file with pandas and preserve blank lines if any
        df = pd.read_csv(file_path, sep='\s+', header=None, names=['wavelength', 'intensity'], skip_blank_lines=False)
        print(f"✅ Loaded {fname} with {len(df)} lines (after pandas).")

        label = fname.upper()
        kind = None
        
        if "B" in label:
            kind = "B"
            df = normalize(df, kind)
            output_path = output_folder / "unkgemB.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Converted {fname} -> {output_path} with {len(df)} lines.")
            
            # Save source file reference
            source_path = output_folder / "unkgemB_source.txt"
            with open(source_path, "w") as f:
                f.write(fname.strip())
                
        elif "L" in label:
            kind = "L"
            df = normalize(df, kind)
            output_path = output_folder / "unkgemL.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Converted {fname} -> {output_path} with {len(df)} lines.")
            
            # Save source file reference
            source_path = output_folder / "unkgemL_source.txt"
            with open(source_path, "w") as f:
                f.write(fname.strip())
                
        elif "U" in label:
            kind = "U"
            df = normalize(df, kind)
            output_path = output_folder / "unkgemU.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Converted {fname} -> {output_path} with {len(df)} lines.")
            
            # Save source file reference
            source_path = output_folder / "unkgemU_source.txt"
            with open(source_path, "w") as f:
                f.write(fname.strip())
                
        else:
            print(f"⚠️ Skipped {fname} (no recognized light source B, L, or U in filename).")

    print("\n🎉 Conversion complete!")
    print(f"📁 All output files saved to: {output_folder}")
    
    # List created files
    created_files = list(output_folder.glob("unkgem*.csv"))
    if created_files:
        print("📄 Created files:")
        for f in created_files:
            print(f"  ✅ {f.name}")
    else:
        print("⚠️ No output files were created. Check your input files.")

if __name__ == "__main__":
    try:
        convert_txt_to_csv()
    except Exception as e:
        print("❌ An unexpected error occurred:", e)
        import traceback
        traceback.print_exc()
