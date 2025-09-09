#!/usr/bin/env python3
"""
txt_to_unkgem.py - Convert selected .txt files to unkgem*.csv format
This program processes .txt files from raw_txt directory and creates 
unkgemB.csv, unkgemL.csv, and unkgemU.csv in data/unknown/ directory
"""

import pandas as pd
import numpy as np
import os
import shutil

# Ensure numpy is available as np
np = np

def normalize_spectrum(wavelengths, intensities, light_source):
    """Apply normalization based on light source"""
    if light_source == 'B':
        # Halogen: normalize to 650nm = 50000
        anchor = 650
        target = 50000
        idx = np.argmin(np.abs(wavelengths - anchor))
        if intensities[idx] != 0:
            scale = target / intensities[idx]
            return intensities * scale
        else:
            print(f"⚠️ Warning: Zero intensity at {anchor}nm for {light_source}")
            return intensities
            
    elif light_source == 'L':
        # Laser: normalize to 450nm = 50000
        anchor = 450
        target = 50000
        idx = np.argmin(np.abs(wavelengths - anchor))
        if intensities[idx] != 0:
            scale = target / intensities[idx]
            return intensities * scale
        else:
            print(f"⚠️ Warning: Zero intensity at {anchor}nm for {light_source}")
            return intensities
            
    elif light_source == 'U':
        # UV: normalize to maximum in 810.5-811.5nm window = 15000
        mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
        window = intensities[mask]
        if len(window) > 0:
            max_811 = window.max()
            if max_811 > 0:
                scale = 15000 / max_811
                return intensities * scale
            else:
                print(f"⚠️ Warning: Zero max intensity in 811nm window for {light_source}")
                return intensities
        else:
            print(f"⚠️ Warning: No data in 810.5-811.5nm range for {light_source}")
            return intensities
    else:
        print(f"⚠️ Unknown light source: {light_source}")
        return intensities

def load_and_convert_spectrum(filepath, light_source):
    """Load .txt file and convert to normalized CSV format"""
    try:
        # Read the .txt file (assuming space/tab separated)
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['wavelength', 'intensity'], skiprows=1)
        
        # Apply normalization
        normalized_intensities = normalize_spectrum(df['wavelength'].values, df['intensity'].values, light_source)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'wavelength': df['wavelength'],
            'intensity': normalized_intensities
        })
        
        return output_df
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None

def main():
    """Main conversion workflow"""
    
    # Check directories
    input_dir = 'raw_txt'
    output_dir = 'data/unknown'
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory '{input_dir}' not found!")
        print("Please ensure .txt files are in the raw_txt directory")
        return False
        
    if not os.path.exists(output_dir):
        print(f"📁 Creating output directory '{output_dir}'")
        os.makedirs(output_dir, exist_ok=True)
    
    # Find .txt files in raw_txt
    txt_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.txt')]
    
    if not txt_files:
        print(f"❌ No .txt files found in '{input_dir}'")
        return False
    
    print(f"📂 Found {len(txt_files)} .txt files in '{input_dir}'")
    
    # Process each light source
    converted_files = {}
    
    for light_source in ['B', 'L', 'U']:
        # Find files for this light source
        source_files = [f for f in txt_files if light_source.upper() in f.upper()]
        
        if not source_files:
            print(f"⚠️ No files found for light source {light_source}")
            continue
            
        if len(source_files) > 1:
            print(f"⚠️ Multiple files found for {light_source}: {source_files}")
            print(f"Using first file: {source_files[0]}")
            
        selected_file = source_files[0]
        filepath = os.path.join(input_dir, selected_file)
        
        print(f"🔄 Processing {light_source}: {selected_file}")
        
        # Convert the file
        converted_df = load_and_convert_spectrum(filepath, light_source)
        
        if converted_df is not None:
            # Save as unkgem*.csv
            output_filename = f"unkgem{light_source}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save without header and index to match expected format
            converted_df.to_csv(output_path, header=False, index=False)
            
            converted_files[light_source] = {
                'input': selected_file,
                'output': output_path,
                'rows': len(converted_df)
            }
            
            print(f"✅ Created {output_filename} ({len(converted_df)} data points)")
        else:
            print(f"❌ Failed to convert {selected_file}")
    
    # Summary
    print(f"\n📋 CONVERSION SUMMARY")
    print("=" * 40)
    
    if converted_files:
        for light_source, info in converted_files.items():
            print(f"  {light_source}: {info['input']} → {info['output']} ({info['rows']} points)")
        
        print(f"\n✅ Successfully converted {len(converted_files)} files")
        print(f"📁 Output files are in: {output_dir}")
        print("\n🔬 Ready for numerical analysis!")
        return True
    else:
        print("❌ No files were successfully converted")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 TROUBLESHOOTING:")
        print("1. Ensure .txt files are in the 'raw_txt' directory")
        print("2. File names should contain B, L, or U to identify light source")
        print("3. Files should be space/tab separated with wavelength,intensity columns")