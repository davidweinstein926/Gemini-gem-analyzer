#!/usr/bin/env python3
"""
TARGETED L LIGHT DATABASE REBUILDER
Fixes only the L light normalization: Maximum -> 50000
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def correct_l_normalization(wavelengths, intensities):
    """L Light: Maximum intensity -> 50000 (CORRECTED method)"""
    max_intensity = intensities.max()
    if max_intensity != 0:
        normalized = intensities * (50000 / max_intensity)
        return normalized
    else:
        return intensities

def scan_l_files(raw_data_dir="data/raw"):
    """Scan for L light files only"""
    if not os.path.exists(raw_data_dir):
        print(f"ERROR: Directory {raw_data_dir} not found!")
        return []
    
    files = [f for f in os.listdir(raw_data_dir) if f.endswith('.txt')]
    l_files = []
    
    for file in files:
        base = os.path.splitext(file)[0]
        if 'L' in base.upper():
            # Check if it's actually L light (not just contains L)
            for i, char in enumerate(base.upper()):
                if char == 'L':
                    # Verify this is the light source indicator
                    if i < len(base) - 1:  # Not the last character
                        l_files.append(file)
                    break
    
    print(f"Found {len(l_files)} L light files")
    return l_files

def process_l_file(filename, raw_data_dir="data/raw"):
    """Process a single L light file"""
    try:
        file_path = os.path.join(raw_data_dir, filename)
        
        # Read raw data
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
        wavelengths = np.array(df['wavelength'])
        raw_intensities = np.array(df['intensity'])
        
        # Apply CORRECTED L light normalization
        normalized_intensities = correct_l_normalization(wavelengths, raw_intensities)
        
        # Create full_name (matching database format)
        base_name = os.path.splitext(filename)[0]
        full_name = base_name
        
        # Create dataframe with database format
        processed_data = pd.DataFrame({
            'wavelength': wavelengths,
            'intensity': normalized_intensities,
            'full_name': full_name
        })
        
        print(f"   {filename}: {len(processed_data)} points")
        print(f"      Raw range: {raw_intensities.min():.3f} - {raw_intensities.max():.3f}")
        print(f"      Max intensity: {raw_intensities.max():.3f}")
        print(f"      Normalized range: {normalized_intensities.min():.3f} - {normalized_intensities.max():.3f}")
        return processed_data
        
    except Exception as e:
        print(f"   ERROR processing {filename}: {e}")
        return None

def rebuild_l_database():
    """Rebuild only the L light database"""
    print("TARGETED L LIGHT DATABASE REBUILDER")
    print("=" * 40)
    print("Fixes L Light: Maximum -> 50000")
    
    # Backup existing L database
    l_db_file = 'gemini_db_long_L.csv'
    if os.path.exists(l_db_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{l_db_file}_{timestamp}_OLD"
        os.rename(l_db_file, backup_name)
        print(f"Backed up {l_db_file} -> {backup_name}")
    
    # Scan L light files
    l_files = scan_l_files()
    if not l_files:
        print("No L light files found!")
        return
    
    # Process all L light files
    all_l_data = []
    processed_count = 0
    
    print(f"\nProcessing {len(l_files)} L light files:")
    print("-" * 40)
    
    for filename in l_files:
        processed_data = process_l_file(filename)
        if processed_data is not None:
            all_l_data.append(processed_data)
            processed_count += 1
    
    if all_l_data:
        # Combine all L light data
        combined_df = pd.concat(all_l_data, ignore_index=True)
        
        # Sort by full_name and wavelength
        combined_df = combined_df.sort_values(['full_name', 'wavelength'])
        
        # Save new L database
        combined_df.to_csv(l_db_file, index=False)
        
        print(f"\nL DATABASE REBUILT SUCCESSFULLY:")
        print(f"   File: {l_db_file}")
        print(f"   Records: {len(combined_df):,}")
        print(f"   Unique gems: {combined_df['full_name'].nunique()}")
        print(f"   Intensity range: {combined_df['intensity'].min():.3f} - {combined_df['intensity'].max():.3f}")
        
        # Check C0034 specifically
        c0034_entries = combined_df[combined_df['full_name'].str.contains('C0034', na=False)]
        if not c0034_entries.empty:
            print(f"\nC0034 entries in new database:")
            for entry_name in c0034_entries['full_name'].unique():
                entry_data = combined_df[combined_df['full_name'] == entry_name]
                intensity_range = f"{entry_data['intensity'].min():.3f} - {entry_data['intensity'].max():.3f}"
                print(f"   {entry_name}: {len(entry_data)} points, range {intensity_range}")
        else:
            print(f"\nWARNING: No C0034 entries found in rebuilt database")
        
        print(f"\nNormalization method: Maximum intensity -> 50000")
        print(f"Files processed: {processed_count}/{len(l_files)}")
        
    else:
        print("ERROR: No L light files were successfully processed")

def validate_l_database():
    """Validate the rebuilt L database"""
    l_db_file = 'gemini_db_long_L.csv'
    
    if not os.path.exists(l_db_file):
        print(f"ERROR: {l_db_file} not found!")
        return
    
    print(f"\nVALIDATING REBUILT L DATABASE:")
    print("-" * 30)
    
    df = pd.read_csv(l_db_file)
    
    print(f"Records: {len(df):,}")
    print(f"Unique gems: {df['full_name'].nunique()}")
    print(f"Intensity range: {df['intensity'].min():.3f} - {df['intensity'].max():.3f}")
    print(f"Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} nm")
    
    # Check for maximum values at 50000
    max_values = df.groupby('full_name')['intensity'].max()
    gems_at_50k = (max_values == 50000.0).sum()
    
    print(f"Gems with max intensity = 50000: {gems_at_50k}/{len(max_values)}")
    
    # Show C0034 specifically
    c0034_entries = df[df['full_name'].str.contains('C0034', na=False)]
    if not c0034_entries.empty:
        print(f"\nC0034 validation:")
        for entry_name in c0034_entries['full_name'].unique():
            entry_data = df[df['full_name'] == entry_name]
            max_int = entry_data['intensity'].max()
            min_int = entry_data['intensity'].min()
            print(f"   {entry_name}: {min_int:.3f} - {max_int:.3f}")
            
            if max_int == 50000.0:
                print(f"      ✅ Correctly normalized (max = 50000)")
            else:
                print(f"      ❌ Incorrect normalization (max = {max_int})")

def main():
    print("L LIGHT DATABASE CORRECTION TOOL")
    print("Applies Maximum -> 50000 normalization")
    print("=" * 50)
    
    # Show current status
    l_db_file = 'gemini_db_long_L.csv'
    if os.path.exists(l_db_file):
        df = pd.read_csv(l_db_file)
        print(f"Current L database: {len(df):,} records, {df['full_name'].nunique()} gems")
        
        # Check C0034 current state
        c0034_entries = df[df['full_name'].str.contains('C0034', na=False)]
        if not c0034_entries.empty:
            print(f"Current C0034 L entries:")
            for entry_name in c0034_entries['full_name'].unique():
                entry_data = df[df['full_name'] == entry_name]
                max_int = entry_data['intensity'].max()
                print(f"   {entry_name}: max = {max_int:.3f}")
    else:
        print(f"L database not found: {l_db_file}")
    
    confirm = input(f"\nRebuild L database with Maximum->50000 normalization? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        rebuild_l_database()
        validate_l_database()
        
        print(f"\nNEXT STEPS:")
        print(f"1. Test C0034 analysis - L light should now match perfectly")
        print(f"2. All L light spectra now use maximum intensity normalization")
        print(f"3. C0034LC1 should score 0.0 against itself")
    else:
        print("L database rebuild cancelled")

if __name__ == "__main__":
    main()