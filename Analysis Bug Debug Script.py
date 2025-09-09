#!/usr/bin/env python3
"""
Analysis Bug Debug Script
Traces exactly where C0045 data becomes 140 data in the analysis workflow
"""

import pandas as pd
import numpy as np
import os

def debug_analysis_workflow():
    """Debug the exact analysis workflow to find where the bug occurs"""
    
    print("ANALYSIS WORKFLOW DEBUG")
    print("=" * 50)
    
    # Step 1: Check the unknown files that were created
    print("\n🔍 STEP 1: CHECK UNKNOWN FILES CREATED BY C0045")
    print("-" * 50)
    
    unknown_files = ['unkgemB.csv', 'unkgemL.csv', 'unkgemU.csv']
    
    for file in unknown_files:
        if os.path.exists(file):
            df = pd.read_csv(file, header=None, names=['wavelength', 'intensity'])
            print(f"\n{file}:")
            print(f"  Records: {len(df)}")
            print(f"  Intensity range: {df['intensity'].min():.3f} to {df['intensity'].max():.3f}")
            print(f"  First 5 intensities: {list(df['intensity'].head())}")
            print(f"  Last 5 intensities: {list(df['intensity'].tail())}")
        else:
            print(f"\n❌ {file} not found")
    
    # Step 2: Check what happens during 0-100 scaling
    print("\n🔍 STEP 2: CHECK 0-100 SCALING OF UNKNOWN")
    print("-" * 50)
    
    if os.path.exists('unkgemB.csv'):
        unknown = pd.read_csv('unkgemB.csv', header=None, names=['wavelength', 'intensity'])
        
        # Apply same 0-100 scaling as analysis
        intensities = unknown['intensity'].values
        min_val, max_val = intensities.min(), intensities.max()
        if max_val != min_val:
            scaled = (intensities - min_val) * 100 / (max_val - min_val)
        else:
            scaled = intensities
            
        print(f"Before scaling: range {min_val:.3f} to {max_val:.3f}")
        print(f"After scaling: range {scaled.min():.3f} to {scaled.max():.3f}")
        print(f"Scaled first 5: {scaled[:5]}")
        print(f"Scaled last 5: {scaled[-5:]}")
    
    # Step 3: Load and check 140BC1 from database
    print("\n🔍 STEP 3: CHECK 140BC1 FROM DATABASE")
    print("-" * 50)
    
    if os.path.exists('gemini_db_long_B.csv'):
        db = pd.read_csv('gemini_db_long_B.csv')
        gem_140 = db[db['full_name'] == '140BC1']
        
        if not gem_140.empty:
            print(f"140BC1 records: {len(gem_140)}")
            print(f"140BC1 intensity range: {gem_140['intensity'].min():.3f} to {gem_140['intensity'].max():.3f}")
            print(f"140BC1 first 5: {list(gem_140['intensity'].head())}")
            
            # Apply 0-100 scaling to 140BC1
            intensities_140 = gem_140['intensity'].values
            min_val, max_val = intensities_140.min(), intensities_140.max()
            if max_val != min_val:
                scaled_140 = (intensities_140 - min_val) * 100 / (max_val - min_val)
            else:
                scaled_140 = intensities_140
                
            print(f"140BC1 after 0-100 scaling: range {scaled_140.min():.3f} to {scaled_140.max():.3f}")
            print(f"140BC1 scaled first 5: {scaled_140[:5]}")
        else:
            print("❌ 140BC1 not found in database")
    
    # Step 4: Simulate the merge operation
    print("\n🔍 STEP 4: SIMULATE MERGE OPERATION")
    print("-" * 50)
    
    if os.path.exists('unkgemB.csv') and os.path.exists('gemini_db_long_B.csv'):
        # Load unknown (C0045)
        unknown = pd.read_csv('unkgemB.csv', header=None, names=['wavelength', 'intensity'])
        
        # Scale unknown
        unknown_scaled = unknown.copy()
        intensities = unknown['intensity'].values
        min_val, max_val = intensities.min(), intensities.max()
        if max_val != min_val:
            unknown_scaled['intensity'] = (intensities - min_val) * 100 / (max_val - min_val)
        
        # Load database and get 140BC1
        db = pd.read_csv('gemini_db_long_B.csv')
        reference = db[db['full_name'] == '140BC1'].copy()
        
        if not reference.empty:
            # Scale 140BC1
            ref_intensities = reference['intensity'].values
            min_val, max_val = ref_intensities.min(), ref_intensities.max()
            if max_val != min_val:
                reference['intensity'] = (ref_intensities - min_val) * 100 / (max_val - min_val)
            
            # Perform merge (same as analysis code)
            merged = pd.merge(unknown_scaled, reference, on='wavelength', suffixes=('_unknown', '_ref'))
            
            print(f"Unknown points: {len(unknown_scaled)}")
            print(f"Reference points: {len(reference)}")
            print(f"Merged points: {len(merged)}")
            
            if len(merged) > 0:
                # Calculate MSE
                mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
                log_score = np.log1p(mse)
                
                print(f"MSE: {mse:.10f}")
                print(f"Log Score: {log_score:.10f}")
                
                # Show first few comparisons
                print(f"\nFirst 5 comparisons:")
                for i in range(min(5, len(merged))):
                    unknown_val = merged.iloc[i]['intensity_unknown']
                    ref_val = merged.iloc[i]['intensity_ref']
                    diff = abs(unknown_val - ref_val)
                    wavelength = merged.iloc[i]['wavelength']
                    print(f"  Wave {wavelength:.1f}: Unknown={unknown_val:.3f}, 140BC1={ref_val:.3f}, Diff={diff:.3f}")
                
                if mse < 1e-10:
                    print("\n❌ BUG FOUND: MSE is near zero when it shouldn't be!")
                    print("   This explains the false 0.000 match")
                    
                    # Check if scaled values are somehow identical
                    if np.allclose(merged['intensity_unknown'], merged['intensity_ref'], atol=1e-10):
                        print("   Scaled values are nearly identical - 0-100 scaling bug!")
                    else:
                        print("   Values are different but MSE calculation is wrong")
                else:
                    print(f"\n✅ MSE is normal - bug must be elsewhere")
            else:
                print("❌ No merged points - wavelength alignment issue!")
        else:
            print("❌ 140BC1 not found for comparison")
    
    # Step 5: Check if there's a file mix-up
    print("\n🔍 STEP 5: CHECK FOR FILE MIX-UP")
    print("-" * 50)
    
    if os.path.exists('unkgemB.csv') and os.path.exists('data/raw/C0045BC1.txt'):
        # Load the unknown file we created
        unknown_csv = pd.read_csv('unkgemB.csv', header=None, names=['wavelength', 'intensity'])
        
        # Load the original raw file and normalize it manually
        raw_data = pd.read_csv('data/raw/C0045BC1.txt', sep=r'\s+', header=None, names=['wavelength', 'intensity'])
        wavelengths = raw_data['wavelength'].values
        intensities = raw_data['intensity'].values
        
        # Apply B normalization (650nm -> 50000)
        anchor_idx = np.argmin(np.abs(wavelengths - 650))
        if intensities[anchor_idx] != 0:
            normalized = intensities * (50000 / intensities[anchor_idx])
        else:
            normalized = intensities
        
        print(f"Raw C0045BC1 range: {intensities.min():.3f} to {intensities.max():.3f}")
        print(f"Manually normalized range: {normalized.min():.3f} to {normalized.max():.3f}")
        print(f"Unknown CSV range: {unknown_csv['intensity'].min():.3f} to {unknown_csv['intensity'].max():.3f}")
        
        # Check if they match
        if len(normalized) == len(unknown_csv):
            mse_check = np.mean((normalized - unknown_csv['intensity'])**2)
            print(f"MSE between manual normalization and CSV: {mse_check:.6f}")
            
            if mse_check < 1e-6:
                print("✅ Unknown CSV matches manual normalization")
            else:
                print("❌ Unknown CSV doesn't match - file creation bug!")
        else:
            print("❌ Length mismatch between manual and CSV")

def check_c0045_database_entries():
    """Check what C0045 entries actually exist in database"""
    print("\n🔍 STEP 6: CHECK C0045 DATABASE ENTRIES")
    print("-" * 50)
    
    db_files = ['gemini_db_long_B.csv', 'gemini_db_long_L.csv', 'gemini_db_long_U.csv']
    
    for db_file in db_files:
        if os.path.exists(db_file):
            df = pd.read_csv(db_file)
            c0045_entries = df[df['full_name'].str.contains('C0045', na=False)]
            
            print(f"\n{db_file}:")
            if not c0045_entries.empty:
                for entry_name in c0045_entries['full_name'].unique():
                    entry_data = df[df['full_name'] == entry_name]
                    print(f"  {entry_name}: {len(entry_data)} points, "
                          f"range {entry_data['intensity'].min():.3f} to {entry_data['intensity'].max():.3f}")
            else:
                print("  No C0045 entries found")

if __name__ == "__main__":
    debug_analysis_workflow()
    check_c0045_database_entries()