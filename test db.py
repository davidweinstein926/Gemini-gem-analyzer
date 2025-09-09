#!/usr/bin/env python3
"""
Database Corruption Check Script
Checks if your gemini databases have corruption or mislabeling issues
"""

import pandas as pd
import numpy as np
import os

def check_database_corruption():
    """Check for systematic database corruption or mislabeling"""
    
    print("DATABASE CORRUPTION CHECK")
    print("=" * 50)
    
    db_files = ['gemini_db_long_B.csv', 'gemini_db_long_L.csv', 'gemini_db_long_U.csv']
    
    for db_file in db_files:
        if os.path.exists(db_file):
            print(f"\n🔍 CHECKING {db_file}")
            print("-" * 30)
            
            try:
                df = pd.read_csv(db_file)
                
                print(f"Total records: {len(df):,}")
                print(f"Unique gems: {df['full_name'].nunique()}")
                print(f"Columns: {list(df.columns)}")
                
                # Check for duplicate entries (same gem, same data)
                if 'full_name' in df.columns:
                    print(f"\nFirst 10 gem entries: {list(df['full_name'].unique()[:10])}")
                    
                    # Check if C0045 and 140 exist
                    c0045_entries = df[df['full_name'].str.contains('C0045', na=False)]
                    gem140_entries = df[df['full_name'].str.contains('^140', na=False, regex=True)]
                    
                    print(f"\nC0045 entries: {list(c0045_entries['full_name'].unique()) if not c0045_entries.empty else 'NONE'}")
                    print(f"140 entries: {list(gem140_entries['full_name'].unique()) if not gem140_entries.empty else 'NONE'}")
                    
                    # If both exist, compare their data
                    if not c0045_entries.empty and not gem140_entries.empty:
                        print(f"\n🔬 COMPARING C0045 vs 140 DATA:")
                        
                        # Get first entries of each
                        c0045_first = df[df['full_name'] == c0045_entries['full_name'].iloc[0]]
                        gem140_first = df[df['full_name'] == gem140_entries['full_name'].iloc[0]]
                        
                        if len(c0045_first) > 0 and len(gem140_first) > 0:
                            c0045_intensities = c0045_first['intensity'].values
                            gem140_intensities = gem140_first['intensity'].values
                            
                            print(f"C0045 intensity range: {c0045_intensities.min():.3f} to {c0045_intensities.max():.3f}")
                            print(f"140 intensity range: {gem140_intensities.min():.3f} to {gem140_intensities.max():.3f}")
                            print(f"C0045 first 5 values: {c0045_intensities[:5]}")
                            print(f"140 first 5 values: {gem140_intensities[:5]}")
                            
                            # Check if they're identical
                            if len(c0045_intensities) == len(gem140_intensities):
                                mse = np.mean((c0045_intensities - gem140_intensities)**2)
                                print(f"MSE between C0045 and 140: {mse:.10f}")
                                
                                if mse < 1e-10:
                                    print("❌ IDENTICAL DATA - This explains the 0.000 match!")
                                    print("   Either database creation error or they're actually the same gem")
                                else:
                                    print("✅ Different data - database seems correct")
                            else:
                                print("⚠️ Different number of points")
                
                # Check for obvious data corruption patterns
                if 'intensity' in df.columns:
                    intensity_stats = df['intensity'].describe()
                    print(f"\nIntensity statistics:")
                    print(f"Min: {intensity_stats['min']:.3f}")
                    print(f"Max: {intensity_stats['max']:.3f}")
                    print(f"Mean: {intensity_stats['mean']:.3f}")
                    print(f"Std: {intensity_stats['std']:.3f}")
                    
                    # Check for suspicious patterns
                    zero_count = (df['intensity'] == 0).sum()
                    negative_count = (df['intensity'] < 0).sum()
                    
                    print(f"Zero values: {zero_count}")
                    print(f"Negative values: {negative_count}")
                    
                    if zero_count > len(df) * 0.1:
                        print("⚠️ HIGH NUMBER OF ZEROS - possible corruption")
                    
                    if negative_count > len(df) * 0.1:
                        print("⚠️ HIGH NUMBER OF NEGATIVES - check normalization")
                
            except Exception as e:
                print(f"❌ Error reading {db_file}: {e}")
        else:
            print(f"❌ {db_file} not found")
    
    print(f"\n🎯 CORRUPTION CHECK SUMMARY:")
    print("If C0045 and 140 have identical data in the database, that explains")
    print("the perfect 0.000 match. This could mean:")
    print("1. Database creation script mislabeled files")
    print("2. C0045 and 140 are actually the same physical gem")
    print("3. File copying error during database creation")
    print("4. Raw data files were accidentally duplicated")

def check_raw_files():
    """Check if raw data files show the same issue"""
    print(f"\n📁 CHECKING RAW DATA FILES")
    print("=" * 30)
    
    raw_dir = 'data/raw'
    if os.path.exists(raw_dir):
        c0045_files = [f for f in os.listdir(raw_dir) if 'C0045' in f]
        gem140_files = [f for f in os.listdir(raw_dir) if '140' in f and 'C0045' not in f]
        
        print(f"C0045 files: {c0045_files}")
        print(f"140 files: {gem140_files}")
        
        if c0045_files and gem140_files:
            # Compare B light files if they exist
            c0045_b = [f for f in c0045_files if 'B' in f]
            gem140_b = [f for f in gem140_files if 'B' in f]
            
            if c0045_b and gem140_b:
                try:
                    print(f"\n🔬 COMPARING RAW FILES:")
                    print(f"C0045 file: {c0045_b[0]}")
                    print(f"140 file: {gem140_b[0]}")
                    
                    c0045_data = pd.read_csv(os.path.join(raw_dir, c0045_b[0]), 
                                           sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                    gem140_data = pd.read_csv(os.path.join(raw_dir, gem140_b[0]), 
                                            sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                    
                    print(f"C0045 raw range: {c0045_data['intensity'].min():.3f} to {c0045_data['intensity'].max():.3f}")
                    print(f"140 raw range: {gem140_data['intensity'].min():.3f} to {gem140_data['intensity'].max():.3f}")
                    print(f"C0045 first 5: {list(c0045_data['intensity'].head())}")
                    print(f"140 first 5: {list(gem140_data['intensity'].head())}")
                    
                    if len(c0045_data) == len(gem140_data):
                        mse = np.mean((c0045_data['intensity'] - gem140_data['intensity'])**2)
                        print(f"Raw MSE: {mse:.6f}")
                        
                        if mse < 1e-6:
                            print("❌ RAW FILES ARE IDENTICAL!")
                            print("   This means the issue is in your source data, not the database")
                        else:
                            print("✅ Raw files are different - database creation issue")
                    
                except Exception as e:
                    print(f"Error comparing raw files: {e}")
        else:
            print("Missing C0045 or 140 raw files for comparison")
    else:
        print(f"Raw directory {raw_dir} not found")

if __name__ == "__main__":
    check_database_corruption()
    check_raw_files()