#!/usr/bin/env python3
"""
REBUILD SPECTRAL DATABASES - Correct Normalization
Recreates gemini_db_long_B.csv, gemini_db_long_L.csv, gemini_db_long_U.csv
with scientifically correct normalization
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class SpectralDatabaseRebuilder:
    def __init__(self, raw_data_dir="data/raw"):
        self.raw_data_dir = raw_data_dir
        self.output_files = {
            'B': 'gemini_db_long_B.csv',
            'L': 'gemini_db_long_L.csv', 
            'U': 'gemini_db_long_U.csv'
        }
    
    def correct_normalize_spectrum(self, wavelengths, intensities, light_source):
        """SCIENTIFICALLY CORRECT NORMALIZATION"""
        
        if light_source == 'B':
            # B Light: 650nm -> 50000 (NO 0-100 scaling for database compatibility)
            anchor_idx = np.argmin(np.abs(wavelengths - 650))
            if intensities[anchor_idx] != 0:
                normalized = intensities * (50000 / intensities[anchor_idx])
                return normalized
            else:
                return intensities
        
        elif light_source == 'L':
            # L Light: MAXIMUM -> 50000 (CORRECTED from 450nm)
            max_intensity = intensities.max()
            if max_intensity != 0:
                normalized = intensities * (50000 / max_intensity)
                return normalized
            else:
                return intensities
        
        elif light_source == 'U':
            # U Light: 811nm window -> 15000 (CORRECTED from divide by max)
            mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
            window = intensities[mask]
            if len(window) > 0 and window.max() > 0:
                normalized = intensities * (15000 / window.max())
                return normalized
            else:
                return intensities
        
        else:
            return intensities
    
    def scan_raw_files(self):
        """Scan data/raw directory for spectral files"""
        if not os.path.exists(self.raw_data_dir):
            print(f"ERROR: Directory {self.raw_data_dir} not found!")
            return None
        
        files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.txt')]
        if not files:
            print(f"ERROR: No .txt files found in {self.raw_data_dir}")
            return None
        
        # Group files by gem and light source
        gems = defaultdict(lambda: {'B': [], 'L': [], 'U': []})
        
        for file in files:
            base = os.path.splitext(file)[0]
            
            # Find light source
            light = None
            for ls in ['B', 'L', 'U']:
                if ls in base.upper():
                    light = ls
                    break
            
            if light:
                # Extract gem number
                for i, char in enumerate(base.upper()):
                    if char == light:
                        gem_num = base[:i]
                        break
                gems[gem_num][light].append(file)
        
        print(f"Found {len(files)} total files")
        print(f"Organized into {len(gems)} gems")
        
        # Show summary
        complete_gems = 0
        for gem_num, gem_files in gems.items():
            available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]
            if len(available) == 3:
                complete_gems += 1
        
        print(f"Complete gems (B+L+U): {complete_gems}")
        print(f"Total light source files: {sum(len(gems[g][ls]) for g in gems for ls in ['B', 'L', 'U'])}")
        
        return dict(gems)
    
    def backup_existing_databases(self):
        """Backup existing database files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for light, filename in self.output_files.items():
            if os.path.exists(filename):
                backup_name = f"{filename}_{timestamp}_OLD"
                os.rename(filename, backup_name)
                print(f"Backed up {filename} -> {backup_name}")
        
        return timestamp
    
    def process_gem_file(self, gem_num, filename, light_source):
        """Process a single gem file and return normalized data"""
        try:
            file_path = os.path.join(self.raw_data_dir, filename)
            
            # Read raw data
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
            wavelengths = np.array(df['wavelength'])
            raw_intensities = np.array(df['intensity'])
            
            # Apply correct normalization
            normalized_intensities = self.correct_normalize_spectrum(wavelengths, raw_intensities, light_source)
            
            # Create full_name (matching database format)
            base_name = os.path.splitext(filename)[0]
            full_name = base_name  # Keep original filename as full_name
            
            # Create dataframe with database format
            processed_data = pd.DataFrame({
                'wavelength': wavelengths,
                'intensity': normalized_intensities,
                'full_name': full_name
            })
            
            print(f"   {filename}: {len(processed_data)} points, range {normalized_intensities.min():.3f}-{normalized_intensities.max():.3f}")
            return processed_data
            
        except Exception as e:
            print(f"   ERROR processing {filename}: {e}")
            return None
    
    def rebuild_databases(self):
        """Main function to rebuild all spectral databases"""
        print("REBUILDING SPECTRAL DATABASES WITH CORRECT NORMALIZATION")
        print("=" * 60)
        
        # Backup existing files
        backup_timestamp = self.backup_existing_databases()
        
        # Scan raw files
        gems = self.scan_raw_files()
        if not gems:
            return
        
        # Initialize databases for each light source
        databases = {
            'B': [],
            'L': [], 
            'U': []
        }
        
        # Process each light source separately
        for light_source in ['B', 'L', 'U']:
            print(f"\nProcessing {light_source} Light Source:")
            print("-" * 30)
            
            file_count = 0
            for gem_num, gem_files in gems.items():
                if gem_files[light_source]:
                    for filename in gem_files[light_source]:
                        processed_data = self.process_gem_file(gem_num, filename, light_source)
                        if processed_data is not None:
                            databases[light_source].append(processed_data)
                            file_count += 1
            
            print(f"   Processed {file_count} {light_source} files")
        
        # Save databases
        for light_source in ['B', 'L', 'U']:
            if databases[light_source]:
                # Combine all data for this light source
                combined_df = pd.concat(databases[light_source], ignore_index=True)
                
                # Sort by full_name and wavelength
                combined_df = combined_df.sort_values(['full_name', 'wavelength'])
                
                # Save to CSV
                output_file = self.output_files[light_source]
                combined_df.to_csv(output_file, index=False)
                
                print(f"\n{light_source} Database: {output_file}")
                print(f"   Records: {len(combined_df):,}")
                print(f"   Unique gems: {combined_df['full_name'].nunique()}")
                print(f"   Intensity range: {combined_df['intensity'].min():.3f} - {combined_df['intensity'].max():.3f}")
            else:
                print(f"\nWARNING: No data processed for {light_source} light source")
        
        print(f"\nREBUILD COMPLETE!")
        print(f"Backup timestamp: {backup_timestamp}")
        print(f"Database stores normalized values for precision:")
        print(f"   B Light: 650nm -> 50000 (no change)")
        print(f"   L Light: Maximum -> 50000 (CORRECTED from 450nm)")  
        print(f"   U Light: 811nm intensity -> 15000 (CORRECTED)")
        print(f"Analysis will apply 0-100 scaling for comparison and visualization")
    
    def validate_new_databases(self):
        """Validate the newly created databases"""
        print("\nVALIDATING NEW DATABASES")
        print("=" * 30)
        
        for light_source in ['B', 'L', 'U']:
            output_file = self.output_files[light_source]
            
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                
                print(f"\n{light_source} Database ({output_file}):")
                print(f"   Records: {len(df):,}")
                print(f"   Unique gems: {df['full_name'].nunique()}")
                print(f"   Intensity range: {df['intensity'].min():.3f} - {df['intensity'].max():.3f}")
                print(f"   Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} nm")
                
                # Show sample entries
                print(f"   Sample entries:")
                for i, (_, row) in enumerate(df.head(3).iterrows()):
                    print(f"      {row['full_name']}: {row['wavelength']:.1f}nm = {row['intensity']:.3f}")
                
                # Check for C0034 specifically
                c0034_entries = df[df['full_name'].str.contains('C0034', na=False)]
                if not c0034_entries.empty:
                    print(f"   C0034 entries: {len(c0034_entries)} records")
                    unique_c0034 = c0034_entries['full_name'].unique()
                    print(f"   C0034 files: {list(unique_c0034)}")
                else:
                    print(f"   WARNING: No C0034 entries found")
            else:
                print(f"\nERROR: {output_file} not found!")
    
    def test_c0034_self_match(self):
        """Test if C0034 will now match itself perfectly"""
        print("\nTESTING C0034 SELF-MATCH")
        print("=" * 30)
        
        # This would require implementing the matching logic
        # For now, just verify the data exists
        for light_source in ['B', 'L', 'U']:
            output_file = self.output_files[light_source]
            
            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                c0034_entries = df[df['full_name'].str.contains('C0034', na=False)]
                
                if not c0034_entries.empty:
                    print(f"   {light_source}: {len(c0034_entries)} C0034 records ready for matching")
                else:
                    print(f"   {light_source}: WARNING - No C0034 entries")
            else:
                print(f"   {light_source}: ERROR - Database not found")

def main():
    """Main execution function"""
    print("SPECTRAL DATABASE REBUILDER")
    print("Fixes L Light normalization: Max -> 50,000")
    print("Fixes U Light normalization: 811 nm -> *15000")
    print("=" * 50)
    
    rebuilder = SpectralDatabaseRebuilder()
    
    # Check if raw data directory exists
    if not os.path.exists(rebuilder.raw_data_dir):
        print(f"ERROR: Raw data directory not found: {rebuilder.raw_data_dir}")
        print("Please ensure data/raw contains the .txt spectral files")
        return
    
    # Show current database status
    print("CURRENT DATABASE STATUS:")
    for light, filename in rebuilder.output_files.items():
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"   {filename}: {len(df):,} records, {df['full_name'].nunique()} gems")
        else:
            print(f"   {filename}: Not found")
    
    # Confirm rebuild
    print(f"\nThis will:")
    print(f"   1. Backup existing databases with timestamp")
    print(f"   2. Rebuild with CORRECT normalization")
    print(f"   3. Fix L Light: Maximum  -> 50,000")
    print(f"   4. Fix U Light: window 811nm peak -> 15,000")
    
    confirm = input(f"\nProceed with database rebuild? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        rebuilder.rebuild_databases()
        rebuilder.validate_new_databases()
        rebuilder.test_c0034_self_match()
        
        print(f"\nNEXT STEPS:")
        print(f"   1. Test C0034 analysis - should now score 0.0 against itself")
        print(f"   2. All gems will now use scientifically correct normalization")
        print(f"   3. L Light properly uses maximum intensity anchor")
        print(f"   4. U Light properly uses 15000 target value")
    else:
        print("Rebuild cancelled")

if __name__ == "__main__":
    main()