#!/usr/bin/env python3
"""
Gemini Database Normalization System - CORRECT SPECIFICATION
Two-step normalization process:
1. Anchor normalization: B(650nm→50K), L(max→50K), U(810-812nm→15K)
2. Scale to 0-100

This ensures all spectra are consistently normalized for accurate matching.
"""

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime

class GeminiNormalizationSystem:
    def __init__(self):
        self.database_files = {
            'B': 'database/reference_spectra/gemini_db_long_B.csv',
            'L': 'database/reference_spectra/gemini_db_long_L.csv', 
            'U': 'database/reference_spectra/gemini_db_long_U.csv'
        }
        
        self.backup_dir = f"database/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def normalize_spectrum_two_step(self, wavelengths, intensities, light_source):
        """Apply the correct two-step normalization process"""
        
        # Step 1: Anchor normalization
        if light_source == 'B':
            # B Light: 650nm → 50000
            anchor_wavelength = 650.0
            target_intensity = 50000.0
            
            # Find closest wavelength to 650nm
            idx = np.argmin(np.abs(wavelengths - anchor_wavelength))
            anchor_intensity = intensities[idx]
            
            if anchor_intensity != 0:
                scale_factor = target_intensity / anchor_intensity
                step1_normalized = intensities * scale_factor
                print(f"      B: 650nm ({anchor_intensity:.1f} → 50000), scale factor: {scale_factor:.3f}")
            else:
                print(f"      B: WARNING - Zero intensity at 650nm, keeping raw")
                step1_normalized = intensities
                
        elif light_source == 'L':
            # L Light: max intensity → 50000
            max_intensity = intensities.max()
            target_intensity = 50000.0
            
            if max_intensity != 0:
                scale_factor = target_intensity / max_intensity
                step1_normalized = intensities * scale_factor
                print(f"      L: max ({max_intensity:.1f} → 50000), scale factor: {scale_factor:.3f}")
            else:
                print(f"      L: WARNING - Zero max intensity, keeping raw")
                step1_normalized = intensities
                
        elif light_source == 'U':
            # U Light: peak max in 810-812nm range → 15000
            target_intensity = 15000.0
            
            # Find wavelengths in 810-812nm range
            mask = (wavelengths >= 810.0) & (wavelengths <= 812.0)
            window_intensities = intensities[mask]
            
            if len(window_intensities) > 0:
                window_max = window_intensities.max()
                if window_max != 0:
                    scale_factor = target_intensity / window_max
                    step1_normalized = intensities * scale_factor
                    print(f"      U: 810-812nm peak ({window_max:.1f} → 15000), scale factor: {scale_factor:.3f}")
                else:
                    print(f"      U: WARNING - Zero intensity in 810-812nm range, keeping raw")
                    step1_normalized = intensities
            else:
                print(f"      U: WARNING - No data in 810-812nm range, keeping raw")
                step1_normalized = intensities
        else:
            print(f"      Unknown light source: {light_source}, keeping raw")
            step1_normalized = intensities
        
        # Step 2: Scale to 0-100
        min_val = step1_normalized.min()
        max_val = step1_normalized.max()
        
        if max_val == min_val:
            # Flat spectrum
            step2_normalized = np.zeros_like(step1_normalized)
            print(f"      Flat spectrum detected, setting to zeros")
        else:
            step2_normalized = (step1_normalized - min_val) * 100.0 / (max_val - min_val)
            print(f"      0-100 scaling: {min_val:.1f}-{max_val:.1f} → 0.0-100.0")
        
        return step2_normalized
    
    def is_spectrum_already_normalized(self, wavelengths, intensities, light_source):
        """Check if spectrum is already properly normalized"""
        
        # Must be in 0-100 range
        min_val = intensities.min()
        max_val = intensities.max()
        
        if not (abs(min_val) < 0.1 and abs(max_val - 100.0) < 0.1):
            return False
        
        # Check if the anchor normalization was applied correctly
        # This is tricky to verify after 0-100 scaling, so we'll be conservative
        # and re-normalize unless it's exactly 0.0-100.0
        return abs(min_val) < 0.001 and abs(max_val - 100.0) < 0.001
    
    def normalize_database_file(self, light_source, filepath):
        """Normalize entire database file using two-step process"""
        print(f"NORMALIZING {light_source} LIGHT DATABASE (TWO-STEP PROCESS)")
        print("=" * 60)
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return 0
        
        # Load database
        df = pd.read_csv(filepath)
        print(f"   Loaded: {len(df):,} rows, {len(df['full_name'].unique())} spectra")
        
        # Process each spectrum individually
        normalized_spectra = []
        processed_count = 0
        skipped_count = 0
        
        spectra = df['full_name'].unique()
        
        for i, spectrum in enumerate(spectra, 1):
            print(f"\n   Processing {i}/{len(spectra)}: {spectrum}")
            
            spectrum_data = df[df['full_name'] == spectrum].copy()
            wavelengths = spectrum_data['wavelength'].values
            intensities = spectrum_data['intensity'].values
            
            # Check if already normalized
            if self.is_spectrum_already_normalized(wavelengths, intensities, light_source):
                print(f"      Already normalized, skipping")
                normalized_spectra.append(spectrum_data)
                skipped_count += 1
                continue
            
            # Apply two-step normalization
            normalized_intensities = self.normalize_spectrum_two_step(
                wavelengths, intensities, light_source)
            
            # Update dataframe
            spectrum_data['intensity'] = normalized_intensities
            normalized_spectra.append(spectrum_data)
            processed_count += 1
        
        # Combine all spectra
        normalized_df = pd.concat(normalized_spectra, ignore_index=True)
        
        # Save normalized database
        normalized_df.to_csv(filepath, index=False)
        
        print(f"\n   SUMMARY:")
        print(f"   Processed: {processed_count} spectra")
        print(f"   Already normalized: {skipped_count} spectra")
        print(f"   Saved to: {filepath}")
        
        return processed_count
    
    def normalize_unknown_spectrum(self, filepath, light_source):
        """Normalize unknown spectrum using two-step process"""
        try:
            # Read spectrum file
            df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
            
            # Apply two-step normalization
            normalized_intensities = self.normalize_spectrum_two_step(
                df['wavelength'].values, df['intensity'].values, light_source)
            
            # Create normalized dataframe
            normalized_df = pd.DataFrame({
                'wavelength': df['wavelength'],
                'intensity': normalized_intensities
            })
            
            print(f"   Two-step normalized {light_source}: {filepath}")
            
            return normalized_df
            
        except Exception as e:
            print(f"   ERROR normalizing {filepath}: {e}")
            return None
    
    def create_normalized_unkgem_files(self, gem_names):
        """Create properly normalized unkgem files"""
        print(f"CREATING NORMALIZED UNKGEM FILES (TWO-STEP PROCESS)")
        print("=" * 60)
        
        # Find source files
        raw_dir = "data/raw"
        if not os.path.exists(raw_dir):
            print(f"ERROR: {raw_dir} directory not found!")
            return False
        
        # Extract gem ID
        first_gem = gem_names[0]
        import re
        match = re.match(r'(\d+)[A-Z]+\d*', first_gem)
        if match:
            gem_id = match.group(1)
        else:
            print(f"ERROR: Could not extract gem ID from {first_gem}")
            return False
        
        # Find gem files
        txt_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
        gem_files = {}
        
        for txt_file in txt_files:
            if txt_file.startswith(gem_id):
                file_upper = txt_file.upper()
                for light in ['B', 'L', 'U']:
                    if light in file_upper:
                        gem_files[light] = txt_file
                        break
        
        if not gem_files:
            print(f"ERROR: No source files found for gem {gem_id}")
            return False
        
        print(f"Found source files: {gem_files}")
        
        # Clean existing unkgem files
        for location in [".", "data/unknown", "data/unknown/numerical"]:
            if os.path.exists(location):
                for light in ['B', 'L', 'U']:
                    unkgem_file = os.path.join(location, f"unkgem{light}.csv")
                    if os.path.exists(unkgem_file):
                        os.remove(unkgem_file)
                        print(f"   Cleaned {unkgem_file}")
        
        # Create normalized unkgem files
        success_count = 0
        
        for light, filename in gem_files.items():
            input_path = os.path.join(raw_dir, filename)
            
            print(f"\nProcessing {light}: {filename}")
            
            # Apply two-step normalization
            normalized_df = self.normalize_unknown_spectrum(input_path, light)
            
            if normalized_df is not None:
                # Save to multiple locations
                for location in [".", "data/unknown", "data/unknown/numerical"]:
                    os.makedirs(location, exist_ok=True)
                    output_path = os.path.join(location, f"unkgem{light}.csv")
                    normalized_df.to_csv(output_path, header=False, index=False)
                    print(f"   Created {output_path}")
                
                success_count += 1
            else:
                print(f"   FAILED to process {filename}")
        
        print(f"\nSUMMARY: Created {success_count}/{len(gem_files)} unkgem files")
        return success_count == len(gem_files)
    
    def create_backup(self):
        """Create backup of original database files"""
        print("CREATING BACKUP OF ORIGINAL DATABASE FILES")
        print("=" * 60)
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
        for light_source, filepath in self.database_files.items():
            if os.path.exists(filepath):
                backup_path = os.path.join(self.backup_dir, f"original_{os.path.basename(filepath)}")
                shutil.copy2(filepath, backup_path)
                print(f"   Backed up {light_source}: {backup_path}")
            else:
                print(f"   WARNING: {filepath} not found")
        
        print(f"\nBackup created in: {self.backup_dir}")
    
    def normalize_all_databases(self):
        """Normalize all database files using correct two-step process"""
        print("\nNORMALIZING ALL DATABASES (TWO-STEP PROCESS)")
        print("=" * 60)
        
        total_processed = 0
        
        for light_source, filepath in self.database_files.items():
            processed = self.normalize_database_file(light_source, filepath)
            total_processed += processed
            print()
        
        print(f"NORMALIZATION COMPLETE")
        print(f"Total spectra processed: {total_processed}")
        
        return total_processed
    
    def verify_normalization(self):
        """Verify all databases are properly normalized"""
        print("VERIFYING NORMALIZATION RESULTS")
        print("=" * 50)
        
        all_verified = True
        
        for light_source, filepath in self.database_files.items():
            if not os.path.exists(filepath):
                continue
                
            df = pd.read_csv(filepath)
            spectra = df['full_name'].unique()
            
            verified_count = 0
            for spectrum in spectra:
                spectrum_data = df[df['full_name'] == spectrum]
                min_val = spectrum_data['intensity'].min()
                max_val = spectrum_data['intensity'].max()
                
                # Check for proper 0-100 scaling
                if abs(min_val) < 0.1 and abs(max_val - 100.0) < 0.1:
                    verified_count += 1
            
            percent_verified = (verified_count / len(spectra)) * 100
            
            print(f"{light_source} Light: {verified_count}/{len(spectra)} spectra verified ({percent_verified:.1f}%)")
            
            if percent_verified < 100:
                all_verified = False
        
        if all_verified:
            print(f"\nSUCCESS: All databases properly normalized")
        else:
            print(f"\nWARNING: Some normalization issues detected")
        
        return all_verified

def main():
    """Main normalization process"""
    print("GEMINI TWO-STEP NORMALIZATION SYSTEM")
    print("=" * 60)
    print("B Light: 650nm → 50000, then 0-100 scale")
    print("L Light: max → 50000, then 0-100 scale") 
    print("U Light: 810-812nm peak → 15000, then 0-100 scale")
    print("=" * 60)
    
    normalizer = GeminiNormalizationSystem()
    
    # Confirm normalization
    confirm = input("\nProceed with two-step database normalization? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Normalization cancelled")
        return
    
    # Create backup
    normalizer.create_backup()
    
    # Normalize all databases
    processed = normalizer.normalize_all_databases()
    
    # Verify results
    success = normalizer.verify_normalization()
    
    if success:
        print(f"\nNORMALIZATION COMPLETE!")
        print(f"All databases normalized using two-step process")
        print(f"Ready for consistent matching!")
    else:
        print(f"\nNORMALIZATION ISSUES DETECTED")
        print(f"Check results and backup files")

if __name__ == "__main__":
    main()