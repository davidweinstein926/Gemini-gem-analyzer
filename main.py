#!/usr/bin/env python3
"""
main.py - COMPLETE GEMINI GEMOLOGICAL ANALYSIS SYSTEM WITH DEBUG
Complete version with self-matching validation and debugging tools
Save as: gemini_gemological_analysis/main.py
"""

import os
import sys
import subprocess
import sqlite3
import shutil
import pandas as pd
import numpy as np
import time
import stat
from collections import defaultdict

class IntegratedGeminiSystem:
    def __init__(self):
        self.db_path = "multi_structural_gem_data.db"
        
        # System configuration
        self.spectral_files = ['gemini_db_long_B.csv', 'gemini_db_long_L.csv', 'gemini_db_long_U.csv']
        self.programs = {
            'structural_hub': 'src/structural_analysis/main.py',
            'launcher': 'src/structural_analysis/gemini_launcher.py', 
            'numerical': 'src/numerical_analysis/gemini1.py',
            'converter': 'src/numerical_analysis/txt_to_unkgem.py',
            'fast_analysis': 'fast_gem_analysis.py'
        }
        
        self.data_dirs = ['data/raw', 'data/unknown']
        self.gem_descriptions = {}
        self.load_gem_library()
    
    def load_gem_library(self):
        """Load gem descriptions from gemlib_structural_ready.csv"""
        try:
            gemlib = pd.read_csv('gemlib_structural_ready.csv')
            gemlib.columns = gemlib.columns.str.strip()
            
            if 'Reference' in gemlib.columns:
                gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                
                if all(col in gemlib.columns for col in expected_columns):
                    gemlib['Description'] = gemlib[expected_columns].apply(
                        lambda x: ' | '.join([str(v).strip() for v in x 
                                            if pd.notnull(v) and str(v).strip()]), axis=1)
                    self.gem_descriptions = dict(zip(gemlib['Reference'], gemlib['Description']))
                    print(f"✅ Loaded {len(self.gem_descriptions)} gem descriptions from gemlib")
                else:
                    print(f"⚠️ Missing columns in gemlib: {[c for c in expected_columns if c not in gemlib.columns]}")
            else:
                print("⚠️ 'Reference' column not found in gemlib_structural_ready.csv")
                
        except FileNotFoundError:
            print("⚠️ gemlib_structural_ready.csv not found - descriptions will be generic")
        except Exception as e:
            print(f"⚠️ Error loading gemlib: {e}")
    
    def get_gem_description(self, gem_id):
        """Get descriptive name for gem ID"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        if base_id in self.gem_descriptions:
            return f"{self.gem_descriptions[base_id]} (Gem {base_id})"
        return f"Gem {base_id}"
    
    def check_system_components(self):
        """Check system status with optimized validation"""
        print("GEMINI GEMOLOGICAL ANALYSIS SYSTEM STATUS")
        print("=" * 50)
        
        # Database files
        db_status = {}
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                size_mb = os.path.getsize(db_file) // (1024*1024)
                print(f"✅ {db_file} ({size_mb} MB)")
                db_status[db_file] = True
            else:
                print(f"❌ {db_file} (missing)")
                db_status[db_file] = False
        
        # Program files
        program_status = {}
        for name, path in self.programs.items():
            if os.path.exists(path):
                print(f"✅ {name.replace('_', ' ').title()}")
                program_status[name] = True
            else:
                print(f"❌ {name.replace('_', ' ').title()} (missing)")
                program_status[name] = False
        
        # Gem library
        gemlib_status = "✅ Loaded" if self.gem_descriptions else "❌ Not available"
        print(f"📚 Gem Library (gemlib_structural_ready.csv): {gemlib_status}")
        
        # Data directories
        for data_dir in self.data_dirs:
            if os.path.exists(data_dir):
                files = len([f for f in os.listdir(data_dir) 
                           if f.endswith(('.txt', '.csv'))])
                print(f"✅ {data_dir} ({files} files)")
            else:
                print(f"❌ {data_dir} (missing)")
        
        # System health summary
        db_ok = sum(db_status.values())
        prog_ok = sum(program_status.values())
        print(f"\nSystem Health: {db_ok}/3 databases, {prog_ok}/{len(self.programs)} programs")
        print("=" * 50)
        
        return db_ok >= 3 and prog_ok >= 3
    
    def scan_available_gems(self):
        """Scan and organize available gem files"""
        raw_dir = 'data/raw'
        if not os.path.exists(raw_dir):
            print(f"❌ Directory {raw_dir} not found!")
            return None
        
        files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
        if not files:
            print(f"❌ No .txt files in {raw_dir}")
            return None
        
        # Group files by gem number
        gems = defaultdict(lambda: {'B': [], 'L': [], 'U': []})
        
        for file in files:
            base = os.path.splitext(file)[0].upper()
            
            # Extract light source and gem number
            for light in ['B', 'L', 'U']:
                if light in base:
                    idx = base.index(light)
                    gem_num = base[:idx]
                    gems[gem_num][light].append(file)
                    break
        
        return dict(gems)
    
    def display_gem_options_enhanced(self, gems):
        """Display available gems with enhanced descriptions"""
        print("\n📂 AVAILABLE GEMS FOR ANALYSIS")
        print("=" * 80)
        
        complete_gems, partial_gems = [], []
        
        # Safe sorting that handles mixed string/numeric gem IDs
        try:
            sorted_gem_keys = sorted(gems.keys(), key=lambda x: (len(str(x)), str(x)))
        except Exception:
            sorted_gem_keys = sorted(gems.keys(), key=str)
        
        for gem_num in sorted_gem_keys:
            gem_data = gems[gem_num]
            available_lights = [ls for ls in ['B', 'L', 'U'] if gem_data[ls]]
            description = self.get_gem_description(gem_num)
            
            if len(available_lights) == 3:
                complete_gems.append(gem_num)
                file_counts = [f"{ls}:{len(gem_data[ls])}" for ls in ['B', 'L', 'U']]
                print(f"   ✅ {description}")
                print(f"      Files: {', '.join(file_counts)}")
            else:
                partial_gems.append((gem_num, available_lights))
                print(f"   🟡 {description}")
                print(f"      Only available: {'+'.join(available_lights)}")
        
        return complete_gems, partial_gems
    
    def select_gem_for_analysis(self):
        """Enhanced gem selection workflow"""
        print("\n🎯 GEM SELECTION AND ANALYSIS")
        print("=" * 40)
        
        # Scan available gems
        gems = self.scan_available_gems()
        if not gems:
            return False
        
        # Display options with descriptions
        complete_gems, partial_gems = self.display_gem_options_enhanced(gems)
        
        if not complete_gems:
            print("\n❌ No complete gem sets found!")
            print("Analysis requires gems with B, L, and U files")
            return False
        
        # Get user selection
        print(f"\n🔍 Complete gems available: {', '.join(complete_gems)}")
        
        while True:
            choice = input(f"\nEnter gem number to analyze (or 'back'): ").strip()
            
            if choice.lower() == 'back':
                return False
            
            if choice in complete_gems:
                return self.prepare_gem_for_analysis(choice, gems[choice])
            
            print(f"❌ Invalid choice. Available: {', '.join(complete_gems)}")
    
    def prepare_gem_for_analysis(self, gem_num, gem_files):
        """Prepare gem files with enhanced feedback"""
        gem_desc = self.get_gem_description(gem_num)
        print(f"\n💎 PREPARING {gem_desc.upper()}")
        print("-" * 60)
        
        # Auto-select first file of each type
        selected_files = {}
        for light in ['B', 'L', 'U']:
            if gem_files[light]:
                selected_files[light] = gem_files[light][0]
                print(f"   {light}: {selected_files[light]}")
                
                # Show alternatives if available
                if len(gem_files[light]) > 1:
                    alts = gem_files[light][1:]
                    print(f"       (alternatives: {', '.join(alts)})")
        
        # Convert using proper normalization
        success = self.convert_with_fixed_normalization(selected_files, gem_num)
        
        if success:
            print(f"\n✅ {gem_desc.upper()} READY FOR ANALYSIS")
            return self.offer_analysis_options(gem_num, gem_desc)
        else:
            print(f"\n❌ Failed to prepare {gem_desc}")
            return False
    
    def convert_with_fixed_normalization(self, selected_files, gem_number):
        """Convert files using FIXED normalization scheme with Windows compatibility"""
        try:
            # Setup directories with Windows compatibility
            success, raw_txt_path = self.setup_analysis_directories_safe()
            if not success:
                return False
            
            # Copy files safely
            print("   📁 Copying files to raw_txt...")
            copied_files = self.safe_copy_files('data/raw', raw_txt_path, selected_files)
            
            if len(copied_files) != len(selected_files):
                print(f"   ⚠️ Only copied {len(copied_files)}/{len(selected_files)} files")
            
            if not copied_files:
                print("   ❌ No files copied successfully")
                return False
            
            # Apply FIXED normalization and create unkgem files
            print("   🔧 Applying FIXED normalization...")
            
            for light, filename in copied_files.items():
                input_path = os.path.join(raw_txt_path, filename)
                output_path = f'data/unknown/unkgem{light}.csv'
                
                try:
                    # Load spectrum data
                    df = pd.read_csv(input_path, sep=r'\s+', header=None, 
                                   names=['wavelength', 'intensity'])
                    wavelengths = np.array(df['wavelength'])
                    intensities = np.array(df['intensity'])
                    
                    # Apply correct FIXED normalization
                    normalized = self.apply_fixed_normalization(wavelengths, intensities, light)
                    
                    # Remove 0-100 scaling to match database format
                    final_intensities = normalized
                    
                    # Save with proper format
                    output_df = pd.DataFrame({
                        'wavelength': wavelengths, 
                        'intensity': final_intensities
                    })
                    
                    # Safe file writing with Windows compatibility
                    try:
                        output_df.to_csv(output_path, header=False, index=False)
                        print(f"     ✅ {light}: {len(output_df)} points, "
                              f"range {final_intensities.min():.1f}-{final_intensities.max():.1f}")
                    except PermissionError:
                        # Try alternative output location
                        alt_output = f'unkgem{light}_{gem_number}.csv'
                        output_df.to_csv(alt_output, header=False, index=False)
                        print(f"     ✅ {light}: Saved as {alt_output} (permission workaround)")
                        
                except Exception as e:
                    print(f"     ❌ Error processing {light}: {e}")
                    continue
            
            return True
            
        except Exception as e:
            print(f"     ❌ Conversion error: {e}")
            return False
    
    def apply_fixed_normalization(self, wavelengths, intensities, light_source):
        """Apply correct FIXED normalization per project documentation"""
        try:
            if light_source == 'B':
                # Halogen: 650nm → 50,000
                idx = np.argmin(np.abs(wavelengths - 650))
                if intensities[idx] > 0:
                    return intensities * (50000 / intensities[idx])
                else:
                    print(f"     ⚠️ Zero intensity at 650nm for {light_source}, using max normalization")
                    max_val = intensities.max()
                    return intensities * (50000 / max_val) if max_val > 0 else intensities
                    
            elif light_source == 'L':
                # Laser: Max intensity → 50,000 (NOT 450nm!)
                max_intensity = intensities.max()
                if max_intensity > 0:
                    return intensities * (50000 / max_intensity)
                else:
                    print(f"     ⚠️ Zero max intensity for {light_source}")
                    return intensities
                    
            elif light_source == 'U':
                # UV: 811nm window → 15,000
                mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
                window_values = intensities[mask]
                if len(window_values) > 0 and window_values.max() > 0:
                    return intensities * (15000 / window_values.max())
                else:
                    print(f"     ⚠️ No valid 811nm window for {light_source}, using max normalization")
                    max_val = intensities.max()
                    return intensities * (15000 / max_val) if max_val > 0 else intensities
            
            return intensities
            
        except Exception as e:
            print(f"     ❌ Normalization error for {light_source}: {e}")
            return intensities
    
    def setup_analysis_directories_safe(self):
        """Windows-safe version of directory setup"""
        try:
            # Define paths
            raw_txt_path = 'raw_txt'
            data_unknown_path = 'data/unknown'
            
            # Create data directory if needed
            os.makedirs('data', exist_ok=True)
            
            # Handle raw_txt directory
            if os.path.exists(raw_txt_path):
                # Try to clean existing directory
                try:
                    for file in os.listdir(raw_txt_path):
                        file_path = os.path.join(raw_txt_path, file)
                        if os.path.isfile(file_path):
                            try:
                                os.chmod(file_path, stat.S_IWRITE)
                                os.remove(file_path)
                            except Exception:
                                pass
                except Exception:
                    # If cleaning fails, try to remove and recreate
                    if not self.force_remove_directory(raw_txt_path):
                        # Use timestamp-based alternative name
                        timestamp = str(int(time.time()))
                        raw_txt_path = f'raw_txt_{timestamp}'
            
            # Create directories
            os.makedirs(raw_txt_path, exist_ok=True)
            os.makedirs(data_unknown_path, exist_ok=True)
            
            return True, raw_txt_path
            
        except Exception as e:
            print(f"     ❌ Directory setup error: {e}")
            return False, None
    
    def force_remove_directory(self, path):
        """Force remove directory with Windows permission handling"""
        try:
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    return True
                except PermissionError:
                    # Try permission fix
                    def handle_remove_readonly(func, path, exc):
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    
                    shutil.rmtree(path, onerror=handle_remove_readonly)
                    return True
        except Exception:
            return False
    
    def safe_copy_files(self, source_dir, dest_dir, selected_files):
        """Safely copy files with Windows compatibility"""
        copied_files = {}
        
        for light, filename in selected_files.items():
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            
            try:
                if not os.path.exists(source_path):
                    print(f"     ❌ Source file not found: {source_path}")
                    continue
                
                # Copy with retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        shutil.copy2(source_path, dest_path)
                        print(f"     ✅ {light}: {filename}")
                        copied_files[light] = filename
                        break
                    except PermissionError:
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                        else:
                            print(f"     ❌ Permission denied copying {filename}")
                    except Exception as e:
                        print(f"     ❌ Error copying {filename}: {e}")
                        break
                        
            except Exception as e:
                print(f"     ❌ Unexpected error with {filename}: {e}")
        
        return copied_files
    
    def offer_analysis_options(self, gem_num, gem_desc):
        """Enhanced analysis options"""
        while True:
            print(f"\n🔬 ANALYSIS OPTIONS FOR {gem_desc.upper()}:")
            print("-" * 50)
            print("1. 🧮 Run Numerical Analysis (gemini1.py)")
            print("2. 📊 Advanced Analysis with txt_to_unkgem.py")
            print("3. 🎯 Structural Analysis (Manual/Auto)")
            print("4. 🔙 Back to main menu")
            
            choice = input(f"\nSelect option (1-4): ").strip()
            
            if choice == '1':
                self.run_numerical_analysis()
                break
            elif choice == '2':
                self.run_txt_to_unkgem_analysis()
                break
            elif choice == '3':
                self.launch_structural_analysis()
                break
            elif choice == '4':
                break
            else:
                print("❌ Invalid choice")
        
        return True
    
    def debug_database_structure(self):
        """Debug database structure and entries"""
        print("\n" + "="*60)
        print("🔍 DATABASE STRUCTURE ANALYSIS")
        print("="*60)
        
        for light in ['B', 'L', 'U']:
            db_file = f'gemini_db_long_{light}.csv'
            if os.path.exists(db_file):
                df = pd.read_csv(db_file)
                print(f"\n📊 {light} Database:")
                print(f"   Columns: {list(df.columns)}")
                print(f"   Total entries: {len(df)}")
                
                # Look for C0034 entries specifically
                c0034_entries = df[df['full_name'].str.contains('C0034', na=False)]
                print(f"   C0034 entries found: {len(c0034_entries)}")
                if len(c0034_entries) > 0:
                    print(f"   C0034 full_names: {c0034_entries['full_name'].tolist()}")
                    # Show sample intensity values
                    if 'intensity' in df.columns:
                        sample_intensities = c0034_entries['intensity'].head(5).tolist()
                        print(f"   Sample intensities: {sample_intensities}")
                        print(f"   Intensity range: {c0034_entries['intensity'].min():.3f} to {c0034_entries['intensity'].max():.3f}")
                
                # Show sample of other entries for context
                print(f"   Sample full_names from database:")
                sample_names = df['full_name'].head(10).tolist()
                for name in sample_names:
                    print(f"     {name}")
            else:
                print(f"❌ {db_file} not found")
    
    def debug_normalization_pipeline(self, gem_id='C0034'):
        """Debug normalization pipeline for specific gem"""
        print(f"\n" + "="*60)
        print(f"🔧 NORMALIZATION DEBUG FOR {gem_id}")
        print("="*60)
        
        for light in ['B', 'L', 'U']:
            print(f"\n📈 {light} Light Source Analysis:")
            
            # Check raw file
            raw_file = f'data/raw/{gem_id}{light}C1.txt'
            if os.path.exists(raw_file):
                try:
                    raw_df = pd.read_csv(raw_file, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                    print(f"   Raw file: {raw_file}")
                    print(f"   Raw data points: {len(raw_df)}")
                    print(f"   Raw intensity range: {raw_df['intensity'].min():.3f} to {raw_df['intensity'].max():.3f}")
                    print(f"   Raw wavelength range: {raw_df['wavelength'].min():.1f} to {raw_df['wavelength'].max():.1f} nm")
                    
                    # Apply your normalization step by step
                    wavelengths = raw_df['wavelength'].values
                    intensities = raw_df['intensity'].values
                    
                    print(f"   Applying {light} normalization...")
                    normalized = self.apply_fixed_normalization(wavelengths, intensities, light)
                    print(f"   After normalization: {normalized.min():.3f} to {normalized.max():.3f}")
                    
                    # Final 0-100 scaling
                    final = normalized * (100.0 / normalized.max()) if normalized.max() > 0 else normalized
                    print(f"   After 0-100 scaling: {final.min():.3f} to {final.max():.3f}")
                    
                    # Check what gets saved to unkgem file
                    unkgem_file = f'data/unknown/unkgem{light}.csv'
                    if os.path.exists(unkgem_file):
                        unkgem_df = pd.read_csv(unkgem_file, header=None, names=['wavelength', 'intensity'])
                        print(f"   Unkgem file range: {unkgem_df['intensity'].min():.3f} to {unkgem_df['intensity'].max():.3f}")
                        
                        # Compare first few values
                        print(f"   First 5 processed values: {final[:5]}")
                        print(f"   First 5 unkgem values: {unkgem_df['intensity'].head(5).tolist()}")
                    
                except Exception as e:
                    print(f"   ❌ Error reading raw file: {e}")
            else:
                print(f"   ❌ Raw file not found: {raw_file}")
            
            # Check database entry
            db_file = f'gemini_db_long_{light}.csv'
            if os.path.exists(db_file):
                try:
                    db_df = pd.read_csv(db_file)
                    # Look for exact match and similar matches
                    exact_match = db_df[db_df['full_name'] == f'{gem_id}{light}C1']
                    similar_matches = db_df[db_df['full_name'].str.contains(gem_id, na=False)]
                    
                    print(f"   Database entries for {gem_id}:")
                    if len(exact_match) > 0:
                        print(f"     Exact match ({gem_id}{light}C1): Found")
                        print(f"     DB intensity range: {exact_match['intensity'].min():.3f} to {exact_match['intensity'].max():.3f}")
                        if 'wavelength' in exact_match.columns:
                            print(f"     DB wavelength range: {exact_match['wavelength'].min():.1f} to {exact_match['wavelength'].max():.1f} nm")
                        print(f"     First 5 DB values: {exact_match['intensity'].head(5).tolist()}")
                    else:
                        print(f"     Exact match ({gem_id}{light}C1): NOT FOUND")
                    
                    if len(similar_matches) > 0:
                        print(f"     Similar matches found: {len(similar_matches)}")
                        print(f"     Similar full_names: {similar_matches['full_name'].tolist()}")
                    else:
                        print(f"     No similar matches found for {gem_id}")
                        
                except Exception as e:
                    print(f"   ❌ Error reading database: {e}")
    
    def debug_matching_algorithm(self, gem_id='C0034'):
        """Debug the actual matching computation"""
        print(f"\n" + "="*60)
        print(f"🎯 MATCHING ALGORITHM DEBUG FOR {gem_id}")
        print("="*60)
        
        try:
            # Load the unkgem files (your processed unknown)
            unknown_data = {}
            for light in ['B', 'L', 'U']:
                unkgem_file = f'data/unknown/unkgem{light}.csv'
                if os.path.exists(unkgem_file):
                    df = pd.read_csv(unkgem_file, header=None, names=['wavelength', 'intensity'])
                    unknown_data[light] = df
                    print(f"Unknown {light}: {len(df)} points, range {df['intensity'].min():.3f}-{df['intensity'].max():.3f}")
            
            # Now test matching against database entries for the same gem
            for light in ['B', 'L', 'U']:
                if light not in unknown_data:
                    continue
                    
                print(f"\n🔍 {light} Light Matching Test:")
                db_file = f'gemini_db_long_{light}.csv'
                if os.path.exists(db_file):
                    db_df = pd.read_csv(db_file)
                    
                    # Find entries for this gem
                    gem_entries = db_df[db_df['full_name'].str.contains(gem_id, na=False)]
                    
                    for _, entry in gem_entries.head(3).iterrows():  # Test first 3 entries
                        entry_name = entry['full_name']
                        print(f"   Testing against: {entry_name}")
                        
                        # Create reference dataframe for this entry
                        reference_df = db_df[db_df['full_name'] == entry_name].copy()
                        
                        # Compute score manually
                        unknown_df = unknown_data[light]
                        try:
                            # Merge on wavelength
                            merged = pd.merge(unknown_df, reference_df, on='wavelength', suffixes=('_unknown', '_ref'))
                            if len(merged) > 0:
                                # Calculate MSE
                                mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
                                log_score = np.log1p(mse)
                                print(f"     Points compared: {len(merged)}")
                                print(f"     MSE: {mse:.6f}")
                                print(f"     Log score: {log_score:.6f}")
                                
                                # Show some sample differences
                                diff = merged['intensity_unknown'] - merged['intensity_ref']
                                print(f"     Sample differences: {diff.head(5).tolist()}")
                                print(f"     Max difference: {diff.abs().max():.6f}")
                                print(f"     Mean abs difference: {diff.abs().mean():.6f}")
                            else:
                                print(f"     ❌ No wavelength overlap for merging!")
                        except Exception as e:
                            print(f"     ❌ Error computing score: {e}")
                else:
                    print(f"   ❌ Database file not found: {db_file}")
                    
        except Exception as e:
            print(f"❌ Debug matching error: {e}")
    
    def run_complete_debug_analysis(self, gem_id='C0034'):
        """Run complete debug analysis for a gem"""
        print("\n" + "🔍" + "="*60)
        print("COMPLETE DEBUG ANALYSIS - SELF MATCHING VALIDATION")
        print("="*60 + "🔍")
        
        # Step 1: Database structure
        self.debug_database_structure()
        
        # Step 2: Normalization pipeline
        self.debug_normalization_pipeline(gem_id)
        
        # Step 3: Matching algorithm
        self.debug_matching_algorithm(gem_id)
        
        print(f"\n" + "="*60)
        print("🎯 SUMMARY AND RECOMMENDATIONS")
        print("="*60)
        print("If C0034 scores 0.0 against itself, the system is working correctly.")
        print("If C0034 scores > 0.0 against itself, there's a fundamental issue:")
        print("  1. Database normalization differs from analysis normalization")
        print("  2. Wavelength ranges don't match between files and database")
        print("  3. Database entries don't exist for the expected gem ID")
        print("  4. Floating point precision issues in matching algorithm")
        print("="*60)
    
    def debug_menu(self):
        """Debug menu for self-matching validation"""
        print("\n🔍 DEBUG SELF-MATCHING VALIDATION")
        print("=" * 50)
        print("This analysis tests if gems score 0.0 when compared to themselves")
        print("Critical for validating the normalization and matching pipeline")
        print()
        print("1. Debug C0034 (Client gem)")
        print("2. Debug 140 (Personal collection)")
        print("3. Debug 58 (Personal collection)")
        print("4. Debug custom gem ID")
        print("5. Database structure only")
        print("6. Back to main menu")
        
        choice = input("\nSelect option (1-6): ").strip()
        
        if choice == '1':
            self.run_complete_debug_analysis('C0034')
        elif choice == '2':
            self.run_complete_debug_analysis('140')
        elif choice == '3':
            self.run_complete_debug_analysis('58')
        elif choice == '4':
            custom_id = input("Enter gem ID to debug: ").strip()
            if custom_id:
                self.run_complete_debug_analysis(custom_id)
        elif choice == '5':
            self.debug_database_structure()
        elif choice == '6':
            return
        else:
            print("Invalid choice")
    
    def run_txt_to_unkgem_analysis(self):
        """Run analysis using txt_to_unkgem.py"""
        converter_path = self.programs['converter']
        
        if os.path.exists(converter_path):
            print(f"\n🚀 RUNNING txt_to_unkgem.py ANALYSIS...")
            try:
                result = subprocess.run([sys.executable, converter_path], 
                                      capture_output=True, text=True, timeout=300,
                                      encoding='utf-8', errors='ignore')
                if result.stdout:
                    print("Analysis Results:")
                    print(result.stdout)
                if result.stderr:
                    print("Warnings/Errors:")
                    print(result.stderr)
            except subprocess.TimeoutExpired:
                print("   ⚠️ Analysis timed out")
            except Exception as e:
                print(f"   ❌ Analysis error: {e}")
        else:
            print(f"❌ {converter_path} not found")
    
    def run_numerical_analysis(self):
        """Run numerical analysis with complete results display"""
        print(f"\n🚀 RUNNING NUMERICAL ANALYSIS...")
        
        # Try fast analysis first, then standard
        analysis_programs = [
            (self.programs['fast_analysis'], "optimized fast analysis"),
            (self.programs['numerical'], "standard gemini1.py")
        ]
        
        for prog_path, description in analysis_programs:
            if os.path.exists(prog_path):
                try:
                    print(f"   Using {description}...")
                    result = subprocess.run([sys.executable, prog_path], 
                                          timeout=120, capture_output=True, text=True,
                                          encoding='utf-8', errors='ignore')
                    if result.stdout:
                        print("Results:")
                        print(result.stdout)  # Show ALL results, not truncated
                    return
                except subprocess.TimeoutExpired:
                    print(f"   ⚠️ {description} timed out")
                except Exception as e:
                    print(f"   ❌ {description} error: {e}")
        
        print("   ❌ No working analysis program found")
    
    def launch_structural_analysis(self):
        """Launch structural analysis tools"""
        launcher_path = self.programs['launcher']
        if os.path.exists(launcher_path):
            try:
                subprocess.run([sys.executable, launcher_path], encoding='utf-8', errors='ignore')
            except Exception as e:
                print(f"Error launching structural analysis: {e}")
        else:
            print(f"❌ {launcher_path} not found")
    
    def show_database_statistics(self):
        """Display comprehensive database statistics"""
        print("\n📊 DATABASE STATISTICS")
        print("=" * 50)
        
        # Spectral databases
        total_gems = set()
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                try:
                    df = pd.read_csv(db_file)
                    unique_gems = df['full_name'].nunique() if 'full_name' in df.columns else 'N/A'
                    print(f"✅ {db_file}:")
                    print(f"   Records: {len(df):,}")
                    print(f"   Unique gems: {unique_gems}")
                    
                    # Add to total gem count
                    if 'full_name' in df.columns:
                        gem_ids = df['full_name'].apply(lambda x: str(x).split('B')[0].split('L')[0].split('U')[0])
                        total_gems.update(gem_ids.unique())
                        
                except Exception as e:
                    print(f"❌ {db_file}: Error - {e}")
            else:
                print(f"❌ {db_file}: Missing")
        
        # Gem library statistics
        if self.gem_descriptions:
            print(f"\n📚 Gem Library Information:")
            print(f"   Described gems: {len(self.gem_descriptions)}")
            print(f"   Coverage: {len(total_gems & set(self.gem_descriptions.keys()))}/{len(total_gems)} database gems")
        
        # Structural database
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                count = cursor.execute("SELECT COUNT(*) FROM structural_features").fetchone()[0]
                print(f"\n✅ Structural database: {count:,} records")
                conn.close()
            except Exception as e:
                print(f"❌ Structural database error: {e}")
    
    def launch_program(self, program_key):
        """Generic program launcher"""
        program_path = self.programs.get(program_key)
        if program_path and os.path.exists(program_path):
            try:
                subprocess.run([sys.executable, program_path], encoding='utf-8', errors='ignore')
            except Exception as e:
                print(f"Error launching {program_key}: {e}")
        else:
            print(f"❌ Program not found: {program_key}")
    
    def main_menu(self):
        """Enhanced main menu system with debug option"""
        menu_items = [
            ("🔬 Launch Structural Analysis Hub", lambda: self.launch_program('structural_hub')),
            ("🎯 Launch Structural Analyzers", lambda: self.launch_program('launcher')),
            ("💎 Select and Analyze Gem", self.select_gem_for_analysis),
            ("🧮 Run Numerical Analysis (current)", self.run_numerical_analysis),
            ("🔍 Debug Self-Matching Validation", self.debug_menu),
            ("📈 Show Database Statistics", self.show_database_statistics),
            ("❌ Exit", lambda: None)
        ]
        
        while True:
            print("\n" + "="*80)
            print("🔬 INTEGRATED GEMINI GEMOLOGICAL ANALYSIS SYSTEM")
            if self.gem_descriptions:
                print(f"   📚 Gem Library: {len(self.gem_descriptions)} described gems")
            print("="*80)
            
            # System status check
            self.check_system_components()
            
            # Menu display
            print(f"\n📋 MAIN MENU:")
            print("-" * 40)
            
            for i, (description, _) in enumerate(menu_items, 1):
                print(f"{i:2}. {description}")
            
            # Handle user input
            try:
                choice = input(f"\nChoice (1-{len(menu_items)}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == len(menu_items) - 1:  # Exit
                    print("\n👋 Goodbye!")
                    break
                
                if 0 <= choice_idx < len(menu_items):
                    description, action = menu_items[choice_idx]
                    print(f"\n🚀 {description.upper()}")
                    print("-" * 60)
                    
                    if action:
                        action()
                    
                    input("\n⏎ Press Enter to return to main menu...")
                else:
                    print("❌ Invalid choice")
                    
            except ValueError:
                print("❌ Please enter a number")
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Menu error: {e}")

def main():
    """Main entry point with enhanced error handling"""
    try:
        print("🔬 Starting Integrated Gemini Gemological Analysis System...")
        print("📚 Loading gem library integration...")
        
        system = IntegratedGeminiSystem()
        system.main_menu()
        
    except KeyboardInterrupt:
        print("\n\nSystem interrupted - goodbye!")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
