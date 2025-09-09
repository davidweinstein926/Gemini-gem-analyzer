#!/usr/bin/env python3
"""
COMPLETE FIXED MAIN.PY - GEMINI GEMOLOGICAL ANALYSIS SYSTEM
All indentation errors corrected, full file selection implemented
"""

import os
import sys
import subprocess
import sqlite3
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict

class FixedGeminiAnalysisSystem:
    def __init__(self):
        self.db_path = "multi_structural_gem_data.db"
        
        # System files to check
        self.spectral_files = ['gemini_db_long_B.csv', 'gemini_db_long_L.csv', 'gemini_db_long_U.csv']
        self.program_files = {
            'src/structural_analysis/main.py': 'Structural Analysis Hub',
            'src/structural_analysis/gemini_launcher.py': 'Structural Analyzers Launcher',
            'src/numerical_analysis/gemini1.py': 'Numerical Analysis Engine',
            'fast_gem_analysis.py': 'Fast Analysis Tool'
        }
    
    def correct_normalize_spectrum(self, wavelengths, intensities, light_source):
        """DATABASE-MATCHING NORMALIZATION - matches corrected database exactly"""
    
        if light_source == 'B':
            # B Light: 650nm -> 50000
            anchor_idx = np.argmin(np.abs(wavelengths - 650))
            if intensities[anchor_idx] != 0:
                normalized = intensities * (50000 / intensities[anchor_idx])
                return normalized
            else:
                return intensities
    
        elif light_source == 'L':
            # L Light: Maximum -> 50000 (CORRECTED from 450nm)
            max_intensity = intensities.max()
            if max_intensity != 0:
                normalized = intensities * (50000 / max_intensity)
                return normalized
            else:
                return intensities
        elif light_source == 'U':
            # U Light: 811nm window max -> 15000 (matches database method)
            mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
            window = intensities[mask]
            if len(window) > 0 and window.max() > 0:
                normalized = intensities * (15000 / window.max())
                return normalized
        else:
            return intensities
    
    def apply_0_100_scaling(self, wavelengths, intensities):
        """Apply 0-100 scaling for analysis and visualization"""
        min_val, max_val = intensities.min(), intensities.max()
        if max_val != min_val:
            scaled = (intensities - min_val) * 100 / (max_val - min_val)
            return scaled
        else:
            return intensities
    
    def check_system_status(self):
        """Check overall system status"""
        print("FIXED GEMINI GEMOLOGICAL ANALYSIS SYSTEM STATUS")
        print("=" * 50)
        
        # Check database files
        db_files_ok = 0
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                size = os.path.getsize(db_file) // (1024*1024)  # MB
                print(f"✅ {db_file} ({size} MB)")
                db_files_ok += 1
            else:
                print(f"❌ {db_file} (missing)")
        
        # Check program files
        programs_ok = 0
        for prog_file, description in self.program_files.items():
            if os.path.exists(prog_file):
                print(f"✅ {description}")
                programs_ok += 1
            else:
                print(f"❌ {description} (missing)")
        
        # Check data directories
        data_dirs = ['data/raw', 'data/unknown']
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                files = len([f for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.csv')])
                print(f"✅ {data_dir} ({files} files)")
            else:
                print(f"❌ {data_dir} (missing)")
        
        print(f"\nSystem Status: {db_files_ok}/3 databases, {programs_ok}/{len(self.program_files)} programs")
        print("=" * 50)
        
        return db_files_ok >= 3 and programs_ok >= 2
    
    def scan_available_gems(self):
        """Scan data/raw for available gems"""
        raw_dir = 'data/raw'
        if not os.path.exists(raw_dir):
            print(f"❌ Directory {raw_dir} not found!")
            return None
        
        files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
        if not files:
            print(f"❌ No .txt files in {raw_dir}")
            return None
        
        # Group by gem number
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
        
        return dict(gems)
    
    def show_available_gems(self, gems):
        """Display available gems"""
        print("\n📂 AVAILABLE GEMS FOR ANALYSIS")
        print("=" * 50)
        
        complete_gems = []
        partial_gems = []
        
        for gem_num in sorted(gems.keys()):
            gem_files = gems[gem_num]
            available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]
            
            if len(available) == 3:
                complete_gems.append(gem_num)
                files_summary = []
                for ls in ['B', 'L', 'U']:
                    count = len(gems[gem_num][ls])
                    files_summary.append(f"{ls}:{count}")
                print(f"   ✅ Gem {gem_num} ({', '.join(files_summary)})")
            else:
                partial_gems.append((gem_num, available))
                print(f"   🟡 Gem {gem_num} (only: {'+'.join(available)})")
        
        return complete_gems, partial_gems
    
    def select_and_analyze_gem(self):
        """Complete gem selection and analysis workflow with full file selection"""
        print("\n🎯 GEM SELECTION AND ANALYSIS")
        print("=" * 40)

        # Clear any previous analysis results to prevent caching issues
        for file in ['unkgemB.csv', 'unkgemL.csv', 'unkgemU.csv']:
            if os.path.exists(file):
                os.remove(file)
            if os.path.exists(f'data/unknown/{file}'):
                os.remove(f'data/unknown/{file}')

        # Scan gems
        gems = self.scan_available_gems()
        if not gems:
            return

        # Show ALL available files with numbers
        print("\n📂 AVAILABLE FILES FOR ANALYSIS")
        print("=" * 50)

        all_files = []
        for gem_num in sorted(gems.keys()):
            gem_files = gems[gem_num]
            available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]

            if len(available) == 3:  # Only show complete gems
                print(f"\n✅ Gem {gem_num}:")
                for light in ['B', 'L', 'U']:
                    for file in gem_files[light]:
                        file_base = file.replace('.txt', '')
                        all_files.append((file_base, file, gem_num, light))
                        print(f"   {len(all_files)}. {file_base}")

        if not all_files:
            print("\n❌ No complete gem sets found!")
            return

        print(f"\n🔍 SELECTION METHOD:")
        print("Enter 3 file numbers (B, L, U) separated by spaces")
        print("Example: 1 5 9 (for files 1, 5, and 9)")
        print("Or enter a gem base number like 'C0045' for auto-selection")

        choice = input("\nYour selection: ").strip()

        selected = {}

        # Try parsing as numbers first
        try:
            numbers = [int(x) for x in choice.split()]
            if len(numbers) == 3:
                selected_files = []
                for num in numbers:
                    if 1 <= num <= len(all_files):
                        selected_files.append(all_files[num-1])
                    else:
                        print(f"❌ Number {num} out of range (1-{len(all_files)})")
                        return

                # Check if we have B, L, U
                lights_found = {f[3] for f in selected_files}
                if lights_found != {'B', 'L', 'U'}:
                    print(f"❌ Need one file from each light source (B, L, U)")
                    print(f"You selected: {lights_found}")
                    return

                # Store selected files
                for file_info in selected_files:
                    file_base, file_full, gem_num, light = file_info
                    selected[light] = file_full
                    print(f"   Selected {light}: {file_base}")

                gem_choice = selected_files[0][2]  # Use gem number from first file

            else:
                print("❌ Please enter exactly 3 numbers")
                return

        except ValueError:
            # Try as gem base number (old method)
            if choice in gems:
                gem_choice = choice
                gem_files = gems[gem_choice]

                print(f"\n💎 AUTO-SELECTING FILES FOR GEM {gem_choice}:")
                for light in ['B', 'L', 'U']:
                    if gem_files[light]:
                        selected[light] = gem_files[light][0]
                        file_base = selected[light].replace('.txt', '')
                        print(f"   {light}: {file_base}")
            else:
                print(f"❌ Invalid selection. Use numbers or gem base like 'C0045'")
                return

        if len(selected) != 3:
            print("\n❌ Incomplete selection - need B, L, and U files")
            return

        # Convert files with CORRECTED normalization
        print(f"\n🔄 PREPARING ANALYSIS...")
        success = self.convert_gem_files_corrected(selected, gem_choice)

        if success:
            # Run validation check
            print(f"\n🔍 VALIDATING NORMALIZATION...")
            self.validate_normalization(gem_choice)

            # Run analysis
            print(f"\n✅ FILES READY FOR ANALYSIS")
            analysis_choice = input(f"Run numerical analysis now? (y/n): ").strip().lower()

            if analysis_choice == 'y':
                self.run_numerical_analysis_fixed()

                # Offer visualization
                viz_choice = input(f"\nShow spectral comparison plots? (y/n): ").strip().lower()
                if viz_choice == 'y':
                    self.create_spectral_comparison_plots(gem_choice)
        else:
            print(f"\n❌ Failed to prepare analysis")
    
    def convert_gem_files_corrected(self, selected_files, gem_number):
        """Convert selected gem files with CORRECTED normalization - Windows Permission Fix"""
        try:
            # Try the normal method first
            if os.path.exists('raw_txt'):
                shutil.rmtree('raw_txt')
            os.makedirs('raw_txt')
            
            # Copy files to raw_txt
            print("   📁 Copying files to raw_txt...")
            for light, filename in selected_files.items():
                src = os.path.join('data/raw', filename)
                dst = os.path.join('raw_txt', filename)
                shutil.copy2(src, dst)
                print(f"     ✅ {light}: {filename}")
            
            # Create data/unknown directory
            os.makedirs('data/unknown', exist_ok=True)
            
            # Convert each file with CORRECTED normalization
            print("   🔧 Converting and normalizing (CORRECTED)...")
            
            for light, filename in selected_files.items():
                input_path = os.path.join('raw_txt', filename)
                output_path = f'data/unknown/unkgem{light}.csv'
                
                # Read file
                df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                wavelengths = np.array(df['wavelength'])
                intensities = np.array(df['intensity'])
                
                # Apply CORRECTED normalization
                normalized = self.correct_normalize_spectrum(wavelengths, intensities, light)
                
                # Save normalized data
                output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
                output_df.to_csv(output_path, header=False, index=False)
                
                print(f"     ✅ {light}: {len(output_df)} points, range {normalized.min():.3f}-{normalized.max():.3f}")
            
            return True
            
        except (PermissionError, OSError) as e:
            print(f"     ⚠️ Permission error with raw_txt: {e}")
            print("     🔄 Switching to BYPASS MODE (direct conversion)...")
            return self.convert_gem_files_bypass(selected_files, gem_number)
        except Exception as e:
            print(f"     ❌ Conversion error: {e}")
            return False
    
    def convert_gem_files_bypass(self, selected_files, gem_number):
        """Convert files directly without raw_txt copying - Windows Permission Bypass"""
        try:
            # Create data/unknown directory only
            os.makedirs('data/unknown', exist_ok=True)
            
            print("   🔧 Converting directly (BYPASS MODE)...")
            
            for light, filename in selected_files.items():
                input_path = os.path.join('data/raw', filename)
                output_path = f'data/unknown/unkgem{light}.csv'
                
                # Read and normalize directly from data/raw
                df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                wavelengths = np.array(df['wavelength'])
                intensities = np.array(df['intensity'])
                
                # Apply CORRECTED normalization
                normalized = self.correct_normalize_spectrum(wavelengths, intensities, light)
                
                # Save normalized data
                output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
                output_df.to_csv(output_path, header=False, index=False)
                
                print(f"     ✅ {light}: {len(output_df)} points, range {normalized.min():.3f}-{normalized.max():.3f}")
            
            return True
            
        except Exception as e:
            print(f"     ❌ Bypass conversion error: {e}")
            return False
    
    def validate_normalization(self, gem_number):
        """Validate that normalization produces expected results"""
        print("   🔍 Checking normalization against database...")
        
        for light in ['B', 'L', 'U']:
            try:
                # Load our normalized data
                unknown_path = f'data/unknown/unkgem{light}.csv'
                unknown_df = pd.read_csv(unknown_path, header=None, names=['wavelength', 'intensity'])
                
                # Load database
                db_path = f'gemini_db_long_{light}.csv'
                if os.path.exists(db_path):
                    db_df = pd.read_csv(db_path)
                    
                    # Look for exact gem match in database
                    gem_matches = db_df[db_df['full_name'].str.contains(gem_number, na=False)]
                    
                    if not gem_matches.empty:
                        # Get first match
                        match = gem_matches.iloc[0]
                        print(f"     🎯 {light}: Found {match['full_name']} in database")
                        
                        # Compare ranges
                        unknown_range = f"{unknown_df['intensity'].min():.3f}-{unknown_df['intensity'].max():.3f}"
                        db_subset = db_df[db_df['full_name'] == match['full_name']]
                        db_range = f"{db_subset['intensity'].min():.3f}-{db_subset['intensity'].max():.3f}"
                        
                        print(f"         Unknown range: {unknown_range}")
                        print(f"         Database range: {db_range}")
                    else:
                        print(f"     ⚠️ {light}: No match for {gem_number} in database")
                else:
                    print(f"     ❌ {light}: Database file {db_path} not found")
                    
            except Exception as e:
                print(f"     ❌ {light}: Validation error - {e}")
    
    def run_numerical_analysis_fixed(self):
        """Run numerical analysis with fixed normalization - DIRECT IN-PROCESS VERSION"""
        print(f"\n🚀 RUNNING FIXED NUMERICAL ANALYSIS (DIRECT)...")
        
        try:
            # Run analysis directly in this process to avoid encoding issues
            self.direct_numerical_analysis()
                
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def direct_numerical_analysis(self):
        """Direct numerical analysis with CLEAR result caching and exact file tracking - SUPPORTS PARTIAL GEMS"""
        print("   📊 Starting analysis with cache clearing and exact file tracking...")
        
        # Clear any previous analysis variables to prevent caching
        self.current_analysis_results = {}
        self.current_gem_identifier = None
        
        # Check for unknown files - allow partial sets
        unknown_files = {}
        available_lights = []
        
        for light in ['B', 'L', 'U']:
            found = False
            for base_path in ['data/unknown', '.']:
                test_path = os.path.join(base_path, f'unkgem{light}.csv')
                if os.path.exists(test_path):
                    unknown_files[light] = test_path
                    available_lights.append(light)
                    found = True
                    break
        
        if len(available_lights) < 2:
            print(f"   ❌ Need at least 2 light sources, found: {available_lights}")
            return
        
        print(f"   ✅ Found {len(available_lights)} light sources: {'+'.join(available_lights)}")
        
        db_files = {'B': 'gemini_db_long_B.csv', 'L': 'gemini_db_long_L.csv', 'U': 'gemini_db_long_U.csv'}
        
        # Check database files for available lights
        for light in available_lights:
            if not os.path.exists(db_files[light]):
                print(f"   ❌ Database file {db_files[light]} not found")
                return
        
        print("   ✅ All required database files found")
        
        # Determine which gem we're analyzing by checking unknown file contents
        actual_gem_id = self.identify_unknown_gem(unknown_files)
        print(f"   🎯 Analyzing unknown gem: {actual_gem_id}")
        print(f"   📋 Using light sources: {'+'.join(available_lights)}")
        
        # Load gem library for descriptions
        gem_name_map = {}
        try:
            gemlib = pd.read_csv('gemlib_structural_ready.csv')
            gemlib.columns = gemlib.columns.str.strip()
            if 'Reference' in gemlib.columns:
                gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                if all(col in gemlib.columns for col in expected_columns):
                    gemlib['Gem Description'] = gemlib[expected_columns].apply(
                        lambda x: ' '.join([v if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                    gem_name_map = dict(zip(gemlib['Reference'], gemlib['Gem Description']))
        except Exception as e:
            print(f"   ⚠️ Could not load gem descriptions: {e}")
        
        # Process each available light source
        all_matches = {}
        gem_best_scores = {}
        gem_best_names = {}
        
        for light_source in available_lights:
            print(f"\n   🔍 Processing {light_source} light (PARTIAL ANALYSIS)...")
            
            try:
                # Load unknown spectrum (normalized values from our processing)
                unknown = pd.read_csv(unknown_files[light_source], header=None, names=['wavelength', 'intensity'])
                print(f"      Unknown: {len(unknown)} points, range {unknown['intensity'].min():.3f}-{unknown['intensity'].max():.3f}")
                
                # Load database (normalized values)
                db = pd.read_csv(db_files[light_source])
                print(f"      Database: {len(db)} points, {db['full_name'].nunique()} unique gems")
                
                # Apply 0-100 scaling to unknown data
                unknown_scaled = unknown.copy()
                unknown_scaled['intensity'] = self.apply_0_100_scaling(unknown['wavelength'].values, unknown['intensity'].values)
                print(f"      Unknown scaled: range {unknown_scaled['intensity'].min():.3f}-{unknown_scaled['intensity'].max():.3f}")
                
                # Compute scores for all gems with fresh variables
                current_scores = []  # Use fresh variable name
                for gem_name in db['full_name'].unique():
                    reference = db[db['full_name'] == gem_name].copy()
                    
                    # Apply 0-100 scaling to database reference
                    reference_scaled = reference.copy()
                    reference_scaled['intensity'] = self.apply_0_100_scaling(reference['wavelength'].values, reference['intensity'].values)
                    
                    # Compute match score using 0-100 scaled values
                    merged = pd.merge(unknown_scaled, reference_scaled, on='wavelength', suffixes=('_unknown', '_ref'))
                    if len(merged) > 0:
                        mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
                        log_score = np.log1p(mse)
                        current_scores.append((gem_name, log_score))
                
                # Sort by score (best = lowest) with fresh variable
                current_sorted_scores = sorted(current_scores, key=lambda x: x[1])
                all_matches[light_source] = current_sorted_scores
                
                print(f"      ✅ Best matches for {light_source} (PARTIAL ANALYSIS):")
                for i, (gem, score) in enumerate(current_sorted_scores[:5], 1):
                    print(f"         {i}. {gem}: {score:.6f}")
                
                # Track best scores per gem ID with fresh tracking
                for gem_name, score in current_sorted_scores:
                    base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                    if base_id not in gem_best_scores:
                        gem_best_scores[base_id] = {}
                        gem_best_names[base_id] = {}
                    if score < gem_best_scores[base_id].get(light_source, np.inf):
                        gem_best_scores[base_id][light_source] = score
                        gem_best_names[base_id][light_source] = gem_name
                
            except Exception as e:
                print(f"      ❌ Error processing {light_source}: {e}")
        
        # Filter to gems with ALL available light sources (not necessarily all 3)
        complete_gems = {gid: scores for gid, scores in gem_best_scores.items() 
                        if set(scores.keys()) >= set(available_lights)}
        
        # Calculate combined scores with fresh aggregation
        fresh_aggregated_scores = {base_id: sum(scores.values()) 
                                 for base_id, scores in complete_gems.items()}
        
        # Sort final results with fresh sorting
        fresh_final_sorted = sorted(fresh_aggregated_scores.items(), key=lambda x: x[1])
        
        print(f"\n🏆 PARTIAL ANALYSIS RESULTS - TOP 20 MATCHES:")
        print("=" * 70)
        print(f"   Analysis using: {'+'.join(available_lights)} light sources")
        print("=" * 70)
        
        for i, (base_id, total_score) in enumerate(fresh_final_sorted[:20], start=1):
            gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
            sources = complete_gems.get(base_id, {})
            
            print(f"  Rank {i:2}: {gem_desc} (ID: {base_id})")
            print(f"          Total Score: {total_score:.6f}")
            for ls in sorted(available_lights):
                if ls in sources:
                    score_val = sources[ls]
                    best_file = gem_best_names[base_id][ls]
                    print(f"          {ls} Score: {score_val:.6f} (vs {best_file})")
            print()
        
        # Check for self-matching with the ACTUAL gem we analyzed
        if actual_gem_id in fresh_aggregated_scores:
            self_rank = next(i for i, (gid, _) in enumerate(fresh_final_sorted, 1) if gid == actual_gem_id)
            self_score = fresh_aggregated_scores[actual_gem_id]
            print(f"🎯 {actual_gem_id} SELF-MATCH RESULT (PARTIAL ANALYSIS):")
            print(f"   Rank: {self_rank}")
            print(f"   Total Score: {self_score:.6f}")
            print(f"   Light sources used: {'+'.join(available_lights)}")
            
            if self_score < 1e-10:
                print(f"   ✅ PERFECT SELF-MATCH!")
            elif self_score < 1e-6:
                print(f"   ✅ EXCELLENT SELF-MATCH!")
            elif self_score < 1e-3:
                print(f"   ✅ GOOD SELF-MATCH!")
            else:
                print(f"   ⚠️ POOR SELF-MATCH - check normalization")
        else:
            print(f"🎯 {actual_gem_id} NOT FOUND in results - check database entries")
        
        print(f"\n📊 PARTIAL ANALYSIS SUMMARY:")
        print(f"   Analyzed gem: {actual_gem_id}")
        print(f"   Light sources: {'+'.join(available_lights)} ({len(available_lights)}/3)")
        print(f"   Database: Normalized values stored (~15K-50K ranges)")
        print(f"   Analysis: 0-100 scaling applied to both sides for comparison")
        print(f"   Total gems analyzed: {len(fresh_final_sorted)}")
        print(f"   Perfect matches (score < 1e-10): {sum(1 for _, score in fresh_final_sorted if score < 1e-10)}")
        
        # Store results for visualization
        self.current_analysis_results = fresh_final_sorted
        self.current_gem_identifier = actual_gem_id
        
        return fresh_final_sorted
    
    def identify_unknown_gem(self, unknown_files):
        """Identify which gem we're actually analyzing by checking file contents"""
        # Try to determine from source files in raw_txt if available
        if os.path.exists('raw_txt'):
            txt_files = [f for f in os.listdir('raw_txt') if f.endswith('.txt')]
            if txt_files:
                # Extract gem ID from first file
                first_file = txt_files[0]
                file_base = first_file.replace('.txt', '')
                # Extract everything before the light source letter
                for light in ['B', 'L', 'U']:
                    if light in file_base.upper():
                        return file_base[:file_base.upper().find(light)]
        
        # Fallback: check if unknown file matches known patterns in database
        try:
            unknown_b = pd.read_csv(unknown_files['B'], header=None, names=['wavelength', 'intensity'])
            db_b = pd.read_csv('gemini_db_long_B.csv')
            
            # Find exact matches in database
            for gem_name in db_b['full_name'].unique():
                reference = db_b[db_b['full_name'] == gem_name]
                merged = pd.merge(unknown_b, reference, on='wavelength', suffixes=('_unknown', '_ref'))
                if len(merged) > 0:
                    mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
                    if mse < 1e-10:  # Perfect match
                        base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                        return base_id
        except:
            pass
        
        return "UNKNOWN"
    
    def create_spectral_comparison_plots(self, gem_identifier):
        """Create comprehensive spectral comparison plots"""
        print(f"\n📊 CREATING SPECTRAL COMPARISON PLOTS FOR {gem_identifier}")
        print("=" * 60)
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("❌ matplotlib not available - cannot create plots")
            return
        
        if not hasattr(self, 'current_analysis_results') or not self.current_analysis_results:
            print("❌ No analysis results available - run analysis first")
            return
        
        # Get top 5 matches
        top_matches = self.current_analysis_results[:5]
        
        # Create comprehensive plots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Spectral Analysis: {gem_identifier} vs Top Matches', fontsize=16, fontweight='bold')
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for light_idx, light in enumerate(['B', 'L', 'U']):
            print(f"🔍 Creating {light} light plots...")
            
            try:
                # Load unknown spectrum
                unknown_file = f'unkgem{light}.csv'
                if os.path.exists(unknown_file):
                    unknown = pd.read_csv(unknown_file, header=None, names=['wavelength', 'intensity'])
                else:
                    unknown_file = f'data/unknown/unkgem{light}.csv'
                    unknown = pd.read_csv(unknown_file, header=None, names=['wavelength', 'intensity'])
                
                # Load database
                db = pd.read_csv(f'gemini_db_long_{light}.csv')
                
                # Plot 1: Raw normalized comparison
                axes[light_idx, 0].plot(unknown['wavelength'], unknown['intensity'], 
                                      'black', linewidth=2, label=f'{gem_identifier} (Unknown)', alpha=0.8)
                
                for i, (match_id, score) in enumerate(top_matches[:3]):
                    match_entries = db[db['full_name'].str.startswith(match_id)]
                    if not match_entries.empty:
                        # Get best matching file for this light
                        match_file = f"{match_id}{light}C1"
                        if match_file not in match_entries['full_name'].values:
                            match_file = match_entries['full_name'].iloc[0]
                        
                        match_data = db[db['full_name'] == match_file]
                        if not match_data.empty:
                            axes[light_idx, 0].plot(match_data['wavelength'], match_data['intensity'], 
                                                  colors[i], linewidth=1, label=f'{match_file} (Score: {score:.3f})', alpha=0.7)
                
                axes[light_idx, 0].set_title(f'{light} Light - Normalized Spectra')
                axes[light_idx, 0].set_xlabel('Wavelength (nm)')
                axes[light_idx, 0].set_ylabel('Normalized Intensity')
                axes[light_idx, 0].legend(fontsize=8)
                axes[light_idx, 0].grid(True, alpha=0.3)
                
                # Plot 2: 0-100 Scaled comparison
                unknown_scaled = unknown.copy()
                unknown_scaled['intensity'] = self.apply_0_100_scaling(unknown['wavelength'].values, unknown['intensity'].values)
                
                axes[light_idx, 1].plot(unknown_scaled['wavelength'], unknown_scaled['intensity'], 
                                      'black', linewidth=2, label=f'{gem_identifier} (0-100 scaled)', alpha=0.8)
                
                for i, (match_id, score) in enumerate(top_matches[:3]):
                    match_entries = db[db['full_name'].str.startswith(match_id)]
                    if not match_entries.empty:
                        match_file = f"{match_id}{light}C1"
                        if match_file not in match_entries['full_name'].values:
                            match_file = match_entries['full_name'].iloc[0]
                        
                        match_data = db[db['full_name'] == match_file]
                        if not match_data.empty:
                            match_scaled = match_data.copy()
                            match_scaled['intensity'] = self.apply_0_100_scaling(match_data['wavelength'].values, match_data['intensity'].values)
                            axes[light_idx, 1].plot(match_scaled['wavelength'], match_scaled['intensity'], 
                                                  colors[i], linewidth=1, label=f'{match_file}', alpha=0.7)
                
                axes[light_idx, 1].set_title(f'{light} Light - 0-100 Scaled (Analysis Method)')
                axes[light_idx, 1].set_xlabel('Wavelength (nm)')
                axes[light_idx, 1].set_ylabel('Scaled Intensity (0-100)')
                axes[light_idx, 1].legend(fontsize=8)
                axes[light_idx, 1].grid(True, alpha=0.3)
                
                # Plot 3: Difference plot (unknown vs best match)
                best_match_id = top_matches[0][0]
                best_match_entries = db[db['full_name'].str.startswith(best_match_id)]
                if not best_match_entries.empty:
                    best_match_file = f"{best_match_id}{light}C1"
                    if best_match_file not in best_match_entries['full_name'].values:
                        best_match_file = best_match_entries['full_name'].iloc[0]
                    
                    best_match_data = db[db['full_name'] == best_match_file]
                    if not best_match_data.empty:
                        # Scale both for difference calculation
                        unknown_for_diff = self.apply_0_100_scaling(unknown['wavelength'].values, unknown['intensity'].values)
                        match_for_diff = self.apply_0_100_scaling(best_match_data['wavelength'].values, best_match_data['intensity'].values)
                        
                        # Calculate difference
                        merged_for_diff = pd.merge(pd.DataFrame({'wavelength': unknown['wavelength'], 'intensity': unknown_for_diff}),
                                                 pd.DataFrame({'wavelength': best_match_data['wavelength'], 'intensity': match_for_diff}),
                                                 on='wavelength', suffixes=('_unknown', '_match'))
                        
                        if not merged_for_diff.empty:
                            difference = merged_for_diff['intensity_unknown'] - merged_for_diff['intensity_match']
                            axes[light_idx, 2].plot(merged_for_diff['wavelength'], difference, 'red', linewidth=1)
                            axes[light_idx, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                            
                            rms_diff = np.sqrt(np.mean(difference**2))
                            axes[light_idx, 2].set_title(f'{light} Light - Difference (RMS: {rms_diff:.3f})')
                            axes[light_idx, 2].text(0.02, 0.98, f'vs {best_match_file}', transform=axes[light_idx, 2].transAxes,
                                                   verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                axes[light_idx, 2].set_xlabel('Wavelength (nm)')
                axes[light_idx, 2].set_ylabel('Intensity Difference')
                axes[light_idx, 2].grid(True, alpha=0.3)
                
                # Plot 4: Match scores histogram
                all_scores = [score for _, score in self.current_analysis_results]
                axes[light_idx, 3].hist(all_scores, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
                axes[light_idx, 3].axvline(x=top_matches[0][1], color='red', linestyle='--', linewidth=2, label='Best Match')
                if len(top_matches) > 1:
                    axes[light_idx, 3].axvline(x=top_matches[1][1], color='orange', linestyle='--', linewidth=1, label='2nd Best')
                
                axes[light_idx, 3].set_title(f'Score Distribution (All Gems)')
                axes[light_idx, 3].set_xlabel('Log Score')
                axes[light_idx, 3].set_ylabel('Number of Gems')
                axes[light_idx, 3].legend(fontsize=8)
                axes[light_idx, 3].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"❌ Error creating {light} light plots: {e}")
                # Fill with error message
                axes[light_idx, 0].text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=axes[light_idx, 0].transAxes)
                for j in range(1, 4):
                    axes[light_idx, j].text(0.5, 0.5, 'Error', ha='center', va='center', transform=axes[light_idx, j].transAxes)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f'spectral_analysis_{gem_identifier}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"💾 Plot saved as: {plot_filename}")
        
        try:
            plt.show()
        except:
            print("⚠️ Cannot display plot interactively, but file saved successfully")
        
        # Create summary report
        self.create_analysis_summary_report(gem_identifier, top_matches)
    
    def create_enhanced_analysis_report(self, gem_identifier, top_matches):
        """Create enhanced text summary report with detailed analysis"""
        report_filename = f'enhanced_report_{gem_identifier}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_filename, 'w') as f:
            f.write(f"ENHANCED GEMINI SPECTRAL ANALYSIS REPORT\n")
            f.write(f"========================================\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Unknown Gem: {gem_identifier}\n")
            f.write(f"Analysis Type: 0-100 Scaled Comparison\n\n")
            
            f.write(f"TOP 10 MATCHES (with detailed scores):\n")
            f.write(f"-" * 60 + "\n")
            
            for i, (match_id, score) in enumerate(top_matches[:10], 1):
                # Load gem description if available
                try:
                    gemlib = pd.read_csv('gemlib_structural_ready.csv')
                    gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                    match_desc = gemlib[gemlib['Reference'] == str(match_id)]
                    if not match_desc.empty:
                        desc = match_desc.iloc[0]['Nat./Syn.'] + " " + match_desc.iloc[0]['Spec.']
                    else:
                        desc = f"Gem {match_id}"
                except:
                    desc = f"Gem {match_id}"
                
                f.write(f"Rank {i:2}: {desc} (ID: {match_id})\n")
                f.write(f"         Total Score: {score:.6f}\n")
                
                # Quality assessment with more detail
                if score < 1e-10:
                    f.write(f"         Quality: ★★★★★ PERFECT MATCH (Identical spectra)\n")
                elif score < 1e-6:
                    f.write(f"         Quality: ★★★★☆ EXCELLENT MATCH (Near identical)\n")
                elif score < 1e-3:
                    f.write(f"         Quality: ★★★☆☆ GOOD MATCH (Very similar)\n")
                elif score < 1:
                    f.write(f"         Quality: ★★☆☆☆ MODERATE MATCH (Some similarity)\n")
                else:
                    f.write(f"         Quality: ★☆☆☆☆ POOR MATCH (Different spectra)\n")
                f.write(f"\n")
            
            # Add statistical summary
            all_scores = [score for _, score in self.current_analysis_results]
            f.write(f"\nSTATISTICAL SUMMARY:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Total gems analyzed: {len(self.current_analysis_results)}\n")
            f.write(f"Best match score: {min(all_scores):.6f}\n")
            f.write(f"Median score: {np.median(all_scores):.6f}\n")
            f.write(f"Average score: {np.mean(all_scores):.6f}\n")
            f.write(f"Perfect matches (< 1e-10): {sum(1 for s in all_scores if s < 1e-10)}\n")
            f.write(f"Excellent matches (< 1e-6): {sum(1 for s in all_scores if s < 1e-6)}\n")
            
            f.write(f"\nANALYSIS PARAMETERS:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Normalization: B(650nm→50K), L(Max→50K), U(811nm→15K)\n")
            f.write(f"Comparison: 0-100 scaling applied to both unknown and database\n")
            f.write(f"Scoring: Mean Squared Error with log transformation\n")
            f.write(f"Line thickness: 0.5 for matches, 1.5 for unknown\n")
            f.write(f"Plot resolution: 200 DPI for detailed viewing\n")
        
        print(f"📄 Enhanced analysis report saved as: {report_filename}")
    
    def show_enhanced_file_listing(self, gems, all_analyzable_gems):
        """Show enhanced, organized file listing for better gem selection"""
        print("\n📂 ENHANCED FILE SELECTION INTERFACE")
        print("=" * 80)
        
        all_files = []
        current_number = 1
        
        # Organize by gem type (Complete vs Partial)
        complete_gems = []
        partial_gems = []
        
        for gem_num in sorted(all_analyzable_gems):
            gem_files = gems[gem_num]
            available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]
            
            if len(available) == 3:
                complete_gems.append(gem_num)
            else:
                partial_gems.append(gem_num)
        
        # Show complete gems first
        if complete_gems:
            print(f"\n🟢 COMPLETE GEMS (B+L+U available) - {len(complete_gems)} gems")
            print("-" * 50)
            
            for gem_num in complete_gems:
                gem_files = gems[gem_num]
                print(f"\n💎 Gem {gem_num}:")
                
                # Group by light source for cleaner display
                for light in ['B', 'L', 'U']:
                    if light in gem_files and gem_files[light]:
                        print(f"  {light} Light:")
                        for file in gem_files[light]:
                            file_base = file.replace('.txt', '')
                            all_files.append((file_base, file, gem_num, light))
                            # Show measurement type
                            measurement_type = "C1 (Standard)" if "C1" in file else "P1/P2 (Alternative)"
                            print(f"    {current_number:3}. {file_base:<15} ({measurement_type})")
                            current_number += 1
        
        # Show partial gems
        if partial_gems:
            print(f"\n🟡 PARTIAL GEMS (2 light sources) - {len(partial_gems)} gems")
            print("-" * 50)
            
            for gem_num in partial_gems:
                gem_files = gems[gem_num]
                available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]
                
                print(f"\n💎 Gem {gem_num} - Available: {'+'.join(available)}:")
                
                for light in ['B', 'L', 'U']:
                    if light in available and gem_files[light]:
                        print(f"  {light} Light:")
                        for file in gem_files[light]:
                            file_base = file.replace('.txt', '')
                            all_files.append((file_base, file, gem_num, light))
                            measurement_type = "C1 (Standard)" if "C1" in file else "P1/P2 (Alternative)"
                            print(f"    {current_number:3}. {file_base:<15} ({measurement_type})")
                            current_number += 1
        
        return all_files
    
    def visualize_individual_spectrum(self):
        """Visualize a specific spectrum from the database"""
        print("\n📊 INDIVIDUAL SPECTRUM VISUALIZATION")
        print("=" * 40)
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("❌ matplotlib not available")
            return
        
        # Get gem selection
        gem_name = input("Enter full gem name (e.g., C0045BC1, 140LC1): ").strip()
        
        if not gem_name:
            print("❌ No gem name provided")
            return
        
        # Find in databases
        found_spectra = {}
        db_files = {'B': 'gemini_db_long_B.csv', 'L': 'gemini_db_long_L.csv', 'U': 'gemini_db_long_U.csv'}
        
        for light, db_file in db_files.items():
            if os.path.exists(db_file):
                df = pd.read_csv(db_file)
                matches = df[df['full_name'] == gem_name]
                if not matches.empty:
                    found_spectra[light] = matches
        
        if not found_spectra:
            print(f"❌ {gem_name} not found in any database")
            return
        
        print(f"✅ Found {gem_name} in {len(found_spectra)} light sources")
    
    def run_debug_analysis(self):
        """Run comprehensive debug analysis to validate fixes"""
        print("\n🔍 RUNNING COMPREHENSIVE DEBUG ANALYSIS")
        print("=" * 50)
        
        # Check which gem files are available
        gems = self.scan_available_gems()
        if not gems:
            print("❌ No gems available for debug analysis")
            return
        
        # Find a gem that exists in database
        test_gems = ['C0034', '140', 'C0001']  # Common test gems
        found_gem = None
        
        for test_gem in test_gems:
            if test_gem in gems and len(gems[test_gem]) >= 3:
                available_lights = [ls for ls in ['B', 'L', 'U'] if gems[test_gem][ls]]
                if len(available_lights) == 3:
                    found_gem = test_gem
                    break
        
        if not found_gem:
            # Use any complete gem
            complete_gems = [g for g in gems.keys() if len([ls for ls in ['B', 'L', 'U'] if gems[g][ls]]) == 3]
            if complete_gems:
                found_gem = complete_gems[0]
        
        if not found_gem:
            print("❌ No complete gems found for debug analysis")
            return
        
        print(f"🎯 Testing with Gem {found_gem}")
        
        # Process the gem with corrected normalization
        selected = {}
        for light in ['B', 'L', 'U']:
            if gems[found_gem][light]:
                selected[light] = gems[found_gem][light][0]
        
        # Convert with corrected normalization
        success = self.convert_gem_files_corrected(selected, found_gem)
        
        if success:
            # Run validation
            self.validate_normalization(found_gem)
            
            # Quick analysis to check scores
            print("\n🔬 RUNNING QUICK MATCH TEST...")
            self.quick_match_test(found_gem)
        else:
            print("❌ Failed to process test gem")
    
    def quick_match_test(self, gem_number):
        """Quick test to see if gem matches itself with score ~0"""
        for light in ['B', 'L', 'U']:
            try:
                # Load our processed data
                unknown = pd.read_csv(f'data/unknown/unkgem{light}.csv', header=None, names=['wavelength', 'intensity'])
                
                # Load database
                db = pd.read_csv(f'gemini_db_long_{light}.csv')
                
                # Find matching entries
                matches = db[db['full_name'].str.contains(gem_number, na=False)]
                
                if not matches.empty:
                    # Test against first match
                    match_name = matches.iloc[0]['full_name']
                    reference = db[db['full_name'] == match_name]
                    
                    # Compute score
                    merged = pd.merge(unknown, reference, on='wavelength', suffixes=('_unknown', '_ref'))
                    mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
                    log_score = np.log1p(mse)
                    
                    print(f"   {light}: vs {match_name} -> MSE: {mse:.3f}, Log Score: {log_score:.3f}")
                    
                    if log_score < 1.0:
                        print(f"      ✅ Excellent match!")
                    elif log_score < 5.0:
                        print(f"      ✅ Good match!")
                    else:
                        print(f"      ⚠️ Poor match - may need further normalization adjustment")
                else:
                    print(f"   {light}: No database match found for {gem_number}")
            
            except Exception as e:
                print(f"   {light}: Test error - {e}")
    
    def debug_exact_normalization(self):
        """Deep debug of exact normalization discrepancy"""
        print("\n🔬 DEEP NORMALIZATION DEBUGGING")
        print("=" * 50)
        print("Finding EXACT discrepancy between our normalization and database...")
        
        # Test with C0034 specifically
        gem_files = self.scan_available_gems()
        if 'C0034' not in gem_files:
            print("❌ C0034 not found")
            return
        
        print("\n🎯 ANALYZING C0034 NORMALIZATION DISCREPANCY")
        
        for light in ['B', 'L', 'U']:
            print(f"\n📊 {light} LIGHT ANALYSIS:")
            print("-" * 30)
            
            try:
                # Load raw data
                raw_file = os.path.join('data/raw', gem_files['C0034'][light][0])
                df = pd.read_csv(raw_file, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                wavelengths = np.array(df['wavelength'])
                raw_intensities = np.array(df['intensity'])
                
                print(f"   Raw data: {len(df)} points")
                print(f"   Raw wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
                print(f"   Raw intensity range: {raw_intensities.min():.3f} - {raw_intensities.max():.3f}")
                
                # Apply our normalization
                normalized = self.correct_normalize_spectrum(wavelengths, raw_intensities, light)
                print(f"   Our normalized range: {normalized.min():.3f} - {normalized.max():.3f}")
                
                # Load database and find C0034 entries
                db_file = f'gemini_db_long_{light}.csv'
                if os.path.exists(db_file):
                    db_df = pd.read_csv(db_file)
                    c0034_entries = db_df[db_df['full_name'].str.contains('C0034', na=False)]
                    
                    if not c0034_entries.empty:
                        print(f"   Database C0034 entries found: {len(c0034_entries['full_name'].unique())}")
                        
                        for entry_name in c0034_entries['full_name'].unique():
                            entry_data = db_df[db_df['full_name'] == entry_name]
                            db_wavelengths = entry_data['wavelength'].values
                            db_intensities = entry_data['intensity'].values
                            
                            print(f"   \n   📋 Database entry: {entry_name}")
                            print(f"      DB wavelength range: {db_wavelengths.min():.1f} - {db_wavelengths.max():.1f} nm")
                            print(f"      DB intensity range: {db_intensities.min():.3f} - {db_intensities.max():.3f}")
                            
                            # Check if wavelength ranges match
                            wave_match = np.allclose(wavelengths, db_wavelengths, rtol=1e-6)
                            print(f"      Wavelength match: {wave_match}")
                            
                            if wave_match:
                                # Compare intensities directly
                                intensity_diff = normalized - db_intensities
                                mse = np.mean(intensity_diff**2)
                                max_diff = np.max(np.abs(intensity_diff))
                                
                                print(f"      Intensity MSE: {mse:.6f}")
                                print(f"      Max difference: {max_diff:.6f}")
                                print(f"      First 5 differences: {intensity_diff[:5]}")
                                
                                if mse < 1e-10:
                                    print(f"      ✅ PERFECT MATCH!")
                                elif mse < 1e-6:
                                    print(f"      ✅ Very close match")
                                elif mse < 1e-3:
                                    print(f"      ⚠️ Small differences")
                                else:
                                    print(f"      ❌ SIGNIFICANT DIFFERENCES")
                            else:
                                print(f"      ❌ Wavelength ranges don't match - interpolation needed")
                    else:
                        print(f"   ❌ No C0034 entries found in {db_file}")
                else:
                    print(f"   ❌ Database file {db_file} not found")
                    
            except Exception as e:
                print(f"   ❌ Error analyzing {light}: {e}")
    
    def direct_analysis_bypass(self):
        """Direct analysis that bypasses file copying issues"""
        print("\n🚀 DIRECT ANALYSIS (BYPASS MODE)")
        print("=" * 40)
        print("This mode processes files directly without copying to avoid permission issues.")
        
        # Scan gems
        gems = self.scan_available_gems()
        if not gems:
            return
        
        # Show options
        complete_gems, partial_gems = self.show_available_gems(gems)
        
        if not complete_gems:
            print("\n❌ No complete gem sets found!")
            return
        
        # Get choice
        print(f"\n🔍 Available complete gems: {', '.join(complete_gems)}")
        
        while True:
            gem_choice = input(f"\nEnter gem number to analyze (or 'back'): ").strip()
            
            if gem_choice.lower() == 'back':
                return
            
            if gem_choice in gems:
                break
            
            print(f"❌ Not found. Available: {', '.join(sorted(gems.keys()))}")
        
        # Process directly
        selected = {}
        gem_files = gems[gem_choice]
        
        print(f"\n💎 PROCESSING GEM {gem_choice} DIRECTLY:")
        for light in ['B', 'L', 'U']:
            if gem_files[light]:
                selected[light] = gem_files[light][0]
                print(f"   {light}: {selected[light]}")
        
        # Direct conversion
        try:
            for light, filename in selected.items():
                # Read directly from data/raw
                input_path = os.path.join('data/raw', filename)
                output_path = f'unkgem{light}.csv'
                
                # Read and normalize
                df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                wavelengths = np.array(df['wavelength'])
                intensities = np.array(df['intensity'])
                
                # Apply corrected normalization
                normalized = self.correct_normalize_spectrum(wavelengths, intensities, light)
                
                # Save to current directory
                output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
                output_df.to_csv(output_path, header=False, index=False)
                
                print(f"   ✅ {light}: {len(output_df)} points, range {normalized.min():.3f}-{normalized.max():.3f}")
            
            print(f"\n✅ Files created in current directory: unkgemB.csv, unkgemL.csv, unkgemU.csv")
            
            print("🚀 Running analysis...")
            
            # Run analysis
            self.run_numerical_analysis_fixed()
            
        except Exception as e:
            print(f"\n❌ Direct analysis error: {e}")
    
    def run_structural_analysis_hub(self):
        """Launch structural analysis hub"""
        hub_path = 'src/structural_analysis/main.py'
        if os.path.exists(hub_path):
            try:
                subprocess.run([sys.executable, hub_path])
            except Exception as e:
                print(f"Error launching structural hub: {e}")
        else:
            print(f"❌ {hub_path} not found")
    
    def run_structural_launcher(self):
        """Launch structural analyzers launcher"""
        launcher_path = 'src/structural_analysis/gemini_launcher.py'
        if os.path.exists(launcher_path):
            try:
                subprocess.run([sys.executable, launcher_path])
            except Exception as e:
                print(f"Error launching structural launcher: {e}")
        else:
            print(f"❌ {launcher_path} not found")
    
    def run_raw_data_browser(self):
        """Run raw data browser if available"""
        if os.path.exists('raw_data_browser.py'):
            try:
                subprocess.run([sys.executable, 'raw_data_browser.py'])
            except Exception as e:
                print(f"Error launching raw data browser: {e}")
        else:
            print("❌ raw_data_browser.py not found")
    
    def run_analytical_workflow(self):
        """Run analytical workflow"""
        workflow_path = 'src/numerical_analysis/analytical_workflow.py'
        if os.path.exists(workflow_path):
            try:
                subprocess.run([sys.executable, workflow_path])
            except Exception as e:
                print(f"Error launching analytical workflow: {e}")
        else:
            print(f"❌ {workflow_path} not found")
    
    def show_database_stats(self):
        """Show database statistics"""
        print("\n📊 DATABASE STATISTICS")
        print("=" * 30)
        
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                try:
                    df = pd.read_csv(db_file)
                    if 'full_name' in df.columns:
                        unique_gems = df['full_name'].nunique()
                        print(f"✅ {db_file}:")
                        print(f"   Records: {len(df):,}")
                        print(f"   Unique gems: {unique_gems}")
                        
                        # Show sample intensity ranges
                        intensity_range = f"{df['intensity'].min():.3f} to {df['intensity'].max():.3f}"
                        print(f"   Intensity range: {intensity_range}")
                    else:
                        print(f"⚠️ {db_file}: {len(df):,} records (no gem names)")
                except Exception as e:
                    print(f"❌ {db_file}: Error reading - {e}")
            else:
                print(f"❌ {db_file}: Missing")
        
        # Check structural database
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                count = cursor.execute("SELECT COUNT(*) FROM structural_features").fetchone()[0]
                print(f"\n✅ Structural database: {count:,} records")
                conn.close()
            except Exception as e:
                print(f"❌ Structural database error: {e}")
    
    def emergency_fix_files(self):
        """Run emergency fix if available"""
        if os.path.exists('emergency_fix.py'):
            try:
                subprocess.run([sys.executable, 'emergency_fix.py'])
            except Exception as e:
                print(f"Error running emergency fix: {e}")
        else:
            print("❌ emergency_fix.py not found")
    
    def main_menu(self):
        """Main menu system"""
        
        menu_options = [
            ("🔬 Launch Structural Analysis Hub", self.run_structural_analysis_hub),
            ("🎯 Launch Structural Analyzers", self.run_structural_launcher),
            ("📊 Analytical Analysis Workflow", self.run_analytical_workflow),
            ("💎 Select Gem for Analysis (FIXED)", self.select_and_analyze_gem),
            ("⚡ Direct Analysis (BYPASS PERMISSIONS)", self.direct_analysis_bypass),
            ("🔍 Run Debug Analysis (VALIDATION)", self.run_debug_analysis),
            ("📊 Visualize Individual Spectrum", self.visualize_individual_spectrum),
            ("🔬 Deep Normalization Debug", self.debug_exact_normalization),
            ("📂 Browse Raw Data Files", self.run_raw_data_browser),
            ("🧮 Run Fixed Numerical Analysis", self.run_numerical_analysis_fixed),
            ("📈 Show Database Statistics", self.show_database_stats),
            ("🔧 Emergency Fix", self.emergency_fix_files),
            ("❌ Exit", lambda: None)
        ]
        
        while True:
            print("\n" + "="*80)
            print("🔬 FIXED GEMINI GEMOLOGICAL ANALYSIS SYSTEM")
            print("="*80)
            
            # Show system status
            system_ok = self.check_system_status()
            
            print(f"\n📋 MAIN MENU (NORMALIZATION FIXED):")
            print("-" * 40)
            
            for i, (description, _) in enumerate(menu_options, 1):
                print(f"{i:2}. {description}")
            
            # Get user choice
            try:
                choice = input(f"\nChoice (1-{len(menu_options)}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == len(menu_options) - 1:  # Exit
                    print("\n👋 Goodbye!")
                    break
                
                if 0 <= choice_idx < len(menu_options) - 1:
                    description, action = menu_options[choice_idx]
                    print(f"\n🚀 {description.upper()}")
                    print("-" * 50)
                    
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
    """Main entry point"""
    try:
        print("🔬 Starting FIXED Gemini Gemological Analysis System...")
        system = FixedGeminiAnalysisSystem()
        system.main_menu()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted - goodbye!")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
