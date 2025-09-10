#!/usr/bin/env python3
"""
COMPLETE FIXED MAIN.PY - GEMINI GEMOLOGICAL ANALYSIS SYSTEM
All indentation errors corrected, full file selection implemented
Enhanced visualization system with CONSOLIDATED SINGLE LEGEND
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
        """Create enhanced spectral comparison plots - focused and detailed with SINGLE CONSOLIDATED LEGEND"""
        print(f"\n📊 CREATING ENHANCED SPECTRAL PLOTS FOR {gem_identifier}")
        print("=" * 60)
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
        except ImportError:
            print("❌ matplotlib not available - cannot create plots")
            return
        
        if not hasattr(self, 'current_analysis_results') or not self.current_analysis_results:
            print("❌ No analysis results available - run analysis first")
            return
        
        # Get top 5 matches
        top_matches = self.current_analysis_results[:5]
        
        # Create enhanced layout: 3 columns x 3 rows (normalized + scaled + distribution)
        # Make spectral plots larger than distribution
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.2)
        
        fig.suptitle(f'Enhanced Spectral Analysis: {gem_identifier} vs Top Matches', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Collect all legend elements for consolidated legend
        legend_elements = []
        
        # Process each light source
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
                
                # ROW 1: Normalized Spectra (Prominent) - NO INDIVIDUAL LEGEND
                ax1 = fig.add_subplot(gs[0, light_idx])
                
                # Plot unknown (thick black line)
                unknown_line = ax1.plot(unknown['wavelength'], unknown['intensity'], 
                        'black', linewidth=0.5, alpha=0.9, zorder=10)
                
                # Add to consolidated legend only once (from first light source)
                if light_idx == 0:
                    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=0.5, 
                                                    label=f'{gem_identifier} (Unknown)', alpha=0.9))
                
                # Plot top matches with scores
                match_lines = []
                for i, (match_id, total_score) in enumerate(top_matches[:3]):
                    match_entries = db[db['full_name'].str.startswith(match_id)]
                    if not match_entries.empty:
                        # Get best matching file for this light
                        match_file = f"{match_id}{light}C1"
                        if match_file not in match_entries['full_name'].values:
                            match_file = match_entries['full_name'].iloc[0]
                        
                        match_data = db[db['full_name'] == match_file]
                        if not match_data.empty:
                            match_line = ax1.plot(match_data['wavelength'], match_data['intensity'], 
                                    color=colors[i], linewidth=0.5, alpha=0.8)
                            
                            # Add to consolidated legend only once (from first light source)
                            if light_idx == 0:
                                legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=0.5,
                                                        label=f'#{i+1}: {match_id} (Score: {total_score:.3f})', alpha=0.8))
                
                ax1.set_title(f'{light} Light - Normalized Spectra', fontsize=14, fontweight='bold')
                #ax1.set_xlabel('Wavelength (nm)', fontsize=12)
                ax1.set_ylabel('Normalized Intensity', fontsize=12)
                ax1.grid(True, alpha=0.3)
                
                # ROW 2: 0-100 Scaled Spectra (Analysis Method) - NO INDIVIDUAL LEGEND
                ax2 = fig.add_subplot(gs[1, light_idx])
                
                # Apply 0-100 scaling to unknown
                unknown_scaled = unknown.copy()
                unknown_scaled['intensity'] = self.apply_0_100_scaling(unknown['wavelength'].values, unknown['intensity'].values)
                
                # Plot scaled unknown
                ax2.plot(unknown_scaled['wavelength'], unknown_scaled['intensity'], 
                        'black', linewidth=0.5, alpha=0.9, zorder=10)
                
                # Plot scaled matches
                for i, (match_id, total_score) in enumerate(top_matches[:3]):
                    match_entries = db[db['full_name'].str.startswith(match_id)]
                    if not match_entries.empty:
                        match_file = f"{match_id}{light}C1"
                        if match_file not in match_entries['full_name'].values:
                            match_file = match_entries['full_name'].iloc[0]
                        
                        match_data = db[db['full_name'] == match_file]
                        if not match_data.empty:
                            match_scaled = match_data.copy()
                            match_scaled['intensity'] = self.apply_0_100_scaling(match_data['wavelength'].values, match_data['intensity'].values)
                            ax2.plot(match_scaled['wavelength'], match_scaled['intensity'], 
                                    color=colors[i], linewidth=0.5, alpha=0.8)
                
                ax2.set_title(f'{light} Light - 0-100 Scaled', fontsize=14, fontweight='bold')
                # Wavelength label removed as requested
                ax2.set_ylabel('Scaled Intensity (0-100)', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)  # Fixed scale for consistency
                
            except Exception as e:
                print(f"❌ Error creating {light} light plots: {e}")
                # Fill with error message
                ax1 = fig.add_subplot(gs[0, light_idx])
                ax2 = fig.add_subplot(gs[1, light_idx])
                ax1.text(0.5, 0.5, f'Error loading {light} data: {e}', ha='center', va='center', transform=ax1.transAxes, fontsize=12, color='red')
                ax2.text(0.5, 0.5, f'Error loading {light} data', ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='red')
        
        # ROW 3: Score Distribution (spans all columns for prominence)
        ax_dist = fig.add_subplot(gs[2, :])
        
        # Create comprehensive score distribution
        all_scores = [score for _, score in self.current_analysis_results]
        
        # Create histogram with better binning
        counts, bins, patches = ax_dist.hist(all_scores, bins=50, alpha=0.7, color='lightblue', edgecolor='black', linewidth=0.5)
        
        # Highlight top matches with vertical lines
        dist_legend_elements = []
        for i, (match_id, score) in enumerate(top_matches[:5]):
            color = colors[i] if i < len(colors) else 'gray'
            ax_dist.axvline(x=score, color=color, linestyle='--', linewidth=2, alpha=0.8)
            # Add to distribution-specific legend
            dist_legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2,
                                                 label=f'#{i+1}: {match_id} ({score:.4f})', alpha=0.8))
        
        # Add statistics
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        ax_dist.axvline(x=mean_score, color='orange', linestyle=':', linewidth=2, alpha=0.6)
        ax_dist.axvline(x=median_score, color='purple', linestyle=':', linewidth=2, alpha=0.6)
        
        # Add statistics to distribution legend
        dist_legend_elements.extend([
            plt.Line2D([0], [0], color='orange', linestyle=':', linewidth=2, alpha=0.6, label=f'Mean: {mean_score:.4f}'),
            plt.Line2D([0], [0], color='purple', linestyle=':', linewidth=2, alpha=0.6, label=f'Median: {median_score:.4f}')
        ])
        
        ax_dist.set_title('Score Distribution - All Database Matches', fontsize=14, fontweight='bold')
        ax_dist.set_xlabel('Log Score (Lower = Better Match)', fontsize=12)
        ax_dist.set_ylabel('Number of Gems', fontsize=12)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.set_yscale('log')  # Log scale for better visualization
        
        # Distribution-specific legend (positioned on the right of distribution plot)
        ax_dist.legend(handles=dist_legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        # Add summary text box
        #perfect_matches = sum(1 for s in all_scores if s < 1e-10)
        #excellent_matches = sum(1 for s in all_scores if s < 1e-6)
        #good_matches = sum(1 for s in all_scores if s < 1e-3)
        
        #summary_text = f'Analysis Summary:\n'
        #summary_text += f'Total gems: {len(all_scores)}\n'
        #summary_text += f'Perfect matches (<1e-10): {perfect_matches}\n'
        #summary_text += f'Excellent matches (<1e-6): {excellent_matches}\n'
        #summary_text += f'Good matches (<1e-3): {good_matches}'
        
        #ax_dist.text(0.02, 0.98, summary_text, transform=ax_dist.transAxes, fontsize=10,
                    #verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # CREATE SINGLE CONSOLIDATED LEGEND FOR SPECTRAL PLOTS
        # Position it prominently in the upper right area of the entire figure
        consolidated_legend = fig.legend(handles=legend_elements, 
                                       loc='upper right', 
                                       bbox_to_anchor=(0.98, 0.88),
                                       fontsize=11,
                                       title='Spectral Plot Legend',
                                       title_fontsize=12,
                                       frameon=True,
                                       fancybox=True,
                                       shadow=True,
                                       framealpha=0.9)
        
        # Style the consolidated legend
        consolidated_legend.get_frame().set_facecolor('white')
        consolidated_legend.get_frame().set_edgecolor('black')
        consolidated_legend.get_frame().set_linewidth(1)
        
        # Make layout tight and save
        plt.tight_layout()
        
        # Adjust layout to accommodate the consolidated legend
        plt.subplots_adjust(right=0.85)
        
        # Save the plot with high DPI for detailed viewing
        plot_filename = f'enhanced_spectral_analysis_{gem_identifier}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"💾 Enhanced plot saved as: {plot_filename}")
        
        # Add interactive features for enlargement
        def on_click(event):
            """Handle click events for plot enlargement"""
            if event.inaxes in [fig.get_axes()[0], fig.get_axes()[1], fig.get_axes()[2], fig.get_axes()[3], fig.get_axes()[4], fig.get_axes()[5]]:
                # Create enlarged view of clicked spectral plot
                enlarged_fig, enlarged_ax = plt.subplots(1, 1, figsize=(16, 10))
                
                # Determine which plot was clicked and recreate it enlarged
                ax_index = fig.get_axes().index(event.inaxes)
                light_names = ['B', 'L', 'U']
                
                if ax_index < 3:  # Normalized plots
                    light = light_names[ax_index]
                    enlarged_ax.set_title(f'ENLARGED: {light} Light - Normalized Spectra', fontsize=16, fontweight='bold')
                else:  # Scaled plots
                    light = light_names[ax_index - 3]
                    enlarged_ax.set_title(f'ENLARGED: {light} Light - 0-100 Scaled', fontsize=16, fontweight='bold')
                
                # Copy the data from original plot to enlarged plot
                for line in event.inaxes.get_lines():
                    enlarged_ax.plot(line.get_xdata(), line.get_ydata(), 
                                   color=line.get_color(), linewidth=line.get_linewidth()*1.5, 
                                   alpha=line.get_alpha())
                
                # Add the consolidated legend to the enlarged plot
                enlarged_ax.legend(handles=legend_elements, fontsize=12, loc='upper right')
                
                enlarged_ax.set_xlabel('Wavelength (nm)', fontsize=14)
                enlarged_ax.set_ylabel(event.inaxes.get_ylabel(), fontsize=14)
                enlarged_ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
        
        # Connect the click handler
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        try:
            plt.show()
            print("\n💡 TIP: Click on any spectral plot to see enlarged view!")
            print("📋 LEGEND: Single consolidated legend shows all matches (upper right)")
            print("📊 DISTRIBUTION: Separate legend shows score markers (right side)")
        except:
            print("⚠️ Cannot display plot interactively, but file saved successfully")
        
        # Create enhanced summary report
        self.create_enhanced_analysis_report(gem_identifier, top_matches)
    
    def create_enhanced_analysis_report(self, gem_identifier, top_matches):
        """Create enhanced text summary report with detailed analysis"""
        report_filename = f'enhanced_report_{gem_identifier}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt'
        
        with open(report_filename, 'w') as f:
            f.write(f"ENHANCED GEMINI SPECTRAL ANALYSIS REPORT\n")
            f.write(f"========================================\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Unknown Gem: {gem_identifier}\n")
            f.write(f"Analysis Method: Normalized + 0-100 Scaled Comparison\n")
            f.write(f"Visualization: Enhanced 3-panel layout with consolidated single legend\n\n")
            
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
                
                # Enhanced quality assessment
                if score < 1e-10:
                    f.write(f"         Quality: ★★★★★ PERFECT MATCH (Identical spectra)\n")
                    f.write(f"         Confidence: ABSOLUTE (Self-match or identical gem)\n")
                elif score < 1e-6:
                    f.write(f"         Quality: ★★★★☆ EXCELLENT MATCH (Near identical)\n")
                    f.write(f"         Confidence: VERY HIGH (Same gem type/treatment)\n")
                elif score < 1e-3:
                    f.write(f"         Quality: ★★★☆☆ GOOD MATCH (Very similar)\n")
                    f.write(f"         Confidence: HIGH (Same species, possible var. difference)\n")
                elif score < 1:
                    f.write(f"         Quality: ★★☆☆☆ MODERATE MATCH (Some similarity)\n")
                    f.write(f"         Confidence: MEDIUM (Related gem family)\n")
                else:
                    f.write(f"         Quality: ★☆☆☆☆ POOR MATCH (Different spectra)\n")
                    f.write(f"         Confidence: LOW (Different gem type)\n")
                f.write(f"\n")
            
            # Enhanced statistical summary
            all_scores = [score for _, score in self.current_analysis_results]
            f.write(f"\nENHANCED STATISTICAL SUMMARY:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"Total gems analyzed: {len(self.current_analysis_results)}\n")
            f.write(f"Best match score: {min(all_scores):.6f}\n")
            f.write(f"Second best score: {sorted(all_scores)[1]:.6f}\n")
            f.write(f"Score gap (confidence): {sorted(all_scores)[1] - min(all_scores):.6f}\n")
            f.write(f"Median score: {np.median(all_scores):.6f}\n")
            f.write(f"Average score: {np.mean(all_scores):.6f}\n")
            f.write(f"Standard deviation: {np.std(all_scores):.6f}\n")
            
            # Match quality distribution
            perfect_matches = sum(1 for s in all_scores if s < 1e-10)
            excellent_matches = sum(1 for s in all_scores if s < 1e-6)
            good_matches = sum(1 for s in all_scores if s < 1e-3)
            moderate_matches = sum(1 for s in all_scores if s < 1)
            
            f.write(f"\nMATCH QUALITY DISTRIBUTION:\n")
            f.write(f"Perfect matches (< 1e-10): {perfect_matches}\n")
            f.write(f"Excellent matches (< 1e-6): {excellent_matches}\n")
            f.write(f"Good matches (< 1e-3): {good_matches}\n")
            f.write(f"Moderate matches (< 1): {moderate_matches}\n")
            f.write(f"Poor matches (>= 1): {len(all_scores) - moderate_matches}\n")
            
            f.write(f"\nVISUALIZATION ENHANCEMENTS:\n")
            f.write(f"-" * 30 + "\n")
            f.write(f"Plot Layout: 3x3 grid (Normalized + Scaled + Distribution)\n")
            f.write(f"Spectral Plot Prominence: 2:1 height ratio over distribution\n")
            f.write(f"Line Thickness: 0.5 for matches, 2.5 for unknown (enhanced)\n")
            f.write(f"Legend System: Single consolidated legend for spectral plots\n")
            f.write(f"Interactive Features: Click-to-enlarge for detailed viewing\n")
            f.write(f"Score Integration: Match scores shown in consolidated legend\n")
            f.write(f"Distribution Analysis: Log-scale with statistical markers\n")
            f.write(f"Plot Resolution: 200 DPI for high-quality detailed viewing\n")
            
            f.write(f"\nANALYSIS PARAMETERS:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Normalization: B(650nm→50K), L(Max→50K), U(811nm→15K)\n")
            f.write(f"Comparison Method: Hybrid (normalized storage + 0-100 scaled analysis)\n")
            f.write(f"Scoring Algorithm: MSE with log transformation for score distribution\n")
            f.write(f"Database Coverage: B+L+U multi-spectral complete analysis\n")
        
        print(f"📄 Enhanced analysis report saved as: {report_filename}")

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
            ("💎 Select Gem for Analysis (ENHANCED)", self.select_and_analyze_gem),
            ("📂 Browse Raw Data Files", self.run_raw_data_browser),
            ("🧮 Run Fixed Numerical Analysis", self.run_numerical_analysis_fixed),
            ("📈 Show Database Statistics", self.show_database_stats),
            ("🔧 Emergency Fix", self.emergency_fix_files),
            ("❌ Exit", lambda: None)
        ]
        
        while True:
            print("\n" + "="*80)
            print("🔬 ENHANCED GEMINI GEMOLOGICAL ANALYSIS SYSTEM")
            print("="*80)
            
            # Show system status
            system_ok = self.check_system_status()
            
            print(f"\n📋 MAIN MENU (CONSOLIDATED LEGEND VISUALIZATION):")
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
        print("🔬 Starting ENHANCED Gemini Gemological Analysis System...")
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
