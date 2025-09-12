#!/usr/bin/env python3
"""
SIMPLIFIED UNKNOWN STONE ANALYZER
Works with existing structural data files - no complex pipeline needed

Author: David
Version: 2024.08.18
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

class SimplifiedUnknownAnalyzer:
    """Simplified analyzer that works with existing structural CSV files"""
    
    def __init__(self, db_path="multi_structural_gem_data.db"):
        self.db_path = db_path
        self.structural_data_dir = r"C:\users\david\gemini sp10 structural data"
        
        # Matching weights
        self.light_source_weights = {
            'Halogen': 1.0,    # Most reliable
            'Laser': 0.9,      # High resolution  
            'UV': 0.8          # Specialized
        }
        
        print(f"🔬 Simplified Unknown Stone Analyzer")
        print(f"   📊 Database: {self.db_path}")
        print(f"   📁 Structural data: {self.structural_data_dir}")
    
    def browse_structural_data(self):
        """Browse and display available structural CSV files"""
        
        print(f"\n📁 BROWSING STRUCTURAL DATA DIRECTORY")
        print("="*60)
        
        if not os.path.exists(self.structural_data_dir):
            print(f"❌ Structural data directory not found: {self.structural_data_dir}")
            return {}
        
        # Check subdirectories
        subdirs = ['halogen', 'laser', 'uv']
        all_files = {}
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.structural_data_dir, subdir)
            if os.path.exists(subdir_path):
                csv_files = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
                if csv_files:
                    all_files[subdir] = [(f, subdir_path) for f in csv_files]
                    print(f"📂 {subdir.upper()}: {len(csv_files)} CSV files")
                else:
                    print(f"📂 {subdir.upper()}: No CSV files")
            else:
                print(f"📂 {subdir.upper()}: Directory not found")
        
        if not any(all_files.values()):
            print("❌ No CSV files found in any subdirectory")
            return {}
        
        # Organize by stone ID
        stone_groups = defaultdict(lambda: {'halogen': [], 'laser': [], 'uv': []})
        
        for light_source, files_and_paths in all_files.items():
            for filename, filepath in files_and_paths:
                # Extract stone ID from filename
                stone_id = self.extract_stone_id(filename)
                stone_groups[stone_id][light_source].append((filename, filepath))
        
        # Display organized files
        print(f"\n🔍 AVAILABLE STONES AND FILES:")
        print("-"*60)
        
        stone_list = []
        for i, (stone_id, files) in enumerate(sorted(stone_groups.items()), 1):
            available_lights = []
            for light in ['halogen', 'laser', 'uv']:
                if files[light]:
                    available_lights.append(light[0].upper())  # H, L, U
            
            lights_str = '/'.join(available_lights) if available_lights else 'None'
            
            print(f"{i:2d}. {stone_id:<15} | Available: {lights_str}")
            
            # Show individual files
            for light in ['halogen', 'laser', 'uv']:
                if files[light]:
                    for filename, filepath in files[light]:
                        size_kb = os.path.getsize(os.path.join(filepath, filename)) // 1024
                        print(f"    📄 {light[0].upper()}: {filename:<30} ({size_kb} KB)")
            
            stone_groups[stone_id]['index'] = i
            stone_list.append((stone_id, files))
            print()
        
        return stone_list
    
    def extract_stone_id(self, filename):
        """Extract stone ID from filename"""
        # Remove common suffixes
        base = filename.replace('.csv', '')
        base = base.replace('_structural', '').replace('_peaks', '').replace('_auto', '').replace('_manual', '')
        
        # Try to remove light source suffixes
        for suffix in ['BC1', 'LC1', 'UC1', 'B1', 'L1', 'U1', 'H1']:
            if base.endswith(suffix):
                return base[:-len(suffix)]
        
        return base
    
    def select_files_for_analysis(self, stone_list):
        """Interactive file selection for unknown analysis"""
        
        if not stone_list:
            print("❌ No stones available")
            return None
        
        print(f"📝 SELECT FILES FOR UNKNOWN ANALYSIS:")
        print("-"*50)
        print("Choose up to 3 files (Halogen, Laser, UV)")
        print("You can select by stone number or mix files from different stones")
        print()
        
        selected_files = {}
        
        while len(selected_files) < 3:
            remaining = 3 - len(selected_files)
            print(f"🎯 Select file {len(selected_files) + 1}/3 (or Enter to finish)")
            
            # Show current selection
            if selected_files:
                print("   Current selection:")
                for light, (file, path) in selected_files.items():
                    print(f"      {light}: {file}")
                print()
            
            choice = input("Enter stone number or 'q' to finish: ").strip()
            
            if choice.lower() == 'q' or choice == '':
                break
            
            try:
                stone_index = int(choice)
                if 1 <= stone_index <= len(stone_list):
                    stone_id, files = stone_list[stone_index - 1]
                    
                    # Show available files for this stone
                    print(f"\n📋 Available files for {stone_id}:")
                    available_options = []
                    
                    for light in ['halogen', 'laser', 'uv']:
                        if files[light]:
                            light_name = {'halogen': 'Halogen', 'laser': 'Laser', 'uv': 'UV'}[light]
                            if light_name not in selected_files:
                                for j, (filename, filepath) in enumerate(files[light]):
                                    option_id = f"{light[0].upper()}{j+1}"
                                    available_options.append((option_id, light_name, filename, filepath))
                                    print(f"      {option_id}: {light_name} - {filename}")
                    
                    if not available_options:
                        print("   ❌ No available files (all light sources already selected)")
                        continue
                    
                    file_choice = input("   Select file (H1, L1, U1, etc.): ").strip().upper()
                    
                    # Find matching option
                    selected_option = None
                    for opt_id, light_name, filename, filepath in available_options:
                        if opt_id == file_choice:
                            selected_option = (light_name, filename, filepath)
                            break
                    
                    if selected_option:
                        light_name, filename, filepath = selected_option
                        full_path = os.path.join(filepath, filename)
                        selected_files[light_name] = (filename, full_path)
                        print(f"   ✅ Selected {light_name}: {filename}")
                    else:
                        print("   ❌ Invalid file choice")
                else:
                    print("❌ Invalid stone number")
            except ValueError:
                print("❌ Please enter a number")
        
        if not selected_files:
            print("❌ No files selected")
            return None
        
        print(f"\n✅ FINAL SELECTION:")
        for light, (filename, filepath) in selected_files.items():
            print(f"   {light}: {filename}")
        
        return selected_files
    
    def calculate_stone_similarity(self, unknown_files, db_file):
        """Calculate similarity between unknown files and database stone"""
        
        try:
            # Load database features
            conn = sqlite3.connect(self.db_path)
            db_query = """
                SELECT * FROM structural_features 
                WHERE file = ? AND processing LIKE '%Normalized%'
            """
            db_df = pd.read_sql_query(db_query, conn, params=[db_file])
            conn.close()
            
            if db_df.empty:
                return 0.0, "No database features"
            
            # Calculate similarity for each light source
            light_scores = {}
            total_weighted_score = 0.0
            total_weight = 0.0
            
            for light_name, (filename, filepath) in unknown_files.items():
                # Load unknown features
                try:
                    unknown_df = pd.read_csv(filepath)
                    
                    # Filter database features for this light source
                    light_source_map = {'Halogen': 'Halogen', 'Laser': 'Laser', 'UV': 'UV'}
                    db_light = light_source_map[light_name]
                    db_light_df = db_df[db_df['light_source'] == db_light]
                    
                    if db_light_df.empty:
                        continue
                    
                    # Calculate similarity
                    similarity = self.calculate_light_similarity(unknown_df, db_light_df)
                    light_scores[light_name] = similarity
                    
                    # Weight the score
                    weight = self.light_source_weights.get(light_name, 1.0)
                    total_weighted_score += similarity * weight
                    total_weight += weight
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing {light_name}: {e}")
                    continue
            
            overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
            
            return overall_score, {
                'overall': overall_score,
                'light_scores': light_scores,
                'common_lights': list(light_scores.keys())
            }
            
        except Exception as e:
            return 0.0, f"Error: {e}"
    
    def calculate_light_similarity(self, unknown_df, db_df):
        """Calculate similarity for a specific light source"""
        
        try:
            # Simple wavelength-based comparison
            if 'wavelength' not in unknown_df.columns or 'wavelength' not in db_df.columns:
                return 0.0
            
            unknown_wavelengths = unknown_df['wavelength'].dropna().values
            db_wavelengths = db_df['wavelength'].dropna().values
            
            if len(unknown_wavelengths) == 0 or len(db_wavelengths) == 0:
                return 0.0
            
            # For each unknown wavelength, find best match in database
            total_score = 0.0
            
            for unknown_wl in unknown_wavelengths:
                best_score = 0.0
                for db_wl in db_wavelengths:
                    diff = abs(unknown_wl - db_wl)
                    
                    # Similarity scoring
                    if diff <= 1.0:
                        score = 1.0
                    elif diff <= 2.0:
                        score = 0.9
                    elif diff <= 5.0:
                        score = 0.8 - (diff - 2.0) * 0.1
                    else:
                        score = max(0.0, 0.5 - (diff - 5.0) * 0.05)
                    
                    best_score = max(best_score, score)
                
                total_score += best_score
            
            return total_score / len(unknown_wavelengths)
            
        except Exception as e:
            return 0.0
    
    def find_database_matches(self, unknown_files):
        """Find best database matches"""
        
        print(f"\n🎯 FINDING DATABASE MATCHES")
        print("="*50)
        
        try:
            # Get all database stones
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT DISTINCT file, 
                       GROUP_CONCAT(DISTINCT light_source) as light_sources
                FROM structural_features 
                WHERE processing LIKE '%Normalized%'
                GROUP BY file
                ORDER BY file
            """
            db_stones_df = pd.read_sql_query(query, conn)
            conn.close()
            
            if db_stones_df.empty:
                print("❌ No database stones found")
                return []
            
            print(f"📊 Comparing against {len(db_stones_df)} database stones...")
            
            matches = []
            unknown_lights = set(unknown_files.keys())
            
            for idx, row in db_stones_df.iterrows():
                db_file = row['file']
                db_lights_str = row['light_sources']
                db_lights = set(db_lights_str.split(',')) if db_lights_str else set()
                
                # Calculate match score
                score, details = self.calculate_stone_similarity(unknown_files, db_file)
                
                if score > 0:
                    if isinstance(details, dict):
                        matches.append({
                            'file': db_file,
                            'overall_score': score,
                            'light_scores': details['light_scores'],
                            'common_lights': details['common_lights'],
                            'db_lights': list(db_lights),
                            'max_light_score': max(details['light_scores'].values()) if details['light_scores'] else 0
                        })
                    else:
                        matches.append({
                            'file': db_file,
                            'overall_score': score,
                            'light_scores': {},
                            'common_lights': [],
                            'db_lights': list(db_lights),
                            'max_light_score': 0
                        })
                
                # Progress
                if (idx + 1) % 20 == 0:
                    print(f"   Progress: {idx + 1}/{len(db_stones_df)}...")
            
            # Sort with tie-breaking rule: highest overall, then lowest max individual
            matches.sort(key=lambda x: (-x['overall_score'], x['max_light_score']))
            
            print(f"✅ Found {len(matches)} matches")
            return matches
            
        except Exception as e:
            print(f"❌ Error finding matches: {e}")
            return []
    
    def display_results(self, matches, unknown_files):
        """Display match results"""
        
        if not matches:
            print("❌ No matches found")
            return
        
        unknown_lights = list(unknown_files.keys())
        
        print(f"\n🏆 TOP MATCHES")
        print(f"   Unknown files: {[f[0] for f in unknown_files.values()]}")
        print(f"   Light sources: {unknown_lights}")
        print("="*80)
        print(f"{'Rank':<4} {'Database Stone':<25} {'Overall':<8} {'H':<6} {'L':<6} {'U':<6} {'Quality'}")
        print("-"*80)
        
        for i, match in enumerate(matches[:15], 1):
            file_name = match['file']
            overall = match['overall_score']
            light_scores = match['light_scores']
            
            # Extract individual scores
            h_score = light_scores.get('Halogen', 0.0)
            l_score = light_scores.get('Laser', 0.0)
            u_score = light_scores.get('UV', 0.0)
            
            # Quality indicator
            if overall > 0.9:
                indicator = "🟢 EXCELLENT"
            elif overall > 0.8:
                indicator = "🟡 VERY GOOD"
            elif overall > 0.7:
                indicator = "🟠 GOOD"
            elif overall > 0.5:
                indicator = "🔵 FAIR"
            else:
                indicator = "🔴 POOR"
            
            print(f"{i:<4} {file_name:<25} {overall:<8.3f} {h_score:<6.3f} {l_score:<6.3f} {u_score:<6.3f} {indicator}")
        
        # Show detailed analysis for top match
        if matches:
            top_match = matches[0]
            print(f"\n🔬 TOP MATCH DETAILS:")
            print(f"   📄 Database file: {top_match['file']}")
            print(f"   🎯 Overall score: {top_match['overall_score']:.3f}")
            print(f"   💡 Common lights: {top_match['common_lights']}")
            print(f"   📊 Individual scores:")
            for light, score in top_match['light_scores'].items():
                print(f"      {light}: {score:.3f}")
    
    def analyze_unknown(self):
        """Main analysis workflow"""
        
        print(f"\n🔬 UNKNOWN STONE ANALYSIS")
        print("="*50)
        
        # Browse available files
        stone_list = self.browse_structural_data()
        if not stone_list:
            return
        
        # Select files for analysis
        unknown_files = self.select_files_for_analysis(stone_list)
        if not unknown_files:
            return
        
        # Find database matches
        matches = self.find_database_matches(unknown_files)
        
        # Display results
        self.display_results(matches, unknown_files)
        
        # TODO: Add visualization option here
        if matches:
            print(f"\n📊 Future enhancement: Visual comparison graphs")
    
    def interactive_menu(self):
        """Interactive menu"""
        
        while True:
            print(f"\n🔬 SIMPLIFIED UNKNOWN STONE ANALYZER")
            print("="*50)
            print("1. 🎯 Analyze Unknown Stone")
            print("2. 📊 Database Statistics")
            print("3. 📁 Browse Structural Data")
            print("4. ❌ Exit")
            
            choice = input("\nChoice (1-4): ").strip()
            
            if choice == "4":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                self.analyze_unknown()
                input("\nPress Enter to continue...")
            elif choice == "2":
                self.show_database_stats()
                input("\nPress Enter to continue...")
            elif choice == "3":
                self.browse_structural_data()
                input("\nPress Enter to continue...")
            else:
                print("❌ Invalid choice")
    
    def show_database_stats(self):
        """Show database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features WHERE processing LIKE '%Normalized%'")
            stones = cursor.fetchone()[0]
            
            cursor.execute("SELECT light_source, COUNT(DISTINCT file) FROM structural_features WHERE processing LIKE '%Normalized%' GROUP BY light_source")
            by_light = cursor.fetchall()
            
            conn.close()
            
            print(f"\n📊 DATABASE STATISTICS:")
            print(f"   🔬 Stones with normalized data: {stones}")
            print(f"   💡 By light source:")
            for light, count in by_light:
                icon = {'Halogen': '🔥', 'Laser': '⚡', 'UV': '🟣'}.get(light, '💡')
                print(f"      {icon} {light}: {count} stones")
                
        except Exception as e:
            print(f"❌ Error reading database: {e}")

def main():
    """Main entry point"""
    
    print("🔬 SIMPLIFIED UNKNOWN STONE ANALYZER")
    print("Works with existing structural CSV files")
    print("="*50)
    
    # Check database
    db_path = "multi_structural_gem_data.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return
    
    # Check structural data directory
    structural_dir = r"C:\users\david\gemini sp10 structural data"
    if not os.path.exists(structural_dir):
        print(f"❌ Structural data directory not found: {structural_dir}")
        print("💡 Create this directory and organize CSV files in halogen/, laser/, uv/ subfolders")
        return
    
    # Initialize analyzer
    analyzer = SimplifiedUnknownAnalyzer(db_path)
    analyzer.interactive_menu()

if __name__ == "__main__":
    main() 