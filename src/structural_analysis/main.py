#!/usr/bin/env python3
"""
main.py - COMPLETE GEMINI GEMOLOGICAL ANALYSIS SYSTEM
Root directory main program with integrated gem selection
Save as: gemini_gemological_analysis/main.py
"""

import os
import sys
import subprocess
import sqlite3
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict

class GeminiAnalysisSystem:
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
    
    def check_system_status(self):
        """Check overall system status"""
        print("GEMINI GEMOLOGICAL ANALYSIS SYSTEM STATUS")
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
        """Complete gem selection and analysis workflow"""
        print("\n🎯 GEM SELECTION AND ANALYSIS")
        print("=" * 40)
        
        # Scan gems
        gems = self.scan_available_gems()
        if not gems:
            return
        
        # Show options
        complete_gems, partial_gems = self.show_available_gems(gems)
        
        if not complete_gems:
            print("\n❌ No complete gem sets found!")
            print("Need gems with B, L, and U files for analysis")
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
        
        # Auto-select files (first file of each type)
        selected = {}
        gem_files = gems[gem_choice]
        
        print(f"\n💎 SELECTING FILES FOR GEM {gem_choice}:")
        for light in ['B', 'L', 'U']:
            if gem_files[light]:
                selected[light] = gem_files[light][0]
                print(f"   {light}: {selected[light]}")
                
                # Show alternatives if available
                if len(gem_files[light]) > 1:
                    alternatives = gem_files[light][1:]
                    print(f"       (alternatives: {', '.join(alternatives)})")
        
        if len(selected) < 3:
            print(f"\n❌ Incomplete gem set - need B, L, and U files")
            return
        
        # Convert files
        print(f"\n🔄 PREPARING GEM {gem_choice} FOR ANALYSIS...")
        success = self.convert_gem_files(selected, gem_choice)
        
        if success:
            # Run analysis
            print(f"\n✅ GEM {gem_choice} READY FOR ANALYSIS")
            choice = input(f"Run numerical analysis now? (y/n): ").strip().lower()
            
            if choice == 'y':
                self.run_numerical_analysis()
        else:
            print(f"\n❌ Failed to prepare Gem {gem_choice}")
    
    def convert_gem_files(self, selected_files, gem_number):
        """Convert selected gem files for analysis"""
        try:
            # Clear and create raw_txt
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
            
            # Convert each file
            print("   🔧 Converting and normalizing...")
            
            for light, filename in selected_files.items():
                input_path = os.path.join('raw_txt', filename)
                output_path = f'data/unknown/unkgem{light}.csv'
                
                # Read file
                df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                wavelengths = np.array(df['wavelength'])
                intensities = np.array(df['intensity'])
                
                # Apply normalization
                if light == 'B':
                    # Halogen: 650nm → 50000
                    idx = np.argmin(np.abs(wavelengths - 650))
                    if intensities[idx] != 0:
                        normalized = intensities * (50000 / intensities[idx])
                    else:
                        normalized = intensities
                elif light == 'L':
                    # Laser: 450nm → 50000
                    idx = np.argmin(np.abs(wavelengths - 450))
                    if intensities[idx] != 0:
                        normalized = intensities * (50000 / intensities[idx])
                    else:
                        normalized = intensities
                elif light == 'U':
                    # UV: 811nm window → 15000
                    mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
                    window = intensities[mask]
                    if len(window) > 0 and window.max() > 0:
                        normalized = intensities * (15000 / window.max())
                    else:
                        normalized = intensities
                
                # Save normalized data
                output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
                output_df.to_csv(output_path, header=False, index=False)
                
                print(f"     ✅ {light}: {len(output_df)} points, range {normalized.min():.1f}-{normalized.max():.1f}")
            
            return True
            
        except Exception as e:
            print(f"     ❌ Conversion error: {e}")
            return False
    
    def run_numerical_analysis(self):
        """Run numerical analysis"""
        print(f"\n🚀 RUNNING NUMERICAL ANALYSIS...")
        
        try:
            # Try fast analysis first
            if os.path.exists('fast_gem_analysis.py'):
                print("   Using optimized fast analysis...")
                subprocess.run([sys.executable, 'fast_gem_analysis.py'])
            elif os.path.exists('src/numerical_analysis/gemini1.py'):
                print("   Using standard gemini1.py...")
                result = subprocess.run([sys.executable, 'src/numerical_analysis/gemini1.py'], 
                                      timeout=120, capture_output=True, text=True)
                if result.stdout:
                    print("Results:")
                    print(result.stdout[-1000:])  # Show last 1000 chars
            else:
                print("   ❌ No analysis program found")
                
        except subprocess.TimeoutExpired:
            print("   ⚠️ Analysis timed out - try fast_gem_analysis.py")
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
    
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
            ("💎 Select Gem for Analysis", self.select_and_analyze_gem),
            ("📂 Browse Raw Data Files", self.run_raw_data_browser),
            ("🧮 Run Numerical Analysis (current files)", self.run_numerical_analysis),
            ("📈 Show Database Statistics", self.show_database_stats),
            ("🔧 Emergency Fix", self.emergency_fix_files),
            ("❌ Exit", lambda: None)
        ]
        
        while True:
            print("\n" + "="*80)
            print("🔬 GEMINI GEMOLOGICAL ANALYSIS SYSTEM")
            print("="*80)
            
            # Show system status
            system_ok = self.check_system_status()
            
            print(f"\n📋 MAIN MENU:")
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
        print("🔬 Starting Gemini Gemological Analysis System...")
        system = GeminiAnalysisSystem()
        system.main_menu()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted - goodbye!")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
