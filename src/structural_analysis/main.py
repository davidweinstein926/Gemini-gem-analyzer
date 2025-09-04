#!/usr/bin/env python3
"""
Enhanced Gemstone Analysis System Hub - Main Menu
Supports both analytical analysis and database entry workflows
"""

import os
import sys
import subprocess
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime

class GeminiWorkflowManager:
    """Manages analytical analysis and database entry workflows"""
    
    def __init__(self):
        # Set up project paths
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.raw_txt_dir = self.data_dir / 'raw_txt'
        self.unknown_dir = self.data_dir / 'unknown'
        self.database_dir = self.project_root / 'database'
        self.reference_spectra_dir = self.database_dir / 'reference_spectra'
        self.src_dir = self.project_root / 'src'
        self.numerical_analysis_dir = self.src_dir / 'numerical_analysis'
        
        # Ensure directories exist
        self.raw_txt_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Project Root: {self.project_root}")
        print(f"Data Directory: {self.data_dir}")
    
    def show_raw_data_files(self):
        """Show available raw data files for selection"""
        if not self.raw_dir.exists():
            print("❌ Raw data directory not found")
            return []
        
        # Find all .txt files in raw directory
        txt_files = list(self.raw_dir.glob('*.txt'))
        
        if not txt_files:
            print("❌ No .txt files found in data/raw directory")
            return []
        
        print(f"\n📁 Available raw data files in {self.raw_dir}:")
        print("=" * 60)
        
        for i, file_path in enumerate(txt_files, 1):
            file_size = file_path.stat().st_size
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            print(f"{i:2d}. {file_path.name}")
            print(f"     Size: {file_size:,} bytes | Modified: {modified_time.strftime('%Y-%m-%d %H:%M')}")
        
        return txt_files
    
    def select_raw_files_for_analysis(self):
        """Interactive selection of raw files for analytical analysis"""
        print("\n🔍 RAW DATA SELECTION FOR ANALYTICAL ANALYSIS")
        print("=" * 60)
        print("Select up to 3 spectra files (B/H, L, U light sources)")
        
        txt_files = self.show_raw_data_files()
        if not txt_files:
            return None
        
        selected_files = {}
        light_sources = ['B', 'L', 'U']
        
        for light_source in light_sources:
            print(f"\n🔬 Select file for {light_source} light source (or press Enter to skip):")
            
            try:
                choice = input(f"Enter file number for {light_source} spectrum (1-{len(txt_files)}, or Enter to skip): ").strip()
                
                if not choice:
                    print(f"   Skipping {light_source} light source")
                    continue
                
                file_idx = int(choice) - 1
                if 0 <= file_idx < len(txt_files):
                    selected_file = txt_files[file_idx]
                    selected_files[light_source] = selected_file
                    print(f"   ✅ Selected for {light_source}: {selected_file.name}")
                else:
                    print(f"   ❌ Invalid selection for {light_source}")
            
            except (ValueError, KeyboardInterrupt):
                print(f"   ⚠️ Skipping {light_source} light source")
                continue
        
        if not selected_files:
            print("❌ No files selected")
            return None
        
        print(f"\n✅ Selected {len(selected_files)} files for analysis:")
        for ls, file_path in selected_files.items():
            print(f"   {ls}: {file_path.name}")
        
        return selected_files
    
    def copy_to_raw_txt(self, selected_files):
        """Copy selected files to data/raw_txt with appropriate names"""
        print(f"\n📋 COPYING FILES TO {self.raw_txt_dir}")
        print("-" * 40)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        copied_files = {}
        
        for light_source, source_file in selected_files.items():
            # Create destination filename with light source identifier
            dest_filename = f"spectrum_{timestamp}_{light_source}.txt"
            dest_path = self.raw_txt_dir / dest_filename
            
            try:
                shutil.copy2(source_file, dest_path)
                copied_files[light_source] = dest_path
                print(f"✅ {light_source}: {source_file.name} → {dest_filename}")
            except Exception as e:
                print(f"❌ Error copying {light_source} file: {e}")
        
        return copied_files
    
    def convert_txt_to_unkgem_csv(self, raw_txt_files):
        """Convert txt files to unkgem*.csv format in data/unknown"""
        print(f"\n🔄 CONVERTING TXT TO UNKGEM CSV FORMAT")
        print("-" * 40)
        
        converted_files = {}
        
        for light_source, txt_file in raw_txt_files.items():
            try:
                # Read txt file (assume wavelength, intensity format)
                df = pd.read_csv(txt_file, sep='\s+', header=None, names=['wavelength', 'intensity'])
                
                if len(df) == 0:
                    print(f"❌ {light_source}: Empty file")
                    continue
                
                # Create unkgem filename
                unkgem_file = self.unknown_dir / f'unkgem{light_source}.csv'
                
                # Save as CSV
                df.to_csv(unkgem_file, index=False, header=False)
                converted_files[light_source] = unkgem_file
                
                print(f"✅ {light_source}: {len(df)} data points → {unkgem_file.name}")
                print(f"   Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} nm")
                
            except Exception as e:
                print(f"❌ Error converting {light_source}: {e}")
        
        return converted_files
    
    def run_numerical_analysis(self):
        """Launch the numerical analysis (gemini1.py)"""
        print(f"\n🚀 LAUNCHING NUMERICAL ANALYSIS")
        print("-" * 40)
        
        gemini1_path = self.numerical_analysis_dir / 'gemini1.py'
        
        if not gemini1_path.exists():
            print(f"❌ gemini1.py not found at: {gemini1_path}")
            return False
        
        try:
            print(f"📊 Starting spectral analysis...")
            print(f"💻 Command: python {gemini1_path}")
            
            # Change to the numerical analysis directory
            os.chdir(self.numerical_analysis_dir)
            
            # Run gemini1.py
            result = subprocess.run([sys.executable, str(gemini1_path)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"✅ Numerical analysis completed successfully")
                return True
            else:
                print(f"❌ Numerical analysis failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Error launching numerical analysis: {e}")
            return False
        finally:
            # Change back to project root
            os.chdir(self.project_root)
    
    def check_database_files(self):
        """Check if database files exist"""
        db_files = {
            'B': self.reference_spectra_dir / 'gemini_db_long_B.csv',
            'L': self.reference_spectra_dir / 'gemini_db_long_L.csv',
            'U': self.reference_spectra_dir / 'gemini_db_long_U.csv'
        }
        
        existing_files = {}
        for ls, db_path in db_files.items():
            if db_path.exists():
                existing_files[ls] = db_path
        
        return existing_files
    
    def append_to_database(self):
        """Append unkgem*.csv files to corresponding database files"""
        print(f"\n📚 DATABASE APPEND OPERATION")
        print("=" * 60)
        
        # Check for unkgem files
        unkgem_files = {
            'B': self.unknown_dir / 'unkgemB.csv',
            'L': self.unknown_dir / 'unkgemL.csv',
            'U': self.unknown_dir / 'unkgemU.csv'
        }
        
        available_unkgem = {ls: path for ls, path in unkgem_files.items() if path.exists()}
        
        if not available_unkgem:
            print("❌ No unkgem*.csv files found in data/unknown")
            print("   Run analytical analysis first to generate these files")
            return False
        
        print(f"📁 Found unkgem files: {list(available_unkgem.keys())}")
        
        # Check database files
        db_files = self.check_database_files()
        
        if not db_files:
            print("❌ No database files found in database/reference_spectra")
            return False
        
        print(f"📚 Found database files: {list(db_files.keys())}")
        
        # Get gem name for database entry
        gem_name = input("\n💎 Enter gem name for database entry (e.g., 'ruby_natural_myanmar_001'): ").strip()
        
        if not gem_name:
            print("❌ No gem name provided")
            return False
        
        # Process each light source
        for light_source in available_unkgem.keys():
            if light_source not in db_files:
                print(f"⚠️ Skipping {light_source}: No corresponding database file")
                continue
            
            try:
                print(f"\n🔍 Processing {light_source} spectrum...")
                
                # Load unknown spectrum
                unkgem_path = available_unkgem[light_source]
                unknown_df = pd.read_csv(unkgem_path, header=None, names=['wavelength', 'intensity'])
                
                # Load existing database
                db_path = db_files[light_source]
                db_df = pd.read_csv(db_path)
                
                # Check for duplicates
                full_name = f"{gem_name}{light_source}"
                
                if 'full_name' in db_df.columns:
                    if full_name in db_df['full_name'].values:
                        print(f"   ⚠️ Duplicate found: {full_name} already exists in database")
                        overwrite = input(f"   Overwrite existing entry? (y/n): ").strip().lower()
                        
                        if overwrite != 'y':
                            print(f"   ⏭️ Skipping {light_source}")
                            continue
                        else:
                            # Remove existing entry
                            db_df = db_df[db_df['full_name'] != full_name]
                            print(f"   🗑️ Removed existing entry: {full_name}")
                
                # Prepare new entries
                new_entries = []
                for _, row in unknown_df.iterrows():
                    new_entry = {
                        'wavelength': row['wavelength'],
                        'intensity': row['intensity'],
                        'full_name': full_name
                    }
                    new_entries.append(new_entry)
                
                # Append to database
                new_df = pd.DataFrame(new_entries)
                updated_db = pd.concat([db_df, new_df], ignore_index=True)
                
                # Sort by wavelength within each gem
                updated_db = updated_db.sort_values(['full_name', 'wavelength'])
                
                # Save updated database
                backup_path = db_path.with_suffix('.backup.csv')
                shutil.copy2(db_path, backup_path)
                print(f"   💾 Backup created: {backup_path.name}")
                
                updated_db.to_csv(db_path, index=False)
                print(f"   ✅ Added {len(new_entries)} entries to {db_path.name}")
                print(f"   📊 Database now has {len(updated_db)} total entries")
                
            except Exception as e:
                print(f"   ❌ Error processing {light_source}: {e}")
        
        print(f"\n✅ Database append operation completed")
        return True
    
    def analytical_analysis_workflow(self):
        """Complete analytical analysis workflow"""
        print("\n🔬 ANALYTICAL ANALYSIS WORKFLOW")
        print("=" * 60)
        print("This workflow will:")
        print("1. Select raw data files from data/raw")
        print("2. Copy selected files to data/raw_txt")
        print("3. Convert to unkgem*.csv format in data/unknown")
        print("4. Run numerical analysis for best match comparison")
        print("5. Display top 10 matches with visual comparisons")
        
        # Step 1: Select files
        selected_files = self.select_raw_files_for_analysis()
        if not selected_files:
            return
        
        # Step 2: Copy to raw_txt
        raw_txt_files = self.copy_to_raw_txt(selected_files)
        if not raw_txt_files:
            print("❌ Failed to copy files to raw_txt")
            return
        
        # Step 3: Convert to unkgem CSV
        unkgem_files = self.convert_txt_to_unkgem_csv(raw_txt_files)
        if not unkgem_files:
            print("❌ Failed to convert files to unkgem format")
            return
        
        # Step 4: Run numerical analysis
        print(f"\n🎯 Ready to run numerical analysis with {len(unkgem_files)} light sources")
        proceed = input("Proceed with numerical analysis? (y/n): ").strip().lower()
        
        if proceed == 'y':
            self.run_numerical_analysis()
        else:
            print("⏸️ Analysis paused. unkgem files are ready in data/unknown")
    
    def database_entry_workflow(self):
        """Database entry workflow"""
        print("\n📚 DATABASE ENTRY WORKFLOW")
        print("=" * 60)
        print("This workflow will:")
        print("1. Check for existing unkgem*.csv files in data/unknown")
        print("2. Check for duplicate entries in database")
        print("3. Append new spectral data to reference database")
        
        success = self.append_to_database()
        if success:
            print("\n✅ Database entry workflow completed successfully")
        else:
            print("\n❌ Database entry workflow failed")

def show_main_menu():
    """Display the enhanced main menu"""
    print("\n" + "=" * 80)
    print("🔬 ENHANCED GEMSTONE ANALYSIS SYSTEM HUB")
    print("=" * 80)
    print()
    print("ANALYSIS WORKFLOWS:")
    print("1. Launch Structural Analyzers (manual + automated)")
    print("2. Import CSV Structural Data") 
    print("3. Analyze Unknown Stone (spectral matching)")
    print("4. Launch Standalone Peak Detector")
    print()
    print("NEW: RAW DATA PROCESSING:")
    print("5. 📊 Analytical Analysis Workflow")
    print("   → Select raw data → Convert → Run numerical analysis")
    print("6. 📚 Database Entry Workflow") 
    print("   → Add unkgem data to reference database")
    print()
    print("SYSTEM MANAGEMENT:")
    print("7. Show Database Statistics")
    print("8. Show Analysis Options Guide")
    print("9. System Status & Health Check")
    print("10. Exit")
    print()

def main():
    """Enhanced main function with new workflow options"""
    workflow_manager = GeminiWorkflowManager()
    
    while True:
        show_main_menu()
        
        try:
            choice = input("Choice (1-10): ").strip()
            
            if choice == '1':
                print("\n🚀 LAUNCHING STRUCTURAL ANALYZERS...")
                # Your existing launcher code here
                
            elif choice == '2':
                print("\n📥 IMPORTING CSV STRUCTURAL DATA...")
                # Your existing CSV import code here
                
            elif choice == '3':
                print("\n🔍 ANALYZING UNKNOWN STONE...")
                # Your existing analysis code here
                
            elif choice == '4':
                print("\n🎯 LAUNCHING PEAK DETECTOR...")
                # Your existing peak detector code here
                
            elif choice == '5':
                workflow_manager.analytical_analysis_workflow()
                
            elif choice == '6':
                workflow_manager.database_entry_workflow()
                
            elif choice == '7':
                print("\n📊 DATABASE STATISTICS...")
                # Your existing database stats code here
                
            elif choice == '8':
                print("\n📖 ANALYSIS OPTIONS GUIDE...")
                # Your existing guide code here
                
            elif choice == '9':
                print("\n🏥 SYSTEM STATUS & HEALTH CHECK...")
                # Your existing health check code here
                
            elif choice == '10':
                print("\n👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-10.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
