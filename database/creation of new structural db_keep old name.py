#!/usr/bin/env python3
"""
NEW DATABASE SYSTEM - Perfect schema for normalized structural data
"""

import sqlite3
import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

class NewStructuralDatabaseManager:
    def __init__(self, db_path="multi_structural_gem_data.db"):
        self.db_path = db_path
        self.init_perfect_database()
    
    def backup_old_database(self):
        """Rename old database to _OLD if it exists"""
        if os.path.exists(self.db_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_name = f"multi_structural_gem_data_OLD_{timestamp}.db"
            os.rename(self.db_path, old_name)
            print(f"✅ Old database renamed to: {old_name}")
            return old_name
        return None
    
    def init_perfect_database(self):
        """Create perfect database schema for new normalized data"""
        # Backup old database first
        self.backup_old_database()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Perfect schema for new data format
            cursor.execute("""
                CREATE TABLE structural_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature TEXT NOT NULL,
                    file TEXT NOT NULL,
                    light_source TEXT NOT NULL,
                    wavelength REAL NOT NULL,
                    intensity REAL NOT NULL,
                    point_type TEXT NOT NULL,
                    feature_group TEXT NOT NULL,
                    processing TEXT,
                    baseline_used REAL,
                    norm_factor REAL,
                    snr REAL,
                    symmetry_ratio REAL,
                    skew_description TEXT,
                    width_nm REAL,
                    height REAL,
                    local_slope REAL,
                    slope_r_squared REAL,
                    timestamp TEXT DEFAULT (datetime('now')),
                    file_source TEXT,
                    UNIQUE(file, feature, point_type, wavelength)
                )
            """)
            
            # Create optimized indexes
            indexes = [
                ("idx_file", "file"),
                ("idx_light_source", "light_source"), 
                ("idx_feature_group", "feature_group"),
                ("idx_wavelength", "wavelength"),
                ("idx_point_type", "point_type"),
                ("idx_file_feature", "file, feature_group")
            ]
            
            for idx_name, columns in indexes:
                cursor.execute(f"CREATE INDEX {idx_name} ON structural_features({columns})")
            
            conn.commit()
            conn.close()
            
            print(f"✅ Perfect database created: {self.db_path}")
            print("📋 Schema: Optimized for normalized wavelength/intensity data")
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
    
    def extract_core_filename(self, file_path):
        """Extract core filename (remove light source and timestamp suffixes)"""
        filename = Path(file_path).stem
        
        # Remove common suffixes: _halogen_structural_20231201_143022
        # Keep just the core stone name
        for suffix in ['_halogen_structural_', '_laser_structural_', '_uv_structural_']:
            if suffix in filename:
                return filename.split(suffix)[0]
        
        # Fallback: remove any _timestamp pattern
        import re
        core = re.sub(r'_\d{8}_\d{6}$', '', filename)
        return core
    
    def check_file_exists(self, core_filename):
        """Check if core filename already exists in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for any file that starts with the core filename
            cursor.execute("SELECT DISTINCT file FROM structural_features WHERE file LIKE ?", (f"{core_filename}%",))
            existing = cursor.fetchall()
            conn.close()
            
            return [row[0] for row in existing] if existing else []
            
        except Exception as e:
            print(f"❌ Error checking file existence: {e}")
            return []
    
    def ask_override_skip(self, core_filename, existing_files):
        """Ask user what to do with duplicate files"""
        print(f"\n⚠️ DUPLICATE DETECTED:")
        print(f"   Core filename: {core_filename}")
        print(f"   Existing files: {existing_files}")
        
        while True:
            choice = input("   Choose action: (O)verride, (S)kip, (A)bort? ").strip().upper()
            if choice in ['O', 'S', 'A']:
                return choice
            print("   Please enter O, S, or A")
    
    def delete_existing_records(self, core_filename):
        """Delete existing records for this core filename"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM structural_features WHERE file LIKE ?", (f"{core_filename}%",))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"   🗑️ Deleted {deleted_count} existing records for {core_filename}")
            return deleted_count
            
        except Exception as e:
            print(f"❌ Error deleting records: {e}")
            return 0
    
    def import_csv_file(self, file_path):
        """Import single CSV file with perfect field mapping"""
        try:
            # Extract core filename for duplicate checking
            core_filename = self.extract_core_filename(file_path)
            existing_files = self.check_file_exists(core_filename)
            
            if existing_files:
                choice = self.ask_override_skip(core_filename, existing_files)
                
                if choice == 'S':
                    print(f"   ⏭️ Skipping {file_path}")
                    return 0, "skipped"
                elif choice == 'A':
                    print(f"   🛑 Aborting import process")
                    return 0, "aborted"
                elif choice == 'O':
                    self.delete_existing_records(core_filename)
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            imported_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Perfect field mapping for new CSV format
                    cursor.execute("""
                        INSERT INTO structural_features 
                        (feature, file, light_source, wavelength, intensity, point_type, 
                         feature_group, processing, baseline_used, norm_factor, snr,
                         symmetry_ratio, skew_description, width_nm, height, 
                         local_slope, slope_r_squared, file_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('Feature', ''),
                        row.get('File', ''),
                        row.get('Light_Source', ''),
                        row.get('Wavelength', 0),
                        row.get('Intensity', 0),
                        row.get('Point_Type', ''),
                        row.get('Feature_Group', ''),
                        row.get('Processing', ''),
                        row.get('Baseline_Used', None),
                        row.get('Norm_Factor', None),
                        row.get('SNR', None),
                        row.get('Symmetry_Ratio', None),
                        row.get('Skew_Description', None),
                        row.get('Width_nm', None),
                        row.get('Height', None),
                        row.get('Local_Slope', None),
                        row.get('Slope_R_Squared', None),
                        str(file_path)
                    ))
                    imported_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Skipped record in {file_path}: {e}")
            
            conn.commit()
            conn.close()
            
            print(f"✅ Imported {imported_count} records from {Path(file_path).name}")
            return imported_count, "imported"
            
        except Exception as e:
            print(f"❌ Error importing {file_path}: {e}")
            return 0, "error"
    
    def batch_import_from_light_folders(self):
        """Import from halogen/, laser/, uv/ folders with smart duplicate handling"""
        base_path = Path(r"c:\users\david\gemini sp10 structural data")
        
        if not base_path.exists():
            print(f"❌ Directory not found: {base_path}")
            return
        
        # Scan all three light source folders
        folders = ['halogen', 'laser', 'uv']
        all_csv_files = []
        
        for folder in folders:
            folder_path = base_path / folder
            if folder_path.exists():
                csv_files = list(folder_path.glob("*_structural_*.csv"))
                all_csv_files.extend(csv_files)
                print(f"📁 {folder}/: Found {len(csv_files)} CSV files")
            else:
                print(f"📁 {folder}/: Directory not found")
        
        if not all_csv_files:
            print("❌ No structural CSV files found in any light folders")
            return
        
        print(f"\n📊 TOTAL: {len(all_csv_files)} CSV files to import")
        print("🔧 Smart duplicate handling: Override/Skip options for existing files")
        
        # Import statistics
        total_imported = 0
        files_imported = 0
        files_skipped = 0
        files_errors = 0
        
        for file_path in all_csv_files:
            print(f"\n📥 Processing: {file_path.name}")
            count, status = self.import_csv_file(file_path)
            
            if status == "imported":
                total_imported += count
                files_imported += 1
            elif status == "skipped":
                files_skipped += 1
            elif status == "aborted":
                print("🛑 Import process aborted by user")
                break
            else:
                files_errors += 1
        
        print(f"\n📊 IMPORT SUMMARY:")
        print(f"   ✅ Files imported: {files_imported}")
        print(f"   ⏭️ Files skipped: {files_skipped}")
        print(f"   ❌ Files with errors: {files_errors}")
        print(f"   📝 Total records: {total_imported}")
        print(f"   🗄️ Database: {self.db_path}")
    
    def view_database_statistics(self):
        """Show statistics for the new perfect database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features")
            unique_files = cursor.fetchone()[0]
            
            # By light source
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source")
            by_light = cursor.fetchall()
            
            # By feature group
            cursor.execute("SELECT feature_group, COUNT(*) FROM structural_features GROUP BY feature_group ORDER BY COUNT(*) DESC")
            by_feature = cursor.fetchall()
            
            # Processing status
            cursor.execute("SELECT processing, COUNT(*) FROM structural_features GROUP BY processing")
            by_processing = cursor.fetchall()
            
            conn.close()
            
            print(f"\n📊 PERFECT DATABASE STATISTICS:")
            print(f"   📝 Total records: {total_records:,}")
            print(f"   📁 Unique files: {unique_files:,}")
            
            print(f"\n   💡 By Light Source:")
            for light, count in by_light:
                light_name = {'Halogen': '🔥 Halogen', 'Laser': '⚡ Laser', 'UV': '🟣 UV'}.get(light, light)
                print(f"      {light_name}: {count:,}")
            
            print(f"\n   🏷️ By Feature Type:")
            for feature, count in by_feature[:8]:  # Top 8
                print(f"      {feature}: {count:,}")
            
            print(f"\n   🔧 By Processing Status:")
            for processing, count in by_processing:
                print(f"      {processing or 'None'}: {count:,}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main_menu():
    """Main menu for new database system"""
    print("🔬 NEW STRUCTURAL DATABASE SYSTEM")
    print("🎯 PERFECT SCHEMA FOR NORMALIZED DATA")
    
    db_manager = NewStructuralDatabaseManager()
    
    while True:
        print(f"\n🎯 MAIN MENU:")
        print("1. 🚀 BATCH IMPORT from light folders (halogen/laser/uv)")
        print("2. 📊 View database statistics") 
        print("3. 🗄️ Show database schema")
        print("4. ❌ Exit")
        
        choice = input("Choice (1-4): ").strip()
        
        if choice == "4":
            print("👋 Goodbye!")
            return
        elif choice == "1":
            db_manager.batch_import_from_light_folders()
        elif choice == "2":
            db_manager.view_database_statistics()
        elif choice == "3":
            print("\n🗄️ PERFECT DATABASE SCHEMA:")
            print("✅ Individual wavelength/intensity records")
            print("✅ Full normalization metadata") 
            print("✅ Processing information")
            print("✅ Smart duplicate handling")
            print("✅ Optimized indexes")
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main_menu()