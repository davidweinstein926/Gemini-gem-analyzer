#!/usr/bin/env python3
"""
CORRECTED DATABASE SYSTEM - Handles ACTUAL peak detection CSV format
FIXES: 
- Handles your actual CSV structure: Peak_Number, Wavelength_nm, Intensity, Prominence, Category
- Proper UV file detection: *UC1.csv, *UP1*.csv, *_uv_structural_*.csv
- Maps peak data to structural features schema
- Appends to existing database
"""

import sqlite3
import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

class CorrectedStructuralDatabaseManager:
    def __init__(self, db_path="multi_structural_gem_data.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Create database ONLY if it doesn't exist - APPEND MODE"""
        if os.path.exists(self.db_path):
            print(f"✅ Using existing database: {self.db_path}")
            self.update_database_schema()
            return
        
        print(f"🆕 Creating new database: {self.db_path}")
        self.create_fresh_database()
    
    def update_database_schema(self):
        """Update existing database schema if needed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get existing columns
            cursor.execute("PRAGMA table_info(structural_features)")
            existing_cols = [row[1] for row in cursor.fetchall()]
            
            # Add missing columns for peak data
            required_columns = [
                ('peak_number', 'INTEGER'),
                ('prominence', 'REAL'),
                ('category', 'TEXT'),
                ('processing', 'TEXT'),
                ('baseline_used', 'REAL'),
                ('norm_factor', 'REAL'),
                ('snr', 'REAL'),
                ('symmetry_ratio', 'REAL'),
                ('skew_description', 'TEXT'),
                ('width_nm', 'REAL'),
                ('height', 'REAL'),
                ('local_slope', 'REAL'),
                ('slope_r_squared', 'REAL'),
                ('file_source', 'TEXT')
            ]
            
            for col_name, col_type in required_columns:
                if col_name not in existing_cols:
                    cursor.execute(f"ALTER TABLE structural_features ADD COLUMN {col_name} {col_type}")
                    print(f"➕ Added column: {col_name}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Schema update warning: {e}")
    
    def create_fresh_database(self):
        """Create new database with schema for peak detection data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structural_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature TEXT NOT NULL,
                    file TEXT NOT NULL,
                    light_source TEXT NOT NULL,
                    wavelength REAL NOT NULL,
                    intensity REAL NOT NULL,
                    point_type TEXT NOT NULL,
                    feature_group TEXT NOT NULL,
                    peak_number INTEGER,
                    prominence REAL,
                    category TEXT,
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
                    UNIQUE(file, peak_number, wavelength)
                )
            """)
            
            # Create indexes
            indexes = [
                ("idx_file", "file"),
                ("idx_light_source", "light_source"), 
                ("idx_wavelength", "wavelength"),
                ("idx_category", "category"),
                ("idx_peak_number", "peak_number")
            ]
            
            for idx_name, columns in indexes:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON structural_features({columns})")
            
            conn.commit()
            conn.close()
            
            print(f"✅ Database created for peak detection data: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Error creating database: {e}")
    
    def detect_light_source_from_filename(self, file_path):
        """Detect light source from YOUR ACTUAL file naming patterns"""
        file_path_str = str(file_path).lower()
        filename = Path(file_path).stem.lower()
        
        print(f"   🔍 Analyzing: {filename}")
        
        # Check folder path first
        if '/uv/' in file_path_str or '\\uv\\' in file_path_str:
            print(f"   📁 Found UV in path")
            return 'UV'
        elif '/halogen/' in file_path_str or '\\halogen\\' in file_path_str:
            print(f"   📁 Found Halogen in path")
            return 'Halogen'
        elif '/laser/' in file_path_str or '\\laser\\' in file_path_str:
            print(f"   📁 Found Laser in path")
            return 'Laser'
        
        # Check YOUR ACTUAL UV naming patterns
        if filename.endswith('uc1') or 'uc1' in filename:
            print(f"   🟣 Detected UC1 pattern -> UV")
            return 'UV'
        elif filename.endswith('up1') or 'up1' in filename:
            print(f"   🟣 Detected UP1 pattern -> UV")
            return 'UV'
        elif '_uv_structural_' in filename:
            print(f"   🟣 Detected _uv_structural_ pattern -> UV")
            return 'UV'
        
        # Check for Halogen patterns
        elif filename.endswith('bc1') or 'bc1' in filename or filename.endswith('_b'):
            print(f"   🔥 Detected Halogen pattern")
            return 'Halogen'
        elif '_halogen_' in filename:
            print(f"   🔥 Detected _halogen_ pattern")
            return 'Halogen'
        
        # Check for Laser patterns  
        elif filename.endswith('lc1') or 'lc1' in filename or filename.endswith('_l'):
            print(f"   ⚡ Detected Laser pattern")
            return 'Laser'
        elif '_laser_' in filename:
            print(f"   ⚡ Detected _laser_ pattern")
            return 'Laser'
        
        print(f"   ❓ Could not determine light source")
        return 'Unknown'
    
    def check_exact_duplicate(self, file_path):
        """Check for exact file duplicate in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            filename = Path(file_path).name
            
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE file LIKE ? OR file_source LIKE ?", 
                          (f"%{Path(file_path).stem}%", f"%{filename}"))
            count = cursor.fetchone()[0]
            conn.close()
            
            return count > 0
            
        except Exception as e:
            print(f"⚠️ Error checking duplicates: {e}")
            return False
    
    def import_peak_detection_csv(self, file_path, skip_duplicates=False):
        """Import YOUR ACTUAL CSV format: Peak_Number, Wavelength_nm, Intensity, Prominence, Category"""
        try:
            print(f"📥 Processing: {Path(file_path).name}")
            
            # Check for duplicates
            if not skip_duplicates and self.check_exact_duplicate(file_path):
                choice = input("   Duplicate detected. (O)verride, (S)kip, (A)bort, (C)ontinue all? ").strip().upper()
                
                if choice == 'S':
                    print(f"   ⭐ Skipping")
                    return 0, "skipped", skip_duplicates
                elif choice == 'A':
                    return 0, "aborted", skip_duplicates
                elif choice == 'C':
                    skip_duplicates = True
                elif choice == 'O':
                    # Delete existing records
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    filename_pattern = Path(file_path).stem
                    cursor.execute("DELETE FROM structural_features WHERE file LIKE ?", (f"%{filename_pattern}%",))
                    deleted = cursor.rowcount
                    conn.commit()
                    conn.close()
                    print(f"   🗑️ Deleted {deleted} existing records")
            
            # Detect light source using YOUR patterns
            light_source = self.detect_light_source_from_filename(file_path)
            
            # Read YOUR CSV format
            try:
                df = pd.read_csv(file_path)
                print(f"   📊 Read {len(df)} rows")
                print(f"   📋 Columns: {list(df.columns)}")
            except Exception as e:
                print(f"   ❌ Failed to read CSV: {e}")
                return 0, "error", skip_duplicates
            
            # Check if this is YOUR format
            expected_cols = ['Peak_Number', 'Wavelength_nm', 'Intensity', 'Prominence', 'Category']
            if not all(col in df.columns for col in expected_cols):
                print(f"   ⚠️ Not peak detection format. Expected: {expected_cols}")
                print(f"   📋 Found: {list(df.columns)}")
                return 0, "wrong_format", skip_duplicates
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            imported_count = 0
            base_filename = Path(file_path).stem
            
            for idx, row in df.iterrows():
                try:
                    # Map YOUR CSV format to database schema
                    cursor.execute("""
                        INSERT OR IGNORE INTO structural_features 
                        (feature, file, light_source, wavelength, intensity, point_type, 
                         feature_group, peak_number, prominence, category, file_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"Peak_{int(row['Peak_Number'])}", # feature
                        base_filename,                     # file
                        light_source,                      # light_source  
                        float(row['Wavelength_nm']),       # wavelength
                        float(row['Intensity']),           # intensity
                        'Peak',                            # point_type
                        str(row['Category']),              # feature_group
                        int(row['Peak_Number']),           # peak_number
                        float(row['Prominence']),          # prominence
                        str(row['Category']),              # category
                        str(file_path)                     # file_source
                    ))
                    
                    if cursor.rowcount > 0:
                        imported_count += 1
                    
                except Exception as e:
                    print(f"   ⚠️ Row {idx} error: {e}")
            
            conn.commit()
            conn.close()
            
            print(f"   ✅ Imported {imported_count} peak records as {light_source}")
            return imported_count, "imported", skip_duplicates
            
        except Exception as e:
            print(f"   ❌ Import error: {e}")
            return 0, "error", skip_duplicates
    
    def scan_for_csv_files(self):
        """Scan for CSV files in the current directory and subdirectories"""
        current_dir = Path.cwd()
        base_path = Path(r"c:\users\david\gemini sp10 structural data")
        
        all_csv_files = []
        
        # Check the expected base path first
        if base_path.exists():
            print(f"🔍 Scanning base path: {base_path}")
            folders = ['halogen', 'laser', 'uv']
            
            for folder in folders:
                folder_path = base_path / folder
                if folder_path.exists():
                    csv_files = list(folder_path.glob("*.csv"))
                    all_csv_files.extend(csv_files)
                    print(f"   📁 {folder}/: {len(csv_files)} CSV files")
        
        # Also check current directory
        local_csvs = list(current_dir.glob("*.csv"))
        if local_csvs:
            all_csv_files.extend(local_csvs)
            print(f"📁 Current directory: {len(local_csvs)} CSV files")
        
        return all_csv_files
    
    def batch_import_csv_files(self):
        """Import all found CSV files"""
        all_csv_files = self.scan_for_csv_files()
        
        if not all_csv_files:
            print("❌ No CSV files found")
            return
        
        print(f"\n📊 FOUND {len(all_csv_files)} CSV files total")
        
        # Show examples of each naming pattern
        uv_files = [f for f in all_csv_files if self.detect_light_source_from_filename(f) == 'UV']
        halogen_files = [f for f in all_csv_files if self.detect_light_source_from_filename(f) == 'Halogen']
        laser_files = [f for f in all_csv_files if self.detect_light_source_from_filename(f) == 'Laser']
        unknown_files = [f for f in all_csv_files if self.detect_light_source_from_filename(f) == 'Unknown']
        
        print(f"\n🔍 LIGHT SOURCE DETECTION:")
        print(f"   🟣 UV files: {len(uv_files)}")
        if uv_files:
            for f in uv_files[:3]:
                print(f"      📄 {f.name}")
        
        print(f"   🔥 Halogen files: {len(halogen_files)}")
        if halogen_files:
            for f in halogen_files[:3]:
                print(f"      📄 {f.name}")
        
        print(f"   ⚡ Laser files: {len(laser_files)}")
        if laser_files:
            for f in laser_files[:3]:
                print(f"      📄 {f.name}")
        
        if unknown_files:
            print(f"   ❓ Unknown files: {len(unknown_files)}")
            for f in unknown_files[:3]:
                print(f"      📄 {f.name}")
        
        proceed = input(f"\nProceed with import? (y/N): ").strip().lower()
        if proceed != 'y':
            print("❌ Import cancelled")
            return
        
        # Import files
        total_imported = 0
        files_imported = 0
        files_skipped = 0
        files_errors = 0
        skip_duplicates = False
        
        for file_path in all_csv_files:
            count, status, skip_duplicates = self.import_peak_detection_csv(file_path, skip_duplicates)
            
            if status == "imported":
                total_imported += count
                files_imported += 1
            elif status == "skipped":
                files_skipped += 1
            elif status == "aborted":
                print("🛑 Import aborted")
                break
            else:
                files_errors += 1
        
        print(f"\n📊 IMPORT SUMMARY:")
        print(f"   ✅ Files imported: {files_imported}")
        print(f"   ⭐ Files skipped: {files_skipped}")
        print(f"   ❌ Files with errors: {files_errors}")
        print(f"   📊 Total peak records: {total_imported}")
        
        # Show updated stats
        self.view_database_statistics()
    
    def view_database_statistics(self):
        """Show current database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features")
            unique_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source ORDER BY light_source")
            by_light = cursor.fetchall()
            
            cursor.execute("SELECT category, COUNT(*) FROM structural_features WHERE category IS NOT NULL GROUP BY category ORDER BY COUNT(*) DESC LIMIT 5")
            by_category = cursor.fetchall()
            
            conn.close()
            
            print(f"\n📊 DATABASE STATISTICS:")
            print(f"   📁 Total records: {total_records:,}")
            print(f"   📄 Unique files: {unique_files:,}")
            
            print(f"\n   💡 By Light Source:")
            for light, count in by_light:
                icons = {'Halogen': '🔥', 'Laser': '⚡', 'UV': '🟣', 'Unknown': '❓'}
                icon = icons.get(light, '💡')
                print(f"      {icon} {light}: {count:,}")
            
            if by_category:
                print(f"\n   📋 Top Categories:")
                for category, count in by_category:
                    print(f"      {category}: {count:,}")
                
        except Exception as e:
            print(f"❌ Error reading database: {e}")

def main_menu():
    """Main menu for CORRECTED database system"""
    print("🔬 CORRECTED DATABASE SYSTEM")
    print("✅ Handles YOUR actual CSV format: Peak_Number, Wavelength_nm, Intensity, Prominence, Category")
    print("🟣 Fixed UV detection: UC1, UP1, _uv_structural_ patterns")
    print("📊 APPEND MODE - preserves existing data")
    
    db_manager = CorrectedStructuralDatabaseManager()
    
    while True:
        print(f"\n🎯 MAIN MENU:")
        print("1. 🚀 IMPORT CSV FILES (auto-detect peak format)")
        print("2. 📊 View database statistics")
        print("3. 🔍 Test single file (debug)")
        print("4. 📁 Show detected light sources")
        print("5. ❌ Exit")
        
        try:
            choice = input("Choice (1-5): ").strip()
            
            if choice == "5":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                print("\n🚀 STARTING CSV IMPORT...")
                db_manager.batch_import_csv_files()
            elif choice == "2":
                db_manager.view_database_statistics()
            elif choice == "3":
                test_file = input("Enter CSV file path: ").strip()
                if os.path.exists(test_file):
                    print(f"\n🔍 Testing: {test_file}")
                    light_source = db_manager.detect_light_source_from_filename(test_file)
                    print(f"   Detected light source: {light_source}")
                    
                    # Try reading structure
                    try:
                        df = pd.read_csv(test_file)
                        print(f"   Columns: {list(df.columns)}")
                        print(f"   Rows: {len(df)}")
                    except Exception as e:
                        print(f"   Error reading: {e}")
                else:
                    print("❌ File not found")
            elif choice == "4":
                print("\n🔍 TESTING LIGHT SOURCE DETECTION:")
                test_names = [
                    "190UC1.csv",
                    "189UC1.csv", 
                    "140UP1_uv_structural_20250815_150616.csv",
                    "60UC1_uv_structural_20250815_150616.csv",
                    "some_halogen_file_BC1.csv",
                    "laser_file_LC1.csv"
                ]
                
                for name in test_names:
                    light_source = db_manager.detect_light_source_from_filename(name)
                    print(f"   📄 {name} -> {light_source}")
            else:
                print(f"❌ Invalid choice. Please enter 1-5")
                continue
        
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
        
        if choice in ['1', '2', '3', '4']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main_menu() 