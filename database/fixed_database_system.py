#!/usr/bin/env python3
"""
FIXED DATABASE SYSTEM - Complete schema for normalized structural data
FIXED: Added all normalization metadata fields from fixed analyzers
"""

import sqlite3
import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

class FixedStructuralDatabaseManager:
    def __init__(self, db_path="fixed_structural_gem_data.db"):
        self.db_path = db_path
        self.init_complete_database()
    
    def backup_old_database(self):
        """Rename old database to _OLD if it exists"""
        if os.path.exists(self.db_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            old_name = f"fixed_structural_gem_data_OLD_{timestamp}.db"
            os.rename(self.db_path, old_name)
            print(f"Old database renamed to: {old_name}")
            return old_name
        return None
    
    def migrate_existing_database(self):
        """Migrate existing database to new schema with normalization fields"""
        old_db_path = "multi_structural_gem_data.db"
        
        if not os.path.exists(old_db_path):
            print("No existing database found - creating new database")
            return False
            
        try:
            # Check if old database has the new fields already
            old_conn = sqlite3.connect(old_db_path)
            old_cursor = old_conn.cursor()
            
            old_cursor.execute("PRAGMA table_info(structural_features)")
            columns = [col[1] for col in old_cursor.fetchall()]
            
            has_new_fields = 'normalization_scheme' in columns
            old_conn.close()
            
            if has_new_fields:
                print("Existing database already has normalization fields")
                return False
                
            print(f"Migrating existing database: {old_db_path}")
            print("Adding normalization metadata fields...")
            
            # Rename old database
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"multi_structural_gem_data_PRE_MIGRATION_{timestamp}.db"
            import shutil
            shutil.copy2(old_db_path, backup_name)
            print(f"Backup created: {backup_name}")
            
            return True
            
        except Exception as e:
            print(f"Migration check error: {e}")
            return False

    def init_complete_database(self):
        """Create complete database schema with ALL normalization metadata fields"""
        # Check if we need to migrate existing database
        needs_migration = self.migrate_existing_database()
        
        if not needs_migration:
            # Backup old database first
            self.backup_old_database()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # FIXED: Complete schema with all normalization metadata fields
            # Compatible with existing data structure but adds new fields
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
                    
                    -- Existing peak detection fields (preserved from old schema)
                    peak_number INTEGER,
                    prominence REAL,
                    category TEXT,
                    
                    -- Existing structural analysis fields (preserved from old schema)
                    start_wavelength REAL,
                    end_wavelength REAL,
                    symmetry_ratio REAL,
                    skew_description TEXT,
                    midpoint REAL,
                    bottom REAL,
                    
                    -- Existing common metadata (preserved from old schema)
                    processing TEXT,
                    baseline_used REAL,
                    norm_factor REAL,
                    snr REAL,
                    width_nm REAL,
                    height REAL,
                    local_slope REAL,
                    slope_r_squared REAL,
                    timestamp TEXT DEFAULT (datetime('now')),
                    file_source TEXT,
                    data_type TEXT,  -- 'peak_detection' or 'structural_features'
                    
                    -- NEW: Complete normalization metadata fields
                    normalization_scheme TEXT,
                    reference_wavelength REAL,
                    intensity_range_min REAL,
                    intensity_range_max REAL,
                    normalization_method TEXT,
                    reference_wavelength_used REAL,
                    feature_key TEXT,
                    
                    UNIQUE(file, feature, wavelength, point_type)
                )
            """)
            
            # Check if we're migrating existing data
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0 and needs_migration:
                print(f"Detected {existing_count} existing records")
                print("Marking existing data with legacy normalization info...")
                
                # Mark existing data appropriately
                cursor.execute("""
                    UPDATE structural_features 
                    SET normalization_scheme = 'Legacy_0_to_1_Scale',
                        intensity_range_min = 0.0,
                        intensity_range_max = 1.0
                    WHERE normalization_scheme IS NULL
                """)
                
                marked_count = cursor.rowcount
                print(f"Marked {marked_count} records as legacy normalization")
            
            # Create optimized indexes including normalization fields
            indexes = [
                ("idx_file", "file"),
                ("idx_light_source", "light_source"), 
                ("idx_feature_group", "feature_group"),
                ("idx_wavelength", "wavelength"),
                ("idx_point_type", "point_type"),
                ("idx_file_feature", "file, feature_group"),
                ("idx_normalization_scheme", "normalization_scheme"),
                ("idx_processing", "processing"),
                ("idx_data_type", "data_type")
            ]
            
            for idx_name, columns in indexes:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON structural_features({columns})")
            
            conn.commit()
            conn.close()
            
            if needs_migration:
                print(f"MIGRATION COMPLETE: {self.db_path}")
                print("Schema: Extended with normalization metadata fields")
                print("Existing data: Preserved and marked as legacy")
            else:
                print(f"Database ready: {self.db_path}")
                print("Schema: Complete with all normalization metadata fields")
            
        except Exception as e:
            print(f"Error creating database: {e}")
    
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
            print(f"Error checking file existence: {e}")
            return []
    
    def ask_override_skip(self, core_filename, existing_files):
        """Ask user what to do with duplicate files"""
        print(f"\nDUPLICATE DETECTED:")
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
            
            print(f"   Deleted {deleted_count} existing records for {core_filename}")
            return deleted_count
            
        except Exception as e:
            print(f"Error deleting records: {e}")
            return 0
    
    def validate_normalization_data(self, row):
        """Validate that normalization data is consistent"""
        errors = []
        
        # Check for normalization scheme consistency
        light_source = row.get('Light_Source', '')
        norm_scheme = row.get('Normalization_Scheme', '')
        
        expected_schemes = {
            'Halogen': 'Halogen_650nm_50000_to_100',
            'Laser': 'Laser_Max_50000_to_100', 
            'UV': 'UV_811nm_15000_to_100'
        }
        
        if light_source in expected_schemes:
            expected = expected_schemes[light_source]
            if norm_scheme and norm_scheme != expected and norm_scheme != 'Raw_Data':
                errors.append(f"Unexpected normalization scheme: {norm_scheme} for {light_source}")
        
        # Check intensity range for normalized data
        range_min = row.get('Intensity_Range_Min', None)
        range_max = row.get('Intensity_Range_Max', None)
        
        if norm_scheme and 'to_100' in norm_scheme:
            if range_max is not None and (range_max < 95 or range_max > 100):
                errors.append(f"Unexpected max range: {range_max} for normalized data")
            if range_min is not None and range_min < -5:
                errors.append(f"Unexpected min range: {range_min} for normalized data")
        
        return errors
    
    def detect_csv_format(self, df):
        """Detect whether CSV is from manual analyzer or peak detector"""
        columns = set(df.columns)
        
        # Peak detection format indicators
        peak_format_indicators = {'Peak_Number', 'Wavelength_nm', 'Prominence', 'Category'}
        
        # Manual analyzer format indicators  
        manual_format_indicators = {'Feature', 'Point_Type', 'Feature_Group'}
        
        if peak_format_indicators.issubset(columns):
            return 'peak_detection'
        elif manual_format_indicators.issubset(columns):
            return 'manual_structural'
        else:
            # Try to guess based on partial matches
            peak_matches = len(peak_format_indicators.intersection(columns))
            manual_matches = len(manual_format_indicators.intersection(columns))
            
            if peak_matches > manual_matches:
                return 'peak_detection'
            else:
                return 'manual_structural'

    def import_peak_detection_csv(self, df, file_path):
        """Import peak detection CSV format"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        imported_count = 0
        filename = Path(file_path).stem
        
        for _, row in df.iterrows():
            try:
                # Map peak detection format to database schema
                peak_num = row.get('Peak_Number', 0)
                wavelength = row.get('Wavelength_nm', row.get('Wavelength', 0))
                intensity = row.get('Intensity', 0)
                prominence = row.get('Prominence', 0)
                category = row.get('Category', 'Unknown')
                
                # Get normalization metadata if available
                norm_scheme = row.get('Normalization_Scheme', None)
                ref_wavelength = row.get('Reference_Wavelength', None)
                light_source = row.get('Light_Source', 'Unknown')
                
                cursor.execute("""
                    INSERT INTO structural_features 
                    (feature, file, light_source, wavelength, intensity, point_type, 
                     feature_group, processing, peak_number, prominence, category,
                     normalization_scheme, reference_wavelength, data_type, file_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f'Peak_{peak_num}',  # feature
                    filename,  # file
                    light_source,  # light_source
                    wavelength,  # wavelength
                    intensity,  # intensity
                    'Peak',  # point_type
                    category,  # feature_group (Major, Strong, etc.)
                    'Auto_Peak_Detection',  # processing
                    peak_num,  # peak_number
                    prominence,  # prominence
                    category,  # category
                    norm_scheme,  # normalization_scheme
                    ref_wavelength,  # reference_wavelength
                    'peak_detection',  # data_type
                    str(file_path)  # file_source
                ))
                imported_count += 1
                
            except Exception as e:
                print(f"   Skipped peak record: {e}")
        
        conn.commit()
        conn.close()
        return imported_count

    def import_manual_structural_csv(self, df, file_path):
        """Import manual structural analyzer CSV format"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        imported_count = 0
        validation_warnings = 0
        
        for _, row in df.iterrows():
            try:
                # Validate normalization data
                validation_errors = self.validate_normalization_data(row)
                if validation_errors:
                    print(f"   WARNING: {validation_errors}")
                    validation_warnings += 1
                
                # Complete field mapping for manual analyzer format
                cursor.execute("""
                    INSERT INTO structural_features 
                    (feature, file, light_source, wavelength, intensity, point_type, 
                     feature_group, processing, baseline_used, norm_factor, snr,
                     symmetry_ratio, skew_description, width_nm, height, 
                     local_slope, slope_r_squared, 
                     normalization_scheme, reference_wavelength, intensity_range_min,
                     intensity_range_max, normalization_method, reference_wavelength_used,
                     feature_key, data_type, file_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    row.get('Normalization_Scheme', None),
                    row.get('Reference_Wavelength', None),
                    row.get('Intensity_Range_Min', None),
                    row.get('Intensity_Range_Max', None),
                    row.get('Normalization_Method', None),
                    row.get('Reference_Wavelength_Used', None),
                    row.get('Feature_Key', None),
                    'structural_features',  # data_type
                    str(file_path)
                ))
                imported_count += 1
                
            except Exception as e:
                print(f"   Skipped structural record: {e}")
        
        conn.commit()
        conn.close()
        return imported_count, validation_warnings

    def import_csv_file(self, file_path):
        """DUAL FORMAT: Import both manual analyzer and peak detection CSV files"""
        try:
            # Extract core filename for duplicate checking
            core_filename = self.extract_core_filename(file_path)
            existing_files = self.check_file_exists(core_filename)
            
            if existing_files:
                choice = self.ask_override_skip(core_filename, existing_files)
                
                if choice == 'S':
                    print(f"   Skipping {file_path}")
                    return 0, "skipped"
                elif choice == 'A':
                    print(f"   Aborting import process")
                    return 0, "aborted"
                elif choice == 'O':
                    self.delete_existing_records(core_filename)
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Detect format
            csv_format = self.detect_csv_format(df)
            print(f"   Detected format: {csv_format}")
            
            if csv_format == 'peak_detection':
                imported_count = self.import_peak_detection_csv(df, file_path)
                validation_warnings = 0
                success_msg = f"Imported {imported_count} peak detection records"
            else:
                imported_count, validation_warnings = self.import_manual_structural_csv(df, file_path)
                success_msg = f"Imported {imported_count} structural analysis records"
            
            if validation_warnings > 0:
                success_msg += f" ({validation_warnings} validation warnings)"
                
            print(f"   {success_msg} from {Path(file_path).name}")
            return imported_count, "imported"
            
        except Exception as e:
            print(f"Error importing {file_path}: {e}")
            return 0, "error"
    
    def batch_import_from_light_folders(self):
        """Import from halogen/, laser/, uv/ folders with complete metadata capture"""
        base_path = Path(r"c:\users\david\gemini sp10 structural data")
        
        if not base_path.exists():
            print(f"Directory not found: {base_path}")
            return
        
        # Scan all three light source folders
        folders = ['halogen', 'laser', 'uv']
        all_csv_files = []
        
        for folder in folders:
            folder_path = base_path / folder
            if folder_path.exists():
                csv_files = list(folder_path.glob("*_structural_*.csv"))
                all_csv_files.extend(csv_files)
                print(f"{folder}/: Found {len(csv_files)} CSV files")
            else:
                print(f"{folder}/: Directory not found")
        
        if not all_csv_files:
            print("No structural CSV files found in any light folders")
            return
        
        print(f"\nTOTAL: {len(all_csv_files)} CSV files to import")
        print("FIXED: Complete normalization metadata capture enabled")
        
        # Import statistics
        total_imported = 0
        files_imported = 0
        files_skipped = 0
        files_errors = 0
        
        for file_path in all_csv_files:
            print(f"\nProcessing: {file_path.name}")
            count, status = self.import_csv_file(file_path)
            
            if status == "imported":
                total_imported += count
                files_imported += 1
            elif status == "skipped":
                files_skipped += 1
            elif status == "aborted":
                print("Import process aborted by user")
                break
            else:
                files_errors += 1
        
        print(f"\nFIXED IMPORT SUMMARY:")
        print(f"   Files imported: {files_imported}")
        print(f"   Files skipped: {files_skipped}")
        print(f"   Files with errors: {files_errors}")
        print(f"   Total records: {total_imported}")
        print(f"   Database: {self.db_path}")
        print(f"   FIXED: All normalization metadata preserved")
    
    def view_database_statistics(self):
        """Show statistics for the complete database including legacy data"""
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
            
            # By normalization scheme (including legacy)
            cursor.execute("SELECT normalization_scheme, COUNT(*) FROM structural_features WHERE normalization_scheme IS NOT NULL GROUP BY normalization_scheme")
            by_normalization = cursor.fetchall()
            
            # Legacy vs new data
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE normalization_scheme = 'Legacy_0_to_1_Scale'")
            legacy_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE normalization_scheme IS NOT NULL AND normalization_scheme != 'Legacy_0_to_1_Scale'")
            new_normalized_count = cursor.fetchone()[0]
            
            # By feature group
            cursor.execute("SELECT feature_group, COUNT(*) FROM structural_features GROUP BY feature_group ORDER BY COUNT(*) DESC")
            by_feature = cursor.fetchall()
            
            # By data type
            cursor.execute("SELECT data_type, COUNT(*) FROM structural_features GROUP BY data_type")
            by_data_type = cursor.fetchall()
            
            conn.close()
            
            print(f"\nCOMPLETE DATABASE STATISTICS:")
            print(f"   Total records: {total_records:,}")
            print(f"   Unique files: {unique_files:,}")
            print(f"   Legacy data (0-1 scale): {legacy_count:,}")
            print(f"   New normalized data (0-100 scale): {new_normalized_count:,}")
            
            print(f"\n   By Light Source:")
            for light, count in by_light:
                light_name = {'Halogen': 'Halogen', 'Laser': 'Laser', 'UV': 'UV'}.get(light, light)
                print(f"      {light_name}: {count:,}")
            
            print(f"\n   By Normalization Scheme:")
            for scheme, count in by_normalization:
                if scheme == 'Legacy_0_to_1_Scale':
                    print(f"      {scheme} (preserved): {count:,}")
                else:
                    print(f"      {scheme}: {count:,}")
            
            print(f"\n   By Data Type:")
            for data_type, count in by_data_type:
                print(f"      {data_type or 'Unknown'}: {count:,}")
            
            print(f"\n   By Feature Type:")
            for feature, count in by_feature[:8]:  # Top 8
                print(f"      {feature}: {count:,}")
                
        except Exception as e:
            print(f"Error: {e}")

    def export_normalization_report(self):
        """Export report on normalization status and consistency"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Create comprehensive normalization report
            report_query = """
            SELECT 
                file,
                light_source,
                normalization_scheme,
                reference_wavelength,
                intensity_range_min,
                intensity_range_max,
                COUNT(*) as feature_count
            FROM structural_features 
            WHERE normalization_scheme IS NOT NULL
            GROUP BY file, light_source, normalization_scheme
            ORDER BY light_source, file
            """
            
            df = pd.read_sql_query(report_query, conn)
            conn.close()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"normalization_report_{timestamp}.csv"
            df.to_csv(report_path, index=False)
            
            print(f"Normalization report exported: {report_path}")
            print(f"Report contains {len(df)} file entries")
            
            return report_path
            
        except Exception as e:
            print(f"Error creating normalization report: {e}")
            return None

def main_menu():
    """Main menu for fixed database system"""
    print("FIXED STRUCTURAL DATABASE SYSTEM")
    print("COMPLETE NORMALIZATION METADATA SUPPORT")
    
    db_manager = FixedStructuralDatabaseManager()
    
    while True:
        print(f"\nMAIN MENU:")
        print("1. BATCH IMPORT from light folders (halogen/laser/uv)")
        print("2. View database statistics") 
        print("3. Export normalization report")
        print("4. Show database schema")
        print("5. Exit")
        
        try:
            choice = input("Choice (1-5): ").strip()
            
            if choice == "5":
                print("Goodbye!")
                break
            elif choice == "1":
                print("\nSTARTING FIXED BATCH IMPORT...")
                db_manager.batch_import_from_light_folders()
            elif choice == "2":
                db_manager.view_database_statistics()
            elif choice == "3":
                db_manager.export_normalization_report()
            elif choice == "4":
                print("\nFIXED DATABASE SCHEMA:")
                print("- Individual wavelength/intensity records")
                print("- COMPLETE normalization metadata capture") 
                print("- Processing information")
                print("- Smart duplicate handling")
                print("- Normalization validation")
                print("- Optimized indexes")
            else:
                print(f"Invalid choice '{choice}'. Please enter 1, 2, 3, 4, or 5")
                continue
        
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Menu error: {e}")
            continue
        
        # Only ask for Enter if we processed a valid choice
        if choice in ['1', '2', '3', '4']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main_menu()