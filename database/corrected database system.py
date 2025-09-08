#!/usr/bin/env python3
"""
FIXED DATABASE SYSTEM - OPTIMIZED - Compatible with 0-100 Normalization Metadata
FIXES: Added normalization metadata, validates 0-100 intensity ranges, tracks schemes
OPTIMIZED: Reduced line count while maintaining all functionality
"""
import sqlite3
import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

class FixedStructuralDatabaseManager:
    def __init__(self, db_path="multi_structural_gem_data.db"):
        self.db_path = db_path
        self.base_path = Path(r"c:\users\david\gemini sp10 structural data")
        
        # Expected normalization schemes for validation
        self.expected_schemes = {
            'UV': 'UV_811nm_15000_to_100',
            'Halogen': 'Halogen_650nm_50000_to_100',
            'Laser': 'Laser_max_50000_to_100'
        }
        
    def create_fresh_database(self):
        """OPTIMIZED: Create database with FIXED normalization metadata fields"""
        if os.path.exists(self.db_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"multi_structural_gem_data_OLD_{timestamp}.db"
            os.rename(self.db_path, backup_name)
            print(f"Old database backed up as: {backup_name}")
        
        print(f"Creating FIXED database with normalization metadata: {self.db_path}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # OPTIMIZED: Combined schema creation
            cursor.execute("""
                CREATE TABLE structural_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature TEXT NOT NULL, file TEXT NOT NULL, light_source TEXT NOT NULL,
                    wavelength REAL NOT NULL, intensity REAL NOT NULL, point_type TEXT NOT NULL,
                    feature_group TEXT NOT NULL,
                    
                    -- Detection and analysis fields
                    peak_number INTEGER, prominence REAL, category TEXT,
                    start_wavelength REAL, end_wavelength REAL, symmetry_ratio REAL,
                    skew_description TEXT, midpoint REAL, bottom REAL,
                    
                    -- FIXED: Normalization metadata fields
                    normalization_scheme TEXT, reference_wavelength REAL, laser_normalization_wavelength REAL,
                    intensity_range_min REAL, intensity_range_max REAL, normalization_compatible BOOLEAN DEFAULT 1,
                    
                    -- Common metadata
                    processing TEXT, baseline_used REAL, norm_factor REAL, snr REAL, width_nm REAL,
                    height REAL, local_slope REAL, slope_r_squared REAL, timestamp TEXT DEFAULT (datetime('now')),
                    file_source TEXT, data_type TEXT,
                    
                    UNIQUE(file, feature, wavelength, point_type)
                )
            """)
            
            # OPTIMIZED: Index creation in loop
            indexes = [
                ("idx_file", "file"), ("idx_light_source", "light_source"), ("idx_wavelength", "wavelength"),
                ("idx_data_type", "data_type"), ("idx_category", "category"), ("idx_feature_group", "feature_group"),
                ("idx_normalization_scheme", "normalization_scheme"), ("idx_intensity_range", "intensity_range_min, intensity_range_max"),
                ("idx_normalization_compatible", "normalization_compatible")
            ]
            
            for idx_name, columns in indexes:
                cursor.execute(f"CREATE INDEX {idx_name} ON structural_features({columns})")
            
            conn.commit()
            conn.close()
            print("FIXED database created with normalization metadata support!")
            
        except Exception as e:
            print(f"Error creating FIXED database: {e}")
    
    def detect_light_source_from_folder(self, file_path):
        """OPTIMIZED: Detect light source from folder location"""
        file_path_str = str(file_path).lower()
        
        # Check folder path first
        for light_source, patterns in [
            ('UV', ['\\uv\\', '/uv/']),
            ('Halogen', ['\\halogen\\', '/halogen/']),
            ('Laser', ['\\laser\\', '/laser/'])
        ]:
            if any(pattern in file_path_str for pattern in patterns):
                return light_source
        
        # Fallback to filename patterns
        filename = Path(file_path).stem.lower()
        filename_patterns = {
            'UV': ['uc1', 'up1', '_uv_'],
            'Halogen': ['bc1', '_halogen_', '_b'],
            'Laser': ['lc1', '_laser_', '_l']
        }
        
        for light_source, patterns in filename_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return light_source
        
        return 'Unknown'
    
    def detect_csv_format(self, file_path):
        """OPTIMIZED: Detect CSV format including new normalization metadata"""
        try:
            df = pd.read_csv(file_path)
            columns = set(df.columns)
            
            # Check format types
            if {'Peak_Number', 'Wavelength_nm', 'Intensity', 'Prominence', 'Category'}.issubset(columns):
                return ('peak_detection_fixed' if {'Normalization_Scheme', 'Reference_Wavelength', 'Light_Source'}.issubset(columns) 
                       else 'peak_detection_legacy'), df
            elif {'Feature', 'File'}.issubset(columns) and ('Crest' in columns or 'Wavelength' in columns):
                return 'structural_features', df
            
            return 'unknown', df
            
        except Exception as e:
            print(f"   Error reading {file_path}: {e}")
            return 'error', None
    
    def validate_data(self, intensities, file_path, light_source, normalization_scheme=None):
        """OPTIMIZED: Combined intensity and normalization validation"""
        if not intensities:
            return False, "No intensity values found", False
        
        min_int, max_int = min(intensities), max(intensities)
        print(f"      Intensity range: {min_int:.3f} - {max_int:.3f}")
        
        # Intensity validation
        intensity_issues = [
            (max_int <= 1.0, "ERROR: 0-1 normalized data detected (breaks UV analysis)"),
            (max_int > 150.0, "WARNING: Intensities exceed expected 0-100 range"),
            (min_int < -5.0, "WARNING: Unusually negative intensities"),
            (max_int < 10.0, "WARNING: Very low maximum intensity for 0-100 scale")
        ]
        
        for condition, message in intensity_issues:
            if condition:
                return False, message, False
        
        # UV-specific validation
        if light_source == 'UV' and not any(i > 10.0 for i in intensities):
            return False, "WARNING: No substantial peaks found for UV analysis", False
        
        # Normalization scheme validation
        scheme_valid = True
        if normalization_scheme:
            expected = self.expected_schemes.get(light_source)
            if expected and normalization_scheme != expected:
                print(f"      Unexpected scheme: {normalization_scheme}, expected: {expected}")
                scheme_valid = False
            else:
                print(f"      Scheme validated: {normalization_scheme}")
        else:
            print("      WARNING: No normalization scheme provided")
            scheme_valid = False
        
        return True, "Validated: 0-100 scale compatible", scheme_valid
    
    def import_csv_unified(self, file_path, df, light_source, format_type):
        """OPTIMIZED: Unified CSV import for all formats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        imported_count = 0
        base_filename = Path(file_path).stem
        
        # Extract intensities and metadata based on format
        if format_type.startswith('peak_detection'):
            intensities = [float(row['Intensity']) for _, row in df.iterrows() if pd.notna(row['Intensity'])]
            
            # Extract normalization metadata for fixed format
            if format_type == 'peak_detection_fixed':
                first_row = df.iloc[0]
                normalization_scheme = first_row.get('Normalization_Scheme')
                reference_wavelength = first_row.get('Reference_Wavelength')
                csv_light_source = first_row.get('Light_Source', light_source)
            else:
                normalization_scheme = None
                reference_wavelength = None
        else:
            # Structural features
            intensities = []
            for _, row in df.iterrows():
                if 'Intensity' in df.columns and pd.notna(row['Intensity']):
                    intensities.append(float(row['Intensity']))
                else:
                    intensities.append(1.0)
            normalization_scheme = None
            reference_wavelength = None
        
        # Validate data
        intensity_valid, intensity_msg, scheme_valid = self.validate_data(intensities, file_path, light_source, normalization_scheme)
        print(f"      {intensity_msg}")
        
        if not intensity_valid and "ERROR" in intensity_msg:
            print("      SKIPPING file due to validation failure")
            conn.close()
            return 0
        
        min_intensity = min(intensities) if intensities else 0
        max_intensity = max(intensities) if intensities else 0
        
        # Import records based on format
        for idx, row in df.iterrows():
            try:
                if format_type.startswith('peak_detection'):
                    imported_count += self._insert_peak_record(cursor, row, base_filename, light_source, file_path,
                                                             normalization_scheme, reference_wavelength, 
                                                             min_intensity, max_intensity, scheme_valid and intensity_valid,
                                                             format_type == 'peak_detection_legacy')
                else:
                    imported_count += self._insert_structural_record(cursor, row, idx, base_filename, light_source, file_path,
                                                                   df, min_intensity, max_intensity, intensity_valid and "ERROR" not in intensity_msg)
                    
            except Exception as e:
                print(f"   Row {idx} error: {e}")
        
        conn.commit()
        conn.close()
        return imported_count
    
    def _insert_peak_record(self, cursor, row, base_filename, light_source, file_path, normalization_scheme, 
                          reference_wavelength, min_intensity, max_intensity, compatible, is_legacy):
        """OPTIMIZED: Insert peak detection record"""
        cursor.execute("""
            INSERT OR IGNORE INTO structural_features 
            (feature, file, light_source, wavelength, intensity, point_type, feature_group, peak_number, 
             prominence, category, file_source, data_type, normalization_scheme, reference_wavelength, 
             intensity_range_min, intensity_range_max, normalization_compatible)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            f"Peak_{int(row['Peak_Number'])}", base_filename, light_source, float(row['Wavelength_nm']), 
            float(row['Intensity']), 'Peak', str(row['Category']), int(row['Peak_Number']), 
            float(row['Prominence']), str(row['Category']), str(file_path), 'peak_detection',
            normalization_scheme, self.safe_float(reference_wavelength), min_intensity, max_intensity,
            compatible and not is_legacy
        ))
        return 1 if cursor.rowcount > 0 else 0
    
    def _insert_structural_record(self, cursor, row, idx, base_filename, light_source, file_path, df,
                                min_intensity, max_intensity, compatible):
        """OPTIMIZED: Insert structural features record"""
        # Extract wavelength and intensity
        wavelength = None
        intensity = 1.0
        
        for col in ['Crest', 'Wavelength', 'Midpoint']:
            if col in df.columns and pd.notna(row[col]):
                wavelength = float(row[col])
                break
        
        if wavelength is None:
            return 0
        
        if 'Intensity' in df.columns and pd.notna(row['Intensity']):
            intensity = float(row['Intensity'])
        
        cursor.execute("""
            INSERT OR IGNORE INTO structural_features 
            (feature, file, light_source, wavelength, intensity, point_type, feature_group, start_wavelength, 
             end_wavelength, symmetry_ratio, skew_description, midpoint, bottom, file_source, data_type,
             intensity_range_min, intensity_range_max, normalization_compatible)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(row.get('Feature', f'Feature_{idx}')), str(row.get('File', base_filename)), light_source,
            wavelength, intensity, 'Structural', str(row.get('Feature', 'Unknown')),
            self.safe_float(row.get('Start')), self.safe_float(row.get('End')), self.safe_float(row.get('Symmetry_Ratio')),
            str(row.get('Skew_Description', '')), self.safe_float(row.get('Midpoint')), self.safe_float(row.get('Bottom')),
            str(file_path), 'structural_features', min_intensity, max_intensity, compatible
        ))
        return 1 if cursor.rowcount > 0 else 0
    
    def safe_float(self, value):
        """Safely convert to float or None"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return float(value)
        except:
            return None
    
    def scan_subdirectories(self):
        """OPTIMIZED: Scan the 3 specific subdirectories"""
        if not self.base_path.exists():
            print(f"Base directory not found: {self.base_path}")
            return []
        
        print(f"Scanning subdirectories in: {self.base_path}")
        
        all_csv_files = []
        for folder in ['halogen', 'laser', 'uv']:
            folder_path = self.base_path / folder
            
            if folder_path.exists():
                csv_files = list(folder_path.glob("*.csv"))
                all_csv_files.extend(csv_files)
                print(f"   {folder}/: {len(csv_files)} CSV files")
                
                # Show first few files
                for f in csv_files[:2]:
                    print(f"      {f.name}")
                if len(csv_files) > 2:
                    print(f"      ... and {len(csv_files)-2} more")
            else:
                print(f"   {folder}/: Directory not found")
        
        print(f"\nTOTAL: {len(all_csv_files)} CSV files found")
        return all_csv_files
    
    def batch_import_fixed(self):
        """OPTIMIZED: Import with normalization metadata support"""
        all_csv_files = self.scan_subdirectories()
        
        if not all_csv_files:
            print("No CSV files found in the 3 subdirectories")
            return
        
        # Analyze file formats
        format_counts = {'peak_detection_fixed': [], 'peak_detection_legacy': [], 'structural_features': [], 'unknown': []}
        
        for file_path in all_csv_files:
            format_type, df = self.detect_csv_format(file_path)
            if format_type in format_counts:
                format_counts[format_type].append(file_path)
            else:
                format_counts['unknown'].append(file_path)
        
        print(f"\nFIXED FILE FORMAT ANALYSIS:")
        print(f"   Peak detection (FIXED with metadata): {len(format_counts['peak_detection_fixed'])}")
        print(f"   Peak detection (Legacy): {len(format_counts['peak_detection_legacy'])}")
        print(f"   Structural feature files: {len(format_counts['structural_features'])}")
        print(f"   Unknown format files: {len(format_counts['unknown'])}")
        
        if format_counts['peak_detection_legacy']:
            print(f"\n   WARNING: {len(format_counts['peak_detection_legacy'])} legacy files lack normalization metadata")
        
        if format_counts['unknown']:
            print(f"\n   Unknown format files:")
            for f in format_counts['unknown'][:5]:
                print(f"      {f.name}")
        
        if input(f"\nCreate FIXED database and import? (y/N): ").strip().lower() != 'y':
            print("Import cancelled")
            return
        
        # Create FIXED database
        self.create_fresh_database()
        
        # Import all files
        total_imported, files_imported, validation_failures = 0, 0, 0
        
        for format_type, files in format_counts.items():
            if format_type == 'unknown' or not files:
                continue
                
            print(f"\nImporting {format_type} files...")
            for file_path in files:
                light_source = self.detect_light_source_from_folder(file_path)
                format_detected, df = self.detect_csv_format(file_path)
                
                print(f"   {file_path.name} -> {light_source}" + (" (LEGACY)" if format_type == 'peak_detection_legacy' else ""))
                count = self.import_csv_unified(file_path, df, light_source, format_type)
                
                if count > 0:
                    total_imported += count
                    files_imported += 1
                else:
                    validation_failures += 1
        
        print(f"\nFIXED IMPORT COMPLETED:")
        print(f"   Files processed: {files_imported}")
        print(f"   Total records: {total_imported}")
        print(f"   Validation failures: {validation_failures}")
        print(f"   Database: {self.db_path}")
        
        self.view_database_statistics_fixed()
    
    def view_database_statistics_fixed(self):
        """OPTIMIZED: Enhanced statistics with normalization metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Single query for multiple statistics
            stats_query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT file) as unique_files,
                    COUNT(CASE WHEN normalization_scheme IS NOT NULL THEN 1 END) as with_normalization,
                    COUNT(CASE WHEN normalization_compatible = 1 THEN 1 END) as compatible_records,
                    AVG(CASE WHEN intensity_range_max IS NOT NULL THEN intensity_range_max END) as avg_max_intensity,
                    MIN(CASE WHEN intensity_range_max IS NOT NULL THEN intensity_range_max END) as min_max_intensity,
                    MAX(CASE WHEN intensity_range_max IS NOT NULL THEN intensity_range_max END) as max_max_intensity
                FROM structural_features
            """
            
            cursor.execute(stats_query)
            stats = cursor.fetchone()
            total_records, unique_files, with_normalization, compatible_records, avg_max, min_max, max_max = stats
            
            # Get breakdowns
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source ORDER BY light_source")
            by_light = cursor.fetchall()
            
            cursor.execute("SELECT data_type, COUNT(*) FROM structural_features GROUP BY data_type")
            by_type = cursor.fetchall()
            
            cursor.execute("SELECT normalization_scheme, COUNT(*) FROM structural_features WHERE normalization_scheme IS NOT NULL GROUP BY normalization_scheme")
            by_scheme = cursor.fetchall()
            
            conn.close()
            
            # Display results
            print(f"\nFIXED DATABASE STATISTICS:")
            print(f"   Total records: {total_records:,}")
            print(f"   Unique files: {unique_files:,}")
            
            print(f"\n   By Light Source:")
            icons = {'Halogen': 'HALOGEN', 'Laser': 'LASER', 'UV': 'UV', 'Unknown': 'UNKNOWN'}
            for light, count in by_light:
                print(f"      {icons.get(light, 'OTHER')}: {count:,}")
            
            print(f"\n   By Data Type:")
            type_icons = {'peak_detection': 'PEAKS', 'structural_features': 'STRUCT'}
            for dtype, count in by_type:
                print(f"      {type_icons.get(dtype, 'OTHER')}: {count:,}")
            
            # Normalization metadata reporting
            print(f"\n   NORMALIZATION METADATA:")
            print(f"      Records with schemes: {with_normalization:,} / {total_records:,}")
            print(f"      Compatible records: {compatible_records:,} / {total_records:,}")
            
            if by_scheme:
                print(f"\n   By Normalization Scheme:")
                for scheme, count in by_scheme:
                    print(f"      {scheme}: {count:,}")
            
            if avg_max:
                print(f"\n   Intensity Range Analysis:")
                print(f"      Average max intensity: {avg_max:.2f}")
                print(f"      Range of max intensities: {min_max:.2f} - {max_max:.2f}")
                
                if avg_max <= 1.0:
                    print("      WARNING: Data appears to be 0-1 normalized (breaks UV analysis)")
                elif avg_max > 150.0:
                    print("      WARNING: Intensities exceed expected 0-100 range")
                else:
                    print("      VALIDATED: Compatible with 0-100 normalization")
            
        except Exception as e:
            print(f"Error reading FIXED database: {e}")
    
    def append_mode_import(self):
        """OPTIMIZED: Append mode with normalization validation"""
        if not os.path.exists(self.db_path):
            print("No database exists. Use FIXED import first.")
            return
        
        print("APPEND MODE - Adding to existing FIXED database")
        all_csv_files = self.scan_subdirectories()
        
        if not all_csv_files:
            print("No CSV files found")
            return
        
        # Check for existing files
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT file FROM structural_features")
        existing_files = {row[0] for row in cursor.fetchall()}
        conn.close()
        
        new_files = [f for f in all_csv_files if Path(f).stem not in existing_files]
        
        print(f"Found {len(new_files)} new files to import")
        
        if not new_files:
            print("No new files to import")
            return
        
        # Import new files
        validation_failures = 0
        
        for file_path in new_files:
            light_source = self.detect_light_source_from_folder(file_path)
            format_type, df = self.detect_csv_format(file_path)
            
            print(f"   {file_path.name} -> {light_source}")
            
            if format_type in ['peak_detection_fixed', 'peak_detection_legacy', 'structural_features']:
                count = self.import_csv_unified(file_path, df, light_source, format_type)
                if count == 0:
                    validation_failures += 1
            else:
                print("      Unknown format, skipping")
                validation_failures += 1
        
        if validation_failures > 0:
            print(f"\nWARNING: {validation_failures} files had validation failures")
        
        self.view_database_statistics_fixed()
    
    def validate_existing_database(self):
        """OPTIMIZED: Validate existing database for normalization compatibility"""
        if not os.path.exists(self.db_path):
            print("No database found to validate")
            return
        
        print(f"VALIDATING EXISTING DATABASE: {self.db_path}")
        print("=" * 60)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check schema compatibility
            cursor.execute("PRAGMA table_info(structural_features)")
            columns = [col[1] for col in cursor.fetchall()]
            
            norm_columns = ['normalization_scheme', 'reference_wavelength', 'intensity_range_min', 'intensity_range_max']
            missing_columns = [col for col in norm_columns if col not in columns]
            
            if missing_columns:
                print(f"MISSING COLUMNS: {missing_columns}")
                print("This database needs to be recreated with FIXED schema")
                conn.close()
                return False
            
            print("Database schema is compatible with FIXED version")
            
            # Validate data quality
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN intensity_range_max <= 1.0 THEN 1 END) as broken_records,
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN normalization_scheme IS NULL THEN 1 END) as missing_metadata
                FROM structural_features
            """)
            
            broken_records, total_records, missing_metadata = cursor.fetchone()
            
            if broken_records > 0:
                print(f"WARNING: {broken_records}/{total_records} records have 0-1 normalized data")
                print("These records will break UV analysis")
            
            if missing_metadata > 0:
                print(f"INFO: {missing_metadata}/{total_records} records lack normalization metadata")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error validating database: {e}")
            return False

def main_menu():
    """OPTIMIZED: Main menu for FIXED database system"""
    print("FIXED STRUCTURAL DATABASE SYSTEM")
    print("Compatible with 0-100 normalization and metadata")
    print("=" * 60)
    
    db_manager = FixedStructuralDatabaseManager()
    
    menu_options = {
        "1": ("FIXED IMPORT (fresh database with normalization metadata)", lambda: db_manager.batch_import_fixed()),
        "2": ("APPEND MODE (add new files with validation)", lambda: db_manager.append_mode_import()),
        "3": ("View FIXED database statistics", lambda: db_manager.view_database_statistics_fixed()),
        "4": ("Preview files in subdirectories", lambda: preview_files(db_manager)),
        "5": ("Validate existing database compatibility", lambda: db_manager.validate_existing_database()),
        "6": ("Exit", lambda: None)
    }
    
    while True:
        print(f"\nMAIN MENU:")
        for key, (description, _) in menu_options.items():
            print(f"{key}. {description}")
        
        try:
            choice = input("Choice (1-6): ").strip()
            
            if choice == "6":
                print("Goodbye!")
                break
            
            if choice in menu_options:
                _, action = menu_options[choice]
                if action:
                    action()
                    if choice in ['1', '2', '3', '4', '5']:
                        input("\nPress Enter to continue...")
            else:
                print("Invalid choice. Please enter 1-6")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

def preview_files(db_manager):
    """Helper function for file preview"""
    print("\nPREVIEWING SUBDIRECTORY CONTENTS:")
    all_files = db_manager.scan_subdirectories()
    
    if all_files:
        print(f"\nFirst few files by directory:")
        for folder in ['halogen', 'laser', 'uv']:
            folder_files = [f for f in all_files if f'\\{folder}\\' in str(f) or f'/{folder}/' in str(f)]
            print(f"\n{folder}/:")
            for f in folder_files[:3]:
                format_type, _ = db_manager.detect_csv_format(f)
                print(f"   {f.name} ({format_type})")

if __name__ == "__main__":
    main_menu()