#!/usr/bin/env python3
"""
PERFECT STRUCTURAL IMPORTER V2
- Prevents duplicate file imports (same filename = skip)
- Clean rebuild from scratch
- Preserves exact CSV data (including negative intensities)
- Validates: 58BC1_TS1 + 58BC2_TS1 + 58BC1_TS2 = OK
- Rejects: 58BC1_TS1 + 58BC1_TS1 = DUPLICATE (second one skipped)
"""

import sqlite3
import pandas as pd
import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

class PerfectImporterV2:
    """Perfect importer with file-level duplicate detection"""
    
    def __init__(self):
        self.project_root = self.find_project_root()
        
        # Paths
        self.source_dir = self.project_root / "data" / "structural_data"
        self.archive_dir = self.project_root / "data" / "structural(archive)"
        self.db_path = self.project_root / "database" / "structural_spectra" / "gemini_structural.db"
        self.backup_dir = self.project_root / "database" / "backups"
        
        # Ensure directories
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'files_found': 0,
            'files_processed': 0,
            'files_skipped_duplicate': 0,
            'records_inserted': 0,
            'records_rejected': 0,
            'by_light_source': {'Halogen': 0, 'Laser': 0, 'UV': 0},
            'by_gem': {}
        }
        
        # Track imported files
        self.imported_files = set()
        
        print("PERFECT STRUCTURAL IMPORTER V2")
        print("=" * 70)
        print("No duplicate files allowed (same filename = skip)")
        print(f"Source: {self.source_dir}")
        print(f"Database: {self.db_path}")
        
    def find_project_root(self) -> Path:
        """Find project root"""
        cwd = Path.cwd()
        if (cwd / "main.py").exists():
            return cwd
        
        current = Path(__file__).parent.absolute()
        for path in [current] + list(current.parents):
            if (path / "main.py").exists():
                return path
        
        return cwd
    
    def backup_database(self):
        """Backup existing database"""
        if not self.db_path.exists():
            print("No existing database - creating fresh")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"gemini_structural_backup_{timestamp}.db"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(self.db_path, backup_path)
            print(f"Backed up to: {backup_name}")
            return backup_path
        except Exception as e:
            print(f"Backup failed: {e}")
            return None
    
    def create_fresh_database(self):
        """Create fresh database with file_source in UNIQUE constraint"""
        
        if self.db_path.exists():
            self.db_path.unlink()
            print("Deleted old database")
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("Creating fresh database schema...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table - KEY CHANGE: file_source in UNIQUE constraint
        cursor.execute('''
            CREATE TABLE structural_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gem_id TEXT NOT NULL,
                file_source TEXT NOT NULL,
                import_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                light_source TEXT NOT NULL CHECK(light_source IN ('Halogen', 'Laser', 'UV')),
                light_source_code TEXT,
                orientation TEXT DEFAULT 'C',
                scan_number INTEGER DEFAULT 1,
                
                feature TEXT NOT NULL,
                wavelength REAL NOT NULL CHECK(wavelength > 0),
                intensity REAL NOT NULL,
                point_type TEXT NOT NULL,
                feature_group TEXT NOT NULL,
                processing TEXT,
                
                snr REAL,
                feature_key TEXT,
                
                baseline_used REAL,
                norm_factor REAL,
                normalization_method TEXT,
                reference_wavelength_used REAL,
                symmetry_ratio REAL,
                skew_description TEXT,
                width_nm REAL,
                intensity_range_min REAL,
                intensity_range_max REAL,
                
                peak_number INTEGER,
                prominence REAL,
                category TEXT,
                detection_method TEXT,
                
                normalization_scheme TEXT,
                reference_wavelength REAL,
                
                directory_structure TEXT,
                output_location TEXT,
                
                data_quality TEXT DEFAULT 'Good',
                analysis_date DATE,
                temporal_sequence INTEGER DEFAULT 1,
                
                UNIQUE(file_source, wavelength, feature, point_type)
            )
        ''')
        
        # Create indexes
        indexes = [
            ("idx_file_source", "file_source"),
            ("idx_gem_lookup", "gem_id, light_source"),
            ("idx_wavelength", "wavelength, light_source"),
            ("idx_analysis_date", "gem_id, analysis_date")
        ]
        
        for idx_name, columns in indexes:
            cursor.execute(f"CREATE INDEX {idx_name} ON structural_features({columns})")
        
        conn.commit()
        conn.close()
        
        print("Fresh database created with file-level duplicate protection")
    
    def check_file_already_imported(self, filename: str) -> bool:
        """Check if file already imported in this session or in database"""
        
        # Check session tracking
        if filename in self.imported_files:
            return True
        
        # Check database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM structural_features WHERE file_source = ?", (filename,))
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def parse_filename(self, filename: str) -> Dict:
        """Parse filename with robust light source detection"""
        base_name = Path(filename).stem
        
        # Extract analysis date
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        date_match = re.search(r'_(\d{8})_', filename)
        if date_match:
            try:
                date_str = date_match.group(1)
                dt = datetime.strptime(date_str, '%Y%m%d')
                analysis_date = dt.strftime('%Y-%m-%d')
            except:
                pass
        
        # Determine light source - FIXED: Check gem ID pattern first
        light_source = None
        light_code = None
        
        # Method 1: Look for light code in gem ID pattern (e.g., 140LC2, 199BP1, 58UC1)
        # Pattern: digits followed by B/L/U followed by letter/digit
        pattern = re.search(r'\d+([BLU])([CP]?\d*)', base_name, re.IGNORECASE)
        if pattern:
            light_char = pattern.group(1).upper()
            if light_char == 'B':
                light_source, light_code = 'Halogen', 'B'
            elif light_char == 'L':
                light_source, light_code = 'Laser', 'L'
            elif light_char == 'U':
                light_source, light_code = 'UV', 'U'
        
        # Method 2: Fallback to keyword detection
        if not light_source:
            fname_lower = filename.lower()
            if 'halogen' in fname_lower:
                light_source, light_code = 'Halogen', 'B'
            elif 'laser' in fname_lower:
                light_source, light_code = 'Laser', 'L'
            elif 'uv' in fname_lower:
                light_source, light_code = 'UV', 'U'
            else:
                # Default
                light_source, light_code = 'UV', 'U'
        
        # Extract gem_id
        gem_id = base_name
        gem_id = re.sub(r'_(uv|halogen|laser).*$', '', gem_id, flags=re.IGNORECASE)
        gem_id = re.sub(r'_(structural|auto).*$', '', gem_id, flags=re.IGNORECASE)
        gem_id = re.sub(r'_\d{8}_\d{6}$', '', gem_id)
        
        match = re.match(r'^([A-Za-z]*)(\d+)', gem_id)
        if match:
            prefix, number = match.groups()
            gem_id = number if number else gem_id
        
        return {
            'gem_id': gem_id,
            'light_source': light_source,
            'light_source_code': light_code,
            'analysis_date': analysis_date,
            'file_source': filename
        }
    
    def detect_format(self, df: pd.DataFrame) -> str:
        """Detect CSV format"""
        columns = set(df.columns)
        
        if 'Peak_Number' in columns and 'Wavelength_nm' in columns:
            return 'uv_auto'
        elif 'Symmetry_Ratio' in columns:
            return 'halogen_structural'
        elif 'Feature' in columns and 'SNR' in columns:
            return 'laser_structural'
        else:
            return 'unknown'
    
    def validate_and_fix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Validate data - NO auto-fixing, preserve exact CSV data"""
        fixed_count = 0
        
        # Just report statistics, don't modify data
        if 'Intensity' in df.columns:
            neg_mask = df['Intensity'] < 0
            neg_count = neg_mask.sum()
            if neg_count > 0:
                print(f"      Note: {neg_count} negative intensities (preserved as-is)")
        
        return df, fixed_count
    
    def import_uv_auto(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import UV auto format"""
        inserted, rejected = 0, 0
        
        for _, row in df.iterrows():
            try:
                wavelength = float(row['Wavelength_nm'])
                intensity = float(row['Intensity'])
                peak_num = int(row['Peak_Number'])
                
                if wavelength <= 0:
                    rejected += 1
                    continue
                
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'light_source': 'UV',
                    'light_source_code': 'U',
                    'analysis_date': file_info['analysis_date'],
                    'feature': f"Peak_{peak_num}",
                    'wavelength': wavelength,
                    'intensity': intensity,
                    'point_type': 'Peak',
                    'feature_group': 'UV_Auto_Detection',
                    'peak_number': peak_num,
                    'prominence': float(row.get('Prominence', 0)),
                    'category': str(row.get('Category', 'Unknown')),
                    'detection_method': str(row.get('Detection_Method', 'UV_Auto'))
                }
                
                if self.insert_record(cursor, data):
                    inserted += 1
                else:
                    rejected += 1
                    
            except Exception:
                rejected += 1
        
        return inserted, rejected
    
    def import_halogen(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import Halogen format"""
        inserted, rejected = 0, 0
        
        for _, row in df.iterrows():
            try:
                wavelength = float(row.get('Wavelength', 0))
                intensity = float(row.get('Intensity', 0))
                
                if wavelength <= 0:
                    rejected += 1
                    continue
                
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'light_source': 'Halogen',
                    'light_source_code': 'B',
                    'analysis_date': file_info['analysis_date'],
                    'feature': str(row.get('Feature', 'Unknown')),
                    'wavelength': wavelength,
                    'intensity': intensity,
                    'point_type': str(row.get('Point_Type', 'Unknown')),
                    'feature_group': str(row.get('Feature_Group', 'Halogen_Manual')),
                    'processing': str(row.get('Processing', '')),
                    'snr': self.safe_float(row.get('SNR')),
                    'baseline_used': self.safe_float(row.get('Baseline_Used')),
                    'norm_factor': self.safe_float(row.get('Norm_Factor')),
                    'symmetry_ratio': self.safe_float(row.get('Symmetry_Ratio')),
                    'skew_description': str(row.get('Skew_Description', '')),
                    'width_nm': self.safe_float(row.get('Width_nm'))
                }
                
                if self.insert_record(cursor, data):
                    inserted += 1
                else:
                    rejected += 1
                    
            except Exception:
                rejected += 1
        
        return inserted, rejected
    
    def import_laser(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import Laser format"""
        inserted, rejected = 0, 0
        
        for _, row in df.iterrows():
            try:
                wavelength = float(row.get('Wavelength', 0))
                intensity = float(row.get('Intensity', 0))
                
                if wavelength <= 0:
                    rejected += 1
                    continue
                
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'light_source': 'Laser',
                    'light_source_code': 'L',
                    'analysis_date': file_info['analysis_date'],
                    'feature': str(row.get('Feature', 'Unknown')),
                    'wavelength': wavelength,
                    'intensity': intensity,
                    'point_type': str(row.get('Point_Type', 'Unknown')),
                    'feature_group': str(row.get('Feature_Group', 'Laser_Manual')),
                    'processing': str(row.get('Processing', '')),
                    'snr': self.safe_float(row.get('SNR')),
                    'baseline_used': self.safe_float(row.get('Baseline_Used')),
                    'norm_factor': self.safe_float(row.get('Norm_Factor'))
                }
                
                if self.insert_record(cursor, data):
                    inserted += 1
                else:
                    rejected += 1
                    
            except Exception:
                rejected += 1
        
        return inserted, rejected
    
    def safe_float(self, value):
        """Safely convert to float"""
        if pd.isna(value) or value == '':
            return None
        try:
            return float(value)
        except:
            return None
    
    def insert_record(self, cursor: sqlite3.Cursor, data: Dict) -> bool:
        """Insert record"""
        fields = []
        values = []
        placeholders = []
        
        for key, value in data.items():
            if value is not None and value != '':
                fields.append(key)
                values.append(value)
                placeholders.append('?')
        
        sql = f'''
            INSERT INTO structural_features 
            ({', '.join(fields)}) 
            VALUES ({', '.join(placeholders)})
        '''
        
        try:
            cursor.execute(sql, values)
            return True
        except Exception:
            return False
    
    def process_all_files(self):
        """Process all CSV files with duplicate detection"""
        csv_files = sorted(self.source_dir.glob("*.csv"))
        self.stats['files_found'] = len(csv_files)
        
        if not csv_files:
            print(f"\nNo CSV files found in {self.source_dir}")
            return False
        
        print(f"\nFound {len(csv_files)} files")
        print("=" * 70)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        successful_files = []
        
        for i, csv_file in enumerate(csv_files, 1):
            print(f"\n[{i}/{len(csv_files)}] {csv_file.name}")
            
            # Check for duplicate
            if self.check_file_already_imported(csv_file.name):
                print(f"   SKIPPED: Duplicate file (already imported)")
                self.stats['files_skipped_duplicate'] += 1
                continue
            
            file_info = self.parse_filename(csv_file.name)
            print(f"   Gem: {file_info['gem_id']} | Light: {file_info['light_source']} | Date: {file_info['analysis_date']}")
            
            try:
                df = pd.read_csv(csv_file)
                csv_format = self.detect_format(df)
                print(f"   Format: {csv_format} | Rows: {len(df)}")
                
                # Validate and fix
                df, _ = self.validate_and_fix(df)
                
                # Import based on format
                if csv_format == 'uv_auto':
                    inserted, rejected = self.import_uv_auto(df, file_info, cursor)
                elif csv_format == 'halogen_structural':
                    inserted, rejected = self.import_halogen(df, file_info, cursor)
                elif csv_format == 'laser_structural':
                    inserted, rejected = self.import_laser(df, file_info, cursor)
                else:
                    print(f"   Unknown format - skipping")
                    continue
                
                # Update statistics
                self.stats['files_processed'] += 1
                self.stats['records_inserted'] += inserted
                self.stats['records_rejected'] += rejected
                self.stats['by_light_source'][file_info['light_source']] += inserted
                
                gem_id = file_info['gem_id']
                if gem_id not in self.stats['by_gem']:
                    self.stats['by_gem'][gem_id] = {'Halogen': 0, 'Laser': 0, 'UV': 0}
                self.stats['by_gem'][gem_id][file_info['light_source']] += inserted
                
                # Mark as imported
                self.imported_files.add(csv_file.name)
                
                # Report
                if inserted > 0:
                    print(f"   Inserted: {inserted} records", end='')
                    if rejected > 0:
                        print(f" | Rejected: {rejected}")
                    else:
                        print()
                    successful_files.append(csv_file)
                else:
                    print(f"   No records inserted (rejected: {rejected})")
                
            except Exception as e:
                print(f"   Error: {e}")
        
        conn.commit()
        conn.close()
        
        # Archive successful files
        if successful_files:
            self.archive_files(successful_files)
        
        return True
    
    def archive_files(self, files: List[Path]):
        """Archive imported files"""
        print(f"\nArchiving {len(files)} imported files...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in files:
            try:
                archive_name = f"{file_path.stem}_imported_{timestamp}{file_path.suffix}"
                archive_path = self.archive_dir / archive_name
                shutil.move(str(file_path), str(archive_path))
            except Exception as e:
                print(f"   Could not archive {file_path.name}: {e}")
    
    def generate_report(self):
        """Generate final report"""
        print("\n" + "=" * 70)
        print("IMPORT COMPLETE")
        print("=" * 70)
        
        print(f"\nSUMMARY:")
        print(f"   Files found: {self.stats['files_found']}")
        print(f"   Files processed: {self.stats['files_processed']}")
        print(f"   Files skipped (duplicates): {self.stats['files_skipped_duplicate']}")
        print(f"   Records inserted: {self.stats['records_inserted']:,}")
        print(f"   Records rejected: {self.stats['records_rejected']:,}")
        
        print(f"\nBY LIGHT SOURCE:")
        for source, count in self.stats['by_light_source'].items():
            if count > 0:
                print(f"   {source}: {count:,} records")
        
        print(f"\nBY GEM:")
        for gem_id in sorted(self.stats['by_gem'].keys(), key=lambda x: int(x) if x.isdigit() else x):
            gem_stats = self.stats['by_gem'][gem_id]
            total = sum(gem_stats.values())
            details = [f"{k}={v}" for k, v in gem_stats.items() if v > 0]
            print(f"   Gem {gem_id}: {total} total ({', '.join(details)})")
        
        # Verify database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT gem_id) FROM structural_features")
            unique_gems = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT file_source) FROM structural_features")
            unique_files = cursor.fetchone()[0]
            
            print(f"\nDATABASE VERIFICATION:")
            print(f"   Total records: {total:,}")
            print(f"   Unique gems: {unique_gems}")
            print(f"   Unique files: {unique_files}")
            print(f"   Location: {self.db_path}")
            
            conn.close()
            
        except Exception as e:
            print(f"\nVerification error: {e}")
    
    def run(self):
        """Execute import"""
        print("\nSTARTING IMPORT")
        print("=" * 70)
        
        # Backup
        self.backup_database()
        
        # Create fresh database
        self.create_fresh_database()
        
        # Process files
        success = self.process_all_files()
        
        if success:
            self.generate_report()
        
        return success


def perfect_import_v2():
    """Main entry point"""
    try:
        importer = PerfectImporterV2()
        success = importer.run()
        
        if success:
            print("\nImport completed successfully")
        else:
            print("\nImport failed")
        
        return success
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    perfect_import_v2()