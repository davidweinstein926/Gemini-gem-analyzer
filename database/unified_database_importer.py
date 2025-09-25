#!/usr/bin/env python3
"""
UNIFIED STRUCTURAL DATABASE IMPORTER
Handles ALL structural data formats in ONE database:
- Manual marking files (Halogen, Laser structural features)
- UV Auto detection files (automated peak detection)
- Legacy format support
- Single source of truth
"""

import sqlite3
import pandas as pd
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

class UnifiedStructuralImporter:
    """Unified importer for all structural data formats"""
    
    def __init__(self):
        self.project_root = self.find_project_root()
        self.db_path = self.project_root / "database" / "structural_spectra" / "unified_structural_data.db"
        self.source_dir = self.project_root / "data" / "structural_data"
        
        # Create database directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def find_project_root(self) -> Path:
        """Find project root"""
        current = Path.cwd()
        if (current / "main.py").exists():
            return current
        
        # Check parent directories
        for path in list(current.parents):
            if (path / "main.py").exists():
                return path
        
        return current
    
    def create_unified_schema(self):
        """Create the unified database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Drop old tables if they exist
            cursor.execute("DROP TABLE IF EXISTS unified_structural_features")
            cursor.execute("DROP VIEW IF EXISTS current_gem_analysis")
            cursor.execute("DROP VIEW IF EXISTS detection_method_summary")
            
            # Create unified table
            cursor.execute('''
                CREATE TABLE unified_structural_features (
                    -- Primary identification
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gem_id TEXT NOT NULL,
                    file_source TEXT NOT NULL,
                    analysis_date DATE NOT NULL,
                    analysis_time TIME,
                    
                    -- Light source and measurement
                    light_source TEXT NOT NULL,
                    light_source_code TEXT,
                    orientation TEXT,
                    scan_number INTEGER DEFAULT 1,
                    
                    -- Core spectral data
                    wavelength REAL NOT NULL,
                    intensity REAL NOT NULL,
                    
                    -- Feature classification
                    feature_name TEXT NOT NULL,
                    feature_type TEXT NOT NULL,
                    feature_group TEXT NOT NULL,
                    point_type TEXT NOT NULL,
                    
                    -- Manual marking fields
                    start_wavelength REAL,
                    end_wavelength REAL,
                    midpoint REAL,
                    bottom REAL,
                    width_nm REAL,
                    height REAL,
                    
                    -- Auto detection fields
                    peak_number INTEGER,
                    prominence REAL,
                    category TEXT,
                    
                    -- Advanced analysis
                    snr REAL,
                    symmetry_ratio REAL,
                    skew_description TEXT,
                    
                    -- Normalization
                    normalization_scheme TEXT,
                    reference_wavelength REAL,
                    norm_factor REAL,
                    baseline_used REAL,
                    processing TEXT,
                    
                    -- Metadata
                    detection_method TEXT,
                    data_quality TEXT DEFAULT 'Good',
                    notes TEXT,
                    import_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Constraints
                    UNIQUE(gem_id, light_source, wavelength, feature_name, analysis_date),
                    CHECK(wavelength > 0),
                    CHECK(intensity >= 0),
                    CHECK(light_source IN ('Halogen', 'Laser', 'UV'))
                )
            ''')
            
            # Create performance indexes
            cursor.execute("CREATE INDEX idx_gem_lookup ON unified_structural_features(gem_id, light_source)")
            cursor.execute("CREATE INDEX idx_wavelength_search ON unified_structural_features(wavelength)")
            cursor.execute("CREATE INDEX idx_feature_analysis ON unified_structural_features(feature_type, light_source)")
            cursor.execute("CREATE INDEX idx_date_analysis ON unified_structural_features(analysis_date, gem_id)")
            
            # Create helpful views
            cursor.execute('''
                CREATE VIEW current_gem_analysis AS
                SELECT gem_id, light_source, MAX(analysis_date) as latest_date, 
                       COUNT(*) as feature_count, detection_method
                FROM unified_structural_features 
                GROUP BY gem_id, light_source
            ''')
            
            cursor.execute('''
                CREATE VIEW detection_method_summary AS
                SELECT detection_method, light_source, COUNT(*) as record_count,
                       COUNT(DISTINCT gem_id) as unique_gems
                FROM unified_structural_features 
                GROUP BY detection_method, light_source
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Unified database schema created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Schema creation error: {e}")
            return False
    
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse filename to extract metadata"""
        base_name = Path(filename).stem
        
        # Extract timestamp if present
        timestamp_match = re.search(r'_(\d{8})_(\d{6})', base_name)
        if timestamp_match:
            date_str, time_str = timestamp_match.groups()
            analysis_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            analysis_time = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            base_name = re.sub(r'_\d{8}_\d{6}', '', base_name)
        else:
            analysis_date = datetime.now().strftime('%Y-%m-%d')
            analysis_time = datetime.now().strftime('%H:%M:%S')
        
        # Determine light source
        light_source = 'Unknown'
        light_code = ''
        
        if '_halogen_' in filename.lower():
            light_source, light_code = 'Halogen', 'B'
        elif '_laser_' in filename.lower():
            light_source, light_code = 'Laser', 'L'
        elif '_uv_' in filename.lower():
            light_source, light_code = 'UV', 'U'
        else:
            # Standard format parsing
            pattern = r'^(.+?)([BLU])([CP]?)(\d+)'
            match = re.match(pattern, base_name, re.IGNORECASE)
            if match:
                base_name, light_char, orientation, scan = match.groups()
                light_mapping = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                light_source = light_mapping.get(light_char.upper(), 'Unknown')
                light_code = light_char.upper()
        
        # Extract gem ID (remove prefixes)
        gem_id = re.sub(r'^[A-Za-z]*', '', base_name)
        if not gem_id:
            gem_id = base_name
        
        return {
            'gem_id': gem_id,
            'light_source': light_source,
            'light_source_code': light_code,
            'analysis_date': analysis_date,
            'analysis_time': analysis_time,
            'file_source': filename
        }
    
    def detect_file_format(self, df: pd.DataFrame) -> str:
        """Detect what type of structural file this is"""
        columns = set(df.columns)
        
        if 'Peak_Number' in columns and 'Wavelength_nm' in columns:
            return 'uv_auto'
        elif 'Feature' in columns and 'Point_Type' in columns:
            return 'manual_structural'
        elif 'feature' in columns and 'point_type' in columns:
            return 'legacy_structural'
        else:
            return 'unknown'
    
    def import_uv_auto_file(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import UV auto detection file"""
        print(f"   📊 Processing UV Auto Detection file")
        
        inserted, skipped = 0, 0
        
        for _, row in df.iterrows():
            try:
                # Map UV auto columns to unified schema
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'analysis_date': file_info['analysis_date'],
                    'analysis_time': file_info['analysis_time'],
                    'light_source': 'UV',
                    'light_source_code': 'U',
                    'wavelength': float(row['Wavelength_nm']),
                    'intensity': float(row['Intensity']),
                    'feature_name': f"Peak_{int(row['Peak_Number'])}",
                    'feature_type': 'Peak',
                    'feature_group': 'UV_Auto_Detection',
                    'point_type': 'Peak',
                    'peak_number': int(row['Peak_Number']),
                    'prominence': float(row.get('Prominence', 0)),
                    'category': str(row.get('Category', 'Unknown')),
                    'normalization_scheme': str(row.get('Normalization_Scheme', '')),
                    'reference_wavelength': float(row.get('Reference_Wavelength', 811)),
                    'detection_method': str(row.get('Detection_Method', 'GeminiUVPeakDetector_Auto'))
                }
                
                self.insert_record(cursor, data)
                inserted += 1
                
            except Exception as e:
                print(f"     Row error: {e}")
                skipped += 1
                
        return inserted, skipped
    
    def import_manual_structural_file(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import manual structural marking file"""
        print(f"   📊 Processing Manual Structural file")
        
        inserted, skipped = 0, 0
        
        for _, row in df.iterrows():
            try:
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'analysis_date': file_info['analysis_date'],
                    'analysis_time': file_info['analysis_time'],
                    'light_source': file_info['light_source'],
                    'light_source_code': file_info['light_source_code'],
                    'wavelength': float(row.get('Wavelength', row.get('wavelength', 0))),
                    'intensity': float(row.get('Intensity', row.get('intensity', 0))),
                    'feature_name': str(row.get('Feature', row.get('feature', 'Unknown'))),
                    'feature_type': self.extract_feature_type(row.get('Feature', row.get('feature', 'Unknown'))),
                    'feature_group': str(row.get('Feature_Group', row.get('feature_group', 'Manual_Marking'))),
                    'point_type': str(row.get('Point_Type', row.get('point_type', 'Unknown'))),
                    'start_wavelength': self.safe_float(row.get('start_wavelength')),
                    'end_wavelength': self.safe_float(row.get('end_wavelength')),
                    'width_nm': self.safe_float(row.get('Width_nm', row.get('width_nm'))),
                    'snr': self.safe_float(row.get('SNR', row.get('snr'))),
                    'skew_description': str(row.get('Skew_Description', row.get('skew_description', ''))),
                    'detection_method': 'Manual_Marking'
                }
                
                self.insert_record(cursor, data)
                inserted += 1
                
            except Exception as e:
                print(f"     Row error: {e}")
                skipped += 1
                
        return inserted, skipped
    
    def extract_feature_type(self, feature_name: str) -> str:
        """Extract feature type from feature name"""
        feature_name = str(feature_name).lower()
        
        if 'peak' in feature_name:
            return 'Peak'
        elif 'mound' in feature_name:
            return 'Mound'
        elif 'trough' in feature_name:
            return 'Trough'
        elif 'valley' in feature_name:
            return 'Valley'
        elif 'shoulder' in feature_name:
            return 'Shoulder'
        elif 'plateau' in feature_name:
            return 'Plateau'
        else:
            return 'Unknown'
    
    def safe_float(self, value) -> Optional[float]:
        """Safely convert to float"""
        try:
            return float(value) if pd.notna(value) else None
        except:
            return None
    
    def insert_record(self, cursor: sqlite3.Cursor, data: Dict):
        """Insert record into unified database"""
        cursor.execute('''
            INSERT OR REPLACE INTO unified_structural_features (
                gem_id, file_source, analysis_date, analysis_time,
                light_source, light_source_code, wavelength, intensity,
                feature_name, feature_type, feature_group, point_type,
                peak_number, prominence, category, start_wavelength, end_wavelength,
                width_nm, snr, skew_description, normalization_scheme,
                reference_wavelength, detection_method
            ) VALUES (
                :gem_id, :file_source, :analysis_date, :analysis_time,
                :light_source, :light_source_code, :wavelength, :intensity,
                :feature_name, :feature_type, :feature_group, :point_type,
                :peak_number, :prominence, :category, :start_wavelength, :end_wavelength,
                :width_nm, :snr, :skew_description, :normalization_scheme,
                :reference_wavelength, :detection_method
            )
        ''', data)
    
    def run_unified_import(self):
        """Run the unified import process"""
        print("UNIFIED STRUCTURAL DATABASE IMPORTER")
        print("="*50)
        
        # Create schema
        if not self.create_unified_schema():
            return False
        
        # Find files to import
        csv_files = list(self.source_dir.glob("*.csv"))
        if not csv_files:
            print(f"❌ No CSV files found in {self.source_dir}")
            return False
        
        print(f"✅ Found {len(csv_files)} files to import")
        
        # Import all files
        total_inserted, total_skipped = 0, 0
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            for csv_file in csv_files:
                print(f"\n📄 Processing: {csv_file.name}")
                
                df = pd.read_csv(csv_file)
                file_info = self.parse_filename(csv_file.name)
                file_format = self.detect_file_format(df)
                
                print(f"   Format: {file_format}")
                print(f"   Gem: {file_info['gem_id']}")
                print(f"   Light: {file_info['light_source']}")
                print(f"   Records: {len(df)}")
                
                if file_format == 'uv_auto':
                    inserted, skipped = self.import_uv_auto_file(df, file_info, cursor)
                elif file_format in ['manual_structural', 'legacy_structural']:
                    inserted, skipped = self.import_manual_structural_file(df, file_info, cursor)
                else:
                    print(f"   ❌ Unknown format - skipping")
                    continue
                
                total_inserted += inserted
                total_skipped += skipped
                print(f"   ✅ Imported: {inserted}, Skipped: {skipped}")
            
            conn.commit()
            
            print(f"\n🎉 UNIFIED IMPORT COMPLETE")
            print(f"   Total imported: {total_inserted}")
            print(f"   Total skipped: {total_skipped}")
            
            # Show database stats
            cursor.execute("SELECT COUNT(*) FROM unified_structural_features")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT detection_method, COUNT(*) FROM unified_structural_features GROUP BY detection_method")
            method_counts = cursor.fetchall()
            
            cursor.execute("SELECT light_source, COUNT(*) FROM unified_structural_features GROUP BY light_source")  
            light_counts = cursor.fetchall()
            
            print(f"\n📊 DATABASE STATISTICS:")
            print(f"   Total records: {total_records}")
            print(f"   By detection method:")
            for method, count in method_counts:
                print(f"     {method}: {count}")
            print(f"   By light source:")
            for light, count in light_counts:
                print(f"     {light}: {count}")
            
        except Exception as e:
            print(f"❌ Import error: {e}")
            return False
        finally:
            conn.close()
        
        return True

def main():
    """Main entry point"""
    importer = UnifiedStructuralImporter()
    success = importer.run_unified_import()
    
    if success:
        print("\n✅ Unified database created successfully!")
        print("   Single source of truth for all structural data")
        print("   Supports both manual marking and auto detection")
        print("   No more dual database confusion!")
    else:
        print("\n❌ Import failed")

if __name__ == "__main__":
    main()