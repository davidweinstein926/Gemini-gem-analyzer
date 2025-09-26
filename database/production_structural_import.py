#!/usr/bin/env python3
"""
PRODUCTION STRUCTURAL DATABASE IMPORTER - GEMINI EDITION
🎯 Built on proven SuperSafeGeminiSystem architecture
🔬 Imports structural data from root/data/structural_data/
💎 Handles all light sources (B|H, L, U) with forensic precision
🗄️ Creates/updates database/structural_spectra/gemini_structural.db
📦 Auto-archives imported files to root/data/structural(archive)
⚡ Integrates with main.py Option 6

PRODUCTION WORKFLOW:
1. Fresh structural data → root/data/structural_data/
2. Option 6 imports → database/structural_spectra/gemini_structural.db
3. Successfully imported files → root/data/structural(archive)
"""

import sqlite3
import pandas as pd
import os
import re
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ProductionStructuralImporter:
    """Production structural importer for normal workflow"""
    
    def __init__(self):
        self.project_root = self.find_project_root()
        
        # PRODUCTION LOCATIONS (as requested)
        self.source_dir = self.project_root / "data" / "structural_data"  # Source: fresh data
        self.archive_dir = self.project_root / "data" / "structural(archive)"  # Destination: after import
        self.db_path = self.project_root / "database" / "structural_spectra" / "gemini_structural.db"  # Main database
        self.csv_output_path = self.project_root / "database" / "structural_spectra" / "gemini_structural_unified.csv"
        
        # Backup location for safety
        self.backup_dir = self.project_root / "database" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.import_stats = {
            'total_files_found': 0,
            'files_processed': 0,
            'records_inserted': 0,
            'records_skipped': 0,
            'files_archived': 0,
            'light_sources': {'Halogen': 0, 'Laser': 0, 'UV': 0, 'Unknown': 0},
            'feature_types': {},
            'unique_gems': set()
        }
        
        print("🎯 PRODUCTION STRUCTURAL DATABASE IMPORTER")
        print("=" * 60)
        print("📍 Built on SuperSafeGeminiSystem architecture")
        print(f"📂 Source: {self.source_dir}")
        print(f"🗄️  Database: {self.db_path}")
        print(f"📦 Archive: {self.archive_dir}")
        print("🔬 Production workflow: Import → Archive")
        
    def find_project_root(self) -> Path:
        """Find project root using SuperSafe patterns"""
        print("🔍 Locating project root...")
        
        # Check current working directory first (main.py location)
        cwd = Path.cwd()
        if (cwd / "main.py").exists():
            print(f"✅ Project root: {cwd}")
            return cwd
        
        # Check script location and parents
        current = Path(__file__).parent.absolute()
        for path in [current] + list(current.parents):
            if (path / "main.py").exists():
                print(f"✅ Project root: {path}")
                return path
        
        print(f"⚠️  Using current directory: {cwd}")
        return cwd
    
    def clear_existing_file_data(self, csv_files: List[Path]):
        """FIXED: Clear any existing data for files about to be imported"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='structural_features'")
            if not cursor.fetchone():
                conn.close()
                return  # Table doesn't exist yet
            
            files_to_clear = []
            for csv_file in csv_files:
                file_info = self.parse_structural_filename(csv_file.name)
                files_to_clear.append({
                    'filename': csv_file.name,
                    'gem_id': file_info['gem_id'],
                    'light_source': file_info['light_source']
                })
            
            if files_to_clear:
                print(f"🧹 Clearing existing data for {len(files_to_clear)} files to prevent duplicates...")
                
                cleared_count = 0
                for file_info in files_to_clear:
                    # Delete by gem_id and light_source combination
                    cursor.execute('''
                        DELETE FROM structural_features 
                        WHERE gem_id = ? AND light_source = ?
                    ''', (file_info['gem_id'], file_info['light_source']))
                    
                    deleted = cursor.rowcount
                    if deleted > 0:
                        cleared_count += deleted
                        print(f"   🗑️  Cleared {deleted} existing records for {file_info['gem_id']} ({file_info['light_source']})")
                
                if cleared_count > 0:
                    conn.commit()
                    print(f"✅ Cleared {cleared_count} existing records to prevent duplicates")
                else:
                    print("ℹ️  No existing records found - proceeding with fresh import")
                
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Could not clear existing data: {e}")
            # Continue anyway - this is just duplicate prevention
        """Backup existing database before updating"""
        if not self.db_path.exists():
            print("📊 No existing database - creating new one")
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"gemini_structural_backup_{timestamp}.db"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(self.db_path, backup_path)
            print(f"💾 Backed up existing database: {backup_name}")
            return backup_path
        except Exception as e:
            print(f"⚠️  Could not backup database: {e}")
            return None
    
    def create_or_verify_schema(self):
        """Create or verify unified database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='structural_features'")
            table_exists = cursor.fetchone() is not None
            
            if not table_exists:
                print("🔧 Creating new database schema...")
                
                # Create unified table with all fields from your CSV examples
                cursor.execute('''
                    CREATE TABLE structural_features (
                        -- Primary key and identification
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gem_id TEXT NOT NULL,
                        file_source TEXT NOT NULL,
                        import_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        
                        -- Light source information
                        light_source TEXT NOT NULL,
                        light_source_code TEXT,
                        orientation TEXT DEFAULT 'C',
                        scan_number INTEGER DEFAULT 1,
                        
                        -- Core spectral data (unified from all formats)
                        feature TEXT NOT NULL,
                        wavelength REAL NOT NULL,  -- Unified: Wavelength/Wavelength_nm
                        intensity REAL NOT NULL,
                        point_type TEXT NOT NULL,
                        feature_group TEXT NOT NULL,
                        processing TEXT,
                        
                        -- Spectral analysis fields
                        snr REAL,
                        feature_key TEXT,
                        
                        -- Halogen-specific fields (23 columns format)
                        baseline_used REAL,
                        norm_factor REAL,
                        normalization_method TEXT,
                        reference_wavelength_used REAL,
                        symmetry_ratio REAL,
                        skew_description TEXT,
                        width_nm REAL,
                        intensity_range_min REAL,
                        intensity_range_max REAL,
                        
                        -- UV-specific fields (11 columns format)
                        peak_number INTEGER,
                        prominence REAL,
                        category TEXT,
                        detection_method TEXT,
                        
                        -- Unified normalization fields
                        normalization_scheme TEXT,
                        reference_wavelength REAL,
                        
                        -- File structure information
                        directory_structure TEXT,
                        output_location TEXT,
                        
                        -- Data quality and forensics
                        data_quality TEXT DEFAULT 'Good',
                        analysis_date DATE,
                        temporal_sequence INTEGER DEFAULT 1,
                        
                        -- Constraints for data integrity
                        UNIQUE(gem_id, light_source, wavelength, feature, point_type),
                        CHECK(wavelength > 0),
                        CHECK(intensity >= 0),
                        CHECK(light_source IN ('Halogen', 'Laser', 'UV', 'Unknown'))
                    )
                ''')
                
                # Create performance indexes
                indexes = [
                    ("idx_gem_lookup", "gem_id, light_source"),
                    ("idx_wavelength_analysis", "wavelength, light_source"),
                    ("idx_feature_classification", "feature_group, point_type"),
                    ("idx_temporal_analysis", "gem_id, analysis_date, temporal_sequence"),
                    ("idx_light_source_features", "light_source, feature"),
                    ("idx_file_source", "file_source")
                ]
                
                for idx_name, columns in indexes:
                    cursor.execute(f"CREATE INDEX {idx_name} ON structural_features({columns})")
                
                # Create analytical views
                cursor.execute('''
                    CREATE VIEW light_source_summary AS
                    SELECT 
                        light_source,
                        COUNT(*) as total_records,
                        COUNT(DISTINCT gem_id) as unique_gems,
                        COUNT(DISTINCT feature_group) as feature_groups,
                        MIN(wavelength) as min_wavelength,
                        MAX(wavelength) as max_wavelength,
                        AVG(intensity) as avg_intensity
                    FROM structural_features 
                    GROUP BY light_source
                ''')
                
                cursor.execute('''
                    CREATE VIEW gem_analysis_summary AS
                    SELECT 
                        gem_id,
                        COUNT(DISTINCT light_source) as light_sources_used,
                        COUNT(*) as total_features,
                        MIN(analysis_date) as first_analysis,
                        MAX(analysis_date) as latest_analysis,
                        MAX(temporal_sequence) as total_analyses,
                        GROUP_CONCAT(DISTINCT light_source) as light_source_list
                    FROM structural_features 
                    GROUP BY gem_id
                    HAVING light_sources_used >= 2
                ''')
                
                print("✅ New database schema created")
            else:
                print("✅ Existing database schema verified")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Schema creation/verification error: {e}")
            return False
    
    def parse_structural_filename(self, filename: str) -> Dict[str, str]:
        """Parse structural filename using your established patterns"""
        base_name = Path(filename).stem
        original_name = base_name
        
        # Remove timestamps if present
        base_name = re.sub(r'_\d{8}_\d{6}$', '', base_name)
        
        # Extract analysis date if present
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        date_match = re.search(r'_(\d{8})_', original_name)
        if date_match:
            try:
                date_str = date_match.group(1)
                dt = datetime.strptime(date_str, '%Y%m%d')
                analysis_date = dt.strftime('%Y-%m-%d')
            except:
                pass
        
        # Determine light source using your patterns
        light_source = 'Unknown'
        light_code = ''
        
        # Check for explicit light source markers
        if 'halogen' in filename.lower() or '_h_' in filename.lower():
            light_source, light_code = 'Halogen', 'B'
        elif 'laser' in filename.lower() or '_l_' in filename.lower():
            light_source, light_code = 'Laser', 'L'
        elif 'uv' in filename.lower() or '_u_' in filename.lower():
            light_source, light_code = 'UV', 'U'
        else:
            # Standard format parsing (your BC1, LC1, UC1 pattern)
            pattern = r'^(.+?)([BLU])([CP]?)(\d+)'
            match = re.match(pattern, base_name, re.IGNORECASE)
            if match:
                prefix, light_char, orientation, scan = match.groups()
                light_mapping = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                light_source = light_mapping.get(light_char.upper(), 'Unknown')
                light_code = light_char.upper()
                base_name = prefix
        
        # Extract gem ID (remove prefixes, keep numeric/identifier part)
        gem_id = re.sub(r'^[A-Za-z]*', '', base_name)
        if not gem_id or not re.search(r'\d', gem_id):
            gem_id = base_name
        
        return {
            'gem_id': gem_id,
            'light_source': light_source,
            'light_source_code': light_code,
            'analysis_date': analysis_date,
            'file_source': filename,
            'orientation': 'C',
            'scan_number': 1
        }
    
    def detect_csv_format(self, df: pd.DataFrame, filename: str) -> str:
        """Detect CSV format using your established patterns"""
        columns = set(df.columns)
        
        # UV Auto Detection format (11 columns)
        if 'Peak_Number' in columns and 'Wavelength_nm' in columns and 'Prominence' in columns:
            return 'uv_auto'
        
        # Halogen format (23 columns) - has symmetry_ratio, skew_description
        if 'Symmetry_Ratio' in columns and 'Skew_Description' in columns:
            return 'halogen_structural'
        
        # Laser format (20 columns) - has standard fields but not UV/Halogen specific
        if 'Feature' in columns and 'Point_Type' in columns and 'SNR' in columns:
            if 'Prominence' not in columns and 'Symmetry_Ratio' not in columns:
                return 'laser_structural'
        
        # Legacy format
        if 'feature' in columns and 'point_type' in columns:
            return 'legacy_structural'
        
        print(f"⚠️  Unknown format in {filename}: {list(columns)[:5]}...")
        return 'unknown'
    
    def import_uv_auto_format(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import UV auto detection format (11 columns) - FIXED: Prevents nested loop bug"""
        print(f"   📊 UV Auto Detection format ({len(df)} peaks)")
        
        inserted, skipped = 0, 0
        duplicates_prevented = 0
        
        # FIXED: Process DataFrame row by row with explicit indexing
        for row_idx in range(len(df)):
            try:
                row = df.iloc[row_idx]  # Get specific row by index
                
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'light_source': 'UV',
                    'light_source_code': 'U',
                    'analysis_date': file_info['analysis_date'],
                    
                    # Core data
                    'feature': f"Peak_{int(row['Peak_Number'])}",
                    'wavelength': float(row['Wavelength_nm']),
                    'intensity': float(row['Intensity']),
                    'point_type': 'Peak',
                    'feature_group': 'UV_Auto_Detection',
                    
                    # UV-specific
                    'peak_number': int(row['Peak_Number']),
                    'prominence': float(row.get('Prominence', 0)),
                    'category': str(row.get('Category', 'Unknown')),
                    'detection_method': str(row.get('Detection_Method', 'UV_Auto')),
                    
                    # Normalization
                    'normalization_scheme': str(row.get('Normalization_Scheme', '')),
                    'reference_wavelength': float(row.get('Reference_Wavelength', 811)),
                    
                    # Structure
                    'directory_structure': str(row.get('Directory_Structure', '')),
                    'output_location': str(row.get('Output_Location', ''))
                }
                
                # FIXED: Single insert per row with explicit debug tracking
                print(f"     Processing row {row_idx + 1}/{len(df)}: Peak {int(row['Peak_Number'])} at {float(row['Wavelength_nm'])}nm")
                
                if self.insert_unified_record(cursor, data):
                    inserted += 1
                    print(f"     ✓ Inserted peak {int(row['Peak_Number'])}")
                else:
                    duplicates_prevented += 1
                    print(f"     - Duplicate prevented for peak {int(row['Peak_Number'])}")
                
            except Exception as e:
                print(f"     ❌ Row {row_idx + 1} error: {e}")
                skipped += 1
        
        print(f"   📈 SUMMARY: {inserted} inserted, {skipped} skipped, {duplicates_prevented} duplicates prevented")
        
        if duplicates_prevented > 0:
            print(f"   🛡️  Prevented {duplicates_prevented} duplicates")
        
        return inserted, skipped
    
    def import_halogen_structural_format(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import Halogen structural format (23 columns) - FIXED: Prevents nested loop bug"""
        print(f"   🔥 Halogen Structural format ({len(df)} features)")
        
        inserted, skipped = 0, 0
        duplicates_prevented = 0
        
        # FIXED: Process DataFrame row by row with explicit indexing to prevent nested loops
        for row_idx in range(len(df)):
            try:
                row = df.iloc[row_idx]  # Get specific row by index
                
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'light_source': file_info.get('light_source', 'Halogen'),
                    'light_source_code': file_info.get('light_source_code', 'B'),
                    'analysis_date': file_info['analysis_date'],
                    
                    # Core data
                    'feature': str(row.get('Feature', 'Unknown')),
                    'wavelength': float(row.get('Wavelength', 0)),
                    'intensity': float(row.get('Intensity', 0)),
                    'point_type': str(row.get('Point_Type', 'Unknown')),
                    'feature_group': str(row.get('Feature_Group', 'Halogen_Manual')),
                    'processing': str(row.get('Processing', '')),
                    
                    # Analysis data
                    'snr': self.safe_float(row.get('SNR')),
                    'feature_key': str(row.get('Feature_Key', '')),
                    
                    # Halogen-specific analysis
                    'baseline_used': self.safe_float(row.get('Baseline_Used')),
                    'norm_factor': self.safe_float(row.get('Norm_Factor')),
                    'normalization_method': str(row.get('Normalization_Method', '')),
                    'reference_wavelength_used': self.safe_float(row.get('Reference_Wavelength_Used')),
                    'symmetry_ratio': self.safe_float(row.get('Symmetry_Ratio')),
                    'skew_description': str(row.get('Skew_Description', '')),
                    'width_nm': self.safe_float(row.get('Width_nm')),
                    'intensity_range_min': self.safe_float(row.get('Intensity_Range_Min')),
                    'intensity_range_max': self.safe_float(row.get('Intensity_Range_Max')),
                    
                    # Unified normalization
                    'normalization_scheme': str(row.get('Normalization_Scheme', '')),
                    'reference_wavelength': self.safe_float(row.get('Reference_Wavelength')),
                    
                    # Structure
                    'directory_structure': str(row.get('Directory_Structure', '')),
                    'output_location': str(row.get('Output_Location', ''))
                }
                
                # FIXED: Single insert per row with explicit debug tracking
                feature_name = str(row.get('Feature', 'Unknown'))
                wavelength = float(row.get('Wavelength', 0))
                print(f"     Processing row {row_idx + 1}/{len(df)}: {feature_name} at {wavelength}nm")
                
                if self.insert_unified_record(cursor, data):
                    inserted += 1
                    print(f"     ✓ Inserted {feature_name}")
                else:
                    duplicates_prevented += 1
                    print(f"     - Duplicate prevented for {feature_name}")
                
            except Exception as e:
                print(f"     ❌ Row {row_idx + 1} error: {e}")
                skipped += 1
        
        print(f"   📈 SUMMARY: {inserted} inserted, {skipped} skipped, {duplicates_prevented} duplicates prevented")
        
        if duplicates_prevented > 0:
            print(f"   🛡️  Prevented {duplicates_prevented} duplicates")
        
        return inserted, skipped
    
    def import_laser_structural_format(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import Laser structural format (20 columns) - FIXED: Proper duplicate tracking"""
        print(f"   ⚡ Laser Structural format ({len(df)} features)")
        
        inserted, skipped = 0, 0
        duplicates_prevented = 0
        
        for _, row in df.iterrows():
            try:
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'light_source': file_info.get('light_source', 'Laser'),
                    'light_source_code': file_info.get('light_source_code', 'L'),
                    'analysis_date': file_info['analysis_date'],
                    
                    # Core data
                    'feature': str(row.get('Feature', 'Unknown')),
                    'wavelength': float(row.get('Wavelength', 0)),
                    'intensity': float(row.get('Intensity', 0)),
                    'point_type': str(row.get('Point_Type', 'Unknown')),
                    'feature_group': str(row.get('Feature_Group', 'Laser_Manual')),
                    'processing': str(row.get('Processing', '')),
                    
                    # Analysis data
                    'snr': self.safe_float(row.get('SNR')),
                    'feature_key': str(row.get('Feature_Key', '')),
                    
                    # Normalization (Laser has these but not Halogen-specific fields)
                    'baseline_used': self.safe_float(row.get('Baseline_Used')),
                    'norm_factor': self.safe_float(row.get('Norm_Factor')),
                    'normalization_method': str(row.get('Normalization_Method', '')),
                    'reference_wavelength_used': self.safe_float(row.get('Reference_Wavelength_Used')),
                    
                    # Unified normalization
                    'normalization_scheme': str(row.get('Normalization_Scheme', '')),
                    'reference_wavelength': self.safe_float(row.get('Reference_Wavelength')),
                    
                    # Structure
                    'directory_structure': str(row.get('Directory_Structure', '')),
                    'output_location': str(row.get('Output_Location', '')),
                    
                    # Intensity range (Laser specific)
                    'intensity_range_min': self.safe_float(row.get('Intensity_Range_Min')),
                    'intensity_range_max': self.safe_float(row.get('Intensity_Range_Max'))
                }
                
                # FIXED: Check return value to track actual inserts
                if self.insert_unified_record(cursor, data):
                    inserted += 1
                else:
                    duplicates_prevented += 1
                
            except Exception as e:
                print(f"     Laser row error: {e}")
                skipped += 1
        
        if duplicates_prevented > 0:
            print(f"   🛡️  Prevented {duplicates_prevented} duplicates")
        
        return inserted, skipped
    
    def safe_float(self, value) -> Optional[float]:
        """Safely convert to float"""
        if pd.isna(value) or value == '':
            return None
        try:
            return float(value)
        except:
            return None
    
    def insert_unified_record(self, cursor: sqlite3.Cursor, data: Dict):
        """Insert record into unified database - FIXED: Better duplicate prevention"""
        # Build dynamic SQL for all possible fields
        fields = []
        values = []
        placeholders = []
        
        for key, value in data.items():
            if value is not None and value != '':
                fields.append(key)
                values.append(value)
                placeholders.append('?')
        
        # FIXED: Use INSERT OR IGNORE to prevent duplicates completely
        # This will silently skip if the UNIQUE constraint would be violated
        sql = f'''
            INSERT OR IGNORE INTO structural_features 
            ({', '.join(fields)}) 
            VALUES ({', '.join(placeholders)})
        '''
        
        # Execute and check if row was actually inserted
        cursor.execute(sql, values)
        
        # DEBUG: Check if the insert actually worked
        if cursor.rowcount == 0:
            # Row was not inserted (duplicate detected)
            print(f"     DEBUG: Duplicate prevented - {data.get('gem_id', 'Unknown')} {data.get('light_source', 'Unknown')} {data.get('wavelength', 'Unknown')}nm")
            return False
        else:
            return True
    
    def archive_imported_files(self, successful_files: List[Path]) -> int:
        """Archive successfully imported files to structural(archive)"""
        if not successful_files:
            return 0
        
        print(f"\n📦 Archiving {len(successful_files)} imported files...")
        
        archived_count = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in successful_files:
            try:
                # Create archive filename with timestamp
                archive_name = f"{file_path.stem}_archived_{timestamp}{file_path.suffix}"
                archive_path = self.archive_dir / archive_name
                
                # Move file to archive (not copy)
                shutil.move(str(file_path), str(archive_path))
                archived_count += 1
                print(f"   📦 Archived: {file_path.name} → {archive_name}")
                
            except Exception as e:
                print(f"   ❌ Archive error for {file_path.name}: {e}")
        
        if archived_count > 0:
            print(f"✅ Successfully archived {archived_count} files")
            print(f"📁 Archive location: {self.archive_dir}")
        
        self.import_stats['files_archived'] = archived_count
        return archived_count
    
    def run_production_import(self) -> bool:
        """Execute production import process - COMPREHENSIVE DUPLICATE FIX"""
        print("🚀 STARTING PRODUCTION STRUCTURAL IMPORT")
        print("=" * 50)
        
        # Verify source directory
        if not self.source_dir.exists():
            print(f"❌ Source directory not found: {self.source_dir}")
            print("💡 Expected location: root/data/structural_data")
            return False
        
        # Find CSV files
        csv_files = list(self.source_dir.glob("*.csv"))
        self.import_stats['total_files_found'] = len(csv_files)
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.source_dir}")
            print("💡 Use Option 2 to mark structural features first")
            return False
        
        print(f"📊 Found {len(csv_files)} structural files to import")
        
        # Backup existing database
        self.backup_existing_database()
        
        # Create or verify schema
        if not self.create_or_verify_schema():
            return False
        
        # COMPREHENSIVE FIX: Clear ALL existing data and import fresh
        self.clear_all_existing_data()
        
        # Process files with comprehensive debugging
        print(f"\n📋 Processing {len(csv_files)} files with full debugging...")
        
        successful_files = []
        conn = None
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN IMMEDIATE")  # FIXED: Use IMMEDIATE transaction to prevent conflicts
            cursor = conn.cursor()
            
            # FIXED: Process files one at a time with explicit isolation
            for file_index, csv_file in enumerate(csv_files):
                print(f"\n" + "="*60)
                print(f"📄 PROCESSING FILE {file_index + 1}/{len(csv_files)}: {csv_file.name}")
                print("="*60)
                
                # Check if file has already been processed in this session
                if csv_file in successful_files:
                    print(f"   ⚠️  SKIPPING: File already processed in this session")
                    continue
                
                # Parse filename ONCE
                file_info = self.parse_structural_filename(csv_file.name)
                print(f"   📝 Parsed info: Gem={file_info['gem_id']}, Light={file_info['light_source']}")
                
                # Read CSV ONCE and validate
                try:
                    df = pd.read_csv(csv_file)
                    original_row_count = len(df)
                    print(f"   📊 CSV contains {original_row_count} rows")
                    
                    if original_row_count == 0:
                        print(f"   ⚠️  SKIPPING: Empty CSV file")
                        continue
                        
                except Exception as e:
                    print(f"   ❌ CSV read error: {e}")
                    continue
                
                # Detect format ONCE
                csv_format = self.detect_csv_format(df, csv_file.name)
                print(f"   🔍 Detected format: {csv_format}")
                
                # Get initial database count
                cursor.execute("SELECT COUNT(*) FROM structural_features")
                initial_db_count = cursor.fetchone()[0]
                print(f"   📊 Database records before import: {initial_db_count}")
                
                # SINGLE IMPORT CALL - No loops, no repetition
                print(f"   🔄 Starting SINGLE import operation...")
                
                try:
                    if csv_format == 'uv_auto':
                        inserted, skipped = self.import_uv_auto_format_SINGLE(df, file_info, cursor)
                    elif csv_format == 'halogen_structural':
                        inserted, skipped = self.import_halogen_structural_format_SINGLE(df, file_info, cursor)
                    elif csv_format == 'laser_structural':
                        inserted, skipped = self.import_laser_structural_format_SINGLE(df, file_info, cursor)
                    else:
                        print(f"   ❌ Unknown format - skipping")
                        continue
                        
                    print(f"   ✅ Import function returned: inserted={inserted}, skipped={skipped}")
                    
                except Exception as e:
                    print(f"   ❌ Import function error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Verify database count after import
                cursor.execute("SELECT COUNT(*) FROM structural_features")
                final_db_count = cursor.fetchone()[0]
                actual_added = final_db_count - initial_db_count
                
                print(f"   📊 Database records after import: {final_db_count}")
                print(f"   📊 Actually added to database: {actual_added}")
                print(f"   📊 Expected vs Actual: {original_row_count} rows → {actual_added} records")
                
                # Validation check
                if actual_added == original_row_count:
                    print(f"   ✅ PERFECT: Added exactly {actual_added} records as expected")
                    successful_files.append(csv_file)
                    self.import_stats['files_processed'] += 1
                    self.import_stats['records_inserted'] += actual_added
                    self.import_stats['light_sources'][file_info['light_source']] += actual_added
                    self.import_stats['unique_gems'].add(file_info['gem_id'])
                elif actual_added > original_row_count:
                    print(f"   🚨 DUPLICATE BUG DETECTED: Added {actual_added} but expected {original_row_count}")
                    print(f"   🚨 This indicates the duplication bug is still active!")
                    # Don't mark as successful
                else:
                    print(f"   ⚠️  WARNING: Added {actual_added} but expected {original_row_count}")
                    print(f"   ⚠️  Some records may have been skipped or failed")
                    if actual_added > 0:
                        successful_files.append(csv_file)
                        self.import_stats['files_processed'] += 1
                        self.import_stats['records_inserted'] += actual_added
                
                print(f"   🏁 Completed processing {csv_file.name}")
            
            # Commit transaction once at the end
            print(f"\n💾 Committing transaction to database...")
            conn.commit()
            print(f"✅ All changes committed successfully")
            
            # Archive successful files
            if successful_files:
                self.archive_imported_files(successful_files)
            
            # Generate report
            self.generate_final_report()
            
            return len(successful_files) > 0
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR in run_production_import: {e}")
            if conn:
                try:
                    conn.rollback()
                    print(f"🔄 Transaction rolled back")
                except:
                    pass
            import traceback
            traceback.print_exc()
            return False
        finally:
            if conn:
                conn.close()
    
    def clear_all_existing_data(self):
        """Clear all existing data to start fresh and prevent any duplication"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='structural_features'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM structural_features")
                existing_count = cursor.fetchone()[0]
                
                if existing_count > 0:
                    print(f"🧹 Clearing all {existing_count} existing records for fresh import...")
                    cursor.execute("DELETE FROM structural_features")
                    conn.commit()
                    print(f"✅ Database cleared - starting with clean slate")
                else:
                    print("ℹ️  Database is already empty")
            
            conn.close()
            
        except Exception as e:
            print(f"⚠️  Could not clear database: {e}")
    
    def import_uv_auto_format_SINGLE(self, df: pd.DataFrame, file_info: Dict, cursor: sqlite3.Cursor) -> Tuple[int, int]:
        """Import UV format with SINGLE processing guarantee"""
        print(f"   📊 Processing {len(df)} UV peaks - SINGLE PASS ONLY")
        
        inserted, skipped = 0, 0
        
        # Process each row exactly once
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            try:
                data = {
                    'gem_id': file_info['gem_id'],
                    'file_source': file_info['file_source'],
                    'light_source': 'UV',
                    'feature': f"Peak_{int(row['Peak_Number'])}",
                    'wavelength': float(row['Wavelength_nm']),
                    'intensity': float(row['Intensity']),
                    'point_type': 'Peak',
                    'feature_group': 'UV_Auto_Detection',
                    'peak_number': int(row['Peak_Number']),
                    'analysis_date': file_info['analysis_date']
                }
                
                cursor.execute('''
                    INSERT INTO structural_features 
                    (gem_id, file_source, light_source, feature, wavelength, intensity, 
                     point_type, feature_group, peak_number, analysis_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['gem_id'], data['file_source'], data['light_source'], 
                     data['feature'], data['wavelength'], data['intensity'],
                     data['point_type'], data['feature_group'], data['peak_number'], 
                     data['analysis_date']))
                
                inserted += 1
                print(f"     Row {idx+1}: Inserted Peak_{int(row['Peak_Number'])} at {float(row['Wavelength_nm'])}nm")
                
            except Exception as e:
                print(f"     Row {idx+1} error: {e}")
                skipped += 1
        
        return inserted, skipped Find CSV files
        csv_files = list(self.source_dir.glob("*.csv"))
        self.import_stats['total_files_found'] = len(csv_files)
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.source_dir}")
            print("💡 Use Option 2 to mark structural features first")
            return False
        
        print(f"📊 Found {len(csv_files)} structural files to import")
        
        # Backup existing database
        self.backup_existing_database()
        
        # Create or verify schema
        if not self.create_or_verify_schema():
            return False
        
        # FIXED: Clear any existing data for these files first
        self.clear_existing_file_data(csv_files)
        
        # Process files
        print(f"\n📋 Processing {len(csv_files)} files...")
        
        successful_files = []
        conn = None
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("BEGIN TRANSACTION")  # FIXED: Explicit transaction control
            cursor = conn.cursor()
            
            for i, csv_file in enumerate(csv_files, 1):
                print(f"\n📄 File {i}/{len(csv_files)}: {csv_file.name}")
                
                # FIXED: Check if this file was already processed in this session
                if csv_file in successful_files:
                    print(f"   ⚠️  File already processed in this session - skipping")
                    continue
                
                # Parse filename
                file_info = self.parse_structural_filename(csv_file.name)
                self.import_stats['unique_gems'].add(file_info['gem_id'])
                
                print(f"   Gem: {file_info['gem_id']}")
                print(f"   Light: {file_info['light_source']}")
                print(f"   Date: {file_info['analysis_date']}")
                
                # Read and detect format
                try:
                    df = pd.read_csv(csv_file)
                    print(f"   📊 CSV rows to process: {len(df)}")
                    csv_format = self.detect_csv_format(df, csv_file.name)
                    print(f"   Format: {csv_format}")
                    
                    # FIXED: Add file processing marker to prevent reprocessing
                    file_marker = f"PROCESSING_{csv_file.name}_{file_info['light_source']}"
                    
                    # Import based on format
                    if csv_format == 'uv_auto':
                        inserted, skipped = self.import_uv_auto_format(df, file_info, cursor)
                    elif csv_format == 'halogen_structural':
                        inserted, skipped = self.import_halogen_structural_format(df, file_info, cursor)
                    elif csv_format == 'laser_structural':
                        inserted, skipped = self.import_laser_structural_format(df, file_info, cursor)
                    else:
                        print(f"   ❌ Unknown format - skipping")
                        continue
                    
                    # Update statistics
                    self.import_stats['files_processed'] += 1
                    self.import_stats['records_inserted'] += inserted
                    self.import_stats['records_skipped'] += skipped
                    self.import_stats['light_sources'][file_info['light_source']] += inserted
                    
                    if inserted > 0:
                        successful_files.append(csv_file)
                        print(f"   ✅ Inserted: {inserted}, Skipped: {skipped}")
                        print(f"   📊 Expected: {len(df)} rows, Actual: {inserted} records")
                    else:
                        print(f"   ⚠️  No records inserted: {skipped} skipped")
                    
                except Exception as e:
                    print(f"   ❌ File error: {e}")
                    print(f"   DEBUG: Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # FIXED: Commit transaction once at the end
            print(f"\n💾 Committing all changes to database...")
            conn.commit()
            print(f"✅ Database transaction committed successfully")
            
            # Archive successfully imported files
            if successful_files:
                self.archive_imported_files(successful_files)
            
            # Generate final report
            self.generate_final_report()
            
            return len(successful_files) > 0
            
        except Exception as e:
            print(f"❌ Import process error: {e}")
            if conn:
                try:
                    conn.rollback()
                    print(f"🔄 Database transaction rolled back")
                except:
                    pass
            import traceback
            traceback.print_exc()
            return False
        finally:
            if conn:
                conn.close()
                print(f"🔌 Database connection closed")
    
    def generate_final_report(self):
        """Generate comprehensive final import report"""
        print(f"\n🎉 PRODUCTION IMPORT COMPLETED!")
        print("=" * 60)
        
        # Basic statistics
        print(f"📊 IMPORT STATISTICS:")
        print(f"   Files found: {self.import_stats['total_files_found']}")
        print(f"   Files processed: {self.import_stats['files_processed']}")
        print(f"   Files archived: {self.import_stats['files_archived']}")
        print(f"   Records inserted: {self.import_stats['records_inserted']:,}")
        print(f"   Records skipped: {self.import_stats['records_skipped']:,}")
        print(f"   Unique gems: {len(self.import_stats['unique_gems'])}")
        
        # Light source breakdown
        print(f"\n💡 BY LIGHT SOURCE:")
        for source, count in self.import_stats['light_sources'].items():
            if count > 0:
                print(f"   {source}: {count:,} records")
        
        # Database validation
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT gem_id) FROM structural_features")
            unique_gems = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT light_source) FROM structural_features")
            light_sources = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"\n🗄️  FINAL DATABASE STATUS:")
            print(f"   Total records: {total_records:,}")
            print(f"   Unique gems: {unique_gems}")
            print(f"   Light sources: {light_sources}")
            print(f"   Database: {self.db_path}")
            
            # Export CSV if requested
            if total_records > 0:
                self.export_unified_csv()
            
            print(f"\n✅ PRODUCTION WORKFLOW COMPLETE!")
            print(f"🔄 Fresh data imported → Database updated → Files archived")
            
        except Exception as e:
            print(f"⚠️  Validation error: {e}")
    
    def export_unified_csv(self):
        """Export unified database to CSV (optional compatibility)"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    gem_id, light_source, wavelength, intensity, feature, 
                    feature_group, point_type, snr, analysis_date,
                    normalization_scheme, reference_wavelength,
                    symmetry_ratio, width_nm, prominence, category,
                    file_source, import_timestamp
                FROM structural_features
                ORDER BY gem_id, light_source, wavelength
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Ensure CSV output directory exists
            self.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(self.csv_output_path, index=False)
            print(f"📄 CSV export: {self.csv_output_path}")
            print(f"   Records: {len(df):,}")
            
        except Exception as e:
            print(f"⚠️  CSV export error: {e}")


def production_structural_import():
    """Main function for integration with main.py Option 6 - DEBUG VERSION"""
    print("🎯 PRODUCTION STRUCTURAL DATABASE IMPORT - DEBUG MODE")
    print("Built on SuperSafeGeminiSystem architecture")
    print("=" * 60)
    
    try:
        print("DEBUG: Creating importer instance...")
        importer = ProductionStructuralImporter()
        
        print("DEBUG: About to call run_production_import()...")
        success = importer.run_production_import()
        
        print(f"DEBUG: Import result = {success}")
        
        if success:
            print(f"\n🎉 SUCCESS! Production import completed!")
            print("🗄️  database/structural_spectra/gemini_structural.db updated")
            print("📦 Fresh data files archived to structural(archive)")
            print("💎 All light sources (Halogen/Laser/UV) unified")
            print("⚡ Ready for structural analysis workflows")
        else:
            print(f"\n❌ Import failed. Check error messages above.")
        
        return success
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in production_structural_import: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("DEBUG: Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    production_structural_import()