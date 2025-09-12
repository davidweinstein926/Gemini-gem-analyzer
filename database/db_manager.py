#!/usr/bin/env python3
"""
INTEGRATED DATABASE MANAGEMENT SYSTEM - COMPLETE CSV-TO-SQLITE SOLUTION
Enhanced system that automatically detects, imports, and manages structural data from
manual analyzers with full normalization metadata, audio feedback, and advanced features.

Major Features:
- Automatic CSV detection from manual analyzers
- Multi-spectral import with B/L/U light source detection
- Complete normalization metadata preservation
- Audio feedback for import operations
- Real-time database statistics and validation
- Enhanced comparison tools and relative height measurements
- Seamless integration with main analysis system
"""

import sqlite3
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import sys
from collections import defaultdict
import threading

# Audio support for feedback
try:
    import winsound
    HAS_AUDIO = True
except ImportError:
    try:
        import pygame
        pygame.mixer.init()
        HAS_AUDIO = True
    except ImportError:
        HAS_AUDIO = False

class IntegratedDatabaseManager:
    """Complete database management system with enhanced features"""
    
    def __init__(self, db_path="multi_structural_gem_data.db"):
        self.db_path = db_path
        self.base_path = Path(r"c:\users\david\gemini sp10 structural data")
        
        # Enhanced configuration
        self.audio_enabled = True
        self.auto_scan_enabled = True
        self.validation_enabled = True
        self.backup_enabled = True
        
        # Expected normalization schemes for validation
        self.expected_schemes = {
            'UV': 'UV_811nm_15000_to_100',
            'Halogen': 'Halogen_650nm_50000_to_100',
            'Laser': 'Laser_Max_50000_to_100'
        }
        
        # Statistical tracking
        self.import_stats = {
            'total_files_processed': 0,
            'total_records_imported': 0,
            'validation_failures': 0,
            'successful_imports': 0,
            'duplicate_files_skipped': 0
        }
        
        # Initialize system
        self.init_complete_database()
        self.setup_monitoring()
        
        print("🗄️ Integrated Database Management System initialized")
        if HAS_AUDIO and self.audio_enabled:
            self.play_bleep("completion")
    
    def play_bleep(self, bleep_type="standard", duration=200):
        """Play audio feedback for database operations"""
        if not self.audio_enabled or not HAS_AUDIO:
            return
        
        try:
            freq_map = {
                "import": 800,
                "validation": 600,
                "completion": 1000,
                "error": 400,
                "batch_complete": 1200
            }
            
            freq = freq_map.get(bleep_type, 800)
            
            if 'winsound' in sys.modules:
                winsound.Beep(freq, duration)
            elif 'pygame' in sys.modules:
                sample_rate = 22050
                frames = int(duration * sample_rate / 1000)
                arr = np.zeros(frames)
                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * freq * i / sample_rate)
                arr = (arr * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(duration / 1000)
                
        except Exception as e:
            print(f"Audio error: {e}")
    
    def init_complete_database(self):
        """Initialize complete database schema with all enhanced features"""
        backup_created = False
        
        if os.path.exists(self.db_path) and self.backup_enabled:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"multi_structural_gem_data_BACKUP_{timestamp}.db"
            
            try:
                import shutil
                shutil.copy2(self.db_path, backup_name)
                backup_created = True
                print(f"✅ Database backup created: {backup_name}")
            except Exception as e:
                print(f"⚠️ Backup failed: {e}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Complete enhanced schema
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
                    
                    -- Enhanced detection and analysis fields
                    peak_number INTEGER,
                    prominence REAL,
                    category TEXT,
                    start_wavelength REAL,
                    end_wavelength REAL,
                    symmetry_ratio REAL,
                    skew_description TEXT,
                    midpoint REAL,
                    bottom REAL,
                    
                    -- Enhanced processing metadata
                    processing TEXT,
                    baseline_used REAL,
                    norm_factor REAL,
                    snr REAL,
                    width_nm REAL,
                    height REAL,
                    local_slope REAL,
                    slope_r_squared REAL,
                    
                    -- Complete normalization metadata
                    normalization_scheme TEXT,
                    reference_wavelength REAL,
                    laser_normalization_wavelength REAL,
                    intensity_range_min REAL,
                    intensity_range_max REAL,
                    normalization_compatible BOOLEAN DEFAULT 1,
                    normalization_method TEXT,
                    reference_wavelength_used REAL,
                    feature_key TEXT,
                    
                    -- Enhanced metadata and tracking
                    timestamp TEXT DEFAULT (datetime('now')),
                    file_source TEXT,
                    data_type TEXT,
                    gem_id TEXT,
                    analysis_session_id TEXT,
                    import_batch_id TEXT,
                    validation_status TEXT DEFAULT 'pending',
                    quality_score REAL,
                    
                    -- Constraints
                    UNIQUE(file, feature, wavelength, point_type)
                )
            """)
            
            # Enhanced indexes for performance
            enhanced_indexes = [
                ("idx_file", "file"),
                ("idx_light_source", "light_source"),
                ("idx_wavelength", "wavelength"),
                ("idx_feature_group", "feature_group"),
                ("idx_normalization_scheme", "normalization_scheme"),
                ("idx_gem_id", "gem_id"),
                ("idx_timestamp", "timestamp"),
                ("idx_validation_status", "validation_status"),
                ("idx_quality_score", "quality_score"),
                ("idx_data_type", "data_type"),
                ("idx_batch_id", "import_batch_id"),
                ("idx_intensity_range", "intensity_range_min, intensity_range_max"),
                ("idx_normalization_compatible", "normalization_compatible")
            ]
            
            for idx_name, columns in enhanced_indexes:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON structural_features({columns})")
            
            # Create metadata tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS import_batches (
                    batch_id TEXT PRIMARY KEY,
                    timestamp TEXT DEFAULT (datetime('now')),
                    files_processed INTEGER DEFAULT 0,
                    records_imported INTEGER DEFAULT 0,
                    validation_failures INTEGER DEFAULT 0,
                    processing_time_seconds REAL,
                    status TEXT DEFAULT 'in_progress',
                    notes TEXT
                )
            """)
            
            # Create validation log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS validation_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    file_path TEXT,
                    light_source TEXT,
                    validation_type TEXT,
                    status TEXT,
                    message TEXT,
                    batch_id TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            
            print("✅ Enhanced database schema initialized")
            if backup_created:
                print("📦 Previous database backed up safely")
            
        except Exception as e:
            print(f"❌ Database initialization error: {e}")
            if HAS_AUDIO and self.audio_enabled:
                self.play_bleep("error")
    
    def setup_monitoring(self):
        """Setup file system monitoring for automatic import"""
        if not self.auto_scan_enabled:
            return
        
        # Ensure monitoring directories exist
        for light_source in ['halogen', 'laser', 'uv']:
            source_dir = self.base_path / light_source
            source_dir.mkdir(parents=True, exist_ok=True)
        
        print("📡 File system monitoring setup complete")
    
    def detect_light_source_from_path(self, file_path):
        """Enhanced light source detection from file path and content"""
        path_str = str(file_path).lower()
        
        # Path-based detection (most reliable)
        if 'halogen' in path_str or '/b/' in path_str or '\\b\\' in path_str:
            return 'Halogen'
        elif 'laser' in path_str or '/l/' in path_str or '\\l\\' in path_str:
            return 'Laser'
        elif 'uv' in path_str or '/u/' in path_str or '\\u\\' in path_str:
            return 'UV'
        
        # Filename-based detection
        filename = Path(file_path).stem.lower()
        if any(pattern in filename for pattern in ['_b', 'bc1', 'bp1', '_halogen']):
            return 'Halogen'
        elif any(pattern in filename for pattern in ['_l', 'lc1', 'lp1', '_laser']):
            return 'Laser'
        elif any(pattern in filename for pattern in ['_u', 'uc1', 'up1', '_uv']):
            return 'UV'
        
        # Content-based detection (analyze first few rows)
        try:
            df = pd.read_csv(file_path, nrows=5)
            if 'Light_Source' in df.columns:
                light_source = df['Light_Source'].iloc[0]
                return light_source
        except:
            pass
        
        return 'Unknown'
    
    def detect_csv_format(self, file_path):
        """Enhanced CSV format detection with validation"""
        try:
            df = pd.read_csv(file_path, nrows=10)
            columns = set(df.columns)
            
            # Enhanced format detection
            if {'Peak_Number', 'Wavelength_nm', 'Intensity', 'Prominence', 'Category'}.issubset(columns):
                # Check for normalization metadata
                if {'Normalization_Scheme', 'Reference_Wavelength', 'Light_Source'}.issubset(columns):
                    return 'peak_detection_enhanced', df
                else:
                    return 'peak_detection_legacy', df
                    
            elif {'Feature', 'File'}.issubset(columns) and any(col in columns for col in ['Crest', 'Wavelength', 'Start', 'End']):
                # Check for enhanced features
                if any(col in columns for col in ['Normalization_Scheme', 'Enhanced_Analysis', 'Light_Source']):
                    return 'manual_structural_enhanced', df
                else:
                    return 'manual_structural_legacy', df
            
            return 'unknown', df
            
        except Exception as e:
            return 'error', None
    
    def validate_data_quality(self, df, file_path, light_source, format_type):
        """Enhanced data quality validation with detailed reporting"""
        validation_results = {
            'status': 'passed',
            'warnings': [],
            'errors': [],
            'quality_score': 1.0,
            'compatible': True
        }
        
        try:
            # Basic data validation
            if df.empty:
                validation_results['errors'].append("Empty dataset")
                validation_results['status'] = 'failed'
                return validation_results
            
            # Intensity validation
            if 'Intensity' in df.columns:
                intensities = df['Intensity'].dropna()
                
                if len(intensities) == 0:
                    validation_results['errors'].append("No valid intensity values")
                    validation_results['status'] = 'failed'
                    return validation_results
                
                min_int = intensities.min()
                max_int = intensities.max()
                
                # Enhanced intensity range validation
                if max_int <= 1.0:
                    validation_results['errors'].append("0-1 normalized data detected (incompatible with UV analysis)")
                    validation_results['compatible'] = False
                    validation_results['quality_score'] *= 0.3
                elif max_int > 150.0:
                    validation_results['warnings'].append(f"High maximum intensity: {max_int:.1f}")
                    validation_results['quality_score'] *= 0.9
                
                if min_int < -10.0:
                    validation_results['warnings'].append(f"Very negative minimum intensity: {min_int:.1f}")
                    validation_results['quality_score'] *= 0.8
            
            # Wavelength validation
            wavelength_cols = [col for col in df.columns if 'wavelength' in col.lower() or col in ['Start', 'End', 'Crest', 'Midpoint']]
            if wavelength_cols:
                for col in wavelength_cols:
                    wl_values = df[col].dropna()
                    if len(wl_values) > 0:
                        if wl_values.min() < 200 or wl_values.max() > 3000:
                            validation_results['warnings'].append(f"Unusual wavelength range in {col}: {wl_values.min():.1f}-{wl_values.max():.1f}nm")
                            validation_results['quality_score'] *= 0.9
            
            # Normalization scheme validation
            if 'Normalization_Scheme' in df.columns:
                scheme = df['Normalization_Scheme'].iloc[0] if not df['Normalization_Scheme'].empty else None
                expected = self.expected_schemes.get(light_source)
                
                if scheme and expected:
                    if scheme != expected:
                        validation_results['warnings'].append(f"Unexpected normalization scheme: {scheme} (expected: {expected})")
                        validation_results['quality_score'] *= 0.8
                    else:
                        validation_results['quality_score'] *= 1.1  # Bonus for correct scheme
            
            # Light source specific validation
            if light_source == 'UV':
                # UV should have substantial peaks
                if 'Intensity' in df.columns:
                    substantial_peaks = sum(1 for i in df['Intensity'] if i > 10.0)
                    if substantial_peaks == 0:
                        validation_results['warnings'].append("No substantial peaks found for UV analysis")
                        validation_results['quality_score'] *= 0.7
            
            # Format-specific validation
            if format_type.startswith('peak_detection'):
                required_cols = ['Peak_Number', 'Wavelength_nm', 'Intensity']
                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    validation_results['errors'].append(f"Missing required columns: {missing}")
                    validation_results['status'] = 'failed'
            
            # Final quality score adjustment
            if validation_results['errors']:
                validation_results['quality_score'] *= 0.1
                validation_results['status'] = 'failed'
            elif len(validation_results['warnings']) > 3:
                validation_results['quality_score'] *= 0.6
                validation_results['status'] = 'warning'
            
            return validation_results
            
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
            validation_results['status'] = 'failed'
            validation_results['quality_score'] = 0.0
            return validation_results
    
    def log_validation_result(self, batch_id, file_path, light_source, validation_type, status, message):
        """Log validation results to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO validation_log 
                (file_path, light_source, validation_type, status, message, batch_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (str(file_path), light_source, validation_type, status, message, batch_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Validation logging error: {e}")
    
    def extract_gem_id(self, filename):
        """Enhanced gem ID extraction from filename"""
        base_name = Path(filename).stem
        
        # Remove common suffixes to get core gem ID
        patterns_to_remove = [
            r'_enhanced_features_\d{8}_\d{6}$',
            r'_structural_\d{8}_\d{6}$',
            r'_[BLU]C\d+$',
            r'_[BLU]P\d+$',
            r'_[BLU]$',
            r'_(halogen|laser|uv)$'
        ]
        
        import re
        for pattern in patterns_to_remove:
            base_name = re.sub(pattern, '', base_name, flags=re.IGNORECASE)
        
        return base_name if base_name else 'unknown'
    
    def import_csv_file_enhanced(self, file_path, batch_id):
        """Enhanced CSV file import with complete metadata handling"""
        try:
            # Enhanced file detection
            light_source = self.detect_light_source_from_path(file_path)
            format_type, df = self.detect_csv_format(file_path)
            gem_id = self.extract_gem_id(file_path)
            
            print(f"📥 Processing: {Path(file_path).name}")
            print(f"    Light source: {light_source}")
            print(f"    Format: {format_type}")
            print(f"    Gem ID: {gem_id}")
            
            if format_type == 'error':
                print(f"    ❌ Failed to read file")
                self.log_validation_result(batch_id, file_path, light_source, "file_read", "error", "Failed to read CSV file")
                return 0, False
            
            if format_type == 'unknown':
                print(f"    ⚠️ Unknown format - attempting basic import")
                self.log_validation_result(batch_id, file_path, light_source, "format_detection", "warning", "Unknown CSV format")
            
            # Enhanced data validation
            if self.validation_enabled:
                validation_results = self.validate_data_quality(df, file_path, light_source, format_type)
                
                print(f"    📊 Quality score: {validation_results['quality_score']:.2f}")
                
                if validation_results['status'] == 'failed':
                    print(f"    ❌ Validation failed: {'; '.join(validation_results['errors'])}")
                    self.log_validation_result(batch_id, file_path, light_source, "data_validation", "failed", 
                                             '; '.join(validation_results['errors']))
                    if self.audio_enabled:
                        self.play_bleep("error")
                    return 0, False
                
                if validation_results['warnings']:
                    print(f"    ⚠️ Warnings: {'; '.join(validation_results['warnings'][:2])}")
                    self.log_validation_result(batch_id, file_path, light_source, "data_validation", "warning",
                                             '; '.join(validation_results['warnings']))
            
            # Enhanced database import
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            imported_count = 0
            session_id = f"{gem_id}_{light_source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Format-specific import handling
            if format_type.startswith('peak_detection'):
                imported_count = self.import_peak_detection_data(cursor, df, file_path, light_source, gem_id, batch_id, session_id, format_type)
            elif format_type.startswith('manual_structural'):
                imported_count = self.import_manual_structural_data(cursor, df, file_path, light_source, gem_id, batch_id, session_id, format_type)
            else:
                imported_count = self.import_generic_data(cursor, df, file_path, light_source, gem_id, batch_id, session_id)
            
            conn.commit()
            conn.close()
            
            if imported_count > 0:
                print(f"    ✅ Imported {imported_count} records")
                if self.audio_enabled:
                    self.play_bleep("import")
                return imported_count, True
            else:
                print(f"    ❌ No records imported")
                return 0, False
            
        except Exception as e:
            print(f"    ❌ Import error: {e}")
            self.log_validation_result(batch_id, file_path, light_source, "import", "error", str(e))
            if self.audio_enabled:
                self.play_bleep("error")
            return 0, False
    
    def import_peak_detection_data(self, cursor, df, file_path, light_source, gem_id, batch_id, session_id, format_type):
        """Import peak detection format data"""
        imported_count = 0
        base_filename = Path(file_path).stem
        
        for _, row in df.iterrows():
            try:
                peak_num = row.get('Peak_Number', 0)
                wavelength = row.get('Wavelength_nm', row.get('Wavelength', 0))
                intensity = row.get('Intensity', 0)
                prominence = row.get('Prominence', 0)
                category = row.get('Category', 'Unknown')
                
                # Enhanced normalization metadata
                norm_scheme = row.get('Normalization_Scheme', self.expected_schemes.get(light_source))
                ref_wavelength = row.get('Reference_Wavelength')
                
                cursor.execute("""
                    INSERT OR IGNORE INTO structural_features 
                    (feature, file, light_source, wavelength, intensity, point_type, 
                     feature_group, processing, peak_number, prominence, category,
                     normalization_scheme, reference_wavelength, intensity_range_min,
                     intensity_range_max, data_type, gem_id, analysis_session_id, 
                     import_batch_id, file_source, normalization_compatible)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f'Peak_{int(peak_num)}',
                    base_filename,
                    light_source,
                    float(wavelength),
                    float(intensity),
                    'Peak',
                    category,
                    f'Auto_Peak_Detection_{format_type}',
                    int(peak_num),
                    float(prominence),
                    category,
                    norm_scheme,
                    self.safe_float(ref_wavelength),
                    0.0,  # intensity_range_min
                    100.0,  # intensity_range_max
                    'peak_detection',
                    gem_id,
                    session_id,
                    batch_id,
                    str(file_path),
                    format_type == 'peak_detection_enhanced'
                ))
                
                if cursor.rowcount > 0:
                    imported_count += 1
                    
            except Exception as e:
                print(f"      ⚠️ Skipped peak record: {e}")
        
        return imported_count
    
    def import_manual_structural_data(self, cursor, df, file_path, light_source, gem_id, batch_id, session_id, format_type):
        """Import manual structural analysis data"""
        imported_count = 0
        base_filename = Path(file_path).stem
        
        for _, row in df.iterrows():
            try:
                # Extract primary wavelength (prefer Crest, then Wavelength, then Start)
                wavelength = None
                for col in ['Crest', 'Wavelength', 'Midpoint', 'Start', 'Max']:
                    if col in df.columns and pd.notna(row.get(col)):
                        wavelength = float(row[col])
                        break
                
                if wavelength is None:
                    continue
                
                intensity = self.safe_float(row.get('Intensity', 1.0)) or 1.0
                
                # Enhanced metadata extraction
                cursor.execute("""
                    INSERT OR IGNORE INTO structural_features 
                    (feature, file, light_source, wavelength, intensity, point_type, 
                     feature_group, processing, start_wavelength, end_wavelength, 
                     symmetry_ratio, skew_description, midpoint, bottom,
                     baseline_used, norm_factor, snr, width_nm, height,
                     local_slope, slope_r_squared, normalization_scheme, 
                     reference_wavelength, intensity_range_min, intensity_range_max,
                     data_type, gem_id, analysis_session_id, import_batch_id, 
                     file_source, normalization_compatible)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row.get('Feature', f'Feature_{imported_count+1}')),
                    base_filename,
                    light_source,
                    wavelength,
                    intensity,
                    'Manual_Enhanced' if format_type == 'manual_structural_enhanced' else 'Manual',
                    str(row.get('Feature', 'Unknown')),
                    f'Manual_Structural_{format_type}',
                    self.safe_float(row.get('Start')),
                    self.safe_float(row.get('End')),
                    self.safe_float(row.get('Symmetry_Ratio')),
                    str(row.get('Skew_Description', '')),
                    self.safe_float(row.get('Midpoint')),
                    self.safe_float(row.get('Bottom')),
                    self.safe_float(row.get('Baseline_Used')),
                    self.safe_float(row.get('Norm_Factor')),
                    self.safe_float(row.get('SNR')),
                    self.safe_float(row.get('Width_nm')),
                    self.safe_float(row.get('Height')),
                    self.safe_float(row.get('Local_Slope')),
                    self.safe_float(row.get('Slope_R_Squared')),
                    row.get('Normalization_Scheme', self.expected_schemes.get(light_source)),
                    self.safe_float(row.get('Reference_Wavelength')),
                    0.0,  # intensity_range_min
                    100.0,  # intensity_range_max
                    'manual_structural',
                    gem_id,
                    session_id,
                    batch_id,
                    str(file_path),
                    format_type == 'manual_structural_enhanced'
                ))
                
                if cursor.rowcount > 0:
                    imported_count += 1
                    
            except Exception as e:
                print(f"      ⚠️ Skipped structural record: {e}")
        
        return imported_count
    
    def import_generic_data(self, cursor, df, file_path, light_source, gem_id, batch_id, session_id):
        """Import data with unknown/generic format"""
        imported_count = 0
        base_filename = Path(file_path).stem
        
        # Try to find wavelength and intensity columns
        wavelength_col = None
        intensity_col = None
        
        for col in df.columns:
            if 'wavelength' in col.lower() or col.lower() in ['wl', 'lambda']:
                wavelength_col = col
            if 'intensity' in col.lower() or col.lower() in ['i', 'counts', 'signal']:
                intensity_col = col
        
        if wavelength_col is None or intensity_col is None:
            return 0
        
        for i, row in df.iterrows():
            try:
                wavelength = float(row[wavelength_col])
                intensity = float(row[intensity_col])
                
                cursor.execute("""
                    INSERT OR IGNORE INTO structural_features 
                    (feature, file, light_source, wavelength, intensity, point_type, 
                     feature_group, processing, data_type, gem_id, analysis_session_id, 
                     import_batch_id, file_source, normalization_compatible)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f'Generic_Feature_{i+1}',
                    base_filename,
                    light_source,
                    wavelength,
                    intensity,
                    'Generic',
                    'Unknown',
                    'Generic_Import',
                    'generic',
                    gem_id,
                    session_id,
                    batch_id,
                    str(file_path),
                    False
                ))
                
                if cursor.rowcount > 0:
                    imported_count += 1
                    
            except Exception as e:
                print(f"      ⚠️ Skipped generic record: {e}")
        
        return imported_count
    
    def safe_float(self, value):
        """Safely convert value to float or return None"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return float(value)
        except:
            return None
    
    def scan_and_import_all(self):
        """Enhanced batch import with monitoring and statistics"""
        print("🚀 ENHANCED BATCH IMPORT SYSTEM")
        print("=" * 50)
        
        # Create batch ID
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()
        
        # Log batch start
        self.log_batch_start(batch_id)
        
        # Scan for CSV files
        all_csv_files = []
        light_source_counts = {'halogen': 0, 'laser': 0, 'uv': 0, 'unknown': 0}
        
        # Scan standard directories
        for light_source in ['halogen', 'laser', 'uv']:
            source_dir = self.base_path / light_source
            if source_dir.exists():
                csv_files = list(source_dir.glob("*.csv"))
                for csv_file in csv_files:
                    all_csv_files.append((csv_file, light_source))
                light_source_counts[light_source] = len(csv_files)
        
        # Scan root directory
        if self.base_path.exists():
            root_csv_files = list(self.base_path.glob("*.csv"))
            for csv_file in root_csv_files:
                all_csv_files.append((csv_file, 'unknown'))
                light_source_counts['unknown'] += 1
        
        print(f"📁 Found CSV files:")
        for light_source, count in light_source_counts.items():
            if count > 0:
                print(f"   {light_source.capitalize()}: {count} files")
        
        total_files = len(all_csv_files)
        if total_files == 0:
            print("❌ No CSV files found!")
            return
        
        print(f"\n📊 Total files to process: {total_files}")
        
        # Confirm import
        if total_files > 20:
            confirm = input(f"\n🔄 Process {total_files} files? This may take a while (y/n): ").strip().lower()
            if confirm != 'y':
                print("❌ Import cancelled")
                return
        
        # Process files with progress tracking
        successful_imports = 0
        total_records = 0
        failed_imports = 0
        
        print(f"\n🔄 Processing files...")
        
        for i, (file_path, detected_light_source) in enumerate(all_csv_files, 1):
            print(f"\n[{i}/{total_files}] {file_path.name}")
            
            record_count, success = self.import_csv_file_enhanced(file_path, batch_id)
            
            if success:
                successful_imports += 1
                total_records += record_count
                self.import_stats['successful_imports'] += 1
            else:
                failed_imports += 1
                self.import_stats['validation_failures'] += 1
            
            self.import_stats['total_files_processed'] += 1
            self.import_stats['total_records_imported'] += record_count
            
            # Progress indicator
            if i % 10 == 0 or i == total_files:
                progress = (i / total_files) * 100
                print(f"    📊 Progress: {progress:.1f}% ({i}/{total_files})")
        
        # Complete batch logging
        processing_time = time.time() - start_time
        self.log_batch_complete(batch_id, total_files, total_records, failed_imports, processing_time)
        
        print(f"\n📊 ENHANCED BATCH IMPORT COMPLETED:")
        print("=" * 50)
        print(f"   ✅ Files processed: {successful_imports}/{total_files}")
        print(f"   📝 Records imported: {total_records:,}")
        print(f"   ❌ Failed imports: {failed_imports}")
        print(f"   ⏱️ Processing time: {processing_time:.1f} seconds")
        print(f"   📊 Import rate: {total_records/processing_time:.1f} records/sec")
        
        if total_records > 0:
            if self.audio_enabled:
                self.play_bleep("batch_complete")
            
            # Show enhanced database statistics
            self.show_enhanced_statistics()
        
        return successful_imports, total_records
    
    def log_batch_start(self, batch_id):
        """Log batch import start"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO import_batches (batch_id, status, notes)
                VALUES (?, ?, ?)
            """, (batch_id, 'in_progress', 'Enhanced batch import started'))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Batch logging error: {e}")
    
    def log_batch_complete(self, batch_id, files_processed, records_imported, failures, processing_time):
        """Log batch import completion"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE import_batches 
                SET files_processed = ?, records_imported = ?, validation_failures = ?,
                    processing_time_seconds = ?, status = ?, notes = ?
                WHERE batch_id = ?
            """, (files_processed, records_imported, failures, processing_time, 'completed',
                  f'Enhanced batch import completed successfully', batch_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Batch completion logging error: {e}")
    
    def show_enhanced_statistics(self):
        """Show enhanced database statistics with detailed analysis"""
        print("\n📊 ENHANCED DATABASE STATISTICS")
        print("=" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Basic statistics
            basic_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT file) as unique_files,
                    COUNT(DISTINCT gem_id) as unique_gems,
                    COUNT(DISTINCT light_source) as light_sources,
                    COUNT(DISTINCT data_type) as data_types
                FROM structural_features
            """, conn)
            
            stats = basic_stats.iloc[0]
            print(f"📁 Total records: {stats['total_records']:,}")
            print(f"📄 Unique files: {stats['unique_files']:,}")
            print(f"💎 Unique gems: {stats['unique_gems']:,}")
            print(f"💡 Light sources: {stats['light_sources']}")
            print(f"📊 Data types: {stats['data_types']}")
            
            # Light source breakdown
            light_stats = pd.read_sql_query("""
                SELECT light_source, COUNT(*) as count, COUNT(DISTINCT gem_id) as gems
                FROM structural_features 
                GROUP BY light_source 
                ORDER BY count DESC
            """, conn)
            
            print(f"\n💡 BY LIGHT SOURCE:")
            for _, row in light_stats.iterrows():
                print(f"   {row['light_source']}: {row['count']:,} records ({row['gems']} gems)")
            
            # Data type breakdown
            type_stats = pd.read_sql_query("""
                SELECT data_type, COUNT(*) as count, COUNT(DISTINCT gem_id) as gems
                FROM structural_features 
                GROUP BY data_type 
                ORDER BY count DESC
            """, conn)
            
            print(f"\n📊 BY DATA TYPE:")
            for _, row in type_stats.iterrows():
                print(f"   {row['data_type']}: {row['count']:,} records ({row['gems']} gems)")
            
            # Normalization status
            norm_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(CASE WHEN normalization_scheme IS NOT NULL THEN 1 END) as with_scheme,
                    COUNT(CASE WHEN normalization_compatible = 1 THEN 1 END) as compatible,
                    COUNT(CASE WHEN validation_status = 'passed' THEN 1 END) as validated
                FROM structural_features
            """, conn)
            
            norm = norm_stats.iloc[0]
            print(f"\n🔧 NORMALIZATION STATUS:")
            print(f"   With scheme: {norm['with_scheme']:,} / {stats['total_records']:,}")
            print(f"   Compatible: {norm['compatible']:,} / {stats['total_records']:,}")
            
            # Recent activity
            recent_stats = pd.read_sql_query("""
                SELECT COUNT(*) as recent_count
                FROM structural_features 
                WHERE timestamp > datetime('now', '-24 hours')
            """, conn)
            
            recent_count = recent_stats.iloc[0]['recent_count']
            print(f"\n📅 Recent activity (24h): {recent_count:,} records")
            
            # Top gems by feature count
            top_gems = pd.read_sql_query("""
                SELECT gem_id, COUNT(*) as feature_count, COUNT(DISTINCT light_source) as light_sources
                FROM structural_features 
                GROUP BY gem_id 
                ORDER BY feature_count DESC 
                LIMIT 10
            """, conn)
            
            print(f"\n🏆 TOP GEMS BY FEATURES:")
            for _, row in top_gems.iterrows():
                print(f"   {row['gem_id']}: {row['feature_count']} features ({row['light_sources']} light sources)")
            
            # Import batch summary
            batch_stats = pd.read_sql_query("""
                SELECT COUNT(*) as total_batches, 
                       SUM(files_processed) as total_files,
                       SUM(records_imported) as total_imported,
                       AVG(processing_time_seconds) as avg_processing_time
                FROM import_batches
                WHERE status = 'completed'
            """, conn)
            
            if not batch_stats.empty and batch_stats.iloc[0]['total_batches'] > 0:
                batch = batch_stats.iloc[0]
                print(f"\n📦 IMPORT HISTORY:")
                print(f"   Completed batches: {batch['total_batches']}")
                print(f"   Total files processed: {int(batch['total_files'])}")
                print(f"   Total records imported: {int(batch['total_imported'])}")
                print(f"   Average processing time: {batch['avg_processing_time']:.1f} seconds")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Statistics error: {e}")
    
    def search_database(self):
        """Enhanced database search with multiple criteria"""
        print("\n🔍 ENHANCED DATABASE SEARCH")
        print("=" * 35)
        
        search_options = [
            "Search by gem ID",
            "Search by wavelength range", 
            "Search by light source",
            "Search by feature type",
            "Search recent imports",
            "Search by validation status",
            "Advanced multi-criteria search"
        ]
        
        print("Search options:")
        for i, option in enumerate(search_options, 1):
            print(f"{i}. {option}")
        
        try:
            choice = int(input(f"\nSelect search type (1-{len(search_options)}): "))
            
            if choice < 1 or choice > len(search_options):
                print("❌ Invalid choice")
                return
            
            conn = sqlite3.connect(self.db_path)
            
            if choice == 1:  # Gem ID
                gem_id = input("Enter gem ID (partial match): ").strip()
                query = "SELECT * FROM structural_features WHERE gem_id LIKE ? ORDER BY wavelength"
                df = pd.read_sql_query(query, conn, params=[f"%{gem_id}%"])
                
            elif choice == 2:  # Wavelength range
                min_wl = float(input("Minimum wavelength: "))
                max_wl = float(input("Maximum wavelength: "))
                query = "SELECT * FROM structural_features WHERE wavelength BETWEEN ? AND ? ORDER BY wavelength"
                df = pd.read_sql_query(query, conn, params=[min_wl, max_wl])
                
            elif choice == 3:  # Light source
                light_source = input("Enter light source: ").strip()
                query = "SELECT * FROM structural_features WHERE light_source LIKE ? ORDER BY gem_id, wavelength"
                df = pd.read_sql_query(query, conn, params=[f"%{light_source}%"])
                
            elif choice == 4:  # Feature type
                feature_type = input("Enter feature type: ").strip()
                query = "SELECT * FROM structural_features WHERE feature_group LIKE ? ORDER BY wavelength"
                df = pd.read_sql_query(query, conn, params=[f"%{feature_type}%"])
                
            elif choice == 5:  # Recent imports
                hours = int(input("Hours back (default 24): ") or "24")
                query = f"SELECT * FROM structural_features WHERE timestamp > datetime('now', '-{hours} hours') ORDER BY timestamp DESC"
                df = pd.read_sql_query(query, conn)
                
            elif choice == 6:  # Validation status
                status = input("Enter validation status (passed/warning/failed): ").strip()
                query = "SELECT * FROM structural_features WHERE validation_status = ? ORDER BY timestamp DESC"
                df = pd.read_sql_query(query, conn, params=[status])
                
            elif choice == 7:  # Advanced search
                print("Advanced search - enter criteria (press Enter to skip):")
                gem_id = input("Gem ID: ").strip()
                light_source = input("Light source: ").strip()
                min_wl = input("Min wavelength: ").strip()
                max_wl = input("Max wavelength: ").strip()
                
                conditions = []
                params = []
                
                if gem_id:
                    conditions.append("gem_id LIKE ?")
                    params.append(f"%{gem_id}%")
                if light_source:
                    conditions.append("light_source LIKE ?")
                    params.append(f"%{light_source}%")
                if min_wl:
                    conditions.append("wavelength >= ?")
                    params.append(float(min_wl))
                if max_wl:
                    conditions.append("wavelength <= ?")
                    params.append(float(max_wl))
                
                if not conditions:
                    print("❌ No search criteria provided")
                    conn.close()
                    return
                
                query = f"SELECT * FROM structural_features WHERE {' AND '.join(conditions)} ORDER BY gem_id, wavelength"
                df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            if df.empty:
                print("❌ No results found")
                return
            
            print(f"\n📊 Found {len(df)} results:")
            print("=" * 100)
            
            # Enhanced result display
            for i, (_, row) in enumerate(df.iterrows(), 1):
                gem_id = row['gem_id']
                light = row['light_source']
                wavelength = row['wavelength']
                feature = row['feature_group']
                intensity = row['intensity']
                data_type = row.get('data_type', 'unknown')
                
                print(f"{i:4}. {gem_id} | {light} | {wavelength:7.1f}nm | {feature:15} | I:{intensity:8.2f} | {data_type}")
                
                if i >= 50:  # Limit display
                    remaining = len(df) - 50
                    if remaining > 0:
                        print(f"       ... and {remaining} more results")
                    break
            
            # Enhanced export option
            export = input(f"\n💾 Export results to CSV? (y/n): ").strip().lower()
            if export == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"enhanced_search_results_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"✅ Results exported to {filename}")
                
                if self.audio_enabled:
                    self.play_bleep("completion")
                    
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    def toggle_audio(self):
        """Toggle audio feedback"""
        self.audio_enabled = not self.audio_enabled
        status = "ENABLED" if self.audio_enabled else "DISABLED"
        print(f"🔊 Audio feedback: {status}")
        
        if self.audio_enabled and HAS_AUDIO:
            self.play_bleep("completion")
        elif self.audio_enabled and not HAS_AUDIO:
            print("   ⚠️ Audio system not available (install winsound/pygame)")
    
    def toggle_validation(self):
        """Toggle data validation"""
        self.validation_enabled = not self.validation_enabled
        status = "ENABLED" if self.validation_enabled else "DISABLED"
        print(f"✅ Data validation: {status}")
    
    def run_interactive_menu(self):
        """Run interactive database management menu"""
        menu_options = [
            ("🚀 Enhanced Batch Import", self.scan_and_import_all),
            ("📊 Show Enhanced Statistics", self.show_enhanced_statistics),
            ("🔍 Enhanced Database Search", self.search_database),
            ("🔊 Toggle Audio Feedback", self.toggle_audio),
            ("✅ Toggle Data Validation", self.toggle_validation),
            ("❌ Exit", None)
        ]
        
        while True:
            print(f"\n🗄️ INTEGRATED DATABASE MANAGEMENT SYSTEM")
            print("=" * 50)
            print(f"Database: {self.db_path}")
            print(f"Audio: {'ON' if self.audio_enabled else 'OFF'}")
            print(f"Validation: {'ON' if self.validation_enabled else 'OFF'}")
            print("=" * 50)
            
            for i, (description, _) in enumerate(menu_options, 1):
                print(f"{i}. {description}")
            
            # Show current statistics
            try:
                conn = sqlite3.connect(self.db_path)
                count = pd.read_sql_query("SELECT COUNT(*) as count FROM structural_features", conn).iloc[0]['count']
                conn.close()
                print(f"\nCurrent database: {count:,} records")
            except:
                print(f"\nCurrent database: Error reading")
            
            try:
                choice = int(input(f"\nChoice (1-{len(menu_options)}): "))
                
                if choice == len(menu_options):  # Exit
                    print("\n👋 Database management system closing...")
                    if self.audio_enabled:
                        self.play_bleep("completion")
                    break
                
                if 1 <= choice < len(menu_options):
                    description, action = menu_options[choice - 1]
                    print(f"\n🚀 {description.upper()}")
                    print("-" * 50)
                    
                    if action:
                        action()
                    
                    input("\n⏎ Press Enter to continue...")
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
    """Main function for integrated database management"""
    print("🗄️ INTEGRATED DATABASE MANAGEMENT SYSTEM")
    print("Complete CSV-to-SQLite solution with enhanced features")
    print("=" * 60)
    
    try:
        manager = IntegratedDatabaseManager()
        manager.run_interactive_menu()
        
    except KeyboardInterrupt:
        print("\nDatabase management interrupted - goodbye!")
    except Exception as e:
        print(f"Database management error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
