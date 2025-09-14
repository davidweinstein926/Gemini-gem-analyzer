#!/usr/bin/env python3
"""
ENHANCED GEMINI GEMOLOGICAL ANALYSIS SYSTEM - COMPLETE DATABASE INTEGRATION
Enhanced version with full database management, auto-import, enhanced comparison tools,
bleep features, relative height measurements, streamlined workflow automation, and
ULTRA_OPTIMIZED CSV import support.

Major Enhancements:
- Full database import/export system integrated
- Automatic CSV-to-SQLite import after manual analysis
- ULTRA_OPTIMIZED format support (41+ columns)
- Enhanced comparison tools using database queries
- Audio bleep feedback for significant features
- Relative height measurements across light sources
- Comprehensive database statistics and management
- Automated workflow from analysis → database → comparison
- Option 11 - Structural matching analysis with self-validation
- Option 17 - Enhanced CSV import with intelligent column mapping
- Fixed Unicode encoding issues and folder organization
"""

import os
import sys
import subprocess
import sqlite3
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict
import threading
import time
from datetime import datetime

# Audio support for bleep functionality
try:
    import winsound  # Windows audio support
    HAS_AUDIO = True
except ImportError:
    try:
        import pygame
        pygame.mixer.init()
        HAS_AUDIO = True
    except ImportError:
        HAS_AUDIO = False

class EnhancedGeminiAnalysisSystem:
    def __init__(self):
        self.db_path = "multi_structural_gem_data.db"
        
        # Database import system path
        self.structural_data_dir = r"c:\users\david\gemini sp10 structural data"
        
        # System files to check - UPDATED to point to database folder
        self.spectral_files = [
            'database/reference_spectra/gemini_db_long_B.csv', 
            'database/reference_spectra/gemini_db_long_L.csv', 
            'database/reference_spectra/gemini_db_long_U.csv'
        ]
        self.program_files = {
            'src/structural_analysis/main.py': 'Structural Analysis Hub',
            'src/structural_analysis/gemini_launcher.py': 'Structural Analyzers Launcher',
            'src/numerical_analysis/gemini1.py': 'Numerical Analysis Engine',
            'fast_gem_analysis.py': 'Fast Analysis Tool'
        }
        
        # Enhanced features state
        self.bleep_enabled = True
        self.auto_import_enabled = True
        self.relative_height_cache = {}
        
        # Initialize database system
        self.init_database_system()
    
    def init_database_system(self):
        """Initialize the integrated database management system"""
        print("Initializing integrated database system...")
        
        # Check if database exists, if not create basic structure
        if not os.path.exists(self.db_path):
            self.create_database_schema()
        
        # Ensure structural data directory exists
        os.makedirs(self.structural_data_dir, exist_ok=True)
        for subdir in ['halogen', 'laser', 'uv']:
            os.makedirs(os.path.join(self.structural_data_dir, subdir), exist_ok=True)
    
    def create_database_schema(self):
        """Create the database schema for structural features"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced schema with normalization metadata
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
                    
                    -- Enhanced fields for analysis
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
                    
                    -- Normalization metadata
                    normalization_scheme TEXT,
                    reference_wavelength REAL,
                    intensity_range_min REAL,
                    intensity_range_max REAL,
                    normalization_compatible BOOLEAN DEFAULT 1,
                    
                    -- Additional wavelength fields for structural matching
                    start_wavelength REAL,
                    end_wavelength REAL,
                    midpoint_wavelength REAL,
                    crest_wavelength REAL,
                    max_wavelength REAL,
                    
                    -- ULTRA_OPTIMIZED additional fields
                    feature_key TEXT,
                    baseline_quality TEXT,
                    baseline_width_nm REAL,
                    baseline_cv_percent REAL,
                    baseline_std_dev REAL,
                    target_ref_intensity REAL,
                    click_order REAL,
                    total_width_nm REAL,
                    left_width_nm REAL,
                    right_width_nm REAL,
                    skew_severity TEXT,
                    width_class TEXT,
                    target_reference_intensity REAL,
                    scaling_factor REAL,
                    total_spectrum_points INTEGER,
                    wavelength_range_min REAL,
                    wavelength_range_max REAL,
                    wavelength_span_nm REAL,
                    baseline_snr REAL,
                    analysis_date TEXT,
                    analysis_time TEXT,
                    analyzer_version TEXT,
                    
                    -- Timestamps
                    timestamp TEXT DEFAULT (datetime('now')),
                    file_source TEXT,
                    data_type TEXT,
                    
                    UNIQUE(file, feature, wavelength, point_type)
                )
            """)
            
            # Create indexes for performance
            indexes = [
                ("idx_file", "file"),
                ("idx_light_source", "light_source"),
                ("idx_wavelength", "wavelength"),
                ("idx_feature_group", "feature_group"),
                ("idx_normalization_scheme", "normalization_scheme")
            ]
            
            for idx_name, columns in indexes:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON structural_features({columns})")
            
            conn.commit()
            conn.close()
            print("Database schema created successfully with ULTRA_OPTIMIZED support")
            
        except Exception as e:
            print(f"Error creating database schema: {e}")

    def auto_import_csv_to_database_enhanced(self, csv_file_path):
        """Enhanced CSV import that handles BOTH regular (23) and ULTRA_OPTIMIZED (41) column formats"""
        if not self.auto_import_enabled:
            return False
        
        try:
            print(f"Analyzing CSV format: {os.path.basename(csv_file_path)}")
            
            # Determine light source from file path or filename
            light_source = self.detect_light_source_from_path(csv_file_path)
            
            # Read CSV file and analyze its structure
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                print(f"   Empty CSV file")
                return False
            
            # Analyze CSV format
            column_count = len(df.columns)
            csv_format = "ULTRA_OPTIMIZED" if column_count >= 35 else "STANDARD"
            
            print(f"   Detected format: {csv_format} ({column_count} columns)")
            
            # Check if database schema can handle all columns
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current database schema
            cursor.execute("PRAGMA table_info(structural_features)")
            db_columns = {col[1]: col[2] for col in cursor.fetchall()}  # name: type
            
            print(f"   Database has {len(db_columns)} columns")
            
            # Map CSV columns to database columns
            column_mapping = self.create_column_mapping(df.columns, db_columns, csv_format)
            
            imported_count = 0
            skipped_columns = []
            base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
            
            # Build dynamic INSERT statement based on available columns
            available_db_columns = [db_col for csv_col, db_col in column_mapping.items() 
                                   if db_col in db_columns and db_col != 'SKIP']
            
            # Add required columns that might not be in CSV
            if 'file' not in available_db_columns:
                available_db_columns.append('file')
            if 'light_source' not in available_db_columns:
                available_db_columns.append('light_source')
            if 'file_source' not in available_db_columns:
                available_db_columns.append('file_source')
            if 'data_type' not in available_db_columns:
                available_db_columns.append('data_type')
            
            # Remove duplicates
            available_db_columns = list(set(available_db_columns))
            
            # Build INSERT statement
            columns_str = ', '.join(available_db_columns)
            placeholders = ', '.join(['?' for _ in available_db_columns])
            
            insert_query = f"""
                INSERT OR IGNORE INTO structural_features 
                ({columns_str})
                VALUES ({placeholders})
            """
            
            print(f"   Will import {len(available_db_columns)} columns to database")
            
            # Import each row from the CSV
            for _, row in df.iterrows():
                try:
                    # Build values list for this row
                    values = []
                    
                    for db_column in available_db_columns:
                        if db_column == 'file':
                            values.append(base_filename)
                        elif db_column == 'light_source':
                            values.append(light_source)
                        elif db_column == 'file_source':
                            values.append(csv_file_path)
                        elif db_column == 'data_type':
                            values.append(f'manual_{csv_format.lower()}')
                        else:
                            # Find corresponding CSV column
                            csv_column = next((csv_col for csv_col, mapped_db_col in column_mapping.items() 
                                             if mapped_db_col == db_column), None)
                            
                            if csv_column and csv_column in row:
                                value = row[csv_column]
                                
                                # Convert to appropriate type
                                if pd.isna(value) or value == '':
                                    values.append(None)
                                elif db_columns[db_column] in ['REAL', 'FLOAT']:
                                    values.append(self.safe_float(value))
                                elif db_columns[db_column] in ['INTEGER', 'INT']:
                                    values.append(self.safe_int(value))
                                else:
                                    values.append(str(value))
                            else:
                                values.append(None)
                    
                    # Execute insert
                    cursor.execute(insert_query, values)
                    
                    if cursor.rowcount > 0:
                        imported_count += 1
                        
                except Exception as e:
                    print(f"   Error importing row: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            # Report results
            if imported_count > 0:
                print(f"   Successfully imported {imported_count} features")
                print(f"   Format: {csv_format} → Database mapping successful")
                
                if csv_format == "ULTRA_OPTIMIZED":
                    print(f"   Enhanced data preserved: width details, baseline quality, skew analysis")
                
                self.play_bleep(feature_type="completion")
                return True
            else:
                print(f"   No features imported")
                return False
                
        except Exception as e:
            print(f"   Enhanced import error: {e}")
            return False

    def create_column_mapping(self, csv_columns, db_columns, csv_format):
        """Create intelligent mapping between CSV and database columns"""
        
        # Base mapping that works for both formats
        base_mapping = {
            # Core columns (present in both formats)
            'Feature': 'feature',
            'File': 'SKIP',  # We'll use filename instead
            'Light_Source': 'SKIP',  # We'll detect from path
            'Wavelength': 'wavelength', 
            'Intensity': 'intensity',
            'Point_Type': 'point_type',
            'Feature_Group': 'feature_group',
            'Processing': 'processing',
            'SNR': 'snr',
            'Baseline_Used': 'baseline_used',
            'Norm_Factor': 'norm_factor',
            'Normalization_Method': 'normalization_scheme',
            'Reference_Wavelength_Used': 'reference_wavelength',
            'Symmetry_Ratio': 'symmetry_ratio',
            'Skew_Description': 'skew_description',
            'Width_nm': 'width_nm',
            'Normalization_Scheme': 'normalization_scheme',
            'Reference_Wavelength': 'reference_wavelength',
            'Intensity_Range_Min': 'intensity_range_min',
            'Intensity_Range_Max': 'intensity_range_max',
        }
        
        if csv_format == "ULTRA_OPTIMIZED":
            # Additional mappings for ULTRA_OPTIMIZED format
            ultra_mapping = {
                'Feature_Key': 'feature_key',
                'Baseline_Quality': 'baseline_quality', 
                'Baseline_Width_nm': 'baseline_width_nm',
                'Baseline_CV_Percent': 'baseline_cv_percent',
                'Baseline_Std_Dev': 'baseline_std_dev',
                'Target_Ref_Intensity': 'target_ref_intensity',
                'Click_Order': 'click_order',
                'total_width_nm': 'total_width_nm',
                'left_width_nm': 'left_width_nm', 
                'right_width_nm': 'right_width_nm',
                'symmetry_ratio': 'symmetry_ratio',
                'skew_description': 'skew_description',
                'skew_severity': 'skew_severity',
                'width_class': 'width_class',
                'Target_Reference_Intensity': 'target_reference_intensity',
                'Scaling_Factor': 'scaling_factor',
                'Total_Spectrum_Points': 'total_spectrum_points',
                'Wavelength_Range_Min': 'wavelength_range_min',
                'Wavelength_Range_Max': 'wavelength_range_max',
                'Wavelength_Span_nm': 'wavelength_span_nm',
                'Baseline_SNR': 'baseline_snr',
                'Analysis_Date': 'analysis_date',
                'Analysis_Time': 'analysis_time',
                'Analyzer_Version': 'analyzer_version'
            }
            base_mapping.update(ultra_mapping)
        
        # Filter mapping to only include columns that exist in CSV and database
        final_mapping = {}
        
        for csv_col in csv_columns:
            if csv_col in base_mapping:
                db_col = base_mapping[csv_col]
                if db_col == 'SKIP':
                    final_mapping[csv_col] = 'SKIP'
                elif db_col in db_columns:
                    final_mapping[csv_col] = db_col
                else:
                    final_mapping[csv_col] = 'SKIP'  # Database doesn't have this column
            else:
                # Try fuzzy matching for unmapped columns
                fuzzy_match = self.find_fuzzy_column_match(csv_col, db_columns.keys())
                final_mapping[csv_col] = fuzzy_match if fuzzy_match else 'SKIP'
        
        return final_mapping

    def find_fuzzy_column_match(self, csv_column, db_columns):
        """Find best fuzzy match for unmapped columns"""
        csv_lower = csv_column.lower().replace('_', '').replace(' ', '')
        
        for db_col in db_columns:
            db_lower = db_col.lower().replace('_', '').replace(' ', '')
            
            # Exact match (ignore case/underscores)
            if csv_lower == db_lower:
                return db_col
            
            # Partial match
            if csv_lower in db_lower or db_lower in csv_lower:
                return db_col
        
        return None

    def safe_int(self, value):
        """Safely convert value to int or return None"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return int(float(value))  # Convert through float first to handle decimals
        except:
            return None

    def update_database_schema_for_ultra_optimized(self):
        """Add missing columns to database schema to support ULTRA_OPTIMIZED format"""
        print("\nUPDATING DATABASE SCHEMA FOR ULTRA_OPTIMIZED FORMAT")
        print("=" * 60)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get current columns
            cursor.execute("PRAGMA table_info(structural_features)")
            existing_columns = {col[1] for col in cursor.fetchall()}
            
            # Define additional columns needed for ULTRA_OPTIMIZED
            ultra_columns = {
                'feature_key': 'TEXT',
                'baseline_quality': 'TEXT',
                'baseline_width_nm': 'REAL',
                'baseline_cv_percent': 'REAL',
                'baseline_std_dev': 'REAL', 
                'target_ref_intensity': 'REAL',
                'click_order': 'REAL',
                'total_width_nm': 'REAL',
                'left_width_nm': 'REAL',
                'right_width_nm': 'REAL',
                'skew_severity': 'TEXT',
                'width_class': 'TEXT',
                'target_reference_intensity': 'REAL',
                'scaling_factor': 'REAL',
                'total_spectrum_points': 'INTEGER',
                'wavelength_range_min': 'REAL',
                'wavelength_range_max': 'REAL',
                'wavelength_span_nm': 'REAL',
                'baseline_snr': 'REAL',
                'analysis_date': 'TEXT',
                'analysis_time': 'TEXT',
                'analyzer_version': 'TEXT'
            }
            
            # Add missing columns
            added_columns = 0
            for col_name, col_type in ultra_columns.items():
                if col_name not in existing_columns:
                    try:
                        cursor.execute(f"ALTER TABLE structural_features ADD COLUMN {col_name} {col_type}")
                        print(f"   Added column: {col_name} ({col_type})")
                        added_columns += 1
                    except Exception as e:
                        print(f"   Could not add {col_name}: {e}")
            
            if added_columns > 0:
                conn.commit()
                print(f"\nSuccessfully added {added_columns} columns for ULTRA_OPTIMIZED support!")
            else:
                print(f"\nDatabase already supports ULTRA_OPTIMIZED format")
            
            conn.close()
            return added_columns > 0
            
        except Exception as e:
            print(f"Schema update error: {e}")
            return False

    def test_enhanced_import_system(self):
        """Test the enhanced import system with both CSV formats"""
        print("\nTESTING ENHANCED IMPORT SYSTEM")
        print("=" * 50)
        
        # First, update schema to support ULTRA_OPTIMIZED
        self.update_database_schema_for_ultra_optimized()
        
        # Test directory
        test_dir = "data/structural_data"
        
        if not os.path.exists(test_dir):
            print(f"Test directory not found: {test_dir}")
            return
        
        # Find test files
        test_files = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.csv'):
                    full_path = os.path.join(root, file)
                    test_files.append(full_path)
        
        if not test_files:
            print(f"No CSV files found in {test_dir}")
            return
        
        print(f"Found {len(test_files)} CSV files to test")
        
        # Test each file
        results = {'success': 0, 'failed': 0, 'ultra_optimized': 0, 'standard': 0}
        
        for test_file in test_files[:5]:  # Test first 5 files
            print(f"\nTesting: {os.path.basename(test_file)}")
            
            success = self.auto_import_csv_to_database_enhanced(test_file)
            
            if success:
                results['success'] += 1
                # Check if it was ultra optimized
                df = pd.read_csv(test_file)
                if len(df.columns) >= 35:
                    results['ultra_optimized'] += 1
                else:
                    results['standard'] += 1
            else:
                results['failed'] += 1
        
        # Show results
        print(f"\nENHANCED IMPORT TEST RESULTS:")
        print("=" * 40)
        print(f"Successful imports: {results['success']}")
        print(f"Failed imports: {results['failed']}")
        print(f"ULTRA_OPTIMIZED files: {results['ultra_optimized']}")
        print(f"Standard files: {results['standard']}")
        
        if results['success'] > 0:
            print(f"\nEnhanced import system working!")
            print(f"Both 23-column and 41-column formats supported")
            if self.bleep_enabled:
                self.play_bleep(feature_type="completion")
        else:
            print(f"\nImport system needs debugging")
        
        return results
    
    def correct_normalize_spectrum(self, wavelengths, intensities, light_source):
        """DATABASE-MATCHING NORMALIZATION - matches corrected database exactly"""
    
        if light_source == 'B':
            # B Light: 650nm -> 50000
            anchor_idx = np.argmin(np.abs(wavelengths - 650))
            if intensities[anchor_idx] != 0:
                normalized = intensities * (50000 / intensities[anchor_idx])
                return normalized
            else:
                return intensities
    
        elif light_source == 'L':
            # L Light: Maximum -> 50000 (CORRECTED from 450nm)
            max_intensity = intensities.max()
            if max_intensity != 0:
                normalized = intensities * (50000 / max_intensity)
                return normalized
            else:
                return intensities
        elif light_source == 'U':
            # U Light: 811nm window max -> 15000 (matches database method)
            mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
            window = intensities[mask]
            if len(window) > 0 and window.max() > 0:
                normalized = intensities * (15000 / window.max())
                return normalized
        else:
            return intensities
    
    def apply_0_100_scaling(self, wavelengths, intensities):
        """Apply 0-100 scaling for analysis and visualization"""
        min_val, max_val = intensities.min(), intensities.max()
        if max_val != min_val:
            scaled = (intensities - min_val) * 100 / (max_val - min_val)
            return scaled
        else:
            return intensities
    
    def play_bleep(self, frequency=800, duration=200, feature_type="standard"):
        """Play audio bleep for feature detection"""
        if not self.bleep_enabled or not HAS_AUDIO:
            return
        
        try:
            # Different frequencies for different feature types
            freq_map = {
                "peak": 1000,
                "valley": 600,
                "plateau": 800,
                "significant": 1200,
                "completion": 400
            }
            
            freq = freq_map.get(feature_type, frequency)
            
            if 'winsound' in sys.modules:
                winsound.Beep(freq, duration)
            elif 'pygame' in sys.modules:
                # Generate tone using pygame
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
            print(f"Audio bleep error: {e}")
    
    def calculate_relative_height(self, gem_id, wavelength_target, tolerance=5.0):
        """Calculate relative height measurements across light sources"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query for features near the target wavelength across all light sources
            query = """
                SELECT light_source, wavelength, intensity, file, feature_group
                FROM structural_features 
                WHERE file LIKE ? AND ABS(wavelength - ?) <= ?
                ORDER BY light_source, ABS(wavelength - ?)
            """
            
            df = pd.read_sql_query(query, conn, params=[f"%{gem_id}%", wavelength_target, tolerance, wavelength_target])
            conn.close()
            
            if df.empty:
                return None
            
            # Group by light source and get closest match
            relative_measurements = {}
            
            for light_source in ['B', 'L', 'U']:
                light_data = df[df['light_source'] == light_source]
                if not light_data.empty:
                    closest = light_data.iloc[0]
                    relative_measurements[light_source] = {
                        'wavelength': closest['wavelength'],
                        'intensity': closest['intensity'],
                        'feature_group': closest['feature_group'],
                        'file': closest['file'],
                        'wavelength_diff': abs(closest['wavelength'] - wavelength_target)
                    }
            
            # Calculate relative ratios
            if len(relative_measurements) >= 2:
                intensities = [data['intensity'] for data in relative_measurements.values()]
                max_intensity = max(intensities)
                
                for light_source, data in relative_measurements.items():
                    data['relative_height'] = data['intensity'] / max_intensity
                    data['percentage'] = (data['intensity'] / max_intensity) * 100
            
            return relative_measurements
            
        except Exception as e:
            print(f"Error calculating relative height: {e}")
            return None
    
    def auto_import_csv_to_database(self, csv_file_path):
        """Legacy method - redirects to enhanced version"""
        return self.auto_import_csv_to_database_enhanced(csv_file_path)
    
    def detect_light_source_from_path(self, file_path):
        """Detect light source from file path or filename"""
        path_lower = file_path.lower()
        
        if 'halogen' in path_lower or '_b' in path_lower:
            return 'Halogen'
        elif 'laser' in path_lower or '_l' in path_lower:
            return 'Laser'
        elif 'uv' in path_lower or '_u' in path_lower:
            return 'UV'
        else:
            return 'Unknown'
    
    def safe_float(self, value):
        """Safely convert value to float or return None"""
        if pd.isna(value) or value == '' or value is None:
            return None
        try:
            return float(value)
        except:
            return None
    
    def batch_import_structural_data(self):
        """Enhanced batch import with auto-detection and validation"""
        print("\nENHANCED BATCH IMPORT SYSTEM")
        print("=" * 50)
        
        # Update schema first to support ULTRA_OPTIMIZED
        self.update_database_schema_for_ultra_optimized()
        
        # Scan for structural CSV files
        all_csv_files = []
        light_source_counts = {'halogen': 0, 'laser': 0, 'uv': 0}
        
        for light_source in ['halogen', 'laser', 'uv']:
            source_dir = os.path.join(self.structural_data_dir, light_source)
            if os.path.exists(source_dir):
                csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
                light_source_counts[light_source] = len(csv_files)
                
                for csv_file in csv_files:
                    full_path = os.path.join(source_dir, csv_file)
                    all_csv_files.append((full_path, light_source))
        
        print(f"Found structural data files:")
        for light_source, count in light_source_counts.items():
            print(f"   {light_source.capitalize()}: {count} files")
        
        if not all_csv_files:
            print("No structural CSV files found!")
            return
        
        print(f"\nTotal files to import: {len(all_csv_files)}")
        
        # Confirm import
        confirm = input("\nProceed with enhanced batch import? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Import cancelled")
            return
        
        # Import files using enhanced system
        successful_imports = 0
        failed_imports = 0
        ultra_optimized_count = 0
        standard_count = 0
        
        print(f"\nStarting enhanced batch import...")
        
        for file_path, light_source in all_csv_files:
            print(f"   Processing: {os.path.basename(file_path)}")
            
            # Check format before importing
            try:
                df = pd.read_csv(file_path)
                csv_format = "ULTRA_OPTIMIZED" if len(df.columns) >= 35 else "STANDARD"
                
                success = self.auto_import_csv_to_database_enhanced(file_path)
                if success:
                    successful_imports += 1
                    if csv_format == "ULTRA_OPTIMIZED":
                        ultra_optimized_count += 1
                    else:
                        standard_count += 1
                else:
                    failed_imports += 1
            except Exception as e:
                print(f"      Error processing file: {e}")
                failed_imports += 1
        
        print(f"\nENHANCED BATCH IMPORT COMPLETED:")
        print(f"   Successful: {successful_imports}")
        print(f"   Failed: {failed_imports}")
        print(f"   ULTRA_OPTIMIZED: {ultra_optimized_count}")
        print(f"   Standard: {standard_count}")
        
        if successful_imports > 0:
            self.play_bleep(feature_type="completion")
            
        # Show database statistics
        self.show_enhanced_database_stats()
    
    def show_enhanced_database_stats(self):
        """Show enhanced database statistics with detailed breakdowns"""
        print("\nENHANCED DATABASE STATISTICS")
        print("=" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Basic counts
            basic_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT file) as unique_files,
                    COUNT(DISTINCT light_source) as light_sources
                FROM structural_features
            """, conn)
            
            if basic_stats.iloc[0]['total_records'] == 0:
                print("Database is empty - no structural features found")
                conn.close()
                return
            
            total_records = basic_stats.iloc[0]['total_records']
            unique_files = basic_stats.iloc[0]['unique_files']
            light_sources = basic_stats.iloc[0]['light_sources']
            
            print(f"Total records: {total_records:,}")
            print(f"Unique files: {unique_files:,}")
            print(f"Light sources: {light_sources}")
            
            # By light source
            light_stats = pd.read_sql_query("""
                SELECT light_source, COUNT(*) as count, COUNT(DISTINCT file) as files
                FROM structural_features 
                GROUP BY light_source 
                ORDER BY count DESC
            """, conn)
            
            print(f"\nBY LIGHT SOURCE:")
            for _, row in light_stats.iterrows():
                print(f"   {row['light_source']}: {row['count']:,} records ({row['files']} files)")
            
            # Check for ULTRA_OPTIMIZED data
            ultra_stats = pd.read_sql_query("""
                SELECT COUNT(*) as ultra_count
                FROM structural_features 
                WHERE data_type LIKE '%ultra_optimized%'
            """, conn)
            
            ultra_count = ultra_stats.iloc[0]['ultra_count']
            print(f"\nULTRA_OPTIMIZED records: {ultra_count:,}")
            
            # By feature type
            feature_stats = pd.read_sql_query("""
                SELECT feature_group, COUNT(*) as count
                FROM structural_features 
                GROUP BY feature_group 
                ORDER BY count DESC
                LIMIT 10
            """, conn)
            
            print(f"\nTOP FEATURE TYPES:")
            for _, row in feature_stats.iterrows():
                print(f"   {row['feature_group']}: {row['count']:,}")
            
            # Recent activity
            recent_stats = pd.read_sql_query("""
                SELECT COUNT(*) as recent_count
                FROM structural_features 
                WHERE timestamp > datetime('now', '-7 days')
            """, conn)
            
            recent_count = recent_stats.iloc[0]['recent_count']
            print(f"\nRecent activity (7 days): {recent_count:,} records")
            
            # Normalization status
            norm_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(CASE WHEN normalization_scheme IS NOT NULL THEN 1 END) as with_normalization,
                    COUNT(CASE WHEN normalization_compatible = 1 THEN 1 END) as compatible
                FROM structural_features
            """, conn)
            
            with_norm = norm_stats.iloc[0]['with_normalization']
            compatible = norm_stats.iloc[0]['compatible']
            
            print(f"\nNORMALIZATION STATUS:")
            print(f"   With metadata: {with_norm:,} / {total_records:,}")
            print(f"   Compatible: {compatible:,} / {total_records:,}")
            
            conn.close()
            
        except Exception as e:
            print(f"Error reading database: {e}")
    
    def enhanced_database_search(self):
        """Enhanced database search with multiple criteria"""
        print("\nENHANCED DATABASE SEARCH")
        print("=" * 40)
        
        if not os.path.exists(self.db_path):
            print("Database not found")
            return
        
        # Search options
        print("Search options:")
        print("1. By gem/file name")
        print("2. By wavelength range")
        print("3. By feature type")
        print("4. By light source")
        print("5. Recent features (last 7 days)")
        print("6. ULTRA_OPTIMIZED data only")
        
        choice = input("\nSelect search type (1-6): ").strip()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            if choice == '1':
                search_term = input("Enter gem/file name (partial match): ").strip()
                query = "SELECT * FROM structural_features WHERE file LIKE ? ORDER BY wavelength"
                df = pd.read_sql_query(query, conn, params=[f"%{search_term}%"])
                
            elif choice == '2':
                min_wl = float(input("Minimum wavelength: "))
                max_wl = float(input("Maximum wavelength: "))
                query = "SELECT * FROM structural_features WHERE wavelength BETWEEN ? AND ? ORDER BY wavelength"
                df = pd.read_sql_query(query, conn, params=[min_wl, max_wl])
                
            elif choice == '3':
                feature_type = input("Enter feature type: ").strip()
                query = "SELECT * FROM structural_features WHERE feature_group LIKE ? ORDER BY wavelength"
                df = pd.read_sql_query(query, conn, params=[f"%{feature_type}%"])
                
            elif choice == '4':
                light_source = input("Enter light source (B/L/U/Halogen/Laser/UV): ").strip()
                query = "SELECT * FROM structural_features WHERE light_source LIKE ? ORDER BY wavelength"
                df = pd.read_sql_query(query, conn, params=[f"%{light_source}%"])
                
            elif choice == '5':
                query = "SELECT * FROM structural_features WHERE timestamp > datetime('now', '-7 days') ORDER BY timestamp DESC"
                df = pd.read_sql_query(query, conn)
                
            elif choice == '6':
                query = "SELECT * FROM structural_features WHERE data_type LIKE '%ultra_optimized%' ORDER BY wavelength"
                df = pd.read_sql_query(query, conn)
                
            else:
                print("Invalid choice")
                conn.close()
                return
            
            conn.close()
            
            if df.empty:
                print("No results found")
                return
            
            print(f"\nFound {len(df)} results:")
            print("=" * 80)
            
            # Display results with enhanced formatting
            for i, (_, row) in enumerate(df.iterrows(), 1):
                data_type = row.get('data_type', 'standard')
                format_indicator = "ULTRA" if 'ultra_optimized' in str(data_type) else "STD"
                
                print(f"{i:3}. {format_indicator} {row['file']} | {row['light_source']} | {row['wavelength']:.1f}nm | {row['feature_group']} | {row['intensity']:.2f}")
                
                if i >= 20:  # Limit display
                    remaining = len(df) - 20
                    if remaining > 0:
                        print(f"    ... and {remaining} more results")
                    break
            
            # Offer to save results
            save = input(f"\nSave results to CSV? (y/n): ").strip().lower()
            if save == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"database_search_results_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"Results saved to {filename}")
                
        except Exception as e:
            print(f"Search error: {e}")
    
    def enhanced_comparison_analysis(self):
        """Enhanced comparison analysis using database"""
        print("\nENHANCED COMPARISON ANALYSIS")
        print("=" * 45)
        
        if not os.path.exists(self.db_path):
            print("Database not found")
            return
        
        # Get available gems for comparison
        try:
            conn = sqlite3.connect(self.db_path)
            
            gems_df = pd.read_sql_query("""
                SELECT DISTINCT file, light_source, COUNT(*) as feature_count
                FROM structural_features 
                GROUP BY file, light_source
                HAVING feature_count >= 3
                ORDER BY file, light_source
            """, conn)
            
            if gems_df.empty:
                print("No gems with sufficient features found")
                conn.close()
                return
            
            # Group by gem (file without light source suffix)
            gem_groups = {}
            for _, row in gems_df.iterrows():
                base_name = row['file']
                # Remove common light source suffixes
                for suffix in ['_B', '_L', '_U', '_halogen', '_laser', '_uv']:
                    if base_name.endswith(suffix):
                        base_name = base_name[:-len(suffix)]
                        break
                
                if base_name not in gem_groups:
                    gem_groups[base_name] = []
                gem_groups[base_name].append(row['light_source'])
            
            # Show available gems
            print("Available gems for comparison:")
            gem_list = []
            for i, (gem_name, light_sources) in enumerate(gem_groups.items(), 1):
                print(f"   {i}. {gem_name} ({'+'.join(light_sources)})")
                gem_list.append(gem_name)
            
            # Select gems for comparison
            if len(gem_list) < 2:
                print("Need at least 2 gems for comparison")
                conn.close()
                return
            
            choice1 = int(input(f"\nSelect first gem (1-{len(gem_list)}): ")) - 1
            choice2 = int(input(f"Select second gem (1-{len(gem_list)}): ")) - 1
            
            if choice1 < 0 or choice1 >= len(gem_list) or choice2 < 0 or choice2 >= len(gem_list):
                print("Invalid selection")
                conn.close()
                return
            
            gem1 = gem_list[choice1]
            gem2 = gem_list[choice2]
            
            # Perform comparison
            self.perform_gem_comparison(conn, gem1, gem2)
            conn.close()
            
        except Exception as e:
            print(f"Comparison error: {e}")
    
    def perform_gem_comparison(self, conn, gem1, gem2):
        """Perform detailed comparison between two gems"""
        print(f"\nCOMPARING: {gem1} vs {gem2}")
        print("=" * 60)
        
        # Get features for both gems
        for gem_name in [gem1, gem2]:
            query = """
                SELECT light_source, wavelength, intensity, feature_group, data_type
                FROM structural_features 
                WHERE file LIKE ?
                ORDER BY light_source, wavelength
            """
            
            gem_data = pd.read_sql_query(query, conn, params=[f"%{gem_name}%"])
            
            if gem_data.empty:
                print(f"No data found for {gem_name}")
                continue
            
            # Check if has ULTRA_OPTIMIZED data
            ultra_count = len(gem_data[gem_data['data_type'].str.contains('ultra_optimized', na=False)])
            format_indicator = f" ({ultra_count} ULTRA_OPTIMIZED)" if ultra_count > 0 else ""
            
            print(f"\n{gem_name.upper()}{format_indicator}:")
            
            # Group by light source
            for light_source in ['B', 'L', 'U', 'Halogen', 'Laser', 'UV']:
                light_data = gem_data[gem_data['light_source'] == light_source]
                if not light_data.empty:
                    print(f"   {light_source} Light ({len(light_data)} features):")
                    
                    # Show top features by intensity
                    top_features = light_data.nlargest(3, 'intensity')
                    for _, feature in top_features.iterrows():
                        ultra_indicator = "ULTRA" if 'ultra_optimized' in str(feature.get('data_type', '')) else "STD"
                        print(f"      {ultra_indicator} {feature['wavelength']:.1f}nm: {feature['feature_group']} (I:{feature['intensity']:.2f})")
                    
                    # Calculate relative heights
                    if len(light_data) >= 2:
                        wavelength_range = light_data['wavelength'].max() - light_data['wavelength'].min()
                        intensity_range = light_data['intensity'].max() - light_data['intensity'].min()
                        print(f"      Range: {wavelength_range:.1f}nm, Intensity span: {intensity_range:.2f}")
        
        # Find common wavelengths for direct comparison
        print(f"\nDIRECT COMPARISON:")
        
        gem1_data = pd.read_sql_query("""
            SELECT wavelength, intensity, light_source, feature_group, data_type
            FROM structural_features WHERE file LIKE ?
        """, conn, params=[f"%{gem1}%"])
        
        gem2_data = pd.read_sql_query("""
            SELECT wavelength, intensity, light_source, feature_group, data_type
            FROM structural_features WHERE file LIKE ?
        """, conn, params=[f"%{gem2}%"])
        
        # Find similar wavelengths (within 5nm)
        common_features = []
        tolerance = 5.0
        
        for _, feature1 in gem1_data.iterrows():
            close_features = gem2_data[
                (gem2_data['light_source'] == feature1['light_source']) &
                (abs(gem2_data['wavelength'] - feature1['wavelength']) <= tolerance)
            ]
            
            if not close_features.empty:
                closest = close_features.iloc[np.argmin(abs(close_features['wavelength'] - feature1['wavelength']))]
                
                ratio = feature1['intensity'] / closest['intensity'] if closest['intensity'] != 0 else float('inf')
                
                common_features.append({
                    'light_source': feature1['light_source'],
                    'wavelength1': feature1['wavelength'],
                    'wavelength2': closest['wavelength'],
                    'intensity1': feature1['intensity'],
                    'intensity2': closest['intensity'],
                    'ratio': ratio,
                    'feature1': feature1['feature_group'],
                    'feature2': closest['feature_group'],
                    'data_type1': feature1.get('data_type', ''),
                    'data_type2': closest.get('data_type', '')
                })
        
        if common_features:
            print(f"   Found {len(common_features)} comparable features:")
            
            for cf in common_features[:10]:  # Show top 10
                wl_diff = abs(cf['wavelength1'] - cf['wavelength2'])
                ratio_str = f"{cf['ratio']:.2f}" if cf['ratio'] != float('inf') else "∞"
                
                ultra1 = "ULTRA" if 'ultra_optimized' in str(cf['data_type1']) else "STD"
                ultra2 = "ULTRA" if 'ultra_optimized' in str(cf['data_type2']) else "STD"
                
                print(f"   {cf['light_source']} | {cf['wavelength1']:.1f}nm vs {cf['wavelength2']:.1f}nm")
                print(f"      {ultra1} {gem1}: {cf['feature1']} (I:{cf['intensity1']:.2f})")
                print(f"      {ultra2} {gem2}: {cf['feature2']} (I:{cf['intensity2']:.2f})")
                print(f"      Ratio: {ratio_str}, Δλ: {wl_diff:.1f}nm")
                print()
        else:
            print("   No comparable features found within tolerance")
        
        # Play completion bleep
        self.play_bleep(feature_type="completion")
    
    def toggle_bleep_system(self):
        """Toggle audio bleep feedback system"""
        self.bleep_enabled = not self.bleep_enabled
        status = "ENABLED" if self.bleep_enabled else "DISABLED"
        
        if HAS_AUDIO:
            print(f"Audio bleep system: {status}")
            if self.bleep_enabled:
                self.play_bleep(feature_type="completion")
                print("   Features will play audio feedback")
            else:
                print("   No audio feedback will be played")
        else:
            print(f"Audio system not available (no winsound/pygame)")
            self.bleep_enabled = False
    
    def toggle_auto_import(self):
        """Toggle automatic CSV import system"""
        self.auto_import_enabled = not self.auto_import_enabled
        status = "ENABLED" if self.auto_import_enabled else "DISABLED"
        print(f"Auto-import system: {status}")
        
        if self.auto_import_enabled:
            print("   CSV files will be automatically imported to database")
            print("   Supports both STANDARD (23) and ULTRA_OPTIMIZED (41+) formats")
        else:
            print("   Manual import required for CSV files")
    
    def relative_height_analysis(self):
        """Perform relative height analysis across light sources"""
        print("\nRELATIVE HEIGHT ANALYSIS")
        print("=" * 40)
        
        gem_id = input("Enter gem ID for analysis: ").strip()
        
        try:
            wavelength_input = input("Enter target wavelength (nm): ").strip()
            if not wavelength_input:
                print("No wavelength provided")
                return
            wavelength = float(wavelength_input)
        except ValueError:
            print("Invalid wavelength - please enter a number")
            return
        
        tolerance = float(input("Enter tolerance (nm) [default: 5.0]: ") or "5.0")
        
        measurements = self.calculate_relative_height(gem_id, wavelength, tolerance)
        
        if not measurements:
            print(f"No measurements found for {gem_id} near {wavelength}nm")
            return
        
        print(f"\nRELATIVE HEIGHT ANALYSIS: {gem_id}")
        print(f"Target wavelength: {wavelength}nm (±{tolerance}nm)")
        print("=" * 50)
        
        # Display measurements
        for light_source, data in measurements.items():
            actual_wl = data['wavelength']
            intensity = data['intensity']
            relative = data.get('relative_height', 0)
            percentage = data.get('percentage', 0)
            feature = data['feature_group']
            wl_diff = data['wavelength_diff']
            
            print(f"{light_source} Light:")
            print(f"   Wavelength: {actual_wl:.1f}nm (Δ{wl_diff:.1f}nm)")
            print(f"   Intensity: {intensity:.2f}")
            print(f"   Relative height: {relative:.3f} ({percentage:.1f}%)")
            print(f"   Feature type: {feature}")
            print()
        
        # Find dominant light source
        if len(measurements) >= 2:
            intensities = [(ls, data['intensity']) for ls, data in measurements.items()]
            dominant = max(intensities, key=lambda x: x[1])
            
            print(f"DOMINANT: {dominant[0]} light (intensity: {dominant[1]:.2f})")
            
            # Calculate ratios
            print(f"\nINTENSITY RATIOS:")
            for light_source, data in measurements.items():
                if light_source != dominant[0]:
                    ratio = data['intensity'] / dominant[1]
                    print(f"   {light_source}/{dominant[0]}: {ratio:.3f}")
        
        # Cache results
        cache_key = f"{gem_id}_{wavelength:.1f}"
        self.relative_height_cache[cache_key] = measurements
        
        self.play_bleep(feature_type="completion")
    
    def structural_matching_analysis(self):
        """Option 11: Structural matching analysis with self-validation testing"""
        print("\nSTRUCTURAL MATCHING ANALYSIS")
        print("=" * 50)
        
        if not os.path.exists(self.db_path):
            print(f"Structural database not found: {self.db_path}")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get all gems with structural data
            query = """
                SELECT 
                    CASE 
                        WHEN file LIKE '%B%' THEN SUBSTR(file, 1, INSTR(file, 'B') - 1)
                        WHEN file LIKE '%L%' THEN SUBSTR(file, 1, INSTR(file, 'L') - 1)
                        WHEN file LIKE '%U%' THEN SUBSTR(file, 1, INSTR(file, 'U') - 1)
                        ELSE SUBSTR(file, 1, INSTR(file || '_', '_') - 1)
                    END as gem_id,
                    GROUP_CONCAT(DISTINCT light_source) as light_sources,
                    COUNT(DISTINCT light_source) as light_count,
                    COUNT(*) as total_features,
                    COUNT(CASE WHEN data_type LIKE '%ultra_optimized%' THEN 1 END) as ultra_features
                FROM structural_features 
                WHERE file NOT LIKE '%unknown%'
                GROUP BY gem_id
                HAVING light_count >= 2
                ORDER BY light_count DESC, gem_id
            """
            
            gems_df = pd.read_sql_query(query, conn)
            
            if gems_df.empty:
                print("No gems with multi-light structural data found")
                print("Need gems with at least 2 light sources (B, L, U) for testing")
                conn.close()
                return
            
            print(f"Found {len(gems_df)} gems suitable for structural matching:")
            print("=" * 70)
            print(f"{'#':<3} {'Gem ID':<15} {'Light Sources':<20} {'Features':<10} {'ULTRA':<8} {'Status'}")
            print("-" * 70)
            
            for i, row in gems_df.iterrows():
                gem_id = row['gem_id']
                light_sources = row['light_sources']
                feature_count = row['total_features']
                light_count = row['light_count']
                ultra_count = row['ultra_features']
                
                # Status indicator
                if light_count == 3:
                    status = "COMPLETE (B+L+U)"
                elif light_count == 2:
                    status = "PARTIAL (2/3)"
                else:
                    status = "INSUFFICIENT"
                
                ultra_indicator = f"ULTRA {ultra_count}" if ultra_count > 0 else f"STD 0"
                
                print(f"{i+1:<3} {gem_id:<15} {light_sources:<20} {feature_count:<10} {ultra_indicator:<8} {status}")
            
            # Selection menu
            print(f"\nSELECT GEM FOR STRUCTURAL MATCHING TEST:")
            choice = input(f"Enter gem number (1-{len(gems_df)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                conn.close()
                return
            
            try:
                gem_idx = int(choice) - 1
                if gem_idx < 0 or gem_idx >= len(gems_df):
                    print("Invalid selection")
                    conn.close()
                    return
                
                selected_gem = gems_df.iloc[gem_idx]
                gem_id = selected_gem['gem_id']
                available_lights = selected_gem['light_sources'].split(',')
                ultra_count = selected_gem['ultra_features']
                
                print(f"\nSELECTED: {gem_id}")
                print(f"   Available light sources: {', '.join(available_lights)}")
                print(f"   ULTRA_OPTIMIZED features: {ultra_count}")
                
                # Show detailed breakdown
                self.show_gem_structural_details(conn, gem_id, available_lights)
                
                # Confirm testing
                confirm = input(f"\nTest structural matching for {gem_id}? (y/n): ").strip().lower()
                if confirm != 'y':
                    conn.close()
                    return
                
                # Extract gem data and run matching test
                self.run_structural_matching_test(conn, gem_id, available_lights)
                
            except ValueError:
                print("Please enter a valid number")
            
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")
    
    def show_gem_structural_details(self, conn, gem_id, available_lights):
        """Show detailed structural data for selected gem"""
        print(f"\nSTRUCTURAL DATA DETAILS: {gem_id}")
        print("=" * 60)
        
        for light_source in available_lights:
            # Get features for this light source
            query = """
                SELECT file, feature_group, wavelength, intensity, data_type, COUNT(*) as count
                FROM structural_features 
                WHERE file LIKE ? AND light_source = ?
                GROUP BY file, feature_group
                ORDER BY feature_group, wavelength
            """
            
            features_df = pd.read_sql_query(query, conn, params=[f"{gem_id}%", light_source])
            
            if not features_df.empty:
                total_features = features_df['count'].sum()
                feature_types = features_df['feature_group'].unique()
                ultra_count = len(features_df[features_df['data_type'].str.contains('ultra_optimized', na=False)])
                
                ultra_indicator = f" ({ultra_count} ULTRA_OPTIMIZED)" if ultra_count > 0 else ""
                
                print(f"{light_source} Light Source{ultra_indicator}:")
                print(f"   Files: {', '.join(features_df['file'].unique())}")
                print(f"   Total features: {total_features}")
                print(f"   Feature types: {', '.join(feature_types)}")
                
                # Show feature breakdown
                for feature_type in feature_types:
                    type_data = features_df[features_df['feature_group'] == feature_type]
                    count = type_data['count'].sum()
                    wavelengths = pd.read_sql_query("""
                        SELECT wavelength, data_type FROM structural_features 
                        WHERE file LIKE ? AND light_source = ? AND feature_group = ?
                        ORDER BY wavelength
                    """, conn, params=[f"{gem_id}%", light_source, feature_type])
                    
                    if not wavelengths.empty:
                        wl_range = f"{wavelengths['wavelength'].min():.1f}-{wavelengths['wavelength'].max():.1f}nm"
                        ultra_in_type = len(wavelengths[wavelengths['data_type'].str.contains('ultra_optimized', na=False)])
                        ultra_type_indicator = f" (ULTRA {ultra_in_type})" if ultra_in_type > 0 else ""
                        print(f"      • {feature_type}: {count} features ({wl_range}){ultra_type_indicator}")
                print()
    
    def run_structural_matching_test(self, conn, gem_id, available_lights):
        """Run structural matching test with self-validation"""
        print(f"\nSTRUCTURAL MATCHING TEST: {gem_id}")
        print("=" * 50)
        
        # For each light source, extract data and test matching
        test_results = {}
        
        for light_source in available_lights:
            print(f"\nTesting {light_source} Light Source...")
            
            # Extract structural features for this light source
            query = """
                SELECT file, feature_group, wavelength, intensity, data_type,
                       start_wavelength, end_wavelength, midpoint_wavelength,
                       crest_wavelength, max_wavelength
                FROM structural_features 
                WHERE file LIKE ? AND light_source = ?
                ORDER BY wavelength
            """
            
            test_data_df = pd.read_sql_query(query, conn, params=[f"{gem_id}%", light_source])
            
            if test_data_df.empty:
                print(f"   No data found for {light_source}")
                continue
            
            # Count ULTRA_OPTIMIZED features
            ultra_count = len(test_data_df[test_data_df['data_type'].str.contains('ultra_optimized', na=False)])
            ultra_indicator = f" ({ultra_count} ULTRA_OPTIMIZED)" if ultra_count > 0 else ""
            
            # Convert to format expected by matching algorithms
            test_features = []
            for _, row in test_data_df.iterrows():
                feature = {
                    'feature_type': row['feature_group'],
                    'wavelength': row['wavelength'],
                    'intensity': row['intensity'],
                    'is_ultra_optimized': 'ultra_optimized' in str(row.get('data_type', ''))
                }
                
                # Add optional wavelength fields
                for field in ['start_wavelength', 'end_wavelength', 'midpoint_wavelength', 
                             'crest_wavelength', 'max_wavelength']:
                    if pd.notna(row[field]):
                        feature[field] = row[field]
                
                test_features.append(feature)
            
            print(f"   Extracted {len(test_features)} features{ultra_indicator}")
            
            # Find all potential matches in database for this light source
            match_query = """
                SELECT 
                    CASE 
                        WHEN file LIKE '%B%' THEN SUBSTR(file, 1, INSTR(file, 'B') - 1)
                        WHEN file LIKE '%L%' THEN SUBSTR(file, 1, INSTR(file, 'L') - 1)
                        WHEN file LIKE '%U%' THEN SUBSTR(file, 1, INSTR(file, 'U') - 1)
                        ELSE SUBSTR(file, 1, INSTR(file || '_', '_') - 1)
                    END as candidate_gem_id,
                    file, COUNT(*) as feature_count,
                    COUNT(CASE WHEN data_type LIKE '%ultra_optimized%' THEN 1 END) as ultra_count
                FROM structural_features 
                WHERE light_source = ? AND file NOT LIKE ?
                GROUP BY candidate_gem_id, file
                HAVING feature_count >= 3
                ORDER BY feature_count DESC
            """
            
            candidates_df = pd.read_sql_query(match_query, conn, params=[light_source, f"{gem_id}%"])
            
            if candidates_df.empty:
                print(f"   No candidates found for comparison")
                continue
            
            print(f"   Found {len(candidates_df)} candidates for comparison")
            
            # Run matching against each candidate
            match_scores = []
            
            for _, candidate_row in candidates_df.iterrows()[:10]:  # Test top 10 candidates
                candidate_gem_id = candidate_row['candidate_gem_id']
                candidate_file = candidate_row['file']
                candidate_ultra_count = candidate_row['ultra_count']
                
                # Get candidate features
                candidate_query = """
                    SELECT file, feature_group, wavelength, intensity, data_type,
                           start_wavelength, end_wavelength, midpoint_wavelength,
                           crest_wavelength, max_wavelength
                    FROM structural_features 
                    WHERE file = ?
                    ORDER BY wavelength
                """
                
                candidate_df = pd.read_sql_query(candidate_query, conn, params=[candidate_file])
                
                candidate_features = []
                for _, row in candidate_df.iterrows():
                    feature = {
                        'feature_type': row['feature_group'],
                        'wavelength': row['wavelength'],
                        'intensity': row['intensity'],
                        'is_ultra_optimized': 'ultra_optimized' in str(row.get('data_type', ''))
                    }
                    
                    # Add optional wavelength fields
                    for field in ['start_wavelength', 'end_wavelength', 'midpoint_wavelength',
                                 'crest_wavelength', 'max_wavelength']:
                        if pd.notna(row[field]):
                            feature[field] = row[field]
                    
                    candidate_features.append(feature)
                
                # Calculate match score using simple wavelength comparison
                match_score = self.calculate_simple_structural_match(test_features, candidate_features)
                
                match_scores.append({
                    'candidate_gem_id': candidate_gem_id,
                    'candidate_file': candidate_file,
                    'score': match_score,
                    'feature_count': len(candidate_features),
                    'ultra_count': candidate_ultra_count
                })
            
            # Sort by score
            match_scores.sort(key=lambda x: x['score'], reverse=True)
            test_results[light_source] = match_scores
            
            # Show results
            print(f"   TOP MATCHES:")
            for i, match in enumerate(match_scores[:5], 1):
                score_indicator = "GOOD" if match['score'] > 90 else "OK" if match['score'] > 70 else "POOR"
                self_match = "SELF-MATCH!" if match['candidate_gem_id'] == gem_id else ""
                ultra_indicator = f" (ULTRA {match['ultra_count']})" if match['ultra_count'] > 0 else ""
                print(f"      {i}. {match['candidate_gem_id']}{ultra_indicator}: {match['score']:.1f}% {score_indicator} {self_match}")
        
        # Summary results
        print(f"\nSTRUCTURAL MATCHING TEST SUMMARY: {gem_id}")
        print("=" * 60)
        
        self_match_found = False
        for light_source, matches in test_results.items():
            if matches:
                top_match = matches[0]
                is_self = top_match['candidate_gem_id'] == gem_id
                if is_self:
                    self_match_found = True
                
                status = "SELF-MATCH!" if is_self else f"Matched: {top_match['candidate_gem_id']}"
                ultra_indicator = f" (ULTRA {top_match['ultra_count']})" if top_match['ultra_count'] > 0 else ""
                print(f"   {light_source}: {top_match['score']:.1f}% - {status}{ultra_indicator}")
        
        print(f"\nOVERALL TEST RESULT:")
        if self_match_found:
            print("   SUCCESS: Structural matching can identify gems correctly!")
            print("   Ready to test with unknown gems")
        else:
            print("   WARNING: Self-matching failed - need to tune algorithms")
            print("   Consider adjusting wavelength tolerances")
        
        # Show algorithm suggestions
        print(f"\nALGORITHM STATUS:")
        print("   Current: Simple wavelength comparison")
        print("   Available: Enhanced UV ratio analysis, multi-light integration")
        print("   ULTRA_OPTIMIZED: Enhanced baseline analysis, width classification")
        print("   Recommendation: Integrate enhanced_gem_analyzer.py for production use")
    
    def calculate_simple_structural_match(self, test_features, candidate_features):
        """Simple structural matching for testing"""
        if not test_features or not candidate_features:
            return 0.0
        
        # Group features by type
        test_by_type = {}
        candidate_by_type = {}
        
        for feature in test_features:
            feature_type = feature['feature_type']
            if feature_type not in test_by_type:
                test_by_type[feature_type] = []
            test_by_type[feature_type].append(feature)
        
        for feature in candidate_features:
            feature_type = feature['feature_type']
            if feature_type not in candidate_by_type:
                candidate_by_type[feature_type] = []
            candidate_by_type[feature_type].append(feature)
        
        # Find common feature types
        common_types = set(test_by_type.keys()) & set(candidate_by_type.keys())
        
        if not common_types:
            return 0.0
        
        # Calculate matching score for common types
        type_scores = []
        
        for feature_type in common_types:
            test_features_type = test_by_type[feature_type]
            candidate_features_type = candidate_by_type[feature_type]
            
            # For each test feature, find best match in candidates
            total_score = 0.0
            for test_feature in test_features_type:
                test_wl = test_feature.get('wavelength', 0)
                test_ultra = test_feature.get('is_ultra_optimized', False)
                
                best_score = 0.0
                for candidate_feature in candidate_features_type:
                    candidate_wl = candidate_feature.get('wavelength', 0)
                    candidate_ultra = candidate_feature.get('is_ultra_optimized', False)
                    diff = abs(test_wl - candidate_wl)
                    
                    # Simple scoring based on wavelength difference
                    if diff <= 0.5:
                        score = 100.0
                    elif diff <= 1.0:
                        score = 95.0
                    elif diff <= 2.0:
                        score = 85.0
                    elif diff <= 5.0:
                        score = 70.0
                    elif diff <= 10.0:
                        score = 50.0
                    else:
                        score = max(0.0, 50.0 - diff * 2)
                    
                    # Bonus for ULTRA_OPTIMIZED matches
                    if test_ultra and candidate_ultra:
                        score *= 1.05  # 5% bonus for both being ULTRA_OPTIMIZED
                    
                    best_score = max(best_score, score)
                
                total_score += best_score
            
            type_score = total_score / len(test_features_type) if test_features_type else 0.0
            type_scores.append(type_score)
        
        # Average across all feature types
        final_score = sum(type_scores) / len(type_scores) if type_scores else 0.0
        return final_score
    
    def debug_structural_database(self):
        """Debug function to examine structural database content"""
        print("\nSTRUCTURAL DATABASE DEBUG")
        print("=" * 50)
        
        if not os.path.exists(self.db_path):
            print(f"Database not found: {self.db_path}")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check table structure
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(structural_features)")
            columns = cursor.fetchall()
            
            print("TABLE STRUCTURE:")
            ultra_columns = 0
            for col in columns:
                is_ultra = col[1] in ['feature_key', 'baseline_quality', 'baseline_width_nm', 
                                     'baseline_cv_percent', 'total_width_nm', 'skew_severity', 
                                     'width_class', 'analysis_date', 'analyzer_version']
                if is_ultra:
                    ultra_columns += 1
                    print(f"   ULTRA {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULLABLE'} [ULTRA_OPTIMIZED]")
                else:
                    print(f"   STD {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULLABLE'}")
            
            print(f"\nULTRA_OPTIMIZED columns available: {ultra_columns}")
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total_records = cursor.fetchone()[0]
            print(f"\nTotal records: {total_records:,}")
            
            if total_records == 0:
                print("No structural features found in database")
                conn.close()
                return
            
            # Check for ULTRA_OPTIMIZED data
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE data_type LIKE '%ultra_optimized%'")
            ultra_records = cursor.fetchone()[0]
            print(f"ULTRA_OPTIMIZED records: {ultra_records:,} ({ultra_records/total_records*100:.1f}%)")
            
            # Check light sources
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source")
            light_sources = cursor.fetchall()
            print(f"\nLight Sources:")
            for light, count in light_sources:
                print(f"   {light}: {count:,} records")
            
            # Check format distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN data_type LIKE '%ultra_optimized%' THEN 'ULTRA_OPTIMIZED'
                        ELSE 'STANDARD'
                    END as format_type,
                    COUNT(*) as count
                FROM structural_features 
                GROUP BY format_type
            """)
            format_dist = cursor.fetchall()
            print(f"\nFORMAT DISTRIBUTION:")
            for format_type, count in format_dist:
                indicator = "ULTRA" if format_type == "ULTRA_OPTIMIZED" else "STD"
                print(f"   {indicator} {format_type}: {count:,} records")
            
            # Find complete gems (with multiple light sources)
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN file LIKE '%B%' THEN SUBSTR(file, 1, INSTR(file, 'B') - 1)
                        WHEN file LIKE '%L%' THEN SUBSTR(file, 1, INSTR(file, 'L') - 1)
                        WHEN file LIKE '%U%' THEN SUBSTR(file, 1, INSTR(file, 'U') - 1)
                        ELSE SUBSTR(file, 1, INSTR(file || '_', '_') - 1)
                    END as gem_id,
                    GROUP_CONCAT(DISTINCT light_source) as lights,
                    COUNT(DISTINCT light_source) as light_count,
                    COUNT(*) as total_features,
                    COUNT(CASE WHEN data_type LIKE '%ultra_optimized%' THEN 1 END) as ultra_features
                FROM structural_features 
                GROUP BY gem_id
                HAVING light_count >= 2
                ORDER BY light_count DESC, total_features DESC
            """)
            
            complete_gems = cursor.fetchall()
            
            print(f"\nGEMS SUITABLE FOR TESTING ({len(complete_gems)} found):")
            print(f"{'Gem ID':<15} {'Lights':<15} {'Count':<8} {'Features':<10} {'ULTRA':<8} {'Status'}")
            print("-" * 75)
            
            for gem_id, lights, light_count, features, ultra_features in complete_gems:
                if light_count == 3:
                    status = "PERFECT"
                elif light_count == 2:
                    status = "GOOD"
                else:
                    status = "MINIMAL"
                
                ultra_indicator = f"ULTRA {ultra_features}" if ultra_features > 0 else f"STD 0"
                
                print(f"{gem_id:<15} {lights:<15} {light_count:<8} {features:<10} {ultra_indicator:<8} {status}")
            
            if complete_gems:
                print(f"\nRECOMMENDATION:")
                best_gem = complete_gems[0]
                ultra_note = f" with ULTRA {best_gem[4]} ULTRA_OPTIMIZED features" if best_gem[4] > 0 else ""
                print(f"   Test with: {best_gem[0]} ({best_gem[1]} - {best_gem[3]} features{ultra_note})")
                print(f"   Expected result: 100% self-match")
            else:
                print(f"\nNO SUITABLE GEMS FOR TESTING")
                print(f"   Need gems with at least 2 light sources")
                print(f"   Add more structural data to database")
            
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")
            import traceback
            traceback.print_exc()
    
    def check_both_database_files(self):
        """Check both database files to see which has data"""
        print("\nCHECKING BOTH DATABASE FILES")
        print("=" * 50)
        
        database_files = [
            "database/structural_spectra/multi_structural_gem_data.db",
            "database/structural_spectra/fixed_structural_gem_data.db"
        ]
        
        for db_path in database_files:
            print(f"\nChecking: {db_path}")
            
            if not os.path.exists(db_path):
                print(f"   File not found")
                continue
                
            try:
                # Get file size
                size_kb = os.path.getsize(db_path) / 1024
                print(f"   Size: {size_kb:.1f} KB")
                
                # Connect and explore
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                print(f"   Tables: {[t[0] for t in tables]}")
                
                # Check each table for records
                for table_name, in tables:
                    if table_name.startswith('sqlite_'):
                        continue  # Skip system tables
                        
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"      {table_name}: {count:,} records")
                        
                        # Check for ULTRA_OPTIMIZED data
                        if table_name == 'structural_features' and count > 0:
                            try:
                                cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE data_type LIKE '%ultra_optimized%'")
                                ultra_count = cursor.fetchone()[0]
                                if ultra_count > 0:
                                    print(f"         ULTRA_OPTIMIZED: {ultra_count:,} records")
                            except:
                                pass
                        
                        if count > 0:
                            # Show sample data
                            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                            samples = cursor.fetchall()
                            print(f"      Sample data:")
                            for i, row in enumerate(samples, 1):
                                sample_str = str(row)[:100] + "..." if len(str(row)) > 100 else str(row)
                                print(f"         {i}. {sample_str}")
                                
                    except Exception as e:
                        print(f"      Error reading {table_name}: {e}")
                
                conn.close()
                
            except Exception as e:
                print(f"   Database error: {e}")

    def export_structural_gems_and_test_matching(self):
        """Export complete structural gems and test matching analysis"""
        print("\nSTRUCTURAL GEMS EXPORT & MATCHING ANALYSIS")
        print("=" * 60)
        
        if not os.path.exists(self.db_path):
            print(f"Structural database not found: {self.db_path}")
            return
        
        # Create data/structural directory
        structural_export_dir = "data/structural"
        os.makedirs(structural_export_dir, exist_ok=True)
        print(f"Export directory: {structural_export_dir}")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Find all complete gems (with B, L, U data)
            query = """
                SELECT 
                    CASE 
                        WHEN file LIKE '%B%' THEN SUBSTR(file, 1, INSTR(file, 'B') - 1)
                        WHEN file LIKE '%L%' THEN SUBSTR(file, 1, INSTR(file, 'L') - 1)
                        WHEN file LIKE '%U%' THEN SUBSTR(file, 1, INSTR(file, 'U') - 1)
                        ELSE SUBSTR(file, 1, INSTR(file || '_', '_') - 1)
                    END as gem_id,
                    GROUP_CONCAT(DISTINCT light_source) as light_sources,
                    COUNT(DISTINCT light_source) as light_count,
                    COUNT(*) as total_features,
                    COUNT(CASE WHEN data_type LIKE '%ultra_optimized%' THEN 1 END) as ultra_features
                FROM structural_features 
                WHERE file NOT LIKE '%unknown%'
                GROUP BY gem_id
                HAVING light_count >= 3
                ORDER BY CAST(SUBSTR(gem_id, 2) AS INTEGER)
            """
            
            complete_gems_df = pd.read_sql_query(query, conn)
            
            if complete_gems_df.empty:
                print("No complete gems found with B, L, U light sources")
                conn.close()
                return
            
            print(f"\nFound {len(complete_gems_df)} complete gems with B+L+U data:")
            print("=" * 80)
            print(f"{'Gem ID':<10} {'Lights':<15} {'Features':<10} {'ULTRA':<8} {'Export Status'}")
            print("-" * 80)
            
            # Export each complete gem
            exported_count = 0
            gem_58_found = False
            
            for _, row in complete_gems_df.iterrows():
                gem_id = row['gem_id']
                feature_count = row['total_features']
                ultra_count = row['ultra_features']
                
                # Check if this is gem 58
                if '58' in str(gem_id):
                    gem_58_found = True
                
                # Export data for this gem
                export_success = self.export_gem_structural_data(conn, gem_id, structural_export_dir)
                
                if export_success:
                    exported_count += 1
                    status = "Exported"
                else:
                    status = "Failed"
                
                ultra_indicator = f"ULTRA {ultra_count}" if ultra_count > 0 else f"STD 0"
                print(f"{gem_id:<10} {'B+L+U':<15} {feature_count:<10} {ultra_indicator:<8} {status}")
            
            print(f"\nEXPORT SUMMARY:")
            print(f"   Successfully exported: {exported_count}/{len(complete_gems_df)} gems")
            print(f"   Location: {structural_export_dir}")
            
            # Special note about gem 58
            if gem_58_found:
                print(f"   Gem 58: FOUND and exported!")
            else:
                print(f"   Gem 58: Not found in complete gems list")
            
            conn.close()
            
            # Now offer to run structural matching analysis
            if exported_count > 0:
                print(f"\nSTRUCTURAL MATCHING ANALYSIS OPTIONS:")
                print("1. Test specific gem (like gem 58)")
                print("2. Run comprehensive matching test on all exported gems")
                print("3. Skip matching analysis")
                
                choice = input("\nSelect option (1-3): ").strip()
                
                if choice == '1':
                    self.test_specific_gem_matching(complete_gems_df)
                elif choice == '2':
                    self.run_comprehensive_matching_test(complete_gems_df)
                else:
                    print("Matching analysis skipped")
            
            # Play completion bleep
            if self.bleep_enabled:
                self.play_bleep(feature_type="completion")
                
        except Exception as e:
            print(f"Error in structural export and matching: {e}")
            import traceback
            traceback.print_exc()

    def export_gem_structural_data(self, conn, gem_id, export_dir):
        """Export structural data for a specific gem to CSV files"""
        try:
            # Create gem-specific directory
            gem_dir = os.path.join(export_dir, gem_id)
            os.makedirs(gem_dir, exist_ok=True)
            
            success_count = 0
            
            # Export data for each light source
            for light_source in ['B', 'L', 'U', 'Halogen', 'Laser', 'UV']:
                query = """
                    SELECT * FROM structural_features 
                    WHERE file LIKE ? AND (light_source = ? OR light_source LIKE ?)
                    ORDER BY wavelength
                """
                
                # Try both exact match and partial match for light source
                light_patterns = [light_source, f"%{light_source.lower()}%"]
                
                df = pd.read_sql_query(query, conn, params=[f"%{gem_id}%", light_source, light_patterns[1]])
                
                if not df.empty:
                    # Export to CSV
                    filename = f"{gem_id}_{light_source}.csv"
                    filepath = os.path.join(gem_dir, filename)
                    df.to_csv(filepath, index=False)
                    success_count += 1
            
            return success_count >= 3  # At least 3 light sources exported
            
        except Exception as e:
            print(f"   Error exporting {gem_id}: {e}")
            return False

    def test_specific_gem_matching(self, complete_gems_df):
        """Test structural matching for a specific gem"""
        print(f"\nSPECIFIC GEM MATCHING TEST")
        print("=" * 50)
        
        # Show available complete gems
        print("Available complete gems:")
        gem_list = []
        for i, (_, row) in enumerate(complete_gems_df.iterrows(), 1):
            gem_id = row['gem_id']
            feature_count = row['total_features']
            ultra_count = row['ultra_features']
            
            special_note = ""
            if '58' in str(gem_id):
                special_note = " (This is gem 58!)"
            
            ultra_indicator = f"ULTRA {ultra_count}" if ultra_count > 0 else ""
            print(f"   {i}. {gem_id} - {feature_count} features {ultra_indicator}{special_note}")
            gem_list.append((gem_id, row))
        
        # Get user selection
        try:
            choice = input(f"\nSelect gem to test (1-{len(gem_list)}) or enter gem ID: ").strip()
            
            # Try as number first
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(gem_list):
                    selected_gem_id, selected_row = gem_list[idx]
                else:
                    print("Invalid selection")
                    return
            except ValueError:
                # Try as direct gem ID
                selected_gem_id = choice
                selected_row = None
                for gem_id, row in gem_list:
                    if gem_id == choice:
                        selected_row = row
                        break
                
                if selected_row is None:
                    print(f"Gem ID '{choice}' not found in complete gems")
                    return
            
            print(f"\nTesting structural matching for: {selected_gem_id}")
            
            # Run the structural matching analysis
            conn = sqlite3.connect(self.db_path)
            self.run_comprehensive_gem_matching_test(conn, selected_gem_id)
            conn.close()
            
        except Exception as e:
            print(f"Error in specific gem testing: {e}")

    def run_comprehensive_matching_test(self, complete_gems_df):
        """Run comprehensive structural matching test on all complete gems"""
        print(f"\nCOMPREHENSIVE MATCHING TEST")
        print("=" * 50)
        
        print(f"Testing {len(complete_gems_df)} complete gems...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            results = []
            
            for i, (_, row) in enumerate(complete_gems_df.iterrows(), 1):
                gem_id = row['gem_id']
                print(f"\nTesting {i}/{len(complete_gems_df)}: {gem_id}")
                
                # Run matching test for this gem
                match_result = self.run_comprehensive_gem_matching_test(conn, gem_id, verbose=False)
                results.append((gem_id, match_result))
                
                # Progress indicator
                if i % 5 == 0:
                    print(f"   Progress: {i}/{len(complete_gems_df)} gems tested")
            
            # Summary results
            print(f"\nCOMPREHENSIVE MATCHING RESULTS:")
            print("=" * 60)
            
            perfect_matches = 0
            good_matches = 0
            
            for gem_id, result in results:
                if result and result.get('self_match_found', False):
                    avg_score = result.get('average_score', 0)
                    if avg_score > 95:
                        perfect_matches += 1
                        status = "PERFECT"
                    elif avg_score > 80:
                        good_matches += 1
                        status = "GOOD"
                    else:
                        status = "POOR"
                    
                    print(f"   {gem_id}: {avg_score:.1f}% - {status}")
                else:
                    print(f"   {gem_id}: FAILED - No self-match")
            
            print(f"\nSUMMARY:")
            print(f"   Perfect matches (>95%): {perfect_matches}")
            print(f"   Good matches (>80%): {good_matches}")
            print(f"   Total tested: {len(results)}")
            
            conn.close()
            
        except Exception as e:
            print(f"Error in comprehensive testing: {e}")

    def run_comprehensive_gem_matching_test(self, conn, gem_id, verbose=True):
        """Run structural matching test for a specific gem and return results"""
        if verbose:
            print(f"Analyzing structural data for {gem_id}...")
        
        try:
            # Get available light sources for this gem
            available_lights_query = """
                SELECT DISTINCT light_source 
                FROM structural_features 
                WHERE file LIKE ?
                ORDER BY light_source
            """
            
            lights_df = pd.read_sql_query(available_lights_query, conn, params=[f"%{gem_id}%"])
            available_lights = lights_df['light_source'].tolist()
            
            if len(available_lights) < 3:
                if verbose:
                    print(f"   Insufficient light sources: {available_lights}")
                return None
            
            if verbose:
                print(f"   Light sources: {', '.join(available_lights)}")
            
            # Test matching for each light source
            light_results = {}
            total_score = 0
            
            for light_source in available_lights:
                if verbose:
                    print(f"   Testing {light_source} light...")
                
                # Get test features
                test_query = """
                    SELECT wavelength, intensity, feature_group, data_type
                    FROM structural_features 
                    WHERE file LIKE ? AND light_source = ?
                    ORDER BY wavelength
                """
                
                test_df = pd.read_sql_query(test_query, conn, params=[f"%{gem_id}%", light_source])
                
                if test_df.empty:
                    if verbose:
                        print(f"      No data for {light_source}")
                    continue
                
                # Find best matches in database
                candidates_query = """
                    SELECT 
                        CASE 
                            WHEN file LIKE '%B%' THEN SUBSTR(file, 1, INSTR(file, 'B') - 1)
                            WHEN file LIKE '%L%' THEN SUBSTR(file, 1, INSTR(file, 'L') - 1)
                            WHEN file LIKE '%U%' THEN SUBSTR(file, 1, INSTR(file, 'U') - 1)
                            ELSE SUBSTR(file, 1, INSTR(file || '_', '_') - 1)
                        END as candidate_gem_id,
                        file, COUNT(*) as feature_count,
                        COUNT(CASE WHEN data_type LIKE '%ultra_optimized%' THEN 1 END) as ultra_count
                    FROM structural_features 
                    WHERE light_source = ?
                    GROUP BY candidate_gem_id, file
                    HAVING feature_count >= 3
                    ORDER BY feature_count DESC
                """
                
                candidates_df = pd.read_sql_query(candidates_query, conn, params=[light_source])
                
                if candidates_df.empty:
                    continue
                
                # Calculate match scores
                best_match_score = 0
                best_match_gem = None
                self_match_found = False
                
                for _, candidate_row in candidates_df.iterrows()[:10]:  # Top 10 candidates
                    candidate_gem_id = candidate_row['candidate_gem_id']
                    candidate_file = candidate_row['file']
                    
                    # Get candidate features
                    candidate_query = """
                        SELECT wavelength, intensity, feature_group, data_type
                        FROM structural_features 
                        WHERE file = ?
                        ORDER BY wavelength
                    """
                    
                    candidate_df = pd.read_sql_query(candidate_query, conn, params=[candidate_file])
                    
                    if candidate_df.empty:
                        continue
                    
                    # Convert to feature format
                    test_features = []
                    for _, row in test_df.iterrows():
                        test_features.append({
                            'feature_type': row['feature_group'],
                            'wavelength': row['wavelength'],
                            'intensity': row['intensity']
                        })
                    
                    candidate_features = []
                    for _, row in candidate_df.iterrows():
                        candidate_features.append({
                            'feature_type': row['feature_group'],
                            'wavelength': row['wavelength'],
                            'intensity': row['intensity']
                        })
                    
                    # Calculate match score
                    match_score = self.calculate_simple_structural_match(test_features, candidate_features)
                    
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_match_gem = candidate_gem_id
                    
                    if candidate_gem_id == gem_id:
                        self_match_found = True
                        if verbose:
                            print(f"      SELF-MATCH found: {match_score:.1f}%")
                
                light_results[light_source] = {
                    'best_score': best_match_score,
                    'best_match': best_match_gem,
                    'self_match_found': self_match_found
                }
                
                total_score += best_match_score
            
            # Calculate average score
            average_score = total_score / len(light_results) if light_results else 0
            overall_self_match = any(result['self_match_found'] for result in light_results.values())
            
            result = {
                'gem_id': gem_id,
                'average_score': average_score,
                'self_match_found': overall_self_match,
                'light_results': light_results
            }
            
            if verbose:
                print(f"   Average matching score: {average_score:.1f}%")
                print(f"   Self-match result: {'SUCCESS' if overall_self_match else 'FAILED'}")
            
            return result
            
        except Exception as e:
            if verbose:
                print(f"   Error in matching test: {e}")
            return None
    
    def run_main_menu(self):
        """Main menu interface for the Enhanced Gemini Analysis System"""
        
        print("\n" + "=" * 80)
        print("  ENHANCED GEMINI GEMOLOGICAL ANALYSIS SYSTEM v2.0")
        print("  Complete Database Integration + ULTRA_OPTIMIZED Support")
        print("=" * 80)
        
        while True:
            print(f"\nMAIN MENU:")
            print("=" * 50)
            
            # Core Analysis Options
            print("ANALYSIS OPTIONS:")
            print("1. Run Structural Analysis (gemini_launcher.py)")
            print("2. Run Numerical Analysis (gemini1.py)")
            print("3. Fast Gem Analysis Tool")
            print("4. Enhanced Comparison Analysis")
            print("5. Relative Height Analysis")
            
            # Database Management
            print("\nDATABASE MANAGEMENT:")
            print("6. Enhanced Database Search")
            print("7. Show Enhanced Database Statistics")
            print("8. Batch Import Structural Data")
            print("9. Test Enhanced Import System")
            print("10. Update Database Schema (ULTRA_OPTIMIZED)")
            
            # Advanced Features
            print("\nADVANCED FEATURES:")
            print("11. Structural Matching Analysis")
            print("12. Export & Test Structural Gems")
            print("13. Debug Structural Database")
            print("14. Check Both Database Files")
            
            # System Configuration
            print("\nSYSTEM CONFIGURATION:")
            print("15. Toggle Audio Bleep System")
            print("16. Toggle Auto-Import System")
            print("17. Enhanced CSV Import Test")
            print("18. System Status Check")
            
            print("\n0. Exit System")
            
            # Show current system status
            bleep_status = "ON" if self.bleep_enabled else "OFF"
            import_status = "ON" if self.auto_import_enabled else "OFF"
            print(f"\nCurrent Status: Bleep [{bleep_status}] | Auto-Import [{import_status}]")
            
            try:
                choice = input("\nSelect option (0-18): ").strip()
                
                if choice == '0':
                    print("\nExiting Enhanced Gemini Analysis System...")
                    print("Database connections closed.")
                    if self.bleep_enabled:
                        self.play_bleep(feature_type="completion")
                    break
                    
                elif choice == '1':
                    self.run_structural_analysis()
                    
                elif choice == '2':
                    self.run_numerical_analysis()
                    
                elif choice == '3':
                    self.run_fast_analysis()
                    
                elif choice == '4':
                    self.enhanced_comparison_analysis()
                    
                elif choice == '5':
                    self.relative_height_analysis()
                    
                elif choice == '6':
                    self.enhanced_database_search()
                    
                elif choice == '7':
                    self.show_enhanced_database_stats()
                    
                elif choice == '8':
                    self.batch_import_structural_data()
                    
                elif choice == '9':
                    self.test_enhanced_import_system()
                    
                elif choice == '10':
                    self.update_database_schema_for_ultra_optimized()
                    
                elif choice == '11':
                    self.structural_matching_analysis()
                    
                elif choice == '12':
                    self.export_structural_gems_and_test_matching()
                    
                elif choice == '13':
                    self.debug_structural_database()
                    
                elif choice == '14':
                    self.check_both_database_files()
                    
                elif choice == '15':
                    self.toggle_bleep_system()
                    
                elif choice == '16':
                    self.toggle_auto_import()
                    
                elif choice == '17':
                    self.test_enhanced_import_system()
                    
                elif choice == '18':
                    self.system_status_check()
                    
                else:
                    print("Invalid choice. Please select 0-18.")
                    
                # Brief pause between operations
                if choice != '0':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\nSystem interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nError in menu system: {e}")
                input("Press Enter to continue...")
    
    def run_structural_analysis(self):
        """Launch structural analysis system"""
        print("\nLAUNCHING STRUCTURAL ANALYSIS")
        print("=" * 50)
        
        structural_main = self.program_files.get('src/structural_analysis/main.py')
        launcher_path = 'src/structural_analysis/gemini_launcher.py'
        
        if os.path.exists(launcher_path):
            print(f"Starting {structural_main}...")
            try:
                # Try to run the launcher
                result = subprocess.run([sys.executable, launcher_path], 
                                      capture_output=False, 
                                      text=True)
                
                if result.returncode == 0:
                    print("Structural analysis completed successfully")
                    
                    # Check for new CSV files to auto-import
                    if self.auto_import_enabled:
                        self.scan_and_auto_import_new_csvs()
                else:
                    print(f"Structural analysis exited with code: {result.returncode}")
                    
            except Exception as e:
                print(f"Error launching structural analysis: {e}")
        else:
            print(f"Structural analysis file not found: {launcher_path}")
            print("Please ensure the structural analysis system is properly installed")
    
    def run_numerical_analysis(self):
        """Launch numerical analysis system"""
        print("\nLAUNCHING NUMERICAL ANALYSIS")
        print("=" * 50)
        
        numerical_path = 'src/numerical_analysis/gemini1.py'
        
        if os.path.exists(numerical_path):
            print(f"Starting {self.program_files[numerical_path]}...")
            try:
                result = subprocess.run([sys.executable, numerical_path], 
                                      capture_output=False, 
                                      text=True)
                
                if result.returncode == 0:
                    print("Numerical analysis completed successfully")
                else:
                    print(f"Numerical analysis exited with code: {result.returncode}")
                    
            except Exception as e:
                print(f"Error launching numerical analysis: {e}")
        else:
            print(f"Numerical analysis file not found: {numerical_path}")
            print("Please ensure the numerical analysis system is properly installed")
    
    def run_fast_analysis(self):
        """Launch fast analysis tool"""
        print("\nLAUNCHING FAST GEM ANALYSIS")
        print("=" * 50)
        
        fast_analysis_path = 'fast_gem_analysis.py'
        
        if os.path.exists(fast_analysis_path):
            print(f"Starting {self.program_files[fast_analysis_path]}...")
            try:
                result = subprocess.run([sys.executable, fast_analysis_path], 
                                      capture_output=False, 
                                      text=True)
                
                if result.returncode == 0:
                    print("Fast analysis completed successfully")
                    
                    # Check for auto-import opportunities
                    if self.auto_import_enabled:
                        self.scan_and_auto_import_new_csvs()
                else:
                    print(f"Fast analysis exited with code: {result.returncode}")
                    
            except Exception as e:
                print(f"Error launching fast analysis: {e}")
        else:
            print(f"Fast analysis file not found: {fast_analysis_path}")
            print("Please ensure the fast analysis tool is available")
    
    def scan_and_auto_import_new_csvs(self):
        """Scan for new CSV files and auto-import them"""
        print("\nSCANNING FOR NEW CSV FILES...")
        
        # Common directories to scan for new CSV files
        scan_dirs = [
            "data/structural_data",
            self.structural_data_dir,
            "exports",
            "output"
        ]
        
        new_files_found = []
        
        for scan_dir in scan_dirs:
            if os.path.exists(scan_dir):
                for root, dirs, files in os.walk(scan_dir):
                    for file in files:
                        if file.endswith('.csv') and not file.startswith('.'):
                            full_path = os.path.join(root, file)
                            # Check if file is recent (modified in last hour)
                            file_mod_time = os.path.getmtime(full_path)
                            current_time = time.time()
                            
                            if current_time - file_mod_time < 3600:  # 1 hour
                                new_files_found.append(full_path)
        
        if new_files_found:
            print(f"Found {len(new_files_found)} recent CSV files:")
            
            imported_count = 0
            for csv_file in new_files_found:
                print(f"   Auto-importing: {os.path.basename(csv_file)}")
                success = self.auto_import_csv_to_database_enhanced(csv_file)
                if success:
                    imported_count += 1
            
            print(f"Auto-imported {imported_count}/{len(new_files_found)} files")
            
            if imported_count > 0 and self.bleep_enabled:
                self.play_bleep(feature_type="completion")
        else:
            print("No recent CSV files found for auto-import")
    
    def system_status_check(self):
        """Comprehensive system status check"""
        print("\nSYSTEM STATUS CHECK")
        print("=" * 50)
        
        # Check database status
        print("DATABASE STATUS:")
        if os.path.exists(self.db_path):
            db_size = os.path.getsize(self.db_path) / 1024  # KB
            print(f"   ✓ Main database: {self.db_path} ({db_size:.1f} KB)")
            
            # Quick record count
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM structural_features")
                record_count = cursor.fetchone()[0]
                print(f"   ✓ Records: {record_count:,}")
                
                # ULTRA_OPTIMIZED count
                cursor.execute("SELECT COUNT(*) FROM structural_features WHERE data_type LIKE '%ultra_optimized%'")
                ultra_count = cursor.fetchone()[0]
                print(f"   ✓ ULTRA_OPTIMIZED: {ultra_count:,}")
                
                conn.close()
            except Exception as e:
                print(f"   ✗ Database error: {e}")
        else:
            print(f"   ✗ Main database not found: {self.db_path}")
        
        # Check system files
        print(f"\nSYSTEM FILES:")
        for file_path, description in self.program_files.items():
            if os.path.exists(file_path):
                print(f"   ✓ {description}: {file_path}")
            else:
                print(f"   ✗ {description}: {file_path} (missing)")
        
        # Check spectral reference files
        print(f"\nREFERENCE SPECTRA:")
        for spec_file in self.spectral_files:
            if os.path.exists(spec_file):
                file_size = os.path.getsize(spec_file) / 1024  # KB
                print(f"   ✓ {os.path.basename(spec_file)}: {file_size:.1f} KB")
            else:
                print(f"   ✗ {os.path.basename(spec_file)}: missing")
        
        # Check audio system
        print(f"\nAUDIO SYSTEM:")
        if HAS_AUDIO:
            audio_lib = "winsound" if 'winsound' in sys.modules else "pygame"
            print(f"   ✓ Audio available: {audio_lib}")
            print(f"   ✓ Bleep system: {'enabled' if self.bleep_enabled else 'disabled'}")
        else:
            print(f"   ✗ Audio not available (no winsound/pygame)")
        
        # Check directories
        print(f"\nDIRECTORIES:")
        dirs_to_check = [
            ("Structural data", self.structural_data_dir),
            ("Database folder", "database"),
            ("Data folder", "data"),
            ("Source folder", "src")
        ]
        
        for dir_name, dir_path in dirs_to_check:
            if os.path.exists(dir_path):
                files_count = sum([len(files) for r, d, files in os.walk(dir_path)])
                print(f"   ✓ {dir_name}: {dir_path} ({files_count} files)")
            else:
                print(f"   ✗ {dir_name}: {dir_path} (missing)")
        
        # System configuration
        print(f"\nCONFIGURATION:")
        print(f"   Auto-import: {'enabled' if self.auto_import_enabled else 'disabled'}")
        print(f"   Bleep feedback: {'enabled' if self.bleep_enabled else 'disabled'}")
        print(f"   Database path: {self.db_path}")
        print(f"   Structural data: {self.structural_data_dir}")
        
        # Performance check
        print(f"\nPERFORMANCE:")
        start_time = time.time()
        
        # Quick database query
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE wavelength > 500")
            test_result = cursor.fetchone()[0]
            conn.close()
            
            query_time = (time.time() - start_time) * 1000  # milliseconds
            print(f"   ✓ Database query: {query_time:.1f}ms ({test_result:,} records)")
            
        except Exception as e:
            print(f"   ✗ Database performance test failed: {e}")
        
        print(f"\nSYSTEM READY FOR GEMOLOGICAL ANALYSIS")
        
        if self.bleep_enabled:
            self.play_bleep(feature_type="completion")


def main():
    """Main entry point for the Enhanced Gemini Analysis System"""
    try:
        # Initialize the enhanced system
        system = EnhancedGeminiAnalysisSystem()
        
        # Check if running in interactive mode or with command line args
        if len(sys.argv) > 1:
            # Command line mode
            command = sys.argv[1].lower()
            
            if command == '--status':
                system.system_status_check()
            elif command == '--batch-import':
                system.batch_import_structural_data()
            elif command == '--db-stats':
                system.show_enhanced_database_stats()
            elif command == '--test-import':
                system.test_enhanced_import_system()
            elif command == '--debug-db':
                system.debug_structural_database()
            elif command == '--help':
                print("Enhanced Gemini Analysis System - Command Line Options:")
                print("  --status       : System status check")
                print("  --batch-import : Batch import structural data")
                print("  --db-stats     : Show database statistics")
                print("  --test-import  : Test enhanced import system")
                print("  --debug-db     : Debug structural database")
                print("  --help         : Show this help message")
                print("  (no args)      : Interactive menu mode")
            else:
                print(f"Unknown command: {command}")
                print("Use --help for available commands")
        else:
            # Interactive menu mode
            system.run_main_menu()
            
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user")
    except Exception as e:
        print(f"\nCritical system error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
