#!/usr/bin/env python3
"""
ENHANCED GEMINI GEMOLOGICAL ANALYSIS SYSTEM - COMPLETE DATABASE INTEGRATION
Enhanced version with full database management, auto-import, enhanced comparison tools,
bleep features, relative height measurements, and streamlined workflow automation.

Major Enhancements:
- Full database import/export system integrated
- Automatic CSV-to-SQLite import after manual analysis
- Enhanced comparison tools using database queries
- Audio bleep feedback for significant features
- Relative height measurements across light sources
- Comprehensive database statistics and management
- Automated workflow from analysis → database → comparison
- NEW: Option 11 - Structural matching analysis with self-validation
- FIXED: Correct database paths and queries for your system
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
        # CORRECTED DATABASE PATH - Full absolute path to your 448KB database
        self.db_path = r"C:\Users\David\OneDrive\Desktop\gemini_gemological_analysis\database\structural_spectra\multi_structural_gem_data.db"
        
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
        
        # Ensure database directory exists - use absolute path
        db_dir = r"C:\Users\David\OneDrive\Desktop\gemini_gemological_analysis\database\structural_spectra"
        os.makedirs(db_dir, exist_ok=True)
        
        # Check if database exists, if not create basic structure
        if not os.path.exists(self.db_path):
            self.create_database_schema()
        else:
            print(f"Database found: {os.path.getsize(self.db_path) / 1024:.1f} KB")
        
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
            print("Database schema created successfully")
            
        except Exception as e:
            print(f"Error creating database schema: {e}")
    
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
        """Automatically import CSV file to database after manual analysis"""
        if not self.auto_import_enabled:
            return False
        
        try:
            # Determine light source from file path or filename
            light_source = self.detect_light_source_from_path(csv_file_path)
            
            # Read CSV file
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            imported_count = 0
            base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
            
            # Import each row from the CSV
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO structural_features 
                        (feature, file, light_source, wavelength, intensity, point_type, 
                         feature_group, processing, baseline_used, norm_factor, snr,
                         symmetry_ratio, skew_description, width_nm, height, 
                         local_slope, slope_r_squared, file_source, data_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('Feature', ''),
                        base_filename,
                        light_source,
                        float(row.get('Wavelength', 0)),
                        float(row.get('Intensity', 0)),
                        row.get('Point_Type', 'Manual'),
                        row.get('Feature_Group', 'Unknown'),
                        'Manual_Analysis',
                        self.safe_float(row.get('Baseline_Used')),
                        self.safe_float(row.get('Norm_Factor')),
                        self.safe_float(row.get('SNR')),
                        self.safe_float(row.get('Symmetry_Ratio')),
                        row.get('Skew_Description', ''),
                        self.safe_float(row.get('Width_nm')),
                        self.safe_float(row.get('Height')),
                        self.safe_float(row.get('Local_Slope')),
                        self.safe_float(row.get('Slope_R_Squared')),
                        csv_file_path,
                        'manual_structural'
                    ))
                    
                    if cursor.rowcount > 0:
                        imported_count += 1
                        
                except Exception as e:
                    print(f"Error importing row: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            if imported_count > 0:
                print(f"Auto-imported {imported_count} features from {os.path.basename(csv_file_path)}")
                self.play_bleep(feature_type="completion")
                return True
            
            return False
            
        except Exception as e:
            print(f"Auto-import error: {e}")
            return False
    
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
        print("\n🚀 ENHANCED BATCH IMPORT SYSTEM")
        print("=" * 50)
        
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
        
        print(f"📁 Found structural data files:")
        for light_source, count in light_source_counts.items():
            print(f"   {light_source.capitalize()}: {count} files")
        
        if not all_csv_files:
            print("❌ No structural CSV files found!")
            return
        
        print(f"\n📊 Total files to import: {len(all_csv_files)}")
        
        # Confirm import
        confirm = input("\n🔄 Proceed with batch import? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Import cancelled")
            return
        
        # Import files
        successful_imports = 0
        failed_imports = 0
        
        print(f"\n🔄 Starting batch import...")
        
        for file_path, light_source in all_csv_files:
            print(f"   📥 Processing: {os.path.basename(file_path)}")
            
            success = self.auto_import_csv_to_database(file_path)
            if success:
                successful_imports += 1
            else:
                failed_imports += 1
        
        print(f"\n📊 BATCH IMPORT COMPLETED:")
        print(f"   ✅ Successful: {successful_imports}")
        print(f"   ❌ Failed: {failed_imports}")
        
        if successful_imports > 0:
            self.play_bleep(feature_type="completion")
            
        # Show database statistics
        self.show_enhanced_database_stats()
    
    def show_enhanced_database_stats(self):
        """Show enhanced database statistics with detailed breakdowns"""
        print("\n📊 ENHANCED DATABASE STATISTICS")
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
                print("📭 Database is empty - no structural features found")
                conn.close()
                return
            
            total_records = basic_stats.iloc[0]['total_records']
            unique_files = basic_stats.iloc[0]['unique_files']
            light_sources = basic_stats.iloc[0]['light_sources']
            
            print(f"📁 Total records: {total_records:,}")
            print(f"📄 Unique files: {unique_files:,}")
            print(f"💡 Light sources: {light_sources}")
            
            # By light source
            light_stats = pd.read_sql_query("""
                SELECT light_source, COUNT(*) as count, COUNT(DISTINCT file) as files
                FROM structural_features 
                GROUP BY light_source 
                ORDER BY count DESC
            """, conn)
            
            print(f"\n💡 BY LIGHT SOURCE:")
            for _, row in light_stats.iterrows():
                print(f"   {row['light_source']}: {row['count']:,} records ({row['files']} files)")
            
            # By feature type
            feature_stats = pd.read_sql_query("""
                SELECT feature_group, COUNT(*) as count
                FROM structural_features 
                GROUP BY feature_group 
                ORDER BY count DESC
                LIMIT 10
            """, conn)
            
            print(f"\n🏷️  TOP FEATURE TYPES:")
            for _, row in feature_stats.iterrows():
                print(f"   {row['feature_group']}: {row['count']:,}")
            
            # Recent activity
            recent_stats = pd.read_sql_query("""
                SELECT COUNT(*) as recent_count
                FROM structural_features 
                WHERE timestamp > datetime('now', '-7 days')
            """, conn)
            
            recent_count = recent_stats.iloc[0]['recent_count']
            print(f"\n📅 Recent activity (7 days): {recent_count:,} records")
            
            # Normalization status
            norm_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(CASE WHEN normalization_scheme IS NOT NULL THEN 1 END) as with_normalization,
                    COUNT(CASE WHEN normalization_compatible = 1 THEN 1 END) as compatible
                FROM structural_features
            """, conn)
            
            with_norm = norm_stats.iloc[0]['with_normalization']
            compatible = norm_stats.iloc[0]['compatible']
            
            print(f"\n🔧 NORMALIZATION STATUS:")
            print(f"   With metadata: {with_norm:,} / {total_records:,}")
            print(f"   Compatible: {compatible:,} / {total_records:,}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    
    def enhanced_database_search(self):
        """Enhanced database search with multiple criteria"""
        print("\n🔍 ENHANCED DATABASE SEARCH")
        print("=" * 40)
        
        if not os.path.exists(self.db_path):
            print("❌ Database not found")
            return
        
        # Search options
        print("Search options:")
        print("1. By gem/file name")
        print("2. By wavelength range")
        print("3. By feature type")
        print("4. By light source")
        print("5. Recent features (last 7 days)")
        
        choice = input("\nSelect search type (1-5): ").strip()
        
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
                
            else:
                print("❌ Invalid choice")
                conn.close()
                return
            
            conn.close()
            
            if df.empty:
                print("❌ No results found")
                return
            
            print(f"\n📊 Found {len(df)} results:")
            print("=" * 80)
            
            # Display results with enhanced formatting
            for i, (_, row) in enumerate(df.iterrows(), 1):
                print(f"{i:3}. {row['file']} | {row['light_source']} | {row['wavelength']:.1f}nm | {row['feature_group']} | {row['intensity']:.2f}")
                
                if i >= 20:  # Limit display
                    remaining = len(df) - 20
                    if remaining > 0:
                        print(f"    ... and {remaining} more results")
                    break
            
            # Offer to save results
            save = input(f"\n💾 Save results to CSV? (y/n): ").strip().lower()
            if save == 'y':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"database_search_results_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"✅ Results saved to {filename}")
                
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    def enhanced_comparison_analysis(self):
        """Enhanced comparison analysis using database"""
        print("\n📊 ENHANCED COMPARISON ANALYSIS")
        print("=" * 45)
        
        if not os.path.exists(self.db_path):
            print("❌ Database not found")
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
                print("❌ No gems with sufficient features found")
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
            print("📁 Available gems for comparison:")
            gem_list = []
            for i, (gem_name, light_sources) in enumerate(gem_groups.items(), 1):
                print(f"   {i}. {gem_name} ({'+'.join(light_sources)})")
                gem_list.append(gem_name)
            
            # Select gems for comparison
            if len(gem_list) < 2:
                print("❌ Need at least 2 gems for comparison")
                conn.close()
                return
            
            choice1 = int(input(f"\nSelect first gem (1-{len(gem_list)}): ")) - 1
            choice2 = int(input(f"Select second gem (1-{len(gem_list)}): ")) - 1
            
            if choice1 < 0 or choice1 >= len(gem_list) or choice2 < 0 or choice2 >= len(gem_list):
                print("❌ Invalid selection")
                conn.close()
                return
            
            gem1 = gem_list[choice1]
            gem2 = gem_list[choice2]
            
            # Perform comparison
            self.perform_gem_comparison(conn, gem1, gem2)
            conn.close()
            
        except Exception as e:
            print(f"❌ Comparison error: {e}")
    
    def perform_gem_comparison(self, conn, gem1, gem2):
        """Perform detailed comparison between two gems"""
        print(f"\n🔬 COMPARING: {gem1} vs {gem2}")
        print("=" * 60)
        
        # Get features for both gems
        for gem_name in [gem1, gem2]:
            query = """
                SELECT light_source, wavelength, intensity, feature_group
                FROM structural_features 
                WHERE file LIKE ?
                ORDER BY light_source, wavelength
            """
            
            gem_data = pd.read_sql_query(query, conn, params=[f"%{gem_name}%"])
            
            if gem_data.empty:
                print(f"❌ No data found for {gem_name}")
                continue
            
            print(f"\n📊 {gem_name.upper()}:")
            
            # Group by light source
            for light_source in ['B', 'L', 'U', 'Halogen', 'Laser', 'UV']:
                light_data = gem_data[gem_data['light_source'] == light_source]
                if not light_data.empty:
                    print(f"   💡 {light_source} Light ({len(light_data)} features):")
                    
                    # Show top features by intensity
                    top_features = light_data.nlargest(3, 'intensity')
                    for _, feature in top_features.iterrows():
                        print(f"      {feature['wavelength']:.1f}nm: {feature['feature_group']} (I:{feature['intensity']:.2f})")
                    
                    # Calculate relative heights
                    if len(light_data) >= 2:
                        wavelength_range = light_data['wavelength'].max() - light_data['wavelength'].min()
                        intensity_range = light_data['intensity'].max() - light_data['intensity'].min()
                        print(f"      Range: {wavelength_range:.1f}nm, Intensity span: {intensity_range:.2f}")
        
        # Find common wavelengths for direct comparison
        print(f"\n🎯 DIRECT COMPARISON:")
        
        gem1_data = pd.read_sql_query("""
            SELECT wavelength, intensity, light_source, feature_group
            FROM structural_features WHERE file LIKE ?
        """, conn, params=[f"%{gem1}%"])
        
        gem2_data = pd.read_sql_query("""
            SELECT wavelength, intensity, light_source, feature_group
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
                    'feature2': closest['feature_group']
                })
        
        if common_features:
            print(f"   Found {len(common_features)} comparable features:")
            
            for cf in common_features[:10]:  # Show top 10
                wl_diff = abs(cf['wavelength1'] - cf['wavelength2'])
                ratio_str = f"{cf['ratio']:.2f}" if cf['ratio'] != float('inf') else "∞"
                
                print(f"   {cf['light_source']} | {cf['wavelength1']:.1f}nm vs {cf['wavelength2']:.1f}nm")
                print(f"      {gem1}: {cf['feature1']} (I:{cf['intensity1']:.2f})")
                print(f"      {gem2}: {cf['feature2']} (I:{cf['intensity2']:.2f})")
                print(f"      Ratio: {ratio_str}, Δλ: {wl_diff:.1f}nm")
                print()
        else:
            print("   ❌ No comparable features found within tolerance")
        
        # Play completion bleep
        self.play_bleep(feature_type="completion")
    
    def toggle_bleep_system(self):
        """Toggle audio bleep feedback system"""
        self.bleep_enabled = not self.bleep_enabled
        status = "ENABLED" if self.bleep_enabled else "DISABLED"
        
        if HAS_AUDIO:
            print(f"🔊 Audio bleep system: {status}")
            if self.bleep_enabled:
                self.play_bleep(feature_type="completion")
                print("   Features will play audio feedback")
            else:
                print("   No audio feedback will be played")
        else:
            print(f"🔇 Audio system not available (no winsound/pygame)")
            self.bleep_enabled = False
    
    def toggle_auto_import(self):
        """Toggle automatic CSV import system"""
        self.auto_import_enabled = not self.auto_import_enabled
        status = "ENABLED" if self.auto_import_enabled else "DISABLED"
        print(f"🔄 Auto-import system: {status}")
        
        if self.auto_import_enabled:
            print("   CSV files will be automatically imported to database")
        else:
            print("   Manual import required for CSV files")
    
    def relative_height_analysis(self):
        """Perform relative height analysis across light sources"""
        print("\n📏 RELATIVE HEIGHT ANALYSIS")
        print("=" * 40)
        
        gem_id = input("Enter gem ID for analysis: ").strip()
        wavelength = float(input("Enter target wavelength (nm): "))
        tolerance = float(input("Enter tolerance (nm) [default: 5.0]: ") or "5.0")
        
        measurements = self.calculate_relative_height(gem_id, wavelength, tolerance)
        
        if not measurements:
            print(f"❌ No measurements found for {gem_id} near {wavelength}nm")
            return
        
        print(f"\n📊 RELATIVE HEIGHT ANALYSIS: {gem_id}")
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
            
            print(f"💡 {light_source} Light:")
            print(f"   Wavelength: {actual_wl:.1f}nm (Δ{wl_diff:.1f}nm)")
            print(f"   Intensity: {intensity:.2f}")
            print(f"   Relative height: {relative:.3f} ({percentage:.1f}%)")
            print(f"   Feature type: {feature}")
            print()
        
        # Find dominant light source
        if len(measurements) >= 2:
            intensities = [(ls, data['intensity']) for ls, data in measurements.items()]
            dominant = max(intensities, key=lambda x: x[1])
            
            print(f"🏆 DOMINANT: {dominant[0]} light (intensity: {dominant[1]:.2f})")
            
            # Calculate ratios
            print(f"\n📊 INTENSITY RATIOS:")
            for light_source, data in measurements.items():
                if light_source != dominant[0]:
                    ratio = data['intensity'] / dominant[1]
                    print(f"   {light_source}/{dominant[0]}: {ratio:.3f}")
        
        # Cache results
        cache_key = f"{gem_id}_{wavelength:.1f}"
        self.relative_height_cache[cache_key] = measurements
        
        self.play_bleep(feature_type="completion")
    
    def structural_matching_analysis(self):
        """FIXED: Option 11 - Structural matching analysis with correct queries for your database"""
        print("\n🔍 STRUCTURAL MATCHING ANALYSIS")
        print("=" * 50)
        
        if not os.path.exists(self.db_path):
            print(f"❌ Structural database not found: {self.db_path}")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # CORRECTED QUERY: Works with your actual data format
            # Your files are like: 51BC1, 51UP1, 58BP1, etc.
            # Extract base gem number from these patterns
            query = """
                SELECT 
                    CASE 
                        WHEN file LIKE '%BC%' THEN SUBSTR(file, 1, INSTR(file, 'BC') - 1)
                        WHEN file LIKE '%BP%' THEN SUBSTR(file, 1, INSTR(file, 'BP') - 1)
                        WHEN file LIKE '%LC%' THEN SUBSTR(file, 1, INSTR(file, 'LC') - 1)
                        WHEN file LIKE '%LP%' THEN SUBSTR(file, 1, INSTR(file, 'LP') - 1)
                        WHEN file LIKE '%UC%' THEN SUBSTR(file, 1, INSTR(file, 'UC') - 1)
                        WHEN file LIKE '%UP%' THEN SUBSTR(file, 1, INSTR(file, 'UP') - 1)
                        ELSE SUBSTR(file, 1, LENGTH(file) - 3)
                    END as gem_id,
                    GROUP_CONCAT(DISTINCT light_source) as light_sources,
                    COUNT(DISTINCT light_source) as light_count,
                    COUNT(*) as total_features
                FROM structural_features 
                WHERE file NOT LIKE '%unknown%'
                GROUP BY gem_id
                HAVING light_count >= 2 AND gem_id != ''
                ORDER BY light_count DESC, total_features DESC, gem_id
            """
            
            gems_df = pd.read_sql_query(query, conn)
            
            if gems_df.empty:
                print("❌ No gems with multi-light structural data found")
                print("💡 Need gems with at least 2 light sources (Halogen, Laser, UV) for testing")
                
                # Show what we DO have
                print(f"\n🔍 Available data summary:")
                summary_query = """
                    SELECT file, light_source, COUNT(*) as features
                    FROM structural_features 
                    GROUP BY file, light_source 
                    ORDER BY file, light_source
                """
                summary_df = pd.read_sql_query(summary_query, conn)
                
                if not summary_df.empty:
                    print(f"📊 Found data for {len(summary_df)} file/light combinations")
                    current_file = None
                    for _, row in summary_df.iterrows():
                        if row['file'] != current_file:
                            if current_file:
                                print()
                            current_file = row['file']
                            print(f"   📁 {row['file']}:")
                        print(f"      {row['light_source']}: {row['features']} features")
                
                conn.close()
                return
            
            print(f"📊 Found {len(gems_df)} gems suitable for structural matching:")
            print("=" * 70)
            print(f"{'#':<3} {'Gem ID':<15} {'Light Sources':<25} {'Features':<10} {'Status'}")
            print("-" * 70)
            
            for i, row in gems_df.iterrows():
                gem_id = row['gem_id']
                light_sources = row['light_sources']
                feature_count = row['total_features']
                light_count = row['light_count']
                
                # Status indicator
                if light_count == 3:
                    status = "✅ COMPLETE (H+L+U)"
                elif light_count == 2:
                    status = "🟡 PARTIAL (2/3)"
                else:
                    status = "🔴 INSUFFICIENT"
                
                print(f"{i+1:<3} {gem_id:<15} {light_sources:<25} {feature_count:<10} {status}")
            
            # Selection menu
            print(f"\n📋 SELECT GEM FOR STRUCTURAL MATCHING TEST:")
            choice = input(f"Enter gem number (1-{len(gems_df)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                conn.close()
                return
            
            try:
                gem_idx = int(choice) - 1
                if gem_idx < 0 or gem_idx >= len(gems_df):
                    print("❌ Invalid selection")
                    conn.close()
                    return
                
                selected_gem = gems_df.iloc[gem_idx]
                gem_id = selected_gem['gem_id']
                available_lights = selected_gem['light_sources'].split(',')
                
                print(f"\n🎯 SELECTED: Gem {gem_id}")
                print(f"   Available light sources: {', '.join(available_lights)}")
                
                # Show detailed breakdown
                self.show_gem_structural_details(conn, gem_id, available_lights)
                
                # Confirm testing
                confirm = input(f"\n🧪 Test structural matching for Gem {gem_id}? (y/n): ").strip().lower()
                if confirm != 'y':
                    conn.close()
                    return
                
                # Extract gem data and run matching test
                self.run_structural_matching_test(conn, gem_id, available_lights)
                
            except ValueError:
                print("❌ Please enter a valid number")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            import traceback
            traceback.print_exc()
    
    def show_gem_structural_details(self, conn, gem_id, available_lights):
        """Show detailed structural data for selected gem"""
        print(f"\n📋 STRUCTURAL DATA DETAILS: Gem {gem_id}")
        print("=" * 60)
        
        for light_source in available_lights:
            # Get features for this light source - match files containing gem_id
            query = """
                SELECT file, feature_group, wavelength, intensity, COUNT(*) as count
                FROM structural_features 
                WHERE file LIKE ? AND light_source = ?
                GROUP BY file, feature_group
                ORDER BY file, feature_group, wavelength
            """
            
            # Create patterns to match your file naming convention
            patterns = [
                f"{gem_id}BC%", f"{gem_id}BP%",  # Halogen patterns
                f"{gem_id}LC%", f"{gem_id}LP%",  # Laser patterns  
                f"{gem_id}UC%", f"{gem_id}UP%"   # UV patterns
            ]
            
            all_features = []
            for pattern in patterns:
                features_df = pd.read_sql_query(query, conn, params=[pattern, light_source])
                if not features_df.empty:
                    all_features.append(features_df)
            
            if all_features:
                combined_features = pd.concat(all_features, ignore_index=True)
                
                total_features = combined_features['count'].sum()
                feature_types = combined_features['feature_group'].unique()
                unique_files = combined_features['file'].unique()
                
                print(f"💡 {light_source} Light Source:")
                print(f"   📄 Files: {', '.join(unique_files)}")
                print(f"   🔢 Total features: {total_features}")
                print(f"   🏷️  Feature types: {', '.join(feature_types)}")
                
                # Show feature breakdown
                for feature_type in feature_types:
                    type_data = combined_features[combined_features['feature_group'] == feature_type]
                    count = type_data['count'].sum()
                    
                    # Get wavelength range for this feature type
                    wavelength_query = """
                        SELECT MIN(wavelength) as min_wl, MAX(wavelength) as max_wl
                        FROM structural_features 
                        WHERE file LIKE ? AND light_source = ? AND feature_group = ?
                    """
                    
                    for pattern in patterns:
                        wl_result = pd.read_sql_query(wavelength_query, conn, 
                                                    params=[pattern, light_source, feature_type])
                        if not wl_result.empty and pd.notna(wl_result.iloc[0]['min_wl']):
                            min_wl = wl_result.iloc[0]['min_wl']
                            max_wl = wl_result.iloc[0]['max_wl']
                            wl_range = f"{min_wl:.1f}-{max_wl:.1f}nm"
                            print(f"      • {feature_type}: {count} features ({wl_range})")
                            break
                print()
    
    def run_structural_matching_test(self, conn, gem_id, available_lights):
        """Run structural matching test with self-validation - FIXED for your data"""
        print(f"\n🧪 STRUCTURAL MATCHING TEST: Gem {gem_id}")
        print("=" * 50)
        
        # For each light source, extract data and test matching
        test_results = {}
        
        for light_source in available_lights:
            print(f"\n🔍 Testing {light_source} Light Source...")
            
            # Extract structural features for this light source using correct patterns
            patterns = [
                f"{gem_id}BC%", f"{gem_id}BP%",  # Halogen patterns
                f"{gem_id}LC%", f"{gem_id}LP%",  # Laser patterns  
                f"{gem_id}UC%", f"{gem_id}UP%"   # UV patterns
            ]
            
            test_features = []
            files_found = []
            
            for pattern in patterns:
                query = """
                    SELECT file, feature_group, wavelength, intensity, 
                           start_wavelength, end_wavelength, midpoint_wavelength,
                           crest_wavelength, max_wavelength
                    FROM structural_features 
                    WHERE file LIKE ? AND light_source = ?
                    ORDER BY wavelength
                """
                
                test_data_df = pd.read_sql_query(query, conn, params=[pattern, light_source])
                
                if not test_data_df.empty:
                    files_found.extend(test_data_df['file'].unique())
                    
                    # Convert to format expected by matching algorithms
                    for _, row in test_data_df.iterrows():
                        feature = {
                            'feature_type': row['feature_group'],
                            'wavelength': row['wavelength'],
                            'intensity': row['intensity']
                        }
                        
                        # Add optional wavelength fields
                        for field in ['start_wavelength', 'end_wavelength', 'midpoint_wavelength', 
                                     'crest_wavelength', 'max_wavelength']:
                            if pd.notna(row[field]):
                                feature[field] = row[field]
                        
                        test_features.append(feature)
            
            if not test_features:
                print(f"   ❌ No data found for {light_source}")
                continue
            
            print(f"   ✅ Extracted {len(test_features)} features from files: {', '.join(set(files_found))}")
            
            # Find all potential matches in database for this light source
            match_query = """
                SELECT 
                    CASE 
                        WHEN file LIKE '%BC%' THEN SUBSTR(file, 1, INSTR(file, 'BC') - 1)
                        WHEN file LIKE '%BP%' THEN SUBSTR(file, 1, INSTR(file, 'BP') - 1)
                        WHEN file LIKE '%LC%' THEN SUBSTR(file, 1, INSTR(file, 'LC') - 1)
                        WHEN file LIKE '%LP%' THEN SUBSTR(file, 1, INSTR(file, 'LP') - 1)
                        WHEN file LIKE '%UC%' THEN SUBSTR(file, 1, INSTR(file, 'UC') - 1)
                        WHEN file LIKE '%UP%' THEN SUBSTR(file, 1, INSTR(file, 'UP') - 1)
                        ELSE SUBSTR(file, 1, LENGTH(file) - 3)
                    END as candidate_gem_id,
                    file, COUNT(*) as feature_count
                FROM structural_features 
                WHERE light_source = ? AND candidate_gem_id != ? AND candidate_gem_id != ''
                GROUP BY candidate_gem_id, file
                HAVING feature_count >= 3
                ORDER BY feature_count DESC
            """
            
            candidates_df = pd.read_sql_query(match_query, conn, params=[light_source, gem_id])
            
            if candidates_df.empty:
                print(f"   ❌ No candidates found for comparison")
                continue
            
            print(f"   🎯 Found {len(candidates_df)} candidates for comparison")
            
            # Run matching against each candidate
            match_scores = []
            
            for _, candidate_row in candidates_df.iterrows()[:10]:  # Test top 10 candidates
                candidate_gem_id = candidate_row['candidate_gem_id']
                candidate_file = candidate_row['file']
                
                # Get candidate features
                candidate_query = """
                    SELECT file, feature_group, wavelength, intensity,
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
                        'intensity': row['intensity']
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
                    'feature_count': len(candidate_features)
                })
            
            # Sort by score
            match_scores.sort(key=lambda x: x['score'], reverse=True)
            test_results[light_source] = match_scores
            
            # Show results
            print(f"   📊 TOP MATCHES:")
            for i, match in enumerate(match_scores[:5], 1):
                score_indicator = "🟢" if match['score'] > 90 else "🟡" if match['score'] > 70 else "🔴"
                self_match = "⭐ SELF-MATCH!" if match['candidate_gem_id'] == gem_id else ""
                print(f"      {i}. Gem {match['candidate_gem_id']}: {match['score']:.1f}% {score_indicator} {self_match}")
        
        # Summary results
        print(f"\n📋 STRUCTURAL MATCHING TEST SUMMARY: Gem {gem_id}")
        print("=" * 60)
        
        self_match_found = False
        for light_source, matches in test_results.items():
            if matches:
                top_match = matches[0]
                is_self = top_match['candidate_gem_id'] == gem_id
                if is_self:
                    self_match_found = True
                
                status = "✅ SELF-MATCH!" if is_self else f"❌ Matched: Gem {top_match['candidate_gem_id']}"
                print(f"   {light_source}: {top_match['score']:.1f}% - {status}")
        
        print(f"\n🎯 OVERALL TEST RESULT:")
        if self_match_found:
            print("   ✅ SUCCESS: Structural matching can identify gems correctly!")
            print("   💡 Ready to test with unknown gems")
            if self.bleep_enabled:
                self.play_bleep(feature_type="completion")
        else:
            print("   ⚠️ MIXED RESULTS: Some light sources found self-match, others didn't")
            print("   🔧 This is normal for testing - algorithms may need tuning")
        
        # Show algorithm suggestions
        print(f"\n🔬 ALGORITHM STATUS:")
        print("   📌 Current: Simple wavelength comparison")
        print("   🚀 Available: Enhanced UV ratio analysis, multi-light integration")
        print("   💡 Next step: Test with unknown gems vs your reference database")
    
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
                
                best_score = 0.0
                for candidate_feature in candidate_features_type:
                    candidate_wl = candidate_feature.get('wavelength', 0)
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
                    
                    best_score = max(best_score, score)
                
                total_score += best_score
            
            type_score = total_score / len(test_features_type) if test_features_type else 0.0
            type_scores.append(type_score)
        
        # Average across all feature types
        final_score = sum(type_scores) / len(type_scores) if type_scores else 0.0
        return final_score
    
    def debug_structural_database(self):
        """Debug function to examine structural database content"""
        print("\n🔧 STRUCTURAL DATABASE DEBUG")
        print("=" * 50)
        
        # DEBUG: Show exactly what path we're trying to access
        print(f"🎯 Attempting to access database at:")
        print(f"   {self.db_path}")
        print(f"🔍 File exists: {os.path.exists(self.db_path)}")
        if os.path.exists(self.db_path):
            actual_size = os.path.getsize(self.db_path) / 1024
            print(f"📊 Actual file size: {actual_size:.1f} KB")
        print()
        
        if not os.path.exists(self.db_path):
            print(f"❌ Database not found: {self.db_path}")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check table structure
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(structural_features)")
            columns = cursor.fetchall()
            
            print("📋 TABLE STRUCTURE:")
            for col in columns:
                print(f"   {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULLABLE'}")
            
            # Basic counts
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total_records = cursor.fetchone()[0]
            print(f"\n📊 Total records: {total_records:,}")
            
            if total_records == 0:
                print("❌ No structural features found in database")
                
                # Check if there are other tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                print(f"\n📋 Found {len(tables)} tables:")
                for table_name, in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    print(f"   {table_name}: {count} records")
                
                conn.close()
                return
            
            # Check light sources
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source")
            light_sources = cursor.fetchall()
            print(f"\n💡 Light Sources:")
            for light, count in light_sources:
                print(f"   {light}: {count:,} records")
            
            # Check file distribution
            cursor.execute("SELECT file, light_source, COUNT(*) FROM structural_features GROUP BY file, light_source ORDER BY file")
            files = cursor.fetchall()
            print(f"\n📄 Files in database ({len(files)} total):")
            
            # Show first 20 files
            for i, (file, light, count) in enumerate(files[:20]):
                print(f"   {i+1:2d}. {file}: {light} ({count} features)")
            
            if len(files) > 20:
                print(f"   ... and {len(files) - 20} more files")
            
            # Check feature types
            cursor.execute("SELECT feature_group, COUNT(*) FROM structural_features GROUP BY feature_group ORDER BY COUNT(*) DESC")
            feature_types = cursor.fetchall()
            print(f"\n🏷️  Feature Types:")
            for feature_type, count in feature_types:
                print(f"   {feature_type}: {count:,}")
            
            # Test the matching query
            print(f"\n🔍 Testing structural matching query...")
            matching_query = """
                SELECT 
                    CASE 
                        WHEN file LIKE '%BC%' THEN SUBSTR(file, 1, INSTR(file, 'BC') - 1)
                        WHEN file LIKE '%BP%' THEN SUBSTR(file, 1, INSTR(file, 'BP') - 1)
                        WHEN file LIKE '%LC%' THEN SUBSTR(file, 1, INSTR(file, 'LC') - 1)
                        WHEN file LIKE '%LP%' THEN SUBSTR(file, 1, INSTR(file, 'LP') - 1)
                        WHEN file LIKE '%UC%' THEN SUBSTR(file, 1, INSTR(file, 'UC') - 1)
                        WHEN file LIKE '%UP%' THEN SUBSTR(file, 1, INSTR(file, 'UP') - 1)
                        ELSE SUBSTR(file, 1, LENGTH(file) - 3)
                    END as gem_id,
                    GROUP_CONCAT(DISTINCT light_source) as light_sources,
                    COUNT(DISTINCT light_source) as light_count,
                    COUNT(*) as total_features
                FROM structural_features 
                WHERE file NOT LIKE '%unknown%'
                GROUP BY gem_id
                HAVING light_count >= 2 AND gem_id != ''
                ORDER BY light_count DESC, total_features DESC
            """
            
            matching_results = pd.read_sql_query(matching_query, conn)
            
            if not matching_results.empty:
                print(f"🎯 Found {len(matching_results)} testable gems:")
                for i, row in matching_results.iterrows():
                    status = "✅ COMPLETE" if row['light_count'] == 3 else "🟡 PARTIAL"
                    print(f"   {i+1}. Gem {row['gem_id']}: {row['light_sources']} ({row['light_count']} sources, {row['total_features']} features) {status}")
            else:
                print("❌ No testable gems found by matching query")
            
            # Sample some actual data
            cursor.execute("SELECT * FROM structural_features LIMIT 5")
            sample_data = cursor.fetchall()
            
            print(f"\n📋 SAMPLE DATA (First 5 records):")
            for i, row in enumerate(sample_data, 1):
                print(f"   {i}. File: {row[2]}, Light: {row[3]}, Wavelength: {row[4]:.1f}nm, Feature: {row[7]}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            import traceback
            traceback.print_exc()
    
    def check_both_database_files(self):
        """Check both database files to see which has data"""
        print("\n🔍 CHECKING BOTH DATABASE FILES")
        print("=" * 50)
        
        database_files = [
            r"C:\Users\David\OneDrive\Desktop\gemini_gemological_analysis\database\structural_spectra\multi_structural_gem_data.db",
            r"C:\Users\David\OneDrive\Desktop\gemini_gemological_analysis\database\structural_spectra\fixed_structural_gem_data.db"
        ]
        
        for db_path in database_files:
            print(f"\n📄 Checking: {db_path}")
            
            if not os.path.exists(db_path):
                print(f"   ❌ File not found")
                continue
                
            try:
                # Get file size
                size_kb = os.path.getsize(db_path) / 1024
                print(f"   📊 Size: {size_kb:.1f} KB")
                
                # Connect and explore
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                print(f"   📋 Tables: {[t[0] for t in tables]}")
                
                # Check each table for records
                for table_name, in tables:
                    if table_name.startswith('sqlite_'):
                        continue  # Skip system tables
                        
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        count = cursor.fetchone()[0]
                        print(f"      📊 {table_name}: {count:,} records")
                        
                        if count > 0:
                            # Show sample data
                            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                            samples = cursor.fetchall()
                            print(f"      🔬 Sample data:")
                            for i, row in enumerate(samples, 1):
                                sample_str = str(row)[:100] + "..." if len(str(row)) > 100 else str(row)
                                print(f"         {i}. {sample_str}")
                                
                    except Exception as e:
                        print(f"      ❌ Error reading {table_name}: {e}")
                
                conn.close()
                
            except Exception as e:
                print(f"   ❌ Database error: {e}")
    
    def check_system_status(self):
        """Check overall system status with enhanced database info"""
        print("ENHANCED GEMINI GEMOLOGICAL ANALYSIS SYSTEM STATUS")
        print("=" * 55)
        
        # Check database files
        db_files_ok = 0
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                size = os.path.getsize(db_file) // (1024*1024)  # MB
                print(f"✅ {db_file} ({size} MB)")
                db_files_ok += 1
            else:
                print(f"❌ {db_file} (missing)")
        
        # Check program files
        programs_ok = 0
        for prog_file, description in self.program_files.items():
            if os.path.exists(prog_file):
                print(f"✅ {description}")
                programs_ok += 1
            else:
                print(f"❌ {description} (missing)")
        
        # Check structural database - CORRECTED
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                count = pd.read_sql_query("SELECT COUNT(*) as count FROM structural_features", conn).iloc[0]['count']
                print(f"✅ Structural database ({count:,} features)")
                conn.close()
            except Exception as e:
                print(f"❌ Structural database (error reading: {e})")
        else:
            print(f"❌ Structural database (missing: {self.db_path})")
        
        # Check data directories
        data_dirs = ['data/raw', 'data/unknown']
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                files = len([f for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.csv')])
                print(f"✅ {data_dir} ({files} files)")
            else:
                print(f"❌ {data_dir} (missing)")
        
        # Enhanced features status
        print(f"\n🔧 ENHANCED FEATURES:")
        print(f"   🔊 Audio bleep: {'ON' if self.bleep_enabled else 'OFF'}")
        print(f"   🔄 Auto-import: {'ON' if self.auto_import_enabled else 'OFF'}")
        print(f"   📏 Relative height cache: {len(self.relative_height_cache)} entries")
        print(f"   🎵 Audio system: {'Available' if HAS_AUDIO else 'Not available'}")
        print(f"   🗄️ Database path: {self.db_path}")
        
        print(f"\nSystem Status: {db_files_ok}/3 databases, {programs_ok}/{len(self.program_files)} programs")
        print("=" * 55)
        
        return db_files_ok >= 3 and programs_ok >= 2
    
    def scan_available_gems(self):
        """Scan data/raw for available gems"""
        raw_dir = 'data/raw'
        if not os.path.exists(raw_dir):
            print(f"❌ Directory {raw_dir} not found!")
            return None
        
        files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
        if not files:
            print(f"❌ No .txt files in {raw_dir}")
            return None
        
        # Group by gem number
        gems = defaultdict(lambda: {'B': [], 'L': [], 'U': []})
        
        for file in files:
            base = os.path.splitext(file)[0]
            
            # Find light source
            light = None
            for ls in ['B', 'L', 'U']:
                if ls in base.upper():
                    light = ls
                    break
            
            if light:
                # Extract gem number
                for i, char in enumerate(base.upper()):
                    if char == light:
                        gem_num = base[:i]
                        break
                gems[gem_num][light].append(file)
        
        return dict(gems)
    
    def show_available_gems(self, gems):
        """Display available gems"""
        print("\n📂 AVAILABLE GEMS FOR ANALYSIS")
        print("=" * 50)
        
        complete_gems = []
        partial_gems = []
        
        for gem_num in sorted(gems.keys()):
            gem_files = gems[gem_num]
            available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]
            
            if len(available) == 3:
                complete_gems.append(gem_num)
                files_summary = []
                for ls in ['B', 'L', 'U']:
                    count = len(gems[gem_num][ls])
                    files_summary.append(f"{ls}:{count}")
                print(f"   ✅ Gem {gem_num} ({', '.join(files_summary)})")
            else:
                partial_gems.append((gem_num, available))
                print(f"   🟡 Gem {gem_num} (only: {'+'.join(available)})")
        
        return complete_gems, partial_gems
    
    def select_and_analyze_gem(self):
        """Complete gem selection and analysis workflow with enhanced features"""
        print("\n🎯 ENHANCED GEM SELECTION AND ANALYSIS")
        print("=" * 45)

        # Clear any previous analysis results to prevent caching issues
        for file in ['unkgemB.csv', 'unkgemL.csv', 'unkgemU.csv']:
            if os.path.exists(file):
                os.remove(file)
            if os.path.exists(f'data/unknown/{file}'):
                os.remove(f'data/unknown/{file}')

        # Scan gems
        gems = self.scan_available_gems()
        if not gems:
            return

        # Show ALL available files with numbers
        print("\n📂 AVAILABLE FILES FOR ANALYSIS")
        print("=" * 50)

        all_files = []
        for gem_num in sorted(gems.keys()):
            gem_files = gems[gem_num]
            available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]

            if len(available) == 3:  # Only show complete gems
                print(f"\n✅ Gem {gem_num}:")
                for light in ['B', 'L', 'U']:
                    for file in gem_files[light]:
                        file_base = file.replace('.txt', '')
                        all_files.append((file_base, file, gem_num, light))
                        print(f"   {len(all_files)}. {file_base}")

        if not all_files:
            print("\n❌ No complete gem sets found!")
            return

        print(f"\n🔍 SELECTION METHOD:")
        print("Enter 3 file numbers (B, L, U) separated by spaces")
        print("Example: 1 5 9 (for files 1, 5, and 9)")
        print("Or enter a gem base number like 'C0045' for auto-selection")

        choice = input("\nYour selection: ").strip()

        selected = {}

        # Try parsing as numbers first
        try:
            numbers = [int(x) for x in choice.split()]
            if len(numbers) == 3:
                selected_files = []
                for num in numbers:
                    if 1 <= num <= len(all_files):
                        selected_files.append(all_files[num-1])
                    else:
                        print(f"❌ Number {num} out of range (1-{len(all_files)})")
                        return

                # Check if we have B, L, U
                lights_found = {f[3] for f in selected_files}
                if lights_found != {'B', 'L', 'U'}:
                    print(f"❌ Need one file from each light source (B, L, U)")
                    print(f"You selected: {lights_found}")
                    return

                # Store selected files
                for file_info in selected_files:
                    file_base, file_full, gem_num, light = file_info
                    selected[light] = file_full
                    print(f"   Selected {light}: {file_base}")

                gem_choice = selected_files[0][2]  # Use gem number from first file

            else:
                print("❌ Please enter exactly 3 numbers")
                return

        except ValueError:
            # Try as gem base number (old method)
            if choice in gems:
                gem_choice = choice
                gem_files = gems[gem_choice]

                print(f"\n💎 AUTO-SELECTING FILES FOR GEM {gem_choice}:")
                for light in ['B', 'L', 'U']:
                    if gem_files[light]:
                        selected[light] = gem_files[light][0]
                        file_base = selected[light].replace('.txt', '')
                        print(f"   {light}: {file_base}")
            else:
                print(f"❌ Invalid selection. Use numbers or gem base like 'C0045'")
                return

        if len(selected) != 3:
            print("\n❌ Incomplete selection - need B, L, and U files")
            return

        # Convert files with CORRECTED normalization
        print(f"\n🔄 PREPARING ENHANCED ANALYSIS...")
        success = self.convert_gem_files_corrected(selected, gem_choice)

        if success:
            # Run validation check
            print(f"\n🔍 VALIDATING NORMALIZATION...")
            self.validate_normalization(gem_choice)

            # Run analysis with enhanced features
            print(f"\n✅ FILES READY FOR ENHANCED ANALYSIS")
            analysis_choice = input(f"Run enhanced numerical analysis now? (y/n): ").strip().lower()

            if analysis_choice == 'y':
                self.run_numerical_analysis_fixed()

                # Auto-import to database if enabled
                if self.auto_import_enabled:
                    print(f"\n🔄 Auto-importing analysis results to database...")
                    # Import any generated CSV files
                    for light in ['B', 'L', 'U']:
                        csv_path = f'data/unknown/unkgem{light}.csv'
                        if os.path.exists(csv_path):
                            self.auto_import_csv_to_database(csv_path)

                # Offer enhanced visualization
                viz_choice = input(f"\nShow enhanced spectral comparison plots? (y/n): ").strip().lower()
                if viz_choice == 'y':
                    self.create_spectral_comparison_plots(gem_choice)
                
                # Offer relative height analysis
                rel_choice = input(f"\nPerform relative height analysis? (y/n): ").strip().lower()
                if rel_choice == 'y':
                    target_wl = float(input("Enter target wavelength for comparison: "))
                    measurements = self.calculate_relative_height(gem_choice, target_wl)
                    if measurements:
                        print(f"\nRelative height measurements calculated and cached")
                        self.play_bleep(feature_type="completion")
        else:
            print(f"\n❌ Failed to prepare analysis")
    
    def convert_gem_files_corrected(self, selected_files, gem_number):
        """Convert selected gem files with CORRECTED normalization"""
        try:
            # Try the normal method first
            if os.path.exists('raw_txt'):
                shutil.rmtree('raw_txt')
            os.makedirs('raw_txt')
            
            # Copy files to raw_txt
            print("   📁 Copying files to raw_txt...")
            for light, filename in selected_files.items():
                src = os.path.join('data/raw', filename)
                dst = os.path.join('raw_txt', filename)
                shutil.copy2(src, dst)
                print(f"     ✅ {light}: {filename}")
            
            # Create data/unknown directory
            os.makedirs('data/unknown', exist_ok=True)
            
            # Convert each file with CORRECTED normalization
            print("   🔧 Converting and normalizing (ENHANCED)...")
            
            for light, filename in selected_files.items():
                input_path = os.path.join('raw_txt', filename)
                output_path = f'data/unknown/unkgem{light}.csv'
                
                # Read file
                df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                wavelengths = np.array(df['wavelength'])
                intensities = np.array(df['intensity'])
                
                # Apply CORRECTED normalization
                normalized = self.correct_normalize_spectrum(wavelengths, intensities, light)
                
                # Save normalized data
                output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
                output_df.to_csv(output_path, header=False, index=False)
                
                print(f"     ✅ {light}: {len(output_df)} points, range {normalized.min():.3f}-{normalized.max():.3f}")
                
                # Play bleep for each successful conversion
                if self.bleep_enabled:
                    self.play_bleep(feature_type="peak")
            
            return True
            
        except (PermissionError, OSError) as e:
            print(f"     ⚠️ Permission error with raw_txt: {e}")
            print("     🔄 Switching to BYPASS MODE (direct conversion)...")
            return self.convert_gem_files_bypass(selected_files, gem_number)
        except Exception as e:
            print(f"     ❌ Conversion error: {e}")
            return False
    
    def convert_gem_files_bypass(self, selected_files, gem_number):
        """Convert files directly without raw_txt copying"""
        try:
            # Create data/unknown directory only
            os.makedirs('data/unknown', exist_ok=True)
            
            print("   🔧 Converting directly (BYPASS MODE)...")
            
            for light, filename in selected_files.items():
                input_path = os.path.join('data/raw', filename)
                output_path = f'data/unknown/unkgem{light}.csv'
                
                # Read and normalize directly from data/raw
                df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                wavelengths = np.array(df['wavelength'])
                intensities = np.array(df['intensity'])
                
                # Apply CORRECTED normalization
                normalized = self.correct_normalize_spectrum(wavelengths, intensities, light)
                
                # Save normalized data
                output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
                output_df.to_csv(output_path, header=False, index=False)
                
                print(f"     ✅ {light}: {len(output_df)} points, range {normalized.min():.3f}-{normalized.max():.3f}")
                
                # Play bleep for each successful conversion
                if self.bleep_enabled:
                    self.play_bleep(feature_type="peak")
            
            return True
            
        except Exception as e:
            print(f"     ❌ Bypass conversion error: {e}")
            return False
    
    def validate_normalization(self, gem_number):
        """Validate that normalization produces expected results"""
        print("   🔍 Checking normalization against database...")
        
        for light in ['B', 'L', 'U']:
            try:
                # Load our normalized data
                unknown_path = f'data/unknown/unkgem{light}.csv'
                unknown_df = pd.read_csv(unknown_path, header=None, names=['wavelength', 'intensity'])
                
                # Load database
                db_path = f'database/reference_spectra/gemini_db_long_{light}.csv'
                if os.path.exists(db_path):
                    db_df = pd.read_csv(db_path)
                    
                    # Look for exact gem match in database
                    gem_matches = db_df[db_df['full_name'].str.contains(gem_number, na=False)]
                    
                    if not gem_matches.empty:
                        # Get first match
                        match = gem_matches.iloc[0]
                        print(f"     🎯 {light}: Found {match['full_name']} in database")
                        
                        # Compare ranges
                        unknown_range = f"{unknown_df['intensity'].min():.3f}-{unknown_df['intensity'].max():.3f}"
                        db_subset = db_df[db_df['full_name'] == match['full_name']]
                        db_range = f"{db_subset['intensity'].min():.3f}-{db_subset['intensity'].max():.3f}"
                        
                        print(f"         Unknown range: {unknown_range}")
                        print(f"         Database range: {db_range}")
                        
                        # Play validation bleep
                        if self.bleep_enabled:
                            self.play_bleep(feature_type="valley")
                    else:
                        print(f"     ⚠️ {light}: No match for {gem_number} in database")
                else:
                    print(f"     ❌ {light}: Database file {db_path} not found")
                    
            except Exception as e:
                print(f"     ❌ {light}: Validation error - {e}")
    
    def run_numerical_analysis_fixed(self):
        """Run numerical analysis with enhanced features"""
        print(f"\n🚀 RUNNING ENHANCED NUMERICAL ANALYSIS...")
        
        try:
            # Run analysis directly in this process
            self.direct_numerical_analysis()
                
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def direct_numerical_analysis(self):
        """Direct numerical analysis with enhanced features and caching"""
        print("   📊 Starting enhanced analysis with advanced features...")
        
        # Clear any previous analysis variables
        self.current_analysis_results = {}
        self.current_gem_identifier = None
        
        # Check for unknown files
        unknown_files = {}
        available_lights = []
        
        for light in ['B', 'L', 'U']:
            found = False
            for base_path in ['data/unknown', '.']:
                test_path = os.path.join(base_path, f'unkgem{light}.csv')
                if os.path.exists(test_path):
                    unknown_files[light] = test_path
                    available_lights.append(light)
                    found = True
                    break
        
        if len(available_lights) < 2:
            print(f"   ❌ Need at least 2 light sources, found: {available_lights}")
            return
        
        print(f"   ✅ Found {len(available_lights)} light sources: {'+'.join(available_lights)}")
        
        # Database files
        db_files = {
            'B': 'database/reference_spectra/gemini_db_long_B.csv', 
            'L': 'database/reference_spectra/gemini_db_long_L.csv', 
            'U': 'database/reference_spectra/gemini_db_long_U.csv'
        }
        
        # Check database files
        for light in available_lights:
            if not os.path.exists(db_files[light]):
                print(f"   ❌ Database file {db_files[light]} not found")
                return
        
        print("   ✅ All required database files found")
        
        # Determine gem ID
        actual_gem_id = self.identify_unknown_gem(unknown_files)
        print(f"   🎯 Analyzing enhanced gem: {actual_gem_id}")
        
        # Load gem library for descriptions
        gem_name_map = {}
        try:
            gemlib = pd.read_csv('gemlib_structural_ready.csv')
            gemlib.columns = gemlib.columns.str.strip()
            if 'Reference' in gemlib.columns:
                gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                if all(col in gemlib.columns for col in expected_columns):
                    gemlib['Gem Description'] = gemlib[expected_columns].apply(
                        lambda x: ' '.join([v if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                    gem_name_map = dict(zip(gemlib['Reference'], gemlib['Gem Description']))
        except Exception as e:
            print(f"   ⚠️ Could not load gem descriptions: {e}")
        
        # Process each light source with enhanced features
        all_matches = {}
        gem_best_scores = {}
        gem_best_names = {}
        
        for light_source in available_lights:
            print(f"\n   🔍 Processing {light_source} light (ENHANCED)...")
            
            try:
                # Load unknown spectrum
                unknown = pd.read_csv(unknown_files[light_source], header=None, names=['wavelength', 'intensity'])
                print(f"      Unknown: {len(unknown)} points, range {unknown['intensity'].min():.3f}-{unknown['intensity'].max():.3f}")
                
                # Load database
                db = pd.read_csv(db_files[light_source])
                print(f"      Database: {len(db)} points, {db['full_name'].nunique()} unique gems")
                
                # Apply 0-100 scaling
                unknown_scaled = unknown.copy()
                unknown_scaled['intensity'] = self.apply_0_100_scaling(unknown['wavelength'].values, unknown['intensity'].values)
                print(f"      Unknown scaled: range {unknown_scaled['intensity'].min():.3f}-{unknown_scaled['intensity'].max():.3f}")
                
                # Compute enhanced scores with progress feedback
                current_scores = []
                unique_gems = db['full_name'].unique()
                
                for i, gem_name in enumerate(unique_gems):
                    if i % 100 == 0:  # Progress indicator
                        print(f"      Processing gem {i+1}/{len(unique_gems)}...")
                    
                    reference = db[db['full_name'] == gem_name].copy()
                    
                    # Apply 0-100 scaling to database reference
                    reference_scaled = reference.copy()
                    reference_scaled['intensity'] = self.apply_0_100_scaling(reference['wavelength'].values, reference['intensity'].values)
                    
                    # Compute match score
                    merged = pd.merge(unknown_scaled, reference_scaled, on='wavelength', suffixes=('_unknown', '_ref'))
                    if len(merged) > 0:
                        mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
                        log_score = np.log1p(mse)
                        current_scores.append((gem_name, log_score))
                
                # Sort by score
                current_sorted_scores = sorted(current_scores, key=lambda x: x[1])
                all_matches[light_source] = current_sorted_scores
                
                print(f"      ✅ Enhanced matches for {light_source}:")
                for i, (gem, score) in enumerate(current_sorted_scores[:5], 1):
                    print(f"         {i}. {gem}: {score:.6f}")
                    
                    # Play bleep for very good matches
                    if self.bleep_enabled and score < 1e-6:
                        self.play_bleep(feature_type="significant")
                
                # Track best scores per gem ID
                for gem_name, score in current_sorted_scores:
                    base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                    if base_id not in gem_best_scores:
                        gem_best_scores[base_id] = {}
                        gem_best_names[base_id] = {}
                    if score < gem_best_scores[base_id].get(light_source, np.inf):
                        gem_best_scores[base_id][light_source] = score
                        gem_best_names[base_id][light_source] = gem_name
                
            except Exception as e:
                print(f"      ❌ Error processing {light_source}: {e}")
        
        # Filter to gems with all available light sources
        complete_gems = {gid: scores for gid, scores in gem_best_scores.items() 
                        if set(scores.keys()) >= set(available_lights)}
        
        # Calculate combined scores
        aggregated_scores = {base_id: sum(scores.values()) 
                           for base_id, scores in complete_gems.items()}
        
        # Sort final results
        final_sorted = sorted(aggregated_scores.items(), key=lambda x: x[1])
        
        print(f"\n🏆 ENHANCED ANALYSIS RESULTS - TOP 20 MATCHES:")
        print("=" * 75)
        print(f"   Analysis using: {'+'.join(available_lights)} light sources")
        print("=" * 75)
        
        for i, (base_id, total_score) in enumerate(final_sorted[:20], start=1):
            gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
            sources = complete_gems.get(base_id, {})
            
            print(f"  Rank {i:2}: {gem_desc} (ID: {base_id})")
            print(f"          Total Score: {total_score:.6f}")
            for ls in sorted(available_lights):
                if ls in sources:
                    score_val = sources[ls]
                    best_file = gem_best_names[base_id][ls]
                    print(f"          {ls} Score: {score_val:.6f} (vs {best_file})")
            print()
        
        # Enhanced self-matching analysis
        if actual_gem_id in aggregated_scores:
            self_rank = next(i for i, (gid, _) in enumerate(final_sorted, 1) if gid == actual_gem_id)
            self_score = aggregated_scores[actual_gem_id]
            print(f"🎯 {actual_gem_id} ENHANCED SELF-MATCH RESULT:")
            print(f"   Rank: {self_rank}")
            print(f"   Total Score: {self_score:.6f}")
            print(f"   Light sources used: {'+'.join(available_lights)}")
            
            if self_score < 1e-10:
                print(f"   ✅ PERFECT SELF-MATCH!")
                if self.bleep_enabled:
                    self.play_bleep(feature_type="completion")
            elif self_score < 1e-6:
                print(f"   ✅ EXCELLENT SELF-MATCH!")
                if self.bleep_enabled:
