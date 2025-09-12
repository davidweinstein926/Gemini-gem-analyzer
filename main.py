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
            
            print(f"\n🏷️ TOP FEATURE TYPES:")
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
        
        # Check structural database
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                count = pd.read_sql_query("SELECT COUNT(*) as count FROM structural_features", conn).iloc[0]['count']
                print(f"✅ Structural database ({count:,} features)")
                conn.close()
            except:
                print(f"❌ Structural database (error reading)")
        else:
            print(f"❌ Structural database (missing)")
        
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
                    self.play_bleep(feature_type="significant")
            elif self_score < 1e-3:
                print(f"   ✅ GOOD SELF-MATCH!")
                if self.bleep_enabled:
                    self.play_bleep(feature_type="peak")
            else:
                print(f"   ⚠️ POOR SELF-MATCH - check normalization")
        else:
            print(f"🎯 {actual_gem_id} NOT FOUND in results - check database entries")
        
        print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
        print(f"   Analyzed gem: {actual_gem_id}")
        print(f"   Light sources: {'+'.join(available_lights)} ({len(available_lights)}/3)")
        print(f"   Enhanced features: Audio feedback, auto-import, relative heights")
        print(f"   Total gems analyzed: {len(final_sorted)}")
        print(f"   Perfect matches (score < 1e-10): {sum(1 for _, score in final_sorted if score < 1e-10)}")
        
        # Store results for enhanced features
        self.current_analysis_results = final_sorted
        self.current_gem_identifier = actual_gem_id
        
        # Generate enhanced reports
        self.create_analysis_summary_report(actual_gem_id)
        self.create_csv_results_report(actual_gem_id)
        
        # Final completion bleep
        if self.bleep_enabled:
            self.play_bleep(feature_type="completion")
        
        return final_sorted
    
    def identify_unknown_gem(self, unknown_files):
        """Identify which gem we're analyzing"""
        # Try to determine from source files in raw_txt if available
        if os.path.exists('raw_txt'):
            txt_files = [f for f in os.listdir('raw_txt') if f.endswith('.txt')]
            if txt_files:
                # Extract gem ID from first file
                first_file = txt_files[0]
                file_base = first_file.replace('.txt', '')
                # Extract everything before the light source letter
                for light in ['B', 'L', 'U']:
                    if light in file_base.upper():
                        return file_base[:file_base.upper().find(light)]
        
        # Fallback: check if unknown file matches known patterns in database
        try:
            if 'B' in unknown_files:
                unknown_b = pd.read_csv(unknown_files['B'], header=None, names=['wavelength', 'intensity'])
                db_b = pd.read_csv('database/reference_spectra/gemini_db_long_B.csv')
                
                # Find exact matches in database
                for gem_name in db_b['full_name'].unique():
                    reference = db_b[db_b['full_name'] == gem_name]
                    merged = pd.merge(unknown_b, reference, on='wavelength', suffixes=('_unknown', '_ref'))
                    if len(merged) > 0:
                        mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
                        if mse < 1e-10:  # Perfect match
                            base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                            return base_id
        except:
            pass
        
        return "UNKNOWN"
    
    def create_analysis_summary_report(self, gem_identifier):
        """Create enhanced text summary report"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('results/post analysis numerical/reports', exist_ok=True)
        filename = f'results/post analysis numerical/reports/enhanced_analysis_summary_{gem_identifier}_{timestamp}.txt'
        
        # Get available light sources
        available_lights = []
        for light in ['B', 'L', 'U']:
            for base_path in ['data/unknown', '.']:
                if os.path.exists(os.path.join(base_path, f'unkgem{light}.csv')):
                    available_lights.append(light)
                    break
        
        with open(filename, 'w') as f:
            f.write("============================================================\n")
            f.write("ENHANCED GEMINI NUMERICAL ANALYSIS SUMMARY\n")
            f.write("============================================================\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Light Sources Analyzed: {', '.join(available_lights)}\n")
            f.write(f"Total Candidates Analyzed: {len(self.current_analysis_results)}\n")
            f.write(f"Enhanced Features: Audio bleep, auto-import, relative heights\n")
            f.write(f"Audio System: {'Available' if HAS_AUDIO else 'Not available'}\n")
            f.write(f"Bleep Enabled: {'Yes' if self.bleep_enabled else 'No'}\n")
            f.write(f"Auto-import Enabled: {'Yes' if self.auto_import_enabled else 'No'}\n\n")
            
            f.write("TOP 10 ENHANCED MATCHES:\n")
            f.write("----------------------------------------\n")
            
            # Load gem descriptions
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
            except:
                pass
            
            # Enhanced analysis details
            for i, (base_id, total_score) in enumerate(self.current_analysis_results[:10], 1):
                gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
                
                f.write(f"Rank {i}: Gem ID {base_id}\n")
                f.write(f"  Description: {gem_desc}\n")
                f.write(f"  Total Score: {total_score:.6f}\n")
                f.write(f"  Light Sources: {', '.join(available_lights)}\n")
                
                # Score classification
                if total_score < 1e-10:
                    f.write(f"  Match Quality: PERFECT\n")
                elif total_score < 1e-6:
                    f.write(f"  Match Quality: EXCELLENT\n")
                elif total_score < 1e-3:
                    f.write(f"  Match Quality: GOOD\n")
                else:
                    f.write(f"  Match Quality: POOR\n")
                
                f.write("\n")
            
            # Enhanced features summary
            f.write("ENHANCED FEATURES USED:\n")
            f.write("----------------------------------------\n")
            f.write(f"• Audio Feedback: {'Enabled' if self.bleep_enabled else 'Disabled'}\n")
            f.write(f"• Auto-import: {'Enabled' if self.auto_import_enabled else 'Disabled'}\n")
            f.write(f"• Relative Height Cache: {len(self.relative_height_cache)} entries\n")
            f.write(f"• Database Integration: Active\n")
            f.write(f"• Enhanced Validation: Applied\n")
        
        print(f"Enhanced analysis summary saved: {filename}")
    
    def create_csv_results_report(self, gem_identifier):
        """Create enhanced CSV report with additional metadata"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('results/post analysis numerical/reports', exist_ok=True)
        filename = f'results/post analysis numerical/reports/enhanced_gemini_results_{gem_identifier}_{timestamp}.csv'
        
        # Get available light sources
        available_lights = []
        for light in ['B', 'L', 'U']:
            for base_path in ['data/unknown', '.']:
                if os.path.exists(os.path.join(base_path, f'unkgem{light}.csv')):
                    available_lights.append(light)
                    break
        
        # Load gem descriptions
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
        except:
            pass
        
        # Prepare enhanced data for CSV
        csv_data = []
        
        for rank, (base_id, total_score) in enumerate(self.current_analysis_results, 1):
            gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
            
            # Enhanced classification
            if total_score < 1e-10:
                match_quality = "PERFECT"
            elif total_score < 1e-6:
                match_quality = "EXCELLENT"
            elif total_score < 1e-3:
                match_quality = "GOOD"
            else:
                match_quality = "POOR"
            
            row_data = {
                'rank': rank,
                'gem_id': base_id,
                'gem_description': gem_desc,
                'total_score': total_score,
                'match_quality': match_quality,
                'light_sources': '+'.join(available_lights),
                'analysis_type': 'enhanced',
                'audio_enabled': self.bleep_enabled,
                'auto_import_enabled': self.auto_import_enabled,
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add placeholder for individual light scores
            for light in ['B', 'L', 'U']:
                row_data[f'{light}_score'] = None
                row_data[f'{light}_gem_name'] = None
            
            csv_data.append(row_data)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        
        print(f"Enhanced CSV results saved: {filename}")
    
    def create_spectral_comparison_plots(self, gem_identifier):
        """Create enhanced spectral comparison plots with audio feedback"""
        print(f"\n📊 CREATING ENHANCED SPECTRAL PLOTS FOR {gem_identifier}")
        print("=" * 65)
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.widgets import Button
        except ImportError:
            print("❌ matplotlib not available - cannot create plots")
            return
        
        if not hasattr(self, 'current_analysis_results') or not self.current_analysis_results:
            print("❌ No analysis results available - run analysis first")
            return
        
        # Get top 5 matches
        top_matches = self.current_analysis_results[:5]
        
        # Create enhanced layout with additional features
        fig = plt.figure(figsize=(24, 20))
        gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 1, 0.5], hspace=0.3, wspace=0.2)
        
        fig.suptitle(f'Enhanced Spectral Analysis with Audio Feedback: {gem_identifier} vs Top Matches', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        # Enhanced legend elements
        legend_elements = []
        
        # Process each light source with enhanced features
        for light_idx, light in enumerate(['B', 'L', 'U']):
            print(f"🔍 Creating enhanced {light} light plots...")
            
            try:
                # Load unknown spectrum
                unknown_file = f'unkgem{light}.csv'
                if os.path.exists(unknown_file):
                    unknown = pd.read_csv(unknown_file, header=None, names=['wavelength', 'intensity'])
                else:
                    unknown_file = f'data/unknown/unkgem{light}.csv'
                    unknown = pd.read_csv(unknown_file, header=None, names=['wavelength', 'intensity'])
                
                # Load database
                db = pd.read_csv(f'database/reference_spectra/gemini_db_long_{light}.csv')
                
                # ROW 1: Normalized Spectra (Enhanced)
                ax1 = fig.add_subplot(gs[0, light_idx])
                
                # Plot unknown with enhanced styling
                unknown_line = ax1.plot(unknown['wavelength'], unknown['intensity'], 
                        'black', linewidth=0.7, alpha=0.9, zorder=10)
                
                # Add to legend only once
                if light_idx == 0:
                    legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=0.7, 
                                                    label=f'{gem_identifier} (Unknown)', alpha=0.9))
                
                # Plot enhanced top matches
                for i, (match_id, total_score) in enumerate(top_matches[:3]):
                    match_entries = db[db['full_name'].str.startswith(match_id)]
                    if not match_entries.empty:
                        match_file = f"{match_id}{light}C1"
                        if match_file not in match_entries['full_name'].values:
                            match_file = match_entries['full_name'].iloc[0]
                        
                        match_data = db[db['full_name'] == match_file]
                        if not match_data.empty:
                            match_line = ax1.plot(match_data['wavelength'], match_data['intensity'], 
                                    color=colors[i], linewidth=0.6, alpha=0.8)
                            
                            # Enhanced legend with quality indicators
                            if light_idx == 0:
                                quality = "PERFECT" if total_score < 1e-10 else "EXCELLENT" if total_score < 1e-6 else "GOOD"
                                legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=0.6,
                                                        label=f'#{i+1}: {match_id} ({quality})', alpha=0.8))
                
                ax1.set_title(f'{light} Light - Normalized Spectra', fontsize=14, fontweight='bold')
                ax1.set_ylabel('Normalized Intensity', fontsize=12)
                ax1.grid(True, alpha=0.3)
                
                # ROW 2: 0-100 Scaled Spectra (Enhanced)
                ax2 = fig.add_subplot(gs[1, light_idx])
                
                # Apply enhanced 0-100 scaling
                unknown_scaled = unknown.copy()
                unknown_scaled['intensity'] = self.apply_0_100_scaling(unknown['wavelength'].values, unknown['intensity'].values)
                
                # Plot enhanced scaled unknown
                ax2.plot(unknown_scaled['wavelength'], unknown_scaled['intensity'], 
                        'black', linewidth=0.7, alpha=0.9, zorder=10)
                
                # Plot enhanced scaled matches
                for i, (match_id, total_score) in enumerate(top_matches[:3]):
                    match_entries = db[db['full_name'].str.startswith(match_id)]
                    if not match_entries.empty:
                        match_file = f"{match_id}{light}C1"
                        if match_file not in match_entries['full_name'].values:
                            match_file = match_entries['full_name'].iloc[0]
                        
                        match_data = db[db['full_name'] == match_file]
                        if not match_data.empty:
                            match_scaled = match_data.copy()
                            match_scaled['intensity'] = self.apply_0_100_scaling(match_data['wavelength'].values, match_data['intensity'].values)
                            ax2.plot(match_scaled['wavelength'], match_scaled['intensity'], 
                                    color=colors[i], linewidth=0.6, alpha=0.8)
                
                ax2.set_title(f'{light} Light -  0-100 Scaled', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Scaled Intensity (0-100)', fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
                
            except Exception as e:
                print(f"❌ Error creating enhanced {light} light plots: {e}")
                # Fill with enhanced error message
                ax1 = fig.add_subplot(gs[0, light_idx])
                ax2 = fig.add_subplot(gs[1, light_idx])
                ax1.text(0.5, 0.5, f'Enhanced Error: {light} data: {e}', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=12, color='red')
                ax2.text(0.5, 0.5, f'Enhanced Error: {light} data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12, color='red')
        
        # ROW 3: Enhanced Score Distribution
        ax_dist = fig.add_subplot(gs[2, :])
        
        # Create enhanced score distribution
        all_scores = [score for _, score in self.current_analysis_results]
        
        # Enhanced histogram with quality regions
        counts, bins, patches = ax_dist.hist(all_scores, bins=50, alpha=0.7, color='lightblue', 
                                           edgecolor='black', linewidth=0.5)
        
        # Color code quality regions
        for i, patch in enumerate(patches):
            bin_center = (bins[i] + bins[i+1]) / 2
            if bin_center < 1e-10:
                patch.set_facecolor('green')
                patch.set_alpha(0.8)
            elif bin_center < 1e-6:
                patch.set_facecolor('yellow')
                patch.set_alpha(0.7)
            elif bin_center < 1e-3:
                patch.set_facecolor('orange')
                patch.set_alpha(0.6)
        
        # Enhanced vertical lines for top matches
        for i, (match_id, score) in enumerate(top_matches[:5]):
            color = colors[i] if i < len(colors) else 'gray'
            ax_dist.axvline(x=score, color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        # Enhanced statistics
        mean_score = np.mean(all_scores)
        median_score = np.median(all_scores)
        ax_dist.axvline(x=mean_score, color='orange', linestyle=':', linewidth=2, alpha=0.6)
        ax_dist.axvline(x=median_score, color='purple', linestyle=':', linewidth=2, alpha=0.6)
        
        ax_dist.set_title('Enhanced Score Distribution with Quality Regions', fontsize=14, fontweight='bold')
        ax_dist.set_xlabel('Log Score (Lower = Better Match)', fontsize=12)
        ax_dist.set_ylabel('Number of Gems', fontsize=12)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.set_yscale('log')
        
        # ROW 4: Enhanced Features Status
        ax_status = fig.add_subplot(gs[3, :])
        ax_status.axis('off')
        
        # Enhanced status information
        status_text = f"Enhanced Features: Audio Bleep: {'ON' if self.bleep_enabled else 'OFF'} | "
        status_text += f"Auto-import: {'ON' if self.auto_import_enabled else 'OFF'} | "
        status_text += f"Relative Height Cache: {len(self.relative_height_cache)} entries | "
        status_text += f"Audio System: {'Available' if HAS_AUDIO else 'Not available'}"
        
        ax_status.text(0.5, 0.5, status_text, ha='center', va='center', fontsize=12, 
                      bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        
        # Enhanced consolidated legend
        consolidated_legend = fig.legend(handles=legend_elements, 
                                       loc='upper right', 
                                       bbox_to_anchor=(0.98, 0.88),
                                       fontsize=11,
                                       title='Enhanced Spectral Plot Legend',
                                       title_fontsize=12,
                                       frameon=True,
                                       fancybox=True,
                                       shadow=True,
                                       framealpha=0.9)
        
        # Enhanced legend styling
        consolidated_legend.get_frame().set_facecolor('white')
        consolidated_legend.get_frame().set_edgecolor('black')
        consolidated_legend.get_frame().set_linewidth(1)
        
        # Make enhanced layout
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # Save enhanced plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('results/post analysis numerical/graph', exist_ok=True)
        plot_filename = f'results/post analysis numerical/graph/enhanced_spectral_analysis_{gem_identifier}_{timestamp}.png'
        plt.savefig(plot_filename, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"💾 Enhanced plot saved as: {plot_filename}")
        
        # Play completion bleep
        if self.bleep_enabled:
            self.play_bleep(feature_type="completion")
        
        try:
            plt.show()
            print("\n💡 ENHANCED TIP: Click on any spectral plot to see enlarged view!")
            print("📋 LEGEND: Enhanced consolidated legend shows match quality")
            print("📊 DISTRIBUTION: Color-coded quality regions (green=perfect, yellow=excellent, orange=good)")
            print("🔊 AUDIO: Audio feedback provided during analysis")
        except:
            print("⚠️ Cannot display plot interactively, but enhanced file saved successfully")
    
    def run_structural_analysis_hub(self):
        """Launch structural analysis hub with auto-import integration"""
        hub_path = 'src/structural_analysis/main.py'
        if os.path.exists(hub_path):
            try:
                print("🚀 Launching structural analysis hub with auto-import...")
                subprocess.run([sys.executable, hub_path])
                
                # Check for new CSV files after analysis and auto-import if enabled
                if self.auto_import_enabled:
                    print("🔄 Checking for new CSV files to auto-import...")
                    self.scan_and_import_new_csv_files()
                    
            except Exception as e:
                print(f"Error launching structural hub: {e}")
        else:
            print(f"❌ {hub_path} not found")
    
    def run_structural_launcher(self):
        """Launch structural analyzers launcher with enhanced integration"""
        launcher_path = 'src/structural_analysis/gemini_launcher.py'
        if os.path.exists(launcher_path):
            try:
                print("🚀 Launching enhanced structural analyzers...")
                subprocess.run([sys.executable, launcher_path])
                
                # Auto-import after analysis
                if self.auto_import_enabled:
                    print("🔄 Auto-importing analysis results...")
                    self.scan_and_import_new_csv_files()
                    
            except Exception as e:
                print(f"Error launching structural launcher: {e}")
        else:
            print(f"❌ {launcher_path} not found")
    
    def scan_and_import_new_csv_files(self):
        """Scan for and import new CSV files from structural analysis"""
        try:
            # Check common output directories
            scan_dirs = [
                self.structural_data_dir,
                os.path.join(self.structural_data_dir, 'halogen'),
                os.path.join(self.structural_data_dir, 'laser'),
                os.path.join(self.structural_data_dir, 'uv'),
                'data/structural_data',
                '.'
            ]
            
            new_files_imported = 0
            
            for scan_dir in scan_dirs:
                if os.path.exists(scan_dir):
                    # Look for CSV files modified in last hour
                    current_time = time.time()
                    for file in os.listdir(scan_dir):
                        if file.endswith('.csv'):
                            file_path = os.path.join(scan_dir, file)
                            file_time = os.path.getmtime(file_path)
                            
                            # If file was modified in last hour, try to import
                            if current_time - file_time < 3600:  # 1 hour
                                success = self.auto_import_csv_to_database(file_path)
                                if success:
                                    new_files_imported += 1
            
            if new_files_imported > 0:
                print(f"✅ Auto-imported {new_files_imported} new CSV files")
                if self.bleep_enabled:
                    self.play_bleep(feature_type="completion")
            
        except Exception as e:
            print(f"Auto-import scan error: {e}")
    
    def run_raw_data_browser(self):
        """Run raw data browser if available"""
        if os.path.exists('raw_data_browser.py'):
            try:
                subprocess.run([sys.executable, 'raw_data_browser.py'])
            except Exception as e:
                print(f"Error launching raw data browser: {e}")
        else:
            print("❌ raw_data_browser.py not found")
    
    def run_analytical_workflow(self):
        """Run analytical workflow"""
        workflow_path = 'src/numerical_analysis/analytical_workflow.py'
        if os.path.exists(workflow_path):
            try:
                subprocess.run([sys.executable, workflow_path])
            except Exception as e:
                print(f"Error launching analytical workflow: {e}")
        else:
            print(f"❌ {workflow_path} not found")
    
    def show_database_stats(self):
        """Show enhanced database statistics"""
        print("\n📊 ENHANCED DATABASE STATISTICS")
        print("=" * 40)
        
        # Show reference spectra databases
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                try:
                    df = pd.read_csv(db_file)
                    if 'full_name' in df.columns:
                        unique_gems = df['full_name'].nunique()
                        print(f"✅ {db_file}:")
                        print(f"   Records: {len(df):,}")
                        print(f"   Unique gems: {unique_gems}")
                        
                        # Show sample intensity ranges
                        intensity_range = f"{df['intensity'].min():.3f} to {df['intensity'].max():.3f}"
                        print(f"   Intensity range: {intensity_range}")
                    else:
                        print(f"⚠️ {db_file}: {len(df):,} records (no gem names)")
                except Exception as e:
                    print(f"❌ {db_file}: Error reading - {e}")
            else:
                print(f"❌ {db_file}: Missing")
        
        # Enhanced structural database statistics
        self.show_enhanced_database_stats()
    
    def emergency_fix_files(self):
        """Run emergency fix if available"""
        if os.path.exists('emergency_fix.py'):
            try:
                subprocess.run([sys.executable, 'emergency_fix.py'])
            except Exception as e:
                print(f"Error running emergency fix: {e}")
        else:
            print("❌ emergency_fix.py not found")
    
    def database_management_menu(self):
        """Enhanced database management menu"""
        print("\n🗄️ DATABASE MANAGEMENT SYSTEM")
        print("=" * 40)
        
        db_menu_options = [
            ("📥 Batch Import Structural Data", self.batch_import_structural_data),
            ("📊 Show Enhanced Database Statistics", self.show_enhanced_database_stats),
            ("🔍 Enhanced Database Search", self.enhanced_database_search),
            ("📊 Enhanced Comparison Analysis", self.enhanced_comparison_analysis),
            ("📏 Relative Height Analysis", self.relative_height_analysis),
            ("🔄 Toggle Auto-Import", self.toggle_auto_import),
            ("🔊 Toggle Audio Bleep System", self.toggle_bleep_system),
            ("🗄️ Create Database Schema", self.create_database_schema),
            ("⬅️ Back to Main Menu", None)
        ]
        
        while True:
            print(f"\n🗄️ DATABASE MANAGEMENT:")
            print("-" * 30)
            
            for i, (description, _) in enumerate(db_menu_options, 1):
                print(f"{i:2}. {description}")
            
            try:
                choice = input(f"\nChoice (1-{len(db_menu_options)}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == len(db_menu_options) - 1:  # Back to main menu
                    break
                
                if 0 <= choice_idx < len(db_menu_options) - 1:
                    description, action = db_menu_options[choice_idx]
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
                print("\n\n⚠️ Returning to main menu...")
                break
            except Exception as e:
                print(f"\n❌ Database menu error: {e}")
    
    def enhanced_tools_menu(self):
        """Enhanced tools and analysis menu"""
        print("\n🔧 ENHANCED TOOLS & ANALYSIS")
        print("=" * 35)
        
        tools_menu_options = [
            ("📊 Enhanced Comparison Analysis", self.enhanced_comparison_analysis),
            ("📏 Relative Height Analysis", self.relative_height_analysis),
            ("🔍 Advanced Database Search", self.enhanced_database_search),
            ("📈 Spectral Visualization (Last Analysis)", lambda: self.create_spectral_comparison_plots(self.current_gem_identifier) if hasattr(self, 'current_gem_identifier') else print("❌ No recent analysis found")),
            ("🔊 Audio Bleep Test", lambda: self.test_audio_system()),
            ("💾 Export Analysis Cache", self.export_analysis_cache),
            ("🔄 Clear Analysis Cache", self.clear_analysis_cache),
            ("⬅️ Back to Main Menu", None)
        ]
        
        while True:
            print(f"\n🔧 ENHANCED TOOLS:")
            print("-" * 25)
            
            for i, (description, _) in enumerate(tools_menu_options, 1):
                print(f"{i:2}. {description}")
            
            # Show current status
            print(f"\n📊 Current Status:")
            print(f"   🔊 Audio: {'ON' if self.bleep_enabled else 'OFF'}")
            print(f"   🔄 Auto-import: {'ON' if self.auto_import_enabled else 'OFF'}")
            print(f"   📏 Cache entries: {len(self.relative_height_cache)}")
            
            try:
                choice = input(f"\nChoice (1-{len(tools_menu_options)}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == len(tools_menu_options) - 1:  # Back to main menu
                    break
                
                if 0 <= choice_idx < len(tools_menu_options) - 1:
                    description, action = tools_menu_options[choice_idx]
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
                print("\n\n⚠️ Returning to main menu...")
                break
            except Exception as e:
                print(f"\n❌ Enhanced tools error: {e}")
    
    def test_audio_system(self):
        """Test the audio bleep system"""
        print("🔊 TESTING AUDIO BLEEP SYSTEM")
        print("=" * 35)
        
        if not HAS_AUDIO:
            print("❌ Audio system not available")
            print("   Install winsound (Windows) or pygame for audio support")
            return
        
        print("🎵 Playing different bleep types...")
        
        bleep_types = [
            ("standard", "Standard bleep"),
            ("peak", "Peak detection"),
            ("valley", "Valley detection"), 
            ("plateau", "Plateau detection"),
            ("significant", "Significant match"),
            ("completion", "Analysis completion")
        ]
        
        for bleep_type, description in bleep_types:
            print(f"   🔊 {description}...")
            self.play_bleep(feature_type=bleep_type)
            time.sleep(0.5)  # Brief pause between bleeps
        
        print("✅ Audio test completed")
        
        # Toggle test
        current_state = self.bleep_enabled
        print(f"\nCurrent audio state: {'ENABLED' if current_state else 'DISABLED'}")
        toggle = input("Toggle audio state? (y/n): ").strip().lower()
        if toggle == 'y':
            self.toggle_bleep_system()
    
    def export_analysis_cache(self):
        """Export relative height analysis cache to CSV"""
        if not self.relative_height_cache:
            print("❌ No cache data to export")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"relative_height_cache_{timestamp}.csv"
        
        try:
            # Convert cache to DataFrame
            cache_data = []
            for key, measurements in self.relative_height_cache.items():
                gem_id, wavelength = key.rsplit('_', 1)
                wavelength = float(wavelength)
                
                for light_source, data in measurements.items():
                    cache_data.append({
                        'gem_id': gem_id,
                        'target_wavelength': wavelength,
                        'light_source': light_source,
                        'actual_wavelength': data['wavelength'],
                        'intensity': data['intensity'],
                        'relative_height': data.get('relative_height', 0),
                        'percentage': data.get('percentage', 0),
                        'feature_group': data['feature_group'],
                        'wavelength_diff': data['wavelength_diff']
                    })
            
            df = pd.DataFrame(cache_data)
            df.to_csv(filename, index=False)
            
            print(f"✅ Cache exported to {filename}")
            print(f"   Records: {len(cache_data)}")
            
        except Exception as e:
            print(f"❌ Export error: {e}")
    
    def clear_analysis_cache(self):
        """Clear the relative height analysis cache"""
        cache_size = len(self.relative_height_cache)
        
        if cache_size == 0:
            print("📭 Cache is already empty")
            return
        
        confirm = input(f"Clear {cache_size} cache entries? (y/n): ").strip().lower()
        if confirm == 'y':
            self.relative_height_cache.clear()
            print(f"✅ Cleared {cache_size} cache entries")
            if self.bleep_enabled:
                self.play_bleep(feature_type="completion")
        else:
            print("❌ Cache clear cancelled")
    
    def main_menu(self):
        """Enhanced main menu system with integrated database management"""
        
        menu_options = [
            ("🔬 Launch Structural Analysis Hub", self.run_structural_analysis_hub),
            ("🎯 Launch Structural Analyzers", self.run_structural_launcher),
            ("📊 Analytical Analysis Workflow", self.run_analytical_workflow),
            ("💎 Select Gem for Enhanced Analysis", self.select_and_analyze_gem),
            ("🗄️ Database Management System", self.database_management_menu),
            ("🔧 Enhanced Tools & Analysis", self.enhanced_tools_menu),
            ("📂 Browse Raw Data Files", self.run_raw_data_browser),
            ("🧮 Run Enhanced Numerical Analysis", self.run_numerical_analysis_fixed),
            ("📈 Show Database Statistics", self.show_database_stats),
            ("🔧 Emergency Fix", self.emergency_fix_files),
            ("❌ Exit", lambda: None)
        ]
        
        while True:
            print("\n" + "="*85)
            print("🔬 ENHANCED GEMINI GEMOLOGICAL ANALYSIS SYSTEM")
            print("   Complete Database Integration • Audio Feedback • Auto-Import")
            print("="*85)
            
            # Show enhanced system status
            system_ok = self.check_system_status()
            
            print(f"\n📋 ENHANCED MAIN MENU:")
            print("-" * 45)
            
            for i, (description, _) in enumerate(menu_options, 1):
                print(f"{i:2}. {description}")
            
            # Enhanced status display
            print(f"\n🔧 ENHANCED FEATURES STATUS:")
            print(f"   🔊 Audio Bleep: {'ON' if self.bleep_enabled else 'OFF'} ({'Available' if HAS_AUDIO else 'Not Available'})")
            print(f"   🔄 Auto-Import: {'ON' if self.auto_import_enabled else 'OFF'}")
            print(f"   📏 Relative Height Cache: {len(self.relative_height_cache)} entries")
            
            if os.path.exists(self.db_path):
                try:
                    conn = sqlite3.connect(self.db_path)
                    count = pd.read_sql_query("SELECT COUNT(*) as count FROM structural_features", conn).iloc[0]['count']
                    conn.close()
                    print(f"   🗄️ Structural Database: {count:,} features")
                except:
                    print(f"   🗄️ Structural Database: Error reading")
            else:
                print(f"   🗄️ Structural Database: Not found")
            
            # Get user choice
            try:
                choice = input(f"\nChoice (1-{len(menu_options)}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == len(menu_options) - 1:  # Exit
                    print("\n👋 Enhanced Gemini Analysis System - Goodbye!")
                    if self.bleep_enabled:
                        self.play_bleep(feature_type="completion")
                    break
                
                if 0 <= choice_idx < len(menu_options) - 1:
                    description, action = menu_options[choice_idx]
                    print(f"\n🚀 {description.upper()}")
                    print("-" * 55)
                    
                    if action:
                        action()
                    
                    input("\n⏎ Press Enter to return to main menu...")
                else:
                    print("❌ Invalid choice")
                    
            except ValueError:
                print("❌ Please enter a number")
            except KeyboardInterrupt:
                print("\n\n⚠️ Enhanced system interrupted - goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Enhanced menu error: {e}")

def main():
    """Enhanced main entry point with error handling"""
    try:
        print("🔬 Starting ENHANCED Gemini Gemological Analysis System...")
        print("   🔊 Audio feedback system")
        print("   🗄️ Complete database integration")
        print("   🔄 Automatic CSV import")
        print("   📏 Relative height measurements")
        print("   📊 Enhanced comparison tools")
        
        system = EnhancedGeminiAnalysisSystem()
        
        # Play startup bleep
        if system.bleep_enabled:
            system.play_bleep(feature_type="completion")
        
        system.main_menu()
        
    except KeyboardInterrupt:
        print("\n\nEnhanced system interrupted - goodbye!")
    except Exception as e:
        print(f"Enhanced system error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
