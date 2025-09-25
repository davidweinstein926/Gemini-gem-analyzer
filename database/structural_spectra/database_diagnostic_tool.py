#!/usr/bin/env python3
"""
Database Diagnostic Tool - UPDATED FOR NEW DATABASE STRUCTURE
Check what's actually in the structural database - supports both new and legacy formats
Compatible with: database/structural_spectra/gemini_structural.db and legacy databases
"""

import sqlite3
import pandas as pd
from pathlib import Path
import os

def find_database_files():
    """Find all available database files"""
    current_path = Path.cwd()
    
    # Database locations to check
    db_locations = [
        current_path / "database" / "structural_spectra",
        current_path,
        current_path / "database",
        current_path.parent / "database" / "structural_spectra",
        # Absolute path as fallback (from original)
        Path("C:/Users/David/OneDrive/Desktop/gemini_gemological_analysis/database/structural_spectra")
    ]
    
    found_databases = []
    
    for db_path in db_locations:
        if not db_path.exists():
            continue
        
        # Check for NEW database files
        new_sqlite = db_path / "gemini_structural.db"
        new_csv = db_path / "gemini_structural_unified.csv"
        
        # Check for LEGACY database files  
        legacy_sqlite = db_path / "multi_structural_gem_data.db"
        legacy_csv = db_path / "gemini_structural_db.csv"
        
        if new_sqlite.exists():
            found_databases.append(("sqlite", new_sqlite, "NEW"))
        if new_csv.exists():
            found_databases.append(("csv", new_csv, "NEW"))
        if legacy_sqlite.exists():
            found_databases.append(("sqlite", legacy_sqlite, "LEGACY"))
        if legacy_csv.exists():
            found_databases.append(("csv", legacy_csv, "LEGACY"))
    
    return found_databases

def diagnose_sqlite_database(db_path, version):
    """Diagnose SQLite database"""
    print(f"\n🗄️  SQLITE DATABASE ANALYSIS ({version})")
    print("=" * 60)
    print(f"Database path: {db_path}")
    print(f"Database exists: {db_path.exists()}")
    print(f"Database size: {db_path.stat().st_size / 1024:.1f} KB")
    
    if not db_path.exists():
        print("❌ Database file does not exist!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"\n📊 Tables found: {len(tables)}")
        
        if not tables:
            print("❌ No tables found in database!")
            conn.close()
            return
        
        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            print(f"\n🔍 TABLE: {table_name}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print(f"   📋 Columns ({len(columns)}):")
            
            # Show columns in organized way
            for col in columns:
                col_name, col_type, not_null, default, pk = col[1], col[2], col[3], col[4], col[5]
                nullable = "NOT NULL" if not_null else "nullable"
                primary_key = " [PK]" if pk else ""
                print(f"     • {col_name} ({col_type}) {nullable}{primary_key}")
            
            # Get record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   📈 Records: {count:,}")
            
            if count > 0:
                # NEW: Enhanced analysis for new schema
                if version == "NEW" and table_name == "structural_features":
                    analyze_new_structural_features_table(cursor, table_name)
                else:
                    analyze_legacy_table(cursor, table_name)
            else:
                print("   ❌ TABLE IS EMPTY!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        import traceback
        traceback.print_exc()

def analyze_new_structural_features_table(cursor, table_name):
    """Analyze the new structural_features table"""
    print(f"   🔬 NEW SCHEMA ANALYSIS:")
    
    try:
        # Light source distribution
        cursor.execute(f"SELECT light_source, COUNT(*) FROM {table_name} GROUP BY light_source ORDER BY COUNT(*) DESC")
        light_sources = cursor.fetchall()
        if light_sources:
            print(f"   💡 Light source distribution:")
            total_records = sum(count for _, count in light_sources)
            for source, count in light_sources:
                percentage = (count / total_records) * 100
                print(f"     • {source}: {count:,} ({percentage:.1f}%)")
        
        # Unique gems
        cursor.execute(f"SELECT COUNT(DISTINCT gem_id) FROM {table_name}")
        unique_gems = cursor.fetchone()[0]
        print(f"   💎 Unique gems: {unique_gems}")
        
        # Wavelength range
        cursor.execute(f"SELECT MIN(wavelength), MAX(wavelength), AVG(wavelength) FROM {table_name} WHERE wavelength IS NOT NULL")
        min_wl, max_wl, avg_wl = cursor.fetchone()
        if min_wl and max_wl:
            print(f"   🌈 Wavelength range: {min_wl:.1f} - {max_wl:.1f} nm (avg: {avg_wl:.1f})")
        
        # Feature groups
        cursor.execute(f"SELECT feature_group, COUNT(*) FROM {table_name} GROUP BY feature_group ORDER BY COUNT(*) DESC LIMIT 5")
        feature_groups = cursor.fetchall()
        if feature_groups:
            print(f"   🏷️  Top feature groups:")
            for group, count in feature_groups:
                print(f"     • {group}: {count:,}")
        
        # Recent imports
        cursor.execute(f"SELECT DATE(import_timestamp), COUNT(*) FROM {table_name} WHERE import_timestamp IS NOT NULL GROUP BY DATE(import_timestamp) ORDER BY DATE(import_timestamp) DESC LIMIT 3")
        recent_imports = cursor.fetchall()
        if recent_imports:
            print(f"   📅 Recent import activity:")
            for date, count in recent_imports:
                print(f"     • {date}: {count:,} records")
        
        # Sample data from each light source
        print(f"   📄 Sample data by light source:")
        for light_source in ['Halogen', 'Laser', 'UV']:
            cursor.execute(f"SELECT gem_id, wavelength, intensity FROM {table_name} WHERE light_source = ? LIMIT 2", (light_source,))
            samples = cursor.fetchall()
            if samples:
                print(f"     • {light_source} samples:")
                for gem_id, wavelength, intensity in samples:
                    print(f"       - Gem {gem_id}: {wavelength}nm, {intensity:.2f}")
        
    except Exception as e:
        print(f"     ❌ Analysis error: {e}")

def analyze_legacy_table(cursor, table_name):
    """Analyze legacy table structure"""
    print(f"   🔍 LEGACY ANALYSIS:")
    
    try:
        # Show sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
        sample_data = cursor.fetchall()
        print(f"   📄 Sample data (first 3 rows):")
        for i, row in enumerate(sample_data):
            print(f"     Row {i+1}: {str(row)[:100]}...")  # First 100 chars
        
        # Check for specific identifier columns
        identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
        for col_name in identifier_columns:
            try:
                cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} LIKE '%51%' OR {col_name} LIKE '%197%' LIMIT 5")
                matches = cursor.fetchall()
                if matches:
                    print(f"   🔍 Sample gems in column '{col_name}': {matches}")
                break
            except:
                continue
                
    except Exception as e:
        print(f"     ❌ Legacy analysis error: {e}")

def diagnose_csv_database(csv_path, version):
    """Diagnose CSV database"""
    print(f"\n📄 CSV DATABASE ANALYSIS ({version})")
    print("=" * 60)
    print(f"CSV path: {csv_path}")
    print(f"CSV exists: {csv_path.exists()}")
    
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print(f"CSV size: {size_mb:.2f} MB")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"📊 Rows: {len(df):,}")
        print(f"📊 Columns: {len(df.columns)}")
        print(f"📋 Column names: {df.columns.tolist()}")
        
        if len(df) > 0:
            # NEW: Enhanced CSV analysis
            if version == "NEW":
                analyze_new_csv_structure(df)
            else:
                analyze_legacy_csv_structure(df)
        else:
            print("❌ CSV IS EMPTY!")
            
    except Exception as e:
        print(f"❌ CSV error: {e}")

def analyze_new_csv_structure(df):
    """Analyze new CSV structure"""
    print(f"🔬 NEW CSV SCHEMA ANALYSIS:")
    
    # Light source distribution
    if 'light_source' in df.columns:
        light_dist = df['light_source'].value_counts()
        print(f"💡 Light source distribution:")
        for source, count in light_dist.items():
            percentage = (count / len(df)) * 100
            print(f"   • {source}: {count:,} ({percentage:.1f}%)")
    
    # Unique gems
    gem_col = 'gem_id' if 'gem_id' in df.columns else 'file_source'
    if gem_col in df.columns:
        unique_gems = df[gem_col].nunique()
        print(f"💎 Unique gems: {unique_gems}")
    
    # Wavelength analysis
    if 'wavelength' in df.columns:
        wl_stats = df['wavelength'].describe()
        print(f"🌈 Wavelength statistics:")
        print(f"   Range: {wl_stats['min']:.1f} - {wl_stats['max']:.1f} nm")
        print(f"   Average: {wl_stats['mean']:.1f} nm")
    
    # Feature groups
    if 'feature_group' in df.columns:
        feature_dist = df['feature_group'].value_counts().head(5)
        print(f"🏷️  Top feature groups:")
        for group, count in feature_dist.items():
            print(f"   • {group}: {count:,}")
    
    # Sample data
    print(f"📄 Sample data (first 2 rows):")
    sample_cols = ['gem_id', 'light_source', 'wavelength', 'intensity', 'feature']
    available_cols = [col for col in sample_cols if col in df.columns]
    if available_cols:
        print(df[available_cols].head(2).to_string())

def analyze_legacy_csv_structure(df):
    """Analyze legacy CSV structure"""
    print(f"🔍 LEGACY CSV ANALYSIS:")
    
    # Show sample data
    print(f"📄 Sample data (first 2 rows):")
    print(df.head(2).to_string())
    
    # Check for key columns
    key_columns = ['file', 'filename', 'light_source', 'wavelength', 'intensity']
    found_columns = [col for col in key_columns if col in df.columns]
    missing_columns = [col for col in key_columns if col not in df.columns]
    
    if found_columns:
        print(f"✅ Found key columns: {found_columns}")
    if missing_columns:
        print(f"❌ Missing key columns: {missing_columns}")

def check_import_file_structure():
    """Check the structure of files being imported - UPDATED"""
    
    # Check both current and legacy locations
    data_dirs = [
        Path.cwd() / "data" / "structural_data",
        Path("C:/Users/David/OneDrive/Desktop/gemini_gemological_analysis/data/structural_data"),
        Path.cwd() / "data" / "structural(archive)"
    ]
    
    print(f"\n📥 IMPORT FILES ANALYSIS")
    print("=" * 60)
    
    for data_dir in data_dirs:
        if data_dir.exists():
            print(f"\n📁 Checking: {data_dir}")
            csv_files = list(data_dir.glob("*.csv"))
            print(f"   CSV files found: {len(csv_files)}")
            
            # Show file details
            for file_path in csv_files[:5]:  # Limit to first 5 files
                print(f"\n   📄 FILE: {file_path.name}")
                try:
                    df = pd.read_csv(file_path)
                    print(f"      Rows: {len(df)}")
                    print(f"      Columns ({len(df.columns)}): {df.columns.tolist()}")
                    
                    # Check file format
                    if 'Peak_Number' in df.columns and 'Wavelength_nm' in df.columns:
                        file_type = "UV Auto Detection (11 columns)"
                    elif 'Symmetry_Ratio' in df.columns and 'Skew_Description' in df.columns:
                        file_type = "Halogen Structural (23 columns)"
                    elif 'Feature' in df.columns and 'Point_Type' in df.columns and 'SNR' in df.columns:
                        file_type = "Laser Structural (20 columns)"
                    else:
                        file_type = "Unknown format"
                    
                    print(f"      Format: {file_type}")
                    
                    # Show key fields if available
                    if len(df) > 0:
                        key_fields = ['Light_Source', 'Feature', 'Wavelength', 'Wavelength_nm', 'Intensity']
                        available_keys = [field for field in key_fields if field in df.columns]
                        if available_keys:
                            sample = df[available_keys].iloc[0]
                            print(f"      Sample: {dict(sample)}")
                    
                except Exception as e:
                    print(f"      ❌ Error reading file: {e}")
            
            if len(csv_files) > 5:
                print(f"   ... and {len(csv_files) - 5} more files")

def main():
    """Run comprehensive diagnostics"""
    print("🔍 COMPREHENSIVE DATABASE DIAGNOSTIC TOOL")
    print("=" * 70)
    print("Checking both NEW and LEGACY database formats...")
    
    # Find all available databases
    databases = find_database_files()
    
    if not databases:
        print("❌ NO DATABASE FILES FOUND!")
        print("\nExpected locations:")
        print("• database/structural_spectra/gemini_structural.db (NEW)")
        print("• database/structural_spectra/gemini_structural_unified.csv (NEW)")
        print("• database/structural_spectra/multi_structural_gem_data.db (LEGACY)")
        print("• gemini_structural_db.csv (LEGACY)")
    else:
        print(f"✅ Found {len(databases)} database files:")
        for db_type, db_path, version in databases:
            print(f"   • {db_type.upper()} - {db_path.name} ({version})")
        
        # Analyze each database
        for db_type, db_path, version in databases:
            if db_type == "sqlite":
                diagnose_sqlite_database(db_path, version)
            else:
                diagnose_csv_database(db_path, version)
    
    # Check import files
    check_import_file_structure()
    
    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    print("=" * 40)
    
    # Count new vs legacy databases
    new_dbs = [db for db in databases if db[2] == "NEW"]
    legacy_dbs = [db for db in databases if db[2] == "LEGACY"]
    
    if new_dbs:
        print("✅ NEW database format detected - system up to date")
        print("• Use Option 6 for importing fresh structural data")
        print("• New format supports all light sources unified")
    
    if legacy_dbs and not new_dbs:
        print("⚠️  LEGACY database format only - consider updating:")
        print("• Use Option 6 to create new unified database")
        print("• New format provides better performance and structure")
    
    if not databases:
        print("❌ No databases found:")
        print("1. Run Option 2 to mark structural features")
        print("2. Run Option 6 to import and create database")
        print("3. Check file permissions and paths")
    
    empty_dbs = []
    for db_type, db_path, version in databases:
        try:
            if db_type == "sqlite":
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                if tables:
                    table_name = tables[0][0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    if count == 0:
                        empty_dbs.append(db_path.name)
                conn.close()
            else:
                df = pd.read_csv(db_path)
                if len(df) == 0:
                    empty_dbs.append(db_path.name)
        except:
            pass
    
    if empty_dbs:
        print(f"⚠️  Empty databases detected: {empty_dbs}")
        print("• Check import process for errors")
        print("• Verify source files in data/structural_data")
    
    print(f"\n🎯 SYSTEM STATUS: {'HEALTHY' if databases and not empty_dbs else 'NEEDS ATTENTION'}")

if __name__ == "__main__":
    main()