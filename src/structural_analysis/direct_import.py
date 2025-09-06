#!/usr/bin/env python3
"""
Direct CSV Import Script - Bypass menu issues
Run this directly to import your structural data
"""

import sqlite3
import pandas as pd
from pathlib import Path
import os

def create_database():
    """Create the structural features database"""
    db_path = "multi_structural_gem_data.db"
    
    # Backup existing database
    if os.path.exists(db_path):
        backup_name = f"multi_structural_gem_data_BACKUP_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.db"
        os.rename(db_path, backup_name)
        print(f"✅ Backed up old database to: {backup_name}")
    
    # Create new database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE structural_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feature TEXT NOT NULL,
            file TEXT NOT NULL,
            light_source TEXT NOT NULL,
            wavelength REAL NOT NULL,
            intensity REAL NOT NULL,
            point_type TEXT NOT NULL,
            feature_group TEXT NOT NULL,
            processing TEXT,
            baseline_used REAL,
            norm_factor REAL,
            snr REAL,
            symmetry_ratio REAL,
            skew_description TEXT,
            width_nm REAL,
            timestamp TEXT DEFAULT (datetime('now')),
            file_source TEXT
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX idx_file ON structural_features(file)")
    cursor.execute("CREATE INDEX idx_light_source ON structural_features(light_source)")
    cursor.execute("CREATE INDEX idx_feature_group ON structural_features(feature_group)")
    
    conn.commit()
    conn.close()
    print(f"✅ Created database: {db_path}")
    return db_path

def import_csv_files():
    """Import all CSV files from light source folders"""
    
    # Create database
    db_path = create_database()
    
    # Define folder paths
    base_path = Path(r"c:\users\david\gemini sp10 structural data")
    folders = ['halogen', 'laser', 'uv']
    
    print(f"\n🔍 Scanning folders in: {base_path}")
    
    all_files = []
    for folder in folders:
        folder_path = base_path / folder
        if folder_path.exists():
            csv_files = list(folder_path.glob("*_structural_*.csv"))
            all_files.extend(csv_files)
            print(f"📁 {folder}/: Found {len(csv_files)} files")
        else:
            print(f"❌ {folder}/: Directory not found")
    
    if not all_files:
        print("❌ No structural CSV files found!")
        return
    
    print(f"\n🚀 IMPORTING {len(all_files)} FILES...")
    
    conn = sqlite3.connect(db_path)
    total_imported = 0
    successful_files = 0
    
    for file_path in all_files:
        try:
            print(f"📥 Processing: {file_path.name}")
            
            # Read CSV
            df = pd.read_csv(file_path)
            file_records = 0
            
            # Import each row
            for _, row in df.iterrows():
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO structural_features 
                        (feature, file, light_source, wavelength, intensity, point_type, 
                         feature_group, processing, baseline_used, norm_factor, snr,
                         symmetry_ratio, skew_description, width_nm, file_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('Feature', ''),
                        row.get('File', ''),
                        row.get('Light_Source', ''),
                        float(row.get('Wavelength', 0)),
                        float(row.get('Intensity', 0)),
                        row.get('Point_Type', ''),
                        row.get('Feature_Group', ''),
                        row.get('Processing', ''),
                        float(row.get('Baseline_Used', 0)) if pd.notna(row.get('Baseline_Used')) else None,
                        float(row.get('Norm_Factor', 0)) if pd.notna(row.get('Norm_Factor')) else None,
                        float(row.get('SNR', 0)) if pd.notna(row.get('SNR')) else None,
                        float(row.get('Symmetry_Ratio', 0)) if pd.notna(row.get('Symmetry_Ratio')) else None,
                        row.get('Skew_Description', ''),
                        float(row.get('Width_nm', 0)) if pd.notna(row.get('Width_nm')) else None,
                        str(file_path)
                    ))
                    file_records += 1
                except Exception as e:
                    print(f"      ⚠️ Skipped record: {e}")
            
            total_imported += file_records
            successful_files += 1
            print(f"      ✅ Imported {file_records} records")
            
        except Exception as e:
            print(f"      ❌ Failed to process {file_path.name}: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"\n📊 IMPORT COMPLETE!")
    print(f"   ✅ Files processed: {successful_files}/{len(all_files)}")
    print(f"   📁 Total records: {total_imported}")
    print(f"   🗄️ Database: {db_path}")
    
    return total_imported

def show_database_stats():
    """Show what was imported"""
    db_path = "multi_structural_gem_data.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Basic stats
    cursor.execute("SELECT COUNT(*) FROM structural_features")
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features")
    files = cursor.fetchone()[0]
    
    cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source")
    by_light = cursor.fetchall()
    
    cursor.execute("SELECT feature_group, COUNT(*) FROM structural_features GROUP BY feature_group ORDER BY COUNT(*) DESC LIMIT 10")
    by_feature = cursor.fetchall()
    
    conn.close()
    
    print(f"\n📊 DATABASE STATISTICS:")
    print(f"   📁 Total records: {total:,}")
    print(f"   📁 Unique files: {files:,}")
    
    print(f"\n   💡 By Light Source:")
    for light, count in by_light:
        icon = {'Halogen': '🔥', 'Laser': '⚡', 'UV': '🟣'}.get(light, '💡')
        print(f"      {icon} {light}: {count:,}")
    
    print(f"\n   🏷️ Top Feature Types:")
    for feature, count in by_feature:
        print(f"      {feature}: {count:,}")

def main():
    """Main execution"""
    print("🚀 DIRECT CSV IMPORT SCRIPT")
    print("=" * 50)
    
    # Import all CSV files
    total_imported = import_csv_files()
    
    if total_imported > 0:
        # Show results
        show_database_stats()
        print(f"\n✅ SUCCESS! Your structural data is now in the database.")
        print(f"   You can now use the main menu Option 6 to see full statistics.")
    else:
        print(f"\n❌ No data was imported. Check folder paths and CSV files.")

if __name__ == "__main__":
    main()
