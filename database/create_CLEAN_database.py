#!/usr/bin/env python3
"""
Quick script to create clean database with only intensity data
"""

import sqlite3
import pandas as pd
import os

def create_clean_database():
    """Create clean database from existing data"""
    
    # Check if original exists
    if not os.path.exists("multi_structural_gem_data.db"):
        print("❌ Original database not found!")
        return
    
    print("🔍 Reading original database...")
    
    # Read original data
    conn = sqlite3.connect("multi_structural_gem_data.db")
    df = pd.read_sql_query("SELECT * FROM structural_features", conn)
    conn.close()
    
    print(f"📊 Found {len(df)} total records")
    
    # Filter for records with valid intensity values
    clean_data = df[
        df['intensity'].notna() & 
        (df['intensity'] != '') & 
        (df['intensity'] != 0)
    ].copy()
    
    print(f"✅ {len(clean_data)} records have intensity values")
    print(f"❌ {len(df) - len(clean_data)} records missing intensities (will be excluded)")
    
    if len(clean_data) == 0:
        print("⚠️ No records with intensity found!")
        return
    
    # Create clean database
    clean_conn = sqlite3.connect("multi_structural_gem_data_CLEAN.db")
    
    # Create table structure
    cursor = clean_conn.cursor()
    cursor.execute("""
        CREATE TABLE structural_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stone_id TEXT NOT NULL,
            feature_type TEXT NOT NULL,
            wavelength_start REAL NOT NULL,
            wavelength_end REAL,
            intensity REAL NOT NULL,
            light_source TEXT NOT NULL,
            symmetry_ratio REAL,
            skew_description TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            file_source TEXT,
            UNIQUE(stone_id, feature_type, wavelength_start, wavelength_end)
        )
    """)
    
    # Insert clean data
    for _, row in clean_data.iterrows():
        try:
            cursor.execute("""
                INSERT INTO structural_features 
                (stone_id, feature_type, wavelength_start, wavelength_end, 
                 intensity, light_source, symmetry_ratio, skew_description, 
                 timestamp, file_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['stone_id'], row['feature_type'], row['wavelength_start'], 
                row['wavelength_end'], row['intensity'], row['light_source'],
                row['symmetry_ratio'], row['skew_description'], 
                row['timestamp'], row['file_source']
            ))
        except Exception as e:
            print(f"⚠️ Skipped record: {e}")
    
    clean_conn.commit()
    clean_conn.close()
    
    print(f"✅ Created: multi_structural_gem_data_CLEAN.db")
    print(f"📊 Contains {len(clean_data)} records with wavelength + intensity")
    
    # Show sample
    print(f"\n📋 Sample clean data:")
    for _, row in clean_data.head(3).iterrows():
        print(f"   {row['stone_id']} | {row['feature_type']} | {row['wavelength_start']}nm | {row['intensity']}")

if __name__ == "__main__":
    create_clean_database()