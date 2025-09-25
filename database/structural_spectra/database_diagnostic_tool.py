#!/usr/bin/env python3
"""
Database Diagnostic Tool
Check what's actually in the structural database
"""

import sqlite3
import pandas as pd
from pathlib import Path

def diagnose_database():
    """Diagnose the structural database"""
    
    # Database path
    db_path = Path("C:/Users/David/OneDrive/Desktop/gemini_gemological_analysis/database/structural_spectra/multi_structural_gem_data.db")
    
    print("🔍 DATABASE DIAGNOSTIC REPORT")
    print("=" * 50)
    print(f"Database path: {db_path}")
    print(f"Database exists: {db_path.exists()}")
    
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
            print(f"   Columns: {len(columns)}")
            for col in columns:
                print(f"     - {col[1]} ({col[2]})")
            
            # Get record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"   Records: {count}")
            
            if count > 0:
                # Show sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                print(f"   Sample data (first 3 rows):")
                for i, row in enumerate(sample_data):
                    print(f"     Row {i+1}: {row[:5]}...")  # First 5 columns
                
                # Check for specific gems
                identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
                for col_name in identifier_columns:
                    try:
                        cursor.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} LIKE '%51UC1%' OR {col_name} LIKE '%51UP1%' OR {col_name} LIKE '%197UC1%' LIMIT 5")
                        matches = cursor.fetchall()
                        if matches:
                            print(f"   Found test gems in column '{col_name}': {matches}")
                        break
                    except:
                        continue
            else:
                print("   ❌ TABLE IS EMPTY!")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

def check_import_file_structure():
    """Check the structure of files being imported"""
    
    data_dir = Path("C:/Users/David/OneDrive/Desktop/gemini_gemological_analysis/data/structural_data")
    print(f"\n🔍 IMPORT FILES ANALYSIS")
    print("=" * 50)
    print(f"Source directory: {data_dir}")
    
    if not data_dir.exists():
        print("❌ Source directory does not exist!")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    print(f"CSV files found: {len(csv_files)}")
    
    for file_path in csv_files:
        print(f"\n📄 FILE: {file_path.name}")
        try:
            df = pd.read_csv(file_path)
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Show sample data
            if len(df) > 0:
                print("   Sample data:")
                print(df.head(2).to_string())
            
            # Check for key columns
            key_columns = ['Peak_Number', 'Wavelength_nm', 'Intensity', 'Light_Source']
            found_columns = [col for col in key_columns if col in df.columns]
            missing_columns = [col for col in key_columns if col not in df.columns]
            
            if found_columns:
                print(f"   ✅ Found key columns: {found_columns}")
            if missing_columns:
                print(f"   ❌ Missing key columns: {missing_columns}")
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")

def main():
    """Run diagnostics"""
    diagnose_database()
    check_import_file_structure()
    
    print(f"\n🔧 RECOMMENDATIONS:")
    print("1. Check if database is completely empty")
    print("2. Verify file structure matches expected schema")  
    print("3. Check database import error logs")
    print("4. May need to recreate database schema")

if __name__ == "__main__":
    main()