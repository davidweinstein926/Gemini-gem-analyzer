#!/usr/bin/env python3
"""
Database Analysis Script
Analyzes SQLite .db and CSV files in database/structural_spectra directory
"""

import sqlite3
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime

def analyze_sqlite_database(db_path):
    """Analyze SQLite database structure and content"""
    print(f"\n{'='*60}")
    print(f"ANALYZING SQLITE DATABASE: {db_path}")
    print(f"{'='*60}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get database file size
        file_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables found: {len(tables)}")
        
        for table_name, in tables:
            print(f"\n--- TABLE: {table_name} ---")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print("Column structure:")
            for col in columns:
                col_id, col_name, col_type, not_null, default_val, primary_key = col
                pk_indicator = " (PRIMARY KEY)" if primary_key else ""
                null_indicator = " NOT NULL" if not_null else ""
                default_indicator = f" DEFAULT {default_val}" if default_val else ""
                print(f"  {col_name}: {col_type}{pk_indicator}{null_indicator}{default_indicator}")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            print(f"Total rows: {row_count:,}")
            
            # Get sample data (first 5 rows)
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
                sample_rows = cursor.fetchall()
                column_names = [description[0] for description in cursor.description]
                
                print("\nSample data (first 5 rows):")
                print(f"Columns: {', '.join(column_names)}")
                for i, row in enumerate(sample_rows, 1):
                    print(f"Row {i}: {row}")
            
            # Analyze specific structural data if this looks like the main table
            if 'feature' in table_name.lower() or 'structural' in table_name.lower():
                analyze_structural_table(cursor, table_name)
        
        conn.close()
        
    except Exception as e:
        print(f"Error analyzing SQLite database: {e}")

def analyze_structural_table(cursor, table_name):
    """Deep analysis of structural features table"""
    print(f"\n--- STRUCTURAL ANALYSIS: {table_name} ---")
    
    try:
        # Analyze light sources
        cursor.execute(f"SELECT light_source, COUNT(*) FROM {table_name} GROUP BY light_source ORDER BY COUNT(*) DESC;")
        light_sources = cursor.fetchall()
        print("Light source distribution:")
        for light, count in light_sources:
            print(f"  {light}: {count:,} records")
        
        # Analyze feature types/groups
        feature_columns = ['feature_type', 'feature_group', 'data_type', 'Feature']
        for col in feature_columns:
            try:
                cursor.execute(f"SELECT {col}, COUNT(*) FROM {table_name} GROUP BY {col} ORDER BY COUNT(*) DESC LIMIT 10;")
                features = cursor.fetchall()
                if features:
                    print(f"\nTop feature types ({col}):")
                    for feature, count in features:
                        print(f"  {feature}: {count:,}")
                    break
            except:
                continue
        
        # Analyze wavelength ranges
        wavelength_columns = ['wavelength', 'Wavelength', 'wavelength_nm', 'Wavelength_nm']
        for col in wavelength_columns:
            try:
                cursor.execute(f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM {table_name} WHERE {col} IS NOT NULL;")
                wl_stats = cursor.fetchone()
                if wl_stats and wl_stats[0] is not None:
                    print(f"\nWavelength range ({col}): {wl_stats[0]:.1f} - {wl_stats[1]:.1f} nm (avg: {wl_stats[2]:.1f})")
                    break
            except:
                continue
        
        # Analyze normalization schemes
        norm_columns = ['normalization_scheme', 'Normalization_Scheme']
        for col in norm_columns:
            try:
                cursor.execute(f"SELECT {col}, COUNT(*) FROM {table_name} GROUP BY {col} ORDER BY COUNT(*) DESC;")
                norms = cursor.fetchall()
                if norms:
                    print(f"\nNormalization schemes ({col}):")
                    for norm, count in norms:
                        print(f"  {norm}: {count:,}")
                    break
            except:
                continue
        
        # Check for unique gems
        file_columns = ['file', 'File', 'filename']
        for col in file_columns:
            try:
                cursor.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table_name};")
                unique_files = cursor.fetchone()[0]
                print(f"\nUnique files ({col}): {unique_files:,}")
                
                # Show some example filenames
                cursor.execute(f"SELECT DISTINCT {col} FROM {table_name} LIMIT 10;")
                sample_files = cursor.fetchall()
                print("Sample filenames:")
                for file, in sample_files:
                    print(f"  {file}")
                break
            except:
                continue
                
    except Exception as e:
        print(f"Error in structural analysis: {e}")

def analyze_csv_file(csv_path):
    """Analyze CSV file structure and content"""
    print(f"\n{'='*60}")
    print(f"ANALYZING CSV FILE: {csv_path}")
    print(f"{'='*60}")
    
    try:
        # Get file size
        file_size = os.path.getsize(csv_path) / (1024 * 1024)  # MB
        print(f"File size: {file_size:.2f} MB")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        
        # Show column info
        print("\nColumn information:")
        for i, col in enumerate(df.columns):
            dtype = df[col].dtype
            non_null = df[col].count()
            null_count = len(df) - non_null
            unique_vals = df[col].nunique()
            print(f"  {col}: {dtype} ({non_null:,} non-null, {null_count:,} null, {unique_vals:,} unique)")
        
        # Show sample data
        print(f"\nFirst 5 rows:")
        print(df.head().to_string())
        
        # Analyze structural data if applicable
        if any(col.lower() in ['feature', 'light_source', 'wavelength'] for col in df.columns):
            analyze_csv_structural_data(df)
            
    except Exception as e:
        print(f"Error analyzing CSV file: {e}")

def analyze_csv_structural_data(df):
    """Analyze structural data in CSV format"""
    print(f"\n--- CSV STRUCTURAL ANALYSIS ---")
    
    # Light source analysis
    light_cols = [col for col in df.columns if 'light' in col.lower()]
    for col in light_cols:
        if col in df.columns:
            print(f"\nLight source distribution ({col}):")
            light_counts = df[col].value_counts()
            for light, count in light_counts.items():
                print(f"  {light}: {count:,}")
            break
    
    # Feature type analysis
    feature_cols = [col for col in df.columns if 'feature' in col.lower()]
    for col in feature_cols:
        if col in df.columns:
            print(f"\nFeature type distribution ({col}):")
            feature_counts = df[col].value_counts().head(10)
            for feature, count in feature_counts.items():
                print(f"  {feature}: {count:,}")
            break
    
    # Wavelength analysis
    wl_cols = [col for col in df.columns if 'wavelength' in col.lower()]
    for col in wl_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            wl_stats = df[col].describe()
            print(f"\nWavelength statistics ({col}):")
            print(f"  Range: {wl_stats['min']:.1f} - {wl_stats['max']:.1f} nm")
            print(f"  Mean: {wl_stats['mean']:.1f} nm")
            print(f"  Std: {wl_stats['std']:.1f} nm")
            break
    
    # File analysis
    file_cols = [col for col in df.columns if 'file' in col.lower()]
    for col in file_cols:
        if col in df.columns:
            unique_files = df[col].nunique()
            print(f"\nUnique files ({col}): {unique_files:,}")
            print("Sample filenames:")
            for filename in df[col].unique()[:10]:
                print(f"  {filename}")
            break

def compare_databases(db_files, csv_files):
    """Compare different database formats"""
    print(f"\n{'='*60}")
    print("DATABASE COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"SQLite databases found: {len(db_files)}")
    for db_file in db_files:
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
            table_count = cursor.fetchone()[0]
            
            # Try to get total record count from main table
            total_records = 0
            for table_name in ['structural_features', 'features', 'data']:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                    total_records = cursor.fetchone()[0]
                    break
                except:
                    continue
            
            file_size = os.path.getsize(db_file) / (1024 * 1024)
            print(f"  {os.path.basename(db_file)}: {table_count} tables, ~{total_records:,} records, {file_size:.2f} MB")
            conn.close()
        except Exception as e:
            print(f"  {os.path.basename(db_file)}: Error - {e}")
    
    print(f"\nCSV files found: {len(csv_files)}")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            file_size = os.path.getsize(csv_file) / (1024 * 1024)
            print(f"  {os.path.basename(csv_file)}: {len(df):,} records, {len(df.columns)} columns, {file_size:.2f} MB")
        except Exception as e:
            print(f"  {os.path.basename(csv_file)}: Error - {e}")

def main():
    """Main analysis function"""
    print("DATABASE ANALYSIS SCRIPT")
    print(f"Analysis started: {datetime.now()}")
    
    # Set the directory path
    db_directory = Path("database/structural_spectra")
    
    if not db_directory.exists():
        print(f"ERROR: Directory {db_directory} not found!")
        print("Please run this script from the project root directory.")
        return
    
    # Find all database files
    db_files = list(db_directory.glob("*.db"))
    csv_files = list(db_directory.glob("*.csv"))
    
    print(f"\nScanning directory: {db_directory}")
    print(f"Found {len(db_files)} SQLite databases and {len(csv_files)} CSV files")
    
    # Analyze each SQLite database
    for db_file in db_files:
        analyze_sqlite_database(db_file)
    
    # Analyze each CSV file
    for csv_file in csv_files:
        analyze_csv_file(csv_file)
    
    # Comparison summary
    if db_files or csv_files:
        compare_databases(db_files, csv_files)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"Finished: {datetime.now()}")
    print(f"{'='*60}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    if len(db_files) > 1:
        print("- Multiple SQLite databases found - determine which is current")
    if len(csv_files) > 1:
        print("- Multiple CSV files found - determine which is current")
    if db_files and csv_files:
        print("- Both SQLite and CSV formats present - choose primary format")
    print("- Analyze schema compatibility for bleep feature integration")
    print("- Plan absolute vs relative height measurement fields")

if __name__ == "__main__":
    main()