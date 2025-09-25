#!/usr/bin/env python3
"""
APPEND FULL NAME TO STRUCTURAL DB - UPDATED VERSION
Update structural database with enhanced gem name information
Works with SQLite database created by batch_importer.py
"""

import sqlite3
import pandas as pd
import os
import re
from pathlib import Path
from datetime import datetime

class StructuralDBUpdater:
    """Update structural database with enhanced gem name information"""
    
    def __init__(self):
        # Find project root and set paths
        self.project_root = self.find_project_root()
        self.db_path = self.project_root / "database" / "structural_spectra" / "multi_structural_gem_data.db"
        self.csv_path = self.project_root / "database" / "structural_spectra" / "gemini_structural_db.csv"
        
        print(f"🔧 Structural DB Updater Initialized")
        print(f"💾 Database: {self.db_path}")
        print(f"📄 CSV file: {self.csv_path}")
    
    def find_project_root(self) -> Path:
        """Find project root by looking for key directories"""
        current = Path(__file__).parent.absolute()
        
        for path in [current] + list(current.parents):
            if (path / "database").exists() and (path / "data").exists():
                return path
            if (path / "main.py").exists():
                return path
        
        return current.parent
    
    def parse_full_gem_name(self, filename: str) -> dict:
        """Parse filename to extract full gem information"""
        base_name = Path(filename).stem
        
        # Remove timestamp if present
        base_name = re.sub(r'_\d{8}_\d{6}$', '', base_name)
        
        # Parse gem name components
        pattern = r'^(.+?)([BLU])([CP])(\d+)$'
        match = re.match(pattern, base_name, re.IGNORECASE)
        
        if match:
            prefix, light, orientation, scan = match.groups()
            light_mapping = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
            
            return {
                'gem_id': prefix,
                'light_source': light_mapping.get(light.upper(), 'Unknown'),
                'light_code': light.upper(),
                'orientation': orientation.upper(),
                'scan_number': int(scan),
                'full_identifier': base_name,
                'base_gem_name': prefix,
                'complete_name': f"{prefix}_{light.upper()}{orientation.upper()}{scan}"
            }
        
        # Fallback for non-standard names
        return {
            'gem_id': base_name,
            'light_source': 'Unknown',
            'light_code': 'U',
            'orientation': 'C', 
            'scan_number': 1,
            'full_identifier': base_name,
            'base_gem_name': base_name,
            'complete_name': base_name
        }
    
    def update_database_schema(self):
        """Add additional columns to database if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check current schema
            cursor.execute("PRAGMA table_info(structural_features)")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Add missing columns
            new_columns = {
                'gem_id': 'TEXT',
                'light_code': 'TEXT', 
                'orientation': 'TEXT',
                'scan_number': 'INTEGER',
                'full_identifier': 'TEXT',
                'base_gem_name': 'TEXT',
                'complete_name': 'TEXT'
            }
            
            added_columns = []
            for col_name, col_type in new_columns.items():
                if col_name not in columns:
                    cursor.execute(f'ALTER TABLE structural_features ADD COLUMN {col_name} {col_type}')
                    added_columns.append(col_name)
            
            if added_columns:
                print(f"✅ Added columns: {', '.join(added_columns)}")
            else:
                print(f"✅ Database schema already up to date")
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Schema update error: {e}")
            return False
    
    def update_full_names(self):
        """Update all records with full name information"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all unique filenames
            cursor.execute("SELECT DISTINCT file FROM structural_features")
            files = [row[0] for row in cursor.fetchall()]
            
            print(f"📁 Updating full names for {len(files)} files...")
            
            updated_count = 0
            
            for filename in files:
                # Parse the full gem information
                gem_info = self.parse_full_gem_name(filename)
                
                # Update all records for this file
                cursor.execute('''
                    UPDATE structural_features 
                    SET gem_id = ?, light_code = ?, orientation = ?, scan_number = ?,
                        full_identifier = ?, base_gem_name = ?, complete_name = ?
                    WHERE file = ?
                ''', (
                    gem_info['gem_id'],
                    gem_info['light_code'],
                    gem_info['orientation'], 
                    gem_info['scan_number'],
                    gem_info['full_identifier'],
                    gem_info['base_gem_name'],
                    gem_info['complete_name'],
                    filename
                ))
                
                updated_files = cursor.rowcount
                updated_count += updated_files
                
                if updated_files > 0:
                    print(f"   📄 {filename} → {gem_info['complete_name']} ({updated_files} records)")
            
            conn.commit()
            conn.close()
            
            print(f"✅ Updated {updated_count} records with full name information")
            return True
            
        except Exception as e:
            print(f"❌ Update error: {e}")
            return False
    
    def update_csv_export(self):
        """Update the CSV export with new columns"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Export with all columns including new ones
            query = '''
                SELECT file, light_source, wavelength, intensity, feature_group, data_type,
                       start_wavelength, end_wavelength, midpoint, bottom,
                       normalization_scheme, reference_wavelength,
                       gem_id, light_code, orientation, scan_number, 
                       full_identifier, base_gem_name, complete_name, timestamp
                FROM structural_features
                ORDER BY base_gem_name, light_source, wavelength
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Save updated CSV
            df.to_csv(self.csv_path, index=False)
            print(f"✅ Updated CSV export: {self.csv_path} ({len(df)} records)")
            
            # Show sample of the data
            if len(df) > 0:
                print(f"\n📋 Sample data (first 3 records):")
                sample_cols = ['complete_name', 'light_source', 'wavelength', 'feature_group']
                available_cols = [col for col in sample_cols if col in df.columns]
                if available_cols:
                    print(df[available_cols].head(3).to_string(index=False))
            
            return True
            
        except Exception as e:
            print(f"❌ CSV update error: {e}")
            return False
    
    def show_statistics(self):
        """Display updated database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total = cursor.fetchone()[0]
            
            # Unique gems
            cursor.execute("SELECT COUNT(DISTINCT base_gem_name) FROM structural_features WHERE base_gem_name IS NOT NULL")
            unique_gems = cursor.fetchone()[0]
            
            # By light source
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source ORDER BY COUNT(*) DESC")
            by_light = cursor.fetchall()
            
            # Recent additions
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE date(timestamp) = date('now')")
            today_count = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"\n📊 UPDATED DATABASE STATISTICS:")
            print(f"   Total records: {total:,}")
            print(f"   Unique gems: {unique_gems}")
            print(f"   Records added today: {today_count:,}")
            print(f"   By light source:")
            for light, count in by_light:
                print(f"     {light}: {count:,}")
            
        except Exception as e:
            print(f"❌ Statistics error: {e}")
    
    def run_update_process(self):
        """Main update process"""
        print(f"\n🚀 STARTING FULL NAME UPDATE PROCESS")
        print("=" * 55)
        
        # Check if database exists
        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            print(f"💡 Run batch import first to create the database")
            return False
        
        # Update schema
        if not self.update_database_schema():
            return False
        
        # Update full names
        if not self.update_full_names():
            return False
        
        # Update CSV export
        if not self.update_csv_export():
            return False
        
        # Show statistics
        self.show_statistics()
        
        print(f"\n✅ FULL NAME UPDATE COMPLETE")
        return True

def main():
    """Main entry point"""
    print("STRUCTURAL DB FULL NAME UPDATER - UPDATED VERSION")
    print("=" * 65)
    
    updater = StructuralDBUpdater()
    success = updater.run_update_process()
    
    if success:
        print(f"\n🎉 Full name update completed successfully!")
        print(f"💡 Your structural database now has enhanced gem identification")
    else:
        print(f"\n❌ Full name update failed")
    
    return success

if __name__ == "__main__":
    main()
