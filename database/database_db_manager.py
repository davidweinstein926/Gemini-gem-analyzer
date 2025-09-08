#!/usr/bin/env python3
"""
DATABASE MANAGER
Handles all database operations for the multi-spectral system

Author: David
Version: 2024.08.06
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List
from config.settings import get_config

class MultiSpectralGemstoneDB:
    """Database manager for multi-spectral gemstone analysis"""
    
    def __init__(self, db_path: str = None):
        self.config = get_config()
        self.db_path = db_path or self.config['db']['db_path']
        self.init_multi_spectral_database()
        print(f"🗄️ Database initialized: {self.db_path}")
    
    def file_exists(self, path: str) -> bool:
        """Check if file exists"""
        return os.path.exists(path)
    
    def init_multi_spectral_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Stone catalog table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stone_catalog (
            reference VARCHAR(20) PRIMARY KEY,
            species VARCHAR(50),
            variety VARCHAR(50),
            stone_type VARCHAR(20),
            shape VARCHAR(30),
            weight VARCHAR(20),
            color VARCHAR(50),
            diamond_type VARCHAR(20),
            nitrogen VARCHAR(30),
            hydrogen VARCHAR(30),
            platelets VARCHAR(30),
            fluorescence VARCHAR(100),
            treatment VARCHAR(100),
            origin VARCHAR(100),
            certification VARCHAR(100),
            manufacturer VARCHAR(50),
            dqs VARCHAR(50),
            notes TEXT,
            date_added DATE DEFAULT CURRENT_DATE
        )
        ''')
        
        # Spectral data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS spectral_data (
            spectral_id INTEGER PRIMARY KEY AUTOINCREMENT,
            stone_reference VARCHAR(20) NOT NULL,
            light_source CHAR(1) NOT NULL CHECK (light_source IN ('B', 'L', 'U')),
            orientation CHAR(1) CHECK (orientation IN ('C', 'P')),
            scan_number INTEGER DEFAULT 1,
            full_stone_id VARCHAR(30),
            date_analyzed DATE,
            analyst VARCHAR(50),
            spectrum_file VARCHAR(100),
            analysis_notes TEXT,
            FOREIGN KEY (stone_reference) REFERENCES stone_catalog (reference),
            UNIQUE(stone_reference, light_source, orientation, scan_number)
        )
        ''')
        
        # Structural features table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS structural_features (
            feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
            spectral_id INTEGER NOT NULL,
            feature_type VARCHAR(20) NOT NULL,
            start_wavelength DECIMAL(7,3),
            midpoint_wavelength DECIMAL(7,3),
            end_wavelength DECIMAL(7,3),
            crest_wavelength DECIMAL(7,3),
            max_wavelength DECIMAL(7,3),
            bottom_wavelength DECIMAL(7,3),
            fwhm DECIMAL(7,3),
            intensity DECIMAL(10,3),
            symmetry_ratio DECIMAL(6,3),
            skew_description VARCHAR(50),
            feature_notes TEXT,
            FOREIGN KEY (spectral_id) REFERENCES spectral_data (spectral_id)
        )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stone_type ON stone_catalog(stone_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON stone_catalog(species)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_light_source ON spectral_data(light_source)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feature_type ON structural_features(feature_type)')
        
        conn.commit()
        conn.close()
        print("✅ Database schema initialized")
    
    def import_stone_catalog(self, csv_file_path: str, light_source_hint: str = None) -> int:
        """Import stone catalog from CSV"""
        try:
            print(f"📥 Loading catalog: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            
            # Auto-detect light source from filename
            if light_source_hint is None:
                filename = os.path.basename(csv_file_path).upper()
                if '_B.CSV' in filename or 'BROADBAND' in filename:
                    light_source_hint = 'B'
                elif '_L.CSV' in filename or 'LASER' in filename:
                    light_source_hint = 'L'
                elif '_U.CSV' in filename or 'UV' in filename:
                    light_source_hint = 'U'
                else:
                    light_source_hint = 'B'
            
            conn = sqlite3.connect(self.db_path)
            imported_count = 0
            
            for idx, row in df.iterrows():
                try:
                    cursor = conn.cursor()
                    
                    # Extract reference
                    reference = None
                    for ref_col in ['Reference', 'reference', 'ID', 'Stone_ID']:
                        if ref_col in row and pd.notna(row[ref_col]):
                            reference = str(row[ref_col]).strip()
                            break
                    
                    if not reference:
                        reference = f'Stone_{imported_count + 1}'
                    
                    # Determine stone type
                    stone_type = 'Unknown'
                    for type_col in ['Nat./Syn.', 'stone_type', 'Type']:
                        if type_col in row and pd.notna(row[type_col]):
                            val = str(row[type_col]).strip().lower()
                            if val in ['natural', 'nat', 'nat.', 'n']:
                                stone_type = 'Natural'
                            elif val in ['synthetic', 'syn', 'syn.', 's']:
                                stone_type = 'Synthetic'
                            break
                    
                    cursor.execute('''
                    INSERT OR REPLACE INTO stone_catalog 
                    (reference, species, variety, stone_type, notes)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (
                        reference,
                        str(row.get('Spec.', 'Unknown')).strip() if pd.notna(row.get('Spec.')) else 'Unknown',
                        str(row.get('Var.', 'Unknown')).strip() if pd.notna(row.get('Var.')) else 'Unknown',
                        stone_type,
                        f"{light_source_hint}-source catalog entry"
                    ))
                    imported_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Error importing row {idx}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            print(f"✅ Imported {imported_count} stones ({light_source_hint}-source)")
            return imported_count
            
        except Exception as e:
            print(f"❌ Error importing catalog: {e}")
            return 0
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            stats = {}
            
            # Count entries
            stats['catalog_count'] = pd.read_sql_query("SELECT COUNT(*) as count FROM stone_catalog", conn).iloc[0]['count']
            stats['spectral_count'] = pd.read_sql_query("SELECT COUNT(*) as count FROM spectral_data", conn).iloc[0]['count']
            stats['features_count'] = pd.read_sql_query("SELECT COUNT(*) as count FROM structural_features", conn).iloc[0]['count']
            
            # Light source breakdown
            light_source_stats = pd.read_sql_query("""
                SELECT light_source, COUNT(*) as count 
                FROM spectral_data 
                GROUP BY light_source
            """, conn)
            
            stats['light_sources'] = {}
            for _, row in light_source_stats.iterrows():
                stats['light_sources'][row['light_source']] = row['count']
            
            return stats
            
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}
        finally:
            conn.close()
    
    def verify_database_contents(self):
        """Verify database contents and check for duplicates"""
        print("🔍 DATABASE VERIFICATION REPORT")
        print("=" * 50)
        
        stats = self.get_database_stats()
        
        print(f"📊 Stone catalog entries: {stats.get('catalog_count', 0)}")
        print(f"📊 Total spectral analyses: {stats.get('spectral_count', 0)}")
        print(f"📊 Total structural features: {stats.get('features_count', 0)}")
        
        if stats.get('light_sources'):
            print(f"\n🔬 LIGHT SOURCES:")
            light_names = {'B': 'Broadband', 'L': 'Laser', 'U': 'UV'}
            for source, count in stats['light_sources'].items():
                name = light_names.get(source, source)
                print(f"   {name} ({source}): {count} analyses")
        
        # Check for duplicates
        conn = sqlite3.connect(self.db_path)
        try:
            duplicates = pd.read_sql_query("""
                SELECT full_stone_id, COUNT(*) as count
                FROM spectral_data 
                GROUP BY full_stone_id 
                HAVING COUNT(*) > 1
                ORDER BY count DESC
            """, conn)
            
            if duplicates.empty:
                print("\n✅ NO DUPLICATES FOUND!")
            else:
                print(f"\n⚠️ FOUND {len(duplicates)} DUPLICATE STONE IDs:")
                for _, row in duplicates.iterrows():
                    print(f"   {row['full_stone_id']}: {row['count']} copies")
        finally:
            conn.close()
        
        print(f"\n✅ VERIFICATION COMPLETE")
    
    def view_raw_data(self):
        """View raw database data"""
        print("\n📊 RAW DATABASE DATA")
        print("=" * 60)
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Show sample data from each table
            tables = [
                ("STONE CATALOG", "stone_catalog"),
                ("SPECTRAL DATA", "spectral_data"), 
                ("STRUCTURAL FEATURES", "structural_features")
            ]
            
            for table_name, table in tables:
                print(f"\n📋 {table_name} (First 5 entries):")
                df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 5", conn)
                if not df.empty:
                    print(df.to_string(index=False))
                else:
                    print(f"   No {table_name.lower()} found")
                    
        except Exception as e:
            print(f"❌ Error reading database: {e}")
        finally:
            conn.close()
    
    def view_statistics(self):
        """View detailed database statistics"""
        print("\n📈 DATABASE STATISTICS")
        print("=" * 60)
        
        stats = self.get_database_stats()
        
        # Basic counts
        print("📊 BASIC COUNTS:")
        print(f"   Catalog entries: {stats.get('catalog_count', 0)}")
        print(f"   Spectral analyses: {stats.get('spectral_count', 0)}")
        print(f"   Structural features: {stats.get('features_count', 0)}")
        
        # Light source breakdown
        if stats.get('light_sources'):
            print(f"\n🔬 ANALYSES BY LIGHT SOURCE:")
            light_names = {'B': 'Broadband', 'L': 'Laser', 'U': 'UV'}
            for source, count in stats['light_sources'].items():
                name = light_names.get(source, source)
                print(f"   {name}: {count} analyses")
        
        # Additional statistics
        conn = sqlite3.connect(self.db_path)
        try:
            # Stone types
            stone_type_stats = pd.read_sql_query("""
                SELECT sc.stone_type, COUNT(DISTINCT sd.full_stone_id) as count
                FROM stone_catalog sc
                JOIN spectral_data sd ON sc.reference = sd.stone_reference
                GROUP BY sc.stone_type
                ORDER BY count DESC
            """, conn)
            
            if not stone_type_stats.empty:
                print(f"\n📊 STONES BY TYPE:")
                for _, row in stone_type_stats.iterrows():
                    print(f"   {row['stone_type']}: {row['count']} stones")
            
            # Feature types
            feature_stats = pd.read_sql_query("""
                SELECT feature_type, COUNT(*) as count
                FROM structural_features
                GROUP BY feature_type
                ORDER BY count DESC
            """, conn)
            
            if not feature_stats.empty:
                print(f"\n📊 STRUCTURAL FEATURES:")
                for _, row in feature_stats.iterrows():
                    print(f"   {row['feature_type']}: {row['count']} features")
                    
        except Exception as e:
            print(f"❌ Error generating statistics: {e}")
        finally:
            conn.close()
    
    def search_by_reference(self):
        """Search stones by reference number"""
        print("\n🔎 SEARCH STONE BY REFERENCE NUMBER")
        print("=" * 60)
        
        reference = input("Enter stone reference number: ").strip()
        
        if not reference:
            print("❌ Please enter a reference number")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Search catalog
            catalog_result = pd.read_sql_query("""
                SELECT * FROM stone_catalog 
                WHERE reference LIKE ?
            """, conn, params=[f"%{reference}%"])
            
            if not catalog_result.empty:
                print(f"\n📋 CATALOG MATCHES:")
                for _, row in catalog_result.iterrows():
                    print(f"   Reference: {row['reference']}")
                    print(f"   Species: {row['species']} {row['variety']}")
                    print(f"   Type: {row['stone_type']}")
                    print(f"   Notes: {row['notes']}")
                    print()
            
            # Search spectral data  
            spectral_result = pd.read_sql_query("""
                SELECT * FROM spectral_data 
                WHERE stone_reference LIKE ? OR full_stone_id LIKE ?
            """, conn, params=[f"%{reference}%", f"%{reference}%"])
            
            if not spectral_result.empty:
                print(f"📊 SPECTRAL ANALYSES:")
                for _, row in spectral_result.iterrows():
                    print(f"   ID: {row['full_stone_id']}")
                    print(f"   Light Source: {row['light_source']}")
                    print(f"   Date: {row['date_analyzed']}")
                    print()
            
            if catalog_result.empty and spectral_result.empty:
                print(f"❌ No stones found matching '{reference}'")
                
        except Exception as e:
            print(f"❌ Search error: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    # Test database operations
    db = MultiSpectralGemstoneDB()
    db.verify_database_contents()