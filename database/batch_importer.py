#!/usr/bin/env python3
"""
BATCH IMPORTER
Handles structural data import operations

Author: David  
Version: 2024.08.06
"""

import os
import pandas as pd
import sqlite3
from datetime import datetime
from typing import Dict, List
from config.settings import get_config

class StructuralDataImporter:
    """Handles batch import of structural analysis files"""
    
    def __init__(self, database):
        self.db = database
        self.config = get_config('system')
        self.structural_dir = self.config['structural_dir']
        print("📥 Structural data importer initialized")
    
    def scan_for_structural_files(self, directory: str = None) -> Dict[str, List[Dict]]:
        """Scan for structural analysis CSV files"""
        if directory is None:
            directory = self.structural_dir
            
        structural_files = {'B': [], 'L': [], 'U': []}
        
        if not os.path.exists(directory):
            print(f"⚠️ Directory not found: {directory}")
            directory = "."
        
        print(f"📁 Scanning directory: {directory}")
        print("🔬 Looking for B/L/U structural files...")
        
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.csv') and 'features' in filename.lower():
                    try:
                        stone_id = filename.split('_')[0]
                        file_path = os.path.join(directory, filename)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        # Detect light source
                        light_source = self._detect_light_source(stone_id)
                        
                        file_info = {
                            'filename': filename,
                            'stone_id': stone_id,
                            'full_path': file_path,
                            'modified': mod_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'file_size': os.path.getsize(file_path),
                            'light_source': light_source
                        }
                        
                        structural_files[light_source].append(file_info)
                        
                    except Exception as e:
                        print(f"⚠️ Error reading {filename}: {e}")
                        continue
        except Exception as e:
            print(f"⚠️ Error scanning directory: {e}")
        
        # Sort by modification time
        for light_source in structural_files:
            structural_files[light_source].sort(key=lambda x: x['modified'], reverse=True)
        
        total_files = sum(len(files) for files in structural_files.values())
        print(f"\n📊 STRUCTURAL FILES FOUND:")
        print(f"   🔬 B (Broadband): {len(structural_files['B'])} files")
        print(f"   🔬 L (Laser): {len(structural_files['L'])} files")
        print(f"   🔬 U (UV): {len(structural_files['U'])} files")
        print(f"   📊 Total: {total_files} files")
        
        return structural_files
    
    def _detect_light_source(self, stone_id: str) -> str:
        """Detect light source from stone ID"""
        light_source = 'B'  # Default to Broadband
        
        for char_pos, char in enumerate(stone_id):
            if char == 'L' and char_pos + 1 < len(stone_id) and stone_id[char_pos + 1] in ['C', 'P']:
                light_source = 'L'
                break
            elif char == 'U' and char_pos + 1 < len(stone_id) and stone_id[char_pos + 1] in ['C', 'P']:
                light_source = 'U'
                break
        
        return light_source
    
    def enhanced_batch_import_with_duplicate_removal(self, files_by_source: Dict[str, List[Dict]]) -> int:
        """Multi-spectral batch import with duplicate removal"""
        print("🔄 Starting MULTI-SPECTRAL batch import...")
        print("🔬 Processing B (Broadband), L (Laser), and U (UV) data")
        
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        
        total_success = 0
        total_errors = 0
        total_removed = 0
        
        for light_source, files in files_by_source.items():
            if not files:
                continue
                
            print(f"\n🔬 PROCESSING {light_source}-SOURCE ({len(files)} files)")
            print("=" * 50)
            
            source_success = 0
            source_errors = 0
            source_removed = 0
            
            for file_info in files:
                try:
                    stone_id = file_info['stone_id']
                    print(f"  📊 Processing: {stone_id}")
                    
                    # Remove existing entries
                    existing_check = pd.read_sql_query("""
                        SELECT spectral_id FROM spectral_data WHERE full_stone_id = ?
                    """, conn, params=[stone_id])
                    
                    if not existing_check.empty:
                        for _, old_entry in existing_check.iterrows():
                            cursor.execute("DELETE FROM structural_features WHERE spectral_id = ?", 
                                         (old_entry['spectral_id'],))
                        cursor.execute("DELETE FROM spectral_data WHERE full_stone_id = ?", (stone_id,))
                        source_removed += len(existing_check)
                    
                    # Load new data
                    features_df = pd.read_csv(file_info['full_path'])
                    features = self._process_features(features_df)
                    
                    if features:
                        # Parse stone ID components
                        orientation, scan_number, base_reference = self._parse_stone_id(stone_id)
                        file_date = datetime.fromtimestamp(os.path.getmtime(file_info['full_path']))
                        
                        # Insert spectral data
                        cursor.execute('''
                        INSERT INTO spectral_data 
                        (stone_reference, light_source, orientation, scan_number, 
                         full_stone_id, date_analyzed, analyst, spectrum_file)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (base_reference, light_source, orientation, scan_number,
                              stone_id, file_date.date(), "David", f"{stone_id}.txt"))
                        
                        spectral_id = cursor.lastrowid
                        
                        # Insert features
                        for feature in features:
                            cursor.execute('''
                            INSERT INTO structural_features 
                            (spectral_id, feature_type, start_wavelength, midpoint_wavelength,
                             end_wavelength, crest_wavelength, max_wavelength, bottom_wavelength)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (spectral_id, feature['feature_type'],
                                  feature.get('start_wavelength'), feature.get('midpoint_wavelength'),
                                  feature.get('end_wavelength'), feature.get('crest_wavelength'),
                                  feature.get('max_wavelength'), feature.get('bottom_wavelength')))
                        
                        source_success += 1
                        print(f"    ✅ Added {len(features)} features")
                    else:
                        source_errors += 1
                        print(f"    ❌ No valid features found")
                
                except Exception as e:
                    source_errors += 1
                    print(f"    ❌ Error: {e}")
            
            print(f"\n📊 {light_source}-SOURCE SUMMARY:")
            print(f"   ✅ Successfully imported: {source_success}")
            print(f"   🗑️ Old entries removed: {source_removed}")
            print(f"   ❌ Errors: {source_errors}")
            
            total_success += source_success
            total_errors += source_errors
            total_removed += source_removed
        
        conn.commit()
        conn.close()
        
        print(f"\n📊 BATCH IMPORT COMPLETED:")
        print("=" * 50)
        print(f"   ✅ Total imported: {total_success}")
        print(f"   🔄 Total removed: {total_removed}")
        print(f"   ❌ Total errors: {total_errors}")
        
        return total_success
    
    def _process_features(self, features_df: pd.DataFrame) -> List[Dict]:
        """Process features from DataFrame"""
        features = []
        for _, row in features_df.iterrows():
            feature = {
                'feature_type': row.get('Feature'),
                'start_wavelength': row.get('Start') if pd.notna(row.get('Start')) else None,
                'midpoint_wavelength': row.get('Midpoint') if pd.notna(row.get('Midpoint')) else None,
                'end_wavelength': row.get('End') if pd.notna(row.get('End')) else None,
                'crest_wavelength': row.get('Crest') if pd.notna(row.get('Crest')) else None,
                'max_wavelength': row.get('Max') if pd.notna(row.get('Max')) else None,
                'bottom_wavelength': row.get('Bottom') if pd.notna(row.get('Bottom')) else None
            }
            if feature['feature_type']:
                features.append(feature)
        return features
    
    def _parse_stone_id(self, stone_id: str) -> tuple:
        """Parse stone ID into components"""
        orientation = 'C'
        scan_number = 1
        base_reference = stone_id
        
        for i, char in enumerate(stone_id):
            if char in ['B', 'L', 'U']:
                if i + 1 < len(stone_id) and stone_id[i + 1] in ['C', 'P']:
                    orientation = stone_id[i + 1]
                if i + 2 < len(stone_id) and stone_id[i + 2:].isdigit():
                    scan_number = int(stone_id[i + 2:])
                base_reference = stone_id[:i]
                break
        
        return orientation, scan_number, base_reference
    
    def batch_import_menu(self):
        """Interactive batch import menu"""
        print("\n🚀 MULTI-SPECTRAL BATCH IMPORT")
        print("=" * 50)
        
        files_by_source = self.scan_for_structural_files()
        total_files = sum(len(files) for files in files_by_source.values())
        
        if total_files == 0:
            print("❌ No structural files found")
            print(f"   💡 Check directory: {self.structural_dir}")
            return
        
        print(f"\n📁 Ready to import {total_files} files:")
        for light_source, files in files_by_source.items():
            if files:
                print(f"   🔬 {light_source}-source: {len(files)} files")
        
        print(f"\n🔬 FEATURES:")
        print("   ✅ Automatic light source detection")
        print("   ✅ Duplicate removal")
        print("   ✅ Progress tracking")
        
        confirm = input(f"\n🔄 Import all {total_files} files? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            try:
                success_count = self.enhanced_batch_import_with_duplicate_removal(files_by_source)
                if success_count > 0:
                    print("\n✅ BATCH IMPORT SUCCESS!")
                    print("   💡 Use menu option 4 to verify results")
                else:
                    print("\n⚠️ No new data imported")
            except Exception as e:
                print(f"\n❌ Batch import error: {e}")
        else:
            print("❌ Import cancelled")
    
    def interactive_import(self):
        """Single file import menu"""
        print("\n📥 SINGLE FILE IMPORT")
        print("=" * 50)
        
        files_by_source = self.scan_for_structural_files()
        total_files = sum(len(files) for files in files_by_source.values())
        
        if total_files == 0:
            print("❌ No structural files found")
            return
        
        print(f"📁 Found {total_files} files")
        print("💡 Use menu option 2 for batch import")

if __name__ == "__main__":
    # Test importer
    from database.db_manager import MultiSpectralGemstoneDB
    db = MultiSpectralGemstoneDB()
    importer = StructuralDataImporter(db)
    files = importer.scan_for_structural_files()
    print(f"Found files: {sum(len(f) for f in files.values())}")