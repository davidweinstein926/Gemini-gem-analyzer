#!/usr/bin/env python3
"""
PRODUCTION BATCH IMPORTER - FORENSIC GEMOLOGICAL EDITION
Complete working version with all features:
- Forensic duplicate handling with temporal analysis preservation
- Proper column mapping for database schema
- Automatic archiving for Option 8 testing
- Enhanced error handling and reporting
"""

import sqlite3
import pandas as pd
import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class ProductionBatchImporter:
    """Production-grade structural data importer with forensic gemological features"""
    
    def __init__(self):
        self.project_root = self.find_project_root()
        self.structural_data_dir = self.project_root / "data" / "structural_data"
        self.archive_dir = self.project_root / "data" / "structural(archive)"
        self.db_path = self.project_root / "database" / "structural_spectra" / "multi_structural_gem_data.db"
        self.csv_output_path = self.project_root / "database" / "structural_spectra" / "gemini_structural_db.csv"
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        print("Production Forensic Gemological Batch Importer")
        print(f"Data source: {self.structural_data_dir}")
        print(f"Archive destination: {self.archive_dir}")
        print(f"Database: {self.db_path}")
        print(f"CSV output: {self.csv_output_path}")
        print("Forensic mode: Preserves temporal analysis for treatment detection")
    
    def find_project_root(self) -> Path:
        """Find project root by looking for key directories"""
        current = Path(__file__).parent.absolute()
        
        for path in [current] + list(current.parents):
            if (path / "database").exists() and (path / "data").exists():
                return path
            if (path / "main.py").exists():
                return path
        
        return current.parent
    
    def create_database_schema(self):
        """Create or verify database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS structural_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file TEXT NOT NULL,
                    light_source TEXT NOT NULL,
                    wavelength REAL NOT NULL,
                    intensity REAL NOT NULL,
                    feature TEXT NOT NULL DEFAULT 'unknown',
                    feature_group TEXT NOT NULL DEFAULT 'unknown',
                    point_type TEXT NOT NULL DEFAULT 'unknown',
                    data_type TEXT,
                    start_wavelength REAL,
                    end_wavelength REAL,
                    midpoint REAL,
                    bottom REAL,
                    normalization_scheme TEXT,
                    reference_wavelength REAL,
                    timestamp TEXT DEFAULT (datetime('now')),
                    UNIQUE(file, light_source, wavelength, feature)
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_features_lookup 
                ON structural_features(light_source, file, feature_group)
            ''')
            
            conn.commit()
            conn.close()
            print("Database schema verified")
            return True
            
        except Exception as e:
            print(f"Database schema error: {e}")
            return False
    
    def parse_gem_filename(self, filename: str) -> Dict[str, str]:
        """Parse gem filename to extract metadata"""
        base_name = Path(filename).stem
        
        # Remove timestamp if present
        base_name = re.sub(r'_\d{8}_\d{6}$', '', base_name)
        
        # Handle various naming formats
        light_source = 'Unknown'
        
        # Check for light source in filename
        if '_halogen_' in filename.lower() or 'halogen' in filename.lower():
            light_source = 'Halogen'
        elif '_laser_' in filename.lower() or 'laser' in filename.lower():
            light_source = 'Laser'
        elif '_uv_' in filename.lower() or 'uv' in filename.lower():
            light_source = 'UV'
        else:
            # Try standard format parsing
            pattern = r'^(.+?)([BLU])([CP])(\d+)$'
            match = re.match(pattern, base_name, re.IGNORECASE)
            if match:
                prefix, light, orientation, scan = match.groups()
                light_mapping = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                light_source = light_mapping.get(light.upper(), 'Unknown')
                base_name = prefix
        
        # Clean up base name
        base_name = re.sub(r'_\w+_structural$', '', base_name)
        
        return {
            'gem_id': base_name,
            'light_source': light_source,
            'orientation': 'C',
            'scan_number': '1',
            'full_name': base_name,
            'original_filename': filename
        }
    
    def check_for_gem_duplicates(self, csv_files: List[Path]) -> Dict[str, List[Path]]:
        """Check for multiple versions of the same gem"""
        print("\nChecking for gem duplicates...")
        
        gem_groups = {}
        
        for csv_file in csv_files:
            file_info = self.parse_gem_filename(csv_file.name)
            gem_key = f"{file_info['gem_id']}_{file_info['light_source']}"
            
            if gem_key not in gem_groups:
                gem_groups[gem_key] = []
            gem_groups[gem_key].append(csv_file)
        
        duplicates = {k: v for k, v in gem_groups.items() if len(v) > 1}
        
        if duplicates:
            print(f"Found duplicate gems:")
            for gem_key, files in duplicates.items():
                print(f"   {gem_key}: {len(files)} versions")
        else:
            print("No duplicate gems found")
        
        return duplicates
    
    def handle_duplicates(self, duplicates: Dict[str, List[Path]]) -> List[Path]:
        """Handle duplicate gems with forensic considerations"""
        if not duplicates:
            return []
        
        print("\nDUPLICATE HANDLING - GEMOLOGICAL CONSIDERATIONS:")
        print("IMPORTANT: Gems may undergo treatments between analyses!")
        print("• Heat treatment changes spectral features")
        print("• Irradiation alters UV characteristics")
        print("• Fracture filling affects laser response")
        print("• Keeping both versions preserves treatment evidence")
        print()
        print("OPTIONS:")
        print("1. Ask for each duplicate (recommended - preserves evidence)")
        print("2. Keep all versions (safest for forensics)")
        print("3. Keep latest only (risky - may lose treatment history)")
        
        try:
            choice = input("Choose strategy (1-3): ").strip()
        except:
            choice = "1"
        
        files_to_skip = []
        
        if choice == "2":
            print("Keeping all versions - maximum forensic preservation")
            return []
        elif choice == "3":
            print("WARNING: This may destroy treatment evidence!")
            confirm = input("Are you sure you want to keep only latest? (y/N): ").strip().lower()
            if confirm != 'y':
                choice = "1"
        
        if choice == "1" or choice not in ["2", "3"]:
            # Ask for each duplicate
            for gem_key, files in duplicates.items():
                print(f"\n{gem_key} - Multiple temporal versions found:")
                print("GEMOLOGICAL ANALYSIS: Same gem analyzed at different times")
                
                for i, file_path in enumerate(files, 1):
                    timestamp_match = re.search(r'_(\d{8}_\d{6})', file_path.stem)
                    if timestamp_match:
                        timestamp_str = timestamp_match.group(1)
                        try:
                            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            formatted_date = dt.strftime("%B %d, %Y at %I:%M %p")
                        except:
                            formatted_date = timestamp_str
                    else:
                        formatted_date = "unknown date"
                    
                    print(f"     {i}. {file_path.name}")
                    print(f"        Analyzed: {formatted_date}")
                
                print("\nTREATMENT CONSIDERATIONS:")
                print("• If gem was treated between analyses → KEEP BOTH")
                print("• If just re-analysis for verification → Keep latest")
                print("• If unsure → KEEP BOTH (safer for evidence)")
                
                try:
                    decision = input(f"Decision (1-{len(files)}, 'all' to keep all, 'latest' for newest): ").strip().lower()
                    
                    if decision == 'all':
                        print("KEEPING ALL: Preserving complete temporal record")
                        continue
                    elif decision == 'latest':
                        # Sort by timestamp and keep latest
                        files_with_time = []
                        for file_path in files:
                            timestamp_match = re.search(r'_(\d{8}_\d{6})', file_path.stem)
                            if timestamp_match:
                                try:
                                    timestamp = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S")
                                except:
                                    timestamp = datetime.min
                            else:
                                timestamp = datetime.min
                            files_with_time.append((file_path, timestamp))
                        
                        files_with_time.sort(key=lambda x: x[1], reverse=True)
                        latest_file = files_with_time[0][0]
                        older_files = [f[0] for f in files_with_time[1:]]
                        
                        print(f"KEEPING LATEST: {latest_file.name}")
                        files_to_skip.extend(older_files)
                        
                    elif decision.isdigit():
                        choice_idx = int(decision) - 1
                        if 0 <= choice_idx < len(files):
                            keep_file = files[choice_idx]
                            skip_files = [f for f in files if f != keep_file]
                            files_to_skip.extend(skip_files)
                            print(f"KEEPING: {keep_file.name}")
                        else:
                            print("Invalid choice, keeping all for safety")
                    else:
                        print("Invalid choice, keeping all for safety")
                        
                except Exception:
                    print("Error in selection, keeping all for safety")
        
        elif choice == "3":
            # Keep only latest of each duplicate set
            for gem_key, files in duplicates.items():
                files_with_time = []
                for file_path in files:
                    timestamp_match = re.search(r'_(\d{8}_\d{6})', file_path.stem)
                    if timestamp_match:
                        try:
                            timestamp = datetime.strptime(timestamp_match.group(1), "%Y%m%d_%H%M%S")
                        except:
                            timestamp = datetime.min
                    else:
                        timestamp = datetime.min
                    files_with_time.append((file_path, timestamp))
                
                files_with_time.sort(key=lambda x: x[1], reverse=True)
                latest_file = files_with_time[0][0]
                older_files = [f[0] for f in files_with_time[1:]]
                
                print(f"KEEPING LATEST: {latest_file.name}")
                files_to_skip.extend(older_files)
        
        return files_to_skip
    
    def get_column_value(self, row: pd.Series, possible_names: List[str]) -> Optional[str]:
        """Get value from row using multiple possible column names"""
        for name in possible_names:
            if name in row and pd.notna(row[name]):
                return row[name]
        return None
    
    def import_csv_file(self, csv_path: Path, conn: sqlite3.Connection) -> Tuple[int, int]:
        """Import a single CSV file with proper column mapping"""
        try:
            df = pd.read_csv(csv_path)
            file_info = self.parse_gem_filename(csv_path.name)
            
            print(f"Processing: {csv_path.name}")
            print(f"   Gem: {file_info['gem_id']}, Light: {file_info['light_source']}")
            
            inserted = 0
            skipped = 0
            
            for index, row in df.iterrows():
                try:
                    # Extract required fields with proper column mapping
                    wavelength = self.get_column_value(row, ['Wavelength', 'wavelength', 'Wavelength_nm'])
                    intensity = self.get_column_value(row, ['Intensity', 'intensity'])
                    feature_id = self.get_column_value(row, ['Feature', 'feature'])
                    feature_group = self.get_column_value(row, ['Feature_Group', 'feature_group'])
                    point_type = self.get_column_value(row, ['Point_Type', 'point_type', 'Type'])
                    
                    # Validate required fields
                    missing_fields = []
                    if not wavelength: missing_fields.append("Wavelength")
                    if not intensity: missing_fields.append("Intensity")
                    if not feature_id: missing_fields.append("Feature")
                    if not feature_group: missing_fields.append("Feature_Group")
                    if not point_type: missing_fields.append("Point_Type")
                    
                    if missing_fields:
                        skipped += 1
                        continue
                    
                    # Optional fields
                    data_type = self.get_column_value(row, ['data_type', 'Data_Type'])
                    start_wl = self.get_column_value(row, ['start_wavelength', 'Start'])
                    end_wl = self.get_column_value(row, ['end_wavelength', 'End'])
                    midpoint = self.get_column_value(row, ['midpoint', 'Midpoint'])
                    bottom = self.get_column_value(row, ['bottom', 'Bottom'])
                    norm_scheme = self.get_column_value(row, ['Normalization_Scheme', 'normalization_scheme'])
                    ref_wl = self.get_column_value(row, ['Reference_Wavelength', 'reference_wavelength'])
                    
                    # Insert into database
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO structural_features 
                        (file, light_source, wavelength, intensity, feature, feature_group, point_type, data_type,
                         start_wavelength, end_wavelength, midpoint, bottom, 
                         normalization_scheme, reference_wavelength)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        file_info['original_filename'],
                        file_info['light_source'],
                        float(wavelength),
                        float(intensity),
                        str(feature_id),
                        str(feature_group),
                        str(point_type),
                        str(data_type) if data_type else None,
                        float(start_wl) if start_wl is not None else None,
                        float(end_wl) if end_wl is not None else None,
                        float(midpoint) if midpoint is not None else None,
                        float(bottom) if bottom is not None else None,
                        str(norm_scheme) if norm_scheme else None,
                        float(ref_wl) if ref_wl is not None else None
                    ))
                    inserted += 1
                    
                except Exception as row_error:
                    skipped += 1
                    continue
            
            print(f"   Inserted: {inserted}, Skipped: {skipped}")
            return inserted, skipped
            
        except Exception as e:
            print(f"File error {csv_path.name}: {e}")
            return 0, 0
    
    def archive_imported_files(self, successful_files: List[Path]) -> int:
        """Move successfully imported files to archive for Option 8"""
        if not successful_files:
            return 0
        
        print(f"\nArchiving {len(successful_files)} imported files for Option 8...")
        
        archived_count = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in successful_files:
            try:
                archive_name = f"{file_path.stem}_archived_{timestamp}{file_path.suffix}"
                archive_path = self.archive_dir / archive_name
                
                shutil.move(str(file_path), str(archive_path))
                archived_count += 1
                
            except Exception as e:
                print(f"Error archiving {file_path.name}: {e}")
        
        if archived_count > 0:
            print(f"Successfully archived {archived_count} files")
            print(f"Archive location: {self.archive_dir}")
            print("Files ready for main.py Option 8 (Structural Matching Test)")
        
        return archived_count
    
    def export_to_csv(self):
        """Export database contents to CSV"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT file, light_source, wavelength, intensity, feature, feature_group, point_type,
                       data_type, start_wavelength, end_wavelength, midpoint, bottom,
                       normalization_scheme, reference_wavelength, timestamp
                FROM structural_features
                ORDER BY file, light_source, wavelength
            '''
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            df.to_csv(self.csv_output_path, index=False)
            print(f"CSV export saved: {self.csv_output_path} ({len(df)} records)")
            return True
            
        except Exception as e:
            print(f"CSV export error: {e}")
            return False
    
    def get_database_stats(self):
        """Display database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source")
            by_light = cursor.fetchall()
            
            cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features")
            unique_files = cursor.fetchone()[0]
            
            conn.close()
            
            print(f"\nDATABASE STATISTICS:")
            print(f"   Total records: {total:,}")
            print(f"   Unique files: {unique_files}")
            print(f"   By light source:")
            for light, count in by_light:
                print(f"     {light}: {count:,}")
            
        except Exception as e:
            print(f"Stats error: {e}")
    
    def run_batch_import(self):
        """Main batch import process"""
        print(f"\nStarting forensic gemological batch import")
        print("=" * 60)
        
        if not self.structural_data_dir.exists():
            print(f"Source directory not found: {self.structural_data_dir}")
            return False
        
        csv_files = list(self.structural_data_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {self.structural_data_dir}")
            return False
        
        print(f"Found {len(csv_files)} CSV files to import")
        
        # Handle duplicates with forensic considerations
        duplicates = self.check_for_gem_duplicates(csv_files)
        files_to_skip = []
        
        if duplicates:
            files_to_skip = self.handle_duplicates(duplicates)
        
        # Filter out skipped files
        final_csv_files = [f for f in csv_files if f not in files_to_skip]
        
        if len(final_csv_files) < len(csv_files):
            skipped_count = len(csv_files) - len(final_csv_files)
            print(f"\nFinal file list:")
            print(f"   Original files: {len(csv_files)}")
            print(f"   Duplicate files skipped: {skipped_count}")
            print(f"   Files to import: {len(final_csv_files)}")
        
        if not final_csv_files:
            print("No files left to import")
            return True
        
        # Create database schema
        if not self.create_database_schema():
            return False
        
        # Import files
        total_inserted = 0
        total_skipped = 0
        successfully_imported_files = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            for csv_file in final_csv_files:
                inserted, skipped = self.import_csv_file(csv_file, conn)
                total_inserted += inserted
                total_skipped += skipped
                if inserted > 0:
                    successfully_imported_files.append(csv_file)
            
            conn.commit()
            conn.close()
            
            print(f"\nIMPORT COMPLETE")
            print(f"   Files processed: {len(final_csv_files)}")
            print(f"   Successful imports: {len(successfully_imported_files)}")
            print(f"   Total records inserted: {total_inserted:,}")
            print(f"   Total records skipped: {total_skipped:,}")
            
            if total_inserted > 0:
                self.export_to_csv()
                self.get_database_stats()
                
                print(f"\nFORENSIC DUPLICATE PROTECTION ACTIVE:")
                print(f"   Temporal analysis preservation (treatment detection)")
                print(f"   User-controlled duplicate handling (evidence protection)")
                print(f"   Database duplicates prevented (UNIQUE constraint)")
                
                # Archive imported files for Option 8
                archived_count = self.archive_imported_files(successfully_imported_files)
                
                if archived_count > 0:
                    print(f"\nAUTOMATIC ARCHIVING COMPLETE")
                    print(f"   {archived_count} files moved to structural(archive)")
                    print(f"   Ready for main.py Option 8: Structural Matching (Test)")
                
                return True
            else:
                print("No records were imported")
                return False
                
        except Exception as e:
            print(f"Import process error: {e}")
            return False

def main():
    """Main entry point"""
    print("PRODUCTION FORENSIC GEMOLOGICAL BATCH IMPORTER")
    print("=" * 70)
    print("Features:")
    print("• Temporal analysis preservation (before/after treatment)")
    print("• Smart duplicate detection with forensic considerations")
    print("• Treatment evidence protection (heat, irradiation, filling)")
    print("• Database duplicate prevention (UNIQUE constraints)")
    print("• Automatic archiving for Option 8 testing")
    print("=" * 70)
    
    importer = ProductionBatchImporter()
    success = importer.run_batch_import()
    
    if success:
        print(f"\nForensic gemological import completed successfully!")
        print(f"Temporal analysis data preserved for treatment detection")
    else:
        print(f"\nBatch import failed")
    
    return success

if __name__ == "__main__":
    main()
