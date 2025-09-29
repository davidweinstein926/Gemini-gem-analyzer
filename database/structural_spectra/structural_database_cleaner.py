#!/usr/bin/env python3
"""
STRUCTURAL DATABASE DUPLICATE CLEANER
Removes duplicate records from structural databases based on gem name and date (TIME IGNORED)

DUPLICATE DEFINITION:
- Same gem name (e.g., 58BC1, 58LC1, 58UC1) 
- Same date (e.g., 20250915) - TIME PORTION IGNORED
- Keeps only the most recent TIME for each gem/date combination

EXAMPLE:
- 58BC1_20250915_143022.csv (14:30:22)
- 58BC1_20250915_151203.csv (15:12:03)  ← KEEPS THIS (most recent time)
- 58BC1_20250915_162445.csv (16:24:45)  ← KEEPS THIS (most recent time)

Databases cleaned:
1. root/database/structural_spectra/gemini_structural_db.csv
2. root/database/structural_spectra/multi_structural_gem_data.db
"""

import pandas as pd
import sqlite3
import re
from pathlib import Path
from datetime import datetime
import shutil
import sys

class StructuralDatabaseCleaner:
    def __init__(self):
        self.project_root = self.find_project_root()
        self.db_dir = self.project_root / "database" / "structural_spectra"
        self.csv_path = self.db_dir / "gemini_structural_db.csv"
        self.sqlite_path = self.db_dir / "multi_structural_gem_data.db"
        
        # Statistics
        self.stats = {
            'csv_total': 0,
            'csv_duplicates': 0,
            'csv_kept': 0,
            'sqlite_total': 0,
            'sqlite_duplicates': 0,
            'sqlite_kept': 0
        }
        
    def find_project_root(self):
        """Find project root directory"""
        current_path = Path(__file__).parent.absolute()
        
        for path in [current_path] + list(current_path.parents):
            if (path / "database" / "structural_spectra").exists():
                return path
            if (path / "main.py").exists() and (path / "database").exists():
                return path
        
        return current_path
    
    def extract_gem_info(self, filename_or_identifier):
        """Extract gem name and date from filename, ignoring time portion"""
        # Handle various formats:
        # 58BC1_20250915_143022.csv → Gem: 58BC1, Date: 20250915 (ignore 143022 time)
        # 58BC1_20250915_151203.csv → Gem: 58BC1, Date: 20250915 (ignore 151203 time)
        # 214LC3_structural_20250920_092845.csv → Gem: 214LC3, Date: 20250920
        
        if pd.isna(filename_or_identifier):
            return None, None
            
        identifier = str(filename_or_identifier)
        
        # Extract gem name (base pattern: prefix + light + orientation + scan)
        gem_pattern = r'^(.+?)([BLU])([CP])(\d+)'
        gem_match = re.match(gem_pattern, identifier, re.IGNORECASE)
        
        if gem_match:
            prefix, light, orientation, scan = gem_match.groups()
            gem_name = f"{prefix}{light.upper()}{orientation.upper()}{scan}"
        else:
            # Fallback - use everything before first underscore or dot
            gem_name = identifier.split('_')[0].split('.')[0]
        
        # Extract date (YYYYMMDD format) - IGNORE time portion
        # Look for 8-digit date pattern, ignore any 6-digit time that follows
        date_pattern = r'(\d{8})(?:_\d{6})?'  # Date optionally followed by _HHMMSS
        date_match = re.search(date_pattern, identifier)
        
        if date_match:
            date_str = date_match.group(1)  # Only capture the 8-digit date part
            try:
                # Validate date format
                datetime.strptime(date_str, '%Y%m%d')
                return gem_name, date_str
            except ValueError:
                pass
        
        # If no date found, use a default (will group all no-date records together)
        return gem_name, "no_date"
    
    def backup_databases(self):
        """Create backups of both databases before cleaning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backups_created = []
        
        # Backup CSV
        if self.csv_path.exists():
            csv_backup = self.csv_path.parent / f"gemini_structural_db_backup_{timestamp}.csv"
            shutil.copy2(self.csv_path, csv_backup)
            backups_created.append(csv_backup)
            print(f"✅ CSV backup created: {csv_backup.name}")
        
        # Backup SQLite
        if self.sqlite_path.exists():
            sqlite_backup = self.sqlite_path.parent / f"multi_structural_gem_data_backup_{timestamp}.db"
            shutil.copy2(self.sqlite_path, sqlite_backup)
            backups_created.append(sqlite_backup)
            print(f"✅ SQLite backup created: {sqlite_backup.name}")
        
        return backups_created
    
    def clean_csv_database(self, dry_run=True):
        """Clean duplicates from CSV database"""
        print(f"\n🔍 Analyzing CSV database: {self.csv_path}")
        
        if not self.csv_path.exists():
            print(f"❌ CSV database not found: {self.csv_path}")
            return
        
        try:
            # Load CSV
            df = pd.read_csv(self.csv_path)
            self.stats['csv_total'] = len(df)
            print(f"📊 Total records: {len(df)}")
            
            if len(df) == 0:
                print("⚠️ CSV database is empty")
                return
            
            # Print column names to understand structure
            print(f"📋 Columns: {list(df.columns)}")
            
            # Try to find the identifier column
            identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
            identifier_col = None
            
            for col in identifier_columns:
                if col in df.columns:
                    identifier_col = col
                    break
            
            if identifier_col is None:
                # Try first column that looks like a filename
                for col in df.columns:
                    if df[col].dtype == 'object':  # String column
                        sample_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                        if any(pattern in sample_val.upper() for pattern in ['BC', 'LC', 'UC', '.CSV', '.TXT']):
                            identifier_col = col
                            break
            
            if identifier_col is None:
                print(f"❌ Could not identify gem identifier column in CSV")
                print(f"Available columns: {list(df.columns)}")
                return
            
            print(f"🎯 Using identifier column: '{identifier_col}'")
            
            # Extract gem info
            gem_info = df[identifier_col].apply(self.extract_gem_info)
            df['gem_name'] = [info[0] for info in gem_info]
            df['date_extracted'] = [info[1] for info in gem_info]
            
            # Show sample of extracted info
            print(f"\n📋 Sample gem info extraction (TIME IGNORED):")
            sample_df = df[[identifier_col, 'gem_name', 'date_extracted']].head()
            for _, row in sample_df.iterrows():
                original = row[identifier_col]
                gem = row['gem_name']
                date = row['date_extracted']
                print(f"   {original}")
                print(f"      → Gem: {gem}, Date: {date} (time ignored)")
                print()
            
            # Find duplicates (same gem_name and date, ignoring time)
            duplicate_groups = df.groupby(['gem_name', 'date_extracted'])
            
            duplicates_found = []
            records_to_keep = []
            
            print(f"\n🔍 Searching for duplicates (same gem + same date, ignoring time)...")
            
            for (gem_name, date), group in duplicate_groups:
                if len(group) > 1:
                    # Multiple records for same gem and same date
                    # Sort by the original identifier to get most recent time
                    # (assuming later alphabetically = later time for same date)
                    group_sorted = group.sort_values(identifier_col)
                    
                    # Keep the last one (most recent time), delete others
                    duplicates_found.extend(group_sorted.index[:-1].tolist())  # All but last
                    records_to_keep.append(group_sorted.index[-1])  # Keep last one
                    
                    print(f"\n🔍 Found {len(group)} records for Gem {gem_name} on Date {date}:")
                    for idx, row in group_sorted.iterrows():
                        status = "KEEP (most recent)" if idx == group_sorted.index[-1] else "DELETE (older)"
                        print(f"   {status}: {row[identifier_col]}")
                else:
                    # Single record - keep it
                    records_to_keep.append(group.index[0])
            
            self.stats['csv_duplicates'] = len(duplicates_found)
            self.stats['csv_kept'] = len(records_to_keep)
            
            print(f"\n📊 CSV Analysis Summary:")
            print(f"   Total records: {self.stats['csv_total']}")
            print(f"   Duplicate records to delete: {self.stats['csv_duplicates']}")
            print(f"   Records to keep: {self.stats['csv_kept']}")
            
            if len(duplicates_found) == 0:
                print("✅ No duplicates found in CSV database!")
                return
            
            if dry_run:
                print("\n🔍 DRY RUN - No changes made to CSV database")
                print("   Run with --execute to actually remove duplicates")
            else:
                # Remove duplicates
                print(f"\n🗑️ Removing {len(duplicates_found)} duplicate records from CSV...")
                df_cleaned = df.drop(index=duplicates_found)
                
                # Remove our temporary columns
                df_cleaned = df_cleaned.drop(columns=['gem_name', 'date_extracted'])
                
                # Save cleaned CSV
                df_cleaned.to_csv(self.csv_path, index=False)
                print(f"✅ CSV database cleaned! {len(df_cleaned)} records remaining")
                
        except Exception as e:
            print(f"❌ Error cleaning CSV database: {e}")
            import traceback
            traceback.print_exc()
    
    def clean_sqlite_database(self, dry_run=True):
        """Clean duplicates from SQLite database"""
        print(f"\n🔍 Analyzing SQLite database: {self.sqlite_path}")
        
        if not self.sqlite_path.exists():
            print(f"❌ SQLite database not found: {self.sqlite_path}")
            return
        
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"📋 Tables found: {[table[0] for table in tables]}")
            
            if not tables:
                print("⚠️ No tables found in SQLite database")
                conn.close()
                return
            
            # Assume main table is the first one or look for common names
            table_name = None
            common_table_names = ['structural_features', 'features', 'gems', 'data']
            
            for common_name in common_table_names:
                if any(common_name in table[0].lower() for table in tables):
                    table_name = next(table[0] for table in tables if common_name in table[0].lower())
                    break
            
            if table_name is None:
                table_name = tables[0][0]  # Use first table
            
            print(f"🎯 Using table: '{table_name}'")
            
            # Get table structure
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            columns = [col[1] for col in columns_info]
            print(f"📋 Columns: {columns}")
            
            # Get total record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = cursor.fetchone()[0]
            self.stats['sqlite_total'] = total_records
            print(f"📊 Total records: {total_records}")
            
            if total_records == 0:
                print("⚠️ SQLite database is empty")
                conn.close()
                return
            
            # Find identifier column
            identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
            identifier_col = None
            
            for col in identifier_columns:
                if col in columns:
                    identifier_col = col
                    break
            
            if identifier_col is None:
                print(f"❌ Could not identify gem identifier column in SQLite table")
                print(f"Available columns: {columns}")
                conn.close()
                return
            
            print(f"🎯 Using identifier column: '{identifier_col}'")
            
            # Load data for analysis
            query = f"SELECT rowid, {identifier_col} FROM {table_name}"
            df = pd.read_sql_query(query, conn)
            
            # Extract gem info
            gem_info = df[identifier_col].apply(self.extract_gem_info)
            df['gem_name'] = [info[0] for info in gem_info]
            df['date_extracted'] = [info[1] for info in gem_info]
            
            # Show sample
            print(f"\n📋 Sample gem info extraction (TIME IGNORED):")
            sample_df = df[[identifier_col, 'gem_name', 'date_extracted']].head()
            for _, row in sample_df.iterrows():
                original = row[identifier_col]
                gem = row['gem_name']
                date = row['date_extracted']
                print(f"   {original}")
                print(f"      → Gem: {gem}, Date: {date} (time ignored)")
                print()
            
            # Find duplicates (same gem_name and date, ignoring time)
            duplicate_groups = df.groupby(['gem_name', 'date_extracted'])
            
            rowids_to_delete = []
            
            print(f"\n🔍 Searching for duplicates (same gem + same date, ignoring time)...")
            
            for (gem_name, date), group in duplicate_groups:
                if len(group) > 1:
                    # Multiple records for same gem and same date
                    # Sort by identifier to get most recent time (assuming later = more recent)
                    group_sorted = group.sort_values(identifier_col)
                    
                    # Keep the last rowid (most recent time), delete others
                    rowids_to_delete.extend(group_sorted['rowid'].iloc[:-1].tolist())
                    
                    print(f"\n🔍 Found {len(group)} records for Gem {gem_name} on Date {date}:")
                    for _, row in group_sorted.iterrows():
                        status = "KEEP (most recent)" if row['rowid'] == group_sorted['rowid'].iloc[-1] else "DELETE (older)"
                        print(f"   {status}: {row[identifier_col]} (rowid: {row['rowid']})")
            
            self.stats['sqlite_duplicates'] = len(rowids_to_delete)
            self.stats['sqlite_kept'] = total_records - len(rowids_to_delete)
            
            print(f"\n📊 SQLite Analysis Summary:")
            print(f"   Total records: {self.stats['sqlite_total']}")
            print(f"   Duplicate records to delete: {self.stats['sqlite_duplicates']}")
            print(f"   Records to keep: {self.stats['sqlite_kept']}")
            
            if len(rowids_to_delete) == 0:
                print("✅ No duplicates found in SQLite database!")
                conn.close()
                return
            
            if dry_run:
                print("\n🔍 DRY RUN - No changes made to SQLite database")
                print("   Run with --execute to actually remove duplicates")
            else:
                # Delete duplicates
                print(f"\n🗑️ Removing {len(rowids_to_delete)} duplicate records from SQLite...")
                
                # Delete in batches to avoid SQL limits
                batch_size = 100
                for i in range(0, len(rowids_to_delete), batch_size):
                    batch = rowids_to_delete[i:i + batch_size]
                    placeholders = ','.join(['?' for _ in batch])
                    delete_query = f"DELETE FROM {table_name} WHERE rowid IN ({placeholders})"
                    cursor.execute(delete_query, batch)
                
                conn.commit()
                print(f"✅ SQLite database cleaned! {self.stats['sqlite_kept']} records remaining")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Error cleaning SQLite database: {e}")
            import traceback
            traceback.print_exc()
    
    def display_all_records_csv(self, df, identifier_col):
        """Display all CSV records in a nice format for manual selection"""
        print(f"\n📋 ALL RECORDS IN CSV DATABASE ({len(df)} total)")
        print("=" * 80)
        
        # Add gem info to dataframe for display
        gem_info = df[identifier_col].apply(self.extract_gem_info)
        df_display = df.copy()
        df_display['gem_name'] = [info[0] for info in gem_info]
        df_display['date_extracted'] = [info[1] for info in gem_info]
        
        # Group by gem name and date for easier viewing
        grouped = df_display.groupby(['gem_name', 'date_extracted'])
        
        record_index = 0
        index_map = {}  # Maps display index to dataframe index
        
        for (gem_name, date), group in grouped:
            print(f"\n🔸 Gem: {gem_name}, Date: {date}")
            print("-" * 50)
            
            # Sort by identifier to show time order
            group_sorted = group.sort_values(identifier_col)
            
            for _, row in group_sorted.iterrows():
                index_map[record_index] = row.name  # Store original dataframe index
                
                # Show key information
                timestamp_info = ""
                if "_" in row[identifier_col]:
                    parts = row[identifier_col].split("_")
                    if len(parts) >= 3 and len(parts[2]) == 6:  # Has time
                        time_part = parts[2].replace('.csv', '')
                        if len(time_part) == 6:
                            timestamp_info = f" Time: {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                
                # Display additional columns if available
                extra_info = ""
                if 'light_source' in df.columns:
                    extra_info += f" | Light: {row.get('light_source', 'N/A')}"
                if 'wavelength' in df.columns:
                    extra_info += f" | Wavelength: {row.get('wavelength', 'N/A')}"
                if len(df.columns) > 3:
                    extra_info += f" | Cols: {len(df.columns)}"
                
                print(f"   [{record_index:3d}] {row[identifier_col]}{timestamp_info}{extra_info}")
                record_index += 1
        
        return index_map
    
    def display_all_records_sqlite(self, conn, table_name, identifier_col):
        """Display all SQLite records in a nice format for manual selection"""
        cursor = conn.cursor()
        
        # Get all records
        cursor.execute(f"SELECT rowid, * FROM {table_name} ORDER BY {identifier_col}")
        records = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        columns = ['rowid'] + [col[1] for col in columns_info]
        
        print(f"\n📋 ALL RECORDS IN SQLITE DATABASE ({len(records)} total)")
        print("=" * 80)
        
        # Create dataframe for easier processing
        df = pd.DataFrame(records, columns=columns)
        
        # Add gem info
        gem_info = df[identifier_col].apply(self.extract_gem_info)
        df['gem_name'] = [info[0] for info in gem_info]
        df['date_extracted'] = [info[1] for info in gem_info]
        
        # Group by gem name and date
        grouped = df.groupby(['gem_name', 'date_extracted'])
        
        record_index = 0
        index_map = {}  # Maps display index to rowid
        
        for (gem_name, date), group in grouped:
            print(f"\n🔸 Gem: {gem_name}, Date: {date}")
            print("-" * 50)
            
            # Sort by identifier to show time order
            group_sorted = group.sort_values(identifier_col)
            
            for _, row in group_sorted.iterrows():
                index_map[record_index] = row['rowid']  # Store rowid for deletion
                
                # Show key information
                timestamp_info = ""
                if "_" in row[identifier_col]:
                    parts = row[identifier_col].split("_")
                    if len(parts) >= 3 and len(parts[2]) >= 6:  # Has time
                        time_part = parts[2][:6]
                        if time_part.isdigit():
                            timestamp_info = f" Time: {time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                
                # Display additional columns if available
                extra_info = ""
                if 'light_source' in df.columns:
                    extra_info += f" | Light: {row.get('light_source', 'N/A')}"
                if 'wavelength' in df.columns:
                    extra_info += f" | Wavelength: {row.get('wavelength', 'N/A')}"
                extra_info += f" | RowID: {row['rowid']}"
                
                print(f"   [{record_index:3d}] {row[identifier_col]}{timestamp_info}{extra_info}")
                record_index += 1
        
        return index_map
    
    def interactive_selection(self, total_records):
        """Interactive selection of records to delete"""
        print(f"\n🎯 INTERACTIVE RECORD SELECTION")
        print("=" * 50)
        print(f"📊 Total records shown: {total_records}")
        print()
        print("📋 Selection options:")
        print("   • Single: 5")
        print("   • Multiple: 5,7,12,15")
        print("   • Range: 5-10")
        print("   • Mixed: 1,5-8,12,20-25")
        print("   • All: all")
        print("   • None: none (or just press Enter)")
        print()
        
        while True:
            try:
                selection = input("🔍 Enter record numbers to DELETE: ").strip()
                
                if not selection or selection.lower() == 'none':
                    return []
                
                if selection.lower() == 'all':
                    confirm = input(f"⚠️  DELETE ALL {total_records} records? Type 'yes' to confirm: ")
                    if confirm.lower() == 'yes':
                        return list(range(total_records))
                    else:
                        continue
                
                # Parse selection
                selected_indices = set()
                
                for part in selection.split(','):
                    part = part.strip()
                    
                    if '-' in part:
                        # Range
                        start, end = map(int, part.split('-'))
                        if start > end:
                            start, end = end, start
                        selected_indices.update(range(start, end + 1))
                    else:
                        # Single number
                        selected_indices.add(int(part))
                
                # Validate indices
                valid_indices = []
                invalid_indices = []
                
                for idx in selected_indices:
                    if 0 <= idx < total_records:
                        valid_indices.append(idx)
                    else:
                        invalid_indices.append(idx)
                
                if invalid_indices:
                    print(f"⚠️  Invalid indices (out of range): {invalid_indices}")
                    print(f"   Valid range: 0-{total_records-1}")
                    continue
                
                # Show confirmation
                print(f"\n📋 Selected {len(valid_indices)} records for deletion:")
                if len(valid_indices) <= 20:  # Show details if not too many
                    print(f"   Indices: {sorted(valid_indices)}")
                else:
                    print(f"   Indices: {sorted(valid_indices)[:10]}... and {len(valid_indices)-10} more")
                
                confirm = input(f"\n⚠️  Proceed with deleting {len(valid_indices)} records? (y/n): ")
                if confirm.lower() in ['y', 'yes']:
                    return sorted(valid_indices)
                else:
                    print("Selection cancelled. Try again:")
                    continue
                    
            except ValueError as e:
                print(f"❌ Invalid input format: {e}")
                print("Please use format like: 1,5-8,12 or 'all' or 'none'")
                continue
            except KeyboardInterrupt:
                print("\n❌ Selection cancelled")
                return []
    
    def interactive_clean_csv_database(self):
        """Interactive cleaning of CSV database"""
        print(f"\n🔍 INTERACTIVE CSV DATABASE CLEANING")
        print("=" * 60)
        
        if not self.csv_path.exists():
            print(f"❌ CSV database not found: {self.csv_path}")
            return
        
        try:
            # Load CSV
            df = pd.read_csv(self.csv_path)
            print(f"📊 Loaded {len(df)} records from CSV database")
            
            if len(df) == 0:
                print("⚠️ CSV database is empty")
                return
            
            # Find identifier column
            identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
            identifier_col = None
            
            for col in identifier_columns:
                if col in df.columns:
                    identifier_col = col
                    break
            
            if identifier_col is None:
                print(f"❌ Could not identify gem identifier column")
                return
            
            # Display all records
            index_map = self.display_all_records_csv(df, identifier_col)
            
            # Interactive selection
            selected_indices = self.interactive_selection(len(index_map))
            
            if not selected_indices:
                print("✅ No records selected for deletion")
                return
            
            # Map selected display indices to dataframe indices
            df_indices_to_delete = [index_map[i] for i in selected_indices]
            
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_backup = self.csv_path.parent / f"gemini_structural_db_backup_{timestamp}.csv"
            shutil.copy2(self.csv_path, csv_backup)
            print(f"✅ Backup created: {csv_backup.name}")
            
            # Delete selected records
            df_cleaned = df.drop(index=df_indices_to_delete)
            df_cleaned.to_csv(self.csv_path, index=False)
            
            print(f"✅ Deleted {len(selected_indices)} records from CSV database")
            print(f"📊 Remaining records: {len(df_cleaned)}")
            
        except Exception as e:
            print(f"❌ Error in interactive CSV cleaning: {e}")
    
    def interactive_clean_sqlite_database(self):
        """Interactive cleaning of SQLite database"""
        print(f"\n🔍 INTERACTIVE SQLITE DATABASE CLEANING")
        print("=" * 60)
        
        if not self.sqlite_path.exists():
            print(f"❌ SQLite database not found: {self.sqlite_path}")
            return
        
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            # Get table info
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            if not tables:
                print("⚠️ No tables found in SQLite database")
                conn.close()
                return
            
            # Find main table
            table_name = tables[0][0]  # Use first table
            common_table_names = ['structural_features', 'features', 'gems', 'data']
            for common_name in common_table_names:
                if any(common_name in table[0].lower() for table in tables):
                    table_name = next(table[0] for table in tables if common_name in table[0].lower())
                    break
            
            # Get record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = cursor.fetchone()[0]
            print(f"📊 Found {total_records} records in table '{table_name}'")
            
            if total_records == 0:
                print("⚠️ SQLite table is empty")
                conn.close()
                return
            
            # Find identifier column
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            columns = [col[1] for col in columns_info]
            
            identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
            identifier_col = None
            for col in identifier_columns:
                if col in columns:
                    identifier_col = col
                    break
            
            if identifier_col is None:
                print(f"❌ Could not identify gem identifier column")
                conn.close()
                return
            
            # Display all records
            index_map = self.display_all_records_sqlite(conn, table_name, identifier_col)
            
            # Interactive selection
            selected_indices = self.interactive_selection(len(index_map))
            
            if not selected_indices:
                print("✅ No records selected for deletion")
                conn.close()
                return
            
            # Map selected display indices to rowids
            rowids_to_delete = [index_map[i] for i in selected_indices]
            
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sqlite_backup = self.sqlite_path.parent / f"multi_structural_gem_data_backup_{timestamp}.db"
            shutil.copy2(self.sqlite_path, sqlite_backup)
            print(f"✅ Backup created: {sqlite_backup.name}")
            
            # Delete selected records
            batch_size = 100
            for i in range(0, len(rowids_to_delete), batch_size):
                batch = rowids_to_delete[i:i + batch_size]
                placeholders = ','.join(['?' for _ in batch])
                delete_query = f"DELETE FROM {table_name} WHERE rowid IN ({placeholders})"
                cursor.execute(delete_query, batch)
            
            conn.commit()
            conn.close()
            
            print(f"✅ Deleted {len(selected_indices)} records from SQLite database")
            print(f"📊 Remaining records: {total_records - len(selected_indices)}")
            
        except Exception as e:
            print(f"❌ Error in interactive SQLite cleaning: {e}")
    
    def run_interactive_mode(self):
        """Run interactive mode for manual record selection"""
        print("🎮 INTERACTIVE MODE - MANUAL RECORD SELECTION")
        print("=" * 60)
        print("View all records and manually select which ones to delete")
        print()
        
        if not self.db_dir.exists():
            print(f"❌ Database directory not found: {self.db_dir}")
            return
        
        # Choose which database to clean
        print("📋 Available databases:")
        csv_exists = self.csv_path.exists()
        sqlite_exists = self.sqlite_path.exists()
        
        if csv_exists:
            print("   1. CSV Database (gemini_structural_db.csv)")
        if sqlite_exists:
            print("   2. SQLite Database (multi_structural_gem_data.db)")
        if csv_exists and sqlite_exists:
            print("   3. Both databases")
        
        if not csv_exists and not sqlite_exists:
            print("❌ No databases found")
            return
        
        while True:
            try:
                if csv_exists and sqlite_exists:
                    choice = input("\n🔍 Select database to clean (1/2/3): ").strip()
                    if choice == '1':
                        self.interactive_clean_csv_database()
                        break
                    elif choice == '2':
                        self.interactive_clean_sqlite_database()
                        break
                    elif choice == '3':
                        self.interactive_clean_csv_database()
                        self.interactive_clean_sqlite_database()
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1, 2, or 3")
                        continue
                elif csv_exists:
                    self.interactive_clean_csv_database()
                    break
                elif sqlite_exists:
                    self.interactive_clean_sqlite_database()
                    break
                    
            except KeyboardInterrupt:
                print("\n❌ Interactive mode cancelled")
                break
    
    def run_analysis(self, dry_run=True):
        """Run the complete duplicate analysis and cleaning"""
        print("🧹 STRUCTURAL DATABASE DUPLICATE CLEANER")
        print("=" * 50)
        print(f"Database directory: {self.db_dir}")
        print(f"Mode: {'DRY RUN (analysis only)' if dry_run else 'EXECUTE (will delete duplicates)'}")
        print()
        print("🎯 DUPLICATE DEFINITION:")
        print("   • Same gem name (e.g., 58BC1, 58LC1, 58UC1)")
        print("   • Same date (e.g., 20250915) - TIME IGNORED")
        print("   • Keeps: Most recent time for each gem/date")
        print("   • Deletes: Earlier times from same day")
        print()
        
        if not self.db_dir.exists():
            print(f"❌ Database directory not found: {self.db_dir}")
            return
        
        # Create backups if executing
        if not dry_run:
            print("📋 Creating backups before cleaning...")
            backups = self.backup_databases()
            if not backups:
                print("⚠️ No databases found to backup")
            print()
        
        # Clean CSV database
        self.clean_csv_database(dry_run)
        
        # Clean SQLite database  
        self.clean_sqlite_database(dry_run)
        
        # Final summary
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 30)
        print(f"Duplicate Definition: Same gem name + same date (time ignored)")
        print(f"Retention Policy: Keep most recent time for each gem/date")
        print()
        print(f"CSV Database:")
        print(f"   Total records: {self.stats['csv_total']}")
        print(f"   Duplicates found: {self.stats['csv_duplicates']}")
        print(f"   Records kept: {self.stats['csv_kept']}")
        print()
        print(f"SQLite Database:")
        print(f"   Total records: {self.stats['sqlite_total']}")
        print(f"   Duplicates found: {self.stats['sqlite_duplicates']}")
        print(f"   Records kept: {self.stats['sqlite_kept']}")
        print()
        
        total_duplicates = self.stats['csv_duplicates'] + self.stats['sqlite_duplicates']
        if total_duplicates > 0:
            if dry_run:
                print(f"🔍 Found {total_duplicates} total duplicates across both databases")
                print("💡 Run with --execute flag to actually remove duplicates")
            else:
                print(f"✅ Successfully removed {total_duplicates} duplicate records!")
                print("🎉 Databases are now clean!")
        else:
            print("✅ No duplicates found in either database!")

def main():
    """Main entry point"""
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--interactive', '-i', '--manual']:
            # Interactive mode
            try:
                cleaner = StructuralDatabaseCleaner()
                cleaner.run_interactive_mode()
            except KeyboardInterrupt:
                print("\n❌ Interactive mode cancelled by user")
            except Exception as e:
                print(f"\n❌ Error: {e}")
            return
        elif sys.argv[1] in ['--execute', '-e', '--run']:
            dry_run = False
        elif sys.argv[1] in ['--help', '-h']:
            print("🧹 STRUCTURAL DATABASE DUPLICATE CLEANER")
            print()
            print("Usage:")
            print("  python structural_database_cleaner.py                 # Dry run (analysis only)")
            print("  python structural_database_cleaner.py --execute       # Execute cleanup")
            print("  python structural_database_cleaner.py --interactive   # Manual selection")
            print("  python structural_database_cleaner.py --help          # Show this help")
            print()
            print("Modes:")
            print("  Dry Run:      Analyze and show what would be deleted (safe)")
            print("  Execute:      Actually delete duplicate records")
            print("  Interactive:  View all records and manually select which to delete")
            print()
            print("Duplicate Definition:")
            print("  • Same gem name (e.g., 58BC1, 58LC1, 58UC1)")
            print("  • Same date (e.g., 20250915) - TIME IGNORED")
            print("  • Keeps: Most recent time for each gem/date")
            return
        else:
            print(f"❌ Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    else:
        dry_run = True
    
    try:
        cleaner = StructuralDatabaseCleaner()
        cleaner.run_analysis(dry_run)
        
        if dry_run:
            print(f"\n💡 Available modes:")
            print(f"   python {Path(__file__).name} --execute      # Remove duplicates")
            print(f"   python {Path(__file__).name} --interactive  # Manual selection")
        
    except KeyboardInterrupt:
        print("\n❌ Cleaning cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()