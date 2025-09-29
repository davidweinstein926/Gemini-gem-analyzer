import sqlite3
import pandas as pd
from datetime import datetime
import hashlib
import logging
from typing import List, Dict, Any

class ColumnMappingImporter:
    """
    Import CSV data with column mapping to match database structure.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_table_structure(self, table_name: str) -> Dict:
        """Get the structure of the target table"""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        table_structure = {}
        for col_info in columns:
            col_name = col_info[1]
            col_type = col_info[2]
            not_null = col_info[3]
            table_structure[col_name] = {
                'type': col_type,
                'not_null': bool(not_null)
            }
        
        self.logger.info(f"Table '{table_name}' has {len(table_structure)} columns")
        return table_structure
    
    def create_column_mapping(self, csv_columns: List[str], table_structure: Dict) -> Dict:
        """Create mapping between CSV columns and database columns"""
        
        # Define mapping rules
        column_mappings = {
            # CSV column -> Database column
            'file': 'gem_id',  # file column maps to gem_id
            'timestamp': 'analysis_date',  # timestamp maps to analysis_date
            'light_source': 'light_source',  # direct match
            'wavelength': 'wavelength',  # direct match
            'intensity': 'intensity',  # direct match
            'feature': 'feature',  # direct match
            'feature_group': 'feature_group',  # direct match
            'point_type': 'point_type',  # direct match
            'normalization_scheme': 'normalization_scheme',  # direct match
            'reference_wavelength': 'reference_wavelength',  # direct match
        }
        
        # Create the actual mapping for available columns
        final_mapping = {}
        unmapped_csv = []
        
        for csv_col in csv_columns:
            if csv_col in column_mappings:
                db_col = column_mappings[csv_col]
                if db_col in table_structure:
                    final_mapping[csv_col] = db_col
                else:
                    unmapped_csv.append(f"{csv_col} -> {db_col} (DB col not found)")
            else:
                # Check for direct match
                if csv_col in table_structure:
                    final_mapping[csv_col] = csv_col
                else:
                    unmapped_csv.append(csv_col)
        
        self.logger.info("Column Mapping:")
        for csv_col, db_col in final_mapping.items():
            self.logger.info(f"  {csv_col} -> {db_col}")
        
        if unmapped_csv:
            self.logger.warning(f"Unmapped CSV columns: {unmapped_csv}")
        
        return final_mapping
    
    def transform_dataframe(self, df: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Transform DataFrame using column mapping"""
        
        # Select only mapped columns and rename them
        mapped_df = df[list(column_mapping.keys())].copy()
        mapped_df = mapped_df.rename(columns=column_mapping)
        
        # Add required columns that might be missing with sensible defaults
        # Add import_timestamp
        mapped_df['import_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add file_source (required NOT NULL field) - use the original filename from gem_id
        if 'gem_id' in mapped_df.columns and 'file_source' not in mapped_df.columns:
            mapped_df['file_source'] = df['file'].values  # Use original file column
        
        # Transform gem_id: extract just the identifier part from filename
        if 'gem_id' in mapped_df.columns:
            # Convert file names like "197BC3_halogen_structural_20250925_210557.csv" 
            # to gem IDs like "197BC3_halogen_structural"
            pattern = r'_\d{8}_\d{6}\.csv$'
            replacement = '_structural'
            mapped_df['gem_id'] = mapped_df['gem_id'].str.replace(pattern, replacement, regex=True)
        
        # Add other required fields with defaults if missing
        required_defaults = {
            'light_source_code': 'U',  # U for Unknown if not provided
            'orientation': 'C',        # C for Center if not provided  
            'scan_number': 1,          # Default scan number
            'processing': 'Unknown',   # Default processing method
            'temporal_sequence': 1     # Default temporal sequence
        }
        
        for col, default_val in required_defaults.items():
            if col not in mapped_df.columns:
                mapped_df[col] = default_val
        
        self.logger.info(f"Transformed DataFrame shape: {mapped_df.shape}")
        self.logger.info(f"Transformed columns: {list(mapped_df.columns)}")
        
        return mapped_df
    
    def smart_import_with_mapping(self, csv_path: str, table_name: str) -> Dict:
        """Import CSV with automatic column mapping"""
        
        self.logger.info(f"Starting import with column mapping: {csv_path} -> {table_name}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        self.logger.info(f"Read CSV: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Get table structure
        table_structure = self.get_table_structure(table_name)
        
        # Create column mapping
        column_mapping = self.create_column_mapping(list(df.columns), table_structure)
        
        if not column_mapping:
            raise ValueError("No column mappings found - cannot import data")
        
        # Transform DataFrame
        transformed_df = self.transform_dataframe(df, column_mapping)
        
        # Check for existing records
        stats = {
            'total_input': len(df),
            'imported': 0,
            'skipped': 0,
            'errors': 0
        }
        
        try:
            # Get existing gem_ids to check for duplicates
            existing_query = f"SELECT DISTINCT gem_id FROM {table_name}"
            try:
                existing_gems = pd.read_sql_query(existing_query, self.conn)['gem_id'].tolist()
            except:
                existing_gems = []
            
            # Filter out existing records
            if existing_gems:
                new_records = transformed_df[~transformed_df['gem_id'].isin(existing_gems)]
                duplicates = transformed_df[transformed_df['gem_id'].isin(existing_gems)]
                
                stats['skipped'] = len(duplicates)
                if len(duplicates) > 0:
                    self.logger.info(f"Skipping {len(duplicates)} existing gem_ids:")
                    for gem_id in duplicates['gem_id'].tolist():
                        self.logger.info(f"  {gem_id}")
            else:
                new_records = transformed_df
            
            # Import new records
            if len(new_records) > 0:
                new_records.to_sql(table_name, self.conn, if_exists='append', index=False)
                stats['imported'] = len(new_records)
                self.conn.commit()
                self.logger.info(f"Successfully imported {len(new_records)} new records")
                
                # Show what was imported
                self.logger.info("Imported gem_ids:")
                for gem_id in new_records['gem_id'].tolist():
                    self.logger.info(f"  {gem_id}")
            else:
                self.logger.info("No new records to import")
                
        except Exception as e:
            self.logger.error(f"Import error: {e}")
            stats['errors'] = len(transformed_df)
            import traceback
            traceback.print_exc()
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# Main execution
if __name__ == "__main__":
    # Your configuration
    DATABASE_PATH = "gemini_structural.db"
    CSV_FILE_PATH = "gemini_structural_unified.csv"
    TABLE_NAME = "structural_features"
    
    # Initialize importer
    importer = ColumnMappingImporter(DATABASE_PATH)
    
    try:
        print("Starting CSV import with automatic column mapping...")
        
        # Import with column mapping
        stats = importer.smart_import_with_mapping(CSV_FILE_PATH, TABLE_NAME)
        
        print(f"\nImport Summary:")
        print(f"  Total input records: {stats['total_input']}")
        print(f"  Records imported: {stats['imported']}")
        print(f"  Records skipped (duplicates): {stats['skipped']}")
        print(f"  Errors: {stats['errors']}")
        
        if stats['imported'] > 0:
            print(f"\nSuccessfully added {stats['imported']} new records to database!")
        elif stats['skipped'] > 0:
            print(f"\nAll records already exist in database - no imports needed")
        else:
            print(f"\nNo records were imported")
        
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        importer.close()