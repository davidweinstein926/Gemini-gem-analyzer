import sqlite3
import pandas as pd
from datetime import datetime
import logging

class GemDatabaseCleaner:
    """
    A tool to clean up duplicate gem records in SQLite database.
    Keeps only unique gems based on identifier, preserving records with different day timestamps.
    """
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def get_table_info(self, table_name):
        """Get information about the table structure"""
        cursor = self.conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        self.logger.info(f"Table '{table_name}' has {len(columns)} columns:")
        for col in columns:
            self.logger.info(f"  {col[1]} ({col[2]})")
        
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_records = cursor.fetchone()[0]
        self.logger.info(f"Total records: {total_records}")
        
        return columns, total_records
    
    def find_duplicates(self, table_name, gem_id_column='gem_id', timestamp_column='timestamp'):
        """
        Find duplicate records based on gem identifier.
        Returns DataFrame with duplicate analysis.
        """
        query = f"""
        SELECT {gem_id_column}, {timestamp_column}, COUNT(*) as count, 
               GROUP_CONCAT(rowid) as rowids,
               MIN(rowid) as min_rowid,
               MAX(rowid) as max_rowid
        FROM {table_name} 
        GROUP BY {gem_id_column}
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        """
        
        duplicates_df = pd.read_sql_query(query, self.conn)
        self.logger.info(f"Found {len(duplicates_df)} gem IDs with duplicates")
        self.logger.info(f"Total duplicate records: {duplicates_df['count'].sum() - len(duplicates_df)}")
        
        return duplicates_df
    
    def analyze_timestamp_differences(self, table_name, gem_id_column='gem_id', timestamp_column='timestamp'):
        """
        Analyze timestamp differences for duplicates to identify legitimate different-day records.
        """
        query = f"""
        SELECT {gem_id_column}, {timestamp_column}, rowid,
               DATE({timestamp_column}) as date_only
        FROM {table_name}
        WHERE {gem_id_column} IN (
            SELECT {gem_id_column} FROM {table_name} 
            GROUP BY {gem_id_column} HAVING COUNT(*) > 1
        )
        ORDER BY {gem_id_column}, {timestamp_column}
        """
        
        timestamp_df = pd.read_sql_query(query, self.conn)
        
        # Group by gem_id and check for different dates
        legitimate_multiples = []
        for gem_id, group in timestamp_df.groupby(gem_id_column):
            unique_dates = group['date_only'].nunique()
            if unique_dates > 1:
                legitimate_multiples.append({
                    'gem_id': gem_id,
                    'unique_dates': unique_dates,
                    'record_count': len(group),
                    'dates': group['date_only'].unique().tolist()
                })
        
        self.logger.info(f"Found {len(legitimate_multiples)} gems with records on different days")
        return legitimate_multiples, timestamp_df
    
    def cleanup_duplicates_smart(self, table_name, gem_id_column='gem_id', 
                                timestamp_column='timestamp', dry_run=True):
        """
        Smart cleanup that preserves records with different day timestamps.
        """
        self.logger.info("Starting smart duplicate cleanup...")
        
        # Get duplicate analysis
        duplicates_df = self.find_duplicates(table_name, gem_id_column, timestamp_column)
        legitimate_multiples, timestamp_df = self.analyze_timestamp_differences(
            table_name, gem_id_column, timestamp_column
        )
        
        # Create set of gem_ids that should have multiple records (different dates)
        legitimate_gem_ids = {item['gem_id'] for item in legitimate_multiples}
        
        rowids_to_delete = []
        records_to_keep_info = []
        
        for _, duplicate_row in duplicates_df.iterrows():
            gem_id = duplicate_row[gem_id_column]
            rowids = [int(x) for x in duplicate_row['rowids'].split(',')]
            
            if gem_id in legitimate_gem_ids:
                # For legitimate multiples, keep one record per unique date
                gem_timestamps = timestamp_df[timestamp_df[gem_id_column] == gem_id]
                
                # Group by date and keep the earliest record (min rowid) for each date
                keep_rowids = []
                for date, date_group in gem_timestamps.groupby('date_only'):
                    min_rowid = date_group['rowid'].min()
                    keep_rowids.append(min_rowid)
                
                # Mark others for deletion
                for rowid in rowids:
                    if rowid not in keep_rowids:
                        rowids_to_delete.append(rowid)
                
                records_to_keep_info.append({
                    'gem_id': gem_id,
                    'kept_rowids': keep_rowids,
                    'deleted_count': len(rowids) - len(keep_rowids),
                    'reason': 'Multiple dates - kept one per date'
                })
            else:
                # Regular duplicates - keep only the first one (minimum rowid)
                keep_rowid = min(rowids)
                delete_rowids = [r for r in rowids if r != keep_rowid]
                rowids_to_delete.extend(delete_rowids)
                
                records_to_keep_info.append({
                    'gem_id': gem_id,
                    'kept_rowids': [keep_rowid],
                    'deleted_count': len(delete_rowids),
                    'reason': 'Same date duplicates - kept earliest'
                })
        
        # Log summary
        total_deletions = len(rowids_to_delete)
        self.logger.info(f"Summary:")
        self.logger.info(f"  Records to delete: {total_deletions}")
        self.logger.info(f"  Gems with legitimate multiple dates: {len(legitimate_gem_ids)}")
        
        if dry_run:
            self.logger.info("DRY RUN - No actual deletions performed")
            self.logger.info("Rowids that would be deleted:")
            for i in range(0, min(50, len(rowids_to_delete))):  # Show first 50
                self.logger.info(f"  {rowids_to_delete[i]}")
            if len(rowids_to_delete) > 50:
                self.logger.info(f"  ... and {len(rowids_to_delete) - 50} more")
        else:
            # Perform actual deletion in batches
            self.delete_records_batch(table_name, rowids_to_delete)
        
        return rowids_to_delete, records_to_keep_info
    
    def delete_records_batch(self, table_name, rowids_to_delete, batch_size=100):
        """Delete records in batches to handle large deletions efficiently"""
        if not rowids_to_delete:
            self.logger.info("No records to delete")
            return
        
        cursor = self.conn.cursor()
        total_deleted = 0
        
        for i in range(0, len(rowids_to_delete), batch_size):
            batch = rowids_to_delete[i:i + batch_size]
            placeholders = ','.join('?' * len(batch))
            
            delete_query = f"DELETE FROM {table_name} WHERE rowid IN ({placeholders})"
            cursor.execute(delete_query, batch)
            
            batch_deleted = cursor.rowcount
            total_deleted += batch_deleted
            
            self.logger.info(f"Batch {i//batch_size + 1}: deleted {batch_deleted} records")
        
        self.conn.commit()
        self.logger.info(f"Total deleted: {total_deleted} records")
    
    def create_backup(self, table_name):
        """Create a backup table before cleanup"""
        backup_table = f"{table_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        cursor = self.conn.cursor()
        cursor.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}")
        self.conn.commit()
        
        self.logger.info(f"Backup created: {backup_table}")
        return backup_table
    
    def get_cleanup_statistics(self, table_name, gem_id_column='gem_id'):
        """Get statistics after cleanup"""
        cursor = self.conn.cursor()
        
        # Total records
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_records = cursor.fetchone()[0]
        
        # Unique gems
        cursor.execute(f"SELECT COUNT(DISTINCT {gem_id_column}) FROM {table_name}")
        unique_gems = cursor.fetchone()[0]
        
        # Remaining duplicates
        cursor.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT {gem_id_column} FROM {table_name} 
                GROUP BY {gem_id_column} HAVING COUNT(*) > 1
            )
        """)
        remaining_duplicates = cursor.fetchone()[0]
        
        stats = {
            'total_records': total_records,
            'unique_gems': unique_gems,
            'remaining_duplicates': remaining_duplicates
        }
        
        self.logger.info("Post-cleanup statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")
        
        return stats
    
    def close(self):
        """Close database connection"""
        self.conn.close()

# Main execution with YOUR EXACT CONFIGURATION
if __name__ == "__main__":
    # Your exact database configuration
    DATABASE_PATH = "gemini_structural.db"
    TABLE_NAME = "structural_features"     # Your actual table name
    GEM_ID_COLUMN = "gem_id"              # Contains IDs like "140BP2_halogen_structural"
    TIMESTAMP_COLUMN = "analysis_date"     # Date when gem was analyzed
    
    # Initialize cleaner
    cleaner = GemDatabaseCleaner(DATABASE_PATH)
    
    try:
        # Get table info
        print("="*60)
        print("DATABASE ANALYSIS")
        print("="*60)
        columns, total_records = cleaner.get_table_info(TABLE_NAME)
        
        # Create backup
        print("\n" + "="*60)
        print("CREATING BACKUP")
        print("="*60)
        backup_table = cleaner.create_backup(TABLE_NAME)
        
        # Analyze duplicates
        print("\n" + "="*60)
        print("ANALYZING DUPLICATES")
        print("="*60)
        
        # First run as dry run
        print("Running DRY RUN analysis...")
        rowids_to_delete, keep_info = cleaner.cleanup_duplicates_smart(
            TABLE_NAME, GEM_ID_COLUMN, TIMESTAMP_COLUMN, dry_run=True
        )
        
        # Ask for confirmation
        if rowids_to_delete:
            print(f"\nReady to delete {len(rowids_to_delete)} duplicate records.")
            confirm = input("Proceed with deletion? (yes/no): ").lower().strip()
            
            if confirm == 'yes':
                print("\n" + "="*60)
                print("PERFORMING CLEANUP")
                print("="*60)
                cleaner.cleanup_duplicates_smart(
                    TABLE_NAME, GEM_ID_COLUMN, TIMESTAMP_COLUMN, dry_run=False
                )
                
                print("\n" + "="*60)
                print("POST-CLEANUP STATISTICS")
                print("="*60)
                cleaner.get_cleanup_statistics(TABLE_NAME, GEM_ID_COLUMN)
            else:
                print("Cleanup cancelled.")
        else:
            print("No duplicates found to clean up.")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleaner.close()
        print("\nDatabase connection closed.")