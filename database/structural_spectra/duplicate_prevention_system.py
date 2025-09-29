import sqlite3
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import hashlib

class DuplicatePreventionSystem:
    """
    Complete system for preventing duplicates and validating data integrity.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_database_constraints(self, table_name: str):
        """Create database-level constraints to prevent duplicates"""
        cursor = self.conn.cursor()
        
        try:
            # 1. Unique constraint on gem_id + analysis_date (allows same gem on different dates)
            constraint_sql = f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_gem_date 
            ON {table_name}(gem_id, DATE(analysis_date))
            """
            cursor.execute(constraint_sql)
            self.logger.info("✓ Created unique constraint: gem_id + analysis_date")
            
            # 2. Unique constraint on file_source (prevents same file being imported twice)
            constraint_sql = f"""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_file_source 
            ON {table_name}(file_source)
            """
            cursor.execute(constraint_sql)
            self.logger.info("✓ Created unique constraint: file_source")
            
            # 3. Check constraint for reasonable wavelength values
            try:
                cursor.execute(f"""
                ALTER TABLE {table_name} ADD CONSTRAINT check_wavelength 
                CHECK (wavelength > 0 AND wavelength < 10000)
                """)
                self.logger.info("✓ Created check constraint: wavelength range")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e):
                    self.logger.info("✓ Wavelength constraint already exists or not needed")
            
            self.conn.commit()
            self.logger.info("✓ All database constraints created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating constraints: {e}")
            # If constraints already exist, that's okay
            if "already exists" not in str(e).lower():
                raise
    
    def validate_csv_before_import(self, csv_path: str, gem_id_col: str = None) -> Dict:
        """Validate CSV file before importing to catch issues early"""
        self.logger.info(f"Validating CSV: {csv_path}")
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            df = pd.read_csv(csv_path)
            validation_results['stats']['total_rows'] = len(df)
            validation_results['stats']['total_columns'] = len(df.columns)
            
            # Auto-detect gem_id column if not provided
            if not gem_id_col:
                potential_cols = [col for col in df.columns if 
                                any(keyword in col.lower() for keyword in ['id', 'gem', 'file', 'name'])]
                if potential_cols:
                    gem_id_col = potential_cols[0]
                    self.logger.info(f"Auto-detected gem_id column: {gem_id_col}")
                else:
                    validation_results['errors'].append("No suitable ID column found")
                    validation_results['valid'] = False
                    return validation_results
            
            # Check for required columns
            if gem_id_col not in df.columns:
                validation_results['errors'].append(f"Required column '{gem_id_col}' not found")
                validation_results['valid'] = False
            
            # Check for internal CSV duplicates
            if gem_id_col in df.columns:
                duplicates = df[df.duplicated(subset=[gem_id_col], keep=False)]
                if len(duplicates) > 0:
                    unique_dupes = duplicates[gem_id_col].nunique()
                    validation_results['warnings'].append(
                        f"Found {len(duplicates)} duplicate rows ({unique_dupes} unique IDs) in CSV"
                    )
                    validation_results['stats']['csv_duplicates'] = len(duplicates)
                
                validation_results['stats']['unique_gems'] = df[gem_id_col].nunique()
            
            # Check for null/empty values in critical columns
            if gem_id_col in df.columns:
                null_gems = df[gem_id_col].isnull().sum()
                if null_gems > 0:
                    validation_results['errors'].append(f"Found {null_gems} rows with null gem IDs")
                    validation_results['valid'] = False
            
            # Check for reasonable data ranges
            numeric_checks = {
                'wavelength': (0, 10000),
                'intensity': (0, None)
            }
            
            for col, (min_val, max_val) in numeric_checks.items():
                if col in df.columns:
                    col_data = pd.to_numeric(df[col], errors='coerce')
                    if min_val is not None and (col_data < min_val).any():
                        validation_results['warnings'].append(f"Found {col} values below {min_val}")
                    if max_val is not None and (col_data > max_val).any():
                        validation_results['warnings'].append(f"Found {col} values above {max_val}")
            
            self.logger.info("CSV validation completed")
            
        except Exception as e:
            validation_results['errors'].append(f"Error reading CSV: {e}")
            validation_results['valid'] = False
        
        return validation_results
    
    def check_database_integrity(self, table_name: str) -> Dict:
        """Check database for duplicates and integrity issues after import"""
        self.logger.info(f"Checking database integrity: {table_name}")
        
        integrity_results = {
            'clean': True,
            'issues': [],
            'stats': {}
        }
        
        try:
            cursor = self.conn.cursor()
            
            # 1. Total record count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_records = cursor.fetchone()[0]
            integrity_results['stats']['total_records'] = total_records
            
            # 2. Check for gem_id duplicates (same gem, same date)
            cursor.execute(f"""
                SELECT gem_id, DATE(analysis_date) as date_part, COUNT(*) as count
                FROM {table_name}
                GROUP BY gem_id, DATE(analysis_date)
                HAVING COUNT(*) > 1
                ORDER BY count DESC
            """)
            date_duplicates = cursor.fetchall()
            
            if date_duplicates:
                integrity_results['clean'] = False
                total_dupes = sum(row[2] - 1 for row in date_duplicates)  # -1 because one copy is legitimate
                integrity_results['issues'].append(f"Found {len(date_duplicates)} gem IDs with same-date duplicates ({total_dupes} extra records)")
                integrity_results['stats']['same_date_duplicates'] = total_dupes
                
                # Show worst offenders
                worst_duplicates = date_duplicates[:5]
                for gem_id, date_part, count in worst_duplicates:
                    integrity_results['issues'].append(f"  {gem_id} on {date_part}: {count} copies")
            
            # 3. Check for file_source duplicates
            cursor.execute(f"""
                SELECT file_source, COUNT(*) as count
                FROM {table_name}
                GROUP BY file_source
                HAVING COUNT(*) > 1
                ORDER BY count DESC
            """)
            file_duplicates = cursor.fetchall()
            
            if file_duplicates:
                integrity_results['clean'] = False
                total_file_dupes = sum(row[1] - 1 for row in file_duplicates)
                integrity_results['issues'].append(f"Found {len(file_duplicates)} duplicate file sources ({total_file_dupes} extra records)")
                integrity_results['stats']['file_duplicates'] = total_file_dupes
            
            # 4. Check for data quality issues
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE wavelength IS NULL OR wavelength <= 0 OR wavelength > 10000
            """)
            bad_wavelengths = cursor.fetchone()[0]
            if bad_wavelengths > 0:
                integrity_results['issues'].append(f"Found {bad_wavelengths} records with invalid wavelengths")
            
            # 5. Statistics
            cursor.execute(f"SELECT COUNT(DISTINCT gem_id) FROM {table_name}")
            unique_gems = cursor.fetchone()[0]
            integrity_results['stats']['unique_gems'] = unique_gems
            integrity_results['stats']['duplication_ratio'] = total_records / unique_gems if unique_gems > 0 else 0
            
            self.logger.info("Database integrity check completed")
            
        except Exception as e:
            integrity_results['issues'].append(f"Error checking integrity: {e}")
            integrity_results['clean'] = False
        
        return integrity_results
    
    def auto_cleanup_duplicates(self, table_name: str, dry_run: bool = True) -> Dict:
        """Automatically clean up duplicates found during integrity check"""
        self.logger.info(f"Auto-cleanup duplicates (dry_run={dry_run})")
        
        cleanup_results = {
            'cleaned': False,
            'deleted_count': 0,
            'kept_count': 0,
            'actions': []
        }
        
        try:
            cursor = self.conn.cursor()
            
            # Find duplicates (same gem_id and same date)
            cursor.execute(f"""
                SELECT gem_id, DATE(analysis_date) as date_part, 
                       GROUP_CONCAT(rowid) as rowids
                FROM {table_name}
                GROUP BY gem_id, DATE(analysis_date)
                HAVING COUNT(*) > 1
            """)
            duplicates = cursor.fetchall()
            
            total_to_delete = 0
            
            for gem_id, date_part, rowids_str in duplicates:
                rowids = [int(x) for x in rowids_str.split(',')]
                # Keep the first (lowest rowid), delete the rest
                keep_rowid = min(rowids)
                delete_rowids = [r for r in rowids if r != keep_rowid]
                
                cleanup_results['actions'].append({
                    'gem_id': gem_id,
                    'date': date_part,
                    'kept_rowid': keep_rowid,
                    'deleted_rowids': delete_rowids
                })
                
                total_to_delete += len(delete_rowids)
                
                if not dry_run:
                    # Delete the duplicates
                    placeholders = ','.join('?' * len(delete_rowids))
                    cursor.execute(f"DELETE FROM {table_name} WHERE rowid IN ({placeholders})", delete_rowids)
            
            if not dry_run and total_to_delete > 0:
                self.conn.commit()
                cleanup_results['cleaned'] = True
                self.logger.info(f"Deleted {total_to_delete} duplicate records")
            
            cleanup_results['deleted_count'] = total_to_delete
            cleanup_results['kept_count'] = len(duplicates)
            
        except Exception as e:
            self.logger.error(f"Error during auto-cleanup: {e}")
            cleanup_results['actions'].append(f"Error: {e}")
        
        return cleanup_results
    
    def generate_integrity_report(self, table_name: str, csv_path: str = None) -> str:
        """Generate a comprehensive integrity report"""
        report = []
        report.append("=" * 80)
        report.append("DATABASE INTEGRITY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Database: {self.db_path}")
        report.append(f"Table: {table_name}")
        
        # CSV validation if provided
        if csv_path:
            report.append("\n" + "-" * 40)
            report.append("CSV VALIDATION")
            report.append("-" * 40)
            validation = self.validate_csv_before_import(csv_path)
            
            if validation['valid']:
                report.append("✓ CSV validation PASSED")
            else:
                report.append("✗ CSV validation FAILED")
            
            for error in validation['errors']:
                report.append(f"  ERROR: {error}")
            for warning in validation['warnings']:
                report.append(f"  WARNING: {warning}")
            
            for key, value in validation['stats'].items():
                report.append(f"  {key}: {value}")
        
        # Database integrity check
        report.append("\n" + "-" * 40)
        report.append("DATABASE INTEGRITY")
        report.append("-" * 40)
        
        integrity = self.check_database_integrity(table_name)
        
        if integrity['clean']:
            report.append("✓ Database integrity CHECK PASSED")
        else:
            report.append("✗ Database integrity CHECK FAILED")
        
        for issue in integrity['issues']:
            report.append(f"  ISSUE: {issue}")
        
        for key, value in integrity['stats'].items():
            report.append(f"  {key}: {value}")
        
        # Recommendations
        report.append("\n" + "-" * 40)
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if not integrity['clean']:
            report.append("1. Run auto-cleanup to remove duplicates")
            report.append("2. Investigate root cause of duplicate creation")
            report.append("3. Implement database constraints if not already present")
        else:
            report.append("✓ No action needed - database is clean")
        
        return '\n'.join(report)
    
    def close(self):
        self.conn.close()

# Complete workflow function
def setup_complete_prevention_system(db_path: str, table_name: str, csv_path: str = None):
    """Set up complete duplicate prevention system"""
    
    system = DuplicatePreventionSystem(db_path)
    
    try:
        print("Setting up complete duplicate prevention system...")
        
        # 1. Create database constraints
        print("\n1. Creating database constraints...")
        system.create_database_constraints(table_name)
        
        # 2. Validate CSV if provided
        if csv_path:
            print(f"\n2. Validating CSV: {csv_path}")
            validation = system.validate_csv_before_import(csv_path)
            if not validation['valid']:
                print("❌ CSV validation failed - fix issues before importing")
                return False
            else:
                print("✓ CSV validation passed")
        
        # 3. Check current database integrity
        print(f"\n3. Checking database integrity...")
        integrity = system.check_database_integrity(table_name)
        
        if not integrity['clean']:
            print("❌ Database has integrity issues")
            print("\n4. Running auto-cleanup (DRY RUN)...")
            cleanup_preview = system.auto_cleanup_duplicates(table_name, dry_run=True)
            
            if cleanup_preview['deleted_count'] > 0:
                confirm = input(f"\nFound {cleanup_preview['deleted_count']} duplicates. Auto-clean them? (yes/no): ")
                if confirm.lower() == 'yes':
                    print("Running auto-cleanup...")
                    cleanup_result = system.auto_cleanup_duplicates(table_name, dry_run=False)
                    print(f"✓ Cleaned up {cleanup_result['deleted_count']} duplicates")
        else:
            print("✓ Database integrity is clean")
        
        # 5. Generate final report
        print("\n5. Generating integrity report...")
        report = system.generate_integrity_report(table_name, csv_path)
        
        # Save report to file
        report_filename = f"integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to: {report_filename}")
        print("\nComplete prevention system is now active!")
        return True
        
    except Exception as e:
        print(f"Error setting up prevention system: {e}")
        return False
    finally:
        system.close()

# Main execution
if __name__ == "__main__":
    # Your configuration
    DATABASE_PATH = "gemini_structural.db"
    TABLE_NAME = "structural_features"
    CSV_PATH = "gemini_structural_unified.csv"  # Optional
    
    # Set up the complete system
    success = setup_complete_prevention_system(DATABASE_PATH, TABLE_NAME, CSV_PATH)
    
    if success:
        print("\n🛡️  Duplicate prevention system is now protecting your database!")
        print("\nNext steps:")
        print("1. All future imports will be protected by database constraints")
        print("2. Run integrity checks periodically with the validation functions") 
        print("3. Use the CSV validation before any imports")
    else:
        print("\n❌ Failed to set up prevention system")