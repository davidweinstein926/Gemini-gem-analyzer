#!/usr/bin/env python3
"""
ENHANCED GEM ANALYZER v2.2 - ULTRA OPTIMIZED - OPTION 8 CONFIGURED
Modified for main.py option 8: structural matching from archive data
Input: root/data/structural(archive)
Output: root/outputs/structural_results/reports (CSV/TXT), /graphs (PNG)
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

class UltraOptimizedGemAnalyzer:
    """ULTRA OPTIMIZED: All-in-one gem analyzer configured for Option 8"""
    
    def __init__(self, db_path=None):
        # OPTION 8: Use project root detection for better path handling
        self.project_root = self.find_project_root()
        
        # Database path configuration
        if db_path is None:
            possible_db_paths = [
                self.project_root / "database" / "structural_spectra" / "multi_structural_gem_data.db",
                Path("../../database/structural_spectra/multi_structural_gem_data.db"),
                Path("../database/structural_spectra/multi_structural_gem_data.db"),
                Path("database/structural_spectra/multi_structural_gem_data.db"),
                Path("multi_structural_gem_data.db"),
            ]
            
            print(f"🔍 Searching for database...")
            print(f"📁 Project root detected: {self.project_root}")
            
            self.db_path = None
            for i, db_path_candidate in enumerate(possible_db_paths, 1):
                print(f"   {i}. Checking: {db_path_candidate}")
                if db_path_candidate.exists():
                    self.db_path = str(db_path_candidate)
                    print(f"✅ Found database: {self.db_path}")
                    break
                else:
                    print(f"   ❌ Not found")
            
            if self.db_path is None:
                self.db_path = str(possible_db_paths[0])
                print(f"⚠️  Database not found, will use: {self.db_path}")
        else:
            self.db_path = db_path
        
        # OPTION 8 MODIFICATION: Input from structural(archive) instead of unknown
        self.archive_path = self.setup_archive_path()
        
        # OPTION 8 MODIFICATION: Output to organized structural_results directories
        self.setup_option8_output_directories()
        
        # Configuration parameters
        self.config = {
            'penalties': {
                'missing_feature': 10.0, 'extra_feature': 10.0, 'uv_missing_peak': 5.0,
                'tolerance_per_nm': 5.0, 'max_tolerance': 20.0
            },
            'tolerances': {
                'peak_top': 2.0, 'trough_bottom': 2.0, 'valley_midpoint': 5.0,
                'trough_start_end': 5.0, 'mound_plateau_start': 7.0,
                'mound_plateau_top': 5.0, 'mound_plateau_end': 7.0
            },
            'uv_params': {
                'reference_wavelength': 811.0, 'reference_expected_intensity': 15.0,
                'minimum_real_peak_intensity': 2.0, 'real_peak_standards': [296.7, 302.1, 415.6, 419.6, 922.7],
                'diagnostic_peaks': {507.0: "Diamond ID (natural=absorb, synthetic=transmit)", 302.0: "Corundum natural vs synthetic"}
            },
            'normalization_schemes': {
                'UV': 'UV_811nm_15000_to_100', 'Halogen': 'Halogen_650nm_50000_to_100', 'Laser': 'Laser_max_50000_to_100'
            },
            'score_thresholds': {
                'excellent': 90.0, 'strong': 75.0, 'moderate': 60.0, 'weak': 40.0
            },
            'light_mapping': {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'},
            'wavelength_fields': ['Wavelength_nm', 'wavelength', 'Wavelength', 'crest_wavelength', 
                                'max_wavelength', 'midpoint_wavelength', 'peak_wavelength', 'Crest', 'Midpoint']
        }
    
    def find_project_root(self) -> Path:
        """Find the project root directory by looking for common indicators"""
        current_path = Path(__file__).parent.absolute()
        
        # Look for common project indicators
        indicators = ["main.py", "src", "database", "data", ".git", "requirements.txt", "README.md"]
        
        for path in [current_path] + list(current_path.parents):
            indicator_count = sum(1 for indicator in indicators if (path / indicator).exists())
            
            if indicator_count >= 2:
                print(f"🎯 Project root detected: {path} (found {indicator_count} indicators)")
                return path
            
            if (path / "database" / "structural_spectra").exists():
                print(f"🎯 Project root found via database directory: {path}")
                return path
        
        print(f"⚠️  Could not detect project root, using current directory: {current_path}")
        return current_path
    
    def setup_archive_path(self):
        """OPTION 8: Setup path to structural archive directory"""
        possible_archive_paths = [
            self.project_root / "data" / "structural(archive)",
            Path("../../data/structural(archive)"),
            Path("../data/structural(archive)"),
            Path("data/structural(archive)"),
        ]
        
        print(f"🔍 Searching for structural archive directory...")
        
        for archive_path_candidate in possible_archive_paths:
            if archive_path_candidate.exists():
                self.archive_path = archive_path_candidate
                print(f"✅ Found structural archive: {self.archive_path}")
                return archive_path_candidate
        
        # Use the primary expected location even if it doesn't exist
        self.archive_path = possible_archive_paths[0]
        print(f"⚠️  Structural archive not found, will use: {self.archive_path}")
        return self.archive_path
    
    def setup_option8_output_directories(self):
        """OPTION 8: Setup output directories for reports and graphs"""
        # Try multiple possible locations for output directories
        possible_output_roots = [
            self.project_root / "outputs" / "structural_results",
            Path("../../outputs/structural_results"),
            Path("../outputs/structural_results"),
            Path("outputs/structural_results"),
            Path("structural_results"),
        ]
        
        # Find or create output root directory
        self.output_root = None
        for output_root_candidate in possible_output_roots:
            parent = output_root_candidate.parent
            if parent.exists() or output_root_candidate.exists():
                self.output_root = output_root_candidate
                break
        
        if self.output_root is None:
            self.output_root = possible_output_roots[0]
        
        # Create required subdirectories
        self.reports_dir = self.output_root / "reports"
        self.graphs_dir = self.output_root / "graphs"
        
        # Ensure directories exist
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            self.graphs_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ OPTION 8 Output directories configured:")
            print(f"   📄 Reports (CSV/TXT): {self.reports_dir}")
            print(f"   📊 Graphs (PNG): {self.graphs_dir}")
        except Exception as e:
            print(f"⚠️  Could not create output directories: {e}")
            # Fallback to current directory
            self.output_root = Path.cwd() / "structural_results"
            self.reports_dir = self.output_root / "reports"
            self.graphs_dir = self.output_root / "graphs"
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            self.graphs_dir.mkdir(parents=True, exist_ok=True)
            print(f"🔄 Using fallback directories:")
            print(f"   📄 Reports: {self.reports_dir}")
            print(f"   📊 Graphs: {self.graphs_dir}")
    
    def check_database_connection(self):
        """Check database connection with better error reporting"""
        try:
            if not Path(self.db_path).exists():
                print(f"❌ Database file not found: {self.db_path}")
                print(f"📍 Expected location: root/database/structural_spectra/multi_structural_gem_data.db")
                return False
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            conn.close()
            
            if table_count == 0:
                print(f"❌ Database appears to be empty (no tables found)")
                return False
                
            print(f"✅ Database connection successful: {table_count} tables found")
            return True
            
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def parse_gem_filename(self, filename: str) -> Dict[str, Union[str, int]]:
        """Parse gem filename with enhanced validation"""
        base_name = Path(filename).stem.split('_')[0] if '_' in Path(filename).stem else Path(filename).stem
        pattern = r'^(.+?)([BLU])([CP])(\d+)$'
        match = re.match(pattern, base_name, re.IGNORECASE)
        
        if match:
            prefix, light, orientation, scan = match.groups()
            return {
                'gem_id': prefix, 'light_source': self.config['light_mapping'].get(light.upper(), 'Unknown'),
                'orientation': orientation.upper(), 'scan_number': int(scan), 'full_identifier': base_name,
                'original_filename': filename, 'is_valid': True
            }
        
        return {
            'gem_id': base_name, 'light_source': 'Unknown', 'orientation': 'Unknown', 'scan_number': 1,
            'full_identifier': base_name, 'original_filename': filename, 'is_valid': False
        }
    
    def load_archive_data_optimized(self, file_path: Path) -> List[Dict]:
        """OPTION 8: Load data from structural archive files"""
        try:
            df = pd.read_csv(file_path)
            features = []
            
            # Peak detection format
            if 'Peak_Number' in df.columns:
                for _, row in df.iterrows():
                    feature = {
                        'feature_type': 'Peak', 'wavelength': row['Wavelength_nm'], 
                        'max_wavelength': row['Wavelength_nm'], 'intensity': row['Intensity'],
                        'prominence': row.get('Prominence', 1.0)
                    }
                    # Add metadata fields
                    for key in ['Normalization_Scheme', 'Reference_Wavelength', 'Light_Source']:
                        if key in row and pd.notna(row[key]):
                            feature[key] = row[key]
                    features.append(feature)
            
            # Structural features format
            elif 'Feature' in df.columns:
                field_mapping = {'Crest': 'crest_wavelength', 'Midpoint': 'midpoint_wavelength', 
                               'Start': 'start_wavelength', 'End': 'end_wavelength'}
                
                for _, row in df.iterrows():
                    feature = {
                        'feature_type': row.get('Feature', 'unknown'),
                        'wavelength': row.get('Wavelength', row.get('Crest')),
                        'intensity': row.get('Intensity', 1.0)
                    }
                    
                    for col, field in field_mapping.items():
                        if col in row:
                            feature[field] = row[col]
                    
                    if 'Normalization_Scheme' in row and pd.notna(row['Normalization_Scheme']):
                        feature['Normalization_Scheme'] = row['Normalization_Scheme']
                    
                    features.append(feature)
            
            return features
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def analyze_archive_files_optimized(self):
        """OPTION 8: Analyze structural files from archive directory"""
        if not self.check_database_connection():
            print("❌ Cannot proceed without database connection")
            return
        
        if not self.archive_path.exists():
            print(f"❌ Structural archive directory not found: {self.archive_path}")
            print("💡 Please ensure the archive directory exists with CSV files to analyze")
            return
        
        csv_files = list(self.archive_path.glob("*.csv"))
        if not csv_files:
            print("❌ No CSV files found in structural archive directory")
            print(f"📍 Checked: {self.archive_path}")
            return
        
        print(f"✅ Found {len(csv_files)} archive files to analyze")
        print(f"📁 Input: {self.archive_path}")
        print(f"📄 Output Reports: {self.reports_dir}")
        print(f"📊 Output Graphs: {self.graphs_dir}")
        
        all_results = {}
        
        for file_path in csv_files:
            try:
                result = self.analyze_archive_file_optimized(file_path)
                all_results[file_path.name] = result
            except Exception as e:
                print(f"❌ Error analyzing {file_path.name}: {e}")
        
        # Generate summary and save results
        self.generate_archive_analysis_summary(all_results)
        
        # Save results automatically
        if all_results:
            print(f"\n💾 Saving analysis results...")
            try:
                results_package = {
                    'results': all_results,
                    'analysis_type': 'archive_structural_analysis',
                    'total_files': len(csv_files),
                    'successful_files': len([r for r in all_results.values() if 'matches' in r and r['matches']])
                }
                
                self.save_analysis_report(results_package, "archive_analysis")
                print(f"📄 Analysis reports saved to: {self.reports_dir}")
                print(f"📊 Graph directory ready: {self.graphs_dir}")
                
            except Exception as e:
                print(f"❌ Error saving results: {e}")
        
        return all_results
    
    def analyze_archive_file_optimized(self, file_path: Path) -> Dict:
        """OPTION 8: Analyze individual archive file"""
        file_info = self.parse_gem_filename(file_path.name)
        print(f"\nAnalyzing archive file: {file_info['original_filename']}")
        print(f"   Gem: {file_info['gem_id']}, Light: {file_info['light_source']}")
        
        # Load archive data
        archive_data = self.load_archive_data_optimized(file_path)
        if not archive_data:
            print("   Could not load archive file data")
            return {'error': 'Could not load data'}
        
        print(f"   Found {len(archive_data)} spectral features")
        
        # Find database matches
        matches = self.find_database_matches_optimized(archive_data, file_info)
        
        if matches:
            print(f"   Found {len(matches)} potential matches:")
            for i, match in enumerate(matches[:3], 1):
                print(f"      {i}. {match['db_gem_id']} - {match['score']:.1f}%")
            if len(matches) > 3:
                print(f"      ... and {len(matches) - 3} more matches")
        else:
            print("   No similar gems found in database")
        
        return {
            'file_info': file_info, 'archive_data': archive_data, 'matches': matches
        }
    
    def find_database_matches_optimized(self, archive_data: List[Dict], file_info: Dict, top_n: int = 10) -> List[Dict]:
        """Find database matches for archive data"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """SELECT file, light_source, wavelength, intensity, feature_group, data_type, 
                             start_wavelength, end_wavelength, midpoint, bottom, normalization_scheme, reference_wavelength
                       FROM structural_features WHERE light_source = ? ORDER BY file, wavelength"""
            
            db_df = pd.read_sql_query(query, conn, params=(file_info['light_source'],))
            conn.close()
            
            if db_df.empty:
                print(f"   No {file_info['light_source']} data found in database")
                return []
            
            matches = []
            
            # Process each database file
            for db_file in db_df['file'].unique():
                file_data = db_df[db_df['file'] == db_file]
                db_file_info = self.parse_gem_filename(db_file)
                
                # Build database features
                db_features = []
                for _, row in file_data.iterrows():
                    feature = {
                        'feature_type': row.get('feature_group', 'unknown'), 'wavelength': row['wavelength'],
                        'intensity': row['intensity'], 'midpoint_wavelength': row.get('midpoint'),
                        'start_wavelength': row.get('start_wavelength'), 'end_wavelength': row.get('end_wavelength'),
                        'crest_wavelength': row['wavelength'], 'max_wavelength': row['wavelength']
                    }
                    db_features.append(feature)
                
                # Calculate match score
                score = self.match_features_by_light_source_optimized(
                    archive_data, db_features, file_info['light_source'], file_info['gem_id'], db_file_info['gem_id']
                )
                
                if score > 0:
                    matches.append({
                        'db_gem_id': db_file_info['gem_id'], 'db_full_id': db_file_info['full_identifier'],
                        'score': score, 'db_features': len(db_features), 'light_source': db_file_info['light_source'],
                        'orientation': db_file_info['orientation'], 'scan_number': db_file_info['scan_number']
                    })
            
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[:top_n]
            
        except Exception as e:
            print(f"Error finding matches: {e}")
            return []
    
    def generate_archive_analysis_summary(self, all_results: Dict):
        """Generate summary for archive analysis"""
        print(f"\nARCHIVE ANALYSIS SUMMARY:")
        print("=" * 50)
        
        # Categorize results
        categories = {'excellent': [], 'strong': [], 'moderate': [], 'weak': [], 'no_match': []}
        thresholds = self.config['score_thresholds']
        
        for filename, result in all_results.items():
            if 'error' in result:
                categories['no_match'].append((filename, result['error']))
            elif result.get('matches'):
                best_score = result['matches'][0]['score']
                if best_score >= thresholds['excellent']:
                    categories['excellent'].append((filename, result))
                elif best_score >= thresholds['strong']:
                    categories['strong'].append((filename, result))
                elif best_score >= thresholds['moderate']:
                    categories['moderate'].append((filename, result))
                else:
                    categories['weak'].append((filename, result))
            else:
                categories['no_match'].append((filename, "No matches found"))
        
        # Display results by category
        category_descriptions = {
            'excellent': f"EXCELLENT MATCHES (≥{thresholds['excellent']}%)",
            'strong': f"STRONG MATCHES (≥{thresholds['strong']}%)",
            'moderate': f"MODERATE MATCHES (≥{thresholds['moderate']}%)",
            'weak': f"WEAK MATCHES (<{thresholds['moderate']}%)",
            'no_match': "NO MATCHES"
        }
        
        for category, description in category_descriptions.items():
            items = categories[category]
            if items:
                print(f"\n{description}:")
                for filename, data in items:
                    if category == 'no_match' and isinstance(data, str):
                        print(f"   • {filename}: {data}")
                    elif category != 'no_match':
                        file_info = data['file_info']
                        best_match = data['matches'][0]
                        print(f"   • {filename} ({file_info['gem_id']}) → {best_match['db_gem_id']} ({best_match['score']:.1f}%)")
    
    def save_analysis_report(self, results: Dict, report_type: str = "analysis") -> str:
        """Save analysis results as CSV and TXT reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"structural_analysis_{report_type}_{timestamp}"
        
        # Generate CSV report
        csv_path = self.reports_dir / f"{base_filename}.csv"
        try:
            self.save_csv_report(results, csv_path)
            print(f"✅ CSV report saved: {csv_path}")
        except Exception as e:
            print(f"❌ Error saving CSV report: {e}")
        
        # Generate TXT report
        txt_path = self.reports_dir / f"{base_filename}.txt"
        try:
            self.save_txt_report(results, txt_path)
            print(f"✅ TXT report saved: {txt_path}")
        except Exception as e:
            print(f"❌ Error saving TXT report: {e}")
        
        return str(csv_path)
    
    def save_csv_report(self, results: Dict, csv_path: Path):
        """Save detailed CSV report of analysis results"""
        csv_data = []
        
        if isinstance(results, dict) and 'results' in results:
            for filename, result in results['results'].items():
                if 'error' in result:
                    csv_data.append({
                        'Filename': filename,
                        'Status': 'Error',
                        'Error': result['error'],
                        'Gem_ID': '',
                        'Light_Source': '',
                        'Best_Match': '',
                        'Score': 0
                    })
                elif result.get('matches'):
                    file_info = result['file_info']
                    best_match = result['matches'][0]
                    csv_data.append({
                        'Filename': filename,
                        'Status': 'Success',
                        'Error': '',
                        'Gem_ID': file_info['gem_id'],
                        'Light_Source': file_info['light_source'],
                        'Best_Match': best_match['db_gem_id'],
                        'Score': best_match['score'],
                        'Features_Found': len(result.get('archive_data', [])),
                        'DB_Features': best_match['db_features']
                    })
                else:
                    file_info = result['file_info']
                    csv_data.append({
                        'Filename': filename,
                        'Status': 'No Matches',
                        'Error': '',
                        'Gem_ID': file_info['gem_id'],
                        'Light_Source': file_info['light_source'],
                        'Best_Match': 'None',
                        'Score': 0
                    })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
    
    def save_txt_report(self, results: Dict, txt_path: Path):
        """Save detailed TXT report of analysis results"""
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("GEMINI STRUCTURAL ANALYSIS REPORT - OPTION 8\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Type: Archive Structural Analysis\n")
            f.write(f"Input Directory: {self.archive_path}\n")
            f.write(f"Database: {self.db_path}\n")
            f.write(f"Reports Directory: {self.reports_dir}\n")
            f.write(f"Graphs Directory: {self.graphs_dir}\n")
            f.write("\n")
            
            if isinstance(results, dict) and 'results' in results:
                f.write("ANALYSIS RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                success_count = 0
                total_count = len(results['results'])
                
                for filename, result in results['results'].items():
                    f.write(f"\nFile: {filename}\n")
                    
                    if 'error' in result:
                        f.write(f"  Status: ERROR - {result['error']}\n")
                    elif result.get('matches'):
                        success_count += 1
                        file_info = result['file_info']
                        best_match = result['matches'][0]
                        
                        f.write(f"  Gem ID: {file_info['gem_id']}\n")
                        f.write(f"  Light Source: {file_info['light_source']}\n")
                        f.write(f"  Features Found: {len(result.get('archive_data', []))}\n")
                        f.write(f"  Best Match: {best_match['db_gem_id']} ({best_match['score']:.1f}%)\n")
                        
                        if len(result['matches']) > 1:
                            f.write("  Other Matches:\n")
                            for i, match in enumerate(result['matches'][1:4], 2):
                                f.write(f"    {i}. {match['db_gem_id']} ({match['score']:.1f}%)\n")
                    else:
                        file_info = result['file_info']
                        f.write(f"  Gem ID: {file_info['gem_id']}\n")
                        f.write(f"  Light Source: {file_info['light_source']}\n")
                        f.write(f"  Status: No matches found\n")
                
                f.write(f"\nSUMMARY:\n")
                f.write(f"  Total files analyzed: {total_count}\n")
                f.write(f"  Successful matches: {success_count}\n")
                f.write(f"  Success rate: {100*success_count/total_count:.1f}%\n")
    
    # Include simplified versions of the matching methods (keeping core functionality)
    def match_features_by_light_source_optimized(self, archive_data, db_features, light_source, archive_gem_id, db_gem_id):
        """Simplified matching for archive analysis"""
        if not archive_data or not db_features:
            return 0.0
            
        if archive_gem_id == db_gem_id:
            return 100.0
        
        # Basic wavelength matching
        total_score = 0.0
        matched_count = 0
        
        for archive_feature in archive_data:
            archive_wl = archive_feature.get('wavelength')
            if archive_wl is None:
                continue
                
            best_score = 0.0
            for db_feature in db_features:
                db_wl = db_feature.get('wavelength')
                if db_wl is None:
                    continue
                    
                diff = abs(archive_wl - db_wl)
                if diff <= 2.0:
                    score = max(0.0, 100.0 - (diff * 25))
                    best_score = max(best_score, score)
            
            total_score += best_score
            matched_count += 1
        
        return total_score / matched_count if matched_count > 0 else 0.0
    
    def run_option8_analysis(self):
        """Main entry point for Option 8 analysis"""
        print("ENHANCED GEM ANALYZER - OPTION 8: STRUCTURAL ARCHIVE ANALYSIS")
        print("=" * 70)
        print(f"📁 Input: {self.archive_path}")
        print(f"📄 Reports Output: {self.reports_dir}")
        print(f"📊 Graphs Output: {self.graphs_dir}")
        print("=" * 70)
        
        return self.analyze_archive_files_optimized()

def main():
    """Main entry point for Option 8"""
    analyzer = UltraOptimizedGemAnalyzer()
    analyzer.run_option8_analysis()

if __name__ == "__main__":
    main()