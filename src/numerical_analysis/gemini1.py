#!/usr/bin/env python3
"""
Gemini Numerical Analysis Module - Simplified Standalone Version
Advanced spectral comparison and gem identification system
No external dependencies on config/utils modules
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
from pathlib import Path

class GeminiNumericalAnalyzer:
    """Simplified numerical analysis system for gemological spectral data"""
    
    def __init__(self):
        print("🚀 Initializing Gemini Numerical Analyzer...")
        
        # Get current script location and derive project paths
        current_file = Path(__file__).resolve()
        print(f"📍 Script location: {current_file}")
        
        # Navigate up to find project root
        # current_file is in src/numerical_analysis/gemini1.py
        self.project_root = current_file.parent.parent.parent
        print(f"📁 Project root: {self.project_root}")
        
        # Set up directory paths
        self.data_dir = self.project_root / 'data'
        self.unknown_dir = self.data_dir / 'unknown'         
        self.database_dir = self.project_root / 'database'
        self.reference_spectra_dir = self.database_dir / 'reference_spectra'  
        self.output_dir = self.project_root / 'output' / 'numerical_analysis'
        
        # Print all paths for debugging
        print(f"📂 Data directory: {self.data_dir}")
        print(f"🔍 Unknown directory: {self.unknown_dir}")
        print(f"📚 Database directory: {self.database_dir}")
        print(f"🔬 Reference spectra: {self.reference_spectra_dir}")
        print(f"💾 Output directory: {self.output_dir}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: {self.output_dir}")
        
        # Analysis data storage
        self.unknown_spectra = {}
        self.database_spectra = {}
        self.analysis_results = {}
        
        print("✅ Gemini Numerical Analyzer initialized successfully")
        print("="*60)
    
    def check_file_exists(self, file_path, description="File"):
        """Check if file exists and report status"""
        if file_path.exists():
            print(f"✅ {description} found: {file_path}")
            return True
        else:
            print(f"❌ {description} NOT FOUND: {file_path}")
            return False
    
    def load_unknown_spectra(self):
        """Load unknown gem spectra from the data/unknown directory"""
        print("\n🔍 LOADING UNKNOWN SPECTRA:")
        print("-" * 40)
        
        unknown_files = {
            'B': self.unknown_dir / 'unkgemB.csv',
            'L': self.unknown_dir / 'unkgemL.csv', 
            'U': self.unknown_dir / 'unkgemU.csv'
        }
        
        loaded_sources = []
        
        for light_source, file_path in unknown_files.items():
            print(f"\n📊 Loading {light_source} spectrum...")
            
            if not self.check_file_exists(file_path, f"Unknown {light_source} spectrum"):
                continue
            
            try:
                # Try multiple parsing methods
                df = None
                
                # Method 1: Standard CSV
                try:
                    df = pd.read_csv(file_path, header=None, names=['wavelength', 'intensity'])
                    print(f"   ✅ Loaded using standard CSV parser")
                except:
                    # Method 2: Flexible separator
                    try:
                        df = pd.read_csv(file_path, sep='[\s,]+', header=None, 
                                       names=['wavelength', 'intensity'], skiprows=1, engine='python')
                        print(f"   ✅ Loaded using flexible separator parser")
                    except:
                        # Method 3: Tab separated
                        try:
                            df = pd.read_csv(file_path, sep='\t', header=None, names=['wavelength', 'intensity'])
                            print(f"   ✅ Loaded using tab separator parser")
                        except Exception as e:
                            print(f"   ❌ Failed to parse file: {e}")
                            continue
                
                if df is not None and len(df) > 0:
                    self.unknown_spectra[light_source] = df
                    loaded_sources.append(light_source)
                    print(f"   📈 Data points: {len(df)}")
                    print(f"   📊 Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} nm")
                    print(f"   💡 Intensity range: {df['intensity'].min():.2f} - {df['intensity'].max():.2f}")
                else:
                    print(f"   ❌ File appears to be empty or invalid")
                    
            except Exception as e:
                print(f"   ❌ Error loading {light_source} spectrum: {e}")
        
        print(f"\n✅ Loaded spectra for: {loaded_sources}")
        return loaded_sources
    
    def load_database_spectra(self):
        """Load database spectra for comparison"""
        print("\n📚 LOADING DATABASE SPECTRA:")
        print("-" * 40)
        
        db_files = {
            'B': 'gemini_db_long_B.csv',
            'L': 'gemini_db_long_L.csv',
            'U': 'gemini_db_long_U.csv'
        }
        
        loaded_db = []
        
        for light_source, filename in db_files.items():
            print(f"\n🔬 Loading {light_source} database...")
            
            db_path = self.reference_spectra_dir / filename
            if not self.check_file_exists(db_path, f"Database {light_source} file"):
                # Try alternative locations
                alt_path = self.database_dir / filename
                if self.check_file_exists(alt_path, f"Database {light_source} file (alternative)"):
                    db_path = alt_path
                else:
                    continue
            
            try:
                df = pd.read_csv(db_path)
                self.database_spectra[light_source] = df
                loaded_db.append(light_source)
                
                print(f"   ✅ Database loaded successfully")
                print(f"   📊 Total records: {len(df)}")
                if 'full_name' in df.columns:
                    unique_gems = df['full_name'].nunique()
                    print(f"   💎 Unique gems: {unique_gems}")
                else:
                    print(f"   ⚠️  No 'full_name' column found - may cause issues")
                    print(f"   📋 Available columns: {list(df.columns)}")
                
            except Exception as e:
                print(f"   ❌ Error loading {light_source} database: {e}")
        
        print(f"\n✅ Loaded databases for: {loaded_db}")
        return loaded_db
    
    def compute_match_score(self, unknown_df, reference_df):
        """Compute match score between unknown and reference spectra"""
        try:
            merged = pd.merge(unknown_df, reference_df, on='wavelength', suffixes=('_unknown', '_ref'))
            if len(merged) == 0:
                return float('inf')
            
            # Calculate MSE and convert to log score
            score = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
            log_score = np.log1p(score)
            return log_score
        except Exception as e:
            print(f"      ⚠️ Error computing match score: {e}")
            return float('inf')
    
    def analyze_spectral_matches(self):
        """Main spectral analysis and matching function"""
        print("\n" + "="*60)
        print("🔬 GEMINI SPECTRAL ANALYSIS STARTING")
        print("="*60)
        
        # Load unknown spectra
        available_sources = self.load_unknown_spectra()
        
        if not available_sources:
            print("\n❌ ANALYSIS FAILED: No unknown spectra found")
            print(f"   📁 Please ensure unkgem*.csv files are in: {self.unknown_dir}")
            return False
        
        # Load database spectra
        db_sources = self.load_database_spectra()
        
        if not db_sources:
            print("\n❌ ANALYSIS FAILED: No database spectra found")
            print(f"   📁 Please ensure database files are in: {self.reference_spectra_dir}")
            return False
        
        # Find common light sources
        common_sources = set(available_sources) & set(db_sources)
        if not common_sources:
            print("\n❌ ANALYSIS FAILED: No common light sources between unknown and database")
            print(f"   🔍 Available: {available_sources}")
            print(f"   📚 Database: {db_sources}")
            return False
        
        print(f"\n🎯 ANALYZING {len(common_sources)} LIGHT SOURCES: {sorted(common_sources)}")
        print("="*60)
        
        # Perform analysis for each light source
        all_matches = {}
        gem_best_scores = {}
        gem_best_names = {}
        
        for light_source in sorted(common_sources):
            print(f"\n🔍 ANALYZING {light_source} SPECTRA...")
            print("-" * 30)
            
            unknown_df = self.unknown_spectra[light_source]
            db_df = self.database_spectra[light_source]
            
            gem_names = db_df['full_name'].unique() if 'full_name' in db_df.columns else []
            
            if len(gem_names) == 0:
                print(f"   ❌ No gem names found in database for {light_source}")
                continue
            
            print(f"   📊 Comparing against {len(gem_names)} reference gems...")
            
            scores = []
            for i, gem_name in enumerate(gem_names):
                if i % 100 == 0:  # Progress indicator
                    print(f"   🔄 Progress: {i}/{len(gem_names)} gems processed...")
                
                try:
                    reference_df = db_df[db_df['full_name'] == gem_name]
                    score = self.compute_match_score(unknown_df, reference_df)
                    scores.append((gem_name, score))
                except Exception as e:
                    continue
            
            if not scores:
                print(f"   ❌ No valid scores computed for {light_source}")
                continue
            
            # Sort by score (lower is better)
            sorted_scores = sorted(scores, key=lambda x: x[1])
            all_matches[light_source] = sorted_scores
            
            print(f"\n   🏆 TOP 5 MATCHES for {light_source}:")
            for i, (gem, score) in enumerate(sorted_scores[:5], 1):
                print(f"      {i}. {gem}: Score = {score:.2f}")
            
            # Track best scores per gem ID
            for gem_name, score in sorted_scores:
                base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                if base_id not in gem_best_scores:
                    gem_best_scores[base_id] = {}
                    gem_best_names[base_id] = {}
                if score < gem_best_scores[base_id].get(light_source, float('inf')):
                    gem_best_scores[base_id][light_source] = score
                    gem_best_names[base_id][light_source] = gem_name
        
        # Calculate final rankings
        print(f"\n🏆 COMPUTING FINAL RANKINGS...")
        print("-" * 40)
        
        # Only consider gems with data for all available light sources
        valid_gems = {gid: scores for gid, scores in gem_best_scores.items() 
                     if set(scores.keys()) == common_sources}
        
        if not valid_gems:
            print("❌ No gems found with complete spectral data for all light sources")
            return False
        
        # Calculate aggregated scores
        aggregated_scores = {
            base_id: sum(scores[ls] for ls in common_sources) 
            for base_id, scores in valid_gems.items()
        }
        
        final_rankings = sorted(aggregated_scores.items(), key=lambda x: x[1])
        
        # Display results
        print(f"\n🎉 FINAL IDENTIFICATION RESULTS:")
        print("="*80)
        print(f"📊 Analysis Summary:")
        print(f"   🔍 Light sources analyzed: {', '.join(sorted(common_sources))}")
        print(f"   💎 Total candidates: {len(final_rankings)}")
        print(f"   🏆 Best matches shown below:")
        print("="*80)
        
        for rank, (base_id, total_score) in enumerate(final_rankings[:10], 1):
            sources_analyzed = list(gem_best_scores[base_id].keys())
            
            print(f"\n🏅 RANK {rank}:")
            print(f"   💎 Gem ID: {base_id}")
            print(f"   📊 Total Score: {total_score:.2f} (lower = better match)")
            print(f"   🔍 Light Sources: {', '.join(sorted(sources_analyzed))}")
            print(f"   📋 Individual Scores:")
            
            for ls in sorted(sources_analyzed):
                individual_score = gem_best_scores[base_id][ls]
                gem_name = gem_best_names[base_id][ls]
                print(f"      {ls}: {individual_score:.2f} ({gem_name})")
        
        # Save results
        print(f"\n💾 SAVING RESULTS...")
        self.save_analysis_results(final_rankings, gem_best_scores, gem_best_names, common_sources)
        
        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("="*60)
        
        return True
    
    def save_analysis_results(self, rankings, scores, gem_names, sources):
        """Save analysis results to files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create results DataFrame
            results_data = []
            for rank, (gem_id, total_score) in enumerate(rankings, 1):
                
                result = {
                    'rank': rank,
                    'gem_id': gem_id,
                    'total_score': total_score,
                    'light_sources': ', '.join(sorted(scores[gem_id].keys()))
                }
                
                # Add individual light source scores and names
                for ls in ['B', 'L', 'U']:
                    result[f'{ls}_score'] = scores[gem_id].get(ls, 'N/A')
                    result[f'{ls}_gem_name'] = gem_names[gem_id].get(ls, 'N/A')
                
                results_data.append(result)
            
            # Save to CSV
            results_df = pd.DataFrame(results_data)
            results_file = self.output_dir / f'gemini_analysis_results_{timestamp}.csv'
            results_df.to_csv(results_file, index=False)
            print(f"   ✅ Detailed results: {results_file}")
            
            # Save summary report
            report_file = self.output_dir / f'analysis_summary_{timestamp}.txt'
            with open(report_file, 'w') as f:
                f.write("="*60 + "\n")
                f.write("GEMINI NUMERICAL ANALYSIS SUMMARY\n")
                f.write("="*60 + "\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Light Sources Analyzed: {', '.join(sorted(sources))}\n")
                f.write(f"Total Candidates Analyzed: {len(rankings)}\n\n")
                
                f.write("TOP 10 BEST MATCHES:\n")
                f.write("-" * 40 + "\n")
                for rank, (gem_id, total_score) in enumerate(rankings[:10], 1):
                    f.write(f"Rank {rank}: Gem ID {gem_id}\n")
                    f.write(f"  Total Score: {total_score:.2f}\n")
                    f.write(f"  Light Sources: {', '.join(sorted(scores[gem_id].keys()))}\n")
                    for ls in sorted(scores[gem_id].keys()):
                        individual_score = scores[gem_id][ls]
                        gem_name = gem_names[gem_id][ls]
                        f.write(f"    {ls}: {individual_score:.2f} ({gem_name})\n")
                    f.write("\n")
            
            print(f"   ✅ Summary report: {report_file}")
            
        except Exception as e:
            print(f"   ❌ Error saving results: {e}")

def main():
    """Main execution function"""
    print("🚀 GEMINI NUMERICAL ANALYSIS SYSTEM")
    print("="*60)
    
    try:
        analyzer = GeminiNumericalAnalyzer()
        success = analyzer.analyze_spectral_matches()
        
        if success:
            print(f"\n🎉 SUCCESS! Analysis completed!")
        else:
            print(f"\n❌ FAILED! Analysis could not be completed")
            
        # Keep window open
        input("\n⏸️  Press Enter to close...")
        
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        print("🔍 Please check your data files and directory structure")
        input("\n⏸️  Press Enter to close...")

if __name__ == "__main__":
    main()
