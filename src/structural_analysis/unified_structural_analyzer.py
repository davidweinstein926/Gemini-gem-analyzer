#!/usr/bin/env python3
"""
COMPLETE UNIFIED STRUCTURAL ANALYZER
Comprehensive diagnostic + full analysis functionality
Save as: src/structural_analysis/unified_structural_analyzer.py

Usage:
    python unified_structural_analyzer.py current    # Analyze current work files
    python unified_structural_analyzer.py archive    # Analyze archived files
    python unified_structural_analyzer.py diagnostic # Run diagnostic mode only
"""

import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re
import sys
import traceback
import json
import warnings
warnings.filterwarnings('ignore')

# Optional imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
    print("✅ Matplotlib available")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ Matplotlib not available - plots will be skipped")

try:
    import seaborn as sns
    HAS_SEABORN = True
    print("✅ Seaborn available")
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Seaborn not available - using basic matplotlib styling")

try:
    from scipy import stats
    from scipy.signal import find_peaks
    HAS_SCIPY = True
    print("✅ SciPy available")
except ImportError:
    HAS_SCIPY = False
    print("⚠️ SciPy not available - using basic peak detection")

print("🔬 COMPLETE UNIFIED STRUCTURAL ANALYZER STARTING")
print("=" * 60)

try:
    print(f"📍 Script location: {__file__}")
    print(f"📍 Current working directory: {Path.cwd()}")
    print(f"📍 Python executable: {sys.executable}")
    print(f"📍 Command line arguments: {sys.argv}")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        print(f"✅ Mode argument: '{mode}'")
    else:
        print("❌ No mode argument provided!")
        print("💡 Usage: python unified_structural_analyzer.py current|archive|diagnostic")
        sys.exit(1)
    
    if mode not in ["current", "archive", "diagnostic"]:
        print(f"❌ Invalid mode: '{mode}'")
        print("💡 Must be 'current', 'archive', or 'diagnostic'")
        sys.exit(1)

    class CompleteUnifiedStructuralAnalyzer:
        """Complete structural analyzer with diagnostic and analysis capabilities"""
        
        def __init__(self, mode="current"):
            print(f"\n🔬 Initializing CompleteUnifiedStructuralAnalyzer")
            print(f"   Mode: {mode}")
            
            try:
                # Calculate project root from src/structural_analysis/ location
                script_dir = Path(__file__).parent  # src/structural_analysis
                src_dir = script_dir.parent  # src
                self.project_root = src_dir.parent  # project root
                
                print(f"   Script dir: {script_dir}")
                print(f"   Src dir: {src_dir}")
                print(f"   Project root: {self.project_root}")
                
                self.mode = mode
                
                # Set source directory based on mode
                if mode == "current":
                    self.source_dir = self.project_root / "data" / "structural_data"
                    self.analysis_name = "Current Work Analysis"
                    self.description = "work-in-progress files (not yet in database)"
                elif mode == "archive":
                    self.source_dir = self.project_root / "data" / "structural(archive)"
                    self.analysis_name = "Archived Work Analysis"
                    self.description = "completed files (already in database)"
                else:  # diagnostic
                    self.source_dir = self.project_root / "data" / "structural_data"
                    self.analysis_name = "Diagnostic Mode"
                    self.description = "testing system functionality"
                
                print(f"   Source directory: {self.source_dir}")
                print(f"   Analysis name: {self.analysis_name}")
                
                # Common directories
                self.unknown_dir = self.project_root / "data" / "unknown" / "structural"
                self.raw_archive_dir = self.project_root / "data" / "raw (archive)"
                self.db_path = self.project_root / "database" / "structural_spectra" / "multi_structural_gem_data.db"
                self.reports_dir = self.project_root / "outputs" / "structural_results" / "reports"
                self.graphs_dir = self.project_root / "outputs" / "structural_results" / "graphs"
                
                print(f"   Unknown dir: {self.unknown_dir}")
                print(f"   Reports dir: {self.reports_dir}")
                print(f"   Database: {self.db_path}")
                
                # Ensure directories exist
                self.unknown_dir.mkdir(parents=True, exist_ok=True)
                self.reports_dir.mkdir(parents=True, exist_ok=True)
                self.graphs_dir.mkdir(parents=True, exist_ok=True)
                
                # Analysis results storage
                self.analysis_results = {}
                self.database_matches = []
                self.spectral_features = {}
                
                print(f"✅ Initialization completed successfully")
                
            except Exception as e:
                print(f"❌ Error in __init__: {e}")
                traceback.print_exc()
                raise
        
        def check_source_directory(self):
            """Check if source directory exists and has files"""
            print(f"\n🔍 Checking source directory: {self.source_dir}")
            
            if not self.source_dir.exists():
                print(f"❌ Source directory does not exist: {self.source_dir}")
                return False
            
            csv_files = list(self.source_dir.glob("*.csv"))
            print(f"📄 Found {len(csv_files)} CSV files in source directory")
            
            if not csv_files:
                print(f"❌ No CSV files found in {self.source_dir}")
                if self.mode == "current":
                    print("💡 Use Option 2 to mark structural features first")
                else:
                    print("💡 Use Option 6 to import files to archive first")
                return False
            
            # Show first few files for verification
            print(f"📋 First few CSV files found:")
            for i, file_path in enumerate(csv_files[:5], 1):
                size_kb = file_path.stat().st_size / 1024
                print(f"   {i}. {file_path.name} ({size_kb:.1f} KB)")
            
            if len(csv_files) > 5:
                print(f"   ... and {len(csv_files) - 5} more files")
            
            return True
        
        def parse_gem_info(self, filename):
            """Parse gem ID and light source from filename"""
            
            # Handle structured format with explicit light source names
            if '_halogen_' in filename.lower():
                prefix = filename.split('_halogen_')[0]
                # Extract base gem ID from prefix like "140BP2" -> "140"
                match = re.match(r'^(\d+)', prefix)
                if match:
                    gem_id = match.group(1)
                else:
                    gem_id = prefix  # fallback
                light_source = 'Halogen'
                return gem_id, light_source
                
            elif '_laser_' in filename.lower():
                prefix = filename.split('_laser_')[0]
                # Extract base gem ID from prefix like "140LC1" -> "140"
                match = re.match(r'^(\d+)', prefix)
                if match:
                    gem_id = match.group(1)
                else:
                    gem_id = prefix  # fallback
                light_source = 'Laser'
                return gem_id, light_source
                
            elif '_uv_' in filename.lower():
                prefix = filename.split('_uv_')[0]
                # Extract base gem ID from prefix like "140UC1" -> "140"
                match = re.match(r'^(\d+)', prefix)
                if match:
                    gem_id = match.group(1)
                else:
                    gem_id = prefix  # fallback
                light_source = 'UV'
                return gem_id, light_source
                
            else:
                # Standard format: 58BC1, 58LC1, 58UC1 -> Gem "58"
                # Pattern: [letters/numbers][B/L/U][orientation][scan]
                match = re.match(r'^([A-Za-z]*\d+)([BLU])([CP]?)(\d+)', filename)
                if match:
                    base_part, light, orientation, scan = match.groups()
                    light_mapping = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                    
                    # Extract numeric part for gem ID
                    gem_match = re.search(r'(\d+)
        
        def group_files_by_gem(self, csv_files):
            """Group CSV files by gem ID and light source"""
            print(f"\n🧪 Grouping files by gem ID and light source...")
            print(f"🔍 Debugging file parsing:")
            
            gem_groups = defaultdict(lambda: {'B': [], 'L': [], 'U': []})
            
            for file_path in csv_files[:10]:  # Debug first 10 files
                filename = file_path.name
                gem_id, light_source = self.parse_gem_info(filename)
                
                print(f"   📄 {filename}")
                print(f"      → Gem ID: '{gem_id}', Light: '{light_source}'")
                
                if gem_id and light_source:
                    light_code = {'Halogen': 'B', 'Laser': 'L', 'UV': 'U'}.get(light_source)
                    if light_code:
                        gem_groups[gem_id][light_code].append(file_path)
                        print(f"      → Added to group: {gem_id}[{light_code}]")
            
            # Process remaining files without debug output
            for file_path in csv_files[10:]:
                gem_id, light_source = self.parse_gem_info(file_path.name)
                if gem_id and light_source:
                    light_code = {'Halogen': 'B', 'Laser': 'L', 'UV': 'U'}.get(light_source)
                    if light_code:
                        gem_groups[gem_id][light_code].append(file_path)
            
            print(f"\n📊 Found {len(gem_groups)} unique gems")
            
            # Show complete vs partial gems
            complete_gems = []
            partial_gems = []
            
            for gem_id, light_files in gem_groups.items():
                b_count = len(light_files['B'])
                l_count = len(light_files['L'])
                u_count = len(light_files['U'])
                
                if b_count > 0 and l_count > 0 and u_count > 0:
                    complete_gems.append((gem_id, b_count, l_count, u_count))
                else:
                    partial_gems.append((gem_id, b_count, l_count, u_count))
            
            print(f"🟢 Complete gems (B+L+U): {len(complete_gems)}")
            for gem_id, b, l, u in complete_gems[:5]:
                print(f"   {gem_id}: B({b}), L({l}), U({u})")
            
            print(f"🟡 Partial gems: {len(partial_gems)}")
            for gem_id, b, l, u in partial_gems[:3]:
                coverage = []
                if b > 0: coverage.append(f"B({b})")
                if l > 0: coverage.append(f"L({l})")
                if u > 0: coverage.append(f"U({u})")
                print(f"   {gem_id}: {', '.join(coverage)}")
            
            return gem_groups
        
        def interactive_gem_selection(self, gem_groups):
            """Interactive gem selection for analysis"""
            print(f"\n🎯 Available gems for analysis:")
            print("=" * 50)
            
            # Separate complete and partial gems
            complete_gems = []
            partial_gems = []
            
            for gem_id, light_files in sorted(gem_groups.items()):
                b_count = len(light_files['B'])
                l_count = len(light_files['L'])
                u_count = len(light_files['U'])
                total = b_count + l_count + u_count
                
                if b_count > 0 and l_count > 0 and u_count > 0:
                    complete_gems.append((gem_id, light_files, b_count, l_count, u_count, total))
                else:
                    partial_gems.append((gem_id, light_files, b_count, l_count, u_count, total))
            
            gem_options = []
            option_num = 1
            
            # Show complete gems first (recommended)
            if complete_gems:
                print(f"🟢 COMPLETE GEMS (Recommended - B+L+U coverage):")
                for gem_id, light_files, b_count, l_count, u_count, total in complete_gems:
                    print(f"{option_num:2d}. 🟢 Gem {gem_id}: B({b_count})+L({l_count})+U({u_count}) = {total} files [COMPLETE]")
                    gem_options.append((gem_id, light_files))
                    option_num += 1
                print()
            
            # Show partial gems  
            if partial_gems:
                print(f"🟡 PARTIAL GEMS (Limited analysis possible):")
                for gem_id, light_files, b_count, l_count, u_count, total in partial_gems[:20]:  # Limit to first 20
                    coverage = []
                    if b_count > 0: coverage.append(f"B({b_count})")
                    if l_count > 0: coverage.append(f"L({l_count})")
                    if u_count > 0: coverage.append(f"U({u_count})")
                    
                    print(f"{option_num:2d}. 🟡 Gem {gem_id}: {'+'.join(coverage)} = {total} files [Partial]")
                    gem_options.append((gem_id, light_files))
                    option_num += 1
                
                if len(partial_gems) > 20:
                    print(f"    ... and {len(partial_gems) - 20} more partial gems")
            
            if not gem_options:
                print("❌ No valid gem options found")
                return None
            
            # Recommend complete gems
            if complete_gems:
                print(f"\n💡 RECOMMENDATION: Choose a complete gem (1-{len(complete_gems)}) for best analysis")
                print(f"   Complete gems have all three light sources (B+L+U) like your example:")
                print(f"   Gem 58: 58BC1 + 58LC1 + 58UC1")
            
            # For diagnostic mode, auto-select first complete gem if available
            if self.mode == "diagnostic":
                if complete_gems:
                    selected_gem_id, selected_files = gem_options[0]
                    print(f"\n🤖 DIAGNOSTIC MODE: Auto-selecting first complete gem...")
                    print(f"   Selected: Gem {selected_gem_id} (Complete)")
                else:
                    selected_gem_id, selected_files = gem_options[0]
                    print(f"\n🤖 DIAGNOSTIC MODE: Auto-selecting first available gem...")
                    print(f"   Selected: Gem {selected_gem_id} (Partial)")
            else:
                # Interactive selection
                print(f"\n🎯 Interactive Gem Selection")
                print(f"   Please select a gem number from the list above")
                
                # Try multiple input attempts
                max_attempts = 3
                selected_gem_id, selected_files = None, None
                
                for attempt in range(max_attempts):
                    try:
                        raw_input = input(f"\nSelect gem number for analysis (1-{len(gem_options)}, or 'q' to quit): ")
                        
                        # Handle InputSubmission wrapper (same as main.py safe_input)
                        choice = str(raw_input).strip()
                        if "InputSubmission(data='" in choice and choice.endswith("')"):
                            choice = choice[22:-2]
                        choice = choice.replace('\\n', '').replace('\n', '').strip("'\"")
                        
                        print(f"   User input received: '{choice}'")  # Debug output
                        
                        if choice.lower() == 'q':
                            print("   Analysis cancelled by user")
                            return None
                        
                        gem_num = int(choice)
                        if 1 <= gem_num <= len(gem_options):
                            selected_gem_id, selected_files = gem_options[gem_num - 1]
                            print(f"✅ Selected: Gem {selected_gem_id}")
                            break
                        else:
                            print(f"❌ Invalid selection. Choose 1-{len(gem_options)}")
                            
                    except ValueError:
                        print("❌ Please enter a number or 'q' to quit")
                    except EOFError:
                        print("⚠️ Input stream ended - using auto-selection")
                        selected_gem_id, selected_files = gem_options[0]
                        print(f"✅ Auto-selected: Gem {selected_gem_id}")
                        break
                    except Exception as e:
                        print(f"⚠️ Input error: {e}")
                        
                    if attempt == max_attempts - 1:
                        print(f"⚠️ Max input attempts reached - auto-selecting first gem")
                        selected_gem_id, selected_files = gem_options[0]
                        print(f"✅ Auto-selected: Gem {selected_gem_id}")
                
                if not selected_gem_id:
                    print("❌ No gem selected")
                    return None
            
            # Select files for each light source
            stone_of_interest = {}
            light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
            
            for light_code in ['B', 'L', 'U']:
                files = selected_files[light_code]
                if files:
                    # Take the first file for each light source
                    stone_of_interest[light_code] = files[0]
                    print(f"   ✅ {light_names[light_code]} ({light_code}): {files[0].name}")
                else:
                    print(f"   ❌ {light_names[light_code]} ({light_code}): No files available")
            
            if not stone_of_interest:
                print("❌ No files selected for analysis")
                return None
            
            return selected_gem_id, stone_of_interest
        
        def load_spectral_data(self, file_path):
            """Load and validate spectral data from CSV file"""
            try:
                print(f"   📄 Loading: {file_path.name}")
                df = pd.read_csv(file_path)
                
                # Try to identify wavelength and intensity columns
                if 'Wavelength' in df.columns and 'Intensity' in df.columns:
                    wavelength = df['Wavelength'].values
                    intensity = df['Intensity'].values
                elif len(df.columns) >= 2:
                    # Assume first two columns are wavelength and intensity
                    wavelength = df.iloc[:, 0].values
                    intensity = df.iloc[:, 1].values
                else:
                    print(f"      ❌ Cannot identify wavelength/intensity columns")
                    return None, None
                
                # Validate data
                if len(wavelength) < 10:
                    print(f"      ⚠️ Warning: Only {len(wavelength)} data points")
                
                print(f"      ✅ Loaded {len(wavelength)} data points")
                print(f"      📊 Range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
                
                return wavelength, intensity
                
            except Exception as e:
                print(f"      ❌ Error loading {file_path.name}: {e}")
                return None, None
        
        def extract_spectral_features(self, wavelength, intensity, light_source):
            """Extract key spectral features from the data"""
            try:
                # Find peaks using scipy if available, otherwise use simple method
                if HAS_SCIPY:
                    peaks, peak_properties = find_peaks(intensity, height=np.percentile(intensity, 75), distance=5)
                    valleys, valley_properties = find_peaks(-intensity, height=-np.percentile(intensity, 25), distance=5)
                else:
                    # Simple peak detection without scipy
                    threshold = np.percentile(intensity, 75)
                    peaks = []
                    for i in range(1, len(intensity) - 1):
                        if intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1] and intensity[i] > threshold:
                            peaks.append(i)
                    
                    # Simple valley detection
                    valley_threshold = np.percentile(intensity, 25)
                    valleys = []
                    for i in range(1, len(intensity) - 1):
                        if intensity[i] < intensity[i-1] and intensity[i] < intensity[i+1] and intensity[i] < valley_threshold:
                            valleys.append(i)
                
                # Calculate statistical features
                features = {
                    'light_source': light_source,
                    'data_points': len(wavelength),
                    'wavelength_range': [float(wavelength.min()), float(wavelength.max())],
                    'intensity_stats': {
                        'mean': float(np.mean(intensity)),
                        'std': float(np.std(intensity)),
                        'max': float(np.max(intensity)),
                        'min': float(np.min(intensity))
                    },
                    'peaks': {
                        'count': len(peaks),
                        'positions': [float(wavelength[i]) for i in peaks] if len(peaks) > 0 else [],
                        'intensities': [float(intensity[i]) for i in peaks] if len(peaks) > 0 else []
                    },
                    'valleys': {
                        'count': len(valleys),
                        'positions': [float(wavelength[i]) for i in valleys] if len(valleys) > 0 else [],
                        'intensities': [float(intensity[i]) for i in valleys] if len(valleys) > 0 else []
                    }
                }
                
                print(f"      🔍 Features: {len(peaks)} peaks, {len(valleys)} valleys")
                
                return features
                
            except Exception as e:
                print(f"      ❌ Feature extraction error: {e}")
                return None
        
        def query_database_matches(self, gem_id):
            """Query database for similar gems"""
            print(f"\n🗄️ Querying database for matches...")
            
            matches = []
            
            try:
                if self.db_path.exists():
                    with sqlite3.connect(self.db_path) as conn:
                        # Try different table structures
                        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                        print(f"   📋 Database tables: {', '.join(tables['name'].tolist())}")
                        
                        # Query for similar gems
                        for table_name in tables['name']:
                            try:
                                query = f"SELECT * FROM {table_name} WHERE gem_id LIKE '%{gem_id}%' OR gem_id LIKE '%{gem_id[:3]}%' LIMIT 10"
                                results = pd.read_sql_query(query, conn)
                                
                                if not results.empty:
                                    print(f"   ✅ Found {len(results)} matches in {table_name}")
                                    matches.extend(results.to_dict('records'))
                                    
                            except Exception as e:
                                print(f"   ⚠️ Query error for {table_name}: {e}")
                                
                else:
                    print(f"   ⚠️ Database not found: {self.db_path}")
                    
            except Exception as e:
                print(f"   ❌ Database query error: {e}")
            
            print(f"   📊 Total matches found: {len(matches)}")
            return matches
        
        def generate_comparison_plots(self, stone_data, gem_id):
            """Generate comparison plots for the spectral data"""
            print(f"\n📈 Generating comparison plots...")
            
            if not HAS_MATPLOTLIB:
                print(f"   ⚠️ Matplotlib not available - skipping plot generation")
                return None
            
            try:
                # Set up the plot style
                plt.style.use('default')
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Structural Analysis: Gem {gem_id} ({self.analysis_name})', fontsize=16, fontweight='bold')
                
                # Color mapping for light sources
                colors = {'B': 'blue', 'L': 'red', 'U': 'purple'}
                light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                
                # Plot 1: Overlaid spectra
                ax1 = axes[0, 0]
                for light_code, (wavelength, intensity) in stone_data.items():
                    if wavelength is not None and intensity is not None:
                        ax1.plot(wavelength, intensity, color=colors[light_code], 
                               label=f'{light_names[light_code]} ({light_code})', linewidth=2)
                
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('Intensity')
                ax1.set_title('Multi-Source Spectral Overlay')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Peak analysis
                ax2 = axes[0, 1]
                peak_data = []
                for light_code in stone_data:
                    if light_code in self.spectral_features:
                        features = self.spectral_features[light_code]
                        if features and 'peaks' in features:
                            peak_positions = features['peaks']['positions']
                            peak_intensities = features['peaks']['intensities']
                            if peak_positions and peak_intensities:
                                ax2.scatter(peak_positions, peak_intensities, 
                                          color=colors[light_code], label=f'{light_names[light_code]} Peaks',
                                          s=50, alpha=0.7)
                                peak_data.extend(peak_positions)
                
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Peak Intensity')
                ax2.set_title('Identified Peaks Comparison')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Statistical comparison
                ax3 = axes[1, 0]
                stats_data = {'Light Source': [], 'Mean Intensity': [], 'Std Dev': [], 'Peak Count': []}
                for light_code in stone_data:
                    if light_code in self.spectral_features:
                        features = self.spectral_features[light_code]
                        if features:
                            stats_data['Light Source'].append(light_names[light_code])
                            stats_data['Mean Intensity'].append(features['intensity_stats']['mean'])
                            stats_data['Std Dev'].append(features['intensity_stats']['std'])
                            stats_data['Peak Count'].append(features['peaks']['count'])
                
                if stats_data['Light Source']:
                    x_pos = np.arange(len(stats_data['Light Source']))
                    ax3.bar(x_pos, stats_data['Mean Intensity'], alpha=0.7, 
                           color=[colors[k] for k in stone_data.keys() if k in colors])
                    ax3.set_xticks(x_pos)
                    ax3.set_xticklabels(stats_data['Light Source'])
                    ax3.set_ylabel('Mean Intensity')
                    ax3.set_title('Statistical Comparison')
                    ax3.grid(True, alpha=0.3)
                
                # Plot 4: Feature summary
                ax4 = axes[1, 1]
                feature_summary = []
                for light_code in stone_data:
                    if light_code in self.spectral_features:
                        features = self.spectral_features[light_code]
                        if features:
                            feature_summary.append([
                                light_names[light_code],
                                features['data_points'],
                                features['peaks']['count'],
                                features['valleys']['count']
                            ])
                
                if feature_summary:
                    feature_df = pd.DataFrame(feature_summary, columns=['Light Source', 'Data Points', 'Peaks', 'Valleys'])
                    feature_df.set_index('Light Source').plot(kind='bar', ax=ax4, rot=45)
                    ax4.set_title('Feature Count Summary')
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save the plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_file = self.graphs_dir / f"structural_analysis_{gem_id}_{self.mode}_{timestamp}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Plot saved: {plot_file.name}")
                return plot_file
                
            except Exception as e:
                print(f"   ❌ Plot generation error: {e}")
                traceback.print_exc()
                return None
        
        def generate_comprehensive_report(self, gem_id, stone_data, plot_file):
            """Generate comprehensive analysis report"""
            print(f"\n📄 Generating comprehensive report...")
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = self.reports_dir / f"structural_analysis_{gem_id}_{self.mode}_{timestamp}.txt"
                
                with open(report_file, 'w') as f:
                    # Header
                    f.write(f"STRUCTURAL ANALYSIS REPORT\n")
                    f.write(f"=" * 60 + "\n")
                    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Stone of Interest: Gem {gem_id}\n")
                    f.write(f"Analysis Type: {self.analysis_name}\n")
                    f.write(f"Source Directory: {self.source_dir}\n")
                    f.write(f"Mode: {self.mode.upper()}\n")
                    f.write(f"\n")
                    
                    # Spectral Data Summary
                    f.write(f"SPECTRAL DATA SUMMARY\n")
                    f.write(f"-" * 40 + "\n")
                    light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                    
                    for light_code, (wavelength, intensity) in stone_data.items():
                        if wavelength is not None and intensity is not None:
                            f.write(f"\n{light_names[light_code]} ({light_code}) Light Source:\n")
                            f.write(f"   Data Points: {len(wavelength)}\n")
                            f.write(f"   Wavelength Range: {wavelength.min():.1f} - {wavelength.max():.1f} nm\n")
                            f.write(f"   Intensity Range: {intensity.min():.1f} - {intensity.max():.1f}\n")
                            
                            if light_code in self.spectral_features:
                                features = self.spectral_features[light_code]
                                if features:
                                    f.write(f"   Identified Peaks: {features['peaks']['count']}\n")
                                    f.write(f"   Identified Valleys: {features['valleys']['count']}\n")
                                    f.write(f"   Mean Intensity: {features['intensity_stats']['mean']:.2f}\n")
                                    f.write(f"   Std Deviation: {features['intensity_stats']['std']:.2f}\n")
                    
                    # Feature Analysis
                    f.write(f"\n\nFEATURE ANALYSIS\n")
                    f.write(f"-" * 40 + "\n")
                    
                    total_peaks = sum(self.spectral_features[lc]['peaks']['count'] 
                                    for lc in self.spectral_features if self.spectral_features[lc])
                    total_valleys = sum(self.spectral_features[lc]['valleys']['count'] 
                                      for lc in self.spectral_features if self.spectral_features[lc])
                    
                    f.write(f"Total Peaks Identified: {total_peaks}\n")
                    f.write(f"Total Valleys Identified: {total_valleys}\n")
                    f.write(f"Light Sources Analyzed: {len([lc for lc in stone_data if stone_data[lc][0] is not None])}\n")
                    
                    # Peak Details
                    for light_code in stone_data:
                        if light_code in self.spectral_features and self.spectral_features[light_code]:
                            features = self.spectral_features[light_code]
                            if features['peaks']['count'] > 0:
                                f.write(f"\n{light_names[light_code]} Peak Details:\n")
                                for i, (pos, intensity) in enumerate(zip(features['peaks']['positions'], 
                                                                        features['peaks']['intensities'])):
                                    f.write(f"   Peak {i+1}: {pos:.1f} nm (intensity: {intensity:.1f})\n")
                    
                    # Database Matches
                    f.write(f"\n\nDATABASE MATCHES\n")
                    f.write(f"-" * 40 + "\n")
                    if self.database_matches:
                        f.write(f"Found {len(self.database_matches)} similar gems in database:\n")
                        for i, match in enumerate(self.database_matches[:5], 1):
                            f.write(f"   {i}. {match}\n")
                    else:
                        f.write("No matches found in database\n")
                    
                    # Analysis Summary
                    f.write(f"\n\nANALYSIS SUMMARY\n")
                    f.write(f"-" * 40 + "\n")
                    f.write(f"✅ Spectral data successfully loaded for {len(stone_data)} light sources\n")
                    f.write(f"✅ Feature extraction completed\n")
                    f.write(f"✅ Database query completed\n")
                    if plot_file:
                        f.write(f"✅ Visualization plots generated: {plot_file.name}\n")
                    
                    # Recommendations
                    f.write(f"\n\nRECOMMENDATIONS\n")
                    f.write(f"-" * 40 + "\n")
                    if len(stone_data) == 3:
                        f.write(f"• Complete analysis performed with all three light sources\n")
                    else:
                        missing = set(['B', 'L', 'U']) - set(stone_data.keys())
                        f.write(f"• Consider acquiring data for missing light sources: {', '.join(missing)}\n")
                    
                    if total_peaks < 5:
                        f.write(f"• Low peak count - consider different analysis parameters\n")
                    
                    if not self.database_matches:
                        f.write(f"• No database matches found - this may be a unique specimen\n")
                    
                    f.write(f"\nAnalysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                print(f"   ✅ Report saved: {report_file.name}")
                return report_file
                
            except Exception as e:
                print(f"   ❌ Report generation error: {e}")
                traceback.print_exc()
                return None
        
        def run_full_analysis(self):
            """Run complete structural analysis workflow"""
            print(f"\n🚀 RUNNING COMPLETE STRUCTURAL ANALYSIS")
            print(f"Analysis Type: {self.analysis_name}")
            print(f"Source: {self.source_dir}")
            
            # Step 1: Check source directory
            if not self.check_source_directory():
                print("❌ Source directory check failed")
                return False
            
            # Step 2: Get and group CSV files
            csv_files = list(self.source_dir.glob("*.csv"))
            gem_groups = self.group_files_by_gem(csv_files)
            
            if not gem_groups:
                print("❌ No gem groups found after parsing")
                return False
            
            # Step 3: Interactive gem selection
            selection = self.interactive_gem_selection(gem_groups)
            if not selection:
                print("❌ No gem selected for analysis")
                return False
            
            stone_gem_id, stone_files = selection
            print(f"\n🔬 Analyzing Gem {stone_gem_id}...")
            
            # Step 4: Load spectral data and extract features
            stone_data = {}
            light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
            
            for light_code, file_path in stone_files.items():
                print(f"\n📊 Processing {light_names[light_code]} data...")
                wavelength, intensity = self.load_spectral_data(file_path)
                stone_data[light_code] = (wavelength, intensity)
                
                if wavelength is not None and intensity is not None:
                    features = self.extract_spectral_features(wavelength, intensity, light_names[light_code])
                    self.spectral_features[light_code] = features
            
            # Step 5: Query database for matches
            self.database_matches = self.query_database_matches(stone_gem_id)
            
            # Step 6: Generate visualizations
            plot_file = self.generate_comparison_plots(stone_data, stone_gem_id)
            
            # Step 7: Generate comprehensive report
            report_file = self.generate_comprehensive_report(stone_gem_id, stone_data, plot_file)
            
            # Step 8: Summary
            print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"   Gem ID: {stone_gem_id}")
            print(f"   Light sources analyzed: {len([lc for lc in stone_data if stone_data[lc][0] is not None])}")
            print(f"   Total peaks found: {sum(self.spectral_features[lc]['peaks']['count'] for lc in self.spectral_features if self.spectral_features[lc])}")
            print(f"   Database matches: {len(self.database_matches)}")
            
            if report_file:
                print(f"   📄 Report: {report_file.name}")
            if plot_file:
                print(f"   📈 Plots: {plot_file.name}")
            
            return True
        
        def run_diagnostic_only(self):
            """Run diagnostic tests only"""
            print(f"\n🔧 RUNNING DIAGNOSTIC TESTS ONLY")
            
            # Test 1: Directory structure
            print(f"\n1️⃣ Testing directory structure...")
            structure_ok = self.check_source_directory()
            
            # Test 2: File parsing
            print(f"\n2️⃣ Testing file parsing...")
            csv_files = list(self.source_dir.glob("*.csv"))
            gem_groups = self.group_files_by_gem(csv_files)
            parsing_ok = len(gem_groups) > 0
            
            # Test 3: Data loading
            print(f"\n3️⃣ Testing data loading...")
            loading_ok = False
            if csv_files:
                test_file = csv_files[0]
                wavelength, intensity = self.load_spectral_data(test_file)
                loading_ok = wavelength is not None and intensity is not None
            
            # Test 4: Feature extraction
            print(f"\n4️⃣ Testing feature extraction...")
            features_ok = False
            if loading_ok:
                features = self.extract_spectral_features(wavelength, intensity, "Test")
                features_ok = features is not None
            
            # Test 5: Visualization import
            print(f"\n5️⃣ Testing visualization capabilities...")
            viz_ok = HAS_MATPLOTLIB
            if HAS_MATPLOTLIB:
                print(f"   ✅ Matplotlib available")
            else:
                print(f"   ⚠️ Matplotlib not available - plots will be skipped")
                print(f"   💡 Install with: pip install matplotlib")
                
            if HAS_SEABORN:
                print(f"   ✅ Seaborn available (enhanced styling)")
            else:
                print(f"   ⚠️ Seaborn not available - using basic styling")
                print(f"   💡 Install with: pip install seaborn")
                
            if HAS_SCIPY:
                print(f"   ✅ SciPy available (advanced peak detection)")
            else:
                print(f"   ⚠️ SciPy not available - using basic peak detection")
                print(f"   💡 Install with: pip install scipy")
            
            # Test 6: Database connectivity
            print(f"\n6️⃣ Testing database connectivity...")
            db_ok = self.db_path.exists()
            if db_ok:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                        print(f"   ✅ Database accessible with {len(tables)} tables")
                except Exception as e:
                    print(f"   ⚠️ Database connection issue: {e}")
                    db_ok = False
            else:
                print(f"   ⚠️ Database file not found: {self.db_path}")
            
            # Summary
            print(f"\n📋 DIAGNOSTIC SUMMARY")
            print(f"=" * 40)
            print(f"{'✅' if structure_ok else '❌'} Directory structure")
            print(f"{'✅' if parsing_ok else '❌'} File parsing")
            print(f"{'✅' if loading_ok else '❌'} Data loading")
            print(f"{'✅' if features_ok else '❌'} Feature extraction")
            print(f"{'✅' if HAS_MATPLOTLIB else '⚠️'} Matplotlib (optional)")
            print(f"{'✅' if HAS_SEABORN else '⚠️'} Seaborn (optional)")
            print(f"{'✅' if HAS_SCIPY else '⚠️'} SciPy (optional)")
            print(f"{'✅' if db_ok else '⚠️'} Database connectivity")
            
            # Core functionality only requires basic components
            core_ok = all([structure_ok, parsing_ok, loading_ok, features_ok])
            
            if core_ok:
                print(f"\n🎉 CORE FUNCTIONALITY READY!")
                print(f"   System can perform basic structural analysis")
                if not HAS_MATPLOTLIB:
                    print(f"   📊 Note: Install matplotlib for visualization plots")
                if not HAS_SCIPY:
                    print(f"   🔍 Note: Install scipy for advanced peak detection")
            else:
                print(f"\n⚠️ CORE FUNCTIONALITY ISSUES DETECTED")
                print(f"   Address the ❌ issues above before running analysis")
            
            return core_ok

    # Main execution
    print(f"\n🚀 Creating analyzer instance...")
    analyzer = CompleteUnifiedStructuralAnalyzer(mode)
    
    if mode == "diagnostic":
        print(f"\n🔧 Running diagnostic mode...")
        success = analyzer.run_diagnostic_only()
    else:
        print(f"\n🔬 Running full analysis mode...")
        success = analyzer.run_full_analysis()
    
    if success:
        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Check outputs/structural_results/ for results")
    else:
        print(f"\n❌ ANALYSIS FAILED")
        print(f"📋 Check the error messages above for details")

except Exception as main_error:
    print(f"\n💥 MAIN ERROR OCCURRED:")
    print(f"Error: {main_error}")
    print(f"\nFull traceback:")
    traceback.print_exc()
    
    print(f"\n🔍 DEBUGGING SUGGESTIONS:")
    print(f"1. Check that you're running from the project root directory")
    print(f"2. Verify that src/structural_analysis/unified_structural_analyzer.py exists")
    print(f"3. Check that data/structural_data/ or data/structural(archive)/ has CSV files")
    print(f"4. Ensure Python dependencies: pandas, numpy, matplotlib, seaborn, scipy")
    print(f"5. Install missing packages: pip install pandas numpy matplotlib seaborn scipy")

print(f"\n🔬 COMPLETE STRUCTURAL ANALYZER ENDING")
print("=" * 60), base_part)
                    if gem_match:
                        gem_id = gem_match.group(1)
                    else:
                        gem_id = base_part
                    
                    light_source = light_mapping.get(light.upper())
                    return gem_id, light_source
                else:
                    return None, None
        
        def group_files_by_gem(self, csv_files):
            """Group CSV files by gem ID and light source"""
            print(f"\n🧪 Grouping files by gem ID and light source...")
            print(f"🔍 Debugging file parsing:")
            
            gem_groups = defaultdict(lambda: {'B': [], 'L': [], 'U': []})
            
            for file_path in csv_files[:10]:  # Debug first 10 files
                filename = file_path.name
                gem_id, light_source = self.parse_gem_info(filename)
                
                print(f"   📄 {filename}")
                print(f"      → Gem ID: '{gem_id}', Light: '{light_source}'")
                
                if gem_id and light_source:
                    light_code = {'Halogen': 'B', 'Laser': 'L', 'UV': 'U'}.get(light_source)
                    if light_code:
                        gem_groups[gem_id][light_code].append(file_path)
                        print(f"      → Added to group: {gem_id}[{light_code}]")
            
            # Process remaining files without debug output
            for file_path in csv_files[10:]:
                gem_id, light_source = self.parse_gem_info(file_path.name)
                if gem_id and light_source:
                    light_code = {'Halogen': 'B', 'Laser': 'L', 'UV': 'U'}.get(light_source)
                    if light_code:
                        gem_groups[gem_id][light_code].append(file_path)
            
            print(f"\n📊 Found {len(gem_groups)} unique gems")
            
            # Show complete vs partial gems
            complete_gems = []
            partial_gems = []
            
            for gem_id, light_files in gem_groups.items():
                b_count = len(light_files['B'])
                l_count = len(light_files['L'])
                u_count = len(light_files['U'])
                
                if b_count > 0 and l_count > 0 and u_count > 0:
                    complete_gems.append((gem_id, b_count, l_count, u_count))
                else:
                    partial_gems.append((gem_id, b_count, l_count, u_count))
            
            print(f"🟢 Complete gems (B+L+U): {len(complete_gems)}")
            for gem_id, b, l, u in complete_gems[:5]:
                print(f"   {gem_id}: B({b}), L({l}), U({u})")
            
            print(f"🟡 Partial gems: {len(partial_gems)}")
            for gem_id, b, l, u in partial_gems[:3]:
                coverage = []
                if b > 0: coverage.append(f"B({b})")
                if l > 0: coverage.append(f"L({l})")
                if u > 0: coverage.append(f"U({u})")
                print(f"   {gem_id}: {', '.join(coverage)}")
            
            return gem_groups
        
        def interactive_gem_selection(self, gem_groups):
            """Interactive gem selection for analysis"""
            print(f"\n🎯 Available gems for analysis:")
            print("=" * 50)
            
            # Separate complete and partial gems
            complete_gems = []
            partial_gems = []
            
            for gem_id, light_files in sorted(gem_groups.items()):
                b_count = len(light_files['B'])
                l_count = len(light_files['L'])
                u_count = len(light_files['U'])
                total = b_count + l_count + u_count
                
                if b_count > 0 and l_count > 0 and u_count > 0:
                    complete_gems.append((gem_id, light_files, b_count, l_count, u_count, total))
                else:
                    partial_gems.append((gem_id, light_files, b_count, l_count, u_count, total))
            
            gem_options = []
            option_num = 1
            
            # Show complete gems first (recommended)
            if complete_gems:
                print(f"🟢 COMPLETE GEMS (Recommended - B+L+U coverage):")
                for gem_id, light_files, b_count, l_count, u_count, total in complete_gems:
                    print(f"{option_num:2d}. 🟢 Gem {gem_id}: B({b_count})+L({l_count})+U({u_count}) = {total} files [COMPLETE]")
                    gem_options.append((gem_id, light_files))
                    option_num += 1
                print()
            
            # Show partial gems  
            if partial_gems:
                print(f"🟡 PARTIAL GEMS (Limited analysis possible):")
                for gem_id, light_files, b_count, l_count, u_count, total in partial_gems[:20]:  # Limit to first 20
                    coverage = []
                    if b_count > 0: coverage.append(f"B({b_count})")
                    if l_count > 0: coverage.append(f"L({l_count})")
                    if u_count > 0: coverage.append(f"U({u_count})")
                    
                    print(f"{option_num:2d}. 🟡 Gem {gem_id}: {'+'.join(coverage)} = {total} files [Partial]")
                    gem_options.append((gem_id, light_files))
                    option_num += 1
                
                if len(partial_gems) > 20:
                    print(f"    ... and {len(partial_gems) - 20} more partial gems")
            
            if not gem_options:
                print("❌ No valid gem options found")
                return None
            
            # Recommend complete gems
            if complete_gems:
                print(f"\n💡 RECOMMENDATION: Choose a complete gem (1-{len(complete_gems)}) for best analysis")
                print(f"   Complete gems have all three light sources (B+L+U) like your example:")
                print(f"   Gem 58: 58BC1 + 58LC1 + 58UC1")
            
            # For diagnostic mode, auto-select first complete gem if available
            if self.mode == "diagnostic":
                if complete_gems:
                    selected_gem_id, selected_files = gem_options[0]
                    print(f"\n🤖 DIAGNOSTIC MODE: Auto-selecting first complete gem...")
                    print(f"   Selected: Gem {selected_gem_id} (Complete)")
                else:
                    selected_gem_id, selected_files = gem_options[0]
                    print(f"\n🤖 DIAGNOSTIC MODE: Auto-selecting first available gem...")
                    print(f"   Selected: Gem {selected_gem_id} (Partial)")
            else:
                # Interactive selection
                print(f"\n🎯 Interactive Gem Selection")
                print(f"   Please select a gem number from the list above")
                
                # Try multiple input attempts
                max_attempts = 3
                selected_gem_id, selected_files = None, None
                
                for attempt in range(max_attempts):
                    try:
                        raw_input = input(f"\nSelect gem number for analysis (1-{len(gem_options)}, or 'q' to quit): ")
                        
                        # Handle InputSubmission wrapper (same as main.py safe_input)
                        choice = str(raw_input).strip()
                        if "InputSubmission(data='" in choice and choice.endswith("')"):
                            choice = choice[22:-2]
                        choice = choice.replace('\\n', '').replace('\n', '').strip("'\"")
                        
                        print(f"   User input received: '{choice}'")  # Debug output
                        
                        if choice.lower() == 'q':
                            print("   Analysis cancelled by user")
                            return None
                        
                        gem_num = int(choice)
                        if 1 <= gem_num <= len(gem_options):
                            selected_gem_id, selected_files = gem_options[gem_num - 1]
                            print(f"✅ Selected: Gem {selected_gem_id}")
                            break
                        else:
                            print(f"❌ Invalid selection. Choose 1-{len(gem_options)}")
                            
                    except ValueError:
                        print("❌ Please enter a number or 'q' to quit")
                    except EOFError:
                        print("⚠️ Input stream ended - using auto-selection")
                        selected_gem_id, selected_files = gem_options[0]
                        print(f"✅ Auto-selected: Gem {selected_gem_id}")
                        break
                    except Exception as e:
                        print(f"⚠️ Input error: {e}")
                        
                    if attempt == max_attempts - 1:
                        print(f"⚠️ Max input attempts reached - auto-selecting first gem")
                        selected_gem_id, selected_files = gem_options[0]
                        print(f"✅ Auto-selected: Gem {selected_gem_id}")
                
                if not selected_gem_id:
                    print("❌ No gem selected")
                    return None
            
            # Select files for each light source
            stone_of_interest = {}
            light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
            
            for light_code in ['B', 'L', 'U']:
                files = selected_files[light_code]
                if files:
                    # Take the first file for each light source
                    stone_of_interest[light_code] = files[0]
                    print(f"   ✅ {light_names[light_code]} ({light_code}): {files[0].name}")
                else:
                    print(f"   ❌ {light_names[light_code]} ({light_code}): No files available")
            
            if not stone_of_interest:
                print("❌ No files selected for analysis")
                return None
            
            return selected_gem_id, stone_of_interest
        
        def load_spectral_data(self, file_path):
            """Load and validate spectral data from CSV file"""
            try:
                print(f"   📄 Loading: {file_path.name}")
                df = pd.read_csv(file_path)
                
                # Try to identify wavelength and intensity columns
                if 'Wavelength' in df.columns and 'Intensity' in df.columns:
                    wavelength = df['Wavelength'].values
                    intensity = df['Intensity'].values
                elif len(df.columns) >= 2:
                    # Assume first two columns are wavelength and intensity
                    wavelength = df.iloc[:, 0].values
                    intensity = df.iloc[:, 1].values
                else:
                    print(f"      ❌ Cannot identify wavelength/intensity columns")
                    return None, None
                
                # Validate data
                if len(wavelength) < 10:
                    print(f"      ⚠️ Warning: Only {len(wavelength)} data points")
                
                print(f"      ✅ Loaded {len(wavelength)} data points")
                print(f"      📊 Range: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
                
                return wavelength, intensity
                
            except Exception as e:
                print(f"      ❌ Error loading {file_path.name}: {e}")
                return None, None
        
        def extract_spectral_features(self, wavelength, intensity, light_source):
            """Extract key spectral features from the data"""
            try:
                # Find peaks using scipy if available, otherwise use simple method
                if HAS_SCIPY:
                    peaks, peak_properties = find_peaks(intensity, height=np.percentile(intensity, 75), distance=5)
                    valleys, valley_properties = find_peaks(-intensity, height=-np.percentile(intensity, 25), distance=5)
                else:
                    # Simple peak detection without scipy
                    threshold = np.percentile(intensity, 75)
                    peaks = []
                    for i in range(1, len(intensity) - 1):
                        if intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1] and intensity[i] > threshold:
                            peaks.append(i)
                    
                    # Simple valley detection
                    valley_threshold = np.percentile(intensity, 25)
                    valleys = []
                    for i in range(1, len(intensity) - 1):
                        if intensity[i] < intensity[i-1] and intensity[i] < intensity[i+1] and intensity[i] < valley_threshold:
                            valleys.append(i)
                
                # Calculate statistical features
                features = {
                    'light_source': light_source,
                    'data_points': len(wavelength),
                    'wavelength_range': [float(wavelength.min()), float(wavelength.max())],
                    'intensity_stats': {
                        'mean': float(np.mean(intensity)),
                        'std': float(np.std(intensity)),
                        'max': float(np.max(intensity)),
                        'min': float(np.min(intensity))
                    },
                    'peaks': {
                        'count': len(peaks),
                        'positions': [float(wavelength[i]) for i in peaks] if len(peaks) > 0 else [],
                        'intensities': [float(intensity[i]) for i in peaks] if len(peaks) > 0 else []
                    },
                    'valleys': {
                        'count': len(valleys),
                        'positions': [float(wavelength[i]) for i in valleys] if len(valleys) > 0 else [],
                        'intensities': [float(intensity[i]) for i in valleys] if len(valleys) > 0 else []
                    }
                }
                
                print(f"      🔍 Features: {len(peaks)} peaks, {len(valleys)} valleys")
                
                return features
                
            except Exception as e:
                print(f"      ❌ Feature extraction error: {e}")
                return None
        
        def query_database_matches(self, gem_id):
            """Query database for similar gems"""
            print(f"\n🗄️ Querying database for matches...")
            
            matches = []
            
            try:
                if self.db_path.exists():
                    with sqlite3.connect(self.db_path) as conn:
                        # Try different table structures
                        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                        print(f"   📋 Database tables: {', '.join(tables['name'].tolist())}")
                        
                        # Query for similar gems
                        for table_name in tables['name']:
                            try:
                                query = f"SELECT * FROM {table_name} WHERE gem_id LIKE '%{gem_id}%' OR gem_id LIKE '%{gem_id[:3]}%' LIMIT 10"
                                results = pd.read_sql_query(query, conn)
                                
                                if not results.empty:
                                    print(f"   ✅ Found {len(results)} matches in {table_name}")
                                    matches.extend(results.to_dict('records'))
                                    
                            except Exception as e:
                                print(f"   ⚠️ Query error for {table_name}: {e}")
                                
                else:
                    print(f"   ⚠️ Database not found: {self.db_path}")
                    
            except Exception as e:
                print(f"   ❌ Database query error: {e}")
            
            print(f"   📊 Total matches found: {len(matches)}")
            return matches
        
        def generate_comparison_plots(self, stone_data, gem_id):
            """Generate comparison plots for the spectral data"""
            print(f"\n📈 Generating comparison plots...")
            
            if not HAS_MATPLOTLIB:
                print(f"   ⚠️ Matplotlib not available - skipping plot generation")
                return None
            
            try:
                # Set up the plot style
                plt.style.use('default')
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'Structural Analysis: Gem {gem_id} ({self.analysis_name})', fontsize=16, fontweight='bold')
                
                # Color mapping for light sources
                colors = {'B': 'blue', 'L': 'red', 'U': 'purple'}
                light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                
                # Plot 1: Overlaid spectra
                ax1 = axes[0, 0]
                for light_code, (wavelength, intensity) in stone_data.items():
                    if wavelength is not None and intensity is not None:
                        ax1.plot(wavelength, intensity, color=colors[light_code], 
                               label=f'{light_names[light_code]} ({light_code})', linewidth=2)
                
                ax1.set_xlabel('Wavelength (nm)')
                ax1.set_ylabel('Intensity')
                ax1.set_title('Multi-Source Spectral Overlay')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Peak analysis
                ax2 = axes[0, 1]
                peak_data = []
                for light_code in stone_data:
                    if light_code in self.spectral_features:
                        features = self.spectral_features[light_code]
                        if features and 'peaks' in features:
                            peak_positions = features['peaks']['positions']
                            peak_intensities = features['peaks']['intensities']
                            if peak_positions and peak_intensities:
                                ax2.scatter(peak_positions, peak_intensities, 
                                          color=colors[light_code], label=f'{light_names[light_code]} Peaks',
                                          s=50, alpha=0.7)
                                peak_data.extend(peak_positions)
                
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Peak Intensity')
                ax2.set_title('Identified Peaks Comparison')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Statistical comparison
                ax3 = axes[1, 0]
                stats_data = {'Light Source': [], 'Mean Intensity': [], 'Std Dev': [], 'Peak Count': []}
                for light_code in stone_data:
                    if light_code in self.spectral_features:
                        features = self.spectral_features[light_code]
                        if features:
                            stats_data['Light Source'].append(light_names[light_code])
                            stats_data['Mean Intensity'].append(features['intensity_stats']['mean'])
                            stats_data['Std Dev'].append(features['intensity_stats']['std'])
                            stats_data['Peak Count'].append(features['peaks']['count'])
                
                if stats_data['Light Source']:
                    x_pos = np.arange(len(stats_data['Light Source']))
                    ax3.bar(x_pos, stats_data['Mean Intensity'], alpha=0.7, 
                           color=[colors[k] for k in stone_data.keys() if k in colors])
                    ax3.set_xticks(x_pos)
                    ax3.set_xticklabels(stats_data['Light Source'])
                    ax3.set_ylabel('Mean Intensity')
                    ax3.set_title('Statistical Comparison')
                    ax3.grid(True, alpha=0.3)
                
                # Plot 4: Feature summary
                ax4 = axes[1, 1]
                feature_summary = []
                for light_code in stone_data:
                    if light_code in self.spectral_features:
                        features = self.spectral_features[light_code]
                        if features:
                            feature_summary.append([
                                light_names[light_code],
                                features['data_points'],
                                features['peaks']['count'],
                                features['valleys']['count']
                            ])
                
                if feature_summary:
                    feature_df = pd.DataFrame(feature_summary, columns=['Light Source', 'Data Points', 'Peaks', 'Valleys'])
                    feature_df.set_index('Light Source').plot(kind='bar', ax=ax4, rot=45)
                    ax4.set_title('Feature Count Summary')
                    ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save the plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_file = self.graphs_dir / f"structural_analysis_{gem_id}_{self.mode}_{timestamp}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Plot saved: {plot_file.name}")
                return plot_file
                
            except Exception as e:
                print(f"   ❌ Plot generation error: {e}")
                traceback.print_exc()
                return None
        
        def generate_comprehensive_report(self, gem_id, stone_data, plot_file):
            """Generate comprehensive analysis report"""
            print(f"\n📄 Generating comprehensive report...")
            
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = self.reports_dir / f"structural_analysis_{gem_id}_{self.mode}_{timestamp}.txt"
                
                with open(report_file, 'w') as f:
                    # Header
                    f.write(f"STRUCTURAL ANALYSIS REPORT\n")
                    f.write(f"=" * 60 + "\n")
                    f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Stone of Interest: Gem {gem_id}\n")
                    f.write(f"Analysis Type: {self.analysis_name}\n")
                    f.write(f"Source Directory: {self.source_dir}\n")
                    f.write(f"Mode: {self.mode.upper()}\n")
                    f.write(f"\n")
                    
                    # Spectral Data Summary
                    f.write(f"SPECTRAL DATA SUMMARY\n")
                    f.write(f"-" * 40 + "\n")
                    light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
                    
                    for light_code, (wavelength, intensity) in stone_data.items():
                        if wavelength is not None and intensity is not None:
                            f.write(f"\n{light_names[light_code]} ({light_code}) Light Source:\n")
                            f.write(f"   Data Points: {len(wavelength)}\n")
                            f.write(f"   Wavelength Range: {wavelength.min():.1f} - {wavelength.max():.1f} nm\n")
                            f.write(f"   Intensity Range: {intensity.min():.1f} - {intensity.max():.1f}\n")
                            
                            if light_code in self.spectral_features:
                                features = self.spectral_features[light_code]
                                if features:
                                    f.write(f"   Identified Peaks: {features['peaks']['count']}\n")
                                    f.write(f"   Identified Valleys: {features['valleys']['count']}\n")
                                    f.write(f"   Mean Intensity: {features['intensity_stats']['mean']:.2f}\n")
                                    f.write(f"   Std Deviation: {features['intensity_stats']['std']:.2f}\n")
                    
                    # Feature Analysis
                    f.write(f"\n\nFEATURE ANALYSIS\n")
                    f.write(f"-" * 40 + "\n")
                    
                    total_peaks = sum(self.spectral_features[lc]['peaks']['count'] 
                                    for lc in self.spectral_features if self.spectral_features[lc])
                    total_valleys = sum(self.spectral_features[lc]['valleys']['count'] 
                                      for lc in self.spectral_features if self.spectral_features[lc])
                    
                    f.write(f"Total Peaks Identified: {total_peaks}\n")
                    f.write(f"Total Valleys Identified: {total_valleys}\n")
                    f.write(f"Light Sources Analyzed: {len([lc for lc in stone_data if stone_data[lc][0] is not None])}\n")
                    
                    # Peak Details
                    for light_code in stone_data:
                        if light_code in self.spectral_features and self.spectral_features[light_code]:
                            features = self.spectral_features[light_code]
                            if features['peaks']['count'] > 0:
                                f.write(f"\n{light_names[light_code]} Peak Details:\n")
                                for i, (pos, intensity) in enumerate(zip(features['peaks']['positions'], 
                                                                        features['peaks']['intensities'])):
                                    f.write(f"   Peak {i+1}: {pos:.1f} nm (intensity: {intensity:.1f})\n")
                    
                    # Database Matches
                    f.write(f"\n\nDATABASE MATCHES\n")
                    f.write(f"-" * 40 + "\n")
                    if self.database_matches:
                        f.write(f"Found {len(self.database_matches)} similar gems in database:\n")
                        for i, match in enumerate(self.database_matches[:5], 1):
                            f.write(f"   {i}. {match}\n")
                    else:
                        f.write("No matches found in database\n")
                    
                    # Analysis Summary
                    f.write(f"\n\nANALYSIS SUMMARY\n")
                    f.write(f"-" * 40 + "\n")
                    f.write(f"✅ Spectral data successfully loaded for {len(stone_data)} light sources\n")
                    f.write(f"✅ Feature extraction completed\n")
                    f.write(f"✅ Database query completed\n")
                    if plot_file:
                        f.write(f"✅ Visualization plots generated: {plot_file.name}\n")
                    
                    # Recommendations
                    f.write(f"\n\nRECOMMENDATIONS\n")
                    f.write(f"-" * 40 + "\n")
                    if len(stone_data) == 3:
                        f.write(f"• Complete analysis performed with all three light sources\n")
                    else:
                        missing = set(['B', 'L', 'U']) - set(stone_data.keys())
                        f.write(f"• Consider acquiring data for missing light sources: {', '.join(missing)}\n")
                    
                    if total_peaks < 5:
                        f.write(f"• Low peak count - consider different analysis parameters\n")
                    
                    if not self.database_matches:
                        f.write(f"• No database matches found - this may be a unique specimen\n")
                    
                    f.write(f"\nAnalysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                print(f"   ✅ Report saved: {report_file.name}")
                return report_file
                
            except Exception as e:
                print(f"   ❌ Report generation error: {e}")
                traceback.print_exc()
                return None
        
        def run_full_analysis(self):
            """Run complete structural analysis workflow"""
            print(f"\n🚀 RUNNING COMPLETE STRUCTURAL ANALYSIS")
            print(f"Analysis Type: {self.analysis_name}")
            print(f"Source: {self.source_dir}")
            
            # Step 1: Check source directory
            if not self.check_source_directory():
                print("❌ Source directory check failed")
                return False
            
            # Step 2: Get and group CSV files
            csv_files = list(self.source_dir.glob("*.csv"))
            gem_groups = self.group_files_by_gem(csv_files)
            
            if not gem_groups:
                print("❌ No gem groups found after parsing")
                return False
            
            # Step 3: Interactive gem selection
            selection = self.interactive_gem_selection(gem_groups)
            if not selection:
                print("❌ No gem selected for analysis")
                return False
            
            stone_gem_id, stone_files = selection
            print(f"\n🔬 Analyzing Gem {stone_gem_id}...")
            
            # Step 4: Load spectral data and extract features
            stone_data = {}
            light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
            
            for light_code, file_path in stone_files.items():
                print(f"\n📊 Processing {light_names[light_code]} data...")
                wavelength, intensity = self.load_spectral_data(file_path)
                stone_data[light_code] = (wavelength, intensity)
                
                if wavelength is not None and intensity is not None:
                    features = self.extract_spectral_features(wavelength, intensity, light_names[light_code])
                    self.spectral_features[light_code] = features
            
            # Step 5: Query database for matches
            self.database_matches = self.query_database_matches(stone_gem_id)
            
            # Step 6: Generate visualizations
            plot_file = self.generate_comparison_plots(stone_data, stone_gem_id)
            
            # Step 7: Generate comprehensive report
            report_file = self.generate_comprehensive_report(stone_gem_id, stone_data, plot_file)
            
            # Step 8: Summary
            print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"   Gem ID: {stone_gem_id}")
            print(f"   Light sources analyzed: {len([lc for lc in stone_data if stone_data[lc][0] is not None])}")
            print(f"   Total peaks found: {sum(self.spectral_features[lc]['peaks']['count'] for lc in self.spectral_features if self.spectral_features[lc])}")
            print(f"   Database matches: {len(self.database_matches)}")
            
            if report_file:
                print(f"   📄 Report: {report_file.name}")
            if plot_file:
                print(f"   📈 Plots: {plot_file.name}")
            
            return True
        
        def run_diagnostic_only(self):
            """Run diagnostic tests only"""
            print(f"\n🔧 RUNNING DIAGNOSTIC TESTS ONLY")
            
            # Test 1: Directory structure
            print(f"\n1️⃣ Testing directory structure...")
            structure_ok = self.check_source_directory()
            
            # Test 2: File parsing
            print(f"\n2️⃣ Testing file parsing...")
            csv_files = list(self.source_dir.glob("*.csv"))
            gem_groups = self.group_files_by_gem(csv_files)
            parsing_ok = len(gem_groups) > 0
            
            # Test 3: Data loading
            print(f"\n3️⃣ Testing data loading...")
            loading_ok = False
            if csv_files:
                test_file = csv_files[0]
                wavelength, intensity = self.load_spectral_data(test_file)
                loading_ok = wavelength is not None and intensity is not None
            
            # Test 4: Feature extraction
            print(f"\n4️⃣ Testing feature extraction...")
            features_ok = False
            if loading_ok:
                features = self.extract_spectral_features(wavelength, intensity, "Test")
                features_ok = features is not None
            
            # Test 5: Visualization import
            print(f"\n5️⃣ Testing visualization capabilities...")
            viz_ok = HAS_MATPLOTLIB
            if HAS_MATPLOTLIB:
                print(f"   ✅ Matplotlib available")
            else:
                print(f"   ⚠️ Matplotlib not available - plots will be skipped")
                print(f"   💡 Install with: pip install matplotlib")
                
            if HAS_SEABORN:
                print(f"   ✅ Seaborn available (enhanced styling)")
            else:
                print(f"   ⚠️ Seaborn not available - using basic styling")
                print(f"   💡 Install with: pip install seaborn")
                
            if HAS_SCIPY:
                print(f"   ✅ SciPy available (advanced peak detection)")
            else:
                print(f"   ⚠️ SciPy not available - using basic peak detection")
                print(f"   💡 Install with: pip install scipy")
            
            # Test 6: Database connectivity
            print(f"\n6️⃣ Testing database connectivity...")
            db_ok = self.db_path.exists()
            if db_ok:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                        print(f"   ✅ Database accessible with {len(tables)} tables")
                except Exception as e:
                    print(f"   ⚠️ Database connection issue: {e}")
                    db_ok = False
            else:
                print(f"   ⚠️ Database file not found: {self.db_path}")
            
            # Summary
            print(f"\n📋 DIAGNOSTIC SUMMARY")
            print(f"=" * 40)
            print(f"{'✅' if structure_ok else '❌'} Directory structure")
            print(f"{'✅' if parsing_ok else '❌'} File parsing")
            print(f"{'✅' if loading_ok else '❌'} Data loading")
            print(f"{'✅' if features_ok else '❌'} Feature extraction")
            print(f"{'✅' if HAS_MATPLOTLIB else '⚠️'} Matplotlib (optional)")
            print(f"{'✅' if HAS_SEABORN else '⚠️'} Seaborn (optional)")
            print(f"{'✅' if HAS_SCIPY else '⚠️'} SciPy (optional)")
            print(f"{'✅' if db_ok else '⚠️'} Database connectivity")
            
            # Core functionality only requires basic components
            core_ok = all([structure_ok, parsing_ok, loading_ok, features_ok])
            
            if core_ok:
                print(f"\n🎉 CORE FUNCTIONALITY READY!")
                print(f"   System can perform basic structural analysis")
                if not HAS_MATPLOTLIB:
                    print(f"   📊 Note: Install matplotlib for visualization plots")
                if not HAS_SCIPY:
                    print(f"   🔍 Note: Install scipy for advanced peak detection")
            else:
                print(f"\n⚠️ CORE FUNCTIONALITY ISSUES DETECTED")
                print(f"   Address the ❌ issues above before running analysis")
            
            return core_ok

    # Main execution
    print(f"\n🚀 Creating analyzer instance...")
    analyzer = CompleteUnifiedStructuralAnalyzer(mode)
    
    if mode == "diagnostic":
        print(f"\n🔧 Running diagnostic mode...")
        success = analyzer.run_diagnostic_only()
    else:
        print(f"\n🔬 Running full analysis mode...")
        success = analyzer.run_full_analysis()
    
    if success:
        print(f"\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Check outputs/structural_results/ for results")
    else:
        print(f"\n❌ ANALYSIS FAILED")
        print(f"📋 Check the error messages above for details")

except Exception as main_error:
    print(f"\n💥 MAIN ERROR OCCURRED:")
    print(f"Error: {main_error}")
    print(f"\nFull traceback:")
    traceback.print_exc()
    
    print(f"\n🔍 DEBUGGING SUGGESTIONS:")
    print(f"1. Check that you're running from the project root directory")
    print(f"2. Verify that src/structural_analysis/unified_structural_analyzer.py exists")
    print(f"3. Check that data/structural_data/ or data/structural(archive)/ has CSV files")
    print(f"4. Ensure Python dependencies: pandas, numpy, matplotlib, seaborn, scipy")
    print(f"5. Install missing packages: pip install pandas numpy matplotlib seaborn scipy")

print(f"\n🔬 COMPLETE STRUCTURAL ANALYZER ENDING")
print("=" * 60)