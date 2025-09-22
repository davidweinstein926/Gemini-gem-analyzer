#!/usr/bin/env python3
"""
structural_visualizer.py - Structural Analysis Visualization System
Matches the style and workflow of gemini1.py and result_visualizer.py
Saves to root/outputs/structural_results/reports;graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import re
import json

class StructuralVisualizer:
    """Structural analysis visualizer matching gemini1.py style"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        
        # Match your exact directory structure
        self.reports_dir = self.project_root / 'outputs' / 'structural_results' / 'reports'
        self.graphs_dir = self.project_root / 'outputs' / 'structural_results' / 'graphs'
        self.raw_archive_dir = self.project_root / 'data' / 'raw (archive)'
        
        # Ensure output directories exist
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        
        # Use your exact color scheme from result_visualizer.py
        self.light_colors = {
            'B': '#FF6B35',  # Orange for Halogen/Broadband
            'L': '#004E98',  # Blue for Laser  
            'U': '#7209B7'   # Purple for UV
        }
        
        # Load gem library (same as your gemini1.py)
        self.gem_name_map = self.load_gem_library()
        
        # Configure matplotlib style (same as result_visualizer.py)
        self.setup_plot_style()
    
    def load_gem_library(self):
        """Load gemstone library - same logic as gemini1.py"""
        gem_name_map = {}
        gemlib_path = self.project_root / 'gemlib_structural_ready.csv'
        
        try:
            if gemlib_path.exists():
                gemlib = pd.read_csv(gemlib_path)
                gemlib.columns = gemlib.columns.str.strip()
                if 'Reference' in gemlib.columns:
                    gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                    expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                    if all(col in gemlib.columns for col in expected_columns):
                        gemlib['Gem Description'] = gemlib[expected_columns].apply(
                            lambda x: ' '.join([v if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                        gem_name_map = dict(zip(gemlib['Reference'], gemlib['Gem Description']))
                        print(f"✅ Loaded gem library: {len(gem_name_map)} entries")
                    else:
                        print(f"⚠️ Expected columns {expected_columns} not found in gemlib_structural_ready.csv")
                else:
                    print("⚠️ 'Reference' column not found in gemlib_structural_ready.csv")
            else:
                print(f"⚠️ Gem library not found: {gemlib_path}")
        except Exception as e:
            print(f"⚠️ Could not load gemlib_structural_ready.csv: {e}")
        
        return gem_name_map
    
    def setup_plot_style(self):
        """Configure matplotlib style - same as result_visualizer.py"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        plot_config = {
            'figure.figsize': (18, 6),  # Horizontal layout like gemini1.py
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10
        }
        
        for key, value in plot_config.items():
            plt.rcParams[key] = value
    
    def find_raw_spectral_files(self, gem_id):
        """Find raw spectral files for a gem - adapted from your system"""
        print(f"🔍 Searching for raw spectral files for gem: {gem_id}")
        
        if not self.raw_archive_dir.exists():
            print(f"❌ Raw archive directory not found: {self.raw_archive_dir}")
            return {}
        
        found_files = {}
        light_mapping = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
        
        # Search patterns - match your gemini1.py logic
        search_patterns = [
            f"{gem_id}B*.txt",  # 58BC1.txt
            f"{gem_id}L*.txt",  # 58LC1.txt  
            f"{gem_id}U*.txt",  # 58UC1.txt
            f"{gem_id}_*.txt",  # Alternative format
        ]
        
        for pattern in search_patterns:
            for file_path in self.raw_archive_dir.glob(pattern):
                filename = file_path.stem
                
                # Determine light source from filename
                light_source = None
                for code, name in light_mapping.items():
                    if code in filename.upper():
                        light_source = code
                        break
                
                if light_source:
                    found_files[light_source] = file_path
                    print(f"✅ Found {light_mapping[light_source]} ({light_source}): {file_path.name}")
        
        if not found_files:
            print(f"⚠️ No raw spectral files found for gem {gem_id}")
        
        return found_files
    
    def load_raw_spectrum(self, file_path):
        """Load raw spectrum - same format as gemini1.py"""
        try:
            # Use same logic as your load_spectrum function
            df = pd.read_csv(file_path, header=None, names=['wavelength', 'intensity'])
            return df['wavelength'].values, df['intensity'].values
        except Exception as e:
            print(f"❌ Error loading {file_path.name}: {e}")
            return None, None
    
    def create_comparison_plot(self, stone_gem_id, stone_files, top5_matches):
        """Create comparison plot - matches your gemini1.py plot_and_save_comparison style"""
        
        # Get raw files for stone of interest
        stone_raw_files = self.find_raw_spectral_files(stone_gem_id)
        
        # Create figure - same layout as gemini1.py
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
        
        for i, light_code in enumerate(['B', 'L', 'U']):
            ax = axes[i]
            light_name = light_names[light_code]
            
            try:
                # Plot stone of interest spectrum if available
                if light_code in stone_raw_files:
                    stone_wavelengths, stone_intensities = self.load_raw_spectrum(stone_raw_files[light_code])
                    
                    if stone_wavelengths is not None:
                        ax.plot(stone_wavelengths, stone_intensities, 
                               label=f"Stone of Interest (Gem {stone_gem_id})", 
                               color=self.light_colors[light_code], 
                               linewidth=2, alpha=0.8)
                
                # Plot top match if available
                if top5_matches:
                    top_match_id = top5_matches[0][0]  # Get top match gem ID
                    top_match_score = top5_matches[0][1]['combined_score']
                    
                    top_match_raw_files = self.find_raw_spectral_files(top_match_id)
                    
                    if light_code in top_match_raw_files:
                        match_wavelengths, match_intensities = self.load_raw_spectrum(top_match_raw_files[light_code])
                        
                        if match_wavelengths is not None:
                            gem_desc = self.gem_name_map.get(str(top_match_id), f"Gem {top_match_id}")
                            ax.plot(match_wavelengths, match_intensities, 
                                   label=f"Top Match: {gem_desc}", 
                                   color='black', linestyle='--', 
                                   linewidth=2, alpha=0.8)
                    
                    # Add score annotation - same style as your system
                    breakdown = top5_matches[0][1]['light_breakdown']
                    if light_name in breakdown and breakdown[light_name] is not None:
                        score_pct = breakdown[light_name] * 100
                        score_text = f"Match: {score_pct:.1f}%"
                        ax.text(0.02, 0.98, score_text, transform=ax.transAxes, 
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                
                # Style the plot - same as gemini1.py
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity')
                ax.set_title(f'{light_name} Light Source Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f"{light_name}: Error loading data\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{light_name}: Data Error")
        
        # Overall title
        stone_desc = self.gem_name_map.get(str(stone_gem_id), f"Gem {stone_gem_id}")
        plt.suptitle(f'Structural Analysis: Stone of Interest vs Top Match\n{stone_desc}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot - same naming as gemini1.py
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"structural_comparison_gem_{stone_gem_id}_{timestamp}.png"
        output_path = self.graphs_dir / filename
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory - same as gemini1.py
        
        print(f"📊 Saved structural comparison plot: {filename}")
        return filename
    
    def create_top5_overview_plot(self, stone_gem_id, top5_matches):
        """Create overview plot of all top 5 matches"""
        
        if len(top5_matches) < 2:
            print("⚠️ Need at least 2 matches for overview plot")
            return None
        
        # Create subplot grid for top 5
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        stone_desc = self.gem_name_map.get(str(stone_gem_id), f"Gem {stone_gem_id}")
        
        # Get stone of interest raw files
        stone_raw_files = self.find_raw_spectral_files(stone_gem_id)
        
        for idx, (match_gem_id, match_data) in enumerate(top5_matches[:5]):
            ax = axes[idx]
            
            combined_score = match_data['combined_score']
            match_desc = self.gem_name_map.get(str(match_gem_id), f"Gem {match_gem_id}")
            
            # Get raw files for this match
            match_raw_files = self.find_raw_spectral_files(match_gem_id)
            
            # Plot available light sources
            plotted_sources = []
            for light_code in ['B', 'L', 'U']:
                if light_code in stone_raw_files and light_code in match_raw_files:
                    try:
                        # Stone spectrum
                        stone_wl, stone_int = self.load_raw_spectrum(stone_raw_files[light_code])
                        if stone_wl is not None:
                            ax.plot(stone_wl, stone_int, 
                                   color=self.light_colors[light_code], 
                                   linewidth=1.5, alpha=0.7, 
                                   label=f"Stone ({light_code})")
                        
                        # Match spectrum  
                        match_wl, match_int = self.load_raw_spectrum(match_raw_files[light_code])
                        if match_wl is not None:
                            ax.plot(match_wl, match_int, 
                                   color=self.light_colors[light_code], 
                                   linestyle='--', linewidth=1.5, alpha=0.7,
                                   label=f"Match ({light_code})")
                        
                        plotted_sources.append(light_code)
                        
                    except Exception as e:
                        continue
            
            # Style the subplot
            ax.set_title(f"Rank {idx+1}: {match_desc}\nScore: {combined_score*100:.1f}%", 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Wavelength (nm)', fontsize=9)
            ax.set_ylabel('Intensity', fontsize=9)
            if plotted_sources:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplot
        if len(top5_matches) < 6:
            axes[5].set_visible(False)
        
        plt.suptitle(f'Top 5 Structural Matches Overview\nStone of Interest: {stone_desc}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save overview plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"structural_top5_overview_{stone_gem_id}_{timestamp}.png"
        output_path = self.graphs_dir / filename
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved top 5 overview plot: {filename}")
        return filename
    
    def save_detailed_report(self, stone_gem_id, stone_files, top5_matches):
        """Save detailed analysis report - same style as gemini1.py save_analysis_results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV Report
        results_list = []
        for rank, (gem_id, data) in enumerate(top5_matches, 1):
            gem_desc = self.gem_name_map.get(str(gem_id), f"Gem {gem_id}")
            combined_score = data['combined_score']
            breakdown = data['light_breakdown']
            light_count = data['light_count']
            
            result_row = {
                'rank': rank,
                'gem_id': gem_id,
                'gem_description': gem_desc,
                'combined_score_percent': combined_score * 100,
                'light_sources_matched': light_count,
                'B_score_percent': breakdown.get('Halogen', 0) * 100 if breakdown.get('Halogen') else None,
                'L_score_percent': breakdown.get('Laser', 0) * 100 if breakdown.get('Laser') else None,
                'U_score_percent': breakdown.get('UV', 0) * 100 if breakdown.get('UV') else None,
                'stone_of_interest': stone_gem_id,
                'analysis_type': 'Structural Matching'
            }
            results_list.append(result_row)
        
        # Save CSV
        results_df = pd.DataFrame(results_list)
        csv_filename = f"structural_analysis_gem_{stone_gem_id}_{timestamp}.csv"
        csv_path = self.reports_dir / csv_filename
        results_df.to_csv(csv_path, index=False)
        print(f"📊 Saved detailed results: {csv_filename}")
        
        # JSON Report - same structure as gemini1.py
        json_results = {
            'analysis_timestamp': timestamp,
            'stone_of_interest': stone_gem_id,
            'analysis_type': 'Structural Matching',
            'light_sources_used': list(stone_files.keys()),
            'top_match': {
                'gem_id': top5_matches[0][0],
                'description': self.gem_name_map.get(str(top5_matches[0][0]), f"Gem {top5_matches[0][0]}"),
                'combined_score_percent': top5_matches[0][1]['combined_score'] * 100
            },
            'all_matches': results_list,
            'raw_files_found': {
                'stone_of_interest': len(self.find_raw_spectral_files(stone_gem_id)),
                'matches_with_raw_data': sum(1 for gem_id, _ in top5_matches if self.find_raw_spectral_files(gem_id))
            }
        }
        
        json_filename = f"structural_results_gem_{stone_gem_id}_{timestamp}.json"
        json_path = self.reports_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"📊 Saved JSON results: {json_filename}")
        
        # Summary Text Report - same style as gemini1.py
        summary_filename = f"structural_summary_gem_{stone_gem_id}_{timestamp}.txt"
        summary_path = self.reports_dir / summary_filename
        
        stone_desc = self.gem_name_map.get(str(stone_gem_id), f"Gem {stone_gem_id}")
        
        with open(summary_path, 'w') as f:
            f.write(f"GEMINI STRUCTURAL ANALYSIS RESULTS\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Stone of Interest: {stone_desc} (Gem {stone_gem_id})\n")
            f.write(f"Light Sources Used: {', '.join(stone_files.keys())}\n")
            f.write(f"Analysis Type: Combined Structural Matching\n")
            f.write(f"="*70 + "\n\n")
            
            f.write(f"TOP MATCH:\n")
            top_gem_id, top_data = top5_matches[0]
            top_desc = self.gem_name_map.get(str(top_gem_id), f"Gem {top_gem_id}")
            f.write(f"Gem ID: {top_gem_id}\n")
            f.write(f"Description: {top_desc}\n")
            f.write(f"Combined Score: {top_data['combined_score']*100:.1f}%\n\n")
            
            f.write(f"TOP 5 STRUCTURAL MATCHES:\n")
            f.write(f"-" * 50 + "\n")
            
            for rank, (gem_id, data) in enumerate(top5_matches, 1):
                gem_desc = self.gem_name_map.get(str(gem_id), f"Gem {gem_id}")
                combined = data['combined_score']
                breakdown = data['light_breakdown']
                light_count = data['light_count']
                
                f.write(f"Rank {rank}: {gem_desc} (Gem {gem_id})\n")
                f.write(f"   Combined Score: {combined*100:.1f}%\n")
                f.write(f"   Light sources: {light_count}\n")
                
                for light_name in ['Halogen', 'Laser', 'UV']:
                    if light_name in breakdown and breakdown[light_name] is not None:
                        score = breakdown[light_name] * 100
                        f.write(f"      {light_name}: {score:.1f}%\n")
                f.write("\n")
            
            f.write(f"\nVISUALIZATION FILES:\n")
            f.write(f"- Comparison plots saved to: outputs/structural_results/graphs/\n")
            f.write(f"- Raw spectral data retrieved from: data/raw(archive)/\n")
        
        print(f"📊 Saved summary report: {summary_filename}")
        
        return [csv_filename, json_filename, summary_filename]
    
    def generate_complete_visualization(self, stone_gem_id, stone_files, top5_matches):
        """Generate complete visualization suite - main entry point"""
        print(f"\n📈 GENERATING STRUCTURAL VISUALIZATION SUITE")
        print("=" * 60)
        print(f"Stone of Interest: Gem {stone_gem_id}")
        print(f"Light Sources: {', '.join(stone_files.keys())}")
        print(f"Top Matches: {len(top5_matches)}")
        print(f"Graphs Directory: {self.graphs_dir}")
        print(f"Reports Directory: {self.reports_dir}")
        
        generated_files = []
        
        # 1. Create main comparison plot
        try:
            comparison_plot = self.create_comparison_plot(stone_gem_id, stone_files, top5_matches)
            if comparison_plot:
                generated_files.append(comparison_plot)
        except Exception as e:
            print(f"❌ Error creating comparison plot: {e}")
        
        # 2. Create top 5 overview plot
        try:
            overview_plot = self.create_top5_overview_plot(stone_gem_id, top5_matches)
            if overview_plot:
                generated_files.append(overview_plot)
        except Exception as e:
            print(f"❌ Error creating overview plot: {e}")
        
        # 3. Save detailed reports
        try:
            report_files = self.save_detailed_report(stone_gem_id, stone_files, top5_matches)
            generated_files.extend(report_files)
        except Exception as e:
            print(f"❌ Error saving reports: {e}")
        
        # Summary
        print(f"\n✅ VISUALIZATION COMPLETE!")
        print(f"📁 Generated {len(generated_files)} files:")
        for filename in generated_files:
            print(f"   📄 {filename}")
        
        print(f"\n📊 Output Locations:")
        print(f"   📈 Graphs: {self.graphs_dir}")
        print(f"   📋 Reports: {self.reports_dir}")
        
        return generated_files

def main():
    """Main entry point for testing"""
    print("🎨 STRUCTURAL VISUALIZATION SYSTEM")
    print("=" * 50)
    print("This system generates visualizations for structural matching analysis")
    print("Matches the style and workflow of gemini1.py and result_visualizer.py")
    print(f"Saves to: outputs/structural_results/reports;graphs")
    
    # This would normally be called by the unified analysis system
    print("\n💡 To use this system:")
    print("1. Import: from structural_visualizer import StructuralVisualizer")  
    print("2. Create: visualizer = StructuralVisualizer()")
    print("3. Generate: visualizer.generate_complete_visualization(gem_id, files, top5)")

if __name__ == "__main__":
    main()