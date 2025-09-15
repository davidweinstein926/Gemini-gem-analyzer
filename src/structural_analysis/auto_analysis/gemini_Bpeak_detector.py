#!/usr/bin/env python3
"""
Gemini B Peak Detector - Advanced B-line spectral analysis
Enhanced for new directory structure and improved functionality
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
import pandas as pd
from pathlib import Path

# Add parent directories to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
project_root = src_dir.parent
sys.path.extend([str(project_root), str(src_dir)])

# Import project modules
try:
    from config.settings import *
    from utils.file_handler import FileHandler
    from utils.data_processor import DataProcessor
    from utils.visualization import VisualizationManager
    from utils.logger import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are in the correct directory structure")
    sys.exit(1)

class GeminiBPeakDetector:
    """Advanced B-line peak detection and analysis system"""
    
    def __init__(self):
        self.logger = setup_logger('GeminiBPeakDetector')
        self.file_handler = FileHandler()
        self.data_processor = DataProcessor()
        self.viz_manager = VisualizationManager()
        
        # Analysis parameters
        self.current_data = None
        self.peaks_detected = []
        self.analysis_results = {}
        self.output_dir = Path(project_root) / 'output' / 'b_peak_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Gemini B Peak Detector initialized")
    
    def load_spectra_data(self, file_path):
        """Load B-line spectral data from file"""
        try:
            self.current_data = self.file_handler.load_spectral_data(file_path)
            if self.current_data is not None:
                self.logger.info(f"Loaded B spectra data: {file_path}")
                return True
            else:
                self.logger.error(f"Failed to load data from: {file_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error loading spectra data: {e}")
            return False
    
    def detect_b_peaks(self, prominence_threshold=0.1, height_threshold=None):
        """Detect B-line peaks in the spectral data"""
        if self.current_data is None:
            self.logger.error("No data loaded for peak detection")
            return False
        
        try:
            from scipy.signal import find_peaks, peak_prominences, peak_widths
            
            # Extract intensity data
            if isinstance(self.current_data, dict):
                intensities = self.current_data.get('intensity', [])
                wavelengths = self.current_data.get('wavelength', [])
            else:
                intensities = self.current_data[:, 1] if self.current_data.shape[1] > 1 else self.current_data
                wavelengths = self.current_data[:, 0] if self.current_data.shape[1] > 1 else np.arange(len(intensities))
            
            # Peak detection
            peaks, properties = find_peaks(
                intensities, 
                prominence=prominence_threshold,
                height=height_threshold
            )
            
            # Calculate additional peak properties
            prominences = peak_prominences(intensities, peaks)[0]
            widths = peak_widths(intensities, peaks, rel_height=0.5)[0]
            
            self.peaks_detected = []
            for i, peak_idx in enumerate(peaks):
                peak_data = {
                    'index': peak_idx,
                    'wavelength': wavelengths[peak_idx],
                    'intensity': intensities[peak_idx],
                    'prominence': prominences[i],
                    'width': widths[i],
                    'type': 'B-line'
                }
                self.peaks_detected.append(peak_data)
            
            self.logger.info(f"Detected {len(self.peaks_detected)} B-line peaks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in B peak detection: {e}")
            return False
    
    def analyze_b_characteristics(self):
        """Analyze B-line characteristics and mineral associations"""
        if not self.peaks_detected:
            self.logger.error("No peaks detected for analysis")
            return False
        
        try:
            # B-line characteristic wavelengths and associations
            b_line_references = {
                'Al_Ka': {'wavelength': 1.487, 'mineral': 'Aluminum-bearing'},
                'Si_Ka': {'wavelength': 1.740, 'mineral': 'Silicate'},
                'P_Ka': {'wavelength': 2.013, 'mineral': 'Phosphate'},
                'S_Ka': {'wavelength': 2.307, 'mineral': 'Sulfate'},
                'Cl_Ka': {'wavelength': 2.622, 'mineral': 'Chloride'},
                'K_Ka': {'wavelength': 3.312, 'mineral': 'Potassium-bearing'},
                'Ca_Ka': {'wavelength': 3.691, 'mineral': 'Calcium-bearing'},
                'Ti_Ka': {'wavelength': 4.508, 'mineral': 'Titanium-bearing'},
                'V_Ka': {'wavelength': 4.952, 'mineral': 'Vanadium-bearing'},
                'Cr_Ka': {'wavelength': 5.415, 'mineral': 'Chromium-bearing'},
                'Mn_Ka': {'wavelength': 5.899, 'mineral': 'Manganese-bearing'},
                'Fe_Ka': {'wavelength': 6.404, 'mineral': 'Iron-bearing'},
                'Co_Ka': {'wavelength': 6.930, 'mineral': 'Cobalt-bearing'},
                'Ni_Ka': {'wavelength': 7.478, 'mineral': 'Nickel-bearing'},
                'Cu_Ka': {'wavelength': 8.048, 'mineral': 'Copper-bearing'},
                'Zn_Ka': {'wavelength': 8.639, 'mineral': 'Zinc-bearing'}
            }
            
            self.analysis_results = {
                'identified_elements': [],
                'mineral_associations': [],
                'peak_statistics': {
                    'total_peaks': len(self.peaks_detected),
                    'max_intensity': max(peak['intensity'] for peak in self.peaks_detected),
                    'avg_prominence': np.mean([peak['prominence'] for peak in self.peaks_detected])
                }
            }
            
            # Match peaks to known B-line wavelengths
            tolerance = 0.1  # keV tolerance
            for peak in self.peaks_detected:
                best_match = None
                min_difference = float('inf')
                
                for element, ref_data in b_line_references.items():
                    difference = abs(peak['wavelength'] - ref_data['wavelength'])
                    if difference < tolerance and difference < min_difference:
                        min_difference = difference
                        best_match = (element, ref_data)
                
                if best_match:
                    element, ref_data = best_match
                    self.analysis_results['identified_elements'].append({
                        'element': element,
                        'wavelength_detected': peak['wavelength'],
                        'wavelength_reference': ref_data['wavelength'],
                        'difference': min_difference,
                        'intensity': peak['intensity'],
                        'mineral_type': ref_data['mineral']
                    })
                    
                    if ref_data['mineral'] not in self.analysis_results['mineral_associations']:
                        self.analysis_results['mineral_associations'].append(ref_data['mineral'])
            
            self.logger.info(f"B-line analysis complete: {len(self.analysis_results['identified_elements'])} elements identified")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in B-line characteristic analysis: {e}")
            return False
    
    def create_interactive_plot(self):
        """Create interactive plot for B-peak analysis"""
        if self.current_data is None:
            self.logger.error("No data available for plotting")
            return
        
        try:
            # Extract data for plotting
            if isinstance(self.current_data, dict):
                wavelengths = self.current_data.get('wavelength', [])
                intensities = self.current_data.get('intensity', [])
            else:
                wavelengths = self.current_data[:, 0] if self.current_data.shape[1] > 1 else np.arange(len(self.current_data))
                intensities = self.current_data[:, 1] if self.current_data.shape[1] > 1 else self.current_data
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.suptitle('Gemini B-Line Peak Detector - Interactive Analysis', fontsize=16, fontweight='bold')
            
            # Plot spectrum
            ax.plot(wavelengths, intensities, 'b-', linewidth=1.5, label='B-line Spectrum')
            ax.set_xlabel('Energy (keV)', fontsize=12)
            ax.set_ylabel('Intensity (counts)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Mark detected peaks
            if self.peaks_detected:
                peak_wavelengths = [peak['wavelength'] for peak in self.peaks_detected]
                peak_intensities = [peak['intensity'] for peak in self.peaks_detected]
                ax.scatter(peak_wavelengths, peak_intensities, c='red', s=100, marker='v', 
                          label=f'Detected Peaks ({len(self.peaks_detected)})', zorder=5)
                
                # Annotate peaks with element identification
                for peak in self.peaks_detected:
                    ax.annotate(f"{peak['wavelength']:.2f} keV", 
                              (peak['wavelength'], peak['intensity']),
                              xytext=(10, 10), textcoords='offset points',
                              fontsize=8, ha='left',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.legend(loc='upper right')
            
            # Add analysis results text box
            if hasattr(self, 'analysis_results') and self.analysis_results:
                results_text = f"Elements Identified: {len(self.analysis_results.get('identified_elements', []))}\n"
                results_text += f"Mineral Types: {len(self.analysis_results.get('mineral_associations', []))}\n"
                results_text += f"Total Peaks: {self.analysis_results.get('peak_statistics', {}).get('total_peaks', 0)}"
                
                ax.text(0.02, 0.98, results_text, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Add control buttons
            ax_button = plt.axes([0.85, 0.01, 0.13, 0.05])
            button_save = Button(ax_button, 'Save Results')
            
            def save_results(event):
                self.save_analysis_results()
                print("Results saved to output directory")
            
            button_save.on_clicked(save_results)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.error(f"Error creating interactive plot: {e}")
    
    def save_analysis_results(self):
        """Save analysis results to files"""
        try:
            # Save peak data
            if self.peaks_detected:
                peak_df = pd.DataFrame(self.peaks_detected)
                peak_file = self.output_dir / 'b_peaks_detected.csv'
                peak_df.to_csv(peak_file, index=False)
                self.logger.info(f"Peak data saved to: {peak_file}")
            
            # Save analysis results
            if self.analysis_results:
                # Element identification results
                if self.analysis_results.get('identified_elements'):
                    elements_df = pd.DataFrame(self.analysis_results['identified_elements'])
                    elements_file = self.output_dir / 'b_elements_identified.csv'
                    elements_df.to_csv(elements_file, index=False)
                
                # Summary report
                report_file = self.output_dir / 'b_analysis_summary.txt'
                with open(report_file, 'w') as f:
                    f.write("=== Gemini B-Line Peak Analysis Summary ===\n\n")
                    f.write(f"Total peaks detected: {len(self.peaks_detected)}\n")
                    f.write(f"Elements identified: {len(self.analysis_results.get('identified_elements', []))}\n")
                    f.write(f"Mineral associations: {', '.join(self.analysis_results.get('mineral_associations', []))}\n\n")
                    
                    if self.analysis_results.get('identified_elements'):
                        f.write("Detected Elements:\n")
                        for element_data in self.analysis_results['identified_elements']:
                            f.write(f"  - {element_data['element']}: {element_data['wavelength_detected']:.3f} keV "
                                  f"(ref: {element_data['wavelength_reference']:.3f} keV)\n")
                
                self.logger.info(f"Analysis results saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def run_interactive_analysis(self):
        """Main interactive analysis workflow"""
        print("=== Gemini B-Line Peak Detector ===")
        print("Enhanced for new directory structure")
        print()
        
        # File selection
        data_dir = Path(project_root) / 'data' / 'raw'
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            print("Please ensure data files are in the correct location")
            return
        
        # List available files
        spectral_files = list(data_dir.glob('*.txt')) + list(data_dir.glob('*.csv')) + list(data_dir.glob('*.dat'))
        if not spectral_files:
            print("No spectral data files found in data directory")
            return
        
        print("Available spectral files:")
        for i, file in enumerate(spectral_files, 1):
            print(f"{i}. {file.name}")
        
        try:
            choice = int(input("Select file number: ")) - 1
            if 0 <= choice < len(spectral_files):
                selected_file = spectral_files[choice]
                
                # Load and analyze
                if self.load_spectra_data(selected_file):
                    print("Detecting B-line peaks...")
                    if self.detect_b_peaks():
                        print("Analyzing B-line characteristics...")
                        if self.analyze_b_characteristics():
                            print("Creating interactive visualization...")
                            self.create_interactive_plot()
                        else:
                            print("Error in characteristic analysis")
                    else:
                        print("Error in peak detection")
                else:
                    print("Error loading spectral data")
            else:
                print("Invalid selection")
                
        except (ValueError, KeyboardInterrupt):
            print("Analysis cancelled by user")
        except Exception as e:
            self.logger.error(f"Error in interactive analysis: {e}")
            print(f"Analysis error: {e}")

def main():
    """Main execution function"""
    detector = GeminiBPeakDetector()
    detector.run_interactive_analysis()

if __name__ == "__main__":
    main()