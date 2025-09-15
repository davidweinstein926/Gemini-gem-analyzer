#!/usr/bin/env python3
"""
Gemini L Peak Detector - Advanced L-line spectral analysis
Enhanced for new directory structure and improved functionality

Optimized for laser-induced spectra with sharp, well-defined peaks
and high signal-to-noise ratios. Designed for natural/synthetic
discrimination and precision wavelength measurements.

Version: 2.0 (Updated for new directory structure)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import warnings

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

@dataclass
class LSpectralFeature:
    """Represents a detected feature in L spectra"""
    wavelength: float
    intensity: float
    feature_type: str  # 'peak', 'mound_crest', 'mound_start', 'mound_end', 'trough_bottom', 'trough_start', 'trough_end', 'baseline_start', 'baseline_end'
    feature_group: str  # 'peak', 'mound', 'trough', 'baseline'
    prominence: float
    snr: float
    confidence: float  # 0-1 confidence score
    detection_method: str  # 'laser_algorithm', 'region_based', 'hybrid'
    width_nm: float = 0.0
    start_wavelength: float = 0.0
    end_wavelength: float = 0.0
    effective_height: float = 0.0  # Peak height above local baseline

class GeminiLPeakDetector:
    """
    Advanced L-line peak detection system optimized for laser spectra
    Enhanced for new directory structure
    """
    
    def __init__(self):
        self.logger = setup_logger('GeminiLPeakDetector')
        self.file_handler = FileHandler()
        self.data_processor = DataProcessor()
        self.viz_manager = VisualizationManager()
        
        # Laser algorithm thresholds (ADJUSTED for 0-100 normalized scale)
        self.laser_prominence_weak = 1.0      # 1% prominence
        self.laser_prominence_medium = 3.0    # 3% prominence  
        self.laser_prominence_major = 8.0     # 8% prominence
        self.laser_snr_weak = 3.0             # Lower SNR requirements
        self.laser_snr_medium = 5.0
        self.laser_intensity_major = 20.0     # 20% of max intensity
        
        # L spectra specific thresholds (ADJUSTED)
        self.baseline_noise_excellent = 0.01   
        self.baseline_noise_good = 0.05        
        self.baseline_intensity_threshold = 1.0  # Reasonable for 0-100 scale
        
        # Region-based detection parameters (DISABLED for L spectra)
        self.mound_min_width = 200  # Very large to prevent detection
        self.mound_min_prominence = 50.0  # Very high to prevent detection
        self.smoothing_sigma = 0.1  # Minimal smoothing
        
        # Peak detection parameters (REASONABLE for 0-100 scale)
        self.peak_min_distance = 15  # 15nm minimum separation
        self.peak_min_prominence = 2.0  # 2% prominence requirement
        self.peak_width_range = (5, 20)  # Narrower acceptable range
        
        # Analysis data storage
        self.current_data = None
        self.analysis_results = {}
        self.detected_features = []
        
        # Output directory setup
        self.output_dir = Path(project_root) / 'output' / 'l_peak_analysis'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Gemini L Peak Detector initialized")
    
    def load_spectra_data(self, file_path):
        """Load L-line spectral data from file"""
        try:
            # Use the file handler to load data
            self.current_data = self.file_handler.load_spectral_data(file_path)
            if self.current_data is not None:
                self.logger.info(f"Loaded L spectra data: {file_path}")
                return True
            else:
                # Fallback to direct loading if file handler doesn't work
                wavelengths, intensities = self._load_l_spectrum_direct(file_path)
                if wavelengths is not None and intensities is not None:
                    self.current_data = {'wavelength': wavelengths, 'intensity': intensities}
                    self.logger.info(f"Loaded L spectra data (direct): {file_path}")
                    return True
                else:
                    self.logger.error(f"Failed to load data from: {file_path}")
                    return False
        except Exception as e:
            self.logger.error(f"Error loading spectra data: {e}")
            return False
    
    def _load_l_spectrum_direct(self, filepath):
        """
        Direct loading method for L spectrum from file with robust parsing
        Supports both .txt and .csv formats, handles multiple delimiters
        """
        try:
            if filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
                wavelengths = data.iloc[:, 0].values
                intensities = data.iloc[:, 1].values
            else:
                # Try multiple delimiters for .txt files
                try:
                    # First try tab-separated
                    data = np.loadtxt(filepath, delimiter='\t')
                except ValueError:
                    try:
                        # Try space-separated
                        data = np.loadtxt(filepath, delimiter=None)  # None = any whitespace
                    except ValueError:
                        try:
                            # Try comma-separated
                            data = np.loadtxt(filepath, delimiter=',')
                        except ValueError:
                            # Final fallback: pandas with flexible parsing
                            data = pd.read_csv(filepath, sep=None, engine='python', header=None)
                            data = data.values
                
                wavelengths = data[:, 0]
                intensities = data[:, 1]
            
            return wavelengths, intensities
        except Exception as e:
            self.logger.error(f"Error in direct spectrum loading: {e}")
            return None, None
    
    def normalize_l_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """
        Apply L spectra normalization: max intensity → 50000, then scale to 0-100
        """
        # Find maximum intensity in raw data
        max_intensity = np.max(intensities)
        max_idx = np.argmax(intensities)
        max_wavelength = wavelengths[max_idx]
        
        self.logger.debug(f"L normalization: max at {max_wavelength:.2f}nm, intensity = {max_intensity:.6f}")
        self.logger.debug(f"L normalization: min intensity = {np.min(intensities):.6f}")
        
        if max_intensity <= 0:
            raise ValueError(f"Invalid maximum intensity: {max_intensity}")
        
        # Step 1: Normalize so max = 50000
        normalized_to_50000 = (intensities / max_intensity) * 50000
        
        # Step 2: Scale to 0-100 range
        normalized = normalized_to_50000 / 500
        
        self.logger.debug(f"L normalization: After normalization min = {np.min(normalized):.6f}, max = {np.max(normalized):.6f}")
        
        # Store normalization metadata
        self.norm_reference_wavelength = max_wavelength
        self.norm_reference_intensity = max_intensity
        
        return normalized
    
    def assess_baseline_noise(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[float, str]:
        """
        Assess baseline noise level optimized for L spectra - FIXED to use 300-350nm
        Returns: (std_dev, classification)
        """
        # FIXED: Use exact 300-350nm range for baseline assessment
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 350)
        if not np.any(baseline_mask):
            # Fallback: use first 10% of data as baseline
            n_baseline = max(10, len(wavelengths) // 10)
            baseline_mask = np.zeros(len(wavelengths), dtype=bool)
            baseline_mask[:n_baseline] = True
        
        baseline_intensities = intensities[baseline_mask]
        baseline_std = np.std(baseline_intensities)
        
        # Classify noise level (stricter thresholds for L spectra)
        if baseline_std <= self.baseline_noise_excellent:
            classification = "excellent"
        elif baseline_std <= self.baseline_noise_good:
            classification = "good"
        else:
            classification = "poor"
        
        return baseline_std, classification
    
    def detect_l_peaks(self, prominence_threshold=1.0, height_threshold=2.0):
        """Detect L-line peaks using laser algorithm"""
        if self.current_data is None:
            self.logger.error("No data loaded for peak detection")
            return False
        
        try:
            # Extract data
            if isinstance(self.current_data, dict):
                wavelengths = self.current_data['wavelength']
                intensities = self.current_data['intensity']
            else:
                wavelengths = self.current_data[:, 0]
                intensities = self.current_data[:, 1]
            
            # Normalize spectrum
            normalized_intensities = self.normalize_l_spectrum(wavelengths, intensities)
            
            # Apply laser algorithm detection
            self.detected_features = self.laser_algorithm_detection(wavelengths, normalized_intensities)
            
            # Add baseline features
            baseline_features = self.detect_baseline(wavelengths, normalized_intensities)
            self.detected_features.extend(baseline_features)
            
            # Sort by wavelength
            self.detected_features.sort(key=lambda x: x.wavelength)
            
            self.logger.info(f"Detected {len([f for f in self.detected_features if f.feature_group == 'peak'])} L-line peaks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in L peak detection: {e}")
            return False
    
    def laser_algorithm_detection(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[LSpectralFeature]:
        """
        Apply optimized laser algorithm for sharp peak detection
        """
        features = []
        
        # Convert distance from nm to index units
        wavelength_step = np.mean(np.diff(wavelengths))
        min_distance_idx = max(1, int(self.peak_min_distance / wavelength_step))
        
        # Calculate dynamic threshold based on baseline noise (300-350nm region)
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 350)
        if np.any(baseline_mask):
            baseline_max = np.max(intensities[baseline_mask])
        else:
            baseline_max = np.min(intensities)
        
        # FIXED: Use absolute intensity threshold of 2.0 (normalized scale 0-100)
        absolute_threshold = 2.0
        self.logger.debug(f"Using absolute threshold = {absolute_threshold} (normalized scale)")
        
        # Find peaks using more permissive settings to catch small but real peaks
        peaks, properties = signal.find_peaks(intensities, 
                                            height=absolute_threshold,  # Must be >2.0 normalized
                                            prominence=0.2,  # Lower to catch small peaks like 575nm
                                            distance=min_distance_idx,  # 15nm minimum spacing
                                            width=1)
        
        self.logger.debug(f"scipy.find_peaks found {len(peaks)} candidates above {absolute_threshold}")
        
        # Filter peaks with absolute intensity threshold (already done in find_peaks, but double-check)
        filtered_peaks = []
        for peak_idx in peaks:
            intensity = intensities[peak_idx]
            
            # Must be above 2.0 normalized intensity  
            if intensity >= absolute_threshold:
                filtered_peaks.append(peak_idx)
        
        self.logger.debug(f"After intensity filtering: {len(filtered_peaks)} peaks remain")
        
        # FIXED: Merge peaks within 1nm (same peak)
        merged_peaks = []
        if len(filtered_peaks) > 0:
            # Sort by wavelength
            filtered_peaks.sort(key=lambda idx: wavelengths[idx])
            
            merged_peaks.append(filtered_peaks[0])  # Always keep first peak
            
            for peak_idx in filtered_peaks[1:]:
                current_wavelength = wavelengths[peak_idx]
                last_wavelength = wavelengths[merged_peaks[-1]]
                
                # If within 1nm of previous peak, it's the same peak
                if abs(current_wavelength - last_wavelength) <= 1.0:
                    # Keep the higher intensity peak
                    if intensities[peak_idx] > intensities[merged_peaks[-1]]:
                        merged_peaks[-1] = peak_idx
                else:
                    # Different peak, add it
                    merged_peaks.append(peak_idx)
        
        self.logger.debug(f"After merging within 1nm: {len(merged_peaks)} peaks remain")
        
        # Sort final peaks by wavelength for consistent output
        merged_peaks.sort(key=lambda idx: wavelengths[idx])
        
        for peak_idx in merged_peaks:
            wavelength = wavelengths[peak_idx]
            intensity = intensities[peak_idx]
            
            # FILTER 1: Ignore peaks below 400nm (UV artifacts in laser spectra)
            if wavelength < 400.0:
                self.logger.debug(f"Rejecting peak at {wavelength:.1f}nm - below 400nm threshold")
                continue
            
            # Calculate prominence and SNR
            prominence = self._calculate_prominence(intensities, peak_idx)
            snr = self._calculate_snr(intensities, peak_idx)
            
            # Calculate peak width
            width_nm = self._calculate_peak_width(wavelengths, intensities, peak_idx)
            
            # FILTER 2: Ignore peaks with invalid widths
            if width_nm > 100.0:
                self.logger.debug(f"Rejecting peak at {wavelength:.1f}nm - width {width_nm:.1f}nm too large (>100nm)")
                continue
            elif width_nm < 1.0:
                self.logger.debug(f"Rejecting peak at {wavelength:.1f}nm - width {width_nm:.1f}nm too small (<1nm)")  
                continue
            
            # FILTER 3: Calculate baseline-corrected intensity for better assessment
            baseline_corrected_intensity = intensity - self._estimate_local_baseline(intensities, peak_idx)
            
            # Use baseline-corrected intensity for quality assessment
            if baseline_corrected_intensity < 1.0:  # Peak must be >1% above local baseline
                self.logger.debug(f"Rejecting peak at {wavelength:.1f}nm - baseline-corrected intensity {baseline_corrected_intensity:.2f} too low")
                continue
            
            # Calculate peak width and top flatness
            width_nm = self._calculate_peak_width(wavelengths, intensities, peak_idx)
            flatness_ratio = self._calculate_top_flatness(intensities, peak_idx)
            
            # ACCEPT ALL PEAKS that pass filters - classify by width AND flatness
            if width_nm > 25.0 and flatness_ratio > 0.6:
                # Wide features with flat tops are mounds
                feature_type = "mound_crest"
                feature_group = "mound"
            else:
                # Everything else is a peak (narrow, or wide but pointed)
                feature_type = "peak"
                feature_group = "peak"
            
            # Determine confidence based on prominence and intensity
            if prominence > self.laser_prominence_major and baseline_corrected_intensity > self.laser_intensity_major:
                confidence = 0.95
            elif prominence >= self.laser_prominence_medium and snr >= self.laser_snr_medium:
                confidence = 0.8
            elif prominence >= self.laser_prominence_weak and snr >= self.laser_snr_weak:
                confidence = 0.6
            else:
                confidence = 0.4  # Low confidence but still keep the peak
            
            feature = LSpectralFeature(
                wavelength=wavelength,
                intensity=intensity,  # Use raw normalized intensity for consistency with graph
                feature_type=feature_type,
                feature_group=feature_group,
                prominence=prominence,
                snr=snr,
                confidence=confidence,
                detection_method="laser_algorithm",
                width_nm=width_nm,
                effective_height=baseline_corrected_intensity  # Peak height above local baseline
            )
            features.append(feature)
            
            self.logger.debug(f"Accepted {feature_type} at {wavelength:.1f}nm - raw: {intensity:.2f}, effective height: {baseline_corrected_intensity:.2f}, width: {width_nm:.1f}nm")
        
        return features
    
    def detect_baseline(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[LSpectralFeature]:
        """Detect baseline start and end points for L spectra - FIXED to 300-350nm"""
        features = []
        
        # FIXED: Force baseline region to exactly 300-350nm as specified
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 350)
        
        if not np.any(baseline_mask):
            # Fallback if no data in 300-350nm range
            self.logger.warning("No data in 300-350nm range, using first 10% of spectrum")
            n_baseline = len(wavelengths) // 10
            baseline_mask = np.zeros(len(wavelengths), dtype=bool)
            baseline_mask[:n_baseline] = True
        
        baseline_wavelengths = wavelengths[baseline_mask]
        baseline_intensities = intensities[baseline_mask]
        
        if len(baseline_wavelengths) < 2:
            return features
        
        # Find actual start and end points within 300-350nm range
        start_idx = 0
        end_idx = len(baseline_intensities) - 1
        
        start_wavelength = baseline_wavelengths[start_idx]
        start_intensity = baseline_intensities[start_idx]
        end_wavelength = baseline_wavelengths[end_idx]
        end_intensity = baseline_intensities[end_idx]
        
        # Calculate baseline quality metrics
        baseline_std = np.std(baseline_intensities)
        snr = 1.0 / (baseline_std + 1e-6)
        
        features.extend([
            LSpectralFeature(
                wavelength=start_wavelength,
                intensity=start_intensity,
                feature_type="baseline_start",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9 if baseline_std < 0.02 else 0.6,
                detection_method="region_based"
            ),
            LSpectralFeature(
                wavelength=end_wavelength,
                intensity=end_intensity,
                feature_type="baseline_end",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9 if baseline_std < 0.02 else 0.6,
                detection_method="region_based"
            )
        ])
        
        return features
    
    def analyze_l_characteristics(self):
        """Analyze L-line characteristics and mineral associations"""
        if not self.detected_features:
            self.logger.error("No features detected for analysis")
            return False
        
        try:
            # L-line characteristic wavelengths and associations
            l_line_references = {
                'Al_L': {'wavelength': 72.4, 'mineral': 'Aluminum-bearing minerals'},
                'Si_L': {'wavelength': 91.5, 'mineral': 'Silicate minerals'},
                'P_L': {'wavelength': 120.0, 'mineral': 'Phosphate minerals'},
                'S_L': {'wavelength': 149.0, 'mineral': 'Sulfate/sulfide minerals'},
                'Cl_L': {'wavelength': 181.0, 'mineral': 'Chloride minerals'},
                'K_L': {'wavelength': 297.3, 'mineral': 'Potassium feldspars'},
                'Ca_L': {'wavelength': 341.3, 'mineral': 'Calcium-bearing minerals'},
                'Ti_L': {'wavelength': 456.8, 'mineral': 'Titanium minerals'},
                'V_L': {'wavelength': 511.3, 'mineral': 'Vanadium minerals'},
                'Cr_L': {'wavelength': 572.8, 'mineral': 'Chromium minerals'},
                'Mn_L': {'wavelength': 637.4, 'mineral': 'Manganese minerals'},
                'Fe_L': {'wavelength': 705.0, 'mineral': 'Iron-bearing minerals'},
                'Co_L': {'wavelength': 776.2, 'mineral': 'Cobalt minerals'},
                'Ni_L': {'wavelength': 851.5, 'mineral': 'Nickel minerals'},
                'Cu_L': {'wavelength': 929.7, 'mineral': 'Copper minerals'},
                'Zn_L': {'wavelength': 1011.7, 'mineral': 'Zinc minerals'}
            }
            
            self.analysis_results = {
                'identified_elements': [],
                'mineral_associations': [],
                'feature_statistics': {
                    'total_features': len(self.detected_features),
                    'peaks': len([f for f in self.detected_features if f.feature_group == 'peak']),
                    'baseline_points': len([f for f in self.detected_features if f.feature_group == 'baseline'])
                }
            }
            
            # Match peaks to known L-line wavelengths
            tolerance = 5.0  # nm tolerance for L-lines
            peak_features = [f for f in self.detected_features if f.feature_group == 'peak']
            
            for feature in peak_features:
                best_match = None
                min_difference = float('inf')
                
                for element, ref_data in l_line_references.items():
                    difference = abs(feature.wavelength - ref_data['wavelength'])
                    if difference < tolerance and difference < min_difference:
                        min_difference = difference
                        best_match = (element, ref_data)
                
                if best_match:
                    element, ref_data = best_match
                    self.analysis_results['identified_elements'].append({
                        'element': element,
                        'wavelength_detected': feature.wavelength,
                        'wavelength_reference': ref_data['wavelength'],
                        'difference': min_difference,
                        'intensity': feature.intensity,
                        'confidence': feature.confidence,
                        'mineral_type': ref_data['mineral']
                    })
                    
                    if ref_data['mineral'] not in self.analysis_results['mineral_associations']:
                        self.analysis_results['mineral_associations'].append(ref_data['mineral'])
            
            self.logger.info(f"L-line analysis complete: {len(self.analysis_results['identified_elements'])} elements identified")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in L-line characteristic analysis: {e}")
            return False
    
    def create_interactive_plot(self):
        """Create interactive plot for L-peak analysis"""
        if self.current_data is None:
            self.logger.error("No data available for plotting")
            return
        
        try:
            # Extract data for plotting
            if isinstance(self.current_data, dict):
                wavelengths = self.current_data['wavelength']
                intensities = self.current_data['intensity']
            else:
                wavelengths = self.current_data[:, 0]
                intensities = self.current_data[:, 1]
            
            # Normalize for display
            normalized_intensities = self.normalize_l_spectrum(wavelengths, intensities)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(16, 10))
            fig.suptitle('Gemini L-Line Peak Detector - Interactive Analysis', fontsize=16, fontweight='bold')
            
            # Plot spectrum
            ax.plot(wavelengths, normalized_intensities, 'b-', linewidth=1.5, label='L-line Spectrum (Normalized)')
            ax.set_xlabel('Wavelength (nm)', fontsize=12)
            ax.set_ylabel('Intensity (normalized)', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Mark detected features
            if self.detected_features:
                peak_features = [f for f in self.detected_features if f.feature_group == 'peak']
                baseline_features = [f for f in self.detected_features if f.feature_group == 'baseline']
                
                if peak_features:
                    peak_wavelengths = [f.wavelength for f in peak_features]
                    peak_intensities = [f.intensity for f in peak_features]
                    ax.scatter(peak_wavelengths, peak_intensities, c='red', s=100, marker='v', 
                              label=f'Detected L Peaks ({len(peak_features)})', zorder=5)
                    
                    # Annotate peaks
                    for feature in peak_features:
                        ax.annotate(f"{feature.wavelength:.1f} nm\n({feature.confidence:.2f})", 
                                  (feature.wavelength, feature.intensity),
                                  xytext=(10, 10), textcoords='offset points',
                                  fontsize=8, ha='left',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                if baseline_features:
                    baseline_wavelengths = [f.wavelength for f in baseline_features]
                    baseline_intensities = [f.intensity for f in baseline_features]
                    ax.scatter(baseline_wavelengths, baseline_intensities, c='green', s=50, marker='s', 
                              label=f'Baseline Points ({len(baseline_features)})', zorder=4)
            
            ax.legend(loc='upper right')
            
            # Add analysis results text box
            if hasattr(self, 'analysis_results') and self.analysis_results:
                results_text = f"Elements Identified: {len(self.analysis_results.get('identified_elements', []))}\n"
                results_text += f"Mineral Types: {len(self.analysis_results.get('mineral_associations', []))}\n"
                results_text += f"Total Features: {self.analysis_results.get('feature_statistics', {}).get('total_features', 0)}"
                
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
            # Save detected features
            if self.detected_features:
                features_data = []
                for feature in self.detected_features:
                    features_data.append({
                        'wavelength': feature.wavelength,
                        'intensity': feature.intensity,
                        'feature_type': feature.feature_type,
                        'feature_group': feature.feature_group,
                        'prominence': feature.prominence,
                        'snr': feature.snr,
                        'confidence': feature.confidence,
                        'detection_method': feature.detection_method,
                        'width_nm': feature.width_nm,
                        'effective_height': feature.effective_height
                    })
                
                features_df = pd.DataFrame(features_data)
                features_file = self.output_dir / 'l_features_detected.csv'
                features_df.to_csv(features_file, index=False)
                self.logger.info(f"Features data saved to: {features_file}")
            
            # Save analysis results
            if self.analysis_results:
                # Element identification results
                if self.analysis_results.get('identified_elements'):
                    elements_df = pd.DataFrame(self.analysis_results['identified_elements'])
                    elements_file = self.output_dir / 'l_elements_identified.csv'
                    elements_df.to_csv(elements_file, index=False)
                
                # Summary report
                report_file = self.output_dir / 'l_analysis_summary.txt'
                with open(report_file, 'w') as f:
                    f.write("=== Gemini L-Line Peak Analysis Summary ===\n\n")
                    f.write(f"Total features detected: {len(self.detected_features)}\n")
                    f.write(f"Peak features: {len([f for f in self.detected_features if f.feature_group == 'peak'])}\n")
                    f.write(f"Elements identified: {len(self.analysis_results.get('identified_elements', []))}\n")
                    f.write(f"Mineral associations: {', '.join(self.analysis_results.get('mineral_associations', []))}\n\n")
                    
                    if self.analysis_results.get('identified_elements'):
                        f.write("Detected Elements:\n")
                        for element_data in self.analysis_results['identified_elements']:
                            f.write(f"  - {element_data['element']}: {element_data['wavelength_detected']:.1f} nm "
                                  f"(ref: {element_data['wavelength_reference']:.1f} nm)\n")
                
                self.logger.info(f"Analysis results saved to: {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def run_interactive_analysis(self):
        """Main interactive analysis workflow"""
        print("=== Gemini L-Line Peak Detector ===")
        print("Enhanced for new directory structure")
        print("Optimized for laser-induced high-resolution spectra")
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
                    print("Detecting L-line peaks...")
                    if self.detect_l_peaks():
                        print("Analyzing L-line characteristics...")
                        if self.analyze_l_characteristics():
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
    
    # Helper methods
    def _estimate_local_baseline(self, intensities: np.ndarray, peak_idx: int, window: int = 20) -> float:
        """Estimate local baseline around a peak for baseline-corrected intensity"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        local_region = intensities[start:end]
        
        # Find the minimum values on left and right sides of peak
        left_idx = peak_idx - start
        left_region = local_region[:left_idx] if left_idx > 0 else np.array([intensities[peak_idx]])
        right_region = local_region[left_idx + 1:] if left_idx < len(local_region) - 1 else np.array([intensities[peak_idx]])
        
        # Estimate baseline as average of minimum regions on both sides
        left_baseline = np.min(left_region) if len(left_region) > 0 else intensities[peak_idx]
        right_baseline = np.min(right_region) if len(right_region) > 0 else intensities[peak_idx]
        
        return (left_baseline + right_baseline) / 2
    
    def _calculate_prominence(self, intensities: np.ndarray, peak_idx: int, window: int = 10) -> float:
        """Calculate prominence of a peak (smaller window for L spectra)"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        local_region = intensities[start:end]
        peak_value = intensities[peak_idx]
        
        left_idx = peak_idx - start
        left_min = np.min(local_region[:left_idx]) if left_idx > 0 else peak_value
        right_min = np.min(local_region[left_idx + 1:]) if left_idx < len(local_region) - 1 else peak_value
        
        baseline = max(left_min, right_min)
        return max(0, peak_value - baseline)
    
    def _calculate_snr(self, intensities: np.ndarray, peak_idx: int, window: int = 10) -> float:
        """Calculate signal-to-noise ratio (smaller window for L spectra)"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        local_region = intensities[start:end]
        peak_value = intensities[peak_idx]
        
        local_avg = np.mean(local_region)
        noise_points = local_region[local_region < local_avg]
        
        if len(noise_points) > 1:
            noise_level = np.std(noise_points)
            return peak_value / noise_level if noise_level > 0 else float('inf')
        
        return peak_value
    
    def _calculate_peak_width(self, wavelengths: np.ndarray, intensities: np.ndarray, peak_idx: int) -> float:
        """Calculate peak width using slope analysis for peaks on shoulders"""
        
        # Method 1: Slope-based detection for overlapping peaks
        left_idx = peak_idx
        right_idx = peak_idx
        
        # Find left boundary - where declining slope starts rising
        for i in range(peak_idx - 1, max(0, peak_idx - 10), -1):
            if i > 0:
                # Calculate slopes
                slope_before = intensities[i] - intensities[i-1]
                slope_after = intensities[i+1] - intensities[i]
                
                # Found where decline transitions to rise
                if slope_before <= 0 and slope_after > 0:
                    left_idx = i
                    break
        
        # Find right boundary - where rising slope resumes declining
        for i in range(peak_idx + 1, min(len(intensities), peak_idx + 10)):
            if i < len(intensities) - 1:
                # Calculate slopes
                slope_before = intensities[i] - intensities[i-1] 
                slope_after = intensities[i+1] - intensities[i]
                
                # Found where rise transitions back to decline
                if slope_before > 0 and slope_after <= 0:
                    right_idx = i
                    break
        
        # Fallback to half-maximum if slope method finds nothing
        if left_idx == peak_idx and right_idx == peak_idx:
            peak_intensity = intensities[peak_idx]
            half_max = peak_intensity * 0.5
            
            for i in range(peak_idx - 1, -1, -1):
                if intensities[i] < half_max:
                    left_idx = i
                    break
                    
            for i in range(peak_idx + 1, len(intensities)):
                if intensities[i] < half_max:
                    right_idx = i
                    break
        
        return abs(wavelengths[right_idx] - wavelengths[left_idx])
    
    def _calculate_top_flatness(self, intensities: np.ndarray, peak_idx: int, window: int = 3) -> float:
        """Measure how flat the peak top is (mounds have flat tops, peaks are pointed)"""
        start = max(0, peak_idx - window)  
        end = min(len(intensities), peak_idx + window + 1)
        
        peak_region = intensities[start:end]
        peak_max = intensities[peak_idx]
        
        # Count points within 5% of peak maximum (indicates flat top)
        tolerance = peak_max * 0.05
        flat_points = np.sum(np.abs(peak_region - peak_max) <= tolerance)
        
        # Return ratio of flat points to total points in region
        flatness_ratio = flat_points / len(peak_region)
        
        return flatness_ratio

def main():
    """Main execution function"""
    detector = GeminiLPeakDetector()
    detector.run_interactive_analysis()

if __name__ == "__main__":
    main()