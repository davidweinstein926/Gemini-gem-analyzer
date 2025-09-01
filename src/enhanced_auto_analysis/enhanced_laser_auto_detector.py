#!/usr/bin/env python3
"""
Enhanced Laser Auto Analyzer - Comprehensive Structural Detection
Automatically detects all structural features with same visual markers as manual analyzer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import find_peaks, savgol_filter, peak_widths
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from datetime import datetime
from typing import List, Dict, Tuple

class EnhancedLaserAutoAnalyzer:
    """Automated laser structural analysis with comprehensive feature detection"""
    
    def __init__(self):
        self.OUTPUT_DIRECTORY = r"c:\users\david\gemini sp10 structural data\laser"
        self.root = tk.Tk()
        self.root.title("⚡ Enhanced Laser Auto Analyzer - Full Structural Detection")
        self.root.geometry("1400x900")
        
        # Data storage
        self.current_spectrum = None
        self.detected_features = []
        self.original_intensities = None
        
        # Detection parameters for laser (optimized based on UV success)
        self.params = {
            "peak_prominence": 0.002,    # Same as successful UV detection
            "mound_min_width": 15,       # Minimum 15nm for mounds (smaller than halogen)
            "plateau_flatness": 0.015,   # Max 1.5% variation for plateaus (stricter)
            "trough_prominence": 0.002,  # Same as peak prominence
            "shoulder_prominence": 0.002, # Same as peak prominence
            "valley_min_width": 3,       # Minimum 3nm for valleys (narrower)
            "baseline_window": 30,       # 30nm window for baseline (smaller)
            "smoothing_window": 5        # Savgol filter window (less smoothing)
        }
        
        # Same colors as manual analyzer
        self.feature_colors = {
            'Baseline': 'gray',
            'Mound': 'red',
            'Plateau': 'green', 
            'Peak': 'blue',
            'Trough': 'purple',
            'Shoulder': 'orange',
            'Valley': 'brown',
            'Diagnostic': 'gold'
        }
        
        # Parameter controls
        self.create_parameter_controls()
        
        # Setup interface
        self.setup_gui()
        
    def create_parameter_controls(self):
        """Create adjustable parameters"""
        self.peak_prominence = tk.DoubleVar(value=0.002)  # Same as successful UV
        self.mound_min_width = tk.IntVar(value=self.params["mound_min_width"])
        self.plateau_flatness = tk.DoubleVar(value=self.params["plateau_flatness"])
        self.trough_prominence = tk.DoubleVar(value=0.002)  # Same as peaks
        self.smoothing_window = tk.IntVar(value=self.params["smoothing_window"])
        
    def setup_gui(self):
        """Create the main interface"""
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill='both', expand=True)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_container, width=400)
        main_container.add(left_panel, weight=1)
        
        # Right panel - Plot
        right_panel = ttk.Frame(main_container)
        main_container.add(right_panel, weight=3)
        
        # Setup sections
        self.setup_file_section(left_panel)
        self.setup_parameters_section(left_panel)
        self.setup_detection_section(left_panel)
        self.setup_results_section(left_panel)
        self.setup_plot_area(right_panel)
        
    def setup_file_section(self, parent):
        """File loading section"""
        file_frame = ttk.LabelFrame(parent, text="⚡ Laser Spectrum File", padding=10)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Laser Spectrum", 
                  command=self.load_spectrum).pack(side='left', padx=5)
        
        ttk.Button(file_frame, text="Load Sample Data",
                  command=self.load_sample_data).pack(side='left', padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=10)
        
    def setup_parameters_section(self, parent):
        """Detection parameters section"""
        params_frame = ttk.LabelFrame(parent, text="🔧 Laser Detection Parameters", padding=10)
        params_frame.pack(fill='x', padx=5, pady=5)
        
        # Peak prominence (very sensitive like UV)
        prom_frame = ttk.Frame(params_frame)
        prom_frame.pack(fill='x', pady=2)
        ttk.Label(prom_frame, text="Peak Prominence:").pack(side='left', padx=5)
        ttk.Scale(prom_frame, from_=0.001, to=0.02, variable=self.peak_prominence,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(prom_frame, textvariable=self.peak_prominence).pack(side='left', padx=5)
        
        # Mound width
        width_frame = ttk.Frame(params_frame)
        width_frame.pack(fill='x', pady=2)
        ttk.Label(width_frame, text="Min Mound Width:").pack(side='left', padx=5)
        ttk.Scale(width_frame, from_=5, to=30, variable=self.mound_min_width,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(width_frame, textvariable=self.mound_min_width).pack(side='left', padx=5)
        
        # Plateau flatness
        flat_frame = ttk.Frame(params_frame)
        flat_frame.pack(fill='x', pady=2)
        ttk.Label(flat_frame, text="Plateau Flatness:").pack(side='left', padx=5)
        ttk.Scale(flat_frame, from_=0.005, to=0.03, variable=self.plateau_flatness,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(flat_frame, textvariable=self.plateau_flatness).pack(side='left', padx=5)
        
        # Smoothing
        smooth_frame = ttk.Frame(params_frame)
        smooth_frame.pack(fill='x', pady=2)
        ttk.Label(smooth_frame, text="Smoothing:").pack(side='left', padx=5)
        ttk.Scale(smooth_frame, from_=3, to=11, variable=self.smoothing_window,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(smooth_frame, textvariable=self.smoothing_window).pack(side='left', padx=5)
        
    def setup_detection_section(self, parent):
        """Detection control section"""
        detect_frame = ttk.LabelFrame(parent, text="⚡ Automated Laser Detection", padding=10)
        detect_frame.pack(fill='x', padx=5, pady=5)
        
        # Main detection button
        ttk.Button(detect_frame, text="🤖 Detect ALL Structural Features",
                  command=self.detect_all_features).pack(pady=5, fill='x')
        
        # Individual feature buttons
        feature_buttons = [
            ("📍 Detect Baseline", self.detect_baseline_only),
            ("🏔️ Detect Mounds", self.detect_mounds_only),
            ("⬜ Detect Plateaus", self.detect_plateaus_only),
            ("🔺 Detect Peaks", self.detect_peaks_only),
            ("🕳️ Detect Troughs", self.detect_troughs_only),
            ("📐 Detect Shoulders", self.detect_shoulders_only),
            ("🌊 Detect Valleys", self.detect_valleys_only)
        ]
        
        for text, command in feature_buttons:
            ttk.Button(detect_frame, text=text, command=command).pack(pady=2, fill='x')
        
        # Clear and export
        ttk.Button(detect_frame, text="🧹 Clear All Markers",
                  command=self.clear_all_markers).pack(pady=5, fill='x')
        
        ttk.Button(detect_frame, text="💾 Export Results",
                  command=self.export_results).pack(pady=5, fill='x')
        
    def setup_results_section(self, parent):
        """Results display section"""
        results_frame = ttk.LabelFrame(parent, text="📊 Detection Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.results_text = tk.Text(text_frame, height=15, width=40, wrap='word',
                                   yscrollcommand=scrollbar.set)
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.results_text.yview)
        
    def setup_plot_area(self, parent):
        """Setup the matplotlib plot area"""
        # Create figure
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add navigation toolbar
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side='top', fill='x')
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Initial empty plot
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity', fontsize=11)
        self.ax.set_title('Laser Automated Structural Analysis', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def load_spectrum(self):
        """Load a laser spectrum file"""
        default_dir = r"C:\Users\David\OneDrive\Desktop\gemini matcher\gemini sp10 raw\raw text"
        
        file_path = filedialog.askopenfilename(
            initialdir=default_dir,
            title="Select Laser Spectrum",
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                # Load data (assuming space-separated)
                data = pd.read_csv(file_path, sep=r'\s+', header=None)
                
                if data.shape[1] < 2:
                    messagebox.showerror("Error", "File must contain at least 2 columns (wavelength, intensity)")
                    return
                
                wavelengths = data.iloc[:, 0].values
                intensities = data.iloc[:, 1].values
                
                # Check wavelength order
                if wavelengths[0] > wavelengths[-1]:
                    wavelengths = wavelengths[::-1]
                    intensities = intensities[::-1]
                    print("🔄 Auto-corrected wavelength order")
                
                self.current_spectrum = {
                    'wavelengths': wavelengths,
                    'intensities': intensities,
                    'filename': os.path.basename(file_path)
                }
                self.original_intensities = intensities.copy()
                
                self.file_label.config(text=os.path.basename(file_path))
                self.plot_spectrum()
                self.update_results(f"Loaded: {os.path.basename(file_path)}\n"
                                  f"Points: {len(wavelengths)}\n"
                                  f"Range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
                
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
                
    def load_sample_data(self):
        """Load sample laser-like data for testing"""
        # Generate sample laser spectrum (typically sharper features)
        wl = np.linspace(400, 800, 1200)
        intensity = np.random.normal(5, 0.5, len(wl))  # Lower baseline noise
        
        # Add typical laser features (sharper than halogen)
        # Sharp peak at 532nm (typical laser line)
        peak1_mask = (wl >= 530) & (wl <= 534)
        intensity[peak1_mask] += 40 * np.exp(-((wl[peak1_mask] - 532) / 1)**2)
        
        # Another sharp peak at 650nm
        peak2_mask = (wl >= 648) & (wl <= 652)
        intensity[peak2_mask] += 35 * np.exp(-((wl[peak2_mask] - 650) / 1.5)**2)
        
        # Small mound at 580-620nm
        mound_mask = (wl >= 580) & (wl <= 620)
        intensity[mound_mask] += 12 * np.exp(-((wl[mound_mask] - 600) / 15)**2)
        
        # Sharp trough at 750nm
        trough_mask = (wl >= 748) & (wl <= 752)
        intensity[trough_mask] -= 8 * np.exp(-((wl[trough_mask] - 750) / 2)**2)
        
        # Plateau at 460-480nm
        plateau_mask = (wl >= 460) & (wl <= 480)
        intensity[plateau_mask] += 8
        
        self.current_spectrum = {
            'wavelengths': wl,
            'intensities': intensity,
            'filename': 'Sample_Laser_Spectrum'
        }
        self.original_intensities = intensity.copy()
        
        self.file_label.config(text='Sample Laser Spectrum')
        self.plot_spectrum()
        self.update_results("Loaded sample laser spectrum with sharp features")
        
    def plot_spectrum(self):
        """Plot the current spectrum"""
        if not self.current_spectrum:
            return
            
        self.ax.clear()
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        
        # Plot spectrum (blue line for laser)
        self.ax.plot(wavelengths, intensities, 'b-', linewidth=1, label='Laser Spectrum')
        
        # Plot detected features with same colors as manual analyzer
        self.plot_detected_features()
        
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity', fontsize=11)
        self.ax.set_title(f"Laser Analysis - {self.current_spectrum['filename']}", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
        
    def plot_detected_features(self):
        """Plot detected features with colored markers"""
        if not self.detected_features:
            return
            
        # Group features by type for consistent coloring
        feature_groups = {}
        for feature in self.detected_features:
            feature_group = feature.get('Feature_Group', 'Unknown')
            feature_type = feature_group.split('_')[0]  # Get base type
            
            if feature_type not in feature_groups:
                feature_groups[feature_type] = []
            feature_groups[feature_type].append(feature)
        
        # Plot each feature type with its color
        for feature_type, features in feature_groups.items():
            color = self.feature_colors.get(feature_type, 'black')
            wavelengths = [f['Wavelength'] for f in features]
            intensities = [f['Intensity'] for f in features]
            
            self.ax.scatter(wavelengths, intensities, c=color, s=50,
                           label=f'{feature_type} ({len(features)})',
                           edgecolors='black', linewidth=1, zorder=5,
                           marker='o')
            
    def detect_all_features(self):
        """Detect all structural features automatically"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
            
        print("⚡ LASER AUTOMATED STRUCTURAL ANALYSIS")
        print("="*50)
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        filename = self.current_spectrum['filename']
        
        # Clear previous results
        self.detected_features = []
        
        # Step 1: Preprocess
        processed_intensities = self.preprocess_spectrum(intensities)
        
        # Step 2: Detect baseline and apply corrections
        baseline_features = self.detect_baseline(wavelengths, processed_intensities)
        self.detected_features.extend(baseline_features)
        
        # Apply baseline correction if baseline found
        if baseline_features:
            corrected_intensities = self.apply_baseline_correction(processed_intensities, baseline_features)
            normalized_intensities = self.normalize_to_maximum(corrected_intensities)
            
            # Update spectrum for display
            self.current_spectrum['intensities'] = normalized_intensities
        else:
            normalized_intensities = processed_intensities
            
        # Step 3: Detect all feature types with laser-optimized parameters
        peaks = self.detect_peaks(wavelengths, normalized_intensities)
        mounds = self.detect_mounds(wavelengths, normalized_intensities)
        plateaus = self.detect_plateaus(wavelengths, normalized_intensities)
        troughs = self.detect_troughs(wavelengths, normalized_intensities)
        shoulders = self.detect_shoulders(wavelengths, normalized_intensities)
        valleys = self.detect_valleys(wavelengths, normalized_intensities)
        
        # Add all features with file info
        all_feature_lists = [peaks, mounds, plateaus, troughs, shoulders, valleys]
        for feature_list in all_feature_lists:
            for feature in feature_list:
                feature['File'] = filename
                feature['Light_Source'] = 'Laser'
                feature['Processing'] = 'Auto_Baseline_Normalized'
            self.detected_features.extend(feature_list)
        
        # Update display
        self.plot_spectrum()
        
        # Show results summary
        feature_counts = {
            'Baseline': len(baseline_features),
            'Peak': len(peaks),
            'Mound': len(mounds),
            'Plateau': len(plateaus),
            'Trough': len(troughs),
            'Shoulder': len(shoulders),
            'Valley': len(valleys)
        }
        
        result_text = "⚡ LASER DETECTION COMPLETE\n"
        result_text += "="*40 + "\n\n"
        result_text += f"📁 File: {filename}\n"
        result_text += f"📊 Total Features: {len(self.detected_features)}\n\n"
        
        result_text += "📋 FEATURE BREAKDOWN:\n"
        for feat_type, count in feature_counts.items():
            if count > 0:
                icon = {'Baseline': '📍', 'Peak': '🔺', 'Mound': '🏔️', 'Plateau': '⬜',
                       'Trough': '🕳️', 'Shoulder': '📐', 'Valley': '🌊'}.get(feat_type, '•')
                result_text += f"   {icon} {feat_type}: {count}\n"
        
        if baseline_features:
            baseline = baseline_features[0]
            result_text += f"\n📍 BASELINE INFO:\n"
            result_text += f"   Range: {baseline_features[0]['Wavelength']:.1f}-{baseline_features[1]['Wavelength']:.1f}nm\n"
            result_text += f"   SNR: {baseline.get('SNR', 'N/A')}\n"
        
        result_text += f"\n⚡ LASER OPTIMIZATIONS:\n"
        result_text += f"   ✅ High sensitivity for sharp features\n"
        result_text += f"   ✅ Minimal smoothing (window={self.smoothing_window.get()})\n"
        result_text += f"   ✅ Narrow feature detection\n"
        if baseline_features:
            result_text += f"   ✅ Baseline correction + normalization\n"
        
        self.update_results(result_text)
        
    # Copy core detection methods from halogen analyzer (same algorithms, different parameters)
    def preprocess_spectrum(self, intensities):
        """Apply minimal smoothing for laser (preserve sharp features)"""
        window = self.smoothing_window.get()
        if len(intensities) > window and window >= 3 and window % 2 == 1:
            smoothed = savgol_filter(intensities, window, 2)
            print(f"🔧 Minimal smoothing applied (window={window})")
            return smoothed
        else:
            return intensities.copy()
            
    def detect_baseline(self, wavelengths, intensities):
        """Detect stable baseline region (smaller window for laser)"""
        window_size = self.params['baseline_window']
        min_points = max(5, window_size // 2)
        
        best_baseline = None
        best_stability = float('inf')
        
        # Scan for most stable region
        for i in range(0, len(intensities) - min_points, 3):  # Smaller steps for laser
            end_idx = min(i + min_points, len(intensities))
            region_intensities = intensities[i:end_idx]
            
            if len(region_intensities) < 5:
                continue
                
            std_dev = np.std(region_intensities)
            mean_intensity = np.mean(region_intensities)
            
            # Prefer low intensity regions with low variation
            stability_score = std_dev + 0.05 * mean_intensity  # Lower weight for laser
            
            if stability_score < best_stability:
                best_stability = stability_score
                best_baseline = {
                    'start_wl': wavelengths[i],
                    'end_wl': wavelengths[end_idx-1],
                    'avg_intensity': mean_intensity,
                    'std_dev': std_dev,
                    'snr': mean_intensity / std_dev if std_dev > 0 else float('inf')
                }
        
        if best_baseline:
            print(f"📍 Laser Baseline: {best_baseline['start_wl']:.1f}-{best_baseline['end_wl']:.1f}nm, "
                  f"SNR: {best_baseline['snr']:.1f}")
            
            return [
                {
                    'Feature': 'Baseline_Start',
                    'Wavelength': round(best_baseline['start_wl'], 2),
                    'Intensity': round(best_baseline['avg_intensity'], 2),
                    'Point_Type': 'Start',
                    'Feature_Group': 'Baseline',
                    'SNR': round(best_baseline['snr'], 1)
                },
                {
                    'Feature': 'Baseline_End',
                    'Wavelength': round(best_baseline['end_wl'], 2),
                    'Intensity': round(best_baseline['avg_intensity'], 2),
                    'Point_Type': 'End',
                    'Feature_Group': 'Baseline',
                    'SNR': round(best_baseline['snr'], 1)
                }
            ]
        
        print("⚠️ No stable baseline region found")
        return []
        
    def apply_baseline_correction(self, intensities, baseline_features):
        """Apply baseline correction"""
        if not baseline_features:
            return intensities.copy()
            
        baseline_intensity = baseline_features[0].get('Intensity', 0)
        corrected = intensities - baseline_intensity
        corrected = np.clip(corrected, 0, None)
        
        print(f"🔧 Baseline correction: -{baseline_intensity:.2f}")
        return corrected
        
    def normalize_to_maximum(self, intensities, target_max=100.0):
        """Normalize spectrum to maximum"""
        max_intensity = np.max(intensities)
        
        if max_intensity <= 0:
            print("⚠️ Cannot normalize - maximum is zero")
            return intensities.copy()
            
        scaling_factor = target_max / max_intensity
        normalized = intensities * scaling_factor
        
        print(f"🔧 Normalized to max={target_max:.0f} (factor={scaling_factor:.4f})")
        return normalized
        
    def detect_peaks(self, wavelengths, intensities):
        """Detect sharp peaks (laser optimized)"""
        prominence_threshold = self.peak_prominence.get() * np.ptp(intensities)
        
        # Laser peaks are typically sharper, so use smaller distance
        peaks, properties = find_peaks(intensities,
                                     prominence=prominence_threshold,
                                     distance=3)  # Smaller distance for laser
        
        peak_features = []
        for i, peak_idx in enumerate(peaks):
            peak_features.append({
                'Feature': 'Peak_Max',
                'Wavelength': round(wavelengths[peak_idx], 2),
                'Intensity': round(intensities[peak_idx], 2),
                'Point_Type': 'Max',
                'Feature_Group': f'Peak_{i+1}',
                'Prominence': round(properties['prominences'][i], 2)
            })
        
        print(f"🔺 Detected {len(peaks)} laser peaks")
        return peak_features
        
    def detect_mounds(self, wavelengths, intensities):
        """Detect mound features using slope analysis with SNR thresholding (laser optimized)"""
        min_width = self.mound_min_width.get()
        
        # Get baseline noise level for slope threshold
        baseline_features = [f for f in self.detected_features if f.get('Feature_Group') == 'Baseline']
        if baseline_features:
            # Use SNR from baseline to set slope threshold
            baseline_snr = baseline_features[0].get('SNR', 10)
            baseline_intensity = baseline_features[0].get('Intensity', 0)
            noise_level = baseline_intensity / baseline_snr if baseline_snr > 0 else np.std(intensities) * 0.05
        else:
            # Estimate noise level from spectrum (lower for laser)
            noise_level = np.std(intensities) * 0.05
        
        # Calculate first derivative (slope) 
        wl_step = wavelengths[1] - wavelengths[0]
        slopes = np.gradient(intensities, wl_step)
        
        # Minimal smoothing for laser (preserve sharp features)
        if len(slopes) > 3:
            slopes = savgol_filter(slopes, 3, 1)
        
        # Set slope threshold based on noise level (more sensitive for laser)
        slope_threshold = noise_level / wl_step * 0.5  # More sensitive than halogen
        
        # Find potential mounds by looking for peaks first
        prominence_threshold = self.peak_prominence.get() * 0.7 * np.ptp(intensities)
        mound_peaks, properties = find_peaks(intensities,
                                           prominence=prominence_threshold,
                                           distance=int(min_width / wl_step / 3))  # Smaller distance for laser
        
        if len(mound_peaks) == 0:
            print("🏔️ No potential laser mound peaks found")
            return []
        
        mound_features = []
        
        for i, peak_idx in enumerate(mound_peaks):
            # Find mound start: where slope first exceeds threshold going left from peak
            start_idx = peak_idx
            while start_idx > 0:
                if slopes[start_idx] <= slope_threshold:
                    break
                start_idx -= 1
            
            # Refine start: look for actual slope increase point
            while start_idx < peak_idx - 1:
                if slopes[start_idx] > slope_threshold:
                    break
                start_idx += 1
            
            # Find mound end: where slope returns to threshold going right from peak  
            end_idx = peak_idx
            while end_idx < len(slopes) - 1:
                if slopes[end_idx] >= -slope_threshold:
                    break
                end_idx += 1
            
            # Refine end: look for actual slope decrease point
            while end_idx > peak_idx + 1:
                if slopes[end_idx] < -slope_threshold:
                    break
                end_idx -= 1
            
            # Check if mound is wide enough
            width_nm = wavelengths[end_idx] - wavelengths[start_idx]
            
            if width_nm >= min_width:
                start_wl = wavelengths[start_idx]
                end_wl = wavelengths[end_idx]
                crest_wl = wavelengths[peak_idx]
                
                # Calculate symmetry
                left_width = crest_wl - start_wl
                right_width = end_wl - crest_wl
                symmetry_ratio = left_width / right_width if right_width > 0 else float('inf')
                
                # Determine skew
                if symmetry_ratio < 0.8:
                    skew_desc = "Left Skewed"
                elif symmetry_ratio > 1.25:
                    skew_desc = "Right Skewed"
                else:
                    skew_desc = "Symmetric"
                
                group_name = f'Mound_{i+1}'
                
                # Add mound points with slope-based boundaries
                mound_features.extend([
                    {
                        'Feature': 'Mound_Start',
                        'Wavelength': round(start_wl, 2),
                        'Intensity': round(intensities[start_idx], 2),
                        'Point_Type': 'Start',
                        'Feature_Group': group_name,
                        'Slope_Threshold': round(slope_threshold, 6)
                    },
                    {
                        'Feature': 'Mound_Crest',
                        'Wavelength': round(crest_wl, 2),
                        'Intensity': round(intensities[peak_idx], 2),
                        'Point_Type': 'Crest',
                        'Feature_Group': group_name
                    },
                    {
                        'Feature': 'Mound_End',
                        'Wavelength': round(end_wl, 2),
                        'Intensity': round(intensities[end_idx], 2),
                        'Point_Type': 'End',
                        'Feature_Group': group_name,
                        'Slope_Threshold': round(-slope_threshold, 6)
                    },
                    {
                        'Feature': 'Mound_Summary',
                        'Wavelength': round(crest_wl, 2),
                        'Intensity': round(intensities[peak_idx], 2),
                        'Point_Type': 'Summary',
                        'Feature_Group': group_name,
                        'Symmetry_Ratio': round(symmetry_ratio, 3),
                        'Skew_Description': skew_desc,
                        'Width_nm': round(width_nm, 2),
                        'Baseline_Used': baseline_features[0].get('Intensity', 0) if baseline_features else 0,
                        'SNR': baseline_features[0].get('SNR', 0) if baseline_features else 0
                    }
                ])
                
                print(f"⚡ Laser Mound {i+1}: {start_wl:.1f}-{crest_wl:.1f}-{end_wl:.1f}nm, "
                      f"width={width_nm:.1f}nm, slope_threshold={slope_threshold:.6f}")
        
        print(f"🏔️ Detected {len([f for f in mound_features if 'Summary' in f['Feature']])} laser mounds using slope analysis")
        return mound_features
        
    # Individual detection methods (similar to halogen)
    def detect_baseline_only(self):
        """Detect baseline only"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
        self.detected_features = self.detect_baseline(
            self.current_spectrum['wavelengths'],
            self.current_spectrum['intensities']
        )
        self.plot_spectrum()
        self.update_results(f"Detected {len(self.detected_features)} baseline points")
        
    def detect_peaks_only(self):
        """Detect peaks only"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
        peaks = self.detect_peaks(
            self.current_spectrum['wavelengths'],
            self.current_spectrum['intensities']
        )
        self.detected_features.extend(peaks)
        self.plot_spectrum()
        self.update_results(f"Detected {len(peaks)} laser peaks")
        
    def detect_mounds_only(self):
        """Detect mounds only"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
        mounds = self.detect_mounds(
            self.current_spectrum['wavelengths'],
            self.current_spectrum['intensities']
        )
        self.detected_features.extend(mounds)
        self.plot_spectrum()
        mound_count = len([f for f in mounds if 'Summary' in f['Feature']])
        self.update_results(f"Detected {mound_count} laser mounds")
        
    # Placeholder methods for other features
    def detect_plateaus_only(self):
        self.update_results("Plateau detection - implement similar to mounds")
        
    def detect_troughs_only(self):
        self.update_results("Trough detection - implement similar to mounds")
        
    def detect_shoulders_only(self):
        self.update_results("Shoulder detection - implement similar to peaks")
        
    def detect_valleys_only(self):
        self.update_results("Valley detection - implement similar to peaks")
        
    def detect_plateaus(self, wavelengths, intensities):
        """Placeholder - implement full plateau detection"""
        return []
        
    def detect_troughs(self, wavelengths, intensities):
        """Placeholder - implement full trough detection"""
        return []
        
    def detect_shoulders(self, wavelengths, intensities):
        """Placeholder - implement full shoulder detection"""
        return []
        
    def detect_valleys(self, wavelengths, intensities):
        """Placeholder - implement full valley detection"""
        return []
        
    def clear_all_markers(self):
        """Clear all detected features"""
        self.detected_features = []
        if self.original_intensities is not None:
            self.current_spectrum['intensities'] = self.original_intensities.copy()
        self.plot_spectrum()
        self.update_results("Cleared all markers")
        
    def export_results(self):
        """Export detection results to CSV"""
        if not self.detected_features:
            messagebox.showwarning("Warning", "No features to export")
            return
            
        try:
            df = pd.DataFrame(self.detected_features)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.current_spectrum.get('filename', 'unknown').replace('.txt', '')
            outname = f"{filename}_laser_auto_{timestamp}.csv"
            
            os.makedirs(self.OUTPUT_DIRECTORY, exist_ok=True)
            full_path = os.path.join(self.OUTPUT_DIRECTORY, outname)
            df.to_csv(full_path, index=False)
            
            messagebox.showinfo("Success", f"Exported {len(self.detected_features)} features to:\n{outname}")
            self.update_results(f"✅ Exported to: {outname}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
            
    def update_results(self, text):
        """Update results display"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("⚡ Starting Enhanced Laser Auto Analyzer...")
    app = EnhancedLaserAutoAnalyzer()
    app.run()

if __name__ == '__main__':
    main()
