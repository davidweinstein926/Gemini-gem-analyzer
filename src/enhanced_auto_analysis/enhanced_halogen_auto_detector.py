#!/usr/bin/env python3
"""
Enhanced Halogen Auto Analyzer - Comprehensive Structural Detection
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

class EnhancedHalogenAutoAnalyzer:
    """Automated halogen structural analysis with comprehensive feature detection"""
    
    def __init__(self):
        self.OUTPUT_DIRECTORY = r"c:\users\david\gemini sp10 structural data\halogen"
        self.root = tk.Tk()
        self.root.title("🤖+🎯 Hybrid Halogen Analyzer - Auto-Detect + Manual Edit")
        self.root.geometry("1400x900")
        
        # Data storage
        self.current_spectrum = None
        self.detected_features = []
        self.removed_features = []  # Track manually removed features
        self.original_intensities = None
        self.click_tolerance = 5.0  # nm tolerance for clicking features
        
        # Detection parameters (tightened to match manual analysis quality)
        self.params = {
            "peak_prominence": 0.008,    # Increased from 0.002 - more selective
            "mound_min_width": 25,       # Increased from 15 - fewer small mounds
            "plateau_flatness": 0.008,   # Much stricter from 0.015 - quality plateaus only
            "trough_prominence": 0.008,  # Increased from 0.002 - significant troughs only
            "shoulder_prominence": 0.008, # Increased from 0.002 - clear shoulders only
            "valley_min_width": 15,      # Increased from 8 - fewer minor valleys
            "baseline_window": 25,       # 300-325nm specific region
            "smoothing_window": 9        # Keep noise reduction
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
        """Create adjustable parameters (tightened for quality over quantity)"""
        self.peak_prominence = tk.DoubleVar(value=0.008)  # More selective default
        self.mound_min_width = tk.IntVar(value=25)  # Larger mounds only
        self.plateau_flatness = tk.DoubleVar(value=0.008)  # Much stricter
        self.trough_prominence = tk.DoubleVar(value=0.008)  # Significant troughs only
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
        file_frame = ttk.LabelFrame(parent, text="📁 Halogen Spectrum File", padding=10)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load Halogen Spectrum", 
                  command=self.load_spectrum).pack(side='left', padx=5)
        
        ttk.Button(file_frame, text="Load Sample Data",
                  command=self.load_sample_data).pack(side='left', padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=10)
        
    def setup_parameters_section(self, parent):
        """Detection parameters section"""
        params_frame = ttk.LabelFrame(parent, text="🔧 Detection Parameters", padding=10)
        params_frame.pack(fill='x', padx=5, pady=5)
        
        # Peak prominence (more selective range)
        prom_frame = ttk.Frame(params_frame)
        prom_frame.pack(fill='x', pady=2)
        ttk.Label(prom_frame, text="Peak Prominence:").pack(side='left', padx=5)
        ttk.Scale(prom_frame, from_=0.002, to=0.05, variable=self.peak_prominence,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(prom_frame, textvariable=self.peak_prominence).pack(side='left', padx=5)
        
        # Mound width (larger minimum for quality)
        width_frame = ttk.Frame(params_frame)
        width_frame.pack(fill='x', pady=2)
        ttk.Label(width_frame, text="Min Mound Width:").pack(side='left', padx=5)
        ttk.Scale(width_frame, from_=15, to=50, variable=self.mound_min_width,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(width_frame, textvariable=self.mound_min_width).pack(side='left', padx=5)
        
        # Plateau flatness (much stricter range)
        flat_frame = ttk.Frame(params_frame)
        flat_frame.pack(fill='x', pady=2)
        ttk.Label(flat_frame, text="Plateau Flatness:").pack(side='left', padx=5)
        ttk.Scale(flat_frame, from_=0.002, to=0.03, variable=self.plateau_flatness,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(flat_frame, textvariable=self.plateau_flatness).pack(side='left', padx=5)
        
        # Smoothing (important for noisy halogen spectra)
        smooth_frame = ttk.Frame(params_frame)
        smooth_frame.pack(fill='x', pady=2)
        ttk.Label(smooth_frame, text="Smoothing Window:").pack(side='left', padx=5)
        ttk.Scale(smooth_frame, from_=3, to=15, variable=self.smoothing_window,
                 orient='horizontal', length=150).pack(side='left')
        ttk.Label(smooth_frame, textvariable=self.smoothing_window).pack(side='left', padx=5)
        
    def setup_detection_section(self, parent):
        """Detection control section"""
        detect_frame = ttk.LabelFrame(parent, text="🔬 Automated Halogen Detection", padding=10)
        detect_frame.pack(fill='x', padx=5, pady=5)
        
        # Detection and visualization
        ttk.Button(detect_frame, text="🤖 Detect ALL Structural Features",
                  command=self.detect_all_features).pack(pady=5, fill='x')
        
        ttk.Button(detect_frame, text="🔍 Debug Mound Detection",
                  command=self.debug_mound_detection).pack(pady=2, fill='x')
                  
        ttk.Button(detect_frame, text="🔍 Debug Plateau Detection", 
                  command=self.debug_plateau_detection).pack(pady=2, fill='x')
                  
        ttk.Button(detect_frame, text="📈 Show Slope Analysis",
                  command=self.show_slope_analysis).pack(pady=2, fill='x')
        
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
        
        # Hybrid editing controls
        edit_frame = ttk.LabelFrame(detect_frame, text="🎛️ Manual Editing", padding=5)
        edit_frame.pack(fill='x', pady=5)
        
        ttk.Label(edit_frame, text="Click on markers to remove/restore", 
                 font=('Arial', 9), foreground='blue').pack()
        
        ttk.Button(edit_frame, text="🔄 Undo Last Remove",
                  command=self.undo_last_remove).pack(pady=2, fill='x')
                  
        ttk.Button(edit_frame, text="✅ Restore All Features",
                  command=self.restore_all_features).pack(pady=2, fill='x')
        
        # Clear and export
        ttk.Button(detect_frame, text="🧹 Clear All Markers",
                  command=self.clear_all_markers).pack(pady=5, fill='x')
        
        ttk.Button(detect_frame, text="💾 Export Final Results",
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
        
        # Add instructions
        instruction_frame = ttk.Frame(parent)
        instruction_frame.pack(side='top', fill='x')
        ttk.Label(instruction_frame, text="💡 After auto-detection: Click on markers to remove unwanted features", 
                 font=('Arial', 9), foreground='darkblue').pack(pady=2)
        
        # Connect mouse events for hybrid editing
        self.canvas.mpl_connect('button_press_event', self.on_marker_click)
        
        # Initial empty plot
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity', fontsize=11)
        self.ax.set_title('Halogen Automated Structural Analysis', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
    def load_spectrum(self):
        """Load a halogen spectrum file"""
        default_dir = r"C:\Users\David\OneDrive\Desktop\gemini matcher\gemini sp10 raw\raw text"
        
        file_path = filedialog.askopenfilename(
            initialdir=default_dir,
            title="Select Halogen Spectrum",
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
        """Load sample halogen-like data for testing"""
        # Generate sample halogen spectrum
        wl = np.linspace(400, 800, 1000)
        intensity = np.random.normal(10, 1, len(wl))  # Baseline noise
        
        # Add typical halogen features
        # Broad mound at 500-600nm
        mound_mask = (wl >= 500) & (wl <= 600)
        intensity[mound_mask] += 20 * np.exp(-((wl[mound_mask] - 550) / 25)**2)
        
        # Peak at 650nm
        peak_mask = (wl >= 645) & (wl <= 655)
        intensity[peak_mask] += 30 * np.exp(-((wl[peak_mask] - 650) / 3)**2)
        
        # Plateau at 720-750nm
        plateau_mask = (wl >= 720) & (wl <= 750)
        intensity[plateau_mask] += 15
        
        # Trough at 460-480nm
        trough_mask = (wl >= 460) & (wl <= 480)
        intensity[trough_mask] -= 8 * np.exp(-((wl[trough_mask] - 470) / 8)**2)
        
        self.current_spectrum = {
            'wavelengths': wl,
            'intensities': intensity,
            'filename': 'Sample_Halogen_Spectrum'
        }
        self.original_intensities = intensity.copy()
        
        self.file_label.config(text='Sample Halogen Spectrum')
        self.plot_spectrum()
        self.update_results("Loaded sample halogen spectrum with typical features")
        
    def plot_spectrum(self):
        """Plot the current spectrum"""
        if not self.current_spectrum:
            return
            
        self.ax.clear()
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        
        # Plot spectrum (black line for halogen)
        self.ax.plot(wavelengths, intensities, 'k-', linewidth=1, label='Halogen Spectrum')
        
        # Plot detected features with same colors as manual analyzer
        self.plot_detected_features()
        
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity', fontsize=11)
        self.ax.set_title(f"Halogen Analysis - {self.current_spectrum['filename']}", fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
        
    def plot_detected_features(self):
        """Plot detected features with colored markers (active vs removed)"""
        if not self.detected_features:
            return
            
        # Group features by type for consistent coloring
        active_features = [f for f in self.detected_features if f not in self.removed_features]
        
        feature_groups = {}
        removed_groups = {}
        
        # Group active features
        for feature in active_features:
            feature_group = feature.get('Feature_Group', 'Unknown')
            feature_type = feature_group.split('_')[0]
            
            if feature_type not in feature_groups:
                feature_groups[feature_type] = []
            feature_groups[feature_type].append(feature)
        
        # Group removed features  
        for feature in self.removed_features:
            feature_group = feature.get('Feature_Group', 'Unknown')
            feature_type = feature_group.split('_')[0]
            
            if feature_type not in removed_groups:
                removed_groups[feature_type] = []
            removed_groups[feature_type].append(feature)
        
        # Plot active features with full color
        for feature_type, features in feature_groups.items():
            color = self.feature_colors.get(feature_type, 'black')
            wavelengths = [f['Wavelength'] for f in features]
            intensities = [f['Intensity'] for f in features]
            
            self.ax.scatter(wavelengths, intensities, c=color, s=50, 
                           label=f'{feature_type} ({len(features)})',
                           edgecolors='black', linewidth=1, zorder=5,
                           marker='o', alpha=0.8)
        
        # Plot removed features with faded appearance
        for feature_type, features in removed_groups.items():
            color = self.feature_colors.get(feature_type, 'black')
            wavelengths = [f['Wavelength'] for f in features]
            intensities = [f['Intensity'] for f in features]
            
            self.ax.scatter(wavelengths, intensities, c=color, s=30,
                           label=f'{feature_type} removed ({len(features)})',
                           edgecolors='red', linewidth=2, zorder=3,
                           marker='x', alpha=0.4)
    
    def on_marker_click(self, event):
        """Handle clicks on feature markers for removal/restoration"""
        if event.inaxes != self.ax or not self.detected_features:
            return
            
        # Check if toolbar is in zoom/pan mode
        if hasattr(self.canvas, 'toolbar') and self.canvas.toolbar:
            toolbar = self.canvas.toolbar
            if hasattr(toolbar, 'mode') and toolbar.mode in ['zoom rect', 'pan']:
                return  # Don't interfere with toolbar functions
        
        click_wl = event.xdata
        click_int = event.ydata
        
        if click_wl is None or click_int is None:
            return
        
        # Find nearest feature
        nearest_feature = self.find_nearest_feature(click_wl, click_int)
        
        if nearest_feature:
            if nearest_feature in self.removed_features:
                # Restore removed feature
                self.removed_features.remove(nearest_feature)
                feature_name = nearest_feature.get('Feature', 'Unknown')
                print(f"✅ Restored: {feature_name} at {nearest_feature['Wavelength']:.1f}nm")
                self.update_status_after_edit()
            else:
                # Remove active feature
                self.removed_features.append(nearest_feature)
                feature_name = nearest_feature.get('Feature', 'Unknown')
                print(f"❌ Removed: {feature_name} at {nearest_feature['Wavelength']:.1f}nm")
                self.update_status_after_edit()
            
            # Update plot
            self.plot_spectrum()
    
    def find_nearest_feature(self, click_wl, click_int):
        """Find the nearest feature to the click location"""
        min_distance = float('inf')
        nearest_feature = None
        
        for feature in self.detected_features:
            feat_wl = feature['Wavelength']
            feat_int = feature['Intensity']
            
            # Calculate distance (weighted more heavily by wavelength)
            wl_diff = abs(feat_wl - click_wl)
            int_diff = abs(feat_int - click_int)
            
            # Normalize intensity difference by spectrum range
            if hasattr(self, 'current_spectrum') and self.current_spectrum:
                int_range = np.ptp(self.current_spectrum['intensities'])
                int_diff_normalized = int_diff / int_range if int_range > 0 else 0
            else:
                int_diff_normalized = int_diff / 100  # Fallback normalization
            
            # Combined distance (wavelength has 3x weight)
            distance = wl_diff + (int_diff_normalized * 20)
            
            if distance < min_distance and wl_diff <= self.click_tolerance:
                min_distance = distance
                nearest_feature = feature
        
        return nearest_feature
    
    def update_status_after_edit(self):
        """Update results display after manual editing"""
        active_count = len([f for f in self.detected_features if f not in self.removed_features])
        removed_count = len(self.removed_features)
        
        result_text = f"🎛️ HYBRID EDITING STATUS\n"
        result_text += "="*30 + "\n\n"
        result_text += f"📊 Active Features: {active_count}\n"
        result_text += f"❌ Removed Features: {removed_count}\n"
        result_text += f"📋 Total Detected: {len(self.detected_features)}\n\n"
        
        # Group active features by type
        active_features = [f for f in self.detected_features if f not in self.removed_features]
        feature_counts = {}
        
        for feature in active_features:
            feature_group = feature.get('Feature_Group', 'Unknown')
            feature_type = feature_group.split('_')[0]
            feature_counts[feature_type] = feature_counts.get(feature_type, 0) + 1
        
        result_text += "📋 ACTIVE FEATURES BY TYPE:\n"
        for feat_type, count in feature_counts.items():
            if count > 0:
                icon = {'Baseline': '📍', 'Peak': '🔺', 'Mound': '🏔️', 'Plateau': '⬜',
                       'Trough': '🕳️', 'Shoulder': '📐', 'Valley': '🌊'}.get(feat_type, '•')
                result_text += f"   {icon} {feat_type}: {count}\n"
        
        result_text += f"\n💡 Click markers to remove/restore\n"
        result_text += f"💾 Export saves only active features"
        
        self.update_results(result_text)
    
    def undo_last_remove(self):
        """Undo the last feature removal"""
        if self.removed_features:
            restored_feature = self.removed_features.pop()
            feature_name = restored_feature.get('Feature', 'Unknown')
            print(f"🔄 Undid removal of: {feature_name} at {restored_feature['Wavelength']:.1f}nm")
            self.plot_spectrum()
            self.update_status_after_edit()
        else:
            print("❌ No removals to undo")
            self.update_results("❌ No removals to undo")
    
    def restore_all_features(self):
        """Restore all removed features"""
        if self.removed_features:
            restored_count = len(self.removed_features)
            self.removed_features.clear()
            print(f"✅ Restored all {restored_count} removed features")
            self.plot_spectrum()
            self.update_status_after_edit()
        else:
            print("❌ No removed features to restore")
            self.update_results("❌ No removed features to restore")
    
    def detect_baseline(self, wavelengths, intensities):
        """Detect baseline in 300-325 nm range (standard for halogen)"""
        
        # Define baseline region for halogen spectroscopy
        baseline_start = 300.0
        baseline_end = 325.0
        
        # Find indices for baseline region
        baseline_mask = (wavelengths >= baseline_start) & (wavelengths <= baseline_end)
        
        if not np.any(baseline_mask):
            print(f"⚠️ No data in baseline region {baseline_start}-{baseline_end}nm")
            # Fall back to first 50 points if no 300-325nm data
            baseline_indices = np.arange(min(50, len(wavelengths)))
            baseline_start = wavelengths[0]
            baseline_end = wavelengths[min(49, len(wavelengths)-1)]
        else:
            baseline_indices = np.where(baseline_mask)[0]
            baseline_start = wavelengths[baseline_indices[0]]
            baseline_end = wavelengths[baseline_indices[-1]]
        
        # Calculate baseline statistics
        baseline_intensities = intensities[baseline_indices]
        avg_intensity = np.mean(baseline_intensities)
        std_dev = np.std(baseline_intensities)
        snr = avg_intensity / std_dev if std_dev > 0 else float('inf')
        
        print(f"📍 Halogen Baseline: {baseline_start:.1f}-{baseline_end:.1f}nm, "
              f"Avg={avg_intensity:.2f}, SNR={snr:.1f}")
        
        return [
            {
                'Feature': 'Baseline_Start',
                'Wavelength': round(baseline_start, 2),
                'Intensity': round(avg_intensity, 2),
                'Point_Type': 'Start',
                'Feature_Group': 'Baseline',
                'SNR': round(snr, 1),
                'Baseline_Used': round(avg_intensity, 2),
                'Processing': 'Auto_300-325nm_Baseline'
            },
            {
                'Feature': 'Baseline_End',
                'Wavelength': round(baseline_end, 2),
                'Intensity': round(avg_intensity, 2),
                'Point_Type': 'End',
                'Feature_Group': 'Baseline',
                'SNR': round(snr, 1),
                'Baseline_Used': round(avg_intensity, 2),
                'Processing': 'Auto_300-325nm_Baseline'
            }
        ]
    
    def detect_mounds(self, wavelengths, intensities):
        """Detect broad mound features using comprehensive analysis"""
        min_width = self.mound_min_width.get()
        
        # Get baseline info
        baseline_features = [f for f in self.detected_features if f.get('Feature_Group') == 'Baseline']
        if baseline_features:
            baseline_intensity = baseline_features[0].get('Intensity', 0)
            baseline_snr = baseline_features[0].get('SNR', 10)
            noise_level = baseline_intensity / baseline_snr if baseline_snr > 0 else np.std(intensities) * 0.1
        else:
            baseline_intensity = np.min(intensities)
            noise_level = np.std(intensities) * 0.1
        
        print(f"🏔️ Mound detection: baseline={baseline_intensity:.2f}, noise={noise_level:.4f}")
        
        # Step 1: Find all significant peaks (potential mound crests)
        # Use very low prominence to catch broad mounds
        prominence_threshold = max(noise_level * 3, np.ptp(intensities) * self.peak_prominence.get())
        
        peaks, properties = find_peaks(intensities,
                                     prominence=prominence_threshold,
                                     distance=int(min_width / (wavelengths[1] - wavelengths[0]) / 2))
        
        if len(peaks) == 0:
            print("🏔️ No potential mound peaks found")
            return []
        
        print(f"🔍 Found {len(peaks)} potential mound crests")
        
        mound_features = []
        
        # Step 2: For each peak, determine if it's part of a mound
        for i, peak_idx in enumerate(peaks):
            peak_wl = wavelengths[peak_idx]
            peak_intensity = intensities[peak_idx]
            
            print(f"  Analyzing peak {i+1} at {peak_wl:.1f}nm, intensity={peak_intensity:.2f}")
            
            # Step 3: Find mound start (going left from peak)
            start_idx = peak_idx
            start_intensity = peak_intensity
            
            # Look for where intensity first rises significantly above baseline
            for j in range(peak_idx, -1, -1):
                current_intensity = intensities[j]
                
                # Check if we've gone too far (back to baseline level)
                if current_intensity <= baseline_intensity + noise_level * 2:
                    # Found baseline level, now look for start of rise
                    for k in range(j, peak_idx):
                        if intensities[k] > baseline_intensity + noise_level * 3:
                            start_idx = k
                            start_intensity = intensities[k]
                            break
                    break
                    
                start_idx = j
                start_intensity = current_intensity
            
            # Step 4: Find mound end (going right from peak)
            end_idx = peak_idx
            end_intensity = peak_intensity
            
            # Look for where intensity returns to near baseline
            for j in range(peak_idx, len(intensities)):
                current_intensity = intensities[j]
                
                # Check if we've returned to baseline level
                if current_intensity <= baseline_intensity + noise_level * 2:
                    # Found baseline level, but use the point just before
                    if j > peak_idx:
                        end_idx = j - 1
                        end_intensity = intensities[j - 1]
                    break
                    
                end_idx = j
                end_intensity = current_intensity
            
            # Step 5: Validate mound dimensions
            start_wl = wavelengths[start_idx]
            end_wl = wavelengths[end_idx]
            width_nm = end_wl - start_wl
            
            # Check if wide enough and significant enough (stricter validation)
            height_above_baseline = peak_intensity - baseline_intensity
            
            print(f"    Mound candidate: {start_wl:.1f}-{peak_wl:.1f}-{end_wl:.1f}nm")
            print(f"    Width: {width_nm:.1f}nm, Height: {height_above_baseline:.2f}")
            
            # Much stricter validation criteria
            min_height = noise_level * 8  # Increased from 5 - must be clearly significant
            min_prominence = np.ptp(intensities) * self.peak_prominence.get()
            
            is_wide_enough = width_nm >= min_width
            is_tall_enough = height_above_baseline > min_height  
            is_prominent_enough = peak_intensity > baseline_intensity + min_prominence
            
            print(f"    Validation: wide={is_wide_enough}, tall={is_tall_enough}, prominent={is_prominent_enough}")
            
            if is_wide_enough and is_tall_enough and is_prominent_enough:
                # Step 6: Calculate mound characteristics
                left_width = peak_wl - start_wl
                right_width = end_wl - peak_wl
                symmetry_ratio = left_width / right_width if right_width > 0 else float('inf')
                
                # Determine skew
                if symmetry_ratio < 0.8:
                    skew_desc = "Left Skewed"
                elif symmetry_ratio > 1.25:
                    skew_desc = "Right Skewed"
                else:
                    skew_desc = "Symmetric"
                
                group_name = f'Mound_{i+1}'
                
                # Step 7: Create mound feature entries
                mound_features.extend([
                    {
                        'Feature': 'Mound_Start',
                        'Wavelength': round(start_wl, 2),
                        'Intensity': round(start_intensity, 2),
                        'Point_Type': 'Start',
                        'Feature_Group': group_name,
                        'Processing': 'Auto_Baseline_Normalized',
                        'Baseline_Used': round(baseline_intensity, 2),
                        'SNR': round(baseline_snr, 1) if baseline_features else None
                    },
                    {
                        'Feature': 'Mound_Crest',
                        'Wavelength': round(peak_wl, 2),
                        'Intensity': round(peak_intensity, 2),
                        'Point_Type': 'Crest',
                        'Feature_Group': group_name,
                        'Processing': 'Auto_Baseline_Normalized',
                        'Baseline_Used': round(baseline_intensity, 2),
                        'SNR': round(baseline_snr, 1) if baseline_features else None
                    },
                    {
                        'Feature': 'Mound_End',
                        'Wavelength': round(end_wl, 2),
                        'Intensity': round(end_intensity, 2),
                        'Point_Type': 'End',
                        'Feature_Group': group_name,
                        'Processing': 'Auto_Baseline_Normalized',
                        'Baseline_Used': round(baseline_intensity, 2),
                        'SNR': round(baseline_snr, 1) if baseline_features else None
                    },
                    {
                        'Feature': 'Mound_Summary',
                        'Wavelength': round(peak_wl, 2),
                        'Intensity': round(peak_intensity, 2),
                        'Point_Type': 'Summary',
                        'Feature_Group': group_name,
                        'Processing': 'Auto_Baseline_Normalized',
                        'Symmetry_Ratio': round(symmetry_ratio, 3),
                        'Skew_Description': skew_desc,
                        'Width_nm': round(width_nm, 2),
                        'Baseline_Used': round(baseline_intensity, 2),
                        'Norm_Factor': 1.0,  # After normalization
                        'SNR': round(baseline_snr, 1) if baseline_features else None
                    }
                ])
                
                print(f"    ✅ VALID MOUND: {start_wl:.1f}-{peak_wl:.1f}-{end_wl:.1f}nm")
                print(f"       Width={width_nm:.1f}nm, {skew_desc}")
            else:
                print(f"    ❌ Rejected: width={width_nm:.1f}nm (min={min_width}), height={height_above_baseline:.2f}")
        
        mound_count = len([f for f in mound_features if 'Summary' in f['Feature']])
        print(f"🏔️ Final result: {mound_count} valid mounds detected")
        
        return mound_features
    
    def detect_all_features(self):
        """Detect all structural features automatically"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
            
        print("🤖 HALOGEN AUTOMATED STRUCTURAL ANALYSIS")
        print("="*50)
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        filename = self.current_spectrum['filename']
        
        # Clear previous results and removals
        self.detected_features = []
        self.removed_features = []
        
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
            
        # Step 3: Detect all feature types
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
                feature['Light_Source'] = 'Halogen'
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
        
        result_text = "🤖 SELECTIVE DETECTION COMPLETE\n"
        result_text += "="*40 + "\n\n"
        result_text += f"📁 File: {filename}\n"
        result_text += f"📊 Total Features: {len(self.detected_features)}\n"
        result_text += f"🎯 Quality over Quantity: Strict thresholds applied\n\n"
        
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
        
        result_text += f"\n🔧 PROCESSING APPLIED:\n"
        result_text += f"   ✅ Smoothing (window={self.smoothing_window.get()})\n"
        if baseline_features:
            result_text += f"   ✅ Baseline correction\n"
            result_text += f"   ✅ Normalization to max=100\n"
        
        result_text += f"\n🎛️ HYBRID EDITING:\n"
        result_text += f"   💡 Click on markers to remove unwanted features\n"
        result_text += f"   🔄 Use 'Undo Last Remove' to restore\n"
        result_text += f"   💾 Export saves only your final selection\n"
        
        self.update_results(result_text)
    
    def debug_plateau_detection(self):
        """Debug plateau detection process step by step"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
            
        # First ensure we have baseline
        if not any(f.get('Feature_Group') == 'Baseline' for f in self.detected_features):
            print("🔧 No baseline found, detecting baseline first...")
            baseline_features = self.detect_baseline(
                self.current_spectrum['wavelengths'],
                self.current_spectrum['intensities']
            )
            self.detected_features.extend(baseline_features)
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        
        # Get baseline info
        baseline_features = [f for f in self.detected_features if f.get('Feature_Group') == 'Baseline']
        baseline_intensity = baseline_features[0].get('Intensity', 0)
        baseline_snr = baseline_features[0].get('SNR', 10)
        noise_level = baseline_intensity / baseline_snr if baseline_snr > 0 else np.std(intensities) * 0.1
        
        print("\n" + "="*60)
        print("⬜ PLATEAU DETECTION DEBUG")
        print("="*60)
        print(f"📊 Spectrum: {len(wavelengths)} points, {wavelengths[0]:.1f}-{wavelengths[-1]:.1f}nm")
        print(f"📍 Baseline: {baseline_intensity:.2f} (SNR: {baseline_snr:.1f})")
        print(f"📈 Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")
        
        # Calculate flatness threshold
        flatness_threshold = self.plateau_flatness.get() * np.ptp(intensities)
        elevation_threshold = baseline_intensity + noise_level * 2
        
        print(f"🎯 Flatness threshold: {flatness_threshold:.4f}")
        print(f"🎯 Elevation threshold: {elevation_threshold:.4f}")
        
        # Analyze potential plateau regions
        print(f"\n🔍 SCANNING FOR FLAT REGIONS:")
        
        wl_step = wavelengths[1] - wavelengths[0]
        window_size = max(5, int(10 / wl_step))  # 10nm window
        
        flat_regions = []
        
        for i in range(0, len(intensities) - window_size, 5):
            window_intensities = intensities[i:i+window_size]
            intensity_range = np.ptp(window_intensities)
            mean_intensity = np.mean(window_intensities)
            
            is_flat = intensity_range <= flatness_threshold
            is_elevated = mean_intensity > elevation_threshold
            
            if is_flat or is_elevated:  # Show both flat AND elevated regions
                start_wl = wavelengths[i]
                end_wl = wavelengths[i+window_size-1]
                
                status = "✅ PLATEAU" if (is_flat and is_elevated) else "❓ PARTIAL"
                if is_flat and not is_elevated:
                    status += " (flat but low)"
                elif not is_flat and is_elevated:
                    status += " (elevated but not flat)"
                
                print(f"  {start_wl:.1f}-{end_wl:.1f}nm: range={intensity_range:.4f}, "
                      f"mean={mean_intensity:.2f} {status}")
                
                if is_flat and is_elevated:
                    flat_regions.append((i, i+window_size, start_wl, end_wl, mean_intensity))
        
        print(f"\n📊 Found {len(flat_regions)} potential plateau regions")
        
        # Now run the actual detection
        print(f"\n⬜ RUNNING PLATEAU DETECTION:")
        self.detected_features = [f for f in self.detected_features if f.get('Feature_Group', '').split('_')[0] != 'Plateau']  # Clear old plateaus
        plateaus = self.detect_plateaus(wavelengths, intensities)
        self.detected_features.extend(plateaus)
        
        # Update plot
        self.plot_spectrum()
        
        # Show results
        if plateaus:
            plateau_count = len([f for f in plateaus if f['Point_Type'] == 'Midpoint'])
            result_text = f"⬜ DEBUG COMPLETE - Found {plateau_count} plateaus\n\n"
            for f in plateaus:
                if f['Point_Type'] in ['Start', 'Midpoint', 'End']:
                    result_text += f"{f['Feature']}: {f['Wavelength']}nm (I={f['Intensity']:.2f})\n"
        else:
            result_text = "⬜ DEBUG COMPLETE - No plateaus detected\n\n"
            result_text += "Possible reasons:\n"
            result_text += f"• No regions flat enough (threshold={flatness_threshold:.4f})\n"
            result_text += f"• No regions elevated enough (threshold={elevation_threshold:.4f})\n"
            result_text += "• Try adjusting plateau flatness parameter\n"
            result_text += "• Check if baseline is correct"
        
        self.update_results(result_text)
        
    def debug_mound_detection(self):
        """Debug mound detection process step by step"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
            
        # First ensure we have baseline
        if not any(f.get('Feature_Group') == 'Baseline' for f in self.detected_features):
            print("🔧 No baseline found, detecting baseline first...")
            baseline_features = self.detect_baseline(
                self.current_spectrum['wavelengths'],
                self.current_spectrum['intensities']
            )
            self.detected_features.extend(baseline_features)
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        
        # Get baseline info
        baseline_features = [f for f in self.detected_features if f.get('Feature_Group') == 'Baseline']
        baseline_intensity = baseline_features[0].get('Intensity', 0)
        baseline_snr = baseline_features[0].get('SNR', 10)
        
        print("\n" + "="*60)
        print("🔍 MOUND DETECTION DEBUG")
        print("="*60)
        print(f"📊 Spectrum: {len(wavelengths)} points, {wavelengths[0]:.1f}-{wavelengths[-1]:.1f}nm")
        print(f"📍 Baseline: {baseline_intensity:.2f} (SNR: {baseline_snr:.1f})")
        print(f"📈 Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")
        print(f"🎯 Looking for features > {baseline_intensity + (baseline_intensity/baseline_snr)*3:.2f}")
        
        # Show spectrum statistics
        above_baseline = intensities > baseline_intensity + (baseline_intensity/baseline_snr)*3
        significant_points = np.sum(above_baseline)
        print(f"📊 Points above threshold: {significant_points}/{len(intensities)} ({significant_points/len(intensities)*100:.1f}%)")
        
        if significant_points > 0:
            significant_wl_range = f"{wavelengths[above_baseline].min():.1f}-{wavelengths[above_baseline].max():.1f}nm"
            print(f"🌊 Significant activity in: {significant_wl_range}")
        
        # Test different prominence thresholds
        print(f"\n🧪 TESTING PEAK DETECTION:")
        noise_level = baseline_intensity / baseline_snr
        
        test_prominences = [
            noise_level * 2,
            noise_level * 3, 
            noise_level * 5,
            np.ptp(intensities) * 0.01,
            np.ptp(intensities) * 0.02,
            np.ptp(intensities) * 0.05
        ]
        
        for i, prom in enumerate(test_prominences):
            peaks, _ = find_peaks(intensities, prominence=prom, distance=10)
            print(f"  Prominence {prom:.4f}: {len(peaks)} peaks")
            if len(peaks) > 0 and len(peaks) < 20:
                peak_wls = wavelengths[peaks]
                print(f"    Peaks at: {', '.join([f'{wl:.1f}' for wl in peak_wls[:10]])}nm")
        
        # Now run the actual detection
        print(f"\n🏔️ RUNNING MOUND DETECTION:")
        self.detected_features = [f for f in self.detected_features if f.get('Feature_Group') != 'Mound']  # Clear old mounds
        mounds = self.detect_mounds(wavelengths, intensities)
        self.detected_features.extend(mounds)
        
        # Update plot
        self.plot_spectrum()
        
        # Show results
        if mounds:
            mound_count = len([f for f in mounds if 'Summary' in f['Feature']])
            result_text = f"🔍 DEBUG COMPLETE - Found {mound_count} mounds\n\n"
            for f in mounds:
                if f['Point_Type'] in ['Start', 'Crest', 'End']:
                    result_text += f"{f['Feature']}: {f['Wavelength']}nm (I={f['Intensity']:.2f})\n"
        else:
            result_text = "🔍 DEBUG COMPLETE - No mounds detected\n\n"
            result_text += "Try adjusting parameters:\n"
            result_text += "• Lower peak prominence\n" 
            result_text += "• Reduce minimum mound width\n"
            result_text += "• Check if spectrum has baseline issues"
        
        self.update_results(result_text)
    
    def preprocess_spectrum(self, intensities):
        """Apply smoothing preprocessing"""
        window = self.smoothing_window.get()
        if len(intensities) > window and window >= 3 and window % 2 == 1:
            smoothed = savgol_filter(intensities, window, 2)
            print(f"🔧 Applied smoothing (window={window})")
            return smoothed
        else:
            return intensities.copy()
            
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
        """Detect sharp peaks"""
        prominence_threshold = self.peak_prominence.get() * np.ptp(intensities)
        
        peaks, properties = find_peaks(intensities,
                                     prominence=prominence_threshold,
                                     distance=5)
        
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
        
        print(f"🔺 Detected {len(peaks)} peaks")
        return peak_features
    
    # Individual detection methods for interactive use
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
        self.update_results(f"Detected {len(peaks)} peaks")
        
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
        self.update_results(f"Detected {mound_count} mounds")
    
    # Placeholder methods for other features (implement similar to mounds)
    def detect_plateaus_only(self):
        """Detect plateaus only"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
            
        # Ensure baseline exists first
        if not any(f.get('Feature_Group') == 'Baseline' for f in self.detected_features):
            print("🔧 No baseline found, detecting baseline first...")
            baseline_features = self.detect_baseline(
                self.current_spectrum['wavelengths'],
                self.current_spectrum['intensities']
            )
            self.detected_features.extend(baseline_features)
            
        plateaus = self.detect_plateaus(
            self.current_spectrum['wavelengths'],
            self.current_spectrum['intensities']
        )
        self.detected_features.extend(plateaus)
        self.plot_spectrum()
        plateau_count = len([f for f in plateaus if f['Point_Type'] == 'Midpoint'])
        self.update_results(f"Detected {plateau_count} plateaus")
        
    def detect_troughs_only(self):
        self.update_results("Trough detection - implement similar to mounds")
        
    def detect_shoulders_only(self):
        self.update_results("Shoulder detection - implement similar to peaks")
        
    def detect_valleys_only(self):
        self.update_results("Valley detection - implement similar to peaks")
        
    def detect_plateaus(self, wavelengths, intensities):
        """Detect flat plateau regions based on manual analysis patterns"""
        
        # Get baseline info for thresholding
        baseline_features = [f for f in self.detected_features if f.get('Feature_Group') == 'Baseline']
        if baseline_features:
            baseline_intensity = baseline_features[0].get('Intensity', 0)
            baseline_snr = baseline_features[0].get('SNR', 10)
            noise_level = baseline_intensity / baseline_snr if baseline_snr > 0 else np.std(intensities) * 0.1
        else:
            baseline_intensity = np.min(intensities)
            noise_level = np.std(intensities) * 0.1
        
        plateau_features = []
        flatness_threshold = self.plateau_flatness.get() * np.ptp(intensities)
        min_width = 20  # Increased minimum plateau width
        
        print(f"⬜ Plateau detection: flatness_threshold={flatness_threshold:.4f}, min_width={min_width}nm")
        
        # Convert min_width to points
        wl_step = wavelengths[1] - wavelengths[0]
        min_points = max(8, int(min_width / wl_step))  # Increased minimum points
        
        # Use sliding window to find flat regions
        window_size = max(8, min_points)
        
        i = 0
        plateau_count = 0
        
        while i < len(intensities) - window_size:
            # Check if region is flat enough
            window_intensities = intensities[i:i+window_size]
            intensity_range = np.ptp(window_intensities)
            mean_intensity = np.mean(window_intensities)
            
            # Much stricter criteria for plateaus
            is_flat = intensity_range <= flatness_threshold
            is_significantly_elevated = mean_intensity > baseline_intensity + noise_level * 4  # Increased from 2
            is_wide_enough_initially = True  # Will check properly after extension
            
            if is_flat and is_significantly_elevated:
                print(f"  Found potential plateau at {wavelengths[i]:.1f}nm")
                
                # Found potential plateau, extend it
                start_idx = i
                end_idx = i + window_size
                
                # Extend backward while maintaining flatness
                while start_idx > 0:
                    extended_window = intensities[start_idx-1:end_idx]
                    extended_range = np.ptp(extended_window)
                    extended_mean = np.mean(extended_window)
                    
                    if (extended_range <= flatness_threshold and 
                        extended_mean > baseline_intensity + noise_level * 2):
                        start_idx -= 1
                    else:
                        break
                
                # Extend forward while maintaining flatness
                while end_idx < len(intensities):
                    extended_window = intensities[start_idx:end_idx+1]
                    extended_range = np.ptp(extended_window)
                    extended_mean = np.mean(extended_window)
                    
                    if (extended_range <= flatness_threshold and 
                        extended_mean > baseline_intensity + noise_level * 2):
                        end_idx += 1
                    else:
                        break
                
                # Check if wide enough (stricter validation)
                width_nm = wavelengths[end_idx-1] - wavelengths[start_idx]
                
                # Additional validation for quality plateaus
                plateau_intensities = intensities[start_idx:end_idx]
                avg_intensity = np.mean(plateau_intensities)
                height_above_baseline = avg_intensity - baseline_intensity
                
                # Stricter criteria
                is_wide_enough = width_nm >= min_width
                is_significantly_elevated = height_above_baseline > noise_level * 4
                is_consistently_flat = np.std(plateau_intensities) < flatness_threshold * 0.5
                
                if is_wide_enough and is_significantly_elevated and is_consistently_flat:
                    plateau_count += 1
                    group_name = f'Plateau_{plateau_count}'
                    
                    start_wl = wavelengths[start_idx]
                    mid_idx = start_idx + (end_idx - start_idx) // 2
                    mid_wl = wavelengths[mid_idx]
                    end_wl = wavelengths[end_idx-1]
                    
                    # Use consistent intensity (average of plateau)
                    plateau_intensities = intensities[start_idx:end_idx]
                    avg_intensity = np.mean(plateau_intensities)
                    
                    print(f"    ✅ VALID PLATEAU: {start_wl:.1f}-{mid_wl:.1f}-{end_wl:.1f}nm")
                    print(f"       Width={width_nm:.1f}nm, Avg_intensity={avg_intensity:.2f}")
                    
                    # Add plateau points (matching manual structure)
                    plateau_features.extend([
                        {
                            'Feature': 'Plateau_Start',
                            'Wavelength': round(start_wl, 2),
                            'Intensity': round(intensities[start_idx], 2),
                            'Point_Type': 'Start',
                            'Feature_Group': group_name,
                            'Processing': 'Auto_Baseline_Normalized',
                            'Width_nm': round(width_nm, 2),
                            'Baseline_Used': round(baseline_intensity, 2),
                            'SNR': round(baseline_snr, 1) if baseline_features else None
                        },
                        {
                            'Feature': 'Plateau_Midpoint',
                            'Wavelength': round(mid_wl, 2),
                            'Intensity': round(avg_intensity, 2),
                            'Point_Type': 'Midpoint',
                            'Feature_Group': group_name,
                            'Processing': 'Auto_Baseline_Normalized',
                            'Width_nm': round(width_nm, 2),
                            'Baseline_Used': round(baseline_intensity, 2),
                            'SNR': round(baseline_snr, 1) if baseline_features else None
                        },
                        {
                            'Feature': 'Plateau_End',
                            'Wavelength': round(end_wl, 2),
                            'Intensity': round(intensities[end_idx-1], 2),
                            'Point_Type': 'End',
                            'Feature_Group': group_name,
                            'Processing': 'Auto_Baseline_Normalized',
                            'Width_nm': round(width_nm, 2),
                            'Baseline_Used': round(baseline_intensity, 2),
                            'SNR': round(baseline_snr, 1) if baseline_features else None
                        }
                    ])
                
                # Skip past this plateau
                i = end_idx
            else:
                i += 1
        
        print(f"⬜ Detected {plateau_count} plateaus")
        return plateau_features
        
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
        """Clear all detected features and removals"""
        self.detected_features = []
        self.removed_features = []
        if self.original_intensities is not None:
            self.current_spectrum['intensities'] = self.original_intensities.copy()
        self.plot_spectrum()
        self.update_results("Cleared all markers and edits")
        
    def export_results(self):
        """Export ONLY the active (non-removed) features to CSV"""
        # Get only active features
        active_features = [f for f in self.detected_features if f not in self.removed_features]
        
        if not active_features:
            messagebox.showwarning("Warning", "No active features to export")
            return
            
        try:
            df = pd.DataFrame(active_features)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.current_spectrum.get('filename', 'unknown').replace('.txt', '')
            outname = f"{filename}_halogen_hybrid_{timestamp}.csv"
            
            os.makedirs(self.OUTPUT_DIRECTORY, exist_ok=True)
            full_path = os.path.join(self.OUTPUT_DIRECTORY, outname)
            df.to_csv(full_path, index=False)
            
            removed_count = len(self.removed_features)
            messagebox.showinfo("Success", 
                f"Exported {len(active_features)} ACTIVE features to:\n{outname}\n\n"
                f"(Excluded {removed_count} manually removed features)")
            
            result_text = f"✅ HYBRID EXPORT COMPLETE\n\n"
            result_text += f"📁 File: {outname}\n"
            result_text += f"💾 Active features exported: {len(active_features)}\n"
            result_text += f"❌ Removed features excluded: {removed_count}\n"
            result_text += f"🎯 Final result represents your expert selection"
            
            self.update_results(result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Export failed: {e}")
    
    def show_slope_analysis(self):
        """Show slope analysis for debugging mound detection"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
            
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        
        # Calculate slopes
        wl_step = wavelengths[1] - wavelengths[0]
        slopes = np.gradient(intensities, wl_step)
        
        # Get baseline info for threshold
        baseline_features = [f for f in self.detected_features if f.get('Feature_Group') == 'Baseline']
        if baseline_features:
            baseline_snr = baseline_features[0].get('SNR', 10)
            baseline_intensity = baseline_features[0].get('Intensity', 0)
            noise_level = baseline_intensity / baseline_snr if baseline_snr > 0 else np.std(intensities) * 0.1
        else:
            noise_level = np.std(intensities) * 0.1
            
        slope_threshold = noise_level / wl_step
        
        # Create slope analysis window
        slope_window = tk.Toplevel(self.root)
        slope_window.title("Slope Analysis - Mound Detection Debug")
        slope_window.geometry("1000x600")
        
        # Create matplotlib figure for slope analysis
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
        fig = Figure(figsize=(12, 8), dpi=100)
        
        # Top subplot: Original spectrum
        ax1 = fig.add_subplot(211)
        ax1.plot(wavelengths, intensities, 'k-', linewidth=1, label='Spectrum')
        
        # Plot detected mounds if any
        mound_features = [f for f in self.detected_features if 'Mound' in f.get('Feature_Group', '')]
        if mound_features:
            mound_wls = [f['Wavelength'] for f in mound_features]
            mound_ints = [f['Intensity'] for f in mound_features]
            ax1.scatter(mound_wls, mound_ints, c='red', s=50, zorder=5, label='Detected Mounds')
        
        ax1.set_ylabel('Intensity')
        ax1.set_title('Spectrum with Detected Mounds')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Bottom subplot: Slope analysis
        ax2 = fig.add_subplot(212)
        ax2.plot(wavelengths, slopes, 'b-', linewidth=1, label='Slope (dI/dλ)')
        ax2.axhline(y=slope_threshold, color='red', linestyle='--', 
                   label=f'Pos Threshold = {slope_threshold:.6f}')
        ax2.axhline(y=-slope_threshold, color='red', linestyle='--', 
                   label=f'Neg Threshold = {-slope_threshold:.6f}')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Slope (dI/dλ)')
        ax2.set_title(f'Slope Analysis (Noise Level = {noise_level:.4f}, Threshold = ±{slope_threshold:.6f})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add canvas to window
        canvas = FigureCanvasTkAgg(fig, slope_window)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add info text
        info_frame = tk.Frame(slope_window)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        info_text = (f"Slope Analysis Debug Info:\n"
                    f"• Baseline SNR: {baseline_features[0].get('SNR', 'N/A') if baseline_features else 'No baseline'}\n"
                    f"• Noise Level: {noise_level:.6f}\n" 
                    f"• Slope Threshold: ±{slope_threshold:.6f}\n"
                    f"• Wavelength Step: {wl_step:.3f} nm\n"
                    f"• Mound starts when slope > +threshold\n"
                    f"• Mound ends when slope < -threshold")
        
        tk.Label(info_frame, text=info_text, font=('Courier', 9), 
                justify='left', bg='lightyellow').pack(anchor='w')
        
        canvas.draw()
            
    def update_results(self, text):
        """Update results display"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("🤖+🎯 Starting Hybrid Halogen Auto Analyzer (Auto-Detect + Manual Edit)...")
    app = EnhancedHalogenAutoAnalyzer()
    app.run()

if __name__ == '__main__':
    main()
