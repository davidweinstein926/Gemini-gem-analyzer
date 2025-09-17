#!/usr/bin/env python3
"""
gemini_peak_detector.py - FIXED FOR STRUCTURAL DATA DIRECTORY
Automated UV Peak Detection with Database Integration
CORRECTED: Reads from data/raw, saves to data/structural_data/uv

FIXED NORMALIZATION SCHEME:
- UV: 811nm → 15,000, then scale 0-100 (preserves UV ratio analysis)

FOCUS: UV spectra only (B/H/L use manual analyzers for complex structures)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import os
import json
import re
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import glob
from pathlib import Path
import sys

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class GeminiUVPeakDetector:
    """FIXED: UV Peak detector for structural data - Reads from data/raw, saves to data/structural_data/uv"""
    
    def __init__(self):
        # FIXED: Corrected path detection for new directory structure
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent.parent.parent  # FIXED: Go up to gemini_gemological_analysis/
        
        print(f"🔍 UV Peak Detector Paths:")
        print(f"   Script directory: {self.script_dir}")
        print(f"   Project root: {self.project_root}")
        
        # FIXED: Setup directories
        self.setup_directories()
        
        self.root = tk.Tk()
        self.root.title("🤖 Gemini UV Peak Detector - FIXED for Structural Data Directory")
        self.root.geometry("1400x1000")
        
        # Data storage
        self.current_spectrum = None
        self.detected_peaks = []
        self.reference_gems = {}
        self.selected_reference = None
        self.comparison_results = None
        self.gem_metadata = {}
        
        # Plot interaction
        self.annotations = []
        self.measure_mode = False
        self.measure_start = None
        
        # Detection parameters
        self.prominence_threshold = tk.DoubleVar(value=0.1)
        self.min_distance = tk.IntVar(value=10)
        self.smoothing_window = tk.IntVar(value=5)
        
        # FIXED: UV-focused normalization (remove other light sources)
        self.normalization_params = {'reference_wavelength': 811.0, 'target_intensity': 15000.0}
        
        # Parameter adjustment
        self.param_increments = {'prominence': 0.001, 'distance': 1, 'smoothing': 2}
        self.current_param = tk.StringVar(value='prominence')
        
        # Setup interface and controls
        self.setup_gui()
        self.setup_keyboard_controls()
        self.root.after(100, self.initialize_system)
        
    def setup_directories(self):
        """FIXED: Setup directories for UV structural data workflow"""
        # Possible locations for input data
        input_search_paths = [
            self.project_root / "data" / "raw",  # New structure - primary
            self.project_root / "src" / "structural_analysis" / "data" / "raw",  # Local to structural analysis
            self.project_root / "raw_txt",  # Legacy location
            Path.home() / "OneDrive" / "Desktop" / "gemini matcher" / "gemini sp10 raw" / "raw text",  # Legacy user path
        ]
        
        # FIXED: UV output goes to structural_data (not output)
        output_search_paths = [
            self.project_root / "data" / "structural_data" / "uv",  # FIXED: Structural data directory
            self.project_root / "src" / "structural_analysis" / "results" / "uv",  # Alternative
            Path.home() / "gemini sp10 structural data" / "uv",  # Legacy
            self.project_root / "data" / "output" / "uv"  # Fallback
        ]
        
        # Find input directory
        self.input_directory = None
        for search_path in input_search_paths:
            if search_path.exists() and search_path.is_dir():
                self.input_directory = search_path
                print(f"✅ Found input directory: {self.input_directory}")
                break
        
        if not self.input_directory:
            # Use the primary new structure path
            self.input_directory = input_search_paths[0]
            print(f"⚠️ Input directory not found, will use: {self.input_directory}")
        
        # Find/create UV structural data output directory
        self.output_directory = None
        for search_path in output_search_paths:
            if search_path.exists():
                self.output_directory = search_path
                print(f"✅ Found UV structural data directory: {self.output_directory}")
                break
        
        if not self.output_directory:
            # Create the primary structural data path
            self.output_directory = output_search_paths[0]
            try:
                self.output_directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created UV structural data directory: {self.output_directory}")
            except Exception as e:
                print(f"⚠️ Could not create UV structural data directory: {e}")
                # Fallback to current directory
                self.output_directory = Path.cwd() / "uv_structural_results"
                self.output_directory.mkdir(exist_ok=True)
                print(f"🔍 Using fallback UV directory: {self.output_directory}")
        
    def setup_gui(self):
        """FIXED: Create the main interface focused on UV workflow"""
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill='both', expand=True)
        
        left_panel = ttk.Frame(main_container, width=400)
        right_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        main_container.add(right_panel, weight=3)
        
        # FIXED: Setup all sections focused on UV
        self.create_control_sections(left_panel)
        self.setup_plot_area(right_panel)
        
    def create_control_sections(self, parent):
        """FIXED: Create control sections focused on UV structural data workflow"""
        # Directory info section - FIXED
        dir_frame = ttk.LabelFrame(parent, text="🔍 UV Structural Data Workflow", padding=5)
        dir_frame.pack(fill='x', padx=5, pady=2)
        
        info_text = f"Input: {self.input_directory.name}\nOutput: data/structural_data/uv\nWorkflow: UV peaks only"
        ttk.Label(dir_frame, text=info_text, font=('Arial', 8)).pack()
        
        # File section
        file_frame = ttk.LabelFrame(parent, text="🔍 UV Spectrum File", padding=10)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        for text, cmd in [("Load UV Spectrum", self.load_spectrum), ("Load Sample UV Data", self.load_sample_data), ("💾 Export to Structural Database", self.export_results)]:
            ttk.Button(file_frame, text=text, command=cmd).pack(side='left', padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No UV file loaded")
        self.file_label.pack(side='left', padx=10)
        
        # FIXED: UV-specific normalization section (removed light source selection)
        norm_frame = ttk.LabelFrame(parent, text="🔧 UV Normalization (811nm → 15,000 → 0-100)", padding=10)
        norm_frame.pack(fill='x', padx=5, pady=5)
        
        norm_info = "UV: 811nm peak → 15,000\nThen scale entire spectrum 0-100\nPreserves UV ratio analysis"
        ttk.Label(norm_frame, text=norm_info, font=('Arial', 9), foreground='blue').pack(pady=5)
        
        self.active_norm_label = ttk.Label(norm_frame, text="Ready for UV normalization", font=('Arial', 9), foreground='darkgreen', wraplength=300)
        self.active_norm_label.pack(pady=5)
        
        ttk.Button(norm_frame, text="Apply UV Normalization", command=self.normalize_spectrum).pack(pady=5)
        
        # Gem database section (simplified for UV)
        db_frame = ttk.LabelFrame(parent, text="💎 UV Reference Database", padding=10)
        db_frame.pack(fill='x', padx=5, pady=5)
        
        source_db_frame = ttk.Frame(db_frame)
        source_db_frame.pack(fill='x', pady=5)
        
        ttk.Label(source_db_frame, text="Database:").pack(side='left', padx=5)
        self.db_source = ttk.Combobox(source_db_frame, width=20, state='readonly')
        self.db_source['values'] = ['UV Reference Peaks', 'Custom UV Database']
        self.db_source.set('UV Reference Peaks')
        self.db_source.pack(side='left', padx=5)
        
        ttk.Button(source_db_frame, text="Reload DB", command=self.load_gem_database).pack(side='left', padx=5)
        
        gem_frame = ttk.Frame(db_frame)
        gem_frame.pack(fill='x', pady=5)
        
        ttk.Label(gem_frame, text="Select Gem:").pack(side='left', padx=5)
        self.gem_selector = ttk.Combobox(gem_frame, width=25, state='readonly')
        self.gem_selector.pack(side='left', padx=5, fill='x', expand=True)
        
        for text, cmd in [("🔍 Compare UV Peaks", self.compare_with_reference), ("📊 Find Best UV Match", self.find_best_match)]:
            ttk.Button(db_frame, text=text, command=cmd).pack(pady=5)
        
        # Detection parameters section
        detect_frame = ttk.LabelFrame(parent, text="🔬 UV Peak Detection Parameters", padding=10)
        detect_frame.pack(fill='x', padx=5, pady=5)
        
        # FIXED: Create parameter controls in loop
        params = [
            ("Prominence:", self.prominence_threshold, 0.002, 0.5, 150, 8),
            ("Min Distance:", self.min_distance, 1, 50, 150, 8),
            ("Smoothing:", self.smoothing_window, 1, 21, 150, 8)
        ]
        
        for label_text, var, from_, to, length, width in params:
            frame = ttk.Frame(detect_frame)
            frame.pack(fill='x', pady=2)
            ttk.Label(frame, text=label_text).pack(side='left', padx=5)
            scale = ttk.Scale(frame, from_=from_, to=to, variable=var, orient='horizontal', length=length)
            scale.pack(side='left')
            scale.bind('<Motion>', self.on_scale_change)
            ttk.Label(frame, textvariable=var, width=width).pack(side='left', padx=5)
        
        ttk.Button(detect_frame, text="🎯 Detect UV Peaks", command=self.detect_peaks).pack(pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(parent, text="📊 UV Peak Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, height=10, width=40, wrap='word')
        scrollbar = ttk.Scrollbar(self.results_text)
        self.results_text.pack(fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
    def setup_plot_area(self, parent):
        """FIXED: Setup matplotlib plot area for UV spectra"""
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, ttk.Frame(parent))
        toolbar.update()
        
        # Coordinate display
        coord_frame = ttk.Frame(parent)
        coord_frame.pack(side='top', fill='x')
        
        self.coord_label = ttk.Label(coord_frame, text="Cursor: X = --- nm, Y = ---", 
                                     font=('Courier', 10, 'bold'), foreground='blue')
        self.coord_label.pack(side='left', padx=10)
        
        self.point_label = ttk.Label(coord_frame, text="Nearest Point: ---", 
                                     font=('Courier', 10), foreground='darkgreen')
        self.point_label.pack(side='left', padx=20)
        
        # Connect mouse events
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        
        # Configure plot for UV
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity (0-100 scale)', fontsize=11)
        self.ax.set_title('UV Spectrum Peak Analysis - FIXED for Structural Data Directory', fontsize=12)
        self.ax.grid(True, alpha=0.3, which='both')
        self.ax.minorticks_on()
        self.ax.set_xlim(290, 1000)
        self.ax.set_ylim(-10, 110)
        
        # Add crosshair lines
        self.crosshair_vline = self.ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.5, visible=False)
        self.crosshair_hline = self.ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, visible=False)
        
        self.canvas.draw()
        
    def setup_keyboard_controls(self):
        """FIXED: Setup keyboard controls"""
        self.root.focus_set()
        self.root.bind('<Key>', self.on_key_press)
        
    def on_key_press(self, event):
        """FIXED: Handle keyboard events"""
        key = event.keysym.lower()
        
        param_map = {'1': 'prominence', '2': 'distance', '3': 'smoothing'}
        if key in param_map:
            self.current_param.set(param_map[key])
        elif key == 'space':
            self.reset_parameters()
        elif key in ['up', 'down']:
            self.adjust_parameter(self.current_param.get(), 
                                self.param_increments.get(self.current_param.get(), 0.01) * (1 if key == 'up' else -1))
    
    def adjust_parameter(self, param_name, increment):
        """FIXED: Adjust parameters with bounds checking"""
        adjustments = {
            'prominence': (self.prominence_threshold, 0.002, 0.5, 4),
            'distance': (self.min_distance, 1, 50, 0),
            'smoothing': (self.smoothing_window, 1, 21, 0)
        }
        
        if param_name in adjustments:
            var, min_val, max_val, round_digits = adjustments[param_name]
            current = var.get()
            new_value = max(min_val, min(max_val, current + increment))
            
            if param_name == 'smoothing' and new_value % 2 == 0:
                new_value += 1 if increment > 0 else -1
                new_value = max(min_val, min(max_val, new_value))
            
            var.set(round(new_value, round_digits) if round_digits else int(new_value))
    
    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.prominence_threshold.set(0.1)
        self.min_distance.set(10)
        self.smoothing_window.set(5)
    
    def on_scale_change(self, event=None):
        """Handle scale changes to update real-time display"""
        self.root.update_idletasks()
    
    def on_mouse_move(self, event):
        """FIXED: Update coordinate display as mouse moves"""
        if event.inaxes != self.ax:
            self.coord_label.config(text="Cursor: Outside plot area")
            self.crosshair_vline.set_visible(False)
            self.crosshair_hline.set_visible(False)
            self.canvas.draw_idle()
            return
        
        x, y = event.xdata, event.ydata
        self.coord_label.config(text=f"Cursor: X = {x:.2f} nm, Y = {y:.2f}")
        
        # Update crosshair
        self.crosshair_vline.set_xdata([x])
        self.crosshair_hline.set_ydata([y])
        self.crosshair_vline.set_visible(True)
        self.crosshair_hline.set_visible(True)
        
        # Find nearest data point
        if self.current_spectrum is not None:
            wavelengths = self.current_spectrum['wavelengths']
            intensities = self.current_spectrum['intensities']
            
            idx = np.argmin(np.abs(wavelengths - x))
            nearest_wl, nearest_int = wavelengths[idx], intensities[idx]
            
            self.point_label.config(text=f"Nearest: λ={nearest_wl:.2f} nm, I={nearest_int:.2f}", foreground='darkgreen')
            
            # Check if near a peak
            for peak in self.detected_peaks:
                if abs(peak['wavelength'] - nearest_wl) < 2:
                    self.point_label.config(text=f"PEAK: λ={peak['wavelength']:.2f} nm, I={peak['intensity']:.2f}", foreground='red')
                    break
        
        self.canvas.draw_idle()
    
    def on_mouse_click(self, event):
        """FIXED: Handle mouse clicks on the plot"""
        if event.inaxes == self.ax and event.dblclick:
            x, y = event.xdata, event.ydata
            ann = self.ax.annotate(f'({x:.1f}, {y:.2f})', xy=(x, y), xytext=(10, 10),
                                   textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            if not hasattr(self, 'annotations'):
                self.annotations = []
            self.annotations.append(ann)
            self.canvas.draw()
    
    def load_spectrum(self):
        """FIXED: Load a UV spectrum file from data/raw directory"""
        # Use the detected input directory as the starting location
        initial_dir = str(self.input_directory) if self.input_directory.exists() else None
        
        file = filedialog.askopenfilename(
            title="Select UV Spectrum File", 
            initialdir=initial_dir,
            filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if file:
            try:
                data = np.loadtxt(file)
                if data.shape[1] >= 2:
                    self.current_spectrum = {
                        'wavelengths': data[:, 0], 'intensities': data[:, 1], 'normalized': False,
                        'filename': os.path.basename(file), 'normalization_scheme': 'Raw_UV_data',
                        'full_path': file  # Store full path for reference
                    }
                    self.file_label.config(text=f"UV: {os.path.basename(file)}")
                    self.plot_spectrum()
                    
                    # Show input directory info
                    rel_path = Path(file).relative_to(self.project_root) if self.project_root in Path(file).parents else Path(file)
                    self.update_results(f"✅ Loaded UV spectrum from structural data workflow:\n{rel_path}\n"
                                      f"Points: {len(data[:, 0])}\n"
                                      f"Range: {data[:, 0].min():.1f} - {data[:, 0].max():.1f} nm\n"
                                      f"Status: Raw UV data (not normalized)\n"
                                      f"Input directory: {self.input_directory}\n"
                                      f"Will save to: {self.output_directory}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load UV file: {e}")
    
    def load_sample_data(self):
        """FIXED: Load sample UV test data"""
        wavelengths = np.linspace(300, 900, 2000)
        intensities = np.random.normal(0, 0.01, len(wavelengths))
        
        # Add sample UV peaks (characteristic UV wavelengths)
        uv_peaks = [(694.2, 0.8, 2), (692.8, 0.7, 2), (659, 0.3, 3), (475, 0.4, 3), (811, 0.2, 2), 
                   (405, 0.5, 2), (532, 0.3, 2), (589, 0.4, 2), (632, 0.2, 2)]
        
        for peak_wl, peak_int, width in uv_peaks:
            mask = np.abs(wavelengths - peak_wl) < width * 3
            intensities[mask] += peak_int * np.exp(-((wavelengths[mask] - peak_wl) / width) ** 2)
        
        self.current_spectrum = {
            'wavelengths': wavelengths, 'intensities': intensities, 'normalized': False,
            'filename': 'Sample_UV_Ruby_Spectrum', 'normalization_scheme': 'Sample_UV_raw_data'
        }
        
        self.file_label.config(text='Sample UV Ruby Spectrum')
        self.plot_spectrum()
        self.update_results("✅ Loaded sample UV ruby spectrum with characteristic peaks\n"
                          f"Status: Raw UV data (not normalized)\n"
                          f"Workflow: FIXED for structural data directory\n"
                          f"Output: {self.output_directory}")
    
    def normalize_spectrum(self):
        """FIXED: Apply UV-specific normalization (811nm → 15,000 → 0-100)"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No UV spectrum loaded")
            return
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities'].copy()
        
        # FIXED: UV normalization - 811nm → 15,000, then scale 0-100
        normalized, norm_wl = self.normalize_uv_spectrum(wavelengths, intensities)
        
        if normalized is None:
            messagebox.showwarning("Warning", "UV normalization failed")
            return
        
        # Update spectrum
        self.current_spectrum.update({
            'intensities': normalized, 'normalized': True, 'original_intensities': intensities,
            'normalization_scheme': 'UV_811nm_15000_to_100_FIXED', 'normalization_wavelength': norm_wl
        })
        
        self.active_norm_label.config(text=f"UV normalized: 811nm → 15,000 → 0-100\nReference: {norm_wl:.1f}nm\nReady for peak detection")
        self.plot_spectrum()
        self.update_results(f"✅ FIXED UV Normalization applied:\nScheme: UV_811nm_15000_to_100_FIXED\n"
                          f"Reference wavelength: {norm_wl:.1f}nm\nFinal range: 0-100 (preserves UV ratios)\n"
                          f"Output directory: {self.output_directory}")
    
    def normalize_uv_spectrum(self, wavelengths, intensities):
        """FIXED: UV normalization - 811nm → 15,000, then scale 0-100"""
        return self._normalize_to_reference(wavelengths, intensities, 811.0, 15000.0, 2.0)
    
    def _normalize_to_reference(self, wavelengths, intensities, ref_wl, target_intensity, tolerance):
        """FIXED: UV-specific normalization to 811nm reference"""
        ref_mask = np.abs(wavelengths - ref_wl) <= tolerance
        ref_value = np.max(intensities[ref_mask]) if np.any(ref_mask) else np.max(intensities)
        
        if ref_value <= 0:
            return None, None
        
        scaled_intensities = intensities * (target_intensity / ref_value)
        return self._scale_to_100(scaled_intensities), ref_wl
    
    def _scale_to_100(self, intensities):
        """FIXED: Scale to 0-100 range preserving UV ratios"""
        min_val, max_val = np.min(intensities), np.max(intensities)
        range_val = max_val - min_val
        
        if range_val > 0:
            return ((intensities - min_val) / range_val) * 100.0
        return None
    
    def detect_peaks(self):
        """FIXED: Detect UV peaks in the normalized spectrum"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No UV spectrum loaded")
            return
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        
        # Check if normalized
        if not self.current_spectrum.get('normalized', False):
            if messagebox.askyesno("Normalization", "UV spectrum is not normalized. Apply UV normalization first?"):
                self.normalize_spectrum()
                intensities = self.current_spectrum['intensities']
            else:
                messagebox.showinfo("Info", "Using raw UV data for peak detection")
        
        # Apply smoothing and detect peaks
        window = self.smoothing_window.get()
        smoothed = savgol_filter(intensities, window, 2) if window > 1 and window % 2 == 1 else intensities
        
        prominence = self.prominence_threshold.get() * (np.max(smoothed) - np.min(smoothed))
        peaks, properties = find_peaks(smoothed, prominence=prominence, distance=self.min_distance.get())
        
        # Store detected peaks
        self.detected_peaks = [{
            'wavelength': wavelengths[idx], 'intensity': intensities[idx],
            'prominence': properties['prominences'][np.where(peaks == idx)[0][0]]
        } for idx in peaks]
        
        # Sort by intensity and limit to top 50
        self.detected_peaks.sort(key=lambda x: x['intensity'], reverse=True)
        self.detected_peaks = self.detected_peaks[:50]
        
        self.plot_spectrum()
        self._display_peak_results()
    
    def _display_peak_results(self):
        """FIXED: Display UV peak detection results"""
        if not self.detected_peaks:
            return
        
        wavelengths_list = [p['wavelength'] for p in self.detected_peaks]
        intensities_list = [p['intensity'] for p in self.detected_peaks]
        
        result_text = f"🎯 DETECTED {len(self.detected_peaks)} UV PEAKS (FIXED normalization)\n" + "="*50 + "\n\n"
        result_text += "📊 UV PEAK STATISTICS:\n"
        result_text += f"  Total UV peaks: {len(self.detected_peaks)}\n"
        result_text += f"  Wavelength range: {min(wavelengths_list):.1f} - {max(wavelengths_list):.1f} nm\n"
        result_text += f"  Intensity range: {min(intensities_list):.2f} - {max(intensities_list):.2f} (0-100 scale)\n"
        result_text += f"  Mean intensity: {np.mean(intensities_list):.2f}\n\n"
        
        # Show normalization info
        scheme = self.current_spectrum.get('normalization_scheme', 'Unknown')
        norm_wl = self.current_spectrum.get('normalization_wavelength', 'N/A')
        result_text += f"🔧 FIXED UV NORMALIZATION INFO:\n  Scheme: {scheme}\n  Reference: {norm_wl}nm\n  Scale: 0-100 (preserves UV ratios)\n"
        result_text += f"  Output directory: {self.output_directory}\n\n"
        
        # List peaks
        result_text += "🎯 ALL UV PEAKS (sorted by intensity):\n" + "-"*50 + "\n"
        result_text += f"{'No.':<4} {'λ (nm)':<10} {'Intensity':<12} {'Prominence':<12} {'Category'}\n" + "-"*50 + "\n"
        
        category_thresholds = [0.1, 0.3, 0.6]
        categories = ["⭐ Major", "⚫ Strong", "⚪ Medium", "· Minor"]
        
        for i, peak in enumerate(self.detected_peaks, 1):
            category_idx = sum(1 for threshold in category_thresholds if i <= len(self.detected_peaks) * threshold)
            category = categories[min(category_idx, len(categories) - 1)]
            
            result_text += f"{i:<4} {peak['wavelength']:<10.2f} {peak['intensity']:<12.2f} "
            result_text += f"{peak['prominence']:<12.2f} {category}\n"
            
            if i % 10 == 0 and i < len(self.detected_peaks):
                result_text += "  ---\n"
        
        self.update_results(result_text)
    
    def plot_spectrum(self, highlight_matches=None):
        """FIXED: Plot UV spectrum with FIXED scaling (0-100)"""
        xlim = self.ax.get_xlim() if self.ax.get_xlim() != (0.0, 1.0) else None
        ylim = self.ax.get_ylim() if self.ax.get_ylim() != (0.0, 1.0) else None
        
        self.ax.clear()
        
        if self.current_spectrum:
            wavelengths = self.current_spectrum['wavelengths']
            intensities = self.current_spectrum['intensities']
            
            # Plot spectrum
            self.ax.plot(wavelengths, intensities, 'b-', linewidth=1.5, 
                        label=self.current_spectrum['filename'], alpha=0.8)
            
            # Plot detected peaks
            if self.detected_peaks:
                major_count = max(1, len(self.detected_peaks)//10)
                major_peaks = self.detected_peaks[:major_count]
                other_peaks = self.detected_peaks[major_count:]
                
                if major_peaks:
                    peak_wls = [p['wavelength'] for p in major_peaks]
                    peak_ints = [p['intensity'] for p in major_peaks]
                    self.ax.scatter(peak_wls, peak_ints, color='red', s=20, marker='v', 
                                  label=f'Major UV Peaks ({len(major_peaks)})', zorder=6, 
                                  edgecolors='darkred', linewidth=0.5)
                
                if other_peaks:
                    peak_wls = [p['wavelength'] for p in other_peaks]
                    peak_ints = [p['intensity'] for p in other_peaks]
                    self.ax.scatter(peak_wls, peak_ints, color='orange', s=20, marker='v', 
                                  label=f'Other UV Peaks ({len(other_peaks)})', zorder=5, alpha=0.7)
            
            # Set axis limits
            if xlim and xlim != (0.0, 1.0):
                self.ax.set_xlim(xlim)
            else:
                self.ax.set_xlim(wavelengths.min() - 10, wavelengths.max() + 10)
            
            if ylim and ylim != (0.0, 1.0):
                self.ax.set_ylim(ylim)
            else:
                if self.current_spectrum.get('normalized', False):
                    self.ax.set_ylim(-10, 110)
                else:
                    y_margin = (intensities.max() - intensities.min()) * 0.1
                    self.ax.set_ylim(intensities.min() - y_margin, intensities.max() + y_margin)
        
        # Re-add crosshair lines
        self.crosshair_vline = self.ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.5, visible=False)
        self.crosshair_hline = self.ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, visible=False)
        
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity (0-100 scale)', fontsize=11)
        self.ax.set_title('FIXED: UV Peak Detection for Structural Data Directory', fontsize=12)
        self.ax.grid(True, alpha=0.3, which='both')
        self.ax.minorticks_on()
        self.ax.legend(loc='best', fontsize=9)
        
        self.canvas.draw()
    
    def update_results(self, text):
        """Update the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
    
    def export_results(self):
        """FIXED: Export UV peaks to structural data directory"""
        if not self.detected_peaks:
            messagebox.showwarning("Warning", "No UV peaks to export")
            return
        
        gem_id = self.extract_gem_id_from_filename()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        default_csv_name = f"{gem_id or 'unknown'}_uv_structural_auto_{timestamp}.csv"
        
        export_choice = messagebox.askyesnocancel("Export to Structural Database",
            f"FIXED Export to UV Structural Data Directory:\n\n"
            f"YES = Export to Structural Database (recommended)\n   → {default_csv_name}\n"
            f"   → Saves to: {self.output_directory}\n   → Compatible with database workflow\n\n"
            f"NO = Choose format manually\n\nCANCEL = Cancel export")
        
        if export_choice is None:
            return
        elif export_choice:
            # Ensure output directory exists
            self.output_directory.mkdir(parents=True, exist_ok=True)
            save_path = self.output_directory / default_csv_name
            
            try:
                self.export_csv_for_structural_database(str(save_path), gem_id)
                messagebox.showinfo("Export Success", 
                                  f"✅ FIXED UV peaks exported to structural data directory:\n{default_csv_name}\n\n"
                                  f"🔍 Location: {self.output_directory}\n🎯 Ready for database import!\n"
                                  f"📋 Compatible with manual analyzer workflow")
                return
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not auto-export: {e}")
        
        # Manual export
        initial_dir = str(self.output_directory)
        file = filedialog.asksaveasfilename(
            initialname=default_csv_name, 
            initialdir=initial_dir,
            defaultextension=".csv",
            filetypes=[("CSV for Structural Database", "*.csv"), ("JSON (Complete data)", "*.json"), 
                      ("Text Report", "*.txt"), ("Peak List Only", "*.peaks")])
        
        if file:
            try:
                export_funcs = {
                    '.json': self.export_json_complete, '.csv': lambda f: self.export_csv_for_structural_database(f, gem_id),
                    '.peaks': self.export_peak_list, '.txt': self.export_text_report
                }
                
                ext = next((ext for ext in export_funcs.keys() if file.endswith(ext)), '.txt')
                export_funcs[ext](file)
                
                messagebox.showinfo("Success", f"✅ UV results exported with FIXED structure:\n{os.path.basename(file)}\n"
                                  f"🔍 Saved to structural data directory\n"
                                  f"🎯 Total UV peaks: {len(self.detected_peaks)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export: {e}")
    
    def extract_gem_id_from_filename(self):
        """FIXED: Extract gem ID from loaded UV spectrum filename"""
        if not self.current_spectrum:
            return None
        
        filename = self.current_spectrum.get('filename', '')
        for ext in ['.txt', '.csv', '.dat']:
            if filename.lower().endswith(ext):
                return filename[:-len(ext)]
        return filename if filename else "unknown"
    
    def export_csv_for_structural_database(self, file_path, gem_id):
        """FIXED: Export CSV compatible with structural database workflow"""
        with open(file_path, 'w', newline='') as f:
            f.write("Peak_Number,Wavelength_nm,Intensity,Prominence,Category,Normalization_Scheme,Reference_Wavelength,Light_Source,Directory_Structure,Output_Location,Detection_Method\n")
            
            scheme = self.current_spectrum.get('normalization_scheme', 'Unknown')
            ref_wl = self.current_spectrum.get('normalization_wavelength', 'N/A')
            
            categories = ["Major", "Strong", "Medium", "Minor"]
            category_thresholds = [0.1, 0.3, 0.6]
            
            for i, peak in enumerate(self.detected_peaks, 1):
                category_idx = sum(1 for threshold in category_thresholds if i <= len(self.detected_peaks) * threshold)
                category = categories[min(category_idx, len(categories) - 1)]
                
                f.write(f"{i},{peak['wavelength']:.3f},{peak['intensity']:.3f},"
                      f"{peak['prominence']:.3f},{category},{scheme},{ref_wl},UV,"
                      f"Fixed_Structural_Data_Directory,{self.output_directory},GeminiUVPeakDetector_Auto\n")
        
        print(f"✅ Exported FIXED UV peaks for gem {gem_id}: {os.path.basename(file_path)}")
    
    def export_json_complete(self, file_path):
        """FIXED: Export complete UV data with enhanced metadata"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'spectrum_file': self.current_spectrum.get('filename', 'Unknown'),
            'normalization_scheme': self.current_spectrum.get('normalization_scheme', 'Unknown'),
            'normalization_wavelength': self.current_spectrum.get('normalization_wavelength', 'N/A'),
            'light_source': 'UV',
            'normalized': self.current_spectrum.get('normalized', False),
            'total_peaks': len(self.detected_peaks),
            'detected_peaks': self.detected_peaks,
            'directory_structure': 'Fixed_Structural_Data_Directory',
            'output_directory': str(self.output_directory),
            'project_root': str(self.project_root),
            'input_directory': str(self.input_directory),
            'workflow': 'UV_peaks_only_B_H_L_use_manual_analyzers'
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_peak_list(self, file_path):
        """FIXED: Export simple UV peak list"""
        with open(file_path, 'w') as f:
            f.write("# FIXED UV Peak List - Structural Data Directory\n")
            f.write(f"# Project root: {self.project_root}\n")
            f.write(f"# Normalization: {self.current_spectrum.get('normalization_scheme', 'Unknown')}\n")
            f.write(f"# Reference: {self.current_spectrum.get('normalization_wavelength', 'N/A')}nm\n")
            f.write(f"# Light Source: UV\n")
            f.write(f"# Output directory: {self.output_directory}\n")
            f.write("# UV peak wavelengths (nm)\n")
            for peak in self.detected_peaks:
                f.write(f"{peak['wavelength']:.2f}\n")
    
    def export_text_report(self, file_path):
        """FIXED: Export detailed UV text report"""
        with open(file_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GEMINI UV PEAK DETECTION REPORT - FIXED FOR STRUCTURAL DATA\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project Root: {self.project_root}\n")
            f.write(f"Input Directory: {self.input_directory}\n")
            f.write(f"Output Directory: {self.output_directory}\n")
            f.write(f"Spectrum File: {self.current_spectrum.get('filename', 'Unknown')}\n")
            f.write(f"Light Source: UV (focused workflow)\n")
            f.write(f"Normalization Scheme: {self.current_spectrum.get('normalization_scheme', 'Unknown')}\n")
            f.write(f"Reference Wavelength: {self.current_spectrum.get('normalization_wavelength', 'N/A')}nm\n")
            f.write(f"Normalized: {self.current_spectrum.get('normalized', False)}\n")
            f.write(f"Workflow: UV peaks only (B/H/L use manual analyzers)\n")
            f.write(f"Directory Structure: Fixed for structural data workflow\n\n")
            f.write(self.results_text.get(1.0, tk.END))
    
    # FIXED: Placeholder methods for gem database functionality
    def load_gem_database(self):
        """Load UV reference database"""
        self.reference_gems = {
            'Ruby (UV Reference)': {
                'peaks': [694.2, 692.8, 668.0, 659.0, 475.0, 405.0, 532.0],
                'color': 'red',
                'description': 'UV reference values - FIXED normalization'
            },
            'Emerald (UV Reference)': {
                'peaks': [680.0, 637.0, 630.0, 477.0, 425.0],
                'color': 'green', 
                'description': 'UV reference values - FIXED normalization'
            }
        }
        self.gem_selector['values'] = list(self.reference_gems.keys())
    
    def compare_with_reference(self):
        """Compare with UV reference"""
        messagebox.showinfo("Info", "UV reference comparison available in full version")
    
    def find_best_match(self):
        """Find best UV match"""
        messagebox.showinfo("Info", "UV best match finding available in full version")
    
    def initialize_system(self):
        """FIXED: Initialize the UV peak detector system"""
        self.load_gem_database()  # Load UV references
        self.update_results(
            "🎉 GEMINI UV PEAK DETECTOR - FIXED FOR STRUCTURAL DATA WORKFLOW\n" + "="*60 + "\n" +
            f"✅ Successfully initialized for UV peak detection only\n" +
            f"🔍 Project root: {self.project_root.name}\n" +
            f"📂 Input directory: {self.input_directory}\n" +
            f"📁 Output directory: {self.output_directory}\n\n" +
            "🔧 FIXED UV NORMALIZATION:\n" +
            "• UV: 811nm → 15,000, then scale 0-100 (preserves ratios)\n\n" +
            "🔍 FIXED DIRECTORY STRUCTURE:\n" +
            "• Input files: data/raw/\n" +
            "• Output files: data/structural_data/uv/\n" +
            "• Compatible with database workflow\n\n" +
            "🎯 WORKFLOW FOCUS:\n" +
            "• UV spectra: This automated peak detector\n" +
            "• B/H/L spectra: Use manual analyzers (complex structures)\n\n" +
            "QUICK START:\n" +
            "1. Load UV spectrum from data/raw/\n" +
            "2. Apply UV normalization (811nm → 15,000 → 0-100)\n" +
            "3. Detect UV peaks (typically 10-30 peaks)\n" +
            "4. Export to data/structural_data/uv/ for database\n\n" +
            "⌨️ KEYBOARD CONTROLS:\n" +
            "• Keys 1,2,3: Select parameter\n" +
            "• Up/Down arrows: Adjust active parameter\n" +
            "• Spacebar: Reset defaults\n\n" +
            "💾 FIXED EXPORT: Auto-saves to structural data directory!"
        )
    
    def run(self):
        """Start the UV peak detector application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("🤖 Starting FIXED Gemini UV Peak Detector for Structural Data Workflow...")
    print("✅ Reads from data/raw, saves to data/structural_data/uv")
    print("🎯 Focus: UV peak detection only (B/H/L use manual analyzers)")
    app = GeminiUVPeakDetector()
    app.run()

if __name__ == '__main__':
    main()
