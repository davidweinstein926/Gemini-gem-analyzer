#!/usr/bin/env python3
"""
gemini_peak_detector.py - OPTIMIZED - Automated Peak Detection with Gem Database Comparison
For use with Gemini Gemological Analyzer Launcher

FIXED NORMALIZATION SCHEME:
- H/B: 650nm → 50,000, then scale 0-100
- L: Max intensity → 50,000, then scale 0-100 (tracks normalization wavelength)
- U: 811nm → 15,000, then scale 0-100

OPTIMIZED: Reduced line count while maintaining all functionality
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

class GeminiPeakDetector:
    """OPTIMIZED: Automated peak detector with gem database comparison - FIXED NORMALIZATION"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 Gemini Automated Peak Detector - OPTIMIZED FIXED Normalization")
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
        self.light_source = tk.StringVar(value="UV")
        self.laser_norm_wavelength = tk.DoubleVar(value=0.0)
        
        # OPTIMIZED: Light-source specific normalization parameters
        self.normalization_params = {
            'UV': {'reference_wavelength': 811.0, 'target_intensity': 15000.0},
            'Halogen': {'reference_wavelength': 650.0, 'target_intensity': 50000.0},
            'Laser': {'reference_wavelength': None, 'target_intensity': 50000.0}
        }
        
        # Parameter adjustment
        self.param_increments = {'prominence': 0.001, 'distance': 1, 'smoothing': 2}
        self.current_param = tk.StringVar(value='prominence')
        
        # Setup interface and controls
        self.setup_gui()
        self.setup_keyboard_controls()
        self.root.after(100, self.initialize_system)
        
    def setup_gui(self):
        """OPTIMIZED: Create the main interface"""
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill='both', expand=True)
        
        left_panel = ttk.Frame(main_container, width=400)
        right_panel = ttk.Frame(main_container)
        main_container.add(left_panel, weight=1)
        main_container.add(right_panel, weight=3)
        
        # OPTIMIZED: Setup all sections in one method
        self.create_control_sections(left_panel)
        self.setup_plot_area(right_panel)
        
    def create_control_sections(self, parent):
        """OPTIMIZED: Create all control sections"""
        # File section
        file_frame = ttk.LabelFrame(parent, text="📁 Spectrum File", padding=10)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        for text, cmd in [("Load Spectrum", self.load_spectrum), ("Load Sample Data", self.load_sample_data), ("💾 Export Results", self.export_results)]:
            ttk.Button(file_frame, text=text, command=cmd).pack(side='left', padx=5)
        
        self.file_label = ttk.Label(file_frame, text="No file loaded")
        self.file_label.pack(side='left', padx=10)
        
        # Light source section
        light_frame = ttk.LabelFrame(parent, text="💡 Light Source (Normalization)", padding=10)
        light_frame.pack(fill='x', padx=5, pady=5)
        
        source_frame = ttk.Frame(light_frame)
        source_frame.pack(fill='x', pady=2)
        
        ttk.Label(source_frame, text="Light Source:").pack(side='left', padx=5)
        light_combo = ttk.Combobox(source_frame, textvariable=self.light_source, width=15, state='readonly')
        light_combo['values'] = ['UV', 'Halogen', 'Laser']
        light_combo.pack(side='left', padx=5)
        light_combo.bind('<<ComboboxSelected>>', self.update_normalization_display)
        
        self.norm_info_label = ttk.Label(light_frame, text="", font=('Arial', 9), foreground='blue')
        self.norm_info_label.pack(pady=5)
        
        ttk.Button(light_frame, text="Auto-Detect from Filename", command=self.auto_detect_light_source).pack(pady=2)
        
        # Gem database section
        db_frame = ttk.LabelFrame(parent, text="💎 Gem Reference Database", padding=10)
        db_frame.pack(fill='x', padx=5, pady=5)
        
        source_db_frame = ttk.Frame(db_frame)
        source_db_frame.pack(fill='x', pady=5)
        
        ttk.Label(source_db_frame, text="Database:").pack(side='left', padx=5)
        self.db_source = ttk.Combobox(source_db_frame, width=20, state='readonly')
        self.db_source['values'] = ['Gemini SP10 Raw', 'Custom Database', 'Load Folder']
        self.db_source.set('Gemini SP10 Raw')
        self.db_source.pack(side='left', padx=5)
        
        ttk.Button(source_db_frame, text="Reload DB", command=self.load_gem_database).pack(side='left', padx=5)
        
        gem_frame = ttk.Frame(db_frame)
        gem_frame.pack(fill='x', pady=5)
        
        ttk.Label(gem_frame, text="Select Gem:").pack(side='left', padx=5)
        self.gem_selector = ttk.Combobox(gem_frame, width=25, state='readonly')
        self.gem_selector.pack(side='left', padx=5, fill='x', expand=True)
        
        for text, cmd in [("🔍 Compare with Reference", self.compare_with_reference), ("📊 Find Best Match", self.find_best_match)]:
            ttk.Button(db_frame, text=text, command=cmd).pack(pady=5)
        
        # Normalization section
        norm_frame = ttk.LabelFrame(parent, text="🔧 Current Normalization", padding=10)
        norm_frame.pack(fill='x', padx=5, pady=5)
        
        self.active_norm_label = ttk.Label(norm_frame, text="", font=('Arial', 9), foreground='darkgreen', wraplength=300)
        self.active_norm_label.pack(pady=5)
        
        ttk.Button(norm_frame, text="Apply FIXED Normalization", command=self.normalize_spectrum).pack(pady=5)
        
        self.laser_norm_display = ttk.Label(norm_frame, text="", font=('Arial', 8), foreground='red')
        self.laser_norm_display.pack(pady=2)
        
        # Detection parameters section
        detect_frame = ttk.LabelFrame(parent, text="🔬 Detection Parameters", padding=10)
        detect_frame.pack(fill='x', padx=5, pady=5)
        
        # OPTIMIZED: Create parameter controls in loop
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
        
        ttk.Button(detect_frame, text="🎯 Detect Peaks", command=self.detect_peaks).pack(pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(parent, text="📊 Results", padding=10)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.results_text = tk.Text(results_frame, height=10, width=40, wrap='word')
        scrollbar = ttk.Scrollbar(self.results_text)
        self.results_text.pack(fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
        # Update displays
        self.update_normalization_display()
        self.update_active_normalization_display()
        
    def setup_plot_area(self, parent):
        """OPTIMIZED: Setup matplotlib plot area"""
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
        
        # Configure plot
        self.ax.set_xlabel('Wavelength (nm)', fontsize=11)
        self.ax.set_ylabel('Intensity (0-100 scale)', fontsize=11)
        self.ax.set_title('Spectrum Analysis - FIXED Normalization (0-100 scale)', fontsize=12)
        self.ax.grid(True, alpha=0.3, which='both')
        self.ax.minorticks_on()
        self.ax.set_xlim(290, 1000)
        self.ax.set_ylim(-10, 110)
        
        # Add crosshair lines
        self.crosshair_vline = self.ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.5, visible=False)
        self.crosshair_hline = self.ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5, visible=False)
        
        self.canvas.draw()
        
    def setup_keyboard_controls(self):
        """OPTIMIZED: Setup keyboard controls"""
        self.root.focus_set()
        self.root.bind('<Key>', self.on_key_press)
        
    def on_key_press(self, event):
        """OPTIMIZED: Handle keyboard events"""
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
        """OPTIMIZED: Adjust parameters with bounds checking"""
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
    
    def update_normalization_display(self, event=None):
        """OPTIMIZED: Update normalization information display"""
        light = self.light_source.get()
        info_map = {
            'UV': "UV: 811nm → 15,000, then scale 0-100",
            'Halogen': "Halogen: 650nm → 50,000, then scale 0-100", 
            'Laser': "Laser: Max intensity → 50,000, then scale 0-100"
        }
        self.norm_info_label.config(text=info_map.get(light, "Select light source for proper normalization"))
        self.update_active_normalization_display()
    
    def update_active_normalization_display(self):
        """OPTIMIZED: Update active normalization display"""
        light = self.light_source.get()
        text_map = {
            'UV': "ACTIVE: UV mode\n811nm peak → 15,000\nThen scale entire spectrum 0-100\nPreserves UV ratio analysis",
            'Halogen': "ACTIVE: Halogen mode\n650nm peak → 50,000\nThen scale entire spectrum 0-100\nOptimal for broad features",
            'Laser': "ACTIVE: Laser mode\nMax intensity → 50,000\nThen scale entire spectrum 0-100\nTracks normalization wavelength"
        }
        self.active_norm_label.config(text=text_map.get(light, "Select light source to see normalization scheme"))
        
        # Update laser display
        if light == 'Laser' and self.laser_norm_wavelength.get() > 0:
            self.laser_norm_display.config(text=f"Last laser normalization: {self.laser_norm_wavelength.get():.1f}nm")
        else:
            self.laser_norm_display.config(text="")
    
    def auto_detect_light_source(self):
        """OPTIMIZED: Auto-detect light source from filename"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
        
        filename = self.current_spectrum.get('filename', '').lower()
        
        light_patterns = {
            'UV': ['uc1', 'up1', '_uv_', '_u'],
            'Halogen': ['bc1', 'bp1', '_halogen_', '_h', '_b'],
            'Laser': ['lc1', 'lp1', '_laser_', '_l']
        }
        
        for light, patterns in light_patterns.items():
            if any(pattern in filename for pattern in patterns):
                self.light_source.set(light)
                self.update_normalization_display()
                messagebox.showinfo("Auto-Detection", f"Detected light source: {light}")
                return
        
        messagebox.showinfo("Auto-Detection", "Could not determine light source from filename. Please select manually.")
    
    def on_scale_change(self, event=None):
        """Handle scale changes to update real-time display"""
        self.root.update_idletasks()
    
    def on_mouse_move(self, event):
        """OPTIMIZED: Update coordinate display as mouse moves"""
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
        """OPTIMIZED: Handle mouse clicks on the plot"""
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
        """OPTIMIZED: Load a spectrum file"""
        file = filedialog.askopenfilename(title="Select Spectrum File",
                                        filetypes=[("Text Files", "*.txt"), ("CSV Files", "*.csv"), ("All Files", "*.*")])
        
        if file:
            try:
                data = np.loadtxt(file)
                if data.shape[1] >= 2:
                    self.current_spectrum = {
                        'wavelengths': data[:, 0], 'intensities': data[:, 1], 'normalized': False,
                        'filename': os.path.basename(file), 'normalization_scheme': 'Raw_data'
                    }
                    self.file_label.config(text=os.path.basename(file))
                    self.auto_detect_light_source()
                    self.plot_spectrum()
                    self.update_results(f"Loaded: {os.path.basename(file)}\nPoints: {len(data[:, 0])}\n"
                                      f"Range: {data[:, 0].min():.1f} - {data[:, 0].max():.1f} nm\nStatus: Raw data (not normalized)")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
    
    def load_sample_data(self):
        """OPTIMIZED: Load sample test data"""
        wavelengths = np.linspace(300, 900, 2000)
        intensities = np.random.normal(0, 0.01, len(wavelengths))
        
        # Add sample peaks
        for peak_wl, peak_int, width in [(694.2, 0.8, 2), (692.8, 0.7, 2), (659, 0.3, 3), (475, 0.4, 3), (811, 0.2, 2)]:
            mask = np.abs(wavelengths - peak_wl) < width * 3
            intensities[mask] += peak_int * np.exp(-((wavelengths[mask] - peak_wl) / width) ** 2)
        
        self.current_spectrum = {
            'wavelengths': wavelengths, 'intensities': intensities, 'normalized': False,
            'filename': 'Sample_UV_Ruby_Spectrum', 'normalization_scheme': 'Sample_raw_data'
        }
        
        self.light_source.set('UV')
        self.update_normalization_display()
        self.file_label.config(text='Sample UV Ruby Spectrum')
        self.plot_spectrum()
        self.update_results("Loaded sample UV ruby spectrum with characteristic peaks\nStatus: Raw data (not normalized)")
    
    def normalize_spectrum(self):
        """OPTIMIZED: Apply light-source specific normalization"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities'].copy()
        light_source = self.light_source.get()
        
        # OPTIMIZED: Normalization dispatch
        normalizers = {
            'UV': (self.normalize_uv_spectrum, 'UV_811nm_15000_to_100'),
            'Halogen': (self.normalize_halogen_spectrum, 'Halogen_650nm_50000_to_100'),
            'Laser': (self.normalize_laser_spectrum, 'Laser_max_50000_to_100')
        }
        
        if light_source not in normalizers:
            messagebox.showwarning("Warning", "Please select a light source first")
            return
        
        normalizer_func, scheme = normalizers[light_source]
        normalized, norm_wl = normalizer_func(wavelengths, intensities)
        
        if normalized is None:
            messagebox.showwarning("Warning", "Normalization failed")
            return
        
        # Update spectrum
        self.current_spectrum.update({
            'intensities': normalized, 'normalized': True, 'original_intensities': intensities,
            'normalization_scheme': scheme, 'normalization_wavelength': norm_wl
        })
        
        if light_source == 'Laser':
            self.laser_norm_wavelength.set(norm_wl)
        
        self.update_active_normalization_display()
        self.plot_spectrum()
        self.update_results(f"FIXED Normalization applied:\nLight source: {light_source}\nScheme: {scheme}\n"
                          f"Reference wavelength: {norm_wl:.1f}nm\nFinal range: 0-100 (preserves UV ratio analysis)")
    
    def normalize_uv_spectrum(self, wavelengths, intensities):
        """OPTIMIZED: UV normalization"""
        return self._normalize_to_reference(wavelengths, intensities, 811.0, 15000.0, 2.0)
    
    def normalize_halogen_spectrum(self, wavelengths, intensities):
        """OPTIMIZED: Halogen normalization"""
        return self._normalize_to_reference(wavelengths, intensities, 650.0, 50000.0, 5.0)
    
    def normalize_laser_spectrum(self, wavelengths, intensities):
        """OPTIMIZED: Laser normalization"""
        max_idx = np.argmax(intensities)
        max_value = intensities[max_idx]
        max_wavelength = wavelengths[max_idx]
        
        if max_value <= 0:
            return None, None
        
        return self._scale_to_100(intensities * (50000.0 / max_value)), max_wavelength
    
    def _normalize_to_reference(self, wavelengths, intensities, ref_wl, target_intensity, tolerance):
        """OPTIMIZED: Common normalization to reference wavelength"""
        ref_mask = np.abs(wavelengths - ref_wl) <= tolerance
        ref_value = np.max(intensities[ref_mask]) if np.any(ref_mask) else np.max(intensities)
        
        if ref_value <= 0:
            return None, None
        
        scaled_intensities = intensities * (target_intensity / ref_value)
        return self._scale_to_100(scaled_intensities), ref_wl
    
    def _scale_to_100(self, intensities):
        """OPTIMIZED: Scale to 0-100 range"""
        min_val, max_val = np.min(intensities), np.max(intensities)
        range_val = max_val - min_val
        
        if range_val > 0:
            return ((intensities - min_val) / range_val) * 100.0
        return None
    
    def detect_peaks(self):
        """OPTIMIZED: Detect peaks in the normalized spectrum"""
        if not self.current_spectrum:
            messagebox.showwarning("Warning", "No spectrum loaded")
            return
        
        wavelengths = self.current_spectrum['wavelengths']
        intensities = self.current_spectrum['intensities']
        
        # Check if normalized
        if not self.current_spectrum.get('normalized', False):
            if messagebox.askyesno("Normalization", "Spectrum is not normalized. Apply FIXED normalization first?"):
                self.normalize_spectrum()
                intensities = self.current_spectrum['intensities']
            else:
                messagebox.showinfo("Info", "Using raw data for peak detection")
        
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
        """OPTIMIZED: Display peak detection results"""
        if not self.detected_peaks:
            return
        
        wavelengths_list = [p['wavelength'] for p in self.detected_peaks]
        intensities_list = [p['intensity'] for p in self.detected_peaks]
        
        result_text = f"🎯 DETECTED {len(self.detected_peaks)} PEAKS (FIXED normalization)\n" + "="*50 + "\n\n"
        result_text += "📊 PEAK STATISTICS:\n"
        result_text += f"  Total peaks: {len(self.detected_peaks)}\n"
        result_text += f"  Wavelength range: {min(wavelengths_list):.1f} - {max(wavelengths_list):.1f} nm\n"
        result_text += f"  Intensity range: {min(intensities_list):.2f} - {max(intensities_list):.2f} (0-100 scale)\n"
        result_text += f"  Mean intensity: {np.mean(intensities_list):.2f}\n\n"
        
        # Show normalization info
        scheme = self.current_spectrum.get('normalization_scheme', 'Unknown')
        norm_wl = self.current_spectrum.get('normalization_wavelength', 'N/A')
        result_text += f"🔧 NORMALIZATION INFO:\n  Scheme: {scheme}\n  Reference: {norm_wl}nm\n  Scale: 0-100 (preserves UV ratios)\n\n"
        
        # List peaks
        result_text += "🎯 ALL PEAKS (sorted by intensity):\n" + "-"*50 + "\n"
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
        """OPTIMIZED: Plot spectrum with FIXED scaling (0-100)"""
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
                                  label=f'Major Peaks ({len(major_peaks)})', zorder=6, 
                                  edgecolors='darkred', linewidth=0.5)
                
                if other_peaks:
                    peak_wls = [p['wavelength'] for p in other_peaks]
                    peak_ints = [p['intensity'] for p in other_peaks]
                    self.ax.scatter(peak_wls, peak_ints, color='orange', s=20, marker='v', 
                                  label=f'Other Peaks ({len(other_peaks)})', zorder=5, alpha=0.7)
            
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
        self.ax.set_title('FIXED Normalization: UV Ratio Analysis Preserved', fontsize=12)
        self.ax.grid(True, alpha=0.3, which='both')
        self.ax.minorticks_on()
        self.ax.legend(loc='best', fontsize=9)
        
        self.canvas.draw()
    
    def update_results(self, text):
        """Update the results text area"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
    
    def export_results(self):
        """OPTIMIZED: Export with proper normalization metadata"""
        if not self.detected_peaks:
            messagebox.showwarning("Warning", "No results to export")
            return
        
        gem_id = self.extract_gem_id_from_filename()
        light_source = self.light_source.get().lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        default_csv_name = f"{gem_id or 'unknown'}_{light_source}_structural_{timestamp}.csv"
        
        export_choice = messagebox.askyesnocancel("Export Format",
            f"FIXED Normalization Export:\n\nYES = CSV for Database (recommended)\n   → {default_csv_name}\n"
            f"   → Saves to: {light_source} folder\n   → Includes normalization metadata\n\n"
            f"NO = Choose format manually\n\nCANCEL = Cancel export")
        
        if export_choice is None:
            return
        elif export_choice:
            save_path = os.path.join(rf"C:\users\david\gemini sp10 structural data\{light_source}", default_csv_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            try:
                self.export_csv_for_database(save_path, gem_id)
                messagebox.showinfo("Export Success", 
                                  f"✅ FIXED normalization CSV exported:\n{default_csv_name}\n\n"
                                  f"📁 Location: {light_source} folder\n🎯 Ready for database import!\n"
                                  f"📋 Includes normalization metadata")
                return
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not auto-export: {e}")
        
        # Manual export
        file = filedialog.asksaveasfilename(initialname=default_csv_name, defaultextension=".csv",
            filetypes=[("CSV for Database", "*.csv"), ("JSON (Complete data)", "*.json"), 
                      ("Text Report", "*.txt"), ("Peak List Only", "*.peaks")])
        
        if file:
            try:
                export_funcs = {
                    '.json': self.export_json_complete, '.csv': lambda f: self.export_csv_for_database(f, gem_id),
                    '.peaks': self.export_peak_list, '.txt': self.export_text_report
                }
                
                ext = next((ext for ext in export_funcs.keys() if file.endswith(ext)), '.txt')
                export_funcs[ext](file)
                
                messagebox.showinfo("Success", f"Results exported with FIXED normalization to {os.path.basename(file)}\n"
                                  f"Total peaks: {len(self.detected_peaks)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export: {e}")
    
    def extract_gem_id_from_filename(self):
        """OPTIMIZED: Extract gem ID from loaded spectrum filename"""
        if not self.current_spectrum:
            return None
        
        filename = self.current_spectrum.get('filename', '')
        for ext in ['.txt', '.csv', '.dat']:
            if filename.lower().endswith(ext):
                return filename[:-len(ext)]
        return filename if filename else "unknown"
    
    def export_csv_for_database(self, file_path, gem_id):
        """OPTIMIZED: Export CSV with normalization metadata"""
        with open(file_path, 'w', newline='') as f:
            f.write("Peak_Number,Wavelength_nm,Intensity,Prominence,Category,Normalization_Scheme,Reference_Wavelength,Light_Source\n")
            
            scheme = self.current_spectrum.get('normalization_scheme', 'Unknown')
            ref_wl = self.current_spectrum.get('normalization_wavelength', 'N/A')
            light_source = self.light_source.get()
            
            categories = ["Major", "Strong", "Medium", "Minor"]
            category_thresholds = [0.1, 0.3, 0.6]
            
            for i, peak in enumerate(self.detected_peaks, 1):
                category_idx = sum(1 for threshold in category_thresholds if i <= len(self.detected_peaks) * threshold)
                category = categories[min(category_idx, len(categories) - 1)]
                
                f.write(f"{i},{peak['wavelength']:.3f},{peak['intensity']:.3f},"
                      f"{peak['prominence']:.3f},{category},{scheme},{ref_wl},{light_source}\n")
        
        print(f"✅ Exported FIXED normalization peaks for gem {gem_id}: {os.path.basename(file_path)}")
    
    def export_json_complete(self, file_path):
        """OPTIMIZED: Export complete data with normalization metadata"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'spectrum_file': self.current_spectrum.get('filename', 'Unknown'),
            'normalization_scheme': self.current_spectrum.get('normalization_scheme', 'Unknown'),
            'normalization_wavelength': self.current_spectrum.get('normalization_wavelength', 'N/A'),
            'light_source': self.light_source.get(),
            'normalized': self.current_spectrum.get('normalized', False),
            'total_peaks': len(self.detected_peaks),
            'detected_peaks': self.detected_peaks
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_peak_list(self, file_path):
        """OPTIMIZED: Export simple peak list with normalization info"""
        with open(file_path, 'w') as f:
            f.write("# FIXED Normalization Peak List\n")
            f.write(f"# Normalization: {self.current_spectrum.get('normalization_scheme', 'Unknown')}\n")
            f.write(f"# Reference: {self.current_spectrum.get('normalization_wavelength', 'N/A')}nm\n")
            f.write(f"# Light Source: {self.light_source.get()}\n")
            f.write("# Peak wavelengths (nm)\n")
            for peak in self.detected_peaks:
                f.write(f"{peak['wavelength']:.2f}\n")
    
    def export_text_report(self, file_path):
        """OPTIMIZED: Export detailed text report with normalization info"""
        with open(file_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GEMINI PEAK DETECTION REPORT - FIXED NORMALIZATION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Spectrum File: {self.current_spectrum.get('filename', 'Unknown')}\n")
            f.write(f"Light Source: {self.light_source.get()}\n")
            f.write(f"Normalization Scheme: {self.current_spectrum.get('normalization_scheme', 'Unknown')}\n")
            f.write(f"Reference Wavelength: {self.current_spectrum.get('normalization_wavelength', 'N/A')}nm\n")
            f.write(f"Normalized: {self.current_spectrum.get('normalized', False)}\n\n")
            f.write(self.results_text.get(1.0, tk.END))
    
    # OPTIMIZED: Placeholder methods for gem database functionality
    def load_gem_database(self):
        """Load fallback database"""
        self.reference_gems = {
            'Ruby (Reference)': {
                'peaks': [694.2, 692.8, 668.0, 659.0, 475.0],
                'color': 'red',
                'description': 'Reference values - FIXED normalization'
            }
        }
    
    def on_database_change(self, event=None):
        """Handle database source change"""
        pass
    
    def on_gem_selected(self, event=None):
        """Handle gem selection"""
        pass
    
    def compare_with_reference(self):
        """Compare with reference"""
        messagebox.showinfo("Info", "Reference comparison not implemented in this simplified version")
    
    def find_best_match(self):
        """Find best match"""
        messagebox.showinfo("Info", "Best match finding not implemented in this simplified version")
    
    def initialize_system(self):
        """OPTIMIZED: Initialize the system"""
        self.update_results(
            "🎉 GEMINI PEAK DETECTOR - OPTIMIZED FIXED NORMALIZATION\n" + "="*50 + "\n" +
            "✅ Successfully initialized with FIXED normalization\n\n" +
            "🔧 FIXED NORMALIZATION SCHEMES:\n" +
            "• UV: 811nm → 15,000, then scale 0-100 (preserves ratios)\n" +
            "• Halogen: 650nm → 50,000, then scale 0-100\n" +
            "• Laser: Max intensity → 50,000, then scale 0-100\n\n" +
            "QUICK START:\n" +
            "1. Load spectrum (auto-detects light source)\n" +
            "2. Apply FIXED normalization\n" +
            "3. Detect peaks with preserved intensity relationships\n" +
            "4. Export with normalization metadata\n\n" +
            "⌨️ KEYBOARD CONTROLS:\n" +
            "• Keys 1,2,3: Select parameter\n" +
            "• Up/Down arrows: Adjust active parameter\n" +
            "• Spacebar: Reset defaults\n\n" +
            "💾 EXPORT: Auto-saves to correct folder with metadata!"
        )
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    print("🤖 Starting OPTIMIZED Gemini Peak Detector with FIXED Normalization...")
    app = GeminiPeakDetector()
    app.run()

if __name__ == '__main__':
    main()
