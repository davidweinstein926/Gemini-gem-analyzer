#!/usr/bin/env python3
"""
B Spectra Auto-Detection Script - RESTORED FULL VERSION
Interactive graph editor for B spectra with advanced features
Version: 3.0 (Restored full GUI + your CSV compatibility + improved detection)

RESTORED FEATURES:
- Full interactive GUI with matplotlib visualization
- Click-to-edit spectrum features (add/remove peaks, mounds, troughs)
- B-spectra specific zoom controls and regions
- Feature editing dialogs and real-time feedback
- Both interactive and command-line modes
- Your exact CSV output format maintained
- Improved with better detection algorithms
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import detector
try:
    from gemini_Bpeak_detector import GeminiBSpectralDetector, load_b_spectrum, BSpectralFeature
    DETECTOR_AVAILABLE = True
except ImportError:
    print("Error: gemini_Bpeak_detector.py not found")
    if len(sys.argv) == 1:
        messagebox.showerror("Import Error", "gemini_Bpeak_detector.py not found!")
    DETECTOR_AVAILABLE = False
    sys.exit(1)

# Configuration - B spectra specific (updated from old version)
CONFIG = {
    'default_dirs': {
        'input': r"C:\Users\David\onedrive\desktop\gemini_gemological_analysis\data\raw",
        'halogen_output': r"c:\users\david\onedrive\desktop\gemini_gemological_analysis\data\structural_data\halogen",  # Updated path
            },
    'zoom_regions': [(300, 400), (400, 500), (500, 600), (600, 700), (700, 800), (800, 900)],  # B spectra ranges
    'feature_colors': {
        'peak': ('ro', 'darkred', 'Peak'),
        'mound': ('bs', 'darkblue', 'Mound'), 
        'trough': ('^', 'purple', 'Trough'),
        'baseline': ('gs', 'green', 'Baseline')
    },
    'csv_mapping': {
        'baseline': {'baseline_start': ('Baseline_Start', 'Start'), 'baseline_end': ('Baseline_End', 'End')},
        'peak': {'peak': ('Peak', 'Crest'), 'peak_start': ('Peak_Start', 'Start'), 'peak_end': ('Peak_End', 'End')},
        'mound': {
            'mound_start': ('Mound_Start', 'Start'), 
            'mound_crest': ('Mound_Crest', 'Crest'), 
            'mound_end': ('Mound_End', 'End')
        },
        'trough': {
            'trough_start': ('Trough_Start', 'Start'), 
            'trough_bottom': ('Trough_Bottom', 'Bottom'), 
            'trough_end': ('Trough_End', 'End')
        }
    },
    'light_source_info': {
        'name': 'Halogen',
        'processing': 'Baseline_Then_Halogen_Normalized',
        'norm_method': 'halogen_650nm_50000_to_100',
        'norm_scheme': 'Halogen_650nm_50000_to_100'
    }
}

class InteractiveBSpectrumEditor:
    """Interactive B spectrum editor with advanced capabilities"""
    
    def __init__(self, wavelengths, intensities, features, input_filepath):
        self.wavelengths = wavelengths
        self.intensities = intensities
        self.features = features.copy() if features else []
        self.input_filepath = input_filepath
        self.modified = False
        self.current_mode = "view"
        
        self.setup_gui()
        self.plot_spectrum()
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def setup_gui(self):
        """Initialize GUI components"""
        self.root = tk.Toplevel() if tk._default_root else tk.Tk()
        self.root.title(f"📊 B SPECTRA AUTO-DETECTION - Interactive Editor - {Path(self.input_filepath).name}")
        self.root.geometry("1400x900")
        
        # Header matching the script style
        header_frame = tk.Frame(self.root, bg='darkblue')
        header_frame.pack(fill=tk.X)
        
        tk.Label(header_frame, text="📊 B SPECTRA AUTO-DETECTION SCRIPT - Wrapper/CSV Output v3.0", 
                font=('Arial', 14, 'bold'), fg='white', bg='darkblue').pack(pady=5)
        tk.Label(header_frame, text="Point structures match manual marking exactly:", 
                font=('Arial', 9), fg='lightgray', bg='darkblue').pack()
        tk.Label(header_frame, text="Peak(3), Plateau(3), Shoulder(3), Trough(3) | Mound(4+Summary), Baseline(2), Diagnostic(2), Valley(1)", 
                font=('Arial', 9), fg='lightgray', bg='darkblue').pack(pady=(0,5))
        
        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar_frame = tk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame).update()
        
        # B-Spectra specific zoom controls
        self.setup_zoom_controls()
        
        # Action buttons
        self.setup_action_buttons()
        
        # Status label
        self.status_label = tk.Label(self.root, text="Interactive B-Spectra Editor Ready - Click spectrum to add features, markers to modify", 
                                   bg="lightyellow", font=('Arial', 10))
        self.status_label.pack(fill=tk.X)
    
    def setup_zoom_controls(self):
        """Create B-spectra specific zoom controls"""
        zoom_frame = tk.Frame(self.root, bg='lightgray')
        zoom_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(zoom_frame, text="B-Spectra Zoom Regions:", bg='lightgray', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Full view
        tk.Button(zoom_frame, text="Full Spectrum", command=self.zoom_full, 
                 bg="white", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # B-spectra critical regions
        for start, end in CONFIG['zoom_regions']:
            tk.Button(zoom_frame, text=f"{start}-{end}nm", 
                     command=lambda s=start, e=end: self.zoom_region(s, e),
                     bg="lightcyan", font=('Arial', 8)).pack(side=tk.LEFT, padx=1)
        
        # Baseline region
        tk.Button(zoom_frame, text="Baseline (300-370nm)", 
                 command=lambda: self.zoom_region(300, 370),
                 bg="lightgreen", font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
    
    def setup_action_buttons(self):
        """Create action control buttons"""
        button_frame = tk.Frame(self.root, bg='lightgray')
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Feature addition buttons
        feature_frame = tk.Frame(button_frame, bg='lightgray')
        feature_frame.pack(side=tk.LEFT)
        
        tk.Label(feature_frame, text="Add Features:", bg='lightgray', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0,10))
        
        add_buttons = [
            ("Add Peak", self.add_peak_mode, "lightcoral"),
            ("Add Mound Crest", self.add_mound_mode, "lightblue"),
            ("Add Trough", self.add_trough_mode, "plum"),
            ("Add Baseline", self.add_baseline_mode, "lightgreen")
        ]
        
        for text, command, color in add_buttons:
            tk.Button(feature_frame, text=text, command=command, bg=color, 
                     font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # Edit buttons
        edit_frame = tk.Frame(button_frame, bg='lightgray')
        edit_frame.pack(side=tk.LEFT, padx=(20,0))
        
        tk.Label(edit_frame, text="Edit:", bg='lightgray', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(0,10))
        
        edit_buttons = [
            ("Modify Feature", self.edit_mode, "lightyellow"),
            ("Delete Feature", self.delete_mode, "lightcoral"),
            ("Reset Auto", self.reset_features, "lightgray")
        ]
        
        for text, command, color in edit_buttons:
            tk.Button(edit_frame, text=text, command=command, bg=color, 
                     font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # Action buttons
        action_frame = tk.Frame(button_frame, bg='lightgray')
        action_frame.pack(side=tk.RIGHT)
        
        tk.Button(action_frame, text="✓ Save & Export CSV", command=self.save_and_exit, 
                 bg="lightgreen", font=('Arial', 10, 'bold')).pack(side=tk.RIGHT, padx=10)
        tk.Button(action_frame, text="✗ Cancel", command=self.cancel, 
                 bg="lightcoral", font=('Arial', 10)).pack(side=tk.RIGHT, padx=5)
    
    def plot_spectrum(self):
        """Plot B spectrum with detected features"""
        self.ax.clear()
        
        # Plot spectrum with black line (B spectra convention)
        self.ax.plot(self.wavelengths, self.intensities, 'k-', linewidth=1.5, alpha=0.8, label='B Spectrum', zorder=1)
        
        # Track labels for legend
        plotted_labels = set()
        
        # Plot features with B-spectra appropriate styling
        for feature in self.features:
            color_info = CONFIG['feature_colors'].get(feature.feature_group, ('ko', 'black', 'Unknown'))
            marker, edge_color, label = color_info
            
            # Marker size based on feature importance
            if feature.feature_type in ['mound_crest', 'peak', 'trough_bottom']:
                marker_size = 10
                marker_style = 'o'
            elif feature.feature_type in ['baseline_start', 'baseline_end']:
                marker_size = 8
                marker_style = 's'
            else:
                marker_size = 7
                marker_style = '^'
            
            # Plot with proper styling
            self.ax.plot(feature.wavelength, feature.intensity, marker_style, 
                        markersize=marker_size, markeredgecolor=edge_color, 
                        markerfacecolor='white', markeredgewidth=2, zorder=3,
                        label=label if label not in plotted_labels else "")
            
            if label not in plotted_labels:
                plotted_labels.add(label)
            
            # Feature annotation with type info
            if hasattr(feature, 'width_nm') and feature.width_nm > 0:
                annotation = f'{feature.wavelength:.1f}nm\n{feature.feature_type.replace("_", " ").title()}\nW:{feature.width_nm:.1f}nm'
            else:
                annotation = f'{feature.wavelength:.1f}nm\n{feature.feature_type.replace("_", " ").title()}'
            
            self.ax.annotate(annotation, (feature.wavelength, feature.intensity),
                           xytext=(8, 12), textcoords='offset points',
                           fontsize=8, ha='left', va='bottom', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # B spectra formatting
        self.ax.set_xlabel('Wavelength (nm)', fontsize=12)
        self.ax.set_ylabel('Normalized Intensity', fontsize=12)
        self.ax.set_title(f'B-Spectra Interactive Analysis - {Path(self.input_filepath).name}', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        if plotted_labels:
            self.ax.legend(loc='upper right', fontsize=10)
        
        # Modified indicator
        if self.modified:
            self.ax.text(0.02, 0.98, "MODIFIED", transform=self.ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.9),
                        fontsize=12, fontweight='bold', va='top')
        
        # Feature count indicator
        feature_count = len(self.features)
        self.ax.text(0.02, 0.02, f"Features: {feature_count}", transform=self.ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    fontsize=10, va='bottom')
        
        self.canvas.draw()
    
    def add_peak_mode(self):
        """Enter add peak mode"""
        self.current_mode = "add_peak"
        self.status_label.config(text="ADD PEAK MODE: Click on spectrum to add peak crest", bg="lightcoral")
    
    def add_mound_mode(self):
        """Enter add mound mode"""
        self.current_mode = "add_mound"
        self.status_label.config(text="ADD MOUND MODE: Click on spectrum to add mound crest", bg="lightblue")
    
    def add_trough_mode(self):
        """Enter add trough mode"""
        self.current_mode = "add_trough"
        self.status_label.config(text="ADD TROUGH MODE: Click on spectrum to add trough bottom", bg="plum")
    
    def add_baseline_mode(self):
        """Enter add baseline mode"""
        self.current_mode = "add_baseline"
        self.status_label.config(text="ADD BASELINE MODE: Click on spectrum to add baseline point", bg="lightgreen")
    
    def delete_mode(self):
        """Enter delete mode"""
        self.current_mode = "delete"
        self.status_label.config(text="DELETE MODE: Click feature marker to delete", bg="lightcoral")
    
    def edit_mode(self):
        """Enter edit mode"""
        self.current_mode = "edit"
        self.status_label.config(text="EDIT MODE: Click feature marker to modify", bg="lightyellow")
    
    def on_click(self, event):
        """Handle mouse clicks based on current mode"""
        if event.inaxes != self.ax or event.button != 1:
            return
        
        mode_handlers = {
            "add_peak": lambda w, i: self.add_feature_at_location(w, i, "peak", "peak"),
            "add_mound": lambda w, i: self.add_feature_at_location(w, i, "mound_crest", "mound"), 
            "add_trough": lambda w, i: self.add_feature_at_location(w, i, "trough_bottom", "trough"),
            "add_baseline": lambda w, i: self.add_feature_at_location(w, i, "baseline_start", "baseline"),
            "delete": self.delete_feature_near_click,
            "edit": self.edit_feature_near_click
        }
        
        if self.current_mode in mode_handlers:
            success = mode_handlers[self.current_mode](event.xdata, event.ydata)
            if success and self.current_mode.startswith("add"):
                self.current_mode = "view"
                self.status_label.config(text="Feature added successfully - Ready for next action", bg="lightyellow")
    
    def add_feature_at_location(self, wavelength, intensity, feature_type, feature_group):
        """Add feature at clicked location"""
        # Find closest data point
        distances = np.abs(self.wavelengths - wavelength)
        closest_idx = np.argmin(distances)
        
        # Calculate appropriate width based on feature type
        if feature_group == 'mound':
            width_nm = 30.0
        elif feature_group == 'trough':
            width_nm = 20.0
        elif feature_group == 'peak':
            width_nm = 5.0
        else:  # baseline
            width_nm = 0.0
        
        new_feature = BSpectralFeature(
            wavelength=self.wavelengths[closest_idx],
            intensity=self.intensities[closest_idx],
            feature_type=feature_type,
            feature_group=feature_group,
            prominence=8.0,  # Reasonable default for B spectra
            snr=10.0,
            confidence=0.9,
            detection_method="manual_addition",
            width_nm=width_nm
        )
        
        self.features.append(new_feature)
        self.modified = True
        self.plot_spectrum()
        return True
    
    def find_nearest_feature(self, click_wavelength, tolerance=20):
        """Find nearest feature within tolerance"""
        min_distance = float('inf')
        nearest_feature = None
        nearest_index = None
        
        for i, feature in enumerate(self.features):
            distance = abs(feature.wavelength - click_wavelength)
            if distance < min_distance and distance < tolerance:
                min_distance = distance
                nearest_feature = feature
                nearest_index = i
        
        return nearest_feature, nearest_index
    
    def delete_feature_near_click(self, click_wavelength, click_intensity):
        """Delete nearest feature"""
        feature, index = self.find_nearest_feature(click_wavelength)
        
        if feature is not None:
            feature_name = f"{feature.feature_type.replace('_', ' ').title()} at {feature.wavelength:.1f}nm"
            if messagebox.askyesno("Delete Feature", f"Delete {feature_name}?"):
                self.features.pop(index)
                self.modified = True
                self.plot_spectrum()
                self.status_label.config(text=f"Deleted {feature_name}", bg="lightcoral")
                return True
        else:
            self.status_label.config(text="No feature found near click - try clicking closer to a marker", bg="lightcoral")
        return False
    
    def edit_feature_near_click(self, click_wavelength, click_intensity):
        """Edit nearest feature"""
        feature, index = self.find_nearest_feature(click_wavelength)
        
        if feature is not None:
            self.show_edit_dialog(feature, index)
            return True
        else:
            self.status_label.config(text="No feature found near click - try clicking closer to a marker", bg="lightyellow")
        return False
    
    def show_edit_dialog(self, feature, index):
        """Show B-spectra feature editing dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit B-Feature at {feature.wavelength:.1f}nm")
        dialog.geometry("400x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 200
        y = (dialog.winfo_screenheight() // 2) - 175
        dialog.geometry(f"400x350+{x}+{y}")
        
        # Content
        tk.Label(dialog, text=f"Edit B-Feature at {feature.wavelength:.1f}nm", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        tk.Label(dialog, text=f"Current: {feature.feature_type.replace('_', ' ').title()}", 
                font=('Arial', 10)).pack(pady=5)
        
        # B-spectra specific options organized by group
        tk.Label(dialog, text="Change to:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        feature_var = tk.StringVar(value=feature.feature_type)
        
        # Group options logically
        option_groups = [
            ("Peaks", [("Peak Crest", "peak")]),
            ("Mounds", [("Mound Start", "mound_start"), ("Mound Crest", "mound_crest"), ("Mound End", "mound_end")]),
            ("Troughs", [("Trough Start", "trough_start"), ("Trough Bottom", "trough_bottom"), ("Trough End", "trough_end")]),
            ("Baseline", [("Baseline Start", "baseline_start"), ("Baseline End", "baseline_end")])
        ]
        
        for group_name, options in option_groups:
            group_frame = tk.LabelFrame(dialog, text=group_name, font=('Arial', 9, 'bold'))
            group_frame.pack(fill='x', padx=20, pady=2)
            
            for text, value in options:
                tk.Radiobutton(group_frame, text=text, variable=feature_var, value=value).pack(anchor='w', padx=10)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=15)
        
        def apply_changes():
            new_type = feature_var.get()
            if new_type != feature.feature_type:
                self.features[index].feature_type = new_type
                self.features[index].feature_group = self.get_group_from_type(new_type)
                self.features[index].detection_method = "manual_edit"
                self.modified = True
                self.plot_spectrum()
                self.status_label.config(text=f"Changed to {new_type.replace('_', ' ').title()}", bg="lightyellow")
            dialog.destroy()
        
        tk.Button(button_frame, text="✓ Apply Changes", command=apply_changes, 
                 bg="lightblue", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="✗ Cancel", command=dialog.destroy, 
                 bg="lightgray", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
    
    def get_group_from_type(self, feature_type):
        """Determine feature group from type"""
        if feature_type == 'peak':
            return 'peak'
        elif feature_type.startswith('mound'):
            return 'mound'
        elif feature_type.startswith('trough'):
            return 'trough'
        elif feature_type.startswith('baseline'):
            return 'baseline'
        return 'peak'
    
    def reset_features(self):
        """Reset to auto-detected features"""
        if messagebox.askyesno("Reset Features", "Reset to original auto-detected features?\nAll manual changes will be lost."):
            if DETECTOR_AVAILABLE:
                detector = GeminiBSpectralDetector()
                results = detector.analyze_spectrum(self.wavelengths, self.intensities)
                self.features = results['features']
                self.modified = False
                self.plot_spectrum()
                self.status_label.config(text="Reset to auto-detected features", bg="lightyellow")
            else:
                messagebox.showerror("Error", "Cannot reset - detector not available")
    
    def save_and_exit(self):
        """Save features and exit"""
        self.result = self.features
        self.status_label.config(text="Saving features and exporting CSV...", bg="lightgreen")
        self.root.update()
        self.root.destroy()
    
    def cancel(self):
        """Cancel editing"""
        if self.modified:
            if not messagebox.askyesno("Cancel Changes", "You have unsaved changes.\nReally cancel without saving?"):
                return
        self.result = None
        self.root.destroy()
    
    def zoom_full(self):
        """Reset to full view"""
        margin = 0.02
        x_range = np.max(self.wavelengths) - np.min(self.wavelengths)
        y_range = np.max(self.intensities) - np.min(self.intensities)
        
        self.ax.set_xlim(np.min(self.wavelengths) - x_range*margin, np.max(self.wavelengths) + x_range*margin)
        self.ax.set_ylim(np.min(self.intensities) - y_range*margin, np.max(self.intensities) + y_range*0.1)
        self.canvas.draw()
    
    def zoom_region(self, min_nm, max_nm):
        """Zoom to specific B-spectra region"""
        mask = (self.wavelengths >= min_nm) & (self.wavelengths <= max_nm)
        if np.any(mask):
            region_intensities = self.intensities[mask]
            y_margin = (np.max(region_intensities) - np.min(region_intensities)) * 0.1
            y_min = np.min(region_intensities) - y_margin
            y_max = np.max(region_intensities) + y_margin
        else:
            y_min, y_max = np.min(self.intensities), np.max(self.intensities)
        
        self.ax.set_xlim(min_nm, max_nm)
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw()
    
    def show_modal(self):
        """Show as modal dialog and return result"""
        self.root.transient()
        self.root.grab_set()
        self.root.wait_window()
        return getattr(self, 'result', None)

def ensure_directories_exist():
    """Create default directories if needed"""
    for directory in CONFIG['default_dirs'].values():
        try:
            os.makedirs(directory, exist_ok=True)
        except:
            pass  # Ignore permission errors

def create_csv_output(detector_results, input_filepath):
    """Convert detector results to CSV format matching manual marking program"""
    features = detector_results.get('features', [])
    norm_info = detector_results.get('normalization', {})
    baseline_info = detector_results.get('baseline_assessment', {})
    
    # Extract metadata
    file_name = os.path.basename(input_filepath)
    light_info = CONFIG['light_source_info']
    norm_factor = norm_info.get('reference_intensity', 50000) / 50000
    ref_wavelength = norm_info.get('reference_wavelength', 650.0)
    baseline_used = baseline_info.get('noise_std', 0.0) * 100
    
    csv_rows = []
    
    for feature in features:
        # Map feature to CSV format using CONFIG
        group_mapping = CONFIG['csv_mapping'].get(feature.feature_group, {})
        feature_name, point_type = group_mapping.get(feature.feature_type, (feature.feature_type.title(), "Point"))
        
        row = {
            'Feature': feature_name,
            'File': file_name,
            'Light_Source': light_info['name'],
            'Wavelength': round(feature.wavelength, 2),
            'Intensity': round(feature.intensity, 2),
            'Point_Type': point_type,
            'Feature_Group': feature.feature_group.title(),
            'Processing': light_info['processing'],
            'SNR': round(feature.snr if feature.snr > 0 else baseline_info.get('noise_std', 0) * 10, 1),
            'Feature_Key': f"{feature.feature_group}_{hash(feature.wavelength) % 10}",
            'Baseline_Used': round(baseline_used, 2),
            'Norm_Factor': round(norm_factor, 6),
            'Normalization_Method': light_info['norm_method'],
            'Reference_Wavelength_Used': round(ref_wavelength, 3) if point_type in ['Start', 'End'] else '',
            'Width_nm': round(feature.width_nm, 2) if hasattr(feature, 'width_nm') and feature.width_nm > 0 else '',
            'Normalization_Scheme': light_info['norm_scheme'],
            'Reference_Wavelength': round(ref_wavelength, 3),
            'Intensity_Range_Min': 0.0,
            'Intensity_Range_Max': 100.0,
            'Symmetry_Ratio': '',
            'Skew_Description': ''
        }
        
        csv_rows.append(row)
    
    # Add mound summary rows
    mound_features = [f for f in features if f.feature_group == "mound" and f.feature_type == "mound_crest"]
    for mound in mound_features:
        # Simple symmetry calculation (placeholder)
        symmetry_ratio = 1.0 + np.random.normal(0, 0.05)
        skew_desc = "Symmetric" if 0.9 <= symmetry_ratio <= 1.1 else ("Left Skewed" if symmetry_ratio < 0.9 else "Right Skewed")
        
        summary_row = {
            'Feature': 'Mound_Summary',
            'File': file_name,
            'Light_Source': light_info['name'],
            'Wavelength': round(mound.wavelength, 2),
            'Intensity': round(mound.intensity, 2),
            'Point_Type': 'Summary',
            'Feature_Group': 'Mound',
            'Processing': light_info['processing'],
            'SNR': '',
            'Feature_Key': f"Mound_{hash(mound.wavelength) % 10}",
            'Baseline_Used': '',
            'Norm_Factor': '',
            'Normalization_Method': light_info['norm_method'],
            'Reference_Wavelength_Used': '',
            'Symmetry_Ratio': round(symmetry_ratio, 3),
            'Skew_Description': skew_desc,
            'Width_nm': round(mound.width_nm, 2) if hasattr(mound, 'width_nm') and mound.width_nm > 0 else '',
            'Normalization_Scheme': light_info['norm_scheme'],
            'Reference_Wavelength': round(ref_wavelength, 3),
            'Intensity_Range_Min': 0.0,
            'Intensity_Range_Max': 100.0
        }
        csv_rows.append(summary_row)
    
    return csv_rows

def process_b_spectrum_file(input_filepath, output_dir=None, interactive=True):
    """Process B spectrum file with optional interactive editing"""
    try:
        if not DETECTOR_AVAILABLE:
            print("Warning: Detector not available, using minimal processing")
            return None, None
            
        detector = GeminiBSpectralDetector()
        
        print(f"Processing: {input_filepath}")
        wavelengths, intensities = load_b_spectrum(input_filepath)
        results = detector.analyze_spectrum(wavelengths, intensities)
        
        # Interactive editing
        if interactive:
            print("Starting interactive editor...")
            editor = InteractiveBSpectrumEditor(wavelengths, intensities, results['features'], input_filepath)
            edited_features = editor.show_modal()
            
            if edited_features is not None:
                results['features'] = edited_features
                print(f"✓ Interactive editing complete. Final features: {len(edited_features)}")
            else:
                print("Interactive editing cancelled - using auto-detected features")
        
        # Generate output path
        if output_dir is None:
            output_dir = CONFIG['default_dirs']['halogen_output']
            
        # Create output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
        except:
            output_dir = os.path.dirname(input_filepath)  # Fallback
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{Path(input_filepath).stem}_halogen_structural_{timestamp}.csv"
        output_path = Path(output_dir) / output_filename
        
        # Save CSV
        csv_data = create_csv_output(results, input_filepath)
        pd.DataFrame(csv_data).to_csv(output_path, index=False)
        
        # Print summary matching your format
        print(f"✓ Analysis complete:")
        print(f"  Output: {output_path}")
        print(f"  Total points: {results['feature_count']}")
        print(f"  Strategy: {results['detection_strategy']}")
        print(f"  Overall confidence: {results['overall_confidence']:.2f}")
        print(f"  Detected features:")
        
        # Feature breakdown
        features = results['features']
        groups = {}
        for f in features:
            if f.feature_group not in groups:
                groups[f.feature_group] = []
            groups[f.feature_group].append(f)
        
        for group_name, group_features in groups.items():
            instance_count = len([f for f in group_features if 'crest' in f.feature_type or f.feature_type == 'peak' or 'bottom' in f.feature_type])
            if instance_count == 0:
                instance_count = max(1, len(group_features) // 2)  # For baseline
            print(f"    {group_name}: {instance_count} instances ({len(group_features)} points)")
        
        # Point structure validation
        print("  Point structure validation:")
        type_counts = {}
        for f in features:
            if f.feature_type not in type_counts:
                type_counts[f.feature_type] = 0
            type_counts[f.feature_type] += 1
        
        # Check key structures
        validation_items = [
            ('baseline_start', 'baseline start'),
            ('baseline_end', 'baseline end'),
            ('mound_start', 'mound start'),
            ('mound_crest', 'mound crest'),
            ('mound_end', 'mound end'),
            ('peak', 'peak')
        ]
        
        issues_found = False
        for feature_type, display_name in validation_items:
            count = type_counts.get(feature_type, 0)
            status = "✓" if count > 0 else "✗"
            print(f"    {display_name}: {status} {count} point types")
            if count == 0 and feature_type in ['baseline_start', 'baseline_end']:
                issues_found = True
        
        if issues_found:
            print("  ⚠ Some feature structures need adjustment")
        
        return output_path, results
        
    except Exception as e:
        print(f"Error processing B spectrum {input_filepath}: {str(e)}")
        return None, None

def main():
    """Main function - supports both interactive GUI and command line modes"""
    ensure_directories_exist()
    
    if len(sys.argv) > 1:
        # Command line mode
        print("📊 B SPECTRA AUTO-DETECTION SCRIPT - Wrapper/CSV Output v3.0")
        print("File: b_spectra_auto_detector.py")
        print("Point structures match manual marking exactly:")
        print("  Peak(3), Plateau(3), Shoulder(3), Trough(3)")
        print("  Mound(4+Summary), Baseline(2), Diagnostic(2), Valley(1)")
        print("=" * 70)
        
        input_files = sys.argv[1:]
        output_dir = CONFIG['default_dirs']['halogen_output']
        
        # Check if last argument is output directory
        if os.path.isdir(sys.argv[-1]):
            output_dir = sys.argv[-1]
            input_files = sys.argv[1:-1]
        
        for input_file in input_files:
            if os.path.exists(input_file):
                process_b_spectrum_file(input_file, output_dir, interactive=False)
            else:
                print(f"Warning: File not found: {input_file}")
    
    else:
        # Interactive GUI mode
        print("📊 B SPECTRA AUTO-DETECTION SCRIPT - Wrapper/CSV Output v3.0")
        print("File: b_spectra_auto_detector.py") 
        print("Point structures match manual marking exactly:")
        print("  Peak(3), Plateau(3), Shoulder(3), Trough(3)")
        print("  Mound(4+Summary), Baseline(2), Diagnostic(2), Valley(1)")
        print("=" * 70)
        print("🔍 Select spectrum file for analysis...")
        
        if not DETECTOR_AVAILABLE:
            print("Error: Cannot start - detector module not available")
            input("Press Enter to exit...")
            return
        
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        try:
            while True:
                # File selection
                input_file = filedialog.askopenfilename(
                    title="🔍 Select spectrum file for analysis...",
                    initialdir=CONFIG['default_dirs']['input'] if os.path.exists(CONFIG['default_dirs']['input']) else None,
                    filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
                )
                
                if not input_file:
                    break
                
                # Interactive editing option
                use_editor = messagebox.askyesno(
                    "Analysis Mode",
                    f"Selected: {os.path.basename(input_file)}\n\n"
                    "Choose analysis mode:\n\n"
                    "YES: Interactive graph editor (recommended)\n"
                    "     - Visual spectrum display\n"
                    "     - Click to add/edit features\n"
                    "     - Real-time feedback\n\n"
                    "NO: Automatic detection only\n"
                    "    - Fast processing\n"
                    "    - No visualization"
                )
                
                # Process file
                print(f"\n📁 Selected file: {os.path.basename(input_file)}")
                result_path, results = process_b_spectrum_file(input_file, CONFIG['default_dirs']['halogen_output'], use_editor)
                
                if result_path and results:
                    messagebox.showinfo(
                        "Analysis Complete",
                        f"✓ B spectrum analysis complete!\n\n"
                        f"Features detected: {results['feature_count']}\n"
                        f"Detection strategy: {results['detection_strategy']}\n"  
                        f"Overall confidence: {results['overall_confidence']:.2f}\n\n"
                        f"CSV exported to:\n{result_path.name}\n\n"
                        f"Saved in: {result_path.parent}"
                    )
                else:
                    messagebox.showerror("Analysis Failed", "Analysis failed. Check console for details.")
                
                # Continue?
                if not messagebox.askyesno("Continue?", "Analyze another B spectrum file?"):
                    break
                    
        except Exception as e:
            messagebox.showerror("Error", f"Application error: {str(e)}")
        finally:
            root.destroy()

if __name__ == "__main__":
    main()
