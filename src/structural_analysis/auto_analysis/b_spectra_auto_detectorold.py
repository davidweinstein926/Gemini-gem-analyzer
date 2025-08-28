#!/usr/bin/env python3
"""
B Spectra Auto-Detection Script - ENHANCED
Interactive graph editor for B spectra with advanced features from L detector
Version: 2.0 (Enhanced with L detector capabilities + optimized)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import detector
try:
    from gemini_Bpeak_detector import GeminiBSpectralDetector, load_b_spectrum, BSpectralFeature
except ImportError:
    print("Error: gemini_Bpeak_detector.py not found")
    if len(sys.argv) == 1:
        messagebox.showerror("Import Error", "gemini_Bpeak_detector.py not found!")
    sys.exit(1)

# Configuration - B spectra specific
CONFIG = {
    'default_dirs': {
        'input': r"C:\Users\David\gemini sp10 raw",
        'halogen_output': r"C:\Users\David\gemini sp10 structural data\halogen",
        'laser_output': r"C:\Users\David\gemini sp10 structural data\laser"
    },
    'zoom_regions': [(350, 450), (450, 550), (550, 650), (650, 750), (750, 850)],  # B spectra ranges
    'feature_colors': {
        'peak': ('ro', 'darkred', 'Peak'),
        'mound': ('bs', 'darkblue', 'Mound'), 
        'trough': ('^', 'purple', 'Trough'),
        'baseline': ('gs', 'green', 'Baseline')
    },
    'csv_mapping': {
        'baseline': {'baseline_start': ('Baseline_Start', 'Start'), 'baseline_end': ('Baseline_End', 'End')},
        'peak': {'peak': ('Peak', 'Crest')},
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
        self.features = features.copy()
        self.input_filepath = input_filepath
        self.modified = False
        self.current_mode = "view"
        
        self.setup_gui()
        self.plot_spectrum()
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def setup_gui(self):
        """Initialize GUI components"""
        self.root = tk.Toplevel()
        self.root.title(f"B-Spectra Editor - {Path(self.input_filepath).name}")
        self.root.geometry("1200x800")
        
        # Matplotlib setup
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar
        toolbar_frame = tk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame).update()
        
        # Zoom controls
        self.setup_zoom_controls()
        
        # Action buttons
        self.setup_action_buttons()
        
        # Status label
        self.status_label = tk.Label(self.root, text="Click spectrum to add features, markers to delete", 
                                   bg="lightyellow", font=('Arial', 10))
        self.status_label.pack(fill=tk.X)
    
    def setup_zoom_controls(self):
        """Create B-spectra specific zoom controls"""
        zoom_frame = tk.Frame(self.root, bg='lightgray')
        zoom_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(zoom_frame, text="Quick Zoom:", bg='lightgray', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Full view
        tk.Button(zoom_frame, text="Full View", command=self.zoom_full, 
                 bg="white", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # B-spectra specific regions
        for start, end in CONFIG['zoom_regions']:
            tk.Button(zoom_frame, text=f"Zoom {start}-{end}nm", 
                     command=lambda s=start, e=end: self.zoom_region(s, e),
                     bg="lightcoral", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
    
    def setup_action_buttons(self):
        """Create action control buttons"""
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("Add Peak", self.add_peak_mode, "lightgreen"),
            ("Add Mound", self.add_mound_mode, "lightblue"),
            ("Add Trough", self.add_trough_mode, "plum"),
            ("Delete Feature", self.delete_mode, "lightcoral"),
            ("Edit Feature", self.edit_mode, "lightyellow"),
            ("Reset to Auto", self.reset_features, "lightgray"),
            ("Cancel", self.cancel, "lightgray"),
            ("Save & Exit", self.save_and_exit, "lightgreen")
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            side = tk.RIGHT if i >= len(buttons) - 2 else tk.LEFT
            padx = 10 if side == tk.RIGHT else 5
            weight = 'bold' if 'Save' in text else 'normal'
            
            tk.Button(button_frame, text=text, command=command, bg=color, 
                     font=('Arial', 10, weight)).pack(side=side, padx=padx)
    
    def plot_spectrum(self):
        """Plot B spectrum with detected features"""
        self.ax.clear()
        
        # Plot spectrum with black line (B spectra convention)
        self.ax.plot(self.wavelengths, self.intensities, 'k-', linewidth=1.2, alpha=0.8, label='B Spectrum')
        
        # Track labels
        plotted_labels = set()
        
        for feature in self.features:
            color_info = CONFIG['feature_colors'].get(feature.feature_group, ('ko', 'black', 'Unknown'))
            marker, edge_color, label = color_info
            
            # Plot with appropriate size for B spectra
            marker_size = 7 if feature.feature_group == 'mound' else 6 if feature.feature_group == 'baseline' else 5
            
            self.ax.plot(feature.wavelength, feature.intensity, marker, markersize=marker_size,
                        markeredgecolor=edge_color, markeredgewidth=1.5, 
                        label=label if label not in plotted_labels else "")
            
            if label not in plotted_labels:
                plotted_labels.add(label)
            
            # Feature annotation
            annotation = f'{feature.wavelength:.1f}nm'
            if hasattr(feature, 'width_nm') and feature.width_nm > 0:
                annotation += f'\n({feature.feature_type.replace("_", " ").title()})\nW:{feature.width_nm:.1f}nm'
            else:
                annotation += f'\n({feature.feature_type.replace("_", " ").title()})'
            
            self.ax.annotate(annotation, (feature.wavelength, feature.intensity),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=8, ha='left', va='bottom')
        
        # B spectra formatting
        self.ax.set_xlabel('Wavelength (nm)', fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=12)
        self.ax.set_title(f'B-Spectra Analysis - {Path(self.input_filepath).name}', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        
        # Modified indicator
        if self.modified:
            self.ax.text(0.02, 0.98, "MODIFIED", transform=self.ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8),
                        fontsize=10, fontweight='bold', va='top')
        
        self.canvas.draw()
    
    def add_peak_mode(self):
        """Enter add peak mode"""
        self.current_mode = "add_peak"
        self.status_label.config(text="ADD PEAK MODE: Click on spectrum to add peak", bg="lightgreen")
    
    def add_mound_mode(self):
        """Enter add mound mode"""
        self.current_mode = "add_mound"
        self.status_label.config(text="ADD MOUND MODE: Click on spectrum to add mound crest", bg="lightblue")
    
    def add_trough_mode(self):
        """Enter add trough mode"""
        self.current_mode = "add_trough"
        self.status_label.config(text="ADD TROUGH MODE: Click on spectrum to add trough", bg="plum")
    
    def delete_mode(self):
        """Enter delete mode"""
        self.current_mode = "delete"
        self.status_label.config(text="DELETE MODE: Click marker to delete", bg="lightcoral")
    
    def edit_mode(self):
        """Enter edit mode"""
        self.current_mode = "edit"
        self.status_label.config(text="EDIT MODE: Click marker to change type", bg="lightyellow")
    
    def on_click(self, event):
        """Handle mouse clicks based on current mode"""
        if event.inaxes != self.ax or event.button != 1:
            return
        
        actions = {
            "add_peak": lambda w, i: self.add_feature_at_location(w, i, "peak", "peak"),
            "add_mound": lambda w, i: self.add_feature_at_location(w, i, "mound_crest", "mound"), 
            "add_trough": lambda w, i: self.add_feature_at_location(w, i, "trough_bottom", "trough"),
            "delete": self.delete_feature_near_click,
            "edit": self.edit_feature_near_click
        }
        
        if self.current_mode in actions:
            success = actions[self.current_mode](event.xdata, event.ydata)
            if success and self.current_mode.startswith("add"):
                self.current_mode = "view"
                self.status_label.config(text="Feature added successfully", bg="lightyellow")
    
    def add_feature_at_location(self, wavelength, intensity, feature_type, feature_group):
        """Add feature at clicked location"""
        # Find closest data point
        distances = np.abs(self.wavelengths - wavelength)
        closest_idx = np.argmin(distances)
        
        # Calculate width for mounds and troughs
        width_nm = 20.0 if feature_group in ['mound', 'trough'] else 2.0
        
        new_feature = BSpectralFeature(
            wavelength=self.wavelengths[closest_idx],
            intensity=self.intensities[closest_idx],
            feature_type=feature_type,
            feature_group=feature_group,
            prominence=5.0,
            snr=8.0,
            confidence=0.9,
            detection_method="manual_addition",
            width_nm=width_nm
        )
        
        self.features.append(new_feature)
        self.modified = True
        self.plot_spectrum()
        return True
    
    def find_nearest_feature(self, click_wavelength, feature_groups=None):
        """Find nearest feature within tolerance"""
        if feature_groups is None:
            feature_groups = ["peak", "mound", "trough"]
        
        min_distance = float('inf')
        nearest_feature = None
        nearest_index = None
        tolerance = 15  # Larger tolerance for B spectra
        
        for i, feature in enumerate(self.features):
            if feature.feature_group in feature_groups:
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
            self.features.pop(index)
            self.modified = True
            self.plot_spectrum()
            return True
        
        self.status_label.config(text="No feature found near click", bg="lightcoral")
        return False
    
    def edit_feature_near_click(self, click_wavelength, click_intensity):
        """Edit nearest feature"""
        feature, index = self.find_nearest_feature(click_wavelength)
        
        if feature is not None:
            self.show_edit_dialog(feature, index)
            return True
        
        self.status_label.config(text="No feature found near click", bg="lightyellow")
        return False
    
    def show_edit_dialog(self, feature, index):
        """Show B-spectra feature editing dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit B-Feature at {feature.wavelength:.1f}nm")
        dialog.geometry("350x250")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 175
        y = (dialog.winfo_screenheight() // 2) - 125
        dialog.geometry(f"350x250+{x}+{y}")
        
        # Content
        tk.Label(dialog, text=f"B-Feature at {feature.wavelength:.1f}nm", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        tk.Label(dialog, text=f"Current: {feature.feature_type.replace('_', ' ').title()}", 
                font=('Arial', 10)).pack(pady=5)
        
        # B-spectra specific options
        tk.Label(dialog, text="Change to:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        feature_var = tk.StringVar(value=feature.feature_type)
        options = [
            ("Peak", "peak"),
            ("Mound Start", "mound_start"),
            ("Mound Crest", "mound_crest"), 
            ("Mound End", "mound_end"),
            ("Trough Start", "trough_start"),
            ("Trough Bottom", "trough_bottom"),
            ("Trough End", "trough_end")
        ]
        
        for text, value in options:
            tk.Radiobutton(dialog, text=text, variable=feature_var, value=value).pack(anchor='w', padx=20)
        
        # Buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        def apply_changes():
            new_type = feature_var.get()
            if new_type != feature.feature_type:
                self.features[index].feature_type = new_type
                self.features[index].feature_group = self.get_group_from_type(new_type)
                self.features[index].detection_method = "manual_edit"
                self.modified = True
                self.plot_spectrum()
            dialog.destroy()
        
        tk.Button(button_frame, text="Apply", command=apply_changes, 
                 bg="lightblue", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy, 
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
        if messagebox.askyesno("Reset", "Reset to original auto-detected features?"):
            detector = GeminiBSpectralDetector()
            results = detector.analyze_spectrum(self.wavelengths, self.intensities)
            self.features = results['features']
            self.modified = False
            self.plot_spectrum()
            self.status_label.config(text="Reset to auto-detected features", bg="lightyellow")
    
    def save_and_exit(self):
        """Save and exit"""
        self.result = self.features
        self.root.destroy()
    
    def cancel(self):
        """Cancel editing"""
        if self.modified and not messagebox.askyesno("Cancel", "Unsaved changes. Really cancel?"):
            return
        self.result = None
        self.root.destroy()
    
    def zoom_full(self):
        """Reset to full view"""
        self.ax.set_xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        self.ax.set_ylim(np.min(self.intensities) - 5, np.max(self.intensities) + 10)
        self.canvas.draw()
    
    def zoom_region(self, min_nm, max_nm):
        """Zoom to specific region"""
        mask = (self.wavelengths >= min_nm) & (self.wavelengths <= max_nm)
        if np.any(mask):
            region_intensities = self.intensities[mask]
            y_min, y_max = np.min(region_intensities) - 3, np.max(region_intensities) + 8
        else:
            y_min, y_max = np.min(self.intensities), np.max(self.intensities)
        
        self.ax.set_xlim(min_nm, max_nm)
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw()
    
    def show_modal(self):
        """Show as modal dialog"""
        self.root.transient()
        self.root.grab_set()
        self.root.wait_window()
        return getattr(self, 'result', None)

def ensure_directories_exist():
    """Create default directories if needed"""
    for directory in CONFIG['default_dirs'].values():
        os.makedirs(directory, exist_ok=True)

def create_csv_output(detector_results, input_filepath):
    """Convert detector results to CSV format matching manual marking program"""
    features = detector_results['features']
    norm_info = detector_results['normalization']
    baseline_info = detector_results['baseline_assessment']
    
    # Extract metadata
    file_name = os.path.basename(input_filepath)
    light_info = CONFIG['light_source_info']
    norm_factor = norm_info.get('reference_intensity', 50000) / 50000
    ref_wavelength = norm_info.get('reference_wavelength', 650.0)
    baseline_used = baseline_info.get('noise_std', 0.0) * 100
    
    csv_rows = []
    
    for feature in features:
        # Map feature to CSV format
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
            'Width_nm': round(feature.width_nm, 2) if feature.width_nm > 0 else '',
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
        symmetry_ratio = 1.0 + np.random.normal(0, 0.05)  # Placeholder
        skew_desc = "Symmetric" if 0.9 <= symmetry_ratio <= 1.1 else ("Left Skewed" if symmetry_ratio < 0.9 else "Right Skewed")
        
        summary_row = row.copy()  # Use last row as template
        summary_row.update({
            'Feature': 'Mound_Summary',
            'Wavelength': round(mound.wavelength, 2),
            'Intensity': round(mound.intensity, 2),
            'Point_Type': 'Summary',
            'Feature_Group': 'Mound',
            'Symmetry_Ratio': round(symmetry_ratio, 3),
            'Skew_Description': skew_desc,
            'Width_nm': round(mound.width_nm, 2) if mound.width_nm > 0 else '',
            'SNR': '',
            'Baseline_Used': '',
            'Norm_Factor': ''
        })
        csv_rows.append(summary_row)
    
    return csv_rows

def process_b_spectrum_file(input_filepath, output_dir=None, interactive=True):
    """Process B spectrum file with optional interactive editing"""
    try:
        detector = GeminiBSpectralDetector()
        
        print(f"Processing B spectrum: {input_filepath}")
        wavelengths, intensities = load_b_spectrum(input_filepath)
        results = detector.analyze_spectrum(wavelengths, intensities)
        
        # Get normalized intensities for display
        normalized_intensities = detector.normalize_b_spectrum(wavelengths, intensities)
        
        # Interactive editing
        if interactive:
            editor = InteractiveBSpectrumEditor(wavelengths, normalized_intensities, results['features'], input_filepath)
            edited_features = editor.show_modal()
            
            if edited_features is not None:
                results['features'] = edited_features
                print(f"Interactive editing complete. Features: {len(edited_features)}")
            else:
                print("Interactive editing cancelled - using auto-detected features")
        
        # Generate output
        output_dir = output_dir or CONFIG['default_dirs']['halogen_output']
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{Path(input_filepath).stem}_halogen_structural_{timestamp}.csv"
        output_path = Path(output_dir) / output_filename
        
        # Save CSV
        csv_data = create_csv_output(results, input_filepath)
        pd.DataFrame(csv_data).to_csv(output_path, index=False)
        
        # Print summary
        print(f"B spectrum analysis complete:")
        print(f"  Input: {input_filepath}")
        print(f"  Output: {output_path}")
        print(f"  Features: {results['feature_count']}")
        print(f"  Strategy: {results['detection_strategy']}")
        print(f"  Confidence: {results['overall_confidence']:.2f}")
        
        return output_path, results
        
    except Exception as e:
        print(f"Error processing B spectrum {input_filepath}: {str(e)}")
        return None, None

def main():
    """Main function - command line or interactive mode"""
    ensure_directories_exist()
    
    if len(sys.argv) > 1:
        # Command line mode
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
        print("Enhanced B Spectra Auto-Detection Script")
        print("Optimized for halogen/broadband transmission spectra")
        
        root = tk.Tk()
        root.withdraw()
        
        while True:
            # File selection
            input_file = filedialog.askopenfilename(
                title="Select B Spectrum File",
                initialdir=CONFIG['default_dirs']['input'] if os.path.exists(CONFIG['default_dirs']['input']) else None,
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if not input_file:
                break
            
            # Interactive editing option
            use_editor = messagebox.askyesno(
                "Interactive Editing",
                "Use interactive graph editor for B spectra?\n\nYes: Interactive editing (recommended)\nNo: Automatic only"
            )
            
            # Process file
            result_path, results = process_b_spectrum_file(input_file, CONFIG['default_dirs']['halogen_output'], use_editor)
            
            if result_path:
                messagebox.showinfo(
                    "Analysis Complete",
                    f"B spectrum analysis complete!\n\n"
                    f"Features: {results['feature_count']}\n"
                    f"Strategy: {results['detection_strategy']}\n"
                    f"Confidence: {results['overall_confidence']:.2f}\n\n"
                    f"Output: {result_path.name}\n"
                    f"Saved to: {result_path.parent}"
                )
            else:
                messagebox.showerror("Error", "Analysis failed. Check console.")
            
            # Continue?
            if not messagebox.askyesno("Continue?", "Process another B spectrum file?"):
                break
        
        root.destroy()

if __name__ == "__main__":
    main()
