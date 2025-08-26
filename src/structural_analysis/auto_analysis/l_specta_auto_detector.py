#!/usr/bin/env python3
"""
Enhanced L Spectra Auto-Detection Script
Processes L spectra files with interactive graph editor for manual refinement.

Features:
- Default directories for input/output  
- Interactive graph editor for adding/removing detections
- Compatible with Gemini Launcher system
- Optimized for laser-induced high-resolution spectra
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches
from matplotlib.widgets import Button

# Import the detector
try:
    from gemini_Lpeak_detector import GeminiLSpectralDetector, load_l_spectrum
except ImportError:
    print("Error: gemini_Lpeak_detector.py not found in Python path")
    print("Please ensure the detector file is in the same directory or add to PYTHONPATH")
    if len(sys.argv) == 1:  # Interactive mode
        messagebox.showerror("Import Error", "gemini_Lpeak_detector.py not found!\nPlace it in the same directory as this script.")
    sys.exit(1)

# Default directories
DEFAULT_INPUT_DIR = r"C:\Users\David\gemini sp10 raw"
DEFAULT_LASER_OUTPUT_DIR = r"C:\Users\David\gemini sp10 structural data\laser"
DEFAULT_HALOGEN_OUTPUT_DIR = r"C:\Users\David\gemini sp10 structural data\halogen"

class InteractiveSpectrumEditor:
    """Interactive graph editor for manual refinement of auto-detected features"""
    
    def __init__(self, wavelengths, intensities, features, input_filepath):
        self.wavelengths = wavelengths
        self.intensities = intensities
        self.features = features.copy()  # Make a copy we can edit
        self.input_filepath = input_filepath
        self.modified = False
        
        # Create GUI
        self.root = tk.Toplevel()
        self.root.title(f"L-Spectra Editor - {Path(input_filepath).name}")
        self.root.geometry("1200x800")
        
        # Create matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar (zoom, pan tools)
        toolbar_frame = tk.Frame(self.root)
        toolbar_frame.pack(fill=tk.X)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        # Additional zoom controls
        zoom_frame = tk.Frame(self.root, bg='lightgray')
        zoom_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(zoom_frame, text="Quick Zoom:", bg='lightgray', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(zoom_frame, text="Full View", command=self.zoom_full, 
                 bg="white", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Zoom 400-500nm", command=lambda: self.zoom_region(400, 500), 
                 bg="lightblue", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Zoom 500-600nm", command=lambda: self.zoom_region(500, 600), 
                 bg="lightblue", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Zoom 600-700nm", command=lambda: self.zoom_region(600, 700), 
                 bg="lightblue", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        tk.Button(zoom_frame, text="Zoom 700-800nm", command=lambda: self.zoom_region(700, 800), 
                 bg="lightblue", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(button_frame, text="Add Peak", command=self.add_peak_mode, 
                 bg="lightgreen", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Delete Feature", command=self.delete_mode,
                 bg="lightcoral", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Edit Feature", command=self.edit_mode,
                 bg="lightyellow", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset to Auto", command=self.reset_features,
                 bg="lightgray", font=('Arial', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Save & Exit", command=self.save_and_exit,
                 bg="lightblue", font=('Arial', 12, 'bold')).pack(side=tk.RIGHT, padx=10)
        tk.Button(button_frame, text="Cancel", command=self.cancel,
                 bg="lightgray", font=('Arial', 10)).pack(side=tk.RIGHT, padx=5)
        
        # Status label
        self.status_label = tk.Label(self.root, text="Click on spectrum to add peaks, click on markers to delete features", 
                                   bg="lightyellow", font=('Arial', 10))
        self.status_label.pack(fill=tk.X)
        
        # Mode tracking
        self.current_mode = "view"
        self.click_handler = None
        
        # Plot the spectrum
        self.plot_spectrum()
        
        # Connect click handler
        self.canvas.mpl_connect('button_press_event', self.on_click)
        
    def plot_spectrum(self):
        """Plot spectrum with detected features"""
        self.ax.clear()
        
        # Plot spectrum
        self.ax.plot(self.wavelengths, self.intensities, 'b-', linewidth=1, alpha=0.8, label='Spectrum')
        
        # Plot features with different markers for different types
        for feature in self.features:
            if feature.feature_group == "peak":
                self.ax.plot(feature.wavelength, feature.intensity, 'ro', markersize=5, 
                           markeredgecolor='darkred', markeredgewidth=1, 
                           label='Peak' if 'Peak' not in [l.get_label() for l in self.ax.get_children()] else "")
                self.ax.annotate(f'{feature.wavelength:.1f}nm\n(Peak)', 
                               (feature.wavelength, feature.intensity),
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=8, ha='left')
            elif feature.feature_group == "mound":
                self.ax.plot(feature.wavelength, feature.intensity, 'bs', markersize=5, 
                           markeredgecolor='darkblue', markeredgewidth=1,
                           label='Mound' if 'Mound' not in [l.get_label() for l in self.ax.get_children()] else "")
                self.ax.annotate(f'{feature.wavelength:.1f}nm\n(Mound)', 
                               (feature.wavelength, feature.intensity),
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=8, ha='left')
            elif feature.feature_group == "trough":
                self.ax.plot(feature.wavelength, feature.intensity, '^', color='purple', markersize=5, 
                           markeredgecolor='darkmagenta', markeredgewidth=1,
                           label='Trough' if 'Trough' not in [l.get_label() for l in self.ax.get_children()] else "")
                self.ax.annotate(f'{feature.wavelength:.1f}nm\n(Trough)', 
                               (feature.wavelength, feature.intensity),
                               xytext=(5, 10), textcoords='offset points',
                               fontsize=8, ha='left')
            elif feature.feature_group == "baseline":
                if feature.feature_type == "baseline_start":
                    self.ax.plot(feature.wavelength, feature.intensity, 'gs', markersize=6,
                               label='Baseline' if 'Baseline' not in [l.get_label() for l in self.ax.get_children()] else "")
                else:
                    self.ax.plot(feature.wavelength, feature.intensity, 'gs', markersize=6)
        
        self.ax.set_xlabel('Wavelength (nm)', fontsize=12)
        self.ax.set_ylabel('Intensity', fontsize=12)
        self.ax.set_title(f'L-Spectra Analysis - {Path(self.input_filepath).name}', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Highlight modified status
        if self.modified:
            self.ax.text(0.02, 0.98, "MODIFIED", transform=self.ax.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8),
                        fontsize=10, fontweight='bold', va='top')
        
        self.canvas.draw()
    
    def add_peak_mode(self):
        """Enter add peak mode"""
        self.current_mode = "add_peak"
        self.status_label.config(text="ADD PEAK MODE: Click on spectrum where you want to add a peak", bg="lightgreen")
    
    def delete_mode(self):
        """Enter delete mode"""
        self.current_mode = "delete"
        self.status_label.config(text="DELETE MODE: Click on a red marker to delete that peak", bg="lightcoral")
    
    def edit_mode(self):
        """Enter edit feature mode"""
        self.current_mode = "edit"
        self.status_label.config(text="EDIT MODE: Click on a marker to change its feature type (peak ↔ mound)", bg="lightyellow")
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax or event.button != 1:  # Only left clicks in plot area
            return
        
        click_wavelength = event.xdata
        click_intensity = event.ydata
        
        if self.current_mode == "add_peak":
            self.add_peak_at_location(click_wavelength, click_intensity)
            self.current_mode = "view"
            self.status_label.config(text="Peak added! Click buttons above to add more or delete features", bg="lightyellow")
            
        elif self.current_mode == "delete":
            deleted = self.delete_feature_near_click(click_wavelength, click_intensity)
            if deleted:
                self.current_mode = "view"
                self.status_label.config(text="Feature deleted! Click buttons above to add more or delete features", bg="lightyellow")
            else:
                self.status_label.config(text="No feature found near click. Try clicking closer to a marker.", bg="lightcoral")
        
        elif self.current_mode == "edit":
            edited = self.edit_feature_near_click(click_wavelength, click_intensity)
            if edited:
                self.current_mode = "view"
                self.status_label.config(text="Feature edited! Click buttons above for more actions", bg="lightyellow")
            else:
                self.status_label.config(text="No feature found near click. Try clicking closer to a marker.", bg="lightyellow")
    
    def add_peak_at_location(self, wavelength, intensity):
        """Add a peak at the specified location"""
        # Find the actual data point closest to the click
        distances = np.abs(self.wavelengths - wavelength)
        closest_idx = np.argmin(distances)
        actual_wavelength = self.wavelengths[closest_idx]
        actual_intensity = self.intensities[closest_idx]
        
        # Create new peak feature
        from gemini_Lpeak_detector import LSpectralFeature
        new_feature = LSpectralFeature(
            wavelength=actual_wavelength,
            intensity=actual_intensity,
            feature_type="peak",
            feature_group="peak",
            prominence=5.0,  # Default values for manually added peaks
            snr=10.0,
            confidence=0.9,  # High confidence for manual selection
            detection_method="manual_addition",
            width_nm=2.0
        )
        
        self.features.append(new_feature)
        self.modified = True
        self.plot_spectrum()
    
    def edit_feature_near_click(self, click_wavelength, click_intensity):
        """Edit feature near click location"""
        min_distance = float('inf')
        feature_to_edit = None
        feature_index = None
        
        for i, feature in enumerate(self.features):
            if feature.feature_group in ["peak", "mound"]:  # Only allow editing of peaks/mounds
                distance = abs(feature.wavelength - click_wavelength)
                if distance < min_distance and distance < 10:  # Within 10nm
                    min_distance = distance
                    feature_to_edit = feature
                    feature_index = i
        
        if feature_to_edit is not None:
            # Create edit dialog
            edit_dialog = tk.Toplevel(self.root)
            edit_dialog.title(f"Edit Feature at {feature_to_edit.wavelength:.1f}nm")
            edit_dialog.geometry("300x200")
            edit_dialog.transient(self.root)
            edit_dialog.grab_set()
            
            # Center the dialog
            edit_dialog.update_idletasks()
            x = (edit_dialog.winfo_screenwidth() // 2) - 150
            y = (edit_dialog.winfo_screenheight() // 2) - 100
            edit_dialog.geometry(f"300x200+{x}+{y}")
            
            tk.Label(edit_dialog, text=f"Feature at {feature_to_edit.wavelength:.1f}nm", 
                    font=('Arial', 12, 'bold')).pack(pady=10)
            tk.Label(edit_dialog, text=f"Current type: {feature_to_edit.feature_type.title()}", 
                    font=('Arial', 10)).pack(pady=5)
            
            # Feature type selection
            tk.Label(edit_dialog, text="Change to:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
            
            feature_var = tk.StringVar(value=feature_to_edit.feature_type)
            feature_options = [
                ("Peak", "peak"),
                ("Mound Crest", "mound_crest"),
                ("Trough Bottom", "trough_bottom")
            ]
            
            for text, value in feature_options:
                tk.Radiobutton(edit_dialog, text=text, variable=feature_var, value=value).pack(anchor='w', padx=20)
            
            # Buttons
            button_frame = tk.Frame(edit_dialog)
            button_frame.pack(pady=10)
            
            def apply_changes():
                new_type = feature_var.get()
                if new_type != feature_to_edit.feature_type:
                    # Update feature type and group
                    self.features[feature_index].feature_type = new_type
                    if new_type == "peak":
                        self.features[feature_index].feature_group = "peak"
                    elif new_type == "mound_crest":
                        self.features[feature_index].feature_group = "mound"
                    elif new_type == "trough_bottom":
                        self.features[feature_index].feature_group = "trough"
                    
                    self.features[feature_index].detection_method = "manual_edit"
                    self.modified = True
                    self.plot_spectrum()
                
                edit_dialog.destroy()
            
            tk.Button(button_frame, text="Apply", command=apply_changes, 
                     bg="lightblue", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
            tk.Button(button_frame, text="Cancel", command=edit_dialog.destroy, 
                     bg="lightgray", font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
            
            return True
        
        return False
    
    def delete_feature_near_click(self, click_wavelength, click_intensity):
        """Delete feature near click location"""
        min_distance = float('inf')
        feature_to_delete = None
        
        for i, feature in enumerate(self.features):
            if feature.feature_group == "peak":  # Only allow deletion of peaks
                distance = abs(feature.wavelength - click_wavelength)
                if distance < min_distance and distance < 10:  # Within 10nm
                    min_distance = distance
                    feature_to_delete = i
        
        if feature_to_delete is not None:
            deleted_feature = self.features.pop(feature_to_delete)
            self.modified = True
            self.plot_spectrum()
            return True
        
        return False
    
    def reset_features(self):
        """Reset to original auto-detected features"""
        if messagebox.askyesno("Reset", "Reset to original auto-detected features?"):
            # Re-run the detector to get original features
            detector = GeminiLSpectralDetector()
            results = detector.analyze_spectrum(self.wavelengths, self.intensities)
            self.features = results['features']
            self.modified = False
            self.plot_spectrum()
            self.status_label.config(text="Reset to auto-detected features", bg="lightyellow")
    
    def save_and_exit(self):
        """Save current features and exit"""
        self.result = self.features
        self.root.destroy()
    
    def cancel(self):
        """Cancel editing and exit"""
        if self.modified:
            if not messagebox.askyesno("Cancel", "You have unsaved changes. Really cancel?"):
                return
        self.result = None
        self.root.destroy()
    
    def zoom_full(self):
        """Reset to full spectrum view"""
        self.ax.set_xlim(np.min(self.wavelengths), np.max(self.wavelengths))
        self.ax.set_ylim(np.min(self.intensities) - 5, np.max(self.intensities) + 5)
        self.canvas.draw()
        self.status_label.config(text="Zoomed to full spectrum view", bg="lightyellow")
    
    def zoom_region(self, min_nm, max_nm):
        """Zoom to specific wavelength region"""
        # Find intensity range in this region
        mask = (self.wavelengths >= min_nm) & (self.wavelengths <= max_nm)
        if np.any(mask):
            region_intensities = self.intensities[mask]
            y_min = np.min(region_intensities) - 2
            y_max = np.max(region_intensities) + 5
        else:
            y_min = np.min(self.intensities)
            y_max = np.max(self.intensities)
        
        self.ax.set_xlim(min_nm, max_nm)
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw()
        self.status_label.config(text=f"Zoomed to {min_nm}-{max_nm}nm region", bg="lightyellow")
    
    def show_modal(self):
        """Show the editor as modal dialog"""
        self.root.transient()
        self.root.grab_set()
        self.root.wait_window()
        return getattr(self, 'result', None)

def ensure_directories_exist():
    """Create default directories if they don't exist"""
    dirs = [DEFAULT_INPUT_DIR, DEFAULT_LASER_OUTPUT_DIR, DEFAULT_HALOGEN_OUTPUT_DIR]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def create_csv_output(detector_results, input_filepath):
    """
    Convert detector results to CSV format matching manual marking program
    """
    features = detector_results['features']
    norm_info = detector_results['normalization']
    baseline_info = detector_results['baseline_assessment']
    
    # Extract file info
    file_name = os.path.basename(input_filepath)
    light_source = "Laser"  # L spectra are laser-based
    processing = "Baseline_Then_Laser_Normalized"
    norm_method = "laser_650nm_50000_to_100"
    norm_scheme = "Laser_650nm_50000_to_100"
    
    # Calculate normalization factor (original_intensity / 50000)
    norm_factor = norm_info['reference_intensity'] / 50000 if 'reference_intensity' in norm_info else 1.0
    ref_wavelength = norm_info.get('reference_wavelength', 650.0)
    baseline_used = baseline_info.get('noise_std', 0.0) * 100  # Convert to percentage scale
    
    csv_rows = []
    
    for feature in features:
        # Determine feature naming based on type
        if feature.feature_group == "baseline":
            if feature.feature_type == "baseline_start":
                feature_name = "Baseline_Start"
                point_type = "Start"
            else:  # baseline_end
                feature_name = "Baseline_End"
                point_type = "End"
            feature_key = "Baseline_0"
            
        elif feature.feature_group == "peak":
            feature_name = "Peak"
            point_type = "Crest"
            feature_key = f"Peak_{hash(feature.wavelength) % 10}"
            
        elif feature.feature_group == "mound":
            if feature.feature_type == "mound_start":
                feature_name = "Mound_Start"
                point_type = "Start"
            elif feature.feature_type == "mound_crest":
                feature_name = "Mound_Crest"
                point_type = "Crest"
            else:  # mound_end
                feature_name = "Mound_End"
                point_type = "End"
            feature_key = f"Mound_{hash(feature.wavelength) % 10}"
            
        elif feature.feature_group == "trough":
            if feature.feature_type == "trough_start":
                feature_name = "Trough_Start"
                point_type = "Start"
            elif feature.feature_type == "trough_bottom":
                feature_name = "Trough_Bottom"
                point_type = "Bottom"
            else:  # trough_end
                feature_name = "Trough_End"
                point_type = "End"
            feature_key = f"Trough_{hash(feature.wavelength) % 10}"
            
        else:
            feature_name = feature.feature_type.title()
            point_type = "Point"
            feature_key = f"Feature_{hash(feature.wavelength) % 10}"
        
        # Create row
        row = {
            'Feature': feature_name,
            'File': file_name,
            'Light_Source': light_source,
            'Wavelength': round(feature.wavelength, 2),
            'Intensity': round(feature.intensity, 2),
            'Effective_Height': round(feature.effective_height, 2) if hasattr(feature, 'effective_height') else 0.0,
            'Point_Type': point_type,
            'Feature_Group': feature.feature_group.title(),
            'Processing': processing,
            'SNR': round(feature.snr, 1) if feature.snr > 0 else round(baseline_info.get('noise_std', 0) * 20, 1),
            'Feature_Key': feature_key,
            'Baseline_Used': round(baseline_used, 2),
            'Norm_Factor': round(norm_factor, 6),
            'Normalization_Method': norm_method,
            'Reference_Wavelength_Used': round(ref_wavelength, 3) if point_type in ['Start', 'End'] else '',
            'Symmetry_Ratio': '',  # Will be filled for summary rows
            'Skew_Description': '',  # Will be filled for summary rows
            'Width_nm': round(feature.width_nm, 2) if feature.width_nm > 0 else '',
            'Normalization_Scheme': norm_scheme,
            'Reference_Wavelength': round(ref_wavelength, 3),
            'Intensity_Range_Min': 0.0,
            'Intensity_Range_Max': 100.0
        }
        
        csv_rows.append(row)
    
    return csv_rows

def process_l_spectrum_file(input_filepath, output_dir=None, interactive=True):
    """
    Process an L spectrum file with optional interactive editing
    """
    try:
        # Initialize detector
        detector = GeminiLSpectralDetector()
        
        # Load and analyze spectrum
        print(f"Processing L spectrum: {input_filepath}")
        wavelengths, intensities = load_l_spectrum(input_filepath)
        results = detector.analyze_spectrum(wavelengths, intensities)
        
        # Get normalized intensities for display (0-100 scale)
        normalized_intensities = detector.normalize_l_spectrum(wavelengths, intensities)
        
        # Interactive editing if requested
        if interactive:
            editor = InteractiveSpectrumEditor(wavelengths, normalized_intensities, results['features'], input_filepath)
            edited_features = editor.show_modal()
            
            if edited_features is not None:  # User didn't cancel
                results['features'] = edited_features
                print(f"Interactive editing complete. Final feature count: {len(edited_features)}")
            else:
                print("Interactive editing cancelled - using auto-detected features")
        
        # Generate output filename
        input_path = Path(input_filepath)
        if output_dir is None:
            output_dir = DEFAULT_LASER_OUTPUT_DIR
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_laser_structural_{timestamp}.csv"
        output_path = Path(output_dir) / output_filename
        
        # Convert to CSV format
        csv_data = create_csv_output(results, input_filepath)
        df = pd.DataFrame(csv_data)
        
        # Save CSV file
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"L spectrum analysis complete:")
        print(f"  Input: {input_filepath}")
        print(f"  Output: {output_path}")
        print(f"  Features detected: {results['feature_count']}")
        print(f"  Detection strategy: {results['detection_strategy']}")
        print(f"  Baseline classification: {results['baseline_assessment']['noise_classification']}")
        print(f"  Overall confidence: {results['overall_confidence']:.2f}")
        
        return output_path, results
        
    except Exception as e:
        print(f"Error processing L spectrum {input_filepath}: {str(e)}")
        return None, None

def main():
    """
    Main function - processes command line arguments or runs interactively
    """
    # Ensure default directories exist
    ensure_directories_exist()
    
    if len(sys.argv) > 1:
        # Command line mode
        input_files = sys.argv[1:]
        output_dir = DEFAULT_LASER_OUTPUT_DIR
        
        # Check if last argument is output directory
        if os.path.isdir(sys.argv[-1]):
            output_dir = sys.argv[-1]
            input_files = sys.argv[1:-1]
        
        for input_file in input_files:
            if os.path.exists(input_file):
                process_l_spectrum_file(input_file, output_dir, interactive=False)
            else:
                print(f"Warning: File not found: {input_file}")
    
    else:
        # Interactive mode with GUI
        print("Enhanced L Spectra Auto-Detection Script")
        print("=======================================")
        print("Optimized for laser-induced high-resolution spectra")
        print("Features: Interactive graph editor, default directories")
        
        # Create simple GUI for file selection
        root = tk.Tk()
        root.withdraw()  # Hide main window
        
        while True:
            # File selection dialog with default directory
            input_file = filedialog.askopenfilename(
                title="Select L Spectrum File",
                initialdir=DEFAULT_INPUT_DIR if os.path.exists(DEFAULT_INPUT_DIR) else None,
                filetypes=[
                    ("Text files", "*.txt"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            
            if not input_file:  # User cancelled
                break
                
            # Ask about interactive editing
            use_editor = messagebox.askyesno(
                "Interactive Editing",
                "Would you like to use the interactive graph editor to review and modify the auto-detected features?\n\n" +
                "Click Yes for interactive editing (recommended)\n" +
                "Click No for automatic processing only"
            )
            
            # Process the file
            result_path, results = process_l_spectrum_file(input_file, DEFAULT_LASER_OUTPUT_DIR, interactive=use_editor)
            
            if result_path:
                messagebox.showinfo(
                    "Analysis Complete", 
                    f"L spectrum analysis complete!\n\n" +
                    f"Features detected: {results['feature_count']}\n" +
                    f"Strategy: {results['detection_strategy']}\n" +
                    f"Confidence: {results['overall_confidence']:.2f}\n\n" +
                    f"Output: {result_path.name}\n" +
                    f"Saved to: {result_path.parent}"
                )
            else:
                messagebox.showerror("Error", "Analysis failed. Check console for details.")
            
            # Ask if user wants to process another file
            if not messagebox.askyesno("Continue?", "Process another L spectrum file?"):
                break
        
        root.destroy()

if __name__ == "__main__":
    main()
