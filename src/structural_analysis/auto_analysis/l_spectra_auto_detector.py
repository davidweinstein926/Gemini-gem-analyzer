#!/usr/bin/env python3
"""
Enhanced L Spectra Auto-Detection Script - UPDATED FOR NEW DIRECTORY STRUCTURE
Interactive graph editor for manual refinement of auto-detected features
Version: 3.0 (Updated for new directory structure + enhanced path detection)
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

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class LSpectraAutoDetector:
    def __init__(self):
        # UPDATED: Dynamic path detection for new directory structure
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent.parent  # Go up to gemini_gemological_analysis/
        
        print(f"🔍 L Spectra Auto Detector Paths:")
        print(f"   Script directory: {self.script_dir}")
        print(f"   Project root: {self.project_root}")
        
        # UPDATED: Setup directories
        self.setup_directories()
        
        # Import the detector
        self.import_detector()
        
        # UPDATED: Configuration with new paths
        self.config = {
            'zoom_regions': [(400, 500), (500, 600), (600, 700), (700, 800)],
            'feature_colors': {
                'peak': ('ro', 'darkred', 'Peak'),
                'mound': ('bs', 'darkblue', 'Mound'),
                'trough': ('^', 'purple', 'Trough'),
                'baseline': ('gs', 'green', 'Baseline')
            },
            'csv_mapping': {
                'baseline': {'baseline_start': ('Baseline_Start', 'Start'), 'baseline_end': ('Baseline_End', 'End')},
                'peak': {'peak': ('Peak', 'Crest')},
                'mound': {'mound_start': ('Mound_Start', 'Start'), 'mound_crest': ('Mound_Crest', 'Crest'), 'mound_end': ('Mound_End', 'End')},
                'trough': {'trough_start': ('Trough_Start', 'Start'), 'trough_bottom': ('Trough_Bottom', 'Bottom'), 'trough_end': ('Trough_End', 'End')}
            }
        }
        
    def setup_directories(self):
        """UPDATED: Setup directories for new project structure"""
        # Possible locations for input data
        input_search_paths = [
            self.project_root / "data" / "raw",  # New structure - primary
            self.project_root / "src" / "structural_analysis" / "data" / "raw",  # Local to structural analysis
            self.project_root / "raw_txt",  # Legacy location
            Path.home() / "OneDrive" / "Desktop" / "gemini matcher" / "gemini sp10 raw" / "raw text",  # Legacy user path
        ]
        
        # Possible locations for output data
        output_search_paths = [
            self.project_root / "data" / "output" / "laser",  # New structure - primary
            self.project_root / "src" / "structural_analysis" / "results" / "laser",  # Results in structural analysis
            self.project_root / "output" / "laser",  # Alternative root location
            self.project_root / "data" / "structural_data" / "laser",  # Alternative data location
            Path.home() / "OneDrive" / "Desktop" / "gemini_gemological_analysis" / "data" / "structural_data" / "laser",  # Legacy path
        ]
        
        # Find input directory
        self.input_directory = None
        for search_path in input_search_paths:
            if search_path.exists() and search_path.is_dir():
                self.input_directory = search_path
                print(f"✅ Found input directory: {self.input_directory}")
                break
        
        if not self.input_directory:
            # Use the primary new structure path (will be created if needed)
            self.input_directory = input_search_paths[0]
            print(f"⚠️  Input directory not found, will use: {self.input_directory}")
        
        # Find/create output directory
        self.output_directory = None
        for search_path in output_search_paths:
            if search_path.exists():
                self.output_directory = search_path
                print(f"✅ Found output directory: {self.output_directory}")
                break
        
        if not self.output_directory:
            # Use the primary new structure path and create it
            self.output_directory = output_search_paths[0]
            try:
                self.output_directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created output directory: {self.output_directory}")
            except Exception as e:
                print(f"⚠️  Could not create output directory: {e}")
                # Fallback to a guaranteed writable location
                self.output_directory = Path.cwd() / "l_spectra_results"
                self.output_directory.mkdir(exist_ok=True)
                print(f"📁 Using fallback output directory: {self.output_directory}")
    
    def import_detector(self):
        """UPDATED: Import detector with enhanced error handling"""
        # Try multiple import paths for the detector
        detector_import_paths = [
            "gemini_Lpeak_detector",  # Same directory
            "src.structural_analysis.auto_analysis.gemini_Lpeak_detector",  # Full project path
            "auto_analysis.gemini_Lpeak_detector",  # Relative from structural_analysis
        ]
        
        self.detector_class = None
        self.load_l_spectrum_func = None
        self.LSpectralFeature = None
        
        for import_path in detector_import_paths:
            try:
                if import_path == "gemini_Lpeak_detector":
                    from gemini_Lpeak_detector import GeminiLSpectralDetector, load_l_spectrum, LSpectralFeature
                else:
                    # Dynamic import for other paths
                    import importlib
                    module = importlib.import_module(import_path)
                    GeminiLSpectralDetector = getattr(module, 'GeminiLSpectralDetector')
                    load_l_spectrum = getattr(module, 'load_l_spectrum')
                    LSpectralFeature = getattr(module, 'LSpectralFeature')
                
                self.detector_class = GeminiLSpectralDetector
                self.load_l_spectrum_func = load_l_spectrum
                self.LSpectralFeature = LSpectralFeature
                print(f"✅ Successfully imported L detector from: {import_path}")
                return True
                
            except ImportError as e:
                print(f"⚠️  Failed to import from {import_path}: {e}")
                continue
        
        # If all imports failed
        print("❌ Error: gemini_Lpeak_detector.py not found in any expected location")
        print("Expected locations:")
        for path in detector_import_paths:
            print(f"   - {path}")
        print("Please ensure the detector file is accessible")
        return False

class InteractiveSpectrumEditor:
    def __init__(self, wavelengths, intensities, features, input_filepath, detector_instance):
        self.wavelengths = wavelengths
        self.intensities = intensities
        self.features = features.copy()
        self.input_filepath = input_filepath
        self.detector = detector_instance
        self.modified = False
        self.current_mode = "view"
        
        self.setup_gui()
        self.plot_spectrum()
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def setup_gui(self):
        """Initialize GUI components"""
        self.root = tk.Toplevel()
        self.root.title(f"L-Spectra Editor (UPDATED v3.0) - {Path(self.input_filepath).name}")
        
        # UPDATED: Better window sizing for your screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1200, int(screen_width * 0.85))
        window_height = min(800, int(screen_height * 0.8))
        self.root.geometry(f"{window_width}x{window_height}")
        
        # Header with update notice
        header_frame = tk.Frame(self.root, bg='darkgreen')
        header_frame.pack(fill=tk.X)
        
        tk.Label(header_frame, text="L-SPECTRA AUTO-DETECTION - UPDATED v3.0", 
                font=('Arial', 14, 'bold'), fg='white', bg='darkgreen').pack(pady=5)
        tk.Label(header_frame, text="UPDATED: New directory structure + enhanced path detection", 
                font=('Arial', 9), fg='lightgreen', bg='darkgreen').pack(pady=(0,5))
        
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
        self.status_label = tk.Label(self.root, text="Click spectrum to add peaks, markers to delete", 
                                   bg="lightyellow", font=('Arial', 10))
        self.status_label.pack(fill=tk.X)
    
    def setup_zoom_controls(self):
        """Create zoom control buttons"""
        zoom_frame = tk.Frame(self.root, bg='lightgray')
        zoom_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(zoom_frame, text="L-Spectra Quick Zoom:", bg='lightgray', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # Full view button
        tk.Button(zoom_frame, text="Full View", command=self.zoom_full, 
                 bg="white", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
        
        # Region zoom buttons
        for start, end in self.detector.config['zoom_regions']:
            tk.Button(zoom_frame, text=f"Zoom {start}-{end}nm", 
                     command=lambda s=start, e=end: self.zoom_region(s, e),
                     bg="lightblue", font=('Arial', 9)).pack(side=tk.LEFT, padx=2)
    
    def setup_action_buttons(self):
        """Create action control buttons"""
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        buttons = [
            ("Add Peak", self.add_peak_mode, "lightgreen"),
            ("Delete Feature", self.delete_mode, "lightcoral"),
            ("Edit Feature", self.edit_mode, "lightyellow"),
            ("Reset to Auto", self.reset_features, "lightgray"),
            ("Cancel", self.cancel, "lightgray"),
            ("Save & Exit", self.save_and_exit, "lightblue")
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            side = tk.RIGHT if i >= len(buttons) - 2 else tk.LEFT
            padx = 10 if side == tk.RIGHT else 5
            weight = 'bold' if 'Save' in text else 'normal'
            
            tk.Button(button_frame, text=text, command=command, bg=color, 
                     font=('Arial', 10, weight)).pack(side=side, padx=padx)
    
    def plot_spectrum(self):
        """Plot spectrum with detected features"""
        self.ax.clear()
        self.ax.plot(self.wavelengths, self.intensities, 'b-', linewidth=1, alpha=0.8, label='L Spectrum')
        
        # Track labels to avoid duplicates
        plotted_labels = set()
        
        for feature in self.features:
            color_info = self.detector.config['feature_colors'].get(feature.feature_group, ('ko', 'black', 'Unknown'))
            marker, edge_color, label = color_info
            
            # Plot marker
            self.ax.plot(feature.wavelength, feature.intensity, marker, markersize=5 if feature.feature_group != 'baseline' else 6,
                        markeredgecolor=edge_color, markeredgewidth=1, 
                        label=label if label not in plotted_labels else "")
            
            if label not in plotted_labels:
                plotted_labels.add(label)
            
            # Add annotation
            self.ax.annotate(f'{feature.wavelength:.1f}nm\n({label})', 
                           (feature.wavelength, feature.intensity),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=8, ha='left')
        
        # Formatting
        self.ax.set_xlabel('Wavelength (nm)', fontsize=12)
        self.ax.set_ylabel('Normalized Intensity (0-100 scale)', fontsize=12)
        self.ax.set_title(f'L-Spectra Analysis (UPDATED v3.0) - {Path(self.input_filepath).name}', fontsize=14)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
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
            "add_peak": self.add_peak_at_location,
            "delete": self.delete_feature_near_click,
            "edit": self.edit_feature_near_click
        }
        
        if self.current_mode in actions:
            success = actions[self.current_mode](event.xdata, event.ydata)
            if success and self.current_mode != "edit":
                self.current_mode = "view"
                self.status_label.config(text="Action completed", bg="lightyellow")
    
    def add_peak_at_location(self, wavelength, intensity):
        """Add peak at clicked location"""
        # Find closest data point
        distances = np.abs(self.wavelengths - wavelength)
        closest_idx = np.argmin(distances)
        
        new_feature = self.detector.LSpectralFeature(
            wavelength=self.wavelengths[closest_idx],
            intensity=self.intensities[closest_idx],
            feature_type="peak",
            feature_group="peak",
            prominence=5.0,
            snr=10.0,
            confidence=0.9,
            detection_method="manual_addition",
            width_nm=2.0
        )
        
        self.features.append(new_feature)
        self.modified = True
        self.plot_spectrum()
        return True
    
    def find_nearest_feature(self, click_wavelength, feature_groups=None):
        """Find nearest feature within tolerance"""
        if feature_groups is None:
            feature_groups = ["peak", "mound"]
        
        min_distance = float('inf')
        nearest_feature = None
        nearest_index = None
        
        for i, feature in enumerate(self.features):
            if feature.feature_group in feature_groups:
                distance = abs(feature.wavelength - click_wavelength)
                if distance < min_distance and distance < 10:  # 10nm tolerance
                    min_distance = distance
                    nearest_feature = feature
                    nearest_index = i
        
        return nearest_feature, nearest_index
    
    def delete_feature_near_click(self, click_wavelength, click_intensity):
        """Delete nearest feature"""
        feature, index = self.find_nearest_feature(click_wavelength, ["peak"])
        
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
        """Show feature editing dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Edit L-Feature at {feature.wavelength:.1f}nm")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 150
        y = (dialog.winfo_screenheight() // 2) - 100
        dialog.geometry(f"300x200+{x}+{y}")
        
        # Content
        tk.Label(dialog, text=f"L-Feature at {feature.wavelength:.1f}nm", 
                font=('Arial', 12, 'bold')).pack(pady=10)
        tk.Label(dialog, text=f"Current: {feature.feature_type.title()}", 
                font=('Arial', 10)).pack(pady=5)
        
        # Options
        tk.Label(dialog, text="Change to:", font=('Arial', 10, 'bold')).pack(pady=(10, 5))
        
        feature_var = tk.StringVar(value=feature.feature_type)
        options = [("Peak", "peak"), ("Mound Crest", "mound_crest"), ("Trough Bottom", "trough_bottom")]
        
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
        type_mapping = {
            'peak': 'peak',
            'mound_crest': 'mound',
            'mound_start': 'mound',
            'mound_end': 'mound',
            'trough_bottom': 'trough',
            'trough_start': 'trough',
            'trough_end': 'trough'
        }
        return type_mapping.get(feature_type, 'peak')
    
    def reset_features(self):
        """Reset to auto-detected features"""
        if messagebox.askyesno("Reset", "Reset to original auto-detected features?"):
            detector_instance = self.detector.detector_class()
            results = detector_instance.analyze_spectrum(self.wavelengths, self.intensities)
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
        self.ax.set_ylim(np.min(self.intensities) - 5, np.max(self.intensities) + 5)
        self.canvas.draw()
    
    def zoom_region(self, min_nm, max_nm):
        """Zoom to specific region"""
        mask = (self.wavelengths >= min_nm) & (self.wavelengths <= max_nm)
        if np.any(mask):
            region_intensities = self.intensities[mask]
            y_min, y_max = np.min(region_intensities) - 2, np.max(region_intensities) + 5
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

    def create_csv_output(self, detector_results, input_filepath):
        """UPDATED: Convert detector results to CSV format"""
        features = detector_results['features']
        norm_info = detector_results['normalization']
        baseline_info = detector_results['baseline_assessment']
        
        # Extract metadata
        file_name = os.path.basename(input_filepath)
        norm_factor = norm_info.get('reference_intensity', 50000) / 50000
        ref_wavelength = norm_info.get('reference_wavelength', 650.0)
        baseline_used = baseline_info.get('noise_std', 0.0) * 100
        
        csv_rows = []
        
        for feature in features:
            # Map feature to CSV format
            group_mapping = self.config['csv_mapping'].get(feature.feature_group, {})
            feature_name, point_type = group_mapping.get(feature.feature_type, (feature.feature_type.title(), "Point"))
            
            row = {
                'Feature': feature_name,
                'File': file_name,
                'Light_Source': "Laser",
                'Wavelength': round(feature.wavelength, 2),
                'Intensity': round(feature.intensity, 2),
                'Effective_Height': round(getattr(feature, 'effective_height', 0.0), 2),
                'Point_Type': point_type,
                'Feature_Group': feature.feature_group.title(),
                'Processing': "Baseline_Then_Laser_Normalized_UPDATED",
                'SNR': round(feature.snr if feature.snr > 0 else baseline_info.get('noise_std', 0) * 20, 1),
                'Feature_Key': f"{feature.feature_group}_{hash(feature.wavelength) % 10}",
                'Baseline_Used': round(baseline_used, 2),
                'Norm_Factor': round(norm_factor, 6),
                'Normalization_Method': "laser_max_50000_to_100_UPDATED",
                'Reference_Wavelength_Used': round(ref_wavelength, 3) if point_type in ['Start', 'End'] else '',
                'Width_nm': round(feature.width_nm, 2) if feature.width_nm > 0 else '',
                'Normalization_Scheme': "Laser_max_50000_to_100_UPDATED",
                'Reference_Wavelength': round(ref_wavelength, 3),
                'Intensity_Range_Min': 0.0,
                'Intensity_Range_Max': 100.0,
                'Directory_Structure': 'Updated_New_Structure',  # UPDATED: Add metadata
                'Output_Location': str(self.output_directory),  # UPDATED: Add output location
                'Auto_Detection_Method': 'GeminiLSpectralDetector',  # UPDATED: Add detection method
                'Symmetry_Ratio': '',
                'Skew_Description': ''
            }
            
            csv_rows.append(row)
        
        return csv_rows

    def process_l_spectrum_file(self, input_filepath, output_dir=None, interactive=True):
        """UPDATED: Process L spectrum file with optional interactive editing"""
        try:
            # Check if detector was imported successfully
            if not self.detector_class or not self.load_l_spectrum_func:
                print("❌ Detector not available - cannot process file")
                return None, None
            
            detector = self.detector_class()
            
            print(f"🔬 Processing L spectrum: {Path(input_filepath).name}")
            wavelengths, intensities = self.load_l_spectrum_func(input_filepath)
            results = detector.analyze_spectrum(wavelengths, intensities)
            
            # Get normalized intensities for display
            normalized_intensities = detector.normalize_l_spectrum(wavelengths, intensities)
            
            # Interactive editing
            if interactive:
                editor = InteractiveSpectrumEditor(wavelengths, normalized_intensities, results['features'], input_filepath, self)
                edited_features = editor.show_modal()
                
                if edited_features is not None:
                    results['features'] = edited_features
                    print(f"Interactive editing complete. Features: {len(edited_features)}")
                else:
                    print("Interactive editing cancelled - using auto-detected features")
            
            # Generate output
            output_dir = output_dir or self.output_directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{Path(input_filepath).stem}_laser_structural_auto_{timestamp}.csv"
            output_path = output_dir / output_filename
            
            # Save CSV
            csv_data = self.create_csv_output(results, input_filepath)
            pd.DataFrame(csv_data).to_csv(output_path, index=False)
            
            # Print summary
            print(f"✅ L spectrum analysis complete:")
            print(f"   Input: {Path(input_filepath).name}")
            print(f"   Output: {output_path}")
            print(f"   Output directory: {output_dir}")
            print(f"   Features: {results['feature_count']}")
            print(f"   Strategy: {results['detection_strategy']}")
            print(f"   Confidence: {results['overall_confidence']:.2f}")
            print(f"   Normalization: {results['normalization']['method']}")
            
            return output_path, results
            
        except Exception as e:
            print(f"❌ Error processing L spectrum {input_filepath}: {str(e)}")
            return None, None

    def file_selection_dialog(self):
        """UPDATED: File selection dialog with new directory structure support"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            
            # Check if input directory exists and has files
            if self.input_directory.exists():
                txt_files = list(self.input_directory.glob("*.txt"))
                if txt_files:
                    initial_dir = str(self.input_directory)
                    print(f"📂 Found {len(txt_files)} txt files in: {self.input_directory}")
                else:
                    initial_dir = str(self.input_directory)
                    print(f"📁 Using input directory (no txt files found): {self.input_directory}")
            else:
                initial_dir = str(self.project_root)
                print(f"📁 Input directory not found, using project root: {self.project_root}")
            
            file_path = filedialog.askopenfilename(
                parent=root, 
                initialdir=initial_dir,
                title=f"Select L Spectrum for UPDATED Auto Analysis\nLooking in: {Path(initial_dir).name}",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")])
            
            root.quit()
            root.destroy()
            
            if file_path:
                print(f"✅ Selected file: {Path(file_path).name}")
                print(f"   Full path: {file_path}")
            
            return file_path if file_path else None
            
        except Exception as e:
            print(f"File dialog error: {e}")
            return None

    def run_interactive(self):
        """UPDATED: Run in interactive mode with new directory structure support"""
        print("="*70)
        print("L SPECTRA AUTO-DETECTION - UPDATED FOR NEW DIRECTORY STRUCTURE")
        print("="*70)
        print(f"📁 Project root: {self.project_root}")
        print(f"📂 Input directory: {self.input_directory}")
        print(f"📁 Output directory: {self.output_directory}")
        print("="*70)
        
        # Check if detector is available
        if not self.detector_class or not self.load_l_spectrum_func:
            print("❌ Detector not available - cannot run interactive mode")
            messagebox.showerror("Import Error", "gemini_Lpeak_detector.py not found!\nPlace it in the auto_analysis directory.")
            return
        
        root = tk.Tk()
        root.withdraw()
        
        while True:
            # File selection
            input_file = self.file_selection_dialog()
            
            if not input_file:
                break
            
            # Interactive editing option
            use_editor = messagebox.askyesno(
                "Interactive Editing",
                "Use interactive graph editor?\n\n"
                "Yes: Interactive editing (recommended)\n"
                "- Visual spectrum display\n"
                "- Click to add/edit peaks\n"
                "- UPDATED normalization applied\n\n"
                "No: Automatic only"
            )
            
            # Process file
            result_path, results = self.process_l_spectrum_file(input_file, self.output_directory, use_editor)
            
            if result_path:
                messagebox.showinfo(
                    "Analysis Complete - UPDATED",
                    f"✅ L spectrum analysis complete!\n\n"
                    f"Features: {results['feature_count']}\n"
                    f"Strategy: {results['detection_strategy']}\n"
                    f"Confidence: {results['overall_confidence']:.2f}\n"
                    f"Normalization: UPDATED 2-step process\n\n"
                    f"Output: {result_path.name}\n"
                    f"Saved to: {result_path.parent}"
                )
            else:
                messagebox.showerror("Error", "Analysis failed. Check console.")
            
            # Continue?
            if not messagebox.askyesno("Continue?", "Process another file?"):
                break
        
        root.destroy()

def main():
    """UPDATED: Main function with enhanced command line and interactive support"""
    detector = LSpectraAutoDetector()
    
    if len(sys.argv) > 1:
        # Command line mode
        print("Enhanced L Spectra Auto-Detection Script - UPDATED v3.0")
        print("UPDATED: New directory structure + enhanced path detection")
        print("Optimized for laser-induced high-resolution spectra")
        print("=" * 70)
        
        input_files = sys.argv[1:]
        output_dir = detector.output_directory
        
        # Check if last argument is output directory
        if os.path.isdir(sys.argv[-1]):
            output_dir = sys.argv[-1]
            input_files = sys.argv[1:-1]
        
        processed = 0
        for input_file in input_files:
            if os.path.exists(input_file):
                output_path, results = detector.process_l_spectrum_file(input_file, output_dir, interactive=False)
                if results:
                    processed += 1
            else:
                print(f"⚠️ Warning: File not found: {input_file}")
        
        print(f"\n✅ Command line processing complete: {processed}/{len(input_files)} files processed")
    
    else:
        # Interactive GUI mode
        detector.run_interactive()

if __name__ == "__main__":
    main()