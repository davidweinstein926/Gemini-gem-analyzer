# gemini_core_base.py
# Core functionality shared across all light sources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime
from scipy import stats

class GeminiCoreAnalyzer:
    """Base class for all Gemini spectral analyzers"""
    
    def __init__(self, output_directory):
        self.OUTPUT_DIRECTORY = output_directory
        self.reset_session()
        
    def reset_session(self):
        """Reset all session variables"""
        self.features = []
        self.current_type = None
        self.persistent_mode = True
        self.clicks = []
        self.filename = ""
        self.lines_drawn = []
        self.magnify_mode = False
        self.baseline_data = None
        self.spectrum_df = None
        self.ax = None
        self.fig = None
        
    def load_spectrum_file(self, file_path):
        """Load and validate spectrum file"""
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None)
            
            if df.shape[1] < 2:
                return None, "File does not contain two columns of data"
            
            # Check wavelength ordering
            wavelengths = df.iloc[:, 0]
            if wavelengths.iloc[0] > wavelengths.iloc[-1]:
                df = df.iloc[::-1].reset_index(drop=True)
                print("🔄 Auto-corrected wavelength order to ascending")
            
            return df, "success"
            
        except Exception as e:
            return None, f"Failed to load file: {e}"
    
    def calculate_baseline_stats(self, start_wl, end_wl):
        """Calculate baseline statistics"""
        try:
            wavelengths = self.spectrum_df.iloc[:, 0]
            intensities = self.spectrum_df.iloc[:, 1]
            
            mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            baseline_intensities = intensities[mask]
            baseline_wavelengths = wavelengths[mask]
            
            if len(baseline_intensities) < 3:
                return None
            
            avg_intensity = np.mean(baseline_intensities)
            std_dev = np.std(baseline_intensities)
            snr = avg_intensity / std_dev if std_dev > 0 else float('inf')
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                baseline_wavelengths, baseline_intensities)
            
            return {
                'wavelength_start': start_wl,
                'wavelength_end': end_wl,
                'avg_intensity': round(avg_intensity, 2),
                'std_deviation': round(std_dev, 3),
                'snr': round(snr, 1),
                'slope': round(slope, 6),
                'r_squared': round(r_value**2, 4),
                'data_points': len(baseline_intensities)
            }
            
        except Exception as e:
            print(f"❌ Baseline calculation error: {e}")
            return None
    
    def get_intensity_at_wavelength(self, target_wavelength):
        """Get precise intensity at wavelength"""
        try:
            wavelengths = self.spectrum_df.iloc[:, 0]
            intensities = self.spectrum_df.iloc[:, 1]
            
            if target_wavelength in wavelengths.values:
                idx = wavelengths[wavelengths == target_wavelength].index[0]
                return intensities.iloc[idx]
            else:
                return np.interp(target_wavelength, wavelengths, intensities)
                
        except Exception as e:
            print(f"❌ Intensity lookup error: {e}")
            return None
    
    def calculate_local_slope(self, wavelength, window=2.0):
        """Calculate local slope around wavelength"""
        try:
            wavelengths = self.spectrum_df.iloc[:, 0]
            intensities = self.spectrum_df.iloc[:, 1]
            
            mask = (wavelengths >= wavelength - window) & (wavelengths <= wavelength + window)
            local_wl = wavelengths[mask]
            local_int = intensities[mask]
            
            if len(local_wl) < 3:
                return None
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(local_wl, local_int)
            
            return {
                'slope': round(slope, 3),
                'r_squared': round(r_value**2, 4),
                'std_error': round(std_err, 3),
                'window_size': window,
                'data_points': len(local_wl)
            }
            
        except Exception as e:
            return None
    
    def save_features(self):
        """Save features to CSV"""
        if not self.features:
            print("⚠️ No features to save")
            return False
            
        try:
            df = pd.DataFrame(self.features)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = self.filename.replace('.txt', '') if self.filename else "unknown"
            outname = f"{base_name}_{self.get_light_source_name()}_features_{timestamp}.csv"
            
            os.makedirs(self.OUTPUT_DIRECTORY, exist_ok=True)
            full_path = os.path.join(self.OUTPUT_DIRECTORY, outname)
            
            df.to_csv(full_path, index=False)
            print(f"✅ Saved {len(self.features)} features to {full_path}")
            return True
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return False
    
    def get_light_source_name(self):
        """Override in subclasses"""
        return "generic"
    
    def get_structure_types(self):
        """Override in subclasses to define light-source specific features"""
        return {
            'Peak': ['Max'],
            'Baseline Region': ['Start', 'End']
        }
    
    def create_base_ui(self):
        """Create basic UI - extend in subclasses"""
        button_width = 0.13
        button_height = 0.04
        button_x = 0.84
        
        buttons = []
        
        # Always include baseline
        ax_baseline = self.fig.add_axes([button_x, 0.88, button_width, button_height])
        btn_baseline = Button(ax_baseline, 'Baseline\n(Key: 7)', 
                            color='lightgray', hovercolor='gray')
        btn_baseline.on_clicked(lambda e: self.select_feature('Baseline Region'))
        buttons.append(btn_baseline)
        
        # Utility buttons
        ax_save = self.fig.add_axes([button_x, 0.20, button_width, button_height])
        btn_save = Button(ax_save, 'Save\n(Key: S)', 
                         color='lightcyan', hovercolor='cyan')
        btn_save.on_clicked(lambda e: self.save_features())
        buttons.append(btn_save)
        
        return buttons
    
    def select_feature(self, feature_type):
        """Select feature type for marking"""
        self.current_type = feature_type
        self.clicks = []
        print(f"✅ Selected: {feature_type}")
        
    def file_selection_dialog(self, default_dir):
        """Handle file selection with retry logic"""
        while True:
            try:
                root = tk.Tk()
                root.withdraw()
                root.lift()
                root.attributes('-topmost', True)
                
                file_path = filedialog.askopenfilename(
                    parent=root,
                    initialdir=default_dir,
                    title=f"Select {self.get_light_source_name()} Spectrum File",
                    filetypes=[("Text files", "*.txt")]
                )
                
                root.quit()
                root.destroy()
                
                if not file_path:
                    retry = messagebox.askyesno(
                        "No File Selected",
                        "No file selected. Try again?"
                    )
                    if not retry:
                        return None
                    continue
                
                return file_path
                
            except Exception as e:
                print(f"❌ File selection error: {e}")
                retry = messagebox.askyesno(
                    "Error",
                    f"File selection error: {e}\nTry again?"
                )
                if not retry:
                    return None