# gemini_laser_analyzer.py - UPDATED FOR NEW DIRECTORY STRUCTURE
# Enhanced path detection for data/raw input and data/output results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class GeminiLaserAnalyzer:
    def __init__(self):
        # FIXED: Corrected path detection for new directory structure
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent.parent.parent  # FIXED: Go up to gemini_gemological_analysis/
        
        print(f"🔍 Laser Analyzer Paths:")
        print(f"   Script directory: {self.script_dir}")
        print(f"   Project root: {self.project_root}")
        
        # UPDATED: Dynamic output directory detection
        self.setup_directories()
        
        # OPTIMIZED: Consolidated configuration data
        self.feature_config = {
            'Baseline': {'points': ['Start', 'End'], 'color': 'gray', 'key': 'b'},
            'Mound': {'points': ['Start', 'Crest', 'End'], 'color': 'red', 'key': '1'},
            'Plateau': {'points': ['Start', 'Midpoint', 'End'], 'color': 'green', 'key': '2'},
            'Peak': {'points': ['Max'], 'color': 'blue', 'key': '3'},
            'Trough': {'points': ['Start', 'Bottom', 'End'], 'color': 'purple', 'key': '4'},
            'Shoulder': {'points': ['Start', 'Peak', 'End'], 'color': 'orange', 'key': '5'},
            'Valley': {'points': ['Midpoint'], 'color': 'brown', 'key': '6'},
            'Diagnostic Region': {'points': ['Start', 'End'], 'color': 'gold', 'key': '7'}
        }
        
        self.button_config = [
            ('Baseline\n(B)', 'lightgray', 'Baseline'),
            ('Mound\n(1)', 'lightcoral', 'Mound'),
            ('Plateau\n(2)', 'lightgreen', 'Plateau'),
            ('Peak\n(3)', 'lightblue', 'Peak'),
            ('Trough\n(4)', 'plum', 'Trough'),
            ('Shoulder\n(5)', 'moccasin', 'Shoulder'),
            ('Valley\n(6)', 'burlywood', 'Valley'),
            ('Diagnostic Region\n(7)', 'lightyellow', 'Diagnostic Region'),
            ('Undo\n(U)', 'mistyrose', 'undo'),
            ('Save\n(S)', 'lightcyan', 'save'),
            ('Persistent\n(P)', 'lavender', 'persistent')
        ]
        
        self.reset_session()
        
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
            self.project_root / "data" / "structural_data" / "laser",  # CORRECTED: Structural data directory
            self.project_root / "src" / "structural_analysis" / "results" / "laser",  # Results in structural analysis
            self.project_root / "output" / "laser",  # Alternative root location
            Path.home() / "gemini sp10 structural data" / "laser",  # Legacy user path
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
            print(f"⚠️ Input directory not found, will use: {self.input_directory}")
        
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
                print(f"⚠️ Could not create output directory: {e}")
                # Fallback to a guaranteed writable location
                self.output_directory = Path.cwd() / "laser_results"
                self.output_directory.mkdir(exist_ok=True)
                print(f"🔍 Using fallback output directory: {self.output_directory}")
        
        # Store as string for compatibility with existing code
        self.OUTPUT_DIRECTORY = str(self.output_directory)
        
    def reset_session(self):
        """OPTIMIZED: Reset all session variables"""
        session_vars = ['features', 'clicks', 'lines_drawn', 'buttons', 'completed_feature_visuals']
        for var in session_vars:
            setattr(self, var, [])
        
        simple_vars = {
            'current_type': None, 'persistent_mode': True, 'filename': "",
            'baseline_data': None, 'spectrum_df': None, 'ax': None, 'fig': None,
            'normalization_applied': False, 'normalization_info': None, 'feature_ready': False
        }
        for var, value in simple_vars.items():
            setattr(self, var, value)
        
        print("Session reset - ready for new spectrum")
        
    def load_spectrum_file(self, file_path):
        """OPTIMIZED: Load and validate spectrum file"""
        try:
            df = pd.read_csv(file_path, sep='\s+', header=None)
            if df.shape[1] < 2:
                return None, "File does not contain two columns of data"
            if df.iloc[0, 0] > df.iloc[-1, 0]:
                df = df.iloc[::-1].reset_index(drop=True)
                print("Auto-corrected wavelength order")
            self.original_spectrum_df = df.copy()
            return df, "success"
        except Exception as e:
            return None, f"Failed to load file: {e}"
    
    def calculate_baseline_stats(self, start_wl, end_wl):
        """Calculate baseline statistics"""
        try:
            wl, intens = self.spectrum_df.iloc[:, 0], self.spectrum_df.iloc[:, 1]
            mask = (wl >= start_wl) & (wl <= end_wl)
            baseline_int = intens[mask]
            if len(baseline_int) < 3: 
                return None
            
            avg_int = np.mean(baseline_int)
            std_dev = np.std(baseline_int)
            snr = avg_int / std_dev if std_dev > 0 else float('inf')
            
            return {
                'wavelength_start': start_wl, 'wavelength_end': end_wl,
                'avg_intensity': round(avg_int, 2), 'std_deviation': round(std_dev, 3),
                'snr': round(snr, 1), 'data_points': len(baseline_int)
            }
        except:
            return None
    
    def apply_processing_pipeline(self):
        """OPTIMIZED: Apply baseline correction and normalization"""
        if not self.baseline_data:
            print("No baseline established")
            return False
        
        try:
            # Step 1: Baseline correction
            baseline_avg = self.baseline_data['avg_intensity']
            print(f"Applying baseline correction: subtracting {baseline_avg:.2f}")
            corrected = self.spectrum_df.iloc[:, 1] - baseline_avg
            self.spectrum_df.iloc[:, 1] = corrected.clip(lower=0)
            
            # Step 2: FIXED Laser normalization
            self.normalization_info = self.normalize_laser_spectrum_fixed()
            if self.normalization_info:
                self.normalization_applied = True
                self.update_plot("FIXED Normalized (0-100)")
                print("FIXED processing complete: Baseline + Laser Normalized")
                return True
            else:
                print("Normalization failed")
                return False
                
        except Exception as e:
            print(f"Processing pipeline error: {e}")
            return False

    def normalize_laser_spectrum_fixed(self):
        """OPTIMIZED: FIXED Laser normalization - Maximum intensity → 50,000, then scale 0-100"""
        try:
            wl = self.spectrum_df.iloc[:, 0]
            intens = self.spectrum_df.iloc[:, 1].values.copy()
            
            print(f"FIXED LASER NORMALIZATION: Using maximum intensity as reference")
            
            # Find maximum intensity in spectrum
            ref_value = np.max(intens)
            ref_idx = np.argmax(intens)
            actual_ref_wl = wl.iloc[ref_idx]
            
            print(f"   Found maximum at {actual_ref_wl:.1f}nm = {ref_value:.2f}")
            
            if ref_value <= 0:
                print("Cannot normalize - maximum intensity is zero or negative")
                return None
            
            # Step 1: Scale to 50,000
            target_ref_intensity = 50000.0
            scaling_factor = target_ref_intensity / ref_value
            scaled_intensities = intens * scaling_factor
            
            # Step 2: Scale to 0-100 range
            min_val, max_val = np.min(scaled_intensities), np.max(scaled_intensities)
            range_val = max_val - min_val
            
            if range_val > 0:
                normalized = ((scaled_intensities - min_val) / range_val) * 100.0
                self.spectrum_df.iloc[:, 1] = normalized
                
                final_ref = normalized[ref_idx]
                final_max, final_min = np.max(normalized), np.min(normalized)
                
                print(f"   Step 1: Scale maximum to {target_ref_intensity:.0f}")
                print(f"   Step 2: Scale to 0-100 range ({final_min:.2f} - {final_max:.2f})")
                print(f"   Maximum final intensity: {final_ref:.2f}")
                print("LASER NORMALIZATION COMPLETE")
                
                return {
                    'method': 'laser_max_50000_to_100',
                    'reference_wavelength': actual_ref_wl,
                    'original_intensity': ref_value,
                    'final_intensity': final_ref,
                    'scaling_factor': scaling_factor,
                    'target_ref_intensity': target_ref_intensity,
                    'final_range_min': final_min,
                    'final_range_max': final_max,
                    'normalization_scheme': 'Laser_Max_50000_to_100'
                }
            else:
                print("Cannot normalize - intensity range is zero")
                return None
                
        except Exception as e:
            print(f"FIXED normalization error: {e}")
            return None
    
    def update_plot(self, title_suffix):
        """OPTIMIZED: Update plot with proper scaling"""
        wl, intens = self.spectrum_df.iloc[:, 0], self.spectrum_df.iloc[:, 1]
        
        self.ax.clear()
        self.ax.plot(wl, intens, 'g-', linewidth=0.8)  # Green for laser
        self.ax.set_title(f"FIXED Laser Analysis ({title_suffix}) - {self.filename}")
        self.ax.set_xlabel("Wavelength (nm)")
        
        # Set Y-axis based on processing state
        if "Normalized" in title_suffix:
            self.ax.set_ylabel("Intensity (0-100 scale)")
            self.ax.set_ylim(-5, 105)
            print("FIXED DISPLAY: Y-axis set to -5 to 105 for 0-100 scale")
        else:
            self.ax.set_ylabel("Raw Intensity")
            if len(intens) > 0:
                y_min, y_max = intens.min(), intens.max()
                y_range = y_max - y_min
                padding = 0.05 * y_range if y_range > 0 else 5
                self.ax.set_ylim(y_min - padding, y_max + padding)
        
        self.ax.grid(True, alpha=0.3)
        self.clear_visual_markers()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def clear_visual_markers(self):
        """OPTIMIZED: Clear all visual markers"""
        if hasattr(self, 'lines_drawn') and self.lines_drawn:
            for item in self.lines_drawn:
                try:
                    if hasattr(item, 'remove'): 
                        item.remove()
                    elif hasattr(item, 'set_visible'): 
                        item.set_visible(False)
                    else:
                        item.set_offsets([])
                except:
                    pass
            self.lines_drawn.clear()
        if not hasattr(self, 'lines_drawn'):
            self.lines_drawn = []
    
    def get_intensity_at_wavelength(self, target_wl):
        """Get precise intensity at wavelength"""
        try:
            wl, intens = self.spectrum_df.iloc[:, 0], self.spectrum_df.iloc[:, 1]
            return intens.iloc[wl[wl == target_wl].index[0]] if target_wl in wl.values else np.interp(target_wl, wl, intens)
        except:
            return None
    
    def calculate_mound_symmetry(self, start_wl, crest_wl, end_wl):
        """Calculate mound symmetry ratio"""
        total_width = end_wl - start_wl
        if total_width == 0: 
            return 0.0, "Invalid"
        
        left_width = crest_wl - start_wl
        right_width = end_wl - crest_wl
        if right_width == 0: 
            return float('inf'), "Extreme Right Skew"
        
        ratio = left_width / right_width
        desc = "Left Skewed" if ratio < 0.8 else "Right Skewed" if ratio > 1.25 else "Symmetric"
        return round(ratio, 3), desc
    
    def select_feature(self, feature_type):
        """OPTIMIZED: Select feature type for marking"""
        if hasattr(self, 'feature_ready') and self.feature_ready and self.clicks:
            print(f"Auto-completing previous {self.current_type}")
            self.complete_feature()
        
        self.current_type = feature_type
        self.clicks = []
        self.feature_ready = False
        
        expected_points = self.feature_config[feature_type]['points']
        print(f"Selected: {feature_type} - need {len(expected_points)} points")
        
        if self.fig: 
            self.fig.canvas.draw()
    
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
                    print(f"🔍 Using input directory (no txt files found): {self.input_directory}")
            else:
                initial_dir = str(self.project_root)
                print(f"🔍 Input directory not found, using project root: {self.project_root}")
            
            file_path = filedialog.askopenfilename(
                parent=root, 
                initialdir=initial_dir,
                title=f"Select Gem Spectrum for FIXED Laser Analysis\nLooking in: {Path(initial_dir).name}",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            
            root.quit()
            root.destroy()
            
            if file_path:
                print(f"✅ Selected file: {Path(file_path).name}")
                print(f"   Full path: {file_path}")
            
            return file_path if file_path else None
            
        except Exception as e:
            print(f"File dialog error: {e}")
            return None
    
    def save_features(self):
        """UPDATED: Save features with normalization metadata to new directory structure"""
        if hasattr(self, 'feature_ready') and self.feature_ready and self.clicks:
            print(f"Completing {self.current_type} before saving...")
            self.complete_feature()
        
        if not self.features:
            print("No features to save")
            return "continue"
            
        try:
            df = pd.DataFrame(self.features)
            
            # Add normalization metadata columns
            metadata_cols = {
                'Normalization_Scheme': self.normalization_info['normalization_scheme'] if self.normalization_info else 'Raw_Data',
                'Reference_Wavelength': self.normalization_info['reference_wavelength'] if self.normalization_info else None,
                'Light_Source': 'Laser',
                'Intensity_Range_Min': self.normalization_info['final_range_min'] if self.normalization_info else None,
                'Intensity_Range_Max': self.normalization_info['final_range_max'] if self.normalization_info else None,
                'Directory_Structure': 'Updated_New_Structure',
                'Output_Location': str(self.output_directory)
            }
            
            for col, value in metadata_cols.items():
                df[col] = value
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = self.filename.replace('.txt', '')
            outname = f"{base_name}_laser_structural_{timestamp}.csv"
            
            # Ensure output directory exists
            self.output_directory.mkdir(parents=True, exist_ok=True)
            full_path = self.output_directory / outname
            
            df.to_csv(full_path, index=False)
            
            print(f"✅ FIXED: Saved {len(self.features)} features with enhanced metadata")
            print(f"🔍 Output directory: {self.output_directory}")
            print(f"📄 File: {outname}")
            
            if self.normalization_info:
                scheme = self.normalization_info['normalization_scheme']
                ref_wl = self.normalization_info['reference_wavelength']
                print(f"🔧 Normalization: {scheme} (max at {ref_wl:.1f}nm)")
            
            return self.ask_next_action()
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            print(f"   Attempted to save to: {self.output_directory}")
            return "continue"
    
    def ask_next_action(self):
        """Ask user for next action with updated messaging"""
        try:
            result = messagebox.askyesnocancel(
                "FIXED Analysis Complete",
                f"✅ FIXED Laser analysis saved for: {self.filename}\n"
                f"🔍 Location: {self.output_directory.name}/\n"
                f"🔧 Includes enhanced metadata for new directory structure\n\n"
                f"YES = Analyze another gem with laser\n"
                f"NO = Close this analyzer (return to launcher)\n"
                f"CANCEL = Exit completely")
            
            if result is True:
                plt.close('all')
                self.reset_session()
                print("Ready for next FIXED laser analysis...")
                return "continue"
            elif result is False:
                plt.close('all')
                print("Closing FIXED laser analyzer - launcher should still be open...")
                import sys
                sys.exit(0)
            else:
                plt.close('all')
                print("Exiting completely...")
                import sys
                sys.exit(1)
                
        except Exception as e:
            print(f"Dialog error: {e}")
            plt.close('all')
            import sys
            sys.exit(0)
    
    def create_laser_ui(self):
        """OPTIMIZED: Create UI buttons"""
        self.buttons = []
        bw, bh, bx, bs = 0.12, 0.035, 0.845, 0.045
        
        for i, (label, color, action) in enumerate(self.button_config):
            y_pos = 0.92 - (i * bs)
            if y_pos <= 0.05:
                break
                
            ax_btn = self.fig.add_axes([bx, y_pos, bw, bh])
            btn = Button(ax_btn, label, color=color)
            
            if action in self.feature_config:
                btn.on_clicked(lambda e, ft=action: self.select_feature(ft))
            elif action == 'undo':
                btn.on_clicked(self.undo_last)
            elif action == 'save':
                btn.on_clicked(lambda e: self.save_features())
            elif action == 'persistent':
                btn.on_clicked(self.toggle_persistent)
            
            self.buttons.append(btn)
        
        return self.buttons
    
    def undo_last(self, event):
        """OPTIMIZED: Undo last action"""
        try:
            if self.clicks and self.current_type:
                removed_click = self.clicks.pop()
                if self.lines_drawn:
                    try:
                        last_dot = self.lines_drawn.pop()
                        if hasattr(last_dot, 'remove'):
                            last_dot.remove()
                    except:
                        pass
                
                expected = len(self.feature_config[self.current_type]['points'])
                remaining = expected - len(self.clicks)
                print(f"Undid click: {removed_click[0]:.2f}nm - {remaining} more needed")
                
                if len(self.clicks) < expected:
                    self.feature_ready = False
                    
            elif self.features:
                last_feature = self.features[-1] if self.features else None
                last_group = last_feature.get('Feature_Group', '') if last_feature else ''
                
                if last_group:
                    self.features = [f for f in self.features if f.get('Feature_Group', '') != last_group]
                    print(f"Removed completed feature: {last_group}")
                    self.clear_visual_markers()
                
                if last_group == 'Baseline':
                    self.baseline_data = None
                    self.normalization_applied = False
                    self.normalization_info = None
                    if hasattr(self, 'original_spectrum_df') and self.original_spectrum_df is not None:
                        self.spectrum_df = self.original_spectrum_df.copy()
                        self.update_plot("Original")
            else:
                print("Nothing to undo")
                return
            
            if self.fig and self.ax:
                self.fig.canvas.draw_idle()
                
        except Exception as e:
            print(f"Undo error: {e}")
            self.clear_visual_markers()
    
    def toggle_persistent(self, event):
        """Toggle persistent mode"""
        self.persistent_mode = not self.persistent_mode
        print(f"PERSISTENT: {'ON' if self.persistent_mode else 'OFF'}")
    
    def onclick(self, event):
        """OPTIMIZED: Handle mouse clicks"""
        # Check toolbar state
        if (hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar and 
            hasattr(self.fig.canvas.toolbar, 'mode') and 
            self.fig.canvas.toolbar.mode in ['zoom rect', 'pan']):
            print(f"TOOLBAR {self.fig.canvas.toolbar.mode.upper()} MODE ACTIVE")
            return
        
        if not self.current_type:
            print("Select feature first! (B for baseline, 1-7 for structures)")
            return
        
        if event.inaxes != self.ax or event.xdata is None:
            return

        wl = event.xdata
        intens = event.ydata
        precise_int = self.get_intensity_at_wavelength(wl)
        if precise_int is None:
            precise_int = intens
        
        self.clicks.append((wl, precise_int))
        
        color = self.feature_config[self.current_type]['color']
        dot = self.ax.scatter(wl, intens, c=color, s=35, marker='o', 
                             edgecolors='black', linewidth=1.5, zorder=10)
        
        if not hasattr(self, 'lines_drawn') or self.lines_drawn is None:
            self.lines_drawn = []
        self.lines_drawn.append(dot)
        
        expected = len(self.feature_config[self.current_type]['points'])
        current = len(self.clicks)
        
        print(f"{self.current_type} {current}/{expected}: {wl:.2f}nm, intensity: {precise_int:.2f}")
        
        if current == expected:
            print(f"{self.current_type} READY - Press S to save or U to undo")
            self.feature_ready = True
        else:
            print(f"   Need {expected - current} more clicks")
            self.feature_ready = False
        
        self.fig.canvas.draw_idle()
    
    def complete_feature(self):
        """OPTIMIZED: Complete feature marking"""
        if not self.clicks or not self.current_type:
            return
        
        self.feature_ready = False
        feature_key = f"{self.current_type}_{len(self.features)}"
        
        # Handle baseline specially
        if self.current_type == 'Baseline':
            if len(self.clicks) == 2:
                start_wl = min(self.clicks[0][0], self.clicks[1][0])
                end_wl = max(self.clicks[0][0], self.clicks[1][0])
                
                baseline_stats = self.calculate_baseline_stats(start_wl, end_wl)
                if baseline_stats:
                    self.baseline_data = baseline_stats
                    self.normalization_applied = False
                    self.normalization_info = None
                    
                    print("Starting FIXED laser processing...")
                    if self.apply_processing_pipeline():
                        # Add baseline features with metadata
                        for i, (wl, intens) in enumerate(self.clicks):
                            pt_type = 'Start' if i == 0 else 'End'
                            entry = {
                                'Feature': f'Baseline_{pt_type}', 'File': self.filename,
                                'Light_Source': 'Laser', 'Wavelength': round(wl, 2),
                                'Intensity': round(intens, 2), 'Point_Type': pt_type,
                                'Feature_Group': 'Baseline', 'Processing': 'Baseline_Then_Laser_Normalized',
                                'SNR': baseline_stats['snr'], 'Feature_Key': feature_key,
                                'Directory_Structure': 'Updated_New_Structure'
                            }
                            
                            if self.normalization_info:
                                entry.update({
                                    'Baseline_Used': baseline_stats['avg_intensity'],
                                    'Norm_Factor': self.normalization_info['scaling_factor'],
                                    'Normalization_Method': self.normalization_info['method']
                                })
                            
                            self.features.append(entry)
                        
                        print(f"BASELINE: {start_wl:.1f}-{end_wl:.1f}nm, SNR: {baseline_stats['snr']:.1f}")
                else:
                    print("Failed to calculate baseline statistics")
                    return
            else:
                print(f"Baseline needs exactly 2 clicks, got {len(self.clicks)}")
                return
        else:
            # Handle all other features
            labels = self.feature_config[self.current_type]['points']
            for i, (wl, intens) in enumerate(self.clicks):
                label = labels[i]
                entry = {
                    'Feature': f'{self.current_type}_{label}', 'File': self.filename,
                    'Light_Source': 'Laser', 'Wavelength': round(wl, 2),
                    'Intensity': round(intens, 2), 'Point_Type': label,
                    'Feature_Group': self.current_type, 'Feature_Key': feature_key,
                    'Directory_Structure': 'Updated_New_Structure'
                }
                
                if self.baseline_data and self.normalization_applied:
                    entry.update({
                        'Processing': 'Baseline_Then_Laser_Normalized',
                        'Baseline_Used': self.baseline_data['avg_intensity'],
                        'Norm_Factor': self.normalization_info['scaling_factor'],
                        'Normalization_Method': self.normalization_info['method'],
                        'Reference_Wavelength_Used': self.normalization_info['reference_wavelength']
                    })
                
                self.features.append(entry)
            
            # Add mound summary if applicable
            if self.current_type == 'Mound' and len(self.clicks) == 3:
                s_wl, s_int = self.clicks[0]
                c_wl, c_int = self.clicks[1]
                e_wl, e_int = self.clicks[2]
                ratio, desc = self.calculate_mound_symmetry(s_wl, c_wl, e_wl)
                
                summary = {
                    'Feature': 'Mound_Summary', 'File': self.filename,
                    'Light_Source': 'Laser', 'Wavelength': round(c_wl, 2),
                    'Intensity': round(c_int, 2), 'Point_Type': 'Summary',
                    'Feature_Group': 'Mound', 'Symmetry_Ratio': ratio,
                    'Skew_Description': desc, 'Width_nm': round(e_wl - s_wl, 2),
                    'Feature_Key': feature_key, 'Directory_Structure': 'Updated_New_Structure'
                }
                
                if self.normalization_applied and self.normalization_info:
                    summary.update({
                        'Processing': 'Baseline_Then_Laser_Normalized',
                        'Normalization_Method': self.normalization_info['method']
                    })
                
                self.features.append(summary)
        
        self.clicks.clear()
        self.lines_drawn = []
        if not self.persistent_mode:
            self.current_type = None
        print(f"SAVED {self.current_type or 'feature'} to feature list")
    
    def onkey(self, event):
        """OPTIMIZED: Handle keyboard events"""
        # Create reverse key mapping
        key_map = {config['key']: feature for feature, config in self.feature_config.items()}
        
        if event.key in key_map:
            self.select_feature(key_map[event.key])
        elif event.key == 's':
            self.save_features()
        elif event.key in ['enter', 'return']:
            if self.current_type in ['Peak', 'Valley'] and len(self.clicks) > 0:
                print(f"Manual completion of {self.current_type}")
                self.complete_feature()
        elif event.key == 'p':
            self.toggle_persistent(event)
        elif event.key == 'u':
            self.undo_last(event)
    
    def run_analysis(self):
        """UPDATED: Main analysis loop with enhanced directory information"""
        print("="*70)
        print("GEMINI LASER ANALYZER - FIXED FOR NEW DIRECTORY STRUCTURE")
        print("="*70)
        print(f"🔍 Project root: {self.project_root}")
        print(f"📂 Input directory: {self.input_directory}")
        print(f"🔍 Output directory: {self.output_directory}")
        print("="*70)
        print("FIXED: Corrected path detection to reach true project root")
        print("FIXED: Maximum intensity → 50,000, then scale 0-100")
        print("Workflow: Mark baseline → Auto-process → Mark features")
        print("Use TOOLBAR MAGNIFYING GLASS for zoom")
        
        while True:
            file_path = self.file_selection_dialog()
            if not file_path: 
                print("No file selected - exiting analyzer")
                break
            
            self.filename = os.path.basename(file_path)
            self.spectrum_df, load_info = self.load_spectrum_file(file_path)
            
            if self.spectrum_df is None:
                print(f"Error: {load_info}")
                break
            
            print(f"Loaded: {self.filename} ({len(self.spectrum_df)} points)")
            
            plt.close('all')
            self.fig, self.ax = plt.subplots(figsize=(13, 7))
            
            wl, intens = self.spectrum_df.iloc[:, 0], self.spectrum_df.iloc[:, 1]
            self.ax.plot(wl, intens, 'g-', linewidth=0.8)  # Green for laser
            self.ax.set_title(f"FIXED Laser Analysis - {self.filename}")
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("Raw Intensity")
            self.ax.grid(True, alpha=0.3)
            
            self.create_laser_ui()
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onkey)
            
            plt.subplots_adjust(right=0.82)
            print("READY! Mark baseline first (B key), then features (1-7)")
            print("FIXED normalization will be applied automatically after baseline")
            print(f"Results will be saved to: {self.output_directory}")
            
            plt.show()
            print("Analysis window closed")

def main():
    analyzer = GeminiLaserAnalyzer()
    analyzer.run_analysis()

if __name__ == '__main__':
    main()