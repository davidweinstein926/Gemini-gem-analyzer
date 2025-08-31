# gemini_uv_analyzer.py - FIXED NORMALIZATION FOR 0-100 COMPATIBILITY
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime

class GeminiUVAnalyzer:
    def __init__(self):
        self.OUTPUT_DIRECTORY = r"c:\users\david\onedrive\desktop\gemini_gemological_analysis\data\structural_data\uv"
        self.reset_session()
        
    def reset_session(self):
        self.features = []
        self.current_type = None
        self.persistent_mode = True
        self.clicks = []
        self.filename = ""
        self.lines_drawn = []
        self.baseline_data = None
        self.spectrum_df = None
        self.ax = None
        self.fig = None
        self.buttons = []
        self.normalization_applied = False
        self.normalization_info = None
        self.feature_ready = False
        self.completed_feature_visuals = {}
        print("Session reset - ready for new spectrum")
        
    def get_structure_types(self):
        return {'Baseline': ['Start', 'End'], 'Mound': ['Start', 'Crest', 'End'],
                'Plateau': ['Start', 'Midpoint', 'End'], 'Peak': ['Max'],
                'Trough': ['Start', 'Bottom', 'End'], 'Shoulder': ['Start', 'Peak', 'End'],
                'Valley': ['Midpoint'], 'Diagnostic Region': ['Start', 'End']}
    
    def get_feature_colors(self):
        return {'Baseline': 'gray', 'Mound': 'red', 'Plateau': 'green', 'Peak': 'blue',
                'Trough': 'purple', 'Shoulder': 'orange', 'Valley': 'brown', 'Diagnostic Region': 'gold'}
    
    def load_spectrum_file(self, file_path):
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
        try:
            wl = self.spectrum_df.iloc[:, 0]
            intens = self.spectrum_df.iloc[:, 1]
            mask = (wl >= start_wl) & (wl <= end_wl)
            baseline_int = intens[mask]
            if len(baseline_int) < 3: return None
            avg_int = np.mean(baseline_int)
            std_dev = np.std(baseline_int)
            snr = avg_int / std_dev if std_dev > 0 else float('inf')
            return {'wavelength_start': start_wl, 'wavelength_end': end_wl,
                   'avg_intensity': round(avg_int, 2), 'std_deviation': round(std_dev, 3),
                   'snr': round(snr, 1), 'data_points': len(baseline_int)}
        except: return None
    
    def apply_baseline_correction_to_spectrum(self):
        if not self.baseline_data:
            print("No baseline established")
            return False
        try:
            baseline_avg = self.baseline_data['avg_intensity']
            print(f"Applying baseline correction: subtracting {baseline_avg:.2f}")
            corrected = self.spectrum_df.iloc[:, 1] - baseline_avg
            self.spectrum_df.iloc[:, 1] = corrected.clip(lower=0)
            print(f"BASELINE CORRECTION APPLIED: -{baseline_avg:.2f}")
            self.update_plot("Baseline Corrected")
            return True
        except Exception as e:
            print(f"Baseline correction error: {e}")
            return False

    def normalize_uv_spectrum_fixed(self):
        """FIXED: UV normalization - 811nm → 15,000, then scale 0-100"""
        try:
            wl = self.spectrum_df.iloc[:, 0]
            intens = self.spectrum_df.iloc[:, 1].values.copy()
            
            # FIXED: Find 811nm reference peak for UV
            ref_wl = 811.0
            tolerance = 5.0  # Tolerance for UV peak finding
            
            print(f"FIXED UV NORMALIZATION:")
            print(f"   Looking for 811nm reference peak (±{tolerance}nm)")
            
            # Find 811nm reference peak
            ref_mask = np.abs(wl - ref_wl) <= tolerance
            if np.any(ref_mask):
                ref_value = np.max(intens[ref_mask])
                ref_idx = np.where(ref_mask)[0][np.argmax(intens[ref_mask])]
                actual_ref_wl = wl.iloc[ref_idx]
                print(f"   Found reference at {actual_ref_wl:.1f}nm = {ref_value:.2f}")
            else:
                print(f"   WARNING: No 811nm peak found, using maximum intensity")
                ref_value = np.max(intens)
                ref_idx = np.argmax(intens)
                actual_ref_wl = wl.iloc[ref_idx]
                print(f"   Using maximum at {actual_ref_wl:.1f}nm = {ref_value:.2f}")
            
            if ref_value <= 0:
                print("Cannot normalize - reference intensity is zero or negative")
                return None
            
            # FIXED: Step 1 - Scale 811nm to 15,000
            target_ref_intensity = 15000.0
            scaling_factor = target_ref_intensity / ref_value
            scaled_intensities = intens * scaling_factor
            
            print(f"   Step 1: Scale 811nm reference to {target_ref_intensity:.0f}")
            print(f"   Scaling factor: {scaling_factor:.6f}")
            print(f"   Reference after scaling: {scaled_intensities[ref_idx]:.1f}")
            
            # FIXED: Step 2 - Scale entire spectrum to 0-100 range
            min_val = np.min(scaled_intensities)
            max_val = np.max(scaled_intensities)
            range_val = max_val - min_val
            
            if range_val > 0:
                normalized = ((scaled_intensities - min_val) / range_val) * 100.0
                self.spectrum_df.iloc[:, 1] = normalized
                
                final_ref = normalized[ref_idx]
                final_max = np.max(normalized)
                final_min = np.min(normalized)
                
                print(f"   Step 2: Scale to 0-100 range")
                print(f"   Final range: {final_min:.2f} - {final_max:.2f}")
                print(f"   811nm final intensity: {final_ref:.2f}")
                print(f"UV NORMALIZATION COMPLETE")
                
                self.update_plot("FIXED Normalized (0-100)")
                
                return {
                    'method': 'uv_811nm_15000_to_100',
                    'reference_wavelength': actual_ref_wl,
                    'original_intensity': ref_value,
                    'intermediate_intensity': scaled_intensities[ref_idx],
                    'final_intensity': final_ref,
                    'scaling_factor': scaling_factor,
                    'target_ref_intensity': target_ref_intensity,
                    'final_range_min': final_min,
                    'final_range_max': final_max,
                    'normalization_scheme': 'UV_811nm_15000_to_100'
                }
            else:
                print("Cannot normalize - intensity range is zero")
                return None
                
        except Exception as e:
            print(f"FIXED normalization error: {e}")
            return None
    
    def clear_all_visual_markers(self):
        try:
            if hasattr(self, 'lines_drawn') and self.lines_drawn:
                for item in self.lines_drawn:
                    try:
                        if hasattr(item, 'remove'): item.remove()
                        elif hasattr(item, 'set_visible'): item.set_visible(False)
                        else:
                            try: item.set_offsets([])
                            except: pass
                    except: pass
                self.lines_drawn.clear()
            if not hasattr(self, 'lines_drawn'):
                self.lines_drawn = []
            print("Cleared all visual markers")
        except Exception as e:
            print(f"Error clearing visual markers: {e}")
            self.lines_drawn = []

    def update_plot(self, title_suffix):
        """FIXED: Update plot with 0-100 scale awareness"""
        wl = self.spectrum_df.iloc[:, 0]
        intens = self.spectrum_df.iloc[:, 1]
        
        # Clear and redraw
        self.ax.clear()
        self.ax.plot(wl, intens, 'b-', linewidth=0.8)  # Blue for UV
        self.ax.set_title(f"FIXED UV Analysis ({title_suffix}) - {self.filename}")
        self.ax.set_xlabel("Wavelength (nm)")
        
        # FIXED: Update Y-axis label based on processing state
        if "Normalized" in title_suffix:
            self.ax.set_ylabel("Intensity (0-100 scale)")
        else:
            self.ax.set_ylabel("Raw Intensity")
            
        self.ax.grid(True, alpha=0.3)
        
        # FIXED: Proper Y-axis scaling
        if len(intens) > 0:
            y_min = intens.min()
            y_max = intens.max()
            y_range = y_max - y_min
            
            # FIXED: Better padding for 0-100 scale
            if "Normalized" in title_suffix and y_max <= 100:
                # Use fixed range for normalized data
                self.ax.set_ylim(-5, 105)
                print(f"FIXED DISPLAY: Y-axis set to -5 to 105 for 0-100 scale")
            else:
                # Dynamic range for raw data
                padding = 0.05 * y_range if y_range > 0 else 5
                self.ax.set_ylim(y_min - padding, y_max + padding)
                print(f"UV DISPLAY: Y-axis set to {y_min:.1f} - {y_max:.1f}")
        
        # Clear visual markers tracking
        if hasattr(self, 'lines_drawn'):
            self.lines_drawn.clear()
        else:
            self.lines_drawn = []
        
        # Force canvas redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def get_intensity_at_wavelength(self, target_wl):
        try:
            wl = self.spectrum_df.iloc[:, 0]
            intens = self.spectrum_df.iloc[:, 1]
            if target_wl in wl.values:
                idx = wl[wl == target_wl].index[0]
                return intens.iloc[idx]
            else:
                return np.interp(target_wl, wl, intens)
        except: return None
    
    def calculate_mound_symmetry(self, start_wl, crest_wl, end_wl):
        total_width = end_wl - start_wl
        if total_width == 0: return 0.0, "Invalid"
        left_width = crest_wl - start_wl
        right_width = end_wl - crest_wl
        if right_width == 0: return float('inf'), "Extreme Right Skew"
        ratio = left_width / right_width
        desc = "Left Skewed" if ratio < 0.8 else "Right Skewed" if ratio > 1.25 else "Symmetric"
        return round(ratio, 3), desc
    
    def select_feature(self, feature_type):
        if hasattr(self, 'feature_ready') and self.feature_ready and self.clicks:
            print(f"Auto-completing previous {self.current_type}")
            self.complete_feature()
        self.current_type = feature_type
        self.clicks = []
        self.feature_ready = False
        expected_points = self.get_structure_types()[feature_type]
        print(f"Selected: {feature_type} - need {len(expected_points)} points")
        if self.fig: self.fig.canvas.draw()
    
    def file_selection_dialog(self):
        """No more loop - single dialog attempt"""
        default_dir = r"C:\Users\David\OneDrive\Desktop\gemini matcher\gemini sp10 raw\raw text"
        try:
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            file_path = filedialog.askopenfilename(
                parent=root, initialdir=default_dir,
                title="Select Gem Spectrum for FIXED UV Analysis",
                filetypes=[("Text files", "*.txt")])
            root.quit()
            root.destroy()
            
            return file_path if file_path else None
            
        except Exception as e:
            print(f"File dialog error: {e}")
            return None
    
    def save_features(self):
        """FIXED: Save with normalization metadata for database compatibility"""
        if hasattr(self, 'feature_ready') and self.feature_ready and self.clicks:
            print(f"Completing {self.current_type} before saving...")
            self.complete_feature()
        if not self.features:
            print("No features to save")
            return "continue"
            
        try:
            # FIXED: Create DataFrame with normalization metadata
            df = pd.DataFrame(self.features)
            
            # FIXED: Add normalization metadata columns for database compatibility
            if self.normalization_info:
                df['Normalization_Scheme'] = self.normalization_info['normalization_scheme']
                df['Reference_Wavelength'] = self.normalization_info['reference_wavelength']
                df['Light_Source'] = 'UV'
                df['Intensity_Range_Min'] = self.normalization_info['final_range_min']
                df['Intensity_Range_Max'] = self.normalization_info['final_range_max']
            else:
                df['Normalization_Scheme'] = 'Raw_Data'
                df['Reference_Wavelength'] = None
                df['Light_Source'] = 'UV'
                df['Intensity_Range_Min'] = None
                df['Intensity_Range_Max'] = None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = self.filename.replace('.txt', '')
            outname = f"{base_name}_uv_structural_{timestamp}.csv"
            os.makedirs(self.OUTPUT_DIRECTORY, exist_ok=True)
            full_path = os.path.join(self.OUTPUT_DIRECTORY, outname)
            df.to_csv(full_path, index=False)
            
            print(f"FIXED: Saved {len(self.features)} features with normalization metadata")
            print(f"File: {outname}")
            
            # FIXED: Display normalization info in save message
            if self.normalization_info:
                scheme = self.normalization_info['normalization_scheme']
                ref_wl = self.normalization_info['reference_wavelength']
                print(f"Normalization: {scheme} (ref: {ref_wl:.1f}nm)")
            
            return self.ask_next_action()
            
        except Exception as e:
            print(f"Save error: {e}")
            return "continue"
    
    def ask_next_action(self):
        """Return action instead of recursive call"""
        try:
            result = messagebox.askyesnocancel(
                "FIXED Analysis Complete",
                f"FIXED UV analysis saved for: {self.filename}\n"
                f"Includes normalization metadata for database compatibility\n\n"
                f"YES = Analyze another gem with UV\n"
                f"NO = Close this analyzer (return to launcher)\n"
                f"CANCEL = Exit completely")
            
            if result is True:
                plt.close('all')
                self.reset_session()
                print("Ready for next FIXED UV analysis...")
                return "continue"
            elif result is False:
                plt.close('all')
                print("Closing FIXED UV analyzer - launcher should still be open...")
                print("Switch to the launcher window to select another analyzer")
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
    
    def create_uv_ui(self):
        self.buttons = []
        bw, bh, bx = 0.12, 0.035, 0.845
        bs = 0.045
        
        # Baseline
        ax_btn = self.fig.add_axes([bx, 0.92, bw, bh])
        btn = Button(ax_btn, 'Baseline\n(B)', color='lightgray')
        btn.on_clicked(lambda e: self.select_feature('Baseline'))
        self.buttons.append(btn)
        
        # Features
        features = [('Mound', 'lightcoral', '1'), ('Plateau', 'lightgreen', '2'),
                   ('Peak', 'lightblue', '3'), ('Trough', 'plum', '4'),
                   ('Shoulder', 'moccasin', '5'), ('Valley', 'burlywood', '6'),
                   ('Diagnostic Region', 'lightyellow', '7')]
        
        for i, (name, color, key) in enumerate(features):
            y_pos = 0.87 - (i * bs)
            ax_btn = self.fig.add_axes([bx, y_pos, bw, bh])
            btn = Button(ax_btn, f'{name}\n({key})', color=color)
            btn.on_clicked(lambda e, ft=name: self.select_feature(ft))
            self.buttons.append(btn)
        
        # Utilities
        utility_y = 0.87 - (len(features) * bs) - 0.04
        utilities = [('Undo\n(U)', 'mistyrose', self.undo_last),
                    ('Save\n(S)', 'lightcyan', lambda e: self.save_features()),
                    ('Persistent\n(P)', 'lavender', self.toggle_persistent)]
        
        for i, (label, color, callback) in enumerate(utilities):
            y_pos = utility_y - (i * bs)
            if y_pos > 0.05:
                ax_btn = self.fig.add_axes([bx, y_pos, bw, bh])
                btn = Button(ax_btn, label, color=color)
                btn.on_clicked(callback)
                self.buttons.append(btn)
        return self.buttons
    
    def undo_last(self, event):
        try:
            if self.clicks and self.current_type:
                removed_click = self.clicks.pop()
                if self.lines_drawn and len(self.lines_drawn) > 0:
                    try:
                        last_dot = self.lines_drawn.pop()
                        self.remove_visual_element(last_dot)
                    except: pass
                expected = len(self.get_structure_types().get(self.current_type, []))
                remaining = expected - len(self.clicks)
                print(f"Undid click: {removed_click[0]:.2f}nm - {remaining} more needed")
                if len(self.clicks) < expected: self.feature_ready = False
            elif self.features:
                last_feature = self.features[-1] if self.features else None
                last_group = last_feature.get('Feature_Group', '') if last_feature else ''
                last_key = last_feature.get('Feature_Key', '') if last_feature else ''
                if last_group:
                    self.features = [f for f in self.features if f.get('Feature_Group', '') != last_group]
                    print(f"Removed completed feature: {last_group}")
                    if last_key and last_key in self.completed_feature_visuals:
                        for element in self.completed_feature_visuals[last_key]:
                            self.remove_visual_element(element)
                        del self.completed_feature_visuals[last_key]
                    else:
                        self.clear_all_visual_markers()
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
            if self.fig and self.ax: self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Undo error: {e}")
            self.force_visual_resync()

    def remove_visual_element(self, element):
        try:
            if hasattr(element, 'remove'):
                element.remove()
                return True
        except: pass
        try:
            if hasattr(element, 'set_visible'):
                element.set_visible(False)
                return True
        except: pass
        try:
            if hasattr(element, 'set_offsets'):
                element.set_offsets(np.empty((0, 2)))
                return True
        except: pass
        return False

    def force_visual_resync(self):
        try:
            self.clear_all_visual_markers()
            if self.clicks and self.current_type:
                color = self.get_feature_colors().get(self.current_type, 'black')
                for wl, intens in self.clicks:
                    dot = self.ax.scatter(wl, intens, c=color, s=35, 
                                         marker='o', edgecolors='black', linewidth=1.5, zorder=10)
                    self.lines_drawn.append(dot)
                print(f"Rebuilt: {len(self.clicks)} clicks")
            if self.fig: self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Force resync error: {e}")
            self.lines_drawn = []
    
    def toggle_persistent(self, event):
        self.persistent_mode = not self.persistent_mode
        print(f"PERSISTENT: {'ON' if self.persistent_mode else 'OFF'}")
    
    def onclick(self, event):
        # Check toolbar zoom/pan
        if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar:
            toolbar = self.fig.canvas.toolbar
            if hasattr(toolbar, 'mode') and toolbar.mode in ['zoom rect', 'pan']:
                print(f"TOOLBAR {toolbar.mode.upper()} MODE ACTIVE")
                return
        
        if not self.current_type:
            print("Select feature first! (B for baseline, 1-7 for structures)")
            return
        if event.inaxes != self.ax or event.xdata is None: return

        wl = event.xdata
        intens = event.ydata
        precise_int = self.get_intensity_at_wavelength(wl)
        if precise_int is None: precise_int = intens
        
        self.clicks.append((wl, precise_int))
        color = self.get_feature_colors().get(self.current_type, 'black')
        dot = self.ax.scatter(wl, intens, c=color, s=35, 
                             marker='o', edgecolors='black', linewidth=1.5, zorder=10)
        
        if not hasattr(self, 'lines_drawn') or self.lines_drawn is None:
            self.lines_drawn = []
        self.lines_drawn.append(dot)
        
        expected = len(self.get_structure_types()[self.current_type])
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
        if not self.clicks or not self.current_type: return
        
        self.feature_ready = False
        feature_key = f"{self.current_type}_{len(self.features)}"
        if self.lines_drawn:
            self.completed_feature_visuals[feature_key] = self.lines_drawn.copy()
        
        # FIXED: Handle baseline with proper UV normalization
        if self.current_type == 'Baseline':
            if len(self.clicks) == 2:
                start_wl = min(self.clicks[0][0], self.clicks[1][0])
                end_wl = max(self.clicks[0][0], self.clicks[1][0])
                
                baseline_stats = self.calculate_baseline_stats(start_wl, end_wl)
                if baseline_stats:
                    self.baseline_data = baseline_stats
                    self.normalization_applied = False
                    self.normalization_info = None
                    
                    print(f"Starting FIXED UV processing...")
                    if self.apply_baseline_correction_to_spectrum():
                        # FIXED: Use proper UV normalization
                        normalization_info = self.normalize_uv_spectrum_fixed()
                        if normalization_info:
                            self.normalization_applied = True
                            self.normalization_info = normalization_info
                            print("FIXED processing complete: Baseline + UV Normalized")
                    
                    for i, (wl, intens) in enumerate(self.clicks):
                        pt_type = 'Start' if i == 0 else 'End'
                        entry = {'Feature': f'Baseline_{pt_type}', 'File': self.filename,
                               'Light_Source': 'UV', 'Wavelength': round(wl, 2),
                               'Intensity': round(intens, 2), 'Point_Type': pt_type,
                               'Feature_Group': 'Baseline', 'Processing': 'Baseline_Then_UV_Normalized',
                               'SNR': baseline_stats['snr'], 'Feature_Key': feature_key}
                        
                        # FIXED: Add normalization metadata to baseline features
                        if self.normalization_info:
                            entry['Baseline_Used'] = baseline_stats['avg_intensity']
                            entry['Norm_Factor'] = self.normalization_info['scaling_factor']
                            entry['Normalization_Method'] = self.normalization_info['method']
                        
                        self.features.append(entry)
                    print(f"BASELINE: {start_wl:.1f}-{end_wl:.1f}nm, SNR: {baseline_stats['snr']:.1f}")
            else:
                print(f"Baseline needs exactly 2 clicks, got {len(self.clicks)}")
                return
            
            self.clicks.clear()
            self.lines_drawn = []
            if not self.persistent_mode: self.current_type = None
            return
        
        # Handle other features with FIXED normalization metadata
        labels = self.get_structure_types()[self.current_type]
        for i, (wl, intens) in enumerate(self.clicks):
            label = labels[i]
            entry = {'Feature': f'{self.current_type}_{label}', 'File': self.filename,
                   'Light_Source': 'UV', 'Wavelength': round(wl, 2),
                   'Intensity': round(intens, 2), 'Point_Type': label,
                   'Feature_Group': self.current_type, 'Feature_Key': feature_key}
            
            # FIXED: Enhanced metadata for normalized features
            if self.baseline_data and self.normalization_applied:
                entry['Processing'] = 'Baseline_Then_UV_Normalized'
                entry['Baseline_Used'] = self.baseline_data['avg_intensity']
                entry['Norm_Factor'] = self.normalization_info['scaling_factor']
                entry['Normalization_Method'] = self.normalization_info['method']
                entry['Reference_Wavelength_Used'] = self.normalization_info['reference_wavelength']
                
            self.features.append(entry)
        
        # Mound summary with FIXED metadata
        if self.current_type == 'Mound' and len(self.clicks) == 3:
            s_wl, s_int = self.clicks[0]
            c_wl, c_int = self.clicks[1]
            e_wl, e_int = self.clicks[2]
            ratio, desc = self.calculate_mound_symmetry(s_wl, c_wl, e_wl)
            summary = {'Feature': 'Mound_Summary', 'File': self.filename,
                      'Light_Source': 'UV', 'Wavelength': round(c_wl, 2),
                      'Intensity': round(c_int, 2), 'Point_Type': 'Summary',
                      'Feature_Group': 'Mound', 'Symmetry_Ratio': ratio,
                      'Skew_Description': desc, 'Width_nm': round(e_wl - s_wl, 2),
                      'Feature_Key': feature_key}
            
            # FIXED: Add normalization info to mound summary
            if self.normalization_applied and self.normalization_info:
                summary['Processing'] = 'Baseline_Then_UV_Normalized'
                summary['Normalization_Method'] = self.normalization_info['method']
                
            self.features.append(summary)
        
        self.clicks.clear()
        self.lines_drawn = []
        if not self.persistent_mode: self.current_type = None
        print(f"SAVED {self.current_type} to feature list")
    
    def onkey(self, event):
        key_map = {'b': 'Baseline', '1': 'Mound', '2': 'Plateau', '3': 'Peak',
                  '4': 'Trough', '5': 'Shoulder', '6': 'Valley', '7': 'Diagnostic Region'}
        if event.key in key_map:
            self.select_feature(key_map[event.key])
        elif event.key == 's':
            action = self.save_features()
        elif event.key in ['enter', 'return']:
            if self.current_type in ['Peak', 'Valley'] and len(self.clicks) > 0:
                print(f"Manual completion of {self.current_type}")
                self.complete_feature()
        elif event.key == 'p':
            self.toggle_persistent(event)
        elif event.key == 'u':
            self.undo_last(event)
    
    def run_analysis(self):
        """FIXED: Main loop for UV analysis with proper normalization"""
        print("="*70)
        print("GEMINI UV ANALYZER - FIXED NORMALIZATION")
        print("="*70)
        print("FIXED: 811nm → 15,000, then scale 0-100")
        print("Workflow: Mark baseline → Auto-process → Mark features")
        print("Use TOOLBAR MAGNIFYING GLASS for zoom")
        print("FIXED: Proper Y-axis scaling for 0-100 range")
        print("FIXED: Exports normalization metadata for database")
        
        # Main analysis loop
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
            
            wl = self.spectrum_df.iloc[:, 0]
            intens = self.spectrum_df.iloc[:, 1]
            self.ax.plot(wl, intens, 'b-', linewidth=0.8)
            self.ax.set_title(f"FIXED UV Analysis - {self.filename}")
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("Raw Intensity")
            self.ax.grid(True, alpha=0.3)
            
            self.create_uv_ui()
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onkey)
            
            plt.subplots_adjust(right=0.82)
            print("READY! Mark baseline first (B key), then features (1-7)")
            print("FIXED normalization will be applied automatically after baseline")
            
            plt.show()
            print("Analysis window closed")

def main():
    analyzer = GeminiUVAnalyzer()
    analyzer.run_analysis()

if __name__ == '__main__':
    main()
