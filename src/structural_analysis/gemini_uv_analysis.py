# gemini_uv_analyzer.py - FIXED DISPLAY VERSION
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
        self.OUTPUT_DIRECTORY = r"c:\users\david\gemini sp10 structural data\uv"
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
        print("ðŸ”„ Session reset - ready for new spectrum")
        
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
                print("ðŸ”„ Auto-corrected wavelength order")
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
            print("âš ï¸ No baseline established")
            return False
        try:
            baseline_avg = self.baseline_data['avg_intensity']
            print(f"ðŸ”„ Applying baseline correction: subtracting {baseline_avg:.2f}")
            corrected = self.spectrum_df.iloc[:, 1] - baseline_avg
            self.spectrum_df.iloc[:, 1] = corrected.clip(lower=0)
            print(f"ðŸ“Š BASELINE CORRECTION APPLIED: -{baseline_avg:.2f}")
            self.update_plot("Baseline Corrected")
            return True
        except Exception as e:
            print(f"âš ï¸ Baseline correction error: {e}")
            return False

    def normalize_spectrum_to_maximum(self, target_max=100.0):
        """FIXED: Normalize to actual maximum, not fixed wavelength"""
        try:
            wl = self.spectrum_df.iloc[:, 0]
            intens = self.spectrum_df.iloc[:, 1].values.copy()
            
            # Find MAXIMUM intensity
            max_int = intens.max()
            max_idx = intens.argmax()
            max_wl = wl.iloc[max_idx]
            
            print(f"ðŸ“Š NORMALIZING TO MAXIMUM:")
            print(f"   Maximum found at {max_wl:.1f}nm = {max_int:.2f}")
            print(f"   Target maximum = {target_max:.1f}")
            
            if max_int <= 0:
                print("âš ï¸ Cannot normalize - maximum intensity is zero or negative")
                return None
            
            scaling_factor = target_max / max_int
            print(f"   Scaling factor = {scaling_factor:.6f}")
            
            self.spectrum_df.iloc[:, 1] = intens * scaling_factor
            new_max = self.spectrum_df.iloc[:, 1].max()
            print(f"âœ… After normalization: max = {new_max:.2f}")
            
            self.update_plot(f"Normalized (Max={target_max:.0f})")
            
            return {'method': 'maximum', 'reference_wavelength': max_wl,
                   'original_intensity': max_int, 'final_intensity': new_max,
                   'scaling_factor': scaling_factor, 'target_intensity': target_max}
        except Exception as e:
            print(f"âš ï¸ Normalization error: {e}")
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
            print("ðŸ§¹ Cleared all visual markers")
        except Exception as e:
            print(f"âš ï¸ Error clearing visual markers: {e}")
            self.lines_drawn = []

    def update_plot(self, title_suffix):
        """FIXED: Force Y-axis update after normalization + UV-specific styling"""
        wl = self.spectrum_df.iloc[:, 0]
        intens = self.spectrum_df.iloc[:, 1]
        
        # Clear and redraw
        self.ax.clear()
        self.ax.plot(wl, intens, 'indigo', linewidth=0.8)  # Indigo for UV
        self.ax.set_title(f"UV Analysis ({title_suffix}) - {self.filename}")
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel("Processed Intensity")
        self.ax.grid(True, alpha=0.3, color='purple')  # Purple grid for UV
        
        # UV-specific wavelength range focus
        if len(wl) > 0:
            min_wl = wl.min()
            max_wl = wl.max()
            if min_wl < 200:
                self.ax.set_xlim(200, min(800, max_wl))
        
        # FORCE Y-axis update after normalization
        if len(intens) > 0:
            y_min = intens.min()
            y_max = intens.max()
            y_range = y_max - y_min
            
            # Add 5% padding and force the limits
            padding = 0.05 * y_range if y_range > 0 else 5
            self.ax.set_ylim(y_min - padding, y_max + padding)
            
            # Force immediate update
            self.ax.relim()
            self.ax.autoscale_view()
            
            print(f"ðŸ”§ UV DISPLAY FIX: Y-axis set to {y_min:.1f} - {y_max:.1f}")
        
        # Clear visual markers tracking
        if hasattr(self, 'lines_drawn'):
            self.lines_drawn.clear()
        else:
            self.lines_drawn = []
        
        # Force canvas redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()  # Additional force update
    
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
            print(f"ðŸ”„ Auto-completing previous {self.current_type}")
            self.complete_feature()
        self.current_type = feature_type
        self.clicks = []
        self.feature_ready = False
        expected_points = self.get_structure_types()[feature_type]
        print(f"âœ… Selected: {feature_type} - need {len(expected_points)} points")
        if self.fig: self.fig.canvas.draw()
    
    def file_selection_dialog(self):
        default_dir = r"C:\Users\David\OneDrive\Desktop\gemini matcher\gemini sp10 raw\raw text"
        while True:
            try:
                root = tk.Tk()
                root.withdraw()
                root.lift()
                root.attributes('-topmost', True)
                file_path = filedialog.askopenfilename(
                    parent=root, initialdir=default_dir,
                    title="Select Gem Spectrum for UV Analysis",
                    filetypes=[("Text files", "*.txt")])
                root.quit()
                root.destroy()
                if not file_path:
                    if not messagebox.askyesno("No File", "Try again?"): return None
                    continue
                return file_path
            except Exception as e:
                if not messagebox.askyesno("Error", f"Error: {e}\nTry again?"): return None
    
    def save_features(self):
        if hasattr(self, 'feature_ready') and self.feature_ready and self.clicks:
            print(f"ðŸ”„ Completing {self.current_type} before saving...")
            self.complete_feature()
        if not self.features:
            print("âš ï¸ No features to save")
            return
        try:
            df = pd.DataFrame(self.features)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = self.filename.replace('.txt', '')
            outname = f"{base_name}_uv_structural_{timestamp}.csv"
            os.makedirs(self.OUTPUT_DIRECTORY, exist_ok=True)
            full_path = os.path.join(self.OUTPUT_DIRECTORY, outname)
            df.to_csv(full_path, index=False)
            print(f"âœ… Saved {len(self.features)} features to UV folder")
            self.ask_next_action()
        except Exception as e:
            print(f"âš ï¸ Save error: {e}")
    
    def ask_next_action(self):
        try:
            result = messagebox.askyesnocancel(
                "Analysis Complete",
                f"âœ… UV analysis saved for: {self.filename}\n\n"
                f"YES = Analyze another gem with UV\n"
                f"NO = Close this analyzer (return to launcher)\n"
                f"CANCEL = Exit completely")
            if result is True:
                plt.close('all')
                self.reset_session()
                self.run_analysis()
            elif result is False:
                plt.close('all')
                print("ðŸ”„ Closing UV analyzer - launcher should still be open...")
                print("ðŸ’¡ Switch to the launcher window to select another analyzer or return to main menu")
                # Close this analyzer process cleanly
                import sys
                sys.exit(0)  # This ends the subprocess cleanly
            else:
                plt.close('all')
                print("ðŸ‘‹ Exiting completely...")
                import sys
                sys.exit(1)  # Exit with code 1 to signal complete shutdown
        except Exception as e:
            print(f"âš ï¸ Dialog error: {e}")
            plt.close('all')
            import sys
            sys.exit(0)  # Clean exit on error
    
    def create_uv_ui(self):
        self.buttons = []
        bw, bh, bx = 0.12, 0.035, 0.845
        bs = 0.045
        
        # Baseline
        ax_btn = self.fig.add_axes([bx, 0.92, bw, bh])
        btn = Button(ax_btn, 'Baseline\n(B)', color='lightgray')
        btn.on_clicked(lambda e: self.select_feature('Baseline'))
        self.buttons.append(btn)
        
        # Features with UV-themed purple colors
        features = [('Mound', 'plum', '1'), ('Plateau', 'mediumorchid', '2'),
                   ('Peak', 'mediumpurple', '3'), ('Trough', 'darkviolet', '4'),
                   ('Shoulder', 'violet', '5'), ('Valley', 'indigo', '6'),
                   ('Diagnostic Region', 'lavender', '7')]
        
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
                print(f"â†©ï¸ Undid click: {removed_click[0]:.2f}nm - {remaining} more needed")
                if len(self.clicks) < expected: self.feature_ready = False
            elif self.features:
                last_feature = self.features[-1] if self.features else None
                last_group = last_feature.get('Feature_Group', '') if last_feature else ''
                last_key = last_feature.get('Feature_Key', '') if last_feature else ''
                if last_group:
                    self.features = [f for f in self.features if f.get('Feature_Group', '') != last_group]
                    print(f"â†©ï¸ Removed completed feature: {last_group}")
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
                print("âš ï¸ Nothing to undo")
                return
            if self.fig and self.ax: self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"âš ï¸ Undo error: {e}")
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
                print(f"âœ… Rebuilt: {len(self.clicks)} clicks")
            if self.fig: self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"âš ï¸ Force resync error: {e}")
            self.lines_drawn = []
    
    def toggle_persistent(self, event):
        self.persistent_mode = not self.persistent_mode
        print(f"ðŸ”„ PERSISTENT: {'ON' if self.persistent_mode else 'OFF'}")
    
    def onclick(self, event):
        # Check toolbar zoom/pan
        if hasattr(self.fig.canvas, 'toolbar') and self.fig.canvas.toolbar:
            toolbar = self.fig.canvas.toolbar
            if hasattr(toolbar, 'mode') and toolbar.mode in ['zoom rect', 'pan']:
                print(f"ðŸ” TOOLBAR {toolbar.mode.upper()} MODE ACTIVE")
                return
        
        if not self.current_type:
            print("âš ï¸ Select feature first! (B for baseline, 1-7 for structures)")
            return
        if event.inaxes != self.ax or event.xdata is None: 
            print("âš ï¸ Click inside the plot area")
            return

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
        
        print(f"ðŸŽ¯ {self.current_type} {current}/{expected}: {wl:.2f}nm, intensity: {precise_int:.2f}")
        
        if current == expected:
            print(f"âœ… {self.current_type} READY - Press S to save or U to undo")
            self.feature_ready = True
            # Auto-complete single-point features
            if self.current_type in ['Peak', 'Valley']:
                print(f"ðŸ”„ Auto-completing {self.current_type}")
                self.complete_feature()
        else:
            print(f"   â³ Need {expected - current} more clicks")
            self.feature_ready = False
        
        self.fig.canvas.draw_idle()
    
    def complete_feature(self):
        if not self.clicks or not self.current_type: 
            print("âš ï¸ No clicks or feature type to complete")
            return
        
        self.feature_ready = False
        feature_key = f"{self.current_type}_{len(self.features)}"
        if self.lines_drawn:
            self.completed_feature_visuals[feature_key] = self.lines_drawn.copy()
        
        print(f"ðŸ”„ Completing feature: {self.current_type} with {len(self.clicks)} clicks")
        
        # Handle baseline
        if self.current_type == 'Baseline':
            if len(self.clicks) == 2:
                start_wl = min(self.clicks[0][0], self.clicks[1][0])
                end_wl = max(self.clicks[0][0], self.clicks[1][0])
                
                baseline_stats = self.calculate_baseline_stats(start_wl, end_wl)
                if baseline_stats:
                    self.baseline_data = baseline_stats
                    self.normalization_applied = False
                    self.normalization_info = None
                    
                    print(f"ðŸ”„ Starting spectrum processing...")
                    if self.apply_baseline_correction_to_spectrum():
                        normalization_info = self.normalize_spectrum_to_maximum(100.0)
                        if normalization_info:
                            self.normalization_applied = True
                            self.normalization_info = normalization_info
                            print("âœ… Processing complete: Baseline + Normalized")
                    
                    for i, (wl, intens) in enumerate(self.clicks):
                        pt_type = 'Start' if i == 0 else 'End'
                        entry = {'Feature': f'Baseline_{pt_type}', 'File': self.filename,
                               'Light_Source': 'UV', 'Wavelength': round(wl, 2),
                               'Intensity': round(intens, 2), 'Point_Type': pt_type,
                               'Feature_Group': 'Baseline', 'Processing': 'Baseline_Then_Normalized',
                               'SNR': baseline_stats['snr'], 'Feature_Key': feature_key}
                        self.features.append(entry)
                        print(f"âœ… Added baseline entry: {entry['Feature']}")
                    print(f"âœ… BASELINE: {start_wl:.1f}-{end_wl:.1f}nm, SNR: {baseline_stats['snr']:.1f}")
            else:
                print(f"âš ï¸ Baseline needs exactly 2 clicks, got {len(self.clicks)}")
                return
            
            self.clicks.clear()
            self.lines_drawn = []
            if not self.persistent_mode: self.current_type = None
            return
        
        # Handle other features
        labels = self.get_structure_types()[self.current_type]
        print(f"ðŸ“ Processing {self.current_type} with labels: {labels}")
        
        for i, (wl, intens) in enumerate(self.clicks):
            label = labels[i]
            entry = {'Feature': f'{self.current_type}_{label}', 'File': self.filename,
                   'Light_Source': 'UV', 'Wavelength': round(wl, 2),
                   'Intensity': round(intens, 2), 'Point_Type': label,
                   'Feature_Group': self.current_type, 'Feature_Key': feature_key}
            if self.baseline_data and self.normalization_applied:
                entry['Processing'] = 'Baseline_Then_Normalized'
                entry['Baseline_Used'] = self.baseline_data['avg_intensity']
                entry['Norm_Factor'] = self.normalization_info['scaling_factor']
            else:
                entry['Processing'] = 'Raw_Data'
            self.features.append(entry)
            print(f"âœ… Added feature entry: {entry['Feature']} at {wl:.2f}nm")
        
        # Mound summary
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
            self.features.append(summary)
            print(f"âœ… Added mound summary at {c_wl:.2f}nm")
        
        self.clicks.clear()
        self.lines_drawn = []
        if not self.persistent_mode: self.current_type = None
        print(f"ðŸ’¾ COMPLETED {self.current_type} - total features now: {len(self.features)}")
        print(f"ðŸ” Current features list: {[f.get('Feature', 'Unknown') for f in self.features]}")
    
    def onkey(self, event):
        key_map = {'b': 'Baseline', '1': 'Mound', '2': 'Plateau', '3': 'Peak',
                  '4': 'Trough', '5': 'Shoulder', '6': 'Valley', '7': 'Diagnostic Region'}
        if event.key in key_map:
            self.select_feature(key_map[event.key])
        elif event.key == 's':
            self.save_features()
        elif event.key in ['enter', 'return']:
            if self.current_type in ['Peak', 'Valley'] and len(self.clicks) > 0:
                print(f"âŽ Manual completion of {self.current_type}")
                self.complete_feature()
        elif event.key == 'p':
            self.toggle_persistent(event)
        elif event.key == 'u':
            self.undo_last(event)
    
    def run_analysis(self):
        print("="*60)
        print("ðŸŸ£ GEMINI UV ANALYZER - DISPLAY FIXED")
        print("="*60)
        print("ðŸ”§ Workflow: Mark baseline â†’ Auto-process â†’ Mark features")
        print("ðŸ” Use TOOLBAR MAGNIFYING GLASS for zoom")
        print("ðŸ“Š DISPLAY FIX: Y-axis now updates properly after normalization")
        print("ðŸŸ£ UV-specific styling: Indigo spectrum with purple grid")
        
        # Main analysis loop - allows multiple analyses in one session
        while True:
            file_path = self.file_selection_dialog()
            if not file_path: 
                break  # User cancelled, exit
            
            self.filename = os.path.basename(file_path)
            self.spectrum_df, load_info = self.load_spectrum_file(file_path)
            
            if self.spectrum_df is None:
                print(f"âš ï¸ {load_info}")
                continue  # Try again with different file
            
            print(f"âœ… Loaded: {self.filename} ({len(self.spectrum_df)} points)")
            
            plt.close('all')
            self.fig, self.ax = plt.subplots(figsize=(13, 7))
            
            wl = self.spectrum_df.iloc[:, 0]
            intens = self.spectrum_df.iloc[:, 1]
            self.ax.plot(wl, intens, 'indigo', linewidth=0.8)
            self.ax.set_title(f"UV Analysis - {self.filename}")
            self.ax.set_xlabel("Wavelength (nm)")
            self.ax.set_ylabel("Intensity")
            self.ax.grid(True, alpha=0.3, color='purple')
            
            self.create_uv_ui()
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onkey)
            
            plt.subplots_adjust(right=0.82)
            print("ðŸ’Ž READY! Mark baseline first (B key), then features (1-7)")
            
            plt.show()
            print("ðŸ“Š Analysis window closed")

def main():
    analyzer = GeminiUVAnalyzer()
    analyzer.run_analysis()

if __name__ == '__main__':
    main()
