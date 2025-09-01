# Enhanced Gemini Structural Marker - Phase 1: Baseline & Slope Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime
import json
import traceback
import sys
from scipy import stats

# MODIFIED: Set the output directory for CSV files
OUTPUT_DIRECTORY = r"c:\users\david\gemini sp10 structural data"

# Enhanced global variables with persistent feature mode
features = []
current_type = None
persistent_mode = True  # NEW: Stay in selected feature mode
clicks = []
filename = ""
lines_drawn = []
texts_drawn = []
magnify_mode = False
baseline_data = None
spectrum_df = None  # Store loaded spectrum data
ax = None

# ENHANCED: Structure types with complete data capture
STRUCTURE_TYPES = {
    'Plateau': ['Start', 'Midpoint', 'End'],
    'Mound': ['Start', 'Crest', 'End'],
    'Trough': ['Start', 'Bottom', 'End'],
    'Valley': ['Midpoint'],
    'Peak': ['Max'],
    'Diagnostic Region': ['Start', 'End'],
    'Baseline Region': ['Start', 'End']  # NEW: Baseline region marking
}

def calculate_baseline_stats(df, start_wl, end_wl):
    """Calculate baseline statistics for a wavelength region"""
    try:
        label = STRUCTURE_TYPES[current_type][idx - 1]
    except IndexError:
        print("⚠️ Too many clicks for this feature type.")
        return

    # Visual feedback with SMALLER dots and enhanced colors
    colors = {
        'Mound': 'red', 'Plateau': 'green', 'Peak': 'blue', 
        'Trough': 'purple', 'Valley': 'orange', 'Diagnostic Region': 'gold',
        'Baseline Region': 'gray'
    }
    color = colors.get(current_type, 'black')
    
    # SMALLER dots: reduced from s=60 to s=25
    dot = ax.scatter(wavelength, precise_intensity, c=color, s=25, marker='o', 
                    edgecolors='black', linewidth=1, zorder=10)
    lines_drawn.append(dot)
    
    expected_clicks = len(STRUCTURE_TYPES[current_type])
    print(f"🎯 Marked {current_type} {label} ({idx}/{expected_clicks})")
    plt.draw()

    # Complete feature if all points marked
    if idx == expected_clicks:
        complete_feature()

def run_enhanced_marker():
    """Main enhanced marker function with proper restart capability"""
    global filename, ax, spectrum_df
    
    print("🔬 ENHANCED GEMINI STRUCTURAL MARKER - Phase 1")
    print("=" * 60)
    print("✨ NEW FEATURES:")
    print("   📊 Baseline region analysis with SNR calculation")
    print("   📈 Slope analysis for each feature point")
    print("   🎯 Precise intensity measurement")
    print("   🔧 Baseline-corrected intensities")
    print("=" * 60)
    
    # File selection with retry capability
    file_path = None
    while not file_path:
        try:
            print("📂 Opening file selection dialog...")
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            
            default_dir = r"C:\Users\David\OneDrive\Desktop\gemini matcher\gemini sp10 raw\raw text"
            
            file_path = filedialog.askopenfilename(
                parent=root,
                initialdir=default_dir,
                title="Select Spectrum Text File",
                filetypes=[("Text files", "*.txt")]
            )
            root.quit()
            root.destroy()
            
            # Force cleanup
            import gc
            gc.collect()
            
            if not file_path or file_path == "":
                print("❌ No file selected")
                
                # Ask if user wants to try again or exit
                root2 = tk.Tk()
                root2.withdraw()
                root2.lift()
                root2.attributes('-topmost', True)
                
                try_again = messagebox.askyesno(
                    "No File Selected",
                    "No file was selected.\n\nWould you like to try again?",
                    parent=root2
                )
                
                root2.quit()
                root2.destroy()
                gc.collect()
                
                if not try_again:
                    print("👋 File selection cancelled - exiting")
                    return
                else:
                    print("🔄 Trying file selection again...")
                    file_path = None  # Continue the while loop
                    continue
            else:
                print(f"✅ Selected: {os.path.basename(file_path)}")
                break  # Exit the while loop
                
        except Exception as e:
            print(f"❌ File selection error: {e}")
            
            # Ask if user wants to try again
            try:
                root3 = tk.Tk()
                root3.withdraw()
                
                try_again = messagebox.askyesno(
                    "File Selection Error",
                    f"Error during file selection: {e}\n\nTry again?",
                    parent=root3
                )
                
                root3.quit()
                root3.destroy()
                
                if not try_again:
                    print("👋 Exiting due to file selection error")
                    return
                else:
                    file_path = None  # Continue trying
                    continue
            except:
                print("👋 Exiting due to multiple errors")
                return
    
    # Continue with file processing...
    filename = os.path.basename(file_path)
    print(f"📁 Loading: {filename}")
    
    # Load and process spectrum
    spectrum_df, load_info = load_and_process_spectrum_file(file_path)
    if spectrum_df is None:
        print(f"❌ {load_info}")
        return
    
    print(f"✅ Loaded spectrum with {len(spectrum_df)} data points")
    
    # Create enhanced plot
    try:
        # Force matplotlib cleanup before creating new figure
        plt.close('all')
        import matplotlib
        matplotlib.pyplot.close('all')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot spectrum
        wavelengths = spectrum_df.iloc[:, 0]
        intensities = spectrum_df.iloc[:, 1]
        ax.plot(wavelengths, intensities, 'k-', linewidth=0.8)
        
        ax.set_title(f"Enhanced Structural Marker - {filename}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        
        # Create enhanced UI
        buttons = create_enhanced_ui(fig, ax)
        fig._enhanced_buttons = buttons
        
        # Connect events
        fig.canvas.mpl_connect('button_press_event', onclick)
        fig.canvas.mpl_connect('key_press_event', onkey)
        
        plt.subplots_adjust(right=0.80)
        
        # Force window positioning
        try:
            manager = fig.canvas.manager
            manager.window.wm_geometry("+100+100")
            manager.window.lift()
            manager.window.attributes('-topmost', True)
            manager.window.attributes('-topmost', False)
        except:
            pass
        
        print("\n🎯 ENHANCED ANALYSIS READY!")
        print("=" * 40)
        print("🆕 STEP 1: Mark BASELINE REGION first (gray button)")
        print("   - Click start and end of a flat, featureless region")
        print("   - This establishes noise level and SNR")
        print("\n🎯 STEP 2: Mark spectral features normally")
        print("   - Tool now captures intensity + slope data")
        print("   - Baseline correction applied automatically")
        print("\n📊 Enhanced data captured:")
        print("   ✅ Wavelength positions")
        print("   ✅ Raw and corrected intensities") 
        print("   ✅ Local slope at each point")
        print("   ✅ Baseline statistics and SNR")
        print("\n⌨️ KEYBOARD SHORTCUTS:")
        print("   1-7: Select feature types")
        print("   P: Toggle persistent mode")
        print("   M: Toggle magnify mode")
        print("   U: Undo last action")
        print("   S: Save data")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Plot creation error: {e}")
        print("💡 Try restarting the program if this persists")

def main():
    """Enhanced main function"""
    print("🔬 ENHANCED GEMINI STRUCTURAL MARKER")
    print("🚀 Phase 1: Baseline & Slope Analysis")
    run_enhanced_marker()

if __name__ == '__main__':
    main()
        # Filter data to baseline region
    wavelengths = df.iloc[:, 0]
    intensities = df.iloc[:, 1]
        
    mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
    baseline_intensities = intensities[mask]
    baseline_wavelengths = wavelengths[mask]
        
if len(baseline_intensities) < 3:
   return None
        
        # Calculate statistics
        avg_intensity = np.mean(baseline_intensities)
        std_dev = np.std(baseline_intensities)
        snr = avg_intensity / std_dev if std_dev > 0 else float('inf')
        
        # Calculate slope (should be near zero for good baseline)
        slope, intercept, r_value, p_value, std_err = stats.linregress(baseline_wavelengths, baseline_intensities)
        
        baseline_stats = {
            'wavelength_start': start_wl,
            'wavelength_end': end_wl,
            'avg_intensity': round(avg_intensity, 2),
            'std_deviation': round(std_dev, 3),
            'snr': round(snr, 1),
            'slope': round(slope, 6),  # Should be near zero
            'r_squared': round(r_value**2, 4),
            'data_points': len(baseline_intensities)
        }
        
        print(f"📊 BASELINE ANALYSIS:")
        print(f"   Region: {start_wl:.1f} - {end_wl:.1f} nm")
        print(f"   Average Intensity: {avg_intensity:.2f}")
        print(f"   Std Deviation: {std_dev:.3f}")
        print(f"   Signal-to-Noise: {snr:.1f}")
        print(f"   Slope: {slope:.6f} (closer to 0 = better baseline)")
        print(f"   Data Points: {len(baseline_intensities)}")
        
        return baseline_stats
        
    except Exception as e:
        print(f"❌ Baseline calculation error: {e}")
        return None

def get_intensity_at_wavelength(df, target_wavelength):
    """Get intensity at specific wavelength using interpolation"""
    try:
        wavelengths = df.iloc[:, 0]
        intensities = df.iloc[:, 1]
        
        # Find closest wavelength or interpolate
        if target_wavelength in wavelengths.values:
            # Exact match
            idx = wavelengths[wavelengths == target_wavelength].index[0]
            return intensities.iloc[idx]
        else:
            # Interpolate between closest points
            intensity = np.interp(target_wavelength, wavelengths, intensities)
            return intensity
            
    except Exception as e:
        print(f"❌ Intensity lookup error: {e}")
        return None

def calculate_local_slope(df, wavelength, window=2.0):
    """Calculate local slope around a wavelength point"""
    try:
        wavelengths = df.iloc[:, 0]
        intensities = df.iloc[:, 1]
        
        # Define window around the point
        mask = (wavelengths >= wavelength - window) & (wavelengths <= wavelength + window)
        local_wl = wavelengths[mask]
        local_int = intensities[mask]
        
        if len(local_wl) < 3:
            return None
        
        # Calculate slope using linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(local_wl, local_int)
        
        return {
            'slope': round(slope, 3),
            'r_squared': round(r_value**2, 4),
            'std_error': round(std_err, 3),
            'window_size': window,
            'data_points': len(local_wl)
        }
        
    except Exception as e:
        print(f"❌ Slope calculation error: {e}")
        return None

def calculate_baseline_corrected_intensity(raw_intensity, baseline_stats):
    """Calculate baseline-corrected intensity"""
    if baseline_stats is None:
        return raw_intensity
    
    baseline_avg = baseline_stats.get('avg_intensity', 0)
    corrected = raw_intensity - baseline_avg
    
    return max(0, corrected)  # Ensure non-negative

def calculate_mound_symmetry(start, crest, end):
    """Calculate symmetry/skew of a mound feature"""
    total_width = end - start
    left_width = crest - start
    right_width = end - crest
    
    if total_width == 0:
        return 0.0, "Invalid"
    
    if right_width == 0:
        symmetry_ratio = float('inf')
        skew_description = "Extreme Right Skew"
    else:
        symmetry_ratio = left_width / right_width
        
        if symmetry_ratio < 0.8:
            skew_description = "Left Skewed"
        elif symmetry_ratio > 1.25:
            skew_description = "Right Skewed"
        else:
            skew_description = "Symmetric"
    
    return round(symmetry_ratio, 3), skew_description

def complete_feature():
    """Enhanced feature completion with persistent mode"""
    global current_type, clicks, features, baseline_data, spectrum_df, persistent_mode
    
    if current_type == 'Baseline Region':
        # Handle baseline region specially
        if len(clicks) == 2:
            start_wl = min(clicks[0][0], clicks[1][0])
            end_wl = max(clicks[0][0], clicks[1][0])
            
            baseline_stats = calculate_baseline_stats(spectrum_df, start_wl, end_wl)
            
            if baseline_stats:
                baseline_data = baseline_stats
                
                # Add to features for export
                entry = {
                    'Feature': 'Baseline Region',
                    'File': filename,
                    'Start': start_wl,
                    'End': end_wl,
                    'Avg_Intensity': baseline_stats['avg_intensity'],
                    'Std_Deviation': baseline_stats['std_deviation'],
                    'SNR': baseline_stats['snr'],
                    'Slope': baseline_stats['slope'],
                    'R_Squared': baseline_stats['r_squared'],
                    'Data_Points': baseline_stats['data_points']
                }
                
                features.append(entry)
                print(f"✅ BASELINE REGION ESTABLISHED!")
                print(f"   This baseline will be used for all subsequent features")
            else:
                print(f"❌ Failed to calculate baseline statistics")
        
        clicks.clear()
        current_type = None  # Reset baseline mode
        return
    
    # Handle regular features with enhanced data capture
    entry = {'Feature': current_type, 'File': filename}
    
    # Add basic wavelength points
    for i, label in enumerate(STRUCTURE_TYPES[current_type]):
        entry[label] = round(clicks[i][0], 2)
    
    # ENHANCED: Add intensity and slope data for each point
    for i, (wavelength, raw_intensity) in enumerate(clicks):
        label = STRUCTURE_TYPES[current_type][i]
        
        # Get precise intensity at this wavelength
        precise_intensity = get_intensity_at_wavelength(spectrum_df, wavelength)
        if precise_intensity is None:
            precise_intensity = raw_intensity
        
        # Calculate baseline-corrected intensity
        corrected_intensity = calculate_baseline_corrected_intensity(precise_intensity, baseline_data)
        
        # Calculate local slope
        slope_data = calculate_local_slope(spectrum_df, wavelength)
        
        # Add enhanced data to entry
        entry[f'{label}_Intensity'] = round(precise_intensity, 2)
        if baseline_data:
            entry[f'{label}_Corrected_Intensity'] = round(corrected_intensity, 2)
        
        if slope_data:
            entry[f'{label}_Slope'] = slope_data['slope']
            entry[f'{label}_Slope_R2'] = slope_data['r_squared']
        
        print(f"📍 {label}: {wavelength:.2f}nm, Intensity: {precise_intensity:.2f}, Slope: {slope_data['slope'] if slope_data else 'N/A'}")
    
    # Add symmetry calculation for mounds (enhanced)
    if current_type == 'Mound' and len(clicks) == 3:
        start_wl = clicks[0][0]
        crest_wl = clicks[1][0]
        end_wl = clicks[2][0]
        
        # Enhanced symmetry with intensity consideration
        symmetry_ratio, skew_desc = calculate_mound_symmetry(start_wl, crest_wl, end_wl)
        entry['Symmetry_Ratio'] = symmetry_ratio
        entry['Skew_Description'] = skew_desc
        
        # Calculate intensity-based symmetry
        start_int = entry.get('Start_Corrected_Intensity', entry.get('Start_Intensity', 0))
        crest_int = entry.get('Crest_Corrected_Intensity', entry.get('Crest_Intensity', 0))
        end_int = entry.get('End_Corrected_Intensity', entry.get('End_Intensity', 0))
        
        left_height = crest_int - start_int
        right_height = crest_int - end_int
        
        if right_height > 0:
            intensity_ratio = left_height / right_height
            entry['Intensity_Symmetry_Ratio'] = round(intensity_ratio, 3)
        
        print(f"📊 Mound Symmetry: {symmetry_ratio} ({skew_desc})")
        print(f"📊 Intensity Symmetry: {entry.get('Intensity_Symmetry_Ratio', 'N/A')}")
    
    # Add baseline information to entry
    if baseline_data:
        entry['Baseline_Applied'] = True
        entry['Baseline_Level'] = baseline_data['avg_intensity']
        entry['Baseline_SNR'] = baseline_data['snr']
    else:
        entry['Baseline_Applied'] = False
        print("⚠️ No baseline region marked - using raw intensities")
    
    features.append(entry)
    clicks.clear()
    
    # NEW: Don't reset current_type if in persistent mode
    if persistent_mode and current_type != 'Baseline Region':
        print(f"🎯 COMPLETED {entry['Feature']} feature!")
        print(f"🔄 PERSISTENT MODE: Ready to mark another {current_type} (or select different feature)")
    else:
        current_type = None
        print(f"🎯 COMPLETED {entry['Feature']} feature!")
        print("✨ Ready for next feature - select another structure type")

# Button callback functions
def select_plateau(event):
    global current_type, clicks
    print("🟢 PLATEAU BUTTON CLICKED!")
    current_type = 'Plateau'
    clicks = []
    print(f"✅ Selected: {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_mound(event):
    global current_type, clicks
    print("🔴 MOUND BUTTON CLICKED!")
    current_type = 'Mound'
    clicks = []
    print(f"✅ Selected: {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_trough(event):
    global current_type, clicks
    print("🟣 TROUGH BUTTON CLICKED!")
    current_type = 'Trough'
    clicks = []
    print(f"✅ Selected: {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_valley(event):
    global current_type, clicks
    print("🟠 VALLEY BUTTON CLICKED!")
    current_type = 'Valley'
    clicks = []
    print(f"✅ Selected: {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_peak(event):
    global current_type, clicks
    print("🔵 PEAK BUTTON CLICKED!")
    current_type = 'Peak'
    clicks = []
    print(f"✅ Selected: {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_diagnostic_region(event):
    global current_type, clicks
    print("🟡 DIAGNOSTIC REGION BUTTON CLICKED!")
    current_type = 'Diagnostic Region'
    clicks = []
    print(f"✅ Selected: {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_baseline_region(event):
    global current_type, clicks
    print("⚪ BASELINE REGION BUTTON CLICKED!")
    current_type = 'Baseline Region'
    clicks = []
    print(f"✅ Selected: {current_type} - Click: {STRUCTURE_TYPES[current_type]}")
    print("📊 Mark a flat region with no features for baseline calculation")

def toggle_persistent_mode(event):
    """Toggle persistent feature selection mode"""
    global persistent_mode
    persistent_mode = not persistent_mode
    print(f"🔄 PERSISTENT MODE: {'ON' if persistent_mode else 'OFF'}")
    if persistent_mode:
        print("   ✅ Feature type stays selected after marking multiple features")
        print("   💡 Great for marking many peaks, mounds, etc.")
    else:
        print("   ❌ Must reselect feature type after each feature")
        print("   💡 Good for marking different feature types")

def toggle_magnify(event):
    """Toggle magnify/zoom mode"""
    global magnify_mode
    magnify_mode = not magnify_mode
    
    try:
        fig = plt.gcf()
        if hasattr(fig.canvas, 'toolbar') and fig.canvas.toolbar is not None:
            if magnify_mode:
                print("🔍 MAGNIFY MODE ON - Use zoom/pan tools, markers disabled")
                fig.canvas.toolbar.zoom()
            else:
                print("🔍 MAGNIFY MODE OFF - Click to place markers")
                # Deactivate zoom/pan
                try:
                    fig.canvas.toolbar.pan()
                    fig.canvas.toolbar.pan()  # Call twice to deactivate
                except:
                    pass
        else:
            if magnify_mode:
                print("🔍 MAGNIFY MODE ON - Use mouse scroll wheel to zoom")
            else:
                print("🔍 MAGNIFY MODE OFF - Click to place markers")
                
    except Exception as e:
        if magnify_mode:
            print("🔍 MAGNIFY MODE ON - markers disabled")
        else:
            print("🔍 MAGNIFY MODE OFF - Click to place markers")

def undo_callback(event):
    """Enhanced undo function with better logic"""
    global features, clicks, lines_drawn, current_type, baseline_data
    
    # First priority: undo incomplete clicks for current feature
    if clicks:
        clicks.pop()
        if lines_drawn:
            dot = lines_drawn.pop()
            dot.remove()
        plt.draw()
        remaining_clicks = len(STRUCTURE_TYPES.get(current_type, [])) - len(clicks) if current_type else 0
        print(f"↩️ Removed last click - {remaining_clicks} more needed for {current_type}")
        return
    
    # Second priority: undo last completed feature
    if features:
        last_feature = features[-1]
        feature_type = last_feature['Feature']
        
        # Calculate how many dots this feature should have had
        expected_dots = len(STRUCTURE_TYPES.get(feature_type, []))
        
        # Remove the dots for this feature
        dots_removed = 0
        while lines_drawn and dots_removed < expected_dots:
            dot = lines_drawn.pop()
            dot.remove()
            dots_removed += 1
        
        # Remove the feature
        features.pop()
        
        print(f"↩️ REMOVED COMPLETED FEATURE: {feature_type}")
        print(f"   Features remaining: {len([f for f in features if f['Feature'] != 'Baseline Region'])}")
        
        # If we removed baseline, clear baseline data
        if feature_type == 'Baseline Region':
            baseline_data = None
            print("⚠️ Baseline removed - raw intensities will be used")
        
        plt.draw()
        return
    
    print("⚠️ Nothing to undo")

def onkey(event):
    """Enhanced keyboard shortcuts"""
    global current_type, clicks, persistent_mode
    
    if event.key == '1':
        current_type = 'Plateau'
        clicks = []
        print(f"⌨️ Selected {current_type} - Persistent mode: {persistent_mode}")
    elif event.key == '2':
        current_type = 'Mound'
        clicks = []
        print(f"⌨️ Selected {current_type} - Persistent mode: {persistent_mode}")
    elif event.key == '3':
        current_type = 'Peak'
        clicks = []
        print(f"⌨️ Selected {current_type} - Persistent mode: {persistent_mode}")
    elif event.key == '4':
        current_type = 'Trough'
        clicks = []
        print(f"⌨️ Selected {current_type} - Persistent mode: {persistent_mode}")
    elif event.key == '5':
        current_type = 'Valley'
        clicks = []
        print(f"⌨️ Selected {current_type} - Persistent mode: {persistent_mode}")
    elif event.key == '6':
        current_type = 'Diagnostic Region'
        clicks = []
        print(f"⌨️ Selected {current_type} - Persistent mode: {persistent_mode}")
    elif event.key == '7':
        current_type = 'Baseline Region'
        clicks = []
        print(f"⌨️ Selected {current_type}")
    elif event.key == 'p':
        toggle_persistent_mode(event)
    elif event.key == 'm':
        toggle_magnify(event)
    elif event.key == 'u':
        undo_callback(event)
    elif event.key == 's':
        save_callback(event)

def save_callback(event):
    """Enhanced save with option to continue session"""
    if features:
        df = pd.DataFrame(features)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = filename.replace('.txt', '') if filename else "unknown"
        outname = f"{base_name}_enhanced_features_{timestamp}.csv"
        
        # Create output directory
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        full_output_path = os.path.join(OUTPUT_DIRECTORY, outname)
        
        try:
            df.to_csv(full_output_path, index=False)
            print(f"✅ Saved {len(features)} enhanced features to {full_output_path}")
            
            # Print summary of data captured
            print(f"\n📊 ENHANCED DATA SUMMARY:")
            print(f"   Features marked: {len([f for f in features if f['Feature'] != 'Baseline Region'])}")
            print(f"   Baseline established: {'Yes' if baseline_data else 'No'}")
            if baseline_data:
                print(f"   SNR: {baseline_data['snr']:.1f}")
            
            feature_types = [f['Feature'] for f in features if f['Feature'] != 'Baseline Region']
            for ftype in set(feature_types):
                count = feature_types.count(ftype)
                print(f"   {ftype}: {count}")
            
            print(f"   Data includes: Wavelengths, Raw Intensities, Slopes, Baseline Corrections")
            
        except Exception as e:
            print(f"❌ Save error: {e}")
            return
        
        # Ask if user wants to continue or finish
        ask_continue_or_finish()
    else:
        print("⚠️ No features marked.")
        ask_continue_or_finish()

def ask_continue_or_finish():
    """Ask if user wants to continue marking or finish session"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        
        choice = messagebox.askyesnocancel(
            "Session Options",
            f"📊 Analysis saved successfully!\n\n"
            f"What would you like to do next?\n\n"
            f"YES = Continue marking more features on {filename}\n"
            f"NO = Finish this gem and analyze another spectrum\n" 
            f"CANCEL = Exit completely",
            parent=root
        )
        
        root.quit()
        root.destroy()
        
        if choice is True:  # YES - Continue marking
            print("🔄 CONTINUING SESSION...")
            if baseline_data:
                print(f"   Baseline preserved (SNR: {baseline_data['snr']:.1f})")
            print("   Ready to mark more features!")
            # Don't close the plot - keep session active
            return
        elif choice is False:  # NO - New spectrum
            print("🔄 Starting analysis of new spectrum...")
            plt.close('all')
            reset_globals()
            # Add small delay to ensure cleanup
            import time
            time.sleep(0.5)
            run_enhanced_marker()  # This will go back to file selection
        else:  # CANCEL - Exit
            print("👋 Goodbye! Enhanced analysis session complete.")
            plt.close('all')
            
    except Exception as e:
        print(f"❌ Dialog error: {e}")
        print("🔄 Continuing session by default...")

def load_and_process_spectrum_file(file_path):
    """Load spectrum file and handle wavelength ordering issues"""
    try:
        df = pd.read_csv(file_path, sep='\s+', header=None)
        
        if df.shape[1] < 2:
            return None, "File does not contain two columns of data"
        
        # Check and fix wavelength ordering
        wavelengths = df.iloc[:, 0]
        first_wl = wavelengths.iloc[0]
        last_wl = wavelengths.iloc[-1]
        
        if first_wl > last_wl:
            # Descending order - needs to be flipped
            df = df.iloc[::-1].reset_index(drop=True)
            print("🔄 Auto-corrected wavelength order to ascending")
        
        return df, "success"
        
    except Exception as e:
        return None, f"Failed to load file: {e}"

def reset_globals():
    """Reset all global variables for new analysis"""
    global features, current_type, clicks, filename, lines_drawn, texts_drawn, magnify_mode, baseline_data, spectrum_df, ax
    features = []
    current_type = None
    clicks = []
    filename = ""
    lines_drawn = []
    texts_drawn = []
    magnify_mode = False
    baseline_data = None
    spectrum_df = None
    ax = None
    print("🔄 Reset complete - ready for new analysis")

def create_enhanced_ui(fig, ax):
    """Create enhanced UI with all buttons including magnify and undo"""
    button_width = 0.13
    button_height = 0.04
    button_x = 0.84
    
    buttons = []
    
    # Baseline Region button (FIRST - most important)
    ax_baseline = fig.add_axes([button_x, 0.88, button_width, button_height])
    ax_baseline.set_facecolor('lightgray')
    btn_baseline = Button(ax_baseline, 'Baseline\n(Key: 7)', color='lightgray', hovercolor='gray')
    btn_baseline.on_clicked(select_baseline_region)
    buttons.append(btn_baseline)
    
    # Feature buttons with keyboard shortcuts
    ax_plateau = fig.add_axes([button_x, 0.82, button_width, button_height])
    ax_plateau.set_facecolor('lightgreen')
    btn_plateau = Button(ax_plateau, 'Plateau\n(Key: 1)', color='lightgreen', hovercolor='green')
    btn_plateau.on_clicked(select_plateau)
    buttons.append(btn_plateau)
    
    ax_mound = fig.add_axes([button_x, 0.76, button_width, button_height])
    ax_mound.set_facecolor('lightcoral')
    btn_mound = Button(ax_mound, 'Mound\n(Key: 2)', color='lightcoral', hovercolor='red')
    btn_mound.on_clicked(select_mound)
    buttons.append(btn_mound)
    
    ax_peak = fig.add_axes([button_x, 0.70, button_width, button_height])
    ax_peak.set_facecolor('lightblue')
    btn_peak = Button(ax_peak, 'Peak\n(Key: 3)', color='lightblue', hovercolor='blue')
    btn_peak.on_clicked(select_peak)
    buttons.append(btn_peak)
    
    ax_trough = fig.add_axes([button_x, 0.64, button_width, button_height])
    ax_trough.set_facecolor('plum')
    btn_trough = Button(ax_trough, 'Trough\n(Key: 4)', color='plum', hovercolor='purple')
    btn_trough.on_clicked(select_trough)
    buttons.append(btn_trough)
    
    ax_valley = fig.add_axes([button_x, 0.58, button_width, button_height])
    ax_valley.set_facecolor('orange')
    btn_valley = Button(ax_valley, 'Valley\n(Key: 5)', color='orange', hovercolor='darkorange')
    btn_valley.on_clicked(select_valley)
    buttons.append(btn_valley)
    
    ax_diagnostic = fig.add_axes([button_x, 0.52, button_width, button_height])
    ax_diagnostic.set_facecolor('lightyellow')
    btn_diagnostic = Button(ax_diagnostic, 'Diagnostic\n(Key: 6)', color='lightyellow', hovercolor='gold')
    btn_diagnostic.on_clicked(select_diagnostic_region)
    buttons.append(btn_diagnostic)
    
    # Utility buttons
    ax_magnify = fig.add_axes([button_x, 0.44, button_width, button_height])
    btn_magnify = Button(ax_magnify, 'Magnify\n(Key: M)', color='lightsteelblue', hovercolor='steelblue')
    btn_magnify.on_clicked(toggle_magnify)
    buttons.append(btn_magnify)
    
    ax_undo = fig.add_axes([button_x, 0.38, button_width, button_height])
    btn_undo = Button(ax_undo, 'Undo\n(Key: U)', color='mistyrose', hovercolor='pink')
    btn_undo.on_clicked(undo_callback)
    buttons.append(btn_undo)
    
    ax_save = fig.add_axes([button_x, 0.32, button_width, button_height])
    btn_save = Button(ax_save, 'Save\n(Key: S)', color='lightcyan', hovercolor='cyan')
    btn_save.on_clicked(save_callback)
    buttons.append(btn_save)
    
    # Persistent Mode Toggle
    ax_persist = fig.add_axes([button_x, 0.26, button_width, button_height])
    btn_persist = Button(ax_persist, 'Persistent\n(Key: P)', color='lavender', hovercolor='mediumpurple')
    btn_persist.on_clicked(toggle_persistent_mode)
    buttons.append(btn_persist)
    
    return buttons

def onclick(event):
    """Enhanced click handler with smaller dots and persistent mode"""
    global current_type, spectrum_df
    
    if magnify_mode:
        print("🔍 Magnify mode active - markers disabled")
        return
        
    if current_type is None:
        print("❌ Please select a feature type first!")
        return

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    wavelength = event.xdata
    raw_intensity = event.ydata
    
    # Get precise intensity from spectrum data
    precise_intensity = get_intensity_at_wavelength(spectrum_df, wavelength)
    if precise_intensity is None:
        precise_intensity = raw_intensity
    
    # Calculate baseline-corrected intensity if baseline exists
    corrected_intensity = calculate_baseline_corrected_intensity(precise_intensity, baseline_data)
    
    # Calculate local slope
    slope_data = calculate_local_slope(spectrum_df, wavelength)
    
    print(f"✅ Click at {wavelength:.2f}nm:")
    print(f"   Raw Intensity: {precise_intensity:.2f}")
    if baseline_data:
        print(f"   Corrected Intensity: {corrected_intensity:.2f}")
    if slope_data:
        print(f"   Local Slope: {slope_data['slope']:.3f}")

    clicks.append((wavelength, precise_intensity))
    idx = len(clicks)
    
    try:
