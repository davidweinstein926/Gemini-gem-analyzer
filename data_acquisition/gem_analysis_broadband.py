#gem_analysis_broadband
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime
import glob
from libspectr import ASQESpectrometer
from darkfield_utils import load_latest_darkfield
def ensure_ascending_wavelength_order(wl, intensities):
    """Fix wavelength order to always be 293->1004"""
    if len(wl) > 1 and wl[0] > wl[-1]:  # If descending
        print("🔄 Fixing wavelength order: 1004→293 becomes 293→1004")
        return wl[::-1], intensities[::-1]  # Reverse both arrays
    return wl, intensities

dark_wl, dark_intensity = load_latest_darkfield()

SAVE_DIR = r"C:\Users\David\Gemini SP10 Calibration"
NORMALIZATION_WAVELENGTH = 650
NORMALIZATION_INTENSITY = 50000
NUM_AVERAGES = 10

def normalize_at(wavelengths, intensities, target_wavelength=NORMALIZATION_WAVELENGTH, target_intensity=NORMALIZATION_INTENSITY):
    idx = np.argmin(np.abs(wavelengths - target_wavelength))
    if intensities[idx] == 0:
        return intensities
    factor = target_intensity / intensities[idx]
    return intensities * factor

def load_latest_white_light(directory):
    try:
        files = sorted([f for f in os.listdir(directory) if f.startswith("white_light_") and f.endswith(".txt")], reverse=True)
        if not files:
            raise FileNotFoundError("No white light reference files starting with 'white_light_' and ending in '.txt'.")
        latest_file = os.path.join(directory, files[0])
        data = np.loadtxt(latest_file)
        wl, intensity = data[:, 0], data[:, 1]
        return wl, intensity, latest_file
    except Exception as e:
        raise FileNotFoundError(f"Error loading white light reference: {e}")

# This function should only be used for GEM analysis
# Not for white light capture. All white light saving should be handled in whitelight_capture.py
def capture_broadband_scan():
    root = tk.Tk()
    root.withdraw()

    full_name = simpledialog.askstring("Gem Name", "Enter full gem name (e.g., 140BC1):", parent=root)
    gem_type_code = simpledialog.askstring("Gem Type", "Enter gem type: C for Colorless, CS for Colored Stone:", parent=root)
    if gem_type_code is None:
        return
    gem_type_code = gem_type_code.strip().upper()
    if gem_type_code == 'C':
        EXPOSURE_TIME_MS = 5000
    elif gem_type_code == 'CS':
        EXPOSURE_TIME_MS = 5000
    else:
        messagebox.showerror("Input Error", "Invalid gem type. Use 'C' or 'CS'.")
        return

    messagebox.showinfo("Begin Capture", f"Capturing {full_name} with broadband (white) light.")

    try:
        spectrometer = ASQESpectrometer()
        spectrometer.set_parameters(exposure_time=EXPOSURE_TIME_MS)
        spectrometer.configure_acquisition()

        print("📷 Starting continuous capture loop... Press Spacebar to freeze.")

        final_wl, final_intensities = None, None
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=0.5)
        ax.set_title(f"Live Spectrum: {full_name}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        plt.grid(True)
        plt.tight_layout()

        stop_capture = [False]

        def on_key(event):
            if event.key == ' ':
                stop_capture[0] = True
                print("🛑 Capture frozen by user.")

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.ion()
        plt.show()

        while not stop_capture[0]:
            all_frames = []
            for _ in range(NUM_AVERAGES):
                wl, frame = spectrometer.get_calibrated_spectrum()
                wl, frame = ensure_ascending_wavelength_order(wl, frame)  # FIX ADDED

                all_frames.append(frame)
            avg_intensity = np.mean(all_frames, axis=0)
            try:
                if dark_intensity is not None and len(dark_intensity) == len(avg_intensity):
                    avg_intensity = avg_intensity - dark_intensity
            except Exception as e:
                print('⚠️ Failed to apply darkfield correction:', e)

            line.set_data(wl, avg_intensity)
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            final_wl, final_intensities = wl, avg_intensity

        plt.ioff()
        plt.show()

        keep = messagebox.askyesno("Confirm Capture", "Do you want to keep this capture?")
        if not keep:
            return capture_broadband_scan()

        os.makedirs(SAVE_DIR, exist_ok=True)
        raw_path = os.path.join(SAVE_DIR, f"{full_name}.txt")
        final_wl, final_intensities = ensure_ascending_wavelength_order(final_wl, final_intensities)  # FIX ADDED
        np.savetxt(raw_path, np.column_stack((final_wl, final_intensities)), delimiter='\t', fmt="%.4f")
        print(f"📊 Saved wavelength range: {final_wl[0]:.1f} → {final_wl[-1]:.1f} nm")  # CONFIRMATION

        print(f"✅ Raw GEM spectrum saved: {raw_path}")

        try:
            white_wl, white_int, white_path = load_latest_white_light(SAVE_DIR)
        except Exception as e:
            messagebox.showerror("White Light Missing", f"Failed to load white light: {e}")
            return

        gem_norm = normalize_at(final_wl, final_intensities)
        white_norm = normalize_at(white_wl, white_int)

        plt.figure()
        plt.plot(white_wl, white_norm, label='White Light', color='blue', lw=0.5)
        plt.plot(final_wl, gem_norm, label='Gem Spectrum', color='red', lw=0.5)
        plt.title("Gem vs White Light (Normalized)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        proceed = messagebox.askyesno("Difference", "Do you want to see the difference spectrum?")
        if not proceed:
            return

        diff = white_norm - gem_norm
        if np.mean(diff) < 0:
            diff *= -1

        plt.figure()
        plt.plot(final_wl, diff, label='White - Gem', lw=0.5)
        plt.title("Difference Spectrum")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        save_diff = messagebox.askyesno("Save Difference", "Save difference spectrum?")
        if save_diff:
            diff_path = os.path.join(SAVE_DIR, f"{full_name}_DIFF.csv")
            np.savetxt(diff_path, np.column_stack((final_wl, diff)), delimiter=',')
            print(f"✅ Difference spectrum saved: {diff_path}")

    except Exception as e:
        messagebox.showerror("Capture Error", f"An error occurred: {e}")
