#gem_capture_uv_laser (UV + laser only)
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime
import glob
from libspectr import ASQESpectrometer
from darkfield_utils import load_latest_darkfield

SAVE_DIR = r"C:\Users\David\Gemini SP10 Calibration"
NORMALIZATION_WAVELENGTH = 650
NORMALIZATION_INTENSITY = 50000
NUM_AVERAGES = 10
def ensure_ascending_wavelength_order(wl, intensities):
    """Fix wavelength order to always be 293->1004"""
    if len(wl) > 1 and wl[0] > wl[-1]:  # If descending
        print("🔄 Fixing wavelength order: 1004→293 becomes 293→1004")
        return wl[::-1], intensities[::-1]  # Reverse both arrays
    return wl, intensities

dark_wl, dark_intensity = load_latest_darkfield()

def normalize_at(wavelengths, intensities, target_wavelength=NORMALIZATION_WAVELENGTH, target_intensity=NORMALIZATION_INTENSITY):
    idx = np.argmin(np.abs(wavelengths - target_wavelength))
    if intensities[idx] == 0:
        return intensities
    factor = target_intensity / intensities[idx]
    return intensities * factor

def capture_uv_or_laser(last_used_name=""):
    root = tk.Tk()
    root.withdraw()

    full_name = simpledialog.askstring("Gem Name", "Enter full gem name (e.g., 140UC1, 140LC1):", parent=root, initialvalue=last_used_name)

    gem_type_code = simpledialog.askstring("Gem Type", "Enter gem type: C for Colorless, CS for Colored Stone:", parent=root)
    if gem_type_code is None:
        return
    gem_type_code = gem_type_code.strip().upper()
    if gem_type_code == 'C':
        gem_type = 'colorless'
    elif gem_type_code == 'CS':
        gem_type = 'colored'
    else:
        messagebox.showerror("Input Error", "Invalid gem type. Use 'C' or 'CS'.")
        return

    light_code = simpledialog.askstring("Light Source", "Enter light source: L for Laser, U for Ultraviolet:", parent=root)
    if light_code is None:
        return
    light_source = light_code.strip().upper()
    if light_source not in ['L', 'U']:
        messagebox.showerror("Input Error", "Invalid light source. Use 'L' or 'U'.")
        return

    # Exposure mapping
    exposure_map = {
        ('colorless', 'L'): 11000,
        ('colorless', 'U'): 7500,
        ('colored', 'L'): 2500,
        ('colored', 'U'): 3500
    }
    EXPOSURE_TIME_MS = exposure_map.get((gem_type, light_source), 5000)

    messagebox.showinfo("Begin Capture", f"Capturing {full_name} with light source {light_source} (Exposure: {EXPOSURE_TIME_MS} ms)")

    try:
        spectrometer = ASQESpectrometer()
        spectrometer.set_parameters(exposure_time=EXPOSURE_TIME_MS)
        spectrometer.configure_acquisition()

        print("📷 Starting continuous capture loop... Press Spacebar to freeze.")

        final_wl, final_intensities = None, None
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=0.5)
        ax.set_title(f"Live Spectrum: {full_name}")
        ax.set_xlabel("Wavelength (nm")
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
            if dark_intensity is not None and len(dark_intensity) == len(avg_intensity):
                avg_intensity = avg_intensity - dark_intensity

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
            return capture_uv_or_laser(full_name)

        os.makedirs(SAVE_DIR, exist_ok=True)
        raw_path = os.path.join(SAVE_DIR, f"{full_name}.txt")
        final_wl, final_intensities = ensure_ascending_wavelength_order(final_wl, final_intensities)  # FIX ADDED
        np.savetxt(raw_path, np.column_stack((final_wl, final_intensities)), delimiter='\t', fmt="%.4f")
        print(f"📊 Saved wavelength range: {final_wl[0]:.1f} → {final_wl[-1]:.1f} nm")  # CONFIRMATION
        print(f"✅ Raw GEM spectrum saved: {raw_path}")

        another = messagebox.askyesno("Repeat Capture", "Do you want to capture another UV or Laser spectrum?")
        if another:
            return capture_uv_or_laser(full_name)

        if full_name[-3].upper() == 'U':
            gem_norm = normalize_at(final_wl, final_intensities, 811, 15000)
        elif full_name[-3].upper() == 'L':
            gem_norm = normalize_at(final_wl, final_intensities, 450, 50000)

        plt.figure()
        plt.plot(final_wl, gem_norm, label='Gem Spectrum', color='red', lw=0.5)
        plt.title("Gem Spectra (Normalized)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return

    except Exception as e:
        messagebox.showerror("Capture Error", f"An error occurred: {e}")
