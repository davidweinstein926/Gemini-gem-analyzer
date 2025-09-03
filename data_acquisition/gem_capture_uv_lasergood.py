#gem_capture_uv_laser (broadband clone)
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime
import glob
from libspectr import ASQESpectrometer

SAVE_DIR = r"C:\Users\David\Gemini SP10 Calibration"
NORMALIZATION_WAVELENGTH = 650
NORMALIZATION_INTENSITY = 50000
NUM_AVERAGES = 10
EXPOSURE_TIME_MS = 5000

def normalize_at(wavelengths, intensities, target_wavelength=NORMALIZATION_WAVELENGTH, target_intensity=NORMALIZATION_INTENSITY):
    idx = np.argmin(np.abs(wavelengths - target_wavelength))
    if intensities[idx] == 0:
        return intensities
    factor = target_intensity / intensities[idx]
    return intensities * factor

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
def capture_uv_or_laser(last_used_name=""):
    root = tk.Tk()
    root.withdraw()
    full_name = simpledialog.askstring("Gem Name", "Enter full gem name (e.g., 140UC1, 140LC1):", initialvalue=last_used_name)
    messagebox.showinfo("Begin Capture", f"Capturing {full_name} with UV or Laser light.")

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
                all_frames.append(frame)
            avg_intensity = np.mean(all_frames, axis=0)

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
        np.savetxt(raw_path, np.column_stack((final_wl, final_intensities)), delimiter='\t', fmt="%.4f")
        print(f"✅ Raw GEM spectrum saved: {raw_path}")


        another = messagebox.askyesno("Repeat Capture", "Do you want to capture another UV or Laser spectrum?")
        if another:
            return capture_uv_or_laser(full_name)

        gem_norm = normalize_at(final_wl, final_intensities)

        plt.figure()
        plt.plot(final_wl, gem_norm, label='Gem Spectrum', color='red', lw=0.5)
        plt.title("Gem Spectra (Normalized)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

#         if not proceed:
        return

        diff = white_norm - gem_norm
        if np.mean(diff) < 0:
            diff *= -1

        plt.figure()
        plt.plot(final_wl, diff, label='White - Gem', lw=0.5)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        if save_diff:
            diff_path = os.path.join(SAVE_DIR, f"{full_name}_DIFF.csv")
            np.savetxt(diff_path, np.column_stack((final_wl, diff)), delimiter=',')

        if another:
            return capture_uv_or_laser()

    except Exception as e:
        messagebox.showerror("Capture Error", f"An error occurred: {e}")
