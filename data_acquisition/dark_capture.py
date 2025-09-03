import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from libspectr import ASQESpectrometer
import tkinter as tk
from tkinter import messagebox

SAVE_DIR = r"C:\\Users\\David\\Gemini SP10 Calibration\\DarkFields"
NUM_AVERAGES = 10
EXPOSURE_TIME_MS = 5000

def ensure_ascending_wavelength_order(wl, intensities):
    """Fix wavelength order to always be 293->1004"""
    if len(wl) > 1 and wl[0] > wl[-1]:  # If descending
        print("🔄 Fixing wavelength order: 1004→293 becomes 293→1004")
        return wl[::-1], intensities[::-1]  # Reverse both arrays
    return wl, intensities

def save_dark_spectrum(wavelengths, intensities):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"darkfield_{timestamp}.txt"
    filepath = os.path.join(SAVE_DIR, filename)
    os.makedirs(SAVE_DIR, exist_ok=True)
    wavelengths, intensities = ensure_ascending_wavelength_order(wavelengths, intensities)  # FIX ADDED
    np.savetxt(filepath, np.column_stack((wavelengths, intensities)), delimiter="\t", fmt="%.6f")
    print(f"📊 Saved dark field wavelength range: {wavelengths[0]:.1f} → {wavelengths[-1]:.1f} nm")  # CONFIRMATION
    return filepath

def capture_dark_field():
    spectrometer = ASQESpectrometer()
    spectrometer.set_parameters(exposure_time=EXPOSURE_TIME_MS)
    spectrometer.configure_acquisition()

    print("📷 Starting dark field capture loop... Press Spacebar to freeze.")

    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=0.5)
    ax.set_title("Live Dark Field")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()

    stop_capture = [False]
    final_wl, final_intensities = None, None

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

        line.set_data(wl, avg_intensity)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

        final_wl = wl
        final_intensities = avg_intensity

    plt.ioff()
    plt.close()

    if final_intensities is None:
        print("⚠️ No data captured.")
        return

    root = tk.Tk()
    root.withdraw()
    choice = messagebox.askyesno("Save Dark Field", "Do you want to save this dark field?")
    root.destroy()

    if choice:
        saved_path = save_dark_spectrum(final_wl, final_intensities)
        print(f"✅ Dark field saved to {saved_path}")
    else:
        print("❌ Dark field not saved.")

if __name__ == "__main__":
    capture_dark_field()