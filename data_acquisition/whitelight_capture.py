#WHITE LIGHT CAPTURE
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, simpledialog, Toplevel
from datetime import datetime
from libspectr import ASQESpectrometer
from darkfield_utils import load_latest_darkfield
dark_wl, dark_intensity = load_latest_darkfield()
from time import sleep

# Configuration
CALIBRATION_DIR = r"C:\\Users\\David\\Gemini SP10 Calibration"
NORMALIZATION_TARGETS = {
    "B": (650, 50000),
    "L": (450, 50000),
    "U": (811, 15000)
}
NUM_AVERAGES = 5

def ensure_ascending_wavelength_order(wl, intensities):
    """Fix wavelength order to always be 293->1004"""
    if len(wl) > 1 and wl[0] > wl[-1]:  # If descending
        print("🔄 Fixing wavelength order: 1004→293 becomes 293→1004")
        return wl[::-1], intensities[::-1]  # Reverse both arrays
    return wl, intensities

def capture_average_spectrum(spectrometer):
    all_spectra = []
    for _ in range(NUM_AVERAGES):
        wavelengths, intensity = spectrometer.get_calibrated_spectrum()
        wavelengths, intensity = ensure_ascending_wavelength_order(wavelengths, intensity)  # FIX ADDED
        all_spectra.append(intensity)
        sleep(0.5)
    avg_intensity = np.mean(all_spectra, axis=0)
    if dark_intensity is not None and len(dark_intensity) == len(avg_intensity):
        avg_intensity = avg_intensity - dark_intensity

    return wavelengths, avg_intensity

def plot_and_confirm(title, wavelengths, intensity):
    try:
        wavelengths = np.array(wavelengths)
        intensity = np.array(intensity)
        if len(wavelengths) != len(intensity):
            raise ValueError("Wavelength and intensity arrays must match in length.")
    except Exception as e:
        messagebox.showerror("Plot Error", f"Unable to plot {title}.\n\n{e}")
        return False

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelengths, intensity, linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.grid(True)
    plt.tight_layout()
    plt.show(block=False)

    confirm = messagebox.askyesno("Confirm", f"Approve {title.lower()} capture?")
    plt.close(fig)
    if confirm:
        return True
    else:
        retry = messagebox.askyesno("Retry", f"Would you like to recapture the {title.lower()}?")
        return None if retry else False

def normalize_spectrum(wavelengths, intensities, source):
    anchor_wavelength, target_value = NORMALIZATION_TARGETS[source]
    idx = np.argmin(np.abs(wavelengths - anchor_wavelength))
    factor = target_value / intensities[idx] if intensities[idx] != 0 else 1
    return intensities * factor

def collect_white_light_spectrum():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("White Light Capture", "Capturing white light spectrum for broadband calibration.")

    try:
        spectrometer = ASQESpectrometer()
        exposure = 5000
        spectrometer.set_parameters(exposure_time=exposure)
        spectrometer.configure_acquisition()
        print(f"📷 Exposure set to {exposure / 1000:.0f} ms")

        while True:
            print("Scanning new spectrum...")
            wavelengths, corrected = capture_average_spectrum(spectrometer)
            corrected -= np.min(corrected)

            approval = plot_and_confirm("White Light Calibration", wavelengths, corrected)
            if approval is True:
                os.makedirs(CALIBRATION_DIR, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_filename = os.path.join(CALIBRATION_DIR, f"white_light_{timestamp}.txt")
                norm_filename = os.path.join(CALIBRATION_DIR, f"white_light_{timestamp}_normalized.txt")

                wavelengths, corrected = ensure_ascending_wavelength_order(wavelengths, corrected)  # FIX ADDED
                np.savetxt(raw_filename, np.column_stack((wavelengths, corrected)), delimiter="\t", fmt="%.4f")
                print(f"✅ Raw spectrum saved as {raw_filename}")
                print(f"📊 Saved white light wavelength range: {wavelengths[0]:.1f} → {wavelengths[-1]:.1f} nm")  # CONFIRMATION

                norm_corrected = normalize_spectrum(wavelengths, corrected, "B")
                wavelengths, norm_corrected = ensure_ascending_wavelength_order(wavelengths, norm_corrected)  # FIX ADDED
                np.savetxt(norm_filename, np.column_stack((wavelengths, norm_corrected)), delimiter="\t", fmt="%.4f")
                print(f"✅ Normalized spectrum saved as {norm_filename}")

                try:
                    os.startfile(raw_filename)
                except:
                    pass
                break
            elif approval is False:
                break

    except Exception as e:
        messagebox.showerror("Capture Error", f"Failed during acquisition.\n\n{e}")