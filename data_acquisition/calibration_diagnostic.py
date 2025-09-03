# calibration_diagnostic.py

import numpy as np
import matplotlib.pyplot as plt
from libspectr import ASQESpectrometer
from darkfield_utils import load_latest_darkfield
dark_wl, dark_intensity = load_latest_darkfield()
from scipy.signal import find_peaks
import csv
import datetime
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

# Full reference Hg/Ar peaks (used for plotting)
theoretical_peaks = [
    302.15, 313.155, 334.148, 365.015, 404.656, 435.833,
    546.074, 576.960, 579.066, 696.543, 706.722, 738.398,
    763.511, 772.367, 794.818, 811.531, 912.297, 922.450
]

# Peaks selected for calibration matching
calibration_peaks = [
    313.155, 365.015, 404.656, 435.833, 546.074,
    579.066, 696.543, 738.398, 750.387, 763.511,
    772.367, 811.531, 842.465, 912.297, 922.450
]

def run_diagnostics():
    print("\U0001f527 Running Calibration Diagnostic...")

    # Prompt user to place probe
    root = tk.Tk()
    root.withdraw()
    ready = messagebox.askyesno("Calibration Setup", "Place the probe over the UV lamp with the glass slide in place.\n\nReady to capture?")
    if not ready:
        print("❌ Capture canceled by user.")
        return

    spectrometer = ASQESpectrometer()
    spectrometer.configure_acquisition()

    capturing = True
    while capturing:
        print("📡 Capturing spectrum...")
        wavelengths, intensities = spectrometer.get_calibrated_spectrum()
        if dark_intensity is not None and len(dark_intensity) == len(intensities):
            intensities = intensities - dark_intensity


        # Plot spectrum and wait for spacebar to freeze
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wavelengths, intensities, label="Live Spectrum", linewidth=0.5)
        ax.set_title("Calibration Spectrum Capture")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(True)
        plt.tight_layout()

        print("🔁 Displaying captured spectrum. Close the window or press spacebar when satisfied.")

        # Wait for user to close or press spacebar
        confirmed = [False]

        def on_key(event):
            if event.key == ' ':
                confirmed[0] = True
                plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        if confirmed[0]:
            keep = messagebox.askyesno("Capture Review", "Do you want to keep this capture?")
            if keep:
                capturing = False
            else:
                print("🔄 Recapturing spectrum...")
        else:
            print("❌ Capture not confirmed. Restarting loop...")

    # Detect peaks (broad sweep, not used directly for calibration matching)
    peak_indices, _ = find_peaks(intensities, height=np.max(intensities)*0.1, distance=10)
    measured_peaks = wavelengths[peak_indices]

    matched_results = []
    for expected in calibration_peaks:
        # Find max intensity within ±2 nm of expected peak
        nearby_indices = np.where((wavelengths >= expected - 2) & (wavelengths <= expected + 2))[0]
        if len(nearby_indices) > 0:
            max_index = nearby_indices[np.argmax(intensities[nearby_indices])]
            measured = wavelengths[max_index]
            delta = abs(measured - expected)
            matched_results.append((expected, measured, delta))
        else:
            matched_results.append((expected, np.nan, np.nan))

    # Show report before saving
    summary = "Expected (nm)\tMeasured (nm)\tΔ (nm)\n"
    summary += "-" * 45 + "\n"
    for expected, measured, delta in matched_results:
        if not np.isnan(measured):
            summary += f"{expected:.1f}\t\t{measured:.2f}\t\t{delta:.2f}\n"
        else:
            summary += f"{expected:.1f}\t\tN/A\t\tN/A\n"

    print("\n📈 Matched Peaks Report:")
    print(summary)

    approve = messagebox.askyesno("Save Calibration Report?", f"Here is the match report:\n\n{summary}\nSave this as CSV?")

    if approve:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_results_{timestamp}.csv"
        output_path = os.path.join(os.getcwd(), filename)

        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Expected (nm)", "Measured (nm)", "Delta (nm)"])
            for row in matched_results:
                writer.writerow([f"{row[0]:.3f}", f"{row[1]:.3f}" if not np.isnan(row[1]) else "N/A", f"{row[2]:.3f}" if not np.isnan(row[2]) else "N/A"])

        print(f"📁 Results saved to: {output_path}")
    else:
        print("❌ Report not saved.")

    # Plot the final spectrum with all detected peaks and expected lines
    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, intensities, label="Captured Spectrum", linewidth=0.5)
    plt.plot(measured_peaks, intensities[peak_indices], 'ro', label="Detected Peaks", markersize=3)

    for expected in theoretical_peaks:
        plt.axvline(x=expected, color='g', linestyle='--', alpha=0.5)

    for _, measured, _ in matched_results:
        if not np.isnan(measured):
            plt.axvline(x=measured, color='r', linestyle=':', alpha=0.5)

    plt.title("Calibration Diagnostic Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("✅ Diagnostic complete.")
