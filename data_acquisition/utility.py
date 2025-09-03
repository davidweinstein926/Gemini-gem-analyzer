# utility.py

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from libspectr import ASQESpectrometer

LIGHT_SOURCE_EXPOSURES = {
    'B': 5000,
    'L': 1000,
    'U': 10000
}

LIGHT_SOURCE_FRAMES = {
    'B': 3,
    'L': 5,
    'U': 10
}

RAW_OUTPUT_DIR = os.path.join(os.getcwd(), "raw_spectra")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)

REFERENCE_PEAKS = {
    'Hg': [302.15, 313.155, 334.148, 365.015, 404.656, 435.833, 546.074, 579.066],
    'Ar': [696.543, 706.722, 738.398, 763.511, 772.376, 811.531, 842.465, 912.297, 922.450]
}

def capture_average_spectrum(gem_name, light_type):
    exposure = int(LIGHT_SOURCE_EXPOSURES.get(light_type, 5000))
    frames = int(LIGHT_SOURCE_FRAMES.get(light_type, 1))

    spectrometer = ASQESpectrometer()
    spectrometer.configure_acquisition()
    spectrometer.set_exposure(exposure)

    all_intensities = []
    wavelength = None

    for _ in range(frames):
        wl, intensity = spectrometer.capture()
        if wavelength is None:
            wavelength = wl
        all_intensities.append(intensity)

    avg_intensity = np.mean(all_intensities, axis=0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{gem_name}_{light_type}_{timestamp}.csv"
    filepath = os.path.join(RAW_OUTPUT_DIR, filename)

    np.savetxt(filepath, np.column_stack((wavelength, avg_intensity)), delimiter=",", fmt="%.6f")
    return wavelength, avg_intensity, filename

def plot_spectrum(wavelength, intensity, title="Spectrum"):
    plt.figure(figsize=(10, 5))
    plt.plot(wavelength, intensity, linewidth=0.5)
    plt.title(title)
    plt.xlabel("Wavelength (nm")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
