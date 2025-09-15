# gemini_utils.py

import pandas as pd
import numpy as np

def normalize_spectrum(df, src, target_intensity=50000):
    """
    Normalizes the spectrum intensity based on the specified light source.
    For B spectra: normalize to 50,000 at 650 nm
    For L spectra: normalize to 50,000 at 450 nm
    For U spectra: normalize to 15,000 at the 811 nm peak region
    """
    if src == 'B':
        anchor_wavelength = 650
        target_intensity = 50000
    elif src == 'L':
        anchor_wavelength = 450
        target_intensity = 50000
    elif src == 'U':
        anchor_wavelength = 811
        target_intensity = 15000
    else:
        raise ValueError(f"Unknown source type: {src}")

    # Find the intensity at the anchor wavelength using interpolation
    anchor_intensity = np.interp(anchor_wavelength, df['wavelength'], df['intensity'])

    if anchor_intensity == 0:
        raise ValueError(f"Normalization point {anchor_wavelength} nm not found in the spectrum.")

    normalization_factor = target_intensity / anchor_intensity
    df['intensity'] = df['intensity'] * normalization_factor
    return df

def compute_match_score(unk_df, ref_df):
    """
    Calculates a simple match score (e.g. Mean Absolute Error) between two normalized spectra.
    """
    merged = pd.merge(unk_df, ref_df, on='wavelength', suffixes=('_unk', '_ref'))
    if merged.empty:
        return float('inf')  # Cannot compare if no overlap
    mae = np.mean(np.abs(merged['intensity_unk'] - merged['intensity_ref']))
    return mae

def extract_gem_id(full_name):
    """
    Extracts the numeric gem ID from a full_name string.
    """
    return ''.join(filter(str.isdigit, full_name))

def load_unknown_csv_files(files_dict):
    """
    Loads and normalizes unknown spectra from CSV files for each light source.
    """
    unknown_spectra = {}
    for src, file in files_dict.items():
        try:
            df = pd.read_csv(file, names=['wavelength', 'intensity'])
            df = normalize_spectrum(df, src)
            unknown_spectra[src] = df
            print(f"✅ Loaded and normalized {file}")
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
    return unknown_spectra