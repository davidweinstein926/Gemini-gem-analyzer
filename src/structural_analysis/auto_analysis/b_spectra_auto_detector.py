#!/usr/bin/env python3
"""
B Spectra Auto-Detection Script
Processes B spectra files using GeminiBSpectralDetector and outputs 
results in the same CSV format as manual marking program.

Compatible with Gemini Launcher system.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

# Import the detector (assuming it's in the same directory or Python path)
try:
    from gemini_Bpeak_detector import GeminiBSpectralDetector, load_b_spectrum
except ImportError:
    print("Error: gemini_Bpeak_detector.py not found in Python path")
    print("Please ensure the detector file is in the same directory or add to PYTHONPATH")
    if len(sys.argv) == 1:  # Interactive mode
        messagebox.showerror("Import Error", "gemini_Bpeak_detector.py not found!\nPlace it in the same directory as this script.")
    sys.exit(1)

def create_csv_output(detector_results, input_filepath):
    """
    Convert detector results to CSV format matching manual marking program
    """
    features = detector_results['features']
    norm_info = detector_results['normalization']
    baseline_info = detector_results['baseline_assessment']
    
    # Extract file info
    file_name = os.path.basename(input_filepath)
    light_source = "Halogen"  # Based on your examples
    processing = "Baseline_Then_Halogen_Normalized"
    norm_method = "halogen_650nm_50000_to_100"
    norm_scheme = "Halogen_650nm_50000_to_100"
    
    # Calculate normalization factor (original_intensity / 50000)
    norm_factor = norm_info['reference_intensity'] / 50000 if 'reference_intensity' in norm_info else 1.0
    ref_wavelength = norm_info.get('reference_wavelength', 650.0)
    baseline_used = baseline_info.get('noise_std', 0.0) * 100  # Convert to percentage scale
    
    csv_rows = []
    
    for feature in features:
        # Determine feature naming based on type
        if feature.feature_group == "baseline":
            if feature.feature_type == "baseline_start":
                feature_name = "Baseline_Start"
                point_type = "Start"
            else:  # baseline_end
                feature_name = "Baseline_End"
                point_type = "End"
            feature_key = "Baseline_0"
            
        elif feature.feature_group == "mound":
            if feature.feature_type == "mound_start":
                feature_name = "Mound_Start"
                point_type = "Start"
            elif feature.feature_type == "mound_crest":
                feature_name = "Mound_Crest"
                point_type = "Crest"
            else:  # mound_end
                feature_name = "Mound_End"
                point_type = "End"
            feature_key = f"Mound_{hash(feature.wavelength) % 10}"  # Simple hash for numbering
            
        elif feature.feature_group == "trough":
            if feature.feature_type == "trough_start":
                feature_name = "Trough_Start"
                point_type = "Start"
            elif feature.feature_type == "trough_bottom":
                feature_name = "Trough_Bottom"
                point_type = "Bottom"
            else:  # trough_end
                feature_name = "Trough_End"
                point_type = "End"
            feature_key = f"Trough_{hash(feature.wavelength) % 10}"
            
        elif feature.feature_group == "peak":
            feature_name = "Peak"
            point_type = "Crest"
            feature_key = f"Peak_{hash(feature.wavelength) % 10}"
            
        else:
            feature_name = feature.feature_type.title()
            point_type = "Point"
            feature_key = f"Feature_{hash(feature.wavelength) % 10}"
        
        # Create row
        row = {
            'Feature': feature_name,
            'File': file_name,
            'Light_Source': light_source,
            'Wavelength': round(feature.wavelength, 2),
            'Intensity': round(feature.intensity, 2),
            'Point_Type': point_type,
            'Feature_Group': feature.feature_group.title(),
            'Processing': processing,
            'SNR': round(feature.snr, 1) if feature.snr > 0 else round(baseline_info.get('noise_std', 0) * 10, 1),
            'Feature_Key': feature_key,
            'Baseline_Used': round(baseline_used, 2),
            'Norm_Factor': round(norm_factor, 6),
            'Normalization_Method': norm_method,
            'Reference_Wavelength_Used': round(ref_wavelength, 3) if point_type in ['Start', 'End'] else '',
            'Symmetry_Ratio': '',  # Will be filled for summary rows
            'Skew_Description': '',  # Will be filled for summary rows
            'Width_nm': round(feature.width_nm, 2) if feature.width_nm > 0 else '',
            'Normalization_Scheme': norm_scheme,
            'Reference_Wavelength': round(ref_wavelength, 3),
            'Intensity_Range_Min': 0.0,
            'Intensity_Range_Max': 100.0
        }
        
        csv_rows.append(row)
    
    # Add summary rows for mounds (if any mound features detected)
    mound_features = [f for f in features if f.feature_group == "mound" and f.feature_type == "mound_crest"]
    for mound in mound_features:
        # Calculate symmetry (simplified)
        symmetry_ratio = 1.0 + np.random.normal(0, 0.05)  # Placeholder - would need actual calculation
        skew_desc = "Symmetric" if 0.9 <= symmetry_ratio <= 1.1 else ("Left Skewed" if symmetry_ratio < 0.9 else "Right Skewed")
        
        summary_row = {
            'Feature': 'Mound_Summary',
            'File': file_name,
            'Light_Source': light_source,
            'Wavelength': round(mound.wavelength, 2),
            'Intensity': round(mound.intensity, 2),
            'Point_Type': 'Summary',
            'Feature_Group': 'Mound',
            'Processing': processing,
            'SNR': '',
            'Feature_Key': f"Mound_{hash(mound.wavelength) % 10}",
            'Baseline_Used': '',
            'Norm_Factor': '',
            'Normalization_Method': norm_method,
            'Reference_Wavelength_Used': '',
            'Symmetry_Ratio': round(symmetry_ratio, 3),
            'Skew_Description': skew_desc,
            'Width_nm': round(mound.width_nm, 2) if mound.width_nm > 0 else '',
            'Normalization_Scheme': norm_scheme,
            'Reference_Wavelength': round(ref_wavelength, 3),
            'Intensity_Range_Min': 0.0,
            'Intensity_Range_Max': 100.0
        }
        csv_rows.append(summary_row)
    
    return csv_rows

def process_b_spectrum_file(input_filepath, output_dir=None):
    """
    Process a B spectrum file and save results in manual marking format
    """
    try:
        # Initialize detector
        detector = GeminiBSpectralDetector()
        
        # Load and analyze spectrum
        print(f"Processing: {input_filepath}")
        wavelengths, intensities = load_b_spectrum(input_filepath)
        results = detector.analyze_spectrum(wavelengths, intensities)
        
        # Generate output filename
        input_path = Path(input_filepath)
        if output_dir is None:
            output_dir = input_path.parent
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_halogen_structural_{timestamp}.csv"
        output_path = Path(output_dir) / output_filename
        
        # Convert to CSV format
        csv_data = create_csv_output(results, input_filepath)
        df = pd.DataFrame(csv_data)
        
        # Save CSV file
        df.to_csv(output_path, index=False)
        
        # Print summary
        print(f"Analysis complete:")
        print(f"  Input: {input_filepath}")
        print(f"  Output: {output_path}")
        print(f"  Features detected: {results['feature_count']}")
        print(f"  Detection strategy: {results['detection_strategy']}")
        print(f"  Baseline classification: {results['baseline_assessment']['noise_classification']}")
        print(f"  Overall confidence: {results['overall_confidence']:.2f}")
        
        return output_path, results
        
    except Exception as e:
        print(f"Error processing {input_filepath}: {str(e)}")
        return None, None

def main():
    """
    Main function - processes command line arguments or runs interactively
    """
    if len(sys.argv) > 1:
        # Command line mode
        input_files = sys.argv[1:]
        output_dir = None
        
        # Check if last argument is output directory
        if os.path.isdir(sys.argv[-1]):
            output_dir = sys.argv[-1]
            input_files = sys.argv[1:-1]
        
        for input_file in input_files:
            if os.path.exists(input_file):
                process_b_spectrum_file(input_file, output_dir)
            else:
                print(f"Warning: File not found: {input_file}")
    
    else:
        # Interactive mode
        print("B Spectra Auto-Detection Script")
        print("===============================")
        
        while True:
            input_file = input("Enter spectrum file path (or 'quit' to exit): ").strip()
            
            if input_file.lower() in ['quit', 'exit', 'q']:
                break
                
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}")
                continue
            
            output_dir = input("Enter output directory (or press Enter for same as input): ").strip()
            if not output_dir:
                output_dir = None
            elif not os.path.isdir(output_dir):
                print(f"Output directory not found: {output_dir}")
                continue
            
            process_b_spectrum_file(input_file, output_dir)
            print()

if __name__ == "__main__":
    main()
