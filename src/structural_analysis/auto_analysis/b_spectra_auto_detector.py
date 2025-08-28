#!/usr/bin/env python3
"""
📊 B SPECTRA AUTO-DETECTION SCRIPT (Wrapper/CSV Output)
File: b_spectra_auto_detector.py
Processes files using gemini_Bpeak_detector.py and outputs CSV results

8 Structural Features:
- Peak: 3 points (Start, Crest, End)
- Plateau: 3 points (Start, Mid, End) 
- Shoulder: 3 points (Start, Mid, End)
- Trough: 3 points (Start, Bottom, End) - B/H only
- Mound: 4 points (Start, Crest, End, Summary) - only Summary row
- Baseline: 2 points (Start, End) - 300-350nm
- Diagnostic Region: 2 points (Start, End) - complex regions
- Valley: 1 point (Mid) - midpoint between features

Version: 3.0 (Standardized point structures for CSV/DB compatibility)
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    from gemini_Bpeak_detector import GeminiBSpectralDetector, load_b_spectrum
except ImportError:
    print("Error: gemini_Bpeak_detector.py not found")
    sys.exit(1)

def create_csv_output(detector_results, input_filepath):
    """
    Convert detector results to streamlined CSV format matching manual marking preferences
    
    REMOVED COLUMNS (H,I,K,L,M,Q,R,S,T,AB,AC,AD,AG,AH,AI,AJ,AK,AL,AM,AN,AO):
    - Processing, Feature_Key, Baseline_Quality, Baseline_Width_nm, Baseline_CV_Percent
    - Normalization_Method, Target_Ref_Intensity, Click_Order, Reference_Wavelength_Used  
    - Normalization_Scheme, Reference_Wavelength, Analyzer_Version
    - Intensity_Range_Min/Max, Target_Reference_Intensity, Scaling_Factor
    - Total_Spectrum_Points, Wavelength_Range_Min/Max, Wavelength_Span_nm, Baseline_SNR
    
    KEPT COLUMNS: Feature, File, Light_Source, Wavelength, Intensity, Point_Type,
    Feature_Group, SNR, Baseline_Std_Dev, Baseline_Used, Norm_Factor, width info,
    symmetry info, Analysis_Date, Analysis_Time
    """
    features = detector_results['features']
    norm_info = detector_results['normalization']
    
    # File metadata
    file_name = os.path.basename(input_filepath)
    light_source = "B"  # Halogen/Broadband
    
    # Normalization data
    norm_factor = norm_info['reference_intensity'] / 50000 if 'reference_intensity' in norm_info else 1.0
    global_baseline = detector_results.get('global_baseline', 0.0)
    
    csv_rows = []
    
    for feature in features:
        # Standardized feature naming
        feature_name = f"{feature.feature_type.title().replace('_', ' ')}_{feature.point_type}"
        
        # Handle specific naming conventions to match manual marking
        if feature.feature_type == "baseline":
            feature_name = f"Baseline_{feature.point_type}"
        elif feature.feature_type == "peak":
            feature_name = f"Peak_{feature.point_type}" if feature.point_type != "Crest" else "Peak"
        elif feature.feature_type == "mound":
            if feature.point_type == "Summary":
                feature_name = "Mound_Summary"
            else:
                feature_name = f"Mound_{feature.point_type}"
        elif feature.feature_type == "plateau":
            feature_name = f"Plateau_{feature.point_type}"
        elif feature.feature_type == "shoulder":
            feature_name = f"Shoulder_{feature.point_type}" if feature.point_type != "Mid" else "Shoulder"
        elif feature.feature_type == "trough":
            feature_name = f"Trough_{feature.point_type}"
        elif feature.feature_type == "valley":
            feature_name = "Valley"  # Only 1 point (Mid)
        elif feature.feature_type == "diagnostic_region":
            feature_name = f"Diagnostic_{feature.point_type}"  # Start or End only
        
        # Calculate symmetry for summary rows (only mounds have summary)
        symmetry_ratio = ''
        skew_description = ''
        skew_severity = ''
        width_class = ''
        
        if feature.point_type == "Summary" and feature.feature_type == "mound":
            # Calculate symmetry based on feature geometry
            if feature.width_nm > 0:
                mid_pos = (feature.start_wavelength + feature.end_wavelength) / 2
                crest_offset = abs(feature.wavelength - mid_pos)
                symmetry_ratio = round(1.0 + (crest_offset / (feature.width_nm / 2)) * 0.1, 3)
                
                if 0.95 <= symmetry_ratio <= 1.05:
                    skew_description = "Symmetric"
                    skew_severity = "None"
                elif symmetry_ratio < 0.95:
                    skew_description = "Left_Skewed"
                    skew_severity = "Moderate" if symmetry_ratio < 0.85 else "Mild"
                else:
                    skew_description = "Right_Skewed"
                    skew_severity = "Moderate" if symmetry_ratio > 1.15 else "Mild"
                
                # Width classification
                if feature.width_nm < 100:
                    width_class = "Narrow"
                elif feature.width_nm < 300:
                    width_class = "Medium"
                else:
                    width_class = "Wide"
        
        # Calculate left and right widths for mound features
        left_width_nm = ''
        right_width_nm = ''
        total_width_nm = ''
        
        if feature.feature_type == "mound" and feature.width_nm > 0:
            total_width_nm = round(feature.width_nm, 2)
            if feature.point_type == "Crest":
                left_width_nm = round(feature.wavelength - feature.start_wavelength, 2)
                right_width_nm = round(feature.end_wavelength - feature.wavelength, 2)
        
        # STREAMLINED CSV - Removed unwanted columns H,I,K,L,M,Q,R,S,T,AB,AC,AD,AG,AH,AI,AJ,AK,AL,AM,AN,AO
        row = {
            'Feature': feature_name,                              # A
            'File': file_name,                                   # B  
            'Light_Source': light_source,                        # C
            'Wavelength': round(feature.wavelength, 2),          # D
            'Intensity': round(feature.intensity, 2),            # E
            'Point_Type': feature.point_type,                    # F
            'Feature_Group': feature.feature_group.title(),      # G
            # H (Processing) - REMOVED
            # I (Feature_Key) - REMOVED  
            'SNR': round(feature.snr, 1) if feature.snr > 0 else 0.0,  # J (was column 10)
            # K (Baseline_Quality) - REMOVED
            # L (Baseline_Width_nm) - REMOVED
            # M (Baseline_CV_Percent) - REMOVED
            'Baseline_Std_Dev': 0.0,                            # N (was column 14)
            'Baseline_Used': round(global_baseline, 2),         # O (was column 15)
            'Norm_Factor': round(norm_factor, 6),               # P (was column 16)
            # Q (Normalization_Method) - REMOVED
            # R (Target_Ref_Intensity) - REMOVED
            # S (Click_Order) - REMOVED  
            # T (Reference_Wavelength_Used) - REMOVED
            'total_width_nm': total_width_nm,                   # U (was column 21)
            'left_width_nm': left_width_nm,                     # V (was column 22)
            'right_width_nm': right_width_nm,                   # W (was column 23)
            'symmetry_ratio': symmetry_ratio,                   # X (was column 24)
            'skew_description': skew_description,               # Y (was column 25)
            'skew_severity': skew_severity,                     # Z (was column 26)
            'width_class': width_class,                         # AA (was column 27)
            # AB (Normalization_Scheme) - REMOVED
            # AC (Reference_Wavelength) - REMOVED
            # AD (Analyzer_Version) - REMOVED
            'Analysis_Date': datetime.now().strftime("%Y-%m-%d"), # AE (was column 31)
            'Analysis_Time': datetime.now().strftime("%H:%M:%S")  # AF (was column 32)
            # AG through AO - ALL REMOVED
        }
        
        csv_rows.append(row)
    
    return csv_rows

def validate_point_structure(features):
    """Validate that detected features have correct point structures"""
    feature_counts = {}
    
    for feature in features:
        key = feature.feature_type
        if key not in feature_counts:
            feature_counts[key] = []
        feature_counts[key].append(feature.point_type)
    
    # Define expected point structures
    expected_structures = {
        'peak': ['Start', 'Crest', 'End'],                    # 3 points
        'plateau': ['Start', 'Mid', 'End'],                   # 3 points
        'shoulder': ['Start', 'Mid', 'End'],                  # 3 points
        'trough': ['Start', 'Bottom', 'End'],                 # 3 points
        'mound': ['Start', 'Crest', 'End', 'Summary'],        # 4 points (only Summary)
        'baseline': ['Start', 'End'],                         # 2 points
        'diagnostic_region': ['Start', 'End'],                # 2 points
        'valley': ['Mid']                                     # 1 point
    }
    
    validation_results = {}
    
    for feature_type, detected_points in feature_counts.items():
        expected_points = expected_structures.get(feature_type, [])
        
        # Count occurrences of each point type
        point_counts = {}
        for point in detected_points:
            point_counts[point] = point_counts.get(point, 0) + 1
        
        # Check if structure matches expected
        is_valid = True
        issues = []
        
        for expected_point in expected_points:
            if expected_point not in point_counts:
                is_valid = False
                issues.append(f"Missing {expected_point}")
            elif point_counts[expected_point] > 1 and feature_type != 'valley':
                # Multiple instances allowed for valley, but not others per feature instance
                pass  # This is actually ok if we have multiple features of same type
        
        validation_results[feature_type] = {
            'valid': is_valid,
            'expected': expected_points,
            'detected': list(point_counts.keys()),
            'issues': issues
        }
    
    return validation_results

def process_b_spectrum_file(input_filepath, output_dir="c:\\users\\david\\gemini sp10 structural data\\B\\"):
    """Process B spectrum file with standardized structural analysis"""
    try:
        # Initialize detector
        detector = GeminiBSpectralDetector()
        
        # Load and analyze
        print(f"Processing: {input_filepath}")
        wavelengths, intensities = load_b_spectrum(input_filepath)
        results = detector.analyze_spectrum(wavelengths, intensities)
        
        # Validate point structures
        validation = validate_point_structure(results['features'])
        
        # Generate output filename
        input_path = Path(input_filepath)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{input_path.stem}_halogen_structural_{timestamp}.csv"
        
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename
        
        # Convert to CSV and save
        csv_data = create_csv_output(results, input_filepath)
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
        
        # Print detailed summary
        feature_summary = results['feature_summary']['by_type'] if 'by_type' in results['feature_summary'] else {}
        
        print(f"✓ Analysis complete:")
        print(f"  Output: {output_path}")
        print(f"  Total points: {results['feature_count']}")
        print(f"  Strategy: {results['detection_strategy']}")
        print(f"  Overall confidence: {results['overall_confidence']:.2f}")
        
        # Feature breakdown with point counts
        if feature_summary:
            print(f"  Detected features:")
            for feature_type, count in feature_summary.items():
                expected_points = {
                    'peak': 3, 'plateau': 3, 'shoulder': 3, 'trough': 3,
                    'mound': 4, 'baseline': 2, 'diagnostic_region': 2, 'valley': 1
                }.get(feature_type, '?')
                
                feature_instances = count // expected_points if expected_points != '?' else count
                print(f"    {feature_type}: {feature_instances} instances ({count} points)")
        
        # Validation results
        print(f"  Point structure validation:")
        all_valid = True
        for feature_type, val_result in validation.items():
            status = "✓" if val_result['valid'] else "✗"
            print(f"    {feature_type}: {status} {len(val_result['detected'])} point types")
            if val_result['issues']:
                print(f"      Issues: {', '.join(val_result['issues'])}")
            all_valid = all_valid and val_result['valid']
        
        if all_valid:
            print("  ✓ All feature point structures match manual marking standards")
        else:
            print("  ⚠ Some feature structures need adjustment")
        
        return output_path, results
        
    except Exception as e:
        print(f"✗ Error processing {input_filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function with validation reporting"""
    print("📊 B SPECTRA AUTO-DETECTION SCRIPT - Wrapper/CSV Output v3.0")
    print("File: b_spectra_auto_detector.py")
    print("Point structures match manual marking exactly:")
    print("  Peak(3), Plateau(3), Shoulder(3), Trough(3)")
    print("  Mound(4+Summary), Baseline(2), Diagnostic(2), Valley(1)")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        # Command line mode
        input_files = sys.argv[1:]
        
        for input_file in input_files:
            if os.path.exists(input_file):
                process_b_spectrum_file(input_file)
                print()  # Add spacing between files
            else:
                print(f"File not found: {input_file}")
    
    else:
        # Interactive mode with file dialog
        print("\n🔍 Select spectrum file for analysis...")
        
        # Create root window for file dialog (hidden)
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        while True:
            # Open file dialog
            file_types = [
                ('Spectrum files', '*.txt *.csv'),
                ('Text files', '*.txt'),
                ('CSV files', '*.csv'),
                ('All files', '*.*')
            ]
            
            input_file = filedialog.askopenfilename(
                title="Select Gemini SP10 Raw Spectrum File",
                filetypes=file_types,
                initialdir=r"C:\users\david\gemini sp10 raw"
            )
            
            if not input_file:  # User cancelled
                print("Analysis cancelled by user.")
                break
                
            if not os.path.exists(input_file):
                messagebox.showerror("File Error", f"File not found: {input_file}")
                continue
            
            # Process the selected file
            print(f"\n📁 Selected file: {os.path.basename(input_file)}")
            result = process_b_spectrum_file(input_file)
            
            if result[0]:  # Success
                # Ask if user wants to analyze another file
                another = messagebox.askyesno("Analysis Complete", 
                    f"Analysis completed successfully!\n\nWould you like to analyze another spectrum file?")
                if not another:
                    break
            else:
                # Ask if user wants to try another file after error
                retry = messagebox.askyesno("Analysis Failed", 
                    f"Analysis failed. Would you like to try another file?")
                if not retry:
                    break
        
        root.destroy()

if __name__ == "__main__":
    main()
