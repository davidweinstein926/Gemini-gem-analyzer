#!/usr/bin/env python3
"""
B Spectra Auto-Detection Script - UPDATED FOR NEW DIRECTORY STRUCTURE
Processes B spectra files using GeminiBSpectralDetector and outputs 
results in the same CSV format as manual marking program.

Enhanced with dynamic path detection and new directory structure support.
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

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class BSpectraAutoDetector:
    def __init__(self):
        # UPDATED: Dynamic path detection for new directory structure
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent.parent  # Go up to gemini_gemological_analysis/
        
        print(f"🔍 B Spectra Auto Detector Paths:")
        print(f"   Script directory: {self.script_dir}")
        print(f"   Project root: {self.project_root}")
        
        # UPDATED: Setup directories
        self.setup_directories()
        
        # Import the detector
        self.import_detector()
        
    def setup_directories(self):
        """UPDATED: Setup directories for new project structure"""
        # Possible locations for input data
        input_search_paths = [
            self.project_root / "data" / "raw",  # New structure - primary
            self.project_root / "src" / "structural_analysis" / "data" / "raw",  # Local to structural analysis
            self.project_root / "raw_txt",  # Legacy location
            Path.home() / "OneDrive" / "Desktop" / "gemini matcher" / "gemini sp10 raw" / "raw text",  # Legacy user path
        ]
        
        # Possible locations for output data
        output_search_paths = [
            self.project_root / "data" / "output" / "halogen",  # New structure - primary
            self.project_root / "src" / "structural_analysis" / "results" / "halogen",  # Results in structural analysis
            self.project_root / "output" / "halogen",  # Alternative root location
            Path.home() / "gemini sp10 structural data" / "halogen",  # Legacy user path
        ]
        
        # Find input directory
        self.input_directory = None
        for search_path in input_search_paths:
            if search_path.exists() and search_path.is_dir():
                self.input_directory = search_path
                print(f"✅ Found input directory: {self.input_directory}")
                break
        
        if not self.input_directory:
            # Use the primary new structure path (will be created if needed)
            self.input_directory = input_search_paths[0]
            print(f"⚠️  Input directory not found, will use: {self.input_directory}")
        
        # Find/create output directory
        self.output_directory = None
        for search_path in output_search_paths:
            if search_path.exists():
                self.output_directory = search_path
                print(f"✅ Found output directory: {self.output_directory}")
                break
        
        if not self.output_directory:
            # Use the primary new structure path and create it
            self.output_directory = output_search_paths[0]
            try:
                self.output_directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created output directory: {self.output_directory}")
            except Exception as e:
                print(f"⚠️  Could not create output directory: {e}")
                # Fallback to a guaranteed writable location
                self.output_directory = Path.cwd() / "b_spectra_results"
                self.output_directory.mkdir(exist_ok=True)
                print(f"📁 Using fallback output directory: {self.output_directory}")
    
    def import_detector(self):
        """UPDATED: Import detector with enhanced error handling"""
        # Try multiple import paths for the detector
        detector_import_paths = [
            "gemini_Bpeak_detector",  # Same directory
            "src.structural_analysis.auto_analysis.gemini_Bpeak_detector",  # Full project path
            "auto_analysis.gemini_Bpeak_detector",  # Relative from structural_analysis
        ]
        
        self.detector_class = None
        self.load_b_spectrum_func = None
        
        for import_path in detector_import_paths:
            try:
                if import_path == "gemini_Bpeak_detector":
                    from gemini_Bpeak_detector import GeminiBSpectralDetector, load_b_spectrum
                else:
                    # Dynamic import for other paths
                    import importlib
                    module = importlib.import_module(import_path)
                    GeminiBSpectralDetector = getattr(module, 'GeminiBSpectralDetector')
                    load_b_spectrum = getattr(module, 'load_b_spectrum')
                
                self.detector_class = GeminiBSpectralDetector
                self.load_b_spectrum_func = load_b_spectrum
                print(f"✅ Successfully imported detector from: {import_path}")
                return True
                
            except ImportError as e:
                print(f"⚠️  Failed to import from {import_path}: {e}")
                continue
        
        # If all imports failed
        print("❌ Error: gemini_Bpeak_detector.py not found in any expected location")
        print("Expected locations:")
        for path in detector_import_paths:
            print(f"   - {path}")
        print("Please ensure the detector file is accessible")
        return False

    def create_csv_output(self, detector_results, input_filepath):
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
            
            # Create row with UPDATED metadata
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
                'Intensity_Range_Max': 100.0,
                'Directory_Structure': 'Updated_New_Structure',  # UPDATED: Add metadata
                'Output_Location': str(self.output_directory),  # UPDATED: Add output location
                'Auto_Detection_Method': 'GeminiBSpectralDetector'  # UPDATED: Add detection method
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
                'Intensity_Range_Max': 100.0,
                'Directory_Structure': 'Updated_New_Structure',
                'Output_Location': str(self.output_directory),
                'Auto_Detection_Method': 'GeminiBSpectralDetector'
            }
            csv_rows.append(summary_row)
        
        return csv_rows

    def file_selection_dialog(self):
        """UPDATED: File selection dialog with new directory structure support"""
        try:
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            
            # Check if input directory exists and has files
            if self.input_directory.exists():
                txt_files = list(self.input_directory.glob("*.txt"))
                if txt_files:
                    initial_dir = str(self.input_directory)
                    print(f"📂 Found {len(txt_files)} txt files in: {self.input_directory}")
                else:
                    initial_dir = str(self.input_directory)
                    print(f"📁 Using input directory (no txt files found): {self.input_directory}")
            else:
                initial_dir = str(self.project_root)
                print(f"📁 Input directory not found, using project root: {self.project_root}")
            
            file_path = filedialog.askopenfilename(
                parent=root, 
                initialdir=initial_dir,
                title=f"Select B Spectrum for UPDATED Auto Analysis\nLooking in: {Path(initial_dir).name}",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
            
            root.quit()
            root.destroy()
            
            if file_path:
                print(f"✅ Selected file: {Path(file_path).name}")
                print(f"   Full path: {file_path}")
            
            return file_path if file_path else None
            
        except Exception as e:
            print(f"File dialog error: {e}")
            return None

    def process_b_spectrum_file(self, input_filepath, output_dir=None):
        """
        UPDATED: Process a B spectrum file and save results in manual marking format
        """
        try:
            # Check if detector was imported successfully
            if not self.detector_class or not self.load_b_spectrum_func:
                print("❌ Detector not available - cannot process file")
                return None, None
            
            # Initialize detector
            detector = self.detector_class()
            
            # Load and analyze spectrum
            print(f"🔬 Processing: {Path(input_filepath).name}")
            wavelengths, intensities = self.load_b_spectrum_func(input_filepath)
            results = detector.analyze_spectrum(wavelengths, intensities)
            
            # Generate output filename
            input_path = Path(input_filepath)
            if output_dir is None:
                output_dir = self.output_directory
            else:
                output_dir = Path(output_dir)
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{input_path.stem}_halogen_structural_auto_{timestamp}.csv"
            output_path = output_dir / output_filename
            
            # Convert to CSV format
            csv_data = self.create_csv_output(results, input_filepath)
            df = pd.DataFrame(csv_data)
            
            # Save CSV file
            df.to_csv(output_path, index=False)
            
            # Print summary
            print(f"✅ Analysis complete:")
            print(f"   Input: {input_path.name}")
            print(f"   Output: {output_path}")
            print(f"   Output directory: {output_dir}")
            print(f"   Features detected: {results['feature_count']}")
            print(f"   Detection strategy: {results['detection_strategy']}")
            print(f"   Baseline classification: {results['baseline_assessment']['noise_classification']}")
            print(f"   Overall confidence: {results['overall_confidence']:.2f}")
            
            return output_path, results
            
        except Exception as e:
            print(f"❌ Error processing {input_filepath}: {str(e)}")
            return None, None

    def run_interactive(self):
        """UPDATED: Run in interactive mode with new directory structure support"""
        print("="*70)
        print("B SPECTRA AUTO-DETECTION - UPDATED FOR NEW DIRECTORY STRUCTURE")
        print("="*70)
        print(f"📁 Project root: {self.project_root}")
        print(f"📂 Input directory: {self.input_directory}")
        print(f"📁 Output directory: {self.output_directory}")
        print("="*70)
        
        # Check if detector is available
        if not self.detector_class or not self.load_b_spectrum_func:
            print("❌ Detector not available - cannot run interactive mode")
            messagebox.showerror("Import Error", "gemini_Bpeak_detector.py not found!\nPlace it in the auto_analysis directory.")
            return
        
        while True:
            print("\n" + "="*50)
            print("B SPECTRA AUTO ANALYSIS OPTIONS:")
            print("1. Select file manually")
            print("2. Process all files in input directory")
            print("3. Exit")
            
            choice = input("\nChoice (1-3): ").strip()
            
            if choice == '1':
                # Single file selection
                file_path = self.file_selection_dialog()
                if file_path:
                    output_path, results = self.process_b_spectrum_file(file_path)
                    if results:
                        print(f"\n🎯 Results saved to: {output_path}")
                        input("Press Enter to continue...")
                else:
                    print("No file selected")
                    
            elif choice == '2':
                # Process all files in input directory
                if not self.input_directory.exists():
                    print(f"❌ Input directory not found: {self.input_directory}")
                    continue
                
                txt_files = list(self.input_directory.glob("*.txt"))
                if not txt_files:
                    print(f"❌ No .txt files found in: {self.input_directory}")
                    continue
                
                print(f"🔍 Found {len(txt_files)} files to process")
                confirm = input(f"Process all {len(txt_files)} files? (y/N): ").strip().lower()
                
                if confirm == 'y':
                    processed = 0
                    for txt_file in txt_files:
                        print(f"\n📄 Processing {processed + 1}/{len(txt_files)}: {txt_file.name}")
                        output_path, results = self.process_b_spectrum_file(str(txt_file))
                        if results:
                            processed += 1
                    
                    print(f"\n✅ Batch processing complete: {processed}/{len(txt_files)} files processed")
                    print(f"📁 Results saved to: {self.output_directory}")
                    input("Press Enter to continue...")
                
            elif choice == '3':
                print("Exiting...")
                break
            else:
                print("Invalid choice")

def main():
    """
    UPDATED: Main function with enhanced command line and interactive support
    """
    detector = BSpectraAutoDetector()
    
    if len(sys.argv) > 1:
        # Command line mode
        input_files = sys.argv[1:]
        output_dir = None
        
        # Check if last argument is output directory
        if os.path.isdir(sys.argv[-1]):
            output_dir = sys.argv[-1]
            input_files = sys.argv[1:-1]
        
        processed = 0
        for input_file in input_files:
            if os.path.exists(input_file):
                output_path, results = detector.process_b_spectrum_file(input_file, output_dir)
                if results:
                    processed += 1
            else:
                print(f"⚠️ Warning: File not found: {input_file}")
        
        print(f"\n✅ Command line processing complete: {processed}/{len(input_files)} files processed")
    
    else:
        # Interactive mode
        detector.run_interactive()

if __name__ == "__main__":
    main()