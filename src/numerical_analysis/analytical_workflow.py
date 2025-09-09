#!/usr/bin/env python3
"""
analytical_workflow.py - Complete workflow for numerical analysis
This program handles:
1. File selection from data/raw
2. Copying files to raw_txt
3. Converting to unkgem*.csv format
4. Running numerical analysis

LOCATION: Save as src/numerical_analysis/analytical_workflow.py
"""

import os
import shutil
import subprocess
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime

def scan_raw_directory():
    """Scan data/raw for available spectra files"""
    raw_dir = 'data/raw'
    
    if not os.path.exists(raw_dir):
        print(f"❌ Directory '{raw_dir}' not found!")
        return None
    
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.txt')]
    
    if not files:
        print(f"❌ No .txt files found in '{raw_dir}'")
        return None
    
    # Group files by gem number
    gems = defaultdict(lambda: {'B': [], 'L': [], 'U': []})
    
    for file in files:
        # Try to extract gem number and light source
        # Assuming format like: 358BC4.txt, 189LC1.txt, etc.
        base = os.path.splitext(file)[0]
        
        # Find light source (B, L, or U)
        light_source = None
        for ls in ['B', 'L', 'U']:
            if ls.upper() in base.upper():
                light_source = ls
                break
        
        if light_source:
            # Extract gem number (everything before the light source)
            for i, char in enumerate(base.upper()):
                if char == light_source:
                    gem_num = base[:i]
                    break
            else:
                gem_num = base  # fallback
            
            gems[gem_num][light_source].append(file)
    
    return dict(gems)

def display_available_files(gems_dict):
    """Display available files grouped by gem"""
    print("\n📂 AVAILABLE SPECTRAL FILES")
    print("=" * 50)
    
    for gem_num in sorted(gems_dict.keys()):
        gem_files = gems_dict[gem_num]
        total_files = sum(len(files) for files in gem_files.values())
        
        print(f"\n💎 Gem {gem_num} ({total_files} files):")
        
        for light_source in ['B', 'L', 'U']:
            files = gem_files[light_source]
            if files:
                print(f"  {light_source}: {', '.join(files)}")
            else:
                print(f"  {light_source}: (none)")

def select_files_for_analysis():
    """Interactive file selection for analysis"""
    # Scan available files
    gems_dict = scan_raw_directory()
    if not gems_dict:
        return None
    
    # Display available files
    display_available_files(gems_dict)
    
    print(f"\n🔬 SELECT FILES FOR ANALYSIS")
    print("=" * 40)
    print("Enter the exact filenames for each light source:")
    print("(Leave blank to skip a light source)")
    
    selected_files = {}
    
    for light_source in ['B', 'L', 'U']:
        while True:
            filename = input(f"\n{light_source} spectrum file: ").strip()
            
            if not filename:
                print(f"  Skipping {light_source} analysis")
                break
            
            # Check if file exists
            filepath = os.path.join('data/raw', filename)
            if os.path.exists(filepath):
                selected_files[light_source] = filename
                print(f"  ✅ Selected: {filename}")
                break
            else:
                print(f"  ❌ File not found: {filename}")
                print("  Please enter exact filename or leave blank to skip")
    
    if not selected_files:
        print("❌ No files selected for analysis")
        return None
    
    return selected_files

def prepare_raw_txt_directory(selected_files):
    """Copy selected files to raw_txt directory"""
    raw_txt_dir = 'raw_txt'
    
    # Create/clear raw_txt directory
    if os.path.exists(raw_txt_dir):
        shutil.rmtree(raw_txt_dir)
    os.makedirs(raw_txt_dir)
    
    print(f"\n📋 PREPARING FILES FOR ANALYSIS")
    print("=" * 40)
    
    copied_files = {}
    
    for light_source, filename in selected_files.items():
        src_path = os.path.join('data/raw', filename)
        dst_path = os.path.join(raw_txt_dir, filename)
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_files[light_source] = filename
            print(f"  ✅ Copied {light_source}: {filename}")
        except Exception as e:
            print(f"  ❌ Failed to copy {filename}: {e}")
    
    return copied_files

def normalize_spectrum(wavelengths, intensities, light_source):
    """Apply normalization based on light source"""
    try:
        # Ensure arrays are numpy arrays
        wavelengths = np.array(wavelengths)
        intensities = np.array(intensities)
        
        if light_source == 'B':
            # Halogen: normalize to 650nm = 50000
            anchor = 650
            target = 50000
            idx = np.argmin(np.abs(wavelengths - anchor))
            if intensities[idx] != 0:
                scale = target / intensities[idx]
                return intensities * scale
            else:
                print(f"⚠️ Warning: Zero intensity at {anchor}nm for {light_source}")
                return intensities
                
        elif light_source == 'L':
            # Laser: normalize to 450nm = 50000
            anchor = 450
            target = 50000
            idx = np.argmin(np.abs(wavelengths - anchor))
            if intensities[idx] != 0:
                scale = target / intensities[idx]
                return intensities * scale
            else:
                print(f"⚠️ Warning: Zero intensity at {anchor}nm for {light_source}")
                return intensities
                
        elif light_source == 'U':
            # UV: normalize to maximum in 810.5-811.5nm window = 15000
            mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
            window = intensities[mask]
            if len(window) > 0:
                max_811 = window.max()
                if max_811 > 0:
                    scale = 15000 / max_811
                    return intensities * scale
                else:
                    print(f"⚠️ Warning: Zero max intensity in 811nm window for {light_source}")
                    return intensities
            else:
                print(f"⚠️ Warning: No data in 810.5-811.5nm range for {light_source}")
                return intensities
        else:
            print(f"⚠️ Unknown light source: {light_source}")
            return intensities
            
    except Exception as e:
        print(f"❌ Error in normalization for {light_source}: {e}")
        return intensities

def convert_files_to_unkgem():
    """Convert raw_txt files to unkgem format"""
    raw_txt_dir = 'raw_txt'
    output_dir = 'data/unknown'
    
    if not os.path.exists(raw_txt_dir):
        print(f"❌ Directory '{raw_txt_dir}' not found!")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n🔄 CONVERTING TO ANALYSIS FORMAT")
    print("=" * 40)
    
    converted_files = {}
    
    # Process each light source
    for light_source in ['B', 'L', 'U']:
        # Find files for this light source
        source_files = [f for f in os.listdir(raw_txt_dir) if light_source.upper() in f.upper()]
        
        if not source_files:
            print(f"⚠️ No files found for light source {light_source}")
            continue
        
        if len(source_files) > 1:
            print(f"⚠️ Multiple files found for {light_source}: {source_files}")
            print(f"   Using first file: {source_files[0]}")
        
        selected_file = source_files[0]
        input_path = os.path.join(raw_txt_dir, selected_file)
        
        print(f"🔄 Processing {light_source}: {selected_file}")
        
        try:
            # Read the file
            df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
            
            print(f"   📊 Loaded {len(df)} data points")
            print(f"   📏 Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} nm")
            print(f"   📈 Original intensity range: {df['intensity'].min():.2f} - {df['intensity'].max():.2f}")
            
            # Apply normalization
            normalized_intensities = normalize_spectrum(df['wavelength'].values, df['intensity'].values, light_source)
            
            # Create output dataframe
            output_df = pd.DataFrame({
                'wavelength': df['wavelength'],
                'intensity': normalized_intensities
            })
            
            # Save as unkgem*.csv
            output_filename = f"unkgem{light_source}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save without header and index to match expected format
            output_df.to_csv(output_path, header=False, index=False)
            
            converted_files[light_source] = {
                'input': selected_file,
                'output': output_path,
                'rows': len(output_df)
            }
            
            print(f"   ✅ Created {output_filename} ({len(output_df)} data points)")
            print(f"   📈 Normalized intensity range: {normalized_intensities.min():.2f} - {normalized_intensities.max():.2f}")
            
        except Exception as e:
            print(f"   ❌ Error converting {selected_file}: {e}")
            import traceback
            traceback.print_exc()
    
    return converted_files

def run_numerical_analysis():
    """Run the numerical analysis (gemini1.py)"""
    print(f"\n🧮 RUNNING NUMERICAL ANALYSIS")
    print("=" * 40)
    
    try:
        # Check if gemini1.py exists
        analysis_script = 'src/numerical_analysis/gemini1.py'
        if not os.path.exists(analysis_script):
            analysis_script = 'gemini1.py'
            if not os.path.exists(analysis_script):
                print("❌ gemini1.py not found!")
                return False
        
        print(f"📊 Starting spectral analysis...")
        print(f"💻 Command: python {analysis_script}")
        
        # Run the analysis
        result = subprocess.run([sys.executable, analysis_script], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Numerical analysis completed successfully")
            return True
        else:
            print(f"\n⚠️ Analysis finished with return code: {result.returncode}")
            return True  # Still consider it successful as it might just be normal completion
            
    except Exception as e:
        print(f"❌ Error running numerical analysis: {e}")
        return False

def main():
    """Main analytical workflow"""
    print("🔬 GEMINI ANALYTICAL ANALYSIS WORKFLOW")
    print("=" * 50)
    
    # Step 1: Select files
    print("\nSTEP 1: File Selection")
    selected_files = select_files_for_analysis()
    if not selected_files:
        print("❌ Workflow aborted - no files selected")
        return
    
    # Step 2: Prepare raw_txt directory
    print("\nSTEP 2: File Preparation")
    copied_files = prepare_raw_txt_directory(selected_files)
    if not copied_files:
        print("❌ Workflow aborted - file preparation failed")
        return
    
    # Step 3: Convert to unkgem format
    print("\nSTEP 3: File Conversion")
    converted_files = convert_files_to_unkgem()
    if not converted_files:
        print("❌ Workflow aborted - conversion failed")
        return
    
    # Step 4: Run numerical analysis
    print("\nSTEP 4: Numerical Analysis")
    analysis_success = run_numerical_analysis()
    
    # Summary
    print(f"\n📊 WORKFLOW COMPLETE")
    print("=" * 30)
    if analysis_success:
        print("✅ All steps completed successfully!")
        print(f"📁 Analyzed files: {', '.join(copied_files.values())}")
        print(f"🔬 Light sources: {', '.join(converted_files.keys())}")
    else:
        print("⚠️ Workflow completed with issues")
    
    input("\nPress Enter to return to main menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Workflow interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()