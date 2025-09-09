#!/usr/bin/env python3
"""
quick_fix_conversion.py - Fix immediate conversion and path issues
This script will:
1. Convert your current raw_txt files to proper unkgem format
2. Put them in the correct location
3. Fix numpy import issues
"""

import pandas as pd
import numpy as np
import os
import shutil
from pathlib import Path

def normalize_spectrum(wavelengths, intensities, light_source):
    """Apply normalization based on light source - WITH NUMPY FIX"""
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

def find_raw_txt_files():
    """Find raw_txt directory and files"""
    possible_dirs = [
        'raw_txt',
        'data/raw_txt',
        '../raw_txt',
        '../../raw_txt',
        'src/numerical_analysis/raw_txt'
    ]
    
    for raw_dir in possible_dirs:
        if os.path.exists(raw_dir):
            txt_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.txt')]
            if txt_files:
                print(f"✅ Found raw_txt directory: {raw_dir}")
                print(f"   Files found: {txt_files}")
                return raw_dir, txt_files
    
    print("❌ No raw_txt directory with .txt files found")
    return None, None

def create_output_directory():
    """Create and ensure data/unknown directory exists"""
    output_dir = 'data/unknown'
    
    # Try multiple possible locations
    possible_parents = ['.', '..', '../..']
    
    for parent in possible_parents:
        full_path = os.path.join(parent, output_dir)
        try:
            os.makedirs(full_path, exist_ok=True)
            # Test if we can write to it
            test_file = os.path.join(full_path, 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            print(f"✅ Using output directory: {os.path.abspath(full_path)}")
            return full_path
        except Exception as e:
            print(f"❌ Cannot use {full_path}: {e}")
    
    # Fallback: create in current directory
    fallback_dir = 'unknown_output'
    os.makedirs(fallback_dir, exist_ok=True)
    print(f"⚠️ Using fallback directory: {os.path.abspath(fallback_dir)}")
    return fallback_dir

def convert_file(input_path, light_source):
    """Convert a single file with robust error handling"""
    try:
        print(f"\n🔄 Converting {light_source}: {input_path}")
        
        # Try different reading methods
        try:
            # Method 1: Space/tab separated
            df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
        except Exception as e1:
            try:
                # Method 2: Comma separated
                df = pd.read_csv(input_path, sep=',', header=None, names=['wavelength', 'intensity'])
            except Exception as e2:
                try:
                    # Method 3: Tab separated
                    df = pd.read_csv(input_path, sep='\t', header=None, names=['wavelength', 'intensity'])
                except Exception as e3:
                    print(f"❌ Could not read file with any separator method")
                    print(f"   Error 1 (space): {e1}")
                    print(f"   Error 2 (comma): {e2}")  
                    print(f"   Error 3 (tab): {e3}")
                    return None
        
        print(f"   📊 Loaded {len(df)} data points")
        print(f"   📏 Wavelength range: {df['wavelength'].min():.1f} - {df['wavelength'].max():.1f} nm")
        print(f"   📈 Intensity range: {df['intensity'].min():.2f} - {df['intensity'].max():.2f}")
        
        # Apply normalization
        print(f"   🔧 Applying {light_source} normalization...")
        normalized_intensities = normalize_spectrum(df['wavelength'].values, df['intensity'].values, light_source)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'wavelength': df['wavelength'],
            'intensity': normalized_intensities
        })
        
        print(f"   ✅ Normalization complete")
        print(f"   📈 New intensity range: {output_df['intensity'].min():.2f} - {output_df['intensity'].max():.2f}")
        
        return output_df
        
    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main quick fix function"""
    print("🔧 QUICK FIX - CONVERSION AND PATH ISSUES")
    print("=" * 50)
    
    # Step 1: Find raw_txt files
    raw_dir, txt_files = find_raw_txt_files()
    if not raw_dir:
        print("\n💡 TROUBLESHOOTING:")
        print("1. Make sure you have .txt files in a 'raw_txt' directory")
        print("2. Or run the analytical workflow to create them")
        return False
    
    # Step 2: Create output directory
    output_dir = create_output_directory()
    
    # Step 3: Process files by light source
    converted_files = {}
    
    for light_source in ['B', 'L', 'U']:
        # Find files for this light source
        source_files = [f for f in txt_files if light_source.upper() in f.upper()]
        
        if not source_files:
            print(f"\n⚠️ No files found for light source {light_source}")
            continue
        
        if len(source_files) > 1:
            print(f"\n⚠️ Multiple files found for {light_source}: {source_files}")
            print(f"   Using first file: {source_files[0]}")
        
        selected_file = source_files[0]
        input_path = os.path.join(raw_dir, selected_file)
        
        # Convert the file
        converted_df = convert_file(input_path, light_source)
        
        if converted_df is not None:
            # Save as unkgem*.csv
            output_filename = f"unkgem{light_source}.csv"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save without header and index to match expected format
            converted_df.to_csv(output_path, header=False, index=False)
            
            converted_files[light_source] = {
                'input': selected_file,
                'output': output_path,
                'rows': len(converted_df)
            }
            
            print(f"✅ Created {output_filename} ({len(converted_df)} data points)")
        else:
            print(f"❌ Failed to convert {selected_file}")
    
    # Step 4: Summary and next steps
    print(f"\n📋 QUICK FIX SUMMARY")
    print("=" * 40)
    
    if converted_files:
        print("✅ Successfully converted files:")
        for light_source, info in converted_files.items():
            print(f"   {light_source}: {info['input']} → {info['output']}")
            print(f"      ({info['rows']} data points)")
        
        print(f"\n📁 Output directory: {os.path.abspath(output_dir)}")
        
        # Check if gemini1.py can find these files
        print(f"\n🔍 CHECKING GEMINI1.PY COMPATIBILITY:")
        
        expected_paths = [
            'data/unknown/unkgemB.csv',
            'data/unknown/unkgemL.csv', 
            'data/unknown/unkgemU.csv'
        ]
        
        all_found = True
        for expected_path in expected_paths:
            if os.path.exists(expected_path):
                print(f"   ✅ {expected_path}")
            else:
                print(f"   ❌ {expected_path}")
                all_found = False
        
        if all_found:
            print(f"\n🎉 ALL FILES IN CORRECT LOCATION!")
            print(f"   You can now run gemini1.py directly")
        else:
            print(f"\n⚠️ Files created but not in expected location for gemini1.py")
            print(f"   Consider copying files to data/unknown/ or updating gemini1.py paths")
        
        # Try to run gemini1.py
        print(f"\n🧮 ATTEMPTING TO RUN NUMERICAL ANALYSIS:")
        try:
            import subprocess
            import sys
            
            # Look for gemini1.py
            gemini_paths = [
                'src/numerical_analysis/gemini1.py',
                '../numerical_analysis/gemini1.py',
                'gemini1.py'
            ]
            
            gemini_script = None
            for path in gemini_paths:
                if os.path.exists(path):
                    gemini_script = path
                    break
            
            if gemini_script:
                print(f"   Found gemini1.py: {gemini_script}")
                result = subprocess.run([sys.executable, gemini_script], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print("   ✅ Numerical analysis completed successfully!")
                else:
                    print("   ⚠️ Numerical analysis had issues:")
                    if result.stderr:
                        print(f"      Error: {result.stderr[:200]}...")
                    if result.stdout:
                        print(f"      Output: {result.stdout[:200]}...")
            else:
                print("   ❌ gemini1.py not found")
                
        except Exception as e:
            print(f"   ❌ Error running numerical analysis: {e}")
        
        return True
    else:
        print("❌ No files were successfully converted")
        return False

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n💡 TROUBLESHOOTING STEPS:")
        print("1. Check that raw_txt directory exists with .txt files")
        print("2. Verify file formats (space/comma/tab separated)")
        print("3. Ensure numpy is installed: pip install numpy")
        print("4. Check file permissions for writing output")
    
    input("\nPress Enter to exit...")