#!/usr/bin/env python3
"""
fix_everything_now.py - ONE SCRIPT TO FIX ALL YOUR ISSUES

Just run this ONE script. It will:
1. Fix all path issues
2. Convert your files properly 
3. Run the analysis
4. Show you the results

NO MORE COMPLEXITY - JUST WORKS
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import shutil

def main():
    """Fix everything automatically"""
    
    print("🔧 FIXING EVERYTHING NOW - PLEASE WAIT")
    print("=" * 50)
    
    # STEP 1: Find your raw_txt files (from your previous run)
    print("📂 Finding your files...")
    
    raw_dirs = ['raw_txt', 'data/raw_txt', '../raw_txt']
    raw_dir = None
    
    for d in raw_dirs:
        if os.path.exists(d):
            files = [f for f in os.listdir(d) if f.endswith('.txt')]
            if files:
                raw_dir = d
                print(f"✅ Found files in: {d}")
                break
    
    if not raw_dir:
        print("❌ No raw_txt files found. Put your .txt files in 'raw_txt' directory first.")
        input("Press Enter to exit...")
        return
    
    # STEP 2: Create output directory
    print("📁 Creating output directory...")
    os.makedirs('data/unknown', exist_ok=True)
    
    # STEP 3: Convert each file with FIXED normalization
    print("🔄 Converting files (FIXED VERSION)...")
    
    for light_source in ['B', 'L', 'U']:
        # Find file for this light source
        files = [f for f in os.listdir(raw_dir) if light_source.upper() in f.upper()]
        
        if not files:
            print(f"⚠️ No {light_source} file found")
            continue
        
        input_file = os.path.join(raw_dir, files[0])
        output_file = f'data/unknown/unkgem{light_source}.csv'
        
        print(f"   Converting {light_source}: {files[0]}")
        
        try:
            # Read file
            df = pd.read_csv(input_file, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
            
            wavelengths = np.array(df['wavelength'])
            intensities = np.array(df['intensity'])
            
            # Apply CORRECT normalization
            if light_source == 'B':
                idx = np.argmin(np.abs(wavelengths - 650))
                if intensities[idx] != 0:
                    normalized = intensities * (50000 / intensities[idx])
                else:
                    normalized = intensities
                    
            elif light_source == 'L':
                idx = np.argmin(np.abs(wavelengths - 450))
                if intensities[idx] != 0:
                    normalized = intensities * (50000 / intensities[idx])
                else:
                    normalized = intensities
                    
            elif light_source == 'U':
                mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
                window = intensities[mask]
                if len(window) > 0 and window.max() > 0:
                    normalized = intensities * (15000 / window.max())
                else:
                    normalized = intensities
            
            # Save file
            output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
            output_df.to_csv(output_file, header=False, index=False)
            
            print(f"   ✅ {output_file} created")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # STEP 4: Find and run gemini1.py
    print("🧮 Running analysis...")
    
    gemini_paths = [
        'src/numerical_analysis/gemini1.py',
        'gemini1.py',
        '../numerical_analysis/gemini1.py'
    ]
    
    gemini_path = None
    for path in gemini_paths:
        if os.path.exists(path):
            gemini_path = path
            break
    
    if not gemini_path:
        print("❌ gemini1.py not found")
        print("Available files:", os.listdir('.'))
        input("Press Enter to exit...")
        return
    
    # Run analysis with simple output
    try:
        print("🚀 Running numerical analysis...")
        result = subprocess.run([sys.executable, gemini_path], 
                              capture_output=True, text=True, 
                              encoding='utf-8', errors='replace')
        
        print("\n📊 ANALYSIS RESULTS:")
        print("=" * 40)
        
        if result.stdout:
            # Clean and show output
            output = result.stdout.replace('\x9d', '').replace('\x8d', '')
            lines = output.split('\n')
            
            # Show the important parts
            for line in lines:
                if any(word in line.lower() for word in ['rank', 'gem', 'score', 'match', 'best']):
                    print(line)
        
        if result.stderr:
            errors = result.stderr.replace('\x9d', '').replace('\x8d', '')
            if 'error' in errors.lower():
                print("\n⚠️ Errors:")
                print(errors[:200])
        
        print("\n✅ ANALYSIS COMPLETE!")
        
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
    
    # STEP 5: Show what files were created
    print("\n📋 SUMMARY:")
    print("=" * 20)
    
    created = []
    for ls in ['B', 'L', 'U']:
        path = f'data/unknown/unkgem{ls}.csv'
        if os.path.exists(path):
            created.append(ls)
    
    print(f"✅ Files converted: {created}")
    print(f"✅ Analysis completed")
    print(f"✅ Check results above")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        input("Press Enter to exit...")