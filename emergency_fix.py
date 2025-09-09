#!/usr/bin/env python3
"""
emergency_fix.py - Quick fix for immediate issues
Run this RIGHT NOW to fix your current problem
"""

import os
import pandas as pd
import numpy as np

def emergency_conversion():
    """Quick fix to convert your current raw_txt files"""
    
    print("🚨 EMERGENCY FIX - Converting your current files")
    print("=" * 50)
    
    # Find your raw_txt directory
    raw_dir = None
    for possible in ['data/raw_txt', 'raw_txt', '../raw_txt']:
        if os.path.exists(possible):
            raw_dir = possible
            break
    
    if not raw_dir:
        print("❌ No raw_txt directory found")
        return False
    
    print(f"✅ Found raw_txt: {raw_dir}")
    
    # Create data/unknown directory
    os.makedirs('data/unknown', exist_ok=True)
    
    # Convert each light source
    for light_source in ['B', 'L', 'U']:
        # Find the file
        files = [f for f in os.listdir(raw_dir) if light_source in f.upper() and f.endswith('.txt')]
        
        if not files:
            print(f"⚠️ No {light_source} file found")
            continue
        
        input_file = os.path.join(raw_dir, files[0])
        output_file = f'data/unknown/unkgem{light_source}.csv'
        
        print(f"\n🔄 Converting {light_source}: {files[0]}")
        
        try:
            # Read the file
            df = pd.read_csv(input_file, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
            
            # Apply normalization WITH NUMPY
            wavelengths = np.array(df['wavelength'])
            intensities = np.array(df['intensity'])
            
            if light_source == 'B':
                # B: 650nm → 50000
                idx = np.argmin(np.abs(wavelengths - 650))
                if intensities[idx] != 0:
                    scale = 50000 / intensities[idx]
                    normalized = intensities * scale
                else:
                    normalized = intensities
                    
            elif light_source == 'L':
                # L: 450nm → 50000  
                idx = np.argmin(np.abs(wavelengths - 450))
                if intensities[idx] != 0:
                    scale = 50000 / intensities[idx]
                    normalized = intensities * scale
                else:
                    normalized = intensities
                    
            elif light_source == 'U':
                # U: 811nm window → 15000
                mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
                window = intensities[mask]
                if len(window) > 0 and window.max() > 0:
                    scale = 15000 / window.max()
                    normalized = intensities * scale
                else:
                    normalized = intensities
            
            # Save as CSV
            output_df = pd.DataFrame({
                'wavelength': wavelengths,
                'intensity': normalized
            })
            
            output_df.to_csv(output_file, header=False, index=False)
            
            print(f"   ✅ Created {output_file}")
            print(f"   📊 {len(output_df)} data points")
            print(f"   📈 Intensity range: {normalized.min():.2f} - {normalized.max():.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Check if files were created
    created_files = []
    for ls in ['B', 'L', 'U']:
        path = f'data/unknown/unkgem{ls}.csv'
        if os.path.exists(path):
            created_files.append(ls)
    
    print(f"\n📋 EMERGENCY FIX RESULTS:")
    print(f"   Created files: {created_files}")
    
    if created_files:
        print("✅ EMERGENCY FIX SUCCESSFUL!")
        print("Now you can run gemini1.py")
        
        # Try to run gemini1.py immediately
        try:
            print("\n🚀 Running numerical analysis...")
            import subprocess
            import sys
            
            # Find gemini1.py
            gemini_path = None
            for path in ['src/numerical_analysis/gemini1.py', 'gemini1.py']:
                if os.path.exists(path):
                    gemini_path = path
                    break
            
            if gemini_path:
                result = subprocess.run([sys.executable, gemini_path], 
                                      capture_output=True, text=True)
                
                if result.stdout:
                    print("📊 ANALYSIS OUTPUT:")
                    print(result.stdout[-1000:])  # Last 1000 chars
                
                if result.stderr:
                    print("⚠️ WARNINGS:")
                    print(result.stderr[-500:])  # Last 500 chars
            else:
                print("❌ gemini1.py not found")
                
        except Exception as e:
            print(f"❌ Error running analysis: {e}")
        
        return True
    else:
        print("❌ EMERGENCY FIX FAILED")
        return False

if __name__ == "__main__":
    emergency_conversion()
    input("\nPress Enter to exit...")