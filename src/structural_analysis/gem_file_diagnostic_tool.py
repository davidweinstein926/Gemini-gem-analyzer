#!/usr/bin/env python3
"""
Gem File Diagnostic Tool
Analyzes why normalization might be failing for specific gems
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

def load_spectrum_file(filepath):
    """Load spectrum file with error handling"""
    try:
        print(f"\n📁 Loading: {filepath}")
        
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
            wavelengths = data.iloc[:, 0].values
            intensities = data.iloc[:, 1].values
            print(f"  Format: CSV file")
        else:
            data = np.loadtxt(filepath, delimiter='\t')
            wavelengths = data[:, 0]
            intensities = data[:, 1]
            print(f"  Format: Tab-delimited text file")
        
        print(f"  Data points: {len(wavelengths):,}")
        print(f"  Wavelength range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
        print(f"  Intensity range: {intensities.min():.2f} - {intensities.max():.2f}")
        
        return wavelengths, intensities, None
        
    except Exception as e:
        print(f"  ❌ ERROR loading file: {e}")
        return None, None, str(e)

def check_normalization_requirements(wavelengths, intensities, gem_name):
    """Check if file meets B/H normalization requirements"""
    print(f"\n🔍 Checking normalization requirements for {gem_name}:")
    
    issues = []
    
    # Check 1: Does file contain 650nm?
    distances_to_650 = np.abs(wavelengths - 650.0)
    closest_idx = np.argmin(distances_to_650)
    closest_wavelength = wavelengths[closest_idx]
    closest_distance = distances_to_650[closest_idx]
    
    print(f"  650nm reference check:")
    print(f"    Closest wavelength: {closest_wavelength:.2f}nm")
    print(f"    Distance from 650nm: {closest_distance:.2f}nm")
    
    if closest_distance > 10.0:
        issues.append(f"No wavelength close to 650nm (closest: {closest_wavelength:.2f}nm)")
        print(f"    ❌ ISSUE: Too far from 650nm")
    else:
        print(f"    ✅ OK: Within acceptable range")
    
    # Check 2: Intensity at 650nm
    intensity_at_650 = intensities[closest_idx]
    print(f"  Intensity at ~650nm:")
    print(f"    Value: {intensity_at_650:.3f}")
    
    if intensity_at_650 <= 0:
        issues.append(f"Invalid intensity at 650nm: {intensity_at_650}")
        print(f"    ❌ ISSUE: Intensity is zero or negative")
    else:
        print(f"    ✅ OK: Positive intensity")
    
    # Check 3: Baseline region (300-325nm)
    baseline_mask = (wavelengths >= 300) & (wavelengths <= 325)
    baseline_points = np.sum(baseline_mask)
    
    print(f"  Baseline region (300-325nm):")
    print(f"    Data points in region: {baseline_points}")
    
    if baseline_points == 0:
        issues.append("No data points in baseline region (300-325nm)")
        print(f"    ❌ ISSUE: No baseline data")
    else:
        baseline_intensities = intensities[baseline_mask]
        baseline_std = np.std(baseline_intensities)
        print(f"    Baseline noise (std): {baseline_std:.4f}")
        print(f"    ✅ OK: Baseline region exists")
    
    # Check 4: Data validity
    invalid_wavelengths = np.sum(~np.isfinite(wavelengths))
    invalid_intensities = np.sum(~np.isfinite(intensities))
    
    print(f"  Data validity:")
    print(f"    Invalid wavelengths: {invalid_wavelengths}")
    print(f"    Invalid intensities: {invalid_intensities}")
    
    if invalid_wavelengths > 0 or invalid_intensities > 0:
        issues.append(f"Invalid data points: {invalid_wavelengths} wavelengths, {invalid_intensities} intensities")
        print(f"    ❌ ISSUE: Contains invalid data")
    else:
        print(f"    ✅ OK: All data is valid")
    
    # Check 5: Intensity distribution
    zero_intensities = np.sum(intensities <= 0)
    negative_intensities = np.sum(intensities < 0)
    
    print(f"  Intensity distribution:")
    print(f"    Zero or negative intensities: {zero_intensities}")
    print(f"    Negative intensities: {negative_intensities}")
    print(f"    Mean intensity: {np.mean(intensities):.2f}")
    print(f"    Median intensity: {np.median(intensities):.2f}")
    
    if zero_intensities > len(intensities) * 0.1:  # More than 10% zero/negative
        issues.append(f"Too many zero/negative intensities: {zero_intensities}")
        print(f"    ⚠️  WARNING: Many zero/negative values")
    else:
        print(f"    ✅ OK: Mostly positive intensities")
    
    return issues

def simulate_normalization(wavelengths, intensities, gem_name):
    """Simulate the normalization process to see where it fails"""
    print(f"\n🧪 Simulating normalization for {gem_name}:")
    
    try:
        # Step 1: Find 650nm reference
        idx_650 = np.argmin(np.abs(wavelengths - 650.0))
        intensity_650 = intensities[idx_650]
        wavelength_650 = wavelengths[idx_650]
        
        print(f"  Step 1 - Find 650nm reference:")
        print(f"    Reference wavelength: {wavelength_650:.2f}nm")
        print(f"    Reference intensity: {intensity_650:.3f}")
        
        if intensity_650 <= 0:
            print(f"    ❌ FAILED: Invalid reference intensity")
            return False
        
        # Step 2: Normalize 650nm to 50000
        step1_normalized = (intensities / intensity_650) * 50000
        print(f"  Step 2 - Normalize to 50000:")
        print(f"    Normalization factor: {50000/intensity_650:.3f}")
        print(f"    Range after step 1: {np.min(step1_normalized):.1f} - {np.max(step1_normalized):.1f}")
        
        # Step 3: Scale to 0-100
        min_val = np.min(step1_normalized)
        max_val = np.max(step1_normalized)
        range_val = max_val - min_val
        
        print(f"  Step 3 - Scale to 0-100:")
        print(f"    Min value: {min_val:.1f}")
        print(f"    Max value: {max_val:.1f}")
        print(f"    Range: {range_val:.1f}")
        
        if range_val > 0:
            normalized = ((step1_normalized - min_val) / range_val) * 100.0
            print(f"    Final range: {np.min(normalized):.2f} - {np.max(normalized):.2f}")
            print(f"    ✅ SUCCESS: Normalization completed")
            return True
        else:
            print(f"    ❌ FAILED: Zero range (all intensities identical)")
            return False
            
    except Exception as e:
        print(f"    ❌ FAILED: Exception occurred: {e}")
        return False

def compare_gems(gem1_path, gem2_path):
    """Compare two gem files to find differences"""
    print(f"\n🔄 COMPARISON ANALYSIS")
    print(f"=" * 50)
    
    # Load both files
    wl1, int1, err1 = load_spectrum_file(gem1_path)
    wl2, int2, err2 = load_spectrum_file(gem2_path)
    
    gem1_name = Path(gem1_path).stem
    gem2_name = Path(gem2_path).stem
    
    if wl1 is None or wl2 is None:
        print("❌ Cannot compare - one or both files failed to load")
        return
    
    # Check normalization requirements
    issues1 = check_normalization_requirements(wl1, int1, gem1_name)
    issues2 = check_normalization_requirements(wl2, int2, gem2_name)
    
    # Simulate normalization
    norm1_success = simulate_normalization(wl1, int1, gem1_name)
    norm2_success = simulate_normalization(wl2, int2, gem2_name)
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"  {gem1_name}: {len(issues1)} issues, normalization {'✅ SUCCESS' if norm1_success else '❌ FAILED'}")
    print(f"  {gem2_name}: {len(issues2)} issues, normalization {'✅ SUCCESS' if norm2_success else '❌ FAILED'}")
    
    if issues1:
        print(f"\n❌ Issues with {gem1_name}:")
        for issue in issues1:
            print(f"    • {issue}")
    
    if issues2:
        print(f"\n❌ Issues with {gem2_name}:")
        for issue in issues2:
            print(f"    • {issue}")
    
    if not issues1 and not issues2:
        print(f"\n✅ Both files appear suitable for normalization")
        if norm1_success != norm2_success:
            print(f"⚠️  WARNING: Normalization success differs despite no obvious issues")
    
    # File format comparison
    print(f"\n📋 FILE FORMAT COMPARISON:")
    print(f"  {gem1_name}: {len(wl1):,} points, range {wl1.min():.1f}-{wl1.max():.1f}nm")
    print(f"  {gem2_name}: {len(wl2):,} points, range {wl2.min():.1f}-{wl2.max():.1f}nm")

def main():
    """Main diagnostic function"""
    print("🔬 GEM FILE DIAGNOSTIC TOOL")
    print("=" * 50)
    print("This tool will help diagnose normalization issues")
    
    # Default paths (adjust as needed)
    data_dir = r"C:\users\david\onedrive\desktop\gemini_gemological_analysis\data\raw"
    
    gem1_file = "51bc1.txt"  # Replace with actual filename
    gem2_file = "92bc1.txt"  # Replace with actual filename
    
    gem1_path = os.path.join(data_dir, gem1_file)
    gem2_path = os.path.join(data_dir, gem2_file)
    
    # Check if files exist
    if not os.path.exists(gem1_path):
        print(f"❌ File not found: {gem1_path}")
        print("Please update the gem1_file variable with the correct filename")
        return
    
    if not os.path.exists(gem2_path):
        print(f"❌ File not found: {gem2_path}")
        print("Please update the gem2_file variable with the correct filename")
        return
    
    # Run comparison
    compare_gems(gem1_path, gem2_path)

if __name__ == "__main__":
    main()