#!/usr/bin/env python3
"""
Gemini Database Normalization Analyzer
Analyzes gemini_db_long_*.csv files to understand normalization schemes
"""

import pandas as pd
import numpy as np
import os

def analyze_database_file(filepath, light_source):
    """Analyze a single database file"""
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {light_source} LIGHT DATABASE")
    print(f"File: {filepath}")
    print(f"{'='*60}")
    
    # Load the file
    df = pd.read_csv(filepath)
    
    print(f"📊 BASIC STATISTICS:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    # Check if it has the expected columns
    if 'full_name' not in df.columns or 'wavelength' not in df.columns or 'intensity' not in df.columns:
        print(f"❌ Unexpected column structure!")
        return None
    
    # Get unique gems
    unique_gems = df['full_name'].unique()
    print(f"   Unique spectra: {len(unique_gems)}")
    
    # Intensity analysis
    print(f"\n📈 INTENSITY ANALYSIS:")
    print(f"   Min intensity: {df['intensity'].min():.3f}")
    print(f"   Max intensity: {df['intensity'].max():.3f}")
    print(f"   Mean intensity: {df['intensity'].mean():.3f}")
    print(f"   Std intensity: {df['intensity'].std():.3f}")
    
    # Check for common normalization patterns
    max_intensity = df['intensity'].max()
    min_intensity = df['intensity'].min()
    
    print(f"\n🔍 NORMALIZATION PATTERN DETECTION:")
    if abs(max_intensity - 100.0) < 0.1 and abs(min_intensity - 0.0) < 0.1:
        print(f"   ✅ DETECTED: 0-100 scaling")
    elif abs(max_intensity - 50000) < 100:
        print(f"   ✅ DETECTED: ~50000 max normalization")
    elif abs(max_intensity - 15000) < 100:
        print(f"   ✅ DETECTED: ~15000 max normalization")
    else:
        print(f"   ❓ UNKNOWN: Custom normalization range {min_intensity:.1f} - {max_intensity:.1f}")
    
    # Wavelength analysis
    print(f"\n📏 WAVELENGTH ANALYSIS:")
    print(f"   Min wavelength: {df['wavelength'].min():.1f} nm")
    print(f"   Max wavelength: {df['wavelength'].max():.1f} nm")
    print(f"   Wavelength span: {df['wavelength'].max() - df['wavelength'].min():.1f} nm")
    
    # Look for gem 140 specifically
    gem_140_data = df[df['full_name'].str.contains('140', na=False)]
    if not gem_140_data.empty:
        print(f"\n🎯 GEM 140 ANALYSIS:")
        gem_140_spectra = gem_140_data['full_name'].unique()
        print(f"   Found spectra: {list(gem_140_spectra)}")
        
        for spectrum in gem_140_spectra[:3]:  # Show first 3
            spectrum_data = gem_140_data[gem_140_data['full_name'] == spectrum]
            if not spectrum_data.empty:
                intensity_range = f"{spectrum_data['intensity'].min():.3f} - {spectrum_data['intensity'].max():.3f}"
                points = len(spectrum_data)
                print(f"   {spectrum}: {points} points, intensity {intensity_range}")
    
    # Sample some spectra to understand per-spectrum normalization
    print(f"\n📋 SAMPLE SPECTRA ANALYSIS:")
    sample_spectra = unique_gems[:5]  # First 5 spectra
    
    for spectrum in sample_spectra:
        spectrum_data = df[df['full_name'] == spectrum]
        intensity_min = spectrum_data['intensity'].min()
        intensity_max = spectrum_data['intensity'].max()
        points = len(spectrum_data)
        print(f"   {spectrum}: {points} pts, range {intensity_min:.3f}-{intensity_max:.3f}")
    
    return {
        'light_source': light_source,
        'total_spectra': len(unique_gems),
        'total_points': len(df),
        'intensity_min': df['intensity'].min(),
        'intensity_max': df['intensity'].max(),
        'intensity_mean': df['intensity'].mean(),
        'wavelength_min': df['wavelength'].min(),
        'wavelength_max': df['wavelength'].max(),
        'gem_140_found': len(gem_140_data) > 0,
        'gem_140_spectra': gem_140_data['full_name'].unique().tolist() if len(gem_140_data) > 0 else []
    }

def compare_normalization_across_lights(results):
    """Compare normalization patterns across light sources"""
    print(f"\n{'='*60}")
    print(f"CROSS-LIGHT NORMALIZATION COMPARISON")
    print(f"{'='*60}")
    
    print(f"{'Light':<8} {'Min':<10} {'Max':<10} {'Mean':<10} {'Pattern'}")
    print(f"-" * 60)
    
    for result in results:
        if result:
            light = result['light_source']
            min_int = result['intensity_min']
            max_int = result['intensity_max']
            mean_int = result['intensity_mean']
            
            # Detect pattern
            if abs(max_int - 100.0) < 0.1 and abs(min_int - 0.0) < 0.1:
                pattern = "0-100 scaled"
            elif abs(max_int - 50000) < 100:
                pattern = "~50000 max"
            elif abs(max_int - 15000) < 100:
                pattern = "~15000 max"
            else:
                pattern = f"Custom ({min_int:.0f}-{max_int:.0f})"
            
            print(f"{light:<8} {min_int:<10.3f} {max_int:<10.3f} {mean_int:<10.3f} {pattern}")
    
    # Check for inconsistencies
    print(f"\n🔍 NORMALIZATION CONSISTENCY CHECK:")
    max_intensities = [r['intensity_max'] for r in results if r]
    min_intensities = [r['intensity_min'] for r in results if r]
    
    if len(set([round(x) for x in max_intensities])) == 1:
        print(f"   ✅ All light sources have same max intensity: {max_intensities[0]:.1f}")
    else:
        print(f"   ❌ Different max intensities detected: {max_intensities}")
        print(f"      This indicates different normalization schemes per light source!")
    
    if len(set([round(x) for x in min_intensities])) == 1:
        print(f"   ✅ All light sources have same min intensity: {min_intensities[0]:.1f}")
    else:
        print(f"   ❌ Different min intensities detected: {min_intensities}")

def analyze_gem_140_specifically(results):
    """Detailed analysis of gem 140 across all light sources"""
    print(f"\n{'='*60}")
    print(f"DETAILED GEM 140 ANALYSIS")
    print(f"{'='*60}")
    
    for result in results:
        if result and result['gem_140_found']:
            light = result['light_source']
            spectra = result['gem_140_spectra']
            print(f"\n{light} Light - Gem 140 spectra found:")
            for spectrum in spectra:
                print(f"   {spectrum}")
        else:
            print(f"\n{result['light_source'] if result else 'Unknown'} Light - No gem 140 data found")

def main():
    """Main analysis function"""
    print("GEMINI DATABASE NORMALIZATION ANALYZER")
    print("=" * 60)
    
    # Database file paths
    database_files = [
        ("database/reference_spectra/gemini_db_long_B.csv", "B"),
        ("database/reference_spectra/gemini_db_long_L.csv", "L"), 
        ("database/reference_spectra/gemini_db_long_U.csv", "U")
    ]
    
    results = []
    
    # Analyze each database file
    for filepath, light_source in database_files:
        result = analyze_database_file(filepath, light_source)
        results.append(result)
    
    # Compare normalization across light sources
    compare_normalization_across_lights(results)
    
    # Detailed gem 140 analysis
    analyze_gem_140_specifically(results)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"This analysis will help identify the normalization mismatch")
    print(f"between your unkgem*.csv files and the reference database.")

if __name__ == "__main__":
    main()