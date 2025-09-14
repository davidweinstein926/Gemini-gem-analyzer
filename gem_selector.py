#!/usr/bin/env python3
"""
Standalone Gem Selector with CORRECTED txt_to_unkgem functionality
CORRECTED: Laser normalization is MAX → 50000 (not 450nm → 50000)
Includes: Database-matching normalization + 0-100 scaling

Run this when main.py asks for gem selection - it will:
1. Show data/raw files organized by gem
2. Let you select which gem you want
3. Convert with CORRECTED normalization + 0-100 scaling
4. Save to data/unknown/numerical/unkgem*.csv
5. Ready for gemini1.py analysis

Usage: python gem_selector.py
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def normalize_spectrum(wavelengths, intensities, light_source):
    """Apply CORRECTED normalization - matches main.py database normalization"""
    if light_source == 'B':
        # B Light: 650nm → 50000
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
        # L Light: Maximum → 50000 (CORRECTED from 450nm)
        max_intensity = intensities.max()
        if max_intensity != 0:
            normalized = intensities * (50000 / max_intensity)
            return normalized
        else:
            print(f"⚠️ Warning: Zero max intensity for {light_source}")
            return intensities
            
    elif light_source == 'U':
        # U Light: 811nm window max → 15000 (matches database method)
        mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
        window = intensities[mask]
        if len(window) > 0 and window.max() > 0:
            normalized = intensities * (15000 / window.max())
            return normalized
        else:
            print(f"⚠️ Warning: No valid data in 810.5-811.5nm range for {light_source}")
            return intensities
    else:
        print(f"⚠️ Unknown light source: {light_source}")
        return intensities

def apply_0_100_scaling(wavelengths, intensities):
    """Apply 0-100 scaling for analysis and visualization"""
    min_val, max_val = intensities.min(), intensities.max()
    if max_val != min_val:
        scaled = (intensities - min_val) * 100 / (max_val - min_val)
        return scaled
    else:
        return intensities

def load_and_convert_spectrum(filepath, light_source):
    """Load .txt file and convert with CORRECTED normalization + 0-100 scaling"""
    try:
        # Read the .txt file (assuming space/tab separated)
        df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
        
        # Step 1: Apply database-matching normalization
        normalized_intensities = normalize_spectrum(df['wavelength'].values, df['intensity'].values, light_source)
        
        # Step 2: Apply 0-100 scaling (CRITICAL for matching)
        scaled_intensities = apply_0_100_scaling(df['wavelength'].values, normalized_intensities)
        
        # Create output dataframe
        output_df = pd.DataFrame({
            'wavelength': df['wavelength'],
            'intensity': scaled_intensities
        })
        
        print(f"      Normalization: {light_source} light, range {scaled_intensities.min():.3f}-{scaled_intensities.max():.3f}")
        
        return output_df
        
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")
        return None

def show_data_raw_files():
    """Show available files in data/raw"""
    print("🎯 GEM SELECTOR - BYPASS INPUT VALIDATION BUG")
    print("=" * 55)
    
    # Check data/raw directory
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        print(f"❌ {raw_dir} directory not found!")
        return None, None
    
    # Get all .txt files
    txt_files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
    if not txt_files:
        print(f"❌ No .txt files found in {raw_dir}")
        return None, None
    
    print(f"📁 Found {len(txt_files)} spectrum files in {raw_dir}:")
    print()
    
    # Show numbered list
    print("📋 ALL SPECTRUM FILES:")
    print("-" * 50)
    for i, file in enumerate(sorted(txt_files), 1):
        file_path = os.path.join(raw_dir, file)
        size_kb = os.path.getsize(file_path) / 1024
        
        # Detect light source and gem ID
        light_source = "?"
        gem_id = "?"
        
        file_upper = file.upper()
        for ls in ['B', 'L', 'U']:
            if ls in file_upper:
                light_source = ls
                idx = file_upper.find(ls)
                gem_id = file[:idx]
                break
        
        print(f"   {i:2}. {file:<25} | Gem: {gem_id:<10} | Light: {light_source} | {size_kb:.1f} KB")
    
    # Group by gem
    gems = {}
    for file in txt_files:
        file_upper = file.upper()
        for ls in ['B', 'L', 'U']:
            if ls in file_upper:
                idx = file_upper.find(ls)
                gem_id = file[:idx]
                if gem_id not in gems:
                    gems[gem_id] = {'B': [], 'L': [], 'U': []}
                gems[gem_id][ls].append(file)
                break
    
    # Show organized by gem
    print(f"\n📊 ORGANIZED BY GEM:")
    print("-" * 50)
    
    complete_gems = []
    for gem_id in sorted(gems.keys()):
        gem_data = gems[gem_id]
        available = [ls for ls in ['B', 'L', 'U'] if gem_data[ls]]
        
        if len(available) == 3:
            status = "✅ COMPLETE"
            complete_gems.append(gem_id)
        else:
            status = "🟡 PARTIAL"
        
        print(f"\n{status} Gem {gem_id}:")
        for light in ['B', 'L', 'U']:
            if gem_data[light]:
                files_str = ', '.join(gem_data[light])
                print(f"   {light}: {files_str}")
            else:
                print(f"   {light}: ❌ Missing")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Complete gems (B+L+U): {len(complete_gems)} → {complete_gems}")
    
    return gems, complete_gems

def select_gem_interactively(gems, complete_gems):
    """Interactive gem selection"""
    print(f"\n🎯 SELECTION OPTIONS:")
    print("1. Select complete gem by ID")
    print("2. Quick selection for gem 58")
    print("3. Show file details and exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        if not complete_gems:
            print("❌ No complete gems available")
            return None
        
        print(f"\n📋 COMPLETE GEMS:")
        for i, gem_id in enumerate(complete_gems, 1):
            print(f"   {i}. {gem_id}")
        
        gem_choice = input(f"\nEnter gem ID: ").strip()
        
        if gem_choice in gems:
            gem_data = gems[gem_choice]
            available = [ls for ls in ['B', 'L', 'U'] if gem_data[ls]]
            
            if len(available) == 3:
                selected_files = {}
                for light in ['B', 'L', 'U']:
                    selected_files[light] = gem_data[light][0]
                
                print(f"\n✅ SELECTED GEM {gem_choice}:")
                for light, filename in selected_files.items():
                    print(f"   {light}: {filename}")
                
                return selected_files, gem_choice
            else:
                print(f"❌ Gem {gem_choice} incomplete. Available: {available}")
                return None
        else:
            print(f"❌ Gem {gem_choice} not found")
            return None
    
    elif choice == "2":
        # Quick gem 58 selection
        if "58" in gems:
            gem_data = gems["58"]
            available = [ls for ls in ['B', 'L', 'U'] if gem_data[ls]]
            
            if len(available) == 3:
                selected_files = {}
                for light in ['B', 'L', 'U']:
                    selected_files[light] = gem_data[light][0]
                
                print(f"\n⚡ QUICK SELECTED GEM 58:")
                for light, filename in selected_files.items():
                    print(f"   {light}: {filename}")
                
                return selected_files, "58"
            else:
                print(f"❌ Gem 58 incomplete. Available: {available}")
                return None
        else:
            print("❌ Gem 58 not found")
            return None
    
    elif choice == "3":
        print("\n📋 File details shown above. Run again to make selection.")
        return None
    
    else:
        print("❌ Invalid choice")
        return None

def convert_to_unkgem_format(selected_files, gem_id):
    """Convert selected files to unkgem*.csv format"""
    print(f"\n🔄 CONVERTING TO UNKGEM FORMAT")
    print("=" * 40)
    
    # Create output directories
    output_dir = "data/unknown/numerical"
    standard_dir = "data/unknown"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(standard_dir, exist_ok=True)
    
    raw_dir = "data/raw"
    converted_count = 0
    
    for light, filename in selected_files.items():
        input_path = os.path.join(raw_dir, filename)
        output_path_numerical = os.path.join(output_dir, f"unkgem{light}.csv")
        output_path_standard = os.path.join(standard_dir, f"unkgem{light}.csv")
        
        print(f"🔄 Processing {light}: {filename}")
        
        # Convert using CORRECTED txt_to_unkgem functionality
        converted_df = load_and_convert_spectrum(input_path, light)
        
        if converted_df is not None:
            # Save to both locations (no header, no index) for maximum compatibility
            converted_df.to_csv(output_path_numerical, header=False, index=False)
            converted_df.to_csv(output_path_standard, header=False, index=False)
            
            print(f"   ✅ unkgem{light}.csv ({len(converted_df)} data points)")
            converted_count += 1
        else:
            print(f"   ❌ Failed to convert {filename}")
            return False
    
    if converted_count == 3:
        print(f"\n📋 CONVERSION SUMMARY")
        print("=" * 40)
        print(f"Gem ID: {gem_id}")
        print(f"Applied CORRECTED normalization:")
        print(f"   B Light: 650nm → 50000, then 0-100 scaled")
        print(f"   L Light: MAX → 50000, then 0-100 scaled") 
        print(f"   U Light: 811nm window → 15000, then 0-100 scaled")
        print(f"Files created:")
        print(f"   📁 {output_dir}/unkgemB.csv, unkgemL.csv, unkgemU.csv")
        print(f"   📁 {standard_dir}/unkgemB.csv, unkgemL.csv, unkgemU.csv")
        
        print(f"\n🎉 SUCCESS: Files ready for numerical analysis!")
        print(f"🔬 Now run: python src/numerical_analysis/gemini1.py")
        print(f"🔬 Or run: python main.py (option 2)")
        
        return True
    else:
        print(f"\n❌ FAILED: Could not convert all files")
        return False

def clean_previous_files():
    """Clean previous analysis files"""
    print("🧹 CLEANING PREVIOUS ANALYSIS FILES...")
    
    files_to_clean = [
        "data/unknown/unkgemB.csv",
        "data/unknown/unkgemL.csv", 
        "data/unknown/unkgemU.csv",
        "data/unknown/numerical/unkgemB.csv",
        "data/unknown/numerical/unkgemL.csv",
        "data/unknown/numerical/unkgemU.csv",
        "unkgemB.csv",
        "unkgemL.csv",
        "unkgemU.csv"
    ]
    
    cleaned = 0
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            os.remove(file_path)
            cleaned += 1
    
    if cleaned > 0:
        print(f"   ✅ Cleaned {cleaned} previous files")
    else:
        print(f"   ✅ No previous files to clean")

def main():
    """Main standalone gem selector"""
    print("STANDALONE GEM SELECTOR")
    print("=" * 30)
    print("Bypasses main.py input validation bug")
    print("Integrates CORRECTED txt_to_unkgem.py functionality")
    print("CORRECTED: L Light MAX → 50000 (not 450nm → 50000)")
    print("Includes: Database-matching normalization + 0-100 scaling")
    print()
    
    # Clean previous files
    clean_previous_files()
    
    # Show available files
    gems, complete_gems = show_data_raw_files()
    if not gems:
        return False
    
    # Interactive selection
    selection_result = select_gem_interactively(gems, complete_gems)
    if not selection_result:
        print("Selection cancelled")
        return False
    
    selected_files, gem_id = selection_result
    
    # Convert to unkgem format
    success = convert_to_unkgem_format(selected_files, gem_id)
    
    if success:
        print(f"\n✅ READY FOR ANALYSIS!")
        print("Files prepared with CORRECTED normalization + 0-100 scaling")
        
        # Ask if user wants to run analysis now
        run_now = input(f"\nRun numerical analysis now? (y/n): ").strip().lower()
        if run_now == 'y':
            try:
                import subprocess
                result = subprocess.run([sys.executable, "src/numerical_analysis/gemini1.py"], 
                                      capture_output=False, text=True)
                if result.returncode == 0:
                    print("✅ Analysis completed!")
                else:
                    print(f"❌ Analysis failed with code: {result.returncode}")
            except Exception as e:
                print(f"❌ Could not run analysis: {e}")
                print("Run manually: python src/numerical_analysis/gemini1.py")
        
        return True
    else:
        print(f"\n❌ FAILED")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎯 SUCCESS: Gem selection and conversion completed!")
        print(f"📊 Applied CORRECTED normalization: L Light MAX → 50000")
        print(f"📊 Applied 0-100 scaling for proper database matching")
    else:
        print(f"\n❌ FAILED: Could not complete gem selection")
    
    input("\nPress Enter to exit...")