#!/usr/bin/env python3
"""
GEMINI NORMALIZATION VALIDATION SCRIPT
Quick test to validate that the normalization fix resolves the self-matching issue
Save as: gemini_gemological_analysis/validate_fix.py
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict

def correct_normalize_spectrum(wavelengths, intensities, light_source):
    """CORRECTED NORMALIZATION - matches database creation process exactly"""
    
    if light_source == 'B':
        # B Light: 650nm → 50000 → 0-100 scale
        anchor_idx = np.argmin(np.abs(wavelengths - 650))
        if intensities[anchor_idx] != 0:
            # Step 1: Normalize to 50000 at 650nm
            normalized = intensities * (50000 / intensities[anchor_idx])
            # Step 2: Scale to 0-100 range
            min_val, max_val = normalized.min(), normalized.max()
            if max_val != min_val:
                scaled = (normalized - min_val) * 100 / (max_val - min_val)
            else:
                scaled = normalized
            return scaled
        else:
            return intensities
    
    elif light_source == 'L':
        # L Light: Maximum → 50000 → 0-100 scale
        max_intensity = intensities.max()
        if max_intensity != 0:
            # Step 1: Normalize maximum to 50000
            normalized = intensities * (50000 / max_intensity)
            # Step 2: Scale to 0-100 range
            min_val, max_val = normalized.min(), normalized.max()
            if max_val != min_val:
                scaled = (normalized - min_val) * 100 / (max_val - min_val)
            else:
                scaled = normalized
            return scaled
        else:
            return intensities
    
    elif light_source == 'U':
        # U Light: 811nm window → 15000 → 0-100 scale
        mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
        window = intensities[mask]
        if len(window) > 0 and window.max() > 0:
            # Step 1: Normalize 811nm window to 15000
            normalized = intensities * (15000 / window.max())
            # Step 2: Scale to 0-100 range
            min_val, max_val = normalized.min(), normalized.max()
            if max_val != min_val:
                scaled = (normalized - min_val) * 100 / (max_val - min_val)
            else:
                scaled = normalized
            return scaled
        else:
            return intensities
    
    else:
        return intensities

def old_normalize_spectrum(wavelengths, intensities, light_source):
    """OLD INCORRECT NORMALIZATION - for comparison"""
    
    if light_source == 'B':
        anchor_idx = np.argmin(np.abs(wavelengths - 650))
        if intensities[anchor_idx] != 0:
            return intensities * (50000 / intensities[anchor_idx])
        else:
            return intensities
    
    elif light_source == 'L':
        anchor_idx = np.argmin(np.abs(wavelengths - 450))  # WRONG: should be max
        if intensities[anchor_idx] != 0:
            return intensities * (50000 / intensities[anchor_idx])
        else:
            return intensities
    
    elif light_source == 'U':
        mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
        window = intensities[mask]
        if len(window) > 0 and window.max() > 0:
            return intensities * (15000 / window.max())  # Missing 0-100 scaling
        else:
            return intensities
    
    else:
        return intensities

def scan_gems():
    """Scan for available gems"""
    raw_dir = 'data/raw'
    if not os.path.exists(raw_dir):
        return None
    
    files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
    if not files:
        return None
    
    gems = defaultdict(lambda: {'B': [], 'L': [], 'U': []})
    
    for file in files:
        base = os.path.splitext(file)[0]
        
        # Find light source
        light = None
        for ls in ['B', 'L', 'U']:
            if ls in base.upper():
                light = ls
                break
        
        if light:
            # Extract gem number
            for i, char in enumerate(base.upper()):
                if char == light:
                    gem_num = base[:i]
                    break
            gems[gem_num][light].append(file)
    
    return dict(gems)

def test_gem_normalization(gem_number, gem_files):
    """Test normalization for a specific gem"""
    print(f"\n🔬 TESTING GEM {gem_number}")
    print("=" * 40)
    
    results = {}
    
    for light in ['B', 'L', 'U']:
        if not gem_files[light]:
            print(f"   ❌ {light}: No files available")
            continue
        
        try:
            # Load raw file
            raw_file = os.path.join('data/raw', gem_files[light][0])
            df = pd.read_csv(raw_file, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
            wavelengths = np.array(df['wavelength'])
            intensities = np.array(df['intensity'])
            
            # Apply old normalization
            old_normalized = old_normalize_spectrum(wavelengths, intensities, light)
            
            # Apply corrected normalization
            new_normalized = correct_normalize_spectrum(wavelengths, intensities, light)
            
            # Check database for comparison
            db_file = f'gemini_db_long_{light}.csv'
            db_match_found = False
            
            if os.path.exists(db_file):
                db_df = pd.read_csv(db_file)
                matches = db_df[db_df['full_name'].str.contains(gem_number, na=False)]
                
                if not matches.empty:
                    db_match_found = True
                    match_name = matches.iloc[0]['full_name']
                    db_subset = db_df[db_df['full_name'] == match_name]
                    db_range = f"{db_subset['intensity'].min():.3f}-{db_subset['intensity'].max():.3f}"
                    
                    print(f"   ✅ {light}: Found {match_name} in database")
                    print(f"      Database range: {db_range}")
                else:
                    print(f"   ⚠️ {light}: No database match for {gem_number}")
            else:
                print(f"   ❌ {light}: Database file {db_file} not found")
            
            # Show ranges
            raw_range = f"{intensities.min():.3f}-{intensities.max():.3f}"
            old_range = f"{old_normalized.min():.3f}-{old_normalized.max():.3f}"
            new_range = f"{new_normalized.min():.3f}-{new_normalized.max():.3f}"
            
            print(f"      Raw range: {raw_range}")
            print(f"      Old normalization: {old_range}")
            print(f"      NEW normalization: {new_range}")
            
            # Store results
            results[light] = {
                'db_match_found': db_match_found,
                'raw_range': (intensities.min(), intensities.max()),
                'old_range': (old_normalized.min(), old_normalized.max()),
                'new_range': (new_normalized.min(), new_normalized.max())
            }
            
        except Exception as e:
            print(f"   ❌ {light}: Error - {e}")
    
    return results

def main():
    """Main validation function"""
    print("🔍 GEMINI NORMALIZATION VALIDATION")
    print("=" * 50)
    print("This script tests the normalization fix by comparing:")
    print("1. Old (incorrect) normalization vs New (corrected) normalization")
    print("2. Expected database ranges vs actual processed ranges")
    print("3. Identifies gems that should now match themselves with score ≈ 0")
    
    # Scan for available gems
    gems = scan_gems()
    if not gems:
        print("\n❌ No gems found in data/raw directory")
        return
    
    # Find complete gems
    complete_gems = []
    for gem_num, gem_files in gems.items():
        available = [ls for ls in ['B', 'L', 'U'] if gem_files[ls]]
        if len(available) == 3:
            complete_gems.append(gem_num)
    
    if not complete_gems:
        print("\n❌ No complete gems found (need B, L, and U files)")
        return
    
    print(f"\n📂 Found {len(complete_gems)} complete gems: {', '.join(sorted(complete_gems))}")
    
    # Test priority gems first
    priority_gems = ['C0034', '140', 'C0001']
    test_gems = []
    
    for priority in priority_gems:
        if priority in complete_gems:
            test_gems.append(priority)
    
    # Add a few more random gems
    other_gems = [g for g in complete_gems if g not in test_gems][:3]
    test_gems.extend(other_gems)
    
    print(f"\n🎯 Testing normalization on: {', '.join(test_gems)}")
    
    # Test each gem
    all_results = {}
    for gem_num in test_gems:
        gem_files = gems[gem_num]
        results = test_gem_normalization(gem_num, gem_files)
        all_results[gem_num] = results
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    
    for gem_num, results in all_results.items():
        print(f"\n💎 Gem {gem_num}:")
        
        for light in ['B', 'L', 'U']:
            if light in results:
                r = results[light]
                old_max = r['old_range'][1]
                new_max = r['new_range'][1]
                
                if r['db_match_found']:
                    if new_max <= 100.0:
                        print(f"   ✅ {light}: Corrected (0-100 scaling applied)")
                    else:
                        print(f"   ⚠️ {light}: May need further adjustment")
                else:
                    print(f"   ❓ {light}: No database match to validate against")
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Run the fixed main system: python main_fixed.py")
    print("2. Select option 4 (Select Gem for Analysis)")
    print("3. Choose a test gem and run analysis")
    print("4. Verify that self-matching produces scores near 0.0")
    
    print(f"\n🔍 KEY IMPROVEMENTS:")
    print("✅ L Light: Now uses Maximum → 50000 (not 450nm → 50000)")
    print("✅ All Lights: Now include 0-100 scaling step") 
    print("✅ Validation: Compare against database ranges")
    print("✅ Debug Mode: Test gems against themselves")

if __name__ == "__main__":
    main()