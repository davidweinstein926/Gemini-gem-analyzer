#!/usr/bin/env python3
"""
enhanced_gemini1.py - Complete Numerical Analysis with Visualization
Integrated with comprehensive results display system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
import os
import sys

# Try to import visualization module
try:
    from src.visualization.result_visualizer import display_analysis_results
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Visualization module not available - using basic display")

def normalize_spectrum(wavelengths, intensities, light_source):
    """Enhanced normalization with better error handling"""
    if light_source == 'B':
        anchor = 650
        target = 50000
        idx = np.argmin(np.abs(wavelengths - anchor))
        if intensities[idx] != 0:
            scale = target / intensities[idx]
            normalized = intensities * scale
            print(f"   B normalization: {anchor}nm ({intensities[idx]:.0f}) → {target}")
            return normalized
        else:
            print(f"   ⚠️ B normalization failed: zero intensity at {anchor}nm")
            return intensities

    elif light_source == 'L':
        anchor = 450
        target = 50000
        idx = np.argmin(np.abs(wavelengths - anchor))
        if intensities[idx] != 0:
            scale = target / intensities[idx]
            normalized = intensities * scale
            print(f"   L normalization: {anchor}nm ({intensities[idx]:.0f}) → {target}")
            return normalized
        else:
            print(f"   ⚠️ L normalization failed: zero intensity at {anchor}nm")
            return intensities

    elif light_source == 'U':
        mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
        window = intensities[mask]
        if len(window) > 0 and window.max() > 0:
            max_811 = window.max()
            target = 15000
            normalized = intensities * (target / max_811)
            print(f"   U normalization: 811nm window ({max_811:.0f}) → {target}")
            return normalized
        else:
            print(f"   ⚠️ U normalization failed: no valid data in 811nm window")
            return intensities

    else:
        print(f"   ⚠️ Unknown light source: {light_source}")
        return intensities

def load_spectrum(filename):
    """Load spectrum with enhanced error handling"""
    try:
        df = pd.read_csv(filename, header=None, names=['wavelength', 'intensity'])
        print(f"   📊 Loaded {len(df)} data points from {filename}")
        return df['wavelength'].values, df['intensity'].values
    except Exception as e:
        print(f"   ❌ Error loading {filename}: {e}")
        return None, None

def compare_spectra(unk_wave, unk_int, db_wave, db_int):
    """Enhanced spectral comparison with multiple metrics"""
    try:
        mse = np.mean((unk_int - db_int)**2)
        mae = np.mean(np.abs(unk_int - db_int))
        mape = np.mean(np.abs((unk_int - db_int) / (unk_int + 1e-8))) * 100  # Added small epsilon
        corr, _ = pearsonr(unk_int, db_int)
        area_diff = np.abs(np.trapz(unk_int, unk_wave) - np.trapz(db_int, db_wave))
        return mse, mae, mape, corr, area_diff
    except Exception as e:
        print(f"   ⚠️ Comparison error: {e}")
        return float('inf'), float('inf'), float('inf'), 0.0, float('inf')

def compute_match_score(unknown, reference):
    """Enhanced matching score computation"""
    try:
        merged = pd.merge(unknown, reference, on='wavelength', suffixes=('_unknown', '_ref'))
        if len(merged) == 0:
            return float('inf')
        
        score = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
        log_score = np.log1p(score)
        return log_score
    except Exception as e:
        print(f"   ⚠️ Score computation error: {e}")
        return float('inf')

def create_summary_statistics(all_matches, gem_best_scores, final_sorted):
    """Create comprehensive analysis statistics"""
    stats = {
        'total_gems_analyzed': len(final_sorted),
        'light_sources_used': list(all_matches.keys()),
        'best_match_score': final_sorted[0][1] if final_sorted else float('inf'),
        'score_range': (final_sorted[0][1], final_sorted[-1][1]) if final_sorted else (0, 0),
        'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Light source statistics
    for light in all_matches:
        matches = all_matches[light]
        scores = [score for _, score in matches]
        stats[f'{light}_matches'] = len(matches)
        stats[f'{light}_best_score'] = min(scores) if scores else float('inf')
        stats[f'{light}_avg_score'] = np.mean(scores) if scores else float('inf')
    
    return stats

def display_enhanced_results(all_matches, gem_best_scores, gem_best_names, final_sorted, gem_name_map, stats):
    """Display results with enhanced formatting and statistics"""
    
    print("\n" + "="*80)
    print("🔬 GEMINI GEMOLOGICAL ANALYSIS - COMPREHENSIVE RESULTS")
    print("="*80)
    
    # Analysis Summary
    print(f"\n📊 ANALYSIS SUMMARY:")
    print(f"   🕒 Completed: {stats['analysis_timestamp']}")
    print(f"   💎 Total gems analyzed: {stats['total_gems_analyzed']:,}")
    print(f"   💡 Light sources: {', '.join(stats['light_sources_used'])}")
    print(f"   🏆 Best match score: {stats['best_match_score']:.3f}")
    print(f"   📈 Score range: {stats['score_range'][0]:.3f} - {stats['score_range'][1]:.3f}")
    
    # Light source breakdown
    print(f"\n💡 LIGHT SOURCE BREAKDOWN:")
    for light in ['B', 'L', 'U']:
        if f'{light}_matches' in stats:
            print(f"   {light}: {stats[f'{light}_matches']:,} matches, "
                  f"best: {stats[f'{light}_best_score']:.3f}, "
                  f"avg: {stats[f'{light}_avg_score']:.3f}")
    
    # Top matches with enhanced display
    print(f"\n🏆 TOP 20 MATCHES:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Gem ID':<8} {'Total Score':<12} {'B':<8} {'L':<8} {'U':<8} {'Description'}")
    print("-" * 80)
    
    for i, (base_id, total_score) in enumerate(final_sorted[:20], start=1):
        gem_desc = gem_name_map.get(str(base_id), f"Unknown Gem {base_id}")
        scores = gem_best_scores.get(base_id, {})
        
        # Format scores
        b_score = f"{scores.get('B', 0):.2f}" if 'B' in scores else "N/A"
        l_score = f"{scores.get('L', 0):.2f}" if 'L' in scores else "N/A"
        u_score = f"{scores.get('U', 0):.2f}" if 'U' in scores else "N/A"
        
        # Truncate description for display
        desc_short = gem_desc[:35] + "..." if len(gem_desc) > 35 else gem_desc
        
        print(f"{i:<5} {base_id:<8} {total_score:<12.3f} {b_score:<8} {l_score:<8} {u_score:<8} {desc_short}")
    
    print("-" * 80)
    
    # Offer visualization
    if VISUALIZATION_AVAILABLE:
        print(f"\n🎨 ENHANCED VISUALIZATION AVAILABLE")
        choice = input(f"Launch interactive results display? (y/n): ").strip().lower()
        
        if choice == 'y':
            print(f"🚀 Launching comprehensive results visualization...")
            try:
                display_analysis_results(all_matches, gem_best_scores, gem_best_names, 
                                       final_sorted, list(all_matches.keys()))
            except Exception as e:
                print(f"❌ Visualization error: {e}")
                print("Continuing with basic display...")
    else:
        print(f"\n📋 For enhanced visualization:")
        print(f"   Copy result_visualizer.py content to src/visualization/match_display.py")
        print(f"   Then re-run analysis for interactive results display")

def main():
    """Enhanced main analysis function"""
    
    print("🔬 Starting Enhanced Gemini Numerical Analysis")
    print("=" * 60)
    
    # File paths - support both old and new directory structures
    unknown_files = {
        'B': 'data/unknown/unkgemB.csv' if os.path.exists('data/unknown/unkgemB.csv') else 'unkgemB.csv',
        'L': 'data/unknown/unkgemL.csv' if os.path.exists('data/unknown/unkgemL.csv') else 'unkgemL.csv', 
        'U': 'data/unknown/unkgemU.csv' if os.path.exists('data/unknown/unkgemU.csv') else 'unkgemU.csv'
    }
    
    db_files = {
        'B': 'databases/gemini_db_long_B.csv' if os.path.exists('databases/gemini_db_long_B.csv') else 'gemini_db_long_B.csv',
        'L': 'databases/gemini_db_long_L.csv' if os.path.exists('databases/gemini_db_long_L.csv') else 'gemini_db_long_L.csv',
        'U': 'databases/gemini_db_long_U.csv' if os.path.exists('databases/gemini_db_long_U.csv') else 'gemini_db_long_U.csv'
    }
    
    # Detect available light sources
    raw_sources = set()
    raw_txt_dir = 'raw_txt'
    if os.path.isdir(raw_txt_dir):
        for f in os.listdir(raw_txt_dir):
            if f.lower().endswith('.txt'):
                base = os.path.splitext(f)[0]
                if len(base) >= 3:
                    light_source = base[-3].upper()
                    if light_source in {'B', 'L', 'U'}:
                        raw_sources.add(light_source)
    else:
        print("⚠️ raw_txt directory not found. Checking for unknown files...")
        # Check which unknown files exist
        for light in ['B', 'L', 'U']:
            if os.path.exists(unknown_files[light]):
                raw_sources.add(light)
    
    if not raw_sources:
        print("❌ No spectral data found!")
        print("Place .txt files in data/raw_txt/ or .csv files in data/unknown/")
        return
    
    print(f"🔍 Analyzing {len(raw_sources)} light sources: {', '.join(sorted(raw_sources))}")
    
    # Initialize results storage
    all_matches = {}
    gem_best_scores = {}
    gem_best_names = {}
    gems_by_light_source = {'B': set(), 'L': set(), 'U': set()}
    
    # Analyze each light source
    for light_source in ['B', 'L', 'U']:
        if light_source not in raw_sources:
            continue
            
        print(f"\n🔬 Analyzing {light_source} light source...")
        
        try:
            # Load unknown spectrum
            unknown = pd.read_csv(unknown_files[light_source], sep='[\s,]+', header=None, 
                                names=['wavelength', 'intensity'], skiprows=1, engine='python')
            print(f"   📊 Unknown spectrum: {len(unknown)} data points")
            
            # Load reference database
            db = pd.read_csv(db_files[light_source])
            unique_gems = db['full_name'].nunique()
            print(f"   💾 Reference database: {len(db)} records, {unique_gems} unique gems")
            
            # Compute matches
            scores = []
            for gem_name in db['full_name'].unique():
                reference = db[db['full_name'] == gem_name]
                score = compute_match_score(unknown, reference)
                scores.append((gem_name, score))
                gems_by_light_source[light_source].add(gem_name)
            
            # Sort and store results
            sorted_scores = sorted(scores, key=lambda x: x[1])
            all_matches[light_source] = sorted_scores
            
            print(f"   ✅ Computed {len(sorted_scores)} match scores")
            print(f"   🏆 Best match: {sorted_scores[0][0]} (score: {sorted_scores[0][1]:.3f})")
            
            # Update best scores per gem
            for gem_name, score in sorted_scores:
                base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                if base_id not in gem_best_scores:
                    gem_best_scores[base_id] = {}
                    gem_best_names[base_id] = {}
                if score < gem_best_scores[base_id].get(light_source, np.inf):
                    gem_best_scores[base_id][light_source] = score
                    gem_best_names[base_id][light_source] = gem_name
            
        except FileNotFoundError:
            print(f"   ❌ Files not found for {light_source} light source")
            print(f"   Unknown: {unknown_files[light_source]}")
            print(f"   Database: {db_files[light_source]}")
        except Exception as e:
            print(f"   ❌ Error analyzing {light_source}: {e}")
    
    # Filter gems that have all required light sources
    gem_best_scores = {gid: s for gid, s in gem_best_scores.items() 
                      if set(s.keys()) == raw_sources}
    gem_best_names = {gid: n for gid, n in gem_best_names.items() 
                     if gid in gem_best_scores}
    
    print(f"\n🎯 Found {len(gem_best_scores)} gems with complete light source data")
    
    # Calculate aggregated scores
    aggregated_scores = {base_id: sum(scores[ls] for ls in raw_sources) 
                        for base_id, scores in gem_best_scores.items()}
    final_sorted = sorted(aggregated_scores.items(), key=lambda x: x[1])
    
    # Load gemstone library
    gem_name_map = {}
    gemlib_path = 'database/gem_library/gemlib_structural_ready.csv'
    
    try:
        gemlib = pd.read_csv(gemlib_path)
        gemlib.columns = gemlib.columns.str.strip()
        if 'Reference' in gemlib.columns:
            gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
            expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
            if all(col in gemlib.columns for col in expected_columns):
                gemlib['Gem Description'] = gemlib[expected_columns].apply(
                    lambda x: ' '.join([v if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                gem_name_map = dict(zip(gemlib['Reference'], gemlib['Gem Description']))
                print(f"   📚 Loaded {len(gem_name_map)} gem descriptions")
            else:
                print(f"   ⚠️ Expected columns not found in gemstone library")
        else:
            print("   ⚠️ 'Reference' column not found in gemstone library")
    except Exception as e:
        print(f"   ⚠️ Could not load gemstone library: {e}")
    
    # Generate comprehensive statistics
    stats = create_summary_statistics(all_matches, gem_best_scores, final_sorted)
    
    # Display enhanced results
    display_enhanced_results(all_matches, gem_best_scores, gem_best_names, 
                           final_sorted, gem_name_map, stats)
    
    print(f"\n✅ Analysis complete! Total processing time: ~{datetime.now().second}s")

if __name__ == "__main__":
    main()
