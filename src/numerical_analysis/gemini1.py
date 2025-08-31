import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
import os
from pathlib import Path

def normalize_spectrum(wavelengths, intensities, light_source):
    if light_source == 'B':
        anchor = 650
        target = 50000
        idx = np.argmin(np.abs(wavelengths - anchor))
        scale = target / intensities[idx] if intensities[idx] != 0 else 1
        return intensities * scale

    elif light_source == 'L':
        anchor = 450
        target = 50000
        idx = np.argmin(np.abs(wavelengths - anchor))
        scale = target / intensities[idx] if intensities[idx] != 0 else 1
        return intensities * scale

    elif light_source == 'U':
        mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
        window = intensities[mask]
        max_811 = window.max() if len(window) > 0 else 1
        return intensities / max_811 if max_811 > 0 else intensities

    else:
        return intensities

def load_spectrum(filename):
    df = pd.read_csv(filename, header=None, names=['wavelength', 'intensity'])
    return df['wavelength'].values, df['intensity'].values

def compare_spectra(unk_wave, unk_int, db_wave, db_int):
    mse = np.mean((unk_int - db_int)**2)
    mae = np.mean(np.abs(unk_int - db_int))
    mape = np.mean(np.abs((unk_int - db_int) / unk_int)) * 100
    corr, _ = pearsonr(unk_int, db_int)
    area_diff = np.abs(np.trapz(unk_int, unk_wave) - np.trapz(db_int, db_wave))
    return mse, mae, mape, corr, area_diff

def compute_match_score(unknown, reference):
    merged = pd.merge(unknown, reference, on='wavelength', suffixes=('_unknown', '_ref'))
    score = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
    log_score = np.log1p(score)
    return log_score

def plot_horizontal_comparison(unknown_files, existing_db, base_id, gem_best_names, gem_name_map):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, light_source in enumerate(['B', 'L', 'U']):
        try:
            # Use the full path to unknown files
            unknown_file_path = unknown_files[light_source]
            if unknown_file_path.exists():
                unknown = pd.read_csv(unknown_file_path, sep='[\s,]+', header=None, names=['wavelength', 'intensity'], skiprows=1, engine='python')
                
                # Check if database file exists in existing_db
                if light_source in existing_db:
                    db_file_path = existing_db[light_source]
                    db = pd.read_csv(db_file_path)
                    match_name = gem_best_names.get(base_id, {}).get(light_source)
                    if match_name:
                        reference = db[db['full_name'] == match_name]
                        gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
                        axs[i].plot(unknown['wavelength'], unknown['intensity'], label=f"Unknown {light_source}", color='orange', linewidth=0.5)
                        axs[i].plot(reference['wavelength'], reference['intensity'], label=f"{gem_desc} {match_name}", color='black', linestyle='--', linewidth=0.5)
                        axs[i].set_xlabel('Wavelength (nm)')
                        axs[i].set_ylabel('Intensity')
                        axs[i].set_title(f"{light_source} vs {gem_desc} {match_name}")
                        axs[i].legend()
                    else:
                        axs[i].set_title(f"{light_source}: No Match Found")
                else:
                    axs[i].set_title(f"{light_source}: Database Missing")
            else:
                axs[i].set_title(f"{light_source}: Data Missing")
        except FileNotFoundError:
            axs[i].set_title(f"{light_source}: Data Missing")
        except Exception as e:
            axs[i].set_title(f"{light_source}: Error - {str(e)}")
    plt.tight_layout()
    plt.show()

def main():
    # Set up paths relative to script location
    script_dir = Path(__file__).parent  # src/numerical_analysis/
    project_root = script_dir.parent.parent  # Go up to gemini_gemological_analysis
    
    # Define file paths using proper directory structure
    unknown_dir = project_root / 'data' / 'unknown'
    raw_txt_dir = script_dir / 'raw_txt'  # src/numerical_analysis/raw_txt
    
    unknown_files = {
        'B': unknown_dir / 'unkgemB.csv', 
        'L': unknown_dir / 'unkgemL.csv', 
        'U': unknown_dir / 'unkgemU.csv'
    }
    
    # Database files in database/reference_spectra directory
    db_dir = project_root / 'database' / 'reference_spectra'
    db_files = {
        'B': db_dir / 'gemini_db_long_B.csv', 
        'L': db_dir / 'gemini_db_long_L.csv', 
        'U': db_dir / 'gemini_db_long_U.csv'
    }
    
    print("🔍 Gemini Gem Identification System")
    print("=" * 50)
    print(f"Project root: {project_root}")
    print(f"Unknown files directory: {unknown_dir}")
    print(f"Raw txt directory: {raw_txt_dir}")
    print(f"Database directory: {db_dir}")
    print()

    # Determine available light sources from raw files
    raw_sources = set()
    if raw_txt_dir.exists():
        for f in raw_txt_dir.glob('*.txt'):
            filename = f.stem
            if len(filename) >= 1:
                # Look for B, L, U in filename
                for char in filename.upper():
                    if char in {'B', 'L', 'U'}:
                        raw_sources.add(char)
                        break
        print(f"📁 Found raw_txt directory: {raw_txt_dir}")
    else:
        print("⚠️ raw_txt directory not found. Assuming all light sources present.")
        raw_sources = {'B', 'L', 'U'}
    
    if not raw_sources:
        print("⚠️ Could not determine light sources from filenames. Assuming B, L, U.")
        raw_sources = {'B', 'L', 'U'}
    
    print(f"🔍 Unknown gem uses {len(raw_sources)} light sources: {', '.join(sorted(raw_sources))}")
    
    # Check which unknown files exist
    existing_unknown = {}
    for light_source in ['B', 'L', 'U']:
        if unknown_files[light_source].exists():
            existing_unknown[light_source] = unknown_files[light_source]
            print(f"✅ Found: {unknown_files[light_source]}")
        else:
            print(f"⚠️ Missing: {unknown_files[light_source]}")
    
    if not existing_unknown:
        print("\n❌ No unknown gem files found!")
        print(f"Expected location: {unknown_dir}")
        print("Please run txt_to_unkgem.py first to convert your raw files.")
        return
    
    # Initialize existing_db dictionary BEFORE using it
    existing_db = {}
    
    # Check if database directory exists
    if not db_dir.exists():
        print(f"\n❌ Database directory not found: {db_dir}")
        print("Please make sure gemini_db_long_*.csv files are in database/reference_spectra/")
        return
    
    # Check which database files exist
    for light_source in ['B', 'L', 'U']:
        if db_files[light_source].exists():
            existing_db[light_source] = db_files[light_source]
            print(f"✅ Found database: {db_files[light_source].name}")
        else:
            print(f"⚠️ Missing database: {db_files[light_source]}")
    
    if not existing_db:
        print(f"\n❌ No database files found in: {db_dir}")
        print("Please make sure gemini_db_long_*.csv files are available.")
        return

    all_matches = {}
    gem_best_scores = {}
    gem_best_names = {}
    gems_by_light_source = {'B': set(), 'L': set(), 'U': set()}

    for light_source in ['B', 'L', 'U']:
        if light_source not in existing_unknown:
            print(f"⚠️ Skipping {light_source} - unknown file not found")
            continue
            
        if light_source not in existing_db:
            print(f"⚠️ Skipping {light_source} - database file not found")
            continue

        try:
            print(f"🔄 Processing {light_source} spectra...")
            unknown = pd.read_csv(existing_unknown[light_source], sep='[\s,]+', header=None, names=['wavelength', 'intensity'], skiprows=1, engine='python')
            db = pd.read_csv(existing_db[light_source])

            scores = []
            for gem_name in db['full_name'].unique():
                reference = db[db['full_name'] == gem_name]
                score = compute_match_score(unknown, reference)
                scores.append((gem_name, score))
                gems_by_light_source[light_source].add(gem_name)

            sorted_scores = sorted(scores, key=lambda x: x[1])
            all_matches[light_source] = sorted_scores

            print(f"\n✅ Best Matches for {light_source}:")
            for gem, score in sorted_scores[:5]:
                print(f"  {gem}: Log Score = {score:.2f}")

            for gem_name, score in sorted_scores:
                base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                if base_id not in gem_best_scores:
                    gem_best_scores[base_id] = {}
                    gem_best_names[base_id] = {}
                if score < gem_best_scores[base_id].get(light_source, np.inf):
                    gem_best_scores[base_id][light_source] = score
                    gem_best_names[base_id][light_source] = gem_name

        except FileNotFoundError as e:
            print(f"❌ File not found for {light_source}: {e}")
        except Exception as e:
            print(f"❌ Error processing {light_source}: {e}")

    # Filter gems that have matches for available light sources
    available_sources = set(existing_unknown.keys())
    gem_best_scores = {gid: s for gid, s in gem_best_scores.items() if set(s.keys()) >= available_sources}
    gem_best_names = {gid: n for gid, n in gem_best_names.items() if gid in gem_best_scores}

    if not gem_best_scores:
        print("\n❌ No gems found with matches across all available light sources")
        return

    aggregated_scores = {base_id: sum(scores[ls] for ls in available_sources) for base_id, scores in gem_best_scores.items()}
    final_sorted = sorted(aggregated_scores.items(), key=lambda x: x[1])

    # Load gem library
    gem_name_map = {}
    gemlib_path = project_root / 'gemlib_structural_ready.csv'
    
    try:
        if gemlib_path.exists():
            gemlib = pd.read_csv(gemlib_path)
            gemlib.columns = gemlib.columns.str.strip()
            if 'Reference' in gemlib.columns:
                gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                if all(col in gemlib.columns for col in expected_columns):
                    gemlib['Gem Description'] = gemlib[expected_columns].apply(lambda x: ' '.join([v if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                    gem_name_map = dict(zip(gemlib['Reference'], gemlib['Gem Description']))
                    print(f"✅ Loaded gem library: {len(gem_name_map)} entries")
                else:
                    print(f"⚠️ Expected columns {expected_columns} not found in gemlib_structural_ready.csv")
            else:
                print("⚠️ 'Reference' column not found in gemlib_structural_ready.csv")
        else:
            print(f"⚠️ Gem library not found: {gemlib_path}")
    except Exception as e:
        print(f"⚠️ Could not load gemlib_structural_ready.csv: {e}")

    print("\n" + "="*70)
    print("✅ Overall Best Matches:")
    print("="*70)
    
    for i, (base_id, total_score) in enumerate(final_sorted, start=1):
        gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
        sources = gem_best_scores.get(base_id, {}).keys()
        print(f"\n🏆 Rank {i}: {gem_desc} (Gem {base_id})")
        print(f"   💎 Total Log Score = {total_score:.2f}")
        print(f"   🔎 Light sources matched: {', '.join(sorted(sources))} ({len(sources)})")
        for ls in sorted(sources):
            val = gem_best_scores[base_id][ls]
            gem_name = gem_best_names[base_id][ls]
            print(f"      {ls} Score: {val:.2f} ({gem_name})")
        
        # Show comparison plot for top 3 matches
        if i <= 3 and existing_db:  # Make sure existing_db is available
            print(f"   📊 Showing comparison plot...")
            plot_horizontal_comparison(unknown_files, existing_db, base_id, gem_best_names, gem_name_map)
    
    if final_sorted:
        top_gem = gem_name_map.get(str(final_sorted[0][0]), f'Gem {final_sorted[0][0]}')
        print(f"\n🎉 Analysis complete! Top match: {top_gem}")

if __name__ == "__main__":
    main()
