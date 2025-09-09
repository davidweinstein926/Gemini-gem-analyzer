import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
import os

def normalize_spectrum(wavelengths, intensities, light_source):
    """CORRECTED NORMALIZATION - implements proper scientific method"""
    
    if light_source == 'B':
        # B Light: 650nm -> 50000 -> 0-100 scale
        anchor = 650
        target = 50000
        idx = np.argmin(np.abs(wavelengths - anchor))
        if intensities[idx] != 0:
            # Step 1: Normalize to 50000 at 650nm
            normalized = intensities * (target / intensities[idx])
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
        # L Light: Maximum -> 50000 -> 0-100 scale (CORRECTED from 450nm)
        target = 50000
        max_intensity = intensities.max()
        if max_intensity != 0:
            # Step 1: Normalize maximum to 50000
            normalized = intensities * (target / max_intensity)
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
        # U Light: 811nm window -> 15000 -> 0-100 scale (CORRECTED from divide by max)
        mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
        window = intensities[mask]
        if len(window) > 0 and window.max() > 0:
            # Step 1: Normalize 811nm window max to 15000
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

def plot_horizontal_comparison(unknown_files, db_files, base_id, gem_best_names, gem_name_map):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, light_source in enumerate(['B', 'L', 'U']):
        try:
            unknown = pd.read_csv(unknown_files[light_source], header=None, names=['wavelength', 'intensity'])
            db = pd.read_csv(db_files[light_source])
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
        except FileNotFoundError:
            axs[i].set_title(f"{light_source}: Data Missing")
    plt.tight_layout()
    plt.show()

def main():
    # Updated to handle different file locations
    unknown_files = {}
    for light in ['B', 'L', 'U']:
        found = False
        for base_path in ['data/unknown', '.']:
            test_path = os.path.join(base_path, f'unkgem{light}.csv')
            if os.path.exists(test_path):
                unknown_files[light] = test_path
                found = True
                break
        if not found:
            print(f"Warning: unkgem{light}.csv not found in data/unknown or current directory")
            return

    db_files = {'B': 'gemini_db_long_B.csv', 'L': 'gemini_db_long_L.csv', 'U': 'gemini_db_long_U.csv'}

    raw_sources = set()
    if os.path.isdir('raw_txt'):
        for f in os.listdir('raw_txt'):
            if f.lower().endswith('.txt'):
                base = os.path.splitext(f)[0]
                if len(base) >= 3:
                    light_source = base[-3].upper()
                    if light_source in {'B', 'L', 'U'}:
                        raw_sources.add(light_source)
    else:
        print("Warning: raw_txt directory not found. Assuming all light sources present.")
        raw_sources = {'B', 'L', 'U'}
    print(f"Unknown gem uses {len(raw_sources)} light sources: {', '.join(sorted(raw_sources))}")

    all_matches = {}
    gem_best_scores = {}
    gem_best_names = {}
    gems_by_light_source = {'B': set(), 'L': set(), 'U': set()}

    for light_source in ['B', 'L', 'U']:
        try:
            unknown = pd.read_csv(unknown_files[light_source], header=None, names=['wavelength', 'intensity'])
            db = pd.read_csv(db_files[light_source])

            scores = []
            for gem_name in db['full_name'].unique():
                reference = db[db['full_name'] == gem_name]
                score = compute_match_score(unknown, reference)
                scores.append((gem_name, score))
                gems_by_light_source[light_source].add(gem_name)

            sorted_scores = sorted(scores, key=lambda x: x[1])
            all_matches[light_source] = sorted_scores

            print(f"\nBest Matches for {light_source}:")
            for gem, score in sorted_scores[:10]:  # Show top 10
                print(f"  {gem}: Log Score = {score:.3f}")

            for gem_name, score in sorted_scores:
                base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                if base_id not in gem_best_scores:
                    gem_best_scores[base_id] = {}
                    gem_best_names[base_id] = {}
                if score < gem_best_scores[base_id].get(light_source, np.inf):
                    gem_best_scores[base_id][light_source] = score
                    gem_best_names[base_id][light_source] = gem_name

        except FileNotFoundError:
            print(f"Warning: Unknown file '{unknown_files[light_source]}' not found.")
        except Exception as e:
            print(f"Error processing {light_source}: {e}")

    gem_best_scores = {gid: s for gid, s in gem_best_scores.items() if set(s.keys()) == raw_sources}
    gem_best_names = {gid: n for gid, n in gem_best_names.items() if gid in gem_best_scores}

    aggregated_scores = {base_id: sum(scores[ls] for ls in raw_sources) for base_id, scores in gem_best_scores.items()}
    final_sorted = sorted(aggregated_scores.items(), key=lambda x: x[1])

    gem_name_map = {}
    try:
        gemlib = pd.read_csv('gemlib_structural_ready.csv')
        gemlib.columns = gemlib.columns.str.strip()
        if 'Reference' in gemlib.columns:
            gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
            expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
            if all(col in gemlib.columns for col in expected_columns):
                gemlib['Gem Description'] = gemlib[expected_columns].apply(lambda x: ' '.join([v if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                gem_name_map = dict(zip(gemlib['Reference'], gemlib['Gem Description']))
            else:
                print(f"Warning: Expected columns {expected_columns} not found in gemlib_structural_ready.csv")
        else:
            print("Warning: 'Reference' column not found in gemlib_structural_ready.csv")
    except Exception as e:
        print(f"Warning: Could not load gemlib_structural_ready.csv: {e}")

    print("\nCOMPLETE RANKING - ALL MATCHES:")
    print("=" * 80)
    
    # Show more results and highlight C0034 specifically
    for i, (base_id, total_score) in enumerate(final_sorted[:50], start=1):  # Show top 50
        gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
        sources = gem_best_scores.get(base_id, {}).keys()
        
        # Highlight C0034 specifically
        marker = " *** C0034 SELF-MATCH ***" if base_id == 'C0034' else ""
        
        print(f"  Rank {i:2}: {gem_desc} (Gem {base_id}) - Total Log Score = {total_score:.3f}{marker}")
        for ls in sorted(sources):
            val = gem_best_scores[base_id][ls]
            print(f"         {ls} Score: {val:.3f}")
        
        # Show detailed comparison for C0034
        if base_id == 'C0034':
            print(f"    C0034 ANALYSIS:")
            if total_score < 1.0:
                print(f"    ✅ PERFECT MATCH - Normalization successful!")
            elif total_score < 5.0:
                print(f"    ✅ EXCELLENT MATCH - Very close!")
            elif total_score < 15.0:
                print(f"    ⚠️ GOOD MATCH - Some differences remain")
            else:
                print(f"    ❌ POOR MATCH - Significant normalization differences")
        
        print()
        
        # Stop after showing C0034 or first 20 if C0034 not in top 50
        if base_id == 'C0034' or i >= 20:
            if base_id != 'C0034' and i >= 20:
                # Look for C0034 in remaining results
                c0034_result = next((r for r in final_sorted if r[0] == 'C0034'), None)
                if c0034_result:
                    c0034_rank = next(i for i, r in enumerate(final_sorted, 1) if r[0] == 'C0034')
                    print(f"  ...")
                    print(f"  Rank {c0034_rank:2}: C0034 - Total Log Score = {c0034_result[1]:.3f} *** C0034 SELF-MATCH ***")
            break

    return final_sorted

if __name__ == "__main__":
    main()