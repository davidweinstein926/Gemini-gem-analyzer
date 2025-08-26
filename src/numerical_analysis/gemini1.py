import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import pearsonr
import os

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

def plot_horizontal_comparison(unknown_files, db_files, base_id, gem_best_names, gem_name_map):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for i, light_source in enumerate(['B', 'L', 'U']):
        try:
            unknown = pd.read_csv(unknown_files[light_source], sep='[\s,]+', header=None, names=['wavelength', 'intensity'], skiprows=1, engine='python')
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
    unknown_files = {'B': 'unkgemB.csv', 'L': 'unkgemL.csv', 'U': 'unkgemU.csv'}
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
        print("⚠️ raw_txt directory not found. Assuming all light sources present.")
        raw_sources = {'B', 'L', 'U'}
    print(f"🔍 Unknown gem uses {len(raw_sources)} light sources: {', '.join(sorted(raw_sources))}")

    all_matches = {}
    gem_best_scores = {}
    gem_best_names = {}
    gems_by_light_source = {'B': set(), 'L': set(), 'U': set()}

    for light_source in ['B', 'L', 'U']:
        try:
            unknown = pd.read_csv(unknown_files[light_source], sep='[\s,]+', header=None, names=['wavelength', 'intensity'], skiprows=1, engine='python')
            db = pd.read_csv(db_files[light_source])

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

        except FileNotFoundError:
            print(f"⚠️ Unknown file '{unknown_files[light_source]}' not found.")

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
                print(f"⚠️ Expected columns {expected_columns} not found in gemlib_structural_ready.csv")
        else:
            print("⚠️ 'Reference' column not found in gemlib_structural_ready.csv")
    except Exception as e:
        print(f"⚠️ Could not load gemlib_structural_ready.csv: {e}")

    print("\n✅ Overall Best Matches:")
    for i, (base_id, total_score) in enumerate(final_sorted, start=1):
        gem_desc = gem_name_map.get(str(base_id), f"Gem {base_id}")
        sources = gem_best_scores.get(base_id, {}).keys()
        print(f"  Rank {i}: {gem_desc} (Gem {base_id}) - Total Log Score = {total_score:.2f}")
        print(f"  🔎 Light sources matched: {', '.join(sorted(sources))} ({len(sources)})")
        for ls in sorted(sources):
            val = gem_best_scores[base_id][ls]
            print(f"     {ls} Score: {val:.2f}")
        plot_horizontal_comparison(unknown_files, db_files, base_id, gem_best_names, gem_name_map)
    return

if __name__ == "__main__":
    main()

