# structural_match_finder.py (UPDATED to use gemini_db_long_B/L/U.csv for plotting)

import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
from datetime import datetime

def get_full_name(source_file):
    with open(source_file, 'r') as f:
        return os.path.splitext(f.readline().strip())[0]

def load_and_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace('﻿', '')
    return df

db = load_and_clean("gemini_structural_db.csv")
b = load_and_clean("structural_features_output_B.csv")
l = load_and_clean("structural_features_output_L.csv")
u = load_and_clean("structural_features_output_U.csv")

print("B columns:", b.columns.tolist())
print("L columns:", l.columns.tolist())
print("U columns:", u.columns.tolist())
print("DB columns:", db.columns.tolist())

b_name = get_full_name("unkgemB_source.txt")
l_name = get_full_name("unkgemL_source.txt")
u_name = get_full_name("unkgemU_source.txt")

def extract_weighted_features(df):
    weights = {'T': 5, 'P': 4, 'V': 3, 'C': 2, 'S': 1}
    entries = df[['Type', 'Wavelength', 'Intensity']].copy()
    entries['Wavelength'] = entries['Wavelength'].round(1)
    entries['Intensity'] = entries['Intensity'].astype(float)
    entries['Tag'] = entries['Type'].astype(str) + '_' + entries['Wavelength'].astype(str)
    entries['Weight'] = entries['Type'].map(weights).fillna(0)
    entries['Raw'] = list(zip(entries['Wavelength'], entries['Intensity'], entries['Weight']))
    return {tag: (wl, inten, weight) for tag, wl, inten, weight in zip(entries['Tag'], entries['Wavelength'], entries['Intensity'], entries['Weight'])}

u['Wavelength'] = u['Wavelength'].astype(float)
u['Intensity'] = u['Intensity'].astype(float)
window = u[(u['Wavelength'] >= 810.5) & (u['Wavelength'] <= 811.5)]
if not window.empty:
    max_811 = window['Intensity'].max()
    if max_811 > 0:
        u['Intensity'] = u['Intensity'] / max_811

unknown_b = extract_weighted_features(b)
unknown_l = extract_weighted_features(l)
unknown_u = extract_weighted_features(u)

def parse_db_features(feature_str, light):
    features = {}
    weights = {'T': 5, 'P': 4, 'V': 3, 'C': 2, 'S': 1}
    if pd.isna(feature_str): return features
    items = [x.strip() for x in str(feature_str).split(',') if x.strip()]
    raw_features = []
    for item in items:
        match = re.match(r"(\d+(?:\.\d+)?)([A-Z]):(\d+)", item)
        if match:
            wl = float(match.group(1))
            typ = match.group(2)
            inten = float(match.group(3))
            tag = f"{typ}_{round(wl, 1)}"
            weight = weights.get(typ, 0)
            raw_features.append((tag, wl, inten, weight))

    if light == 'U':
        max_811 = max([inten for tag, wl, inten, _ in raw_features if 810.5 <= wl <= 811.5] + [0])
        if max_811 > 0:
            raw_features = [(tag, wl, inten / max_811, weight) for tag, wl, inten, weight in raw_features]

    return {tag: (wl, inten, weight) for tag, wl, inten, weight in raw_features}

WAVELENGTH_TOLERANCE = 2.0
INTENSITY_TOLERANCE = 0.10
match_scores = defaultdict(lambda: {'B': None, 'L': None, 'U': None})

for light, unknown_features in zip(['B', 'L', 'U'], [unknown_b, unknown_l, unknown_u]):
    subset = db[db['LightSource'] == light]
    for idx, row in subset.iterrows():
        gem_id = row['GemID']
        orientation = row['Orientation']
        scan = row['ScanNo']
        gem_key = f"{gem_id}{light}{orientation}{scan}"
        db_features = parse_db_features(row['Expected_Features'], light)

        missing_count = 0
        extra_count = 0
        score = 0
        matched_db_tags = set()

        for tag, (uwl, uinten, weight) in unknown_features.items():
            found = False
            for dtag, (dwl, dinten, dweight) in db_features.items():
                if tag[0] == dtag[0] and abs(uwl - dwl) <= WAVELENGTH_TOLERANCE and abs(uinten - dinten) / uinten <= INTENSITY_TOLERANCE:
                    matched_db_tags.add(dtag)
                    found = True
                    break
            if not found:
                score += weight
                missing_count += 1

        for dtag, (dwl, dinten, dweight) in db_features.items():
            if dtag not in matched_db_tags:
                found = False
                for utag, (uwl, uinten, _) in unknown_features.items():
                    if dtag[0] == utag[0] and abs(uwl - dwl) <= WAVELENGTH_TOLERANCE and abs(uinten - dinten) / uinten <= INTENSITY_TOLERANCE:
                        found = True
                        break
                if not found:
                    score += dweight
                    extra_count += 1

        match_scores[gem_id][light] = {
            'score': score,
            'missing': missing_count,
            'extra': extra_count,
            'ID': gem_id,
            'Orientation': orientation,
            'Scan': scan,
            'Features': row['Expected_Features'],
            'GemFullName': gem_key
        }

results = []
for gem_id, data in match_scores.items():
    scores = [data[k]['score'] for k in ['B', 'L', 'U'] if data[k]]
    if len(scores) < 2:
        continue
    total = sum(scores)
    max_score = max(scores)
    results.append((gem_id, total, max_score, data))

results.sort(key=lambda x: (x[1], x[2]))
best_gem_id, best_score, best_max_score, best_data = results[0]
print(f"\n✅ Final Structural Match: Gem {best_gem_id} (total score: {best_score})")

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
rows = []
for gem_id, data in match_scores.items():
    for light in ['B', 'L', 'U']:
        if data[light]:
            r = data[light]
            rows.append([
                r['ID'], r['Orientation'], r['Scan'], light,
                r['GemFullName'], r['score'], r['missing'], r['extra'], r['Features']
            ])

df_export = pd.DataFrame(rows, columns=[
    'GemID', 'Orientation', 'ScanNo', 'LightSource',
    'GemFullName', 'LogScore', 'MissingCount', 'ExtraCount', 'Expected_Features'
])

outname = f"structural_results_gemini_{now_str}.csv"
df_export.to_csv(outname, index=False)
print(f"📁 Exported full results to: {outname}")

# -------- NEW PLOTTING --------
def load_db_spectrum(gem_full_name, light):
    path_map = {
        'B': 'gemini_db_long_B.csv',
        'L': 'gemini_db_long_L.csv',
        'U': 'gemini_db_long_U.csv'
    }
    if light not in path_map:
        return None
    try:
        df = pd.read_csv(path_map[light])
        df['full_name'] = df['full_name'].astype(str).str.strip().str.upper()
        gem_full_name = gem_full_name.strip().upper()
        df = df[df['full_name'] == gem_full_name]
        return df[['wavelength', 'intensity']].reset_index(drop=True)
    except Exception as e:
        print(f"❌ Error loading spectrum for {gem_full_name}: {e}")
        return None

def plot_overlay(ax, gem_light_id, unk_path, label1, label2, title):
    light = next((ch for ch in gem_light_id if ch in 'BLU'), '?')
    db_df = load_db_spectrum(gem_light_id, light)
    if db_df is None or not os.path.exists(unk_path):
        ax.set_title(f"⚠️ Missing data for {title}")
        return
    try:
        unk_df = pd.read_csv(unk_path)
    except Exception:
        unk_df = pd.read_csv(unk_path, sep=',', header=0)

    unk_df = unk_df.apply(pd.to_numeric, errors='coerce').dropna()

    if light == 'U':
        for df in [unk_df, db_df]:
            window = df[(df['wavelength'] >= 810.5) & (df['wavelength'] <= 811.5)]
            if not window.empty:
                max_811 = window['intensity'].max()
                if max_811 > 0:
                    df['intensity'] = df['intensity'] / max_811

    ax.plot(db_df['wavelength'], db_df['intensity'], color='orange', label="Match", linewidth=0.5, zorder=1)
    ax.plot(unk_df['wavelength'], unk_df['intensity'], linestyle='--', color='Black', label="Unknown", linewidth=0.5, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend()

def plot_top_matches(n=20):
    top_n = results[:n]
    for i, (gem_id, total_score, max_score, data) in enumerate(top_n):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for j, light in enumerate(['B', 'L', 'U']):
            match = data[light]
            if not match: continue
            label = match['GemFullName']
            unk_file = {
                'B': 'unkgemB.csv',
                'L': 'unkgemL.csv',
                'U': 'unkgemU.csv'
            }.get(light)
            title = f"Gem {gem_id} | {light} | Score: {match['score']}"
            plot_overlay(axes[j], label, unk_file, label, f"Unknown {light}", title)
        plt.suptitle(f"Rank {i+1}: Gem {gem_id} (Total Score: {total_score})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

plot_top_matches(n=20)
