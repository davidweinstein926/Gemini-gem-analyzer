# gemini_total_matcher.py

import pandas as pd
import os
from gemini_utils import normalize_spectrum
from match_metrics import compute_match_score
import matplotlib.pyplot as plt

DB_FILE = 'gemini_db_long.csv'
UNK_FOLDER = '.'

def extract_metadata(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    gem_id = ''.join(filter(str.isdigit, name))
    light_source = [c.upper() for c in name if c.upper() in 'BLU']
    source = light_source[0] if light_source else None
    return name, int(gem_id) if gem_id else None, source

def load_unk_csv(file):
    df = pd.read_csv(file)
    if 'wavelength' not in df.columns or 'intensity' not in df.columns:
        raise ValueError(f"❌ '{file}' is missing required columns.")
    return df

def main():
    db = pd.read_csv(DB_FILE)
    unks = [f for f in os.listdir(UNK_FOLDER) if f.lower().startswith('unkgem') and f.lower().endswith('.csv')]
    if not unks:
        print("❌ No unkgem*.csv files found.")
        return

    for unk_file in unks:
        unk_name, _, source = extract_metadata(unk_file)
        print(f"🔍 Matching for {unk_name} ({source})...")

        unk_df = load_unk_csv(unk_file)
        unk_df = normalize_spectrum(unk_df, source)

        matches = []
        for full_name, group in db.groupby('full_name'):
            db_source = group['light_source'].iloc[0]
            if db_source != source:
                continue

            db_df = group[['wavelength', 'intensity']].copy()
            db_df = normalize_spectrum(db_df, source)

            score = compute_match_score(unk_df, db_df)
            matches.append((full_name, score))

        top_matches = sorted(matches, key=lambda x: x[1])[:5]
        print("\n🏆 Top 5 Matches:")
        for name, score in top_matches:
            print(f"{name}: Score = {score:.2f}")

        out_df = pd.DataFrame(top_matches, columns=['full_name', 'score'])
        out_name = f"results_{unk_name}.csv"
        out_df.to_csv(out_name, index=False)
        print(f"💾 Results saved to {out_name}")

        # Plot
        fig, ax = plt.subplots()
        ax.plot(unk_df['wavelength'], unk_df['intensity'], label=f"Unknown: {unk_name}", linewidth=2)
        for match_name, _ in top_matches:
            match_df = db[db['full_name'] == match_name][['wavelength', 'intensity']]
            ax.plot(match_df['wavelength'], match_df['intensity'], label=match_name, linestyle='--')

        ax.set_title(f"Spectral Match for {unk_name}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
