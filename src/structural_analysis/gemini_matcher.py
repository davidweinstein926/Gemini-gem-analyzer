import numpy as np
from gemini_utils import normalize_spectrum

def find_best_matches(unknown_spectra, gem_db, required_sources, top_n=5):
    results = []
    for gem_id in gem_db['gem_id'].unique():
        total_score = 0
        valid = True
        for src in required_sources:
            unk_df = unknown_spectra[src]
            db_spectra = gem_db[(gem_db['gem_id'] == gem_id) & (gem_db['light_source'] == src)]
            if db_spectra.empty:
                valid = False
                break
            avg_df = db_spectra.groupby('wavelength', as_index=False)['intensity'].mean()
            avg_df = normalize_spectrum(avg_df, src)
            # Here, you can choose any matching metric, for now using simple absolute difference sum
            if len(unk_df) != len(avg_df):
                # Interpolate to match wavelengths if needed
                avg_df = avg_df.set_index('wavelength').reindex(unk_df['wavelength']).interpolate().reset_index()
            score = np.sum(np.abs(unk_df['intensity'] - avg_df['intensity']))
            total_score += score
        if valid:
            results.append((gem_id, total_score))
    results.sort(key=lambda x: x[1])
    return results[:top_n]