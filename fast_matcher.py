# fast_matcher.py
import pandas as pd
from collections import defaultdict
from gemini_matcher import default_match_func as compute_match_score

def compute_match_scores(unknown_spectra, db):
    results = []
    required_sources = set(unknown_spectra.keys())
    gem_counts = db.groupby('gem_id')['light_source'].apply(set)
    eligible_gems = gem_counts[gem_counts == required_sources].index
    db = db[db['gem_id'].isin(eligible_gems)]

    for gem_id in db['gem_id'].unique():
        gem_result = {'gem_id': gem_id, 'scores': {}}
        gem_spectra = db[db['gem_id'] == gem_id]
        skip = False

        for src in required_sources:
            if src not in unknown_spectra:
                skip = True
                break
            unk_df = unknown_spectra[src]
            db_df = gem_spectra[gem_spectra['light_source'] == src]

            # Group and sort
            grouped = db_df.groupby(['position', 'pass_number'])
            best_score = float('inf')

            # Try all variants and keep the lowest match score
            for (_, _), df2 in grouped:
                try:
                    score = compute_match_score(unk_df, df2)
                    if score == 0.0:
                        best_score = 0.0
                        break  # Perfect match found, stop looking
                    elif score < best_score:
                        best_score = score
                except Exception:
                    continue

            if best_score == float('inf'):
                skip = True
                break
            gem_result['scores'][src] = best_score

        if not skip:
            gem_result['total_score'] = sum(gem_result['scores'].values())
            results.append(gem_result)

    return results

def apply_loh_rule(candidates):
    return min(candidates, key=lambda g: max(g['scores'].values()))['gem_id']

def rank_top_matches(candidates, best_gem_id):
    sorted_matches = sorted(candidates, key=lambda r: r['total_score'])
    return sorted_matches[:5]
