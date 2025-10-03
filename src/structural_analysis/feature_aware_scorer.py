#!/usr/bin/env python3
"""
FEATURE-AWARE STRUCTURAL ANALYZER
Replaces the broken wavelength-only comparison with intelligent feature matching

Key improvements:
- Matches features by type (Peak to Peak, Mound to Mound)
- Type-specific tolerances (Peak ±1nm, Mound ±5nm)
- Excludes Summary/Baseline rows from comparison
- UV normalization to 811nm reference
- Penalty-based scoring (lower = better match)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class FeatureAwareScorer:
    """Feature-aware scoring engine for structural spectral data"""
    
    def __init__(self):
        # Type-specific wavelength tolerances (nm)
        self.tolerances = {
            'Peak': 1.0,      # Peak positions very consistent
            'Crest': 1.0,     # Crest positions consistent  
            'Start': 5.0,     # Mound/feature starts vary more
            'End': 5.0,       # Mound/feature ends vary more
            'Shoulder': 3.0,  # Shoulders medium variance
            'Trough': 2.0,    # Troughs fairly consistent
            'Valley': 2.0     # Valleys fairly consistent
        }
        
        # Feature importance weights
        self.weights = {
            'Trough': 5,
            'Peak': 4,
            'Valley': 3,
            'Crest': 2,
            'Start': 1,
            'End': 1,
            'Shoulder': 1
        }
        
        # Penalty values
        self.missing_penalty = 5.0
        self.extra_penalty = 20.0
        
        # Intensity tolerance (relative)
        self.intensity_tolerance = 0.10  # 10%
        
        # UV normalization reference
        self.uv_ref_wavelength = 811.0
        self.uv_ref_window = 0.5  # ±0.5nm around 811
    
    def extract_halogen_features(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Extract Halogen/Laser structural features
        
        Returns dict of features keyed by Feature name with Start/Crest/End positions
        """
        features = {}
        
        # Exclude Summary rows
        df = df[df['Point_Type'] != 'Summary'].copy()
        
        # Group by Feature column
        if 'Feature' not in df.columns:
            return features
        
        for feature_name, group in df.groupby('Feature'):
            feature_data = {
                'name': feature_name,
                'points': {}
            }
            
            for _, row in group.iterrows():
                point_type = row.get('Point_Type', '')
                wavelength = row.get('Wavelength', 0)
                intensity = row.get('Intensity', 0)
                
                if point_type and wavelength > 0:
                    feature_data['points'][point_type] = {
                        'wavelength': float(wavelength),
                        'intensity': float(intensity)
                    }
            
            # Store additional metadata
            if 'Symmetry_Ratio' in df.columns:
                sym_vals = group['Symmetry_Ratio'].dropna()
                if not sym_vals.empty:
                    feature_data['symmetry_ratio'] = float(sym_vals.iloc[0])
            
            if 'Width_nm' in df.columns:
                width_vals = group['Width_nm'].dropna()
                if not width_vals.empty:
                    feature_data['width_nm'] = float(width_vals.iloc[0])
            
            features[feature_name] = feature_data
        
        return features
    
    def extract_uv_features(self, df: pd.DataFrame) -> Dict[int, Dict]:
        """Extract UV peak features with 811nm normalization
        
        Returns dict of peaks keyed by Peak_Number
        """
        features = {}
        
        if 'Peak_Number' not in df.columns:
            return features
        
        # Find 811nm reference peak for normalization
        ref_intensity = None
        ref_window = (self.uv_ref_wavelength - self.uv_ref_window, 
                      self.uv_ref_wavelength + self.uv_ref_window)
        
        if 'Wavelength_nm' in df.columns:
            ref_peaks = df[
                (df['Wavelength_nm'] >= ref_window[0]) & 
                (df['Wavelength_nm'] <= ref_window[1])
            ]
            if not ref_peaks.empty:
                ref_intensity = ref_peaks['Intensity'].max()
        
        # Extract peaks
        for _, row in df.iterrows():
            peak_num = int(row['Peak_Number'])
            wavelength = float(row.get('Wavelength_nm', 0))
            intensity = float(row.get('Intensity', 0))
            
            # Normalize intensity to 811nm peak
            if ref_intensity and ref_intensity > 0:
                intensity = (intensity / ref_intensity) * 100.0
            
            features[peak_num] = {
                'wavelength': wavelength,
                'intensity': intensity,
                'prominence': float(row.get('Prominence', 0))
            }
        
        return features
    
    def match_halogen_features(self, goi_features: Dict, db_features: Dict) -> float:
        """Match Halogen features between GOI and database
        
        Returns penalty score (lower = better match)
        """
        score = 0.0
        matched_db = set()
        
        # Match GOI features to DB features
        for goi_name, goi_data in goi_features.items():
            best_match = None
            best_diff = float('inf')
            
            # Try to match with DB features
            for db_name, db_data in db_features.items():
                if db_name in matched_db:
                    continue
                
                # Compare key points (Crest is most important)
                if 'Crest' in goi_data['points'] and 'Crest' in db_data['points']:
                    goi_crest = goi_data['points']['Crest']['wavelength']
                    db_crest = db_data['points']['Crest']['wavelength']
                    
                    diff = abs(goi_crest - db_crest)
                    tolerance = self.tolerances['Crest']
                    
                    if diff <= tolerance and diff < best_diff:
                        # Check intensity match
                        goi_int = goi_data['points']['Crest']['intensity']
                        db_int = db_data['points']['Crest']['intensity']
                        
                        max_int = max(abs(goi_int), abs(db_int))
                        if max_int > 0:
                            int_diff = abs(goi_int - db_int) / max_int
                            if int_diff <= self.intensity_tolerance:
                                best_match = db_name
                                best_diff = diff
            
            if best_match:
                matched_db.add(best_match)
            else:
                # Missing feature penalty
                weight = self.weights.get('Crest', 1)
                score += self.missing_penalty * weight
        
        # Penalize extra DB features not in GOI
        unmatched_db = set(db_features.keys()) - matched_db
        for db_name in unmatched_db:
            weight = self.weights.get('Crest', 1)
            score += self.extra_penalty * weight
        
        return score
    
    def match_uv_features(self, goi_features: Dict, db_features: Dict) -> float:
        """Match UV peaks between GOI and database
        
        UV peaks have consistent wavelengths, so we match by proximity
        Returns penalty score (lower = better match)
        """
        score = 0.0
        matched_db = set()
        
        # Match GOI peaks to DB peaks
        for goi_num, goi_data in goi_features.items():
            best_match = None
            best_diff = float('inf')
            
            goi_wl = goi_data['wavelength']
            goi_int = goi_data['intensity']
            
            for db_num, db_data in db_features.items():
                if db_num in matched_db:
                    continue
                
                db_wl = db_data['wavelength']
                db_int = db_data['intensity']
                
                # Check wavelength proximity
                wl_diff = abs(goi_wl - db_wl)
                if wl_diff <= self.tolerances['Peak']:
                    # Check intensity ratio (normalized, so compare directly)
                    int_diff = abs(goi_int - db_int) / max(goi_int, 1.0)
                    
                    if int_diff <= self.intensity_tolerance and wl_diff < best_diff:
                        best_match = db_num
                        best_diff = wl_diff
            
            if best_match:
                matched_db.add(best_match)
            else:
                # Missing peak penalty
                score += self.missing_penalty * self.weights['Peak']
        
        # Penalize extra DB peaks
        unmatched_db = set(db_features.keys()) - matched_db
        score += len(unmatched_db) * self.extra_penalty * self.weights['Peak']
        
        return score
    
    def score_light_source(self, goi_df: pd.DataFrame, db_df: pd.DataFrame, 
                          light_source: str) -> Optional[float]:
        """Score match between GOI and database for one light source
        
        Returns penalty score (lower = better), or None if can't compare
        """
        try:
            if light_source in ['Halogen', 'Laser']:
                goi_features = self.extract_halogen_features(goi_df)
                db_features = self.extract_halogen_features(db_df)
                
                if not goi_features or not db_features:
                    return None
                
                return self.match_halogen_features(goi_features, db_features)
            
            elif light_source == 'UV':
                goi_features = self.extract_uv_features(goi_df)
                db_features = self.extract_uv_features(db_df)
                
                if not goi_features or not db_features:
                    return None
                
                return self.match_uv_features(goi_features, db_features)
            
            return None
            
        except Exception as e:
            print(f"Error scoring {light_source}: {e}")
            return None
    
    def check_self_match(self, goi_df: pd.DataFrame, db_df: pd.DataFrame,
                        light_source: str) -> bool:
        """Check if this is a perfect self-match (same data)
        
        Returns True if data appears identical
        """
        try:
            # Check row counts match
            if len(goi_df) != len(db_df):
                return False
            
            # For Halogen/Laser, compare wavelengths and features
            if light_source in ['Halogen', 'Laser']:
                if 'Wavelength' not in goi_df.columns or 'wavelength' not in db_df.columns:
                    return False
                
                goi_wl = sorted(goi_df['Wavelength'].dropna().round(1).tolist())
                db_wl = sorted(db_df['wavelength'].dropna().round(1).tolist())
                
                if goi_wl != db_wl:
                    return False
                
                # Check intensities match closely
                goi_sorted = goi_df.sort_values('Wavelength')
                db_sorted = db_df.sort_values('wavelength')
                
                max_diff = abs(goi_sorted['Intensity'].values - db_sorted['intensity'].values).max()
                if max_diff > 0.1:  # Allow tiny floating point differences
                    return False
                
                return True
            
            # For UV, compare peak numbers and wavelengths
            elif light_source == 'UV':
                if 'Peak_Number' not in goi_df.columns:
                    return False
                
                goi_peaks = set(goi_df['Peak_Number'].astype(int))
                
                # DB might not have Peak_Number, use wavelengths
                if 'Wavelength_nm' in goi_df.columns and 'wavelength' in db_df.columns:
                    goi_wl = sorted(goi_df['Wavelength_nm'].round(1).tolist())
                    db_wl = sorted(db_df['wavelength'].round(1).tolist())
                    
                    if goi_wl == db_wl:
                        return True
            
            return False
            
        except Exception:
            return False


# Integration function for use with existing analyzer
def calculate_feature_aware_score(goi_df: pd.DataFrame, db_df: pd.DataFrame,
                                 light_source: str, goi_filename: str,
                                 db_filename: str, goi_ts: str, db_ts: str,
                                 goi_base_id: str, db_base_id: str) -> Optional[Dict]:
    """Calculate feature-aware score between GOI and database gem
    
    Returns dict with score and metadata, or None if can't score
    """
    scorer = FeatureAwareScorer()
    
    # Check for self-match first
    if goi_base_id == db_base_id and goi_ts == db_ts:
        if scorer.check_self_match(goi_df, db_df, light_source):
            return {
                'score': 0.001,  # Near-perfect self-match score
                'is_perfect': True,
                'db_gem_id': db_filename,
                'db_ts': db_ts,
                'match_type': 'self_match'
            }
    
    # Calculate feature-aware score
    score = scorer.score_light_source(goi_df, db_df, light_source)
    
    if score is None:
        return None
    
    return {
        'score': score,
        'is_perfect': False,
        'db_gem_id': db_filename,
        'db_ts': db_ts,
        'match_type': 'feature_match'
    }


def convert_penalty_to_percentage(penalty_score: float) -> float:
    """Convert penalty score to match percentage (0-100)
    
    Lower penalty = higher percentage
    """
    # Assuming typical penalty range 0-100
    # 0 penalty = 100% match
    # 100+ penalty = 0% match
    
    if penalty_score <= 0:
        return 100.0
    elif penalty_score >= 100:
        return 0.0
    else:
        return max(0.0, 100.0 - penalty_score)


if __name__ == "__main__":
    print("Feature-Aware Structural Analyzer")
    print("Replace scoring functions in multi_gem_structural_analyzer.py with:")
    print("  - calculate_feature_aware_score()")
    print("  - convert_penalty_to_percentage()")
    print("\nFeatures:")
    print("  - Type-specific tolerances (Peak ±1nm, Mound ±5nm)")
    print("  - Feature-to-feature matching (Peak to Peak, etc.)")
    print("  - UV 811nm normalization")
    print("  - Penalty-based scoring")
    print("  - Self-match detection")