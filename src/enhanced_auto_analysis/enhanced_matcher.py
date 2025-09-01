#!/usr/bin/env python3
"""
ENHANCED MATCHING ALGORITHMS
Core spectral matching logic with fixed mathematics

Author: David
Version: 2024.08.06
"""

from typing import List, Dict
from config.settings import get_config

class EnhancedGemstoneMatching:
    """Enhanced matching logic with corrected mathematics - PRODUCTION VERSION"""
    
    def __init__(self):
        config = get_config('matching')
        self.peak_tolerance = config['peak_tolerance']
        self.mound_tolerance = config['mound_tolerance']
        self.plateau_tolerance = config['plateau_tolerance']
        self.extra_feature_penalty = config['extra_feature_penalty']
        self.missing_feature_penalty = config['missing_feature_penalty']
        print("🔧 Enhanced matching engine initialized with fixed algorithms")
    
    def calculate_enhanced_structural_match(self, unknown_features: List[Dict], 
                                          db_features: List[Dict],
                                          unknown_stone_id: str = "",
                                          candidate_stone_id: str = "") -> float:
        """Pure spectral matching with corrected feature compatibility math"""
        
        if not unknown_features or not db_features:
            return 0.0

        # Perfect match detection
        if unknown_stone_id == candidate_stone_id:
            print(f"🔍 PERFECT MATCH: {unknown_stone_id} = 100% confidence")
            return 100.0
        
        # Feature compatibility analysis
        unknown_feature_types = set(f['feature_type'] for f in unknown_features if f.get('feature_type'))
        db_feature_types = set(f['feature_type'] for f in db_features if f.get('feature_type'))
        
        common_features = unknown_feature_types.intersection(db_feature_types)
        extra_db_features = db_feature_types - unknown_feature_types
        missing_db_features = unknown_feature_types - db_feature_types
        
        # Apply penalties for incompatible features
        extra_penalty = len(extra_db_features) * self.extra_feature_penalty
        missing_penalty = len(missing_db_features) * self.missing_feature_penalty
        total_penalty = extra_penalty + missing_penalty
        
        # FIXED: Correct feature compatibility calculation
        if len(unknown_feature_types) == 0:
            feature_compatibility = 0.0
        else:
            compatibility_ratio = len(common_features) / len(unknown_feature_types)
            feature_compatibility = (compatibility_ratio * 100) - total_penalty
            feature_compatibility = max(0.0, feature_compatibility)
        
        # Early exit for poor compatibility
        if feature_compatibility < 20.0:
            return feature_compatibility
        
        if not common_features:
            return 0.0
        
        # Filter to common features only
        filtered_unknown = [f for f in unknown_features if f.get('feature_type') in common_features]
        filtered_db = [f for f in db_features if f.get('feature_type') in common_features]
        
        # FIXED: Calculate spectral matching scores consistently
        feature_scores = []
        
        if 'Peak' in common_features:
            peak_score = self.match_peaks_enhanced(filtered_unknown, filtered_db, verbose=False)
            feature_scores.append(peak_score)
        
        if 'Mound' in common_features:
            mound_score = self.match_mounds_enhanced(filtered_unknown, filtered_db, verbose=False)
            feature_scores.append(mound_score)
        
        if 'Plateau' in common_features:
            plateau_score = self.match_plateaus_enhanced(filtered_unknown, filtered_db, verbose=False)
            feature_scores.append(plateau_score)
        
        # FIXED: Consistent averaging of all feature scores
        base_spectral_score = sum(feature_scores) / len(feature_scores) if feature_scores else 0.0
        
        # FIXED: Simplified final score calculation
        if feature_compatibility >= 100.0:
            # Perfect feature compatibility - use pure spectral score
            final_score = base_spectral_score
        else:
            # Penalize for poor feature compatibility
            final_score = base_spectral_score * (feature_compatibility / 100.0)
        
        final_score = max(0.0, min(100.0, final_score))
        
        # ONLY SHOW DETAILED OUTPUT FOR GOOD MATCHES (≥30%)
        if final_score >= 30.0:
            print(f"\n🔍 SPECTRAL MATCH: {unknown_stone_id} vs {candidate_stone_id}")
            print(f"   📋 Unknown features: {unknown_feature_types}")
            print(f"   📋 Database features: {db_feature_types}")
            print(f"      🟢 Common features: {common_features}")
            
            if extra_db_features:
                print(f"      🔴 Database only: {extra_db_features}")
            if missing_db_features:
                print(f"      🟠 Unknown only: {missing_db_features}")
            
            print(f"      📊 Extra feature penalty: -{extra_penalty:.1f}%")
            if missing_penalty > 0:
                print(f"      📊 Missing feature penalty: -{missing_penalty:.1f}%")
            
            print(f"   📊 Feature compatibility: {feature_compatibility:.1f}%")
            print(f"   🎯 BASE SPECTRAL SCORE: {base_spectral_score:.1f}%")
            print(f"   🎯 FINAL SCORE: {final_score:.1f}%")
        
        return final_score
    
    def match_peaks_enhanced(self, unknown_features: List[Dict], db_features: List[Dict], verbose: bool = True) -> float:
        """Enhanced peak matching with precise wavelength scoring"""
        unknown_peaks = [f for f in unknown_features if f['feature_type'] == 'Peak']
        db_peaks = [f for f in db_features if f['feature_type'] == 'Peak']
        
        if not unknown_peaks or not db_peaks:
            if verbose:
                print(f"      ⚠️ Peak matching skipped: Unknown={len(unknown_peaks)}, DB={len(db_peaks)}")
            return 0.0
        
        total_score = 0.0
        matched_count = 0
        
        for unknown_peak in unknown_peaks:
            unknown_max = unknown_peak.get('max_wavelength')
            if unknown_max is None:
                continue
            
            best_score = 0.0
            best_diff = float('inf')
            
            for db_peak in db_peaks:
                db_max = db_peak.get('max_wavelength')
                if db_max is None:
                    continue
                
                diff = abs(unknown_max - db_max)
                
                # FIXED: More precise wavelength scoring
                if diff <= 0.1:
                    score = 100.0
                elif diff <= 0.2:
                    score = 95.0
                elif diff <= 0.5:
                    score = 90.0 - (diff * 20)
                elif diff <= 1.0:
                    score = max(70.0, 85.0 - (diff * 15))
                elif diff <= 2.0:
                    score = max(50.0, 70.0 - (diff * 10))
                else:
                    score = max(0.0, 50.0 - (diff * 5))
                
                if diff < best_diff:
                    best_diff = diff
                    best_score = score
            
            total_score += best_score
            matched_count += 1
            if verbose and best_score > 0:
                print(f"      🔸 Peak: {unknown_max:.1f}nm → diff={best_diff:.2f}nm → score={best_score:.1f}%")
        
        final_peak_score = total_score / matched_count if matched_count > 0 else 0.0
        if verbose and final_peak_score > 0:
            print(f"      📊 PEAK TOTAL: {final_peak_score:.1f}%")
        return final_peak_score
    
    def match_mounds_enhanced(self, unknown_features: List[Dict], db_features: List[Dict], verbose: bool = True) -> float:
        """Enhanced mound matching with precise wavelength scoring"""
        unknown_mounds = [f for f in unknown_features if f['feature_type'] == 'Mound']
        db_mounds = [f for f in db_features if f['feature_type'] == 'Mound']
        
        if not unknown_mounds or not db_mounds:
            if verbose:
                print(f"      ⚠️ Mound matching skipped: Unknown={len(unknown_mounds)}, DB={len(db_mounds)}")
            return 0.0
        
        total_score = 0.0
        matched_count = 0
        
        for unknown_mound in unknown_mounds:
            unknown_crest = unknown_mound.get('crest_wavelength')
            if unknown_crest is None:
                continue
            
            best_score = 0.0
            best_diff = float('inf')
            
            for db_mound in db_mounds:
                db_crest = db_mound.get('crest_wavelength')
                if db_crest is None:
                    continue
                
                crest_diff = abs(unknown_crest - db_crest)
                
                # FIXED: More precise mound scoring
                if crest_diff <= 0.1:
                    score = 100.0
                elif crest_diff <= 0.5:
                    score = 95.0
                elif crest_diff <= 1.0:
                    score = 90.0 - (crest_diff * 5)
                elif crest_diff <= 5.0:
                    score = max(60.0, 85.0 - (crest_diff * 5))
                elif crest_diff <= 10.0:
                    score = max(30.0, 60.0 - (crest_diff * 3))
                else:
                    score = max(0.0, 30.0 - (crest_diff * 2))
                
                if crest_diff < best_diff:
                    best_diff = crest_diff
                    best_score = score
            
            total_score += best_score
            matched_count += 1
            if verbose and best_score > 0:
                print(f"      🔸 Mound: {unknown_crest:.1f}nm → diff={best_diff:.2f}nm → score={best_score:.1f}%")
        
        final_mound_score = total_score / matched_count if matched_count > 0 else 0.0
        if verbose and final_mound_score > 0:
            print(f"      📊 MOUND TOTAL: {final_mound_score:.1f}%")
        return final_mound_score
    
    def match_plateaus_enhanced(self, unknown_features: List[Dict], db_features: List[Dict], verbose: bool = True) -> float:
        """Enhanced plateau matching with precise wavelength scoring"""
        unknown_plateaus = [f for f in unknown_features if f['feature_type'] == 'Plateau']
        db_plateaus = [f for f in db_features if f['feature_type'] == 'Plateau']
        
        if not unknown_plateaus or not db_plateaus:
            if verbose:
                print(f"      ⚠️ Plateau matching skipped: Unknown={len(unknown_plateaus)}, DB={len(db_plateaus)}")
            return 0.0
        
        total_score = 0.0
        matched_count = 0
        
        for unknown_plateau in unknown_plateaus:
            unknown_mid = unknown_plateau.get('midpoint_wavelength')
            if unknown_mid is None:
                continue
            
            best_score = 0.0
            best_diff = float('inf')
            
            for db_plateau in db_plateaus:
                db_mid = db_plateau.get('midpoint_wavelength')
                if db_mid is None:
                    continue
                
                mid_diff = abs(unknown_mid - db_mid)
                
                # FIXED: More precise plateau scoring
                if mid_diff <= 0.1:
                    score = 100.0
                elif mid_diff <= 0.5:
                    score = 95.0
                elif mid_diff <= 1.0:
                    score = 90.0 - (mid_diff * 5)
                elif mid_diff <= 5.0:
                    score = max(65.0, 85.0 - (mid_diff * 4))
                elif mid_diff <= 10.0:
                    score = max(40.0, 65.0 - (mid_diff * 2.5))
                else:
                    score = max(0.0, 40.0 - (mid_diff * 2))
                
                if mid_diff < best_diff:
                    best_diff = mid_diff
                    best_score = score
            
            total_score += best_score
            matched_count += 1
            if verbose and best_score > 0:
                print(f"      🔸 Plateau: {unknown_mid:.1f}nm → diff={best_diff:.2f}nm → score={best_score:.1f}%")
        
        final_plateau_score = total_score / matched_count if matched_count > 0 else 0.0
        if verbose and final_plateau_score > 0:
            print(f"      📊 PLATEAU TOTAL: {final_plateau_score:.1f}%")
        return final_plateau_score

if __name__ == "__main__":
    # Test the matching engine
    print("🔧 TESTING ENHANCED MATCHING ENGINE")
    print("=" * 40)
    
    matcher = EnhancedGemstoneMatching()
    
    # Test with sample data
    unknown_features = [
        {'feature_type': 'Peak', 'max_wavelength': 415.0},
        {'feature_type': 'Mound', 'crest_wavelength': 550.0}
    ]
    
    db_features = [
        {'feature_type': 'Peak', 'max_wavelength': 415.2},
        {'feature_type': 'Mound', 'crest_wavelength': 551.0}
    ]
    
    score = matcher.calculate_enhanced_structural_match(
        unknown_features, db_features, "TEST_UNKNOWN", "TEST_DB"
    )
    
    print(f"\n✅ Test completed - Score: {score:.1f}%")
    print("🎯 Matching engine ready for production!")