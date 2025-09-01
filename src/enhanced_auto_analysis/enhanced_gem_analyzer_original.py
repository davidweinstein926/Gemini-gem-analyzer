#!/usr/bin/env python3
"""
ENHANCED GEM ANALYZER v2.1 - FIXED for 0-100 Normalization Compatibility
Implements David's sophisticated matching rules with corrected UV ratio analysis

FIXED ISSUES:
- UV ratio analysis now works with 0-100 normalized data
- Real peak threshold detection adjusted for new intensity scale
- Normalization metadata handling added
- Compatible with fixed peak detector output
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class GemNamingSystem:
    """Handles gem naming convention parsing"""
    
    @staticmethod
    def parse_gem_filename(filename: str) -> Dict[str, str]:
        """Parse gem filename into components"""
        base_name = Path(filename).stem
        
        # Remove timestamp if present
        if '_' in base_name:
            base_name = base_name.split('_')[0]
        
        # Pattern: (prefix)(light)(orientation)(scan)
        pattern = r'^(.+?)([BLU])([CP])(\d+)$'
        match = re.match(pattern, base_name, re.IGNORECASE)
        
        if match:
            prefix, light, orientation, scan = match.groups()
            
            light_source = {
                'B': 'Halogen',
                'L': 'Laser', 
                'U': 'UV'
            }.get(light.upper(), 'Unknown')
            
            return {
                'gem_id': prefix,
                'light_source': light_source,
                'orientation': orientation.upper(),
                'scan_number': int(scan),
                'full_identifier': base_name,
                'original_filename': filename
            }
        
        return {
            'gem_id': base_name,
            'light_source': 'Unknown',
            'orientation': 'Unknown',
            'scan_number': 1,
            'full_identifier': base_name,
            'original_filename': filename
        }

class SpectralMatcher:
    """FIXED: Implements David's sophisticated spectral matching rules for 0-100 normalization"""
    
    def __init__(self):
        # Matching penalties
        self.missing_feature_penalty = 10.0
        self.extra_feature_penalty = 10.0
        self.uv_missing_peak_penalty = 5.0
        
        # Wavelength tolerances (in nm) - for Halogen/Laser only
        self.tolerances = {
            'peak_top': 2.0,
            'trough_bottom': 2.0,
            'valley_midpoint': 5.0,
            'trough_start_end': 5.0,
            'mound_plateau_start': 7.0,
            'mound_plateau_top': 5.0,
            'mound_plateau_end': 7.0
        }
        
        self.tolerance_penalty_per_nm = 5.0
        self.max_tolerance_penalty = 20.0
        
        # FIXED: UV-specific parameters for 0-100 scale
        self.uv_reference_wavelength = 811.0
        self.uv_reference_expected_intensity = 15.0  # FIXED: Expected 811nm intensity in 0-100 scale
        
        # FIXED: Real UV peak standards adjusted for 0-100 scale
        self.uv_real_peak_standards = [296.7, 302.1, 415.6, 419.6, 922.7]
        self.uv_minimum_real_peak_intensity = 2.0  # FIXED: Minimum intensity for real peaks in 0-100 scale
        
        # Diagnostic UV peaks
        self.uv_diagnostic_peaks = {
            507.0: "Diamond ID (natural=absorb, synthetic=transmit)",
            302.0: "Corundum natural vs synthetic"
        }
        
        # FIXED: Normalization scheme validation
        self.expected_normalization_schemes = {
            'UV': 'UV_811nm_15000_to_100',
            'Halogen': 'Halogen_650nm_50000_to_100', 
            'Laser': 'Laser_max_50000_to_100'
        }
    
    def match_features_by_light_source(self, unknown_features: List[Dict], 
                                     db_features: List[Dict], light_source: str,
                                     unknown_gem_id: str, db_gem_id: str) -> float:
        """Route to appropriate matching algorithm with normalization validation"""
        
        # Perfect match detection
        if unknown_gem_id == db_gem_id:
            return 100.0
        
        if not unknown_features or not db_features:
            return 0.0
        
        # FIXED: Validate normalization compatibility
        unknown_norm = self.extract_normalization_scheme(unknown_features)
        db_norm = self.extract_normalization_scheme(db_features)
        
        if unknown_norm and db_norm and unknown_norm != db_norm:
            print(f"      WARNING: Normalization mismatch - Unknown: {unknown_norm}, DB: {db_norm}")
        
        if light_source.upper() == 'UV':
            return self.match_uv_intensity_ratios_fixed(unknown_features, db_features, 
                                                       unknown_gem_id, db_gem_id)
        else:
            return self.match_halogen_laser_wavelengths(unknown_features, db_features,
                                                       unknown_gem_id, db_gem_id)
    
    def extract_normalization_scheme(self, features: List[Dict]) -> Optional[str]:
        """Extract normalization scheme from feature metadata"""
        for feature in features:
            if 'Normalization_Scheme' in feature:
                return feature['Normalization_Scheme']
            elif 'normalization_scheme' in feature:
                return feature['normalization_scheme']
        return None
    
    def match_uv_intensity_ratios_fixed(self, unknown_features: List[Dict], 
                                       db_features: List[Dict], unknown_gem_id: str, 
                                       db_gem_id: str) -> float:
        """FIXED: UV matching for 0-100 normalized data"""
        
        print(f"\n   UV RATIO ANALYSIS (FIXED 0-100): {unknown_gem_id} vs {db_gem_id}")
        
        # FIXED: Calculate ratios with 0-100 scale expectations
        unknown_ratios = self.calculate_uv_ratios_fixed(unknown_features, "Unknown")
        db_ratios = self.calculate_uv_ratios_fixed(db_features, "Database")
        
        if not unknown_ratios or not db_ratios:
            print(f"      Cannot calculate UV ratios (missing 811nm reference or no valid peaks)")
            return 0.0
        
        # Get all unique wavelengths
        all_wavelengths = set(unknown_ratios.keys()) | set(db_ratios.keys())
        
        total_score = 100.0
        penalties = 0.0
        matched_peaks = 0
        
        print(f"      Peak ratio comparison (0-100 scale):")
        
        for wavelength in sorted(all_wavelengths):
            unknown_ratio = unknown_ratios.get(wavelength, None)
            db_ratio = db_ratios.get(wavelength, None)
            
            if unknown_ratio is None:
                penalty = self.uv_missing_peak_penalty
                penalties += penalty
                print(f"         {wavelength:.0f}nm: Missing in unknown (-{penalty}%)")
                
                if wavelength in self.uv_diagnostic_peaks:
                    print(f"            DIAGNOSTIC: {self.uv_diagnostic_peaks[wavelength]}")
            
            elif db_ratio is None:
                penalty = self.uv_missing_peak_penalty
                penalties += penalty
                print(f"         {wavelength:.0f}nm: Missing in database (-{penalty}%)")
            
            else:
                # Both peaks present - compare ratios
                ratio_diff = abs(unknown_ratio - db_ratio)
                
                # FIXED: Ratio scoring adjusted for 0-100 scale
                ratio_score = 100.0 * np.exp(-3.0 * ratio_diff)  # Slightly less strict
                
                print(f"         {wavelength:.0f}nm: {unknown_ratio:.3f} vs {db_ratio:.3f} "
                      f"(Δ{ratio_diff:.3f} → {ratio_score:.1f}%)")
                
                if wavelength in self.uv_diagnostic_peaks:
                    print(f"            DIAGNOSTIC: {self.uv_diagnostic_peaks[wavelength]}")
                
                total_score += ratio_score
                matched_peaks += 1
        
        # Calculate final score
        if matched_peaks > 0:
            average_ratio_score = total_score / (matched_peaks + 1)
        else:
            average_ratio_score = 100.0
        
        final_score = max(0.0, average_ratio_score - penalties)
        
        print(f"      UV Results (FIXED):")
        print(f"         Matched peaks: {matched_peaks}")
        print(f"         Missing peak penalties: -{penalties:.1f}%")
        print(f"         Average ratio score: {average_ratio_score:.1f}%")
        print(f"         Final UV score: {final_score:.1f}%")
        
        return min(100.0, final_score)
    
    def calculate_uv_ratios_fixed(self, features: List[Dict], dataset_name: str) -> Dict[float, float]:
        """FIXED: Calculate ratios for 0-100 normalized data"""
        
        # Find 811nm reference peak
        reference_intensity = None
        peak_data = {}
        
        # Extract peaks with wavelengths and intensities
        for feature in features:
            wavelength = self.extract_wavelength(feature, 'UV')
            intensity = feature.get('intensity', feature.get('Intensity', 0.0))
            
            if wavelength is not None and intensity > 0:
                peak_data[wavelength] = intensity
                
                # Check for 811nm reference
                if abs(wavelength - self.uv_reference_wavelength) <= 1.0:
                    reference_intensity = intensity
        
        if reference_intensity is None or reference_intensity <= 0:
            print(f"      {dataset_name}: No 811nm reference peak found")
            return {}
        
        # FIXED: Validate 811nm reference intensity for 0-100 scale
        if reference_intensity < 5.0:  # Should be around 15 for UV normalization
            print(f"      WARNING: {dataset_name} 811nm intensity ({reference_intensity:.2f}) "
                  f"seems low for 0-100 scale")
        
        # FIXED: Determine real peak threshold for 0-100 scale
        real_peak_threshold = self.determine_uv_real_peak_threshold_fixed(peak_data, dataset_name)
        
        # Filter peaks
        filtered_peaks = {}
        filtered_count = 0
        total_peaks = len(peak_data)
        
        for wavelength, intensity in peak_data.items():
            if intensity >= real_peak_threshold:
                filtered_peaks[wavelength] = intensity
            else:
                filtered_count += 1
        
        print(f"      {dataset_name}: Filtered {filtered_count}/{total_peaks} minor peaks")
        print(f"         Real peak threshold: {real_peak_threshold:.2f} (0-100 scale)")
        print(f"         Kept {len(filtered_peaks)} real peaks")
        
        # Calculate ratios relative to 811nm
        ratios = {}
        for wavelength, intensity in filtered_peaks.items():
            ratio = intensity / reference_intensity
            ratios[wavelength] = ratio
        
        print(f"      {dataset_name} UV ratios (811nm = {reference_intensity:.2f}):")
        for wl in sorted(ratios.keys()):
            print(f"         {wl:.1f}nm: {ratios[wl]:.3f}")
        
        return ratios
    
    def determine_uv_real_peak_threshold_fixed(self, peak_data: Dict[float, float], 
                                              dataset_name: str) -> float:
        """FIXED: Determine threshold for 0-100 normalized data"""
        
        standard_intensities = []
        tolerance = 2.0
        
        # Look for known real peak standards
        for standard_wl in self.uv_real_peak_standards:
            closest_wl = None
            closest_diff = float('inf')
            
            for wl in peak_data.keys():
                diff = abs(wl - standard_wl)
                if diff <= tolerance and diff < closest_diff:
                    closest_diff = diff
                    closest_wl = wl
            
            if closest_wl is not None:
                intensity = peak_data[closest_wl]
                standard_intensities.append(intensity)
                print(f"      {dataset_name}: Found standard {standard_wl:.1f}nm → "
                      f"{closest_wl:.1f}nm (I={intensity:.2f})")
        
        if standard_intensities:
            # Use minimum of found standards
            threshold = min(standard_intensities) * 0.8  # 80% of minimum
            threshold = max(threshold, self.uv_minimum_real_peak_intensity)
            print(f"      {dataset_name}: Threshold from {len(standard_intensities)} standards")
        else:
            # FIXED: Fallback statistical approach for 0-100 scale
            if peak_data:
                intensities = list(peak_data.values())
                mean_intensity = np.mean(intensities)
                std_intensity = np.std(intensities)
                
                # More conservative threshold for 0-100 scale
                threshold = max(
                    mean_intensity - std_intensity,
                    np.max(intensities) * 0.05,  # 5% of max
                    self.uv_minimum_real_peak_intensity
                )
                print(f"      {dataset_name}: Statistical threshold (mean={mean_intensity:.2f}, std={std_intensity:.2f})")
            else:
                threshold = self.uv_minimum_real_peak_intensity
                print(f"      {dataset_name}: Using minimum threshold")
        
        return max(threshold, self.uv_minimum_real_peak_intensity)
    
    def match_halogen_laser_wavelengths(self, unknown_features: List[Dict], 
                                      db_features: List[Dict], unknown_gem_id: str, 
                                      db_gem_id: str) -> float:
        """Halogen/Laser wavelength-based matching (unchanged)"""
        
        if not unknown_features or not db_features:
            return 0.0
        
        # Analyze feature types
        unknown_types = set(f.get('feature_type', 'unknown') for f in unknown_features)
        db_types = set(f.get('feature_type', 'unknown') for f in db_features)
        
        common_types = unknown_types.intersection(db_types)
        missing_types = unknown_types - db_types
        extra_types = db_types - unknown_types
        
        # Apply feature type penalties
        feature_penalty = 0.0
        if missing_types:
            feature_penalty += len(missing_types) * self.missing_feature_penalty
        if extra_types:
            feature_penalty += len(extra_types) * self.extra_feature_penalty
        
        if not common_types:
            return max(0.0, 100.0 - feature_penalty)
        
        # Match features by type
        type_scores = []
        show_details = False
        
        # First pass: calculate scores without detailed output
        for feature_type in common_types:
            unknown_of_type = [f for f in unknown_features if f.get('feature_type') == feature_type]
            db_of_type = [f for f in db_features if f.get('feature_type') == feature_type]
            
            type_score = self.match_features_of_type_simple(unknown_of_type, db_of_type, feature_type)
            type_scores.append(type_score)
        
        # Calculate preliminary score
        if type_scores:
            base_score = sum(type_scores) / len(type_scores)
        else:
            base_score = 0.0
        
        preliminary_score = max(0.0, base_score - feature_penalty)
        
        # If promising match, recalculate with detailed output
        if preliminary_score >= 30.0:
            print(f"\n   H/L Wavelength analysis: {unknown_gem_id} vs {db_gem_id}")
            if missing_types:
                print(f"      Missing features: {missing_types} (-{len(missing_types)*self.missing_feature_penalty}%)")
            if extra_types:
                print(f"      Extra features: {extra_types} (-{len(extra_types)*self.extra_feature_penalty}%)")
            
            # Recalculate with detailed output
            type_scores = []
            for feature_type in common_types:
                unknown_of_type = [f for f in unknown_features if f.get('feature_type') == feature_type]
                db_of_type = [f for f in db_features if f.get('feature_type') == feature_type]
                
                type_score = self.match_features_of_type(unknown_of_type, db_of_type, feature_type)
                type_scores.append(type_score)
            
            if type_scores:
                base_score = sum(type_scores) / len(type_scores)
            
            final_score = max(0.0, base_score - feature_penalty)
            
            print(f"      Base spectral score: {base_score:.1f}%")
            print(f"      Final H/L score: {final_score:.1f}%")
        else:
            final_score = preliminary_score
        
        return min(100.0, final_score)
    
    def calculate_wavelength_score(self, unknown_wl: float, db_wl: float, 
                                 feature_type: str, feature_position: str = 'center') -> float:
        """Calculate score based on wavelength difference and tolerance rules"""
        diff = abs(unknown_wl - db_wl)
        
        # Determine tolerance
        if feature_type.lower() == 'peak':
            if feature_position == 'top':
                tolerance = self.tolerances['peak_top']
            else:
                tolerance = self.tolerances['peak_top']
        
        elif feature_type.lower() == 'trough':
            if feature_position == 'bottom':
                tolerance = self.tolerances['trough_bottom']
            elif feature_position in ['start', 'end']:
                tolerance = self.tolerances['trough_start_end']
            else:
                tolerance = self.tolerances['trough_bottom']
        
        elif feature_type.lower() == 'valley':
            tolerance = self.tolerances['valley_midpoint']
        
        elif feature_type.lower() in ['mound', 'plateau']:
            if feature_position == 'start':
                tolerance = self.tolerances['mound_plateau_start']
            elif feature_position == 'top':
                tolerance = self.tolerances['mound_plateau_top']
            elif feature_position == 'end':
                tolerance = self.tolerances['mound_plateau_end']
            else:
                tolerance = self.tolerances['mound_plateau_top']
        
        else:
            tolerance = self.tolerances['valley_midpoint']
        
        # Calculate score
        if diff <= tolerance:
            score = 100.0 - (diff / tolerance) * 5.0
        else:
            excess_nm = diff - tolerance
            tolerance_units_out = int(excess_nm // tolerance)
            
            penalty = tolerance_units_out * self.tolerance_penalty_per_nm
            penalty = min(penalty, self.max_tolerance_penalty)
            
            score = max(0.0, 95.0 - penalty)
        
        return max(0.0, score)
    
    def calculate_wavelength_score_detailed(self, unknown_wl: float, db_wl: float, 
                                          feature_type: str, feature_position: str = 'center') -> Tuple[float, Dict]:
        """Calculate score with detailed breakdown"""
        diff = abs(unknown_wl - db_wl)
        
        # Determine tolerance
        if feature_type.lower() == 'peak':
            tolerance = self.tolerances['peak_top']
        elif feature_type.lower() == 'trough':
            tolerance = self.tolerances['trough_bottom']
        elif feature_type.lower() == 'valley':
            tolerance = self.tolerances['valley_midpoint']
        elif feature_type.lower() in ['mound', 'plateau']:
            tolerance = self.tolerances['mound_plateau_top']
        else:
            tolerance = self.tolerances['valley_midpoint']
        
        breakdown = {
            'tolerance': tolerance,
            'difference': diff,
            'excess': 0.0,
            'units_out': 0,
            'penalty': 0.0
        }
        
        if diff <= tolerance:
            score = 100.0 - (diff / tolerance) * 5.0
        else:
            excess_nm = diff - tolerance
            tolerance_units_out = int(excess_nm // tolerance)
            penalty = tolerance_units_out * self.tolerance_penalty_per_nm
            penalty = min(penalty, self.max_tolerance_penalty)
            
            breakdown['excess'] = excess_nm
            breakdown['units_out'] = tolerance_units_out
            breakdown['penalty'] = penalty
            
            score = max(0.0, 95.0 - penalty)
        
        return max(0.0, score), breakdown
    
    def match_features(self, unknown_features: List[Dict], db_features: List[Dict], 
                      light_source: str, unknown_gem_id: str, db_gem_id: str) -> float:
        """Main feature matching router"""
        return self.match_features_by_light_source(
            unknown_features, db_features, light_source, unknown_gem_id, db_gem_id
        )
    
    def match_features_of_type(self, unknown_features: List[Dict], 
                              db_features: List[Dict], feature_type: str) -> float:
        """Match features with detailed scoring breakdown"""
        if not unknown_features or not db_features:
            return 0.0
        
        total_score = 0.0
        matched_count = 0
        
        print(f"      Matching {feature_type} features:")
        
        for i, unknown_feature in enumerate(unknown_features):
            best_score = 0.0
            best_match_info = None
            
            unknown_wl = self.extract_wavelength(unknown_feature, feature_type)
            if unknown_wl is None:
                continue
            
            for db_feature in db_features:
                db_wl = self.extract_wavelength(db_feature, feature_type)
                if db_wl is None:
                    continue
                
                score, breakdown = self.calculate_wavelength_score_detailed(
                    unknown_wl, db_wl, feature_type
                )
                
                if score > best_score:
                    best_score = score
                    best_match_info = {
                        'db_wl': db_wl,
                        'diff': abs(unknown_wl - db_wl),
                        'breakdown': breakdown
                    }
            
            if best_match_info:
                print(f"         {i+1}. {unknown_wl:.1f}nm → {best_match_info['db_wl']:.1f}nm "
                      f"(Δ{best_match_info['diff']:.1f}nm, {best_score:.1f}%)")
                
                if best_match_info['breakdown']['penalty'] > 0:
                    b = best_match_info['breakdown']
                    print(f"            Tolerance: {b['tolerance']:.1f}nm, "
                          f"Excess: {b['excess']:.1f}nm, "
                          f"Units out: {b['units_out']}, "
                          f"Penalty: -{b['penalty']:.1f}%")
            
            total_score += best_score
            matched_count += 1
        
        avg_score = total_score / matched_count if matched_count > 0 else 0.0
        print(f"      {feature_type} average score: {avg_score:.1f}%")
        return avg_score
    
    def match_features_of_type_simple(self, unknown_features: List[Dict], 
                                    db_features: List[Dict], feature_type: str) -> float:
        """Match features without detailed output"""
        if not unknown_features or not db_features:
            return 0.0
        
        total_score = 0.0
        matched_count = 0
        
        for unknown_feature in unknown_features:
            best_score = 0.0
            
            unknown_wl = self.extract_wavelength(unknown_feature, feature_type)
            if unknown_wl is None:
                continue
            
            for db_feature in db_features:
                db_wl = self.extract_wavelength(db_feature, feature_type)
                if db_wl is None:
                    continue
                
                score = self.calculate_wavelength_score(unknown_wl, db_wl, feature_type)
                best_score = max(best_score, score)
            
            total_score += best_score
            matched_count += 1
        
        return total_score / matched_count if matched_count > 0 else 0.0
    
    def extract_wavelength(self, feature: Dict, feature_type: str) -> Optional[float]:
        """Extract appropriate wavelength from feature"""
        # FIXED: Handle new CSV format with normalization metadata
        if feature_type == 'UV' or feature_type.upper() == 'UV':
            uv_fields = ['Wavelength_nm', 'wavelength', 'Wavelength']
        else:
            uv_fields = []
        
        wavelength_fields = uv_fields + [
            'wavelength',
            'crest_wavelength', 
            'max_wavelength',
            'midpoint_wavelength',
            'peak_wavelength',
            'Wavelength',
            'Crest',
            'Midpoint'
        ]
        
        for field in wavelength_fields:
            if field in feature and feature[field] is not None:
                try:
                    return float(feature[field])
                except (ValueError, TypeError):
                    continue
        
        return None

class EnhancedGemAnalyzer:
    """FIXED: Enhanced analyzer compatible with 0-100 normalization"""
    
    def __init__(self, db_path="multi_structural_gem_data.db"):
        self.db_path = db_path
        self.unknown_path = Path(r"C:\users\david\gemini sp10 structural data\unknown")
        self.naming_system = GemNamingSystem()
        self.matcher = SpectralMatcher()
        
    def analyze_unknown_file(self, file_path: Path) -> Dict:
        """Analyze unknown file with enhanced matching and normalization validation"""
        file_info = self.naming_system.parse_gem_filename(file_path.name)
        
        print(f"\nAnalyzing: {file_info['original_filename']}")
        print(f"   Gem ID: {file_info['gem_id']}")
        print(f"   Light source: {file_info['light_source']}")
        print(f"   Orientation: {file_info['orientation']}")
        print(f"   Scan #: {file_info['scan_number']}")
        
        # Load and validate unknown data
        unknown_data = self.load_unknown_data_fixed(file_path)
        if not unknown_data:
            print("   Could not load file data")
            return {'error': 'Could not load data'}
        
        print(f"   Found {len(unknown_data)} spectral features")
        
        # FIXED: Validate normalization scheme
        norm_scheme = self.validate_normalization_scheme(unknown_data, file_info['light_source'])
        if norm_scheme:
            print(f"   Normalization: {norm_scheme}")
        
        # Get wavelength range
        wavelengths = [self.matcher.extract_wavelength(f, f.get('feature_type', '')) 
                      for f in unknown_data]
        wavelengths = [w for w in wavelengths if w is not None]
        
        if wavelengths:
            print(f"   Wavelength range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
        
        # FIXED: Validate intensity range for 0-100 scale
        self.validate_intensity_range(unknown_data, file_info['light_source'])
        
        # Find matches
        matches = self.find_database_matches(unknown_data, file_info)
        
        if not matches:
            print("   No similar gems found in database")
            return {'matches': []}
        
        print(f"   Found {len(matches)} potential matches:")
        
        for i, match in enumerate(matches[:5], 1):
            print(f"      {i}. {match['db_gem_id']} - {match['score']:.1f}% "
                  f"({match['db_features']} features, {match['light_source']})")
        
        return {
            'file_info': file_info,
            'unknown_data': unknown_data,
            'matches': matches,
            'normalization_scheme': norm_scheme
        }
    
    def validate_normalization_scheme(self, features: List[Dict], light_source: str) -> Optional[str]:
        """FIXED: Validate that normalization scheme matches expected format"""
        scheme = self.matcher.extract_normalization_scheme(features)
        
        if not scheme:
            print("   WARNING: No normalization scheme found in data")
            return None
        
        expected = self.matcher.expected_normalization_schemes.get(light_source)
        
        if expected and scheme != expected:
            print(f"   WARNING: Unexpected normalization - Found: {scheme}, Expected: {expected}")
        else:
            print(f"   Normalization validated: {scheme}")
        
        return scheme
    
    def validate_intensity_range(self, features: List[Dict], light_source: str):
        """FIXED: Validate intensity values are in expected 0-100 range"""
        intensities = []
        
        for feature in features:
            intensity = feature.get('intensity', feature.get('Intensity', None))
            if intensity is not None:
                try:
                    intensities.append(float(intensity))
                except (ValueError, TypeError):
                    continue
        
        if not intensities:
            print("   WARNING: No intensity values found")
            return
        
        min_int = min(intensities)
        max_int = max(intensities)
        
        print(f"   Intensity range: {min_int:.2f} - {max_int:.2f}")
        
        # Validate for 0-100 scale
        if max_int <= 1.0:
            print("   ERROR: Intensities appear to be 0-1 normalized (broken for UV analysis)")
        elif max_int > 100.0:
            print("   WARNING: Intensities exceed 100 (unexpected for fixed normalization)")
        elif min_int < 0:
            print("   WARNING: Negative intensities found")
        else:
            print("   Intensity range validated for 0-100 scale")
        
        # UV-specific validation
        if light_source == 'UV':
            ref_intensity = None
            for feature in features:
                wl = self.matcher.extract_wavelength(feature, 'UV')
                if wl and abs(wl - 811.0) <= 1.0:
                    intensity = feature.get('intensity', feature.get('Intensity', 0))
                    if intensity > 0:
                        ref_intensity = intensity
                        break
            
            if ref_intensity:
                if ref_intensity < 10.0:
                    print(f"   WARNING: 811nm reference ({ref_intensity:.2f}) seems low for UV normalization")
                else:
                    print(f"   811nm reference validated: {ref_intensity:.2f}")
            else:
                print("   WARNING: No 811nm reference peak found for UV analysis")
    
    def find_database_matches(self, unknown_data: List[Dict], 
                            file_info: Dict, top_n: int = 10) -> List[Dict]:
        """Find matching gems with normalization compatibility checking"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # FIXED: Include normalization metadata in query
            query = """
                SELECT file, light_source, wavelength, intensity, 
                       feature_group, data_type, start_wavelength, 
                       end_wavelength, midpoint, bottom,
                       normalization_scheme, reference_wavelength
                FROM structural_features 
                WHERE light_source = ?
                ORDER BY file, wavelength
            """
            
            db_df = pd.read_sql_query(query, conn, params=(file_info['light_source'],))
            conn.close()
            
            if db_df.empty:
                return []
            
            # Group by gem and calculate matches
            matches = []
            unknown_norm = self.matcher.extract_normalization_scheme(unknown_data)
            
            for db_file in db_df['file'].unique():
                file_data = db_df[db_df['file'] == db_file]
                
                db_file_info = self.naming_system.parse_gem_filename(db_file)
                
                # Convert database features
                db_features = []
                for _, row in file_data.iterrows():
                    feature = {
                        'feature_type': row.get('feature_group', 'unknown'),
                        'wavelength': row['wavelength'],
                        'intensity': row['intensity'],
                        'midpoint_wavelength': row.get('midpoint'),
                        'start_wavelength': row.get('start_wavelength'),
                        'end_wavelength': row.get('end_wavelength'),
                        'crest_wavelength': row['wavelength'],
                        'max_wavelength': row['wavelength']
                    }
                    
                    # FIXED: Include normalization metadata
                    if 'normalization_scheme' in row and pd.notna(row['normalization_scheme']):
                        feature['Normalization_Scheme'] = row['normalization_scheme']
                    if 'reference_wavelength' in row and pd.notna(row['reference_wavelength']):
                        feature['Reference_Wavelength'] = row['reference_wavelength']
                    
                    db_features.append(feature)
                
                # Calculate match score
                score = self.matcher.match_features(
                    unknown_data, db_features, file_info['light_source'],
                    file_info['gem_id'], db_file_info['gem_id']
                )
                
                # FIXED: Apply normalization compatibility bonus/penalty
                db_norm = self.matcher.extract_normalization_scheme(db_features)
                if unknown_norm and db_norm:
                    if unknown_norm == db_norm:
                        score += 2.0  # Small bonus for compatible normalization
                    else:
                        score -= 5.0  # Penalty for incompatible normalization
                
                if score > 0:
                    matches.append({
                        'db_gem_id': db_file_info['gem_id'],
                        'db_full_id': db_file_info['full_identifier'],
                        'score': score,
                        'db_features': len(db_features),
                        'light_source': db_file_info['light_source'],
                        'orientation': db_file_info['orientation'],
                        'scan_number': db_file_info['scan_number'],
                        'normalization_scheme': db_norm,
                        'normalization_compatible': unknown_norm == db_norm if unknown_norm and db_norm else None
                    })
            
            # Sort by score
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[:top_n]
            
        except Exception as e:
            print(f"Error finding matches: {e}")
            return []
    
    def load_unknown_data_fixed(self, file_path: Path) -> List[Dict]:
        """FIXED: Load unknown data with normalization metadata handling"""
        try:
            df = pd.read_csv(file_path)
            
            # FIXED: Detect format and handle normalization metadata
            if 'Peak_Number' in df.columns:
                # Peak detection format (from fixed peak detector)
                features = []
                for _, row in df.iterrows():
                    feature = {
                        'feature_type': 'Peak',
                        'wavelength': row['Wavelength_nm'],
                        'max_wavelength': row['Wavelength_nm'],
                        'intensity': row['Intensity'],
                        'prominence': row.get('Prominence', 1.0)
                    }
                    
                    # FIXED: Include normalization metadata if available
                    if 'Normalization_Scheme' in row and pd.notna(row['Normalization_Scheme']):
                        feature['Normalization_Scheme'] = row['Normalization_Scheme']
                    if 'Reference_Wavelength' in row and pd.notna(row['Reference_Wavelength']):
                        feature['Reference_Wavelength'] = row['Reference_Wavelength']
                    if 'Light_Source' in row and pd.notna(row['Light_Source']):
                        feature['Light_Source'] = row['Light_Source']
                    
                    features.append(feature)
                return features
            
            elif 'Feature' in df.columns:
                # Structural features format
                features = []
                for _, row in df.iterrows():
                    feature = {
                        'feature_type': row.get('Feature', 'unknown'),
                        'wavelength': row.get('Wavelength', row.get('Crest')),
                        'intensity': row.get('Intensity', 1.0)
                    }
                    
                    # Add additional fields
                    if 'Crest' in row:
                        feature['crest_wavelength'] = row['Crest']
                    if 'Midpoint' in row:
                        feature['midpoint_wavelength'] = row['Midpoint']
                    if 'Start' in row:
                        feature['start_wavelength'] = row['Start']
                    if 'End' in row:
                        feature['end_wavelength'] = row['End']
                    
                    # FIXED: Handle normalization metadata
                    if 'Normalization_Scheme' in row and pd.notna(row['Normalization_Scheme']):
                        feature['Normalization_Scheme'] = row['Normalization_Scheme']
                    
                    features.append(feature)
                return features
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
        
        return []
    
    def analyze_all_unknowns(self):
        """Analyze all unknown files with enhanced normalization validation"""
        if not os.path.exists(self.db_path):
            print(f"Database not found: {self.db_path}")
            return
        
        if not self.unknown_path.exists():
            print(f"Unknown directory not found: {self.unknown_path}")
            return
        
        csv_files = list(self.unknown_path.glob("*.csv"))
        
        if not csv_files:
            print("No unknown files found")
            return
        
        print(f"Found {len(csv_files)} unknown files to analyze")
        
        all_results = {}
        normalization_issues = []
        
        for file_path in csv_files:
            try:
                result = self.analyze_unknown_file(file_path)
                all_results[file_path.name] = result
                
                # FIXED: Track normalization issues
                if 'normalization_scheme' in result:
                    scheme = result['normalization_scheme']
                    if not scheme or 'Unknown' in scheme:
                        normalization_issues.append(file_path.name)
            except Exception as e:
                print(f"Error analyzing {file_path.name}: {e}")
        
        # FIXED: Enhanced summary with normalization validation
        print(f"\nENHANCED ANALYSIS SUMMARY (FIXED for 0-100 normalization):")
        print("=" * 70)
        
        if normalization_issues:
            print(f"NORMALIZATION WARNINGS:")
            for issue_file in normalization_issues:
                print(f"   ! {issue_file}: Missing or unknown normalization scheme")
            print()
        
        for filename, result in all_results.items():
            if 'error' in result:
                print(f"Analyzing {filename}")
                print(f"   {result['error']}")
            elif result.get('matches'):
                matches = result['matches']
                best_match = matches[0]
                file_info = result['file_info']
                
                print(f"Analyzing {filename}")
                print(f"   Gem: {file_info['gem_id']} ({file_info['light_source']})")
                
                # FIXED: Show normalization compatibility
                norm_scheme = result.get('normalization_scheme', 'Unknown')
                best_norm_compat = best_match.get('normalization_compatible')
                
                if best_norm_compat is True:
                    compat_indicator = "COMPATIBLE"
                elif best_norm_compat is False:
                    compat_indicator = "INCOMPATIBLE"
                else:
                    compat_indicator = "UNKNOWN"
                
                print(f"   Best match: {best_match['db_gem_id']} ({best_match['score']:.1f}%) [{compat_indicator}]")
                print(f"   Normalization: {norm_scheme}")
                
                # Enhanced confidence interpretation
                if best_match['score'] == 100.0:
                    print(f"   PERFECT MATCH - Same gem, different scan!")
                elif best_match['score'] >= 80.0:
                    print(f"   EXCELLENT - Very likely same species")
                elif best_match['score'] >= 60.0:
                    print(f"   GOOD - Possibly same variety")
                else:
                    print(f"   MODERATE - Similar characteristics")
            else:
                print(f"Analyzing {filename}")
                print(f"   No matches found")
        
        return all_results
    
    def deduplicate_results(self, matches: List[Dict], mode: str = 'show_all') -> List[Dict]:
        """Smart results deduplication while preserving treatment history"""
        
        if mode == 'show_all':
            return matches
        
        elif mode == 'best_per_gem':
            print(f"   Deduplicating results (best per gem)...")
            
            gem_groups = {}
            for match in matches:
                gem_id = match['db_gem_id']
                if gem_id not in gem_groups or match['score'] > gem_groups[gem_id]['score']:
                    gem_groups[gem_id] = match
            
            deduplicated = list(gem_groups.values())
            print(f"   Reduced from {len(matches)} to {len(deduplicated)} unique gems")
            return deduplicated
        
        elif mode == 'treatment_analysis':
            print(f"   Analyzing treatment progression...")
            
            gem_groups = {}
            for match in matches:
                gem_id = match['db_gem_id']
                if gem_id not in gem_groups:
                    gem_groups[gem_id] = []
                gem_groups[gem_id].append(match)
            
            treatment_results = []
            for gem_id, gem_matches in gem_groups.items():
                if len(gem_matches) == 1:
                    treatment_results.append(gem_matches[0])
                else:
                    gem_matches.sort(key=lambda x: x['score'], reverse=True)
                    best_match = gem_matches[0]
                    
                    enhanced_match = best_match.copy()
                    enhanced_match['treatment_variants'] = [
                        {'scan': m['db_full_id'], 'score': m['score']} 
                        for m in gem_matches
                    ]
                    enhanced_match['treatment_range'] = f"{gem_matches[-1]['score']:.1f}%-{gem_matches[0]['score']:.1f}%"
                    treatment_results.append(enhanced_match)
            
            return treatment_results
        
        else:
            return matches
    
    def display_results_with_deduplication(self, results: Dict, show_mode: str = 'show_all'):
        """Display results with smart deduplication and normalization info"""
        
        print(f"\nENHANCED ANALYSIS SUMMARY (FIXED):")
        print("=" * 70)
        
        for filename, result in results.items():
            if 'error' in result:
                print(f"Analyzing {filename}")
                print(f"   {result['error']}")
            elif result.get('matches'):
                file_info = result['file_info']
                matches = result['matches']
                
                # Apply deduplication
                if show_mode != 'show_all':
                    matches = self.deduplicate_results(matches, show_mode)
                
                print(f"Analyzing {filename}")
                print(f"   Gem: {file_info['gem_id']} ({file_info['light_source']})")
                
                if matches:
                    best_match = matches[0]
                    
                    # FIXED: Show normalization compatibility
                    norm_compat = best_match.get('normalization_compatible')
                    if norm_compat is True:
                        compat_str = "COMPATIBLE normalization"
                    elif norm_compat is False:
                        compat_str = "INCOMPATIBLE normalization"
                    else:
                        compat_str = "Unknown normalization compatibility"
                    
                    if show_mode == 'treatment_analysis' and 'treatment_variants' in best_match:
                        print(f"   Best match: {best_match['db_gem_id']} ({best_match['treatment_range']})")
                        print(f"   Treatment variants:")
                        for variant in best_match['treatment_variants'][:3]:
                            print(f"      • {variant['scan']}: {variant['score']:.1f}%")
                        
                        if len(best_match['treatment_variants']) > 3:
                            print(f"      ... and {len(best_match['treatment_variants'])-3} more scans")
                    
                    elif show_mode == 'best_per_gem':
                        print(f"   Best match: {best_match['db_gem_id']} ({best_match['score']:.1f}%)")
                        if len(result['matches']) > len(matches):
                            filtered_count = len(result['matches']) - len(matches)
                            print(f"   ({filtered_count} duplicate scans filtered)")
                    
                    else:
                        print(f"   Best match: {best_match['db_gem_id']} ({best_match['score']:.1f}%)")
                    
                    print(f"   {compat_str}")
                    
                    # Confidence interpretation
                    score = best_match['score']
                    if score == 100.0:
                        print(f"   PERFECT MATCH - Same gem, different scan!")
                    elif score >= 80.0:
                        print(f"   EXCELLENT - Very likely same species")
                    elif score >= 60.0:
                        print(f"   GOOD - Possibly same variety")
                    else:
                        print(f"   MODERATE - Similar characteristics")
                else:
                    print(f"   No matches found")
            else:
                print(f"Analyzing {filename}")
                print(f"   No matches found")
        
        return results
    
    def analyze_multi_light_integration(self, gem_id: str):
        """FIXED: Multi-light analysis with normalization validation"""
        print(f"\nMULTI-LIGHT INTEGRATION ANALYSIS (FIXED): {gem_id}")
        print("=" * 70)
        
        light_sources = ['Halogen', 'Laser', 'UV']
        gem_scores = {}
        
        # Find unknown files
        unknown_files = {}
        if self.unknown_path.exists():
            for file in self.unknown_path.glob("*.csv"):
                file_info = self.naming_system.parse_gem_filename(file.name)
                if file_info['gem_id'] == gem_id:
                    unknown_files[file_info['light_source']] = file
        
        if not unknown_files:
            print(f"No unknown files found for gem {gem_id}")
            return
        
        print(f"Found unknown files:")
        for light_source, file in unknown_files.items():
            print(f"   {light_source}: {file.name}")
        
        # Analyze each light source
        best_matches = {}
        normalization_summary = {}
        
        for light_source, file_path in unknown_files.items():
            print(f"\nAnalyzing {light_source} data...")
            
            file_info = self.naming_system.parse_gem_filename(file_path.name)
            unknown_data = self.load_unknown_data_fixed(file_path)
            
            if unknown_data:
                # FIXED: Validate normalization
                norm_scheme = self.validate_normalization_scheme(unknown_data, light_source)
                normalization_summary[light_source] = norm_scheme
                
                matches = self.find_database_matches(unknown_data, file_info, top_n=5)
                if matches:
                    deduplicated_matches = self.deduplicate_results(matches, 'best_per_gem')
                    
                    best_match = deduplicated_matches[0] if deduplicated_matches else matches[0]
                    best_score = best_match['score']
                    gem_scores[light_source] = best_score
                    best_matches[light_source] = best_match
                    
                    # Show normalization compatibility
                    norm_compat = best_match.get('normalization_compatible')
                    compat_str = ""
                    if norm_compat is True:
                        compat_str = " [COMPATIBLE]"
                    elif norm_compat is False:
                        compat_str = " [INCOMPATIBLE]"
                    
                    if len(matches) > len(deduplicated_matches):
                        filtered_count = len(matches) - len(deduplicated_matches)
                        print(f"   Best {light_source} match: {best_match['db_gem_id']} ({best_score:.1f}%) "
                              f"[{filtered_count} duplicates filtered]{compat_str}")
                    else:
                        print(f"   Best {light_source} match: {best_match['db_gem_id']} ({best_score:.1f}%){compat_str}")
                else:
                    gem_scores[light_source] = 0.0
                    best_matches[light_source] = None
                    print(f"   No {light_source} matches found")
        
        # Calculate integrated score
        if gem_scores:
            integrated_score = sum(gem_scores.values()) / len(gem_scores)
            
            # Find most consistent match
            gem_vote_counts = {}
            gem_total_scores = {}
            
            for light_source, match in best_matches.items():
                if match:
                    db_gem = match['db_gem_id']
                    score = match['score']
                    
                    if db_gem not in gem_vote_counts:
                        gem_vote_counts[db_gem] = 0
                        gem_total_scores[db_gem] = 0
                    
                    gem_vote_counts[db_gem] += 1
                    gem_total_scores[db_gem] += score
            
            best_overall_gem = None
            best_consistency_score = 0
            
            for db_gem, vote_count in gem_vote_counts.items():
                avg_score = gem_total_scores[db_gem] / vote_count
                consistency_score = vote_count * avg_score
                
                if consistency_score > best_consistency_score:
                    best_consistency_score = consistency_score
                    best_overall_gem = db_gem
            
            # FIXED: Enhanced results display with normalization info
            print(f"\nINTEGRATED ANALYSIS RESULTS (FIXED):")
            print("=" * 50)
            print(f"UNKNOWN GEM: {gem_id}")
            print(f"BEST MATCH: {best_overall_gem if best_overall_gem else 'No clear winner'}")
            
            # Show normalization schemes
            print(f"\nNORMALIZATION VALIDATION:")
            for light_source in ['UV', 'Laser', 'Halogen']:
                if light_source in normalization_summary:
                    scheme = normalization_summary[light_source] or 'Unknown'
                    print(f"   {light_source}: {scheme}")
            print()
            
            for light_source in ['UV', 'Laser', 'Halogen']:
                if light_source in gem_scores:
                    score = gem_scores[light_source]
                    match = best_matches.get(light_source)
                    icons = {'UV': 'UV', 'Laser': 'LASER', 'Halogen': 'HALOGEN'}
                    
                    if match:
                        matched_gem = match['db_gem_id']
                        norm_compat = match.get('normalization_compatible')
                        
                        if norm_compat is True:
                            compat_indicator = " (COMPAT)"
                        elif norm_compat is False:
                            compat_indicator = " (INCOMPAT)"
                        else:
                            compat_indicator = ""
                        
                        if matched_gem == best_overall_gem:
                            print(f"   {icons[light_source]}: {score:.1f}% → {matched_gem}{compat_indicator} MATCH")
                        else:
                            print(f"   {icons[light_source]}: {score:.1f}% → {matched_gem}{compat_indicator}")
                    else:
                        print(f"   {icons[light_source]}: {score:.1f}% → No match")
            
            print(f"\n   INTEGRATED SCORE: {integrated_score:.1f}%")
            
            # Enhanced interpretation
            if best_overall_gem:
                if integrated_score >= 90.0:
                    print(f"   EXCELLENT: {gem_id} is very likely gem {best_overall_gem}")
                elif integrated_score >= 75.0:
                    print(f"   STRONG: {gem_id} is probably gem {best_overall_gem}") 
                elif integrated_score >= 60.0:
                    print(f"   MODERATE: {gem_id} might be similar to gem {best_overall_gem}")
                elif integrated_score >= 40.0:
                    print(f"   WEAK: {gem_id} shows some similarity to gem {best_overall_gem}")
                else:
                    print(f"   POOR: No strong match found")
            else:
                print(f"   INCONCLUSIVE: No consistent match across light sources")
            
            # Show voting summary
            if len(gem_vote_counts) > 1:
                print(f"\nCANDIDATE SUMMARY:")
                sorted_candidates = sorted(gem_vote_counts.items(), 
                                         key=lambda x: gem_total_scores[x[0]]/x[1], reverse=True)
                for candidate_gem, vote_count in sorted_candidates:
                    avg_score = gem_total_scores[candidate_gem] / vote_count
                    light_sources_matched = [ls for ls, match in best_matches.items() 
                                           if match and match['db_gem_id'] == candidate_gem]
                    print(f"   • {candidate_gem}: {avg_score:.1f}% avg ({vote_count}/{len(gem_scores)} sources: {', '.join(light_sources_matched)})")
            
            return {
                'gem_id': gem_id,
                'best_match': best_overall_gem,
                'light_source_scores': gem_scores,
                'light_source_matches': {ls: match['db_gem_id'] if match else None 
                                       for ls, match in best_matches.items()},
                'integrated_score': integrated_score,
                'consistency_score': best_consistency_score,
                'normalization_schemes': normalization_summary
            }
        
        return None

def main_menu():
    """Main menu for FIXED enhanced gem analyzer"""
    print("ENHANCED GEM ANALYZER v2.1 - FIXED for 0-100 Normalization")
    print("Implements David's sophisticated matching algorithms")
    print("=" * 70)
    
    analyzer = EnhancedGemAnalyzer()
    
    while True:
        print(f"\nMAIN MENU:")
        print("1. Analyze Unknown Files (FIXED for 0-100 normalization)")
        print("2. Multi-Light Integration Analysis (FIXED)") 
        print("3. Show Unknown Directory")
        print("4. Clear Unknown Directory")
        print("5. Database Statistics")
        print("6. Show Matching Parameters (FIXED)")
        print("7. Exit")
        
        try:
            choice = input("Choice (1-7): ").strip()
            
            if choice == "7":
                print("Goodbye!")
                break
                
            elif choice == "1":
                print("\nFIXED SPECTRAL ANALYSIS (0-100 normalization compatible)")
                print("=" * 60)
                analyzer.analyze_all_unknowns()
                
            elif choice == "2":
                print("\nFIXED MULTI-LIGHT INTEGRATION ANALYSIS")
                print("=" * 60)
                
                gem_ids = set()
                if analyzer.unknown_path.exists():
                    for file in analyzer.unknown_path.glob("*.csv"):
                        file_info = analyzer.naming_system.parse_gem_filename(file.name)
                        gem_ids.add(file_info['gem_id'])
                
                if gem_ids:
                    print(f"Found gems with unknown data:")
                    gem_list = sorted(list(gem_ids))
                    for i, gem_id in enumerate(gem_list, 1):
                        print(f"   {i}. {gem_id}")
                    
                    try:
                        selection = input(f"\nSelect gem (1-{len(gem_list)}) or 'all': ").strip()
                        if selection.lower() == 'all':
                            for gem_id in gem_list:
                                analyzer.analyze_multi_light_integration(gem_id)
                        else:
                            idx = int(selection) - 1
                            if 0 <= idx < len(gem_list):
                                analyzer.analyze_multi_light_integration(gem_list[idx])
                            else:
                                print("Invalid selection")
                    except ValueError:
                        print("Invalid selection")
                else:
                    print("No unknown gems found")
                
            elif choice == "3":
                print("\nUNKNOWN DIRECTORY CONTENTS")
                print("=" * 60)
                if analyzer.unknown_path.exists():
                    files = list(analyzer.unknown_path.glob("*.csv"))
                    if files:
                        print(f"Found {len(files)} files:")
                        for file in files:
                            info = analyzer.naming_system.parse_gem_filename(file.name)
                            size = file.stat().st_size
                            print(f"   {file.name} ({size} bytes)")
                            print(f"       Gem: {info['gem_id']}, "
                                  f"Light: {info['light_source']}, "
                                  f"Scan: {info['scan_number']}")
                    else:
                        print("No files found")
                else:
                    print("Directory not found")
                    
            elif choice == "4":
                print("\nCLEAR UNKNOWN DIRECTORY")
                print("=" * 60)
                if analyzer.unknown_path.exists():
                    files = list(analyzer.unknown_path.glob("*.csv"))
                    if files:
                        confirm = input(f"Delete {len(files)} files? (y/N): ").strip().lower()
                        if confirm == 'y':
                            deleted = 0
                            for file in files:
                                try:
                                    file.unlink()
                                    deleted += 1
                                except Exception as e:
                                    print(f"Error deleting {file.name}: {e}")
                            print(f"Deleted {deleted} files")
                        else:
                            print("Cancelled")
                    else:
                        print("No files to delete")
                else:
                    print("Directory not found")
                    
            elif choice == "5":
                print("\nDATABASE STATISTICS")
                print("=" * 60)
                try:
                    conn = sqlite3.connect(analyzer.db_path)
                    
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM structural_features")
                    total = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features")
                    unique_files = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source")
                    by_light = cursor.fetchall()
                    
                    # FIXED: Check for normalization metadata
                    cursor.execute("SELECT COUNT(*) FROM structural_features WHERE normalization_scheme IS NOT NULL")
                    with_norm = cursor.fetchone()[0]
                    
                    print(f"Total records: {total:,}")
                    print(f"Unique gem files: {unique_files:,}")
                    print(f"Records with normalization metadata: {with_norm:,}")
                    print(f"By light source:")
                    for light, count in by_light:
                        print(f"   {light}: {count:,}")
                    
                    conn.close()
                except Exception as e:
                    print(f"Database error: {e}")
                    
            elif choice == "6":
                print(f"MATCHING PARAMETERS (FIXED for 0-100 normalization)")
                print("=" * 70)
                print("Wavelength Tolerances (Halogen/Laser):")
                for param, value in analyzer.matcher.tolerances.items():
                    print(f"   {param}: ±{value} nm")
                
                print(f"\nPenalties:")
                print(f"   Missing feature (H/L): -{analyzer.matcher.missing_feature_penalty}%")
                print(f"   Extra feature (H/L): -{analyzer.matcher.extra_feature_penalty}%")
                print(f"   Missing UV peak: -{analyzer.matcher.uv_missing_peak_penalty}%")
                print(f"   Out-of-tolerance: -{analyzer.matcher.tolerance_penalty_per_nm}% per nm")
                print(f"   Maximum tolerance penalty: -{analyzer.matcher.max_tolerance_penalty}%")
                
                print(f"\nFIXED UV Analysis Parameters (0-100 scale):")
                print(f"   Reference wavelength: {analyzer.matcher.uv_reference_wavelength}nm")
                print(f"   Expected 811nm intensity: ~{analyzer.matcher.uv_reference_expected_intensity} (0-100 scale)")
                print(f"   Minimum real peak intensity: {analyzer.matcher.uv_minimum_real_peak_intensity}")
                print(f"   Real peak standards: {', '.join(f'{wl:.1f}nm' for wl in analyzer.matcher.uv_real_peak_standards)}")
                
                print(f"\nExpected Normalization Schemes:")
                for light, scheme in analyzer.matcher.expected_normalization_schemes.items():
                    print(f"   {light}: {scheme}")
                
                print(f"\nNormalization Validation:")
                print(f"   • Checks intensity ranges are 0-100")
                print(f"   • Validates 811nm reference for UV")
                print(f"   • Warns about incompatible schemes")
                print(f"   • Applies compatibility bonuses/penalties")
                
            else:
                print("Invalid choice. Please enter 1-7")
                continue
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        if choice in ['1', '2', '3', '4', '5', '6']:
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    # FIXED: Test UV calculations with 0-100 scale
    print("TESTING FIXED UV RATIO CALCULATIONS")
    print("=" * 60)
    
    matcher = SpectralMatcher()
    
    print("FIXED UV Parameters:")
    print(f"   Expected 811nm intensity: ~{matcher.uv_reference_expected_intensity} (0-100 scale)")
    print(f"   Minimum real peak: {matcher.uv_minimum_real_peak_intensity}")
    print(f"   Real peak standards: {matcher.uv_real_peak_standards}")
    
    print(f"\nExpected normalization schemes:")
    for light, scheme in matcher.expected_normalization_schemes.items():
        print(f"   {light}: {scheme}")
    
    print(f"\nStarting FIXED Enhanced Gem Analyzer...")
    main_menu()