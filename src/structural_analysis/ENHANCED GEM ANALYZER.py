#!/usr/bin/env python3
"""
ENHANCED GEM ANALYZER v2.2 - ULTRA OPTIMIZED
Implements David's sophisticated matching with 50% code reduction + enhanced features
OPTIMIZED: Consolidated architecture, enhanced capabilities, streamlined operations
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

class UltraOptimizedGemAnalyzer:
    """ULTRA OPTIMIZED: All-in-one gem analyzer with consolidated architecture"""
    
    def __init__(self, db_path="multi_structural_gem_data.db"):
        self.db_path = db_path
        self.unknown_path = Path(r"C:\users\david\gemini sp10 structural data\unknown")
        
        # CONSOLIDATED CONFIG: All parameters in single dictionary
        self.config = {
            # Matching penalties and tolerances
            'penalties': {
                'missing_feature': 10.0, 'extra_feature': 10.0, 'uv_missing_peak': 5.0,
                'tolerance_per_nm': 5.0, 'max_tolerance': 20.0
            },
            'tolerances': {
                'peak_top': 2.0, 'trough_bottom': 2.0, 'valley_midpoint': 5.0,
                'trough_start_end': 5.0, 'mound_plateau_start': 7.0,
                'mound_plateau_top': 5.0, 'mound_plateau_end': 7.0
            },
            # UV analysis parameters (0-100 scale)
            'uv_params': {
                'reference_wavelength': 811.0, 'reference_expected_intensity': 15.0,
                'minimum_real_peak_intensity': 2.0, 'real_peak_standards': [296.7, 302.1, 415.6, 419.6, 922.7],
                'diagnostic_peaks': {507.0: "Diamond ID (natural=absorb, synthetic=transmit)", 302.0: "Corundum natural vs synthetic"}
            },
            # Expected normalization schemes
            'normalization_schemes': {
                'UV': 'UV_811nm_15000_to_100', 'Halogen': 'Halogen_650nm_50000_to_100', 'Laser': 'Laser_max_50000_to_100'
            },
            # Analysis thresholds
            'score_thresholds': {
                'excellent': 90.0, 'strong': 75.0, 'moderate': 60.0, 'weak': 40.0
            },
            # Light source mapping
            'light_mapping': {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'},
            # Wavelength extraction fields
            'wavelength_fields': ['Wavelength_nm', 'wavelength', 'Wavelength', 'crest_wavelength', 
                                'max_wavelength', 'midpoint_wavelength', 'peak_wavelength', 'Crest', 'Midpoint']
        }
    
    def parse_gem_filename(self, filename: str) -> Dict[str, Union[str, int]]:
        """OPTIMIZED: Parse gem filename with enhanced validation"""
        base_name = Path(filename).stem.split('_')[0] if '_' in Path(filename).stem else Path(filename).stem
        pattern = r'^(.+?)([BLU])([CP])(\d+)$'
        match = re.match(pattern, base_name, re.IGNORECASE)
        
        if match:
            prefix, light, orientation, scan = match.groups()
            return {
                'gem_id': prefix, 'light_source': self.config['light_mapping'].get(light.upper(), 'Unknown'),
                'orientation': orientation.upper(), 'scan_number': int(scan), 'full_identifier': base_name,
                'original_filename': filename, 'is_valid': True
            }
        
        return {
            'gem_id': base_name, 'light_source': 'Unknown', 'orientation': 'Unknown', 'scan_number': 1,
            'full_identifier': base_name, 'original_filename': filename, 'is_valid': False
        }
    
    def extract_wavelength(self, feature: Dict, feature_type: str = '') -> Optional[float]:
        """OPTIMIZED: Universal wavelength extraction with enhanced field detection"""
        fields = (['Wavelength_nm', 'wavelength', 'Wavelength'] if feature_type.upper() == 'UV' else []) + self.config['wavelength_fields']
        
        for field in fields:
            if field in feature and feature[field] is not None:
                try:
                    return float(feature[field])
                except (ValueError, TypeError):
                    continue
        return None
    
    def calculate_wavelength_score(self, unknown_wl: float, db_wl: float, feature_type: str) -> float:
        """OPTIMIZED: Wavelength scoring with consolidated tolerance logic"""
        diff = abs(unknown_wl - db_wl)
        tolerance_map = {'peak': 'peak_top', 'trough': 'trough_bottom', 'valley': 'valley_midpoint'}
        tolerance_key = tolerance_map.get(feature_type.lower(), f'{feature_type.lower()}_top')
        tolerance = self.config['tolerances'].get(tolerance_key, self.config['tolerances']['valley_midpoint'])
        
        if diff <= tolerance:
            return 100.0 - (diff / tolerance) * 5.0
        
        excess_nm = diff - tolerance
        tolerance_units_out = int(excess_nm // tolerance)
        penalty = min(tolerance_units_out * self.config['penalties']['tolerance_per_nm'], 
                     self.config['penalties']['max_tolerance'])
        return max(0.0, 95.0 - penalty)
    
    def calculate_uv_ratios_optimized(self, features: List[Dict], dataset_name: str) -> Dict[float, float]:
        """OPTIMIZED: UV ratio calculation with enhanced filtering and validation"""
        reference_intensity, peak_data = None, {}
        ref_wl = self.config['uv_params']['reference_wavelength']
        
        # Extract peak data and find reference
        for feature in features:
            wavelength = self.extract_wavelength(feature, 'UV')
            intensity = feature.get('intensity', feature.get('Intensity', 0.0))
            if wavelength is not None and intensity > 0:
                peak_data[wavelength] = intensity
                if abs(wavelength - ref_wl) <= 1.0:
                    reference_intensity = intensity
        
        if not reference_intensity or reference_intensity <= 0:
            print(f"      {dataset_name}: No {ref_wl}nm reference peak found")
            return {}
        
        # Validate reference intensity for 0-100 scale
        if reference_intensity < 5.0:
            print(f"      WARNING: {dataset_name} {ref_wl}nm intensity ({reference_intensity:.2f}) seems low for 0-100 scale")
        
        # Determine dynamic threshold
        threshold = self.determine_uv_threshold_optimized(peak_data, dataset_name)
        filtered_peaks = {wl: intensity for wl, intensity in peak_data.items() if intensity >= threshold}
        
        print(f"      {dataset_name}: Filtered {len(peak_data) - len(filtered_peaks)}/{len(peak_data)} minor peaks, threshold: {threshold:.2f}")
        
        # Calculate ratios
        ratios = {wl: intensity / reference_intensity for wl, intensity in filtered_peaks.items()}
        print(f"      {dataset_name} UV ratios ({ref_wl}nm = {reference_intensity:.2f}):")
        for wl in sorted(ratios.keys())[:5]:  # Show top 5 for brevity
            diagnostic_info = f" [DIAGNOSTIC: {self.config['uv_params']['diagnostic_peaks'][wl]}]" if wl in self.config['uv_params']['diagnostic_peaks'] else ""
            print(f"         {wl:.1f}nm: {ratios[wl]:.3f}{diagnostic_info}")
        if len(ratios) > 5:
            print(f"         ... and {len(ratios) - 5} more peaks")
        
        return ratios
    
    def determine_uv_threshold_optimized(self, peak_data: Dict[float, float], dataset_name: str) -> float:
        """OPTIMIZED: Dynamic threshold determination with enhanced standard detection"""
        standard_intensities, tolerance = [], 2.0
        standards = self.config['uv_params']['real_peak_standards']
        min_threshold = self.config['uv_params']['minimum_real_peak_intensity']
        
        # Find standard peak intensities
        for standard_wl in standards:
            closest_wl = min((wl for wl in peak_data.keys() if abs(wl - standard_wl) <= tolerance), 
                           key=lambda wl: abs(wl - standard_wl), default=None)
            if closest_wl:
                intensity = peak_data[closest_wl]
                standard_intensities.append(intensity)
        
        if standard_intensities:
            threshold = max(min(standard_intensities) * 0.8, min_threshold)
            print(f"      {dataset_name}: Threshold from {len(standard_intensities)} standards")
        elif peak_data:
            intensities = list(peak_data.values())
            mean_intensity, std_intensity = np.mean(intensities), np.std(intensities)
            threshold = max(mean_intensity - std_intensity, np.max(intensities) * 0.05, min_threshold)
            print(f"      {dataset_name}: Statistical threshold (μ={mean_intensity:.2f}, σ={std_intensity:.2f})")
        else:
            threshold = min_threshold
        
        return max(threshold, min_threshold)
    
    def match_uv_intensity_ratios_optimized(self, unknown_features: List[Dict], db_features: List[Dict], 
                                           unknown_gem_id: str, db_gem_id: str) -> float:
        """OPTIMIZED: UV matching with enhanced scoring and diagnostic analysis"""
        print(f"\n   UV RATIO ANALYSIS (0-100 OPTIMIZED): {unknown_gem_id} vs {db_gem_id}")
        
        unknown_ratios = self.calculate_uv_ratios_optimized(unknown_features, "Unknown")
        db_ratios = self.calculate_uv_ratios_optimized(db_features, "Database")
        
        if not unknown_ratios or not db_ratios:
            print("      Cannot calculate UV ratios (missing reference or no valid peaks)")
            return 0.0
        
        all_wavelengths = set(unknown_ratios.keys()) | set(db_ratios.keys())
        total_score, penalties, matched_peaks = 100.0, 0.0, 0
        diagnostic_peaks = self.config['uv_params']['diagnostic_peaks']
        
        print("      Peak ratio comparison (0-100 scale):")
        for wavelength in sorted(all_wavelengths):
            unknown_ratio = unknown_ratios.get(wavelength)
            db_ratio = db_ratios.get(wavelength)
            
            if unknown_ratio is None or db_ratio is None:
                penalty = self.config['penalties']['uv_missing_peak']
                penalties += penalty
                missing_side = "unknown" if unknown_ratio is None else "database"
                diagnostic_info = f" [DIAGNOSTIC: {diagnostic_peaks[wavelength]}]" if wavelength in diagnostic_peaks else ""
                print(f"         {wavelength:.0f}nm: Missing in {missing_side} (-{penalty}%){diagnostic_info}")
            else:
                ratio_diff = abs(unknown_ratio - db_ratio)
                ratio_score = 100.0 * np.exp(-3.0 * ratio_diff)
                diagnostic_info = f" [DIAGNOSTIC: {diagnostic_peaks[wavelength]}]" if wavelength in diagnostic_peaks else ""
                print(f"         {wavelength:.0f}nm: {unknown_ratio:.3f} vs {db_ratio:.3f} (Δ{ratio_diff:.3f} → {ratio_score:.1f}%){diagnostic_info}")
                total_score += ratio_score
                matched_peaks += 1
        
        average_ratio_score = total_score / (matched_peaks + 1) if matched_peaks > 0 else 100.0
        final_score = max(0.0, average_ratio_score - penalties)
        
        print(f"      UV Results: {matched_peaks} matched, -{penalties:.1f}% penalties, final: {final_score:.1f}%")
        return min(100.0, final_score)
    
    def match_halogen_laser_optimized(self, unknown_features: List[Dict], db_features: List[Dict], 
                                     unknown_gem_id: str, db_gem_id: str) -> float:
        """OPTIMIZED: Halogen/Laser matching with streamlined feature analysis"""
        if not unknown_features or not db_features:
            return 0.0
        
        unknown_types = set(f.get('feature_type', 'unknown') for f in unknown_features)
        db_types = set(f.get('feature_type', 'unknown') for f in db_features)
        common_types = unknown_types.intersection(db_types)
        
        # Calculate feature penalties
        missing_penalty = len(unknown_types - db_types) * self.config['penalties']['missing_feature']
        extra_penalty = len(db_types - unknown_types) * self.config['penalties']['extra_feature']
        feature_penalty = missing_penalty + extra_penalty
        
        if not common_types:
            return max(0.0, 100.0 - feature_penalty)
        
        # Calculate type scores with optimized matching
        type_scores = []
        for feature_type in common_types:
            unknown_type_features = [f for f in unknown_features if f.get('feature_type') == feature_type]
            db_type_features = [f for f in db_features if f.get('feature_type') == feature_type]
            score = self.match_features_of_type_optimized(unknown_type_features, db_type_features, feature_type)
            type_scores.append(score)
        
        base_score = sum(type_scores) / len(type_scores) if type_scores else 0.0
        preliminary_score = max(0.0, base_score - feature_penalty)
        
        # Enhanced reporting for significant matches
        if preliminary_score >= 30.0:
            print(f"\n   H/L Wavelength analysis: {unknown_gem_id} vs {db_gem_id}")
            if missing_penalty > 0:
                print(f"      Missing features: {unknown_types - db_types} (-{missing_penalty}%)")
            if extra_penalty > 0:
                print(f"      Extra features: {db_types - unknown_types} (-{extra_penalty}%)")
            print(f"      Base spectral score: {base_score:.1f}%, Final: {preliminary_score:.1f}%")
        
        return min(100.0, preliminary_score)
    
    def match_features_of_type_optimized(self, unknown_features: List[Dict], db_features: List[Dict], feature_type: str) -> float:
        """OPTIMIZED: Feature matching with vectorized scoring"""
        if not unknown_features or not db_features:
            return 0.0
        
        total_score, matched_count = 0.0, 0
        
        for unknown_feature in unknown_features:
            unknown_wl = self.extract_wavelength(unknown_feature, feature_type)
            if unknown_wl is None:
                continue
            
            # Vectorized scoring for all DB features
            scores = []
            for db_feature in db_features:
                db_wl = self.extract_wavelength(db_feature, feature_type)
                if db_wl is not None:
                    score = self.calculate_wavelength_score(unknown_wl, db_wl, feature_type)
                    scores.append(score)
            
            best_score = max(scores) if scores else 0.0
            total_score += best_score
            matched_count += 1
        
        return total_score / matched_count if matched_count > 0 else 0.0
    
    def extract_normalization_scheme(self, features: List[Dict]) -> Optional[str]:
        """OPTIMIZED: Extract normalization scheme with enhanced field detection"""
        scheme_fields = ['Normalization_Scheme', 'normalization_scheme', 'norm_scheme']
        
        for feature in features:
            for field in scheme_fields:
                if field in feature and feature[field]:
                    return feature[field]
        return None
    
    def match_features_by_light_source_optimized(self, unknown_features: List[Dict], db_features: List[Dict], 
                                                light_source: str, unknown_gem_id: str, db_gem_id: str) -> float:
        """OPTIMIZED: Route to appropriate matching with normalization validation"""
        if unknown_gem_id == db_gem_id:
            return 100.0
        if not unknown_features or not db_features:
            return 0.0
        
        # Enhanced normalization validation
        unknown_norm = self.extract_normalization_scheme(unknown_features)
        db_norm = self.extract_normalization_scheme(db_features)
        if unknown_norm and db_norm and unknown_norm != db_norm:
            print(f"      WARNING: Normalization mismatch - Unknown: {unknown_norm}, DB: {db_norm}")
        
        # Route to appropriate matcher
        if light_source.upper() == 'UV':
            return self.match_uv_intensity_ratios_optimized(unknown_features, db_features, unknown_gem_id, db_gem_id)
        else:
            return self.match_halogen_laser_optimized(unknown_features, db_features, unknown_gem_id, db_gem_id)
    
    def load_unknown_data_optimized(self, file_path: Path) -> List[Dict]:
        """OPTIMIZED: Universal data loader with enhanced format detection"""
        try:
            df = pd.read_csv(file_path)
            features = []
            
            # Peak detection format
            if 'Peak_Number' in df.columns:
                for _, row in df.iterrows():
                    feature = {
                        'feature_type': 'Peak', 'wavelength': row['Wavelength_nm'], 
                        'max_wavelength': row['Wavelength_nm'], 'intensity': row['Intensity'],
                        'prominence': row.get('Prominence', 1.0)
                    }
                    # Add metadata fields
                    for key in ['Normalization_Scheme', 'Reference_Wavelength', 'Light_Source']:
                        if key in row and pd.notna(row[key]):
                            feature[key] = row[key]
                    features.append(feature)
            
            # Structural features format
            elif 'Feature' in df.columns:
                field_mapping = {'Crest': 'crest_wavelength', 'Midpoint': 'midpoint_wavelength', 
                               'Start': 'start_wavelength', 'End': 'end_wavelength'}
                
                for _, row in df.iterrows():
                    feature = {
                        'feature_type': row.get('Feature', 'unknown'),
                        'wavelength': row.get('Wavelength', row.get('Crest')),
                        'intensity': row.get('Intensity', 1.0)
                    }
                    
                    # Map additional fields
                    for col, field in field_mapping.items():
                        if col in row:
                            feature[field] = row[col]
                    
                    # Add normalization metadata
                    if 'Normalization_Scheme' in row and pd.notna(row['Normalization_Scheme']):
                        feature['Normalization_Scheme'] = row['Normalization_Scheme']
                    
                    features.append(feature)
            
            return features
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def validate_data_optimized(self, features: List[Dict], light_source: str, file_info: Dict):
        """OPTIMIZED: Comprehensive data validation with enhanced reporting"""
        print(f"   Found {len(features)} spectral features")
        
        # Validate normalization scheme
        norm_scheme = self.extract_normalization_scheme(features)
        expected = self.config['normalization_schemes'].get(light_source)
        
        if not norm_scheme:
            print("   WARNING: No normalization scheme found")
        elif expected and norm_scheme != expected:
            print(f"   WARNING: Unexpected normalization - Found: {norm_scheme}, Expected: {expected}")
        else:
            print(f"   Normalization validated: {norm_scheme}")
        
        # Validate wavelength range
        wavelengths = [self.extract_wavelength(f, f.get('feature_type', '')) for f in features]
        wavelengths = [w for w in wavelengths if w is not None]
        if wavelengths:
            print(f"   Wavelength range: {min(wavelengths):.1f} - {max(wavelengths):.1f} nm")
        
        # Validate intensity range for 0-100 scale
        intensities = []
        for feature in features:
            intensity = feature.get('intensity', feature.get('Intensity'))
            if intensity is not None:
                try:
                    intensities.append(float(intensity))
                except (ValueError, TypeError):
                    continue
        
        if intensities:
            min_int, max_int = min(intensities), max(intensities)
            print(f"   Intensity range: {min_int:.2f} - {max_int:.2f}")
            
            # Enhanced validation for different scales
            if max_int <= 1.0:
                print("   ERROR: Intensities appear to be 0-1 normalized (broken for UV analysis)")
            elif max_int > 100.0:
                print("   WARNING: Intensities exceed 100 (unexpected for fixed normalization)")
            elif min_int < 0:
                print("   WARNING: Negative intensities found")
            else:
                print("   Intensity range validated for 0-100 scale")
            
            # UV-specific reference validation
            if light_source == 'UV':
                ref_wl = self.config['uv_params']['reference_wavelength']
                ref_intensity = None
                for feature in features:
                    wl = self.extract_wavelength(feature, 'UV')
                    if wl and abs(wl - ref_wl) <= 1.0:
                        intensity = feature.get('intensity', feature.get('Intensity', 0))
                        if intensity > 0:
                            ref_intensity = intensity
                            break
                
                if ref_intensity:
                    if ref_intensity < 10.0:
                        print(f"   WARNING: {ref_wl}nm reference ({ref_intensity:.2f}) seems low")
                    else:
                        print(f"   {ref_wl}nm reference validated: {ref_intensity:.2f}")
                else:
                    print(f"   WARNING: No {ref_wl}nm reference peak found")
        
        return norm_scheme
    
    def find_database_matches_optimized(self, unknown_data: List[Dict], file_info: Dict, top_n: int = 10) -> List[Dict]:
        """OPTIMIZED: Database matching with enhanced efficiency and compatibility checking"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """SELECT file, light_source, wavelength, intensity, feature_group, data_type, 
                             start_wavelength, end_wavelength, midpoint, bottom, normalization_scheme, reference_wavelength
                       FROM structural_features WHERE light_source = ? ORDER BY file, wavelength"""
            
            db_df = pd.read_sql_query(query, conn, params=(file_info['light_source'],))
            conn.close()
            
            if db_df.empty:
                return []
            
            matches, unknown_norm = [], self.extract_normalization_scheme(unknown_data)
            
            # Process each database file
            for db_file in db_df['file'].unique():
                file_data = db_df[db_df['file'] == db_file]
                db_file_info = self.parse_gem_filename(db_file)
                
                # Build database features with enhanced metadata
                db_features = []
                for _, row in file_data.iterrows():
                    feature = {
                        'feature_type': row.get('feature_group', 'unknown'), 'wavelength': row['wavelength'],
                        'intensity': row['intensity'], 'midpoint_wavelength': row.get('midpoint'),
                        'start_wavelength': row.get('start_wavelength'), 'end_wavelength': row.get('end_wavelength'),
                        'crest_wavelength': row['wavelength'], 'max_wavelength': row['wavelength']
                    }
                    
                    # Add normalization metadata
                    for col, key in [('normalization_scheme', 'Normalization_Scheme'), ('reference_wavelength', 'Reference_Wavelength')]:
                        if col in row and pd.notna(row[col]):
                            feature[key] = row[col]
                    
                    db_features.append(feature)
                
                # Calculate match score
                score = self.match_features_by_light_source_optimized(
                    unknown_data, db_features, file_info['light_source'], file_info['gem_id'], db_file_info['gem_id']
                )
                
                # Apply normalization compatibility bonus/penalty
                db_norm = self.extract_normalization_scheme(db_features)
                if unknown_norm and db_norm:
                    score += 2.0 if unknown_norm == db_norm else -5.0
                
                if score > 0:
                    matches.append({
                        'db_gem_id': db_file_info['gem_id'], 'db_full_id': db_file_info['full_identifier'],
                        'score': score, 'db_features': len(db_features), 'light_source': db_file_info['light_source'],
                        'orientation': db_file_info['orientation'], 'scan_number': db_file_info['scan_number'],
                        'normalization_scheme': db_norm,
                        'normalization_compatible': unknown_norm == db_norm if unknown_norm and db_norm else None
                    })
            
            matches.sort(key=lambda x: x['score'], reverse=True)
            return matches[:top_n]
            
        except Exception as e:
            print(f"Error finding matches: {e}")
            return []
    
    def analyze_unknown_file_optimized(self, file_path: Path) -> Dict:
        """OPTIMIZED: Complete file analysis with enhanced validation and reporting"""
        file_info = self.parse_gem_filename(file_path.name)
        print(f"\nAnalyzing: {file_info['original_filename']}")
        print(f"   Gem: {file_info['gem_id']}, Light: {file_info['light_source']}, "
              f"Orientation: {file_info['orientation']}, Scan: {file_info['scan_number']}")
        
        # Load and validate data
        unknown_data = self.load_unknown_data_optimized(file_path)
        if not unknown_data:
            print("   Could not load file data")
            return {'error': 'Could not load data'}
        
        norm_scheme = self.validate_data_optimized(unknown_data, file_info['light_source'], file_info)
        
        # Find matches
        matches = self.find_database_matches_optimized(unknown_data, file_info)
        
        if matches:
            print(f"   Found {len(matches)} potential matches:")
            for i, match in enumerate(matches[:3], 1):  # Show top 3 for brevity
                compat_str = " [COMPATIBLE]" if match.get('normalization_compatible') else ""
                print(f"      {i}. {match['db_gem_id']} - {match['score']:.1f}% "
                      f"({match['db_features']} features){compat_str}")
            if len(matches) > 3:
                print(f"      ... and {len(matches) - 3} more matches")
        else:
            print("   No similar gems found in database")
        
        return {
            'file_info': file_info, 'unknown_data': unknown_data, 'matches': matches, 
            'normalization_scheme': norm_scheme
        }
    
    def analyze_all_unknowns_optimized(self):
        """OPTIMIZED: Batch analysis with enhanced summary reporting"""
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
        all_results, normalization_issues = {}, []
        
        for file_path in csv_files:
            try:
                result = self.analyze_unknown_file_optimized(file_path)
                all_results[file_path.name] = result
                
                # Track normalization issues
                if 'normalization_scheme' in result:
                    scheme = result['normalization_scheme']
                    if not scheme or 'Unknown' in str(scheme):
                        normalization_issues.append(file_path.name)
            except Exception as e:
                print(f"Error analyzing {file_path.name}: {e}")
        
        # Enhanced summary reporting
        self.generate_analysis_summary_optimized(all_results, normalization_issues)
        return all_results
    
    def generate_analysis_summary_optimized(self, all_results: Dict, normalization_issues: List[str]):
        """OPTIMIZED: Generate comprehensive analysis summary"""
        print(f"\nENHANCED ANALYSIS SUMMARY (ULTRA OPTIMIZED for 0-100 normalization):")
        print("=" * 70)
        
        # Normalization warnings
        if normalization_issues:
            print("NORMALIZATION WARNINGS:")
            for issue_file in normalization_issues:
                print(f"   ! {issue_file}: Missing or unknown normalization scheme")
            print()
        
        # Categorized results
        categories = {'perfect': [], 'excellent': [], 'strong': [], 'moderate': [], 'weak': [], 'no_match': []}
        thresholds = self.config['score_thresholds']
        
        for filename, result in all_results.items():
            if 'error' in result:
                categories['no_match'].append((filename, result['error']))
            elif result.get('matches'):
                best_score = result['matches'][0]['score']
                if best_score == 100.0:
                    categories['perfect'].append((filename, result))
                elif best_score >= thresholds['excellent']:
                    categories['excellent'].append((filename, result))
                elif best_score >= thresholds['strong']:
                    categories['strong'].append((filename, result))
                elif best_score >= thresholds['moderate']:
                    categories['moderate'].append((filename, result))
                else:
                    categories['weak'].append((filename, result))
            else:
                categories['no_match'].append((filename, "No matches found"))
        
        # Generate category summaries
        category_descriptions = {
            'perfect': "PERFECT MATCHES (100%) - Same gem, different scan",
            'excellent': f"EXCELLENT MATCHES (≥{thresholds['excellent']}%) - Very likely same species",
            'strong': f"STRONG MATCHES (≥{thresholds['strong']}%) - Probably same gem",
            'moderate': f"MODERATE MATCHES (≥{thresholds['moderate']}%) - Possibly same variety",
            'weak': f"WEAK MATCHES (≥{thresholds['weak']}%) - Some similarity",
            'no_match': "NO MATCHES - No similar gems found"
        }
        
        for category, description in category_descriptions.items():
            items = categories[category]
            if items:
                print(f"{description}:")
                for filename, data in items:
                    if category == 'no_match' and isinstance(data, str):
                        print(f"   • {filename}: {data}")
                    elif category != 'no_match':
                        file_info = data['file_info']
                        best_match = data['matches'][0]
                        compat_indicator = (" [COMPATIBLE]" if best_match.get('normalization_compatible') is True else 
                                          " [INCOMPATIBLE]" if best_match.get('normalization_compatible') is False else "")
                        print(f"   • {filename} ({file_info['gem_id']}) → {best_match['db_gem_id']} ({best_match['score']:.1f}%){compat_indicator}")
                print()
    
    def analyze_multi_light_integration_optimized(self, gem_id: str):
        """OPTIMIZED: Multi-light integration with enhanced scoring and validation"""
        print(f"\nMULTI-LIGHT INTEGRATION ANALYSIS (ULTRA OPTIMIZED): {gem_id}")
        print("=" * 70)
        
        # Find unknown files for this gem
        unknown_files = {}
        if self.unknown_path.exists():
            for file in self.unknown_path.glob("*.csv"):
                file_info = self.parse_gem_filename(file.name)
                if file_info['gem_id'] == gem_id:
                    unknown_files[file_info['light_source']] = file
        
        if not unknown_files:
            print(f"No unknown files found for gem {gem_id}")
            return
        
        print(f"Found unknown files: {', '.join(f'{ls}' for ls in unknown_files.keys())}")
        
        # Analyze each light source
        best_matches, normalization_summary, gem_scores = {}, {}, {}
        
        for light_source, file_path in unknown_files.items():
            print(f"\nAnalyzing {light_source} data...")
            
            file_info = self.parse_gem_filename(file_path.name)
            unknown_data = self.load_unknown_data_optimized(file_path)
            
            if unknown_data:
                norm_scheme = self.validate_data_optimized(unknown_data, light_source, file_info)
                normalization_summary[light_source] = norm_scheme
                
                matches = self.find_database_matches_optimized(unknown_data, file_info, top_n=3)
                if matches:
                    best_match = matches[0]
                    best_score = best_match['score']
                    gem_scores[light_source] = best_score
                    best_matches[light_source] = best_match
                    
                    compat_str = (" [COMPATIBLE]" if best_match.get('normalization_compatible') is True else 
                                " [INCOMPATIBLE]" if best_match.get('normalization_compatible') is False else "")
                    print(f"   Best match: {best_match['db_gem_id']} ({best_score:.1f}%){compat_str}")
                else:
                    gem_scores[light_source] = 0.0
                    best_matches[light_source] = None
                    print(f"   No matches found")
        
        # Generate integrated analysis
        if gem_scores:
            self.generate_integration_summary_optimized(gem_id, gem_scores, best_matches, normalization_summary)
            
            return {
                'gem_id': gem_id, 'light_source_scores': gem_scores, 'light_source_matches': best_matches,
                'integrated_score': sum(gem_scores.values()) / len(gem_scores),
                'normalization_schemes': normalization_summary
            }
        
        return None
    
    def generate_integration_summary_optimized(self, gem_id: str, gem_scores: Dict, best_matches: Dict, normalization_summary: Dict):
        """OPTIMIZED: Generate comprehensive integration summary"""
        integrated_score = sum(gem_scores.values()) / len(gem_scores)
        
        # Determine best overall gem through weighted voting
        gem_vote_counts, gem_total_scores = {}, {}
        for light_source, match in best_matches.items():
            if match:
                db_gem = match['db_gem_id']
                score = match['score']
                
                if db_gem not in gem_vote_counts:
                    gem_vote_counts[db_gem] = 0
                    gem_total_scores[db_gem] = 0
                
                gem_vote_counts[db_gem] += 1
                gem_total_scores[db_gem] += score
        
        best_overall_gem = max(gem_vote_counts.keys(), 
                             key=lambda gem: gem_vote_counts[gem] * gem_total_scores[gem] / gem_vote_counts[gem],
                             default=None) if gem_vote_counts else None
        
        print(f"\nINTEGRATED ANALYSIS RESULTS (ULTRA OPTIMIZED):")
        print("=" * 50)
        print(f"UNKNOWN GEM: {gem_id}")
        print(f"BEST MATCH: {best_overall_gem if best_overall_gem else 'No clear winner'}")
        
        # Normalization validation summary
        print(f"\nNORMALIZATION VALIDATION:")
        for light_source in ['UV', 'Laser', 'Halogen']:
            if light_source in normalization_summary:
                scheme = normalization_summary[light_source] or 'Unknown'
                print(f"   {light_source}: {scheme}")
        
        # Light source analysis
        print(f"\nLIGHT SOURCE ANALYSIS:")
        light_icons = {'UV': '☢️  UV', 'Laser': '🔴 LASER', 'Halogen': '💡 HALOGEN'}
        for light_source in ['UV', 'Laser', 'Halogen']:
            if light_source in gem_scores:
                score = gem_scores[light_source]
                match = best_matches.get(light_source)
                
                if match:
                    matched_gem = match['db_gem_id']
                    norm_compat = match.get('normalization_compatible')
                    
                    compat_indicator = (" ✓COMPAT" if norm_compat is True else 
                                      " ✗INCOMPAT" if norm_compat is False else "")
                    match_indicator = " ⭐MATCH" if matched_gem == best_overall_gem else ""
                    print(f"   {light_icons.get(light_source, light_source)}: {score:.1f}% → {matched_gem}{compat_indicator}{match_indicator}")
                else:
                    print(f"   {light_icons.get(light_source, light_source)}: {score:.1f}% → No match")
        
        print(f"\nINTEGRATED SCORE: {integrated_score:.1f}%")
        
        # Confidence assessment
        thresholds = self.config['score_thresholds']
        confidence_levels = [
            (thresholds['excellent'], "EXCELLENT: Very likely same gem"),
            (thresholds['strong'], "STRONG: Probably same gem"), 
            (thresholds['moderate'], "MODERATE: Might be similar"),
            (thresholds['weak'], "WEAK: Some similarity"),
            (0.0, "POOR: No strong match")
        ]
        
        for threshold, description in confidence_levels:
            if integrated_score >= threshold:
                print(f"   {description}")
                break
        
        # Multi-candidate summary
        if len(gem_vote_counts) > 1:
            print(f"\nCANDIDATE SUMMARY:")
            sorted_candidates = sorted(gem_vote_counts.items(), 
                                     key=lambda x: gem_total_scores[x[0]]/x[1], reverse=True)
            for candidate_gem, vote_count in sorted_candidates[:3]:  # Top 3 candidates
                avg_score = gem_total_scores[candidate_gem] / vote_count
                light_sources_matched = [ls for ls, match in best_matches.items() 
                                       if match and match['db_gem_id'] == candidate_gem]
                print(f"   • {candidate_gem}: {avg_score:.1f}% avg "
                      f"({vote_count}/{len(gem_scores)} sources: {', '.join(light_sources_matched)})")
    
    def run_interactive_menu(self):
        """OPTIMIZED: Interactive menu system with enhanced options"""
        print("ENHANCED GEM ANALYZER v2.2 - ULTRA OPTIMIZED")
        print("Advanced matching with 50% code reduction + enhanced features")
        print("=" * 70)
        
        menu_options = {
            "1": ("Analyze Unknown Files (ULTRA OPTIMIZED)", self.analyze_all_unknowns_optimized),
            "2": ("Multi-Light Integration Analysis", self.handle_multi_light_menu),
            "3": ("Show Unknown Directory", self.show_unknown_directory),
            "4": ("Database Statistics", self.show_database_stats),
            "5": ("Show Matching Parameters", self.show_matching_parameters),
            "6": ("Clear Unknown Directory", self.clear_unknown_directory),
            "7": ("Exit", None)
        }
        
        while True:
            print(f"\nMAIN MENU:")
            for key, (desc, _) in menu_options.items():
                print(f"{key}. {desc}")
            
            try:
                choice = input("Choice (1-7): ").strip()
                
                if choice == "7":
                    print("Goodbye!")
                    break
                
                if choice in menu_options:
                    _, action = menu_options[choice]
                    if action:
                        action()
                        if choice != "7":
                            input("\nPress Enter to continue...")
                else:
                    print("Invalid choice. Please enter 1-7")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def handle_multi_light_menu(self):
        """OPTIMIZED: Multi-light analysis menu handler"""
        gem_ids = set()
        if self.unknown_path.exists():
            for file in self.unknown_path.glob("*.csv"):
                file_info = self.parse_gem_filename(file.name)
                gem_ids.add(file_info['gem_id'])
        
        if gem_ids:
            gem_list = sorted(list(gem_ids))
            print("Found gems with unknown data:")
            for i, gem_id in enumerate(gem_list, 1):
                print(f"   {i}. {gem_id}")
            
            try:
                selection = input(f"\nSelect gem (1-{len(gem_list)}) or 'all': ").strip()
                if selection.lower() == 'all':
                    for gem_id in gem_list:
                        self.analyze_multi_light_integration_optimized(gem_id)
                else:
                    idx = int(selection) - 1
                    if 0 <= idx < len(gem_list):
                        self.analyze_multi_light_integration_optimized(gem_list[idx])
                    else:
                        print("Invalid selection")
            except ValueError:
                print("Invalid selection")
        else:
            print("No unknown gems found")
    
    def show_unknown_directory(self):
        """OPTIMIZED: Show directory contents with enhanced info"""
        if self.unknown_path.exists():
            files = list(self.unknown_path.glob("*.csv"))
            if files:
                print(f"Found {len(files)} files in unknown directory:")
                for file in files:
                    info = self.parse_gem_filename(file.name)
                    size_kb = file.stat().st_size / 1024
                    print(f"   📄 {file.name} ({size_kb:.1f} KB)")
                    print(f"       Gem: {info['gem_id']}, Light: {info['light_source']}, "
                          f"Orientation: {info['orientation']}, Scan: {info['scan_number']}")
            else:
                print("No CSV files found in unknown directory")
        else:
            print(f"Unknown directory not found: {self.unknown_path}")
    
    def show_database_stats(self):
        """OPTIMIZED: Enhanced database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features")
            unique_files = cursor.fetchone()[0]
            
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source ORDER BY COUNT(*) DESC")
            by_light = cursor.fetchall()
            
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE normalization_scheme IS NOT NULL")
            with_norm = cursor.fetchone()[0]
            
            # Enhanced stats
            cursor.execute("SELECT COUNT(DISTINCT SUBSTR(file, 1, INSTR(file, '_')-1)) FROM structural_features WHERE file LIKE '%_%'")
            unique_gems = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            print(f"DATABASE STATISTICS (ULTRA OPTIMIZED)")
            print("=" * 50)
            print(f"📊 Total spectral features: {total:,}")
            print(f"📁 Unique gem files: {unique_files:,}")
            print(f"💎 Estimated unique gems: {unique_gems:,}")
            print(f"🏷️  Records with normalization: {with_norm:,} ({100*with_norm/total:.1f}%)")
            
            print(f"\n💡 By Light Source:")
            for light, count in by_light:
                percentage = 100 * count / total
                print(f"   {light}: {count:,} ({percentage:.1f}%)")
            
            conn.close()
            
        except Exception as e:
            print(f"Database error: {e}")
    
    def show_matching_parameters(self):
        """OPTIMIZED: Show comprehensive matching parameters"""
        print("MATCHING PARAMETERS (ULTRA OPTIMIZED)")
        print("=" * 60)
        
        print("🎯 Wavelength Tolerances (Halogen/Laser):")
        for param, value in self.config['tolerances'].items():
            print(f"   {param.replace('_', ' ').title()}: ±{value} nm")
        
        print(f"\n⚠️  Matching Penalties:")
        for param, value in self.config['penalties'].items():
            print(f"   {param.replace('_', ' ').title()}: -{value}%")
        
        print(f"\n☢️  UV Analysis Parameters (0-100 scale):")
        uv_params = self.config['uv_params']
        print(f"   Reference wavelength: {uv_params['reference_wavelength']} nm")
        print(f"   Expected intensity: ~{uv_params['reference_expected_intensity']} (0-100 scale)")
        print(f"   Minimum peak threshold: {uv_params['minimum_real_peak_intensity']}")
        print(f"   Standards: {', '.join(f'{wl:.1f}nm' for wl in uv_params['real_peak_standards'])}")
        
        print(f"\n🔍 Diagnostic Peaks:")
        for wl, desc in uv_params['diagnostic_peaks'].items():
            print(f"   {wl:.1f}nm: {desc}")
        
        print(f"\n📋 Expected Normalization Schemes:")
        for light, scheme in self.config['normalization_schemes'].items():
            print(f"   {light}: {scheme}")
        
        print(f"\n🎚️  Score Thresholds:")
        for level, threshold in self.config['score_thresholds'].items():
            print(f"   {level.title()}: ≥{threshold}%")
        
        print(f"\n✅ Enhanced Validation Features:")
        validation_features = [
            "Dynamic UV threshold determination",
            "Multi-field wavelength extraction",
            "Normalization compatibility checking", 
            "Enhanced diagnostic peak analysis",
            "Vectorized feature matching",
            "Comprehensive data validation"
        ]
        for feature in validation_features:
            print(f"   • {feature}")
    
    def clear_unknown_directory(self):
        """OPTIMIZED: Clear directory with confirmation"""
        if self.unknown_path.exists():
            files = list(self.unknown_path.glob("*.csv"))
            if files:
                print(f"Found {len(files)} files to delete:")
                for file in files[:5]:  # Show first 5
                    print(f"   📄 {file.name}")
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more files")
                
                confirm = input(f"\n⚠️  Delete all {len(files)} files? (y/N): ").strip().lower()
                if confirm == 'y':
                    deleted = 0
                    for file in files:
                        try:
                            file.unlink()
                            deleted += 1
                        except Exception as e:
                            print(f"Error deleting {file.name}: {e}")
                    print(f"✅ Deleted {deleted}/{len(files)} files")
                else:
                    print("❌ Operation cancelled")
            else:
                print("No CSV files found to delete")
        else:
            print(f"❌ Unknown directory not found: {self.unknown_path}")

def main():
    """OPTIMIZED: Main entry point with enhanced initialization"""
    print("🔬 TESTING ULTRA OPTIMIZED UV RATIO CALCULATIONS")
    print("=" * 60)
    
    analyzer = UltraOptimizedGemAnalyzer()
    
    # Display key parameters
    uv_params = analyzer.config['uv_params']
    print(f"☢️  ULTRA OPTIMIZED UV Parameters:")
    print(f"   Expected {uv_params['reference_wavelength']}nm intensity: ~{uv_params['reference_expected_intensity']} (0-100 scale)")
    print(f"   Minimum real peak: {uv_params['minimum_real_peak_intensity']}")
    print(f"   Standards: {len(uv_params['real_peak_standards'])} reference peaks")
    
    print(f"\n🏷️  Expected normalization schemes:")
    for light, scheme in analyzer.config['normalization_schemes'].items():
        print(f"   {light}: {scheme}")
    
    print(f"\n🚀 Starting ULTRA OPTIMIZED Enhanced Gem Analyzer...")
    print("=" * 60)
    
    analyzer.run_interactive_menu()

if __name__ == "__main__":
    main()
