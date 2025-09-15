#!/usr/bin/env python3
"""
GEMINI NUMERICAL ANALYSIS ENGINE v2.0
Optimized version with reduced line count while preserving all functionality
"""

import os, sys, pandas as pd, numpy as np, matplotlib.pyplot as plt, sqlite3, shutil, glob, re, json
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GeminiNumericalAnalyzer:
    def __init__(self):
        self.database_dir = "database"
        self.reference_files = {k: os.path.join(self.database_dir, "reference_spectra", f"gemini_db_long_{k}.csv") for k in ['B','L','U']}
        self.structural_db = "multi_structural_gem_data.db"
        self.temp_output_dir = "output/numerical_analysis"
        self.final_graphs_dir = "results/post_analysis_numerical/graphs"
        self.final_reports_dir = "results/post_analysis_numerical/reports"
        self.wavelength_tolerance = 2.0
        self.min_overlap_percentage = 70
        self.all_gem_spectra = {}
        self.available_base_gems = {}
        self.selected_gem_names = []
        self.selected_base_gem_id = None
        self.unknown_data = {}
        self.match_results = []
        self.generated_files = {'png_files': [], 'csv_files': []}
        print("=" * 80 + "\n  GEMINI NUMERICAL ANALYSIS ENGINE v2.0\n" + "=" * 80)
    
    def parse_gem_name(self, gem_name):
        patterns = [r'^([CS]\d+)([BLU])([CP])(\d+)$', r'^(\d+)([BLU])([CP])(\d+)$']
        for pattern in patterns:
            match = re.match(pattern, str(gem_name).strip())
            if match:
                return {'base_id': match.group(1), 'light_source': match.group(2), 
                       'orientation': match.group(3), 'scan_number': int(match.group(4)), 'full_name': gem_name}
        return None
    
    def load_reference_database(self):
        print(f"\nLOADING REFERENCE DATABASE\n" + "=" * 50)
        all_gem_spectra = {}
        
        for light_source, filepath in self.reference_files.items():
            print(f"Loading {light_source} reference: {filepath}")
            if not os.path.exists(filepath):
                print(f"   Warning: File not found: {filepath}")
                continue
            
            try:
                df = pd.read_csv(filepath)
                if df.empty:
                    print(f"   Warning: {light_source} reference is empty")
                    continue
                
                print(f"   File loaded: {len(df)} rows, {len(df.columns)} columns")
                
                # Find columns flexibly
                cols = df.columns
                gem_id_col = next((col for col in cols if any(x in str(col).lower() for x in ['gem','id','sample','name'])), cols[0])
                wavelength_col = next((col for col in cols if any(x in str(col).lower() for x in ['wavelength','wl'])), cols[1])
                intensity_col = next((col for col in cols if any(x in str(col).lower() for x in ['intensity','int','signal'])), cols[2])
                
                print(f"   Using columns: ID='{gem_id_col}', WL='{wavelength_col}', INT='{intensity_col}'")
                
                valid_gems = 0
                for gem_name in df[gem_id_col].dropna().unique():
                    gem_name_str = str(gem_name).strip()
                    if not gem_name_str or gem_name_str.lower() in ['nan','none','null']:
                        continue
                    
                    parsed = self.parse_gem_name(gem_name_str)
                    if not parsed:
                        continue
                    
                    gem_data = df[df[gem_id_col] == gem_name]
                    try:
                        wavelengths = pd.to_numeric(gem_data[wavelength_col], errors='coerce').dropna().values
                        intensities = pd.to_numeric(gem_data[intensity_col], errors='coerce').dropna().values
                        
                        if len(wavelengths) >= 10:
                            all_gem_spectra[gem_name_str] = {
                                'parsed': parsed, 'wavelengths': wavelengths, 'intensities': intensities,
                                'light_source': light_source, 'points': len(wavelengths),
                                'wl_range': f"{wavelengths.min():.1f}-{wavelengths.max():.1f}nm"
                            }
                            valid_gems += 1
                    except:
                        continue
                
                print(f"   ✓ Loaded {valid_gems} valid gem spectra")
            except Exception as e:
                print(f"   Error loading {light_source}: {e}")
        
        self.all_gem_spectra = all_gem_spectra
        self.organize_gems_by_base_id()
        self.load_structural_database()
        print(f"\nTotal individual spectra loaded: {len(all_gem_spectra)}\nTotal base gems available: {len(self.available_base_gems)}")
        return len(all_gem_spectra) > 0
    
    def organize_gems_by_base_id(self):
        base_gems = defaultdict(list)
        for gem_name, spectrum_data in self.all_gem_spectra.items():
            parsed = spectrum_data['parsed']
            base_gems[parsed['base_id']].append({
                'full_name': gem_name, 'spectrum_data': spectrum_data,
                'light_source': parsed['light_source'], 'orientation': parsed['orientation'], 'scan_number': parsed['scan_number']
            })
        
        self.available_base_gems = dict(base_gems)
        for base_id in self.available_base_gems:
            self.available_base_gems[base_id].sort(key=lambda x: (x['light_source'], x['orientation'], x['scan_number']))
    
    def load_structural_database(self):
        if os.path.exists(self.structural_db):
            try:
                conn = sqlite3.connect(self.structural_db)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM structural_features")
                count = cursor.fetchone()[0]
                if count > 0:
                    print(f"   ✓ Found {count:,} structural features (not integrated in this version)")
                conn.close()
                return True
            except Exception as e:
                print(f"   Error accessing structural database: {e}")
        else:
            print(f"   Structural database not found: {self.structural_db}")
        return False
    
    def display_available_base_gems(self):
        print(f"\nAVAILABLE BASE GEMS\n" + "=" * 80)
        if not self.available_base_gems:
            print("No gems found in database")
            return False
        
        sorted_base_gems = sorted(self.available_base_gems.items(), key=lambda x: self.sort_key_for_gem_id(x[0]))
        print(f"{'Base Gem':<15} {'Spectra':<8} {'Light Sources':<15} {'Sample Spectra Names'}")
        print("-" * 80)
        
        for base_id, spectra_list in sorted_base_gems[:50]:
            light_sources = '+'.join(sorted(set(s['light_source'] for s in spectra_list)))
            sample_names = [s['full_name'] for s in spectra_list[:3]]
            if len(spectra_list) > 3:
                sample_names.append(f"... +{len(spectra_list)-3} more")
            print(f"{base_id:<15} {len(spectra_list):<8} {light_sources:<15} {', '.join(sample_names)}")
        
        if len(sorted_base_gems) > 50:
            print(f"... and {len(sorted_base_gems) - 50} more base gems")
        
        print(f"\nTotal base gems: {len(sorted_base_gems)}")
        light_counts = defaultdict(int)
        for base_id, spectra_list in sorted_base_gems:
            for spectrum in spectra_list:
                light_counts[spectrum['light_source']] += 1
        
        print(f"Spectrum distribution by light source:")
        for light, count in sorted(light_counts.items()):
            print(f"   {light}: {count} spectra")
        return True
    
    def sort_key_for_gem_id(self, gem_id):
        if gem_id.startswith(('C', 'S')):
            return (1, gem_id)
        try:
            return (0, int(gem_id))
        except:
            return (2, gem_id)
    
    def select_gem_spectra_for_testing(self):
        print(f"\nSELECT GEM SPECTRA FOR TESTING\n" + "=" * 50)
        print("Enter up to 3 full gem names representing the same physical gem\nunder different light sources (B, L, U).")
        print("Examples:\n   Single light: 58BC1\n   Two lights: 58BC1, 58LC2\n   Three lights: 58BC1, 58LC2, 58UC3")
        print("   Client gem: C0045BC1, C0045LC1, C0045UC1\n   Source gem: S20250909BC1, S20250909LC2")
        
        while True:
            gem_input = input("\nEnter gem names (comma-separated) or 'q' to quit: ").strip()
            if gem_input.lower() == 'q':
                return False
            
            gem_names = [name.strip() for name in gem_input.split(',') if name.strip()]
            if not gem_names:
                print("Please enter at least one gem name")
                continue
            if len(gem_names) > 3:
                print("Maximum 3 gem spectra allowed")
                continue
            
            valid_gems, base_ids = [], set()
            for gem_name in gem_names:
                parsed = self.parse_gem_name(gem_name)
                if not parsed:
                    print(f"Invalid gem name format: '{gem_name}'\nExpected format: [prefix]number + light + orientation + scan\nExamples: 58BC1, C0045LC2, S20250909UP3")
                    break
                
                if gem_name not in self.all_gem_spectra:
                    print(f"Gem not found in database: '{gem_name}'\nAvailable gems for this base ID:")
                    if parsed['base_id'] in self.available_base_gems:
                        for spectrum in self.available_base_gems[parsed['base_id']][:10]:
                            print(f"   {spectrum['full_name']}")
                    else:
                        print(f"   No gems found for base ID: {parsed['base_id']}")
                    break
                
                valid_gems.append(gem_name)
                base_ids.add(parsed['base_id'])
            else:
                if len(base_ids) > 1:
                    print(f"All gems must have the same base ID. Found: {', '.join(base_ids)}")
                    continue
                
                print(f"\nSelected gems for testing:")
                base_gem_id = next(iter(base_ids))
                for gem_name in valid_gems:
                    spectrum_data = self.all_gem_spectra[gem_name]
                    parsed = spectrum_data['parsed']
                    print(f"   {gem_name}: {parsed['light_source']} light, {parsed['orientation']} orientation, scan {parsed['scan_number']}, {spectrum_data['points']} points")
                
                if input(f"\nTest base gem {base_gem_id} using these spectra? (y/n): ").strip().lower() == 'y':
                    self.selected_gem_names = valid_gems
                    self.selected_base_gem_id = base_gem_id
                    return True
    
    def extract_selected_gem_data(self):
        print(f"\nEXTRACTING SELECTED GEM: {self.selected_base_gem_id}\n" + "=" * 50)
        if not self.selected_gem_names:
            print("No gem spectra selected")
            return False
        
        extracted_data = {}
        for gem_name in self.selected_gem_names:
            spectrum_data = self.all_gem_spectra[gem_name]
            parsed = spectrum_data['parsed']
            light_source = parsed['light_source']
            
            extracted_data[light_source] = {
                'gem_name': gem_name, 'wavelengths': spectrum_data['wavelengths'],
                'intensities': spectrum_data['intensities'], 'parsed': parsed, 'spectrum_data': spectrum_data
            }
            print(f"   ✓ Extracted {gem_name}: {light_source} light, {spectrum_data['points']} points, {spectrum_data['wl_range']}")
        
        self.unknown_data = extracted_data
        total_spectra = len(self.all_gem_spectra)
        available_base_gems = len(self.available_base_gems)
        
        print(f"   ✓ Keeping all {total_spectra} spectra in database for matching")
        print(f"   ✓ Base gem {self.selected_base_gem_id} remains available for self-matching")
        print(f"   ✓ Total base gems available for search: {available_base_gems}")
        print(f"\n{self.selected_base_gem_id} now treated as UNKNOWN GEM")
        print(f"Test spectra: {', '.join(self.selected_gem_names)}")
        print(f"Light sources: {', '.join(extracted_data.keys())}")
        return True
    
    def calculate_numerical_match_score(self, unknown_wl, unknown_int, ref_wl, ref_int):
        min_wl, max_wl = max(unknown_wl.min(), ref_wl.min()), min(unknown_wl.max(), ref_wl.max())
        if min_wl >= max_wl:
            return 0.0, 0.0, {"error": "No wavelength overlap"}
        
        wl_grid = np.linspace(min_wl, max_wl, 200)
        unknown_interp = np.interp(wl_grid, unknown_wl, unknown_int)
        ref_interp = np.interp(wl_grid, ref_wl, ref_int)
        
        unknown_norm = (unknown_interp - unknown_interp.min()) / (unknown_interp.max() - unknown_interp.min()) * 100
        ref_norm = (ref_interp - ref_interp.min()) / (ref_interp.max() - ref_interp.min()) * 100
        
        correlation = np.corrcoef(unknown_norm, ref_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        mse = np.mean((unknown_norm - ref_norm) ** 2)
        mse_similarity = max(0, 100 - mse)
        
        unknown_peaks = self.find_spectral_peaks(wl_grid, unknown_norm)
        ref_peaks = self.find_spectral_peaks(wl_grid, ref_norm)
        peak_score = self.calculate_peak_alignment_score(unknown_peaks, ref_peaks)
        shape_score = self.calculate_shape_similarity(unknown_norm, ref_norm)
        
        weights = {'correlation': 0.3, 'mse': 0.25, 'peaks': 0.25, 'shape': 0.2}
        final_score = (correlation * 100 * weights['correlation'] + mse_similarity * weights['mse'] + 
                      peak_score * weights['peaks'] + shape_score * weights['shape'])
        
        overlap_percentage = (max_wl - min_wl) / (max(unknown_wl.max(), ref_wl.max()) - min(unknown_wl.min(), ref_wl.min())) * 100
        confidence = min(100, overlap_percentage * (final_score / 100))
        
        details = {'correlation': correlation, 'mse_similarity': mse_similarity, 'peak_score': peak_score,
                  'shape_score': shape_score, 'overlap_percentage': overlap_percentage, 'wavelength_range': f"{min_wl:.1f}-{max_wl:.1f}nm"}
        
        return final_score, confidence, details
    
    def find_spectral_peaks(self, wavelengths, intensities, prominence=5):
        peaks = []
        for i in range(1, len(intensities) - 1):
            if (intensities[i] > intensities[i-1] and intensities[i] > intensities[i+1] and intensities[i] > prominence):
                peaks.append({'wavelength': wavelengths[i], 'intensity': intensities[i],
                            'prominence': min(intensities[i] - intensities[i-1], intensities[i] - intensities[i+1])})
        peaks.sort(key=lambda x: x['intensity'], reverse=True)
        return peaks[:10]
    
    def calculate_peak_alignment_score(self, peaks1, peaks2):
        if not peaks1 or not peaks2:
            return 0.0
        
        total_score, matches = 0.0, 0
        for peak1 in peaks1:
            best_match_score = 0.0
            for peak2 in peaks2:
                wl_diff = abs(peak1['wavelength'] - peak2['wavelength'])
                if wl_diff <= self.wavelength_tolerance * 2:
                    wl_score = max(0, 100 - (wl_diff / (self.wavelength_tolerance * 2)) * 100)
                    int_ratio = min(peak1['intensity'], peak2['intensity']) / max(peak1['intensity'], peak2['intensity'])
                    int_score = int_ratio * 100
                    peak_match_score = (wl_score * 0.6 + int_score * 0.4)
                    best_match_score = max(best_match_score, peak_match_score)
            
            if best_match_score > 50:
                total_score += best_match_score
                matches += 1
        
        return total_score / matches if matches > 0 else 0.0
    
    def calculate_shape_similarity(self, spectrum1, spectrum2):
        if len(spectrum1) != len(spectrum2):
            return 0.0
        
        deriv1 = np.gradient(spectrum1)
        deriv2 = np.gradient(spectrum2)
        deriv1_norm = deriv1 / (np.std(deriv1) + 1e-10)
        deriv2_norm = deriv2 / (np.std(deriv2) + 1e-10)
        shape_correlation = np.corrcoef(deriv1_norm, deriv2_norm)[0, 1]
        if np.isnan(shape_correlation):
            shape_correlation = 0.0
        return (shape_correlation + 1) * 50
    
    def analyze_unknown_gem(self):
        print(f"\nANALYZING UNKNOWN GEM: {self.selected_base_gem_id}\n" + "=" * 50)
        print(f"Testing if system can correctly identify base gem {self.selected_base_gem_id}")
        print(f"Using spectra: {', '.join(self.selected_gem_names)}")
        
        if not self.unknown_data:
            print("No unknown gem data available for analysis")
            return False
        if not self.all_gem_spectra:
            print("No reference database available for comparison")
            return False
        
        all_matches = []
        for light_source, unknown_spectrum in self.unknown_data.items():
            print(f"\nAnalyzing {light_source} light source...")
            print(f"   Unknown spectrum: {unknown_spectrum['gem_name']}")
            
            unknown_wl, unknown_int = unknown_spectrum['wavelengths'], unknown_spectrum['intensities']
            ref_spectra = [(gem_name, spectrum_data) for gem_name, spectrum_data in self.all_gem_spectra.items() 
                          if spectrum_data['parsed']['light_source'] == light_source]
            
            print(f"   Comparing against {len(ref_spectra)} reference spectra...")
            
            light_matches = []
            for gem_name, spectrum_data in ref_spectra:
                ref_wl, ref_int = spectrum_data['wavelengths'], spectrum_data['intensities']
                try:
                    score, confidence, details = self.calculate_numerical_match_score(unknown_wl, unknown_int, ref_wl, ref_int)
                    if score > 30:
                        parsed = spectrum_data['parsed']
                        match = {'gem_name': gem_name, 'base_gem_id': parsed['base_id'], 'light_source': light_source,
                                'score': score, 'confidence': confidence, 'details': details,
                                'ref_data': {'wavelengths': ref_wl, 'intensities': ref_int}}
                        light_matches.append(match)
                except:
                    continue
            
            light_matches.sort(key=lambda x: x['score'], reverse=True)
            print(f"   Processed {len(ref_spectra)} reference spectra")
            print(f"   Found {len(light_matches)} potential matches for {light_source}")
            
            if light_matches:
                top_match = light_matches[0]
                if top_match['base_gem_id'] == self.selected_base_gem_id:
                    print(f"   ✓ SELF-MATCH FOUND! {top_match['gem_name']} ({top_match['score']:.1f}%)")
                else:
                    print(f"   ✗ Best match: {top_match['gem_name']} (base: {top_match['base_gem_id']}) ({top_match['score']:.1f}%)")
                
                print(f"   Top 5 {light_source} matches:")
                for i, match in enumerate(light_matches[:5], 1):
                    self_indicator = " (SELF)" if match['base_gem_id'] == self.selected_base_gem_id else ""
                    print(f"      {i}. {match['gem_name']}: {match['score']:.1f}%{self_indicator}")
            
            all_matches.extend(light_matches)
        
        if len(all_matches) == 0:
            print("No matches found - analysis failed")
            return False
        
        self.combine_multi_light_results(all_matches)
        return True
    
    def combine_multi_light_results(self, all_matches):
        print(f"\nCOMBINING MULTI-LIGHT RESULTS\n" + "=" * 50)
        
        base_gem_matches = defaultdict(list)
        for match in all_matches:
            base_gem_matches[match['base_gem_id']].append(match)
        
        combined_results = []
        for base_gem_id, matches in base_gem_matches.items():
            light_sources = [m['light_source'] for m in matches]
            scores = [m['score'] for m in matches]
            confidences = [m['confidence'] for m in matches]
            
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            avg_confidence = np.mean(confidences)
            multi_light_bonus = len(set(light_sources)) * 5
            combined_score = min(100, (avg_score * 0.6 + max_score * 0.4) + multi_light_bonus)
            
            combined_result = {
                'base_gem_id': base_gem_id, 'combined_score': combined_score, 'average_score': avg_score,
                'max_score': max_score, 'confidence': avg_confidence, 'light_sources': light_sources,
                'num_light_sources': len(set(light_sources)), 'individual_matches': matches,
                'is_self_match': base_gem_id == self.selected_base_gem_id,
                'sample_gem_names': [m['gem_name'] for m in matches[:3]]
            }
            combined_results.append(combined_result)
        
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        self.match_results = combined_results
        self.display_ranking_results()
        self.create_match_visualizations()
    
    def display_ranking_results(self):
        print(f"\nCOMPLETE RANKING - ALL MATCHES:\n" + "=" * 80)
        if not self.match_results:
            print("No matches found")
            return
        
        print(f"{'Rank':<4} {'Base Gem':<10} {'Score':<8} {'Conf.':<6} {'Lights':<8} {'Best Score':<10} {'Status'}")
        print("-" * 80)
        
        self_match_rank = None
        for i, result in enumerate(self.match_results[:20], 1):
            base_gem_id = result['base_gem_id']
            combined_score = result['combined_score']
            confidence = result['confidence']
            light_sources = '+'.join(sorted(set(result['light_sources'])))
            
            best_match = max(result['individual_matches'], key=lambda x: x['score'])
            best_score = f"{best_match['light_source']}({best_match['score']:.1f}%)"
            
            if result['is_self_match']:
                status = "SELF-MATCH!"
                self_match_rank = i
            elif combined_score >= 90:
                status = "EXCELLENT"
            elif combined_score >= 80:
                status = "VERY GOOD"
            elif combined_score >= 70:
                status = "GOOD"
            else:
                status = "FAIR"
            
            marker = "★" if i <= 5 else " "
            if i <= 5:
                print(f"{marker}{i:<3} {base_gem_id:<10} {combined_score:<7.1f}% {confidence:<5.1f}% {light_sources:<8} {best_score:<10} {status}")
            else:
                print(f"{i:<4} {base_gem_id:<10} {combined_score:<7.1f}% {confidence:<5.1f}% {light_sources:<8} {best_score:<10} {status}")
        
        if len(self.match_results) > 20:
            print(f"... and {len(self.match_results) - 20} more matches")
        
        print(f"\n⭐ TOP 5 DETAILED BREAKDOWN:\n" + "-" * 50)
        top_5 = self.match_results[:min(5, len(self.match_results))]
        for i, result in enumerate(top_5, 1):
            status_icon = "🏆" if i == 1 else f"#{i}"
            self_indicator = " (SELF-MATCH)" if result['is_self_match'] else ""
            print(f"{status_icon} {result['base_gem_id']}: {result['combined_score']:.1f}%{self_indicator}")
            print(f"   Light sources: {'+'.join(sorted(set(result['light_sources'])))}")
            print(f"   Sample spectra: {', '.join(result['sample_gem_names'])}")
            if i < len(top_5):
                print()
        
        print(f"\nTEST RESULTS SUMMARY:\n" + "=" * 50)
        if self_match_rank:
            if self_match_rank == 1:
                print(f"✓ SUCCESS: Base gem {self.selected_base_gem_id} correctly identified as #1 match!")
                print(f"   Combined score: {self.match_results[0]['combined_score']:.1f}%")
                print(f"   System accuracy: EXCELLENT")
            else:
                print(f"⚠ PARTIAL: Base gem {self.selected_base_gem_id} found at rank #{self_match_rank}")
                print(f"   Combined score: {self.match_results[self_match_rank-1]['combined_score']:.1f}%")
                print(f"   System accuracy: NEEDS IMPROVEMENT")
        else:
            print(f"✗ FAILED: Base gem {self.selected_base_gem_id} not found in top matches")
            print(f"   System accuracy: POOR - requires algorithm tuning")
        
        if self.match_results:
            best_match = self.match_results[0]
            print(f"\nBest overall match: Base gem {best_match['base_gem_id']} ({best_match['combined_score']:.1f}%)")
            print(f"Sample spectra: {', '.join(best_match['sample_gem_names'])}")
    
    def create_match_visualizations(self):
        print(f"\nCREATING MATCH VISUALIZATIONS\n" + "=" * 50)
        if len(self.match_results) < 1:
            print("No matches to visualize")
            return
        
        os.makedirs(self.temp_output_dir, exist_ok=True)
        num_matches_to_show = min(5, len(self.match_results))
        top_matches = self.match_results[:num_matches_to_show]
        
        print(f"Will create {num_matches_to_show} individual match visualizations:")
        for i, match in enumerate(top_matches, 1):
            print(f"Creating visualization {i}: Base gem {match['base_gem_id']} ({match['combined_score']:.1f}%)")
            png_file = self.create_single_match_visualization(match, i, self.temp_output_dir)
            if png_file:
                self.generated_files['png_files'].append(png_file)
        
        print(f"Creating summary comparison plot with {num_matches_to_show} matches...")
        summary_png = self.create_summary_comparison_plot(top_matches, self.temp_output_dir)
        if summary_png:
            self.generated_files['png_files'].append(summary_png)
        
        print(f"Visualizations saved to: {self.temp_output_dir}/")
    
    def create_single_match_visualization(self, match, rank, output_dir):
        base_gem_id = match['base_gem_id']
        
        light_matches = {}
        for individual_match in match['individual_matches']:
            light = individual_match['light_source']
            if light not in light_matches:
                light_matches[light] = []
            light_matches[light].append(individual_match)
        
        fig, axes = plt.subplots(len(light_matches), 1, figsize=(12, 4 * len(light_matches)))
        if len(light_matches) == 1:
            axes = [axes]
        
        title = f"Rank #{rank}: Base Gem {base_gem_id} (Score: {match['combined_score']:.1f}%)"
        if match['is_self_match']:
            title += " - SELF-MATCH!"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for idx, (light_source, matches) in enumerate(light_matches.items()):
            ax = axes[idx]
            
            if light_source in self.unknown_data:
                unknown_data = self.unknown_data[light_source]
                ax.plot(unknown_data['wavelengths'], unknown_data['intensities'], 
                       'b-', linewidth=2, label=f'Unknown: {unknown_data["gem_name"]}', alpha=0.8)
            
            best_match = max(matches, key=lambda x: x['score'])
            ref_wl = best_match['ref_data']['wavelengths']
            ref_int = best_match['ref_data']['intensities']
            ax.plot(ref_wl, ref_int, 'r-', linewidth=2, 
                   label=f'Reference: {best_match["gem_name"]}', alpha=0.7)
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'{light_source} Light - Match Score: {best_match["score"]:.1f}%')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            details = best_match['details']
            details_text = (f"Correlation: {details['correlation']:.3f}\nPeak Score: {details['peak_score']:.1f}%\n"
                          f"Shape Score: {details['shape_score']:.1f}%\nOverlap: {details['overlap_percentage']:.1f}%")
            ax.text(0.02, 0.98, details_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        filename = f"match_rank_{rank}_gem_{base_gem_id}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        return filepath
    
    def create_summary_comparison_plot(self, top_matches, output_dir):
        fig, axes = plt.subplots(len(self.unknown_data), 1, figsize=(14, 5 * len(self.unknown_data)))
        if len(self.unknown_data) == 1:
            axes = [axes]
        
        num_matches = len(top_matches)
        fig.suptitle(f"Top {num_matches} Matches vs Unknown: Base Gem {self.selected_base_gem_id}", 
                     fontsize=16, fontweight='bold')
        
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        
        for light_idx, (light_source, unknown_spectrum) in enumerate(self.unknown_data.items()):
            ax = axes[light_idx]
            ax.plot(unknown_spectrum['wavelengths'], unknown_spectrum['intensities'], 
                   'b-', linewidth=3, label=f'Unknown: {unknown_spectrum["gem_name"]}', alpha=0.9)
            
            for rank, match in enumerate(top_matches):
                light_matches = [m for m in match['individual_matches'] if m['light_source'] == light_source]
                if light_matches:
                    best_match = max(light_matches, key=lambda x: x['score'])
                    ref_wl = best_match['ref_data']['wavelengths']
                    ref_int = best_match['ref_data']['intensities']
                    
                    status = " (SELF)" if match['is_self_match'] else ""
                    label = f"#{rank+1}: {match['base_gem_id']} ({match['combined_score']:.1f}%){status}"
                    ax.plot(ref_wl, ref_int, color=colors[rank], linewidth=2, label=label, alpha=0.7)
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'{light_source} Light Source Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"top_{num_matches}_matches_summary.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath
    
    def export_results(self):
        os.makedirs(self.temp_output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.match_results:
            results_data = []
            for rank, result in enumerate(self.match_results, 1):
                for individual_match in result['individual_matches']:
                    row = {
                        'Test_Base_Gem': self.selected_base_gem_id, 'Test_Spectra': '; '.join(self.selected_gem_names),
                        'Rank': rank, 'Match_Base_Gem': result['base_gem_id'], 'Match_Spectrum': individual_match['gem_name'],
                        'Is_Self_Match': result['is_self_match'], 'Combined_Score': result['combined_score'],
                        'Light_Source': individual_match['light_source'], 'Individual_Score': individual_match['score'],
                        'Individual_Confidence': individual_match['confidence'], 'Correlation': individual_match['details']['correlation'],
                        'Peak_Score': individual_match['details']['peak_score'], 'Shape_Score': individual_match['details']['shape_score'],
                        'Overlap_Percentage': individual_match['details']['overlap_percentage']
                    }
                    results_data.append(row)
            
            csv_filename = f"numerical_test_gem_{self.selected_base_gem_id}_{timestamp}.csv"
            csv_filepath = os.path.join(self.temp_output_dir, csv_filename)
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(csv_filepath, index=False)
            self.generated_files['csv_files'].append(csv_filepath)
            print(f"Results exported to: {csv_filepath}")
    
    def transfer_results_to_permanent_storage(self):
        print(f"\nTRANSFERRING RESULTS TO PERMANENT STORAGE\n" + "=" * 50)
        
        os.makedirs(self.final_graphs_dir, exist_ok=True)
        os.makedirs(self.final_reports_dir, exist_ok=True)
        
        transferred_files = {'graphs': [], 'reports': []}
        
        for png_file in self.generated_files['png_files']:
            if os.path.exists(png_file):
                filename = os.path.basename(png_file)
                destination = os.path.join(self.final_graphs_dir, filename)
                try:
                    shutil.move(png_file, destination)
                    transferred_files['graphs'].append(destination)
                    print(f"   📊 Graph: {filename} → {self.final_graphs_dir}")
                except Exception as e:
                    print(f"   ❌ Failed to transfer {filename}: {e}")
        
        for csv_file in self.generated_files['csv_files']:
            if os.path.exists(csv_file):
                filename = os.path.basename(csv_file)
                destination = os.path.join(self.final_reports_dir, filename)
                try:
                    shutil.move(csv_file, destination)
                    transferred_files['reports'].append(destination)
                    print(f"   📄 Report: {filename} → {self.final_reports_dir}")
                except Exception as e:
                    print(f"   ❌ Failed to transfer {filename}: {e}")
        
        total_graphs = len(transferred_files['graphs'])
        total_reports = len(transferred_files['reports'])
        
        print(f"\n📋 TRANSFER SUMMARY:")
        print(f"   Graphs transferred: {total_graphs} (up to 5 individual + 1 summary)")
        print(f"   Reports transferred: {total_reports}")
        print(f"   Final locations:\n      📊 Graphs: {self.final_graphs_dir}\n      📄 Reports: {self.final_reports_dir}")
        
        if total_graphs > 0 or total_reports > 0:
            print(f"\n✅ Results successfully archived to permanent storage!")
            print(f"🧹 Temporary files automatically cleaned up from {self.temp_output_dir}")
        else:
            print(f"\n⚠️ No files were transferred")
        
        return total_graphs + total_reports > 0
    
    def run_complete_analysis(self):
        print("Starting Numerical Analysis Engine...")
        
        if not self.load_reference_database():
            print("Failed to load reference database")
            return False
        if not self.display_available_base_gems():
            return False
        if not self.select_gem_spectra_for_testing():
            print("No gem spectra selected for testing")
            return False
        if not self.extract_selected_gem_data():
            return False
        if not self.analyze_unknown_gem():
            return False
        
        self.export_results()
        transfer_success = self.transfer_results_to_permanent_storage()
        
        print(f"\n" + "=" * 80)
        print("NUMERICAL ANALYSIS COMPLETED SUCCESSFULLY")
        if transfer_success:
            print("RESULTS ARCHIVED TO PERMANENT STORAGE")
        print("=" * 80)
        return True

def main():
    try:
        analyzer = GeminiNumericalAnalyzer()
        success = analyzer.run_complete_analysis()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nCritical error in numerical analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()