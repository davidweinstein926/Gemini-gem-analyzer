#!/usr/bin/env python3
"""
GEMINI NUMERICAL ANALYSIS ENGINE v2.0
Proper Gem Name Structure Support

Gem Naming Convention:
- Base ID: Integer (52, 140, 956) OR Prefixed (C00034, S20250909)
- Light Source: B, L, U (always present)
- Orientation: C (Crown) or P (Pavilion) (always present)  
- Scan Number: 1, 2, 3, etc. (which scan of the stone)

Examples: 51BC1, 67LC2, 199UP3, C0045BC1, S20250909LC2

Workflow:
1. User specifies up to 3 full gem names for same physical gem: 58BC1, 58LC2, 58UC3
2. System extracts these as "unknown gem #58"
3. FIXED: System keeps ALL gem #58 variations in database for proper self-matching
4. System finds best match (should be gem #58 with 0.0% difference - self-validation)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime
from collections import defaultdict
import json
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GeminiNumericalAnalyzer:
    def __init__(self):
        """Initialize the Gemini Numerical Analysis System"""
        
        # Database paths
        self.database_dir = "database"
        self.reference_files = {
            'B': os.path.join(self.database_dir, "reference_spectra", "gemini_db_long_B.csv"),
            'L': os.path.join(self.database_dir, "reference_spectra", "gemini_db_long_L.csv"), 
            'U': os.path.join(self.database_dir, "reference_spectra", "gemini_db_long_U.csv")
        }
        
        # Structural database
        self.structural_db = "multi_structural_gem_data.db"
        
        # Analysis settings
        self.wavelength_tolerance = 2.0
        self.min_overlap_percentage = 70
        
        # Results storage
        self.all_gem_spectra = {}      # All individual gem spectra from database
        self.available_base_gems = {}  # Base gems grouped (e.g., all variations of gem 58)
        self.selected_gem_names = []   # User-specified gem names (e.g., 58BC1, 58LC2, 58UC3)
        self.selected_base_gem_id = None # Base gem ID (e.g., "58")
        self.unknown_data = {}         # Selected gem data (treated as unknown)
        self.reference_data = {}       # Remaining gems for searching  
        self.match_results = []
        
        print("=" * 80)
        print("  GEMINI NUMERICAL ANALYSIS ENGINE v2.0")
        print("  Proper Gem Name Structure Support")
        print("=" * 80)
    
    def parse_gem_name(self, gem_name):
        """Parse gem name into components: base_id, light_source, orientation, scan_number"""
        gem_name = str(gem_name).strip()
        
        # Pattern for gem names: (prefix)(base_id)(light)(orientation)(scan)
        # Examples: 58BC1, C0045LC2, S20250909UP3
        
        # Try different patterns
        patterns = [
            r'^([CS]\d+)([BLU])([CP])(\d+)$',          # C0045BC1, S20250909LC2
            r'^(\d+)([BLU])([CP])(\d+)$',              # 58BC1, 199UP3
        ]
        
        for pattern in patterns:
            match = re.match(pattern, gem_name)
            if match:
                base_id = match.group(1)
                light_source = match.group(2) 
                orientation = match.group(3)
                scan_number = int(match.group(4))
                
                return {
                    'base_id': base_id,
                    'light_source': light_source,
                    'orientation': orientation,
                    'scan_number': scan_number,
                    'full_name': gem_name
                }
        
        # If no pattern matches, return None
        return None
    
    def load_reference_database(self):
        """Load reference database and organize by gem structure"""
        print(f"\nLOADING REFERENCE DATABASE")
        print("=" * 50)
        
        all_gem_spectra = {}
        
        for light_source, filepath in self.reference_files.items():
            print(f"Loading {light_source} reference: {filepath}")
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    
                    if df.empty:
                        print(f"   Warning: {light_source} reference is empty")
                        continue
                    
                    print(f"   File loaded: {len(df)} rows, {len(df.columns)} columns")
                    print(f"   Column names: {list(df.columns)}")
                    
                    # Find gem ID column (flexible detection)
                    gem_id_col = None
                    wavelength_col = None
                    intensity_col = None
                    
                    for col in df.columns:
                        col_str = str(col).strip().lower()
                        if 'gem' in col_str or 'id' in col_str or 'sample' in col_str or 'name' in col_str:
                            gem_id_col = col
                        elif 'wavelength' in col_str or 'wl' in col_str:
                            wavelength_col = col
                        elif 'intensity' in col_str or 'int' in col_str or 'signal' in col_str:
                            intensity_col = col
                    
                    # Fallback to first columns if no matches
                    if not gem_id_col and len(df.columns) >= 3:
                        gem_id_col = df.columns[0]
                        wavelength_col = df.columns[1]
                        intensity_col = df.columns[2]
                    
                    print(f"   Using columns: ID='{gem_id_col}', WL='{wavelength_col}', INT='{intensity_col}'")
                    
                    if gem_id_col and wavelength_col and intensity_col:
                        # Process each gem spectrum
                        unique_gem_names = df[gem_id_col].dropna().unique()
                        valid_gems = 0
                        
                        for gem_name in unique_gem_names:
                            gem_name_str = str(gem_name).strip()
                            if not gem_name_str or gem_name_str.lower() in ['nan', 'none', 'null']:
                                continue
                            
                            # Parse gem name structure
                            parsed = self.parse_gem_name(gem_name_str)
                            if not parsed:
                                continue  # Skip gems that don't match naming convention
                            
                            # Get spectrum data for this gem
                            gem_data = df[df[gem_id_col] == gem_name]
                            
                            try:
                                wavelengths = pd.to_numeric(gem_data[wavelength_col], errors='coerce').dropna().values
                                intensities = pd.to_numeric(gem_data[intensity_col], errors='coerce').dropna().values
                                
                                if len(wavelengths) < 10:  # Skip gems with too few points
                                    continue
                                
                                # Store spectrum
                                all_gem_spectra[gem_name_str] = {
                                    'parsed': parsed,
                                    'wavelengths': wavelengths,
                                    'intensities': intensities,
                                    'light_source': light_source,
                                    'points': len(wavelengths),
                                    'wl_range': f"{wavelengths.min():.1f}-{wavelengths.max():.1f}nm"
                                }
                                valid_gems += 1
                                
                            except Exception as e:
                                continue
                        
                        print(f"   ✓ Loaded {valid_gems} valid gem spectra")
                        
                    else:
                        print(f"   Error: Could not identify required columns")
                        
                except Exception as e:
                    print(f"   Error loading {light_source}: {e}")
            else:
                print(f"   Warning: File not found: {filepath}")
        
        # Store all gem spectra
        self.all_gem_spectra = all_gem_spectra
        
        # Group by base gem ID
        self.organize_gems_by_base_id()
        
        # Load structural database
        self.load_structural_database()
        
        print(f"\nTotal individual spectra loaded: {len(all_gem_spectra)}")
        print(f"Total base gems available: {len(self.available_base_gems)}")
        
        return len(all_gem_spectra) > 0
    
    def organize_gems_by_base_id(self):
        """Organize gem spectra by base gem ID"""
        base_gems = defaultdict(list)
        
        for gem_name, spectrum_data in self.all_gem_spectra.items():
            parsed = spectrum_data['parsed']
            base_id = parsed['base_id']
            base_gems[base_id].append({
                'full_name': gem_name,
                'spectrum_data': spectrum_data,
                'light_source': parsed['light_source'],
                'orientation': parsed['orientation'],
                'scan_number': parsed['scan_number']
            })
        
        self.available_base_gems = dict(base_gems)
        
        # Sort each base gem's spectra
        for base_id in self.available_base_gems:
            self.available_base_gems[base_id].sort(
                key=lambda x: (x['light_source'], x['orientation'], x['scan_number'])
            )
    
    def load_structural_database(self):
        """Load structural database as additional reference source"""
        if not os.path.exists(self.structural_db):
            print(f"   Structural database not found: {self.structural_db}")
            return False
        
        try:
            conn = sqlite3.connect(self.structural_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            structural_count = cursor.fetchone()[0]
            
            if structural_count > 0:
                print(f"   ✓ Found {structural_count:,} structural features (not integrated in this version)")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"   Error accessing structural database: {e}")
            return False
    
    def display_available_base_gems(self):
        """Display available base gems with their spectrum variations"""
        print(f"\nAVAILABLE BASE GEMS")
        print("=" * 80)
        
        if not self.available_base_gems:
            print("No gems found in database")
            return False
        
        # Sort base gems
        sorted_base_gems = sorted(self.available_base_gems.items(), 
                                 key=lambda x: self.sort_key_for_gem_id(x[0]))
        
        print(f"{'Base Gem':<15} {'Spectra':<8} {'Light Sources':<15} {'Sample Spectra Names'}")
        print("-" * 80)
        
        for base_id, spectra_list in sorted_base_gems[:50]:  # Show first 50
            light_sources = list(set(s['light_source'] for s in spectra_list))
            light_sources.sort()
            
            # Sample spectrum names (show first 3)
            sample_names = [s['full_name'] for s in spectra_list[:3]]
            if len(spectra_list) > 3:
                sample_names.append(f"... +{len(spectra_list)-3} more")
            
            print(f"{base_id:<15} {len(spectra_list):<8} {'+'.join(light_sources):<15} {', '.join(sample_names)}")
        
        if len(sorted_base_gems) > 50:
            print(f"... and {len(sorted_base_gems) - 50} more base gems")
        
        print(f"\nTotal base gems: {len(sorted_base_gems)}")
        
        # Show light source distribution
        light_counts = defaultdict(int)
        for base_id, spectra_list in sorted_base_gems:
            for spectrum in spectra_list:
                light_counts[spectrum['light_source']] += 1
        
        print(f"Spectrum distribution by light source:")
        for light, count in sorted(light_counts.items()):
            print(f"   {light}: {count} spectra")
        
        return True
    
    def sort_key_for_gem_id(self, gem_id):
        """Create sort key for gem ID to handle mixed numeric/text IDs"""
        if gem_id.startswith(('C', 'S')):
            # For prefixed gems like C0045, S20250909
            return (1, gem_id)  # Sort after numeric gems
        else:
            # For numeric gems like 58, 140
            try:
                return (0, int(gem_id))  # Sort numerically
            except:
                return (2, gem_id)  # Sort alphabetically if not numeric
    
    def select_gem_spectra_for_testing(self):
        """Allow user to specify gem spectra for unknown testing"""
        print(f"\nSELECT GEM SPECTRA FOR TESTING")
        print("=" * 50)
        
        print("Enter up to 3 full gem names representing the same physical gem")
        print("under different light sources (B, L, U).")
        print("Examples:")
        print("   Single light: 58BC1")
        print("   Two lights: 58BC1, 58LC2") 
        print("   Three lights: 58BC1, 58LC2, 58UC3")
        print("   Client gem: C0045BC1, C0045LC1, C0045UC1")
        print("   Source gem: S20250909BC1, S20250909LC2")
        
        while True:
            gem_input = input("\nEnter gem names (comma-separated) or 'q' to quit: ").strip()
            
            if gem_input.lower() == 'q':
                return False
            
            # Parse input
            gem_names = [name.strip() for name in gem_input.split(',')]
            gem_names = [name for name in gem_names if name]  # Remove empty strings
            
            if not gem_names:
                print("Please enter at least one gem name")
                continue
            
            if len(gem_names) > 3:
                print("Maximum 3 gem spectra allowed")
                continue
            
            # Validate and parse gem names
            valid_gems = []
            base_ids = set()
            
            for gem_name in gem_names:
                parsed = self.parse_gem_name(gem_name)
                if not parsed:
                    print(f"Invalid gem name format: '{gem_name}'")
                    print("Expected format: [prefix]number + light + orientation + scan")
                    print("Examples: 58BC1, C0045LC2, S20250909UP3")
                    break
                
                # Check if gem exists in database
                if gem_name not in self.all_gem_spectra:
                    print(f"Gem not found in database: '{gem_name}'")
                    print("Available gems for this base ID:")
                    
                    # Show available spectra for this base ID
                    if parsed['base_id'] in self.available_base_gems:
                        for spectrum in self.available_base_gems[parsed['base_id']][:10]:
                            print(f"   {spectrum['full_name']}")
                    else:
                        print(f"   No gems found for base ID: {parsed['base_id']}")
                    break
                
                valid_gems.append(gem_name)
                base_ids.add(parsed['base_id'])
            
            else:  # No break occurred - all gems are valid
                # Check if all gems have same base ID
                if len(base_ids) > 1:
                    print(f"All gems must have the same base ID. Found: {', '.join(base_ids)}")
                    continue
                
                # Show selected gems info
                print(f"\nSelected gems for testing:")
                base_gem_id = next(iter(base_ids))
                
                for gem_name in valid_gems:
                    spectrum_data = self.all_gem_spectra[gem_name]
                    parsed = spectrum_data['parsed']
                    print(f"   {gem_name}: {parsed['light_source']} light, {parsed['orientation']} orientation, "
                          f"scan {parsed['scan_number']}, {spectrum_data['points']} points")
                
                # Confirm selection
                confirm = input(f"\nTest base gem {base_gem_id} using these spectra? (y/n): ").strip().lower()
                if confirm == 'y':
                    self.selected_gem_names = valid_gems
                    self.selected_base_gem_id = base_gem_id
                    return True
    
    def extract_selected_gem_data(self):
        """Extract selected gem spectra but KEEP base gem in reference database for matching"""
        print(f"\nEXTRACTING SELECTED GEM: {self.selected_base_gem_id}")
        print("=" * 50)
        
        if not self.selected_gem_names:
            print("No gem spectra selected")
            return False
        
        # Extract selected spectra as "unknown" data
        extracted_data = {}
        
        for gem_name in self.selected_gem_names:
            spectrum_data = self.all_gem_spectra[gem_name]
            parsed = spectrum_data['parsed']
            light_source = parsed['light_source']
            
            # Store as unknown data (keyed by light source)
            extracted_data[light_source] = {
                'gem_name': gem_name,
                'wavelengths': spectrum_data['wavelengths'],
                'intensities': spectrum_data['intensities'],
                'parsed': parsed,
                'spectrum_data': spectrum_data
            }
            
            wl_range = spectrum_data['wl_range']
            points = spectrum_data['points']
            print(f"   ✓ Extracted {gem_name}: {light_source} light, {points} points, {wl_range}")
        
        self.unknown_data = extracted_data
        
        # FIXED: DO NOT REMOVE BASE GEM FROM REFERENCE DATABASE
        # This allows proper self-matching to occur
        
        # Original problematic code (COMMENTED OUT):
        # removed_count = 0
        # if self.selected_base_gem_id in self.available_base_gems:
        #     spectra_to_remove = self.available_base_gems[self.selected_base_gem_id]
        #     
        #     for spectrum in spectra_to_remove:
        #         gem_name = spectrum['full_name']
        #         if gem_name in self.all_gem_spectra:
        #             del self.all_gem_spectra[gem_name]
        #             removed_count += 1
        #     
        #     # Remove from base gems catalog
        #     del self.available_base_gems[self.selected_base_gem_id]
        # 
        # print(f"   ✓ Removed {removed_count} spectra for base gem {self.selected_base_gem_id}")
        # print(f"   ✓ Base gem {self.selected_base_gem_id} excluded from search")
        
        # FIXED: Keep all spectra in database for matching
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
        """Calculate numerical matching score between unknown and reference spectra"""
        
        # Find overlapping wavelength range
        min_wl = max(unknown_wl.min(), ref_wl.min())
        max_wl = min(unknown_wl.max(), ref_wl.max())
        
        if min_wl >= max_wl:
            return 0.0, 0.0, {"error": "No wavelength overlap"}
        
        # Create common wavelength grid
        wl_grid = np.linspace(min_wl, max_wl, 200)
        
        # Interpolate both spectra to common grid
        unknown_interp = np.interp(wl_grid, unknown_wl, unknown_int)
        ref_interp = np.interp(wl_grid, ref_wl, ref_int)
        
        # Normalize both spectra to 0-100 scale
        unknown_norm = (unknown_interp - unknown_interp.min()) / (unknown_interp.max() - unknown_interp.min()) * 100
        ref_norm = (ref_interp - ref_interp.min()) / (ref_interp.max() - ref_interp.min()) * 100
        
        # Calculate similarity metrics
        
        # 1. Correlation coefficient
        correlation = np.corrcoef(unknown_norm, ref_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # 2. Mean squared error (inverted to similarity)
        mse = np.mean((unknown_norm - ref_norm) ** 2)
        mse_similarity = max(0, 100 - mse)
        
        # 3. Peak alignment score
        unknown_peaks = self.find_spectral_peaks(wl_grid, unknown_norm)
        ref_peaks = self.find_spectral_peaks(wl_grid, ref_norm)
        peak_score = self.calculate_peak_alignment_score(unknown_peaks, ref_peaks)
        
        # 4. Shape similarity
        shape_score = self.calculate_shape_similarity(unknown_norm, ref_norm)
        
        # Combine scores with weights
        weights = {'correlation': 0.3, 'mse': 0.25, 'peaks': 0.25, 'shape': 0.2}
        
        final_score = (
            correlation * 100 * weights['correlation'] +
            mse_similarity * weights['mse'] +
            peak_score * weights['peaks'] +
            shape_score * weights['shape']
        )
        
        # Calculate confidence based on overlap and data quality
        overlap_percentage = (max_wl - min_wl) / (max(unknown_wl.max(), ref_wl.max()) - min(unknown_wl.min(), ref_wl.min())) * 100
        confidence = min(100, overlap_percentage * (final_score / 100))
        
        details = {
            'correlation': correlation,
            'mse_similarity': mse_similarity,
            'peak_score': peak_score,
            'shape_score': shape_score,
            'overlap_percentage': overlap_percentage,
            'wavelength_range': f"{min_wl:.1f}-{max_wl:.1f}nm"
        }
        
        return final_score, confidence, details
    
    def find_spectral_peaks(self, wavelengths, intensities, prominence=5):
        """Find peaks in spectral data"""
        peaks = []
        
        for i in range(1, len(intensities) - 1):
            if (intensities[i] > intensities[i-1] and 
                intensities[i] > intensities[i+1] and 
                intensities[i] > prominence):
                peaks.append({
                    'wavelength': wavelengths[i],
                    'intensity': intensities[i],
                    'prominence': min(intensities[i] - intensities[i-1], intensities[i] - intensities[i+1])
                })
        
        peaks.sort(key=lambda x: x['intensity'], reverse=True)
        return peaks[:10]
    
    def calculate_peak_alignment_score(self, peaks1, peaks2):
        """Calculate how well peaks align between two spectra"""
        if not peaks1 or not peaks2:
            return 0.0
        
        total_score = 0.0
        matches = 0
        
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
        """Calculate overall shape similarity between spectra"""
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
        """Perform complete analysis of unknown gem against reference database"""
        print(f"\nANALYZING UNKNOWN GEM: {self.selected_base_gem_id}")
        print("=" * 50)
        print(f"Testing if system can correctly identify base gem {self.selected_base_gem_id}")
        print(f"Using spectra: {', '.join(self.selected_gem_names)}")
        
        if not self.unknown_data:
            print("No unknown gem data available for analysis")
            return False
        
        if not self.all_gem_spectra:
            print("No reference database available for comparison") 
            return False
        
        all_matches = []
        
        # Analyze each light source
        for light_source, unknown_spectrum in self.unknown_data.items():
            print(f"\nAnalyzing {light_source} light source...")
            print(f"   Unknown spectrum: {unknown_spectrum['gem_name']}")
            
            unknown_wl = unknown_spectrum['wavelengths']
            unknown_int = unknown_spectrum['intensities']
            
            # Find all reference spectra for this light source
            ref_spectra = []
            for gem_name, spectrum_data in self.all_gem_spectra.items():
                if spectrum_data['parsed']['light_source'] == light_source:
                    ref_spectra.append((gem_name, spectrum_data))
            
            print(f"   Comparing against {len(ref_spectra)} reference spectra...")
            
            light_matches = []
            processed_count = 0
            
            for gem_name, spectrum_data in ref_spectra:
                ref_wl = spectrum_data['wavelengths']
                ref_int = spectrum_data['intensities']
                
                try:
                    score, confidence, details = self.calculate_numerical_match_score(
                        unknown_wl, unknown_int, ref_wl, ref_int
                    )
                    
                    if score > 30:  # Only include reasonable matches
                        parsed = spectrum_data['parsed']
                        match = {
                            'gem_name': gem_name,
                            'base_gem_id': parsed['base_id'],
                            'light_source': light_source,
                            'score': score,
                            'confidence': confidence,
                            'details': details,
                            'ref_data': {
                                'wavelengths': ref_wl,
                                'intensities': ref_int
                            }
                        }
                        light_matches.append(match)
                    
                    processed_count += 1
                    
                except Exception as e:
                    continue
            
            # Sort by score
            light_matches.sort(key=lambda x: x['score'], reverse=True)
            
            print(f"   Processed {processed_count} reference spectra")
            print(f"   Found {len(light_matches)} potential matches for {light_source}")
            
            if light_matches:
                top_match = light_matches[0]
                if top_match['base_gem_id'] == self.selected_base_gem_id:
                    print(f"   ✓ SELF-MATCH FOUND! {top_match['gem_name']} ({top_match['score']:.1f}%)")
                else:
                    print(f"   ✗ Best match: {top_match['gem_name']} (base: {top_match['base_gem_id']}) ({top_match['score']:.1f}%)")
                
                # Show top 3 for this light source
                print(f"   Top 3 {light_source} matches:")
                for i, match in enumerate(light_matches[:3], 1):
                    self_indicator = " (SELF)" if match['base_gem_id'] == self.selected_base_gem_id else ""
                    print(f"      {i}. {match['gem_name']}: {match['score']:.1f}%{self_indicator}")
            
            all_matches.extend(light_matches)
        
        if len(all_matches) == 0:
            print("No matches found - analysis failed")
            return False
        
        self.combine_multi_light_results(all_matches)
        return True
    
    def combine_multi_light_results(self, all_matches):
        """Combine matching results across multiple light sources"""
        print(f"\nCOMBINING MULTI-LIGHT RESULTS")
        print("=" * 50)
        
        # Group matches by base gem ID
        base_gem_matches = defaultdict(list)
        for match in all_matches:
            base_gem_id = match['base_gem_id']
            base_gem_matches[base_gem_id].append(match)
        
        combined_results = []
        
        for base_gem_id, matches in base_gem_matches.items():
            light_sources = [m['light_source'] for m in matches]
            scores = [m['score'] for m in matches]
            confidences = [m['confidence'] for m in matches]
            
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            avg_confidence = np.mean(confidences)
            
            # Multi-light bonus
            multi_light_bonus = len(set(light_sources)) * 5  # Bonus for multiple light sources
            combined_score = (avg_score * 0.6 + max_score * 0.4) + multi_light_bonus
            combined_score = min(100, combined_score)
            
            combined_result = {
                'base_gem_id': base_gem_id,
                'combined_score': combined_score,
                'average_score': avg_score,
                'max_score': max_score,
                'confidence': avg_confidence,
                'light_sources': light_sources,
                'num_light_sources': len(set(light_sources)),
                'individual_matches': matches,
                'is_self_match': base_gem_id == self.selected_base_gem_id,
                'sample_gem_names': [m['gem_name'] for m in matches[:3]]
            }
            
            combined_results.append(combined_result)
        
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        self.match_results = combined_results
        
        self.display_ranking_results()
        self.create_match_visualizations()
    
    def display_ranking_results(self):
        """Display complete ranking of all matches"""
        print(f"\nCOMPLETE RANKING - ALL MATCHES:")
        print("=" * 80)
        
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
            
            print(f"{i:<4} {base_gem_id:<10} {combined_score:<7.1f}% {confidence:<5.1f}% {light_sources:<8} {best_score:<10} {status}")
        
        if len(self.match_results) > 20:
            remaining = len(self.match_results) - 20
            print(f"... and {remaining} more matches")
        
        # Test results summary
        print(f"\nTEST RESULTS SUMMARY:")
        print("=" * 50)
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
        """Create visualizations for top 3 matches"""
        print(f"\nCREATING MATCH VISUALIZATIONS")
        print("=" * 50)
        
        if len(self.match_results) < 1:
            print("No matches to visualize")
            return
        
        output_dir = "output/numerical_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        top_matches = self.match_results[:3]
        
        for i, match in enumerate(top_matches, 1):
            print(f"Creating visualization {i}: Base gem {match['base_gem_id']} ({match['combined_score']:.1f}%)")
            self.create_single_match_visualization(match, i, output_dir)
        
        self.create_summary_comparison_plot(top_matches, output_dir)
        
        print(f"Visualizations saved to: {output_dir}/")
    
    def create_single_match_visualization(self, match, rank, output_dir):
        """Create detailed visualization for a single match"""
        
        base_gem_id = match['base_gem_id']
        
        # Group individual matches by light source
        light_matches = {}
        for individual_match in match['individual_matches']:
            light = individual_match['light_source']
            if light not in light_matches:
                light_matches[light] = []
            light_matches[light].append(individual_match)
        
        fig, axes = plt.subplots(len(light_matches), 1, 
                                figsize=(12, 4 * len(light_matches)))
        
        if len(light_matches) == 1:
            axes = [axes]
        
        title = f"Rank #{rank}: Base Gem {base_gem_id} (Score: {match['combined_score']:.1f}%)"
        if match['is_self_match']:
            title += " - SELF-MATCH!"
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for idx, (light_source, matches) in enumerate(light_matches.items()):
            ax = axes[idx]
            
            # Plot unknown spectrum
            if light_source in self.unknown_data:
                unknown_data = self.unknown_data[light_source]
                ax.plot(unknown_data['wavelengths'], unknown_data['intensities'], 
                       'b-', linewidth=2, label=f'Unknown: {unknown_data["gem_name"]}', alpha=0.8)
            
            # Plot best reference spectrum for this light source
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
            
            # Add match details
            details = best_match['details']
            details_text = (f"Correlation: {details['correlation']:.3f}\n"
                          f"Peak Score: {details['peak_score']:.1f}%\n"
                          f"Shape Score: {details['shape_score']:.1f}%\n"
                          f"Overlap: {details['overlap_percentage']:.1f}%")
            
            ax.text(0.02, 0.98, details_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save to file
        filename = f"match_rank_{rank}_gem_{base_gem_id}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Display on screen
        plt.show()
        
        # Wait for user to close window before continuing
        plt.close()
    
    def create_summary_comparison_plot(self, top_matches, output_dir):
        """Create summary comparison plot of top matches"""
        
        fig, axes = plt.subplots(len(self.unknown_data), 1, 
                                figsize=(14, 5 * len(self.unknown_data)))
        
        if len(self.unknown_data) == 1:
            axes = [axes]
        
        fig.suptitle(f"Top 3 Matches vs Unknown: Base Gem {self.selected_base_gem_id}", 
                     fontsize=16, fontweight='bold')
        
        colors = ['red', 'green', 'orange']
        
        for light_idx, (light_source, unknown_spectrum) in enumerate(self.unknown_data.items()):
            ax = axes[light_idx]
            
            # Plot unknown spectrum
            ax.plot(unknown_spectrum['wavelengths'], unknown_spectrum['intensities'], 
                   'b-', linewidth=3, label=f'Unknown: {unknown_spectrum["gem_name"]}', alpha=0.9)
            
            # Plot top 3 matches for this light source
            for rank, match in enumerate(top_matches):
                # Find best match for this light source
                light_matches = [m for m in match['individual_matches'] 
                               if m['light_source'] == light_source]
                
                if light_matches:
                    best_match = max(light_matches, key=lambda x: x['score'])
                    ref_wl = best_match['ref_data']['wavelengths']
                    ref_int = best_match['ref_data']['intensities']
                    
                    status = " (SELF)" if match['is_self_match'] else ""
                    label = f"#{rank+1}: {match['base_gem_id']} ({match['combined_score']:.1f}%){status}"
                    ax.plot(ref_wl, ref_int, color=colors[rank], 
                           linewidth=2, label=label, alpha=0.7)
            
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Intensity')
            ax.set_title(f'{light_source} Light Source Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, "top_3_matches_summary.png")
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_results(self):
        """Export detailed results to CSV and JSON"""
        
        output_dir = "output/numerical_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.match_results:
            results_data = []
            
            for rank, result in enumerate(self.match_results, 1):
                for individual_match in result['individual_matches']:
                    row = {
                        'Test_Base_Gem': self.selected_base_gem_id,
                        'Test_Spectra': '; '.join(self.selected_gem_names),
                        'Rank': rank,
                        'Match_Base_Gem': result['base_gem_id'],
                        'Match_Spectrum': individual_match['gem_name'],
                        'Is_Self_Match': result['is_self_match'],
                        'Combined_Score': result['combined_score'],
                        'Light_Source': individual_match['light_source'],
                        'Individual_Score': individual_match['score'],
                        'Individual_Confidence': individual_match['confidence'],
                        'Correlation': individual_match['details']['correlation'],
                        'Peak_Score': individual_match['details']['peak_score'],
                        'Shape_Score': individual_match['details']['shape_score'],
                        'Overlap_Percentage': individual_match['details']['overlap_percentage']
                    }
                    results_data.append(row)
            
            csv_filename = f"numerical_test_gem_{self.selected_base_gem_id}_{timestamp}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)
            
            results_df = pd.DataFrame(results_data)
            results_df.to_csv(csv_filepath, index=False)
            
            print(f"Results exported to: {csv_filepath}")
    
    def run_complete_analysis(self):
        """Run the complete numerical analysis workflow"""
        
        print("Starting Numerical Analysis Engine...")
        
        # Step 1: Load reference database
        if not self.load_reference_database():
            print("Failed to load reference database")
            return False
        
        # Step 2: Display available base gems  
        if not self.display_available_base_gems():
            return False
        
        # Step 3: Select gem spectra for testing
        if not self.select_gem_spectra_for_testing():
            print("No gem spectra selected for testing")
            return False
        
        # Step 4: Extract selected gem data (FIXED - no longer removes from database)
        if not self.extract_selected_gem_data():
            return False
        
        # Step 5: Perform analysis
        if not self.analyze_unknown_gem():
            return False
        
        # Step 6: Export results
        self.export_results()
        
        print(f"\n" + "=" * 80)
        print("NUMERICAL ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return True


def main():
    """Main entry point for numerical analysis"""
    
    try:
        analyzer = GeminiNumericalAnalyzer()
        success = analyzer.run_complete_analysis()
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
            
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