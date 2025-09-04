
#!/usr/bin/env python3
"""
SPECTRAL CURVE PLOTTER
Creates individual spectral curve comparisons for top matches

Author: David
Version: 2024.08.06
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Tuple, Optional
from config.settings import get_config

class SpectralCurvePlotter:
    """Creates individual spectral curve plots from raw data files"""
    
    def __init__(self):
        self.config = get_config('system')
        self.raw_data_dir = r"c:\users\david\gemini sp10 raw"
        self.line_thickness = 0.5
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        print("📈 Spectral curve plotter initialized")
    
    def load_spectral_data(self, stone_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load raw spectral data for a stone ID"""
        
        # Try different file extensions and formats
        possible_files = [
            f"{stone_id}.txt",
            f"{stone_id}_spectrum.txt",
            f"{stone_id}_raw.txt"
        ]
        
        for filename in possible_files:
            filepath = os.path.join(self.raw_data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    # Try to read the file
                    print(f"   📄 Loading spectral data: {filename}")
                    
                    # Read as space or tab delimited
                    data = pd.read_csv(filepath, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
                    
                    if len(data) > 0:
                        wavelengths = data['wavelength'].values
                        intensities = data['intensity'].values
                        
                        print(f"   ✅ Loaded {len(wavelengths)} data points ({wavelengths.min():.1f}-{wavelengths.max():.1f}nm)")
                        return wavelengths, intensities
                    
                except Exception as e:
                    print(f"   ⚠️ Error reading {filename}: {e}")
                    continue
        
        print(f"   ❌ No spectral data found for {stone_id}")
        return None
    
    def normalize_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize spectrum for comparison"""
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(intensities) & np.isfinite(wavelengths)
        clean_wavelengths = wavelengths[valid_mask]
        clean_intensities = intensities[valid_mask]
        
        if len(clean_intensities) == 0:
            return wavelengths, intensities
        
        # Normalize to 0-1 range
        min_intensity = np.min(clean_intensities)
        max_intensity = np.max(clean_intensities)
        
        if max_intensity != min_intensity:
            normalized_intensities = (clean_intensities - min_intensity) / (max_intensity - min_intensity)
        else:
            normalized_intensities = clean_intensities
        
        return clean_wavelengths, normalized_intensities
    
    def create_spectral_comparison(self, unknown_stone_id: str, unknown_features: List[Dict], 
                                  best_match_id: str, confidence: float) -> bool:
        """Create single overlaid spectral curve for unknown vs best match"""
        
        print(f"📈 Creating spectral curve comparison:")
        print(f"   🔍 Unknown: {unknown_stone_id}")
        print(f"   🎯 Best Match: {best_match_id} ({confidence:.1f}%)")
        
        # Load spectral data
        unknown_data = self.load_spectral_data(unknown_stone_id)
        match_data = self.load_spectral_data(best_match_id)
        
        if unknown_data is None and match_data is None:
            print("❌ No spectral data available for either stone")
            return False
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot unknown stone spectrum
        if unknown_data is not None:
            unknown_wavelengths, unknown_intensities = unknown_data
            norm_unknown_wav, norm_unknown_int = self.normalize_spectrum(unknown_wavelengths, unknown_intensities)
            
            ax.plot(norm_unknown_wav, norm_unknown_int, 
                   color='black', linewidth=2.0, label=f'{unknown_stone_id} (Unknown)', 
                   alpha=0.8, zorder=5)
        
        # Plot best match spectrum
        if match_data is not None:
            match_wavelengths, match_intensities = match_data
            norm_match_wav, norm_match_int = self.normalize_spectrum(match_wavelengths, match_intensities)
            
            ax.plot(norm_match_wav, norm_match_int, 
                   color='red', linewidth=1.5, label=f'{best_match_id} (Match: {confidence:.1f}%)', 
                   alpha=0.7, zorder=4)
        
        # Add feature markers from structural analysis
        self._add_feature_markers(ax, unknown_features, unknown_stone_id)
        
        # Formatting
        ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
        ax.set_title(f'Spectral Curve Comparison: {unknown_stone_id} vs {best_match_id}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Set reasonable wavelength range
        if unknown_data is not None or match_data is not None:
            all_wavelengths = []
            if unknown_data is not None:
                all_wavelengths.extend(unknown_data[0])
            if match_data is not None:
                all_wavelengths.extend(match_data[0])
            
            if all_wavelengths:
                min_wav = np.min(all_wavelengths)
                max_wav = np.max(all_wavelengths)
                margin = (max_wav - min_wav) * 0.02
                ax.set_xlim(min_wav - margin, max_wav + margin)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Spectral curve comparison displayed")
        return True
    
    def create_individual_comparisons(self, unknown_stone_id: str, unknown_features: List[Dict], 
                                    top_matches: List, max_matches: int = 6) -> bool:
        """Create 6 separate individual comparison graphs"""
        
        matches_to_plot = top_matches[:max_matches]
        
        print(f"📈 Creating {len(matches_to_plot)} individual spectral comparisons:")
        print(f"   🔍 Unknown: {unknown_stone_id}")
        
        # Load unknown stone data first
        unknown_data = self.load_spectral_data(unknown_stone_id)
        if unknown_data is None:
            print("❌ No spectral data available for unknown stone")
            return False
        
        unknown_wavelengths, unknown_intensities = unknown_data
        norm_unknown_wav, norm_unknown_int = self.normalize_spectrum(unknown_wavelengths, unknown_intensities)
        
        # Create individual plots for each match
        plots_created = 0
        
        for i, match in enumerate(matches_to_plot):
            print(f"   📈 Creating comparison {i+1}/{len(matches_to_plot)}: {match.stone_reference}")
            
            # Load match data
            match_data = self.load_spectral_data(match.stone_reference)
            
            if match_data is not None:
                match_wavelengths, match_intensities = match_data
                norm_match_wav, norm_match_int = self.normalize_spectrum(match_wavelengths, match_intensities)
                
                # Create individual figure for this comparison
                fig, ax = plt.subplots(1, 1, figsize=(14, 8))
                
                # Plot unknown stone (thick black line)
                ax.plot(norm_unknown_wav, norm_unknown_int, 
                       color='black', linewidth=2.0, label=f'{unknown_stone_id} (Unknown)', 
                       alpha=0.8, zorder=5)
                
                # Plot this match (colored line)
                color = self.colors[i % len(self.colors)]
                ax.plot(norm_match_wav, norm_match_int, 
                       color=color, linewidth=self.line_thickness, 
                       label=f'{match.stone_reference} ({match.overall_confidence:.1f}%)', 
                       alpha=0.7, zorder=4)
                
                # Add feature markers
                self._add_feature_markers(ax, unknown_features, unknown_stone_id)
                
                # Formatting
                ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
                ax.set_title(f'Comparison #{i+1}: {unknown_stone_id} vs {match.stone_reference} ({match.overall_confidence:.1f}%)', 
                            fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=11)
                
                # Set wavelength range
                all_wavelengths = list(unknown_wavelengths) + list(match_wavelengths)
                min_wav = np.min(all_wavelengths)
                max_wav = np.max(all_wavelengths)
                margin = (max_wav - min_wav) * 0.02
                ax.set_xlim(min_wav - margin, max_wav + margin)
                
                plt.tight_layout()
                plt.show()
                
                plots_created += 1
                
                # Brief pause between plots for user to view
                if i < len(matches_to_plot) - 1:  # Don't pause after the last plot
                    input(f"   📊 Press Enter to see comparison {i+2}...")
            
            else:
                print(f"   ⚠️ No spectral data found for {match.stone_reference}")
        
        if plots_created == 0:
            print("❌ No spectral data available for any matches")
            return False
        
        print(f"✅ Created {plots_created} individual spectral comparisons")
        return True
    
    def create_multi_spectral_comparison(self, unknown_stone_id: str, unknown_features: List[Dict], 
                                       top_matches: List, max_matches: int = 6) -> bool:
        """Create overlaid spectral curves for unknown vs top N matches (original method)"""
        
        matches_to_plot = top_matches[:max_matches]
        
        print(f"📈 Creating overlaid multi-spectral comparison:")
        print(f"   🔍 Unknown: {unknown_stone_id}")
        print(f"   🎯 Top {len(matches_to_plot)} matches overlaid")
        
        # Load unknown stone data
        unknown_data = self.load_spectral_data(unknown_stone_id)
        if unknown_data is None:
            print("❌ No spectral data available for unknown stone")
            return False
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Plot unknown stone (thick black line)
        unknown_wavelengths, unknown_intensities = unknown_data
        norm_unknown_wav, norm_unknown_int = self.normalize_spectrum(unknown_wavelengths, unknown_intensities)
        
        ax.plot(norm_unknown_wav, norm_unknown_int, 
               color='black', linewidth=2.5, label=f'{unknown_stone_id} (Unknown)', 
               alpha=0.9, zorder=10)
        
        # Plot each match with different colors
        matches_plotted = 0
        for i, match in enumerate(matches_to_plot):
            match_data = self.load_spectral_data(match.stone_reference)
            
            if match_data is not None:
                match_wavelengths, match_intensities = match_data
                norm_match_wav, norm_match_int = self.normalize_spectrum(match_wavelengths, match_intensities)
                
                color = self.colors[i % len(self.colors)]
                ax.plot(norm_match_wav, norm_match_int, 
                       color=color, linewidth=self.line_thickness, 
                       label=f'{match.stone_reference} ({match.overall_confidence:.1f}%)', 
                       alpha=0.7, zorder=5-i)
                matches_plotted += 1
        
        if matches_plotted == 0:
            print("❌ No spectral data available for any matches")
            plt.close()
            return False
        
        # Add feature markers
        self._add_feature_markers(ax, unknown_features, unknown_stone_id)
        
        # Formatting
        ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
        ax.set_title(f'Multi-Spectral Comparison: {unknown_stone_id} vs Top {matches_plotted} Matches', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Legend positioning
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Set wavelength range
        min_wav = np.min(unknown_wavelengths)
        max_wav = np.max(unknown_wavelengths)
        margin = (max_wav - min_wav) * 0.02
        ax.set_xlim(min_wav - margin, max_wav + margin)
        
        plt.tight_layout()
        plt.show()
        
        print(f"✅ Multi-spectral comparison displayed ({matches_plotted} matches)")
        return True
    
    def _add_feature_markers(self, ax, features: List[Dict], stone_id: str):
        """Add vertical lines for identified features"""
        
        feature_colors = {'Peak': 'red', 'Mound': 'blue', 'Plateau': 'green'}
        feature_styles = {'Peak': '--', 'Mound': '-.', 'Plateau': ':'}
        
        for feature in features:
            feature_type = feature['feature_type']
            wavelength = None
            
            # Get the appropriate wavelength for each feature type
            if feature_type == 'Peak' and feature.get('max_wavelength'):
                wavelength = feature['max_wavelength']
            elif feature_type == 'Mound' and feature.get('crest_wavelength'):
                wavelength = feature['crest_wavelength']
            elif feature_type == 'Plateau' and feature.get('midpoint_wavelength'):
                wavelength = feature['midpoint_wavelength']
            
            if wavelength:
                color = feature_colors.get(feature_type, 'gray')
                style = feature_styles.get(feature_type, '-')
                
                ax.axvline(wavelength, color=color, linestyle=style, alpha=0.6, linewidth=1.0)
                
                # Add feature label
                ax.annotate(f'{feature_type}\n{wavelength:.1f}nm', 
                           xy=(wavelength, ax.get_ylim()[1] * 0.9), 
                           xytext=(5, -5), textcoords='offset points',
                           fontsize=8, color=color, ha='left', va='top',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    def test_data_loading(self, stone_ids: List[str]):
        """Test loading spectral data for multiple stone IDs"""
        print(f"🧪 Testing spectral data loading for {len(stone_ids)} stones:")
        print("=" * 50)
        
        found = 0
        for stone_id in stone_ids:
            data = self.load_spectral_data(stone_id)
            if data is not None:
                found += 1
        
        print(f"\n📊 Results: {found}/{len(stone_ids)} spectral files found")
        return found > 0

if __name__ == "__main__":
    # Test the spectral curve plotter
    print("📈 TESTING SPECTRAL CURVE PLOTTER")
    print("=" * 40)
    
    try:
        plotter = SpectralCurvePlotter()
        
        # Test with some common stone IDs
        test_ids = ['C0010BP1', 'C0012BP1', 'C0013BP1', 'testBP1']
        
        if plotter.test_data_loading(test_ids):
            print("✅ Spectral curve plotter ready!")
        else:
            print("⚠️ No spectral data files found - check directory path")
            
    except Exception as e:
        print(f"❌ Error initializing spectral curve plotter: {e}")
