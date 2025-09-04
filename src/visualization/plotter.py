#!/usr/bin/env python3
"""
VISUALIZATION MODULE
Generates normalized comparison plots for top matches

Author: David
Version: 2024.08.06
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict
from config.settings import get_config

class SpectralPlotter:
    """Creates normalized comparison plots for spectral matches"""
    
    def __init__(self):
        self.config = get_config('system')
        self.line_thickness = 0.5
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        print("📊 Spectral plotter initialized")
    
    def create_comparison_plot(self, unknown_stone_id: str, unknown_features: List[Dict], 
                             top_matches: List, db_connection):
        """Create normalized comparison plot for top 6 matches"""
        
        if len(top_matches) == 0:
            print("❌ No matches to plot")
            return
        
        # Limit to top 6 matches
        matches_to_plot = top_matches[:6]
        
        print(f"📊 Creating comparison plot for {unknown_stone_id}")
        print(f"   Plotting top {len(matches_to_plot)} matches")
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f'Spectral Analysis Comparison: {unknown_stone_id}', fontsize=14, fontweight='bold')
        
        # Plot 1: Feature positions
        self._plot_feature_positions(ax1, unknown_stone_id, unknown_features, matches_to_plot, db_connection)
        
        # Plot 2: Confidence scores
        self._plot_confidence_scores(ax2, matches_to_plot)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Comparison plot displayed")
    
    def _plot_feature_positions(self, ax, unknown_stone_id, unknown_features, matches, db_connection):
        """Plot normalized feature positions"""
        
        # Plot unknown stone features first
        unknown_positions = []
        unknown_types = []
        
        for feature in unknown_features:
            feature_type = feature['feature_type']
            
            if feature_type == 'Peak' and feature.get('max_wavelength'):
                unknown_positions.append(feature['max_wavelength'])
                unknown_types.append('Peak')
            elif feature_type == 'Mound' and feature.get('crest_wavelength'):
                unknown_positions.append(feature['crest_wavelength'])
                unknown_types.append('Mound')
            elif feature_type == 'Plateau' and feature.get('midpoint_wavelength'):
                unknown_positions.append(feature['midpoint_wavelength'])
                unknown_types.append('Plateau')
        
        # Plot unknown stone (thick black line)
        if unknown_positions:
            normalized_unknown = self._normalize_positions(unknown_positions)
            ax.scatter(unknown_positions, [0] * len(unknown_positions), 
                      color='black', s=100, marker='o', linewidth=2,
                      label=f'{unknown_stone_id} (Unknown)', zorder=10)
            
            # Add feature type labels
            for i, (pos, feat_type) in enumerate(zip(unknown_positions, unknown_types)):
                ax.annotate(feat_type, (pos, 0), xytext=(0, 15), 
                           textcoords='offset points', ha='center', fontsize=8)
        
        # Plot database matches
        for idx, match in enumerate(matches):
            if idx >= 6:  # Limit to top 6
                break
                
            # Get features for this match
            match_features = self._get_match_features(match.stone_reference, db_connection)
            
            if match_features:
                match_positions = []
                match_types = []
                
                for feature in match_features:
                    feature_type = feature['feature_type']
                    y_offset = -(idx + 1) * 0.5  # Offset each match vertically
                    
                    if feature_type == 'Peak' and feature.get('max_wavelength'):
                        match_positions.append(feature['max_wavelength'])
                        match_types.append('Peak')
                    elif feature_type == 'Mound' and feature.get('crest_wavelength'):
                        match_positions.append(feature['crest_wavelength'])
                        match_types.append('Mound')
                    elif feature_type == 'Plateau' and feature.get('midpoint_wavelength'):
                        match_positions.append(feature['midpoint_wavelength'])
                        match_types.append('Plateau')
                
                if match_positions:
                    color = self.colors[idx % len(self.colors)]
                    ax.scatter(match_positions, [y_offset] * len(match_positions),
                              color=color, s=60, marker='s', linewidth=self.line_thickness,
                              label=f'{match.stone_reference} ({match.overall_confidence:.1f}%)',
                              alpha=0.8)
        
        ax.set_xlabel('Wavelength (nm)', fontweight='bold')
        ax.set_ylabel('Relative Position', fontweight='bold')
        ax.set_title('Feature Position Comparison', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set wavelength range
        if unknown_positions:
            all_positions = unknown_positions.copy()
            for match in matches[:6]:
                match_features = self._get_match_features(match.stone_reference, db_connection)
                for feature in match_features:
                    if feature.get('max_wavelength'):
                        all_positions.append(feature['max_wavelength'])
                    if feature.get('crest_wavelength'):
                        all_positions.append(feature['crest_wavelength'])
                    if feature.get('midpoint_wavelength'):
                        all_positions.append(feature['midpoint_wavelength'])
            
            if all_positions:
                margin = (max(all_positions) - min(all_positions)) * 0.1
                ax.set_xlim(min(all_positions) - margin, max(all_positions) + margin)
    
    def _plot_confidence_scores(self, ax, matches):
        """Plot confidence scores as bar chart"""
        
        matches_to_plot = matches[:6]
        stone_ids = [match.stone_reference for match in matches_to_plot]
        scores = [match.overall_confidence for match in matches_to_plot]
        colors = [self.colors[i % len(self.colors)] for i in range(len(matches_to_plot))]
        
        bars = ax.bar(range(len(stone_ids)), scores, color=colors, alpha=0.7, linewidth=self.line_thickness)
        
        ax.set_xlabel('Database Stones', fontweight='bold')
        ax.set_ylabel('Confidence Score (%)', fontweight='bold')
        ax.set_title('Match Confidence Scores', fontweight='bold')
        ax.set_xticks(range(len(stone_ids)))
        ax.set_xticklabels(stone_ids, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # Add score labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{score:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    def _normalize_positions(self, positions):
        """Normalize positions to 0-1 range"""
        if not positions:
            return []
        
        min_pos = min(positions)
        max_pos = max(positions)
        
        if max_pos == min_pos:
            return [0.5] * len(positions)
        
        return [(pos - min_pos) / (max_pos - min_pos) for pos in positions]
    
    def _get_match_features(self, stone_reference, db_connection):
        """Get features for a matched stone"""
        try:
            features_query = '''
            SELECT sf.* FROM structural_features sf
            JOIN spectral_data sd ON sf.spectral_id = sd.spectral_id
            WHERE sd.full_stone_id = ?
            '''
            features_df = pd.read_sql_query(features_query, db_connection, params=[stone_reference])
            return features_df.to_dict('records')
        except Exception as e:
            print(f"⚠️ Error getting features for {stone_reference}: {e}")
            return []
    
    def create_detailed_comparison(self, unknown_stone_id: str, unknown_features: List[Dict], 
                                 best_match, db_connection):
        """Create detailed comparison plot between unknown and best match"""
        
        print(f"📊 Creating detailed comparison: {unknown_stone_id} vs {best_match.stone_reference}")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get best match features
        match_features = self._get_match_features(best_match.stone_reference, db_connection)
        
        # Plot both stones
        self._plot_single_stone(ax, unknown_stone_id, unknown_features, 'black', 0.2, 'Unknown')
        self._plot_single_stone(ax, best_match.stone_reference, match_features, 'red', -0.2, 
                               f'Match ({best_match.overall_confidence:.1f}%)')
        
        ax.set_xlabel('Wavelength (nm)', fontweight='bold')
        ax.set_ylabel('Stone Comparison', fontweight='bold')
        ax.set_title(f'Detailed Comparison: {unknown_stone_id} vs {best_match.stone_reference}', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Detailed comparison plot displayed")
    
    def _plot_single_stone(self, ax, stone_id, features, color, y_offset, label):
        """Plot features for a single stone"""
        
        positions = []
        feature_types = []
        
        for feature in features:
            feature_type = feature['feature_type']
            
            if feature_type == 'Peak' and feature.get('max_wavelength'):
                positions.append(feature['max_wavelength'])
                feature_types.append('Peak')
            elif feature_type == 'Mound' and feature.get('crest_wavelength'):
                positions.append(feature['crest_wavelength'])
                feature_types.append('Mound')
            elif feature_type == 'Plateau' and feature.get('midpoint_wavelength'):
                positions.append(feature['midpoint_wavelength'])
                feature_types.append('Plateau')
        
        if positions:
            ax.scatter(positions, [y_offset] * len(positions),
                      color=color, s=80, linewidth=self.line_thickness,
                      label=label, alpha=0.8)
            
            # Add feature labels
            for pos, feat_type in zip(positions, feature_types):
                ax.annotate(feat_type, (pos, y_offset), xytext=(0, 10), 
                           textcoords='offset points', ha='center', fontsize=8)

if __name__ == "__main__":
    # Test plotter
    print("📊 TESTING SPECTRAL PLOTTER")
    print("=" * 30)
    
    try:
        import matplotlib.pyplot as plt
        plotter = SpectralPlotter()
        print("✅ Plotter initialized successfully")
        print("📊 Matplotlib available - ready for plotting")
    except ImportError:
        print("❌ Matplotlib not available - install with: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error initializing plotter: {e}")

