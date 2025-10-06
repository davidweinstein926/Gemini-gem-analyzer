#!/usr/bin/env python3
"""
FEATURE MATCHING MODULE
Location: src/structural_analysis/feature_matcher.py

Handles matching of structural features between GOI and database gems.
Implements equivalence rules (Plateau ≈ Shoulder).

Author: Gemini Gem Analyzer Team
Version: 2.0 - Updated with equivalence support
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Import configuration - handle both relative and absolute imports
try:
    # Try relative import first (when used as module)
    from .config import (
        FEATURE_EQUIVALENCE,
        FEATURE_POINT_MAPPINGS,
        are_types_equivalent
    )
except ImportError:
    # Fall back to absolute import (when run as standalone)
    from config import (
        FEATURE_EQUIVALENCE,
        FEATURE_POINT_MAPPINGS,
        are_types_equivalent
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Feature:
    """
    Represents a single structural feature.
    
    Attributes:
        feature_type: Type of feature (Mound, Plateau, Shoulder, etc.)
        light_source: Light source code (B, L, U)
        wavelength: Primary wavelength (for single-point features)
        wavelength_start: Start wavelength (for multi-point features)
        wavelength_crest: Crest wavelength (for mounds)
        wavelength_middle: Middle wavelength (for plateaus/shoulders/troughs)
        wavelength_end: End wavelength (for multi-point features)
        intensity: Intensity value
        metadata: Additional metadata dictionary
    """
    feature_type: str
    light_source: str
    wavelength: Optional[float] = None
    wavelength_start: Optional[float] = None
    wavelength_crest: Optional[float] = None
    wavelength_middle: Optional[float] = None
    wavelength_end: Optional[float] = None
    intensity: Optional[float] = None
    metadata: Optional[Dict] = None
    
    def __repr__(self):
        wl_info = []
        if self.wavelength is not None:
            wl_info.append(f"{self.wavelength:.1f}nm")
        if self.wavelength_start is not None:
            wl_info.append(f"start:{self.wavelength_start:.1f}")
        if self.wavelength_crest is not None:
            wl_info.append(f"crest:{self.wavelength_crest:.1f}")
        if self.wavelength_middle is not None:
            wl_info.append(f"mid:{self.wavelength_middle:.1f}")
        if self.wavelength_end is not None:
            wl_info.append(f"end:{self.wavelength_end:.1f}")
        
        wl_str = ", ".join(wl_info) if wl_info else "no wavelength"
        return f"Feature({self.feature_type}, {self.light_source}, {wl_str})"


@dataclass
class FeatureMatch:
    """
    Represents a match between GOI and database features.
    """
    goi_feature: Feature
    db_feature: Feature
    is_equivalent: bool  # True if types are different but equivalent (Plateau ≈ Shoulder)
    match_quality: str  # 'exact', 'equivalent', or 'none'


# =============================================================================
# FEATURE MATCHER CLASS
# =============================================================================

class FeatureMatcher:
    """
    Matches structural features between GOI and database gems.
    """
    
    def __init__(self):
        """Initialize matcher with equivalence rules."""
        self.equivalence_map = FEATURE_EQUIVALENCE
    
    # =========================================================================
    # EQUIVALENCE CHECKING
    # =========================================================================
    
    def are_types_equivalent(self, type1: str, type2: str) -> bool:
        """
        Check if two feature types are equivalent.
        
        KEY RULE: Plateau ≈ Shoulder
        
        Args:
            type1: First feature type
            type2: Second feature type
        
        Returns:
            True if types are equivalent, False otherwise
        
        Examples:
            >>> matcher = FeatureMatcher()
            >>> matcher.are_types_equivalent('Plateau', 'Shoulder')
            True
            >>> matcher.are_types_equivalent('Mound', 'Peak')
            False
        """
        return are_types_equivalent(type1, type2)
    
    # =========================================================================
    # FEATURE MATCHING
    # =========================================================================
    
    def find_matching_feature(self, goi_feature: Feature, 
                             db_features: List[Feature]) -> Optional[FeatureMatch]:
        """
        Find best matching feature in database for a GOI feature.
        
        Matching priority:
        1. Exact type match (Plateau → Plateau)
        2. Equivalent type match (Plateau → Shoulder)
        3. No match
        
        Args:
            goi_feature: GOI Feature object
            db_features: List of database Feature objects
        
        Returns:
            FeatureMatch object or None if no match found
        """
        goi_type = goi_feature.feature_type
        
        # First pass: exact type match
        for db_feature in db_features:
            if db_feature.feature_type == goi_type:
                return FeatureMatch(
                    goi_feature=goi_feature,
                    db_feature=db_feature,
                    is_equivalent=False,
                    match_quality='exact'
                )
        
        # Second pass: equivalent type match (e.g., Plateau → Shoulder)
        for db_feature in db_features:
            if self.are_types_equivalent(goi_type, db_feature.feature_type):
                return FeatureMatch(
                    goi_feature=goi_feature,
                    db_feature=db_feature,
                    is_equivalent=True,
                    match_quality='equivalent'
                )
        
        # No match found
        return FeatureMatch(
            goi_feature=goi_feature,
            db_feature=None,
            is_equivalent=False,
            match_quality='none'
        )
    
    def match_feature_sets(self, goi_features: List[Feature], 
                          db_features: List[Feature]) -> List[FeatureMatch]:
        """
        Match all GOI features against database features.
        
        Args:
            goi_features: List of GOI Feature objects
            db_features: List of database Feature objects
        
        Returns:
            List of FeatureMatch objects
        """
        matches = []
        
        for goi_feature in goi_features:
            match = self.find_matching_feature(goi_feature, db_features)
            matches.append(match)
        
        return matches
    
    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================
    
    def extract_feature_from_row(self, row: Dict, light_source: str) -> Optional[Feature]:
        """
        Extract a Feature object from a CSV row.
        
        Args:
            row: Dictionary representing a row from CSV
            light_source: Light source code (B, L, U)
        
        Returns:
            Feature object or None if row is invalid
        """
        # Get feature type from 'Feature' column
        feature_str = row.get('Feature', '')
        if not feature_str or 'Summary' in feature_str:
            return None
        
        # Parse feature type and point
        # Format: "Mound_Start", "Plateau_Midpoint", "Peak_Top", etc.
        parts = feature_str.split('_')
        
        # Handle "Diagnostic Region" which has space in name
        if 'Diagnostic' in feature_str:
            feature_type = 'Diagnostic Region'
            point_name = parts[-1] if parts else ''
        else:
            feature_type = parts[0] if parts else feature_str
            point_name = '_'.join(parts[1:]) if len(parts) > 1 else ''
        
        # Get wavelength and intensity
        wavelength = row.get('Wavelength')
        intensity = row.get('Intensity')
        
        if wavelength is not None:
            try:
                wavelength = float(wavelength)
            except (ValueError, TypeError):
                wavelength = None
        
        if intensity is not None:
            try:
                intensity = float(intensity)
            except (ValueError, TypeError):
                intensity = None
        
        # Create Feature object
        feature = Feature(
            feature_type=feature_type,
            light_source=light_source
        )
        
        # Map point name to appropriate wavelength attribute
        point_mappings = {
            'Start': 'wavelength_start',
            'Crest': 'wavelength_crest',
            'Midpoint': 'wavelength_middle',
            'Middle': 'wavelength_middle',
            'End': 'wavelength_end',
            'Top': 'wavelength',
            'Point': 'wavelength'
        }
        
        attr_name = point_mappings.get(point_name, 'wavelength')
        if wavelength is not None:
            setattr(feature, attr_name, wavelength)
        
        if intensity is not None:
            feature.intensity = intensity
        
        return feature
    
    def extract_features_from_dataframe(self, df, light_source: str) -> List[Feature]:
        """
        Extract all features from a dataframe.
        
        Args:
            df: Pandas DataFrame with structural data
            light_source: Light source code (B, L, U)
        
        Returns:
            List of Feature objects
        """
        features = []
        
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            feature = self.extract_feature_from_row(row_dict, light_source)
            if feature is not None:
                features.append(feature)
        
        # Group related points into single features
        features = self._consolidate_features(features)
        
        return features
    
    def _consolidate_features(self, features: List[Feature]) -> List[Feature]:
        """
        Consolidate multiple rows for the same feature into single Feature object.
        
        For example, combine:
        - Mound_Start, Mound_Crest, Mound_End → single Mound feature
        - Plateau_Start, Plateau_Midpoint, Plateau_End → single Plateau feature
        
        Args:
            features: List of Feature objects (one per row)
        
        Returns:
            List of consolidated Feature objects
        """
        # Group by feature type
        feature_groups = {}
        
        for feature in features:
            feature_type = feature.feature_type
            
            if feature_type not in feature_groups:
                feature_groups[feature_type] = Feature(
                    feature_type=feature_type,
                    light_source=feature.light_source
                )
            
            # Merge wavelength data
            consolidated = feature_groups[feature_type]
            
            if feature.wavelength is not None:
                consolidated.wavelength = feature.wavelength
            if feature.wavelength_start is not None:
                consolidated.wavelength_start = feature.wavelength_start
            if feature.wavelength_crest is not None:
                consolidated.wavelength_crest = feature.wavelength_crest
            if feature.wavelength_middle is not None:
                consolidated.wavelength_middle = feature.wavelength_middle
            if feature.wavelength_end is not None:
                consolidated.wavelength_end = feature.wavelength_end
            if feature.intensity is not None:
                consolidated.intensity = feature.intensity
        
        return list(feature_groups.values())
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_feature_summary(self, features: List[Feature]) -> Dict[str, int]:
        """
        Get summary of feature types in a list.
        
        Args:
            features: List of Feature objects
        
        Returns:
            Dictionary {feature_type: count}
        """
        summary = {}
        for feature in features:
            feature_type = feature.feature_type
            summary[feature_type] = summary.get(feature_type, 0) + 1
        
        return summary
    
    def compare_feature_sets(self, goi_features: List[Feature], 
                            db_features: List[Feature]) -> Dict:
        """
        Compare two sets of features and identify differences.
        
        Args:
            goi_features: List of GOI Feature objects
            db_features: List of database Feature objects
        
        Returns:
            Dictionary with comparison results
        """
        goi_types = set(f.feature_type for f in goi_features)
        db_types = set(f.feature_type for f in db_features)
        
        # Find missing and extra features considering equivalence
        missing = []
        for goi_type in goi_types:
            found = False
            if goi_type in db_types:
                found = True
            else:
                # Check for equivalent types
                for db_type in db_types:
                    if self.are_types_equivalent(goi_type, db_type):
                        found = True
                        break
            
            if not found:
                missing.append(goi_type)
        
        extra = []
        for db_type in db_types:
            found = False
            if db_type in goi_types:
                found = True
            else:
                # Check for equivalent types
                for goi_type in goi_types:
                    if self.are_types_equivalent(db_type, goi_type):
                        found = True
                        break
            
            if not found:
                extra.append(db_type)
        
        return {
            'goi_types': list(goi_types),
            'db_types': list(db_types),
            'missing_in_db': missing,
            'extra_in_db': extra,
            'exact_matches': len(goi_types & db_types),
            'equivalent_matches': len([t for t in goi_types if any(self.are_types_equivalent(t, dt) and t != dt for dt in db_types)])
        }


# =============================================================================
# STANDALONE FUNCTION FOR BACKWARD COMPATIBILITY
# =============================================================================

def extract_features_from_dataframe(df, light_source: str) -> List[Feature]:
    """
    Standalone function to extract features from dataframe.
    For backward compatibility with existing code.
    
    Args:
        df: Pandas DataFrame with structural data
        light_source: Light source code (B, L, U)
    
    Returns:
        List of Feature objects
    """
    matcher = FeatureMatcher()
    return matcher.extract_features_from_dataframe(df, light_source)


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FEATURE MATCHER TEST")
    print("=" * 70)
    
    matcher = FeatureMatcher()
    
    # Test 1: Equivalence checking
    print("\nTest 1: Feature type equivalence")
    test_cases = [
        ('Plateau', 'Plateau', True, "Same type"),
        ('Plateau', 'Shoulder', True, "Equivalent types"),
        ('Shoulder', 'Plateau', True, "Equivalent types (reverse)"),
        ('Mound', 'Peak', False, "Different types"),
        ('Mound', 'Mound', True, "Same type"),
    ]
    
    for type1, type2, expected, description in test_cases:
        result = matcher.are_types_equivalent(type1, type2)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {type1} ≈ {type2}: {result} ({description})")
        assert result == expected, f"Failed: {type1} ≈ {type2}"
    
    # Test 2: Feature extraction from row
    print("\nTest 2: Feature extraction from row")
    test_row = {
        'Feature': 'Mound_Start',
        'Wavelength': 500.0,
        'Intensity': 75.5
    }
    feature = matcher.extract_feature_from_row(test_row, 'B')
    print(f"  Extracted: {feature}")
    assert feature is not None
    assert feature.feature_type == 'Mound'
    assert feature.wavelength_start == 500.0
    print("  ✓ PASS")
    
    # Test 3: Diagnostic Region (special case with space)
    print("\nTest 3: Diagnostic Region extraction")
    test_row = {
        'Feature': 'Diagnostic Region_Start',
        'Wavelength': 650.0,
        'Intensity': 50.0
    }
    feature = matcher.extract_feature_from_row(test_row, 'B')
    print(f"  Extracted: {feature}")
    assert feature is not None
    assert feature.feature_type == 'Diagnostic Region'
    print("  ✓ PASS")
    
    # Test 4: Feature comparison
    print("\nTest 4: Feature set comparison")
    goi_features = [
        Feature('Mound', 'B'),
        Feature('Plateau', 'B')
    ]
    db_features = [
        Feature('Mound', 'B'),
        Feature('Shoulder', 'B'),  # Equivalent to Plateau
        Feature('Peak', 'B')        # Extra
    ]
    
    comparison = matcher.compare_feature_sets(goi_features, db_features)
    print(f"  GOI types: {comparison['goi_types']}")
    print(f"  DB types: {comparison['db_types']}")
    print(f"  Missing in DB: {comparison['missing_in_db']}")
    print(f"  Extra in DB: {comparison['extra_in_db']}")
    print(f"  Equivalent matches: {comparison['equivalent_matches']}")
    
    # Plateau should match Shoulder (equivalent), so nothing should be missing
    assert len(comparison['missing_in_db']) == 0, "Plateau should match Shoulder"
    assert 'Peak' in comparison['extra_in_db'], "Peak should be extra"
    print("  ✓ PASS")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nFeature equivalence rules:")
    for feature_type, equivalents in FEATURE_EQUIVALENCE.items():
        equiv_str = ", ".join([e for e in equivalents if e != feature_type])
        if equiv_str:
            print(f"  • {feature_type} ≈ {equiv_str}")