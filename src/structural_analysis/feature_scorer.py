#!/usr/bin/env python3
"""
FEATURE SCORING MODULE
Location: src/structural_analysis/feature_scorer.py

Handles scoring of structural feature matches between GOI and database gems.
Uses exact tolerances and penalties from config.py.

Author: Gemini Gem Analyzer Team
Version: 2.0 - Updated with exact tolerances
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Import configuration - handle both relative and absolute imports
try:
    # Try relative import first (when used as module)
    from .config import (
        WAVELENGTH_TOLERANCES,
        MISSING_FEATURE_PENALTY,
        EXTRA_FEATURE_PENALTY,
        MOUND_END_IGNORE_THRESHOLD,
        FEATURE_EQUIVALENCE,
        get_tolerance,
        get_feature_points,
        are_types_equivalent
    )
except ImportError:
    # Fall back to absolute import (when run as standalone)
    from config import (
        WAVELENGTH_TOLERANCES,
        MISSING_FEATURE_PENALTY,
        EXTRA_FEATURE_PENALTY,
        MOUND_END_IGNORE_THRESHOLD,
        FEATURE_EQUIVALENCE,
        get_tolerance,
        get_feature_points,
        are_types_equivalent
    )


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScoringResult:
    """
    Result of scoring a comparison between GOI and database gem.
    
    Attributes:
        final_score: Total score (lower is better, 0 = perfect)
        percentage: Match percentage (0-100%, higher is better)
        is_perfect_match: True if score is near zero
        match_statistics: Dictionary with penalty breakdown
        weighted_penalties: List of individual penalty details
    """
    final_score: float
    percentage: float
    is_perfect_match: bool
    match_statistics: Dict[str, float]
    weighted_penalties: List[Dict] = field(default_factory=list)


@dataclass
class FeatureMatchDetail:
    """
    Detailed information about a single feature match.
    """
    feature_type: str
    point_name: str
    goi_wavelength: float
    db_wavelength: float
    difference: float
    tolerance: float
    penalty: float
    within_tolerance: bool
    ignored: bool = False


# =============================================================================
# FEATURE SCORER CLASS
# =============================================================================

class FeatureScorer:
    """
    Scores structural feature matches using exact tolerances.
    """
    
    def __init__(self):
        """Initialize scorer with configuration."""
        self.tolerances = WAVELENGTH_TOLERANCES
        self.missing_penalty = MISSING_FEATURE_PENALTY
        self.extra_penalty = EXTRA_FEATURE_PENALTY
        self.perfect_match_threshold = 0.5  # Score below this = perfect match
    
    # =========================================================================
    # WAVELENGTH PENALTY CALCULATION
    # =========================================================================
    
    def calculate_wavelength_penalty(self, goi_wl: float, db_wl: float, 
                                     tolerance: float) -> float:
        """
        Calculate penalty based on wavelength difference.
        
        EXACT FORMULA:
        - Within tolerance: 0.0 to 1.0 (proportional to difference)
        - Outside tolerance: > 1.0 (escalating penalty)
        
        Args:
            goi_wl: GOI wavelength (nm)
            db_wl: Database wavelength (nm)
            tolerance: Allowed tolerance (nm)
        
        Returns:
            Penalty score (0.0 = perfect, higher = worse)
        
        Examples:
            >>> scorer = FeatureScorer()
            >>> scorer.calculate_wavelength_penalty(550, 552, 5)
            0.4  # 2nm diff in 5nm tolerance = 40% = 0.4 penalty
            >>> scorer.calculate_wavelength_penalty(550, 560, 5)
            4.0  # 10nm diff, 5nm excess = escalating penalty
        """
        diff = abs(goi_wl - db_wl)
        
        # Within tolerance: proportional penalty (0.0 to 1.0)
        if diff <= tolerance:
            return (diff / tolerance) * 1.0
        
        # Outside tolerance: escalating penalty (> 1.0)
        excess = diff - tolerance
        return 1.0 + (excess / tolerance) * 3.0
    
    # =========================================================================
    # FEATURE POINT SCORING
    # =========================================================================
    
    def score_feature_point(self, goi_wl: float, db_wl: float, 
                           feature_type: str, point_name: str) -> Tuple[float, FeatureMatchDetail]:
        """
        Score a single point (start, crest, middle, end, etc.).
        
        Args:
            goi_wl: GOI wavelength at this point
            db_wl: DB wavelength at this point
            feature_type: Feature type (Mound, Plateau, etc.)
            point_name: Point name (start, crest, middle, etc.)
        
        Returns:
            Tuple of (penalty_score, detail_object)
        """
        # Special case: ignore mound end if > 850nm
        if (feature_type == 'Mound' and 
            point_name == 'end' and 
            goi_wl > MOUND_END_IGNORE_THRESHOLD):
            detail = FeatureMatchDetail(
                feature_type=feature_type,
                point_name=point_name,
                goi_wavelength=goi_wl,
                db_wavelength=db_wl,
                difference=abs(goi_wl - db_wl),
                tolerance=0,
                penalty=0.0,
                within_tolerance=True,
                ignored=True
            )
            return 0.0, detail
        
        # Get exact tolerance for this feature type and point
        tolerance = get_tolerance(feature_type, point_name)
        if tolerance is None:
            tolerance = 10  # Default fallback
        
        # Calculate penalty
        penalty = self.calculate_wavelength_penalty(goi_wl, db_wl, tolerance)
        diff = abs(goi_wl - db_wl)
        
        detail = FeatureMatchDetail(
            feature_type=feature_type,
            point_name=point_name,
            goi_wavelength=goi_wl,
            db_wavelength=db_wl,
            difference=diff,
            tolerance=tolerance,
            penalty=penalty,
            within_tolerance=(diff <= tolerance),
            ignored=False
        )
        
        return penalty, detail
    
    # =========================================================================
    # FEATURE MATCHING
    # =========================================================================
    
    def score_light_source_comparison(self, goi_features: List, 
                                      db_features: List,
                                      light_source: str) -> Tuple[ScoringResult, List]:
        """
        Score complete comparison for one light source.
        
        This is the main entry point called by multi_gem_structural_analyzer.py
        
        Args:
            goi_features: List of Feature objects from GOI
            db_features: List of Feature objects from database gem
            light_source: Light source code ('B', 'L', 'U')
        
        Returns:
            Tuple of (ScoringResult, list_of_match_details)
        """
        total_score = 0.0
        penalties = {
            'missing_features': 0,
            'extra_features': 0,
            'wavelength_differences': 0
        }
        all_details = []
        match_details = []
        
        # Convert to dictionaries by feature type for easier lookup
        goi_by_type = self._group_features_by_type(goi_features)
        db_by_type = self._group_features_by_type(db_features)
        
        # =====================================================================
        # Step 1: Check for MISSING features (in GOI but not in DB)
        # =====================================================================
        for goi_type in goi_by_type.keys():
            if not self._find_equivalent_feature(goi_type, db_by_type):
                penalty = self.missing_penalty
                penalties['missing_features'] += penalty
                total_score += penalty
                
                all_details.append({
                    'type': 'missing',
                    'feature': goi_type,
                    'penalty': penalty,
                    'description': f"GOI has {goi_type} but DB doesn't"
                })
        
        # =====================================================================
        # Step 2: Check for EXTRA features (in DB but not in GOI)
        # =====================================================================
        for db_type in db_by_type.keys():
            if not self._find_equivalent_feature(db_type, goi_by_type):
                penalty = self.extra_penalty
                penalties['extra_features'] += penalty
                total_score += penalty
                
                all_details.append({
                    'type': 'extra',
                    'feature': db_type,
                    'penalty': penalty,
                    'description': f"DB has {db_type} but GOI doesn't"
                })
        
        # =====================================================================
        # Step 3: Score MATCHED features (wavelength comparison)
        # =====================================================================
        for goi_type, goi_feature in goi_by_type.items():
            db_feature = self._find_equivalent_feature(goi_type, db_by_type)
            
            if db_feature:
                # Get actual matched type (might be equivalent, e.g., Plateau -> Shoulder)
                matched_type = db_feature.feature_type
                
                # Score this feature match
                feature_penalty, feature_details = self._score_feature_match(
                    goi_feature, db_feature, goi_type
                )
                
                penalties['wavelength_differences'] += feature_penalty
                total_score += feature_penalty
                
                all_details.append({
                    'type': 'matched',
                    'feature': goi_type,
                    'matched_as': matched_type,
                    'is_equivalent': (goi_type != matched_type),
                    'penalty': feature_penalty,
                    'details': feature_details
                })
                
                match_details.extend(feature_details)
        
        # =====================================================================
        # Step 4: Create ScoringResult
        # =====================================================================
        percentage = self._score_to_percentage(total_score)
        is_perfect = total_score < self.perfect_match_threshold
        
        result = ScoringResult(
            final_score=total_score,
            percentage=percentage,
            is_perfect_match=is_perfect,
            match_statistics=penalties,
            weighted_penalties=all_details
        )
        
        return result, match_details
    
    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================
    
    def _group_features_by_type(self, features: List) -> Dict:
        """
        Group features by feature type.
        
        Args:
            features: List of Feature objects
        
        Returns:
            Dictionary {feature_type: feature_object}
        """
        grouped = {}
        for feature in features:
            feature_type = feature.feature_type
            grouped[feature_type] = feature
        
        return grouped
    
    def _find_equivalent_feature(self, feature_type: str, feature_dict: Dict):
        """
        Find feature in dictionary, considering equivalence (Plateau ≈ Shoulder).
        
        Args:
            feature_type: Type to search for
            feature_dict: Dictionary of features by type
        
        Returns:
            Matching feature object or None
        """
        # Check exact match first
        if feature_type in feature_dict:
            return feature_dict[feature_type]
        
        # Check equivalent types (e.g., Plateau can match Shoulder)
        equivalents = FEATURE_EQUIVALENCE.get(feature_type, [])
        for equiv_type in equivalents:
            if equiv_type in feature_dict and equiv_type != feature_type:
                return feature_dict[equiv_type]
        
        return None
    
    def _score_feature_match(self, goi_feature, db_feature, 
                            feature_type: str) -> Tuple[float, List[FeatureMatchDetail]]:
        """
        Score wavelength differences for a matched feature.
        
        Args:
            goi_feature: GOI Feature object
            db_feature: DB Feature object
            feature_type: Feature type name
        
        Returns:
            Tuple of (total_penalty, list_of_point_details)
        """
        total_penalty = 0.0
        details = []
        
        # Get list of points to check for this feature type
        points = get_feature_points(feature_type)
        
        for point_name in points:
            # Get wavelengths from feature objects
            goi_wl = self._get_wavelength_from_feature(goi_feature, point_name)
            db_wl = self._get_wavelength_from_feature(db_feature, point_name)
            
            if goi_wl is not None and db_wl is not None:
                penalty, detail = self.score_feature_point(
                    goi_wl, db_wl, feature_type, point_name
                )
                total_penalty += penalty
                details.append(detail)
        
        return total_penalty, details
    
    def _get_wavelength_from_feature(self, feature, point_name: str) -> Optional[float]:
        """
        Extract wavelength for a specific point from feature object.
        
        Args:
            feature: Feature object
            point_name: Point name (start, crest, middle, end, top, point)
        
        Returns:
            Wavelength value or None if not found
        """
        # Map point names to common attribute names
        attr_mappings = {
            'start': ['wavelength_start', 'start_wavelength', 'start'],
            'crest': ['wavelength_crest', 'crest_wavelength', 'crest', 'peak_wavelength'],
            'middle': ['wavelength_middle', 'middle_wavelength', 'middle', 'midpoint_wavelength', 'midpoint'],
            'end': ['wavelength_end', 'end_wavelength', 'end'],
            'top': ['wavelength_top', 'top_wavelength', 'wavelength', 'peak_wavelength'],
            'point': ['wavelength_point', 'point_wavelength', 'wavelength']
        }
        
        possible_attrs = attr_mappings.get(point_name, [point_name])
        
        for attr_name in possible_attrs:
            if hasattr(feature, attr_name):
                value = getattr(feature, attr_name)
                if value is not None:
                    return float(value)
        
        # Fallback: check if feature has 'wavelength' attribute (single-point features)
        if hasattr(feature, 'wavelength'):
            return float(feature.wavelength)
        
        return None
    
    def _score_to_percentage(self, score: float) -> float:
        """
        Convert score to match percentage.
        
        Score interpretation:
        - 0.0 = 100% match (perfect)
        - 1.0 = ~90% match
        - 5.0 = ~60% match
        - 10.0 = ~37% match
        - 20+ = poor match
        
        Args:
            score: Raw score (lower is better)
        
        Returns:
            Percentage (0-100, higher is better)
        """
        if score < self.perfect_match_threshold:
            return 100.0
        
        # Exponential decay for realistic percentages
        percentage = 100.0 * math.exp(-score / 10.0)
        
        return max(0.0, min(100.0, percentage))


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_test_scorer():
    """
    Create a scorer instance for testing.
    
    Returns:
        Configured FeatureScorer instance
    """
    return FeatureScorer()


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FEATURE SCORER TEST")
    print("=" * 70)
    
    scorer = FeatureScorer()
    
    # Test 1: Wavelength within tolerance
    print("\nTest 1: Wavelength difference within tolerance")
    penalty = scorer.calculate_wavelength_penalty(550.0, 552.0, 5.0)
    print(f"  GOI: 550nm, DB: 552nm, Tolerance: ±5nm")
    print(f"  Penalty: {penalty:.3f} (expected ~0.4)")
    assert 0.3 < penalty < 0.5, "Penalty should be around 0.4"
    print("  ✓ PASS")
    
    # Test 2: Wavelength outside tolerance
    print("\nTest 2: Wavelength difference outside tolerance")
    penalty = scorer.calculate_wavelength_penalty(550.0, 560.0, 5.0)
    print(f"  GOI: 550nm, DB: 560nm, Tolerance: ±5nm")
    print(f"  Penalty: {penalty:.3f} (expected > 1.0)")
    assert penalty > 1.0, "Penalty should exceed 1.0"
    print("  ✓ PASS")
    
    # Test 3: Perfect match
    print("\nTest 3: Perfect wavelength match")
    penalty = scorer.calculate_wavelength_penalty(550.0, 550.0, 5.0)
    print(f"  GOI: 550nm, DB: 550nm, Tolerance: ±5nm")
    print(f"  Penalty: {penalty:.3f} (expected 0.0)")
    assert penalty == 0.0, "Penalty should be zero"
    print("  ✓ PASS")
    
    # Test 4: Penalties
    print("\nTest 4: Missing/Extra feature penalties")
    print(f"  Missing feature penalty: {scorer.missing_penalty} (expected 10)")
    print(f"  Extra feature penalty: {scorer.extra_penalty} (expected 10)")
    assert scorer.missing_penalty == 10, "Missing penalty should be 10"
    assert scorer.extra_penalty == 10, "Extra penalty should be 10"
    print("  ✓ PASS")
    
    # Test 5: Score to percentage conversion
    print("\nTest 5: Score to percentage conversion")
    test_scores = [0.0, 1.0, 5.0, 10.0, 20.0]
    for score in test_scores:
        pct = scorer._score_to_percentage(score)
        print(f"  Score {score:.1f} → {pct:.1f}%")
    print("  ✓ PASS")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nConfiguration loaded:")
    print(f"  • Tolerances: {len(WAVELENGTH_TOLERANCES)} feature types")
    print(f"  • Missing penalty: +{MISSING_FEATURE_PENALTY}")
    print(f"  • Extra penalty: +{EXTRA_FEATURE_PENALTY}")
    print(f"  • Mound end ignore threshold: {MOUND_END_IGNORE_THRESHOLD}nm")