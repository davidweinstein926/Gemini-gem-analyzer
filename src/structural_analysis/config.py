#!/usr/bin/env python3
"""
STRUCTURAL MATCHING CONFIGURATION
Location: src/structural_analysis/config.py

Centralized configuration for structural feature matching.
Contains exact tolerances and penalties as specified.

Author: Gemini Gem Analyzer Team
Version: 1.0
"""

# =============================================================================
# WAVELENGTH TOLERANCES (±nm)
# These define the acceptable wavelength difference for each feature type
# =============================================================================

WAVELENGTH_TOLERANCES = {
    'Mound': {
        'start': 7,    # ±7nm for mound start
        'crest': 5,    # ±5nm for mound crest (peak of mound)
        'end': 7,      # ±7nm for mound end (ignored if > 850nm)
    },
    'Plateau': {
        'start': 7,    # ±7nm for plateau start
        'middle': 5,   # ±5nm for plateau midpoint (most critical)
        'end': 7,      # ±7nm for plateau end
    },
    'Shoulder': {
        'start': 7,    # ±7nm for shoulder start
        'middle': 5,   # ±5nm for shoulder midpoint (most critical)
        'end': 7,      # ±7nm for shoulder end
    },
    'Peak': {
        'top': 1,      # ±1nm for peak (single point, very tight!)
    },
    'Trough': {
        'start': 5,    # ±5nm for trough start
        'middle': 3,   # ±3nm for trough midpoint
        'end': 5,      # ±5nm for trough end
    },
    'Valley': {
        'point': 3,    # ±3nm for valley (single point)
    },
    'Diagnostic Region': {
        'start': 5,    # ±5nm for diagnostic region start
        'end': 5,      # ±5nm for diagnostic region end
    },
    'Baseline': {
        'start': 50,   # ±50nm for baseline start (very loose, not critical)
        'end': 50,     # ±50nm for baseline end (very loose, not critical)
    }
}

# =============================================================================
# STRUCTURAL MISMATCH PENALTIES
# Fixed point penalties for missing or extra features
# =============================================================================

# If GOI has a feature but DB doesn't: add this penalty
MISSING_FEATURE_PENALTY = 10

# If DB has a feature but GOI doesn't: add this penalty
EXTRA_FEATURE_PENALTY = 35

# =============================================================================
# FEATURE EQUIVALENCE RULES
# Features that can match each other despite different names
# =============================================================================

FEATURE_EQUIVALENCE = {
    'Plateau': ['Plateau', 'Shoulder'],    # Plateau can match Plateau or Shoulder
    'Shoulder': ['Plateau', 'Shoulder'],   # Shoulder can match Plateau or Shoulder
}

# Note: The key insight is that Plateau and Shoulder are structurally similar
# What matters is the wavelength position, not the type name

# =============================================================================
# SPECIAL RULES
# =============================================================================

# Ignore mound end wavelength if it's above this threshold (nm)
# Rationale: Mound end above 850nm has no diagnostic meaning
MOUND_END_IGNORE_THRESHOLD = 850

# Baseline has minimal discriminatory power (always present)
BASELINE_MINIMAL_WEIGHT = True

# =============================================================================
# FEATURE IMPORTANCE HIERARCHY (for reference/documentation)
# Not currently used in scoring but documents the importance levels
# =============================================================================

FEATURE_IMPORTANCE = {
    'Baseline': 1,           # Always present, least discriminatory
    'Mound': 3,              # Moderate importance
    'Diagnostic Region': 3,  # Moderate importance
    'Plateau': 4,            # High importance
    'Shoulder': 4,           # High importance (same as Plateau)
    'Trough': 4,             # High importance
    'Valley': 4,             # High importance
    'Peak': 5,               # Highest importance (wavelength + intensity critical)
}

# =============================================================================
# POINT NAME MAPPINGS
# Maps feature types to their constituent points
# =============================================================================

FEATURE_POINT_MAPPINGS = {
    'Baseline': ['start', 'end'],
    'Mound': ['start', 'crest', 'end'],
    'Plateau': ['start', 'middle', 'end'],
    'Shoulder': ['start', 'middle', 'end'],
    'Peak': ['top'],
    'Trough': ['start', 'middle', 'end'],
    'Valley': ['point'],
    'Diagnostic Region': ['start', 'end']
}

# =============================================================================
# COLUMN NAME MAPPINGS (for CSV parsing)
# Maps various column naming conventions to standard internal names
# =============================================================================

COLUMN_NAME_ALIASES = {
    'wavelength': ['Wavelength', 'wavelength', 'wl', 'WL'],
    'intensity': ['Intensity', 'intensity', 'int', 'INT'],
    'feature': ['Feature', 'feature', 'feature_type', 'FeatureType'],
    'light_source': ['Light_Source', 'light_source', 'LightSource', 'Light Source'],
}

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Minimum number of features required for valid analysis
MIN_FEATURES_REQUIRED = 2  # At minimum: Baseline + one other feature

# Maximum allowed score before gem is considered "no match"
MAX_ACCEPTABLE_SCORE = 100

# Wavelength range limits (nm)
MIN_WAVELENGTH = 280
MAX_WAVELENGTH = 1100

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tolerance(feature_type, point_name):
    """
    Get tolerance for a specific feature type and point.
    
    Args:
        feature_type: Feature type (e.g., 'Mound', 'Plateau')
        point_name: Point name (e.g., 'start', 'crest', 'middle')
    
    Returns:
        Tolerance in nm, or None if not found
    """
    tolerance_set = WAVELENGTH_TOLERANCES.get(feature_type, {})
    return tolerance_set.get(point_name)


def get_feature_points(feature_type):
    """
    Get list of points for a feature type.
    
    Args:
        feature_type: Feature type (e.g., 'Mound', 'Plateau')
    
    Returns:
        List of point names, or empty list if not found
    """
    return FEATURE_POINT_MAPPINGS.get(feature_type, [])


def are_types_equivalent(type1, type2):
    """
    Check if two feature types are equivalent.
    
    Args:
        type1: First feature type
        type2: Second feature type
    
    Returns:
        True if types are equivalent, False otherwise
    """
    if type1 == type2:
        return True
    
    # Check if type1's equivalents include type2
    equiv1 = FEATURE_EQUIVALENCE.get(type1, [])
    if type2 in equiv1:
        return True
    
    # Check if type2's equivalents include type1
    equiv2 = FEATURE_EQUIVALENCE.get(type2, [])
    if type1 in equiv2:
        return True
    
    return False


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config():
    """
    Validate that configuration is complete and consistent.
    Raises ValueError if configuration has issues.
    """
    errors = []
    
    # Check that all tolerances are positive
    for feature_type, points in WAVELENGTH_TOLERANCES.items():
        for point_name, tolerance in points.items():
            if tolerance <= 0:
                errors.append(f"{feature_type}.{point_name} has non-positive tolerance: {tolerance}")
    
    # Check that penalties are positive
    if MISSING_FEATURE_PENALTY <= 0:
        errors.append(f"MISSING_FEATURE_PENALTY must be positive: {MISSING_FEATURE_PENALTY}")
    if EXTRA_FEATURE_PENALTY <= 0:
        errors.append(f"EXTRA_FEATURE_PENALTY must be positive: {EXTRA_FEATURE_PENALTY}")
    
    # Check that equivalence is symmetric
    for feature_type, equivalents in FEATURE_EQUIVALENCE.items():
        for equiv in equivalents:
            if equiv != feature_type:  # Don't check self-reference
                equiv_list = FEATURE_EQUIVALENCE.get(equiv, [])
                if feature_type not in equiv_list:
                    errors.append(f"Equivalence not symmetric: {feature_type} -> {equiv}, but {equiv} doesn't reference {feature_type}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True


# Run validation on import
try:
    validate_config()
except ValueError as e:
    print(f"⚠️  Configuration Warning: {e}")


# =============================================================================
# USAGE INFORMATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("STRUCTURAL MATCHING CONFIGURATION")
    print("=" * 70)
    
    print("\n📏 WAVELENGTH TOLERANCES:")
    for feature_type, points in WAVELENGTH_TOLERANCES.items():
        print(f"\n  {feature_type}:")
        for point_name, tolerance in points.items():
            print(f"    • {point_name}: ±{tolerance}nm")
    
    print("\n🚨 PENALTIES:")
    print(f"  • Missing feature (in GOI, not in DB): +{MISSING_FEATURE_PENALTY}")
    print(f"  • Extra feature (in DB, not in GOI): +{EXTRA_FEATURE_PENALTY}")
    
    print("\n🔄 FEATURE EQUIVALENCE:")
    for feature_type, equivalents in FEATURE_EQUIVALENCE.items():
        equiv_str = ", ".join([e for e in equivalents if e != feature_type])
        if equiv_str:
            print(f"  • {feature_type} ≈ {equiv_str}")
    
    print("\n⚙️  SPECIAL RULES:")
    print(f"  • Mound end ignored if > {MOUND_END_IGNORE_THRESHOLD}nm")
    print(f"  • Baseline has minimal weight: {BASELINE_MINIMAL_WEIGHT}")
    
    print("\n✅ Configuration validated successfully!")
    print("=" * 70)