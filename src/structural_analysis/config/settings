
#!/usr/bin/env python3
"""
CONFIGURATION SETTINGS
All system parameters in one place for easy modification

Author: David
Version: 2024.08.06
"""

import os

# Database Configuration
DB_CONFIG = {
    'db_path': "multi_spectral_gemstone_db.db",
    'backup_enabled': True,
    'auto_optimize': True
}

# Light Source Configuration
LIGHT_SOURCES = {
    'B': {'name': 'Broadband', 'weight': 0.4, 'tolerance': 0.5},
    'L': {'name': 'Laser', 'weight': 0.35, 'tolerance': 0.3},
    'U': {'name': 'UV', 'weight': 0.25, 'tolerance': 0.8}
}

# Matching Algorithm Parameters
MATCHING_CONFIG = {
    'peak_tolerance': 0.5,
    'mound_tolerance': 1.0,
    'plateau_tolerance': 2.0,
    'extra_feature_penalty': 20.0,
    'missing_feature_penalty': 5.0,
    'min_confidence_threshold': 30.0
}

# File Paths Configuration
STRUCTURAL_DATA_DIR = r"c:\users\david\gemini sp10 structural data"

CATALOG_FILES = {
    'B': [
        'gemini_db_long_B.csv',
        r'gemini matcher\gemini_db_long_B.csv',
        r'C:\Users\David\OneDrive\Desktop\gemini matcher\gemini_db_long_B.csv',
        r'C:\Users\David\Desktop\gemini matcher\gemini_db_long_B.csv'
    ],
    'L': [
        'gemini_db_long_L.csv',
        r'gemini matcher\gemini_db_long_L.csv',
        r'C:\Users\David\OneDrive\Desktop\gemini matcher\gemini_db_long_L.csv',
        r'C:\Users\David\Desktop\gemini matcher\gemini_db_long_L.csv'
    ],
    'U': [
        'gemini_db_long_U.csv',
        r'gemini matcher\gemini_db_long_U.csv',
        r'C:\Users\David\OneDrive\Desktop\gemini matcher\gemini_db_long_U.csv',
        r'C:\Users\David\Desktop\gemini matcher\gemini_db_long_U.csv'
    ]
}

# System Configuration
SYSTEM_CONFIG = {
    'catalog_files': CATALOG_FILES,
    'structural_dir': STRUCTURAL_DATA_DIR,
    'debug_mode': True,
    'verbose_output': True,
    'progress_interval': 10  # Show progress every N stones
}

# Validation Rules
VALIDATION_RULES = {
    'required_columns': ['Feature', 'Start', 'End', 'Midpoint', 'Crest', 'Max', 'Bottom'],
    'valid_feature_types': ['Peak', 'Mound', 'Plateau'],
    'wavelength_range': (200, 1000),
    'max_file_size_mb': 50
}

def get_config(section=None):
    """Get configuration section or all config"""
    configs = {
        'db': DB_CONFIG,
        'light_sources': LIGHT_SOURCES,
        'matching': MATCHING_CONFIG,
        'system': SYSTEM_CONFIG,
        'validation': VALIDATION_RULES
    }
    
    if section:
        return configs.get(section, {})
    return configs

def validate_paths():
    """Validate all configured paths exist"""
    issues = []
    
    # Check structural data directory
    if not os.path.exists(STRUCTURAL_DATA_DIR):
        issues.append(f"Structural data directory not found: {STRUCTURAL_DATA_DIR}")
    
    # Check catalog files
    for light_source, paths in CATALOG_FILES.items():
        found = any(os.path.exists(path) for path in paths)
        if not found:
            issues.append(f"No {light_source}-source catalog found in configured paths")
    
    return issues

if __name__ == "__main__":
    # Test configuration
    print("🔧 CONFIGURATION TEST")
    print("=" * 40)
    
    issues = validate_paths()
    if issues:
        print("⚠️ Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ All paths configured correctly")
    
    print(f"\n📊 Light Sources: {list(LIGHT_SOURCES.keys())}")
    print(f"🗄️ Database: {DB_CONFIG['db_path']}")
    print(f"🎯 Confidence Threshold: {MATCHING_CONFIG['min_confidence_threshold']}%")
