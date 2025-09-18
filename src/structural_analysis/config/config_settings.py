#!/usr/bin/env python3
"""
UPDATED CONFIGURATION SETTINGS
Updated for current Gemini project structure

Author: David (Updated for current project)
Version: 2024.12.18
"""

import os
from pathlib import Path

def find_project_root():
    """Find the project root directory"""
    current = Path(__file__).parent.absolute()
    
    for path in [current] + list(current.parents):
        if (path / "database").exists() and (path / "data").exists():
            return path
        if (path / "main.py").exists():
            return path
    
    return current.parent.parent.parent  # Fallback from config folder

# Get project root for dynamic paths
PROJECT_ROOT = find_project_root()

# Database Configuration - UPDATED for current structure
DB_CONFIG = {
    'db_path': str(PROJECT_ROOT / "database" / "structural_spectra" / "multi_structural_gem_data.db"),
    'csv_path': str(PROJECT_ROOT / "database" / "structural_spectra" / "gemini_structural_db.csv"),
    'backup_enabled': True,
    'auto_optimize': True
}

# Light Source Configuration - ENHANCED
LIGHT_SOURCES = {
    'B': {'name': 'Halogen', 'weight': 0.4, 'tolerance': 0.5},
    'L': {'name': 'Laser', 'weight': 0.35, 'tolerance': 0.3},
    'U': {'name': 'UV', 'weight': 0.25, 'tolerance': 0.8}
}

# Matching Algorithm Parameters - ORIGINAL SOPHISTICATED VALUES
MATCHING_CONFIG = {
    'peak_tolerance': 0.5,
    'mound_tolerance': 1.0,
    'plateau_tolerance': 2.0,
    'extra_feature_penalty': 20.0,
    'missing_feature_penalty': 5.0,
    'min_confidence_threshold': 30.0
}

# File Paths Configuration - UPDATED for current project
STRUCTURAL_DATA_DIR = str(PROJECT_ROOT / "data" / "structural_data")
STRUCTURAL_ARCHIVE_DIR = str(PROJECT_ROOT / "data" / "structural(archive)")

# Database reference files - UPDATED paths
CATALOG_FILES = {
    'B': [
        str(PROJECT_ROOT / "database" / "reference_spectra" / "gemini_db_long_B.csv"),
        'gemini_db_long_B.csv',
        r'database\reference_spectra\gemini_db_long_B.csv'
    ],
    'L': [
        str(PROJECT_ROOT / "database" / "reference_spectra" / "gemini_db_long_L.csv"),
        'gemini_db_long_L.csv', 
        r'database\reference_spectra\gemini_db_long_L.csv'
    ],
    'U': [
        str(PROJECT_ROOT / "database" / "reference_spectra" / "gemini_db_long_U.csv"),
        'gemini_db_long_U.csv',
        r'database\reference_spectra\gemini_db_long_U.csv'
    ]
}

# System Configuration - UPDATED
SYSTEM_CONFIG = {
    'catalog_files': CATALOG_FILES,
    'structural_dir': STRUCTURAL_DATA_DIR,
    'structural_archive_dir': STRUCTURAL_ARCHIVE_DIR,
    'output_dir': str(PROJECT_ROOT / "outputs" / "structural_results"),
    'reports_dir': str(PROJECT_ROOT / "outputs" / "structural_results" / "reports"),
    'graphs_dir': str(PROJECT_ROOT / "outputs" / "structural_results" / "graphs"),
    'debug_mode': True,
    'verbose_output': True,
    'progress_interval': 10
}

# Validation Rules - ENHANCED
VALIDATION_RULES = {
    'required_columns': ['Feature', 'Start', 'End', 'Midpoint', 'Crest', 'Max', 'Bottom'],
    'valid_feature_types': ['Peak', 'Mound', 'Plateau'],
    'wavelength_range': (200, 1000),
    'max_file_size_mb': 50,
    'expected_light_sources': ['Halogen', 'Laser', 'UV', 'B', 'L', 'U']
}

# UV Analysis Parameters - NEW (from enhanced analyzer)
UV_ANALYSIS_CONFIG = {
    'reference_wavelength': 811.0,
    'reference_expected_intensity': 15.0,
    'minimum_real_peak_intensity': 2.0,
    'real_peak_standards': [296.7, 302.1, 415.6, 419.6, 922.7],
    'diagnostic_peaks': {
        507.0: "Diamond ID (natural=absorb, synthetic=transmit)", 
        302.0: "Corundum natural vs synthetic"
    }
}

def get_config(section=None):
    """Get configuration section or all config"""
    configs = {
        'db': DB_CONFIG,
        'light_sources': LIGHT_SOURCES,
        'matching': MATCHING_CONFIG,
        'system': SYSTEM_CONFIG,
        'validation': VALIDATION_RULES,
        'uv_analysis': UV_ANALYSIS_CONFIG
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
    
    # Check database directory
    db_dir = Path(DB_CONFIG['db_path']).parent
    if not db_dir.exists():
        issues.append(f"Database directory not found: {db_dir}")
    
    # Check catalog files
    for light_source, paths in CATALOG_FILES.items():
        found = any(os.path.exists(path) for path in paths)
        if not found:
            issues.append(f"No {light_source}-source catalog found in configured paths")
    
    return issues

def get_project_info():
    """Get project information"""
    return {
        'project_root': PROJECT_ROOT,
        'config_location': Path(__file__).parent,
        'database_path': DB_CONFIG['db_path'],
        'structural_data': STRUCTURAL_DATA_DIR,
        'output_directory': SYSTEM_CONFIG['output_dir']
    }

if __name__ == "__main__":
    # Test configuration
    print("🔧 UPDATED CONFIGURATION TEST")
    print("=" * 50)
    
    info = get_project_info()
    print(f"📁 Project Root: {info['project_root']}")
    print(f"⚙️  Config Location: {info['config_location']}")
    print(f"💾 Database: {info['database_path']}")
    print(f"📂 Structural Data: {info['structural_data']}")
    
    issues = validate_paths()
    if issues:
        print(f"\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ All paths configured correctly")
    
    print(f"\n📊 Light Sources: {list(LIGHT_SOURCES.keys())}")
    print(f"🎯 Confidence Threshold: {MATCHING_CONFIG['min_confidence_threshold']}%")
    print(f"⚖️  Light Source Weights: B={LIGHT_SOURCES['B']['weight']}, L={LIGHT_SOURCES['L']['weight']}, U={LIGHT_SOURCES['U']['weight']}")