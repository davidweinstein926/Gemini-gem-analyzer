#!/usr/bin/env python3
"""
migrate_to_gemini_analysis.py - Automated Migration Script
Migrates from gemini_matcher to gemini_gemological_analysis structure
"""

import os
import shutil
import sys
from pathlib import Path

class GeminiMigrator:
    """Automated migration system"""
    
    def __init__(self):
        self.source_dir = "."  # Current directory (gemini_matcher)
        self.target_dir = "gemini_gemological_analysis"
        
        # File mapping: source -> target
        self.file_mappings = {
            # Core files
            "main.py": "main.py",
            "gemlib_structural_ready.csv": "databases/gemlib_structural_ready.csv",
            
            # Database files  
            "gemini_db_long_B.csv": "databases/gemini_db_long_B.csv",
            "gemini_db_long_L.csv": "databases/gemini_db_long_L.csv", 
            "gemini_db_long_U.csv": "databases/gemini_db_long_U.csv",
            "multi_structural_gem_data.db": "databases/multi_structural_gem_data.db",
            
            # Structural analysis
            "src/structural_analysis/gemini_launcher.py": "src/structural_analysis/gemini_launcher.py",
            "src/structural_analysis/gemini_halogen_analyzer.py": "src/structural_analysis/gemini_halogen_analyzer.py",
            "src/structural_analysis/gemini_laser_analyzer.py": "src/structural_analysis/gemini_laser_analyzer.py",
            "src/structural_analysis/gemini_uv_analyzer.py": "src/structural_analysis/gemini_uv_analyzer.py",
            "src/structural_analysis/auto_analysis/gemini_peak_detector.py": "src/structural_analysis/peak_detectors/gemini_peak_detector.py",
            "src/structural_analysis/auto_analysis/b_spectra_auto_detector.py": "src/structural_analysis/peak_detectors/b_spectra_auto_detector.py",
            "src/structural_analysis/auto_analysis/l_spectra_auto_detector.py": "src/structural_analysis/peak_detectors/l_spectra_auto_detector.py",
            
            # Numerical analysis
            "src/numerical_analysis/gemini1.py": "src/numerical_analysis/gemini1.py",
            "enhanced_gem_analyzer.py": "src/numerical_analysis/enhanced_gem_analyzer.py",
            
            # Other analysis files
            "fast_gem_analysis.py": "src/core/fast_gem_analysis.py",
        }
        
        # Directories to create
        self.directories = [
            "data/raw",
            "data/unknown", 
            "data/reference",
            "raw_txt",
            "databases",
            "src/core",
            "src/structural_analysis",
            "src/structural_analysis/peak_detectors", 
            "src/numerical_analysis",
            "src/visualization",
            "src/utils",
            "config",
            "tests",
            "docs",
            "examples/sample_data",
            "examples/tutorials"
        ]
        
        # Create __init__.py files
        self.init_files = [
            "src/__init__.py",
            "src/core/__init__.py",
            "src/structural_analysis/__init__.py", 
            "src/numerical_analysis/__init__.py",
            "src/visualization/__init__.py",
            "src/utils/__init__.py",
            "tests/__init__.py"
        ]
    
    def create_directory_structure(self):
        """Create the new directory structure"""
        print("🏗️  Creating directory structure...")
        
        base_path = Path(self.target_dir)
        base_path.mkdir(exist_ok=True)
        
        for directory in self.directories:
            dir_path = base_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created: {directory}")
        
        print(f"✅ Directory structure created in {self.target_dir}/")
    
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        print("\n📦 Creating package __init__.py files...")
        
        base_path = Path(self.target_dir)
        
        for init_file in self.init_files:
            init_path = base_path / init_file
            with open(init_path, 'w') as f:
                f.write('"""Gemini Gemological Analysis Package"""\n')
            print(f"   ✅ Created: {init_file}")
    
    def migrate_files(self):
        """Migrate files to new structure"""
        print("\n📁 Migrating files...")
        
        base_path = Path(self.target_dir)
        
        for source_file, target_file in self.file_mappings.items():
            source_path = Path(source_file)
            target_path = base_path / target_file
            
            if source_path.exists():
                # Ensure target directory exists
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(source_path, target_path)
                print(f"   ✅ {source_file} → {target_file}")
            else:
                print(f"   ⚠️  Not found: {source_file}")
    
    def update_import_paths(self):
        """Update import paths in migrated files"""
        print("\n🔧 Updating import paths...")
        
        # Files that need path updates
        files_to_update = [
            "main.py",
            "src/structural_analysis/gemini_launcher.py",
            "src/numerical_analysis/gemini1.py"
        ]
        
        base_path = Path(self.target_dir)
        
        for file_path in files_to_update:
            full_path = base_path / file_path
            if full_path.exists():
                self.update_file_paths(full_path)
                print(f"   ✅ Updated imports: {file_path}")
    
    def update_file_paths(self, file_path):
        """Update paths in a specific file"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Common path updates
        replacements = {
            # Database paths
            '"gemini_db_long_B.csv"': '"databases/gemini_db_long_B.csv"',
            '"gemini_db_long_L.csv"': '"databases/gemini_db_long_L.csv"',
            '"gemini_db_long_U.csv"': '"databases/gemini_db_long_U.csv"',
            '"multi_structural_gem_data.db"': '"databases/multi_structural_gem_data.db"',
            '"gemlib_structural_ready.csv"': '"databases/gemlib_structural_ready.csv"',
            
            # Unknown file paths
            '"unkgemB.csv"': '"data/unknown/unkgemB.csv"',
            '"unkgemL.csv"': '"data/unknown/unkgemL.csv"',
            '"unkgemU.csv"': '"data/unknown/unkgemU.csv"',
            
            # Program paths
            '"src/structural_analysis/main.py"': '"src/structural_analysis/gemini_launcher.py"',
            '"src/numerical_analysis/gemini1.py"': '"src/numerical_analysis/gemini1.py"',
            '"fast_gem_analysis.py"': '"src/core/fast_gem_analysis.py"',
        }
        
        for old_path, new_path in replacements.items():
            content = content.replace(old_path, new_path)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def create_config_files(self):
        """Create configuration files"""
        print("\n⚙️  Creating configuration files...")
        
        base_path = Path(self.target_dir)
        
        # settings.yaml
        settings_content = """# Gemini Gemological Analysis Settings
system:
  name: "Gemini Gemological Analysis"
  version: "2.0.0"
  
data_paths:
  raw_data: "data/raw"
  unknown_data: "data/unknown"
  reference_data: "data/reference"
  databases: "databases"
  
analysis:
  max_matches_display: 50
  export_formats: ["csv", "json", "xlsx"]
  
visualization:
  plot_style: "professional"
  color_scheme: "gemini"
  figure_size: [15, 8]
"""
        
        with open(base_path / "config/settings.yaml", 'w') as f:
            f.write(settings_content)
        
        # light_source_config.yaml
        light_config_content = """# Light Source Configuration
normalization:
  B:  # Halogen/Broadband
    reference_wavelength: 650
    target_intensity: 50000
    final_scale: 100
    
  L:  # Laser
    reference_type: "maximum"
    target_intensity: 50000
    final_scale: 100
    
  U:  # UV
    reference_wavelength: 811
    reference_window: [810.5, 811.5]
    target_intensity: 15000
    final_scale: 100

colors:
  B: "#FF6B35"  # Orange
  L: "#004E98"  # Blue  
  U: "#7209B7"  # Purple
"""
        
        with open(base_path / "config/light_source_config.yaml", 'w') as f:
            f.write(light_config_content)
        
        print("   ✅ Created configuration files")
    
    def create_documentation(self):
        """Create basic documentation"""
        print("\n📚 Creating documentation...")
        
        base_path = Path(self.target_dir)
        
        # README.md
        readme_content = """# Gemini Gemological Analysis System

Advanced multi-spectral gemstone identification and analysis system.

## Features

- Multi-light source spectral analysis (Halogen, Laser, UV)
- Manual and automated feature marking
- Advanced peak detection algorithms
- Comprehensive database matching
- Interactive results visualization
- Score summaries and rankings

## Quick Start

```bash
python main.py
```

## Directory Structure

- `src/` - Source code modules
- `data/` - Data files and analysis results
- `databases/` - Reference databases
- `config/` - Configuration files
- `docs/` - Documentation

## Analysis Workflow

1. Place raw .txt spectral files in `data/raw/`
2. Run main.py and select "Select Gem for Analysis"
3. Choose unknown gem from available files
4. System automatically converts and analyzes
5. View results in interactive display

## Support

For issues and questions, refer to the technical documentation in `docs/`.
"""
        
        with open(base_path / "README.md", 'w') as f:
            f.write(readme_content)
        
        print("   ✅ Created README.md")
    
    def copy_data_files(self):
        """Copy existing data files to appropriate locations"""
        print("\n📊 Copying data files...")
        
        # Copy raw_txt to new location if it exists
        if os.path.exists("raw_txt"):
            shutil.copytree("raw_txt", f"{self.target_dir}/raw_txt", dirs_exist_ok=True)
            print("   ✅ Copied raw_txt directory")
        
        # Copy any existing unknown files
        unknown_files = ["unkgemB.csv", "unkgemL.csv", "unkgemU.csv"]
        for unk_file in unknown_files:
            if os.path.exists(unk_file):
                shutil.copy2(unk_file, f"{self.target_dir}/data/unknown/")
                print(f"   ✅ Copied {unk_file}")
    
    def create_results_visualizer(self):
        """Create the results visualizer module"""
        print("\n🎨 Creating results visualizer...")
        
        # The result_visualizer.py content would be inserted here
        # For now, create a placeholder that references the artifact
        
        base_path = Path(self.target_dir)
        viz_path = base_path / "src/visualization/match_display.py"
        
        placeholder_content = '''"""
Match Display Module - Placeholder
Copy the content from result_visualizer.py artifact to this file
"""

# TODO: Implement comprehensive results visualization
# See result_visualizer.py artifact for full implementation

def display_analysis_results(all_matches, gem_best_scores, gem_best_names, final_sorted, light_sources):
    """Placeholder for results display"""
    print("🎨 Results visualization not yet implemented")
    print("📋 Copy result_visualizer.py content to src/visualization/match_display.py")
'''
        
        with open(viz_path, 'w') as f:
            f.write(placeholder_content)
        
        print("   ✅ Created visualization placeholder")
    
    def run_full_migration(self):
        """Run complete migration process"""
        print("🚀 Starting Gemini Gemological Analysis Migration")
        print("=" * 60)
        
        try:
            self.create_directory_structure()
            self.create_init_files()
            self.migrate_files()
            self.copy_data_files()
            self.update_import_paths()
            self.create_config_files()
            self.create_documentation()
            self.create_results_visualizer()
            
            print("\n" + "=" * 60)
            print("✅ MIGRATION COMPLETE!")
            print("=" * 60)
            print(f"🎯 New system location: {self.target_dir}/")
            print("\n📋 Next Steps:")
            print("1. Copy result_visualizer.py content to src/visualization/match_display.py")
            print("2. Test the system: cd gemini_gemological_analysis && python main.py")
            print("3. Place sample .txt files in data/raw/ for testing")
            print("4. Verify database files are accessible")
            print("5. Check data/raw_txt/, data/unknown/, and data/structural_data/ directories")
            print("\n🔬 Your Gemini system is ready for advanced analysis!")
            
        except Exception as e:
            print(f"\n❌ Migration error: {e}")
            print("Please check file permissions and try again.")
            return False
        
        return True

def main():
    """Main migration function"""
    print("🔬 Gemini Gemological Analysis - Migration Tool")
    print("This will create a new 'gemini_gemological_analysis' directory")
    
    response = input("\nProceed with migration? (y/n): ").strip().lower()
    
    if response == 'y':
        migrator = GeminiMigrator()
        migrator.run_full_migration()
    else:
        print("Migration cancelled.")

if __name__ == "__main__":
    main()