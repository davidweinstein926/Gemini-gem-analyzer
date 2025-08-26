#!/usr/bin/env python3
"""MAIN.PY - OPTIMIZED - Enhanced Gemstone Analysis System Hub with UV Peak Validation
OPTIMIZED: Reduced line count while maintaining all functionality
"""

import os
import sys
import subprocess
import sqlite3

class SystemHub:
    def __init__(self):
        self.db_path = "multi_structural_gem_data.db"
        
        # System components configuration
        self.programs = {
            'gemini_launcher.py': 'Structural Analyzers (Manual + Automated)',
            'gemini_peak_detector.py': 'Standalone Peak Detector',
            'direct_import.py': 'CSV Data Import System',
            'uv_peak_validator.py': 'UV Peak Validator'
        }
        
        self.spectral_files = ['gemini_db_long_B.csv', 'gemini_db_long_L.csv', 'gemini_db_long_U.csv']
        
    def check_database_exists(self):
        """OPTIMIZED: Combined database existence and stats check"""
        if not os.path.exists(self.db_path):
            print(f"Database not found: {self.db_path}")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get counts in single query batch
            queries = [
                "SELECT COUNT(*) FROM structural_features",
                "SELECT COUNT(DISTINCT file) FROM structural_features",
                "SELECT name FROM sqlite_master WHERE type='table' AND name='automated_peaks'"
            ]
            
            count = cursor.execute(queries[0]).fetchone()[0]
            files = cursor.execute(queries[1]).fetchone()[0]
            auto_table_exists = cursor.execute(queries[2]).fetchone() is not None
            
            if auto_table_exists:
                auto_count = cursor.execute("SELECT COUNT(*) FROM automated_peaks").fetchone()[0]
                print(f"Database: {count:,} manual + {auto_count:,} automated records, {files} files")
            else:
                print(f"Database: {count:,} records, {files} files")
            
            conn.close()
            return True
        except Exception as e:
            print(f"Database exists but has issues: {e}")
            return False
    
    def check_system_components(self):
        """OPTIMIZED: Check all system components at once"""
        # Check spectral files
        spectral_found = sum(1 for f in self.spectral_files if os.path.exists(f))
        
        # Check program files
        program_status = {prog: os.path.exists(prog) for prog in self.programs.keys()}
        
        # Check advanced analyzer
        analyzer_ok = False
        try:
            from analysis.advanced_analyzer import AdvancedGemstoneAnalyzer
            analyzer_ok = True
            analyzer_location = "analysis/advanced_analyzer.py"
        except ImportError:
            analyzer_ok = os.path.exists('advanced_analyzer.py')
            analyzer_location = "advanced_analyzer.py"
        
        return spectral_found, program_status, analyzer_ok, analyzer_location
    
    def run_program(self, program_name, description):
        """OPTIMIZED: Unified program launcher"""
        if not os.path.exists(program_name):
            print(f"{program_name} not found")
            return False
        try:
            print(f"Launching {description}...")
            result = subprocess.run([sys.executable, program_name], capture_output=False, text=True)
            status = "completed" if result.returncode == 0 else "finished with warnings"
            print(f"{description} {status}")
            return True
        except Exception as e:
            print(f"Error running {description}: {e}")
            return False
    
    def show_system_status(self):
        """OPTIMIZED: Combined system status display"""
        print("ENHANCED GEMSTONE ANALYSIS SYSTEM STATUS")
        print("=" * 50)
        
        db_ok = self.check_database_exists()
        spectral_found, program_status, analyzer_ok, analyzer_location = self.check_system_components()
        
        print(f"\nSpectral files: {spectral_found}/3 found")
        
        print(f"\nProgram Files:")
        for prog, desc in self.programs.items():
            status = "OK" if program_status[prog] else "MISSING"
            print(f"   {status} {desc}: {prog}")
        
        status = "OK" if analyzer_ok else "MISSING"
        print(f"   {status} Advanced Unknown Stone Analyzer: {analyzer_location}")
        
        system_ready = db_ok and spectral_found == 3
        print(f"\nSystem Ready: {'YES' if system_ready else 'PARTIAL'}")
        print("=" * 50)
        
        return db_ok, system_ready
    
    def get_database_stats(self):
        """OPTIMIZED: Get database statistics"""
        if not os.path.exists(self.db_path):
            return None
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Combined statistics query
            stats = {}
            stats['total_manual'] = cursor.execute("SELECT COUNT(*) FROM structural_features").fetchone()[0]
            stats['files'] = cursor.execute("SELECT COUNT(DISTINCT file) FROM structural_features").fetchone()[0]
            stats['by_light'] = cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source").fetchall()
            
            # Check for automated peaks
            auto_table_exists = cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='automated_peaks'").fetchone() is not None
            
            if auto_table_exists:
                stats['total_auto'] = cursor.execute("SELECT COUNT(*) FROM automated_peaks").fetchone()[0]
                stats['auto_files'] = cursor.execute("SELECT COUNT(DISTINCT file) FROM automated_peaks").fetchone()[0]
            
            conn.close()
            return stats
        except Exception as e:
            print(f"Error reading database: {e}")
            return None
    
    def show_quick_stats(self):
        """OPTIMIZED: Display quick statistics"""
        stats = self.get_database_stats()
        if not stats:
            print("No database found")
            return
        
        print(f"\nQUICK STATISTICS:")
        print(f"   Total files analyzed: {stats['files']:,}")
        print(f"   Manual structural records: {stats['total_manual']:,}")
        
        if 'total_auto' in stats:
            print(f"   Automated peak records: {stats['total_auto']:,}")
            print(f"   Files with automated peaks: {stats['auto_files']:,}")
        
        print(f"   Manual records by light source:")
        light_icons = {'Halogen': 'HALOGEN', 'Laser': 'LASER', 'UV': 'UV'}
        for light, count in stats['by_light']:
            icon = light_icons.get(light, 'OTHER')
            print(f"      {icon}: {count:,}")
    
    def launch_unknown_analyzer(self):
        """OPTIMIZED: Launch unknown stone analyzer"""
        if not self.check_database_exists():
            print("Database required for unknown stone analysis")
            print("Use option 2 to import structural data first")
            return False
        
        # Try analysis module first, then root directory
        try:
            from analysis.advanced_analyzer import AdvancedGemstoneAnalyzer
            print("Starting advanced unknown stone analysis...")
            analyzer = AdvancedGemstoneAnalyzer(self.db_path)
            analyzer.batch_analysis_menu()
            return True
        except ImportError:
            if os.path.exists('advanced_analyzer.py'):
                print("Using root directory analyzer...")
                return self.run_program('advanced_analyzer.py', 'Advanced Unknown Stone Analyzer')
            else:
                print("No analyzer found!")
                print("Make sure advanced_analyzer.py is in analysis/ module or current directory")
                return False
        except Exception as e:
            print(f"Analyzer error: {e}")
            return False
    
    def launch_specialized_program(self, program_key, description_override=None):
        """OPTIMIZED: Launch specialized programs with validation"""
        program_name = program_key
        description = description_override or self.programs.get(program_key, program_key)
        
        if not os.path.exists(program_name):
            print(f"{program_name} not found!")
            if program_key == 'uv_peak_validator.py':
                print("Create uv_peak_validator.py from the provided code")
            else:
                print(f"Make sure {program_name} is in the current directory")
            return False
        
        # Special descriptions for certain programs
        special_descriptions = {
            'gemini_peak_detector.py': "AUTOMATED PEAK DETECTOR\nComputational peak detection with comparison capabilities",
            'uv_peak_validator.py': "UV PEAK VALIDATION SYSTEM\nFilter auto-detected UV peaks against reference and SNR thresholds\nValidates against UV source reference (S010123UC1)\nRemoves noise artifacts and spurious peaks",
            'direct_import.py': "CSV STRUCTURAL DATA IMPORT\nImporting from halogen/, laser/, uv/ folders\nScans for all *_structural_*.csv files"
        }
        
        if program_key in special_descriptions:
            print(special_descriptions[program_key])
        
        return self.run_program(program_name, description)
    
    def show_analysis_options(self):
        """OPTIMIZED: Show analysis options guide"""
        options_guide = {
            "STRUCTURAL ANALYZERS (Comprehensive Suite)": [
                "Manual feature marking - Interactive analysis",
                "Automated peak detection - Computational analysis",
                "Choose analysis type and light source",
                "Integrated results in unified interface"
            ],
            "CSV STRUCTURAL DATA IMPORT": [
                "Import all manual structural CSV files",
                "Scans halogen/, laser/, uv/ folders automatically",
                "Proven reliable (imported your 69 files successfully)",
                "Handles duplicates intelligently"
            ],
            "UNKNOWN STONE ANALYZER": [
                "Spectral matching against database",
                "Statistical comparison algorithms",
                "Identification confidence scoring"
            ],
            "STANDALONE PEAK DETECTOR": [
                "Direct computational peak detection",
                "Compare against reference peak lists",
                "Adjustable algorithm parameters",
                "Export results in multiple formats"
            ],
            "UV PEAK VALIDATOR": [
                "Validate auto-detected UV peaks",
                "Filter against UV source reference",
                "SNR-based noise rejection",
                "Quality assurance for database import"
            ]
        }
        
        print(f"\nANALYSIS OPTIONS EXPLAINED:")
        print("=" * 45)
        
        for i, (title, features) in enumerate(options_guide.items(), 1):
            print(f"\n{i}. {title}:")
            for feature in features:
                print(f"   {feature}")
        
        print(f"\nWORKFLOW RECOMMENDATIONS:")
        recommendations = [
            "Import data: Use option 2 (proven reliable with your 69 files)",
            "New UV data: Validate auto-peaks first (option 5)",
            "Complex spectra: Manual analysis for detailed interpretation (option 1)",
            "Unknown stones: Use analyzer after building database (option 3)",
            "Database stats: Check your data anytime (option 6)"
        ]
        
        for rec in recommendations:
            print(f"   {rec}")

    def main_menu(self):
        """OPTIMIZED: Main menu system"""
        menu_options = [
            ("Launch Structural Analyzers (manual + automated)", lambda: self.run_program('gemini_launcher.py', 'Structural Analyzers')),
            ("Import CSV Structural Data", lambda: self.launch_specialized_program('direct_import.py')),
            ("Analyze Unknown Stone (spectral matching)", self.launch_unknown_analyzer),
            ("Launch Standalone Peak Detector", lambda: self.launch_specialized_program('gemini_peak_detector.py')),
            ("Validate UV Auto-Detected Peaks", lambda: self.launch_specialized_program('uv_peak_validator.py')),
            ("Show Database Statistics", self.show_quick_stats),
            ("Show Analysis Options Guide", self.show_analysis_options),
            ("System Status & Health Check", lambda: self.show_system_status()),
            ("Exit", lambda: None)
        ]
        
        while True:
            print("\n" + "="*55)
            print("ENHANCED GEMSTONE ANALYSIS SYSTEM HUB")
            print("="*55)
            self.show_system_status()
            
            print(f"\nMAIN MENU:")
            for i, (description, _) in enumerate(menu_options, 1):
                print(f"{i}. {description}")
            
            choice = input(f"\nChoice (1-{len(menu_options)}): ").strip()
            
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(menu_options):
                    description, action = menu_options[choice_idx]
                    
                    if choice_idx == len(menu_options) - 1:  # Exit
                        print("Goodbye!")
                        break
                    
                    print(f"\n{description.upper()}")
                    if action:
                        action()
                    
                    if choice_idx < len(menu_options) - 1:  # Not exit
                        input("\nPress Enter to return to main menu...")
                else:
                    print("Invalid choice")
            except ValueError:
                print("Invalid choice")

def main():
    """OPTIMIZED: Main entry point"""
    try:
        hub = SystemHub()
        hub.main_menu()
    except KeyboardInterrupt:
        print("\nSystem interrupted - goodbye!")
    except Exception as e:
        print(f"System error: {e}")


if __name__ == "__main__":
    main()
