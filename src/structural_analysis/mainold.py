#!/usr/bin/env python3
"""MAIN.PY - ULTRA OPTIMIZED - Enhanced Gemstone Analysis System Hub
TRUE OPTIMIZATION: Significant code reduction while maintaining all functionality
"""

import os, sys, subprocess, sqlite3

class UltraOptimizedSystemHub:
    def __init__(self):
        self.db_path = "multi_structural_gem_data.db"
        # CONSOLIDATED CONFIG: All system configuration in single structure
        self.config = {
            'programs': {
                'gemini_launcher.py': 'Structural Analyzers (Manual + Automated)',
                'gemini_peak_detector.py': 'Standalone Peak Detector',
                'direct_import.py': 'CSV Data Import System',
                'uv_peak_validator.py': 'UV Peak Validator'
            },
            'spectral_files': ['gemini_db_long_B.csv', 'gemini_db_long_L.csv', 'gemini_db_long_U.csv'],
            'light_icons': {'Halogen': 'HALOGEN', 'Laser': 'LASER', 'UV': 'UV'},
            'analyzer_paths': ['analysis/advanced_analyzer.py', 'advanced_analyzer.py'],
            'special_descriptions': {
                'gemini_peak_detector.py': "AUTOMATED PEAK DETECTOR\nComputational peak detection with comparison capabilities",
                'uv_peak_validator.py': "UV PEAK VALIDATION SYSTEM\nFilter auto-detected UV peaks against reference and SNR thresholds",
                'direct_import.py': "CSV STRUCTURAL DATA IMPORT\nImporting from halogen/, laser/, uv/ folders"
            }
        }
    
    def execute_db_query(self, queries):
        """OPTIMIZED: Universal database query executor"""
        if not os.path.exists(self.db_path):
            return None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            results = [cursor.execute(q).fetchone() if 'COUNT' in q else cursor.execute(q).fetchall() for q in queries]
            conn.close()
            return results
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def get_system_status(self):
        """OPTIMIZED: Get complete system status in single method"""
        # Database status
        db_queries = [
            "SELECT COUNT(*) FROM structural_features",
            "SELECT COUNT(DISTINCT file) FROM structural_features", 
            "SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source",
            "SELECT name FROM sqlite_master WHERE type='table' AND name='automated_peaks'"
        ]
        
        db_results = self.execute_db_query(db_queries)
        db_ok = db_results is not None
        
        # System components status
        spectral_found = sum(1 for f in self.config['spectral_files'] if os.path.exists(f))
        programs_ok = {prog: os.path.exists(prog) for prog in self.config['programs']}
        
        # Advanced analyzer status
        analyzer_ok = False
        analyzer_location = None
        for path in self.config['analyzer_paths']:
            if os.path.exists(path):
                analyzer_ok = True
                analyzer_location = path
                break
        
        # Try import if file-based check failed
        if not analyzer_ok:
            try:
                from analysis.advanced_analyzer import AdvancedGemstoneAnalyzer
                analyzer_ok = True
                analyzer_location = "analysis/advanced_analyzer.py"
            except ImportError:
                pass
        
        return {
            'db_ok': db_ok, 'db_results': db_results,
            'spectral_found': spectral_found, 'programs_ok': programs_ok,
            'analyzer_ok': analyzer_ok, 'analyzer_location': analyzer_location,
            'system_ready': db_ok and spectral_found == 3
        }
    
    def display_system_status(self, status=None):
        """OPTIMIZED: Display comprehensive system status"""
        if not status:
            status = self.get_system_status()
        
        print("ENHANCED GEMSTONE ANALYSIS SYSTEM STATUS")
        print("=" * 50)
        
        # Database info
        if status['db_ok'] and status['db_results']:
            manual_count, files_count, by_light, auto_table = status['db_results']
            auto_exists = auto_table[0] is not None if auto_table else False
            
            if auto_exists:
                auto_results = self.execute_db_query(["SELECT COUNT(*) FROM automated_peaks"])
                auto_count = auto_results[0][0] if auto_results else 0
                print(f"Database: {manual_count[0]:,} manual + {auto_count:,} automated records, {files_count[0]} files")
            else:
                print(f"Database: {manual_count[0]:,} records, {files_count[0]} files")
        else:
            print(f"Database: NOT FOUND ({self.db_path})")
        
        # System components
        print(f"Spectral files: {status['spectral_found']}/3 found")
        print("Program Files:")
        for prog, desc in self.config['programs'].items():
            status_str = "OK" if status['programs_ok'][prog] else "MISSING"
            print(f"   {status_str} {desc}: {prog}")
        
        analyzer_status = "OK" if status['analyzer_ok'] else "MISSING"
        location = status['analyzer_location'] or "not found"
        print(f"   {analyzer_status} Advanced Unknown Stone Analyzer: {location}")
        
        print(f"System Ready: {'YES' if status['system_ready'] else 'PARTIAL'}")
        print("=" * 50)
        
        return status
    
    def show_quick_stats(self):
        """OPTIMIZED: Display database statistics"""
        queries = [
            "SELECT COUNT(*) FROM structural_features",
            "SELECT COUNT(DISTINCT file) FROM structural_features",
            "SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source",
            "SELECT name FROM sqlite_master WHERE type='table' AND name='automated_peaks'"
        ]
        
        results = self.execute_db_query(queries)
        if not results:
            print("No database found")
            return
        
        manual_count, files_count, by_light, auto_table = results
        print(f"\nQUICK STATISTICS:")
        print(f"   Total files analyzed: {files_count[0]:,}")
        print(f"   Manual structural records: {manual_count[0]:,}")
        
        # Check for automated data
        if auto_table and auto_table[0]:
            auto_results = self.execute_db_query(["SELECT COUNT(*) FROM automated_peaks", "SELECT COUNT(DISTINCT file) FROM automated_peaks"])
            if auto_results:
                print(f"   Automated peak records: {auto_results[0][0]:,}")
                print(f"   Files with automated peaks: {auto_results[1][0]:,}")
        
        print("   Manual records by light source:")
        for light, count in by_light:
            icon = self.config['light_icons'].get(light, 'OTHER')
            print(f"      {icon}: {count:,}")
    
    def run_program(self, program_name, description=None):
        """OPTIMIZED: Universal program launcher with validation"""
        if not os.path.exists(program_name):
            print(f"{program_name} not found!")
            if program_name == 'uv_peak_validator.py':
                print("Create uv_peak_validator.py from the provided code")
            return False
        
        # Show special descriptions
        if program_name in self.config['special_descriptions']:
            print(self.config['special_descriptions'][program_name])
        
        try:
            desc = description or self.config['programs'].get(program_name, program_name)
            print(f"Launching {desc}...")
            result = subprocess.run([sys.executable, program_name], capture_output=False, text=True)
            status = "completed" if result.returncode == 0 else "finished with warnings"
            print(f"{desc} {status}")
            return True
        except Exception as e:
            print(f"Error running {description or program_name}: {e}")
            return False
    
    def launch_unknown_analyzer(self):
        """OPTIMIZED: Launch unknown stone analyzer with fallback logic"""
        status = self.get_system_status()
        if not status['db_ok']:
            print("Database required for unknown stone analysis\nUse option 2 to import structural data first")
            return False
        
        # Try import method first, then file-based
        try:
            from analysis.advanced_analyzer import AdvancedGemstoneAnalyzer
            print("Starting advanced unknown stone analysis...")
            analyzer = AdvancedGemstoneAnalyzer(self.db_path)
            analyzer.batch_analysis_menu()
            return True
        except ImportError:
            if status['analyzer_ok']:
                print("Using root directory analyzer...")
                return self.run_program(status['analyzer_location'], 'Advanced Unknown Stone Analyzer')
            else:
                print("No analyzer found!\nMake sure advanced_analyzer.py is in analysis/ module or current directory")
                return False
        except Exception as e:
            print(f"Analyzer error: {e}")
            return False
    
    def show_analysis_guide(self):
        """OPTIMIZED: Comprehensive analysis options guide"""
        guide_data = [
            ("STRUCTURAL ANALYZERS (Comprehensive Suite)", [
                "Manual feature marking - Interactive analysis",
                "Automated peak detection - Computational analysis", 
                "Choose analysis type and light source",
                "Integrated results in unified interface"
            ]),
            ("CSV STRUCTURAL DATA IMPORT", [
                "Import all manual structural CSV files",
                "Scans halogen/, laser/, uv/ folders automatically",
                "Proven reliable (imported your 69 files successfully)",
                "Handles duplicates intelligently"
            ]),
            ("UNKNOWN STONE ANALYZER", [
                "Spectral matching against database",
                "Statistical comparison algorithms", 
                "Identification confidence scoring"
            ]),
            ("STANDALONE PEAK DETECTOR", [
                "Direct computational peak detection",
                "Compare against reference peak lists",
                "Adjustable algorithm parameters",
                "Export results in multiple formats"
            ]),
            ("UV PEAK VALIDATOR", [
                "Validate auto-detected UV peaks",
                "Filter against UV source reference",
                "SNR-based noise rejection",
                "Quality assurance for database import"
            ])
        ]
        
        recommendations = [
            "Import data: Use option 2 (proven reliable with your 69 files)",
            "New UV data: Validate auto-peaks first (option 5)",
            "Complex spectra: Manual analysis for detailed interpretation (option 1)",
            "Unknown stones: Use analyzer after building database (option 3)",
            "Database stats: Check your data anytime (option 6)"
        ]
        
        print(f"\nANALYSIS OPTIONS EXPLAINED:")
        print("=" * 45)
        for i, (title, features) in enumerate(guide_data, 1):
            print(f"\n{i}. {title}:")
            for feature in features:
                print(f"   {feature}")
        
        print(f"\nWORKFLOW RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")
    
    def main_menu(self):
        """OPTIMIZED: Streamlined main menu system"""
        # CONSOLIDATED MENU: All options in single data structure
        menu_actions = [
            ("Launch Structural Analyzers (manual + automated)", lambda: self.run_program('gemini_launcher.py')),
            ("Import CSV Structural Data", lambda: self.run_program('direct_import.py')),
            ("Analyze Unknown Stone (spectral matching)", self.launch_unknown_analyzer),
            ("Launch Standalone Peak Detector", lambda: self.run_program('gemini_peak_detector.py')),
            ("Validate UV Auto-Detected Peaks", lambda: self.run_program('uv_peak_validator.py')),
            ("Show Database Statistics", self.show_quick_stats),
            ("Show Analysis Options Guide", self.show_analysis_guide),
            ("System Status & Health Check", lambda: self.display_system_status()),
            ("Exit", None)
        ]
        
        while True:
            print("\n" + "="*55)
            print("ENHANCED GEMSTONE ANALYSIS SYSTEM HUB")
            print("="*55)
            status = self.display_system_status()
            
            print(f"\nMAIN MENU:")
            for i, (description, _) in enumerate(menu_actions, 1):
                print(f"{i}. {description}")
            
            try:
                choice = int(input(f"\nChoice (1-{len(menu_actions)}): ").strip()) - 1
                if 0 <= choice < len(menu_actions):
                    description, action = menu_actions[choice]
                    
                    if action is None:  # Exit
                        print("Goodbye!")
                        break
                    
                    print(f"\n{description.upper()}")
                    action()
                    
                    if choice < len(menu_actions) - 1:  # Not exit
                        input("\nPress Enter to return to main menu...")
                else:
                    print("Invalid choice")
            except (ValueError, KeyboardInterrupt):
                if input("\nExit? (y/N): ").lower().startswith('y'):
                    print("Goodbye!")
                    break

def main():
    """OPTIMIZED: Streamlined main entry point with enhanced error handling"""
    try:
        UltraOptimizedSystemHub().main_menu()
    except KeyboardInterrupt:
        print("\nSystem interrupted - goodbye!")
    except Exception as e:
        print(f"System error: {e}")

if __name__ == "__main__":
    main()
