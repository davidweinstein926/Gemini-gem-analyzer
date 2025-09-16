#!/usr/bin/env python3
"""
COMPLETE GEMINI ANALYSIS SYSTEM - FIXED WITH SMART FALLBACK
Smart directory fallback: raw_temp → raw (archive) + InputSubmission fixes
"""
import os, sys, subprocess, sqlite3, pandas as pd, numpy as np, threading, time, shutil, json, re
from datetime import datetime
from pathlib import Path

# Audio support
try: import winsound; HAS_AUDIO = True
except ImportError:
    try: import pygame; pygame.mixer.init(); HAS_AUDIO = True
    except ImportError: HAS_AUDIO = False

class CompactGeminiSystem:
    def __init__(self):
        self.db_path = "database/structural_spectra/multi_structural_gem_data.db"
        self.program_files = {
            'data_acquisition/main_menu.py': 'Data Acquisition System',
            'src/structural_analysis/gemini_launcher.py': 'Structural Marking System',
            'src/numerical_analysis/gemini1.py': 'Numerical Analysis Engine',
            'src/structural_analysis/enhanced_gem_analyzer.py': 'Structural Matching Engine',
            'txt_to_split_long_format.py': 'Numerical DB Import Tool',
            'database/batch_importer.py': 'Structural DB Import Tool'
        }
        self.bleep_enabled = True
        self.init_directories()
    
    def init_directories(self):
        """Initialize all required directories"""
        directories = [
            "data/raw_temp", "data/raw (archive)", "data/structural_data", "data/structural(archive)",
            "outputs/numerical_analysis/reports", "outputs/numerical_analysis/graphs",
            "outputs/structural_results/reports", "outputs/structural_results/graphs", 
            "database/reference_spectra", "database/structural_spectra",
            "results(archive)/post_analysis_numerical/reports", "results(archive)/post_analysis_numerical/graphs",
            "results(archive)/post_analysis_structural/reports", "results(archive)/post_analysis_structural/graphs"
        ]
        for directory in directories: Path(directory).mkdir(parents=True, exist_ok=True)
    
    def play_bleep(self, feature_type="standard"):
        """Play audio bleep"""
        if not self.bleep_enabled or not HAS_AUDIO: return
        try:
            freq = {'peak': 1000, 'valley': 600, 'plateau': 800, 'significant': 1200, 'completion': 400}.get(feature_type, 800)
            if 'winsound' in sys.modules: winsound.Beep(freq, 200)
            elif 'pygame' in sys.modules:
                arr = np.zeros(4410); [arr.__setitem__(i, np.sin(2 * np.pi * freq * i / 22050)) for i in range(4410)]
                pygame.sndarray.make_sound((arr * 32767).astype(np.int16)).play(); time.sleep(0.2)
        except Exception as e: print(f"Audio bleep error: {e}")
    
    def safe_input(self, prompt):
        """Safe input handling - fixes InputSubmission wrapper"""
        try:
            user_input = str(input(prompt)).strip()
            if "InputSubmission(data='" in user_input and user_input.endswith("')"): user_input = user_input[22:-2]
            return user_input.replace('\\n', '').replace('\n', '').strip("'\"")
        except Exception as e: print(f"Input error: {e}"); return ""
    
    def check_directory_files(self, directory, file_pattern="*.txt"):
        """Check if directory has files matching pattern"""
        if not Path(directory).exists(): return []
        return list(Path(directory).glob(file_pattern))
    
    def setup_numerical_environment(self, data_directory):
        """Setup environment variables for gemini1.py to use specific directory"""
        os.environ['GEMINI_DATA_PATH'] = str(Path(data_directory).absolute())
        print(f"📁 Data source: {data_directory}")
        return str(Path(data_directory).absolute())
    
    # MENU OPTION 1: DATA ACQUISITION
    def data_acquisition(self):
        """Option 1: Data acquisition - capture spectra to data/raw_temp"""
        print("\n📡 DATA ACQUISITION - SPECTRAL CAPTURE\n" + "=" * 60)
        print("Input: Spectrophotometer (LR1)")
        print("Output: root/data/raw_temp/")
        if os.path.exists('data_acquisition/main_menu.py'):
            try:
                result = subprocess.run([sys.executable, 'data_acquisition/main_menu.py'], capture_output=False, text=True)
                if result.returncode == 0: print("✅ Data acquisition completed"); self.play_bleep("completion")
            except Exception as e: print(f"❌ Error: {e}")
        else: print("❌ data_acquisition/main_menu.py not found")
    
    # MENU OPTION 2: STRUCTURAL MARKING  
    def structural_marking(self):
        """Option 2: Structural marking - mark features from raw_temp (with fallback)"""
        print("\n🔬 STRUCTURAL MARKING SYSTEM\n" + "=" * 60)
        
        # Smart fallback logic
        raw_temp_files = self.check_directory_files("data/raw_temp")
        archive_files = self.check_directory_files("data/raw (archive)")
        
        if raw_temp_files:
            print("Input: root/data/raw_temp/ (work-in-progress)")
            data_source = "data/raw_temp"
        elif archive_files:
            print("Input: root/data/raw (archive)/ (fallback - no new captures)")
            data_source = "data/raw (archive)"
            print(f"📁 Found {len(archive_files)} archived files for marking")
        else:
            print("❌ No spectral files found in raw_temp or archive")
            print("💡 Use Option 1 to capture spectra first")
            return
        
        print("Output: root/data/structural_data/ (halogen,laser,uv in one folder)")
        
        if os.path.exists('src/structural_analysis/gemini_launcher.py'):
            try:
                # Set environment for structural marking
                self.setup_numerical_environment(data_source)
                result = subprocess.run([sys.executable, 'src/structural_analysis/gemini_launcher.py'], capture_output=False, text=True)
                if result.returncode == 0: print("✅ Structural marking completed"); self.play_bleep("completion")
            except Exception as e: print(f"❌ Error: {e}")
        else: print("❌ src/structural_analysis/gemini_launcher.py not found")
    
    # MENU OPTION 3: NUMERICAL MATCHING
    def numerical_matching(self):
        """Option 3: Numerical matching - smart fallback from raw_temp to archive"""
        print("\n📊 NUMERICAL MATCHING\n" + "=" * 60)
        
        # Smart fallback logic - this is the CORRECT behavior
        raw_temp_files = self.check_directory_files("data/raw_temp")
        archive_files = self.check_directory_files("data/raw (archive)")
        
        if raw_temp_files:
            print("Input: root/data/raw_temp/ (work-in-progress)")
            data_source = "data/raw_temp"
            print(f"📁 Found {len(raw_temp_files)} new files for analysis")
        elif archive_files:
            print("Input: root/data/raw (archive)/ (smart fallback - no new captures)")
            data_source = "data/raw (archive)"
            print(f"📁 Found {len(archive_files)} archived files for analysis")
        else:
            print("❌ No spectral files found in raw_temp or archive")
            print("💡 Use Option 1 to capture spectra first")
            return
        
        print("Output: root/outputs/numerical_analysis/reports;graphs")
        
        if os.path.exists('src/numerical_analysis/gemini1.py'):
            try:
                # Setup environment and run with InputSubmission fix
                data_path = self.setup_numerical_environment(data_source)
                
                # Create input file to bypass InputSubmission issues
                input_file = Path("temp_numerical_input.txt")
                with open(input_file, 'w') as f:
                    f.write("auto_analysis\n")  # Signal for auto-analysis mode
                
                # Run with stdin redirection
                with open(input_file, 'r') as f:
                    result = subprocess.run([sys.executable, 'src/numerical_analysis/gemini1.py'], 
                                          stdin=f, capture_output=False, text=True)
                
                # Cleanup
                if input_file.exists(): input_file.unlink()
                
                if result.returncode == 0: print("✅ Numerical matching completed"); self.play_bleep("completion")
            except Exception as e: print(f"❌ Error: {e}")
        else: print("❌ src/numerical_analysis/gemini1.py not found")
    
    # MENU OPTION 4: STRUCTURAL MATCHING
    def structural_matching(self):
        """Option 4: Structural matching - match structural data vs database"""
        print("\n🔬 STRUCTURAL MATCHING (UNKNOWN GEMS)\n" + "=" * 60)
        
        # Check for structural files
        structural_files = self.check_directory_files("data/structural_data", "*.csv")
        if not structural_files:
            print("❌ No structural files found in data/structural_data/")
            print("💡 Use Option 2 to mark structural features first")
            return
        
        print("Input: root/data/structural_data/ (halogen,laser,uv)")
        print("Output: root/outputs/structural_results/reports;graphs")
        print(f"📁 Found {len(structural_files)} structural files for matching")
        
        if os.path.exists('src/structural_analysis/enhanced_gem_analyzer.py'):
            try:
                # Copy structural files to analyzer's expected location for unknown analysis
                unknown_dir = Path("data/unknown/structural")
                unknown_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy files from structural_data to unknown location
                for file in structural_files:
                    if 'structural' in file.name.lower():
                        shutil.copy2(file, unknown_dir / file.name)
                
                result = subprocess.run([sys.executable, 'src/structural_analysis/enhanced_gem_analyzer.py'], capture_output=False, text=True)
                if result.returncode == 0: 
                    print("✅ Structural matching completed"); 
                    # Move results to proper output location
                    self.move_structural_results_to_outputs()
                    self.play_bleep("completion")
            except Exception as e: print(f"❌ Error: {e}")
        else: print("❌ src/structural_analysis/enhanced_gem_analyzer.py not found")
    
    def move_structural_results_to_outputs(self):
        """Move structural results to proper output location"""
        try:
            # Move from results/structural/ to outputs/structural_results/
            if Path("results/structural").exists():
                if Path("results/structural/reports").exists():
                    for file in Path("results/structural/reports").glob("*"):
                        shutil.move(str(file), f"outputs/structural_results/reports/{file.name}")
                if Path("results/structural/graphs").exists():
                    for file in Path("results/structural/graphs").glob("*"):
                        shutil.move(str(file), f"outputs/structural_results/graphs/{file.name}")
                print("📦 Results moved to outputs/structural_results/")
        except Exception as e: print(f"⚠️ Result move error: {e}")
    
    # MENU OPTION 5: IMPORT TO NUMERICAL DB
    def import_to_numerical_db(self):
        """Option 5: Import to numerical database"""
        print("\n💾 IMPORT TO NUMERICAL DATABASE\n" + "=" * 60)
        
        # Check for files to import
        raw_temp_files = self.check_directory_files("data/raw_temp")
        if not raw_temp_files:
            print("❌ No files found in data/raw_temp/")
            print("💡 Use Option 1 to capture spectra first")
            return
        
        print("Input: root/data/raw_temp")
        print("Output: root/database/reference_spectra/gemini_db_long_*.csv")
        print(f"📁 Found {len(raw_temp_files)} files to import")
        
        if os.path.exists('txt_to_split_long_format.py'):
            try:
                self.setup_numerical_environment("data/raw_temp")
                result = subprocess.run([sys.executable, 'txt_to_split_long_format.py'], capture_output=False, text=True)
                if result.returncode == 0: print("✅ Numerical database import completed"); self.play_bleep("completion")
            except Exception as e: print(f"❌ Error: {e}")
        else: print("❌ txt_to_split_long_format.py not found")
    
    # MENU OPTION 6: IMPORT TO STRUCTURAL DB
    def import_to_structural_db(self):
        """Option 6: Import to structural database"""
        print("\n💾 IMPORT TO STRUCTURAL DATABASE\n" + "=" * 60)
        
        structural_files = self.check_directory_files("data/structural_data", "*.csv")
        if not structural_files:
            print("❌ No structural files found in data/structural_data/")
            print("💡 Use Option 2 to mark structural features first")
            return
        
        print("Input: root/data/structural_data/ (halogen,laser,uv)")
        print("Output: root/database/structural_spectra/multi_structural_gem_data.db")
        print("        root/database/structural_spectra/gemini_structural_db.csv")
        print(f"📁 Found {len(structural_files)} structural files")
        
        confirm = self.safe_input(f"Import {len(structural_files)} files to structural databases? (y/n): ")
        if confirm.lower() != 'y': return
        
        try:
            success_count = 0
            # Import using batch_importer.py
            if os.path.exists('database/batch_importer.py'):
                result = subprocess.run([sys.executable, 'database/batch_importer.py'], capture_output=False, text=True)
                if result.returncode == 0: success_count += 1
            
            # Import using append_full_name_to_structuraldb.py  
            if os.path.exists('database/append_full_name_to_structuraldb.py'):
                result = subprocess.run([sys.executable, 'database/append_full_name_to_structuraldb.py'], capture_output=False, text=True)
                if result.returncode == 0: success_count += 1
            
            if success_count > 0: print("✅ Structural database import completed"); self.play_bleep("completion")
            else: print("❌ Database import failed")
        except Exception as e: print(f"❌ Import error: {e}")
    
    # MENU OPTION 7: NUMERICAL MATCHING (TEST)
    def numerical_matching_test(self):
        """Option 7: Numerical matching test - specifically use archived data"""
        print("\n📊 NUMERICAL MATCHING (TEST)\n" + "=" * 60)
        
        archive_files = self.check_directory_files("data/raw (archive)")
        if not archive_files:
            print("❌ No archived files found in data/raw (archive)/")
            print("💡 Use Option 9 to archive files from raw_temp first")
            return
        
        print("Input: root/data/raw (archive)/ (archived data for testing)")
        print("Output: root/outputs/numerical_analysis/reports;graphs")
        print(f"📁 Found {len(archive_files)} archived files for testing")
        
        if os.path.exists('src/numerical_analysis/gemini1.py'):
            try:
                # Setup for archived data testing
                data_path = self.setup_numerical_environment("data/raw (archive)")
                
                # Create test input file
                input_file = Path("temp_test_input.txt")
                with open(input_file, 'w') as f:
                    f.write("test_mode\n")  # Signal for test mode
                
                # Run with test configuration
                with open(input_file, 'r') as f:
                    result = subprocess.run([sys.executable, 'src/numerical_analysis/gemini1.py'], 
                                          stdin=f, capture_output=False, text=True)
                
                # Cleanup
                if input_file.exists(): input_file.unlink()
                
                if result.returncode == 0: print("✅ Numerical test completed"); self.play_bleep("completion")
            except Exception as e: print(f"❌ Error: {e}")
        else: print("❌ src/numerical_analysis/gemini1.py not found")
    
    # MENU OPTION 8: STRUCTURAL MATCHING (TEST)
    def structural_matching_test(self):
        """Option 8: Structural matching test - use archived structural data"""
        print("\n🔬 STRUCTURAL MATCHING (TEST)\n" + "=" * 60)
        
        archive_files = self.check_directory_files("data/structural(archive)", "*.csv")
        if not archive_files:
            print("❌ No archived structural files found in data/structural(archive)/")
            print("💡 Use Option 10 to archive structural files first")
            return
        
        print("Input: root/data/structural(archive)/ (archived data for testing)")
        print("Output: root/outputs/structural_results/reports;graphs")
        print(f"📁 Found {len(archive_files)} archived structural files for testing")
        
        if os.path.exists('src/structural_analysis/enhanced_gem_analyzer.py'):
            try:
                # Copy archived files to unknown location for testing
                unknown_dir = Path("data/unknown/structural")
                unknown_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy sample of archived files for testing
                test_files = archive_files[:10]  # Test with first 10 files
                for file in test_files:
                    shutil.copy2(file, unknown_dir / file.name)
                
                print(f"🧪 Testing with {len(test_files)} archived files")
                result = subprocess.run([sys.executable, 'src/structural_analysis/enhanced_gem_analyzer.py'], capture_output=False, text=True)
                if result.returncode == 0: 
                    print("✅ Structural test completed"); 
                    self.move_structural_results_to_outputs()
                    self.play_bleep("completion")
            except Exception as e: print(f"❌ Error: {e}")
        else: print("❌ src/structural_analysis/enhanced_gem_analyzer.py not found")
    
    # MENU OPTION 9: CLEAN UP NUMERICAL
    def cleanup_numerical(self):
        """Option 9: Clean up numerical - archive data and results"""
        print("\n🧹 CLEAN UP NUMERICAL\n" + "=" * 60)
        print("a) Archive: root/data/raw_temp → root/data/raw (archive)")
        print("b) Archive: root/outputs/numerical_analysis/ → root/results(archive)/post_analysis_numerical/")
        
        try:
            archived_files = 0
            # 9a: Archive raw_temp to raw (archive)
            raw_temp_files = self.check_directory_files("data/raw_temp")
            if raw_temp_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for file in raw_temp_files:
                    if file.is_file():
                        dest_name = f"{file.stem}_{timestamp}{file.suffix}"
                        shutil.move(str(file), f"data/raw (archive)/{dest_name}")
                        archived_files += 1
                print(f"✅ Archived {archived_files} files from raw_temp to raw (archive)")
            
            # 9b: Archive numerical results
            results_moved = 0
            if Path("outputs/numerical_analysis/reports").exists():
                for file in Path("outputs/numerical_analysis/reports").glob("*"):
                    shutil.move(str(file), f"results(archive)/post_analysis_numerical/reports/{file.name}")
                    results_moved += 1
            if Path("outputs/numerical_analysis/graphs").exists():
                for file in Path("outputs/numerical_analysis/graphs").glob("*"):
                    shutil.move(str(file), f"results(archive)/post_analysis_numerical/graphs/{file.name}")
                    results_moved += 1
            
            if results_moved > 0: print(f"✅ Archived {results_moved} result files")
            print("✅ Numerical cleanup completed"); self.play_bleep("completion")
            
        except Exception as e: print(f"❌ Cleanup error: {e}")
    
    # MENU OPTION 10: CLEAN UP STRUCTURAL
    def cleanup_structural(self):
        """Option 10: Clean up structural - archive data and results"""
        print("\n🧹 CLEAN UP STRUCTURAL\n" + "=" * 60)
        print("a) Archive: root/data/structural_data/ → root/data/structural(archive)")
        print("b) Archive: root/outputs/structural_results/ → root/results(archive)/post_analysis_structural/")
        
        try:
            archived_files = 0
            # 10a: Archive structural_data to structural(archive)
            structural_files = self.check_directory_files("data/structural_data", "*.csv")
            if structural_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for file in structural_files:
                    dest_name = f"{file.stem}_{timestamp}{file.suffix}"
                    shutil.move(str(file), f"data/structural(archive)/{dest_name}")
                    archived_files += 1
                print(f"✅ Archived {archived_files} structural files")
            
            # 10b: Archive structural results
            results_moved = 0
            if Path("outputs/structural_results/reports").exists():
                for file in Path("outputs/structural_results/reports").glob("*"):
                    shutil.move(str(file), f"results(archive)/post_analysis_structural/reports/{file.name}")
                    results_moved += 1
            if Path("outputs/structural_results/graphs").exists():
                for file in Path("outputs/structural_results/graphs").glob("*"):
                    shutil.move(str(file), f"results(archive)/post_analysis_structural/graphs/{file.name}")
                    results_moved += 1
            
            if results_moved > 0: print(f"✅ Archived {results_moved} result files")
            print("✅ Structural cleanup completed"); self.play_bleep("completion")
            
        except Exception as e: print(f"❌ Cleanup error: {e}")
    
    # MENU OPTION 11: SYSTEM STATUS
    def system_status(self):
        """Option 11: System status with smart directory checking"""
        print("\nSYSTEM STATUS\n" + "=" * 40)
        
        # Check program files
        print("📋 Program Files:")
        for file_path, description in self.program_files.items():
            status = "✅" if os.path.exists(file_path) else "❌"
            print(f"{status} {description}: {file_path}")
        
        # Check key directories with file counts
        print(f"\n📁 Directory Status:")
        key_dirs = [
            ("data/raw_temp", "Raw temp files", "*.txt"),
            ("data/raw (archive)", "Archived raw files", "*.txt"), 
            ("data/structural_data", "Structural marked files", "*.csv"),
            ("data/structural(archive)", "Archived structural files", "*.csv"),
            ("outputs/numerical_analysis/reports", "Numerical reports", "*"),
            ("outputs/structural_results/reports", "Structural reports", "*")
        ]
        
        for dir_path, description, pattern in key_dirs:
            files = self.check_directory_files(dir_path, pattern)
            file_count = len(files)
            status = "✅" if file_count > 0 else "⚪"
            print(f"{status} {description}: {file_count} files in {dir_path}")
        
        # Database status
        if os.path.exists(self.db_path):
            size_kb = os.path.getsize(self.db_path) / 1024
            print(f"✅ Structural database: {size_kb:.1f} KB")
        else:
            print(f"❌ Structural database not found: {self.db_path}")
        
        # Environment check
        gemini_path = os.environ.get('GEMINI_DATA_PATH', 'Not set')
        print(f"\n🔧 Environment: GEMINI_DATA_PATH = {gemini_path}")
        print(f"🔊 Audio: {'Available' if HAS_AUDIO else 'Not available'} | Bleep: {'ON' if self.bleep_enabled else 'OFF'}")
    
    # MENU OPTION 12: TOGGLE BLEEP
    def toggle_bleep(self):
        """Option 12: Toggle bleep system"""
        self.bleep_enabled = not self.bleep_enabled
        print(f"🔊 Bleep system: {'ENABLED' if self.bleep_enabled else 'DISABLED'}")
        if self.bleep_enabled and HAS_AUDIO: self.play_bleep("completion")
    
    # MAIN MENU
    def run_main_menu(self):
        """Complete main menu with smart fallback logic"""
        print(f"\n{'='*70}\n  COMPLETE GEMINI ANALYSIS SYSTEM\n  Smart Directory Fallback + InputSubmission Fixes\n{'='*70}")
        
        while True:
            print(f"\nMAIN MENU (Smart Fallback Logic):")
            print("=" * 45)
            print("📡 DATA CAPTURE & ANALYSIS:")
            print("1. Data Acquisition (Spectral Capture)")
            print("2. Structural Marking (smart fallback)")  
            print("3. Numerical Matching (smart fallback)")
            print("4. Structural Matching")
            print("")
            print("💾 DATABASE OPERATIONS:")
            print("5. Import to Numerical DB")
            print("6. Import to Structural DB")
            print("")
            print("🧪 TESTING (Archived Data):")
            print("7. Numerical Matching (Test)")
            print("8. Structural Matching (Test)")
            print("")
            print("🧹 CLEANUP & ARCHIVING:")
            print("9. Clean Up Numerical")
            print("10. Clean Up Structural")
            print("")
            print("⚙️ SYSTEM:")
            print("11. System Status")
            print("12. Toggle Bleep System") 
            print("13. Exit")
            
            print(f"\nStatus: Bleep [{'ON' if self.bleep_enabled else 'OFF'}] | Smart Fallback Active")
            
            try:
                choice = self.safe_input("\nSelect (1-13): ")
                
                actions = {
                    '1': self.data_acquisition, '2': self.structural_marking, '3': self.numerical_matching,
                    '4': self.structural_matching, '5': self.import_to_numerical_db, '6': self.import_to_structural_db,
                    '7': self.numerical_matching_test, '8': self.structural_matching_test, 
                    '9': self.cleanup_numerical, '10': self.cleanup_structural,
                    '11': self.system_status, '12': self.toggle_bleep
                }
                
                if choice == '13':
                    print("Exiting system..."); self.play_bleep("completion") if self.bleep_enabled else None; break
                elif choice in actions:
                    actions[choice]()
                else:
                    print("Invalid choice. Please select 1-13.")
                
                if choice != '13': input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt: print("\n\nSystem interrupted. Exiting..."); break
            except Exception as e: print(f"\nError: {e}"); input("Press Enter to continue...")

def main():
    """Main entry point"""
    try:
        system = CompactGeminiSystem()
        system.run_main_menu()
    except KeyboardInterrupt: print("\n\nSystem interrupted")
    except Exception as e: print(f"\nCritical error: {e}")

if __name__ == "__main__": main()
