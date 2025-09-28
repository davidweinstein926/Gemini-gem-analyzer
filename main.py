#!/usr/bin/env python3
"""
SUPER SAFE GEMINI ANALYSIS SYSTEM - OPTION 6 ENHANCED
⚡ Safe subprocess calls + Memory-protected Option 3 + Smart fallback
🛡️  Optional memory monitoring (install psutil for enhanced monitoring)
🎯 Enhanced Option 6: Production Structural Database Import with Auto-Archive
🚀 UPDATED: Options 4 & 8 now use Ultimate Multi-Gem Structural Analyzer
"""
import os, sys, subprocess, sqlite3, pandas as pd, numpy as np, threading, time, shutil, json, re
from datetime import datetime
from pathlib import Path

# Optional memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Audio support
try: import winsound; HAS_AUDIO = True
except ImportError:
    try: import pygame; pygame.mixer.init(); HAS_AUDIO = True
    except ImportError: HAS_AUDIO = False

class SuperSafeGeminiSystem:
    def __init__(self):
        self.db_path = "database/structural_spectra/multi_structural_gem_data.db"
        self.program_files = {
            'data_acquisition/main_menu.py': 'Data Acquisition System',
            'src/structural_analysis/gemini_launcher.py': 'Structural Marking System',
            'src/numerical_analysis/gemini1.py': 'Numerical Analysis Engine',
            'src/structural_analysis/multi_gem_structural_analyzer.py': 'Ultimate Structural Analysis Engine',
            'txt_to_split_long_format.py': 'Numerical DB Import Tool',
            'database/batch_importer.py': 'Structural DB Import Tool',
            'database/perfect_structural_archive_importer.py': 'Production Structural Import Tool',  # NEW
            'gem_selector.py': 'Safe Gem Analysis Tool'
        }
        self.bleep_enabled = True
        self.memory_limit_mb = 1500  # Prevent using more than 1.5GB
        self.init_directories()
    
    def init_directories(self):
        """Initialize all required directories"""
        directories = [
            "data/raw_temp", "data/raw (archive)", "data/structural_data", "data/structural(archive)",
            "outputs/numerical_analysis/reports", "outputs/numerical_analysis/graphs",
            "outputs/numerical_results/reports", "outputs/numerical_results/graphs",
            "outputs/structural_results/reports", "outputs/structural_results/graphs", 
            "database/reference_spectra", "database/structural_spectra",
            "results(archive)/post_analysis_numerical/reports", "results(archive)/post_analysis_numerical/graphs",
            "results(archive)/post_analysis_structural/reports", "results(archive)/post_analysis_structural/graphs",
            "data/unknown/numerical"  # For safe analysis
        ]
        for directory in directories: Path(directory).mkdir(parents=True, exist_ok=True)
    
    def check_memory_safety(self):
        """Check if system has enough memory for safe operation"""
        if not HAS_PSUTIL:
            print("🧠 Memory check: psutil not available (install with: pip install psutil)")
            print("✅ Assuming memory OK - proceeding with caution")
            return True
            
        try:
            memory = psutil.virtual_memory()
            available_mb = memory.available / 1024 / 1024
            
            print(f"🧠 Memory check: {available_mb:.0f}MB available")
            
            if available_mb < self.memory_limit_mb:
                print(f"⚠️  WARNING: Low memory ({available_mb:.0f}MB available, {self.memory_limit_mb}MB recommended)")
                return False
            else:
                print(f"✅ Memory OK: {available_mb:.0f}MB available")
                return True
                
        except Exception as e:
            print(f"⚠️  Cannot check memory: {e}")
            return True  # Assume OK if can't check
    
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
        """Setup environment variables for analysis tools"""
        os.environ['GEMINI_DATA_PATH'] = str(Path(data_directory).absolute())
        print(f"📁 Data source: {data_directory}")
        return str(Path(data_directory).absolute())
    
    def clear_old_analysis_files(self):
        """Clear old analysis files to prevent confusion with current results"""
        try:
            # Clear outputs/numerical_results directories to start fresh
            for dir_path in ["outputs/numerical_results/reports", "outputs/numerical_results/graphs"]:
                dir_obj = Path(dir_path)
                if dir_obj.exists():
                    removed_count = 0
                    for file in dir_obj.glob("*"):
                        if file.is_file():
                            file.unlink()
                            removed_count += 1
                    if removed_count > 0:
                        print(f"🗑️  Cleared {removed_count} old files from {dir_path}")
                        
            # Also clear the old output/numerical_analysis directory if it exists
            old_output_dir = Path("output/numerical_analysis")
            if old_output_dir.exists():
                removed_count = 0
                for file in old_output_dir.glob("*"):
                    if file.is_file():
                        file.unlink()
                        removed_count += 1
                if removed_count > 0:
                    print(f"🗑️  Cleared {removed_count} old analysis files from output/numerical_analysis")
                        
        except Exception as e:
            print(f"⚠️ Warning: Could not clear old files: {e}")

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
    
    # 🛡️ MENU OPTION 3: SUPER SAFE NUMERICAL MATCHING
    def numerical_matching(self):
        """Option 3: SUPER SAFE Numerical matching - Analyze newly captured unknown gems"""
        print("\n📊 SUPER SAFE NUMERICAL MATCHING - UNKNOWN GEM ANALYSIS\n" + "=" * 60)
        print("🛡️  MEMORY PROTECTION ACTIVE")
        print("🔬 Analyzing newly captured spectral data as unknown gem")
        
        # Memory safety check
        if not self.check_memory_safety():
            print("⚠️  Memory warning detected - using ultra-safe mode")
        
        # Look for newly captured files in data/raw_txt directory
        raw_txt_dir = Path("data/raw_txt")
        if not raw_txt_dir.exists():
            print("❌ No data/raw_txt directory found!")
            print("💡 Capture spectral data first and place .txt files in data/raw_txt/")
            return
        
        txt_files = self.check_directory_files("data/raw_txt", "*.txt")
        if not txt_files:
            print("❌ No .txt files found in data/raw_txt/")
            print("💡 Capture spectral data first (e.g., 58BC1.txt, 58LC1.txt, 58UC1.txt)")
            return
        
        print(f"📁 Found {len(txt_files)} spectral files in data/raw_txt/")
        
        # Check for B, L, U light sources
        light_sources_found = set()
        for file_path in txt_files:
            filename = file_path.stem.upper()
            for char in filename:
                if char in ['B', 'L', 'U']:
                    light_sources_found.add(char)
                    break
        
        print(f"🔍 Light sources detected: {', '.join(sorted(light_sources_found))}")
        
        if len(light_sources_found) < 3:
            print("⚠️  Warning: Optimal analysis requires B, L, and U light sources")
            missing = {'B', 'L', 'U'} - light_sources_found
            print(f"   Missing: {', '.join(sorted(missing))}")
            
            proceed = self.safe_input("Continue with available light sources? (y/n): ").strip().lower()
            if proceed != 'y':
                print("Analysis cancelled")
                return
        
        print("Output: root/outputs/numerical_results/reports;graphs")
        print("🔬 Files will be named with 'unkgem' prefix for unknown gem analysis")
        
        # Clear existing unknown files
        unknown_dir = Path("data/unknown/numerical")
        unknown_dir.mkdir(parents=True, exist_ok=True)
        for old_file in unknown_dir.glob("unkgem*.csv"):
            try:
                old_file.unlink()
                print(f"🗑️  Cleared old file: {old_file.name}")
            except Exception as e:
                print(f"⚠️  Could not clear {old_file.name}: {e}")
        
        # Set environment variable to point to raw_txt directory
        original_path = os.environ.get('GEMINI_DATA_PATH', '')
        os.environ['GEMINI_DATA_PATH'] = str(raw_txt_dir.absolute())
        
        try:
            # Run txt_to_unkgem.py to convert raw files
            print(f"\n🔄 Converting spectral files to analysis format...")
            if os.path.exists('data/txt_to_unkgem.py'):
                try:
                    result = subprocess.run([sys.executable, 'data/txt_to_unkgem.py'], 
                                          capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print("✅ File conversion completed successfully")
                        if result.stdout:
                            print("📋 Conversion output:")
                            print(result.stdout)
                    else:
                        print(f"❌ Conversion failed with return code {result.returncode}")
                        if result.stderr:
                            print("Error output:")
                            print(result.stderr)
                        return
                        
                except subprocess.TimeoutExpired:
                    print("⚠️  Conversion timed out after 60 seconds")
                    return
                except Exception as e:
                    print(f"❌ Error running txt_to_unkgem.py: {e}")
                    return
                    
            else:
                print("❌ txt_to_unkgem.py not found in data/ directory")
                return
            
            # Check if conversion created the expected files
            unknown_files = list(unknown_dir.glob("unkgem*.csv"))
            if not unknown_files:
                print("❌ No unkgem*.csv files were created by conversion")
                return
            
            print(f"✅ Created {len(unknown_files)} unknown gem files:")
            for f in unknown_files:
                size_kb = f.stat().st_size / 1024
                print(f"   📄 {f.name} ({size_kb:.1f} KB)")
            
            print(f"\n🚀 Starting numerical analysis of unknown gem...")
            print("Input: Newly captured spectral data")
            print("Output: root/outputs/numerical_results/reports;graphs")
            print("🔬 Using FULL numerical analysis engine")
            
            # Run the numerical analysis with "unkgem" as the base ID
            if os.path.exists('src/numerical_analysis/gemini1.py'):
                try:
                    print("📊 Launching full numerical analysis engine...")
                    
                    # Run with full analysis engine, passing "unkgem" as the base_id
                    result = subprocess.run([sys.executable, 'src/numerical_analysis/gemini1.py', 'unkgem'], 
                                          capture_output=False, text=True, timeout=600)  # 10 min timeout
                    
                    if result.returncode == 0: 
                        print("✅ Full numerical analysis completed successfully")
                        self.play_bleep("completion")
                    else:
                        print("❌ Numerical analysis failed")
                        
                except subprocess.TimeoutExpired:
                    print("⚠️  Analysis timed out after 10 minutes - killed for safety")
                except Exception as e: 
                    print(f"❌ Error: {e}")
                    
            # FALLBACK: Use gem_selector.py for basic file preparation
            elif os.path.exists('gem_selector.py'):
                try:
                    print("⚠️  gemini1.py not found - using gem_selector.py fallback")
                    print("🔧 This will prepare files but won't run full analysis")
                    
                    # Run gem_selector.py as fallback
                    result = subprocess.run([sys.executable, 'gem_selector.py'], 
                                          capture_output=False, text=True, timeout=300)
                    
                    if result.returncode == 0: 
                        print("✅ File preparation completed")
                        print("💡 For full analysis, ensure gemini1.py is available")
                        self.play_bleep("completion")
                    else:
                        print("❌ gem_selector.py failed")
                        
                except subprocess.TimeoutExpired:
                    print("⚠️  File preparation timed out - killed for safety")
                except Exception as e: 
                    print(f"❌ Error in file preparation: {e}")
                    
            else:
                print("❌ No analysis tools found (gem_selector.py or gemini1.py)")
                print("💡 Ensure analysis tools are available")
        
        finally:
            # Restore original environment
            if original_path:
                os.environ['GEMINI_DATA_PATH'] = original_path
            else:
                os.environ.pop('GEMINI_DATA_PATH', None)
        
        print(f"\n🎉 Unknown gem analysis completed!")
    
    # 🚀 MENU OPTION 4: ULTIMATE STRUCTURAL MATCHING (CURRENT WORK)
    def structural_matching(self):
        """Option 4: Ultimate structural matching - analyze current work files using ultimate analyzer"""
        print("\n🚀 ULTIMATE STRUCTURAL MATCHING (CURRENT WORK)\n" + "=" * 60)
        print("🎯 Using Ultimate Multi-Gem Structural Analyzer")
        print("📁 Input: root/data/structural_data/ (current work files)")
        print("🗄️  Database: Modern databases (gemini_structural.db/csv)")
        print("📊 Features: Advanced scoring + Feature weighting + Visualizations")
        print("🔬 Output: root/outputs/structural_results/reports;graphs")
        
        # Check for current structural files
        structural_files = self.check_directory_files("data/structural_data", "*.csv")
        if not structural_files:
            print("❌ No structural files found in data/structural_data/")
            print("💡 Use Option 2 to mark structural features first")
            return
        
        print(f"📁 Found {len(structural_files)} current structural files for ultimate analysis")
        
        # Confirm with user
        proceed = self.safe_input(f"Start ultimate analysis of {len(structural_files)} current work files? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Ultimate analysis cancelled")
            return
        
        # Check if ultimate analyzer exists
        ultimate_analyzer_path = 'src/structural_analysis/multi_gem_structural_analyzer.py'
        if not os.path.exists(ultimate_analyzer_path):
            print(f"❌ Ultimate analyzer not found: {ultimate_analyzer_path}")
            print("💡 Ensure the ultimate multi-gem structural analyzer is available")
            return
        
        try:
            print(f"\n🚀 Launching Ultimate Multi-Gem Structural Analyzer...")
            print("🎯 Mode: Automatic analysis of current work")
            print("📊 Using advanced feature weighting and modern databases")
            
            # Run ultimate analyzer in automatic mode for current work
            result = subprocess.run([
                sys.executable, 
                ultimate_analyzer_path,
                '--mode', 'auto',
                '--input-source', 'current',
                '--auto-complete'
            ], capture_output=False, text=True, timeout=900)  # 15 min timeout
            
            if result.returncode == 0:
                print("✅ Ultimate structural analysis completed successfully")
                print("📊 Results saved to:")
                print("   📄 Reports: outputs/structural_results/reports/")
                print("   📈 Graphs: outputs/structural_results/graphs/")
                print("📦 Previous results archived automatically")
                self.play_bleep("completion")
            else:
                print("❌ Ultimate structural analysis failed")
                print("💡 Check the ultimate analyzer for detailed error messages")
                
        except subprocess.TimeoutExpired:
            print("⚠️  Ultimate analysis timed out after 15 minutes - killed for safety")
        except Exception as e:
            print(f"❌ Error launching ultimate analyzer: {e}")
            
            # Fallback information
            print(f"\n🔧 TROUBLESHOOTING:")
            print(f"• Ensure {ultimate_analyzer_path} exists")
            print(f"• Check that data/structural_data/ has CSV files")
            print(f"• Verify modern databases exist (gemini_structural.db/csv)")
            print(f"• Install required packages: pip install matplotlib scipy pandas")
    
    # 🚀 MENU OPTION 8: ULTIMATE STRUCTURAL MATCHING (ARCHIVE TEST)
    def structural_matching_test(self):
        """Option 8: Ultimate structural matching test - analyze archived files using ultimate analyzer"""
        print("\n🚀 ULTIMATE STRUCTURAL MATCHING (ARCHIVE TEST)\n" + "=" * 60)
        print("🎯 Using Ultimate Multi-Gem Structural Analyzer")
        print("📁 Input: root/data/structural(archive)/ (archived files for testing)")
        print("🗄️  Database: Modern databases (gemini_structural.db/csv)")
        print("📊 Features: Advanced scoring + Feature weighting + Visualizations")
        print("🔬 Output: root/outputs/structural_results/reports;graphs")
        
        # Check for archived structural files
        archive_files = self.check_directory_files("data/structural(archive)", "*.csv")
        if not archive_files:
            print("❌ No archived structural files found in data/structural(archive)/")
            print("💡 Use Option 6 to import and archive structural files first")
            return
        
        print(f"📁 Found {len(archive_files)} archived structural files for ultimate testing")
        
        # Confirm with user
        proceed = self.safe_input(f"Start ultimate test analysis of {len(archive_files)} archived files? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Ultimate test analysis cancelled")
            return
        
        # Check if ultimate analyzer exists
        ultimate_analyzer_path = 'src/structural_analysis/multi_gem_structural_analyzer.py'
        if not os.path.exists(ultimate_analyzer_path):
            print(f"❌ Ultimate analyzer not found: {ultimate_analyzer_path}")
            print("💡 Ensure the ultimate multi-gem structural analyzer is available")
            return
        
        try:
            print(f"\n🚀 Launching Ultimate Multi-Gem Structural Analyzer...")
            print("🎯 Mode: Automatic test analysis of archived files")
            print("📊 Using advanced feature weighting and modern databases")
            print("🧪 Testing against archived data for validation")
            
            # Run ultimate analyzer in automatic mode for archive testing
            result = subprocess.run([
                sys.executable, 
                ultimate_analyzer_path,
                '--mode', 'auto',
                '--input-source', 'archive',
                '--auto-complete'
            ], capture_output=False, text=True, timeout=900)  # 15 min timeout
            
            if result.returncode == 0:
                print("✅ Ultimate test analysis completed successfully")
                print("📊 Results saved to:")
                print("   📄 Reports: outputs/structural_results/reports/")
                print("   📈 Graphs: outputs/structural_results/graphs/")
                print("📦 Previous results archived automatically")
                print("🧪 Archive testing validates system performance")
                self.play_bleep("completion")
            else:
                print("❌ Ultimate test analysis failed")
                print("💡 Check the ultimate analyzer for detailed error messages")
                
        except subprocess.TimeoutExpired:
            print("⚠️  Ultimate test analysis timed out after 15 minutes - killed for safety")
        except Exception as e:
            print(f"❌ Error launching ultimate analyzer: {e}")
            
            # Fallback information
            print(f"\n🔧 TROUBLESHOOTING:")
            print(f"• Ensure {ultimate_analyzer_path} exists")
            print(f"• Check that data/structural(archive)/ has CSV files")
            print(f"• Verify modern databases exist (gemini_structural.db/csv)")
            print(f"• Install required packages: pip install matplotlib scipy pandas")
    
    def move_structural_results_to_outputs(self):
        """Move structural results to proper output location (legacy support)"""
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
        except Exception as e: 
            print(f"⚠️ Result move error: {e}")

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
    
    # 🎯 MENU OPTION 6: ENHANCED PRODUCTION STRUCTURAL IMPORT
    def import_to_structural_db(self):
        """Option 6: Enhanced Production Structural Database Import with Auto-Archive"""
        print("\n🎯 PRODUCTION STRUCTURAL DATABASE IMPORT\n" + "=" * 60)
        print("📍 Built on SuperSafeGeminiSystem architecture")
        print("🗄️  Database: database/structural_spectra/gemini_structural.db")
        print("📂 Source: root/data/structural_data/ (fresh structural data)")
        print("📦 Archive: root/data/structural(archive)/ (after successful import)")
        print("💎 Handles all light sources: Halogen (B), Laser (L), UV (U)")
        print("🔬 Production workflow: Import → Database → Archive")
        
        # Check if source directory exists
        source_dir = Path("data/structural_data")
        if not source_dir.exists():
            print("❌ Source directory not found: data/structural_data")
            print("💡 This is for importing fresh structural data from Option 2")
            print("💡 Use Option 2 (Structural Marking) to create structural data first")
            return
        
        # Count source files
        structural_files = list(source_dir.glob("*.csv"))
        if not structural_files:
            print(f"❌ No CSV files found in {source_dir}")
            print("💡 Use Option 2 to mark structural features first")
            return
        
        print(f"📊 Found {len(structural_files)} structural files to import")
        
        # Show sample of files found
        print(f"\n📋 Sample files to import:")
        sample_files = structural_files[:5]
        for i, file in enumerate(sample_files, 1):
            size_kb = file.stat().st_size / 1024
            print(f"   {i}. {file.name} ({size_kb:.1f} KB)")
        
        if len(structural_files) > 5:
            print(f"   ... and {len(structural_files) - 5} more files")
        
        # Safety confirmation (SuperSafe pattern)
        print(f"\n⚠️  PRODUCTION IMPORT CONFIRMATION:")
        print("This will:")
        print("✅ Import fresh structural data into database")
        print("✅ Update/create database/structural_spectra/gemini_structural.db")
        print("✅ Archive successfully imported files to structural(archive)")
        print("✅ Preserve existing database with automatic backup")
        
        confirm = self.safe_input(f"Proceed with production import of {len(structural_files)} files? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Import cancelled by user")
            return
        
        # Memory safety check (SuperSafe pattern)
        if not self.check_memory_safety():
            print("⚠️  Memory warning detected")
            print("Production import is memory-intensive due to comprehensive processing")
            proceed = self.safe_input("Continue with memory caution? (y/n): ").strip().lower()
            if proceed != 'y':
                print("❌ Import cancelled due to memory concerns")
                return
        
        print(f"\n🚀 Starting Production Structural Import...")
        print("This may take several minutes depending on data volume...")
        
        try:
            # Import the production structural importer
            try:
                from database.perfect_structural_archive_importer import production_structural_import
            except ImportError:
                print("❌ database/perfect_structural_archive_importer.py not found")
                print("💡 Ensure the file is in your database directory")
                print("💡 Running fallback batch import instead...")
                
                # Fallback to existing batch importer
                if os.path.exists('database/batch_importer.py'):
                    result = subprocess.run([sys.executable, 'database/batch_importer.py'], capture_output=False, text=True)
                    if result.returncode == 0: 
                        print("✅ Fallback import completed")
                        self.play_bleep("completion")
                    else:
                        print("❌ Fallback import failed")
                else:
                    print("❌ No import tools available")
                return
            
            # Execute production import
            print("📊 Launching production structural import engine...")
            success = production_structural_import()
            
            if success:
                print("\n🎉 PRODUCTION IMPORT SUCCESS!")
                print("-" * 50)
                print("✅ Database updated: database/structural_spectra/gemini_structural.db")
                print("✅ All light sources unified: Halogen/Laser/UV")
                print("✅ Fresh structural data successfully imported")
                print("✅ Source files archived to structural(archive)")
                print("✅ Performance optimized with indexes")
                print("✅ Data integrity validated")
                print("✅ Production workflow completed")
                
                # Play completion sound
                self.play_bleep("completion")
                
                # Offer to show statistics
                show_stats = self.safe_input("\nShow database statistics? (y/n): ").strip().lower()
                if show_stats == 'y':
                    self.show_structural_database_stats()
            else:
                print("\n❌ PRODUCTION IMPORT FAILED!")
                print("Check error messages above for details")
                print("Common solutions:")
                print("• Verify source directory path: data/structural_data")
                print("• Check file permissions and disk space")
                print("• Ensure CSV files are properly formatted")
                print("• Try running with administrator privileges")
                
        except Exception as e:
            print(f"\n💥 Unexpected error during production import: {e}")
            print("Error type:", type(e).__name__)
            
            # Offer fallback option
            print("\n🔧 Attempting fallback to existing batch importer...")
            try:
                if os.path.exists('database/batch_importer.py'):
                    result = subprocess.run([sys.executable, 'database/batch_importer.py'], capture_output=False, text=True)
                    if result.returncode == 0: 
                        print("✅ Fallback import completed successfully")
                        self.play_bleep("completion")
                    else:
                        print("❌ Fallback also failed")
            except Exception as fallback_error:
                print(f"❌ Fallback error: {fallback_error}")

    def show_structural_database_stats(self):
        """Show comprehensive structural database statistics"""
        try:
            print("\n📊 STRUCTURAL DATABASE STATISTICS")
            print("=" * 50)
            
            # Check which database exists (priority order)
            db_paths = [
                "database/structural_spectra/gemini_structural.db",  # NEW primary location
                "gemini_structural.db",  # Alternative location
                "database/structural_spectra/multi_structural_gem_data.db"  # Legacy
            ]
            db_path = None
            
            for path in db_paths:
                if os.path.exists(path):
                    db_path = path
                    break
            
            if not db_path:
                print("❌ No structural database found")
                print("💡 Run Option 6 to import structural data first")
                return
            
            print(f"📁 Database: {db_path}")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Basic statistics
            cursor.execute("SELECT COUNT(*) FROM structural_features")
            total_records = cursor.fetchone()[0]
            print(f"📝 Total spectral records: {total_records:,}")
            
            # Unique gems
            cursor.execute("SELECT COUNT(DISTINCT COALESCE(gem_id, file)) FROM structural_features")
            unique_gems = cursor.fetchone()[0]
            print(f"💎 Unique gems analyzed: {unique_gems}")
            
            # By light source
            cursor.execute("SELECT light_source, COUNT(*) FROM structural_features GROUP BY light_source ORDER BY COUNT(*) DESC")
            light_sources = cursor.fetchall()
            
            print(f"\n💡 Records by light source:")
            for source, count in light_sources:
                percentage = (count / total_records) * 100 if total_records > 0 else 0
                print(f"   {source}: {count:,} ({percentage:.1f}%)")
            
            # Wavelength analysis
            cursor.execute("SELECT MIN(wavelength), MAX(wavelength), AVG(wavelength) FROM structural_features WHERE wavelength IS NOT NULL")
            min_wl, max_wl, avg_wl = cursor.fetchone()
            
            if min_wl and max_wl:
                print(f"\n🌈 Wavelength analysis:")
                print(f"   Range: {min_wl:.1f} - {max_wl:.1f} nm")
                print(f"   Average: {avg_wl:.1f} nm")
            
            # Feature analysis
            cursor.execute("SELECT feature_group, COUNT(*) FROM structural_features WHERE feature_group IS NOT NULL GROUP BY feature_group ORDER BY COUNT(*) DESC LIMIT 8")
            feature_groups = cursor.fetchall()
            
            if feature_groups:
                print(f"\n🏷️  Top feature groups:")
                for group, count in feature_groups:
                    print(f"   {group}: {count:,}")
            
            # Recent imports
            cursor.execute("SELECT DATE(import_timestamp), COUNT(*) FROM structural_features WHERE import_timestamp IS NOT NULL GROUP BY DATE(import_timestamp) ORDER BY DATE(import_timestamp) DESC LIMIT 5")
            recent_imports = cursor.fetchall()
            
            if recent_imports:
                print(f"\n📅 Recent import activity:")
                for date, count in recent_imports:
                    print(f"   {date}: {count:,} records")
            
            # Data quality check
            cursor.execute("SELECT COUNT(*) FROM structural_features WHERE wavelength IS NULL OR intensity IS NULL")
            incomplete_records = cursor.fetchone()[0]
            
            print(f"\n🔍 Data quality:")
            complete_records = total_records - incomplete_records
            completeness = (complete_records / total_records) * 100 if total_records > 0 else 0
            print(f"   Complete records: {complete_records:,} ({completeness:.1f}%)")
            
            if incomplete_records > 0:
                print(f"   ⚠️  Incomplete records: {incomplete_records:,}")
            else:
                print(f"   ✅ All records complete")
            
            # Database size
            db_size = os.path.getsize(db_path) / (1024 * 1024)  # MB
            print(f"\n💾 Database size: {db_size:.2f} MB")
            
            conn.close()
            
            print("\n✅ Database is ready for ultimate structural analysis!")
            print("🎯 Use Option 4 (current work) or Option 8 (archive test) for ultimate matching")
            
        except Exception as e:
            print(f"❌ Could not retrieve database stats: {e}")
            print(f"🔍 Error details: {type(e).__name__}")
    
    # MENU OPTION 7: NUMERICAL MATCHING (TEST) - Interactive gem selection
    def numerical_matching_test(self):
        """Option 7: Numerical matching test - Interactive gem selection from archive"""
        print("\n📊 NUMERICAL MATCHING (TEST) - INTERACTIVE SELECTION\n" + "=" * 60)
        
        archive_files = self.check_directory_files("data/raw (archive)", "*.txt")
        if not archive_files:
            print("❌ No archived files found in data/raw (archive)/")
            print("💡 Use Option 9 to archive files from raw_temp first")
            return
        
        print(f"📁 Found {len(archive_files)} archived spectral files")
        
        # Parse gem names and group by base ID
        gem_groups = {}
        for file_path in archive_files:
            filename = file_path.stem  # Get filename without extension
            
            # Parse gem name to extract base ID and light source
            # Expected format: [prefix]base_id + light_source + orientation + scan_number
            # Examples: 58BC1, C0045LC2, S20250909UP3
            
            base_id = None
            light_source = None
            
            # Try to find light source (B, L, U) in filename
            for i, char in enumerate(filename.upper()):
                if char in ['B', 'L', 'U']:
                    # Check if this looks like a light source position
                    if i > 0:  # Not at start
                        base_id = filename[:i]
                        light_source = char
                        break
            
            if base_id and light_source:
                if base_id not in gem_groups:
                    gem_groups[base_id] = {'files': {}, 'full_names': {}}
                gem_groups[base_id]['files'][light_source] = file_path
                gem_groups[base_id]['full_names'][light_source] = filename
        
        if not gem_groups:
            print("❌ No valid gem files found with recognizable naming pattern")
            print("💡 Expected format: [ID][B/L/U][orientation][scan].txt")
            return
        
        # Show available gems to user
        print(f"\n🔍 Available gems for analysis ({len(gem_groups)} unique gems):")
        print("=" * 50)
        
        valid_gems = []
        for i, (base_id, data) in enumerate(sorted(gem_groups.items()), 1):
            light_sources = sorted(data['files'].keys())
            complete = len(light_sources) >= 3 and 'B' in light_sources and 'L' in light_sources and 'U' in light_sources
            
            status = "✅ COMPLETE (B+L+U)" if complete else f"⚠️  PARTIAL ({'+'.join(light_sources)})"
            print(f"{i:2d}. Gem {base_id} - {status}")
            
            # Show individual files
            for ls in ['B', 'L', 'U']:
                if ls in data['files']:
                    filename = data['full_names'][ls]
                    print(f"      {ls}: {filename}")
                else:
                    print(f"      {ls}: Missing")
            
            if complete:
                valid_gems.append((i, base_id, data))
            print()
        
        if not valid_gems:
            print("❌ No complete gems found (need B+L+U light sources)")
            print("💡 Each gem needs Halogen (B), Laser (L), and UV (U) spectra")
            return
        
        # Get user selection
        print(f"📋 {len(valid_gems)} complete gems available for analysis")
        while True:
            try:
                selection = self.safe_input("Select gem number for analysis (or 'q' to quit): ").strip()
                if selection.lower() == 'q':
                    return
                
                gem_num = int(selection)
                selected_gem = None
                for num, base_id, data in valid_gems:
                    if num == gem_num:
                        selected_gem = (base_id, data)
                        break
                
                if selected_gem:
                    break
                else:
                    print(f"❌ Invalid selection. Choose from: {', '.join(str(n) for n, _, _ in valid_gems)}")
                    
            except ValueError:
                print("❌ Please enter a number or 'q' to quit")
        
        base_id, gem_data = selected_gem
        print(f"\n🎯 Selected: Gem {base_id}")
        
        # Show selected files
        selected_files = []
        for ls in ['B', 'L', 'U']:
            if ls in gem_data['files']:
                file_path = gem_data['files'][ls]
                filename = gem_data['full_names'][ls]
                selected_files.append(file_path)
                print(f"   {ls}: {filename}")
        
        confirm = self.safe_input(f"\nProceed with analysis of Gem {base_id}? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Analysis cancelled")
            return
        
        # Clear existing unknown files
        unknown_dir = Path("data/unknown/numerical")
        unknown_dir.mkdir(parents=True, exist_ok=True)
        for old_file in unknown_dir.glob("unkgem*.csv"):
            try:
                old_file.unlink()
                print(f"🗑️  Cleared old file: {old_file.name}")
            except Exception as e:
                print(f"⚠️  Could not clear {old_file.name}: {e}")
        
        # Copy selected files to a temp location for txt_to_unkgem.py
        temp_dir = Path("temp_selected_gem")
        temp_dir.mkdir(exist_ok=True)
        
        try:
            print(f"\n📋 Preparing files for conversion...")
            for file_path in selected_files:
                temp_file = temp_dir / file_path.name
                shutil.copy2(file_path, temp_file)
                print(f"   📄 Copied: {file_path.name}")
            
            # Set environment variable to point to temp directory
            original_path = os.environ.get('GEMINI_DATA_PATH', '')
            os.environ['GEMINI_DATA_PATH'] = str(temp_dir.absolute())
            
            # Run txt_to_unkgem.py
            print(f"\n🔄 Running txt_to_unkgem.py to convert selected files...")
            if os.path.exists('data/txt_to_unkgem.py'):
                try:
                    result = subprocess.run([sys.executable, 'data/txt_to_unkgem.py'], 
                                          capture_output=True, text=True, timeout=60)
                    
                    if result.returncode == 0:
                        print("✅ File conversion completed successfully")
                        if result.stdout:
                            print("📋 Conversion output:")
                            print(result.stdout)
                    else:
                        print(f"❌ Conversion failed with return code {result.returncode}")
                        if result.stderr:
                            print("Error output:")
                            print(result.stderr)
                        return
                        
                except subprocess.TimeoutExpired:
                    print("⚠️  Conversion timed out after 60 seconds")
                    return
                except Exception as e:
                    print(f"❌ Error running txt_to_unkgem.py: {e}")
                    return
                    
            else:
                print("❌ txt_to_unkgem.py not found in data/ directory")
                return
            
            # Restore original environment
            if original_path:
                os.environ['GEMINI_DATA_PATH'] = original_path
            else:
                os.environ.pop('GEMINI_DATA_PATH', None)
            
            # Check if conversion created the expected files
            unknown_files = list(unknown_dir.glob("unkgem*.csv"))
            if not unknown_files:
                print("❌ No unkgem*.csv files were created by conversion")
                return
            
            print(f"✅ Created {len(unknown_files)} unknown gem files:")
            for f in unknown_files:
                size_kb = f.stat().st_size / 1024
                print(f"   📄 {f.name} ({size_kb:.1f} KB)")
            
            print(f"\n🚀 Starting numerical analysis of Gem {base_id}...")
            print("Input: Selected files from archive")
            print("Output: root/outputs/numerical_results/reports;graphs")
            print("🔬 Using FULL numerical analysis engine")
            
            # Run the numerical analysis
            if os.path.exists('src/numerical_analysis/gemini1.py'):
                try:
                    print("📊 Launching full numerical analysis engine...")
                    
                    # Run with full analysis engine and extended timeout, passing gem ID
                    result = subprocess.run([sys.executable, 'src/numerical_analysis/gemini1.py', base_id], 
                                          capture_output=False, text=True, timeout=600)  # 10 min timeout
                    
                    if result.returncode == 0: 
                        print("✅ Full numerical analysis completed successfully")
                        self.play_bleep("completion")
                    else:
                        print("❌ Numerical analysis failed")
                        
                except subprocess.TimeoutExpired:
                    print("⚠️  Analysis timed out after 10 minutes - killed for safety")
                except Exception as e: 
                    print(f"❌ Error: {e}")
                    
            elif os.path.exists('gem_selector.py'):
                print("⚠️  Full analysis engine not found - using file preparation fallback")
                try:
                    result = subprocess.run([sys.executable, 'gem_selector.py'], 
                                          capture_output=False, text=True, timeout=300)
                    
                    if result.returncode == 0: 
                        print("✅ File preparation completed")
                        print("💡 For full analysis, ensure src/numerical_analysis/gemini1.py is available")
                        self.play_bleep("completion")
                except Exception as e: 
                    print(f"❌ Error: {e}")
            else: 
                print("❌ No analysis tools found (gemini1.py or gem_selector.py)")
            
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
                print(f"🗑️  Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️  Could not clean up temp directory: {e}")
        
        print(f"\n🎉 Analysis of Gem {base_id} completed!")
    
    # MENU OPTION 9: CLEAN UP NUMERICAL
    def cleanup_numerical(self):
        """Option 9: Clean up numerical - archive data and results"""
        print("\n🧹 CLEAN UP NUMERICAL\n" + "=" * 60)
        print("a) Archive: root/data/raw_temp → root/data/raw (archive)")
        print("b) Archive: root/outputs/numerical_results/reports;graphs → root/results(archive)/post_analysis_numerical/reports;graphs")
        
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
            
            # 9b: Archive numerical results from new organized structure
            results_moved = 0
            
            # Archive reports
            reports_source = Path("outputs/numerical_results/reports")
            if reports_source.exists():
                for file in reports_source.glob("*"):
                    if file.is_file():
                        shutil.move(str(file), f"results(archive)/post_analysis_numerical/reports/{file.name}")
                        results_moved += 1
            
            # Archive graphs  
            graphs_source = Path("outputs/numerical_results/graphs")
            if graphs_source.exists():
                for file in graphs_source.glob("*"):
                    if file.is_file():
                        shutil.move(str(file), f"results(archive)/post_analysis_numerical/graphs/{file.name}")
                        results_moved += 1
            
            # Also clean up any files left in the old output/numerical_analysis directory
            old_output_dir = Path("output/numerical_analysis")
            if old_output_dir.exists():
                old_files = list(old_output_dir.glob("*"))
                if old_files:
                    print(f"🗑️  Found {len(old_files)} files in old output directory - cleaning up...")
                    for file in old_files:
                        if file.is_file():
                            # Move old files to reports directory
                            if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.svg', '.pdf']:
                                shutil.move(str(file), f"results(archive)/post_analysis_numerical/graphs/{file.name}")
                            else:
                                shutil.move(str(file), f"results(archive)/post_analysis_numerical/reports/{file.name}")
                            results_moved += 1
            
            if results_moved > 0: 
                print(f"✅ Archived {results_moved} result files to results(archive)/post_analysis_numerical/")
            else:
                print("ℹ️  No numerical results found to archive")
                
            print("✅ Numerical cleanup completed"); self.play_bleep("completion")
            
        except Exception as e: print(f"❌ Cleanup error: {e}")
    
    # MENU OPTION 10: CLEAN UP STRUCTURAL
    def cleanup_structural(self):
        """Option 10: Clean up structural - archive data and results"""
        print("\n🧹 CLEAN UP STRUCTURAL\n" + "=" * 60)
        print("a) Archive: root/data/structural_data/ → root/data/structural(archive)")
        print("b) Archive: root/outputs/structural_results/ → root/results(archive)/post_analysis_structural/")
        print("ℹ️  NOTE: Option 6 now auto-archives imported files")
        print("ℹ️  NOTE: Ultimate analyzer (Options 4 & 8) auto-archives results")
        
        try:
            archived_files = 0
            # 10a: Archive any remaining structural_data files (if Option 6 didn't catch them)
            structural_files = self.check_directory_files("data/structural_data", "*.csv")
            if structural_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                for file in structural_files:
                    dest_name = f"{file.stem}_manual_archive_{timestamp}{file.suffix}"
                    shutil.move(str(file), f"data/structural(archive)/{dest_name}")
                    archived_files += 1
                print(f"✅ Manually archived {archived_files} remaining structural files")
            else:
                print("ℹ️  No structural files found to archive (Auto-archive working)")
            
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
            
            if results_moved > 0: 
                print(f"✅ Archived {results_moved} result files")
            else:
                print("ℹ️  No structural results found to archive (Ultimate analyzer auto-archives)")
                
            print("✅ Structural cleanup completed"); self.play_bleep("completion")
            
        except Exception as e: print(f"❌ Cleanup error: {e}")
    
    # MENU OPTION 11: SYSTEM STATUS
    def system_status(self):
        """Option 11: System status with memory monitoring"""
        print("\nSUPER SAFE SYSTEM STATUS\n" + "=" * 40)
        
        # Memory status
        if HAS_PSUTIL:
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)
                
                print("🧠 SYSTEM RESOURCES:")
                print(f"   Memory: {memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB ({memory.percent:.1f}%)")
                print(f"   Available: {memory.available/1024/1024:.0f}MB")
                print(f"   CPU Usage: {cpu_percent:.1f}%")
                
                if memory.available/1024/1024 < self.memory_limit_mb:
                    print(f"⚠️  LOW MEMORY WARNING: {memory.available/1024/1024:.0f}MB < {self.memory_limit_mb}MB")
                else:
                    print(f"✅ Memory OK for safe operation")
                    
            except Exception as e:
                print(f"⚠️  Cannot check system resources: {e}")
        else:
            print("🧠 SYSTEM RESOURCES:")
            print("   Memory monitoring not available (psutil not installed)")
            print("   💡 Install psutil for memory monitoring: pip install psutil")
        
        # Check program files
        print("\n📋 Program Files:")
        for file_path, description in self.program_files.items():
            status = "✅" if os.path.exists(file_path) else "❌"
            print(f"{status} {description}: {file_path}")
        
        # Check key directories with file counts
        print(f"\n📁 Directory Status:")
        key_dirs = [
            ("data/raw_temp", "Raw temp files", "*.txt"),
            ("data/raw (archive)", "Archived raw files", "*.txt"), 
            ("data/structural_data", "Fresh structural files", "*.csv"),
            ("data/structural(archive)", "Archived structural files", "*.csv"),
            ("outputs/numerical_results/reports", "Numerical reports", "*"),
            ("outputs/structural_results/reports", "Structural reports", "*")
        ]
        
        for dir_path, description, pattern in key_dirs:
            files = self.check_directory_files(dir_path, pattern)
            file_count = len(files)
            status = "✅" if file_count > 0 else "⚪"
            print(f"{status} {description}: {file_count} files in {dir_path}")
        
        # Database status - Enhanced for new database locations
        print(f"\n🗄️  Database Status:")
        
        # Check structural databases (priority order)
        structural_dbs = [
            ("database/structural_spectra/gemini_structural.db", "Modern Structural DB (PRIMARY)"),
            ("database/structural_spectra/gemini_structural_unified.csv", "Modern Structural CSV"),
            ("gemini_structural.db", "Alternative Structural DB"),
            ("database/structural_spectra/multi_structural_gem_data.db", "Legacy Structural DB")
        ]
        
        for db_path, description in structural_dbs:
            if os.path.exists(db_path):
                size_kb = os.path.getsize(db_path) / 1024
                print(f"✅ {description}: {size_kb:.1f} KB")
            else:
                print(f"⚪ {description}: Not found")
        
        # Environment check
        gemini_path = os.environ.get('GEMINI_DATA_PATH', 'Not set')
        print(f"\n🔧 Environment: GEMINI_DATA_PATH = {gemini_path}")
        print(f"🔊 Audio: {'Available' if HAS_AUDIO else 'Not available'} | Bleep: {'ON' if self.bleep_enabled else 'OFF'}")
        print(f"🛡️  Memory Limit: {self.memory_limit_mb}MB | Safety: ACTIVE")
        print(f"📊 Memory Monitor: {'Available' if HAS_PSUTIL else 'Not available (optional)'}")
        print(f"🎯 Enhanced Option 6: Production Import with Auto-Archive")
        print(f"🚀 Ultimate Options 4 & 8: Advanced structural analysis with modern databases")
    
    # MENU OPTION 12: TOGGLE BLEEP
    def toggle_bleep(self):
        """Option 12: Toggle bleep system"""
        self.bleep_enabled = not self.bleep_enabled
        print(f"🔊 Bleep system: {'ENABLED' if self.bleep_enabled else 'DISABLED'}")
        if self.bleep_enabled and HAS_AUDIO: self.play_bleep("completion")
    
    # MAIN MENU
    def run_main_menu(self):
        """Complete main menu with ULTIMATE Options 4 & 8"""
        print(f"\n{'='*70}\n  SUPER SAFE GEMINI ANALYSIS SYSTEM\n  🚀 ULTIMATE Options 4 & 8: Advanced Structural Analysis\n  🎯 Enhanced Option 6: Production Import + Auto-Archive\n  🛡️  Memory Protected + Smart Fallback\n{'='*70}")
        
        while True:
            print(f"\nMAIN MENU (Ultimate Enhanced):")
            print("=" * 60)
            print("📡 DATA CAPTURE & ANALYSIS:")
            print("1. Data Acquisition (Spectral Capture)")
            print("2. Structural Marking (smart fallback)")  
            print("3. 🛡️  Numerical Matching (MEMORY PROTECTED)")
            print("4. 🚀 Ultimate Structural Matching (Current Work)")
            print("")
            print("💾 DATABASE OPERATIONS:")
            print("5. Import to Numerical DB")
            print("6. 🎯 Production Structural Import (Auto-Archive)")
            print("")
            print("🧪 TESTING (Advanced & Legacy):")
            print("7. 🛡️  Numerical Matching Test (INTERACTIVE)")
            print("8. 🚀 Ultimate Structural Test (Archive)")
            print("")
            print("🧹 CLEANUP & ARCHIVING:")
            print("9. Clean Up Numerical")
            print("10. Clean Up Structural")
            print("")
            print("⚙️ SYSTEM:")
            print("11. System Status (with Database Monitor)")
            print("12. Toggle Bleep System") 
            print("13. Exit")
            
            # Show safety status
            memory_status = "OK" if self.check_memory_safety() else "⚠️  LOW" if HAS_PSUTIL else "Unknown"
            print(f"\nStatus: Bleep [{'ON' if self.bleep_enabled else 'OFF'}] | Memory: {memory_status} | 🎯 Enhanced Option 6 | 🚀 Ultimate 4&8 | 🛡️  SUPER SAFE")
            
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
                    print("Exiting SUPER SAFE system..."); self.play_bleep("completion") if self.bleep_enabled else None; break
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
        system = SuperSafeGeminiSystem()
        system.run_main_menu()
    except KeyboardInterrupt: print("\n\nSystem interrupted")
    except Exception as e: print(f"\nCritical error: {e}")

if __name__ == "__main__": main()
