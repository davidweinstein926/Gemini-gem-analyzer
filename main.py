#!/usr/bin/env python3
"""
COMPACT GEMINI ANALYSIS SYSTEM - FINAL COMPLETE FIX
Auto-feeds validated input to gemini1.py via stdin

Fixed Issues:
- gem_selector kick-in when input validation fails
- Proper input handling for different environments (jupyter, terminal, etc.)
- Reduced file size while preserving essential functionality
- FINAL FIX: Auto-feeds input to gemini1.py to bypass InputSubmission issue
"""

import os
import sys
import subprocess
import sqlite3
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime

# Audio support
try:
    import winsound
    HAS_AUDIO = True
except ImportError:
    try:
        import pygame
        pygame.mixer.init()
        HAS_AUDIO = True
    except ImportError:
        HAS_AUDIO = False

class CompactGeminiSystem:
    def __init__(self):
        self.db_path = "multi_structural_gem_data.db"
        self.structural_data_dir = r"c:\users\david\gemini sp10 structural data"
        
        # Core files
        self.program_files = {
            'src/structural_analysis/main.py': 'Structural Analysis Hub',
            'src/numerical_analysis/gemini1.py': 'Numerical Analysis Engine',
            'gem_selector.py': 'Gem Selector Tool',
            'fast_gem_analysis.py': 'Fast Analysis Tool'
        }
        
        # System settings
        self.bleep_enabled = True
        self.auto_import_enabled = True
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize database system"""
        if not os.path.exists(self.db_path):
            self.create_database_schema()
        os.makedirs(self.structural_data_dir, exist_ok=True)
    
    def create_database_schema(self):
        """Create database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structural_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature TEXT NOT NULL,
                    file TEXT NOT NULL,
                    light_source TEXT NOT NULL,
                    wavelength REAL NOT NULL,
                    intensity REAL NOT NULL,
                    point_type TEXT NOT NULL,
                    feature_group TEXT NOT NULL,
                    processing TEXT,
                    baseline_used REAL,
                    norm_factor REAL,
                    snr REAL,
                    symmetry_ratio REAL,
                    skew_description TEXT,
                    width_nm REAL,
                    height REAL,
                    normalization_scheme TEXT,
                    reference_wavelength REAL,
                    intensity_range_min REAL,
                    intensity_range_max REAL,
                    timestamp TEXT DEFAULT (datetime('now')),
                    file_source TEXT,
                    data_type TEXT,
                    UNIQUE(file, feature, wavelength, point_type)
                )
            """)
            
            conn.commit()
            conn.close()
            print("Database schema created successfully")
            
        except Exception as e:
            print(f"Error creating database schema: {e}")
    
    def play_bleep(self, feature_type="standard"):
        """Play audio bleep"""
        if not self.bleep_enabled or not HAS_AUDIO:
            return
        
        try:
            freq_map = {
                "peak": 1000,
                "valley": 600, 
                "plateau": 800,
                "significant": 1200,
                "completion": 400
            }
            
            freq = freq_map.get(feature_type, 800)
            duration = 200
            
            if 'winsound' in sys.modules:
                winsound.Beep(freq, duration)
            elif 'pygame' in sys.modules:
                sample_rate = 22050
                frames = int(duration * sample_rate / 1000)
                arr = np.zeros(frames)
                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * freq * i / sample_rate)
                arr = (arr * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(duration / 1000)
                
        except Exception as e:
            print(f"Audio bleep error: {e}")
    
    def safe_input(self, prompt):
        """Safe input handling - FIXES GEM_SELECTOR ISSUE"""
        try:
            user_input = input(prompt)
            
            # Handle wrapped input (jupyter/interactive environments)
            if hasattr(user_input, 'data'):
                user_input = user_input.data
            
            # Convert to string and clean
            user_input = str(user_input).strip()
            
            # Remove common wrapper patterns
            if user_input.startswith("InputSubmission(data='") and user_input.endswith("')"):
                user_input = user_input[22:-2]  # Remove wrapper
            
            # Remove newlines and extra quotes
            user_input = user_input.replace('\\n', '').replace('\n', '').strip("'\"")
            
            return user_input
            
        except Exception as e:
            print(f"Input error: {e}")
            return ""
    
    def validate_gem_format(self, gem_name):
        """Validate gem name format with fallback to gem_selector"""
        # Clean the input first
        gem_name = gem_name.strip()
        
        print(f"DEBUG: Raw input received: '{gem_name}'")
        print(f"DEBUG: Input type: {type(gem_name)}")
        
        # Handle wrapped inputs from jupyter/interactive environments
        if "InputSubmission" in str(gem_name):
            print("DEBUG: Detected wrapped input, extracting...")
            # Extract the actual gem name from wrapped input
            import re
            match = re.search(r"data='([^']*)", str(gem_name))
            if match:
                gem_name = match.group(1).strip()
                print(f"DEBUG: Extracted gem name: '{gem_name}'")
            else:
                print("DEBUG: Could not extract gem name from wrapped input")
                return False
        
        # Remove any trailing newlines or quotes
        gem_name = gem_name.replace('\\n', '').replace('\n', '').strip("'\"")
        
        print(f"DEBUG: Final cleaned gem name: '{gem_name}'")
        
        # Simple validation patterns
        valid_patterns = [
            r'^\d+[A-Z]+\d*$',        # 58BC1, 45LC2, etc.
            r'^[A-Z]\d+[A-Z]+\d*$',   # C0045LC2, S20250909UP3, etc.
            r'^\d+$',                 # Simple numbers like 58
            r'^[A-Z]+\d+$'            # BC58, LC45, etc.
        ]
        
        import re
        for pattern in valid_patterns:
            if re.match(pattern, gem_name):
                print(f"DEBUG: Gem name '{gem_name}' matches pattern {pattern}")
                return True
        
        print(f"DEBUG: Gem name '{gem_name}' does not match any valid pattern")
        return False
    
    def launch_gem_selector(self):
        """Launch gem_selector.py when validation fails"""
        print("\n" + "="*60)
        print("🎯 INPUT VALIDATION FAILED - LAUNCHING GEM SELECTOR")
        print("="*60)
        print("The gem name format was invalid.")
        print("Launching gem_selector.py to bypass this issue...")
        print()
        
        try:
            if os.path.exists('gem_selector.py'):
                print("Running gem_selector.py...")
                result = subprocess.run([sys.executable, 'gem_selector.py'], 
                                      capture_output=False, text=True)
                
                if result.returncode == 0:
                    print("\n✅ gem_selector.py completed successfully!")
                    print("Files should be ready for analysis now.")
                    
                    # Check if unkgem files were created
                    unkgem_files = ['unkgemB.csv', 'unkgemL.csv', 'unkgemU.csv']
                    created_files = [f for f in unkgem_files if os.path.exists(f) or 
                                   os.path.exists(f'data/unknown/{f}') or 
                                   os.path.exists(f'data/unknown/numerical/{f}')]
                    
                    if created_files:
                        print(f"✅ Found converted files: {created_files}")
                        return True
                    else:
                        print("⚠️ gem_selector completed but no unkgem files found")
                        return False
                else:
                    print(f"❌ gem_selector.py failed with exit code: {result.returncode}")
                    return False
            else:
                print("❌ gem_selector.py not found!")
                print("Please ensure gem_selector.py is in the current directory")
                return False
                
        except Exception as e:
            print(f"❌ Error running gem_selector.py: {e}")
            return False
    
    def get_gem_input_with_selector_fallback(self):
        """Get gem input with automatic gem_selector fallback"""
        while True:
            try:
                print("\nEnter gem names (comma-separated) or 'q' to quit:")
                print("Examples: 58BC1, C0045LC2, S20250909UP3")
                
                gem_input = self.safe_input("Gem names: ")
                
                if gem_input.lower() == 'q':
                    return None
                
                if not gem_input:
                    print("Empty input received. Please try again.")
                    continue
                
                # Split and validate gem names
                gem_names = [name.strip() for name in gem_input.split(',') if name.strip()]
                
                if not gem_names:
                    print("No valid gem names found.")
                    continue
                
                # Validate each gem name
                all_valid = True
                for gem_name in gem_names:
                    if not self.validate_gem_format(gem_name):
                        print(f"Invalid gem name format: '{gem_name}'")
                        print("Expected format: [prefix]number + light + orientation + scan")
                        print("Examples: 58BC1, C0045LC2, S20250909UP3")
                        all_valid = False
                        break
                
                if all_valid:
                    return gem_names
                
                # If validation failed, offer gem_selector
                print(f"\n🎯 VALIDATION FAILED - OFFERING GEM_SELECTOR BYPASS")
                use_selector = self.safe_input("Use gem_selector.py to bypass validation? (y/n): ")
                
                if use_selector.lower() == 'y':
                    if self.launch_gem_selector():
                        # Return a special indicator that gem_selector was used
                        return ['GEM_SELECTOR_USED']
                    else:
                        print("gem_selector failed. Please try manual input again.")
                        continue
                else:
                    print("Please try entering gem names again with correct format.")
                    continue
                    
            except KeyboardInterrupt:
                print("\nOperation cancelled by user")
                return None
            except Exception as e:
                print(f"Error in gem input: {e}")
                continue
    
    def run_numerical_analysis_with_auto_input(self, gem_names):
        """Launch numerical analysis with automatic input feeding"""
        print("\n🚀 LAUNCHING NUMERICAL ANALYSIS (AUTO-INPUT)")
        print("   Automatically feeding validated input to gemini1.py")
        print("=" * 60)
        
        numerical_path = 'src/numerical_analysis/gemini1.py'
        
        if not os.path.exists(numerical_path):
            print(f"Numerical analysis file not found: {numerical_path}")
            return
        
        try:
            # Prepare the input that gemini1.py expects
            gem_input_string = ','.join(gem_names) + '\n'
            confirmation_input = 'y\n'  # Auto-confirm the analysis
            
            # Combine all inputs
            full_input = gem_input_string + confirmation_input
            
            print(f"Auto-feeding input to gemini1.py:")
            print(f"  1. Gem names: {gem_input_string.strip()}")
            print(f"  2. Confirmation: y")
            
            # Launch gemini1.py with automatic input feeding
            process = subprocess.Popen(
                [sys.executable, numerical_path],
                stdin=subprocess.PIPE,
                stdout=None,  # Let output go to console
                stderr=None,  # Let errors go to console
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Feed all the input and close stdin
            process.stdin.write(full_input)
            process.stdin.close()
            
            # Wait for completion
            return_code = process.wait()
            
            if return_code == 0:
                print("\n✅ Numerical analysis completed successfully!")
                self.play_bleep("completion")
            else:
                print(f"\n⚠️ Numerical analysis exited with code: {return_code}")
                
        except Exception as e:
            print(f"❌ Error launching numerical analysis: {e}")
    
    def run_numerical_analysis_final_fix(self):
        """Launch numerical analysis with final complete fix"""
        print("\nNUMERICAL ANALYSIS (FINAL COMPLETE FIX)")
        print("=" * 50)
        
        # Get gem names using our fixed input handling FIRST
        gem_names = self.get_gem_input_with_selector_fallback()
        
        if gem_names is None:
            print("No gem names provided. Returning to menu.")
            return
        
        if gem_names == ['GEM_SELECTOR_USED']:
            print("✅ gem_selector was used. Files should be ready.")
            print("Launching numerical analysis with default input handling...")
            # For gem_selector case, just run normally since files are ready
            self.run_numerical_analysis()
        else:
            print(f"✅ Using pre-validated gems: {gem_names}")
            
            # Launch with automatic input feeding
            self.run_numerical_analysis_with_auto_input(gem_names)
    
    def run_numerical_analysis(self):
        """Launch numerical analysis (standard)"""
        print("\nLAUNCHING NUMERICAL ANALYSIS")
        print("=" * 50)
        
        numerical_path = 'src/numerical_analysis/gemini1.py'
        
        if os.path.exists(numerical_path):
            try:
                result = subprocess.run([sys.executable, numerical_path], 
                                      capture_output=False, text=True)
                
                if result.returncode == 0:
                    print("Numerical analysis completed successfully")
                    self.play_bleep("completion")
                else:
                    print(f"Numerical analysis exited with code: {result.returncode}")
                    
            except Exception as e:
                print(f"Error launching numerical analysis: {e}")
        else:
            print(f"Numerical analysis file not found: {numerical_path}")
    
    def run_structural_analysis(self):
        """Launch structural analysis"""
        print("\nLAUNCHING STRUCTURAL ANALYSIS")
        print("=" * 50)
        
        launcher_path = 'src/structural_analysis/gemini_launcher.py'
        
        if os.path.exists(launcher_path):
            try:
                result = subprocess.run([sys.executable, launcher_path], 
                                      capture_output=False, text=True)
                
                if result.returncode == 0:
                    print("Structural analysis completed successfully")
                    self.play_bleep("completion")
                else:
                    print(f"Structural analysis exited with code: {result.returncode}")
                    
            except Exception as e:
                print(f"Error launching structural analysis: {e}")
        else:
            print(f"Structural analysis file not found: {launcher_path}")
    
    def run_fast_analysis(self):
        """Launch fast analysis"""
        print("\nLAUNCHING FAST ANALYSIS")
        print("=" * 50)
        
        fast_path = 'fast_gem_analysis.py'
        
        if os.path.exists(fast_path):
            try:
                result = subprocess.run([sys.executable, fast_path], 
                                      capture_output=False, text=True)
                
                if result.returncode == 0:
                    print("Fast analysis completed successfully")
                    self.play_bleep("completion")
                else:
                    print(f"Fast analysis exited with code: {result.returncode}")
                    
            except Exception as e:
                print(f"Error launching fast analysis: {e}")
        else:
            print(f"Fast analysis file not found: {fast_path}")
    
    def system_status(self):
        """Show system status"""
        print("\nSYSTEM STATUS")
        print("=" * 30)
        
        # Check database
        if os.path.exists(self.db_path):
            size = os.path.getsize(self.db_path) / 1024
            print(f"✓ Database: {self.db_path} ({size:.1f} KB)")
        else:
            print(f"✗ Database missing: {self.db_path}")
        
        # Check program files
        print("\nProgram Files:")
        for file_path, description in self.program_files.items():
            status = "✓" if os.path.exists(file_path) else "✗"
            print(f"{status} {description}: {file_path}")
        
        # Check for unkgem files
        print("\nUnkgem Files:")
        unkgem_locations = [".", "data/unknown", "data/unknown/numerical"]
        for location in unkgem_locations:
            if os.path.exists(location):
                for light in ['B', 'L', 'U']:
                    unkgem_file = os.path.join(location, f"unkgem{light}.csv")
                    status = "✓" if os.path.exists(unkgem_file) else "✗"
                    print(f"{status} {unkgem_file}")
        
        # Audio status
        audio_status = "Available" if HAS_AUDIO else "Not available"
        bleep_status = "ON" if self.bleep_enabled else "OFF"
        print(f"\nAudio: {audio_status} | Bleep: {bleep_status}")
    
    def toggle_bleep(self):
        """Toggle bleep system"""
        self.bleep_enabled = not self.bleep_enabled
        status = "ENABLED" if self.bleep_enabled else "DISABLED"
        print(f"Bleep system: {status}")
        if self.bleep_enabled and HAS_AUDIO:
            self.play_bleep("completion")
    
    def run_main_menu(self):
        """Main menu interface"""
        print("\n" + "=" * 60)
        print("  COMPACT GEMINI ANALYSIS SYSTEM")
        print("  Final fix - auto-feeds input to gemini1.py")
        print("=" * 60)
        
        while True:
            print(f"\nMAIN MENU:")
            print("=" * 30)
            print("1. Run Structural Analysis")
            print("2. Run Numerical Analysis (FINAL FIX)") 
            print("3. Fast Gem Analysis")
            print("4. Test Gem Input (with selector fallback)")
            print("5. Launch Gem Selector Directly")
            print("6. System Status")
            print("7. Toggle Bleep System")
            print("0. Exit")
            
            bleep_status = "ON" if self.bleep_enabled else "OFF"
            print(f"\nStatus: Bleep [{bleep_status}]")
            
            try:
                choice = self.safe_input("\nSelect option (0-7): ")
                
                if choice == '0':
                    print("Exiting system...")
                    if self.bleep_enabled:
                        self.play_bleep("completion")
                    break
                    
                elif choice == '1':
                    self.run_structural_analysis()
                    
                elif choice == '2':
                    # Use the final fix that auto-feeds input to gemini1.py
                    self.run_numerical_analysis_final_fix()
                    
                elif choice == '3':
                    self.run_fast_analysis()
                    
                elif choice == '4':
                    # Test the fixed gem input system
                    print("\nTesting gem input with selector fallback...")
                    gem_names = self.get_gem_input_with_selector_fallback()
                    if gem_names:
                        if gem_names == ['GEM_SELECTOR_USED']:
                            print("✅ gem_selector was used successfully!")
                        else:
                            print(f"✅ Valid gem names received: {gem_names}")
                    else:
                        print("No gem names received")
                    
                elif choice == '5':
                    # Direct gem_selector launch
                    self.launch_gem_selector()
                    
                elif choice == '6':
                    self.system_status()
                    
                elif choice == '7':
                    self.toggle_bleep()
                    
                else:
                    print("Invalid choice. Please select 0-7.")
                    
                if choice != '0':
                    input("\nPress Enter to continue...")
                    
            except KeyboardInterrupt:
                print("\n\nSystem interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nError in menu system: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    try:
        system = CompactGeminiSystem()
        
        # Command line mode support
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == '--gem-selector':
                system.launch_gem_selector()
            elif command == '--status':
                system.system_status()
            elif command == '--test-input':
                gem_names = system.get_gem_input_with_selector_fallback()
                if gem_names:
                    print(f"Success: {gem_names}")
            elif command == '--help':
                print("Compact Gemini Analysis System")
                print("  --gem-selector : Launch gem selector directly")
                print("  --status      : Show system status")
                print("  --test-input  : Test gem input with fallback")
                print("  --help        : Show this help")
                print("  (no args)     : Interactive menu")
            else:
                print(f"Unknown command: {command}")
                print("Use --help for available commands")
        else:
            # Interactive mode
            system.run_main_menu()
            
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user")
    except Exception as e:
        print(f"\nCritical system error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()