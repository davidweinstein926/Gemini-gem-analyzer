#!/usr/bin/env python3
"""
run_gemini1_fixed.py - Run gemini1.py without Unicode issues
Your files are already converted - this just runs the analysis safely

LOCATION: Save as run_gemini1_fixed.py (in main directory)
"""

import os
import sys
import subprocess

def run_gemini1_safely():
    """Run gemini1.py with proper encoding handling"""
    
    print("🧮 RUNNING NUMERICAL ANALYSIS (Unicode-Safe)")
    print("=" * 50)
    
    # Check that converted files exist
    required_files = [
        'data/unknown/unkgemB.csv',
        'data/unknown/unkgemL.csv', 
        'data/unknown/unkgemU.csv'
    ]
    
    print("📋 Checking converted files:")
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} ({size:,} bytes)")
        else:
            print(f"   ❌ {file_path} (missing)")
            missing.append(file_path)
    
    if missing:
        print(f"\n❌ Missing files: {missing}")
        print("Run the emergency fix again to create them")
        return False
    
    # Find gemini1.py
    possible_paths = [
        'src/numerical_analysis/gemini1.py',
        '../src/numerical_analysis/gemini1.py',
        'numerical_analysis/gemini1.py',
        'gemini1.py'
    ]
    
    gemini_path = None
    for path in possible_paths:
        if os.path.exists(path):
            gemini_path = path
            print(f"✅ Found gemini1.py: {path}")
            break
    
    if not gemini_path:
        print("❌ gemini1.py not found in expected locations")
        print("Searched:", possible_paths)
        return False
    
    # Run gemini1.py with proper encoding
    print(f"\n🚀 Launching numerical analysis...")
    print("   (This may take 30-60 seconds)")
    
    try:
        # Method 1: Try with UTF-8 encoding
        result = subprocess.run(
            [sys.executable, gemini_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace problematic characters
            timeout=180  # 3 minute timeout
        )
        
        print(f"\n📊 ANALYSIS COMPLETED!")
        print("=" * 30)
        
        if result.returncode == 0:
            print("✅ Analysis successful!")
        else:
            print(f"⚠️ Analysis completed with return code: {result.returncode}")
        
        # Display output
        if result.stdout:
            print("\n📈 RESULTS:")
            print("-" * 20)
            # Filter out problematic characters and show clean output
            clean_output = result.stdout.replace('\x9d', '').replace('\x8d', '')
            print(clean_output)
        
        if result.stderr:
            print("\n⚠️ WARNINGS:")
            print("-" * 20)
            clean_errors = result.stderr.replace('\x9d', '').replace('\x8d', '')
            print(clean_errors)
        
        return True
        
    except subprocess.TimeoutExpired:
        print("❌ Analysis timed out (>3 minutes)")
        return False
        
    except UnicodeDecodeError as e:
        print(f"⚠️ Unicode issue detected: {e}")
        print("Trying alternative method...")
        
        # Method 2: Run without capturing output (direct to console)
        try:
            print(f"\n🔄 Running in direct mode (output will appear below):")
            print("-" * 50)
            
            result = subprocess.run([sys.executable, gemini_path])
            
            print("-" * 50)
            print(f"✅ Analysis completed (return code: {result.returncode})")
            return True
            
        except Exception as e2:
            print(f"❌ Alternative method failed: {e2}")
            return False
            
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        return False

def verify_results():
    """Check if analysis produced expected results"""
    
    print(f"\n🔍 VERIFYING RESULTS")
    print("=" * 30)
    
    # Look for any output files or plots that gemini1.py might create
    current_files = os.listdir('.')
    
    # Check for common analysis outputs
    result_indicators = [
        'plot', 'graph', 'result', 'match', 'analysis', 
        '.png', '.jpg', '.pdf', '.html'
    ]
    
    found_outputs = []
    for file in current_files:
        for indicator in result_indicators:
            if indicator.lower() in file.lower():
                found_outputs.append(file)
                break
    
    if found_outputs:
        print("📊 Possible output files created:")
        for file in found_outputs:
            print(f"   📄 {file}")
    else:
        print("ℹ️ No obvious output files detected")
        print("   (Results may have been displayed in console)")
    
    # Check if unkgem files are still there
    unkgem_files = [
        'data/unknown/unkgemB.csv',
        'data/unknown/unkgemL.csv',
        'data/unknown/unkgemU.csv'
    ]
    
    print(f"\n✅ Confirmed - Your converted files are ready:")
    for file_path in unkgem_files:
        if os.path.exists(file_path):
            lines = sum(1 for line in open(file_path))
            print(f"   📄 {file_path} ({lines:,} data points)")

def check_database_files():
    """Check if database files exist"""
    
    print(f"\n🗃️ CHECKING DATABASE FILES")
    print("=" * 30)
    
    db_files = [
        'gemini_db_long_B.csv',
        'gemini_db_long_L.csv', 
        'gemini_db_long_U.csv',
        'gemlib_structural_ready.csv'
    ]
    
    all_found = True
    for db_file in db_files:
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            print(f"   ✅ {db_file} ({size:,} bytes)")
        else:
            print(f"   ❌ {db_file} (missing)")
            all_found = False
    
    if not all_found:
        print(f"\n⚠️ Some database files are missing.")
        print("   Numerical analysis may not work properly without them.")
        print("   Make sure database files are in the main directory.")
    
    return all_found

def main():
    """Main function"""
    
    print("🎯 SAFE GEMINI1 ANALYSIS RUNNER")
    print("=" * 50)
    print("This program safely runs gemini1.py without Unicode issues")
    print("Your files should already be converted by the emergency fix")
    
    # Check database files first
    db_ok = check_database_files()
    
    if not db_ok:
        choice = input(f"\n⚠️ Database files missing. Continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("❌ Aborted - fix database files first")
            return
    
    success = run_gemini1_safely()
    
    if success:
        verify_results()
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("Check the output above for gem identification results")
        
        # Show summary of what was analyzed
        print(f"\n📋 ANALYSIS SUMMARY:")
        print("   Files analyzed: data/unknown/unkgem*.csv")
        print("   These contain your converted spectral data")
        print("   Results show best gem matches from database")
        
    else:
        print(f"\n⚠️ Analysis had issues, but your files are converted correctly.")
        print("You can try running gemini1.py manually:")
        print("   python src/numerical_analysis/gemini1.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    input(f"\nPress Enter to exit...")