#!/usr/bin/env python3
"""
Directory Setup Guide for Organized Structural Analysis System
Run this to create the proper directory structure and file locations
"""

from pathlib import Path
import shutil

def create_directory_structure():
    """Create the organized directory structure"""
    print("📁 CREATING ORGANIZED DIRECTORY STRUCTURE")
    print("=" * 60)
    
    project_root = Path.cwd()
    print(f"Project root: {project_root}")
    
    # Create src directories
    directories = [
        "src/structural_analysis",
        "src/visualization", 
        "outputs/structural_results/reports",
        "outputs/structural_results/graphs",
        "data/structural_data",
        "data/structural(archive)",
        "data/raw (archive)",
        "data/unknown/structural"
    ]
    
    print(f"\n📂 Creating directories:")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print(f"\n📄 File placement guide:")
    print(f"   📍 unified_structural_analyzer.py → src/structural_analysis/")
    print(f"   📍 structural_visualizer.py → src/visualization/")
    print(f"   📍 Updated main.py methods → replace in existing main.py")
    
    return True

def check_existing_files():
    """Check if files need to be moved"""
    print(f"\n🔍 CHECKING EXISTING FILES")
    print("-" * 40)
    
    project_root = Path.cwd()
    files_to_move = []
    
    # Check for files in project root that should be moved
    if (project_root / "unified_structural_analyzer.py").exists():
        files_to_move.append({
            'current': project_root / "unified_structural_analyzer.py",
            'target': project_root / "src/structural_analysis/unified_structural_analyzer.py"
        })
    
    if (project_root / "structural_visualizer.py").exists():
        files_to_move.append({
            'current': project_root / "structural_visualizer.py", 
            'target': project_root / "src/visualization/structural_visualizer.py"
        })
    
    if files_to_move:
        print(f"📋 Found {len(files_to_move)} files to move:")
        for file_info in files_to_move:
            print(f"   📄 {file_info['current'].name} → {file_info['target']}")
        
        move_files = input("\nMove files to organized structure? (y/n): ").strip().lower()
        if move_files == 'y':
            for file_info in files_to_move:
                try:
                    # Ensure target directory exists
                    file_info['target'].parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move the file
                    shutil.move(str(file_info['current']), str(file_info['target']))
                    print(f"   ✅ Moved {file_info['current'].name}")
                except Exception as e:
                    print(f"   ❌ Error moving {file_info['current'].name}: {e}")
    else:
        print("   ✅ No files found to move")

def verify_setup():
    """Verify the setup is correct"""
    print(f"\n✅ SETUP VERIFICATION")
    print("-" * 40)
    
    project_root = Path.cwd()
    
    required_files = [
        "src/structural_analysis/unified_structural_analyzer.py",
        "src/visualization/structural_visualizer.py",
        "main.py"
    ]
    
    print(f"📋 Checking required files:")
    all_good = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            all_good = False
    
    print(f"\n📂 Checking directories:")
    required_dirs = [
        "src/structural_analysis",
        "src/visualization",
        "outputs/structural_results/reports", 
        "outputs/structural_results/graphs"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - MISSING")
            all_good = False
    
    if all_good:
        print(f"\n🎉 SETUP COMPLETE!")
        print(f"   All files and directories are in place")
        print(f"   Ready to run Options 4 and 8 with visualization")
    else:
        print(f"\n⚠️  SETUP INCOMPLETE")
        print(f"   Please create missing files/directories")
    
    return all_good

def show_usage_instructions():
    """Show how to use the new organized system"""
    print(f"\n📋 USAGE INSTRUCTIONS")
    print("=" * 60)
    
    print(f"🔬 OPTION 4 (Current Work Analysis):")
    print(f"   • Source: data/structural_data/")
    print(f"   • Runs: src/structural_analysis/unified_structural_analyzer.py current")
    print(f"   • Uses: src/visualization/structural_visualizer.py")
    print(f"   • Output: outputs/structural_results/reports;graphs")
    
    print(f"\n🧪 OPTION 8 (Archived Work Analysis):")
    print(f"   • Source: data/structural(archive)/")
    print(f"   • Runs: src/structural_analysis/unified_structural_analyzer.py archive") 
    print(f"   • Uses: src/visualization/structural_visualizer.py")
    print(f"   • Output: outputs/structural_results/reports;graphs")
    
    print(f"\n📁 ORGANIZED STRUCTURE:")
    print(f"   src/structural_analysis/    ← Analysis algorithms")
    print(f"   src/visualization/          ← Graph generation")
    print(f"   outputs/structural_results/ ← All output files")
    print(f"   data/structural_data/       ← Work in progress")
    print(f"   data/structural(archive)/   ← Completed work")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Update your main.py with new Option 4 and 8 methods")
    print(f"2. Test with: python main.py → Option 4 or 8")
    print(f"3. Select stone of interest and analyze")
    print(f"4. Check outputs/structural_results/ for results")

def main():
    """Main setup workflow"""
    print("🔧 STRUCTURAL ANALYSIS ORGANIZATION SETUP")
    print("=" * 70)
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Check and move existing files  
    check_existing_files()
    
    # Step 3: Verify setup
    verify_setup()
    
    # Step 4: Show usage instructions
    show_usage_instructions()

if __name__ == "__main__":
    main()