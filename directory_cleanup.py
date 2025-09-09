#!/usr/bin/env python3
"""
directory_cleanup.py - Fix incorrect directory placement
Moves misplaced folders and ensures correct structure
"""

import os
import shutil
import sys
from pathlib import Path

class DirectoryCleanup:
    """Fix directory structure issues"""
    
    def __init__(self):
        self.base_dir = "gemini_gemological_analysis"
        
    def check_misplaced_folders(self):
        """Check for folders in wrong locations"""
        print("Checking for misplaced directories...")
        
        issues_found = []
        
        # Check for data and raw_txt in numerical_analysis
        numerical_analysis_path = Path(self.base_dir) / "src" / "numerical_analysis"
        
        if (numerical_analysis_path / "data").exists():
            issues_found.append(("data", "src/numerical_analysis/data", "data"))
            
        if (numerical_analysis_path / "raw_txt").exists():
            issues_found.append(("raw_txt", "src/numerical_analysis/raw_txt", "raw_txt"))
        
        # Check for other common misplacements
        structural_analysis_path = Path(self.base_dir) / "src" / "structural_analysis"
        
        if (structural_analysis_path / "database").exists():
            issues_found.append(("database", "src/structural_analysis/database", "database"))
            
        return issues_found
    
    def display_issues(self, issues):
        """Display found issues"""
        if not issues:
            print("No directory placement issues found.")
            return False
        
        print("\nDirectory placement issues found:")
        print("-" * 50)
        
        for folder_name, wrong_location, correct_location in issues:
            print(f"ISSUE: {folder_name}")
            print(f"  Currently at: {wrong_location}")
            print(f"  Should be at:  {correct_location}")
            print()
        
        return True
    
    def fix_directory_structure(self, issues):
        """Fix the directory placement issues"""
        print("Fixing directory structure...")
        
        base_path = Path(self.base_dir)
        
        for folder_name, wrong_location, correct_location in issues:
            source_path = base_path / Path(wrong_location).relative_to(Path(self.base_dir))
            target_path = base_path / correct_location
            
            print(f"\nFixing {folder_name}:")
            print(f"  Moving from: {source_path}")
            print(f"  Moving to:   {target_path}")
            
            try:
                # If target already exists, we need to merge or replace
                if target_path.exists():
                    print(f"  Target already exists, merging...")
                    self.merge_directories(source_path, target_path)
                    shutil.rmtree(source_path)
                else:
                    # Simple move
                    shutil.move(str(source_path), str(target_path))
                
                print(f"  ✅ Fixed: {folder_name}")
                
            except Exception as e:
                print(f"  ❌ Error fixing {folder_name}: {e}")
    
    def merge_directories(self, source, target):
        """Merge source directory into target directory"""
        for item in source.iterdir():
            target_item = target / item.name
            
            if item.is_file():
                if target_item.exists():
                    print(f"    Replacing file: {item.name}")
                shutil.copy2(item, target_item)
            elif item.is_dir():
                if target_item.exists():
                    print(f"    Merging directory: {item.name}")
                    self.merge_directories(item, target_item)
                else:
                    print(f"    Moving directory: {item.name}")
                    shutil.copytree(item, target_item)
    
    def verify_correct_structure(self):
        """Verify the directory structure is now correct"""
        print("\nVerifying correct directory structure...")
        
        base_path = Path(self.base_dir)
        expected_structure = {
            "data": ["raw", "raw_txt", "structrual_data", "unknown"],
            "database": ["reference_spectra", "structural_spectra", "gem_library", "gemasign"],
            "src": ["numerical_analysis", "structural_analysis", "visualization"],
            "docs": [],
            "data_acquisition": ["aseq", "Lib", "user"]
        }
        
        all_good = True
        
        for main_folder, subfolders in expected_structure.items():
            main_path = base_path / main_folder
            
            if main_path.exists():
                print(f"✅ {main_folder}/ exists")
                
                for subfolder in subfolders:
                    sub_path = main_path / subfolder
                    if sub_path.exists():
                        print(f"  ✅ {main_folder}/{subfolder}/ exists")
                    else:
                        print(f"  ⚠️  {main_folder}/{subfolder}/ missing")
                        all_good = False
            else:
                print(f"❌ {main_folder}/ missing")
                all_good = False
        
        return all_good
    
    def create_directory_tree_display(self):
        """Create a visual display of current directory structure"""
        print("\nCurrent directory structure:")
        print("=" * 50)
        
        base_path = Path(self.base_dir)
        
        if not base_path.exists():
            print(f"Directory {self.base_dir} does not exist!")
            return
        
        self.display_tree(base_path, "", True)
    
    def display_tree(self, path, prefix="", is_last=True):
        """Recursively display directory tree"""
        if path.name.startswith('.'):
            return
            
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{path.name}/")
        
        if path.is_dir():
            try:
                children = sorted([p for p in path.iterdir() if p.is_dir() and not p.name.startswith('.')])
                for i, child in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    extension = "    " if is_last else "│   "
                    self.display_tree(child, prefix + extension, is_last_child)
            except PermissionError:
                pass
    
    def run_cleanup(self):
        """Run the complete cleanup process"""
        print("Gemini Directory Structure Cleanup Tool")
        print("=" * 50)
        
        if not Path(self.base_dir).exists():
            print(f"Error: {self.base_dir} directory not found!")
            print("Run this script from the directory containing gemini_gemological_analysis/")
            return False
        
        # Check for issues
        issues = self.check_misplaced_folders()
        has_issues = self.display_issues(issues)
        
        if not has_issues:
            print("Directory structure appears correct!")
            self.verify_correct_structure()
            return True
        
        # Ask for confirmation
        response = input("\nFix these directory placement issues? (y/n): ").strip().lower()
        
        if response == 'y':
            # Create backup
            print("\nCreating backup...")
            backup_path = f"{self.base_dir}_backup"
            if Path(backup_path).exists():
                shutil.rmtree(backup_path)
            shutil.copytree(self.base_dir, backup_path)
            print(f"Backup created: {backup_path}")
            
            # Fix issues
            self.fix_directory_structure(issues)
            
            # Verify
            if self.verify_correct_structure():
                print("\n✅ Directory structure cleanup completed successfully!")
                
                # Ask about removing backup
                remove_backup = input("\nRemove backup? (y/n): ").strip().lower()
                if remove_backup == 'y':
                    shutil.rmtree(backup_path)
                    print("Backup removed.")
                else:
                    print(f"Backup kept at: {backup_path}")
            else:
                print("\n⚠️ Some issues may remain. Check the structure manually.")
        else:
            print("Cleanup cancelled.")
        
        return True

def main():
    """Main function"""
    cleanup = DirectoryCleanup()
    
    # Show current structure first
    cleanup.create_directory_tree_display()
    
    # Run cleanup
    cleanup.run_cleanup()

if __name__ == "__main__":
    main()