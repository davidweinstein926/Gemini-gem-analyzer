#!/usr/bin/env python3
"""
raw_data_browser.py - Complete Standalone Raw Data File Browser
Shows all available .txt files organized by gem number for easy selection

USAGE:
1. Save this file as: raw_data_browser.py
2. Put it in your main project directory (gemini_gemological_analysis)
3. Run: python raw_data_browser.py
4. Browse your available spectral files organized by gem number
"""

import os
import re
from collections import defaultdict
from pathlib import Path
import datetime

def scan_raw_data_directory(raw_dir='data/raw'):
    """Scan raw data directory and organize files by gem number"""
    
    if not os.path.exists(raw_dir):
        print(f"❌ Raw data directory '{raw_dir}' not found!")
        return None
    
    # Get all .txt files
    txt_files = []
    for file in os.listdir(raw_dir):
        if file.lower().endswith('.txt'):
            txt_files.append(file)
    
    if not txt_files:
        print(f"❌ No .txt files found in '{raw_dir}'")
        return None
    
    print(f"📂 Found {len(txt_files)} .txt files in '{raw_dir}'")
    
    # Organize files by gem number
    gems = defaultdict(lambda: {'B': [], 'L': [], 'U': [], 'Other': []})
    
    for file in txt_files:
        # Extract gem info from filename
        gem_info = analyze_filename(file)
        gem_num = gem_info['gem_number']
        light_source = gem_info['light_source']
        
        gems[gem_num][light_source].append({
            'filename': file,
            'full_info': gem_info
        })
    
    return dict(gems)

def analyze_filename(filename):
    """Analyze filename to extract gem number, light source, and other info"""
    base = os.path.splitext(filename)[0]  # Remove .txt extension
    
    info = {
        'filename': filename,
        'gem_number': 'Unknown',
        'light_source': 'Other',
        'variant': '',
        'full_code': base
    }
    
    # Pattern matching for different filename formats
    patterns = [
        # Pattern 1: 189BC1, 358LC3, etc. (number + light + variant)
        r'^(\d+)([BLU])([CP]?\d*)$',
        # Pattern 2: gem189B1, sample358L2, etc.
        r'^(?:gem|sample)?(\d+)([BLU])(\d*)$',
        # Pattern 3: 189_B_C1, 358_L_3, etc.
        r'^(\d+)[_-]([BLU])[_-]?([CP]?\d*)$'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, base.upper())
        if match:
            info['gem_number'] = match.group(1)
            info['light_source'] = match.group(2)
            info['variant'] = match.group(3) if match.group(3) else ''
            break
    
    # If no pattern matches, try to find gem number and light source separately
    if info['gem_number'] == 'Unknown':
        # Look for gem number (consecutive digits)
        gem_match = re.search(r'(\d+)', base)
        if gem_match:
            info['gem_number'] = gem_match.group(1)
        
        # Look for light source
        light_match = re.search(r'[BLU]', base.upper())
        if light_match:
            info['light_source'] = light_match.group(0)
    
    return info

def display_organized_files(gems_dict):
    """Display files organized by gem number"""
    
    print("\n" + "="*80)
    print("📊 RAW SPECTRAL DATA FILES ORGANIZED BY GEM")
    print("="*80)
    
    # Sort gems by number
    try:
        sorted_gems = sorted(gems_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
    except:
        sorted_gems = sorted(gems_dict.keys())
    
    total_gems = len(sorted_gems)
    complete_sets = 0
    
    for gem_num in sorted_gems:
        gem_files = gems_dict[gem_num]
        
        # Count available light sources
        available_sources = []
        for source in ['B', 'L', 'U']:
            if gem_files[source]:
                available_sources.append(source)
        
        if len(available_sources) == 3:
            complete_sets += 1
            status_icon = "✅"
            status_text = "COMPLETE SET"
        elif len(available_sources) >= 2:
            status_icon = "🟡"
            status_text = "PARTIAL SET"
        else:
            status_icon = "🔴"
            status_text = "INCOMPLETE"
        
        print(f"\n{status_icon} 💎 GEM {gem_num} ({status_text})")
        print(f"   Available light sources: {', '.join(available_sources) if available_sources else 'None'}")
        
        # Show files for each light source
        for light_source in ['B', 'L', 'U']:
            files = gem_files[light_source]
            if files:
                print(f"   🔥 {light_source} ({len(files)} files):")
                for file_info in files:
                    filename = file_info['filename']
                    variant = file_info['full_info']['variant']
                    variant_text = f" (variant: {variant})" if variant else ""
                    print(f"      📄 {filename}{variant_text}")
            else:
                print(f"   ⚪ {light_source}: (no files)")
        
        # Show other files (if any)
        if gem_files['Other']:
            print(f"   🔍 Other files:")
            for file_info in gem_files['Other']:
                print(f"      📄 {file_info['filename']}")
    
    # Summary statistics
    print(f"\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    print(f"Total gems found: {total_gems}")
    print(f"Complete sets (B+L+U): {complete_sets}")
    print(f"Incomplete sets: {total_gems - complete_sets}")
    
    # Light source availability
    light_counts = {'B': 0, 'L': 0, 'U': 0}
    for gem_files in gems_dict.values():
        for source in ['B', 'L', 'U']:
            if gem_files[source]:
                light_counts[source] += 1
    
    print(f"\nLight source availability:")
    for source, count in light_counts.items():
        percentage = (count / total_gems * 100) if total_gems > 0 else 0
        print(f"   {source}: {count}/{total_gems} gems ({percentage:.1f}%)")

def show_gem_recommendations(gems_dict):
    """Show recommendations for best gems to analyze"""
    
    print(f"\n" + "="*60)
    print("🎯 RECOMMENDED GEMS FOR ANALYSIS")
    print("="*60)
    
    # Find complete sets
    complete_gems = []
    for gem_num, gem_files in gems_dict.items():
        available_sources = []
        for source in ['B', 'L', 'U']:
            if gem_files[source]:
                available_sources.append(source)
        
        if len(available_sources) == 3:
            complete_gems.append((gem_num, gem_files))
    
    if complete_gems:
        print(f"\n✅ COMPLETE SETS (B+L+U available):")
        
        # Sort by gem number
        try:
            complete_gems.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        except:
            complete_gems.sort(key=lambda x: x[0])
        
        for i, (gem_num, gem_files) in enumerate(complete_gems[:10]):  # Show top 10
            print(f"\n   {i+1}. 💎 Gem {gem_num}")
            
            # Show recommended files (first file of each type)
            for source in ['B', 'L', 'U']:
                if gem_files[source]:
                    recommended_file = gem_files[source][0]['filename']
                    alternatives = len(gem_files[source]) - 1
                    alt_text = f" (+{alternatives} alternatives)" if alternatives > 0 else ""
                    print(f"      {source}: {recommended_file}{alt_text}")
        
        if len(complete_gems) > 10:
            print(f"\n   ... and {len(complete_gems) - 10} more complete sets")
    else:
        print("\n❌ No complete sets (B+L+U) found")
    
    # Show partial sets
    partial_gems = []
    for gem_num, gem_files in gems_dict.items():
        available_sources = []
        for source in ['B', 'L', 'U']:
            if gem_files[source]:
                available_sources.append(source)
        
        if len(available_sources) == 2:
            partial_gems.append((gem_num, available_sources))
    
    if partial_gems:
        print(f"\n🟡 PARTIAL SETS (2 light sources):")
        try:
            partial_gems.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        except:
            partial_gems.sort(key=lambda x: x[0])
        
        for gem_num, sources in partial_gems[:5]:  # Show top 5
            print(f"      Gem {gem_num}: {'+'.join(sources)}")
        
        if len(partial_gems) > 5:
            print(f"      ... and {len(partial_gems) - 5} more partial sets")

def interactive_file_selector(gems_dict):
    """Interactive file selector"""
    
    print(f"\n" + "="*60)
    print("🎯 INTERACTIVE FILE SELECTOR")
    print("="*60)
    
    while True:
        gem_choice = input(f"\nEnter gem number to analyze (or 'quit' to exit): ").strip()
        
        if gem_choice.lower() in ['quit', 'exit', 'q']:
            break
        
        if gem_choice not in gems_dict:
            print(f"❌ Gem {gem_choice} not found.")
            
            # Show similar gems
            similar = [g for g in gems_dict.keys() if gem_choice in g]
            if similar:
                print(f"   Similar gems found: {', '.join(similar)}")
            continue
        
        gem_files = gems_dict[gem_choice]
        
        print(f"\n💎 Files available for Gem {gem_choice}:")
        
        selected_files = {}
        
        for source in ['B', 'L', 'U']:
            files = gem_files[source]
            
            if not files:
                print(f"   {source}: ❌ No files available")
                continue
            elif len(files) == 1:
                filename = files[0]['filename']
                selected_files[source] = filename
                print(f"   {source}: ✅ {filename} (auto-selected)")
            else:
                print(f"   {source}: Multiple files available:")
                for i, file_info in enumerate(files):
                    print(f"      {i+1}. {file_info['filename']}")
                
                while True:
                    choice = input(f"   Select file for {source} (1-{len(files)}, or Enter for first): ").strip()
                    
                    if not choice:
                        selected_files[source] = files[0]['filename']
                        print(f"   Selected: {files[0]['filename']}")
                        break
                    
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(files):
                            selected_files[source] = files[idx]['filename']
                            print(f"   Selected: {files[idx]['filename']}")
                            break
                        else:
                            print(f"   Invalid choice. Enter 1-{len(files)}")
                    except ValueError:
                        print(f"   Invalid input. Enter a number 1-{len(files)}")
        
        if selected_files:
            print(f"\n📋 SELECTION SUMMARY for Gem {gem_choice}:")
            for source in ['B', 'L', 'U']:
                if source in selected_files:
                    print(f"   {source}: {selected_files[source]}")
            
            # Generate command for analytical workflow
            print(f"\n💡 TO ANALYZE THESE FILES:")
            print(f"   1. Run the analytical workflow from main menu (option 5)")
            print(f"   2. When prompted, enter these exact filenames:")
            for source in ['B', 'L', 'U']:
                if source in selected_files:
                    print(f"      {source} spectrum file: {selected_files[source]}")
            
            print(f"\n🚀 OR copy this selection for easy reference:")
            print(f"   Gem {gem_choice} files:")
            for source in ['B', 'L', 'U']:
                if source in selected_files:
                    print(f"   {source}: {selected_files[source]}")
        else:
            print(f"❌ No files selected for Gem {gem_choice}")

def export_file_list(gems_dict, raw_dir):
    """Export file list to text file"""
    
    output_file = "raw_data_file_list.txt"
    
    try:
        with open(output_file, 'w') as f:
            f.write("RAW SPECTRAL DATA FILES ORGANIZED BY GEM\n")
            f.write("="*60 + "\n")
            f.write(f"Source directory: {os.path.abspath(raw_dir)}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Sort gems by number
            try:
                sorted_gems = sorted(gems_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
            except:
                sorted_gems = sorted(gems_dict.keys())
            
            for gem_num in sorted_gems:
                gem_files = gems_dict[gem_num]
                
                # Count available light sources
                available_sources = []
                for source in ['B', 'L', 'U']:
                    if gem_files[source]:
                        available_sources.append(source)
                
                f.write(f"\nGem {gem_num} ({len(available_sources)} light sources)\n")
                
                for light_source in ['B', 'L', 'U']:
                    files = gem_files[light_source]
                    if files:
                        f.write(f"  {light_source}: ")
                        filenames = [file_info['filename'] for file_info in files]
                        f.write(", ".join(filenames) + "\n")
                    else:
                        f.write(f"  {light_source}: (none)\n")
        
        print(f"✅ File list exported to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error exporting file list: {e}")

def show_quick_selections(gems_dict):
    """Show quick selection commands for complete gem sets"""
    
    print(f"\n" + "="*60)
    print("⚡ QUICK SELECTION COMMANDS")
    print("="*60)
    
    # Find complete sets
    complete_gems = []
    for gem_num, gem_files in gems_dict.items():
        available_sources = []
        for source in ['B', 'L', 'U']:
            if gem_files[source]:
                available_sources.append(source)
        
        if len(available_sources) == 3:
            complete_gems.append((gem_num, gem_files))
    
    if complete_gems:
        print("\n📋 Ready-to-use file selections for analytical workflow:")
        
        # Sort by gem number
        try:
            complete_gems.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        except:
            complete_gems.sort(key=lambda x: x[0])
        
        for i, (gem_num, gem_files) in enumerate(complete_gems[:5]):  # Show top 5
            print(f"\n🔸 Gem {gem_num}:")
            
            for source in ['B', 'L', 'U']:
                if gem_files[source]:
                    recommended_file = gem_files[source][0]['filename']
                    print(f"   {source}: {recommended_file}")
        
        if len(complete_gems) > 5:
            print(f"\n... and {len(complete_gems) - 5} more complete sets available")
        
        print(f"\n💡 Copy the filenames above and paste them when the analytical workflow asks!")
    else:
        print("\n❌ No complete gem sets found for quick selection")

def main():
    """Main function"""
    print("🔍 RAW DATA FILE BROWSER")
    print("="*50)
    print("This tool helps you browse and select spectral files for analysis")
    print("Place this file in your main project directory and run it")
    
    # Try different possible raw data locations
    possible_dirs = [
        'data/raw',
        '../data/raw', 
        '../../data/raw',
        'raw_data',
        '../raw_data',
        '../../raw_data',
        'src/numerical_analysis/raw_data',
        '.'  # Current directory
    ]
    
    gems_dict = None
    used_dir = None
    
    for raw_dir in possible_dirs:
        if os.path.exists(raw_dir):
            print(f"📂 Checking directory: {raw_dir}")
            gems_dict = scan_raw_data_directory(raw_dir)
            if gems_dict:
                used_dir = raw_dir
                break
    
    if not gems_dict:
        print("❌ No raw data files found in any expected location")
        print(f"\nSearched directories:")
        for dir_path in possible_dirs:
            exists = "✅" if os.path.exists(dir_path) else "❌"
            print(f"   {exists} {os.path.abspath(dir_path)}")
        
        print(f"\n💡 SOLUTION:")
        print(f"   1. Create a 'data/raw' directory")
        print(f"   2. Put your .txt spectral files in it")
        print(f"   3. Run this browser again")
        return
    
    print(f"\n✅ Using raw data directory: {os.path.abspath(used_dir)}")
    
    # Display organized files
    display_organized_files(gems_dict)
    
    # Show recommendations
    show_gem_recommendations(gems_dict)
    
    # Show quick selections
    show_quick_selections(gems_dict)
    
    # Interactive menu
    while True:
        print(f"\n" + "="*50)
        print("🎛️  MENU OPTIONS:")
        print("1. Show organized file list again")
        print("2. Show recommendations")
        print("3. Interactive file selector")
        print("4. Quick selection commands") 
        print("5. Export file list to text")
        print("6. Exit")
        
        choice = input(f"\nChoose option (1-6): ").strip()
        
        if choice == '1':
            display_organized_files(gems_dict)
        elif choice == '2':
            show_gem_recommendations(gems_dict)
        elif choice == '3':
            interactive_file_selector(gems_dict)
        elif choice == '4':
            show_quick_selections(gems_dict)
        elif choice == '5':
            export_file_list(gems_dict, used_dir)
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")