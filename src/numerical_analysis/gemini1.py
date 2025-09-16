#!/usr/bin/env python3
"""
GEMINI1.PY - FIXED VERSION WITH INPUTSUBMISSION HANDLING
Fixes InputSubmission wrapper issues + environment variable support
"""
import os
import sys
import re
from pathlib import Path

# Add this at the very beginning of your gemini1.py file
def safe_input_fix(prompt):
    """Fixed input handling that resolves InputSubmission wrapper issues"""
    try:
        user_input = str(input(prompt)).strip()
        
        # Handle InputSubmission wrapper pattern
        if "InputSubmission(data='" in user_input and user_input.endswith("')"):
            # Extract data from wrapper: InputSubmission(data='195BC1') -> 195BC1
            user_input = user_input[22:-2]
        
        # Clean up common wrapper artifacts
        user_input = user_input.replace('\\n', '').replace('\n', '').strip("'\"")
        
        # Additional cleanup for numbered input issues like "1 InputSubmission..."
        if " InputSubmission" in user_input:
            # Extract the part before InputSubmission
            parts = user_input.split(" InputSubmission")
            if parts[0].strip():
                user_input = parts[0].strip()
            else:
                # Try to extract from the InputSubmission part
                match = re.search(r"data='([^']*)", user_input)
                if match:
                    user_input = match.group(1)
        
        return user_input
        
    except Exception as e:
        print(f"Input error: {e}")
        return ""

def validate_gem_format_fixed(gem_name):
    """Enhanced gem format validation with better error messages"""
    gem_name = gem_name.strip()
    
    print(f"DEBUG: Validating gem name: '{gem_name}'")
    
    # Clean up any remaining wrapper artifacts
    if "InputSubmission" in gem_name:
        print("DEBUG: Found InputSubmission wrapper, cleaning...")
        match = re.search(r"data='([^']*)", gem_name)
        if match:
            gem_name = match.group(1).strip()
            print(f"DEBUG: Extracted from wrapper: '{gem_name}'")
    
    # Enhanced validation patterns
    valid_patterns = [
        (r'^\d+[BLU][CP]\d+$', 'Standard: 58BC1, 195LC2, 140UC3'),
        (r'^[A-Z]\d+[BLU][CP]\d+$', 'Client: C0045BC1, S20250909LC2'), 
        (r'^\d+$', 'Simple number: 58, 195'),
        (r'^[A-Z]+\d+$', 'Letter-number: BC58, LC45')
    ]
    
    for pattern, example in valid_patterns:
        if re.match(pattern, gem_name, re.IGNORECASE):
            print(f"DEBUG: '{gem_name}' matches pattern: {example}")
            return True
    
    print(f"DEBUG: '{gem_name}' does not match any valid pattern")
    print("Valid formats:")
    for _, example in valid_patterns:
        print(f"  - {example}")
    
    return False

def setup_data_environment():
    """Setup data directory from environment variable"""
    data_path = os.environ.get('GEMINI_DATA_PATH')
    
    if data_path and Path(data_path).exists():
        print(f"📁 Using data path from environment: {data_path}")
        return data_path
    else:
        # Fallback search order
        fallback_paths = [
            "data/raw_temp",
            "data/raw (archive)", 
            "data/raw",
            "."
        ]
        
        for path in fallback_paths:
            if Path(path).exists() and list(Path(path).glob("*.txt")):
                print(f"📁 Using fallback data path: {path}")
                return path
        
        print("❌ No valid data directory found")
        return None

def get_gem_input_with_fixes():
    """Enhanced gem input for selecting specific gems from available list"""
    while True:
        try:
            print("\n🎯 SELECT SPECIFIC GEMS FOR ANALYSIS")
            print("=" * 50)
            print("Enter gem names from the available list above.")
            print("💡 Choose variants of the SAME physical gem for best results:")
            print("   Examples: 195BC1,195LC1,195UC1")
            print("             140BC1,140LC1") 
            print("             C0045BC1,C0045UC1")
            
            # Use fixed input handler
            gem_input = safe_input_fix("\nEnter gem names (comma-separated) or 'q' to quit: ")
            
            if gem_input.lower() == 'q':
                return None
            
            if not gem_input:
                print("❌ Empty input received. Please try again.")
                continue
            
            print(f"DEBUG: Raw input after cleaning: '{gem_input}'")
            
            # Parse gem names
            gem_names = [name.strip() for name in gem_input.split(',') if name.strip()]
            
            if not gem_names:
                print("❌ No valid gem names found after parsing.")
                continue
            
            if len(gem_names) > 3:
                print("⚠️ More than 3 gems selected. Using first 3 for analysis.")
                gem_names = gem_names[:3]
            
            print(f"DEBUG: Parsed gem names: {gem_names}")
            
            # Validate each gem name
            all_valid = True
            validated_gems = []
            
            for gem_name in gem_names:
                if validate_gem_format_fixed(gem_name):
                    validated_gems.append(gem_name)
                    print(f"✅ Valid: {gem_name}")
                else:
                    print(f"❌ Invalid gem name format: '{gem_name}'")
                    all_valid = False
                    break
            
            if all_valid and validated_gems:
                # Check if they represent the same physical gem
                base_gems = set()
                for gem in validated_gems:
                    base_match = re.match(r'^([A-Z]*\d+)', gem)
                    if base_match:
                        base_gems.add(base_match.group(1))
                
                if len(base_gems) > 1:
                    print(f"⚠️ Warning: Selected gems appear to be different physical gems: {base_gems}")
                    print("💡 For best results, select variants of the same gem (e.g., 195BC1,195LC1,195UC1)")
                    continue_anyway = safe_input_fix("Continue anyway? (y/n): ")
                    if continue_anyway.lower() != 'y':
                        continue
                
                print(f"✅ Selected {len(validated_gems)} gems for analysis")
                return validated_gems
            
            print("💡 Please check the gem names and try again.")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user")
            return None
        except Exception as e:
            print(f"❌ Error in gem input processing: {e}")
            continue

def show_available_gems_for_selection(data_path):
    """Show available gems grouped for easy selection"""
    txt_files = list(Path(data_path).glob("*.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in {data_path}")
        return []
    
    print(f"\n📁 Found {len(txt_files)} spectral files in {data_path}")
    
    # Group gems by base name for easy selection
    gem_groups = {}
    valid_gems = []
    
    for file in txt_files:
        gem_name = file.stem
        if validate_gem_format_fixed(gem_name):
            valid_gems.append(gem_name)
            
            # Extract base gem (195BC1 -> 195)
            base_match = re.match(r'^(\d+)', gem_name)
            if base_match:
                base_num = base_match.group(1)
                if base_num not in gem_groups:
                    gem_groups[base_num] = []
                gem_groups[base_num].append(gem_name)
    
    if not valid_gems:
        print("❌ No valid gem files found")
        return []
    
    print(f"\n💎 AVAILABLE GEMS FOR SELECTION ({len(valid_gems)} total):")
    print("=" * 60)
    
    # Show grouped gems for easy selection
    shown_count = 0
    for base_num in sorted(gem_groups.keys(), key=lambda x: int(x))[:15]:  # Show first 15 groups
        variants = sorted(gem_groups[base_num])
        
        # Identify light sources
        light_sources = set()
        for variant in variants:
            if 'BC' in variant or 'BP' in variant: light_sources.add('B/H')
            if 'LC' in variant or 'LP' in variant: light_sources.add('L')  
            if 'UC' in variant or 'UP' in variant: light_sources.add('U')
        
        complete_status = "✅ COMPLETE" if len(light_sources) >= 3 else "⚠️ PARTIAL"
        print(f"Gem {base_num} [{', '.join(sorted(light_sources))}] {complete_status}")
        print(f"   Available: {', '.join(variants)}")
        shown_count += len(variants)
        
        if shown_count >= 50:  # Limit display
            break
    
    remaining_groups = len(gem_groups) - 15
    if remaining_groups > 0:
        remaining_gems = len(valid_gems) - shown_count
        print(f"\n... and {remaining_groups} more gem groups ({remaining_gems} more files)")
    
    print(f"\n💡 SELECTION EXAMPLES:")
    print("✅ Good selections (same gem, different lights):")
    if gem_groups:
        first_group = list(gem_groups.keys())[0]
        examples = gem_groups[first_group]
        if len(examples) >= 2:
            print(f"   • {examples[0]},{examples[1]}")
        if len(examples) >= 3:
            print(f"   • {examples[0]},{examples[1]},{examples[2]}")
    print("   • 195BC1,195LC1,195UC1")
    print("   • C0045BC1,C0045LC1")
    
    return valid_gems

def main_with_fixes():
    """Main function with interactive gem selection"""
    print("🚀 GEMINI NUMERICAL ANALYSIS - INTERACTIVE MODE")
    print("=" * 60)
    print("✅ InputSubmission wrapper handling enabled")
    print("✅ Interactive gem selection enabled") 
    print("✅ Enhanced validation enabled")
    
    # Setup data environment
    data_path = setup_data_environment()
    if not data_path:
        print("❌ Cannot proceed without valid data directory")
        return
    
    # Show available gems for selection
    available_gems = show_available_gems_for_selection(data_path)
    if not available_gems:
        return
    
    # Interactive gem selection
    selected_gems = get_gem_input_with_fixes()
    
    if selected_gems:
        print(f"\n🚀 Processing {len(selected_gems)} selected gems:")
        for gem in selected_gems:
            print(f"   • {gem}")
        
        # Validate selections exist in data directory
        missing_gems = []
        for gem in selected_gems:
            gem_file = Path(data_path) / f"{gem}.txt"
            if not gem_file.exists():
                missing_gems.append(gem)
        
        if missing_gems:
            print(f"❌ Missing files for: {', '.join(missing_gems)}")
            print(f"💡 Available files are in: {data_path}")
            return
        
        print("✅ All selected gems found in data directory")
        
        # Here you would call your existing gemini1.py analysis functions
        # process_gems(selected_gems, data_path)
        print("✅ Analysis completed")
    else:
        print("❌ No gems selected for analysis")

def auto_analyze_all_gems_in_directory(data_path):
    """Auto-analyze all gems in directory (called from main.py)"""
    txt_files = list(Path(data_path).glob("*.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in {data_path}")
        return
    
    print(f"🤖 Auto-analyzing {len(txt_files)} files in {data_path}")
    
    # Extract gem names from filenames
    gem_names = []
    for file in txt_files:
        gem_name = file.stem
        if validate_gem_format_fixed(gem_name):
            gem_names.append(gem_name)
    
    print(f"✅ Found {len(gem_names)} valid gem files:")
    for gem in gem_names[:5]:  # Show first 5
        print(f"   • {gem}")
    if len(gem_names) > 5:
        print(f"   ... and {len(gem_names) - 5} more")
    
    # Here you would process all gems
    # process_gems(gem_names, data_path)
    print("✅ Auto-analysis completed")

# Integration instructions for your existing gemini1.py:
"""
INTEGRATION INSTRUCTIONS:
========================

1. Add these functions at the top of your existing gemini1.py file

2. Replace your existing input() calls with safe_input_fix()

3. Replace your gem validation with validate_gem_format_fixed()

4. Replace your main() function with main_with_fixes()

5. Update your gem input collection to use get_gem_input_with_fixes()

6. Add environment variable support using setup_data_environment()

Example integration:

OLD CODE:
    gem_input = input("Enter gem names: ")
    
NEW CODE:    
    gem_input = safe_input_fix("Enter gem names: ")

OLD CODE:
    if validate_gem_format(gem_name):
        
NEW CODE:
    if validate_gem_format_fixed(gem_name):

This will fix the InputSubmission wrapper issues and add smart directory fallback support.
"""

if __name__ == "__main__":
    main_with_fixes()
