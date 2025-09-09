#!/usr/bin/env python3
"""
fast_gem_analysis.py - OPTIMIZED ANALYSIS FOR LARGE DATABASE
Your database has 968K+ records! No wonder gemini1.py was slow.
This script is optimized for fast results.
"""

import pandas as pd
import numpy as np
import time

def fast_gem_analysis():
    """Fast analysis optimized for large database"""
    
    print("🚀 FAST GEM ANALYSIS - OPTIMIZED FOR LARGE DATABASE")
    print("=" * 60)
    
    # Load your unknown files
    print("📂 Loading your unknown gem files...")
    unknown_files = {}
    
    for light in ['B', 'L', 'U']:
        path = f'data/unknown/unkgem{light}.csv'
        try:
            df = pd.read_csv(path, header=None, names=['wavelength', 'intensity'])
            unknown_files[light] = df
            print(f"   ✅ {light}: {len(df)} points loaded")
        except Exception as e:
            print(f"   ❌ {light}: Error loading - {e}")
    
    if not unknown_files:
        print("❌ No unknown files loaded!")
        return
    
    # Load databases (with progress)
    print(f"\n🗃️ Loading reference database...")
    databases = {}
    
    for light in ['B', 'L', 'U']:
        if light not in unknown_files:
            continue
            
        db_file = f'gemini_db_long_{light}.csv'
        print(f"   Loading {light} database...")
        start_time = time.time()
        
        try:
            db = pd.read_csv(db_file)
            databases[light] = db
            load_time = time.time() - start_time
            gems = db['full_name'].nunique() if 'full_name' in db.columns else 0
            print(f"   ✅ {light}: {len(db):,} records, {gems} gems ({load_time:.1f}s)")
        except Exception as e:
            print(f"   ❌ {light}: Error loading - {e}")
    
    if not databases:
        print("❌ No databases loaded!")
        return
    
    # Fast analysis for each light source
    print(f"\n🔍 RUNNING FAST ANALYSIS...")
    print("=" * 40)
    
    all_results = {}
    
    for light in unknown_files.keys():
        if light not in databases:
            continue
            
        print(f"\n📊 Analyzing {light} spectrum...")
        start_time = time.time()
        
        unknown = unknown_files[light]
        db = databases[light]
        
        # Get unique gems (much faster than processing all records)
        unique_gems = db['full_name'].unique()
        print(f"   Comparing against {len(unique_gems)} unique gems...")
        
        scores = []
        
        # Sample only every 10th gem for speed (can be adjusted)
        sample_gems = unique_gems[::max(1, len(unique_gems)//50)]  # Max 50 gems for speed
        print(f"   Quick sampling: {len(sample_gems)} gems")
        
        for i, gem_name in enumerate(sample_gems):
            if i % 10 == 0:
                print(f"     Progress: {i}/{len(sample_gems)}")
            
            try:
                # Get gem data
                gem_data = db[db['full_name'] == gem_name]
                
                # Quick merge and comparison
                merged = pd.merge(unknown, gem_data[['wavelength', 'intensity']], 
                                on='wavelength', suffixes=('_unk', '_gem'))
                
                if len(merged) > 100:  # Need decent overlap
                    # Fast MSE calculation
                    mse = np.mean((merged['intensity_unk'] - merged['intensity_gem'])**2)
                    scores.append((gem_name, mse))
                    
            except Exception:
                continue
        
        # Sort results
        scores.sort(key=lambda x: x[1])
        all_results[light] = scores[:10]  # Top 10
        
        analysis_time = time.time() - start_time
        print(f"   ✅ {light} analysis complete ({analysis_time:.1f}s)")
        
        # Show top 5 results for this light source
        print(f"   🏆 Top 5 matches:")
        for i, (gem, score) in enumerate(scores[:5]):
            print(f"      {i+1}. {gem}: {score:.2f}")
    
    # Combined results
    print(f"\n🎯 COMBINED RESULTS:")
    print("=" * 30)
    
    # Find gems that appear in multiple light sources
    gem_appearances = {}
    for light, results in all_results.items():
        for gem, score in results:
            base_gem = gem.split('B')[0].split('L')[0].split('U')[0]  # Extract base gem number
            if base_gem not in gem_appearances:
                gem_appearances[base_gem] = {}
            gem_appearances[base_gem][light] = score
    
    # Calculate combined scores for gems with multiple light sources
    combined_scores = []
    for base_gem, light_scores in gem_appearances.items():
        if len(light_scores) >= 2:  # At least 2 light sources
            avg_score = np.mean(list(light_scores.values()))
            combined_scores.append((base_gem, avg_score, len(light_scores), light_scores))
    
    combined_scores.sort(key=lambda x: x[1])
    
    print(f"🏆 BEST OVERALL MATCHES:")
    for i, (gem, avg_score, num_lights, light_scores) in enumerate(combined_scores[:10]):
        print(f"\n{i+1}. 💎 Gem {gem}")
        print(f"   Average Score: {avg_score:.2f}")
        print(f"   Light sources: {num_lights} ({', '.join(light_scores.keys())})")
        for light, score in light_scores.items():
            print(f"     {light}: {score:.2f}")
        
        # Special highlighting for gem 189 (since that's what you're analyzing)
        if "189" in gem:
            print(f"   🎯 THIS IS YOUR GEM! (You're analyzing Gem 189)")
            if avg_score < 1000:  # Adjust threshold as needed
                print(f"   ✅ EXCELLENT MATCH - Very likely correct identification")
            elif avg_score < 5000:
                print(f"   ✅ GOOD MATCH - Likely correct identification")
            else:
                print(f"   ⚠️ MODERATE MATCH - May need verification")
    
    # Load gem library for descriptions
    print(f"\n📚 LOADING GEM DESCRIPTIONS...")
    try:
        gemlib = pd.read_csv('gemlib_structural_ready.csv')
        
        if not combined_scores:
            print("No combined results to describe")
        else:
            best_gem = combined_scores[0][0]
            if 'Reference' in gemlib.columns:
                gem_info = gemlib[gemlib['Reference'].astype(str) == str(best_gem)]
                if not gem_info.empty:
                    print(f"\n💎 BEST MATCH DETAILS:")
                    for col in ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']:
                        if col in gem_info.columns:
                            value = gem_info.iloc[0][col]
                            if pd.notna(value):
                                print(f"   {col}: {value}")
                else:
                    print(f"   No description found for Gem {best_gem}")
            else:
                print("   Gem library format issue")
                
    except Exception as e:
        print(f"   ⚠️ Could not load gem descriptions: {e}")
    
    print(f"\n✅ FAST ANALYSIS COMPLETE!")
    print("This quick analysis compared your Gem 189 against a sample of the database.")
    print("For more comprehensive results, the full analysis would take much longer.")

def main():
    """Main function"""
    print("🎯 FAST ANALYSIS FOR YOUR LARGE DATABASE")
    print("Your database has 968K+ records - that's huge!")
    print("This optimized analysis will give you quick results...\n")
    
    fast_gem_analysis()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    input(f"\nPress Enter to exit...")