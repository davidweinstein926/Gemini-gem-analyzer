#!/usr/bin/env python3
"""
EXTRACT DETAILED SCORES FROM FOUND JSON FILE
Read the actual analysis results and show valid gems only
"""

import json
import re
from pathlib import Path

def analyze_found_json():
    """Analyze the specific JSON file we found"""
    
    json_file = Path("C:/Users/David/OneDrive/Desktop/gemini_gemological_analysis/outputs/structural_results/reports/structural_analysis_full_20250928_135329.json")
    
    if not json_file.exists():
        print(f"❌ File not found: {json_file}")
        return
    
    print(f"📄 Reading: {json_file.name}")
    
    # Load the JSON data
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    print(f"✅ Loaded data for {len(results)} gems")
    
    # Find gem 197 data
    gem_197_result = None
    for result in results:
        if str(result['gem_id']) == '197':
            gem_197_result = result
            break
    
    if not gem_197_result:
        print("❌ Gem 197 not found in results!")
        return
    
    print(f"\n🎯 ANALYZING GEM 197 vs VALID GEMS ONLY")
    print("=" * 60)
    print("📋 GOI: Gem 197 (B+L+U light sources)")
    print("✅ Valid database gems: 140, 196, 197, 199 (have B+L+U)")
    print("🚫 Invalid: Any gem missing B, L, or U (like 189)")
    
    # Valid gems we want to focus on
    valid_gems = ['140', '196', '197', '199']
    
    # Extract scores for each light source
    all_scores = {}
    
    for light_source, ls_data in gem_197_result['light_source_results'].items():
        print(f"\n📊 {light_source} LIGHT SOURCE:")
        print(f"   File analyzed: {ls_data['file']}")
        print(f"   All matches found:")
        
        # Get all matches (not just top 5)
        all_matches = ls_data.get('top_5', [])
        
        valid_found = []
        invalid_found = []
        
        for rank, match in enumerate(all_matches, 1):
            db_gem_id = str(match['db_gem_id'])
            score = match['score']
            
            # Extract base gem ID
            base_id_match = re.search(r'(\d+)', db_gem_id)
            base_id = base_id_match.group(1) if base_id_match else None
            
            # Calculate percentage
            if score <= 0.01:
                percentage = 100.0 - (score * 1000)
                percentage = max(90.0, min(100.0, percentage))
            else:
                percentage = max(0, 100.0 * (1 - score))
            
            # Categorize as valid or invalid
            if base_id in valid_gems:
                valid_found.append({
                    'rank': rank,
                    'base_id': base_id,
                    'db_gem_id': db_gem_id,
                    'score': score,
                    'percentage': percentage
                })
                
                # Store for later combination
                if base_id not in all_scores:
                    all_scores[base_id] = {}
                all_scores[base_id][light_source] = {
                    'db_gem_id': db_gem_id,
                    'score': score,
                    'percentage': percentage,
                    'rank': rank
                }
                
                status = "🎯 SELF-MATCH" if base_id == '197' else "✅ Valid"
                print(f"      {rank}. {db_gem_id} → Gem {base_id} | {score:.6f} | {percentage:.2f}% {status}")
            else:
                invalid_found.append({
                    'rank': rank,
                    'base_id': base_id,
                    'db_gem_id': db_gem_id,
                    'score': score,
                    'percentage': percentage
                })
                print(f"      {rank}. {db_gem_id} → Gem {base_id} | {score:.6f} | {percentage:.2f}% 🚫 INVALID")
        
        # Summary for this light source
        print(f"   📊 Summary: {len(valid_found)} valid, {len(invalid_found)} invalid gems found")
        
        # Check if gem 197 found itself
        gem_197_self = [v for v in valid_found if v['base_id'] == '197']
        if gem_197_self:
            self_match = gem_197_self[0]
            if self_match['score'] < 0.01:
                print(f"   ✅ Gem 197 self-match: Rank {self_match['rank']}, Score {self_match['score']:.6f}")
            else:
                print(f"   ⚠️  Gem 197 self-match: Rank {self_match['rank']}, Score {self_match['score']:.6f} (should be < 0.01)")
        else:
            print(f"   ❌ Gem 197 NOT found in its own light source results!")
    
    # Calculate combined scores for valid gems only
    print(f"\n🏆 COMBINED SCORES (VALID GEMS ONLY):")
    print("=" * 60)
    
    final_rankings = []
    
    for gem_id in valid_gems:
        if gem_id not in all_scores:
            print(f"❌ Gem {gem_id}: No data found")
            continue
        
        gem_data = all_scores[gem_id]
        available_sources = list(gem_data.keys())
        
        # Check completeness (should have B+L+U)
        required = {'B', 'L', 'U'}
        has_all = required.issubset(set(available_sources))
        
        if not has_all:
            missing = required - set(available_sources)
            print(f"⚠️  Gem {gem_id}: Missing {missing} - Partial data only")
        
        # Calculate weighted average
        light_weights = {'B': 1.0, 'L': 0.9, 'U': 0.8}
        
        total_weighted = 0.0
        total_weight = 0.0
        details = []
        
        for ls in available_sources:
            if ls in gem_data:
                weight = light_weights[ls]
                percentage = gem_data[ls]['percentage']
                score = gem_data[ls]['score']
                rank = gem_data[ls]['rank']
                
                total_weighted += percentage * weight
                total_weight += weight
                details.append(f"{ls}:{percentage:.1f}%(rank:{rank})")
        
        if total_weight > 0:
            combined_percentage = total_weighted / total_weight
            
            # Completeness factor
            completeness = 1.0 if has_all else 0.85
            final_percentage = combined_percentage * completeness
            
            final_rankings.append({
                'gem_id': gem_id,
                'percentage': final_percentage,
                'sources': available_sources,
                'details': details,
                'complete': has_all
            })
    
    # Sort by percentage (highest first)
    final_rankings.sort(key=lambda x: x['percentage'], reverse=True)
    
    print(f"\n📊 FINAL RANKING (VALID GEMS ONLY):")
    print("-" * 60)
    
    for rank, result in enumerate(final_rankings, 1):
        gem_id = result['gem_id']
        percentage = result['percentage']
        complete = "✅" if result['complete'] else "⚠️ "
        
        # Special status for gem 197
        if gem_id == '197':
            if rank == 1 and percentage > 95:
                status = "🎯 PERFECT SELF-MATCH"
            elif percentage > 95:
                status = f"🎯 EXCELLENT SELF-MATCH (should be rank 1)"
            else:
                status = f"❌ POOR SELF-MATCH (expected >95%)"
        else:
            status = "✅ Valid competitor"
        
        print(f"{rank}. GEM {gem_id}: {percentage:.2f}% {complete} {status}")
        print(f"   Sources: {'+'.join(result['sources'])}")
        print(f"   Details: {' | '.join(result['details'])}")
        print()
    
    # Analysis and diagnostics
    print(f"🔍 DIAGNOSTIC ANALYSIS:")
    print("-" * 60)
    
    if final_rankings:
        winner = final_rankings[0]
        
        if winner['gem_id'] == '197':
            print(f"✅ CORRECT: Gem 197 is ranking #1 (self-match working)")
            if winner['percentage'] > 95:
                print(f"✅ EXCELLENT: Score {winner['percentage']:.2f}% indicates perfect self-match")
            else:
                print(f"⚠️  CONCERN: Score {winner['percentage']:.2f}% lower than expected for perfect self-match")
        else:
            print(f"❌ PROBLEM: Gem {winner['gem_id']} ranking #1 instead of Gem 197")
            
            # Find gem 197's position
            for rank, result in enumerate(final_rankings, 1):
                if result['gem_id'] == '197':
                    print(f"   Gem 197 is at rank #{rank} with {result['percentage']:.2f}%")
                    print(f"   This suggests self-match detection is not working properly")
                    break
    
    # Check what the original analyzer was doing wrong
    print(f"\n❓ WHY WAS GEM 189 WINNING ORIGINALLY?")
    print("-" * 60)
    print("Based on console output, gem 189 had very low scores:")
    print("   B Light: 0.0022 (very good)")
    print("   L Light: Not best for this source")
    print("   U Light: Unknown")
    print("")
    print("🚫 Problem: Gem 189 shouldn't be considered at all!")
    print("   Reason: It's missing light sources (doesn't have B+L+U)")
    print("   Fix needed: Filter database to only include complete gems")

def main():
    """Main function"""
    print("🔍 DETAILED ANALYSIS OF ACTUAL RESULTS")
    print("=" * 60)
    print("📋 Extracting scores for valid gems: 140, 196, 197, 199")
    print("🚫 Ignoring invalid gems (like 189)")
    
    analyze_found_json()
    
    print(f"\n💡 RECOMMENDED FIXES:")
    print("1. Add database filtering: Only gems with same light sources as GOI")
    print("2. Improve self-match detection: Gem 197 should score ~0.001 with itself") 
    print("3. Exclude incomplete gems: Gem 189 and others missing B/L/U")

if __name__ == "__main__":
    main()