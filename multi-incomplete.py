"""
MODIFIED METHODS FOR multi_gem_structural_analyzer.py
Copy and paste each method, replacing the existing versions

SEARCH FOR THE METHOD NAME in multi_gem_structural_analyzer.py
Then replace the entire method (from 'def' to the next method)
"""

# ============================================================================
# METHOD 1: Replace calculate_similarity_scores (around line 500)
# SEARCH FOR: def calculate_similarity_scores(self, unknown_df, db_matches
# ============================================================================

def calculate_similarity_scores(self, unknown_df, db_matches, light_source, file_path):
    """Calculate similarity scores - FEATURE AWARE VERSION"""
    
    # Import the feature-aware scorer
    try:
        from feature_aware_scorer import calculate_feature_aware_score
    except ImportError:
        print("ERROR: feature_aware_scorer.py not found")
        return []
    
    scores = []
    file_col = self.database_schema['file_column']
    
    # Extract GOI metadata
    goi_base_id, goi_ts = self.extract_base_id_and_ts(file_path.name)
    
    # Score each database gem
    for file_id, gem_data in db_matches.groupby(file_col):
        try:
            # Extract database metadata
            db_base_id, db_ts_from_file = self.extract_base_id_and_ts(str(file_id))
            
            # Get timestamp from analysis_date if available
            date_col = self.database_schema.get('analysis_date_column')
            if date_col and date_col in db_matches.columns:
                date_values = gem_data[date_col].dropna().unique()
                if len(date_values) > 0:
                    date_str = str(date_values[0])
                    if '-' in date_str:
                        db_ts = date_str.replace('-', '')[:8]
                    else:
                        db_ts = date_str[:8] if len(date_str) >= 8 else date_str
                else:
                    db_ts = db_ts_from_file
            else:
                db_ts = db_ts_from_file
            
            # Map light source code to full name
            light_name_map = {
                'B': 'Halogen',
                'L': 'Laser',
                'U': 'UV'
            }
            light_full_name = light_name_map.get(light_source, light_source)
            
            # Call feature-aware scorer
            result = calculate_feature_aware_score(
                unknown_df,
                gem_data,
                light_full_name,
                file_path.name,
                file_id,
                goi_ts if goi_ts else '',
                db_ts if db_ts else '',
                goi_base_id if goi_base_id else '',
                db_base_id if db_base_id else ''
            )
            
            if result:
                # Check if self-match was detected
                if result.get('is_perfect'):
                    print(f"🎯 SELF-MATCH: {file_path.name} ↔ {file_id} (score: {result['score']:.6f})")
                
                scores.append(result)
        
        except Exception as e:
            print(f"Error scoring {file_id}: {e}")
            continue
    
    return scores


# ============================================================================
# METHOD 2: Replace score_to_percentage (around line 580)
# SEARCH FOR: def score_to_percentage(self, score)
# ============================================================================

def score_to_percentage(self, score):
    """Convert feature-aware penalty score to percentage"""
    try:
        from feature_aware_scorer import convert_penalty_to_percentage
        return convert_penalty_to_percentage(score)
    except ImportError:
        # Fallback if import fails
        if score is None:
            return 0.0
        if score <= 0:
            return 100.0
        elif score >= 100:
            return 0.0
        else:
            return max(0.0, 100.0 - score)


# ============================================================================
# METHOD 3: Replace calculate_combined_scores (around line 350)
# SEARCH FOR: def calculate_combined_scores(self, light_source_results)
# ============================================================================

def calculate_combined_scores(self, light_source_results):
    """Calculate combined scores - UPDATED for penalty-based scoring"""
    
    try:
        from feature_aware_scorer import convert_penalty_to_percentage
    except ImportError:
        print("ERROR: feature_aware_scorer.py not found")
        return []
    
    gem_ts_combinations = {}
    
    for ls, ls_data in light_source_results.items():
        for match in ls_data['all_scores']:
            gem_id = match['db_gem_id']
            
            base_id, _ = self.extract_base_id_and_ts(gem_id)
            ts = match.get('db_ts')
            
            ts_key = f"{base_id}_{ts}" if ts else base_id
            
            if ts_key not in gem_ts_combinations:
                gem_ts_combinations[ts_key] = {
                    'base_id': base_id,
                    'ts': ts,
                    'light_data': {}
                }
            
            # Convert penalty score to percentage
            percentage = convert_penalty_to_percentage(match['score'])
            
            gem_ts_combinations[ts_key]['light_data'][ls] = {
                'gem_id': gem_id,
                'score': match['score'],
                'percentage': percentage,
                'is_perfect': match.get('is_perfect', False)
            }
    
    # Calculate combined scores
    valid_combinations = []
    
    for ts_key, combo_data in gem_ts_combinations.items():
        light_data = combo_data['light_data']
        base_id = combo_data['base_id']
        ts = combo_data['ts']
        
        light_scores = {}
        perfect_count = 0
        
        for ls, ls_info in light_data.items():
            light_scores[ls] = ls_info['percentage']
            if ls_info['is_perfect']:
                perfect_count += 1
        
        if light_scores:
            # Weighted average of percentages
            weighted_sum = sum(score * self.light_weights.get(ls, 1.0) 
                             for ls, score in light_scores.items())
            total_weight = sum(self.light_weights.get(ls, 1.0) 
                             for ls in light_scores.keys())
            
            avg_percentage = weighted_sum / total_weight
            completeness = 1.0 if len(light_scores) == 3 else 0.8
            perfect_bonus = 1.0 + perfect_count * 0.05
            
            final_percentage = min(100.0, avg_percentage * completeness * perfect_bonus)
            
            # Convert back to score for sorting (lower is better)
            combined_score = (100.0 - final_percentage) / 25.0
            
            ts_display = f" ({ts})" if ts else ""
            print(f"🔄 Combined score for {base_id}{ts_display}: {combined_score:.4f} ({final_percentage:.1f}%)")
            
            valid_combinations.append({
                'db_gem_id': f"{base_id}{ts_display}",
                'base_id': base_id,
                'ts': ts,
                'score': combined_score,
                'percentage': final_percentage,
                'light_sources': list(light_scores.keys()),
                'perfect_count': perfect_count,
                'light_details': light_data
            })
    
    return valid_combinations


"""
HOW TO APPLY:

1. Open: src/structural_analysis/multi_gem_structural_analyzer.py

2. Use Ctrl+F to search for "def calculate_similarity_scores"
   - Delete the entire existing method
   - Paste METHOD 1 from above

3. Use Ctrl+F to search for "def score_to_percentage" 
   - Delete the entire existing method
   - Paste METHOD 2 from above

4. Use Ctrl+F to search for "def calculate_combined_scores"
   - Delete the entire existing method
   - Paste METHOD 3 from above

5. Save the file

6. Test: python main.py -> Option 4 -> gem 197

Expected: "🎯 SELF-MATCH:" messages and gem 197 scores ~100%
"""