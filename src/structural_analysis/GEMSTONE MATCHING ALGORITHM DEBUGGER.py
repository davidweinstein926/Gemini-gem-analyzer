#!/usr/bin/env python3
"""
GEMSTONE MATCHING ALGORITHM DEBUGGER
Debug why stone 197 (1 mound) isn't matching itself but matches stone 56 (2 mounds, iolite)

This will step through the matching process and show exactly what's happening.
"""

import sqlite3
import pandas as pd
import os
from typing import Dict, List
import json

class MatchingDebugger:
    def __init__(self, db_path: str = "multi_spectral_gemstone_db.db"):
        self.db_path = db_path
        
    def debug_self_test_failure(self, unknown_stone_id: str = "unkBC3", 
                               expected_match: str = "197BC3"):
        """Debug why a self-test is failing"""
        
        print("🐛 GEMSTONE MATCHING ALGORITHM DEBUGGER")
        print("=" * 60)
        print(f"Unknown Stone: {unknown_stone_id}")
        print(f"Expected Match: {expected_match}")
        print(f"Database: {self.db_path}")
        
        # Step 1: Check if database exists
        if not os.path.exists(self.db_path):
            print("❌ CRITICAL: Database file doesn't exist!")
            print(f"   Looking for: {self.db_path}")
            print("💡 Make sure you're running this in the same directory as your database")
            return
        
        print("✅ Database file found")
        
        # Step 2: Connect and examine database structure
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📊 Database tables: {[table[0] for table in tables]}")
        
        # Step 3: Check if stone 197 actually exists in database
        print(f"\n🔍 STEP 3: Checking if stone {expected_match} exists in database...")
        
        # Check if we're looking for full stone ID or just stone number
        print(f"🔍 Looking for stone ID: {expected_match}")
        
        # First check: Look for the full stone ID in spectral_data table
        spectral_check_query = "SELECT * FROM spectral_data WHERE full_stone_id = ?"
        spectral_check_result = pd.read_sql_query(spectral_check_query, conn, params=[expected_match])
        
        if spectral_check_result.empty:
            # Fallback: Extract just the stone number and check stone_catalog
            stone_number = expected_match[:3] if expected_match.startswith(('197', '056', '058', '060')) else expected_match.split('B')[0].split('L')[0].split('U')[0]
            print(f"🔍 Extracted stone number: {stone_number}")
            stone_catalog_query = "SELECT * FROM stone_catalog WHERE reference = ?"
            stone_catalog_result = pd.read_sql_query(stone_catalog_query, conn, params=[stone_number])
        else:
            print(f"✅ Found full stone ID {expected_match} in spectral_data")
            stone_number = spectral_check_result.iloc[0]['stone_reference']
            stone_catalog_query = "SELECT * FROM stone_catalog WHERE reference = ?"
            stone_catalog_result = pd.read_sql_query(stone_catalog_query, conn, params=[stone_number])
        
        if stone_catalog_result.empty:
            print(f"❌ Stone {expected_match} NOT FOUND in stone_catalog table")
            print("💡 This means the stone was never properly added to the database")
            
            # Show what stones ARE in the database
            all_stones = pd.read_sql_query("SELECT reference, species, variety, stone_type FROM stone_catalog LIMIT 10", conn)
            print(f"\n📋 First 10 stones in database:")
            for _, row in all_stones.iterrows():
                print(f"   {row['reference']}: {row['species']} {row['variety']} ({row['stone_type']})")
        else:
            print(f"✅ Stone {expected_match} found in stone_catalog:")
            for col in stone_catalog_result.columns:
                print(f"   {col}: {stone_catalog_result.iloc[0][col]}")
        
        # Check spectral data - look for the specific full stone ID
        print(f"\n🔍 SPECTRAL DATA CHECK:")
        spectral_data_query = "SELECT * FROM spectral_data WHERE full_stone_id = ?"
        spectral_result = pd.read_sql_query(spectral_data_query, conn, params=[expected_match])
        
        if spectral_result.empty:
            print(f"❌ Stone ID {expected_match} has NO SPECTRAL DATA")
            # Check if there are other variants of this stone
            base_stone_number = expected_match[:3] if expected_match.startswith(('197', '056', '058', '060')) else expected_match.split('B')[0].split('L')[0].split('U')[0]
            variants_query = "SELECT full_stone_id FROM spectral_data WHERE stone_reference = ?"
            variants_result = pd.read_sql_query(variants_query, conn, params=[base_stone_number])
            if not variants_result.empty:
                print(f"📊 But found these variants of stone {base_stone_number}:")
                for _, row in variants_result.iterrows():
                    print(f"   {row['full_stone_id']}")
        else:
            print(f"✅ Stone {expected_match} has spectral data:")
            for _, row in spectral_result.iterrows():
                print(f"   Full ID: {row['full_stone_id']}")
                print(f"   Light source: {row['light_source']}, Orientation: {row['orientation']}, Scan: {row['scan_number']}")
        
        # Step 4: Load unknown stone features (from CSV file)
        print(f"\n🔍 STEP 4: Loading unknown stone {unknown_stone_id} features...")
        
        structural_data_dir = r"c:\users\david\gemini sp10 structural data"
        unknown_csv_file = None
        
        if os.path.exists(structural_data_dir):
            for filename in os.listdir(structural_data_dir):
                if filename.startswith(unknown_stone_id) and filename.endswith('.csv'):
                    unknown_csv_file = os.path.join(structural_data_dir, filename)
                    break
        
        if not unknown_csv_file:
            print(f"❌ Could not find CSV file for {unknown_stone_id}")
            print(f"   Searched in: {structural_data_dir}")
            print("💡 Please provide the path to the UNKBC1 CSV file")
            return
        
        print(f"✅ Found CSV file: {os.path.basename(unknown_csv_file)}")
        
        # Load unknown features
        unknown_df = pd.read_csv(unknown_csv_file)
        unknown_features = []
        
        for _, row in unknown_df.iterrows():
            feature = {
                'feature_type': row.get('Feature'),
                'start_wavelength': row.get('Start') if pd.notna(row.get('Start')) else None,
                'midpoint_wavelength': row.get('Midpoint') if pd.notna(row.get('Midpoint')) else None,
                'end_wavelength': row.get('End') if pd.notna(row.get('End')) else None,
                'crest_wavelength': row.get('Crest') if pd.notna(row.get('Crest')) else None,
                'max_wavelength': row.get('Max') if pd.notna(row.get('Max')) else None,
                'bottom_wavelength': row.get('Bottom') if pd.notna(row.get('Bottom')) else None,
                'symmetry_ratio': row.get('Symmetry_Ratio') if pd.notna(row.get('Symmetry_Ratio')) else None
            }
            unknown_features.append(feature)
        
        print(f"📊 Unknown stone features ({len(unknown_features)} total):")
        for i, feature in enumerate(unknown_features, 1):
            print(f"   {i}. {feature['feature_type']}")
            if feature['feature_type'] == 'Mound':
                print(f"      Start: {feature.get('start_wavelength')}")
                print(f"      Crest: {feature.get('crest_wavelength')}")
                print(f"      End: {feature.get('end_wavelength')}")
        
        # Step 5: Get stone 197 features from database
        print(f"\n🔍 STEP 5: Loading stone {expected_match} features from database...")
        
        if not spectral_result.empty:
            spectral_id = spectral_result.iloc[0]['spectral_id']
            features_query = "SELECT * FROM structural_features WHERE spectral_id = ?"
            db_features_df = pd.read_sql_query(features_query, conn, params=[spectral_id])
            
            if db_features_df.empty:
                print(f"❌ Stone {expected_match} has NO STRUCTURAL FEATURES in database")
            else:
                print(f"📊 Database stone features ({len(db_features_df)} total):")
                for _, row in db_features_df.iterrows():
                    print(f"   {row['feature_type']}")
                    if row['feature_type'] == 'Mound':
                        print(f"      Start: {row['start_wavelength']}")
                        print(f"      Crest: {row['crest_wavelength']}") 
                        print(f"      End: {row['end_wavelength']}")
                
                # Step 6: Compare features directly
                print(f"\n🔍 STEP 6: Direct feature comparison...")
                self.compare_features_directly(unknown_features, db_features_df.to_dict('records'))
        
        # Step 7: Check what stone 56 (iolite) looks like
        print(f"\n🔍 STEP 7: Examining the incorrect match (stone 56 iolite)...")
        
        stone_56_catalog = pd.read_sql_query("SELECT * FROM stone_catalog WHERE reference = '56'", conn)
        if not stone_56_catalog.empty:
            print(f"📊 Stone 56 info:")
            print(f"   Species: {stone_56_catalog.iloc[0]['species']}")
            print(f"   Variety: {stone_56_catalog.iloc[0]['variety']}")
            print(f"   Type: {stone_56_catalog.iloc[0]['stone_type']}")
            
            # Get stone 56 spectral data
            stone_56_spectral = pd.read_sql_query("SELECT * FROM spectral_data WHERE stone_reference = '56'", conn)
            if not stone_56_spectral.empty:
                spectral_id_56 = stone_56_spectral.iloc[0]['spectral_id']
                stone_56_features = pd.read_sql_query("SELECT * FROM structural_features WHERE spectral_id = ?", conn, params=[spectral_id_56])
                
                print(f"📊 Stone 56 features ({len(stone_56_features)} total):")
                for _, row in stone_56_features.iterrows():
                    print(f"   {row['feature_type']}")
                    if row['feature_type'] == 'Mound':
                        print(f"      Start: {row['start_wavelength']}")
                        print(f"      Crest: {row['crest_wavelength']}")
                        print(f"      End: {row['end_wavelength']}")
        
        # Step 8: Run the actual matching algorithm and trace it
        print(f"\n🔍 STEP 8: Tracing the matching algorithm...")
        self.trace_matching_algorithm(unknown_features, conn)
        
        conn.close()
    
    def compare_features_directly(self, unknown_features: List[Dict], db_features: List[Dict]):
        """Direct comparison of features"""
        print("🔍 DIRECT FEATURE COMPARISON:")
        print("-" * 40)
        
        unknown_mounds = [f for f in unknown_features if f['feature_type'] == 'Mound']
        db_mounds = [f for f in db_features if f['feature_type'] == 'Mound']
        
        print(f"Unknown mounds: {len(unknown_mounds)}")
        print(f"Database mounds: {len(db_mounds)}")
        
        if unknown_mounds and db_mounds:
            unknown_mound = unknown_mounds[0]
            db_mound = db_mounds[0]
            
            print(f"\nMound comparison:")
            print(f"   Unknown crest: {unknown_mound.get('crest_wavelength')}")
            print(f"   Database crest: {db_mound.get('crest_wavelength')}")
            
            if unknown_mound.get('crest_wavelength') and db_mound.get('crest_wavelength'):
                crest_diff = abs(unknown_mound['crest_wavelength'] - db_mound['crest_wavelength'])
                print(f"   Crest difference: ±{crest_diff:.1f} nm")
                
                if crest_diff < 5.0:
                    print(f"   ✅ This should be an EXCELLENT match!")
                else:
                    print(f"   ⚠️ Significant difference - may not be same stone")
        
        # Check if the wavelength values are identical (perfect self-test)
        print(f"\n🔍 PERFECT MATCH CHECK:")
        if unknown_mounds and db_mounds:
            um = unknown_mounds[0]
            dm = db_mounds[0]
            
            identical_values = []
            if um.get('start_wavelength') == dm.get('start_wavelength'):
                identical_values.append('start')
            if um.get('crest_wavelength') == dm.get('crest_wavelength'):
                identical_values.append('crest')
            if um.get('end_wavelength') == dm.get('end_wavelength'):
                identical_values.append('end')
            
            if len(identical_values) >= 2:
                print(f"   🎯 IDENTICAL VALUES: {', '.join(identical_values)}")
                print(f"   🚨 This IS the same stone - algorithm should score 100%!")
            else:
                print(f"   📊 Different values - may be different analyses of same stone")
    
    def trace_matching_algorithm(self, unknown_features: List[Dict], conn: sqlite3.Connection):
        """Trace through the matching algorithm step by step"""
        print("🔍 MATCHING ALGORITHM TRACE:")
        print("-" * 40)
        
        # Get all candidates from database - prioritize same light source and orientation
        candidates_query = """
        SELECT DISTINCT sc.reference, sc.species, sc.variety, sc.stone_type,
               sd.full_stone_id, sd.light_source, sd.orientation, sd.scan_number
        FROM stone_catalog sc
        JOIN spectral_data sd ON sc.reference = sd.stone_reference
        ORDER BY 
            CASE WHEN sd.light_source = 'B' THEN 1 ELSE 2 END,
            CASE WHEN sd.orientation = 'C' THEN 1 ELSE 2 END,
            sc.reference
        """
        candidates_df = pd.read_sql_query(candidates_query, conn)
        
        print(f"📊 Total candidates in database: {len(candidates_df)}")
        print(f"🔍 Looking for matches with same light source (B) and orientation (C)...")
        
        # Show compatible candidates first
        compatible_candidates = candidates_df[
            (candidates_df['light_source'] == 'B') & 
            (candidates_df['orientation'] == 'C')
        ]
        print(f"🎯 Compatible candidates (B-C): {len(compatible_candidates)}")
        
        # Score each candidate
        scores = []
        
        for _, candidate in candidates_df.iterrows():
            stone_ref = candidate['reference']
            full_stone_id = candidate['full_stone_id']
            
            # Get candidate features using the spectral_id
            candidate_features_query = """
            SELECT sf.* FROM structural_features sf
            JOIN spectral_data sd ON sf.spectral_id = sd.spectral_id
            WHERE sd.full_stone_id = ?
            """
            candidate_features_df = pd.read_sql_query(candidate_features_query, conn, params=[full_stone_id])
            candidate_features = candidate_features_df.to_dict('records')
            
            if not candidate_features:
                continue
            
            # Calculate score using simplified logic
            score = self.calculate_simple_match_score(unknown_features, candidate_features)
            
            scores.append({
                'stone_id': stone_ref,
                'full_stone_id': full_stone_id,
                'species': candidate['species'],
                'variety': candidate['variety'],
                'stone_type': candidate['stone_type'],
                'light_source': candidate['light_source'],
                'orientation': candidate['orientation'],
                'score': score,
                'feature_count': len(candidate_features)
            })
        
        # Sort by score
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n🏆 TOP 5 MATCHES:")
        for i, match in enumerate(scores[:5], 1):
            compatibility = "✅" if match['light_source'] == 'B' and match['orientation'] == 'C' else "⚠️"
            print(f"   {i}. {match['full_stone_id']}: {match['score']:.1f}% {compatibility}")
            print(f"      {match['species']} {match['variety']} ({match['stone_type']})")
            print(f"      Light: {match['light_source']}, Orient: {match['orientation']}, Features: {match['feature_count']}")
            
            if match['full_stone_id'] == expected_match:
                print(f"      🎯 THIS IS THE EXPECTED PERFECT MATCH!")
            elif match['stone_id'] == '56':
                print(f"      🚨 THIS IS THE INCORRECT MATCH (stone 56 iolite)!")
        
        # Find the expected match in the results
        expected_result = next((s for s in scores if s['full_stone_id'] == expected_match), None)
        if expected_result:
            expected_rank = scores.index(expected_result) + 1
            print(f"\n📊 Expected match {expected_match} ranked #{expected_rank} with {expected_result['score']:.1f}% confidence")
            if expected_rank > 1:
                print(f"🚨 PROBLEM: {expected_match} should be #1 with ~100% confidence!")
            if expected_result['score'] < 95:
                print(f"🚨 PROBLEM: Self-test should score 95-100%, got {expected_result['score']:.1f}%")
        else:
            print(f"\n❌ Expected match {expected_match} not found in matching results!")
            print(f"💡 Check if {expected_match} was properly saved to database")
    
    def calculate_simple_match_score(self, unknown_features: List[Dict], 
                                   candidate_features: List[Dict]) -> float:
        """Simplified matching score calculation for debugging"""
        
        if not unknown_features or not candidate_features:
            return 0.0
        
        total_score = 0.0
        matches_found = 0
        
        for unknown_feature in unknown_features:
            best_match_score = 0.0
            
            for candidate_feature in candidate_features:
                if unknown_feature['feature_type'] != candidate_feature['feature_type']:
                    continue
                
                feature_type = unknown_feature['feature_type']
                
                if feature_type == 'Mound':
                    # Compare crest wavelengths (most important)
                    unknown_crest = unknown_feature.get('crest_wavelength')
                    candidate_crest = candidate_feature.get('crest_wavelength')
                    
                    if unknown_crest and candidate_crest:
                        diff = abs(unknown_crest - candidate_crest)
                        if diff <= 10.0:  # ±10nm tolerance
                            score = 100 * (1 - diff / 10.0)
                            if score > best_match_score:
                                best_match_score = score
                
                elif feature_type == 'Peak':
                    # Compare max wavelengths
                    unknown_max = unknown_feature.get('max_wavelength')
                    candidate_max = candidate_feature.get('max_wavelength')
                    
                    if unknown_max and candidate_max:
                        diff = abs(unknown_max - candidate_max)
                        if diff <= 2.0:  # ±2nm tolerance for peaks
                            score = 100 * (1 - diff / 2.0)
                            if score > best_match_score:
                                best_match_score = score
            
            if best_match_score > 0:
                total_score += best_match_score
                matches_found += 1
        
        # Average score across all unknown features
        return total_score / len(unknown_features) if len(unknown_features) > 0 else 0.0

def main():
    """Main debugging function"""
    debugger = MatchingDebugger()
    
    print("🔧 GEMSTONE MATCHING DEBUGGER")
    print("=" * 50)
    print("This will debug why UNKBC1 isn't matching 197BC3")
    print("Key: unk-B-C-3 vs 197-B-C-3 (same light source & orientation)")
    
    # Check for different database file names
    possible_db_names = [
        "multi_spectral_gemstone_db.db",
        "gemstone_features.db", 
        "gemstone_database.db"
    ]
    
    db_found = None
    for db_name in possible_db_names:
        if os.path.exists(db_name):
            db_found = db_name
            break
    
    if db_found:
        print(f"✅ Using database: {db_found}")
        debugger.db_path = db_found
    else:
        print("⚠️ No database found. Trying default name.")
    
    # Debug the specific self-test failure
    debugger.debug_self_test_failure("unkBC3", "197BC3")

if __name__ == "__main__":
    main()