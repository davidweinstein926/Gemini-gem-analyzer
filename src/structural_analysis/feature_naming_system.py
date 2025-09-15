#!/usr/bin/env python3
"""
feature_naming_system.py
Enhanced integration with gemlib_structural_ready.csv
Handles naming inconsistencies like "plateau" vs "shoulder"
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from collections import defaultdict
from difflib import SequenceMatcher
import re

class StructuralFeatureNamingSystem:
    """
    System to handle structural feature naming consistency
    and integrate gemlib_structural_ready.csv data more thoroughly
    """
    
    def __init__(self, csv_file='gemlib_structural_ready.csv', db_file='multi_structural_gem_data.db'):
        self.csv_file = csv_file
        self.db_file = db_file
        
        # Load existing data
        self.gemlib_data = self.load_gemlib_data()
        self.feature_mapping = self.create_feature_mapping()
        self.similarity_threshold = 0.8
        
        # Initialize database connection
        self.init_database()
    
    def load_gemlib_data(self):
        """Load and clean gemlib_structural_ready.csv data"""
        if not os.path.exists(self.csv_file):
            print(f"Warning: {self.csv_file} not found")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.csv_file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Ensure Reference column exists and is string type
            if 'Reference' in df.columns:
                df['Reference'] = df['Reference'].astype(str).str.strip()
            
            # Create comprehensive gem description
            desc_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
            available_columns = [col for col in desc_columns if col in df.columns]
            
            if available_columns:
                df['Full_Description'] = df[available_columns].apply(
                    lambda x: ' '.join([str(v).strip() for v in x if pd.notnull(v) and str(v).strip() != '']), 
                    axis=1
                )
            
            print(f"✅ Loaded {len(df)} gems from {self.csv_file}")
            return df
            
        except Exception as e:
            print(f"Error loading {self.csv_file}: {e}")
            return pd.DataFrame()
    
    def create_feature_mapping(self):
        """
        Create mapping for similar structural features to standardize naming
        This addresses the plateau vs shoulder inconsistency issue
        """
        feature_groups = {
            # Broad features - similar shapes but different names
            'plateau': {
                'primary': 'plateau',
                'aliases': ['plateau', 'broad_absorption', 'flat_region', 'shelf', 'terrace'],
                'description': 'Wide, flat absorption feature with gradual slopes',
                'characteristics': ['wide_width', 'low_slope', 'flat_top']
            },
            'shoulder': {
                'primary': 'shoulder', 
                'aliases': ['shoulder', 'side_peak', 'asymmetric_extension', 'wing'],
                'description': 'Asymmetric extension from main peak',
                'characteristics': ['asymmetric', 'attached_to_peak', 'medium_width']
            },
            
            # Peak-related features
            'sharp_peak': {
                'primary': 'peak_sharp',
                'aliases': ['sharp_peak', 'narrow_peak', 'spike', 'needle_peak'],
                'description': 'Narrow, intense absorption peak',
                'characteristics': ['narrow_width', 'high_intensity', 'steep_slopes']
            },
            'broad_peak': {
                'primary': 'peak_broad',
                'aliases': ['broad_peak', 'wide_peak', 'mound', 'hill'],
                'description': 'Wide absorption peak with gentle slopes',
                'characteristics': ['wide_width', 'moderate_intensity', 'gentle_slopes']
            },
            
            # Valley/minimum features
            'valley': {
                'primary': 'valley',
                'aliases': ['valley', 'trough', 'dip', 'minimum', 'depression'],
                'description': 'Low transmission region between features',
                'characteristics': ['low_transmission', 'between_peaks']
            },
            
            # Edge/transition features
            'edge': {
                'primary': 'absorption_edge',
                'aliases': ['edge', 'absorption_edge', 'cutoff', 'onset', 'transition'],
                'description': 'Sharp increase in absorption at specific wavelength',
                'characteristics': ['sharp_transition', 'threshold_like']
            },
            
            # Complex features
            'doublet': {
                'primary': 'doublet',
                'aliases': ['doublet', 'double_peak', 'split_peak', 'twin_peak'],
                'description': 'Two closely spaced peaks appearing as one feature',
                'characteristics': ['two_components', 'close_spacing', 'similar_intensity']
            },
            'multiplet': {
                'primary': 'multiplet',
                'aliases': ['multiplet', 'multiple_peak', 'peak_cluster', 'group'],
                'description': 'Group of multiple closely related peaks',
                'characteristics': ['multiple_components', 'grouped', 'related_origin']
            }
        }
        
        # Create reverse mapping for quick lookup
        alias_to_primary = {}
        for primary, data in feature_groups.items():
            for alias in data['aliases']:
                alias_to_primary[alias.lower()] = primary
        
        return {
            'groups': feature_groups,
            'alias_mapping': alias_to_primary
        }
    
    def standardize_feature_name(self, feature_name, context=None):
        """
        Standardize a feature name using the mapping system
        Handles cases like "plateau" vs "shoulder" by looking at context
        """
        if not feature_name or pd.isna(feature_name):
            return 'unknown_feature'
        
        # Clean the input
        clean_name = str(feature_name).lower().strip()
        clean_name = re.sub(r'[^a-z0-9_\s]', '', clean_name)
        clean_name = re.sub(r'\s+', '_', clean_name)
        
        # Direct mapping first
        if clean_name in self.feature_mapping['alias_mapping']:
            primary = self.feature_mapping['alias_mapping'][clean_name]
            return self.feature_mapping['groups'][primary]['primary']
        
        # Similarity matching for fuzzy cases
        best_match = None
        best_score = 0
        
        for alias, primary_key in self.feature_mapping['alias_mapping'].items():
            similarity = SequenceMatcher(None, clean_name, alias).ratio()
            if similarity > best_score and similarity > self.similarity_threshold:
                best_score = similarity
                best_match = self.feature_mapping['groups'][primary_key]['primary']
        
        if best_match:
            print(f"Fuzzy match: '{feature_name}' -> '{best_match}' (score: {best_score:.2f})")
            return best_match
        
        # Context-based disambiguation for plateau vs shoulder
        if context:
            if self.is_plateau_like(clean_name, context):
                return 'plateau'
            elif self.is_shoulder_like(clean_name, context):
                return 'shoulder'
        
        # Return original if no match found
        print(f"No standardization for: '{feature_name}' -> keeping original")
        return clean_name
    
    def is_plateau_like(self, feature_name, context):
        """Determine if a feature is plateau-like based on context"""
        plateau_indicators = [
            'flat', 'wide', 'broad', 'extended', 'stable', 'constant',
            'level', 'horizontal', 'uniform'
        ]
        
        # Check feature name
        name_score = sum(1 for indicator in plateau_indicators 
                        if indicator in feature_name.lower())
        
        # Check context (wavelength range, intensity characteristics, etc.)
        context_score = 0
        if context:
            if context.get('width', 0) > 50:  # Wide feature
                context_score += 2
            if context.get('slope_variation', 1) < 0.1:  # Flat
                context_score += 2
            if context.get('intensity_std', 1) < 0.05:  # Stable intensity
                context_score += 1
        
        return (name_score + context_score) >= 2
    
    def is_shoulder_like(self, feature_name, context):
        """Determine if a feature is shoulder-like based on context"""
        shoulder_indicators = [
            'side', 'wing', 'asymmetric', 'extension', 'attached',
            'adjacent', 'secondary', 'off', 'tail'
        ]
        
        # Check feature name
        name_score = sum(1 for indicator in shoulder_indicators 
                        if indicator in feature_name.lower())
        
        # Check context
        context_score = 0
        if context:
            if context.get('asymmetry', 0) > 0.3:  # Asymmetric
                context_score += 2
            if context.get('attached_to_main_peak', False):  # Attached to main feature
                context_score += 2
            if context.get('width', 0) < 30:  # Narrower than plateau
                context_score += 1
        
        return (name_score + context_score) >= 2
    
    def init_database(self):
        """Initialize enhanced structural database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Enhanced structural features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structural_features_enhanced (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gem_id TEXT,
                    light_source TEXT,
                    wavelength REAL,
                    feature_type_raw TEXT,
                    feature_type_standardized TEXT,
                    intensity REAL,
                    width REAL,
                    asymmetry REAL,
                    slope_left REAL,
                    slope_right REAL,
                    context_data TEXT,
                    confidence_score REAL,
                    analysis_method TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Gem information table (from CSV data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gem_information (
                    reference TEXT PRIMARY KEY,
                    nat_syn TEXT,
                    species TEXT,
                    variety TEXT,
                    treatment TEXT,
                    origin TEXT,
                    full_description TEXT,
                    additional_info TEXT
                )
            """)
            
            # Feature standardization rules table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feature_standardization_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_name TEXT,
                    standardized_name TEXT,
                    rule_type TEXT,
                    confidence REAL,
                    context_required BOOLEAN,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            print("✅ Enhanced database schema created/verified")
            
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def populate_gem_information(self):
        """Populate gem information table from CSV data"""
        if self.gemlib_data.empty:
            print("No gemlib data to populate")
            return
        
        try:
            conn = sqlite3.connect(self.db_file)
            
            # Clear existing data
            conn.execute("DELETE FROM gem_information")
            
            # Insert CSV data
            for _, row in self.gemlib_data.iterrows():
                conn.execute("""
                    INSERT INTO gem_information 
                    (reference, nat_syn, species, variety, treatment, origin, full_description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('Reference', ''),
                    row.get('Nat./Syn.', ''),
                    row.get('Spec.', ''),
                    row.get('Var.', ''),
                    row.get('Treatment', ''),
                    row.get('Origin', ''),
                    row.get('Full_Description', '')
                ))
            
            conn.commit()
            conn.close()
            print(f"✅ Populated {len(self.gemlib_data)} gem records in database")
            
        except Exception as e:
            print(f"Error populating gem information: {e}")
    
    def add_structural_feature(self, gem_id, light_source, wavelength, feature_type, 
                             intensity=None, width=None, context=None, analysis_method='manual'):
        """Add a structural feature with standardized naming"""
        
        # Standardize the feature name
        standardized_type = self.standardize_feature_name(feature_type, context)
        
        # Calculate additional metrics if context provided
        asymmetry = context.get('asymmetry', 0) if context else 0
        slope_left = context.get('slope_left', 0) if context else 0
        slope_right = context.get('slope_right', 0) if context else 0
        confidence = context.get('confidence', 0.8) if context else 0.8
        
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO structural_features_enhanced
                (gem_id, light_source, wavelength, feature_type_raw, feature_type_standardized,
                 intensity, width, asymmetry, slope_left, slope_right, context_data,
                 confidence_score, analysis_method)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                gem_id, light_source, wavelength, feature_type, standardized_type,
                intensity, width, asymmetry, slope_left, slope_right,
                str(context) if context else None, confidence, analysis_method
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Added feature: {gem_id} {light_source} {wavelength}nm - {feature_type} -> {standardized_type}")
            return True
            
        except Exception as e:
            print(f"Error adding structural feature: {e}")
            return False
    
    def get_gem_description(self, gem_id):
        """Get comprehensive gem description from database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            result = cursor.execute("""
                SELECT reference, nat_syn, species, variety, treatment, origin, full_description
                FROM gem_information 
                WHERE reference = ?
            """, (gem_id,)).fetchone()
            
            conn.close()
            
            if result:
                return {
                    'reference': result[0],
                    'nat_syn': result[1], 
                    'species': result[2],
                    'variety': result[3],
                    'treatment': result[4],
                    'origin': result[5],
                    'full_description': result[6]
                }
            else:
                # Fallback to direct CSV lookup
                if not self.gemlib_data.empty and 'Reference' in self.gemlib_data.columns:
                    match = self.gemlib_data[self.gemlib_data['Reference'].astype(str) == str(gem_id)]
                    if not match.empty:
                        row = match.iloc[0]
                        return {
                            'reference': gem_id,
                            'species': row.get('Spec.', ''),
                            'variety': row.get('Var.', ''),
                            'full_description': row.get('Full_Description', f'Gem {gem_id}')
                        }
                
                return {'reference': gem_id, 'full_description': f'Gem {gem_id}'}
                
        except Exception as e:
            print(f"Error getting gem description: {e}")
            return {'reference': gem_id, 'full_description': f'Gem {gem_id}'}
    
    def analyze_feature_naming_inconsistencies(self):
        """Analyze existing structural database for naming inconsistencies"""
        try:
            conn = sqlite3.connect(self.db_file)
            
            # Get all existing features
            features_df = pd.read_sql_query("""
                SELECT feature_type_raw, COUNT(*) as count,
                       COUNT(DISTINCT gem_id) as unique_gems,
                       light_source
                FROM structural_features 
                GROUP BY feature_type_raw, light_source
                ORDER BY count DESC
            """, conn)
            
            conn.close()
            
            print("\n📊 EXISTING FEATURE NAMING ANALYSIS:")
            print("=" * 50)
            
            # Group similar features
            similar_groups = defaultdict(list)
            for _, row in features_df.iterrows():
                feature_name = row['feature_type_raw']
                standardized = self.standardize_feature_name(feature_name)
                similar_groups[standardized].append({
                    'original': feature_name,
                    'count': row['count'],
                    'gems': row['unique_gems'],
                    'light': row['light_source']
                })
            
            # Report on inconsistencies
            inconsistencies = []
            for standardized, variants in similar_groups.items():
                if len(variants) > 1:
                    inconsistencies.append((standardized, variants))
                    
                    print(f"\n🔍 STANDARDIZED: '{standardized}'")
                    total_count = sum(v['count'] for v in variants)
                    print(f"   Total occurrences: {total_count}")
                    print(f"   Variants found: {len(variants)}")
                    
                    for variant in sorted(variants, key=lambda x: x['count'], reverse=True):
                        print(f"     • '{variant['original']}': {variant['count']} times, "
                              f"{variant['gems']} gems, {variant['light']} light")
            
            print(f"\n📈 SUMMARY:")
            print(f"   Total unique features: {len(features_df)}")
            print(f"   Standardized groups: {len(similar_groups)}")
            print(f"   Groups with inconsistencies: {len(inconsistencies)}")
            
            return inconsistencies
            
        except Exception as e:
            print(f"Error analyzing inconsistencies: {e}")
            return []
    
    def create_standardization_report(self):
        """Create a comprehensive report on feature standardization"""
        report_lines = [
            "STRUCTURAL FEATURE STANDARDIZATION REPORT",
            "=" * 50,
            "",
            "FEATURE MAPPING GROUPS:",
            "-" * 25
        ]
        
        for group_name, data in self.feature_mapping['groups'].items():
            report_lines.extend([
                f"\n🔹 {data['primary'].upper()}",
                f"   Description: {data['description']}",
                f"   Aliases: {', '.join(data['aliases'])}",
                f"   Characteristics: {', '.join(data['characteristics'])}"
            ])
        
        report_lines.extend([
            "\n",
            "NAMING RULES:",
            "-" * 13,
            "",
            "1. PLATEAU vs SHOULDER disambiguation:",
            "   • Plateau: Wide (>50nm), flat, stable intensity",
            "   • Shoulder: Asymmetric, attached to peak, narrower",
            "",
            "2. Peak classifications:",
            "   • Sharp peaks: Narrow, steep slopes, high intensity",
            "   • Broad peaks: Wide, gentle slopes, moderate intensity", 
            "",
            "3. Complex features:",
            "   • Doublet: Two closely spaced components",
            "   • Multiplet: Multiple grouped components",
            "",
            "4. Context considerations:",
            "   • Wavelength position relative to other features",
            "   • Light source type (B/H, L, UV)",
            "   • Gem type and expected features"
        ])
        
        return "\n".join(report_lines)
    
    def migrate_existing_data(self):
        """Migrate existing structural features to use standardized naming"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Check if old table exists
            tables = cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='structural_features'
            """).fetchall()
            
            if not tables:
                print("No existing structural_features table to migrate")
                conn.close()
                return
            
            # Get all existing features
            old_features = cursor.execute("""
                SELECT gem_id, light_source, wavelength, feature_type, intensity
                FROM structural_features
            """).fetchall()
            
            print(f"🔄 Migrating {len(old_features)} existing features...")
            
            migrated_count = 0
            for feature in old_features:
                gem_id, light_source, wavelength, feature_type, intensity = feature
                
                # Add to enhanced table with standardized naming
                success = self.add_structural_feature(
                    gem_id=gem_id,
                    light_source=light_source, 
                    wavelength=wavelength,
                    feature_type=feature_type,
                    intensity=intensity,
                    analysis_method='migrated'
                )
                
                if success:
                    migrated_count += 1
            
            conn.close()
            print(f"✅ Migrated {migrated_count}/{len(old_features)} features")
            
        except Exception as e:
            print(f"Migration error: {e}")
    
    def generate_feature_consistency_rules(self, output_file='feature_consistency_rules.txt'):
        """Generate a file with feature consistency rules for manual reference"""
        
        rules_content = f"""GEMINI STRUCTURAL FEATURE CONSISTENCY RULES
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{self.create_standardization_report()}

IMPLEMENTATION GUIDELINES:
========================

For Manual Analysis:
• Use standardized names from the mapping above
• When unsure between plateau/shoulder, consider context:
  - Width: Plateau (>50nm) vs Shoulder (<30nm)
  - Shape: Plateau (flat) vs Shoulder (asymmetric)
  - Position: Shoulder attached to main peak

For Database Storage:
• Store both original and standardized names
• Include context data for disambiguation
• Record confidence scores for automated decisions

Recommended Workflow:
1. Identify feature using traditional methods
2. Apply standardization mapping
3. Check for context-based disambiguation
4. Store with both original and standardized names
5. Review periodically for new inconsistencies

CONTEXT PARAMETERS FOR DISAMBIGUATION:
====================================
• width: Feature width in nm
• asymmetry: 0-1 score (0=symmetric, 1=highly asymmetric)
• slope_left: Left side slope steepness
• slope_right: Right side slope steepness
• attached_to_main_peak: Boolean flag
• intensity_std: Standard deviation of intensity in feature region

QUALITY CONTROL:
===============
• Review naming decisions with confidence < 0.7
• Periodically analyze database for new inconsistencies
• Update rules based on new gem types and features discovered
• Maintain consistency across different analysts
"""
        
        with open(output_file, 'w') as f:
            f.write(rules_content)
        
        print(f"📄 Feature consistency rules saved to: {output_file}")

# Integration functions for main.py and gemini1.py
def enhance_main_py_integration():
    """Integration suggestions for main.py"""
    integration_code = '''
# Add to main.py FixedGeminiAnalysisSystem class:

def __init__(self):
    # ... existing init code ...
    
    # Add structural naming system
    self.naming_system = StructuralFeatureNamingSystem()
    self.naming_system.populate_gem_information()

def get_enhanced_gem_description(self, gem_id):
    """Get enhanced gem description using naming system"""
    return self.naming_system.get_gem_description(gem_id)

def create_enhanced_analysis_report(self, gem_identifier, top_matches):
    """Enhanced report with structural feature analysis"""
    # ... existing report code ...
    
    # Add structural feature consistency analysis
    with open(report_filename, 'a') as f:
        f.write("\\n\\nSTRUCTURAL FEATURE ANALYSIS:\\n")
        f.write("-" * 30 + "\\n")
        
        gem_desc = self.naming_system.get_gem_description(gem_identifier)
        f.write(f"Gem Description: {gem_desc.get('full_description', gem_identifier)}\\n")
        f.write(f"Species: {gem_desc.get('species', 'Unknown')}\\n")
        f.write(f"Variety: {gem_desc.get('variety', 'Unknown')}\\n")
        
        # Add standardized feature naming report
        f.write("\\nFeature Naming Standards Applied\\n")
        f.write(self.naming_system.create_standardization_report())
'''