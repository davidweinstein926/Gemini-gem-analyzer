#!/usr/bin/env python3
"""
optimized_gemlib_integration.py - OPTIMIZED GEMLIB INTEGRATION
Tailored integration for your gemlib_structural_ready.csv structure
Based on analysis showing 736 gems × 22 columns with selective data quality
Save as: gemini_gemological_analysis/optimized_gemlib_integration.py
"""

import pandas as pd
import numpy as np

class OptimizedGemlibIntegration:
    def __init__(self, csv_file='gemlib_structural_ready.csv'):
        self.csv_file = csv_file
        self.gem_descriptions = {}
        self.gem_details = {}
        self.statistics = {}
        
        # Define core columns for rich descriptions (based on analysis)
        self.core_columns = {
            'id': 'Reference',
            'natural_synthetic': 'Nat./Syn.',
            'species': 'Spec.',
            'variety': 'Var.',
            'color': 'Color',
            'treatment': 'Treatment',
            'origin': 'Origin'
        }
        
        # Optional columns for detailed analysis
        self.optional_columns = {
            'shape': 'Shape',
            'weight': 'Weight',
            'type': 'Type',
            'note': 'Note'
        }
        
        self.load_gem_library()
    
    def load_gem_library(self):
        """Load and process gemlib with optimized handling of null values"""
        try:
            print(f"📚 Loading gem library from {self.csv_file}")
            
            # Load with proper handling
            gemlib = pd.read_csv(self.csv_file)
            gemlib.columns = gemlib.columns.str.strip()
            
            print(f"📊 Loaded {len(gemlib)} entries with {len(gemlib.columns)} columns")
            
            # Validate core columns exist
            missing_core = [col for col in self.core_columns.values() if col not in gemlib.columns]
            if missing_core:
                print(f"❌ Missing core columns: {missing_core}")
                return
            
            # Clean and prepare Reference IDs
            gemlib[self.core_columns['id']] = gemlib[self.core_columns['id']].astype(str).str.strip()
            
            # Create rich descriptions and detailed info
            self.process_gem_data(gemlib)
            
            # Generate statistics
            self.generate_statistics(gemlib)
            
            print(f"✅ Successfully processed {len(self.gem_descriptions)} gem descriptions")
            print(f"📈 Data completeness by field:")
            for field, col in self.core_columns.items():
                if field != 'id':
                    completeness = ((~gemlib[col].isnull()) & (gemlib[col] != ' ') & (gemlib[col] != '')).sum()
                    percentage = (completeness / len(gemlib)) * 100
                    print(f"   {field}: {completeness}/{len(gemlib)} ({percentage:.1f}%)")
                    
        except FileNotFoundError:
            print(f"❌ File not found: {self.csv_file}")
        except Exception as e:
            print(f"❌ Error loading gem library: {e}")
            import traceback
            traceback.print_exc()
    
    def process_gem_data(self, gemlib):
        """Process gem data into descriptions and details"""
        for _, row in gemlib.iterrows():
            gem_id = str(row[self.core_columns['id']])
            
            # Build rich description from core fields
            description_parts = []
            details = {}
            
            for field, col_name in self.core_columns.items():
                if field == 'id':
                    continue
                    
                value = self.clean_value(row.get(col_name, ''))
                details[field] = value
                
                if value:  # Only add non-empty values to description
                    description_parts.append(value)
            
            # Add optional fields to details only
            for field, col_name in self.optional_columns.items():
                if col_name in gemlib.columns:
                    value = self.clean_value(row.get(col_name, ''))
                    if value:
                        details[field] = value
            
            # Create rich description
            if description_parts:
                rich_description = ' | '.join(description_parts)
            else:
                rich_description = f"Gem {gem_id}"
            
            self.gem_descriptions[gem_id] = rich_description
            self.gem_details[gem_id] = details
    
    def clean_value(self, value):
        """Clean and validate field values"""
        if pd.isna(value) or value is None:
            return ''
        
        value_str = str(value).strip()
        
        # Remove common empty indicators
        if value_str in [' ', '', 'nan', 'NaN', 'None', '#SPILL!']:
            return ''
        
        return value_str
    
    def generate_statistics(self, gemlib):
        """Generate comprehensive statistics"""
        self.statistics = {
            'total_gems': len(gemlib),
            'loaded_descriptions': len(self.gem_descriptions),
            'completeness': {},
            'top_species': {},
            'top_varieties': {},
            'natural_vs_synthetic': {},
            'treatment_summary': {},
            'origin_summary': {}
        }
        
        # Field completeness
        for field, col_name in self.core_columns.items():
            if field != 'id':
                non_empty = ((~gemlib[col_name].isnull()) & 
                           (gemlib[col_name] != ' ') & 
                           (gemlib[col_name] != '')).sum()
                self.statistics['completeness'][field] = {
                    'count': non_empty,
                    'percentage': (non_empty / len(gemlib)) * 100
                }
        
        # Content analysis
        if self.core_columns['species'] in gemlib.columns:
            species_counts = gemlib[self.core_columns['species']].value_counts().head(10)
            self.statistics['top_species'] = species_counts.to_dict()
        
        if self.core_columns['variety'] in gemlib.columns:
            variety_counts = gemlib[self.core_columns['variety']].value_counts().head(10)
            self.statistics['top_varieties'] = variety_counts.to_dict()
        
        if self.core_columns['natural_synthetic'] in gemlib.columns:
            nat_syn_counts = gemlib[self.core_columns['natural_synthetic']].value_counts()
            self.statistics['natural_vs_synthetic'] = nat_syn_counts.to_dict()
        
        if self.core_columns['treatment'] in gemlib.columns:
            treatment_counts = gemlib[self.core_columns['treatment']].value_counts().head(10)
            self.statistics['treatment_summary'] = treatment_counts.to_dict()
        
        if self.core_columns['origin'] in gemlib.columns:
            origin_counts = gemlib[self.core_columns['origin']].value_counts().head(10)
            self.statistics['origin_summary'] = origin_counts.to_dict()
    
    def get_gem_description(self, gem_id):
        """Get rich description for gem ID with fallback"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        
        if base_id in self.gem_descriptions:
            description = self.gem_descriptions[base_id]
            return f"{description} (#{base_id})"
        else:
            return f"Gem #{base_id}"
    
    def get_gem_details(self, gem_id):
        """Get detailed properties for gem ID"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        return self.gem_details.get(base_id, {
            'natural_synthetic': '',
            'species': '',
            'variety': '',
            'color': '',
            'treatment': '',
            'origin': ''
        })
    
    def search_gems(self, search_term, limit=20):
        """Search gems by description or properties"""
        search_term = search_term.lower()
        matches = []
        
        for gem_id, description in self.gem_descriptions.items():
            if search_term in description.lower():
                details = self.gem_details.get(gem_id, {})
                matches.append({
                    'gem_id': gem_id,
                    'description': description,
                    'details': details
                })
                
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_gems_by_species(self, species, limit=50):
        """Get all gems of a specific species"""
        species_lower = species.lower()
        matches = []
        
        for gem_id, details in self.gem_details.items():
            if details.get('species', '').lower() == species_lower:
                matches.append({
                    'gem_id': gem_id,
                    'description': self.gem_descriptions.get(gem_id, f"Gem {gem_id}"),
                    'details': details
                })
                
                if len(matches) >= limit:
                    break
        
        return matches
    
    def get_statistics(self):
        """Get comprehensive library statistics"""
        return self.statistics
    
    def print_statistics(self):
        """Print formatted statistics"""
        print("\n📊 GEMLIB STATISTICS SUMMARY")
        print("=" * 50)
        
        stats = self.statistics
        print(f"📚 Total Gems: {stats['total_gems']}")
        print(f"✅ Loaded Descriptions: {stats['loaded_descriptions']}")
        
        print(f"\n📈 Data Completeness:")
        for field, data in stats['completeness'].items():
            print(f"   {field.replace('_', ' ').title()}: {data['count']} ({data['percentage']:.1f}%)")
        
        if stats['top_species']:
            print(f"\n💎 Top Species:")
            for species, count in list(stats['top_species'].items())[:5]:
                print(f"   {species}: {count}")
        
        if stats['natural_vs_synthetic']:
            print(f"\n🔬 Natural vs Synthetic:")
            for type_val, count in stats['natural_vs_synthetic'].items():
                print(f"   {type_val}: {count}")
    
    def export_processed_data(self, output_file='processed_gemlib.csv'):
        """Export processed data for verification"""
        try:
            export_data = []
            
            for gem_id, description in self.gem_descriptions.items():
                details = self.gem_details.get(gem_id, {})
                
                row = {
                    'gem_id': gem_id,
                    'rich_description': description,
                    **details
                }
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(output_file, index=False)
            print(f"💾 Processed data exported to: {output_file}")
            
        except Exception as e:
            print(f"❌ Export error: {e}")

def main():
    """Test the optimized integration"""
    print("🔬 OPTIMIZED GEMLIB INTEGRATION TEST")
    print("=" * 50)
    
    # Initialize integration
    integrator = OptimizedGemlibIntegration()
    
    # Print statistics
    integrator.print_statistics()
    
    # Test sample lookups
    if integrator.gem_descriptions:
        print(f"\n🔍 SAMPLE LOOKUPS:")
        sample_ids = list(integrator.gem_descriptions.keys())[:5]
        
        for gem_id in sample_ids:
            description = integrator.get_gem_description(gem_id)
            details = integrator.get_gem_details(gem_id)
            print(f"\n💎 {description}")
            for key, value in details.items():
                if value:  # Only show non-empty details
                    print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Test search functionality
        print(f"\n🔍 SEARCH TESTS:")
        
        # Search for diamonds
        diamond_results = integrator.search_gems("diamond", limit=3)
        if diamond_results:
            print(f"\nDiamond search results:")
            for result in diamond_results:
                print(f"   {result['gem_id']}: {result['description']}")
        
        # Search for rubies
        ruby_results = integrator.search_gems("ruby", limit=3)
        if ruby_results:
            print(f"\nRuby search results:")
            for result in ruby_results:
                print(f"   {result['gem_id']}: {result['description']}")
        
        # Export processed data for verification
        print(f"\n💾 EXPORTING PROCESSED DATA:")
        integrator.export_processed_data()
    
    else:
        print("\n❌ No gem descriptions loaded - check file and structure")

# Integration class for use in main system
class GemlibIntegration:
    """Simplified interface for main system integration"""
    
    def __init__(self):
        self.integrator = OptimizedGemlibIntegration()
    
    def get_gem_description(self, gem_id):
        """Get rich description for gem ID"""
        return self.integrator.get_gem_description(gem_id)
    
    def get_gem_details(self, gem_id):
        """Get detailed gem properties"""
        return self.integrator.get_gem_details(gem_id)
    
    def is_loaded(self):
        """Check if library loaded successfully"""
        return len(self.integrator.gem_descriptions) > 0
    
    def get_count(self):
        """Get number of loaded gems"""
        return len(self.integrator.gem_descriptions)

if __name__ == "__main__":
    main()