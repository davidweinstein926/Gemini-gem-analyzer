#!/usr/bin/env python3
"""
gemlib_structure_analyzer.py - GEMLIB CSV STRUCTURE ANALYZER
Comprehensive analysis tool to determine structure of gemlib_structural_ready.csv
Save as: gemini_gemological_analysis/gemlib_structure_analyzer.py
"""

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import json
from datetime import datetime

class GemlibStructureAnalyzer:
    def __init__(self, csv_file='gemlib_structural_ready.csv'):
        self.csv_file = csv_file
        self.df = None
        self.analysis_results = {}
    
    def analyze_complete_structure(self):
        """Run complete structure analysis"""
        print("🔍 GEMLIB STRUCTURE ANALYZER")
        print("=" * 60)
        
        if not self.load_csv():
            return False
        
        print(f"📂 Analyzing file: {self.csv_file}")
        print(f"📊 File size: {os.path.getsize(self.csv_file) / 1024:.1f} KB")
        print(f"📅 Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all analyses
        self.basic_info_analysis()
        self.column_analysis()
        self.data_type_analysis()
        self.content_analysis()
        self.pattern_analysis()
        self.gemstone_specific_analysis()
        self.integration_recommendations()
        
        # Generate reports
        self.generate_summary_report()
        self.generate_integration_code()
        
        return True
    
    def load_csv(self):
        """Load and validate CSV file"""
        try:
            if not os.path.exists(self.csv_file):
                print(f"❌ File not found: {self.csv_file}")
                return False
            
            # Try different encodings and separators
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            separators = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        self.df = pd.read_csv(self.csv_file, encoding=encoding, sep=sep)
                        if len(self.df.columns) > 1:  # Valid if more than 1 column
                            print(f"✅ Successfully loaded with encoding: {encoding}, separator: '{sep}'")
                            return True
                    except Exception:
                        continue
            
            # Last resort - try reading as text to see raw content
            with open(self.csv_file, 'r', encoding='utf-8', errors='replace') as f:
                sample_lines = f.readlines()[:5]
                print("❌ Failed to parse CSV. Sample content:")
                for i, line in enumerate(sample_lines, 1):
                    print(f"Line {i}: {line.strip()}")
            
            return False
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def basic_info_analysis(self):
        """Analyze basic file information"""
        print(f"\n📋 BASIC FILE INFORMATION")
        print("-" * 40)
        
        info = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024,  # KB
            'has_index': not isinstance(self.df.index, pd.RangeIndex) or self.df.index.name is not None
        }
        
        print(f"📊 Rows: {info['total_rows']:,}")
        print(f"📊 Columns: {info['total_columns']}")
        print(f"💾 Memory Usage: {info['memory_usage']:.1f} KB")
        print(f"🔢 Index Type: {type(self.df.index).__name__}")
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"🔄 Duplicate Rows: {duplicates}")
        
        # Check for completely empty rows
        empty_rows = self.df.isnull().all(axis=1).sum()
        print(f"🕳️  Empty Rows: {empty_rows}")
        
        self.analysis_results['basic_info'] = info
    
    def column_analysis(self):
        """Analyze column structure and names"""
        print(f"\n📊 COLUMN ANALYSIS")
        print("-" * 40)
        
        columns_info = {}
        
        print(f"📝 Column Names ({len(self.df.columns)} total):")
        for i, col in enumerate(self.df.columns, 1):
            col_str = str(col).strip()
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique_count = self.df[col].nunique()
            
            print(f"  {i:2d}. '{col_str}' ({len(col_str)} chars)")
            print(f"      Nulls: {null_count:,} ({null_pct:.1f}%)")
            print(f"      Unique: {unique_count:,}")
            
            columns_info[col_str] = {
                'position': i,
                'null_count': null_count,
                'null_percentage': null_pct,
                'unique_count': unique_count,
                'length': len(col_str)
            }
        
        # Look for potential key columns
        print(f"\n🔑 Potential Key Columns:")
        key_candidates = []
        for col in self.df.columns:
            col_str = str(col).strip().lower()
            unique_ratio = self.df[col].nunique() / len(self.df)
            
            # Check for key-like names
            key_indicators = ['id', 'ref', 'reference', 'number', 'code', 'key', 'gem_id', 'specimen']
            if any(indicator in col_str for indicator in key_indicators):
                key_candidates.append((col, unique_ratio, 'name_match'))
            elif unique_ratio > 0.8:  # High uniqueness
                key_candidates.append((col, unique_ratio, 'high_uniqueness'))
        
        for col, ratio, reason in key_candidates:
            print(f"  🔑 '{col}' - {ratio:.1%} unique ({reason})")
        
        self.analysis_results['columns'] = columns_info
        self.analysis_results['key_candidates'] = key_candidates
    
    def data_type_analysis(self):
        """Analyze data types and formats"""
        print(f"\n🔢 DATA TYPE ANALYSIS")
        print("-" * 40)
        
        type_info = {}
        
        for col in self.df.columns:
            col_str = str(col).strip()
            dtype = str(self.df[col].dtype)
            
            # Get sample values (non-null)
            non_null_values = self.df[col].dropna()
            samples = non_null_values.head(5).tolist() if len(non_null_values) > 0 else []
            
            # Analyze value patterns
            patterns = self.analyze_value_patterns(non_null_values)
            
            print(f"📊 '{col_str}':")
            print(f"    Type: {dtype}")
            print(f"    Samples: {samples}")
            print(f"    Patterns: {patterns}")
            
            type_info[col_str] = {
                'dtype': dtype,
                'samples': [str(s) for s in samples],
                'patterns': patterns
            }
        
        self.analysis_results['data_types'] = type_info
    
    def analyze_value_patterns(self, series):
        """Analyze patterns in values"""
        if len(series) == 0:
            return {'empty': True}
        
        patterns = {}
        
        # Convert to string for pattern analysis
        str_values = series.astype(str)
        
        # Length distribution
        lengths = str_values.str.len()
        patterns['min_length'] = int(lengths.min())
        patterns['max_length'] = int(lengths.max())
        patterns['avg_length'] = float(lengths.mean())
        
        # Common value analysis
        value_counts = str_values.value_counts()
        patterns['most_common'] = value_counts.head(3).to_dict()
        patterns['unique_values'] = len(value_counts)
        
        # Pattern detection
        patterns['has_numbers'] = str_values.str.contains(r'\d').any()
        patterns['has_letters'] = str_values.str.contains(r'[a-zA-Z]').any()
        patterns['has_spaces'] = str_values.str.contains(r'\s').any()
        patterns['has_special_chars'] = str_values.str.contains(r'[^a-zA-Z0-9\s]').any()
        
        # Format patterns
        if patterns['has_numbers'] and not patterns['has_letters']:
            patterns['format'] = 'numeric'
        elif patterns['has_letters'] and not patterns['has_numbers']:
            patterns['format'] = 'alphabetic'
        elif patterns['has_letters'] and patterns['has_numbers']:
            patterns['format'] = 'alphanumeric'
        else:
            patterns['format'] = 'mixed'
        
        return patterns
    
    def content_analysis(self):
        """Analyze content for gemological relevance"""
        print(f"\n💎 GEMOLOGICAL CONTENT ANALYSIS")
        print("-" * 40)
        
        # Define gemological terms to look for
        gemological_terms = {
            'gem_types': ['diamond', 'ruby', 'sapphire', 'emerald', 'topaz', 'quartz', 'garnet', 
                         'peridot', 'amethyst', 'citrine', 'tourmaline', 'beryl', 'corundum'],
            'origins': ['natural', 'synthetic', 'lab', 'created', 'grown', 'cultured'],
            'treatments': ['heat', 'irradiation', 'oiling', 'filling', 'coating', 'diffusion', 'hpht'],
            'locations': ['myanmar', 'thailand', 'sri lanka', 'madagascar', 'brazil', 'africa', 
                         'colombia', 'kashmir', 'burma', 'ceylon', 'botswana'],
            'properties': ['color', 'clarity', 'cut', 'carat', 'hardness', 'refractive', 'specific gravity']
        }
        
        content_analysis = {}
        
        for col in self.df.columns:
            col_str = str(col).strip()
            
            # Combine all text in column for analysis
            text_content = ' '.join(self.df[col].astype(str).str.lower())
            
            # Count gemological terms
            term_counts = {}
            for category, terms in gemological_terms.items():
                found_terms = []
                for term in terms:
                    if term in text_content:
                        found_terms.append(term)
                term_counts[category] = found_terms
            
            # Check if column looks gemologically relevant
            total_terms = sum(len(terms) for terms in term_counts.values())
            
            if total_terms > 0:
                print(f"💎 '{col_str}' - {total_terms} gemological terms found:")
                for category, terms in term_counts.items():
                    if terms:
                        print(f"    {category}: {', '.join(terms)}")
                
                content_analysis[col_str] = term_counts
        
        self.analysis_results['gemological_content'] = content_analysis
    
    def pattern_analysis(self):
        """Analyze data patterns and relationships"""
        print(f"\n🔍 PATTERN ANALYSIS")
        print("-" * 40)
        
        # Analyze relationships between columns
        print("🔗 Column Relationships:")
        
        # Look for columns that might be related
        for i, col1 in enumerate(self.df.columns):
            for j, col2 in enumerate(self.df.columns[i+1:], i+1):
                try:
                    # Check correlation for numeric columns
                    if self.df[col1].dtype in ['int64', 'float64'] and self.df[col2].dtype in ['int64', 'float64']:
                        corr = self.df[col1].corr(self.df[col2])
                        if abs(corr) > 0.5:
                            print(f"    📊 '{col1}' ↔ '{col2}': correlation = {corr:.3f}")
                    
                    # Check for text pattern relationships
                    elif self.df[col1].dtype == 'object' and self.df[col2].dtype == 'object':
                        # Check if one column contains parts of another
                        sample_1 = str(self.df[col1].iloc[0]) if len(self.df) > 0 else ""
                        sample_2 = str(self.df[col2].iloc[0]) if len(self.df) > 0 else ""
                        
                        if len(sample_1) > 0 and len(sample_2) > 0:
                            if sample_1.lower() in sample_2.lower() or sample_2.lower() in sample_1.lower():
                                print(f"    🔤 '{col1}' ↔ '{col2}': potential text relationship")
                
                except Exception:
                    continue
        
        # Analyze value distributions
        print(f"\n📈 Value Distributions:")
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                value_counts = self.df[col].value_counts()
                if len(value_counts) <= 20:  # Show distribution for categorical data
                    print(f"    📊 '{col}' distribution:")
                    for value, count in value_counts.head(5).items():
                        pct = (count / len(self.df)) * 100
                        print(f"        '{value}': {count} ({pct:.1f}%)")
    
    def gemstone_specific_analysis(self):
        """Analyze specifically for gemstone database integration"""
        print(f"\n💎 GEMSTONE DATABASE INTEGRATION ANALYSIS")
        print("-" * 40)
        
        integration_analysis = {}
        
        # Look for ID-like columns that could match gem numbers
        print("🔍 Potential Gem ID Columns:")
        for col in self.df.columns:
            col_str = str(col).strip()
            
            # Check if column contains numeric-like IDs
            sample_values = self.df[col].dropna().head(10)
            numeric_pattern = True
            
            for val in sample_values:
                val_str = str(val).strip()
                if not (val_str.isdigit() or (val_str.replace('.', '').isdigit())):
                    numeric_pattern = False
                    break
            
            if numeric_pattern and len(sample_values) > 0:
                print(f"    🔢 '{col_str}': {list(sample_values)}")
                integration_analysis[col_str] = {
                    'type': 'potential_gem_id',
                    'samples': [str(v) for v in sample_values]
                }
        
        # Look for description-building columns
        print(f"\n📝 Potential Description Columns:")
        description_candidates = []
        
        for col in self.df.columns:
            col_str = str(col).strip().lower()
            
            # Check for gemological property indicators
            property_indicators = {
                'natural_synthetic': ['nat', 'syn', 'natural', 'synthetic', 'origin', 'type'],
                'species': ['spec', 'species', 'mineral', 'gem'],
                'variety': ['var', 'variety', 'color', 'type'],
                'treatment': ['treat', 'treatment', 'enhanced', 'heated'],
                'location': ['origin', 'location', 'country', 'mine', 'locality']
            }
            
            for prop_type, indicators in property_indicators.items():
                if any(indicator in col_str for indicator in indicators):
                    sample_vals = self.df[col].dropna().head(5).tolist()
                    print(f"    💎 '{col}' ({prop_type}): {sample_vals}")
                    description_candidates.append((col, prop_type, sample_vals))
                    break
        
        integration_analysis['description_candidates'] = description_candidates
        self.analysis_results['integration'] = integration_analysis
    
    def integration_recommendations(self):
        """Generate integration recommendations"""
        print(f"\n🚀 INTEGRATION RECOMMENDATIONS")
        print("-" * 40)
        
        # Recommend ID column
        id_recommendations = []
        for col in self.df.columns:
            col_str = str(col).strip().lower()
            if any(keyword in col_str for keyword in ['id', 'ref', 'number', 'code']):
                unique_ratio = self.df[col].nunique() / len(self.df)
                id_recommendations.append((col, unique_ratio))
        
        if id_recommendations:
            best_id = max(id_recommendations, key=lambda x: x[1])
            print(f"🔑 Recommended ID Column: '{best_id[0]}' ({best_id[1]:.1%} unique)")
        else:
            print("⚠️  No obvious ID column found")
        
        # Recommend description columns
        desc_columns = []
        for col in self.df.columns:
            col_str = str(col).strip()
            # Skip likely ID columns
            if not any(keyword in col_str.lower() for keyword in ['id', 'ref', 'number']):
                if self.df[col].dtype == 'object':  # Text columns
                    desc_columns.append(col)
        
        if desc_columns:
            print(f"📝 Recommended Description Columns:")
            for col in desc_columns:
                sample = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else "N/A"
                print(f"    - '{col}': {sample}")
        
        # Generate integration code template
        self.generate_integration_template()
    
    def generate_integration_template(self):
        """Generate code template for integration"""
        print(f"\n💻 INTEGRATION CODE TEMPLATE:")
        print("-" * 40)
        
        # Find best ID column
        id_col = "Reference"  # Default assumption
        for col in self.df.columns:
            if any(keyword in str(col).lower() for keyword in ['ref', 'id', 'number']):
                id_col = col
                break
        
        # Find description columns (non-ID text columns)
        desc_cols = []
        for col in self.df.columns:
            if (self.df[col].dtype == 'object' and 
                col != id_col and 
                not any(keyword in str(col).lower() for keyword in ['id', 'ref', 'number'])):
                desc_cols.append(col)
        
        # Generate code
        code_template = f'''
# Integration code for gemlib_structural_ready.csv
def load_gem_library():
    try:
        gemlib = pd.read_csv('gemlib_structural_ready.csv')
        gemlib.columns = gemlib.columns.str.strip()
        
        # Use '{id_col}' as the reference ID column
        if '{id_col}' in gemlib.columns:
            gemlib['{id_col}'] = gemlib['{id_col}'].astype(str).str.strip()
            
            # Build descriptions from available columns
            description_columns = {desc_cols}
            
            # Create rich descriptions
            gemlib['Description'] = gemlib[description_columns].apply(
                lambda x: ' | '.join([str(v).strip() for v in x 
                                    if pd.notnull(v) and str(v).strip()]), axis=1)
            
            # Create mapping dictionary
            gem_descriptions = dict(zip(gemlib['{id_col}'], gemlib['Description']))
            
            print(f"✅ Loaded {{len(gem_descriptions)}} gem descriptions")
            return gem_descriptions
        else:
            print("⚠️ ID column '{id_col}' not found")
            return {{}}
            
    except Exception as e:
        print(f"⚠️ Error loading gemlib: {{e}}")
        return {{}}
'''
        
        print(code_template)
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(f"\n📋 COMPREHENSIVE SUMMARY REPORT")
        print("=" * 60)
        
        # File overview
        print(f"📂 File: {self.csv_file}")
        print(f"📊 Dimensions: {len(self.df)} rows × {len(self.df.columns)} columns")
        print(f"💾 Size: {os.path.getsize(self.csv_file) / 1024:.1f} KB")
        
        # Column summary
        print(f"\n📝 Columns Summary:")
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            nulls = self.df[col].isnull().sum()
            unique = self.df[col].nunique()
            print(f"  {i:2d}. '{col}' ({dtype}) - {nulls} nulls, {unique} unique")
        
        # Data quality
        total_cells = len(self.df) * len(self.df.columns)
        null_cells = self.df.isnull().sum().sum()
        quality_score = ((total_cells - null_cells) / total_cells) * 100
        
        print(f"\n📊 Data Quality Score: {quality_score:.1f}%")
        print(f"   Total cells: {total_cells:,}")
        print(f"   Null cells: {null_cells:,}")
        print(f"   Complete cells: {total_cells - null_cells:,}")
        
        # Save detailed analysis to file
        self.save_analysis_report()
    
    def save_analysis_report(self):
        """Save detailed analysis to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"gemlib_analysis_{timestamp}.json"
        
        try:
            # Prepare analysis results for JSON serialization
            json_results = {}
            for key, value in self.analysis_results.items():
                json_results[key] = self.make_json_serializable(value)
            
            # Add metadata
            json_results['metadata'] = {
                'analysis_time': datetime.now().isoformat(),
                'file_analyzed': self.csv_file,
                'file_size_kb': os.path.getsize(self.csv_file) / 1024,
                'analyzer_version': '1.0'
            }
            
            with open(report_file, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"\n💾 Detailed analysis saved to: {report_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving analysis report: {e}")
    
    def make_json_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self.make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    def generate_integration_code(self):
        """Generate complete integration code"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_file = f"gemlib_integration_{timestamp}.py"
        
        # Analyze structure to generate optimal code
        id_col = "Reference"  # Default
        desc_cols = []
        
        # Find best ID column
        for col in self.df.columns:
            if any(keyword in str(col).lower() for keyword in ['ref', 'id', 'number']):
                id_col = col
                break
        
        # Find description columns
        for col in self.df.columns:
            if (str(col) != id_col and 
                self.df[col].dtype == 'object' and
                not any(keyword in str(col).lower() for keyword in ['id', 'ref', 'number'])):
                desc_cols.append(str(col))
        
        integration_code = f'''#!/usr/bin/env python3
"""
gemlib_integration.py - GENERATED GEMLIB INTEGRATION CODE
Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Based on analysis of {self.csv_file}
"""

import pandas as pd
import numpy as np

class GemlibIntegration:
    def __init__(self, csv_file='gemlib_structural_ready.csv'):
        self.csv_file = csv_file
        self.gem_descriptions = {{}}
        self.gem_details = {{}}
        self.load_gem_library()
    
    def load_gem_library(self):
        """Load gem descriptions from CSV file"""
        try:
            gemlib = pd.read_csv(self.csv_file)
            gemlib.columns = gemlib.columns.str.strip()
            
            print(f"📚 Loading gem library from {{self.csv_file}}")
            print(f"📊 Found {{len(gemlib)}} entries with {{len(gemlib.columns)}} columns")
            
            # Columns found in your file:
            # {list(self.df.columns)}
            
            if '{id_col}' in gemlib.columns:
                gemlib['{id_col}'] = gemlib['{id_col}'].astype(str).str.strip()
                
                # Description columns found:
                description_columns = {desc_cols}
                
                if all(col in gemlib.columns for col in description_columns):
                    # Create rich descriptions
                    gemlib['Rich_Description'] = gemlib[description_columns].apply(
                        lambda x: ' | '.join([str(v).strip() for v in x 
                                            if pd.notnull(v) and str(v).strip()]), axis=1)
                    
                    # Create mapping dictionaries
                    self.gem_descriptions = dict(zip(gemlib['{id_col}'], gemlib['Rich_Description']))
                    
                    # Store detailed information for each gem
                    for _, row in gemlib.iterrows():
                        ref = str(row['{id_col}'])
                        self.gem_details[ref] = {{}}
                        for col in description_columns:
                            self.gem_details[ref][col.lower().replace('.', '_').replace('/', '_')] = str(row.get(col, '')).strip()
                    
                    print(f"✅ Successfully loaded {{len(self.gem_descriptions)}} gem descriptions")
                    print(f"📝 Available properties: {{description_columns}}")
                    
                else:
                    missing_cols = [col for col in description_columns if col not in gemlib.columns]
                    print(f"⚠️ Missing expected columns: {{missing_cols}}")
                    print(f"📝 Available columns: {{list(gemlib.columns)}}")
                    
                    # Fallback: use available text columns
                    available_desc_cols = [col for col in gemlib.columns 
                                         if col != '{id_col}' and gemlib[col].dtype == 'object']
                    
                    if available_desc_cols:
                        gemlib['Description'] = gemlib[available_desc_cols].apply(
                            lambda x: ' | '.join([str(v).strip() for v in x 
                                                if pd.notnull(v) and str(v).strip()]), axis=1)
                        self.gem_descriptions = dict(zip(gemlib['{id_col}'], gemlib['Description']))
                        print(f"✅ Using fallback columns: {{available_desc_cols}}")
                        print(f"✅ Loaded {{len(self.gem_descriptions)}} basic descriptions")
            else:
                print(f"❌ ID column '{id_col}' not found in file")
                print(f"📝 Available columns: {{list(gemlib.columns)}}")
                
        except FileNotFoundError:
            print(f"❌ File not found: {{self.csv_file}}")
        except Exception as e:
            print(f"❌ Error loading gem library: {{e}}")
            import traceback
            traceback.print_exc()
    
    def get_gem_description(self, gem_id):
        """Get rich description for gem ID"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        return self.gem_descriptions.get(base_id, f"Gem {{base_id}}")
    
    def get_gem_details(self, gem_id):
        """Get detailed properties for gem ID"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        return self.gem_details.get(base_id, {{}})
    
    def search_gems(self, search_term):
        """Search gems by description"""
        search_term = search_term.lower()
        matches = []
        
        for gem_id, description in self.gem_descriptions.items():
            if search_term in description.lower():
                matches.append((gem_id, description))
        
        return matches
    
    def get_statistics(self):
        """Get library statistics"""
        return {{
            'total_gems': len(self.gem_descriptions),
            'total_details': len(self.gem_details),
            'sample_descriptions': list(self.gem_descriptions.items())[:5]
        }}

def main():
    """Test the integration"""
    integrator = GemlibIntegration()
    
    print("\\n🧪 TESTING INTEGRATION")
    print("-" * 40)
    
    stats = integrator.get_statistics()
    print(f"📊 Statistics: {{stats}}")
    
    # Test sample lookups
    if stats['total_gems'] > 0:
        sample_ids = list(integrator.gem_descriptions.keys())[:3]
        print(f"\\n🔍 Sample Lookups:")
        for gem_id in sample_ids:
            description = integrator.get_gem_description(gem_id)
            details = integrator.get_gem_details(gem_id)
            print(f"  {{gem_id}}: {{description}}")
            print(f"    Details: {{details}}")

if __name__ == "__main__":
    main()
'''
        
        try:
            with open(code_file, 'w') as f:
                f.write(integration_code)
            print(f"\n💻 Integration code generated: {code_file}")
        except Exception as e:
            print(f"⚠️ Error generating integration code: {e}")

def main():
    """Main function"""
    print("🔍 GEMLIB STRUCTURE ANALYZER")
    print("="*50)
    
    # Check for file
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'gemlib' in f.lower()]
    
    if csv_files:
        print(f"📂 Found potential gemlib files: {csv_files}")
        
        if 'gemlib_structural_ready.csv' in csv_files:
            file_to_analyze = 'gemlib_structural_ready.csv'
        else:
            file_to_analyze = csv_files[0]
            print(f"⚠️ Using: {file_to_analyze}")
    else:
        file_to_analyze = input("📂 Enter CSV filename to analyze: ").strip()
        if not file_to_analyze:
            file_to_analyze = 'gemlib_structural_ready.csv'
    
    # Run analysis
    analyzer = GemlibStructureAnalyzer(file_to_analyze)
    success = analyzer.analyze_complete_structure()
    
    if success:
        print(f"\n✅ Analysis complete!")
        print(f"📋 Check generated files for detailed results and integration code")
    else:
        print(f"\n❌ Analysis failed - check file path and format")

if __name__ == "__main__":
    main()