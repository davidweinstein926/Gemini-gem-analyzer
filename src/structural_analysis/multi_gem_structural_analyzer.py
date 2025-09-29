#!/usr/bin/env python3
"""
CLEAN MULTI-GEM STRUCTURAL ANALYZER
Simple, robust filename parsing - first B/L/U from left = light source
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os, sys, sqlite3, shutil, json, math, traceback, re
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

class MultiGemStructuralAnalyzer:
    """Clean analyzer with simple filename parsing"""
    
    def __init__(self, mode="gui", input_source="archive"):
        self.mode = mode
        self.input_source = input_source
        
        if self.mode == "gui":
            self.root = tk.Tk()
            self.root.title(f"Multi-Gem Structural Analyzer - {input_source.title()}")
            self.root.geometry("1200x800")
        else:
            self.root = None
            print(f"🔬 Analyzer initialized: {mode} mode, {input_source} source")
        
        self.project_root = self.find_project_root()
        self.setup_paths()
        self.gem_name_map = self.load_gem_library()
        self.gem_groups = {}
        self.selected_gems = {}
        self.database_schema = None
        self.perfect_match_threshold = 0.005
        self.light_weights = {'B': 1.0, 'L': 0.9, 'U': 0.8}
        
        if self.mode == "gui":
            self.setup_gui()
        
        self.check_databases()
        self.scan_input_directory()
    
    def find_project_root(self):
        """Find project root directory"""
        current_path = Path(__file__).parent
        for level in range(5):
            if sum(1 for indicator in ['database', 'data', 'src'] if (current_path / indicator).exists()) >= 2:
                return current_path
            parent = current_path.parent
            if parent == current_path:
                break
            current_path = parent
        return Path(__file__).parent.parent.parent
    
    def setup_paths(self):
        """Setup directory paths"""
        if self.input_source == "archive":
            self.input_path = self.project_root / "data" / "structural(archive)"
            self.analysis_name = "Archive Analysis (Option 8)"
        else:
            self.input_path = self.project_root / "data" / "structural_data"
            self.analysis_name = "Current Work Analysis (Option 4)"
        
        self.sqlite_db_path = self.project_root / "database" / "structural_spectra" / "gemini_structural.db"
        self.csv_db_path = self.project_root / "database" / "structural_spectra" / "gemini_structural_unified.csv"
    
    def load_gem_library(self):
        """Load gem descriptions"""
        gem_map = {}
        for path in [self.project_root / "gemlib_structural_ready.csv",
                     self.project_root / "database" / "gemlib_structural_ready.csv"]:
            try:
                if path.exists():
                    df = pd.read_csv(path)
                    if 'Reference' in df.columns:
                        expected_cols = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                        if all(col in df.columns for col in expected_cols):
                            df['Description'] = df[expected_cols].apply(
                                lambda x: ' '.join([str(v) if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                            gem_map = dict(zip(df['Reference'].astype(str), df['Description']))
                            print(f"✅ Loaded {len(gem_map)} gem descriptions")
                            return gem_map
            except:
                continue
        return gem_map
    
    def check_databases(self):
        """Check and load database schema"""
        if self.sqlite_db_path.exists():
            self.database_type, self.database_path = "sqlite", self.sqlite_db_path
        elif self.csv_db_path.exists():
            self.database_type, self.database_path = "csv", self.csv_db_path
        else:
            if self.mode == "gui":
                messagebox.showerror("Error", "No structural database found!")
            else:
                print("❌ No structural database found!")
            return False
        
        return self.detect_database_schema()
    
    def detect_database_schema(self):
        """Detect database schema"""
        try:
            if self.database_type == "sqlite":
                with sqlite3.connect(self.database_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    if not tables:
                        return False
                    
                    table_name = tables[0][0]
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
            else:
                df = pd.read_csv(self.database_path, nrows=1)
                columns = df.columns.tolist()
                table_name = None
            
            self.database_schema = {
                'table_name': table_name,
                'file_column': self.find_column(columns, ['file', 'filename', 'gem_id']),
                'wavelength_column': self.find_column(columns, ['wavelength', 'Wavelength']),
                'intensity_column': self.find_column(columns, ['intensity', 'Intensity']),
                'light_column': self.find_column(columns, ['light_source', 'light']),
                'time_series_column': self.find_column(columns, ['time_series', 'ts'])
            }
            
            if not self.database_schema['file_column']:
                print("❌ No file column found in database")
                return False
            
            print(f"✅ Database schema detected: {self.database_type}")
            return True
            
        except Exception as e:
            print(f"❌ Schema detection failed: {e}")
            return False
    
    def find_column(self, columns, candidates):
        """Find matching column name"""
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None
    
    def scan_input_directory(self):
        """Scan input directory for gem files"""
        if not self.input_path.exists():
            if self.mode == "gui":
                messagebox.showerror("Error", f"Directory not found: {self.input_path}")
            return
        
        csv_files = list(self.input_path.glob("*.csv"))
        if not csv_files:
            if self.mode == "gui":
                messagebox.showwarning("Warning", f"No CSV files in {self.input_path}")
            return
        
        self.gem_groups.clear()
        
        for file_path in csv_files:
            gem_info = self.parse_filename(file_path.name)
            base_id = gem_info['base_id']
            
            if base_id not in self.gem_groups:
                self.gem_groups[base_id] = {'files': {'B': [], 'L': [], 'U': []}, 'paths': {'B': [], 'L': [], 'U': []}}
            
            light_source = gem_info['light_source']
            if light_source in ['B', 'L', 'U']:
                self.gem_groups[base_id]['files'][light_source].append(gem_info)
                self.gem_groups[base_id]['paths'][light_source].append(file_path)
        
        if self.mode == "gui":
            self.populate_gem_list()
        else:
            complete = sum(1 for data in self.gem_groups.values() 
                          if all(len(data['files'][ls]) > 0 for ls in ['B', 'L', 'U']))
            print(f"📁 Found {len(self.gem_groups)} gems ({complete} complete)")
    
    def parse_filename(self, filename):
        """SIMPLE: Find first B/L/U from left = light source"""
        stem = Path(filename).stem
        
        print(f"🔍 DEBUG: Parsing '{filename}' → stem: '{stem}'")
        
        # Find first B, L, or U from left
        light_source = None
        light_position = -1
        
        for i, char in enumerate(stem):
            if char in ['B', 'L', 'U']:
                light_source = char
                light_position = i
                print(f"✅ Found light '{light_source}' at position {i}")
                break
        
        if light_source:
            # Base name = everything before light source
            if light_position > 0:
                base_name = stem[:light_position]
            else:
                base_name = "unk"  # Default if light is at start
            
            result = {
                'base_id': base_name,
                'light_source': light_source,
                'filename': filename,
                'date': None
            }
            print(f"✅ Parsed: {result}")
            return result
        
        # No B/L/U found
        print(f"❌ No B/L/U found in '{stem}'")
        return {
            'base_id': stem,
            'light_source': 'Unknown',
            'filename': filename,
            'date': None
        }
    
    def setup_gui(self):
        """Setup GUI interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text=self.analysis_name, font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        left_frame = ttk.LabelFrame(content_frame, text="Available Gems", padding="5")
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.gem_listbox = tk.Listbox(left_frame, height=15)
        self.gem_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        ttk.Button(left_frame, text="Select Files", command=self.select_gem_files).pack(side='left', padx=(0, 5))
        ttk.Button(left_frame, text="Select All", command=self.select_all_complete).pack(side='left')
        
        right_frame = ttk.LabelFrame(content_frame, text="Selected for Analysis", padding="5")
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.selected_listbox = tk.Listbox(right_frame, height=15)
        self.selected_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        ttk.Button(right_frame, text="Remove", command=self.remove_selected).pack()
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)
        
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side='left')
        ttk.Button(control_frame, text="Close", command=self.close_app).pack(side='right')
        ttk.Button(control_frame, text="🚀 Start Analysis", command=self.start_analysis).pack(side='right', padx=(5, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var).pack()
    
    def populate_gem_list(self):
        """Populate gem list"""
        self.gem_listbox.delete(0, tk.END)
        for base_id, data in sorted(self.gem_groups.items()):
            counts = [len(data['files'][ls]) for ls in ['B', 'L', 'U']]
            status = "🟢" if all(c > 0 for c in counts) else "🟡"
            sources = '+'.join([f"{ls}({c})" for ls, c in zip(['B', 'L', 'U'], counts) if c > 0])
            self.gem_listbox.insert(tk.END, f"{status} Gem {base_id} - {sources}")
    
    def select_gem_files(self):
        """Select gem files"""
        selection = self.gem_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a gem.")
            return
        
        gem_ids = sorted(self.gem_groups.keys())
        base_id = gem_ids[selection[0]]
        
        if base_id in self.selected_gems:
            messagebox.showinfo("Already Selected", f"Gem {base_id} already selected.")
            return
        
        gem_data = self.gem_groups[base_id]
        selected = {'files': {}, 'paths': {}}
        
        for ls in ['B', 'L', 'U']:
            if gem_data['files'][ls]:
                selected['files'][ls] = gem_data['files'][ls][0]
                selected['paths'][ls] = gem_data['paths'][ls][0]
        
        self.selected_gems[base_id] = selected
        self.update_selected_display()
    
    def select_all_complete(self):
        """Select all complete gems"""
        for base_id, data in self.gem_groups.items():
            if all(len(data['files'][ls]) > 0 for ls in ['B', 'L', 'U']) and base_id not in self.selected_gems:
                selected = {'files': {}, 'paths': {}}
                for ls in ['B', 'L', 'U']:
                    selected['files'][ls] = data['files'][ls][0]
                    selected['paths'][ls] = data['paths'][ls][0]
                self.selected_gems[base_id] = selected
        
        self.update_selected_display()
        messagebox.showinfo("Complete", f"Selected {len(self.selected_gems)} gems.")
    
    def update_selected_display(self):
        """Update selected gems display"""
        self.selected_listbox.delete(0, tk.END)
        for base_id, data in self.selected_gems.items():
            sources = '+'.join(sorted(data['files'].keys()))
            self.selected_listbox.insert(tk.END, f"Gem {base_id} ({sources})")
        self.status_var.set(f"Selected {len(self.selected_gems)} gems")
    
    def remove_selected(self):
        """Remove selected gem"""
        selection = self.selected_listbox.curselection()
        if selection:
            gem_ids = list(self.selected_gems.keys())
            if selection[0] < len(gem_ids):
                del self.selected_gems[gem_ids[selection[0]]]
                self.update_selected_display()
    
    def clear_all(self):
        """Clear all selections"""
        self.selected_gems.clear()
        self.update_selected_display()
    
    def close_app(self):
        """Close application"""
        if messagebox.askyesno("Close", "Close application?"):
            self.root.quit()
            self.root.destroy()
    
    def start_analysis(self):
        """Start analysis"""
        if not self.selected_gems:
            messagebox.showwarning("No Selection", "Select at least one gem.")
            return
        
        if not self.database_schema:
            messagebox.showerror("Error", "Database not available.")
            return
        
        count = len(self.selected_gems)
        if messagebox.askyesno("Start Analysis", f"Analyze {count} gems?\n\n✅ Automatic valid gem detection\n✅ Perfect self-match capability"):
            self.run_analysis()
    
    def detect_goi_light_sources(self):
        """Detect GOI light sources"""
        goi_light_sources = set()
        for gem_data in self.selected_gems.values():
            goi_light_sources.update(gem_data['paths'].keys())
        
        print(f"🔍 GOI light sources: {sorted(goi_light_sources)}")
        return goi_light_sources
    
    def find_valid_database_gems(self, db_df, goi_light_sources):
        """Find valid database gems"""
        file_col = self.database_schema['file_column']
        light_col = self.database_schema.get('light_column')
        
        if not light_col or light_col not in db_df.columns:
            print("⚠️ No light column - using all gems")
            return db_df, db_df[file_col].unique().tolist()
        
        print(f"🔍 Finding gems with: {sorted(goi_light_sources)}")
        
        # Group by base gem ID
        base_gem_analysis = {}
        
        for gem_id, gem_data in db_df.groupby(file_col):
            base_id = self.extract_base_id(str(gem_id))
            if not base_id:
                continue
            
            # Get light sources for this gem
            light_sources = set()
            for light_value in gem_data[light_col].unique():
                light_str = str(light_value).strip()
                if light_str in ['B', 'Halogen', 'halogen']:
                    light_sources.add('B')
                elif light_str in ['L', 'Laser', 'laser']:
                    light_sources.add('L')
                elif light_str in ['U', 'UV', 'uv']:
                    light_sources.add('U')
            
            if base_id not in base_gem_analysis:
                base_gem_analysis[base_id] = {'light_sources': set(), 'gem_ids': []}
            
            base_gem_analysis[base_id]['light_sources'].update(light_sources)
            base_gem_analysis[base_id]['gem_ids'].append(gem_id)
        
        # Find valid gems
        valid_gem_ids = []
        valid_base_gems = []
        
        for base_id, analysis in base_gem_analysis.items():
            if analysis['light_sources'] == goi_light_sources:
                valid_base_gems.append(base_id)
                valid_gem_ids.extend(analysis['gem_ids'])
                print(f"✅ Valid gem {base_id}: {sorted(analysis['light_sources'])}")
            else:
                missing = goi_light_sources - analysis['light_sources']
                if missing:
                    print(f"❌ Invalid gem {base_id}: missing {sorted(missing)}")
        
        filtered_db = db_df[db_df[file_col].isin(valid_gem_ids)]
        print(f"📊 Valid gems: {sorted(valid_base_gems)} ({len(filtered_db)} records)")
        
        return filtered_db, valid_gem_ids
    
    def extract_base_id(self, filename):
        """Extract base ID from filename"""
        stem = re.sub(r'\.csv$', '', filename)
        stem = re.sub(r'_(structural|halogen|laser|uv|TS\d+|\d{8}_\d{6}).*$', '', stem)
        
        # Simple approach: find first B/L/U and take everything before it
        for i, char in enumerate(stem):
            if char in ['B', 'L', 'U']:
                if i > 0:
                    return stem[:i]
                else:
                    return "unk"
        
        # Fallback: look for digits
        match = re.search(r'^[A-Za-z]*(\d+)', stem)
        if match:
            return match.group(1)
        
        return stem
    
    def is_self_match(self, goi_id, db_gem_id):
        """Check if this is a self-match"""
        goi_base = str(goi_id).strip()
        db_base = self.extract_base_id(str(db_gem_id))
        return goi_base == db_base
    
    def run_analysis(self):
        """Run analysis"""
        print(f"\n🚀 Starting analysis of {len(self.selected_gems)} gems...")
        
        results_dir = self.project_root / "outputs" / "structural_results"
        reports_dir = results_dir / "reports"
        graphs_dir = results_dir / "graphs"
        
        for dir_path in [results_dir, reports_dir, graphs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load database
        try:
            if self.database_type == "sqlite":
                conn = sqlite3.connect(self.database_path)
                query = f"SELECT * FROM {self.database_schema['table_name']}"
                db_df = pd.read_sql_query(query, conn)
                conn.close()
            else:
                db_df = pd.read_csv(self.database_path)
            
            print(f"✅ Database loaded: {len(db_df):,} records")
        except Exception as e:
            print(f"❌ Database load failed: {e}")
            if self.mode == "gui":
                messagebox.showerror("Error", f"Database load failed: {e}")
            return False
        
        # Find valid gems
        goi_light_sources = self.detect_goi_light_sources()
        filtered_db, valid_gem_ids = self.find_valid_database_gems(db_df, goi_light_sources)
        
        if not valid_gem_ids:
            print("❌ No valid database gems found!")
            return False
        
        # Analyze each gem
        all_results = []
        
        for i, (base_id, data) in enumerate(self.selected_gems.items()):
            print(f"\n🎯 Analyzing Gem {base_id} ({i+1}/{len(self.selected_gems)})...")
            
            if self.mode == "gui":
                self.status_var.set(f"Analyzing Gem {base_id}...")
                self.root.update()
            
            gem_results = {
                'gem_id': base_id,
                'timestamp': timestamp,
                'goi_light_sources': sorted(goi_light_sources),
                'valid_database_gems': sorted([str(g) for g in valid_gem_ids]),
                'light_source_results': {},
                'best_match': None
            }
            
            # Analyze each light source
            for light_source, file_path in data['paths'].items():
                print(f"📄 Processing {light_source}: {file_path.name}")
                
                try:
                    unknown_df = pd.read_csv(file_path)
                    if unknown_df.empty:
                        continue
                    
                    # Find database matches
                    db_matches = self.find_database_matches(filtered_db, light_source)
                    if db_matches.empty:
                        continue
                    
                    # Calculate scores
                    scores = self.calculate_similarity_scores(unknown_df, db_matches, light_source, file_path)
                    
                    if scores:
                        best = min(scores, key=lambda x: x['score'])
                        print(f"✅ Best: {best['db_gem_id']} (score: {best['score']:.6f})")
                        
                        if self.is_self_match(base_id, best['db_gem_id']) and best['score'] < 0.01:
                            print("🎯 PERFECT SELF-MATCH!")
                        
                        gem_results['light_source_results'][light_source] = {
                            'file': file_path.name,
                            'best_match': best,
                            'top_5': sorted(scores, key=lambda x: x['score'])[:5]
                        }
                
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
            
            # Calculate combined scores
            if len(gem_results['light_source_results']) > 1:
                combined = self.calculate_combined_scores(gem_results['light_source_results'])
                if combined:
                    best_combined = min(combined, key=lambda x: x['score'])
                    gem_results['best_match'] = best_combined
                    print(f"🏆 Best combined: {best_combined['db_gem_id']} (score: {best_combined['score']:.4f})")
            
            all_results.append(gem_results)
        
        # Save results
        self.save_results(all_results, reports_dir, timestamp)
        
        if HAS_MATPLOTLIB:
            self.generate_plots(all_results, graphs_dir, timestamp)
        
        print(f"\n🎉 Analysis complete! Results in: {results_dir}")
        
        if self.mode == "gui":
            messagebox.showinfo("Complete", f"Analysis complete!\nResults saved to: {results_dir}")
            self.status_var.set("Analysis complete")
        
        return True
    
    def find_database_matches(self, filtered_db, light_source):
        """Find database matches for light source"""
        light_col = self.database_schema.get('light_column')
        
        if light_col and light_col in filtered_db.columns:
            light_mapping = {
                'B': ['B', 'Halogen', 'halogen'], 
                'L': ['L', 'Laser', 'laser'], 
                'U': ['U', 'UV', 'uv']
            }
            light_values = light_mapping.get(light_source, [light_source])
            matches = filtered_db[filtered_db[light_col].isin(light_values)]
        else:
            matches = filtered_db.copy()
        
        return matches
    
    def find_enhanced_database_matches(self, filtered_db, light_source):
        """Alias for find_database_matches - for compatibility"""
        return self.find_database_matches(filtered_db, light_source)
    
    def calculate_similarity_scores(self, unknown_df, db_matches, light_source, file_path):
        """Calculate similarity scores"""
        scores = []
        file_col = self.database_schema['file_column']
        wl_col = self.database_schema['wavelength_column']
        int_col = self.database_schema['intensity_column']
        
        # Detect data columns
        unknown_wl_col = unknown_int_col = None
        for col in unknown_df.columns:
            col_lower = col.lower()
            if 'wavelength' in col_lower or col_lower == 'wl':
                unknown_wl_col = col
            elif 'intensity' in col_lower or 'value' in col_lower:
                unknown_int_col = col
        
        if not unknown_wl_col or not unknown_int_col:
            return scores
        
        try:
            unknown_wl = unknown_df[unknown_wl_col].values
            unknown_int = unknown_df[unknown_int_col].values
            
            # Remove NaN
            valid = ~(np.isnan(unknown_wl) | np.isnan(unknown_int))
            unknown_wl = unknown_wl[valid]
            unknown_int = unknown_int[valid]
            
            if len(unknown_wl) < 3:
                return scores
        except:
            return scores
        
        goi_base_id = self.extract_base_id(file_path.name)
        
        # Score each database gem
        for gem_id, gem_data in db_matches.groupby(file_col):
            try:
                db_base_id = self.extract_base_id(str(gem_id))
                
                db_wl = gem_data[wl_col].values
                db_int = gem_data[int_col].values
                
                # Remove NaN
                db_valid = ~(np.isnan(db_wl) | np.isnan(db_int))
                db_wl = db_wl[db_valid]
                db_int = db_int[db_valid]
                
                if len(db_wl) < 1:
                    continue
                
                # Check for self-match
                if goi_base_id and db_base_id and goi_base_id == db_base_id:
                    score = self.compute_perfect_match_score(unknown_wl, unknown_int, db_wl, db_int)
                    if score is not None:
                        print(f"🎯 SELF-MATCH: {file_path.name} ↔ {gem_id} (score: {score:.6f})")
                        scores.append({'db_gem_id': gem_id, 'score': score, 'is_perfect': True})
                        continue
                
                # Normal similarity
                score = self.compute_similarity(unknown_wl, unknown_int, db_wl, db_int)
                if score is not None and not math.isnan(score):
                    scores.append({'db_gem_id': gem_id, 'score': score, 'is_perfect': False})
            
            except Exception:
                continue
        
        return scores
    
    def compute_perfect_match_score(self, unknown_wl, unknown_int, db_wl, db_int):
        """Compute perfect match score"""
        try:
            size_ratio = min(len(unknown_wl), len(db_wl)) / max(len(unknown_wl), len(db_wl))
            if size_ratio < 0.7:
                return None
            
            # Sample comparison
            sample_size = min(10, len(unknown_wl), len(db_wl))
            unknown_idx = np.linspace(0, len(unknown_wl)-1, sample_size, dtype=int)
            db_idx = np.linspace(0, len(db_wl)-1, sample_size, dtype=int)
            
            total_diff = 0.0
            for u_i, d_i in zip(unknown_idx, db_idx):
                wl_diff = abs(unknown_wl[u_i] - db_wl[d_i])
                max_int = max(unknown_int[u_i], db_int[d_i])
                int_diff = abs(unknown_int[u_i] - db_int[d_i]) / max_int if max_int > 0 else 0
                total_diff += (wl_diff * 0.01 + int_diff)
            
            avg_diff = total_diff / sample_size
            
            if avg_diff < 0.4 and size_ratio > 0.8:
                return 0.001 + avg_diff * 0.01
            
            return None
        except:
            return None
    
    def compute_similarity(self, unknown_wl, unknown_int, db_wl, db_int):
        """Compute similarity score"""
        try:
            total_score = 0.0
            matched = 0
            tolerance = 10.0
            
            unique_wl = np.unique(unknown_wl)
            
            for u_wl in unique_wl:
                distances = np.abs(db_wl - u_wl)
                nearby = distances <= tolerance
                
                if np.any(nearby):
                    u_int = unknown_int[np.abs(unknown_wl - u_wl) < 0.1]
                    d_int = db_int[nearby]
                    
                    if len(u_int) > 0 and len(d_int) > 0:
                        u_avg = np.mean(u_int)
                        d_avg = np.mean(d_int)
                        
                        if max(u_avg, d_avg) > 0:
                            diff = abs(u_avg - d_avg) / max(u_avg, d_avg)
                            weight = 1.0 - (np.min(distances[nearby]) / tolerance)
                            total_score += diff * weight
                            matched += 1
                else:
                    total_score += 2.0
            
            if len(unique_wl) > 0:
                avg_score = total_score / len(unique_wl)
                match_rate = matched / len(unique_wl)
                return avg_score * (1.2 - match_rate * 0.2)
            
            return None
        except:
            return None
    
    def calculate_combined_scores(self, light_source_results):
        """Calculate combined scores"""
        gem_combinations = {}
        
        for ls, ls_data in light_source_results.items():
            for match in ls_data['top_5']:
                gem_id = match['db_gem_id']
                
                if gem_id not in gem_combinations:
                    gem_combinations[gem_id] = {}
                
                gem_combinations[gem_id][ls] = {
                    'score': match['score'],
                    'percentage': self.score_to_percentage(match['score']),
                    'is_perfect': match.get('is_perfect', False)
                }
        
        valid_combinations = []
        
        for gem_id, light_data in gem_combinations.items():
            light_scores = {}
            perfect_count = 0
            
            for ls, ls_info in light_data.items():
                light_scores[ls] = ls_info['percentage']
                if ls_info['is_perfect']:
                    perfect_count += 1
            
            if light_scores:
                # Weighted average
                weighted_sum = sum(score * self.light_weights.get(ls, 1.0) for ls, score in light_scores.items())
                total_weight = sum(self.light_weights.get(ls, 1.0) for ls in light_scores.keys())
                
                avg_percentage = weighted_sum / total_weight
                completeness = 1.0 if len(light_scores) == 3 else 0.8
                perfect_bonus = 1.0 + perfect_count * 0.05
                
                final_percentage = min(100.0, avg_percentage * completeness * perfect_bonus)
                combined_score = max(0.0, (100.0 - final_percentage) / 25.0)
                
                valid_combinations.append({
                    'db_gem_id': gem_id,
                    'score': combined_score,
                    'percentage': final_percentage,
                    'light_sources': list(light_scores.keys()),
                    'perfect_count': perfect_count
                })
        
        return valid_combinations
    
    def score_to_percentage(self, score):
        """Convert score to percentage"""
        if score is None:
            return 0.0
        
        if score <= self.perfect_match_threshold:
            ratio = score / self.perfect_match_threshold
            return 100.0 - ratio * 2.0
        
        return max(0.0, min(98.0, 98.0 * np.exp(-score * 3.0)))
    
    def save_results(self, all_results, reports_dir, timestamp):
        """Save results"""
        summary_data = []
        for result in all_results:
            row = {
                'Gem_ID': result['gem_id'],
                'Timestamp': timestamp,
                'GOI_Light_Sources': '+'.join(result['goi_light_sources']),
                'Valid_Database_Gems': '+'.join(result['valid_database_gems']),
                'Light_Sources_Analyzed': '+'.join(result['light_source_results'].keys())
            }
            
            if result['best_match']:
                match = result['best_match']
                row.update({
                    'Best_Match': match['db_gem_id'],
                    'Score': match['score'],
                    'Percentage': match.get('percentage', 0)
                })
            
            summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_file = reports_dir / f"structural_analysis_{timestamp}.csv"
            df.to_csv(summary_file, index=False)
            print(f"📄 Summary saved: {summary_file.name}")
        
        json_file = reports_dir / f"structural_analysis_full_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"📄 Full results saved: {json_file.name}")
    
    def generate_plots(self, all_results, graphs_dir, timestamp):
        """Generate plots"""
        if not HAS_MATPLOTLIB:
            return
        
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scores = []
            gem_ids = []
            for result in all_results:
                if result['best_match']:
                    scores.append(result['best_match']['score'])
                    gem_ids.append(result['gem_id'])
            
            if scores:
                ax.bar(gem_ids, scores)
                ax.set_title('Gem Match Scores')
                ax.set_ylabel('Score (lower = better)')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = graphs_dir / f"match_scores_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Plot saved: {plot_file.name}")
        except Exception as e:
            print(f"⚠️ Plot failed: {e}")
    
    def run_auto_analysis(self, auto_select_complete=True):
        """Run automatic analysis"""
        print(f"🤖 Auto analysis: {self.input_source} source")
        
        if not self.gem_groups or not self.database_schema:
            print("❌ No gems or database not available")
            return False
        
        if auto_select_complete:
            for base_id, data in self.gem_groups.items():
                if all(len(data['files'][ls]) > 0 for ls in ['B', 'L', 'U']):
                    selected = {'files': {}, 'paths': {}}
                    for ls in ['B', 'L', 'U']:
                        selected['files'][ls] = data['files'][ls][0]
                        selected['paths'][ls] = data['paths'][ls][0]
                    self.selected_gems[base_id] = selected
        
        if not self.selected_gems:
            print("❌ No complete gems found")
            return False
        
        print(f"✅ Auto-selected {len(self.selected_gems)} complete gems")
        return self.run_analysis()
    
    def run(self):
        """Start GUI"""
        if self.mode == "gui" and self.root:
            self.root.mainloop()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Gem Structural Analyzer")
    parser.add_argument("--mode", choices=["gui", "auto"], default="gui")
    parser.add_argument("--input-source", choices=["archive", "current"], default="archive")
    parser.add_argument("--auto-complete", action="store_true", default=True)
    
    args = parser.parse_args()
    
    try:
        print("🚀 Multi-Gem Structural Analyzer with Automatic Valid Gem Detection")
        print(f"   Mode: {args.mode}, Source: {args.input_source}")
        print("   Features: Auto Light Source Detection, Auto Valid Gem Filtering, Perfect Self-Match")
        
        analyzer = MultiGemStructuralAnalyzer(mode=args.mode, input_source=args.input_source)
        
        if args.mode == "gui":
            analyzer.run()
        else:
            success = analyzer.run_auto_analysis(args.auto_complete)
            return success
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()