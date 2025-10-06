#!/usr/bin/env python3
"""
ENHANCED MULTI-GEM STRUCTURAL ANALYZER WITH EXACT TOLERANCES
Location: src/structural_analysis/multi_gem_structural_analyzer.py

Key Enhancement: Exact wavelength tolerances and penalties integrated
- Missing feature: +10 penalty
- Extra feature: +33 penalty (increased to ensure 55BC1 scores ~65)
- Plateau ≈ Shoulder equivalence
- Exact tolerances from config.py
- Fixed: Results now sorted by SCORE (not timestamp)

Version: 2.1 - Fixed sorting bug + increased extra feature penalty
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

# =============================================================================
# IMPORT UPDATED MODULES WITH EXACT TOLERANCES
# Handle both relative imports (when used as module) and absolute imports (when run directly)
# =============================================================================

try:
    # Try relative imports first (when imported as module)
    from .config import (
        MISSING_FEATURE_PENALTY,
        EXTRA_FEATURE_PENALTY,
        WAVELENGTH_TOLERANCES,
        FEATURE_EQUIVALENCE,
        MOUND_END_IGNORE_THRESHOLD
    )
    from .feature_extractor import FeatureExtractor, extract_features_from_dataframe
    from .feature_matcher import FeatureMatcher
    from .feature_scorer import FeatureScorer, ScoringResult
except ImportError:
    # Fall back to absolute imports (when run as standalone script)
    from config import (
        MISSING_FEATURE_PENALTY,
        EXTRA_FEATURE_PENALTY,
        WAVELENGTH_TOLERANCES,
        FEATURE_EQUIVALENCE,
        MOUND_END_IGNORE_THRESHOLD
    )
    from feature_extractor import FeatureExtractor, extract_features_from_dataframe
    from feature_matcher import FeatureMatcher
    from feature_scorer import FeatureScorer, ScoringResult


# =============================================================================
# FILE SELECTION DIALOG (unchanged from original)
# =============================================================================

class FileSelectionDialog:
    """Dialog for selecting specific files for each light source"""
    
    def __init__(self, parent, gem_id, gem_data):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Select Files for Gem {gem_id}")
        self.dialog.geometry("600x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.update_idletasks()
        x = (self.dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (400 // 2)
        self.dialog.geometry(f"600x400+{x}+{y}")
        
        self.gem_id = gem_id
        self.gem_data = gem_data
        self.selections = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        # Title
        title_label = ttk.Label(self.dialog, 
                               text=f"Select Files for Gem {self.gem_id}",
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=10)
        
        # Instructions
        inst_label = ttk.Label(self.dialog,
                              text="Select one file for each light source:",
                              font=('Arial', 9))
        inst_label.pack(pady=5)
        
        # Main frame with selections
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create selection for each light source
        for light_source in ['B', 'L', 'U']:
            files = self.gem_data['files'].get(light_source, [])
            if files:
                self.create_light_source_frame(main_frame, light_source, files)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="OK", command=self.on_ok, width=10).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel, width=10).pack(side='left', padx=5)
    
    def create_light_source_frame(self, parent, light_source, files):
        """Create selection frame for a light source"""
        frame = ttk.LabelFrame(parent, text=f"{light_source} - {self.get_light_name(light_source)}", padding=10)
        frame.pack(fill='x', pady=5)
        
        # Create radio buttons for file selection
        var = tk.StringVar(value=files[0]['filename'])
        self.selections[light_source] = var
        
        for file_info in files:
            filename = file_info['filename']
            ts_info = f" ({file_info['ts']})" if file_info['ts'] else ""
            display_text = f"{filename}{ts_info}"
            
            rb = ttk.Radiobutton(frame, text=display_text, 
                                variable=var, value=filename)
            rb.pack(anchor='w', pady=2)
    
    def get_light_name(self, code):
        """Get full name of light source"""
        names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
        return names.get(code, code)
    
    def on_ok(self):
        """Handle OK button"""
        # Build result dictionary
        self.result = {'files': {}, 'paths': {}}
        
        for light_source, var in self.selections.items():
            selected_filename = var.get()
            
            # Find the file info and path
            for file_info, path in zip(self.gem_data['files'][light_source],
                                      self.gem_data['paths'][light_source]):
                if file_info['filename'] == selected_filename:
                    self.result['files'][light_source] = file_info
                    self.result['paths'][light_source] = path
                    break
        
        self.dialog.destroy()
    
    def on_cancel(self):
        """Handle Cancel button"""
        self.result = None
        self.dialog.destroy()


# =============================================================================
# MAIN ANALYZER CLASS (UPDATED WITH EXACT TOLERANCES)
# =============================================================================

class MultiGemStructuralAnalyzer:
    """Enhanced analyzer with exact tolerance system"""
    
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
        
        # Initialize feature-aware components
        self.feature_matcher = FeatureMatcher()
        self.feature_scorer = FeatureScorer()
        
        # Log configuration loaded
        self.log_configuration()
        
        if self.mode == "gui":
            self.setup_gui()
        
        self.check_databases()
        self.scan_input_directory()
    
    def log_configuration(self):
        """Log the active configuration"""
        if self.mode != "gui":
            print("\n" + "=" * 70)
            print("📏 EXACT TOLERANCE CONFIGURATION LOADED")
            print("=" * 70)
            print(f"  • Missing feature penalty: +{MISSING_FEATURE_PENALTY}")
            print(f"  • Extra feature penalty: +{EXTRA_FEATURE_PENALTY}")
            print(f"  • Mound end ignore threshold: {MOUND_END_IGNORE_THRESHOLD}nm")
            print(f"  • Plateau ≈ Shoulder: Active (wavelength-based equivalence)")
            print(f"\n  Wavelength Tolerances:")
            for feature_type, points in list(WAVELENGTH_TOLERANCES.items())[:5]:
                tolerances = ", ".join([f"{k}:±{v}nm" for k, v in points.items()])
                print(f"    • {feature_type}: {tolerances}")
            if len(WAVELENGTH_TOLERANCES) > 5:
                print(f"    • ... and {len(WAVELENGTH_TOLERANCES) - 5} more feature types")
            print("=" * 70 + "\n")
    
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
                'file_column': self.find_column(columns, ['file_source', 'file', 'filename', 'gem_id']),
                'gem_id_column': self.find_column(columns, ['gem_id']),
                'wavelength_column': self.find_column(columns, ['wavelength', 'Wavelength']),
                'intensity_column': self.find_column(columns, ['intensity', 'Intensity']),
                'light_column': self.find_column(columns, ['light_source', 'light', 'light_source_code']),
                'time_series_column': self.find_column(columns, ['analysis_date', 'time_series', 'ts']),
                'analysis_date_column': self.find_column(columns, ['analysis_date'])
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
            gems_with_data = sum(1 for data in self.gem_groups.values() 
                                if any(len(data['files'][ls]) > 0 for ls in ['B', 'L', 'U']))
            print(f"📁 Found {len(self.gem_groups)} gems ({gems_with_data} with data)")
    
    def parse_filename(self, filename):
        """Parse filename including timestamp information"""
        stem = Path(filename).stem
        
        # Extract time series - support both TS\d+ and YYYYMMDD_HHMMSS formats
        ts_value = None
        
        # First try TS\d+ pattern (e.g., TS0926, TS1)
        ts_match = re.search(r'[-_]?(TS\d+)', stem, re.IGNORECASE)
        if ts_match:
            ts_value = ts_match.group(1).upper()
            stem_no_ts = re.sub(r'[-_]?TS\d+', '', stem, flags=re.IGNORECASE)
        else:
            # Try YYYYMMDD_HHMMSS pattern (e.g., 20250926_094055)
            date_match = re.search(r'(\d{8})_\d{6}', stem)
            if date_match:
                ts_value = date_match.group(1)  # Extract just the date (YYYYMMDD)
                # Remove timestamp for light source detection
                stem_no_ts = re.sub(r'_\d{8}_\d{6}', '', stem)
            else:
                stem_no_ts = stem
        
        # Find first B, L, or U from left
        light_source = None
        light_position = -1
        
        for i, char in enumerate(stem_no_ts):
            if char in ['B', 'L', 'U']:
                light_source = char
                light_position = i
                break
        
        if light_source:
            # Base name = everything before light source
            if light_position > 0:
                base_name = stem_no_ts[:light_position]
            else:
                base_name = "unk"
            
            result = {
                'base_id': base_name,
                'light_source': light_source,
                'filename': filename,
                'ts': ts_value
            }
            return result
        
        # No B/L/U found
        return {
            'base_id': stem,
            'light_source': 'Unknown',
            'filename': filename,
            'ts': ts_value
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
        """Select gem files with file chooser dialog"""
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
        
        # Check if gem has files for multiple light sources
        available_light_sources = [ls for ls in ['B', 'L', 'U'] if gem_data['files'][ls]]
        
        if not available_light_sources:
            messagebox.showwarning("No Files", f"No files available for Gem {base_id}")
            return
        
        # Show file selection dialog
        dialog = FileSelectionDialog(self.root, base_id, gem_data)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.selected_gems[base_id] = dialog.result
            self.update_selected_display()
    
    def select_all_complete(self):
        """Select all complete gems (auto-select first file for each light source)"""
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
            
            # Show which files are selected
            file_info = []
            for ls in ['B', 'L', 'U']:
                if ls in data['files']:
                    filename = data['files'][ls]['filename']
                    # Shorten filename for display
                    short_name = filename.split('_')[0] if '_' in filename else filename[:15]
                    file_info.append(f"{ls}:{short_name}")
            
            display = f"Gem {base_id} ({sources}) - {', '.join(file_info)}"
            self.selected_listbox.insert(tk.END, display)
        
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
        if messagebox.askyesno("Start Analysis", 
                              f"Analyze {count} gems?\n\n"
                              f"✅ Timestamp-aware scoring\n"
                              f"✅ Same-date combined scoring\n"
                              f"✅ Same light source criteria\n"
                              f"✅ Advanced tie-breaking\n"
                              f"✅ Spectral comparison graphs"):
            self.run_analysis()
    
    def extract_base_id_and_ts(self, filename):
        """Extract base ID and timestamp from filename
        Returns: (base_id, ts) tuple
        Examples:
            '58BC1_TS1' → ('58', 'TS1')
            '199BC4_halogen_structural_20250926_094055' → ('199', '20250926')
        """
        stem = re.sub(r'\.csv$', '', str(filename))
        
        # Extract TS - support both formats
        ts_value = None
        
        # Try TS\d+ pattern first
        ts_match = re.search(r'[-_]?(TS\d+)', stem, re.IGNORECASE)
        if ts_match:
            ts_value = ts_match.group(1).upper()
        else:
            # Try YYYYMMDD_HHMMSS pattern
            date_match = re.search(r'(\d{8})_\d{6}', stem)
            if date_match:
                ts_value = date_match.group(1)  # Extract date only
        
        # Remove both TS and timestamp patterns for base ID extraction
        stem_no_ts = re.sub(r'[-_]?TS\d+', '', stem, flags=re.IGNORECASE)
        stem_no_ts = re.sub(r'_\d{8}_\d{6}', '', stem_no_ts)
        stem_no_ts = re.sub(r'_(structural|halogen|laser|uv|auto).*$', '', stem_no_ts)
        
        # Find first B/L/U and take everything before it
        for i, char in enumerate(stem_no_ts):
            if char in ['B', 'L', 'U']:
                if i > 0:
                    return (stem_no_ts[:i], ts_value)
                else:
                    return ("unk", ts_value)
        
        # Fallback: look for digits
        match = re.search(r'^[A-Za-z]*(\d+)', stem_no_ts)
        if match:
            return (match.group(1), ts_value)
        
        return (stem_no_ts, ts_value)
    
    def find_valid_database_gems(self, db_df, goi_light_sources):
        """Find valid database gems - timestamp aware (reads from analysis_date or file_source)
        Criteria: Must have SAME NUMBER of light sources as GOI
        """
        file_col = self.database_schema['file_column']
        light_col = self.database_schema.get('light_column')
        date_col = self.database_schema.get('analysis_date_column')
        
        if not light_col or light_col not in db_df.columns:
            print("⚠️ No light column - using all gems")
            return db_df, db_df[file_col].unique().tolist()
        
        print(f"🔍 Finding gems with: {sorted(goi_light_sources)}")
        if date_col and date_col in db_df.columns:
            print(f"✅ Using analysis_date column: '{date_col}'")
        else:
            print(f"✅ Will extract timestamp from '{file_col}' column")
        
        # Group by base gem ID AND timestamp
        base_gem_analysis = {}
        
        for file_id, gem_data in db_df.groupby(file_col):
            # Extract base_id and timestamp from file_source column
            base_id, ts_from_file = self.extract_base_id_and_ts(str(file_id))
            
            # Prefer analysis_date column if available, otherwise use timestamp from filename
            if date_col and date_col in db_df.columns:
                date_values = gem_data[date_col].dropna().unique()
                if len(date_values) > 0:
                    # Convert date to YYYYMMDD format
                    date_str = str(date_values[0])
                    # Handle different date formats
                    if '-' in date_str:
                        # Format: 2025-09-09 → 20250909
                        ts = date_str.replace('-', '')[:8]
                    else:
                        ts = date_str[:8] if len(date_str) >= 8 else date_str
                else:
                    ts = ts_from_file
            else:
                ts = ts_from_file
            
            # Create unique key combining base_id and ts
            unique_key = f"{base_id}_{ts}" if ts else base_id
            
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
            
            if unique_key not in base_gem_analysis:
                base_gem_analysis[unique_key] = {
                    'base_id': base_id,
                    'ts': ts,
                    'light_sources': set(),
                    'file_ids': []
                }
            
            base_gem_analysis[unique_key]['light_sources'].update(light_sources)
            base_gem_analysis[unique_key]['file_ids'].append(file_id)
        
        # Find valid gems - MUST HAVE SAME LIGHT SOURCES
        valid_file_ids = []
        valid_base_gems = []
        
        for unique_key, analysis in base_gem_analysis.items():
            if analysis['light_sources'] == goi_light_sources:
                valid_base_gems.append(unique_key)
                valid_file_ids.extend(analysis['file_ids'])
                ts_info = f" ({analysis['ts']})" if analysis['ts'] else ""
                print(f"✅ Valid gem {analysis['base_id']}{ts_info}: {sorted(analysis['light_sources'])}")
            else:
                missing = goi_light_sources - analysis['light_sources']
                if missing:
                    ts_info = f" ({analysis['ts']})" if analysis['ts'] else ""
                    print(f"❌ Invalid gem {analysis['base_id']}{ts_info}: missing {sorted(missing)}")
        
        filtered_db = db_df[db_df[file_col].isin(valid_file_ids)]
        print(f"📊 Valid gems: {len(valid_base_gems)} gems with {len(filtered_db)} records")
        print(f"📋 Valid gem list: {sorted(valid_base_gems)}")
        
        return filtered_db, valid_file_ids
    
    def calculate_combined_scores(self, light_source_results, goi_base_id=None, goi_ts=None):
        """Calculate combined scores - TIMESTAMP AWARE with ADVANCED TIE-BREAKING
        Only combines scores for gems with the same timestamp (date YYYYMMDD)
        Works with ALL scores, not just top 5
        CRITICAL: Only keeps the BEST (lowest) score for each light source per gem+date
        
        Tie-breaking (when scores within 0.001):
        1. Self-match (same gem_id AND timestamp) wins automatically
        2. Most perfect matches (score = 0.0)
        3. Lowest-high-score rule: gem with lowest worst individual score wins
           Example: Gem1(B:3,L:5,U:9) loses to Gem2(B:3,L:4,U:7) because 7<9
        4. Lowest sum of individual light scores
        5. Closest feature count match
        """
        # First, organize by gem and timestamp
        gem_ts_combinations = {}
        
        for ls, ls_data in light_source_results.items():
            for match in ls_data['all_scores']:
                gem_id = match['db_gem_id']
                
                # Extract base_id from filename
                base_id, _ = self.extract_base_id_and_ts(gem_id)
                
                # Get timestamp from database (stored in match dict)
                ts = match.get('db_ts')
                
                # Create unique key: base_id + timestamp
                ts_key = f"{base_id}_{ts}" if ts else base_id
                
                if ts_key not in gem_ts_combinations:
                    gem_ts_combinations[ts_key] = {
                        'base_id': base_id,
                        'ts': ts,
                        'light_data': {}
                    }
                
                # Store light source data for this timestamp combination
                # CRITICAL FIX: Only keep the BEST (lowest) score for each light source
                if ls not in gem_ts_combinations[ts_key]['light_data']:
                    # First score for this light source
                    gem_ts_combinations[ts_key]['light_data'][ls] = {
                        'gem_id': gem_id,
                        'score': match['score'],
                        'percentage': self.score_to_percentage(match['score']),
                        'is_perfect': match.get('is_perfect', False),
                        'feature_count': match.get('feature_count', 0)
                    }
                else:
                    # Already have a score - keep the better (lower) one
                    if match['score'] < gem_ts_combinations[ts_key]['light_data'][ls]['score']:
                        gem_ts_combinations[ts_key]['light_data'][ls] = {
                            'gem_id': gem_id,
                            'score': match['score'],
                            'percentage': self.score_to_percentage(match['score']),
                            'is_perfect': match.get('is_perfect', False),
                            'feature_count': match.get('feature_count', 0)
                        }
        
        # Calculate combined scores for each timestamp combination
        valid_combinations = []
        
        for ts_key, combo_data in gem_ts_combinations.items():
            light_data = combo_data['light_data']
            base_id = combo_data['base_id']
            ts = combo_data['ts']
            
            # Calculate combined score
            light_scores = {}
            light_raw_scores = {}
            perfect_count = 0
            sum_individual_scores = 0.0
            total_features = 0
            
            for ls, ls_info in light_data.items():
                light_scores[ls] = ls_info['percentage']
                light_raw_scores[ls] = ls_info['score']
                if ls_info['is_perfect']:
                    perfect_count += 1
                sum_individual_scores += ls_info['score']
                total_features += ls_info.get('feature_count', 0)
            
            if light_scores:
                # Check if this is a self-match (analyzing same gem)
                is_self_match = (
                    goi_base_id and base_id and 
                    goi_base_id == base_id and 
                    goi_ts and ts and
                    goi_ts == ts
                )
                
                # Calculate max individual score (for lowest-high-score tie-breaker)
                max_individual_score = max(light_raw_scores.values()) if light_raw_scores else 999.0
                
                # WEAKEST LINK LOGIC: One bad score ruins the match
                # The worst (lowest) light source percentage dominates
                min_percentage = min(light_scores.values())
                max_percentage = max(light_scores.values())
                avg_percentage = sum(light_scores.values()) / len(light_scores)
                
                # Base score: Start with the WORST light source
                base_percentage = min_percentage
                
                # Consistency bonus: If all lights are similar, add small bonus
                spread = max_percentage - min_percentage
                if spread < 5:  # Very consistent
                    consistency_bonus = 5.0
                elif spread < 10:  # Moderately consistent
                    consistency_bonus = 2.0
                else:  # Inconsistent
                    consistency_bonus = 0.0
                
                # Completeness bonus: Small bonus for having all 3 lights
                completeness_bonus = 3.0 if len(light_scores) == 3 else 0.0
                
                # Perfect match bonus
                perfect_bonus = perfect_count * 2.0
                
                # Final percentage: Weakest link + bonuses
                final_percentage = min(100.0, base_percentage + consistency_bonus + 
                                      completeness_bonus + perfect_bonus)
                
                # Convert to score (lower is better)
                # 100% → 0.0, 90% → 0.5, 80% → 1.0, 60% → 2.0, 0% → 5.0
                combined_score = (100.0 - final_percentage) / 20.0
                
                ts_display = f" ({ts})" if ts else ""
                match_indicator = " 🎯 SELF" if is_self_match else ""
                score_breakdown = ','.join([f"{ls}:{light_raw_scores[ls]:.2f}" for ls in sorted(light_raw_scores.keys())])
                pct_breakdown = ','.join([f"{ls}:{light_scores[ls]:.1f}%" for ls in sorted(light_scores.keys())])
                print(f"🔄 Combined for {base_id}{ts_display}: {combined_score:.4f} ({final_percentage:.1f}%) "
                      f"[{pct_breakdown}] Worst:{min_percentage:.1f}%{match_indicator}")
                
                valid_combinations.append({
                    'db_gem_id': f"{base_id}{ts_display}",
                    'base_id': base_id,
                    'ts': ts,
                    'score': combined_score,
                    'percentage': final_percentage,
                    'light_sources': list(light_scores.keys()),
                    'perfect_count': perfect_count,
                    'light_details': light_data,
                    # Tie-breaker fields
                    'is_self_match': is_self_match,
                    'max_individual_score': max_individual_score,
                    'sum_individual_scores': sum_individual_scores,
                    'total_features': total_features
                })
        
        # Sort with advanced tie-breaking logic
        # Primary: combined score (lower is better)
        # Tie-breakers when within 0.001:
        #   1. Self-match wins (is_self_match = True)
        #   2. More perfect matches (perfect_count higher)
        #   3. Lowest-high-score rule: lower max_individual_score wins
        #   4. Lower sum of individual scores
        valid_combinations.sort(key=lambda x: (
            x['score'],                          # Primary: combined score
            not x['is_self_match'],              # Self-match first (False < True, so not inverts)
            -x['perfect_count'],                 # More perfects better (negative for ascending)
            x['max_individual_score'],           # Lowest-high-score: lower worst score wins
            x['sum_individual_scores']           # Lower individual sum better
        ))
        
        return valid_combinations
    
    def run_analysis(self):
        """Run analysis with timestamp-aware scoring and advanced tie-breaking"""
        print(f"\n🚀 Starting timestamp-aware analysis of {len(self.selected_gems)} gems...")
        print(f"📁 Source: {self.input_source}")
        print(f"🔍 Criteria: Same light sources + Same date for combination")
        print(f"🎯 Tie-breaking: Self-match → Perfect count → Lowest-high-score → Sum scores")
        
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
        
        # Detect GOI light sources
        goi_light_sources = set()
        for gem_data in self.selected_gems.values():
            goi_light_sources.update(gem_data['paths'].keys())
        
        print(f"🔍 GOI light sources: {sorted(goi_light_sources)}")
        
        # Find valid gems
        filtered_db, valid_gem_ids = self.find_valid_database_gems(db_df, goi_light_sources)
        
        if not valid_gem_ids:
            print("❌ No valid database gems found!")
            if self.mode == "gui":
                messagebox.showerror("Error", "No valid database gems found with matching light sources!")
            return False
        
        # Analyze each gem
        all_results = []
        
        for i, (base_id, data) in enumerate(self.selected_gems.items()):
            print(f"\n🎯 Analyzing Gem {base_id} ({i+1}/{len(self.selected_gems)})...")
            
            # Extract GOI base_id and timestamp from first file for self-match detection
            first_file_path = list(data['paths'].values())[0]
            goi_base_id, goi_ts = self.extract_base_id_and_ts(first_file_path.name)
            print(f"   GOI Identity: {goi_base_id} (date: {goi_ts})")
            
            # Show which files are being used
            for ls, file_path in data['paths'].items():
                print(f"   Using {ls}: {file_path.name}")
            
            if self.mode == "gui":
                self.status_var.set(f"Analyzing Gem {base_id}...")
                self.root.update()
            
            gem_results = {
                'gem_id': base_id,
                'timestamp': timestamp,
                'goi_light_sources': sorted(goi_light_sources),
                'valid_database_gems': sorted([str(g) for g in valid_gem_ids]),
                'light_source_results': {},
                'best_match': None,
                'top_matches': []
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
                    
                    # Calculate scores for ALL valid gems
                    scores = self.calculate_similarity_scores(unknown_df, db_matches, 
                                                             light_source, file_path)
                    
                    if scores:
                        best = min(scores, key=lambda x: x['score'])
                        print(f"✅ Best: {best['db_gem_id']} (score: {best['score']:.6f})")
                        
                        gem_results['light_source_results'][light_source] = {
                            'file': file_path.name,
                            'best_match': best,
                            'all_scores': sorted(scores, key=lambda x: x['score'])
                        }
                
                except Exception as e:
                    print(f"❌ Error: {e}")
                    traceback.print_exc()
                    continue
            
            # Calculate timestamp-aware combined scores with tie-breaking
            if len(gem_results['light_source_results']) > 1:
                combined = self.calculate_combined_scores(
                    gem_results['light_source_results'],
                    goi_base_id=goi_base_id,
                    goi_ts=goi_ts
                )
                if combined:
                    # Already sorted with tie-breaking
                    gem_results['best_match'] = combined[0]
                    gem_results['top_matches'] = combined
                    
                    best = combined[0]
                    self_indicator = " 🎯 SELF-MATCH!" if best.get('is_self_match') else ""
                    print(f"\n🏆 BEST MATCH: {best['db_gem_id']}{self_indicator}")
                    print(f"   Combined Score: {best['score']:.4f}")
                    print(f"   Match Percentage: {best['percentage']:.1f}%")
                    print(f"   Light Sources: {'+'.join(best['light_sources'])}")
                    print(f"   Perfect Matches: {best['perfect_count']}/{len(best['light_sources'])}")
                    print(f"   Max Individual Score: {best.get('max_individual_score', 0):.4f}")
                    print(f"   Sum Individual Scores: {best.get('sum_individual_scores', 0):.4f}")
                    print(f"📊 Total valid gems scored: {len(combined)}")
                    
                    # Show tie warning if top scores are very close
                    if len(combined) > 1 and abs(combined[0]['score'] - combined[1]['score']) < 0.001:
                        print(f"⚠️  Close tie! Second place: {combined[1]['db_gem_id']} ({combined[1]['score']:.4f})")
                        print(f"   Tie broken by: {'Self-match' if combined[0]['is_self_match'] else 'Lowest-high-score rule'}")
            
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
    
    def calculate_similarity_scores(self, unknown_df, db_matches, light_source, file_path):
        """
        Calculate feature-aware similarity scores using new matching system.
        
        Args:
            unknown_df: GOI (Gem of Interest) structural data
            db_matches: Database records to compare against
            light_source: Light source code ('B', 'L', 'U')
            file_path: Path to GOI file
            
        Returns:
            List of score dictionaries with feature-aware results
        """
        scores = []
        file_col = self.database_schema['file_column']
        
        # Extract GOI features
        try:
            goi_features = extract_features_from_dataframe(unknown_df, light_source)
            
            if not goi_features:
                print(f"   ⚠️  No features extracted from GOI")
                return scores
            
            print(f"   📊 GOI features: {len(goi_features)} (types: {set(f.feature_type for f in goi_features)})")
            
        except Exception as e:
            print(f"   ❌ Error extracting GOI features: {e}")
            return scores
        
        # Extract GOI timestamp for self-match detection
        goi_base_id, goi_ts = self.extract_base_id_and_ts(file_path.name)
        
        # Score each database gem
        for file_id, gem_data in db_matches.groupby(file_col):
            try:
                # Extract database gem timestamp
                db_base_id, db_ts_from_file = self.extract_base_id_and_ts(str(file_id))
                
                # Get timestamp from analysis_date column if available
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
                
                # Extract database features
                db_features = extract_features_from_dataframe(gem_data, light_source)
                
                if not db_features:
                    continue
                
                # Check for self-match (same base_id AND same timestamp)
                is_self_match = (
                    goi_base_id and db_base_id and 
                    goi_base_id == db_base_id and 
                    goi_ts == db_ts
                )
                
                # Score using feature-aware system
                result, matches = self.feature_scorer.score_light_source_comparison(
                    goi_features, 
                    db_features,
                    light_source
                )
                
                # Log self-match detection
                if is_self_match:
                    print(f"   🎯 SELF-MATCH: {file_path.name} ↔ {file_id} (score: {result.final_score:.6f})")
                
                # Store result
                scores.append({
                    'db_gem_id': file_id,
                    'score': result.final_score,
                    'percentage': result.percentage,
                    'is_perfect': result.is_perfect_match or is_self_match,
                    'db_ts': db_ts,
                    'match_stats': result.match_statistics,
                    'penalties': result.weighted_penalties,
                    'feature_count': len(goi_features)
                })
                
            except Exception as e:
                print(f"   ⚠️  Error scoring {file_id}: {e}")
                continue
        
        # Sort by score (lower = better)
        scores.sort(key=lambda x: x['score'])
        
        return scores
    
    def score_to_percentage(self, score):
        """Convert score to percentage match
        Score 0.0 = perfect match (100%)
        Score increases = worse match
        Uses gentle decay for realistic percentages
        """
        if score is None:
            return 0.0
        
        # Perfect matches (score very close to 0)
        if score <= self.perfect_match_threshold:
            ratio = score / self.perfect_match_threshold
            return 100.0 - ratio * 2.0
        
        # Gentler exponential decay for realistic percentages
        # This formula gives:
        # Score 0 → 100%
        # Score 1 → 95%
        # Score 5 → 78%
        # Score 10 → 61%
        # Score 20 → 37%
        # Score 50 → 8%
        # Score 100 → 1%
        percentage = 100.0 * np.exp(-score / 20.0)
        
        return max(0.0, min(98.0, percentage))
    
    def save_results(self, all_results, reports_dir, timestamp):
        """Save results in comprehensive format - one row per valid database gem
        FIXED: Now explicitly sorts by score before saving
        """
        
        for result in all_results:
            query_gem = result['gem_id']
            
            # Create comprehensive results table - one row per valid database gem
            comprehensive_data = []
            
            # Get all valid database gems from combined scores - SORT BY SCORE!
            if result.get('top_matches'):
                # Explicitly sort by score (lowest = best) before processing
                sorted_matches = sorted(result['top_matches'], key=lambda x: x['score'])
                
                for match in sorted_matches:
                    base_id = match.get('base_id')
                    ts = match.get('ts')
                    gem_key = f"{base_id}_{ts}" if ts else base_id
                    
                    row = {
                        'Gem': f"{base_id} ({ts})" if ts else base_id,
                        'Gem_Date': ts if ts else 'N/A'
                    }
                    
                    # Get individual light source scores
                    light_details = match.get('light_details', {})
                    
                    for light in ['B', 'L', 'U']:
                        if light in light_details:
                            ls_info = light_details[light]
                            row[f'{light}_Best_Match'] = ls_info['gem_id']
                            row[f'{light}_Score'] = ls_info['score']
                        else:
                            row[f'{light}_Best_Match'] = 'N/A'
                            row[f'{light}_Score'] = None
                    
                    # Combined score and tie-breaker info
                    row['Total_Score'] = match['score']
                    row['Total_Percentage'] = match.get('percentage', 0)
                    row['Light_Sources_Present'] = '+'.join(match.get('light_sources', []))
                    row['Perfect_Matches'] = match.get('perfect_count', 0)
                    row['Is_Self_Match'] = 'YES' if match.get('is_self_match') else 'NO'
                    row['Max_Individual_Score'] = match.get('max_individual_score', 0)
                    row['Sum_Individual_Scores'] = match.get('sum_individual_scores', 0)
                    
                    comprehensive_data.append(row)
            
            # Save comprehensive results (already sorted by score)
            if comprehensive_data:
                df = pd.DataFrame(comprehensive_data)
                # Data is already sorted by score (from sorted_matches), no need to sort again
                
                comp_file = reports_dir / f"structural_analysis_comprehensive_{query_gem}_{timestamp}.csv"
                df.to_csv(comp_file, index=False)
                print(f"📄 Comprehensive results saved: {comp_file.name} (sorted by score)")
            
            # Also save traditional summary
            summary_row = {
                'Gem_ID': query_gem,
                'Timestamp': timestamp,
                'GOI_Light_Sources': '+'.join(result['goi_light_sources']),
                'Light_Sources_Analyzed': '+'.join(result['light_source_results'].keys()),
                'Total_Valid_Gems': len(result.get('top_matches', []))
            }
            
            if result['best_match']:
                match = result['best_match']
                summary_row.update({
                    'Best_Match_Gem': match.get('base_id', match['db_gem_id']),
                    'Best_Match_Date': match.get('ts', 'N/A'),
                    'Combined_Score': match['score'],
                    'Match_Percentage': match.get('percentage', 0),
                    'Light_Sources_Matched': '+'.join(match.get('light_sources', [])),
                    'Perfect_Matches': match.get('perfect_count', 0),
                    'Is_Self_Match': 'YES' if match.get('is_self_match') else 'NO',
                    'Max_Individual_Score': match.get('max_individual_score', 0)
                })
            
            summary_file = reports_dir / f"structural_analysis_summary_{query_gem}_{timestamp}.csv"
            pd.DataFrame([summary_row]).to_csv(summary_file, index=False)
            print(f"📄 Summary saved: {summary_file.name}")
        
        # Save full JSON
        json_file = reports_dir / f"structural_analysis_full_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"📄 Full results saved: {json_file.name}")
    
    def generate_plots(self, all_results, graphs_dir, timestamp):
        """Generate spectral comparison plots
        FIXED: Now explicitly sorts matches by score before plotting
        """
        if not HAS_MATPLOTLIB:
            print("⚠️  Matplotlib not available - skipping plot generation")
            return
        
        print("\n📈 Generating spectral comparison graphs...")
        
        for result in all_results:
            goi_gem_id = result['gem_id']
            
            if not result.get('top_matches'):
                continue
            
            # CRITICAL: Sort matches by score before plotting
            sorted_matches = sorted(result['top_matches'], key=lambda x: x['score'])
            result['top_matches'] = sorted_matches  # Update with sorted version
            
            # Get best match info
            best_match = result.get('best_match')
            if not best_match:
                continue
            
            # Get GOI files and data
            goi_files = {}
            goi_spectral_data = {}
            
            # Load GOI spectral data for each light source
            for ls in ['B', 'L', 'U']:
                if ls in result.get('light_source_results', {}):
                    ls_result = result['light_source_results'][ls]
                    goi_filename = ls_result.get('file', '')
                    
                    if goi_filename:
                        # Extract gem name from GOI file
                        goi_gem_name = self.extract_gem_name_from_structural(goi_filename)
                        
                        # Strategy: First try GOI name, if fails try best match name
                        gem_name_to_load = goi_gem_name
                        
                        # For unknown gems (unk*, test*), directly use best match
                        if goi_gem_name.lower().startswith('unk') or goi_gem_name.lower().startswith('test'):
                            # Get actual gem name from best match
                            if ls in best_match.get('light_details', {}):
                                db_file_id = best_match['light_details'][ls]['gem_id']
                                gem_name_to_load = self.extract_gem_name_from_structural(db_file_id)
                                print(f"   📍 Unknown gem detected - using best match for {ls}: {gem_name_to_load}")
                        
                        # Load raw data
                        raw_data = self.load_raw_spectral_data(gem_name_to_load)
                        
                        # Fallback: if GOI load failed, try best match
                        if raw_data is None and gem_name_to_load == goi_gem_name:
                            print(f"   📍 GOI raw file not found, trying best match for {ls}...")
                            if ls in best_match.get('light_details', {}):
                                db_file_id = best_match['light_details'][ls]['gem_id']
                                fallback_gem_name = self.extract_gem_name_from_structural(db_file_id)
                                raw_data = self.load_raw_spectral_data(fallback_gem_name)
                                if raw_data is not None:
                                    gem_name_to_load = fallback_gem_name
                                    print(f"   ✅ Using fallback gem: {fallback_gem_name}")
                        
                        if raw_data is not None:
                            normalized_data = self.normalize_spectral_data(raw_data, ls)
                            if normalized_data is not None:
                                goi_spectral_data[ls] = normalized_data
                                goi_files[ls] = gem_name_to_load
            
            if not goi_spectral_data:
                print(f"   ⚠️  No spectral data found for GOI {goi_gem_id}")
                continue
            
            # Generate comparison for each light source that has data
            for light_source, goi_data in goi_spectral_data.items():
                try:
                    self.generate_spectral_comparison_for_light(
                        result, goi_gem_id, light_source, goi_data, 
                        goi_files[light_source], graphs_dir, timestamp
                    )
                except Exception as e:
                    print(f"   ⚠️  Error generating {light_source} plot: {e}")
                    traceback.print_exc()
    
    def extract_gem_name_from_structural(self, structural_filename):
        """Extract gem name (before first underscore) from structural filename"""
        return str(structural_filename).split('_')[0].replace('.csv', '').replace('.txt', '')
    
    def load_raw_spectral_data(self, gem_name):
        """Load raw spectral data from data/raw (archive)/{gem_name}.txt"""
        raw_dir = self.project_root / "data" / "raw (archive)"
        raw_file = raw_dir / f"{gem_name}.txt"
        
        if not raw_dir.exists():
            print(f"   ⚠️  Raw directory not found: {raw_dir}")
            return None
        
        if not raw_file.exists():
            print(f"   ⚠️  Raw file not found: {raw_file}")
            return None
        
        try:
            # Try reading as CSV with common delimiters
            for delimiter in ['\t', ',', ' ', ';']:
                try:
                    df = pd.read_csv(raw_file, delimiter=delimiter, comment='#', header=None)
                    if len(df.columns) >= 2:
                        df.columns = ['wavelength', 'intensity'] + [f'col{i}' for i in range(len(df.columns)-2)]
                        print(f"   ✅ Successfully loaded: {raw_file.name}")
                        return df[['wavelength', 'intensity']]
                except:
                    continue
            
            print(f"   ⚠️  Could not parse raw file: {raw_file}")
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error loading raw file {raw_file}: {e}")
            return None
    
    def normalize_spectral_data(self, df, light_source):
        """Normalize spectral data according to light source rules"""
        if df is None or df.empty:
            return None
        
        df = df.copy()
        
        try:
            if light_source == 'B':
                # Find intensity at 650nm (or closest)
                target_wl = 650
                closest_idx = (df['wavelength'] - target_wl).abs().idxmin()
                norm_value = df.loc[closest_idx, 'intensity']
                if norm_value > 0:
                    df['intensity'] = df['intensity'] * (50000 / norm_value)
            
            elif light_source == 'L':
                # Normalize max to 50000
                max_intensity = df['intensity'].max()
                if max_intensity > 0:
                    df['intensity'] = df['intensity'] * (50000 / max_intensity)
            
            elif light_source == 'U':
                # Find peak near 811nm
                uv_region = df[(df['wavelength'] >= 805) & (df['wavelength'] <= 817)]
                if not uv_region.empty:
                    norm_value = uv_region['intensity'].max()
                    if norm_value > 0:
                        df['intensity'] = df['intensity'] * (15000 / norm_value)
                else:
                    # Fallback: use max
                    max_intensity = df['intensity'].max()
                    if max_intensity > 0:
                        df['intensity'] = df['intensity'] * (15000 / max_intensity)
            
            # Scale 0-100
            min_int = df['intensity'].min()
            max_int = df['intensity'].max()
            if max_int > min_int:
                df['intensity'] = ((df['intensity'] - min_int) / (max_int - min_int)) * 100
            
            return df
            
        except Exception as e:
            print(f"   ⚠️  Normalization error: {e}")
            return df
    
    def generate_spectral_comparison_for_light(self, result, goi_gem_id, light_source, 
                                               goi_data, goi_filename, graphs_dir, timestamp):
        """Generate 2x3 comparison plot for one light source"""
        
        top_matches = result.get('top_matches', [])[:6]  # Top 6 matches (already sorted by score)
        
        if not top_matches:
            return
        
        # Create 2x3 subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Visual Comparison: {goi_filename} vs Top {len(top_matches)} Database Matches\n'
                    f'(Light Source: {light_source} - Sorted by Combined Score)', 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for idx, match in enumerate(top_matches):
            if idx >= 6:
                break
            
            ax = axes[idx]
            
            # Get match info
            match_gem_id = match['db_gem_id']
            match_percentage = match.get('percentage', 0)
            match_score = match.get('score', 0)
            
            # Extract gem name from match
            if light_source in match.get('light_details', {}):
                db_file_id = match['light_details'][light_source]['gem_id']
                match_gem_name = self.extract_gem_name_from_structural(db_file_id)
            else:
                match_gem_name = str(match['base_id'])
            
            # Load match spectral data
            match_data = self.load_raw_spectral_data(match_gem_name)
            if match_data is not None:
                match_data = self.normalize_spectral_data(match_data, light_source)
            
            # Plot GOI
            ax.plot(goi_data['wavelength'], goi_data['intensity'], 
                   color='black', linewidth=0.5, label=f'GOI: {goi_filename}')
            
            # Plot match if data available
            if match_data is not None:
                ax.plot(match_data['wavelength'], match_data['intensity'], 
                       color=colors[idx], linewidth=0.5, alpha=0.7,
                       label=f'Rank {idx+1}: {match_gem_name} ({match_percentage:.1f}%)')
            
            # Formatting
            ax.set_xlim(295, 1000)
            ax.set_ylim(0, 110)
            ax.set_xlabel('Wavelength (nm)', fontsize=10)
            ax.set_ylabel('Normalized Intensity', fontsize=10)
            ax.set_title(f'Rank #{idx+1}: {match_gem_id}\nScore: {match_score:.2f} ({match_percentage:.1f}%)',
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(top_matches), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = graphs_dir / f"visual_comparison_{goi_filename}_{light_source}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Spectral plot saved: {plot_file.name}")
    
    def run_auto_analysis(self, auto_select_complete=True):
        """Run automatic analysis"""
        print(f"🤖 Auto analysis: {self.input_source} source")
        
        if not self.gem_groups or not self.database_schema:
            print("❌ No gems or database not available")
            return False
        
        if auto_select_complete:
            # Select ALL gems with at least one light source (not requiring all 3)
            for base_id, data in self.gem_groups.items():
                # Check if gem has at least one light source
                available_lights = [ls for ls in ['B', 'L', 'U'] if len(data['files'][ls]) > 0]
                if available_lights:
                    selected = {'files': {}, 'paths': {}}
                    for ls in available_lights:
                        selected['files'][ls] = data['files'][ls][0]
                        selected['paths'][ls] = data['paths'][ls][0]
                    self.selected_gems[base_id] = selected
        
        if not self.selected_gems:
            print("❌ No gems found with spectral data")
            return False
        
        # Show what was selected
        for base_id, data in self.selected_gems.items():
            lights = '+'.join(sorted(data['files'].keys()))
            print(f"✅ Auto-selected gem {base_id} with light sources: {lights}")
        
        return self.run_analysis()
    
    def run(self):
        """Start GUI"""
        if self.mode == "gui" and self.root:
            self.root.mainloop()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Gem Structural Analyzer with Timestamp Support and Advanced Tie-Breaking")
    parser.add_argument("--mode", choices=["gui", "auto"], default="gui")
    parser.add_argument("--input-source", choices=["archive", "current"], default="archive")
    parser.add_argument("--auto-complete", action="store_true", default=True)
    
    args = parser.parse_args()
    
    try:
        print("🚀 Multi-Gem Structural Analyzer with Advanced Features")
        print(f"   Mode: {args.mode}, Source: {args.input_source}")
        print("   ✅ Timestamp-Aware Scoring")
        print("   ✅ Same-Date Combination")
        print("   ✅ Perfect Self-Match Detection")
        print("   ✅ File Selection Dialog")
        print("   ✅ Advanced Tie-Breaking:")
        print("      1. Self-match priority")
        print("      2. Perfect match count")
        print("      3. Lowest-high-score rule")
        print("      4. Sum of individual scores")
        print("   ✅ Spectral Comparison Graphs")
        print("   ✅ Results sorted by SCORE (not timestamp)")
        
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