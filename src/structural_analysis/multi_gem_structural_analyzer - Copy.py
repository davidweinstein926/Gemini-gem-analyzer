#!/usr/bin/env python3
"""
ULTIMATE MULTI-GEM STRUCTURAL ANALYZER - CONSOLIDATED POWERHOUSE
Incorporates:
- Advanced database matching with sophisticated scoring algorithms
- Visualization plots (from unified_structural_analyzer.py)
- Diagnostic mode (from unified_structural_analyzer.py)
- Configurable input sources (archive OR current)
- Full GUI interface with enhanced capabilities

Usage:
- Option 4 (main.py): Current work files - data/structural_data/*.csv
- Option 8 (main.py): Archive files - data/structural(archive)/*.csv
- Standalone: GUI mode with source selection
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import re
import sys
import sqlite3
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
import json
import math
import traceback
warnings.filterwarnings('ignore')

# Enhanced imports with graceful fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
    print("✅ Matplotlib available - visualizations enabled")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ Matplotlib not available - visualizations disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
    print("✅ Seaborn available - enhanced styling enabled")
except ImportError:
    HAS_SEABORN = False
    print("⚠️ Seaborn not available - using basic matplotlib styling")

try:
    from scipy.spatial.distance import euclidean, cosine
    from scipy import stats
    from scipy.signal import find_peaks
    HAS_SCIPY = True
    print("✅ SciPy available - advanced algorithms enabled")
except ImportError:
    HAS_SCIPY = False
    print("⚠️ SciPy not available - using basic algorithms")

class UltimateMultiGemStructuralAnalyzer:
    """Ultimate structural analyzer with all advanced features consolidated"""
    
    def __init__(self, mode="gui", input_source="archive"):
        """
        Initialize analyzer
        
        Args:
            mode: "gui" (interactive), "auto" (programmatic), "diagnostic" (testing)
            input_source: "archive" (Option 8), "current" (Option 4)
        """
        self.mode = mode
        self.input_source = input_source
        
        # Only create GUI if in GUI mode
        if self.mode == "gui":
            self.root = tk.Tk()
            self.root.title(f"Ultimate Multi-Gem Structural Analyzer - {input_source.title()} Mode")
            self.root.geometry("1400x900")
        else:
            self.root = None
            print(f"🔬 Ultimate Analyzer initialized in {mode} mode")
            print(f"📁 Input source: {input_source}")
        
        # Find project root
        self.project_root = self.find_project_root()
        
        # Configure input path based on source
        if input_source == "archive":
            self.input_path = self.project_root / "data" / "structural(archive)"
            self.analysis_name = "Archive Analysis (Option 8)"
            self.description = "completed files (already in database)"
        elif input_source == "current":
            self.input_path = self.project_root / "data" / "structural_data"
            self.analysis_name = "Current Work Analysis (Option 4)"
            self.description = "work-in-progress files (not yet in database)"
        else:
            raise ValueError(f"Invalid input_source: {input_source}")
        
        # Check for modern databases - UPDATED PATHS
        self.sqlite_db_path = self.project_root / "database" / "structural_spectra" / "gemini_structural.db"
        self.csv_db_path = self.project_root / "database" / "structural_spectra" / "gemini_structural_unified.csv"
        self.database_type = None
        self.database_path = None
        
        # Data structures
        self.gem_groups = {}
        self.selected_gems = {}
        self.analysis_results = {}
        self.spectral_features = {}
        
        # Advanced scoring parameters (enhanced from original)
        self.feature_weights = {
            'Mound': 1.0,      # Most important - diagnostic
            'Peak': 0.9,       # Very important - sharp features
            'Trough': 0.8,     # Important - absorption bands
            'Plateau': 0.7,    # Moderately important
            'Shoulder': 0.6,   # Less important
            'Valley': 0.5,     # Least important
            'Baseline': 0.3    # Reference only
        }
        
        self.light_weights = {
            'Halogen': 1.0,    # Most reliable - broad spectrum
            'Laser': 0.9,      # Very reliable - high resolution
            'UV': 0.8          # Good but more specialized
        }
        
        self.wavelength_tolerances = {
            'Peak': 1.0,       # Tight tolerance for sharp features
            'Mound': 3.0,      # Looser for broad features
            'Trough': 2.0,     # Medium tolerance
            'Plateau': 4.0,    # Broadest tolerance
            'Shoulder': 2.5,
            'Valley': 1.5
        }
        
        # Setup components
        if self.mode == "gui":
            self.setup_gui()
        
        # Always check databases and scan directory
        self.check_databases()
        self.scan_input_directory()
    
    def find_project_root(self):
        """Find the project root directory by walking up from script location"""
        current_path = Path(__file__).parent
        
        project_indicators = [
            'database/structural_spectra',
            'data', 'src', 'outputs'
        ]
        
        for level in range(5):
            indicator_count = 0
            for indicator in project_indicators:
                if (current_path / indicator).exists():
                    indicator_count += 1
            
            if indicator_count >= 2:
                if self.mode != "gui":
                    print(f"🎯 Project root found: {current_path}")
                return current_path
            
            parent = current_path.parent
            if parent == current_path:
                break
            current_path = parent
        
        fallback_path = Path(__file__).parent.parent.parent
        if self.mode != "gui":
            print(f"🎯 Using fallback project root: {fallback_path}")
        return fallback_path
    
    def check_databases(self):
        """Check for modern databases with priority order"""
        available_dbs = []
        
        if self.sqlite_db_path.exists():
            available_dbs.append(("sqlite", self.sqlite_db_path))
        if self.csv_db_path.exists():
            available_dbs.append(("csv", self.csv_db_path))
        
        if not available_dbs:
            error_msg = (f"No modern databases found!\n\n"
                        f"Expected files:\n"
                        f"- database/structural_spectra/gemini_structural.db\n"
                        f"- database/structural_spectra/gemini_structural_unified.csv\n\n"
                        f"Please ensure at least one exists.")
            
            if self.mode == "gui":
                messagebox.showerror("Database Error", error_msg)
            else:
                print(f"❌ {error_msg}")
            return False
        
        # Prefer SQLite over CSV
        if len(available_dbs) > 1:
            self.database_type, self.database_path = available_dbs[0]  # SQLite first
        else:
            self.database_type, self.database_path = available_dbs[0]
        
        if self.mode != "gui":
            print(f"✅ Using {self.database_type.upper()} database: {self.database_path.name}")
        
        return self.validate_database()
    
    def validate_database(self):
        """Validate the selected database has proper structure"""
        try:
            if self.database_type == "sqlite":
                with sqlite3.connect(self.database_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    
                    if not tables:
                        error_msg = "SQLite database exists but contains no tables."
                        if self.mode == "gui":
                            messagebox.showwarning("Database Warning", error_msg)
                        else:
                            print(f"⚠️ {error_msg}")
                        return False
                    
                    table_name = tables[0][0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    if count == 0:
                        error_msg = f"SQLite database table '{table_name}' is empty."
                        if self.mode == "gui":
                            messagebox.showwarning("Database Warning", error_msg)
                        else:
                            print(f"⚠️ {error_msg}")
                        return False
                    
                    if self.mode != "gui":
                        print(f"✅ SQLite database validated: {count:,} records in table '{table_name}'")
                    
            else:  # CSV
                df = pd.read_csv(self.database_path)
                
                if df.empty:
                    error_msg = "CSV database is empty."
                    if self.mode == "gui":
                        messagebox.showwarning("Database Warning", error_msg)
                    else:
                        print(f"⚠️ {error_msg}")
                    return False
                
                required_cols = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
                identifier_col = None
                for col in required_cols:
                    if col in df.columns:
                        identifier_col = col
                        break
                
                if not identifier_col:
                    error_msg = f"CSV database missing identifier column."
                    if self.mode == "gui":
                        messagebox.showerror("Database Error", error_msg)
                    else:
                        print(f"❌ {error_msg}")
                    return False
                
                if self.mode != "gui":
                    print(f"✅ CSV database validated: {len(df):,} records with identifier column '{identifier_col}'")
            
            return True
            
        except Exception as e:
            error_msg = f"Database validation failed: {e}"
            if self.mode == "gui":
                messagebox.showerror("Database Error", error_msg)
            else:
                print(f"❌ {error_msg}")
            return False
    
    def scan_input_directory(self):
        """Scan input directory and group gems by base_id"""
        if not self.input_path.exists():
            error_msg = f"Input directory not found: {self.input_path}"
            if self.mode == "gui":
                messagebox.showerror("Error", error_msg)
            else:
                print(f"❌ {error_msg}")
            return
        
        csv_files = list(self.input_path.glob("*.csv"))
        if not csv_files:
            warning_msg = f"No .csv files found in: {self.input_path}"
            if self.mode == "gui":
                messagebox.showwarning("Warning", warning_msg)
            else:
                print(f"⚠️ {warning_msg}")
            return
        
        if self.mode != "gui":
            print(f"📁 Scanning {self.input_source} directory: {self.input_path}")
            print(f"📄 Found {len(csv_files)} CSV files")
        
        self.gem_groups.clear()
        
        for file_path in csv_files:
            gem_info = self.parse_structural_filename(file_path.name)
            base_id = gem_info['base_id']
            
            if base_id not in self.gem_groups:
                self.gem_groups[base_id] = {
                    'files': {'B': [], 'L': [], 'U': []},
                    'file_paths': {'B': [], 'L': [], 'U': []}
                }
            
            light_source = gem_info['light_source']
            if light_source in ['B', 'L', 'U']:
                self.gem_groups[base_id]['files'][light_source].append(gem_info)
                self.gem_groups[base_id]['file_paths'][light_source].append(file_path)
        
        if self.mode == "gui":
            self.populate_gem_list()
        else:
            # Show summary for non-GUI modes
            complete_gems = sum(1 for data in self.gem_groups.values() 
                              if all(len(data['files'][ls]) > 0 for ls in ['B', 'L', 'U']))
            print(f"💎 Found {len(self.gem_groups)} unique gems ({complete_gems} complete with B+L+U)")
    
    def parse_structural_filename(self, filename: str) -> dict:
        """Parse structural CSV filename to extract components"""
        stem = Path(filename).stem
        
        # Handle timestamped files
        if '_20' in stem:
            parts = stem.split('_20')
            stem = parts[0]
        
        # Remove _structural suffix if present
        if stem.endswith('_structural'):
            stem = stem[:-11]
        
        # Handle structured format with explicit light source names
        if '_halogen_' in stem.lower():
            prefix = stem.split('_halogen_')[0]
            match = re.match(r'^(\d+)', prefix)
            gem_id = match.group(1) if match else prefix
            return {
                'base_id': gem_id,
                'light_source': 'B',
                'light_full': 'Halogen',
                'filename': filename,
                'is_valid': True
            }
        elif '_laser_' in stem.lower():
            prefix = stem.split('_laser_')[0]
            match = re.match(r'^(\d+)', prefix)
            gem_id = match.group(1) if match else prefix
            return {
                'base_id': gem_id,
                'light_source': 'L',
                'light_full': 'Laser',
                'filename': filename,
                'is_valid': True
            }
        elif '_uv_' in stem.lower():
            prefix = stem.split('_uv_')[0]
            match = re.match(r'^(\d+)', prefix)
            gem_id = match.group(1) if match else prefix
            return {
                'base_id': gem_id,
                'light_source': 'U',
                'light_full': 'UV',
                'filename': filename,
                'is_valid': True
            }
        else:
            # Standard format: [prefix]base_id + light_source + orientation + scan_number
            match = re.match(r'^([A-Za-z]*\d+)([BLU])([CP]?)(\d+)', stem)
            if match:
                prefix, light, orientation, scan = match.groups()
                
                # Extract numeric gem ID
                gem_match = re.search(r'(\d+)', prefix)
                gem_id = gem_match.group(1) if gem_match else prefix
                
                return {
                    'base_id': gem_id,
                    'light_source': light.upper(),
                    'light_full': {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}[light.upper()],
                    'orientation': orientation.upper(),
                    'scan_number': int(scan),
                    'filename': filename,
                    'is_valid': True
                }
        
        return {
            'base_id': stem,
            'light_source': 'Unknown',
            'filename': filename,
            'is_valid': False
        }
    
    def setup_gui(self):
        """Setup the enhanced GUI interface"""
        # Configure style
        try:
            self.root.tk.call("source", "azure.tcl")
            self.root.tk.call("set_theme", "dark")
        except:
            pass  # Fallback to default theme
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title with mode info
        title_text = f"Ultimate Multi-Gem Structural Analyzer - {self.analysis_name}"
        title_label = ttk.Label(main_frame, text=title_text, font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill='x', pady=(0, 10))
        
        # Directory and database info
        info_text = (f"📁 Input: {self.input_path}\n"
                    f"🗄️  Database: {self.database_path}\n"
                    f"📊 Analysis: {self.description}")
        ttk.Label(info_frame, text=info_text, font=('Arial', 10)).pack(side='left')
        
        # Feature badges
        badges_frame = ttk.Frame(info_frame)
        badges_frame.pack(side='right')
        
        badges = []
        if HAS_MATPLOTLIB: badges.append("📈 Plots")
        if HAS_SCIPY: badges.append("🔬 Advanced")
        badges.append("🎯 Weighting")
        badges.append("🚀 Ultimate")
        
        ttk.Label(badges_frame, text=" | ".join(badges), 
                 font=('Arial', 9), foreground='green').pack()
        
        # Main content frame with notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill='both', expand=True, pady=(0, 10))
        
        # Tab 1: Analysis Selection
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="🔬 Analysis Selection")
        
        # Content frame for analysis tab
        content_frame = ttk.Frame(self.analysis_frame, padding="10")
        content_frame.pack(fill='both', expand=True)
        
        # Left side - Available gems
        left_frame = ttk.LabelFrame(content_frame, text="Available Structural Gems", padding="5")
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Gem listbox with scrollbar
        gem_list_frame = ttk.Frame(left_frame)
        gem_list_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.gem_listbox = tk.Listbox(gem_list_frame, height=15)
        gem_scrollbar = ttk.Scrollbar(gem_list_frame, orient="vertical", command=self.gem_listbox.yview)
        self.gem_listbox.configure(yscrollcommand=gem_scrollbar.set)
        
        self.gem_listbox.pack(side='left', fill='both', expand=True)
        gem_scrollbar.pack(side='right', fill='y')
        
        # Selection controls
        select_frame = ttk.Frame(left_frame)
        select_frame.pack(fill='x', pady=(0, 5))
        
        ttk.Button(select_frame, text="Select Files", command=self.select_gem_files).pack(side='left', padx=(0, 5))
        ttk.Button(select_frame, text="Select All Complete", command=self.select_all_complete).pack(side='left')
        
        # Right side - Selected gems
        right_frame = ttk.LabelFrame(content_frame, text="Selected for Ultimate Analysis", padding="5")
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Selected listbox with scrollbar
        selected_list_frame = ttk.Frame(right_frame)
        selected_list_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.selected_listbox = tk.Listbox(selected_list_frame, height=15)
        selected_scrollbar = ttk.Scrollbar(selected_list_frame, orient="vertical", command=self.selected_listbox.yview)
        self.selected_listbox.configure(yscrollcommand=selected_scrollbar.set)
        
        self.selected_listbox.pack(side='left', fill='both', expand=True)
        selected_scrollbar.pack(side='right', fill='y')
        
        # Remove button
        ttk.Button(right_frame, text="Remove Selected", command=self.remove_selected).pack(pady=(0, 5))
        
        # Tab 2: Visualization (only if matplotlib available)
        if HAS_MATPLOTLIB:
            self.viz_frame = ttk.Frame(self.notebook)
            self.notebook.add(self.viz_frame, text="📈 Visualizations")
            self.setup_visualization_tab()
        
        # Tab 3: Diagnostics
        self.diagnostic_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.diagnostic_frame, text="🔧 Diagnostics")
        self.setup_diagnostic_tab()
        
        # Bottom control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', pady=10)
        
        # Left side controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side='left')
        
        ttk.Button(left_controls, text="Clear All", command=self.clear_all).pack(side='left', padx=(0, 5))
        ttk.Button(left_controls, text="Refresh", command=self.scan_input_directory).pack(side='left', padx=(0, 5))
        ttk.Button(left_controls, text="Run Diagnostics", command=self.run_diagnostics).pack(side='left', padx=(0, 5))
        
        # Right side controls
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side='right')
        
        ttk.Button(right_controls, text="Close", command=self.close_application).pack(side='right', padx=(5, 0))
        ttk.Button(right_controls, text="🚀 Start Ultimate Analysis", 
                  command=self.start_ultimate_analysis, style='Accent.TButton').pack(side='right', padx=(5, 0))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Ultimate Multi-Gem Structural Analyzer")
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill='x')
        
        ttk.Label(status_frame, textvariable=self.status_var).pack(side='left')
        
        # Progress bar (hidden initially)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, mode='determinate')
        self.progress_bar.pack(side='right', fill='x', expand=True, padx=(10, 0))
        self.progress_bar.pack_forget()  # Hide initially
    
    def setup_visualization_tab(self):
        """Setup the visualization tab with matplotlib integration"""
        if not HAS_MATPLOTLIB:
            ttk.Label(self.viz_frame, text="Matplotlib not available - install with: pip install matplotlib").pack(pady=20)
            return
        
        viz_main = ttk.Frame(self.viz_frame, padding="10")
        viz_main.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(viz_main, text="Enhanced Visualization Dashboard", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Control frame
        viz_controls = ttk.Frame(viz_main)
        viz_controls.pack(fill='x', pady=(0, 10))
        
        ttk.Button(viz_controls, text="Generate Preview Plots", 
                  command=self.generate_preview_plots).pack(side='left', padx=(0, 10))
        ttk.Button(viz_controls, text="Save All Plots", 
                  command=self.save_all_plots).pack(side='left', padx=(0, 10))
        
        # Plot options
        options_frame = ttk.LabelFrame(viz_controls, text="Plot Options", padding="5")
        options_frame.pack(side='right')
        
        self.plot_overlays = tk.BooleanVar(value=True)
        self.plot_peaks = tk.BooleanVar(value=True)
        self.plot_statistics = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="Overlays", variable=self.plot_overlays).pack(side='left')
        ttk.Checkbutton(options_frame, text="Peaks", variable=self.plot_peaks).pack(side='left')
        ttk.Checkbutton(options_frame, text="Stats", variable=self.plot_statistics).pack(side='left')
        
        # Canvas frame for plots
        self.canvas_frame = ttk.Frame(viz_main)
        self.canvas_frame.pack(fill='both', expand=True)
        
        # Initial placeholder
        ttk.Label(self.canvas_frame, text="Select gems and run analysis to generate plots", 
                 font=('Arial', 12)).pack(expand=True)
    
    def setup_diagnostic_tab(self):
        """Setup the diagnostic tab with system testing capabilities"""
        diag_main = ttk.Frame(self.diagnostic_frame, padding="10")
        diag_main.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(diag_main, text="System Diagnostics & Health Check", 
                 font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        # Control frame
        diag_controls = ttk.Frame(diag_main)
        diag_controls.pack(fill='x', pady=(0, 10))
        
        ttk.Button(diag_controls, text="🔧 Run Full Diagnostics", 
                  command=self.run_full_diagnostics).pack(side='left', padx=(0, 10))
        ttk.Button(diag_controls, text="📊 Check Database", 
                  command=self.check_database_detailed).pack(side='left', padx=(0, 10))
        ttk.Button(diag_controls, text="🗂️ Check Files", 
                  command=self.check_files_detailed).pack(side='left', padx=(0, 10))
        
        # Results frame with scrollable text
        results_frame = ttk.LabelFrame(diag_main, text="Diagnostic Results", padding="5")
        results_frame.pack(fill='both', expand=True)
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True)
        
        self.diag_text = tk.Text(text_frame, wrap='word', font=('Consolas', 10))
        diag_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.diag_text.yview)
        self.diag_text.configure(yscrollcommand=diag_scrollbar.set)
        
        self.diag_text.pack(side='left', fill='both', expand=True)
        diag_scrollbar.pack(side='right', fill='y')
        
        # Insert initial diagnostic info
        self.diag_text.insert('end', "🔬 Ultimate Multi-Gem Structural Analyzer - Diagnostic Mode\n")
        self.diag_text.insert('end', "=" * 60 + "\n\n")
        self.diag_text.insert('end', f"Input Source: {self.input_source}\n")
        self.diag_text.insert('end', f"Input Path: {self.input_path}\n")
        self.diag_text.insert('end', f"Database: {self.database_path}\n")
        self.diag_text.insert('end', f"Database Type: {self.database_type}\n\n")
        
        # Library availability
        self.diag_text.insert('end', "📚 Library Availability:\n")
        self.diag_text.insert('end', f"✅ Matplotlib: {HAS_MATPLOTLIB}\n" if HAS_MATPLOTLIB else "❌ Matplotlib: False\n")
        self.diag_text.insert('end', f"✅ Seaborn: {HAS_SEABORN}\n" if HAS_SEABORN else "❌ Seaborn: False\n")
        self.diag_text.insert('end', f"✅ SciPy: {HAS_SCIPY}\n" if HAS_SCIPY else "❌ SciPy: False\n")
        
        self.diag_text.insert('end', "\nReady for diagnostics. Click buttons above to run tests.\n")
        self.diag_text.config(state='disabled')
    
    def populate_gem_list(self):
        """Populate the gem list with available structural gems"""
        self.gem_listbox.delete(0, tk.END)
        
        complete_count = 0
        partial_count = 0
        
        for base_id, data in sorted(self.gem_groups.items()):
            # Count files per light source
            b_count = len(data['files']['B'])
            l_count = len(data['files']['L'])
            u_count = len(data['files']['U'])
            total = b_count + l_count + u_count
            
            # Create display string
            sources = []
            if b_count > 0: sources.append(f"B({b_count})")
            if l_count > 0: sources.append(f"L({l_count})")
            if u_count > 0: sources.append(f"U({u_count})")
            
            is_complete = b_count > 0 and l_count > 0 and u_count > 0
            if is_complete:
                complete_count += 1
                status = "🟢 COMPLETE"
            else:
                partial_count += 1
                status = "🟡 Partial"
            
            display_text = f"{status} Gem {base_id} - {'+'.join(sources)} - {total} files"
            self.gem_listbox.insert(tk.END, display_text)
        
        self.status_var.set(f"Found {len(self.gem_groups)} gems ({complete_count} complete, {partial_count} partial)")
    
    def select_gem_files(self):
        """Select specific structural files for a gem"""
        selection = self.gem_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a gem from the list.")
            return
        
        # Get selected gem
        gem_list = list(self.gem_groups.keys())
        gem_index = selection[0]
        base_id = sorted(gem_list)[gem_index]
        
        if base_id in self.selected_gems:
            messagebox.showinfo("Already Selected", f"Gem {base_id} is already selected.")
            return
        
        # Open enhanced file selection dialog
        self.open_enhanced_file_dialog(base_id)
    
    def select_all_complete(self):
        """Select all complete gems (have B+L+U) automatically"""
        complete_gems = []
        
        for base_id, data in self.gem_groups.items():
            b_count = len(data['files']['B'])
            l_count = len(data['files']['L'])
            u_count = len(data['files']['U'])
            
            if b_count > 0 and l_count > 0 and u_count > 0:
                complete_gems.append(base_id)
        
        if not complete_gems:
            messagebox.showinfo("No Complete Gems", "No gems with complete B+L+U coverage found.")
            return
        
        # Confirm selection
        if not messagebox.askyesno("Select All Complete", 
                                  f"Select all {len(complete_gems)} complete gems for analysis?\n\n"
                                  f"This will use the first file for each light source."):
            return
        
        # Add all complete gems
        added_count = 0
        for base_id in complete_gems:
            if base_id not in self.selected_gems:
                gem_data = self.gem_groups[base_id]
                
                # Auto-select first file for each light source
                selected_files = {}
                selected_paths = {}
                
                for light_source in ['B', 'L', 'U']:
                    if gem_data['files'][light_source]:
                        selected_files[light_source] = gem_data['files'][light_source][0]
                        selected_paths[light_source] = gem_data['file_paths'][light_source][0]
                
                self.selected_gems[base_id] = {
                    'selected_files': selected_files,
                    'selected_paths': selected_paths,
                    'options': {
                        'normalize': True,
                        'feature_weighting': True,
                        'multipoint_analysis': True
                    }
                }
                added_count += 1
        
        self.update_selected_display()
        messagebox.showinfo("Selection Complete", f"Added {added_count} complete gems to analysis queue.")
    
    def open_enhanced_file_dialog(self, base_id):
        """Open enhanced file selection dialog with feature preview"""
        gem_data = self.gem_groups[base_id]
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Ultimate File Selection - Gem {base_id}")
        dialog.geometry("900x750")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 450
        y = (dialog.winfo_screenheight() // 2) - 375
        dialog.geometry(f"900x750+{x}+{y}")
        
        # Title
        ttk.Label(dialog, text=f"Ultimate Structural Analysis - Gem {base_id}", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Info
        info_text = (f"Advanced feature weighting and multi-point analysis\n"
                    f"Source: {self.analysis_name}")
        ttk.Label(dialog, text=info_text, font=('Arial', 10), foreground='blue').pack(pady=5)
        
        # File selection variables
        selections = {}
        
        # Create enhanced selection area for each light source
        for light_source in ['B', 'L', 'U']:
            files_list = gem_data['files'][light_source]
            if not files_list:
                continue
            
            light_full = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}[light_source]
            weight = self.light_weights.get(light_full, 1.0)
            
            # Frame for this light source
            frame = ttk.LabelFrame(dialog, 
                                  text=f"{light_full} ({light_source}) - Weight: {weight} - {len(files_list)} files", 
                                  padding="5")
            frame.pack(fill='x', padx=10, pady=5)
            
            # Selection variable
            selections[light_source] = tk.StringVar(value="")
            
            # Skip option
            ttk.Radiobutton(frame, text="Skip this light source", 
                           variable=selections[light_source], value="").pack(anchor='w')
            
            # File options with enhanced descriptions
            for i, file_info in enumerate(files_list):
                file_path = gem_data['file_paths'][light_source][i]
                file_desc = self.analyze_file_features(file_path)
                text = f"{file_info['filename']} - {file_desc}"
                ttk.Radiobutton(frame, text=text, 
                               variable=selections[light_source], value=str(i)).pack(anchor='w')
        
        # Analysis options frame
        options_frame = ttk.LabelFrame(dialog, text="Ultimate Analysis Options", padding="5")
        options_frame.pack(fill='x', padx=10, pady=10)
        
        # Normalization option
        normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Apply spectral normalization", 
                       variable=normalize_var).pack(anchor='w')
        
        # Feature weighting option
        weighting_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use advanced feature weighting", 
                       variable=weighting_var).pack(anchor='w')
        
        # Multi-point analysis option
        multipoint_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Enable multi-point feature analysis", 
                       variable=multipoint_var).pack(anchor='w')
        
        # Visualization option (if available)
        if HAS_MATPLOTLIB:
            visualization_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(options_frame, text="Generate visualization plots", 
                           variable=visualization_var).pack(anchor='w')
        else:
            visualization_var = tk.BooleanVar(value=False)
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        def confirm_selection():
            selected_files = {}
            selected_paths = {}
            
            for light_source in ['B', 'L', 'U']:
                if light_source in selections and selections[light_source].get():
                    try:
                        index = int(selections[light_source].get())
                        files_list = gem_data['files'][light_source]
                        paths_list = gem_data['file_paths'][light_source]
                        
                        if 0 <= index < len(files_list):
                            selected_files[light_source] = files_list[index]
                            selected_paths[light_source] = paths_list[index]
                    except (ValueError, IndexError):
                        continue
            
            if not selected_files:
                messagebox.showwarning("No Selection", "Please select at least one structural file.")
                return
            
            # Add to selected gems with analysis options
            self.selected_gems[base_id] = {
                'selected_files': selected_files,
                'selected_paths': selected_paths,
                'options': {
                    'normalize': normalize_var.get(),
                    'feature_weighting': weighting_var.get(),
                    'multipoint_analysis': multipoint_var.get(),
                    'visualization': visualization_var.get()
                }
            }
            
            self.update_selected_display()
            dialog.destroy()
        
        def cancel_selection():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Cancel", command=cancel_selection).pack(side='right', padx=(5, 0))
        ttk.Button(button_frame, text="Confirm Ultimate Selection", command=confirm_selection).pack(side='right')
    
    def analyze_file_features(self, file_path):
        """Analyze file to show enhanced feature preview"""
        try:
            df = pd.read_csv(file_path, nrows=20)  # Quick preview
            
            feature_types = []
            
            # Check for feature columns
            if 'Feature' in df.columns:
                features = df['Feature'].unique()
                feature_types = [f for f in features if f in self.feature_weights]
                
            if 'feature_type' in df.columns:
                features = df['feature_type'].unique()
                feature_types.extend([f for f in features if f in self.feature_weights])
            
            if feature_types:
                # Calculate weighted importance
                total_weight = sum(self.feature_weights.get(ft, 0) for ft in feature_types)
                return f"{len(feature_types)} features, weight: {total_weight:.1f}"
            else:
                return f"{len(df)} structural points"
                
        except Exception as e:
            return "structural data"
    
    def update_selected_display(self):
        """Update the selected gems display with enhanced info"""
        self.selected_listbox.delete(0, tk.END)
        
        for base_id, data in self.selected_gems.items():
            files = data['selected_files']
            options = data.get('options', {})
            light_sources = sorted(files.keys())
            
            file_details = []
            for ls in light_sources:
                file_info = files[ls]
                file_details.append(f"{ls}:{file_info.get('orientation', 'X')}{file_info.get('scan_number', '1')}")
            
            # Show analysis options
            option_indicators = []
            if options.get('normalize', False): option_indicators.append("N")
            if options.get('feature_weighting', False): option_indicators.append("W")
            if options.get('multipoint_analysis', False): option_indicators.append("M")
            if options.get('visualization', False): option_indicators.append("V")
            
            options_str = f"[{'+'.join(option_indicators)}]" if option_indicators else ""
            
            display_text = f"🎯 Gem {base_id} ({'+'.join(light_sources)}) {options_str} - {' '.join(file_details)}"
            self.selected_listbox.insert(tk.END, display_text)
        
        self.status_var.set(f"Selected {len(self.selected_gems)} gems for ultimate analysis")
    
    def remove_selected(self):
        """Remove selected gem from analysis"""
        selection = self.selected_listbox.curselection()
        if not selection:
            return
        
        gem_ids = list(self.selected_gems.keys())
        index = selection[0]
        if index < len(gem_ids):
            base_id = gem_ids[index]
            del self.selected_gems[base_id]
            self.update_selected_display()
    
    def clear_all(self):
        """Clear all selections"""
        self.selected_gems.clear()
        self.update_selected_display()
    
    def close_application(self):
        """Close the application properly"""
        if messagebox.askyesno("Close Application", "Close Ultimate Multi-Gem Structural Analyzer?"):
            self.root.quit()
            self.root.destroy()
    
    # DIAGNOSTIC METHODS (from unified_structural_analyzer.py)
    
    def run_diagnostics(self):
        """Run quick diagnostics and show results in status"""
        try:
            # Quick checks
            db_ok = self.validate_database()
            files_ok = len(self.gem_groups) > 0
            libs_ok = HAS_MATPLOTLIB and HAS_SCIPY
            
            status = f"Diagnostics: DB={'✅' if db_ok else '❌'} Files={'✅' if files_ok else '❌'} Libs={'✅' if libs_ok else '⚠️'}"
            self.status_var.set(status)
            
            if self.mode == "gui":
                self.notebook.select(self.diagnostic_frame)
                
        except Exception as e:
            self.status_var.set(f"Diagnostic error: {e}")
    
    def run_full_diagnostics(self):
        """Run comprehensive diagnostics and display in diagnostic tab"""
        self.diag_text.config(state='normal')
        self.diag_text.delete('1.0', 'end')
        
        self.diag_text.insert('end', "🔧 FULL SYSTEM DIAGNOSTICS\n")
        self.diag_text.insert('end', "=" * 50 + "\n\n")
        
        # Test 1: Directory structure
        self.diag_text.insert('end', "1️⃣ Testing directory structure...\n")
        structure_ok = self.input_path.exists()
        csv_files = list(self.input_path.glob("*.csv")) if structure_ok else []
        
        if structure_ok and csv_files:
            self.diag_text.insert('end', f"   ✅ Input directory exists: {self.input_path}\n")
            self.diag_text.insert('end', f"   ✅ Found {len(csv_files)} CSV files\n")
        else:
            self.diag_text.insert('end', f"   ❌ Directory issue: {self.input_path}\n")
        
        # Test 2: File parsing
        self.diag_text.insert('end', "\n2️⃣ Testing file parsing...\n")
        parsing_ok = len(self.gem_groups) > 0
        
        if parsing_ok:
            complete_gems = sum(1 for data in self.gem_groups.values() 
                              if all(len(data['files'][ls]) > 0 for ls in ['B', 'L', 'U']))
            self.diag_text.insert('end', f"   ✅ Parsed {len(self.gem_groups)} unique gems\n")
            self.diag_text.insert('end', f"   ✅ Found {complete_gems} complete gems (B+L+U)\n")
        else:
            self.diag_text.insert('end', f"   ❌ No gems parsed successfully\n")
        
        # Test 3: Database connectivity
        self.diag_text.insert('end', "\n3️⃣ Testing database connectivity...\n")
        db_ok = self.validate_database()
        
        if db_ok:
            self.diag_text.insert('end', f"   ✅ Database accessible: {self.database_type.upper()}\n")
            self.diag_text.insert('end', f"   ✅ Path: {self.database_path.name}\n")
        else:
            self.diag_text.insert('end', f"   ❌ Database connection failed\n")
        
        # Test 4: Library availability
        self.diag_text.insert('end', "\n4️⃣ Testing library availability...\n")
        
        self.diag_text.insert('end', f"   {'✅' if HAS_MATPLOTLIB else '❌'} Matplotlib: {HAS_MATPLOTLIB}\n")
        self.diag_text.insert('end', f"   {'✅' if HAS_SEABORN else '❌'} Seaborn: {HAS_SEABORN}\n")
        self.diag_text.insert('end', f"   {'✅' if HAS_SCIPY else '❌'} SciPy: {HAS_SCIPY}\n")
        
        if not HAS_MATPLOTLIB:
            self.diag_text.insert('end', f"      💡 Install: pip install matplotlib\n")
        if not HAS_SCIPY:
            self.diag_text.insert('end', f"      💡 Install: pip install scipy\n")
        
        # Test 5: Data loading test
        self.diag_text.insert('end', "\n5️⃣ Testing data loading...\n")
        loading_ok = False
        
        if csv_files:
            try:
                test_file = csv_files[0]
                df = pd.read_csv(test_file, nrows=5)
                loading_ok = not df.empty
                self.diag_text.insert('end', f"   ✅ Successfully loaded test file: {test_file.name}\n")
                self.diag_text.insert('end', f"   ✅ Columns: {', '.join(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}\n")
            except Exception as e:
                self.diag_text.insert('end', f"   ❌ Data loading error: {e}\n")
        else:
            self.diag_text.insert('end', f"   ❌ No files available for testing\n")
        
        # Summary
        self.diag_text.insert('end', "\n📋 DIAGNOSTIC SUMMARY\n")
        self.diag_text.insert('end', "-" * 30 + "\n")
        
        tests = [
            ("Directory Structure", structure_ok),
            ("File Parsing", parsing_ok),
            ("Database Connectivity", db_ok),
            ("Data Loading", loading_ok),
            ("Matplotlib (Visualization)", HAS_MATPLOTLIB),
            ("SciPy (Advanced Analysis)", HAS_SCIPY)
        ]
        
        for test_name, result in tests:
            status = "✅ PASS" if result else "❌ FAIL"
            self.diag_text.insert('end', f"{status} {test_name}\n")
        
        # Overall assessment
        core_ok = all([structure_ok, parsing_ok, db_ok, loading_ok])
        
        self.diag_text.insert('end', f"\n🎯 OVERALL ASSESSMENT\n")
        
        if core_ok:
            self.diag_text.insert('end', f"✅ CORE FUNCTIONALITY READY\n")
            self.diag_text.insert('end', f"   System can perform ultimate structural analysis\n")
            
            if not HAS_MATPLOTLIB:
                self.diag_text.insert('end', f"   📊 Note: Install matplotlib for visualization features\n")
            if not HAS_SCIPY:
                self.diag_text.insert('end', f"   🔬 Note: Install scipy for advanced algorithms\n")
        else:
            self.diag_text.insert('end', f"❌ CRITICAL ISSUES DETECTED\n")
            self.diag_text.insert('end', f"   Address failed tests above before running analysis\n")
        
        self.diag_text.insert('end', f"\nDiagnostics completed: {datetime.now().strftime('%H:%M:%S')}\n")
        self.diag_text.config(state='disabled')
        
        # Update status
        passed_tests = sum(1 for _, result in tests if result)
        self.status_var.set(f"Diagnostics: {passed_tests}/{len(tests)} tests passed")
    
    def check_database_detailed(self):
        """Detailed database analysis"""
        self.diag_text.config(state='normal')
        self.diag_text.insert('end', f"\n🗄️ DETAILED DATABASE ANALYSIS\n")
        self.diag_text.insert('end', "-" * 40 + "\n")
        
        try:
            if self.database_type == "sqlite":
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                
                # Get table info
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                self.diag_text.insert('end', f"Database Type: SQLite\n")
                self.diag_text.insert('end', f"Tables: {len(tables)}\n")
                
                for table in tables:
                    table_name = table[0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    # Get column info
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = cursor.fetchall()
                    
                    self.diag_text.insert('end', f"\nTable: {table_name}\n")
                    self.diag_text.insert('end', f"  Records: {count:,}\n")
                    self.diag_text.insert('end', f"  Columns: {len(columns)}\n")
                    
                    # Show first few columns
                    for i, col in enumerate(columns[:5]):
                        self.diag_text.insert('end', f"    {col[1]} ({col[2]})\n")
                    if len(columns) > 5:
                        self.diag_text.insert('end', f"    ... and {len(columns)-5} more\n")
                
                conn.close()
                
            else:  # CSV
                df = pd.read_csv(self.database_path)
                
                self.diag_text.insert('end', f"Database Type: CSV\n")
                self.diag_text.insert('end', f"Records: {len(df):,}\n")
                self.diag_text.insert('end', f"Columns: {len(df.columns)}\n")
                
                # Show column info
                for i, col in enumerate(df.columns[:10]):
                    dtype = str(df[col].dtype)
                    non_null = df[col].notna().sum()
                    self.diag_text.insert('end', f"  {col}: {dtype} ({non_null:,} non-null)\n")
                
                if len(df.columns) > 10:
                    self.diag_text.insert('end', f"  ... and {len(df.columns)-10} more columns\n")
                
        except Exception as e:
            self.diag_text.insert('end', f"❌ Database analysis error: {e}\n")
        
        self.diag_text.config(state='disabled')
    
    def check_files_detailed(self):
        """Detailed file analysis"""
        self.diag_text.config(state='normal')
        self.diag_text.insert('end', f"\n🗂️ DETAILED FILE ANALYSIS\n")
        self.diag_text.insert('end', "-" * 40 + "\n")
        
        try:
            csv_files = list(self.input_path.glob("*.csv"))
            
            self.diag_text.insert('end', f"Input Directory: {self.input_path}\n")
            self.diag_text.insert('end', f"Total CSV Files: {len(csv_files)}\n")
            
            # Analyze by light source
            light_counts = {'B': 0, 'L': 0, 'U': 0, 'Unknown': 0}
            
            for file_path in csv_files:
                gem_info = self.parse_structural_filename(file_path.name)
                light_source = gem_info['light_source']
                
                if light_source in light_counts:
                    light_counts[light_source] += 1
                else:
                    light_counts['Unknown'] += 1
            
            self.diag_text.insert('end', f"\nFiles by Light Source:\n")
            for light, count in light_counts.items():
                if count > 0:
                    light_full = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}.get(light, light)
                    self.diag_text.insert('end', f"  {light_full} ({light}): {count}\n")
            
            # Show gem analysis
            self.diag_text.insert('end', f"\nGem Analysis:\n")
            self.diag_text.insert('end', f"  Unique Gems: {len(self.gem_groups)}\n")
            
            complete_gems = []
            partial_gems = []
            
            for base_id, data in self.gem_groups.items():
                b_count = len(data['files']['B'])
                l_count = len(data['files']['L'])
                u_count = len(data['files']['U'])
                
                if b_count > 0 and l_count > 0 and u_count > 0:
                    complete_gems.append(base_id)
                else:
                    partial_gems.append(base_id)
            
            self.diag_text.insert('end', f"  Complete Gems (B+L+U): {len(complete_gems)}\n")
            self.diag_text.insert('end', f"  Partial Gems: {len(partial_gems)}\n")
            
            # Show examples
            if complete_gems:
                self.diag_text.insert('end', f"\nComplete Gem Examples:\n")
                for gem_id in complete_gems[:5]:
                    data = self.gem_groups[gem_id]
                    b_count = len(data['files']['B'])
                    l_count = len(data['files']['L'])
                    u_count = len(data['files']['U'])
                    self.diag_text.insert('end', f"  Gem {gem_id}: B({b_count})+L({l_count})+U({u_count})\n")
                
                if len(complete_gems) > 5:
                    self.diag_text.insert('end', f"  ... and {len(complete_gems)-5} more\n")
            
        except Exception as e:
            self.diag_text.insert('end', f"❌ File analysis error: {e}\n")
        
        self.diag_text.config(state='disabled')
    
    # VISUALIZATION METHODS (from unified_structural_analyzer.py)
    
    def generate_preview_plots(self):
        """Generate preview plots for selected gems"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Matplotlib Required", "Install matplotlib to generate plots: pip install matplotlib")
            return
        
        if not self.selected_gems:
            messagebox.showwarning("No Selection", "Please select gems for analysis first.")
            return
        
        # Clear existing plots
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Create plots for first selected gem as preview
        first_gem_id = list(self.selected_gems.keys())[0]
        gem_data = self.selected_gems[first_gem_id]
        
        try:
            self.create_gem_visualization(first_gem_id, gem_data, preview=True)
            self.status_var.set(f"Generated preview plots for Gem {first_gem_id}")
        except Exception as e:
            messagebox.showerror("Plot Error", f"Error generating preview plots: {e}")
    
    def create_gem_visualization(self, gem_id, gem_data, preview=False):
        """Create comprehensive visualization for a gem"""
        if not HAS_MATPLOTLIB:
            return
        
        # Load spectral data
        stone_data = {}
        light_names = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}
        
        for light_code, file_path in gem_data['selected_paths'].items():
            try:
                df = pd.read_csv(file_path)
                
                # Try to identify wavelength and intensity columns
                if 'Wavelength' in df.columns and 'Intensity' in df.columns:
                    wavelength = df['Wavelength'].values
                    intensity = df['Intensity'].values
                elif len(df.columns) >= 2:
                    wavelength = df.iloc[:, 0].values
                    intensity = df.iloc[:, 1].values
                else:
                    continue
                
                stone_data[light_code] = (wavelength, intensity)
                
                # Extract features
                if HAS_SCIPY:
                    try:
                        peaks, _ = find_peaks(intensity, height=np.percentile(intensity, 75), distance=5)
                        valleys, _ = find_peaks(-intensity, height=-np.percentile(intensity, 25), distance=5)
                    except:
                        peaks = valleys = []
                else:
                    # Simple peak detection
                    threshold = np.percentile(intensity, 75)
                    peaks = []
                    for i in range(1, len(intensity) - 1):
                        if intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1] and intensity[i] > threshold:
                            peaks.append(i)
                    valleys = []
                
                self.spectral_features[light_code] = {
                    'peaks': {'positions': [wavelength[i] for i in peaks], 'intensities': [intensity[i] for i in peaks]},
                    'valleys': {'positions': [wavelength[i] for i in valleys], 'intensities': [intensity[i] for i in valleys]},
                    'stats': {'mean': np.mean(intensity), 'std': np.std(intensity), 'max': np.max(intensity), 'min': np.min(intensity)}
                }
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not stone_data:
            return
        
        # Create matplotlib figure
        if preview:
            # Embed in GUI
            fig = plt.Figure(figsize=(12, 8), dpi=100)
            
            if HAS_SEABORN:
                sns.set_style("whitegrid")
            
            # Create subplots
            if len(stone_data) == 3:
                axes = fig.subplots(2, 2)
                axes = axes.flatten()
            else:
                axes = fig.subplots(1, len(stone_data))
                if len(stone_data) == 1:
                    axes = [axes]
            
            # Color mapping
            colors = {'B': 'blue', 'L': 'red', 'U': 'purple'}
            
            # Plot 1: Overlaid spectra
            if len(stone_data) >= 2:
                ax = axes[0]
                for light_code, (wavelength, intensity) in stone_data.items():
                    ax.plot(wavelength, intensity, color=colors[light_code], 
                           label=f'{light_names[light_code]} ({light_code})', linewidth=2)
                
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity')
                ax.set_title(f'Multi-Source Spectral Overlay - Gem {gem_id}')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Individual spectrum plots
            for i, (light_code, (wavelength, intensity)) in enumerate(stone_data.items()):
                ax_idx = i + 1 if len(stone_data) >= 2 else i
                if ax_idx >= len(axes):
                    break
                    
                ax = axes[ax_idx]
                ax.plot(wavelength, intensity, color=colors[light_code], linewidth=2)
                
                # Add peaks if available
                if light_code in self.spectral_features and self.plot_peaks.get():
                    features = self.spectral_features[light_code]
                    if features['peaks']['positions']:
                        ax.scatter(features['peaks']['positions'], features['peaks']['intensities'], 
                                 color='red', s=50, alpha=0.7, marker='^', label='Peaks')
                
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Intensity')
                ax.set_title(f'{light_names[light_code]} Spectrum')
                ax.grid(True, alpha=0.3)
                if light_code in self.spectral_features and self.plot_peaks.get():
                    ax.legend()
            
            # Hide unused subplots
            for i in range(len(stone_data) + (1 if len(stone_data) >= 2 else 0), len(axes)):
                axes[i].set_visible(False)
            
            fig.suptitle(f'Ultimate Analysis - Gem {gem_id} ({self.analysis_name})', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            # Embed in GUI
            canvas = FigureCanvasTkAgg(fig, self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        else:
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graphs_dir = self.project_root / "outputs" / "structural_results" / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            
            plot_file = graphs_dir / f"ultimate_analysis_{gem_id}_{self.input_source}_{timestamp}.png"
            
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Ultimate Analysis: Gem {gem_id} ({self.analysis_name})', fontsize=16, fontweight='bold')
            
            # ... (similar plotting code but save to file)
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
    
    def save_all_plots(self):
        """Save plots for all selected gems"""
        if not HAS_MATPLOTLIB:
            messagebox.showwarning("Matplotlib Required", "Install matplotlib to save plots: pip install matplotlib")
            return
        
        if not self.selected_gems:
            messagebox.showwarning("No Selection", "Please select gems for analysis first.")
            return
        
        saved_count = 0
        
        for gem_id, gem_data in self.selected_gems.items():
            try:
                plot_file = self.create_gem_visualization(gem_id, gem_data, preview=False)
                if plot_file:
                    saved_count += 1
            except Exception as e:
                print(f"Error saving plot for Gem {gem_id}: {e}")
        
        messagebox.showinfo("Plots Saved", f"Saved {saved_count} visualization plots to outputs/structural_results/graphs/")
    
    # ULTIMATE ANALYSIS METHOD
    
    def start_ultimate_analysis(self):
        """Start the ultimate structural database matching analysis"""
        if not self.selected_gems:
            messagebox.showwarning("No Selection", "Please select at least one gem for ultimate analysis.")
            return
        
        if not self.check_databases():
            return
        
        # Ultimate confirmation
        gem_count = len(self.selected_gems)
        if not messagebox.askyesno("Start Ultimate Analysis", 
                                  f"Start ultimate structural analysis for {gem_count} gems?\n\n"
                                  f"This will:\n"
                                  f"• Archive any existing results\n"
                                  f"• Use advanced feature weighting\n"
                                  f"• Apply multi-point scoring algorithms\n"
                                  f"• Apply spectral normalization\n"
                                  f"• Perform comprehensive database matching\n"
                                  f"• Generate visualization plots (if enabled)\n\n"
                                  f"Analysis may take several minutes."):
            return
        
        print(f"\n🚀 Starting ultimate structural analysis for {gem_count} gems...")
        print(f"📁 Input source: {self.input_source}")
        print(f"🗄️  Database: {self.database_type.upper()}")
        
        try:
            # Show progress bar
            self.progress_bar.pack(side='right', fill='x', expand=True, padx=(10, 0))
            self.progress_var.set(0)
            self.root.update()
            
            success = self.run_ultimate_analysis()
            
            # Hide progress bar
            self.progress_bar.pack_forget()
            
            if success:
                messagebox.showinfo("Ultimate Analysis Complete", 
                                  f"Ultimate analysis completed for {gem_count} gems!\n\n"
                                  f"Results saved to:\n"
                                  f"• Reports: outputs/structural_results/reports/\n"
                                  f"• Graphs: outputs/structural_results/graphs/\n\n"
                                  f"Previous results archived to:\n"
                                  f"• results(archive)/post_analysis_structural/\n\n"
                                  f"Check reports folder for detailed scoring breakdown.")
            else:
                messagebox.showerror("Analysis Failed", "Ultimate analysis encountered errors. Check console for details.")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            print(f"❌ Ultimate analysis error: {e}")
            traceback.print_exc()
            messagebox.showerror("Analysis Error", f"Ultimate analysis failed:\n{e}")
    
    def run_ultimate_analysis(self):
        """Run the ultimate structural database matching analysis"""
        # Archive existing results before creating new ones
        self.archive_previous_results()
        
        # Create output directories
        results_dir = self.project_root / "outputs" / "structural_results"
        reports_dir = results_dir / "reports"
        graphs_dir = results_dir / "graphs"
        
        results_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        graphs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load database with schema adaptation
        print(f"📊 Loading ultimate structural database...")
        
        if self.database_type == "sqlite":
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_name = tables[0][0]
            
            # Check actual column names
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            actual_columns = [col[1] for col in columns_info]
            
            print(f"📋 Database columns: {', '.join(actual_columns[:5])}{'...' if len(actual_columns) > 5 else ''}")
            
            # Find file identifier column
            file_columns = ['file', 'filename', 'gem_id', 'identifier', 'full_name']
            file_col = None
            for col in file_columns:
                if col in actual_columns:
                    file_col = col
                    break
            
            # Find wavelength column
            wl_columns = ['wavelength', 'Wavelength', 'wavelength_nm', 'Wavelength_nm']
            wl_col = None
            for col in wl_columns:
                if col in actual_columns:
                    wl_col = col
                    break
            
            # Build adaptive query
            if file_col and wl_col:
                query = f"SELECT * FROM {table_name} ORDER BY {file_col}, {wl_col}"
                print(f"📊 Using adaptive query: ORDER BY {file_col}, {wl_col}")
            else:
                query = f"SELECT * FROM {table_name}"
                print(f"📊 Using basic query (no sorting)")
            
            try:
                db_df = pd.read_sql_query(query, conn)
                print(f"✅ Database loaded successfully: {len(db_df):,} records")
            except Exception as e:
                print(f"❌ Database query failed: {e}")
                # Try with basic query as fallback
                try:
                    db_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                    print(f"✅ Fallback query successful: {len(db_df):,} records")
                except Exception as e2:
                    print(f"❌ Even fallback query failed: {e2}")
                    conn.close()
                    return False
            
            conn.close()
            
        else:  # CSV
            db_df = pd.read_csv(self.database_path)
            print(f"✅ CSV database loaded: {len(db_df):,} records")
        
        if db_df.empty:
            print(f"❌ Database is empty!")
            return False
        
        # Analyze each selected gem
        all_results = []
        total_gems = len(self.selected_gems)
        analysis_successful = True
        
        for i, (base_id, data) in enumerate(self.selected_gems.items()):
            print(f"\n🎯 Ultimate Analysis: Gem {base_id} ({i+1}/{total_gems})...")
            
            # Update progress
            progress = (i / total_gems) * 100
            if self.mode == "gui":
                self.progress_var.set(progress)
                self.status_var.set(f"Analyzing Gem {base_id} ({i+1}/{total_gems})...")
                self.root.update()
            
            selected_files = data['selected_files']
            selected_paths = data['selected_paths']
            options = data.get('options', {})
            
            gem_results = {
                'gem_id': base_id,
                'analysis_timestamp': timestamp,
                'analysis_type': 'ultimate',
                'input_source': self.input_source,
                'database_type': self.database_type,
                'options_used': options,
                'light_source_results': {},
                'best_overall_match': None,
                'lowest_combined_score': float('inf'),
                'feature_analysis': {}
            }
            
            gem_analysis_successful = False
            
            # Analyze each light source
            for light_source, file_path in selected_paths.items():
                print(f"   📄 Ultimate processing {light_source}: {file_path.name}")
                
                try:
                    # Load and process data
                    unknown_df = pd.read_csv(file_path)
                    print(f"      📊 Loaded {len(unknown_df)} structural points")
                    
                    if unknown_df.empty:
                        print(f"      ⚠️ Empty data file: {file_path.name}")
                        continue
                    
                    # Apply normalization if requested
                    if options.get('normalize', True):
                        unknown_df = self.normalize_spectrum(unknown_df, light_source)
                    
                    # Extract enhanced features
                    features = self.extract_enhanced_features(unknown_df, light_source, options)
                    gem_results['feature_analysis'][light_source] = features
                    
                    # Find database matches
                    db_matches = self.find_enhanced_database_matches(db_df, light_source)
                    
                    if db_matches.empty:
                        print(f"      ⚠️ No database matches for light source {light_source}")
                        continue
                    
                    # Calculate ultimate scores
                    scores = self.calculate_ultimate_scores(unknown_df, db_matches, light_source, options)
                    
                    if scores:
                        best_match = min(scores, key=lambda x: x['score'])
                        print(f"      ✅ Best match: {best_match['db_gem_id']} (score: {best_match['score']:.2f})")
                        
                        gem_results['light_source_results'][light_source] = {
                            'file_analyzed': file_path.name,
                            'features_extracted': features,
                            'best_match': best_match,
                            'top_5_matches': sorted(scores, key=lambda x: x['score'])[:5]
                        }
                        gem_analysis_successful = True
                    else:
                        print(f"      ⚠️ No valid scores calculated for {light_source}")
                
                except Exception as e:
                    print(f"      ❌ Error processing {light_source}: {e}")
                    analysis_successful = False
                    continue
            
            # Calculate ultimate combined score if multiple light sources
            if len(gem_results['light_source_results']) > 1:
                try:
                    combined_scores = self.calculate_ultimate_combined_scores(gem_results['light_source_results'])
                    if combined_scores:
                        best_combined = min(combined_scores, key=lambda x: x['combined_score'])
                        gem_results['best_overall_match'] = best_combined
                        gem_results['lowest_combined_score'] = best_combined['combined_score']
                        print(f"   🏆 Ultimate best match: {best_combined['db_gem_id']} (score: {best_combined['combined_score']:.2f})")
                except Exception as e:
                    print(f"   ⚠️ Error calculating combined scores: {e}")
            
            if gem_analysis_successful:
                all_results.append(gem_results)
                print(f"   ✅ Gem {base_id} analysis completed successfully")
            else:
                print(f"   ❌ Gem {base_id} analysis failed")
                analysis_successful = False
        
        # Complete progress
        if self.mode == "gui":
            self.progress_var.set(100)
            self.status_var.set(f"Saving ultimate analysis results...")
            self.root.update()
        
        if not all_results:
            print(f"\n❌ No gems were successfully analyzed!")
            print(f"💡 Check database schema compatibility and file formats")
            return False
        
        try:
            # Save ultimate results
            self.save_ultimate_results(all_results, results_dir, reports_dir, graphs_dir, timestamp)
            
            # Generate visualizations if enabled and available
            if HAS_MATPLOTLIB:
                try:
                    self.generate_analysis_visualizations(all_results, graphs_dir, timestamp)
                except Exception as e:
                    print(f"⚠️ Visualization generation failed: {e}")
            
            success_count = len(all_results)
            total_count = len(self.selected_gems)
            
            print(f"\n🎉 Ultimate analysis completed!")
            print(f"📊 Successfully analyzed: {success_count}/{total_count} gems")
            print(f"📊 Results saved to:")
            print(f"   Reports: {reports_dir}")
            print(f"   Graphs:  {graphs_dir}")
            
            if success_count < total_count:
                print(f"⚠️ {total_count - success_count} gems failed analysis - check logs above")
                return success_count == total_count  # Return False if any failed
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return False
    
    def generate_analysis_visualizations(self, all_results, graphs_dir, timestamp):
        """Generate visualization plots for analysis results"""
        if not HAS_MATPLOTLIB:
            return
        
        print(f"📈 Generating ultimate analysis visualizations...")
        
        try:
            # Create summary visualization
            plt.style.use('default')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Ultimate Structural Analysis Summary', fontsize=16, fontweight='bold')
            
            # Plot 1: Score distribution
            scores = []
            gem_names = []
            for result in all_results:
                if result['best_overall_match']:
                    scores.append(result['best_overall_match']['combined_score'])
                    gem_names.append(result['gem_id'])
            
            if scores:
                axes[0, 0].bar(range(len(scores)), scores, color='skyblue')
                axes[0, 0].set_title('Ultimate Match Scores by Gem')
                axes[0, 0].set_xlabel('Gem Index')
                axes[0, 0].set_ylabel('Combined Score')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Light source coverage
            light_coverage = {'B': 0, 'L': 0, 'U': 0}
            for result in all_results:
                for ls in result['light_source_results'].keys():
                    if ls in light_coverage:
                        light_coverage[ls] += 1
            
            if any(light_coverage.values()):
                axes[0, 1].bar(light_coverage.keys(), light_coverage.values(), 
                              color=['blue', 'red', 'purple'])
                axes[0, 1].set_title('Light Source Coverage')
                axes[0, 1].set_xlabel('Light Source')
                axes[0, 1].set_ylabel('Number of Gems')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Analysis success metrics
            total_gems = len(all_results)
            successful_matches = len([r for r in all_results if r['best_overall_match']])
            
            success_data = [successful_matches, total_gems - successful_matches]
            success_labels = ['Successful Matches', 'No Matches']
            
            axes[1, 0].pie(success_data, labels=success_labels, autopct='%1.1f%%', 
                          colors=['lightgreen', 'lightcoral'])
            axes[1, 0].set_title('Analysis Success Rate')
            
            # Plot 4: Feature statistics
            total_features = {}
            for result in all_results:
                for ls, features in result['feature_analysis'].items():
                    if 'peaks' in features:
                        peak_count = features['peaks'].get('count', 0)
                        if ls not in total_features:
                            total_features[ls] = []
                        total_features[ls].append(peak_count)
            
            if total_features:
                for i, (ls, counts) in enumerate(total_features.items()):
                    axes[1, 1].hist(counts, alpha=0.7, label=f'{ls} Light', bins=10)
                
                axes[1, 1].set_title('Peak Count Distribution')
                axes[1, 1].set_xlabel('Number of Peaks')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = graphs_dir / f"ultimate_analysis_summary_{timestamp}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Summary visualization saved: {plot_file.name}")
            
        except Exception as e:
            print(f"⚠️ Error generating summary visualization: {e}")
    
    def run_auto_analysis(self, auto_select_complete=True):
        """Run analysis in automatic mode (called from main.py) with proper error handling"""
        print(f"🤖 Running automatic ultimate analysis...")
        print(f"📁 Input source: {self.input_source}")
        print(f"🔍 Auto-select complete gems: {auto_select_complete}")
        
        try:
            if not self.gem_groups:
                print(f"❌ No gems found in {self.input_path}")
                return False
            
            # Auto-select gems
            if auto_select_complete:
                # Select all complete gems
                for base_id, data in self.gem_groups.items():
                    b_count = len(data['files']['B'])
                    l_count = len(data['files']['L'])
                    u_count = len(data['files']['U'])
                    
                    if b_count > 0 and l_count > 0 and u_count > 0:
                        # Auto-select first file for each light source
                        selected_files = {}
                        selected_paths = {}
                        
                        for light_source in ['B', 'L', 'U']:
                            selected_files[light_source] = data['files'][light_source][0]
                            selected_paths[light_source] = data['file_paths'][light_source][0]
                        
                        self.selected_gems[base_id] = {
                            'selected_files': selected_files,
                            'selected_paths': selected_paths,
                            'options': {
                                'normalize': True,
                                'feature_weighting': True,
                                'multipoint_analysis': True,
                                'visualization': HAS_MATPLOTLIB
                            }
                        }
            
            if not self.selected_gems:
                print(f"❌ No complete gems found for automatic analysis")
                return False
            
            print(f"✅ Auto-selected {len(self.selected_gems)} complete gems")
            
            # Run analysis with proper error handling
            success = self.run_ultimate_analysis()
            
            if success:
                print(f"🎉 Automatic ultimate analysis completed successfully")
            else:
                print(f"❌ Automatic ultimate analysis failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Automatic analysis failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # [Include all the sophisticated analysis methods from the original multi_gem_structural_analyzer.py]
    # normalize_spectrum, extract_enhanced_features, find_enhanced_database_matches, etc.
    # [These are the same as in the original, so I'll include placeholders here to keep artifact manageable]
    
    def normalize_spectrum(self, df, light_source, target_intensity=50000):
        """Normalize spectrum intensity based on light source"""
        try:
            # Find wavelength and intensity columns
            wl_columns = ['Wavelength', 'wavelength', 'Wavelength_nm', 'wavelength_nm']
            int_columns = ['Intensity', 'intensity', 'Intensity_Value']
            
            wl_col = None
            int_col = None
            
            for col in wl_columns:
                if col in df.columns:
                    wl_col = col
                    break
            
            for col in int_columns:
                if col in df.columns:
                    int_col = col
                    break
            
            if not wl_col or not int_col:
                print(f"      ⚠️ Could not find wavelength/intensity columns for normalization")
                return df
            
            wavelength = df[wl_col].values
            intensity = df[int_col].values
            
            # Light source specific normalization
            if light_source == 'B' or light_source == 'Halogen':
                # Halogen: normalize to 50,000 at 650nm
                anchor_wavelength = 650
                target_intensity = 50000
                anchor_intensity = np.interp(anchor_wavelength, wavelength, intensity)
                
            elif light_source == 'L' or light_source == 'Laser':
                # Laser: normalize max intensity to 50,000
                anchor_intensity = np.max(intensity)
                target_intensity = 50000
                
            elif light_source == 'U' or light_source == 'UV':
                # UV: normalize to 15,000 at 811nm
                anchor_wavelength = 811
                target_intensity = 15000
                anchor_intensity = np.interp(anchor_wavelength, wavelength, intensity)
                
            else:
                return df  # No normalization for unknown sources
            
            if anchor_intensity == 0:
                return df  # Cannot normalize
                
            # Apply normalization
            normalization_factor = target_intensity / anchor_intensity
            normalized_intensity = intensity * normalization_factor
            
            # Scale to 0-100 range
            min_intensity = np.min(normalized_intensity)
            max_intensity = np.max(normalized_intensity)
            
            if max_intensity > min_intensity:
                scaled_intensity = 100 * (normalized_intensity - min_intensity) / (max_intensity - min_intensity)
            else:
                scaled_intensity = normalized_intensity
            
            # Apply scaling to dataframe
            df[int_col] = scaled_intensity
            
            print(f"      🔧 Applied normalization + 0-100 scaling for {light_source}")
            return df
            
        except Exception as e:
            print(f"   ⚠️ Normalization error: {e}")
            return df
    
    def extract_enhanced_features(self, df, light_source, options):
        """Extract enhanced structural features with weighting"""
        features = {
            'light_source': light_source,
            'data_points': len(df),
            'feature_weights_used': options.get('feature_weighting', True),
            'multipoint_analysis': options.get('multipoint_analysis', True)
        }
        
        # Find wavelength and intensity columns
        wl_columns = ['Wavelength', 'wavelength', 'Wavelength_nm', 'wavelength_nm']
        int_columns = ['Intensity', 'intensity', 'Intensity_Value']
        
        wl_col = None
        int_col = None
        
        for col in wl_columns:
            if col in df.columns:
                wl_col = col
                break
        
        for col in int_columns:
            if col in df.columns:
                int_col = col
                break
        
        if not wl_col or not int_col:
            # Try first two numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                wl_col = numeric_cols[0]
                int_col = numeric_cols[1]
        
        if not wl_col or not int_col:
            print(f"      ⚠️ Could not identify wavelength/intensity columns")
            return features
        
        try:
            wavelength = df[wl_col].values
            intensity = df[int_col].values
            
            features['wavelength_range'] = [float(np.nanmin(wavelength)), float(np.nanmax(wavelength))]
            features['intensity_stats'] = {
                'mean': float(np.nanmean(intensity)),
                'std': float(np.nanstd(intensity)),
                'max': float(np.nanmax(intensity)),
                'min': float(np.nanmin(intensity))
            }
            
            # Feature type analysis
            if 'Feature' in df.columns:
                feature_types = df['Feature'].value_counts().to_dict()
                weighted_importance = 0
                
                for feature_type, count in feature_types.items():
                    weight = self.feature_weights.get(feature_type, 0.1)
                    weighted_importance += count * weight
                
                features['feature_types'] = feature_types
                features['weighted_importance'] = weighted_importance
            
            # Peak detection
            if HAS_SCIPY:
                try:
                    peaks, peak_properties = find_peaks(intensity, height=np.nanpercentile(intensity, 75), distance=5)
                    features['peaks'] = {
                        'count': len(peaks),
                        'positions': [float(wavelength[i]) for i in peaks],
                        'intensities': [float(intensity[i]) for i in peaks],
                        'prominences': peak_properties.get('peak_heights', []).tolist() if 'peak_heights' in peak_properties else []
                    }
                except Exception:
                    features['peaks'] = {'count': 0, 'positions': [], 'intensities': []}
            else:
                # Simple peak detection
                threshold = np.nanpercentile(intensity, 75)
                peaks = []
                for i in range(1, len(intensity) - 1):
                    if not (pd.isna(intensity[i]) or pd.isna(intensity[i-1]) or pd.isna(intensity[i+1])):
                        if intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1] and intensity[i] > threshold:
                            peaks.append(i)
                
                features['peaks'] = {
                    'count': len(peaks),
                    'positions': [float(wavelength[i]) for i in peaks],
                    'intensities': [float(intensity[i]) for i in peaks]
                }
            
        except Exception as e:
            print(f"      ⚠️ Feature extraction error: {e}")
        
        return features
    
    def calculate_ultimate_combined_scores(self, light_source_results):
        """Calculate ultimate combined scores with light source weighting"""
        # Get all unique database gem IDs
        all_gem_ids = set()
        for ls_data in light_source_results.values():
            for match in ls_data['top_5_matches']:
                all_gem_ids.add(match['db_gem_id'])
        
        combined_scores = []
        
        for gem_id in all_gem_ids:
            total_score = 0.0
            total_weight = 0.0
            light_sources_used = []
            
            for light_source, ls_data in light_source_results.items():
                # Get light source weight
                light_full = {'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}[light_source]
                light_weight = self.light_weights.get(light_full, 1.0)
                
                # Find this gem in the results
                gem_score = None
                for match in ls_data['top_5_matches']:
                    if match['db_gem_id'] == gem_id:
                        gem_score = match['score']
                        break
                
                if gem_score is not None:
                    total_score += gem_score * light_weight
                    total_weight += light_weight
                    light_sources_used.append(light_source)
                else:
                    # Penalty for missing light source (weighted)
                    total_score += 50.0 * light_weight  # Moderate penalty
                    total_weight += light_weight
            
            if total_weight > 0:
                combined_score = total_score / total_weight
                
                # Bonus for complete coverage
                completeness_bonus = len(light_sources_used) / 3.0
                final_score = combined_score * (2.0 - completeness_bonus)
                
                combined_scores.append({
                    'db_gem_id': gem_id,
                    'combined_score': final_score,
                    'light_sources_used': light_sources_used,
                    'source_count': len(light_sources_used),
                    'completeness_bonus': completeness_bonus
                })
        
        return combined_scores
    
    def save_ultimate_results(self, all_results, results_dir, reports_dir, graphs_dir, timestamp):
        """Save ultimate analysis results with detailed breakdown"""
        # Create summary report
        summary_data = []
        
        for result in all_results:
            gem_id = result['gem_id']
            options_used = result.get('options_used', {})
            
            summary_row = {
                'Gem_ID': gem_id,
                'Analysis_Timestamp': timestamp,
                'Analysis_Type': 'Ultimate',
                'Input_Source': result.get('input_source', 'unknown'),
                'Database_Type': result.get('database_type', 'unknown'),
                'Light_Sources_Analyzed': '+'.join(result['light_source_results'].keys()),
                'Light_Source_Count': len(result['light_source_results'])
            }
            
            if result['best_overall_match']:
                summary_row.update({
                    'Best_Overall_Match': result['best_overall_match']['db_gem_id'],
                    'Ultimate_Combined_Score': result['best_overall_match']['combined_score'],
                    'Sources_Used': '+'.join(result['best_overall_match']['light_sources_used']),
                    'Completeness_Bonus': result['best_overall_match'].get('completeness_bonus', 0)
                })
            else:
                # Single light source case
                if result['light_source_results']:
                    ls_result = list(result['light_source_results'].values())[0]
                    summary_row.update({
                        'Best_Overall_Match': ls_result['best_match']['db_gem_id'],
                        'Ultimate_Combined_Score': ls_result['best_match']['score'],
                        'Sources_Used': list(result['light_source_results'].keys())[0],
                        'Completeness_Bonus': 0
                    })
            
            summary_data.append(summary_row)
        
        # Save summary report
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_file = reports_dir / f"ultimate_structural_analysis_{timestamp}.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"📄 Ultimate analysis summary saved: {summary_file.name}")
        
        # Save full results as JSON
        json_file = reports_dir / f"ultimate_structural_analysis_full_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"📄 Ultimate full results saved: {json_file.name}")
        
        # Generate text report
        txt_file = reports_dir / f"ultimate_structural_analysis_report_{timestamp}.txt"
        with open(txt_file, 'w') as f:
            f.write("ULTIMATE STRUCTURAL ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Type: Ultimate Multi-Gem Structural Analysis\n")
            f.write(f"Gems Analyzed: {len(all_results)}\n\n")
            
            for result in all_results:
                f.write(f"Gem {result['gem_id']}:\n")
                f.write(f"  Light Sources: {', '.join(result['light_source_results'].keys())}\n")
                if result['best_overall_match']:
                    f.write(f"  Best Match: {result['best_overall_match']['db_gem_id']}\n")
                    f.write(f"  Score: {result['best_overall_match']['combined_score']:.2f}\n")
                f.write("\n")
        
        print(f"📄 Ultimate analysis report saved: {txt_file.name}")
    
    def archive_previous_results(self):
        """Archive existing results before running new analysis"""
        archive_base = self.project_root / "results(archive)" / "post_analysis_structural"
        archive_reports = archive_base / "reports"
        archive_graphs = archive_base / "graphs"
        
        archive_reports.mkdir(parents=True, exist_ok=True)
        archive_graphs.mkdir(parents=True, exist_ok=True)
        
        # Check for existing results
        current_reports = self.project_root / "outputs" / "structural_results" / "reports"
        current_graphs = self.project_root / "outputs" / "structural_results" / "graphs"
        
        archived_count = 0
        
        if current_reports.exists():
            for file_path in current_reports.glob("*"):
                if file_path.is_file():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_name = f"{file_path.stem}_archived_{timestamp}{file_path.suffix}"
                    shutil.move(str(file_path), str(archive_reports / archive_name))
                    archived_count += 1
        
        if current_graphs.exists():
            for file_path in current_graphs.glob("*"):
                if file_path.is_file():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_name = f"{file_path.stem}_archived_{timestamp}{file_path.suffix}"
                    shutil.move(str(file_path), str(archive_graphs / archive_name))
                    archived_count += 1
        
        if archived_count > 0:
            print(f"📦 Archived {archived_count} previous result files")
        else:
            print(f"📁 No previous results to archive")
    
    # PROGRAMMATIC INTERFACE FOR MAIN.PY
    
    def run_auto_analysis(self, auto_select_complete=True):
        """Run analysis in automatic mode (called from main.py)"""
        print(f"🤖 Running automatic ultimate analysis...")
        print(f"📁 Input source: {self.input_source}")
        print(f"🔍 Auto-select complete gems: {auto_select_complete}")
        
        if not self.gem_groups:
            print(f"❌ No gems found in {self.input_path}")
            return False
        
        # Auto-select gems
        if auto_select_complete:
            # Select all complete gems
            for base_id, data in self.gem_groups.items():
                b_count = len(data['files']['B'])
                l_count = len(data['files']['L'])
                u_count = len(data['files']['U'])
                
                if b_count > 0 and l_count > 0 and u_count > 0:
                    # Auto-select first file for each light source
                    selected_files = {}
                    selected_paths = {}
                    
                    for light_source in ['B', 'L', 'U']:
                        selected_files[light_source] = data['files'][light_source][0]
                        selected_paths[light_source] = data['file_paths'][light_source][0]
                    
                    self.selected_gems[base_id] = {
                        'selected_files': selected_files,
                        'selected_paths': selected_paths,
                        'options': {
                            'normalize': True,
                            'feature_weighting': True,
                            'multipoint_analysis': True,
                            'visualization': HAS_MATPLOTLIB
                        }
                    }
        
        if not self.selected_gems:
            print(f"❌ No complete gems found for automatic analysis")
            return False
        
        print(f"✅ Auto-selected {len(self.selected_gems)} complete gems")
        
        # Run analysis
        try:
            return self.run_ultimate_analysis()
        except Exception as e:
            print(f"❌ Automatic analysis failed: {e}")
            return False
    
    def run(self):
        """Start the application (GUI mode only)"""
        if self.mode == "gui" and self.root:
            self.root.mainloop()

def main():
    """Main entry point - can be called from main.py or standalone"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultimate Multi-Gem Structural Analyzer")
    parser.add_argument("--mode", choices=["gui", "auto", "diagnostic"], default="gui",
                      help="Analysis mode")
    parser.add_argument("--input-source", choices=["archive", "current"], default="archive",
                      help="Input source (archive for Option 8, current for Option 4)")
    parser.add_argument("--auto-complete", action="store_true", default=True,
                      help="Auto-select all complete gems in auto mode")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting Ultimate Multi-Gem Structural Analyzer...")
        print(f"   Mode: {args.mode}")
        print(f"   Input Source: {args.input_source}")
        
        analyzer = UltimateMultiGemStructuralAnalyzer(mode=args.mode, input_source=args.input_source)
        
        if args.mode == "gui":
            analyzer.run()
        elif args.mode == "auto":
            success = analyzer.run_auto_analysis(auto_select_complete=args.auto_complete)
            print(f"🎯 Automatic analysis {'completed successfully' if success else 'failed'}")
        elif args.mode == "diagnostic":
            analyzer.run_full_diagnostics()
        
        print("Ultimate Multi-Gem Structural Analyzer finished.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()