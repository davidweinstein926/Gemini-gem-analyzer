#!/usr/bin/env python3
"""
ENHANCED MULTI-GEM STRUCTURAL ANALYZER - ADVANCED DATABASE MATCHING
Incorporates sophisticated scoring algorithms and comprehensive analysis
Input: data/structural(archive)/*.csv files
Compares against: multi_structural_gem_data.db
Output: outputs/structural_results/
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
import re
import sys
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy import stats
from scipy.signal import find_peaks
import json
import math
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class EnhancedMultiGemStructuralAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Multi-Gem Structural Analyzer - Advanced Database Matching")
        self.root.geometry("1200x800")
        
        # Find project root
        self.project_root = self.find_project_root()
        self.archive_path = self.project_root / "data" / "structural(archive)"
        
        # Check for both databases
        self.sqlite_db_path = self.project_root / "multi_structural_gem_data.db"
        self.csv_db_path = self.project_root / "gemini_structural_db.csv"
        self.database_type = None
        self.database_path = None
        
        # Data structures
        self.gem_groups = {}
        self.selected_gems = {}
        
        # Advanced scoring parameters
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
        
        # Setup GUI
        self.setup_gui()
        self.check_databases()
        self.scan_archive_directory()
    
    def check_databases(self):
        """Check for both SQLite and CSV databases"""
        available_dbs = []
        
        if self.sqlite_db_path.exists():
            available_dbs.append(("sqlite", self.sqlite_db_path))
        if self.csv_db_path.exists():
            available_dbs.append(("csv", self.csv_db_path))
        
        if not available_dbs:
            messagebox.showerror("Database Error", 
                f"No structural databases found!\n\n"
                f"Expected files:\n"
                f"- multi_structural_gem_data.db\n"
                f"- gemini_structural_db.csv\n\n"
                f"Please ensure at least one exists in the project root.")
            return False
        
        # If both exist, ask user to choose
        if len(available_dbs) > 1:
            choice = messagebox.askyesno("Database Selection",
                f"Multiple databases found:\n\n"
                f"SQLite: {self.sqlite_db_path.name}\n"
                f"CSV: {self.csv_db_path.name}\n\n"
                f"Use SQLite database? (No = use CSV)")
            
            if choice:
                self.database_type = "sqlite"
                self.database_path = self.sqlite_db_path
            else:
                self.database_type = "csv"
                self.database_path = self.csv_db_path
        else:
            # Use the single available database
            self.database_type, self.database_path = available_dbs[0]
        
        print(f"Using {self.database_type.upper()} database: {self.database_path.name}")
        
        # Validate the selected database
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
                        messagebox.showwarning("Database Warning", 
                            f"SQLite database exists but contains no tables.\n"
                            f"Please import structural data first.")
                        return False
                    
                    # Check for data
                    table_name = tables[0][0]
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    if count == 0:
                        messagebox.showwarning("Database Warning", 
                            f"SQLite database table '{table_name}' is empty.\n"
                            f"Please import structural data first.")
                        return False
                    
                    print(f"SQLite database validated: {count} records in table '{table_name}'")
                    
            else:  # CSV
                df = pd.read_csv(self.database_path)
                
                if df.empty:
                    messagebox.showwarning("Database Warning", 
                        f"CSV database is empty.\n"
                        f"Please import structural data first.")
                    return False
                
                # Check for required columns
                required_cols = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
                identifier_col = None
                for col in required_cols:
                    if col in df.columns:
                        identifier_col = col
                        break
                
                if not identifier_col:
                    messagebox.showerror("Database Error", 
                        f"CSV database missing identifier column.\n"
                        f"Expected one of: {', '.join(required_cols)}")
                    return False
                
                print(f"CSV database validated: {len(df)} records with identifier column '{identifier_col}'")
            
            return True
            
        except Exception as e:
            messagebox.showerror("Database Error", f"Database validation failed:\n{e}")
            return False
    
    def normalize_spectrum(self, df, light_source, target_intensity=50000):
        """Normalize spectrum intensity based on light source, then scale to 0-100"""
        try:
            if 'Wavelength' in df.columns and 'Intensity' in df.columns:
                wavelength = df['Wavelength'].values
                intensity = df['Intensity'].values
                intensity_col = 'Intensity'
            else:
                wavelength = df.iloc[:, 0].values
                intensity = df.iloc[:, 1].values
                intensity_col = None
            
            # Step 1: Light source specific normalization
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
            
            # Step 2: Scale to 0-100 range
            min_intensity = np.min(normalized_intensity)
            max_intensity = np.max(normalized_intensity)
            
            if max_intensity > min_intensity:
                scaled_intensity = 100 * (normalized_intensity - min_intensity) / (max_intensity - min_intensity)
            else:
                scaled_intensity = normalized_intensity  # Avoid division by zero
            
            # Apply scaling to dataframe
            if intensity_col:
                df[intensity_col] = scaled_intensity
            else:
                df.iloc[:, 1] = scaled_intensity
            
            print(f"      🔧 Applied normalization + 0-100 scaling for {light_source}")
            return df
            
        except Exception as e:
            print(f"   ⚠️ Normalization error: {e}")
            return df
    
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
    
    def scan_archive_directory(self):
        """Scan structural archive directory and group gems by base_id"""
        if not self.archive_path.exists():
            messagebox.showerror("Error", f"Structural archive directory not found:\n{self.archive_path}")
            return
        
        csv_files = list(self.archive_path.glob("*.csv"))
        if not csv_files:
            messagebox.showwarning("Warning", f"No .csv files found in:\n{self.archive_path}")
            return
        
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
        
        self.populate_gem_list()
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Enhanced Multi-Gem Structural Analyzer", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Directory info
        dir_label = ttk.Label(main_frame, text=f"Structural Archive: {self.archive_path}")
        dir_label.pack(pady=(0, 5))
        
        db_label = ttk.Label(main_frame, text=f"Database: {self.database_path}")
        db_label.pack(pady=(0, 10))
        
        # Info label
        info_label = ttk.Label(main_frame, 
                              text="Advanced structural analysis with feature weighting and multi-point scoring", 
                              font=('Arial', 10), foreground='blue')
        info_label.pack(pady=(0, 10))
        
        # Main content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Left side - Available gems
        left_frame = ttk.LabelFrame(content_frame, text="Available Structural Gems", padding="5")
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Gem listbox
        self.gem_listbox = tk.Listbox(left_frame, height=20)
        self.gem_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        # Select button
        select_btn = ttk.Button(left_frame, text="Select Structural Files", command=self.select_gem_files)
        select_btn.pack(pady=(0, 5))
        
        # Right side - Selected gems
        right_frame = ttk.LabelFrame(content_frame, text="Selected for Enhanced Analysis", padding="5")
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Selected listbox
        self.selected_listbox = tk.Listbox(right_frame, height=20)
        self.selected_listbox.pack(fill='both', expand=True, pady=(0, 10))
        
        # Remove button
        remove_btn = ttk.Button(right_frame, text="Remove Selected", command=self.remove_selected)
        remove_btn.pack(pady=(0, 5))
        
        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill='x', pady=10)
        
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Refresh", command=self.scan_archive_directory).pack(side='left', padx=(0, 10))
        ttk.Button(button_frame, text="Start Enhanced Analysis", command=self.start_analysis, 
                  style='Accent.TButton').pack(side='right', padx=(10, 0))
        ttk.Button(button_frame, text="Close", command=self.root.quit).pack(side='right')
        
        # Status
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack()
    
    def populate_gem_list(self):
        """Populate the gem list with available structural gems"""
        self.gem_listbox.delete(0, tk.END)
        
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
            
            display_text = f"Gem {base_id} - {'+'.join(sources)} - {total} structural files"
            self.gem_listbox.insert(tk.END, display_text)
        
        self.status_var.set(f"Found {len(self.gem_groups)} unique structural gems")
    
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
    
    def open_enhanced_file_dialog(self, base_id):
        """Open enhanced file selection dialog with feature preview"""
        gem_data = self.gem_groups[base_id]
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Enhanced File Selection - Gem {base_id}")
        dialog.geometry("800x700")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - 400
        y = (dialog.winfo_screenheight() // 2) - 350
        dialog.geometry(f"800x700+{x}+{y}")
        
        # Title
        ttk.Label(dialog, text=f"Enhanced Structural Analysis - Gem {base_id}", 
                 font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Info
        ttk.Label(dialog, text="Select structural files with advanced feature weighting", 
                 font=('Arial', 10), foreground='blue').pack(pady=5)
        
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
        options_frame = ttk.LabelFrame(dialog, text="Analysis Options", padding="5")
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
                    'multipoint_analysis': multipoint_var.get()
                }
            }
            
            self.update_selected_display()
            dialog.destroy()
        
        def cancel_selection():
            dialog.destroy()
        
        ttk.Button(button_frame, text="Cancel", command=cancel_selection).pack(side='right', padx=(5, 0))
        ttk.Button(button_frame, text="Confirm Enhanced Selection", command=confirm_selection).pack(side='right')
    
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
            
            options_str = f"[{'+'.join(option_indicators)}]" if option_indicators else ""
            
            display_text = f"Gem {base_id} ({'+'.join(light_sources)}) {options_str} - {' '.join(file_details)}"
            self.selected_listbox.insert(tk.END, display_text)
        
        self.status_var.set(f"Selected {len(self.selected_gems)} gems for enhanced analysis")
    
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
    
    def start_analysis(self):
        """Start the enhanced structural database matching analysis"""
        if not self.selected_gems:
            messagebox.showwarning("No Selection", "Please select at least one gem for enhanced analysis.")
            return
        
        if not self.check_database():
            return
        
        # Enhanced confirmation
        gem_count = len(self.selected_gems)
        if not messagebox.askyesno("Start Enhanced Analysis", 
                                  f"Start enhanced structural analysis for {gem_count} gems?\n\n"
                                  f"This will use:\n"
                                  f"• Advanced feature weighting\n"
                                  f"• Multi-point scoring algorithms\n"
                                  f"• Spectral normalization\n"
                                  f"• Comprehensive database matching\n\n"
                                  f"Analysis may take several minutes."):
            return
        
        print(f"\n🔬 Starting enhanced structural analysis for {gem_count} gems...")
        
        try:
            self.run_enhanced_analysis()
            messagebox.showinfo("Analysis Complete", 
                              f"Enhanced analysis completed for {gem_count} gems!\n\n"
                              f"Results saved to outputs/structural_results/\n"
                              f"Check reports for detailed scoring breakdown.")
        except Exception as e:
            print(f"❌ Enhanced analysis error: {e}")
            messagebox.showerror("Analysis Error", f"Enhanced analysis failed:\n{e}")
    
    def run_enhanced_analysis(self):
        """Run the enhanced structural database matching analysis"""
        # Create output directories
        results_dir = self.project_root / "outputs" / "structural_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load database with enhanced schema checking
        print(f"📊 Loading enhanced structural database...")
        conn = sqlite3.connect(self.database_path)
        
        # Get database table info
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        if not tables:
            raise Exception("No tables found in structural database")
        
        table_name = tables[0][0]
        print(f"📊 Using table: {table_name}")
        
        # Check for advanced schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        has_advanced_schema = 'feature_group' in columns and 'processing' in columns
        
        # Load database with appropriate query
        if has_advanced_schema:
            db_df = pd.read_sql_query(f"""
                SELECT * FROM {table_name} 
                WHERE processing LIKE '%Normalized%' OR processing IS NULL
                ORDER BY file, feature_group, wavelength
            """, conn)
            print(f"📊 Enhanced database: {len(db_df)} normalized structural records")
        else:
            db_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print(f"📊 Basic database: {len(db_df)} structural records")
        
        # Analyze each selected gem
        all_results = []
        
        for base_id, data in self.selected_gems.items():
            print(f"\n🔍 Enhanced Analysis: Gem {base_id}...")
            
            selected_files = data['selected_files']
            selected_paths = data['selected_paths']
            options = data.get('options', {})
            
            gem_results = {
                'gem_id': base_id,
                'analysis_timestamp': timestamp,
                'analysis_type': 'enhanced',
                'options_used': options,
                'light_source_results': {},
                'best_overall_match': None,
                'lowest_combined_score': float('inf'),
                'feature_analysis': {}
            }
            
            # Analyze each light source with enhanced methods
            for light_source, file_path in selected_paths.items():
                print(f"   📄 Enhanced processing {light_source}: {file_path.name}")
                
                try:
                    # Load unknown gem data
                    unknown_df = pd.read_csv(file_path)
                    print(f"      📊 Loaded {len(unknown_df)} structural points")
                    
                    # Apply normalization if requested
                    if options.get('normalize', True):
                        unknown_df = self.normalize_spectrum(unknown_df, light_source)
                        print(f"      🔧 Applied normalization for {light_source}")
                    
                    # Extract enhanced features
                    features = self.extract_enhanced_features(unknown_df, light_source, options)
                    gem_results['feature_analysis'][light_source] = features
                    
                    # Find matching database gems
                    db_matches = self.find_enhanced_database_matches(db_df, light_source, table_name)
                    print(f"      🔍 Found {len(db_matches)} database gems with {light_source}")
                    
                    if db_matches.empty:
                        print(f"      ⚠️ No database matches for light source {light_source}")
                        continue
                    
                    # Calculate enhanced scores
                    scores = self.calculate_enhanced_scores(unknown_df, db_matches, light_source, options, has_advanced_schema)
                    
                    if scores:
                        best_match = min(scores, key=lambda x: x['score'])
                        print(f"      ✅ Best match: {best_match['db_gem_id']} (score: {best_match['score']:.4f})")
                        
                        gem_results['light_source_results'][light_source] = {
                            'file_analyzed': file_path.name,
                            'features_extracted': features,
                            'best_match': best_match,
                            'top_5_matches': sorted(scores, key=lambda x: x['score'])[:5]
                        }
                    
                except Exception as e:
                    print(f"      ❌ Error processing {light_source}: {e}")
                    continue
            
            # Calculate enhanced combined score
            if len(gem_results['light_source_results']) > 1:
                combined_scores = self.calculate_enhanced_combined_scores(gem_results['light_source_results'])
                if combined_scores:
                    best_combined = min(combined_scores, key=lambda x: x['combined_score'])
                    gem_results['best_overall_match'] = best_combined
                    gem_results['lowest_combined_score'] = best_combined['combined_score']
                    print(f"   🏆 Best enhanced match: {best_combined['db_gem_id']} (score: {best_combined['combined_score']:.4f})")
            
            all_results.append(gem_results)
        
        conn.close()
        
        # Save enhanced results
        self.save_enhanced_results(all_results, results_dir, timestamp)
        
        print(f"\n🎉 Enhanced analysis completed!")
        print(f"📊 Results saved to: {results_dir}")
    
    def extract_enhanced_features(self, df, light_source, options):
        """Extract enhanced structural features with weighting"""
        features = {
            'light_source': light_source,
            'data_points': len(df),
            'feature_weights_used': options.get('feature_weighting', True),
            'multipoint_analysis': options.get('multipoint_analysis', True)
        }
        
        # Extract wavelength and intensity
        if 'Wavelength' in df.columns and 'Intensity' in df.columns:
            wavelength = df['Wavelength'].values
            intensity = df['Intensity'].values
        else:
            wavelength = df.iloc[:, 0].values
            intensity = df.iloc[:, 1].values
        
        features['wavelength_range'] = [float(wavelength.min()), float(wavelength.max())]
        features['intensity_stats'] = {
            'mean': float(np.mean(intensity)),
            'std': float(np.std(intensity)),
            'max': float(np.max(intensity)),
            'min': float(np.min(intensity))
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
        
        # Peak detection with enhanced algorithms
        try:
            peaks, peak_properties = find_peaks(intensity, height=np.percentile(intensity, 75), distance=5)
            features['peaks'] = {
                'count': len(peaks),
                'positions': [float(wavelength[i]) for i in peaks],
                'intensities': [float(intensity[i]) for i in peaks],
                'prominences': peak_properties.get('peak_heights', []).tolist() if 'peak_heights' in peak_properties else []
            }
        except Exception:
            # Fallback simple peak detection
            threshold = np.percentile(intensity, 75)
            peaks = []
            for i in range(1, len(intensity) - 1):
                if intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1] and intensity[i] > threshold:
                    peaks.append(i)
            
            features['peaks'] = {
                'count': len(peaks),
                'positions': [float(wavelength[i]) for i in peaks],
                'intensities': [float(intensity[i]) for i in peaks]
            }
        
        return features
    
    def find_enhanced_database_matches(self, db_df, light_source, table_name):
        """Find database matches with enhanced filtering"""
        # Look for identifier column
        identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
        identifier_col = None
        
        for col in identifier_columns:
            if col in db_df.columns:
                identifier_col = col
                break
        
        if not identifier_col:
            # Use first string column
            for col in db_df.columns:
                if db_df[col].dtype == 'object':
                    identifier_col = col
                    break
        
        if not identifier_col:
            return pd.DataFrame()
        
        # Enhanced filtering by light source
        light_mappings = {
            'B': ['B', 'Halogen', 'halogen'],
            'L': ['L', 'Laser', 'laser'], 
            'U': ['U', 'UV', 'uv']
        }
        
        search_terms = light_mappings.get(light_source, [light_source])
        mask = pd.Series([False] * len(db_df))
        
        for term in search_terms:
            mask |= db_df[identifier_col].str.contains(term, case=False, na=False)
        
        return db_df[mask].copy()
    
    def calculate_enhanced_scores(self, unknown_df, db_matches, light_source, options, has_advanced_schema):
        """Calculate enhanced similarity scores with advanced algorithms"""
        scores = []
        
        for _, db_row in db_matches.iterrows():
            try:
                # Get database gem identifier
                identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
                db_gem_id = None
                
                for col in identifier_columns:
                    if col in db_row.index and pd.notna(db_row[col]):
                        db_gem_id = str(db_row[col])
                        break
                
                if not db_gem_id:
                    continue
                
                # Calculate enhanced score
                if has_advanced_schema and options.get('feature_weighting', True):
                    score = self.compute_advanced_similarity_score(unknown_df, db_row, options)
                else:
                    score = self.compute_basic_similarity_score(unknown_df, db_row)
                
                scores.append({
                    'db_gem_id': db_gem_id,
                    'score': score,
                    'light_source': light_source,
                    'analysis_method': 'advanced' if has_advanced_schema else 'basic',
                    'db_row_data': db_row.to_dict()
                })
                
            except Exception as e:
                print(f"        ⚠️ Error scoring against {db_gem_id}: {e}")
                continue
        
        return scores
    
    def compute_advanced_similarity_score(self, unknown_df, db_row, options):
        """Compute advanced similarity score with feature weighting"""
        score = 0.0
        comparison_count = 0
        
        # Method 1: Enhanced wavelength comparison
        if 'Wavelength' in unknown_df.columns:
            unknown_wl = unknown_df['Wavelength'].values
            unknown_int = unknown_df['Intensity'].values if 'Intensity' in unknown_df.columns else unknown_df.iloc[:, 1].values
            
            # Compare against database wavelength data
            wl_columns = [col for col in db_row.index if 'wavelength' in col.lower()]
            for wl_col in wl_columns:
                if pd.notna(db_row[wl_col]):
                    try:
                        db_wl = float(db_row[wl_col])
                        
                        # Find closest point in unknown spectrum
                        closest_idx = np.argmin(np.abs(unknown_wl - db_wl))
                        if np.abs(unknown_wl[closest_idx] - db_wl) < 5.0:  # Within 5nm
                            # Compare intensities
                            unknown_intensity = unknown_int[closest_idx]
                            db_intensity = db_row.get('intensity', db_row.get('Intensity', 1.0))
                            
                            if pd.notna(db_intensity):
                                db_intensity = float(db_intensity)
                                # Normalized intensity comparison
                                ratio = min(unknown_intensity, db_intensity) / max(unknown_intensity, db_intensity, 0.1)
                                score += (1.0 - ratio) * 0.3
                                comparison_count += 1
                        break
                    except:
                        continue
        
        # Method 2: Feature type weighting
        if 'Feature' in unknown_df.columns and options.get('feature_weighting', True):
            unknown_features = unknown_df['Feature'].value_counts().to_dict()
            
            # Check for feature information in database
            feature_columns = [col for col in db_row.index if 'feature' in col.lower()]
            if feature_columns:
                for feat_col in feature_columns:
                    if pd.notna(db_row[feat_col]):
                        db_feature_info = str(db_row[feat_col])
                        
                        # Calculate weighted feature similarity
                        feature_score = 0.0
                        total_weight = 0.0
                        
                        for unknown_feat, count in unknown_features.items():
                            weight = self.feature_weights.get(unknown_feat, 0.1)
                            
                            if unknown_feat.lower() in db_feature_info.lower():
                                feature_score += weight * count * 1.0  # Perfect match
                            else:
                                feature_score += weight * count * 0.0  # No match
                            
                            total_weight += weight * count
                        
                        if total_weight > 0:
                            normalized_feature_score = 1.0 - (feature_score / total_weight)
                            score += normalized_feature_score * 0.4
                            comparison_count += 1
                        break
        
        # Method 3: Statistical comparison with advanced metrics
        numerical_columns = ['Intensity', 'FWHM', 'Area', 'Height', 'intensity', 'fwhm', 'area', 'height']
        
        for col in numerical_columns:
            if col in unknown_df.columns and col in db_row.index:
                if pd.notna(db_row[col]):
                    unknown_values = unknown_df[col].dropna()
                    if len(unknown_values) > 0:
                        try:
                            db_value = float(db_row[col])
                            
                            # Use multiple statistical measures
                            unknown_mean = unknown_values.mean()
                            unknown_std = unknown_values.std()
                            
                            # Mean comparison
                            mean_diff = abs(unknown_mean - db_value) / max(abs(unknown_mean), abs(db_value), 1)
                            
                            # Check if db_value is within unknown distribution
                            if unknown_std > 0:
                                z_score = abs(db_value - unknown_mean) / unknown_std
                                distribution_score = min(1.0, z_score / 3.0)  # 3-sigma rule
                            else:
                                distribution_score = mean_diff
                            
                            combined_stat_score = (mean_diff + distribution_score) / 2
                            score += combined_stat_score * 0.2
                            comparison_count += 1
                            
                        except:
                            continue
        
        # Method 4: Multi-point analysis if enabled
        if options.get('multipoint_analysis', True) and 'point_type' in db_row.index:
            point_type = db_row.get('point_type', '')
            if point_type in self.wavelength_tolerances:
                # Use point-type specific tolerance
                tolerance = self.wavelength_tolerances[point_type]
                weight = self.feature_weights.get(point_type, 0.5)
                
                # Enhanced point-specific scoring
                point_score = self.calculate_point_specific_score(unknown_df, db_row, point_type, tolerance)
                score += point_score * weight * 0.1
                comparison_count += 1
        
        # Return average score or penalty for insufficient comparisons
        if comparison_count > 0:
            final_score = score / comparison_count
            
            # Apply confidence penalty based on comparison count
            confidence_factor = min(1.0, comparison_count / 3.0)
            return final_score / confidence_factor
        else:
            return 10.0  # High penalty for no valid comparisons
    
    def calculate_point_specific_score(self, unknown_df, db_row, point_type, tolerance):
        """Calculate point-type specific scoring"""
        try:
            db_wl = float(db_row.get('wavelength', db_row.get('Wavelength', 0)))
            db_int = float(db_row.get('intensity', db_row.get('Intensity', 0)))
            
            if 'Wavelength' in unknown_df.columns:
                unknown_wl = unknown_df['Wavelength'].values
                unknown_int = unknown_df['Intensity'].values if 'Intensity' in unknown_df.columns else unknown_df.iloc[:, 1].values
                
                # Find points within tolerance
                within_tolerance = np.abs(unknown_wl - db_wl) <= tolerance
                
                if np.any(within_tolerance):
                    matching_intensities = unknown_int[within_tolerance]
                    # Use best matching intensity within tolerance
                    best_match_intensity = matching_intensities[np.argmin(np.abs(matching_intensities - db_int))]
                    
                    # Calculate similarity
                    intensity_ratio = min(best_match_intensity, db_int) / max(best_match_intensity, db_int, 0.1)
                    return 1.0 - intensity_ratio
                else:
                    # Penalty for no points within tolerance
                    min_distance = np.min(np.abs(unknown_wl - db_wl))
                    return min(2.0, min_distance / tolerance)
            
            return 1.0  # Default moderate score
            
        except:
            return 1.0
    
    def compute_basic_similarity_score(self, unknown_df, db_row):
        """Compute basic similarity score for databases without advanced schema"""
        score = 0.0
        comparison_count = 0
        
        # Basic wavelength and intensity comparison
        numerical_columns = ['Wavelength', 'Intensity', 'wavelength', 'intensity']
        
        for col in numerical_columns:
            if col in unknown_df.columns and col in db_row.index:
                if pd.notna(db_row[col]):
                    unknown_values = unknown_df[col].dropna()
                    if len(unknown_values) > 0:
                        try:
                            db_value = float(db_row[col])
                            unknown_mean = unknown_values.mean()
                            diff = abs(unknown_mean - db_value) / max(abs(unknown_mean), abs(db_value), 1)
                            score += diff
                            comparison_count += 1
                        except:
                            continue
        
        return score / comparison_count if comparison_count > 0 else 5.0
    
    def calculate_enhanced_combined_scores(self, light_source_results):
        """Calculate enhanced combined scores with light source weighting"""
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
                    total_score += 5.0 * light_weight
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
    
    def save_enhanced_results(self, all_results, results_dir, timestamp):
        """Save enhanced analysis results with detailed breakdown"""
        # Create enhanced summary report
        summary_data = []
        detailed_data = []
        
        for result in all_results:
            gem_id = result['gem_id']
            options_used = result.get('options_used', {})
            
            # Enhanced summary row
            summary_row = {
                'Gem_ID': gem_id,
                'Analysis_Timestamp': timestamp,
                'Analysis_Type': 'Enhanced',
                'Normalization_Applied': options_used.get('normalize', False),
                'Feature_Weighting': options_used.get('feature_weighting', False),
                'Multipoint_Analysis': options_used.get('multipoint_analysis', False),
                'Light_Sources_Analyzed': '+'.join(result['light_source_results'].keys()),
                'Light_Source_Count': len(result['light_source_results'])
            }
            
            if result['best_overall_match']:
                summary_row.update({
                    'Best_Overall_Match': result['best_overall_match']['db_gem_id'],
                    'Enhanced_Combined_Score': result['best_overall_match']['combined_score'],
                    'Sources_Used': '+'.join(result['best_overall_match']['light_sources_used']),
                    'Completeness_Bonus': result['best_overall_match'].get('completeness_bonus', 0)
                })
            else:
                # Single light source case
                if result['light_source_results']:
                    ls_result = list(result['light_source_results'].values())[0]
                    summary_row.update({
                        'Best_Overall_Match': ls_result['best_match']['db_gem_id'],
                        'Enhanced_Combined_Score': ls_result['best_match']['score'],
                        'Sources_Used': list(result['light_source_results'].keys())[0],
                        'Completeness_Bonus': 0
                    })
            
            summary_data.append(summary_row)
            
            # Enhanced detailed rows
            for light_source, ls_data in result['light_source_results'].items():
                for i, match in enumerate(ls_data['top_5_matches']):
                    detailed_row = {
                        'Gem_ID': gem_id,
                        'Light_Source': light_source,
                        'File_Analyzed': ls_data['file_analyzed'],
                        'Rank': i + 1,
                        'DB_Gem_Match': match['db_gem_id'],
                        'Enhanced_Score': match['score'],
                        'Analysis_Method': match.get('analysis_method', 'basic'),
                        'Light_Weight': self.light_weights.get({'B': 'Halogen', 'L': 'Laser', 'U': 'UV'}[light_source], 1.0),
                        'Analysis_Timestamp': timestamp
                    }
                    detailed_data.append(detailed_row)
        
        # Save enhanced summary report
        summary_df = pd.DataFrame(summary_data)
        summary_file = results_dir / f"enhanced_structural_matching_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"📄 Enhanced summary saved: {summary_file.name}")
        
        # Save enhanced detailed report
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = results_dir / f"enhanced_structural_matching_detailed_{timestamp}.csv"
        detailed_df.to_csv(detailed_file, index=False)
        print(f"📄 Enhanced detailed results saved: {detailed_file.name}")
        
        # Save full enhanced results as JSON
        json_file = results_dir / f"enhanced_structural_matching_full_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"📄 Enhanced full results saved: {json_file.name}")
        
        # Save analysis configuration
        config_file = results_dir / f"enhanced_analysis_config_{timestamp}.json"
        config_data = {
            'feature_weights': self.feature_weights,
            'light_weights': self.light_weights,
            'wavelength_tolerances': self.wavelength_tolerances,
            'analysis_timestamp': timestamp,
            'gems_analyzed': len(all_results)
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"📄 Analysis configuration saved: {config_file.name}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        print("Starting Enhanced Multi-Gem Structural Analyzer...")
        analyzer = EnhancedMultiGemStructuralAnalyzer()
        analyzer.run()
        print("Enhanced Structural Analyzer closed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
