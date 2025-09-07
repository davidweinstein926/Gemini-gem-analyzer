#!/usr/bin/env python3
"""
main.py - COMPLETE GEMINI GEMOLOGICAL ANALYSIS SYSTEM
Root directory main program with structured gem selection interface
Updated for gemini_gemological_analysis directory structure
"""

import os
import sys
import subprocess
import sqlite3
import shutil
import pandas as pd
import numpy as np
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox
import re

class GeminiAnalysisSystem:
    def __init__(self):
        # Updated for your directory structure
        self.db_path = "database/structural_spectra/fixed_structural_gem_data.db"
        
        # System files to check - updated paths
        self.spectral_files = [
            'database/reference_spectra/gemini_db_long_B.csv', 
            'database/reference_spectra/gemini_db_long_L.csv', 
            'database/reference_spectra/gemini_db_long_U.csv'
        ]
        self.program_files = {
            'src/structural_analysis/gemini_launcher.py': 'Structural Analyzers Launcher',
            'src/numerical_analysis/gemini1.py': 'Numerical Analysis Engine',
        }
    
    def check_system_status(self):
        """Check overall system status"""
        print("GEMINI GEMOLOGICAL ANALYSIS SYSTEM STATUS")
        print("=" * 50)
        
        # Check database files
        db_files_ok = 0
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                size = os.path.getsize(db_file) // (1024*1024)  # MB
                print(f"✅ {db_file} ({size} MB)")
                db_files_ok += 1
            else:
                print(f"❌ {db_file} (missing)")
        
        # Check program files
        programs_ok = 0
        for prog_file, description in self.program_files.items():
            if os.path.exists(prog_file):
                print(f"✅ {description}")
                programs_ok += 1
            else:
                print(f"❌ {description} (missing)")
        
        # Check data directories - updated paths
        data_dirs = ['data/raw', 'data/unknown', 'data/raw_txt']
        for data_dir in data_dirs:
            if os.path.exists(data_dir):
                files = len([f for f in os.listdir(data_dir) if f.endswith('.txt') or f.endswith('.csv')])
                print(f"✅ {data_dir} ({files} files)")
            else:
                print(f"❌ {data_dir} (missing)")
        
        print(f"\nSystem Status: {db_files_ok}/3 databases, {programs_ok}/{len(self.program_files)} programs")
        print("=" * 50)
        
        return db_files_ok >= 3 and programs_ok >= 2
    
    def scan_gem_structure(self):
        """Scan data/raw to understand gem structure: gems, orientations, scan numbers"""
        raw_dir = 'data/raw'
        if not os.path.exists(raw_dir):
            return None
            
        files = [f for f in os.listdir(raw_dir) if f.endswith('.txt')]
        if not files:
            return None
        
        # Parse files to extract structure
        gem_structure = {}
        
        for file in files:
            base = os.path.splitext(file)[0]
            
            # Try to parse filename pattern: [GemID][Light][Orientation][ScanNum]
            # Examples: C0018BC1, 58BP2, C0019LC1
            match = re.match(r'^([A-Z]*\d+)([BLU])([CP])(\d+)$', base.upper())
            
            if match:
                gem_id, light, orientation, scan_num = match.groups()
                
                if gem_id not in gem_structure:
                    gem_structure[gem_id] = {'B': {}, 'L': {}, 'U': {}}
                
                if orientation not in gem_structure[gem_id][light]:
                    gem_structure[gem_id][light][orientation] = []
                
                gem_structure[gem_id][light][orientation].append(scan_num)
        
        # Sort scan numbers numerically
        for gem_id in gem_structure:
            for light in gem_structure[gem_id]:
                for orientation in gem_structure[gem_id][light]:
                    gem_structure[gem_id][light][orientation].sort(key=int)
        
        return gem_structure
    
    def select_analysis_files_gui(self):
        """GUI for structured gem selection"""
        print("\n🎯 Launching structured gem selection interface...")
        
        gem_structure = self.scan_gem_structure()
        if not gem_structure:
            print("❌ No valid gem files found in data/raw/")
            return None
        
        selector = StructuredGemSelector(gem_structure)
        return selector.run()

class StructuredGemSelector:
    """GUI for structured gem file selection"""
    
    def __init__(self, gem_structure):
        self.gem_structure = gem_structure
        self.selected_files = {'B': None, 'L': None, 'U': None}
        self.result = None
        
        self.root = tk.Tk()
        self.root.title("Gemini Structured Gem Selector")
        self.root.geometry("900x700")
        
        self.create_interface()
    
    def create_interface(self):
        """Create structured selection interface"""
        
        # Title
        title_frame = tk.Frame(self.root, bg='darkblue', height=60)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text="GEMINI STRUCTURED GEM SELECTOR", 
                font=('Arial', 16, 'bold'), fg='white', bg='darkblue').pack(pady=15)
        
        # Available gems display
        self.create_gems_display()
        
        # Selection interface
        self.create_selection_interface()
        
        # Selection summary
        self.create_selection_summary()
        
        # Buttons
        self.create_buttons()
    
    def create_gems_display(self):
        """Show available gems"""
        gems_frame = tk.LabelFrame(self.root, text="Available Gems", font=('Arial', 12, 'bold'))
        gems_frame.pack(fill='x', padx=20, pady=10)
        
        gems_text = "Found gems: " + ", ".join(sorted(self.gem_structure.keys()))
        tk.Label(gems_frame, text=gems_text, font=('Arial', 10), wraplength=800).pack(pady=5)
    
    def create_selection_interface(self):
        """Create selection dropdowns for each light source"""
        
        selection_frame = tk.LabelFrame(self.root, text="Build Your Selection", font=('Arial', 12, 'bold'))
        selection_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create selection for each light source
        self.selectors = {}
        colors = {'B': '#FF6B35', 'L': '#004E98', 'U': '#7209B7'}
        
        for i, light in enumerate(['B', 'L', 'U']):
            light_frame = tk.LabelFrame(selection_frame, text=f"{light} Light Source", 
                                       font=('Arial', 11, 'bold'), fg=colors[light])
            light_frame.grid(row=0, column=i, sticky='nsew', padx=10, pady=10)
            
            selection_frame.grid_columnconfigure(i, weight=1)
            
            # Gem selection
            tk.Label(light_frame, text="Gem:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=2)
            gem_var = tk.StringVar()
            gem_combo = ttk.Combobox(light_frame, textvariable=gem_var, state='readonly', width=15)
            gem_combo['values'] = sorted(self.gem_structure.keys())
            gem_combo.grid(row=0, column=1, padx=5, pady=2)
            gem_combo.bind('<<ComboboxSelected>>', lambda e, ls=light: self.on_gem_change(ls))
            
            # Orientation selection
            tk.Label(light_frame, text="Orientation:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=2)
            orientation_var = tk.StringVar()
            orientation_combo = ttk.Combobox(light_frame, textvariable=orientation_var, state='readonly', width=15)
            orientation_combo['values'] = ['C (Crown)', 'P (Pavilion)']
            orientation_combo.grid(row=1, column=1, padx=5, pady=2)
            orientation_combo.bind('<<ComboboxSelected>>', lambda e, ls=light: self.on_orientation_change(ls))
            
            # Scan number selection
            tk.Label(light_frame, text="Scan #:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=2)
            scan_var = tk.StringVar()
            scan_combo = ttk.Combobox(light_frame, textvariable=scan_var, state='readonly', width=15)
            scan_combo.grid(row=2, column=1, padx=5, pady=2)
            scan_combo.bind('<<ComboboxSelected>>', lambda e, ls=light: self.on_scan_change(ls))
            
            # Current filename display
            filename_var = tk.StringVar(value="(not selected)")
            filename_label = tk.Label(light_frame, textvariable=filename_var, 
                                    font=('Arial', 10), fg='gray', relief='sunken', bd=1)
            filename_label.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
            
            # File exists indicator
            exists_var = tk.StringVar()
            exists_label = tk.Label(light_frame, textvariable=exists_var, font=('Arial', 9))
            exists_label.grid(row=4, column=0, columnspan=2, pady=2)
            
            # Store references
            self.selectors[light] = {
                'gem_var': gem_var,
                'gem_combo': gem_combo,
                'orientation_var': orientation_var,
                'orientation_combo': orientation_combo,
                'scan_var': scan_var,
                'scan_combo': scan_combo,
                'filename_var': filename_var,
                'filename_label': filename_label,
                'exists_var': exists_var,
                'exists_label': exists_label
            }
    
    def on_gem_change(self, light):
        """Handle gem selection change"""
        gem_id = self.selectors[light]['gem_var'].get()
        
        if gem_id:
            # Update orientation options based on available data
            orientations = []
            for orientation in ['C', 'P']:
                if orientation in self.gem_structure[gem_id][light]:
                    orientations.append(f"{orientation} ({'Crown' if orientation == 'C' else 'Pavilion'})")
            
            self.selectors[light]['orientation_combo']['values'] = orientations
            self.selectors[light]['orientation_var'].set('')
            self.selectors[light]['scan_var'].set('')
            self.selectors[light]['scan_combo']['values'] = []
        
        self.update_filename(light)
    
    def on_orientation_change(self, light):
        """Handle orientation selection change"""
        gem_id = self.selectors[light]['gem_var'].get()
        orientation_text = self.selectors[light]['orientation_var'].get()
        
        if gem_id and orientation_text:
            orientation = orientation_text[0]  # Extract 'C' or 'P' from 'C (Crown)'
            
            # Update scan number options
            if orientation in self.gem_structure[gem_id][light]:
                scans = self.gem_structure[gem_id][light][orientation]
                self.selectors[light]['scan_combo']['values'] = scans
            else:
                self.selectors[light]['scan_combo']['values'] = []
            
            self.selectors[light]['scan_var'].set('')
        
        self.update_filename(light)
    
    def on_scan_change(self, light):
        """Handle scan number selection change"""
        self.update_filename(light)
    
    def update_filename(self, light):
        """Update filename display and check if file exists"""
        gem_id = self.selectors[light]['gem_var'].get()
        orientation_text = self.selectors[light]['orientation_var'].get()
        scan_num = self.selectors[light]['scan_var'].get()
        
        if gem_id and orientation_text and scan_num:
            orientation = orientation_text[0]  # Extract 'C' or 'P'
            filename = f"{gem_id}{light}{orientation}{scan_num}.txt"
            
            self.selectors[light]['filename_var'].set(filename)
            self.selectors[light]['filename_label'].config(fg='black')
            
            # Check if file exists
            file_path = os.path.join('data/raw', filename)
            if os.path.exists(file_path):
                self.selectors[light]['exists_var'].set("✅ File exists")
                self.selectors[light]['exists_label'].config(fg='green')
                self.selected_files[light] = filename
            else:
                self.selectors[light]['exists_var'].set("❌ File not found")
                self.selectors[light]['exists_label'].config(fg='red')
                self.selected_files[light] = None
        else:
            self.selectors[light]['filename_var'].set("(not selected)")
            self.selectors[light]['filename_label'].config(fg='gray')
            self.selectors[light]['exists_var'].set("")
            self.selected_files[light] = None
        
        self.update_selection_summary()
    
    def create_selection_summary(self):
        """Create selection summary display"""
        
        summary_frame = tk.LabelFrame(self.root, text="Final Selection", font=('Arial', 12, 'bold'))
        summary_frame.pack(fill='x', padx=20, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=4, font=('Courier', 10), bg='lightyellow')
        self.summary_text.pack(fill='x', padx=10, pady=10)
        
        self.update_selection_summary()
    
    def update_selection_summary(self):
        """Update selection summary"""
        self.summary_text.delete(1.0, tk.END)
        
        valid_selections = 0
        summary = "Selected files for analysis:\n"
        
        for light in ['B', 'L', 'U']:
            filename = self.selected_files[light]
            if filename:
                summary += f"  {light}: {filename}\n"
                valid_selections += 1
            else:
                summary += f"  {light}: (not selected)\n"
        
        summary += f"\nValid selections: {valid_selections}/3"
        
        self.summary_text.insert(1.0, summary)
    
    def create_buttons(self):
        """Create action buttons"""
        
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill='x', padx=20, pady=10)
        
        # Clear all
        clear_btn = tk.Button(button_frame, text="Clear All", 
                             command=self.clear_all,
                             bg='orange', fg='white', font=('Arial', 11))
        clear_btn.pack(side='left', padx=5)
        
        # Quick fill (same gem, same scan)
        quick_btn = tk.Button(button_frame, text="Quick Fill", 
                             command=self.quick_fill,
                             bg='blue', fg='white', font=('Arial', 11))
        quick_btn.pack(side='left', padx=5)
        
        # Analyze
        analyze_btn = tk.Button(button_frame, text="Analyze Selected Files", 
                               command=self.analyze_files,
                               bg='green', fg='white', font=('Arial', 12, 'bold'))
        analyze_btn.pack(side='right', padx=5)
        
        # Cancel
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                              command=self.cancel,
                              bg='red', fg='white', font=('Arial', 11))
        cancel_btn.pack(side='right', padx=5)
    
    def clear_all(self):
        """Clear all selections"""
        for light in ['B', 'L', 'U']:
            self.selectors[light]['gem_var'].set('')
            self.selectors[light]['orientation_var'].set('')
            self.selectors[light]['scan_var'].set('')
            self.selectors[light]['orientation_combo']['values'] = []
            self.selectors[light]['scan_combo']['values'] = []
            self.update_filename(light)
    
    def quick_fill(self):
        """Quick fill with same gem ID for all light sources"""
        # Get first available gem
        if not self.gem_structure:
            return
        
        gem_id = sorted(self.gem_structure.keys())[0]
        
        for light in ['B', 'L', 'U']:
            self.selectors[light]['gem_var'].set(gem_id)
            self.on_gem_change(light)
            
            # Try to set Crown orientation if available
            if 'C' in self.gem_structure[gem_id][light]:
                self.selectors[light]['orientation_var'].set('C (Crown)')
                self.on_orientation_change(light)
                
                # Try to set scan 1 if available
                if '1' in self.gem_structure[gem_id][light]['C']:
                    self.selectors[light]['scan_var'].set('1')
                    self.on_scan_change(light)
    
    def analyze_files(self):
        """Confirm and proceed with analysis"""
        valid_files = {k: v for k, v in self.selected_files.items() if v is not None}
        
        if not valid_files:
            messagebox.showwarning("No Selection", "Please select at least one valid file for analysis")
            return
        
        if len(valid_files) < 3:
            message = f"Only {len(valid_files)} light sources selected. Analysis works best with all 3 (B, L, U).\n\nContinue anyway?"
            if not messagebox.askyesno("Incomplete Selection", message):
                return
        
        # Confirm selection
        message = "Proceed with analysis using these files?\n\n"
        for light in ['B', 'L', 'U']:
            filename = self.selected_files[light]
            if filename:
                message += f"{light}: {filename}\n"
        
        if messagebox.askyesno("Confirm Analysis", message):
            self.result = valid_files
            self.root.quit()
    
    def cancel(self):
        """Cancel selection"""
        self.result = None
        self.root.quit()
    
    def run(self):
        """Run the selector and return result"""
        self.root.mainloop()
        self.root.destroy()
        return self.result

# Continue with rest of GeminiAnalysisSystem class
class GeminiAnalysisSystem(GeminiAnalysisSystem):
    
    def select_and_analyze_files(self):
        """Complete file selection and analysis workflow using structured GUI"""
        selected_files = self.select_analysis_files_gui()
        
        if not selected_files:
            print("❌ No files selected for analysis")
            return
        
        print(f"\n✅ Selected files:")
        for light, filename in selected_files.items():
            print(f"   {light}: {filename}")
        
        # Create analysis identifier
        file_bases = [os.path.splitext(f)[0] for f in selected_files.values()]
        analysis_id = "_".join(file_bases)
        
        # Convert files
        print(f"\n🔄 PREPARING FILES FOR ANALYSIS...")
        print("=" * 50)
        success = self.convert_selected_files(selected_files, analysis_id)
        
        if success:
            print(f"\n✅ FILES READY FOR ANALYSIS")
            choice = input(f"Run numerical analysis now? (y/n): ").strip().lower()
            
            if choice == 'y':
                self.run_numerical_analysis()
        else:
            print(f"\n❌ Failed to prepare files for analysis")
    
    def convert_selected_files(self, selected_files, analysis_id):
        """Convert selected files using txt_to_unkgem.py"""
        try:
            # Clear and create data/raw_txt
            if os.path.exists('data/raw_txt'):
                shutil.rmtree('data/raw_txt')
            os.makedirs('data/raw_txt')
            
            # Copy files to data/raw_txt
            print("   📁 Copying selected files to data/raw_txt...")
            for light, filename in selected_files.items():
                src = os.path.join('data/raw', filename)
                dst = os.path.join('data/raw_txt', filename)
                shutil.copy2(src, dst)
                print(f"     ✅ {light}: {filename}")
            
            # Create data/unknown directory
            os.makedirs('data/unknown', exist_ok=True)
            
            # Use txt_to_unkgem.py to convert files
            print("   🔧 Converting files using txt_to_unkgem.py...")
            
            # Check if txt_to_unkgem.py exists
            txt_to_unkgem_path = 'src/numerical_analysis/txt_to_unkgem.py'
            if os.path.exists(txt_to_unkgem_path):
                print(f"     Using {txt_to_unkgem_path}")
                result = subprocess.run([sys.executable, txt_to_unkgem_path], 
                                      capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("     ✅ Conversion completed successfully")
                    if result.stdout:
                        print("     Output:", result.stdout.strip())
                else:
                    print("     ⚠️ Conversion had issues:")
                    if result.stderr:
                        print("     Error:", result.stderr.strip())
                    if result.stdout:
                        print("     Output:", result.stdout.strip())
            else:
                print(f"     ⚠️ {txt_to_unkgem_path} not found, using built-in conversion...")
                # Fallback to built-in conversion
                self.builtin_conversion(selected_files)
            
            # Verify output files were created
            print("   📊 Verifying converted files...")
            for light in selected_files.keys():
                output_path = f'data/unknown/unkgem{light}.csv'
                if os.path.exists(output_path):
                    df = pd.read_csv(output_path, header=None)
                    print(f"     ✅ {light}: {output_path} ({len(df)} data points)")
                else:
                    print(f"     ❌ {light}: {output_path} not created")
            
            print(f"\n   ✅ Files ready for analysis!")
            print(f"   Analysis ID: {analysis_id}")
            return True
            
        except subprocess.TimeoutExpired:
            print("     ❌ Conversion timed out")
            return False
        except Exception as e:
            print(f"     ❌ Conversion error: {e}")
            return False
    
    def builtin_conversion(self, selected_files):
        """Fallback built-in conversion if txt_to_unkgem.py not available"""
        print("     Using built-in conversion...")
        
        for light, filename in selected_files.items():
            input_path = os.path.join('data/raw_txt', filename)
            output_path = f'data/unknown/unkgem{light}.csv'
            
            # Read file
            df = pd.read_csv(input_path, sep=r'\s+', header=None, names=['wavelength', 'intensity'])
            wavelengths = np.array(df['wavelength'])
            intensities = np.array(df['intensity'])
            
            # Apply normalization
            if light == 'B':
                # Halogen: 650nm → 50000
                idx = np.argmin(np.abs(wavelengths - 650))
                if intensities[idx] != 0:
                    normalized = intensities * (50000 / intensities[idx])
                else:
                    normalized = intensities
            elif light == 'L':
                # Laser: 450nm → 50000
                idx = np.argmin(np.abs(wavelengths - 450))
                if intensities[idx] != 0:
                    normalized = intensities * (50000 / intensities[idx])
                else:
                    normalized = intensities
            elif light == 'U':
                # UV: 811nm window → 15000
                mask = (wavelengths >= 810.5) & (wavelengths <= 811.5)
                window = intensities[mask]
                if len(window) > 0 and window.max() > 0:
                    normalized = intensities * (15000 / window.max())
                else:
                    normalized = intensities
            
            # Save normalized data
            output_df = pd.DataFrame({'wavelength': wavelengths, 'intensity': normalized})
            output_df.to_csv(output_path, header=False, index=False)
    
    def run_numerical_analysis(self):
        """Run numerical analysis - updated path"""
        print(f"\n🚀 RUNNING NUMERICAL ANALYSIS...")
        
        try:
            # Use the gemini1.py in src/numerical_analysis
            if os.path.exists('src/numerical_analysis/gemini1.py'):
                print("   Using src/numerical_analysis/gemini1.py...")
                result = subprocess.run([sys.executable, 'src/numerical_analysis/gemini1.py'], 
                                      timeout=120, capture_output=True, text=True)
                if result.stdout:
                    print("Results:")
                    print(result.stdout[-1000:])  # Show last 1000 chars
            else:
                print("   ❌ No analysis program found")
                
        except subprocess.TimeoutExpired:
            print("   ⚠️ Analysis timed out")
        except Exception as e:
            print(f"   ❌ Analysis error: {e}")
    
    def run_structural_launcher(self):
        """Launch structural analyzers launcher - updated path"""
        launcher_path = 'src/structural_analysis/gemini_launcher.py'
        if os.path.exists(launcher_path):
            try:
                subprocess.run([sys.executable, launcher_path])
            except Exception as e:
                print(f"Error launching structural launcher: {e}")
        else:
            print(f"❌ {launcher_path} not found")
    
    def run_analytical_workflow(self):
        """Run analytical workflow - updated path"""
        workflow_path = 'src/numerical_analysis/analytical_workflow.py'
        if os.path.exists(workflow_path):
            try:
                subprocess.run([sys.executable, workflow_path])
            except Exception as e:
                print(f"Error launching analytical workflow: {e}")
        else:
            print(f"❌ {workflow_path} not found")
    
    def show_database_stats(self):
        """Show database statistics - updated paths"""
        print("\n📊 DATABASE STATISTICS")
        print("=" * 30)
        
        for db_file in self.spectral_files:
            if os.path.exists(db_file):
                try:
                    df = pd.read_csv(db_file)
                    if 'full_name' in df.columns:
                        unique_gems = df['full_name'].nunique()
                        print(f"✅ {db_file}:")
                        print(f"   Records: {len(df):,}")
                        print(f"   Unique gems: {unique_gems}")
                    else:
                        print(f"⚠️ {db_file}: {len(df):,} records (no gem names)")
                except Exception as e:
                    print(f"❌ {db_file}: Error reading - {e}")
            else:
                print(f"❌ {db_file}: Missing")
        
        # Check structural database - updated path
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                count = cursor.execute("SELECT COUNT(*) FROM structural_features").fetchone()[0]
                print(f"\n✅ Structural database: {count:,} records")
                conn.close()
            except Exception as e:
                print(f"❌ Structural database error: {e}")
    
    def main_menu(self):
        """Main menu system"""
        
        menu_options = [
            ("🎯 Launch Structural Analyzers", self.run_structural_launcher),
            ("📊 Analytical Analysis Workflow", self.run_analytical_workflow),
            ("💎 Select Files for Analysis", self.select_and_analyze_files),
            ("🧮 Run Numerical Analysis (current files)", self.run_numerical_analysis),
            ("📈 Show Database Statistics", self.show_database_stats),
            ("❌ Exit", lambda: None)
        ]
        
        while True:
            print("\n" + "="*80)
            print("🔬 GEMINI GEMOLOGICAL ANALYSIS SYSTEM")
            print("="*80)
            
            # Show system status
            system_ok = self.check_system_status()
            
            print(f"\n📋 MAIN MENU:")
            print("-" * 40)
            
            for i, (description, _) in enumerate(menu_options, 1):
                print(f"{i:2}. {description}")
            
            # Get user choice
            try:
                choice = input(f"\nChoice (1-{len(menu_options)}): ").strip()
                choice_idx = int(choice) - 1
                
                if choice_idx == len(menu_options) - 1:  # Exit
                    print("\n👋 Goodbye!")
                    break
                
                if 0 <= choice_idx < len(menu_options) - 1:
                    description, action = menu_options[choice_idx]
                    print(f"\n🚀 {description.upper()}")
                    print("-" * 50)
                    
                    if action:
                        action()
                    
                    input("\n⏎ Press Enter to return to main menu...")
                else:
                    print("❌ Invalid choice")
                    
            except ValueError:
                print("❌ Please enter a number")
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user")
                break
            except Exception as e:
                print(f"\n❌ Menu error: {e}")

def main():
    """Main entry point"""
    try:
        print("🔬 Starting Gemini Gemological Analysis System...")
        system = GeminiAnalysisSystem()
        system.main_menu()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted - goodbye!")
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
