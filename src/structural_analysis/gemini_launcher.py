#!/usr/bin/env python3
"""
Enhanced gemini_launcher.py - Added Analytical Workflow Option
Now includes the complete workflow for file selection and numerical analysis
"""

import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data path configuration if available
try:
    from data_path_config import gemini_paths, setup_project_paths
    HAS_PATH_CONFIG = True
    print("✅ Data path configuration available")
except ImportError:
    HAS_PATH_CONFIG = False
    print("⚠️  Data path configuration not available - using manual path resolution")

class GeminiLauncher:
    """Enhanced Gemini launcher with analytical workflow for file selection"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemini Analyzer - ENHANCED WITH ANALYTICAL WORKFLOW")
        self.root.geometry("600x800")  # Increased height for new option
        self.root.resizable(True, True)
        self.root.minsize(600, 800)
        
        # Get paths and setup data configuration
        self.script_dir = Path(__file__).parent.absolute()
        print(f"Launcher directory: {self.script_dir}")
        
        # Setup data paths if configuration is available
        if HAS_PATH_CONFIG:
            setup_project_paths()
            print(f"✅ Project paths configured - Raw data: {gemini_paths.raw_data}")
        
        # ENHANCED: Updated primary analysis choices to include analytical workflow
        self.primary_analysis = {
            "analytical": ("Analytical Workflow", "Complete workflow: file selection → conversion → numerical analysis"),
            "numerical": ("Direct Numerical Analysis", "Run gemini1.py with existing unkgem files"),
            "structural": ("Structural Analysis", "Feature detection and marking (manual or automated)")
        }
        
        # Structural analysis methods (unchanged)
        self.structural_methods = {
            "manual": ("Manual Marking", "Interactive point-and-click feature marking"),
            "auto": ("Automated Detection", "Computational feature detection algorithms")
        }
        
        # Light sources (unchanged)
        self.light_sources = {
            "bh": ("B/H (Halogen)", "Broad mounds, plateaus, mineral identification"),
            "l": ("L (Laser)", "Sharp features, natural/synthetic detection"),
            "u": ("U (UV)", "Electronic transitions, color centers")
        }
        
        # ENHANCED: Updated file mappings to include analytical workflow
        self.file_mappings = {
            "analytical": {
                # NEW: Analytical workflow option
                "paths": [
                    "analytical_workflow.py",
                    "../analytical_workflow.py",
                    "../../analytical_workflow.py",
                    "../src/numerical_analysis/analytical_workflow.py",
                    "../../src/numerical_analysis/analytical_workflow.py"
                ],
                "name": "Analytical Workflow System"
            },
            "numerical": {
                # Direct numerical analysis (existing)
                "paths": [
                    "../numerical_analysis/gemini1.py",
                    "../../numerical_analysis/gemini1.py", 
                    "numerical_analysis/gemini1.py",
                    "../src/numerical_analysis/gemini1.py",
                    "../../src/numerical_analysis/gemini1.py",
                    "gemini1.py"
                ],
                "name": "Direct Gemini Identification"
            },
            "structural": {
                # Structural analysis (unchanged)
                "manual": {
                    "bh": {
                        "paths": [
                            "manual_analyzers/gemini_halogen_analyzer.py",
                            "../manual_analyzers/gemini_halogen_analyzer.py",
                            "../../manual_analyzers/gemini_halogen_analyzer.py",
                            "../structural_analysis/manual_analyzers/gemini_halogen_analyzer.py",
                            "gemini_halogen_analyzer.py"
                        ],
                        "name": "Manual Halogen Analyzer"
                    },
                    "l": {
                        "paths": [
                            "manual_analyzers/gemini_laser_analyzer.py",
                            "../manual_analyzers/gemini_laser_analyzer.py",
                            "../../manual_analyzers/gemini_laser_analyzer.py",
                            "../structural_analysis/manual_analyzers/gemini_laser_analyzer.py",
                            "gemini_laser_analyzer.py"
                        ],
                        "name": "Manual Laser Analyzer"
                    },
                    "u": {
                        "paths": [
                            "manual_analyzers/gemini_uv_analyzer.py",
                            "../manual_analyzers/gemini_uv_analyzer.py",
                            "../../manual_analyzers/gemini_uv_analyzer.py", 
                            "../structural_analysis/manual_analyzers/gemini_uv_analyzer.py",
                            "gemini_uv_analyzer.py"
                        ],
                        "name": "Manual UV Analyzer"
                    }
                },
                "auto": {
                    "bh": {
                        "paths": [
                            "auto_analysis/b_spectra_auto_detector.py",
                            "../auto_analysis/b_spectra_auto_detector.py",
                            "../../auto_analysis/b_spectra_auto_detector.py",
                            "../structural_analysis/auto_analysis/b_spectra_auto_detector.py",
                            "b_spectra_auto_detector.py"
                        ],
                        "name": "B Spectra Auto Detector"
                    },
                    "l": {
                        "paths": [
                            "auto_analysis/l_spectra_auto_detector.py",
                            "../auto_analysis/l_spectra_auto_detector.py", 
                            "../../auto_analysis/l_spectra_auto_detector.py",
                            "../structural_analysis/auto_analysis/l_spectra_auto_detector.py",
                            "l_spectra_auto_detector.py"
                        ],
                        "name": "L Spectra Auto Detector"
                    },
                    "u": {
                        "paths": [
                            "auto_analysis/gemini_peak_detector.py",
                            "../auto_analysis/gemini_peak_detector.py",
                            "../../auto_analysis/gemini_peak_detector.py", 
                            "../structural_analysis/auto_analysis/gemini_peak_detector.py",
                            "gemini_peak_detector.py"
                        ],
                        "name": "UV Peak Detector"
                    }
                }
            }
        }
        
        # Setup interface
        self.center_window()
        self.create_interface()
        
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (600 // 2)
        y = (screen_height // 2) - (800 // 2)
        self.root.geometry(f"600x800+{x}+{y}")
        print(f"Window sized for full visibility: 600x800")
        
    def create_interface(self):
        """Create enhanced interface with analytical workflow"""
        # Title - ENHANCED
        title_frame = tk.Frame(self.root, bg='darkblue')
        title_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(title_frame, text="GEMINI GEMOLOGICAL ANALYZER", 
                font=('Arial', 16, 'bold'), fg='white', bg='darkblue').pack(pady=8)
        tk.Label(title_frame, text="ENHANCED: Analytical Workflow → File Selection → Numerical Analysis", 
                font=('Arial', 9), fg='lightblue', bg='darkblue').pack()
        
        # Data path status if available
        if HAS_PATH_CONFIG:
            status_text = f"✅ Data paths configured - Raw data: {len(gemini_paths.get_raw_data_files())} files"
        else:
            status_text = "⚠️  Manual path resolution mode"
        tk.Label(title_frame, text=status_text, 
                font=('Arial', 8), fg='lightyellow', bg='darkblue').pack()
        
        # Button frame at bottom first
        self.button_frame = tk.Frame(self.root, bg='lightgray', relief='raised', bd=2)
        self.button_frame.pack(fill='x', side='bottom', pady=10)
        self.create_buttons()
        
        # Main content frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # STEP 1: Enhanced Primary Analysis Type
        self.create_primary_selection(main_frame)
        
        # STEP 2: Structural Method Selection (initially hidden)
        self.structural_frame = tk.LabelFrame(main_frame, text="Step 2: Structural Analysis Method", 
                                            font=('Arial', 11, 'bold'), padx=8, pady=8)
        
        self.structural_method_var = tk.StringVar(value="auto")
        self.create_structural_method_options()
        
        # STEP 3: Light Source Selection (initially hidden)  
        self.light_frame = tk.LabelFrame(main_frame, text="Step 3: Light Source Selection", 
                                       font=('Arial', 11, 'bold'), padx=8, pady=8)
        
        self.light_source_var = tk.StringVar(value="bh")
        self.create_light_source_options()
        
        # Info section
        self.create_info_section(main_frame)
        
    def create_primary_selection(self, parent):
        """Create Enhanced Step 1: Primary analysis type selection"""
        frame = tk.LabelFrame(parent, text="Step 1: Select Analysis Type", 
                             font=('Arial', 11, 'bold'), padx=8, pady=8)
        frame.pack(fill='x', pady=(0, 8))
        
        self.primary_var = tk.StringVar(value="analytical")  # Default to analytical workflow
        
        # Enhanced options with analytical workflow
        option_configs = {
            "analytical": {"icon": "🔬", "color": "darkgreen", "bg": "lightgreen"},
            "numerical": {"icon": "🔢", "color": "darkblue", "bg": "lightblue"}, 
            "structural": {"icon": "🎯", "color": "darkred", "bg": "lightcoral"}
        }
        
        for key, (name, description) in self.primary_analysis.items():
            option_frame = tk.Frame(frame)
            option_frame.pack(fill='x', pady=3)
            
            config = option_configs[key]
            
            # Highlight analytical workflow option
            if key == "analytical":
                highlight_frame = tk.Frame(option_frame, bg=config["bg"], relief='raised', bd=2)
                highlight_frame.pack(fill='x', padx=2, pady=2)
                container = highlight_frame
                
                tk.Label(highlight_frame, text="⭐ RECOMMENDED ⭐", 
                        font=('Arial', 8, 'bold'), fg='darkgreen', bg=config["bg"]).pack()
            else:
                container = option_frame
            
            tk.Radiobutton(container, text=f"{config['icon']} {name}", 
                          variable=self.primary_var, value=key,
                          font=('Arial', 10, 'bold'), fg=config["color"],
                          command=self.update_workflow).pack(anchor='w')
            tk.Label(container, text=f"   {description}", 
                    font=('Arial', 8), fg='gray').pack(anchor='w')
    
    def create_structural_method_options(self):
        """Create Step 2: Structural method options (unchanged)"""
        for key, (name, description) in self.structural_methods.items():
            option_frame = tk.Frame(self.structural_frame)
            option_frame.pack(fill='x', pady=3)
            
            icon = "🎯" if key == "manual" else "🤖"
            color = "darkgreen" if key == "manual" else "darkred"
            
            tk.Radiobutton(option_frame, text=f"{icon} {name}", 
                          variable=self.structural_method_var, value=key,
                          font=('Arial', 10, 'bold'), fg=color,
                          command=self.update_workflow).pack(anchor='w')
            tk.Label(option_frame, text=f"   {description}", 
                    font=('Arial', 8), fg='gray').pack(anchor='w')
    
    def create_light_source_options(self):
        """Create Step 3: Light source options (unchanged)"""
        for key, (name, description) in self.light_sources.items():
            option_frame = tk.Frame(self.light_frame)
            option_frame.pack(fill='x', pady=3)
            
            icons = {"bh": "🔥", "l": "⚡", "u": "🟣"}
            colors = {"bh": "red", "l": "blue", "u": "purple"}
            
            tk.Radiobutton(option_frame, text=f"{icons[key]} {name}", 
                          variable=self.light_source_var, value=key,
                          font=('Arial', 10, 'bold'), fg=colors[key],
                          command=self.update_workflow).pack(anchor='w')
            tk.Label(option_frame, text=f"   {description}", 
                    font=('Arial', 8), fg='gray').pack(anchor='w')
    
    def create_info_section(self, parent):
        """Create enhanced info section"""
        self.info_frame = tk.LabelFrame(parent, text="Configuration Summary", 
                                       font=('Arial', 11, 'bold'), padx=10, pady=10)
        self.info_frame.pack(fill='both', expand=True, pady=(10, 10))
        
        text_frame = tk.Frame(self.info_frame)
        text_frame.pack(fill='both', expand=True)
        
        self.info_text = tk.Text(text_frame, height=8, width=60, font=('Arial', 8), 
                                wrap='word', bg='lightyellow', relief='sunken', bd=1)
        
        scrollbar = tk.Scrollbar(text_frame, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.update_workflow()
    
    def create_buttons(self):
        """Create button controls"""
        # Dynamic status indicator
        self.status_label = tk.Label(self.button_frame, text="▼ ENHANCED WITH ANALYTICAL WORKFLOW ▼", 
                                    font=('Arial', 8, 'bold'), fg='darkgreen', bg='lightgray')
        self.status_label.pack(pady=(2,0))
        
        button_container = tk.Frame(self.button_frame, bg='lightgray')
        button_container.pack(fill='x', pady=5)
        
        # Main launch button
        self.launch_btn = tk.Button(button_container, text="🚀 LAUNCH ANALYZER", 
                                   font=('Arial', 14, 'bold'), 
                                   bg="green", fg="white", 
                                   command=self.launch_analyzer, 
                                   padx=40, pady=12)
        self.launch_btn.pack(side='left', padx=10)
        
        # Other buttons
        tk.Button(button_container, text="🔄 Reset", 
                 font=('Arial', 10), bg="lightyellow", fg="black", 
                 command=self.reset_selections, padx=15, pady=8).pack(side='left', padx=5)
        
        tk.Button(button_container, text="📁 Browse Files", 
                 font=('Arial', 10), bg="lightgreen", fg="black", 
                 command=self.browse_files, padx=15, pady=8).pack(side='left', padx=5)
        
        tk.Button(button_container, text="❓ Help", 
                 font=('Arial', 10), bg="lightblue", fg="black", 
                 command=self.show_help, padx=15, pady=8).pack(side='right', padx=10)
        
        tk.Button(button_container, text="❌ Exit", 
                 font=('Arial', 10), bg="lightcoral", fg="black", 
                 command=self.root.quit, padx=15, pady=8).pack(side='right', padx=5)
    
    def update_workflow(self):
        """Enhanced workflow update with analytical option"""
        primary_choice = self.primary_var.get()
        
        # Show/hide workflow steps based on primary choice
        if primary_choice in ["analytical", "numerical"]:
            # Hide structural options for analytical and numerical analysis
            self.structural_frame.pack_forget()
            self.light_frame.pack_forget()
            print(f"Workflow: {primary_choice} mode - Steps 2&3 hidden")
            
            # Update status label
            if hasattr(self, 'status_label'):
                if primary_choice == "analytical":
                    self.status_label.config(text="▼ ANALYTICAL WORKFLOW: Complete File Selection Process ▼")
                else:
                    self.status_label.config(text="▼ DIRECT NUMERICAL: Uses Existing unkgem Files ▼")
            
        else:  # structural
            # Show structural method selection
            self.structural_frame.pack(fill='x', pady=(0, 8))
            self.light_frame.pack(fill='x', pady=(0, 8))
            print("Workflow: Structural mode - Steps 2&3 visible")
            
            # Update status label
            if hasattr(self, 'status_label'):
                self.status_label.config(text="▼ STRUCTURAL ANALYSIS: All Steps Visible ▼")
        
        # Update info display
        self.update_info_display()
    
    def update_info_display(self):
        """Enhanced info display with analytical workflow details"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        
        primary_choice = self.primary_var.get()
        
        info_lines = []
        info_lines.append("ENHANCED WORKFLOW SUMMARY:")
        info_lines.append("=" * 45)
        
        if primary_choice == "analytical":
            info_lines.append("⭐ ANALYTICAL WORKFLOW (RECOMMENDED)")
            
            # Get file info
            paths = self.file_mappings["analytical"]["paths"]
            name = self.file_mappings["analytical"]["name"]
            found_file = self.find_file_from_paths(paths)
            
            info_lines.append("→ Will launch: Complete Analytical Workflow")
            if found_file:
                info_lines.append("→ File Status: ✅ FOUND")
                info_lines.append(f"→ Location: {found_file}")
            else:
                info_lines.append("→ File Status: ❌ MISSING")
            
            info_lines.append("")
            info_lines.append("COMPLETE WORKFLOW PROCESS:")
            info_lines.append("1. 📂 Scan data/raw directory for available spectra")
            info_lines.append("2. 🎯 Select specific B, L, U files for analysis")
            info_lines.append("3. 📋 Copy selected files to raw_txt directory")
            info_lines.append("4. 🔄 Convert .txt files to unkgem*.csv format")
            info_lines.append("5. 🧮 Run numerical analysis with proper normalization")
            info_lines.append("6. 📊 Display ranked gem matches with confidence scores")
            info_lines.append("")
            info_lines.append("ADVANTAGES:")
            info_lines.append("• Ensures you analyze the correct gem files")
            info_lines.append("• Proper normalization applied (B: 650nm, L: 450nm, U: 811nm)")
            info_lines.append("• No risk of analyzing old/wrong data")
            info_lines.append("• Complete traceability of analysis workflow")
            
        elif primary_choice == "numerical":
            info_lines.append("🔢 DIRECT NUMERICAL ANALYSIS")
            
            # Get file info
            paths = self.file_mappings["numerical"]["paths"]
            name = self.file_mappings["numerical"]["name"]
            found_file = self.find_file_from_paths(paths)
            
            info_lines.append("→ Will launch: gemini1.py directly")
            if found_file:
                info_lines.append("→ File Status: ✅ FOUND")
                info_lines.append(f"→ Location: {found_file}")
            else:
                info_lines.append("→ File Status: ❌ MISSING")
            
            info_lines.append("")
            info_lines.append("DIRECT ANALYSIS PROCESS:")
            info_lines.append("• Uses existing unkgemB.csv, unkgemL.csv, unkgemU.csv")
            info_lines.append("• No file selection or conversion")
            info_lines.append("• Assumes files are already properly normalized")
            info_lines.append("")
            info_lines.append("⚠️ CAUTION:")
            info_lines.append("• May analyze old/incorrect data if unkgem files exist")
            info_lines.append("• No verification of which gem is being analyzed")
            info_lines.append("• Recommend using Analytical Workflow instead")
            
        else:  # structural
            structural_method = self.structural_method_var.get()
            light_source = self.light_source_var.get()
            
            info_lines.append("🎯 STRUCTURAL ANALYSIS")
            info_lines.append(f"✅ Method: {self.structural_methods[structural_method][0]}")
            info_lines.append(f"✅ Light Source: {self.light_sources[light_source][0]}")
            
            # Get file info
            file_config = self.file_mappings["structural"][structural_method][light_source]
            paths = file_config["paths"]
            analyzer_name = file_config["name"]
            
            found_file = self.find_file_from_paths(paths)
            file_status = "✅ FOUND" if found_file else "❌ MISSING"
            
            info_lines.append(f"→ Will launch: {analyzer_name}")
            info_lines.append(f"→ File Status: {file_status}")
            if found_file:
                info_lines.append(f"→ Location: {found_file}")
            
            info_lines.append("")
            info_lines.append("STRUCTURAL ANALYSIS FEATURES:")
            
            if structural_method == "manual":
                info_lines.append("• Interactive point-and-click marking")
                info_lines.append("• Manual control over all features")
                info_lines.append("• Zoom, undo, persistent mode tools")
            else:
                info_lines.append("• Computational feature detection")
                info_lines.append("• Advanced algorithmic analysis")  
                info_lines.append("• Automated baseline, peaks, mounds")
            
            light_descriptions = {
                "bh": "• Broad absorption features and mounds\n• General mineral identification\n• Wide wavelength range analysis",
                "l": "• High-resolution sharp features\n• Natural vs synthetic detection\n• Precision wavelength analysis",
                "u": "• Electronic transitions and color centers\n• UV range spectral analysis\n• Sharp peak identification"
            }
            info_lines.append("")
            info_lines.append("LIGHT SOURCE SPECIALTY:")
            info_lines.append(light_descriptions[light_source])
        
        # Add data status if available
        if HAS_PATH_CONFIG:
            raw_files = gemini_paths.get_raw_data_files()
            info_lines.append("")
            info_lines.append("DATA STATUS:")
            info_lines.append(f"• Raw data directory: {len(raw_files)} files found")
            info_lines.append(f"• Location: {gemini_paths.raw_data}")
        
        self.info_text.insert(1.0, "\n".join(info_lines))
        self.info_text.config(state='disabled')
        
        # Update launch button text
        if primary_choice == "analytical":
            self.launch_btn.config(text="🚀 LAUNCH ANALYTICAL WORKFLOW")
        elif primary_choice == "numerical":
            self.launch_btn.config(text="🚀 LAUNCH GEMINI1.PY DIRECT")
        else:
            method_name = self.structural_methods[self.structural_method_var.get()][0]
            light_name = self.light_sources[self.light_source_var.get()][0].split()[0]
            self.launch_btn.config(text=f"🚀 LAUNCH {method_name.upper()} {light_name}")
    
    def find_file_from_paths(self, paths):
        """Find file from list of possible paths"""
        print(f"\n🔍 SEARCHING FOR FILE:")
        print(f"Script directory: {self.script_dir}")
        
        all_paths_to_try = []
        
        for path_str in paths:
            candidates = [
                Path(path_str),  # Absolute path
                self.script_dir / path_str,  # Relative to script dir
                self.script_dir.parent / path_str,  # Relative to parent
                Path.cwd() / path_str,  # Relative to current working directory
            ]
            all_paths_to_try.extend(candidates)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paths = []
        for path in all_paths_to_try:
            path_str = str(path)
            if path_str not in seen:
                seen.add(path_str)
                unique_paths.append(path)
        
        print(f"Trying {len(unique_paths)} possible locations:")
        for i, path in enumerate(unique_paths):
            exists = path.exists()
            print(f"  {i+1:2d}. {'✅' if exists else '❌'} {path}")
            if exists and path.is_file():
                print(f"🎯 FOUND: {path}")
                return path
        
        print("❌ File not found in any location")
        return None
    
    def launch_analyzer(self):
        """Enhanced launcher with analytical workflow support"""
        primary_choice = self.primary_var.get()
        
        if primary_choice == "analytical":
            # Launch analytical workflow
            paths = self.file_mappings["analytical"]["paths"]
            name = self.file_mappings["analytical"]["name"]
            found_file = self.find_file_from_paths(paths)
            
            if not found_file:
                self.show_file_not_found_dialog(name, paths)
                return
            
            self.run_analyzer(str(found_file), name)
            
        elif primary_choice == "numerical":
            # Launch gemini1.py directly
            paths = self.file_mappings["numerical"]["paths"]
            name = self.file_mappings["numerical"]["name"]
            found_file = self.find_file_from_paths(paths)
            
            if not found_file:
                self.show_file_not_found_dialog(name, paths)
                return
            
            self.run_analyzer(str(found_file), name)
            
        else:  # structural
            structural_method = self.structural_method_var.get()
            light_source = self.light_source_var.get()
            
            file_config = self.file_mappings["structural"][structural_method][light_source]
            paths = file_config["paths"]
            analyzer_name = file_config["name"]
            
            found_file = self.find_file_from_paths(paths)
            
            if not found_file:
                self.show_file_not_found_dialog(analyzer_name, paths)
                return
            
            self.run_analyzer(str(found_file), analyzer_name)
    
    def show_file_not_found_dialog(self, analyzer_name, searched_paths):
        """Show detailed file not found dialog"""
        error_msg = f"Could not find {analyzer_name}\n\n"
        error_msg += "Searched the following locations:\n"
        
        for i, path in enumerate(searched_paths[:5]):
            resolved_path = self.script_dir / path
            error_msg += f"{i+1}. {resolved_path}\n"
        
        if len(searched_paths) > 5:
            error_msg += f"... and {len(searched_paths) - 5} more locations\n"
            
        error_msg += f"\nCurrent directory: {self.script_dir}\n"
        error_msg += "\n💡 TIP: Use 'Browse Files' button to locate files manually"
        
        messagebox.showerror("File Not Found", error_msg)
    
    def run_analyzer(self, analyzer_file, analyzer_name):
        """Run the selected analyzer"""
        try:
            analyzer_path = Path(analyzer_file)
            print(f"\n🚀 LAUNCHING ANALYZER:")
            print(f"Name: {analyzer_name}")
            print(f"File: {analyzer_path}")
            print(f"Exists: {analyzer_path.exists()}")
            print(f"Is file: {analyzer_path.is_file()}")
            
            if not analyzer_path.exists():
                raise FileNotFoundError(f"File does not exist: {analyzer_path}")
            
            working_dir = str(analyzer_path.parent)
            print(f"Working directory: {working_dir}")
            
            cmd = [sys.executable, str(analyzer_path.absolute())]
            print(f"Command: {' '.join(cmd)}")
            
            if sys.platform == 'win32':
                process = subprocess.Popen(cmd, 
                                         cwd=working_dir,
                                         creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                process = subprocess.Popen(cmd, 
                                         cwd=working_dir,
                                         preexec_fn=os.setsid if hasattr(os, 'setsid') else None)
            
            print(f"✅ {analyzer_name} started (PID: {process.pid})")
            
            # Enhanced success message
            success_msg = f"✅ {analyzer_name} started successfully!\n\n"
            success_msg += f"PID: {process.pid}\n"
            success_msg += f"Location: {analyzer_path}\n"
            success_msg += f"Working Dir: {working_dir}\n"
            
            if HAS_PATH_CONFIG:
                raw_files = gemini_paths.get_raw_data_files()
                success_msg += f"\n📂 Data Status:\n"
                success_msg += f"Raw data files: {len(raw_files)}\n"
                success_msg += f"Data location: {gemini_paths.raw_data}"
            
            messagebox.showinfo("Analyzer Launched", success_msg)
            
            self.root.after(3000, self.minimize_launcher)
            
        except Exception as e:
            error_msg = f"Error launching {analyzer_name}:\n\n{str(e)}\n\n"
            error_msg += f"File: {analyzer_file}\n"
            error_msg += f"Working Dir: {working_dir if 'working_dir' in locals() else 'Unknown'}"
            
            messagebox.showerror("Launch Error", error_msg)
            print(f"❌ Launch Error: {e}")
    
    def minimize_launcher(self):
        """Minimize launcher after launch"""
        self.root.iconify()
        print("📱 Launcher minimized - restore to launch another analyzer")
    
    def browse_files(self):
        """Open file browser to current directory"""
        try:
            if sys.platform == 'win32':
                os.startfile(str(self.script_dir))
            elif sys.platform == 'darwin':  # macOS
                subprocess.run(['open', str(self.script_dir)])
            else:  # Linux
                subprocess.run(['xdg-open', str(self.script_dir)])
                
            print(f"📂 Opened file browser to: {self.script_dir}")
        except Exception as e:
            print(f"Error opening file browser: {e}")
    
    def reset_selections(self):
        """Reset all selections to defaults"""
        self.primary_var.set("analytical")  # Default to analytical workflow
        self.structural_method_var.set("auto")  
        self.light_source_var.set("bh")
        self.update_workflow()
        print("🔄 Reset to analytical workflow (recommended)")
    
    def show_help(self):
        """Show enhanced help with analytical workflow"""
        help_text = """ENHANCED GEMINI ANALYZER HELP - WITH ANALYTICAL WORKFLOW

🆕 NEW FEATURE: ANALYTICAL WORKFLOW
⭐ RECOMMENDED for most gem identification tasks

🔬 ANALYTICAL WORKFLOW (NEW):
• Complete end-to-end gem identification process
• Interactive file selection from data/raw directory
• Automatic file preparation and normalization
• Direct integration with numerical analysis
• Eliminates risk of analyzing wrong/old data

WORKFLOW STEPS:
1. 📂 Scans data/raw for available spectral files
2. 🎯 Shows files grouped by gem number
3. 📋 Lets you select specific B, L, U files 
4. 🔄 Copies files to raw_txt and converts to CSV
5. 🧮 Runs numerical analysis with proper normalization
6. 📊 Displays ranked gem matches with confidence

ADVANTAGES:
• Ensures correct file selection
• Proper normalization (B: 650nm, L: 450nm, U: 811nm)
• Complete traceability of analysis
• No risk of using old/incorrect data

🔢 DIRECT NUMERICAL ANALYSIS:
• Runs gemini1.py with existing unkgem*.csv files
• No file selection or conversion
• Use only if you know unkgem files are current

⚠️ CAUTION: May analyze old data if unkgem files exist

🎯 STRUCTURAL ANALYSIS:
• Feature detection and marking (unchanged)
• Manual or automated methods
• Light source specific analyzers

💡 RECOMMENDATIONS:

FOR GEM IDENTIFICATION:
→ Use "Analytical Workflow" (new, recommended)
→ Provides complete file selection and analysis

FOR STRUCTURAL ANALYSIS:
→ Use existing structural options
→ Manual for precise control
→ Automated for computational analysis

🔧 TROUBLESHOOTING:

File Not Found:
• Check project structure
• Use Browse Files to explore
• Verify analytical_workflow.py exists

Path Issues:
• Launcher searches multiple locations
• Check Configuration Summary for status
• Use absolute paths if needed

🚀 QUICK START:
1. Select "Analytical Workflow" (default)
2. Click "Launch Analytical Workflow" 
3. Select your gem files when prompted
4. View identification results

This enhanced launcher provides the complete workflow you need for accurate gem identification!"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Enhanced Workflow Help")
        help_window.geometry("900x800")
        help_window.resizable(True, True)
        
        text_frame = tk.Frame(help_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        help_text_widget = tk.Text(text_frame, font=('Arial', 9), wrap='word')
        scrollbar = tk.Scrollbar(text_frame, command=help_text_widget.yview)
        help_text_widget.config(yscrollcommand=scrollbar.set)
        
        help_text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        help_text_widget.insert(1.0, help_text)
        help_text_widget.config(state='disabled')
        
        tk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)
    
    def run(self):
        """Start the enhanced launcher"""
        print("🚀 Starting Enhanced Gemini Launcher with Analytical Workflow...")
        print(f"Python executable: {sys.executable}")
        print(f"Platform: {sys.platform}")
        if HAS_PATH_CONFIG:
            print(f"✅ Data path configuration active")
        else:
            print(f"⚠️  Manual path resolution mode")
        self.root.mainloop()

def main():
    """Main function"""
    try:
        launcher = GeminiLauncher()
        launcher.run()
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        messagebox.showerror("Error", f"Launcher error: {e}")

if __name__ == '__main__':
    main()
