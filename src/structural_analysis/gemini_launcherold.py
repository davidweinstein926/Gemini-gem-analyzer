# gemini_launcher.py - LOGICAL WORKFLOW DESIGN
import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class GeminiLauncher:
    """Gemini launcher with logical workflow: Numerical → gemini1.py, Structural → Manual/Auto → Light Source"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemini Analyzer - RESIZED FOR FULL VISIBILITY")
        self.root.geometry("600x750")  # Increased height to fit all sections
        self.root.resizable(True, True)
        self.root.minsize(600, 750)  # Minimum size to prevent cutoff
        
        # Get paths
        self.script_dir = Path(__file__).parent.absolute()
        print(f"Launcher directory: {self.script_dir}")
        
        # STEP 1: Primary analysis choice
        self.primary_analysis = {
            "numerical": ("Numerical Analysis", "Gem identification using gemini1.py database matching"),
            "structural": ("Structural Analysis", "Feature detection and marking (manual or automated)")
        }
        
        # STEP 2: Structural analysis method (only shown if structural selected)
        self.structural_methods = {
            "manual": ("Manual Marking", "Interactive point-and-click feature marking"),
            "auto": ("Automated Detection", "Computational feature detection algorithms")
        }
        
        # STEP 3: Light sources (only shown if structural selected)
        self.light_sources = {
            "bh": ("B/H (Halogen)", "Broad mounds, plateaus, mineral identification"),
            "l": ("L (Laser)", "Sharp features, natural/synthetic detection"),
            "u": ("U (UV)", "Electronic transitions, color centers")
        }
        
        # File mappings based on your specifications
        self.file_mappings = {
            "numerical": {
                # Numerical analysis always launches gemini1.py
                "file": "src/numerical_analysis/gemini1.py",
                "fallback": "../numerical_analysis/gemini1.py",
                "name": "Gemini Identification System"
            },
            "structural": {
                "manual": {
                    "bh": ("structural_analysis/manual_analyzers/gemini_halogen_analyzer.py", 
                          "manual_analyzers/gemini_halogen_analyzer.py",
                          "Manual Halogen Analyzer"),
                    "l": ("structural_analysis/manual_analyzers/gemini_laser_analyzer.py",
                         "manual_analyzers/gemini_laser_analyzer.py", 
                         "Manual Laser Analyzer"),
                    "u": ("structural_analysis/manual_analyzers/gemini_uv_analyzer.py",
                         "manual_analyzers/gemini_uv_analyzer.py",
                         "Manual UV Analyzer")
                },
                "auto": {
                    "bh": ("structural_analysis/auto_analysis/b_spectra_auto_detector.py",
                          "auto_analysis/b_spectra_auto_detector.py",
                          "B Spectra Auto Detector"),
                    "l": ("structural_analysis/auto_analysis/l_spectra_auto_detector.py",
                         "auto_analysis/l_spectra_auto_detector.py",
                         "L Spectra Auto Detector"),
                    "u": ("structural_analysis/auto_analysis/gemini_peak_detector.py",
                         "auto_analysis/gemini_peak_detector.py",
                         "UV Peak Detector")
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
        y = (screen_height // 2) - (750 // 2)
        self.root.geometry(f"600x750+{x}+{y}")
        print(f"Window sized for full visibility: 600x750")
        
    def create_interface(self):
        """Create logical workflow interface"""
        # Title - COMPACT
        title_frame = tk.Frame(self.root, bg='darkblue')
        title_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(title_frame, text="GEMINI GEMOLOGICAL ANALYZER", 
                font=('Arial', 16, 'bold'), fg='white', bg='darkblue').pack(pady=8)
        tk.Label(title_frame, text="Logical Workflow: Numerical → Structural → Manual/Auto → Light Source", 
                font=('Arial', 9), fg='lightblue', bg='darkblue').pack()
        
        # Button frame at bottom first
        self.button_frame = tk.Frame(self.root, bg='lightgray', relief='raised', bd=2)
        self.button_frame.pack(fill='x', side='bottom', pady=10)
        self.create_buttons()
        
        # Main content frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # STEP 1: Primary Analysis Type
        self.create_primary_selection(main_frame)
        
        # STEP 2: Structural Method Selection (initially hidden)
        self.structural_frame = tk.LabelFrame(main_frame, text="Step 2: Structural Analysis Method", 
                                            font=('Arial', 11, 'bold'), padx=8, pady=8)
        # Don't pack initially - will be shown when structural is selected
        
        self.structural_method_var = tk.StringVar(value="auto")
        self.create_structural_method_options()
        
        # STEP 3: Light Source Selection (initially hidden)  
        self.light_frame = tk.LabelFrame(main_frame, text="Step 3: Light Source Selection", 
                                       font=('Arial', 11, 'bold'), padx=8, pady=8)
        # Don't pack initially
        
        self.light_source_var = tk.StringVar(value="bh")
        self.create_light_source_options()
        
        # Info section
        self.create_info_section(main_frame)
        
    def create_primary_selection(self, parent):
        """Create Step 1: Primary analysis type selection - COMPACT"""
        frame = tk.LabelFrame(parent, text="Step 1: Select Analysis Type", 
                             font=('Arial', 11, 'bold'), padx=8, pady=8)
        frame.pack(fill='x', pady=(0, 8))
        
        self.primary_var = tk.StringVar(value="numerical")
        
        for key, (name, description) in self.primary_analysis.items():
            option_frame = tk.Frame(frame)
            option_frame.pack(fill='x', pady=3)
            
            icon = "🔢" if key == "numerical" else "🔬"
            color = "darkgreen" if key == "numerical" else "darkblue"
            
            tk.Radiobutton(option_frame, text=f"{icon} {name}", 
                          variable=self.primary_var, value=key,
                          font=('Arial', 10, 'bold'), fg=color,
                          command=self.update_workflow).pack(anchor='w')
            tk.Label(option_frame, text=f"   {description}", 
                    font=('Arial', 8), fg='gray').pack(anchor='w')
    
    def create_structural_method_options(self):
        """Create Step 2: Structural method options - COMPACT"""
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
        """Create Step 3: Light source options - COMPACT"""
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
        """Create compact info section"""
        self.info_frame = tk.LabelFrame(parent, text="Configuration Summary", 
                                       font=('Arial', 11, 'bold'), padx=10, pady=10)
        self.info_frame.pack(fill='both', expand=True, pady=(10, 10))
        
        text_frame = tk.Frame(self.info_frame)
        text_frame.pack(fill='both', expand=True)
        
        # Reduced height to make room for workflow steps
        self.info_text = tk.Text(text_frame, height=5, width=60, font=('Arial', 8), 
                                wrap='word', bg='lightyellow', relief='sunken', bd=1)
        
        scrollbar = tk.Scrollbar(text_frame, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.update_workflow()
    
    def create_buttons(self):
        """Create button controls with dynamic visibility status"""
        # Dynamic status indicator - will be updated by update_workflow
        self.status_label = tk.Label(self.button_frame, text="▼ LOADING... ▼", 
                                    font=('Arial', 8, 'bold'), fg='darkblue', bg='lightgray')
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
        
        tk.Button(button_container, text="❓ Help", 
                 font=('Arial', 10), bg="lightblue", fg="black", 
                 command=self.show_help, padx=15, pady=8).pack(side='right', padx=10)
        
        tk.Button(button_container, text="❌ Exit", 
                 font=('Arial', 10), bg="lightcoral", fg="black", 
                 command=self.root.quit, padx=15, pady=8).pack(side='right', padx=5)
    
    def update_workflow(self):
        """Update the workflow display based on selections"""
        primary_choice = self.primary_var.get()
        
        # Show/hide workflow steps based on primary choice
        if primary_choice == "numerical":
            # Hide structural options for numerical analysis
            self.structural_frame.pack_forget()
            self.light_frame.pack_forget()
            print("Workflow: Numerical mode - Steps 2&3 hidden")
            
            # Update status label
            if hasattr(self, 'status_label'):
                self.status_label.config(text="▼ NUMERICAL MODE: Only Step 1 Needed ▼")
            
        else:  # structural
            # Show structural method selection with compact spacing
            self.structural_frame.pack(fill='x', pady=(0, 8))
            # Always show light source for structural analysis with compact spacing  
            self.light_frame.pack(fill='x', pady=(0, 8))
            print("Workflow: Structural mode - Steps 2&3 visible, including LIGHT SOURCE")
            
            # Update status label
            if hasattr(self, 'status_label'):
                self.status_label.config(text="▼ ALL SECTIONS VISIBLE: Steps 1, 2 (Method), 3 (Light Source) ▼")
        
        # Update info display
        self.update_info_display()
    
    def update_info_display(self):
        """Update the configuration summary"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        
        primary_choice = self.primary_var.get()
        
        info_lines = []
        info_lines.append("WORKFLOW SUMMARY:")
        info_lines.append("=" * 40)
        
        if primary_choice == "numerical":
            info_lines.append("✅ Step 1: Numerical Analysis")
            info_lines.append("→ Will launch: Gemini Identification System")
            info_lines.append("→ File: gemini1.py")
            info_lines.append("")
            info_lines.append("DESCRIPTION:")
            info_lines.append("• Raw spectrum to unknown format conversion")
            info_lines.append("• Database matching against reference spectra") 
            info_lines.append("• Gem identification with confidence scores")
            info_lines.append("• Comparison plots and detailed results")
            info_lines.append("")
            info_lines.append("WORKFLOW:")
            info_lines.append("1. Put raw files in: src/numerical_analysis/raw_txt/")
            info_lines.append("2. Launch gemini1.py for identification")
            
        else:  # structural
            structural_method = self.structural_method_var.get()
            light_source = self.light_source_var.get()
            
            info_lines.append("✅ Step 1: Structural Analysis")
            info_lines.append(f"✅ Step 2: {self.structural_methods[structural_method][0]}")
            info_lines.append(f"✅ Step 3: {self.light_sources[light_source][0]}")
            
            # Get file info
            file_config = self.file_mappings["structural"][structural_method][light_source]
            primary_file, fallback_file, analyzer_name = file_config
            
            # Check if file exists
            found_file = self.find_file(primary_file, fallback_file)
            file_status = "✅ FOUND" if found_file else "❌ MISSING"
            
            info_lines.append(f"→ Will launch: {analyzer_name}")
            info_lines.append(f"→ File Status: {file_status}")
            if found_file:
                info_lines.append(f"→ Location: {found_file}")
            
            info_lines.append("")
            info_lines.append("DESCRIPTION:")
            
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
        
        self.info_text.insert(1.0, "\n".join(info_lines))
        self.info_text.config(state='disabled')
        
        # Update launch button text
        if primary_choice == "numerical":
            self.launch_btn.config(text="🚀 LAUNCH GEMINI1.PY")
        else:
            method_name = self.structural_methods[self.structural_method_var.get()][0]
            light_name = self.light_sources[self.light_source_var.get()][0].split()[0]  # Get B/H, L, or U
            self.launch_btn.config(text=f"🚀 LAUNCH {method_name.upper()} {light_name}")
    
    def find_file(self, primary_path, fallback_path):
        """Find analyzer file in multiple locations"""
        paths_to_try = [
            Path(primary_path),
            self.script_dir / primary_path,
            self.script_dir.parent / primary_path,
            Path(fallback_path) if fallback_path else None,
            self.script_dir / fallback_path if fallback_path else None,
            self.script_dir.parent / fallback_path if fallback_path else None,
        ]
        
        paths_to_try = [p for p in paths_to_try if p is not None]
        
        for path in paths_to_try:
            if path.exists():
                return path
        
        return None
    
    def launch_analyzer(self):
        """Launch the appropriate analyzer based on workflow selections"""
        primary_choice = self.primary_var.get()
        
        if primary_choice == "numerical":
            # Launch gemini1.py directly
            file_config = self.file_mappings["numerical"]
            found_file = self.find_file(file_config["file"], file_config["fallback"])
            
            if not found_file:
                messagebox.showerror("File Not Found", 
                                   f"Could not find gemini1.py\nLooked for:\n• {file_config['file']}\n• {file_config['fallback']}")
                return
            
            self.run_analyzer(str(found_file), file_config["name"])
            
        else:  # structural
            structural_method = self.structural_method_var.get()
            light_source = self.light_source_var.get()
            
            file_config = self.file_mappings["structural"][structural_method][light_source]
            primary_file, fallback_file, analyzer_name = file_config
            
            found_file = self.find_file(primary_file, fallback_file)
            
            if not found_file:
                messagebox.showerror("File Not Found", 
                                   f"Could not find {analyzer_name}\nLooked for:\n• {primary_file}\n• {fallback_file}")
                return
            
            self.run_analyzer(str(found_file), analyzer_name)
    
    def run_analyzer(self, analyzer_file, analyzer_name):
        """Run the selected analyzer"""
        try:
            print(f"Launching {analyzer_name}...")
            print(f"File: {analyzer_file}")
            
            working_dir = str(Path(analyzer_file).parent)
            
            if sys.platform == 'win32':
                process = subprocess.Popen([sys.executable, analyzer_file], 
                                         cwd=working_dir,
                                         creationflags=0x00000200)
            else:
                process = subprocess.Popen([sys.executable, analyzer_file], 
                                         cwd=working_dir,
                                         preexec_fn=os.setsid if hasattr(os, 'setsid') else None)
            
            print(f"{analyzer_name} started (PID: {process.pid})")
            
            # Show success message
            messagebox.showinfo("Analyzer Launched", 
                               f"✅ {analyzer_name} started successfully!\n\nPID: {process.pid}")
            
            self.root.after(3000, self.minimize_launcher)
            
        except Exception as e:
            messagebox.showerror("Launch Error", f"Error launching {analyzer_name}:\n{str(e)}")
            print(f"Error: {e}")
    
    def minimize_launcher(self):
        """Minimize launcher after launch"""
        self.root.iconify()
        print("Launcher minimized - restore to launch another analyzer")
    
    def reset_selections(self):
        """Reset all selections to defaults"""
        self.primary_var.set("numerical")
        self.structural_method_var.set("auto")  
        self.light_source_var.set("bh")
        self.update_workflow()
    
    def show_help(self):
        """Show workflow help"""
        help_text = """GEMINI ANALYZER LOGICAL WORKFLOW HELP

STEP 1: CHOOSE ANALYSIS TYPE

🔢 NUMERICAL ANALYSIS:
• Direct gem identification using gemini1.py
• No additional selections needed
• Converts raw spectra to unknown format
• Compares against reference database
• Provides ranked gem matches with confidence scores

🔬 STRUCTURAL ANALYSIS: 
• Feature detection and marking
• Requires Steps 2 & 3 (method and light source)

STEP 2: STRUCTURAL METHOD (only for structural)

🎯 MANUAL MARKING:
• Interactive point-and-click analysis
• Full control over feature identification
• Uses gemini_*_analyzer.py files

🤖 AUTOMATED DETECTION:
• Computational feature detection
• Uses *_spectra_auto_detector.py or gemini_peak_detector.py

STEP 3: LIGHT SOURCE (only for structural)

🔥 B/H (HALOGEN): Broad features, mineral ID
⚡ L (LASER): Sharp features, precision analysis  
🟣 U (UV): Electronic transitions, color centers

FILE MAPPINGS:
• Numerical → gemini1.py
• Manual B/H → gemini_halogen_analyzer.py
• Manual L → gemini_laser_analyzer.py  
• Manual U → gemini_uv_analyzer.py
• Auto B/H → b_spectra_auto_detector.py
• Auto L → l_spectra_auto_detector.py
• Auto U → gemini_peak_detector.py

WORKFLOW EXAMPLES:

For Gem ID:
1. Select "Numerical Analysis"
2. Click "Launch" → opens gemini1.py

For Manual B Analysis:  
1. Select "Structural Analysis"
2. Select "Manual Marking"
3. Select "B/H (Halogen)"
4. Click "Launch" → opens manual halogen analyzer

For Auto L Analysis:
1. Select "Structural Analysis" 
2. Select "Automated Detection"
3. Select "L (Laser)"
4. Click "Launch" → opens L spectra auto detector"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Logical Workflow Help")
        help_window.geometry("750x600")
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
        """Start the launcher"""
        print("Starting Logical Workflow Gemini Launcher...")
        self.root.mainloop()

def main():
    """Main function"""
    try:
        launcher = GeminiLauncher()
        launcher.run()
    except Exception as e:
        print(f"Launcher error: {e}")
        messagebox.showerror("Error", f"Launcher error: {e}")

if __name__ == '__main__':
    main()
