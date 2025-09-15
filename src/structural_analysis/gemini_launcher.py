#!/usr/bin/env python3
"""
Gemini Structural Analysis Launcher - Simplified
Focused on structural analysis tools only (manual and automated)
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
    """Simplified Gemini launcher for structural analysis tools only"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemini Structural Analyzer Launcher")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        self.root.minsize(600, 700)
        
        # Get paths and setup data configuration
        self.script_dir = Path(__file__).parent.absolute()
        print(f"Launcher directory: {self.script_dir}")
        
        # Setup data paths if configuration is available
        if HAS_PATH_CONFIG:
            setup_project_paths()
            print(f"✅ Project paths configured - Raw data: {gemini_paths.raw_data}")
        
        # Structural analysis methods
        self.structural_methods = {
            "manual": ("Manual Marking", "Interactive point-and-click feature marking"),
            "auto": ("Automated Detection", "Computational feature detection algorithms")
        }
        
        # Light sources
        self.light_sources = {
            "bh": ("B/H (Halogen)", "Broad mounds, plateaus, mineral identification"),
            "l": ("L (Laser)", "Sharp features, natural/synthetic detection"),
            "u": ("U (UV)", "Electronic transitions, color centers")
        }
        
        # File mappings for structural analysis only
        self.file_mappings = {
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
        
        # Setup interface
        self.center_window()
        self.create_interface()
        
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (600 // 2)
        y = (screen_height // 2) - (700 // 2)
        self.root.geometry(f"600x700+{x}+{y}")
        print(f"Window sized for visibility: 600x700")
        
    def create_interface(self):
        """Create interface for structural analysis"""
        # Title
        title_frame = tk.Frame(self.root, bg='darkred')
        title_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(title_frame, text="GEMINI STRUCTURAL ANALYZER", 
                font=('Arial', 16, 'bold'), fg='white', bg='darkred').pack(pady=8)
        tk.Label(title_frame, text="Manual & Automated Structural Feature Detection", 
                font=('Arial', 9), fg='lightcoral', bg='darkred').pack()
        
        # Data path status if available
        if HAS_PATH_CONFIG:
            status_text = f"✅ Data paths configured - Raw data: {len(gemini_paths.get_raw_data_files())} files"
        else:
            status_text = "⚠️  Manual path resolution mode"
        tk.Label(title_frame, text=status_text, 
                font=('Arial', 8), fg='lightyellow', bg='darkred').pack()
        
        # Button frame at bottom
        self.button_frame = tk.Frame(self.root, bg='lightgray', relief='raised', bd=2)
        self.button_frame.pack(fill='x', side='bottom', pady=10)
        self.create_buttons()
        
        # Main content frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Step 1: Structural Method Selection
        self.structural_frame = tk.LabelFrame(main_frame, text="Step 1: Analysis Method", 
                                            font=('Arial', 11, 'bold'), padx=8, pady=8)
        self.structural_frame.pack(fill='x', pady=(0, 8))
        
        self.structural_method_var = tk.StringVar(value="auto")
        self.create_structural_method_options()
        
        # Step 2: Light Source Selection
        self.light_frame = tk.LabelFrame(main_frame, text="Step 2: Light Source", 
                                       font=('Arial', 11, 'bold'), padx=8, pady=8)
        self.light_frame.pack(fill='x', pady=(0, 8))
        
        self.light_source_var = tk.StringVar(value="bh")
        self.create_light_source_options()
        
        # Info section
        self.create_info_section(main_frame)
        
    def create_structural_method_options(self):
        """Create structural method options"""
        for key, (name, description) in self.structural_methods.items():
            option_frame = tk.Frame(self.structural_frame)
            option_frame.pack(fill='x', pady=3)
            
            icon = "🎯" if key == "manual" else "🤖"
            color = "darkgreen" if key == "manual" else "darkred"
            
            tk.Radiobutton(option_frame, text=f"{icon} {name}", 
                          variable=self.structural_method_var, value=key,
                          font=('Arial', 10, 'bold'), fg=color,
                          command=self.update_info_display).pack(anchor='w')
            tk.Label(option_frame, text=f"   {description}", 
                    font=('Arial', 8), fg='gray').pack(anchor='w')
    
    def create_light_source_options(self):
        """Create light source options"""
        for key, (name, description) in self.light_sources.items():
            option_frame = tk.Frame(self.light_frame)
            option_frame.pack(fill='x', pady=3)
            
            icons = {"bh": "🔥", "l": "⚡", "u": "🟣"}
            colors = {"bh": "red", "l": "blue", "u": "purple"}
            
            tk.Radiobutton(option_frame, text=f"{icons[key]} {name}", 
                          variable=self.light_source_var, value=key,
                          font=('Arial', 10, 'bold'), fg=colors[key],
                          command=self.update_info_display).pack(anchor='w')
            tk.Label(option_frame, text=f"   {description}", 
                    font=('Arial', 8), fg='gray').pack(anchor='w')
    
    def create_info_section(self, parent):
        """Create info section"""
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
        
        self.update_info_display()
    
    def create_buttons(self):
        """Create button controls"""
        # Status indicator
        self.status_label = tk.Label(self.button_frame, text="▼ STRUCTURAL ANALYSIS TOOLS ▼", 
                                    font=('Arial', 8, 'bold'), fg='darkred', bg='lightgray')
        self.status_label.pack(pady=(2,0))
        
        button_container = tk.Frame(self.button_frame, bg='lightgray')
        button_container.pack(fill='x', pady=5)
        
        # Main launch button
        self.launch_btn = tk.Button(button_container, text="🚀 LAUNCH ANALYZER", 
                                   font=('Arial', 14, 'bold'), 
                                   bg="darkred", fg="white", 
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
    
    def update_info_display(self):
        """Update info display"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        
        structural_method = self.structural_method_var.get()
        light_source = self.light_source_var.get()
        
        info_lines = []
        info_lines.append("STRUCTURAL ANALYSIS CONFIGURATION:")
        info_lines.append("=" * 40)
        
        info_lines.append("🎯 STRUCTURAL ANALYSIS")
        info_lines.append(f"✅ Method: {self.structural_methods[structural_method][0]}")
        info_lines.append(f"✅ Light Source: {self.light_sources[light_source][0]}")
        
        # Get file info
        file_config = self.file_mappings[structural_method][light_source]
        paths = file_config["paths"]
        analyzer_name = file_config["name"]
        
        found_file = self.find_file_from_paths(paths)
        file_status = "✅ FOUND" if found_file else "❌ MISSING"
        
        info_lines.append(f"→ Will launch: {analyzer_name}")
        info_lines.append(f"→ File Status: {file_status}")
        if found_file:
            info_lines.append(f"→ Location: {found_file}")
        
        info_lines.append("")
        info_lines.append("ANALYSIS FEATURES:")
        
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
        
        info_lines.append("")
        info_lines.append("📝 NOTE:")
        info_lines.append("For numerical gem identification, use main.py Option 4:")
        info_lines.append("'💎 Select Gem for Analysis (ENHANCED)'")
        
        self.info_text.insert(1.0, "\n".join(info_lines))
        self.info_text.config(state='disabled')
        
        # Update launch button text
        method_name = self.structural_methods[structural_method][0]
        light_name = self.light_sources[light_source][0].split()[0]
        self.launch_btn.config(text=f"🚀 LAUNCH {method_name.upper()} {light_name}")
    
    def find_file_from_paths(self, paths):
        """Find file from list of possible paths"""
        print(f"\n🔍 SEARCHING FOR FILE:")
        print(f"Script directory: {self.script_dir}")
        
        all_paths_to_try = []
        
        for path_str in paths:
            candidates = [
                Path(path_str),
                self.script_dir / path_str,
                self.script_dir.parent / path_str,
                Path.cwd() / path_str,
            ]
            all_paths_to_try.extend(candidates)
        
        # Remove duplicates
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
        """Launch structural analyzer"""
        structural_method = self.structural_method_var.get()
        light_source = self.light_source_var.get()
        
        file_config = self.file_mappings[structural_method][light_source]
        paths = file_config["paths"]
        analyzer_name = file_config["name"]
        
        found_file = self.find_file_from_paths(paths)
        
        if not found_file:
            self.show_file_not_found_dialog(analyzer_name, paths)
            return
        
        self.run_analyzer(str(found_file), analyzer_name)
    
    def show_file_not_found_dialog(self, analyzer_name, searched_paths):
        """Show file not found dialog"""
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
        """Run the selected structural analyzer"""
        try:
            analyzer_path = Path(analyzer_file)
            print(f"\n🚀 LAUNCHING ANALYZER:")
            print(f"Name: {analyzer_name}")
            print(f"File: {analyzer_path}")
            print(f"Exists: {analyzer_path.exists()}")
            print(f"Is file: {analyzer_path.is_file()}")
            
            if not analyzer_path.exists():
                raise FileNotFoundError(f"File does not exist: {analyzer_path}")
            
            # Use analyzer's directory as working directory (analyzers will handle their own navigation)
            working_dir = str(analyzer_path.parent)
            print(f"Working directory: {working_dir}")
            
            structural_method = self.structural_method_var.get()
            if structural_method == "manual":
                print("Manual analyzer will automatically navigate to data/raw for gem selection")
            
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
            
            success_msg = f"✅ {analyzer_name} started successfully!\n\n"
            success_msg += f"PID: {process.pid}\n"
            success_msg += f"Location: {analyzer_path}\n"
            success_msg += f"Working Dir: {working_dir}\n"
            
            if structural_method == "manual":
                success_msg += f"\n📂 Gem Selection:\n"
                success_msg += f"Started in data/raw directory for gem selection\n"
                success_msg += f"Select your gem files for analysis"
            
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
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(self.script_dir)])
            else:
                subprocess.run(['xdg-open', str(self.script_dir)])
                
            print(f"📂 Opened file browser to: {self.script_dir}")
        except Exception as e:
            print(f"Error opening file browser: {e}")
    
    def reset_selections(self):
        """Reset selections to defaults"""
        self.structural_method_var.set("auto")  
        self.light_source_var.set("bh")
        self.update_info_display()
        print("🔄 Reset to automated B/H analysis")
    
    def show_help(self):
        """Show help for structural analysis tools"""
        help_text = """GEMINI STRUCTURAL ANALYZER HELP

🎯 STRUCTURAL ANALYSIS TOOLS
Focused launcher for structural feature detection and marking

ANALYSIS METHODS:

🎯 MANUAL MARKING:
• Interactive point-and-click feature marking
• Manual control over all features  
• Zoom, undo, persistent mode tools
• Precise user-controlled analysis

🤖 AUTOMATED DETECTION:
• Computational feature detection algorithms
• Advanced algorithmic analysis
• Automated baseline, peaks, mounds detection
• High-speed batch processing

LIGHT SOURCES:

🔥 B/H (HALOGEN):
• Broad absorption features and mounds
• General mineral identification
• Wide wavelength range analysis
• Plateau and broad feature detection

⚡ L (LASER):
• High-resolution sharp features
• Natural vs synthetic detection
• Precision wavelength analysis
• Sharp peak identification

🟣 U (UV):
• Electronic transitions and color centers
• UV range spectral analysis
• Sharp peak identification
• Color center analysis

💡 RECOMMENDATIONS:

FOR FEATURE DETECTION:
→ Use Manual methods for precise control
→ Use Automated methods for batch processing
→ Choose light source based on gem type

FOR GEM IDENTIFICATION:
→ Use main.py Option 4: "💎 Select Gem for Analysis (ENHANCED)"
→ Provides complete numerical analysis with reports
→ Includes file selection and proper normalization

🚀 QUICK START:
1. Select Analysis Method (Manual or Automated)
2. Select Light Source (B/H, L, or U)
3. Click "Launch" button
4. Analyzer will open in new window

This launcher focuses on structural analysis tools only.
For numerical gem identification, use main.py Option 4."""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Structural Analysis Help")
        help_window.geometry("800x600")
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
        """Start the structural analysis launcher"""
        print("🚀 Starting Gemini Structural Analysis Launcher...")
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