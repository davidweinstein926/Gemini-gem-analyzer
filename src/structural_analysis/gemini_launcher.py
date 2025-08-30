# gemini_launcher.py - SIMPLIFIED VERSION
import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class GeminiLauncher:
    """Simplified Gemini gemological analyzer launcher - 2 analysis types, 3 light sources"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemini Gemological Analyzer Launcher")
        self.root.geometry("500x600")
        self.root.resizable(True, True)
        
        # SIMPLIFIED Configuration - 2 analysis types
        self.analysis_types = {
            "manual": ("Manual Feature Marking", "Interactive marking of baseline, peaks, mounds, plateaus, and other features"),
            "auto": ("Automated Detection", "Comprehensive automated detection of ALL structural features")
        }
        
        # 3 light sources with clear B/H naming
        self.light_sources = {
            "bh": ("B/H (Halogen)", "Broad mounds, plateaus, general mineral ID, wide spectral range"),
            "l": ("L (Laser)", "Sharp features, natural/synthetic detection, high-resolution analysis"),
            "u": ("U (UV)", "Sharp peaks, electronic transitions, color centers")
        }
        
        # Simplified analyzers mapping - 6 combinations total
        self.analyzers = {
            "manual": {
                "bh": ("structural_analysis/gemini_halogen_analyzer.py", "gemini_halogen_analyzer.py", "Manual Halogen Analyzer"),
                "l": ("structural_analysis/gemini_laser_analyzer.py", "gemini_laser_analyzer.py", "Manual Laser Analyzer"),
                "u": ("structural_analysis/gemini_uv_analyzer.py", "gemini_uv_analyzer.py", "Manual UV Analyzer")
            },
            "auto": {
                "bh": ("b_spectra_auto_detector.py", "enhanced_halogen_auto_analyzer.py", "Auto Halogen Analyzer"),
                "l": ("l_spectra_auto_detector.py", "enhanced_laser_auto_analyzer.py", "Auto Laser Analyzer"),
                "u": ("gemini_peak_detector.py", "gemini_uv_analyzer.py", "Auto UV Analyzer")
            }
        }
        
        # Setup window
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        self.root.focus_force()
        self.center_window()
        self.create_interface()
        
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        x = (screen_width // 2) - (500 // 2)
        y = 50
        self.root.geometry(f"500x600+{x}+{y}")
        print(f"Positioning launcher at: {x}, {y}")
    
    def create_interface(self):
        """Create simplified interface"""
        # Title section
        title_frame = tk.Frame(self.root, bg='darkblue')
        title_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(title_frame, text="GEMINI GEMOLOGICAL ANALYZER", 
                font=('Arial', 16, 'bold'), fg='white', bg='darkblue').pack(pady=10)
        tk.Label(title_frame, text="Manual + Automated Analysis", 
                font=('Arial', 12), fg='lightblue', bg='darkblue').pack()
        
        # Main content
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20)
        
        # Create sections
        self.create_selection_section(main_frame, "Analysis Type", self.analysis_types, "analysis_type", "manual")
        self.create_selection_section(main_frame, "Light Source", self.light_sources, "light_source", "bh")
        self.create_info_section(main_frame)
        self.create_buttons(main_frame)
        
    def create_selection_section(self, parent, title, options, var_name, default):
        """Create selection section with simplified layout"""
        frame = tk.LabelFrame(parent, text=title, font=('Arial', 12, 'bold'), padx=15, pady=15)
        frame.pack(fill='x', pady=(0, 20))
        
        setattr(self, var_name, tk.StringVar(value=default))
        var = getattr(self, var_name)
        
        # Simplified colors and icons
        colors = {
            "manual": "darkgreen", 
            "auto": "darkred",
            "bh": "red", 
            "l": "blue", 
            "u": "purple"
        }
        
        icons = {
            "manual": "🎯", 
            "auto": "🤖",
            "bh": "🔥", 
            "l": "⚡", 
            "u": "🟣"
        }
        
        for key, (name, description) in options.items():
            option_frame = tk.Frame(frame)
            option_frame.pack(fill='x', pady=8)
            
            icon = icons.get(key, "•")
            color = colors.get(key, "black")
            
            tk.Radiobutton(option_frame, text=f"{icon} {name}", variable=var, value=key,
                          font=('Arial', 12, 'bold'), fg=color, command=self.update_info).pack(anchor='w')
            tk.Label(option_frame, text=f"   {description}", font=('Arial', 10), fg='gray').pack(anchor='w')
    
    def create_info_section(self, parent):
        """Create information display"""
        self.info_frame = tk.LabelFrame(parent, text="Selected Configuration", 
                                       font=('Arial', 12, 'bold'), padx=15, pady=15)
        self.info_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        self.info_text = tk.Text(self.info_frame, height=6, width=50, font=('Arial', 10), 
                                wrap='word', bg='lightyellow', relief='sunken', bd=1)
        self.info_text.pack(fill='both', expand=True)
        self.update_info()
    
    def create_buttons(self, parent):
        """Create launch buttons"""
        button_frame = tk.Frame(parent)
        button_frame.pack(fill='x')
        
        tk.Button(button_frame, text="🚀 Launch Analyzer", font=('Arial', 14, 'bold'), 
                 bg="green", fg="white", command=self.launch_analyzer, padx=30, pady=10).pack(side='left')
        
        tk.Button(button_frame, text="❓ Help", font=('Arial', 10, 'bold'), 
                 bg="lightblue", fg="black", command=self.show_help, padx=15, pady=5).pack(side='right', padx=(0, 10))
        
        tk.Button(button_frame, text="❌ Exit", font=('Arial', 10, 'bold'), 
                 bg="lightcoral", fg="black", command=self.root.quit, padx=15, pady=5).pack(side='right')
    
    def update_info(self):
        """Update information display"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        
        analysis_type = self.analysis_type.get()
        light_source = self.light_source.get()
        
        # Get analyzer info
        analyzer_config = self.analyzers[analysis_type][light_source]
        primary_file, fallback_file, analyzer_name = analyzer_config
        
        # Build info text
        analysis_names = {"manual": "Manual Feature Marking", "auto": "Automated Detection"}
        light_names = {"bh": "B/H (Halogen)", "l": "L (Laser)", "u": "U (UV)"}
        
        info_text = f"SELECTED CONFIGURATION:\n\n"
        info_text += f"Analysis Type: {analysis_names[analysis_type]}\n"
        info_text += f"Light Source: {light_names[light_source]}\n"
        info_text += f"Analyzer: {analyzer_name}\n\n"
        
        if analysis_type == "manual":
            info_text += "Interactive GUI with point-and-click feature marking.\n"
            info_text += "Full control over baseline, peaks, mounds, plateaus."
        else:
            info_text += "Fully automated feature detection and analysis.\n"
            info_text += "Computational detection of all structural features."
        
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state='disabled')
    
    def launch_analyzer(self):
        """Launch selected analyzer"""
        analysis_choice = self.analysis_type.get()
        light_choice = self.light_source.get()
        
        analyzer_config = self.analyzers[analysis_choice][light_choice]
        primary_file, fallback_file, analyzer_name = analyzer_config
        
        # Determine which file to use
        if os.path.exists(primary_file):
            analyzer_file = primary_file
            print(f"Using primary analyzer: {analyzer_file}")
        elif fallback_file and os.path.exists(fallback_file):
            analyzer_file = fallback_file
            print(f"Using fallback analyzer: {analyzer_file}")
        else:
            messagebox.showerror("File Not Found", 
                               f"Could not find analyzer files:\n• {primary_file}" + 
                               (f"\n• {fallback_file}" if fallback_file else "") +
                               f"\n\nMake sure the analyzer files are in the correct directory.")
            return
        
        self.run_analyzer(analyzer_file, analyzer_name)
    
    def run_analyzer(self, analyzer_file, analyzer_name):
        """Run the selected analyzer"""
        try:
            print(f"Launching {analyzer_name}...")
            
            # Launch process
            if sys.platform == 'win32':
                process = subprocess.Popen([sys.executable, analyzer_file], cwd=os.getcwd(),
                                         creationflags=0x00000200, stdout=None, stderr=None, stdin=None)
            else:
                process = subprocess.Popen([sys.executable, analyzer_file], cwd=os.getcwd(),
                                         stdout=None, stderr=None, stdin=None,
                                         preexec_fn=os.setsid if hasattr(os, 'setsid') else None)
            
            print(f"{analyzer_name} started (PID: {process.pid})")
            self.root.after(2000, self.minimize_launcher)
            
        except Exception as e:
            messagebox.showerror("Launch Error", f"Error launching {analyzer_name}:\n{e}")
            print(f"Error: {e}")
    
    def minimize_launcher(self):
        """Minimize launcher after launch"""
        self.root.iconify()
        print("Launcher minimized - restore to launch another analyzer")
    
    def show_help(self):
        """Show simplified help"""
        help_content = """SIMPLIFIED GEMINI ANALYZER HELP

ANALYSIS TYPES:

MANUAL FEATURE MARKING:
• Interactive point-and-click analysis
• Mark baseline, peaks, mounds, plateaus, troughs, shoulders, valleys
• Full control over feature identification  
• Best for detailed gemological analysis
• Zoom, undo, persistent mode tools available

AUTOMATED DETECTION:
• Comprehensive computational feature detection
• Detects ALL the same features as manual analysis
• Advanced algorithms with configurable sensitivity
• Best for objective, repeatable analysis
• Same CSV output format as manual analysis

LIGHT SOURCES:

B/H (HALOGEN):
• Broad absorption features & mounds
• General mineral identification
• Wide wavelength range: 400-2500nm
• Best for: Comprehensive gemological analysis

L (LASER):
• High-resolution sharp features
• Natural vs synthetic discrimination  
• Pre-mound structure detection
• Best for: Detailed feature analysis

U (UV):
• Sharp peaks & electronic transitions
• Color center studies: 200-800nm
• Best for: Electronic property analysis

WORKFLOW:
1. Select analysis type (Manual or Auto)
2. Select light source (B/H, L, or U)
3. Click "Launch Analyzer" 
4. Choose your spectrum file
5. Results exported to CSV format

OUTPUT:
• All analyzers export to CSV format
• Compatible with database import
• Same data structure for manual and automated results"""
        
        # Simple help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Gemini Analyzer Help")
        help_window.geometry("600x500")
        help_window.resizable(True, True)
        
        # Center help window
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - 300
        y = (help_window.winfo_screenheight() // 2) - 250
        help_window.geometry(f"600x500+{x}+{y}")
        
        # Create scrollable text
        text_frame = tk.Frame(help_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        help_text_widget = tk.Text(text_frame, font=('Arial', 9), wrap='word', yscrollcommand=scrollbar.set)
        help_text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=help_text_widget.yview)
        
        help_text_widget.insert(1.0, help_content)
        help_text_widget.config(state='disabled')
        
        tk.Button(help_window, text="Close", command=help_window.destroy, pady=5).pack(pady=10)
    
    def run(self):
        """Start the launcher"""
        print("Starting Simplified Gemini Analyzer Launcher...")
        self.root.mainloop()

def main():
    """Main launcher function"""
    try:
        launcher = GeminiLauncher()
        launcher.run()
    except Exception as e:
        print(f"Launcher error: {e}")
        messagebox.showerror("Error", f"Launcher error: {e}")

if __name__ == '__main__':
    main()
