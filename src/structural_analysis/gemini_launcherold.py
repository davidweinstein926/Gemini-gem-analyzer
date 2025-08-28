# gemini_launcher.py - UPDATED with Peak Detectors
import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import subprocess
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class GeminiLauncher:
    """Main launcher for standardized Gemini gemological analyzers with peak detectors"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Gemini Gemological Analyzer Launcher")
        self.root.geometry("550x700")
        self.root.resizable(True, True)
        
        # Configuration data
        self.analysis_types = {
            "manual": ("Manual Feature Marking", "Interactive marking of baseline, peaks, mounds, plateaus, and other features"),
            "automated": ("Automated Structural Detection", "Comprehensive automated detection of ALL structural features"),
            "peak_detection": ("Peak Detection Algorithms", "Advanced algorithmic detection using trained peak detectors")
        }
        
        self.light_sources = {
            "halogen": ("Halogen Light", "Best for: Broad mounds, plateaus, general mineral ID, wide spectral range"),
            "laser": ("Laser Light", "Best for: Sharp features, natural/synthetic detection, high-resolution analysis"),
            "uv": ("UV Light", "Best for: Sharp peaks, electronic transitions, color centers")
        }
        
        # Updated analyzers configuration with peak detectors
        self.analyzers = {
            "manual": {
                "halogen": ("structural_analysis/gemini_halogen_analyzer.py", "gemini_halogen_analyzer.py", "Manual Halogen Analyzer"),
                "laser": ("structural_analysis/gemini_laser_analyzer.py", "gemini_laser_analyzer.py", "Manual Laser Analyzer"),
                "uv": ("structural_analysis/gemini_uv_analyzer.py", "gemini_uv_analyzer.py", "Manual UV Analyzer")
            },
            "automated": {
                "halogen": ("enhanced_halogen_auto_analyzer.py", None, "Enhanced Halogen Auto Analyzer"),
                "laser": ("enhanced_laser_auto_analyzer.py", None, "Enhanced Laser Auto Analyzer"),
                "uv": ("gemini_peak_detector.py", None, "UV Peak Detector")
            },
            "peak_detection": {
                "halogen": ("gemini_b_detector_gui.py", "b_spectra_auto_detector.py", "Gemini B-Spectra Peak Detector"),
                "laser": ("gemini_l_detector_gui.py", "l_spectra_auto_detector.py", "Gemini L-Spectra Peak Detector"),
                "uv": ("gemini_peak_detector.py", None, "UV Peak Detector")
            }
        }
        
        # Force window prominence and center
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        self.root.focus_force()
        self.center_window()
        self.create_interface()
        
    def center_window(self):
        """Center window high on screen for visibility"""
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        x = (screen_width // 2) - (550 // 2)
        y = 50  # High position
        self.root.geometry(f"550x700+{x}+{y}")
        print(f"Positioning launcher at: {x}, {y} (centered and high)")
    
    def create_interface(self):
        """Create main interface"""
        # Title section
        title_frame = tk.Frame(self.root, bg='darkblue')
        title_frame.pack(fill='x', pady=(0, 20))
        
        tk.Label(title_frame, text="GEMINI GEMOLOGICAL ANALYZER", 
                font=('Arial', 16, 'bold'), fg='white', bg='darkblue').pack(pady=10)
        tk.Label(title_frame, text="Manual + Automated + Peak Detection Analysis", 
                font=('Arial', 12), fg='lightblue', bg='darkblue').pack()
        
        # Main content
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20)
        
        # Create sections
        self.create_selection_section(main_frame, "Select Analysis Type", self.analysis_types, "analysis_type", "manual")
        self.create_selection_section(main_frame, "Select Light Source", self.light_sources, "light_source", "halogen")
        self.create_info_section(main_frame)
        self.create_buttons(main_frame)
        
    def create_selection_section(self, parent, title, options, var_name, default):
        """Create standardized selection section"""
        frame = tk.LabelFrame(parent, text=title, font=('Arial', 11, 'bold'), padx=10, pady=10)
        frame.pack(fill='x', pady=(0, 15))
        
        setattr(self, var_name, tk.StringVar(value=default))
        var = getattr(self, var_name)
        
        colors = {
            "manual": "darkgreen", 
            "automated": "darkred", 
            "peak_detection": "purple",
            "halogen": "darkred", 
            "laser": "blue", 
            "uv": "purple"
        }
        
        for key, (name, description) in options.items():
            option_frame = tk.Frame(frame)
            option_frame.pack(fill='x', pady=5)
            
            icons = {
                "manual": "🎯", 
                "automated": "🤖", 
                "peak_detection": "🔬",
                "halogen": "🔥", 
                "laser": "⚡", 
                "uv": "🟣"
            }
            icon = icons.get(key, "•")
            color = colors.get(key, "black")
            
            tk.Radiobutton(option_frame, text=f"{icon} {name}", variable=var, value=key,
                          font=('Arial', 11, 'bold'), fg=color, command=self.update_info).pack(anchor='w')
            tk.Label(option_frame, text=f"   {description}", font=('Arial', 9), fg='gray').pack(anchor='w')
    
    def create_info_section(self, parent):
        """Create information display section"""
        self.info_frame = tk.LabelFrame(parent, text="Analysis Information", 
                                       font=('Arial', 11, 'bold'), padx=10, pady=10)
        self.info_frame.pack(fill='both', expand=True, pady=(0, 15))
        
        self.info_text = tk.Text(self.info_frame, height=8, width=60, font=('Arial', 9), 
                                wrap='word', bg='lightyellow', relief='sunken', bd=1)
        self.info_text.pack(fill='both', expand=True)
        self.update_info()
    
    def create_buttons(self, parent):
        """Create button section"""
        button_frame = tk.Frame(parent)
        button_frame.pack(fill='x')
        
        buttons = [
            ("🚀 Launch Analyzer", self.launch_analyzer, "green", "white", "left", 20),
            ("❓ Help", self.show_help, "lightblue", "black", "right", 15),
            ("❌ Exit", self.root.quit, "lightcoral", "black", "right", 15)
        ]
        
        for text, command, bg, fg, side, padx in buttons:
            btn = tk.Button(button_frame, text=text, font=('Arial', 12 if 'Launch' in text else 10, 'bold'), 
                           bg=bg, fg=fg, command=command, padx=padx, pady=5)
            btn.pack(side=side, padx=(0, 10) if side == 'right' else 0)
    
    def update_info(self):
        """Update information display"""
        self.info_text.config(state='normal')
        self.info_text.delete(1.0, tk.END)
        
        analysis_type = self.analysis_type.get()
        light_source = self.light_source.get()
        
        # Base information for analysis types
        info_data = {
            "manual": {
                "title": "MANUAL FEATURE MARKING:",
                "features": [
                    "Interactive analysis with standardized gemological features",
                    "BASELINE - Reference region for noise analysis & correction",
                    "MOUND - Broad absorption feature (diagnostic for many gems)",
                    "PLATEAU - Flat-topped spectral feature",
                    "PEAK - Sharp absorption maximum",
                    "TROUGH - Broad absorption valley",
                    "SHOULDER - Subsidiary spectral feature",
                    "VALLEY - Narrow spectral depression",
                    "Manual tools: zoom, undo, persistent mode, baseline correction"
                ]
            },
            "automated": {
                "title": "AUTOMATED STRUCTURAL DETECTION:",
                "features": [
                    "Comprehensive computational detection of ALL structural features",
                    "Baseline detection with SNR analysis",
                    "Peak detection with prominence filtering",
                    "Mound detection with width & symmetry analysis",
                    "Plateau detection with flatness metrics",
                    "Trough, shoulder, and valley detection",
                    "Automatic baseline correction & normalization",
                    "Configurable sensitivity and export to CSV"
                ]
            },
            "peak_detection": {
                "title": "ADVANCED PEAK DETECTION ALGORITHMS:",
                "features": [
                    "Trained algorithmic detection using machine learning principles",
                    "Adaptive detection strategy based on noise assessment",
                    "B-Spectra: Optimized for broadband/halogen transmission spectra",
                    "L-Spectra: Optimized for laser-induced sharp features",
                    "Hybrid laser + region-based detection methods",
                    "Automatic baseline assessment and classification",
                    "High-precision wavelength identification",
                    "CSV output compatible with manual marking format"
                ]
            }
        }
        
        # Light source specialties
        light_info = {
            "halogen": "Broad-band absorption, mounds/plateaus, mineral ID, 400-2500nm range",
            "laser": "High-resolution, sharp features, natural/synthetic detection, pre-mound structures", 
            "uv": "Electronic transitions, color centers, sharp peaks, 200-800nm range"
        }
        
        # Special info for peak detection mode
        peak_detection_specialties = {
            "halogen": "B-Spectra detector: Optimized for complex halogen transmission spectra with multiple overlapping features",
            "laser": "L-Spectra detector: Optimized for high-resolution laser spectra with sharp, well-defined peaks",
            "uv": "UV Peak detector: Traditional peak finding for electronic transitions and color centers"
        }
        
        # Build info text
        data = info_data[analysis_type]
        info_text = f"{data['title']}\n\n"
        info_text += "\n".join(f"• {feature}" for feature in data['features'])
        
        if analysis_type == "peak_detection":
            info_text += f"\n\n{light_source.upper()} DETECTOR SPECIALTIES:\n• {peak_detection_specialties[light_source]}"
        else:
            info_text += f"\n\n{light_source.upper()} LIGHT SPECIALTIES:\n• {light_info[light_source]}"
        
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
            # For peak detection, provide guidance on missing files
            if analysis_choice == "peak_detection":
                missing_files = [primary_file]
                if fallback_file:
                    missing_files.append(fallback_file)
                
                messagebox.showerror("Peak Detector Not Found", 
                                   f"Could not find {analyzer_name}:\n" +
                                   "\n".join(f"• {file}" for file in missing_files) +
                                   f"\n\nFor peak detection mode, you need:\n" +
                                   f"• The detector algorithm file (gemini_Bpeak_detector.py or gemini_Lpeak_detector.py)\n" +
                                   f"• A GUI wrapper or the command-line script")
            else:
                messagebox.showerror("File Not Found", 
                                   f"Could not find {primary_file}" + 
                                   (f" or {fallback_file}" if fallback_file else "") +
                                   f"\n\nMake sure the analyzer files are in the correct directory.")
            return
        
        self.run_analyzer(analyzer_file, analyzer_name)
    
    def run_analyzer(self, analyzer_file, analyzer_name):
        """Run analyzer with proper process handling"""
        try:
            print(f"Launching {analyzer_name}...")
            
            # Cross-platform process creation
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
            print(f"Detailed error: {e}")
    
    def minimize_launcher(self):
        """Minimize launcher after successful launch"""
        self.root.iconify()
        print("Launcher minimized - restore it to launch another analyzer")
    
    def show_help(self):
        """Show help information"""
        help_content = """GEMINI GEMOLOGICAL ANALYZER HELP

ANALYSIS TYPES:

MANUAL FEATURE MARKING:
• Interactive point-and-click analysis
• Mark baseline, peaks, mounds, plateaus, troughs, shoulders, valleys
• Full control over feature identification
• Best for detailed gemological analysis
• Automatic baseline correction & normalization

AUTOMATED STRUCTURAL DETECTION:
• Comprehensive computational feature identification
• Detects ALL the same features as manual analysis
• Advanced algorithms with adjustable sensitivity
• Best for objective, repeatable analysis
• Same output format as manual analysis

ADVANCED PEAK DETECTION ALGORITHMS:
• Trained algorithmic detection using machine learning principles
• B-Spectra Detector: For halogen/broadband transmission spectra
• L-Spectra Detector: For laser-induced high-resolution spectra
• Adaptive detection strategy based on baseline noise assessment
• Hybrid laser + region-based detection methods
• High-precision critical wavelength identification
• CSV output compatible with manual marking format

LIGHT SOURCES:

HALOGEN LIGHT:
• Broad absorption features & mounds
• General gemological analysis, mineral identification
• Wide wavelength range: 400-2500nm
• Peak Detection Mode: B-Spectra detector for complex overlapping features

LASER LIGHT:
• High-resolution sharp features
• Natural vs synthetic discrimination
• Pre-mound structure detection
• Peak Detection Mode: L-Spectra detector for well-defined peaks

UV LIGHT:
• Sharp peaks & electronic transitions
• Color center studies, wavelength range: 200-800nm
• Peak detection with database comparison

WORKFLOW RECOMMENDATIONS:
• Unknown stones: Use peak detection first for objective analysis
• Complex spectra: Peak detection for challenging samples like 224BC3
• Reference standards: Peak detection for cataloging, manual for verification
• Routine analysis: Peak detection for speed and consistency

PEAK DETECTION ADVANTAGES:
• Objective, repeatable analysis
• Handles complex multi-component spectra
• Adaptive algorithms adjust to sample quality
• Same CSV output format as manual marking
• Ideal for samples difficult to mark manually

KEYBOARD SHORTCUTS (Manual Mode):
B=Baseline, 1-7=Features, U=Undo, S=Save, P=Persistent mode

OUTPUT:
• All analyzers export to CSV format
• Compatible with database import system
• Same data structure for manual, automated & peak detection results"""
        
        # Create help window
        help_window = tk.Toplevel(self.root)
        help_window.title("Enhanced Gemini Analyzer Help")
        help_window.geometry("700x650")
        help_window.resizable(True, True)
        
        # Center help window
        help_window.update_idletasks()
        x = (help_window.winfo_screenwidth() // 2) - 350
        y = (help_window.winfo_screenheight() // 2) - 325
        help_window.geometry(f"700x650+{x}+{y}")
        
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
        print("Starting Enhanced Gemini Gemological Analyzer Launcher with Peak Detection...")
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
