#!/usr/bin/env python3
"""
ENHANCED HALOGEN ANALYZER (B LIGHT) - GEMINI SPECTRAL ANALYSIS
Enhanced version of: gemini_halogen_analyzer.py
Adds: Structural feature naming consistency, plateau vs shoulder disambiguation, CSV integration
Author: Enhanced with naming consistency system for B/H light analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sys
import os

# Import the naming system
try:
    from feature_naming_system import StructuralFeatureNamingSystem
    HAS_NAMING_SYSTEM = True
    print("✅ Naming consistency system loaded")
except ImportError:
    print("⚠️ Feature naming system not available - using basic mode")
    HAS_NAMING_SYSTEM = False

class EnhancedHalogenAnalyzer:
    """Enhanced B/H Light Manual Analyzer with naming consistency"""
    
    def __init__(self):
        self.light_source = 'B'
        self.current_gem_id = None
        self.wavelengths = None
        self.intensities = None
        self.manual_features = []
        
        # Initialize naming system if available
        if HAS_NAMING_SYSTEM:
            self.naming_system = StructuralFeatureNamingSystem()
            print("✅ Enhanced B/H analyzer with naming consistency")
        else:
            self.naming_system = None
            print("✅ Basic B/H analyzer (no naming system)")
        
        # B/H light specific feature types (broad features, mounds, plateaus)
        self.feature_types = {
            'plateau': 'Wide, flat absorption region (key B light feature)',
            'broad_peak': 'Wide absorption peak or mound', 
            'shoulder': 'Asymmetric extension from main peak',
            'valley': 'Low transmission between features',
            'absorption_edge': 'Sharp increase in absorption',
            'doublet': 'Two closely spaced broad peaks',
            'baseline_shift': 'Overall baseline change',
            'broad_absorption': 'General wide absorption feature'
        }
        
        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Setup B/H light specific GUI"""
        self.root = tk.Tk()
        self.root.title("Enhanced B/H Light Manual Analyzer - Gemini Spectral Analysis")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control panel
        self.create_control_panel(main_frame)
        
        # Plotting area
        self.create_plot_area(main_frame)
        
        # Feature list
        self.create_feature_list(main_frame)
        
        # Status bar
        self.create_status_bar()
    
    def create_control_panel(self, parent):
        """Create B/H light specific control panel"""
        control_frame = tk.LabelFrame(parent, text="B/H Light Controls (Broad Features)", 
                                    font=('Arial', 12, 'bold'), fg='darkred')
        control_frame.pack(fill='x', pady=(0, 10))
        
        # File operations
        file_frame = tk.Frame(control_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Button(file_frame, text="Load B Spectrum", command=self.load_spectrum,
                 bg='lightcoral', font=('Arial', 10)).pack(side='left', padx=5)
        
        tk.Button(file_frame, text="Save B Features", command=self.save_features,
                 bg='lightgreen', font=('Arial', 10)).pack(side='left', padx=5)
        
        tk.Button(file_frame, text="Load Previous", command=self.load_previous_features,
                 bg='lightyellow', font=('Arial', 10)).pack(side='left', padx=5)
        
        # Gem ID entry with enhanced description
        id_frame = tk.Frame(control_frame)
        id_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(id_frame, text="Gem ID:", font=('Arial', 10, 'bold')).pack(side='left')
        self.gem_id_var = tk.StringVar()
        self.gem_id_entry = tk.Entry(id_frame, textvariable=self.gem_id_var, width=15,
                                   font=('Arial', 10))
        self.gem_id_entry.pack(side='left', padx=5)
        self.gem_id_entry.bind('<Return>', self.update_gem_info)
        
        # Enhanced gem description (using CSV data)
        self.gem_desc_var = tk.StringVar(value="Enter Gem ID for enhanced description")
        tk.Label(id_frame, textvariable=self.gem_desc_var, 
                font=('Arial', 9), fg='darkblue', wraplength=400).pack(side='left', padx=10)
        
        # B/H light specific feature selection
        feature_frame = tk.Frame(control_frame)
        feature_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(feature_frame, text="B Light Feature Type:", 
                font=('Arial', 10, 'bold')).pack(side='left')
        self.feature_type_var = tk.StringVar()
        
        # B/H specific feature dropdown with descriptions
        feature_values = [f"{ftype} - {desc}" for ftype, desc in self.feature_types.items()]
        self.feature_combo = ttk.Combobox(feature_frame, textvariable=self.feature_type_var, 
                                        values=feature_values, width=50)
        self.feature_combo.pack(side='left', padx=5)
        
        # Plateau vs Shoulder guidance for B light
        guidance_frame = tk.Frame(control_frame)
        guidance_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Label(guidance_frame, text="💡 B Light Guidance:", 
                font=('Arial', 9, 'bold'), fg='darkred').pack(side='left')
        tk.Label(guidance_frame, text="Plateau: >50nm wide, flat • Shoulder: <30nm, asymmetric, attached", 
                font=('Arial', 8), fg='gray').pack(side='left', padx=5)
        
        # Naming system status
        naming_frame = tk.Frame(control_frame)
        naming_frame.pack(fill='x', padx=5, pady=2)
        
        status_color = 'green' if HAS_NAMING_SYSTEM else 'orange'
        status_text = "✅ Naming Consistency: ON" if HAS_NAMING_SYSTEM else "⚠️ Naming Consistency: OFF"
        tk.Label(naming_frame, text=status_text, fg=status_color, 
                font=('Arial', 9, 'bold')).pack(side='left')
        
        if HAS_NAMING_SYSTEM:
            tk.Button(naming_frame, text="Show B Light Rules", command=self.show_naming_rules,
                     bg='lightcyan', font=('Arial', 8)).pack(side='right', padx=5)
    
    def create_plot_area(self, parent):
        """Create B/H light specific plotting area"""
        plot_frame = tk.LabelFrame(parent, text="B/H Light Spectrum - Enhanced Feature Marking (linewidth=0.5)", 
                                 font=('Arial', 12, 'bold'), fg='darkred')
        plot_frame.pack(fill='both', expand=True)
        
        # Create matplotlib figure with your preferred settings
        self.fig = Figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(111)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bind click event for feature marking
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Initialize B/H light plot
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Transmission (%)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('B/H Light Spectral Analysis - Broad Features & Plateaus')
        
        # Add instructions
        self.ax.text(0.02, 0.98, 'Click on spectrum to mark features\nB Light: Best for plateaus, shoulders, broad peaks', 
                    transform=self.ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def create_feature_list(self, parent):
        """Create enhanced feature list for B/H light"""
        list_frame = tk.LabelFrame(parent, text="B Light Features - Enhanced with Naming Consistency", 
                                 font=('Arial', 12, 'bold'), fg='darkred')
        list_frame.pack(fill='x', pady=(10, 0))
        
        # Create treeview with B/H specific columns
        columns = ('Wavelength', 'B Feature Type', 'Standardized', 'Width', 'Confidence', 'Notes')
        self.feature_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=6)
        
        # Configure columns for B light analysis
        self.feature_tree.heading('Wavelength', text='Wavelength (nm)')
        self.feature_tree.heading('B Feature Type', text='Original B Type')
        self.feature_tree.heading('Standardized', text='Standardized Type') 
        self.feature_tree.heading('Width', text='Width (nm)')
        self.feature_tree.heading('Confidence', text='Confidence')
        self.feature_tree.heading('Notes', text='B Light Notes')
        
        self.feature_tree.column('Wavelength', width=100)
        self.feature_tree.column('B Feature Type', width=150)
        self.feature_tree.column('Standardized', width=150)
        self.feature_tree.column('Width', width=80)
        self.feature_tree.column('Confidence', width=80)
        self.feature_tree.column('Notes', width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.feature_tree.yview)
        self.feature_tree.configure(yscrollcommand=scrollbar.set)
        
        self.feature_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Bind double-click for editing
        self.feature_tree.bind('<Double-1>', self.edit_feature)
    
    def create_status_bar(self):
        """Create B/H light specific status bar"""
        self.status_var = tk.StringVar()
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x')
        self.update_status("B/H Light Analyzer Ready - Enhanced with naming consistency")
    
    def load_spectrum(self):
        """Load B/H light spectrum data"""
        filename = filedialog.askopenfilename(
            title="Load B/H Light Spectrum File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="../../data/raw"  # Point to your data directory
        )
        
        if filename:
            try:
                # Try different formats
                if filename.endswith('.csv'):
                    data = pd.read_csv(filename)
                    if len(data.columns) >= 2:
                        self.wavelengths = data.iloc[:, 0].values
                        self.intensities = data.iloc[:, 1].values
                    else:
                        raise ValueError("CSV must have at least 2 columns")
                else:
                    # Assume space-separated text file
                    data = np.loadtxt(filename)
                    self.wavelengths = data[:, 0]
                    self.intensities = data[:, 1]
                
                # Extract gem ID from filename (B light specific)
                gem_id = Path(filename).stem
                # Remove B light suffixes
                for suffix in ['B', 'b', 'BH', 'bh', 'H', 'h']:
                    if gem_id.endswith(suffix):
                        gem_id = gem_id[:-len(suffix)]
                        break
                
                self.gem_id_var.set(gem_id)
                self.current_gem_id = gem_id
                self.update_gem_info()
                
                # Plot the B/H spectrum
                self.plot_spectrum()
                self.update_status(f"Loaded B spectrum: {Path(filename).name}")
                
            except Exception as e:
                messagebox.showerror("B Spectrum Load Error", f"Error loading B light file: {e}")
                self.update_status(f"B load failed: {e}")
    
    def update_gem_info(self, event=None):
        """Update gem information using enhanced naming system"""
        gem_id = self.gem_id_var.get().strip()
        if not gem_id:
            return
        
        self.current_gem_id = gem_id
        
        if HAS_NAMING_SYSTEM and self.naming_system:
            gem_info = self.naming_system.get_gem_description(gem_id)
            desc = gem_info.get('full_description', f'Gem {gem_id}')
            
            # Add B light specific info
            species = gem_info.get('species', '')
            variety = gem_info.get('variety', '')
            if species:
                desc = f"B Light Analysis: {species} {variety} - {desc}".strip(' -')
        else:
            desc = f"B Light Analysis: Gem {gem_id}"
        
        self.gem_desc_var.set(desc)
        self.update_status(f"B Light - Gem: {gem_id} - {desc}")
    
    def plot_spectrum(self):
        """Plot B/H light spectrum with enhanced feature visualization"""
        if self.wavelengths is None or self.intensities is None:
            return
        
        self.ax.clear()
        
        # Plot B/H spectrum with your preferred linewidth=0.5
        self.ax.plot(self.wavelengths, self.intensities, 'r-', linewidth=0.5, 
                    label='B/H Light Spectrum', color='darkred')
        
        # B/H light specific feature colors
        b_light_colors = {
            'plateau': 'red',
            'shoulder': 'orange', 
            'broad_peak': 'blue',
            'valley': 'purple',
            'absorption_edge': 'brown',
            'doublet': 'pink',
            'baseline_shift': 'gray',
            'broad_absorption': 'darkred'
        }
        
        # Plot B light features
        for feature in self.manual_features:
            wavelength = feature['wavelength']
            orig_type = feature['original_type']
            std_type = feature.get('standardized_type', orig_type)
            width = feature.get('context', {}).get('width', 0)
            
            color = b_light_colors.get(std_type, 'black')
            
            # Vertical line at feature position
            self.ax.axvline(wavelength, color=color, linewidth=0.5, alpha=0.8)
            
            # Find intensity at this wavelength
            idx = np.argmin(np.abs(self.wavelengths - wavelength))
            intensity = self.intensities[idx]
            
            # B light specific annotation
            if orig_type != std_type:
                label = f"{std_type}\n({orig_type})\nW:{width:.0f}nm"
            else:
                label = f"{std_type}\nW:{width:.0f}nm"
                
            self.ax.annotate(label, (wavelength, intensity), 
                           rotation=90, fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Transmission (%)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'B/H Light Analysis: {self.current_gem_id or "Unknown Gem"} - Broad Features')
        
        # Add consolidated legend for B light features (your approach)
        legend_elements = []
        used_types = set(f.get('standardized_type', f['original_type']) for f in self.manual_features)
        for ftype in used_types:
            color = b_light_colors.get(ftype, 'black')
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=0.5, label=ftype.title()))
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, title='B Light Features', loc='upper right')
        
        self.canvas.draw()
    
    def on_plot_click(self, event):
        """Handle B/H light plot click for feature marking"""
        if event.inaxes != self.ax or not event.xdata:
            return
        
        wavelength = event.xdata
        
        # Get B light feature type from combobox
        selected = self.feature_type_var.get()
        if not selected:
            messagebox.showwarning("No B Feature Selected", "Please select a B light feature type first")
            return
        
        # Extract just the feature type name
        feature_type = selected.split(' - ')[0]
        
        # Calculate B light specific context
        context = self.calculate_b_light_context(wavelength)
        
        if HAS_NAMING_SYSTEM and self.naming_system:
            # Get standardized name with B light context
            standardized_type = self.naming_system.standardize_feature_name(feature_type, context)
            confidence = context.get('confidence', 0.8)
            
            # B light specific disambiguation for plateau vs shoulder
            if feature_type.lower() in ['plateau', 'shoulder', 'broad_absorption'] and confidence < 0.9:
                choice = self.show_b_light_disambiguation_dialog(feature_type, standardized_type, context)
                if choice:
                    standardized_type = choice
            
            # Add to database
            success = self.naming_system.add_structural_feature(
                gem_id=self.current_gem_id or 'unknown',
                light_source='B',
                wavelength=wavelength,
                feature_type=feature_type,
                context=context,
                analysis_method='manual_b_light'
            )
            
            if not success:
                messagebox.showerror("Database Error", "Failed to save B light feature to database")
                return
        else:
            standardized_type = feature_type.lower()
            confidence = 0.5
        
        # Add to local B light features list
        feature = {
            'wavelength': wavelength,
            'original_type': feature_type,
            'standardized_type': standardized_type,
            'confidence': confidence,
            'context': context
        }
        
        self.manual_features.append(feature)
        self.update_feature_list()
        self.plot_spectrum()
        
        # Show B light specific standardization result
        width = context.get('width', 0)
        if feature_type.lower() != standardized_type:
            self.update_status(f"B Light: {feature_type} → {standardized_type} at {wavelength:.1f}nm (W:{width:.0f}nm)")
        else:
            self.update_status(f"B Light: {standardized_type} at {wavelength:.1f}nm (W:{width:.0f}nm)")
    
    def calculate_b_light_context(self, wavelength):
        """Calculate B/H light specific context for feature disambiguation"""
        if self.wavelengths is None or self.intensities is None:
            return {'confidence': 0.5, 'width': 0}
        
        # Find nearest data points
        center_idx = np.argmin(np.abs(self.wavelengths - wavelength))
        
        # B light specific window (broader for B light features)
        window_size = 40  # Larger window for B light broad features
        mask = np.abs(self.wavelengths - wavelength) <= window_size
        
        if not np.any(mask):
            return {'confidence': 0.5, 'width': 0}
        
        window_wavelengths = self.wavelengths[mask]
        window_intensities = self.intensities[mask]
        
        # B light specific metrics
        width = len(window_wavelengths) * np.mean(np.diff(self.wavelengths)) if len(window_wavelengths) > 1 else 0
        intensity_std = np.std(window_intensities)
        
        # Calculate asymmetry (important for plateau vs shoulder in B light)
        if len(window_intensities) >= 4:
            mid_point = len(window_intensities) // 2
            left_half = window_intensities[:mid_point]
            right_half = window_intensities[mid_point:]
            asymmetry = abs(np.mean(left_half) - np.mean(right_half)) / np.mean(window_intensities)
        else:
            asymmetry = 0
        
        # B light specific slope analysis for plateau detection
        if len(window_intensities) >= 3:
            slopes = np.gradient(window_intensities, window_wavelengths)
            slope_variation = np.std(slopes)
            mean_abs_slope = np.mean(np.abs(slopes))
        else:
            slope_variation = 0
            mean_abs_slope = 0
        
        # Check if attached to main peak (important for shoulders in B light)
        attached_to_main_peak = self.check_b_light_peak_attachment(wavelength)
        
        context = {
            'width': width,
            'asymmetry': asymmetry,
            'intensity_std': intensity_std,
            'slope_variation': slope_variation,
            'mean_abs_slope': mean_abs_slope,
            'attached_to_main_peak': attached_to_main_peak,
            'light_source': 'B',
            'confidence': 0.8
        }
        
        # B light specific confidence adjustment
        if width > 50 and asymmetry < 0.2:  # Likely plateau
            context['confidence'] = 0.9
        elif attached_to_main_peak and asymmetry > 0.3:  # Likely shoulder
            context['confidence'] = 0.9
        
        return context
    
    def check_b_light_peak_attachment(self, wavelength):
        """Check if B light feature is attached to a main peak"""
        if self.wavelengths is None or self.intensities is None:
            return False
        
        # Find local minima (absorption peaks in transmission spectra)
        try:
            from scipy import signal
            # Invert for peak finding (lower transmission = peak)
            inverted = -self.intensities
            peaks, _ = signal.find_peaks(inverted, height=0, distance=15)  # Broader distance for B light
            peak_wavelengths = self.wavelengths[peaks]
            
            # Check if our feature is within 30nm of any B light peak (broader than L light)
            return np.any(np.abs(peak_wavelengths - wavelength) <= 30)
        except:
            return False
    
    def show_b_light_disambiguation_dialog(self, original_type, suggested_type, context):
        """Show B light specific disambiguation dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("B Light Feature Disambiguation")
        dialog.geometry("450x350")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (225)
        y = (dialog.winfo_screenheight() // 2) - (175)
        dialog.geometry(f"450x350+{x}+{y}")
        
        # B light specific content
        tk.Label(dialog, text="B/H Light Feature Disambiguation", 
                font=('Arial', 14, 'bold'), fg='darkred').pack(pady=10)
        
        tk.Label(dialog, text=f"Original: {original_type}", 
                font=('Arial', 12)).pack(pady=5)
        tk.Label(dialog, text=f"AI Suggested: {suggested_type}", 
                font=('Arial', 12), fg='blue').pack(pady=5)
        
        # B light specific context
        context_frame = tk.LabelFrame(dialog, text="B Light Analysis Context", fg='darkred')
        context_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(context_frame, text=f"Width: {context.get('width', 0):.1f} nm").pack(anchor='w')
        tk.Label(context_frame, text=f"Asymmetry: {context.get('asymmetry', 0):.3f}").pack(anchor='w')
        tk.Label(context_frame, text=f"Slope variation: {context.get('slope_variation', 0):.3f}").pack(anchor='w')
        tk.Label(context_frame, text=f"Attached to peak: {context.get('attached_to_main_peak', False)}").pack(anchor='w')
        
        # B light specific guidance
        guidance_frame = tk.LabelFrame(dialog, text="B/H Light Guidelines", fg='darkred')
        guidance_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(guidance_frame, text="• Plateau (B light): Wide (>50nm), flat, low slope variation", 
                font=('Arial', 9), fg='darkred').pack(anchor='w')
        tk.Label(guidance_frame, text="• Shoulder (B light): Asymmetric, attached to peak, <30nm", 
                font=('Arial', 9), fg='darkorange').pack(anchor='w')
        tk.Label(guidance_frame, text="• Broad peak: Wide but with clear maximum", 
                font=('Arial', 9), fg='darkblue').pack(anchor='w')
        
        # Choice buttons
        button_frame = tk.Frame(dialog)
        button_frame.pack(fill='x', padx=20, pady=10)
        
        result = [None]
        
        def choose_original():
            result[0] = original_type.lower()
            dialog.destroy()
        
        def choose_suggested():
            result[0] = suggested_type
            dialog.destroy()
        
        def choose_custom():
            custom = simpledialog.askstring("Custom B Light Type", "Enter custom B light feature type:")
            if custom:
                result[0] = custom.lower()
            dialog.destroy()
        
        tk.Button(button_frame, text=f"Use Original\n({original_type})", 
                 command=choose_original, bg='lightcoral').pack(side='left', padx=5)
        tk.Button(button_frame, text=f"Use AI Suggested\n({suggested_type})", 
                 command=choose_suggested, bg='lightgreen').pack(side='left', padx=5)
        tk.Button(button_frame, text="Custom B Type", 
                 command=choose_custom, bg='lightyellow').pack(side='left', padx=5)
        
        dialog.wait_window()
        return result[0]
    
    def update_feature_list(self):
        """Update the B light specific feature list display"""
        # Clear existing items
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        
        # Add B light features to tree
        for feature in self.manual_features:
            wavelength = f"{feature['wavelength']:.1f}"
            original = feature['original_type']
            standardized = feature.get('standardized_type', original)
            confidence = f"{feature.get('confidence', 0.5):.2f}"
            
            # B light specific context info
            context = feature.get('context', {})
            width = f"{context.get('width', 0):.0f}"
            notes = "Plateau-like" if context.get('width', 0) > 50 else "Shoulder-like" if context.get('asymmetry', 0) > 0.3 else "Broad"
            
            self.feature_tree.insert('', 'end', values=(
                wavelength, original, standardized, width, confidence, notes
            ))
    
    def show_naming_rules(self):
        """Show B light specific naming rules dialog"""
        if not HAS_NAMING_SYSTEM:
            messagebox.showinfo("Not Available", "Naming system not loaded")
            return
        
        rules_window = tk.Toplevel(self.root)
        rules_window.title("B/H Light Feature Naming Rules")
        rules_window.geometry("700x600")
        
        text_widget = tk.Text(rules_window, wrap='word', font=('Arial', 10))
        scrollbar = tk.Scrollbar(rules_window, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # B light specific rules
        b_light_rules = f"""B/H LIGHT FEATURE NAMING RULES
{'='*40}

SPECIFIC TO B/H LIGHT ANALYSIS:

PLATEAU vs SHOULDER DISAMBIGUATION:
• Plateau (B light): Width >50nm, low asymmetry (<0.2), flat appearance
• Shoulder (B light): Width <30nm, high asymmetry (>0.3), attached to peak
• Context Analysis: Uses slope variation and peak attachment

COMMON B LIGHT FEATURES:
• Broad Peak: Wide absorption with clear maximum
• Valley: Low transmission regions between features  
• Absorption Edge: Sharp absorption increase
• Baseline Shift: Overall transmission level change

B LIGHT CHARACTERISTICS:
• Best for identifying broad, diffuse features
• Excellent for plateau detection
• Good for mineral identification features
• Wide wavelength coverage for comprehensive analysis

NAMING CONSISTENCY RULES:
{self.naming_system.create_standardization_report() if self.naming_system else 'System not available'}

WORKFLOW RECOMMENDATIONS:
1. Load B light spectrum
2. Identify broad features visually
3. Click to mark - system suggests standardized name
4. Review disambiguation for plateau vs shoulder
5. Accept or override AI suggestion
6. Features saved with both original and standardized names
"""
        
        text_widget.insert('1.0', b_light_rules)
        text_widget.config(state='disabled')
    
    def save_features(self):
        """Save B light features to database and file"""
        if not self.manual_features:
            messagebox.showwarning("No B Features", "No B light features to save")
            return
        
        # Save to B light specific file
        filename = f"{self.current_gem_id or 'unknown'}_B_features.csv"
        
        try:
            df = pd.DataFrame(self.manual_features)
            df.to_csv(filename, index=False)
            self.update_status(f"Saved {len(self.manual_features)} B light features to {filename}")
            messagebox.showinfo("B Features Saved", f"B light features saved to {filename}")
        except Exception as e:
            messagebox.showerror("B Save Error", f"Error saving B features: {e}")
    
    def load_previous_features(self):
        """Load previously saved B light features"""
        if not self.current_gem_id:
            messagebox.showwarning("No Gem ID", "Set gem ID first")
            return
        
        filename = f"{self.current_gem_id}_B_features.csv"
        
        if not os.path.exists(filename):
            messagebox.showinfo("Not Found", f"No saved B features found for {self.current_gem_id}")
            return
        
        try:
            df = pd.read_csv(filename)
            self.manual_features = df.to_dict('records')
            self.update_feature_list()
            self.plot_spectrum()
            self.update_status(f"Loaded {len(self.manual_features)} B light features from {filename}")
        except Exception as e:
            messagebox.showerror("B Load Error", f"Error loading B features: {e}")
    
    def edit_feature(self, event):
        """Edit selected B light feature"""
        selection = self.feature_tree.selection()
        if not selection:
            return
        
        item = self.feature_tree.item(selection[0])
        values = item['values']
        
        # Find corresponding B light feature
        wavelength = float(values[0])
        for i, feature in enumerate(self.manual_features):
            if abs(feature['wavelength'] - wavelength) < 0.1:
                # B light specific edit dialog
                new_type = simpledialog.askstring(
                    "Edit B Feature", 
                    f"Edit B light feature type at {wavelength}nm:",
                    initialvalue=feature['original_type']
                )
                if new_type:
                    # Re-standardize for B light
                    if HAS_NAMING_SYSTEM and self.naming_system:
                        standardized = self.naming_system.standardize_feature_name(
                            new_type, feature.get('context', {})
                        )
                    else:
                        standardized = new_type.lower()
                    
                    self.manual_features[i]['original_type'] = new_type
                    self.manual_features[i]['standardized_type'] = standardized
                    
                    self.update_feature_list()
                    self.plot_spectrum()
                    self.update_status(f"Updated B light feature: {new_type} → {standardized}")
                break
    
    def update_status(self, message):
        """Update B light specific status bar"""
        self.status_var.set(f"[B/H Light] {message}")
    
    def run(self):
        """Start the enhanced B/H light analyzer"""
        print("🚀 Starting Enhanced B/H Light Analyzer...")
        self.root.mainloop()

def main():
    """Main function for B/H light analyzer"""
    print("🔴 Enhanced B/H Light Manual Analyzer - Gemini Spectral Analysis")
    print("Features: Naming consistency, plateau vs shoulder disambiguation, CSV integration")
    
    analyzer = EnhancedHalogenAnalyzer()
    analyzer.run()

if __name__ == '__main__':
    main()