#!/usr/bin/env python3
"""
ENHANCED GEMINI LASER ANALYZER (L LIGHT) - GEMINI SPECTRAL ANALYSIS
Enhanced version of: gemini_laser_analyzer.py
Adds: Structural feature naming consistency, sharp peak detection, doublet analysis, CSV integration
Author: Enhanced with naming consistency system for L light analysis
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

class EnhancedGeminiLaserAnalyzer:
    """Enhanced L Light Manual Analyzer with naming consistency"""
    
    def __init__(self):
        self.light_source = 'L'
        self.current_gem_id = None
        self.wavelengths = None
        self.intensities = None
        self.manual_features = []
        
        # Initialize naming system if available
        if HAS_NAMING_SYSTEM:
            self.naming_system = StructuralFeatureNamingSystem()
            print("✅ Enhanced L light analyzer with naming consistency")
        else:
            self.naming_system = None
            print("✅ Basic L light analyzer (no naming system)")
        
        # L light specific feature types (sharp features, peaks, doublets)
        self.feature_types = {
            'peak_sharp': 'Narrow, intense absorption peak (key L light feature)',
            'peak_broad': 'Wide absorption peak',
            'doublet': 'Two closely spaced sharp peaks (L light specialty)',
            'multiplet': 'Multiple closely related peaks',
            'shoulder': 'Asymmetric extension from sharp peak',
            'valley': 'Minimum between sharp features',
            'absorption_edge': 'Sharp absorption onset (high resolution)',
            'split_peak': 'Peak showing fine structure',
            'satellite_peak': 'Small peak near main feature'
        }
        
        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Setup L light specific GUI"""
        self.root = tk.Tk()
        self.root.title("Enhanced L Light Manual Analyzer - Gemini Spectral Analysis")
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
        """Create L light specific control panel"""
        control_frame = tk.LabelFrame(parent, text="L Light Controls (Sharp Features & Doublets)", 
                                    font=('Arial', 12, 'bold'), fg='darkblue')
        control_frame.pack(fill='x', pady=(0, 10))
        
        # File operations
        file_frame = tk.Frame(control_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Button(file_frame, text="Load L Spectrum", command=self.load_spectrum,
                 bg='lightblue', font=('Arial', 10)).pack(side='left', padx=5)
        
        tk.Button(file_frame, text="Save L Features", command=self.save_features,
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
        
        # L light specific feature selection
        feature_frame = tk.Frame(control_frame)
        feature_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(feature_frame, text="L Light Feature Type:", 
                font=('Arial', 10, 'bold')).pack(side='left')
        self.feature_type_var = tk.StringVar()
        
        # L light specific feature dropdown with descriptions
        feature_values = [f"{ftype} - {desc}" for ftype, desc in self.feature_types.items()]
        self.feature_combo = ttk.Combobox(feature_frame, textvariable=self.feature_type_var, 
                                        values=feature_values, width=55)
        self.feature_combo.pack(side='left', padx=5)
        
        # Sharp peak vs doublet guidance for L light
        guidance_frame = tk.Frame(control_frame)
        guidance_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Label(guidance_frame, text="💡 L Light Guidance:", 
                font=('Arial', 9, 'bold'), fg='darkblue').pack(side='left')
        tk.Label(guidance_frame, text="Sharp Peak: <10nm wide • Doublet: Two peaks <20nm apart • High resolution", 
                font=('Arial', 8), fg='gray').pack(side='left', padx=5)
        
        # Naming system status
        naming_frame = tk.Frame(control_frame)
        naming_frame.pack(fill='x', padx=5, pady=2)
        
        status_color = 'green' if HAS_NAMING_SYSTEM else 'orange'
        status_text = "✅ Naming Consistency: ON" if HAS_NAMING_SYSTEM else "⚠️ Naming Consistency: OFF"
        tk.Label(naming_frame, text=status_text, fg=status_color, 
                font=('Arial', 9, 'bold')).pack(side='left')
        
        if HAS_NAMING_SYSTEM:
            tk.Button(naming_frame, text="Show L Light Rules", command=self.show_naming_rules,
                     bg='lightcyan', font=('Arial', 8)).pack(side='right', padx=5)
    
    def create_plot_area(self, parent):
        """Create L light specific plotting area"""
        plot_frame = tk.LabelFrame(parent, text="L Light Spectrum - Enhanced Sharp Feature Detection (linewidth=0.5)", 
                                 font=('Arial', 12, 'bold'), fg='darkblue')
        plot_frame.pack(fill='both', expand=True)
        
        # Create matplotlib figure with your preferred settings
        self.fig = Figure(figsize=(12, 6))
        self.ax = self.fig.add_subplot(111)
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bind click event for feature marking
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        # Initialize L light plot
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Transmission (%)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('L Light Spectral Analysis - Sharp Peaks & Doublets')
        
        # Add instructions
        self.ax.text(0.02, 0.98, 'Click on spectrum to mark features\nL Light: Best for sharp peaks, doublets, high resolution', 
                    transform=self.ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def create_feature_list(self, parent):
        """Create enhanced feature list for L light"""
        list_frame = tk.LabelFrame(parent, text="L Light Features - Enhanced with Sharp Peak Analysis", 
                                 font=('Arial', 12, 'bold'), fg='darkblue')
        list_frame.pack(fill='x', pady=(10, 0))
        
        # Create treeview with L light specific columns
        columns = ('Wavelength', 'L Feature Type', 'Standardized', 'FWHM', 'Intensity', 'Notes')
        self.feature_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=6)
        
        # Configure columns for L light analysis
        self.feature_tree.heading('Wavelength', text='Wavelength (nm)')
        self.feature_tree.heading('L Feature Type', text='Original L Type')
        self.feature_tree.heading('Standardized', text='Standardized Type') 
        self.feature_tree.heading('FWHM', text='FWHM (nm)')
        self.feature_tree.heading('Intensity', text='Intensity')
        self.feature_tree.heading('Notes', text='L Light Notes')
        
        self.feature_tree.column('Wavelength', width=100)
        self.feature_tree.column('L Feature Type', width=150)
        self.feature_tree.column('Standardized', width=150)
        self.feature_tree.column('FWHM', width=80)
        self.feature_tree.column('Intensity', width=80)
        self.feature_tree.column('Notes', width=150)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.feature_tree.yview)
        self.feature_tree.configure(yscrollcommand=scrollbar.set)
        
        self.feature_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Bind double-click for editing
        self.feature_tree.bind('<Double-1>', self.edit_feature)
    
    def create_status_bar(self):
        """Create L light specific status bar"""
        self.status_var = tk.StringVar()
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x')
        self.update_status("L Light Analyzer Ready - Enhanced with sharp peak detection")
    
    def load_spectrum(self):
        """Load L light spectrum data"""
        filename = filedialog.askopenfilename(
            title="Load L Light Spectrum File",
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
                
                # Extract gem ID from filename (L light specific)
                gem_id = Path(filename).stem
                # Remove L light suffixes
                for suffix in ['L', 'l', 'LC1', 'lc1', 'LASER', 'laser']:
                    if gem_id.endswith(suffix):
                        gem_id = gem_id[:-len(suffix)]
                        break
                
                self.gem_id_var.set(gem_id)
                self.current_gem_id = gem_id
                self.update_gem_info()
                
                # Plot the L light spectrum
                self.plot_spectrum()
                self.update_status(f"Loaded L spectrum: {Path(filename).name}")
                
            except Exception as e:
                messagebox.showerror("L Spectrum Load Error", f"Error loading L light file: {e}")
                self.update_status(f"L load failed: {e}")
    
    def update_gem_info(self, event=None):
        """Update gem information using enhanced naming system"""
        gem_id = self.gem_id_var.get().strip()
        if not gem_id:
            return
        
        self.current_gem_id = gem_id
        
        if HAS_NAMING_SYSTEM and self.naming_system:
            gem_info = self.naming_system.get_gem_description(gem_id)
            desc = gem_info.get('full_description', f'Gem {gem_id}')
            
            # Add L light specific info
            species = gem_info.get('species', '')
            variety = gem_info.get('variety', '')
            if species:
                desc = f"L Light Analysis: {species} {variety} - {desc}".strip(' -')
        else:
            desc = f"L Light Analysis: Gem {gem_id}"
        
        self.gem_desc_var.set(desc)
        self.update_status(f"L Light - Gem: {gem_id} - {desc}")
    
    def plot_spectrum(self):
        """Plot L light spectrum with enhanced sharp feature visualization"""
        if self.wavelengths is None or self.intensities is None:
            return
        
        self.ax.clear()
        
        # Plot L light spectrum with your preferred linewidth=0.5
        self.ax.plot(self.wavelengths, self.intensities, 'b-', linewidth=0.5, 
                    label='L Light Spectrum', color='darkblue')
        
        # L light specific feature colors
        l_light_colors = {
            'peak_sharp': 'green',
            'peak_broad': 'blue',
            'doublet': 'red',
            'multiplet': 'purple',
            'shoulder': 'orange', 
            'valley': 'brown',
            'absorption_edge': 'black',
            'split_peak': 'magenta',
            'satellite_peak': 'gray'
        }
        
        # Plot L light features
        for feature in self.manual_features:
            wavelength = feature['wavelength']
            orig_type = feature['original_type']
            std_type = feature.get('standardized_type', orig_type)
            fwhm = feature.get('context', {}).get('fwhm', 0)
            intensity = feature.get('intensity', 0)
            
            color = l_light_colors.get(std_type, 'black')
            
            # Vertical line at feature position
            self.ax.axvline(wavelength, color=color, linewidth=0.5, alpha=0.8)
            
            # Find intensity at this wavelength if not stored
            if intensity == 0:
                idx = np.argmin(np.abs(self.wavelengths - wavelength))
                intensity = self.intensities[idx]
            
            # L light specific annotation with FWHM
            if orig_type != std_type:
                label = f"{std_type}\n({orig_type})\nFWHM:{fwhm:.1f}nm"
            else:
                label = f"{std_type}\nFWHM:{fwhm:.1f}nm"
                
            self.ax.annotate(label, (wavelength, intensity), 
                           rotation=90, fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Transmission (%)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f'L Light Analysis: {self.current_gem_id or "Unknown Gem"} - Sharp Features & Doublets')
        
        # Add consolidated legend for L light features (your approach)
        legend_elements = []
        used_types = set(f.get('standardized_type', f['original_type']) for f in self.manual_features)
        for ftype in used_types:
            color = l_light_colors.get(ftype, 'black')
            legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=0.5, label=ftype.title()))
        
        if legend_elements:
            self.ax.legend(handles=legend_elements, title='L Light Features', loc='upper right')
        
        self.canvas.draw()
    
    def on_plot_click(self, event):
        """Handle L light plot click for feature marking"""
        if event.inaxes != self.ax or not event.xdata:
            return
        
        wavelength = event.xdata
        
        # Get L light feature type from combobox
        selected = self.feature_type_var.get()
        if not selected:
            messagebox.showwarning("No L Feature Selected", "Please select an L light feature type first")
            return
        
        # Extract just the feature type name
        feature_type = selected.split(' - ')[0]
        
        # Calculate L light specific context
        context = self.calculate_l_light_context(wavelength)
        
        if HAS_NAMING_SYSTEM and self.naming_system:
            # Get standardized name with L light context
            standardized_type = self.naming_system.standardize_feature_name(feature_type, context)
            confidence = context.get('confidence', 0.8)
            
            # L light specific disambiguation for sharp peaks vs doublets
            if feature_type.lower() in ['peak_sharp', 'doublet', 'multiplet'] and confidence < 0.9:
                choice = self.show_l_light_disambiguation_dialog(feature_type, standardized_type, context)
                if choice:
                    standardized_type = choice
            
            # Add to database
            success = self.naming_system.add_structural_feature(
                gem_id=self.current_gem_id or 'unknown',
                light_source='L',
                wavelength=wavelength,
                feature_type=feature_type,
                context=context,
                analysis_method='manual_l_light'
            )
            
            if not success:
                messagebox.showerror("Database Error", "Failed to save L light feature to database")
                return
        else:
            standardized_type = feature_type.lower()
            confidence = 0.5
        
        # Add to local L light features list
        feature = {
            'wavelength': wavelength,
            'original_type': feature_type,
            'standardized_type': standardized_type,
            'confidence': confidence,
            'context': context,
            'intensity': self.get_intensity_at_wavelength(wavelength)
        }
        
        self.manual_features.append(feature)
        self.update_feature_list()
        self.plot_spectrum()
        
        # Show L light specific standardization result
        fwhm = context.get('fwhm', 0)
        if feature_type.lower() != standardized_type:
            self.update_status(f"L Light: {feature_type} → {standardized_type} at {wavelength:.1f}nm (FWHM:{fwhm:.1f}nm)")
        else:
            self.update_status(f"L Light: {standardized_type} at {wavelength:.1f}nm (FWHM:{fwhm:.1f}nm)")
    
    def get_intensity_at_wavelength(self, wavelength):
        """Get intensity value at specific wavelength"""
        if self.wavelengths is None or self.intensities is None:
            return 0
        idx = np.argmin(np.abs(self.wavelengths - wavelength))
        return self.intensities[idx]
    
    def calculate_l_light_context(self, wavelength):
        """Calculate L light specific context for feature disambiguation"""
        if self.wavelengths is None or self.intensities is None:
            return {'confidence': 0.5, 'fwhm': 0}
        
        # Find nearest data points
        center_idx = np.argmin(np.abs(self.wavelengths - wavelength))
        
        # L light specific window (smaller for sharp features)
        window_size = 15  # Smaller window for L light sharp features
        mask = np.abs(self.wavelengths - wavelength) <= window_size
        
        if not np.any(mask):
            return {'confidence': 0.5, 'fwhm': 0}
        
        window_wavelengths = self.wavelengths[mask]
        window_intensities = self.intensities[mask]
        
        # L light specific metrics - FWHM calculation
        fwhm = self.calculate_fwhm(window_wavelengths, window_intensities)
        intensity_std = np.std(window_intensities)
        
        # Calculate sharpness (important for L light peak classification)
        if len(window_intensities) >= 3:
            # Second derivative for sharpness
            second_deriv = np.gradient(np.gradient(window_intensities, window_wavelengths), window_wavelengths)
            sharpness = np.max(np.abs(second_deriv))
        else:
            sharpness = 0
        
        # Detect nearby peaks for doublet identification
        nearby_peaks = self.detect_nearby_peaks(wavelength, search_range=20)
        
        # L light specific slope analysis for peak detection
        if len(window_intensities) >= 3:
            slopes = np.gradient(window_intensities, window_wavelengths)
            max_slope = np.max(np.abs(slopes))
        else:
            max_slope = 0
        
        context = {
            'fwhm': fwhm,
            'sharpness': sharpness,
            'intensity_std': intensity_std,
            'max_slope': max_slope,
            'nearby_peaks': nearby_peaks,
            'light_source': 'L',
            'confidence': 0.8
        }
        
        # L light specific confidence adjustment
        if fwhm < 5 and sharpness > 1:  # Very sharp peak
            context['confidence'] = 0.95
        elif len(nearby_peaks) >= 2:  # Multiple peaks nearby = doublet/multiplet
            context['confidence'] = 0.9
        elif fwhm > 20:  # Broad for L light
            context['confidence'] = 0.7
        
        return context
    
    def calculate_fwhm(self, wavelengths, intensities):
        """Calculate Full Width at Half Maximum for L light peaks"""
        if len(intensities) < 3:
            return 0
        
        # Find minimum (peak in absorption spectrum)
        min_idx = np.argmin(intensities)
        min_intensity = intensities[min_idx]
        
        # Find baseline (average of edges)
        baseline = (intensities[0] + intensities[-1]) / 2
        
        # Half maximum
        half_max = min_intensity + (baseline - min_intensity) / 2
        
        # Find points at half maximum
        left_idx = right_idx = min_idx
        
        # Search left
        for i in range(min_idx, -1, -1):
            if intensities[i] >= half_max:
                left_idx = i
                break
        
        # Search right
        for i in range(min_idx, len(intensities)):
            if intensities[i] >= half_max:
                right_idx = i
                break
        
        if left_idx != right_idx:
            fwhm = wavelengths[right_idx] - wavelengths[left_idx]
            return max(0, fwhm)  # Ensure positive
        
        return 0
    
    def detect_nearby_peaks(self, wavelength, search_range=20):
        """Detect peaks near given wavelength for doublet identification"""
        if self.wavelengths is None or self.intensities is None:
            return []
        
        # Find peaks in nearby region
        try:
            from scipy import signal
            # Mask for search range
            mask = np.abs(self.wavelengths - wavelength) <= search_range
            if not np.any(mask):
                return []
            
            region_wavelengths = self.wavelengths[mask]
            region_intensities = self.intensities[mask]
            
            # Find minima (peaks in absorption)
            inverted = -region_intensities
            peaks, properties = signal.find_peaks(inverted, height=0, distance=3, prominence=0.01)
            
            peak_wavelengths = region_wavelengths[peaks]
            return peak_wavelengths.tolist()
        except:
            return []
    
    def show_l_light_disambiguation_dialog(self, original_type, suggested_type, context):
        """Show L light specific disambiguation dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("L Light Feature Disambiguation")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (250)
        y = (dialog.winfo_screenheight() // 2) - (200)
        dialog.geometry(f"500x400+{x}+{y}")
        
        # L light specific content
        tk.Label(dialog, text="L Light Feature Disambiguation", 
                font=('Arial', 14, 'bold'), fg='darkblue').pack(pady=10)
        
        tk.Label(dialog, text=f"Original: {original_type}", 
                font=('Arial', 12)).pack(pady=5)
        tk.Label(dialog, text=f"AI Suggested: {suggested_type}", 
                font=('Arial', 12), fg='blue').pack(pady=5)
        
        # L light specific context
        context_frame = tk.LabelFrame(dialog, text="L Light Analysis Context", fg='darkblue')
        context_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(context_frame, text=f"FWHM: {context.get('fwhm', 0):.1f} nm").pack(anchor='w')
        tk.Label(context_frame, text=f"Sharpness: {context.get('sharpness', 0):.3f}").pack(anchor='w')
        tk.Label(context_frame, text=f"Max slope: {context.get('max_slope', 0):.3f}").pack(anchor='w')
        nearby_peaks = context.get('nearby_peaks', [])
        tk.Label(context_frame, text=f"Nearby peaks: {len(nearby_peaks)} detected").pack(anchor='w')
        
        # L light specific guidance
        guidance_frame = tk.LabelFrame(dialog, text="L Light Guidelines", fg='darkblue')
        guidance_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(guidance_frame, text="• Sharp Peak: FWHM <5nm, high sharpness, single", 
                font=('Arial', 9), fg='darkgreen').pack(anchor='w')
        tk.Label(guidance_frame, text="• Doublet: Two peaks <20nm apart, similar intensities", 
                font=('Arial', 9), fg='darkred').pack(anchor='w')
        tk.Label(guidance_frame, text="• Multiplet: Multiple peaks clustered together", 
                font=('Arial', 9), fg='purple').pack(anchor='w')
        tk.Label(guidance_frame, text="• Broad Peak: FWHM >15nm for L light", 
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
            custom = simpledialog.askstring("Custom L Light Type", "Enter custom L light feature type:")
            if custom:
                result[0] = custom.lower()
            dialog.destroy()
        
        tk.Button(button_frame, text=f"Use Original\n({original_type})", 
                 command=choose_original, bg='lightcoral').pack(side='left', padx=5)
        tk.Button(button_frame, text=f"Use AI Suggested\n({suggested_type})", 
                 command=choose_suggested, bg='lightgreen').pack(side='left', padx=5)
        tk.Button(button_frame, text="Custom L Type", 
                 command=choose_custom, bg='lightyellow').pack(side='left', padx=5)
        
        dialog.wait_window()
        return result[0]
    
    def update_feature_list(self):
        """Update the L light specific feature list display"""
        # Clear existing items
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        
        # Add L light features to tree
        for feature in self.manual_features:
            wavelength = f"{feature['wavelength']:.1f}"
            original = feature['original_type']
            standardized = feature.get('standardized_type', original)
            confidence = f"{feature.get('confidence', 0.5):.2f}"
            
            # L light specific context info
            context = feature.get('context', {})
            fwhm = f"{context.get('fwhm', 0):.1f}"
            intensity = f"{feature.get('intensity', 0):.1f}"
            
            # L light specific notes
            sharpness = context.get('sharpness', 0)
            if sharpness > 1:
                notes = "Very sharp"
            elif context.get('fwhm', 0) < 5:
                notes = "Sharp peak"
            elif len(context.get('nearby_peaks', [])) > 1:
                notes = "Multiple nearby"
            else:
                notes = "Standard"
            
            self.feature_tree.insert('', 'end', values=(
                wavelength, original, standardized, fwhm, intensity, notes
            ))
    
    def show_naming_rules(self):
        """Show L light specific naming rules dialog"""
        if not HAS_NAMING_SYSTEM:
            messagebox.showinfo("Not Available", "Naming system not loaded")
            return
        
        rules_window = tk.Toplevel(self.root)
        rules_window.title("L Light Feature Naming Rules")
        rules_window.geometry("700x600")
        
        text_widget = tk.Text(rules_window, wrap='word', font=('Arial', 10))
        scrollbar = tk.Scrollbar(rules_window, command=text_widget.yview)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # L light specific rules
        l_light_rules = f"""L LIGHT FEATURE NAMING RULES
{'='*40}

SPECIFIC TO L LIGHT ANALYSIS:

SHARP PEAK vs DOUBLET vs MULTIPLET:
• Sharp Peak: FWHM <5nm, single isolated feature, high sharpness
• Doublet: Two peaks <20nm apart, similar intensities
• Multiplet: Multiple peaks clustered together (>2 peaks)
• Context Analysis: Uses FWHM, sharpness, nearby peak detection

COMMON L LIGHT FEATURES:
• Peak Sharp: Narrow intense absorption, key L light feature
• Peak Broad: Wide absorption (>15nm FWHM for L light)
• Shoulder: Asymmetric extension from sharp peak
• Split Peak: Peak showing fine structure splitting
• Satellite Peak: Small peak near main feature

L LIGHT CHARACTERISTICS:
• High resolution analysis capability
• Excellent for sharp peak detection  
• Superior doublet resolution
• Natural vs synthetic discrimination
• Precise wavelength measurements

FWHM GUIDELINES FOR L LIGHT:
• Very Sharp: <3nm FWHM
• Sharp: 3-8nm FWHM  
• Medium: 8-15nm FWHM
• Broad: >15nm FWHM (unusual for L light)

NAMING CONSISTENCY RULES:
{self.naming_system.create_standardization_report() if self.naming_system else 'System not available'}

WORKFLOW RECOMMENDATIONS:
1. Load L light spectrum
2. Identify sharp features and doublets
3. Click to mark - system calculates FWHM and sharpness
4. Review disambiguation for complex features
5. Accept or override AI suggestion based on context
6. Features saved with detailed L light metrics
"""
        
        text_widget.insert('1.0', l_light_rules)
        text_widget.config(state='disabled')
    
    def save_features(self):
        """Save L light features to database and file"""
        if not self.manual_features:
            messagebox.showwarning("No L Features", "No L light features to save")
            return
        
        # Save to L light specific file
        filename = f"{self.current_gem_id or 'unknown'}_L_features.csv"
        
        try:
            df = pd.DataFrame(self.manual_features)
            df.to_csv(filename, index=False)
            self.update_status(f"Saved {len(self.manual_features)} L light features to {filename}")
            messagebox.showinfo("L Features Saved", f"L light features saved to {filename}")
        except Exception as e:
            messagebox.showerror("L Save Error", f"Error saving L features: {e}")
    
    def load_previous_features(self):
        """Load previously saved L light features"""
        if not self.current_gem_id:
            messagebox.showwarning("No Gem ID", "Set gem ID first")
            return
        
        filename = f"{self.current_gem_id}_L_features.csv"
        
        if not os.path.exists(filename):
            messagebox.showinfo("Not Found", f"No saved L features found for {self.current_gem_id}")
            return
        
        try:
            df = pd.read_csv(filename)
            self.manual_features = df.to_dict('records')
            self.update_feature_list()
            self.plot_spectrum()
            self.update_status(f"Loaded {len(self.manual_features)} L light features from {filename}")
        except Exception as e:
            messagebox.showerror("L Load Error", f"Error loading L features: {e}")
    
    def edit_feature(self, event):
        """Edit selected L light feature"""
        selection = self.feature_tree.selection()
        if not selection:
            return
        
        item = self.feature_tree.item(selection[0])
        values = item['values']
        
        # Find corresponding L light feature
        wavelength = float(values[0])
        for i, feature in enumerate(self.manual_features):
            if abs(feature['wavelength'] - wavelength) < 0.1:
                # L light specific edit dialog
                new_type = simpledialog.askstring(
                    "Edit L Feature", 
                    f"Edit L light feature type at {wavelength}nm:",
                    initialvalue=feature['original_type']
                )
                if new_type:
                    # Re-standardize for L light
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
                    self.update_status(f"Updated L light feature: {new_type} → {standardized}")
                break
    
    def update_status(self, message):
        """Update L light specific status bar"""
        self.status_var.set(f"[L Light] {message}")
    
    def run(self):
        """Start the enhanced L light analyzer"""
        print("🚀 Starting Enhanced L Light Analyzer...")
        self.root.mainloop()

def main():
    """Main function for L light analyzer"""
    print("🔵 Enhanced L Light Manual Analyzer - Gemini Spectral Analysis")
    print("Features: Sharp peak detection, doublet analysis, FWHM calculation, CSV integration")
    
    analyzer = EnhancedGeminiLaserAnalyzer()
    analyzer.run()

if __name__ == '__main__':
    main()