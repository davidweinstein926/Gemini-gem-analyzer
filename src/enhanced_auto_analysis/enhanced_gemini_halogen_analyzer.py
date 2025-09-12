#!/usr/bin/env python3
"""
ENHANCED HALOGEN ANALYZER (B LIGHT) - COMPLETE INTEGRATION
Enhanced with: Audio bleep feedback, automatic database import, real-time relative height 
measurements, seamless integration with main analysis system, and advanced feature detection.

Major Enhancements:
- Audio bleep feedback for each feature detected/marked
- Automatic CSV-to-database import after analysis completion
- Real-time relative height calculations across light sources
- Enhanced plateau vs shoulder disambiguation with audio cues
- Progressive feature detection with different audio tones
- Direct integration with main gemological analysis system
- Advanced baseline correction and normalization metadata
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
import time
from datetime import datetime

# Audio support for bleep functionality
try:
    import winsound
    HAS_AUDIO = True
except ImportError:
    try:
        import pygame
        pygame.mixer.init()
        HAS_AUDIO = True
    except ImportError:
        HAS_AUDIO = False

# Enhanced configuration
DATABASE_PATH = "multi_structural_gem_data.db"
OUTPUT_DIRECTORY = r"c:\users\david\gemini sp10 structural data\halogen"

class EnhancedHalogenAnalyzer:
    """Enhanced B/H Light Manual Analyzer with complete integration"""
    
    def __init__(self):
        self.light_source = 'Halogen'
        self.current_gem_id = None
        self.wavelengths = None
        self.intensities = None
        self.manual_features = []
        self.baseline_data = None
        
        # Enhanced features
        self.bleep_enabled = True
        self.auto_import_enabled = True
        self.relative_height_enabled = True
        self.validation_enabled = True
        
        # Audio feedback frequencies for different features
        self.feature_frequencies = {
            'plateau': 800,
            'broad_peak': 900,
            'shoulder': 700,
            'valley': 600,
            'absorption_edge': 1000,
            'doublet': 950,
            'baseline_shift': 500,
            'broad_absorption': 850,
            'completion': 1200,
            'error': 400,
            'validation': 750
        }
        
        # B/H light specific feature types with enhanced descriptions
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
        
        # Enhanced analysis metrics
        self.analysis_session_id = None
        self.feature_count = 0
        self.quality_metrics = {
            'baseline_quality': None,
            'feature_consistency': None,
            'relative_height_coverage': None
        }
        
        # Setup enhanced GUI
        self.setup_enhanced_gui()
        
        print("Enhanced B/H Light Analyzer with complete integration initialized")
        if self.bleep_enabled and HAS_AUDIO:
            self.play_bleep('completion')
    
    def play_bleep(self, feature_type="standard", duration=200):
        """Play enhanced audio bleep for B light feature detection"""
        if not self.bleep_enabled or not HAS_AUDIO:
            return
        
        try:
            freq = self.feature_frequencies.get(feature_type, 800)
            
            if 'winsound' in sys.modules:
                winsound.Beep(freq, duration)
            elif 'pygame' in sys.modules:
                sample_rate = 22050
                frames = int(duration * sample_rate / 1000)
                arr = np.zeros(frames)
                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * freq * i / sample_rate)
                arr = (arr * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
                time.sleep(duration / 1000)
                
        except Exception as e:
            print(f"Enhanced audio error: {e}")
    
    def setup_enhanced_gui(self):
        """Setup enhanced B/H light specific GUI with integration features"""
        self.root = tk.Tk()
        self.root.title("Enhanced B/H Light Analyzer - Complete Integration")
        self.root.geometry("1400x900")
        
        # Create enhanced main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Enhanced control panel
        self.create_enhanced_control_panel(main_frame)
        
        # Enhanced plotting area
        self.create_enhanced_plot_area(main_frame)
        
        # Enhanced feature list
        self.create_enhanced_feature_list(main_frame)
        
        # Enhanced status bar
        self.create_enhanced_status_bar()
    
    def create_enhanced_control_panel(self, parent):
        """Create enhanced B/H light specific control panel with integration controls"""
        control_frame = tk.LabelFrame(parent, text="Enhanced B/H Light Controls (Complete Integration)", 
                                    font=('Arial', 12, 'bold'), fg='darkred')
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Enhanced file operations
        file_frame = tk.Frame(control_frame)
        file_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Button(file_frame, text="Load B Spectrum", command=self.load_enhanced_spectrum,
                 bg='lightcoral', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        tk.Button(file_frame, text="Save Enhanced B Features", command=self.save_enhanced_features,
                 bg='lightgreen', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        tk.Button(file_frame, text="Auto-Import to Database", command=self.manual_database_import,
                 bg='lightblue', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        # Enhanced gem ID entry
        id_frame = tk.Frame(control_frame)
        id_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(id_frame, text="Gem ID:", font=('Arial', 10, 'bold')).pack(side='left')
        self.gem_id_var = tk.StringVar()
        self.gem_id_entry = tk.Entry(id_frame, textvariable=self.gem_id_var, width=15,
                                   font=('Arial', 10))
        self.gem_id_entry.pack(side='left', padx=5)
        self.gem_id_entry.bind('<Return>', self.update_enhanced_gem_info)
        
        # Enhanced gem description with database lookup
        self.gem_desc_var = tk.StringVar(value="Enter Gem ID for enhanced analysis")
        tk.Label(id_frame, textvariable=self.gem_desc_var, 
                font=('Arial', 9), fg='darkblue', wraplength=500).pack(side='left', padx=10)
        
        # Enhanced B/H light specific feature selection
        feature_frame = tk.Frame(control_frame)
        feature_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(feature_frame, text="Enhanced B Light Feature Type:", 
                font=('Arial', 10, 'bold')).pack(side='left')
        self.feature_type_var = tk.StringVar()
        
        # Enhanced feature dropdown with audio preview
        feature_values = [f"{ftype} - {desc}" for ftype, desc in self.feature_types.items()]
        self.feature_combo = ttk.Combobox(feature_frame, textvariable=self.feature_type_var, 
                                        values=feature_values, width=60)
        self.feature_combo.pack(side='left', padx=5)
        self.feature_combo.bind('<<ComboboxSelected>>', self.on_feature_type_selected)
        
        # Enhanced controls frame
        controls_frame = tk.Frame(control_frame)
        controls_frame.pack(fill='x', padx=5, pady=5)
        
        # Audio controls
        audio_frame = tk.LabelFrame(controls_frame, text="Enhanced Audio Controls", fg='purple')
        audio_frame.pack(side='left', padx=5, fill='y')
        
        tk.Button(audio_frame, text="Toggle Audio", command=self.toggle_enhanced_audio,
                 bg='lightyellow', font=('Arial', 8)).pack(padx=2, pady=2)
        
        tk.Button(audio_frame, text="Test Audio", command=self.test_enhanced_audio,
                 bg='lightcyan', font=('Arial', 8)).pack(padx=2, pady=2)
        
        # Integration controls
        integration_frame = tk.LabelFrame(controls_frame, text="Enhanced Integration", fg='darkgreen')
        integration_frame.pack(side='left', padx=5, fill='y')
        
        tk.Button(integration_frame, text="Toggle Auto-Import", command=self.toggle_auto_import,
                 bg='lightgreen', font=('Arial', 8)).pack(padx=2, pady=2)
        
        tk.Button(integration_frame, text="Relative Heights", command=self.toggle_relative_heights,
                 bg='orange', font=('Arial', 8)).pack(padx=2, pady=2)
        
        # Enhanced guidance
        guidance_frame = tk.Frame(control_frame)
        guidance_frame.pack(fill='x', padx=5, pady=2)
        
        tk.Label(guidance_frame, text="Enhanced B Light Guidance:", 
                font=('Arial', 9, 'bold'), fg='darkred').pack(side='left')
        tk.Label(guidance_frame, text="Plateau: >50nm, flat, audio cue 800Hz • Shoulder: <30nm, asymmetric, audio cue 700Hz", 
                font=('Arial', 8), fg='gray').pack(side='left', padx=5)
        
        # Enhanced status indicators
        status_frame = tk.Frame(control_frame)
        status_frame.pack(fill='x', padx=5, pady=2)
        
        self.audio_status_var = tk.StringVar(value=f"Audio: {'ON' if self.bleep_enabled else 'OFF'}")
        tk.Label(status_frame, textvariable=self.audio_status_var, fg='green' if self.bleep_enabled else 'red',
                font=('Arial', 9, 'bold')).pack(side='left', padx=10)
        
        self.import_status_var = tk.StringVar(value=f"Auto-Import: {'ON' if self.auto_import_enabled else 'OFF'}")
        tk.Label(status_frame, textvariable=self.import_status_var, fg='green' if self.auto_import_enabled else 'red',
                font=('Arial', 9, 'bold')).pack(side='left', padx=10)
        
        self.relative_status_var = tk.StringVar(value=f"Relative Heights: {'ON' if self.relative_height_enabled else 'OFF'}")
        tk.Label(status_frame, textvariable=self.relative_status_var, fg='green' if self.relative_height_enabled else 'red',
                font=('Arial', 9, 'bold')).pack(side='left', padx=10)
    
    def create_enhanced_plot_area(self, parent):
        """Create enhanced B/H light specific plotting area with integration features"""
        plot_frame = tk.LabelFrame(parent, text="Enhanced B/H Light Spectrum - Audio Feedback & Real-time Analysis", 
                                 font=('Arial', 12, 'bold'), fg='darkred')
        plot_frame.pack(fill='both', expand=True)
        
        # Create enhanced matplotlib figure
        self.fig = Figure(figsize=(14, 8))
        self.ax = self.fig.add_subplot(111)
        
        # Enhanced canvas
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Bind enhanced click event for feature marking with audio feedback
        self.canvas.mpl_connect('button_press_event', self.on_enhanced_plot_click)
        
        # Initialize enhanced B/H light plot
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Transmission (%)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Enhanced B/H Light Spectral Analysis - Audio Feedback & Database Integration')
        
        # Enhanced instructions with integration info
        instructions = ('Enhanced B Light Analysis:\n'
                       'Click on spectrum to mark features with audio feedback\n'
                       'Auto-import: Features saved directly to database\n'
                       'Relative heights: Calculated in real-time across light sources\n'
                       'Audio cues: Different tones for each feature type')
        
        self.ax.text(0.02, 0.98, instructions, 
                    transform=self.ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def create_enhanced_feature_list(self, parent):
        """Create enhanced feature list for B/H light with integration data"""
        list_frame = tk.LabelFrame(parent, text="Enhanced B Light Features - Real-time Database Integration", 
                                 font=('Arial', 12, 'bold'), fg='darkred')
        list_frame.pack(fill='x', pady=(10, 0))
        
        # Enhanced treeview with integration columns
        columns = ('Wavelength', 'B Feature', 'Width', 'Audio Freq', 'Relative Height', 'DB Status', 'Quality')
        self.feature_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        # Configure enhanced columns
        self.feature_tree.heading('Wavelength', text='Wavelength (nm)')
        self.feature_tree.heading('B Feature', text='B Light Feature')
        self.feature_tree.heading('Width', text='Width (nm)')
        self.feature_tree.heading('Audio Freq', text='Audio (Hz)')
        self.feature_tree.heading('Relative Height', text='Rel. Height')
        self.feature_tree.heading('DB Status', text='Database')
        self.feature_tree.heading('Quality', text='Quality')
        
        self.feature_tree.column('Wavelength', width=100)
        self.feature_tree.column('B Feature', width=150)
        self.feature_tree.column('Width', width=80)
        self.feature_tree.column('Audio Freq', width=80)
        self.feature_tree.column('Relative Height', width=80)
        self.feature_tree.column('DB Status', width=80)
        self.feature_tree.column('Quality', width=80)
        
        # Enhanced scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.feature_tree.yview)
        self.feature_tree.configure(yscrollcommand=scrollbar.set)
        
        self.feature_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Enhanced context menu
        self.feature_tree.bind('<Double-1>', self.edit_enhanced_feature)
        self.feature_tree.bind('<Button-3>', self.show_enhanced_context_menu)
    
    def create_enhanced_status_bar(self):
        """Create enhanced B/H light specific status bar with integration info"""
        self.status_var = tk.StringVar()
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief='sunken', anchor='w')
        status_bar.pack(side='bottom', fill='x')
        self.update_enhanced_status("Enhanced B/H Light Analyzer Ready - Audio, Database & Real-time Integration Active")
    
    def on_feature_type_selected(self, event=None):
        """Handle feature type selection with audio preview"""
        selected = self.feature_type_var.get()
        if selected:
            feature_type = selected.split(' - ')[0]
            if self.bleep_enabled:
                self.play_bleep(feature_type)
            print(f"Enhanced feature selected: {feature_type}")
    
    def toggle_enhanced_audio(self):
        """Toggle enhanced audio feedback system"""
        self.bleep_enabled = not self.bleep_enabled
        self.audio_status_var.set(f"Audio: {'ON' if self.bleep_enabled else 'OFF'}")
        
        if self.bleep_enabled:
            print("Enhanced audio feedback enabled")
            if HAS_AUDIO:
                self.play_bleep('completion')
        else:
            print("Enhanced audio feedback disabled")
    
    def test_enhanced_audio(self):
        """Test enhanced audio system with B light specific tones"""
        if not HAS_AUDIO:
            messagebox.showinfo("Audio Test", "Audio system not available\nInstall winsound (Windows) or pygame")
            return
        
        print("Testing enhanced B light audio system...")
        
        test_features = ['plateau', 'shoulder', 'broad_peak', 'valley', 'completion']
        for feature in test_features:
            print(f"  Playing {feature} tone ({self.feature_frequencies[feature]}Hz)...")
            self.play_bleep(feature)
            time.sleep(0.3)
        
        print("Enhanced audio test completed")
    
    def toggle_auto_import(self):
        """Toggle automatic database import"""
        self.auto_import_enabled = not self.auto_import_enabled
        self.import_status_var.set(f"Auto-Import: {'ON' if self.auto_import_enabled else 'OFF'}")
        
        status = "enabled" if self.auto_import_enabled else "disabled"
        print(f"Enhanced auto-import {status}")
        
        if self.bleep_enabled:
            self.play_bleep('validation')
    
    def toggle_relative_heights(self):
        """Toggle relative height calculations"""
        self.relative_height_enabled = not self.relative_height_enabled
        self.relative_status_var.set(f"Relative Heights: {'ON' if self.relative_height_enabled else 'OFF'}")
        
        status = "enabled" if self.relative_height_enabled else "disabled"
        print(f"Enhanced relative height calculations {status}")
        
        if self.bleep_enabled:
            self.play_bleep('validation')
    
    def load_enhanced_spectrum(self):
        """Load enhanced B/H light spectrum data with validation"""
        filename = filedialog.askopenfilename(
            title="Load Enhanced B/H Light Spectrum File",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="../../data/raw"
        )
        
        if filename:
            try:
                print(f"Loading enhanced spectrum: {os.path.basename(filename)}")
                
                # Enhanced file loading with validation
                if filename.endswith('.csv'):
                    data = pd.read_csv(filename)
                    if len(data.columns) >= 2:
                        self.wavelengths = data.iloc[:, 0].values
                        self.intensities = data.iloc[:, 1].values
                    else:
                        raise ValueError("CSV must have at least 2 columns")
                else:
                    # Enhanced space-separated text file handling
                    data = np.loadtxt(filename)
                    self.wavelengths = data[:, 0]
                    self.intensities = data[:, 1]
                
                # Enhanced validation
                if self.validation_enabled:
                    validation_results = self.validate_spectrum_data()
                    if not validation_results['valid']:
                        messagebox.showwarning("Data Validation", 
                                             f"Spectrum validation issues:\n{validation_results['message']}")
                        if self.bleep_enabled:
                            self.play_bleep('error')
                
                # Enhanced gem ID extraction
                self.current_gem_id = self.extract_enhanced_gem_id(filename)
                self.gem_id_var.set(self.current_gem_id)
                self.update_enhanced_gem_info()
                
                # Create enhanced analysis session
                self.analysis_session_id = f"{self.current_gem_id}_B_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Enhanced spectrum plotting
                self.plot_enhanced_spectrum()
                self.update_enhanced_status(f"Enhanced spectrum loaded: {os.path.basename(filename)}")
                
                if self.bleep_enabled:
                    self.play_bleep('completion')
                
            except Exception as e:
                messagebox.showerror("Enhanced Load Error", f"Error loading B light file: {e}")
                self.update_enhanced_status(f"Enhanced load failed: {e}")
                if self.bleep_enabled:
                    self.play_bleep('error')
    
    def extract_enhanced_gem_id(self, filename):
        """Enhanced gem ID extraction with B light specific patterns"""
        gem_id = Path(filename).stem
        
        # Remove B light specific suffixes
        b_light_patterns = ['B', 'b', 'BH', 'bh', 'H', 'h', 'BC1', 'BC2', 'BP1', 'BP2', '_halogen']
        
        for suffix in b_light_patterns:
            if gem_id.endswith(suffix):
                gem_id = gem_id[:-len(suffix)]
                break
        
        return gem_id if gem_id else 'unknown'
    
    def validate_spectrum_data(self):
        """Enhanced spectrum data validation for B light"""
        if self.wavelengths is None or self.intensities is None:
            return {'valid': False, 'message': 'No spectrum data loaded'}
        
        issues = []
        
        # Wavelength validation
        if len(self.wavelengths) < 10:
            issues.append("Very few data points")
        
        if self.wavelengths.min() < 200 or self.wavelengths.max() > 3000:
            issues.append(f"Unusual wavelength range: {self.wavelengths.min():.1f}-{self.wavelengths.max():.1f}nm")
        
        # Intensity validation
        if self.intensities.max() <= 1.0:
            issues.append("Data appears to be 0-1 normalized (may cause issues)")
        
        if self.intensities.min() < -10:
            issues.append(f"Very negative intensities: {self.intensities.min():.2f}")
        
        # B light specific validation
        broad_features = np.sum(np.abs(np.gradient(self.intensities)) < 0.1) / len(self.intensities)
        if broad_features < 0.3:
            issues.append("Spectrum may be too sharp for B light analysis (prefer broad features)")
        
        return {
            'valid': len(issues) == 0,
            'message': '; '.join(issues) if issues else 'Validation passed',
            'issues': issues
        }
    
    def update_enhanced_gem_info(self, event=None):
        """Update enhanced gem information with database lookup"""
        gem_id = self.gem_id_var.get().strip()
        if not gem_id:
            return
        
        self.current_gem_id = gem_id
        
        # Enhanced database lookup for gem information
        gem_info = self.lookup_gem_in_database(gem_id)
        
        if gem_info:
            desc = f"Enhanced B Light Analysis: {gem_info.get('description', gem_id)}"
            existing_features = gem_info.get('existing_features', 0)
            if existing_features > 0:
                desc += f" ({existing_features} features in database)"
        else:
            desc = f"Enhanced B Light Analysis: New Gem {gem_id}"
        
        self.gem_desc_var.set(desc)
        self.update_enhanced_status(f"Enhanced B Light - Gem: {gem_id}")
        
        if self.bleep_enabled:
            self.play_bleep('validation')
    
    def lookup_gem_in_database(self, gem_id):
        """Lookup gem information in database"""
        try:
            if not os.path.exists(DATABASE_PATH):
                return None
            
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Look for existing features
            query = "SELECT COUNT(*) as count FROM structural_features WHERE gem_id LIKE ?"
            result = pd.read_sql_query(query, conn, params=[f"%{gem_id}%"])
            existing_features = result.iloc[0]['count'] if not result.empty else 0
            
            # Look for gem description (if available)
            try:
                desc_query = "SELECT DISTINCT feature_group FROM structural_features WHERE gem_id LIKE ? LIMIT 5"
                desc_result = pd.read_sql_query(desc_query, conn, params=[f"%{gem_id}%"])
                feature_types = ', '.join(desc_result['feature_group'].tolist()) if not desc_result.empty else 'Unknown'
            except:
                feature_types = 'Unknown'
            
            conn.close()
            
            return {
                'existing_features': existing_features,
                'description': f"Gem with {feature_types} features" if existing_features > 0 else 'New gem'
            }
            
        except Exception as e:
            print(f"Database lookup error: {e}")
            return None
    
    def plot_enhanced_spectrum(self):
        """Plot enhanced B/H light spectrum with real-time integration features"""
        if self.wavelengths is None or self.intensities is None:
            return
        
        self.ax.clear()
        
        # Enhanced B/H spectrum plotting with optimized linewidth
        self.ax.plot(self.wavelengths, self.intensities, 'r-', linewidth=0.5, 
                    label='Enhanced B/H Light Spectrum', color='darkred', alpha=0.8)
        
        # Enhanced B light specific feature colors with audio correlation
        b_light_colors = {
            'plateau': 'red',      # 800Hz
            'shoulder': 'orange',  # 700Hz
            'broad_peak': 'blue',  # 900Hz
            'valley': 'purple',    # 600Hz
            'absorption_edge': 'brown',   # 1000Hz
            'doublet': 'pink',     # 950Hz
            'baseline_shift': 'gray',     # 500Hz
            'broad_absorption': 'darkred' # 850Hz
        }
        
        # Plot enhanced B light features with audio feedback indicators
        for i, feature in enumerate(self.manual_features):
            wavelength = feature['wavelength']
            feature_type = feature['original_type']
            width = feature.get('context', {}).get('width', 0)
            audio_freq = self.feature_frequencies.get(feature_type, 800)
            
            color = b_light_colors.get(feature_type, 'black')
            
            # Enhanced vertical line at feature position
            self.ax.axvline(wavelength, color=color, linewidth=1.0, alpha=0.8)
            
            # Find intensity at this wavelength
            idx = np.argmin(np.abs(self.wavelengths - wavelength))
            intensity = self.intensities[idx]
            
            # Enhanced annotation with audio frequency
            label = f"{feature_type}\nW:{width:.0f}nm\n{audio_freq}Hz"
                
            self.ax.annotate(label, (wavelength, intensity), 
                           rotation=90, fontsize=8, ha='center', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # Enhanced axis labels and title
        self.ax.set_xlabel('Wavelength (nm)')
        self.ax.set_ylabel('Transmission (%)')
        self.ax.grid(True, alpha=0.3)
        
        enhanced_title = f'Enhanced B/H Light: {self.current_gem_id or "Unknown"} - '
        enhanced_title += f'Audio: {"ON" if self.bleep_enabled else "OFF"}, '
        enhanced_title += f'Features: {len(self.manual_features)}'
        self.ax.set_title(enhanced_title)
        
        # Enhanced legend with audio frequencies
        if self.manual_features:
            legend_elements = []
            used_types = set(f['original_type'] for f in self.manual_features)
            for ftype in used_types:
                color = b_light_colors.get(ftype, 'black')
                freq = self.feature_frequencies.get(ftype, 800)
                legend_elements.append(plt.Line2D([0], [0], color=color, linewidth=1.0, 
                                                 label=f'{ftype.title()} ({freq}Hz)'))
            
            self.ax.legend(handles=legend_elements, title='Enhanced B Light Features (Audio)', 
                          loc='upper right', fontsize=8)
        
        self.canvas.draw()
    
    def on_enhanced_plot_click(self, event):
        """Handle enhanced B/H light plot click with audio feedback and real-time analysis"""
        if event.inaxes != self.ax or not event.xdata:
            return
        
        wavelength = event.xdata
        
        # Get enhanced feature type from combobox
        selected = self.feature_type_var.get()
        if not selected:
            messagebox.showwarning("No Enhanced Feature Selected", 
                                 "Please select a B light feature type first")
            if self.bleep_enabled:
                self.play_bleep('error')
            return
        
        # Extract feature type
        feature_type = selected.split(' - ')[0]
        
        # Enhanced B light specific context calculation
        context = self.calculate_enhanced_b_light_context(wavelength)
        
        # Enhanced plateau vs shoulder disambiguation with audio cues
        if feature_type.lower() in ['plateau', 'shoulder', 'broad_absorption']:
            disambiguation = self.enhanced_b_light_disambiguation(feature_type, context)
            if disambiguation != feature_type:
                print(f"Enhanced disambiguation: {feature_type} → {disambiguation}")
                feature_type = disambiguation
                # Play disambiguation audio cue
                if self.bleep_enabled:
                    self.play_bleep(feature_type)
        
        # Calculate enhanced relative height if enabled
        relative_height = None
        if self.relative_height_enabled:
            relative_height = self.calculate_enhanced_relative_height(wavelength)
        
        # Create enhanced feature entry
        feature = {
            'wavelength': wavelength,
            'original_type': feature_type,
            'standardized_type': feature_type,
            'confidence': context.get('confidence', 0.8),
            'context': context,
            'relative_height': relative_height,
            'audio_frequency': self.feature_frequencies.get(feature_type, 800),
            'analysis_session': self.analysis_session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        self.manual_features.append(feature)
        self.feature_count += 1
        
        # Enhanced audio feedback
        if self.bleep_enabled:
            self.play_bleep(feature_type)
        
        # Enhanced real-time database import
        if self.auto_import_enabled:
            db_status = self.import_feature_to_database(feature)
            feature['db_status'] = db_status
        else:
            feature['db_status'] = 'pending'
        
        # Update enhanced displays
        self.update_enhanced_feature_list()
        self.plot_enhanced_spectrum()
        
        # Enhanced status update
        width = context.get('width', 0)
        rel_str = f", Rel.H: {relative_height:.3f}" if relative_height else ""
        freq = self.feature_frequencies.get(feature_type, 800)
        
        status_msg = f"Enhanced B: {feature_type} at {wavelength:.1f}nm (W:{width:.0f}nm, {freq}Hz{rel_str})"
        self.update_enhanced_status(status_msg)
        
        print(f"Enhanced B light feature marked: {feature_type} at {wavelength:.1f}nm")
    
    def calculate_enhanced_b_light_context(self, wavelength):
        """Calculate enhanced B/H light specific context for feature analysis"""
        if self.wavelengths is None or self.intensities is None:
            return {'confidence': 0.5, 'width': 0}
        
        # Enhanced analysis window for B light (broader than L light)
        window_size = 50  # Larger for B light broad features
        mask = np.abs(self.wavelengths - wavelength) <= window_size
        
        if not np.any(mask):
            return {'confidence': 0.5, 'width': 0}
        
        window_wavelengths = self.wavelengths[mask]
        window_intensities = self.intensities[mask]
        
        # Enhanced B light specific metrics
        width = len(window_wavelengths) * np.mean(np.diff(self.wavelengths)) if len(window_wavelengths) > 1 else 0
        intensity_std = np.std(window_intensities)
        
        # Enhanced asymmetry calculation (critical for B light features)
        if len(window_intensities) >= 6:
            mid_point = len(window_intensities) // 2
            left_half = window_intensities[:mid_point]
            right_half = window_intensities[mid_point:]
            asymmetry = abs(np.mean(left_half) - np.mean(right_half)) / np.mean(window_intensities)
        else:
            asymmetry = 0
        
        # Enhanced slope analysis for B light plateau detection
        if len(window_intensities) >= 4:
            slopes = np.gradient(window_intensities, window_wavelengths)
            slope_variation = np.std(slopes)
            mean_abs_slope = np.mean(np.abs(slopes))
        else:
            slope_variation = 0
            mean_abs_slope = 0
        
        # Enhanced peak attachment check for B light shoulders
        attached_to_main_peak = self.check_enhanced_b_light_peak_attachment(wavelength)
        
        # Enhanced flatness metric for B light plateaus
        if len(window_intensities) >= 5:
            # Calculate local flatness
            local_range = np.max(window_intensities) - np.min(window_intensities)
            mean_intensity = np.mean(window_intensities)
            flatness_ratio = local_range / mean_intensity if mean_intensity > 0 else 1.0
        else:
            flatness_ratio = 1.0
        
        context = {
            'width': width,
            'asymmetry': asymmetry,
            'intensity_std': intensity_std,
            'slope_variation': slope_variation,
            'mean_abs_slope': mean_abs_slope,
            'attached_to_main_peak': attached_to_main_peak,
            'flatness_ratio': flatness_ratio,
            'light_source': 'B',
            'confidence': 0.8,
            'window_points': len(window_intensities)
        }
        
        # Enhanced B light confidence adjustment
        if width > 60 and asymmetry < 0.15 and flatness_ratio < 0.2:  # Strong plateau indicators
            context['confidence'] = 0.95
        elif attached_to_main_peak and asymmetry > 0.4 and width < 25:  # Strong shoulder indicators
            context['confidence'] = 0.92
        elif width > 30 and slope_variation < 0.1:  # Moderate broad feature
            context['confidence'] = 0.85
        
        return context
    
    def enhanced_b_light_disambiguation(self, original_type, context):
        """Enhanced B light specific disambiguation with audio cues"""
        width = context.get('width', 0)
        asymmetry = context.get('asymmetry', 0)
        flatness = context.get('flatness_ratio', 1.0)
        attached = context.get('attached_to_main_peak', False)
        
        # Enhanced disambiguation rules for B light
        if original_type.lower() == 'plateau':
            if width < 30:
                return 'shoulder'  # Too narrow for plateau
            elif flatness > 0.5:
                return 'broad_peak'  # Too much variation for plateau
            else:
                return 'plateau'  # Confirmed plateau
        
        elif original_type.lower() == 'shoulder':
            if width > 50 and flatness < 0.3:
                return 'plateau'  # Too broad and flat for shoulder
            elif not attached and asymmetry < 0.2:
                return 'broad_peak'  # Not attached, too symmetric
            else:
                return 'shoulder'  # Confirmed shoulder
        
        elif original_type.lower() == 'broad_absorption':
            if width > 80 and flatness < 0.2:
                return 'plateau'
            elif width < 20:
                return 'valley'
            else:
                return 'broad_absorption'
        
        return original_type
    
    def check_enhanced_b_light_peak_attachment(self, wavelength):
        """Enhanced check for B light feature attachment to main peaks"""
        if self.wavelengths is None or self.intensities is None:
            return False
        
        try:
            from scipy import signal
            # Invert for peak finding in transmission spectra
            inverted = -self.intensities
            peaks, properties = signal.find_peaks(inverted, height=0, distance=20, prominence=0.1)
            peak_wavelengths = self.wavelengths[peaks]
            
            # Enhanced attachment check (broader tolerance for B light)
            attachment_distance = 40  # nm tolerance for B light
            is_attached = np.any(np.abs(peak_wavelengths - wavelength) <= attachment_distance)
            
            if is_attached:
                closest_peak_distance = np.min(np.abs(peak_wavelengths - wavelength))
                return closest_peak_distance <= attachment_distance
            
            return False
        except:
            return False
    
    def calculate_enhanced_relative_height(self, wavelength):
        """Calculate enhanced relative height with database lookup"""
        if not self.current_gem_id or not os.path.exists(DATABASE_PATH):
            return None
        
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            
            # Look for features near this wavelength across all light sources
            query = """
                SELECT light_source, wavelength, intensity
                FROM structural_features 
                WHERE gem_id LIKE ? AND ABS(wavelength - ?) <= 15.0
                ORDER BY ABS(wavelength - ?)
            """
            
            result = pd.read_sql_query(query, conn, params=[f"%{self.current_gem_id}%", wavelength, wavelength])
            conn.close()
            
            if result.empty:
                return None
            
            # Calculate relative height
            max_intensity = result['intensity'].max()
            current_intensity = self.intensities[np.argmin(np.abs(self.wavelengths - wavelength))]
            
            if max_intensity > 0:
                relative_height = current_intensity / max_intensity
                print(f"  Enhanced relative height: {relative_height:.3f} vs {len(result)} features")
                return relative_height
            
            return None
            
        except Exception as e:
            print(f"Enhanced relative height calculation error: {e}")
            return None
    
    def import_feature_to_database(self, feature):
        """Enhanced real-time feature import to database"""
        if not os.path.exists(DATABASE_PATH):
            self.create_database_schema()
        
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Enhanced database insertion
            cursor.execute("""
                INSERT OR IGNORE INTO structural_features 
                (feature, file, light_source, wavelength, intensity, point_type, 
                 feature_group, processing, width_nm, height, normalization_scheme,
                 reference_wavelength, intensity_range_min, intensity_range_max,
                 data_type, gem_id, analysis_session_id, file_source, 
                 normalization_compatible, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feature['original_type'],
                self.current_gem_id or 'unknown',
                'Halogen',
                feature['wavelength'],
                self.intensities[np.argmin(np.abs(self.wavelengths - feature['wavelength']))],
                'Manual_Enhanced_B',
                feature['original_type'],
                'Enhanced_Manual_B_Light_Analysis',
                feature.get('context', {}).get('width', 0),
                feature.get('relative_height', 0),
                'Halogen_650nm_50000_to_100',
                650.0,
                0.0,
                100.0,
                'manual_enhanced_b_light',
                self.current_gem_id,
                feature['analysis_session'],
                'Enhanced_B_Light_Analyzer',
                True,
                feature.get('confidence', 0.8)
            ))
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            if success:
                print(f"  Enhanced database import: SUCCESS")
                return "imported"
            else:
                print(f"  Enhanced database import: DUPLICATE")
                return "duplicate"
                
        except Exception as e:
            print(f"  Enhanced database import error: {e}")
            return "error"
    
    def create_database_schema(self):
        """Create enhanced database schema if needed"""
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS structural_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature TEXT NOT NULL,
                    file TEXT NOT NULL,
                    light_source TEXT NOT NULL,
                    wavelength REAL NOT NULL,
                    intensity REAL NOT NULL,
                    point_type TEXT NOT NULL,
                    feature_group TEXT NOT NULL,
                    processing TEXT,
                    width_nm REAL,
                    height REAL,
                    normalization_scheme TEXT,
                    reference_wavelength REAL,
                    intensity_range_min REAL,
                    intensity_range_max REAL,
                    data_type TEXT,
                    gem_id TEXT,
                    analysis_session_id TEXT,
                    file_source TEXT,
                    normalization_compatible BOOLEAN DEFAULT 1,
                    quality_score REAL,
                    timestamp TEXT DEFAULT (datetime('now')),
                    UNIQUE(file, feature, wavelength, point_type)
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Enhanced database schema creation error: {e}")
    
    def update_enhanced_feature_list(self):
        """Update enhanced feature list display with integration data"""
        # Clear existing items
        for item in self.feature_tree.get_children():
            self.feature_tree.delete(item)
        
        # Add enhanced B light features
        for feature in self.manual_features:
            wavelength = f"{feature['wavelength']:.1f}"
            feature_type = feature['original_type']
            width = f"{feature.get('context', {}).get('width', 0):.0f}"
            audio_freq = str(feature.get('audio_frequency', 800))
            
            relative_height = feature.get('relative_height', 0)
            rel_height_str = f"{relative_height:.3f}" if relative_height else "N/A"
            
            db_status = feature.get('db_status', 'pending')
            quality = f"{feature.get('confidence', 0.8):.2f}"
            
            self.feature_tree.insert('', 'end', values=(
                wavelength, feature_type, width, audio_freq, rel_height_str, db_status, quality
            ))
    
    def save_enhanced_features(self):
        """Save enhanced B light features with complete integration"""
        if not self.manual_features:
            messagebox.showwarning("No Enhanced Features", "No B light features to save")
            if self.bleep_enabled:
                self.play_bleep('error')
            return
        
        # Enhanced filename generation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = self.current_gem_id or 'unknown'
        filename = f"{base_name}_enhanced_B_features_{timestamp}.csv"
        
        # Create enhanced output directory
        os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
        full_path = os.path.join(OUTPUT_DIRECTORY, filename)
        
        try:
            # Enhanced feature data preparation
            enhanced_data = []
            for feature in self.manual_features:
                enhanced_feature = {
                    'Feature': feature['original_type'],
                    'File': self.current_gem_id,
                    'Light_Source': 'Halogen',
                    'Wavelength': feature['wavelength'],
                    'Intensity': self.intensities[np.argmin(np.abs(self.wavelengths - feature['wavelength']))],
                    'Point_Type': 'Enhanced_Manual_B',
                    'Feature_Group': feature['original_type'],
                    'Width_nm': feature.get('context', {}).get('width', 0),
                    'Confidence': feature.get('confidence', 0.8),
                    'Audio_Frequency': feature.get('audio_frequency', 800),
                    'Relative_Height': feature.get('relative_height', 0),
                    'Analysis_Session': feature.get('analysis_session', ''),
                    'Timestamp': feature.get('timestamp', ''),
                    'Enhanced_Analysis': True,
                    'Normalization_Scheme': 'Halogen_650nm_50000_to_100',
                    'Reference_Wavelength': 650.0,
                    'Intensity_Range_Min': 0.0,
                    'Intensity_Range_Max': 100.0,
                    'DB_Status': feature.get('db_status', 'pending')
                }
                enhanced_data.append(enhanced_feature)
            
            # Save enhanced CSV
            df = pd.DataFrame(enhanced_data)
            df.to_csv(full_path, index=False)
            
            # Enhanced completion feedback
            print(f"Enhanced B light features saved: {filename}")
            print(f"  Features: {len(self.manual_features)}")
            print(f"  Audio feedback: {'Used' if self.bleep_enabled else 'Disabled'}")
            print(f"  Database integration: {'Active' if self.auto_import_enabled else 'Inactive'}")
            print(f"  Relative heights: {'Calculated' if self.relative_height_enabled else 'Disabled'}")
            
            self.update_enhanced_status(f"Saved {len(self.manual_features)} enhanced B features to {filename}")
            
            # Enhanced audio completion
            if self.bleep_enabled:
                self.play_bleep('completion')
            
            # Enhanced auto-import to database
            if self.auto_import_enabled:
                self.manual_database_import(full_path)
            
            messagebox.showinfo("Enhanced Save Complete", 
                              f"Enhanced B light features saved successfully!\n\n"
                              f"File: {filename}\n"
                              f"Features: {len(self.manual_features)}\n"
                              f"Audio feedback: {'Active' if self.bleep_enabled else 'Inactive'}\n"
                              f"Database: {'Auto-imported' if self.auto_import_enabled else 'Manual import needed'}")
            
        except Exception as e:
            messagebox.showerror("Enhanced Save Error", f"Error saving enhanced B features: {e}")
            if self.bleep_enabled:
                self.play_bleep('error')
    
    def manual_database_import(self, csv_path=None):
        """Manual database import with enhanced feedback"""
        if csv_path is None:
            # Use last saved file or prompt
            if not self.manual_features:
                messagebox.showinfo("No Features", "No features to import")
                return
            # Save first, then import
            self.save_enhanced_features()
            return
        
        print(f"Enhanced database import starting: {os.path.basename(csv_path)}")
        
        # Import using the integrated database system
        try:
            from integrated_database_system import IntegratedDatabaseManager
            db_manager = IntegratedDatabaseManager()
            
            batch_id = f"manual_b_light_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            record_count, success = db_manager.import_csv_file_enhanced(csv_path, batch_id)
            
            if success and record_count > 0:
                print(f"Enhanced database import: {record_count} records imported")
                if self.bleep_enabled:
                    self.play_bleep('completion')
                messagebox.showinfo("Enhanced Import Success", 
                                  f"Successfully imported {record_count} B light features to database")
            else:
                print("Enhanced database import: No records imported")
                if self.bleep_enabled:
                    self.play_bleep('error')
                messagebox.showwarning("Enhanced Import Warning", "No records were imported to database")
        
        except Exception as e:
            print(f"Enhanced database import error: {e}")
            if self.bleep_enabled:
                self.play_bleep('error')
            messagebox.showerror("Enhanced Import Error", f"Database import failed: {e}")
    
    def show_enhanced_context_menu(self, event):
        """Show enhanced context menu for feature list"""
        selection = self.feature_tree.selection()
        if not selection:
            return
        
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="Play Audio Cue", 
                               command=lambda: self.play_feature_audio_cue(selection[0]))
        context_menu.add_command(label="Recalculate Relative Height", 
                               command=lambda: self.recalculate_relative_height(selection[0]))
        context_menu.add_command(label="Re-import to Database", 
                               command=lambda: self.reimport_feature_to_database(selection[0]))
        context_menu.add_separator()
        context_menu.add_command(label="Edit Feature", 
                               command=lambda: self.edit_enhanced_feature(None))
        
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()
    
    def play_feature_audio_cue(self, item):
        """Play audio cue for selected feature"""
        values = self.feature_tree.item(item)['values']
        if values and len(values) >= 4:
            audio_freq = int(values[3])  # Audio frequency column
            feature_type = values[1]     # Feature type column
            
            print(f"Playing audio cue for {feature_type}: {audio_freq}Hz")
            
            if self.bleep_enabled:
                # Temporarily override frequency
                original_freq = self.feature_frequencies.get(feature_type.lower(), 800)
                self.feature_frequencies[feature_type.lower()] = audio_freq
                self.play_bleep(feature_type.lower())
                self.feature_frequencies[feature_type.lower()] = original_freq
    
    def recalculate_relative_height(self, item):
        """Recalculate relative height for selected feature"""
        values = self.feature_tree.item(item)['values']
        if values and len(values) >= 1:
            wavelength = float(values[0])
            relative_height = self.calculate_enhanced_relative_height(wavelength)
            
            if relative_height is not None:
                print(f"Recalculated relative height for {wavelength}nm: {relative_height:.3f}")
                if self.bleep_enabled:
                    self.play_bleep('validation')
            else:
                print(f"Could not recalculate relative height for {wavelength}nm")
                if self.bleep_enabled:
                    self.play_bleep('error')
    
    def edit_enhanced_feature(self, event):
        """Edit enhanced B light feature with validation"""
        selection = self.feature_tree.selection()
        if not selection:
            return
        
        item = self.feature_tree.item(selection[0])
        values = item['values']
        
        # Find corresponding feature
        wavelength = float(values[0])
        for i, feature in enumerate(self.manual_features):
            if abs(feature['wavelength'] - wavelength) < 0.1:
                # Enhanced edit dialog
                new_type = simpledialog.askstring(
                    "Edit Enhanced B Feature", 
                    f"Edit B light feature type at {wavelength}nm:\n"
                    f"Current: {feature['original_type']}\n"
                    f"Audio frequency: {feature.get('audio_frequency', 800)}Hz",
                    initialvalue=feature['original_type']
                )
                
                if new_type and new_type != feature['original_type']:
                    # Update feature with enhanced data
                    old_type = feature['original_type']
                    feature['original_type'] = new_type
                    feature['standardized_type'] = new_type
                    feature['audio_frequency'] = self.feature_frequencies.get(new_type, 800)
                    
                    # Enhanced audio feedback for change
                    if self.bleep_enabled:
                        self.play_bleep(new_type)
                    
                    # Update displays
                    self.update_enhanced_feature_list()
                    self.plot_enhanced_spectrum()
                    
                    print(f"Enhanced B light feature updated: {old_type} → {new_type} at {wavelength:.1f}nm")
                    self.update_enhanced_status(f"Updated: {old_type} → {new_type}")
                break
    
    def update_enhanced_status(self, message):
        """Update enhanced status bar with integration info"""
        enhanced_message = f"[Enhanced B/H] {message}"
        self.status_var.set(enhanced_message)
    
    def run(self):
        """Start the enhanced B/H light analyzer with complete integration"""
        print("Starting Enhanced B/H Light Analyzer with complete integration...")
        print("  Audio feedback: Available" if HAS_AUDIO else "  Audio feedback: Not available")
        print(f"  Database integration: {DATABASE_PATH}")
        print(f"  Output directory: {OUTPUT_DIRECTORY}")
        
        if self.bleep_enabled and HAS_AUDIO:
            self.play_bleep('completion')
        
        self.root.mainloop()

def main():
    """Main function for enhanced B/H light analyzer"""
    print("Enhanced B/H Light Manual Analyzer - Complete Integration")
    print("Features: Audio feedback, database integration, real-time analysis")
    print("=" * 60)
    
    try:
        analyzer = EnhancedHalogenAnalyzer()
        analyzer.run()
    except Exception as e:
        print(f"Enhanced analyzer error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
