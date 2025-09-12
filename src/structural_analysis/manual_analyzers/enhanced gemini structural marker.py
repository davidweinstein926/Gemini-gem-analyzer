#!/usr/bin/env python3
"""
ENHANCED GEMINI STRUCTURAL MARKER - COMPLETE INTEGRATION
Enhanced with: Audio bleep feedback, automatic database import, relative height measurements,
enhanced normalization metadata capture, and streamlined workflow automation.

Major Enhancements:
- Audio bleep feedback for each feature marked
- Automatic CSV-to-database import after analysis
- Real-time relative height calculations across light sources
- Enhanced metadata capture (baseline, slope, normalization info)
- Direct integration with main analysis system
- Progressive feature detection with audio cues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys
from datetime import datetime
from pathlib import Path
import sqlite3
from scipy import stats
import time

# Audio support for bleep functionality
try:
    import winsound  # Windows audio support
    HAS_AUDIO = True
except ImportError:
    try:
        import pygame
        pygame.mixer.init()
        HAS_AUDIO = True
    except ImportError:
        HAS_AUDIO = False

# ENHANCED: Output directory for CSV files with auto-import capability
OUTPUT_DIRECTORY = r"c:\users\david\gemini sp10 structural data"
DATABASE_PATH = "multi_structural_gem_data.db"

# Enhanced global variables with audio and database integration
features = []
current_type = None
persistent_mode = True
clicks = []
filename = ""
lines_drawn = []
texts_drawn = []
magnify_mode = False
baseline_data = None
spectrum_df = None
ax = None

# Audio and database integration globals
bleep_enabled = True
auto_import_enabled = True
current_gem_id = None
light_source_detected = None

# ENHANCED: Structure types with complete metadata capture
STRUCTURE_TYPES = {
    'Plateau': ['Start', 'Midpoint', 'End'],
    'Mound': ['Start', 'Crest', 'End'],
    'Trough': ['Start', 'Bottom', 'End'],
    'Valley': ['Midpoint'],
    'Peak': ['Max'],
    'Diagnostic Region': ['Start', 'End'],
    'Baseline Region': ['Start', 'End']
}

def play_bleep(frequency=800, duration=200, feature_type="standard"):
    """Play audio bleep for feature detection with different tones"""
    if not bleep_enabled or not HAS_AUDIO:
        return
    
    try:
        # Different frequencies for different feature types
        freq_map = {
            "peak": 1000,
            "valley": 600,
            "plateau": 800,
            "mound": 900,
            "trough": 500,
            "diagnostic": 1100,
            "baseline": 400,
            "completion": 1200,
            "error": 300
        }
        
        freq = freq_map.get(feature_type, frequency)
        
        if 'winsound' in sys.modules:
            winsound.Beep(freq, duration)
        elif 'pygame' in sys.modules:
            # Generate tone using pygame
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
        print(f"Audio bleep error: {e}")

def detect_light_source_from_filename(filename):
    """Detect light source from filename for auto-categorization"""
    global light_source_detected
    
    filename_upper = filename.upper()
    
    if any(pattern in filename_upper for pattern in ['_B', 'HALOGEN', '_H', 'BC1', 'BP1']):
        light_source_detected = 'Halogen'
        return 'Halogen'
    elif any(pattern in filename_upper for pattern in ['_L', 'LASER', 'LC1', 'LP1']):
        light_source_detected = 'Laser' 
        return 'Laser'
    elif any(pattern in filename_upper for pattern in ['_U', '_UV', 'UC1', 'UP1']):
        light_source_detected = 'UV'
        return 'UV'
    else:
        light_source_detected = 'Unknown'
        return 'Unknown'

def extract_gem_id_from_filename(filename):
    """Extract gem ID from filename for database integration"""
    global current_gem_id
    
    base_name = Path(filename).stem
    
    # Remove common light source suffixes to get core gem ID
    for suffix in ['B', 'L', 'U', 'BC1', 'LC1', 'UC1', 'BP1', 'LP1', 'UP1']:
        if base_name.upper().endswith(suffix):
            current_gem_id = base_name[:-len(suffix)]
            return current_gem_id
    
    # Fallback: use full base name
    current_gem_id = base_name
    return current_gem_id

def auto_import_to_database(csv_file_path):
    """Automatically import CSV to database after analysis"""
    if not auto_import_enabled:
        print("   Auto-import disabled")
        return False
    
    try:
        print(f"   🔄 Auto-importing to database...")
        
        # Create database if it doesn't exist
        if not os.path.exists(DATABASE_PATH):
            create_database_schema()
        
        # Read the CSV we just created
        df = pd.read_csv(csv_file_path)
        
        if df.empty:
            print("   ⚠️ CSV file is empty - nothing to import")
            return False
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        imported_count = 0
        base_filename = os.path.splitext(os.path.basename(csv_file_path))[0]
        
        # Import each row with enhanced metadata
        for _, row in df.iterrows():
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO structural_features 
                    (feature, file, light_source, wavelength, intensity, point_type, 
                     feature_group, processing, baseline_used, norm_factor, snr,
                     symmetry_ratio, skew_description, width_nm, height, 
                     local_slope, slope_r_squared, normalization_scheme, 
                     reference_wavelength, intensity_range_min, intensity_range_max,
                     file_source, data_type, normalization_compatible)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.get('Feature', ''),
                    current_gem_id or base_filename,
                    light_source_detected or 'Unknown',
                    float(row.get('Wavelength', 0)) if 'Wavelength' in row else float(row.get('Start', 0)),
                    float(row.get('Intensity', 0)) if 'Intensity' in row else 1.0,
                    'Manual_Enhanced',
                    row.get('Feature', 'Unknown'),
                    'Enhanced_Manual_Analysis',
                    safe_float(row.get('Baseline_Used')),
                    safe_float(row.get('Norm_Factor')),
                    safe_float(row.get('SNR')),
                    safe_float(row.get('Symmetry_Ratio')),
                    row.get('Skew_Description', ''),
                    safe_float(row.get('Width_nm')),
                    safe_float(row.get('Height')),
                    safe_float(row.get('Local_Slope')),
                    safe_float(row.get('Slope_R_Squared')),
                    determine_normalization_scheme(light_source_detected),
                    get_reference_wavelength(light_source_detected),
                    0.0,  # intensity_range_min
                    100.0,  # intensity_range_max (enhanced 0-100 scale)
                    csv_file_path,
                    'manual_enhanced_structural',
                    True  # normalization_compatible
                ))
                
                if cursor.rowcount > 0:
                    imported_count += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error importing row: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        if imported_count > 0:
            print(f"   ✅ Auto-imported {imported_count} features to database")
            play_bleep(feature_type="completion")
            return True
        else:
            print("   ❌ No features imported to database")
            return False
            
    except Exception as e:
        print(f"   ❌ Auto-import error: {e}")
        play_bleep(feature_type="error")
        return False

def create_database_schema():
    """Create database schema for structural features"""
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
                
                -- Enhanced analysis fields
                processing TEXT,
                baseline_used REAL,
                norm_factor REAL,
                snr REAL,
                symmetry_ratio REAL,
                skew_description TEXT,
                width_nm REAL,
                height REAL,
                local_slope REAL,
                slope_r_squared REAL,
                
                -- Enhanced normalization metadata
                normalization_scheme TEXT,
                reference_wavelength REAL,
                intensity_range_min REAL,
                intensity_range_max REAL,
                normalization_compatible BOOLEAN DEFAULT 1,
                
                -- Metadata
                timestamp TEXT DEFAULT (datetime('now')),
                file_source TEXT,
                data_type TEXT,
                
                UNIQUE(file, feature, wavelength, point_type)
            )
        """)
        
        # Create indexes for performance
        indexes = [
            ("idx_file", "file"),
            ("idx_light_source", "light_source"),
            ("idx_wavelength", "wavelength"),
            ("idx_feature_group", "feature_group")
        ]
        
        for idx_name, columns in indexes:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON structural_features({columns})")
        
        conn.commit()
        conn.close()
        print("   ✅ Database schema created")
        
    except Exception as e:
        print(f"   ❌ Database schema creation error: {e}")

def determine_normalization_scheme(light_source):
    """Determine normalization scheme based on light source"""
    schemes = {
        'Halogen': 'Halogen_650nm_50000_to_100',
        'Laser': 'Laser_Max_50000_to_100',
        'UV': 'UV_811nm_15000_to_100'
    }
    return schemes.get(light_source, 'Unknown_to_100')

def get_reference_wavelength(light_source):
    """Get reference wavelength for normalization"""
    ref_wavelengths = {
        'Halogen': 650.0,
        'Laser': None,  # Uses maximum
        'UV': 811.0
    }
    return ref_wavelengths.get(light_source)

def safe_float(value):
    """Safely convert value to float or return None"""
    if pd.isna(value) or value == '' or value is None:
        return None
    try:
        return float(value)
    except:
        return None

def calculate_relative_height_realtime(wavelength, intensity):
    """Calculate relative height measurements in real-time during analysis"""
    if not current_gem_id:
        return None
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Query for similar wavelengths across light sources for this gem
        query = """
            SELECT light_source, wavelength, intensity, feature_group
            FROM structural_features 
            WHERE file LIKE ? AND ABS(wavelength - ?) <= 10.0
            ORDER BY ABS(wavelength - ?)
        """
        
        df = pd.read_sql_query(query, conn, params=[f"%{current_gem_id}%", wavelength, wavelength])
        conn.close()
        
        if not df.empty:
            # Calculate relative heights
            max_intensity = df['intensity'].max()
            relative_height = intensity / max_intensity if max_intensity > 0 else 0
            
            print(f"      📏 Relative height: {relative_height:.3f} ({relative_height*100:.1f}%)")
            print(f"      📊 Compared to {len(df)} similar features")
            
            return {
                'relative_height': relative_height,
                'comparison_count': len(df),
                'max_intensity_found': max_intensity
            }
        
        return None
        
    except Exception as e:
        print(f"      ⚠️ Relative height calculation error: {e}")
        return None

def calculate_baseline_stats(df, start_wl, end_wl):
    """Calculate enhanced baseline statistics for a wavelength region"""
    try:
        # Filter data to baseline region
        wavelengths = df.iloc[:, 0]
        intensities = df.iloc[:, 1]
        
        mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
        baseline_intensities = intensities[mask]
        baseline_wavelengths = wavelengths[mask]
        
        if len(baseline_intensities) < 3:
            return None
        
        # Enhanced statistics
        avg_intensity = np.mean(baseline_intensities)
        std_dev = np.std(baseline_intensities)
        snr = avg_intensity / std_dev if std_dev > 0 else float('inf')
        
        # Enhanced slope analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(baseline_wavelengths, baseline_intensities)
        
        baseline_stats = {
            'wavelength_start': start_wl,
            'wavelength_end': end_wl,
            'avg_intensity': round(avg_intensity, 2),
            'std_deviation': round(std_dev, 3),
            'snr': round(snr, 1),
            'slope': round(slope, 6),
            'r_squared': round(r_value**2, 4),
            'data_points': len(baseline_intensities),
            'width_nm': end_wl - start_wl
        }
        
        print(f"📊 ENHANCED BASELINE ANALYSIS:")
        print(f"   Region: {start_wl:.1f} - {end_wl:.1f} nm (Width: {end_wl-start_wl:.1f}nm)")
        print(f"   Average Intensity: {avg_intensity:.2f}")
        print(f"   Signal-to-Noise: {snr:.1f}")
        print(f"   Slope Quality: {slope:.6f} (R²: {r_value**2:.4f})")
        
        # Audio feedback for baseline quality
        if snr > 50:
            play_bleep(feature_type="baseline", frequency=400)
        elif snr > 20:
            play_bleep(feature_type="baseline", frequency=350)
        else:
            play_bleep(feature_type="error", frequency=300)
        
        return baseline_stats
        
    except Exception as e:
        print(f"❌ Enhanced baseline calculation error: {e}")
        return None

def get_enhanced_intensity_at_wavelength(df, target_wavelength):
    """Get intensity at specific wavelength with enhanced interpolation"""
    try:
        wavelengths = df.iloc[:, 0]
        intensities = df.iloc[:, 1]
        
        if target_wavelength in wavelengths.values:
            # Exact match
            idx = wavelengths[wavelengths == target_wavelength].index[0]
            return intensities.iloc[idx]
        else:
            # Enhanced interpolation
            intensity = np.interp(target_wavelength, wavelengths, intensities)
            return intensity
            
    except Exception as e:
        print(f"❌ Enhanced intensity lookup error: {e}")
        return None

def calculate_enhanced_local_slope(df, wavelength, window=3.0):
    """Calculate enhanced local slope around a wavelength point"""
    try:
        wavelengths = df.iloc[:, 0]
        intensities = df.iloc[:, 1]
        
        # Enhanced window around the point
        mask = (wavelengths >= wavelength - window) & (wavelengths <= wavelength + window)
        local_wl = wavelengths[mask]
        local_int = intensities[mask]
        
        if len(local_wl) < 3:
            return None
        
        # Enhanced slope calculation with multiple metrics
        slope, intercept, r_value, p_value, std_err = stats.linregress(local_wl, local_int)
        
        # Calculate slope variation (curvature indicator)
        slopes = np.gradient(local_int.values, local_wl.values)
        slope_variation = np.std(slopes)
        
        return {
            'slope': round(slope, 4),
            'r_squared': round(r_value**2, 4),
            'std_error': round(std_err, 4),
            'slope_variation': round(slope_variation, 4),
            'window_size': window,
            'data_points': len(local_wl),
            'curvature_indicator': 'High' if slope_variation > 1.0 else 'Low'
        }
        
    except Exception as e:
        print(f"❌ Enhanced slope calculation error: {e}")
        return None

def calculate_enhanced_baseline_corrected_intensity(raw_intensity, baseline_stats):
    """Calculate enhanced baseline-corrected intensity with quality metrics"""
    if baseline_stats is None:
        return {
            'corrected_intensity': raw_intensity,
            'correction_applied': False,
            'baseline_quality': 'No baseline'
        }
    
    baseline_avg = baseline_stats.get('avg_intensity', 0)
    baseline_snr = baseline_stats.get('snr', 0)
    corrected = raw_intensity - baseline_avg
    
    # Quality assessment
    if baseline_snr > 50:
        quality = 'Excellent'
    elif baseline_snr > 20:
        quality = 'Good'
    elif baseline_snr > 10:
        quality = 'Fair'
    else:
        quality = 'Poor'
    
    return {
        'corrected_intensity': max(0, corrected),
        'correction_applied': True,
        'baseline_level': baseline_avg,
        'baseline_quality': quality,
        'snr': baseline_snr
    }

def calculate_enhanced_feature_width(clicks, feature_type):
    """Calculate enhanced feature width with type-specific metrics"""
    if len(clicks) < 2:
        return None
    
    wavelengths = [click[0] for click in clicks]
    min_wl = min(wavelengths)
    max_wl = max(wavelengths)
    width = max_wl - min_wl
    
    # Type-specific width analysis
    width_analysis = {
        'width_nm': round(width, 2),
        'type_classification': None,
        'width_quality': None
    }
    
    if feature_type in ['Plateau', 'Diagnostic Region']:
        if width > 50:
            width_analysis['type_classification'] = 'Broad plateau'
            width_analysis['width_quality'] = 'Excellent for plateau'
        elif width > 20:
            width_analysis['type_classification'] = 'Moderate plateau'
            width_analysis['width_quality'] = 'Good for plateau'
        else:
            width_analysis['type_classification'] = 'Narrow feature'
            width_analysis['width_quality'] = 'May be shoulder, not plateau'
    
    elif feature_type in ['Peak', 'Valley']:
        if width < 5:
            width_analysis['type_classification'] = 'Sharp feature'
            width_analysis['width_quality'] = 'Excellent resolution'
        elif width < 15:
            width_analysis['type_classification'] = 'Moderate feature'
            width_analysis['width_quality'] = 'Good resolution'
        else:
            width_analysis['type_classification'] = 'Broad feature'
            width_analysis['width_quality'] = 'Low resolution'
    
    return width_analysis

def complete_enhanced_feature():
    """Enhanced feature completion with audio feedback and database integration"""
    global current_type, clicks, features, baseline_data, spectrum_df, persistent_mode
    
    if current_type == 'Baseline Region':
        # Enhanced baseline region handling
        if len(clicks) == 2:
            start_wl = min(clicks[0][0], clicks[1][0])
            end_wl = max(clicks[0][0], clicks[1][0])
            
            baseline_stats = calculate_baseline_stats(spectrum_df, start_wl, end_wl)
            
            if baseline_stats:
                baseline_data = baseline_stats
                
                # Enhanced baseline entry
                entry = {
                    'Feature': 'Baseline Region',
                    'File': filename,
                    'Light_Source': light_source_detected,
                    'Gem_ID': current_gem_id,
                    'Start': start_wl,
                    'End': end_wl,
                    'Width_nm': baseline_stats['width_nm'],
                    'Avg_Intensity': baseline_stats['avg_intensity'],
                    'Std_Deviation': baseline_stats['std_deviation'],
                    'SNR': baseline_stats['snr'],
                    'Slope': baseline_stats['slope'],
                    'R_Squared': baseline_stats['r_squared'],
                    'Data_Points': baseline_stats['data_points'],
                    'Processing': 'Enhanced_Baseline_Analysis',
                    'Analysis_Quality': 'Excellent' if baseline_stats['snr'] > 50 else 'Good' if baseline_stats['snr'] > 20 else 'Fair'
                }
                
                features.append(entry)
                print(f"✅ ENHANCED BASELINE REGION ESTABLISHED!")
                play_bleep(feature_type="baseline")
            else:
                print(f"❌ Failed to calculate enhanced baseline statistics")
                play_bleep(feature_type="error")
        
        clicks.clear()
        current_type = None
        return
    
    # Enhanced regular feature processing
    entry = {
        'Feature': current_type, 
        'File': filename,
        'Light_Source': light_source_detected,
        'Gem_ID': current_gem_id,
        'Processing': 'Enhanced_Manual_Analysis'
    }
    
    # Add enhanced wavelength points
    for i, label in enumerate(STRUCTURE_TYPES[current_type]):
        entry[label] = round(clicks[i][0], 2)
    
    # ENHANCED: Add comprehensive intensity and analysis data for each point
    intensity_values = []
    for i, (wavelength, raw_intensity) in enumerate(clicks):
        label = STRUCTURE_TYPES[current_type][i]
        
        # Enhanced intensity measurement
        precise_intensity = get_enhanced_intensity_at_wavelength(spectrum_df, wavelength)
        if precise_intensity is None:
            precise_intensity = raw_intensity
        
        # Enhanced baseline correction
        correction_result = calculate_enhanced_baseline_corrected_intensity(precise_intensity, baseline_data)
        
        # Enhanced slope analysis
        slope_data = calculate_enhanced_local_slope(spectrum_df, wavelength)
        
        # Enhanced real-time relative height
        relative_data = calculate_relative_height_realtime(wavelength, correction_result['corrected_intensity'])
        
        # Add enhanced data to entry
        entry[f'{label}_Intensity'] = round(precise_intensity, 3)
        entry[f'{label}_Corrected_Intensity'] = round(correction_result['corrected_intensity'], 3)
        entry[f'{label}_Baseline_Quality'] = correction_result['baseline_quality']
        
        if slope_data:
            entry[f'{label}_Slope'] = slope_data['slope']
            entry[f'{label}_Slope_R2'] = slope_data['r_squared']
            entry[f'{label}_Curvature'] = slope_data['curvature_indicator']
        
        if relative_data:
            entry[f'{label}_Relative_Height'] = round(relative_data['relative_height'], 3)
            entry[f'{label}_Comparison_Count'] = relative_data['comparison_count']
        
        intensity_values.append(correction_result['corrected_intensity'])
        
        print(f"📍 Enhanced {label}: {wavelength:.2f}nm")
        print(f"   Intensity: {precise_intensity:.3f} → {correction_result['corrected_intensity']:.3f} (corrected)")
        if slope_data:
            print(f"   Slope: {slope_data['slope']:.4f} ({slope_data['curvature_indicator']} curvature)")
        if relative_data:
            print(f"   Relative height: {relative_data['relative_height']:.3f}")
    
    # Enhanced width analysis
    width_analysis = calculate_enhanced_feature_width(clicks, current_type)
    if width_analysis:
        entry['Width_nm'] = width_analysis['width_nm']
        entry['Width_Classification'] = width_analysis['type_classification']
        entry['Width_Quality'] = width_analysis['width_quality']
    
    # Enhanced symmetry calculation for mounds
    if current_type == 'Mound' and len(clicks) == 3:
        start_wl, crest_wl, end_wl = [click[0] for click in clicks]
        start_int, crest_int, end_int = intensity_values
        
        # Enhanced symmetry with multiple metrics
        wavelength_symmetry = abs(crest_wl - start_wl) / abs(end_wl - crest_wl) if abs(end_wl - crest_wl) > 0 else float('inf')
        intensity_left = crest_int - start_int
        intensity_right = crest_int - end_int
        intensity_symmetry = intensity_left / intensity_right if intensity_right > 0 else float('inf')
        
        # Enhanced skew classification
        if 0.8 <= wavelength_symmetry <= 1.25:
            skew_desc = "Symmetric"
        elif wavelength_symmetry < 0.8:
            skew_desc = "Left Skewed (steep left side)"
        else:
            skew_desc = "Right Skewed (steep right side)"
        
        entry['Wavelength_Symmetry_Ratio'] = round(wavelength_symmetry, 3)
        entry['Intensity_Symmetry_Ratio'] = round(intensity_symmetry, 3)
        entry['Enhanced_Skew_Description'] = skew_desc
        
        print(f"📊 Enhanced Mound Analysis:")
        print(f"   Wavelength symmetry: {wavelength_symmetry:.3f}")
        print(f"   Intensity symmetry: {intensity_symmetry:.3f}")
        print(f"   Classification: {skew_desc}")
    
    # Enhanced baseline integration
    if baseline_data:
        entry['Baseline_Applied'] = True
        entry['Baseline_Level'] = baseline_data['avg_intensity']
        entry['Baseline_SNR'] = baseline_data['snr']
        entry['Baseline_Quality'] = 'Excellent' if baseline_data['snr'] > 50 else 'Good' if baseline_data['snr'] > 20 else 'Fair'
        print(f"✅ Enhanced baseline correction applied (SNR: {baseline_data['snr']:.1f})")
    else:
        entry['Baseline_Applied'] = False
        entry['Baseline_Quality'] = 'No baseline'
        print("⚠️ No baseline region marked - using raw intensities")
    
    # Enhanced normalization metadata
    entry['Normalization_Scheme'] = determine_normalization_scheme(light_source_detected)
    entry['Reference_Wavelength'] = get_reference_wavelength(light_source_detected)
    entry['Intensity_Range_Min'] = 0.0
    entry['Intensity_Range_Max'] = 100.0
    entry['Enhanced_Analysis'] = True
    entry['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    features.append(entry)
    clicks.clear()
    
    # Enhanced audio feedback based on feature type
    play_bleep(feature_type=current_type.lower())
    
    # Enhanced persistence mode handling
    if persistent_mode and current_type != 'Baseline Region':
        print(f"🎯 ENHANCED {current_type} feature completed!")
        print(f"🔄 Persistent mode: Ready to mark another {current_type}")
        print(f"📊 Total features marked: {len([f for f in features if f['Feature'] != 'Baseline Region'])}")
    else:
        current_type = None
        print(f"🎯 ENHANCED {current_type} feature completed!")
        print("✨ Ready for next feature - select another structure type")

def onclick_enhanced(event):
    """Enhanced click handler with audio feedback and real-time analysis"""
    global current_type, spectrum_df, clicks, bleep_enabled
    
    if magnify_mode:
        print("🔍 Magnify mode active - markers disabled")
        return
        
    if current_type is None:
        print("❌ Please select a feature type first!")
        play_bleep(feature_type="error")
        return

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    wavelength = event.xdata
    raw_intensity = event.ydata
    
    # Enhanced intensity measurement
    precise_intensity = get_enhanced_intensity_at_wavelength(spectrum_df, wavelength)
    if precise_intensity is None:
        precise_intensity = raw_intensity
    
    # Enhanced baseline correction
    correction_result = calculate_enhanced_baseline_corrected_intensity(precise_intensity, baseline_data)
    
    # Enhanced slope analysis
    slope_data = calculate_enhanced_local_slope(spectrum_df, wavelength)
    
    # Enhanced real-time relative height
    relative_data = calculate_relative_height_realtime(wavelength, correction_result['corrected_intensity'])
    
    print(f"✅ Enhanced click at {wavelength:.2f}nm:")
    print(f"   Raw: {precise_intensity:.3f} → Corrected: {correction_result['corrected_intensity']:.3f}")
    print(f"   Baseline quality: {correction_result['baseline_quality']}")
    if slope_data:
        print(f"   Slope: {slope_data['slope']:.4f} ({slope_data['curvature_indicator']} curvature)")
    if relative_data:
        print(f"   Relative height: {relative_data['relative_height']:.3f} vs {relative_data['comparison_count']} features")

    clicks.append((wavelength, precise_intensity))
    idx = len(clicks)
    
    try:
        label = STRUCTURE_TYPES[current_type][idx - 1]
    except IndexError:
        print("⚠️ Too many clicks for this feature type.")
        play_bleep(feature_type="error")
        return

    # Enhanced visual feedback with feature-specific colors
    colors = {
        'Mound': 'red', 'Plateau': 'green', 'Peak': 'blue', 
        'Trough': 'purple', 'Valley': 'orange', 'Diagnostic Region': 'gold',
        'Baseline Region': 'gray'
    }
    color = colors.get(current_type, 'black')
    
    # Enhanced dot size based on intensity
    dot_size = max(25, min(100, correction_result['corrected_intensity'] / 10))
    
    dot = ax.scatter(wavelength, precise_intensity, c=color, s=dot_size, marker='o', 
                    edgecolors='black', linewidth=1.5, zorder=10, alpha=0.8)
    lines_drawn.append(dot)
    
    expected_clicks = len(STRUCTURE_TYPES[current_type])
    print(f"🎯 Enhanced {current_type} {label} ({idx}/{expected_clicks})")
    
    # Enhanced audio feedback for each click
    if bleep_enabled:
        play_bleep(feature_type=current_type.lower())
    
    plt.draw()

    # Complete feature if all points marked
    if idx == expected_clicks:
        complete_enhanced_feature()
        
        # Enhanced completion feedback
        if bleep_enabled:
            play_bleep(feature_type="completion")

def save_enhanced_callback(event):
    """Enhanced save with auto-import to database"""
    if features:
        df = pd.DataFrame(features)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Enhanced filename with light source and gem ID
        base_name = current_gem_id or filename.replace('.txt', '') or "unknown"
        light_suffix = f"_{light_source_detected.lower()}" if light_source_detected else ""
        outname = f"{base_name}{light_suffix}_enhanced_features_{timestamp}.csv"
        
        # Create enhanced output directory structure
        light_dir = light_source_detected.lower() if light_source_detected else "unknown"
        full_output_dir = os.path.join(OUTPUT_DIRECTORY, light_dir)
        os.makedirs(full_output_dir, exist_ok=True)
        full_output_path = os.path.join(full_output_dir, outname)
        
        try:
            df.to_csv(full_output_path, index=False)
            print(f"✅ Saved {len(features)} enhanced features to {full_output_path}")
            
            # Enhanced summary with audio feedback
            feature_count = len([f for f in features if f['Feature'] != 'Baseline Region'])
            baseline_count = len([f for f in features if f['Feature'] == 'Baseline Region'])
            
            print(f"\n📊 ENHANCED SAVE SUMMARY:")
            print(f"   Features marked: {feature_count}")
            print(f"   Baseline regions: {baseline_count}")
            print(f"   Light source: {light_source_detected}")
            print(f"   Gem ID: {current_gem_id}")
            print(f"   Enhanced metadata: ✅")
            print(f"   Baseline corrections: {'✅' if baseline_data else '❌'}")
            
            # Audio feedback for successful save
            play_bleep(feature_type="completion")
            
            # ENHANCED: Auto-import to database
            if auto_import_enabled:
                success = auto_import_to_database(full_output_path)
                if success:
                    print(f"   Database integration: ✅")
                else:
                    print(f"   Database integration: ❌")
            
        except Exception as e:
            print(f"❌ Enhanced save error: {e}")
            play_bleep(feature_type="error")
            return
        
        # Enhanced session continuation dialog
        ask_enhanced_continue_or_finish()
    else:
        print("⚠️ No enhanced features marked.")
        play_bleep(feature_type="error")
        ask_enhanced_continue_or_finish()

def ask_enhanced_continue_or_finish():
    """Enhanced session continuation with audio feedback"""
    try:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        
        feature_count = len([f for f in features if f['Feature'] != 'Baseline Region'])
        
        choice = messagebox.askyesnocancel(
            "Enhanced Analysis Session Options",
            f"📊 Enhanced analysis saved successfully!\n\n"
            f"Features marked: {feature_count}\n"
            f"Light source: {light_source_detected}\n"
            f"Gem ID: {current_gem_id}\n"
            f"Audio feedback: {'ON' if bleep_enabled else 'OFF'}\n"
            f"Auto-import: {'ON' if auto_import_enabled else 'OFF'}\n\n"
            f"What would you like to do next?\n\n"
            f"YES = Continue marking more features\n"
            f"NO = Analyze another spectrum\n" 
            f"CANCEL = Exit enhanced system",
            parent=root
        )
        
        root.quit()
        root.destroy()
        
        if choice is True:  # YES - Continue
            print("🔄 CONTINUING ENHANCED SESSION...")
            if baseline_data:
                print(f"   Baseline preserved (SNR: {baseline_data['snr']:.1f})")
            print("   Ready to mark more enhanced features!")
            play_bleep(feature_type="completion")
            return
        elif choice is False:  # NO - New spectrum
            print("🔄 Starting enhanced analysis of new spectrum...")
            play_bleep(feature_type="completion")
            plt.close('all')
            reset_enhanced_globals()
            time.sleep(0.5)
            run_enhanced_marker()
        else:  # CANCEL - Exit
            print("👋 Enhanced analysis session complete!")
            play_bleep(feature_type="completion")
            plt.close('all')
            
    except Exception as e:
        print(f"❌ Enhanced dialog error: {e}")
        print("🔄 Continuing enhanced session by default...")

def load_and_process_enhanced_spectrum_file(file_path):
    """Load and process spectrum file with enhanced validation"""
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None)
        
        if df.shape[1] < 2:
            return None, "File does not contain two columns of data"
        
        # Enhanced wavelength validation and correction
        wavelengths = df.iloc[:, 0]
        intensities = df.iloc[:, 1]
        
        # Check for common issues
        if wavelengths.min() < 0 or wavelengths.max() > 10000:
            return None, "Wavelength range appears invalid (not in nm)"
        
        if intensities.min() < 0:
            print("⚠️ Warning: Negative intensities found - may indicate issues")
        
        first_wl = wavelengths.iloc[0]
        last_wl = wavelengths.iloc[-1]
        
        if first_wl > last_wl:
            # Enhanced descending order correction
            df = df.iloc[::-1].reset_index(drop=True)
            print("🔄 Enhanced wavelength order correction applied (ascending)")
        
        # Enhanced data quality assessment
        data_range = wavelengths.max() - wavelengths.min()
        data_density = len(wavelengths) / data_range if data_range > 0 else 0
        
        print(f"📊 Enhanced data quality:")
        print(f"   Range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
        print(f"   Points: {len(wavelengths)} (density: {data_density:.2f} pts/nm)")
        print(f"   Intensity range: {intensities.min():.3f} - {intensities.max():.3f}")
        
        return df, "success"
        
    except Exception as e:
        return None, f"Enhanced load failed: {e}"

def reset_enhanced_globals():
    """Reset all enhanced global variables for new analysis"""
    global features, current_type, clicks, filename, lines_drawn, texts_drawn
    global magnify_mode, baseline_data, spectrum_df, ax, current_gem_id, light_source_detected
    
    features = []
    current_type = None
    clicks = []
    filename = ""
    lines_drawn = []
    texts_drawn = []
    magnify_mode = False
    baseline_data = None
    spectrum_df = None
    ax = None
    current_gem_id = None
    light_source_detected = None
    
    print("🔄 Enhanced system reset complete - ready for new analysis")

def create_enhanced_ui(fig, ax):
    """Create enhanced UI with audio and database integration controls"""
    button_width = 0.13
    button_height = 0.04
    button_x = 0.84
    
    buttons = []
    
    # Enhanced baseline region button (priority)
    ax_baseline = fig.add_axes([button_x, 0.90, button_width, button_height])
    ax_baseline.set_facecolor('lightgray')
    btn_baseline = Button(ax_baseline, 'Enhanced\nBaseline (7)', color='lightgray', hovercolor='gray')
    btn_baseline.on_clicked(select_baseline_region_enhanced)
    buttons.append(btn_baseline)
    
    # Enhanced feature buttons
    enhanced_buttons = [
        (0.84, 'Plateau\n(Key: 1)', 'lightgreen', select_plateau_enhanced),
        (0.78, 'Mound\n(Key: 2)', 'lightcoral', select_mound_enhanced),
        (0.72, 'Peak\n(Key: 3)', 'lightblue', select_peak_enhanced),
        (0.66, 'Trough\n(Key: 4)', 'plum', select_trough_enhanced),
        (0.60, 'Valley\n(Key: 5)', 'orange', select_valley_enhanced),
        (0.54, 'Diagnostic\n(Key: 6)', 'lightyellow', select_diagnostic_region_enhanced),
    ]
    
    for y_pos, label, color, callback in enhanced_buttons:
        ax_btn = fig.add_axes([button_x, y_pos, button_width, button_height])
        ax_btn.set_facecolor(color)
        btn = Button(ax_btn, label, color=color, hovercolor=color.replace('light', ''))
        btn.on_clicked(callback)
        buttons.append(btn)
    
    # Enhanced control buttons
    control_buttons = [
        (0.46, 'Audio Bleep\n(Key: B)', 'lightsteelblue', toggle_enhanced_bleep),
        (0.40, 'Auto-Import\n(Key: I)', 'lightcyan', toggle_enhanced_auto_import),
        (0.34, 'Magnify\n(Key: M)', 'lightsteelblue', toggle_magnify_enhanced),
        (0.28, 'Undo\n(Key: U)', 'mistyrose', undo_enhanced_callback),
        (0.22, 'Save Enhanced\n(Key: S)', 'lightgreen', save_enhanced_callback),
        (0.16, 'Persistent\n(Key: P)', 'lavender', toggle_persistent_enhanced)
    ]
    
    for y_pos, label, color, callback in control_buttons:
        ax_btn = fig.add_axes([button_x, y_pos, button_width, button_height])
        btn = Button(ax_btn, label, color=color, hovercolor=color.replace('light', ''))
        btn.on_clicked(callback)
        buttons.append(btn)
    
    # Enhanced status display
    ax_status = fig.add_axes([button_x, 0.08, button_width, 0.06])
    ax_status.text(0.5, 0.7, f'Audio: {"ON" if bleep_enabled else "OFF"}', ha='center', fontsize=8, weight='bold')
    ax_status.text(0.5, 0.5, f'Auto-Import: {"ON" if auto_import_enabled else "OFF"}', ha='center', fontsize=8, weight='bold')
    ax_status.text(0.5, 0.3, f'Light: {light_source_detected or "Unknown"}', ha='center', fontsize=8, weight='bold')
    ax_status.text(0.5, 0.1, f'Gem: {current_gem_id or "Unknown"}', ha='center', fontsize=8, weight='bold')
    ax_status.set_xlim(0, 1)
    ax_status.set_ylim(0, 1)
    ax_status.axis('off')
    buttons.append(ax_status)
    
    return buttons

# Enhanced button callbacks with audio feedback
def select_plateau_enhanced(event):
    global current_type, clicks
    print("🟢 ENHANCED PLATEAU SELECTED!")
    current_type = 'Plateau'
    clicks = []
    play_bleep(feature_type="plateau")
    print(f"✅ Enhanced {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_mound_enhanced(event):
    global current_type, clicks
    print("🔴 ENHANCED MOUND SELECTED!")
    current_type = 'Mound'
    clicks = []
    play_bleep(feature_type="mound")
    print(f"✅ Enhanced {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_peak_enhanced(event):
    global current_type, clicks
    print("🔵 ENHANCED PEAK SELECTED!")
    current_type = 'Peak'
    clicks = []
    play_bleep(feature_type="peak")
    print(f"✅ Enhanced {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_trough_enhanced(event):
    global current_type, clicks
    print("🟣 ENHANCED TROUGH SELECTED!")
    current_type = 'Trough'
    clicks = []
    play_bleep(feature_type="trough")
    print(f"✅ Enhanced {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_valley_enhanced(event):
    global current_type, clicks
    print("🟠 ENHANCED VALLEY SELECTED!")
    current_type = 'Valley'
    clicks = []
    play_bleep(feature_type="valley")
    print(f"✅ Enhanced {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_diagnostic_region_enhanced(event):
    global current_type, clicks
    print("🟡 ENHANCED DIAGNOSTIC REGION SELECTED!")
    current_type = 'Diagnostic Region'
    clicks = []
    play_bleep(feature_type="diagnostic")
    print(f"✅ Enhanced {current_type} - Click: {STRUCTURE_TYPES[current_type]}")

def select_baseline_region_enhanced(event):
    global current_type, clicks
    print("⚪ ENHANCED BASELINE REGION SELECTED!")
    current_type = 'Baseline Region'
    clicks = []
    play_bleep(feature_type="baseline")
    print(f"✅ Enhanced {current_type} - Click: {STRUCTURE_TYPES[current_type]}")
    print("📊 Mark a flat region with no features for enhanced baseline calculation")

def toggle_enhanced_bleep(event):
    global bleep_enabled
    bleep_enabled = not bleep_enabled
    print(f"🔊 ENHANCED AUDIO BLEEP: {'ON' if bleep_enabled else 'OFF'}")
    
    if bleep_enabled:
        print("   ✅ Audio feedback enabled for all features")
        play_bleep(feature_type="completion")
    else:
        print("   ❌ Audio feedback disabled")

def toggle_enhanced_auto_import(event):
    global auto_import_enabled
    auto_import_enabled = not auto_import_enabled
    print(f"🔄 ENHANCED AUTO-IMPORT: {'ON' if auto_import_enabled else 'OFF'}")
    
    if auto_import_enabled:
        print("   ✅ CSV files will be automatically imported to database")
        if bleep_enabled:
            play_bleep(feature_type="completion")
    else:
        print("   ❌ Manual import required for CSV files")

def toggle_magnify_enhanced(event):
    global magnify_mode
    magnify_mode = not magnify_mode
    
    if magnify_mode:
        print("🔍 ENHANCED MAGNIFY MODE ON - markers disabled")
        if bleep_enabled:
            play_bleep(feature_type="completion")
    else:
        print("🔍 ENHANCED MAGNIFY MODE OFF - enhanced markers active")

def toggle_persistent_enhanced(event):
    global persistent_mode
    persistent_mode = not persistent_mode
    print(f"🔄 ENHANCED PERSISTENT MODE: {'ON' if persistent_mode else 'OFF'}")
    
    if persistent_mode:
        print("   ✅ Feature type stays selected - great for multiple similar features")
    else:
        print("   ❌ Must reselect feature type after each feature")
    
    if bleep_enabled:
        play_bleep(feature_type="completion")

def undo_enhanced_callback(event):
    """Enhanced undo with audio feedback"""
    global features, clicks, lines_drawn, current_type, baseline_data
    
    if clicks:
        clicks.pop()
        if lines_drawn:
            dot = lines_drawn.pop()
            dot.remove()
        plt.draw()
        remaining_clicks = len(STRUCTURE_TYPES.get(current_type, [])) - len(clicks) if current_type else 0
        print(f"↩️ Enhanced undo: {remaining_clicks} more clicks needed for {current_type}")
        if bleep_enabled:
            play_bleep(feature_type="valley")
        return
    
    if features:
        last_feature = features[-1]
        feature_type = last_feature['Feature']
        
        expected_dots = len(STRUCTURE_TYPES.get(feature_type, []))
        
        dots_removed = 0
        while lines_drawn and dots_removed < expected_dots:
            dot = lines_drawn.pop()
            dot.remove()
            dots_removed += 1
        
        features.pop()
        
        print(f"↩️ ENHANCED UNDO: Removed {feature_type}")
        print(f"   Enhanced features remaining: {len([f for f in features if f['Feature'] != 'Baseline Region'])}")
        
        if feature_type == 'Baseline Region':
            baseline_data = None
            print("⚠️ Enhanced baseline removed - raw intensities will be used")
        
        if bleep_enabled:
            play_bleep(feature_type="valley")
        
        plt.draw()
        return
    
    print("⚠️ Nothing to undo")
    if bleep_enabled:
        play_bleep(feature_type="error")

def onkey_enhanced(event):
    """Enhanced keyboard shortcuts with audio feedback"""
    global current_type, clicks, persistent_mode, bleep_enabled, auto_import_enabled
    
    key_actions = {
        '1': ('Plateau', play_bleep),
        '2': ('Mound', play_bleep),
        '3': ('Peak', play_bleep),
        '4': ('Trough', play_bleep),
        '5': ('Valley', play_bleep),
        '6': ('Diagnostic Region', play_bleep),
        '7': ('Baseline Region', play_bleep)
    }
    
    if event.key in key_actions:
        feature_type, audio_func = key_actions[event.key]
        current_type = feature_type
        clicks = []
        if bleep_enabled:
            audio_func(feature_type=feature_type.lower())
        print(f"⌨️ Enhanced {current_type} selected - Persistent: {persistent_mode}")
        
    elif event.key == 'p':
        toggle_persistent_enhanced(event)
    elif event.key == 'm':
        toggle_magnify_enhanced(event)
    elif event.key == 'u':
        undo_enhanced_callback(event)
    elif event.key == 's':
        save_enhanced_callback(event)
    elif event.key == 'b':
        toggle_enhanced_bleep(event)
    elif event.key == 'i':
        toggle_enhanced_auto_import(event)

def run_enhanced_marker():
    """Main enhanced marker function with complete integration"""
    global filename, ax, spectrum_df, current_gem_id, light_source_detected
    
    print("🔬 ENHANCED GEMINI STRUCTURAL MARKER - COMPLETE INTEGRATION")
    print("=" * 65)
    print("✨ ENHANCED FEATURES:")
    print("   🔊 Audio bleep feedback for each feature marked")
    print("   🗄️ Automatic database import after analysis")
    print("   📏 Real-time relative height calculations")
    print("   📊 Enhanced baseline & slope analysis with quality metrics")
    print("   🎯 Progressive feature detection with audio cues")
    print("   🔄 Seamless integration with main analysis system")
    print("=" * 65)
    
    # Enhanced file selection with retry capability
    file_path = None
    while not file_path:
        try:
            print("📂 Opening enhanced file selection dialog...")
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            
            default_dir = r"C:\Users\David\OneDrive\Desktop\gemini matcher\gemini sp10 raw\raw text"
            
            file_path = filedialog.askopenfilename(
                parent=root,
                initialdir=default_dir,
                title="Select Spectrum for Enhanced Analysis",
                filetypes=[("Text files", "*.txt")]
            )
            root.quit()
            root.destroy()
            
            import gc
            gc.collect()
            
            if not file_path or file_path == "":
                print("❌ No file selected")
                
                root2 = tk.Tk()
                root2.withdraw()
                root2.lift()
                root2.attributes('-topmost', True)
                
                try_again = messagebox.askyesno(
                    "Enhanced File Selection",
                    "No file was selected for enhanced analysis.\n\nWould you like to try again?",
                    parent=root2
                )
                
                root2.quit()
                root2.destroy()
                gc.collect()
                
                if not try_again:
                    print("👋 Enhanced file selection cancelled - exiting")
                    return
                else:
                    print("🔄 Trying enhanced file selection again...")
                    file_path = None
                    continue
            else:
                print(f"✅ Selected for enhanced analysis: {os.path.basename(file_path)}")
                break
                
        except Exception as e:
            print(f"❌ Enhanced file selection error: {e}")
            
            try:
                root3 = tk.Tk()
                root3.withdraw()
                
                try_again = messagebox.askyesno(
                    "Enhanced File Selection Error",
                    f"Error during enhanced file selection: {e}\n\nTry again?",
                    parent=root3
                )
                
                root3.quit()
                root3.destroy()
                
                if not try_again:
                    print("👋 Exiting enhanced analysis due to file selection error")
                    return
                else:
                    file_path = None
                    continue
            except:
                print("👋 Exiting enhanced analysis due to multiple errors")
                return
    
    # Enhanced file processing
    filename = os.path.basename(file_path)
    print(f"📁 Loading for enhanced analysis: {filename}")
    
    # Enhanced light source and gem ID detection
    light_source_detected = detect_light_source_from_filename(filename)
    current_gem_id = extract_gem_id_from_filename(filename)
    
    print(f"🔍 Enhanced detection:")
    print(f"   Light source: {light_source_detected}")
    print(f"   Gem ID: {current_gem_id}")
    
    # Load and process spectrum with enhanced validation
    spectrum_df, load_info = load_and_process_enhanced_spectrum_file(file_path)
    if spectrum_df is None:
        print(f"❌ Enhanced load failed: {load_info}")
        if bleep_enabled:
            play_bleep(feature_type="error")
        return
    
    print(f"✅ Enhanced spectrum loaded: {len(spectrum_df)} data points")
    
    # Create enhanced plot
    try:
        plt.close('all')
        import matplotlib
        matplotlib.pyplot.close('all')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Enhanced spectrum plotting
        wavelengths = spectrum_df.iloc[:, 0]
        intensities = spectrum_df.iloc[:, 1]
        ax.plot(wavelengths, intensities, 'k-', linewidth=0.6, alpha=0.8)
        
        # Enhanced title with detection info
        title = f"Enhanced Structural Marker - {filename}\n"
        title += f"Light: {light_source_detected} | Gem: {current_gem_id} | "
        title += f"Audio: {'ON' if bleep_enabled else 'OFF'} | Auto-Import: {'ON' if auto_import_enabled else 'OFF'}"
        ax.set_title(title, fontsize=12, weight='bold')
        
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        
        # Create enhanced UI
        buttons = create_enhanced_ui(fig, ax)
        fig._enhanced_buttons = buttons
        
        # Connect enhanced events
        fig.canvas.mpl_connect('button_press_event', onclick_enhanced)
        fig.canvas.mpl_connect('key_press_event', onkey_enhanced)
        
        plt.subplots_adjust(right=0.80)
        
        # Enhanced window positioning
        try:
            manager = fig.canvas.manager
            manager.window.wm_geometry("+100+100")
            manager.window.lift()
            manager.window.attributes('-topmost', True)
            manager.window.attributes('-topmost', False)
        except:
            pass
        
        print("\n🎯 ENHANCED ANALYSIS READY!")
        print("=" * 45)
        print("🆕 STEP 1: Mark BASELINE REGION first (gray button)")
        print("   - Establishes enhanced SNR and quality metrics")
        print("   - Enables baseline correction for all features")
        print("\n🎯 STEP 2: Mark spectral features with enhanced data capture")
        print("   - Each click captures: intensity, slope, relative height")
        print("   - Audio feedback for each successful marking")
        print("   - Real-time database integration available")
        print("\n📊 ENHANCED DATA CAPTURED:")
        print("   ✅ Wavelength positions with sub-nm precision")
        print("   ✅ Raw and baseline-corrected intensities") 
        print("   ✅ Local slope analysis with curvature detection")
        print("   ✅ Enhanced baseline statistics and SNR")
        print("   ✅ Real-time relative height measurements")
        print("   ✅ Automatic normalization metadata")
        print("   ✅ Light source and gem ID auto-detection")
        print("\n⌨️ ENHANCED KEYBOARD SHORTCUTS:")
        print("   1-7: Select enhanced feature types")
        print("   P: Toggle persistent mode")
        print("   M: Toggle magnify mode")
        print("   U: Enhanced undo with audio")
        print("   S: Save with auto-import")
        print("   B: Toggle audio bleep system")
        print("   I: Toggle auto-import to database")
        print("\n🔊 AUDIO FEATURES:")
        print("   Different tones for each feature type")
        print("   Completion sounds for successful operations")
        print("   Error sounds for invalid actions")
        
        # Enhanced startup audio
        if bleep_enabled:
            play_bleep(feature_type="completion")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Enhanced plot creation error: {e}")
        print("💡 Try restarting the enhanced program if this persists")
        if bleep_enabled:
            play_bleep(feature_type="error")

def main():
    """Enhanced main function with complete integration"""
    print("🔬 ENHANCED GEMINI STRUCTURAL MARKER")
    print("🚀 Complete Integration: Audio + Database + Real-time Analysis")
    
    # Initialize enhanced systems
    if HAS_AUDIO:
        print("🔊 Audio system: Available")
    else:
        print("🔇 Audio system: Not available (install winsound/pygame)")
    
    print(f"🗄️ Database integration: {DATABASE_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIRECTORY}")
    
    run_enhanced_marker()

if __name__ == '__main__':
    main()