#!/usr/bin/env python3
"""
🔧 GEMINI B PEAK DETECTOR (Main Detection Logic)
File: gemini_Bpeak_detector.py
Core detection algorithms with standardized point structures

8 Structural Features with Standard Point Definitions:
- Peak: 3 points (Start, Crest, End)
- Plateau: 3 points (Start, Mid, End) 
- Shoulder: 3 points (Start, Mid, End)
- Trough: 3 points (Start, Bottom, End) - B/H only
- Mound: 4 points (Start, Crest, End, Summary) - only Summary row
- Baseline: 2 points (Start, End) - 300-350nm
- Diagnostic Region: 2 points (Start, End) - complex regions
- Valley: 1 point (Mid) - midpoint between features

Version: 3.0 (Standardized point structures)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class BSpectralFeature:
    """Detected feature with standardized point structure"""
    wavelength: float
    intensity: float
    feature_type: str  # peak, plateau, shoulder, trough, mound, baseline, diagnostic_region, valley
    point_type: str    # Start, Crest, End, Mid, Bottom, Summary
    feature_group: str # Same as feature_type for consistency
    prominence: float
    snr: float
    confidence: float
    detection_method: str
    width_nm: float = 0.0
    width_half_max: float = 0.0  # For peak shape/broad classification
    start_wavelength: float = 0.0
    end_wavelength: float = 0.0
    effective_baseline: float = 0.0  # Local baseline for height calculation
    global_baseline: float = 0.0     # Global baseline for height calculation

class GeminiBSpectralDetector:
    """Standardized B spectra detection matching manual marking system"""
    
    def __init__(self):
        # Peak detection (shape vs broad determined by width at half max)
        self.peak_prominence_threshold = 2.0
        self.peak_shape_width_threshold = 15.0  # nm, width at half max
        
        # Plateau detection (slope → flat → slope)
        self.plateau_flatness_threshold = 0.015  # Max slope in flat section
        self.plateau_min_width = 30.0  # nm
        
        # Shoulder detection (like plateau but never quite flat)
        self.shoulder_slope_threshold = 0.05   # More slope than plateau
        self.shoulder_min_width = 20.0  # nm
        
        # Trough detection (sudden downward slope change in B/H only)
        self.trough_slope_change_threshold = 0.1
        self.trough_min_depth = 1.5
        
        # Mound detection (broad peaks, 4 points including summary)
        self.mound_min_prominence = 5.0
        self.mound_min_width = 50.0  # nm
        
        # Baseline (reliably 300-350nm)
        self.baseline_start = 300.0  # nm
        self.baseline_end = 350.0    # nm
        
        # Valley detection (midpoint between features)
        self.valley_search_window = 50.0  # nm around midpoint
        
        # General parameters
        self.smoothing_sigma = 2.0
        self.global_baseline_value = 0.0
        
    def normalize_b_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Apply B spectra normalization: 650nm → 50000, then scale to 0-100"""
        idx_650 = np.argmin(np.abs(wavelengths - 650.0))
        intensity_650 = intensities[idx_650]
        
        if intensity_650 <= 0:
            raise ValueError(f"Invalid intensity at 650nm: {intensity_650}")
        
        normalized = (intensities / intensity_650) * 50000 / 500
        
        self.norm_reference_wavelength = wavelengths[idx_650]
        self.norm_reference_intensity = intensity_650
        
        return normalized
    
    def calculate_global_baseline(self, wavelengths: np.ndarray, intensities: np.ndarray) -> float:
        """Calculate global baseline from 300-350nm region"""
        baseline_mask = (wavelengths >= self.baseline_start) & (wavelengths <= self.baseline_end)
        if np.any(baseline_mask):
            self.global_baseline_value = np.mean(intensities[baseline_mask])
        else:
            self.global_baseline_value = np.min(intensities)
        return self.global_baseline_value
    
    def detect_baseline(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect baseline - 2 points (Start, End) in 300-350nm range"""
        baseline_mask = (wavelengths >= self.baseline_start) & (wavelengths <= self.baseline_end)
        baseline_indices = np.where(baseline_mask)[0]
        
        if len(baseline_indices) < 2:
            return []
        
        start_idx = baseline_indices[0]
        end_idx = baseline_indices[-1]
        
        baseline_std = np.std(intensities[baseline_indices])
        snr = 1.0 / (baseline_std + 1e-6)
        confidence = 0.9 if baseline_std < 0.1 else 0.7
        
        return [
            BSpectralFeature(
                wavelength=wavelengths[start_idx],
                intensity=intensities[start_idx],
                feature_type="baseline",
                point_type="Start",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=confidence,
                detection_method="standard_range",
                global_baseline=self.global_baseline_value,
                effective_baseline=self.global_baseline_value
            ),
            BSpectralFeature(
                wavelength=wavelengths[end_idx],
                intensity=intensities[end_idx],
                feature_type="baseline",
                point_type="End",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=confidence,
                detection_method="standard_range",
                global_baseline=self.global_baseline_value,
                effective_baseline=self.global_baseline_value
            )
        ]
    
    def detect_peaks(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect peaks - 3 points (Start, Crest, End), shape vs broad by width at half max"""
        peaks, properties = signal.find_peaks(intensities, 
                                            prominence=self.peak_prominence_threshold,
                                            distance=5)
        
        features = []
        
        for peak_idx in peaks:
            # Calculate width at half maximum
            peak_intensity = intensities[peak_idx]
            half_max = (peak_intensity + self.global_baseline_value) / 2
            
            # Find start and end at half max
            start_idx = peak_idx
            for i in range(peak_idx - 1, -1, -1):
                if intensities[i] <= half_max:
                    start_idx = i
                    break
            
            end_idx = peak_idx  
            for i in range(peak_idx + 1, len(intensities)):
                if intensities[i] <= half_max:
                    end_idx = i
                    break
            
            width_half_max = wavelengths[end_idx] - wavelengths[start_idx]
            width_nm = wavelengths[end_idx] - wavelengths[start_idx]
            
            # Calculate effective baseline (local minimum)
            local_window = slice(max(0, start_idx - 10), min(len(intensities), end_idx + 10))
            effective_baseline = np.min(intensities[local_window])
            
            prominence = peak_intensity - effective_baseline
            snr = self._calculate_snr(intensities, peak_idx)
            
            # Classification: shape vs broad based on width at half max
            peak_class = "shape" if width_half_max <= self.peak_shape_width_threshold else "broad"
            
            if prominence >= self.peak_prominence_threshold:
                features.extend([
                    BSpectralFeature(
                        wavelength=wavelengths[start_idx],
                        intensity=intensities[start_idx],
                        feature_type="peak",
                        point_type="Start",
                        feature_group="peak",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method=f"{peak_class}_peak",
                        width_nm=width_nm,
                        width_half_max=width_half_max,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[peak_idx],
                        intensity=peak_intensity,
                        feature_type="peak",
                        point_type="Crest",
                        feature_group="peak",
                        prominence=prominence,
                        snr=snr,
                        confidence=0.9,
                        detection_method=f"{peak_class}_peak",
                        width_nm=width_nm,
                        width_half_max=width_half_max,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[end_idx],
                        intensity=intensities[end_idx],
                        feature_type="peak",
                        point_type="End",
                        feature_group="peak",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method=f"{peak_class}_peak",
                        width_nm=width_nm,
                        width_half_max=width_half_max,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    )
                ])
        
        return features
    
    def detect_plateaus(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect plateaus - 3 points (Start, Mid, End): slope → flat → slope"""
        smoothed = gaussian_filter1d(intensities, sigma=self.smoothing_sigma)
        derivatives = np.gradient(smoothed)
        
        # Find regions with low slope (potential plateau centers)
        flat_mask = np.abs(derivatives) < self.plateau_flatness_threshold
        plateau_regions = self._find_connected_regions(flat_mask, 
                                                      self.plateau_min_width / np.mean(np.diff(wavelengths)))
        
        features = []
        
        for start_idx, end_idx in plateau_regions:
            width_nm = wavelengths[end_idx] - wavelengths[start_idx]
            
            if width_nm >= self.plateau_min_width:
                # Verify slope → flat → slope pattern
                pre_slope = np.mean(derivatives[max(0, start_idx - 10):start_idx])
                post_slope = np.mean(derivatives[end_idx:min(len(derivatives), end_idx + 10)])
                
                # Must have significant slopes before and after
                if abs(pre_slope) > 0.02 and abs(post_slope) > 0.02:
                    mid_idx = (start_idx + end_idx) // 2
                    effective_baseline = np.min(smoothed[start_idx:end_idx+1])
                    prominence = smoothed[mid_idx] - effective_baseline
                    
                    features.extend([
                        BSpectralFeature(
                            wavelength=wavelengths[start_idx],
                            intensity=intensities[start_idx],
                            feature_type="plateau",
                            point_type="Start",
                            feature_group="plateau",
                            prominence=0.0,
                            snr=0.0,
                            confidence=0.7,
                            detection_method="slope_analysis",
                            width_nm=width_nm,
                            start_wavelength=wavelengths[start_idx],
                            end_wavelength=wavelengths[end_idx],
                            effective_baseline=effective_baseline,
                            global_baseline=self.global_baseline_value
                        ),
                        BSpectralFeature(
                            wavelength=wavelengths[mid_idx],
                            intensity=intensities[mid_idx],
                            feature_type="plateau",
                            point_type="Mid",
                            feature_group="plateau",
                            prominence=prominence,
                            snr=0.0,
                            confidence=0.8,
                            detection_method="slope_analysis",
                            width_nm=width_nm,
                            start_wavelength=wavelengths[start_idx],
                            end_wavelength=wavelengths[end_idx],
                            effective_baseline=effective_baseline,
                            global_baseline=self.global_baseline_value
                        ),
                        BSpectralFeature(
                            wavelength=wavelengths[end_idx],
                            intensity=intensities[end_idx],
                            feature_type="plateau",
                            point_type="End",
                            feature_group="plateau",
                            prominence=0.0,
                            snr=0.0,
                            confidence=0.7,
                            detection_method="slope_analysis",
                            width_nm=width_nm,
                            start_wavelength=wavelengths[start_idx],
                            end_wavelength=wavelengths[end_idx],
                            effective_baseline=effective_baseline,
                            global_baseline=self.global_baseline_value
                        )
                    ])
        
        return features
    
    def detect_shoulders(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect shoulders - 3 points (Start, Mid, End): like plateau but never quite flat"""
        smoothed = gaussian_filter1d(intensities, sigma=self.smoothing_sigma)
        derivatives = np.gradient(smoothed)
        
        # Find regions with moderate slope (not flat like plateau)
        shoulder_mask = (np.abs(derivatives) >= self.plateau_flatness_threshold) & \
                       (np.abs(derivatives) <= self.shoulder_slope_threshold)
        
        shoulder_regions = self._find_connected_regions(shoulder_mask, 
                                                       self.shoulder_min_width / np.mean(np.diff(wavelengths)))
        
        features = []
        
        for start_idx, end_idx in shoulder_regions:
            width_nm = wavelengths[end_idx] - wavelengths[start_idx]
            
            if width_nm >= self.shoulder_min_width:
                mid_idx = (start_idx + end_idx) // 2
                effective_baseline = np.min(smoothed[start_idx:end_idx+1])
                prominence = smoothed[mid_idx] - effective_baseline
                
                features.extend([
                    BSpectralFeature(
                        wavelength=wavelengths[start_idx],
                        intensity=intensities[start_idx],
                        feature_type="shoulder",
                        point_type="Start",
                        feature_group="shoulder",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.6,
                        detection_method="slope_analysis",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[mid_idx],
                        intensity=intensities[mid_idx],
                        feature_type="shoulder",
                        point_type="Mid",
                        feature_group="shoulder",
                        prominence=prominence,
                        snr=0.0,
                        confidence=0.7,
                        detection_method="slope_analysis",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[end_idx],
                        intensity=intensities[end_idx],
                        feature_type="shoulder",
                        point_type="End",
                        feature_group="shoulder",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.6,
                        detection_method="slope_analysis",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    )
                ])
        
        return features
    
    def detect_troughs(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect troughs - 3 points (Start, Bottom, End): sudden downward slope change in B/H"""
        derivatives = np.gradient(intensities)
        slope_changes = np.gradient(derivatives)  # Second derivative
        
        # Find sudden downward slope changes
        trough_candidates = np.where(slope_changes < -self.trough_slope_change_threshold)[0]
        
        features = []
        
        for candidate_idx in trough_candidates:
            # Find local minimum near the slope change
            search_start = max(0, candidate_idx - 10)
            search_end = min(len(intensities), candidate_idx + 10)
            local_region = intensities[search_start:search_end]
            
            min_local_idx = search_start + np.argmin(local_region)
            
            # Check if it's a significant trough
            local_baseline = (intensities[search_start] + intensities[search_end-1]) / 2
            depth = local_baseline - intensities[min_local_idx]
            
            if depth >= self.trough_min_depth:
                # Find trough boundaries
                start_idx = min_local_idx
                for i in range(min_local_idx - 1, -1, -1):
                    if intensities[i] >= intensities[min_local_idx] + depth * 0.5:
                        start_idx = i
                        break
                
                end_idx = min_local_idx
                for i in range(min_local_idx + 1, len(intensities)):
                    if intensities[i] >= intensities[min_local_idx] + depth * 0.5:
                        end_idx = i
                        break
                
                width_nm = wavelengths[end_idx] - wavelengths[start_idx]
                effective_baseline = (intensities[start_idx] + intensities[end_idx]) / 2
                
                features.extend([
                    BSpectralFeature(
                        wavelength=wavelengths[start_idx],
                        intensity=intensities[start_idx],
                        feature_type="trough",
                        point_type="Start",
                        feature_group="trough",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.7,
                        detection_method="slope_change_bh",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[min_local_idx],
                        intensity=intensities[min_local_idx],
                        feature_type="trough",
                        point_type="Bottom",
                        feature_group="trough",
                        prominence=depth,
                        snr=0.0,
                        confidence=0.8,
                        detection_method="slope_change_bh",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[end_idx],
                        intensity=intensities[end_idx],
                        feature_type="trough",
                        point_type="End",
                        feature_group="trough",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.7,
                        detection_method="slope_change_bh",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    )
                ])
        
        return features
    
    def detect_mounds(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect mounds - FIXED based on 92BC1 analysis: start at 425nm not 393nm"""
        smoothed = gaussian_filter1d(intensities, sigma=self.smoothing_sigma)
        
        peaks, properties = signal.find_peaks(smoothed, 
                                            prominence=self.mound_min_prominence,
                                            distance=20)
        
        features = []
        
        for peak_idx in peaks:
            # FIXED MOUND START DETECTION based on 92BC1 comparison
            # Manual: starts at 425.06nm (intensity 0.46)
            # Auto was: starts at 393.39nm (intensity 0.1) - too early
            
            peak_intensity = smoothed[peak_idx]
            
            # Use higher threshold for mound start - wait for significant intensity rise
            # Based on 92BC1: manual started when intensity reached ~0.46
            significant_rise_threshold = self.global_baseline_value + 0.4  # Increased from 0.5 to 0.4
            
            # Find start boundary - where intensity significantly rises above baseline
            start_idx = peak_idx
            for i in range(peak_idx - 1, -1, -1):
                if smoothed[i] < significant_rise_threshold:
                    # Found where it drops below significant rise
                    start_idx = min(peak_idx - 1, i + 1)
                    break
            
            # Additional check: ensure mound starts with measurable slope increase
            if start_idx > 5:  # Need some points before to calculate slope
                local_derivatives = np.gradient(smoothed[start_idx-5:start_idx+5])
                avg_slope = np.mean(local_derivatives[3:7])  # Around start point
                
                # If slope is too small, move start point later until we find significant rise
                search_range = min(50, peak_idx - start_idx)  # Search up to 50 points forward
                for j in range(search_range):
                    test_idx = start_idx + j
                    if test_idx >= peak_idx:
                        break
                    
                    test_derivatives = np.gradient(smoothed[test_idx-2:test_idx+3])
                    test_slope = np.mean(test_derivatives)
                    test_intensity = smoothed[test_idx]
                    
                    # 92BC1 analysis: manual mound starts where intensity ~0.46 with clear upward slope
                    if test_intensity >= 0.3 and test_slope > 0.008:  # More restrictive criteria
                        start_idx = test_idx
                        break
            
            # Find end boundary - where intensity drops back toward baseline  
            end_idx = peak_idx
            for i in range(peak_idx + 1, len(smoothed)):
                if smoothed[i] < significant_rise_threshold:
                    end_idx = i
                    break
            
            # Validate width and start intensity
            width_nm = wavelengths[end_idx] - wavelengths[start_idx]
            start_intensity = smoothed[start_idx]
            
            if width_nm >= self.mound_min_width and start_intensity >= 0.25:  # Ensure minimum start intensity
                effective_baseline = min(smoothed[start_idx], smoothed[end_idx])
                prominence = peak_intensity - effective_baseline
                
                features.extend([
                    BSpectralFeature(
                        wavelength=wavelengths[start_idx],
                        intensity=intensities[start_idx],
                        feature_type="mound",
                        point_type="Start",
                        feature_group="mound",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method="refined_rise_threshold",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[peak_idx],
                        intensity=intensities[peak_idx],
                        feature_type="mound",
                        point_type="Crest",
                        feature_group="mound",
                        prominence=prominence,
                        snr=0.0,
                        confidence=0.9,
                        detection_method="refined_rise_threshold",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[end_idx],
                        intensity=intensities[end_idx],
                        feature_type="mound",
                        point_type="End",
                        feature_group="mound",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method="refined_rise_threshold",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    ),
                    BSpectralFeature(
                        wavelength=wavelengths[peak_idx],
                        intensity=intensities[peak_idx],
                        feature_type="mound",
                        point_type="Summary",
                        feature_group="mound",
                        prominence=prominence,
                        snr=0.0,
                        confidence=0.9,
                        detection_method="refined_rise_threshold",
                        width_nm=width_nm,
                        start_wavelength=wavelengths[start_idx],
                        end_wavelength=wavelengths[end_idx],
                        effective_baseline=effective_baseline,
                        global_baseline=self.global_baseline_value
                    )
                ])
        
        return features
    
    def detect_valleys(self, wavelengths: np.ndarray, intensities: np.ndarray, other_features: List[BSpectralFeature]) -> List[BSpectralFeature]:
        """Detect valleys - 1 point (Mid): midpoint between two structural features"""
        # Sort other features by wavelength
        sorted_features = sorted([f for f in other_features if f.point_type in ['Crest', 'Mid', 'Bottom']], 
                                key=lambda x: x.wavelength)
        
        features = []
        
        for i in range(len(sorted_features) - 1):
            # Find midpoint between consecutive features
            wavelength1 = sorted_features[i].wavelength
            wavelength2 = sorted_features[i + 1].wavelength
            mid_wavelength = (wavelength1 + wavelength2) / 2
            
            # Find closest data point to midpoint
            mid_idx = np.argmin(np.abs(wavelengths - mid_wavelength))
            
            # Check if there's sufficient separation to warrant a valley
            if abs(wavelength2 - wavelength1) > self.valley_search_window:
                features.append(BSpectralFeature(
                    wavelength=wavelengths[mid_idx],
                    intensity=intensities[mid_idx],
                    feature_type="valley",
                    point_type="Mid",
                    feature_group="valley",
                    prominence=0.0,
                    snr=0.0,
                    confidence=0.6,
                    detection_method="midpoint_calculation",
                    effective_baseline=intensities[mid_idx],
                    global_baseline=self.global_baseline_value
                ))
        
        return features
    
    def detect_diagnostic_regions(self, wavelengths: np.ndarray, intensities: np.ndarray, other_features: List[BSpectralFeature]) -> List[BSpectralFeature]:
        """Detect diagnostic regions - 2 points (Start, End): complex regions with multiple features"""
        # Find regions with high feature density
        feature_positions = [f.wavelength for f in other_features]
        
        # Skip if no features detected yet
        if len(feature_positions) < 3:
            return []
        
        # Simple approach: look for wavelength ranges with 3+ features within 100nm
        diagnostic_regions = []
        window_size = 100.0  # nm
        
        for i, pos in enumerate(feature_positions):
            nearby_features = [p for p in feature_positions if abs(p - pos) <= window_size]
            
            if len(nearby_features) >= 3:
                region_start = min(nearby_features)
                region_end = max(nearby_features)
                
                # Ensure minimum width
                if abs(region_end - region_start) < 20.0:  # Less than 20nm width
                    continue
                
                # Avoid duplicate regions
                if not any(abs(r[0] - region_start) < 50 for r in diagnostic_regions):
                    diagnostic_regions.append((region_start, region_end))
        
        features = []
        
        for region_start, region_end in diagnostic_regions:
            start_idx = np.argmin(np.abs(wavelengths - region_start))
            end_idx = np.argmin(np.abs(wavelengths - region_end))
            
            # Ensure valid indices and non-empty slice
            if start_idx >= end_idx or start_idx < 0 or end_idx >= len(intensities):
                continue
                
            width_nm = region_end - region_start
            
            # Safe baseline calculation with bounds checking
            region_slice = intensities[start_idx:end_idx+1]
            if len(region_slice) > 0:
                effective_baseline = np.min(region_slice)
            else:
                effective_baseline = self.global_baseline_value  # Fallback
            
            features.extend([
                BSpectralFeature(
                    wavelength=wavelengths[start_idx],
                    intensity=intensities[start_idx],
                    feature_type="diagnostic_region",
                    point_type="Start",
                    feature_group="diagnostic_region",
                    prominence=0.0,
                    snr=0.0,
                    confidence=0.5,
                    detection_method="feature_density",
                    width_nm=width_nm,
                    start_wavelength=wavelengths[start_idx],
                    end_wavelength=wavelengths[end_idx],
                    effective_baseline=effective_baseline,
                    global_baseline=self.global_baseline_value
                ),
                BSpectralFeature(
                    wavelength=wavelengths[end_idx],
                    intensity=intensities[end_idx],
                    feature_type="diagnostic_region",
                    point_type="End",
                    feature_group="diagnostic_region",
                    prominence=0.0,
                    snr=0.0,
                    confidence=0.5,
                    detection_method="feature_density",
                    width_nm=width_nm,
                    start_wavelength=wavelengths[start_idx],
                    end_wavelength=wavelengths[end_idx],
                    effective_baseline=effective_baseline,
                    global_baseline=self.global_baseline_value
                )
            ])
        
        return features
    
    def analyze_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Dict:
        """Main analysis with standardized point structures"""
        if len(wavelengths) != len(intensities):
            raise ValueError("Wavelength and intensity arrays must have same length")
        
        # Normalize spectrum
        normalized_intensities = self.normalize_b_spectrum(wavelengths, intensities)
        
        # Calculate global baseline
        self.calculate_global_baseline(wavelengths, normalized_intensities)
        
        # Detect all feature types with standardized points
        all_features = []
        
        # 1. Baseline: 2 points (Start, End)
        baseline_features = self.detect_baseline(wavelengths, normalized_intensities)
        all_features.extend(baseline_features)
        
        # 2. Peaks: 3 points (Start, Crest, End)
        peak_features = self.detect_peaks(wavelengths, normalized_intensities)
        all_features.extend(peak_features)
        
        # 3. Mounds: 4 points (Start, Crest, End, Summary)
        mound_features = self.detect_mounds(wavelengths, normalized_intensities)
        all_features.extend(mound_features)
        
        # 4. Plateaus: 3 points (Start, Mid, End)
        plateau_features = self.detect_plateaus(wavelengths, normalized_intensities)
        all_features.extend(plateau_features)
        
        # 5. Shoulders: 3 points (Start, Mid, End)
        shoulder_features = self.detect_shoulders(wavelengths, normalized_intensities)
        all_features.extend(shoulder_features)
        
        # 6. Troughs: 3 points (Start, Bottom, End) - B/H only
        trough_features = self.detect_troughs(wavelengths, normalized_intensities)
        all_features.extend(trough_features)
        
        # 7. Valleys: 1 point (Mid) - after other features detected
        valley_features = self.detect_valleys(wavelengths, normalized_intensities, all_features)
        all_features.extend(valley_features)
        
        # 8. Diagnostic Regions: 2 points (Start, End) - after other features detected
        diagnostic_features = self.detect_diagnostic_regions(wavelengths, normalized_intensities, all_features)
        all_features.extend(diagnostic_features)
        
        # Sort by wavelength
        all_features.sort(key=lambda x: x.wavelength)
        
        return {
            'features': all_features,
            'normalization': {
                'reference_wavelength': self.norm_reference_wavelength,
                'reference_intensity': self.norm_reference_intensity,
                'method': '650nm_to_50000_scale_100'
            },
            'global_baseline': self.global_baseline_value,
            'detection_strategy': 'standardized_structural_analysis',
            'overall_confidence': np.mean([f.confidence for f in all_features]) if all_features else 0.0,
            'feature_count': len(all_features),
            'feature_summary': self._summarize_features(all_features)
        }
    
    # Helper methods
    def _calculate_snr(self, intensities: np.ndarray, peak_idx: int, window: int = 15) -> float:
        """Calculate signal-to-noise ratio"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        local_region = intensities[start:end]
        peak_value = intensities[peak_idx]
        
        noise_level = np.std(local_region)
        return peak_value / noise_level if noise_level > 0 else float('inf')
    
    def _find_mound_boundaries(self, intensities: np.ndarray, peak_idx: int) -> Tuple[int, int]:
        """Find mound boundaries at 10% of peak height"""
        threshold = intensities[peak_idx] * 0.1
        
        start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if intensities[i] < threshold:
                start_idx = i
                break
        
        end_idx = peak_idx
        for i in range(peak_idx + 1, len(intensities)):
            if intensities[i] < threshold:
                end_idx = i
                break
        
        return start_idx, end_idx
    
    def _find_connected_regions(self, mask: np.ndarray, min_width: int) -> List[Tuple[int, int]]:
        """Find connected regions in boolean mask"""
        regions = []
        start = None
        
        for i, value in enumerate(mask):
            if value and start is None:
                start = i
            elif not value and start is not None:
                if i - start >= min_width:
                    regions.append((start, i - 1))
                start = None
        
        if start is not None and len(mask) - start >= min_width:
            regions.append((start, len(mask) - 1))
        
        return regions
    
    def _summarize_features(self, features: List[BSpectralFeature]) -> Dict:
        """Generate standardized feature summary"""
        feature_counts = {}
        for feature in features:
            key = f"{feature.feature_type}_{feature.point_type}"
            feature_counts[key] = feature_counts.get(key, 0) + 1
        
        return {
            'detailed_counts': feature_counts,
            'by_type': {ft: len([f for f in features if f.feature_type == ft]) 
                       for ft in set(f.feature_type for f in features)},
            'total_points': len(features),
            'wavelength_range': (min(f.wavelength for f in features), 
                               max(f.wavelength for f in features)) if features else (0, 0)
        }

def load_b_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load B spectrum from file"""
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
        wavelengths = data.iloc[:, 0].values
        intensities = data.iloc[:, 1].values
    else:
        data = np.loadtxt(filepath, delimiter='\t')
        wavelengths = data[:, 0]
        intensities = data[:, 1]
    
    return wavelengths, intensities

if __name__ == "__main__":
    print("🔧 GEMINI B PEAK DETECTOR - Main Detection Logic v3.0")
    print("File: gemini_Bpeak_detector.py")
    print("Point structures match manual marking exactly:")
    print("Peak(3), Plateau(3), Shoulder(3), Trough(3), Mound(4+Summary), Baseline(2), Diagnostic(2), Valley(1)")
