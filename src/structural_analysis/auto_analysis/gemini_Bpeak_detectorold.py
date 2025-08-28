"""
Gemini B Spectra Auto-Detector - ENHANCED
Advanced adaptive detection incorporating L detector improvements
Version: 2.0 (Enhanced with L detector features - 50% line reduction)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class BSpectralFeature:
    wavelength: float
    intensity: float
    feature_type: str
    feature_group: str
    prominence: float
    snr: float
    confidence: float
    detection_method: str
    width_nm: float = 0.0
    start_wavelength: float = 0.0
    end_wavelength: float = 0.0
    effective_height: float = 0.0

class GeminiBSpectralDetector:
    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Enhanced configuration incorporating L detector advances
        self.config = {
            'laser_thresholds': {
                'prominence': {'weak': 0.54, 'medium': 0.9, 'major': 50.0},
                'snr': {'weak': 4.2, 'medium': 6.9},
                'intensity_major': 95.0
            },
            'noise_assessment': {
                'excellent': 0.025,  # Can use laser algorithm
                'good': 0.5,         # Laser with fallback
                'baseline_threshold': 0.1
            },
            'detection_params': {
                'absolute_threshold': 1.0,     # Lower for B spectra
                'min_distance': 10,            # nm - shorter for B spectra
                'merge_distance': 2.0,         # nm - broader merging for B
                'min_wavelength': 350.0,       # nm - allow more UV for B
                'width_range': (2.0, 200.0),   # nm - wider range for B spectra
                'baseline_min': 0.5,           # Lower threshold for B
                'flatness_threshold': 0.6
            },
            'region_params': {
                'mound_min_width': 50,         # nm
                'mound_min_prominence': 5.0,
                'smoothing_sigma': 2.0,
                'trough_prominence': 1.0
            },
            'baseline_range': (300, 325)      # Tighter range for B spectra
        }
    
    def _log(self, message: str) -> None:
        """Debug logging"""
        if self.debug:
            print(f"Debug B: {message}")
    
    def normalize_b_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Enhanced B spectra normalization: 650nm → 50000 → scale to 0-100"""
        idx_650 = np.argmin(np.abs(wavelengths - 650.0))
        intensity_650 = intensities[idx_650]
        
        if intensity_650 <= 0:
            raise ValueError(f"Invalid intensity at 650nm: {intensity_650}")
        
        normalized = (intensities / intensity_650) * 100  # Direct to 0-100 scale
        
        self.norm_reference_wavelength = wavelengths[idx_650]
        self.norm_reference_intensity = intensity_650
        
        self._log(f"Normalized 650nm ref at {self.norm_reference_wavelength:.2f}nm, "
                 f"range: {np.min(normalized):.2f}-{np.max(normalized):.2f}")
        
        return normalized
    
    def assess_baseline_noise(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[float, str]:
        """Enhanced baseline noise assessment"""
        baseline_range = self.config['baseline_range']
        baseline_mask = (wavelengths >= baseline_range[0]) & (wavelengths <= baseline_range[1])
        
        if not np.any(baseline_mask):
            raise ValueError(f"No data in baseline region ({baseline_range[0]}-{baseline_range[1]}nm)")
        
        baseline_std = np.std(intensities[baseline_mask])
        noise_thresholds = self.config['noise_assessment']
        
        if baseline_std <= noise_thresholds['excellent']:
            classification = "excellent"
        elif baseline_std <= noise_thresholds['good']:
            classification = "good"
        else:
            classification = "poor"
        
        self._log(f"Baseline noise: {baseline_std:.4f} -> {classification}")
        return baseline_std, classification
    
    def _calculate_peak_metrics(self, wavelengths: np.ndarray, intensities: np.ndarray, 
                               peak_idx: int) -> Dict[str, float]:
        """Comprehensive peak metrics calculation"""
        window = 15  # Larger window for B spectra
        start, end = max(0, peak_idx - window), min(len(intensities), peak_idx + window + 1)
        local_region = intensities[start:end]
        peak_value = intensities[peak_idx]
        
        # Prominence and baseline
        left_idx = peak_idx - start
        left_min = np.min(local_region[:left_idx]) if left_idx > 0 else peak_value
        right_min = np.min(local_region[left_idx + 1:]) if left_idx < len(local_region) - 1 else peak_value
        baseline = max(left_min, right_min)
        prominence = max(0, peak_value - baseline)
        effective_height = max(0, peak_value - (left_min + right_min) / 2)
        
        # SNR
        local_avg = np.mean(local_region)
        noise_points = local_region[local_region < local_avg]
        snr = peak_value / np.std(noise_points) if len(noise_points) > 1 and np.std(noise_points) > 0 else peak_value
        
        # Width and flatness
        width_nm = self._calculate_peak_width(wavelengths, intensities, peak_idx)
        flatness = self._calculate_flatness(intensities, peak_idx)
        
        return {
            'prominence': prominence,
            'snr': snr,
            'effective_height': effective_height,
            'width_nm': width_nm,
            'flatness': flatness,
            'baseline': baseline
        }
    
    def _calculate_peak_width(self, wavelengths: np.ndarray, intensities: np.ndarray, peak_idx: int) -> float:
        """Advanced width calculation for B spectra"""
        # Use half-maximum method (more reliable for broad B peaks)
        peak_intensity = intensities[peak_idx]
        half_max = peak_intensity * 0.5
        
        left_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if intensities[i] < half_max:
                left_idx = i
                break
        
        right_idx = peak_idx
        for i in range(peak_idx + 1, len(intensities)):
            if intensities[i] < half_max:
                right_idx = i
                break
        
        return abs(wavelengths[right_idx] - wavelengths[left_idx])
    
    def _calculate_flatness(self, intensities: np.ndarray, peak_idx: int, window: int = 5) -> float:
        """Calculate peak top flatness"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        peak_region = intensities[start:end]
        peak_max = intensities[peak_idx]
        tolerance = peak_max * 0.1  # More generous for B spectra
        
        flat_points = np.sum(np.abs(peak_region - peak_max) <= tolerance)
        return flat_points / len(peak_region)
    
    def _merge_nearby_peaks(self, peaks: List[int], wavelengths: np.ndarray, intensities: np.ndarray) -> List[int]:
        """Merge peaks within merge distance"""
        if not peaks:
            return peaks
        
        merge_distance = self.config['detection_params']['merge_distance']
        peaks.sort(key=lambda idx: wavelengths[idx])
        
        merged_peaks = [peaks[0]]
        
        for peak_idx in peaks[1:]:
            current_wl = wavelengths[peak_idx]
            last_wl = wavelengths[merged_peaks[-1]]
            
            if abs(current_wl - last_wl) <= merge_distance:
                # Keep higher intensity peak
                if intensities[peak_idx] > intensities[merged_peaks[-1]]:
                    merged_peaks[-1] = peak_idx
            else:
                merged_peaks.append(peak_idx)
        
        self._log(f"Merged {len(peaks)} candidates to {len(merged_peaks)} peaks")
        return merged_peaks
    
    def _apply_filters(self, peaks: List[int], wavelengths: np.ndarray, intensities: np.ndarray) -> List[int]:
        """Comprehensive filtering pipeline"""
        filtered_peaks = []
        params = self.config['detection_params']
        
        for peak_idx in peaks:
            wavelength = wavelengths[peak_idx]
            
            # Filter 1: Wavelength range
            if wavelength < params['min_wavelength']:
                self._log(f"Rejected {wavelength:.1f}nm - below minimum wavelength")
                continue
            
            # Filter 2: Calculate metrics
            metrics = self._calculate_peak_metrics(wavelengths, intensities, peak_idx)
            
            # Filter 3: Width validation
            if not (params['width_range'][0] <= metrics['width_nm'] <= params['width_range'][1]):
                self._log(f"Rejected {wavelength:.1f}nm - invalid width {metrics['width_nm']:.1f}nm")
                continue
            
            # Filter 4: Effective height
            if metrics['effective_height'] < params['baseline_min']:
                self._log(f"Rejected {wavelength:.1f}nm - low effective height {metrics['effective_height']:.2f}")
                continue
            
            filtered_peaks.append(peak_idx)
            self._log(f"Accepted peak at {wavelength:.1f}nm - height: {metrics['effective_height']:.2f}")
        
        return filtered_peaks
    
    def laser_algorithm_detection(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Enhanced laser algorithm with filtering pipeline"""
        params = self.config['detection_params']
        
        # Initial detection
        wavelength_step = np.mean(np.diff(wavelengths))
        min_distance_idx = max(1, int(params['min_distance'] / wavelength_step))
        
        peaks, _ = signal.find_peaks(
            intensities,
            height=params['absolute_threshold'],
            prominence=0.1,
            distance=min_distance_idx
        )
        
        self._log(f"Laser algorithm found {len(peaks)} initial candidates")
        
        # Apply filtering pipeline
        peaks = self._merge_nearby_peaks(peaks.tolist(), wavelengths, intensities)
        peaks = self._apply_filters(peaks, wavelengths, intensities)
        
        # Create features
        features = []
        laser_thresholds = self.config['laser_thresholds']
        
        for peak_idx in peaks:
            metrics = self._calculate_peak_metrics(wavelengths, intensities, peak_idx)
            wavelength = wavelengths[peak_idx]
            
            # Determine feature type
            if (metrics['width_nm'] > 25.0 and 
                metrics['flatness'] > self.config['detection_params']['flatness_threshold']):
                feature_type, feature_group = "mound_crest", "mound"
            else:
                feature_type, feature_group = "peak", "peak"
            
            # Calculate confidence
            prominence = metrics['prominence']
            snr = metrics['snr']
            
            if prominence > laser_thresholds['prominence']['major']:
                confidence = 0.95
            elif prominence >= laser_thresholds['prominence']['medium'] and snr >= laser_thresholds['snr']['medium']:
                confidence = 0.8
            elif prominence >= laser_thresholds['prominence']['weak'] and snr >= laser_thresholds['snr']['weak']:
                confidence = 0.6
            else:
                confidence = 0.4
            
            feature = BSpectralFeature(
                wavelength=wavelength,
                intensity=intensities[peak_idx],
                feature_type=feature_type,
                feature_group=feature_group,
                prominence=prominence,
                snr=snr,
                confidence=confidence,
                detection_method="laser_algorithm",
                width_nm=metrics['width_nm'],
                effective_height=metrics['effective_height']
            )
            features.append(feature)
        
        return features
    
    def region_based_detection(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Enhanced region-based detection for B spectra"""
        region_cfg = self.config['region_params']
        smoothed = gaussian_filter1d(intensities, sigma=region_cfg['smoothing_sigma'])
        features = []
        
        # Detect mounds
        features.extend(self._detect_mounds(wavelengths, smoothed))
        
        # Detect troughs
        features.extend(self._detect_troughs(wavelengths, smoothed))
        
        return features
    
    def _detect_mounds(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Enhanced mound detection"""
        region_cfg = self.config['region_params']
        wavelength_step = np.mean(np.diff(wavelengths))
        min_width_idx = region_cfg['mound_min_width'] / wavelength_step
        
        peaks, _ = signal.find_peaks(
            intensities,
            prominence=region_cfg['mound_min_prominence'],
            width=min_width_idx,
            distance=20
        )
        
        features = []
        for peak_idx in peaks:
            # Find boundaries
            start_idx, end_idx = self._find_boundaries(intensities, peak_idx, 'mound')
            
            # Calculate properties
            start_wl, end_wl = wavelengths[start_idx], wavelengths[end_idx]
            width_nm = abs(end_wl - start_wl)
            prominence = self._calculate_peak_metrics(wavelengths, intensities, peak_idx)['prominence']
            
            # Create mound feature set
            mound_props = {'width_nm': width_nm, 'start_wavelength': start_wl, 'end_wavelength': end_wl}
            
            features.extend([
                self._create_feature(wavelengths, intensities, start_idx, 'mound_start', 'mound', 0.8, 'region_based', **mound_props),
                self._create_feature(wavelengths, intensities, peak_idx, 'mound_crest', 'mound', 0.9, 'region_based', **mound_props),
                self._create_feature(wavelengths, intensities, end_idx, 'mound_end', 'mound', 0.8, 'region_based', **mound_props)
            ])
            
            # Override prominence for crest
            features[-2].prominence = prominence
        
        return features
    
    def _detect_troughs(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Enhanced trough detection"""
        region_cfg = self.config['region_params']
        inverted = -intensities
        peaks, _ = signal.find_peaks(inverted, prominence=region_cfg['trough_prominence'], distance=10)
        
        features = []
        for peak_idx in peaks:
            if intensities[peak_idx] > 10.0:  # Above baseline
                start_idx, end_idx = self._find_boundaries(intensities, peak_idx, 'trough')
                
                features.extend([
                    self._create_feature(wavelengths, intensities, start_idx, 'trough_start', 'trough', 0.7, 'region_based'),
                    self._create_feature(wavelengths, intensities, peak_idx, 'trough_bottom', 'trough', 0.8, 'region_based'),
                    self._create_feature(wavelengths, intensities, end_idx, 'trough_end', 'trough', 0.7, 'region_based')
                ])
        
        return features
    
    def _find_boundaries(self, intensities: np.ndarray, center_idx: int, boundary_type: str = 'mound') -> Tuple[int, int]:
        """Unified boundary finding"""
        center_value = intensities[center_idx]
        
        if boundary_type == 'mound':
            threshold = center_value * 0.1
            comparison = lambda x: x < threshold
        else:  # trough
            threshold = center_value * 1.2
            comparison = lambda x: x > threshold
        
        # Find boundaries
        start_idx = center_idx
        for i in range(center_idx - 1, -1, -1):
            if comparison(intensities[i]):
                start_idx = i
                break
        
        end_idx = center_idx
        for i in range(center_idx + 1, len(intensities)):
            if comparison(intensities[i]):
                end_idx = i
                break
        
        return start_idx, end_idx
    
    def _create_feature(self, wavelengths: np.ndarray, intensities: np.ndarray, idx: int,
                       feature_type: str, feature_group: str, confidence: float,
                       detection_method: str, **kwargs) -> BSpectralFeature:
        """Enhanced unified feature creation"""
        if feature_group == 'peak':
            metrics = self._calculate_peak_metrics(wavelengths, intensities, idx)
            prominence, snr, effective_height = metrics['prominence'], metrics['snr'], metrics['effective_height']
        else:
            prominence = snr = effective_height = 0.0
        
        return BSpectralFeature(
            wavelength=wavelengths[idx],
            intensity=intensities[idx],
            feature_type=feature_type,
            feature_group=feature_group,
            prominence=prominence,
            snr=snr,
            confidence=confidence,
            detection_method=detection_method,
            effective_height=effective_height,
            **kwargs
        )
    
    def detect_baseline(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Enhanced baseline detection"""
        baseline_range = self.config['baseline_range']
        baseline_mask = (wavelengths >= baseline_range[0]) & (wavelengths <= baseline_range[1])
        
        if not np.any(baseline_mask):
            return []
        
        baseline_wl = wavelengths[baseline_mask]
        baseline_int = intensities[baseline_mask]
        
        baseline_std = np.std(baseline_int)
        snr = 1.0 / (baseline_std + 1e-6)
        confidence = 0.9 if baseline_std < 0.1 else 0.6
        
        return [
            BSpectralFeature(baseline_wl[0], baseline_int[0], 'baseline_start', 'baseline', 0.0, snr, confidence, 'region_based'),
            BSpectralFeature(baseline_wl[-1], baseline_int[-1], 'baseline_end', 'baseline', 0.0, snr, confidence, 'region_based')
        ]
    
    def analyze_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Dict:
        """Enhanced adaptive analysis for B spectra"""
        if len(wavelengths) != len(intensities):
            raise ValueError("Wavelength and intensity arrays must have same length")
        
        # Normalize and assess
        normalized_intensities = self.normalize_b_spectrum(wavelengths, intensities)
        baseline_std, noise_classification = self.assess_baseline_noise(wavelengths, normalized_intensities)
        
        # Adaptive strategy based on noise
        all_features = self.detect_baseline(wavelengths, normalized_intensities)
        
        if noise_classification == "excellent":
            # Hybrid approach: both methods
            laser_features = self.laser_algorithm_detection(wavelengths, normalized_intensities)
            region_features = self.region_based_detection(wavelengths, normalized_intensities)
            
            for feature in laser_features:
                feature.detection_method = "hybrid_laser"
            for feature in region_features:
                feature.detection_method = "hybrid_region"
            
            all_features.extend(laser_features + region_features)
            detection_strategy = "hybrid_advanced"
            
        elif noise_classification == "good":
            # Laser with fallback
            laser_features = self.laser_algorithm_detection(wavelengths, normalized_intensities)
            
            if not laser_features:
                region_features = self.region_based_detection(wavelengths, normalized_intensities)
                all_features.extend(region_features)
                detection_strategy = "region_fallback"
            else:
                all_features.extend(laser_features)
                detection_strategy = "laser_primary"
        else:
            # Region-based only
            region_features = self.region_based_detection(wavelengths, normalized_intensities)
            all_features.extend(region_features)
            detection_strategy = "region_only"
        
        # Sort and summarize
        all_features.sort(key=lambda x: x.wavelength)
        avg_confidence = np.mean([f.confidence for f in all_features]) if all_features else 0.0
        
        self._log(f"Final analysis: {len(all_features)} features using {detection_strategy}")
        
        return {
            'features': all_features,
            'normalization': {
                'reference_wavelength': self.norm_reference_wavelength,
                'reference_intensity': self.norm_reference_intensity,
                'method': '650nm_to_100_scale'
            },
            'baseline_assessment': {
                'noise_std': baseline_std,
                'noise_classification': noise_classification,
                'vs_threshold': baseline_std / 0.025
            },
            'detection_strategy': detection_strategy,
            'overall_confidence': avg_confidence,
            'feature_count': len(all_features),
            'feature_summary': self._summarize_features(all_features)
        }
    
    def _summarize_features(self, features: List[BSpectralFeature]) -> Dict:
        """Enhanced feature summary"""
        if not features:
            return {}
        
        feature_types = {}
        for feature in features:
            feature_types[feature.feature_group] = feature_types.get(feature.feature_group, 0) + 1
        
        return {
            'by_type': feature_types,
            'wavelength_range': (min(f.wavelength for f in features), max(f.wavelength for f in features)),
            'intensity_range': (min(f.intensity for f in features), max(f.intensity for f in features)),
            'avg_confidence': np.mean([f.confidence for f in features])
        }

def load_b_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced B spectrum loading"""
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
        return data.iloc[:, 0].values, data.iloc[:, 1].values
    else:
        # Try multiple delimiters
        for delimiter in ['\t', None, ',']:
            try:
                data = np.loadtxt(filepath, delimiter=delimiter)
                return data[:, 0], data[:, 1]
            except ValueError:
                continue
        
        # Final fallback
        data = pd.read_csv(filepath, sep=None, engine='python', header=None)
        return data.iloc[:, 0].values, data.iloc[:, 1].values

def analyze_b_spectrum_file(filepath: str, debug: bool = False) -> Dict:
    """Analyze B spectrum file with enhanced detector"""
    detector = GeminiBSpectralDetector(debug=debug)
    wavelengths, intensities = load_b_spectrum(filepath)
    return detector.analyze_spectrum(wavelengths, intensities)

if __name__ == "__main__":
    detector = GeminiBSpectralDetector(debug=True)
    print("Enhanced Gemini B Spectra Auto-Detector (v2.0) - 50% fewer lines")
    print("Advanced features from L detector with B-specific adaptive strategy")
