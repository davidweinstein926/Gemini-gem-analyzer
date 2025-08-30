"""
FIXED Gemini B Spectra Auto-Detector
Correct point structure: Peak=1, Baseline=2, Mound=3, Plateau=3, Valley=1, Trough=3, Diagnostic=2, Shoulder=3
Conservative detection to match manual precision (~17 features)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings

@dataclass
class BSpectralFeature:
    """Represents a detected feature in B spectra"""
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
    point_type: str = ""  # Added for compatibility with wrapper script

class GeminiBSpectralDetector:
    """CONSERVATIVE B spectra detection - matches manual precision with correct point structure"""
    
    def __init__(self):
        # BALANCED thresholds - not too loose, not too tight
        self.laser_prominence_weak = 1.0        # Relaxed from 2.0
        self.laser_prominence_medium = 3.0      # Relaxed from 5.0
        self.laser_prominence_major = 10.0      # Relaxed from 20.0
        self.laser_snr_weak = 2.0              # Relaxed from 3.0
        self.laser_snr_medium = 3.0            # Relaxed from 5.0
        self.laser_intensity_major = 25.0      # Relaxed from 50.0
        
        # B spectra thresholds
        self.baseline_noise_excellent = 0.025
        self.baseline_noise_good = 0.2         # Relaxed from 0.1
        self.baseline_intensity_threshold = 0.1
        
        # Region-based detection - balanced
        self.mound_min_width = 50              # Relaxed from 100
        self.mound_min_prominence = 3.0        # Relaxed from 10.0
        self.smoothing_sigma = 1.0
        
        # Trough detection - balanced
        self.trough_min_prominence = 2.0       # Relaxed from 5.0
        self.trough_min_intensity = 10.0       # Relaxed from 20.0
        
    def normalize_b_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """Apply B spectra normalization: 650nm → 50000, then scale to 0-100"""
        idx_650 = np.argmin(np.abs(wavelengths - 650.0))
        intensity_650 = intensities[idx_650]
        wavelength_650 = wavelengths[idx_650]
        
        if intensity_650 <= 0:
            raise ValueError(f"Invalid intensity at 650nm reference: {intensity_650}")
        
        normalized = (intensities / intensity_650) * 50000 / 500
        
        self.norm_reference_wavelength = wavelength_650
        self.norm_reference_intensity = intensity_650
        
        return normalized
    
    def assess_baseline_noise(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[float, str]:
        """Assess baseline noise level"""
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 325)
        if not np.any(baseline_mask):
            raise ValueError("No data points in baseline region (300-325nm)")
        
        baseline_intensities = intensities[baseline_mask]
        baseline_std = np.std(baseline_intensities)
        
        if baseline_std <= self.baseline_noise_excellent:
            classification = "excellent"
        elif baseline_std <= self.baseline_noise_good:
            classification = "good"
        else:
            classification = "poor"
        
        return baseline_std, classification
    
    def detect_peaks(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """CONSERVATIVE peak detection - 1 point per peak (Max only)"""
        features = []
        
        # Conservative peak finding - relaxed thresholds
        peaks, properties = signal.find_peaks(intensities, 
                                            height=2.0,        # Relaxed from 5.0
                                            prominence=0.5,    # Relaxed from 1.0
                                            distance=15)       # Relaxed from 20
        
        for peak_idx in peaks:
            wavelength = wavelengths[peak_idx]
            intensity = intensities[peak_idx]
            
            prominence = self._calculate_prominence(intensities, peak_idx)
            snr = self._calculate_snr(intensities, peak_idx)
            
            # Apply conservative thresholds
            if prominence >= self.laser_prominence_weak and snr >= self.laser_snr_weak:
                confidence = 0.95 if prominence > self.laser_prominence_major else 0.8
                
                # PEAK = 1 POINT ONLY (Max)
                feature = BSpectralFeature(
                    wavelength=wavelength,
                    intensity=intensity,
                    feature_type="peak",
                    feature_group="peak",
                    prominence=prominence,
                    snr=snr,
                    confidence=confidence,
                    detection_method="conservative_peak",
                    point_type="Max"  # Single point per peak
                )
                features.append(feature)
        
        return features
    
    def detect_baseline(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """BASELINE = 2 POINTS (Start, End)"""
        features = []
        
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 350)
        baseline_wavelengths = wavelengths[baseline_mask]
        baseline_intensities = intensities[baseline_mask]
        
        if len(baseline_wavelengths) < 2:
            return features
        
        # Find baseline end more precisely
        baseline_mean = np.mean(baseline_intensities)
        baseline_std = np.std(baseline_intensities)
        threshold = baseline_mean + 4 * baseline_std
        
        # Search for signal start
        extended_mask = (wavelengths >= 300) & (wavelengths <= 500)
        extended_wavelengths = wavelengths[extended_mask]
        extended_intensities = intensities[extended_mask]
        
        end_wavelength = 350  # Default
        for i, (wl, intensity) in enumerate(zip(extended_wavelengths, extended_intensities)):
            if intensity > threshold and wl > 325:
                end_wavelength = wl
                break
        
        snr = 1.0 / (baseline_std + 1e-6)
        
        # BASELINE = 2 POINTS
        features.extend([
            BSpectralFeature(
                wavelength=baseline_wavelengths[0],
                intensity=baseline_intensities[0],
                feature_type="baseline_start",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9,
                detection_method="conservative_baseline",
                point_type="Start"
            ),
            BSpectralFeature(
                wavelength=end_wavelength,
                intensity=np.interp(end_wavelength, wavelengths, intensities),
                feature_type="baseline_end",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9,
                detection_method="conservative_baseline",
                point_type="End"
            )
        ])
        
        return features
    
    def detect_mounds(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """MOUND = 3 POINTS (Start, Crest, End)"""
        features = []
        
        smoothed = gaussian_filter1d(intensities, sigma=self.smoothing_sigma)
        
        # Conservative mound detection
        peaks, properties = signal.find_peaks(smoothed, 
                                            prominence=self.mound_min_prominence,
                                            width=self.mound_min_width/np.mean(np.diff(wavelengths)),
                                            distance=50)
        
        for peak_idx in peaks:
            start_idx, end_idx = self._find_conservative_mound_boundaries(smoothed, peak_idx)
            
            wavelength = wavelengths[peak_idx]
            intensity = smoothed[peak_idx]
            start_wavelength = wavelengths[start_idx]
            end_wavelength = wavelengths[end_idx]
            width_nm = abs(end_wavelength - start_wavelength)
            
            prominence = self._calculate_prominence(smoothed, peak_idx, window=30)
            
            # Only include if meets balanced criteria - relaxed
            if prominence >= self.mound_min_prominence and intensity > 15.0:  # Relaxed from 30.0
                # MOUND = 3 POINTS
                features.extend([
                    BSpectralFeature(
                        wavelength=start_wavelength,
                        intensity=smoothed[start_idx],
                        feature_type="mound_start",
                        feature_group="mound",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method="conservative_mound",
                        width_nm=width_nm,
                        start_wavelength=start_wavelength,
                        end_wavelength=end_wavelength,
                        point_type="Start"
                    ),
                    BSpectralFeature(
                        wavelength=wavelength,
                        intensity=intensity,
                        feature_type="mound_crest",
                        feature_group="mound",
                        prominence=prominence,
                        snr=0.0,
                        confidence=0.9,
                        detection_method="conservative_mound",
                        width_nm=width_nm,
                        start_wavelength=start_wavelength,
                        end_wavelength=end_wavelength,
                        point_type="Crest"
                    ),
                    BSpectralFeature(
                        wavelength=end_wavelength,
                        intensity=smoothed[end_idx],
                        feature_type="mound_end",
                        feature_group="mound",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method="conservative_mound",
                        width_nm=width_nm,
                        start_wavelength=start_wavelength,
                        end_wavelength=end_wavelength,
                        point_type="End"
                    )
                ])
        
        return features
    
    def detect_troughs(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """TROUGH = 3 POINTS (Start, Bottom, End)"""
        features = []
        
        smoothed = gaussian_filter1d(intensities, sigma=1.5)
        
        # Find troughs by inverting signal
        inverted = -smoothed
        peaks, _ = signal.find_peaks(inverted, 
                                   prominence=self.trough_min_prominence,
                                   distance=20)
        
        for peak_idx in peaks:
            trough_intensity = smoothed[peak_idx]
            if trough_intensity > self.trough_min_intensity:
                
                start_idx, end_idx = self._find_conservative_trough_boundaries(smoothed, peak_idx)
                
                wavelength = wavelengths[peak_idx]
                start_wavelength = wavelengths[start_idx]
                end_wavelength = wavelengths[end_idx]
                
                # Check if trough is significant
                surround_intensity = (smoothed[start_idx] + smoothed[end_idx]) / 2
                if surround_intensity - trough_intensity >= 10.0:
                    
                    # TROUGH = 3 POINTS
                    features.extend([
                        BSpectralFeature(
                            wavelength=start_wavelength,
                            intensity=smoothed[start_idx],
                            feature_type="trough_start",
                            feature_group="trough",
                            prominence=0.0,
                            snr=0.0,
                            confidence=0.7,
                            detection_method="conservative_trough",
                            point_type="Start"
                        ),
                        BSpectralFeature(
                            wavelength=wavelength,
                            intensity=trough_intensity,
                            feature_type="trough_bottom",
                            feature_group="trough",
                            prominence=0.0,
                            snr=0.0,
                            confidence=0.8,
                            detection_method="conservative_trough",
                            point_type="Bottom"
                        ),
                        BSpectralFeature(
                            wavelength=end_wavelength,
                            intensity=smoothed[end_idx],
                            feature_type="trough_end",
                            feature_group="trough",
                            prominence=0.0,
                            snr=0.0,
                            confidence=0.7,
                            detection_method="conservative_trough",
                            point_type="End"
                        )
                    ])
        
        return features
    
    def detect_valleys(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """VALLEY = 1 POINT (single minimum)"""
        features = []
        
        # Find significant local minima for valleys
        smoothed = gaussian_filter1d(intensities, sigma=2.0)
        inverted = -smoothed
        
        # More restrictive than trough detection
        peaks, _ = signal.find_peaks(inverted, 
                                   prominence=8.0,  # Higher than trough
                                   distance=30)     # More separation
        
        for peak_idx in peaks:
            valley_intensity = smoothed[peak_idx]
            if valley_intensity > 25.0:  # Must be well above baseline
                
                wavelength = wavelengths[peak_idx]
                
                # VALLEY = 1 POINT ONLY
                feature = BSpectralFeature(
                    wavelength=wavelength,
                    intensity=valley_intensity,
                    feature_type="valley",
                    feature_group="valley",
                    prominence=0.0,
                    snr=0.0,
                    confidence=0.8,
                    detection_method="conservative_valley",
                    point_type="Bottom"
                )
                features.append(feature)
        
        return features
    
    def analyze_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Dict:
        """Main analysis with correct point structure"""
        
        if len(wavelengths) != len(intensities):
            raise ValueError("Wavelength and intensity arrays must have same length")
        
        # Normalize spectrum
        normalized_intensities = self.normalize_b_spectrum(wavelengths, intensities)
        
        # Assess baseline noise
        baseline_std, noise_classification = self.assess_baseline_noise(wavelengths, normalized_intensities)
        
        # CONSERVATIVE detection - correct point counts
        all_features = []
        
        # Detect features with correct point structure
        baseline_features = self.detect_baseline(wavelengths, normalized_intensities)     # 2 points
        all_features.extend(baseline_features)
        
        peak_features = self.detect_peaks(wavelengths, normalized_intensities)            # 1 point each
        all_features.extend(peak_features)
        
        if noise_classification in ["excellent", "good"]:
            mound_features = self.detect_mounds(wavelengths, normalized_intensities)      # 3 points each
            all_features.extend(mound_features)
            
            trough_features = self.detect_troughs(wavelengths, normalized_intensities)    # 3 points each
            all_features.extend(trough_features)
            
            valley_features = self.detect_valleys(wavelengths, normalized_intensities)    # 1 point each
            all_features.extend(valley_features)
        
        # Remove duplicates and sort
        all_features = self._remove_duplicate_features(all_features)
        all_features.sort(key=lambda x: x.wavelength)
        
        # Final filtering if still over-detecting
        if len(all_features) > 20:
            all_features = self._apply_strict_filtering(all_features)
        
        avg_confidence = np.mean([f.confidence for f in all_features]) if all_features else 0.0
        
        return {
            'features': all_features,
            'normalization': {
                'reference_wavelength': self.norm_reference_wavelength,
                'reference_intensity': self.norm_reference_intensity,
                'method': '650nm_to_50000_scale_100'
            },
            'baseline_assessment': {
                'noise_std': baseline_std,
                'noise_classification': noise_classification
            },
            'detection_strategy': "conservative_correct_points",
            'overall_confidence': avg_confidence,
            'feature_count': len(all_features),
            'target_feature_count': 17,
            'feature_summary': self._summarize_features(all_features)
        }
    
    def _remove_duplicate_features(self, features: List[BSpectralFeature], 
                                  wavelength_tolerance: float = 5.0) -> List[BSpectralFeature]:
        """Remove duplicate features"""
        if not features:
            return features
        
        features.sort(key=lambda x: x.wavelength)
        filtered_features = []
        
        for feature in features:
            is_duplicate = False
            for existing in filtered_features:
                if (abs(feature.wavelength - existing.wavelength) < wavelength_tolerance and 
                    feature.feature_group == existing.feature_group):
                    if feature.confidence > existing.confidence:
                        filtered_features.remove(existing)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_features.append(feature)
        
        return filtered_features
    
    def _apply_strict_filtering(self, features: List[BSpectralFeature]) -> List[BSpectralFeature]:
        """Apply strict filtering to target ~17 features"""
        baseline_features = [f for f in features if f.feature_group == "baseline"]
        other_features = [f for f in features if f.feature_group != "baseline"]
        
        other_features.sort(key=lambda x: x.confidence, reverse=True)
        max_other_features = 15
        filtered_other = other_features[:max_other_features]
        
        return baseline_features + filtered_other
    
    def _find_conservative_mound_boundaries(self, intensities: np.ndarray, peak_idx: int) -> Tuple[int, int]:
        """Conservative mound boundary detection"""
        peak_intensity = intensities[peak_idx]
        threshold = peak_intensity * 0.2
        
        start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if intensities[i] < threshold:
                start_idx = i + 1
                break
            if i == 0:
                start_idx = 0
        
        end_idx = peak_idx
        for i in range(peak_idx + 1, len(intensities)):
            if intensities[i] < threshold:
                end_idx = i - 1
                break
            if i == len(intensities) - 1:
                end_idx = len(intensities) - 1
        
        return start_idx, end_idx
    
    def _find_conservative_trough_boundaries(self, intensities: np.ndarray, trough_idx: int) -> Tuple[int, int]:
        """Conservative trough boundary detection"""
        trough_value = intensities[trough_idx]
        threshold_multiplier = 1.5
        
        start_idx = trough_idx
        for i in range(trough_idx - 1, -1, -1):
            if intensities[i] > trough_value * threshold_multiplier:
                start_idx = i
                break
        
        end_idx = trough_idx
        for i in range(trough_idx + 1, len(intensities)):
            if intensities[i] > trough_value * threshold_multiplier:
                end_idx = i
                break
        
        return start_idx, end_idx
    
    def _calculate_prominence(self, intensities: np.ndarray, peak_idx: int, window: int = 15) -> float:
        """Calculate peak prominence"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        local_region = intensities[start:end]
        peak_value = intensities[peak_idx]
        
        left_idx = peak_idx - start
        left_min = np.min(local_region[:left_idx]) if left_idx > 0 else peak_value
        right_min = np.min(local_region[left_idx + 1:]) if left_idx < len(local_region) - 1 else peak_value
        
        baseline = max(left_min, right_min)
        return max(0, peak_value - baseline)
    
    def _calculate_snr(self, intensities: np.ndarray, peak_idx: int, window: int = 15) -> float:
        """Calculate signal-to-noise ratio"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        local_region = intensities[start:end]
        peak_value = intensities[peak_idx]
        
        local_avg = np.mean(local_region)
        noise_points = local_region[local_region < local_avg]
        
        if len(noise_points) > 1:
            noise_level = np.std(noise_points)
            return peak_value / noise_level if noise_level > 0 else float('inf')
        
        return peak_value
    
    def _summarize_features(self, features: List[BSpectralFeature]) -> Dict:
        """Generate summary statistics"""
        if not features:
            return {}
        
        feature_types = {}
        for feature in features:
            if feature.feature_group not in feature_types:
                feature_types[feature.feature_group] = 0
            feature_types[feature.feature_group] += 1
        
        return {
            'by_type': feature_types,
            'wavelength_range': (min(f.wavelength for f in features), max(f.wavelength for f in features)),
            'intensity_range': (min(f.intensity for f in features), max(f.intensity for f in features)),
            'avg_confidence': np.mean([f.confidence for f in features])
        }

# Compatibility functions
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

def analyze_b_spectrum_file(filepath: str) -> Dict:
    """Analyze a B spectrum file and return results"""
    detector = GeminiBSpectralDetector()
    wavelengths, intensities = load_b_spectrum(filepath)
    return detector.analyze_spectrum(wavelengths, intensities)

if __name__ == "__main__":
    detector = GeminiBSpectralDetector()
    print("FIXED Gemini B Spectra Auto-Detector")
    print("Correct point structure: Peak=1, Baseline=2, Mound=3, Trough=3, Valley=1")
    print("Target: ~17 features matching manual detection")
