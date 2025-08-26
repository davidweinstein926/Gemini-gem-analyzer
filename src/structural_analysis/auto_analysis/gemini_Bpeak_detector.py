"""
Gemini B Spectra Auto-Detector
Adaptive detection algorithm for halogen/broadband transmission spectra

Based on analysis of 5 B spectra samples:
- 190BP2: High noise (20.1), incompatible with laser algorithm
- 189BP2: High noise (19.4), incompatible with laser algorithm  
- 79BC1: Medium noise (0.32), partial laser compatibility
- 92BC1: Low noise (0.032), mixed laser compatibility  
- 214BC1: Medium noise (0.332), partial laser compatibility

Key finding: Threshold boundary adjustment needed - samples around 0.3-0.5 std dev
show partial laser compatibility, not complete failure.

Version: 1.1 (Updated thresholds based on 5 sample analysis)
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
    feature_type: str  # 'peak', 'mound_crest', 'mound_start', 'mound_end', 'trough_bottom', 'trough_start', 'trough_end', 'baseline_start', 'baseline_end'
    feature_group: str  # 'peak', 'mound', 'trough', 'baseline'
    prominence: float
    snr: float
    confidence: float  # 0-1 confidence score
    detection_method: str  # 'laser_algorithm', 'region_based', 'hybrid'
    width_nm: float = 0.0
    start_wavelength: float = 0.0
    end_wavelength: float = 0.0

class GeminiBSpectralDetector:
    """
    Adaptive B spectra detection using hybrid approach
    """
    
    def __init__(self):
        # Laser algorithm thresholds (for low-noise samples)
        self.laser_prominence_weak = 0.54
        self.laser_prominence_medium = 0.9
        self.laser_prominence_major = 50.0
        self.laser_snr_weak = 4.2
        self.laser_snr_medium = 6.9
        self.laser_intensity_major = 95.0
        
        # B spectra specific thresholds (updated based on 5 sample analysis)
        self.baseline_noise_excellent = 0.025  # ≤0.5x laser threshold
        self.baseline_noise_good = 0.5        # ≤10x laser threshold (increased from 0.3)
        self.baseline_intensity_threshold = 0.1
        
        # Region-based detection parameters
        self.mound_min_width = 50  # nm
        self.mound_min_prominence = 5.0
        self.smoothing_sigma = 2.0
        
    def normalize_b_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """
        Apply correct B spectra normalization: 650nm → 50000, then scale to 0-100
        """
        # Find intensity at closest wavelength to 650nm
        idx_650 = np.argmin(np.abs(wavelengths - 650.0))
        intensity_650 = intensities[idx_650]
        wavelength_650 = wavelengths[idx_650]
        
        if intensity_650 <= 0:
            raise ValueError(f"Invalid intensity at 650nm reference: {intensity_650}")
        
        # Normalize: (intensity / intensity_at_650nm) * 50000 / 500
        normalized = (intensities / intensity_650) * 50000 / 500
        
        # Store normalization metadata
        self.norm_reference_wavelength = wavelength_650
        self.norm_reference_intensity = intensity_650
        
        return normalized
    
    def assess_baseline_noise(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[float, str]:
        """
        Assess baseline noise level to determine algorithm strategy
        Returns: (std_dev, classification)
        """
        # Extract baseline region (300-325nm)
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 325)
        if not np.any(baseline_mask):
            raise ValueError("No data points in baseline region (300-325nm)")
        
        baseline_intensities = intensities[baseline_mask]
        baseline_std = np.std(baseline_intensities)
        
        # Classify noise level
        if baseline_std <= self.baseline_noise_excellent:
            classification = "excellent"
        elif baseline_std <= self.baseline_noise_good:
            classification = "good"
        else:
            classification = "poor"
        
        return baseline_std, classification
    
    def laser_algorithm_detection(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """
        Apply laser algorithm for sharp peak detection (adapted from L spectra)
        """
        features = []
        
        # Find peaks using scipy
        peaks, properties = signal.find_peaks(intensities, 
                                            height=1.0,
                                            prominence=0.1,
                                            distance=5)
        
        for peak_idx in peaks:
            wavelength = wavelengths[peak_idx]
            intensity = intensities[peak_idx]
            
            # Calculate prominence and SNR
            prominence = self._calculate_prominence(intensities, peak_idx)
            snr = self._calculate_snr(intensities, peak_idx)
            
            # Classify using laser algorithm thresholds
            if prominence > self.laser_prominence_major and intensity > self.laser_intensity_major:
                feature_type = "peak"
                confidence = 0.95
            elif prominence >= self.laser_prominence_medium and snr >= self.laser_snr_medium:
                feature_type = "peak"
                confidence = 0.8
            elif prominence >= self.laser_prominence_weak and snr >= self.laser_snr_weak:
                feature_type = "peak"
                confidence = 0.6
            else:
                continue  # Below threshold
            
            feature = BSpectralFeature(
                wavelength=wavelength,
                intensity=intensity,
                feature_type=feature_type,
                feature_group="peak",
                prominence=prominence,
                snr=snr,
                confidence=confidence,
                detection_method="laser_algorithm"
            )
            features.append(feature)
        
        return features
    
    def region_based_detection(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """
        Region-based detection for mounds, troughs, and broad features
        """
        features = []
        
        # Smooth the data for better region detection
        smoothed = gaussian_filter1d(intensities, sigma=self.smoothing_sigma)
        
        # Detect mounds (broad peaks)
        mound_features = self._detect_mounds(wavelengths, smoothed)
        features.extend(mound_features)
        
        # Detect troughs within mounds
        trough_features = self._detect_troughs(wavelengths, smoothed)
        features.extend(trough_features)
        
        return features
    
    def _detect_mounds(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect broad mound features"""
        features = []
        
        # Find broad peaks with large distance parameter
        peaks, properties = signal.find_peaks(intensities, 
                                            prominence=self.mound_min_prominence,
                                            width=self.mound_min_width/np.mean(np.diff(wavelengths)),
                                            distance=20)
        
        for peak_idx in peaks:
            # Find mound boundaries
            start_idx, end_idx = self._find_mound_boundaries(intensities, peak_idx)
            
            wavelength = wavelengths[peak_idx]
            intensity = intensities[peak_idx]
            start_wavelength = wavelengths[start_idx]
            end_wavelength = wavelengths[end_idx]
            width_nm = abs(end_wavelength - start_wavelength)
            
            prominence = self._calculate_prominence(intensities, peak_idx, window=30)
            
            # Create mound features
            features.extend([
                BSpectralFeature(
                    wavelength=start_wavelength,
                    intensity=intensities[start_idx],
                    feature_type="mound_start",
                    feature_group="mound",
                    prominence=0.0,
                    snr=0.0,
                    confidence=0.8,
                    detection_method="region_based",
                    width_nm=width_nm,
                    start_wavelength=start_wavelength,
                    end_wavelength=end_wavelength
                ),
                BSpectralFeature(
                    wavelength=wavelength,
                    intensity=intensity,
                    feature_type="mound_crest",
                    feature_group="mound",
                    prominence=prominence,
                    snr=0.0,
                    confidence=0.9,
                    detection_method="region_based",
                    width_nm=width_nm,
                    start_wavelength=start_wavelength,
                    end_wavelength=end_wavelength
                ),
                BSpectralFeature(
                    wavelength=end_wavelength,
                    intensity=intensities[end_idx],
                    feature_type="mound_end",
                    feature_group="mound",
                    prominence=0.0,
                    snr=0.0,
                    confidence=0.8,
                    detection_method="region_based",
                    width_nm=width_nm,
                    start_wavelength=start_wavelength,
                    end_wavelength=end_wavelength
                )
            ])
        
        return features
    
    def _detect_troughs(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect trough features (local minima within mounds)"""
        features = []
        
        # Find troughs by inverting the signal
        inverted = -intensities
        peaks, _ = signal.find_peaks(inverted, prominence=1.0, distance=10)
        
        for peak_idx in peaks:
            # Only consider significant troughs in regions above baseline
            if intensities[peak_idx] > 10.0:  # Above baseline threshold
                start_idx, end_idx = self._find_trough_boundaries(intensities, peak_idx)
                
                wavelength = wavelengths[peak_idx]
                intensity = intensities[peak_idx]
                start_wavelength = wavelengths[start_idx]
                end_wavelength = wavelengths[end_idx]
                
                features.extend([
                    BSpectralFeature(
                        wavelength=start_wavelength,
                        intensity=intensities[start_idx],
                        feature_type="trough_start",
                        feature_group="trough",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.7,
                        detection_method="region_based"
                    ),
                    BSpectralFeature(
                        wavelength=wavelength,
                        intensity=intensity,
                        feature_type="trough_bottom",
                        feature_group="trough",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method="region_based"
                    ),
                    BSpectralFeature(
                        wavelength=end_wavelength,
                        intensity=intensities[end_idx],
                        feature_type="trough_end",
                        feature_group="trough",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.7,
                        detection_method="region_based"
                    )
                ])
        
        return features
    
    def detect_baseline(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[BSpectralFeature]:
        """Detect baseline start and end points"""
        features = []
        
        # Find baseline region
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 370)  # Extended range for B spectra
        baseline_wavelengths = wavelengths[baseline_mask]
        baseline_intensities = intensities[baseline_mask]
        
        if len(baseline_wavelengths) < 2:
            return features
        
        # Find start point (first low-intensity point)
        start_idx = 0
        start_wavelength = baseline_wavelengths[start_idx]
        start_intensity = baseline_intensities[start_idx]
        
        # Find end point (last low-intensity point or where signal rises)
        derivatives = np.gradient(baseline_intensities)
        end_candidates = np.where(derivatives > 0.01)[0]
        
        if len(end_candidates) > 0:
            end_idx = end_candidates[0]
        else:
            end_idx = len(baseline_intensities) - 1
        
        end_wavelength = baseline_wavelengths[end_idx]
        end_intensity = baseline_intensities[end_idx]
        
        # Calculate baseline quality metrics
        baseline_std = np.std(baseline_intensities)
        snr = 1.0 / (baseline_std + 1e-6)
        
        features.extend([
            BSpectralFeature(
                wavelength=start_wavelength,
                intensity=start_intensity,
                feature_type="baseline_start",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9 if baseline_std < 0.1 else 0.6,
                detection_method="region_based"
            ),
            BSpectralFeature(
                wavelength=end_wavelength,
                intensity=end_intensity,
                feature_type="baseline_end",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9 if baseline_std < 0.1 else 0.6,
                detection_method="region_based"
            )
        ])
        
        return features
    
    def analyze_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Dict:
        """
        Main analysis function using adaptive detection strategy
        """
        # Input validation
        if len(wavelengths) != len(intensities):
            raise ValueError("Wavelength and intensity arrays must have same length")
        
        # Step 1: Normalize the spectrum
        normalized_intensities = self.normalize_b_spectrum(wavelengths, intensities)
        
        # Step 2: Assess baseline noise
        baseline_std, noise_classification = self.assess_baseline_noise(wavelengths, normalized_intensities)
        
        # Step 3: Select detection strategy
        all_features = []
        
        # Always detect baseline
        baseline_features = self.detect_baseline(wavelengths, normalized_intensities)
        all_features.extend(baseline_features)
        
        # Adaptive main feature detection
        if noise_classification == "excellent":
            # Try laser algorithm first
            laser_features = self.laser_algorithm_detection(wavelengths, normalized_intensities)
            region_features = self.region_based_detection(wavelengths, normalized_intensities)
            
            # Use both methods, mark as hybrid
            for feature in laser_features:
                feature.detection_method = "hybrid_laser"
            for feature in region_features:
                feature.detection_method = "hybrid_region"
            
            all_features.extend(laser_features)
            all_features.extend(region_features)
            detection_strategy = "hybrid"
            
        elif noise_classification == "good":
            # Try laser algorithm with fallback
            laser_features = self.laser_algorithm_detection(wavelengths, normalized_intensities)
            
            if len(laser_features) == 0:
                # Fallback to region-based
                region_features = self.region_based_detection(wavelengths, normalized_intensities)
                all_features.extend(region_features)
                detection_strategy = "region_fallback"
            else:
                # Use laser results
                all_features.extend(laser_features)
                detection_strategy = "laser_primary"
        
        else:  # poor noise
            # Skip to region-based detection
            region_features = self.region_based_detection(wavelengths, normalized_intensities)
            all_features.extend(region_features)
            detection_strategy = "region_only"
        
        # Step 4: Sort features by wavelength
        all_features.sort(key=lambda x: x.wavelength)
        
        # Step 5: Calculate overall confidence
        if all_features:
            avg_confidence = np.mean([f.confidence for f in all_features])
        else:
            avg_confidence = 0.0
        
        return {
            'features': all_features,
            'normalization': {
                'reference_wavelength': self.norm_reference_wavelength,
                'reference_intensity': self.norm_reference_intensity,
                'method': '650nm_to_50000_scale_100'
            },
            'baseline_assessment': {
                'noise_std': baseline_std,
                'noise_classification': noise_classification,
                'vs_laser_threshold': baseline_std / 0.05
            },
            'detection_strategy': detection_strategy,
            'overall_confidence': avg_confidence,
            'feature_count': len(all_features),
            'feature_summary': self._summarize_features(all_features)
        }
    
    def _calculate_prominence(self, intensities: np.ndarray, peak_idx: int, window: int = 15) -> float:
        """Calculate prominence of a peak"""
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
    
    def _find_mound_boundaries(self, intensities: np.ndarray, peak_idx: int) -> Tuple[int, int]:
        """Find start and end indices of a mound"""
        # Look for significant drop on both sides
        threshold = intensities[peak_idx] * 0.1  # 10% of peak height
        
        # Find start (going backwards)
        start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if intensities[i] < threshold:
                start_idx = i
                break
        
        # Find end (going forwards)
        end_idx = peak_idx
        for i in range(peak_idx + 1, len(intensities)):
            if intensities[i] < threshold:
                end_idx = i
                break
        
        return start_idx, end_idx
    
    def _find_trough_boundaries(self, intensities: np.ndarray, trough_idx: int) -> Tuple[int, int]:
        """Find start and end indices of a trough"""
        trough_value = intensities[trough_idx]
        
        # Find boundaries where intensity increases significantly
        start_idx = trough_idx
        for i in range(trough_idx - 1, -1, -1):
            if intensities[i] > trough_value * 1.2:
                start_idx = i
                break
        
        end_idx = trough_idx
        for i in range(trough_idx + 1, len(intensities)):
            if intensities[i] > trough_value * 1.2:
                end_idx = i
                break
        
        return start_idx, end_idx
    
    def _summarize_features(self, features: List[BSpectralFeature]) -> Dict:
        """Generate summary statistics of detected features"""
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

def load_b_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load B spectrum from file
    Supports both .txt and .csv formats
    """
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
        wavelengths = data.iloc[:, 0].values
        intensities = data.iloc[:, 1].values
    else:
        # Assume tab-separated .txt file
        data = np.loadtxt(filepath, delimiter='\t')
        wavelengths = data[:, 0]
        intensities = data[:, 1]
    
    return wavelengths, intensities

def analyze_b_spectrum_file(filepath: str) -> Dict:
    """
    Analyze a B spectrum file and return results
    """
    detector = GeminiBSpectralDetector()
    wavelengths, intensities = load_b_spectrum(filepath)
    return detector.analyze_spectrum(wavelengths, intensities)

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    detector = GeminiBSpectralDetector()
    
    # Example: analyze a file
    # results = analyze_b_spectrum_file("sample_b_spectrum.txt")
    # print(f"Detected {results['feature_count']} features using {results['detection_strategy']} strategy")
    # print(f"Baseline noise: {results['baseline_assessment']['noise_classification']}")
    
    print("Gemini B Spectra Auto-Detector initialized")
    print("Ready to analyze B spectra samples")
    print("Current training data: 4 samples (190BP2, 189BP2, 79BC1, 92BC1)")
