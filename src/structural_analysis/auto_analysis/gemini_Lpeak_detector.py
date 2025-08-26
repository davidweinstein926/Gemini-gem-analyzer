"""
Gemini L Spectra Auto-Detector
Adaptive detection algorithm for laser/high-resolution transmission spectra

Optimized for laser-induced spectra with sharp, well-defined peaks
and high signal-to-noise ratios. Designed for natural/synthetic
discrimination and precision wavelength measurements.

Version: 1.1 (Updated thresholds for 0-100 normalized scale)
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings

@dataclass
class LSpectralFeature:
    """Represents a detected feature in L spectra"""
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
    effective_height: float = 0.0  # Peak height above local baseline

class GeminiLSpectralDetector:
    """
    Adaptive L spectra detection optimized for laser spectra
    """
    
    def __init__(self):
        # Laser algorithm thresholds (ADJUSTED for 0-100 normalized scale)
        self.laser_prominence_weak = 1.0      # 1% prominence
        self.laser_prominence_medium = 3.0    # 3% prominence  
        self.laser_prominence_major = 8.0     # 8% prominence
        self.laser_snr_weak = 3.0             # Lower SNR requirements
        self.laser_snr_medium = 5.0
        self.laser_intensity_major = 20.0     # 20% of max intensity
        
        # L spectra specific thresholds (ADJUSTED)
        self.baseline_noise_excellent = 0.01   
        self.baseline_noise_good = 0.05        
        self.baseline_intensity_threshold = 1.0  # Reasonable for 0-100 scale
        
        # Region-based detection parameters (DISABLED for L spectra)
        self.mound_min_width = 200  # Very large to prevent detection
        self.mound_min_prominence = 50.0  # Very high to prevent detection
        self.smoothing_sigma = 0.1  # Minimal smoothing
        
        # Peak detection parameters (REASONABLE for 0-100 scale)
        self.peak_min_distance = 15  # 15nm minimum separation
        self.peak_min_prominence = 2.0  # 2% prominence requirement
        self.peak_width_range = (5, 20)  # Narrower acceptable range
        
    def normalize_l_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> np.ndarray:
        """
        Apply L spectra normalization: max intensity → 50000, then scale to 0-100
        """
        # Find maximum intensity in raw data
        max_intensity = np.max(intensities)
        max_idx = np.argmax(intensities)
        max_wavelength = wavelengths[max_idx]
        
        print(f"Debug L normalization: max at {max_wavelength:.2f}nm, intensity = {max_intensity:.6f}")
        print(f"Debug L normalization: min intensity = {np.min(intensities):.6f}")
        
        if max_intensity <= 0:
            raise ValueError(f"Invalid maximum intensity: {max_intensity}")
        
        # Step 1: Normalize so max = 50000
        normalized_to_50000 = (intensities / max_intensity) * 50000
        
        # Step 2: Scale to 0-100 range
        normalized = normalized_to_50000 / 500
        
        print(f"Debug L normalization: After normalization min = {np.min(normalized):.6f}, max = {np.max(normalized):.6f}")
        
        # Store normalization metadata
        self.norm_reference_wavelength = max_wavelength
        self.norm_reference_intensity = max_intensity
        
        return normalized
    
    def assess_baseline_noise(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Tuple[float, str]:
        """
        Assess baseline noise level optimized for L spectra - FIXED to use 300-350nm
        Returns: (std_dev, classification)
        """
        # FIXED: Use exact 300-350nm range for baseline assessment
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 350)
        if not np.any(baseline_mask):
            # Fallback: use first 10% of data as baseline
            n_baseline = max(10, len(wavelengths) // 10)
            baseline_mask = np.zeros(len(wavelengths), dtype=bool)
            baseline_mask[:n_baseline] = True
        
        baseline_intensities = intensities[baseline_mask]
        baseline_std = np.std(baseline_intensities)
        
        # Classify noise level (stricter thresholds for L spectra)
        if baseline_std <= self.baseline_noise_excellent:
            classification = "excellent"
        elif baseline_std <= self.baseline_noise_good:
            classification = "good"
        else:
            classification = "poor"
        
        return baseline_std, classification
    
    def laser_algorithm_detection(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[LSpectralFeature]:
        """
        Apply optimized laser algorithm for sharp peak detection
        """
        features = []
        
        # Convert distance from nm to index units
        wavelength_step = np.mean(np.diff(wavelengths))
        min_distance_idx = max(1, int(self.peak_min_distance / wavelength_step))
        
        # Calculate dynamic threshold based on baseline noise (300-350nm region)
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 350)
        if np.any(baseline_mask):
            baseline_max = np.max(intensities[baseline_mask])
        else:
            baseline_max = np.min(intensities)
        
        # FIXED: Use absolute intensity threshold of 2.0 (normalized scale 0-100)
        absolute_threshold = 2.0
        print(f"Debug: Using absolute threshold = {absolute_threshold} (normalized scale)")
        
        # Find peaks using more permissive settings to catch small but real peaks
        peaks, properties = signal.find_peaks(intensities, 
                                            height=absolute_threshold,  # Must be >2.0 normalized
                                            prominence=0.2,  # Lower to catch small peaks like 575nm
                                            distance=min_distance_idx,  # 15nm minimum spacing
                                            width=1)
        
        print(f"Debug: scipy.find_peaks found {len(peaks)} candidates above {absolute_threshold}")
        
        # Filter peaks with absolute intensity threshold (already done in find_peaks, but double-check)
        filtered_peaks = []
        for peak_idx in peaks:
            intensity = intensities[peak_idx]
            
            # Must be above 2.0 normalized intensity  
            if intensity >= absolute_threshold:
                filtered_peaks.append(peak_idx)
        
        print(f"Debug: After intensity filtering: {len(filtered_peaks)} peaks remain")
        
        # FIXED: Merge peaks within 1nm (same peak)
        merged_peaks = []
        if len(filtered_peaks) > 0:
            # Sort by wavelength
            filtered_peaks.sort(key=lambda idx: wavelengths[idx])
            
            merged_peaks.append(filtered_peaks[0])  # Always keep first peak
            
            for peak_idx in filtered_peaks[1:]:
                current_wavelength = wavelengths[peak_idx]
                last_wavelength = wavelengths[merged_peaks[-1]]
                
                # If within 1nm of previous peak, it's the same peak
                if abs(current_wavelength - last_wavelength) <= 1.0:
                    # Keep the higher intensity peak
                    if intensities[peak_idx] > intensities[merged_peaks[-1]]:
                        merged_peaks[-1] = peak_idx
                else:
                    # Different peak, add it
                    merged_peaks.append(peak_idx)
        
        print(f"Debug: After merging within 1nm: {len(merged_peaks)} peaks remain")
        
        # Sort final peaks by wavelength for consistent output
        merged_peaks.sort(key=lambda idx: wavelengths[idx])
        
        for peak_idx in merged_peaks:
            wavelength = wavelengths[peak_idx]
            intensity = intensities[peak_idx]
            
            # FILTER 1: Ignore peaks below 400nm (UV artifacts in laser spectra)
            if wavelength < 400.0:
                print(f"Debug: Rejecting peak at {wavelength:.1f}nm - below 400nm threshold")
                continue
            
            # Calculate prominence and SNR
            prominence = self._calculate_prominence(intensities, peak_idx)
            snr = self._calculate_snr(intensities, peak_idx)
            
            # Calculate peak width
            width_nm = self._calculate_peak_width(wavelengths, intensities, peak_idx)
            
            # FILTER 2: Ignore peaks with invalid widths
            if width_nm > 100.0:
                print(f"Debug: Rejecting peak at {wavelength:.1f}nm - width {width_nm:.1f}nm too large (>100nm)")
                continue
            elif width_nm < 1.0:
                print(f"Debug: Rejecting peak at {wavelength:.1f}nm - width {width_nm:.1f}nm too small (<1nm)")  
                continue
            
            # FILTER 3: Calculate baseline-corrected intensity for better assessment
            baseline_corrected_intensity = intensity - self._estimate_local_baseline(intensities, peak_idx)
            
            # Use baseline-corrected intensity for quality assessment
            if baseline_corrected_intensity < 1.0:  # Peak must be >1% above local baseline
                print(f"Debug: Rejecting peak at {wavelength:.1f}nm - baseline-corrected intensity {baseline_corrected_intensity:.2f} too low")
                continue
            
            # Calculate peak width and top flatness
            width_nm = self._calculate_peak_width(wavelengths, intensities, peak_idx)
            flatness_ratio = self._calculate_top_flatness(intensities, peak_idx)
            
            # ACCEPT ALL PEAKS that pass filters - classify by width AND flatness
            if width_nm > 25.0 and flatness_ratio > 0.6:
                # Wide features with flat tops are mounds
                feature_type = "mound_crest"
                feature_group = "mound"
            else:
                # Everything else is a peak (narrow, or wide but pointed)
                feature_type = "peak"
                feature_group = "peak"
            
            # Determine confidence based on prominence and intensity
            if prominence > self.laser_prominence_major and baseline_corrected_intensity > self.laser_intensity_major:
                confidence = 0.95
            elif prominence >= self.laser_prominence_medium and snr >= self.laser_snr_medium:
                confidence = 0.8
            elif prominence >= self.laser_prominence_weak and snr >= self.laser_snr_weak:
                confidence = 0.6
            else:
                confidence = 0.4  # Low confidence but still keep the peak
            
            feature = LSpectralFeature(
                wavelength=wavelength,
                intensity=intensity,  # Use raw normalized intensity for consistency with graph
                feature_type=feature_type,
                feature_group=feature_group,
                prominence=prominence,
                snr=snr,
                confidence=confidence,
                detection_method="laser_algorithm",
                width_nm=width_nm,
                effective_height=baseline_corrected_intensity  # Peak height above local baseline
            )
            features.append(feature)
            
            print(f"Debug: Accepted {feature_type} at {wavelength:.1f}nm - raw: {intensity:.2f}, effective height: {baseline_corrected_intensity:.2f}, width: {width_nm:.1f}nm")
        
        return features
    
    def region_based_detection(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[LSpectralFeature]:
        """
        Region-based detection optimized for L spectra
        """
        features = []
        
        # Minimal smoothing to preserve sharp features
        smoothed = gaussian_filter1d(intensities, sigma=self.smoothing_sigma)
        
        # Detect mounds (broader features)
        mound_features = self._detect_mounds(wavelengths, smoothed)
        features.extend(mound_features)
        
        # Detect troughs
        trough_features = self._detect_troughs(wavelengths, smoothed)
        features.extend(trough_features)
        
        return features
    
    def _detect_mounds(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[LSpectralFeature]:
        """Detect broad mound features optimized for L spectra"""
        features = []
        
        # Convert width from nm to index units
        wavelength_step = np.mean(np.diff(wavelengths))
        min_width_idx = self.mound_min_width / wavelength_step
        
        # Find broad peaks
        peaks, properties = signal.find_peaks(intensities, 
                                            prominence=self.mound_min_prominence,
                                            width=min_width_idx,
                                            distance=10)
        
        for peak_idx in peaks:
            # Find mound boundaries
            start_idx, end_idx = self._find_mound_boundaries(intensities, peak_idx)
            
            wavelength = wavelengths[peak_idx]
            intensity = intensities[peak_idx]
            start_wavelength = wavelengths[start_idx]
            end_wavelength = wavelengths[end_idx]
            width_nm = abs(end_wavelength - start_wavelength)
            
            prominence = self._calculate_prominence(intensities, peak_idx, window=20)
            
            # Create mound features
            features.extend([
                LSpectralFeature(
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
                LSpectralFeature(
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
                LSpectralFeature(
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
    
    def _detect_troughs(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[LSpectralFeature]:
        """Detect trough features optimized for L spectra"""
        features = []
        
        # Find troughs by inverting the signal
        inverted = -intensities
        peaks, _ = signal.find_peaks(inverted, prominence=0.5, distance=5)
        
        for peak_idx in peaks:
            # Only consider significant troughs
            if intensities[peak_idx] > 5.0:  # Above baseline threshold
                start_idx, end_idx = self._find_trough_boundaries(intensities, peak_idx)
                
                wavelength = wavelengths[peak_idx]
                intensity = intensities[peak_idx]
                start_wavelength = wavelengths[start_idx]
                end_wavelength = wavelengths[end_idx]
                
                features.extend([
                    LSpectralFeature(
                        wavelength=start_wavelength,
                        intensity=intensities[start_idx],
                        feature_type="trough_start",
                        feature_group="trough",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.7,
                        detection_method="region_based"
                    ),
                    LSpectralFeature(
                        wavelength=wavelength,
                        intensity=intensity,
                        feature_type="trough_bottom",
                        feature_group="trough",
                        prominence=0.0,
                        snr=0.0,
                        confidence=0.8,
                        detection_method="region_based"
                    ),
                    LSpectralFeature(
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
    
    def detect_baseline(self, wavelengths: np.ndarray, intensities: np.ndarray) -> List[LSpectralFeature]:
        """Detect baseline start and end points for L spectra - FIXED to 300-350nm"""
        features = []
        
        # FIXED: Force baseline region to exactly 300-350nm as specified
        baseline_mask = (wavelengths >= 300) & (wavelengths <= 350)
        
        if not np.any(baseline_mask):
            # Fallback if no data in 300-350nm range
            print("Warning: No data in 300-350nm range, using first 10% of spectrum")
            n_baseline = len(wavelengths) // 10
            baseline_mask = np.zeros(len(wavelengths), dtype=bool)
            baseline_mask[:n_baseline] = True
        
        baseline_wavelengths = wavelengths[baseline_mask]
        baseline_intensities = intensities[baseline_mask]
        
        if len(baseline_wavelengths) < 2:
            return features
        
        # Find actual start and end points within 300-350nm range
        start_idx = 0
        end_idx = len(baseline_intensities) - 1
        
        start_wavelength = baseline_wavelengths[start_idx]
        start_intensity = baseline_intensities[start_idx]
        end_wavelength = baseline_wavelengths[end_idx]
        end_intensity = baseline_intensities[end_idx]
        
        # Calculate baseline quality metrics
        baseline_std = np.std(baseline_intensities)
        snr = 1.0 / (baseline_std + 1e-6)
        
        features.extend([
            LSpectralFeature(
                wavelength=start_wavelength,
                intensity=start_intensity,
                feature_type="baseline_start",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9 if baseline_std < 0.02 else 0.6,
                detection_method="region_based"
            ),
            LSpectralFeature(
                wavelength=end_wavelength,
                intensity=end_intensity,
                feature_type="baseline_end",
                feature_group="baseline",
                prominence=0.0,
                snr=snr,
                confidence=0.9 if baseline_std < 0.02 else 0.6,
                detection_method="region_based"
            )
        ])
        
        return features
    
    def analyze_spectrum(self, wavelengths: np.ndarray, intensities: np.ndarray) -> Dict:
        """
        Main analysis function optimized for L spectra
        """
        # Input validation
        if len(wavelengths) != len(intensities):
            raise ValueError("Wavelength and intensity arrays must have same length")
        
        # Step 1: Normalize the spectrum
        normalized_intensities = self.normalize_l_spectrum(wavelengths, intensities)
        
        # Step 2: Assess baseline noise
        baseline_std, noise_classification = self.assess_baseline_noise(wavelengths, normalized_intensities)
        
        # Step 3: L spectra strategy - LASER ALGORITHM ONLY (no region-based)
        all_features = []
        
        # Always detect baseline
        baseline_features = self.detect_baseline(wavelengths, normalized_intensities)
        all_features.extend(baseline_features)
        
        # For L spectra: ONLY use laser algorithm with strict thresholds
        # NO region-based detection to prevent over-detection
        laser_features = self.laser_algorithm_detection(wavelengths, normalized_intensities)
        all_features.extend(laser_features)
        detection_strategy = "laser_only_strict"
        
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
                'vs_laser_threshold': baseline_std / 0.01
            },
            'detection_strategy': detection_strategy,
            'overall_confidence': avg_confidence,
            'feature_count': len(all_features),
            'feature_summary': self._summarize_features(all_features)
        }
    
    def _estimate_local_baseline(self, intensities: np.ndarray, peak_idx: int, window: int = 20) -> float:
        """Estimate local baseline around a peak for baseline-corrected intensity"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        local_region = intensities[start:end]
        
        # Find the minimum values on left and right sides of peak
        left_idx = peak_idx - start
        left_region = local_region[:left_idx] if left_idx > 0 else np.array([intensities[peak_idx]])
        right_region = local_region[left_idx + 1:] if left_idx < len(local_region) - 1 else np.array([intensities[peak_idx]])
        
        # Estimate baseline as average of minimum regions on both sides
        left_baseline = np.min(left_region) if len(left_region) > 0 else intensities[peak_idx]
        right_baseline = np.min(right_region) if len(right_region) > 0 else intensities[peak_idx]
        
        return (left_baseline + right_baseline) / 2
    
    def _calculate_prominence(self, intensities: np.ndarray, peak_idx: int, window: int = 10) -> float:
        """Calculate prominence of a peak (smaller window for L spectra)"""
        start = max(0, peak_idx - window)
        end = min(len(intensities), peak_idx + window + 1)
        
        local_region = intensities[start:end]
        peak_value = intensities[peak_idx]
        
        left_idx = peak_idx - start
        left_min = np.min(local_region[:left_idx]) if left_idx > 0 else peak_value
        right_min = np.min(local_region[left_idx + 1:]) if left_idx < len(local_region) - 1 else peak_value
        
        baseline = max(left_min, right_min)
        return max(0, peak_value - baseline)
    
    def _calculate_snr(self, intensities: np.ndarray, peak_idx: int, window: int = 10) -> float:
        """Calculate signal-to-noise ratio (smaller window for L spectra)"""
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
    
    def _calculate_peak_width(self, wavelengths: np.ndarray, intensities: np.ndarray, peak_idx: int) -> float:
        """Calculate peak width using slope analysis for peaks on shoulders"""
        
        # Method 1: Slope-based detection for overlapping peaks
        left_idx = peak_idx
        right_idx = peak_idx
        
        # Find left boundary - where declining slope starts rising (503nm point)
        for i in range(peak_idx - 1, max(0, peak_idx - 10), -1):
            if i > 0:
                # Calculate slopes
                slope_before = intensities[i] - intensities[i-1]
                slope_after = intensities[i+1] - intensities[i]
                
                # Found where decline transitions to rise
                if slope_before <= 0 and slope_after > 0:
                    left_idx = i
                    break
        
        # Find right boundary - where rising slope resumes declining (504.17nm point)  
        for i in range(peak_idx + 1, min(len(intensities), peak_idx + 10)):
            if i < len(intensities) - 1:
                # Calculate slopes
                slope_before = intensities[i] - intensities[i-1] 
                slope_after = intensities[i+1] - intensities[i]
                
                # Found where rise transitions back to decline
                if slope_before > 0 and slope_after <= 0:
                    right_idx = i
                    break
        
        # Fallback to half-maximum if slope method finds nothing
        if left_idx == peak_idx and right_idx == peak_idx:
            peak_intensity = intensities[peak_idx]
            half_max = peak_intensity * 0.5
            
            for i in range(peak_idx - 1, -1, -1):
                if intensities[i] < half_max:
                    left_idx = i
                    break
                    
            for i in range(peak_idx + 1, len(intensities)):
                if intensities[i] < half_max:
                    right_idx = i
                    break
        
        return abs(wavelengths[right_idx] - wavelengths[left_idx])
    
    def _calculate_top_flatness(self, intensities: np.ndarray, peak_idx: int, window: int = 3) -> float:
        """Measure how flat the peak top is (mounds have flat tops, peaks are pointed)"""
        start = max(0, peak_idx - window)  
        end = min(len(intensities), peak_idx + window + 1)
        
        peak_region = intensities[start:end]
        peak_max = intensities[peak_idx]
        
        # Count points within 5% of peak maximum (indicates flat top)
        tolerance = peak_max * 0.05
        flat_points = np.sum(np.abs(peak_region - peak_max) <= tolerance)
        
        # Return ratio of flat points to total points in region
        flatness_ratio = flat_points / len(peak_region)
        
        return flatness_ratio
    
    def _find_mound_boundaries(self, intensities: np.ndarray, peak_idx: int) -> Tuple[int, int]:
        """Find start and end indices of a mound"""
        threshold = intensities[peak_idx] * 0.2  # 20% of peak height for L spectra
        
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
            if intensities[i] > trough_value * 1.5:
                start_idx = i
                break
        
        end_idx = trough_idx
        for i in range(trough_idx + 1, len(intensities)):
            if intensities[i] > trough_value * 1.5:
                end_idx = i
                break
        
        return start_idx, end_idx
    
    def _summarize_features(self, features: List[LSpectralFeature]) -> Dict:
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

def load_l_spectrum(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load L spectrum from file with robust parsing
    Supports both .txt and .csv formats, handles multiple delimiters
    """
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
        wavelengths = data.iloc[:, 0].values
        intensities = data.iloc[:, 1].values
    else:
        # Try multiple delimiters for .txt files
        try:
            # First try tab-separated
            data = np.loadtxt(filepath, delimiter='\t')
        except ValueError:
            try:
                # Try space-separated
                data = np.loadtxt(filepath, delimiter=None)  # None = any whitespace
            except ValueError:
                try:
                    # Try comma-separated
                    data = np.loadtxt(filepath, delimiter=',')
                except ValueError:
                    # Final fallback: pandas with flexible parsing
                    data = pd.read_csv(filepath, sep=None, engine='python', header=None)
                    data = data.values
        
        wavelengths = data[:, 0]
        intensities = data[:, 1]
    
    return wavelengths, intensities

def analyze_l_spectrum_file(filepath: str) -> Dict:
    """
    Analyze an L spectrum file and return results
    """
    detector = GeminiLSpectralDetector()
    wavelengths, intensities = load_l_spectrum(filepath)
    return detector.analyze_spectrum(wavelengths, intensities)

# Example usage and testing
if __name__ == "__main__":
    detector = GeminiLSpectralDetector()
    print("Gemini L Spectra Auto-Detector initialized")
    print("Ready to analyze L spectra samples")
    print("Optimized for laser-induced high-resolution spectra")
