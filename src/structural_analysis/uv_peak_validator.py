#!/usr/bin/env python3
"""
UV Peak Validation System
Filters auto-detected peaks against UV source reference and SNR thresholds
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class UVPeakValidator:
    """Validates auto-detected UV peaks against reference and noise thresholds"""
    
    def __init__(self, uv_source_reference_file: str):
        self.reference_peaks = self.load_uv_source_reference(uv_source_reference_file)
        self.reference_wavelengths = self.extract_reference_wavelengths()
        print(f"🔬 UV Source Reference: {len(self.reference_wavelengths)} characteristic peaks")
    
    def load_uv_source_reference(self, reference_file: str) -> pd.DataFrame:
        """Load UV source reference peaks (S010123UC1)"""
        try:
            df = pd.read_csv(reference_file)
            print(f"✅ Loaded UV source reference: {reference_file}")
            return df
        except Exception as e:
            print(f"❌ Error loading reference: {e}")
            return pd.DataFrame()
    
    def extract_reference_wavelengths(self) -> List[float]:
        """Extract wavelengths from UV source reference"""
        if 'Wavelength' in self.reference_peaks.columns:
            wavelengths = self.reference_peaks['Wavelength'].tolist()
        elif 'Wavelength_nm' in self.reference_peaks.columns:
            wavelengths = self.reference_peaks['Wavelength_nm'].tolist()
        else:
            print("⚠️ No wavelength column found in reference")
            return []
        
        # Sort and remove duplicates
        unique_wavelengths = sorted(list(set(wavelengths)))
        print(f"📊 Reference wavelength range: {min(unique_wavelengths):.1f} - {max(unique_wavelengths):.1f} nm")
        return unique_wavelengths
    
    def calculate_snr_threshold(self, auto_detected_file: str) -> float:
        """Calculate SNR-based threshold from auto-detected peak data"""
        try:
            df = pd.read_csv(auto_detected_file)
            
            if 'Category' in df.columns:
                # Use category-based approach
                minor_peaks = df[df['Category'] == 'Minor']['Prominence']
                if len(minor_peaks) > 0:
                    # Set threshold at 75th percentile of minor peaks
                    threshold = np.percentile(minor_peaks, 75)
                else:
                    threshold = df['Prominence'].quantile(0.25)
            else:
                # Use statistical approach
                prominences = df['Prominence']
                threshold = np.mean(prominences) + 2 * np.std(prominences)
            
            print(f"📈 Calculated SNR threshold: {threshold:.6f}")
            return threshold
            
        except Exception as e:
            print(f"⚠️ Error calculating threshold: {e}")
            return 0.01  # Default conservative threshold
    
    def peak_matches_reference(self, peak_wavelength: float, tolerance: float = 2.0) -> bool:
        """Check if detected peak matches known UV source wavelength"""
        for ref_wl in self.reference_wavelengths:
            if abs(peak_wavelength - ref_wl) <= tolerance:
                return True
        return False
    
    def validate_auto_detected_peaks(self, auto_detected_file: str, 
                                   tolerance: float = 2.0,
                                   use_adaptive_threshold: bool = True) -> Dict:
        """Validate auto-detected peaks against reference and SNR"""
        try:
            df = pd.read_csv(auto_detected_file)
            
            # Calculate SNR threshold
            if use_adaptive_threshold:
                snr_threshold = self.calculate_snr_threshold(auto_detected_file)
            else:
                snr_threshold = 0.01  # Fixed threshold
            
            validated_peaks = []
            rejected_noise = []
            rejected_spurious = []
            
            for _, peak in df.iterrows():
                wavelength = peak['Wavelength_nm']
                prominence = peak['Prominence']
                category = peak.get('Category', 'Unknown')
                
                # Check reference match
                matches_reference = self.peak_matches_reference(wavelength, tolerance)
                
                # Check SNR threshold
                exceeds_threshold = prominence > snr_threshold
                
                if matches_reference and exceeds_threshold:
                    validated_peaks.append({
                        'wavelength': wavelength,
                        'intensity': peak['Intensity'],
                        'prominence': prominence,
                        'category': category,
                        'validation': 'Reference_Match_SNR_Pass'
                    })
                elif not matches_reference:
                    rejected_spurious.append({
                        'wavelength': wavelength,
                        'prominence': prominence,
                        'reason': 'No_Reference_Match'
                    })
                else:  # matches reference but below threshold
                    rejected_noise.append({
                        'wavelength': wavelength,
                        'prominence': prominence,
                        'reason': 'Below_SNR_Threshold'
                    })
            
            results = {
                'filename': auto_detected_file,
                'total_detected': len(df),
                'validated_peaks': validated_peaks,
                'rejected_noise': rejected_noise,
                'rejected_spurious': rejected_spurious,
                'snr_threshold': snr_threshold,
                'tolerance': tolerance
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error validating peaks: {e}")
            return {}
    
    def generate_validation_report(self, results: Dict) -> str:
        """Generate detailed validation report"""
        if not results:
            return "❌ No validation results available"
        
        report = []
        report.append(f"🔬 UV PEAK VALIDATION REPORT")
        report.append(f"=" * 50)
        report.append(f"📁 File: {results['filename']}")
        report.append(f"📊 Total detected peaks: {results['total_detected']}")
        report.append(f"✅ Validated peaks: {len(results['validated_peaks'])}")
        report.append(f"🔇 Rejected (noise): {len(results['rejected_noise'])}")
        report.append(f"❌ Rejected (spurious): {len(results['rejected_spurious'])}")
        report.append(f"📈 SNR threshold: {results['snr_threshold']:.6f}")
        report.append(f"🎯 Tolerance: ±{results['tolerance']} nm")
        
        # Validation efficiency
        efficiency = len(results['validated_peaks']) / results['total_detected'] * 100
        report.append(f"⚡ Validation efficiency: {efficiency:.1f}%")
        
        # Peak quality assessment
        if efficiency > 80:
            quality = "🌟 Excellent (clean baseline)"
        elif efficiency > 60:
            quality = "👍 Good (moderate noise)"
        elif efficiency > 40:
            quality = "⚠️ Fair (noisy baseline)"
        else:
            quality = "❌ Poor (very noisy)"
        
        report.append(f"🏆 Peak quality: {quality}")
        
        return "\n".join(report)
    
    def validate_batch_files(self, auto_detected_files: List[str]) -> List[Dict]:
        """Validate multiple auto-detected files"""
        all_results = []
        
        print(f"🔬 BATCH UV PEAK VALIDATION")
        print(f"📁 Processing {len(auto_detected_files)} files")
        print(f"🎯 Reference: {len(self.reference_wavelengths)} UV source peaks")
        
        for filename in auto_detected_files:
            print(f"\n📊 Validating: {filename}")
            results = self.validate_auto_detected_peaks(filename)
            
            if results:
                report = self.generate_validation_report(results)
                print(report)
                all_results.append(results)
            else:
                print(f"❌ Validation failed for {filename}")
        
        return all_results

def main():
    """Example usage"""
    # Initialize with UV source reference
    validator = UVPeakValidator("S010123UC1_uv_structural_20250815_103412.csv")
    
    # Validate the problematic files you mentioned
    test_files = ["200UC1.csv", "51UC1.csv", "52UC1.csv", "58UC1.csv"]
    
    results = validator.validate_batch_files(test_files)
    
    print(f"\n🎯 BATCH VALIDATION SUMMARY:")
    print(f"=" * 50)
    for result in results:
        efficiency = len(result['validated_peaks']) / result['total_detected'] * 100
        print(f"{result['filename']}: {len(result['validated_peaks'])}/{result['total_detected']} peaks ({efficiency:.1f}%)")

if __name__ == "__main__":
    main()