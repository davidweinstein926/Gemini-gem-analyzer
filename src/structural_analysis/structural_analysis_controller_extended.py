# structural_analysis_controller_extended.py
# Automatically runs BB, LAS, and UV analyzers with option for auto/manual mode and DB export

import os
import pandas as pd
from bb_peak_matcher import analyze_spectrum as analyze_bb
from las_peak_matcher import analyze_spectrum as analyze_las
from uv_peak_matcher import analyze_spectrum as analyze_uv

# File paths for each spectrum
FILES = {
    'B': 'unkgemB.csv',
    'L': 'unkgemL.csv',
    'U': 'unkgemU.csv'
}

# Prompt user for mode selection
mode = input("🔧 Select mode (auto/manual): ").strip().lower()
if mode not in ['auto', 'manual']:
    print("⚠️ Invalid input. Defaulting to auto mode.")
    mode = 'auto'

# Output log
log = []

# Run analyzers and record results
for light, file in FILES.items():
    if not os.path.exists(file):
        msg = f"❌ Missing file for {light}: {file}"
        print(msg)
        log.append(msg)
        continue

    try:
        if mode == 'manual':
            print(f"🖼️  Launching {light} window... please interact manually and close it to continue.")

        if light == 'B':
            result = analyze_bb(file, mode=mode)
        elif light == 'L':
            result = analyze_las(file, mode=mode)
        elif light == 'U':
            result = analyze_uv(file, mode=mode)

        msg = f"✅ {light} analysis complete: {file}"
        print(msg)
        log.append(msg)
    except TypeError:
        # Fallback for legacy matcher without mode argument
        try:
            print(f"⚠️  Legacy mode fallback for {light}. No mode argument in function.")
            if light == 'B':
                result = analyze_bb(file)
            elif light == 'L':
                result = analyze_las(file)
            elif light == 'U':
                result = analyze_uv(file)
            msg = f"✅ {light} analysis complete (legacy): {file}"
            print(msg)
            log.append(msg)
        except Exception as e:
            msg = f"❌ Error in legacy fallback for {light}: {e}"
            print(msg)
            log.append(msg)
    except Exception as e:
        msg = f"❌ Error processing {light}: {e}"
        print(msg)
        log.append(msg)

# Optional: export log
with open("structural_analysis_log.txt", "w", encoding="utf-8") as f:
    for line in log:
        f.write(line + "\n")

print("\n🔍 Structural analysis completed for all available files.")
