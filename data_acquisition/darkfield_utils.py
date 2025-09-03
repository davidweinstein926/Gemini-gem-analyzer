import os
import numpy as np

DARKFIELD_DIR = r"C:\\Users\\David\\Gemini SP10 Calibration\\DarkFields"

def load_latest_darkfield():
    try:
        files = sorted([
            f for f in os.listdir(DARKFIELD_DIR)
            if f.startswith("darkfield_") and f.endswith(".txt")
        ], reverse=True)

        if not files:
            raise FileNotFoundError("No darkfield_*.txt files found.")

        latest_file = os.path.join(DARKFIELD_DIR, files[0])
        data = np.loadtxt(latest_file, delimiter="\t")
        wavelengths, dark_intensity = data[:, 0], data[:, 1]
        return wavelengths, dark_intensity

    except Exception as e:
        print(f"⚠️ Could not load darkfield: {e}")
        return None, None