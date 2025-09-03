# main_menu.py - UPDATED FOR PROPER DIRECTORY STRUCTURE
# Location: data_acquisition/main_menu.py

import tkinter as tk
import subprocess
import importlib
import traceback
import sys
import os
from pathlib import Path

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Safely attempt to run a module's function
def safe_run(func, description):
    try:
        print(f"Launching {description}...")
        func()
    except Exception as e:
        print(f"Error running {description}:", e)
        traceback.print_exc()

# Import necessary modules with proper error handling
try:
    from . import whitelight_capture as white
except ImportError:
    try:
        import whitelight_capture as white
    except Exception as e:
        print("⚠️ Failed to import whitelight_capture:", e)
        white = None

try:
    from . import calibration_diagnostic as calib
except ImportError:
    try:
        import calibration_diagnostic as calib
    except Exception as e:
        print("⚠️ Failed to import calibration_diagnostic:", e)
        calib = None

try:
    from . import gem_analysis_broadband as bb
except ImportError:
    try:
        import gem_analysis_broadband as bb
    except Exception as e:
        print("⚠️ Failed to import gem_analysis_broadband:", e)
        bb = None

try:
    from . import gem_capture_uv_laser as uvlaser
except ImportError:
    try:
        import gem_capture_uv_laser as uvlaser
    except Exception as e:
        print("⚠️ Failed to import gem_capture_uv_laser:", e)
        uvlaser = None

try:
    from . import dark_capture as dark
except ImportError:
    try:
        import dark_capture as dark
    except Exception as e:
        print("⚠️ Failed to import dark_capture:", e)
        dark = None

# GUI setup
root = tk.Tk()
root.title("Gemini Capture Menu")

# Add project info
project_root = current_dir.parent
print(f"Project root: {project_root}")
print(f"Data acquisition directory: {current_dir}")

frame = tk.Frame(root, padx=20, pady=20)
frame.pack()

# Add project path info label
info_label = tk.Label(frame, text=f"Data Acquisition Module\nProject: {project_root.name}", 
                     font=('Arial', 8), fg='gray')
info_label.pack(pady=(0, 10))

btn_white = tk.Button(frame, text="Capture White Light", width=25,
                      command=lambda: safe_run(white.collect_white_light_spectrum, "white light capture") if white else None)
btn_white.pack(pady=5)

btn_calib = tk.Button(frame, text="Calibrate Wavelength", width=25,
                      command=lambda: safe_run(calib.run_diagnostics, "wavelength calibration") if calib else None)
btn_calib.pack(pady=5)

btn_bb = tk.Button(frame, text="Capture Gem Broadband", width=25,
                   command=lambda: safe_run(bb.capture_broadband_scan, "broadband gem capture") if bb else None)
btn_bb.pack(pady=5)

btn_uvlaser = tk.Button(frame, text="Capture Gem UV/Laser", width=25,
                        command=lambda: safe_run(uvlaser.capture_uv_or_laser, "UV/Laser gem capture") if uvlaser else None)
btn_uvlaser.pack(pady=5)

btn_dark = tk.Button(root, text="Capture Dark Field", width=25,
                     command=lambda: safe_run(dark.capture_dark_field, "dark field capture") if dark else None)
btn_dark.pack(pady=10)

root.mainloop()
