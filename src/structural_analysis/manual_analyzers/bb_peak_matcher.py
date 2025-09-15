# Step-by-step integrated version of bb_peak_matcher.py with skew computation option

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import skew
import tkinter as tk
import subprocess

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

SMOOTHING_SIGMA = 0.5

def choose_mode():
    root = tk.Tk()
    selected = {'mode': '3'}
    def set_mode(m):
        selected['mode'] = m
        root.quit()
    root.title("Select Detection Mode")
    tk.Label(root, text="Select Detection Mode", font=('Arial', 14)).pack(pady=10)
    tk.Button(root, text="1 = Manual", width=25, command=lambda: set_mode('1')).pack(pady=5)
    tk.Button(root, text="2 = Guided Auto (targeted)", width=25, command=lambda: set_mode('2')).pack(pady=5)
    tk.Button(root, text="3 = Full Auto (unbounded)", width=25, command=lambda: set_mode('3')).pack(pady=5)
    root.mainloop()
    return selected['mode']

MODE = choose_mode()

# Global variable to store skew
BROADBAND_SKEW = None


def load_spectrum(file):
    df = pd.read_csv(file, header=None, names=['wavelength', 'intensity'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df['wavelength'] >= 290) & (df['wavelength'] <= 1000)]
    return df

def smooth(intensity):
    return gaussian_filter1d(intensity, sigma=SMOOTHING_SIGMA)

def ask_compute_skew(df):
    global BROADBAND_SKEW
    root = tk.Tk()
    root.title("Compute Skew?")
    result = {'compute': False}

    def yes():
        result['compute'] = True
        root.quit()

    def no():
        root.quit()

    tk.Label(root, text="Compute broadband skew?", font=('Arial', 14)).pack(pady=10)
    tk.Button(root, text="Yes", width=20, command=yes).pack(pady=5)
    tk.Button(root, text="No", width=20, command=no).pack(pady=5)
    root.mainloop()

    if result['compute']:
        BROADBAND_SKEW = round(skew(df['smoothed']), 4)
        print(f"📉 Skewness of broadband curve: {BROADBAND_SKEW}")


def analyze_spectrum(filename):
    global BROADBAND_SKEW

    if MODE == '1':
        print("[Mode 1: Manual] Interactive classification mode activated.")
        df = load_spectrum(filename)
        df['smoothed'] = smooth(df['intensity'])

        ask_compute_skew(df)  # Ask user whether to compute skew

        selected_type = {'type': 'Peak'}
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['wavelength'], df['smoothed'], color='orange')
        ax.set_title("Click to mark points; choose classification from popup")
        ax.set_xlabel("Wavelength (nm")
        ax.set_ylabel("Smoothed Intensity")

        root = tk.Tk()
        root.title("Manual Marking Tool - BB")
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack()

        tk.Label(root, text="Select Point Type to Tag:").pack(pady=5)
        for label in ['Peak', 'Trough', 'Crest', 'Valley', 'Shoulder']:
            tk.Button(root, text=label, command=lambda l=label: selected_type.update({'type': l})).pack(fill='x')

        coords = []
        plotted_points = []

        def undo_last():
            if coords and len(plotted_points) >= 2:
                coords.pop()
                plotted_points.pop().remove()
                plotted_points.pop().remove()
                fig.canvas.draw()
                print("↩️ Last mark removed.")

        def onclick(event):
            if not hasattr(toolbar, 'mode') or toolbar.mode == '':
                if event.inaxes:
                    wl = event.xdata
                    intensity = event.ydata
                    label = selected_type['type']
                    color_map = {'Peak': 'green', 'Trough': 'red', 'Crest': 'blue', 'Valley': 'black', 'Shoulder': 'purple'}
                    marker_map = {'Peak': 'o', 'Trough': 'v', 'Crest': 'D', 'Valley': 'h', 'Shoulder': 's'}
                    point, = ax.plot(wl, intensity, marker=marker_map[label], color=color_map[label], markersize=4)
                    label_text = ax.text(wl + 2, intensity, f"{label[0]} {int(round(wl))}", fontsize=8, color=color_map[label])
                    coords.append((label, wl, intensity))
                    plotted_points.append(point)
                    plotted_points.append(label_text)
                    fig.canvas.draw()
                    print(f"Marked {label} at {wl:.2f} nm")

        canvas.mpl_connect('button_press_event', onclick)

        def save_marks():
            if coords:
                manual_df = pd.DataFrame(coords, columns=['Type', 'Wavelength', 'Intensity'])
                manual_df['Wavelength'] = manual_df['Wavelength'].round(0).astype(int)
                if BROADBAND_SKEW is not None:
                    manual_df.loc[len(manual_df)] = ['Skew', 0, BROADBAND_SKEW]  # Store skew as a row
                manual_df.to_csv("structural_features_output_B.csv", index=False)
                print("✅ Manual selections saved to 'structural_features_output_B.csv'")
                try:
                    subprocess.run(['python', 'structural_feature_to_gemlib.py'], check=True)
                    print("✅ Structural features inserted into gem library.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to insert into gem library: {e}")
                root.quit()
                root.destroy()

        tk.Button(root, text="Undo Last Mark", command=undo_last, fg='darkred').pack(pady=10)
        tk.Button(root, text="Save to File + Library", command=save_marks, fg='darkgreen').pack(pady=10)
        root.mainloop()
        return

    print("[Mode 3: Full Auto] Skew handling not yet integrated.")

if __name__ == '__main__':
    analyze_spectrum("unkgemB.csv")
