# las_peak_matcher.py (manual + placeholder + full auto + undo + export)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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

def load_spectrum(file):
    df = pd.read_csv(file, header=None, names=['wavelength', 'intensity'])
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df[(df['wavelength'] >= 290) & (df['wavelength'] <= 1000)]
    return df

def smooth(intensity):
    return gaussian_filter1d(intensity, sigma=SMOOTHING_SIGMA)

def analyze_spectrum(filename):
    if MODE == '1':
        print("[Mode 1: Manual] Interactive classification mode activated.")
        df = load_spectrum(filename)
        df['smoothed'] = smooth(df['intensity'])

        selected_type = {'type': 'Peak'}
        def select_type(t):
            selected_type['type'] = t
            print(f"Selected: {t}")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['wavelength'], df['smoothed'], color='orange')
        ax.set_title("Click to mark points; choose classification from popup")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Smoothed Intensity")

        root = tk.Tk()
        root.title("Manual Marking Tool - LAS")
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas.get_tk_widget().pack()

        tk.Label(root, text="Select Point Type to Tag:").pack(pady=5)
        for label in ['Peak', 'Trough', 'Crest', 'Valley', 'Shoulder']:
            tk.Button(root, text=label, command=lambda l=label: select_type(l)).pack(fill='x')

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
                manual_df.to_csv("structural_features_output_L.csv", index=False)
                print("✅ Manual selections saved to 'structural_features_output_L.csv'")
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

    elif MODE == '2':
        print("[Mode 2: Guided Auto] Not implemented.")
        return

    print("[Mode 3: Full Auto] LAS Structural Feature Detection")
    df = load_spectrum(filename)
    df['smoothed'] = smooth(df['intensity'])
    df['slope'] = np.gradient(df['smoothed'], df['wavelength'])
    df['curvature'] = np.gradient(df['slope'], df['wavelength'])

    reversals = []
    for i in range(1, len(df)):
        prev_slope = df.iloc[i - 1]['slope']
        curr_slope = df.iloc[i]['slope']
        slope_delta = abs(curr_slope - prev_slope)
        if slope_delta < 0.05:
            continue
        wl = df.iloc[i]['wavelength']
        y = df.iloc[i]['smoothed']
        if prev_slope > 0 and curr_slope < 0:
            reversals.append(('positive_to_negative', wl, y))
        elif prev_slope < 0 and curr_slope > 0:
            reversals.append(('negative_to_positive', wl, y))

    CLUSTER_SPAN = 4.0
    clusters = []
    temp = [reversals[0]]
    for i in range(1, len(reversals)):
        if reversals[i][1] - temp[-1][1] <= CLUSTER_SPAN:
            temp.append(reversals[i])
        else:
            if len(temp) >= 3:
                clusters.append(temp)
            temp = [reversals[i]]
    if len(temp) >= 3:
        clusters.append(temp)

    cluster_regions = []
    structural_markers = []
    for cluster in clusters:
        wls = [pt[1] for pt in cluster]
        yvals = [pt[2] for pt in cluster]
        mid = np.mean(wls)
        intensity_range = max(yvals) - min(yvals)
        noise_std = np.std(df['smoothed'][(df['wavelength'] >= 410) & (df['wavelength'] <= 425)])
        max_curvature = max(abs(df['curvature'][(df['wavelength'] >= min(wls)) & (df['wavelength'] <= max(wls))]))
        slope_range = df['slope'][(df['wavelength'] >= min(wls)) & (df['wavelength'] <= max(wls))].max() - \
                      df['slope'][(df['wavelength'] >= min(wls)) & (df['wavelength'] <= max(wls))].min()
        if intensity_range < 20.0 * noise_std or max_curvature < 0.01 or slope_range < 0.01:
            continue
        slope_after = df[df['wavelength'] > max(wls)].iloc[:5]['slope'].mean()
        label = 'Crest' if slope_after < 0 else 'Valley'
        y_val = df.loc[(df['wavelength'] - mid).abs().idxmin(), 'smoothed']
        structural_markers.append((label, mid, y_val))
        cluster_regions.append((min(wls), max(wls)))

    plt.figure(figsize=(14, 6))
    plt.plot(df['wavelength'], df['smoothed'], label='Smoothed', color='orange')
    for i, (lo, hi) in enumerate(cluster_regions):
        plt.axvspan(lo, hi, color='lightblue' if structural_markers[i][0]=='Crest' else 'lightgray', alpha=0.4)

    marker_shapes = {'Crest': ('D', 'blue'), 'Valley': ('h', 'black')}
    for label, wl, y in structural_markers:
        shape, color = marker_shapes[label]
        plt.scatter(wl, y, color=color, marker=shape, s=100, label=f"{label} @ {wl:.2f}")

    plt.title("[Auto Mode] LAS Spectrum Structural Features")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Smoothed Intensity")
    plt.grid(True)
    plt.legend(loc="lower right", fontsize="x-small", ncol=2)
    plt.tight_layout()
    plt.show()

    input("🧠 Save and exit? Press Enter to continue...")

if __name__ == '__main__':
    analyze_spectrum("unkgemL.csv")
