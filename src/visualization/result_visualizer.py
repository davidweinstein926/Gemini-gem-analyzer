#!/usr/bin/env python3
"""
result_visualizer.py - For src/visualization/ folder
Comprehensive Results Display System for gemini_gemological_analysis structure
Save as: src/visualization/result_visualizer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import seaborn as sns
from datetime import datetime
import os

class GeminiResultsVisualizer:
    """Enhanced results display - works with existing directory structure"""
    
    def __init__(self):
        self.gem_name_map = self.load_gemstone_library()
        self.setup_plot_style()
        
    def load_gemstone_library(self):
        """Load gemstone information from database/gem_library/gemlib_structural_ready.csv"""
        try:
            gemlib = pd.read_csv('database/gem_library/gemlib_structural_ready.csv')
            gemlib.columns = gemlib.columns.str.strip()
            
            if 'Reference' in gemlib.columns:
                gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                
                if all(col in gemlib.columns for col in expected_columns):
                    gemlib['Gem Description'] = gemlib[expected_columns].apply(
                        lambda x: ' '.join([v if pd.notnull(v) else '' for v in x]).strip(), axis=1)
                    return dict(zip(gemlib['Reference'], gemlib['Gem Description']))
                    
            print("⚠️ Using fallback gem descriptions")
            return {}
            
        except Exception as e:
            print(f"⚠️ Could not load gemstone library: {e}")
            return {}
    
    def setup_plot_style(self):
        """Configure matplotlib style for professional plots"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Custom color scheme for light sources
        self.light_colors = {
            'B': '#FF6B35',  # Orange for Halogen/Broadband
            'L': '#004E98',  # Blue for Laser  
            'U': '#7209B7'   # Purple for UV
        }
        
        self.plot_config = {
            'figure.figsize': (15, 8),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 10
        }
        
        for key, value in self.plot_config.items():
            plt.rcParams[key] = value
    
    def create_score_summary_gui(self, results_data):
        """Create interactive score summary window"""
        
        # Create main window
        root = tk.Tk()
        root.title("Gemini Analysis Results - Score Summary")
        root.geometry("1200x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Overall Rankings
        self.create_rankings_tab(notebook, results_data)
        
        # Tab 2: Detailed Scores
        self.create_detailed_scores_tab(notebook, results_data)
        
        # Tab 3: Light Source Analysis
        self.create_light_source_tab(notebook, results_data)
        
        # Tab 4: Spectral Comparisons
        self.create_spectral_comparison_tab(notebook, results_data)
        
        # Status bar
        status_frame = tk.Frame(root)
        status_frame.pack(fill='x', side='bottom')
        
        status_label = tk.Label(status_frame, 
                               text=f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                                   f"Total matches: {len(results_data.get('final_sorted', []))} | "
                                   f"Light sources: {', '.join(results_data.get('light_sources', []))}")
        status_label.pack(side='left', padx=10, pady=5)
        
        root.mainloop()
    
    def create_rankings_tab(self, notebook, results_data):
        """Create overall rankings display tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="🏆 Overall Rankings")
        
        # Title
        title = tk.Label(frame, text="GEMINI GEMOLOGICAL ANALYSIS - TOP MATCHES", 
                        font=('Arial', 16, 'bold'), fg='darkblue')
        title.pack(pady=10)
        
        # Results tree
        tree_frame = tk.Frame(frame)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient='vertical')
        h_scrollbar = ttk.Scrollbar(tree_frame, orient='horizontal')
        
        # Treeview
        columns = ('Rank', 'Gem_ID', 'Description', 'Total_Score', 'B_Score', 'L_Score', 'U_Score', 'Sources')
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                           yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Configure columns
        column_widths = {'Rank': 60, 'Gem_ID': 80, 'Description': 300, 'Total_Score': 100, 
                        'B_Score': 80, 'L_Score': 80, 'U_Score': 80, 'Sources': 100}
        
        for col in columns:
            tree.heading(col, text=col.replace('_', ' '))
            tree.column(col, width=column_widths.get(col, 100))
        
        # Populate data
        final_sorted = results_data.get('final_sorted', [])
        gem_best_scores = results_data.get('gem_best_scores', {})
        
        for i, (gem_id, total_score) in enumerate(final_sorted[:50], 1):  # Top 50
            gem_desc = self.gem_name_map.get(str(gem_id), f"Unknown Gem {gem_id}")
            scores = gem_best_scores.get(gem_id, {})
            
            # Color coding for rankings
            tag = 'top10' if i <= 10 else 'top25' if i <= 25 else 'others'
            
            tree.insert('', 'end', values=(
                i, gem_id, gem_desc, f"{total_score:.2f}",
                f"{scores.get('B', 'N/A'):.2f}" if scores.get('B') else 'N/A',
                f"{scores.get('L', 'N/A'):.2f}" if scores.get('L') else 'N/A', 
                f"{scores.get('U', 'N/A'):.2f}" if scores.get('U') else 'N/A',
                ', '.join(sorted(scores.keys()))
            ), tags=(tag,))
        
        # Configure tags for color coding
        tree.tag_configure('top10', background='lightgreen')
        tree.tag_configure('top25', background='lightyellow')
        tree.tag_configure('others', background='white')
        
        # Pack components
        tree.pack(side='left', fill='both', expand=True)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        
        v_scrollbar.config(command=tree.yview)
        h_scrollbar.config(command=tree.xview)
        
        # Export button
        export_btn = tk.Button(frame, text="📊 Export Results to CSV", 
                              command=lambda: self.export_results(results_data))
        export_btn.pack(pady=10)
    
    def create_detailed_scores_tab(self, notebook, results_data):
        """Create detailed score analysis tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📊 Score Analysis")
        
        # Create matplotlib figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        final_sorted = results_data.get('final_sorted', [])[:20]  # Top 20
        gem_best_scores = results_data.get('gem_best_scores', {})
        
        if final_sorted:
            # Prepare data
            gem_ids = [str(item[0]) for item in final_sorted]
            total_scores = [item[1] for item in final_sorted]
            
            gem_descriptions = [self.gem_name_map.get(gid, f"Gem {gid}")[:30] + "..." 
                              if len(self.gem_name_map.get(gid, f"Gem {gid}")) > 30 
                              else self.gem_name_map.get(gid, f"Gem {gid}") 
                              for gid in gem_ids]
            
            # Plot 1: Total Scores Bar Chart
            bars = ax1.bar(range(len(gem_descriptions)), total_scores, color='darkblue', alpha=0.7)
            ax1.set_title('Top 20 Matches - Total Scores', fontweight='bold')
            ax1.set_xlabel('Gemstone Matches')
            ax1.set_ylabel('Total Log Score')
            ax1.set_xticks(range(len(gem_descriptions)))
            ax1.set_xticklabels(gem_descriptions, rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
            
            # Plot 2: Score Distribution Histogram
            ax2.hist(total_scores, bins=10, color='darkgreen', alpha=0.7, edgecolor='black')
            ax2.set_title('Score Distribution', fontweight='bold')
            ax2.set_xlabel('Total Log Score')
            ax2.set_ylabel('Frequency')
            
            # Plot 3: Light Source Scores Comparison
            light_scores = {'B': [], 'L': [], 'U': []}
            for gem_id, _ in final_sorted:
                scores = gem_best_scores.get(gem_id, {})
                for light in ['B', 'L', 'U']:
                    if light in scores:
                        light_scores[light].append(scores[light])
            
            positions = []
            labels = []
            data_for_box = []
            
            for i, (light, scores) in enumerate(light_scores.items()):
                if scores:
                    positions.append(i + 1)
                    labels.append(f"{light} ({len(scores)})")
                    data_for_box.append(scores)
            
            if data_for_box:
                box_plot = ax3.boxplot(data_for_box, positions=positions, patch_artist=True)
                
                # Color the boxes
                colors = [self.light_colors[light] for light in ['B', 'L', 'U'] if light_scores[light]]
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax3.set_title('Score Distribution by Light Source', fontweight='bold')
                ax3.set_xlabel('Light Source')
                ax3.set_ylabel('Individual Scores')
                ax3.set_xticklabels(labels)
            
            # Plot 4: Score Correlation Matrix (if multiple light sources)
            available_lights = [light for light in ['B', 'L', 'U'] if light_scores[light]]
            if len(available_lights) >= 2:
                corr_data = []
                for gem_id, _ in final_sorted:
                    scores = gem_best_scores.get(gem_id, {})
                    row = [scores.get(light, np.nan) for light in available_lights]
                    if not any(np.isnan(row)):
                        corr_data.append(row)
                
                if len(corr_data) > 1:
                    corr_df = pd.DataFrame(corr_data, columns=available_lights)
                    correlation_matrix = corr_df.corr()
                    
                    im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    ax4.set_title('Light Source Score Correlations', fontweight='bold')
                    ax4.set_xticks(range(len(available_lights)))
                    ax4.set_yticks(range(len(available_lights)))
                    ax4.set_xticklabels(available_lights)
                    ax4.set_yticklabels(available_lights)
                    
                    # Add correlation values as text
                    for i in range(len(available_lights)):
                        for j in range(len(available_lights)):
                            text = ax4.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                          ha="center", va="center", color="black", fontweight='bold')
                    
                    plt.colorbar(im, ax=ax4, shrink=0.6)
                else:
                    ax4.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, f'Only {len(available_lights)} light source(s)\navailable', 
                        ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Embed in tkinter
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def create_light_source_tab(self, notebook, results_data):
        """Create light source specific analysis tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="💡 Light Source Analysis")
        
        # Light source specific statistics
        all_matches = results_data.get('all_matches', {})
        
        info_text = tk.Text(frame, height=20, width=80, font=('Courier', 10))
        info_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        analysis_text = "LIGHT SOURCE ANALYSIS SUMMARY\n"
        analysis_text += "=" * 60 + "\n\n"
        
        for light_source in ['B', 'L', 'U']:
            if light_source in all_matches:
                matches = all_matches[light_source]
                analysis_text += f"🔬 {light_source} LIGHT SOURCE ANALYSIS:\n"
                analysis_text += f"   Total matches found: {len(matches)}\n"
                
                if matches:
                    scores = [score for _, score in matches]
                    analysis_text += f"   Best score: {min(scores):.3f}\n"
                    analysis_text += f"   Worst score: {max(scores):.3f}\n"
                    analysis_text += f"   Average score: {np.mean(scores):.3f}\n"
                    analysis_text += f"   Score std dev: {np.std(scores):.3f}\n"
                    
                    analysis_text += f"\n   🏆 TOP 10 MATCHES:\n"
                    for i, (gem_name, score) in enumerate(matches[:10], 1):
                        gem_id = gem_name.split(light_source)[0]
                        gem_desc = self.gem_name_map.get(gem_id, f"Gem {gem_id}")
                        analysis_text += f"   {i:2}. {gem_desc[:40]:<40} Score: {score:.3f}\n"
                
                analysis_text += "\n" + "-" * 60 + "\n\n"
        
        info_text.insert(1.0, analysis_text)
        info_text.config(state='disabled')
    
    def create_spectral_comparison_tab(self, notebook, results_data):
        """Create spectral overlay comparison tab"""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text="📈 Spectral Comparisons")
        
        # Control frame
        control_frame = tk.Frame(frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(control_frame, text="Select match to visualize:", font=('Arial', 12, 'bold')).pack(side='left')
        
        # Dropdown for gem selection
        final_sorted = results_data.get('final_sorted', [])
        gem_options = [f"Rank {i}: {self.gem_name_map.get(str(gem_id), f'Gem {gem_id}')}" 
                      for i, (gem_id, _) in enumerate(final_sorted[:20], 1)]
        
        selected_gem = tk.StringVar(value=gem_options[0] if gem_options else "No matches")
        gem_dropdown = ttk.Combobox(control_frame, textvariable=selected_gem, values=gem_options, width=50)
        gem_dropdown.pack(side='left', padx=10)
        
        plot_button = tk.Button(control_frame, text="📊 Plot Comparison", 
                               command=lambda: self.plot_spectral_comparison(selected_gem.get(), results_data))
        plot_button.pack(side='left', padx=10)
        
        # Placeholder for plots
        plot_frame = tk.Frame(frame)
        plot_frame.pack(fill='both', expand=True)
        
        placeholder_label = tk.Label(plot_frame, text="Select a match and click 'Plot Comparison' to view spectral overlays",
                                    font=('Arial', 14), fg='gray')
        placeholder_label.pack(expand=True)
    
    def plot_spectral_comparison(self, selected_option, results_data):
        """Plot spectral comparison for selected gem"""
        if not selected_option or "No matches" in selected_option:
            return
        
        # Extract rank from selection
        rank = int(selected_option.split(":")[0].replace("Rank ", ""))
        final_sorted = results_data.get('final_sorted', [])
        
        if rank <= len(final_sorted):
            gem_id, total_score = final_sorted[rank - 1]
            gem_best_names = results_data.get('gem_best_names', {})
            
            # Create comparison plot
            self.create_horizontal_comparison_plot(gem_id, gem_best_names, total_score)
    
    def create_horizontal_comparison_plot(self, gem_id, gem_best_names, total_score):
        """Create horizontal spectral comparison plot"""
        
        unknown_files = {
            'B': 'data/unknown/unkgemB.csv', 
            'L': 'data/unknown/unkgemL.csv', 
            'U': 'data/unknown/unkgemU.csv'
        }
        
        db_files = {
            'B': 'database/reference_spectra/gemini_db_long_B.csv',
            'L': 'database/reference_spectra/gemini_db_long_L.csv', 
            'U': 'database/reference_spectra/gemini_db_long_U.csv'
        }
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        gem_desc = self.gem_name_map.get(str(gem_id), f"Gem {gem_id}")
        
        for i, light_source in enumerate(['B', 'L', 'U']):
            try:
                # Load unknown spectrum
                unknown = pd.read_csv(unknown_files[light_source], header=None, names=['wavelength', 'intensity'])
                
                # Load database
                db = pd.read_csv(db_files[light_source])
                
                # Get best match for this light source
                match_name = gem_best_names.get(gem_id, {}).get(light_source)
                
                if match_name:
                    reference = db[db['full_name'] == match_name]
                    
                    # Plot spectra
                    axs[i].plot(unknown['wavelength'], unknown['intensity'], 
                               label=f"Unknown {light_source}", color=self.light_colors[light_source], 
                               linewidth=2, alpha=0.8)
                    
                    axs[i].plot(reference['wavelength'], reference['intensity'], 
                               label=f"Match: {gem_desc}", color='black', 
                               linestyle='--', linewidth=2, alpha=0.8)
                    
                    axs[i].set_xlabel('Wavelength (nm)', fontsize=11)
                    axs[i].set_ylabel('Intensity', fontsize=11)
                    axs[i].set_title(f"{light_source} Light Comparison\n{gem_desc}", fontsize=12, fontweight='bold')
                    axs[i].legend(fontsize=10)
                    axs[i].grid(True, alpha=0.3)
                    
                    # Add score annotation
                    score_text = f"Match Score: {total_score:.2f}"
                    axs[i].text(0.02, 0.98, score_text, transform=axs[i].transAxes, 
                               fontsize=10, verticalalignment='top', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                else:
                    axs[i].text(0.5, 0.5, f"{light_source}: No match data available", 
                               ha='center', va='center', transform=axs[i].transAxes, fontsize=12)
                    axs[i].set_title(f"{light_source} Light: No Data", fontsize=12)
                    
            except FileNotFoundError as e:
                axs[i].text(0.5, 0.5, f"{light_source}: File not found", 
                           ha='center', va='center', transform=axs[i].transAxes, fontsize=12)
                axs[i].set_title(f"{light_source} Light: Missing Data", fontsize=12)
        
        plt.suptitle(f"Spectral Analysis: {gem_desc} (Total Score: {total_score:.3f})", 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def export_results(self, results_data):
        """Export results to CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_analysis_results_{timestamp}.csv"
            
            final_sorted = results_data.get('final_sorted', [])
            gem_best_scores = results_data.get('gem_best_scores', {})
            
            export_data = []
            for i, (gem_id, total_score) in enumerate(final_sorted, 1):
                gem_desc = self.gem_name_map.get(str(gem_id), f"Unknown Gem {gem_id}")
                scores = gem_best_scores.get(gem_id, {})
                
                export_data.append({
                    'Rank': i,
                    'Gem_ID': gem_id,
                    'Description': gem_desc,
                    'Total_Score': total_score,
                    'B_Score': scores.get('B', ''),
                    'L_Score': scores.get('L', ''),
                    'U_Score': scores.get('U', ''),
                    'Light_Sources': ', '.join(sorted(scores.keys())),
                    'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            print(f"✅ Results exported to: {filename}")
            tk.messagebox.showinfo("Export Complete", f"Results exported to:\n{filename}")
            
        except Exception as e:
            print(f"❌ Export error: {e}")
            tk.messagebox.showerror("Export Error", f"Failed to export results:\n{e}")

# Integration function for main analysis workflow
def display_analysis_results(all_matches, gem_best_scores, gem_best_names, final_sorted, light_sources):
    """Main function to display comprehensive analysis results"""
    
    results_data = {
        'all_matches': all_matches,
        'gem_best_scores': gem_best_scores, 
        'gem_best_names': gem_best_names,
        'final_sorted': final_sorted,
        'light_sources': light_sources
    }
    
    visualizer = GeminiResultsVisualizer()
    visualizer.create_score_summary_gui(results_data)

if __name__ == "__main__":
    # Example usage - this would be called from gemini1.py
    print("🔬 Gemini Results Visualizer - Ready for integration")
    print("This module should be imported and called from the main analysis workflow")