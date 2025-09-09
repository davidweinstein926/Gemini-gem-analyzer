#!/usr/bin/env python3
"""
gem_results_visualizer.py - GEMINI RESULTS VISUALIZATION SYSTEM
Advanced visualization and score summary for gemstone analysis results
Save as: gemini_gemological_analysis/gem_results_visualizer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Button, RadioButtons
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

class GemResultsVisualizer:
    def __init__(self):
        self.unknown_files = {'B': 'data/unknown/unkgemB.csv', 'L': 'data/unknown/unkgemL.csv', 'U': 'data/unknown/unkgemU.csv'}
        self.db_files = {'B': 'gemini_db_long_B.csv', 'L': 'gemini_db_long_L.csv', 'U': 'gemini_db_long_U.csv'}
        self.results_data = {}
        self.gem_descriptions = {}
        self.load_gem_library()
    
    def load_gem_library(self):
        """Load gem descriptions from gemlib_structural_ready.csv"""
        try:
            gemlib = pd.read_csv('gemlib_structural_ready.csv')
            gemlib.columns = gemlib.columns.str.strip()
            
            if 'Reference' in gemlib.columns:
                gemlib['Reference'] = gemlib['Reference'].astype(str).str.strip()
                expected_columns = ['Nat./Syn.', 'Spec.', 'Var.', 'Treatment', 'Origin']
                
                if all(col in gemlib.columns for col in expected_columns):
                    gemlib['Description'] = gemlib[expected_columns].apply(
                        lambda x: ' '.join([v if pd.notnull(v) and str(v).strip() else '' 
                                          for v in x]).strip(), axis=1)
                    self.gem_descriptions = dict(zip(gemlib['Reference'], gemlib['Description']))
                    print(f"✅ Loaded {len(self.gem_descriptions)} gem descriptions")
                else:
                    print(f"⚠️ Missing columns in gemlib: {[c for c in expected_columns if c not in gemlib.columns]}")
            else:
                print("⚠️ 'Reference' column not found in gemlib_structural_ready.csv")
                
        except FileNotFoundError:
            print("⚠️ gemlib_structural_ready.csv not found - using generic descriptions")
        except Exception as e:
            print(f"⚠️ Error loading gemlib: {e}")
    
    def get_gem_description(self, gem_id):
        """Get descriptive name for gem ID"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        return self.gem_descriptions.get(base_id, f"Gem {base_id}")
    
    def run_analysis_and_visualize(self):
        """Run complete analysis and show results"""
        print("🔍 RUNNING COMPREHENSIVE GEM ANALYSIS")
        print("=" * 50)
        
        # Run analysis
        results = self.analyze_unknown_gem()
        
        if results:
            # Show score summary
            self.show_score_summary_window(results)
            
            # Create visualizations
            self.create_comparison_plots(results)
            
            # Show interactive results
            self.show_interactive_results(results)
        else:
            print("❌ No analysis results to display")
    
    def analyze_unknown_gem(self):
        """Analyze unknown gem against database"""
        try:
            # Detect available light sources
            available_sources = self.detect_available_sources()
            print(f"🔍 Unknown gem uses {len(available_sources)} light sources: {', '.join(sorted(available_sources))}")
            
            all_matches = {}
            gem_best_scores = {}
            gem_best_names = {}
            
            # Analyze each light source
            for light_source in available_sources:
                matches = self.analyze_light_source(light_source)
                if matches:
                    all_matches[light_source] = matches
                    
                    # Track best scores per gem
                    for gem_name, score in matches:
                        base_id = gem_name.split('B')[0].split('L')[0].split('U')[0]
                        if base_id not in gem_best_scores:
                            gem_best_scores[base_id] = {}
                            gem_best_names[base_id] = {}
                        
                        if score < gem_best_scores[base_id].get(light_source, np.inf):
                            gem_best_scores[base_id][light_source] = score
                            gem_best_names[base_id][light_source] = gem_name
            
            # Filter to gems with all available light sources
            complete_gems = {gid: scores for gid, scores in gem_best_scores.items() 
                           if set(scores.keys()) == available_sources}
            
            # Calculate aggregate scores
            aggregated_scores = {base_id: sum(scores[ls] for ls in available_sources) 
                               for base_id, scores in complete_gems.items()}
            
            # Sort by total score
            final_sorted = sorted(aggregated_scores.items(), key=lambda x: x[1])
            
            return {
                'available_sources': available_sources,
                'all_matches': all_matches,
                'gem_best_scores': complete_gems,
                'gem_best_names': {gid: names for gid, names in gem_best_names.items() if gid in complete_gems},
                'final_ranking': final_sorted[:20]  # Top 20 matches
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return None
    
    def detect_available_sources(self):
        """Detect which light sources are available"""
        available = set()
        
        # Check raw_txt directory
        if os.path.isdir('raw_txt'):
            for f in os.listdir('raw_txt'):
                if f.lower().endswith('.txt'):
                    base = os.path.splitext(f)[0]
                    if len(base) >= 3:
                        light_source = base[-3].upper()
                        if light_source in {'B', 'L', 'U'}:
                            available.add(light_source)
        
        # Check unknown files
        for light in ['B', 'L', 'U']:
            if os.path.exists(self.unknown_files[light]):
                available.add(light)
        
        return available if available else {'B', 'L', 'U'}  # Default assumption
    
    def analyze_light_source(self, light_source):
        """Analyze single light source"""
        try:
            # Load unknown spectrum
            unknown = pd.read_csv(self.unknown_files[light_source], 
                                sep=r'[\s,]+', header=None, 
                                names=['wavelength', 'intensity'], 
                                skiprows=1, engine='python')
            
            # Load database
            db = pd.read_csv(self.db_files[light_source])
            
            scores = []
            for gem_name in db['full_name'].unique():
                reference = db[db['full_name'] == gem_name]
                score = self.compute_match_score(unknown, reference)
                scores.append((gem_name, score))
            
            # Sort by score (lower is better)
            sorted_scores = sorted(scores, key=lambda x: x[1])
            
            print(f"\n✅ Best Matches for {light_source}:")
            for i, (gem, score) in enumerate(sorted_scores[:5], 1):
                desc = self.get_gem_description(gem.split('B')[0].split('L')[0].split('U')[0])
                print(f"  {i}. {desc} ({gem}): Score = {score:.2f}")
            
            return sorted_scores
            
        except FileNotFoundError as e:
            print(f"⚠️ File not found for {light_source}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error analyzing {light_source}: {e}")
            return None
    
    def compute_match_score(self, unknown, reference):
        """Compute match score between unknown and reference"""
        try:
            merged = pd.merge(unknown, reference, on='wavelength', suffixes=('_unknown', '_ref'))
            if len(merged) == 0:
                return np.inf
            
            mse = np.mean((merged['intensity_unknown'] - merged['intensity_ref']) ** 2)
            return np.log1p(mse)  # Log score for better distribution
            
        except Exception:
            return np.inf
    
    def show_score_summary_window(self, results):
        """Show comprehensive score summary in GUI window"""
        root = tk.Tk()
        root.title("🏆 Gemstone Analysis Results - Score Summary")
        root.geometry("1000x700")
        root.configure(bg='#f0f0f0')
        
        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = tk.Label(main_frame, text="🔬 GEMSTONE IDENTIFICATION RESULTS", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0', fg='#2c3e50')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Analysis info
        info_text = f"Analysis Sources: {', '.join(sorted(results['available_sources']))} ({len(results['available_sources'])} light sources)"
        info_label = tk.Label(main_frame, text=info_text, font=('Arial', 10), bg='#f0f0f0', fg='#7f8c8d')
        info_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        # Results table
        self.create_results_table(main_frame, results)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="📊 Show Plots", 
                  command=lambda: self.create_comparison_plots(results)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="📋 Export Results", 
                  command=lambda: self.export_results(results)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔍 Detailed Analysis", 
                  command=lambda: self.show_detailed_analysis(results)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="❌ Close", 
                  command=root.destroy).pack(side=tk.LEFT, padx=5)
        
        root.mainloop()
    
    def create_results_table(self, parent, results):
        """Create results table with scores"""
        # Table frame
        table_frame = ttk.Frame(parent)
        table_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Treeview for results
        columns = ['Rank', 'Gem Description', 'Gem ID', 'Total Score'] + [f'{ls} Score' for ls in sorted(results['available_sources'])]
        tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        tree.heading('Rank', text='Rank')
        tree.heading('Gem Description', text='Gem Description')
        tree.heading('Gem ID', text='Gem ID')
        tree.heading('Total Score', text='Total Score')
        
        tree.column('Rank', width=50)
        tree.column('Gem Description', width=300)
        tree.column('Gem ID', width=80)
        tree.column('Total Score', width=100)
        
        for ls in sorted(results['available_sources']):
            tree.heading(f'{ls} Score', text=f'{ls} Score')
            tree.column(f'{ls} Score', width=80)
        
        # Populate table
        for i, (gem_id, total_score) in enumerate(results['final_ranking'], 1):
            description = self.get_gem_description(gem_id)
            
            # Get individual scores
            scores = results['gem_best_scores'].get(gem_id, {})
            score_values = [f"{scores.get(ls, 'N/A'):.2f}" if isinstance(scores.get(ls), (int, float)) else "N/A" 
                           for ls in sorted(results['available_sources'])]
            
            # Color coding based on rank
            tags = []
            if i <= 3:
                tags = ['top3']
            elif i <= 10:
                tags = ['top10']
            
            values = [str(i), description, str(gem_id), f"{total_score:.2f}"] + score_values
            tree.insert('', 'end', values=values, tags=tags)
        
        # Configure tags for colors
        tree.tag_configure('top3', background='#e8f5e8')  # Light green for top 3
        tree.tag_configure('top10', background='#fff3cd')  # Light yellow for top 10
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
    
    def create_comparison_plots(self, results):
        """Create interactive comparison plots"""
        if not results['final_ranking']:
            print("❌ No results to plot")
            return
        
        # Get top 3 matches for plotting
        top_matches = results['final_ranking'][:3]
        
        # Create subplots for each light source
        n_sources = len(results['available_sources'])
        fig, axes = plt.subplots(n_sources, 1, figsize=(15, 5*n_sources))
        if n_sources == 1:
            axes = [axes]
        
        fig.suptitle('🔬 Gemstone Spectral Comparison - Top Matches', fontsize=16, fontweight='bold')
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']  # Red, Blue, Green for top 3
        
        for i, light_source in enumerate(sorted(results['available_sources'])):
            ax = axes[i]
            
            try:
                # Load unknown spectrum
                unknown = pd.read_csv(self.unknown_files[light_source], 
                                    sep=r'[\s,]+', header=None, 
                                    names=['wavelength', 'intensity'], 
                                    skiprows=1, engine='python')
                
                # Plot unknown spectrum
                ax.plot(unknown['wavelength'], unknown['intensity'], 
                       label='Unknown Sample', color='black', linewidth=2.5, alpha=0.8)
                
                # Load database and plot top matches
                db = pd.read_csv(self.db_files[light_source])
                
                for j, (gem_id, total_score) in enumerate(top_matches):
                    if gem_id in results['gem_best_names'] and light_source in results['gem_best_names'][gem_id]:
                        gem_name = results['gem_best_names'][gem_id][light_source]
                        reference = db[db['full_name'] == gem_name]
                        
                        if not reference.empty:
                            description = self.get_gem_description(gem_id)
                            individual_score = results['gem_best_scores'][gem_id].get(light_source, 0)
                            
                            ax.plot(reference['wavelength'], reference['intensity'], 
                                   label=f'#{j+1}: {description} (Score: {individual_score:.2f})', 
                                   color=colors[j], linestyle='--', linewidth=1.8, alpha=0.7)
                
                ax.set_xlabel('Wavelength (nm)', fontsize=12)
                ax.set_ylabel('Intensity', fontsize=12)
                ax.set_title(f'{light_source} Light Source Analysis', fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                
                # Set background color based on light source
                if light_source == 'B':
                    ax.set_facecolor('#fafafa')
                elif light_source == 'L':
                    ax.set_facecolor('#f0f8ff')
                elif light_source == 'U':
                    ax.set_facecolor('#f5f0ff')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading {light_source} data: {e}', 
                       transform=ax.transAxes, ha='center', va='center', 
                       fontsize=12, color='red')
                ax.set_title(f'{light_source} Light Source - Data Error', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def show_detailed_analysis(self, results):
        """Show detailed analysis window"""
        detail_window = tk.Toplevel()
        detail_window.title("🔍 Detailed Analysis Results")
        detail_window.geometry("900x600")
        
        # Notebook for tabbed interface
        notebook = ttk.Notebook(detail_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Statistical Summary
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="📊 Statistical Summary")
        
        stats_text = ScrolledText(stats_frame, wrap=tk.WORD, width=100, height=30)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate statistical summary
        self.generate_statistical_summary(stats_text, results)
        
        # Tab 2: Match Quality Analysis
        quality_frame = ttk.Frame(notebook)
        notebook.add(quality_frame, text="🎯 Match Quality")
        
        quality_text = ScrolledText(quality_frame, wrap=tk.WORD, width=100, height=30)
        quality_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.generate_quality_analysis(quality_text, results)
        
        # Tab 3: Recommendations
        rec_frame = ttk.Frame(notebook)
        notebook.add(rec_frame, text="💡 Recommendations")
        
        rec_text = ScrolledText(rec_frame, wrap=tk.WORD, width=100, height=30)
        rec_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.generate_recommendations(rec_text, results)
    
    def generate_statistical_summary(self, text_widget, results):
        """Generate statistical summary"""
        text_widget.insert(tk.END, "🔬 STATISTICAL ANALYSIS SUMMARY\n")
        text_widget.insert(tk.END, "=" * 60 + "\n\n")
        
        text_widget.insert(tk.END, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        text_widget.insert(tk.END, f"Light Sources Used: {', '.join(sorted(results['available_sources']))}\n")
        text_widget.insert(tk.END, f"Total Candidates Analyzed: {len(results['final_ranking'])}\n\n")
        
        # Score statistics
        if results['final_ranking']:
            scores = [score for _, score in results['final_ranking']]
            text_widget.insert(tk.END, "📈 SCORE DISTRIBUTION:\n")
            text_widget.insert(tk.END, f"   Best Score: {min(scores):.3f}\n")
            text_widget.insert(tk.END, f"   Worst Score: {max(scores):.3f}\n")
            text_widget.insert(tk.END, f"   Mean Score: {np.mean(scores):.3f}\n")
            text_widget.insert(tk.END, f"   Median Score: {np.median(scores):.3f}\n")
            text_widget.insert(tk.END, f"   Std Deviation: {np.std(scores):.3f}\n\n")
        
        # Top 10 detailed results
        text_widget.insert(tk.END, "🏆 TOP 10 DETAILED RESULTS:\n")
        text_widget.insert(tk.END, "-" * 60 + "\n")
        
        for i, (gem_id, total_score) in enumerate(results['final_ranking'][:10], 1):
            description = self.get_gem_description(gem_id)
            text_widget.insert(tk.END, f"{i:2d}. {description} (ID: {gem_id})\n")
            text_widget.insert(tk.END, f"    Total Score: {total_score:.3f}\n")
            
            # Individual light source scores
            scores = results['gem_best_scores'].get(gem_id, {})
            for ls in sorted(results['available_sources']):
                score = scores.get(ls, 'N/A')
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else str(score)
                text_widget.insert(tk.END, f"    {ls} Light: {score_str}\n")
            text_widget.insert(tk.END, "\n")
    
    def generate_quality_analysis(self, text_widget, results):
        """Generate match quality analysis"""
        text_widget.insert(tk.END, "🎯 MATCH QUALITY ANALYSIS\n")
        text_widget.insert(tk.END, "=" * 60 + "\n\n")
        
        if not results['final_ranking']:
            text_widget.insert(tk.END, "No results to analyze.\n")
            return
        
        # Quality tiers
        scores = [score for _, score in results['final_ranking']]
        q1 = np.percentile(scores, 25)
        q2 = np.percentile(scores, 50)
        q3 = np.percentile(scores, 75)
        
        text_widget.insert(tk.END, "📊 QUALITY TIERS:\n")
        text_widget.insert(tk.END, f"   Excellent (≤{q1:.2f}): {sum(1 for s in scores if s <= q1)} gems\n")
        text_widget.insert(tk.END, f"   Good ({q1:.2f}-{q2:.2f}): {sum(1 for s in scores if q1 < s <= q2)} gems\n")
        text_widget.insert(tk.END, f"   Fair ({q2:.2f}-{q3:.2f}): {sum(1 for s in scores if q2 < s <= q3)} gems\n")
        text_widget.insert(tk.END, f"   Poor (>{q3:.2f}): {sum(1 for s in scores if s > q3)} gems\n\n")
        
        # Best match analysis
        best_gem_id, best_score = results['final_ranking'][0]
        text_widget.insert(tk.END, f"🏆 BEST MATCH ANALYSIS:\n")
        text_widget.insert(tk.END, f"   Gem: {self.get_gem_description(best_gem_id)} (ID: {best_gem_id})\n")
        text_widget.insert(tk.END, f"   Overall Score: {best_score:.3f}\n")
        
        best_scores = results['gem_best_scores'].get(best_gem_id, {})
        text_widget.insert(tk.END, "   Individual Scores:\n")
        for ls in sorted(results['available_sources']):
            score = best_scores.get(ls, 'N/A')
            text_widget.insert(tk.END, f"     {ls} Light: {score:.3f if isinstance(score, (int, float)) else score}\n")
        
        # Confidence assessment
        if len(results['final_ranking']) > 1:
            second_score = results['final_ranking'][1][1]
            confidence = (second_score - best_score) / best_score * 100
            text_widget.insert(tk.END, f"\n🎯 CONFIDENCE ANALYSIS:\n")
            text_widget.insert(tk.END, f"   Score Gap: {second_score - best_score:.3f}\n")
            text_widget.insert(tk.END, f"   Relative Confidence: {confidence:.1f}%\n")
            
            if confidence > 50:
                text_widget.insert(tk.END, "   Assessment: HIGH CONFIDENCE match\n")
            elif confidence > 20:
                text_widget.insert(tk.END, "   Assessment: MODERATE CONFIDENCE match\n")
            else:
                text_widget.insert(tk.END, "   Assessment: LOW CONFIDENCE - multiple candidates\n")
    
    def generate_recommendations(self, text_widget, results):
        """Generate analysis recommendations"""
        text_widget.insert(tk.END, "💡 ANALYSIS RECOMMENDATIONS\n")
        text_widget.insert(tk.END, "=" * 60 + "\n\n")
        
        if not results['final_ranking']:
            text_widget.insert(tk.END, "No results available for recommendations.\n")
            return
        
        best_score = results['final_ranking'][0][1]
        
        text_widget.insert(tk.END, "🔍 IDENTIFICATION CONFIDENCE:\n")
        if best_score < 2.0:
            text_widget.insert(tk.END, "   ✅ EXCELLENT match - High confidence identification\n")
            text_widget.insert(tk.END, "   Recommendation: Proceed with identification\n\n")
        elif best_score < 5.0:
            text_widget.insert(tk.END, "   🟡 GOOD match - Moderate confidence\n")
            text_widget.insert(tk.END, "   Recommendation: Consider additional testing\n\n")
        else:
            text_widget.insert(tk.END, "   ⚠️ POOR match - Low confidence\n")
            text_widget.insert(tk.END, "   Recommendation: Review sample preparation and re-analyze\n\n")
        
        text_widget.insert(tk.END, "🔬 SUGGESTED NEXT STEPS:\n")
        text_widget.insert(tk.END, "1. Review top 3 candidates for gemological consistency\n")
        text_widget.insert(tk.END, "2. Consider structural feature analysis for disambiguation\n")
        text_widget.insert(tk.END, "3. Verify geographic origin if applicable\n")
        text_widget.insert(tk.END, "4. Check treatment indicators in spectral data\n\n")
        
        text_widget.insert(tk.END, "📊 ANALYSIS QUALITY:\n")
        available_count = len(results['available_sources'])
        if available_count == 3:
            text_widget.insert(tk.END, "   ✅ Complete multi-spectral analysis (B+L+U)\n")
        elif available_count == 2:
            text_widget.insert(tk.END, "   🟡 Partial analysis - missing one light source\n")
        else:
            text_widget.insert(tk.END, "   ⚠️ Limited analysis - only one light source\n")
        
        text_widget.insert(tk.END, f"   Database coverage: {len(results['final_ranking'])} reference gems\n")
    
    def export_results(self, results):
        """Export results to CSV file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gem_analysis_results_{timestamp}.csv"
            
            # Prepare data for export
            export_data = []
            for i, (gem_id, total_score) in enumerate(results['final_ranking'], 1):
                description = self.get_gem_description(gem_id)
                scores = results['gem_best_scores'].get(gem_id, {})
                
                row = {
                    'Rank': i,
                    'Gem_ID': gem_id,
                    'Description': description,
                    'Total_Score': total_score
                }
                
                # Add individual scores
                for ls in sorted(results['available_sources']):
                    row[f'{ls}_Score'] = scores.get(ls, 'N/A')
                
                export_data.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Results exported to {filename}")
            print(f"✅ Results exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")
            print(f"❌ Export error: {e}")
    
    def show_interactive_results(self, results):
        """Show interactive matplotlib plot with selectable matches"""
        if not results['final_ranking']:
            return
        
        # Create interactive score plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Score distribution
        scores = [score for _, score in results['final_ranking']]
        ranks = list(range(1, len(scores) + 1))
        
        colors = ['red' if i < 3 else 'orange' if i < 10 else 'blue' for i in ranks]
        bars = ax1.bar(ranks[:20], scores[:20], color=colors, alpha=0.7)
        
        ax1.set_xlabel('Rank')
        ax1.set_ylabel('Log Score (lower is better)')
        ax1.set_title('Top 20 Gem Matches by Score')
        ax1.grid(True, alpha=0.3)
        
        # Add hover functionality
        def on_hover(event):
            if event.inaxes == ax1:
                for i, bar in enumerate(bars):
                    if bar.contains(event)[0]:
                        gem_id = results['final_ranking'][i][0]
                        description = self.get_gem_description(gem_id)
                        score = results['final_ranking'][i][1]
                        ax1.set_title(f'Rank {i+1}: {description} (Score: {score:.3f})')
                        fig.canvas.draw_idle()
                        break
        
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        
        # Plot 2: Light source score comparison for top 5
        top_5 = results['final_ranking'][:5]
        light_sources = sorted(results['available_sources'])
        
        x = np.arange(len(top_5))
        width = 0.8 / len(light_sources)
        
        for i, ls in enumerate(light_sources):
            scores = [results['gem_best_scores'][gem_id].get(ls, 0) for gem_id, _ in top_5]
            ax2.bar(x + i * width, scores, width, label=f'{ls} Light', alpha=0.7)
        
        ax2.set_xlabel('Top 5 Matches')
        ax2.set_ylabel('Individual Light Source Scores')
        ax2.set_title('Score Breakdown by Light Source')
        ax2.set_xticks(x + width * (len(light_sources) - 1) / 2)
        ax2.set_xticklabels([f"#{i+1}" for i in range(len(top_5))])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the visualizer"""
    try:
        visualizer = GemResultsVisualizer()
        
        print("🔬 GEM RESULTS VISUALIZER")
        print("=" * 40)
        print("1. Run Analysis and Visualize")
        print("2. Load Previous Results")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            visualizer.run_analysis_and_visualize()
        elif choice == '2':
            print("⚠️ Previous results loading not yet implemented")
        elif choice == '3':
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()