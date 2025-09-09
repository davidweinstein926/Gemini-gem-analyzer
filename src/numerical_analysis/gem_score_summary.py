#!/usr/bin/env python3
"""
gem_score_summary.py - GEMINI SCORE SUMMARY DISPLAY
Standalone score summary component for gemstone analysis results
Save as: gemini_gemological_analysis/gem_score_summary.py
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class GemScoreSummary:
    def __init__(self, master=None):
        self.master = master if master else tk.Tk()
        self.gem_descriptions = {}
        self.results_data = None
        self.setup_ui()
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
                    # Create rich descriptions
                    def create_description(row):
                        parts = []
                        for col in expected_columns:
                            val = row[col] if pd.notnull(row[col]) and str(row[col]).strip() else ''
                            if val:
                                parts.append(str(val).strip())
                        return ' | '.join(parts) if parts else f"Gem {row['Reference']}"
                    
                    gemlib['Rich_Description'] = gemlib.apply(create_description, axis=1)
                    self.gem_descriptions = dict(zip(gemlib['Reference'], gemlib['Rich_Description']))
                    
                    # Also store individual fields for detailed display
                    self.gem_details = {}
                    for _, row in gemlib.iterrows():
                        ref = str(row['Reference'])
                        self.gem_details[ref] = {
                            'natural_synthetic': str(row.get('Nat./Syn.', '')).strip(),
                            'species': str(row.get('Spec.', '')).strip(),
                            'variety': str(row.get('Var.', '')).strip(),
                            'treatment': str(row.get('Treatment', '')).strip(),
                            'origin': str(row.get('Origin', '')).strip()
                        }
                    
                    print(f"✅ Loaded {len(self.gem_descriptions)} gem descriptions")
                else:
                    print(f"⚠️ Missing columns in gemlib: {[c for c in expected_columns if c not in gemlib.columns]}")
                    self.setup_fallback_descriptions()
            else:
                print("⚠️ 'Reference' column not found in gemlib_structural_ready.csv")
                self.setup_fallback_descriptions()
                
        except FileNotFoundError:
            print("⚠️ gemlib_structural_ready.csv not found - using generic descriptions")
            self.setup_fallback_descriptions()
        except Exception as e:
            print(f"⚠️ Error loading gemlib: {e}")
            self.setup_fallback_descriptions()
    
    def setup_fallback_descriptions(self):
        """Setup fallback descriptions when gemlib is not available"""
        self.gem_descriptions = {}
        self.gem_details = {}
    
    def get_gem_description(self, gem_id):
        """Get descriptive name for gem ID"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        return self.gem_descriptions.get(base_id, f"Gem {base_id}")
    
    def get_gem_details(self, gem_id):
        """Get detailed gem information"""
        base_id = str(gem_id).split('B')[0].split('L')[0].split('U')[0]
        return self.gem_details.get(base_id, {
            'natural_synthetic': 'Unknown',
            'species': 'Unknown',
            'variety': 'Unknown', 
            'treatment': 'Unknown',
            'origin': 'Unknown'
        })
    
    def setup_ui(self):
        """Setup the user interface"""
        self.master.title("🏆 Gemstone Analysis - Score Summary")
        self.master.geometry("1200x800")
        self.master.configure(bg='#f8f9fa')
        
        # Configure grid weights
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        
        # Main container
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Header
        self.create_header()
        
        # Content area
        self.create_content_area()
        
        # Footer with controls
        self.create_footer()
    
    def create_header(self):
        """Create header section"""
        header_frame = tk.Frame(self.main_frame, bg='#2c3e50', height=80)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header_frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(header_frame, text="🔬 GEMSTONE IDENTIFICATION RESULTS", 
                              font=('Arial', 18, 'bold'), bg='#2c3e50', fg='white')
        title_label.pack(pady=20)
        
        # Status bar
        self.status_frame = tk.Frame(self.main_frame, bg='#ecf0f1', height=30)
        self.status_frame.grid(row=0, column=0, sticky="ew", pady=(60, 0))
        self.status_frame.grid_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="Ready to load analysis results", 
                                   font=('Arial', 10), bg='#ecf0f1', fg='#7f8c8d')
        self.status_label.pack(pady=5)
    
    def create_content_area(self):
        """Create main content area with tabs"""
        # Notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Tab 1: Summary Table
        self.create_summary_tab()
        
        # Tab 2: Detailed View
        self.create_detailed_tab()
        
        # Tab 3: Statistics
        self.create_statistics_tab()
    
    def create_summary_tab(self):
        """Create summary results tab"""
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="📊 Summary Results")
        
        # Configure grid
        summary_frame.grid_rowconfigure(0, weight=1)
        summary_frame.grid_columnconfigure(0, weight=1)
        
        # Create treeview with scrollbars
        tree_frame = tk.Frame(summary_frame)
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Define columns
        columns = ('Rank', 'Gem_ID', 'Description', 'Total_Score', 'Confidence', 'B_Score', 'L_Score', 'U_Score')
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=20)
        
        # Configure column headings and widths
        column_config = {
            'Rank': ('Rank', 50),
            'Gem_ID': ('Gem ID', 70),
            'Description': ('Gem Description', 300),
            'Total_Score': ('Total Score', 100),
            'Confidence': ('Confidence', 90),
            'B_Score': ('B Score', 80),
            'L_Score': ('L Score', 80),
            'U_Score': ('U Score', 80)
        }
        
        for col, (heading, width) in column_config.items():
            self.results_tree.heading(col, text=heading)
            self.results_tree.column(col, width=width, anchor='center' if col != 'Description' else 'w')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.results_tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure row colors
        self.results_tree.tag_configure('rank1', background='#d5f4e6')  # Green for #1
        self.results_tree.tag_configure('rank2', background='#ffeaa7')  # Yellow for #2
        self.results_tree.tag_configure('rank3', background='#fab1a0')  # Orange for #3
        self.results_tree.tag_configure('top10', background='#f8f9fa')  # Light gray for top 10
        self.results_tree.tag_configure('normal', background='white')   # White for others
        
        # Bind double-click to show details
        self.results_tree.bind('<Double-1>', self.show_gem_details)
    
    def create_detailed_tab(self):
        """Create detailed view tab"""
        detail_frame = ttk.Frame(self.notebook)
        self.notebook.add(detail_frame, text="🔍 Detailed View")
        
        # Split into selection and detail areas
        detail_frame.grid_rowconfigure(1, weight=1)
        detail_frame.grid_columnconfigure(0, weight=1)
        
        # Selection area
        selection_frame = tk.LabelFrame(detail_frame, text="Select Gem for Details", 
                                       font=('Arial', 10, 'bold'), padx=10, pady=5)
        selection_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
        
        self.detail_var = tk.StringVar()
        self.detail_combo = ttk.Combobox(selection_frame, textvariable=self.detail_var, 
                                        state="readonly", width=80)
        self.detail_combo.grid(row=0, column=0, padx=5, pady=5)
        self.detail_combo.bind('<<ComboboxSelected>>', self.update_detailed_view)
        
        # Detail display area
        self.detail_text = tk.Text(detail_frame, wrap=tk.WORD, font=('Consolas', 10),
                                  bg='#f8f9fa', relief='sunken', bd=2)
        self.detail_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Scrollbar for detail text
        detail_scroll = ttk.Scrollbar(detail_frame, orient="vertical", command=self.detail_text.yview)
        detail_scroll.grid(row=1, column=1, sticky="ns")
        self.detail_text.configure(yscrollcommand=detail_scroll.set)
    
    def create_statistics_tab(self):
        """Create statistics tab"""
        stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(stats_frame, text="📈 Statistics")
        
        # Configure grid
        stats_frame.grid_rowconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(0, weight=1)
        
        # Statistics text area
        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, font=('Consolas', 10),
                                 bg='#f8f9fa', relief='sunken', bd=2)
        self.stats_text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Scrollbar for stats
        stats_scroll = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        stats_scroll.grid(row=0, column=1, sticky="ns")
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
    
    def create_footer(self):
        """Create footer with control buttons"""
        footer_frame = tk.Frame(self.main_frame, bg='#ecf0f1', height=60)
        footer_frame.grid(row=2, column=0, sticky="ew")
        footer_frame.grid_propagate(False)
        
        # Button frame
        button_frame = tk.Frame(footer_frame, bg='#ecf0f1')
        button_frame.pack(expand=True)
        
        # Control buttons
        buttons = [
            ("📂 Load Results", self.load_results, '#3498db'),
            ("💾 Save Report", self.save_report, '#2ecc71'),
            ("📊 Export CSV", self.export_csv, '#e67e22'),
            ("🔄 Refresh", self.refresh_display, '#9b59b6'),
            ("❌ Close", self.close_window, '#e74c3c')
        ]
        
        for i, (text, command, color) in enumerate(buttons):
            btn = tk.Button(button_frame, text=text, command=command,
                           bg=color, fg='white', font=('Arial', 9, 'bold'),
                           relief='raised', bd=2, padx=15, pady=5)
            btn.pack(side=tk.LEFT, padx=5, pady=15)
    
    def load_results(self, results_data=None):
        """Load analysis results"""
        if results_data:
            self.results_data = results_data
        else:
            # Load from file (placeholder for file dialog)
            messagebox.showinfo("Load Results", "File loading dialog would appear here.\nFor now, use direct data loading.")
            return
        
        self.populate_summary_table()
        self.populate_detailed_combo()
        self.generate_statistics()
        self.update_status(f"Loaded {len(self.results_data.get('final_ranking', []))} results")
    
    def populate_summary_table(self):
        """Populate the summary results table"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if not self.results_data or 'final_ranking' not in self.results_data:
            return
        
        # Calculate confidence scores
        scores = [score for _, score in self.results_data['final_ranking']]
        if len(scores) > 1:
            confidence_factors = self.calculate_confidence_scores(scores)
        else:
            confidence_factors = ['N/A']
        
        # Populate table
        for i, (gem_id, total_score) in enumerate(self.results_data['final_ranking'], 1):
            description = self.get_gem_description(gem_id)
            
            # Get individual scores
            gem_scores = self.results_data.get('gem_best_scores', {}).get(gem_id, {})
            b_score = f"{gem_scores.get('B', 0):.2f}" if 'B' in gem_scores else 'N/A'
            l_score = f"{gem_scores.get('L', 0):.2f}" if 'L' in gem_scores else 'N/A'
            u_score = f"{gem_scores.get('U', 0):.2f}" if 'U' in gem_scores else 'N/A'
            
            confidence = confidence_factors[i-1] if i-1 < len(confidence_factors) else 'N/A'
            
            # Determine row tag based on rank
            if i == 1:
                tag = 'rank1'
            elif i == 2:
                tag = 'rank2'
            elif i == 3:
                tag = 'rank3'
            elif i <= 10:
                tag = 'top10'
            else:
                tag = 'normal'
            
            values = (i, gem_id, description, f"{total_score:.3f}", confidence, b_score, l_score, u_score)
            self.results_tree.insert('', 'end', values=values, tags=(tag,))
    
    def calculate_confidence_scores(self, scores):
        """Calculate confidence scores based on score distribution"""
        confidence_scores = []
        
        for i, score in enumerate(scores):
            if i == 0:
                # First place - compare with second place
                if len(scores) > 1:
                    gap = scores[1] - score
                    confidence_pct = min(100, (gap / score) * 100)
                    confidence_scores.append(f"{confidence_pct:.1f}%")
                else:
                    confidence_scores.append("100%")
            else:
                # Other places - compare with first place
                gap = score - scores[0]
                confidence_pct = max(0, 100 - (gap / scores[0]) * 100)
                confidence_scores.append(f"{confidence_pct:.1f}%")
        
        return confidence_scores
    
    def populate_detailed_combo(self):
        """Populate the detailed view combo box"""
        if not self.results_data or 'final_ranking' not in self.results_data:
            return
        
        options = []
        for i, (gem_id, score) in enumerate(self.results_data['final_ranking'][:20], 1):  # Top 20
            description = self.get_gem_description(gem_id)
            options.append(f"#{i} - {description} (ID: {gem_id}, Score: {score:.3f})")
        
        self.detail_combo['values'] = options
        if options:
            self.detail_combo.set(options[0])
            self.update_detailed_view()
    
    def update_detailed_view(self, event=None):
        """Update the detailed view for selected gem"""
        selection = self.detail_var.get()
        if not selection or not self.results_data:
            return
        
        # Extract gem ID from selection
        try:
            gem_id = selection.split('ID: ')[1].split(',')[0]
        except:
            return
        
        # Get gem details
        details = self.get_gem_details(gem_id)
        gem_scores = self.results_data.get('gem_best_scores', {}).get(gem_id, {})
        
        # Clear and populate detail text
        self.detail_text.delete(1.0, tk.END)
        
        detail_info = f"""
🔍 DETAILED GEM ANALYSIS REPORT
{'='*60}

💎 GEM IDENTIFICATION:
   Gem ID: {gem_id}
   Full Description: {self.get_gem_description(gem_id)}

📋 GEMOLOGICAL PROPERTIES:
   Natural/Synthetic: {details.get('natural_synthetic', 'Unknown')}
   Species: {details.get('species', 'Unknown')}
   Variety: {details.get('variety', 'Unknown')}
   Treatment: {details.get('treatment', 'Unknown')}
   Origin: {details.get('origin', 'Unknown')}

📊 ANALYSIS SCORES:
"""
        
        # Add individual light source scores
        available_sources = self.results_data.get('available_sources', ['B', 'L', 'U'])
        for light in sorted(available_sources):
            score = gem_scores.get(light, 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            detail_info += f"   {light} Light Source: {score_str}\n"
        
        # Calculate total score
        total_score = sum(gem_scores.get(ls, 0) for ls in available_sources if ls in gem_scores)
        detail_info += f"   Total Combined Score: {total_score:.4f}\n"
        
        # Add match quality assessment
        detail_info += f"""
🎯 MATCH QUALITY ASSESSMENT:
   Score Interpretation: {'Excellent' if total_score < 2 else 'Good' if total_score < 5 else 'Fair' if total_score < 10 else 'Poor'}
   Confidence Level: {'High' if total_score < 3 else 'Medium' if total_score < 7 else 'Low'}
   
📝 NOTES:
   - Lower scores indicate better matches
   - Scores are logarithmic (log1p of MSE)
   - Multiple light sources increase confidence
   - Consider gemological properties for final identification
"""
        
        self.detail_text.insert(1.0, detail_info)
    
    def generate_statistics(self):
        """Generate and display analysis statistics"""
        if not self.results_data:
            return
        
        self.stats_text.delete(1.0, tk.END)
        
        scores = [score for _, score in self.results_data.get('final_ranking', [])]
        available_sources = self.results_data.get('available_sources', [])
        
        stats_info = f"""
📈 GEMSTONE ANALYSIS STATISTICS
{'='*60}

🔍 ANALYSIS OVERVIEW:
   Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
   Light Sources Used: {', '.join(sorted(available_sources))} ({len(available_sources)} sources)
   Total Candidates: {len(scores)}
   Database Coverage: Complete multi-spectral analysis

📊 SCORE DISTRIBUTION:
"""
        
        if scores:
            stats_info += f"""   Best Score: {min(scores):.4f}
   Worst Score: {max(scores):.4f}
   Mean Score: {np.mean(scores):.4f}
   Median Score: {np.median(scores):.4f}
   Standard Deviation: {np.std(scores):.4f}
   
🎯 QUALITY BREAKDOWN:
"""
            
            # Quality tiers
            excellent = sum(1 for s in scores if s < 2.0)
            good = sum(1 for s in scores if 2.0 <= s < 5.0)
            fair = sum(1 for s in scores if 5.0 <= s < 10.0)
            poor = sum(1 for s in scores if s >= 10.0)
            
            stats_info += f"""   Excellent Matches (< 2.0): {excellent} ({excellent/len(scores)*100:.1f}%)
   Good Matches (2.0-5.0): {good} ({good/len(scores)*100:.1f}%)
   Fair Matches (5.0-10.0): {fair} ({fair/len(scores)*100:.1f}%)
   Poor Matches (≥ 10.0): {poor} ({poor/len(scores)*100:.1f}%)

💡 RECOMMENDATIONS:
"""
            
            best_score = min(scores)
            if best_score < 1.0:
                stats_info += "   ✅ Excellent identification confidence - proceed with classification\n"
            elif best_score < 3.0:
                stats_info += "   🟡 Good identification confidence - verify with additional methods\n"
            else:
                stats_info += "   ⚠️ Low identification confidence - review sample preparation\n"
            
            stats_info += f"""
🔬 TECHNICAL DETAILS:
   Analysis Method: Multi-spectral comparison with logarithmic scoring
   Normalization: Fixed reference wavelength normalization
   Database: Comprehensive gemstone reference library
   Score Calculation: log1p(MSE) between normalized spectra
"""
        
        self.stats_text.insert(1.0, stats_info)
    
    def show_gem_details(self, event):
        """Show detailed information for double-clicked gem"""
        selection = self.results_tree.selection()
        if not selection:
            return
        
        item = self.results_tree.item(selection[0])
        gem_id = item['values'][1]  # Gem ID is in second column
        
        # Switch to detailed tab and select this gem
        self.notebook.select(1)  # Select detailed view tab
        
        # Find and select the gem in combo box
        for i, option in enumerate(self.detail_combo['values']):
            if f"ID: {gem_id}" in option:
                self.detail_combo.current(i)
                self.update_detailed_view()
                break
    
    def save_report(self):
        """Save comprehensive analysis report"""
        if not self.results_data:
            messagebox.showwarning("No Data", "No analysis results to save")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gem_analysis_report_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("GEMSTONE ANALYSIS COMPREHENSIVE REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary results
                f.write("SUMMARY RESULTS:\n")
                f.write("-" * 30 + "\n")
                for i, (gem_id, score) in enumerate(self.results_data['final_ranking'][:10], 1):
                    desc = self.get_gem_description(gem_id)
                    f.write(f"{i:2d}. {desc} (ID: {gem_id}) - Score: {score:.4f}\n")
                
                f.write("\n" + "=" * 60 + "\nEnd of Report\n")
            
            messagebox.showinfo("Report Saved", f"Report saved as {filename}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report: {e}")
    
    def export_csv(self):
        """Export results to CSV"""
        if not self.results_data:
            messagebox.showwarning("No Data", "No analysis results to export")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gem_analysis_export_{timestamp}.csv"
            
            # Prepare data
            export_data = []
            for i, (gem_id, total_score) in enumerate(self.results_data['final_ranking'], 1):
                description = self.get_gem_description(gem_id)
                details = self.get_gem_details(gem_id)
                scores = self.results_data.get('gem_best_scores', {}).get(gem_id, {})
                
                row = {
                    'Rank': i,
                    'Gem_ID': gem_id,
                    'Description': description,
                    'Total_Score': total_score,
                    'Natural_Synthetic': details.get('natural_synthetic', ''),
                    'Species': details.get('species', ''),
                    'Variety': details.get('variety', ''),
                    'Treatment': details.get('treatment', ''),
                    'Origin': details.get('origin', '')
                }
                
                # Add individual scores
                for ls in sorted(self.results_data.get('available_sources', [])):
                    row[f'{ls}_Score'] = scores.get(ls, 'N/A')
                
                export_data.append(row)
            
            df = pd.DataFrame(export_data)
            df.to_csv(filename, index=False)
            
            messagebox.showinfo("Export Complete", f"Data exported to {filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def refresh_display(self):
        """Refresh the display"""
        if self.results_data:
            self.load_results(self.results_data)
            self.update_status("Display refreshed")
        else:
            self.update_status("No data to refresh")
    
    def update_status(self, message):
        """Update status message"""
        self.status_label.config(text=message)
        self.master.after(3000, lambda: self.status_label.config(text="Ready"))
    
    def close_window(self):
        """Close the application"""
        self.master.quit()
        if self.master.master is None:  # Only destroy if we created the root
            self.master.destroy()

def main():
    """Main function for standalone usage"""
    root = tk.Tk()
    app = GemScoreSummary(root)
    
    # Example data for demonstration
    example_data = {
        'available_sources': ['B', 'L', 'U'],
        'final_ranking': [
            ('58', 2.45), ('142', 3.21), ('89', 4.67), ('234', 5.12), ('67', 6.34)
        ],
        'gem_best_scores': {
            '58': {'B': 0.85, 'L': 0.92, 'U': 0.68},
            '142': {'B': 1.12, 'L': 1.05, 'U': 1.04},
            '89': {'B': 1.67, 'L': 1.45, 'U': 1.55},
            '234': {'B': 1.89, 'L': 1.78, 'U': 1.45},
            '67': {'B': 2.12, 'L': 2.01, 'U': 2.21}
        }
    }
    
    # Load example data
    app.load_results(example_data)
    
    root.mainloop()

if __name__ == "__main__":
    main()