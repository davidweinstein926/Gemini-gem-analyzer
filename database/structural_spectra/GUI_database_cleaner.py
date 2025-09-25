#!/usr/bin/env python3
"""
GUI DATABASE CLEANER - UPDATED FOR NEW DATABASE STRUCTURE
Working GUI for selecting and deleting database records from the new unified structural database
Compatible with: database/structural_spectra/gemini_structural.db and gemini_structural_unified.csv
"""

import pandas as pd
import sqlite3
import re
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
from datetime import datetime
import shutil

class UpdatedDatabaseCleanerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Structural Database Record Cleaner (Updated)")
        self.root.geometry("1400x900")
        
        self.df = None
        self.database_type = None
        self.database_path = None
        self.conn = None
        self.table_name = None
        self.identifier_col = None
        self.selected_items = set()
        
        self.setup_gui()
        self.load_database()
    
    def find_database_files(self):
        """Find and select new database files"""
        current_path = Path.cwd()
        
        # NEW: Check for updated database files in priority order
        db_locations = [
            # New primary location
            current_path / "database" / "structural_spectra",
            # Alternative locations
            current_path,
            current_path / "database",
            current_path.parent / "database" / "structural_spectra"
        ]
        
        available_dbs = []
        
        for db_path in db_locations:
            if not db_path.exists():
                continue
                
            # NEW: Look for new database files first
            new_csv = db_path / "gemini_structural_unified.csv"
            new_sqlite = db_path / "gemini_structural.db"
            
            # Legacy files as backup
            legacy_csv = db_path / "gemini_structural_db.csv"  
            legacy_sqlite = db_path / "multi_structural_gem_data.db"
            
            # Priority order: new files first, then legacy
            if new_sqlite.exists():
                available_dbs.append(("sqlite", new_sqlite, "NEW"))
            if new_csv.exists():
                available_dbs.append(("csv", new_csv, "NEW"))
            if legacy_sqlite.exists():
                available_dbs.append(("sqlite", legacy_sqlite, "LEGACY"))
            if legacy_csv.exists():
                available_dbs.append(("csv", legacy_csv, "LEGACY"))
            
            # Stop at first location with databases
            if available_dbs:
                break
        
        if not available_dbs:
            return None, None, None
        
        # Show user the options if multiple databases found
        if len(available_dbs) > 1:
            choice_text = "Multiple databases found:\n\n"
            for i, (db_type, db_file, version) in enumerate(available_dbs, 1):
                choice_text += f"{i}. {db_type.upper()} - {db_file.name} ({version})\n"
            choice_text += "\nEnter number (1-{}):".format(len(available_dbs))
            
            choice = simpledialog.askstring("Database Selection", choice_text)
            try:
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_dbs):
                    db_type, db_file, version = available_dbs[choice_idx]
                    return db_file.parent, db_type, version
            except (ValueError, TypeError):
                pass
            return None, None, None
        
        # Return the single available database
        db_type, db_file, version = available_dbs[0]
        return db_file.parent, db_type, version
    
    def extract_gem_info(self, filename):
        """Extract gem name, date, and time from filename - Enhanced for new format"""
        if pd.isna(filename):
            return "Unknown", "no_date", "000000"
            
        identifier = str(filename)
        
        # Extract gem name - handle both old and new formats
        gem_pattern = r'^(.+?)([BLU])([CP])(\d+)'
        gem_match = re.match(gem_pattern, identifier, re.IGNORECASE)
        
        if gem_match:
            prefix, light, orientation, scan = gem_match.groups()
            gem_name = f"{prefix}{light.upper()}{orientation.upper()}{scan}"
        else:
            # NEW: Handle archived files with timestamps
            archived_pattern = r'^(.+?)_archived_\d{8}_\d{6}'
            archived_match = re.match(archived_pattern, identifier)
            if archived_match:
                gem_name = archived_match.group(1)
            else:
                gem_name = identifier.split('_')[0].split('.')[0]
        
        # Extract date and time - enhanced patterns
        date_patterns = [
            r'(\d{8})_(\d{6})',  # Standard format
            r'archived_(\d{8})_(\d{6})',  # Archived format
            r'(\d{8})(?:_(\d{6}))?'  # Fallback
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, identifier)
            if date_match:
                date_str = date_match.group(1)
                time_str = date_match.group(2) if date_match.group(2) else "000000"
                return gem_name, date_str, time_str
        
        return gem_name, "no_date", "000000"
    
    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title - Updated
        title_label = ttk.Label(main_frame, text="Structural Database Record Cleaner (Updated)", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        info_frame.columnconfigure(1, weight=1)
        
        self.info_label = ttk.Label(info_frame, text="Loading database...", 
                                   font=("Arial", 10))
        self.info_label.grid(row=0, column=0, sticky=tk.W)
        
        self.count_label = ttk.Label(info_frame, text="", 
                                    font=("Arial", 10, "bold"), foreground="blue")
        self.count_label.grid(row=0, column=1, sticky=tk.E)
        
        # Treeview frame
        tree_frame = ttk.Frame(main_frame)
        tree_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Create treeview
        self.tree = ttk.Treeview(tree_frame, show="tree headings", height=20)
        
        # Define columns
        self.tree["columns"] = ("DisplayIndex", "OriginalIndex", "Filename", "Gem", "Date", "Time", "LightSource")
        
        # Configure tree column (for checkboxes)
        self.tree.column("#0", width=60, minwidth=60, anchor=tk.CENTER)
        self.tree.heading("#0", text="Select")
        
        # Configure data columns
        self.tree.column("DisplayIndex", width=60, minwidth=60, anchor=tk.CENTER)
        self.tree.column("OriginalIndex", width=0, minwidth=0)  # Hidden column for real index
        self.tree.column("Filename", width=350, minwidth=300, anchor=tk.W)
        self.tree.column("Gem", width=100, minwidth=80, anchor=tk.CENTER)
        self.tree.column("Date", width=100, minwidth=80, anchor=tk.CENTER)
        self.tree.column("Time", width=80, minwidth=60, anchor=tk.CENTER)
        self.tree.column("LightSource", width=80, minwidth=60, anchor=tk.CENTER)  # NEW
        
        self.tree.heading("DisplayIndex", text="Index")
        self.tree.heading("OriginalIndex", text="")  # Hidden
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("Gem", text="Gem Name")
        self.tree.heading("Date", text="Date")
        self.tree.heading("Time", text="Time")
        self.tree.heading("LightSource", text="Light")  # NEW
        
        # Hide the OriginalIndex column
        self.tree.column("OriginalIndex", width=0, stretch=False)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid treeview and scrollbars
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Bind events
        self.tree.bind("<Button-1>", self.on_tree_click)
        self.tree.bind("<space>", self.on_space_key)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, pady=(10, 0), sticky=tk.W)
        
        # Selection buttons
        ttk.Button(button_frame, text="Select All", 
                  command=self.select_all).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(button_frame, text="Deselect All", 
                  command=self.deselect_all).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Invert Selection", 
                  command=self.invert_selection).grid(row=0, column=2, padx=5)
        
        # NEW: Filter buttons
        ttk.Button(button_frame, text="Select Halogen", 
                  command=lambda: self.select_by_light_source("Halogen")).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Select Laser", 
                  command=lambda: self.select_by_light_source("Laser")).grid(row=0, column=4, padx=5)
        ttk.Button(button_frame, text="Select UV", 
                  command=lambda: self.select_by_light_source("UV")).grid(row=0, column=5, padx=5)
        
        # Separator
        ttk.Separator(button_frame, orient=tk.VERTICAL).grid(row=0, column=6, padx=15, sticky=(tk.N, tk.S))
        
        # Action buttons
        self.delete_button = ttk.Button(button_frame, text="DELETE SELECTED", 
                                       command=self.delete_selected)
        self.delete_button.grid(row=0, column=7, padx=10)
        self.delete_button.configure(state="disabled")
        
        ttk.Button(button_frame, text="Refresh", 
                  command=self.load_database).grid(row=0, column=8, padx=(10, 0))
        
        # Status labels
        self.selected_label = ttk.Label(main_frame, text="Selected: 0 records", 
                                       font=("Arial", 11, "bold"), foreground="red")
        self.selected_label.grid(row=4, column=0, pady=(10, 5), sticky=tk.W)
        
        self.status_label = ttk.Label(main_frame, text="Ready", 
                                     font=("Arial", 9), foreground="green")
        self.status_label.grid(row=5, column=0, sticky=tk.W)
    
    def load_database(self):
        """Load database data - Updated for new structure"""
        try:
            self.status_label.config(text="Loading database...")
            self.root.update()
            
            # Clear any cached data
            self.df = None
            if self.conn:
                self.conn.close()
                self.conn = None
            
            # Find database files
            db_path, db_type, version = self.find_database_files()
            
            if not db_path or not db_type:
                messagebox.showerror("Error", 
                    "No database files found!\n\nExpected files (NEW):\n"
                    "- gemini_structural.db\n"
                    "- gemini_structural_unified.csv\n\n"
                    "Legacy files:\n"
                    "- multi_structural_gem_data.db\n"
                    "- gemini_structural_db.csv")
                self.root.quit()
                return
            
            self.database_path = db_path
            self.database_type = db_type
            
            # Load based on type
            if db_type == "csv":
                self.load_csv_database(version)
            else:
                self.load_sqlite_database(version)
                
            self.populate_tree()
            self.update_counts()
            self.status_label.config(text=f"Loaded {len(self.df)} records from {db_type.upper()} database ({version})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database:\n{str(e)}")
            self.status_label.config(text="Error loading database")
            print(f"DEBUG: Load error details: {e}")
            import traceback
            traceback.print_exc()
    
    def load_csv_database(self, version):
        """Load CSV database - Updated for new format"""
        if version == "NEW":
            csv_file = self.database_path / "gemini_structural_unified.csv"
        else:
            csv_file = self.database_path / "gemini_structural_db.csv"
            
        self.df = pd.read_csv(csv_file)
        self.df = self.df.reset_index(drop=True)
        
        # NEW: Updated identifier column detection for new schema
        identifier_columns = ['file_source', 'gem_id', 'file', 'filename', 'full_name', 'gem_name', 'identifier']
        self.identifier_col = None
        
        for col in identifier_columns:
            if col in self.df.columns:
                self.identifier_col = col
                break
        
        if not self.identifier_col and len(self.df.columns) > 0:
            # Use first string column
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    self.identifier_col = col
                    break
        
        if not self.identifier_col:
            raise Exception("Could not find identifier column in CSV")
        
        self.info_label.config(text=f"CSV Database ({version}): {csv_file.name} | Column: {self.identifier_col}")
    
    def load_sqlite_database(self, version):
        """Load SQLite database - Updated for new structure"""
        if version == "NEW":
            sqlite_file = self.database_path / "gemini_structural.db"
        else:
            sqlite_file = self.database_path / "multi_structural_gem_data.db"
            
        self.conn = sqlite3.connect(sqlite_file)
        cursor = self.conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            raise Exception("No tables found in SQLite database")
        
        # NEW: Handle new table name
        table_names = [t[0] for t in tables]
        if version == "NEW" and "structural_features" in table_names:
            self.table_name = "structural_features"
        else:
            self.table_name = tables[0][0]  # Use first available table
        
        # Get all records with rowid
        cursor.execute(f"SELECT rowid, * FROM {self.table_name}")
        records = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        columns_info = cursor.fetchall()
        columns = ['rowid'] + [col[1] for col in columns_info]
        
        self.df = pd.DataFrame(records, columns=columns)
        self.df = self.df.reset_index(drop=True)
        
        # NEW: Updated identifier column detection for new schema
        identifier_columns = ['file_source', 'gem_id', 'file', 'filename', 'full_name', 'gem_name', 'identifier']
        self.identifier_col = None
        
        for col in identifier_columns:
            if col in self.df.columns:
                self.identifier_col = col
                break
        
        if not self.identifier_col:
            raise Exception("Could not find identifier column in SQLite")
        
        self.info_label.config(text=f"SQLite Database ({version}): {sqlite_file.name} | Table: {self.table_name} | Column: {self.identifier_col}")
    
    def populate_tree(self):
        """Populate the treeview with data - Enhanced for new structure"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.selected_items.clear()
        
        # Add records
        for display_idx, (df_idx, row) in enumerate(self.df.iterrows()):
            filename = row[self.identifier_col]
            gem_name, date_str, time_str = self.extract_gem_info(filename)
            
            # NEW: Extract light source info if available
            light_source = ""
            if 'light_source' in row:
                light_source = str(row['light_source'])
            elif 'Light_Source' in row:
                light_source = str(row['Light_Source'])
            else:
                # Infer from filename
                if 'halogen' in filename.lower() or any(x in filename.upper() for x in ['BC', '_B_']):
                    light_source = "Halogen"
                elif 'laser' in filename.lower() or any(x in filename.upper() for x in ['LC', '_L_']):
                    light_source = "Laser"
                elif 'uv' in filename.lower() or any(x in filename.upper() for x in ['UC', 'UP', '_U_']):
                    light_source = "UV"
            
            # Format time for display
            time_display = ""
            if time_str and time_str != "000000":
                time_display = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            
            # Insert row
            item_id = self.tree.insert("", "end", 
                                     text="☐",  # Checkbox in tree column
                                     values=(display_idx + 1, df_idx, filename, gem_name, date_str, time_display, light_source))
    
    def select_by_light_source(self, light_source):
        """NEW: Select all records with specific light source"""
        for item in self.tree.get_children():
            values = self.tree.item(item)['values']
            item_light_source = values[6] if len(values) > 6 else ""
            
            if item_light_source == light_source and item not in self.selected_items:
                self.selected_items.add(item)
                self.tree.item(item, text="☑")
                self.tree.item(item, tags=("selected",))
        
        self.tree.tag_configure("selected", background="#e6f3ff")
        self.update_counts()
    
    def on_tree_click(self, event):
        """Handle tree click events"""
        item = self.tree.identify_row(event.y)
        if item:
            # Check if clicked on the tree column (checkbox area)
            region = self.tree.identify_region(event.x, event.y)
            if region == "tree" or event.x < 60:  # Checkbox area
                self.toggle_selection(item)
    
    def on_space_key(self, event):
        """Handle space key for toggle selection"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            self.toggle_selection(item)
    
    def toggle_selection(self, item):
        """Toggle selection state of an item"""
        if item in self.selected_items:
            # Deselect
            self.selected_items.remove(item)
            self.tree.item(item, text="☐")
            self.tree.item(item, tags=())
        else:
            # Select
            self.selected_items.add(item)
            self.tree.item(item, text="☑")
            self.tree.item(item, tags=("selected",))
        
        # Configure selection highlighting
        self.tree.tag_configure("selected", background="#e6f3ff")
        self.update_counts()
    
    def select_all(self):
        """Select all items"""
        for item in self.tree.get_children():
            if item not in self.selected_items:
                self.selected_items.add(item)
                self.tree.item(item, text="☑")
                self.tree.item(item, tags=("selected",))
        
        self.tree.tag_configure("selected", background="#e6f3ff")
        self.update_counts()
    
    def deselect_all(self):
        """Deselect all items"""
        for item in self.tree.get_children():
            if item in self.selected_items:
                self.selected_items.remove(item)
                self.tree.item(item, text="☐")
                self.tree.item(item, tags=())
        
        self.update_counts()
    
    def invert_selection(self):
        """Invert current selection"""
        for item in self.tree.get_children():
            if item in self.selected_items:
                self.selected_items.remove(item)
                self.tree.item(item, text="☐")
                self.tree.item(item, tags=())
            else:
                self.selected_items.add(item)
                self.tree.item(item, text="☑")
                self.tree.item(item, tags=("selected",))
        
        self.tree.tag_configure("selected", background="#e6f3ff")
        self.update_counts()
    
    def update_counts(self):
        """Update count displays"""
        total = len(self.df) if self.df is not None else 0
        selected = len(self.selected_items)
        remaining = total - selected
        
        self.count_label.config(text=f"Total: {total} records")
        self.selected_label.config(text=f"Selected: {selected} records | Will keep: {remaining} records")
        
        # Update delete button state
        if selected > 0:
            self.delete_button.config(state="normal")
        else:
            self.delete_button.config(state="disabled")
    
    def delete_selected(self):
        """Delete selected records - Updated for new structure"""
        if not self.selected_items:
            messagebox.showwarning("No Selection", "No records selected for deletion.")
            return
        
        selected_count = len(self.selected_items)
        remaining_count = len(self.df) - selected_count
        
        # Confirmation dialog
        response = messagebox.askyesno(
            "Confirm Deletion",
            f"Delete {selected_count} selected records?\n\n"
            f"Records to delete: {selected_count}\n"
            f"Records to keep: {remaining_count}\n\n"
            f"This action cannot be undone!\n"
            f"(A backup will be created automatically)"
        )
        
        if not response:
            return
        
        try:
            self.status_label.config(text="Creating backup...")
            self.root.update()
            
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.database_type == "csv":
                # Determine current CSV filename
                if (self.database_path / "gemini_structural_unified.csv").exists():
                    csv_file = self.database_path / "gemini_structural_unified.csv"
                    backup_file = self.database_path / f"gemini_structural_unified_backup_{timestamp}.csv"
                else:
                    csv_file = self.database_path / "gemini_structural_db.csv"
                    backup_file = self.database_path / f"gemini_structural_db_backup_{timestamp}.csv"
                
                shutil.copy2(csv_file, backup_file)
                backup_name = backup_file.name
            else:
                # Determine current SQLite filename
                if (self.database_path / "gemini_structural.db").exists():
                    sqlite_file = self.database_path / "gemini_structural.db"
                    backup_file = self.database_path / f"gemini_structural_backup_{timestamp}.db"
                else:
                    sqlite_file = self.database_path / "multi_structural_gem_data.db"
                    backup_file = self.database_path / f"multi_structural_gem_data_backup_{timestamp}.db"
                
                shutil.copy2(sqlite_file, backup_file)
                backup_name = backup_file.name
            
            self.status_label.config(text="Deleting records...")
            self.root.update()
            
            # Get dataframe indices to delete
            df_indices_to_delete = []
            for item in self.selected_items:
                values = self.tree.item(item)['values']
                df_index = int(values[1])  # OriginalIndex
                df_indices_to_delete.append(df_index)
            
            print(f"DEBUG: Deleting dataframe indices: {sorted(df_indices_to_delete)}")
            
            # Delete records
            if self.database_type == "csv":
                self.delete_csv_records(df_indices_to_delete)
            else:
                self.delete_sqlite_records(df_indices_to_delete)
            
            # Show success message
            messagebox.showinfo(
                "Deletion Complete",
                f"Successfully deleted {selected_count} records!\n\n"
                f"Backup created: {backup_name}\n"
                f"Remaining records: {remaining_count}"
            )
            
            # Reload database
            self.load_database()
            
        except Exception as e:
            messagebox.showerror("Deletion Error", f"Failed to delete records:\n{str(e)}")
            self.status_label.config(text="Deletion failed")
            print(f"DEBUG: Error details: {e}")
            import traceback
            traceback.print_exc()
    
    def delete_csv_records(self, df_indices):
        """Delete records from CSV database"""
        # Determine current CSV file
        if (self.database_path / "gemini_structural_unified.csv").exists():
            csv_file = self.database_path / "gemini_structural_unified.csv"
        else:
            csv_file = self.database_path / "gemini_structural_db.csv"
        
        print(f"DEBUG CSV: Original dataframe shape: {self.df.shape}")
        print(f"DEBUG CSV: Indices to delete: {sorted(df_indices)}")
        
        # Create boolean mask for rows to keep
        mask = ~self.df.index.isin(df_indices)
        df_cleaned = self.df[mask].copy()
        
        print(f"DEBUG CSV: Cleaned dataframe shape: {df_cleaned.shape}")
        
        # Reset index to ensure clean sequential indices
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        # Save cleaned CSV
        df_cleaned.to_csv(csv_file, index=False)
        
        print(f"DEBUG CSV: Saved cleaned CSV with {len(df_cleaned)} records")
        
        # Clear cached dataframe to force reload
        self.df = None
    
    def delete_sqlite_records(self, df_indices):
        """Delete records from SQLite database"""
        if not self.conn:
            return
        
        cursor = self.conn.cursor()
        
        print(f"DEBUG SQLite: Original dataframe shape: {self.df.shape}")
        print(f"DEBUG SQLite: Indices to delete: {sorted(df_indices)}")
        
        # Get rowids to delete using the dataframe indices
        rowids_to_delete = []
        for df_index in df_indices:
            if df_index < len(self.df):
                rowid = self.df.iloc[df_index]['rowid']
                rowid = int(rowid)  # Convert to Python int
                rowids_to_delete.append(rowid)
                print(f"DEBUG SQLite: df_index {df_index} -> rowid {rowid}")
        
        print(f"DEBUG SQLite: Rowids to delete: {sorted(rowids_to_delete)}")
        
        # Delete in batches
        batch_size = 100
        deleted_count = 0
        for i in range(0, len(rowids_to_delete), batch_size):
            batch = rowids_to_delete[i:i + batch_size]
            placeholders = ','.join(['?' for _ in batch])
            delete_query = f"DELETE FROM {self.table_name} WHERE rowid IN ({placeholders})"
            result = cursor.execute(delete_query, batch)
            deleted_count += result.rowcount
            print(f"DEBUG SQLite: Batch {i//batch_size + 1}, deleted {result.rowcount} records")
        
        print(f"DEBUG SQLite: Total deleted: {deleted_count} records")
        
        # Commit and close/reopen connection
        self.conn.commit()
        self.conn.close()
        self.conn = None
        
        # Clear cached dataframe to force reload
        self.df = None
    
    def run(self):
        """Run the GUI"""
        try:
            self.root.mainloop()
        finally:
            if self.conn:
                self.conn.close()

def main():
    """Main entry point"""
    try:
        app = UpdatedDatabaseCleanerGUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()