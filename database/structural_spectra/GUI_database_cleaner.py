#!/usr/bin/env python3
"""
GUI DATABASE CLEANER - FIXED DELETION VERSION
Working GUI for selecting and deleting database records with proper index handling
"""

import pandas as pd
import sqlite3
import re
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
from datetime import datetime
import shutil

class DatabaseCleanerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Database Record Cleaner")
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
        """Find and select database files"""
        current_path = Path.cwd()
        
        # Check current directory
        csv_file = current_path / "gemini_structural_db.csv"
        sqlite_file = current_path / "multi_structural_gem_data.db"
        
        available_dbs = []
        if csv_file.exists():
            available_dbs.append(("csv", csv_file))
        if sqlite_file.exists():
            available_dbs.append(("sqlite", sqlite_file))
        
        if not available_dbs:
            # Search in other locations
            possible_paths = [
                current_path / "database" / "structural_spectra",
                current_path / "root" / "database" / "structural_spectra", 
                current_path.parent / "database" / "structural_spectra"
            ]
            
            for db_path in possible_paths:
                if db_path.exists():
                    csv_file = db_path / "gemini_structural_db.csv"
                    sqlite_file = db_path / "multi_structural_gem_data.db"
                    if csv_file.exists():
                        available_dbs.append(("csv", csv_file))
                    if sqlite_file.exists():
                        available_dbs.append(("sqlite", sqlite_file))
                    break
        
        if not available_dbs:
            return None, None
            
        # If both exist, ask user to choose
        if len(available_dbs) > 1:
            choice = simpledialog.askstring(
                "Database Selection",
                "Multiple databases found:\n\n1 = CSV Database\n2 = SQLite Database\n\nEnter 1 or 2:"
            )
            if choice == "1":
                db_type, db_file = next(((t, f) for t, f in available_dbs if t == "csv"), (None, None))
                return db_file.parent if db_file else None, db_type
            elif choice == "2":
                db_type, db_file = next(((t, f) for t, f in available_dbs if t == "sqlite"), (None, None))
                return db_file.parent if db_file else None, db_type
            else:
                return None, None
        
        # Return the single available database
        db_type, db_file = available_dbs[0]
        return db_file.parent, db_type
    
    def extract_gem_info(self, filename):
        """Extract gem name, date, and time from filename"""
        if pd.isna(filename):
            return "Unknown", "no_date", "000000"
            
        identifier = str(filename)
        
        # Extract gem name
        gem_pattern = r'^(.+?)([BLU])([CP])(\d+)'
        gem_match = re.match(gem_pattern, identifier, re.IGNORECASE)
        
        if gem_match:
            prefix, light, orientation, scan = gem_match.groups()
            gem_name = f"{prefix}{light.upper()}{orientation.upper()}{scan}"
        else:
            gem_name = identifier.split('_')[0].split('.')[0]
        
        # Extract date and time
        date_pattern = r'(\d{8})(?:_(\d{6}))?'
        date_match = re.search(date_pattern, identifier)
        
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
        
        # Title
        title_label = ttk.Label(main_frame, text="Database Record Cleaner", 
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
        
        # Define columns - CHANGED: Added OriginalIndex to track real indices
        self.tree["columns"] = ("DisplayIndex", "OriginalIndex", "Filename", "Gem", "Date", "Time")
        
        # Configure tree column (for checkboxes)
        self.tree.column("#0", width=60, minwidth=60, anchor=tk.CENTER)
        self.tree.heading("#0", text="Select")
        
        # Configure data columns
        self.tree.column("DisplayIndex", width=60, minwidth=60, anchor=tk.CENTER)
        self.tree.column("OriginalIndex", width=0, minwidth=0)  # Hidden column for real index
        self.tree.column("Filename", width=400, minwidth=300, anchor=tk.W)
        self.tree.column("Gem", width=100, minwidth=80, anchor=tk.CENTER)
        self.tree.column("Date", width=100, minwidth=80, anchor=tk.CENTER)
        self.tree.column("Time", width=80, minwidth=60, anchor=tk.CENTER)
        
        self.tree.heading("DisplayIndex", text="Index")
        self.tree.heading("OriginalIndex", text="")  # Hidden
        self.tree.heading("Filename", text="Filename")
        self.tree.heading("Gem", text="Gem Name")
        self.tree.heading("Date", text="Date")
        self.tree.heading("Time", text="Time")
        
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
        
        # Separator
        ttk.Separator(button_frame, orient=tk.VERTICAL).grid(row=0, column=3, padx=15, sticky=(tk.N, tk.S))
        
        # Action buttons
        self.delete_button = ttk.Button(button_frame, text="DELETE SELECTED", 
                                       command=self.delete_selected)
        self.delete_button.grid(row=0, column=4, padx=10)
        self.delete_button.configure(state="disabled")
        
        ttk.Button(button_frame, text="Refresh", 
                  command=self.load_database).grid(row=0, column=5, padx=(10, 0))
        
        # Status labels
        self.selected_label = ttk.Label(main_frame, text="Selected: 0 records", 
                                       font=("Arial", 11, "bold"), foreground="red")
        self.selected_label.grid(row=4, column=0, pady=(10, 5), sticky=tk.W)
        
        self.status_label = ttk.Label(main_frame, text="Ready", 
                                     font=("Arial", 9), foreground="green")
        self.status_label.grid(row=5, column=0, sticky=tk.W)
    
    def load_database(self):
        """Load database data - forces fresh reload from disk"""
        try:
            self.status_label.config(text="Loading database...")
            self.root.update()
            
            # Clear any cached data
            self.df = None
            if self.conn:
                self.conn.close()
                self.conn = None
            
            # Find database files
            db_path, db_type = self.find_database_files()
            
            if not db_path or not db_type:
                messagebox.showerror("Error", 
                    "No database files found!\n\nExpected files:\n"
                    "- gemini_structural_db.csv\n"
                    "- multi_structural_gem_data.db")
                self.root.quit()
                return
            
            self.database_path = db_path
            self.database_type = db_type
            
            # Force reload from disk
            if db_type == "csv":
                self.load_csv_database()
            else:
                self.load_sqlite_database()
                
            self.populate_tree()
            self.update_counts()
            self.status_label.config(text=f"Loaded {len(self.df)} records from {db_type.upper()} database")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load database:\n{str(e)}")
            self.status_label.config(text="Error loading database")
    
    def load_csv_database(self):
        """Load CSV database"""
        csv_file = self.database_path / "gemini_structural_db.csv"
        self.df = pd.read_csv(csv_file)
        
        # IMPORTANT: Reset index to ensure sequential indices
        self.df = self.df.reset_index(drop=True)
        
        # Find identifier column
        identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
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
        
        self.info_label.config(text=f"CSV Database: {csv_file.name} | Column: {self.identifier_col}")
    
    def load_sqlite_database(self):
        """Load SQLite database"""
        sqlite_file = self.database_path / "multi_structural_gem_data.db"
        self.conn = sqlite3.connect(sqlite_file)
        cursor = self.conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            raise Exception("No tables found in SQLite database")
        
        self.table_name = tables[0][0]
        
        # Get all records
        cursor.execute(f"SELECT rowid, * FROM {self.table_name}")
        records = cursor.fetchall()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({self.table_name})")
        columns_info = cursor.fetchall()
        columns = ['rowid'] + [col[1] for col in columns_info]
        
        self.df = pd.DataFrame(records, columns=columns)
        
        # IMPORTANT: Reset index to ensure sequential indices
        self.df = self.df.reset_index(drop=True)
        
        # Find identifier column
        identifier_columns = ['file', 'filename', 'full_name', 'gem_name', 'identifier']
        self.identifier_col = None
        
        for col in identifier_columns:
            if col in self.df.columns:
                self.identifier_col = col
                break
        
        if not self.identifier_col:
            raise Exception("Could not find identifier column in SQLite")
        
        self.info_label.config(text=f"SQLite Database: {sqlite_file.name} | Table: {self.table_name} | Column: {self.identifier_col}")
    
    def populate_tree(self):
        """Populate the treeview with data"""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        self.selected_items.clear()
        
        # Add records - CHANGED: Store both display index and original dataframe index
        for display_idx, (df_idx, row) in enumerate(self.df.iterrows()):
            filename = row[self.identifier_col]
            gem_name, date_str, time_str = self.extract_gem_info(filename)
            
            # Format time for display
            time_display = ""
            if time_str and time_str != "000000":
                time_display = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"
            
            # Insert row - store both display index and actual dataframe index
            item_id = self.tree.insert("", "end", 
                                     text="☐",  # Checkbox in tree column
                                     values=(display_idx + 1, df_idx, filename, gem_name, date_str, time_display))
    
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
        """Delete selected records"""
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
                csv_file = self.database_path / "gemini_structural_db.csv"
                backup_file = self.database_path / f"gemini_structural_db_backup_{timestamp}.csv"
                shutil.copy2(csv_file, backup_file)
                backup_name = backup_file.name
            else:
                sqlite_file = self.database_path / "multi_structural_gem_data.db"
                backup_file = self.database_path / f"multi_structural_gem_data_backup_{timestamp}.db"
                shutil.copy2(sqlite_file, backup_file)
                backup_name = backup_file.name
            
            self.status_label.config(text="Deleting records...")
            self.root.update()
            
            # CHANGED: Get actual dataframe indices to delete
            df_indices_to_delete = []
            for item in self.selected_items:
                values = self.tree.item(item)['values']
                # Use the OriginalIndex (second value) instead of DisplayIndex (first value)
                df_index = int(values[1])  # OriginalIndex is the real dataframe index
                df_indices_to_delete.append(df_index)
            
            print(f"DEBUG: Deleting dataframe indices: {sorted(df_indices_to_delete)}")  # Debug info
            
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
            print(f"DEBUG: Error details: {e}")  # Debug info
            import traceback
            traceback.print_exc()
    
    def delete_csv_records(self, df_indices):
        """Delete records from CSV database"""
        csv_file = self.database_path / "gemini_structural_db.csv"
        
        print(f"DEBUG CSV: Original dataframe shape: {self.df.shape}")
        print(f"DEBUG CSV: Indices to delete: {sorted(df_indices)}")
        
        # CHANGED: Use .loc instead of .drop for more reliable deletion
        # Create boolean mask for rows to keep
        mask = ~self.df.index.isin(df_indices)
        df_cleaned = self.df[mask].copy()
        
        print(f"DEBUG CSV: Cleaned dataframe shape: {df_cleaned.shape}")
        
        # Reset index to ensure clean sequential indices
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        # Save cleaned CSV
        df_cleaned.to_csv(csv_file, index=False)
        
        print(f"DEBUG CSV: Saved cleaned CSV with {len(df_cleaned)} records")
        
        # Force file system sync
        import os
        if hasattr(os, 'sync'):
            os.sync()
        
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
                # FIXED: Convert numpy int64 to Python int for SQLite compatibility
                rowid = int(rowid)
                rowids_to_delete.append(rowid)
                print(f"DEBUG SQLite: df_index {df_index} -> rowid {rowid}")
        
        print(f"DEBUG SQLite: Rowids to delete: {sorted(rowids_to_delete)}")
        print(f"DEBUG SQLite: Rowid types: {[type(r) for r in rowids_to_delete[:3]]}")
        
        # Delete in batches
        batch_size = 100
        deleted_count = 0
        for i in range(0, len(rowids_to_delete), batch_size):
            batch = rowids_to_delete[i:i + batch_size]
            placeholders = ','.join(['?' for _ in batch])
            delete_query = f"DELETE FROM {self.table_name} WHERE rowid IN ({placeholders})"
            print(f"DEBUG SQLite: Executing query: {delete_query}")
            print(f"DEBUG SQLite: With batch: {batch[:5]}...")  # Show first 5 values
            result = cursor.execute(delete_query, batch)
            deleted_count += result.rowcount
            print(f"DEBUG SQLite: Batch {i//batch_size + 1}, deleted {result.rowcount} records")
        
        print(f"DEBUG SQLite: Total deleted: {deleted_count} records")
        
        # Force commit and close/reopen connection to ensure changes are written
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
        app = DatabaseCleanerGUI()
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()