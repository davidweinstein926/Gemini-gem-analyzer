#ZIP Aseq_Python dir
import os
import zipfile
from tkinter import Tk, filedialog, messagebox
from datetime import datetime

def zip_active_directory():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select your Aseq_Python directory")
    if not folder:
        return

    today = datetime.now().strftime("%Y_%m_%d")
    zip_filename = f"aseq_python_backup_{today}.zip"
    zip_path = os.path.join(folder, zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root_dir, _, files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root_dir, file)
                arcname = os.path.relpath(file_path, folder)
                if arcname != zip_filename:  # avoid including the zip itself
                    zipf.write(file_path, arcname)

    messagebox.showinfo("Backup Complete", f"✅ Backup saved as:\n\n{zip_path}")

if __name__ == "__main__":
    zip_active_directory()