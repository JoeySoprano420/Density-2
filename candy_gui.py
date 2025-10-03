import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess, sys, os, shlex

def run_file(p):
    try:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),"candywrapper.py"), p], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Candy Wrapper", f"Build/Run failed.\n\n{e}")

def choose():
    p = filedialog.askopenfilename(filetypes=[("Assembly/Objects/Density-2",".asm .obj .den"),
                                              ("All files","*.*")])
    if p:
        run_file(p)

root = tk.Tk()
root.title("Candy Wrapper")
root.geometry("420x160")
lbl = tk.Label(root, text="Drop a .asm / .obj / .den here, or click Browse.", font=("Segoe UI", 11))
lbl.pack(pady=16)
btn = tk.Button(root, text="Browse…", command=choose, width=18)
btn.pack(pady=8)

def dnd(ev):
    # Basic drag support (Windows shell drops a quoted path)
    p = ev.data.strip().strip("{}").strip()
    if p:
        run_file(p)
try:
    # TkDnD might not be present; keep UI minimal
    root.drop_target_register('DND_Files')
    root.dnd_bind('<<Drop>>', dnd)
except Exception:
    pass

root.mainloop()
