#!/usr/bin/env python3
"""
Tkinter UI for the TSP Ant Colony Optimization solver.
Launches tsp_aco.py as a subprocess with the configured parameters.
"""
import os
import queue
import subprocess
import sys
import threading

# Tcl/Tk lives in the base Python install, not inside the venv.
# Set the library paths before importing tkinter so it can find init.tcl.
if sys.platform == "win32" and "TCL_LIBRARY" not in os.environ:
    _base_dir = os.path.dirname(getattr(sys, "_base_executable", sys.executable))
    for _tcl_ver in ("tcl8.6", "tcl8.5"):
        _tcl = os.path.join(_base_dir, "tcl", _tcl_ver)
        _tk  = os.path.join(_base_dir, "tcl", _tcl_ver.replace("tcl", "tk"))
        if os.path.isdir(_tcl):
            os.environ["TCL_LIBRARY"] = _tcl
            os.environ["TK_LIBRARY"]  = _tk
            break

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk

_DIR    = os.path.dirname(os.path.abspath(__file__))
_PYTHON = os.path.join(_DIR, "venv", "Scripts", "python.exe")
_SCRIPT = os.path.join(_DIR, "tsp_aco.py")
_CITIES = os.path.join(_DIR, "cities.txt")

if not os.path.exists(_PYTHON):
    _PYTHON = sys.executable

_CUDA_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"


def _build_env() -> dict:
    env = os.environ.copy()
    if os.path.exists(_CUDA_PATH):
        env["CUDA_PATH"] = _CUDA_PATH
        env["CUDA_HOME"] = _CUDA_PATH
        env["PATH"] = (
            _CUDA_PATH + r"\bin;"
            + _CUDA_PATH + r"\nvvm\bin;"
            + _CUDA_PATH + r"\nvvm\bin\x64;"
            + env.get("PATH", "")
        )
    return env


# ── Main application ──────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TSP  —  Ant Colony Optimization")
        self.minsize(920, 560)
        self._proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        self._build_params(left)

        right = ttk.Frame(self, padding=(0, 10, 10, 10))
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self._build_output(right)

    def _build_params(self, parent):
        f = ttk.LabelFrame(parent, text="Parameters", padding=10)
        f.pack(fill=tk.BOTH, expand=True)

        def row_label(r, text):
            ttk.Label(f, text=text).grid(row=r, column=0, sticky="w", pady=2)

        def separator(r):
            ttk.Separator(f, orient="horizontal").grid(
                row=r, column=0, columnspan=2, sticky="ew", pady=6)

        r = 0

        # Mode
        row_label(r, "Mode")
        self._mode = tk.StringVar(value="cpu")
        mf = ttk.Frame(f)
        mf.grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(mf, text="CPU",  variable=self._mode, value="cpu").pack(side=tk.LEFT)
        ttk.Radiobutton(mf, text="CUDA", variable=self._mode, value="cuda").pack(side=tk.LEFT, padx=(8, 0))
        r += 1; separator(r); r += 1

        # Input source
        row_label(r, "Input")
        self._input_src = tk.StringVar(value="file")
        isf = ttk.Frame(f)
        isf.grid(row=r, column=1, sticky="w")
        ttk.Radiobutton(isf, text="File",   variable=self._input_src, value="file",
                        command=self._toggle_input).pack(side=tk.LEFT)
        ttk.Radiobutton(isf, text="Random", variable=self._input_src, value="random",
                        command=self._toggle_input).pack(side=tk.LEFT, padx=(8, 0))
        r += 1

        # File path
        row_label(r, "File")
        ff = ttk.Frame(f)
        ff.grid(row=r, column=1, sticky="ew")
        self._file = tk.StringVar(value=_CITIES)
        self._file_entry = ttk.Entry(ff, textvariable=self._file, width=20)
        self._file_entry.pack(side=tk.LEFT)
        self._browse_btn = ttk.Button(ff, text="...", width=3, command=self._browse)
        self._browse_btn.pack(side=tk.LEFT, padx=(4, 0))
        r += 1

        # Random city count
        row_label(r, "City count")
        self._cities = tk.IntVar(value=50)
        self._cities_spin = ttk.Spinbox(f, from_=3, to=5000,
                                        textvariable=self._cities, width=8)
        self._cities_spin.grid(row=r, column=1, sticky="w")
        r += 1; separator(r); r += 1

        # ACO parameters
        aco_params = [
            ("Ants",              "_ants",    tk.IntVar,    50,    (1, 5000)),
            ("Iterations",        "_iters",   tk.IntVar,    200,   (1, 99999)),
            ("Alpha (pheromone)", "_alpha",   tk.DoubleVar, 1.0,   None),
            ("Beta (heuristic)",  "_beta",    tk.DoubleVar, 3.0,   None),
            ("Rho (evaporation)", "_rho",     tk.DoubleVar, 0.1,   None),
            ("Q (deposit)",       "_Q",       tk.DoubleVar, 100.0, None),
            ("Init. pheromone",   "_init_ph", tk.DoubleVar, 1.0,   None),
            ("Seed (optional)",   "_seed",    tk.StringVar, "",    None),
        ]

        for label, attr, var_cls, default, spin_range in aco_params:
            row_label(r, label)
            var = var_cls(value=default)
            setattr(self, attr, var)
            if spin_range and var_cls is tk.IntVar:
                w = ttk.Spinbox(f, from_=spin_range[0], to=spin_range[1],
                                textvariable=var, width=9)
            else:
                w = ttk.Entry(f, textvariable=var, width=10)
            w.grid(row=r, column=1, sticky="w", pady=2)
            r += 1

        separator(r); r += 1

        # Verbose
        self._verbose = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Show iteration progress",
                        variable=self._verbose).grid(
            row=r, column=0, columnspan=2, sticky="w")
        r += 1; separator(r); r += 1

        # Action buttons
        bf = ttk.Frame(f)
        bf.grid(row=r, column=0, columnspan=2)
        self._run_btn  = ttk.Button(bf, text="Run",   width=10, command=self._run)
        self._stop_btn = ttk.Button(bf, text="Stop",  width=7,  command=self._stop,
                                    state="disabled")
        self._clr_btn  = ttk.Button(bf, text="Clear", width=7,  command=self._clear_output)
        self._run_btn.pack(side=tk.LEFT)
        self._stop_btn.pack(side=tk.LEFT, padx=(6, 0))
        self._clr_btn.pack(side=tk.LEFT, padx=(6, 0))

        self._toggle_input()

    def _build_output(self, parent):
        self._out = scrolledtext.ScrolledText(
            parent, wrap=tk.NONE, font=("Courier New", 9),
            state="disabled", bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white", width=80, height=30)
        self._out.grid(row=0, column=0, sticky="nsew")

        # Tag colours for stats sections
        self._out.tag_config("header",  foreground="#4ec9b0")
        self._out.tag_config("section", foreground="#9cdcfe")
        self._out.tag_config("value",   foreground="#ce9178")
        self._out.tag_config("cmd",     foreground="#666666")
        self._out.tag_config("error",   foreground="#f44747")

        hbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL,
                             command=self._out.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        self._out.configure(xscrollcommand=hbar.set)

        self._status = ttk.Label(parent, text="Ready", anchor="w")
        self._status.grid(row=2, column=0, sticky="ew", pady=(4, 0))

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _toggle_input(self):
        using_file = self._input_src.get() == "file"
        self._file_entry.config( state="normal"    if using_file else "disabled")
        self._browse_btn.config( state="normal"    if using_file else "disabled")
        self._cities_spin.config(state="disabled"  if using_file else "normal")

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select city file",
            initialdir=_DIR,
            filetypes=[("Text files", "*.txt"), ("TSP files", "*.tsp"),
                       ("All files", "*.*")],
        )
        if path:
            self._file.set(path)

    def _write(self, text: str, tag: str = ""):
        self._out.config(state="normal")
        if tag:
            self._out.insert(tk.END, text, tag)
        else:
            self._out.insert(tk.END, text)
        self._out.see(tk.END)
        self._out.config(state="disabled")

    def _replace_last_line(self, text: str):
        """Handle \\r: overwrite the current last line (progress update)."""
        self._out.config(state="normal")
        self._out.delete("end-1l linestart", "end-1c")
        self._out.insert(tk.END, text)
        self._out.see(tk.END)
        self._out.config(state="disabled")

    def _clear_output(self):
        self._out.config(state="normal")
        self._out.delete("1.0", tk.END)
        self._out.config(state="disabled")
        self._status.config(text="Ready")

    # ── Run logic ─────────────────────────────────────────────────────────────

    def _build_args(self) -> list[str]:
        args = [_PYTHON, _SCRIPT, f"--{self._mode.get()}"]

        if self._input_src.get() == "file":
            args += ["--file", self._file.get()]
        else:
            args += ["--cities", str(self._cities.get())]

        args += [
            "--ants",              str(self._ants.get()),
            "--iterations",        str(self._iters.get()),
            "--alpha",             str(self._alpha.get()),
            "--beta",              str(self._beta.get()),
            "--rho",               str(self._rho.get()),
            "--Q",                 str(self._Q.get()),
            "--initial-pheromone", str(self._init_ph.get()),
        ]

        seed = str(self._seed.get()).strip()
        if seed:
            args += ["--seed", seed]

        if not self._verbose.get():
            args.append("--quiet")

        return args

    def _run(self):
        args = self._build_args()
        self._clear_output()
        self._write(" ".join(
            f'"{a}"' if " " in a else a for a in args) + "\n\n", "cmd")

        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._status.config(text="Running...")

        threading.Thread(target=self._worker, args=(args,), daemon=True).start()
        self._poll()

    def _worker(self, args: list[str]):
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=_build_env(),
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        buf = b""
        while True:
            ch = self._proc.stdout.read(1)
            if not ch:
                break
            if ch == b"\r":
                self._q.put(("cr", buf.decode("utf-8", errors="replace")))
                buf = b""
            elif ch == b"\n":
                self._q.put(("line", buf.decode("utf-8", errors="replace")))
                buf = b""
            else:
                buf += ch
        if buf:
            self._q.put(("line", buf.decode("utf-8", errors="replace")))
        self._proc.wait()
        self._q.put(("done", self._proc.returncode))
        self._proc = None

    def _poll(self):
        try:
            while True:
                msg = self._q.get_nowait()
                kind = msg[0]
                if kind == "cr":
                    self._replace_last_line(msg[1])
                elif kind == "line":
                    self._write(msg[1] + "\n")
                elif kind == "done":
                    rc = msg[1]
                    self._run_btn.config(state="normal")
                    self._stop_btn.config(state="disabled")
                    if rc == 0:
                        self._status.config(text="Completed.")
                    elif rc is None or rc < 0:
                        self._status.config(text="Stopped.")
                    else:
                        self._status.config(text=f"Finished with error (exit {rc})")
                    return
        except queue.Empty:
            pass
        self.after(40, self._poll)

    def _stop(self):
        if self._proc:
            self._proc.terminate()

    def _on_close(self):
        self._stop()
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
