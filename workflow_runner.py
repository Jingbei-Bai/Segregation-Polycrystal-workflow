#!/usr/bin/env python3
"""
workflow_runner.py

Integrated workflow GUI (English UI).

Features (button order):
 1) Generate polycrystal (uses generate_polycrystal)
 2) Generate LAMMPS in file (uses generate_lammps_in)
 3) Extract GB atom ids/indices (optionally via OVITO)
 4) Compute SOAP descriptors and train ML models
 5) Visualizations

Note: This script only prepares files and calls local Python interfaces; it does not
run external binaries (atomsk, lmp) by default. Users can inspect prepared commands
and run them in an environment with those tools installed.
"""
from pathlib import Path
import threading
import traceback
import numpy as np
import subprocess
import sys
import shutil
import json

# Ensure the project/script directory is on sys.path so local modules can be imported
# This is robust when the script is launched from a different working directory.
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
except Exception:
    pass

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
    from tkinter import ttk
except Exception:
    tk = None

# Import project modules
from generate_polycrystal import generate_polycrystal
from generate_lammps_in import generate_lammps_in
from generate_ml_with_soap import (
    load_energy_map,
    compute_soap_features,
    map_ids_to_indices_from_dump,
    train_and_save_models,
    _plot_parity,
    _plot_feature_importance,
    compute_skew_from_y,
)
try:
    from generate_skew_data import plot_skew_distribution
except Exception:
    plot_skew_distribution = None


# Try to import OVITO interface functions (generate_get_gb_ids.py)
try:
    from generate_get_gb_ids import get_gb_ids_and_indices
    HAVE_OVITO_INTERFACE = True
except Exception:
    get_gb_ids_and_indices = None
    HAVE_OVITO_INTERFACE = False


if tk is None:
    print('tkinter is not available in the current environment; GUI cannot be started.')
    raise SystemExit(1)


class WorkflowApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('Segregation Workflow Control Panel')

        frm = tk.Frame(root, padx=8, pady=8)
        frm.pack(fill='both', expand=True)

        # Left: inputs; Right: logs
        left = tk.Frame(frm)
        right = tk.Frame(frm)
        left.grid(row=0, column=0, sticky='n')
        right.grid(row=0, column=1, sticky='n')

        # --- Inputs ---
        tk.Label(left, text='1) Generate polycrystal').grid(row=0, column=0, sticky='w')
        tk.Label(left, text='Box (e.g. 70 or "70 80 90")').grid(row=1, column=0, sticky='w')
        self.ent_box = tk.Entry(left, width=20)
        self.ent_box.insert(0, '70')
        self.ent_box.grid(row=1, column=1)

        tk.Label(left, text='Random seed (integer)').grid(row=2, column=0, sticky='w')
        self.ent_random = tk.Entry(left, width=20)
        self.ent_random.insert(0, '20')
        self.ent_random.grid(row=2, column=1)

        tk.Label(left, text='polycrystal output (polycrystal.txt)').grid(row=3, column=0, sticky='w')
        self.ent_poly_out = tk.Entry(left, width=30)
        self.ent_poly_out.insert(0, 'polycrystal.txt')
        self.ent_poly_out.grid(row=3, column=1)

        tk.Button(left, text='Generate polycrystal file', bg='#4caf50', fg='white', command=self.on_generate_poly).grid(row=4, column=1, pady=6, sticky='w')
        # Preselect XSF (user can preselect for later atomsk steps)
        self.selected_xsf = None
        tk.Button(left, text='Select XSF (optional)', command=self.on_select_xsf).grid(row=4, column=2, padx=6, pady=6, sticky='w')
        self.xsf_label = tk.Label(left, text='No XSF selected', anchor='w')
        self.xsf_label.grid(row=5, column=2, sticky='w')

        # Prepare polycrystal.lmp (command only, do not execute atomsk)
        tk.Button(left, text='Generate polycrystal and prepare atomsk command', bg='#6a1b9a', fg='white', command=self.on_generate_poly_and_prepare_lmp).grid(row=6, column=1, columnspan=2, padx=6, pady=6, sticky='w')

        # LAMMPS in
        tk.Label(left, text='2) Generate LAMMPS in').grid(row=7, column=0, sticky='w', pady=(10,0))
        tk.Label(left, text='read_data (lmp file and extra params)').grid(row=8, column=0, sticky='w')
        self.ent_read_data = tk.Entry(left, width=40)
        self.ent_read_data.insert(0, 'polycrystal.lmp extra/atom/types 1')
        self.ent_read_data.grid(row=8, column=1)

        tk.Label(left, text='EAM file').grid(row=9, column=0, sticky='w')
        self.ent_eam = tk.Entry(left, width=30)
        self.ent_eam.insert(0, 'AlCu.eam')
        self.ent_eam.grid(row=9, column=1)

        tk.Label(left, text='append filename').grid(row=10, column=0, sticky='w')
        self.ent_append = tk.Entry(left, width=30)
        # Default set to project energy/input file Al_Cu.txt (original example used medium_ file)
        self.ent_append.insert(0, 'Al_Cu.txt')
        self.ent_append.grid(row=10, column=1)

        tk.Label(left, text='in output (e.g. in.lmp)').grid(row=11, column=0, sticky='w')
        self.ent_in_out = tk.Entry(left, width=30)
        self.ent_in_out.insert(0, 'in.lmp')
        self.ent_in_out.grid(row=11, column=1)

        tk.Button(left, text='Generate in file', bg='#2196f3', fg='white', command=self.on_generate_in).grid(row=12, column=1, pady=6, sticky='w')

        # Add MPI ranks input and a button to run the generated in file with LAMMPS
        tk.Label(left, text='MPI ranks').grid(row=13, column=0, sticky='w')
        self.ent_mpi = tk.Entry(left, width=8)
        self.ent_mpi.insert(0, '1')
        self.ent_mpi.grid(row=13, column=1, sticky='w')
        tk.Button(left, text='Run in file (LAMMPS)', bg='#e91e63', fg='white', command=self.on_run_in).grid(row=14, column=1, pady=6, sticky='w')

        # GB extraction
        tk.Label(left, text='3) Extract GB atom ids').grid(row=16, column=0, sticky='w', pady=(10,0))
        self.gb_status = tk.Label(left, text='GB not extracted')
        self.gb_status.grid(row=16, column=1, sticky='w')
        tk.Button(left, text='Compute GB ids with OVITO (if available)', command=self.on_compute_gb).grid(row=17, column=0, pady=6, sticky='w')
        tk.Button(left, text='Load GB ids/indices from files', command=self.on_load_gb_files).grid(row=17, column=1, pady=6, sticky='w')

        # SOAP & ML
        tk.Label(left, text='4) Compute SOAP and train ML').grid(row=18, column=0, sticky='w', pady=(10,0))
        tk.Label(left, text='Energy two-column file (id value)').grid(row=19, column=0, sticky='w')
        self.ent_energy = tk.Entry(left, width=40)
        self.ent_energy.insert(0, 'Al_Cu.txt')
        self.ent_energy.grid(row=19, column=1)

        tk.Label(left, text='Dump file (for SOAP)').grid(row=20, column=0, sticky='w')
        self.ent_dump = tk.Entry(left, width=40)
        self.ent_dump.grid(row=20, column=1)

        tk.Button(left, text='Compute SOAP and train', bg='#ff9800', fg='white', command=self.on_compute_soap_and_train).grid(row=21, column=1, pady=6, sticky='w')

        # Visualization controls
        tk.Label(left, text='5) Visualization').grid(row=22, column=0, sticky='w', pady=(10,0))
        tk.Button(left, text='Show parity plot', command=self.on_show_parity).grid(row=23, column=0, sticky='w')
        tk.Button(left, text='Show feature importance', command=self.on_show_fi).grid(row=23, column=1, sticky='w')
        tk.Button(left, text='Show skew distribution', command=self.on_show_skew).grid(row=23, column=2, sticky='w')

        # --- Logs ---
        tk.Label(right, text='Log / Output').pack(anchor='w')
        self.txt_log = scrolledtext.ScrolledText(right, width=80, height=40)
        self.txt_log.pack()

        # internal state
        self.last_poly_path = None
        self.last_in_path = None
        self.last_lmp_path = None
        self.gb_ids = None
        self.gb_indices = None
        self.soap_X = None
        self.y = None
        self.ml_out_dir = Path('ml_output')

    def log(self, msg: str):
        self.txt_log.insert(tk.END, msg + '\n')
        self.txt_log.see(tk.END)

    def on_generate_poly(self):
        box = self.ent_box.get().strip()
        rnd = self.ent_random.get().strip()
        out = self.ent_poly_out.get().strip() or 'polycrystal.txt'
        try:
            rnd_i = int(rnd)
        except Exception:
            messagebox.showerror('Parameter error', 'Random must be an integer')
            return
        try:
            path = generate_polycrystal(box, rnd_i, out=out, show_preview=False, print_command=False)
            self.last_poly_path = Path(path)
            self.log(f'Generated polycrystal file: {path}')
            try:
                self.log('File content preview:')
                self.log(path.read_text(encoding='utf-8'))
            except Exception as e:
                self.log(f'Failed to read generated file: {e}')
        except Exception as e:
            self.log('Failed to generate polycrystal: ' + str(e))
            traceback.print_exc()

    def on_select_xsf(self):
        p = filedialog.askopenfilename(title='Select aluminium.xsf or XSF file (optional)', filetypes=[('XSF files','*.xsf'),('All','*.*')])
        if not p:
            return
        self.selected_xsf = Path(p)
        self.xsf_label.config(text=self.selected_xsf.name)
        self.log(f'Selected XSF: {self.selected_xsf}')

    def on_generate_poly_and_prepare_lmp(self):
        """Generate a polycrystal file and prepare/run atomsk to create polycrystal.lmp.

        Workflow: generate polycrystal -> prompt for XSF if not preselected -> run atomsk in background -> update log and read_data field.
        """
        box = self.ent_box.get().strip()
        rnd = self.ent_random.get().strip()
        out = self.ent_poly_out.get().strip() or 'polycrystal.txt'
        try:
            rnd_i = int(rnd)
        except Exception:
            messagebox.showerror('Parameter error', 'Random must be an integer')
            return
        try:
            path = generate_polycrystal(box, rnd_i, out=out, show_preview=False, print_command=False)
            self.last_poly_path = Path(path)
            self.log(f'Generated polycrystal file: {path}')
            try:
                self.log('File content preview:')
                self.log(path.read_text(encoding='utf-8'))
            except Exception as e:
                self.log(f'Failed to read generated file: {e}')
        except Exception as e:
            self.log('Failed to generate polycrystal: ' + str(e))
            traceback.print_exc()
            return

        # Prepare atomsk command (only prepare, do not execute here)
        poly_path = Path(path)
        xsf_path = self.selected_xsf
        cmd = None
        try:
            txt = poly_path.read_text(encoding='utf-8')
            for line in txt.splitlines():
                if not line:
                    continue
                candidate = line.strip()
                if candidate.startswith('#'):
                    candidate = candidate.lstrip('#').strip()
                if not candidate:
                    continue
                if candidate.startswith('atomsk') or '--polycrystal' in candidate:
                    try:
                        import shlex
                        parsed = shlex.split(candidate)
                        if any(p == 'atomsk' for p in parsed):
                            has_lmp = any(str(p).lower().endswith('.lmp') for p in parsed)
                            if not has_lmp:
                                parsed = parsed + ['polycrystal.lmp']
                                if '-wrap' not in parsed:
                                    parsed = parsed + ['-wrap']
                            cmd = parsed
                            break
                    except Exception:
                        continue
        except Exception:
            cmd = None

        if cmd is None:
            # If user preselected XSF, use it in the placeholder command
            if xsf_path is not None:
                cmd = ['atomsk', '--polycrystal', str(xsf_path), str(poly_path.name), 'polycrystal.lmp', '-wrap']
                self.log('No atomsk command found in polycrystal file; using default (with selected XSF): ' + ' '.join(cmd))
            else:
                cmd = ['atomsk', '--polycrystal', '<xsf>', str(poly_path.name), 'polycrystal.lmp', '-wrap']
                self.log('No atomsk command found in polycrystal file; generated default command (please select XSF or replace manually): ' + ' '.join(cmd))

        # Note: external command preview removed from GUI. Commands are shown in log instead.
        self.log('Prepared atomsk command: ' + ' '.join(cmd))

        # Auto-fill read_data field with the lmp filename
        try:
            self.ent_read_data.delete(0, tk.END)
            self.ent_read_data.insert(0, f'{poly_path.name} extra/atom/types 1')
        except Exception:
            pass

        # Prompt for XSF if needed and run atomsk in background (no extra confirmation)
        if xsf_path is None:
            default_xsf = Path('aluminium.xsf')
            if default_xsf.exists():
                xsf_path = default_xsf
                self.selected_xsf = xsf_path
                self.xsf_label.config(text=self.selected_xsf.name)
                self.log(f'No XSF preselected; automatically using {xsf_path}')
            else:
                p = filedialog.askopenfilename(title='Please select an XSF file to run atomsk', filetypes=[('XSF files','*.xsf'),('All','*.*')])
                if not p:
                    self.log('No XSF selected; canceling atomsk')
                    return
                xsf_path = Path(p)
                self.selected_xsf = xsf_path
                self.xsf_label.config(text=self.selected_xsf.name)

        # Construct command and run in background (uses user-specified output `out`, default polycrystal.txt)
        cmd = ['atomsk', '--polycrystal', str(xsf_path), out, 'polycrystal.lmp', '-wrap']
        self.log('Atomsk command to be run (background): ' + ' '.join(cmd))

        # Run atomsk in background using the original _run_atomsk logic
        def _run_atomsk(prepared_cmd, workdir):
            self.log(f'Starting atomsk in background: {" ".join(prepared_cmd)} (cwd={workdir})')
            try:
                # Record directory entries before execution to identify files created by atomsk
                pre_existing = set(Path(workdir).iterdir())
                proc = subprocess.run(prepared_cmd, capture_output=True, text=True, cwd=str(workdir))
            except FileNotFoundError:
                self.log('Error: atomsk not found. Please ensure atomsk is installed and available in PATH')
                def _show_not_found(*args: object) -> None:
                    messagebox.showerror('Execution failed', 'atomsk not found. Ensure atomsk is installed and on your PATH.')
                self.root.after(0, _show_not_found)  # type: ignore[arg-type]
                return
            except Exception as e:
                self.log('Exception while running atomsk: ' + str(e))
                traceback.print_exc()
                return

            self.log('--- atomsk stdout ---')
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    self.log(line)
            self.log('--- atomsk stderr ---')
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    self.log(line)

            if proc.returncode != 0:
                self.log(f'atomsk returned non-zero exit code: {proc.returncode}')
                def _show_failed(*args: object) -> None:
                    messagebox.showerror('atomsk failed', f'atomsk returned non-zero exit code: {proc.returncode}\nPlease check logs for details.')
                self.root.after(0, _show_failed)  # type: ignore[arg-type]
                return

            # Inspect output .lmp file path
            out_lmp = None
            for t in prepared_cmd[::-1]:
                if str(t).lower().endswith('.lmp'):
                    out_lmp = Path(workdir) / Path(t).name
                    break
            if out_lmp is None:
                out_lmp = Path(workdir) / 'polycrystal.lmp'

            if out_lmp.exists():
                self.last_lmp_path = out_lmp
                self.log(f'atomsk completed successfully, generated: {out_lmp.resolve()}')
                def _show_success(*args: object) -> None:
                    messagebox.showinfo('atomsk complete', f'Generated: {out_lmp}')
                self.root.after(0, _show_success)  # type: ignore[arg-type]
                # Identify and prompt to delete extra files created by atomsk (keep polycrystal.lmp), and optionally delete polycrystal.txt
                try:
                    post_existing = set(Path(workdir).iterdir())
                    created = post_existing - pre_existing
                    # Exclude polycrystal.lmp from deletion candidates
                    to_delete = [p for p in created if p.name != out_lmp.name]
                    # Also include the polycrystal.txt used in this run (if present)
                    try:
                        if poly_path.exists() and poly_path not in to_delete:
                            to_delete.append(poly_path)
                    except Exception:
                        pass
                    if to_delete:
                        preview = '\n'.join([p.name for p in sorted(to_delete)])
                        confirm = messagebox.askyesno('Confirm deletion', f'The following files/folders were created by atomsk (polycrystal.lmp will be kept):\n\n{preview}\n\nDelete these files and polycrystal.txt?')
                        if confirm:
                            deleted = 0
                            for p in to_delete:
                                try:
                                    if p.is_dir():
                                        shutil.rmtree(p)
                                    else:
                                        p.unlink()
                                    deleted += 1
                                    self.log(f'Deleted: {p.name}')
                                except Exception as e:
                                    self.log(f'Failed to delete {p}: {e}')
                            messagebox.showinfo('Cleanup complete', f'Deleted {deleted} items (polycrystal.lmp kept)')
                        else:
                            self.log('User canceled automatic deletion of atomsk-created files')
                except Exception as e:
                    self.log(f'Error while attempting to identify/delete atomsk-created files: {e}')
            else:
                self.log('atomsk did not produce expected .lmp output; please check logs')

        threading.Thread(target=_run_atomsk, args=(cmd, poly_path.parent), daemon=True).start()

    def on_generate_in(self):
        read_data = self.ent_read_data.get().strip()
        eam = self.ent_eam.get().strip() or 'AlCu.eam'
        # If user left append blank, use Al_Cu.txt as fallback default
        append = self.ent_append.get().strip() or 'Al_Cu.txt'
        out = self.ent_in_out.get().strip() or 'in.lmp'
        try:
            # Infer out_tui from out
            p_out = Path(out)
            if p_out.name == 'in':
                 out_curr_tui = p_out.parent / 'in_tui'
            else:
                 out_curr_tui = p_out.parent / (p_out.stem + '_tui' + p_out.suffix)

            path_tui, path_in = generate_lammps_in(read_data, append, eam, out_tui=str(out_curr_tui), out_in=out)
            self.last_in_path = Path(path_in)
            self.log(f'Generated files: {path_tui}, {path_in}')
            try:
                self.log(f'--- {path_tui.name} ---')
                self.log(path_tui.read_text(encoding='utf-8'))
                self.log(f'--- {path_in.name} ---')
                self.log(path_in.read_text(encoding='utf-8'))
            except Exception as e:
                self.log(f'Failed to read generated files: {e}')
        except Exception as e:
            self.log('Failed to generate in file: ' + str(e))
            traceback.print_exc()

    def on_run_in(self):
        """Run a LAMMPS input file (in.lmp) in background. Checks for lmp and mpiexec/mpirun in PATH."""
        # Determine in file path
        in_spec = self.ent_in_out.get().strip()
        in_path = None
        if self.last_in_path is not None and Path(self.last_in_path).exists():
            in_path = Path(self.last_in_path)
        else:
            # if entry contains a path that exists, use it
            if in_spec:
                p = Path(in_spec)
                if p.exists():
                    in_path = p
        if in_path is None:
            sel = filedialog.askopenfilename(title='Select LAMMPS in file (in.lmp)', filetypes=[('LAMMPS in','*.lmp;*.in'),('All','*.*')])
            if not sel:
                self.log('No in file selected; canceling LAMMPS run')
                return
            in_path = Path(sel)

        # MPI ranks
        ranks = 1
        try:
            ranks = int(self.ent_mpi.get().strip())
            if ranks < 1:
                ranks = 1
        except Exception:
            ranks = 1

        # Locate executables
        possible_names = ['lmp', 'lmp_serial', 'lmp_mpi', 'lammps']
        lmp_bin = None
        for name in possible_names:
            p = shutil.which(name)
            if p:
                lmp_bin = p
                break
        mpiexec = shutil.which('mpiexec') or shutil.which('mpirun')

        if lmp_bin is None:
            self.log('LAMMPS executable not found in PATH (searched: ' + ','.join(possible_names) + ').')
            messagebox.showerror('Execution failed', 'LAMMPS executable not found in PATH. Please install LAMMPS or add it to PATH.')
            return

        if mpiexec and ranks > 1:
            cmd = [mpiexec, '-n', str(ranks), lmp_bin, '-in', in_path.name]
        else:
            cmd = [lmp_bin, '-in', in_path.name]

        # Check if there is a corresponding _tui file to run first
        tui_path = None
        # Logic matches generate_lammps_in:
        # If out="in", tui="in_tui"
        # If out="in.lmp", tui="in_tui.lmp"
        candidate_same_ext = in_path.with_name(f"{in_path.stem}_tui{in_path.suffix}")
        if candidate_same_ext.exists():
            tui_path = candidate_same_ext
        else:
            # Fallback: check no extension or different extension common patterns
            candidate_no_ext = in_path.with_name(f"{in_path.stem}_tui")
            if candidate_no_ext.exists():
                tui_path = candidate_no_ext

        def _run_lammps(prepared_cmd, workdir):
            if tui_path:
                self.log(f'Detected setup script {tui_path.name}; running it first...')
                if mpiexec and ranks > 1:
                    cmd_tui = [mpiexec, '-n', str(ranks), lmp_bin, '-in', tui_path.name]
                else:
                    cmd_tui = [lmp_bin, '-in', tui_path.name]

                self.log('Starting LAMMPS (Setup): ' + ' '.join(cmd_tui) + f' (cwd={workdir})')
                try:
                    proc_tui = subprocess.run(cmd_tui, capture_output=True, text=True, cwd=str(workdir))
                    if proc_tui.returncode != 0:
                        self.log(f'Setup script {tui_path.name} failed with code {proc_tui.returncode}. Aborting main run.')
                        self.root.after(0, lambda: messagebox.showerror('Setup failed', f'Setup script {tui_path.name} failed.'))
                        return
                    self.log(f'Setup script {tui_path.name} finished successfully.')
                except Exception as e:
                    self.log(f'Failed to run setup script: {e}')
                    return

            self.log('Starting LAMMPS (Main): ' + ' '.join(prepared_cmd) + f' (cwd={workdir})')
            try:
                pre_existing = set(Path(workdir).iterdir())
                proc = subprocess.run(prepared_cmd, capture_output=True, text=True, cwd=str(workdir))
            except FileNotFoundError:
                self.log('Error: LAMMPS executable not found at runtime. Please ensure LAMMPS is installed and available in PATH')
                def _show_not_found(*args: object) -> None:
                    messagebox.showerror('Execution failed', 'LAMMPS executable not found. Ensure LAMMPS is installed and on your PATH.')
                self.root.after(0, _show_not_found)  # type: ignore[arg-type]
                return
            except Exception as e:
                self.log('Exception while running LAMMPS: ' + str(e))
                traceback.print_exc()
                return

            # Do not stream full LAMMPS stdout/stderr into the GUI log to avoid flooding
            # the visualization pane. Only report a concise status; users can inspect
            # LAMMPS' own log files in the working directory for full details.
            self.log('LAMMPS finished (full stdout/stderr not displayed here).')

            if proc.returncode != 0:
                self.log(f'LAMMPS returned non-zero exit code: {proc.returncode}')
                def _show_failed(*args: object) -> None:
                    messagebox.showerror('LAMMPS failed', f'LAMMPS returned non-zero exit code: {proc.returncode}\nPlease check logs for details.')
                self.root.after(0, _show_failed)  # type: ignore[arg-type]
            else:
                self.log('LAMMPS completed successfully')
                # After success, attempt to auto-sort dump.atom if present in working dir.
                try:
                    dump_path = Path(workdir) / 'dump.atom'
                    if dump_path.exists():
                        self.log(f'Found dump file: {dump_path.name}; attempting to sort')
                        try:
                            sorted_path = self._sort_dump_file(dump_path, header_lines=9)
                            self.log(f'Dump sorting complete: {sorted_path.name}')
                        except Exception as e:
                            self.log(f'Failed to sort dump.atom: {e}')
                except Exception as e:
                    self.log(f'Error while attempting to auto-sort dump.atom: {e}')
                def _show_done(*args: object) -> None:
                    messagebox.showinfo('LAMMPS complete', f'LAMMPS finished (cwd={workdir})')
                self.root.after(0, _show_done)  # type: ignore[arg-type]

        threading.Thread(target=_run_lammps, args=(cmd, in_path.parent), daemon=True).start()

    def on_compute_gb(self):
        # use OVITO-based function if available
        if HAVE_OVITO_INTERFACE and get_gb_ids_and_indices is not None:
            # ask for lmp file
            p = filedialog.askopenfilename(title='Select LAMMPS file (for OVITO GB detection)', filetypes=[('LAMMPS','*.lmp;*.dump;*.data'),('All','*.*')])
            if not p:
                return
            self.log(f'Calling OVITO interface to compute GB (file: {p}) ...')
            try:
                gb_ids, gb_indices = get_gb_ids_and_indices(p)
                self.gb_ids = gb_ids
                self.gb_indices = gb_indices
                try:
                    self.gb_status.config(text=f'GB extracted ({gb_ids.size} atoms)')
                except Exception:
                    # gb_ids may be a list
                    try:
                        self.gb_status.config(text=f'GB extracted ({len(gb_ids)} atoms)')
                    except Exception:
                        pass
                # save copies
                np = __import__('numpy')
                np.save('gb_ids_from_ovito.npy', gb_ids)
                np.save('gb_indices_from_ovito.npy', gb_indices)
                self.log('GB ids/indices saved as gb_ids_from_ovito.npy / gb_indices_from_ovito.npy')
            except Exception as e:
                self.log('OVITO GB computation failed: ' + str(e))
                traceback.print_exc()
        else:
            messagebox.showinfo('OVITO not available', 'OVITO interface not detected in the current environment. Please load existing GB ids/indices from files.')

    def on_load_gb_files(self):
        # allow selecting ids (txt/npy) and indices (npy)
        file = filedialog.askopenfilename(title='Select GB ids or GB indices file (.npy/.txt)', filetypes=[('NumPy files','*.npy'),('Text files','*.txt'),('All','*.*')])
        if not file:
            return
        p = Path(file)
        try:
            import numpy as np
            if p.suffix == '.npy':
                arr = np.load(p)
            else:
                arr = np.loadtxt(p, dtype=np.int64)
            # Guess whether ids or indices by values (ids often large, indices 0..N-1)
            if arr.ndim == 0:
                arr = arr.reshape(1,)
            if arr.dtype.kind in ('i', 'u'):
                if arr.max() > 100000 or arr.max() > 10000:
                    # likely ids
                    self.gb_ids = arr.astype(np.int64)
                    self.log(f'Loaded GB ids ({p.name}), {arr.size} entries')
                else:
                    # ambiguous; ask user
                    ans = messagebox.askyesno('Type detection', f'Selected file {p.name} has maximum value {int(arr.max())}. Treat it as indices? (Yes=indices, No=ids)')
                    if ans:
                        self.gb_indices = arr.astype(np.int64)
                        self.log(f'Loaded GB indices ({p.name}), {arr.size} entries')
                    else:
                        self.gb_ids = arr.astype(np.int64)
                        self.log(f'Loaded GB ids ({p.name}), {arr.size} entries')
            else:
                self.log('Could not recognize integer content in the selected file')
        except Exception as e:
            self.log('Failed to load GB file: ' + str(e))
            traceback.print_exc()

    def on_compute_soap_and_train(self):
        # requires energy file and dump
        energy = self.ent_energy.get().strip()
        dump = self.ent_dump.get().strip()
        if not energy:
            messagebox.showerror('Parameter error', 'Please specify an energy file')
            return
        energy_path = Path(energy)
        if not energy_path.exists():
            messagebox.showerror('File error', f'Energy file not found: {energy_path}')
            return
        if self.gb_ids is None and self.gb_indices is None:
            messagebox.showerror('Missing GB', 'Please extract or load GB ids/indices first')
            return
        # load energy map
        try:
            ids_all, vals_all = load_energy_map(energy_path)
        except Exception as e:
            self.log('Failed to read energy file: ' + str(e))
            traceback.print_exc()
            return
        # Map gb_ids/indices
        gb_ids = self.gb_ids
        gb_indices = self.gb_indices
        # if we only have ids and dump provided, try to map
        try:
            if gb_indices is None and gb_ids is not None and dump:
                mapped = map_ids_to_indices_from_dump(Path(dump), gb_ids)
                if mapped is not None:
                    gb_indices = mapped
                    self.log('Mapped gb_ids -> gb_indices using dump')
            # if only indices but no ids, try to extract ids from dump
            if gb_ids is None and gb_indices is not None and dump:
                try:
                    from ase.io import read as ase_read
                    system = ase_read(str(dump), format='lammps-dump-text')
                    arrays = getattr(system, 'arrays', {})
                    found = None
                    for k in ['id', 'atom_id', 'atomIDs', 'atom_ids']:
                        if k in arrays:
                            found = np.array(arrays[k], dtype=np.int64)
                            break
                    if found is not None:
                        gb_ids = found[gb_indices]
                        self.log('Generated gb_ids from indices using dump')
                except Exception:
                    self.log('Failed to extract ids from dump')
        except Exception as e:
            self.log('Attempt to map ids/indices from dump failed: ' + str(e))
            traceback.print_exc()

        # compute target y (reuse code from generate_ml_with_soap compute_y_from_map logic)
        try:
            # local implementation to avoid importing private symbols
            id_to_val = {int(i): float(v) for i, v in zip(ids_all, vals_all)}
            gb_ids_int = [int(x) for x in (gb_ids if gb_ids is not None else (ids_all[gb_indices]))]
            gb_vals = []
            missing = []
            for gid in gb_ids_int:
                if gid in id_to_val:
                    gb_vals.append(id_to_val[gid])
                else:
                    missing.append(gid)
                    gb_vals.append(float('nan'))
            if missing:
                self.log('Warning: the following GB ids were not found in the energy file: ' + str(missing[:10]) + ('...' if len(missing)>10 else ''))
            non_gb_vals = [v for i, v in zip(ids_all, vals_all) if int(i) not in set(gb_ids_int)]
            yb = float(np.mean(non_gb_vals)) if len(non_gb_vals)>0 else 0.0
            y = (np.array(gb_vals, dtype=float) - yb) * 96.485
            self.y = y
            # Added: compute skew fit and save results
            try:
                skew_info = compute_skew_from_y(y)
                # ensure output directory exists
                out_dir = self.ml_out_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / 'skew_info.json').write_text(json.dumps(skew_info, indent=2))
                np.save(out_dir / 'y_gb.npy', y)
                self.log(f'Saved skew fit info to {out_dir / "skew_info.json"} and y array to {out_dir / "y_gb.npy"}')
                # try to plot (if helper available)
                if plot_skew_distribution is not None:
                    try:
                        plot_skew_distribution(y, skew_info.get('fit_params'), out_dir / 'skew_plot.png', bins=40, show=False)
                        self.log(f'Generated skew distribution plot: {out_dir / "skew_plot.png"}')
                    except Exception as e:
                        self.log('Failed to generate skew distribution plot: ' + str(e))
            except Exception as e:
                self.log('Skew fitting failed: ' + str(e))
        except Exception as e:
            self.log('Failed to compute y: ' + str(e))
            traceback.print_exc()
            return

        # Sort dump file lines before processing (if not already sorted)
        dump_path = Path(dump)
        if dump_path.exists():
            try:
                self.log(f'Sorting dump file: {dump_path}')
                sorted_dump_path = self._sort_dump_file(dump_path, header_lines=9)
                if sorted_dump_path != dump_path:
                    self.log(f'Dump file sorted and saved as: {sorted_dump_path}')
                    dump_path = sorted_dump_path
                else:
                    self.log('Dump file sorting skipped or failed')
            except Exception as e:
                self.log('Error sorting dump file: ' + str(e))
                traceback.print_exc()

        # compute SOAP features (may be heavy). We run in background thread.
        def _do_soap_and_train():
            try:
                self.log('Starting SOAP computation (requires ASE + dscribe); otherwise provide precomputed features')
                X = None
                try:
                    if dump:
                        X = compute_soap_features(Path(dump))
                    else:
                        self.log('No dump file provided; cannot compute SOAP (dump required)')
                except RuntimeError as e:
                    self.log('SOAP computation failed (missing deps or other error): ' + str(e))
                    X = None
                if X is None:
                    self.log('SOAP features not computed; please provide precomputed features.npy or install dependencies')
                    return
                self.soap_X = X
                self.log(f'SOAP computation complete, shape {X.shape}')

                # align X to gb_indices (if SOAP rows are in atom order)
                if gb_indices is not None:
                    X_gb = X[gb_indices]
                else:
                    # assume gb_ids mapping to indices done earlier
                    self.log('gb_indices not available; failed to infer indices from gb_ids')
                    X_gb = X

                # train models
                out_dir = self.ml_out_dir
                self.log('Starting training and saving to ' + str(out_dir))
                results, trained, X_test, y_test, y_preds = train_and_save_models(X_gb, self.y, out_dir)
                self.log('Training complete, results: ' + str(results))
                # generate plots
                try:
                    # Parity plots for each trained model
                    for name, pred in y_preds.items():
                        _plot_parity(y_test, pred, out_dir / f'parity_{name}.png', title=name)
                    # Feature importance for any model that exposes feature_importances_
                    for mname, model in trained.items():
                        try:
                            if hasattr(model, 'feature_importances_'):
                                _plot_feature_importance(model, None, out_dir / f'feature_importance_{mname}.png')
                        except Exception:
                            pass
                    self.log('Generated visualization images saved to ' + str(out_dir))
                except Exception as e:
                    self.log('Error generating images: ' + str(e))
                    traceback.print_exc()
            except Exception as e:
                self.log('SOAP/training workflow failed: ' + str(e))
                traceback.print_exc()

        threading.Thread(target=_do_soap_and_train, daemon=True).start()

    def on_show_parity(self):
        p = filedialog.askopenfilename(title='Select parity image (parity_*.png)', filetypes=[('PNG','*.png'),('All','*.*')])
        if not p:
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(p)
            w, h = img.size
            win = tk.Toplevel(self.root)
            win.title('Parity')
            canvas = tk.Canvas(win, width=w, height=h)
            canvas.pack()
            tkimg = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor='nw', image=tkimg)
            # keep reference
            canvas.image = tkimg
        except Exception as e:
            messagebox.showerror('Display failed', f'Unable to display image: {e}')

    def on_show_fi(self):
        p = filedialog.askopenfilename(title='Select feature importance image', filetypes=[('PNG','*.png'),('All','*.*')])
        if not p:
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(p)
            w, h = img.size
            win = tk.Toplevel(self.root)
            win.title('Feature importance')
            canvas = tk.Canvas(win, width=w, height=h)
            canvas.pack()
            tkimg = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor='nw', image=tkimg)
            canvas.image = tkimg
        except Exception as e:
            messagebox.showerror('Display failed', f'Unable to display image: {e}')

    def on_show_skew(self):
        # Prefer the default generated location, but allow user to pick a file
        default_path = Path(self.ml_out_dir) / 'skew_plot.png'
        if default_path.exists():
            p = str(default_path)
        else:
            p = filedialog.askopenfilename(title='Select skew distribution image (skew_plot.png)', filetypes=[('PNG','*.png'),('All','*.*')])
            if not p:
                return
        try:
            from PIL import Image, ImageTk
            img = Image.open(p)
            w, h = img.size
            win = tk.Toplevel(self.root)
            win.title('Skew distribution')
            canvas = tk.Canvas(win, width=w, height=h)
            canvas.pack()
            tkimg = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor='nw', image=tkimg)
            canvas.image = tkimg
        except Exception as e:
            messagebox.showerror('Display failed', f'Unable to display image: {e}')

    def on_run_atomsk(self):
        """Run atomsk in background to convert a polycrystal file to polycrystal.lmp."""
        # Determine polycrystal file path
        if self.last_poly_path is not None and Path(self.last_poly_path).exists():
            poly_path = Path(self.last_poly_path)
        else:
            p = filedialog.askopenfilename(title='Select polycrystal.txt file', filetypes=[('Text files','*.txt'),('All','*.*')])
            if not p:
                self.log('No polycrystal.txt selected; canceling atomsk')
                return
            poly_path = Path(p)

        # Select XSF file
        xsf = filedialog.askopenfilename(title='Select aluminium.xsf or XSF file', filetypes=[('XSF files','*.xsf'),('All','*.*')])
        if not xsf:
            self.log('No XSF selected; canceling atomsk')
            return
        xsf_path = Path(xsf)

        def _run():
            self.log(f'Calling atomsk to convert {poly_path.name} with {xsf_path.name} to polycrystal.lmp (cwd={poly_path.parent})')
            # Try to parse an atomsk command from the polycrystal file (look for embedded examples)
            cmd = None
            try:
                txt = poly_path.read_text(encoding='utf-8')
                for line in txt.splitlines():
                    if not line:
                        continue
                    # support atomsk commands embedded in comment lines (starting with '#') or plain text
                    candidate = line.strip()
                    if candidate.startswith('#'):
                        candidate = candidate.lstrip('#').strip()
                    if not candidate:
                        continue
                    if candidate.startswith('atomsk') or '--polycrystal' in candidate:
                        try:
                            import shlex
                            parsed = shlex.split(candidate)
                            # if parsed result contains 'atomsk', use it
                            if any(p == 'atomsk' for p in parsed):
                                # if no .lmp output specified, append a default
                                has_lmp = any(str(p).lower().endswith('.lmp') for p in parsed)
                                if not has_lmp:
                                    parsed = parsed + ['polycrystal.lmp']
                                    if '-wrap' not in parsed:
                                        parsed = parsed + ['-wrap']
                                cmd = parsed
                                break
                        except Exception:
                            # parsing failed for this line; continue searching
                            continue
            except Exception:
                cmd = None

            # Fallback to a basic default atomsk command template if nothing was parsed
            if cmd is None:
                cmd = ['atomsk', '--polycrystal', str(xsf_path), str(poly_path), 'polycrystal.lmp', '-wrap']
                self.log('No atomsk command found in polycrystal file; using default: ' + ' '.join(cmd))
            else:
                self.log('Parsed atomsk command from polycrystal file: ' + ' '.join(cmd))

            try:
                pre_existing = set(poly_path.parent.iterdir())
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(poly_path.parent))
            except FileNotFoundError:
                self.log('Error: atomsk not found. Please ensure atomsk is installed and available in PATH')
                def _show_not_found(*args: object) -> None:
                    messagebox.showerror('Execution failed', 'atomsk not found. Ensure atomsk is installed and on your PATH.')
                self.root.after(0, _show_not_found)  # type: ignore[arg-type]
                return
            except Exception as e:
                self.log('Exception while running atomsk: ' + str(e))
                traceback.print_exc()
                return

            self.log('--- atomsk stdout ---')
            if proc.stdout:
                for line in proc.stdout.splitlines():
                    self.log(line)
            self.log('--- atomsk stderr ---')
            if proc.stderr:
                for line in proc.stderr.splitlines():
                    self.log(line)

            if proc.returncode != 0:
                self.log(f'atomsk returned non-zero exit code: {proc.returncode}')
                def _show_failed(*args: object) -> None:
                    messagebox.showerror('atomsk failed', f'atomsk returned non-zero exit code: {proc.returncode}\nPlease check logs for details.')
                self.root.after(0, _show_failed)  # type: ignore[arg-type]
                return

            # On success, inspect and record the output .lmp file path using the command we ran
            out_lmp = None
            for t in cmd[::-1]:
                try:
                    if str(t).lower().endswith('.lmp'):
                        out_lmp = poly_path.parent / Path(t).name
                        break
                except Exception:
                    continue
            if out_lmp is None:
                out_lmp = poly_path.parent / 'polycrystal.lmp'

            if out_lmp.exists():
                self.last_lmp_path = out_lmp
                self.log(f'atomsk completed successfully, generated: {out_lmp.resolve()}')
                def _on_success(*args: object) -> None:
                    messagebox.showinfo('atomsk complete', f'Generated: {out_lmp}')
                    # Auto-fill read_data field with the new lmp filename
                    try:
                        self.ent_read_data.delete(0, tk.END)
                        self.ent_read_data.insert(0, f'{out_lmp.name} extra/atom/types 1')
                    except Exception:
                        pass
                self.root.after(0, _on_success)  # type: ignore[arg-type]
                # identify and optionally cleanup extra files created by atomsk
                try:
                    post_existing = set(poly_path.parent.iterdir())
                    created = post_existing - pre_existing
                    to_delete = [p for p in created if p.name != out_lmp.name]
                    try:
                        if poly_path.exists() and poly_path not in to_delete:
                            to_delete.append(poly_path)
                    except Exception:
                        pass
                    if to_delete:
                        preview = '\n'.join([p.name for p in sorted(to_delete)])
                        confirm = messagebox.askyesno('Confirm deletion', f'The following files/folders were created by atomsk (polycrystal.lmp will be kept):\n\n{preview}\n\nDelete these files and polycrystal.txt?')
                        if confirm:
                            deleted = 0
                            for p in to_delete:
                                try:
                                    if p.is_dir():
                                        shutil.rmtree(p)
                                    else:
                                        p.unlink()
                                    deleted += 1
                                    self.log(f'Deleted: {p.name}')
                                except Exception as e:
                                    self.log(f'Failed to delete {p}: {e}')
                            messagebox.showinfo('Cleanup complete', f'Deleted {deleted} items (polycrystal.lmp kept)')
                        else:
                            self.log('User canceled automatic deletion of atomsk-created files')
                except Exception as e:
                    self.log(f'Error while attempting to identify/delete atomsk-created files: {e}')
            else:
                self.log('atomsk did not produce expected .lmp output; please check logs')

        threading.Thread(target=_run, daemon=True).start()

    def _sort_dump_file(self, dump_path: Path, header_lines: int = 9) -> Path:
        """Sort atom lines in a dump file by the first column, using the user-provided approach.

        This function:
        - Makes a backup of the original file (filename + '.bak')
        - Reads the file (tries utf-8, falls back to latin-1)
        - Skips the first `header_lines`, then treats the remainder starting at line header_lines as data
        - Attempts to parse the first token on each data line as a float and sort by it
        - If parsing fails for any data line, no change is made and the original path is returned
        - Writes the header + sorted data back to the original file and returns the same Path
        """
        dump_path = Path(dump_path)
        if not dump_path.exists():
            self.log(f'Dump file not found: {dump_path}')
            return dump_path

        # Read with utf-8, fallback to latin-1 if decoding fails
        try:
            text = dump_path.read_text(encoding='utf-8')
            lines = text.splitlines(keepends=True)
        except UnicodeDecodeError:
            try:
                text = dump_path.read_text(encoding='latin-1')
                lines = text.splitlines(keepends=True)
            except Exception as e:
                self.log(f'Failed to read dump file {dump_path}: {e}')
                return dump_path

        if len(lines) <= header_lines:
            self.log(f'Dump file has <= {header_lines} lines; skipping sort')
            return dump_path

        header = lines[:header_lines]
        data_lines = lines[header_lines:]

        # Attempt to parse the first token of each data line as a float
        parsed = []
        for ln in data_lines:
            s = ln.strip()
            if not s:
                # keep empty lines (but they will sort to front if key 0) - instead, abort
                self.log('Found empty line in data region; skipping sort')
                return dump_path
            parts = s.split()
            try:
                key = float(parts[0])
            except Exception:
                self.log(f'Unable to parse numeric key from line: {s[:80]}...; skipping sort')
                return dump_path
            parsed.append((key, ln))

        # Sort by key
        parsed.sort(key=lambda t: t[0])
        sorted_data_lines = [t[1] for t in parsed]

        # Backup original file
        backup_path = dump_path.with_name(dump_path.name + '.bak')
        try:
            shutil.copy2(dump_path, backup_path)
            self.log(f'Backup created: {backup_path.name}')
        except Exception as e:
            self.log(f'Failed to create backup of dump file: {e}; aborting sort')
            return dump_path

        # Write header + sorted data back to file
        try:
            with open(dump_path, 'w', encoding='utf-8') as out:
                out.writelines(header)
                out.writelines(sorted_data_lines)
            self.log(f'Sorted dump file written: {dump_path} (backup at {backup_path.name})')
        except Exception as e:
            self.log(f'Failed to write sorted dump file: {e}')
            # Attempt to restore backup
            try:
                shutil.copy2(backup_path, dump_path)
                self.log('Original dump file restored from backup')
            except Exception:
                self.log('Failed to restore original dump from backup')
        return dump_path

def main() -> None:
    """Start the workflow GUI.

    This function creates the Tk root, instantiates the WorkflowApp and runs the
    Tk main loop. It is safe to import this module without starting the GUI.
    """
    # If tkinter was not available, the module already exits earlier; this is
    # a defensive check to avoid raising here.
    if tk is None:
        print('tkinter is not available; cannot start GUI.')
        return

    root = tk.Tk()
    app = WorkflowApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        # allow graceful termination with Ctrl+C when run from a terminal
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == '__main__':
    main()
