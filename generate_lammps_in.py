#!/usr/bin/env python3
"""
generate_lammps_in.py

Generate a LAMMPS input file (an in file). Provides the function interface
`generate_lammps_in(...)` and a simple Tkinter GUI (visual window) for
interactive input.

Users must specify three required items:
1) The lmp file name after read_data (and optional extra args, e.g. `extra/atom/types 1`)
2) The txt file name after print append
3) The eam file used in the pair_coeff line (e.g. AlCu.eam) and optional species names

This script does not execute LAMMPS; it only generates a text file and shows
suggested run commands in the GUI.
"""

from pathlib import Path
from typing import Sequence, Union, Tuple
import subprocess
import threading

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext
except Exception:
    tk = None

# add: import sorter utility
try:
    from dump_utils import sort_dump_file
except Exception:
    sort_dump_file = None


DEFAULT_TEMPLATE_IN_TUI = '''units           metal
atom_style      atomic
boundary        p p p
read_data       {read_data_initial}
mass 1 {mass1}
pair_style  eam/alloy
pair_coeff * * {eam_file} {species_first_only}
timestep {timestep}
thermo {thermo}
thermo_style        {thermo_style}
variable T_anneal equal 373.2
variable T_cool_start equal ${{T_anneal}}
variable T_cool_end equal 0.1
variable CoolingRate equal 3
fix 1 all npt temp ${{T_anneal}} ${{T_anneal}} 0.1 iso 0 0 1.0
dump 1 all custom 1000 dump_anneal.lammpstrj id type x y z
run 250000
unfix 1
fix 2 all npt temp ${{T_cool_start}} ${{T_cool_end}} 2.0 iso 0 0 1.0
dump 2 all custom 1000 dump_cool.lammpstrj id type x y z
run 124360
unfix 2
undump 1
undump 2
min_style cg
min_modify dmax 0.1
minimize 0 1.0e-12 10000 100000
write_dump all atom dump.atom
write_data {data_intermediate}
#mpiexec -np 4 lmp -in in_tui -pk omp 4 -sf omp
'''

DEFAULT_TEMPLATE_IN = '''units           metal
atom_style      atomic
boundary        p p p
read_data       {data_intermediate} extra/atom/types 1

mass 1 {mass1}
mass 2 {mass2}
pair_style  eam/alloy
pair_coeff * * {eam_file} {species_line}

timestep {timestep}
thermo {thermo}
thermo_style        custom step pe

# Initial minimization -> baseline
minimize 0 1e-12 10000 10000

# Write baseline restart for later reuse
write_restart baseline.restart
write_data baseline.data

run 0


# Save total atoms and build loop (in most LAMMPS versions these variables/loop controls persist after clear)
variable total_atoms equal count(all)
variable atom_id loop ${{total_atoms}}

# ------------------ Loop body: restore baseline -> replace -> relax -> record ------------------
label loop_start

# Clear current system so read_restart can define the box
clear

units           metal
atom_style      atomic
boundary        p p p


read_restart baseline.restart

mass 1 {mass1}
mass 2 {mass2}
pair_style  eam/alloy
pair_coeff * * {eam_file} {species_line}

timestep {timestep}
thermo {thermo}
thermo_style        custom step pe
variable totalenergy equal pe

set atom ${{atom_id}} type 2

minimize 0 1e-12 10000 10000
run 0

print "${{atom_id}} ${{totalenergy}}" append {append_txt}

next atom_id
jump SELF loop_start
run 0
print "Done!"
#mpiexec -np 4 lmp -in in -pk omp 4 -sf omp
'''


def generate_lammps_in(read_data: str, append_txt: str, eam_file: str, out_tui: Union[str, Path] = 'in_tui', out_in: Union[str, Path] = 'in', species: Union[Sequence[str], str] = ('Al', 'Cu'), mass1: Union[str,float] = '26.98', mass2: Union[str,float] = '63.55', timestep: float = 0.001, thermo: int = 1000, thermo_style: str = 'custom step pe', read_data_extra: str = '') -> Tuple[Path, Path]:
    """Generate LAMMPS input files (in_tui and in) and return their paths.

    Parameters:
      read_data: arguments after read_data (can include path (e.g. polycrystal.lmp) and extra args), e.g. 'medium_final_atoms.lmp extra/atom/types 1'
                 NOTE: For in_tui, we use the first part (filename) and strip extra args if they look like 'extra/atom/types'.
                 But based on usage, in_tui typically just reads the geometry file.
      append_txt: filename after print append, e.g. 'medium_Al_Cu_eam4.txt'
      eam_file: EAM file in the pair_coeff line, e.g. 'AlCu.eam'
      out_tui: output path for the annealing script (default 'in_tui')
      out_in: output path for the segregation script (default 'in')
      species: species list or space-separated string (default ('Al','Cu'))
      mass1/mass2: element masses
      timestep, thermo, thermo_style: LAMMPS parameters
      read_data_extra: optional extra read_data args

    Returns: (path_to_in_tui, path_to_in)
    """
    # Build the species line.
    if isinstance(species, (list, tuple)):
        species_list = list(map(str, species))
        species_line = ' '.join(species_list)
        # For in_tui, usually only the major species is relevant if we are just annealing a pure structure or if the pot file needs it?
        # In the provided example "in_tui": pair_coeff * * AlCu.eam Al
        # It seems it treats everything as type 1 (Al) initially?
        # Let's assume the first species is the primary one for in_tui if multiple are provided,
        # OR if the user provides the list, maybe we should check if they want 1 or all.
        # However, in_tui provided has "pair_coeff * * AlCu.eam Al".
        # Let's use the first species for in_tui.
        species_first_only = species_list[0] if species_list else 'Al'
    else:
        species_line = str(species)
        species_first_only = species_line.split()[0] if species_line.strip() else 'Al'

    # Parse read_data to get the filename.
    # The user might pass "file.lmp extra/atom/types 1" or just "file.lmp"
    parts = read_data.strip().split()
    input_file_path = parts[0]

    # Construct intermediate data filename (e.g. file.data)
    # If input is 'polycrystal.lmp', intermediate might be 'polycrystal.data'
    # The provided example: read '70+5.lmp', write '70+5.data'
    p = Path(input_file_path)
    data_intermediate = p.with_suffix('.data').name

    # Generte in_tui content
    # in_tui reads the initial file (often without extra/atom/types if it's just a dump turned data, or maybe it needs it?)
    # The provided in_tui has: read_data 70+5.lmp
    # It does NOT have extra/atom/types 1.
    # We will assume for in_tui we just use the filename.

    content_tui = DEFAULT_TEMPLATE_IN_TUI.format(
        read_data_initial=input_file_path,
        mass1=str(mass1),
        mass2=str(mass2),
        eam_file=str(eam_file),
        species_first_only=species_first_only,
        timestep=timestep,
        thermo=thermo,
        thermo_style=thermo_style,
        data_intermediate=data_intermediate
    )

    # Generate in content
    # in reads the intermediate data file.
    # NOTE: The provided in file has 'read_data 70+5.data extra/atom/types 1'
    # So we should append 'extra/atom/types 1' to the intermediate data filename in the read_data line.
    # But wait, read_data_extra parameter might duplicate this if we are not careful.
    # Let's treat read_data_extra as something applied to the INITIAL user input, but here we are generating a secondary step.
    # The user instruction implies the flow: in_tui -> in.
    # So 'in' should read what 'in_tui' writes (data_intermediate).
    # And we add 'extra/atom/types 1' because we are doing segregation (inserting type 2).

    content_in = DEFAULT_TEMPLATE_IN.format(
        data_intermediate=data_intermediate,
        mass1=str(mass1),
        mass2=str(mass2),
        eam_file=str(eam_file),
        species_line=species_line,
        timestep=timestep,
        thermo=thermo,
        append_txt=str(append_txt),
    )

    path_tui = Path(out_tui)
    path_in = Path(out_in)

    path_tui.write_text(content_tui, encoding='utf-8')
    path_in.write_text(content_in, encoding='utf-8')

    return path_tui, path_in


def main_gui():
    """Simple Tkinter GUI for generating an in file interactively."""
    if tk is None:
        print('tkinter is not available in the current Python environment; cannot start GUI.')
        print('Please import and call generate_lammps_in(...) from Python instead.')
        return

    root = tk.Tk()
    root.title('LAMMPS in Generator')

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill=tk.BOTH, expand=True)

    tk.Label(frm, text='read_data (e.g. polycrystal.lmp extra/atom/types 1):').grid(row=0, column=0, sticky='w')
    ent_read = tk.Entry(frm, width=50)
    ent_read.insert(0, 'polycrystal.lmp') # Default changed to just filename, cleaner
    ent_read.grid(row=0, column=1, sticky='w')

    tk.Label(frm, text='append filename (e.g. Al_Cu.txt):').grid(row=1, column=0, sticky='w')
    ent_append = tk.Entry(frm, width=50)
    ent_append.insert(0, 'Al_Cu.txt')
    ent_append.grid(row=1, column=1, sticky='w')

    tk.Label(frm, text='EAM file (e.g. AlCu.eam):').grid(row=2, column=0, sticky='w')
    ent_eam = tk.Entry(frm, width=50)
    ent_eam.insert(0, 'AlCu.eam')
    ent_eam.grid(row=2, column=1, sticky='w')

    tk.Label(frm, text='Species (space-separated, default: Al Cu):').grid(row=3, column=0, sticky='w')
    ent_species = tk.Entry(frm, width=50)
    ent_species.insert(0, 'Al Cu')
    ent_species.grid(row=3, column=1, sticky='w')

    tk.Label(frm, text='Output filename for segregation script (default: in):').grid(row=4, column=0, sticky='w')
    ent_out = tk.Entry(frm, width=50)
    ent_out.insert(0, 'in')
    ent_out.grid(row=4, column=1, sticky='w')

    tk.Label(frm, text='(Will also generate a corresponding *_tui setup script)').grid(row=5, column=0, sticky='w') # informational

    tk.Label(frm, text='Optional: extra read_data args (leave blank if none):').grid(row=6, column=0, sticky='w')
    ent_read_extra = tk.Entry(frm, width=50)
    ent_read_extra.grid(row=6, column=1, sticky='w')

    # New options: auto-run LAMMPS and optionally sync in name with append output name.
    run_lammps_var = tk.BooleanVar(value=False)
    sync_names_var = tk.BooleanVar(value=False)
    chk_run_lammps = tk.Checkbutton(frm, text='Auto-run LAMMPS (in_tui then in)', variable=run_lammps_var)
    chk_run_lammps.grid(row=6, column=2, sticky='w', padx=(10,0))
    chk_sync = tk.Checkbutton(frm, text='Sync -in filename with append output name (append = in basename + .txt)', variable=sync_names_var)
    chk_sync.grid(row=7, column=2, sticky='w', padx=(10,0))

    # Preview/log
    tk.Label(frm, text='Generated content preview:').grid(row=7, column=0, sticky='nw', pady=(10,0))
    txt_preview = scrolledtext.ScrolledText(frm, width=80, height=20)
    txt_preview.grid(row=7, column=1, columnspan=2, pady=(10,0))

    def log(msg: str):
        txt_preview.insert(tk.END, msg + '\n')
        txt_preview.see(tk.END)

    def browse_read():
        p = filedialog.askopenfilename(filetypes=[('LMP files','*.lmp'),('All','*.*')])
        if p:
            ent_read.delete(0, tk.END)
            ent_read.insert(0, p)

    def browse_eam():
        p = filedialog.askopenfilename(filetypes=[('EAM files','*.eam'),('All','*.*')])
        if p:
            ent_eam.delete(0, tk.END)
            ent_eam.insert(0, p)

    btn_browse_read = tk.Button(frm, text='Browse read_data', command=browse_read)
    btn_browse_read.grid(row=0, column=2, padx=5)
    btn_browse_eam = tk.Button(frm, text='Browse EAM', command=browse_eam)
    btn_browse_eam.grid(row=2, column=2, padx=5)

    def on_generate():
        txt_preview.delete('1.0', tk.END)
        rd = ent_read.get().strip()
        ap = ent_append.get().strip()
        eam = ent_eam.get().strip()
        sp = ent_species.get().strip()
        out = ent_out.get().strip()
        rd_extra = ent_read_extra.get().strip()

        if not rd:
            messagebox.showerror('Parameter error', 'read_data cannot be empty')
            return
        if not ap and not sync_names_var.get():
            messagebox.showerror('Parameter error', 'append filename cannot be empty (or enable sync option)')
            return
        if not eam:
            messagebox.showerror('Parameter error', 'EAM file cannot be empty')
            return

        species_list = sp.split()

        # If sync is checked, set append to in basename + .txt
        append_to_use = ap
        if sync_names_var.get():
            try:
                append_to_use = str(Path(out).stem) + '.txt'
                # Update UI
                ent_append.delete(0, tk.END)
                ent_append.insert(0, append_to_use)
            except Exception:
                append_to_use = ap

        # Determine strict filenames based on user request "in" and "in_tui"
        # If user put "in.lmp", we might produce "in_tui.lmp" and "in.lmp"?
        # Or just "in_tui" and "in"?
        # The prompt says: "先运行in_tui，然后运行in" (Run in_tui first, then in).
        # Let's infer the tui name from the out name.
        p_out = Path(out)
        if p_out.name == 'in':
             out_curr_tui = p_out.parent / 'in_tui'
        else:
             # if out is "custom.in", maybe "custom_tui.in"?
             out_curr_tui = p_out.parent / (p_out.stem + '_tui' + p_out.suffix)

        try:
            # Generate both files
            path_tui, path_in = generate_lammps_in(rd, append_to_use, eam, out_tui=str(out_curr_tui), out_in=out, species=species_list, read_data_extra=rd_extra)
        except Exception as e:
            messagebox.showerror('Write failed', f'Error generating files: {e}')
            return

        txt_preview.insert(tk.END, f'--- {path_tui.name} ---\n')
        txt_preview.insert(tk.END, path_tui.read_text(encoding='utf-8'))
        txt_preview.insert(tk.END, f'\n\n--- {path_in.name} ---\n')
        txt_preview.insert(tk.END, path_in.read_text(encoding='utf-8'))

        # Auto-run LAMMPS if enabled
        if run_lammps_var.get():
            dir_for_ops = path_in.parent.resolve()

            # Sequence: tui then in
            # We need to run tui first.
            cmd_tui = ['mpiexec', '-np', '4', 'lmp', '-in', path_tui.name, '-pk', 'omp', '4', '-sf', 'omp']
            cmd_in = ['mpiexec', '-np', '4', 'lmp', '-in', path_in.name, '-pk', 'omp', '4', '-sf', 'omp']

            # run in background thread to avoid blocking GUI
            def _run_lammps():
                def safe_log(s: str):
                    # use after with function+args to avoid lambda-related linter warnings
                    root.after(0, log, s)

                # --- STEP 1: in_tui ---
                safe_log('\nRunning STEP 1 (Annealing/Setup): ' + ' '.join(cmd_tui))
                try:
                    proc_tui = subprocess.run(cmd_tui, capture_output=True, text=True, cwd=str(dir_for_ops))
                except FileNotFoundError:
                    root.after(0, messagebox.showerror, 'Execution failed', 'mpiexec or lmp not found.')
                    safe_log('Error: mpiexec or lmp not found in PATH')
                    return
                except Exception as e:
                    root.after(0, messagebox.showerror, 'Execution failed', f'Error running LAMMPS Step 1: {e}')
                    safe_log(f'LAMMPS Step 1 runtime error: {e}')
                    return

                if proc_tui.returncode != 0:
                    safe_log(f'LAMMPS Step 1 failed with code {proc_tui.returncode}. Aborting Step 2.')
                    safe_log('Last stderr chars: ' + proc_tui.stderr[-500:])
                    root.after(0, messagebox.showerror, 'LAMMPS Step 1 failed', f'Step 1 (in_tui) returned non-zero exit code: {proc_tui.returncode}')
                    return

                safe_log('Step 1 finished successfully.')

                # --- STEP 2: in ---
                safe_log('\nRunning STEP 2 (Segregation Calculation): ' + ' '.join(cmd_in))
                try:
                    proc_in = subprocess.run(cmd_in, capture_output=True, text=True, cwd=str(dir_for_ops))
                except Exception as e:
                    root.after(0, messagebox.showerror, 'Execution failed', f'Error running LAMMPS Step 2: {e}')
                    safe_log(f'LAMMPS Step 2 runtime error: {e}')
                    return

                # Do not stream full LAMMPS stdout/stderr into the GUI preview to avoid
                # flooding the visualization pane. Only report high-level status.
                safe_log('LAMMPS Step 2 finished (see log.lammps in the working directory for full output if needed)')

                if proc_in.returncode != 0:
                    root.after(0, messagebox.showerror, 'LAMMPS Step 2 failed', f'Step 2 (in) returned non-zero exit code: {proc_in.returncode}\nPlease check logs for details.')
                    return

                # Attempt to delete LAMMPS default log file log.lammps
                safe_log('LAMMPS ended; attempting to delete log.lammps (if present)...')
                try:
                    log_path = dir_for_ops / 'log.lammps'
                    if log_path.exists():
                        log_path.unlink()
                        safe_log(f'Deleted log file: {log_path.name}')
                    else:
                        safe_log('log.lammps not found; nothing to delete')
                except Exception as e:
                    safe_log(f'Failed to delete log.lammps: {e}')
                    root.after(0, messagebox.showwarning, 'Log cleanup failed', f'Run completed but could not delete log.lammps: {e}')

                # After successful run, attempt to sort dump.atom (if present).
                try:
                    dump_path = dir_for_ops / 'dump.atom'
                    if dump_path.exists() and sort_dump_file is not None:
                        # call sorter without a GUI logger to avoid streaming logs into preview
                        sort_dump_file(dump_path, header_lines=9, logger=None)
                        root.after(0, messagebox.showinfo, 'LAMMPS complete', 'LAMMPS workflow finished; dump.atom (if present) was auto-sorted.')
                    else:
                        root.after(0, messagebox.showinfo, 'LAMMPS complete', 'LAMMPS workflow finished.')
                except Exception as e:
                    root.after(0, messagebox.showwarning, 'Sort failed', f'Auto-sort of dump.atom failed: {e}')

            threading.Thread(target=_run_lammps, daemon=True).start()

        messagebox.showinfo('Generated', f'Generated files:\n{path_tui}\n{path_in}\n\nRun in_tui first, then in.')

    btn_generate = tk.Button(frm, text='Generate files', command=on_generate, bg='#2196f3', fg='white')
    btn_generate.grid(row=8, column=1, pady=10, sticky='w')

    btn_quit = tk.Button(frm, text='Exit', command=root.destroy)
    btn_quit.grid(row=8, column=2, pady=10, sticky='e')

    root.mainloop()


if __name__ == '__main__':
    main_gui()
