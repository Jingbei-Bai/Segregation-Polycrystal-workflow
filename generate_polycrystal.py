#!/usr/bin/env python3
"""
generate_polycrystal.py

generate_polycrystal(box, random_val, out='polycrystal.txt', ...)
"""

from pathlib import Path
from typing import Sequence, Union
import subprocess
import shutil
import re


def _normalize_box_input(box: Union[Sequence, int, float, str]):
    """Accepts a single scalar (int/float/str) or sequence (length 1 or 3), returns three strings representing box size."""
    # If scalar (not sequence or is string/number), treat as three identical values
    if isinstance(box, (int, float)):
        s = str(box)
        return [s, s, s]
    if isinstance(box, str):
        # Allow strings like "70" or "70 80 90"
        parts = box.split()
        if len(parts) == 1:
            return [parts[0], parts[0], parts[0]]
        if len(parts) == 3:
            return parts
        raise ValueError('String format for box must be one value or three space-separated values, e.g. "70" or "70 80 90"')
    # If sequence (list/tuple)
    try:
        seq = list(box)
    except TypeError:
        raise ValueError('box must be a number/string or sequence')

    if len(seq) == 1:
        val = seq[0]
        return [str(val), str(val), str(val)]
    if len(seq) == 3:
        return [str(seq[0]), str(seq[1]), str(seq[2])]
    raise ValueError('box parameter must be 1 value or 3 values, e.g.: 70 or [70,70,70]')


def write_polycrystal(path: Path, box_vals, random_val: int):
    # Write polycrystal.txt (overwrite)
    lines = []
    lines.append('box {} {} {}'.format(box_vals[0], box_vals[1], box_vals[2]))
    lines.append('random {}'.format(random_val))
    content = "\n".join(lines) + "\n"
    # Ensure parent directory exists
    parent = path.parent
    if not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise OSError(f'Cannot create parent directory {parent}: {e}')

    # If target path exists and is a directory, give explicit error (Windows Errno 22)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f'Target path is a directory, cannot write file: {path}. Please choose another name or delete the folder.')

    try:
        path.write_text(content, encoding='utf-8')
    except OSError as e:
        # Wrap error with context
        raise OSError(f'Failed to write file {path}: {e}')


def _sanitize_filename(name: str) -> str:
    """Simple filename sanitization, removing illegal characters on Windows/UNIX and avoiding reserved names.

    Returns a safe filename (no path)."""
    if not isinstance(name, str) or name == '':
        return 'polycrystal.txt'

    # Keep only basename
    base = Path(name).name
    # Remove control characters
    base = re.sub(r'[\x00-\x1f\x7f]', '', base)
    # Remove illegal characters: <>:"/\\|?*
    base = re.sub(r'[<>:"/\\|?*]', '_', base)
    # Handle Windows reserved names (CON, PRN, AUX, NUL, COM1..COM9, LPT1..LPT9)
    reserved = { 'CON','PRN','AUX','NUL' } | {f'COM{i}' for i in range(1,10)} | {f'LPT{i}' for i in range(1,10)}
    stem = Path(base).stem.upper()
    if stem in reserved:
        base = f'_{base}'
    # If empty after cleanup, use default
    if base == '' or base in ('.', '..'):
        base = 'polycrystal.lmp'
    return base


def generate_polycrystal(box: Union[Sequence, int, float, str], random_val: int, out: Union[str, Path] = 'polycrystal.txt', xsf_path: Union[str, Path] = 'aluminium.xsf', show_preview: bool = True, print_command: bool = True):
    """Functional interface to generate polycrystal file.

    Parameters:
      box: single value, string, or sequence of length 1/3; e.g. 70 or '70 80 90' or [70] or [70,80,90]
      random_val: integer after random
      out: output file path (default 'polycrystal.txt')
      xsf_path: path to aluminium.xsf for suggestions (default 'aluminium.xsf')
      show_preview: whether to print file content after generation
      print_command: whether to print suggested atomsk command

    Returns: Path object of the output file
    """
    box_vals = _normalize_box_input(box)
    # Clean and normalize output path
    out_p = Path(out)
    try:
        parent = out_p.parent
    except Exception:
        parent = Path('.')
    safe_name = _sanitize_filename(out_p.name)
    out_path = parent / safe_name
    if str(out_path) != str(out_p):
        # Notify user of filename correction
        if show_preview:
            print(f"Note: Output filename sanitized to: {out_path.name}")

    try:
        write_polycrystal(out_path, box_vals, int(random_val))
    except Exception as e:
        # Fallback to temp file on failure
        import time, tempfile
        fallback_name = f'polycrystal_{int(time.time())}.txt'
        fallback_path = Path('.') / fallback_name
        try:
            write_polycrystal(fallback_path, box_vals, int(random_val))
        except Exception as e2:
            # If fallback fails, raise original error
            raise OSError(f'Failed to write {out_path} ({e}), and fallback {fallback_path} also failed ({e2}).')
        else:
            if show_preview:
                print(f'Note: Could not write to {out_path}, fell back to: {fallback_path.resolve()}')
            out_path = fallback_path

    if show_preview:
        print('Generated: {}'.format(out_path.resolve()))
        print('\nContent preview:')
        print(out_path.read_text(encoding='utf-8'))

    xsf_p = Path(xsf_path)
    if not xsf_p.exists():
        print('Hint: aluminium.xsf ({}) not found. Ensure correct path or place in same directory when running atomsk.'.format(xsf_p.resolve()))

    if print_command:
        print('\nNext step: Run the following command in a terminal with atomsk installed to generate the polycrystal:')
        print('atomsk --polycrystal {} {} polycrystal.lmp'.format(xsf_p.name, out_path.name))

    return out_path


# Replace original __main__ prompt with GUI (using tkinter)
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog, scrolledtext
except Exception:
    tk = None


def _confirm_and_delete_all_except(directory: Path, keep_name: str, log_fn=None):
    """Delete all files/folders in directory except the one named keep_name.

    Returns list of deleted paths.
    log_fn can be a function for logging.
    """
    deleted = []
    for child in directory.iterdir():
        if child.name == keep_name:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            deleted.append(child)
            if log_fn:
                log_fn(f'Deleted: {child}')
        except Exception as e:
            if log_fn:
                log_fn(f'Failed to delete {child}: {e}')
    return deleted


def main_gui():
    """Start a simple Tkinter GUI for inputing box, random, and file path to generate polycrystal file.

    Features: Optionally run atomsk after generation, and cleanup files other than polycrystal.lmp.
    Prints console hints if tkinter is unavailable.
    """
    if tk is None:
        print('tkinter not available in current Python environment, GUI cannot start.')
        print('Please import and call generate_polycrystal(box, random_val, out, xsf_path) directly.')
        return

    root = tk.Tk()
    root.title('Polycrystal Generator')

    frm = tk.Frame(root, padx=10, pady=10)
    frm.pack(fill=tk.BOTH, expand=True)


    tk.Label(frm, text='Box (single value or three values, e.g. 70 or 70 80 90):').grid(row=0, column=0, sticky='w')
    ent_box = tk.Entry(frm, width=30)
    ent_box.insert(0, '70')
    ent_box.grid(row=0, column=1, sticky='w')

    tk.Label(frm, text='Random (integer):').grid(row=1, column=0, sticky='w')
    ent_random = tk.Entry(frm, width=30)
    ent_random.insert(0, '20')
    ent_random.grid(row=1, column=1, sticky='w')

    tk.Label(frm, text='Output file (polycrystal.lmp):').grid(row=2, column=0, sticky='w')
    ent_out = tk.Entry(frm, width=30)
    ent_out.insert(0, 'polycrystal.lmp')
    ent_out.grid(row=2, column=1, sticky='w')

    def browse_out():
        p = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text files','*.txt'),('All','*.*')], initialfile=ent_out.get())
        if p:
            ent_out.delete(0, tk.END)
            ent_out.insert(0, p)

    btn_browse_out = tk.Button(frm, text='Browse', command=browse_out)
    btn_browse_out.grid(row=2, column=2, padx=5)

    tk.Label(frm, text='aluminium.xsf path:').grid(row=3, column=0, sticky='w')
    ent_xsf = tk.Entry(frm, width=30)
    ent_xsf.insert(0, 'aluminium.xsf')
    ent_xsf.grid(row=3, column=1, sticky='w')

    def browse_xsf():
        p = filedialog.askopenfilename(filetypes=[('XSF files','*.xsf'),('All','*.*')])
        if p:
            ent_xsf.delete(0, tk.END)
            ent_xsf.insert(0, p)

    btn_browse_xsf = tk.Button(frm, text='Browse', command=browse_xsf)
    btn_browse_xsf.grid(row=3, column=2, padx=5)

    # Atomsk and cleanup options
    run_atomsk_var = tk.BooleanVar(value=False)
    clean_var = tk.BooleanVar(value=False)
    chk_run = tk.Checkbutton(frm, text='Auto-run atomsk after generation', variable=run_atomsk_var)
    chk_run.grid(row=6, column=1, sticky='w')
    chk_clean = tk.Checkbutton(frm, text='Delete all files except polycrystal.lmp after generation (confirm required)', variable=clean_var)
    chk_clean.grid(row=7, column=1, sticky='w')

    # Log/Preview
    tk.Label(frm, text='Log / Preview:').grid(row=4, column=0, sticky='nw', pady=(10,0))
    txt_preview = scrolledtext.ScrolledText(frm, width=80, height=12)
    txt_preview.grid(row=4, column=1, columnspan=2, pady=(10,0))

    def log(msg: str):
        txt_preview.insert(tk.END, msg + '\n')
        txt_preview.see(tk.END)

    def on_generate():
        txt_preview.delete('1.0', tk.END)
        box_str = ent_box.get().strip()
        rnd_str = ent_random.get().strip()
        out_path = ent_out.get().strip()
        xsf_path = ent_xsf.get().strip()

        # Parse and validate
        try:
            box_vals = _normalize_box_input(box_str)
        except Exception as e:
            messagebox.showerror('Parameter Error', f'Box parameter parsing failed: {e}')
            return

        try:
            rnd = int(rnd_str)
        except Exception:
            messagebox.showerror('Parameter Error', 'Random must be an integer')
            return

        try:
            outp = generate_polycrystal(box_vals, rnd, out=out_path, xsf_path=xsf_path, show_preview=False, print_command=False)
        except Exception as e:
            messagebox.showerror('Write Failed', f'Error generating file: {e}')
            return

        # Show content
        try:
            content = outp.read_text(encoding='utf-8')
            log('=== Generated Content ===')
            log(content.strip())
        except Exception as e:
            log(f'Cannot read generated file: {e}')

        # If run atomsk checked
        dir_for_ops = Path(outp).parent.resolve()
        atomsk_ran = False
        pre_existing = set(dir_for_ops.iterdir())
        created = set()
        if run_atomsk_var.get():
            cmd = ['atomsk', '--polycrystal', xsf_path, str(outp), 'polycrystal.lmp', '-wrap']
            log('Running atomsk: ' + ' '.join(cmd))
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(dir_for_ops))
            except FileNotFoundError:
                messagebox.showerror('Execution Failed', 'atomsk not found. Ensure atomsk is installed and in PATH.')
                log('Error: atomsk not found in PATH')
                return
            except Exception as e:
                messagebox.showerror('Execution Failed', f'Error running atomsk: {e}')
                log(f'atomsk runtime exception: {e}')
                return

            log('--- atomsk stdout ---')
            log(proc.stdout.strip())
            log('--- atomsk stderr ---')
            log(proc.stderr.strip())

            if proc.returncode != 0:
                messagebox.showerror('atomsk Failed', f'atomsk returned non-zero exit code: {proc.returncode}\nPlease check logs.')
                return
            atomsk_ran = True
            log('atomsk executed successfully, generated polycrystal.lmp (if supported)')
            post_existing = set(dir_for_ops.iterdir())
            created = post_existing - pre_existing
            # Show created files
            if created:
                log('atomsk created the following files/folders:')
                for p in sorted(created):
                    log(f'  {p.name}')
            else:
                log('atomsk did not create new files this run')

        # If cleanup checked
        if clean_var.get():
            # Cleanup logic:
            # - If atomsk ran, delete files created by atomsk (except polycrystal.lmp) AND the polycrystal.txt used
            # - If atomsk did not run, ask to delete only polycrystal.txt
            if atomsk_ran:
                created_list = list(created)
                # List targets to delete (exclude .lmp)
                to_delete = [p for p in created_list if p.name != 'polycrystal.lmp']
                # Ensure polycrystal.txt is also deleted
                if Path(outp).exists() and Path(outp) not in to_delete:
                    to_delete.append(Path(outp))

                if not to_delete:
                    messagebox.showinfo('Cleanup', 'No atomsk-generated files to delete, only polycrystal.txt will be deleted (if exists).')
                    # Only delete polycrystal.txt
                    if Path(outp).exists():
                        Path(outp).unlink()
                        log(f'Deleted: {outp}')
                        messagebox.showinfo('Cleanup Complete', f'Deleted: {outp}')
                    else:
                        messagebox.showinfo('Cleanup', 'polycrystal.txt not found, nothing deleted')
                else:
                    preview_list = '\n'.join([p.name for p in to_delete[:50]])
                    if len(to_delete) > 50:
                        preview_list += '\n...(and %d more)...' % (len(to_delete) - 50)

                    confirm = messagebox.askyesno('Confirm Delete', f'Deleting files created by atomsk (keeping polycrystal.lmp):\n\n{preview_list}\n\nContinue? This cannot be undone.')
                    if confirm:
                        log('Deleting atomsk generated files...')
                        deleted = []
                        for p in to_delete:
                            try:
                                if p.is_dir():
                                    shutil.rmtree(p)
                                else:
                                    p.unlink()
                                deleted.append(p)
                                log(f'Deleted: {p.name}')
                            except Exception as e:
                                log(f'Failed to delete {p}: {e}')
                        messagebox.showinfo('Cleanup Complete', f'Deleted {len(deleted)} items (excluding polycrystal.lmp)')
                    else:
                        log('User cancelled cleanup of atomsk files')
            else:
                # Atomsk not flow, only delete polycrystal.txt (confirm first)
                if Path(outp).exists():
                    confirm = messagebox.askyesno('Confirm Delete', f'Atomsk did not run. Delete generated polycrystal file only:\n{outp} ?')
                    if confirm:
                        try:
                            Path(outp).unlink()
                            log(f'Deleted: {outp}')
                            messagebox.showinfo('Cleanup Complete', f'Deleted: {outp}')
                        except Exception as e:
                            messagebox.showerror('Delete Failed', f'Failed to delete {outp}: {e}')
                    else:
                        log('User cancelled delete of polycrystal.txt')
                else:
                    messagebox.showinfo('Cleanup', 'polycrystal.txt not found, nothing deleted')

        # After cleanup (if any), convert polycrystal.lmp to a dump using atomsk and save both files
        try:
            poly_lmp_path = dir_for_ops / 'polycrystal.lmp'
            if poly_lmp_path.exists():
                cmd_dump = ['atomsk', poly_lmp_path.name, 'dump']
                log('Running atomsk to convert polycrystal.lmp to dump: ' + ' '.join(cmd_dump))
                try:
                    proc_dump = subprocess.run(cmd_dump, capture_output=True, text=True, cwd=str(dir_for_ops))
                except FileNotFoundError:
                    log('Error: atomsk not found. Optional dump generation skipped.')
                except Exception as e:
                    log(f'Exception converting to dump: {e}')
                else:
                    log('--- atomsk(dump) stdout ---')
                    log(proc_dump.stdout.strip())
                    log('--- atomsk(dump) stderr ---')
                    log(proc_dump.stderr.strip())
                    if proc_dump.returncode != 0:
                        log(f'atomsk conversion returned non-zero exit code: {proc_dump.returncode} (see stderr)')
                    else:
                        log('atomsk dump conversion successful. polycrystal.lmp and dump saved in: ' + str(dir_for_ops))
            else:
                log('polycrystal.lmp not found, skipping dump conversion')
        except Exception as e:
            log(f'Unexpected error during dump generation: {e}')

    btn_generate = tk.Button(frm, text='Generate polycrystal.txt', command=on_generate, bg='#4caf50', fg='white')
    btn_generate.grid(row=5, column=1, pady=10, sticky='w')

    btn_quit = tk.Button(frm, text='Exit', command=root.destroy)
    btn_quit.grid(row=5, column=2, pady=10, sticky='e')

    root.mainloop()


if __name__ == '__main__':
    main_gui()
