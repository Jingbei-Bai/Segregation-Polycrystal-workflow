#!/usr/bin/env python3
"""
generate_skew_data.py

Process a two-column text file (id value), extract GB atom values and plot the skewed distribution
(histogram + skewnorm fit curve) directly. Does not save JSON metadata.

Example CLI:
    python generate_skew_data.py --input Al_Cu_eam.txt --gb-ids gb_ids.npy --out segregation_skew_plot.pdf

The script will produce an image file (PDF by default).
"""

from pathlib import Path
from typing import Optional, Sequence
import argparse
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None

try:
    from scipy.stats import skewnorm
except Exception:
    skewnorm = None

import matplotlib.pyplot as plt


def _load_gb_ids(path: Optional[Path]) -> Optional[np.ndarray]:
    """Load a GB id list; supports .npy or text files with one id per line or whitespace-separated.
    If path is None, common filenames are tried. Returns a numpy integer array or None if not found."""
    candidates = []
    if path:
        candidates.append(Path(path))
    else:
        candidates.extend([Path('gb_ids.npy'), Path('GB_ids.npy'), Path('gb_ids.txt')])

    for p in candidates:
        if p is None:
            continue
        if p.exists():
            try:
                if p.suffix == '.npy':
                    arr = np.load(p)
                    return arr.astype(np.int64)
                else:
                    arr = np.loadtxt(p, dtype=np.int64)
                    return arr
            except Exception as e:
                print(f"Warning: failed to load GB id file {p}: {e}")
    return None


def process_file(input_path: Path, gb_ids: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read a two-column text file (id value) and return array (gb_vals - mean(non_gb)) * 96.485.

    Parameters:
      input_path: path to two-column text file
      gb_ids: optional GB atom id list (array-like)

    Returns:
      ndarray, 1-D, dtype=float
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data = np.loadtxt(str(input_path))
    if data.ndim == 1:
        if data.shape[0] < 2:
            raise ValueError('Input file is not two-column format')
        data = data.reshape(-1, 2)

    ids = data[:, 0].astype(np.int64)
    vals = data[:, 1].astype(float)

    if gb_ids is None:
        print('Warning: no GB id list provided; treating all atoms as GB (this affects baseline calculation)')
        gb_mask = np.ones_like(ids, dtype=bool)
    else:
        gb_set = set(map(int, gb_ids))
        gb_mask = np.array([int(i) in gb_set for i in ids], dtype=bool)

    gb_vals = vals[gb_mask]
    non_gb_vals = vals[~gb_mask]

    if non_gb_vals.size == 0:
        print('Warning: no non-GB atoms found; baseline yb set to 0')
        yb = 0.0
    else:
        yb = float(np.mean(non_gb_vals))

    y_arr = (gb_vals - yb) * 96.485
    return y_arr


def fit_skew(y: np.ndarray) -> Optional[tuple]:
    """Fit a skew-normal distribution using scipy.stats.skewnorm; return (a, loc, scale) or None if unavailable."""
    if skewnorm is None:
        print('scipy.stats.skewnorm not available; skipping fit. Install scipy to enable this feature.')
        return None
    try:
        a, loc, scale = skewnorm.fit(y)
        return float(a), float(loc), float(scale)
    except Exception as e:
        print(f'Fit failed: {e}')
        return None


# New plotting routine: replaces saving JSON; directly generate plot
def plot_skew_distribution(y: np.ndarray, params: Optional[tuple], out_path: Path, bins: int = 40, show: bool = False):
    """Plot histogram and, if available, the skewnorm fit curve. Save to out_path (PDF/PNG)."""
    if y is None or y.size == 0:
        print('No data to plot (y is empty)')
        return

    plt.figure(figsize=(8, 5))
    counts, bin_edges, _ = plt.hist(y, bins=bins, density=True, alpha=0.6, color='C0', label='Data')

    x = np.linspace(np.min(y), np.max(y), 1000)
    if params is not None and skewnorm is not None:
        a, loc, scale = params
        pdf = skewnorm.pdf(x, a, loc=loc, scale=scale)
        plt.plot(x, pdf, 'k--', lw=2, label=f'skewnorm fit (a={a:.2f})')
    else:
        if params is not None and skewnorm is None:
            print('Fit parameters available but scipy is not installed; cannot plot fit curve.')

    plt.xlabel('ΔE (kJ/mol)')
    plt.ylabel('Probability Density')
    plt.title('Segregation energy (GB) distribution')
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    print(f'Saved plot: {out_path}')
    if show:
        plt.show()
    plt.close()


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description='Process two-column (id value) text and plot skew distribution')
    parser.add_argument('--input', '-i', type=str, default='Al_Cu_eam.txt', help='Input text file, format: id value')
    parser.add_argument('--gb-ids', '-g', type=str, default=None, help='Optional: GB id file (.npy or .txt)')
    parser.add_argument('--lmp', '-l', type=str, default=None, help='Optional: LAMMPS file path (will attempt to compute GB ids via OVITO)')
    parser.add_argument('--out', '-o', type=str, default='segregation_skew_plot.pdf', help='Output plot file (PDF/PNG)')
    parser.add_argument('--bins', type=int, default=40, help='Number of histogram bins')
    parser.add_argument('--show', action='store_true', help='Show plot after saving (useful in GUI-enabled environments)')
    args = parser.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        print(f'Error: input file not found: {inp}')
        return

    gb = None
    # If LMP provided, try to compute gb ids via helper (OVITO)
    if args.lmp:
        try:
            from generate_get_gb_ids import get_gb_ids_and_indices
        except Exception:
            # fallback: try to import ovito-based function directly if present
            try:
                import generate_get_gb_ids as _g
                get_gb_ids_and_indices = _g.get_gb_ids_and_indices
            except Exception:
                get_gb_ids_and_indices = None

        if 'get_gb_ids_and_indices' in globals() and callable(globals()['get_gb_ids_and_indices']):
            try:
                gb_ids, gb_indices = get_gb_ids_and_indices(args.lmp)
                gb = gb_ids
                print(f'Computed {gb.size} GB ids from LAMMPS file')
            except Exception as e:
                print(f'Failed to compute GB ids with OVITO: {e} (will try --gb-ids or fallback behavior)')
        else:
            print('No function found to compute GB ids from LAMMPS (requires generate_get_gb_ids.py and OVITO), falling back to --gb-ids or default behavior')

    if gb is None:
        gb = _load_gb_ids(Path(args.gb_ids) if args.gb_ids else None)
        if gb is None:
            print('No GB id list found; will use all atoms as GB (this affects baseline calculation).')

    y = process_file(inp, gb)
    params = fit_skew(y)
    plot_skew_distribution(y, params, Path(args.out), bins=args.bins, show=args.show)


def main_gui():
    if tk is None:
        print('tkinter not available in current environment; cannot open GUI. Use command-line mode.')
        return

    root = tk.Tk()
    root.title('Generate skew data (plot)')

    def browse_input():
        p = filedialog.askopenfilename(filetypes=[('Text files','*.txt'),('All files','*.*')])
        if p:
            ent_input.delete(0, tk.END)
            ent_input.insert(0, p)

    def browse_gb():
        p = filedialog.askopenfilename(filetypes=[('NumPy files','*.npy'),('Text files','*.txt'),('All files','*.*')])
        if p:
            ent_gb.delete(0, tk.END)
            ent_gb.insert(0, p)

    def browse_lmp():
        p = filedialog.askopenfilename(filetypes=[('Lammps files','*.lmp;*.dump;*.data'),('All','*.*')])
        if p:
            ent_lmp.delete(0, tk.END)
            ent_lmp.insert(0, p)

    frm = tk.Frame(root, padx=8, pady=8)
    frm.pack(fill=tk.BOTH, expand=True)

    tk.Label(frm, text='Input (id value) txt:').grid(row=0, column=0, sticky='w')
    ent_input = tk.Entry(frm, width=60)
    ent_input.insert(0, 'Al_Cu_eam.txt')
    ent_input.grid(row=0, column=1)
    tk.Button(frm, text='Browse', command=browse_input).grid(row=0, column=2, padx=4)

    tk.Label(frm, text='GB ids (optional):').grid(row=1, column=0, sticky='w')
    ent_gb = tk.Entry(frm, width=60)
    ent_gb.grid(row=1, column=1)
    tk.Button(frm, text='Browse', command=browse_gb).grid(row=1, column=2, padx=4)

    tk.Label(frm, text='LAMMPS file (optional, use OVITO to compute GB ids):').grid(row=2, column=0, sticky='w')
    ent_lmp = tk.Entry(frm, width=60)
    ent_lmp.grid(row=2, column=1)
    tk.Button(frm, text='Browse', command=browse_lmp).grid(row=2, column=2, padx=4)

    tk.Label(frm, text='Output file (PDF/PNG):').grid(row=3, column=0, sticky='w')
    ent_out = tk.Entry(frm, width=60)
    ent_out.insert(0, 'segregation_skew_plot.pdf')
    ent_out.grid(row=3, column=1)

    tk.Label(frm, text='Bins:').grid(row=4, column=0, sticky='w')
    ent_bins = tk.Entry(frm, width=10)
    ent_bins.insert(0, '40')
    ent_bins.grid(row=4, column=1, sticky='w')

    def on_run():
        inp = Path(ent_input.get())
        gbp = ent_gb.get().strip()
        lmp_p = ent_lmp.get().strip()
        outp = ent_out.get().strip()
        bins = int(ent_bins.get().strip() or '40')
        if not inp.exists():
            messagebox.showerror('Error', f'Input file not found: {inp}')
            return
        gb_arr = None
        if lmp_p:
            try:
                from generate_get_gb_ids import get_gb_ids_and_indices
            except Exception:
                try:
                    import generate_get_gb_ids as _g
                    get_gb_ids_and_indices = _g.get_gb_ids_and_indices
                except Exception:
                    get_gb_ids_and_indices = None

            if 'get_gb_ids_and_indices' in globals() and callable(globals()['get_gb_ids_and_indices']):
                try:
                    gb_ids, gb_indices = get_gb_ids_and_indices(lmp_p)
                    gb_arr = gb_ids
                    print(f'Computed {gb_arr.size} GB ids from LAMMPS file')
                except Exception as e:
                    messagebox.showwarning('Warning', f'Failed to compute GB ids with OVITO: {e}\nWill fall back to --gb-ids or default behavior')
            else:
                messagebox.showwarning('Warning', 'generate_get_gb_ids.py or OVITO not found; will not compute GB ids, falling back to --gb-ids or default behavior')

        if gb_arr is None and gbp:
            gb_arr = _load_gb_ids(Path(gbp))

        y = process_file(inp, gb_arr)
        params = fit_skew(y)
        try:
            plot_skew_distribution(y, params, Path(outp), bins=bins, show=True)
            messagebox.showinfo('Done', f'Saved plot to: {outp}')
        except Exception as e:
            messagebox.showerror('Error', f'Plot failed: {e}')

    btn = tk.Button(frm, text='Run', command=on_run, bg='#2196f3', fg='white')
    btn.grid(row=5, column=1, pady=8, sticky='w')

    root.mainloop()


if __name__ == '__main__':
    # 优先 GUI（若可用），否则 CLI
    if tk is not None:
        main_gui()
    else:
        main_cli()
