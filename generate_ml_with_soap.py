#!/usr/bin/env python3
"""
generate_ml_with_soap.py

Pipeline:
 1) Load two-column energy file (id value)
 2) Obtain GB ids and/or GB indices (prefer provided indices; if --lmp provided, may call generate_get_gb_ids.py)
 3) Read LAMMPS dump and compute SOAP descriptors using ASE + dscribe (or load precomputed features.npy)
 4) Align/select SOAP vectors X with target y based on GB indices
 5) Train regression models (LinearRegression) and save models and metrics

Notes: This script requires the following packages to enable full functionality at runtime:
  - numpy, scikit-learn, joblib, matplotlib (recommended)
  - ase, dscribe (for computing SOAP from dump)
  - The OVITO-based helper in generate_get_gb_ids.py is needed only if you want to compute GB ids automatically from an LMP file

Example:
  python generate_ml_with_soap.py --energy Al_Cu.txt --dump medium_final_atoms.dump --gb-indices gb_ids_indices.npy --out-dir ml_output

Or use OVITO to compute GB ids (requires OVITO and generate_get_gb_ids.py):
  python generate_ml_with_soap.py --energy Al_Cu.txt --dump medium_final_atoms.dump --lmp medium_final_atoms.lmp --out-dir ml_output

"""
from pathlib import Path
import argparse
import numpy as np
from typing import Optional, Sequence, Tuple
import subprocess
import json

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Optional heavy imports guarded
try:
    from ase.io import read as ase_read
    from dscribe.descriptors import SOAP
except Exception:
    ase_read = None
    SOAP = None

# Try to import scipy skew-related tools; if unavailable, we'll compute with numpy
try:
    from scipy.stats import skewnorm, skew as scipy_skew
except Exception:
    skewnorm = None
    scipy_skew = None

# Optional: import plotting helper from generate_skew_data
try:
    from generate_skew_data import plot_skew_distribution
except Exception:
    plot_skew_distribution = None

# GUI availability
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None


def load_energy_map(energy_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load two-column energy file (id, value) and return (ids_array, values_array).
    IDs are integers; values are floats.
    """
    data = np.loadtxt(str(energy_file))
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    ids = data[:, 0].astype(np.int64)
    vals = data[:, 1].astype(float)
    return ids, vals


def compute_y_from_map(ids_all: np.ndarray, vals_all: np.ndarray, gb_ids: Sequence[int]) -> np.ndarray:
    """Compute y array for gb_ids in the order of gb_ids.

    Baseline yb is computed as mean(values of atoms NOT in gb_ids).
    Then return y_arr = (vals_gb - yb) * 96.485
    """
    # build id -> value mapping
    id_to_val = {int(i): float(v) for i, v in zip(ids_all, vals_all)}
    gb_ids_int = [int(x) for x in gb_ids]

    # values for GB ids in provided order; check missing
    gb_vals = []
    missing = []
    for gid in gb_ids_int:
        if gid in id_to_val:
            gb_vals.append(id_to_val[gid])
        else:
            missing.append(gid)
            gb_vals.append(np.nan)

    if missing:
        raise RuntimeError(f'The following GB ids were not found in the energy file: {missing}')

    # compute baseline using non-GB values
    non_gb_vals = [v for i, v in zip(ids_all, vals_all) if int(i) not in set(gb_ids_int)]
    if len(non_gb_vals) == 0:
        yb = 0.0
    else:
        yb = float(np.mean(non_gb_vals))

    y_arr = (np.array(gb_vals, dtype=float) - yb) * 96.485
    return y_arr


def compute_skew_from_y(y: np.ndarray) -> dict:
    """Compute basic statistics and (if possible) fit a skew-normal distribution to y.

    Returns a dict with keys:
      - n: number of samples
      - mean: mean of y
      - std: standard deviation (population, ddof=0)
      - skewness: sample skewness (computed via scipy if available, else numpy third moment)
      - fit_params: (a, loc, scale) if skewnorm fit succeeded, else None
    """
    # annotate as object-valued dict to allow values of mixed types (ints, floats, None, tuples)
    res: dict[str, object] = {'n': int(y.size)}
    if y is None or y.size == 0:
        res.update({'mean': None, 'std': None, 'skewness': None, 'fit_params': None})
        return res

    mean = float(np.mean(y))
    std = float(np.std(y, ddof=0))
    # compute skewness: prefer scipy.stats.skew if available
    if scipy_skew is not None:
        try:
            skewness = float(scipy_skew(y, bias=False))
        except Exception:
            # fallback to manual
            m3 = float(np.mean((y - mean) ** 3))
            skewness = float(m3 / (std ** 3)) if std != 0 else 0.0
    else:
        m3 = float(np.mean((y - mean) ** 3))
        skewness = float(m3 / (std ** 3)) if std != 0 else 0.0

    fit_params = None
    if skewnorm is not None:
        try:
            a, loc, scale = skewnorm.fit(y)
            fit_params = (float(a), float(loc), float(scale))
        except Exception:
            fit_params = None

    res.update({'mean': mean, 'std': std, 'skewness': skewness, 'fit_params': fit_params})
    return res


def load_gb_ids_indices(gb_indices_path: Optional[Path], gb_ids_path: Optional[Path], lmp_path: Optional[Path]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (gb_ids, gb_indices). Each may be None if not available.

    Priority:
     - If gb_indices_path provided and exists: load indices (npy) and, if gb_ids_path provided, load ids too.
     - If lmp_path provided: try to use generate_get_gb_ids.get_gb_ids_and_indices (OVITO) to compute both.
     - If only gb_ids_path provided: load ids (indices remain None).
     - Otherwise, attempt to find common filenames in the current working directory.

    """
    gb_ids = None
    gb_indices = None
    # direct loads
    if gb_indices_path and gb_indices_path.exists():
        gb_indices = np.load(gb_indices_path)
    if gb_ids_path and gb_ids_path.exists():
        try:
            gb_ids = np.load(gb_ids_path)
        except Exception:
            gb_ids = np.loadtxt(gb_ids_path, dtype=np.int64)

    # fallback: try common filenames in cwd
    cwd = Path('.').resolve()
    common_candidates = [cwd / 'gb_ids.npy', cwd / 'gb_ids_ids.npy', cwd / 'GB_ids.npy', cwd / 'gb_ids.txt', cwd / 'gb_ids_indices.npy', cwd / 'gb_ids_indices.txt', cwd / 'gb_ids_ids.txt']
    if gb_ids is None or gb_indices is None:
        for p in common_candidates:
            if p.exists():
                try:
                    if 'indices' in p.name or 'indices' in str(gb_indices_path or ''):
                        if gb_indices is None and p.suffix == '.npy':
                            gb_indices = np.load(p)
                    else:
                        if gb_ids is None:
                            if p.suffix == '.npy':
                                gb_ids = np.load(p)
                            else:
                                try:
                                    arr = np.loadtxt(p, dtype=np.int64)
                                    gb_ids = arr
                                except Exception:
                                    pass
                except Exception:
                    continue

    # try OVITO if still missing and lmp provided
    if (gb_ids is None or gb_indices is None) and lmp_path:
        try:
            from generate_get_gb_ids import get_gb_ids_and_indices
            gb_ids2, gb_indices2 = get_gb_ids_and_indices(str(lmp_path))
            gb_ids = gb_ids if gb_ids is not None else gb_ids2
            gb_indices = gb_indices if gb_indices is not None else gb_indices2
        except Exception as e:
            print(f"Warning: failed to compute GB ids/indices via OVITO: {e}")

    return gb_ids, gb_indices


def compute_soap_features(dump_path: Path, r_cut: float = 6.0, n_max: int = 8, l_max: int = 6, sigma: float = 0.5, periodic: bool = False) -> np.ndarray:
    """Compute SOAP descriptors for all atoms in LAMMPS dump using ASE + dscribe.
    SOAP parameters are configurable and passed through from CLI/GUI.
    Returns array shape (n_atoms, n_features).
    """
    if ase_read is None or SOAP is None:
        raise RuntimeError('ASE or dscribe not available. Please pip install ase dscribe')

    system = ase_read(str(dump_path), format='lammps-dump-text')
    # try to infer species from atomic numbers
    atomic_numbers = np.array(system.get_atomic_numbers())
    species = list(sorted(set(atomic_numbers)))

    # Create SOAP with provided parameters
    soap = SOAP(species=species, r_cut=float(r_cut), n_max=int(n_max), l_max=int(l_max), sigma=float(sigma), periodic=bool(periodic))
    print(f'Computing SOAP (r_cut={r_cut}, n_max={n_max}, l_max={l_max}, sigma={sigma}, periodic={periodic}): this may take some time...')
    X = soap.create(system)
    return X


def map_ids_to_indices_from_dump(dump_path: Path, ids: np.ndarray) -> Optional[np.ndarray]:
    """Attempt to map atom ids -> atom indices by reading dump with ASE and checking arrays for 'id' or 'atom_id'.
    Returns array of indices (same order as ids) or None if mapping not possible.
    """
    if ase_read is None:
        return None
    system = ase_read(str(dump_path), format='lammps-dump-text')
    # ASE stores extra arrays in system.arrays
    arrays = getattr(system, 'arrays', {})
    possible_keys = ['id', 'atom_id', 'atomIDs', 'atom_ids']
    found = None
    for k in possible_keys:
        if k in arrays:
            found = arrays[k]
            break
    if found is None:
        # sometimes ASE stores as 'numbers' or 'atom_ids' under info? try to inspect
        try:
            # try reading comment lines if any (less reliable)
            return None
        except Exception:
            return None

    # build mapping from id to index
    id_to_index = {int(a): int(i) for i, a in enumerate(found)}
    indices = []
    for v in ids:
        if int(v) in id_to_index:
            indices.append(id_to_index[int(v)])
        else:
            # missing id
            indices.append(-1)
    indices = np.array(indices, dtype=np.int64)
    if np.any(indices < 0):
        return None
    return indices


def train_and_save_models(X: np.ndarray, y: np.ndarray, out_dir: Path) -> Tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    """Train models, save them, and return results plus trained models and test splits for plotting.

    Returns:
      results: dict of metrics
      models: dict name->trained estimator
      X_test, y_test: arrays for test split
      y_preds: dict name->predictions on X_test
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'LinearRegression': LinearRegression(),
    }
    results = {}
    trained = {}
    y_preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'mae_test': float(mae), 'r2_test': float(r2)}
        trained[name] = model
        y_preds[name] = y_pred
        joblib.dump(model, out_dir / f'{name}.joblib')
        print(f'Saved model: {out_dir / (name + ".joblib")}')
    return results, trained, X_test, y_test, y_preds


def _plot_parity(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str = ''):
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, s=20, alpha=0.7)
        mn = min(float(np.nanmin(y_true)), float(np.nanmin(y_pred)))
        mx = max(float(np.nanmax(y_true)), float(np.nanmax(y_pred)))
        plt.plot([mn, mx], [mn, mx], 'k--')
        plt.xlabel('y_true (kJ/mol)')
        plt.ylabel('y_pred (kJ/mol)')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=200)
        plt.close()
        print(f'Saved parity plot: {out_path}')
    except Exception as e:
        print(f'Failed to save parity plot {out_path}: {e}')


def _plot_feature_importance(model, feature_names: Optional[Sequence[str]], out_path: Path, top_n: int = 30):
    import matplotlib.pyplot as plt
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, 'feature_importances_'):
        print('Model has no feature_importances_, skipping feature importance plot')
        return
    try:
        fi = model.feature_importances_
        idx = np.argsort(fi)[::-1]
        if feature_names is None:
            feature_names = [f'F{i}' for i in range(len(fi))]
        top = idx[:top_n]
        plt.figure(figsize=(8, max(4, int(0.15*len(top)))))
        y_pos = np.arange(len(top))
        plt.barh(y_pos, fi[top][::-1])
        labels = [feature_names[i] for i in top][::-1]
        plt.yticks(y_pos, labels)
        plt.xlabel('Feature importance')
        plt.title('Feature importances (top {})'.format(len(top)))
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=200)
        plt.close()
        print(f'Saved feature importance plot: {out_path}')
    except Exception as e:
        print(f'Failed to save feature importance plot {out_path}: {e}')


def main():
    parser = argparse.ArgumentParser(description='Compute SOAP features for GB atoms and train regression models')
    parser.add_argument('--energy', type=str, default='Al_Cu.txt', help='Two-column energy file (id value)')
    parser.add_argument('--dump', type=str, default=None, help='LAMMPS dump file for SOAP (lammps-dump-text) — required now (no precomputed features option)')
    parser.add_argument('--gb-ids', type=str, default=None, help='GB ids file (.npy or .txt)')
    parser.add_argument('--gb-indices', type=str, default=None, help='GB indices file (.npy) produced by OVITO')
    parser.add_argument('--lmp', type=str, default=None, help='Optional LAMMPS file for OVITO-based GB detection (generate_get_gb_ids.py)')
    # SOAP parameter options
    parser.add_argument('--soap-rcut', type=float, default=6.0, help='SOAP r_cut')
    parser.add_argument('--soap-nmax', type=int, default=8, help='SOAP n_max')
    parser.add_argument('--soap-lmax', type=int, default=6, help='SOAP l_max')
    parser.add_argument('--soap-sigma', type=float, default=0.5, help='SOAP sigma')
    parser.add_argument('--soap-periodic', action='store_true', help='Set SOAP periodic=True (default False unless provided)')
    parser.add_argument('--out-dir', type=str, default='ml_output', help='Output directory to save models')
    args = parser.parse_args()

    energy_path = Path(args.energy)
    if not energy_path.exists():
        raise FileNotFoundError(f'Energy file not found: {energy_path}')

    ids_all, vals_all = load_energy_map(energy_path)

    gb_ids, gb_indices = load_gb_ids_indices(Path(args.gb_indices) if args.gb_indices else None,
                                            Path(args.gb_ids) if args.gb_ids else None,
                                            Path(args.lmp) if args.lmp else None)

    # If we have gb_indices, great; if only gb_ids, try to map to indices using dump
    if gb_indices is None and gb_ids is not None and args.dump:
        mapped = map_ids_to_indices_from_dump(Path(args.dump), gb_ids)
        if mapped is not None:
            gb_indices = mapped
            print('Mapped gb_ids -> gb_indices using dump')

    # If dump not provided but lmp is provided, attempt to convert lmp -> dump using atomsk
    dump_path = Path(args.dump) if args.dump else None
    if dump_path is None and args.lmp:
        lmp_path = Path(args.lmp)
        if lmp_path.exists():
            cwd = lmp_path.parent
            pre_existing = set(cwd.iterdir())
            cmd = ['atomsk', lmp_path.name, 'dump']
            print('Attempting to run atomsk to convert LMP -> dump: ' + ' '.join(cmd))
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
            except FileNotFoundError:
                print('atomsk not found in PATH; cannot convert LMP to dump automatically.')
            except Exception as e:
                print(f'Error running atomsk: {e}')
            else:
                print('atomsk stdout:')
                print(proc.stdout)
                print('atomsk stderr:')
                print(proc.stderr)
                post_existing = set(cwd.iterdir())
                created = post_existing - pre_existing
                # try to find a new dump-like file
                dump_candidates = [p for p in created if p.suffix in ('.dump', '.lammpstrj', '.txt') or 'dump' in p.name.lower()]
                if dump_candidates:
                    dump_path = dump_candidates[0]
                    print(f'Detected generated dump file: {dump_path.name}')
                else:
                    # fallback: look for any existing dump-like file in cwd
                    all_candidates = [p for p in cwd.iterdir() if p.suffix in ('.dump', '.lammpstrj') or 'dump' in p.name.lower()]
                    if all_candidates:
                        dump_path = all_candidates[0]
                        print(f'Found existing dump-like file: {dump_path.name}')
                    else:
                        print('No dump file detected after atomsk run; please provide --dump or run atomsk manually.')

    # assign args.dump to detected dump_path if we found one
    if dump_path is not None:
        args.dump = str(dump_path)

    if gb_ids is None and gb_indices is not None:
        # If only indices provided, we can attempt to read dump and extract ids from ASE arrays if possible
        if args.dump and ase_read is not None:
            system = ase_read(str(args.dump), format='lammps-dump-text')
            arrays = getattr(system, 'arrays', {})
            # attempt to find id array
            for key in ['id', 'atom_id', 'atomIDs', 'atom_ids']:
                if key in arrays:
                    ids_from_dump = np.array(arrays[key], dtype=np.int64)
                    # build gb_ids accordingly
                    gb_ids = ids_from_dump[gb_indices]
                    break

    if gb_ids is None and gb_indices is None:
        raise RuntimeError('No GB ids or GB indices available. Provide --gb-ids/--gb-indices or use --lmp with OVITO')

    # compute target y aligned to gb_indices order
    # We build a mapping id->value from energy file, compute baseline = mean(non_gb)
    # then y array in order corresponding to gb_indices
    if gb_ids is None:
        # try to derive gb_ids by reading dump and selecting by indices
        if args.dump and ase_read is not None:
            system = ase_read(str(args.dump), format='lammps-dump-text')
            arrays = getattr(system, 'arrays', {})
            found = None
            for k in ['id', 'atom_id', 'atomIDs', 'atom_ids']:
                if k in arrays:
                    found = np.array(arrays[k], dtype=np.int64)
                    break
            if found is None:
                raise RuntimeError('gb_ids not available and dump does not contain atom id arrays')
            gb_ids = found[gb_indices]
        else:
            raise RuntimeError('gb_ids not available; provide --gb-ids or a dump with id arrays')

    # compute y in order of gb_ids
    y = compute_y_from_map(ids_all, vals_all, gb_ids)

    # --- New: compute skew / fit information for GB energies and save to out_dir later ---
    # We'll compute it here and save after models are trained (out_dir exists).
    skew_info = compute_skew_from_y(y)

    # compute features X from dump using configurable SOAP parameters (precomputed features removed)
    if args.dump is None:
        raise RuntimeError('This script requires a LAMMPS dump. Provide --dump or a valid --lmp that can be converted by atomsk.')
    X_all = compute_soap_features(Path(args.dump), r_cut=args.soap_rcut, n_max=args.soap_nmax, l_max=args.soap_lmax, sigma=args.soap_sigma, periodic=args.soap_periodic)

    # select features for gb_indices
    if gb_indices is None:
        # try to map gb_ids -> indices using dump
        mapped = map_ids_to_indices_from_dump(Path(args.dump), gb_ids)
        if mapped is None:
            raise RuntimeError('Cannot obtain gb_indices to select SOAP rows; provide --gb-indices or ensure dump contains id arrays')
        gb_indices = mapped

    X = X_all[gb_indices]

    if X.shape[0] != y.shape[0]:
        raise RuntimeError(f'Sample count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} entries')

    out_dir = Path(args.out_dir)
    results, trained, X_test, y_test, y_preds = train_and_save_models(X, y, out_dir)

    # Save skew info and y array to out_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'skew_info.json').write_text(json.dumps(skew_info, indent=2))
        np.save(out_dir / 'y_gb.npy', y)
        print(f'Saved skew info to {out_dir / "skew_info.json"} and GB y array to {out_dir / "y_gb.npy"}')
        # try to generate skew plot if plotting helper is available
        if plot_skew_distribution is not None:
            try:
                plot_skew_distribution(y, skew_info.get('fit_params'), out_dir / 'skew_plot.png', bins=40, show=False)
                print(f'Saved skew plot to {out_dir / "skew_plot.png"}')
            except Exception as e:
                print(f'Warning: failed to generate skew plot: {e}')
    except Exception as e:
        print(f'Warning: failed to save skew info: {e}')

    print('Training results:')
    for k, v in results.items():
        print(k, v)

    # Visualization: parity plots and feature importance for trained models
    viz_dir = out_dir / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)
    for name, pred in y_preds.items():
        _plot_parity(y_test.ravel(), pred.ravel(), viz_dir / f'parity_{name}.png', title=f'Parity: {name}')

    # feature importance: plot for any trained model that exposes feature_importances_
    for mname, model in trained.items():
        try:
            if hasattr(model, 'feature_importances_'):
                _plot_feature_importance(model, None, viz_dir / f'feature_importance_{mname.lower()}.png')
        except Exception:
            # skip plotting failures for particular models
            pass

    print(f'Visualizations saved to {viz_dir}')


def main_gui():
    if tk is None:
        print('tkinter is not available; cannot open GUI.')
        return

    root = tk.Tk()
    root.title('ML with SOAP - GUI')

    frm = tk.Frame(root, padx=8, pady=8)
    frm.pack(fill='both', expand=True)

    def browse_file(entry, filetypes=None):
        p = filedialog.askopenfilename(filetypes=filetypes or [('All','*.*')])
        if p:
            entry.delete(0, tk.END)
            entry.insert(0, p)

    tk.Label(frm, text='Energy file (id value):').grid(row=0, column=0, sticky='w')
    ent_energy = tk.Entry(frm, width=60)
    ent_energy.insert(0, 'Al_Cu.txt')
    ent_energy.grid(row=0, column=1)
    tk.Button(frm, text='Browse', command=lambda: browse_file(ent_energy, [('Text files','*.txt'),('All','*.*')])).grid(row=0, column=2, padx=4)

    tk.Label(frm, text='LAMMPS dump (for SOAP):').grid(row=1, column=0, sticky='w')
    ent_dump = tk.Entry(frm, width=60)
    ent_dump.grid(row=1, column=1)
    tk.Button(frm, text='Browse', command=lambda: browse_file(ent_dump, [('Lammps dump','*.dump;*.lammpstrj;*.txt'),('All','*.*')])).grid(row=1, column=2, padx=4)

    # SOAP parameter inputs (optional)
    tk.Label(frm, text='SOAP r_cut:').grid(row=2, column=0, sticky='w')
    ent_rcut = tk.Entry(frm, width=20)
    ent_rcut.insert(0, '6.0')
    ent_rcut.grid(row=2, column=1, sticky='w')

    tk.Label(frm, text='SOAP n_max:').grid(row=3, column=0, sticky='w')
    ent_nmax = tk.Entry(frm, width=20)
    ent_nmax.insert(0, '8')
    ent_nmax.grid(row=3, column=1, sticky='w')

    tk.Label(frm, text='SOAP l_max:').grid(row=4, column=0, sticky='w')
    ent_lmax = tk.Entry(frm, width=20)
    ent_lmax.insert(0, '6')
    ent_lmax.grid(row=4, column=1, sticky='w')

    tk.Label(frm, text='SOAP sigma:').grid(row=5, column=0, sticky='w')
    ent_sigma = tk.Entry(frm, width=20)
    ent_sigma.insert(0, '0.5')
    ent_sigma.grid(row=5, column=1, sticky='w')

    soap_periodic_var = tk.BooleanVar(value=False)
    tk.Checkbutton(frm, text='SOAP periodic', variable=soap_periodic_var).grid(row=5, column=2, sticky='w')

    tk.Label(frm, text='GB ids (optional .npy/.txt):').grid(row=6, column=0, sticky='w')
    ent_gbids = tk.Entry(frm, width=60)
    ent_gbids.grid(row=6, column=1)
    tk.Button(frm, text='Browse', command=lambda: browse_file(ent_gbids, [('NumPy','*.npy'),('Text','*.txt'),('All','*.*')])).grid(row=6, column=2, padx=4)

    tk.Label(frm, text='GB indices (optional .npy):').grid(row=7, column=0, sticky='w')
    ent_gbind = tk.Entry(frm, width=60)
    ent_gbind.grid(row=7, column=1)
    tk.Button(frm, text='Browse', command=lambda: browse_file(ent_gbind, [('NumPy','*.npy'),('All','*.*')])).grid(row=7, column=2, padx=4)

    # removed precomputed features field per request

    tk.Label(frm, text='Optional LMP (OVITO) to compute GB ids:').grid(row=8, column=0, sticky='w')
    ent_lmp = tk.Entry(frm, width=60)
    ent_lmp.grid(row=8, column=1)
    tk.Button(frm, text='Browse', command=lambda: browse_file(ent_lmp, [('Lammps','*.lmp;*.data'),('All','*.*')])).grid(row=8, column=2, padx=4)

    tk.Label(frm, text='Output dir:').grid(row=9, column=0, sticky='w')
    ent_out = tk.Entry(frm, width=60)
    ent_out.insert(0, 'ml_output')
    ent_out.grid(row=9, column=1)
    tk.Button(frm, text='Browse', command=lambda: browse_file(ent_out, [('All','*.*')])).grid(row=9, column=2, padx=4)

    txt_log = tk.Text(frm, width=80, height=10)
    txt_log.grid(row=10, column=0, columnspan=3, pady=(8,0))

    def log(msg: str):
        txt_log.insert(tk.END, msg + '\n')
        txt_log.see(tk.END)

    def on_run():
        try:
            energy = ent_energy.get().strip()
            dump = ent_dump.get().strip() or None
            # SOAP GUI params
            rcut_val = float(ent_rcut.get().strip() or '6.0')
            nmax_val = int(ent_nmax.get().strip() or '8')
            lmax_val = int(ent_lmax.get().strip() or '6')
            sigma_val = float(ent_sigma.get().strip() or '0.5')
            periodic_val = bool(soap_periodic_var.get())

            gbids = ent_gbids.get().strip() or None
            gbind = ent_gbind.get().strip() or None
            lmp = ent_lmp.get().strip() or None
            outdir = ent_out.get().strip() or 'ml_output'

            if not energy:
                messagebox.showerror('Error', 'Please provide energy file')
                return
            log(f'Loading energy file: {energy}')
            ids_all, vals_all = load_energy_map(Path(energy))

            log('Locating GB ids/indices...')
            gb_ids, gb_indices = load_gb_ids_indices(Path(gbind) if gbind else None,
                                                    Path(gbids) if gbids else None,
                                                    Path(lmp) if lmp else None)

            if gb_indices is None and gb_ids is not None and dump:
                mapped = map_ids_to_indices_from_dump(Path(dump), gb_ids)
                if mapped is not None:
                    gb_indices = mapped
                    log('Mapped gb_ids -> gb_indices using dump')

            if gb_ids is None and gb_indices is not None and dump and ase_read is not None:
                system = ase_read(str(dump), format='lammps-dump-text')
                arrays = getattr(system, 'arrays', {})
                for key in ['id', 'atom_id', 'atomIDs', 'atom_ids']:
                    if key in arrays:
                        ids_from_dump = np.array(arrays[key], dtype=np.int64)
                        gb_ids = ids_from_dump[gb_indices]
                        break

            if gb_ids is None and gb_indices is None:
                messagebox.showerror('Error', 'No GB ids or indices found. Provide GB files or an LMP for OVITO.')
                return

            log('Computing y array...')
            y = compute_y_from_map(ids_all, vals_all, gb_ids)

            log('Preparing features (computing SOAP from dump)...')
            if not dump:
                messagebox.showerror('Error', 'Precomputed features were removed. Please provide a LAMMPS dump to compute SOAP.')
                return
            # compute SOAP with GUI-provided params
            X_all = compute_soap_features(Path(dump), r_cut=rcut_val, n_max=nmax_val, l_max=lmax_val, sigma=sigma_val, periodic=periodic_val)

            if gb_indices is None:
                mapped = map_ids_to_indices_from_dump(Path(dump), gb_ids)
                if mapped is None:
                    messagebox.showerror('Error', 'Cannot obtain gb_indices to select SOAP rows; provide --gb-indices or ensure dump contains id arrays')
                    return
                gb_indices_local = mapped
            else:
                gb_indices_local = gb_indices

            X = X_all[gb_indices_local]
            if X.shape[0] != y.shape[0]:
                messagebox.showerror('Error', f'Sample count mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} entries')
                return

            log('Training models...')
            results, trained, X_test, y_test, y_preds = train_and_save_models(X, y, Path(outdir))
            log('Training finished')
            for k, v in results.items():
                log(f'{k}: {v}')

            viz_dir = Path(outdir) / 'viz'
            for name, pred in y_preds.items():
                _plot_parity(y_test.ravel(), pred.ravel(), viz_dir / f'parity_{name}.png', title=f'Parity: {name}')
            for mname, model in trained.items():
                try:
                    if hasattr(model, 'feature_importances_'):
                        _plot_feature_importance(model, None, viz_dir / f'feature_importance_{mname.lower()}.png')
                except Exception:
                    pass

            log(f'Visualizations saved to {viz_dir}')
            if messagebox.askyesno('Done', f'Finished. Open visualization folder?'):
                import os
                os.startfile(str(viz_dir))

        except Exception as e:
            messagebox.showerror('Error', str(e))
            log(f'ERROR: {e}')

    btn = tk.Button(frm, text='Run', command=on_run, bg='#4caf50', fg='white')
    btn.grid(row=11, column=1, pady=8, sticky='w')

    root.mainloop()


if __name__ == '__main__':
    # Prefer GUI when tkinter is available
    if tk is not None:
        try:
            main_gui()
        except Exception as _:
            # fallback to CLI on GUI failure
            main()
    else:
        main()
