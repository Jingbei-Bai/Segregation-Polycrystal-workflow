#!/usr/bin/env python3
"""
generate_get_gb_ids.py

Use OVITO to run adaptive CNA (and centro-symmetry) on a LAMMPS data/dump file to identify grain boundary (GB) atoms.

Behavior:
- Provides function `get_gb_ids_and_indices(polycrystal)` (adapted from the provided notebook with minor robustness improvements)
- Provides CLI: --lmp specifies the lmp file path; if not provided, prompts interactively
- Results are saved as gb_ids.npy, gb_indices.npy (and an optional text file)

Note: This script must be run in a Python environment with OVITO installed (i.e. `import ovito` must succeed).
"""
from pathlib import Path
import argparse
import numpy as np
from typing import Tuple

# OVITO 依赖在导入阶段进行捕获，便于在没有安装 OVITO 的环境中给出友好提示
try:
    import ovito
    from ovito.modifiers import CentroSymmetryModifier, CoordinationAnalysisModifier
except Exception as e:
    ovito = None


def get_gb_ids_and_indices(polycrystal: str) -> Tuple[np.ndarray, np.ndarray]:
    """Identify GB atoms using OVITO's CNA/coordination/CSM tools.

    Parameters:
      polycrystal: path to a LAMMPS data or dump file (passed to ovito.io.import_file)

    Returns:
      (gb_atom_ids, gb_atom_indices) two 1D numpy arrays:
        - gb_atom_ids: global atom ids (suitable for LAMMPS-style id columns)
        - gb_atom_indices: particle indices used internally by OVITO (0-based, suitable for selecting SOAP rows by index)

    Implementation notes: to avoid interference from modifying the same pipeline repeatedly, the function creates short-lived pipeline objects for each selection step for robustness.
    """
    if ovito is None:
        raise RuntimeError('OVITO Python module not available: please run this script in a Python environment with OVITO installed.')

    poly_path = str(polycrystal)

    # First compute CNA + coordination to get StructureType for each atom
    try:
        node_base = ovito.io.import_file(poly_path)
    except Exception as e:
        raise RuntimeError(f'Failed to open file with OVITO {poly_path}: {e}')

    # Apply coordination and CNA
    coord = CoordinationAnalysisModifier(cutoff=6.0, number_of_bins=200)
    cna = ovito.modifiers.CommonNeighborAnalysisModifier()
    node_base.modifiers.append(coord)
    node_base.modifiers.append(cna)

    data = node_base.compute()

    # CNA IDs: FCC == 1, HCP == 2, BCC ==3
    types = [1, 2, 3]
    counts = []

    # For robust counting, create a fresh pipeline for each StructureType selection
    for t in types:
        node_tmp = ovito.io.import_file(poly_path)
        node_tmp.modifiers.append(coord)
        node_tmp.modifiers.append(cna)
        node_tmp.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression=f"StructureType == {t}"))
        dtmp = node_tmp.compute()
        sel = dtmp.particles['Selection']
        counts.append(int(np.count_nonzero(sel.array == 1)))

    # determine majority structure type
    key = int(np.argmax(counts))
    type_max = types[key]

    # Final pipeline: coordination, cna, centro-symmetry (optionally), and selection for StructureType != type_max
    node_final = ovito.io.import_file(poly_path)
    node_final.modifiers.append(coord)
    node_final.modifiers.append(cna)
    csm = CentroSymmetryModifier()
    # 为 FCC 设置合理的邻居数（常用 12）
    try:
        csm.num_neighbors = 12
    except Exception:
        pass
    node_final.modifiers.append(csm)
    node_final.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(expression=f"StructureType != {type_max}"))

    dfinal = node_final.compute()

    sel = dfinal.particles['Selection']
    pid = dfinal.particles['Particle Identifier']

    nparticles = int(dfinal.particles.count)

    gb_atoms_ids = []
    gb_atoms_indices = []
    for index in range(nparticles):
        # Selection field uses 1 for selected
        try:
            if int(sel.array[index]) == 1:
                gb_atoms_ids.append(int(pid.array[index]))
                gb_atoms_indices.append(int(index))
        except Exception:
            # ignore entries that cannot be parsed
            continue

    return np.array(gb_atoms_ids, dtype=np.int64), np.array(gb_atoms_indices, dtype=np.int64)


def main_cli():
    parser = argparse.ArgumentParser(description='Use OVITO to compute GB atom ids and indices from a LAMMPS file')
    parser.add_argument('--lmp', '-l', type=str, default=None, help='LAMMPS file path (e.g. medium_final_atoms.lmp)')
    parser.add_argument('--out-prefix', '-o', type=str, default='gb_ids', help='Output file prefix (will produce <prefix>_ids.npy and <prefix>_indices.npy)')
    args = parser.parse_args()

    lmp = args.lmp
    if lmp is None:
        lmp = input('Please enter LAMMPS file path (e.g. medium_final_atoms.lmp): ').strip()

    if not lmp:
        print('No LAMMPS file path provided, exiting.')
        return

    p = Path(lmp)
    if not p.exists():
        print(f'Error: file not found {p}')
        return

    try:
        gb_ids, gb_indices = get_gb_ids_and_indices(str(p))
    except Exception as e:
        print(f'Error while computing GB ids: {e}')
        return

    out_prefix = args.out_prefix
    np.save(out_prefix + '_ids.npy', gb_ids)
    np.save(out_prefix + '_indices.npy', gb_indices)

    # 也写入简洁的 txt 方便查看
    with open(out_prefix + '_ids.txt', 'w', encoding='utf-8') as fh:
        for v in gb_ids:
            fh.write(str(int(v)) + '\n')

    print(f'Saved {out_prefix}_ids.npy ({gb_ids.size} entries) and {out_prefix}_indices.npy')
    print(f'Text preview saved to {out_prefix}_ids.txt')


def main_gui():
    if ovito is None:
        print('OVITO is not installed in the current Python environment; GUI is unavailable. Run in an OVITO-enabled environment or use the command-line mode.')
        return

    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception:
        print('tkinter not available, falling back to command-line mode')
        return

    root = tk.Tk()
    root.title('Generate GB ids (OVITO)')

    def browse_file():
        p = filedialog.askopenfilename(filetypes=[('Lammps files', '*.lmp;*.dump;*.data'), ('All','*.*')])
        if p:
            ent_lmp.delete(0, tk.END)
            ent_lmp.insert(0, p)

    frm = tk.Frame(root, padx=8, pady=8)
    frm.pack(fill='both', expand=True)

    tk.Label(frm, text='LAMMPS file:').grid(row=0, column=0, sticky='w')
    ent_lmp = tk.Entry(frm, width=60)
    ent_lmp.insert(0, 'medium_final_atoms.lmp')
    ent_lmp.grid(row=0, column=1)
    tk.Button(frm, text='Browse', command=browse_file).grid(row=0, column=2, padx=4)

    tk.Label(frm, text='Output prefix:').grid(row=1, column=0, sticky='w')
    ent_out = tk.Entry(frm, width=60)
    ent_out.insert(0, 'gb_ids')
    ent_out.grid(row=1, column=1)

    def on_run():
        lmp_path = ent_lmp.get().strip()
        outp = ent_out.get().strip() or 'gb_ids'
        p = Path(lmp_path)
        if not p.exists():
            messagebox.showerror('Error', f'File not found: {p}')
            return
        try:
            gb_ids, gb_indices = get_gb_ids_and_indices(str(p))
        except Exception as e:
            messagebox.showerror('Error', f'Failed to compute GB ids: {e}')
            return

        np.save(outp + '_ids.npy', gb_ids)
        np.save(outp + '_indices.npy', gb_indices)
        with open(outp + '_ids.txt', 'w', encoding='utf-8') as fh:
            for v in gb_ids:
                fh.write(str(int(v)) + '\n')

        messagebox.showinfo('Done', f'Saved {outp}_ids.npy ({gb_ids.size}) and {outp}_indices.npy')

    btn = tk.Button(frm, text='Run', command=on_run, bg='#4caf50', fg='white')
    btn.grid(row=2, column=1, pady=8, sticky='w')

    root.mainloop()


if __name__ == '__main__':
    # 与仓库中其他 generate_* 文件一致：如果有 GUI 支持则优先打开，否则使用 CLI
    if ovito is not None:
        # prefer GUI when tkinter available
        try:
            import tkinter  # type: ignore
            main_gui()
        except Exception:
            main_cli()
    else:
        main_cli()
