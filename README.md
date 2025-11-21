# Segregation / Polycrystal workflow

A small, self-contained workflow for generating polycrystals, preparing LAMMPS input files, extracting grain-boundary (GB) atoms, computing SOAP descriptors, and training simple machine‑learning regressors.

This repository contains a set of command-line utilities and a lightweight Tkinter GUI to help you prepare inputs and run lightweight analyses. External tools such as Atomsk, LAMMPS and OVITO are used only when you explicitly run them — the project does not execute external binaries silently.

Contents
- `workflow_runner.py` — Tkinter GUI that integrates the main workflow (control panel).
- `generate_polycrystal.py` — Create polycrystal description files (output suitable for atomsk's `--polycrystal`).
- `generate_lammps_in.py` — Build a LAMMPS `in.lmp` file from a `read_data` specification plus an append file (potentials / thermo settings).
- `generate_get_gb_ids.py` — OVITO-based helper to detect GB atoms and return atom ids/indices (optional; requires OVITO Python API).
- `generate_ml_with_soap.py` — Compute SOAP descriptors (ASE + dscribe) and train simple ML models (scikit-learn).
- `generate_skew_data.py` — Helpers for skew-fitting and plotting the segregation energy distribution.
- Example data files: `aluminium.xsf`, `AlCu.eam`, `Al_Cu.txt`, `dump.atom`, `polycrystal.lmp`, etc.

What I updated
- Added a `requirements.txt` that lists core and optional dependencies (see below).
- Rewrote this `README.md` into a concise GitHub-style project overview with installation and usage examples.

Quick links
- Requirements: `requirements.txt`
- GUI entry point: `workflow_runner.py`

Table of contents
- Features
- Requirements
- Installation
- Usage examples (CLI + GUI)
- Files & workflow
- Contributing
- License

Features
- Generate polycrystal descriptors suitable for atomsk (does not call atomsk automatically).
- Produce a LAMMPS `in.lmp` template from a `read_data` specification and append file.
- Extract GB atom ids and indices using OVITO (optional).
- Compute SOAP descriptors for atoms (using ASE + dscribe) and train simple scikit-learn regressors (LinearRegression by default).
- Plot skewed segregation energy distributions and save model artifacts and plots.

Requirements
- Python: 3.8+ (3.10 tested during development).
- Core Python packages (used by most scripts):
  - numpy
  - scikit-learn
  - joblib
  - matplotlib
- Optional packages (enable additional features):
  - ase, dscribe — compute SOAP descriptors from LAMMPS dump files
  - scipy — fit skew-normal distributions and related statistics
  - ovito — compute GB atom ids/indices via OVITO's Python API
  - pillow — image handling in GUI

A `requirements.txt` file has been added to the repository with a recommended set of core and optional dependencies. Note that some packages (notably OVITO) are best installed via their recommended installers or conda packages on certain platforms.

Installation

1) Create and activate a virtual environment (recommended):

   python -m venv .venv
   .\.venv\Scripts\activate  # Windows (cmd.exe)

2) Install the minimal/core dependencies:

   python -m pip install -r requirements.txt

3) Install optional packages for full functionality (examples):

   # via pip (if available for your platform)
   python -m pip install ase dscribe scipy

   # OVITO: follow OVITO's official install instructions (may require conda or OVITO installer)

Security / safety note
- The GUI and scripts prepare external commands (e.g. for atomsk or LAMMPS) but do not run them automatically without your action. When the GUI runs external binaries it checks the executable is available and reports errors in dialogs/log. Always review prepared commands before executing them.

Usage examples

- generate_polycrystal.py
  - Generates a `polycrystal.txt` description (does not call atomsk automatically). Example:

      python generate_polycrystal.py -b 70 -r 20 -o polycrystal.txt

  - The generated file usually contains suggested `atomsk` command lines; you can run those manually (or use the GUI helper which will prepare commands and optionally run atomsk if available).

- generate_lammps_in.py

      python generate_lammps_in.py "polycrystal.lmp extra/atom/types 1" Al_Cu.txt AlCu.eam -o in.lmp

  - Produces an `in.lmp` file ready for LAMMPS; the script does not call LAMMPS automatically.

- generate_get_gb_ids.py (OVITO required)
  - Use OVITO's Python API to detect grain-boundary atoms and save `gb_ids.npy` / `gb_indices.npy`.

      python generate_get_gb_ids.py --lmp medium_final_atoms.lmp -o gb_ids

- generate_ml_with_soap.py
  - Compute SOAP features for GB atoms and train simple regressors (requires ASE + dscribe to compute SOAP):

      python generate_ml_with_soap.py --energy Al_Cu.txt --dump medium_final_atoms.dump --gb-ids gb_ids.npy --out-dir ml_output

  - If you have an LAMMPS file but no dump, the script can attempt to call Atomsk to generate a dump (Atomsk must be installed and available in PATH).

- generate_skew_data.py
  - Fit and plot a skew-normal distribution for the segregation energy of GB atoms.

      python generate_skew_data.py --input Al_Cu_eam.txt --gb-ids gb_ids.npy --out segregation_skew_plot.pdf

GUI
- Start the Tkinter GUI (requires tkinter available in your Python runtime):

    python workflow_runner.py

The GUI prepares commands and runs local Python helpers; it will also attempt to run atomsk or LAMMPS only when you explicitly request those operations via the UI and when the executable is available.

Files & workflow overview
- Typical flow:
  1) Create polycrystal description with `generate_polycrystal.py`.
  2) Use Atomsk (manually or via GUI) to convert to a LAMMPS `.lmp` file.
  3) Generate a LAMMPS `in.lmp` with `generate_lammps_in.py` and run LAMMPS separately.
  4) Use OVITO (optional) to detect GB atoms and save `gb_ids.npy`/`gb_indices.npy`.
  5) Compute SOAP for GB atoms with `generate_ml_with_soap.py` and train ML models.
  6) Visualize results with `generate_skew_data.py` or the GUI.

Contributing
- Bug reports, pull requests and suggestions are welcome. Please include a short description, steps to reproduce, and (if relevant) small test data. Consider opening issues first to discuss larger changes.

License
- No license.

Acknowledgements & notes
- Parts of the code rely on optional third-party scientific libraries (ASE, dscribe, OVITO) that may require specific installation instructions for your platform. The code is written so missing optional packages raise runtime errors only when those features are used.
- If you want me to also add an example `pyproject.toml`/`setup.cfg` or CI workflow (GitHub Actions) for tests/linting, tell me and I can add that next.
