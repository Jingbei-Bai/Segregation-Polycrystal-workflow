# Segregation / Polycrystal workflow

This repository contains a small workflow for generating polycrystals, preparing LAMMPS input, extracting grain-boundary (GB) atoms, computing SOAP descriptors, and training simple ML models.

---

Contents
- workflow_runner.py — Tkinter GUI that integrates the full workflow (the main control panel).
- generate_polycrystal.py — Creates polycrystal description files intended for use with atomsk --polycrystal.
- generate_lammps_in.py — Creates a LAMMPS input script (`in.lmp`) from a read_data line and an append file.
- generate_get_gb_ids.py — Optional OVITO helper to detect GB atoms (requires OVITO Python API).
- generate_ml_with_soap.py — Compute SOAP features (requires ASE + dscribe) and train basic models.
- generate_skew_data.py — Helpers for skew-fitting / plotting the energy distribution.
- Example data files: aluminium.xsf, AlCu.eam, Al_Cu.txt, dump.atom, polycrystal.lmp, etc.

What's been fixed
- Fixed syntax/indentation errors in `workflow_runner.py` (removed stray diff markers and corrected mixed indents).
- Ensured the GUI class `WorkflowApp` is syntactically valid and maintains previous behavior (command-preparation but not forced execution of external binaries).

Notes about other files
- I ran static checks across the Python files in the repository; there were only warnings reported (unused imports, type-check hints, or messages related to optional third-party modules such as OVITO/dscribe/ASE). Warnings were left as non-blocking since your request was to fix bugs (errors) rather than stylistic or unused-import warnings.
- `generate_get_gb_ids.py` and `generate_ml_with_soap.py` contain code that interfaces with optional third-party packages (OVITO, ASE, dscribe). Those modules are imported inside functions and guarded with try/except in most places; missing packages will raise runtime errors only when those features are invoked. This is intentional design so the basic parts of the project can be used without installing large scientific stacks.

Requirements (recommended)
- Python 3.8+ (scripts use pathlib and typing-friendly idioms; tested with 3.10+ in dev)
- Optional packages for full functionality:
  - numpy
  - ase (for reading LAMMPS dumps)
  - dscribe (for SOAP descriptors)
  - scikit-learn (for ML training)
  - matplotlib, pillow (for plotting and GUI image display)
  - ovito (optional — only required if using the OVITO-based GB detection helper)

Install minimal Python deps (recommended, but optional):

For a minimal experience (numpy + pillow + matplotlib + scikit-learn):

    python -m pip install numpy pillow matplotlib scikit-learn

For full SOAP & OVITO features:

    python -m pip install ase dscribe ovito

(OVITO may require its own installer or conda package depending on platform.)

Usage — command line utilities (no external binaries executed automatically)
- generate_polycrystal.py
  - Generates a `polycrystal.txt` file; does not call atomsk.
  - Example:

      python generate_polycrystal.py -b 70 -r 20 -o polycrystal.txt

  - The script prints or writes suggested atomsk commands which you can run manually in a terminal that has atomsk installed.

- generate_lammps_in.py
  - Builds a LAMMPS input `in.lmp` from a read_data spec and an append file (for potentials/thermo settings).
  - Example:

      python generate_lammps_in.py "polycrystal.lmp extra/atom/types 1" Al_Cu.txt AlCu.eam -o in.lmp

  - This script does not automatically call LAMMPS; it prepares `in.lmp` for manual execution.

- GUI: workflow_runner.py
  - Launch the integrated GUI if your environment has tkinter available.
  - The GUI will prepare commands, run only local Python functions, and may optionally invoke atomsk or LAMMPS if you explicitly press the corresponding buttons — but by default the code prepares the commands and logs them. When the GUI tries to run atomsk or LAMMPS, the code checks for the executable and will show an error message if it is not found.
  - To start the GUI (from a cmd.exe prompt):

      python workflow_runner.py

  - If tkinter is not installed, the script will exit with a helpful message.

Safety note: this project was intentionally implemented so external commands are shown to users and not executed silently. When using the GUI, if you run atomsk or LAMMPS from the GUI those actions are done in background threads but the code explicitly checks for missing executables and warns the user via dialog boxes.

Extensibility & Integration

- Descriptor modularity: The feature-computation code is written modularly so different atomic descriptors can be plugged in with minimal changes. The current implementation exposes a SOAP entry point (via `compute_soap_features` in `generate_ml_with_soap.py`); to add ACSF (Atom-Centered Symmetry Functions) or other descriptors, implement the same function signature (accepting a dump/path and returning a NumPy 2D array of per-atom feature vectors) and either replace or register it in the workflow. This allows swapping or comparing descriptors (SOAP, ACSF, Behler–Parrinello, etc.) without changing the training pipeline.

- Machine-learning algorithms: The training pipeline uses the scikit-learn API and works with any estimator that implements the standard `fit`/`predict` interface. That means you can use any method from `scikit-learn` (linear models, tree ensembles, kernel methods, SVR, GaussianProcess, pipelines, etc.) or compatible third-party estimators. The `train_and_save_models` helper centralizes model training and persists results so adding new estimators is a matter of registering them in that helper.

- OVITO-based GB detection: The OVITO helper (`generate_get_gb_ids.py`) is implemented as a thin interface around the OVITO Python API. Any GB-detection or selection method that OVITO supports (site property-based selection, common neighbor analysis, centro-symmetry, coordination, custom modifiers, or scripted workflows) can be integrated into the helper. Extend or replace the `get_gb_ids_and_indices` implementation to call the desired OVITO modifiers and return GB atom ids/indices for downstream processing.


License / Attribution
- No license file is provided in this repository. If you intend to publish or share, consider adding a LICENSE file (MIT/Apache/BSD as appropriate).

