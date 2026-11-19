# Penalized Naghdi shell refinement

From `studies/penalized-shell`:

```bash
python run_penalized_convergence_studies.py --dry-run
python run_penalized_convergence_studies.py
```

The dry run writes a manifest under `penalized_convergence_results/` and prints the planned commands. The full driver generates three mesh levels, runs five distinct cases and analyzes their pressure profiles. The medium mesh at 25 microseconds is shared by the spatial and temporal sweeps. Use `--study-dir` to select an independent output directory and `--write-field-output` to save PVD collections.

The included baseline mesh can be used directly with `fsi_naghdi_pressure_pulse_penalized_convergence.py --mesh-dir .`. The `analyze_convergence*.py` scripts analyze pressure and diameter profiles. Run each with `--help` for input and output arguments.
