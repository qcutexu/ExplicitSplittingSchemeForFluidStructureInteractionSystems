# FPSI error-estimate experiments

From `studies/fpsi-error-estimates`, start with two short manufactured Backward Euler tests:

```bash
python NS_Biot_proof_compatible_mpi_BackwardEuler_sweeps_ex1.py --smoke
python NS_Biot_proof_compatible_mpi_BackwardEuler_sweeps_ex2.py --smoke
```

Use `--mode temporal` or `--mode spatial` for a single sweep. The `ex2_variableDt` variant exposes spatial time-step scaling; run it with `--help` for the controls. Use the matching MPI launcher when comparing ranks.

The second-order manufactured sweeps live in `NSBiot2026_2ndOrder/`:

```bash
cd NSBiot2026_2ndOrder
python nsbiot_2nd_BDF2_AB2_spatial_sweep_OK.py
python nsbiot_2nd_BDF2_AB2_temporal_sweep_OK.py
```

The `bloodflow_*.py` scripts run first- and second-order ALE examples. Their case parameters are set in the drivers. Run each script from its own directory and move the previous outputs before rerunning because the drivers use fixed XDMF names. The first-order profiles variant also generates axial wave profile data.
