# Neumann–Robin error-estimate experiments

From `studies/neumann-robin`:

```bash
python nsbiot_Neumann_Robin_spatial_sweep.py --smoke
python nsbiot_Neumann_Robin_temporal_sweep.py --smoke
```

For the full manufactured studies, run without `--smoke`. Use `--spatial-n` and `--spatial-dt` for spatial refinement, or `--temporal-n` and `--temporal-mesh-intervals` for temporal refinement. Use `--dry-run` to inspect the chosen cases. The shared second-order examples are documented in the [FPSI guide](../fpsi-error-estimates/README.md).
