# Naghdi shell pressure pulse

From `studies/naghdi-shell`, run one time step for each method using the included matched fluid and shell meshes.

```bash
python fsi_naghdi_pressure_pulse.py --mesh-dir meshes --beta 1 \
  --dt 0.0001 --final-time 0.0001 --output-dir results_beta_smoke
python fsi_naghdi_pressure_pulse_explicit_scheme.py --mesh-dir meshes --robin-parameter 1000 \
  --dt 0.0001 --final-time 0.0001 --output-dir results_robin_smoke
python fsi_naghdi_pressure_pulse_monolithic.py --mesh-dir meshes \
  --dt 0.0001 --final-time 0.0001 --output-dir results_monolithic_smoke
```

For a pressure-pulse comparison, use `--final-time 0.024 --pressure-stabilization 0.01 --profile-bins 61` for all three methods and choose separate output directories. For Robin parameter sensitivity, vary `--robin-parameter` at fixed time step. For time refinement, hold the Robin parameter fixed and use time steps of 100, 50 and 25 microseconds. Compare CSV profiles at common physical times and use PVD files to inspect fields.

The stented driver uses the fitted shell material tags and reference geometry bundled with the stented case:

```bash
python fsi_naghdi_stented_pressure_pulse.py --scheme explicit \
  --mesh-dir stented_onyx_uniform_study/meshes --dt 0.0001 \
  --final-time 0.0001 --output-dir results_stented_smoke
```

Select `--scheme monolithic` for its coupled solver. `generate_bukac_benchmark_meshes.py` and `fluid_from_shell_surface.geo` support regenerating the unstented benchmark mesh.
