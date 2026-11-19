# Stokes–Biot comparisons

From `studies/stokes-biot`, plan a temporal comparison and run a small manufactured test:

```bash
python nsbiot_comparison_experiments.py --study temporal --dry-run
python nsbiot_comparison_experiments.py --study temporal --mesh 8 \
  --dts 0.125 0.0625 --phi-degree 2 --output-dir out_temporal
python nsbiot_plot_comparison.py out_temporal
```

The comparison driver covers temporal, spatial, energy, stability, parameter and cost studies. Standalone Robin–Robin sweep scripts are available for temporal and spatial refinement. Use `nsbiot_run.py --help` for the 2D Neumann–Robin and monolithic methods. The shell launchers use the active Conda environment or an explicit `FENICS_PREFIX`. Set `RANKS` for the MPI launchers.

Try the 3D manufactured geometry with either `--method nr` or `--method monolithic`:

```bash
python nsbiot_3d_run.py --method nr --mesh-source fixture --T 0.001 \
  --output-dir results/fixture_nr
```

The standalone `stokes_biot_3d_single.py` embeds its own configuration. Use `--mesh-source artery` for the included geometry under `FSI_parallel_3D/mesh/`. Use `--mesh-dir` to select another directory with the same XDMF/HDF5 layout.

The cylinder driver can generate its own mesh. For a short monolithic pressure-wave run:

```bash
mpiexec -n 2 python stokes_biot_cylinder_wave.py --method monolithic \
  --mesh-generator layered --angular 8 --fluid-radial 1 --wall-radial 1 --axial 8 \
  --dt 0.0001 --T 0.0002 --output-dir results/cylinder_smoke
```

Use `--mesh-only` to generate the mesh without solving. The default unstructured Gmsh mesh is larger than the layered example. The cylinder driver writes field data, checkpoints and probe diagnostics.
