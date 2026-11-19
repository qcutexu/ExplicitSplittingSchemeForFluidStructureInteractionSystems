# Source provenance

| Original folder | Destination | Selection |
| --- | --- | --- |
| `NaghdiExplicitSplittingScheme` | `studies/naghdi-shell/` | Pressure-pulse drivers from `corrected_wall_rerun/source`, baseline mesh generation and inputs, and the stented solver with its fitted-mesh inputs |
| `penalized_naghdi_shell_pressure_wave_convergence_study` | `studies/penalized-shell/` | Solver, convergence orchestration, analysis scripts and baseline mesh |
| `stokes_biot_robin_comparison_package` | `studies/stokes-biot/` | Main 2D/3D drivers and dependencies, comparison tools, reference scripts the supplied arterial meshes and the modular 3D boundary configuration |
| `Neumann_Robin_ErrorEstimate_paper_numericalexample_with_Lee` | `studies/neumann-robin/` | Dedicated Neumann–Robin sweeps |
| `ErrorEstimateFPSI_paper_numericalexample` | `studies/fpsi-error-estimates/` | Backward Euler and BDF2/AB2 examples. The four shared second-order drivers were byte-identical across the two error-estimate folders. |

The Naghdi split drivers in the corrected-wall snapshot classify fluid exterior facets by midpoint. Use that snapshot for new unstented comparisons. The source collection also contains historical runs, logs, manuscripts, and machine-specific plotting and rerun pipelines. These remain in the original folders. The included guides cover running the selected examples and inspecting newly generated data. Historical figure layouts depend on the plotting pipelines and inputs retained in the original folders.

The Stokes–Biot launchers and benchmark driver use the active environment in place of machine-specific interpreter paths. The working files retain the original numerical methods. No distribution license has been selected.
