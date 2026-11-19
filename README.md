# FSI Splitting Schemes

Research drivers for fluid–structure interaction (FSI) and fluid–poroelastic structure interaction (FPSI). The collection covers Naghdi shell pressure pulses, Stokes–Biot coupling, and convergence experiments for splitting error estimates.

| Study | Contents |
| --- | --- |
| [Naghdi shell](studies/naghdi-shell/README.md) | β-splitting, explicit Robin–Robin, monolithic coupling, and an idealized stented shell |
| [Penalized shell](studies/penalized-shell/README.md) | Spatial and temporal refinement with pressure and diameter profile analysis |
| [Stokes–Biot](studies/stokes-biot/README.md) | Robin–Robin and Neumann–Robin comparisons, monolithic references, MPI drivers, and 3D pressure waves |
| [Neumann–Robin error estimates](studies/neumann-robin/README.md) | Dedicated spatial and temporal sweeps |
| [FPSI error estimates](studies/fpsi-error-estimates/README.md) | Backward Euler and BDF2/AB2 examples, including ALE blood-flow drivers |

Start with [environment setup and verification](docs/setup.md). Run commands from the study directory shown in each guide. See [source provenance](docs/provenance.md) for the source folders, version choices, and reproduction scope.
