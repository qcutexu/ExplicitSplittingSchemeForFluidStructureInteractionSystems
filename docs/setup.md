# Environment and verification

The solvers use legacy FEniCS/DOLFIN 2019.1 and its matching UFL, FFC and dijitso components. Use Python 3, NumPy, SciPy, Matplotlib, mpi4py and petsc4py in the same environment. Use an MPI launcher from that environment to match the MPI implementation linked into DOLFIN, PETSc and mpi4py. The solver examples use PETSc with MUMPS or SuiteSparse/UMFPACK depending on the driver. Install a working C++ compiler for form compilation.

Check the active environment with:

```bash
python -c 'import dolfin, mpi4py, numpy, petsc4py, scipy; print(dolfin.__version__)'
python -c 'from dolfin import has_lu_solver_method as has; print({s: has(s) for s in ("mumps", "umfpack")})'
```

Mesh generation can require the `gmsh` command-line tool or the `gmsh` Python module. The shell mesh generators write DOLFIN XML directly and can also write XDMF/HDF5 when h5py is available. ParaView can open the PVD and XDMF output. Keep each XDMF file beside its companion HDF5 file. Use fresh output directories for independent runs.

Start with the Stokes–Biot comparison dry run (requires NumPy) and the penalized-shell study dry run (standard library only). They verify CLI parsing and case construction. Then use the documented short solver cases to check assembly, linear solvers and output in your environment.
