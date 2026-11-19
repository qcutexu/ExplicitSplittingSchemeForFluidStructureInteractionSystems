#!/bin/bash
set -eu
cd "$(dirname "$0")"
FENICS_PREFIX=${FENICS_PREFIX:-${CONDA_PREFIX:?Activate the FEniCS environment or set FENICS_PREFIX}}
export PATH="$FENICS_PREFIX/bin:$PATH"
export FI_PROVIDER=tcp
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export DIJITSO_CACHE_DIR="$PWD/validation/cache/dijitso"
exec "$FENICS_PREFIX/bin/python" nsbiot_monolithic.py "$@"
