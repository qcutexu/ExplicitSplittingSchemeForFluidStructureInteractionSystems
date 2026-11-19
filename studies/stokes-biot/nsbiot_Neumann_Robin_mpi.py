#!/usr/bin/env python3
"""MPI entry point for the fixed-domain beta=0 BDF2/EX2 scheme."""
from nsbiot_run import main
if __name__=='__main__':
    main(default_execution='mpi',default_method='nr')
