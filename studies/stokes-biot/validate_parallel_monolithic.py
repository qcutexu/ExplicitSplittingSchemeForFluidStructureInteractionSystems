"""Small independent checks for the new block, MPI and monolithic solvers."""
import csv,json,sys
from pathlib import Path
import numpy as np
from fenics import *
from mpi4py import MPI as mpi
from nsbiot_blocks import Settings,FluidBlock,BiotBlock,owned_records
from nsbiot_monolithic import MonolithicSystem
from nsbiot_compare_core import solve_case

OUT=Path('validation/parallel_implementation');OUT.mkdir(exist_ok=True)
checks={}
# Interface RHS cancellation and independence from artificial L,beta.
matrices=[]
for L,beta in ((1.,0.),(.5,0.),(2.,0.),(1.,1.)):
 cfg=Settings(nm=4,dt=.125,T=.375,L=L,beta=beta).validate()
 fluid=FluidBlock(cfg,mpi.COMM_SELF);biot=BiotBlock(cfg,mpi.COMM_SELF)
 mono=MonolithicSystem(fluid,biot)
 error=mono.check_bulk_rhs()
 assert error<1e-12,error
 checks[f'bulk_rhs_L{L}_beta{beta}']=error
 matrices.append(mono.matrix)
for i,matrix in enumerate(matrices[1:],1):
 difference=matrix-matrices[0]
 error=float(abs(difference.data).max()) if difference.nnz else 0.
 assert error<1e-12,error
 checks[f'monolithic_matrix_partition_parameter_cancellation_{i}']=error
# Refactored production forms reproduce historical results.
old=list(csv.DictReader(Path('validation/energy/energy.csv').open()))
ref=solve_case(.0625,8,.5,2,beta=0.,forcing='zero',initialization='duplicate',initial_mode='interface',monitor=True)
error=max(abs(r['energy']-float(next(v for v in old if v['scheme']=='neumann_robin' and float(v['dt'])==.0625 and float(v['t'])==r['t'])['energy'])) for r in ref['energy_history'])
assert error<1e-16,error;checks['historical_energy_regression']=error
# Save an independent reference-core trajectory and canonical final fields.
for forcing in ('manufactured','zero'):
 fields={}
 def observer(context):
  if context['step']==3:
   for name,fn in context['fields'].items():fields[name]=owned_records(fn)
 mode='smooth' if forcing=='manufactured' else 'interface'
 r=solve_case(.125,8,.375,2,beta=0.,forcing=forcing,initialization='projected' if forcing=='manufactured' else 'duplicate',initial_mode=mode,monitor=True,observer=observer)
 for name,array in fields.items():fields[name]=array[np.lexsort((array[:,2],array[:,1],array[:,0]))]
 np.savez_compressed(OUT/f'core_{forcing}_fields.npz',**fields)
 (OUT/f'core_{forcing}_summary.json').write_text(json.dumps(r,indent=2)+'\n')
(OUT/'form_and_matrix_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
