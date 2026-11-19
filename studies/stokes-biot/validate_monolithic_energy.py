"""Independent homogeneous BDF2 bulk-energy identity for the monolithic reference."""
import csv,json,math
from pathlib import Path
from fenics import *
from mpi4py import MPI as mpi
from nsbiot_blocks import Settings,FluidBlock,BiotBlock,assemble_rhs,collect_records
from nsbiot_monolithic import MonolithicSystem
from nsbiot_forms import epsilon

out=Path('validation/parallel_implementation')
cfg=Settings(nm=8,dt=1/16,T=.5,forcing='zero',initial_mode='interface').validate()
f=FluidBlock(cfg,mpi.COMM_SELF);b=BiotBlock(cfg,mpi.COMM_SELF);system=MonolithicSystem(f,b)
previous=f.energy(f.old,f.older)+b.energy(b.old,b.older);initial=previous;rows=[]
for step in range(2,cfg.steps+1):
 system.solve(system.rhs(step));f.update();b.update()
 now=f.energy(f.current,f.old)+b.energy(b.current,b.old)
 uf=f.current['u']-2*f.old['u']+f.older['u']
 xb={name:b.current[name]-2*b.old[name]+b.older[name] for name in b.names}
 defect=.25*float(assemble(inner(uf,uf)*f.g['dx'])+assemble((inner(xb['xi'],xb['xi'])+xb['phi']**2+2*inner(epsilon(xb['eta']),epsilon(xb['eta']))+div(xb['eta'])**2)*b.g['dx']))
 trace=collect_records(b.current['xi'],b.comm,True);f.proxies['xi'].receive(trace,0)
 slip=f.current['u']-f.proxies['xi'].functions[0]
 diss=float(assemble(2*inner(epsilon(f.current['u']),epsilon(f.current['u']))*f.g['dx'])+
            assemble(inner(grad(b.current['phi']),grad(b.current['phi']))*b.g['dx'])+
            assemble(dot(slip,f.g['tangent'])**2*f.g['ds'](5)))
 residual=now-previous+defect+cfg.dt*diss
 assert abs(residual)/initial<1e-10,(step,residual/initial)
 rows.append({'step':step,'t':step*cfg.dt,'energy':now,'energy_increment':now-previous,
              'BDF2_history_defect':defect,'dt_physical_dissipation':cfg.dt*diss,
              'energy_identity_residual':residual,'relative_to_initial':residual/initial})
 f.shift();b.shift();previous=now
with (out/'monolithic_energy_identity.csv').open('w',newline='') as file:
 writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
print('PASS monolithic energy identity; maximum residual / initial energy =',max(abs(r['relative_to_initial']) for r in rows))
