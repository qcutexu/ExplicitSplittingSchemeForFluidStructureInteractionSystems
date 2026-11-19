#!/usr/bin/env python3
"""Small, resource-labeled NR/monolithic benchmark; no automatic expensive sweep."""
import argparse,csv,json,os,platform,statistics,subprocess,sys,time
from pathlib import Path


def write(path,rows):
    if not rows:return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--python',default=sys.executable)
 p.add_argument('--mesh',type=int,default=16);p.add_argument('--dts',type=float,nargs='+',default=[.125,.0625]);p.add_argument('--T',type=float,default=.5)
 p.add_argument('--repeats',type=int,default=5);p.add_argument('--target-error',type=float,default=1e-4)
 p.add_argument('--output-dir',type=Path,default=Path('validation/parallel_implementation/benchmark'))
 a=p.parse_args();root=Path(__file__).resolve().parent;out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 if a.repeats<1:raise ValueError('At least one repeat required')
 env=os.environ.copy();env.update(FI_PROVIDER='tcp',PATH=str(Path(a.python).parent)+':'+env['PATH'],
  DIJITSO_CACHE_DIR=str(root/'validation/cache/dijitso'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1')
 variants=[('serial_nr','serial','nr',1),('serial_monolithic','serial','monolithic',1),('mpi2_nr','mpi','nr',2)]
 records=[]
 def run(dt,variant,trial,warmup):
  name,execution,method,ranks=variant;folder=out/f'dt_{dt:g}'/name/('warmup' if warmup else f'trial_{trial}')
  cmd=[a.python,str(root/'nsbiot_run.py'),'--execution',execution,'--method',method,'--mesh',str(a.mesh),'--dt',str(dt),'--T',str(a.T),'--forcing','manufactured','--linear-solver','umfpack','--no-field-output','--output-dir',str(folder)]
  if ranks>1:cmd=[str(Path(a.python).parent/'mpiexec'),'-n',str(ranks),*cmd]
  print('RUN',name,'dt',dt,'warmup' if warmup else f'trial {trial}',flush=True)
  start=time.perf_counter();result=subprocess.run(cmd,env=env,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);external=time.perf_counter()-start
  folder.mkdir(exist_ok=True,parents=True);(folder/'execution.log').write_text(result.stdout)
  if result.returncode:raise RuntimeError(result.stdout[-2500:])
  data=json.loads((folder/'summary.json').read_text())
  row={'variant':name,'trial':trial,'is_warmup':warmup,'external_process_wall_s':external,**data}
  records.append(row);write(out/'runs.csv',records)
 for dt in a.dts:
  for variant in variants:run(dt,variant,0,True)
 for trial in range(1,a.repeats+1):
  for dt in (a.dts if trial%2 else list(reversed(a.dts))):
   for variant in (variants if trial%2 else list(reversed(variants))):run(dt,variant,trial,False)
 groups=[]
 for dt in a.dts:
  for variant,_,_,ranks in variants:
   values=[r for r in records if not r['is_warmup'] and r['variant']==variant and r['dt']==dt]
   row={'variant':variant,'nm':a.mesh,'dt':dt,'T':a.T,'ranks':ranks,'linear_solver':'umfpack','repeats':len(values)}
   for metric in ('wall_s','integration_wall_s','matrix_assembly_s','rhs_assembly_s','linear_solve_s','communication_s','diagnostics_s','allocated_rank_seconds','external_process_wall_s','err_u','err_p','err_eta','err_xi','err_phi'):
    sample=[r[metric] for r in values]
    row[metric+'_median']=statistics.median(sample)
    if metric in ('wall_s','integration_wall_s'):
     row[metric+'_min']=min(sample);row[metric+'_max']=max(sample)
     row[metric+'_stdev']=statistics.stdev(sample) if len(sample)>1 else 0.
   groups.append(row)
 write(out/'summary.csv',groups)
 fastest=[]
 for variant,_,_,ranks in variants:
  eligible=[r for r in groups if r['variant']==variant and r['err_u_median']<=a.target_error]
  if eligible:
   best=min(eligible,key=lambda r:r['wall_s_median'])
   fastest.append({'variant':variant,'ranks':ranks,'target_velocity_L2_error':a.target_error,'dt':best['dt'],
      'achieved_error':best['err_u_median'],'median_wall_s':best['wall_s_median'],
      'note':'Fastest of tested settings only; fixed mesh and velocity-only tolerance; polynomial MMS limits general work-precision claims'})
  else:fastest.append({'variant':variant,'ranks':ranks,'target_velocity_L2_error':a.target_error,'dt':None,'achieved_error':None,'median_wall_s':None,'note':'No tested setting qualifies'})
 write(out/'cost_at_velocity_tolerance.csv',fastest)
 protocol={'mesh':a.mesh,'dts':a.dts,'T':a.T,'repeats':a.repeats,'warmups_per_setting':1,'order':'alternating variant and step order',
   'physical':'unit coefficients, L=gamma=1, beta=0; P2/P1 and P2/P2/P2; exact projected starts',
   'backend':'UMFPACK in every case; MPI NR uses one rank per block; monolithic remains serial',
   'resource_comparison':'serial NR vs serial monolithic isolates fixed-configuration one-process algorithm cost; mpi2 NR vs serial NR measures resource-labeled parallel acceleration',
   'threads':{k:env[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS')},
   'machine':platform.platform(),'architecture':platform.machine(),'target_error':a.target_error,
   'timing':'wall_s excludes launch/import and final CSV/JSON serialization; external_process_wall_s includes them',
   'limitations':['small mesh, not large-scale speedup','polynomial manufactured time dependence','velocity tolerance is not a full-solution tolerance','two tested steps are not an optimized work-precision search']}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 print('SAVED',out,flush=True)

if __name__=='__main__':main()
