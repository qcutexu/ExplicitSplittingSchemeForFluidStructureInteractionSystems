#!/usr/bin/env python3
"""Parallel 3D fixed-reference Stokes–Biot NR or monolithic BDF2 solver.

Use --mesh-source artery explicitly for the supplied 4.52-million-DOF meshes.
The default adjacent-cube fixture is only a small implementation smoke test.
"""
import argparse,csv,json,time,os
from pathlib import Path
import numpy as np
from fenics import *
from mpi4py import MPI as mpi
from nsbiot_3d_config import load_config
from nsbiot_3d_blocks import Block3D
from nsbiot_3d_algebra import Coupling3D,NRSystem3D,MonolithicSystem3D

ROOT=Path(__file__).resolve().parent

def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class FieldOutput3D:
    """ParaView time series plus exact native finite-element HDF5 fields."""
    def __init__(self,blocks,output,xdmf_every,checkpoint_every):
        if xdmf_every<0 or checkpoint_every<0:raise ValueError('Output intervals must be nonnegative')
        self.blocks=blocks;self.xdmf_every=xdmf_every;self.checkpoint_every=checkpoint_every
        self.xdmf={};self.hdf5={};self.times=[]
        for b in blocks:
            folder=output/b.kind
            if b.comm.rank==0:folder.mkdir(exist_ok=True)
            b.comm.Barrier()
            if xdmf_every:
                f=XDMFFile(b.comm,str(folder/'fields.xdmf'))
                f.parameters['functions_share_mesh']=True;f.parameters['flush_output']=True;f.parameters['rewrite_function_mesh']=False
                self.xdmf[b.kind]=f
            if checkpoint_every:
                h=HDF5File(b.comm,str(folder/'fields_checkpoint.h5'),'w');h.write(b.mesh,'/mesh');self.hdf5[b.kind]=h
    def write(self,step,t,final=False,level='new'):
        visual=bool(self.xdmf_every and (step==0 or step%self.xdmf_every==0 or final))
        exact=bool(self.checkpoint_every and (step==0 or step%self.checkpoint_every==0 or final))
        if visual or exact:
            for b in self.blocks:
                fields=getattr(b,level).split(True)
                for name,field in zip(b.names,fields):
                    field.rename(name,name)
                    if visual:self.xdmf[b.kind].write(field,float(t))
                    if exact:
                        path=f'/{name}/step_{step:08d}'
                        self.hdf5[b.kind].write(field,path)
                        attrs=self.hdf5[b.kind].attributes(path);attrs['time']=float(t);attrs['step']=int(step)
                if exact:self.hdf5[b.kind].flush()
        self.times.append({'step':step,'t':t,'paraview_written':visual,'native_checkpoint_written':exact})
    def close(self):
        for f in self.xdmf.values():f.close()
        for f in self.hdf5.values():f.close()

def run(cfg,method,output,dt,T,mesh_source='fixture',fixture_n=1,biot_fixture_n=None,phi_degree=2,quadrature_order=4,mesh_dir=None,save_fields=True,xdmf_every=1,projection_refinement=3,checkpoint_every=1):
    w=mpi.COMM_WORLD;rank=w.rank;size=w.size
    if dt<=0 or T<=0 or not np.isfinite(dt+T):raise ValueError('dt and T must be finite positive')
    steps=round(T/dt)
    if steps<1 or not np.isclose(steps*dt,T,rtol=0,atol=1e-12):raise ValueError('dt must divide T')
    if method=='nr' and size>1 and size%2:raise ValueError('NR requires an even world size or a serial reference run')
    if phi_degree not in (1,2):raise ValueError('phi degree must be 1 or 2')
    if not has_lu_solver_method('mumps'):raise ValueError('MUMPS backend is required explicitly')
    if rank==0:output.mkdir(parents=True,exist_ok=True)
    w.Barrier();start=time.perf_counter()
    if method=='nr' and size>1:
        kind='fluid' if rank<size//2 else 'biot';comm=w.Split(0 if kind=='fluid' else 1,rank);kinds=[kind]
    else:comm=w;kinds=['fluid','biot']
    blocks=[]
    for kind in kinds:
        n=(biot_fixture_n or fixture_n) if kind=='biot' else fixture_n
        blocks.append(Block3D(kind,comm,cfg,dt,mesh_dir,None if mesh_source=='artery' else n,phi_degree))
    t0=time.perf_counter();coupling=Coupling3D(blocks,w,cfg,phi_degree,quadrature_order,projection_refinement);interface_setup=w.allreduce(time.perf_counter()-t0,op=mpi.MAX)
    if rank==0:
        (output/'interface_audit.json').write_text(json.dumps(coupling.meta['audit'],indent=2)+'\n')
        print('Interface ready:',json.dumps(coupling.meta['audit']),flush=True)
    systems=[MonolithicSystem3D(blocks,coupling)] if method=='monolithic' else [NRSystem3D(b,coupling) for b in blocks]
    def snapshots(level,name):
        data={}
        for block in blocks:data.update(block.snapshot(level))
        pieces=w.gather(data,root=0)
        if rank==0:np.savez_compressed(output/name,**{k:v for part in pieces for k,v in part.items()})
    if save_fields:snapshots('old','initial_fields.npz')
    field_output=FieldOutput3D(blocks,output,xdmf_every,checkpoint_every)
    field_output.write(0,0.,level='old')
    coupling.ops=None
    for b in blocks:b.surface=None;b.normals=None;b.trace_map=None
    setup=w.allreduce(time.perf_counter()-start,op=mpi.MAX);records=[];energies=[{'step':0,'t':0.,'physical_energy':0.,'bdf2_G_energy':0.}]
    integration_start=time.perf_counter()
    for step in range(1,steps+1):
        step_start=time.perf_counter();first=step==1;t=step*dt
        t0=time.perf_counter()
        for system in systems:system.prepare(first)
        matrix_s=time.perf_counter()-t0;t0=time.perf_counter()
        predictor=coupling.predictor(first) if method=='nr' else None
        exchange_s=time.perf_counter()-t0;t0=time.perf_counter()
        rhs=[system.rhs(t,predictor) for system in systems];rhs_s=time.perf_counter()-t0;t0=time.perf_counter()
        for system,b in zip(systems,rhs):system.solve(b)
        solve_s=time.perf_counter()-t0;t0=time.perf_counter()
        residual=max(system.residual(b) for system,b in zip(systems,rhs));residual=w.allreduce(residual,op=mpi.MAX)
        if not np.isfinite(residual) or residual>1e-7:raise ArithmeticError(f'Unacceptable linear residual {residual} at step {step}')
        bc_violation=0.
        for block in blocks:
            prescribed=block.new.vector().copy()
            for bc in block.bcs:bc.apply(prescribed)
            defect=block.new.vector().copy();defect.axpy(-1.,prescribed)
            bc_violation=max(bc_violation,defect.norm('linf'))
        bc_violation=w.allreduce(bc_violation,op=mpi.MAX)
        if bc_violation>1e-8:raise ArithmeticError(f'Dirichlet violation {bc_violation}')
        local=0.;g_energy=0.
        for b in blocks:
            energy=b.energy_and_volume()
            # A separate physical energy and BDF2 G-energy are both saved;
            # BE startup is labelled and no forced monotonicity is asserted.
            f=dict(zip(b.names,split(b.new)));old=dict(zip(b.names,split(b.old)));m=b.m;dx=b.g['dx']
            d={k:2*f[k]-old[k] for k in b.names}
            from nsbiot_3d_forms import epsilon
            if b.kind=='fluid':density=m['rho_f']*inner(d['u'],d['u'])
            else:density=m['rho_b']*inner(d['xi'],d['xi'])+m['C0']*d['phi']**2+2*m['mu_b']*inner(epsilon(d['eta']),epsilon(d['eta']))+m['lambda_b']*div(d['eta'])**2
            G=.25*(2*energy+float(assemble(density*dx)))
            if b.comm.rank==0:local+=energy;g_energy+=G
        energy=w.allreduce(local,op=mpi.SUM);G=w.allreduce(g_energy,op=mpi.SUM)
        if not np.isfinite(energy+G):raise ArithmeticError('Nonfinite energy')
        energies.append({'step':step,'t':t,'physical_energy':energy,'bdf2_G_energy':G})
        field_output.write(step,t,final=step==steps)
        diagnostic_s=time.perf_counter()-t0
        for block in blocks:block.shift()
        row={'step':step,'t':t,'scheme':'BE/EX1' if first and method=='nr' else ('BE' if first else ('BDF2/EX2' if method=='nr' else 'BDF2')),
             'matrix_prepare_s':matrix_s,'trace_exchange_s':exchange_s,'rhs_assembly_s':rhs_s,'linear_solve_s':solve_s,'diagnostics_s':diagnostic_s,'step_wall_s':time.perf_counter()-step_start,'relative_linear_residual':residual,'maximum_dirichlet_violation':bc_violation}
        for k in ['matrix_prepare_s','trace_exchange_s','rhs_assembly_s','linear_solve_s','diagnostics_s','step_wall_s']:row[k]=w.allreduce(row[k],op=mpi.MAX)
        records.append(row)
        if rank==0:print(f'{method}: step {step}/{steps}, t={t:g}, residual={residual:.3g}, energy={energy:.6g}',flush=True)
    integration=w.allreduce(time.perf_counter()-integration_start,op=mpi.MAX)
    if save_fields:snapshots('new','final_fields.npz')
    field_output.close()
    w.Barrier();wall=w.allreduce(time.perf_counter()-start,op=mpi.MAX)
    summary={'method':method,'dimension':3,'world_ranks':size,'mesh_source':mesh_source,'mesh_directory':str((mesh_dir or ROOT/'FSI_parallel_3D/mesh').resolve()) if mesh_source=='artery' else None,'fixture_n':fixture_n if mesh_source=='fixture' else None,'biot_fixture_n':biot_fixture_n or fixture_n if mesh_source=='fixture' else None,
             'dofs':coupling.meta['fluid_dofs']+coupling.meta['biot_dofs'],'fluid_dofs':coupling.meta['fluid_dofs'],'biot_dofs':coupling.meta['biot_dofs'],'phi_degree':phi_degree,'dt':dt,'T':T,'steps':steps,
             'beta':0,'physical_coefficients':cfg['physical_coefficients'],'L':cfg['numerical_robin_L'],'mesh_scale':cfg['mesh_coordinate_scale'] if mesh_source=='artery' else 1.,'configured_artery_mesh_scale':cfg['mesh_coordinate_scale'],'linear_solver':'mumps',
             'wall_s':wall,'setup_s':setup,'interface_setup_s':interface_setup,'integration_wall_s':integration,'allocated_rank_seconds':size*wall,
             'maximum_relative_linear_residual':max(r['relative_linear_residual'] for r in records),'maximum_dirichlet_violation':max(r['maximum_dirichlet_violation'] for r in records),'energy_final':energies[-1]['physical_energy'],
             'xdmf_every':xdmf_every,'checkpoint_every':checkpoint_every,'field_names':{'fluid':['u','p'],'biot':['eta','xi','phi']},'startup':cfg['startup'],'quadrature_order':quadrature_order,'projection_refinement':projection_refinement,'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},
             'timing_note':'Case wall excludes imports/launch/final CSV serialization; includes interface assembly, startup and fields if enabled. Step breakdown uses rank maxima. NR uses one collective trace predictor exchange before independent group solves.',
             'geometry_note':'Fixture uses adjacent cubes and an explicit small synthetic inlet, not the arterial problem' if mesh_source=='fixture' else 'Supplied meshes scaled by 1/4; sample inlet spatial constants scaled consistently; amplitude -35 cm/s'}
    if rank==0:
        write_csv(output/'field_output_times.csv',field_output.times)
        write_csv(output/'step_timings.csv',records);write_csv(output/'energy.csv',energies)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(output/'protocol.json').write_text(json.dumps({'configuration':cfg,'summary':summary},indent=2)+'\n')
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['nr','monolithic'],required=True)
    p.add_argument('--config',type=Path,default=ROOT/'validation/three_dimensional/boundary_configuration.json')
    p.add_argument('--mesh-source',choices=['fixture','artery'],default='fixture');p.add_argument('--mesh-dir',type=Path,default=None)
    p.add_argument('--fixture-n',type=int,default=1);p.add_argument('--biot-fixture-n',type=int,default=None);p.add_argument('--phi-degree',type=int,choices=[1,2],default=2)
    p.add_argument('--projection-refinement',type=int,default=None);p.add_argument('--quadrature-order',type=int,default=None);p.add_argument('--dt',type=float,default=.001);p.add_argument('--T',type=float,default=.003)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--no-field-output',action='store_true',help='Skip gathered NPZ snapshots only; XDMF/native HDF5 use their own intervals');p.add_argument('--xdmf-every',type=int,default=1,help='ParaView output interval; default 1, 0 disables');p.add_argument('--checkpoint-every',type=int,default=1,help='Native FE HDF5 output interval; default 1, 0 disables')
    a=p.parse_args();set_log_level(LogLevel.WARNING)
    try:
        cfg=load_config(a.config)
        order=a.quadrature_order if a.quadrature_order is not None else cfg['interface_discretization']['quadrature_order']
        refinement=a.projection_refinement if a.projection_refinement is not None else cfg['interface_discretization']['projection_refinement']
        run(cfg,a.method,a.output_dir,a.dt,a.T,a.mesh_source,a.fixture_n,a.biot_fixture_n,a.phi_degree,order,a.mesh_dir,not a.no_field_output,a.xdmf_every,refinement,a.checkpoint_every)
    except Exception:
        import traceback
        traceback.print_exc()
        if mpi.COMM_WORLD.size>1:mpi.COMM_WORLD.Abort(1)
        raise
if __name__=='__main__':main()
