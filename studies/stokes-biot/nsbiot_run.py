#!/usr/bin/env python3
"""Serial or genuinely concurrent MPI Neumann–Robin runs on matched FE meshes."""
import argparse
import csv
import json
import os
import time
from pathlib import Path
import numpy as np
from fenics import *
from mpi4py import MPI as mpi
from nsbiot_blocks import Settings,FluidBlock,BiotBlock,pack_traces,receive_traces,assemble_rhs,snapshot

class InterfaceExchange:
    def __init__(self,block,world,middle):
        self.block=block;self.world=world;self.middle=middle
        packet=pack_traces(block);header=None;self.keys=None
        if block.comm.rank==0:
            self.keys=sorted(packet)
            own=[(key,packet[key][:,:2].copy()) for key in self.keys]
            other=middle if block.kind=='fluid' else 0
            header=world.sendrecv(own,dest=other,sendtag=1,source=other,recvtag=1)
        self.incoming=block.comm.bcast(header,root=0)
        self.in_size=sum(len(coords) for _,coords in self.incoming)
        self.out_size=sum(len(value) for value in packet.values()) if block.comm.rank==0 else 0
    def exchange(self,step):
        packet=pack_traces(self.block)
        received=np.empty(self.in_size,dtype=np.float64)
        if self.block.comm.rank==0:
            sent=np.ascontiguousarray(np.concatenate([packet[key][:,2] for key in self.keys]))
            other=self.middle if self.block.kind=='fluid' else 0
            self.world.Sendrecv(sent,dest=other,sendtag=step+10,recvbuf=received,source=other,recvtag=step+10)
        self.block.comm.Bcast(received,root=0)
        packet={};offset=0
        for key,coords in self.incoming:
            packet[key]=np.column_stack((coords,received[offset:offset+len(coords)]));offset+=len(coords)
        receive_traces(self.block,packet)

def save_csv(path,rows):
    if not rows:return
    keys=list(rows[0])
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def execute(cfg,output,execution='serial',method='nr',save_fields=True):
    world=mpi.COMM_WORLD;rank=world.rank;size=world.size;cfg.validate()
    if execution=='serial' and size!=1:raise ValueError('Serial execution requires one process')
    if execution=='mpi' and (size<2 or size%2):raise ValueError('MPI execution requires an even number of processes >=2')
    if execution=='mpi' and (method=='monolithic' or size//2>1) and cfg.linear_solver=='umfpack':raise ValueError('Multiple ranks per block require explicit --linear-solver mumps')
    if cfg.beta!=0:raise ValueError('This comparison driver implements beta=0 NR')
    if rank==0:output.mkdir(parents=True,exist_ok=True)
    world.Barrier();case_start=time.perf_counter()
    blocks=[];exchange=None;monolithic=None
    if execution=='mpi' and method=='monolithic':
        from nsbiot_monolithic_mpi import DistributedMonolithicSystem
        blocks=[FluidBlock(cfg,world),BiotBlock(cfg,world)]
        monolithic=DistributedMonolithicSystem(*blocks)
    elif execution=='mpi':
        middle=size//2;color=0 if rank<middle else 1
        group=world.Split(color,rank)
        block=(FluidBlock if color==0 else BiotBlock)(cfg,group)
        blocks=[block];exchange=InterfaceExchange(block,world,middle)
    else:
        blocks=[FluidBlock(cfg,mpi.COMM_SELF),BiotBlock(cfg,mpi.COMM_SELF)]
        if method=='monolithic':
            from nsbiot_monolithic import MonolithicSystem
            monolithic=MonolithicSystem(blocks[0],blocks[1])
    setup_elapsed=time.perf_counter()-case_start
    initial={};final={}
    if save_fields:
        for block in blocks:
            for level in ('old','older'):
                data=snapshot(block,level)
                if block.comm.rank==0:
                    for name,array in data.items():initial[level+'_'+name]=array
        pieces=world.gather(initial,root=0)
        if rank==0:
            combined={key:value for piece in pieces for key,value in piece.items()}
            np.savez_compressed(output/'initial_fields.npz',**combined)
    def global_energy(current=False):
        local=0.
        for block in blocks:
            energy=block.energy(block.current,block.old) if current else block.energy(block.old,block.older)
            if block.comm.rank==0:local+=energy
        return world.allreduce(local,op=mpi.SUM)
    energy_rows=[{'step':1,'t':cfg.dt,'energy':global_energy()}]
    errors={key:0. for key in ('err_u','err_p','err_eta','err_xi','err_phi')}
    timings=[];breakdown={'communication_s':0.,'rhs_assembly_s':0.,'linear_solve_s':0.,'diagnostics_s':0.,'output_s':0.}
    integration_start=time.perf_counter()
    for step in range(2,cfg.steps+1):
        step_start=time.perf_counter();t0=time.perf_counter()
        if exchange is not None:exchange.exchange(step)
        elif method=='nr':
            fluid_packet=pack_traces(blocks[0]);biot_packet=pack_traces(blocks[1])
            receive_traces(blocks[0],biot_packet);receive_traces(blocks[1],fluid_packet)
        comm_s=time.perf_counter()-t0
        t0=time.perf_counter()
        if monolithic is not None:
            rhs=monolithic.rhs(step)
        else:rhs=[assemble_rhs(block,step) for block in blocks]
        rhs_s=time.perf_counter()-t0
        t0=time.perf_counter()
        if monolithic is not None:monolithic.solve(rhs)
        else:
            for block,b in zip(blocks,rhs):block.solver.solve(block.new.vector(),b)
        solve_s=time.perf_counter()-t0
        t0=time.perf_counter()
        for block in blocks:block.update()
        energy=global_energy(True)
        if not np.isfinite(energy):raise ArithmeticError('Nonfinite energy at step '+str(step))
        energy_rows.append({'step':step,'t':step*cfg.dt,'energy':energy})
        local_errors={key:0. for key in errors}
        if cfg.forcing=='manufactured':
            for block in blocks:
                values=block.errors()
                if block.comm.rank==0:local_errors.update(values)
        for key in errors:errors[key]=max(errors[key],world.allreduce(local_errors[key],op=mpi.MAX))
        if monolithic is not None:residual=monolithic.relative_residual(rhs)
        else:
            residual=0.
            for block,b in zip(blocks,rhs):
                defect=block.A*block.new.vector();defect.axpy(-1.,b)
                residual=max(residual,defect.norm('l2')/max(b.norm('l2'),1e-300))
        residual=world.allreduce(residual,op=mpi.MAX)
        for block in blocks:block.shift()
        diagnostic_s=time.perf_counter()-t0
        record={'step':step,'t':step*cfg.dt,'communication_s':comm_s,'rhs_assembly_s':rhs_s,
                'linear_solve_s':solve_s,'diagnostics_s':diagnostic_s,
                'step_wall_s':time.perf_counter()-step_start,'relative_linear_residual':residual}
        # Max across simultaneous workers; never sum parallel solve times.
        for key in ('communication_s','rhs_assembly_s','linear_solve_s','diagnostics_s','step_wall_s'):
            record[key]=world.allreduce(record[key],op=mpi.MAX)
        timings.append(record)
        for key in breakdown:
            if key in record:breakdown[key]+=record[key]
    integration_wall=world.allreduce(time.perf_counter()-integration_start,op=mpi.MAX)
    if save_fields:
        t0=time.perf_counter()
        for block in blocks:
            data=snapshot(block)
            if block.comm.rank==0:final.update(data)
        pieces=world.gather(final,root=0)
        if rank==0:np.savez_compressed(output/'final_fields.npz',**{key:value for piece in pieces for key,value in piece.items()})
        breakdown['output_s']=world.allreduce(time.perf_counter()-t0,op=mpi.MAX)
    world.Barrier();wall=world.allreduce(time.perf_counter()-case_start,op=mpi.MAX)
    local_dofs=sum(block.V.dim() for block in blocks if block.comm.rank==0)
    dofs=world.allreduce(local_dofs,op=mpi.SUM)
    summary={'method':'monolithic_bdf2' if method=='monolithic' else 'neumann_robin_bdf2_ex2',
             'execution':execution,'world_ranks':size,'ranks_per_subdomain':(size if method=='monolithic' else size//2) if execution=='mpi' else 1,
             'beta':cfg.beta,'nm':cfg.nm,'dt':cfg.dt,'T':cfg.T,'steps':cfg.steps,'bdf2_updates':cfg.steps-1,
             'phi_degree':cfg.phi_degree,'physical':cfg.material,'L':cfg.L,'gamma':cfg.gamma,
             'forcing':cfg.forcing,'initial_mode':cfg.initial_mode,'linear_solver':cfg.linear_solver,
             'dofs':dofs,'energy_initial':energy_rows[0]['energy'],'energy_final':energy_rows[-1]['energy'],
             'peak_energy_ratio':max(r['energy'] for r in energy_rows)/energy_rows[0]['energy'],
             'wall_s':wall,'integration_wall_s':integration_wall,'allocated_rank_seconds':size*wall,'setup_s':world.allreduce(setup_elapsed,op=mpi.MAX),
             'matrix_assembly_s':world.allreduce(sum(block.matrix_assembly_s for block in blocks)+(monolithic.interface_matrix_assembly_s if monolithic is not None else 0.),op=mpi.MAX),
             'interface_payload_bytes_per_step':world.allreduce(exchange.out_size*8 if exchange is not None else 0,op=mpi.SUM),
             'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
             **breakdown,'maximum_relative_linear_residual':max(r['relative_linear_residual'] for r in timings),
             **{key:(value if cfg.forcing=='manufactured' else None) for key,value in errors.items()},
             'timing_note':'Breakdown sums per-step rank maxima and need not add to wall time; startup includes JIT; field output and diagnostics included; final CSV/summary serialization, imports and launcher excluded. Allocated rank-seconds is ranks times wall, not measured CPU time.'}
    if rank==0:
        save_csv(output/'energy.csv',energy_rows);save_csv(output/'step_timings.csv',timings)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (output/'protocol.json').write_text(json.dumps({'settings':vars(cfg),'summary':summary,
             'interface':'exact matching nodal traces; explicit component and owned-dof maps',
             'time_coupling':('current-level fully coupled interface' if method=='monolithic' else 'both previous history levels exchanged before independent solves; no current-field Gauss-Seidel coupling'),
             'sample_inspiration':'FSI_parallel_3D/really_double.py communicator groups only; no ALE/nonlinear/3D model imported'},indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
    return summary

def main(default_execution='mpi',default_method='nr'):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution',choices=('serial','mpi'),default=default_execution)
    p.add_argument('--method',choices=('nr','monolithic'),default=default_method)
    p.add_argument('--mesh',type=int,default=8);p.add_argument('--dt',type=float,default=.0625);p.add_argument('--T',type=float,default=.5)
    p.add_argument('--phi-degree',type=int,choices=(1,2),default=2)
    p.add_argument('--forcing',choices=('manufactured','zero'),default='manufactured')
    p.add_argument('--initial-mode',choices=('smooth','interface','oscillatory'),default=None)
    p.add_argument('--mode-number',type=int,default=8)
    p.add_argument('--linear-solver',choices=('umfpack','mumps'),default='umfpack')
    p.add_argument('--L',type=float,default=1.);p.add_argument('--gamma',type=float,default=1.)
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--no-field-output',action='store_true')
    a=p.parse_args();set_log_level(LogLevel.WARNING)
    if a.output_dir is None:a.output_dir=Path('nsbiot_results')/(a.method+'_'+a.execution)
    cfg=Settings(nm=a.mesh,dt=a.dt,T=a.T,phi_degree=a.phi_degree,forcing=a.forcing,
          initial_mode=a.initial_mode or ('smooth' if a.forcing=='manufactured' else 'interface'),mode_number=a.mode_number,L=a.L,gamma=a.gamma,linear_solver=a.linear_solver)
    try:execute(cfg,a.output_dir,a.execution,a.method,not a.no_field_output)
    except Exception as exc:
        print(f'Rank {mpi.COMM_WORLD.rank}: {type(exc).__name__}: {exc}',flush=True)
        if mpi.COMM_WORLD.size>1:mpi.COMM_WORLD.Abort(1)
        raise

if __name__=='__main__':main()
