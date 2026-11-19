"""Distributed volume matrices and row-distributed common-surface operators."""
import time
import numpy as np
import scipy.sparse as sp
from fenics import *
from mpi4py import MPI as mpi
from petsc4py import PETSc as petsc
from nsbiot_3d_surface import InterfaceOperators,key

class Coupling3D:
    def __init__(self,blocks,world,cfg,phi_degree,order,refinement=3):
        self.world=world;self.blocks=blocks;rank=world.rank
        bundle={b.kind:{'surface':b.surface,'normals':b.normals,'trace_map':b.trace_map,'dofs':b.V.dim()} for b in blocks if b.comm.rank==0}
        pieces=world.gather(bundle,root=0);ops=None;meta=None
        if rank==0:
            data={k:v for p in pieces for k,v in p.items()}
            ops=InterfaceOperators(data['fluid']['surface'],data['biot']['surface'],data['fluid']['normals'],cfg['physical_coefficients'],cfg['numerical_robin_L'],phi_degree,order,refinement,data['biot']['normals'])
            native={}
            for name,kind,ncomp in [('u','fluid',3),('xi','biot',3),('phi','biot',1)]:
                native[name]=np.asarray([[data[kind]['trace_map'][name,c,key(x)] for c in range(ncomp)] for x in ops.coordinates[name]],dtype=petsc.IntType)
            ids=np.concatenate([native[k].reshape(-1) for k in ['u','xi','phi']])
            meta={'native_ids':ids,'nf_trace':ops.nf,'N_trace':ops.N,'fluid_dofs':data['fluid']['dofs'],'biot_dofs':data['biot']['dofs'],'audit':ops.audit}
        self.meta=world.bcast(meta,root=0);self.ops=ops;self.ids=self.meta['native_ids'];self.nt=self.meta['N_trace'];self.nft=self.meta['nf_trace']
        self.local=[]
        for b in blocks:
            lo,hi=b.old.vector().local_range();kindmask=np.arange(self.nt)<self.nft
            if b.kind=='biot':kindmask=~kindmask
            indices=np.flatnonzero(kindmask & (self.ids>=lo)&(self.ids<hi))
            self.local.append((b,indices,self.ids[indices]-lo))
    def predictor(self,first):
        send=np.zeros(self.nt)
        for block,indices,local in self.local:
            old=block.old.vector().get_local();older=block.older.vector().get_local()
            send[indices]=old[local] if first else 2*old[local]-older[local]
        recv=np.empty_like(send);self.world.Allreduce(send,recv,op=mpi.SUM);return recv
    def trace_current(self):
        send=np.zeros(self.nt)
        for block,indices,local in self.local:send[indices]=block.new.vector().get_local()[local]
        recv=np.empty_like(send);self.world.Allreduce(send,recv,op=mpi.SUM);return recv
    def distribute(self,kind,ownership,offsets=False):
        # Called collectively by the world. NR ranks own native block rows;
        # monolithic ranks own their contiguous coupled matrix row ranges.
        descriptions=self.world.gather(ownership,root=0);packets=None
        if self.world.rank==0:
            matrix=getattr(self.ops,kind).tocoo();joint=self.ids.copy();joint[self.nft:]+=self.meta['fluid_dofs']
            packets=[]
            for desc in descriptions:
                block,lo,hi=desc
                rows=joint[matrix.row] if offsets else self.ids[matrix.row]
                cols=joint[matrix.col] if offsets else self.ids[matrix.col]
                take=(rows>=lo)&(rows<hi)
                if not offsets:
                    which=matrix.row<self.nft
                    take &= which if block=='fluid' else ~which
                if kind=='R':cols=matrix.col
                packets.append((rows[take],cols[take],matrix.data[take]))
        return self.world.scatter(packets,root=0)

def add_coo(mat,packet):
    rows,cols,values=packet
    if len(rows):
        order=np.argsort(rows,kind='stable');rows=rows[order];cols=cols[order];values=values[order]
        unique,starts=np.unique(rows,return_index=True);ends=np.r_[starts[1:],len(rows)]
        for row,start,end in zip(unique,starts,ends):mat.setValues(int(row),np.asarray(cols[start:end],dtype=petsc.IntType),values[start:end],addv=True)

class NRSystem3D:
    def __init__(self,block,coupling):
        self.block=block;self.c=coupling;lo,hi=block.old.vector().local_range()
        self.D=coupling.distribute('D',(block.kind,lo,hi))
        rows,cols,vals=coupling.distribute('R',(block.kind,lo,hi))
        self.R=sp.coo_matrix((vals,(rows-lo,cols)),shape=(hi-lo,coupling.nt)).tocsr()
        self.systems={};self.matrix_setup_s=0.
    def prepare(self,first):
        self.block.set_scheme(first)
        if not first and True in self.systems:
            self.systems.pop(True);self.A=None;self.solver=None
        if first not in self.systems:
            start=time.perf_counter();base=assemble(self.block.a);mat=as_backend_type(base).mat().duplicate(copy=True)
            mat.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,False)
            add_coo(mat,self.D);mat.assemble();A=PETScMatrix(mat)
            for bc in self.block.bcs:bc.apply(A)
            self.systems[first]=(A,LUSolver(A,'mumps'));self.matrix_setup_s+=time.perf_counter()-start
        self.A,self.solver=self.systems[first]
    def rhs(self,t,predictor):
        b=self.block.rhs(t);values=b.get_local()+self.R@predictor;b.set_local(values);b.apply('insert')
        for bc in self.block.bcs:bc.apply(b)
        return b
    def solve(self,rhs):self.solver.solve(self.block.new.vector(),rhs)
    def residual(self,rhs):
        r=self.A*self.block.new.vector();r.axpy(-1.,rhs);return r.norm('l2')/max(rhs.norm('l2'),1e-300)

class MonolithicSystem3D:
    def __init__(self,blocks,coupling):
        self.blocks=blocks;self.c=coupling;self.comm=coupling.world;self.nf=coupling.meta['fluid_dofs'];self.N=self.nf+coupling.meta['biot_dofs']
        self.mat=petsc.Mat().createAIJ(size=(self.N,self.N),nnz=0,comm=self.comm)
        self.mat.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,False);self.mat.assemble()
        lo,hi=self.mat.getOwnershipRange();self.H=coupling.distribute('H',('joint',lo,hi),True)
        self.systems={};self.matrix_setup_s=0.;self.b=None;self.x=None;self.transfers=[]
    def prepare(self,first):
        for block in self.blocks:block.set_scheme(first)
        if not first and True in self.systems:
            self.systems.pop(True);self.A=None;self.solver=None
        if first not in self.systems:
            start=time.perf_counter()
            row_lo,row_hi=self.mat.getOwnershipRange();local_size=row_hi-row_lo
            ends=np.asarray(self.comm.allgather(row_hi));bulk_data=[]
            diag_counts=self.mat.createVecLeft();off_counts=diag_counts.duplicate()
            diag_counts.set(0.);off_counts.set(0.)
            for block in self.blocks:
                offset=0 if block.kind=='fluid' else self.nf
                bulk=assemble(block.a);source=as_backend_type(bulk).mat();lo,hi=source.getOwnershipRange()
                ia,ja,values=source.getValuesCSR();rows=np.arange(lo+offset,hi+offset,dtype=petsc.IntType);cols=ja+offset
                owner=np.searchsorted(ends,rows,side='right')
                col_owner=np.searchsorted(ends,cols,side='right')
                diagonal=col_owner==np.repeat(owner,np.diff(ia))
                cumulative=np.r_[0,np.cumsum(diagonal)]
                dn=np.diff(cumulative[ia]);on=np.diff(ia)-dn
                diag_counts.setValues(rows,dn,addv=True);off_counts.setValues(rows,on,addv=True)
                bulk_data.append((bulk,rows,ia,cols,values))
            diag_counts.assemble();off_counts.assemble()
            dn=diag_counts.getArray().copy();on=off_counts.getArray().copy()
            hrows,hcols,hvalues=self.H
            local_diagonal=(hcols>=row_lo)&(hcols<row_hi)
            dn+=np.bincount((hrows[local_diagonal]-row_lo).astype(int),minlength=local_size)
            on+=np.bincount((hrows[~local_diagonal]-row_lo).astype(int),minlength=local_size)
            # Count volume and trace locations on their actual coupled owners.
            # Summing overlapping locations only overallocates; no volume
            # matrix/state gathering or empty-template duplication is needed.
            mat=petsc.Mat().createAIJ(size=((local_size,self.N),(local_size,self.N)),
                nnz=(dn.astype(petsc.IntType),on.astype(petsc.IntType)),comm=self.comm)
            mat.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,True)
            for bulk,rows,ia,cols,values in bulk_data:
                for j,row in enumerate(rows):
                    sl=slice(ia[j],ia[j+1]);mat.setValues(int(row),cols[sl],values[sl],addv=True)
            add_coo(mat,self.H);mat.assemble()
            constrained=[]
            for block in self.blocks:
                offset=0 if block.kind=='fluid' else self.nf;dm=block.V.dofmap();lg=dm.tabulate_local_to_global_dofs();lo,hi=block.old.vector().local_range()
                for bc in block.bcs:constrained.extend(int(lg[i])+offset for i in bc.get_boundary_values())
            # get_boundary_values may omit an owned boundary DOF if only a
            # different rank owns its marked facet. Retain ghost-side tags,
            # union globally, then zero each coupled row on its owner.
            all_constraints={i for part in self.comm.allgather(constrained) for i in part}
            row_lo,row_hi=mat.getOwnershipRange()
            owned=np.asarray(sorted(i for i in all_constraints if row_lo<=i<row_hi),dtype=petsc.IntType)
            mat.zeroRows(owned,diag=1.);mat.assemble()
            A=PETScMatrix(mat);self.systems[first]=(A,LUSolver(A,'mumps'));self.matrix_setup_s+=time.perf_counter()-start
        self.A,self.solver=self.systems[first]
        if self.b is None:
            self.b=PETScVector(self.comm);self.x=PETScVector(self.comm);self.A.init_vector(self.b,0);self.A.init_vector(self.x,1)
            for block in self.blocks:
                offset=0 if block.kind=='fluid' else self.nf;lo,hi=block.old.vector().local_range();indices=np.arange(lo+offset,hi+offset,dtype=petsc.IntType)
                seq=petsc.Vec().createSeq(len(indices),comm=mpi.COMM_SELF)
                src=petsc.IS().createGeneral(indices,comm=self.comm);dest=petsc.IS().createStride(len(indices),comm=mpi.COMM_SELF)
                scatter=petsc.Scatter().create(as_backend_type(self.x).vec(),src,seq,dest)
                self.transfers.append((block,indices,seq,scatter))
    def rhs(self,t,predictor=None):
        target=as_backend_type(self.b).vec();target.set(0.)
        for block,indices,_,_ in self.transfers:
            b=block.rhs(t)
            for bc in block.bcs:bc.apply(b)
            target.setValues(indices,b.get_local())
        target.assemble();return self.b
    def solve(self,rhs):
        self.solver.solve(self.x,rhs)
        for block,_,seq,scatter in self.transfers:
            scatter.scatter(as_backend_type(self.x).vec(),seq,addv=petsc.InsertMode.INSERT,mode=petsc.ScatterMode.FORWARD)
            block.new.vector().set_local(seq.getArray());block.new.vector().apply('insert');block.new.vector().update_ghost_values()
    def residual(self,rhs):
        r=self.A*self.x;r.axpy(-1.,rhs);return r.norm('l2')/max(rhs.norm('l2'),1e-300)
