"""Distributed current-interface BDF2 operator, with distributed PETSc/MUMPS LU.

Both physical meshes are partitioned on the world communicator. Local subdomain
matrix rows are inserted into one globally coupled operator. Only small trace
maps are replicated; volume matrices and state vectors are never gathered.
"""
import time
import numpy as np
import scipy.sparse as sp
from fenics import *
from mpi4py import MPI as mpi
from petsc4py import PETSc as petsc
from nsbiot_blocks import assemble_rhs
from nsbiot_monolithic import csr

class DistributedMonolithicSystem:
    def __init__(self,fluid,biot):
        start=time.perf_counter();self.fluid=fluid;self.biot=biot;self.comm=fluid.comm
        self.nf=fluid.V.dim();self.N=self.nf+biot.V.dim();cfg=fluid.cfg
        line=IntervalMesh(mpi.COMM_SELF,cfg.nm,0.,1.)
        t2=FunctionSpace(line,'CG',2);tp=FunctionSpace(line,'CG',cfg.phi_degree)
        def mass(test,trial):return csr(assemble(TrialFunction(trial)*TestFunction(test)*dx(domain=line)))
        M22=mass(t2,t2);M2p=mass(t2,tp);Mpp=mass(tp,tp)
        def mapping(block,field,component,trace,offset):
            sub=block.V.sub(field)
            if component is not None:sub=sub.sub(component)
            dm=block.V.dofmap();lg=dm.tabulate_local_to_global_dofs();lo,hi=dm.ownership_range()
            coords=block.V.tabulate_dof_coordinates().reshape(-1,2)
            reverse={int(g):j for j,g in enumerate(lg)}
            local={round(float(coords[reverse[int(i)],0]),12):int(i)+offset for i in sub.dofmap().dofs() if lo<=i<hi and abs(coords[reverse[int(i)],1])<1e-12}
            lookup={k:v for part in self.comm.allgather(local) for k,v in part.items()}
            return np.asarray([lookup[round(float(x),12)] for x in trace.tabulate_dof_coordinates().reshape(-1)],dtype=petsc.IntType)
        ux=mapping(fluid,0,0,t2,0);uy=mapping(fluid,0,1,t2,0)
        xx=mapping(biot,1,0,t2,self.nf);xy=mapping(biot,1,1,t2,self.nf);pp=mapping(biot,2,None,tp,self.nf)
        R=sp.csr_matrix((self.N,self.N))
        def add(M,rows,cols,factor):
            nonlocal R
            coo=M.tocoo();R+=sp.coo_matrix((factor*coo.data,(rows[coo.row],cols[coo.col])),shape=R.shape).tocsr()
        add(M22,uy,uy,cfg.L);add(M22,ux,xx,cfg.gamma);add(M2p,uy,pp,1.)
        add(M22,xx,ux,cfg.gamma)
        if cfg.beta:add(M22,xy,xy,cfg.beta)
        add(M2p.T,pp,uy,-1.);add(Mpp,pp,pp,1./cfg.L)
        constrained=[]
        for block,offset in ((fluid,0),(biot,self.nf)):
            lg=block.V.dofmap().tabulate_local_to_global_dofs()
            for bc in block.bcs:constrained.extend(int(lg[i])+offset for i in bc.get_boundary_values())
        constrained={i for part in self.comm.allgather(constrained) for i in part}
        # Only trace rows need masking; keep sparse storage without a volume mask.
        coo=R.tocoo();keep=np.asarray([int(i) not in constrained for i in coo.row])
        R=sp.coo_matrix((coo.data[keep],(coo.row[keep],coo.col[keep])),shape=R.shape).tocsr()
        self.mat=petsc.Mat().createAIJ(size=(self.N,self.N),nnz=(100,100),comm=self.comm)
        self.mat.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR,False)
        for block,offset in ((fluid,0),(biot,self.nf)):
            A=as_backend_type(block.A).mat();lo,hi=A.getOwnershipRange();ia,ja,values=A.getValuesCSR()
            for j,row in enumerate(range(lo,hi)):
                sl=slice(ia[j],ia[j+1]);self.mat.setValues(row+offset,ja[sl]+offset,values[sl],addv=True)
        # Each trace row inserted exactly once, using coupled-matrix ownership.
        lo,hi=self.mat.getOwnershipRange()
        for row in np.unique(coo.row):
            if lo<=row<hi:
                sl=slice(R.indptr[row],R.indptr[row+1]);self.mat.setValues(int(row),R.indices[sl],-R.data[sl],addv=True)
        self.mat.assemble();self.A=PETScMatrix(self.mat)
        self.b=PETScVector(self.comm);self.x=PETScVector(self.comm)
        self.A.init_vector(self.b,0);self.A.init_vector(self.x,1)
        self.solver=LUSolver(self.A,cfg.linear_solver)
        self.transfers=[]
        for block,offset in ((fluid,0),(biot,self.nf)):
            lo,hi=block.new.vector().local_range()
            indices=np.arange(lo+offset,hi+offset,dtype=petsc.IntType)
            seq=petsc.Vec().createSeq(len(indices),comm=mpi.COMM_SELF)
            source=petsc.IS().createGeneral(indices,comm=self.comm)
            dest=petsc.IS().createStride(len(indices),first=0,step=1,comm=mpi.COMM_SELF)
            scatter=petsc.Scatter().create(as_backend_type(self.x).vec(),source,seq,dest)
            self.transfers.append((block,indices,seq,scatter))
        self.interface_matrix_assembly_s=time.perf_counter()-start
    def rhs(self,step):
        vec=as_backend_type(self.b).vec();vec.set(0.)
        for block,indices,_,_ in self.transfers:
            rhs=assemble_rhs(block,step,True);vec.setValues(indices,rhs.get_local())
        vec.assemble();return self.b
    def solve(self,rhs):
        self.solver.solve(self.x,rhs)
        for block,_,seq,scatter in self.transfers:
            scatter.scatter(as_backend_type(self.x).vec(),seq,addv=petsc.InsertMode.INSERT,mode=petsc.ScatterMode.FORWARD)
            block.new.vector().set_local(seq.getArray());block.new.vector().apply('insert');block.new.vector().update_ghost_values()
    def relative_residual(self,rhs):
        defect=self.mat.createVecLeft();self.mat.mult(as_backend_type(self.x).vec(),defect)
        defect.axpy(-1.,as_backend_type(rhs).vec());return defect.norm()/max(rhs.norm('l2'),1e-300)
