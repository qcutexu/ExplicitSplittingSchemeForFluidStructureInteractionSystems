"""Fully coupled BDF2 reference on the same two Stokes/Biot FE meshes.

Native matching 1D trace mass blocks restore all interface values to the
current time level. Partition-only L and beta terms cancel. No EX2 loads or
new physical penalties are added. Physical unknown counts are unchanged.
"""
import time
import numpy as np
import scipy.sparse as sp
from fenics import *
from mpi4py import MPI as mpi
from petsc4py import PETSc as petsc
from nsbiot_blocks import assemble_rhs,pack_traces,receive_traces


def csr(A):
    p=as_backend_type(A).mat();ia,ja,values=p.getValuesCSR()
    return sp.csr_matrix((values.copy(),ja.copy(),ia.copy()),shape=(A.size(0),A.size(1)))

class MonolithicSystem:
    def __init__(self,fluid,biot):
        if fluid.comm.size!=1 or biot.comm.size!=1:raise ValueError('This monolithic reference uses serial subdomain meshes')
        self.fluid=fluid;self.biot=biot;self.nf=fluid.V.dim();self.nb=biot.V.dim();self.N=self.nf+self.nb
        cfg=fluid.cfg
        matrix_start=time.perf_counter()
        line=IntervalMesh(mpi.COMM_SELF,cfg.nm,0.,1.)
        trace2=FunctionSpace(line,'CG',2);tracep=FunctionSpace(line,'CG',cfg.phi_degree)
        def mass(testspace,trialspace):
            trial=TrialFunction(trialspace);test=TestFunction(testspace)
            return csr(assemble(trial*test*dx(domain=line)))
        M22=mass(trace2,trace2);M2p=mass(trace2,tracep);Mpp=mass(tracep,tracep)
        def map_dofs(block,field,component,trace,offset):
            space=block.V.sub(field)
            if component is not None:space=space.sub(component)
            ids=space.dofmap().dofs();coords=block.V.tabulate_dof_coordinates().reshape(-1,2)
            lookup={round(float(coords[i,0]),12):int(i)+offset for i in ids if abs(coords[i,1])<1e-12}
            return np.array([lookup[round(float(x),12)] for x in trace.tabulate_dof_coordinates().reshape(-1)],dtype=np.int64)
        ux=map_dofs(fluid,0,0,trace2,0);uy=map_dofs(fluid,0,1,trace2,0)
        xx=map_dofs(biot,1,0,trace2,self.nf);xy=map_dofs(biot,1,1,trace2,self.nf)
        pp=map_dofs(biot,2,None,tracep,self.nf)
        R=sp.csr_matrix((self.N,self.N))
        def add(M,rows,cols,factor):
            nonlocal R
            coo=M.tocoo();R=R+sp.coo_matrix((factor*coo.data,(rows[coo.row],cols[coo.col])),shape=(self.N,self.N)).tocsr()
        add(M22,uy,uy,cfg.L)
        add(M22,ux,xx,cfg.gamma)
        add(M2p,uy,pp,1.)
        add(M22,xx,ux,cfg.gamma)
        if cfg.beta:add(M22,xy,xy,cfg.beta)
        add(M2p.T,pp,uy,-1.)
        add(Mpp,pp,pp,1./cfg.L)
        mask=np.ones(self.N)
        for block,offset in ((fluid,0),(biot,self.nf)):
            for bc in block.bcs:
                for dof in bc.get_boundary_values():mask[offset+dof]=0.
        self.R=(sp.diags(mask)@R).tocsr()
        self.matrix=(sp.block_diag((csr(fluid.A),csr(biot.A)),format='csr')-self.R).tocsr()
        self.matrix.eliminate_zeros()
        pmat=petsc.Mat().createAIJ(size=self.matrix.shape,
            csr=(self.matrix.indptr.astype(petsc.IntType),self.matrix.indices.astype(petsc.IntType),self.matrix.data),comm=mpi.COMM_SELF)
        pmat.assemble();self.A=PETScMatrix(pmat)
        self.interface_matrix_assembly_s=time.perf_counter()-matrix_start
        self.b=Vector(mpi.COMM_SELF,self.N);self.x=Vector(mpi.COMM_SELF,self.N)
        self.solver=LUSolver(self.A,cfg.linear_solver)
    def rhs(self,step):
        return np.r_[assemble_rhs(self.fluid,step,True).get_local(),assemble_rhs(self.biot,step,True).get_local()]
    def solve(self,rhs):
        self.b.set_local(rhs);self.b.apply('insert');self.solver.solve(self.x,self.b)
        values=self.x.get_local()
        self.fluid.new.vector().set_local(values[:self.nf]);self.fluid.new.vector().apply('insert')
        self.biot.new.vector().set_local(values[self.nf:]);self.biot.new.vector().apply('insert')
    def relative_residual(self,rhs):
        return float(np.linalg.norm(self.matrix@self.x.get_local()-rhs)/max(np.linalg.norm(rhs),1e-300))
    def pack_history(self,level):
        arrays=[]
        for block in (self.fluid,self.biot):
            functions=[getattr(block,level)[name] for name in block.names]
            mixed=Function(block.V);FunctionAssigner(block.V,[f.function_space() for f in functions]).assign(mixed,functions)
            arrays.append(mixed.vector().get_local())
        return np.concatenate(arrays)
    def check_bulk_rhs(self,step=2):
        # Independent oracle: full partitioned RHS minus its EX2 interface load.
        fluid_packet=pack_traces(self.fluid);biot_packet=pack_traces(self.biot)
        receive_traces(self.fluid,biot_packet);receive_traces(self.biot,fluid_packet)
        full=np.r_[assemble_rhs(self.fluid,step).get_local(),assemble_rhs(self.biot,step).get_local()]
        expected=full-self.R@(2*self.pack_history('old')-self.pack_history('older'))
        bulk=self.rhs(step)
        return float(np.linalg.norm(expected-bulk)/max(np.linalg.norm(bulk),1e-300))

if __name__=="__main__":
    from nsbiot_run import main
    main(default_execution="serial",default_method="monolithic")
