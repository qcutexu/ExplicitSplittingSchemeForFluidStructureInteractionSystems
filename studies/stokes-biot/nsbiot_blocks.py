"""Reusable fixed-domain Stokes/Biot blocks and exact matching trace exchange."""
import hashlib
import math
import time as clock
from dataclasses import dataclass,field
from pathlib import Path
import numpy as np
from fenics import *
from mpi4py import MPI as mpi
import nsbiot_compare_core as reference
from nsbiot_forms import fluid_form,biot_forms,epsilon

@dataclass
class Settings:
    nm:int=8
    dt:float=0.0625
    T:float=0.5
    phi_degree:int=2
    forcing:str='manufactured'
    initial_mode:str='smooth'
    mode_number:int=8
    L:float=1.
    gamma:float=1.
    beta:float=0.
    linear_solver:str='umfpack'
    physical:dict=field(default_factory=dict)
    def validate(self):
        if self.nm<1 or self.phi_degree not in (1,2):raise ValueError('Invalid mesh or phi degree')
        if self.forcing not in ('zero','manufactured'):raise ValueError('Invalid forcing')
        if self.initial_mode not in ('interface','smooth','oscillatory'):raise ValueError('Invalid initial mode')
        if self.forcing=='manufactured' and self.initial_mode!='smooth':raise ValueError('Manufactured tests require smooth exact projected starts')
        if self.mode_number<0 or int(self.mode_number)!=self.mode_number:raise ValueError('Invalid oscillatory mode')
        if not all(math.isfinite(float(v)) and v>0 for v in (self.dt,self.T,self.L,self.gamma)):raise ValueError('dt,T,L,gamma must be positive finite')
        if self.beta<0 or not math.isfinite(self.beta):raise ValueError('Invalid beta')
        self.steps=round(self.T/self.dt)
        if self.steps<2 or not math.isclose(self.steps*self.dt,self.T,abs_tol=1e-13):raise ValueError('dt must divide T into at least two intervals')
        defaults={k:1. for k in ('rho_f','mu_f','rho_b','mu_b','lambda_b','alpha','C0','K')}
        if set(self.physical)-set(defaults):raise ValueError('Unknown physical coefficient')
        defaults.update(self.physical);self.material={k:float(v) for k,v in defaults.items()}
        if not all(math.isfinite(v) and v>0 for v in self.material.values()):raise ValueError('Positive material coefficients required')
        if self.forcing=='manufactured' and any(abs(self.material[k]-1)>1e-13 for k in ('mu_f','mu_b','lambda_b','alpha')):raise ValueError('Manufactured interface stresses require unit stress coefficients')
        if not has_lu_solver_method(self.linear_solver):raise ValueError('Unavailable LU solver: '+self.linear_solver)
        reference.__dict__.update(self.material)
        return self

class Side(SubDomain):
    def __init__(self,axis,value):super().__init__();self.axis=axis;self.value=value
    def inside(self,x,on_boundary):return on_boundary and near(x[self.axis],self.value)

def geometry(comm,nm,kind):
    lo,hi=(0.,1.) if kind=='fluid' else (-1.,0.)
    mesh=RectangleMesh(comm,Point(0.,lo),Point(1.,hi),nm,nm)
    markers=MeshFunction('size_t',mesh,1,0)
    Side(1,0.).mark(markers,5)
    Side(0,1.).mark(markers,2);Side(0,0.).mark(markers,4)
    Side(1,hi if kind=='fluid' else lo).mark(markers,3 if kind=='fluid' else 1)
    normal=FacetNormal(mesh)
    return {'mesh':mesh,'markers':markers,'dx':Measure('dx',domain=mesh),
            'ds':Measure('ds',domain=mesh,subdomain_data=markers),
            'n':normal,'tangent':as_vector((-normal[1],normal[0]))}

def zero_exterior(space,markers,sides,vector):
    return [DirichletBC(space,Constant((0.,0.)) if vector else Constant(0.),markers,side) for side in sides]

def norm2(value,measure):return float(assemble(inner(value,value)*measure))

class FluidBlock:
    kind='fluid';names=('u','p')
    def __init__(self,cfg,comm):
        self.cfg=cfg;self.comm=comm;self.g=geometry(comm,cfg.nm,'fluid');self.mesh=self.g['mesh']
        self.V=FunctionSpace(self.mesh,MixedElement([VectorElement('CG',self.mesh.ufl_cell(),2),FiniteElement('CG',self.mesh.ufl_cell(),1)]))
        self.new=Function(self.V)
        self.old=dict(zip(self.names,Function(self.V).split(True)));self.older=dict(zip(self.names,Function(self.V).split(True)))
        self.time=Constant(0.);self.dt=Constant(cfg.dt)
        self.exact_u,self.exact_p,ff=reference.make_fluid_data(self.mesh,self.time)
        self.source=ff if cfg.forcing=='manufactured' else Constant((0.,0.))
        self.bcs=zero_exterior(self.V.sub(0),self.g['markers'],(4,3,2),True)
        u,p=TrialFunctions(self.V);v,q=TestFunctions(self.V);m=cfg.material;g=self.g
        a=2*m['mu_f']*inner(epsilon(u),epsilon(v))*g['dx']-p*div(v)*g['dx']+div(u)*q*g['dx']+cfg.gamma*dot(u,g['tangent'])*dot(v,g['tangent'])*g['ds'](5)+cfg.L*dot(u,g['n'])*dot(v,g['n'])*g['ds'](5)
        load=2*m['mu_f']*inner(epsilon(self.exact_u),epsilon(v))*g['dx']-self.exact_p*div(v)*g['dx']+div(self.exact_u)*q*g['dx']+cfg.gamma*dot(self.exact_u,g['tangent'])*dot(v,g['tangent'])*g['ds'](5)+cfg.L*dot(self.exact_u,g['n'])*dot(v,g['n'])*g['ds'](5)
        for value,history in ((0.,self.older),(cfg.dt if cfg.forcing=='manufactured' else 0.,self.old)):
            self.time.assign(value)
            if value:
                projected=Function(self.V);solve(a==load,projected,self.bcs,solver_parameters={'linear_solver':cfg.linear_solver})
                for name,fn in zip(self.names,projected.split(True)):history[name].assign(fn)
        self.proxies={'xi':TraceProxy(self.mesh,2,2),'phi':TraceProxy(self.mesh,cfg.phi_degree,1)}
        remote={k+suffix:proxy.functions[i] for k,proxy in self.proxies.items() for i,suffix in enumerate(('', '1'))}
        F=fluid_form(self.V,self.dt,{'u':self.old['u'],'u1':self.older['u']},remote,g,m,cfg.L,cfg.gamma,self.source)
        self.a=lhs(F);self.load=rhs(F)
        self.bulk_load=(m['rho_f']/(2*self.dt))*dot(4*self.old['u']-self.older['u'],v)*g['dx']+dot(self.source,v)*g['dx']
        matrix_start=clock.perf_counter()
        self.A=assemble(self.a)
        for bc in self.bcs:bc.apply(self.A)
        self.matrix_assembly_s=clock.perf_counter()-matrix_start
        self.solver=LUSolver(self.A,cfg.linear_solver)
    def update(self):
        fields=self.new.split(True);self.current=dict(zip(self.names,fields));return self.current
    def shift(self):
        for name in self.names:self.older[name].assign(self.old[name]);self.old[name].assign(self.current[name])
    def energy(self,current,previous):
        return .25*self.cfg.material['rho_f']*(norm2(current['u'],self.g['dx'])+norm2(2*current['u']-previous['u'],self.g['dx']))
    def errors(self):
        return {'err_u':math.sqrt(max(0.,norm2(self.exact_u-self.current['u'],self.g['dx']))),
                'err_p':math.sqrt(max(0.,norm2(self.exact_p-self.current['p'],self.g['dx'])))}

class BiotBlock:
    kind='biot';names=('eta','xi','phi')
    def __init__(self,cfg,comm):
        self.cfg=cfg;self.comm=comm;self.g=geometry(comm,cfg.nm,'biot');self.mesh=self.g['mesh'];g=self.g;m=cfg.material
        self.V=FunctionSpace(self.mesh,MixedElement([VectorElement('CG',self.mesh.ufl_cell(),2),VectorElement('CG',self.mesh.ufl_cell(),2),FiniteElement('CG',self.mesh.ufl_cell(),cfg.phi_degree)]))
        self.new=Function(self.V);self.old=dict(zip(self.names,Function(self.V).split(True)));self.older=dict(zip(self.names,Function(self.V).split(True)))
        self.time=Constant(0.);self.dt=Constant(cfg.dt)
        self.exact_eta,self.exact_xi,self.exact_phi,fe,fp=reference.make_biot_data(self.mesh,self.time)
        self.source_e=fe if cfg.forcing=='manufactured' else Constant((0.,0.))
        self.source_p=fp if cfg.forcing=='manufactured' else Constant(0.)
        self.bcs=[]
        for i in range(3):self.bcs+=zero_exterior(self.V.sub(i),g['markers'],(1,2,4),i<2)
        x,y=SpatialCoordinate(self.mesh);shape=x*x*(1-x)*(1-x)*(1+y)
        mode=cfg.mode_number if cfg.mode_number else max(1,cfg.nm//4)
        initial_phi=self.exact_phi
        if cfg.forcing=='zero' and cfg.initial_mode!='smooth':initial_phi=shape if cfg.initial_mode=='interface' else shape*sin(mode*math.pi*x)
        for value,history in ((0.,self.older),(cfg.dt if cfg.forcing=='manufactured' else 0.,self.old)):
            self.time.assign(value)
            if value:
                for exact,name in ((self.exact_eta,'eta'),(self.exact_xi,'xi')):
                    space=history[name].function_space();trial=TrialFunction(space);test=TestFunction(space)
                    a=(2*m['mu_b']*inner(epsilon(trial),epsilon(test))+m['lambda_b']*div(trial)*div(test))*g['dx']
                    load=(2*m['mu_b']*inner(epsilon(exact),epsilon(test))+m['lambda_b']*div(exact)*div(test))*g['dx']
                    solve(a==load,history[name],zero_exterior(space,g['markers'],(1,2,4),True),solver_parameters={'linear_solver':cfg.linear_solver})
            space=history['phi'].function_space();trial=TrialFunction(space);test=TestFunction(space)
            solve(m['K']*dot(grad(trial),grad(test))*g['dx']==m['K']*dot(grad(initial_phi),grad(test))*g['dx'],history['phi'],zero_exterior(space,g['markers'],(1,2,4),False),solver_parameters={'linear_solver':cfg.linear_solver})
        self.proxies={'u':TraceProxy(self.mesh,2,2)}
        remote={'u':self.proxies['u'].functions[0],'u1':self.proxies['u'].functions[1]}
        old={name+suffix:history[name] for history,suffix in ((self.old,''),(self.older,'1')) for name in self.names}
        Fpe,Fpd=biot_forms(self.V,self.dt,old,remote,g,m,cfg.L,cfg.gamma,cfg.beta,self.source_e,self.source_p)
        self.a=lhs(Fpe+Fpd);self.load=rhs(Fpe+Fpd)
        vp,zeta,psi=TestFunctions(self.V)
        self.bulk_load=(m['rho_b']/(2*self.dt))*dot(4*self.old['xi']-self.older['xi'],zeta)*g['dx']-(1/(2*self.dt))*dot(4*self.old['eta']-self.older['eta'],vp)*g['dx']+(m['C0']/(2*self.dt))*(4*self.old['phi']-self.older['phi'])*psi*g['dx']+dot(self.source_e,zeta)*g['dx']+self.source_p*psi*g['dx']
        matrix_start=clock.perf_counter()
        self.A=assemble(self.a)
        for bc in self.bcs:bc.apply(self.A)
        self.matrix_assembly_s=clock.perf_counter()-matrix_start
        self.solver=LUSolver(self.A,cfg.linear_solver)
    def update(self):self.current=dict(zip(self.names,self.new.split(True)));return self.current
    def shift(self):
        for name in self.names:self.older[name].assign(self.old[name]);self.old[name].assign(self.current[name])
    def energy(self,current,previous):
        m=self.cfg.material;g=self.g;total=0.
        for factor,fields in ((1.,current),(1.,{k:2*current[k]-previous[k] for k in self.names})):
            total+=assemble((m['rho_b']*inner(fields['xi'],fields['xi'])+m['C0']*fields['phi']**2+2*m['mu_b']*inner(epsilon(fields['eta']),epsilon(fields['eta']))+m['lambda_b']*div(fields['eta'])**2)*g['dx'])
        return .25*float(total)
    def errors(self):
        m=self.cfg.material;g=self.g;e=self.exact_eta-self.current['eta']
        return {'err_eta':math.sqrt(max(0.,float(assemble((2*m['mu_b']*inner(epsilon(e),epsilon(e))+m['lambda_b']*div(e)**2)*g['dx'])))),
                'err_xi':math.sqrt(max(0.,norm2(self.exact_xi-self.current['xi'],g['dx']))),
                'err_phi':math.sqrt(max(0.,norm2(self.exact_phi-self.current['phi'],g['dx'])))}

class TraceProxy:
    """Exact nodal trace transfer; no nearest-neighbor approximation."""
    def __init__(self,mesh,degree,components):
        self.S=FunctionSpace(mesh,'CG',degree);self.components=components
        self.V=VectorFunctionSpace(mesh,'CG',degree) if components==2 else self.S
        self.functions=[Function(self.V),Function(self.V)]
        self.parts=[[Function(self.S) for _ in range(components)] for _ in range(2)]
        self.assigner=FunctionAssigner(self.V,[self.S,self.S]) if components==2 else None
    def receive(self,records,level):
        lookup={(round(float(x),12),int(c)):float(value) for x,c,value in records}
        coords=self.S.tabulate_dof_coordinates().reshape(-1,2)
        global_ids=self.S.dofmap().tabulate_local_to_global_dofs();start,end=self.parts[level][0].vector().local_range()
        for c in range(self.components):
            values=np.zeros(end-start)
            for point,global_id in zip(coords,global_ids):
                if start<=global_id<end and abs(point[1])<1e-12:
                    key=(round(float(point[0]),12),c)
                    if key not in lookup:raise ValueError('Missing exact interface dof '+str(key))
                    values[global_id-start]=lookup[key]
            vector=self.parts[level][c].vector();vector.set_local(values);vector.apply('insert');vector.update_ghost_values()
        if self.components==2:self.assigner.assign(self.functions[level],self.parts[level])
        else:self.functions[level].vector().set_local(self.parts[level][0].vector().get_local());self.functions[level].vector().apply('insert')
        self.functions[level].vector().update_ghost_values()


def owned_records(function,interface_only=False):
    components=function.ufl_shape[0] if function.ufl_shape else 1
    parts=function.split(True) if components>1 else [function]
    rows=[]
    for c,part in enumerate(parts):
        space=part.function_space();coords=space.tabulate_dof_coordinates().reshape(-1,2)
        global_ids=space.dofmap().tabulate_local_to_global_dofs();start,end=part.vector().local_range();values=part.vector().get_local()
        for point,global_id in zip(coords,global_ids):
            if start<=global_id<end and (not interface_only or abs(point[1])<1e-12):
                rows.append([float(point[0]),float(c),float(values[global_id-start])] if interface_only else [float(point[0]),float(point[1]),float(c),float(values[global_id-start])])
    return np.asarray(rows,dtype=float).reshape((-1,3 if interface_only else 4))


def collect_records(function,comm,interface_only=False):
    local=owned_records(function,interface_only);parts=comm.gather(local,root=0)
    if comm.rank!=0:return None
    records=np.vstack(parts)
    if interface_only:
        keys=[(round(x,12),int(c)) for x,c,_ in records]
        if len(set(keys))!=len(keys):raise ValueError('Duplicate owned interface dofs')
        return records[np.lexsort((records[:,1],records[:,0]))]
    keys=[(round(x,12),round(y,12),int(c)) for x,y,c,_ in records]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate owned field dofs')
    return records[np.lexsort((records[:,2],records[:,1],records[:,0]))]


def pack_traces(block):
    fields=('u',) if block.kind=='fluid' else ('xi','phi')
    packet={}
    for name in fields:
        for level,history in enumerate((block.old,block.older)):
            packet[(name,level)]=collect_records(history[name],block.comm,True)
    return packet if block.comm.rank==0 else None


def receive_traces(block,packet):
    for (name,level),records in packet.items():block.proxies[name].receive(records,level)


def assemble_rhs(block,step,bulk=False):
    block.time.assign(step*block.cfg.dt)
    b=assemble(block.bulk_load if bulk else block.load)
    for bc in block.bcs:bc.apply(b)
    return b


def snapshot(block,history='current'):
    return {name:collect_records(function,block.comm,False) for name,function in getattr(block,history).items()}
