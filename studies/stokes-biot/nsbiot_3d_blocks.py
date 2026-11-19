"""3D bulk blocks, exact DOF/trace maps and zero-rest histories."""
import numpy as np
from fenics import *
from mpi4py import MPI as mpi
from nsbiot_3d_forms import fluid_bulk,biot_bulk,epsilon
from nsbiot_3d_boundary import exterior_bcs,pore_outward_flux_load,make_inlet
from nsbiot_3d_mesh import read_configured_geometry
from nsbiot_3d_surface import key,triangle_nodes
from nsbiot_3d_config import material_values

class Block3D:
    def __init__(self,kind,comm,cfg,dt,mesh_dir=None,fixture_n=None,phi_degree=2):
        self.kind=kind;self.comm=comm;self.cfg=cfg;self.m=material_values(cfg);self.phi_degree=phi_degree
        if fixture_n is None:
            kw={} if mesh_dir is None else {'mesh_dir':mesh_dir}
            self.g=read_configured_geometry(kind,comm,cfg,**kw)
        else:self.g=fixture_geometry(kind,comm,fixture_n)
        self.mesh=self.g['mesh'];cell=self.mesh.ufl_cell()
        vec=VectorElement('CG',cell,2)
        els=[vec,FiniteElement('CG',cell,1)] if kind=='fluid' else [vec,vec,FiniteElement('CG',cell,phi_degree)]
        self.names=('u','p') if kind=='fluid' else ('eta','xi','phi')
        self.V=FunctionSpace(self.mesh,MixedElement(els));self.new=Function(self.V);self.old=Function(self.V);self.older=Function(self.V)
        self.bcs,self.inlet=exterior_bcs(kind,self.V,self.g['markers'],0.,cfg)
        if fixture_n is not None and kind=='biot':
            self.bcs=[DirichletBC(self.V.sub(i),Constant((0.,0.,0.)),self.g['markers'],6) for i in (0,1)]
        if fixture_n is not None and kind=='fluid':
            # A small, explicitly distinct inlet fixture. Production always
            # uses the supplied profile. This profile vanishes on edge walls.
            self.inlet=Expression(('0','0','-A*16*x[0]*(1-x[0])*x[1]*(1-x[1])*(1-cos(2*pi*t))/2'),degree=4,A=.1,t=0.,mpi_comm=comm)
            self.bcs=[DirichletBC(self.V.sub(0),self.inlet,self.g['markers'],1)]
        self.weights=[Constant(v) for v in (3.,4.,1.)];self.denominator=Constant(2.);self.dt=Constant(dt)
        old=dict(zip(self.names,split(self.old)));older=dict(zip(self.names,split(self.older)))
        history={**old,**{k+'1':v for k,v in older.items()}}
        sources=cfg['initial_fields_and_volume_sources']
        if kind=='fluid':F=fluid_bulk(self.V,self.dt,history,self.m,Constant(tuple(sources['fluid_volume_force'])),self.g['dx'],self.weights,self.denominator)
        else:
            F=biot_bulk(self.V,self.dt,history,self.m,Constant(tuple(sources['biot_volume_force'])),Constant(float(sources['pore_volume_source'])),self.g['dx'],self.weights,self.denominator)
            # Prescribed flux is an outward Darcy flux; subtract its RHS load.
            F-=pore_outward_flux_load(TestFunctions(self.V)[2],self.g['ds'],cfg)
        self.a=lhs(F);self.load=rhs(F)
        self.surface,self.normals=self.extract_surface()
        self.trace_map=self.extract_trace_map()
    def set_scheme(self,first):
        for c,v in zip(self.weights,(1.,1.,0.) if first else (3.,4.,1.)):c.assign(v)
        self.denominator.assign(1. if first else 2.)
    def extract_surface(self):
        mesh=self.mesh;mesh.init(2,0);mesh.init(2,3);tri=[];normals=[]
        for facet in facets(mesh):
            if self.g['markers'][facet]!=5 or not facet.exterior():continue
            owners=[int(i) for i in facet.entities(3) if not Cell(mesh,int(i)).is_ghost()]
            if not owners:continue
            points=mesh.coordinates()[facet.entities(0)]
            n=np.asarray(facet.normal().array());center=Cell(mesh,owners[0]).midpoint().array()
            if np.dot(n,points.mean(axis=0)-center)<0:n=-n
            points=np.asarray(sorted(points,key=key))
            tri.append(points);normals.append(n)
        parts=self.comm.allgather((np.asarray(tri).reshape(-1,3,3),np.asarray(normals).reshape(-1,3)))
        triangles=np.concatenate([a for a,b in parts]);ns=np.concatenate([b for a,b in parts])
        # Deterministic geometric ordering, independent of partition numbering.
        order=sorted(range(len(triangles)),key=lambda i:tuple(sorted(key(x) for x in triangles[i])))
        return triangles[order],ns[order]
    def extract_trace_map(self):
        dm=self.V.dofmap();lg=dm.tabulate_local_to_global_dofs();lo,hi=dm.ownership_range();rev={int(g):j for j,g in enumerate(lg)}
        coords=self.V.tabulate_dof_coordinates().reshape(-1,3)
        specifications=[('u',0,3,2)] if self.kind=='fluid' else [('xi',1,3,2),('phi',2,1,self.phi_degree)]
        local={}
        for name,field,ncomp,degree in specifications:
            nodes={key(x) for x in triangle_nodes(self.surface,degree).reshape(-1,3)}
            for c in range(ncomp):
                sub=self.V.sub(field) if ncomp==1 else self.V.sub(field).sub(c)
                for gid in sub.dofmap().dofs():
                    if lo<=gid<hi:
                        point=key(coords[rev[int(gid)]])
                        if point in nodes:local[(name,c,point)]=int(gid)
        maps={}
        for part in self.comm.allgather(local):
            if set(maps)&set(part):raise ValueError('Duplicate owned 3D trace DOF')
            maps.update(part)
        return maps
    def rhs(self,t):
        if self.inlet is not None:self.inlet.t=t
        return assemble(self.load)
    def shift(self):self.older.assign(self.old);self.old.assign(self.new)
    def energy_and_volume(self):
        f=dict(zip(self.names,split(self.new)));m=self.m;dx=self.g['dx']
        if self.kind=='fluid':density=m['rho_f']*inner(f['u'],f['u'])
        else:density=m['rho_b']*inner(f['xi'],f['xi'])+m['C0']*f['phi']**2+2*m['mu_b']*inner(epsilon(f['eta']),epsilon(f['eta']))+m['lambda_b']*div(f['eta'])**2
        return .5*float(assemble(density*dx))
    def snapshot(self,level='new'):
        function=getattr(self,level);out={}
        for name,fn in zip(self.names,function.split(True)):
            components=fn.split(True) if fn.ufl_shape else [fn];parts=[]
            for c,part in enumerate(components):
                V=part.function_space();coords=V.tabulate_dof_coordinates().reshape(-1,3);ids=V.dofmap().tabulate_local_to_global_dofs();lo,hi=part.vector().local_range();values=part.vector().get_local()
                rows=[[*x,c,values[int(g)-lo]] for x,g in zip(coords,ids) if lo<=g<hi]
                parts.append(np.asarray(rows).reshape(-1,5))
            gathered=self.comm.gather(np.concatenate(parts),root=0)
            if self.comm.rank==0:
                rows=np.concatenate(gathered);out[name]=rows[np.lexsort((rows[:,3],rows[:,2],rows[:,1],rows[:,0]))]
        return out

def fixture_geometry(kind,comm,n):
    # Adjacent cubes, independently triangulated if different n is requested.
    lo,hi=(0.,1.) if kind=='fluid' else (-1.,0.)
    mesh=BoxMesh(comm,Point(0,0,lo),Point(1,1,hi),n,n,n)
    tags=MeshFunction('size_t',mesh,2,0)
    class Boundary(SubDomain):
        def inside(self,x,b):return b
    class Plane(SubDomain):
        def __init__(self,z):super().__init__();self.z=z
        def inside(self,x,b):return b and near(x[2],self.z)
    Boundary().mark(tags,2 if kind=='biot' else 1)
    Plane(hi if kind=='fluid' else lo).mark(tags,1 if kind=='fluid' else 6)
    if kind=='fluid':
        class Sides(SubDomain):
            def inside(self,x,b):return b and (near(x[0],0) or near(x[0],1) or near(x[1],0) or near(x[1],1))
        Sides().mark(tags,2)
    Plane(0.).mark(tags,5)
    return {'mesh':mesh,'markers':tags,'n':FacetNormal(mesh),'dx':Measure('dx',domain=mesh),'ds':Measure('ds',domain=mesh,subdomain_data=tags)}
