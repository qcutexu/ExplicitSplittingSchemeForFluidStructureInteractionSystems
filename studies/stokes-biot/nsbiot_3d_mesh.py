"""Read the provided reference tetrahedral meshes with the FSI label convention."""
from pathlib import Path
import math
from fenics import *
from mpi4py import MPI as mpi

DEFAULT_MESH_DIR=Path(__file__).resolve().parent/'FSI_parallel_3D/mesh'
LABELS={'fluid':{1:'inlet',2:'outlet',5:'interface'},
        'biot':{2:'exterior',3:'narrow outlet end',4:'wide outlet end',5:'interface',6:'inlet end'}}

def read_geometry(kind,comm,mesh_dir=DEFAULT_MESH_DIR,coordinate_scale=1.0):
    coordinate_scale=float(coordinate_scale)
    if not math.isfinite(coordinate_scale) or coordinate_scale<=0:raise ValueError("coordinate_scale must be finite and positive")
    mesh_dir=Path(mesh_dir)
    if kind not in LABELS:raise ValueError('kind must be fluid or biot')
    mesh=Mesh(comm)
    with XDMFFile(comm,str(mesh_dir/('lumen0.xdmf' if kind=='fluid' else 'artery0.xdmf'))) as f:f.read(mesh)
    if mesh.topology().dim()!=3 or mesh.geometry().dim()!=3:raise ValueError('Expected a 3D tetrahedral reference mesh')
    if mesh.ufl_cell().cellname()!='tetrahedron':raise ValueError('Expected tetrahedra')
    if kind=='fluid':
        markers=MeshFunction('size_t',mesh,2,0)
        class Exterior(SubDomain):
            def inside(self,x,on_boundary):return on_boundary
        class End(SubDomain):
            def __init__(self,position):super().__init__();self.position=position
            def inside(self,x,on_boundary):return on_boundary and near(x[0],self.position,1e-2)
        Exterior().mark(markers,5);End(18.894).mark(markers,1);End(1.931).mark(markers,2)
    else:
        mvc=MeshValueCollection('size_t',mesh,2)
        with XDMFFile(comm,str(mesh_dir/'artery0_boundary.xdmf')) as f:f.read(mvc,'marker')
        markers=MeshFunction('size_t',mesh,mvc)
    # Mark/read facets in the source coordinates first; then transform the
    # geometry once. Existing facet identities and tags remain unchanged.
    mesh.coordinates()[:]*=coordinate_scale
    mesh.bounding_box_tree().build(mesh)
    return {'mesh':mesh,'markers':markers,'coordinate_scale':coordinate_scale,'n':FacetNormal(mesh),
            'dx':Measure('dx',domain=mesh),'ds':Measure('ds',domain=mesh,subdomain_data=markers)}

def geometry_summary(g,comm):
    mesh=g['mesh'];mesh.init(2,3)
    owned_cells=sum(not c.is_ghost() for c in cells(mesh))
    areas={str(tag):float(assemble(Constant(1.)*g['ds'](tag))) for tag in sorted(set(LABELS['fluid'])|set(LABELS['biot']))}
    return {'dimension':3,'tetrahedra':comm.allreduce(owned_cells,op=mpi.SUM),'boundary_areas':areas}


def read_configured_geometry(kind,comm,cfg,mesh_dir=DEFAULT_MESH_DIR):
    if cfg.get('mesh_length_unit')!='cm':raise ValueError('The physical coefficients require geometry in cm')
    return read_geometry(kind,comm,mesh_dir,coordinate_scale=cfg['mesh_coordinate_scale'])
