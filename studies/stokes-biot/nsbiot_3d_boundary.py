"""Explicit boundary specification for the reference 3D arterial meshes.

Pore-pressure BCs and physical coefficients have no inferred FSI defaults.
A complete user configuration is required before assembling a 3D problem.
"""
from fenics import *

# Preserve the sample's product profile and time waveform while scaling all
# spatial constants together with the geometry. Velocity is in cm/s, time in s.
INLET=("amplitude*(1-pow((x[1]-yc)/radius,2))*(1-pow((x[2]-zc)/radius,2))*(1-cos(2*pi*t/period))/2", "0", "0")

def make_inlet(cfg,time=0.,comm=None):
    if comm is None:comm=MPI.comm_world
    scale=float(cfg['mesh_coordinate_scale']);p=cfg['inlet_profile']
    return Expression(INLET,mpi_comm=comm,degree=4,t=float(time),amplitude=float(p['amplitude']),
                      period=float(p['period']),yc=float(p['raw_y_center'])*scale,
                      zc=float(p['raw_z_center'])*scale,radius=float(p['raw_radius'])*scale)

def validate_boundary_config(cfg):
    required={'2','3','4','6'}
    if set(cfg.get('pore',{}))!=required:raise ValueError('Specify pore conditions for exactly Biot exterior tags 2, 3, 4, 6; tag 5 is coupled')
    for tag,spec in cfg['pore'].items():
        if spec.get('type') not in ('pressure','flux'):raise ValueError(f'Tag {tag}: type must be pressure or flux')
        if 'value' not in spec:raise ValueError(f'Tag {tag}: specify the value explicitly')
    for tag in ('3','4','6'):
        spec=cfg['biot_mechanics'][tag]
        if spec.get('type')!='clamped_eta_and_xi' or list(spec['value'])!=[0,0,0]:raise ValueError('Only confirmed zero clamps on Biot 3,4,6 are implemented')
    for domain,tag in [('fluid','2'),('biot_mechanics','2')]:
        spec=cfg[domain][tag]
        if spec.get('type')!='total_traction' or list(spec['value'])!=[0,0,0]:raise ValueError('Only confirmed zero natural exterior tractions are implemented')
    if cfg['fluid']['1'].get('type')!='velocity':raise ValueError('Fluid inlet must use the configured velocity')
    return cfg

def exterior_bcs(kind,V,markers,time,cfg):
    validate_boundary_config(cfg)
    zero=Constant((0.,0.,0.))
    if kind=='fluid':
        inlet=make_inlet(cfg,time,V.mesh().mpi_comm())
        return [DirichletBC(V.sub(0),inlet,markers,1)],inlet
    bcs=[DirichletBC(V.sub(field),zero,markers,tag) for field in (0,1) for tag in (3,4,6)]
    for tag,spec in cfg['pore'].items():
        if spec['type']=='pressure':bcs.append(DirichletBC(V.sub(2),Constant(float(spec['value'])),markers,int(tag)))
    return bcs,None

def pore_outward_flux_load(psi,ds,cfg):
    # q_n=-K grad(phi).n: integration by parts puts -q_n psi on the RHS.
    validate_boundary_config(cfg)
    return sum(-Constant(float(spec['value']))*psi*ds(int(tag)) for tag,spec in cfg['pore'].items() if spec['type']=='flux')
