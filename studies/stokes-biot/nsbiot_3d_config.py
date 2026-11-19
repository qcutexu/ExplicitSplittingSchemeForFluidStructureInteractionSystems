"""Explicit CGS configuration for the 3D fixed-reference Stokes–Biot model.

K is hydraulic mobility in Darcy's law q=-K grad(phi). It is not divided by
fluid viscosity. The isotropic scalar represents the tensor K times I.
No mesh-unit conversion or Robin-parameter default is inferred here.
"""
import json
import math
from pathlib import Path

MATERIAL_KEYS=('rho_f','mu_f','rho_b','mu_b','lambda_b','alpha','C0','K','gamma')
UNITS={'rho_f':'g/cm^3','mu_f':'g/(cm*s)','rho_b':'g/cm^3',
       'mu_b':'dyn/cm^2','lambda_b':'dyn/cm^2','alpha':'dimensionless',
       'C0':'cm^2/dyn','K':'cm^3*s/g','gamma':'dyn*s/cm^3','L':'dyn*s/cm^3'}

def material_values(cfg):
    data=cfg.get('physical_coefficients')
    if not isinstance(data,dict) or set(data)!=set(MATERIAL_KEYS):
        raise ValueError('Specify all physical coefficients explicitly: '+', '.join(MATERIAL_KEYS))
    values={key:float(data[key]) for key in MATERIAL_KEYS}
    if any(not math.isfinite(v) or v<=0 for v in values.values()):
        raise ValueError('Physical coefficients must be finite and positive')
    return values

def pending_choices(cfg):
    missing=[]
    if cfg.get('numerical_robin_L') is None:missing.append('numerical_robin_L')
    if cfg.get('mesh_length_unit') is None:missing.append('mesh_length_unit')
    if cfg.get('mesh_coordinate_scale') is None:missing.append('mesh_coordinate_scale')
    if cfg.get('time_unit') is None:missing.append('time_unit')
    if cfg.get('inlet_velocity_unit') is None:missing.append('inlet_velocity_unit')
    if cfg.get('interface_discretization') is None:missing.append('interface_discretization')
    if cfg.get('initial_fields_and_volume_sources') is None:missing.append('initial_fields_and_volume_sources')
    return missing

def load_config(path,require_complete=True):
    cfg=json.loads(Path(path).read_text());material_values(cfg)
    if require_complete:
        missing=pending_choices(cfg)
        if missing:raise ValueError('Unconfirmed 3D setup choices: '+', '.join(missing))
        if cfg['interface_discretization'].get('type')!='common_fluid_surface_closest_point_quadrature':raise ValueError('Unsupported interface discretization')
        if cfg['initial_fields_and_volume_sources'].get('initial_fields')!='zero at t=0':raise ValueError('Only the confirmed zero initial fields are implemented')
        value=float(cfg['numerical_robin_L'])
        if not math.isfinite(value) or value<=0:raise ValueError('L must be finite and positive')
    return cfg
