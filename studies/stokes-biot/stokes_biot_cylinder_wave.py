#!/usr/bin/env python3
"""Standalone legacy FEniCS example: pressure wave in a cylinder with a thick Biot wall.

CGS units. Default inner radius .30 cm, thickness .06 cm, length 6 cm.
A 1000 dyn/cm^2 sine-squared inlet traction pulse lasts .002 s. Fluid outlet
traction is zero; wall end rings are clamped; outer total traction and pore
flux are zero. Initial fields and volume forces/sources are zero.

Both monolithic BDF2 and beta=0 BDF2/EX2 NR are included. A BE first step starts
from rest. This file needs no other project Python files. The default Gmsh mode generates
a genuinely unstructured, interface-conforming 3D tetrahedral mesh using one
fragmented CAD model; the optional layered mode uses NumPy only. The wall has
true volume elements. Gmsh is an additional installed-library dependency.

Mesh-only preparation (current defaults: 24 angular, 3 fluid-radial, 3 wall-radial, 24 axial):
 FI_PROVIDER=tcp "$CONDA_PREFIX/bin/mpiexec" -n 2 "$CONDA_PREFIX/bin/python" stokes_biot_cylinder_wave.py --method monolithic --mesh-generator gmsh --mesh-only --output-dir results/cylinder_unstructured

All five fields are written for ParaView and native HDF5 readback. Wave probes
record pressure, axial velocity and radial wall displacement along the tube.
Zero outlet traction reflects waves; this is not an absorbing outlet model.
"""
import argparse,csv,json,time,os,math,copy,traceback
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from fenics import *
from mpi4py import MPI as mpi
from petsc4py import PETSc as petsc

DEFAULT_CONFIGURATION = {'fluid': {'1': {'type': 'velocity', 'profile': 'literal FSI_parallel_3D/Fluid.py inlet waveform', 'confirmed': True}, '2': {'type': 'total_traction', 'value': [0, 0, 0], 'confirmed': True}, '5': {'type': 'coupled_interface'}}, 'biot_mechanics': {'2': {'type': 'total_traction', 'value': [0, 0, 0], 'confirmed': True}, '3': {'type': 'clamped_eta_and_xi', 'value': [0, 0, 0], 'confirmed': True}, '4': {'type': 'clamped_eta_and_xi', 'value': [0, 0, 0], 'confirmed': True}, '6': {'type': 'clamped_eta_and_xi', 'value': [0, 0, 0], 'confirmed': True}, '5': {'type': 'coupled_interface'}}, 'pore': {'2': {'type': 'flux', 'value': 0}, '3': {'type': 'flux', 'value': 0}, '4': {'type': 'flux', 'value': 0}, '6': {'type': 'flux', 'value': 0}}, 'interface_discretization': {'type': 'common_fluid_surface_closest_point_quadrature', 'biot_pullback': 'closest point on Biot tag 5 triangles', 'normal': 'fluid outward normal; Biot normal is its negative on common surface', 'quadrature_order': 4, 'projection_refinement': 3}, 'physical_coefficients': {'rho_f': 1.0, 'mu_f': 0.035, 'rho_b': 1.1, 'mu_b': 1677900.0, 'lambda_b': 82215000.0, 'alpha': 1.0, 'C0': 0.001, 'K': 1e-08, 'gamma': 1000.0}, 'initial_fields_and_volume_sources': {'initial_fields': 'zero at t=0', 'fluid_volume_force': [0, 0, 0], 'biot_volume_force': [0, 0, 0], 'pore_volume_source': 0}, 'mesh_length_unit': 'cm', 'time_unit': 's', 'coefficient_units': {'rho_f': 'g/cm^3', 'mu_f': 'g/(cm*s)', 'rho_b': 'g/cm^3', 'mu_b': 'dyn/cm^2', 'lambda_b': 'dyn/cm^2', 'alpha': 'dimensionless', 'C0': 'cm^2/dyn', 'K': 'cm^3*s/g', 'gamma': 'dyn*s/cm^3', 'L': 'dyn*s/cm^3'}, 'coefficient_source': 'User-supplied CGS parameter table', 'parameter_mapping': {'rho_p': 'rho_b', 'mu_p': 'mu_b', 'lambda_p': 'lambda_b'}, 'K_representation': 'Isotropic hydraulic mobility tensor K I, implemented as scalar K multiplying grad(phi); no extra division by mu_f', 'numerical_robin_L': 1500.0, 'inlet_velocity_unit': 'cm/s', 'mesh_coordinate_scale': 0.25, 'inlet_profile': {'amplitude': -35.0, 'period': 1.0, 'raw_y_center': 4.38452, 'raw_z_center': 2.96784, 'raw_radius': 1.2383, 'spatial_constants_scale_with_geometry': True}, 'startup': 'backward Euler at first step; NR EX1 then BDF2/EX2, monolithic current coupling throughout'}
MATERIAL_KEYS = ('rho_f', 'mu_f', 'rho_b', 'mu_b', 'lambda_b', 'alpha', 'C0', 'K', 'gamma')
UNITS = {'rho_f': 'g/cm^3', 'mu_f': 'g/(cm*s)', 'rho_b': 'g/cm^3', 'mu_b': 'dyn/cm^2', 'lambda_b': 'dyn/cm^2', 'alpha': 'dimensionless', 'C0': 'cm^2/dyn', 'K': 'cm^3*s/g', 'gamma': 'dyn*s/cm^3', 'L': 'dyn*s/cm^3'}

def material_values(cfg):
    data = cfg.get('physical_coefficients')
    if not isinstance(data, dict) or set(data) != set(MATERIAL_KEYS):
        raise ValueError('Specify all physical coefficients explicitly: ' + ', '.join(MATERIAL_KEYS))
    values = {key: float(data[key]) for key in MATERIAL_KEYS}
    if any((not math.isfinite(v) or v <= 0 for v in values.values())):
        raise ValueError('Physical coefficients must be finite and positive')
    return values

def pending_choices(cfg):
    missing = []
    if cfg.get('numerical_robin_L') is None:
        missing.append('numerical_robin_L')
    if cfg.get('mesh_length_unit') is None:
        missing.append('mesh_length_unit')
    if cfg.get('mesh_coordinate_scale') is None:
        missing.append('mesh_coordinate_scale')
    if cfg.get('time_unit') is None:
        missing.append('time_unit')
    if cfg.get('inlet_velocity_unit') is None:
        missing.append('inlet_velocity_unit')
    if cfg.get('interface_discretization') is None:
        missing.append('interface_discretization')
    if cfg.get('initial_fields_and_volume_sources') is None:
        missing.append('initial_fields_and_volume_sources')
    return missing

def load_config(path, require_complete=True):
    cfg = copy.deepcopy(DEFAULT_CONFIGURATION) if path is None else json.loads(Path(path).read_text())
    material_values(cfg)
    if require_complete:
        missing = pending_choices(cfg)
        if missing:
            raise ValueError('Unconfirmed 3D setup choices: ' + ', '.join(missing))
        if cfg['interface_discretization'].get('type') != 'common_fluid_surface_closest_point_quadrature':
            raise ValueError('Unsupported interface discretization')
        if cfg['initial_fields_and_volume_sources'].get('initial_fields') != 'zero at t=0':
            raise ValueError('Only the confirmed zero initial fields are implemented')
        value = float(cfg['numerical_robin_L'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError('L must be finite and positive')
    return cfg

def epsilon(u):
    return sym(grad(u))

def tangent(u, n):
    return u - dot(u, n) * n

def slip(u, v, n):
    return inner(tangent(u, n), tangent(v, n))

def fluid_bulk(V, dt, old, m, source, measure, weights=(3.0, 4.0, 1.0), denominator=2.0):
    u, p = TrialFunctions(V)
    v, q = TestFunctions(V)
    a0, a1, a2 = weights
    return (m['rho_f'] / (denominator * dt) * dot(a0 * u - a1 * old['u'] + a2 * old['u1'], v) + 2 * m['mu_f'] * inner(epsilon(u), epsilon(v)) - p * div(v) + div(u) * q - dot(source, v)) * measure

def biot_bulk(V, dt, old, m, source_e, source_p, measure, weights=(3.0, 4.0, 1.0), denominator=2.0):
    eta, xi, phi = TrialFunctions(V)
    vp, zeta, psi = TestFunctions(V)
    a0, a1, a2 = weights
    return (m['rho_b'] / (denominator * dt) * dot(a0 * xi - a1 * old['xi'] + a2 * old['xi1'], zeta) + 2 * m['mu_b'] * inner(epsilon(eta), epsilon(zeta)) + m['lambda_b'] * div(eta) * div(zeta) - m['alpha'] * phi * div(zeta) + dot(xi, vp) - dot(a0 * eta - a1 * old['eta'] + a2 * old['eta1'], vp) / (denominator * dt) + m['C0'] / (denominator * dt) * (a0 * phi - a1 * old['phi'] + a2 * old['phi1']) * psi + m['alpha'] * div(xi) * psi + m['K'] * dot(grad(phi), grad(psi)) - dot(source_e, zeta) - source_p * psi) * measure

def fluid_interface(u, v, xi_ex, phi_ex, u_ex, n, L, gamma, measure):
    return (L * dot(u - u_ex, n) * dot(v, n) + gamma * slip(u - xi_ex, v, n) + phi_ex * dot(v, n)) * measure

def biot_interface(xi, phi, zeta, psi, u_ex, phi_ex, n, L, gamma, measure):
    return (gamma * slip(xi - u_ex, zeta, n) + phi * dot(zeta, n) + (phi - phi_ex) * psi / L - dot(xi, n) * psi + dot(u_ex, n) * psi) * measure

def coupled_interface(u, xi, phi, v, zeta, psi, nf, gamma, measure):
    """Common-surface physical coupling; the Biot normal is -nf.

    The pressure/normal-flow terms are skew under energy testing. L and beta
    do not occur. Arguments may be pulled-back traces of different meshes.
    """
    return (gamma * slip(u - xi, v - zeta, nf) + phi * dot(v - zeta, nf) - dot(u - xi, nf) * psi) * measure
DEFAULT_MESH_DIR = Path(__file__).resolve().parent / 'FSI_parallel_3D/mesh'
LABELS = {'fluid': {1: 'inlet', 2: 'outlet', 5: 'interface'}, 'biot': {2: 'exterior', 3: 'narrow outlet end', 4: 'wide outlet end', 5: 'interface', 6: 'inlet end'}}

def read_geometry(kind, comm, mesh_dir=DEFAULT_MESH_DIR, coordinate_scale=1.0):
    coordinate_scale = float(coordinate_scale)
    if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be finite and positive')
    mesh_dir = Path(mesh_dir)
    if kind not in LABELS:
        raise ValueError('kind must be fluid or biot')
    mesh = Mesh(comm)
    with XDMFFile(comm, str(mesh_dir / ('lumen0.xdmf' if kind == 'fluid' else 'artery0.xdmf'))) as f:
        f.read(mesh)
    if mesh.topology().dim() != 3 or mesh.geometry().dim() != 3:
        raise ValueError('Expected a 3D tetrahedral reference mesh')
    if mesh.ufl_cell().cellname() != 'tetrahedron':
        raise ValueError('Expected tetrahedra')
    if kind == 'fluid':
        markers = MeshFunction('size_t', mesh, 2, 0)

        class Exterior(SubDomain):

            def inside(self, x, on_boundary):
                return on_boundary

        class End(SubDomain):

            def __init__(self, position):
                super().__init__()
                self.position = position

            def inside(self, x, on_boundary):
                return on_boundary and near(x[0], self.position, 0.01)
        Exterior().mark(markers, 5)
        End(18.894).mark(markers, 1)
        End(1.931).mark(markers, 2)
    else:
        mvc = MeshValueCollection('size_t', mesh, 2)
        with XDMFFile(comm, str(mesh_dir / 'artery0_boundary.xdmf')) as f:
            f.read(mvc, 'marker')
        markers = MeshFunction('size_t', mesh, mvc)
    mesh.coordinates()[:] *= coordinate_scale
    mesh.bounding_box_tree().build(mesh)
    return {'mesh': mesh, 'markers': markers, 'coordinate_scale': coordinate_scale, 'n': FacetNormal(mesh), 'dx': Measure('dx', domain=mesh), 'ds': Measure('ds', domain=mesh, subdomain_data=markers)}

def geometry_summary(g, comm):
    mesh = g['mesh']
    mesh.init(2, 3)
    owned_cells = sum((not c.is_ghost() for c in cells(mesh)))
    areas = {str(tag): float(assemble(Constant(1.0) * g['ds'](tag))) for tag in sorted(set(LABELS['fluid']) | set(LABELS['biot']))}
    return {'dimension': 3, 'tetrahedra': comm.allreduce(owned_cells, op=mpi.SUM), 'boundary_areas': areas}

def read_configured_geometry(kind, comm, cfg, mesh_dir=DEFAULT_MESH_DIR):
    if cfg.get('mesh_length_unit') != 'cm':
        raise ValueError('The physical coefficients require geometry in cm')
    return read_geometry(kind, comm, mesh_dir, coordinate_scale=cfg['mesh_coordinate_scale'])
INLET = ('amplitude*(1-pow((x[1]-yc)/radius,2))*(1-pow((x[2]-zc)/radius,2))*(1-cos(2*pi*t/period))/2', '0', '0')

def make_inlet(cfg, time=0.0, comm=None):
    if comm is None:
        comm = MPI.comm_world
    scale = float(cfg['mesh_coordinate_scale'])
    p = cfg['inlet_profile']
    return Expression(INLET, mpi_comm=comm, degree=4, t=float(time), amplitude=float(p['amplitude']), period=float(p['period']), yc=float(p['raw_y_center']) * scale, zc=float(p['raw_z_center']) * scale, radius=float(p['raw_radius']) * scale)

def validate_boundary_config(cfg):
    required = {'2', '3', '4', '6'}
    if set(cfg.get('pore', {})) != required:
        raise ValueError('Specify pore conditions for exactly Biot exterior tags 2, 3, 4, 6; tag 5 is coupled')
    for tag, spec in cfg['pore'].items():
        if spec.get('type') not in ('pressure', 'flux'):
            raise ValueError(f'Tag {tag}: type must be pressure or flux')
        if 'value' not in spec:
            raise ValueError(f'Tag {tag}: specify the value explicitly')
    for tag in ('3', '4', '6'):
        spec = cfg['biot_mechanics'][tag]
        if spec.get('type') != 'clamped_eta_and_xi' or list(spec['value']) != [0, 0, 0]:
            raise ValueError('Only confirmed zero clamps on Biot 3,4,6 are implemented')
    for domain, tag in [('fluid', '2'), ('biot_mechanics', '2')]:
        spec = cfg[domain][tag]
        if spec.get('type') != 'total_traction' or list(spec['value']) != [0, 0, 0]:
            raise ValueError('Only confirmed zero natural exterior tractions are implemented')
    if cfg['fluid']['1'].get('type') != 'velocity':
        raise ValueError('Fluid inlet must use the configured velocity')
    return cfg

def exterior_bcs(kind, V, markers, time, cfg):
    validate_boundary_config(cfg)
    zero = Constant((0.0, 0.0, 0.0))
    if kind == 'fluid':
        inlet = make_inlet(cfg, time, V.mesh().mpi_comm())
        return ([DirichletBC(V.sub(0), inlet, markers, 1)], inlet)
    bcs = [DirichletBC(V.sub(field), zero, markers, tag) for field in (0, 1) for tag in (3, 4, 6)]
    for tag, spec in cfg['pore'].items():
        if spec['type'] == 'pressure':
            bcs.append(DirichletBC(V.sub(2), Constant(float(spec['value'])), markers, int(tag)))
    return (bcs, None)

def pore_outward_flux_load(psi, ds, cfg):
    validate_boundary_config(cfg)
    return sum((-Constant(float(spec['value'])) * psi * ds(int(tag)) for tag, spec in cfg['pore'].items() if spec['type'] == 'flux'))

def key(point):
    return tuple(np.round(point, 11))

def closest_on_triangles(points, triangles):
    a, b, c = (triangles[:, 0], triangles[:, 1], triangles[:, 2])
    e0 = b - a
    e1 = c - a
    v = points - a
    d00 = np.einsum('ij,ij->i', e0, e0)
    d01 = np.einsum('ij,ij->i', e0, e1)
    d11 = np.einsum('ij,ij->i', e1, e1)
    d20 = np.einsum('ij,ij->i', v, e0)
    d21 = np.einsum('ij,ij->i', v, e1)
    den = d00 * d11 - d01 * d01
    if np.any(den <= 0):
        raise ValueError('Degenerate interface triangle')
    l1 = (d11 * d20 - d01 * d21) / den
    l2 = (d00 * d21 - d01 * d20) / den
    bary = np.column_stack((1 - l1 - l2, l1, l2))
    inside = np.min(bary, axis=1) >= 0
    projected = np.einsum('ni,nij->nj', bary, triangles)
    best = np.where(inside, np.sum((projected - points) ** 2, axis=1), np.inf)
    for i, j in [(0, 1), (1, 2), (2, 0)]:
        edge = triangles[:, j] - triangles[:, i]
        t = np.clip(np.einsum('ij,ij->i', points - triangles[:, i], edge) / np.sum(edge * edge, axis=1), 0, 1)
        q = triangles[:, i] + t[:, None] * edge
        dist = np.sum((q - points) ** 2, axis=1)
        take = dist < best
        candidate = np.zeros_like(bary)
        candidate[:, i] = 1 - t
        candidate[:, j] = t
        bary[take] = candidate[take]
        projected[take] = q[take]
        best[take] = dist[take]
    return (projected, bary, np.sqrt(best))

class SurfaceProjection:

    def __init__(self, triangles):
        self.triangles = np.asarray(triangles)
        self.centers = self.triangles.mean(axis=1)
        self.radius = float(np.linalg.norm(self.triangles - self.centers[:, None, :], axis=2).max())
        self.tree = cKDTree(self.centers)

    def query(self, points):
        points = np.atleast_2d(points)
        _, seed = self.tree.query(points)
        _, _, upper = closest_on_triangles(points, self.triangles[seed])
        candidates = self.tree.query_ball_point(points, upper + self.radius + 1e-12)
        owners = np.repeat(np.arange(len(points)), [len(a) for a in candidates])
        ids = np.concatenate(candidates)
        q, bary, dist = closest_on_triangles(points[owners], self.triangles[ids])
        order = np.lexsort((ids, dist, owners))
        ordered = owners[order]
        selected = order[np.r_[True, ordered[1:] != ordered[:-1]]]
        return (ids[selected], q[selected], bary[selected], dist[selected])

def triangle_nodes(triangles, degree=2):
    if degree == 1:
        return triangles
    if degree != 2:
        raise ValueError('Surface degree must be 1 or 2')
    return np.concatenate((triangles, (triangles[:, 0:1] + triangles[:, 1:2]) / 2, (triangles[:, 1:2] + triangles[:, 2:3]) / 2, (triangles[:, 2:3] + triangles[:, 0:1]) / 2), axis=1)

def surface_space(triangles, degree=2):
    nodes = triangle_nodes(triangles, degree)
    coords = np.asarray(sorted({key(p) for p in nodes.reshape(-1, 3)}))
    lookup = {key(x): i for i, x in enumerate(coords)}
    ids = np.asarray([[lookup[key(p)] for p in row] for row in nodes], dtype=np.int64)
    return (coords, ids)

def basis(lam, degree=2):
    if degree == 1:
        return lam
    a, b, c = lam.T
    return np.column_stack((a * (2 * a - 1), b * (2 * b - 1), c * (2 * c - 1), 4 * a * b, 4 * b * c, 4 * c * a))

def triangle_rule(order):
    if order < 2:
        raise ValueError('Quadrature order must be >=2')
    x, w = np.polynomial.legendre.leggauss(order)
    x = (x + 1) / 2
    w = w / 2
    lam = []
    weights = []
    for i, u in enumerate(x):
        for j, v in enumerate(x):
            lam.append((1 - u - (1 - u) * v, u, (1 - u) * v))
            weights.append(w[i] * w[j] * (1 - u))
    return (np.asarray(lam), np.asarray(weights))

class InterfaceOperators:

    def __init__(self, fluid_triangles, biot_triangles, normals, material, L, phi_degree=2, order=4, refinement=3, biot_normals=None):
        self.fc, self.fi = surface_space(fluid_triangles, 2)
        self.bc, self.bi = surface_space(biot_triangles, 2)
        self.pc, self.pi = surface_space(biot_triangles, phi_degree)
        self.nf = 3 * len(self.fc)
        self.nx = 3 * len(self.bc)
        self.np = len(self.pc)
        self.N = self.nf + self.nx + self.np
        self.maps = {'u': np.arange(self.nf).reshape(-1, 3), 'xi': self.nf + np.arange(self.nx).reshape(-1, 3), 'phi': (self.nf + self.nx + np.arange(self.np))[:, None]}
        self.coordinates = {'u': self.fc, 'xi': self.bc, 'phi': self.pc}
        projection = SurfaceProjection(biot_triangles)
        rule, rule_weights = triangle_rule(order)
        if refinement < 0:
            raise ValueError('Projection refinement must be nonnegative')
        unresolved_area = 0.0
        patch_count = 0

        def patches(tri, vertices, depth=0):
            lam = rule @ vertices
            ids, q, bl, dist = projection.query(lam @ tri)
            feature = np.sum((bl < 1e-10) * np.array([1, 2, 4]), axis=1)
            consistent = np.all(ids == ids[0]) and np.all(feature == feature[0])
            if consistent or depth == refinement:
                yield (lam, rule_weights / 4 ** depth, ids, bl, dist, consistent)
            else:
                a, b, c = vertices
                ab = (a + b) / 2
                bc = (b + c) / 2
                ca = (c + a) / 2
                for sub in [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]:
                    yield from patches(tri, np.asarray(sub), depth + 1)
        dr = []
        dc = []
        dv = []
        rr = []
        rc = []
        rv = []
        distances = []
        alignments = []
        area = 0.0
        gamma = material['gamma']

        def put(target, rows, cols, values):
            rows = np.asarray(rows).reshape(-1)
            cols = np.asarray(cols).reshape(-1)
            a, b, c = (dr, dc, dv) if target == 'D' else (rr, rc, rv)
            a.append(np.repeat(rows, len(cols)))
            b.append(np.tile(cols, len(rows)))
            c.append(np.asarray(values).reshape(-1))
        D = sp.csr_matrix((self.N, self.N))
        R = D.copy()
        for ti, tri in enumerate(fluid_triangles):
            jac = np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            area += jac / 2
            n = normals[ti]
            P = np.eye(3) - np.outer(n, n)
            NN = np.outer(n, n)
            fids = self.maps['u'][self.fi[ti]].reshape(-1)
            pieces = list(patches(tri, np.eye(3)))
            patch_count += len(pieces)
            for lam, weights, ids, bl, dist, consistent in pieces:
                distances.extend(dist)
                if biot_normals is not None:
                    alignments.extend(-np.asarray(biot_normals)[ids] @ n)
                if not consistent:
                    unresolved_area += jac * sum(weights)
                bf = basis(lam)
                for qi, bt in enumerate(ids):
                    f = bf[qi]
                    b = basis(bl[qi:qi + 1])[0]
                    p = basis(bl[qi:qi + 1], phi_degree)[0]
                    w = weights[qi] * jac
                    xids = self.maps['xi'][self.bi[bt]].reshape(-1)
                    pids = self.maps['phi'][self.pi[bt]].reshape(-1)
                    ff = np.outer(f, f) * w
                    bb = np.outer(b, b) * w
                    fb = np.outer(f, b) * w
                    Nf = np.outer(f, p) * w
                    Nx = np.outer(b, p) * w
                    mass = np.outer(p, p) * w
                    F = np.kron(ff, L * NN)
                    Sff = np.kron(ff, gamma * P)
                    Sxx = np.kron(bb, gamma * P)
                    Sfx = np.kron(fb, gamma * P)
                    Fp = (Nf[:, :, None] * n).transpose(0, 2, 1).reshape(len(fids), len(pids))
                    Xp = (Nx[:, :, None] * n).transpose(0, 2, 1).reshape(len(xids), len(pids))
                    put('D', fids, fids, F + Sff)
                    put('D', xids, xids, Sxx)
                    put('D', xids, pids, -Xp)
                    put('D', pids, xids, Xp.T)
                    put('D', pids, pids, mass / L)
                    put('R', fids, fids, F)
                    put('R', fids, xids, Sfx)
                    put('R', xids, fids, Sfx.T)
                    put('R', fids, pids, -Fp)
                    put('R', pids, fids, Fp.T)
                    put('R', pids, pids, mass / L)
            if (ti + 1) % 64 == 0 or ti + 1 == len(fluid_triangles):
                for target, a, b, c in [('D', dr, dc, dv), ('R', rr, rc, rv)]:
                    block = sp.coo_matrix((np.concatenate(c), (np.concatenate(a), np.concatenate(b))), shape=D.shape).tocsr()
                    if target == 'D':
                        D += block
                    else:
                        R += block
                    a.clear()
                    b.clear()
                    c.clear()
        self.D = D
        self.R = R
        self.H = (D - R).tocsr()
        self.H.eliminate_zeros()
        distances = np.asarray(distances)
        if alignments and min(alignments) <= 0:
            raise ValueError('Closest-point interface has reversed normal alignment; inspect the geometry pairing')
        self.audit = {'normal_alignment_min': float(min(alignments)) if alignments else None, 'fluid_surface_area': area, 'fluid_triangles': len(fluid_triangles), 'biot_triangles': len(biot_triangles), 'quadrature_order': order, 'projection_refinement': refinement, 'integration_patches': patch_count, 'unresolved_crossing_area_fraction': unresolved_area / area, 'quadrature_points': len(distances), 'projection_distance_max': float(distances.max()), 'projection_distance_mean': float(distances.mean()), 'trace_dofs': self.N, 'D_nonzeros': D.nnz, 'R_nonzeros': R.nnz, 'coupling': 'common fluid surface; exact closest triangle point; P2/P1 or P2/P2 pulled-back traces; single paired quadrature'}

class Block3D:

    def __init__(self, kind, comm, cfg, dt, mesh_dir=None, fixture_n=None, phi_degree=2):
        self.kind = kind
        self.comm = comm
        self.cfg = cfg
        self.m = material_values(cfg)
        self.phi_degree = phi_degree
        if fixture_n is None:
            kw = {} if mesh_dir is None else {'mesh_dir': mesh_dir}
            self.g = read_configured_geometry(kind, comm, cfg, **kw)
        else:
            self.g = fixture_geometry(kind, comm, fixture_n)
        self.mesh = self.g['mesh']
        cell = self.mesh.ufl_cell()
        vec = VectorElement('CG', cell, 2)
        els = [vec, FiniteElement('CG', cell, 1)] if kind == 'fluid' else [vec, vec, FiniteElement('CG', cell, phi_degree)]
        self.names = ('u', 'p') if kind == 'fluid' else ('eta', 'xi', 'phi')
        self.V = FunctionSpace(self.mesh, MixedElement(els))
        self.new = Function(self.V)
        self.old = Function(self.V)
        self.older = Function(self.V)
        self.bcs, self.inlet = exterior_bcs(kind, self.V, self.g['markers'], 0.0, cfg)
        if fixture_n is not None and kind == 'biot':
            self.bcs = [DirichletBC(self.V.sub(i), Constant((0.0, 0.0, 0.0)), self.g['markers'], 6) for i in (0, 1)]
        if fixture_n is not None and kind == 'fluid':
            self.inlet = Expression(('0', '0', '-A*16*x[0]*(1-x[0])*x[1]*(1-x[1])*(1-cos(2*pi*t))/2'), degree=4, A=0.1, t=0.0, mpi_comm=comm)
            self.bcs = [DirichletBC(self.V.sub(0), self.inlet, self.g['markers'], 1)]
        self.weights = [Constant(v) for v in (3.0, 4.0, 1.0)]
        self.denominator = Constant(2.0)
        self.dt = Constant(dt)
        old = dict(zip(self.names, split(self.old)))
        older = dict(zip(self.names, split(self.older)))
        history = {**old, **{k + '1': v for k, v in older.items()}}
        sources = cfg['initial_fields_and_volume_sources']
        if kind == 'fluid':
            F = fluid_bulk(self.V, self.dt, history, self.m, Constant(tuple(sources['fluid_volume_force'])), self.g['dx'], self.weights, self.denominator)
        else:
            F = biot_bulk(self.V, self.dt, history, self.m, Constant(tuple(sources['biot_volume_force'])), Constant(float(sources['pore_volume_source'])), self.g['dx'], self.weights, self.denominator)
            F -= pore_outward_flux_load(TestFunctions(self.V)[2], self.g['ds'], cfg)
        self.a = lhs(F)
        self.load = rhs(F)
        self.surface, self.normals = self.extract_surface()
        self.trace_map = self.extract_trace_map()

    def set_scheme(self, first):
        for c, v in zip(self.weights, (1.0, 1.0, 0.0) if first else (3.0, 4.0, 1.0)):
            c.assign(v)
        self.denominator.assign(1.0 if first else 2.0)

    def extract_surface(self):
        mesh = self.mesh
        mesh.init(2, 0)
        mesh.init(2, 3)
        tri = []
        normals = []
        for facet in facets(mesh):
            if self.g['markers'][facet] != 5 or not facet.exterior():
                continue
            owners = [int(i) for i in facet.entities(3) if not Cell(mesh, int(i)).is_ghost()]
            if not owners:
                continue
            points = mesh.coordinates()[facet.entities(0)]
            n = np.asarray(facet.normal().array())
            center = Cell(mesh, owners[0]).midpoint().array()
            if np.dot(n, points.mean(axis=0) - center) < 0:
                n = -n
            points = np.asarray(sorted(points, key=key))
            tri.append(points)
            normals.append(n)
        parts = self.comm.allgather((np.asarray(tri).reshape(-1, 3, 3), np.asarray(normals).reshape(-1, 3)))
        triangles = np.concatenate([a for a, b in parts])
        ns = np.concatenate([b for a, b in parts])
        order = sorted(range(len(triangles)), key=lambda i: tuple(sorted((key(x) for x in triangles[i]))))
        return (triangles[order], ns[order])

    def extract_trace_map(self):
        dm = self.V.dofmap()
        lg = dm.tabulate_local_to_global_dofs()
        lo, hi = dm.ownership_range()
        rev = {int(g): j for j, g in enumerate(lg)}
        coords = self.V.tabulate_dof_coordinates().reshape(-1, 3)
        specifications = [('u', 0, 3, 2)] if self.kind == 'fluid' else [('xi', 1, 3, 2), ('phi', 2, 1, self.phi_degree)]
        local = {}
        for name, field, ncomp, degree in specifications:
            nodes = {key(x) for x in triangle_nodes(self.surface, degree).reshape(-1, 3)}
            for c in range(ncomp):
                sub = self.V.sub(field) if ncomp == 1 else self.V.sub(field).sub(c)
                for gid in sub.dofmap().dofs():
                    if lo <= gid < hi:
                        point = key(coords[rev[int(gid)]])
                        if point in nodes:
                            local[name, c, point] = int(gid)
        maps = {}
        for part in self.comm.allgather(local):
            if set(maps) & set(part):
                raise ValueError('Duplicate owned 3D trace DOF')
            maps.update(part)
        return maps

    def rhs(self, t):
        if self.inlet is not None:
            self.inlet.t = t
        return assemble(self.load)

    def shift(self):
        self.older.assign(self.old)
        self.old.assign(self.new)

    def energy_and_volume(self):
        f = dict(zip(self.names, split(self.new)))
        m = self.m
        dx = self.g['dx']
        if self.kind == 'fluid':
            density = m['rho_f'] * inner(f['u'], f['u'])
        else:
            density = m['rho_b'] * inner(f['xi'], f['xi']) + m['C0'] * f['phi'] ** 2 + 2 * m['mu_b'] * inner(epsilon(f['eta']), epsilon(f['eta'])) + m['lambda_b'] * div(f['eta']) ** 2
        return 0.5 * float(assemble(density * dx))

    def snapshot(self, level='new'):
        function = getattr(self, level)
        out = {}
        for name, fn in zip(self.names, function.split(True)):
            components = fn.split(True) if fn.ufl_shape else [fn]
            parts = []
            for c, part in enumerate(components):
                V = part.function_space()
                coords = V.tabulate_dof_coordinates().reshape(-1, 3)
                ids = V.dofmap().tabulate_local_to_global_dofs()
                lo, hi = part.vector().local_range()
                values = part.vector().get_local()
                rows = [[*x, c, values[int(g) - lo]] for x, g in zip(coords, ids) if lo <= g < hi]
                parts.append(np.asarray(rows).reshape(-1, 5))
            gathered = self.comm.gather(np.concatenate(parts), root=0)
            if self.comm.rank == 0:
                rows = np.concatenate(gathered)
                out[name] = rows[np.lexsort((rows[:, 3], rows[:, 2], rows[:, 1], rows[:, 0]))]
        return out

def fixture_geometry(kind, comm, n):
    lo, hi = (0.0, 1.0) if kind == 'fluid' else (-1.0, 0.0)
    mesh = BoxMesh(comm, Point(0, 0, lo), Point(1, 1, hi), n, n, n)
    tags = MeshFunction('size_t', mesh, 2, 0)

    class Boundary(SubDomain):

        def inside(self, x, b):
            return b

    class Plane(SubDomain):

        def __init__(self, z):
            super().__init__()
            self.z = z

        def inside(self, x, b):
            return b and near(x[2], self.z)
    Boundary().mark(tags, 2 if kind == 'biot' else 1)
    Plane(hi if kind == 'fluid' else lo).mark(tags, 1 if kind == 'fluid' else 6)
    if kind == 'fluid':

        class Sides(SubDomain):

            def inside(self, x, b):
                return b and (near(x[0], 0) or near(x[0], 1) or near(x[1], 0) or near(x[1], 1))
        Sides().mark(tags, 2)
    Plane(0.0).mark(tags, 5)
    return {'mesh': mesh, 'markers': tags, 'n': FacetNormal(mesh), 'dx': Measure('dx', domain=mesh), 'ds': Measure('ds', domain=mesh, subdomain_data=tags)}

class Coupling3D:

    def __init__(self, blocks, world, cfg, phi_degree, order, refinement=3):
        self.world = world
        self.blocks = blocks
        rank = world.rank
        bundle = {b.kind: {'surface': b.surface, 'normals': b.normals, 'trace_map': b.trace_map, 'dofs': b.V.dim()} for b in blocks if b.comm.rank == 0}
        pieces = world.gather(bundle, root=0)
        ops = None
        meta = None
        if rank == 0:
            data = {k: v for p in pieces for k, v in p.items()}
            ops = InterfaceOperators(data['fluid']['surface'], data['biot']['surface'], data['fluid']['normals'], cfg['physical_coefficients'], cfg['numerical_robin_L'], phi_degree, order, refinement, data['biot']['normals'])
            native = {}
            for name, kind, ncomp in [('u', 'fluid', 3), ('xi', 'biot', 3), ('phi', 'biot', 1)]:
                native[name] = np.asarray([[data[kind]['trace_map'][name, c, key(x)] for c in range(ncomp)] for x in ops.coordinates[name]], dtype=petsc.IntType)
            ids = np.concatenate([native[k].reshape(-1) for k in ['u', 'xi', 'phi']])
            meta = {'native_ids': ids, 'nf_trace': ops.nf, 'N_trace': ops.N, 'fluid_dofs': data['fluid']['dofs'], 'biot_dofs': data['biot']['dofs'], 'audit': ops.audit}
        self.meta = world.bcast(meta, root=0)
        self.ops = ops
        self.ids = self.meta['native_ids']
        self.nt = self.meta['N_trace']
        self.nft = self.meta['nf_trace']
        self.local = []
        for b in blocks:
            lo, hi = b.old.vector().local_range()
            kindmask = np.arange(self.nt) < self.nft
            if b.kind == 'biot':
                kindmask = ~kindmask
            indices = np.flatnonzero(kindmask & (self.ids >= lo) & (self.ids < hi))
            self.local.append((b, indices, self.ids[indices] - lo))

    def predictor(self, first):
        send = np.zeros(self.nt)
        for block, indices, local in self.local:
            old = block.old.vector().get_local()
            older = block.older.vector().get_local()
            send[indices] = old[local] if first else 2 * old[local] - older[local]
        recv = np.empty_like(send)
        self.world.Allreduce(send, recv, op=mpi.SUM)
        return recv

    def trace_current(self):
        send = np.zeros(self.nt)
        for block, indices, local in self.local:
            send[indices] = block.new.vector().get_local()[local]
        recv = np.empty_like(send)
        self.world.Allreduce(send, recv, op=mpi.SUM)
        return recv

    def distribute(self, kind, ownership, offsets=False):
        descriptions = self.world.gather(ownership, root=0)
        packets = None
        if self.world.rank == 0:
            matrix = getattr(self.ops, kind).tocoo()
            joint = self.ids.copy()
            joint[self.nft:] += self.meta['fluid_dofs']
            packets = []
            for desc in descriptions:
                block, lo, hi = desc
                rows = joint[matrix.row] if offsets else self.ids[matrix.row]
                cols = joint[matrix.col] if offsets else self.ids[matrix.col]
                take = (rows >= lo) & (rows < hi)
                if not offsets:
                    which = matrix.row < self.nft
                    take &= which if block == 'fluid' else ~which
                if kind == 'R':
                    cols = matrix.col
                packets.append((rows[take], cols[take], matrix.data[take]))
        return self.world.scatter(packets, root=0)

def add_coo(mat, packet):
    rows, cols, values = packet
    if len(rows):
        order = np.argsort(rows, kind='stable')
        rows = rows[order]
        cols = cols[order]
        values = values[order]
        unique, starts = np.unique(rows, return_index=True)
        ends = np.r_[starts[1:], len(rows)]
        for row, start, end in zip(unique, starts, ends):
            mat.setValues(int(row), np.asarray(cols[start:end], dtype=petsc.IntType), values[start:end], addv=True)

class NRSystem3D:

    def __init__(self, block, coupling):
        self.block = block
        self.c = coupling
        lo, hi = block.old.vector().local_range()
        self.D = coupling.distribute('D', (block.kind, lo, hi))
        rows, cols, vals = coupling.distribute('R', (block.kind, lo, hi))
        self.R = sp.coo_matrix((vals, (rows - lo, cols)), shape=(hi - lo, coupling.nt)).tocsr()
        self.systems = {}
        self.matrix_setup_s = 0.0

    def prepare(self, first):
        self.block.set_scheme(first)
        if not first and True in self.systems:
            self.systems.pop(True)
            self.A = None
            self.solver = None
        if first not in self.systems:
            start = time.perf_counter()
            base = assemble(self.block.a)
            mat = as_backend_type(base).mat().duplicate(copy=True)
            mat.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
            add_coo(mat, self.D)
            mat.assemble()
            A = PETScMatrix(mat)
            for bc in self.block.bcs:
                bc.apply(A)
            self.systems[first] = (A, LUSolver(A, 'mumps'))
            self.matrix_setup_s += time.perf_counter() - start
        self.A, self.solver = self.systems[first]

    def rhs(self, t, predictor):
        b = self.block.rhs(t)
        values = b.get_local() + self.R @ predictor
        b.set_local(values)
        b.apply('insert')
        for bc in self.block.bcs:
            bc.apply(b)
        return b

    def solve(self, rhs):
        self.solver.solve(self.block.new.vector(), rhs)

    def residual(self, rhs):
        r = self.A * self.block.new.vector()
        r.axpy(-1.0, rhs)
        return r.norm('l2') / max(rhs.norm('l2'), 1e-300)

class MonolithicSystem3D:

    def __init__(self, blocks, coupling):
        self.blocks = blocks
        self.c = coupling
        self.comm = coupling.world
        self.nf = coupling.meta['fluid_dofs']
        self.N = self.nf + coupling.meta['biot_dofs']
        self.mat = petsc.Mat().createAIJ(size=(self.N, self.N), nnz=0, comm=self.comm)
        self.mat.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        self.mat.assemble()
        lo, hi = self.mat.getOwnershipRange()
        self.H = coupling.distribute('H', ('joint', lo, hi), True)
        self.systems = {}
        self.matrix_setup_s = 0.0
        self.b = None
        self.x = None
        self.transfers = []

    def prepare(self, first):
        for block in self.blocks:
            block.set_scheme(first)
        if not first and True in self.systems:
            self.systems.pop(True)
            self.A = None
            self.solver = None
        if first not in self.systems:
            start = time.perf_counter()
            row_lo, row_hi = self.mat.getOwnershipRange()
            local_size = row_hi - row_lo
            ends = np.asarray(self.comm.allgather(row_hi))
            bulk_data = []
            diag_counts = self.mat.createVecLeft()
            off_counts = diag_counts.duplicate()
            diag_counts.set(0.0)
            off_counts.set(0.0)
            for block in self.blocks:
                offset = 0 if block.kind == 'fluid' else self.nf
                bulk = assemble(block.a)
                source = as_backend_type(bulk).mat()
                lo, hi = source.getOwnershipRange()
                ia, ja, values = source.getValuesCSR()
                rows = np.arange(lo + offset, hi + offset, dtype=petsc.IntType)
                cols = ja + offset
                owner = np.searchsorted(ends, rows, side='right')
                col_owner = np.searchsorted(ends, cols, side='right')
                diagonal = col_owner == np.repeat(owner, np.diff(ia))
                cumulative = np.r_[0, np.cumsum(diagonal)]
                dn = np.diff(cumulative[ia])
                on = np.diff(ia) - dn
                diag_counts.setValues(rows, dn, addv=True)
                off_counts.setValues(rows, on, addv=True)
                bulk_data.append((bulk, rows, ia, cols, values))
            diag_counts.assemble()
            off_counts.assemble()
            dn = diag_counts.getArray().copy()
            on = off_counts.getArray().copy()
            hrows, hcols, hvalues = self.H
            local_diagonal = (hcols >= row_lo) & (hcols < row_hi)
            dn += np.bincount((hrows[local_diagonal] - row_lo).astype(int), minlength=local_size)
            on += np.bincount((hrows[~local_diagonal] - row_lo).astype(int), minlength=local_size)
            mat = petsc.Mat().createAIJ(size=((local_size, self.N), (local_size, self.N)), nnz=(dn.astype(petsc.IntType), on.astype(petsc.IntType)), comm=self.comm)
            mat.setOption(petsc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
            for bulk, rows, ia, cols, values in bulk_data:
                for j, row in enumerate(rows):
                    sl = slice(ia[j], ia[j + 1])
                    mat.setValues(int(row), cols[sl], values[sl], addv=True)
            add_coo(mat, self.H)
            mat.assemble()
            constrained = []
            for block in self.blocks:
                offset = 0 if block.kind == 'fluid' else self.nf
                dm = block.V.dofmap()
                lg = dm.tabulate_local_to_global_dofs()
                lo, hi = block.old.vector().local_range()
                for bc in block.bcs:
                    constrained.extend((int(lg[i]) + offset for i in bc.get_boundary_values()))
            all_constraints = {i for part in self.comm.allgather(constrained) for i in part}
            row_lo, row_hi = mat.getOwnershipRange()
            owned = np.asarray(sorted((i for i in all_constraints if row_lo <= i < row_hi)), dtype=petsc.IntType)
            mat.zeroRows(owned, diag=1.0)
            mat.assemble()
            A = PETScMatrix(mat)
            self.systems[first] = (A, LUSolver(A, 'mumps'))
            self.matrix_setup_s += time.perf_counter() - start
        self.A, self.solver = self.systems[first]
        if self.b is None:
            self.b = PETScVector(self.comm)
            self.x = PETScVector(self.comm)
            self.A.init_vector(self.b, 0)
            self.A.init_vector(self.x, 1)
            for block in self.blocks:
                offset = 0 if block.kind == 'fluid' else self.nf
                lo, hi = block.old.vector().local_range()
                indices = np.arange(lo + offset, hi + offset, dtype=petsc.IntType)
                seq = petsc.Vec().createSeq(len(indices), comm=mpi.COMM_SELF)
                src = petsc.IS().createGeneral(indices, comm=self.comm)
                dest = petsc.IS().createStride(len(indices), comm=mpi.COMM_SELF)
                scatter = petsc.Scatter().create(as_backend_type(self.x).vec(), src, seq, dest)
                self.transfers.append((block, indices, seq, scatter))

    def rhs(self, t, predictor=None):
        target = as_backend_type(self.b).vec()
        target.set(0.0)
        for block, indices, _, _ in self.transfers:
            b = block.rhs(t)
            for bc in block.bcs:
                bc.apply(b)
            target.setValues(indices, b.get_local())
        target.assemble()
        return self.b

    def solve(self, rhs):
        self.solver.solve(self.x, rhs)
        for block, _, seq, scatter in self.transfers:
            scatter.scatter(as_backend_type(self.x).vec(), seq, addv=petsc.InsertMode.INSERT, mode=petsc.ScatterMode.FORWARD)
            block.new.vector().set_local(seq.getArray())
            block.new.vector().apply('insert')
            block.new.vector().update_ghost_values()

    def residual(self, rhs):
        r = self.A * self.x
        r.axpy(-1.0, rhs)
        return r.norm('l2') / max(rhs.norm('l2'), 1e-300)
ROOT = Path(__file__).resolve().parent

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

class FieldOutput3D:
    """ParaView time series plus exact native finite-element HDF5 fields."""

    def __init__(self, blocks, output, xdmf_every, checkpoint_every):
        if xdmf_every < 0 or checkpoint_every < 0:
            raise ValueError('Output intervals must be nonnegative')
        self.blocks = blocks
        self.xdmf_every = xdmf_every
        self.checkpoint_every = checkpoint_every
        self.xdmf = {}
        self.hdf5 = {}
        self.times = []
        for b in blocks:
            folder = output / b.kind
            if b.comm.rank == 0:
                folder.mkdir(exist_ok=True)
            b.comm.Barrier()
            if xdmf_every:
                f = XDMFFile(b.comm, str(folder / 'fields.xdmf'))
                f.parameters['functions_share_mesh'] = True
                f.parameters['flush_output'] = True
                f.parameters['rewrite_function_mesh'] = False
                self.xdmf[b.kind] = f
            if checkpoint_every:
                h = HDF5File(b.comm, str(folder / 'fields_checkpoint.h5'), 'w')
                h.write(b.mesh, '/mesh')
                self.hdf5[b.kind] = h

    def write(self, step, t, final=False, level='new'):
        visual = bool(self.xdmf_every and (step == 0 or step % self.xdmf_every == 0 or final))
        exact = bool(self.checkpoint_every and (step == 0 or step % self.checkpoint_every == 0 or final))
        if visual or exact:
            for b in self.blocks:
                fields = getattr(b, level).split(True)
                for name, field in zip(b.names, fields):
                    field.rename(name, name)
                    if visual:
                        self.xdmf[b.kind].write(field, float(t))
                    if exact:
                        path = f'/{name}/step_{step:08d}'
                        self.hdf5[b.kind].write(field, path)
                        attrs = self.hdf5[b.kind].attributes(path)
                        attrs['time'] = float(t)
                        attrs['step'] = int(step)
                if exact:
                    self.hdf5[b.kind].flush()
        self.times.append({'step': step, 't': t, 'paraview_written': visual, 'native_checkpoint_written': exact})

    def close(self):
        for f in self.xdmf.values():
            f.close()
        for f in self.hdf5.values():
            f.close()

def run(cfg, method, output, dt, T, mesh_source='fixture', fixture_n=1, biot_fixture_n=None, phi_degree=2, quadrature_order=4, mesh_dir=None, save_fields=True, xdmf_every=1, projection_refinement=3, checkpoint_every=1, step_observer=None):
    w = mpi.COMM_WORLD
    rank = w.rank
    size = w.size
    if dt <= 0 or T <= 0 or (not np.isfinite(dt + T)):
        raise ValueError('dt and T must be finite positive')
    steps = round(T / dt)
    if steps < 1 or not np.isclose(steps * dt, T, rtol=0, atol=1e-12):
        raise ValueError('dt must divide T')
    if method == 'nr' and size > 1 and size % 2:
        raise ValueError('NR requires an even world size or a serial reference run')
    if phi_degree not in (1, 2):
        raise ValueError('phi degree must be 1 or 2')
    if not has_lu_solver_method('mumps'):
        raise ValueError('MUMPS backend is required explicitly')
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    w.Barrier()
    start = time.perf_counter()
    if method == 'nr' and size > 1:
        kind = 'fluid' if rank < size // 2 else 'biot'
        comm = w.Split(0 if kind == 'fluid' else 1, rank)
        kinds = [kind]
    else:
        comm = w
        kinds = ['fluid', 'biot']
    blocks = []
    for kind in kinds:
        n = biot_fixture_n or fixture_n if kind == 'biot' else fixture_n
        blocks.append(Block3D(kind, comm, cfg, dt, mesh_dir, None if mesh_source == 'cylinder' else n, phi_degree))
    t0 = time.perf_counter()
    coupling = Coupling3D(blocks, w, cfg, phi_degree, quadrature_order, projection_refinement)
    interface_setup = w.allreduce(time.perf_counter() - t0, op=mpi.MAX)
    if rank == 0:
        (output / 'interface_audit.json').write_text(json.dumps(coupling.meta['audit'], indent=2) + '\n')
        print('Interface ready:', json.dumps(coupling.meta['audit']), flush=True)
    systems = [MonolithicSystem3D(blocks, coupling)] if method == 'monolithic' else [NRSystem3D(b, coupling) for b in blocks]

    def snapshots(level, name):
        data = {}
        for block in blocks:
            data.update(block.snapshot(level))
        pieces = w.gather(data, root=0)
        if rank == 0:
            np.savez_compressed(output / name, **{k: v for part in pieces for k, v in part.items()})
    if save_fields:
        snapshots('old', 'initial_fields.npz')
    field_output = FieldOutput3D(blocks, output, xdmf_every, checkpoint_every)
    field_output.write(0, 0.0, level='old')
    coupling.ops = None
    for b in blocks:
        b.surface = None
        b.normals = None
        b.trace_map = None
    setup = w.allreduce(time.perf_counter() - start, op=mpi.MAX)
    records = []
    energies = [{'step': 0, 't': 0.0, 'physical_energy': 0.0, 'bdf2_G_energy': 0.0}]
    if step_observer is not None:
        step_observer(blocks,0,0.)
    integration_start = time.perf_counter()
    for step in range(1, steps + 1):
        step_start = time.perf_counter()
        first = step == 1
        t = step * dt
        t0 = time.perf_counter()
        if rank==0 and step<=2:print('Preparing '+('BE' if first else 'BDF2')+' matrix',flush=True)
        for system in systems:
            system.prepare(first)
        if rank==0 and step<=2:print('Matrix prepared',flush=True)
        matrix_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        predictor = coupling.predictor(first) if method == 'nr' else None
        exchange_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        rhs = [system.rhs(t, predictor) for system in systems]
        rhs_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        if rank==0 and step<=2:print('Starting linear solve',flush=True)
        for system, b in zip(systems, rhs):
            system.solve(b)
        if rank==0 and step<=2:print('Linear solve complete',flush=True)
        solve_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        residual = max((system.residual(b) for system, b in zip(systems, rhs)))
        residual = w.allreduce(residual, op=mpi.MAX)
        if not np.isfinite(residual) or residual > 1e-07:
            raise ArithmeticError(f'Unacceptable linear residual {residual} at step {step}')
        bc_violation = 0.0
        for block in blocks:
            prescribed = block.new.vector().copy()
            for bc in block.bcs:
                bc.apply(prescribed)
            defect = block.new.vector().copy()
            defect.axpy(-1.0, prescribed)
            bc_violation = max(bc_violation, defect.norm('linf'))
        bc_violation = w.allreduce(bc_violation, op=mpi.MAX)
        if bc_violation > 1e-08:
            raise ArithmeticError(f'Dirichlet violation {bc_violation}')
        local = 0.0
        g_energy = 0.0
        for b in blocks:
            energy = b.energy_and_volume()
            f = dict(zip(b.names, split(b.new)))
            old = dict(zip(b.names, split(b.old)))
            m = b.m
            dx = b.g['dx']
            d = {k: 2 * f[k] - old[k] for k in b.names}
            if b.kind == 'fluid':
                density = m['rho_f'] * inner(d['u'], d['u'])
            else:
                density = m['rho_b'] * inner(d['xi'], d['xi']) + m['C0'] * d['phi'] ** 2 + 2 * m['mu_b'] * inner(epsilon(d['eta']), epsilon(d['eta'])) + m['lambda_b'] * div(d['eta']) ** 2
            G = 0.25 * (2 * energy + float(assemble(density * dx)))
            if b.comm.rank == 0:
                local += energy
                g_energy += G
        energy = w.allreduce(local, op=mpi.SUM)
        G = w.allreduce(g_energy, op=mpi.SUM)
        if not np.isfinite(energy + G):
            raise ArithmeticError('Nonfinite energy')
        energies.append({'step': step, 't': t, 'physical_energy': energy, 'bdf2_G_energy': G})
        field_output.write(step, t, final=step == steps)
        if step_observer is not None:
            step_observer(blocks,step,t)
        diagnostic_s = time.perf_counter() - t0
        for block in blocks:
            block.shift()
        row = {'step': step, 't': t, 'scheme': 'BE/EX1' if first and method == 'nr' else 'BE' if first else 'BDF2/EX2' if method == 'nr' else 'BDF2', 'matrix_prepare_s': matrix_s, 'trace_exchange_s': exchange_s, 'rhs_assembly_s': rhs_s, 'linear_solve_s': solve_s, 'diagnostics_s': diagnostic_s, 'step_wall_s': time.perf_counter() - step_start, 'relative_linear_residual': residual, 'maximum_dirichlet_violation': bc_violation}
        for k in ['matrix_prepare_s', 'trace_exchange_s', 'rhs_assembly_s', 'linear_solve_s', 'diagnostics_s', 'step_wall_s']:
            row[k] = w.allreduce(row[k], op=mpi.MAX)
        records.append(row)
        if rank==0:
            write_csv(output/'energy.csv',energies)
            write_csv(output/'step_timings.csv',records)
        if rank == 0:
            print(f'{method}: step {step}/{steps}, t={t:g}, residual={residual:.3g}, energy={energy:.6g}', flush=True)
    integration = w.allreduce(time.perf_counter() - integration_start, op=mpi.MAX)
    if save_fields:
        snapshots('new', 'final_fields.npz')
    field_output.close()
    w.Barrier()
    wall = w.allreduce(time.perf_counter() - start, op=mpi.MAX)
    summary = {'method': method, 'dimension': 3, 'world_ranks': size, 'mesh_source': mesh_source, 'mesh_directory': str((mesh_dir or ROOT / 'FSI_parallel_3D/mesh').resolve()) if mesh_source == 'cylinder' else None, 'fixture_n': fixture_n if mesh_source == 'fixture' else None, 'biot_fixture_n': biot_fixture_n or fixture_n if mesh_source == 'fixture' else None, 'dofs': coupling.meta['fluid_dofs'] + coupling.meta['biot_dofs'], 'fluid_dofs': coupling.meta['fluid_dofs'], 'biot_dofs': coupling.meta['biot_dofs'], 'phi_degree': phi_degree, 'dt': dt, 'T': T, 'steps': steps, 'beta': 0, 'physical_coefficients': cfg['physical_coefficients'], 'L': cfg['numerical_robin_L'], 'mesh_scale': cfg['mesh_coordinate_scale'] if mesh_source == 'cylinder' else 1.0, 'configured_artery_mesh_scale': cfg['mesh_coordinate_scale'], 'linear_solver': 'mumps', 'wall_s': wall, 'setup_s': setup, 'interface_setup_s': interface_setup, 'integration_wall_s': integration, 'allocated_rank_seconds': size * wall, 'maximum_relative_linear_residual': max((r['relative_linear_residual'] for r in records)), 'maximum_dirichlet_violation': max((r['maximum_dirichlet_violation'] for r in records)), 'energy_final': energies[-1]['physical_energy'], 'xdmf_every': xdmf_every, 'checkpoint_every': checkpoint_every, 'field_names': {'fluid': ['u', 'p'], 'biot': ['eta', 'xi', 'phi']}, 'startup': cfg['startup'], 'quadrature_order': quadrature_order, 'projection_refinement': projection_refinement, 'thread_environment': {k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']}, 'timing_note': 'Case wall excludes imports/launch/final CSV serialization; includes interface assembly, startup and fields if enabled. Step breakdown uses rank maxima. NR uses one collective trace predictor exchange before independent group solves.', 'geometry_note': 'Fixture uses adjacent cubes and an explicit small synthetic inlet, not the arterial problem' if mesh_source == 'fixture' else 'Pressure-driven conforming fluid cylinder plus thick Biot annulus; coordinates in cm, no artery scaling'}
    if rank == 0:
        write_csv(output / 'field_output_times.csv', field_output.times)
        write_csv(output / 'step_timings.csv', records)
        write_csv(output / 'energy.csv', energies)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (output / 'protocol.json').write_text(json.dumps({'configuration': cfg, 'summary': summary}, indent=2) + '\n')
    return summary

def cylinder_arrays(radius, thickness, length, angular, fluid_radial, wall_radial, axial):
    if not all((np.isfinite(x) and x > 0 for x in [radius, thickness, length])):
        raise ValueError('Positive finite dimensions required')
    if angular < 6 or min(fluid_radial, wall_radial, axial) < 1:
        raise ValueError('Need >=6 angular divisions and positive radial/axial counts')

    def cross_section(kind):
        coords = []
        tri = []
        if kind == 'fluid':
            coords.append([0.0, 0.0])
            radii = np.linspace(radius / fluid_radial, radius, fluid_radial)
            offset = 1
        else:
            radii = np.linspace(radius, radius + thickness, wall_radial + 1)
            offset = 0
        for r in radii:
            for j in range(angular):
                theta = 2 * np.pi * j / angular
                coords.append([r * np.cos(theta), r * np.sin(theta)])
        if kind == 'fluid':
            for j in range(angular):
                tri.append([0, 1 + j, 1 + (j + 1) % angular])
        for layer in range(len(radii) - 1):
            for j in range(angular):
                k = (j + 1) % angular
                a = offset + layer * angular + j
                b = offset + layer * angular + k
                c = a + angular
                d = b + angular
                tri.extend([[a, b, d], [a, d, c]])
        return (np.asarray(coords), np.asarray(tri, dtype=np.int64))
    result = {}
    for kind in ['fluid', 'biot']:
        xy, tri = cross_section(kind)
        n = len(xy)
        xyz = np.concatenate([np.column_stack((xy, np.full(n, z))) for z in np.linspace(0, length, axial + 1)])
        cells = []
        for level in range(axial):
            for row in tri:
                a, b, c = np.sort(row) + level * n
                A, B, C = np.array([a, b, c]) + n
                cells.extend([[a, b, c, C], [a, b, B, C], [a, A, B, C]])
        cells = np.asarray(cells, dtype=np.int64)
        det = np.linalg.det(xyz[cells[:, 1:]] - xyz[cells[:, :1]])
        if np.any(abs(det) < 1e-18):
            raise ValueError('Degenerate tetrahedron')
        negative = det < 0
        cells[negative, 0], cells[negative, 1] = (cells[negative, 1].copy(), cells[negative, 0].copy())
        faces = np.sort(np.concatenate([cells[:, k] for k in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]), axis=1)
        boundary, count = np.unique(faces, axis=0, return_counts=True)
        boundary = boundary[count == 1]
        points = xyz[boundary]
        z = points[:, :, 2]
        r = np.linalg.norm(points[:, :, :2], axis=2)
        labels = np.full(len(boundary), 2, dtype=np.uint64)
        if kind == 'fluid':
            labels[:] = 5
        labels[np.all(abs(z) < 1e-12, axis=1)] = 1 if kind == 'fluid' else 6
        labels[np.all(abs(z - length) < 1e-12, axis=1)] = 2 if kind == 'fluid' else 3
        if kind == 'biot':
            labels[np.all(abs(r - radius) < 1e-12, axis=1)] = 5
        result[kind] = {'coordinates': xyz, 'tetrahedra': cells, 'boundary_faces': boundary, 'boundary_labels': labels}

    def interface(data):
        return {tuple(sorted((tuple(np.round(p, 12)) for p in tri))) for tri in data['coordinates'][data['boundary_faces'][data['boundary_labels'] == 5]]}
    if interface(result['fluid']) != interface(result['biot']):
        raise ValueError('Nonconforming cylinder interface')
    return result

def write_cylinder_meshes(folder, settings, generator='layered', size_factor=1.0):
    from pathlib import Path
    from fenics import Mesh, MeshEditor, MeshFunction, XDMFFile, facets
    from mpi4py import MPI
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if generator not in ('layered', 'gmsh'):
        raise ValueError('Unknown mesh generator')
    data = cylinder_arrays(**settings) if generator == 'layered' else gmsh_cylinder_arrays(**settings, size_factor=size_factor)
    audit = {}
    for kind, item in data.items():
        mesh = Mesh(mpi.COMM_SELF)
        editor = MeshEditor()
        editor.open(mesh, 'tetrahedron', 3, 3)
        editor.init_vertices(len(item['coordinates']))
        editor.init_cells(len(item['tetrahedra']))
        for i, x in enumerate(item['coordinates']):
            editor.add_vertex(i, x)
        for i, c in enumerate(item['tetrahedra']):
            editor.add_cell(i, c.astype(np.uintp))
        editor.close()
        mesh.init(2, 0)
        lookup = {tuple(row): int(tag) for row, tag in zip(item['boundary_faces'], item['boundary_labels'])}
        tags = MeshFunction('size_t', mesh, 2, 0)
        for facet in facets(mesh):
            tags[facet] = lookup.get(tuple(sorted((int(i) for i in facet.entities(0)))), 0)
        tags.rename('marker', 'marker')
        with XDMFFile(mpi.COMM_SELF, str(folder / f'{kind}.xdmf')) as f:
            f.write(mesh)
        with XDMFFile(mpi.COMM_SELF, str(folder / f'{kind}_boundary.xdmf')) as f:
            f.write(tags)
        audit[kind] = {'vertices': mesh.num_vertices(), 'tetrahedra': mesh.num_cells(), 'boundary_counts': {str(int(k)): int(np.sum(item['boundary_labels'] == k)) for k in np.unique(item['boundary_labels'])}, 'generator': generator, **item.get('generator_audit', {})}
    return audit

def gmsh_cylinder_arrays(radius, thickness, length, angular, fluid_radial, wall_radial, axial, size_factor=1.0):
    """One CAD fragmentation/3D mesh, then material extraction.

    Unstructured tetrahedra are generated in a sizing coordinate system and
    mapped radially/axially to physical cm. This is not a layered extrusion.
    Shared interface node IDs/faces are retained through material extraction.
    """
    import gmsh
    if min(radius, thickness, length, size_factor) <= 0 or not np.isfinite([radius, thickness, length, size_factor]).all():
        raise ValueError('Positive finite dimensions and size factor required')
    if angular < 6 or min(fluid_radial, wall_radial, axial) < 1:
        raise ValueError('Invalid target resolution')
    hr = radius / fluid_radial
    hw = thickness / wall_radial
    hz = length / axial
    ht = 2 * np.pi * radius / angular
    h0 = min(hr, ht, 2 * hw)
    computational_thickness = wall_radial * h0
    computational_length = axial * h0
    owned = not gmsh.isInitialized()
    if owned:
        gmsh.initialize()
    model = 'stokes_biot_conforming_cylinder'
    try:
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add(model)
        inner = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, computational_length, radius)
        outer = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, computational_length, radius + computational_thickness)
        fragments, mapping = gmsh.model.occ.fragment([(3, outer)], [(3, inner)])
        gmsh.model.occ.synchronize()
        fluid = {tag for dim, tag in mapping[1] if dim == 3}
        allvol = {tag for dim, tag in mapping[0] if dim == 3}
        wall = allvol - fluid
        if len(fluid) != 1 or len(wall) != 1:
            raise ValueError('Expected one fluid and one annular CAD volume')
        volumes = {'fluid': next(iter(fluid)), 'biot': next(iter(wall))}
        surfaces = {kind: {abs(tag) for dim, tag in gmsh.model.getBoundary([(3, v)], oriented=False) if dim == 2} for kind, v in volumes.items()}
        shared = surfaces['fluid'] & surfaces['biot']
        if not shared:
            raise ValueError('CAD volumes have no shared interface; refusing separate nonconforming meshing')
        labels = {}
        for kind, tags in surfaces.items():
            labels[kind] = {}
            for tag in tags:
                box = gmsh.model.getBoundingBox(2, tag)
                if tag in shared:
                    label = 5
                elif abs(box[2]) < 1e-06 and abs(box[5]) < 1e-06:
                    label = 1 if kind == 'fluid' else 6
                elif abs(box[2] - computational_length) < 1e-06 and abs(box[5] - computational_length) < 1e-06:
                    label = 2 if kind == 'fluid' else 3
                elif kind == 'biot':
                    label = 2
                else:
                    raise ValueError('Unclassified fluid CAD surface')
                labels[kind][tag] = label
        for kind, volume in volumes.items():
            gmsh.model.addPhysicalGroup(3, [volume], 1 if kind == 'fluid' else 2)
        grouped = {}
        for tagmap in labels.values():
            for tag, label in tagmap.items():
                grouped.setdefault(label, set()).add(tag)
        for label, tags in grouped.items():
            gmsh.model.addPhysicalGroup(2, sorted(tags), label)
        gmsh.option.setNumber('Mesh.MeshSizeMin', h0 * size_factor)
        gmsh.option.setNumber('Mesh.MeshSizeMax', h0 * size_factor)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.option.setNumber('Mesh.Algorithm3D', 1)
        gmsh.option.setNumber('Mesh.RandomSeed', 2026)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize('')
        node_tags, flat, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(node_tags)
        node_tags = np.asarray(node_tags)[order]
        computational = np.asarray(flat).reshape(-1, 3)[order]
        physical = computational.copy()
        rho = np.linalg.norm(computational[:, :2], axis=1)
        outside = rho > radius
        mapped_radius = radius + (rho[outside] - radius) * thickness / computational_thickness
        physical[outside, :2] *= (mapped_radius / rho[outside])[:, None]
        physical[:, 2] *= length / computational_length

        def elements(dim, tag, expected_type):
            types, ids, nodes = gmsh.model.mesh.getElements(dim, tag)
            rows = []
            for typ, values in zip(types, nodes):
                if typ != expected_type:
                    raise ValueError('Expected first-order tetrahedra/triangles from Gmsh')
                n = 4 if dim == 3 else 3
                values = np.asarray(values).reshape(-1, n)
                loc = np.searchsorted(node_tags, values)
                if not np.array_equal(node_tags[loc], values):
                    raise ValueError('Missing Gmsh node')
                rows.append(loc)
            return np.concatenate(rows)
        result = {}
        for kind, volume in volumes.items():
            global_tets = elements(3, volume, 4)
            boundary = []
            tagvalues = []
            for surface, label in labels[kind].items():
                tri = elements(2, surface, 2)
                boundary.append(tri)
                tagvalues.extend([label] * len(tri))
            global_faces = np.concatenate(boundary)
            used = np.unique(global_tets)
            tets = np.searchsorted(used, global_tets)
            faces = np.searchsorted(used, global_faces)
            xyz = physical[used]
            before = computational[global_tets]
            after = physical[global_tets]
            det0 = np.linalg.det(before[:, 1:] - before[:, :1])
            det1 = np.linalg.det(after[:, 1:] - after[:, :1])
            if np.any(det0 * det1 <= 0):
                raise ValueError('Sizing-to-physical map inverted or degenerated a tetrahedron; reduce --gmsh-size-factor')
            negative = det1 < 0
            tets[negative, 0], tets[negative, 1] = (tets[negative, 1].copy(), tets[negative, 0].copy())
            lengths = np.concatenate([np.linalg.norm(xyz[tets[:, i]] - xyz[tets[:, j]], axis=1) for i, j in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]])
            result[kind] = {'coordinates': xyz, 'tetrahedra': tets, 'boundary_faces': np.sort(faces, axis=1), 'boundary_labels': np.asarray(tagvalues, dtype=np.uint64), 'generator_audit': {'generator': 'Gmsh 3D Delaunay, one fragmented CAD model', 'gmsh_version': gmsh.__version__, 'conforming_CAD_surface_tags': sorted(shared), 'nominal_target_cm': {'fluid_radial': hr, 'wall_radial': hw, 'circumferential': ht, 'axial': hz}, 'computational_mesh_size': h0 * size_factor, 'radial_wall_map_scale': thickness / computational_thickness, 'axial_map_scale': length / computational_length, 'minimum_tetrahedron_volume_cm3': float(abs(det1).min() / 6), 'total_volume_cm3': float(abs(det1).sum() / 6), 'minimum_map_determinant_ratio': float((det1 / det0).min()), 'edge_length_cm_quantiles': np.quantile(lengths, [0, 0.5, 0.95, 1]).tolist()}}

        def boundary_keys(item):
            return {tuple(sorted((tuple(np.round(x, 12)) for x in points))) for points in item['coordinates'][item['boundary_faces'][item['boundary_labels'] == 5]]}
        a = boundary_keys(result['fluid'])
        b = boundary_keys(result['biot'])
        if a != b:
            raise ValueError('Extracted fluid/Biot interface triangles differ')
        for item in result.values():
            item['generator_audit']['interface_triangles_identical'] = True
            item['generator_audit']['interface_triangle_count'] = len(a)
        return result
    finally:
        if owned:
            gmsh.finalize()
        else:
            gmsh.model.remove()

def cylinder_read_geometry(kind, comm, cfg, mesh_dir=None):
    if mesh_dir is None:
        raise ValueError('Cylinder mesh directory is required')
    mesh = Mesh(comm)
    with XDMFFile(comm, str(Path(mesh_dir) / f'{kind}.xdmf')) as f:
        f.read(mesh)
    mvc = MeshValueCollection('size_t', mesh, 2)
    with XDMFFile(comm, str(Path(mesh_dir) / f'{kind}_boundary.xdmf')) as f:
        f.read(mvc, 'marker')
    markers = MeshFunction('size_t', mesh, mvc)
    return {'mesh': mesh, 'markers': markers, 'n': FacetNormal(mesh), 'dx': Measure('dx', domain=mesh), 'ds': Measure('ds', domain=mesh, subdomain_data=markers)}

def cylinder_validate(cfg):
    for spec in cfg['pore'].values():
        if spec['type'] != 'flux' or float(spec['value']) != 0:
            raise ValueError('This cylinder example uses impermeable pore boundaries')
    return cfg

def cylinder_bcs(kind, V, markers, time, cfg):
    cylinder_validate(cfg)
    if kind == 'fluid':
        return ([], None)
    zero = Constant((0.0, 0.0, 0.0))
    return ([DirichletBC(V.sub(i), zero, markers, tag) for i in [0, 1] for tag in [3, 6]], None)

def pulse_pressure(t, amplitude, duration):
    return float(amplitude * np.sin(np.pi * t / duration) ** 2) if 0 < t < duration else 0.0

class CylinderBlock(Block3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pressure_load = Constant(0.0)
        if self.kind == 'fluid':
            v, q = TestFunctions(self.V)
            self.load += -self.pressure_load * dot(v, self.g['n']) * self.g['ds'](1)

    def rhs(self, t):
        if self.kind == 'fluid':
            pulse = self.cfg['pressure_pulse']
            self.pressure_load.assign(pulse_pressure(t, pulse['amplitude'], pulse['duration']))
        return assemble(self.load)

class WaveProbes:

    def __init__(self, cfg, output, world):
        self.cfg = cfg
        self.output = output
        self.world = world
        self.rows = []
        self.stations = np.asarray(cfg['cylinder']['length']) * np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def __call__(self, blocks, step, t):
        values = np.zeros((len(self.stations), 3))
        counts = np.zeros_like(values)
        shape = self.cfg['cylinder']
        theta = np.pi / shape['angular']
        normal = np.array([np.cos(theta), np.sin(theta), 0.0])
        radius = (shape['radius'] + shape['thickness'] / 2) * np.cos(theta)
        for b in blocks:
            fields = dict(zip(b.names, b.new.split(True)))
            tree = b.mesh.bounding_box_tree()
            for i, z in enumerate(self.stations):
                point = Point(0.0, 0.0, z) if b.kind == 'fluid' else Point(radius * normal[0], radius * normal[1], z)
                if tree.compute_first_entity_collision(point) >= b.mesh.num_cells():
                    continue
                if b.kind == 'fluid':
                    values[i, 0] = float(fields['p'](point))
                    values[i, 1] = float(fields['u'](point)[2])
                    counts[i, :2] = 1
                else:
                    values[i, 2] = float(np.dot(fields['eta'](point), normal))
                    counts[i, 2] = 1
        totals = np.empty_like(values)
        number = np.empty_like(counts)
        self.world.Allreduce(values, totals, op=mpi.SUM)
        self.world.Allreduce(counts, number, op=mpi.SUM)
        if np.any(number == 0):
            raise ValueError('A wave probe is outside the distributed cylinder mesh')
        totals /= number
        if self.world.rank == 0:
            for i, z in enumerate(self.stations):
                self.rows.append({'step': step, 't': t, 'z_cm': z, 'pressure_dyn_cm2': totals[i, 0], 'axial_velocity_cm_s': totals[i, 1], 'wall_radial_displacement_cm': totals[i, 2], 'inlet_pressure_dyn_cm2': pulse_pressure(t, self.cfg['pressure_pulse']['amplitude'], self.cfg['pressure_pulse']['duration'])})
            write_csv(self.output / 'wave_probes.csv', self.rows)

def cylinder_configuration(radius, thickness, length, angular, fluid_radial, wall_radial, axial, amplitude, duration):
    cfg = copy.deepcopy(DEFAULT_CONFIGURATION)
    cfg['mesh_coordinate_scale'] = 1.0
    cfg['cylinder'] = {'radius': radius, 'thickness': thickness, 'length': length, 'angular': angular, 'fluid_radial': fluid_radial, 'wall_radial': wall_radial, 'axial': axial}
    cfg['pressure_pulse'] = {'type': 'finite sine-squared normal traction', 'amplitude': amplitude, 'duration': duration, 'unit': 'dyn/cm^2', 'time_unit': 's'}
    cfg['fluid']['1'] = {'type': 'pressure_traction', 'formula': 'sigma_f n = -P_in(t) n'}
    cfg['fluid']['2'] = {'type': 'zero_outlet_total_traction'}
    cfg['biot_mechanics'] = {'2': {'type': 'zero_outer_total_traction'}, '3': {'type': 'clamped_outlet_ring'}, '6': {'type': 'clamped_inlet_ring'}, '5': {'type': 'coupled_interface'}}
    cfg['pore'] = {str(tag): {'type': 'flux', 'value': 0.0} for tag in [2, 3, 6]}
    cfg.pop('inlet_profile', None)
    cfg['initial_fields_and_volume_sources'] = {'initial_fields': 'zero at t=0', 'fluid_volume_force': [0.0, 0.0, 0.0], 'biot_volume_force': [0.0, 0.0, 0.0], 'pore_volume_source': 0.0}
    return cfg
read_configured_geometry = cylinder_read_geometry
exterior_bcs = cylinder_bcs
validate_boundary_config = cylinder_validate
Block3D = CylinderBlock

def cylinder_main():
    p = argparse.ArgumentParser(description='3D pressure pulse in a fluid cylinder with a finite-thickness Biot wall (CGS).')
    p.add_argument('--method', choices=['nr', 'monolithic'], default='monolithic')
    p.add_argument('--radius', type=float, default=0.3)
    p.add_argument('--thickness', type=float, default=0.06)
    p.add_argument('--length', type=float, default=6.0)
    p.add_argument('--angular', type=int, default=24, help='Circumferential target; Gmsh does not impose rings')
    p.add_argument('--fluid-radial', type=int, default=3, help='Fluid radial target: radius/N')
    p.add_argument('--wall-radial', type=int, default=3, help='Wall target: thickness/N, not forced unstructured layers')
    p.add_argument('--axial', type=int, default=24, help='Axial target: length/N; no extruded planes in Gmsh mode')
    p.add_argument('--mesh-generator', choices=['gmsh', 'layered'], default='gmsh')
    p.add_argument('--gmsh-size-factor', type=float, default=1.0)
    p.add_argument('--pressure-amplitude', type=float, default=1000.0)
    p.add_argument('--pulse-duration', type=float, default=0.002)
    p.add_argument('--dt', type=float, default=None, help='Monolithic default 1e-4 s; NR requires an explicit tested choice')
    p.add_argument('--T', type=float, default=None, help='Monolithic default 0.05 s; specify explicitly for NR')
    p.add_argument('--xdmf-every', type=int, default=1)
    p.add_argument('--checkpoint-every', type=int, default=1)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--mesh-only', action='store_true')
    args = p.parse_args()
    set_log_level(LogLevel.WARNING)
    world = mpi.COMM_WORLD
    if args.output_dir is None:
        args.output_dir = Path('results') / ('cylinder_wave_' + args.method)
    if args.method == 'nr' and (args.dt is None or args.T is None):
        p.error('NR needs explicit --dt and --T: dt=1e-4 was unstable; dt=2.5e-5 was bounded only through T=0.006 in the tested mesh')
    if args.dt is None:
        args.dt = 0.0001
    if args.T is None:
        args.T = 0.05
    try:
        if args.pressure_amplitude <= 0 or args.pulse_duration <= 0:
            raise ValueError('Positive pulse amplitude/duration required')
        cfg = cylinder_configuration(args.radius, args.thickness, args.length, args.angular, args.fluid_radial, args.wall_radial, args.axial, args.pressure_amplitude, args.pulse_duration)
        cfg['mesh_generator'] = args.mesh_generator
        cfg['gmsh_size_factor'] = args.gmsh_size_factor
        meshdir = args.output_dir / 'mesh'
        if world.rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            audit = write_cylinder_meshes(meshdir, cfg['cylinder'], args.mesh_generator, args.gmsh_size_factor)
            (args.output_dir / 'mesh_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
            (args.output_dir / 'cylinder_configuration.json').write_text(json.dumps(cfg, indent=2) + '\n')
            print('Cylinder mesh:', json.dumps(audit), flush=True)
        world.Barrier()
        if args.mesh_only:
            return
        observer = WaveProbes(cfg, args.output_dir, world)
        run(cfg, args.method, args.output_dir, args.dt, args.T, mesh_source='cylinder', phi_degree=2, mesh_dir=meshdir, save_fields=False, xdmf_every=args.xdmf_every, checkpoint_every=args.checkpoint_every, step_observer=observer)
    except Exception:
        traceback.print_exc()
        if world.size > 1:
            world.Abort(1)
        raise

if __name__ == "__main__":
    cylinder_main()
