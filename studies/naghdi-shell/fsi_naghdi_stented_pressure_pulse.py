"""Matched fluid--heterogeneous Naghdi shell pressure-pulse experiment.

One self-contained solver supports --scheme monolithic and --scheme explicit.
The reference surface and fitted wall/steel tags are read from --mesh-dir.
Wall and outer-steel layers use separate moduli and through-thickness moments. Both schemes use the same
shell operators, pressure stabilization, ALE extension, and lagged skew fluid
convection. A common reference-interface mass pairing and transpose transfer
preserve interface work. The shell reference is the supplied inward radial
surface, not the separate outward visualization surface.

The artery is present everywhere. An outer steel layer is added in the strut
bands through laminate membrane, bending and coupling moments. Perfect bonding
is assumed; contact, slip, and solid strut sidewalls are not resolved. Densities and Poisson ratio retain the earlier benchmark values.
Output is decimated with --output-every; all diagnostics and six profile times
are retained. Legacy FEniCS, NumPy, SciPy and petsc4py are required.
"""
from dolfin import *
import argparse
import csv
import math
import os
import numpy as np
from ufl import transpose
from pathlib import Path
import json,hashlib,sys
from scipy import sparse


##############################function definitions ################

def inlet_pressure_value(time, maximum_pressure, pulse_duration):
    """Raised-cosine pressure pulse used in the benchmark."""
    if -DOLFIN_EPS <= time <= pulse_duration + DOLFIN_EPS:
        return 0.5 * maximum_pressure * (
            1.0 - math.cos(2.0 * math.pi * time / pulse_duration)
        )
    return 0.0


def copy_matching_function(source, target):
    """Copy coefficients between matching spaces on identical mesh topology.

    The reference and current fluid meshes are loaded independently from the
    same XML mesh.  Their P1 vector spaces therefore have the same global and
    local degree-of-freedom ordering.  Keeping this copy explicit prevents the
    ALE transport from being confused with interpolation in physical space.
    """
    if source.function_space().dim() != target.function_space().dim():
        raise RuntimeError("Cannot copy between nonmatching function spaces.")
    source_values = source.vector().get_local()
    if source_values.size != target.vector().local_size():
        raise RuntimeError("Reference/current local degree-of-freedom layouts differ.")
    target.vector().set_local(source_values)
    target.vector().apply("insert")


class StructureVectorOnReferenceFluid(UserExpression):
    """Extend a vector field on (x,y_arc) by constant radial projection.

    Only the trace on the cylindrical wall is used by the coupling terms.  The
    same expression is also convenient as boundary data for the harmonic ALE
    extension on the fixed reference fluid mesh.
    """

    def __init__(self, structure_field, radius, **kwargs):
        super().__init__(**kwargs)
        self.structure_field = structure_field
        self.structure_field.set_allow_extrapolation(True)
        self.radius = radius
        self.structure_point = np.zeros(2)
        self.value_holder = np.zeros(3)

    def eval(self, values, x):
        theta = math.atan2(x[2], x[1])
        if theta < 0.0:
            theta += 2.0 * math.pi
        self.structure_point[:] = (x[0], self.radius * theta)
        self.structure_field.eval(self.value_holder, self.structure_point)
        values[:] = self.value_holder

    def value_shape(self):
        return (3,)


class StructureScalarOnReferenceFluid(UserExpression):
    """Extend a scalar shell field to the reference fluid mesh."""

    def __init__(self, structure_field, radius, **kwargs):
        super().__init__(**kwargs)
        self.structure_field = structure_field
        self.structure_field.set_allow_extrapolation(True)
        self.radius = radius
        self.structure_point = np.zeros(2)
        self.value_holder = np.zeros(1)

    def eval(self, values, x):
        theta = math.atan2(x[2], x[1])
        if theta < 0.0:
            theta += 2.0 * math.pi
        self.structure_point[:] = (x[0], self.radius * theta)
        self.structure_field.eval(self.value_holder, self.structure_point)
        values[0] = self.value_holder[0]

    def value_shape(self):
        return ()


class CurrentFluidVectorOnShell(UserExpression):
    """Pull a current-domain fluid vector back with Phi_s = phi + eta."""

    def __init__(self, fluid_field, shell_state, radius, **kwargs):
        super().__init__(**kwargs)
        self.fluid_field = fluid_field
        self.shell_state = shell_state
        self.fluid_field.set_allow_extrapolation(True)
        self.shell_state.set_allow_extrapolation(True)
        self.radius = radius
        self.fluid_value = np.zeros(3)
        self.shell_value = np.zeros(6)
        self.shell_point = np.zeros(2)

    def eval(self, values, x):
        x_coord = x[0]
        y_arc = x[1]
        theta = y_arc / self.radius
        self.shell_point[:] = (x_coord, y_arc)
        self.shell_state.eval(self.shell_value, self.shell_point)
        current_point = np.array(
            [
                x_coord + self.shell_value[0],
                self.radius * math.cos(theta) + self.shell_value[1],
                self.radius * math.sin(theta) + self.shell_value[2],
            ]
        )
        self.fluid_field.eval(self.fluid_value, current_point)
        values[:] = self.fluid_value

    def value_shape(self):
        return (3,)

from dolfin import as_matrix
# I am going to define the following two tensor products

def kron_2223_ufl(A, B):
    """
    A: 2×2, B: 2×3 → returns a 4×6 UFL Matrix
    """
    return as_matrix([
        # row 0 = [ A00*B[0,0], A00*B[0,1], A00*B[0,2],
        #           A01*B[0,0], A01*B[0,1], A01*B[0,2] ]
        [A[0,0]*B[0,0], A[0,0]*B[0,1], A[0,0]*B[0,2],
         A[0,1]*B[0,0], A[0,1]*B[0,1], A[0,1]*B[0,2]],
        # row 1 = [ A00*B[1,:], A01*B[1,:] ]
        [A[0,0]*B[1,0], A[0,0]*B[1,1], A[0,0]*B[1,2],
         A[0,1]*B[1,0], A[0,1]*B[1,1], A[0,1]*B[1,2]],
        # row 2 = [ A10*B[0,:], A11*B[0,:] ]
        [A[1,0]*B[0,0], A[1,0]*B[0,1], A[1,0]*B[0,2],
         A[1,1]*B[0,0], A[1,1]*B[0,1], A[1,1]*B[0,2]],
        # row 3 = [ A10*B[1,:], A11*B[1,:] ]
        [A[1,0]*B[1,0], A[1,0]*B[1,1], A[1,0]*B[1,2],
         A[1,1]*B[1,0], A[1,1]*B[1,1], A[1,1]*B[1,2]],
    ])


def kron_2233_ufl(A, B):
    """
    A: 2×2, B: 3×3 → returns a 6×6 UFL Matrix
    """
    rows = []
    for i in range(2):
        for j in range(3):
            # build one 6-entry row: [ A[i,0]*B[j,0:3], A[i,1]*B[j,0:3] ]
            rows.append([
                A[i,0]*B[j,0], A[i,0]*B[j,1], A[i,0]*B[j,2],
                A[i,1]*B[j,0], A[i,1]*B[j,1], A[i,1]*B[j,2],
            ])
    return as_matrix(rows)


# Define the membrane strain (9-component) based on displacements and rotations.
# This is corresponding to [dx(u1),dx(u2),dx(u3),dy(u1),dy(u2),dy(u3),om1,om2,om3] in Dr. Josip's code
def membrane_strain(u):
    displacement = as_vector([u[0], u[1], u[2]])
    rotation = as_vector([u[3], u[4], u[5]])
    disp_dx = as_vector([displacement[0].dx(0), displacement[1].dx(0), displacement[2].dx(0)])
    disp_dy = as_vector([displacement[0].dx(1), displacement[1].dx(1), displacement[2].dx(1)])
    return as_vector([disp_dx[0],
                      disp_dx[1],
                      disp_dx[2],
                      disp_dy[0],
                      disp_dy[1],
                      disp_dy[2],
                      rotation[0],
                      rotation[1],
                      rotation[2]])

# Define the bending strain (6-component)
def bending_strain(u):
    rotation = as_vector([u[3], u[4], u[5]])
    rot_dx = as_vector([rotation[0].dx(0), rotation[1].dx(0), rotation[2].dx(0)])
    rot_dy = as_vector([rotation[0].dx(1), rotation[1].dx(1), rotation[2].dx(1)])
    return as_vector([rot_dx[0],
                      rot_dx[1],
                      rot_dx[2],
                      rot_dy[0],
                      rot_dy[1],
                      rot_dy[2]])

def mass_disp(u):
    return as_vector([u[0], u[1], u[2]])

def mass_rotation(u):
    return as_vector([u[3], u[4], u[5]])


###################################### Parameters ######################################################

parser = argparse.ArgumentParser(
    description="Three-dimensional fluid--Naghdi-shell pressure-pulse benchmark."
)
parser.add_argument(
    "--mesh-dir",
    default=".",
    help="Directory containing fluid.xml and stent.xml (or stent.xdmf/stent.h5).",
)
parser.add_argument("--output-dir", default="stented_onyx_study/results_monolithic", help="Directory for PVD and CSV output.")
parser.add_argument("--dt", type=float, default=1.0e-4, help="Time step [s].")
parser.add_argument("--final-time", type=float, default=2.4e-2, help="Final time [s].")
parser.add_argument('--max-iterations', type=int, default=30)
parser.add_argument('--relative-tolerance', type=float, default=1e-7)
parser.add_argument('--absolute-tolerance', type=float, default=1e-10)
parser.add_argument('--relaxation', type=float, default=1.0)
parser.add_argument('--linear-solver', default='mumps')
parser.add_argument('--output-every', type=int, default=20)
parser.add_argument('--scheme', choices=['monolithic','explicit'], default='monolithic')
parser.add_argument('--robin-parameter', type=float, default=2000.)
parser.add_argument('--wall-young-modulus', type=float, default=4e5)
parser.add_argument('--stent-young-modulus', type=float, default=2.43e10)
parser.add_argument('--wall-thickness', type=float, default=4e-4)
parser.add_argument('--stent-thickness', type=float, default=1.5e-4)
parser.add_argument('--quadrature-degree', type=int, default=2)
parser.add_argument('--maximum-inlet-pressure', type=float, default=2000.0)
parser.add_argument('--pulse-duration', type=float, default=0.005)
parser.add_argument('--fixed-domain', action='store_true', help='Keep the reference fluid domain fixed for verification.')
parser.add_argument("--profile-bins", type=int, default=61, help="Number of axial bins used for profiles.")
parser.add_argument(
    "--pressure-stabilization",
    type=float,
    default=1.0e-2,
    help="Brezzi--Pitkaranta coefficient for the equal-order P1--P1 fluid pair.",
)
args = parser.parse_args()

if args.dt <= 0.0 or args.final_time <= 0.0:
    raise ValueError("The time step and final time must be positive.")
if not (0 < args.relaxation <= 1) or args.max_iterations < 2:
    raise ValueError('Require 0 < relaxation <= 1 and max-iterations >= 2.')
if args.relative_tolerance <= 0 or args.absolute_tolerance <= 0 or args.output_every < 1:
    raise ValueError('Tolerances and output-every must be positive.')
if args.pressure_stabilization <= 0:
    raise ValueError('The P1/P1 pressure stabilization must be positive.')
if args.profile_bins < 2:
    raise ValueError("At least two axial profile bins are required.")

t0 = 0.0
T = args.final_time
dt = args.dt
num_steps = int(round(T / dt))
if not math.isclose(num_steps * dt, T, rel_tol=1.0e-12, abs_tol=1.0e-14):
    raise ValueError("final-time must be an integer multiple of dt.")

# Buka\v{c} pressure-pulse benchmark in SI units.
rho_f = 1.0e3
mu_f = 3.5e-3
rho_s = 1.1e3  # Retained common effective shell density; not a fitted steel-density value.
E = 7.5e4
nu = 0.5
thickness = args.wall_thickness
Radius = 1.5e-3
width = .033
length = 2.0 * math.pi * Radius
maximum_inlet_pressure = args.maximum_inlet_pressure
pulse_duration = args.pulse_duration
if not math.isfinite(maximum_inlet_pressure) or maximum_inlet_pressure < 0 or pulse_duration <= 0:
    raise ValueError("Require finite nonnegative pulse amplitude and positive pulse duration.")


# The reduced plane-stress coefficients remain finite at nu=0.5.  Do not form
# the singular three-dimensional Lame coefficient E*nu/((1+nu)*(1-2*nu)).
mu_shell = E / (2.0 * (1.0 + nu))
coef_trace = E * nu / (1.0 - nu**2)
coef_shear = 2.0 * mu_shell
bending_shear_coefficient = mu_shell

mesh_path = os.path.join(args.mesh_dir, "fluid.xml")
shell_xml_path = os.path.join(args.mesh_dir, "stent.xml")
shell_xdmf_path = os.path.join(args.mesh_dir, "stent.xdmf")
if not os.path.exists(mesh_path):
    raise FileNotFoundError("Missing fluid mesh: {}".format(mesh_path))
if not os.path.exists(shell_xml_path) and not os.path.exists(shell_xdmf_path):
    raise FileNotFoundError(
        "Missing shell mesh: expected {} or {}".format(
            shell_xml_path, shell_xdmf_path
        )
    )
os.makedirs(args.output_dir, exist_ok=True)

########################################### Fluid and ALE initialization #########################################

# Load two identical copies.  fluid_reference_mesh never moves and is used for
# harmonic ALE extension.  fluid_mesh is moved incrementally and carries the
# fluid degrees of freedom from Omega_f^n to Omega_f^{n+1}.
fluid_reference_mesh = Mesh(mesh_path)
fluid_mesh = Mesh(mesh_path)
dim = fluid_mesh.topology().dim()
if dim != 3:
    raise RuntimeError("The benchmark requires a three-dimensional fluid mesh.")
if MPI.size(MPI.comm_world) != 1:
    raise RuntimeError(
        "This reference implementation currently supports serial execution only."
    )
if not np.array_equal(fluid_reference_mesh.cells(), fluid_mesh.cells()):
    raise RuntimeError("The reference and current fluid-mesh connectivities differ.")
if not np.allclose(
    fluid_reference_mesh.coordinates(),
    fluid_mesh.coordinates(),
    rtol=0.0,
    atol=0.0,
):
    raise RuntimeError("The reference and current fluid-mesh coordinates differ.")


def signed_tetrahedron_jacobians(mesh):
    """Return six times each oriented tetrahedron volume."""
    coordinates = mesh.coordinates()
    connectivity = mesh.cells()
    if connectivity.ndim != 2 or connectivity.shape[1] != 4:
        raise RuntimeError("The fluid mesh must contain first-order tetrahedra.")
    x0 = coordinates[connectivity[:, 0]]
    x1 = coordinates[connectivity[:, 1]]
    x2 = coordinates[connectivity[:, 2]]
    x3 = coordinates[connectivity[:, 3]]
    return np.einsum(
        "ij,ij->i",
        x1 - x0,
        np.cross(x2 - x0, x3 - x0),
    )


reference_tetrahedron_jacobians = signed_tetrahedron_jacobians(
    fluid_reference_mesh
)
if np.any(np.abs(reference_tetrahedron_jacobians) <= DOLFIN_EPS):
    raise RuntimeError("The reference fluid mesh contains a degenerate tetrahedron.")

fluid_x_min = float(fluid_reference_mesh.coordinates()[:, 0].min())
fluid_x_max = float(fluid_reference_mesh.coordinates()[:, 0].max())
mesh_radius = float(
    np.sqrt(
        fluid_reference_mesh.coordinates()[:, 1] ** 2
        + fluid_reference_mesh.coordinates()[:, 2] ** 2
    ).max()
)
geometry_tolerance = 1.0e-8
if not math.isclose(fluid_x_min, -.008, abs_tol=geometry_tolerance):
    raise RuntimeError("The benchmark fluid inlet must be located at x=0.")
if not math.isclose(fluid_x_max-fluid_x_min, width, rel_tol=1.0e-8, abs_tol=geometry_tolerance):
    raise RuntimeError("The fluid mesh length does not equal L=0.06 m.")
if not math.isclose(mesh_radius, Radius, rel_tol=1.0e-6, abs_tol=geometry_tolerance):
    raise RuntimeError("The fluid mesh radius does not equal R_0=0.005 m.")

print("Fluid axial interval: [{:.6e}, {:.6e}] m".format(fluid_x_min, fluid_x_max))
print("Fluid reference radius: {:.6e} m".format(mesh_radius))


class Inlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], fluid_x_min, geometry_tolerance)


class Outlet(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], fluid_x_max, geometry_tolerance)


class Wall(SubDomain):
    def inside(self, x, on_boundary):
        return (
            on_boundary
            and not near(x[0], fluid_x_min, geometry_tolerance)
            and not near(x[0], fluid_x_max, geometry_tolerance)
        )


def mark_fluid_boundaries(mesh):
    # Classify exterior facets by their midpoint. A wall triangle touching an
    # end ring is still a wall facet: testing all its vertices would omit it.
    mesh.init(mesh.topology().dim() - 1, mesh.topology().dim())
    markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    for facet in facets(mesh):
        if facet.exterior():
            x = facet.midpoint().x()
            markers[facet] = (1 if near(x, fluid_x_min, geometry_tolerance)
                              else 2 if near(x, fluid_x_max, geometry_tolerance) else 3)
    return markers


fluid_boundary_markers = mark_fluid_boundaries(fluid_mesh)
reference_boundary_markers = mark_fluid_boundaries(fluid_reference_mesh)
ds_f = Measure("ds", domain=fluid_mesh, subdomain_data=fluid_boundary_markers)
dx_f = Measure("dx", domain=fluid_mesh)
dx_f_reference = Measure("dx", domain=fluid_reference_mesh)
geometric_n_f = FacetNormal(fluid_mesh)
h_f = CellDiameter(fluid_mesh)

# Equal-order P1--P1 is retained from the supplied code and stabilized below.
V_f = VectorFunctionSpace(fluid_mesh, "CG", 1)
Q_f = FunctionSpace(fluid_mesh, "CG", 1)
W_f = FunctionSpace(
    fluid_mesh, MixedElement([V_f.ufl_element(), Q_f.ufl_element()])
)
V_f_reference = VectorFunctionSpace(fluid_reference_mesh, "CG", 1)
Q_f_reference = FunctionSpace(fluid_reference_mesh, "CG", 1)

u_f, p_f = TrialFunctions(W_f)
v_f, q_f = TestFunctions(W_f)
up_f_old = Function(W_f, name="previous_fluid_state")
u_f_old, _ = split(up_f_old)
up_f = Function(W_f, name="fluid_state")

# Current-domain ALE velocity.
w_f = Function(V_f, name="ALE_mesh_velocity")

# Total harmonic displacement is solved on the fixed reference mesh.  Its
# increment is copied to the matching current mesh before ALE.move is called.
mesh_displacement_trial = TrialFunction(V_f_reference)
mesh_displacement_test = TestFunction(V_f_reference)
mesh_displacement_total = Function(V_f_reference, name="total_ALE_displacement")
mesh_displacement_old = Function(V_f_reference, name="previous_total_ALE_displacement")
mesh_displacement_increment_reference = Function(V_f_reference)
mesh_displacement_increment_current = Function(V_f)

a_lap = inner(grad(mesh_displacement_trial), grad(mesh_displacement_test)) * dx_f_reference
L_lap = inner(Constant((0.0, 0.0, 0.0)), mesh_displacement_test) * dx_f_reference

p_inlet = Constant(0.0)
########################################### Structure initialization #########################################

if os.path.exists(shell_xml_path):
    structure_mesh = Mesh(shell_xml_path)
else:
    structure_mesh = Mesh()
    with XDMFFile(shell_xdmf_path) as infile:
        infile.read(structure_mesh)
if structure_mesh.topology().dim() != 2:
    raise RuntimeError("The Naghdi shell parameter mesh must be two-dimensional.")

x_s, y_s = SpatialCoordinate(structure_mesh)
dx_s = Measure("dx", domain=structure_mesh, metadata={"quadrature_degree":args.quadrature_degree})
structure_x_min = float(structure_mesh.coordinates()[:, 0].min())
structure_x_max = float(structure_mesh.coordinates()[:, 0].max())
structure_y_min = float(structure_mesh.coordinates()[:, 1].min())
structure_y_max = float(structure_mesh.coordinates()[:, 1].max())
if not math.isclose(structure_x_min, 0.0, abs_tol=geometry_tolerance):
    raise RuntimeError("The shell parameter domain must start at x=0.")
if not math.isclose(structure_x_max, width, rel_tol=1.0e-8, abs_tol=geometry_tolerance):
    raise RuntimeError("The shell parameter-domain length does not equal L=0.06 m.")
if not math.isclose(structure_y_min, 0.0, abs_tol=geometry_tolerance):
    raise RuntimeError("The shell circumferential coordinate must start at zero.")
if not math.isclose(structure_y_max, length, rel_tol=1.0e-8, abs_tol=geometry_tolerance):
    raise RuntimeError("The shell circumference does not equal 2*pi*R_0.")

print("Shell x interval: [{:.6e}, {:.6e}] m".format(structure_x_min, structure_x_max))
print("Shell circumferential interval: [{:.6e}, {:.6e}] m".format(structure_y_min, structure_y_max))


class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.0, geometry_tolerance)

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] - length


pbc = PeriodicBoundary()

# Six shell unknowns: three Cartesian midsurface-displacement components and
# three Cartesian infinitesimal-rotation components.
V_s = VectorFunctionSpace(structure_mesh, "CG", 1, dim=6, constrained_domain=pbc)
V_structure_vec = VectorFunctionSpace(
    structure_mesh, "CG", 1, dim=3, constrained_domain=pbc
)
V_structure_scalar = FunctionSpace(
    structure_mesh, "CG", 1, constrained_domain=pbc
)
u_s_trial = TrialFunction(V_s)
v_s_trial = TrialFunction(V_s)
v_s_test = TestFunction(V_s)

u_s_n = Function(V_s, name="shell_state")
v_s_completed = Function(V_s, name="shell_completed_velocity")

# Exact matching-node reference map from the mesh generator.
geometry_data = np.load(Path(args.mesh_dir)/'reference_shell_geometry.npz')
reference_shell_coordinates = geometry_data['coordinates']
assert np.allclose(geometry_data['parameter_coordinates'],structure_mesh.coordinates(),atol=1e-12)
V_phi = V_structure_vec
phi0 = Function(V_phi,name='reference_midsurface')
phi_coeff = np.zeros(V_phi.dim())
phi_vertex = vertex_to_dof_map(V_phi).reshape((-1,3))
for vertex,xyz in enumerate(reference_shell_coordinates):phi_coeff[phi_vertex[vertex]]=xyz
phi0.vector().set_local(phi_coeff);phi0.vector().apply('insert')

material_tags = MeshFunction('size_t',structure_mesh,str(Path(args.mesh_dir)/'shell_materials.xml'))
material_space = FunctionSpace(structure_mesh,'DG',0)
young = Function(material_space,name='Young_modulus_Pa')
thickness_field = Function(material_space,name='shell_thickness_m')
material_id = Function(material_space,name='material_id')
yv=np.zeros(material_space.dim());hv=yv.copy();iv=yv.copy()
for cell in cells(structure_mesh):
    dof=material_space.dofmap().cell_dofs(cell.index())[0]
    steel=material_tags[cell]==2
    yv[dof]=args.stent_young_modulus if steel else args.wall_young_modulus
    hv[dof]=args.wall_thickness+(args.stent_thickness if steel else 0.)
    iv[dof]=2 if steel else 1
for function,values in [(young,yv),(thickness_field,hv),(material_id,iv)]:
    function.vector().set_local(values);function.vector().apply('insert')
# Thickness coordinate z is positive outward from the artery midsurface.
# Artery: [-hw/2,hw/2]; steel where present: [hw/2,hw/2+hs].
layer_mask=conditional(gt(material_id,Constant(1.5)),Constant(1.),Constant(0.))
hw=Constant(args.wall_thickness);hs=Constant(args.stent_thickness)
Ew=Constant(args.wall_young_modulus);Es=Constant(args.stent_young_modulus)
z0=hw/2;z1=hw/2+hs
E0=Ew*hw+layer_mask*Es*hs
E1=layer_mask*Es*(z1**2-z0**2)/2
E2=Ew*hw**3/12+layer_mask*Es*(z1**3-z0**3)/3
m0=rho_s*(hw+layer_mask*hs)
m1=rho_s*layer_mask*(z1**2-z0**2)/2
m2=rho_s*(hw**3/12+layer_mask*(z1**3-z0**3)/3)
mu_unit=1/(2*(1+nu));trace_unit=nu/(1-nu**2)
# Build an orthonormal frame and directional derivatives on each reference
# triangle. This reduces to the cylinder's local law on an orthonormal chart.
a1=as_vector(tuple(phi0[k].dx(0) for k in range(3)))
a2=as_vector(tuple(phi0[k].dx(1) for k in range(3)))
reference_cross=cross(a1,a2)
reference_area=sqrt(inner(reference_cross,reference_cross))
a_3=reference_cross/reference_area
frame1=a1/sqrt(inner(a1,a1))
frame2=cross(a_3,frame1)
metric=as_matrix([[dot(a1,a1),dot(a1,a2)],[dot(a2,a1),dot(a2,a2)]])
metric_inv=inv(metric)
dual=[metric_inv[i,0]*a1+metric_inv[i,1]*a2 for i in range(2)]
frames=[frame1,frame2,a_3]

def directional(field,direction):
    return as_vector(tuple(sum(dot(dual[j],direction)*field[k].dx(j) for j in range(2)) for k in range(3)))

def shell_strains(state):
    displacement=as_vector(tuple(state[k] for k in range(3)))
    rotation=as_vector(tuple(state[k] for k in range(3,6)))
    membrane=[directional(displacement,e)+cross(e,rotation) for e in frames[:2]]
    bending=[directional(rotation,e) for e in frames[:2]]
    A=as_matrix([[dot(frames[i],membrane[j]) for j in range(2)] for i in range(2)])
    e=as_vector(tuple(dot(a_3,c) for c in membrane))
    B=as_matrix([[dot(frames[i],bending[j]) for j in range(2)] for i in range(2)])
    b=as_vector(tuple(dot(a_3,c) for c in bending))
    return A,e,B,b

######################################## Fluid weak formulation ##################################

F_time = (rho_f / dt) * inner(u_f - u_f_old, v_f) * dx_f
#F_adv = rho_f * dot( dot(convecting_velocity - w_f, nabla_grad(u_f)), v_f) * dx_f

convecting_velocity = Function(V_f, name="Picard_fluid_velocity")
F_adv = 0.5 * rho_f * (
    dot( dot(convecting_velocity - w_f, nabla_grad(u_f)), v_f)  
     -
    dot( dot(convecting_velocity - w_f, nabla_grad(v_f)), u_f )
) * dx_f

F_div = q_f * div(u_f) * dx_f
F_p = -p_f * div(v_f) * dx_f
F_diss = 2.0 * mu_f * inner(sym(grad(u_f)), sym(grad(v_f))) * dx_f
F_stab = (
    Constant(args.pressure_stabilization)
    * h_f**2
    * dot(grad(p_f), grad(q_f))
    * dx_f
)

fluid_residual = (
    F_time
    + F_adv
    + F_div
    + F_p
    + F_diss
    + F_stab

)
a_f = lhs(fluid_residual)
# The inlet is the plane x=0, whose outward normal is -e_x.  Therefore the
# prescribed traction -p_in n equals +p_in e_x.  Writing this vector
# explicitly makes the pressure-load direction independent of imported-cell
# orientation conventions in legacy XML meshes.
inlet_traction = as_vector((p_inlet, Constant(0.0), Constant(0.0)))
L_f = rhs(fluid_residual) + dot(inlet_traction, v_f) * ds_f(1)

######################################## Structure weak formulation ##################################

A_trial,e_trial,B_trial,b_trial=shell_strains(u_s_trial)
A_test,e_test,B_test,b_test=shell_strains(v_s_test)
# Outward layer coordinate corresponds to -a_3 for this parameterization.
J=as_matrix([[0.,-1.],[1.,0.]])
K_trial=J*B_trial;K_test=J*B_test

def tangent_material(A,B):
    return trace_unit*tr(A)*tr(B)+2*mu_unit*inner(A,B)

a_membrane=E0*(tangent_material(A_trial,A_test)+mu_unit*dot(e_trial,e_test))*reference_area*dx_s
a_coupling=E1*(tangent_material(A_trial,K_test)+tangent_material(K_trial,A_test))*reference_area*dx_s
a_bending=E2*(tangent_material(K_trial,K_test)+mu_unit*dot(b_trial,b_test))*reference_area*dx_s
a_structure=a_membrane+a_coupling+a_bending
vs=mass_disp(v_s_trial);ws=mass_rotation(v_s_trial)
vt=mass_disp(v_s_test);wt=mass_rotation(v_s_test)
n_out=-a_3
a_mass=(m0*dot(vs,vt)+m1*(dot(vs,cross(wt,n_out))+dot(cross(ws,n_out),vt))
    +m2*dot(ws,wt))*reference_area*dx_s
a_interface=inner(mass_disp(v_s_trial),mass_disp(v_s_test))*reference_area*dx_s


def clamped_boundary(x, on_boundary):
    return on_boundary and (
        near(x[0], structure_x_min, geometry_tolerance)
        or near(x[0], structure_x_max, geometry_tolerance)
    )


bc_s = DirichletBC(
    V_s,
    Constant((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    clamped_boundary,
)

# Current shell map and surface Jacobian in fixed shell coordinates.
current_shell_map=as_vector(tuple(phi0[k]+u_s_n[k] for k in range(3)))
current_tangent_1 = as_vector(
    tuple(current_shell_map[i].dx(0) for i in range(3))
)
current_tangent_2 = as_vector(
    tuple(current_shell_map[i].dx(1) for i in range(3))
)
current_cross_product = cross(current_tangent_1, current_tangent_2)
J_s_expression = sqrt(inner(current_cross_product, current_cross_product))
orientation_expression = inner(current_cross_product, a_3)
orientation_current = Function(V_structure_scalar, name="surface_orientation")
fluid_outward_normal_shell = Function(
    V_structure_vec, name="current_outward_fluid_normal"
)

# The shell matrices are constant because the benchmark reference geometry is
# the straight cylinder with sqrt(a)=1.
A_s = assemble(a_structure)
M_s = assemble(a_mass)

# Reference-mesh boundary data and ALE boundary conditions.
shell_displacement_on_reference_fluid = Function(V_f_reference)
zero_reference_displacement = Constant((0.0, 0.0, 0.0))
bc_mesh_inlet = DirichletBC(
    V_f_reference, zero_reference_displacement, reference_boundary_markers, 1
)
bc_mesh_outlet = DirichletBC(
    V_f_reference, zero_reference_displacement, reference_boundary_markers, 2
)
bc_mesh_wall = DirichletBC(
    V_f_reference,
    shell_displacement_on_reference_fluid,
    reference_boundary_markers,
    3,
)


def shell_vector_to_reference_fluid(shell_vector):
    expression = StructureVectorOnReferenceFluid(
        shell_vector, Radius, degree=2
    )
    return interpolate(expression, V_f_reference)


def shell_scalar_to_reference_fluid(shell_scalar):
    expression = StructureScalarOnReferenceFluid(
        shell_scalar, Radius, degree=2
    )
    return interpolate(expression, Q_f_reference)


def current_fluid_vector_to_shell(fluid_vector):
    expression = CurrentFluidVectorOnShell(
        fluid_vector, u_s_n, Radius, degree=2
    )
    return interpolate(expression, V_structure_vec)


def make_reference_axial_markers(current_mesh, reference_mesh, number_of_bins):
    """Mark current-mesh cells using the corresponding reference cell centers."""
    markers = MeshFunction("size_t", current_mesh, current_mesh.topology().dim(), 0)
    bin_width = width / float(number_of_bins)
    for reference_cell in cells(reference_mesh):
        index = reference_cell.index()
        axial_coordinate = reference_cell.midpoint().x()
        bin_index = min(
            number_of_bins - 1,
            max(0, int((axial_coordinate - fluid_x_min) / bin_width)),
        )
        markers[index] = bin_index + 1
    return markers


def make_shell_axial_markers(number_of_bins):
    markers = MeshFunction(
        "size_t", structure_mesh, structure_mesh.topology().dim(), 0
    )
    bin_width = width / float(number_of_bins)
    for shell_cell in cells(structure_mesh):
        axial_coordinate = shell_cell.midpoint().x()
        bin_index = min(
            number_of_bins - 1,
            max(0, int((axial_coordinate - structure_x_min) / bin_width)),
        )
        markers[shell_cell] = bin_index + 1
    return markers


fluid_axial_markers = make_reference_axial_markers(
    fluid_mesh, fluid_reference_mesh, args.profile_bins
)
shell_axial_markers = make_shell_axial_markers(args.profile_bins)
dx_f_profile = Measure(
    "dx", domain=fluid_mesh, subdomain_data=fluid_axial_markers
)
dx_s_profile = Measure(
    "dx", domain=structure_mesh, subdomain_data=shell_axial_markers
)
profile_bin_width = width / float(args.profile_bins)
profile_x_coordinates = np.array(
    [(i + 0.5) * profile_bin_width for i in range(args.profile_bins)]
)
midpoint_bin = int(np.argmin(np.abs(profile_x_coordinates - 0.5 * width)))
radial_direction = as_vector(
    (0.0, cos(y_s / Radius), sin(y_s / Radius))
)


def compute_axial_profile(fluid_velocity,fluid_pressure,shell_state):
    """Exact P1 tetrahedron means; 3-point reference-triangle shell quadrature."""
    global profile_radials
    conn=fluid_reference_mesh.cells();xyz=fluid_mesh.coordinates()
    v=fluid_velocity.compute_vertex_values(fluid_mesh).reshape((3,-1)).T
    p=fluid_pressure.compute_vertex_values(fluid_mesh)
    volumes=np.abs(np.einsum('ij,ij->i',xyz[conn[:,1]]-xyz[conn[:,0]],
        np.cross(xyz[conn[:,2]]-xyz[conn[:,0]],xyz[conn[:,3]]-xyz[conn[:,0]])))/6
    volume=np.bincount(fluid_profile_bins,weights=volumes,minlength=args.profile_bins)
    meanp=np.bincount(fluid_profile_bins,weights=volumes*p[conn].mean(axis=1),minlength=args.profile_bins)/volume
    flow=np.bincount(fluid_profile_bins,weights=volumes*v[conn,0].mean(axis=1),minlength=args.profile_bins)/profile_bin_width
    state=shell_state.compute_vertex_values(structure_mesh).reshape((6,-1)).T
    disp=state[:,:3][shell_connectivity]
    radial=np.zeros(len(shell_connectivity));radii=radial.copy()
    for weights in triangle_quadrature:
        uq=np.einsum('j,ijk->ik',weights,disp)
        xq=np.einsum('j,ijk->ik',weights,reference_shell_coordinates[shell_connectivity])
        theta=np.einsum('j,ij->i',weights,shell_parameter_coordinates[shell_connectivity,1])/Radius
        radial+=(uq[:,1]*np.cos(theta)+uq[:,2]*np.sin(theta))/3
        radii+=np.sqrt(xq[:,1]**2+xq[:,2]**2)/3
    area=np.bincount(shell_profile_bins,weights=reference_triangle_areas,minlength=args.profile_bins)
    profile_radials=np.bincount(shell_profile_bins,weights=radial*reference_triangle_areas,minlength=args.profile_bins)/area
    baseline=np.bincount(shell_profile_bins,weights=radii*reference_triangle_areas,minlength=args.profile_bins)/area
    return [(x,float(pp),float(q),float(2*(rr+dr))) for x,pp,q,rr,dr in zip(profile_x_coordinates,meanp,flow,baseline,profile_radials)]


def write_profile(writer, time, profile):
    for axial_coordinate, mean_pressure, flow_rate, diameter in profile:
        writer.writerow(
            [time, axial_coordinate, mean_pressure, flow_rate, diameter]
        )




# Matching-node coupling and a shared reference-surface mass pairing.
def csr(matrix):
    ip,ix,values=as_backend_type(matrix).mat().getValuesCSR()
    return sparse.csr_matrix((values.copy(),ix.copy(),ip.copy()),shape=(matrix.size(0),matrix.size(1)))

def put(function,values):
    function.vector().set_local(values);function.vector().apply('insert')

interface_nodes=[(int(r['shell_vertex']),int(r['fluid_vertex'])) for r in csv.DictReader(open(Path(args.mesh_dir)/'interface_node_map.csv'))]
velocity_space,parent=W_f.sub(0).collapse(collapsed_dofs=True)
fv_dofs=vertex_to_dof_map(velocity_space).reshape((-1,3))
sv_dofs=vertex_to_dof_map(V_s).reshape((-1,6))
ref_v_dofs=vertex_to_dof_map(V_f_reference).reshape((-1,3))
wall_rows={}
for sv,fv in interface_nodes:
    if np.linalg.norm(reference_shell_coordinates[sv]-fluid_reference_mesh.coordinates()[fv])>1e-10:
        raise RuntimeError('Nonmatching reference interface')
    for c in range(3):
        fd=parent[int(fv_dofs[fv,c])];sd=int(sv_dofs[sv,c])
        if fd in wall_rows and wall_rows[fd]!=sd:raise RuntimeError('Periodic interface mismatch')
        wall_rows[fd]=sd
required=set()
fluid_mesh.init(2,0)
for facet in facets(fluid_mesh):
    if fluid_boundary_markers[facet]==3:
        required.update(parent[int(fv_dofs[int(v),c])] for v in facet.entities(0) for c in range(3))
assert required==set(wall_rows)
nf,ns=W_f.dim(),V_s.dim();clamped=set(bc_s.get_boundary_values())
shell_free=np.array([i for i in range(ns) if i not in clamped],dtype=np.int32)
fluid_free=np.array([i for i in range(nf) if i not in wall_rows],dtype=np.int32)
clamped_fluid={fd for fd,sd in wall_rows.items() if sd in clamped}
fluid_explicit_free=np.array([i for i in range(nf) if i not in clamped_fluid],dtype=np.int32)
reverse={sd:fd for fd,sd in wall_rows.items()}
S=sparse.csr_matrix((np.ones(len(reverse)),(list(reverse),list(reverse.values()))),shape=(ns,nf))
rr=list(fluid_free)+list(nf+shell_free)
cc=list(range(len(fluid_free)))+list(len(fluid_free)+np.arange(len(shell_free)))
shell_reduced={int(d):i for i,d in enumerate(shell_free)}
for fd,sd in wall_rows.items():
    if sd in shell_reduced:rr.append(fd);cc.append(len(fluid_free)+shell_reduced[sd])
R=sparse.csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(nf+ns,len(fluid_free)+len(shell_free)))
M,K,G=csr(M_s),csr(A_s),csr(assemble(a_interface))
H=M/dt+dt*K
stiff_explicit=H+args.robin_parameter*G
fluid_robin=args.robin_parameter*(S.T@G@S)
translation_mask=np.asarray(G.getnnz(axis=1)>0,dtype=float)

# Both mass and stiffness must be symmetric; rigid motions must have zero
# elastic strain before clamping. These checks catch reference-frame errors.
def sparse_frobenius(A):return float(np.sqrt(A.multiply(A).sum()))
knorm=sparse_frobenius(K)
assert sparse_frobenius(K-K.T)/max(knorm,1e-30)<1e-10
assert sparse_frobenius(M-M.T)/max(sparse_frobenius(M),1e-30)<1e-10
rigid_defects=[]
for axis in range(6):
    z=np.zeros(ns)
    for sv in range(structure_mesh.num_vertices()):
        if axis<3:z[sv_dofs[sv,axis]]=1.
        else:
            om=np.eye(3)[axis-3]
            z[sv_dofs[sv,:3]]=np.cross(om,reference_shell_coordinates[sv])
            z[sv_dofs[sv,3:]]=om
    defect=float(np.linalg.norm(K@z)/max(knorm*np.linalg.norm(z),1e-30))
    rigid_defects.append(defect)
assert max(rigid_defects)<1e-9,rigid_defects

class DirectSolver:
    def __init__(self,A):
        from petsc4py import PETSc
        A=A.tocsr();A.sort_indices();self.A=A
        self.pmat=PETSc.Mat().createAIJ(size=A.shape,csr=(A.indptr.astype(PETSc.IntType),A.indices.astype(PETSc.IntType),A.data),comm=PETSc.COMM_SELF)
        self.matrix=PETScMatrix(self.pmat)
        self.b=PETScVector(MPI.comm_self);self.matrix.init_vector(self.b,0)
        self.x=self.b.copy();self.solver=LUSolver(self.matrix,args.linear_solver)
    def solve(self,b):
        if not np.any(b):return np.zeros(self.A.shape[0]),0.
        self.b.set_local(np.asarray(b));self.b.apply('insert');self.x.zero()
        self.solver.solve(self.x,self.b)
        z=self.x.get_local();res=float(np.linalg.norm(self.A@z-b)/max(np.linalg.norm(b),1e-30))
        if not np.all(np.isfinite(z)) or res>1e-7:raise RuntimeError('Linear solve failed: '+str(res))
        return z,res

shell_solver=DirectSolver(stiff_explicit[shell_free][:,shell_free]) if args.scheme=='explicit' else None
reference_coordinates=fluid_reference_mesh.coordinates().copy()
fluid_profile_bins=np.minimum(args.profile_bins-1,((reference_coordinates[fluid_reference_mesh.cells()].mean(axis=1)[:,0]-fluid_x_min)/profile_bin_width).astype(int))
shell_connectivity=structure_mesh.cells();shell_parameter_coordinates=structure_mesh.coordinates()
shell_profile_bins=np.minimum(args.profile_bins-1,(shell_parameter_coordinates[shell_connectivity].mean(axis=1)[:,0]/profile_bin_width).astype(int))
reftris=reference_shell_coordinates[shell_connectivity]
reference_triangle_areas=.5*np.linalg.norm(np.cross(reftris[:,1]-reftris[:,0],reftris[:,2]-reftris[:,0]),axis=1)
triangle_quadrature=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
old_shell=Function(V_s);velocity=Function(V_s);candidate=Function(V_s)

# Factor the fixed harmonic extension only once, preserving its old/new trace.
lap_matrix=assemble(a_lap)
for bc in [bc_mesh_inlet,bc_mesh_outlet,bc_mesh_wall]:bc.apply(lap_matrix)
lap_solver=LUSolver(lap_matrix,args.linear_solver)

def geometry(state,old_mesh):
    u_s_n.assign(state)
    if args.fixed_domain:
        mesh_displacement_total.vector().zero();w_f.vector().zero()
        fluid_mesh.coordinates()[:]=reference_coordinates
    else:
        trace=np.zeros(V_f_reference.dim());v=state.vector().get_local()
        for sv,fv in interface_nodes:trace[ref_v_dofs[fv]]=v[sv_dofs[sv,:3]]
        put(shell_displacement_on_reference_fluid,trace)
        rhs=assemble(L_lap)
        for bc in [bc_mesh_inlet,bc_mesh_outlet,bc_mesh_wall]:bc.apply(rhs)
        if np.any(trace):lap_solver.solve(mesh_displacement_total.vector(),rhs)
        else:mesh_displacement_total.vector().zero()
        put(w_f,(mesh_displacement_total.vector().get_local()-old_mesh)/dt)
        displacement=mesh_displacement_total.compute_vertex_values(fluid_reference_mesh).reshape((3,-1)).T
        fluid_mesh.coordinates()[:]=reference_coordinates+displacement
    fluid_mesh.bounding_box_tree().build(fluid_mesh)
    ratios=signed_tetrahedron_jacobians(fluid_mesh)/reference_tetrahedron_jacobians
    if ratios.min()<=0:raise RuntimeError('Inverted ALE cell')
    return float(ratios.min())

Path(args.output_dir).mkdir(parents=True,exist_ok=True)
settings=vars(args).copy();settings.update(solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    fluid_dofs=nf,shell_dofs=ns,reference_surface='common inward radius graph from supplied generator',
    material_model='0.4 mm artery everywhere plus 0.15 mm perfectly bonded outer steel in strut bands; laminate E0,E1,E2 and m0,m1,m2',
    density_kg_per_m3=rho_s,poisson_ratio=nu,rigid_motion_relative_stiffness_defects=rigid_defects,
    full_state_time_levels='step 1, every output-every steps, six profile snapshots, final step',
    convection='same lagged skew form in both schemes',interface_pairing='common reference-surface mass with transpose transfer')
Path(args.output_dir,'run_settings.json').write_text(json.dumps(settings,indent=2))
for name,function in [('Young_modulus_Pa',young),('shell_thickness_m',thickness_field),('material_id',material_id),('reference_midsurface',phi0)]:
    File(str(Path(args.output_dir)/(name+'.pvd')),'compressed')<<function
outputs={name:File(str(Path(args.output_dir)/(name+'.pvd')),'compressed') for name in
    ['fluid_velocity','fluid_pressure','shell_displacement','shell_rotation','shell_velocity','current_shell_midsurface','ALE_total_displacement_reference']}
profile_times=[.002,.004,.006,.008,.010,.012]
snapshots={round(t/dt) for t in profile_times if t<=T+.5*dt}
headers=['time_s','inlet_pressure_load_Pa','inlet_flow_m3_per_s','outlet_flow_m3_per_s',
 'midpoint_mean_pressure_Pa','midpoint_mean_radial_displacement_m','maximum_mean_radial_displacement_m',
 'fluid_kinetic_energy_J','shell_kinetic_energy_J','shell_elastic_energy_J','total_mechanical_energy_J',
 'min_ALE_tetrahedron_jacobian_ratio','nonlinear_iterations','linear_relative_residual',
 'velocity_mismatch_reference_L2','force_imbalance_reference_L2','interface_virtual_work_defect_W',
 'geometry_displacement_max_abs_defect_m','pressure_stabilization_increment_J',
 'viscous_dissipation_increment_J','inlet_work_increment_J','energy_balance_defect_J']
zero_shell=np.zeros(ns);old_velocity=zero_shell.copy();old_fluid=np.zeros(nf)
fluid_trace_old=zero_shell.copy();fluid_load_old=zero_shell.copy();shell_load_old=zero_shell.copy()
previous_energy=0.
print('Starting',args.scheme,'fluid DOFs',nf,'shell DOFs',ns,'rigid-motion check',max(rigid_defects),flush=True)
with open(Path(args.output_dir)/'diagnostics.csv','w',newline='') as dh, \
     open(Path(args.output_dir)/'axial_profiles.csv','w',newline='') as ph, \
     open(Path(args.output_dir)/'nonlinear_iterations.csv','w',newline='') as ih:
    dw,pw,iw=csv.writer(dh),csv.writer(ph),csv.writer(ih)
    dw.writerow(headers);pw.writerow(['time_s','axial_coordinate_m','mean_pressure_Pa','flow_rate_m3_per_s','mean_diameter_m','mean_radial_displacement_m'])
    iw.writerow(['step','iteration','shell_relative_increment','fluid_relative_increment','linear_relative_residual'])
    for step in range(1,num_steps+1):
        t=step*dt;p_inlet.assign(inlet_pressure_value(t,maximum_inlet_pressure,pulse_duration))
        d_old=u_s_n.vector().get_local().copy();v_old=old_velocity.copy()
        mesh_old=mesh_displacement_old.vector().get_local().copy()
        shell_rhs=M@v_old/dt-K@d_old
        # Fix convection at the old material velocity in both schemes.
        put(up_f_old,old_fluid);convecting_velocity.assign(up_f_old.split(deepcopy=True)[0])
        if args.scheme=='monolithic':
            v_guess=v_old.copy();u_guess=old_fluid.copy()
            for iteration in range(1,args.max_iterations+1):
                put(candidate,d_old+dt*v_guess);minimum_j=geometry(candidate,mesh_old)
                Af=csr(assemble(a_f));bf=assemble(L_f).get_local()
                block=sparse.block_diag((Af,H),format='csr');rhs=np.concatenate([bf,shell_rhs])
                reduced=(R.T@block@R).tocsr()
                solver=DirectSolver(reduced);z,linres=solver.solve(np.asarray(R.T@rhs))
                full=np.asarray(R@z);u_new=full[:nf];v_new=full[nf:]
                du=u_new-u_guess;dv=v_new-v_guess
                uv=W_f.sub(0).dofmap().dofs()
                eu=np.linalg.norm(du[uv])/max(np.linalg.norm(u_new[uv]),args.absolute_tolerance)
                ev=np.linalg.norm(dv)/max(np.linalg.norm(v_new),args.absolute_tolerance)
                iw.writerow([step,iteration,ev,eu,linres]);ih.flush()
                print('step',step,'iteration',iteration,'increments',eu,ev,flush=True)
                conv_u=eu<args.relative_tolerance or np.max(np.abs(du[uv]))<args.absolute_tolerance
                conv_v=ev<args.relative_tolerance or np.max(np.abs(dv))<args.absolute_tolerance
                if iteration>=2 and conv_u and conv_v:break
                u_guess+=args.relaxation*du;v_guess+=args.relaxation*dv
            else:raise RuntimeError('Geometry iteration failed at step '+str(step))
            reaction=block@full-rhs
            virtual_work=float(sum(reaction[fd]*u_new[fd] for fd in wall_rows)+reaction[nf:]@v_new)
            geom_defect=dt*float(np.max(np.abs(dv)))
            force_imbalance=0.
        else:
            gamma=args.robin_parameter
            rf=(.5*gamma*(fluid_trace_old+v_old)+.5*(fluid_load_old-shell_load_old))*translation_mask
            rs=(.5*gamma*(fluid_trace_old+v_old)-.5*(fluid_load_old-shell_load_old))*translation_mask
            put(candidate,d_old+dt*v_old);minimum_j=geometry(candidate,mesh_old)
            Af=csr(assemble(a_f));bf=assemble(L_f).get_local()
            matrix=Af+fluid_robin;rhs=bf+S.T@(G@rf)
            solver=DirectSolver(matrix[fluid_explicit_free][:,fluid_explicit_free])
            u_new=np.zeros(nf);u_new[fluid_explicit_free],linres=solver.solve(np.asarray(rhs)[fluid_explicit_free])
            shell_right=shell_rhs+G@rs
            v_new=np.zeros(ns);v_new[shell_free],sr=shell_solver.solve(np.asarray(shell_right)[shell_free])
            linres=max(linres,sr);iteration=1;geom_defect=dt*float(np.max(np.abs(v_new-v_old)))
            fluid_load_new=(rf-gamma*(S@u_new))*translation_mask
            shell_load_new=(rs-gamma*v_new)*translation_mask
            force_imbalance=math.sqrt(max(0.,float((fluid_load_new+shell_load_new)@(G@(fluid_load_new+shell_load_new)))))
            virtual_work=float(fluid_load_new@(G@(S@u_new))+shell_load_new@(G@v_new))
            fluid_load_old=fluid_load_new;shell_load_old=shell_load_new
            iw.writerow([step,1,0,0,linres]);ih.flush()
        put(up_f,u_new);put(velocity,v_new);put(u_s_n,d_old+dt*v_new)
        orientation=project(dot(current_cross_product,reference_cross)/reference_area**2,material_space)
        if orientation.vector().min()<=0:raise RuntimeError('Shell surface orientation lost')
        uf,pf=up_f.split(deepcopy=True)
        displacement=project(mass_disp(u_s_n),V_structure_vec)
        rotation=project(mass_rotation(u_s_n),V_structure_vec)
        translation=project(mass_disp(velocity),V_structure_vec)
        profile=compute_axial_profile(uf,pf,u_s_n)
        radial_means=profile_radials
        midpoint=profile[midpoint_bin]
        ekf=.5*rho_f*assemble(inner(uf,uf)*dx_f)
        eks=.5*float(v_new@(M@v_new));deformed=u_s_n.vector().get_local()
        elastic=.5*float(deformed@(K@deformed));energy=ekf+eks+elastic
        diss=dt*2*mu_f*assemble(inner(sym(grad(uf)),sym(grad(uf)))*dx_f)
        stab=dt*args.pressure_stabilization*assemble(h_f**2*inner(grad(pf),grad(pf))*dx_f)
        qi,qo=assemble(uf[0]*ds_f(1)),assemble(uf[0]*ds_f(2))
        mismatch=S@u_new-v_new
        velocity_mismatch=math.sqrt(max(0.,float(mismatch@(G@mismatch))))
        work=dt*float(p_inlet)*qi
        row=[t,float(p_inlet),qi,qo,midpoint[1],radial_means[midpoint_bin],max(radial_means),ekf,eks,elastic,energy,
             minimum_j,iteration,linres,velocity_mismatch,force_imbalance,virtual_work,geom_defect,stab,diss,work,
             energy-previous_energy+diss+stab-work]
        if not np.all(np.isfinite(row)):raise RuntimeError('Nonfinite diagnostic data')
        dw.writerow(row);dh.flush()
        if step in snapshots:
            for entry,radial in zip(profile,radial_means):pw.writerow([t]+list(entry)+[radial])
            ph.flush()
        if step==1 or step%args.output_every==0 or step==num_steps or step in snapshots:
            fields=dict(fluid_velocity=uf,fluid_pressure=pf,shell_displacement=displacement,
                shell_rotation=rotation,shell_velocity=translation,
                current_shell_midsurface=project(phi0+displacement,V_phi),
                ALE_total_displacement_reference=mesh_displacement_total)
            for name,field in fields.items():field.rename(name,name);outputs[name]<<(field,t)
        old_fluid=u_new.copy();old_velocity=v_new.copy();fluid_trace_old=S@u_new
        mesh_displacement_old.assign(mesh_displacement_total);previous_energy=energy
        print('Step %d/%d: t=%.6e s, midpoint pressure=%.6e Pa, radial=%.6e m, minALE=%.6e, linear=%.3e'%
              (step,num_steps,t,midpoint[1],radial_means[midpoint_bin],minimum_j,linres),flush=True)
print('Simulation complete. Results are in '+args.output_dir,flush=True)
