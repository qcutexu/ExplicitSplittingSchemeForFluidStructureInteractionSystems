"""Monolithic fluid--Naghdi shell benchmark (serial legacy FEniCS).

Fluid velocity/pressure and six shell velocities are solved in one constrained
system. Strong matching-node velocity continuity and transpose assembly preserve
interface virtual work. Backward Euler advances shell displacement; Picard
iteration converges current ALE geometry and current skew convection together.
There is no beta or Robin parameter.

At each time step d = d_old + dt*v and the assembled linear system is
    R.T diag(A_fluid, M_shell/dt + dt*K_shell) R z
      = R.T [b_fluid, M_shell*v_old/dt - K_shell*d_old].
R identifies fluid wall velocity with shell translation and removes clamped
shell DOFs. The transpose gives exactly the same interface virtual work on
both sides. ALE geometry and the skew convecting velocity are Picard-iterated;
this is not a Newton solve. The old fluid coefficients follow reference nodes.
Use --fixed-domain for a fixed-domain verification problem (shell displacement
then does not deform the fluid domain). Serial execution only.

Example (in the FEniCS environment):
    python fsi_naghdi_pressure_pulse_monolithic.py --mesh-dir meshes
Dependencies: legacy dolfin, NumPy, SciPy, petsc4py, and a PETSc LU backend.
Outputs: ParaView PVD/VTU series, axial_profiles.csv, diagnostics.csv,
nonlinear_iterations.csv, and run_settings.json.
The geometry/material/strain setup is copied from the supplied beta solver.
"""
from dolfin import *
import argparse
import csv
import math
import os
import numpy as np
from ufl import transpose


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
parser.add_argument("--output-dir", default="results_naghdi_monolithic", help="Directory for PVD and CSV output.")
parser.add_argument("--dt", type=float, default=1.0e-4, help="Time step [s].")
parser.add_argument("--final-time", type=float, default=2.4e-2, help="Final time [s].")
parser.add_argument('--max-iterations', type=int, default=30)
parser.add_argument('--relative-tolerance', type=float, default=1e-7)
parser.add_argument('--absolute-tolerance', type=float, default=1e-10)
parser.add_argument('--relaxation', type=float, default=1.0)
parser.add_argument('--linear-solver', default='mumps')
parser.add_argument('--output-every', type=int, default=1)
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
rho_s = 1.1e3
E = 7.5e4
nu = 0.5
thickness = 1.0e-3
Radius = 5.0e-3
width = 6.0e-2
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
if not math.isclose(fluid_x_min, 0.0, abs_tol=geometry_tolerance):
    raise RuntimeError("The benchmark fluid inlet must be located at x=0.")
if not math.isclose(fluid_x_max, width, rel_tol=1.0e-8, abs_tol=geometry_tolerance):
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
dx_s = Measure("dx", domain=structure_mesh)
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

initial_shape = Expression(
    ("x[0]", "Radius*cos(x[1]/Radius)", "Radius*sin(x[1]/Radius)"),
    degree=6,
    Radius=Radius,
)
V_phi = V_structure_vec
phi0 = project(initial_shape, V_phi)
phi0.rename("reference_midsurface", "reference_midsurface")

# Reference normal a_3 points into the fluid; the fluid outward normal is -a_3.
a_3 = as_vector((0.0, -cos(y_s / Radius), -sin(y_s / Radius)))


# ------------------------------------------------------------------------------------------------------------
# 2. Precompute Constant Matrices for Geometry & Stiffness
# ------------------------------------------------------------------------------------------------------------
# 2D identity matrix and rotation matrix J
Id2 = Identity(2)
J = as_matrix([[0, 1],
              [-1, 0]])

# "Trace product" matrix, this is obtained by flatten the 2x2 identity matrix to vector u = [1, 0, 0, 1]^T and then perform out product u X u^T
traceProduct = as_matrix([[1, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 1]])

# e12 matrix
e12 = as_matrix([[1, 0, 0],
                [0, 1, 0]])

# Normal vector (normalaCov), normal to a_1, a_2
normalVector = a_3

# Director vector (a3), here the same as the normal vector, in the flat plain
director = normalVector

# Rows of the local orthonormal frame Q=(a_1^T,a_2^T,a_3^T).
covariantBasis = as_matrix([[1, 0, 0],
                            [0, -sin(y_s/Radius),cos(y_s/Radius)],
                            [0, -cos(y_s/Radius),-sin(y_s/Radius)] ])

detcovariantBasis=1 

# On this arc-length parameterized straight cylinder, the covariant and
# contravariant local frames coincide.
contravariantBasis = covariantBasis

# Contravariant metric (matricaAContrav) is identity 2x2
contravariantMetric = as_matrix([[1, 0],
                                [0, 1]])

# Thickness matrix (debelia3): a 2x3 matrix with zeros on first row and director in second row
thicknessMatrix = as_matrix([[0, 0, 0],
                            [director[0], director[1], director[2]]])
# Outer product of thicknessMatrix (outera3), used to span the space of all the linear map between director space->director space.
outerProductDirector = thicknessMatrix.T * thicknessMatrix  # 3x3 matrix


# ------------------------------------------------------------------------------------------------------------
# 4. Compute Stiffness Matrices
# ------------------------------------------------------------------------------------------------------------
# Bending stiffness components:

QT = contravariantBasis
e12QT = e12 * QT
Je12QT = J * e12QT
ACe12QT = contravariantMetric * e12QT
ACJe12QT = contravariantMetric * Je12QT

# ---------------------------------------------------------
#flexural B_f term in equation (2.3) 
# ---------------------------------------------------------
# K1 = J * e12
K1 = ACJe12QT   # 2x3 matrix
Kron1 = kron_2223_ufl(Id2, K1)  # 4x6 matrix
bendingStiffness11 = detcovariantBasis * coef_trace * (transpose(Kron1) * traceProduct * Kron1)

# K2 = (J * e12)' * (J * e12)
K2 = transpose(Je12QT) * ACJe12QT  # 3x3 matrix
Kron2 = kron_2233_ufl(contravariantMetric, K2)  # 6x6 matrix
bendingStiffness12 = detcovariantBasis * coef_shear * Kron2

# K3 = outerProductDirector
Kron3 = kron_2233_ufl(Id2, outerProductDirector)  # 6x6 matrix
bendingStiffness2 = detcovariantBasis * bending_shear_coefficient * Kron3

# Total bending stiffness matrix (6x6), Cflex
bendingStiffness = bendingStiffness11 + bendingStiffness12 + bendingStiffness2


# ---------------------------------------------------------
#membrane B_ms term in equation (2.3)
# ---------------------------------------------------------
# Membrane stiffness components:
# K4 = e12QT
K4 = e12QT  # 2x3 matrix
Kron4 = kron_2223_ufl(Id2, K4)  # 4x6 matrix
membraneStiffness11 = coef_trace * (transpose(Kron4) * traceProduct * Kron4)

# K5 = e12QT' * AC * e12QT
K5 = transpose(e12QT) * ACe12QT  # 3x3 matrix (diagonal with entries 1, 1, 0)
Kron5 = kron_2233_ufl(contravariantMetric, K5)  # 6x6 matrix
membraneStiffness12 = coef_shear * Kron5

# K6 = outerProductDirector
Kron6 = kron_2233_ufl(contravariantMetric, outerProductDirector)  # 6x6 matrix
membraneStiffness2 = mu_shell * Kron6

# Temporary membrane stiffness, CmemTemp
membraneStiffnessTemp = membraneStiffness11 + membraneStiffness12 + membraneStiffness2

# Membrane transformation matrix (MembraneTransform): 6x9 matrix
membraneTransform = as_matrix([
    [1, 0, 0, 0, 0, 0, 0, -covariantBasis[0, 2], covariantBasis[0, 1]],
    [0, 1, 0, 0, 0, 0, covariantBasis[0, 2], 0, -covariantBasis[0, 0]],
    [0, 0, 1, 0, 0, 0, -covariantBasis[0, 1], covariantBasis[0, 0], 0],
    [0, 0, 0, 1, 0, 0, 0, -covariantBasis[1, 2], covariantBasis[1, 1]],
    [0, 0, 0, 0, 1, 0, covariantBasis[1, 2], 0, -covariantBasis[1, 0]],
    [0, 0, 0, 0, 0, 1, -covariantBasis[1, 1], covariantBasis[1, 0], 0]
])

# Final membrane stiffness matrix (9x9)
membraneStiffness = transpose(membraneTransform)  * membraneStiffnessTemp * membraneTransform


# ------------------------------------------------------------------------------------------------------------
# 5. Set Up the FEniCS Problem
# ------------------------------------------------------------------------------------------------------------

# Convert precomputed matrices to FEniCS Constant objects.
membraneStiffness_const = as_matrix(
    [[membraneStiffness[i, j] for j in range(9)]
     for i in range(9)] )  # 9x9 matrix

bendingStiffness_const = as_matrix(
    [[bendingStiffness[i, j] for j in range(6)]
     for i in range(6)] )   # 6x6 matrix



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

a_membrane = (
    thickness
    * inner(
        dot(membraneStiffness_const, membrane_strain(u_s_trial)),
        membrane_strain(v_s_test),
    )
    * dx_s
)
a_bending = (
    (thickness**3 / 12.0)
    * inner(
        dot(bendingStiffness_const, bending_strain(u_s_trial)),
        bending_strain(v_s_test),
    )
    * dx_s
)
a_structure = a_membrane + a_bending
a_mass = (
    thickness
    * rho_s
    * inner(mass_disp(v_s_trial), mass_disp(v_s_test))
    * dx_s
    + (thickness**3 / 12.0)
    * rho_s
    * inner(mass_rotation(v_s_trial), mass_rotation(v_s_test))
    * dx_s
)

structure_test_displacement = as_vector(
    (v_s_test[0], v_s_test[1], v_s_test[2])
)


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
current_shell_map = as_vector(
    (
        x_s + u_s_n[0],
        Radius * cos(y_s / Radius) + u_s_n[1],
        Radius * sin(y_s / Radius) + u_s_n[2],
    )
)
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


def compute_axial_profile(fluid_velocity, fluid_pressure, shell_state):
    """Return slab approximations of cross-sectional pressure, flow, and diameter."""
    profile = []
    shell_displacement = as_vector(
        (shell_state[0], shell_state[1], shell_state[2])
    )
    for i, axial_coordinate in enumerate(profile_x_coordinates):
        marker = i + 1
        slab_volume = assemble(Constant(1.0) * dx_f_profile(marker))
        shell_parameter_area = assemble(Constant(1.0) * dx_s_profile(marker))
        if slab_volume <= 0.0 or shell_parameter_area <= 0.0:
            raise RuntimeError("An axial diagnostic bin contains no cells.")
        mean_pressure = assemble(fluid_pressure * dx_f_profile(marker)) / slab_volume
        flow_rate = (
            assemble(fluid_velocity[0] * dx_f_profile(marker))
            / profile_bin_width
        )
        mean_radial_displacement = (
            assemble(
                dot(shell_displacement, radial_direction)
                * dx_s_profile(marker)
            )
            / shell_parameter_area
        )
        diameter = 2.0 * (Radius + mean_radial_displacement)
        profile.append(
            (axial_coordinate, mean_pressure, flow_rate, diameter)
        )
    return profile


def write_profile(writer, time, profile):
    for axial_coordinate, mean_pressure, flow_rate, diameter in profile:
        writer.writerow(
            [time, axial_coordinate, mean_pressure, flow_rate, diameter]
        )



# ---------------------------------------------------------------------------
# Monolithic constraint reduction. All mesh/space setup above is self-contained.
# ---------------------------------------------------------------------------
from pathlib import Path
import hashlib
import json
import sys
from scipy import sparse


def csr(matrix):
    """Copy a serial DOLFIN matrix to SciPy CSR without changing DOF ordering."""
    indptr, indices, values = as_backend_type(matrix).mat().getValuesCSR()
    return sparse.csr_matrix((values.copy(), indices.copy(), indptr.copy()),
                             shape=(matrix.size(0), matrix.size(1)))


# The supplied interface is matching, including the duplicate periodic seam.
# Verify the map geometrically rather than silently applying nearest neighbors.
map_path = Path(args.mesh_dir) / 'interface_node_map.csv'
if not map_path.exists():
    raise FileNotFoundError('Strong monolithic coupling requires ' + str(map_path))
with map_path.open() as handle:
    interface_nodes = [(int(r['shell_vertex']), int(r['fluid_vertex']))
                       for r in csv.DictReader(handle)]
if {s for s, f in interface_nodes} != set(range(structure_mesh.num_vertices())):
    raise RuntimeError('The interface map must cover every shell vertex.')

velocity_space, velocity_parent = W_f.sub(0).collapse(collapsed_dofs=True)
fluid_vertex_dofs = vertex_to_dof_map(velocity_space).reshape((-1, 3))
shell_vertex_dofs = vertex_to_dof_map(V_s).reshape((-1, 6))
wall_rows = {}
for sv, fv in interface_nodes:
    if not 0 <= fv < fluid_reference_mesh.num_vertices():
        raise RuntimeError('Invalid fluid vertex in interface map.')
    x, arc = structure_mesh.coordinates()[sv]
    expected = np.array([x, Radius * math.cos(arc / Radius), Radius * math.sin(arc / Radius)])
    if np.linalg.norm(expected - fluid_reference_mesh.coordinates()[fv]) > geometry_tolerance:
        raise RuntimeError('Interface map geometry mismatch at shell vertex ' + str(sv))
    for component in range(3):
        fluid_dof = velocity_parent[int(fluid_vertex_dofs[fv, component])]
        shell_dof = int(shell_vertex_dofs[sv, component])
        if fluid_dof in wall_rows and wall_rows[fluid_dof] != shell_dof:
            raise RuntimeError('Inconsistent periodic interface DOFs.')
        wall_rows[fluid_dof] = shell_dof

# Check every wall-facet vertex and the periodic rim; no uncovered wall DOFs.
fluid_mesh.init(2, 0)
marked_wall_vertices = set()
for facet in facets(fluid_mesh):
    if fluid_boundary_markers[facet] == 3:
        marked_wall_vertices.update(int(v) for v in facet.entities(0))
required = {velocity_parent[int(fluid_vertex_dofs[v, c])]
            for v in marked_wall_vertices for c in range(3)}
if required != set(wall_rows):
    raise RuntimeError('Interface map and marked fluid wall do not agree.')

nf, ns = W_f.dim(), V_s.dim()
clamped = set(bc_s.get_boundary_values())
fluid_free = np.array([i for i in range(nf) if i not in wall_rows], dtype=np.int32)
shell_free = np.array([i for i in range(ns) if i not in clamped], dtype=np.int32)
shell_reduced = {int(d): j for j, d in enumerate(shell_free)}
rr = list(fluid_free) + list(nf + shell_free)
cc = list(range(len(fluid_free))) + list(len(fluid_free) + np.arange(len(shell_free)))
for fd, sd in wall_rows.items():
    if sd in shell_reduced:
        rr.append(fd)
        cc.append(len(fluid_free) + shell_reduced[sd])
R = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)),
                      shape=(nf + ns, len(fluid_free) + len(shell_free)))
Ms, Ks = csr(M_s), csr(A_s)
Hs = Ms / dt + dt * Ks


def coupled_solve(fluid_matrix, fluid_rhs, shell_rhs):
    """One linear solve for fluid velocity/pressure and shell velocity."""
    Af = csr(fluid_matrix)
    block = sparse.block_diag((Af, Hs), format='csr')
    rhs = np.concatenate([fluid_rhs.get_local(), shell_rhs])
    reduced = (R.T @ block @ R).tocsr()
    reduced.sort_indices()
    b = np.asarray(R.T @ rhs)
    # Use DOLFIN/PETSc MUMPS, rather than a partitioned interface iteration.
    from petsc4py import PETSc
    pmat = PETSc.Mat().createAIJ(size=reduced.shape,
        csr=(reduced.indptr.astype(PETSc.IntType),
             reduced.indices.astype(PETSc.IntType), reduced.data), comm=PETSc.COMM_SELF)
    matrix = PETScMatrix(pmat)
    right = PETScVector(MPI.comm_self)
    matrix.init_vector(right, 0)
    right.set_local(b); right.apply('insert')
    solution = right.copy(); solution.zero()
    solver = LUSolver(args.linear_solver)
    solver.solve(matrix, solution, right)
    z = solution.get_local()
    residual = reduced @ z - b
    relative = np.linalg.norm(residual) / max(np.linalg.norm(b), 1e-30)
    if not np.all(np.isfinite(z)) or relative > 1e-7:
        raise RuntimeError('Coupled linear solve failed: relative residual %.3e' % relative)
    full = np.asarray(R @ z)
    reaction = block @ full - rhs
    return full[:nf], full[nf:], relative, reaction


reference_coordinates = fluid_reference_mesh.coordinates().copy()
shell_old = Function(V_s)
shell_velocity_old = Function(V_s)
shell_iterate = Function(V_s)
fluid_iterate = Function(W_f)
snapshot_steps = {int(round(t / dt)) for t in (.002, .004, .006, .008, .010, .012)
                  if t <= T + .5 * dt}


def put(function, coefficients):
    function.vector().set_local(coefficients)
    function.vector().apply('insert')


def update_geometry(displacement):
    """Set the candidate current domain, always relative to the fixed mesh."""
    u_s_n.assign(displacement)
    if args.fixed_domain:
        mesh_displacement_total.vector().zero()
        w_f.vector().zero()
        fluid_mesh.coordinates()[:] = reference_coordinates
    else:
        # Exact matching-node trace, including the periodic seam.
        transfer = np.zeros(V_f_reference.dim())
        reference_vertex_dofs = vertex_to_dof_map(V_f_reference).reshape((-1, 3))
        shell_values = u_s_n.vector().get_local()
        for sv, fv in interface_nodes:
            transfer[reference_vertex_dofs[fv]] = shell_values[shell_vertex_dofs[sv, :3]]
        put(shell_displacement_on_reference_fluid, transfer)
        solve(a_lap == L_lap, mesh_displacement_total,
              [bc_mesh_inlet, bc_mesh_outlet, bc_mesh_wall])
        increment = (mesh_displacement_total.vector().get_local()
                     - mesh_displacement_old.vector().get_local()) / dt
        put(w_f, increment)
        dvertices = mesh_displacement_total.compute_vertex_values(fluid_reference_mesh).reshape((3, -1)).T
        fluid_mesh.coordinates()[:] = reference_coordinates + dvertices
    fluid_mesh.bounding_box_tree().build(fluid_mesh)
    ratios = signed_tetrahedron_jacobians(fluid_mesh) / reference_tetrahedron_jacobians
    if ratios.min() <= 0:
        raise RuntimeError('ALE mesh inversion during nonlinear iteration.')
    orientation = project(orientation_expression, V_structure_scalar)
    if orientation.vector().min() <= 0:
        raise RuntimeError('Shell orientation lost during nonlinear iteration.')
    return float(ratios.min())


outputs = {name: File(os.path.join(args.output_dir, name + '.pvd')) for name in
    ['fluid_velocity', 'fluid_pressure', 'shell_displacement', 'shell_rotation',
     'shell_velocity', 'ALE_total_displacement_reference', 'ALE_mesh_velocity',
     'current_shell_midsurface']}
File(os.path.join(args.output_dir, 'fluid_boundary_markers.pvd')) << fluid_boundary_markers
settings = vars(args).copy()
settings.update(scheme='monolithic backward Euler; current skew convection; ALE Picard',
                solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                command=sys.argv, fluid_dofs=nf, shell_dofs=ns, reduced_dofs=R.shape[1],
                constrained_interface_dofs=len(wall_rows),
                mesh_sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(mesh_path), Path(shell_xml_path) if os.path.exists(shell_xml_path)
                              else Path(shell_xdmf_path), map_path]})
Path(args.output_dir, 'run_settings.json').write_text(json.dumps(settings, indent=2))
headers = ['time_s', 'inlet_pressure_load_Pa', 'mean_inlet_pressure_Pa',
    'mean_outlet_pressure_Pa', 'inlet_flow_m3_per_s', 'outlet_flow_m3_per_s',
    'fluid_volume_m3', 'mass_residual_m3_per_s', 'midpoint_mean_pressure_Pa',
    'midpoint_flow_m3_per_s', 'midpoint_mean_radial_displacement_m',
    'maximum_mean_radial_displacement_m', 'fluid_kinetic_energy_J',
    'shell_kinetic_energy_J', 'shell_elastic_energy_J', 'total_mechanical_energy_J',
    'min_ALE_tetrahedron_jacobian_ratio', 'nonlinear_iterations',
    'nonlinear_relative_increment', 'linear_relative_residual',
    'interface_velocity_max_abs_defect_m_per_s', 'interface_virtual_work_defect_W',
    'ALE_wall_velocity_max_abs_defect_m_per_s', 'geometry_displacement_max_abs_defect_m',
    'viscous_dissipation_increment_J', 'pressure_stabilization_increment_J',
    'backward_Euler_dissipation_increment_J', 'inlet_work_increment_J',
    'energy_balance_defect_J']
previous_volume = assemble(Constant(1.) * dx_f)
previous_energy = 0.0
print('MONOLITHIC: fluid %d + shell %d -> %d reduced DOFs' % (nf, ns, R.shape[1]), flush=True)
with open(os.path.join(args.output_dir, 'diagnostics.csv'), 'w', newline='') as dh, \
     open(os.path.join(args.output_dir, 'axial_profiles.csv'), 'w', newline='') as ph, \
     open(os.path.join(args.output_dir, 'nonlinear_iterations.csv'), 'w', newline='') as ih:
    dw, pw, iw = csv.writer(dh), csv.writer(ph), csv.writer(ih)
    dw.writerow(headers)
    pw.writerow(['time_s', 'axial_coordinate_m', 'mean_pressure_Pa',
                 'flow_rate_m3_per_s', 'mean_diameter_m'])
    iw.writerow(['step', 'iteration', 'fluid_velocity_relative_increment',
                 'shell_velocity_relative_increment', 'linear_relative_residual'])
    for step in range(1, num_steps + 1):
        t = step * dt
        p_inlet.assign(inlet_pressure_value(t, maximum_inlet_pressure, pulse_duration))
        shell_old.assign(u_s_n)
        old_d = shell_old.vector().get_local()
        old_v = shell_velocity_old.vector().get_local()
        vs_guess = old_v.copy()
        uf_guess = up_f_old.vector().get_local().copy()
        shell_rhs = Ms @ old_v / dt - Ks @ old_d
        for iteration in range(1, args.max_iterations + 1):
            put(shell_iterate, old_d + dt * vs_guess)
            min_j = update_geometry(shell_iterate)
            put(fluid_iterate, uf_guess)
            convecting_velocity.assign(fluid_iterate.split(deepcopy=True)[0])
            af = assemble(a_f); bf = assemble(L_f)
            uf_new, vs_new, lin_res, reaction = coupled_solve(af, bf, shell_rhs)
            du = uf_new - uf_guess
            dv = vs_new - vs_guess
            # Pressure has different physical units, so assess velocity separately.
            udofs = W_f.sub(0).dofmap().dofs()
            eu = np.linalg.norm(du[udofs]) / max(np.linalg.norm(uf_new[udofs]), args.absolute_tolerance)
            ev = np.linalg.norm(dv) / max(np.linalg.norm(vs_new), args.absolute_tolerance)
            iw.writerow([step, iteration, eu, ev, lin_res]); ih.flush()
            print('  step %d iteration %d: fluid %.3e shell %.3e linear %.3e' %
                  (step, iteration, eu, ev, lin_res), flush=True)
            converged = (eu < args.relative_tolerance or np.max(np.abs(du[udofs])) < args.absolute_tolerance)
            converged = converged and (ev < args.relative_tolerance or np.max(np.abs(dv)) < args.absolute_tolerance)
            if converged and iteration >= 2:
                break
            uf_guess += args.relaxation * du
            vs_guess += args.relaxation * dv
        else:
            raise RuntimeError('Monolithic geometry/convection failed to converge at step %d; no unconverged step accepted.' % step)
        put(up_f, uf_new)
        put(v_s_completed, vs_new)
        # The assembled domain corresponds to vs_guess. At convergence its
        # displacement differs from the accepted shell state only by dt*dv.
        put(u_s_n, old_d + dt * vs_new)
        geometry_defect = dt * float(np.max(np.abs(dv)))
        u_f_sol, p_f_sol = up_f.split(deepcopy=True)
        shell_displacement = project(as_vector((u_s_n[0], u_s_n[1], u_s_n[2])), V_structure_vec)
        shell_rotation = project(as_vector((u_s_n[3], u_s_n[4], u_s_n[5])), V_structure_vec)
        shell_velocity = project(as_vector((v_s_completed[0], v_s_completed[1], v_s_completed[2])), V_structure_vec)
        profile = compute_axial_profile(u_f_sol, p_f_sol, u_s_n)
        inlet_area = assemble(Constant(1.) * ds_f(1))
        outlet_area = assemble(Constant(1.) * ds_f(2))
        qi, qo = assemble(u_f_sol[0] * ds_f(1)), assemble(u_f_sol[0] * ds_f(2))
        volume = assemble(Constant(1.) * dx_f)
        interface_defect = max(abs(uf_new[fd] - vs_new[sd]) for fd, sd in wall_rows.items())
        # Residual forces use the same constraint and its transpose. Summed
        # interface virtual work vanishes without a separate stress projection.
        virtual_work = float(sum(reaction[fd] * uf_new[fd] for fd in wall_rows)
                             + reaction[nf:] @ vs_new)
        ale_values = w_f.vector().get_local()
        vf_vertex = vertex_to_dof_map(V_f).reshape((-1, 3))
        ale_defect = max(abs(ale_values[vf_vertex[fv, c]] - vs_new[shell_vertex_dofs[sv,c]])
                         for sv,fv in interface_nodes for c in range(3)) if not args.fixed_domain else 0.
        ekf = .5 * rho_f * assemble(inner(u_f_sol,u_f_sol)*dx_f)
        eks = .5 * float(vs_new @ (Ms @ vs_new))
        elastic = .5 * float(u_s_n.vector().get_local() @ (Ks @ u_s_n.vector().get_local()))
        # On the fixed domain this is the backward-Euler energy identity.
        # On moving domains the reported defect includes geometric effects;
        # it is a diagnostic, not a moving-domain stability theorem.
        viscous = dt * 2 * mu_f * assemble(inner(sym(grad(u_f_sol)), sym(grad(u_f_sol))) * dx_f)
        pressure_diss = dt * args.pressure_stabilization * assemble(h_f**2 * inner(grad(p_f_sol),grad(p_f_sol)) * dx_f)
        delta_v = vs_new-old_v
        delta_d = dt*vs_new
        be_diss = .5*rho_f*assemble(inner(u_f_sol-u_f_old,u_f_sol-u_f_old)*dx_f)
        be_diss += .5*float(delta_v @ (Ms @ delta_v)) + .5*float(delta_d @ (Ks @ delta_d))
        inlet_work = dt*float(p_inlet)*qi
        total_energy = ekf+eks+elastic
        energy_defect = total_energy-previous_energy+viscous+pressure_diss+be_diss-inlet_work
        mid = profile[midpoint_bin]
        row = [t, float(p_inlet), assemble(p_f_sol*ds_f(1))/inlet_area,
               assemble(p_f_sol*ds_f(2))/outlet_area, qi, qo, volume,
               abs((volume-previous_volume)/dt-(qi-qo)), mid[1], mid[2],
               .5*mid[3]-Radius, max(.5*r[3]-Radius for r in profile), ekf, eks,
               elastic, ekf+eks+elastic, min_j, iteration, max(eu,ev), lin_res,
               interface_defect, virtual_work, ale_defect, geometry_defect, viscous, pressure_diss,
               be_diss, inlet_work, energy_defect]
        if not np.all(np.isfinite(row)):
            raise RuntimeError('Nonfinite monolithic diagnostics.')
        dw.writerow(row); dh.flush()
        if step in snapshot_steps:
            write_profile(pw, t, profile); ph.flush()
        if step % args.output_every == 0 or step == num_steps or step in snapshot_steps:
            fields = dict(fluid_velocity=u_f_sol, fluid_pressure=p_f_sol,
                shell_displacement=shell_displacement, shell_rotation=shell_rotation,
                shell_velocity=shell_velocity, ALE_total_displacement_reference=mesh_displacement_total,
                ALE_mesh_velocity=w_f,
                current_shell_midsurface=project(phi0+shell_displacement,V_phi))
            for name, field in fields.items():
                field.rename(name,name); outputs[name] << (field,t)
        up_f_old.assign(up_f)
        shell_velocity_old.assign(v_s_completed)
        mesh_displacement_old.assign(mesh_displacement_total)
        previous_volume = volume
        previous_energy = total_energy
        print('Step %d/%d: t=%.6e, midpoint p=%.6e Pa, radial=%.6e m, iterations=%d' %
              (step,num_steps,t,mid[1],.5*mid[3]-Radius,iteration), flush=True)
print('Simulation complete. Results are in '+args.output_dir, flush=True)
