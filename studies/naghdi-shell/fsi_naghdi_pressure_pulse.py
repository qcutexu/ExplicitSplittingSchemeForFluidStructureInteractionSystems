"""Three-dimensional fluid--Naghdi-shell pressure-pulse benchmark.

a straight-tube pressure-pulse test considered in our Naghdi-shell-FSI manuscript.
Its dimensions, material parameters, and raised-cosine load are based on the
hemodynamics benchmark used in the kinematically coupled beta-scheme
literature, while the wall here is the manuscript's full two-dimensional
Naghdi shell and the fluid is three-dimensional.  The implementation uses
legacy FEniCS (dolfin), an ALE mesh update, and the structure--ALE--fluid
ordering of the full-traction beta split adopted in the manuscript.

Important time-level convention
-------------------------------
The structure stage advances the shell displacement with the stage velocity.
The subsequent fluid solve produces the completed interface velocity.  The
translational part of that completed velocity, together with the rotational
velocity from the structure stage, is carried into the next structure step.
The shell displacement is not corrected after the fluid solve.
"""

from dolfin import *
import argparse
import csv
import math
import os
import numpy as np
from ufl import max_value, transpose


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
parser.add_argument("--output-dir", default="results_naghdi_original", help="Directory for PVD and CSV output.")
parser.add_argument("--dt", type=float, default=1.0e-4, help="Time step [s].")
parser.add_argument("--final-time", type=float, default=2.4e-2, help="Final time [s].")
parser.add_argument("--beta", type=float, default=1.0, help="FSI splitting parameter in [0,1].")
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
if not 0.0 <= args.beta <= 1.0:
    raise ValueError("The splitting parameter beta must lie in [0,1].")
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
maximum_inlet_pressure = 2e3
pulse_duration = 5.0e-3
beta_split = args.beta

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
    """Mark every exterior facet, including wall triangles at the end rings."""
    facet_dim = mesh.topology().dim() - 1
    mesh.init(facet_dim, mesh.topology().dim())
    markers = MeshFunction("size_t", mesh, facet_dim, 0)
    for facet in facets(mesh):
        if not facet.exterior():
            continue
        x = facet.midpoint().x()
        if near(x, fluid_x_min, geometry_tolerance):
            markers[facet] = 1
        elif near(x, fluid_x_max, geometry_tolerance):
            markers[facet] = 2
        else:
            markers[facet] = 3
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

# Current-domain coefficients used in the ALE and Robin terms.
w_f = Function(V_f, name="ALE_mesh_velocity")
v_s_stage_on_fluid = Function(V_f, name="structure_stage_velocity_on_fluid")
traction_ref_old_on_fluid = Function(V_f, name="previous_pulled_back_fluid_traction")
J_s_current_on_fluid = Function(Q_f, name="current_surface_jacobian")
wall_normal_on_fluid = Function(V_f, name="current_outward_wall_normal")

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
v_s_stage = Function(V_s, name="shell_stage_velocity")
v_s_completed = Function(V_s, name="shell_completed_velocity")
traction_ref_old = Function(V_structure_vec, name="previous_pulled_back_fluid_traction")
shell_load_ref_old = Function(V_structure_vec, name="previous_pulled_back_shell_load")
J_s_current = Function(V_structure_scalar, name="current_surface_jacobian")

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

safe_J_s_on_fluid = max_value(J_s_current_on_fluid, Constant(1.0e-8))
F_time = (rho_f / dt) * inner(u_f - u_f_old, v_f) * dx_f
#F_adv = rho_f * dot( dot(u_f_old - w_f, nabla_grad(u_f)), v_f) * dx_f

F_adv = 0.5 * rho_f * (
    dot( dot(u_f_old - w_f, nabla_grad(u_f)), v_f)  
     -
    dot( dot(u_f_old - w_f, nabla_grad(v_f)), u_f )
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

# Both terms are written on Gamma^{n+1}.  Division by J_s^{n+1} converts
# current-surface integration back to the fixed parameter-area measure used in
# the manuscript.  traction_ref_old_on_fluid stores
# (t_f^n o Phi_s^n) J_s^n.  Its minus sign in the residual places the known
# beta-traction contribution on the right-hand side of the fluid equation.
F_robin_mass = (
    (rho_s * thickness / dt)
    * inner(u_f - v_s_stage_on_fluid, v_f)
    / safe_J_s_on_fluid
    * ds_f(3)
)
F_robin_previous_traction = (
    -beta_split
    * inner(traction_ref_old_on_fluid, v_f)
    / safe_J_s_on_fluid
    * ds_f(3)
)

fluid_residual = (
    F_time
    + F_adv
    + F_div
    + F_p
    + F_diss
    + F_stab
    + F_robin_mass
    + F_robin_previous_traction
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
L_structure = (
    beta_split
    * inner(shell_load_ref_old, structure_test_displacement)
    * dx_s
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


# Time-series output.  Each File object is retained so that one PVD collection
# contains all time levels rather than creating one unrelated PVD per step.
fluid_velocity_file = File(os.path.join(args.output_dir, "fluid_velocity.pvd"))
fluid_pressure_file = File(os.path.join(args.output_dir, "fluid_pressure.pvd"))
fluid_traction_file = File(os.path.join(args.output_dir, "fluid_traction.pvd"))
mesh_velocity_file = File(os.path.join(args.output_dir, "ALE_mesh_velocity.pvd"))
mesh_displacement_file = File(
    os.path.join(args.output_dir, "ALE_total_displacement_reference.pvd")
)
shell_displacement_file = File(
    os.path.join(args.output_dir, "shell_displacement.pvd")
)
shell_rotation_file = File(os.path.join(args.output_dir, "shell_rotation.pvd"))
shell_stage_velocity_file = File(
    os.path.join(args.output_dir, "shell_stage_velocity.pvd")
)
shell_completed_velocity_file = File(
    os.path.join(args.output_dir, "shell_completed_velocity.pvd")
)
current_midsurface_file = File(
    os.path.join(args.output_dir, "current_shell_midsurface.pvd")
)
File(os.path.join(args.output_dir, "fluid_boundary_markers.pvd")) << fluid_boundary_markers

diagnostics_handle = open(
    os.path.join(args.output_dir, "diagnostics.csv"), "w", newline=""
)
diagnostics_writer = csv.writer(diagnostics_handle)
diagnostics_writer.writerow(
    [
        "time_s",
        "inlet_pressure_load_Pa",
        "mean_inlet_pressure_Pa",
        "mean_outlet_pressure_Pa",
        "inlet_flow_m3_per_s",
        "outlet_flow_m3_per_s",
        "fluid_volume_m3",
        "mass_residual_m3_per_s",
        "midpoint_mean_pressure_Pa",
        "midpoint_flow_m3_per_s",
        "midpoint_mean_radial_displacement_m",
        "maximum_mean_radial_displacement_m",
        "fluid_kinetic_energy_J",
        "shell_kinetic_energy_J",
        "shell_elastic_energy_J",
        "total_mechanical_energy_J",
        "viscous_dissipation_increment_J",
        "inlet_work_increment_J",
        "min_surface_jacobian",
        "max_surface_jacobian",
        "min_ALE_tetrahedron_jacobian_ratio",
        "ALE_wall_velocity_error_L2",
        "fluid_stage_velocity_increment_L2",
    ]
)

profiles_handle = open(
    os.path.join(args.output_dir, "axial_profiles.csv"), "w", newline=""
)
profiles_writer = csv.writer(profiles_handle)
profiles_writer.writerow(
    [
        "time_s",
        "axial_coordinate_m",
        "mean_pressure_Pa",
        "flow_rate_m3_per_s",
        "mean_diameter_m",
    ]
)

snapshot_times = [
    value
    for value in (0.002, 0.004, 0.006, 0.008, 0.010, 0.012)
    if value <= T + 0.5 * dt
]
snapshot_steps = {int(round(value / dt)) for value in snapshot_times}

# Initial data are zero.  The initial pulled-back traction and shell load are
# therefore also zero.  A value of one is required for J_s on the first fluid
# assembly, and it is replaced by the projected current value before that solve.
J_s_current_on_fluid.interpolate(Constant(1.0))
initial_volume = assemble(Constant(1.0) * dx_f)
previous_volume = initial_volume
initial_inlet_area = assemble(Constant(1.0) * ds_f(1))
initial_outlet_area = assemble(Constant(1.0) * ds_f(2))
initial_wall_area = assemble(Constant(1.0) * ds_f(3))
reference_fluid_coordinate = SpatialCoordinate(fluid_mesh)
reference_radial_direction_on_fluid = as_vector(
    (
        0.0,
        reference_fluid_coordinate[1] / Radius,
        reference_fluid_coordinate[2] / Radius,
    )
)
mean_geometric_inlet_normal_x = (
    assemble(geometric_n_f[0] * ds_f(1)) / initial_inlet_area
)
mean_geometric_outlet_normal_x = (
    assemble(geometric_n_f[0] * ds_f(2)) / initial_outlet_area
)
mean_geometric_wall_normal_alignment = assemble(
    dot(geometric_n_f, reference_radial_direction_on_fluid) * ds_f(3)
) / initial_wall_area

print("Starting pressure-pulse benchmark")
print("  dt             = {:.6e} s".format(dt))
print("  final time     = {:.6e} s".format(T))
print("  time steps     = {}".format(num_steps))
print("  beta           = {:.3f}".format(beta_split))
print("  fluid cells    = {}".format(fluid_mesh.num_cells()))
print("  shell cells    = {}".format(structure_mesh.num_cells()))
print(
    "  mesh normals   = inlet {:.3f}, outlet {:.3f}, wall alignment {:.3f}".format(
        mean_geometric_inlet_normal_x,
        mean_geometric_outlet_normal_x,
        mean_geometric_wall_normal_alignment,
    )
)
if (
    mean_geometric_inlet_normal_x > -0.5
    or mean_geometric_outlet_normal_x < 0.5
    or mean_geometric_wall_normal_alignment < 0.5
):
    print(
        "WARNING: imported-mesh FacetNormal orientation is nonstandard; "
        "the benchmark uses explicit end and shell normals for its loads."
    )

t = t0
for n in range(num_steps):
    step = n + 1
    t = step * dt

    # ------------------------------------------------------------------
    # Step 1: structure stage
    # ------------------------------------------------------------------
    # (M_s + dt^2 A_s) v_s^{n+1,s}
    #   = M_s v_s^n + dt beta f_s^{ref,n} - dt A_s xi_s^n.
    structure_matrix = M_s.copy()
    structure_matrix.axpy(dt**2, A_s, True)

    structure_rhs = M_s * v_s_completed.vector()
    elastic_old = A_s * u_s_n.vector()
    elastic_old *= -dt
    structure_rhs.axpy(1.0, elastic_old)

    structure_load = assemble(L_structure)
    structure_load *= dt
    structure_rhs.axpy(1.0, structure_load)

    bc_s.apply(structure_matrix, structure_rhs)
    solve(structure_matrix, v_s_stage.vector(), structure_rhs)
    bc_s.apply(v_s_stage.vector())

    u_s_n.vector().axpy(dt, v_s_stage.vector())
    bc_s.apply(u_s_n.vector())

    shell_displacement = project(
        as_vector((u_s_n[0], u_s_n[1], u_s_n[2])),
        V_structure_vec,
    )
    shell_stage_translation = project(
        as_vector((v_s_stage[0], v_s_stage[1], v_s_stage[2])),
        V_structure_vec,
    )

    J_s_current.assign(project(J_s_expression, V_structure_scalar))
    orientation_current.assign(
        project(orientation_expression, V_structure_scalar)
    )
    minimum_orientation = orientation_current.vector().min()
    if minimum_orientation <= 0.0:
        diagnostics_handle.close()
        profiles_handle.close()
        raise RuntimeError(
            "The shell map lost its reference orientation at step {}.".format(step)
        )
    fluid_outward_normal_shell.assign(
        project(
            -current_cross_product / J_s_expression,
            V_structure_vec,
        )
    )

    # ------------------------------------------------------------------
    # Step 2: harmonic ALE update on the fixed reference mesh
    # ------------------------------------------------------------------
    shell_displacement_on_reference_fluid.assign(
        shell_vector_to_reference_fluid(shell_displacement)
    )
    solve(
        a_lap == L_lap,
        mesh_displacement_total,
        [bc_mesh_inlet, bc_mesh_outlet, bc_mesh_wall],
    )

    mesh_displacement_increment_reference.vector().zero()
    mesh_displacement_increment_reference.vector().axpy(
        1.0, mesh_displacement_total.vector()
    )
    mesh_displacement_increment_reference.vector().axpy(
        -1.0, mesh_displacement_old.vector()
    )
    mesh_displacement_increment_reference.vector().apply("insert")
    copy_matching_function(
        mesh_displacement_increment_reference,
        mesh_displacement_increment_current,
    )
    w_f.vector().set_local(
        mesh_displacement_increment_current.vector().get_local() / dt
    )
    w_f.vector().apply("insert")

    # Transfer all reference-coordinate Robin data before moving the matching
    # current mesh.  Their coefficients then follow the ALE nodes.
    stage_velocity_reference = shell_vector_to_reference_fluid(
        shell_stage_translation
    )
    traction_reference = shell_vector_to_reference_fluid(traction_ref_old)
    J_reference = shell_scalar_to_reference_fluid(J_s_current)
    wall_normal_reference = shell_vector_to_reference_fluid(
        fluid_outward_normal_shell
    )
    copy_matching_function(stage_velocity_reference, v_s_stage_on_fluid)
    copy_matching_function(traction_reference, traction_ref_old_on_fluid)
    copy_matching_function(J_reference, J_s_current_on_fluid)
    copy_matching_function(wall_normal_reference, wall_normal_on_fluid)

    ALE.move(fluid_mesh, mesh_displacement_increment_current)
    fluid_mesh.bounding_box_tree().build(fluid_mesh)
    mesh_displacement_old.assign(mesh_displacement_total)

    current_tetrahedron_jacobians = signed_tetrahedron_jacobians(fluid_mesh)
    ALE_jacobian_ratios = (
        current_tetrahedron_jacobians / reference_tetrahedron_jacobians
    )
    minimum_ALE_jacobian_ratio = float(ALE_jacobian_ratios.min())
    if minimum_ALE_jacobian_ratio <= 0.0:
        diagnostics_handle.close()
        profiles_handle.close()
        raise RuntimeError(
            "The ALE map inverted a fluid tetrahedron at step {}.".format(step)
        )

    ALE_wall_velocity_error = math.sqrt(
        max(
            0.0,
            assemble(
                inner(w_f - v_s_stage_on_fluid, w_f - v_s_stage_on_fluid)
                * ds_f(3)
            ),
        )
    )

    # ------------------------------------------------------------------
    # Step 3: semi-implicit fluid solve with the Robin interface condition
    # ------------------------------------------------------------------
    prescribed_inlet_pressure = inlet_pressure_value(
        t, maximum_inlet_pressure, pulse_duration
    )
    p_inlet.assign(prescribed_inlet_pressure)

    fluid_matrix = assemble(a_f)
    fluid_rhs = assemble(L_f)
    solve(fluid_matrix, up_f.vector(), fluid_rhs)
    u_f_sol, p_f_sol = up_f.split(deepcopy=True)
    u_f_sol.rename("fluid_velocity", "fluid_velocity")
    p_f_sol.rename("fluid_pressure", "fluid_pressure")

    # The completed translational interface velocity is the current fluid
    # trace.  It is stored for the next structure step without changing the
    # already-updated shell displacement.
    fluid_velocity_on_shell = current_fluid_vector_to_shell(u_f_sol)
    completed_velocity_expression = as_vector(
        (
            fluid_velocity_on_shell[0],
            fluid_velocity_on_shell[1],
            fluid_velocity_on_shell[2],
            v_s_stage[3],
            v_s_stage[4],
            v_s_stage[5],
        )
    )
    v_s_completed.assign(project(completed_velocity_expression, V_s))
    bc_s.apply(v_s_completed.vector())

    # Current fluid traction t_f^{n+1}=sigma_f^{n+1} n_f^{n+1}.
    traction_expression = dot(
        2.0 * mu_f * sym(grad(u_f_sol))
        - p_f_sol * Identity(3),
        wall_normal_on_fluid,
    )
    traction_trial = TrialFunction(V_f)
    traction_test = TestFunction(V_f)
    projection_regularization = Constant(1.0e-12 / Radius)
    traction_projection_left = (
        inner(traction_trial, traction_test) * ds_f(3)
        + projection_regularization
        * inner(traction_trial, traction_test)
        * dx_f
    )
    traction_projection_right = inner(
        traction_expression, traction_test
    ) * ds_f(3)
    traction_current = Function(V_f, name="fluid_traction")
    solve(
        traction_projection_left == traction_projection_right,
        traction_current,
    )

    traction_current_on_shell = current_fluid_vector_to_shell(
        traction_current
    )
    traction_ref_old.assign(
        project(
            traction_current_on_shell * J_s_expression,
            V_structure_vec,
        )
    )
    shell_load_ref_old.vector().set_local(
        -traction_ref_old.vector().get_local()
    )
    shell_load_ref_old.vector().apply("insert")

    # ------------------------------------------------------------------
    # Diagnostics and output
    # ------------------------------------------------------------------
    inlet_area = assemble(Constant(1.0) * ds_f(1))
    outlet_area = assemble(Constant(1.0) * ds_f(2))
    mean_inlet_pressure = assemble(p_f_sol * ds_f(1)) / inlet_area
    mean_outlet_pressure = assemble(p_f_sol * ds_f(2)) / outlet_area
    # For the straight benchmark tube, positive flow is in the +x direction.
    # These formulas equal -int_{Gamma_in} u.n and int_{Gamma_out} u.n when
    # the geometric normals have their standard outward orientation.
    inlet_flow = assemble(u_f_sol[0] * ds_f(1))
    outlet_flow = assemble(u_f_sol[0] * ds_f(2))
    current_volume = assemble(Constant(1.0) * dx_f)
    mass_residual = abs(
        (current_volume - previous_volume) / dt
        - (inlet_flow - outlet_flow)
    )

    midpoint_marker = midpoint_bin + 1
    midpoint_volume = assemble(
        Constant(1.0) * dx_f_profile(midpoint_marker)
    )
    midpoint_shell_area = assemble(
        Constant(1.0) * dx_s_profile(midpoint_marker)
    )
    midpoint_mean_pressure = (
        assemble(p_f_sol * dx_f_profile(midpoint_marker))
        / midpoint_volume
    )
    midpoint_flow = (
        assemble(u_f_sol[0] * dx_f_profile(midpoint_marker))
        / profile_bin_width
    )
    midpoint_mean_radial_displacement = (
        assemble(
            dot(shell_displacement, radial_direction)
            * dx_s_profile(midpoint_marker)
        )
        / midpoint_shell_area
    )
    sampled_profile = compute_axial_profile(u_f_sol, p_f_sol, u_s_n)
    maximum_mean_radial_displacement = max(
        0.5 * diameter - Radius
        for _, _, _, diameter in sampled_profile
    )
    fluid_kinetic_energy = 0.5 * rho_f * assemble(
        inner(u_f_sol, u_f_sol) * dx_f
    )
    shell_kinetic_energy = 0.5 * v_s_completed.vector().inner(
        M_s * v_s_completed.vector()
    )
    shell_elastic_energy = 0.5 * u_s_n.vector().inner(
        A_s * u_s_n.vector()
    )
    viscous_dissipation_increment = dt * 2.0 * mu_f * assemble(
        inner(sym(grad(u_f_sol)), sym(grad(u_f_sol))) * dx_f
    )
    total_mechanical_energy = (
        fluid_kinetic_energy
        + shell_kinetic_energy
        + shell_elastic_energy
    )
    inlet_work_increment = dt * prescribed_inlet_pressure * inlet_flow
    fluid_stage_velocity_increment = math.sqrt(
        max(
            0.0,
            assemble(
                inner(
                    u_f_sol - v_s_stage_on_fluid,
                    u_f_sol - v_s_stage_on_fluid,
                )
                * ds_f(3)
            ),
        )
    )

    minimum_J_s = J_s_current.vector().min()
    maximum_J_s = J_s_current.vector().max()
    diagnostics_writer.writerow(
        [
            t,
            prescribed_inlet_pressure,
            mean_inlet_pressure,
            mean_outlet_pressure,
            inlet_flow,
            outlet_flow,
            current_volume,
            mass_residual,
            midpoint_mean_pressure,
            midpoint_flow,
            midpoint_mean_radial_displacement,
            maximum_mean_radial_displacement,
            fluid_kinetic_energy,
            shell_kinetic_energy,
            shell_elastic_energy,
            total_mechanical_energy,
            viscous_dissipation_increment,
            inlet_work_increment,
            minimum_J_s,
            maximum_J_s,
            minimum_ALE_jacobian_ratio,
            ALE_wall_velocity_error,
            fluid_stage_velocity_increment,
        ]
    )
    diagnostics_handle.flush()

    if step in snapshot_steps:
        write_profile(
            profiles_writer,
            t,
            sampled_profile,
        )
        profiles_handle.flush()

    shell_rotation = project(
        as_vector((u_s_n[3], u_s_n[4], u_s_n[5])),
        V_structure_vec,
    )
    shell_stage_translation.rename(
        "shell_stage_velocity", "shell_stage_velocity"
    )
    completed_translation = project(
        as_vector(
            (v_s_completed[0], v_s_completed[1], v_s_completed[2])
        ),
        V_structure_vec,
    )
    completed_translation.rename(
        "shell_completed_velocity", "shell_completed_velocity"
    )
    shell_displacement.rename("shell_displacement", "shell_displacement")
    shell_rotation.rename("shell_rotation", "shell_rotation")
    current_midsurface = project(
        phi0 + shell_displacement, V_phi
    )
    current_midsurface.rename(
        "current_shell_midsurface", "current_shell_midsurface"
    )

    fluid_velocity_file << (u_f_sol, t)
    fluid_pressure_file << (p_f_sol, t)
    fluid_traction_file << (traction_current, t)
    mesh_velocity_file << (w_f, t)
    mesh_displacement_file << (mesh_displacement_total, t)
    shell_displacement_file << (shell_displacement, t)
    shell_rotation_file << (shell_rotation, t)
    shell_stage_velocity_file << (shell_stage_translation, t)
    shell_completed_velocity_file << (completed_translation, t)
    current_midsurface_file << (current_midsurface, t)

    up_f_old.assign(up_f)
    previous_volume = current_volume

    print(
        "Step {}/{}: t={:.6e} s, p_in={:.6e} Pa, "
        "Q_in={:.6e} m^3/s, Q_out={:.6e} m^3/s, mass residual={:.3e}".format(
            step,
            num_steps,
            t,
            prescribed_inlet_pressure,
            inlet_flow,
            outlet_flow,
            mass_residual,
        )
    )

diagnostics_handle.close()
profiles_handle.close()
print("Simulation complete. Results are in {}.".format(args.output_dir))

