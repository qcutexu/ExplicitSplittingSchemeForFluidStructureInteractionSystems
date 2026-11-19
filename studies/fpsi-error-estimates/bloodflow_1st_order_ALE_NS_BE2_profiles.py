from fenics import *
import csv
from pathlib import Path
import numpy as np

"""
Half-domain idealized artery benchmark with:
  - physical parameters and boundary data from Kunštek--Bukač--Muha,
    Applied Mathematics and Computation 487 (2025), Example 3
  - harmonic ALE displacement solved on an unchanged reference fluid mesh
  - Navier--Stokes solved on a separate, physically moving fluid mesh
  - interface traces paired by their initial reference coordinates
  - first-order backward Euler at every time step
  - explicit, one-step-lagged interface traces and ALE displacement
  - in-situ centerline/interface profiles and pressure-wave figures

Geometry:
    Fluid:  Omega_f = (0,L) x (0,R)
    Biot:   Omega_p = (0,L) x (R,R+r_p)
"""

parameters["form_compiler"]["quadrature_degree"] = 5


# ==========================================================
# Physical / benchmark parameters
# ==========================================================
R   = 0.5
L   = 6.0
r_p = 0.1
T   = 0.014
DEFAULT_DT = 5.0e-6

rho_f    = 1.0
mu_f     = 0.035

rho_b    = 1.1
mu_b     = 1.6779e6
lambda_b = 8.2215e7
alpha    = 1.0
C0       = 1.0e-3
K_perm   = 1.0e-8

C_BJS    = 1.0e3
# Robin splitting parameters of this method; the benchmark paper does not
# prescribe these because it uses a different coupling algorithm.
L_1      = 500.0
L_2      = 500.0
L_3      = 500.0
beta_s   = 5.0e7

P_max    = 1.333e4
T_max    = 0.003

# These times lie on the time grid for the default dt=5e-6 s.
SNAPSHOT_TIMES = (0.007, 0.0105, 0.014)
PROFILE_DIR = "artery_wave_profiles"

# The ALE Navier--Stokes convection uses the velocity from the preceding step.
# Switch it off to recover the Stokes-like fluid model used in the convergence tests.
INCLUDE_NS_CONVECTION = False



# ==========================================================
# Helpers
# ==========================================================
def epsilon(u):
    return sym(nabla_grad(u))


class InletPressure(UserExpression):
    def __init__(self, **kwargs):
        self.t = 0.0
        super().__init__(**kwargs)

    def eval(self, values, x):
        if self.t <= T_max:
            values[0] = 0.5 * P_max * (1.0 - np.cos(2.0 * np.pi * self.t / T_max))
        else:
            values[0] = 0.0

    def value_shape(self):
        return ()


def build_meshes(nx):
    ny_f = 12
    ny_b = 5
    mesh_f = RectangleMesh(Point(0.0, 0.0), Point(L, R), nx, ny_f)
    mesh_b = RectangleMesh(Point(0.0, R), Point(L, R + r_p), nx, ny_b)
    return mesh_f, mesh_b


def mark_boundaries(mesh_f, mesh_b):
    # Fluid labels:
    # 1 = symmetry y=0
    # 2 = outlet   x=L
    # 3 = interface y=R
    # 4 = inlet    x=0
    class Symmetry_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)

    class Outlet_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L)

    class Interface_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], R)

    class Inlet_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    bmf_f = MeshFunction("size_t", mesh_f, mesh_f.topology().dim() - 1, 0)
    Symmetry_F().mark(bmf_f, 1)
    Outlet_F().mark(bmf_f, 2)
    Interface_F().mark(bmf_f, 3)
    Inlet_F().mark(bmf_f, 4)

    # Biot labels:
    # 1 = external wall y=R+r_p
    # 2 = outlet        x=L
    # 3 = interface     y=R
    # 4 = inlet         x=0
    class External_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], R + r_p)

    class Outlet_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L)

    class Interface_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], R)

    class Inlet_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    bmf_b = MeshFunction("size_t", mesh_b, mesh_b.topology().dim() - 1, 0)
    External_B().mark(bmf_b, 1)
    Outlet_B().mark(bmf_b, 2)
    Interface_B().mark(bmf_b, 3)
    Inlet_B().mark(bmf_b, 4)

    ds_f = Measure("ds", domain=mesh_f, subdomain_data=bmf_f)
    ds_b = Measure("ds", domain=mesh_b, subdomain_data=bmf_b)
    return bmf_f, bmf_b, ds_f, ds_b


# ==========================================================
# Interface transfer utilities
# ==========================================================
def coord_key(x, ndigits=12):
    return tuple(np.round(x, ndigits))


class ScalarInterfaceTransfer:
    """
    Match scalar interface dofs once on the initial meshes. The stored dof
    indices remain valid when the physical fluid mesh moves in place.
    """
    def __init__(self, source_space, target_space, y_interface, tol=1e-12, ndigits=12):
        self.source_space = source_space
        self.target_space = target_space
        self.y_interface = y_interface
        self.tol = tol
        self.ndigits = ndigits

        src_coords = source_space.tabulate_dof_coordinates().reshape(
            (-1, source_space.mesh().geometry().dim())
        )
        tgt_coords = target_space.tabulate_dof_coordinates().reshape(
            (-1, target_space.mesh().geometry().dim())
        )

        self.src_if = np.where(np.abs(src_coords[:, 1] - y_interface) < tol)[0]
        self.tgt_if = np.where(np.abs(tgt_coords[:, 1] - y_interface) < tol)[0]

        if len(self.src_if) == 0:
            raise RuntimeError("No source scalar interface dofs found.")
        if len(self.tgt_if) == 0:
            raise RuntimeError("No target scalar interface dofs found.")

        src_map = {}
        for i in self.src_if:
            src_map[coord_key(src_coords[i], ndigits)] = i

        src_match = []
        tgt_match = []
        missing = []
        for j in self.tgt_if:
            key = coord_key(tgt_coords[j], ndigits)
            if key in src_map:
                src_match.append(src_map[key])
                tgt_match.append(j)
            else:
                missing.append(tuple(tgt_coords[j]))

        if missing:
            raise RuntimeError(
                "Exact scalar interface matching failed. "
                f"Unmatched target coordinates: {missing[:10]}"
            )

        self.src_match = np.array(src_match, dtype=int)
        self.tgt_match = np.array(tgt_match, dtype=int)

    def transfer(self, src_func, tgt_func, zero_elsewhere=True):
        src = src_func.vector().get_local()
        tgt = tgt_func.vector().get_local()
        if zero_elsewhere:
            tgt[:] = 0.0
        tgt[self.tgt_match] = src[self.src_match]
        tgt_func.vector().set_local(tgt)
        tgt_func.vector().apply("insert")


class VectorInterfaceTransfer:
    """
    Match vector dofs componentwise using the actual subspace dofmaps.
    The map can cover the interface or every target dof (for mesh motion).
    Construct it before moving the physical fluid mesh.
    """
    def __init__(self, source_space, target_space, y_interface=None,
                 tol=1e-12, ndigits=12):
        self.source_space = source_space
        self.target_space = target_space
        gdim = source_space.mesh().geometry().dim()
        if target_space.mesh().geometry().dim() != gdim:
            raise ValueError("Source and target must have the same geometric dimension.")
        src_coords_all = source_space.tabulate_dof_coordinates().reshape((-1, gdim))
        tgt_coords_all = target_space.tabulate_dof_coordinates().reshape((-1, gdim))
        src_match, tgt_match = [], []
        for component in range(gdim):
            src_dofs = source_space.sub(component).dofmap().dofs()
            tgt_dofs = target_space.sub(component).dofmap().dofs()
            src_map = {}
            for i in src_dofs:
                if y_interface is None or abs(src_coords_all[i, 1] - y_interface) < tol:
                    key = coord_key(src_coords_all[i], ndigits)
                    if key in src_map:
                        raise RuntimeError("Duplicate source vector dof coordinate.")
                    src_map[key] = i
            missing = []
            selected = 0
            for j in tgt_dofs:
                if y_interface is not None and abs(tgt_coords_all[j, 1] - y_interface) >= tol:
                    continue
                selected += 1
                key = coord_key(tgt_coords_all[j], ndigits)
                if key in src_map:
                    src_match.append(src_map[key])
                    tgt_match.append(j)
                else:
                    missing.append(tuple(tgt_coords_all[j]))
            if not src_map or not selected or missing:
                raise RuntimeError(
                    f"Vector coordinate matching failed for component {component}: "
                    f"{missing[:10]}"
                )

        self.src_match = np.asarray(src_match, dtype=int)
        self.tgt_match = np.asarray(tgt_match, dtype=int)

    def transfer(self, src_func, tgt_func, zero_elsewhere=True):
        src = src_func.vector().get_local()
        tgt = tgt_func.vector().get_local()
        if zero_elsewhere:
            tgt[:] = 0.0
        tgt[self.tgt_match] = src[self.src_match]
        tgt_func.vector().set_local(tgt)
        tgt_func.vector().apply("insert")


def signed_triangle_areas(mesh):
    """Oriented areas detect inverted linear triangles after ALE.move."""
    coords = mesh.coordinates()
    cells = mesh.cells()
    edge_1 = coords[cells[:, 1]] - coords[cells[:, 0]]
    edge_2 = coords[cells[:, 2]] - coords[cells[:, 0]]
    return edge_1[:, 0]*edge_2[:, 1] - edge_1[:, 1]*edge_2[:, 0]


def print_norms(u, p, eta, ksi, phi, step, t):
    print(
        f"step {step:4d}, t={t:.6f} | "
        f"||u||={norm(u.vector(), 'l2'):.6e}, "
        f"||p||={norm(p.vector(), 'l2'):.6e}, "
        f"||eta||={norm(eta.vector(), 'l2'):.6e}, "
        f"||ksi||={norm(ksi.vector(), 'l2'):.6e}, "
        f"||phi||={norm(phi.vector(), 'l2'):.6e}"
    )


def boundary_dofs(space, y_reference, expected_x, component=None):
    """Freeze material-boundary DOF indices before the physical mesh moves."""
    coords = space.tabulate_dof_coordinates().reshape((-1, 2))
    if component is None:
        candidates = np.arange(len(coords))
    else:
        candidates = np.asarray(space.sub(component).dofmap().dofs(), dtype=int)
    on_boundary = candidates[np.isclose(coords[candidates, 1], y_reference,
                                        atol=1e-10, rtol=0.0)]
    order = np.argsort(coords[on_boundary, 0])
    dofs = on_boundary[order]
    x = coords[dofs, 0]
    if len(x) != len(expected_x) or not np.allclose(
            x, expected_x, atol=1e-10, rtol=0.0):
        raise RuntimeError(
            f"Unexpected trace DOFs at reference y={y_reference}: "
            f"found {len(x)}, expected {len(expected_x)}."
        )
    return dofs


class WaveProfiles:
    """Record every step on material x nodes, then plot in physical x."""

    def __init__(self, nx, steps, dt_value, u_space, p_space, ksi_space,
                 motion_space, output_dir=PROFILE_DIR,
                 snapshot_times=SNAPSHOT_TIMES):
        self.directory = Path(output_dir)
        self.dt = dt_value
        self.steps = steps
        self.time = dt_value*np.arange(steps + 1)
        self.x_ref = np.linspace(0.0, L, 2*nx + 1)  # P2 trace nodes
        x_p1 = np.linspace(0.0, L, nx + 1)
        self.x_p1 = x_p1

        # The DOF ordering of each collapsed solution space remains fixed as
        # the fluid mesh moves. Construct all boundary maps at t=0.
        self.u_dofs = {
            edge: [boundary_dofs(u_space, y, self.x_ref, k)
                   for k in range(2)]
            for edge, y in (("interface", R), ("centerline", 0.0))
        }
        self.p_dofs = {
            edge: boundary_dofs(p_space, y, x_p1)
            for edge, y in (("interface", R), ("centerline", 0.0))
        }
        self.motion_dofs = {
            edge: [boundary_dofs(motion_space, y, x_p1, k)
                   for k in range(2)]
            for edge, y in (("interface", R), ("centerline", 0.0))
        }
        self.ksi_dofs = [boundary_dofs(ksi_space, R, self.x_ref, k)
                         for k in range(2)]

        field_names = (
            "p_interface", "p_centerline",
            "u_x_interface", "u_y_interface",
            "u_x_centerline", "u_y_centerline",
            "xi_x_interface", "xi_y_interface",
            "x_interface", "y_interface",
            "x_centerline", "y_centerline",
        )
        self.data = {name: np.empty((steps + 1, len(self.x_ref)))
                     for name in field_names}

        self.snapshot_steps = {}
        for t in snapshot_times:
            step = int(round(t/dt_value))
            if step < 0 or step > steps or abs(step*dt_value - t) > 1e-8*dt_value:
                raise ValueError(
                    f"Snapshot time {t} s is outside the run or off the time grid."
                )
            self.snapshot_steps[step] = t

    def record(self, step, u, p, ksi, mesh_displacement):
        """Values at the current step, using the ALE map on the fluid edge."""
        u_values = u.vector().get_local()
        p_values = p.vector().get_local()
        xi_values = ksi.vector().get_local()
        motion_values = mesh_displacement.vector().get_local()

        for edge, y_ref in (("interface", R), ("centerline", 0.0)):
            self.data[f"p_{edge}"][step] = np.interp(
                self.x_ref, self.x_p1, p_values[self.p_dofs[edge]]
            )
            self.data[f"u_x_{edge}"][step] = u_values[self.u_dofs[edge][0]]
            self.data[f"u_y_{edge}"][step] = u_values[self.u_dofs[edge][1]]
            self.data[f"x_{edge}"][step] = self.x_ref + np.interp(
                self.x_ref, self.x_p1,
                motion_values[self.motion_dofs[edge][0]]
            )
            self.data[f"y_{edge}"][step] = y_ref + np.interp(
                self.x_ref, self.x_p1,
                motion_values[self.motion_dofs[edge][1]]
            )

        # The wall remains on its reference mesh; its values are paired with
        # the current fluid interface using the shared material coordinate X.
        self.data["xi_x_interface"][step] = xi_values[self.ksi_dofs[0]]
        self.data["xi_y_interface"][step] = xi_values[self.ksi_dofs[1]]

    def _two_edge_figure(self, plt, field, ylabel, filename):
        fig, axes = plt.subplots(2, len(self.snapshot_steps),
                                 figsize=(4.3*len(self.snapshot_steps), 6.2),
                                 sharex="col", constrained_layout=True)
        for col, (step, t) in enumerate(self.snapshot_steps.items()):
            for row, edge in enumerate(("interface", "centerline")):
                ax = axes[row, col]
                x = self.data[f"x_{edge}"][step]
                value = self.data[f"{field}_{edge}"][step]
                ax.plot(x, value, color="black", linewidth=1.6)
                ax.set_title(f"t = {t:.4f} s")
                ax.set_ylabel(f"{'Moving interface' if row == 0 else 'Centerline'}\n{ylabel}")
                ax.grid(alpha=0.25)
                if row == 1:
                    ax.set_xlabel("Physical axial coordinate x (cm)")
        fig.savefig(self.directory/filename, dpi=220)
        plt.close(fig)

    def save(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.directory/"wave_profiles_all_times.npz",
                            time_s=self.time, x_reference_cm=self.x_ref,
                            **self.data)

        for step, t in self.snapshot_steps.items():
            filename = self.directory/f"profiles_t{t:.4f}s.csv"
            with filename.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow((
                    "time_s", "x_reference_cm", "x_interface_cm",
                    "y_interface_cm", "x_centerline_cm", "y_centerline_cm",
                    "p_interface_dyn_cm2", "p_centerline_dyn_cm2",
                    "u_x_interface_cm_s", "u_y_interface_cm_s",
                    "u_x_centerline_cm_s", "u_y_centerline_cm_s",
                    "xi_x_interface_cm_s", "xi_y_interface_cm_s",
                ))
                for j, X in enumerate(self.x_ref):
                    writer.writerow((
                        t, X,
                        self.data["x_interface"][step, j],
                        self.data["y_interface"][step, j],
                        self.data["x_centerline"][step, j],
                        self.data["y_centerline"][step, j],
                        self.data["p_interface"][step, j],
                        self.data["p_centerline"][step, j],
                        self.data["u_x_interface"][step, j],
                        self.data["u_y_interface"][step, j],
                        self.data["u_x_centerline"][step, j],
                        self.data["u_y_centerline"][step, j],
                        self.data["xi_x_interface"][step, j],
                        self.data["xi_y_interface"][step, j],
                    ))

        (self.directory/"README.txt").write_text(
            "All steps, including t=0, are stored in wave_profiles_all_times.npz.\n"
            "Arrays have shape (number of time levels, number of P2 boundary nodes).\n"
            "Each column uses the initial/material coordinate x_reference_cm.\n"
            "Fluid boundary values are taken from the physically moved mesh.\n"
            "x_interface and y_interface are the lagged ALE position of the fluid edge.\n"
            "The structure velocity xi is evaluated on its fixed reference edge y=0.5.\n"
            "The centerline stays at y=0 but may move tangentially in the ALE mesh.\n"
            "Pressure is interpolated from its P1 edge nodes to the P2 x grid.\n"
            "The fluid velocity figure plots signed u_x, without absolute values.\n"
            "Snapshot plots use the physical axial coordinate; spacetime plots use\n"
            "the reference axial coordinate to follow the same material samples.\n"
        )

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib unavailable; NPZ and snapshot CSV data were saved.")
            return

        self._two_edge_figure(plt, "p", "Pressure (dyn/cm²)",
                              "fluid_pressure_profiles.png")
        self._two_edge_figure(plt, "u_x", "u_x (cm/s)",
                              "fluid_axial_velocity_profiles.png")

        fig, axes = plt.subplots(2, len(self.snapshot_steps),
                                 figsize=(4.3*len(self.snapshot_steps), 6.2),
                                 sharex="col", constrained_layout=True)
        for col, (step, t) in enumerate(self.snapshot_steps.items()):
            for row, comp in enumerate(("x", "y")):
                ax = axes[row, col]
                ax.plot(self.data["x_interface"][step],
                        self.data[f"xi_{comp}_interface"][step],
                        color="black", linewidth=1.6)
                ax.set_title(f"t = {t:.4f} s")
                ax.set_ylabel(f"Structure xi_{comp} (cm/s)")
                ax.grid(alpha=0.25)
                if row == 1:
                    ax.set_xlabel("Physical interface coordinate x (cm)")
        fig.savefig(self.directory/"structure_velocity_interface_profiles.png", dpi=220)
        plt.close(fig)

        # Time-space maps capture wave propagation between the snapshots.
        for name, label, filename in (
                ("p_centerline", "Centerline pressure (dyn/cm²)",
                 "pressure_centerline_time_space.png"),
                ("p_interface", "Interface pressure (dyn/cm²)",
                 "pressure_interface_time_space.png"),
                ("u_x_interface", "Interface fluid u_x (cm/s)",
                 "fluid_axial_velocity_interface_time_space.png"),
                ("xi_y_interface", "Interface structure xi_y (cm/s)",
                 "structure_normal_velocity_time_space.png")):
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            im = ax.imshow(self.data[name], origin="lower", aspect="auto",
                           extent=(0, L, self.time[0], self.time[-1]))
            ax.set_xlabel("Reference axial coordinate X (cm)")
            ax.set_ylabel("Time (s)")
            fig.colorbar(im, ax=ax, label=label)
            fig.savefig(self.directory/filename, dpi=200)
            plt.close(fig)
        print(f"Saved pressure-wave profiles and figures in {self.directory}",
              flush=True)


# ==========================================================
# Main run
# ==========================================================
def run_case(nx=80, dt_value=DEFAULT_DT, save_output=True,
             profiles_dir=PROFILE_DIR):
    mesh_f_reference, mesh_b = build_meshes(nx)
    mesh_f = Mesh(mesh_f_reference)  # Only this copy moves.
    bmf_f, bmf_b, ds_f, ds_b = mark_boundaries(mesh_f, mesh_b)
    bmf_ale, _, _, _ = mark_boundaries(mesh_f_reference, mesh_b)
    reference_vertices = mesh_f_reference.coordinates().copy()
    initial_areas = signed_triangle_areas(mesh_f)
    if np.any(np.abs(initial_areas) < 1e-14):
        raise RuntimeError("The initial fluid mesh contains a degenerate cell.")
    if MPI.size(mesh_f.mpi_comm()) != 1:
        raise RuntimeError("The moving-mesh coordinate transfer currently requires one MPI rank.")

    dt = Constant(dt_value)
    steps = int(np.round(T / dt_value))
    if abs(steps * dt_value - T) > 1e-12:
        raise ValueError("Choose dt_value so that T/dt is an integer.")
    if steps < 1:
        raise ValueError("The solve requires at least one time step.")

    # Echo the values used by this run, so an older copy of the script is
    # immediately distinguishable in the output.
    print(
        f"[benchmark] T={T:.6g}, dt={dt_value:.6g}, steps={steps}, "
        f"nx={nx}, ny_f=12, ny_b=5",
        flush=True,
    )
    print(
        f"[benchmark] rho_f={rho_f:.6g}, rho_p={rho_b:.6g}, "
        f"mu_f={mu_f:.6g}, mu_p={mu_b:.6g}, lambda_p={lambda_b:.6g}, "
        f"K={K_perm:.6g}, C0={C0:.6g}, alpha={alpha:.6g}, "
        f"gamma_BJS={C_BJS:.6g}, beta={beta_s:.6g}, "
        f"P_max={P_max:.6g}",
        flush=True,
    )

    # -----------------------
    # FE spaces
    # -----------------------
    V_f = FunctionSpace(mesh_f, MixedElement([
        VectorElement("CG", mesh_f.ufl_cell(), 2),   # fluid velocity
        FiniteElement("CG", mesh_f.ufl_cell(), 1)    # fluid pressure
    ]))
    u, p = TrialFunctions(V_f)
    v, q = TestFunctions(V_f)

    V_b = FunctionSpace(mesh_b, MixedElement([
        VectorElement("CG", mesh_b.ufl_cell(), 2),   # eta
        VectorElement("CG", mesh_b.ufl_cell(), 2),   # ksi
        FiniteElement("CG", mesh_b.ufl_cell(), 1)    # phi
    ]))
    eta, ksi, phi = TrialFunctions(V_b)
    vp, zeta, psi = TestFunctions(V_b)

    # Trace fields on the physical fluid mesh and reference Biot mesh.
    Uf_trace = VectorFunctionSpace(mesh_f, "CG", 2)
    Pf_trace = FunctionSpace(mesh_f, "CG", 1)

    Ub_trace = VectorFunctionSpace(mesh_b, "CG", 2)

    # Solve the P1 absolute ALE extension on the fixed reference fluid mesh.
    V_ale = VectorFunctionSpace(mesh_f_reference, "CG", 1)
    eta_ale_trial = TrialFunction(V_ale)
    tau_ale       = TestFunction(V_ale)
    eta_ale_fem   = Function(V_ale)
    eta_ale_old   = Function(V_ale)
    delta_ale_ref = Function(V_ale)
    eta_ext_ale  = Function(V_ale)

    # Both the reference ALE extension and physical-mesh motion use P1.
    # The fluid velocity remains P2 on the moving physical fluid mesh.
    V_motion = VectorFunctionSpace(mesh_f, "CG", 1)
    mesh_increment = Function(V_motion)
    mesh_velocity = Function(V_motion)
    mesh_displacement = Function(V_motion)

    # Record which vertex each physical-mesh displacement dof belongs to.
    # This lets us verify that ALE.move realizes the absolute reference map.
    vertex_by_coordinate = {
        coord_key(x): i for i, x in enumerate(reference_vertices)
    }
    motion_dof_coordinates = V_motion.tabulate_dof_coordinates().reshape((-1, 2))
    motion_dofs_by_component = []
    for component in range(2):
        dofs = np.asarray(V_motion.sub(component).dofmap().dofs(), dtype=int)
        vertices = np.asarray([
            vertex_by_coordinate[coord_key(motion_dof_coordinates[i])]
            for i in dofs
        ], dtype=int)
        motion_dofs_by_component.append((dofs, vertices))

    # Use the same piecewise-linear interface geometry in the Biot terms.
    V_geom_b = VectorFunctionSpace(mesh_b, "CG", 1)
    eta_geom_b = Function(V_geom_b)

    # -----------------------
    # Boundary conditions
    # -----------------------
    zero_scalar = Constant(0.0)
    zero_vec    = Constant((0.0, 0.0))

    # Fluid symmetry: only normal component zero on y=0 => y component zero
    bc_f = [
        DirichletBC(V_f.sub(0).sub(1), zero_scalar, bmf_f, 1)
    ]

    # Biot: clamped and drained outer wall; clamped, impermeable end faces.
    bc_b = [
        DirichletBC(V_b.sub(0), zero_vec, bmf_b, 2),
        DirichletBC(V_b.sub(0), zero_vec, bmf_b, 4),
        DirichletBC(V_b.sub(1), zero_vec, bmf_b, 2),
        DirichletBC(V_b.sub(1), zero_vec, bmf_b, 4),
    ]

    # ALE BCs on the fixed reference fluid mesh:
    # inlet/outlet fixed; symmetry only normal displacement zero; interface gets transferred eta
    ale_bc_in  = DirichletBC(V_ale, zero_vec, bmf_ale, 4)
    ale_bc_out = DirichletBC(V_ale, zero_vec, bmf_ale, 2)
    ale_bc_sym = DirichletBC(V_ale.sub(1), zero_scalar, bmf_ale, 1)

    # -----------------------
    # Inlet pressure
    # -----------------------
    p_in = InletPressure(degree=2)
    traction_in = as_vector((p_in, 0.0))

    # -----------------------
    # Current solution and one previous time level
    # -----------------------
    NS_old = Function(V_f)
    NS_new = Function(V_f)

    Bi_old = Function(V_b)
    Bi_new = Function(V_b)

    u_old, p_old = NS_old.split(True)

    eta_old, ksi_old, phi_old = Bi_old.split(True)

    for f in [u_old, eta_old, ksi_old]:
        f.interpolate(zero_vec)
    for f in [p_old, phi_old]:
        f.interpolate(zero_scalar)

    eta_ale_old.interpolate(zero_vec)
    eta_ale_fem.interpolate(zero_vec)
    delta_ale_ref.interpolate(zero_vec)
    eta_ext_ale.interpolate(zero_vec)
    mesh_increment.interpolate(zero_vec)
    mesh_velocity.interpolate(zero_vec)
    mesh_displacement.interpolate(zero_vec)
    eta_geom_b.interpolate(zero_vec)
    eta_geom_on_f = Function(V_motion)

    # Capture every time level directly from the solver. The ordinary XDMF
    # cadence (steps 1, 11, 21, ...) misses t=0.007 and t=0.0105 s.
    profiles = WaveProfiles(
        nx, steps, dt_value, u_old.function_space(),
        p_old.function_space(), ksi_old.function_space(), V_motion,
        output_dir=profiles_dir,
    )
    profiles.record(0, u_old, p_old, ksi_old, mesh_displacement)

    # -----------------------
    # Same-mesh transferred fields
    # -----------------------
    # Biot -> fluid
    ksi_old_f = Function(Uf_trace)
    phi_old_f = Function(Pf_trace)

    # fluid -> Biot
    u_old_b = Function(Ub_trace)

    # Lagged advecting velocity, used if Navier--Stokes convection is enabled.
    u_adv_be = Function(Uf_trace)

    # -----------------------
    # Build transfer operators
    # -----------------------
    T_ksi_b2f = VectorInterfaceTransfer(ksi_old.function_space(), Uf_trace, R)
    T_eta_b2ale = VectorInterfaceTransfer(eta_old.function_space(), V_ale, R)
    T_u_f2b = VectorInterfaceTransfer(u_old.function_space(), Ub_trace, R)
    T_phi_b2f = ScalarInterfaceTransfer(phi_old.function_space(), Pf_trace, R)
    T_ale_to_mesh = VectorInterfaceTransfer(V_ale, V_motion)
    T_geom_b2mesh = VectorInterfaceTransfer(V_geom_b, V_motion, R)

    def refresh_transfers():
        # Biot -> fluid
        T_ksi_b2f.transfer(ksi_old, ksi_old_f)
        T_phi_b2f.transfer(phi_old, phi_old_f)
        T_eta_b2ale.transfer(eta_old, eta_ext_ale)

        # fluid -> Biot
        T_u_f2b.transfer(u_old, u_old_b)

        # Both the convecting velocity and ALE boundary use the old time level.
        u_adv_be.assign(u_old)

    # Fluid normals and ds below are on the actual moved mesh. The Biot mesh
    # remains material/reference, so pull the physical interface geometry
    # back using the same P1 vertex geometry as the moved fluid mesh.
    n_f = FacetNormal(mesh_f)
    tor_f = as_vector((-n_f[1], n_f[0]))
    tangent_b = as_vector((1.0 + eta_geom_b[0].dx(0),
                           eta_geom_b[1].dx(0)))
    surface_b = sqrt(dot(tangent_b, tangent_b))
    normal_b_ds = as_vector((tangent_b[1], -tangent_b[0]))
    tangent_b_ds = tangent_b  # Matches (1, 0) on the reference interface.

    # -----------------------
    # ALE forms
    # -----------------------
    a_ale = inner(grad(eta_ale_trial), grad(tau_ale))*dx(domain=mesh_f_reference)
    L_ale = dot(Constant((0.0, 0.0)), tau_ale)*dx(domain=mesh_f_reference)

    # -----------------------
    # Output
    # -----------------------
    if save_output:
        xdmf_f = XDMFFile("artery_halfdomain_fluid_physical_BE.xdmf")
        xdmf_b = XDMFFile("artery_halfdomain_biot_physical_BE.xdmf")
        xdmf_f.parameters["flush_output"] = True
        xdmf_b.parameters["flush_output"] = True
        xdmf_f.parameters["functions_share_mesh"] = True
        xdmf_b.parameters["functions_share_mesh"] = True
        xdmf_f.parameters["rewrite_function_mesh"] = True

    solver_parameters = {"linear_solver": "umfpack"}

    # ==========================================================
    # Backward Euler forms: one-step differences and lagged interface traces
    # ==========================================================
    # Physical-domain fluid equations: mesh_f is moved before each solve.
    # Old nodal values travel with the mesh, giving the ALE time difference;
    # convection uses the relative velocity u_adv_be - mesh_velocity.
    F_f_be = (
        (rho_f/dt)*dot(u - u_old, v)*dx
        + (rho_f*dot(dot(u_adv_be - mesh_velocity, nabla_grad(u)), v)*dx
           if INCLUDE_NS_CONVECTION else 0)
        + 2.0*mu_f*inner(epsilon(u), epsilon(v))*dx
        - p*div(v)*dx
        + div(u)*q*dx
        + L_1*dot(u, n_f)*dot(v, n_f)*ds_f(3)
        + C_BJS*dot(u, tor_f)*dot(v, tor_f)*ds_f(3)
        - C_BJS*dot(ksi_old_f, tor_f)*dot(v, tor_f)*ds_f(3)
        - L_1*dot(u_old, n_f)*dot(v, n_f)*ds_f(3)
        + phi_old_f*dot(v, n_f)*ds_f(3)
        - dot(traction_in, v)*ds_f(4)
    )

    F_pe_be = (
        (rho_b/dt)*dot(ksi - ksi_old, zeta)*dx
        + 2.0*mu_b*inner(epsilon(eta), epsilon(zeta))*dx
        + lambda_b*div(eta)*div(zeta)*dx
        - alpha*phi*div(zeta)*dx
        + beta_s*dot(eta, zeta)*dx
        + dot(ksi, vp)*dx
        - (1.0/dt)*dot(eta - eta_old, vp)*dx
        + (L_2/surface_b)*dot(ksi, normal_b_ds)*dot(zeta, normal_b_ds)*ds_b(3)
        + (C_BJS/surface_b)*dot(ksi, tangent_b_ds)*dot(zeta, tangent_b_ds)*ds_b(3)
        + phi*dot(zeta, normal_b_ds)*ds_b(3)
        - (C_BJS/surface_b)*dot(u_old_b, tangent_b_ds)*dot(zeta, tangent_b_ds)*ds_b(3)
        - (L_2/surface_b)*dot(ksi_old, normal_b_ds)*dot(zeta, normal_b_ds)*ds_b(3)
    )

    F_pd_be = (
        (C0/dt)*(phi - phi_old)*psi*dx
        + alpha*div(ksi)*psi*dx
        + K_perm*dot(grad(phi), grad(psi))*dx
        + (surface_b/L_3)*phi*psi*ds_b(3)
        - dot(ksi, normal_b_ds)*psi*ds_b(3)
        + dot(u_old_b, normal_b_ds)*psi*ds_b(3)
        - (surface_b/L_3)*phi_old*psi*ds_b(3)
    )

    a_f_be, L_f_be = lhs(F_f_be), rhs(F_f_be)
    a_b_be, L_b_be = lhs(F_pe_be + F_pd_be), rhs(F_pe_be + F_pd_be)

    # DirichletBC retains a reference to eta_ext_ale, refreshed each step.
    ale_bc_if = DirichletBC(V_ale, eta_ext_ale, bmf_ale, 3)
    bc_ale = [ale_bc_in, ale_bc_out, ale_bc_sym, ale_bc_if]

    for nstep in range(1, steps + 1):
        t = nstep * dt_value
        p_in.t = t

        # All interface data are from the preceding time level.
        refresh_transfers()

        # Solve for absolute ALE displacement on the unchanged reference mesh.
        solve(a_ale == L_ale, eta_ale_fem, bc_ale,
              solver_parameters=solver_parameters)

        # Transfer the reference P1 ALE map and its increment to the matching
        # physical-mesh vertices.
        # The physical mesh receives the change in the absolute map, avoiding
        # accumulated drift when eta_old varies in time.
        delta_ale_ref.vector()[:] = (
            eta_ale_fem.vector() - eta_ale_old.vector()
        )
        T_ale_to_mesh.transfer(delta_ale_ref, mesh_increment)
        T_ale_to_mesh.transfer(eta_ale_fem, mesh_displacement)
        mesh_velocity.vector().set_local(
            mesh_increment.vector().get_local() / dt_value
        )
        mesh_velocity.vector().apply("insert")
        ALE.move(mesh_f, mesh_increment)

        area_ratios = signed_triangle_areas(mesh_f) / initial_areas
        min_area_ratio = np.min(area_ratios)
        if not np.all(np.isfinite(area_ratios)) or min_area_ratio <= 0.05:
            raise RuntimeError(
                f"Fluid mesh has an inverted/nearly collapsed cell at "
                f"step {nstep}, t={t:.6e}; minimum oriented area ratio "
                f"is {min_area_ratio:.6e}."
            )
        if not np.allclose(mesh_f_reference.coordinates(), reference_vertices,
                           rtol=0.0, atol=1e-12):
            raise RuntimeError("The reference ALE mesh moved unexpectedly.")
        displacement_values = mesh_displacement.vector().get_local()
        physical_vertices = mesh_f.coordinates()
        mesh_errors = [
            np.max(np.abs(
                physical_vertices[vertices, component]
                - reference_vertices[vertices, component]
                - displacement_values[dofs]
            ))
            for component, (dofs, vertices) in enumerate(motion_dofs_by_component)
        ]
        if max(mesh_errors) > 1e-9:
            raise RuntimeError(
                f"Physical mesh does not match the reference ALE map "
                f"at step {nstep}; maximum vertex mismatch "
                f"is {max(mesh_errors):.6e}."
            )

        # The Biot bulk stays on its reference mesh. Its interface uses the
        # P1 vertex geometry of the moved fluid mesh at this lagged time.
        eta_geom_b.interpolate(eta_old)
        T_geom_b2mesh.transfer(eta_geom_b, eta_geom_on_f)
        interface_dofs = T_geom_b2mesh.tgt_match
        interface_error = np.max(np.abs(
            displacement_values[interface_dofs]
            - eta_geom_on_f.vector().get_local()[interface_dofs]
        ))
        if interface_error > 1e-9:
            raise RuntimeError(
                f"Fluid and Biot interface positions disagree at step "
                f"{nstep}; maximum displacement mismatch "
                f"is {interface_error:.6e}."
            )
        print(f"step {nstep:4d} ALE | ||eta_ALE||={norm(eta_ale_fem.vector(), 'l2'):.6e}, "
              f"||w||={norm(mesh_velocity.vector(), 'l2'):.6e}, "
              f"min area ratio={min_area_ratio:.6e}")

        # Solve fluid and Biot independently with the lagged interface traces.
        solve(a_f_be == L_f_be, NS_new, bc_f,
              solver_parameters=solver_parameters)
        solve(a_b_be == L_b_be, Bi_new, bc_b,
              solver_parameters=solver_parameters)

        u_new, p_new = NS_new.split(True)
        eta_new, ksi_new, phi_new = Bi_new.split(True)

        profiles.record(nstep, u_new, p_new, ksi_new, mesh_displacement)

        if save_output and (nstep == 1 or nstep % 10 == 1 or nstep == steps
                            or nstep in profiles.snapshot_steps):
            u_new.rename("u_f", "u_f")
            p_new.rename("p_f", "p_f")
            mesh_displacement.rename("disp", "disp")
            eta_new.rename("eta_p", "eta_p")
            ksi_new.rename("ksi_p", "ksi_p")
            phi_new.rename("phi_p", "phi_p")
            xdmf_f.write(u_new, t)
            xdmf_f.write(p_new, t)
            xdmf_f.write(mesh_displacement, t)
            xdmf_b.write(eta_new, t)
            xdmf_b.write(ksi_new, t)
            xdmf_b.write(phi_new, t)

        print_norms(u_new, p_new, eta_new, ksi_new, phi_new, nstep, t)

        # Retain only the current ALE map and solution for the next step.
        eta_ale_old.assign(eta_ale_fem)
        u_old.assign(u_new)
        p_old.assign(p_new)
        eta_old.assign(eta_new)
        ksi_old.assign(ksi_new)
        phi_old.assign(phi_new)

    if save_output:
        xdmf_f.close()
        xdmf_b.close()

    profiles.save()

    return mesh_f, mesh_b, bmf_f, bmf_b


def main():
    run_case(nx=80, dt_value=DEFAULT_DT, save_output=True)


if __name__ == "__main__":
    main()
