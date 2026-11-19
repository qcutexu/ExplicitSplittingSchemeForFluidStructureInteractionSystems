from fenics import *
import numpy as np

"""
Half-domain idealized artery benchmark with:
  - explicit interface trace transfer (same-mesh target fields)
  - ALE extension of Biot displacement into the fluid mesh
  - first-order backward Euler at every time step
  - explicit, one-step-lagged interface traces and ALE displacement

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
T   = 0.0341

rho_f    = 1.0
mu_f     = 0.035

rho_b    = 1.1
mu_b     = 5.575e5
lambda_b = 1.7e6
alpha    = 1.0
C0       = 1.0e-3
K_perm   = 1.0e-8

C_BJS    = 1.0e3
L_1      = 10.0
L_2      = 10.0
L_3      = 10.0
beta_s   = 5e7

P_max    = 13334.0
T_max    = 0.003

# The ALE Navier--Stokes convection uses the velocity from the preceding step.
# Switch it off to recover the Stokes-like fluid model used in the convergence tests.
INCLUDE_NS_CONVECTION = False



# ==========================================================
# Helpers
# ==========================================================
def epsilon(u):
    return sym(nabla_grad(u))


def epsilon_ALE(u, F):
    """Twice the symmetric velocity gradient in physical coordinates,
    pulled back to the reference fluid mesh.

    For F = I, this reduces to grad(u) + grad(u).T = 2*epsilon(u).
    """
    Finv = inv(F)
    return grad(u)*Finv + Finv.T*nabla_grad(u)


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
    ny_f = 5
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
    Exact scalar transfer on the fixed/reference interface by matching dof
    coordinates.  This transfer is built and used on the fixed reference
    meshes.
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
    Exact vector transfer on the fixed/reference interface by matching node
    coordinates, then copying values componentwise.
    """
    def __init__(self, source_space, target_space, y_interface, tol=1e-12, ndigits=12):
        self.source_space = source_space
        self.target_space = target_space
        self.y_interface = y_interface
        self.tol = tol
        self.ndigits = ndigits

        gdim = source_space.mesh().geometry().dim()
        self.gdim = gdim

        src_coords_all = source_space.tabulate_dof_coordinates().reshape((-1, gdim))
        tgt_coords_all = target_space.tabulate_dof_coordinates().reshape((-1, gdim))

        src_node_coords = src_coords_all[::gdim, :]
        tgt_node_coords = tgt_coords_all[::gdim, :]

        self.src_if_nodes = np.where(np.abs(src_node_coords[:, 1] - y_interface) < tol)[0]
        self.tgt_if_nodes = np.where(np.abs(tgt_node_coords[:, 1] - y_interface) < tol)[0]

        if len(self.src_if_nodes) == 0:
            raise RuntimeError("No source vector interface nodes found.")
        if len(self.tgt_if_nodes) == 0:
            raise RuntimeError("No target vector interface nodes found.")

        src_map = {}
        for i in self.src_if_nodes:
            src_map[coord_key(src_node_coords[i], ndigits)] = i

        src_match_nodes = []
        tgt_match_nodes = []
        missing = []
        for j in self.tgt_if_nodes:
            key = coord_key(tgt_node_coords[j], ndigits)
            if key in src_map:
                src_match_nodes.append(src_map[key])
                tgt_match_nodes.append(j)
            else:
                missing.append(tuple(tgt_node_coords[j]))

        if missing:
            raise RuntimeError(
                "Exact vector interface matching failed. "
                f"Unmatched target coordinates: {missing[:10]}"
            )

        self.src_match_nodes = np.array(src_match_nodes, dtype=int)
        self.tgt_match_nodes = np.array(tgt_match_nodes, dtype=int)

    def transfer(self, src_func, tgt_func, zero_elsewhere=True):
        src = src_func.vector().get_local().reshape((-1, self.gdim))
        tgt = tgt_func.vector().get_local().reshape((-1, self.gdim))
        if zero_elsewhere:
            tgt[:, :] = 0.0
        tgt[self.tgt_match_nodes, :] = src[self.src_match_nodes, :]
        tgt_func.vector().set_local(tgt.reshape(-1))
        tgt_func.vector().apply("insert")

def print_norms(u, p, eta, ksi, phi, step, t):
    print(
        f"step {step:4d}, t={t:.6f} | "
        f"||u||={norm(u.vector(), 'l2'):.6e}, "
        f"||p||={norm(p.vector(), 'l2'):.6e}, "
        f"||eta||={norm(eta.vector(), 'l2'):.6e}, "
        f"||ksi||={norm(ksi.vector(), 'l2'):.6e}, "
        f"||phi||={norm(phi.vector(), 'l2'):.6e}"
    )


# ==========================================================
# Main run
# ==========================================================
def run_case(nx=80, dt_value=1e-4, save_output=True):
    mesh_f, mesh_b = build_meshes(nx)
    bmf_f, bmf_b, ds_f, ds_b = mark_boundaries(mesh_f, mesh_b)

    dt = Constant(dt_value)
    steps = int(np.round(T / dt_value))
    if abs(steps * dt_value - T) > 1e-12:
        raise ValueError("Choose dt_value so that T/dt is an integer.")
    if steps < 1:
        raise ValueError("The solve requires at least one time step.")

    n_f   = FacetNormal(mesh_f)
    tor_f = as_vector((-n_f[1], n_f[0]))

    n_b   = FacetNormal(mesh_b)
    tor_b = as_vector((-n_b[1], n_b[0]))

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

    # Same-mesh trace spaces for transfer
    Uf_trace = VectorFunctionSpace(mesh_f, "CG", 2)
    Pf_trace = FunctionSpace(mesh_f, "CG", 1)

    Ub_trace = VectorFunctionSpace(mesh_b, "CG", 2)
    Pb_trace = FunctionSpace(mesh_b, "CG", 1)

    # ALE extension space on fluid mesh
    V_ale = VectorFunctionSpace(mesh_f, "CG", 2)
    eta_ale_trial = TrialFunction(V_ale)
    tau_ale       = TestFunction(V_ale)
    eta_ale_fem   = Function(V_ale)
    eta_ale_old   = Function(V_ale)
    omega_ale     = Function(V_ale)

    # ALE deformation gradient on the reference fluid mesh:
    #   x = A(X) = X + eta_ALE(X),
    #   FF = grad_X A = I + grad_X eta_ALE.
    # The fluid equations below are assembled on the reference mesh using
    # J = det(FF) and Finv = inv(FF), instead of physically moving the mesh.
    V_tensor_f = TensorFunctionSpace(mesh_f, "CG", 2)
    FF = Function(V_tensor_f)
    FF.assign(project(Identity(2), V_tensor_f, solver_type="cg"))
    J = det(FF)
    Finv = inv(FF)

    # -----------------------
    # Boundary conditions
    # -----------------------
    zero_scalar = Constant(0.0)
    zero_vec    = Constant((0.0, 0.0))

    # Fluid symmetry: only normal component zero on y=0 => y component zero
    bc_f = [
        DirichletBC(V_f.sub(0).sub(1), zero_scalar, bmf_f, 1)
    ]

    # Biot
    bc_b = [
        DirichletBC(V_b.sub(0), zero_vec, bmf_b, 2),
        DirichletBC(V_b.sub(0), zero_vec, bmf_b, 4),
        DirichletBC(V_b.sub(1), zero_vec, bmf_b, 2),
        DirichletBC(V_b.sub(1), zero_vec, bmf_b, 4),
        #DirichletBC(V_b.sub(1).sub(1), zero_scalar, bmf_b, 1)
    ]

    # ALE BCs on fluid mesh:
    # inlet/outlet fixed; symmetry only normal displacement zero; interface gets transferred eta
    ale_bc_in  = DirichletBC(V_ale, zero_vec, bmf_f, 4)
    ale_bc_out = DirichletBC(V_ale, zero_vec, bmf_f, 2)
    ale_bc_sym = DirichletBC(V_ale.sub(1), zero_scalar, bmf_f, 1)

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
    omega_ale.interpolate(zero_vec)

    # -----------------------
    # Same-mesh transferred fields
    # -----------------------
    # Biot -> fluid
    ksi_old_f = Function(Uf_trace)
    phi_old_f = Function(Pf_trace)
    eta_old_f = Function(Uf_trace)

    # Lagged wall displacement used as the fluid ALE interface data
    eta_ext_ale = Function(V_ale)

    # fluid -> Biot
    u_old_b = Function(Ub_trace)

    # Lagged advecting velocity, used if Navier--Stokes convection is enabled.
    u_adv_be = Function(Uf_trace)

    # -----------------------
    # Build transfer operators
    # -----------------------
    T_ksi_b2f = VectorInterfaceTransfer(Ub_trace, Uf_trace, R)
    T_eta_b2f = VectorInterfaceTransfer(Ub_trace, Uf_trace, R)
    T_u_f2b   = VectorInterfaceTransfer(Uf_trace, Ub_trace, R)
    T_phi_b2f = ScalarInterfaceTransfer(Pb_trace, Pf_trace, R)

    def refresh_transfers():
        # Biot -> fluid
        T_ksi_b2f.transfer(ksi_old, ksi_old_f)
        T_phi_b2f.transfer(phi_old, phi_old_f)
        T_eta_b2f.transfer(eta_old, eta_old_f)

        # fluid -> Biot
        T_u_f2b.transfer(u_old, u_old_b)

        # Both the convecting velocity and ALE boundary use time level n.
        u_adv_be.assign(u_old)
        eta_ext_ale.interpolate(eta_old_f)

    # -----------------------
    # ALE forms
    # -----------------------
    a_ale = inner(grad(eta_ale_trial), grad(tau_ale))*dx(domain=mesh_f)
    L_ale = dot(Constant((0.0, 0.0)), tau_ale)*dx(domain=mesh_f)

    # -----------------------
    # Output
    # -----------------------
    if save_output:
        xdmf_f = XDMFFile("artery_halfdomain_fluid_BE.xdmf")
        xdmf_b = XDMFFile("artery_halfdomain_biot_BE.xdmf")
        xdmf_f.parameters["flush_output"] = True
        xdmf_b.parameters["flush_output"] = True
        xdmf_f.parameters["functions_share_mesh"] = True
        xdmf_b.parameters["functions_share_mesh"] = True

    solver_parameters = {"linear_solver": "umfpack"}

    # ==========================================================
    # Backward Euler forms: one-step differences and lagged interface traces
    # ==========================================================
    # Fluid equations are pulled back to the fixed reference mesh by FF.
    # For the optional convection, u_adv_be contains the preceding velocity.
    F_f_be = (
        (rho_f/dt)*J*dot(u - u_old, v)*dx
        + (rho_f*J*dot(dot(dot(Finv, u_adv_be - omega_ale), nabla_grad(u)), v)*dx
           if INCLUDE_NS_CONVECTION else 0)
        + mu_f*J*inner(epsilon_ALE(u, FF)*Finv.T, grad(v))*dx
        - J*inner(p*Finv.T, grad(v))*dx
        + div(J*Finv*u)*q*dx
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
        + L_2*dot(ksi, n_b)*dot(zeta, n_b)*ds_b(3)
        + C_BJS*dot(ksi, tor_b)*dot(zeta, tor_b)*ds_b(3)
        + phi*dot(zeta, n_b)*ds_b(3)
        - C_BJS*dot(u_old_b, tor_b)*dot(zeta, tor_b)*ds_b(3)
        - L_2*dot(ksi_old, n_b)*dot(zeta, n_b)*ds_b(3)
    )

    F_pd_be = (
        (C0/dt)*(phi - phi_old)*psi*dx
        + alpha*div(ksi)*psi*dx
        + K_perm*dot(grad(phi), grad(psi))*dx
        + (1.0/L_3)*phi*psi*ds_b(3)
        - dot(ksi, n_b)*psi*ds_b(3)
        + dot(u_old_b, n_b)*psi*ds_b(3)
        - (1.0/L_3)*phi_old*psi*ds_b(3)
    )

    a_f_be, L_f_be = lhs(F_f_be), rhs(F_f_be)
    a_b_be, L_b_be = lhs(F_pe_be + F_pd_be), rhs(F_pe_be + F_pd_be)

    # DirichletBC retains a reference to eta_ext_ale, refreshed each step.
    ale_bc_if = DirichletBC(V_ale, eta_ext_ale, bmf_f, 3)
    bc_ale = [ale_bc_in, ale_bc_out, ale_bc_sym, ale_bc_if]

    for nstep in range(1, steps + 1):
        t = nstep * dt_value
        p_in.t = t

        # All interface data are from the preceding time level.
        refresh_transfers()

        # Solve the harmonic ALE extension with eta^n on the interface.
        solve(a_ale == L_ale, eta_ale_fem, bc_ale,
              solver_parameters=solver_parameters)
        omega_ale.vector()[:] = (
            eta_ale_fem.vector() - eta_ale_old.vector()
        ) / dt_value
        FF.assign(project(Identity(2) + grad(eta_ale_fem), V_tensor_f,
                          solver_type="cg"))
        print(f"step {nstep:4d} ALE | ||eta_ALE||={norm(eta_ale_fem.vector(), 'l2'):.6e}, "
              f"||w||={norm(omega_ale.vector(), 'l2'):.6e}")

        # Solve fluid and Biot independently with the lagged interface traces.
        solve(a_f_be == L_f_be, NS_new, bc_f,
              solver_parameters=solver_parameters)
        solve(a_b_be == L_b_be, Bi_new, bc_b,
              solver_parameters=solver_parameters)

        u_new, p_new = NS_new.split(True)
        eta_new, ksi_new, phi_new = Bi_new.split(True)

        if save_output and (nstep == 1 or nstep % 10 == 1 or nstep == steps):
            u_new.rename("u_f", "u_f")
            p_new.rename("p_f", "p_f")
            eta_ale_fem.rename("disp", "disp")
            eta_new.rename("eta_p", "eta_p")
            ksi_new.rename("ksi_p", "ksi_p")
            phi_new.rename("phi_p", "phi_p")
            xdmf_f.write(u_new, t)
            xdmf_f.write(p_new, t)
            xdmf_f.write(eta_ale_fem, t)
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

    return mesh_f, mesh_b, bmf_f, bmf_b


def main():
    run_case(nx=80, dt_value=1e-4, save_output=True)


if __name__ == "__main__":
    main()
