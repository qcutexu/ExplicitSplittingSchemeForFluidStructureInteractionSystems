"""Spatial refinement for the polynomial manufactured solution from ex1.

The default time step is fixed at 1e-3 on every mesh and T=1. Mesh refinement
changes only the interval count. If a requested step does not divide T, use
one common smaller uniform step so every mesh reaches the same final time.
BDF2--EX2 and the Neumann--Robin normal coupling are retained, with projected
exact fields at both starting levels. Maximum-in-time errors use elastic
energy for displacement and L2 for the other fields.

Run: python3 nsbiot_Neumann_Robin_spatial_sweep.py
Preview: append --dry-run (FEniCS is not required for this preview).
"""

import argparse
import math
from fractions import Fraction

import numpy as np

try:
    from fenics import *
except ImportError as exc:
    _fenics_import_error = exc
else:
    _fenics_import_error = None

# Physical parameters and the existing Neumann--Robin coefficients.
rho_f = 1.0
mu_f = 1.0
rho_b = 1.0
mu_b = 1.0
lambda_b = 1.0
alpha = 1.0
C0 = 1.0
K = 1.0
C_BJS = 1.0
L_1 = 1.0
L_2 = 1.0
L_3 = 1.0

# The attached polynomial fields satisfy the physical interface conditions
# for these unit stress coefficients.
if any(abs(value - 1.0) > 1e-14
       for value in (mu_f, mu_b, lambda_b, alpha)):
    raise ValueError("The polynomial example requires unit stress coefficients")

def epsilon(u):
    return sym(nabla_grad(u))


def spatial_polynomials(mesh):
    """f=x²(1-x)², s=y-3y³+2y⁴, H=(1+y)²(1-2y), J=y(1+y)², B=y²(1+y)."""
    X = SpatialCoordinate(mesh)
    x, y = X[0], X[1]
    f = x*x*(1.0-x)*(1.0-x)
    fp = 2.0*x - 6.0*x*x + 4.0*x*x*x
    s = y - 3.0*y*y*y + 2.0*y*y*y*y
    sp = 1.0 - 9.0*y*y + 8.0*y*y*y
    H = (1.0+y)*(1.0+y)*(1.0-2.0*y)
    J = y*(1.0+y)*(1.0+y)
    B = y*y*(1.0+y)
    return f, fp, s, sp, H, J, B


def make_fluid_data(mesh, time):
    """u=2t(f s',-f' s), p=t²f-4tf'; differentiate the stress for the source."""
    f, fp, s, sp, _, _, _ = spatial_polynomials(mesh)
    u_exact = as_vector((2.0*time*f*sp, -2.0*time*fp*s))
    p_exact = time*time*f - 4.0*time*fp
    u_t = as_vector((2.0*f*sp, -2.0*fp*s))
    stress_f = 2.0*mu_f*epsilon(u_exact) - p_exact*Identity(2)
    f_f = rho_f*u_t - div(stress_f)
    return u_exact, p_exact, f_f


def make_biot_data(mesh, time):
    """eta=t²(fH,-f'J/3), xi=eta_t, phi=f(B+t²H); derive both sources."""
    f, fp, _, _, H, J, B = spatial_polynomials(mesh)
    eta_shape = as_vector((f*H, -(fp*J)/3.0))
    eta_exact = time*time*eta_shape
    ksi_exact = 2.0*time*eta_shape
    phi_exact = f*(B + time*time*H)
    stress_p = (2.0*mu_b*epsilon(eta_exact)
                + lambda_b*div(eta_exact)*Identity(2)
                - alpha*phi_exact*Identity(2))
    f_e = 2.0*rho_b*eta_shape - div(stress_p)
    f_p = C0*(2.0*time*f*H) + alpha*div(ksi_exact) - K*div(grad(phi_exact))
    return eta_exact, ksi_exact, phi_exact, f_e, f_p

def time_grid(dt_target, T):
    """Use a uniform grid ending at T, with at least two BDF2 levels."""
    if not math.isfinite(float(T)) or T <= 0.0:
        raise ValueError("The final time must be positive and finite")
    if not math.isfinite(float(dt_target)) or dt_target <= 0.0:
        raise ValueError("The target time step must be positive and finite")
    end = Fraction(str(T))
    target = (dt_target if isinstance(dt_target, Fraction)
              else Fraction(str(dt_target)))
    ratio = end / target
    steps = max(2, (ratio.numerator + ratio.denominator - 1)
                // ratio.denominator)
    return float(end / steps), steps


def validate_refinements(values):
    if not values or any(value < 1 for value in values):
        raise ValueError("Refinement parameters must be positive integers")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Refinement parameters must be strictly increasing")


def solve_case(dt_val, nm, T, phi_degree=1):
    """Return hmax, dt and maximum errors: eta in elastic energy; others in L2."""
    if _fenics_import_error is not None:
        raise RuntimeError(
            "Legacy FEniCS is required to solve; --dry-run previews the schedule"
        ) from _fenics_import_error
    if nm < 1 or phi_degree not in (1, 2):
        raise ValueError("Use a positive mesh size and pore-pressure degree 1 or 2")
    steps = int(round(T / dt_val))
    if steps < 2 or not math.isclose(steps * dt_val, T,
                                     rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("The uniform time step must divide T into at least two steps")
    dt = Constant(dt_val)

    mesh_f = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), nm, nm)
    mesh_b = RectangleMesh(Point(0.0, -1.0), Point(1.0, 0.0), nm, nm)
    dx_f = Measure("dx", domain=mesh_f)
    dx_b = Measure("dx", domain=mesh_b)
    n_f = FacetNormal(mesh_f)
    tor_f = as_vector((-n_f[1], n_f[0]))
    n_b = FacetNormal(mesh_b)
    tor_b = as_vector((-n_b[1], n_b[0]))

    class Right_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1.0)

    class Left_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    class Top_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1.0)

    class Interface_F(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)

    bmf_f = MeshFunction("size_t", mesh_f, mesh_f.topology().dim() - 1, 0)
    Interface_F().mark(bmf_f, 5)
    Right_F().mark(bmf_f, 2)
    Top_F().mark(bmf_f, 3)
    Left_F().mark(bmf_f, 4)
    ds_f = Measure("ds", domain=mesh_f, subdomain_data=bmf_f)

    class Right_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 1.0)

    class Left_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    class Bottom_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], -1.0)

    class Interface_B(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)

    bmf_b = MeshFunction("size_t", mesh_b, mesh_b.topology().dim() - 1, 0)
    Interface_B().mark(bmf_b, 5)
    Right_B().mark(bmf_b, 2)
    Bottom_B().mark(bmf_b, 1)
    Left_B().mark(bmf_b, 4)
    ds_b = Measure("ds", domain=mesh_b, subdomain_data=bmf_b)

    time = Constant(0.0)
    u_D, p_D, f_f = make_fluid_data(mesh_f, time)
    eta_D, ksi_D, phi_D, f_e, f_p = make_biot_data(mesh_b, time)

    V_f = FunctionSpace(mesh_f, MixedElement([
        VectorElement("CG", mesh_f.ufl_cell(), 2),
        FiniteElement("CG", mesh_f.ufl_cell(), 1),
    ]))
    u, p = TrialFunctions(V_f)
    v, q = TestFunctions(V_f)

    V_b = FunctionSpace(mesh_b, MixedElement([
        VectorElement("CG", mesh_b.ufl_cell(), 2),
        VectorElement("CG", mesh_b.ufl_cell(), 2),
        FiniteElement("CG", mesh_b.ufl_cell(), phi_degree),
    ]))
    eta, ksi, phi = TrialFunctions(V_b)
    vp, zeta, psi = TestFunctions(V_b)

    # Exterior conditions of the selected polynomial example.
    zero_vec = Constant((0.0, 0.0))
    zero_scalar = Constant(0.0)
    bc_f = [DirichletBC(V_f.sub(0), zero_vec, bmf_f, side)
            for side in (4, 3, 2)]
    bc_b = (
        [DirichletBC(V_b.sub(0), zero_vec, bmf_b, side)
         for side in (1, 2, 4)]
        + [DirichletBC(V_b.sub(1), zero_vec, bmf_b, side)
           for side in (1, 2, 4)]
        + [DirichletBC(V_b.sub(2), zero_scalar, bmf_b, side)
           for side in (1, 2, 4)]
    )

    NS_old = Function(V_f)
    u_old, p_old = NS_old.split(True)
    NS_old1 = Function(V_f)
    u_old1, p_old1 = NS_old1.split(True)
    NS_new = Function(V_f)

    Bi_old = Function(V_b)
    eta_old, ksi_old, phi_old = Bi_old.split(True)
    Bi_old1 = Function(V_b)
    eta_old1, ksi_old1, phi_old1 = Bi_old1.split(True)
    Bi_new = Function(V_b)

    # Joint Robin--Stokes projection for the nonzero second starting level.
    a_f0 = (
        2.0 * mu_f * inner(epsilon(u), epsilon(v)) * dx_f
        - p * div(v) * dx_f + div(u) * q * dx_f
        + C_BJS * dot(u, tor_f) * dot(v, tor_f) * ds_f(5)
        + L_1 * dot(u, n_f) * dot(v, n_f) * ds_f(5)
    )
    L_f0 = (
        2.0 * mu_f * inner(epsilon(u_D), epsilon(v)) * dx_f
        - p_D * div(v) * dx_f + div(u_D) * q * dx_f
        + C_BJS * dot(u_D, tor_f) * dot(v, tor_f) * ds_f(5)
        + L_1 * dot(u_D, n_f) * dot(v, n_f) * ds_f(5)
    )
    solver_parameters = {"linear_solver": "umfpack"}

    def initialize_level(value, uh, ph, etah, ksih, phih):
        time.assign(value)
        # At t=0, u=p=eta=xi=0, so their projections are already zero.
        if value != 0.0:
            projected_fluid = Function(V_f)
            solve(a_f0 == L_f0, projected_fluid, bc_f,
                  solver_parameters=solver_parameters)
            up, pp = projected_fluid.split(True)
            uh.assign(up)
            ph.assign(pp)

            # Apply the same elasticity Ritz map to displacement and velocity.
            for exact_field, history in ((eta_D, etah), (ksi_D, ksih)):
                space = history.function_space()
                projected = TrialFunction(space)
                test = TestFunction(space)
                a_e0 = (
                    2.0 * mu_b * inner(epsilon(projected), epsilon(test))
                    + lambda_b * div(projected) * div(test)
                ) * dx_b
                L_e0 = (
                    2.0 * mu_b * inner(epsilon(exact_field), epsilon(test))
                    + lambda_b * div(exact_field) * div(test)
                ) * dx_b
                bc_e0 = [DirichletBC(space, zero_vec, bmf_b, side)
                         for side in (1, 2, 4)]
                solve(a_e0 == L_e0, history, bc_e0,
                      solver_parameters=solver_parameters)

        # phi(0)=f(x) B(y) is nonzero inside: project it as in the attachment.
        space = phih.function_space()
        projected = TrialFunction(space)
        test = TestFunction(space)
        a_phi0 = K * dot(grad(projected), grad(test)) * dx_b
        L_phi0 = K * dot(grad(phi_D), grad(test)) * dx_b
        bc_phi0 = [DirichletBC(space, zero_scalar, bmf_b, side)
                   for side in (1, 2, 4)]
        solve(a_phi0 == L_phi0, phih, bc_phi0,
              solver_parameters=solver_parameters)

    # Exact projected starts at t=0 and t=dt; the first BDF2 solve is at 2*dt.
    initialize_level(0.0, u_old1, p_old1, eta_old1, ksi_old1, phi_old1)
    initialize_level(dt_val, u_old, p_old, eta_old, ksi_old, phi_old)

    for fn in (u_old, u_old1, ksi_old, ksi_old1, phi_old, phi_old1):
        fn.set_allow_extrapolation(True)

    # BDF2 derivatives and EX2 interface predictors, a*=2*a_old-a_old1.
    # The new fluid field is not inserted into the current Biot solve.
    F_f3 = (
        (rho_f / (2.0 * dt)) * dot(3.0*u - 4.0*u_old + u_old1, v) * dx_f
        + 2.0*mu_f*inner(epsilon(u), epsilon(v))*dx_f
        - p*div(v)*dx_f + div(u)*q*dx_f
        + L_1*dot(u, n_f)*dot(v, n_f)*ds_f(5)
        + C_BJS*dot(u, tor_f)*dot(v, tor_f)*ds_f(5)
        - dot(f_f, v)*dx_f
        - C_BJS*dot(2.0*ksi_old - ksi_old1, tor_f)*dot(v, tor_f)*ds_f(5)
        - L_1*dot(2.0*u_old - u_old1, n_f)*dot(v, n_f)*ds_f(5)
        + (2.0*phi_old - phi_old1)*dot(v, n_f)*ds_f(5)
    )

    # Neumann normal mechanics: retain current phi, with no normal-velocity
    # Robin penalty and no extrapolated solid normal-velocity load.
    F_pe3 = (
        (rho_b / (2.0*dt))*dot(3.0*ksi - 4.0*ksi_old + ksi_old1, zeta)*dx_b
        + 2.0*mu_b*inner(epsilon(eta), epsilon(zeta))*dx_b
        + lambda_b*div(eta)*div(zeta)*dx_b
        - alpha*phi*div(zeta)*dx_b
        + dot(ksi, vp)*dx_b
        - (1.0/(2.0*dt))*dot(3.0*eta - 4.0*eta_old + eta_old1, vp)*dx_b
        + C_BJS*dot(ksi, tor_b)*dot(zeta, tor_b)*ds_b(5)
        + phi*dot(zeta, n_b)*ds_b(5)
        - dot(f_e, zeta)*dx_b
        - C_BJS*dot(2.0*u_old - u_old1, tor_b)*dot(zeta, tor_b)*ds_b(5)
    )
    F_pd3 = (
        (C0/(2.0*dt))*(3.0*phi - 4.0*phi_old + phi_old1)*psi*dx_b
        + alpha*div(ksi)*psi*dx_b
        + K*dot(grad(phi), grad(psi))*dx_b
        + L_3*phi*psi*ds_b(5)
        - dot(ksi, n_b)*psi*ds_b(5)
        - f_p*psi*dx_b
        + dot(2.0*u_old - u_old1, n_b)*psi*ds_b(5)
        - L_3*(2.0*phi_old - phi_old1)*psi*ds_b(5)
    )

    a_f3, L_f3 = lhs(F_f3), rhs(F_f3)
    a_b3, L_b3 = lhs(F_pe3 + F_pd3), rhs(F_pe3 + F_pd3)
    # As in the reference driver, reuse the matrices within each uniform-grid
    # run. Time and histories enter only the right-hand sides.
    A_f = assemble(a_f3)
    for bc in bc_f:
        bc.apply(A_f)
    solver_f = LUSolver(A_f, "umfpack")
    A_b = assemble(a_b3)
    for bc in bc_b:
        bc.apply(A_b)
    solver_b = LUSolver(A_b, "umfpack")

    err_u = err_p = err_eta = err_ksi = err_phi = 0.0
    for n in range(2, steps + 1):
        time.assign(n * dt_val)
        b_f = assemble(L_f3)
        for bc in bc_f:
            bc.apply(b_f)
        solver_f.solve(NS_new.vector(), b_f)
        b_b = assemble(L_b3)
        for bc in bc_b:
            bc.apply(b_b)
        solver_b.solve(Bi_new.vector(), b_b)

        u_new, p_new = NS_new.split(True)
        eta_new, ksi_new, phi_new = Bi_new.split(True)
        # Maximum-in-time errors: elastic energy for eta, L2 for other fields.
        err_u = max(err_u, np.sqrt(assemble(dot(u_D-u_new, u_D-u_new)*dx_f)))
        err_p = max(err_p, np.sqrt(assemble((p_D-p_new)**2*dx_f)))
        eta_error = eta_D - eta_new
        err_eta = max(
            err_eta,
            np.sqrt(assemble(
                (2.0*mu_b*inner(epsilon(eta_error), epsilon(eta_error))
                 + lambda_b*div(eta_error)**2)*dx_b
            )),
        )
        err_ksi = max(err_ksi, np.sqrt(assemble(dot(ksi_D-ksi_new, ksi_D-ksi_new)*dx_b)))
        err_phi = max(err_phi, np.sqrt(assemble((phi_D-phi_new)**2*dx_b)))

        # Shift both histories only after both independent block solves.
        u_old1.assign(u_old)
        p_old1.assign(p_old)
        eta_old1.assign(eta_old)
        ksi_old1.assign(ksi_old)
        phi_old1.assign(phi_old)
        u_old.assign(u_new)
        p_old.assign(p_new)
        eta_old.assign(eta_new)
        ksi_old.assign(ksi_new)
        phi_old.assign(phi_new)
        for fn in (u_old, u_old1, ksi_old, ksi_old1, phi_old, phi_old1):
            fn.set_allow_extrapolation(True)

    return mesh_f.hmax(), dt_val, err_u, err_p, err_eta, err_ksi, err_phi


def print_table(title, rows, is_dt):
    print("\n" + title)
    names = ("u", "p", "eta_E", "ksi", "phi")
    print(f"{'n':>6} {'dt':>13} {'dx':>13}" + "".join(
        f" {'err_'+name:>13} {'rate':>8}" for name in names))
    previous = None
    for n, dt_val, dx_val, *errors in rows:
        step = dt_val if is_dt else dx_val
        fields = []
        for i, error in enumerate(errors):
            rate = ""
            if previous is not None and error > 0.0 and previous[1][i] > 0.0:
                rate = f"{np.log(previous[1][i]/error)/np.log(previous[0]/step):8.4f}"
            fields.append(f" {error:13.6e} {rate:>8}")
        print(f"{n:6d} {dt_val:13.6e} {dx_val:13.6e}" + "".join(fields))
        previous = step, errors


def run_case(nm: int, T: float = 1.0, fixed_dt: float = 1e-3,
             phi_degree: int = 1):
    dt_val, _ = time_grid(fixed_dt, T)
    return solve_case(dt_val, nm, T, phi_degree)


def build_spatial_cases(n_list, fixed_dt=1e-3, T=1.0):
    validate_refinements(n_list)
    # Compute this once: the same actual time step is used for every mesh.
    dt_val, steps = time_grid(fixed_dt, T)
    return [(n, n, dt_val, 1.0/n, steps) for n in n_list]


def main():
    parser = argparse.ArgumentParser(
        description="BDF2--EX2 Neumann--Robin spatial sweep, polynomial example")
    parser.add_argument("--spatial-n", nargs="+", type=int,
                        default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--spatial-dt", type=float, default=1e-3)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--phi-degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview the fixed-time-step schedule without solving")
    parser.add_argument("--smoke", action="store_true",
                        help="Run two BDF2 updates on an 8-by-8 mesh")
    args = parser.parse_args()
    T = 0.15 if args.smoke else args.final_time
    fixed_dt = 0.05 if args.smoke else args.spatial_dt
    try:
        cases = build_spatial_cases([8] if args.smoke else args.spatial_n,
                                    fixed_dt, T)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Spatial sweep: polynomial example, BDF2--EX2 Neumann--Robin, T={T:g}")
    print(f"Fixed time step: {cases[0][2]:.6e}")
    print(f"{'n':>6} {'intervals':>10} {'dt':>14} {'dx':>14} {'steps':>10}")
    for n, nm, dt_val, dx_val, steps in cases:
        print(f"{n:6d} {nm:10d} {dt_val:14.6e} {dx_val:14.6e} {steps:10d}")
    if args.dry_run:
        return

    rows = []
    for n, nm, _, dx_val, _ in cases:
        _, actual_dt, *errors = run_case(nm, T, fixed_dt, args.phi_degree)
        rows.append((n, actual_dt, dx_val, *errors))
    print_table("Spatial convergence: max-in-time errors (eta: elastic energy; others: L2), fixed time step",
                rows, is_dt=False)


if __name__ == "__main__":
    main()
