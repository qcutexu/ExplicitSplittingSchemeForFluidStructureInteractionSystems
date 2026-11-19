"""Temporal refinement for the polynomial manufactured solution from ex1.

The mesh is fixed at 100 intervals/side by default, T=1, and dt_target=0.05/n.
BDF2 is used in both blocks with EX2 interface predictors and Neumann--Robin
normal coupling. Both starting levels are projected from the exact fields.
Errors are maxima in time: elastic energy for displacement, L2 for other fields.

Run: python3 nsbiot_Neumann_Robin_temporal_sweep.py
Preview: append --dry-run (FEniCS is not required for this preview).
"""

import argparse
import math
import csv
import json
import time as perf_time
from pathlib import Path
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

# Non-unit stress parameters may violate the manufactured interface conditions.
# The comparison driver rejects such cases with manufactured forcing; zero-source
# energy tests may use other positive material coefficients.

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


def solve_case(dt_val, nm, T, phi_degree=2, *, beta=0.0, L=1.0,
               gamma=1.0, physical=None, forcing="manufactured",
               initialization="projected", monitor=False,
               initial_mode="smooth", mode_number=0, observer=None,
               history_initializer=None, history_label=None, operator_capture=None):
    """Run one coupled case, returning accuracy, energy and wall-time diagnostics.

    observer, when supplied, receives read-only solution/history references and
    algebraic residuals after each update; it must not mutate these functions.
    This optional verification hook does not change the solved equations.

    history_initializer is an optional, zero-forcing-only initializer for
    explicitly labeled algebraic BDF2 disturbances. It runs after the default
    projections and before forms are built; prior defaults are unchanged.

    operator_capture optionally exposes the assembled homogeneous update for
    verification/spectral analysis. It is forbidden with manufactured forcing.

    beta=0 gives Neumann--Robin; beta>0 gives the earlier Robin--Robin.
    L_1=L and L_3=1/L enforce the manuscript's reciprocal normal Robin pair.
    zero forcing with duplicated nontrivial initial pressure is designed for the
    homogeneous algebraic energy comparison, not for time-accuracy verification.
    """
    if _fenics_import_error is not None:
        raise RuntimeError(
            "Legacy FEniCS is required to solve; --dry-run previews the schedule"
        ) from _fenics_import_error
    if operator_capture is not None and (forcing != "zero" or not callable(operator_capture)):
        raise ValueError("Operator capture requires zero forcing and a callable receiver")
    if history_initializer is not None:
        if forcing != "zero" or initialization != "duplicate":
            raise ValueError("Custom histories require homogeneous forcing and duplicate base initialization")
        if not callable(history_initializer) or not isinstance(history_label, str) or not history_label.strip():
            raise ValueError("Custom histories require a callable initializer and an explicit nonempty label")
    elif history_label is not None:
        raise ValueError("history_label requires a custom history initializer")
    if nm < 1 or phi_degree not in (1, 2):
        raise ValueError("Use a positive mesh size and pore-pressure degree 1 or 2")
    if not (math.isfinite(beta) and beta >= 0 and math.isfinite(L) and L > 0
            and math.isfinite(gamma) and gamma > 0):
        raise ValueError("Require beta >= 0, L > 0, and gamma > 0, all finite")
    if forcing not in ("manufactured", "zero"):
        raise ValueError("forcing must be 'manufactured' or 'zero'")
    if initialization not in ("projected", "duplicate"):
        raise ValueError("initialization must be 'projected' or 'duplicate'")
    if initial_mode not in ("smooth", "interface", "oscillatory"):
        raise ValueError("initial_mode must be smooth, interface, or oscillatory")
    if mode_number < 0 or int(mode_number)!=mode_number:
        raise ValueError("mode_number must be a nonnegative integer")
    if forcing == "manufactured" and initial_mode != "smooth":
        raise ValueError("Manufactured convergence requires initial_mode=smooth")
    if forcing == "manufactured" and initialization != "projected":
        raise ValueError("Manufactured convergence tests require projected exact starts")
    if forcing == "zero" and initialization != "duplicate":
        raise ValueError("Homogeneous energy tests require duplicated nonzero initial pressure")
    material = {"rho_f": 1.0, "mu_f": 1.0, "rho_b": 1.0,
                "mu_b": 1.0, "lambda_b": 1.0, "alpha": 1.0,
                "C0": 1.0, "K": 1.0}
    if physical:
        invalid = set(physical) - set(material)
        if invalid:
            raise ValueError(f"Unknown material parameters: {sorted(invalid)}")
        material.update(physical)
    if not all(math.isfinite(float(x)) and float(x)>0 for x in material.values()):
        raise ValueError("This driver requires strictly positive finite material coefficients")
    if forcing == "manufactured" and any(abs(float(material[x])-1.0)>1e-13
             for x in ("mu_f", "mu_b", "lambda_b", "alpha")):
        raise ValueError("The manufactured solution requires unit mu_f, mu_b, lambda_b and alpha")
    # Source factory functions reference the module-level parameters. Reset on
    # EVERY call to prevent parameter leakage between cases.
    globals().update({k:float(v) for k,v in material.items()})
    global C_BJS
    C_BJS = float(gamma)
    L_1, L_3 = float(L), 1.0 / float(L)
    steps = int(round(T / dt_val))
    if steps < 2 or not math.isclose(steps * dt_val, T,
                                     rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("The uniform time step must divide T into at least two steps")
    dt = Constant(dt_val)
    wall_start = perf_time.perf_counter()

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
    f_f_source = f_f if forcing == "manufactured" else Constant((0.0, 0.0))
    f_e_source = f_e if forcing == "manufactured" else Constant((0.0, 0.0))
    f_p_source = f_p if forcing == "manufactured" else Constant(0.0)
    bx,by = SpatialCoordinate(mesh_b)
    normal_shape = bx*bx*(1.0-bx)*(1.0-bx)*(1.0+by)
    effective_mode = max(1, nm//4) if mode_number==0 else int(mode_number)
    if initial_mode == "smooth":
        phi_initial = phi_D
    elif initial_mode == "interface":
        phi_initial = normal_shape
    else:
        phi_initial = normal_shape*sin(float(effective_mode)*math.pi*bx)

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
        L_phi0 = K * dot(grad(phi_initial), grad(test)) * dx_b
        bc_phi0 = [DirichletBC(space, zero_scalar, bmf_b, side)
                   for side in (1, 2, 4)]
        solve(a_phi0 == L_phi0, phih, bc_phi0,
              solver_parameters=solver_parameters)

    # Exact projected starts at t=0 and t=dt; the first BDF2 solve is at 2*dt.
    initialize_level(0.0, u_old1, p_old1, eta_old1, ksi_old1, phi_old1)
    initialize_level(dt_val if initialization == "projected" else 0.0,
                     u_old, p_old, eta_old, ksi_old, phi_old)

    if history_initializer is not None:
        history_initializer({
            "nm":nm, "dt":dt_val, "mesh_f":mesh_f, "mesh_b":mesh_b,
            "boundary_f":bmf_f, "boundary_b":bmf_b,
            "old": {"u":u_old,"p":p_old,"eta":eta_old,"xi":ksi_old,"phi":phi_old},
            "old1": {"u":u_old1,"p":p_old1,"eta":eta_old1,"xi":ksi_old1,"phi":phi_old1},
            "material":dict(material),
        })

    for fn in (u_old, u_old1, ksi_old, ksi_old1, phi_old, phi_old1):
        fn.set_allow_extrapolation(True)

    # BDF2 derivatives and EX2 interface predictors, a*=2*a_old-a_old1.
    # The new fluid field is not inserted into the current Biot solve.
    from nsbiot_forms import fluid_form, biot_forms
    F_f3 = fluid_form(V_f,dt,{'u':u_old,'u1':u_old1},
        {'xi':ksi_old,'xi1':ksi_old1,'phi':phi_old,'phi1':phi_old1},
        {'n':n_f,'tangent':tor_f,'dx':dx_f,'ds':ds_f},material,L,C_BJS,f_f_source)
    F_pe3,F_pd3 = biot_forms(V_b,dt,
        {'eta':eta_old,'eta1':eta_old1,'xi':ksi_old,'xi1':ksi_old1,'phi':phi_old,'phi1':phi_old1},
        {'u':u_old,'u1':u_old1},
        {'n':n_b,'tangent':tor_b,'dx':dx_b,'ds':ds_b},material,L,C_BJS,beta,f_e_source,f_p_source)

    def bdf2_energy(u_n, u_nm1, xi_n, xi_nm1,
                    eta_n, eta_nm1, phi_n, phi_nm1):
        """Positive bulk BDF2 energy; excludes additional interface corrections."""
        u_pair = 2.0*u_n-u_nm1
        xi_pair = 2.0*xi_n-xi_nm1
        eta_pair = 2.0*eta_n-eta_nm1
        phi_pair = 2.0*phi_n-phi_nm1
        ef = assemble(rho_f*(dot(u_n,u_n)+dot(u_pair,u_pair))*dx_f)
        ep = assemble((
            rho_b*(dot(xi_n,xi_n)+dot(xi_pair,xi_pair))
            + C0*(phi_n**2+phi_pair**2)
            + 2.0*mu_b*(inner(epsilon(eta_n),epsilon(eta_n))
                       +inner(epsilon(eta_pair),epsilon(eta_pair)))
            + lambda_b*(div(eta_n)**2+div(eta_pair)**2)
        )*dx_b)
        return 0.25 * float(ef+ep)

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

    if operator_capture is not None:
        operator_capture({
            "nm":nm,"dt":dt_val,"beta":beta,"material":dict(material),
            "V_f":V_f,"V_b":V_b,"mesh_f":mesh_f,"mesh_b":mesh_b,
            "A_f":A_f,"A_b":A_b,"rhs_f":L_f3,"rhs_b":L_b3,
            "bc_f":bc_f,"bc_b":bc_b,"solver_f":solver_f,"solver_b":solver_b,
            "new_f":NS_new,"new_b":Bi_new,
            "old":{"u":u_old,"p":p_old,"eta":eta_old,"xi":ksi_old,"phi":phi_old},
            "old1":{"u":u_old1,"p":p_old1,"eta":eta_old1,"xi":ksi_old1,"phi":phi_old1},
        })

    assembly_elapsed = perf_time.perf_counter()-wall_start
    solve_elapsed = 0.0
    rhs_elapsed = 0.0
    error_elapsed = 0.0
    energy_history = []
    normal_velocity_peak = 0.0
    observer_elapsed = 0.0
    if monitor:
        initial_energy = bdf2_energy(u_old,u_old1,ksi_old,ksi_old1,
                                     eta_old,eta_old1,phi_old,phi_old1)
        energy_history.append({"step":1,"t":dt_val,"energy":initial_energy})
    err_u = err_p = err_eta = err_ksi = err_phi = 0.0
    for n in range(2, steps + 1):
        time.assign(n * dt_val)
        rhs_start = perf_time.perf_counter()
        b_f = assemble(L_f3)
        for bc in bc_f:
            bc.apply(b_f)
        rhs_elapsed += perf_time.perf_counter()-rhs_start
        solve_start = perf_time.perf_counter()
        solver_f.solve(NS_new.vector(), b_f)
        solve_elapsed += perf_time.perf_counter()-solve_start
        rhs_start = perf_time.perf_counter()
        b_b = assemble(L_b3)
        for bc in bc_b:
            bc.apply(b_b)
        rhs_elapsed += perf_time.perf_counter()-rhs_start
        solve_start = perf_time.perf_counter()
        solver_b.solve(Bi_new.vector(), b_b)
        solve_elapsed += perf_time.perf_counter()-solve_start

        u_new, p_new = NS_new.split(True)
        eta_new, ksi_new, phi_new = Bi_new.split(True)
        diagnostic_start = perf_time.perf_counter()
        if forcing == "manufactured":
            err_u = max(err_u, np.sqrt(assemble(dot(u_D-u_new, u_D-u_new)*dx_f)))
            err_p = max(err_p, np.sqrt(assemble((p_D-p_new)**2*dx_f)))
            eta_error = eta_D - eta_new
            err_eta = max(err_eta, np.sqrt(assemble(
                (2.0*mu_b*inner(epsilon(eta_error), epsilon(eta_error))
                 + lambda_b*div(eta_error)**2)*dx_b)))
            err_ksi = max(err_ksi, np.sqrt(assemble(dot(ksi_D-ksi_new, ksi_D-ksi_new)*dx_b)))
            err_phi = max(err_phi, np.sqrt(assemble((phi_D-phi_new)**2*dx_b)))
        if monitor:
            energy = bdf2_energy(u_new,u_old,ksi_new,ksi_old,
                                  eta_new,eta_old,phi_new,phi_old)
            normal = float(np.sqrt(max(0.0,assemble(
                dot(ksi_new,n_b)**2*ds_b(5)))))
            normal_velocity_peak = max(normal_velocity_peak,normal)
            if not math.isfinite(energy):
                raise ArithmeticError(f"Nonfinite discrete energy at step {n}")
            energy_history.append({"step":n,"t":n*dt_val,"energy":energy})
        error_elapsed += perf_time.perf_counter()-diagnostic_start

        if observer is not None:
            observer_start = perf_time.perf_counter()
            residual_f = A_f * NS_new.vector()
            residual_f.axpy(-1.0, b_f)
            residual_b = A_b * Bi_new.vector()
            residual_b.axpy(-1.0, b_b)
            observer({
                "step": n, "t": n*dt_val, "dt": dt_val,
                "bulk_energy": energy if monitor else None,
                "fields": {"u":u_new, "p":p_new, "eta":eta_new,
                           "xi":ksi_new, "phi":phi_new},
                "history": {"eta_old":eta_old, "eta_old1":eta_old1},
                "exact": {"u":u_D, "p":p_D, "eta":eta_D,
                          "xi":ksi_D, "phi":phi_D},
                "residual_f_abs": residual_f.norm("l2"),
                "residual_b_abs": residual_b.norm("l2"),
                "residual_f_relative": residual_f.norm("l2")/max(b_f.norm("l2"),1e-300),
                "residual_b_relative": residual_b.norm("l2")/max(b_b.norm("l2"),1e-300),
            })
            observer_elapsed += perf_time.perf_counter()-observer_start

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

    peak_ratio = (max(row["energy"] for row in energy_history)
                  / energy_history[0]["energy"]
                  if energy_history and energy_history[0]["energy"] > 1e-300
                  else None)
    return {"scheme": "neumann_robin" if beta==0 else "robin_robin",
            "beta":float(beta),"L":float(L),"gamma":float(gamma),
            "hmax":float(mesh_f.hmax()),"h":1.0/nm,
            "nm":nm,"dt":float(dt_val),"T":float(T),"steps":int(steps),
            "initial_mode":("custom" if history_initializer is not None else initial_mode),
            "custom_history_label":history_label,
            "effective_mode":(effective_mode if initial_mode=="oscillatory" else None),
            "phi_degree":phi_degree,"forcing":forcing,
            "err_u":float(err_u),"err_p":float(err_p),
            "err_eta":float(err_eta),"err_xi":float(err_ksi),"err_phi":float(err_phi),
            "energy_initial":(energy_history[0]["energy"] if energy_history else None),
            "energy_final":(energy_history[-1]["energy"] if energy_history else None),
            "peak_energy_ratio":peak_ratio,
            "normal_velocity_peak":normal_velocity_peak,
            "wall_s":perf_time.perf_counter()-wall_start,
            "setup_assembly_s":assembly_elapsed,"rhs_assembly_s":rhs_elapsed,
            "linear_solve_s":solve_elapsed,"diagnostics_s":error_elapsed,
            "observer_s":observer_elapsed,
            "material":material,"energy_history":energy_history}

