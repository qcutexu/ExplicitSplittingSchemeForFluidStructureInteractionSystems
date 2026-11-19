# This code is for the paper: ERROR ANALYSIS OF THE EXPLICIT SPLITTING SCHEME FOR
# FLUID-POROELASTIC STRUCTURE INTERACTION PROBLEMS 
# Yifan Wang, Apr 2026
import os
import sys
import argparse

# --- robust settings for macOS MPICH/OFI ---
os.environ.setdefault("FI_PROVIDER", "tcp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fenics import *
import numpy as np

"""
MPI (2 ranks) backward-Euler splitting for the compatible manufactured
Stokes--Biot example, with temporal and spatial convergence sweeps.

- NO Python multiprocessing/Pipe
- rank 0 solves fluid; rank 1 solves Biot
- coupling exchanged with mpi4py using BLOCKING Send/Recv (avoids OFI nonblocking progress stalls on macOS)
- robust DOF ordering: transfer fluid velocity using a standalone VectorFunctionSpace on both ranks

Run:
  FI_PROVIDER=tcp "$CONDA_PREFIX/bin/mpiexec" -n 2 \
    "$CONDA_PREFIX/bin/python" NS_Biot_proof_compatible_mpi_BackwardEuler_sweeps_ex1.py

Quick two-step check: append --smoke. To run one large temporal case, append
--mode temporal --temporal-n 128. The temporal sweep ends at T=1; the spatial
sweep also ends at T=1. By default, the temporal sweep fixes 100 intervals per
side (dx=0.01) and uses dt=0.05/n; the spatial sweep fixes dt=1e-3 and uses
n intervals per side. Override these fixed values with
--temporal-mesh-intervals or --spatial-dt. All five errors are final-time L2 norms.
The section's P1 pore pressure is the default; --phi-degree 2 matches the
manuscript proof's common k=2 H1 approximation assumption instead.
"""

from mpi4py import MPI as pyMPI
comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size != 2:
    if rank == 0:
        print("ERROR: This script must be run with exactly 2 MPI ranks.", flush=True)
    raise SystemExit(1)

fenics_comm = MPI.comm_self  # keep FEniCS objects serial on each rank

# ----------------------
# Parameters 
# ----------------------
rho_f    = 1.0
mu_f     = 1.0
rho_b    = 1.0
mu_b     = 1.0
lambda_b = 1.0
alpha    = 1.0
C0       = 1.0
K        = 1.0
C_BJS    = 1.0

if any(abs(value - 1.0) > 1e-14 for value in (mu_f, mu_b, lambda_b, alpha)):
    raise ValueError("The manufactured interface fields require unit viscosities, Lamé constants and Biot coefficient")

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

def print_table(title, rows, is_dt):
    if rank != 0:
        return
    print("\n" + "="*170)
    print(title)
    print("="*170)
    print(f"{'n':>5} {'dt':>12} {'dx':>12}" + "".join(
        f" {name:>12} {'rate':>9}" for name in
        ("err_eta", "err_ksi", "err_phi", "err_u", "err_p")))
    previous = None
    for n, dt_float, dx_float, *errors in rows:
        step = dt_float if is_dt else dx_float
        fields = []
        for i, error in enumerate(errors):
            rate = ""
            if previous is not None and error > 0.0 and previous[1][i] > 0.0:
                rate = f"{np.log(previous[1][i]/error)/np.log(previous[0]/step):9.4f}"
            fields.append(f" {error:12.4e} {rate:>9}")
        print(f"{n:5d} {dt_float:12.4e} {dx_float:12.4e}" + "".join(fields))
        previous = step, errors

def solve_case(dt_float: float, nm: int, T: float, phi_degree: int = 1):
    """
    Return the final-time L2 errors for u, p, eta, ksi, phi.
    """
    # broadcast to keep ranks in sync
    dt_float, nm, T, phi_degree = comm.bcast(
        (dt_float, nm, T, phi_degree), root=0)
    nsteps = int(round(T/dt_float))
    if abs(nsteps*dt_float - T) > 1e-12:
        raise ValueError("T must be an integer multiple of dt_float")
    if nsteps < 1:
        raise ValueError("The solve requires at least one time step")
    dt = Constant(dt_float)

    # ----------------------
    # Rank 0: fluid
    # ----------------------
    if rank == 0:
        mesh_f = RectangleMesh(fenics_comm, Point(0.0, 0.0), Point(1.0, 1.0), nm, nm)
        mesh_b = RectangleMesh(fenics_comm, Point(0.0, -1.0), Point(1.0, 0.0), nm, nm)
        time_f = Constant(0.0)
        u_D, p_D, f_f = make_fluid_data(mesh_f, time_f)

        n_f = FacetNormal(mesh_f)
        tor_f = as_vector((-n_f[1], n_f[0]))

        class Right_F(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0)
        class Left_F(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
        class Top_F(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[1], 1.0)
        class Interface_F(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)

        bmf_f = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1, 0)
        Interface_F().mark(bmf_f, 5)
        Right_F().mark(bmf_f, 2)
        Top_F().mark(bmf_f, 3)
        Left_F().mark(bmf_f, 4)
        ds_f = Measure("ds", domain=mesh_f, subdomain_data=bmf_f)

        u_elem = VectorElement("CG", mesh_f.ufl_cell(), 2)
        p_elem = FiniteElement("CG", mesh_f.ufl_cell(), 1)
        V_f = FunctionSpace(mesh_f, MixedElement([u_elem, p_elem]))
        u, p = TrialFunctions(V_f)
        v, q = TestFunctions(V_f)

        # Standalone velocity space for robust MPI transfer
        Vuf0 = VectorFunctionSpace(mesh_f, "CG", 2)
        u_send_fn = Function(Vuf0)

        # Biot coupling spaces on mesh_b (to store received eta/ksi/phi)
        eta_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        ksi_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        phi_elem = FiniteElement("CG", mesh_b.ufl_cell(), phi_degree)
        V_b = FunctionSpace(mesh_b, MixedElement([eta_elem, ksi_elem, phi_elem]))
        Biot_old = Function(V_b)
        eta_old, ksi_old, phi_old = Biot_old.split(True)

        NS_old = Function(V_f)
        u_old, p_old = NS_old.split(True)
        NS_present = Function(V_f)

        # Homogeneous velocity on the three exterior sides. The fluid pressure
        # has no Dirichlet condition: its level is fixed by interface traction.
        zero_vec = Constant((0.0, 0.0))
        bc_f = [DirichletBC(V_f.sub(0), zero_vec, bmf_f, side)
                for side in (4, 3, 2)]

        # u(0)=p(0)=0; zero is exactly the joint Stokes Ritz projection.
        # Function(V_f) and both split fields are zero at construction.

        # constants
        rhoF = Constant(rho_f)
        muF  = Constant(mu_f)
        Cbjs = Constant(C_BJS)

        # BE fluid forms 
        a_f = rhoF*dot(u, v)*dx \
            + 2*dt*muF*inner(epsilon(u), epsilon(v))*dx \
            - dt*p*div(v)*dx \
            + dt*q*div(u)*dx \
            + dt*dot(u, n_f)*dot(v, n_f)*ds_f(5) \
            + dt*Cbjs*dot(u, tor_f)*dot(v, tor_f)*ds_f(5)

        L_f = dt*dot(f_f, v)*dx \
            + rhoF*dot(u_old, v)*dx \
            + dt*Cbjs*dot(ksi_old, tor_f)*dot(v, tor_f)*ds_f(5) \
            + dt*dot(u_old, n_f)*dot(v, n_f)*ds_f(5) \
            - dt*phi_old*dot(v, n_f)*ds_f(5)

        # The matrix is time-independent within a case: assemble it once and
        # reuse the LU solver while reassembling only the changing right side.
        A_f = assemble(a_f)
        for bc in bc_f:
            bc.apply(A_f)
        solver_f = LUSolver(A_f, "umfpack")

        # MPI buffers
        send_u = np.zeros(u_send_fn.vector().local_size(), dtype=np.float64)
        recv_eta = np.zeros(eta_old.vector().local_size(), dtype=np.float64)
        recv_ksi = np.zeros(ksi_old.vector().local_size(), dtype=np.float64)
        recv_phi = np.zeros(phi_old.vector().local_size(), dtype=np.float64)

        for nstep in range(nsteps):
            t = (nstep + 1)*dt_float

            # blocking exchange: send u^n, receive (eta^n,ksi^n,phi^n)
            u_send_fn.assign(u_old)
            send_u[:] = u_send_fn.vector().get_local()
            comm.Send(send_u, dest=1, tag=10)
            comm.Recv(recv_eta, source=1, tag=20)
            comm.Recv(recv_ksi, source=1, tag=21)
            comm.Recv(recv_phi, source=1, tag=22)

            eta_old.vector()[:] = recv_eta; eta_old.set_allow_extrapolation(True)
            ksi_old.vector()[:] = recv_ksi; ksi_old.set_allow_extrapolation(True)
            phi_old.vector()[:] = recv_phi; phi_old.set_allow_extrapolation(True)

            time_f.assign(t)

            b_f = assemble(L_f)
            for bc in bc_f:
                bc.apply(b_f)
            solver_f.solve(NS_present.vector(), b_f)
            u_fem, p_fem = NS_present.split(True)

            u_old.assign(u_fem)
            p_old.assign(p_fem)

        eu = np.sqrt(assemble(dot(u_D - u_fem, u_D - u_fem)*dx(domain=mesh_f)))
        ep = np.sqrt(assemble((p_D - p_fem)*(p_D - p_fem)*dx(domain=mesh_f)))

        # ask rank1 to send its error triple
        comm.Send(np.array([1], dtype=np.int32), dest=1, tag=98)
        biot_err = np.empty(3, dtype=np.float64)
        comm.Recv(biot_err, source=1, tag=77)
        err_eta, err_ksi, err_phi = biot_err.tolist()

        return eu, ep, err_eta, err_ksi, err_phi

    # ----------------------
    # Rank 1: biot
    # ----------------------
    else:
        mesh_f = RectangleMesh(fenics_comm, Point(0.0, 0.0), Point(1.0, 1.0), nm, nm)
        mesh_b = RectangleMesh(fenics_comm, Point(0.0, -1.0), Point(1.0, 0.0), nm, nm)
        time_b = Constant(0.0)
        eta_D, ksi_D, phi_D, f_e, f_p = make_biot_data(mesh_b, time_b)

        n_b = FacetNormal(mesh_b)
        tor_b = as_vector((-n_b[1], n_b[0]))

        class Right_B(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0)
        class Left_B(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
        class Bottom_B(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[1], -1.0)
        class Interface_B(SubDomain):
            def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)

        bmf_b = MeshFunction("size_t", mesh_b, mesh_b.topology().dim()-1, 0)
        Interface_B().mark(bmf_b, 5)
        Right_B().mark(bmf_b, 2)
        Bottom_B().mark(bmf_b, 1)
        Left_B().mark(bmf_b, 4)
        ds_b = Measure("ds", domain=mesh_b, subdomain_data=bmf_b)

        eta_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        ksi_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        phi_elem = FiniteElement("CG", mesh_b.ufl_cell(), phi_degree)
        V_b = FunctionSpace(mesh_b, MixedElement([eta_elem, ksi_elem, phi_elem]))
        eta, ksi, phi = TrialFunctions(V_b)
        vp, zeta, psi = TestFunctions(V_b)

        Biot_old = Function(V_b)
        eta_old, ksi_old, phi_old = Biot_old.split(True)
        Biot_present = Function(V_b)

        # Standalone fluid velocity receive space must match rank0 send space
        Vuf1 = VectorFunctionSpace(mesh_f, "CG", 2)
        u_old = Function(Vuf1)

        # The exact displacement, solid velocity and pore pressure vanish on
        # the three exterior sides, but not all vanish inside at t=0.
        zero_vec = Constant((0.0, 0.0))
        zero_scalar = Constant(0.0)
        bc_b = ([DirichletBC(V_b.sub(0), zero_vec, bmf_b, side)
                 for side in (1, 2, 4)]
                + [DirichletBC(V_b.sub(1), zero_vec, bmf_b, side)
                   for side in (1, 2, 4)]
                + [DirichletBC(V_b.sub(2), zero_scalar, bmf_b, side)
                   for side in (1, 2, 4)])

        # eta(0)=ksi(0)=0, hence their elastic Ritz projections are zero.
        # For phi(0)=x^2(1-x)^2*y^2(1+y), solve the actual K-weighted
        # elliptic Ritz problem with Dirichlet data only on exterior sides.
        Qphi = phi_old.function_space()
        phi_r = TrialFunction(Qphi)
        phi_test = TestFunction(Qphi)
        bc_phi0 = [DirichletBC(Qphi, zero_scalar, bmf_b, side)
                   for side in (1, 2, 4)]
        a_phi0 = K*dot(grad(phi_r), grad(phi_test))*dx(domain=mesh_b)
        L_phi0 = K*dot(grad(phi_D), grad(phi_test))*dx(domain=mesh_b)
        solve(a_phi0 == L_phi0, phi_old, bc_phi0,
              solver_parameters={"linear_solver": "umfpack"})
        if "--smoke" in sys.argv:
            phi0_error = phi_D - phi_old
            print("[rank1] initial pore-pressure Ritz errors: "
                  f"L2={np.sqrt(assemble(phi0_error*phi0_error*dx(domain=mesh_b))):.6e}, "
                  f"H1-seminorm={np.sqrt(assemble(dot(grad(phi0_error), grad(phi0_error))*dx(domain=mesh_b))):.6e}",
                  flush=True)

        # constants
        rhoB = Constant(rho_b)
        muB  = Constant(mu_b)
        lamB = Constant(lambda_b)
        Alp  = Constant(alpha)
        C0c  = Constant(C0)
        Kc   = Constant(K)
        Cbjs = Constant(C_BJS)

        # BE Biot forms 
        a_be = rhoB*dot(ksi, zeta)*dx \
             + 2*dt*muB*inner(epsilon(eta), epsilon(zeta))*dx \
             + dt*lamB*div(eta)*div(zeta)*dx \
             - dt*Alp*phi*div(zeta)*dx \
             + dt*dot(ksi, vp)*dx \
             - dot(eta, vp)*dx \
             + dt*dot(ksi, n_b)*dot(zeta, n_b)*ds_b(5) \
             + dt*Cbjs*dot(ksi, tor_b)*dot(zeta, tor_b)*ds_b(5) \
             + dt*phi*dot(zeta, n_b)*ds_b(5)

        L_be = rhoB*dot(ksi_old, zeta)*dx \
             - dot(eta_old, vp)*dx \
             + dt*dot(f_e, zeta)*dx \
             + dt*Cbjs*dot(u_old, tor_b)*dot(zeta, tor_b)*ds_b(5) \
             + dt*dot(ksi_old, n_b)*dot(zeta, n_b)*ds_b(5)

        a_bp = C0c*phi*psi*dx \
             + dt*Alp*div(ksi)*psi*dx \
             + dt*Kc*dot(grad(phi), grad(psi))*dx \
             + dt*phi*psi*ds_b(5) \
             - dt*dot(ksi, n_b)*psi*ds_b(5)

        L_bp = dt*f_p*psi*dx \
             + C0c*phi_old*psi*dx \
             - dt*dot(u_old, n_b)*psi*ds_b(5) \
             + dt*phi_old*psi*ds_b(5)

        a_b = a_be + a_bp
        L_b = L_be + L_bp

        A_b = assemble(a_b)
        for bc in bc_b:
            bc.apply(A_b)
        solver_b = LUSolver(A_b, "umfpack")

        # MPI buffers
        recv_u = np.zeros(u_old.vector().local_size(), dtype=np.float64)
        send_eta = np.zeros(eta_old.vector().local_size(), dtype=np.float64)
        send_ksi = np.zeros(ksi_old.vector().local_size(), dtype=np.float64)
        send_phi = np.zeros(phi_old.vector().local_size(), dtype=np.float64)

        for nstep in range(nsteps):
            t = (nstep + 1)*dt_float

            # blocking exchange: recv u^n, send (eta^n,ksi^n,phi^n)
            comm.Recv(recv_u, source=0, tag=10)
            u_old.vector()[:] = recv_u
            u_old.set_allow_extrapolation(True)

            send_eta[:] = eta_old.vector().get_local()
            send_ksi[:] = ksi_old.vector().get_local()
            send_phi[:] = phi_old.vector().get_local()
            comm.Send(send_eta, dest=0, tag=20)
            comm.Send(send_ksi, dest=0, tag=21)
            comm.Send(send_phi, dest=0, tag=22)

            time_b.assign(t)

            b_b = assemble(L_b)
            for bc in bc_b:
                bc.apply(b_b)
            solver_b.solve(Biot_present.vector(), b_b)
            eta_fem, ksi_fem, phi_fem = Biot_present.split(True)

            eta_old.assign(eta_fem)
            ksi_old.assign(ksi_fem)
            phi_old.assign(phi_fem)

        eta_error = eta_fem - eta_D
        ee = (np.sqrt(assemble(inner(epsilon(eta_error), epsilon(eta_error)) * dx(domain=mesh_b)))  + np.sqrt(assemble(div(eta_error)**2 * dx(domain=mesh_b))))
        # ee = np.sqrt(assemble(dot(eta_D - eta_fem, eta_D - eta_fem)*dx(domain=mesh_b)))
        ek = np.sqrt(assemble(dot(ksi_D - ksi_fem, ksi_D - ksi_fem)*dx(domain=mesh_b)))
        eph = np.sqrt(assemble((phi_D - phi_fem)*(phi_D - phi_fem)*dx(domain=mesh_b)))

        # wait for end-of-case signal and return errors
        flag = np.empty(1, dtype=np.int32)
        comm.Recv(flag, source=0, tag=98)
        comm.Send(np.array([ee, ek, eph], dtype=np.float64), dest=0, tag=77)
        return None

# ----------------------
# Sweep schedules
# ----------------------
def temporal_sweep(n_list, phi_degree, fixed_nm):
    # Hold the mesh fixed; n changes only the time step.
    T = 1.0
    dx_float = 1/(fixed_nm)

    rows = []
    for n in n_list:
        dt_float = 0.05/n
        errors = solve_case(dt_float, fixed_nm, T, phi_degree)
        if rank == 0:
            eu, ep, ee, ek, eph = errors
            rows.append((n, dt_float, dx_float, ee, ek, eph, eu, ep))
            print(f"[rank0] temporal n={n}: dt={dt_float:.6e}, dx={dx_float:.6e}, "
                  f"eta={ee:.6e}, xi={ek:.6e}, phi={eph:.6e}, "
                  f"u={eu:.6e}, p={ep:.6e}", flush=True)
    print_table(f"Temporal refinement at T={T:g}, fixed dx={dx_float:g} "
                f"({fixed_nm} intervals/side, phi P{phi_degree})",
                rows, is_dt=True)

def spatial_sweep(n_list, phi_degree, fixed_dt):
    # Hold the time step fixed; n changes only the mesh.
    dt_float = fixed_dt
    T = 1.0

    rows = []
    for n in n_list:
        dx_float = 1.0/n
        errors = solve_case(dt_float, n, T, phi_degree)
        if rank == 0:
            eu, ep, ee, ek, eph = errors
            rows.append((n, dt_float, dx_float, ee, ek, eph, eu, ep))
            print(f"[rank0] spatial n={n}: dt={dt_float:.6e}, dx={dx_float:.6e}, "
                  f"eta={ee:.6e}, xi={ek:.6e}, phi={eph:.6e}, "
                  f"u={eu:.6e}, p={ep:.6e}", flush=True)
    print_table(f"Spatial refinement at T={T:g}, fixed dt={dt_float:g} "
                f"(phi P{phi_degree})",
                rows, is_dt=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-rank Stokes--Biot convergence sweeps")
    parser.add_argument("--mode", choices=("both", "temporal", "spatial"),
                        default="both", help="Select which convergence sweep to run")
    parser.add_argument("--temporal-n", nargs="+", type=int,
                        default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--spatial-n", nargs="+", type=int,
                        default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--temporal-mesh-intervals", type=int, default=100,
                        help="Fixed intervals per side in the temporal sweep (default: 100; dx=0.01)")
    parser.add_argument("--spatial-dt", type=float, default=1e-3,
                        help="Fixed time step in the spatial sweep (default: 1e-3)")
    parser.add_argument("--phi-degree", type=int, choices=(1, 2), default=1,
                        help="Pore-pressure degree: P1 in the numerical section, P2 for the theorem")
    parser.add_argument("--smoke", action="store_true", help="Run two short steps")
    args = parser.parse_args()
    if any(n < 1 for n in args.temporal_n + args.spatial_n):
        parser.error("All refinement parameters must be positive integers")
    if args.temporal_mesh_intervals < 1:
        parser.error("--temporal-mesh-intervals must be a positive integer")
    if not (0.0 < args.spatial_dt <= 1.0):
        parser.error("--spatial-dt must be in (0, 1]")
    if abs(round(1.0/args.spatial_dt)*args.spatial_dt - 1.0) > 1e-12:
        parser.error("--spatial-dt must divide T=1 into an integer number of steps")

    if rank == 0:
        print("[rank0] BE sweep starting", flush=True)
    else:
        print("[rank1] BE sweep starting", flush=True)

    if args.smoke:
        result = solve_case(0.05, 8, 0.1, args.phi_degree)
        if rank == 0:
            print("[rank0] smoke errors (u, p, eta, ksi, phi):", result, flush=True)
    else:
        if args.mode in ("both", "temporal"):
            temporal_sweep(args.temporal_n, args.phi_degree,
                           args.temporal_mesh_intervals)
        if args.mode in ("both", "spatial"):
            comm.Barrier()
            spatial_sweep(args.spatial_n, args.phi_degree, args.spatial_dt)
    comm.Barrier()

    if rank == 0:
        print("[rank0] BE sweep done", flush=True)
    else:
        print("[rank1] BE sweep done", flush=True)
